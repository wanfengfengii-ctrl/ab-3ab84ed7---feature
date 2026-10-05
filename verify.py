#!/usr/bin/env python3
"""One-shot verification for the attitude interpolation service.

Runs three stages and summarizes them in the process exit code:

  1. BUILD  - byte-compile every source file and import the ASGI app
  2. TESTS  - run the pytest suite (core math, validation, API)
  3. SMOKE  - black-box checks against the live HTTP API: shortest-arc
              interpolation accuracy/continuity and rejection of
              over-long sample gaps (plus zero-quaternion, 180-degree
              ambiguity and non-increasing-time rejections)

Exit code is a bitmask so a single value summarizes every stage:

  0  all stages passed
  1  build/compile stage failed
  2  test stage failed
  4  smoke stage failed

The service URL is read from API_BASE_URL (default http://localhost:8000).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXIT_OK = 0
EXIT_BUILD_FAILED = 1
EXIT_TESTS_FAILED = 2
EXIT_SMOKE_FAILED = 4

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
HEALTH_TIMEOUT_S = float(os.environ.get("VERIFY_HEALTH_TIMEOUT_S", "60"))

SQRT2_2 = math.sqrt(0.5)
T0 = 1_700_000_000_000_000_000  # epoch-scale ns; exceeds 2**53 as a float


def banner(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


# ---------------------------------------------------------------------------
# stage 1: build
# ---------------------------------------------------------------------------

def check_build() -> bool:
    banner("BUILD: byte-compile sources and import application")
    compile_proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "app", "tests", "verify.py"],
        cwd=REPO_ROOT,
    )
    if compile_proc.returncode != 0:
        print("FAIL: byte-compilation failed")
        return False
    import_proc = subprocess.run(
        [sys.executable, "-c", "from app.main import app; print('import ok:', app.title)"],
        cwd=REPO_ROOT,
    )
    if import_proc.returncode != 0:
        print("FAIL: application import failed")
        return False
    print("BUILD: PASS")
    return True


# ---------------------------------------------------------------------------
# stage 2: tests
# ---------------------------------------------------------------------------

def check_tests() -> bool:
    banner("TESTS: pytest suite")
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=REPO_ROOT)
    print("TESTS:", "PASS" if proc.returncode == 0 else "FAIL")
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# stage 3: API smoke
# ---------------------------------------------------------------------------

def wait_for_health() -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(API_BASE_URL + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def post_interpolate(payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        API_BASE_URL + "/api/attitudes/interpolate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        return exc.code, json.loads(body) if body else {}


def q_z(angle: float) -> list[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def check_smoke() -> bool:
    banner("SMOKE: live API interpolation and rejection checks")
    if not wait_for_health():
        print(f"FAIL: {API_BASE_URL}/health did not respond within "
              f"{HEALTH_TIMEOUT_S:.0f}s")
        return False
    print(f"service healthy at {API_BASE_URL}")

    failures: list[str] = []

    def expect(condition: bool, label: str) -> None:
        print(("PASS: " if condition else "FAIL: ") + label)
        if not condition:
            failures.append(label)

    # -- 1. shortest-arc interpolation accuracy + continuity ---------------
    gap = 10_000
    queries = [T0 + 2_500 * k for k in range(5)]
    payload = {
        "samples": [
            {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
            {"t": T0 + gap, "q": [SQRT2_2, 0.0, 0.0, SQRT2_2]},  # +90 deg about z
        ],
        "queries": queries,
        "max_gap_ns": gap,
    }
    status, body = post_interpolate(payload)
    expect(status == 200, f"interpolation request returns 200 (got {status})")
    if status == 200:
        attitudes = body.get("attitudes", [])
        expect([a.get("t") for a in attitudes] == queries,
               "results are returned in query order")
        analytic_ok = True
        unit_ok = True
        for entry, u in zip(attitudes, (0.0, 0.25, 0.5, 0.75, 1.0)):
            q = entry["q"]
            expected = q_z(u * math.pi / 2.0)
            if any(abs(a - e) > 1e-9 for a, e in zip(q, expected)):
                analytic_ok = False
            if abs(sum(c * c for c in q) - 1.0) > 1e-9:
                unit_ok = False
        expect(analytic_ok, "interpolated quaternions match analytic slerp within 1e-9")
        expect(unit_ok, "returned quaternions are unit norm within 1e-9")
        quats = [a["q"] for a in attitudes]
        if quats:
            leading = next((c for c in quats[0] if c != 0.0), 1.0)
            expect(leading > 0.0, "first result's first non-zero component is positive")
            dots_ok = all(
                sum(x * y for x, y in zip(p, c)) >= 0.0
                for p, c in zip(quats, quats[1:])
            )
            expect(dots_ok, "consecutive results keep non-negative dot product")

    # -- 2. q and -q are the same rotation: negated sample, same answer ----
    flipped = json.loads(json.dumps(payload))
    flipped["samples"][1]["q"] = [-SQRT2_2, 0.0, 0.0, -SQRT2_2]
    status_f, body_f = post_interpolate(flipped)
    same = False
    if status == 200 and status_f == 200:
        same = all(
            abs(a - b) <= 1e-9
            for ea, eb in zip(body["attitudes"], body_f["attitudes"])
            for a, b in zip(ea["q"], eb["q"])
        )
    expect(status_f == 200 and same,
           "sign-flipped sample follows the shortest arc to the same attitudes")

    # -- 3. over-long sample gap is rejected with a locatable index --------
    gap_payload = {
        "samples": [
            {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
            {"t": T0 + 10_000_000, "q": [1.0, 0.0, 0.0, 0.0]},
        ],
        "queries": [T0 + 5_000_000],
        "max_gap_ns": 1_000,
    }
    status, body = post_interpolate(gap_payload)
    detail = body.get("detail", {})
    expect(400 <= status < 500, f"over-long gap rejected with 4xx (got {status})")
    expect(detail.get("code") == "SAMPLE_GAP_EXCEEDED"
           and detail.get("index") == 0
           and detail.get("sample_index") == 0,
           f"gap error is locatable to the query/sample index (got {detail})")
    expect("attitudes" not in body, "rejected request yields no partial results")

    # -- 4. zero quaternion rejected ---------------------------------------
    zero_payload = {
        "samples": [
            {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
            {"t": T0 + 1, "q": [0.0, 0.0, 0.0, 0.0]},
        ],
        "queries": [T0],
        "max_gap_ns": 1,
    }
    status, body = post_interpolate(zero_payload)
    detail = body.get("detail", {})
    expect(400 <= status < 500 and detail.get("code") == "ZERO_QUATERNION"
           and detail.get("index") == 1,
           f"zero quaternion rejected with index (got {status} {detail})")

    # -- 5. 180-degree ambiguity rejected ----------------------------------
    ambiguous_payload = {
        "samples": [
            {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
            {"t": T0 + 1, "q": [0.0, 1.0, 0.0, 0.0]},
        ],
        "queries": [T0],
        "max_gap_ns": 1,
    }
    status, body = post_interpolate(ambiguous_payload)
    detail = body.get("detail", {})
    expect(400 <= status < 500 and detail.get("code") == "AMBIGUOUS_180_DEGREE_ROTATION",
           f"180-degree ambiguity rejected (got {status} {detail})")

    # -- 6. non-increasing sample times rejected ---------------------------
    disorder_payload = {
        "samples": [
            {"t": T0 + 1, "q": [1.0, 0.0, 0.0, 0.0]},
            {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
        ],
        "queries": [T0],
        "max_gap_ns": 10,
    }
    status, body = post_interpolate(disorder_payload)
    detail = body.get("detail", {})
    expect(400 <= status < 500 and detail.get("code") == "NON_INCREASING_SAMPLE_TIME"
           and detail.get("index") == 1,
           f"non-increasing sample times rejected with index (got {status} {detail})")

    if failures:
        print(f"SMOKE: FAIL ({len(failures)} check(s) failed)")
        return False
    print("SMOKE: PASS")
    return True


# ---------------------------------------------------------------------------

def main() -> int:
    print(f"verify: target API {API_BASE_URL}", flush=True)
    build_ok = check_build()
    tests_ok = check_tests()
    smoke_ok = check_smoke()

    exit_code = EXIT_OK
    if not build_ok:
        exit_code |= EXIT_BUILD_FAILED
    if not tests_ok:
        exit_code |= EXIT_TESTS_FAILED
    if not smoke_ok:
        exit_code |= EXIT_SMOKE_FAILED

    banner("SUMMARY")
    print(f"build : {'PASS' if build_ok else 'FAIL'}")
    print(f"tests : {'PASS' if tests_ok else 'FAIL'}")
    print(f"smoke : {'PASS' if smoke_ok else 'FAIL'}")
    print(f"verify exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
