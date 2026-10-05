"""Core attitude (quaternion) interpolation logic.

The aerial imaging platform projects low-rate INS attitude samples onto
camera exposure instants.  Attitudes are interpreted as unit quaternions
with the fixed component order ``[w, x, y, z]``.  Because ``q`` and ``-q``
describe the same rotation, interpolation always follows the shortest
rotation arc between adjacent samples, and the returned sequence is
re-signed so that consecutive exposure attitudes stay continuous.

This module is pure Python (no third-party imports) so it can be unit
tested and reused independently of the HTTP layer.  All validation is
performed up front: any violation raises :class:`AttitudeInputError`
carrying a machine-readable ``code`` plus the offending ``index``/``path``,
and no partial result is ever produced.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Any, Dict, List

MIN_SAMPLES = 2
MAX_SAMPLES = 2000
MIN_QUERIES = 1
MAX_QUERIES = 500
QUATERNION_COMPONENTS = 4  # fixed order: w, x, y, z

# Largest integer exactly representable as an IEEE-754 double.  Float
# timestamps beyond this cannot be trusted to be exact nanoseconds.
_SAFE_INTEGER_FLOAT = 2 ** 53

# Adjacent rotations of exactly 180 degrees have no unique shortest arc.
# |dot(q_i, q_{i+1})| at or below this threshold is treated as the
# ambiguous 180-degree case.  The threshold is orders of magnitude above
# float64 normalization noise (~1e-16) yet far below the dot product of
# any representable strictly-less-than-180-degree spacing a client can
# meaningfully use (179.9999999 deg corresponds to |dot| ~ 8.7e-10).
_AMBIGUOUS_DOT_THRESHOLD = 1e-12

# Dot products above this use the small-angle series for slerp weights,
# avoiding cancellation in acos/sin for nearly identical attitudes.
_SMALL_ANGLE_DOT = 1.0 - 1e-9

_MISSING = object()


class AttitudeInputError(ValueError):
    """A client-correctable problem with the request, locatable by index."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        index: int | None = None,
        path: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.index = index
        self.path = path
        self.context = dict(context or {})

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.index is not None:
            payload["index"] = self.index
        if self.path is not None:
            payload["path"] = self.path
        payload.update(self.context)
        return payload


# ---------------------------------------------------------------------------
# scalar validation helpers
# ---------------------------------------------------------------------------

def _require_field(obj: Dict[str, Any], key: str, path: str, index: int | None) -> Any:
    value = obj.get(key, _MISSING)
    if value is _MISSING:
        raise AttitudeInputError(
            "MISSING_FIELD",
            f"{path} is required",
            index=index,
            path=path,
        )
    return value


def _require_int_ns(value: Any, path: str, index: int | None = None) -> int:
    """Coerce a JSON number to an exact integer nanosecond count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"{path} must be an integer number of nanoseconds, "
            f"got {type(value).__name__}",
            index=index,
            path=path,
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AttitudeInputError(
                "NON_FINITE_TIMESTAMP",
                f"{path} must be a finite integer number of nanoseconds",
                index=index,
                path=path,
            )
        if not value.is_integer():
            raise AttitudeInputError(
                "NON_INTEGER_TIMESTAMP",
                f"{path}={value!r} is not an integer number of nanoseconds",
                index=index,
                path=path,
            )
        if abs(value) > _SAFE_INTEGER_FLOAT:
            raise AttitudeInputError(
                "TIMESTAMP_PRECISION_LOSS",
                f"{path}={value!r} exceeds 2**53; send nanosecond timestamps "
                f"as JSON integer literals to avoid float rounding",
                index=index,
                path=path,
            )
        value = int(value)
    return value


def _require_finite_float(value: Any, path: str, index: int | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"{path} must be a number, got {type(value).__name__}",
            index=index,
            path=path,
        )
    component = float(value)
    if not math.isfinite(component):
        raise AttitudeInputError(
            "NON_FINITE_COMPONENT",
            f"{path} must be a finite number, got {component!r}",
            index=index,
            path=path,
        )
    return component


# ---------------------------------------------------------------------------
# quaternion math (component order is always [w, x, y, z])
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def _negate_in_place(q: List[float]) -> None:
    for i in range(QUATERNION_COMPONENTS):
        q[i] = -q[i]


def _normalize(q: List[float]) -> List[float]:
    # math.hypot is robust against overflow/underflow of the naive sum of
    # squares, so only an exactly-zero quaternion yields a zero norm.
    norm = math.hypot(q[0], q[1], q[2], q[3])
    return [c / norm for c in q]


def _slerp(q0: List[float], q1: List[float], u: float) -> List[float]:
    """Spherical linear interpolation along the shortest rotation arc."""
    if u <= 0.0:
        return list(q0)
    if u >= 1.0:
        return list(q1)

    d = _dot(q0, q1)
    if d < 0.0:  # defensive: callers pre-align, but stay correct standalone
        q1 = [-c for c in q1]
        d = -d
    if d > 1.0:  # clamp float rounding on (nearly) identical quaternions
        d = 1.0

    if d > _SMALL_ANGLE_DOT:
        # sin(a*theta)/sin(theta) = a * (1 + (1 - a^2) * theta^2 / 6) + O(theta^4)
        # with theta^2 ~= 2*(1 - d); the O(theta^4) error is below 1e-17.
        theta2 = max(0.0, 2.0 * (1.0 - d))
        w0 = (1.0 - u) * (1.0 + (1.0 - (1.0 - u) * (1.0 - u)) * theta2 / 6.0)
        w1 = u * (1.0 + (1.0 - u * u) * theta2 / 6.0)
    else:
        theta = math.acos(d)
        sin_theta = math.sin(theta)
        w0 = math.sin((1.0 - u) * theta) / sin_theta
        w1 = math.sin(u * theta) / sin_theta

    result = [w0 * a + w1 * b for a, b in zip(q0, q1)]
    return _normalize(result)


def _apply_output_sign_convention(quats: List[List[float]]) -> None:
    """Re-sign results in place for a continuous exposure sequence.

    The first quaternion is canonicalized so its first non-zero component
    is positive; every subsequent quaternion keeps the equivalent sign
    (q vs -q) whose dot product with its predecessor is non-negative.
    """
    if not quats:
        return
    first = quats[0]
    for component in first:
        if component != 0.0:
            if component < 0.0:
                _negate_in_place(first)
            break
    for previous, current in zip(quats, quats[1:]):
        if _dot(previous, current) < 0.0:
            _negate_in_place(current)


# ---------------------------------------------------------------------------
# request validation + interpolation
# ---------------------------------------------------------------------------

def _validate_samples(raw_samples: Any) -> tuple[List[int], List[List[float]]]:
    if not isinstance(raw_samples, list):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"samples must be an array of {MIN_SAMPLES}..{MAX_SAMPLES} attitude samples",
            path="samples",
        )
    if not MIN_SAMPLES <= len(raw_samples) <= MAX_SAMPLES:
        raise AttitudeInputError(
            "SAMPLE_COUNT_OUT_OF_RANGE",
            f"samples must contain between {MIN_SAMPLES} and {MAX_SAMPLES} "
            f"entries, got {len(raw_samples)}",
            path="samples",
            context={"count": len(raw_samples)},
        )

    times: List[int] = []
    quats: List[List[float]] = []
    for i, sample in enumerate(raw_samples):
        path = f"samples[{i}]"
        if not isinstance(sample, dict):
            raise AttitudeInputError(
                "INVALID_TYPE",
                f"{path} must be an object with 't' and 'q' fields",
                index=i,
                path=path,
            )
        t = _require_int_ns(_require_field(sample, "t", f"{path}.t", i), f"{path}.t", i)
        raw_q = _require_field(sample, "q", f"{path}.q", i)
        if not isinstance(raw_q, list) or len(raw_q) != QUATERNION_COMPONENTS:
            raise AttitudeInputError(
                "INVALID_QUATERNION",
                f"{path}.q must be an array of {QUATERNION_COMPONENTS} "
                f"numbers in [w, x, y, z] order",
                index=i,
                path=f"{path}.q",
            )
        q = [
            _require_finite_float(component, f"{path}.q[{j}]", i)
            for j, component in enumerate(raw_q)
        ]
        if math.hypot(q[0], q[1], q[2], q[3]) == 0.0:
            raise AttitudeInputError(
                "ZERO_QUATERNION",
                f"{path}.q is a zero quaternion; an attitude quaternion "
                f"must be non-zero",
                index=i,
                path=f"{path}.q",
            )
        if times and t <= times[-1]:
            raise AttitudeInputError(
                "NON_INCREASING_SAMPLE_TIME",
                f"samples[{i}].t={t} must be strictly greater than "
                f"samples[{i - 1}].t={times[-1]}",
                index=i,
                path=f"{path}.t",
                context={"previous_index": i - 1, "previous_t": times[-1]},
            )
        times.append(t)
        quats.append(_normalize(q))

    # Adjacent rotations must be strictly less than 180 degrees; align the
    # sign chain so every consecutive pair follows the shortest arc.
    for i in range(1, len(quats)):
        d = _dot(quats[i - 1], quats[i])
        if abs(d) <= _AMBIGUOUS_DOT_THRESHOLD:
            raise AttitudeInputError(
                "AMBIGUOUS_180_DEGREE_ROTATION",
                f"rotation between samples[{i - 1}] and samples[{i}] is "
                f"180 degrees; the shortest interpolation arc is ambiguous",
                index=i,
                path=f"samples[{i}].q",
                context={"previous_index": i - 1},
            )
        if d < 0.0:
            _negate_in_place(quats[i])
    return times, quats


def _validate_queries(raw_queries: Any) -> List[int]:
    if not isinstance(raw_queries, list):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"queries must be an array of {MIN_QUERIES}..{MAX_QUERIES} timestamps",
            path="queries",
        )
    if not MIN_QUERIES <= len(raw_queries) <= MAX_QUERIES:
        raise AttitudeInputError(
            "QUERY_COUNT_OUT_OF_RANGE",
            f"queries must contain between {MIN_QUERIES} and {MAX_QUERIES} "
            f"entries, got {len(raw_queries)}",
            path="queries",
            context={"count": len(raw_queries)},
        )
    times = [
        _require_int_ns(value, f"queries[{i}]", i)
        for i, value in enumerate(raw_queries)
    ]
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            raise AttitudeInputError(
                "NON_INCREASING_QUERY_TIME",
                f"queries[{i}]={times[i]} must be strictly greater than "
                f"queries[{i - 1}]={times[i - 1]}",
                index=i,
                path=f"queries[{i}]",
                context={"previous_index": i - 1, "previous_t": times[i - 1]},
            )
    return times


def _validate_max_gap(raw_max_gap: Any) -> int:
    max_gap = _require_int_ns(raw_max_gap, "max_gap_ns")
    if max_gap < 0:
        raise AttitudeInputError(
            "NEGATIVE_MAX_GAP",
            f"max_gap_ns must be non-negative, got {max_gap}",
            path="max_gap_ns",
        )
    return max_gap


def interpolate_attitudes(payload: Any) -> List[Dict[str, Any]]:
    """Validate the request and interpolate attitudes at the query times.

    Returns a list of ``{"t": <ns>, "q": [w, x, y, z]}`` entries in query
    order.  Raises :class:`AttitudeInputError` on any validation problem;
    either every query is answered or none is (no partial results).
    """
    if not isinstance(payload, dict):
        raise AttitudeInputError(
            "INVALID_BODY",
            "request body must be a JSON object with 'samples', 'queries' "
            "and 'max_gap_ns'",
            path="body",
        )

    times, quats = _validate_samples(_require_field(payload, "samples", "samples", None))
    query_times = _validate_queries(_require_field(payload, "queries", "queries", None))
    max_gap = _validate_max_gap(_require_field(payload, "max_gap_ns", "max_gap_ns", None))

    first_t, last_t = times[0], times[-1]
    last_interval = len(times) - 2

    results: List[Dict[str, Any]] = []
    for query_index, t in enumerate(query_times):
        if t < first_t or t > last_t:
            raise AttitudeInputError(
                "QUERY_OUT_OF_RANGE",
                f"queries[{query_index}]={t} lies outside the sampled "
                f"interval [{first_t}, {last_t}]",
                index=query_index,
                path=f"queries[{query_index}]",
                context={"sample_start": first_t, "sample_end": last_t},
            )
        i = bisect_right(times, t) - 1
        if i > last_interval:  # query exactly at the final sample
            i = last_interval
        gap = times[i + 1] - times[i]
        if gap > max_gap:
            raise AttitudeInputError(
                "SAMPLE_GAP_EXCEEDED",
                f"queries[{query_index}]={t} is enclosed by samples[{i}] "
                f"(t={times[i]}) and samples[{i + 1}] (t={times[i + 1]}) "
                f"whose gap {gap} ns exceeds max_gap_ns={max_gap}",
                index=query_index,
                path=f"queries[{query_index}]",
                context={
                    "sample_index": i,
                    "interval": [times[i], times[i + 1]],
                    "gap_ns": gap,
                    "max_gap_ns": max_gap,
                },
            )
        u = (t - times[i]) / gap
        results.append({"t": t, "q": _slerp(quats[i], quats[i + 1], u)})

    result_quats = [entry["q"] for entry in results]
    _apply_output_sign_convention(result_quats)
    for entry in results:
        # normalize -0.0 to 0.0 for clean, reviewable output
        entry["q"] = [c + 0.0 for c in entry["q"]]
    return results
