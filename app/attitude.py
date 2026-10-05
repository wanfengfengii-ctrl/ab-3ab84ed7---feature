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

# Rolling-shutter (progressive-scan) cameras expose each sensor row at a
# different instant within a frame.  rowCount bounds the sensor size, and
# the readout direction maps a zero-based row to its exposure sequence
# number (0 = first row exposed).
MIN_ROW_COUNT = 2
MAX_ROW_COUNT = 20000
DIRECTION_TOP_TO_BOTTOM = "top_to_bottom"
DIRECTION_BOTTOM_TO_TOP = "bottom_to_top"
_ROLLING_SHUTTER_DIRECTIONS = (DIRECTION_TOP_TO_BOTTOM, DIRECTION_BOTTOM_TO_TOP)

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


# ---------------------------------------------------------------------------
# rolling shutter (progressive exposure) support
# ---------------------------------------------------------------------------

def _validate_rolling_shutter(raw: Any) -> Dict[str, Any] | None:
    """Validate the optional ``rolling_shutter`` camera description.

    Returns a normalized ``{"row_count", "line_period_ns", "direction"}``
    mapping, or ``None`` when the field is omitted (legacy integer-queries
    mode).  Every problem is located to its configuration field path.
    """
    if raw is None:
        return None
    base = "rolling_shutter"
    if not isinstance(raw, dict):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"{base} must be an object with 'rowCount', 'linePeriodNs' "
            f"and 'direction' fields",
            path=base,
        )

    raw_row_count = _require_field(raw, "rowCount", f"{base}.rowCount", None)
    raw_line_period = _require_field(raw, "linePeriodNs", f"{base}.linePeriodNs", None)
    raw_direction = _require_field(raw, "direction", f"{base}.direction", None)

    # rowCount / linePeriodNs are configuration counts, not timestamps:
    # bools are rejected but plain JSON integers (of any magnitude) are fine.
    if isinstance(raw_row_count, bool) or not isinstance(raw_row_count, int):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"{base}.rowCount must be an integer, got "
            f"{type(raw_row_count).__name__}",
            path=f"{base}.rowCount",
        )
    if not MIN_ROW_COUNT <= raw_row_count <= MAX_ROW_COUNT:
        raise AttitudeInputError(
            "ROW_COUNT_OUT_OF_RANGE",
            f"{base}.rowCount must be between {MIN_ROW_COUNT} and "
            f"{MAX_ROW_COUNT}, got {raw_row_count}",
            path=f"{base}.rowCount",
            context={"rowCount": raw_row_count},
        )

    if isinstance(raw_line_period, bool) or not isinstance(raw_line_period, int):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"{base}.linePeriodNs must be a positive integer, got "
            f"{type(raw_line_period).__name__}",
            path=f"{base}.linePeriodNs",
        )
    if raw_line_period <= 0:
        raise AttitudeInputError(
            "NON_POSITIVE_LINE_PERIOD",
            f"{base}.linePeriodNs must be a positive integer, got "
            f"{raw_line_period}",
            path=f"{base}.linePeriodNs",
            context={"linePeriodNs": raw_line_period},
        )

    if not isinstance(raw_direction, str) or raw_direction not in _ROLLING_SHUTTER_DIRECTIONS:
        raise AttitudeInputError(
            "INVALID_DIRECTION",
            f"{base}.direction must be one of "
            f"{list(_ROLLING_SHUTTER_DIRECTIONS)}, got {raw_direction!r}",
            path=f"{base}.direction",
            context={"direction": raw_direction},
        )

    return {
        "row_count": raw_row_count,
        "line_period_ns": raw_line_period,
        "direction": raw_direction,
    }


def _validate_row_queries(
    raw_queries: Any, config: Dict[str, Any]
) -> List[Dict[str, int]]:
    """Validate per-row queries and derive each row's capture instant.

    Each query must be an object carrying an integer ``frameT`` and a
    zero-based ``row``.  The exposure sequence number is ``row`` for
    ``top_to_bottom`` readout and ``row_count - 1 - row`` for
    ``bottom_to_top``; the capture time is
    ``frameT + sequence * line_period_ns`` (exact integer nanoseconds).
    The derived instants must be strictly increasing in request order.
    """
    base = "queries"
    if not isinstance(raw_queries, list):
        raise AttitudeInputError(
            "INVALID_TYPE",
            f"{base} must be an array of {MIN_QUERIES}..{MAX_QUERIES} "
            f"objects with 'frameT' and 'row' when rolling_shutter is set",
            path=base,
        )
    if not MIN_QUERIES <= len(raw_queries) <= MAX_QUERIES:
        raise AttitudeInputError(
            "QUERY_COUNT_OUT_OF_RANGE",
            f"{base} must contain between {MIN_QUERIES} and {MAX_QUERIES} "
            f"entries, got {len(raw_queries)}",
            path=base,
            context={"count": len(raw_queries)},
        )

    row_count = config["row_count"]
    line_period = config["line_period_ns"]
    bottom_to_top = config["direction"] == DIRECTION_BOTTOM_TO_TOP

    queries: List[Dict[str, int]] = []
    last_t: int | None = None
    for i, raw_query in enumerate(raw_queries):
        path = f"{base}[{i}]"
        if not isinstance(raw_query, dict):
            raise AttitudeInputError(
                "INVALID_TYPE",
                f"{path} must be an object with 'frameT' and 'row' fields",
                index=i,
                path=path,
            )
        frame_t = _require_int_ns(
            _require_field(raw_query, "frameT", f"{path}.frameT", i),
            f"{path}.frameT",
            i,
        )
        raw_row = _require_field(raw_query, "row", f"{path}.row", i)
        if isinstance(raw_row, bool) or not isinstance(raw_row, int):
            raise AttitudeInputError(
                "INVALID_TYPE",
                f"{path}.row must be a zero-based integer row index, got "
                f"{type(raw_row).__name__}",
                index=i,
                path=f"{path}.row",
            )
        if not 0 <= raw_row < row_count:
            raise AttitudeInputError(
                "ROW_OUT_OF_RANGE",
                f"{path}.row={raw_row} is outside the valid zero-based row "
                f"range [0, {row_count - 1}] of the {row_count}-row sensor",
                index=i,
                path=f"{path}.row",
                context={"row": raw_row, "rowCount": row_count},
            )

        sequence = row_count - 1 - raw_row if bottom_to_top else raw_row
        t = frame_t + sequence * line_period
        if last_t is not None and t <= last_t:
            raise AttitudeInputError(
                "NON_INCREASING_QUERY_TIME",
                f"{path} derives capture time t={t} which must be strictly "
                f"greater than the previous derived t={last_t}",
                index=i,
                path=path,
                context={"previous_index": i - 1, "previous_t": last_t, "derived_t": t},
            )
        last_t = t
        queries.append({"frame_t": frame_t, "row": raw_row, "t": t})
    return queries


def interpolate_attitudes(payload: Any) -> List[Dict[str, Any]]:
    """Validate the request and interpolate attitudes at the query times.

    Without ``rolling_shutter`` each query is an integer nanosecond
    timestamp and each result is ``{"t": <ns>, "q": [w, x, y, z]}``.

    With ``rolling_shutter`` each query is ``{"frameT": <ns>, "row": k}``;
    the per-row capture instant is derived from the frame time, readout
    direction and line period, and each result additionally echoes
    ``frameT`` and ``row``: ``{"frameT", "row", "t", "q"}``.

    Results are returned in request order.  Raises
    :class:`AttitudeInputError` on any validation problem; either every
    query is answered or none is (no partial results).
    """
    if not isinstance(payload, dict):
        raise AttitudeInputError(
            "INVALID_BODY",
            "request body must be a JSON object with 'samples', 'queries' "
            "and 'max_gap_ns'",
            path="body",
        )

    times, quats = _validate_samples(_require_field(payload, "samples", "samples", None))
    config = _validate_rolling_shutter(payload.get("rolling_shutter"))
    max_gap = _validate_max_gap(_require_field(payload, "max_gap_ns", "max_gap_ns", None))
    raw_queries = _require_field(payload, "queries", "queries", None)
    if config is None:
        queries: List[Dict[str, Any]] = [
            {"t": t} for t in _validate_queries(raw_queries)
        ]
    else:
        queries = _validate_row_queries(raw_queries, config)

    first_t, last_t = times[0], times[-1]
    last_interval = len(times) - 2

    for query_index, query in enumerate(queries):
        t = query["t"]
        if t < first_t or t > last_t:
            context: Dict[str, Any] = {"sample_start": first_t, "sample_end": last_t}
            if config is not None:
                context.update(
                    {"frameT": query["frame_t"], "row": query["row"], "derived_t": t}
                )
                message = (
                    f"queries[{query_index}] derives t={t} which lies outside "
                    f"the sampled interval [{first_t}, {last_t}]"
                )
            else:
                message = (
                    f"queries[{query_index}]={t} lies outside the sampled "
                    f"interval [{first_t}, {last_t}]"
                )
            raise AttitudeInputError(
                "QUERY_OUT_OF_RANGE",
                message,
                index=query_index,
                path=f"queries[{query_index}]",
                context=context,
            )
        i = bisect_right(times, t) - 1
        if i > last_interval:  # query exactly at the final sample
            i = last_interval
        gap = times[i + 1] - times[i]
        if gap > max_gap:
            context = {
                "sample_index": i,
                "interval": [times[i], times[i + 1]],
                "gap_ns": gap,
                "max_gap_ns": max_gap,
            }
            if config is not None:
                context.update(
                    {"frameT": query["frame_t"], "row": query["row"], "derived_t": t}
                )
                subject = f"queries[{query_index}] derives t={t} which is"
            else:
                subject = f"queries[{query_index}]={t} is"
            raise AttitudeInputError(
                "SAMPLE_GAP_EXCEEDED",
                f"{subject} enclosed by samples[{i}] (t={times[i]}) and "
                f"samples[{i + 1}] (t={times[i + 1]}) whose gap {gap} ns "
                f"exceeds max_gap_ns={max_gap}",
                index=query_index,
                path=f"queries[{query_index}]",
                context=context,
            )
        u = (t - times[i]) / gap
        query["q"] = _slerp(quats[i], quats[i + 1], u)

    result_quats = [query["q"] for query in queries]
    _apply_output_sign_convention(result_quats)

    results: List[Dict[str, Any]] = []
    for query in queries:
        # normalize -0.0 to 0.0 for clean, reviewable output
        q = [c + 0.0 for c in query["q"]]
        if config is None:
            results.append({"t": query["t"], "q": q})
        else:
            results.append(
                {"frameT": query["frame_t"], "row": query["row"], "t": query["t"], "q": q}
            )
    return results
