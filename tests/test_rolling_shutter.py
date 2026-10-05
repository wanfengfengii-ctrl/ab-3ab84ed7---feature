"""Unit tests for rolling-shutter (progressive exposure) support.

When ``rolling_shutter`` is present, each query carries a frame time and a
zero-based sensor row; the service derives the row's true capture instant
from the readout direction and line period, then runs the *same* shortest
arc SLERP / sign-continuity pipeline as the legacy integer-query mode.
"""

import math

import pytest

from app.attitude import (
    MAX_QUERIES,
    MAX_ROW_COUNT,
    AttitudeInputError,
    interpolate_attitudes,
)

T0 = 1_700_000_000_000_000_000  # epoch-scale nanoseconds (exceeds 2**53 as float)
SQRT2_2 = math.sqrt(0.5)


def q_z(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def samples(gap=10_000, end_angle=math.pi / 2):
    return [
        {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
        {"t": T0 + gap, "q": q_z(end_angle)},
    ]


def rolling_payload(rows, *, frame_ts=None, row_count=1_000, line_period_ns=2_500,
                    direction="top_to_bottom", gap=10_000, max_gap_ns=None):
    if frame_ts is None:
        frame_ts = [T0] * len(rows)
    return {
        "samples": samples(gap=gap),
        "queries": [{"frameT": ft, "row": row} for ft, row in zip(frame_ts, rows)],
        "max_gap_ns": gap if max_gap_ns is None else max_gap_ns,
        "rolling_shutter": {
            "rowCount": row_count,
            "linePeriodNs": line_period_ns,
            "direction": direction,
        },
    }


def legacy_payload(times, gap=10_000, max_gap_ns=None):
    return {
        "samples": samples(gap=gap),
        "queries": times,
        "max_gap_ns": gap if max_gap_ns is None else max_gap_ns,
    }


def assert_quat_close(actual, expected, tol=1e-12):
    assert len(actual) == 4
    for a, e in zip(actual, expected):
        assert abs(a - e) <= tol, f"{actual} != {expected}"


# ---------------------------------------------------------------------------
# time derivation
# ---------------------------------------------------------------------------

class TestTimeDerivation:
    def test_top_to_bottom_row_zero_coincides_with_frame_time(self):
        result = interpolate_attitudes(rolling_payload([0]))
        assert result[0]["t"] == T0
        assert result[0]["frameT"] == T0
        assert result[0]["row"] == 0

    def test_top_to_bottom_derives_frame_plus_row_times_period(self):
        result = interpolate_attitudes(rolling_payload([0, 1, 2, 3]))
        assert [e["t"] for e in result] == [T0 + 2_500 * k for k in range(4)]
        assert [e["row"] for e in result] == [0, 1, 2, 3]
        assert [e["frameT"] for e in result] == [T0] * 4

    def test_bottom_to_top_reverses_exposure_sequence(self):
        # row 3 is exposed first, row 0 last; request them in exposure
        # order so the derived instants are strictly increasing
        result = interpolate_attitudes(
            rolling_payload([3, 2, 1, 0], row_count=4,
                            direction="bottom_to_top")
        )
        assert [e["row"] for e in result] == [3, 2, 1, 0]
        assert [e["t"] for e in result] == [T0 + 2_500 * k for k in range(4)]

    def test_bottom_to_top_row_zero_is_exposed_last(self):
        result = interpolate_attitudes(
            rolling_payload([0], row_count=4, direction="bottom_to_top")
        )
        assert result[0]["t"] == T0 + 3 * 2_500

    def test_frame_time_advances_between_frames(self):
        # same row in two frames: derived times differ by the frame stride
        frame_ts = [T0, T0 + 1_000_000]
        result = interpolate_attitudes(
            rolling_payload([5, 5], frame_ts=frame_ts, row_count=100,
                            line_period_ns=1_000, gap=2_000_000)
        )
        assert [e["t"] for e in result] == [T0 + 5_000, T0 + 1_005_000]

    def test_derived_times_match_legacy_integer_queries_top_to_bottom(self):
        rows = [0, 1, 2, 3]
        rs = interpolate_attitudes(rolling_payload(rows))
        legacy = interpolate_attitudes(
            legacy_payload([T0 + 2_500 * k for k in range(4)])
        )
        for got, want in zip(rs, legacy):
            assert_quat_close(got["q"], want["q"])

    def test_derived_times_match_legacy_integer_queries_bottom_to_top(self):
        rows = [3, 2, 1, 0]
        rs = interpolate_attitudes(
            rolling_payload(rows, row_count=4, direction="bottom_to_top")
        )
        legacy = interpolate_attitudes(
            legacy_payload([T0 + 2_500 * k for k in range(4)])
        )
        for got, want in zip(rs, legacy):
            assert_quat_close(got["q"], want["q"])

    def test_both_directions_yield_same_attitude_at_same_instant(self):
        # the third exposure (sequence 2) is physical row 2 top-to-bottom
        # and physical row 1 bottom-to-top of a 4-row sensor
        common = dict(row_count=4, line_period_ns=2_500)
        down = interpolate_attitudes(
            rolling_payload([2], direction="top_to_bottom", **common)
        )
        up = interpolate_attitudes(
            rolling_payload([1], direction="bottom_to_top", **common)
        )
        assert down[0]["t"] == up[0]["t"] == T0 + 2 * 2_500
        assert_quat_close(down[0]["q"], up[0]["q"])


# ---------------------------------------------------------------------------
# response shape and sign convention
# ---------------------------------------------------------------------------

class TestResponseShape:
    def test_echoes_frame_t_row_and_derived_t(self):
        result = interpolate_attitudes(
            rolling_payload([1, 3], frame_ts=[T0, T0 + 100])
        )
        assert set(result[0].keys()) == {"frameT", "row", "t", "q"}
        assert result[0]["frameT"] == T0
        assert result[0]["row"] == 1
        assert result[0]["t"] == T0 + 2_500
        assert result[1]["frameT"] == T0 + 100
        assert result[1]["row"] == 3
        assert result[1]["t"] == T0 + 100 + 7_500

    def test_results_returned_in_request_order(self):
        rows = [3, 1, 2, 0]
        frame_ts = [T0 + k * 1_000_000 for k in range(4)]
        result = interpolate_attitudes(
            rolling_payload(rows, frame_ts=frame_ts, gap=5_000_000)
        )
        assert [e["row"] for e in result] == rows
        assert [e["frameT"] for e in result] == frame_ts

    def test_unit_quaternions_and_sign_continuity(self):
        # frame-to-frame jumps interleaved with within-frame rows; derived
        # instants are increasing, so the sign chain must stay continuous
        frame_ts = [T0, T0, T0 + 8_000, T0 + 8_000]
        rows = [0, 3, 0, 1]
        result = interpolate_attitudes(
            rolling_payload(rows, frame_ts=frame_ts, line_period_ns=500)
        )
        quats = [e["q"] for e in result]
        for q in quats:
            assert abs(dot(q, q) - 1.0) <= 1e-9
        leading = next(c for c in quats[0] if c != 0.0)
        assert leading > 0.0
        for previous, current in zip(quats, quats[1:]):
            assert dot(previous, current) >= 0.0


# ---------------------------------------------------------------------------
# validation: configuration
# ---------------------------------------------------------------------------

class TestRollingShutterConfigValidation:
    @pytest.mark.parametrize("bad_count", [1, 0, -1, MAX_ROW_COUNT + 1])
    def test_row_count_out_of_range(self, bad_count):
        payload = rolling_payload([0], row_count=bad_count)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "ROW_COUNT_OUT_OF_RANGE"
        assert err.path == "rolling_shutter.rowCount"

    @pytest.mark.parametrize("count", [2, MAX_ROW_COUNT])
    def test_row_count_bounds_accepted(self, count):
        payload = rolling_payload([0, count - 1], row_count=count,
                                  line_period_ns=1, gap=count)
        result = interpolate_attitudes(payload)
        assert len(result) == 2

    @pytest.mark.parametrize("bad_period", [0, -1, -10**9])
    def test_non_positive_line_period_rejected(self, bad_period):
        payload = rolling_payload([0], line_period_ns=bad_period)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "NON_POSITIVE_LINE_PERIOD"
        assert err.path == "rolling_shutter.linePeriodNs"

    @pytest.mark.parametrize("bad_direction", ["", "left_to_right", "TOP_TO_BOTTOM", 0])
    def test_invalid_direction_rejected(self, bad_direction):
        payload = rolling_payload([0], direction=bad_direction)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "INVALID_DIRECTION"
        assert err.path == "rolling_shutter.direction"

    @pytest.mark.parametrize(
        "field,bad_value",
        [("rowCount", 2.0), ("rowCount", True), ("linePeriodNs", 1.5),
         ("linePeriodNs", False), ("rowCount", "1000"), ("linePeriodNs", None)],
    )
    def test_wrong_config_type_rejected(self, field, bad_value):
        payload = rolling_payload([0])
        payload["rolling_shutter"][field] = bad_value
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.path == f"rolling_shutter.{field}"

    @pytest.mark.parametrize("field", ["rowCount", "linePeriodNs", "direction"])
    def test_missing_config_field_rejected(self, field):
        payload = rolling_payload([0])
        del payload["rolling_shutter"][field]
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "MISSING_FIELD"
        assert err.path == f"rolling_shutter.{field}"

    def test_rolling_shutter_must_be_object(self):
        payload = rolling_payload([0])
        payload["rolling_shutter"] = [1000, 2500, "top_to_bottom"]
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "INVALID_TYPE"
        assert excinfo.value.path == "rolling_shutter"

    def test_explicit_null_rolling_shutter_keeps_legacy_mode(self):
        payload = legacy_payload([T0 + 5_000])
        payload["rolling_shutter"] = None
        result = interpolate_attitudes(payload)
        assert set(result[0].keys()) == {"t", "q"}


# ---------------------------------------------------------------------------
# validation: per-row queries
# ---------------------------------------------------------------------------

class TestRowQueryValidation:
    @pytest.mark.parametrize("bad_row", [-1, 1_000, 10_000])
    def test_row_out_of_range_rejected_with_index(self, bad_row):
        payload = rolling_payload([0, bad_row])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "ROW_OUT_OF_RANGE"
        assert err.index == 1
        assert err.path == "queries[1].row"
        assert err.context["row"] == bad_row
        assert err.context["rowCount"] == 1_000

    def test_first_and_last_rows_accepted(self):
        result = interpolate_attitudes(
            rolling_payload([0, 3], row_count=4)
        )
        assert [e["t"] for e in result] == [T0, T0 + 3 * 2_500]

    @pytest.mark.parametrize("bad_row", [1.0, True, "0", None])
    def test_non_integer_row_rejected(self, bad_row):
        payload = rolling_payload([bad_row])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.path == "queries[0].row"

    @pytest.mark.parametrize("bad_query", [[0, T0], "row", 5, None])
    def test_query_must_be_object(self, bad_query):
        payload = rolling_payload([0])
        payload["queries"] = [bad_query]
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "INVALID_TYPE"
        assert err.index == 0
        assert err.path == "queries[0]"

    @pytest.mark.parametrize("field", ["frameT", "row"])
    def test_missing_query_field_rejected(self, field):
        payload = rolling_payload([0])
        del payload["queries"][0][field]
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "MISSING_FIELD"
        assert err.path == f"queries[0].{field}"
        assert err.index == 0

    def test_integer_query_rejected_when_rolling_shutter_enabled(self):
        payload = rolling_payload([0])
        payload["queries"] = [T0]
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "INVALID_TYPE"
        assert excinfo.value.path == "queries[0]"

    def test_object_query_rejected_in_legacy_mode(self):
        payload = legacy_payload([T0])
        payload["queries"] = [{"frameT": T0, "row": 0}]
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "INVALID_TYPE"
        assert excinfo.value.path == "queries[0]"

    def test_too_many_row_queries_rejected(self):
        payload = rolling_payload([0] * (MAX_QUERIES + 1), line_period_ns=1,
                                  gap=10**9, max_gap_ns=10**9)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "QUERY_COUNT_OUT_OF_RANGE"

    def test_frame_t_float_beyond_2_53_rejected(self):
        payload = rolling_payload([0])
        payload["queries"][0]["frameT"] = 1.7e18
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "TIMESTAMP_PRECISION_LOSS"


# ---------------------------------------------------------------------------
# validation: derived-time ordering, range and gaps
# ---------------------------------------------------------------------------

class TestDerivedTimeValidation:
    def test_descending_rows_top_to_bottom_rejected(self):
        payload = rolling_payload([2, 1])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "NON_INCREASING_QUERY_TIME"
        assert err.index == 1
        assert err.path == "queries[1]"
        assert err.context["derived_t"] == T0 + 2_500
        assert err.context["previous_t"] == T0 + 5_000

    def test_equal_derived_times_rejected(self):
        # same physical row in two frames at the same frameT
        payload = rolling_payload([5, 5], frame_ts=[T0, T0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "NON_INCREASING_QUERY_TIME"
        assert excinfo.value.index == 1

    def test_ascending_rows_bottom_to_top_rejected(self):
        payload = rolling_payload([0, 1], direction="bottom_to_top")
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "NON_INCREASING_QUERY_TIME"
        assert excinfo.value.index == 1

    def test_frame_time_must_compensate_readout_direction(self):
        # bottom-to-top: later physical rows expose earlier, so a later
        # frameT can still rescue strict ordering
        frame_ts = [T0, T0 + 10_000]
        result = interpolate_attitudes(
            rolling_payload([0, 1], frame_ts=frame_ts, row_count=4,
                            direction="bottom_to_top", line_period_ns=1_000,
                            gap=20_000)
        )
        assert [e["t"] for e in result] == [T0 + 3_000, T0 + 12_000]

    def test_derived_time_before_sample_range_rejected(self):
        payload = rolling_payload([0], frame_ts=[T0 - 10_000],
                                  line_period_ns=1)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "QUERY_OUT_OF_RANGE"
        assert err.index == 0
        assert err.context["derived_t"] == T0 - 10_000

    def test_derived_time_after_sample_range_rejected(self):
        payload = rolling_payload([0], frame_ts=[T0 + 10_001])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "QUERY_OUT_OF_RANGE"
        assert excinfo.value.index == 0

    def test_derived_time_at_sample_endpoints_accepted(self):
        result = interpolate_attitudes(
            rolling_payload([0, 0], frame_ts=[T0, T0 + 10_000])
        )
        assert [e["t"] for e in result] == [T0, T0 + 10_000]

    def test_derived_time_in_overlong_gap_rejected(self):
        payload = {
            "samples": [
                {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
                {"t": T0 + 10_000_000, "q": q_z(0.4)},
            ],
            "queries": [{"frameT": T0 + 5_000_000, "row": 0}],
            "max_gap_ns": 1_000,
            "rolling_shutter": {"rowCount": 4, "linePeriodNs": 1,
                                 "direction": "top_to_bottom"},
        }
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "SAMPLE_GAP_EXCEEDED"
        assert err.index == 0
        assert err.context["sample_index"] == 0
        assert err.context["gap_ns"] == 10_000_000
        assert err.context["derived_t"] == T0 + 5_000_000

    def test_row_offset_can_push_otherwise_ok_frame_into_gap(self):
        # frameT itself sits at the first sample, but the row readout delay
        # lands inside the over-long gap
        payload = {
            "samples": [
                {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
                {"t": T0 + 10_000_000, "q": q_z(0.4)},
            ],
            "queries": [{"frameT": T0, "row": 99}],
            "max_gap_ns": 1_000,
            "rolling_shutter": {"rowCount": 100, "linePeriodNs": 100_000,
                                 "direction": "top_to_bottom"},
        }
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "SAMPLE_GAP_EXCEEDED"
        assert excinfo.value.index == 0

    def test_no_partial_results_on_late_failure(self):
        # first query is fine; the second derives a time past the samples
        payload = rolling_payload([0, 0], frame_ts=[T0, T0 + 11_000])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "QUERY_OUT_OF_RANGE"
        assert excinfo.value.index == 1
