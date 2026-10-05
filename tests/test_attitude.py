"""Unit tests for the core attitude interpolation logic."""

import math

import pytest

from app.attitude import (
    MAX_QUERIES,
    MAX_SAMPLES,
    AttitudeInputError,
    interpolate_attitudes,
)

SQRT2_2 = math.sqrt(0.5)
T0 = 1_700_000_000_000_000_000  # epoch-scale nanoseconds (exceeds 2**53 as float)


def q_z(angle):
    """Unit quaternion for a rotation of `angle` radians about +z."""
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def q_x(angle):
    return [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(q):
    return math.sqrt(dot(q, q))


def make_payload(samples, queries, max_gap_ns=10**9):
    return {
        "samples": [{"t": t, "q": list(q)} for t, q in samples],
        "queries": list(queries),
        "max_gap_ns": max_gap_ns,
    }


def simple_payload(angle=math.pi / 2, gap=10_000, queries=(0, 2_500, 5_000, 7_500, 10_000),
                   max_gap_ns=10_000):
    samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + gap, q_z(angle))]
    return make_payload(samples, [T0 + q for q in queries], max_gap_ns)


def assert_quat_close(actual, expected, tol=1e-12):
    assert len(actual) == 4
    for a, e in zip(actual, expected):
        assert abs(a - e) <= tol, f"{actual} != {expected}"


# ---------------------------------------------------------------------------
# happy path / numerics
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_midpoint_is_exact_half_rotation(self):
        result = interpolate_attitudes(simple_payload(queries=(5_000,)))
        assert_quat_close(result[0]["q"], q_z(math.pi / 4))

    def test_quarter_points_match_analytic_rotation(self):
        result = interpolate_attitudes(simple_payload())
        for entry, u in zip(result, (0.0, 0.25, 0.5, 0.75, 1.0)):
            assert_quat_close(entry["q"], q_z(u * math.pi / 2))

    def test_endpoints_return_the_sample_attitudes(self):
        result = interpolate_attitudes(simple_payload(queries=(0, 10_000)))
        assert_quat_close(result[0]["q"], [1.0, 0.0, 0.0, 0.0])
        assert_quat_close(result[1]["q"], q_z(math.pi / 2))

    def test_query_may_land_on_first_and_last_sample(self):
        payload = simple_payload(queries=(0, 10_000), max_gap_ns=10_000)
        result = interpolate_attitudes(payload)
        assert [entry["t"] for entry in result] == [T0, T0 + 10_000]

    def test_shortest_arc_when_sample_sign_flipped(self):
        # -q is the same rotation; interpolation must not take the long way.
        samples = [
            (T0, [1.0, 0.0, 0.0, 0.0]),
            (T0 + 10_000, [-c for c in q_z(math.pi / 2)]),
        ]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 5_000], max_gap_ns=10_000)
        )
        assert_quat_close(result[0]["q"], q_z(math.pi / 4))

    def test_shortest_arc_prefers_smaller_rotation(self):
        # 270 degrees one way is 90 degrees the other; must interpolate the
        # 90-degree arc, i.e. the midpoint is the 45-degree attitude.
        samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10_000, q_z(-math.pi / 2))]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 5_000], max_gap_ns=10_000)
        )
        assert_quat_close(result[0]["q"], q_z(-math.pi / 4))

    def test_small_angle_branch_is_accurate(self):
        angle = 1e-7  # exercises the series branch of slerp
        samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10_000, q_x(angle))]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 3_000], max_gap_ns=10_000)
        )
        assert_quat_close(result[0]["q"], q_x(0.3 * angle), tol=1e-15)

    def test_identical_adjacent_samples(self):
        samples = [(T0, q_z(0.4)), (T0 + 10_000, q_z(0.4))]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 5_000], max_gap_ns=10_000)
        )
        assert_quat_close(result[0]["q"], q_z(0.4))

    def test_non_unit_inputs_are_normalized(self):
        samples = [
            (T0, [2.0, 0.0, 0.0, 0.0]),
            (T0 + 10_000, [3.0 * SQRT2_2, 0.0, 0.0, 3.0 * SQRT2_2]),
        ]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 5_000], max_gap_ns=10_000)
        )
        assert_quat_close(result[0]["q"], q_z(math.pi / 4))

    def test_tiny_quaternion_is_normalized_not_rejected(self):
        samples = [(T0, [1e-300, 0.0, 0.0, 0.0]), (T0 + 10_000, [1e-300, 0.0, 0.0, 1e-300])]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 5_000], max_gap_ns=10_000)
        )
        assert_quat_close(result[0]["q"], q_z(math.pi / 4))

    def test_epoch_scale_timestamps_keep_exact_fraction(self):
        # u must be exactly 0.5 even for ~1.7e18 ns epoch timestamps.
        result = interpolate_attitudes(simple_payload(queries=(5_000,)))
        assert_quat_close(result[0]["q"], q_z(math.pi / 4), tol=1e-15)

    def test_long_chain_matches_analytic_composition(self):
        n = 50
        step = 0.05  # radians per sample
        samples = [(T0 + 1_000 * k, q_z(step * k)) for k in range(n)]
        queries = [T0 + 10_250, T0 + 25_500, T0 + 49_000]
        result = interpolate_attitudes(make_payload(samples, queries, max_gap_ns=1_000))
        assert_quat_close(result[0]["q"], q_z(step * 10.25), tol=1e-12)
        assert_quat_close(result[1]["q"], q_z(step * 25.5), tol=1e-12)
        assert_quat_close(result[2]["q"], q_z(step * 49.0), tol=1e-12)

    def test_results_are_unit_norm_within_1e_9(self):
        payload = simple_payload()
        for entry in interpolate_attitudes(payload):
            assert abs(norm(entry["q"]) - 1.0) <= 1e-9

    def test_results_returned_in_query_order(self):
        result = interpolate_attitudes(simple_payload())
        assert [entry["t"] for entry in result] == [T0 + q for q in (0, 2_500, 5_000, 7_500, 10_000)]


# ---------------------------------------------------------------------------
# output sign convention
# ---------------------------------------------------------------------------

class TestSignConvention:
    def test_first_result_first_nonzero_component_is_positive(self):
        # Samples with negative-leading-component attitudes; the first
        # result must be re-signed to a positive leading component.
        samples = [
            (T0, [-1.0, 0.0, 0.0, 0.0]),
            (T0 + 10_000, [-c for c in q_z(math.pi / 2)]),
        ]
        result = interpolate_attitudes(
            make_payload(samples, [T0, T0 + 5_000, T0 + 10_000], max_gap_ns=10_000)
        )
        first = result[0]["q"]
        leading = next(c for c in first if c != 0.0)
        assert leading > 0.0

    def test_consecutive_results_have_nonnegative_dot(self):
        # A 170-degree sweep forces raw slerp outputs to flip sign relative
        # to each other; the convention must keep dots non-negative.
        samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10_000, q_z(math.radians(170.0)))]
        queries = [T0 + k * 1_000 for k in range(11)]
        result = interpolate_attitudes(make_payload(samples, queries, max_gap_ns=10_000))
        quats = [entry["q"] for entry in result]
        for previous, current in zip(quats, quats[1:]):
            assert dot(previous, current) >= 0.0

    def test_sequence_is_continuous_across_sample_boundary(self):
        samples = [
            (T0, q_z(0.0)),
            (T0 + 10_000, q_z(0.3)),
            (T0 + 20_000, q_z(0.6)),
        ]
        queries = [T0 + 9_999, T0 + 10_000, T0 + 10_001]
        result = interpolate_attitudes(make_payload(samples, queries, max_gap_ns=10_000))
        q_before, q_at, q_after = (entry["q"] for entry in result)
        # adjacent exposures are ~3e-5 rad apart: dots must be ~1
        assert dot(q_before, q_at) > 1.0 - 1e-8
        assert dot(q_at, q_after) > 1.0 - 1e-8

    def test_leading_zero_components_are_skipped_for_sign(self):
        # the sign anchor is the first *non-zero* component of the first
        # result; here the y/z components are exactly zero
        samples = [(T0, q_x(0.2)), (T0 + 10_000, q_x(0.4))]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 5_000], max_gap_ns=10_000)
        )
        leading = next(c for c in result[0]["q"] if c != 0.0)
        assert leading > 0.0


# ---------------------------------------------------------------------------
# validation: samples
# ---------------------------------------------------------------------------

class TestSampleValidation:
    @pytest.mark.parametrize("count", [0, 1, MAX_SAMPLES + 1])
    def test_sample_count_out_of_range(self, count):
        payload = make_payload([(T0 + k, [1.0, 0.0, 0.0, 0.0]) for k in range(count)], [T0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "SAMPLE_COUNT_OUT_OF_RANGE"

    def test_max_sample_count_accepted(self):
        samples = [(T0 + k, [1.0, 0.0, 0.0, 0.0]) for k in range(MAX_SAMPLES)]
        result = interpolate_attitudes(
            make_payload(samples, [T0, T0 + MAX_SAMPLES - 1], max_gap_ns=1)
        )
        assert len(result) == 2

    def test_zero_quaternion_rejected_with_index(self):
        payload = make_payload(
            [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 1, [1.0, 0.0, 0.0, 0.0]), (T0 + 2, [0.0, 0.0, 0.0, 0.0])],
            [T0],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "ZERO_QUATERNION"
        assert err.index == 2
        assert err.path == "samples[2].q"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_component_rejected(self, bad):
        payload = make_payload(
            [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 1, [1.0, bad, 0.0, 0.0])],
            [T0],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "NON_FINITE_COMPONENT"
        assert excinfo.value.index == 1

    def test_wrong_quaternion_length_rejected(self):
        payload = make_payload([(T0, [1.0, 0.0, 0.0]), (T0 + 1, [1.0, 0.0, 0.0, 0.0])], [T0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "INVALID_QUATERNION"
        assert excinfo.value.index == 0

    def test_non_numeric_component_rejected(self):
        payload = make_payload([(T0, [1.0, "x", 0.0, 0.0]), (T0 + 1, [1.0, 0.0, 0.0, 0.0])], [T0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "INVALID_TYPE"

    def test_non_increasing_sample_times_rejected_with_index(self):
        payload = make_payload(
            [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 5, [1.0, 0.0, 0.0, 0.0]), (T0 + 5, [1.0, 0.0, 0.0, 0.0])],
            [T0],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "NON_INCREASING_SAMPLE_TIME"
        assert err.index == 2

    def test_decreasing_sample_times_rejected(self):
        payload = make_payload(
            [(T0 + 5, [1.0, 0.0, 0.0, 0.0]), (T0, [1.0, 0.0, 0.0, 0.0])],
            [T0],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "NON_INCREASING_SAMPLE_TIME"
        assert excinfo.value.index == 1

    def test_non_integer_sample_time_rejected(self):
        payload = make_payload([(0, [1.0, 0.0, 0.0, 0.0]), (1_000_000.5, [1.0, 0.0, 0.0, 0.0])], [0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "NON_INTEGER_TIMESTAMP"

    def test_imprecise_float_timestamp_rejected(self):
        # 1.7e18 as a float literal cannot be represented exactly
        payload = make_payload(
            [(0, [1.0, 0.0, 0.0, 0.0]), (1.7e18, [1.0, 0.0, 0.0, 0.0])],
            [0],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "TIMESTAMP_PRECISION_LOSS"

    def test_boolean_time_rejected(self):
        payload = make_payload([(T0, [1.0, 0.0, 0.0, 0.0]), (True, [1.0, 0.0, 0.0, 0.0])], [T0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "INVALID_TYPE"

    def test_missing_sample_field_rejected(self):
        payload = {"samples": [{"t": T0}, {"t": T0 + 1, "q": [1, 0, 0, 0]}],
                   "queries": [T0], "max_gap_ns": 1}
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "MISSING_FIELD"
        assert excinfo.value.path == "samples[0].q"


# ---------------------------------------------------------------------------
# validation: 180-degree ambiguity
# ---------------------------------------------------------------------------

class TestAmbiguousRotation:
    def test_exact_180_degree_rotation_rejected(self):
        payload = make_payload(
            [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 1, [0.0, 1.0, 0.0, 0.0])],
            [T0],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "AMBIGUOUS_180_DEGREE_ROTATION"
        assert err.index == 1
        assert err.context["previous_index"] == 0

    def test_composed_180_degree_rotation_rejected(self):
        q0 = [math.cos(0.3), math.sin(0.3) / math.sqrt(3.0),
              math.sin(0.3) / math.sqrt(3.0), math.sin(0.3) / math.sqrt(3.0)]
        q1 = quat_mul(q0, [0.0, 1.0, 0.0, 0.0])  # rotate 180 deg about x
        payload = make_payload([(T0, q0), (T0 + 1, q1)], [T0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "AMBIGUOUS_180_DEGREE_ROTATION"

    def test_just_under_180_degrees_accepted(self):
        angle = math.radians(179.999)
        payload = make_payload([(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10, q_z(angle))], [T0 + 5])
        result = interpolate_attitudes(payload)
        assert_quat_close(result[0]["q"], q_z(angle / 2.0), tol=1e-9)

    def test_180_degree_pair_rejected_even_without_queries_inside(self):
        # validation is global: an ambiguous adjacent pair invalidates the
        # request no matter where the queries fall
        samples = [
            (T0, [1.0, 0.0, 0.0, 0.0]),
            (T0 + 1, [1.0, 0.0, 0.0, 0.0]),
            (T0 + 2, [0.0, 0.0, 0.0, 1.0]),
        ]
        payload = make_payload(samples, [T0], max_gap_ns=10)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "AMBIGUOUS_180_DEGREE_ROTATION"
        assert excinfo.value.index == 2


# ---------------------------------------------------------------------------
# validation: queries and gap limit
# ---------------------------------------------------------------------------

class TestQueryValidation:
    @pytest.mark.parametrize("count", [0, MAX_QUERIES + 1])
    def test_query_count_out_of_range(self, count):
        queries = [T0 + k for k in range(count)]
        payload = make_payload([(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + MAX_QUERIES + 1, [1.0, 0.0, 0.0, 0.0])], queries)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "QUERY_COUNT_OUT_OF_RANGE"

    def test_max_query_count_accepted(self):
        samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + MAX_QUERIES - 1, [1.0, 0.0, 0.0, 0.0])]
        queries = [T0 + k for k in range(MAX_QUERIES)]
        result = interpolate_attitudes(make_payload(samples, queries, max_gap_ns=MAX_QUERIES))
        assert len(result) == MAX_QUERIES

    def test_non_increasing_queries_rejected_with_index(self):
        payload = make_payload(
            [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10, [1.0, 0.0, 0.0, 0.0])],
            [T0 + 2, T0 + 5, T0 + 5],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "NON_INCREASING_QUERY_TIME"
        assert err.index == 2

    @pytest.mark.parametrize("offset", [-1, 11])
    def test_query_outside_sample_range_rejected(self, offset):
        payload = make_payload(
            [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10, [1.0, 0.0, 0.0, 0.0])],
            [T0 + offset],
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "QUERY_OUT_OF_RANGE"
        assert err.index == 0

    def test_gap_exceeding_limit_rejected_with_indices(self):
        samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10_000, [1.0, 0.0, 0.0, 0.0])]
        payload = make_payload(samples, [T0 + 5_000], max_gap_ns=9_999)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        err = excinfo.value
        assert err.code == "SAMPLE_GAP_EXCEEDED"
        assert err.index == 0
        assert err.context["sample_index"] == 0
        assert err.context["gap_ns"] == 10_000

    def test_gap_exactly_at_limit_accepted(self):
        samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10_000, q_z(0.5))]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 5_000], max_gap_ns=10_000)
        )
        assert_quat_close(result[0]["q"], q_z(0.25))

    def test_gap_limit_applies_to_enclosing_interval_only(self):
        # a long gap elsewhere does not matter if no query falls inside it
        samples = [
            (T0, [1.0, 0.0, 0.0, 0.0]),
            (T0 + 100, q_z(0.2)),
            (T0 + 10_000_000, q_z(0.4)),
        ]
        result = interpolate_attitudes(
            make_payload(samples, [T0 + 50], max_gap_ns=100)
        )
        assert_quat_close(result[0]["q"], q_z(0.1))

    def test_query_at_endpoint_uses_adjacent_interval_for_gap(self):
        samples = [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 10_000, q_z(0.2))]
        payload = make_payload(samples, [T0 + 10_000], max_gap_ns=5_000)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "SAMPLE_GAP_EXCEEDED"

    def test_negative_max_gap_rejected(self):
        payload = make_payload(
            [(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 1, [1.0, 0.0, 0.0, 0.0])],
            [T0],
            max_gap_ns=-1,
        )
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "NEGATIVE_MAX_GAP"

    def test_no_partial_results_on_late_failure(self):
        # first query is fine, second one is in an over-long gap: the whole
        # request must fail rather than return the first result
        samples = [
            (T0, [1.0, 0.0, 0.0, 0.0]),
            (T0 + 100, q_z(0.2)),
            (T0 + 10_000_000, q_z(0.4)),
        ]
        payload = make_payload(samples, [T0 + 50, T0 + 5_000_000], max_gap_ns=100)
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "SAMPLE_GAP_EXCEEDED"
        assert excinfo.value.index == 1


# ---------------------------------------------------------------------------
# validation: envelope
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_body_must_be_object(self):
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes([1, 2, 3])
        assert excinfo.value.code == "INVALID_BODY"

    @pytest.mark.parametrize("field", ["samples", "queries", "max_gap_ns"])
    def test_missing_top_level_field(self, field):
        payload = make_payload([(T0, [1.0, 0.0, 0.0, 0.0]), (T0 + 1, [1.0, 0.0, 0.0, 0.0])], [T0])
        del payload[field]
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        assert excinfo.value.code == "MISSING_FIELD"
        assert excinfo.value.path == field

    def test_error_payload_shape(self):
        payload = make_payload([(T0, [0.0, 0.0, 0.0, 0.0]), (T0 + 1, [1.0, 0.0, 0.0, 0.0])], [T0])
        with pytest.raises(AttitudeInputError) as excinfo:
            interpolate_attitudes(payload)
        data = excinfo.value.to_payload()
        assert data["code"] == "ZERO_QUATERNION"
        assert data["index"] == 0
        assert data["path"] == "samples[0].q"
        assert "message" in data
