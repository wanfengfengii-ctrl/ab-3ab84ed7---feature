"""API tests for the attitude interpolation service (FastAPI TestClient)."""

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

T0 = 1_700_000_000_000_000_000
SQRT2_2 = math.sqrt(0.5)


def valid_payload(**overrides):
    payload = {
        "samples": [
            {"t": T0, "q": [1.0, 0.0, 0.0, 0.0]},
            {"t": T0 + 10_000, "q": [SQRT2_2, 0.0, 0.0, SQRT2_2]},
        ],
        "queries": [T0, T0 + 2_500, T0 + 5_000, T0 + 7_500, T0 + 10_000],
        "max_gap_ns": 10_000,
    }
    payload.update(overrides)
    return payload


def post(payload):
    return client.post("/api/attitudes/interpolate", json=payload)


class TestHappyPath:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_valid_request_returns_attitudes_in_query_order(self):
        response = post(valid_payload())
        assert response.status_code == 200
        body = response.json()
        attitudes = body["attitudes"]
        assert [a["t"] for a in attitudes] == valid_payload()["queries"]
        midpoint = attitudes[2]["q"]
        expected = [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)]
        for actual, want in zip(midpoint, expected):
            assert abs(actual - want) <= 1e-9

    def test_results_are_unit_and_sign_continuous(self):
        response = post(valid_payload())
        quats = [a["q"] for a in response.json()["attitudes"]]
        for q in quats:
            assert abs(sum(c * c for c in q) - 1.0) <= 1e-9
        leading = next(c for c in quats[0] if c != 0.0)
        assert leading > 0.0
        for previous, current in zip(quats, quats[1:]):
            assert sum(a * b for a, b in zip(previous, current)) >= 0.0

    def test_float_timestamps_that_are_integral_accepted(self):
        payload = valid_payload()
        payload["samples"] = [
            {"t": 1_000_000, "q": [1.0, 0.0, 0.0, 0.0]},
            {"t": 1_010_000.0, "q": [SQRT2_2, 0.0, 0.0, SQRT2_2]},
        ]
        payload["queries"] = [1_005_000.0]
        assert post(payload).status_code == 200


class TestErrorResponses:
    def test_malformed_json_is_400(self):
        response = client.post(
            "/api/attitudes/interpolate",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_JSON"

    def test_zero_quaternion_error_locates_index(self):
        payload = valid_payload()
        payload["samples"][1]["q"] = [0.0, 0.0, 0.0, 0.0]
        response = post(payload)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "ZERO_QUATERNION"
        assert detail["index"] == 1
        assert detail["path"] == "samples[1].q"

    def test_nan_component_rejected(self):
        # strict JSON encoders refuse NaN, so post the raw body a lenient
        # client could send; the server must still reject it with an index
        body = (
            '{"samples": ['
            f'{{"t": {T0}, "q": [1.0, NaN, 0.0, 0.0]}},'
            f'{{"t": {T0 + 10_000}, "q": [1.0, 0.0, 0.0, 0.0]}}'
            f'], "queries": [{T0}], "max_gap_ns": 10000}}'
        )
        response = client.post(
            "/api/attitudes/interpolate",
            content=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "NON_FINITE_COMPONENT"
        assert detail["index"] == 0

    def test_non_increasing_sample_times(self):
        payload = valid_payload()
        payload["samples"][1]["t"] = T0
        response = post(payload)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "NON_INCREASING_SAMPLE_TIME"
        assert detail["index"] == 1

    def test_180_degree_ambiguity(self):
        payload = valid_payload()
        payload["samples"][1]["q"] = [0.0, 1.0, 0.0, 0.0]
        response = post(payload)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "AMBIGUOUS_180_DEGREE_ROTATION"
        assert detail["index"] == 1

    def test_gap_exceeded(self):
        payload = valid_payload(max_gap_ns=5_000)
        response = post(payload)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "SAMPLE_GAP_EXCEEDED"
        assert detail["index"] == 0
        assert detail["sample_index"] == 0
        assert detail["gap_ns"] == 10_000

    def test_query_out_of_range(self):
        payload = valid_payload(queries=[T0 + 20_000])
        response = post(payload)
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "QUERY_OUT_OF_RANGE"

    def test_too_few_samples(self):
        response = post(valid_payload(samples=[{"t": T0, "q": [1, 0, 0, 0]}]))
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "SAMPLE_COUNT_OUT_OF_RANGE"

    def test_too_many_queries(self):
        response = post(valid_payload(queries=list(range(T0, T0 + 501))))
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "QUERY_COUNT_OUT_OF_RANGE"

    def test_missing_field(self):
        payload = valid_payload()
        del payload["max_gap_ns"]
        response = post(payload)
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "MISSING_FIELD"

    def test_error_response_contains_no_partial_results(self):
        payload = valid_payload(max_gap_ns=5_000)
        body = post(payload).json()
        assert "attitudes" not in body
        assert set(body.keys()) == {"detail"}

    def test_get_on_interpolate_not_allowed(self):
        response = client.get("/api/attitudes/interpolate")
        assert 400 <= response.status_code < 500
