"""HTTP API for the attitude interpolation service.

Exposes ``POST /api/attitudes/interpolate`` which projects low-rate INS
attitude samples onto camera exposure instants, plus ``GET /health`` for
the container health check.  All domain validation lives in
:mod:`app.attitude`; this layer only translates between JSON and Python.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .attitude import AttitudeInputError, interpolate_attitudes

app = FastAPI(
    title="Attitude Interpolation Service",
    version="1.0.0",
    description=(
        "Interpolates unit-quaternion INS attitude samples (w, x, y, z) "
        "onto camera exposure times along shortest rotation arcs."
    ),
)


def _error_response(exc: AttitudeInputError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.to_payload()})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/attitudes/interpolate")
async def interpolate(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error_response(
            AttitudeInputError(
                "INVALID_JSON",
                "request body must be a valid JSON object",
                path="body",
            )
        )
    try:
        attitudes = interpolate_attitudes(payload)
    except AttitudeInputError as exc:
        return _error_response(exc)
    return JSONResponse(status_code=200, content={"attitudes": attitudes})
