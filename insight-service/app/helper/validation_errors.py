"""Keep FastAPI's 422 detail contract safe for non-finite invalid inputs.

Pydantic rejects non-finite values, but its error's `input` may retain NaN/Inf.
Starlette refuses those floats when serializing JSON. Only non-finite values
in error details become diagnostic strings; valid response values are untouched.
Register on every application: include_router does not copy exception handlers.
"""

from math import isfinite
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _finite_error_details(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value):
        return "NaN" if value != value else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _finite_error_details(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_error_details(item) for item in value]
    return value


async def request_validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Registration guarantees this subtype; keep the general Exception handler
    # signature expected by Starlette's handler registry.
    details = _finite_error_details(jsonable_encoder(cast(RequestValidationError, exc).errors()))
    return JSONResponse(status_code=422, content={"detail": details})


def install_validation_error_handler(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
