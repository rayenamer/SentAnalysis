"""Compatibility entry point: uvicorn app.aggregation_api:app."""
import sys
from app.api import aggregation as _implementation
sys.modules[__name__] = _implementation
