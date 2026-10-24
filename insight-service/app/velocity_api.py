"""Compatibility entry point: uvicorn app.velocity_api:app."""
import sys
from app.api import velocity as _implementation
sys.modules[__name__] = _implementation
