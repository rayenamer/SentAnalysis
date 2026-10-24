"""Compatibility entry point: uvicorn app.correlation_api:app."""
import sys
from app.api import correlation as _implementation
sys.modules[__name__] = _implementation
