"""Compatibility entry point: uvicorn app.extensions_api:app."""
import sys
from app.api import extensions as _implementation
sys.modules[__name__] = _implementation
