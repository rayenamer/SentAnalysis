"""Compatibility entry point: uvicorn app.granger_api:app."""
import sys
from app.api import granger as _implementation
sys.modules[__name__] = _implementation
