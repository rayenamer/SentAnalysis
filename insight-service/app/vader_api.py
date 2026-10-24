"""Compatibility entry point: uvicorn app.vader_api:app."""
import sys
from app.api import vader as _implementation
sys.modules[__name__] = _implementation
