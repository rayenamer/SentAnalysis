"""Compatibility entry point: uvicorn app.finbert_api:app."""
import sys
from app.api import sentiment as _implementation
sys.modules[__name__] = _implementation
