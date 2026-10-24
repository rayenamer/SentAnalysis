"""Compatibility entry point: uvicorn app.outliers_api:app."""
import sys
from app.api import outliers as _implementation
sys.modules[__name__] = _implementation
