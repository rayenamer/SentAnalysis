"""Compatibility alias for app.models.granger."""
import sys
from app.models import granger as _implementation
sys.modules[__name__] = _implementation
