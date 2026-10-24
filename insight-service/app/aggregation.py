"""Compatibility alias for app.core.aggregate."""
import sys
from app.core import aggregate as _implementation
sys.modules[__name__] = _implementation
