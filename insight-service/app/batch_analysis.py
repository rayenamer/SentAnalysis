"""Compatibility alias for app.core.batch."""
import sys
from app.core import batch as _implementation
sys.modules[__name__] = _implementation
