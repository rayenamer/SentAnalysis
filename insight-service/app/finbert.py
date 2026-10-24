"""Compatibility alias for app.core.sentiment.finbert."""
import sys
from app.core.sentiment import finbert as _implementation
sys.modules[__name__] = _implementation
