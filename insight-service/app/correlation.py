"""Compatibility alias for app.core.signals.correlation."""
import sys
from app.core.signals import correlation as _implementation
sys.modules[__name__] = _implementation
