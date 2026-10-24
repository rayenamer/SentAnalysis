"""Compatibility alias for app.core.signals.velocity."""
import sys
from app.core.signals import velocity as _implementation
sys.modules[__name__] = _implementation
