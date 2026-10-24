"""Compatibility alias for app.core.signals.granger."""
import sys
from app.core.signals import granger as _implementation
sys.modules[__name__] = _implementation
