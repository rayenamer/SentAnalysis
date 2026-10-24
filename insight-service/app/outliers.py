"""Compatibility alias for app.core.signals.outliers."""
import sys
from app.core.signals import outliers as _implementation
sys.modules[__name__] = _implementation
