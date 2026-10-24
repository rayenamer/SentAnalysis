"""Compatibility alias for app.models.finbert."""
import sys
from app.models import finbert as _implementation
sys.modules[__name__] = _implementation
