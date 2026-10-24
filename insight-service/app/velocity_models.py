"""Compatibility alias for app.models.velocity."""
import sys
from app.models import velocity as _implementation
sys.modules[__name__] = _implementation
