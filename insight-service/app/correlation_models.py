"""Compatibility alias for app.models.correlation."""
import sys
from app.models import correlation as _implementation
sys.modules[__name__] = _implementation
