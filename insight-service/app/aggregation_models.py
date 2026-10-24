"""Compatibility alias for app.models.aggregation."""
import sys
from app.models import aggregation as _implementation
sys.modules[__name__] = _implementation
