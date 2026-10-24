"""Compatibility alias for app.core.weighted_aggregation."""
import sys
from app.core import weighted_aggregation as _implementation
sys.modules[__name__] = _implementation
