"""Compatibility alias for app.models.outliers."""
import sys
from app.models import outliers as _implementation
sys.modules[__name__] = _implementation
