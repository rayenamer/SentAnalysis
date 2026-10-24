"""Compatibility alias for app.helper.validation_errors."""
import sys
from app.helper import validation_errors as _implementation
sys.modules[__name__] = _implementation
