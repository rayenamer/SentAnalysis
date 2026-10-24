"""Compatibility alias for app.helper.datetime_utils."""
import sys
from app.helper import datetime_utils as _implementation
sys.modules[__name__] = _implementation
