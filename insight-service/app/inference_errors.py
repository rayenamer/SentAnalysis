"""Compatibility alias for app.helper.inference_errors."""
import sys
from app.helper import inference_errors as _implementation
sys.modules[__name__] = _implementation
