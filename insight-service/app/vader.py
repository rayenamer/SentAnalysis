"""Compatibility alias for app.core.sentiment.lexicon."""
import sys
from app.core.sentiment import lexicon as _implementation
sys.modules[__name__] = _implementation
