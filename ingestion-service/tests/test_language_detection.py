"""Fixed hand-labelled regression set; not a population accuracy estimate."""
import json
from pathlib import Path
import re

import pytest
from app.core.transform import _WORDS, detect_language

CASES = json.loads((Path(__file__).parent / "fixtures/language_cases.json").read_text(encoding="utf-8"))


def baseline(text):
    words = re.findall(r"[^\W\d_]+", text.casefold())
    if len(words) < 12:
        return "und"
    scores = sorted(((len(set(words) & markers), code) for code, markers in _WORDS.items()), reverse=True)
    best, code = scores[0]
    return code if best >= 5 and best >= scores[1][0] + 3 else "und"


@pytest.mark.parametrize("text,expected", CASES)
def test_fixed_language_cases(text, expected):
    assert detect_language(text) == expected
    assert detect_language(text) == detect_language(text)


def test_improves_fixed_set_over_previous_detector():
    old = sum(baseline(text) == expected for text, expected in CASES)
    new = sum(detect_language(text) == expected for text, expected in CASES)
    assert new > old
