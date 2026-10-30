"""Real Loughran–McDonald financial sentiment, academic research use only.

Score = (positive token occurrences - negative token occurrences) /
        (positive token occurrences + negative token occurrences).
No matches => 0; label follows the score sign. Confidence is uncalibrated/None.
English word lookup without stemming, negation, translation or phrase handling.
"""
import csv
from functools import lru_cache
from pathlib import Path
import re
from types import MappingProxyType
import unicodedata

from app.helper.metrics import measure_model
from app.models.sentiment import AnalyzeResponse, SentimentResult

MODEL = "loughran_mcdonald"
DICTIONARY = Path(__file__).resolve().parents[2] / "data/loughran_mcdonald_2025.csv"
CATEGORIES = ("Positive", "Negative", "Uncertainty", "Litigious",
              "Strong_Modal", "Weak_Modal", "Constraining")
TOKENS = re.compile(r"[^\W\d_]+")


@lru_cache(maxsize=1)
def get_dictionary():
    """Load only the compact official sentiment categories once per process."""
    categories = {category: set() for category in CATEGORIES}
    with DICTIONARY.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Word", *CATEGORIES}.issubset(reader.fieldnames or ()):
            raise ValueError("Invalid Loughran–McDonald dictionary columns")
        for row in reader:
            word = row["Word"].casefold()
            if not word.isalpha():
                raise ValueError("Invalid Loughran–McDonald word")
            for category in CATEGORIES:
                # Positive year: membership. Negative year: removed category.
                if int(row[category]) > 0:
                    categories[category].add(word)
    if not categories["Positive"] or not categories["Negative"]:
        raise ValueError("Empty Loughran–McDonald sentiment categories")
    return MappingProxyType({name: frozenset(words) for name, words in categories.items()})


def score_text(text):
    dictionary = get_dictionary()
    tokens = TOKENS.findall(unicodedata.normalize("NFC", text).casefold())
    counts = {category: sum(token in words for token in tokens)
              for category, words in dictionary.items()}
    positive, negative = counts["Positive"], counts["Negative"]
    total = positive + negative
    score = (positive - negative) / total if total else 0.0
    return score, counts


@measure_model(MODEL)
def analyze_documents(request):
    if request.model != MODEL:
        raise ValueError("Expected loughran_mcdonald model")
    results = []
    for document in request.documents:
        score, _ = score_text(document.text)
        results.append(SentimentResult(document_id=document.document_id, model=MODEL,
            label="positive" if score > 0 else "negative" if score < 0 else "neutral",
            sentiment_score=score, confidence=None, tickers=list(document.tickers)))
    return AnalyzeResponse(results=results)
