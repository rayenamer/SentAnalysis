import csv
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.core.sentiment import loughran_mcdonald as lm
from app.main import create_app
from app.models.sentiment import AnalyzeRequest


@pytest.mark.parametrize("text,score", [
    ("Excellent success and improved results.", 1),
    ("Losses and bankruptcy.", -1),
    ("The company is based in Helsinki.", 0),
    ("Liability liabilities capital tax.", 0),
    ("", 0),
    ("Excellent loss", 0),
    ("Excellent excellent loss", 1 / 3),
    ("EXCELLENT! SUCCESS.", 1),
    ("Not excellent", 1),  # explicit bag-of-words limitation
    ("Café 東京 $AAPL 123", 0),
])
def test_real_dictionary_scoring(text, score):
    assert lm.score_text(text)[0] == pytest.approx(score)


def test_real_categories_cache_and_provenance():
    dictionary = lm.get_dictionary()
    assert lm.get_dictionary() is dictionary
    assert isinstance(dictionary["Positive"], frozenset)
    assert len(dictionary["Positive"]) == 347
    assert len(dictionary["Negative"]) == 2345
    assert "uncertain" in dictionary["Uncertainty"]
    assert lm.score_text("Uncertain uncertain")[1]["Uncertainty"] == 2
    metadata = json.loads(lm.DICTIONARY.with_suffix(".provenance.json").read_text())
    assert hashlib.sha256(lm.DICTIONARY.read_bytes()).hexdigest() == metadata["derived_csv_sha256"]
    with lm.DICTIONARY.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for category in lm.CATEGORIES:
                if int(row[category]) < 0:
                    assert row["Word"].casefold() not in dictionary[category]


def test_api_and_batch_preserve_contract_and_metrics():
    body = {"model": lm.MODEL, "documents": [
        {"document_id": "positive", "title": "Report", "text": "Excellent success.", "tickers": ["AAPL"]},
        {"document_id": "negative", "title": "Report", "text": "Losses bankruptcy.", "tickers": []},
        {"document_id": "neutral", "title": "Report", "text": "Liability capital.", "tickers": ["MSFT"]},
    ]}
    with TestClient(create_app(), client=("127.0.0.1", 1)) as client:
        response = client.post("/analyze", json=body)
        assert response.status_code == 200
        results = response.json()["results"]
        assert [row["label"] for row in results] == ["positive", "negative", "neutral"]
        assert [row["document_id"] for row in results] == ["positive", "negative", "neutral"]
        assert [row["tickers"] for row in results] == [["AAPL"], [], ["MSFT"]]
        assert all(row["confidence"] is None and row["model"] == lm.MODEL for row in results)
        batch = client.post("/analyze/batch", json={**body, "chunk_size": 2})
        assert batch.status_code == 200
        assert batch.json()["success_count"] == 3
        assert batch.json()["error_count"] == 0
        assert [row["result"] for row in batch.json()["outcomes"]] == results
        assert 'sentanalysis_documents_analyzed_total{category="loughran_mcdonald"} 6.0' in client.get("/metrics").text
        aggregate = client.post("/aggregate", json={
            "as_of": "2026-10-30T12:00:00Z", "windows": ["1h"],
            "results": [{**results[0], "published_at": "2026-10-30T11:30:00Z"}],
        })
        assert aggregate.status_code == 200
        assert lm.MODEL in aggregate.text


def test_wrong_model_rejected():
    request = AnalyzeRequest(documents=[{"document_id": "x", "title": "x", "text": "x"}])
    with pytest.raises(ValueError):
        lm.analyze_documents(request)


def test_missing_dictionary_is_safe_api_failure(monkeypatch, tmp_path):
    lm.get_dictionary.cache_clear()
    try:
        monkeypatch.setattr(lm, "DICTIONARY", tmp_path / "missing.csv")
        body = {"model": lm.MODEL, "documents": [
            {"document_id": "x", "title": "Report", "text": "Excellent."}]}
        with TestClient(create_app()) as client:
            response = client.post("/analyze", json=body)
            assert response.status_code == 503
            assert str(tmp_path) not in response.text
            batch = client.post("/analyze/batch", json=body)
            assert batch.status_code == 200
            assert batch.json()["error_count"] == 1
    finally:
        lm.get_dictionary.cache_clear()


def test_malformed_dictionary_rejected(monkeypatch, tmp_path):
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("Wrong,Columns\n", encoding="utf-8")
    lm.get_dictionary.cache_clear()
    try:
        monkeypatch.setattr(lm, "DICTIONARY", invalid)
        with pytest.raises(ValueError):
            lm.get_dictionary()
    finally:
        lm.get_dictionary.cache_clear()
