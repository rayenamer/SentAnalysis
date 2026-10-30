from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.core.market_adapter import price_observations
from app.models.market import Candle


def candle(hour=12, minute=0):
    return Candle(ticker="AAPL", provider_symbol="AAPL", asset_type="stock", interval="1h",
                  timestamp=datetime(2026, 10, 27, hour, minute, tzinfo=timezone.utc),
                  close_timestamp=datetime(2026, 10, 27, hour + 1, minute, tzinfo=timezone.utc),
                  open=100, high=110, low=90, close=105, volume=10)


def test_existing_price_projection_reused_with_order_and_dedupe():
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    rows = price_observations([candle(14), candle(), candle()], available_at=now)
    assert len(rows) == 2
    assert rows[0] == candle().insight_price()


@pytest.mark.parametrize("case", ["unaligned", "incomplete", "daily", "conflict", "unknown"])
def test_missing_required_time_series_inputs_rejected(case):
    rows = [candle(minute=30)] if case == "unaligned" else [candle()]
    if case == "conflict":
        rows.append(candle().model_copy(update={"close": 106}))
    now = datetime(2026, 10, 27, 12 if case == "incomplete" else 23, tzinfo=timezone.utc)
    bucket = "daily" if case == "daily" else "unknown" if case == "unknown" else "hourly"
    with pytest.raises(ValueError):
        price_observations(rows, bucket, available_at=now)


def test_actual_analysis_contracts_with_available_aligned_observations():
    rows = price_observations([candle()], available_at=datetime(2026, 10, 28, tzinfo=timezone.utc))
    code = """
import json,sys
from app.models.correlation import CorrelationRequest
from app.models.granger import GrangerRequest
prices=json.load(sys.stdin)
for cls in (CorrelationRequest,GrangerRequest):
    cls.model_validate({'prices':prices,'sentiments':[],'time_bucket':'hourly'})
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2] / "insight-service",
                            input=json.dumps(rows), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
