"""HTTP orchestration only: never turn article scores into historical snapshots.

The current raw aggregate is observed at actual computation time, not its
articles' publication dates. Historical snapshots and prices must be supplied
by the host. No storage, filling, time rounding or automatic market downloads.
Each downstream call is sequential, bounded by the existing Insight timeout
and response-size configuration. A failed stage preserves earlier results.
"""
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from app.helper.provider_http import SourceError, fetch_json
from app.models.analytics import (AnalyticsStage, SignalReports, WireAggregation,
    WireSignal, WireVelocity, WireOutliers, WireCorrelation, WireGranger, bucket_aligned)

SIGNAL_CONTRACTS = {
    "/signals/velocity": (WireVelocity, "velocity_per_hour"),
    "/signals/outliers": (WireOutliers, "z_score"),
    "/signals/correlation": (WireCorrelation, "pearson_r"),
    "/signals/granger": (WireGranger, "f_statistic"),
}


def unavailable(reason):
    return AnalyticsStage(status="unavailable", reason=reason)


class AnalyticsClient:
    def __init__(self, config, client):
        self.config, self.client = config, client

    async def call(self, path, body, contract, model, tickers):
        try:
            payload = await fetch_json(self.client, "POST", self.config.service_url + path,
                timeout=self.config.request_timeout_seconds,
                max_bytes=self.config.max_response_bytes, json=body)
            response_contract = SIGNAL_CONTRACTS[path][0] if contract is WireSignal else contract
            validated = response_contract.model_validate(payload)
            if contract is WireAggregation:
                if validated.as_of.isoformat() != datetime.fromisoformat(body["as_of"]).isoformat():
                    raise ValueError("Aggregation time mismatch")
                if sorted(row.window for row in validated.windows) != sorted(body["windows"]):
                    raise ValueError("Aggregation window mismatch")
                durations = {"1h": timedelta(hours=1), "24h": timedelta(days=1), "7d": timedelta(days=7)}
                if any(window.end != validated.as_of or window.start !=
                       validated.as_of - durations[window.window] for window in validated.windows):
                    raise ValueError("Aggregation interval mismatch")
                rows = [row for window in validated.windows for row in window.aggregates]
            else:
                rows = validated.results
            if any(row.model != model or row.ticker not in tickers for row in rows):
                raise ValueError("Analytics series identity mismatch")
            if contract is WireSignal and not any(
                    getattr(row, SIGNAL_CONTRACTS[path][1]) is not None for row in rows):
                reasons = {getattr(row, "diagnostic_reason", None) for row in rows} - {None}
                reason = next(iter(reasons)) if len(reasons) == 1 else "insufficient_history"
                return AnalyticsStage(status="unavailable", reason=reason, data=payload)
            return AnalyticsStage(status="success", data=payload)
        except SourceError as exc:
            return AnalyticsStage(status="error", reason=exc.code)
        except (ValidationError, ValueError):
            return AnalyticsStage(status="error", reason="invalid_response")

    async def enrich(self, documents, analysis, options):
        as_of = datetime.now(timezone.utc)
        scored = []
        for outcome in analysis.outcomes:
            if outcome.status != "success":
                continue
            document = documents[outcome.input_index]
            row = outcome.result.model_dump(mode="json", exclude={"probabilities"})
            row["published_at"] = document.published_at.isoformat() if document.published_at else None
            scored.append((row, document))
        tickers = {ticker for row, _ in scored for ticker in row["tickers"]}
        if scored:
            body = {"as_of": as_of.isoformat(), "windows": ["1h", "24h", "7d"],
                    "results": [row for row, _ in scored]}
            aggregation = await self.call("/aggregate", body, WireAggregation, analysis.model, tickers)
            weighted_rows = []
            for row, document in scored:
                engagement = document.metadata.get("num_comments")
                if type(engagement) not in {int, float} or not 0 <= engagement <= 1e12:
                    engagement = None
                weighted_rows.append({**row, "source": document.source, "engagement": engagement})
            weighted = await self.call("/aggregate/weighted", {**body, "results": weighted_rows},
                                       WireAggregation, analysis.model, tickers)
        else:
            aggregation = unavailable("no_successful_analysis")
            weighted = unavailable("no_successful_analysis")

        observations = [row.model_dump(mode="json") for row in options.history]
        if aggregation.status == "success":
            for window in aggregation.data["windows"]:
                if window["window"] == options.signal_window:
                    for row in window["aggregates"]:
                        observations.append({"ticker": row["ticker"], "model": row["model"],
                            "sentiment_score": row["mean_sentiment"], "observed_at": as_of.isoformat(),
                            "aggregation_window": options.signal_window, "score_kind": "raw"})
        series_tickers = {row["ticker"] for row in observations}
        if observations:
            velocity_rows = [{key: row[key] for key in
                ("ticker", "model", "sentiment_score", "observed_at")} for row in observations]
            velocity = await self.call("/signals/velocity", {"observations": velocity_rows},
                                      WireSignal, analysis.model, series_tickers)
            outliers = await self.call("/signals/outliers", {"observations": observations},
                                      WireSignal, analysis.model, series_tickers)
        else:
            velocity = unavailable("no_timestamped_ticker_aggregates")
            outliers = unavailable("no_timestamped_ticker_aggregates")

        # Current aggregates generally do not land on UTC bucket boundaries.
        # Never round them backward: that would invent earlier availability.
        aligned = [row for row in observations if bucket_aligned(
            datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00")), options.time_bucket)]
        if not options.prices:
            correlation = granger = unavailable("prices_not_provided")
        elif not aligned:
            correlation = granger = unavailable("no_bucket_aligned_sentiment_history")
        else:
            market_body = {"sentiments": aligned, "prices": [row.model_dump(mode="json") for row in options.prices],
                           "time_bucket": options.time_bucket}
            correlation = await self.call("/signals/correlation", market_body, WireSignal,
                                          analysis.model, series_tickers)
            if len(series_tickers) > 32:
                granger = unavailable("too_many_series")
            else:
                granger = await self.call("/signals/granger", market_body, WireSignal,
                                          analysis.model, series_tickers)
        return aggregation, weighted, SignalReports(velocity=velocity, outliers=outliers,
                                                     correlation=correlation, granger=granger)
