"""Tests use synthetic prices so they run offline and are deterministic."""

from datetime import datetime

import numpy as np
import pandas as pd

from predictor import data, features, report, scoring
from predictor.data import NY, Premarket


def make_daily(start: float, drift: float, n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + drift + rng.normal(0, 0.01, n))
    idx = pd.bdate_range(end="2026-11-06", periods=n)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
         "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float)},
        index=idx,
    )


def test_rsi_bounds():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert features.rsi(up) == 100.0
    assert 0 <= features.rsi(pd.Series(np.random.default_rng(1).random(40))) <= 100


def test_premarket_gap_feature():
    daily = make_daily(100, 0.0)
    prev = float(daily["Close"].iloc[-1])
    f = features.ticker_features(daily, Premarket(prev * 1.05, 50_000, "x"))
    assert abs(f["premarket_gap_pct"] - 5.0) < 1e-9


def test_extract_premarket_filters_to_today_before_open():
    idx = pd.DatetimeIndex([
        "2026-11-06 08:00", "2026-11-09 07:00", "2026-11-09 09:25", "2026-11-09 10:00",
    ]).tz_localize(NY)
    bars = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0], "Volume": [10, 20, 30, 40]}, index=idx)
    pre = data.extract_premarket({"X": bars}, datetime(2026, 11, 9).date())["X"]
    assert pre.last_price == 3.0 and pre.volume == 50


def test_drop_today_bar_before_open():
    daily = {"X": make_daily(100, 0, n=5).set_axis(pd.bdate_range(end="2026-11-09", periods=5))}
    out = data.drop_today_bar(daily, datetime(2026, 11, 9, 8, 0, tzinfo=NY))
    assert out["X"].index[-1].date().isoformat() == "2026-11-06"


def test_strong_gapper_ranks_first(tmp_path):
    daily = {f"T{i}": make_daily(100, 0.0, seed=i) for i in range(20)}
    pre = {t: Premarket(float(df["Close"].iloc[-1]), 10_000, "x") for t, df in daily.items()}
    # T7 gaps up 6% on heavy premarket volume
    pre["T7"] = Premarket(float(daily["T7"]["Close"].iloc[-1]) * 1.06, 400_000, "x")

    scored = scoring.score(features.build_features(daily, pre))
    assert scored.index[0] == "T7"
    assert scored.iloc[0]["reasons"][0]["impact"] == "positive"

    payload = report.build_payload(scored, datetime(2026, 11, 9, 8, 30, tzinfo=NY), top_n=5)
    assert payload["picks"][0]["ticker"] == "T7" and len(payload["picks"]) == 5
    assert (report.write(payload, tmp_path)).exists()
    assert (tmp_path / "history" / "2026-11-09.json").exists()


def test_works_without_premarket_data():
    daily = {f"T{i}": make_daily(100, 0.001 * i, seed=i) for i in range(10)}
    scored = scoring.score(features.build_features(daily, {}))
    assert len(scored) == 10 and scored["score"].between(0, 100).all()


def test_run_window():
    assert report.run_window(datetime(2026, 11, 9, 8, 30, tzinfo=NY)) == "premarket"
    assert report.run_window(datetime(2026, 11, 9, 15, 50, tzinfo=NY)) == "market_hours"
    assert report.run_window(datetime(2026, 11, 8, 8, 30, tzinfo=NY)) == "weekend"
