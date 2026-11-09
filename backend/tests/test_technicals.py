"""Tests for the technical signals: ATR, levels, relative strength, liquidity, RVOL, sessions."""

import math
from datetime import datetime

import numpy as np
import pandas as pd

from predictor import data, features, scoring, sentiment
from predictor.data import NY, Premarket

from test_pipeline import make_daily


def test_atr_simple():
    df = pd.DataFrame({"High": [11.0] * 20, "Low": [9.0] * 20, "Close": [10.0] * 20})
    assert abs(features.atr(df) - 2.0) < 1e-9


def test_levels_breakout_and_ceiling():
    daily = make_daily(100, 0.0, seed=1)
    prev = float(daily["Close"].iloc[-1])
    a = features.atr(daily)
    # Above yesterday's high and above the 20-day high -> breakout (+1)
    hi20 = float(daily["High"].iloc[-20:].max())
    f = features.ticker_features(daily, Premarket(hi20 * 1.01, 1e5, "x"))
    assert f["level_signal"] == 1.0
    # Just under the 20-day high (and not above yesterday's high) -> ceiling (-1)
    daily2 = daily.copy()
    daily2.iloc[-5, daily2.columns.get_loc("High")] = prev + 0.6 * a + 0.2 * a  # recent high a bit above
    p = prev + 0.6 * a
    if p <= float(daily2["High"].iloc[-1]):
        f2 = features.ticker_features(daily2, Premarket(p, 1e5, "x"))
        assert f2["level_signal"] == -1.0


def test_relative_strength_and_overextension():
    daily = make_daily(100, 0.0, seed=2)
    prev = float(daily["Close"].iloc[-1])
    f = features.ticker_features(daily, Premarket(prev * 1.03, 1e5, "x"), ref_gap=1.0)
    assert abs(f["rel_strength_pct"] - 2.0) < 1e-9
    assert f["gap_atr"] > 0
    assert f["overextension"] == max(0.0, f["gap_atr"] - 1.5)


def test_rvol_same_time_window():
    idx = pd.DatetimeIndex([
        "2026-11-05 08:00", "2026-11-05 09:00",           # prior day: 100 + 100 before 9:15
        "2026-11-06 08:00", "2026-11-06 09:00",           # prior day: 100 + 100
        "2026-11-09 08:00", "2026-11-09 09:00",           # today: 300 + 300
    ]).tz_localize(NY)
    bars = pd.DataFrame({"Close": [1, 1, 1, 1, 2, 2.2], "High": [1, 1, 1, 1, 2.1, 2.3],
                         "Low": [1, 1, 1, 1, 1.9, 2.0], "Volume": [100, 100, 100, 100, 300, 300]}, index=idx)
    pre = data.extract_extended({"X": bars}, datetime(2026, 11, 9, 9, 15, tzinfo=NY))["X"]
    assert pre.volume == 600 and abs(pre.rvol - 3.0) < 1e-9
    assert pre.high == 2.3 and pre.low == 1.9 and pre.session == "premarket"


def test_after_hours_session_and_daily_bar():
    idx = pd.DatetimeIndex(["2026-11-09 15:55", "2026-11-09 16:30", "2026-11-09 17:00"]).tz_localize(NY)
    bars = pd.DataFrame({"Close": [10.0, 10.5, 10.6], "Volume": [5, 7, 9]}, index=idx)
    now = datetime(2026, 11, 9, 17, 30, tzinfo=NY)
    pre = data.extract_extended({"X": bars}, now)["X"]
    assert pre.session == "after_hours" and pre.last_price == 10.6 and pre.volume == 16
    daily = {"X": make_daily(100, 0, n=5).set_axis(pd.bdate_range(end="2026-11-09", periods=5))}
    assert data.drop_today_bar(daily, now)["X"].index[-1].date().isoformat() == "2026-11-09"   # today kept
    noon = datetime(2026, 11, 9, 12, 0, tzinfo=NY)
    assert data.drop_today_bar(daily, noon)["X"].index[-1].date().isoformat() == "2026-11-06"  # incomplete dropped


def _universe(n=20, seed=5):
    rng = np.random.default_rng(seed)
    daily = {f"T{i}": make_daily(100, 0.0, seed=i) for i in range(n)}
    pre = {t: Premarket(float(df["Close"].iloc[-1]) * (1 + rng.normal(0.005, 0.005)), 50_000, "x", rvol=1.0)
           for t, df in daily.items()}
    return daily, pre


def test_thin_trading_penalised_and_explained():
    daily, pre = _universe()
    p = pre["T3"]
    pre["T3"] = Premarket(p.last_price, 100, "x", rvol=1.0)   # ~ $10k traded: very thin
    sc = scoring.score(features.build_features(daily, pre))
    base = scoring.score(features.build_features(daily, {**pre, "T3": Premarket(p.last_price, 50_000, "x", rvol=1.0)}))
    assert sc.loc["T3", "raw_score"] < base.loc["T3", "raw_score"]
    assert any("Very little early trading" in r["text"] for r in sc.loc["T3", "reasons"]) or \
           "thin early trading" in sc.loc["T3", "why_short"]


def test_unexplained_jump_flagged_only_when_news_was_checked():
    daily, pre = _universe()
    prev = float(daily["T4"]["Close"].iloc[-1])
    pre["T4"] = Premarket(prev * 1.03, 50_000, "x", rvol=1.0)
    feats = features.build_features(daily, pre)
    unchecked = scoring.score(feats)
    checked = scoring.score(feats, news={}, news_checked={"T4"})
    assert checked.loc["T4", "raw_score"] < unchecked.loc["T4", "raw_score"]
    explained = scoring.score(feats, news={"T4": sentiment.NewsAssessment(0.0, "low", "earnings", "Results out", [])},
                              news_checked={"T4"})
    assert explained.loc["T4", "raw_score"] == unchecked.loc["T4", "raw_score"]


def test_market_context_and_reference_gaps():
    from predictor import market
    gaps = {"SPY": 0.5, "QQQ": 0.7, "IWM": -0.1, "XLK": 1.2}
    ctx = market.context(gaps, {})
    assert ctx["bias"] == "positive" and ctx["indexes"][0]["ticker"] == "SPY"
    assert ctx["sectors"]["Information Technology"] == 1.2
    refs = market.reference_gaps(["AAPL", "XOM"], {"AAPL": "Information Technology", "XOM": "Energy"}, gaps)
    assert refs == {"AAPL": 1.2, "XOM": 0.5}      # XOM falls back to the S&P 500


def test_full_payload_has_new_fields():
    from predictor import report
    daily, pre = _universe()
    sc = scoring.score(features.build_features(daily, pre, {t: 0.2 for t in daily}))
    p = report.build_payload(sc, datetime(2026, 11, 10, 8, 30, tzinfo=NY), top_n=3,
                             market={"bias": "mixed"})
    pick = p["picks"][0]
    assert p["market"]["bias"] == "mixed" and p["run_window"] == "premarket"
    assert pick["levels"]["prev_high"] is not None and pick["atr"] is not None
    assert pick["rel_strength_pct"] is not None
    import json; json.dumps(p)   # must be valid JSON (no NaN objects)


def test_parse_members_handles_symbol_or_ticker_columns():
    from predictor import universe
    a = pd.DataFrame({"Symbol": ["BRK.B", "AAPL"] * 30, "Security": ["Berkshire", "Apple"] * 30, "GICS Sector": ["Financials", "Information Technology"] * 30})
    b = pd.DataFrame({"Ticker": ["NVDA", "ASML"] * 30, "Company": ["Nvidia", "ASML Holding"] * 30})
    small = pd.DataFrame({"Symbol": ["X"], "Security": ["x"]})
    pa = universe.parse_members([small, a])
    assert "BRK-B" in set(pa["ticker"]) and pa.iloc[0]["sector"] == "Financials"
    pb = universe.parse_members([b])
    assert set(pb["ticker"]) == {"NVDA", "ASML"} and pb["sector"].isna().all()


def test_liquidity_filter():
    cheap = make_daily(2.0, 0.0, seed=1)                 # $2 stock -> dropped
    thin = make_daily(100, 0.0, seed=2); thin["Volume"] = 1_000.0   # $100k/day -> dropped
    ok = make_daily(100, 0.0, seed=3)                     # ~$150M/day -> kept
    kept = data.liquid({"CHEAP": cheap, "THIN": thin, "OK": ok})
    assert set(kept) == {"OK"}


def test_price_history_is_90_calendar_days():
    from predictor import report
    df = make_daily(100, 0.001, n=150)
    h = report.price_history(df)
    days = (pd.Timestamp(h["end"]) - pd.Timestamp(h["start"])).days
    assert 85 <= days <= 90 and 55 <= len(h["closes"]) <= 66
    assert h["closes"][-1] == round(float(df["Close"].iloc[-1]), 2)


def test_price_history_includes_moving_averages():
    from predictor import report
    df = make_daily(100, 0.001, n=300)
    h = report.price_history(df)
    assert len(h["ma50"]) == len(h["closes"]) == len(h["ma200"])
    assert None not in h["ma200"]                       # 300 days of data: MA200 defined in window
    assert abs(h["ma50"][-1] - round(float(df["Close"].iloc[-50:].mean()), 2)) < 0.011
    short = report.price_history(make_daily(100, 0.001, n=120))
    assert short["ma200"][-1] is None and short["ma50"][-1] is not None   # not enough history
