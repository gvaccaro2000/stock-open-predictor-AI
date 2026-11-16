"""Buy-the-dip scan: price tests, fundamental filters, Claude cause filter (fake), payload."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from predictor import dips as D
from predictor import longterm as LT
from predictor.data import NY


def series(path):
    idx = pd.bdate_range(end="2026-11-13", periods=len(path))
    c = pd.Series(path, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx)


def dip_path(n=300, fall_to=55.0, settle_days=25):
    up = np.linspace(80, 100, n - 60)
    down = np.linspace(100, fall_to, 60 - settle_days)
    flat = fall_to * (1 + 0.004 * np.arange(1, settle_days + 1))          # slowly recovering, no new low
    return np.concatenate([up, down, flat])


def test_price_signals_and_candidates():
    still_falling = np.concatenate([np.linspace(80, 100, 240), np.linspace(100, 50, 60)])
    small_dip = np.concatenate([np.linspace(80, 100, 260), np.linspace(100, 85, 40)])
    daily = {"DIP": series(dip_path()), "FALL": series(still_falling), "SMALL": series(small_dip)}
    sector = {"Health Care": pd.Series(100.0, index=daily["DIP"].index)}
    s = D.price_signals(daily, sector, {"DIP": "Health Care", "FALL": "Health Care", "SMALL": "Health Care"})
    assert s.loc["DIP", "drawdown"] < -30 and s.loc["DIP", "days_since_low"] >= 10
    assert s.loc["FALL", "days_since_low"] == 0
    assert abs(s.loc["DIP", "rel_drop"] - s.loc["DIP", "drawdown"]) < 1e-9     # flat sector -> all company-specific
    assert list(D.price_candidates(s).index) == ["DIP"]                          # falling knife and small dip excluded


def fake(n=12):
    rng = np.random.default_rng(1)
    info = {}
    for i in range(n):
        info[f"D{i}"] = {"longName": f"Dip {i}", "sector": "Healthcare", "industry": "x", "marketCap": 1e10,
                         "currentPrice": 50.0, "forwardPE": rng.uniform(6, 14), "trailingPE": 12, "enterpriseToEbitda": rng.uniform(5, 10),
                         "freeCashflow": 5e8, "returnOnEquity": 0.2, "operatingMargins": rng.uniform(0.1, 0.3), "grossMargins": 0.5,
                         "debtToEquity": 60, "revenueGrowth": 0.05, "earningsGrowth": 0.1, "trailingEps": 4, "forwardEps": 4.6,
                         "recommendationMean": 1.9, "recommendationKey": "buy", "targetMeanPrice": 75, "numberOfAnalystOpinions": 10,
                         "currentRatio": 1.5}
    info["D0"]["operatingMargins"] = -0.05          # loss-making -> out
    info["D1"]["debtToEquity"] = 400                # too much debt -> out
    info["D2"]["recommendationMean"] = 3.2          # analysts negative -> out
    info["D3"]["currentRatio"] = 0.5                # can't pay short-term bills -> out
    info = pd.DataFrame.from_dict(info, orient="index")
    sig = pd.DataFrame({"last": 50.0, "high_52w": 90.0, "high_date": "2026-06-01", "drawdown": -44.0, "low_60d": 45.0,
                        "days_since_low": 15, "bounce_from_low": 11.0, "vs_sma20": 2.0, "vol_fade": 0.6, "rsi14": 55.0,
                        "sector_change": 3.0, "rel_drop": -47.0}, index=info.index)
    m = LT.metrics(info, pd.Series(dtype=float))
    m["current_ratio"] = info["currentRatio"]
    return info, sig, m


def test_screen_filters_and_scores():
    info, sig, m = fake()
    r = D.screen(m, sig, {"Healthcare": {"forward_pe": 18.0, "ev_ebitda": 14.0, "op_margin": 15.0, "roe": 12.0}})
    assert not {"D0", "D1", "D2", "D3"} & set(r.index)
    assert len(r) == 8 and r["composite"].between(0, 100).all()
    assert (r["sector_median_forward_pe"] == 18.0).all()          # medians come from the full-universe file
    assert r.loc[r["forward_pe"].idxmin(), "value_score"] >= r["value_score"].median()


def test_structural_causes_are_dropped():
    info, sig, m = fake()
    r = D.screen(m, sig, {})
    first = r.index[0]

    def answer(cause):
        return {"cause_type": cause, "why_fell": "Weak quarter.", "why_recover": "Still profitable and cheap.",
                "risks": ["Guidance could fall"], "conviction": "medium", "one_liner": "Cheap after a weak quarter"}

    class Client:
        def __init__(self, cause):
            self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input=answer(cause))]))

    bad = D.analyse(Client("structural"), "m", D.facts_for(first, r.loc[first], info.loc[first].to_dict(), ["Fraud probe"]))
    good = {t: D.analyse(Client("temporary"), "m", "x") for t in r.index[1:]}
    payload = D.build(r, info, {}, {first: bad, **good}, {}, datetime(2026, 11, 13, 17, tzinfo=NY), 900, 40)
    tickers = [p["ticker"] for p in payload["picks"]]
    assert first not in tickers and len(tickers) == 7
    assert payload["excluded_lasting_damage"][0]["ticker"] == first
    assert payload["picks"][0]["analysis"]["why_recover"] and payload["picks"][0]["dip"]["drawdown_pct"] == -44.0
