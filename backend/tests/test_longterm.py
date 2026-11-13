"""Long-term ideas: sector-relative scoring, filters, Claude analysis (fake client), payload."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from predictor import longterm as LT
from predictor.data import NY

from test_pipeline import make_daily


def fake_info(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = {}
    for i in range(n):
        sector = "Information Technology" if i < n // 2 else "Utilities"
        base_pe = 30 if sector == "Information Technology" else 15       # sectors have different "normal" P/Es
        rows[f"T{i}"] = {
            "longName": f"Company {i}", "sector": sector, "industry": "x", "marketCap": 5e10, "currentPrice": 100.0,
            "forwardPE": base_pe * rng.uniform(0.6, 1.4), "trailingPE": base_pe, "enterpriseToEbitda": base_pe * 0.6,
            "freeCashflow": 5e10 * rng.uniform(0.01, 0.06), "returnOnEquity": rng.uniform(0.05, 0.4),
            "operatingMargins": rng.uniform(0.05, 0.35), "grossMargins": rng.uniform(0.2, 0.7),
            "debtToEquity": rng.uniform(10, 200), "revenueGrowth": rng.uniform(-0.05, 0.3),
            "earningsGrowth": rng.uniform(-0.1, 0.4), "trailingEps": 4.0, "forwardEps": 4.0 * rng.uniform(0.9, 1.4),
            "recommendationMean": rng.uniform(1.5, 3.5), "recommendationKey": "buy", "targetMeanPrice": 100 * rng.uniform(0.9, 1.4),
            "numberOfAnalystOpinions": 12,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def test_scores_are_sector_relative_and_oriented():
    info = fake_info()
    mom = pd.Series(np.linspace(-10, 30, len(info)), index=info.index)
    sc = LT.score(LT.metrics(info, mom))
    # A utility with P/E 12 must not look "cheap" only because tech P/Es are higher:
    # value scores are ranked within each sector, so both sectors span 0-100.
    tech, util = (sc[sc["sector"] == s] for s in ("Information Technology", "Utilities"))
    assert util["earnings_yield_fwd"].median() > 1.5 * tech["earnings_yield_fwd"].median()   # utilities look cheaper in absolute terms
    assert abs(util["value_score"].median() - tech["value_score"].median()) < 15             # ...but not within their sector
    # Better fundamentals -> higher composite
    best, worst = sc.index[0], sc.index[-1]
    assert sc.loc[best, "composite"] > sc.loc[worst, "composite"]
    assert sc["composite"].between(0, 100).all()


def test_filters_small_and_loss_making():
    info = fake_info()
    info.loc["T0", "marketCap"] = 5e8                 # too small
    info.loc["T1", "forwardPE"] = -12                 # expected losses
    sc = LT.score(LT.metrics(info, pd.Series(0.0, index=info.index)))
    assert "T0" not in sc.index and "T1" not in sc.index


def test_analysis_with_fake_claude_and_payload():
    info = fake_info()
    sc = LT.score(LT.metrics(info, pd.Series(5.0, index=info.index)))
    t = sc.index[0]
    answer = {"thesis": "Profitable, growing faster than peers, and cheaper than its sector.",
              "strengths": ["Margins above sector", "Low debt"], "risks": ["Competition", "Valuation could compress"],
              "conviction": "medium", "one_liner": "Growth at a sector discount"}

    class Fake:
        def __init__(self): self.messages, self.calls = self, []
        def create(self, **kw):
            self.calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=answer)])

    fake = Fake()
    prompt = LT.facts_for(t, sc.loc[t], info.loc[t].to_dict(), {}, ["Company beats estimates"])
    assert "sector median" in prompt and "Company beats estimates" in prompt
    a = LT.analyse(fake, "test-model", prompt)
    assert a["conviction"] == "medium" and len(a["risks"]) == 2
    daily = {x: make_daily(100, 0.001, n=300, seed=i) for i, x in enumerate(info.index)}
    payload = LT.build(sc, info, daily, {}, {t: a}, {}, datetime(2026, 11, 15, 10, tzinfo=NY), top_n=5)
    p0 = payload["picks"][0]
    assert p0["ticker"] == t and p0["analysis"]["one_liner"] == "Growth at a sector discount"
    assert set(p0["scores"]) == set(LT.WEIGHTS) and p0["history"]["closes"]
    import json; json.dumps(payload)


def test_longterm_tracking(tmp_path):
    import json
    from predictor import evaluate
    h = tmp_path / "longterm_history"; h.mkdir()
    (h / "2026-11-15.json").write_text(json.dumps({"as_of_date": "2026-11-15", "picks": [
        {"rank": 1, "ticker": "AAA", "name": "A", "price": 100.0}, {"rank": 2, "ticker": "BBB", "name": "B", "price": 50.0}]}))
    idx = pd.to_datetime(["2026-11-13", "2026-11-16", "2026-12-15"])
    mk = lambda closes: pd.DataFrame({"Close": closes}, index=idx)
    daily = {"AAA": mk([100, 101, 110]), "BBB": mk([50, 49, 45]), "SPY": mk([500, 502, 510])}
    lt = evaluate.longterm_track(tmp_path, datetime(2026, 12, 15, 17, tzinfo=NY), fetch=lambda t, period, now: daily)
    L = lt["lists"][0]
    assert L["avg_return_pct"] == 0.0 and L["winners"] == 1          # +10% and -10%
    assert abs(L["spy_return_pct"] - 2.0) < 1e-9 and L["beat_spy"] is False
