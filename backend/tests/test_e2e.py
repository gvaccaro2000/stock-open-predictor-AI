"""End to end, offline: train on simulated history, then run the morning pipeline."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from predictor import calendars, data, news as newsmod, run, train, universe
from predictor.data import NY, Premarket

from test_model import synthetic_market


def _patch(monkeypatch, daily_all, pre_moves, now):
    def fake_daily(tickers, period="18mo", now=None):
        return {t: daily_all[t] for t in tickers if t in daily_all}

    def fake_pre(tickers, now=None):
        out = {}
        for t in tickers:
            if t not in daily_all:
                continue
            prev = float(daily_all[t]["Close"].iloc[-1])
            g = pre_moves.get(t, 0.3)
            p = prev * (1 + g / 100)
            out[t] = Premarket(p, 200_000, "x", p * 1.002, p * 0.995, 1.5, "premarket",
                               peak=prev * (1 + pre_moves.get(t + "_peak", g) / 100))
        return out

    monkeypatch.setattr(data, "fetch_daily", fake_daily)
    monkeypatch.setattr(data, "fetch_premarket", fake_pre)
    monkeypatch.setattr(data, "fetch_indexes", lambda now=None: {})
    monkeypatch.setattr(newsmod, "collect", lambda tickers, since, fetcher=None: {})
    monkeypatch.setattr(train, "fetch_vix", lambda period: None)
    monkeypatch.setattr(calendars, "fetch_earnings", lambda tickers, workers=6: {})
    monkeypatch.setattr(calendars, "ex_dividends_today", lambda tickers, day, workers=8: {})
    monkeypatch.setattr(run, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: now),
                                                         "combine": staticmethod(datetime.combine)}))


def test_train_then_predict(tmp_path, monkeypatch):
    stocks = synthetic_market(n_stocks=50, n_days=330)
    etfs = synthetic_market(n_stocks=1, n_days=330, seed=9)["S0"]
    daily_all = {**stocks, "SPY": etfs}
    tickers = list(stocks)
    monkeypatch.setattr(universe, "get_universe", lambda name: tickers)
    now = datetime(2024, 4, 9, 9, 0, tzinfo=NY)   # day after the synthetic history ends
    # S1 jumps 6% but has already faded from a 12% overnight peak (the Sanmina case).
    _patch(monkeypatch, daily_all, {"S1": 6.0, "S1_peak": 12.0, "S2": -1.0}, now)

    train.main(["--universe", "x", "--out", str(tmp_path)])
    assert (tmp_path / "model.json").exists() and (tmp_path / "backtest.json").exists()
    bt = json.loads((tmp_path / "backtest.json").read_text())
    assert bt["overall"]["days"] > 20

    run.main(["--universe", "x", "--top", "10", "--out", str(tmp_path), "--news-candidates", "0"])
    p = json.loads((tmp_path / "today.json").read_text())
    assert p["model"].split("-")[0] in ("ridge", "lgbm") and p["model_info"]["backtest"]["overall"]["days"] > 20
    assert set(p["model_info"]["backtest"]["models"]) == {"ridge", "lgbm"}     # both kinds compared
    exp = [x["expected_after_open_pct"] for x in p["picks"]]
    assert all(isinstance(e, float) for e in exp)
    if p["model_info"]["strategy"] == "raw":                       # other strategies rank risk-adjusted
        assert exp == sorted(exp, reverse=True)
    assert all(isinstance(x["why_short"], str) and x["why_short"] for x in p["picks"])
    assert (tmp_path / "history").exists()                         # pre-market run -> saved for the track record
    logs = list((tmp_path / "features").glob("*.csv.gz"))                # live data logged for future tests
    assert len(logs) == 1
    import pandas as pd
    lg = pd.read_csv(logs[0], index_col=0)
    assert {"pm_trend_30m", "pm_range_pos", "pm_vwap_dev", "news_score", "expected_after_open_pct"} <= set(lg.columns)
    assert len(lg) == 50

    # The faded jumper gets the fading penalty and a clear explanation.
    import predictor.predict as pr
    from predictor import model as M
    mdl = M.load(tmp_path / "model.json")
    pre = data.fetch_premarket(tickers + ["SPY"], now=now)
    live = pr.live_features({t: daily_all[t] for t in tickers}, {t: pre[t] for t in tickers},
                            {"SPY": daily_all["SPY"]}, {"SPY": pre["SPY"]}, {}, now)
    sc = pr.score(mdl, live, pre)
    s1 = sc.loc["S1"]
    assert any(a["kind"] == "fade" for a in s1["adjustments"])
    assert "overnight peak" in " ".join(r["text"] for r in s1["reasons"])


def test_evening_run_does_not_overwrite_history(tmp_path, monkeypatch):
    stocks = synthetic_market(n_stocks=40, n_days=330)
    daily_all = {**stocks, "SPY": synthetic_market(n_stocks=1, n_days=330, seed=3)["S0"]}
    monkeypatch.setattr(universe, "get_universe", lambda name: list(stocks))
    _patch(monkeypatch, daily_all, {}, datetime(2024, 4, 8, 18, 0, tzinfo=NY))
    run.main(["--universe", "x", "--out", str(tmp_path), "--news-candidates", "0"])
    assert (tmp_path / "today.json").exists() and not (tmp_path / "history").exists()
