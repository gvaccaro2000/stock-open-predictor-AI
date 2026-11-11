import json
from datetime import datetime

import pandas as pd

from predictor import data, evaluate
from predictor.data import NY


def test_evaluate_and_summarise(tmp_path, monkeypatch):
    hist = tmp_path / "history"; hist.mkdir()
    morning = {"trading_date": "2026-11-10", "model": "ridge-v1",
               "picks": [{"rank": i + 1, "ticker": t, "name": t, "premarket_price": 100.0,
                          "expected_after_open_pct": 0.5, "premarket_gap_pct": 2.0}
                         for i, t in enumerate(["AAA", "BBB", "CCC"])],
               "high_risk_picks": [{"rank": 1, "ticker": "CCC", "name": "CCC", "premarket_price": 100.0}]}
    (hist / "2026-11-10.json").write_text(json.dumps(morning))
    idx = pd.to_datetime(["2026-11-09", "2026-11-10"])
    bars = {"AAA": (101, 103), "BBB": (102, 100), "CCC": (100, 101), "SPY": (500, 505)}
    daily = {t: pd.DataFrame({"Open": [1, o], "High": [1, max(o, c)], "Low": [1, min(o, c)],
                              "Close": [1, c], "Volume": [1, 1]}, index=idx) for t, (o, c) in bars.items()}
    monkeypatch.setattr(data, "fetch_daily", lambda tickers, period=None, now=None: daily)
    monkeypatch.setattr(evaluate, "datetime", type("D", (), {
        "now": staticmethod(lambda tz=None: datetime(2026, 11, 10, 17, 0, tzinfo=NY)),
        "fromisoformat": staticmethod(datetime.fromisoformat)}))
    evaluate.main(["--out", str(tmp_path)])
    day = json.loads((tmp_path / "track" / "2026-11-10.json").read_text())
    aaa = next(p for p in day["picks"] if p["ticker"] == "AAA")
    assert abs(aaa["open_to_close_pct"] - (103 / 101 - 1) * 100) < 1e-3
    assert abs(aaa["entry_to_close_pct"] - 3.0) < 1e-6
    assert day["top10"]["hit_rate"] == round(2 / 3, 3)
    assert abs(day["spy_open_to_close_pct"] - 1.0) < 1e-9
    s = json.loads((tmp_path / "track_summary.json").read_text())
    assert s["days"] == 1 and s["recent"][0]["date"] == "2026-11-10"
    assert day["high_risk"]["n"] == 1 and s["high_risk"]["days"] == 1
    assert abs(s["high_risk"]["avg_open_to_close_pct"] - 1.0) < 1e-9


def test_not_evaluated_before_close(tmp_path, monkeypatch):
    hist = tmp_path / "history"; hist.mkdir()
    (hist / "2026-11-10.json").write_text(json.dumps({"trading_date": "2026-11-10", "picks": []}))
    monkeypatch.setattr(evaluate, "datetime", type("D", (), {
        "now": staticmethod(lambda tz=None: datetime(2026, 11, 10, 12, 0, tzinfo=NY)),
        "fromisoformat": staticmethod(datetime.fromisoformat)}))
    evaluate.main(["--out", str(tmp_path)])
    assert not list((tmp_path / "track").glob("*.json"))
