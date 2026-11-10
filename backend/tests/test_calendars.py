from datetime import datetime

import numpy as np
import pandas as pd

from predictor import calendars, model
from predictor.data import NY

from test_model import synthetic_market

DAYS = pd.bdate_range("2026-11-02", "2026-11-13")


def test_earnings_session_date():
    after_close = datetime(2026, 11, 9, 16, 5, tzinfo=NY)     # Monday after the close -> Tuesday open
    before_open = datetime(2026, 11, 10, 7, 0, tzinfo=NY)     # Tuesday before the open -> Tuesday
    unknown = datetime(2026, 11, 12, 0, 0, tzinfo=NY)         # time unknown -> that day's open
    friday_pm = datetime(2026, 11, 6, 16, 30, tzinfo=NY)      # Friday after close -> Monday
    assert calendars.earnings_session_date(after_close, DAYS) == pd.Timestamp("2026-11-10")
    assert calendars.earnings_session_date(before_open, DAYS) == pd.Timestamp("2026-11-10")
    assert calendars.earnings_session_date(unknown, DAYS) == pd.Timestamp("2026-11-12")
    assert calendars.earnings_session_date(friday_pm, DAYS) == pd.Timestamp("2026-11-09")


def test_earnings_panel():
    cal = {"AAA": [datetime(2026, 11, 9, 16, 5, tzinfo=NY).isoformat()]}
    p = calendars.earnings_panel(cal, DAYS, ["AAA", "BBB"])
    assert p.loc["2026-11-10", "AAA"] == 1 and p["AAA"].sum() == 1 and p["BBB"].sum() == 0


def test_dividend_adjustment():
    df = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0], "Volume": [1]})
    a = calendars.adjust_for_dividend(df, 1.0)
    assert abs(a["Close"].iloc[-1] - 99.0) < 1e-9 and a["Volume"].iloc[-1] == 1


def test_new_features_use_only_pre_open_info_with_context():
    daily = synthetic_market(n_stocks=4, n_days=300)
    spy = synthetic_market(n_stocks=1, n_days=300, seed=7)["S0"]
    P, E = model.to_panel(daily), model.to_panel({"SPY": spy})
    idx = P["C"].index
    ctx = {"vix": pd.Series(np.linspace(14, 30, len(idx)), index=idx),
           "earnings": pd.DataFrame(0.0, index=idx, columns=P["C"].columns),
           "macro_days": {idx[-1]}}
    ctx["earnings"].iloc[-1, 0] = 1.0
    f1 = model.panel_features(P, {}, E, ctx)
    P2 = {k: v.copy() for k, v in P.items()}
    E2 = {k: v.copy() for k, v in E.items()}
    last = idx[-1]
    for k in "HLCV":
        P2[k].loc[last] *= 1.5
        E2[k].loc[last] *= 1.5
    f2 = model.panel_features(P2, {}, E2, ctx)
    for name in model.FEATURES:
        a, b = f1[name].loc[last].fillna(0), f2[name].loc[last].fillna(0)
        assert np.allclose(a, b), name
    assert f1["beta_60"].loc[last].notna().all()
    assert f1["is_earnings"].loc[last].iloc[0] == 1 and f1["beta_x_macro"].loc[last].notna().all()


def test_live_features_flag_overnight_earnings_and_vix():
    from predictor import predict
    from predictor.data import Premarket
    daily = synthetic_market(n_stocks=5, n_days=300)
    last = daily["S0"].index[-1]                                   # last completed session
    now = datetime.combine((last + pd.offsets.BDay(1)).date(), datetime.min.time().replace(hour=9), tzinfo=NY)
    pre = {t: Premarket(float(df["Close"].iloc[-1]) * 1.02, 1e5, "x") for t, df in daily.items()}
    cal = {"S3": [datetime.combine(last.date(), datetime.min.time().replace(hour=16, minute=10), tzinfo=NY).isoformat()]}
    vix = pd.Series(25.0, index=daily["S0"].index)
    live = predict.live_features(daily, pre, {}, {}, {}, now, vix=vix, earnings_cal=cal, macro_days=[])
    assert live.loc["S3", "is_earnings"] == 1 and live.loc["S1", "is_earnings"] == 0
    assert live.loc["S3", "earn_x_gap"] != 0
    assert (live["gap_x_vix"] > 0).all()                           # VIX 25 > 20 and all gaps positive
