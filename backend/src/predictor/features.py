"""Stage 3 — features: turn raw prices into signals.

Each feature is one number per ticker. The scoring stage combines them.

| Feature              | Meaning                                              | Why it may matter at the open |
|----------------------|------------------------------------------------------|-------------------------------|
| premarket_gap_pct    | premarket price vs yesterday's close, in %           | the open usually starts near the premarket price |
| premarket_activity   | premarket volume / average daily volume              | a gap on heavy volume is more "real" than on a few trades |
| momentum_5d_pct      | 5-day return, in %                                   | short-term trends tend to persist a little |
| trend_vs_ma50_pct    | price vs its 50-day moving average, in %             | stocks in an uptrend get more buyers |
| rsi_14               | Relative Strength Index (0-100)                      | very high = stretched, risk of pullback |
| rel_volume_1d        | yesterday's volume / 20-day average volume           | unusual interest often precedes moves |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Premarket

FEATURE_COLUMNS = [
    "premarket_gap_pct",
    "premarket_activity",
    "momentum_5d_pct",
    "trend_vs_ma50_pct",
    "rsi_14",
    "rel_volume_1d",
]


def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    if len(delta) < period:
        return float("nan")
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def ticker_features(daily: pd.DataFrame, pre: Premarket | None) -> dict[str, float]:
    close = daily["Close"].dropna()
    volume = daily["Volume"].fillna(0)
    if len(close) < 21:
        return {}

    prev_close = float(close.iloc[-1])
    avg_vol_20 = float(volume.iloc[-21:-1].mean()) or np.nan
    ma50 = float(close.iloc[-50:].mean())

    f = {
        "prev_close": prev_close,
        "momentum_5d_pct": (prev_close / float(close.iloc[-6]) - 1) * 100,
        "trend_vs_ma50_pct": (prev_close / ma50 - 1) * 100,
        "rsi_14": rsi(close),
        "rel_volume_1d": float(volume.iloc[-1]) / avg_vol_20,
        "premarket_price": np.nan,
        "premarket_gap_pct": np.nan,
        "premarket_activity": np.nan,
    }
    if pre is not None and prev_close > 0:
        f["premarket_price"] = pre.last_price
        f["premarket_gap_pct"] = (pre.last_price / prev_close - 1) * 100
        f["premarket_activity"] = pre.volume / avg_vol_20
    return f


def build_features(
    daily: dict[str, pd.DataFrame], premarket: dict[str, Premarket]
) -> pd.DataFrame:
    """One row per ticker, one column per feature."""
    rows = {}
    for t, df in daily.items():
        f = ticker_features(df, premarket.get(t))
        if f:
            rows[t] = f
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "ticker"
    return out
