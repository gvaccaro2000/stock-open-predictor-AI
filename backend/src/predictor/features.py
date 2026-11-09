"""Stage 3 — features: turn raw prices into signals.

Each feature is one number per ticker; the scoring stage combines them.

| Feature                | Meaning                                                    | Why it matters at the open |
|------------------------|------------------------------------------------------------|----------------------------|
| premarket_gap_pct      | early price vs previous close, %                          | the open usually starts near it |
| premarket_rvol         | early volume vs the same time window on previous days      | a move with real participation is more reliable |
| premarket_activity     | early volume / average full-day volume                     | fallback when rvol is unknown |
| premarket_dollar_vol   | early volume x price, $                                    | liquidity: thin trading = unreliable move |
| rel_strength_pct       | gap minus its sector ETF's gap (or the S&P 500's), points  | is it moving on its own, or riding the market? |
| atr_14                 | average true range over 14 days, $                          | the stock's normal daily move |
| gap_atr                | gap in $ divided by ATR                                     | how big the move is *for this stock* |
| overextension          | how far the gap exceeds 1.5 ATR (0 if not)                  | very large moves often give some back |
| level_signal           | +1 above yesterday's high, -1 just under a 20-day high      | breakouts vs. ceilings (support/resistance) |
| momentum_5d_pct        | 5-day return, %                                             | short-term trends persist a little |
| trend_vs_ma50_pct      | price vs 50-day average, %                                  | uptrends attract buyers |
| rsi_14                 | Relative Strength Index (0-100)                             | very high = stretched (minor factor) |
| rel_volume_1d          | yesterday's volume / 20-day average                         | unusual interest |
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import Premarket

FEATURE_COLUMNS = [
    "premarket_gap_pct", "premarket_rvol", "premarket_activity", "premarket_dollar_vol",
    "rel_strength_pct", "atr_14", "gap_atr", "overextension", "level_signal",
    "momentum_5d_pct", "trend_vs_ma50_pct", "rsi_14", "rel_volume_1d",
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


def atr(daily: pd.DataFrame, period: int = 14) -> float:
    if not {"High", "Low", "Close"} <= set(daily.columns) or len(daily) < period + 1:
        return float("nan")
    prev_close = daily["Close"].shift(1)
    tr = pd.concat([daily["High"] - daily["Low"],
                    (daily["High"] - prev_close).abs(),
                    (daily["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())


def gap_pct(daily: pd.DataFrame | None, pre: Premarket | None) -> float:
    if daily is None or pre is None or daily.empty:
        return float("nan")
    prev = float(daily["Close"].dropna().iloc[-1])
    return (pre.last_price / prev - 1) * 100 if prev > 0 else float("nan")


def ticker_features(daily: pd.DataFrame, pre: Premarket | None, ref_gap: float = float("nan")) -> dict[str, float]:
    close = daily["Close"].dropna()
    volume = daily["Volume"].fillna(0)
    if len(close) < 21:
        return {}

    prev_close = float(close.iloc[-1])
    avg_vol_20 = float(volume.iloc[-21:-1].mean()) or np.nan
    ma50 = float(close.iloc[-50:].mean())
    a = atr(daily)
    high = daily["High"] if "High" in daily else daily["Close"]
    low = daily["Low"] if "Low" in daily else daily["Close"]

    f = {
        "prev_close": prev_close,
        "prev_high": float(high.iloc[-1]),
        "prev_low": float(low.iloc[-1]),
        "high_20d": float(high.iloc[-20:].max()),
        "low_20d": float(low.iloc[-20:].min()),
        "atr_14": a,
        "momentum_5d_pct": (prev_close / float(close.iloc[-6]) - 1) * 100,
        "trend_vs_ma50_pct": (prev_close / ma50 - 1) * 100,
        "rsi_14": rsi(close),
        "rel_volume_1d": float(volume.iloc[-1]) / avg_vol_20,
        "premarket_price": np.nan, "premarket_gap_pct": np.nan, "premarket_activity": np.nan,
        "premarket_rvol": np.nan, "premarket_dollar_vol": np.nan, "pm_high": np.nan, "pm_low": np.nan,
        "rel_strength_pct": np.nan, "gap_atr": np.nan, "overextension": 0.0, "level_signal": 0.0,
    }
    if pre is None or prev_close <= 0:
        return f

    p = pre.last_price
    gap = (p / prev_close - 1) * 100
    f.update({
        "premarket_price": p,
        "premarket_gap_pct": gap,
        "premarket_activity": pre.volume / avg_vol_20,
        "premarket_rvol": pre.rvol,
        "premarket_dollar_vol": pre.volume * p,
        "pm_high": pre.high, "pm_low": pre.low,
        "rel_strength_pct": gap - ref_gap if math.isfinite(ref_gap) else np.nan,
    })
    if math.isfinite(a) and a > 0:
        g_atr = (p - prev_close) / a
        f["gap_atr"] = g_atr
        f["overextension"] = max(0.0, g_atr - 1.5)
        level = 0.0
        if p > f["prev_high"]:
            level += 1.0                                    # breaking above yesterday's high
        if f["high_20d"] > p and (f["high_20d"] - p) < 0.5 * a:
            level -= 1.0                                    # just under a recent high (ceiling)
        f["level_signal"] = level
    return f


def build_features(daily: dict[str, pd.DataFrame], premarket: dict[str, Premarket],
                   ref_gaps: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per ticker. `ref_gaps` maps ticker -> its sector's (or market's) early move, %."""
    ref_gaps = ref_gaps or {}
    rows = {}
    for t, df in daily.items():
        f = ticker_features(df, premarket.get(t), ref_gaps.get(t, float("nan")))
        if f:
            rows[t] = f
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "ticker"
    return out
