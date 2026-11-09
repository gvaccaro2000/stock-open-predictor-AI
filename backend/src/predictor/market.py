"""Market context: what is the whole market doing before the open?

  * S&P 500 / Nasdaq 100 / small caps via their ETFs (SPY, QQQ, IWM), which trade pre-market
  * the 11 sector ETFs, so each stock can be compared with its own sector
  * VIX (expected volatility, "fear index") and the US 10-year Treasury yield (daily values)

The market bias is context shown in the app, and the sector/market moves feed each
stock's relative-strength signal. It is not a forecast of the market.
"""

from __future__ import annotations

import math

import pandas as pd

from . import features
from .data import MARKET_ETFS, SECTOR_ETFS, Premarket


def etf_gaps(daily: dict[str, pd.DataFrame], pre: dict[str, Premarket]) -> dict[str, float]:
    tickers = list(MARKET_ETFS) + list(SECTOR_ETFS.values())
    out = {}
    for t in tickers:
        g = features.gap_pct(daily.get(t), pre.get(t))
        if math.isfinite(g):
            out[t] = g
    return out


def reference_gaps(tickers: list[str], sectors: dict[str, str | None], gaps: dict[str, float]) -> dict[str, float]:
    """Each stock's benchmark move: its sector ETF if known, otherwise the S&P 500."""
    out = {}
    for t in tickers:
        etf = SECTOR_ETFS.get(sectors.get(t) or "")
        ref = gaps.get(etf) if etf else None
        if ref is None:
            ref = gaps.get("SPY")
        if ref is not None:
            out[t] = ref
    return out


def _last_and_change(df: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if df is None or df.empty or len(df["Close"].dropna()) < 2:
        return None, None
    c = df["Close"].dropna()
    return float(c.iloc[-1]), float(c.iloc[-1] - c.iloc[-2])


def context(gaps: dict[str, float], indexes: dict[str, pd.DataFrame], session: str = "premarket") -> dict:
    vix, vix_chg = _last_and_change(indexes.get("^VIX"))
    tnx, tnx_chg = _last_and_change(indexes.get("^TNX"))
    core = [gaps[t] for t in ("SPY", "QQQ") if t in gaps]
    avg = sum(core) / len(core) if core else None
    when = {"premarket": "before the open", "after_hours": "in after-hours trading"}.get(session, "today")
    if avg is None:
        bias, text = "unknown", "No early market data yet."
    elif avg >= 0.3:
        bias, text = "positive", f"The overall market is pointing higher {when}."
    elif avg <= -0.3:
        bias, text = "negative", f"The overall market is pointing lower {when}, so be more careful."
    else:
        bias, text = "mixed", f"The overall market is roughly flat {when}."
    if vix is not None and vix >= 25:
        text += " Uncertainty is high (VIX above 25), so big swings are more likely."
    sector_moves = {name: round(gaps[etf], 2) for name, etf in SECTOR_ETFS.items() if etf in gaps}
    return {
        "bias": bias,
        "summary": text,
        "indexes": [{"ticker": t, "name": n, "gap_pct": round(gaps[t], 2)} for t, n in MARKET_ETFS.items() if t in gaps],
        "vix": None if vix is None else round(vix, 2),
        "vix_change": None if vix_chg is None else round(vix_chg, 2),
        "us10y": None if tnx is None else round(tnx, 2),
        "us10y_change": None if tnx_chg is None else round(tnx_chg, 2),
        "sectors": dict(sorted(sector_moves.items(), key=lambda kv: -kv[1])),
    }
