"""Stage 5 — report: write the JSON file the iPhone app downloads.

Files written:
  docs/data/today.json               latest run (what the app shows)
  docs/data/history/YYYY-MM-DD.json  archive, used later to measure accuracy

The JSON format is a contract with the app; if you change it, bump SCHEMA_VERSION
and update app/src/types.ts.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from .universe import company_name, sector, size

SCHEMA_VERSION = 1
DISCLAIMER = (
    "Statistical signals for research and education only. Not financial advice. "
    "Past patterns do not guarantee future results."
)


def _num(x, digits: int = 2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, digits)


def run_window(t: datetime) -> str:
    """Which trading session the "early" price comes from."""
    if t.weekday() >= 5:
        return "weekend"
    hm = (t.hour, t.minute)
    if (4, 0) <= hm < (9, 30):
        return "premarket"
    if (9, 30) <= hm < (16, 0):
        return "market_hours"
    if (16, 0) <= hm < (20, 0):
        return "after_hours"
    return "overnight"


HISTORY_DAYS = 90   # calendar days shown in the charts


def price_history(df: pd.DataFrame | None, days: int = HISTORY_DAYS) -> dict | None:
    """Daily closes for the last `days` calendar days (~63 trading days), for the charts."""
    if df is None or df.empty:
        return None
    full = df["Close"].dropna()
    ma50 = full.rolling(50).mean()      # computed on the full history, then cut to the window,
    ma200 = full.rolling(200).mean()    # so the averages are correct from the first day shown
    keep = full.index >= full.index[-1] - pd.Timedelta(days=days)
    c = full[keep]
    if len(c) < 5:
        return None
    r = lambda s: [None if pd.isna(x) else round(float(x), 2) for x in s[keep]]
    return {"start": c.index[0].date().isoformat(), "end": c.index[-1].date().isoformat(),
            "closes": [round(float(x), 2) for x in c], "ma50": r(ma50), "ma200": r(ma200),
            "ma50_now": _num(ma50.iloc[-1]), "ma200_now": _num(ma200.iloc[-1])}


def build_payload(scored: pd.DataFrame, generated_at: datetime, top_n: int = 25,
                  news: dict | None = None, news_checked: int = 0, market: dict | None = None,
                  daily: dict[str, pd.DataFrame] | None = None, model_info: dict | None = None) -> dict:
    """`news` maps ticker -> sentiment.NewsAssessment; `news_checked` = how many stocks we read news for."""
    news = news or {}
    has_premarket = bool(scored["premarket_gap_pct"].notna().any())
    picks = _picks(scored.head(top_n), news, daily)
    universe = len(scored)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "trading_date": generated_at.date().isoformat(),
        "universe_size": int(universe),
        "has_premarket_data": has_premarket,
        "run_window": run_window(generated_at),
        "model": (model_info["name"] if model_info else "rules-v3") + ("-news" if news else ""),
        "model_info": model_info,
        "market": market,
        "news_enabled": bool(news),
        "news_checked": news_checked,
        "disclaimer": DISCLAIMER,
        "picks": picks,
    }


def _picks(rows: pd.DataFrame, news: dict, daily: dict | None) -> list[dict]:
    picks = []
    for rank, (ticker, r) in enumerate(rows.iterrows(), start=1):
        picks.append({
            "rank": rank,
            "ticker": ticker,
            "name": company_name(ticker),
            "score": int(r["score"]) if "score" in r and pd.notna(r["score"]) else None,
            "raw_score": _num(r.get("raw_score"), 3),
            "expected_after_open_pct": _num(r.get("expected_after_open_pct"), 2),
            "model_pct": _num(r.get("model_pct"), 3),
            "adjust_pct": _num(r.get("adjust_pct"), 3),
            "strength": r.get("strength") if isinstance(r.get("strength"), str) else None,
            "adjustments": r.get("adjustments") if isinstance(r.get("adjustments"), list) else [],
            "prev_close": _num(r["prev_close"]),
            "premarket_price": _num(r["premarket_price"]),
            "premarket_gap_pct": _num(r["premarket_gap_pct"]),
            "momentum_5d_pct": _num(r["momentum_5d_pct"]),
            "trend_vs_ma50_pct": _num(r["trend_vs_ma50_pct"]),
            "rsi_14": _num(r["rsi_14"], 1),
            "rel_volume_1d": _num(r["rel_volume_1d"]),
            "sector": sector(ticker),
            "size": size(ticker),
            "history": price_history((daily or {}).get(ticker)),
            "premarket_rvol": _num(r.get("premarket_rvol"), 1),
            "premarket_dollar_vol": _num(r.get("premarket_dollar_vol"), 0),
            "rel_strength_pct": _num(r.get("rel_strength_pct")),
            "atr": _num(r.get("atr_14")),
            "gap_atr": _num(r.get("gap_atr"), 2),
            "levels": {
                "prev_close": _num(r.get("prev_close")),
                "prev_high": _num(r.get("prev_high")),
                "prev_low": _num(r.get("prev_low")),
                "pm_high": _num(r.get("pm_high")),
                "pm_low": _num(r.get("pm_low")),
                "high_20d": _num(r.get("high_20d")),
                "low_20d": _num(r.get("low_20d")),
            },
            "summary": r["summary"],
            "why_short": r["why_short"],
            "reasons": r["reasons"],
            "news": news[ticker].to_json() if ticker in news else None,
        })
    return picks


def write(payload: dict, out_dir: Path, save_history: bool = True) -> Path:
    """today.json always; the dated history copy (used by the track record) only for
    pre-market runs, so a test run in the evening can't overwrite the morning's picks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    today = out_dir / "today.json"
    today.write_text(text)
    if save_history:
        (out_dir / "history").mkdir(exist_ok=True)
        (out_dir / "history" / f"{payload['trading_date']}.json").write_text(text)
    return today
