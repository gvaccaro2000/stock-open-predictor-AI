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

from .universe import company_name

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
    """Signals are only meaningful when computed in the premarket session."""
    if t.weekday() >= 5:
        return "weekend"
    hm = (t.hour, t.minute)
    if (4, 0) <= hm < (9, 30):
        return "premarket"
    if (9, 30) <= hm < (16, 0):
        return "market_hours"
    return "outside_premarket"


def build_payload(scored: pd.DataFrame, generated_at: datetime, top_n: int = 25,
                  news: dict | None = None, news_checked: int = 0) -> dict:
    """`news` maps ticker -> sentiment.NewsAssessment; `news_checked` = how many stocks we read news for."""
    news = news or {}
    has_premarket = bool(scored["premarket_gap_pct"].notna().any())
    picks = []
    for rank, (ticker, r) in enumerate(scored.head(top_n).iterrows(), start=1):
        picks.append({
            "rank": rank,
            "ticker": ticker,
            "name": company_name(ticker),
            "score": int(r["score"]),
            "raw_score": _num(r["raw_score"], 3),
            "prev_close": _num(r["prev_close"]),
            "premarket_price": _num(r["premarket_price"]),
            "premarket_gap_pct": _num(r["premarket_gap_pct"]),
            "momentum_5d_pct": _num(r["momentum_5d_pct"]),
            "trend_vs_ma50_pct": _num(r["trend_vs_ma50_pct"]),
            "rsi_14": _num(r["rsi_14"], 1),
            "rel_volume_1d": _num(r["rel_volume_1d"]),
            "summary": r["summary"],
            "why_short": r["why_short"],
            "reasons": r["reasons"],
            "news": news[ticker].to_json() if ticker in news else None,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "trading_date": generated_at.date().isoformat(),
        "universe_size": int(len(scored)),
        "has_premarket_data": has_premarket,
        "run_window": run_window(generated_at),
        "model": "rules-v2-news" if news else "rules-v1",
        "news_enabled": bool(news),
        "news_checked": news_checked,
        "disclaimer": DISCLAIMER,
        "picks": picks,
    }


def write(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history").mkdir(exist_ok=True)
    text = json.dumps(payload, indent=2)
    today = out_dir / "today.json"
    today.write_text(text)
    (out_dir / "history" / f"{payload['trading_date']}.json").write_text(text)
    return today
