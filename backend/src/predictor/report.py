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


def build_payload(scored: pd.DataFrame, generated_at: datetime, top_n: int = 25) -> dict:
    has_premarket = bool(scored["premarket_gap_pct"].notna().any())
    picks = []
    for rank, (ticker, r) in enumerate(scored.head(top_n).iterrows(), start=1):
        picks.append({
            "rank": rank,
            "ticker": ticker,
            "score": int(r["score"]),
            "prev_close": _num(r["prev_close"]),
            "premarket_price": _num(r["premarket_price"]),
            "premarket_gap_pct": _num(r["premarket_gap_pct"]),
            "momentum_5d_pct": _num(r["momentum_5d_pct"]),
            "trend_vs_ma50_pct": _num(r["trend_vs_ma50_pct"]),
            "rsi_14": _num(r["rsi_14"], 1),
            "rel_volume_1d": _num(r["rel_volume_1d"]),
            "reasons": r["reasons"],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "trading_date": generated_at.date().isoformat(),
        "universe_size": int(len(scored)),
        "has_premarket_data": has_premarket,
        "model": "rules-v1",
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
