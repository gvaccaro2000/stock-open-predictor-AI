"""Track record: how did each morning's picks actually do after the open?

    python -m predictor.evaluate          # evaluate every morning file not yet evaluated

For each trading day with a saved morning file (docs/data/history/YYYY-MM-DD.json):
  * open and close of every pick that day
  * open -> close %        what you'd get buying at the 15:30 CET open, selling at the close
  * pre-market -> close %  buying at the pre-market price the app showed (e.g. on Trade Republic)
  * the same for the S&P 500 (SPY) as a benchmark

Writes docs/data/track/YYYY-MM-DD.json and docs/data/track_summary.json (shown in the app).
Nothing is cherry-picked: every published morning is evaluated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, time
from pathlib import Path

import numpy as np

from . import data
from .model import ROUND_TRIP_COST

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "docs" / "data"
TOP_N = 10

log = logging.getLogger("predictor.evaluate")


def _pct(a, b):
    return None if a is None or b is None or not b or math.isnan(a) or math.isnan(b) else round((a / b - 1) * 100, 3)


def _bar(df, day):
    if df is None or df.empty:
        return None
    rows = df[[d.date() == day for d in df.index]]
    if rows.empty or rows[["Open", "Close"]].isna().any(axis=None):
        return None
    r = rows.iloc[-1]
    return float(r["Open"]), float(r["Close"])


def evaluate_day(morning: dict, daily: dict) -> dict | None:
    day = datetime.fromisoformat(morning["trading_date"]).date()
    spy = _bar(daily.get("SPY"), day)
    picks = []
    for p in morning.get("picks", []):
        b = _bar(daily.get(p["ticker"]), day)
        if b is None:
            continue
        o, c = b
        entry = p.get("premarket_price")
        picks.append({
            "rank": p["rank"], "ticker": p["ticker"], "name": p.get("name"),
            "expected_pct": p.get("expected_after_open_pct"),
            "premarket_gap_pct": p.get("premarket_gap_pct"),
            "premarket_price": entry, "open": round(o, 2), "close": round(c, 2),
            "open_to_close_pct": _pct(c, o), "entry_to_close_pct": _pct(c, entry) if entry else None,
        })
    if not picks or spy is None:
        return None

    def block(rows):
        oc = [r["open_to_close_pct"] for r in rows if r["open_to_close_pct"] is not None]
        ec = [r["entry_to_close_pct"] for r in rows if r["entry_to_close_pct"] is not None]
        return {"n": len(rows),
                "avg_open_to_close_pct": round(float(np.mean(oc)), 3) if oc else None,
                "avg_entry_to_close_pct": round(float(np.mean(ec)), 3) if ec else None,
                "hit_rate": round(float(np.mean([x > 0 for x in oc])), 3) if oc else None}

    top = sorted(picks, key=lambda r: r["rank"])
    return {"date": day.isoformat(), "model": morning.get("model"),
            "spy_open_to_close_pct": _pct(spy[1], spy[0]),
            "top10": block(top[:TOP_N]), "all_picks": block(top), "picks": top}


def summarise(days: list[dict]) -> dict:
    rows = [d for d in days if d["top10"]["avg_open_to_close_pct"] is not None]
    if not rows:
        return {"days": 0}
    top = np.array([d["top10"]["avg_open_to_close_pct"] for d in rows])
    spy = np.array([d["spy_open_to_close_pct"] or 0.0 for d in rows])
    entry = [d["top10"]["avg_entry_to_close_pct"] for d in rows if d["top10"]["avg_entry_to_close_pct"] is not None]
    hits = [d["top10"]["hit_rate"] for d in rows if d["top10"]["hit_rate"] is not None]
    net = top - ROUND_TRIP_COST
    return {
        "days": len(rows), "start": rows[0]["date"], "end": rows[-1]["date"], "top_n": TOP_N,
        "cost_round_trip_pct": ROUND_TRIP_COST,
        "top10_avg_open_to_close_pct": round(float(top.mean()), 3),
        "top10_avg_entry_to_close_pct": round(float(np.mean(entry)), 3) if entry else None,
        "spy_avg_open_to_close_pct": round(float(spy.mean()), 3),
        "top10_hit_rate": round(float(np.mean(hits)), 3) if hits else None,
        "days_beating_spy": round(float((top > spy).mean()), 3),
        "top10_growth_of_100_after_costs": round(float(100 * np.prod(1 + net / 100)), 2),
        "spy_growth_of_100": round(float(100 * np.prod(1 + spy / 100)), 2),
        "recent": [{"date": d["date"], "top10": d["top10"]["avg_open_to_close_pct"],
                    "spy": d["spy_open_to_close_pct"], "hit_rate": d["top10"]["hit_rate"],
                    "model": d.get("model")} for d in rows[-30:]][::-1],
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--redo", action="store_true", help="re-evaluate days already done")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    now = datetime.now(data.NY)

    hist, track = args.out / "history", args.out / "track"
    track.mkdir(parents=True, exist_ok=True)
    todo = []
    for f in sorted(hist.glob("*.json")) if hist.exists() else []:
        day = datetime.fromisoformat(f.stem).date()
        closed = day < now.date() or (day == now.date() and now.time() >= time(16, 5))
        if closed and (args.redo or not (track / f.name).exists()):
            todo.append(json.loads(f.read_text()))
    if todo:
        tickers = sorted({p["ticker"] for m in todo for p in m.get("picks", [])} | {"SPY"})
        daily = data.fetch_daily(tickers, period="3mo", now=now)
        for m in todo:
            res = evaluate_day(m, daily)
            if res is None:
                log.warning("No prices yet for %s", m["trading_date"])
                continue
            (track / f"{res['date']}.json").write_text(json.dumps(res, indent=1))
            t = res["top10"]
            log.info("%s: top-10 open->close %+.2f%% (hit %.0f%%) vs S&P 500 %+.2f%%", res["date"],
                     t["avg_open_to_close_pct"] or 0, 100 * (t["hit_rate"] or 0), res["spy_open_to_close_pct"] or 0)
    days = [json.loads(f.read_text()) for f in sorted(track.glob("*.json"))]
    summary = summarise(days)
    summary["updated_at"] = now.isoformat(timespec="minutes")
    if days:
        summary["latest"] = days[-1]
    (args.out / "track_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("Track record: %d days", summary.get("days", 0))


if __name__ == "__main__":
    main()
