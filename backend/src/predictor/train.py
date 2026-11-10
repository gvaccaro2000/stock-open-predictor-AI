"""Train the after-open model and backtest it on ~3 years of history.

    python -m predictor.train --universe us1500

Writes
    docs/data/model.json      model weights used by the daily run
    docs/data/backtest.json   walk-forward results shown in the app

Runs weekly on GitHub Actions (and automatically if model.json is missing).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from . import data, model, universe

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "docs" / "data"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="us1500")
    p.add_argument("--period", default="4y", help="history to download (first year warms up the averages)")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("predictor.train")
    now = datetime.now(data.NY)

    tickers = universe.get_universe(args.universe)
    etfs = list(data.MARKET_ETFS) + list(data.SECTOR_ETFS.values())
    log.info("Downloading %s of daily history for %d stocks + %d ETFs", args.period, len(tickers), len(etfs))
    daily = data.fetch_daily(tickers, period=args.period, now=now)
    etf_daily = data.fetch_daily(etfs, period=args.period, now=now)
    sector_etf = {t: data.SECTOR_ETFS.get(universe.sector(t) or "") for t in daily}

    final, bt = model.train_all(daily, etf_daily, {t: e for t, e in sector_etf.items() if e})
    final.meta["universe"] = args.universe
    model.save(final, args.out / "model.json")
    bt["generated_at"] = final.meta["trained_at"]
    bt["weights"] = dict(zip(final.features, [round(float(w), 4) for w in final.weights]))
    (args.out / "backtest.json").write_text(json.dumps(bt, indent=1))

    o = bt.get("overall", {})
    log.info("Model trained on %s rows (%s .. %s)", final.meta.get("n_rows"),
             final.meta.get("trained_from"), final.meta.get("trained_through"))
    log.info("Backtest %s days: top-10 avg %.3f%% (after costs %.3f%%) vs all stocks %.3f%%, hit rate %.1f%%, IC %.3f",
             o.get("days", 0), o.get("top_avg_pct", 0), o.get("top_avg_after_costs_pct", 0),
             o.get("all_avg_pct", 0), 100 * o.get("top_hit_rate", 0), o.get("avg_ic", 0))
    log.info("Weights: %s", bt["weights"])


if __name__ == "__main__":
    main()
