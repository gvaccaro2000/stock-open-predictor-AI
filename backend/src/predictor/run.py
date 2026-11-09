"""Entry point: runs all pipeline stages in order.

    python -m predictor.run --universe sample
    python -m predictor.run --universe sp500 --top 25
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from . import data, features, report, scoring, universe

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "docs" / "data"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="sp500", help="sample | sp500 | AAPL,MSFT,...")
    p.add_argument("--top", type=int, default=25, help="how many picks to publish")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("predictor")
    now = datetime.now(data.NY)

    tickers = universe.get_universe(args.universe)
    log.info("Universe: %d tickers", len(tickers))

    daily = data.fetch_daily(tickers, now=now)
    pre = data.fetch_premarket(tickers, now=now)
    log.info("Got daily data for %d, premarket data for %d", len(daily), len(pre))
    if not daily:
        raise SystemExit("No price data downloaded — aborting so we don't publish an empty file")

    feats = features.build_features(daily, pre)
    scored = scoring.score(feats)
    payload = report.build_payload(scored, generated_at=now, top_n=args.top)
    path = report.write(payload, args.out)

    log.info("Wrote %s", path)
    for pick in payload["picks"][:10]:
        log.info("  #%-2d %-6s score %3d  %s", pick["rank"], pick["ticker"], pick["score"],
                 "; ".join(r["text"] for r in pick["reasons"]))


if __name__ == "__main__":
    main()
