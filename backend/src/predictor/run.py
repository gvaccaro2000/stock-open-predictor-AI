"""Entry point: runs all pipeline stages in order.

    python -m predictor.run --universe sample
    python -m predictor.run --universe sp500 --top 25
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, time
from pathlib import Path

from . import data, features, market, news, report, scoring, sentiment, universe

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "docs" / "data"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="us1500",
                   help="sample | sp500 | us1500 (S&P 1500 + Nasdaq-100) | AAPL,MSFT,...")
    p.add_argument("--top", type=int, default=25, help="how many picks to publish")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--news-candidates", type=int, default=60,
                   help="read news for this many top stocks (0 = no news)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("predictor")
    now = datetime.now(data.NY)

    tickers = universe.get_universe(args.universe)
    log.info("Universe: %d tickers", len(tickers))

    etfs = list(data.MARKET_ETFS) + list(data.SECTOR_ETFS.values())
    all_t = tickers + [e for e in etfs if e not in tickers]
    daily_all = data.fetch_daily(all_t, now=now)
    daily = data.liquid({t: d for t, d in daily_all.items() if t in tickers})
    log.info("%d of %d stocks pass the price/liquidity filter", len(daily), len(tickers))
    wanted = list(daily) + [e for e in etfs if e in daily_all]
    pre_all = data.fetch_premarket(wanted, now=now)   # intraday only for stocks we keep
    pre = {t: x for t, x in pre_all.items() if t in daily}
    log.info("Got daily data for %d, extended-hours data for %d (session: %s)",
             len(daily), len(pre), data.session_for(now))
    if not daily:
        raise SystemExit("No price data downloaded — aborting so we don't publish an empty file")

    # Market context + each stock's sector benchmark for relative strength.
    gaps = market.etf_gaps(daily_all, pre_all)
    try:
        indexes = data.fetch_indexes(now)
    except Exception as exc:
        log.warning("Could not load VIX / 10-year yield: %s", exc)
        indexes = {}
    ctx = market.context(gaps, indexes, data.session_for(now))
    log.info("Market: %s", ctx["summary"])
    refs = market.reference_gaps(list(daily), {t: universe.sector(t) for t in daily}, gaps)

    feats = features.build_features(daily, pre, refs)
    scored = scoring.score(feats)  # price-only first pass

    # Phase 2: read the overnight news for the most promising stocks, then re-score.
    assessments: dict = {}
    candidates = list(scored.index[: args.news_candidates])
    if candidates:
        since = last_close_time(daily)
        found = news.collect(candidates, since)
        names = {t: universe.company_name(t) for t in candidates}
        assessments = sentiment.analyze(found, names)
        if assessments:
            scored = scoring.score(feats, news=assessments, news_checked=set(candidates))

    payload = report.build_payload(scored, generated_at=now, top_n=args.top, market=ctx, daily=daily,
                                   news=assessments, news_checked=len(candidates) if assessments else 0)
    path = report.write(payload, args.out)

    log.info("Wrote %s", path)
    for pick in payload["picks"][:10]:
        log.info("  #%-2d %-6s score %3d  %s", pick["rank"], pick["ticker"], pick["score"],
                 "; ".join(r["text"] for r in pick["reasons"]))


def last_close_time(daily: dict) -> datetime:
    """16:00 New York time on the last completed trading day in the data."""
    last_day = max(df.index[-1] for df in daily.values()).date()
    return datetime.combine(last_day, time(16, 0), tzinfo=data.NY)


if __name__ == "__main__":
    main()
