"""Entry point: runs all pipeline stages in order.

    python -m predictor.run --universe sample
    python -m predictor.run --universe sp500 --top 25
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from . import calendars, data, features, market, news, predict, report, scoring, sentiment, universe
from . import model as model_mod

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
    scored = scoring.score(feats)  # rule-based pass: levels, ATR etc. for the app (and fallback ranking)

    high_risk = None
    # The after-open model (trained by predictor.train), if available.
    mdl = model_mod.load(args.out / "model.json")
    live = None
    if mdl is not None:
        etf_daily = {e: daily_all[e] for e in etfs if e in daily_all}
        etf_pre = {e: pre_all[e] for e in etfs if e in pre_all}
        sector_etf = {t: data.SECTOR_ETFS.get(universe.sector(t) or "") for t in daily}
        vix = indexes.get("^VIX", pd.DataFrame()).get("Close") if indexes else None
        cal = calendars.load(args.out / "earnings_calendar.json", {})
        macro = calendars.load(args.out / "macro_days.json", [])
        sector_map = {t: e for t, e in sector_etf.items() if e}

        def features_now():
            return predict.live_features(daily, pre, etf_daily, etf_pre, sector_map, now,
                                         vix=vix, earnings_cal=cal, macro_days=macro)

        live = features_now()
        first = predict.score(mdl, live, pre)
        # Ex-dividend correction for the top candidates (a mechanical drop isn't a real dip).
        session = predict.next_session_date(now, max(pd.Timestamp(d.index[-1]).tz_localize(None) for d in daily.values()))
        try:
            divs = calendars.ex_dividends_today(list(first.index[:150]), session.date())
        except Exception as exc:
            log.warning("Ex-dividend check failed: %s", exc)
            divs = {}
        if divs:
            log.info("Ex-dividend today, adjusting: %s", ", ".join(f"{t} ${d:.2f}" for t, d in divs.items()))
            for t, d in divs.items():
                daily[t] = calendars.adjust_for_dividend(daily[t], d)
            live = features_now()
            first = predict.score(mdl, live, pre)
        live["is_ex_dividend"] = live.index.isin(list(divs)).astype(float)
        candidates = list(first.index[: args.news_candidates])
        log.info("Model %s/%s (trained through %s) scored %d stocks; earnings today: %d",
                 mdl.kind, mdl.meta.get("strategy"), mdl.meta.get("trained_through"), len(first),
                 int(live["is_earnings"].fillna(0).sum()))
    else:
        log.warning("No model.json yet — using the rule-based ranking (run predictor.train)")
        candidates = list(scored.index[: args.news_candidates])

    # Read the overnight news for the most promising stocks.
    assessments: dict = {}
    if candidates:
        since = last_close_time(daily)
        found = news.collect(candidates, since)
        names = {t: universe.company_name(t) for t in candidates}
        assessments = sentiment.analyze(found, names)

    if mdl is not None:
        ranked = predict.score(mdl, live, pre, news=assessments, news_checked=set(candidates))
        base = scored.drop(columns=["reasons", "summary", "why_short"], errors="ignore")
        scored = ranked.join(base, how="inner").sort_values(["rank_score", "expected_after_open_pct"], ascending=False)
        # The most volatile stocks are left out of the main list (they made results worse on
        # average in the backtest) but get their own "high-risk" list, ranked by expected move.
        main = scored[~scored["excluded_volatile"]]
        shown = set(main.index[: args.top])
        high_risk = scored[scored["is_volatile"] & ~scored.index.isin(list(shown))
                           & (scored["expected_after_open_pct"] > 0)].sort_values("expected_after_open_pct", ascending=False)
        scored = main
    elif assessments:
        scored = scoring.score(feats, news=assessments, news_checked=set(candidates))

    payload = report.build_payload(scored, generated_at=now, top_n=args.top, market=ctx, daily=daily,
                                   news=assessments, news_checked=len(candidates) if assessments else 0,
                                   model_info=model_info(mdl, args.out),
                                   high_risk=high_risk if mdl is not None else None)
    is_morning = data.session_for(now) == "premarket"
    path = report.write(payload, args.out, save_history=is_morning)
    if is_morning and live is not None:
        log_path = write_feature_log(args.out, payload["trading_date"], live, pre, assessments, scored)
        log.info("Logged %d stocks' live data to %s", len(live), log_path.name)

    log.info("Wrote %s", path)
    for pick in payload["picks"][:10]:
        log.info("  #%-2d %-6s exp %s  pre %s  %s", pick["rank"], pick["ticker"],
                 pick.get("expected_after_open_pct"), pick.get("premarket_gap_pct"), pick.get("why_short"))


def write_feature_log(out: Path, day: str, live, pre: dict, assessments: dict, scored) -> Path:
    """Everything known this morning, for all stocks, so live-only factors (pre-market
    shape, news, fading) can be backtested once a few months have accumulated."""
    rows = live.copy()
    for f in ("last_price", "volume", "rvol", "peak", "high", "low", "trend_30m", "range_pos", "vwap_dev"):
        rows[f"pm_{f}"] = [getattr(pre.get(t), f, None) if pre.get(t) else None for t in rows.index]
    rows["news_score"] = [assessments[t].score if t in assessments else None for t in rows.index]
    rows["news_event"] = [assessments[t].event if t in assessments else None for t in rows.index]
    for c in ("model_pct", "adjust_pct", "expected_after_open_pct"):
        if c in scored:
            rows[c] = scored[c].reindex(rows.index)
    path = out / "features" / f"{day}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.round(5).to_csv(path, compression="gzip")
    return path


def model_info(mdl, out: Path) -> dict | None:
    """Model + backtest summary for the app."""
    if mdl is None:
        return None
    import json
    info = {"name": f"{mdl.kind}-v2", "kind": mdl.kind, "strategy": mdl.meta.get("strategy", "raw"),
            "trained_through": mdl.meta.get("trained_through"),
            "trained_from": mdl.meta.get("trained_from"), "n_stocks": mdl.meta.get("n_stocks")}
    try:
        bt = json.loads((out / "backtest.json").read_text())
        info["backtest"] = {k: bt.get(k) for k in ("start", "end", "top_n", "cost_round_trip_pct", "overall",
                                                     "last_12_months", "strategy", "strategy_label", "strategies",
                                                     "model_kind", "models")}
    except (OSError, ValueError):
        pass
    return info


def last_close_time(daily: dict) -> datetime:
    """16:00 New York time on the last completed trading day in the data."""
    last_day = max(df.index[-1] for df in daily.values()).date()
    return datetime.combine(last_day, time(16, 0), tzinfo=data.NY)


if __name__ == "__main__":
    main()
