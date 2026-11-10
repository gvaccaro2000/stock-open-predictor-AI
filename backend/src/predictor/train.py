"""Train the after-open model and backtest it on ~3 years of history.

    python -m predictor.train --universe us1500            # full: compare models + backtest
    python -m predictor.train --universe us1500 --quick    # fast: fit only (no backtest)

Writes
    docs/data/model.json              model used by the daily run
    docs/data/backtest.json           walk-forward results shown in the app
    docs/data/earnings_calendar.json  earnings dates (past + upcoming) for the daily run
    docs/data/macro_days.json         US macro release days (if FRED_API_KEY is set)

Runs weekly on GitHub Actions (and in quick mode if model.json is missing).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import calendars, data, model, universe

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "docs" / "data"
log = logging.getLogger("predictor.train")


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return (idx.tz_localize(None) if idx.tz is not None else idx).normalize()


def build_context(daily: dict[str, pd.DataFrame], vix: pd.Series | None, cal: dict, macro: list[str]) -> dict:
    """Context panels aligned to the daily price index (used by training and the daily run)."""
    index = model.to_panel(daily)["C"].index
    naive = _naive(index)
    ctx: dict = {"macro_days": {pd.Timestamp(d) for d in macro}}
    if vix is not None and len(vix):
        v = vix.copy()
        v.index = _naive(v.index)
        ctx["vix"] = v[~v.index.duplicated()].reindex(naive, method="ffill").set_axis(index)
    if cal:
        ctx["earnings"] = calendars.earnings_panel(cal, naive, list(daily)).set_axis(index)
    return ctx


def fetch_vix(period: str) -> pd.Series | None:
    df = data._download(["^VIX"], period=period, interval="1d", auto_adjust=False).get("^VIX")
    return df["Close"].dropna() if df is not None else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="us1500")
    p.add_argument("--period", default="4y", help="history to download (first year warms up the averages)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--quick", action="store_true", help="skip the backtest (fast)")
    p.add_argument("--kinds", default="ridge,lgbm", help="model kinds to compare")
    p.add_argument("--no-earnings", action="store_true", help="don't fetch earnings dates")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    now = datetime.now(data.NY)

    tickers = universe.get_universe(args.universe)
    etfs = list(data.MARKET_ETFS) + list(data.SECTOR_ETFS.values())
    log.info("Downloading %s of daily history for %d stocks + %d ETFs", args.period, len(tickers), len(etfs))
    daily = data.fetch_daily(tickers, period=args.period, now=now)
    etf_daily = data.fetch_daily(etfs, period=args.period, now=now)
    sector_etf = {t: data.SECTOR_ETFS.get(universe.sector(t) or "") for t in daily}

    # Context: VIX, earnings dates, macro release days (also saved for the daily run).
    cal_path, macro_path = args.out / "earnings_calendar.json", args.out / "macro_days.json"
    cal = calendars.load(cal_path, {})
    if not args.no_earnings:
        fresh = calendars.fetch_earnings(list(daily))
        if fresh:
            cal = {**cal, **fresh}
            calendars.save(cal, cal_path)
    macro = calendars.fetch_macro_days() or calendars.load(macro_path, [])
    if macro:
        calendars.save(macro, macro_path)
    try:
        vix = fetch_vix(args.period)
    except Exception as exc:
        log.warning("VIX history failed: %s", exc)
        vix = None
    ctx = build_context(daily, vix, cal, macro)
    log.info("Context: VIX %s days, earnings for %d stocks, %d macro days",
             0 if vix is None else len(vix), len(cal), len(macro))

    kinds = tuple(k for k in args.kinds.split(",") if k)
    final, bt = model.train_all(daily, etf_daily, {t: e for t, e in sector_etf.items() if e},
                                ctx=ctx, kinds=kinds, backtest=not args.quick)
    final.meta["universe"] = args.universe
    model.save(final, args.out / "model.json")
    if args.quick and (args.out / "backtest.json").exists():
        log.info("Quick mode: model saved, previous backtest.json kept")
        return
    bt["generated_at"] = final.meta["trained_at"]
    if final.kind == "ridge":
        bt["weights"] = dict(zip(final.features, [round(float(w), 4) for w in final.weights]))
    (args.out / "backtest.json").write_text(json.dumps(bt, indent=1))

    o = bt.get("overall", {})
    log.info("Model trained on %s rows (%s .. %s), kind %s, strategy %s", final.meta.get("n_rows"),
             final.meta.get("trained_from"), final.meta.get("trained_through"), final.kind, final.meta.get("strategy"))
    for k, v in (bt.get("models") or {}).items():
        log.info("  %-6s out-of-sample IC %.4f | last 12m top-10 after costs %+.3f%%/day", k, v["avg_ic"],
                 v["last_12_months"]["top_avg_after_costs_pct"])
    log.info("Chosen: %s days, top-10 avg %.3f%% (after costs %.3f%%) vs all %.3f%%, hit %.1f%%",
             o.get("days", 0), o.get("top_avg_pct", 0), o.get("top_avg_after_costs_pct", 0),
             o.get("all_avg_pct", 0), 100 * o.get("top_hit_rate", 0))


if __name__ == "__main__":
    main()
