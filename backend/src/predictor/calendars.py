"""Event calendars: earnings dates, US macro release days, ex-dividend dates.

earnings   Yahoo Finance earnings dates (past ~4 years + upcoming), fetched weekly during
           training and saved to docs/data/earnings_calendar.json so the daily run can
           see whether results came out between yesterday's close and today's open.
macro      Release days of major US data (CPI, jobs report, GDP, PCE, retail sales...)
           from FRED (St. Louis Fed). Needs a free FRED_API_KEY; skipped without one.
dividends  On an ex-dividend morning the price drops by the dividend for purely
           mechanical reasons. The training data is dividend-adjusted, so we apply the
           same adjustment live for the top candidates.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from .data import NY

log = logging.getLogger(__name__)
OPEN, CLOSE = time(9, 30), time(16, 0)


# ----------------------------------------------------------------------------- earnings
def _earnings_one(ticker: str, limit: int = 20) -> list[str]:
    import yfinance as yf

    for attempt in (1, 2, 3):
        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
            if df is None or df.empty:
                return []
            idx = pd.DatetimeIndex(df.index)
            idx = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
            return sorted({d.isoformat() for d in idx})
        except Exception as exc:  # rate limits, missing data
            if attempt == 3:
                log.debug("earnings dates failed for %s: %s", ticker, exc)
                return []
            _time.sleep(2 * attempt)
    return []


def fetch_earnings(tickers: list[str], workers: int = 6) -> dict[str, list[str]]:
    log.info("Fetching earnings dates for %d stocks", len(tickers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = dict(zip(tickers, pool.map(_earnings_one, tickers)))
    got = sum(1 for v in out.values() if v)
    log.info("Earnings dates found for %d of %d stocks", got, len(tickers))
    return {t: v for t, v in out.items() if v}


def earnings_session_date(ts: datetime, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The trading day whose *open* first reflects an announcement at `ts`."""
    t = ts.astimezone(NY) if ts.tzinfo else ts.replace(tzinfo=NY)
    d = pd.Timestamp(t.date())
    if t.time() >= CLOSE or (OPEN <= t.time() < CLOSE and t.time() != time(0, 0)):
        later = trading_days[trading_days > d]          # after the close (or during the day): next open
        return later[0] if len(later) else d + pd.offsets.BDay(1)
    same = trading_days[trading_days >= d]               # before the open (or time unknown): that open
    return same[0] if len(same) else d


def earnings_panel(cal: dict[str, list[str]], index: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """1 on the trading day each stock's results first hit the open, else 0."""
    out = pd.DataFrame(0.0, index=index, columns=list(tickers))
    idx = pd.DatetimeIndex(index)
    for t, stamps in cal.items():
        if t not in out.columns:
            continue
        for s in stamps:
            d = earnings_session_date(datetime.fromisoformat(s), idx)
            if d is not None and d in out.index:
                out.loc[d, t] = 1.0
    return out


# ----------------------------------------------------------------------------- macro (FRED)
FRED = "https://api.stlouisfed.org/fred"
MACRO_RELEASES = [  # name fragments of FRED releases that move the whole market
    "Consumer Price Index", "Employment Situation", "Gross Domestic Product",
    "Personal Income and Outlays", "Advance Monthly Sales for Retail", "Producer Price Index",
    "FOMC Press Release",
]


def _fred(path: str, **params) -> dict:
    params |= {"api_key": os.environ["FRED_API_KEY"], "file_type": "json"}
    with urllib.request.urlopen(f"{FRED}/{path}?{urllib.parse.urlencode(params)}", timeout=20) as r:
        return json.load(r)


def fetch_macro_days(start: str = "2021-01-01") -> list[str]:
    """Dates of major US releases (past and scheduled). [] without FRED_API_KEY."""
    if not os.environ.get("FRED_API_KEY"):
        log.info("FRED_API_KEY not set — macro release days skipped")
        return []
    try:
        releases = _fred("releases", limit=1000)["releases"]
        ids = {r["id"]: r["name"] for r in releases
               if any(frag.lower() in r["name"].lower() for frag in MACRO_RELEASES)}
        days: set[str] = set()
        for rid in ids:
            d = _fred("release/dates", release_id=rid, realtime_start=start, limit=10000,
                      include_release_dates_with_no_data="true")
            days |= {x["date"] for x in d.get("release_dates", [])}
        log.info("Macro release days: %d from %d FRED releases", len(days), len(ids))
        return sorted(days)
    except Exception as exc:
        log.warning("FRED macro calendar failed: %s", exc)
        return []


# ----------------------------------------------------------------------------- dividends
def _exdiv_one(ticker: str, day: date) -> float | None:
    """Dividend per share if `day` is the ex-dividend date, else None."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    ex = info.get("exDividendDate")
    amt = info.get("lastDividendValue")
    if not ex or not amt:
        return None
    ex_day = datetime.fromtimestamp(ex, tz=NY).date() if isinstance(ex, (int, float)) else pd.Timestamp(ex).date()
    return float(amt) if ex_day == day else None


def ex_dividends_today(tickers: list[str], day: date, workers: int = 8) -> dict[str, float]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = dict(zip(tickers, pool.map(lambda t: _exdiv_one(t, day), tickers)))
    return {t: v for t, v in res.items() if v}


def adjust_for_dividend(df: pd.DataFrame, dividend: float) -> pd.DataFrame:
    """Back-adjust a price history for a dividend (same convention as the training data)."""
    prev = float(df["Close"].dropna().iloc[-1])
    if prev <= dividend:
        return df
    f = 1 - dividend / prev
    out = df.copy()
    for c in ("Open", "High", "Low", "Close"):
        out[c] = out[c] * f
    return out


# ----------------------------------------------------------------------------- io
def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default
