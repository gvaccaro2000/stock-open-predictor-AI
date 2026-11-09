"""Stage 2 — data: download prices.

  * daily bars (last ~6 months): trend, momentum, volatility (ATR), key levels
  * extended-hours bars (5-minute): the price *before* the regular session
      - morning run (before 9:30 ET): today's pre-market (4:00-9:30)
      - evening run (after 16:00 ET): today's after-hours (16:00-20:00)
      - during the day: the latest price (marked as not a real pre-open signal)
  * market context: S&P 500 / Nasdaq / small caps (SPY, QQQ, IWM), sector ETFs,
    VIX ("fear index") and the US 10-year Treasury yield

Data source is Yahoo Finance via `yfinance` (free, unofficial, fine for personal
use). Everything else only sees plain pandas objects / dataclasses, so another
provider can be swapped in here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
PRE_START, MARKET_OPEN, MARKET_CLOSE, POST_END = time(4, 0), time(9, 30), time(16, 0), time(20, 0)

MARKET_ETFS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Small caps (Russell 2000)"}
SECTOR_ETFS = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
INDEX_TICKERS = {"^VIX": "VIX", "^TNX": "US 10-year yield"}


@dataclass
class Premarket:
    """Extended-hours snapshot (pre-market in the morning, after-hours in the evening)."""
    last_price: float
    volume: float
    as_of: str
    high: float = math.nan
    low: float = math.nan
    rvol: float = math.nan          # volume vs the same time window on previous days
    session: str = "premarket"      # premarket | after_hours | regular


def session_for(now: datetime) -> str:
    t = now.time()
    if t < MARKET_OPEN:
        return "premarket"
    if t >= MARKET_CLOSE:
        return "after_hours"
    return "regular"


def _split_by_ticker(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """yfinance returns one wide table; split it into one table per ticker."""
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t in df.columns.get_level_values(0):
                sub = df[t].dropna(how="all")
                if not sub.empty:
                    out[t] = sub
    elif len(tickers) == 1:
        out[tickers[0]] = df.dropna(how="all")
    return out


def fetch_daily(tickers: list[str], period: str = "6mo", now: datetime | None = None) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    now = now or datetime.now(NY)
    log.info("Downloading daily bars for %d tickers", len(tickers))
    df = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                     auto_adjust=True, progress=False, threads=True)
    return drop_today_bar(_split_by_ticker(df, tickers), now)


def drop_today_bar(daily: dict[str, pd.DataFrame], now: datetime) -> dict[str, pd.DataFrame]:
    """The last row must be the last *completed* session ("the previous close").
    Before the close, today's bar is missing or incomplete, so drop it."""
    if now.time() >= MARKET_CLOSE:
        return daily
    today = now.date()
    return {t: df[pd.Index(df.index).map(lambda d: d.date() < today)] for t, df in daily.items()}


def fetch_premarket(tickers: list[str], now: datetime | None = None) -> dict[str, Premarket]:
    import yfinance as yf

    now = now or datetime.now(NY)
    log.info("Downloading extended-hours bars for %d tickers", len(tickers))
    df = yf.download(tickers, period="10d", interval="5m", prepost=True, group_by="ticker",
                     auto_adjust=True, progress=False, threads=True)
    return extract_extended(_split_by_ticker(df, tickers), now)


def _window(session: str, now: datetime) -> tuple[time, time]:
    if session == "premarket":
        return PRE_START, min(now.time(), MARKET_OPEN)
    if session == "after_hours":
        return MARKET_CLOSE, min(now.time(), POST_END)
    return PRE_START, now.time()  # regular: everything so far today


def extract_extended(intraday: dict[str, pd.DataFrame], now: datetime) -> dict[str, Premarket]:
    """Today's extended-hours price, range and relative volume. Separate so it's testable."""
    session = session_for(now)
    start, end = _window(session, now)
    out: dict[str, Premarket] = {}
    for t, bars in intraday.items():
        idx = bars.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        local = bars.set_axis(idx.tz_convert(NY)).dropna(subset=["Close"])
        times = local.index.time
        in_window = local[(times >= start) & (times <= end)]
        today = in_window[in_window.index.date == now.date()]
        if today.empty:
            continue
        vol = float(today["Volume"].fillna(0).sum())
        prior = in_window[in_window.index.date < now.date()]
        per_day = prior["Volume"].fillna(0).groupby(prior.index.date).sum()
        per_day = per_day[per_day > 0]
        rvol = vol / float(per_day.mean()) if len(per_day) else math.nan
        out[t] = Premarket(
            last_price=float(today["Close"].iloc[-1]), volume=vol, as_of=today.index[-1].isoformat(),
            high=float(today.get("High", today["Close"]).max()), low=float(today.get("Low", today["Close"]).min()),
            rvol=rvol, session=session,
        )
    return out


# Backwards-compatible name used by earlier tests/code.
def extract_premarket(intraday: dict[str, pd.DataFrame], day) -> dict[str, Premarket]:
    return extract_extended(intraday, datetime.combine(day, time(9, 29), tzinfo=NY))


def fetch_indexes(now: datetime | None = None) -> dict[str, pd.DataFrame]:
    """VIX and 10-year yield (daily). These indexes don't trade pre-market."""
    import yfinance as yf

    df = yf.download(list(INDEX_TICKERS), period="1mo", interval="1d", group_by="ticker",
                     auto_adjust=False, progress=False, threads=True)
    return _split_by_ticker(df, list(INDEX_TICKERS))
