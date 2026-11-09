"""Stage 2 — data: download prices.

Two kinds of data:
  * daily bars (last ~6 months): trend, momentum, normal volume
  * today's premarket bars (4:00-9:30 ET): the strongest hint of how a stock will open

Data source in v1 is Yahoo Finance via `yfinance` (free, unofficial, fine for
personal use). Everything else in the pipeline only sees plain pandas objects, so
swapping in Polygon / Alpaca / Finnhub later only means changing this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)


@dataclass
class Premarket:
    last_price: float
    volume: float
    as_of: str  # ISO timestamp of the last premarket bar


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


def fetch_daily(
    tickers: list[str], period: str = "6mo", now: datetime | None = None
) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    now = now or datetime.now(NY)
    log.info("Downloading daily bars for %d tickers", len(tickers))
    df = yf.download(
        tickers, period=period, interval="1d", group_by="ticker",
        auto_adjust=True, progress=False, threads=True,
    )
    return drop_today_bar(_split_by_ticker(df, tickers), now)


def drop_today_bar(daily: dict[str, pd.DataFrame], now: datetime) -> dict[str, pd.DataFrame]:
    """Before the open, Yahoo may already return a half-empty bar for today.
    We want the last *completed* day to be the final row ("yesterday's close")."""
    if now.time() >= MARKET_OPEN:
        return daily
    today = now.date()
    return {t: df[pd.Index(df.index).map(lambda d: d.date() < today)] for t, df in daily.items()}


def fetch_premarket(tickers: list[str], now: datetime | None = None) -> dict[str, Premarket]:
    """Today's premarket price and volume for each ticker (empty if none yet)."""
    import yfinance as yf

    now = now or datetime.now(NY)
    log.info("Downloading premarket bars for %d tickers", len(tickers))
    df = yf.download(
        tickers, period="5d", interval="5m", prepost=True, group_by="ticker",
        auto_adjust=True, progress=False, threads=True,
    )
    return extract_premarket(_split_by_ticker(df, tickers), now.date())


def extract_premarket(intraday: dict[str, pd.DataFrame], day) -> dict[str, Premarket]:
    """Keep only bars from `day` before 9:30 ET. Separate function so it is testable."""
    out: dict[str, Premarket] = {}
    for t, bars in intraday.items():
        idx = bars.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        local = bars.set_axis(idx.tz_convert(NY))
        pre = local[(local.index.date == day) & (local.index.time < MARKET_OPEN)]
        pre = pre.dropna(subset=["Close"])
        if pre.empty:
            continue
        out[t] = Premarket(
            last_price=float(pre["Close"].iloc[-1]),
            volume=float(pre["Volume"].fillna(0).sum()),
            as_of=pre.index[-1].isoformat(),
        )
    return out
