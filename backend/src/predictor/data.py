"""Stage 2 — data: download prices.

  * daily bars (last ~18 months): trend, 50/200-day averages, momentum, volatility (ATR), key levels
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
    peak: float = math.nan          # highest price since the last regular close (after-hours + pre-market)
    trend_30m: float = math.nan     # % change over the last 30 minutes of the session so far
    range_pos: float = math.nan     # 0 = at the session low, 1 = at the session high
    vwap_dev: float = math.nan      # % above/below the session's volume-weighted average price


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


CHUNK = 200  # tickers per Yahoo request; big lists in one go get rate-limited


def _download(tickers: list[str], **kw) -> dict[str, pd.DataFrame]:
    """yf.download in chunks, with one retry per chunk."""
    import time as _time

    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        for attempt in (1, 2):
            try:
                df = yf.download(chunk, group_by="ticker", progress=False, threads=True, **kw)
                got = _split_by_ticker(df, chunk)
                out.update(got)
                if got or attempt == 2:
                    break
            except Exception as exc:
                log.warning("Download chunk %d failed (attempt %d): %s", i // CHUNK + 1, attempt, exc)
            _time.sleep(3)
        _time.sleep(0.5)
    return out


def fetch_daily(tickers: list[str], period: str = "18mo", now: datetime | None = None) -> dict[str, pd.DataFrame]:
    now = now or datetime.now(NY)
    log.info("Downloading daily bars for %d tickers", len(tickers))
    daily = _download(tickers, period=period, interval="1d", auto_adjust=True)
    return drop_today_bar(daily, now)


def liquid(daily: dict[str, pd.DataFrame], min_price: float = 3.0, min_dollar_vol: float = 5e6) -> dict[str, pd.DataFrame]:
    """Drop stocks that are too cheap or too rarely traded for reliable signals."""
    out = {}
    for t, df in daily.items():
        c, v = df["Close"].dropna(), df["Volume"].fillna(0)
        if len(c) < 21:
            continue
        dollar_vol = float((c.iloc[-20:] * v.reindex(c.index).iloc[-20:]).mean())
        if float(c.iloc[-1]) >= min_price and dollar_vol >= min_dollar_vol:
            out[t] = df
    return out


def drop_today_bar(daily: dict[str, pd.DataFrame], now: datetime) -> dict[str, pd.DataFrame]:
    """The last row must be the last *completed* session ("the previous close").
    Before the close, today's bar is missing or incomplete, so drop it."""
    if now.time() >= MARKET_CLOSE:
        return daily
    today = now.date()
    return {t: df[pd.Index(df.index).map(lambda d: d.date() < today)] for t, df in daily.items()}


def fetch_premarket(tickers: list[str], now: datetime | None = None) -> dict[str, Premarket]:
    now = now or datetime.now(NY)
    log.info("Downloading extended-hours bars for %d tickers", len(tickers))
    bars = _download(tickers, period="10d", interval="5m", prepost=True, auto_adjust=True)
    return extract_extended(bars, now)


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
        # Peak since the last regular-session close: previous evening's after-hours + today so far.
        if session == "premarket":
            prev_days = sorted(d for d in set(local.index.date) if d < now.date())
            evening = local[(local.index.date == prev_days[-1]) & (times >= MARKET_CLOSE)] if prev_days else local.iloc[0:0]
            since = pd.concat([evening, today])
        else:
            since = today
        hi_col = since["High"] if "High" in since else since["Close"]
        peak = float(hi_col.max()) if len(since) else math.nan
        last = float(today["Close"].iloc[-1])
        recent = today[today.index >= today.index[-1] - pd.Timedelta(minutes=30)]
        trend_30m = (last / float(recent["Close"].iloc[0]) - 1) * 100 if len(recent) > 1 else math.nan
        s_hi = float(today.get("High", today["Close"]).max())
        s_lo = float(today.get("Low", today["Close"]).min())
        range_pos = (last - s_lo) / (s_hi - s_lo) if s_hi > s_lo else math.nan
        v = today["Volume"].fillna(0)
        vwap = float((today["Close"] * v).sum() / v.sum()) if v.sum() > 0 else math.nan
        vwap_dev = (last / vwap - 1) * 100 if vwap and math.isfinite(vwap) else math.nan
        out[t] = Premarket(
            last_price=float(today["Close"].iloc[-1]), volume=vol, as_of=today.index[-1].isoformat(),
            high=float(today.get("High", today["Close"]).max()), low=float(today.get("Low", today["Close"]).min()),
            rvol=rvol, session=session, peak=peak, trend_30m=trend_30m, range_pos=range_pos, vwap_dev=vwap_dev,
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
