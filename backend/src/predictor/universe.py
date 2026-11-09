"""Stage 1 — the universe: which tickers we analyse.

We start with large, liquid US stocks. Illiquid small caps have noisy prices that
make any signal unreliable.
"""

from __future__ import annotations

import io
import logging
import urllib.request

import pandas as pd

log = logging.getLogger(__name__)

# A small, fast list for development and tests.
SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "COST", "NFLX",
    "AMD", "CRM", "ORCL", "ADBE", "PEP", "KO", "BAC", "CVX", "MRK", "INTC",
]

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500() -> list[str]:
    """Current S&P 500 constituents from Wikipedia (falls back to SAMPLE)."""
    try:
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": "stock-open-predictor/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8")
        table = pd.read_html(io.StringIO(html))[0]
        # Yahoo uses "-" instead of "." (e.g. BRK.B -> BRK-B)
        return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception as exc:  # network down, page layout changed, ...
        log.warning("Could not load S&P 500 list (%s); using sample list", exc)
        return SAMPLE


def get_universe(name: str) -> list[str]:
    if name == "sample":
        return SAMPLE
    if name == "sp500":
        return sp500()
    # Allow a comma-separated custom list: --universe AAPL,MSFT
    return [t.strip().upper() for t in name.split(",") if t.strip()]
