"""Stage 1 — the universe: which tickers we analyse.

We start with large, liquid US stocks. Illiquid small caps have noisy prices that
make any signal unreliable.
"""

from __future__ import annotations

import io
import re
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

SAMPLE_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)", "META": "Meta Platforms", "TSLA": "Tesla",
    "AVGO": "Broadcom", "JPM": "JPMorgan Chase", "V": "Visa", "UNH": "UnitedHealth",
    "XOM": "ExxonMobil", "LLY": "Eli Lilly", "JNJ": "Johnson & Johnson",
    "WMT": "Walmart", "MA": "Mastercard", "PG": "Procter & Gamble",
    "HD": "Home Depot", "COST": "Costco", "NFLX": "Netflix", "AMD": "AMD",
    "CRM": "Salesforce", "ORCL": "Oracle", "ADBE": "Adobe", "PEP": "PepsiCo",
    "KO": "Coca-Cola", "BAC": "Bank of America", "CVX": "Chevron", "MRK": "Merck",
    "INTC": "Intel",
}

IT, FIN, HC, CD, CS, EN, CM = ("Information Technology", "Financials", "Health Care",
                               "Consumer Discretionary", "Consumer Staples", "Energy",
                               "Communication Services")
SAMPLE_SECTORS = {
    "AAPL": IT, "MSFT": IT, "NVDA": IT, "AVGO": IT, "AMD": IT, "CRM": IT, "ORCL": IT, "ADBE": IT, "INTC": IT,
    "AMZN": CD, "TSLA": CD, "HD": CD, "GOOGL": CM, "META": CM, "NFLX": CM,
    "JPM": FIN, "V": FIN, "MA": FIN, "BAC": FIN, "UNH": HC, "LLY": HC, "JNJ": HC, "MRK": HC,
    "XOM": EN, "CVX": EN, "WMT": CS, "PG": CS, "COST": CS, "PEP": CS, "KO": CS,
}

# Filled in when the S&P 500 list is loaded: ticker -> company name / GICS sector.
NAMES: dict[str, str] = dict(SAMPLE_NAMES)
SECTORS: dict[str, str] = dict(SAMPLE_SECTORS)

# Wikipedia lists of index members. Each page has a table with a ticker column
# ("Symbol" or "Ticker"), a name column and usually a "GICS Sector" column.
INDEX_PAGES = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Large cap"),
    "sp400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Mid cap"),
    "sp600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Small cap"),
    "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Large cap"),
}
SIZES: dict[str, str] = {}


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "stock-open-predictor/0.2"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return resp.read().decode("utf-8")


def parse_members(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Find the constituents table and normalise it to columns: ticker, name, sector."""
    for t in tables:
        cols = {str(c).strip(): c for c in t.columns}
        tick = next((cols[c] for c in ("Symbol", "Ticker", "Ticker symbol") if c in cols), None)
        name = next((cols[c] for c in ("Security", "Company", "Company Name") if c in cols), None)
        if tick is None or name is None or len(t) < 50:
            continue
        sec = cols.get("GICS Sector")
        out = pd.DataFrame({
            "ticker": t[tick].astype(str).str.strip().str.replace(".", "-", regex=False),
            "name": t[name].astype(str).str.strip(),
            "sector": t[sec].astype(str).str.strip() if sec is not None else None,
        })
        return out[out["ticker"].str.fullmatch(r"[A-Z][A-Z0-9-]{0,6}")]
    raise ValueError("no constituents table found")


def load_index(key: str) -> list[str]:
    url, size = INDEX_PAGES[key]
    members = parse_members(pd.read_html(io.StringIO(_fetch_html(url))))
    for r in members.itertuples():
        NAMES.setdefault(r.ticker, r.name)
        if isinstance(r.sector, str) and r.sector not in ("", "nan"):
            SECTORS.setdefault(r.ticker, r.sector)
        SIZES.setdefault(r.ticker, size)
    return members["ticker"].tolist()


def load_indexes(keys: list[str]) -> list[str]:
    """Union of several index lists; an index that fails to load is skipped."""
    tickers: list[str] = []
    for k in keys:
        try:
            got = load_index(k)
            log.info("Loaded %d members of %s", len(got), k)
            tickers += got
        except Exception as exc:  # network down, page layout changed, ...
            log.warning("Could not load %s (%s)", k, exc)
    if not tickers:
        log.warning("No index lists loaded; using the sample list")
        return SAMPLE
    return sorted(set(tickers))


def sp500() -> list[str]:
    return load_indexes(["sp500"])


def clean_name(name: str) -> str:
    """'Hershey Company (The)' -> 'Hershey Company'; 'Apple Inc.' -> 'Apple'."""
    name = re.sub(r"\s*\(The\)$", "", name.strip())
    name = re.sub(r",?\s+(Inc\.?|Incorporated|Corp\.?|Co\.|plc|Ltd\.?|N\.V\.|S\.A\.)$", "", name)
    return name


def company_name(ticker: str) -> str | None:
    n = NAMES.get(ticker)
    return clean_name(n) if n else None


def get_universe(name: str) -> list[str]:
    if name == "sample":
        return SAMPLE
    if name == "sp500":
        return sp500()
    if name == "us1500":   # S&P 500 + MidCap 400 + SmallCap 600 + Nasdaq-100 (~1,500 stocks)
        return load_indexes(["sp500", "sp400", "sp600", "nasdaq100"])
    # Allow a comma-separated custom list: --universe AAPL,MSFT
    return [t.strip().upper() for t in name.split(",") if t.strip()]


def sector(ticker: str) -> str | None:
    return SECTORS.get(ticker)


def size(ticker: str) -> str | None:
    return SIZES.get(ticker)
