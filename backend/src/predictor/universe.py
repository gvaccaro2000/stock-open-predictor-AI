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

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500() -> list[str]:
    """Current S&P 500 constituents from Wikipedia (falls back to SAMPLE)."""
    try:
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": "stock-open-predictor/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8")
        table = pd.read_html(io.StringIO(html))[0]
        # Yahoo uses "-" instead of "." (e.g. BRK.B -> BRK-B)
        table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
        if "Security" in table.columns:
            NAMES.update(dict(zip(table["Symbol"], table["Security"])))
        if "GICS Sector" in table.columns:
            SECTORS.update(dict(zip(table["Symbol"], table["GICS Sector"])))
        return sorted(table["Symbol"].tolist())
    except Exception as exc:  # network down, page layout changed, ...
        log.warning("Could not load S&P 500 list (%s); using sample list", exc)
        return SAMPLE


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
    # Allow a comma-separated custom list: --universe AAPL,MSFT
    return [t.strip().upper() for t in name.split(",") if t.strip()]


def sector(ticker: str) -> str | None:
    return SECTORS.get(ticker)
