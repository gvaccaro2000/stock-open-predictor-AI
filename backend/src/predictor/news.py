"""Phase 2, step 1 — news: collect recent headlines for each stock.

Sources:
  * Yahoo Finance news + press releases (free, via yfinance; always on)
  * Finnhub company news (free key; used only if FINNHUB_API_KEY is set)

We keep only items published since the previous market close, since older news
is already reflected in yesterday's price. Each item is a small `NewsItem`, so the
rest of the pipeline never depends on a specific provider's format.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

MAX_ITEMS_PER_TICKER = 8


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    url: str | None
    published: datetime  # timezone-aware

    def to_json(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat()
        return d


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def parse_yahoo(article: dict) -> NewsItem | None:
    """Handles both the current ({"content": {...}}) and the older flat format."""
    c = article.get("content") or article
    title = (c.get("title") or "").strip()
    published = _parse_time(c.get("pubDate") or c.get("providerPublishTime"))
    if not title or published is None:
        return None
    provider = c.get("provider") or {}
    url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link")
    return NewsItem(
        title=title,
        summary=(c.get("summary") or c.get("description") or "").strip()[:600],
        source=(provider.get("displayName") if isinstance(provider, dict) else None) or c.get("publisher") or "Yahoo Finance",
        url=url,
        published=published,
    )


def fetch_yahoo(ticker: str) -> list[NewsItem]:
    import yfinance as yf

    try:
        raw = yf.Ticker(ticker).get_news(count=15, tab="all")
    except Exception as exc:
        log.warning("Yahoo news failed for %s: %s", ticker, exc)
        return []
    return [i for i in (parse_yahoo(a) for a in raw or []) if i]


def fetch_finnhub(ticker: str, since: datetime, api_key: str) -> list[NewsItem]:
    q = urllib.parse.urlencode({
        "symbol": ticker,
        "from": (since - timedelta(days=1)).date().isoformat(),
        "to": datetime.now(timezone.utc).date().isoformat(),
        "token": api_key,
    })
    try:
        with urllib.request.urlopen(f"https://finnhub.io/api/v1/company-news?{q}", timeout=15) as r:
            rows = json.load(r)
    except Exception as exc:
        log.warning("Finnhub news failed for %s: %s", ticker, exc)
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        published = _parse_time(row.get("datetime"))
        if row.get("headline") and published:
            out.append(NewsItem(row["headline"].strip(), (row.get("summary") or "")[:600],
                                row.get("source") or "Finnhub", row.get("url"), published))
    return out


def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    seen, out = set(), []
    for i in sorted(items, key=lambda x: x.published, reverse=True):
        key = "".join(ch for ch in i.title.lower() if ch.isalnum())[:80]
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def news_for(ticker: str, since: datetime, finnhub_key: str | None = None) -> list[NewsItem]:
    items = fetch_yahoo(ticker)
    if finnhub_key:
        items += fetch_finnhub(ticker, since, finnhub_key)
    fresh = [i for i in items if i.published >= since]
    return _dedupe(fresh)[:MAX_ITEMS_PER_TICKER]


def collect(tickers: list[str], since: datetime, fetcher=None) -> dict[str, list[NewsItem]]:
    """News for many tickers in parallel. `fetcher` can be swapped in tests."""
    key = os.environ.get("FINNHUB_API_KEY") or None
    fetcher = fetcher or (lambda t: news_for(t, since, key))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(tickers, pool.map(fetcher, tickers)))
    n = sum(len(v) for v in results.values())
    log.info("Collected %d fresh news items for %d tickers (Finnhub %s)", n, len(tickers), "on" if key else "off")
    return results
