"""Stage 4 — scoring: combine features into one score with reasons.

v1 is a transparent, rule-based model:
  1. Standardise each feature across all stocks (how unusual is this value today?)
  2. Multiply by a weight (how much we believe in it) and add up
  3. Convert to a 0-100 rank so the app can show "signal strength"

Phase 2 adds a news signal: Claude's assessment of the company's overnight news
(-1..+1, already scaled by importance) times NEWS_WEIGHT. It is added directly,
not standardised, because most stocks have no news (score 0) on a given day.

The weights are educated guesses. Phase 3 replaces them with weights learned from
history and checked by a backtest. Until then, treat the scores as hypotheses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# feature -> weight. Negative weight = higher value is *bad* for an up-open.
WEIGHTS: dict[str, float] = {
    "premarket_gap_pct": 0.45,
    "premarket_confirmation": 0.15,  # gap direction x premarket activity (derived below)
    "momentum_5d_pct": 0.15,
    "trend_vs_ma50_pct": 0.10,
    "rel_volume_1d": 0.10,
    "rsi_14": -0.05,
}
NEWS_WEIGHT = 1.5      # strong, important news ~ about half the maximum pre-market effect
NEWS_MIN_EFFECT = 0.1  # below this, news isn't mentioned in the explanation

# Written for people with no finance background: no jargon, numbers only where helpful.
REASON_TEXT = {
    "premarket_gap_pct": ("Already up {premarket_gap_pct:+.1f}% in early trading before the market opens",
                          "Down {premarket_gap_pct:+.1f}% in early trading before the market opens"),
    "premarket_confirmation": ("Lots of people are trading it this morning, so the early move looks solid",
                               "Heavy early selling this morning"),
    "momentum_5d_pct": ("Has been rising this week ({momentum_5d_pct:+.1f}% in 5 days)",
                        "Has been falling this week ({momentum_5d_pct:+.1f}% in 5 days)"),
    "trend_vs_ma50_pct": ("On an upward trend over the last few months",
                          "On a downward trend over the last few months"),
    "rel_volume_1d": ("More people than usual traded it yesterday",
                      "Fewer people than usual traded it yesterday"),
    "rsi_14": ("Hasn't climbed too far, too fast",
               "Has climbed a lot recently, so it may need a breather"),
}


def _is_true(feat: str, positive: bool, row: pd.Series) -> bool:
    """Scores are relative to other stocks, but the words must be literally true.
    E.g. a +0.6% week can score above average, but we can't call it "a strong week"."""
    v = row.get(feat, np.nan)
    gap = row.get("premarket_gap_pct", np.nan)
    checks = {
        "premarket_gap_pct": (v > 0, v < 0),
        "premarket_confirmation": (gap > 0, gap < 0),
        "momentum_5d_pct": (v > 1.0, v < 0),
        "trend_vs_ma50_pct": (v > 0, v < 0),
        "rel_volume_1d": (v > 1, v < 1),
        "rsi_14": (v < 65, v > 70),
    }
    ok_pos, ok_neg = checks.get(feat, (True, True))
    return bool(ok_pos if positive else ok_neg)


def robust_z(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Like a z-score, but using median/MAD so one crazy outlier doesn't distort it."""
    med = s.median()
    scale = (s - med).abs().median() * 1.4826
    if not np.isfinite(scale) or scale == 0:
        # Most values identical (e.g. few stocks moving premarket): fall back to std
        scale = s.std()
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - med) / scale).clip(-clip, clip).fillna(0.0)


def score(features: pd.DataFrame, news: dict | None = None, n_reasons: int = 3) -> pd.DataFrame:
    """`news` maps ticker -> sentiment.NewsAssessment (optional)."""
    df = features.copy()
    # Gap in the same direction as heavy premarket volume = "confirmed" gap.
    df["premarket_confirmation"] = (
        np.sign(df["premarket_gap_pct"]) * np.log1p(df["premarket_activity"] * 100)
    )

    contrib = pd.DataFrame(index=df.index)
    for feat, w in WEIGHTS.items():
        contrib[feat] = robust_z(df[feat].astype(float)) * w

    news = news or {}
    df["news_score"] = pd.Series({t: a.score for t, a in news.items()}, dtype=float).reindex(df.index)
    df["news_summary"] = pd.Series({t: a.summary for t, a in news.items()}, dtype=object).reindex(df.index)
    contrib["news"] = df["news_score"].fillna(0.0) * NEWS_WEIGHT

    df["raw_score"] = contrib.sum(axis=1)
    df["score"] = (df["raw_score"].rank(pct=True) * 100).round(0)
    df["reasons"] = [
        _news_reason(df.loc[t], contrib.loc[t]) + _reasons(df.loc[t], contrib.loc[t].drop("news"), n_reasons)
        for t in df.index
    ]
    df["summary"] = [summary(df.loc[t], contrib.loc[t]) for t in df.index]
    df["why_short"] = [short_why(df.loc[t], contrib.loc[t]) for t in df.index]
    return df.sort_values("raw_score", ascending=False)


def _news_reason(row: pd.Series, contrib: pd.Series) -> list[dict]:
    c = float(contrib.get("news", 0.0))
    if abs(c) < NEWS_MIN_EFFECT or not isinstance(row.get("news_summary"), str):
        return []
    return [{"text": row["news_summary"], "impact": "positive" if c > 0 else "negative", "kind": "news"}]


def _reasons(row: pd.Series, contrib: pd.Series, n: int) -> list[dict]:
    """The n features that moved this stock's score the most, as readable text."""
    out = []
    for feat in contrib.abs().sort_values(ascending=False).index:
        if len(out) >= n:
            break
        c = float(contrib[feat])
        if abs(c) < 0.05 or not np.isfinite(row.get(feat, np.nan)):
            continue
        if not _is_true(feat, c > 0, row):
            continue
        pos, neg = REASON_TEXT[feat]
        # For a negative-weight feature, a positive contribution means a LOW value.
        out.append({
            "text": (pos if c > 0 else neg).format(**row.to_dict()),
            "impact": "positive" if c > 0 else "negative",
        })
    return out


# Short clauses used to build a one or two sentence "why" for each stock.
_UP_CLAUSE = {
    "premarket_gap_pct": "up {premarket_gap_pct:.1f}% before the open",
    "premarket_confirmation": "on heavy early trading",
    "momentum_5d_pct": "after a strong week ({momentum_5d_pct:+.1f}%)",
    "trend_vs_ma50_pct": "in a steady uptrend for months",
    "rel_volume_1d": "with busy trading yesterday",
    "rsi_14": "without looking overheated",
}
_CAUTION = {
    "premarket_gap_pct": "it is trading lower before the open",
    "momentum_5d_pct": "it fell over the past week",
    "trend_vs_ma50_pct": "its longer-term trend is down",
    "rsi_14": "it has already climbed a lot recently",
    "premarket_confirmation": "early selling is heavy",
}


def summary(row: pd.Series, contrib: pd.Series) -> str:
    """One or two plain-English sentences explaining why the stock scores well."""
    price = _price_summary(row, contrib.drop("news", errors="ignore"))
    return _with_news(row, contrib, price, lead_full=True)


def _with_news(row: pd.Series, contrib: pd.Series, price_text: str, lead_full: bool) -> str:
    """Put strong good news first; add strong bad news as a caution."""
    c = float(contrib.get("news", 0.0))
    news_text = row.get("news_summary")
    if not isinstance(news_text, str) or abs(c) < NEWS_MIN_EFFECT:
        return price_text
    news_text = news_text.rstrip(".") + "."
    if c > 0:
        return f"{news_text} {price_text}" if lead_full else news_text
    return f"{price_text} Caution: {news_text[0].lower() + news_text[1:]}"


def _price_summary(row: pd.Series, contrib: pd.Series) -> str:
    pos = [f for f in contrib.sort_values(ascending=False).index
           if f in _UP_CLAUSE and contrib[f] > 0.05 and _is_true(f, True, row)]
    clauses = [_UP_CLAUSE[f].format(**row.to_dict()) for f in pos[:3]]
    if not clauses:
        text = "Several small positive signs, none stand-out on its own."
    else:
        first = clauses[0][0].upper() + clauses[0][1:]
        if len(clauses) == 1:
            text = first + "."
        else:
            text = first + ", " + " and ".join(clauses[1:]).replace(" and on", " on") + "."
    negs = [f for f in contrib.sort_values().index
            if contrib[f] < -0.15 and f in _CAUTION and _is_true(f, False, row)]
    neg = negs[0] if negs else None
    if neg:
        text += f" Caution: {_CAUTION[neg]}."
    return text


# Very short version for the list view, where the expected % is already shown.
_SHORT = {
    "premarket_confirmation": "heavy early trading behind the move",
    "momentum_5d_pct": "a strong week ({momentum_5d_pct:+.1f}%)",
    "trend_vs_ma50_pct": "a steady uptrend for months",
    "rel_volume_1d": "busy trading yesterday",
    "rsi_14": "no sign of overheating",
}
_SHORT_CAUTION = {
    "momentum_5d_pct": "fell last week",
    "trend_vs_ma50_pct": "longer-term trend is down",
    "rsi_14": "already up a lot recently",
    "premarket_confirmation": "heavy early selling",
    "premarket_gap_pct": "trading lower before the open",
}


def short_why(row: pd.Series, contrib: pd.Series) -> str:
    price = _price_short(row, contrib.drop("news", errors="ignore"))
    return _with_news(row, contrib, price, lead_full=False)


def _price_short(row: pd.Series, contrib: pd.Series) -> str:
    pos = [f for f in contrib.sort_values(ascending=False).index
           if f in _SHORT and contrib[f] > 0.05 and _is_true(f, True, row)]
    parts = [_SHORT[f].format(**row.to_dict()) for f in pos[:2]]
    if parts:
        text = " and ".join(parts)
        text = text[0].upper() + text[1:] + "."
    elif row.get("premarket_gap_pct", 0) > 0:
        text = "Rising before the open."
    else:
        text = "Several small positive signs."
    negs = [f for f in contrib.sort_values().index
            if contrib[f] < -0.15 and f in _SHORT_CAUTION and _is_true(f, False, row)]
    neg = negs[0] if negs else None
    if neg:
        text += f" Caution: {_SHORT_CAUTION[neg]}."
    return text
