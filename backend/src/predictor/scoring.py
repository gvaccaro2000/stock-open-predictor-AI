"""Stage 4 — scoring: combine signals into one score with plain-English reasons.

A transparent, rule-based model:
  1. Standardise each signal across all stocks (how unusual is this value today?)
  2. Multiply by a weight (how much we believe in it) and add up
  3. Add direct adjustments that don't make sense to standardise:
       news        Claude's news assessment (-1..+1, scaled by importance) x NEWS_WEIGHT
       thin        very little early trading -> the move is less reliable
       unexplained a big jump with no news found to explain it
  4. Convert to a 0-100 rank

The weights follow the usual pre-open priorities (catalyst, early move, participation,
relative strength, price levels, then trend; RSI only minor). They are still educated
guesses: a later phase learns them from history and checks them with a backtest.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# signal -> weight. Negative weight = a higher value is *bad* for an up-open.
WEIGHTS: dict[str, float] = {
    "premarket_gap_pct": 0.40,
    "premarket_confirmation": 0.15,   # gap direction x relative early volume (derived below)
    "rel_strength_pct": 0.10,         # beating its sector / the market this morning
    "level_signal": 0.08,             # above yesterday's high (+) / just under a recent high (-)
    "momentum_5d_pct": 0.10,
    "trend_vs_ma50_pct": 0.07,
    "rel_volume_1d": 0.05,
    "overextension": -0.08,           # gap already far beyond the stock's normal daily range
    "rsi_14": -0.03,
}
NEWS_WEIGHT = 1.5        # strong, important news ~ about half the maximum early-move effect
NEWS_MIN_EFFECT = 0.1    # below this, news isn't mentioned in the explanation
THIN_DOLLAR_VOL = 500_000
THIN_PENALTY = -0.30
UNEXPLAINED_GAP = 2.0
UNEXPLAINED_PENALTY = -0.15

# Written for people with no finance background: no jargon, numbers only where helpful.
REASON_TEXT = {
    "premarket_gap_pct": ("Already up {premarket_gap_pct:+.1f}% in early trading before the market opens",
                          "Down {premarket_gap_pct:+.1f}% in early trading before the market opens"),
    "premarket_confirmation": ("Much busier early trading than on a normal morning, so the move looks solid",
                               "Heavy early selling this morning"),
    "rel_strength_pct": ("Rising more than its sector this morning",
                         "Doing worse than its sector this morning"),
    "level_signal": ("Trading above yesterday's high, a possible breakout",
                     "Just below a recent high, which can act as a ceiling"),
    "momentum_5d_pct": ("Has been rising this week ({momentum_5d_pct:+.1f}% in 5 days)",
                        "Has been falling this week ({momentum_5d_pct:+.1f}% in 5 days)"),
    "trend_vs_ma50_pct": ("On an upward trend over the last few months",
                          "On a downward trend over the last few months"),
    "rel_volume_1d": ("More people than usual traded it yesterday",
                      "Fewer people than usual traded it yesterday"),
    "overextension": ("", "Already moved a lot compared with its normal daily range"),
    "rsi_14": ("Hasn't climbed too far, too fast",
               "Has climbed a lot recently, so it may need a breather"),
    "thin": ("", "Very little early trading, so the move is less reliable"),
    "unexplained": ("", "No news found to explain the jump"),
}

# Short clauses for the 1-2 sentence explanation ("summary").
_UP_CLAUSE = {
    "premarket_gap_pct": "up {premarket_gap_pct:.1f}% before the open",
    "premarket_confirmation": "on much busier early trading than usual",
    "rel_strength_pct": "ahead of its sector",
    "level_signal": "breaking above yesterday's high",
    "momentum_5d_pct": "after a strong week ({momentum_5d_pct:+.1f}%)",
    "trend_vs_ma50_pct": "in a steady uptrend for months",
    "rel_volume_1d": "with busy trading yesterday",
    "rsi_14": "without looking overheated",
}
_CAUTION = {
    "premarket_gap_pct": "it is trading lower before the open",
    "premarket_confirmation": "early selling is heavy",
    "rel_strength_pct": "it is doing worse than its sector",
    "level_signal": "it is just below a recent high that may act as a ceiling",
    "momentum_5d_pct": "it fell over the past week",
    "trend_vs_ma50_pct": "its longer-term trend is down",
    "overextension": "it has already moved a lot compared with a normal day",
    "rsi_14": "it has already climbed a lot recently",
    "thin": "very few shares have traded so far, so the move is less reliable",
    "unexplained": "we found no news explaining the jump",
}
# Very short version for list rows, where the expected % is already shown.
_SHORT = {
    "premarket_confirmation": "much busier early trading than usual",
    "rel_strength_pct": "beating its sector",
    "level_signal": "breaking above yesterday's high",
    "momentum_5d_pct": "a strong week ({momentum_5d_pct:+.1f}%)",
    "trend_vs_ma50_pct": "a steady uptrend for months",
    "rel_volume_1d": "busy trading yesterday",
    "rsi_14": "no sign of overheating",
}
_SHORT_CAUTION = {
    "premarket_gap_pct": "trading lower before the open",
    "premarket_confirmation": "heavy early selling",
    "rel_strength_pct": "lagging its sector",
    "level_signal": "just below a recent high",
    "momentum_5d_pct": "fell last week",
    "trend_vs_ma50_pct": "longer-term trend is down",
    "overextension": "already a big move for this stock",
    "rsi_14": "already up a lot recently",
    "thin": "very thin early trading",
    "unexplained": "no news found for the jump",
}


def _is_true(feat: str, positive: bool, row: pd.Series) -> bool:
    """Scores are relative to other stocks, but the words must be literally true.
    E.g. a +0.6% week can score above average, but we can't call it "a strong week"."""
    v = row.get(feat, np.nan)
    gap = row.get("premarket_gap_pct", np.nan)
    rvol = row.get("premarket_rvol", np.nan)
    # Only call early trading "busier than usual" if we could actually measure it.
    busy = bool(rvol > 1.3) if isinstance(rvol, float) and math.isfinite(rvol) else False
    checks = {
        "premarket_gap_pct": (v > 0, v < 0),
        "premarket_confirmation": (gap > 0 and busy, gap < 0 and busy),
        "rel_strength_pct": (v > 0.3, v < -0.3),
        "level_signal": (v > 0, v < 0),
        "momentum_5d_pct": (v > 2.0, v < 0),
        "trend_vs_ma50_pct": (v > 0, v < 0),
        "rel_volume_1d": (v > 1, v < 1),
        "overextension": (False, v > 0),
        "rsi_14": (v < 65, v > 70),
        "thin": (False, True),
        "unexplained": (False, True),
    }
    ok_pos, ok_neg = checks.get(feat, (True, True))
    return bool(ok_pos if positive else ok_neg)


def robust_z(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Like a z-score, but using median/MAD so one crazy outlier doesn't distort it."""
    med = s.median()
    scale = (s - med).abs().median() * 1.4826
    if not np.isfinite(scale) or scale == 0:
        scale = s.std()  # most values identical (e.g. few stocks moving): fall back to std
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - med) / scale).clip(-clip, clip).fillna(0.0)


def _confirmation(df: pd.DataFrame) -> pd.Series:
    """Gap direction x how busy early trading is vs. a normal morning."""
    sign = np.sign(df["premarket_gap_pct"])
    rvol = np.log1p(df["premarket_rvol"].astype(float))
    fallback = np.log1p(df["premarket_activity"].astype(float) * 20)   # when no history for rvol
    return sign * rvol.fillna(fallback)


def score(features: pd.DataFrame, news: dict | None = None, n_reasons: int = 3,
          news_checked: set[str] | None = None) -> pd.DataFrame:
    """`news`: ticker -> sentiment.NewsAssessment. `news_checked`: tickers whose news we read."""
    df = features.copy()
    for col in ("premarket_rvol", "premarket_dollar_vol", "rel_strength_pct", "level_signal", "overextension"):
        if col not in df:
            df[col] = np.nan if col != "overextension" else 0.0
    df["premarket_confirmation"] = _confirmation(df)

    contrib = pd.DataFrame(index=df.index)
    for feat, w in WEIGHTS.items():
        contrib[feat] = robust_z(df[feat].astype(float)) * w

    news = news or {}
    df["news_score"] = pd.Series({t: a.score for t, a in news.items()}, dtype=float).reindex(df.index)
    df["news_summary"] = pd.Series({t: a.summary for t, a in news.items()}, dtype=object).reindex(df.index)
    contrib["news"] = df["news_score"].fillna(0.0) * NEWS_WEIGHT

    dv = df["premarket_dollar_vol"].astype(float)
    contrib["thin"] = np.where(dv.notna() & (dv < THIN_DOLLAR_VOL), THIN_PENALTY, 0.0)
    checked = news_checked or set()
    no_news = [t not in news or news[t].event == "none" for t in df.index]
    unexplained = df.index.isin(list(checked)) & np.array(no_news) & (df["premarket_gap_pct"] > UNEXPLAINED_GAP)
    contrib["unexplained"] = np.where(unexplained, UNEXPLAINED_PENALTY, 0.0)

    df["raw_score"] = contrib.sum(axis=1)
    df["score"] = (df["raw_score"].rank(pct=True) * 100).round(0)
    df["reasons"] = [_news_reason(df.loc[t], contrib.loc[t]) + _reasons(df.loc[t], contrib.loc[t].drop("news"), n_reasons)
                     for t in df.index]
    df["summary"] = [summary(df.loc[t], contrib.loc[t]) for t in df.index]
    df["why_short"] = [short_why(df.loc[t], contrib.loc[t]) for t in df.index]
    return df.sort_values("raw_score", ascending=False)


def _news_reason(row: pd.Series, contrib: pd.Series) -> list[dict]:
    c = float(contrib.get("news", 0.0))
    if abs(c) < NEWS_MIN_EFFECT or not isinstance(row.get("news_summary"), str):
        return []
    return [{"text": row["news_summary"], "impact": "positive" if c > 0 else "negative", "kind": "news"}]


def _reasons(row: pd.Series, contrib: pd.Series, n: int) -> list[dict]:
    """The n signals that moved this stock's score the most, as readable text."""
    out = []
    for feat in contrib.abs().sort_values(ascending=False).index:
        if len(out) >= n:
            break
        c = float(contrib[feat])
        if abs(c) < 0.05 or feat not in REASON_TEXT or not _is_true(feat, c > 0, row):
            continue
        if feat in row and not (isinstance(row[feat], (int, float)) and np.isfinite(row[feat])):
            continue
        pos, neg = REASON_TEXT[feat]
        out.append({"text": (pos if c > 0 else neg).format(**row.to_dict()),
                    "impact": "positive" if c > 0 else "negative"})
    return out


def _caution(row: pd.Series, contrib: pd.Series, table: dict) -> str | None:
    negs = [f for f in contrib.sort_values().index
            if contrib[f] < -0.15 + 1e-9 and f in table and _is_true(f, False, row)]
    return table[negs[0]] if negs else None


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


def summary(row: pd.Series, contrib: pd.Series) -> str:
    """One or two plain-English sentences explaining why the stock scores well."""
    c = contrib.drop("news", errors="ignore")
    pos = [f for f in c.sort_values(ascending=False).index
           if f in _UP_CLAUSE and c[f] > 0.05 and _is_true(f, True, row)]
    clauses = [_UP_CLAUSE[f].format(**row.to_dict()) for f in pos[:3]]
    if not clauses:
        text = "Several small positive signs, none stand-out on its own."
    else:
        first = clauses[0][0].upper() + clauses[0][1:]
        text = first + ("." if len(clauses) == 1 else ", " + " and ".join(clauses[1:]) + ".")
    caution = _caution(row, c, _CAUTION)
    if caution:
        text += f" Caution: {caution}."
    return _with_news(row, contrib, text, lead_full=True)


def short_why(row: pd.Series, contrib: pd.Series) -> str:
    c = contrib.drop("news", errors="ignore")
    pos = [f for f in c.sort_values(ascending=False).index
           if f in _SHORT and c[f] > 0.05 and _is_true(f, True, row)]
    parts = [_SHORT[f].format(**row.to_dict()) for f in pos[:2]]
    if parts:
        text = " and ".join(parts)
        text = text[0].upper() + text[1:] + "."
    elif row.get("premarket_gap_pct", 0) > 0:
        text = "Rising before the open."
    else:
        text = "Several small positive signs."
    caution = _caution(row, c, _SHORT_CAUTION)
    if caution:
        text += f" Caution: {caution}."
    return _with_news(row, contrib, text, lead_full=False)
