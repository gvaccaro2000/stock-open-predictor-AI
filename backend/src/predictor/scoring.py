"""Stage 4 — scoring: combine features into one score with reasons.

v1 is a transparent, rule-based model:
  1. Standardise each feature across all stocks (how unusual is this value today?)
  2. Multiply by a weight (how much we believe in it) and add up
  3. Convert to a 0-100 rank so the app can show "signal strength"

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


def score(features: pd.DataFrame, n_reasons: int = 3) -> pd.DataFrame:
    df = features.copy()
    # Gap in the same direction as heavy premarket volume = "confirmed" gap.
    df["premarket_confirmation"] = (
        np.sign(df["premarket_gap_pct"]) * np.log1p(df["premarket_activity"] * 100)
    )

    contrib = pd.DataFrame(index=df.index)
    for feat, w in WEIGHTS.items():
        contrib[feat] = robust_z(df[feat].astype(float)) * w

    df["raw_score"] = contrib.sum(axis=1)
    df["score"] = (df["raw_score"].rank(pct=True) * 100).round(0)
    df["reasons"] = [
        _reasons(df.loc[t], contrib.loc[t], n_reasons) for t in df.index
    ]
    df["summary"] = [summary(df.loc[t], contrib.loc[t]) for t in df.index]
    df["why_short"] = [short_why(df.loc[t], contrib.loc[t]) for t in df.index]
    return df.sort_values("raw_score", ascending=False)


def _reasons(row: pd.Series, contrib: pd.Series, n: int) -> list[dict]:
    """The n features that moved this stock's score the most, as readable text."""
    out = []
    for feat in contrib.abs().sort_values(ascending=False).index:
        if len(out) >= n:
            break
        c = float(contrib[feat])
        if abs(c) < 0.05 or not np.isfinite(row.get(feat, np.nan)):
            continue
        # Only call volume "busier"/"quieter" if it really is above/below 1x.
        if feat == "rel_volume_1d" and (row[feat] > 1) != (c > 0):
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
    pos = [f for f in contrib.sort_values(ascending=False).index
           if contrib[f] > 0.05 and np.isfinite(row.get(f, np.nan))]
    if "rel_volume_1d" in pos and not row["rel_volume_1d"] > 1:
        pos.remove("rel_volume_1d")
    if "premarket_gap_pct" in pos and not row["premarket_gap_pct"] > 0:
        pos.remove("premarket_gap_pct")
    clauses = [_UP_CLAUSE[f].format(**row.to_dict()) for f in pos[:3]]
    if not clauses:
        text = "Several small positive signs, none stand-out on its own."
    else:
        first = clauses[0][0].upper() + clauses[0][1:]
        if len(clauses) == 1:
            text = first + "."
        else:
            text = first + ", " + " and ".join(clauses[1:]).replace(" and on", " on") + "."
    neg = contrib.sort_values().index[0]
    if contrib[neg] < -0.15 and neg in _CAUTION:
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
    pos = [f for f in contrib.sort_values(ascending=False).index
           if f in _SHORT and contrib[f] > 0.05 and np.isfinite(row.get(f, np.nan))]
    if "rel_volume_1d" in pos and not row["rel_volume_1d"] > 1:
        pos.remove("rel_volume_1d")
    parts = [_SHORT[f].format(**row.to_dict()) for f in pos[:2]]
    if parts:
        text = " and ".join(parts)
        text = text[0].upper() + text[1:] + "."
    elif row.get("premarket_gap_pct", 0) > 0:
        text = "Rising before the open."
    else:
        text = "Several small positive signs."
    neg = contrib.sort_values().index[0]
    if contrib[neg] < -0.15 and neg in _SHORT_CAUTION:
        text += f" Caution: {_SHORT_CAUTION[neg]}."
    return text
