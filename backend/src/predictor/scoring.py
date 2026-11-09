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

REASON_TEXT = {
    "premarket_gap_pct": ("Trading up {premarket_gap_pct:+.1f}% premarket",
                          "Trading down {premarket_gap_pct:+.1f}% premarket"),
    "premarket_confirmation": ("Premarket move backed by heavy volume",
                               "Premarket weakness on heavy volume"),
    "momentum_5d_pct": ("Strong 5-day momentum ({momentum_5d_pct:+.1f}%)",
                        "Weak 5-day momentum ({momentum_5d_pct:+.1f}%)"),
    "trend_vs_ma50_pct": ("Above its 50-day average ({trend_vs_ma50_pct:+.1f}%)",
                          "Below its 50-day average ({trend_vs_ma50_pct:+.1f}%)"),
    "rel_volume_1d": ("Unusual volume yesterday ({rel_volume_1d:.1f}x normal)",
                      "Quiet trading yesterday ({rel_volume_1d:.1f}x normal)"),
    "rsi_14": ("Not overbought (RSI {rsi_14:.0f})",
               "Overbought, pullback risk (RSI {rsi_14:.0f})"),
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
    return df.sort_values("raw_score", ascending=False)


def _reasons(row: pd.Series, contrib: pd.Series, n: int) -> list[dict]:
    """The n features that moved this stock's score the most, as readable text."""
    out = []
    for feat in contrib.abs().sort_values(ascending=False).index[:n]:
        c = float(contrib[feat])
        if abs(c) < 0.05 or not np.isfinite(row.get(feat, np.nan)):
            continue
        pos, neg = REASON_TEXT[feat]
        # For a negative-weight feature, a positive contribution means a LOW value.
        out.append({
            "text": (pos if c > 0 else neg).format(**row.to_dict()),
            "impact": "positive" if c > 0 else "negative",
        })
    return out
