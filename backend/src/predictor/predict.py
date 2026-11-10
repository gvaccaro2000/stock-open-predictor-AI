"""Daily predictions with the after-open model.

1. Live features, computed with exactly the same code as in training: we append
   "today" to the daily history with the pre-market price as today's open.
2. The model's estimate of each stock's open->close move vs the average stock.
3. Adjustments for things the history can't teach the model (yet), in % points:
     fading       falling back from its overnight peak (the Sanmina case)
     news         Claude's read of the overnight news
     thin         very little pre-market trading
     unexplained  a big jump with no news found
   These are clearly labelled as not yet validated; the daily track record will
   show whether they help.
4. Plain-English reasons from the per-feature contributions.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import model as M
from .data import Premarket

FADE_START_ATR, FADE_PCT_PER_ATR, FADE_MAX_ATR = 0.5, 0.10, 3.0
NEWS_PCT = 0.30
THIN_DOLLAR_VOL, THIN_PCT = 500_000, -0.15
UNEXPLAINED_GAP, UNEXPLAINED_PCT = 2.0, -0.10
MIN_SHOWN = 0.01    # % points; smaller contributions aren't mentioned
MIN_CAUTION = 0.05  # % points; a negative factor must matter at least this much to be a "Caution"
MIN_GAP_WORDS = 1.0 # %; smaller pre-market moves aren't described as jumps or dips


def next_session_date(now: datetime, daily_last: pd.Timestamp) -> pd.Timestamp:
    """The trading day the prediction is for (today before the close, else the next weekday)."""
    d = pd.Timestamp(now.date())
    if d <= daily_last:
        d = daily_last + pd.offsets.BDay(1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _with_today(daily: dict[str, pd.DataFrame], pre: dict[str, Premarket], day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    out = {}
    for t, df in daily.items():
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        price = pre[t].last_price if t in pre else np.nan
        df.loc[day] = [price, np.nan, np.nan, np.nan, np.nan]
        out[t] = df
    return out


def live_features(daily: dict[str, pd.DataFrame], pre: dict[str, Premarket],
                  etf_daily: dict[str, pd.DataFrame], etf_pre: dict[str, Premarket],
                  sector_etf: dict[str, str], now: datetime) -> pd.DataFrame:
    last = max(pd.Timestamp(df.index[-1]).tz_localize(None).normalize() for df in daily.values())
    day = next_session_date(now, last)
    P = M.to_panel(_with_today(daily, pre, day))
    E = M.to_panel(_with_today(etf_daily, etf_pre, day)) if etf_daily else None
    f = M.panel_features(P, sector_etf, E)
    row = pd.DataFrame({k: v.loc[day] for k, v in f.items()})
    row.index.name = "ticker"
    return row[row["gap_pct"].notna()]


# ----------------------------------------------------------------------------- wording
def _phrase(feat: str, v: float) -> str | None:
    if v is None or not np.isfinite(v):
        return None
    if feat == "rel_gap_sector":
        return ("Rising more than its sector before the open" if v > 0 else "Doing worse than its sector before the open")
    if feat == "above_prev_high":
        return "Opening above yesterday's high" if v >= 1 else None
    if feat == "dist_high_20d":
        if v <= 0:
            return "Trading at a new 20-day high"
        return "Just below its 20-day high" if v < 0.5 else f"{v:.1f} normal daily moves below its 20-day high"
    if feat == "ret_1d":
        return f"{'Rose' if v > 0 else 'Fell'} {abs(v):.1f}% yesterday"
    if feat == "ret_5d":
        return f"{'Up' if v > 0 else 'Down'} {abs(v):.1f}% over the last 5 days"
    if feat == "ret_20d":
        return f"{'Up' if v > 0 else 'Down'} {abs(v):.1f}% over the last month"
    if feat == "trend_ma50":
        return f"{abs(v):.0f}% {'above' if v > 0 else 'below'} its 50-day average"
    if feat == "trend_ma200":
        return f"{abs(v):.0f}% {'above' if v > 0 else 'below'} its 200-day average"
    if feat == "ma50_vs_ma200":
        return ("50-day average above the 200-day (long-term uptrend)" if v > 0
                else "50-day average below the 200-day (long-term downtrend)")
    if feat == "rsi_14":
        if v > 70:
            return f"Looks overbought after a strong run (RSI {v:.0f})"
        if v < 30:
            return f"Looks oversold after a drop (RSI {v:.0f})"
        return f"Neither overbought nor oversold (RSI {v:.0f})"
    if feat == "log_rel_volume_1d":
        return f"Traded {math.exp(v):.1f}x its normal volume yesterday"
    if feat == "log_atr_pct":
        return f"Normally moves about {math.exp(v):.1f}% a day"
    return None


def _gap_text(gap: float, gap_atr: float, c: float) -> str:
    size = ""
    if np.isfinite(gap_atr) and abs(gap_atr) > 2:
        size = ", a very big move for this stock"
    elif np.isfinite(gap_atr) and abs(gap_atr) > 1:
        size = ", a big move for this stock"
    head = f"{'Up' if gap > 0 else 'Down'} {abs(gap):.1f}% before the open{size}"
    if gap > 0:
        tail = "jumps like this have tended to continue after the open" if c > 0 else \
               "jumps like this have tended to fade after the open"
    else:
        tail = "dips like this have tended to bounce back after the open" if c > 0 else \
               "drops like this have tended to keep falling after the open"
    return f"{head}: {tail}"


def _gap_short(gap: float, c: float) -> str:
    if gap > 0:
        return f"Up {gap:.1f}% before the open, {'and such moves tend to continue' if c > 0 else 'and such jumps tend to fade'}"
    return f"Dipped {abs(gap):.1f}% before the open, {'and dips like this tend to bounce back' if c > 0 else 'and drops like this tend to continue'}"


def explain(row: pd.Series, contrib: pd.Series, adjustments: list[dict]) -> dict:
    """Reasons list + 1-2 sentence summary + very short 'why' for the list."""
    items = []  # (abs effect, text, short, impact, kind)
    g = float(sum(contrib.get(f, 0.0) for f in M.GAP_GROUP))
    if abs(g) >= MIN_SHOWN and np.isfinite(row.get("gap_pct", np.nan)) and abs(row["gap_pct"]) >= MIN_GAP_WORDS:
        txt = _gap_text(row["gap_pct"], row.get("gap_atr", np.nan), g)
        items.append((abs(g), txt, _gap_short(row["gap_pct"], g), "positive" if g > 0 else "negative", "model"))
    for f, c in contrib.items():
        if f in M.GAP_GROUP or abs(c) < MIN_SHOWN:
            continue
        ph = _phrase(f, row.get(f, np.nan))
        if not ph:
            continue
        sign = "historically a good sign for the rest of the day" if c > 0 else \
               "historically a bad sign for the rest of the day"
        items.append((abs(c), f"{ph}: {sign}", ph, "positive" if c > 0 else "negative", "model"))
    for a in adjustments:
        items.append((abs(a["pct"]), a["text"], a["short"], "positive" if a["pct"] > 0 else "negative", a["kind"]))
    items.sort(key=lambda x: -x[0])

    reasons = [{"text": t, "impact": imp, "kind": k} for _, t, _, imp, k in items[:6]]
    pos = [s for _, _, s, imp, _ in items if imp == "positive"]
    neg = [s for e, _, s, imp, _ in items if imp == "negative" and e >= MIN_CAUTION]
    news_pos = next((s for _, _, s, imp, k in items if k == "news" and imp == "positive"), None)

    def lower(s: str) -> str:
        return s[0].lower() + s[1:] if s and not s[:2].isupper() else s

    lead = [news_pos] if news_pos else []
    lead += [p for p in pos if p != news_pos][: 2 - len(lead)]
    short = ". ".join(x.rstrip(".") for x in lead) if lead else "Small positive signs only"
    if neg:
        short += f". Caution: {lower(neg[0])}"
    short = short.rstrip(".") + "."
    full_pos = [t for _, t, _, imp, _ in items if imp == "positive"][:3]
    summary = " ".join(x.rstrip(".") + "." for x in full_pos) or "Several small positive signs."
    if neg:
        neg_full = next(t for e, t, _, imp, _ in items if imp == "negative" and e >= MIN_CAUTION)
        summary += f" Caution: {lower(neg_full).rstrip('.')}."
    return {"reasons": reasons, "summary": summary, "why_short": short}


# ----------------------------------------------------------------------------- adjustments
def adjustments(t: str, row: pd.Series, pre: Premarket | None, news: dict, news_checked: set) -> list[dict]:
    out = []
    atr, prev = row.get("_atr", np.nan), row.get("_prev_close", np.nan)
    if pre is not None and np.isfinite(pre.peak) and np.isfinite(atr) and atr > 0 and pre.peak > prev:
        fade = (pre.peak - pre.last_price) / atr
        if fade > FADE_START_ATR:
            pct = -FADE_PCT_PER_ATR * min(fade - FADE_START_ATR, FADE_MAX_ATR)
            drop = (pre.last_price / pre.peak - 1) * 100
            out.append({"kind": "fade", "pct": pct,
                        "text": f"Already {abs(drop):.1f}% below its overnight peak of ${pre.peak:,.2f}: "
                                "early buyers are taking profits, which often continues after the open",
                        "short": f"Fading from its overnight peak ({drop:+.1f}%)"})
    a = news.get(t)
    if a is not None and abs(a.score) >= 0.05:
        out.append({"kind": "news", "pct": NEWS_PCT * a.score, "text": a.summary, "short": a.summary.rstrip(".")})
    if pre is not None and pre.volume * pre.last_price < THIN_DOLLAR_VOL:
        out.append({"kind": "thin", "pct": THIN_PCT,
                    "text": "Very little pre-market trading so far, so the early price is less reliable",
                    "short": "Very thin pre-market trading"})
    gap = row.get("gap_pct", np.nan)
    if t in news_checked and (a is None or a.event == "none") and np.isfinite(gap) and gap > UNEXPLAINED_GAP:
        out.append({"kind": "unexplained", "pct": UNEXPLAINED_PCT,
                    "text": "Big jump but we found no news explaining it", "short": "No news found for the jump"})
    return out


def strength(pred: float, thr: dict) -> str:
    if not thr:
        return "Moderate"
    if pred >= thr.get("q99", math.inf):
        return "Strong"
    if pred >= thr.get("q95", math.inf):
        return "Good"
    if pred >= thr.get("q80", math.inf):
        return "Moderate"
    return "Weak"


def score(mdl: M.Model, live: pd.DataFrame, pre: dict[str, Premarket],
          news: dict | None = None, news_checked: set | None = None) -> pd.DataFrame:
    news, news_checked = news or {}, news_checked or set()
    contrib = mdl.contributions(live)
    pred = contrib.sum(axis=1)
    rows = {}
    for t in live.index:
        r = live.loc[t]
        adj = adjustments(t, r, pre.get(t), news, news_checked)
        adj_pct = float(sum(a["pct"] for a in adj))
        e = explain(r, contrib.loc[t], adj)
        rows[t] = {
            "model_pct": float(pred[t]), "adjust_pct": adj_pct, "expected_after_open_pct": float(pred[t]) + adj_pct,
            "strength": strength(float(pred[t]) + adj_pct, mdl.thresholds),
            "adjustments": [{"kind": a["kind"], "pct": round(a["pct"], 3), "text": a["text"]} for a in adj],
            **e,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "ticker"
    strategy = mdl.meta.get("strategy", "raw")
    out["rank_score"] = M.rank_scores(out["expected_after_open_pct"], live["log_atr_pct"].reindex(out.index), strategy)
    out["excluded_volatile"] = out["rank_score"].isna()
    out["rank_score"] = out["rank_score"].fillna(-np.inf)
    return out.sort_values(["rank_score", "expected_after_open_pct"], ascending=False)
