"""The after-open model: which stocks tend to rise from the US open (15:30 CET) to the close?

Target
    y = close / open - 1, in %, minus the same day's average across all stocks
    ("excess" return). Removing the day's average makes the model about *which*
    stocks do better, not about whether the whole market goes up.

Features (all known before the open; the open price stands in for the pre-market
price we see live at ~9:00 New York time)
    gap_pct, gap_atr, gap_sq      size of the overnight jump, raw and vs normal range
    rel_gap_sector                jump vs its sector ETF's jump
    above_prev_high               opening above yesterday's high
    dist_high_20d                 distance below the 20-day high, in daily ranges
    ret_1d, ret_5d, ret_20d       recent returns
    trend_ma50, trend_ma200,      price vs 50/200-day averages, and 50 vs 200
    ma50_vs_ma200
    rsi_14                        overbought / oversold
    log_rel_volume_1d             yesterday's volume vs normal
    log_atr_pct                   how volatile the stock normally is

Method
    Ridge regression on standardised, clipped features. Simple, fast, hard to
    overfit, and every prediction splits into per-feature contributions, so the
    app can explain *why* a stock ranks high.

Validation
    Walk-forward: train only on data before a month, predict that month, roll on.
    The backtest therefore never sees the future.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "gap_pct", "gap_atr", "gap_sq", "rel_gap_sector", "above_prev_high", "dist_high_20d",
    "ret_1d", "ret_5d", "ret_20d", "trend_ma50", "trend_ma200", "ma50_vs_ma200",
    "rsi_14", "log_rel_volume_1d", "log_atr_pct",
]
GAP_GROUP = {"gap_pct", "gap_atr", "gap_sq"}
CLIP_Z = 4.0
TARGET_CLIP = 8.0           # winsorise daily open->close excess returns at +-8%
MIN_PRICE, MIN_DOLLAR_VOL = 3.0, 5e6
RIDGE_LAMBDA = 0.05         # shrinkage, as a fraction of the number of rows
ROUND_TRIP_COST = 0.10      # %, rough cost of buying and selling (spread + fees)


# ----------------------------------------------------------------------------- panels
def to_panel(daily: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV} -> {"O","H","L","C","V": dates x tickers}."""
    cols = {"O": "Open", "H": "High", "L": "Low", "C": "Close", "V": "Volume"}
    out = {}
    for k, c in cols.items():
        out[k] = pd.concat({t: df[c] for t, df in daily.items() if c in df}, axis=1).sort_index()
    idx = out["C"].index
    return {k: v.reindex(idx) for k, v in out.items()}


def _rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(loss != 0, 100.0)


def panel_features(P: dict[str, pd.DataFrame], sector_etf: dict[str, str] | None = None,
                   etf_P: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Feature panels (dates x tickers). Row t uses the open of day t and data up to t-1."""
    O, H, L, C, V = (P[k] for k in "OHLCV")
    C1 = C.shift(1)
    tr = np.fmax(np.fmax(H - L, (H - C.shift(1)).abs()), (L - C.shift(1)).abs())
    atr = tr.rolling(14, min_periods=10).mean().shift(1)
    gap_atr = (O - C1) / atr
    ma50 = C.rolling(50, min_periods=40).mean().shift(1)
    ma200 = C.rolling(200, min_periods=150).mean().shift(1)
    vol_avg = V.shift(2).rolling(20, min_periods=15).mean()
    f = {
        "gap_pct": (O / C1 - 1) * 100,
        "gap_atr": gap_atr,
        "gap_sq": (gap_atr * gap_atr.abs()).clip(-9, 9),
        "above_prev_high": (O > H.shift(1)).astype(float).where(O.notna()),
        "dist_high_20d": ((H.rolling(20, min_periods=15).max().shift(1) - O) / atr).clip(-5, 10),
        "ret_1d": (C1 / C.shift(2) - 1) * 100,
        "ret_5d": (C1 / C.shift(6) - 1) * 100,
        "ret_20d": (C1 / C.shift(21) - 1) * 100,
        "trend_ma50": (C1 / ma50 - 1) * 100,
        "trend_ma200": (C1 / ma200 - 1) * 100,
        "ma50_vs_ma200": (ma50 / ma200 - 1) * 100,
        "rsi_14": _rsi(C).shift(1),
        "log_rel_volume_1d": np.log((V.shift(1) / vol_avg).clip(lower=0.05)),
        "log_atr_pct": np.log((atr / C1 * 100).clip(lower=0.05)),
        "_prev_close": C1,
        "_atr": atr,
        "_dollar_vol": (C * V).rolling(20, min_periods=15).mean().shift(1),
    }
    # Gap vs the stock's sector ETF (or the S&P 500 if the sector is unknown).
    if etf_P is not None and "O" in etf_P and len(etf_P["O"].columns):
        egap = ((etf_P["O"] / etf_P["C"].shift(1) - 1) * 100).reindex(C.index)
        ref = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for t in C.columns:
            e = (sector_etf or {}).get(t)
            col = e if e in egap.columns else ("SPY" if "SPY" in egap.columns else None)
            if col:
                ref[t] = egap[col]
        f["rel_gap_sector"] = f["gap_pct"] - ref
    else:
        f["rel_gap_sector"] = f["gap_pct"] * np.nan
    return f


def target_excess(P: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open->close return (%) and the same minus the day's cross-sectional mean."""
    y = (P["C"] / P["O"] - 1) * 100
    y = y.clip(-TARGET_CLIP * 3, TARGET_CLIP * 3)
    return y, y.sub(y.mean(axis=1), axis=0)


def long_table(feats: dict[str, pd.DataFrame], y: pd.DataFrame | None = None,
               y_ex: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack panels into rows (date, ticker) and apply the liquidity filter."""
    cols = {k: v.stack(future_stack=True) for k, v in feats.items()}
    if y is not None:
        cols["y"] = y.stack(future_stack=True)
        cols["y_ex"] = y_ex.stack(future_stack=True)
    df = pd.DataFrame(cols)
    df.index.names = ["date", "ticker"]
    ok = (df["_prev_close"] >= MIN_PRICE) & (df["_dollar_vol"] >= MIN_DOLLAR_VOL) & df["gap_pct"].notna()
    if y is not None:
        ok &= df["y"].notna()
    return df[ok]


# ----------------------------------------------------------------------------- ridge
@dataclass
class Model:
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    thresholds: dict[str, float]
    meta: dict

    def z(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.features].to_numpy(dtype=float) - self.mean) / self.std
        Z = np.clip(Z, -CLIP_Z, CLIP_Z)
        return np.nan_to_num(Z, nan=0.0)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.z(X) * self.weights, index=X.index, columns=self.features)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.z(X) @ self.weights, index=X.index)

    def to_json(self) -> dict:
        return {"features": self.features, "mean": self.mean.round(6).tolist(),
                "std": self.std.round(6).tolist(), "weights": self.weights.round(6).tolist(),
                "thresholds": self.thresholds, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        return cls(d["features"], np.array(d["mean"]), np.array(d["std"]),
                   np.array(d["weights"]), d.get("thresholds", {}), d.get("meta", {}))


def fit(train: pd.DataFrame, lam: float = RIDGE_LAMBDA) -> Model:
    X = train[FEATURES].to_numpy(dtype=float)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    Z = np.nan_to_num(np.clip((X - mean) / std, -CLIP_Z, CLIP_Z), nan=0.0)
    yv = train["y_ex"].clip(-TARGET_CLIP, TARGET_CLIP).to_numpy(dtype=float)
    n, k = Z.shape
    w = np.linalg.solve(Z.T @ Z + lam * n * np.eye(k), Z.T @ yv)
    pred = Z @ w
    q = np.quantile(pred, [0.5, 0.8, 0.95, 0.99]) if n else [0, 0, 0, 0]
    thresholds = {"q50": float(q[0]), "q80": float(q[1]), "q95": float(q[2]), "q99": float(q[3])}
    return Model(list(FEATURES), mean, std, w, thresholds, {"n_rows": int(n)})


# ----------------------------------------------------------------------------- backtest
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return math.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def walk_forward(table: pd.DataFrame, market_oc: pd.Series | None = None, min_train_days: int = 252,
                 top_n: int = 10) -> dict:
    """Monthly expanding-window backtest. Returns summary + per-day series."""
    dates = table.index.get_level_values("date")
    udays = np.array(sorted(set(dates)))
    if len(udays) <= min_train_days + 5:
        return {"error": "not enough history for a backtest", "days": 0}
    months = pd.Series(udays[min_train_days:]).dt.to_period("M").unique()
    rows = []
    for m in months:
        start, end = m.start_time, m.end_time
        tr = table[dates < start]
        te = table[(dates >= start) & (dates <= end)]
        if len(tr) < 1000 or te.empty:
            continue
        mdl = fit(tr)
        te = te.assign(pred=mdl.predict(te).to_numpy())
        for d, g in te.groupby(level="date"):
            if len(g) < 3 * top_n:
                continue
            g = g.sort_values("pred", ascending=False)
            top, bot = g.head(top_n), g.tail(top_n)
            rows.append({
                "date": pd.Timestamp(d).date().isoformat(),
                "top_mean": float(top["y"].mean()),
                "top_hit": float((top["y"] > 0).mean()),
                "bottom_mean": float(bot["y"].mean()),
                "all_mean": float(g["y"].mean()),
                "market": float(market_oc.get(d, np.nan)) if market_oc is not None else math.nan,
                "ic": _spearman(g["pred"].to_numpy(), g["y_ex"].to_numpy()),
                "n": int(len(g)),
            })
    if not rows:
        return {"error": "backtest produced no test days", "days": 0}
    s = pd.DataFrame(rows)
    return summarise(s, top_n) | {"daily": rows[-260:]}


def summarise(s: pd.DataFrame, top_n: int) -> dict:
    def block(x: pd.DataFrame) -> dict:
        net = x["top_mean"] - ROUND_TRIP_COST
        return {
            "days": int(len(x)),
            "top_avg_pct": round(float(x["top_mean"].mean()), 3),
            "top_avg_after_costs_pct": round(float(net.mean()), 3),
            "all_avg_pct": round(float(x["all_mean"].mean()), 3),
            "market_avg_pct": round(float(x["market"].mean()), 3) if x["market"].notna().any() else None,
            "bottom_avg_pct": round(float(x["bottom_mean"].mean()), 3),
            "top_hit_rate": round(float(x["top_hit"].mean()), 3),
            "days_top_beats_all": round(float((x["top_mean"] > x["all_mean"]).mean()), 3),
            "avg_ic": round(float(x["ic"].mean()), 4),
            "top_growth_of_100": round(float(100 * np.prod(1 + net / 100)), 1),
            "all_growth_of_100": round(float(100 * np.prod(1 + x["all_mean"] / 100)), 1),
        }
    last = s.tail(252)
    return {"top_n": top_n, "cost_round_trip_pct": ROUND_TRIP_COST,
            "start": s["date"].iloc[0], "end": s["date"].iloc[-1],
            "overall": block(s), "last_12_months": block(last)}


# ----------------------------------------------------------------------------- io
def save(model: Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=1))


def load(path: Path) -> Model | None:
    try:
        return Model.from_json(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError):
        return None


def train_all(daily: dict[str, pd.DataFrame], etf_daily: dict[str, pd.DataFrame],
              sector_etf: dict[str, str]) -> tuple[Model, dict]:
    """Full pipeline used by train.py: features -> walk-forward backtest -> final model."""
    P = to_panel(daily)
    E = to_panel(etf_daily) if etf_daily else None
    feats = panel_features(P, sector_etf, E)
    y, y_ex = target_excess(P)
    table = long_table(feats, y, y_ex)
    market = None
    if E is not None and "SPY" in E["C"].columns:
        market = ((E["C"]["SPY"] / E["O"]["SPY"] - 1) * 100)
    bt = walk_forward(table, market)
    final = fit(table)
    final.meta.update({
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_from": str(table.index.get_level_values("date").min().date()),
        "trained_through": str(table.index.get_level_values("date").max().date()),
        "n_stocks": int(table.index.get_level_values("ticker").nunique()),
        "target": "open-to-close return minus the day's average stock, %",
    })
    return final, bt
