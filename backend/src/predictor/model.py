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
    # the pre-market move
    "gap_pct", "gap_atr", "gap_sq", "rel_gap_sector", "idio_gap", "above_prev_high", "dist_high_20d",
    # recent price action
    "ret_1d", "ret_5d", "ret_20d", "prev_intraday", "close_loc_1d", "up_days_5", "dist_52w_high",
    # overnight vs daytime behaviour ("tug of war")
    "on_avg_20", "id_avg_20", "on_avg_60", "id_avg_60",
    # trend, momentum, volume, volatility, market sensitivity
    "trend_ma50", "trend_ma200", "ma50_vs_ma200", "rsi_14", "log_rel_volume_1d", "log_atr_pct", "beta_60",
    # events and conditions (as interactions, because day-wide effects cancel out in a ranking)
    "is_earnings", "earn_x_gap", "gap_x_vix", "gap_x_monday", "gap_x_opex", "beta_x_macro",
]
GAP_GROUP = {"gap_pct", "gap_atr", "gap_sq", "idio_gap"}
CLIP_Z = 4.0
TARGET_CLIP = 8.0           # winsorise daily open->close excess returns at +-8%
MIN_PRICE, MIN_DOLLAR_VOL = 3.0, 5e6
RIDGE_LAMBDA = 0.05         # shrinkage, as a fraction of the number of rows
ROUND_TRIP_COST = 0.10      # %, rough cost of buying and selling (spread + fees)
VOL_CUT_QUANTILE = 0.80     # "calm only": skip the most volatile 20% of stocks each day

# Ways to turn predictions into a top-10. All are tested; training keeps the best.
STRATEGIES = {
    "raw": "Highest expected move after the open",
    "calm": "Highest expected move, skipping the most volatile 20% of stocks",
    "risk_adj": "Highest expected move relative to the stock's normal daily swing",
    "calm_risk_adj": "Risk-adjusted, among the calmer 80% of stocks",
}


def rank_scores(pred: pd.Series, log_atr_pct: pd.Series, strategy: str) -> pd.Series:
    """Ranking score for one day's cross-section; NaN = excluded."""
    atr_pct = np.exp(log_atr_pct.astype(float)).fillna(np.exp(log_atr_pct.astype(float)).median())
    score = pred / atr_pct if strategy in ("risk_adj", "calm_risk_adj") else pred.copy()
    if strategy in ("calm", "calm_risk_adj"):
        score = score.where(atr_pct <= atr_pct.quantile(VOL_CUT_QUANTILE))
    return score


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


def _third_friday(d: pd.Timestamp) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def panel_features(P: dict[str, pd.DataFrame], sector_etf: dict[str, str] | None = None,
                   etf_P: dict[str, pd.DataFrame] | None = None, ctx: dict | None = None) -> dict[str, pd.DataFrame]:
    """Feature panels (dates x tickers). Row t uses the open of day t and data up to t-1.

    ctx (optional): {"vix": Series of VIX closes by date,
                     "earnings": DataFrame (dates x tickers) of 1 where results came out
                                 between the previous close and this open,
                     "macro_days": set of dates with a major US data release / Fed decision}
    """
    ctx = ctx or {}
    O, H, L, C, V = (P[k] for k in "OHLCV")
    idx = C.index
    C1, H1, L1 = C.shift(1), H.shift(1), L.shift(1)
    tr = np.fmax(np.fmax(H - L, (H - C.shift(1)).abs()), (L - C.shift(1)).abs())
    atr = tr.rolling(14, min_periods=10).mean().shift(1)
    gap_pct = (O / C1 - 1) * 100
    gap_atr = (O - C1) / atr
    ma50 = C.rolling(50, min_periods=40).mean().shift(1)
    ma200 = C.rolling(200, min_periods=150).mean().shift(1)
    vol_avg = V.shift(2).rolling(20, min_periods=15).mean()
    overnight = (O / C.shift(1) - 1) * 100
    daytime = (C / O - 1) * 100
    rng = (H1 - L1).where(lambda x: x > 0)

    # Market (SPY) for beta and the stock-specific gap.
    spy_c = spy_gap = None
    if etf_P is not None and "SPY" in etf_P.get("C", pd.DataFrame()).columns:
        spy_c = etf_P["C"]["SPY"].reindex(idx)
        spy_gap = ((etf_P["O"]["SPY"] / etf_P["C"]["SPY"].shift(1) - 1) * 100).reindex(idx)
    if spy_c is not None:
        r, rs = C.pct_change(fill_method=None), spy_c.pct_change(fill_method=None)
        cov = r.mul(rs, axis=0).rolling(60, min_periods=40).mean() - r.rolling(60, min_periods=40).mean().mul(rs.rolling(60, min_periods=40).mean(), axis=0)
        var = rs.rolling(60, min_periods=40).var(ddof=0)
        beta = cov.div(var, axis=0).shift(1).clip(-1, 4)
        idio_gap = gap_pct - beta.mul(spy_gap, axis=0)
    else:
        beta = gap_pct * np.nan
        idio_gap = gap_pct * np.nan

    dates = pd.Series(idx, index=idx)
    monday = dates.map(lambda d: float(d.weekday() == 0))
    opex = dates.map(lambda d: float(_third_friday(d)))
    macro = dates.map(lambda d: float(d.normalize() in ctx.get("macro_days", set()) if hasattr(d, "normalize") else 0.0))
    vix = ctx.get("vix")
    vix_prev = vix.reindex(idx, method="ffill").shift(1) if vix is not None and len(vix) else pd.Series(np.nan, index=idx)
    vix_term = np.log((vix_prev / 20).clip(lower=0.3)).fillna(0.0)
    earn = ctx.get("earnings")
    is_earn = earn.reindex(index=idx, columns=C.columns).fillna(0.0) if earn is not None else (gap_pct * 0).fillna(0.0)

    f = {
        "gap_pct": gap_pct,
        "gap_atr": gap_atr,
        "gap_sq": (gap_atr * gap_atr.abs()).clip(-9, 9),
        "idio_gap": idio_gap,
        "above_prev_high": (O > H1).astype(float).where(O.notna()),
        "dist_high_20d": ((H.rolling(20, min_periods=15).max().shift(1) - O) / atr).clip(-5, 10),
        "ret_1d": (C1 / C.shift(2) - 1) * 100,
        "ret_5d": (C1 / C.shift(6) - 1) * 100,
        "ret_20d": (C1 / C.shift(21) - 1) * 100,
        "prev_intraday": daytime.shift(1),
        "close_loc_1d": (C1 - L1) / rng,
        "up_days_5": (C.diff() > 0).astype(float).where(C.diff().notna()).rolling(5, min_periods=5).mean().shift(1) - 0.5,
        "dist_52w_high": (C1 / H.rolling(252, min_periods=150).max().shift(1) - 1) * 100,
        "on_avg_20": overnight.rolling(20, min_periods=15).mean().shift(1),
        "id_avg_20": daytime.rolling(20, min_periods=15).mean().shift(1),
        "on_avg_60": overnight.rolling(60, min_periods=40).mean().shift(1),
        "id_avg_60": daytime.rolling(60, min_periods=40).mean().shift(1),
        "trend_ma50": (C1 / ma50 - 1) * 100,
        "trend_ma200": (C1 / ma200 - 1) * 100,
        "ma50_vs_ma200": (ma50 / ma200 - 1) * 100,
        "rsi_14": _rsi(C).shift(1),
        "log_rel_volume_1d": np.log((V.shift(1) / vol_avg).clip(lower=0.05)),
        "log_atr_pct": np.log((atr / C1 * 100).clip(lower=0.05)),
        "beta_60": beta,
        "is_earnings": is_earn.where(O.notna()),
        "earn_x_gap": (is_earn * gap_atr).clip(-10, 10),
        "gap_x_vix": gap_atr.mul(vix_term, axis=0).clip(-10, 10),
        "gap_x_monday": gap_atr.mul(monday, axis=0).clip(-10, 10),
        "gap_x_opex": gap_atr.mul(opex, axis=0).clip(-10, 10),
        "beta_x_macro": beta.mul(macro, axis=0),
        "_prev_close": C1,
        "_atr": atr,
        "_dollar_vol": (C * V).rolling(20, min_periods=15).mean().shift(1),
    }
    # Gap vs the stock's sector ETF (or the S&P 500 if the sector is unknown).
    if etf_P is not None and "O" in etf_P and len(etf_P["O"].columns):
        egap = ((etf_P["O"] / etf_P["C"].shift(1) - 1) * 100).reindex(idx)
        ref = pd.DataFrame(index=idx, columns=C.columns, dtype=float)
        for t in C.columns:
            e = (sector_etf or {}).get(t)
            col = e if e in egap.columns else ("SPY" if "SPY" in egap.columns else None)
            if col:
                ref[t] = egap[col]
        f["rel_gap_sector"] = gap_pct - ref
    else:
        f["rel_gap_sector"] = gap_pct * np.nan
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
LGBM_PARAMS = {
    "objective": "huber", "alpha": 3.0, "learning_rate": 0.05, "num_leaves": 15,
    "min_data_in_leaf": 3000, "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 10.0, "verbose": -1, "num_threads": 0, "seed": 7,
}
LGBM_ROUNDS = 200


@dataclass
class Model:
    """kind="ridge": linear weights on standardised features.
    kind="lgbm": gradient-boosted trees (captures combinations like "big gap *on earnings day*").
    Both give per-feature contributions that add up to the prediction (for explanations)."""
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    thresholds: dict[str, float]
    meta: dict
    kind: str = "ridge"
    booster: object = None

    def z(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X.reindex(columns=self.features).to_numpy(dtype=float) - self.mean) / self.std
        Z = np.clip(Z, -CLIP_Z, CLIP_Z)
        return np.nan_to_num(Z, nan=0.0)

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        return X.reindex(columns=self.features).to_numpy(dtype=float)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.kind == "lgbm":
            c = self.booster.predict(self._raw(X), pred_contrib=True)
            return pd.DataFrame(c[:, :-1], index=X.index, columns=self.features)   # last column = baseline
        return pd.DataFrame(self.z(X) * self.weights, index=X.index, columns=self.features)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.kind == "lgbm":
            return pd.Series(self.booster.predict(self._raw(X)), index=X.index)
        return pd.Series(self.z(X) @ self.weights, index=X.index)

    def to_json(self) -> dict:
        d = {"kind": self.kind, "features": self.features, "mean": self.mean.round(6).tolist(),
             "std": self.std.round(6).tolist(), "weights": self.weights.round(6).tolist(),
             "thresholds": self.thresholds, "meta": self.meta}
        if self.kind == "lgbm":
            d["booster"] = self.booster.model_to_string()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        booster = None
        if d.get("kind") == "lgbm":
            import lightgbm as lgb
            booster = lgb.Booster(model_str=d["booster"])
        return cls(d["features"], np.array(d["mean"]), np.array(d["std"]), np.array(d["weights"]),
                   d.get("thresholds", {}), d.get("meta", {}), d.get("kind", "ridge"), booster)


def _thresholds(pred: np.ndarray) -> dict[str, float]:
    q = np.quantile(pred, [0.5, 0.8, 0.95, 0.99]) if len(pred) else [0, 0, 0, 0]
    return {"q50": float(q[0]), "q80": float(q[1]), "q95": float(q[2]), "q99": float(q[3])}


def fit(train: pd.DataFrame, lam: float = RIDGE_LAMBDA, kind: str = "ridge") -> Model:
    X = train.reindex(columns=FEATURES).to_numpy(dtype=float)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    yv = train["y_ex"].clip(-TARGET_CLIP, TARGET_CLIP).to_numpy(dtype=float)
    n, k = X.shape
    if kind == "lgbm":
        import lightgbm as lgb
        params = dict(LGBM_PARAMS, min_data_in_leaf=max(50, min(LGBM_PARAMS["min_data_in_leaf"], n // 200)))
        booster = lgb.train(params, lgb.Dataset(X, yv, feature_name=list(FEATURES), free_raw_data=True),
                            num_boost_round=LGBM_ROUNDS)
        pred = booster.predict(X)
        return Model(list(FEATURES), mean, std, np.zeros(k), _thresholds(pred), {"n_rows": int(n)}, "lgbm", booster)
    Z = np.nan_to_num(np.clip((X - mean) / std, -CLIP_Z, CLIP_Z), nan=0.0)
    w = np.linalg.solve(Z.T @ Z + lam * n * np.eye(k), Z.T @ yv)
    return Model(list(FEATURES), mean, std, w, _thresholds(Z @ w), {"n_rows": int(n)})


# ----------------------------------------------------------------------------- backtest
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return math.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def walk_forward(table: pd.DataFrame, market_oc: pd.Series | None = None, min_train_days: int = 252,
                 top_n: int = 10, kind: str = "ridge") -> dict:
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
        mdl = fit(tr, kind=kind)
        te = te.assign(pred=mdl.predict(te).to_numpy())
        for d, g in te.groupby(level="date"):
            if len(g) < 3 * top_n:
                continue
            base = {"date": pd.Timestamp(d).date().isoformat(),
                    "all_mean": float(g["y"].mean()),
                    "market": float(market_oc.get(d, np.nan)) if market_oc is not None else math.nan,
                    "ic": _spearman(g["pred"].to_numpy(), g["y_ex"].to_numpy()), "n": int(len(g))}
            for strat in STRATEGIES:
                sc = rank_scores(g["pred"], g["log_atr_pct"], strat).dropna().sort_values(ascending=False)
                top, bot = g.loc[sc.index[:top_n]], g.loc[sc.index[-top_n:]]
                rows.append({**base, "strategy": strat, "top_mean": float(top["y"].mean()),
                             "top_hit": float((top["y"] > 0).mean()), "bottom_mean": float(bot["y"].mean())})
    if not rows:
        return {"error": "backtest produced no test days", "days": 0}
    s = pd.DataFrame(rows)
    per = {k: summarise(g.reset_index(drop=True), top_n) for k, g in s.groupby("strategy")}
    best = choose_strategy(per)
    out = per[best] | {"strategy": best, "strategy_label": STRATEGIES[best],
                       "strategies": {k: {"label": STRATEGIES[k], "overall": v["overall"], "last_12_months": v["last_12_months"]}
                                      for k, v in per.items()}}
    daily = s[s["strategy"] == best].drop(columns="strategy").to_dict("records")
    return out | {"daily": daily[-260:]}


def choose_strategy(per: dict) -> str:
    """Best growth after costs over the last 12 months, among strategies that are
    also positive after costs over the whole test. Falls back to "raw"."""
    ok = {k: v for k, v in per.items() if v["overall"]["top_avg_after_costs_pct"] > 0}
    if not ok:
        return "raw"
    return max(ok, key=lambda k: ok[k]["last_12_months"]["top_growth_of_100"])


def summarise(s: pd.DataFrame, top_n: int) -> dict:
    def block(x: pd.DataFrame) -> dict:
        net = x["top_mean"] - ROUND_TRIP_COST
        return {
            "days": int(len(x)),
            "top_avg_pct": round(float(x["top_mean"].mean()), 3),
            "top_avg_after_costs_pct": round(float(net.mean()), 3),
            "top_avg_after_double_costs_pct": round(float((x["top_mean"] - 2 * ROUND_TRIP_COST).mean()), 3),
            "all_avg_pct": round(float(x["all_mean"].mean()), 3),
            "market_avg_pct": round(float(x["market"].mean()), 3) if x["market"].notna().any() else None,
            "bottom_avg_pct": round(float(x["bottom_mean"].mean()), 3),
            "top_hit_rate": round(float(x["top_hit"].mean()), 3),
            "days_top_beats_all": round(float((x["top_mean"] > x["all_mean"]).mean()), 3),
            "avg_ic": round(float(x["ic"].mean()), 4),
            "top_growth_of_100": round(float(100 * np.prod(1 + net / 100)), 1),
            "top_volatility_pct": round(float(x["top_mean"].std()), 3),
            "sharpe": round(float(net.mean() / net.std() * math.sqrt(252)), 2) if net.std() > 0 else None,
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


MODEL_KINDS = {"ridge": "Linear (ridge regression)", "lgbm": "Gradient-boosted trees (LightGBM)"}


def train_all(daily: dict[str, pd.DataFrame], etf_daily: dict[str, pd.DataFrame],
              sector_etf: dict[str, str], ctx: dict | None = None, kinds: tuple = ("ridge", "lgbm"),
              backtest: bool = True) -> tuple[Model, dict]:
    """Features -> walk-forward backtest of each model kind -> keep the kind that ranks
    best out of sample (average IC) -> pick its strategy -> fit the final model on all data."""
    P = to_panel(daily)
    E = to_panel(etf_daily) if etf_daily else None
    feats = panel_features(P, sector_etf, E, ctx)
    y, y_ex = target_excess(P)
    table = long_table(feats, y, y_ex)
    market = None
    if E is not None and "SPY" in E["C"].columns:
        market = ((E["C"]["SPY"] / E["O"]["SPY"] - 1) * 100)
    results = {}
    if backtest:
        for k in kinds:
            results[k] = walk_forward(table, market, kind=k)
    ok = {k: v for k, v in results.items() if v.get("overall")}
    chosen = max(ok, key=lambda k: ok[k]["overall"]["avg_ic"]) if ok else kinds[0]
    bt = dict(results.get(chosen, {"days": 0}))
    if ok:
        bt["model_kind"] = chosen
        bt["models"] = {k: {"label": MODEL_KINDS.get(k, k), "avg_ic": v["overall"]["avg_ic"],
                            "strategy": v["strategy"], "overall": v["overall"], "last_12_months": v["last_12_months"]}
                        for k, v in ok.items()}
    final = fit(table, kind=chosen)
    final.meta["strategy"] = bt.get("strategy", "raw")
    final.meta.update({
        "kind": chosen,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_from": str(table.index.get_level_values("date").min().date()),
        "trained_through": str(table.index.get_level_values("date").max().date()),
        "n_stocks": int(table.index.get_level_values("ticker").nunique()),
        "target": "open-to-close return minus the day's average stock, %",
    })
    return final, bt
