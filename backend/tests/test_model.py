"""The after-open model: features use only pre-open info, it learns a planted
pattern, and the walk-forward backtest detects it."""

import numpy as np
import pandas as pd

from predictor import model


def synthetic_market(n_stocks=60, n_days=420, seed=0, fade=-0.35, momentum=0.04):
    """Prices where open->close moves depend on the gap (fade) and 5-day momentum."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    daily = {}
    for i in range(n_stocks):
        close_prev = 50 + 100 * rng.random()
        rows = []
        hist = [close_prev] * 6
        for d in days:
            gap = rng.normal(0, 1.2) + (rng.random() < 0.05) * rng.normal(0, 5)       # % overnight
            o = hist[-1] * (1 + gap / 100)
            mom5 = (hist[-1] / hist[-6] - 1) * 100
            oc = fade * gap + momentum * mom5 + rng.normal(0, 1.5)                     # % open->close
            c = o * (1 + oc / 100)
            h, l = max(o, c) * (1 + abs(rng.normal(0, .004))), min(o, c) * (1 - abs(rng.normal(0, .004)))
            rows.append((o, h, l, c, float(rng.integers(5e5, 2e6))))
            hist.append(c)
        daily[f"S{i}"] = pd.DataFrame(rows, index=days, columns=["Open", "High", "Low", "Close", "Volume"])
    return daily


def test_features_use_only_pre_open_information():
    daily = synthetic_market(n_stocks=3, n_days=260)
    P = model.to_panel(daily)
    f1 = model.panel_features(P)
    # Changing today's close/high/low/volume must not change today's features.
    P2 = {k: v.copy() for k, v in P.items()}
    last = P2["C"].index[-1]
    for k in "HLCV":
        P2[k].loc[last] = P2[k].loc[last] * 1.5
    f2 = model.panel_features(P2)
    for name in model.FEATURES:
        if name == "rel_gap_sector":
            continue
        a, b = f1[name].loc[last], f2[name].loc[last]
        assert np.allclose(a.fillna(0), b.fillna(0)), name


def test_learns_planted_pattern_and_backtest_finds_edge():
    daily = synthetic_market()
    P = model.to_panel(daily)
    feats = model.panel_features(P)
    y, y_ex = model.target_excess(P)
    table = model.long_table(feats, y, y_ex)
    m = model.fit(table)
    w = dict(zip(m.features, m.weights))
    assert w["gap_pct"] + w["gap_atr"] + w["gap_sq"] < 0      # learned: big gaps fade
    assert w["ret_5d"] > 0                                       # learned: momentum continues
    bt = model.walk_forward(table, min_train_days=200)
    o = bt["overall"]
    assert o["days"] > 100
    assert o["top_avg_pct"] > o["all_avg_pct"] > o["bottom_avg_pct"]
    assert o["avg_ic"] > 0.1


def test_no_edge_on_pure_noise():
    daily = synthetic_market(fade=0.0, momentum=0.0, seed=4)
    P = model.to_panel(daily)
    feats = model.panel_features(P)
    y, y_ex = model.target_excess(P)
    bt = model.walk_forward(model.long_table(feats, y, y_ex), min_train_days=200)
    assert abs(bt["overall"]["avg_ic"]) < 0.05                 # nothing to find -> ~0


def test_model_roundtrip(tmp_path):
    daily = synthetic_market(n_stocks=20, n_days=300)
    P = model.to_panel(daily)
    feats = model.panel_features(P)
    y, y_ex = model.target_excess(P)
    t = model.long_table(feats, y, y_ex)
    m = model.fit(t)
    model.save(m, tmp_path / "m.json")
    m2 = model.load(tmp_path / "m.json")
    assert np.allclose(m.predict(t.head(50)), m2.predict(t.head(50)), atol=1e-4)
    c = m2.contributions(t.head(5))
    assert np.allclose(c.sum(axis=1), m2.predict(t.head(5)))


def test_strategies_compared_and_one_chosen():
    daily = synthetic_market()
    P = model.to_panel(daily)
    feats = model.panel_features(P)
    y, y_ex = model.target_excess(P)
    bt = model.walk_forward(model.long_table(feats, y, y_ex), min_train_days=200)
    assert set(bt["strategies"]) == set(model.STRATEGIES)
    assert bt["strategy"] in model.STRATEGIES
    chosen = bt["strategies"][bt["strategy"]]
    assert chosen["overall"]["top_avg_after_costs_pct"] > 0
    assert all("sharpe" in v["overall"] for v in bt["strategies"].values())


def test_rank_scores_calm_excludes_most_volatile():
    import pandas as pd
    pred = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6], index=list("abcde"))
    vol = pd.Series(np.log([10.0, 1.0, 1.0, 1.0, 1.0]), index=pred.index)   # "a" is very volatile
    assert pd.isna(model.rank_scores(pred, vol, "calm")["a"])
    ra = model.rank_scores(pred, vol, "risk_adj")
    assert ra.idxmax() == "b"                       # a's high pred is divided by its big swing
