"""Phase 2 tests: news parsing, Claude assessment (with a fake client) and scoring."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from predictor import features, news, report, scoring, sentiment
from predictor.data import NY, Premarket
from predictor.news import NewsItem

from test_pipeline import make_daily

NOW = datetime(2026, 11, 10, 13, 0, tzinfo=timezone.utc)


def item(title, hours_ago=2):
    return NewsItem(title, "", "Reuters", "https://example.com", NOW - timedelta(hours=hours_ago))


class FakeClaude:
    """Mimics anthropic.Anthropic().messages.create with a canned tool answer per ticker."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        prompt = kw["messages"][0]["content"]
        ticker = prompt.split("(ticker ")[1].split(")")[0]
        block = SimpleNamespace(type="tool_use", input=self.answers[ticker])
        return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=300, output_tokens=60))


def test_parse_yahoo_current_and_old_format():
    new_fmt = {"id": "1", "content": {"title": "Acme beats estimates", "summary": "Strong quarter.",
               "pubDate": "2026-11-10T11:00:00Z", "provider": {"displayName": "Reuters"},
               "canonicalUrl": {"url": "https://r.com/a"}}}
    old_fmt = {"title": "Acme wins contract", "publisher": "AP", "link": "https://ap.org/b",
               "providerPublishTime": 1794300000}
    a, b = news.parse_yahoo(new_fmt), news.parse_yahoo(old_fmt)
    assert a.title == "Acme beats estimates" and a.source == "Reuters" and a.url == "https://r.com/a"
    assert b.source == "AP" and b.published.tzinfo is not None
    assert news.parse_yahoo({"content": {"title": ""}}) is None


def test_dedupe_and_since_filter():
    items = [item("Acme beats estimates", 1), item("ACME beats estimates!", 2), item("Old story", 40)]
    kept = [i for i in news._dedupe(items) if i.published >= NOW - timedelta(hours=20)]
    assert [i.title for i in kept] == ["Acme beats estimates"]


def test_analyze_with_fake_claude():
    fake = FakeClaude({
        "ACME": {"sentiment": 0.8, "importance": "high", "event": "earnings",
                 "summary": "Profits beat expectations and it raised its forecast.",
                 "items": [{"index": 0, "sentiment": "positive"}]},
    })
    out = sentiment.analyze({"ACME": [item("Acme beats estimates")], "NONE": []},
                            {"ACME": "Acme Corp"}, client=fake, model="test-model")
    assert set(out) == {"ACME"} and len(fake.calls) == 1          # no call for stocks without news
    a = out["ACME"]
    assert a.label == "positive" and abs(a.score - 0.8) < 1e-9
    assert a.items[0]["sentiment"] == "positive"
    assert fake.calls[0]["tool_choice"]["name"] == "record_news_assessment"
    assert "Acme Corp" in fake.calls[0]["messages"][0]["content"]


def test_analyze_without_key_skips(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert sentiment.analyze({"ACME": [item("x")]}, {}) == {}


def test_news_moves_ranking_and_explanation():
    daily = {f"T{i}": make_daily(100, 0.0, seed=i) for i in range(20)}
    pre = {t: Premarket(float(df["Close"].iloc[-1]) * 1.01, 20_000, "x") for t, df in daily.items()}
    feats = features.build_features(daily, pre)
    base = scoring.score(feats)
    underdog = base.index[-1]   # worst-ranked stock on price alone

    good = sentiment.NewsAssessment(1.0, "high", "earnings", "Profits beat expectations", [])
    bad = sentiment.NewsAssessment(-1.0, "high", "legal_regulatory", "Regulators opened an investigation", [])
    leader = base.index[0]
    scored = scoring.score(feats, news={underdog: good, leader: bad})

    assert scored.index.get_loc(underdog) < base.index.get_loc(underdog)   # good news lifts it
    assert scored.index.get_loc(leader) > base.index.get_loc(leader)       # bad news drops it
    up = scored.loc[underdog]
    assert up["why_short"] == "Profits beat expectations."
    assert up["summary"].startswith("Profits beat expectations.")
    assert up["reasons"][0] == {"text": "Profits beat expectations", "impact": "positive", "kind": "news"}
    assert "Caution: regulators opened an investigation." in scored.loc[leader]["summary"]

    payload = report.build_payload(scored, datetime(2026, 11, 10, 8, 30, tzinfo=NY), top_n=20,
                                   news={underdog: good}, news_checked=20)
    assert payload["news_enabled"] and payload["model"] == "rules-v2-news"
    pick = next(p for p in payload["picks"] if p["ticker"] == underdog)
    assert pick["news"]["label"] == "positive"


def test_explanations_are_literally_true():
    import numpy as np
    rng = np.random.default_rng(3)
    daily = {f"T{i}": make_daily(100, rng.normal(0, 0.003), seed=i) for i in range(40)}
    pre = {t: Premarket(float(df["Close"].iloc[-1]) * (1 + rng.normal(0.004, 0.012)), rng.uniform(5e3, 3e5), "x")
           for t, df in daily.items()}
    sc = scoring.score(features.build_features(daily, pre))
    for t, r in sc.iterrows():
        text = r["summary"] + " " + r["why_short"]
        if "early selling" in text:
            assert r["premarket_gap_pct"] < 0, t
        if "strong week" in text:
            assert r["momentum_5d_pct"] > 1, t
        if "fell" in text:
            assert r["momentum_5d_pct"] < 0, t
        if "busy trading yesterday" in text:
            assert r["rel_volume_1d"] > 1, t


def test_falls_back_when_forced_tool_rejected():
    class Picky(FakeClaude):
        def create(self, **kw):
            if kw.get("tool_choice", {}).get("type") == "tool":
                raise Exception("tool_choice: type \"tool\" and \"any\" are not supported for this model.")
            assert kw["tools"][0]["strict"] is True
            return super().create(**kw)
    sentiment._FORCED_TOOL_OK["value"] = True
    fake = Picky({"ACME": {"sentiment": -0.5, "importance": "medium", "event": "legal_regulatory",
                           "summary": "Faces a new lawsuit.", "items": []}})
    out = sentiment.analyze({"ACME": [item("Acme sued")]}, {}, client=fake, model="claude-sonnet-5-5")
    assert out["ACME"].label == "negative"
    sentiment._FORCED_TOOL_OK["value"] = True
