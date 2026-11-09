# Stock Open Predictor

An app that, every US trading day, ranks stocks by signals historically associated with
a price rise at the market open — and tracks how often it is right.

> ⚠️ **Not financial advice.** This project is for educational and research purposes.
> Scores are statistical signals, not recommendations to buy or sell any security.

## How it works

```
 BACKEND (Python, runs daily ~8:30 ET)            iPHONE APP (Expo / React Native)
 ────────────────────────────────────             ────────────────────────────────
 1. Ingest   prices, premarket, news...    ──►    Reads docs/data/today.json
 2. Features one row of signals per ticker        Shows ranked picks + reasons
 3. Score    rule-based (v1) → ML later
 4. Publish  docs/data/today.json (GitHub Pages)
```

## Repo layout

| Folder | What it contains |
|---|---|
| `backend/` | Python pipeline: data fetching, features, scoring, report |
| `app/` | Expo iPhone/Android app |
| `docs/data/` | Published output (`today.json`), served by GitHub Pages |
| `.github/workflows/` | Scheduled daily job |

## Quick start

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m predictor.run --universe sample      # small ticker list, fast
python -m predictor.run --universe sp500       # full S&P 500
pytest
```

### App
```bash
cd app
npm install
npx expo start      # scan the QR code with the Expo Go app on your iPhone
```

## Roadmap
- [x] Phase 1 — Price-based features, rule-based score, daily JSON, Expo app
- [ ] Phase 2 — News & SEC filings analysed with the Claude API
- [ ] Phase 3 — Historical dataset, ML model, backtesting & live accuracy tracking
- [ ] Phase 4 — Supabase backend, push notifications, watchlists, App Store

## License
MIT
