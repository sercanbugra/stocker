# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stocker** — a Flask web app for stock prediction and analysis. Single-file backend (`app.py`, ~2100 lines) with one Jinja2 template (`templates/index.html`).

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server
flask run

# Or with gunicorn (matches production)
gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 120
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `FLASK_SECRET_KEY` | Session signing (defaults to `dev-secret-key` in dev) |
| `GOOGLE_OAUTH_CLIENT_ID` | Google OAuth (optional; skipped if unset) |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth (optional; skipped if unset) |

## Deployment

- **Fly.io**: `fly deploy` — configured in `fly.toml` (app: `stocker-4ii2vw`, region: `ams`, port 8080)
- **Render**: `render.yaml` — auto-deploys from main branch

## Architecture

All logic lives in `app.py`. Key sections:

- **Data fetching** (`fetch_with_retry`, `fetch_stooq_history`): yfinance primary, Stooq fallback for historical OHLCV data. Rate-limit errors are detected and retried.
- **Analyst insights** (`fetch_analyst_insights`): Yahoo Finance QuoteSummary API with crumb authentication; falls back to yfinance `upgrades_downgrades`.
- **Remarkables** (`_compute_remarkables`, `get_remarkables`): Scans S&P 500 + NASDAQ symbols in batch to find notable movers. Results cached to `cache/remarkables_nasdaq.json` (keyed by date). A background thread (`_refresh_remarkables_worker`) refreshes the cache without blocking requests.
- **Per-symbol cache** (`load_cached_response`, `save_cached_response`): JSON files in `cache/<SYMBOL>.json`. `_upgrade_cached_payload` migrates old cache shapes on load.
- **ML prediction** (`train_prediction_models`, `run_prediction`): Trains XGBoost, ExtraTreesRegressor, and RandomForestRegressor ensemble on engineered features (`build_feature_frame`). Forecasts are done recursively via `_forecast_tree_recursive`.
- **Pattern detection** (`detect_pattern` and helpers): Pure-Python technical pattern detectors (head-and-shoulders, double top/bottom, triangles, flags, cup-and-handle, etc.) operating on closing price arrays.
- **Chart building** (`_build_pattern_chart_from_series`, `_build_rsi_chart_from_series`): Returns Plotly JSON consumed by the frontend.
- **Watchlist** (`watchlist_api`): Per-user JSON files stored on disk at `watchlists/<sanitized_email>.json`; requires Google OAuth session.
- **News + sentiment** (`fetch_news`, `_sentiment_score`): Pulls news from yfinance, scores titles/summaries with VADER.

### API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home — loads S&P 500 list + remarkables |
| `/predict` | POST | Run full analysis for a symbol; returns JSON |
| `/predict_excel` | POST | Same as `/predict` but returns `.xlsx` download |
| `/api/remarkables` | GET | Get/refresh remarkables cache (`?refresh=1`) |
| `/api/watchlist` | GET/POST | Read or update authenticated user's watchlist |
| `/logout` | GET | Clear OAuth session |
