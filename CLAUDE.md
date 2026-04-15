# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stocker** — a Flask web app for stock prediction and analysis. Single-file backend (`app.py`, ~5200 lines) with one main Jinja2 template (`templates/index.html`, ~5000 lines).

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server
flask run

# Or with gunicorn (matches production)
gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 120
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `FLASK_SECRET_KEY` | Session signing (defaults to `dev-secret-key` in dev) |
| `GOOGLE_OAUTH_CLIENT_ID` | Google OAuth (optional; skipped if unset) |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth (optional; skipped if unset) |
| `ANTHROPIC_API_KEY` | Claude Haiku for trade thesis / earnings / portfolio features |
| `STRIPE_SECRET_KEY` | Stripe payments |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook verification |
| `STRIPE_PRO_PRICE_ID` | Stripe Pro tier price ID |
| `STRIPE_PREMIUM_PRICE_ID` | Stripe Premium tier price ID |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | Email (Privateemail.com) |
| `SITE_BASE_URL` | Canonical domain (default: `https://stocker.gultechs.net`) |
| `PERSISTENT_DATA_DIR` | Data directory on Fly.io volume (default: `data/`) |

## Deployment

- **Fly.io**: `fly deploy` — configured in `fly.toml` (app: `stocker-2xbjqq`, region: `ams`, port 8080). Deploy token via `fly tokens create deploy -a stocker-2xbjqq`, stored as `FLY_API_TOKEN` GitHub secret.
- **Render**: `render.yaml` — auto-deploys from main branch
- **CI**: `.github/workflows/fly-deploy.yml` — pushes to `main` trigger Fly.io deploy via `flyctl deploy --remote-only`

## Architecture

All logic lives in `app.py`. Key sections:

- **Data fetching** (`fetch_with_retry`, `fetch_stooq_history`): yfinance primary, Stooq fallback for historical OHLCV data. Rate-limit errors are detected and retried.
- **Analyst insights** (`fetch_analyst_insights`): Yahoo Finance QuoteSummary API with crumb authentication; falls back to yfinance `upgrades_downgrades`.
- **Remarkables** (`_compute_remarkables`, `get_remarkables`): Scans S&P 500 + NASDAQ symbols in batch to find notable movers. Results cached keyed by `REMARKABLES_RULE_VERSION` + date — bump the version constant to force a cache invalidation on next deploy. A background thread refreshes the cache without blocking requests.
- **Per-symbol cache** (`load_cached_response`, `save_cached_response`): JSON files in `cache/<SYMBOL>.json`, 12-hour TTL. `_upgrade_cached_payload` migrates old cache shapes on load. Cache hits do **not** consume daily quota.
- **ML prediction** (`train_prediction_models`, `run_prediction`): Trains XGBoost, ExtraTreesRegressor, and RandomForestRegressor ensemble on engineered features (`build_feature_frame`). Forecasts are done recursively via `_forecast_tree_recursive`. Minimum 60 trading days of data required (`validate_stock_data`).
- **Pattern detection** (`detect_pattern` and helpers): Pure-Python technical pattern detectors (head-and-shoulders, double top/bottom, triangles, flags, cup-and-handle, etc.) operating on closing price arrays.
- **Chart building** (`_build_pattern_chart_from_series`, `_build_rsi_chart_from_series`): Returns Plotly JSON consumed by the frontend.
- **Watchlist** (`watchlist_api`): Per-user JSON files at `watchlists/<sanitized_email>.json`; requires Pro tier. Save button transforms to "Watchlisted" state after saving.
- **News + sentiment** (`fetch_news`, `_sentiment_score`): Pulls news from yfinance, scores titles/summaries with VADER.
- **AI features** (Claude Haiku via HTTP): Trade thesis (Pro), earnings summary (Premium), portfolio advisor (Premium).
- **Payments** (Stripe): Checkout sessions, webhook handler, billing portal. Tiers: free (3/day), pro (unlimited + watchlist + AI), premium (pro + earnings + portfolio).
- **Auth**: Google OAuth 2.0 (Flask-Dance) + email/password (bcrypt). Both stored in `data/users.json`.
- **Market data**: `get_full_market_data("lse"|"bist")` returns cached data and triggers background refresh — never blocks the home route.

## Data & Cache Files

| Path | Purpose |
|---|---|
| `data/users.json` | User accounts, tiers, Stripe IDs, daily usage |
| `data/watchlists/<email>.json` | Per-user watchlist symbols |
| `data/sp500_symbols.txt` | Static S&P 500 symbol list — loaded first by `fetch_sp500_stocks()` to avoid Wikipedia scraping on every home page load |
| `data/cache/industry_db.json` | Symbol metadata (sector, name, market) built by `build_industry_db.py` |
| `cache/<SYMBOL>.json` | Per-symbol prediction cache (12h TTL) |
| `cache/remarkables_nasdaq.json` | Daily remarkables cache |

## Subscription Tiers

| Tier | Daily analyses | Watchlist | Trade thesis | Peer comparison | Earnings summary | Portfolio advisor |
|---|---|---|---|---|---|---|
| Anonymous | 1 | — | — | — | — | — |
| Free | 3 | — | — | — | — | — |
| Pro | Unlimited | Yes | Yes | Yes | — | — |
| Premium | Unlimited | Yes | Yes | Yes | Yes | Yes |
| Admin | Unlimited | All | All | All | All | All |

Admin emails are hard-coded in `ADMIN_EMAILS` set in `app.py`.

## API Routes

| Route | Method | Tier | Description |
|---|---|---|---|
| `/` | GET | — | Home — loads S&P 500 list + remarkables |
| `/predict` | POST | Free+ | Run full analysis; returns JSON. Cache hits skip quota. |
| `/predict_excel` | POST | Free+ | Same as `/predict` but returns `.xlsx` |
| `/api/remarkables` | GET | — | Get/refresh remarkables cache (`?refresh=1`) |
| `/api/watchlist` | GET/POST | Pro+ | Read or update user's watchlist |
| `/api/trade-thesis` | POST | Pro+ | Claude Haiku bull/bear/verdict |
| `/api/earnings-summary` | POST | Premium+ | Claude earnings call summary |
| `/api/portfolio-advisor` | POST | Premium+ | Claude portfolio advice |
| `/api/peers` | POST | Pro+ | Peer comparison (fundamentals, returns) |
| `/api/me` | GET | — | Current user profile + usage |
| `/api/profile` | POST | — | Update nickname, avatar, theme, markets |
| `/create-checkout-session` | POST | — | Stripe checkout |
| `/webhook/stripe` | POST | — | Stripe webhook (subscription events) |
| `/billing-portal` | GET | Pro+ | Stripe billing portal redirect |
| `/login/email` | POST | — | Email + password sign-in |
| `/register/email` | POST | — | Email + password registration |
| `/verify-email` | GET | — | Email verification |
| `/logout` | GET | — | Clear session |
| `/api/admin/users` | GET | Admin | List all users |
| `/api/admin/delete-user` | POST | Admin | Delete a user |

## Known Pitfalls

- **Home page blocking**: `fetch_sp500_stocks()` falls through to Wikipedia if `data/sp500_symbols.txt` is missing. Always keep this file in the repo.
- **Remarkables cache invalidation**: Changing `REMARKABLES_RULE_VERSION` forces all clients to discard cached remarkables and recompute. Do this when the data shape changes.
- **Minimum data threshold**: `validate_stock_data` requires 60 trading days minimum. New/small stocks with less than 60 days of history will error.
- **Daily quota**: Only fresh (non-cached) predictions consume the daily limit. Cache hits are free for all tiers.
- **Fly.io volume**: User data (`users.json`, watchlists, cache) lives on a persistent volume at `/data`. The `data/` folder in the repo is only for static assets bundled into the Docker image.
