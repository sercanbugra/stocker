# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stocker** — a Flask web app for stock prediction and analysis. Single-file backend (`app.py`, ~5520 lines) with one main Jinja2 template (`templates/index.html`, ~6400 lines).

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server
flask run

# Or with gunicorn (matches production)
gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 120 --preload
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

- **Fly.io**: `fly deploy` — configured in `fly.toml` (app: `stocker-2xbjqq`, region: `ams`, port 8080). `auto_stop_machines = 'suspend'` (faster cold starts than `stop`), `min_machines_running = 1`. Gunicorn uses `--preload` to avoid slow-startup false positives on the Fly.io deployment health check. Deploy token via `fly tokens create deploy -a stocker-2xbjqq`, stored as `FLY_API_TOKEN` GitHub secret.
- **Render**: `render.yaml` — auto-deploys from main branch
- **CI**: `.github/workflows/fly-deploy.yml` — pushes to `main` trigger Fly.io deploy via `flyctl deploy --remote-only`

## Architecture

All logic lives in `app.py`. Key sections:

- **Data fetching** (`fetch_with_retry`, `fetch_stooq_history`): yfinance primary, Stooq fallback for historical OHLCV data. Rate-limit errors are detected and retried.
- **Analyst insights** (`fetch_analyst_insights`): Yahoo Finance QuoteSummary API with crumb authentication; falls back to yfinance `upgrades_downgrades`.
- **Remarkables** (`_compute_remarkables`, `get_remarkables`): Scans S&P 500 + NASDAQ symbols in batch to find notable movers. Results cached keyed by `REMARKABLES_RULE_VERSION` + date — bump the version constant to force a cache invalidation on next deploy. A background thread refreshes the cache without blocking requests.
- **Market data — LSE & BIST** (`get_full_market_data("lse"|"bist")`): Returns cached data and triggers background refresh — never blocks the home route. Cache keyed by `MARKET_RULE_VERSION`; bump to force invalidation. BIST symbol source priority: isyatirim.com.tr JSON API → Wikipedia BIST 100 → hardcoded ~500-symbol fallback list. FTSE 100 symbol source priority: local `data/ftse100_symbols.txt` → Wikipedia → hardcoded fallback.
- **Per-symbol cache** (`load_cached_response`, `save_cached_response`): JSON files in `cache/<SYMBOL>.json`, 12-hour TTL. `_upgrade_cached_payload` migrates old cache shapes on load. Cache hits do **not** consume daily quota.
- **ML prediction** (`train_prediction_models`, `run_prediction`): Trains XGBoost, ExtraTreesRegressor, and RandomForestRegressor ensemble on engineered features (`build_feature_frame`). Forecasts are done recursively via `_forecast_tree_recursive`. Minimum 60 trading days of data required (`validate_stock_data`).
- **Pattern detection** (`detect_pattern` and helpers): Pure-Python technical pattern detectors (head-and-shoulders, double top/bottom, triangles, flags, cup-and-handle, etc.) operating on closing price arrays.
- **Chart building** (`_build_pattern_chart_from_series`, `_build_rsi_chart_from_series`): Returns Plotly JSON consumed by the frontend. RSI chart uses a 45-day window (for warm-up), displaying ~22 days.
- **Watchlist** (`watchlist_api`): Per-user JSON files at `watchlists/<sanitized_email>.json`; requires Pro tier. Save button transforms to "Watchlisted" state after saving.
- **News + sentiment** (`fetch_news`, `_sentiment_score`): Pulls news from yfinance, scores titles/summaries with VADER.
- **AI features** (Claude Haiku via HTTP `requests`): Trade thesis (Pro), earnings summary (Premium), portfolio advisor (Premium). No `anthropic` SDK — uses direct HTTP calls to `api.anthropic.com/v1/messages`.
- **Payments** (Stripe): Checkout sessions, webhook handler, billing portal. Tiers: free (3/day), pro (unlimited + watchlist + AI), premium (pro + earnings + portfolio).
- **Auth**: Google OAuth 2.0 (Flask-Dance) + email/password (bcrypt). Both stored in `data/users.json`.
- **Undervalued stocks** (`fetch_top_undervalued_stocks`): Scans curated candidate lists for S&P 500, LSE, and BIST; cached daily in `undervalued_stocks.json`.
- **Market ticker** (`/api/ticker`, `MARKET_TICKER_DEFS`): 11 instruments (S&P 500, Nasdaq, FTSE 100, BIST 100, Gold, Bitcoin, Crude Oil, USD/EUR, USD/TRY, USD/GBP, GBP/TRY) fetched via `yf.Tickers.fast_info`. 5-minute in-memory cache (`_TICKER_CACHE`). Forex symbols with `"invert": True` return `1/rate`.
- **Admin broadcast email** (`_send_broadcast_email`, `/api/admin/send-mail`): Sends welcome-template-style HTML email to users filtered by tier. Dispatched in a background thread to avoid blocking.
- **Admin tier override** (`/api/admin/set-tier`): Lets admins change any user's tier without Stripe charge. Sets `subscription_status = "active"` and `admin_granted = True` for non-free tiers.
- **Currency display** (`currencySymbol(sym)` in JS): Detects market from symbol suffix — `.IS` → ₺ (TRY), `.L` → £ (GBP), default → $ (USD). Applied in Prediction Results and Peer Comparison.

## Data & Cache Files

| Path | Purpose |
|---|---|
| `data/users.json` | User accounts, tiers, Stripe IDs, daily usage |
| `data/watchlists/<email>.json` | Per-user watchlist symbols |
| `data/sp500_symbols.txt` | Static S&P 500 symbol list — loaded first by `fetch_sp500_stocks()` to avoid Wikipedia scraping |
| `data/ftse100_symbols.txt` | Static FTSE 100 symbol list (~98 symbols, `.L` suffix) |
| `data/bist_symbols.txt` | Static BIST symbol list (~359 symbols, `.IS` suffix) |
| `data/cache/industry_db.json` | Symbol metadata (sector, name, market) built by `build_industry_db.py` |
| `cache/<SYMBOL>.json` | Per-symbol prediction cache (12h TTL) |
| `cache/remarkables_nasdaq.json` | Daily remarkables cache (S&P 500 + NASDAQ) |
| `cache/remarkables_lse.json` | Daily LSE remarkables cache |
| `cache/remarkables_bist.json` | Daily BIST remarkables cache |
| `cache/undervalued_stocks.json` | Daily undervalued stocks cache |

## Subscription Tiers

| Tier | Daily analyses | Watchlist | Trade thesis | Peer comparison | Earnings summary | Portfolio advisor |
|---|---|---|---|---|---|---|
| Anonymous | 1 | — | — | — | — | — |
| Free | 3 | — | — | — | — | — |
| Pro | Unlimited | Yes | Yes | Yes | — | — |
| Premium | Unlimited | Yes | Yes | Yes | Yes | Yes |
| Admin | Unlimited | All | All | All | All | All |

Admin emails are hard-coded in `ADMIN_EMAILS` set in `app.py`. Pricing: Pro £3.99/mo, Premium £5.99/mo.

## API Routes

| Route | Method | Tier | Description |
|---|---|---|---|
| `/` | GET | — | Home — loads S&P 500 list + remarkables |
| `/predict` | POST | Free+ | Run full analysis; returns JSON. Cache hits skip quota. |
| `/predict_excel` | POST | Free+ | Same as `/predict` but returns `.xlsx` |
| `/health` | GET | — | Health check for Fly.io (`{"status":"ok"}`) |
| `/api/ticker` | GET | — | Live market ticker data (11 instruments, 5-min cache) |
| `/api/remarkables` | GET | — | Get/refresh remarkables cache (`?refresh=1`) |
| `/api/watchlist` | GET/POST | Pro+ | Read or update user's watchlist |
| `/api/trade-thesis` | POST | Pro+ | Claude Haiku bull/bear/verdict |
| `/api/earnings-summary` | POST | Premium+ | Claude earnings call summary |
| `/api/portfolio-advisor` | POST | Premium+ | Claude portfolio advice |
| `/api/peers` | POST | Pro+ | Peer comparison (fundamentals, returns) |
| `/api/me` | GET | — | Current user profile + usage |
| `/api/profile` | POST | — | Update nickname, avatar, theme, markets |
| `/create-checkout-session` | POST | — | Stripe checkout (requires login; returns `{url}` or `{upgraded: true}`) |
| `/webhook/stripe` | POST | — | Stripe webhook (subscription events) |
| `/billing-portal` | GET | Pro+ | Stripe billing portal redirect |
| `/login/email` | POST | — | Email + password sign-in |
| `/register/email` | POST | — | Email + password registration |
| `/verify-email` | GET | — | Email verification |
| `/logout` | GET | — | Clear session |
| `/api/admin/users` | GET | Admin | List all users |
| `/api/admin/delete-user` | POST | Admin | Delete a user |
| `/api/admin/set-tier` | POST | Admin | Override a user's tier without Stripe charge |
| `/api/admin/send-mail` | POST | Admin | Broadcast HTML email to users filtered by tier |

## Frontend — Welcome Modal

A full-screen onboarding popup (`#welcome-modal`) is shown on page load for non-logged-in users (`{% if not user_email %}`). Dismissed via `sessionStorage` key `stocker_welcome_seen` — shows once per browser session.

- **Continue with Google** → `/login/google?prompt=consent&select_account=1`
- **Continue with Email** → closes welcome modal, opens `#auth-modal` on the email view
- **Continue as Guest** → closes modal, sets session flag (1 analysis/day)
- **Pro / Premium plan cards** → clickable; stores `stocker_pending_tier` in `sessionStorage`, opens auth modal. After login the page reloads as a logged-in user and the jQuery ready block auto-calls `/create-checkout-session` with the stored tier, then redirects to Stripe Checkout.

Plan cards displayed: Guest (1/day), Logged-in User (£0, 3/day), Pro (£3.99/mo), Premium (£5.99/mo) — matching the Plans & Pricing section in the upgrade modal.

## Frontend — Market Ticker Bar

Fixed bottom bar (`#mkt-ticker`, `position:fixed; bottom:0; z-index:900`) displaying 11 instruments in a grid. Updates every 5 minutes via `/api/ticker`. Features:

- Scrolling news headlines from yfinance (clickable, open in new tab); hover slows animation to 15% speed via Web Animations API `playbackRate`
- Hide/show toggle persisted in `localStorage` (`stocker_ticker_hidden`)
- Colors use CSS variables (`--parliament-blue-deep`, `--panel-text`, `--accent`, etc.) so all themes apply automatically

## Frontend — Themes

Four themes selectable in the Profile modal. Stored in `localStorage` (`stocker_theme`) and also saved server-side via `/api/profile`.

| Key | Name | Character |
|---|---|---|
| `""` | Ocean Blue | Dark navy — default |
| `"beige"` | Warm Beige | Dark amber |
| `"crimson"` | Dark Crimson | Dark red |
| `"sand"` | Light Sand | Light beige, **black text** |

Light Sand (`[data-theme="sand"]`) has an extended override block that fixes hardcoded dark-theme colors: semantic greens → `#14532d`, reds → `#7f1d1d`, yield badges → dark orange `#92400e`, FCF yield badges → dark blue `#1e3a8a`, ticker/symbol links → `var(--panel-text)` (near-black). Plotly chart grid lines switch from `rgba(255,255,255,0.1)` to `rgba(0,0,0,0.1)` for the light background. Both `applyTheme` and `applyChartTheme` detect `data-theme="sand"` and use dark grid colors.

## Frontend — Admin Console

Accessible in the Profile modal for admin-tier users. Features:

- **Members list** (`#admin-members-list`): shows all users with email + tier dropdown. Changing the dropdown triggers a `confirm()` dialog then calls `/api/admin/set-tier`. Tier color-coded: admin (red), premium (gold), pro (blue), free (muted).
- **Send Mail button**: opens `#admin-mail-modal` (z-index 1500). Compose form: tier filter pills (Admin / Premium / Pro / Free, all pre-selected), Subject, Header tab, Content tab. Recipient count shown live. Send calls `/api/admin/send-mail`; email dispatched in background thread. Success auto-closes modal after 2.2s.

## Known Pitfalls

- **Home page blocking**: `fetch_sp500_stocks()` falls through to Wikipedia if `data/sp500_symbols.txt` is missing. Always keep this file in the repo.
- **Remarkables cache invalidation**: Changing `REMARKABLES_RULE_VERSION` forces all clients to discard S&P 500/NASDAQ remarkables and recompute. Changing `MARKET_RULE_VERSION` does the same for LSE and BIST caches.
- **BIST symbol fetch**: Primary source is isyatirim.com.tr JSON API (~500 symbols). If it fails, falls back to Wikipedia BIST 100, then to `data/bist_symbols.txt`, then to a hardcoded list.
- **Minimum data threshold**: `validate_stock_data` requires 60 trading days minimum. New/small stocks with less than 60 days of history will error.
- **Daily quota**: Only fresh (non-cached) predictions consume the daily limit. Cache hits are free for all tiers.
- **Fly.io volume**: User data (`users.json`, watchlists, cache) lives on a persistent volume at `/data`. The `data/` folder in the repo is only for static assets bundled into the Docker image.
- **Fly.io slow startup**: Heavy ML imports (numpy, pandas, sklearn, xgboost) take 15–30s. `--preload` in gunicorn CMD ensures the port is bound only after all imports complete, preventing false-positive deployment health-check failures.
- **Admin tier override vs Stripe**: When an admin sets a user's tier via `/api/admin/set-tier`, `admin_granted: true` is written to `users.json`. If that user later subscribes via Stripe, the webhook will overwrite the tier with the Stripe-derived value.
- **Mobile layout**: Results section uses Bootstrap order classes — predictions panel (`order-1 order-lg-2`) appears above the chart on mobile, below on desktop.
