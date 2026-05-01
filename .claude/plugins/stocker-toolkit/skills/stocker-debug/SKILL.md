---
name: stocker-debug
description: Use this skill when the user reports a bug in Stocker, a broken route, ML prediction failures, yfinance errors, quota not counting correctly, Stripe webhook issues, OAuth problems, or anything that requires diagnosing app.py or the frontend template. Also activates for "why is X broken", "users getting errors", "prediction not working".
version: 1.0.0
---

# Stocker Debug Guide

Fast triage for the most common failure modes in the Stocker app.

## Architecture reminder (relevant to most bugs)

- **Single-file backend**: all logic is in `app.py` (~5520 lines). No separate modules.
- **Single template**: `templates/index.html` (~6400 lines). All frontend JS lives here.
- **Data source chain**: yfinance primary → Stooq fallback for OHLCV. yfinance also for news, analyst data, fundamentals.
- **ML minimum**: `validate_stock_data` requires 60 trading days. Stocks with less history hard-fail.
- **Quota**: Only non-cached predictions consume the daily limit. Cache hits are free.

## Triage by symptom

### `/predict` returns an error

1. Check `validate_stock_data` — symbol may have < 60 days of history.
2. Check yfinance rate limits — `fetch_with_retry` logs rate-limit detection. If yfinance is down, Stooq fallback kicks in for history only (not for fundamentals).
3. Check the ML pipeline — `train_prediction_models` requires enough variance in the feature frame. Flat-price stocks (e.g., ETFs pegged to a fixed value) can produce degenerate models.
4. Check quota — user may have hit their daily limit. Look at `data/users.json` for `daily_count` vs. tier limit.

### Quota not resetting / users locked out

Quota resets daily based on UTC date stored in `users.json` (`last_reset_date`). If the server clock drifts or the date comparison uses local time instead of UTC, resets silently fail.

```python
# The correct comparison in app.py:
from datetime import datetime, timezone
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
```

Verify the stored `last_reset_date` format matches this.

### Google OAuth broken

- Callback URL must match exactly what's registered in Google Cloud Console.
- `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET` env vars must be set.
- Flask-Dance requires `FLASK_SECRET_KEY` to be stable across restarts — on Fly.io this must be set as a secret, not defaulted.
- Symptom: redirect loop or "OAuth error" → check Fly secrets: `fly secrets list -a stocker-2xbjqq`.

### Stripe webhook returning 400

- `STRIPE_WEBHOOK_SECRET` must match the signing secret for the **live** endpoint, not the CLI test secret.
- The webhook handler reads raw request body before any JSON parsing — if middleware touches the body first, signature verification fails.
- Check Stripe Dashboard → Webhooks → recent deliveries for the actual error payload.

### Market ticker shows stale/wrong prices

- Ticker has a 5-minute in-memory cache (`_TICKER_CACHE`). A fresh deploy resets it.
- Forex symbols with `"invert": True` return `1/rate` — verify the symbol definition in `MARKET_TICKER_DEFS`.
- `.IS` suffix → TRY, `.L` suffix → GBP, default → USD in `currencySymbol()` (frontend JS).

### Admin features not showing

Admin emails are hard-coded in `ADMIN_EMAILS` set in `app.py`. If a user's email isn't in that set, they won't see the admin console even if their tier is set to "admin".

### AI features (trade thesis / earnings / portfolio) failing

These use direct HTTP to `api.anthropic.com/v1/messages` — no `anthropic` SDK. Check:
1. `ANTHROPIC_API_KEY` is set: `fly secrets list -a stocker-2xbjqq | grep ANTHROPIC`
2. The request body matches the Messages API format (model, max_tokens, messages array).
3. Tier check: trade thesis requires Pro+, earnings/portfolio require Premium+.

## Reading Fly.io logs

```bash
fly logs -a stocker-2xbjqq              # live tail
fly logs -a stocker-2xbjqq --since 1h  # last hour
```

Python tracebacks appear here. Gunicorn worker crashes show as `[ERROR] Worker with pid ... was terminated`.

## Inspecting live data files

```bash
fly ssh console -a stocker-2xbjqq
cat /data/users.json | python3 -m json.tool | head -40
ls -lh /data/cache/
```

## Minimum data threshold edge cases

`validate_stock_data` fails with < 60 trading days. This affects:
- Newly listed stocks (IPOs < 3 months old)
- Stocks with extended trading halts
- Some BIST symbols with thin history on yfinance

The error message surfaces to the user as a friendly "not enough data" response — not a 500.

## Frontend JS debug

All frontend logic is inline in `templates/index.html`. Key patterns:
- `currencySymbol(sym)` — detects market from symbol suffix
- `applyTheme()` / `applyChartTheme()` — detect `data-theme="sand"` for light theme overrides
- Plotly charts are built server-side and returned as JSON in the `/predict` response; the frontend calls `Plotly.react()` to render them
- Ticker bar updates every 5 minutes via `setInterval` polling `/api/ticker`
