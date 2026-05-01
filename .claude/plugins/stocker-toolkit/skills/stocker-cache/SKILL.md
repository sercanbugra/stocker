---
name: stocker-cache
description: Use this skill when the user mentions cache invalidation, stale data, old remarkables, "users see old results", bumping rule versions, clearing cache, or anything about the Stocker caching layer (remarkables, market data, per-symbol JSON, undervalued stocks).
version: 1.0.0
---

# Stocker Cache Management

Stocker has four distinct caching layers. Know which one you're targeting before touching anything.

## Cache layers at a glance

| Layer | Files | TTL | Invalidation |
|---|---|---|---|
| Per-symbol prediction | `cache/<SYMBOL>.json` | 12 hours | Delete file or wait |
| Remarkables (S&P/NASDAQ) | `cache/remarkables_nasdaq.json` | Daily | Bump `REMARKABLES_RULE_VERSION` |
| Market data (LSE/BIST) | `cache/remarkables_lse.json`, `cache/remarkables_bist.json` | Daily | Bump `MARKET_RULE_VERSION` |
| Undervalued stocks | `cache/undervalued_stocks.json` | Daily | Delete file or wait |

## Bumping version constants (force cache invalidation on next deploy)

Open `app.py` and find the constants near the top:

```python
REMARKABLES_RULE_VERSION = "v3"   # bump this → clears S&P 500 + NASDAQ remarkables
MARKET_RULE_VERSION = "v2"        # bump this → clears LSE + BIST remarkables
```

Increment the number (`v3` → `v4`). The cache key is built as `f"{version}_{date}"` so any mismatch triggers a full recompute on the next background refresh. The old file is not deleted — it's just ignored.

**When to bump:**
- Changed the scoring logic inside `_compute_remarkables`
- Changed which symbols are scanned
- Added/removed a field from the remarkables payload
- Suspecting corrupted cache data

**When NOT to bump:** fixing a UI bug, changing auth logic, updating Stripe — cache is unrelated.

## Deleting a specific symbol's cache

On Fly.io, exec into the running machine:

```bash
fly ssh console -a stocker-2xbjqq
rm /data/cache/AAPL.json   # replace AAPL with the symbol
```

Locally:

```bash
rm cache/AAPL.json
```

The next `/predict` call for that symbol will re-run the full ML pipeline.

## Force-refresh remarkables via API

```bash
curl -s "https://stocker.gultechs.net/api/remarkables?refresh=1"
```

This triggers a background recompute without waiting for the daily cron. The response returns the current (possibly stale) cache immediately; the new data lands within minutes.

## Cache hit vs. quota

Cache hits do **not** consume the user's daily analysis quota. Only fresh (non-cached) predictions count. This is enforced in `load_cached_response` / `save_cached_response` — the quota check happens after the cache lookup returns `None`.

## Debugging cache issues

1. SSH into Fly machine: `fly ssh console -a stocker-2xbjqq`
2. List cache files: `ls -lh /data/cache/`
3. Inspect a file: `python3 -c "import json,sys; d=json.load(open('/data/cache/AAPL.json')); print(d.get('cached_at'), d.get('rule_version', 'no version'))"`
4. Check version match: compare file's `rule_version` against constant in `app.py`

## The background thread pattern

Remarkables and market data use a "return stale, refresh in background" pattern — the home route never blocks. The background thread (`threading.Thread(target=..., daemon=True)`) runs `_compute_remarkables` and writes the new JSON file atomically. If the thread crashes, the old cache stays in place and users see stale-but-valid data.
