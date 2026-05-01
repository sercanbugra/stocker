---
name: stocker-deploy
description: Use this skill when the user wants to deploy Stocker to Fly.io, run "fly deploy", push a new release, check deployment health, or asks about CI/CD for this project. Also activates for "ship it", "deploy", "push to prod", or "release" in the context of this repo.
version: 1.0.0
---

# Stocker Deploy

Deploy the Stocker Flask app to Fly.io correctly and verify it landed.

## Pre-flight checklist (always run before deploying)

1. **Uncommitted changes** — `git status`. Never deploy with untracked sensitive files (`.env`, credentials).
2. **Tests / lint** — if any exist, run them. (`grep -r "def test_" app.py` to check quickly.)
3. **requirements.txt in sync** — any new `import` in `app.py` must be in `requirements.txt`.
4. **Version constants** — if remarkables or market cache logic changed, bump `REMARKABLES_RULE_VERSION` or `MARKET_RULE_VERSION` in `app.py` so stale caches are invalidated on first request after deploy.
5. **Env vars** — confirm any new env var is set in Fly secrets (`fly secrets list -a stocker-2xbjqq`).

## Deploy command

```bash
fly deploy -a stocker-2xbjqq
```

- **Remote build** is the default and correct path — do NOT add `--local-only` unless debugging a Dockerfile issue.
- Fly uses `gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 120 --preload` (see `fly.toml`). The `--preload` flag is critical: heavy ML imports (numpy, pandas, sklearn, xgboost) take 15–30s; without it the health check fires before the port is bound.

## Post-deploy verification

Run these in order:

```bash
# 1. Health check
curl -s https://stocker.gultechs.net/health

# 2. Ticker API (5 instruments must return valid prices)
curl -s https://stocker.gultechs.net/api/ticker | python3 -m json.tool | head -30

# 3. Remarkables (should return cached data immediately)
curl -s "https://stocker.gultechs.net/api/remarkables" | python3 -m json.tool | head -10
```

Expected `/health` response: `{"status":"ok"}`. Any other response or connection error means the machine didn't start cleanly — check `fly logs -a stocker-2xbjqq`.

## Rollback

```bash
fly releases list -a stocker-2xbjqq          # find the last good version
fly deploy --image <image-ref> -a stocker-2xbjqq
```

## CI path (GitHub Actions)

Pushes to `main` auto-trigger `.github/workflows/fly-deploy.yml` via `flyctl deploy --remote-only`. Manual deploys with `fly deploy` are fine for hotfixes — both paths are equivalent.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Health check times out | `--preload` missing or ML import crash | Check `fly logs`; verify `fly.toml` CMD |
| `ModuleNotFoundError` in logs | New import not in `requirements.txt` | Add it, redeploy |
| Old cache still served after deploy | Rule version not bumped | Bump `REMARKABLES_RULE_VERSION` / `MARKET_RULE_VERSION` |
| Stripe webhook 400s | `STRIPE_WEBHOOK_SECRET` wrong after redeploy | Re-run `fly secrets set STRIPE_WEBHOOK_SECRET=...` |
| Volume data missing | Mounted path wrong in `fly.toml` | Verify `[mounts] source="stocker_data" destination="/data"` |
