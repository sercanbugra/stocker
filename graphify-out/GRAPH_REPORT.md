# Graph Report - .  (2026-05-14)

## Corpus Check
- 35 files · ~363,032 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 441 nodes · 718 edges · 27 communities (20 shown, 7 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 58 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Flask App & Admin API|Flask App & Admin API]]
- [[_COMMUNITY_Auth System Tests|Auth System Tests]]
- [[_COMMUNITY_Core App Architecture|Core App Architecture]]
- [[_COMMUNITY_Frontend UI Modals|Frontend UI Modals]]
- [[_COMMUNITY_Chart & Pattern Detection|Chart & Pattern Detection]]
- [[_COMMUNITY_AI Premium Features UI|AI Premium Features UI]]
- [[_COMMUNITY_AI & Peers API Endpoints|AI & Peers API Endpoints]]
- [[_COMMUNITY_Market Data Pipeline|Market Data Pipeline]]
- [[_COMMUNITY_App Infrastructure & Frontend|App Infrastructure & Frontend]]
- [[_COMMUNITY_Remarkables & Screener|Remarkables & Screener]]
- [[_COMMUNITY_ML Prediction Engine|ML Prediction Engine]]
- [[_COMMUNITY_News & Sentiment|News & Sentiment]]
- [[_COMMUNITY_Watchlist Trends Charts|Watchlist Trends Charts]]
- [[_COMMUNITY_Brand Assets & PWA Icons|Brand Assets & PWA Icons]]
- [[_COMMUNITY_Industry DB Builder|Industry DB Builder]]
- [[_COMMUNITY_Google OAuth Tests|Google OAuth Tests]]
- [[_COMMUNITY_Service Worker PWA|Service Worker PWA]]
- [[_COMMUNITY_Market Symbol Lists|Market Symbol Lists]]
- [[_COMMUNITY_External Platform Badges|External Platform Badges]]
- [[_COMMUNITY_GulTechs Brand Logos|GulTechs Brand Logos]]
- [[_COMMUNITY_Google OAuth Integration|Google OAuth Integration]]
- [[_COMMUNITY_Android TWA|Android TWA]]
- [[_COMMUNITY_lxml Dependency|lxml Dependency]]
- [[_COMMUNITY_BeautifulSoup4 Dependency|BeautifulSoup4 Dependency]]
- [[_COMMUNITY_python-dotenv Dependency|python-dotenv Dependency]]
- [[_COMMUNITY_Antler Co-founder Pitch|Antler Co-founder Pitch]]
- [[_COMMUNITY_Feature Screenshots Modal|Feature Screenshots Modal]]

## God Nodes (most connected - your core abstractions)
1. `app.py Backend` - 30 edges
2. `_load_users()` - 23 edges
3. `post_json()` - 19 edges
4. `register()` - 19 edges
5. `index.html — Main App Template` - 17 edges
6. `_save_users()` - 15 edges
7. `_get_current_user_email()` - 14 edges
8. `detect_pattern()` - 14 edges
9. `run_prediction()` - 13 edges
10. `TestRegister` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Three-Model ML Ensemble (CV)` --semantically_similar_to--> `ML Ensemble (XGBoost + LightGBM + ExtraTreesRegressor)`  [INFERRED] [semantically similar]
  cv_antler.md → CLAUDE.md
- `Fly.io Deployment` --semantically_similar_to--> `Fly.io and Render Deployment (CV)`  [INFERRED] [semantically similar]
  CLAUDE.md → cv_antler.md
- `Stripe Subscription Revenue Model (CV)` --semantically_similar_to--> `Stripe Payment Integration`  [INFERRED] [semantically similar]
  cv_antler.md → CLAUDE.md
- `Claude AI Integration (CV)` --semantically_similar_to--> `Claude AI Features (trade thesis, earnings, portfolio)`  [INFERRED] [semantically similar]
  cv_antler.md → CLAUDE.md
- `PWA and Android TWA (CV)` --semantically_similar_to--> `Android TWA`  [INFERRED] [semantically similar]
  cv_antler.md → CLAUDE.md

## Hyperedges (group relationships)
- **ML Prediction Pipeline** — claude_train_prediction_models, claude_build_feature_frame, claude_forecast_tree_recursive, claude_ml_ensemble [EXTRACTED 0.95]
- **Cache Management System** — claude_load_cached_response, claude_save_cached_response, claude_upgrade_cached_payload, claude_per_symbol_cache [EXTRACTED 0.95]
- **Auth + Subscription + Access Control** — claude_subscription_tiers, claude_google_oauth, claude_stripe_payments, claude_users_json [EXTRACTED 0.95]
- **Promo Code Sync Across Welcome, Upgrade, and Profile Modals** — index_welcome_modal, index_upgrade_modal, index_profile_modal, index_fn_wirepromobox, index_promo_box [EXTRACTED 0.95]
- **Prediction trigger -> Pro/Premium AI panels (Thesis, Peers, Portfolio, Earnings)** — index_fn_predictbtn, index_fn_loadtradethesis, index_fn_loadpeers, index_fn_loadportfolioadvice, index_fn_loadearningssummary, tier_rationale_gating [EXTRACTED 0.95]
- **Static symbol files used by backend to seed Remarkables scanner** — sp500_symbols_txt, ftse100_symbols_txt, bist_symbols_txt, index_remarkables_card [INFERRED 0.85]
- **AI Features Gated Behind Pro Tier (Trade Thesis, Portfolio Advisor, Peer Comparison)** — ai_trade_thesis_feature, ai_portfolio_adviser_feature, peer_comparison_feature, pro_tier [EXTRACTED 1.00]
- **AI Features Powered by Claude Haiku (Trade Thesis, Portfolio Advisor, Earnings Summarizer)** — ai_trade_thesis_feature, ai_portfolio_adviser_feature, earning_summarizer_feature, claude_haiku_ai [INFERRED 0.95]
- **Features Exclusive to Premium Tier (Earnings Summarizer)** — earning_summarizer_feature, premium_tier [EXTRACTED 1.00]
- **ASTS Symbol Demonstrated Across All Four Feature Panels** — symbol_asts, peer_comparison_feature, ai_trade_thesis_feature, ai_portfolio_adviser_feature, earning_summarizer_feature [INFERRED 0.85]
- **Stocker Brand Logo Variants** — stocker_logo_sm_logo, stocker_logo_webp_logo, stocker_logo_png_logo, stocker_logo_small_webp_logo [INFERRED 0.95]
- **Stocker PWA and Touch Icons** — icon512_icon, icon192_icon, favicon192_icon, apple_touch_icon_icon [INFERRED 0.95]
- **GulTechs Brand Logo Variants** — logo_jpg_brand, logo_png_brand, logo_small_webp_brand [INFERRED 0.95]
- **External Platform Listing Badges** — badge_shipit_badge, badge_producthunt_badge, badge_backlinklog_badge [INFERRED 0.95]

## Communities (27 total, 7 thin omitted)

### Community 0 - "Flask App & Admin API"
Cohesion: 0.06
Nodes (64): api_admin_delete_user(), api_admin_send_mail(), api_admin_send_watchlist_mails(), api_admin_set_tier(), api_admin_users(), api_anon_usage(), api_cancel_subscription(), api_delete_account() (+56 more)

### Community 1 - "Auth System Tests"
Cohesion: 0.05
Nodes (23): app(), login(), post_json(), Comprehensive tests for the Stocker auth system:   - Email registration  (/regis, Password must never be stored as plaintext., Each test registers a fresh user via the client fixture., Login with UPPER@example.com should succeed for a lower-registered account., Logging out when not logged in must not crash. (+15 more)

### Community 2 - "Core App Architecture"
Cohesion: 0.05
Nodes (47): app.py Backend, data/bist_symbols.txt, build_feature_frame, _build_pattern_chart_from_series, _build_rsi_chart_from_series, Claude AI Features (trade thesis, earnings, portfolio), _compute_remarkables, detect_pattern (+39 more)

### Community 3 - "Frontend UI Modals"
Cohesion: 0.06
Nodes (40): Delete Account Page, Admin Broadcast Mail Modal, AI Analysis Terminal Overlay (Loading Spinner), Auth Modal (Sign In / Register), Cookie Consent Banner & Preferences Panel, JS: loadAdminMembers(), JS: applyChartTheme(layout, traceCount) — Plotly theme adapter, JS: applyTheme(theme) (+32 more)

### Community 4 - "Chart & Pattern Detection"
Cohesion: 0.08
Nodes (33): _build_pattern_chart_48h_fallback_from_series(), _build_pattern_chart_48h_from_df(), _build_pattern_chart_from_series(), _build_rsi_chart_from_series(), _detect_cup_handle(), _detect_double_bottom(), _detect_double_top(), _detect_flags_pennants() (+25 more)

### Community 5 - "AI Premium Features UI"
Cohesion: 0.11
Nodes (25): Entry Strategy Section (AI Portfolio Advisor), Exit Strategy Section (AI Portfolio Advisor), AI Portfolio Advisor Feature, Position Size Section (AI Portfolio Advisor), PRO Tier Badge (AI Portfolio Advisor), Risk Profile Section (AI Portfolio Advisor), Bear Case Section (AI Trade Thesis), Bull Case Section (AI Trade Thesis) (+17 more)

### Community 6 - "AI & Peers API Endpoints"
Cohesion: 0.1
Nodes (24): api_earnings_summary(), api_peers(), api_portfolio_advisor(), api_trade_thesis(), _avg_sentiment(), fetch_peers(), fetch_sp500_stocks(), _fetch_ticker_fundamentals() (+16 more)

### Community 7 - "Market Data Pipeline"
Cohesion: 0.1
Nodes (21): _compute_full_market(), _dividend_refresh_worker(), fetch_bist_symbols(), _fetch_dividend_for_symbols(), _fetch_dividend_stocks_live(), _fetch_undervalued_for_symbols(), _fetch_undervalued_stocks_live(), _full_market_cache_is_today() (+13 more)

### Community 8 - "App Infrastructure & Frontend"
Cohesion: 0.1
Nodes (20): Admin Console Frontend, currencySymbol JS Function, Fly.io Deployment, Hero Carousel Frontend, templates/index.html, Market Ticker Bar Frontend, Render Deployment, Stocker Flask App (+12 more)

### Community 9 - "Remarkables & Screener"
Cohesion: 0.16
Nodes (20): _compute_remarkables(), _compute_remarkables_from_local_cache(), _compute_risk_trending_for_market(), _default_remarkables_payload(), _extract_close_from_batch(), _extract_field_from_batch(), fetch_nasdaq_symbols(), get_remarkables() (+12 more)

### Community 10 - "ML Prediction Engine"
Cohesion: 0.22
Nodes (10): build_feature_frame(), compute_rsi(), _forecast_tree_recursive(), prepare_feature_windows(), Compute RSI on a price series., Generate a richer feature set from closing prices only., Prepare windowed feature matrices and scalers for the models., Recursive multi-step forecast using engineered features and scaled targets. (+2 more)

### Community 11 - "News & Sentiment"
Cohesion: 0.22
Nodes (10): _collect_watchlist_news_for_user(), _company_search_query(), fetch_news(), _google_news_rss(), _parse_yf_news_item(), Return [{symbol, items:[{title,link,publisher}]}] from last 10 days., Parse a single yfinance news item regardless of old vs new API shape., Fetch news from Google News RSS for a free-text query. Returns parsed items. (+2 more)

### Community 12 - "Watchlist Trends Charts"
Cohesion: 0.25
Nodes (8): JS: buildLayout(rows) — dynamic Plotly layout builder, JS: buildTraces(sym) — builds Plotly traces for price/RSI/MACD, JS: calcEMA(arr, period) — Exponential Moving Average, JS: calcMACD(closes) — MACD indicator, JS: calcRSI(closes, period) — RSI indicator, JS: loadData(tf) — fetches /api/watchlist-trends, JS: renderAll() — renders all chart cards via Plotly.react, Watchlist Trends Page

### Community 13 - "Brand Assets & PWA Icons"
Cohesion: 0.5
Nodes (8): Stocker Apple Touch Icon, Stocker Favicon 192px, Stocker PWA Icon 192px, Stocker PWA Icon 512px, Stocker Logo (PNG), Stocker Logo Small (PNG), Stocker Logo Small (WebP), Stocker Logo (WebP)

### Community 14 - "Industry DB Builder"
Cohesion: 0.43
Nodes (6): build(), _fetch_meta(), _get_bist(), _get_ftse100(), _get_sp500(), Return a metadata dict for one symbol. Never raises.

### Community 15 - "Google OAuth Tests"
Cohesion: 0.29
Nodes (4): If GOOGLE_OAUTH_CLIENT_ID/SECRET are empty, google_bp must be None., Simulate a Google OAuth callback: monkeypatch both user email and tier         s, Without a Google token or session, _get_current_user_email returns None., TestGoogleOAuth

### Community 16 - "Service Worker PWA"
Cohesion: 0.5
Nodes (3): clone, PRECACHE, url

### Community 17 - "Market Symbol Lists"
Cohesion: 1.0
Nodes (3): BIST Symbols Static File (.IS suffix, ~500 symbols), FTSE 100 Symbols Static File (.L suffix), S&P 500 Symbols Static Fallback File

### Community 18 - "External Platform Badges"
Cohesion: 1.0
Nodes (3): BacklinkLog Listed Badge, Product Hunt Find Us Badge, Shipit Featured Badge

### Community 19 - "GulTechs Brand Logos"
Cohesion: 1.0
Nodes (3): GulTechs Brand Logo (JPG), GulTechs Brand Logo (PNG), GulTechs Brand Logo Small (WebP)

## Knowledge Gaps
- **143 isolated node(s):** `Return a metadata dict for one symbol. Never raises.`, `Best-effort conversion of Stripe (or similar) objects to plain dicts.`, `Map a Stripe price_id to our tier, with fallbacks.`, `Send a welcome email in a background thread; never raises.`, `Send a subscription-ended email; never raises.` (+138 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `app.py Backend` connect `Core App Architecture` to `App Infrastructure & Frontend`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `Stocker Flask App` connect `App Infrastructure & Frontend` to `Core App Architecture`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `index.html — Main App Template` (e.g. with `Privacy Policy Page` and `Delete Account Page`) actually correct?**
  _`index.html — Main App Template` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Return a metadata dict for one symbol. Never raises.`, `Best-effort conversion of Stripe (or similar) objects to plain dicts.`, `Map a Stripe price_id to our tier, with fallbacks.` to the rest of the system?**
  _143 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Flask App & Admin API` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Auth System Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Core App Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._