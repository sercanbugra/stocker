import os
from dotenv import load_dotenv
load_dotenv()
import logging
import json
import time
import re
import threading
import concurrent.futures
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import requests
from datetime import datetime, timezone
from requests.exceptions import HTTPError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from plotly.utils import PlotlyJSONEncoder
from io import StringIO

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler

# Flask imports
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_dance.contrib.google import make_google_blueprint, google
from oauthlib.oauth2 import TokenExpiredError
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import stripe

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
_USER_AGENT = "Mozilla/5.0"

# Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID", "")
STRIPE_PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID", "")

# Subscription tiers
TIER_RANK = {"free": 0, "pro": 1, "premium": 2, "admin": 99}
FREE_DAILY_LIMIT = 5
ANON_DAILY_LIMIT = 3

# Accounts that are always treated as admin regardless of users.json
ADMIN_EMAILS = {"sercan.bugra@gmail.com"}

google_bp = None
google_client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
google_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
if google_client_id and google_client_secret:
    google_bp = make_google_blueprint(
        client_id=google_client_id,
        client_secret=google_client_secret,
        scope=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"
        ]
    )
    app.register_blueprint(google_bp, url_prefix="/login")
else:
    logger.warning("Google OAuth not configured. Set GOOGLE_OAUTH_CLIENT_ID/SECRET.")

def _rating_score(rating: str) -> int:
    text = (rating or "").lower()
    if "strong buy" in text:
        return 90
    if "buy" in text:
        return 80
    if "outperform" in text:
        return 70
    if "hold" in text or "neutral" in text or "equal-weight" in text:
        return 55
    if "underperform" in text:
        return 35
    if "sell" in text:
        return 20
    return 50

def fetch_analyst_insights(symbol: str):
    empty_payload = {
        "top_analysts": [],
        "recommendations": [],
        "top_analysts_source": "Unavailable",
        "recommendations_source": "Unavailable"
    }
    def _has_rows(payload):
        return bool((payload.get("top_analysts") or []) or (payload.get("recommendations") or []))

    def _fallback_from_yfinance(sym: str):
        out = {
            "top_analysts": [],
            "recommendations": [],
            "top_analysts_source": "Unavailable",
            "recommendations_source": "Unavailable"
        }
        try:
            t = yf.Ticker(sym)
        except Exception:
            return out

        # Recent upgrades/downgrades table -> Top Analysts rows.
        try:
            ud = getattr(t, "upgrades_downgrades", None)
            if isinstance(ud, pd.DataFrame) and not ud.empty:
                df = ud.reset_index().copy()
                # Normalize expected columns from different yfinance versions.
                col_map = {c.lower(): c for c in df.columns}
                firm_col = col_map.get("firm")
                grade_col = col_map.get("tograde") or col_map.get("to grade")
                date_col = col_map.get("grade date") or col_map.get("date")
                if firm_col and grade_col:
                    rows = df.tail(8).iloc[::-1]
                    top = []
                    for _, row in rows.iterrows():
                        rating = str(row.get(grade_col) or "")
                        score = _rating_score(rating)
                        date_val = row.get(date_col) if date_col else ""
                        if hasattr(date_val, "strftime"):
                            date_val = date_val.strftime("%Y-%m-%d")
                        top.append({
                            "analyst": row.get(firm_col) or "",
                            "overallScore": score,
                            "directionScore": max(10, score - 10),
                            "priceScore": min(100, score + 15),
                            "latestRating": rating,
                            "priceTarget": "",
                            "date": str(date_val or "")
                        })
                    out["top_analysts"] = top
                    out["top_analysts_source"] = "Yahoo Finance via yfinance (upgrades_downgrades)"
        except Exception:
            pass

        # Recommendation summary/trend.
        try:
            rs = getattr(t, "recommendations_summary", None)
            if isinstance(rs, pd.DataFrame) and not rs.empty:
                recs = []
                for period, row in rs.tail(4).iloc[::-1].iterrows():
                    recs.append({
                        "period": str(period),
                        "strongBuy": int(row.get("strongBuy", 0) or 0),
                        "buy": int(row.get("buy", 0) or 0),
                        "hold": int(row.get("hold", 0) or 0),
                        "sell": int(row.get("sell", 0) or 0),
                        "strongSell": int(row.get("strongSell", 0) or 0),
                        "underperform": int(row.get("underperform", 0) or 0)
                    })
                out["recommendations"] = recs
                out["recommendations_source"] = "Yahoo Finance via yfinance (recommendations_summary)"
        except Exception:
            pass

        return out

    try:
        session_req = requests.Session()
        session_req.get("https://fc.yahoo.com", headers={"User-Agent": _USER_AGENT}, timeout=8)
        crumb_resp = session_req.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            headers={"User-Agent": _USER_AGENT},
            timeout=8
        )
        if not crumb_resp.ok or not crumb_resp.text:
            fallback = _fallback_from_yfinance(symbol)
            return fallback if _has_rows(fallback) else empty_payload
        crumb = crumb_resp.text.strip()
        url = (
            "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
            f"{symbol}?modules=upgradeDowngradeHistory,recommendationTrend&crumb={crumb}"
        )
        resp = session_req.get(url, headers={"User-Agent": _USER_AGENT}, timeout=8)
        if not resp.ok:
            fallback = _fallback_from_yfinance(symbol)
            return fallback if _has_rows(fallback) else empty_payload
        payload = resp.json().get("quoteSummary", {})
        result = (payload.get("result") or [{}])[0]

        history = (result.get("upgradeDowngradeHistory") or {}).get("history") or []
        top_analysts = []
        for item in history[:8]:
            rating = item.get("toGrade") or ""
            score = _rating_score(rating)
            epoch = item.get("epochGradeDate")
            date_str = ""
            if epoch:
                date_str = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
            top_analysts.append({
                "analyst": item.get("firm") or "",
                "overallScore": score,
                "directionScore": max(10, score - 10),
                "priceScore": min(100, score + 15),
                "latestRating": rating,
                "priceTarget": item.get("currentPriceTarget") or "",
                "date": date_str
            })

        rec_trend = (result.get("recommendationTrend") or {}).get("trend") or []
        recommendations = []
        for row in rec_trend:
            recommendations.append({
                "period": row.get("period"),
                "strongBuy": row.get("strongBuy", 0),
                "buy": row.get("buy", 0),
                "hold": row.get("hold", 0),
                "sell": row.get("sell", 0),
                "strongSell": row.get("strongSell", 0),
                "underperform": row.get("underperform", 0)
            })

        direct_payload = {
            "top_analysts": top_analysts,
            "recommendations": recommendations,
            "top_analysts_source": "Yahoo Finance QuoteSummary (upgradeDowngradeHistory)",
            "recommendations_source": "Yahoo Finance QuoteSummary (recommendationTrend)"
        }
        if _has_rows(direct_payload):
            return direct_payload
        fallback = _fallback_from_yfinance(symbol)
        return fallback if _has_rows(fallback) else empty_payload
    except Exception as exc:
        logger.warning(f"Analyst insights fetch failed for {symbol}: {exc}")
        fallback = _fallback_from_yfinance(symbol)
        return fallback if _has_rows(fallback) else empty_payload

def fetch_company_profile(symbol: str, stock_obj=None):
    # 1) yfinance info (preferred when available)
    try:
        if stock_obj is not None and hasattr(stock_obj, "info"):
            info = stock_obj.info or {}
            if isinstance(info, dict) and (
                info.get("longBusinessSummary") or info.get("sector") or info.get("industry")
                or info.get("shortName") or info.get("longName")
            ):
                return _normalize_company_info(symbol, info), "Yahoo Finance via yfinance (info)"
    except Exception:
        pass
    try:
        info = yf.Ticker(symbol).info or {}
        if isinstance(info, dict) and (
            info.get("longBusinessSummary") or info.get("sector") or info.get("industry")
            or info.get("shortName") or info.get("longName")
        ):
            return _normalize_company_info(symbol, info), "Yahoo Finance via yfinance (info)"
    except Exception:
        pass

    # 2) QuoteSummary assetProfile fallback
    try:
        session_req = requests.Session()
        session_req.get("https://fc.yahoo.com", headers={"User-Agent": _USER_AGENT}, timeout=8)
        crumb_resp = session_req.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            headers={"User-Agent": _USER_AGENT},
            timeout=8
        )
        if crumb_resp.ok and crumb_resp.text:
            crumb = crumb_resp.text.strip()
            url = (
                "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
                f"{symbol}?modules=assetProfile,price&crumb={crumb}"
            )
            resp = session_req.get(url, headers={"User-Agent": _USER_AGENT}, timeout=8)
            if resp.ok:
                result = (resp.json().get("quoteSummary", {}).get("result") or [{}])[0]
                profile = result.get("assetProfile") or {}
                price = result.get("price") or {}
                merged = {
                    "name": price.get("shortName") or price.get("longName") or symbol,
                    "sector": profile.get("sector"),
                    "industry": profile.get("industry"),
                    "summary": profile.get("longBusinessSummary")
                }
                if merged.get("summary") or merged.get("sector") or merged.get("industry") or merged.get("name") != symbol:
                    return _normalize_company_info(symbol, merged), "Yahoo Finance QuoteSummary (assetProfile)"
    except Exception:
        pass

    return _normalize_company_info(symbol, {}), "Unavailable"

# On Fly.io, PERSISTENT_DATA_DIR is set to the mounted volume path (/data).
# Locally it falls back to the repo's data/ folder.
_DATA_DIR = os.getenv("PERSISTENT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
WATCHLIST_DIR = os.path.join(_DATA_DIR, "watchlists")
USERS_FILE = os.path.join(_DATA_DIR, "users.json")

def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            return {}
        # Migrate old flat format: {email: "hash_string"} → {email: {password_hash, tier, …}}
        migrated = False
        for email, val in list(users.items()):
            if isinstance(val, str):
                users[email] = {"password_hash": val, "tier": "free"}
                migrated = True
        if migrated:
            _save_users(users)
        return users
    return {}

def _save_users(users: dict) -> None:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f)

def _get_user_tier(email: str) -> str:
    if not email:
        return "free"
    if email.lower() in ADMIN_EMAILS:
        return "admin"
    users = _load_users()
    info = users.get(email, {})
    if not isinstance(info, dict):
        return "free"
    tier = info.get("tier", "free")
    status = info.get("subscription_status", "active")
    if tier != "free" and status not in ("active", "trialing"):
        return "free"
    return tier

def _check_and_increment_daily_usage(email: str):
    """Returns (allowed, used_today, limit). Increments usage count if allowed."""
    tier = _get_user_tier(email)
    if TIER_RANK.get(tier, 0) >= TIER_RANK["pro"]:
        return True, 0, -1  # paid users: unlimited

    today = datetime.now(timezone.utc).date().isoformat()
    users = _load_users()
    info = users.get(email, {})
    if not isinstance(info, dict):
        info = {"password_hash": info, "tier": "free"}
    daily_usage = info.get("daily_usage", {})
    used = daily_usage.get(today, 0)
    if used >= FREE_DAILY_LIMIT:
        return False, used, FREE_DAILY_LIMIT
    daily_usage[today] = used + 1
    info["daily_usage"] = daily_usage
    users[email] = info
    _save_users(users)
    return True, used + 1, FREE_DAILY_LIMIT

def subscription_required(min_tier="pro"):
    """Decorator that gates a route to users with at least `min_tier`."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            email = _get_current_user_email()
            if not email:
                return jsonify({"error": "Login required"}), 401
            tier = _get_user_tier(email)
            if TIER_RANK.get(tier, 0) < TIER_RANK[min_tier]:
                return jsonify({"error": "upgrade_required", "required_tier": min_tier}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator
REMARKABLES_CACHE_PATH = os.path.join(_DATA_DIR, "cache", "remarkables_nasdaq.json")
REMARKABLES_CACHE_TTL_SECONDS = 12 * 60 * 60
REMARKABLES_BATCH_SIZE = 25
REMARKABLES_RULE_VERSION = "2026-02-16-v8-risk-near-fill-up1.20-down0.9"
REMARKABLES_REFRESH_LOCK = threading.Lock()
REMARKABLES_REFRESH_IN_PROGRESS = False
REMARKABLES_MEMORY_CACHE = None
REMARKABLES_MEMORY_DAY = None

DIVIDEND_CACHE_PATH = os.path.join(_DATA_DIR, "cache", "dividend_stocks.json")
_DIVIDEND_MEM: list | None = None
_DIVIDEND_MEM_DAY: str | None = None

UNDERVALUED_CACHE_PATH = os.path.join(_DATA_DIR, "cache", "undervalued_stocks.json")
_UNDERVALUED_MEM: list | None = None
_UNDERVALUED_MEM_DAY: str | None = None
_UNDERVALUED_REFRESH_LOCK = threading.Lock()
_UNDERVALUED_REFRESH_IN_PROGRESS = False

def _sanitize_watchlist_key(value: str) -> str:
    if not value:
        return "anonymous"
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)

def _get_watchlist_path(email: str) -> str:
    os.makedirs(WATCHLIST_DIR, exist_ok=True)
    return os.path.join(WATCHLIST_DIR, f"{_sanitize_watchlist_key(email)}.json")

def _get_current_user_email():
    email = session.get("user_email")
    if email:
        return email
    if google_client_id and google_client_secret and google.authorized:
        resp = google.get("/oauth2/v2/userinfo")
        if resp.ok:
            info = resp.json()
            email = info.get("email")
            if email:
                session["user_email"] = email
                return email
    return None

def fetch_with_retry(symbol: str, period: str = '1y', attempts: int = 3, delay: int = 2):
    """Fetch price history with simple backoff to handle transient rate limits."""
    last_exc = None
    for attempt in range(attempts):
        try:
            ticker = yf.Ticker(symbol)
            return ticker.history(period=period)
        except Exception as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            is_rate_limit = status == 429 or "Too Many Requests" in str(exc)
            if is_rate_limit and attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
                continue
            break
    raise last_exc

def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or "Too Many Requests" in str(exc)

def fetch_stooq_history(symbol: str) -> pd.DataFrame:
    """Fallback data source when Yahoo is rate limited."""
    stooq_symbol = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    resp = requests.get(url, timeout=12, headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    text = (resp.text or "").strip()
    if not text or "No data" in text:
        raise ValueError(f"No Stooq data for {symbol}")

    df = pd.read_csv(StringIO(text))
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"Invalid Stooq data for {symbol}")

    # Stooq columns: Date,Open,High,Low,Close,Volume
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan if col != "Volume" else 0
    return df[["Open", "High", "Low", "Close", "Volume"]]

def fetch_sp500_stocks():
    """Fetch S&P 500 stocks with robust error handling and local fallback."""
    # 1) Local list
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        local_list = os.path.join(data_dir, 'sp500_symbols.txt')
        if os.path.exists(local_list):
            with open(local_list, 'r', encoding='utf-8') as f:
                symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if symbols:
                logger.info(f"Loaded {len(symbols)} symbols from local list")
                return symbols
    except Exception as e:
        logger.warning(f"Could not read local S&P 500 list: {e}")

    # 2) Wikipedia with custom User-Agent
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Safari/537.36"}
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = tables[0]
        stocks = df['Symbol'].astype(str).str.strip().tolist()[:500]
        logger.info(f"Successfully fetched {len(stocks)} stocks from Wikipedia")
        return stocks
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")

    # 3) Fallback small list
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
        'NVDA', 'JPM', 'V', 'JNJ', 'WMT', 'MA', 'UNH', 'DIS', 'BAC'
    ]

def fetch_nasdaq_symbols():
    """Fetch full Nasdaq listed symbols."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    resp = requests.get(url, timeout=12, headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    lines = resp.text.splitlines()
    symbols = []
    for line in lines[1:]:
        if not line or line.startswith("File Creation Time"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        symbol = (parts[0] or "").strip().upper()
        test_issue = (parts[6] or "").strip().upper() if len(parts) > 6 else "N"
        if not symbol or test_issue == "Y":
            continue
        # Keep common-stock style tickers to avoid warrants/units/rights noise.
        if not re.fullmatch(r"[A-Z]{1,5}", symbol):
            continue
        symbols.append(symbol)
    return sorted(set(symbols))

def _extract_close_from_batch(df_batch: pd.DataFrame, symbol: str):
    if df_batch is None or df_batch.empty:
        return pd.Series(dtype=float)
    try:
        if isinstance(df_batch.columns, pd.MultiIndex):
            # group_by='ticker' => (ticker, field)
            if symbol in df_batch.columns.get_level_values(0):
                series = df_batch[symbol].get("Close", pd.Series(dtype=float))
                return pd.to_numeric(series, errors="coerce").dropna()
            # Fallback orientation => (field, ticker)
            if "Close" in df_batch.columns.get_level_values(0) and symbol in df_batch.columns.get_level_values(1):
                series = df_batch["Close"][symbol]
                return pd.to_numeric(series, errors="coerce").dropna()
            return pd.Series(dtype=float)
        if "Close" in df_batch.columns:
            return pd.to_numeric(df_batch["Close"], errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)
    return pd.Series(dtype=float)

def _extract_field_from_batch(df_batch: pd.DataFrame, symbol: str, field: str):
    if df_batch is None or df_batch.empty:
        return pd.Series(dtype=float)
    try:
        if isinstance(df_batch.columns, pd.MultiIndex):
            # group_by='ticker' => (ticker, field)
            if symbol in df_batch.columns.get_level_values(0):
                series = df_batch[symbol].get(field, pd.Series(dtype=float))
                return pd.to_numeric(series, errors="coerce").dropna()
            # Fallback orientation => (field, ticker)
            if field in df_batch.columns.get_level_values(0) and symbol in df_batch.columns.get_level_values(1):
                series = df_batch[field][symbol]
                return pd.to_numeric(series, errors="coerce").dropna()
            return pd.Series(dtype=float)
        if field in df_batch.columns:
            return pd.to_numeric(df_batch[field], errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)
    return pd.Series(dtype=float)

def _trend_percent(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("-inf")
    start_val = float(series.iloc[0])
    end_val = float(series.iloc[-1])
    if start_val <= 0:
        return float("-inf")
    return ((end_val - start_val) / start_val) * 100.0

def _prevday_highlow_hits(close_series: pd.Series, high_series: pd.Series, low_series: pd.Series,
                          up_mult: float = 1.5, down_mult: float = 0.75):
    df = pd.concat(
        [
            pd.to_numeric(close_series, errors="coerce"),
            pd.to_numeric(high_series, errors="coerce"),
            pd.to_numeric(low_series, errors="coerce"),
        ],
        axis=1,
    )
    df.columns = ["close", "high", "low"]
    df = df.dropna()
    if len(df) < 2:
        return 0, 0
    prev_close = df["close"].shift(1)
    up_hits = int((df["high"] >= (prev_close * up_mult)).sum())
    down_hits = int((df["low"] <= (prev_close * down_mult)).sum())
    return up_hits, down_hits

def _default_remarkables_payload():
    return {
        "updated_at": None,
        "source": "cache",
        "rule_version": REMARKABLES_RULE_VERSION,
        "scanned_symbols": 0,
        "total_symbols": 0,
        "for_risk_lovers": [],
        "no_pain_but_gain": []
    }

def _load_remarkables_cache():
    if not os.path.exists(REMARKABLES_CACHE_PATH):
        return None
    try:
        with open(REMARKABLES_CACHE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.warning(f"Failed to read remarkables cache: {exc}")
        return None

def _save_remarkables_cache(payload: dict):
    os.makedirs(os.path.dirname(REMARKABLES_CACHE_PATH), exist_ok=True)
    try:
        with open(REMARKABLES_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception as exc:
        logger.warning(f"Failed to write remarkables cache: {exc}")

def _remarkables_day_key(dt=None) -> str:
    val = dt or datetime.now(timezone.utc)
    return val.astimezone(timezone.utc).date().isoformat()

def _remarkables_payload_day(payload: dict):
    ts = (payload or {}).get("updated_at")
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return _remarkables_day_key(parsed)
    except Exception:
        return None

def _remarkables_cache_is_today(payload: dict) -> bool:
    if (payload or {}).get("rule_version") != REMARKABLES_RULE_VERSION:
        return False
    return _remarkables_payload_day(payload) == _remarkables_day_key()


def _compute_remarkables():
    symbols = fetch_nasdaq_symbols()
    risk_candidates = []
    risk_near_candidates = []
    steady_candidates = []
    scanned = 0

    rate_limit_hits = 0
    # yfinance can emit noisy per-batch error logs on 429; quiet it during bulk scan.
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)

    for idx in range(0, len(symbols), REMARKABLES_BATCH_SIZE):
        batch = symbols[idx: idx + REMARKABLES_BATCH_SIZE]
        try:
            data = yf.download(
                tickers=" ".join(batch),
                period="6mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=False
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                rate_limit_hits += 1
                logger.warning(f"Remarkables rate limited at batch {idx}; hit {rate_limit_hits}")
                if rate_limit_hits >= 1:
                    logger.warning("Stopping remarkable scan early due to repeated rate limits.")
                    break
                time.sleep(1.5 * rate_limit_hits)
                continue
            logger.warning(f"Remarkables batch download failed at {idx}: {exc}")
            continue

        for sym in batch:
            close_6m = _extract_close_from_batch(data, sym)
            high_6m = _extract_field_from_batch(data, sym, "High")
            low_6m = _extract_field_from_batch(data, sym, "Low")
            scanned += 1
            if close_6m.empty or len(close_6m) < 60:
                continue
            # Filter out extreme penny/illiquid-like behavior that dominates NASDAQ-wide scans.
            if float(close_6m.median()) < 1.0:
                continue

            trend_6m = _trend_percent(close_6m)
            daily_loss = close_6m.pct_change().dropna()
            max_daily_drop = abs(float(daily_loss.min() * 100.0)) if not daily_loss.empty and daily_loss.min() < 0 else 0.0

            # Last-day change %
            day_chg = 0.0
            if len(close_6m) >= 2:
                day_chg = round(float((close_6m.iloc[-1] - close_6m.iloc[-2]) / close_6m.iloc[-2] * 100), 2)

            # No Pain But Gain: +10% in 6M, and no daily drop >9%
            if trend_6m >= 10.0 and (daily_loss >= -0.09).all():
                steady_candidates.append({
                    "symbol": sym,
                    "trend_percent": round(trend_6m, 2),
                    "max_daily_drop_percent": round(max_daily_drop, 2),
                    "day_change_pct": day_chg,
                    "score": trend_6m - max_daily_drop,
                    "match_type": "strict"
                })

            # For Risk Lovers: evaluate last ~3 months
            close_3m = close_6m.tail(63)
            high_3m = high_6m.tail(63)
            low_3m = low_6m.tail(63)
            if close_3m.empty or len(close_3m) < 40:
                continue
            trend_3m = _trend_percent(close_3m)
            up_hits, down_hits = _prevday_highlow_hits(close_3m, high_3m, low_3m, up_mult=1.20, down_mult=0.9)
            if trend_3m >= 15.0 and down_hits >= 2 and up_hits >= 3:
                risk_candidates.append({
                    "symbol": sym,
                    "trend_percent": round(trend_3m, 2),
                    "loss_hits": down_hits,
                    "gain_hits": up_hits,
                    "down_hits": down_hits,
                    "up_hits": up_hits,
                    "day_change_pct": day_chg,
                    "score": (trend_3m + up_hits * 8 + down_hits * 8),
                    "match_type": "strict"
                })
            else:
                # Keep near matches so list can still be filled when strict results are scarce.
                trend_gap = max(0.0, 15.0 - trend_3m)
                down_gap = max(0, 2 - down_hits)
                up_gap = max(0, 3 - up_hits)
                miss_points = (trend_gap / 5.0) + (down_gap * 1.0) + (up_gap * 1.0)
                if trend_3m >= 5.0 or up_hits >= 1 or down_hits >= 1:
                    risk_near_candidates.append({
                        "symbol": sym,
                        "trend_percent": round(trend_3m, 2),
                        "loss_hits": down_hits,
                        "gain_hits": up_hits,
                        "down_hits": down_hits,
                        "up_hits": up_hits,
                        "day_change_pct": day_chg,
                        "score": (trend_3m + up_hits * 4 + down_hits * 4 - miss_points * 5),
                        "match_type": "near_match",
                        "_miss_points": miss_points
                    })

    yf_logger.setLevel(prev_level)

    risk_sorted = sorted(risk_candidates, key=lambda x: x["score"], reverse=True)
    steady_sorted = sorted(steady_candidates, key=lambda x: x["score"], reverse=True)
    risk_near_sorted = sorted(
        risk_near_candidates,
        key=lambda x: (x.get("_miss_points", 9999), -x.get("trend_percent", 0.0), -(x.get("up_hits", 0) + x.get("down_hits", 0)))
    )
    risk_sorted = risk_sorted[:5]
    # Disjoint rule: symbols in risk list cannot appear in steady list.
    risk_symbols = {x["symbol"] for x in risk_sorted}
    steady_sorted = [x for x in steady_sorted if x["symbol"] not in risk_symbols][:5]
    if len(risk_sorted) < 5 and risk_near_sorted:
        used = {x["symbol"] for x in risk_sorted}
        for item in risk_near_sorted:
            if item["symbol"] in used:
                continue
            # Internal field used only for ranking; do not expose it.
            item.pop("_miss_points", None)
            risk_sorted.append(item)
            used.add(item["symbol"])
            if len(risk_sorted) >= 5:
                break

    # Fallback: if live NASDAQ-wide scan produced no candidates due API throttling,
    # harvest candidates from existing local symbol caches.
    if len(risk_sorted) < 5 or len(steady_sorted) < 5:
        local_risk, local_steady, local_risk_near, local_steady_near = _compute_remarkables_from_local_cache()
        if len(risk_sorted) < 5 and (local_risk or local_risk_near):
            used = {x["symbol"] for x in risk_sorted}
            risk_sorted.extend([x for x in local_risk if x["symbol"] not in used])
            used = {x["symbol"] for x in risk_sorted}
            if len(risk_sorted) < 5:
                risk_sorted.extend([x for x in local_risk_near if x["symbol"] not in used])
            risk_sorted = risk_sorted[:5]
        if len(steady_sorted) < 5 and (local_steady or local_steady_near):
            used = {x["symbol"] for x in steady_sorted}
            steady_sorted.extend([x for x in local_steady if x["symbol"] not in used and x["symbol"] not in risk_symbols])
            used = {x["symbol"] for x in steady_sorted}
            if len(steady_sorted) < 5:
                steady_sorted.extend([x for x in local_steady_near if x["symbol"] not in used and x["symbol"] not in risk_symbols])
            steady_sorted = steady_sorted[:5]

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "live_scan",
        "rule_version": REMARKABLES_RULE_VERSION,
        "scanned_symbols": scanned,
        "total_symbols": len(symbols),
        "for_risk_lovers": risk_sorted,
        "no_pain_but_gain": steady_sorted
    }

def _compute_remarkables_from_local_cache():
    cache_dir = os.path.join(_DATA_DIR, "cache")
    if not os.path.isdir(cache_dir):
        return [], [], [], []
    risk_strict = []
    steady_strict = []
    risk_near = []
    steady_near = []
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".json"):
            continue
        if fname == "remarkables_nasdaq.json":
            continue
        symbol = os.path.splitext(fname)[0].upper()
        try:
            with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            closes = payload.get("actual_close") or []
            highs = payload.get("actual_high") or []
            lows = payload.get("actual_low") or []
            s = pd.Series(pd.to_numeric(closes, errors="coerce")).dropna()
            if len(s) < 60:
                continue
            if float(s.median()) < 1.0:
                continue
            s6 = s.tail(min(126, len(s)))
            trend_6m = _trend_percent(s6)
            daily = s6.pct_change().dropna()
            max_drop = abs(float(daily.min() * 100.0)) if not daily.empty and daily.min() < 0 else 0.0
            day_chg = round(float((s6.iloc[-1] - s6.iloc[-2]) / s6.iloc[-2] * 100), 2) if len(s6) >= 2 else 0.0
            if trend_6m >= 10.0 and (daily >= -0.09).all():
                steady_strict.append({
                    "symbol": symbol,
                    "trend_percent": round(trend_6m, 2),
                    "max_daily_drop_percent": round(max_drop, 2),
                    "day_change_pct": day_chg,
                    "score": trend_6m - max_drop,
                    "match_type": "cache_strict"
                })
            elif trend_6m >= 8.0 and (daily >= -0.12).all():
                steady_near.append({
                    "symbol": symbol,
                    "trend_percent": round(trend_6m, 2),
                    "max_daily_drop_percent": round(max_drop, 2),
                    "day_change_pct": day_chg,
                    "score": trend_6m - max_drop,
                    "match_type": "cache_near_match"
                })
            s3 = s6.tail(min(63, len(s6)))
            if len(s3) < 40:
                continue
            start = float(s3.iloc[0])
            if start <= 0:
                continue
            trend_3m = _trend_percent(s3)
            hs = pd.Series(pd.to_numeric(highs, errors="coerce")).dropna() if highs else s.copy()
            ls = pd.Series(pd.to_numeric(lows, errors="coerce")).dropna() if lows else s.copy()
            hs3 = hs.tail(len(s3))
            ls3 = ls.tail(len(s3))
            up_hits, down_hits = _prevday_highlow_hits(s3, hs3, ls3, up_mult=1.20, down_mult=0.9)
            if trend_3m >= 15.0 and down_hits >= 2 and up_hits >= 3:
                risk_strict.append({
                    "symbol": symbol,
                    "trend_percent": round(trend_3m, 2),
                    "loss_hits": down_hits,
                    "gain_hits": up_hits,
                    "down_hits": down_hits,
                    "up_hits": up_hits,
                    "day_change_pct": day_chg,
                    "score": (trend_3m + up_hits * 8 + down_hits * 8),
                    "match_type": "cache_strict"
                })
            else:
                trend_gap = max(0.0, 15.0 - trend_3m)
                down_gap = max(0, 2 - down_hits)
                up_gap = max(0, 3 - up_hits)
                miss_points = (trend_gap / 5.0) + (down_gap * 1.0) + (up_gap * 1.0)
                if trend_3m >= 5.0 or up_hits >= 1 or down_hits >= 1:
                    risk_near.append({
                        "symbol": symbol,
                        "trend_percent": round(trend_3m, 2),
                        "loss_hits": down_hits,
                        "gain_hits": up_hits,
                        "down_hits": down_hits,
                        "up_hits": up_hits,
                        "day_change_pct": day_chg,
                        "score": (trend_3m + up_hits * 4 + down_hits * 4 - miss_points * 5),
                        "match_type": "cache_near_match",
                        "_miss_points": miss_points
                    })
        except Exception:
            continue
    risk_strict = sorted(risk_strict, key=lambda x: x["score"], reverse=True)
    steady_strict = sorted(steady_strict, key=lambda x: x["score"], reverse=True)
    risk_near = sorted(
        risk_near,
        key=lambda x: (x.get("_miss_points", 9999), -x.get("trend_percent", 0.0), -(x.get("up_hits", 0) + x.get("down_hits", 0)))
    )
    for item in risk_near:
        item.pop("_miss_points", None)
    steady_near = sorted(steady_near, key=lambda x: x["score"], reverse=True)
    return risk_strict, steady_strict, risk_near, steady_near

def _refresh_remarkables_worker():
    global REMARKABLES_REFRESH_IN_PROGRESS, REMARKABLES_MEMORY_CACHE, REMARKABLES_MEMORY_DAY
    try:
        payload = _compute_remarkables()
        _save_remarkables_cache(payload)
        REMARKABLES_MEMORY_CACHE = payload
        REMARKABLES_MEMORY_DAY = _remarkables_payload_day(payload)
    except Exception as exc:
        logger.warning(f"Remarkables background refresh failed: {exc}")
    finally:
        with REMARKABLES_REFRESH_LOCK:
            REMARKABLES_REFRESH_IN_PROGRESS = False

def _start_remarkables_refresh_if_needed():
    global REMARKABLES_REFRESH_IN_PROGRESS
    with REMARKABLES_REFRESH_LOCK:
        if REMARKABLES_REFRESH_IN_PROGRESS:
            return
        REMARKABLES_REFRESH_IN_PROGRESS = True
    t = threading.Thread(target=_refresh_remarkables_worker, daemon=True)
    t.start()

def get_remarkables(force_refresh: bool = False):
    global REMARKABLES_MEMORY_CACHE, REMARKABLES_MEMORY_DAY
    today_key = _remarkables_day_key()

    if not force_refresh and REMARKABLES_MEMORY_CACHE and REMARKABLES_MEMORY_DAY == today_key:
        payload = dict(REMARKABLES_MEMORY_CACHE)
        payload["source"] = "memory_cache"
        return payload

    cached = _load_remarkables_cache()
    if cached and cached.get("rule_version") != REMARKABLES_RULE_VERSION:
        cached = None
    if cached:
        risk_list = cached.get("for_risk_lovers") or []
        steady_list = cached.get("no_pain_but_gain") or []
        risk_symbols = {x.get("symbol") for x in risk_list if x.get("symbol")}
        if steady_list:
            steady_list = [x for x in steady_list if x.get("symbol") not in risk_symbols]
            cached["no_pain_but_gain"] = steady_list[:5]
        if len(risk_list) < 5 or len(steady_list) < 5:
            local_risk, local_steady, local_risk_near, local_steady_near = _compute_remarkables_from_local_cache()
            if len(risk_list) < 5 and (local_risk or local_risk_near):
                used = {x.get("symbol") for x in risk_list}
                risk_list.extend([x for x in local_risk if x.get("symbol") not in used])
                used = {x.get("symbol") for x in risk_list}
                if len(risk_list) < 5:
                    risk_list.extend([x for x in local_risk_near if x.get("symbol") not in used])
                cached["for_risk_lovers"] = risk_list[:5]
                risk_symbols = {x.get("symbol") for x in cached["for_risk_lovers"] if x.get("symbol")}
            if len(steady_list) < 5 and (local_steady or local_steady_near):
                used = {x.get("symbol") for x in steady_list}
                steady_list.extend([x for x in local_steady if x.get("symbol") not in used and x.get("symbol") not in risk_symbols])
                used = {x.get("symbol") for x in steady_list}
                if len(steady_list) < 5:
                    steady_list.extend([x for x in local_steady_near if x.get("symbol") not in used and x.get("symbol") not in risk_symbols])
                cached["no_pain_but_gain"] = steady_list[:5]
            if (cached.get("for_risk_lovers") or []) or (cached.get("no_pain_but_gain") or []):
                cached["source"] = "cache_recovered"
                cached["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_remarkables_cache(cached)

    if cached and _remarkables_cache_is_today(cached):
        REMARKABLES_MEMORY_CACHE = cached
        REMARKABLES_MEMORY_DAY = today_key
        payload = dict(cached)
        payload["source"] = "file_cache_today"
        return payload

    if force_refresh:
        try:
            payload = _compute_remarkables()
            _save_remarkables_cache(payload)
            REMARKABLES_MEMORY_CACHE = payload
            REMARKABLES_MEMORY_DAY = _remarkables_payload_day(payload)
            return payload
        except Exception as exc:
            logger.warning(f"Remarkables forced computation failed: {exc}")
            if cached:
                cached["source"] = "stale_cache"
                return cached
            return _default_remarkables_payload()

    # Daily refresh mode: trigger background job at most once when day changed.
    _start_remarkables_refresh_if_needed()
    if cached:
        REMARKABLES_MEMORY_CACHE = cached
        REMARKABLES_MEMORY_DAY = _remarkables_payload_day(cached)
        cached["source"] = "stale_cache_refreshing_daily"
        return cached
    payload = _default_remarkables_payload()
    local_risk, local_steady, local_risk_near, local_steady_near = _compute_remarkables_from_local_cache()
    payload["for_risk_lovers"] = (local_risk + local_risk_near)[:5]
    risk_symbols = {x.get("symbol") for x in payload["for_risk_lovers"] if x.get("symbol")}
    payload["no_pain_but_gain"] = [
        x for x in (local_steady + local_steady_near)
        if x.get("symbol") not in risk_symbols
    ][:5]
    payload["source"] = "warming_up_daily_refresh"
    return payload

def load_cached_response(symbol: str):
    """Load cached API-style response if available."""
    cache_dir = os.path.join(_DATA_DIR, 'cache')
    cache_path = os.path.join(cache_dir, f'{symbol}.json')
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        payload = _upgrade_cached_payload(payload, symbol)
        payload['cache_used'] = True
        return payload
    except Exception as exc:
        logger.warning(f"Failed to load cache for {symbol}: {exc}")
        return None

def save_cached_response(symbol: str, payload: dict):
    """Persist API-style response to cache for reuse."""
    cache_dir = os.path.join(_DATA_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{symbol}.json')
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
    except Exception as exc:
        logger.warning(f"Failed to write cache for {symbol}: {exc}")

def _find_local_extrema(values):
    peaks = []
    troughs = []
    for i in range(1, len(values) - 1):
        if values[i - 1] < values[i] > values[i + 1]:
            peaks.append(i)
        if values[i - 1] > values[i] < values[i + 1]:
            troughs.append(i)
    return peaks, troughs

def _detect_head_shoulders(values):
    if len(values) < 7:
        return None
    peaks, troughs = _find_local_extrema(values)
    for i in range(len(peaks) - 3, -1, -1):
        p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]
        if not (p1 < p2 < p3):
            continue
        left = values[p1]
        head = values[p2]
        right = values[p3]
        shoulder_diff = abs(left - right) / max(left, right)
        if shoulder_diff > 0.06:
            continue
        if not (head > left * 1.03 and head > right * 1.03):
            continue
        t1 = next((t for t in reversed(troughs) if p1 < t < p2), None)
        t2 = next((t for t in reversed(troughs) if p2 < t < p3), None)
        if t1 is None or t2 is None:
            continue
        if values[t1] >= min(left, head, right) or values[t2] >= min(left, head, right):
            continue
        return {
            'name': 'Head and Shoulders',
            'start_idx': p1,
            'end_idx': p3
        }
    return None

def _detect_inverse_head_shoulders(values):
    if len(values) < 7:
        return None
    peaks, troughs = _find_local_extrema(values)
    for i in range(len(troughs) - 3, -1, -1):
        t1, t2, t3 = troughs[i], troughs[i + 1], troughs[i + 2]
        if not (t1 < t2 < t3):
            continue
        left = values[t1]
        head = values[t2]
        right = values[t3]
        shoulder_diff = abs(left - right) / max(left, right)
        if shoulder_diff > 0.06:
            continue
        if not (head < left * 0.97 and head < right * 0.97):
            continue
        p1 = next((p for p in reversed(peaks) if t1 < p < t2), None)
        p2 = next((p for p in reversed(peaks) if t2 < p < t3), None)
        if p1 is None or p2 is None:
            continue
        if values[p1] <= max(left, head, right) or values[p2] <= max(left, head, right):
            continue
        return {
            'name': 'Inverse Head and Shoulders',
            'start_idx': t1,
            'end_idx': t3
        }
    return None

def _detect_double_top(values):
    if len(values) < 6:
        return None
    peaks, troughs = _find_local_extrema(values)
    for i in range(len(peaks) - 2, -1, -1):
        p1, p2 = peaks[i], peaks[i + 1]
        if p1 >= p2:
            continue
        peak_diff = abs(values[p1] - values[p2]) / max(values[p1], values[p2])
        if peak_diff > 0.04:
            continue
        t = next((t for t in troughs if p1 < t < p2), None)
        if t is None:
            continue
        if values[t] >= min(values[p1], values[p2]) * 0.97:
            continue
        return {'name': 'Double Top', 'start_idx': p1, 'end_idx': p2}
    return None

def _detect_double_bottom(values):
    if len(values) < 6:
        return None
    peaks, troughs = _find_local_extrema(values)
    for i in range(len(troughs) - 2, -1, -1):
        t1, t2 = troughs[i], troughs[i + 1]
        if t1 >= t2:
            continue
        trough_diff = abs(values[t1] - values[t2]) / max(values[t1], values[t2])
        if trough_diff > 0.04:
            continue
        p = next((p for p in peaks if t1 < p < t2), None)
        if p is None:
            continue
        if values[p] <= max(values[t1], values[t2]) * 1.03:
            continue
        return {'name': 'Double Bottom', 'start_idx': t1, 'end_idx': t2}
    return None

def _detect_rounding_bottom(values):
    if len(values) < 10:
        return None
    mid = len(values) // 2
    left = np.mean(values[:mid])
    right = np.mean(values[mid:])
    low_idx = int(np.argmin(values))
    if low_idx < len(values) * 0.25 or low_idx > len(values) * 0.75:
        return None
    if values[0] > values[low_idx] * 1.05 and values[-1] > values[low_idx] * 1.05:
        if left > values[low_idx] * 1.02 and right > values[low_idx] * 1.02:
            return {'name': 'Rounding Bottom', 'start_idx': 0, 'end_idx': len(values) - 1}
    return None

def _detect_rounding_top(values):
    if len(values) < 10:
        return None
    mid = len(values) // 2
    left = np.mean(values[:mid])
    right = np.mean(values[mid:])
    high_idx = int(np.argmax(values))
    if high_idx < len(values) * 0.25 or high_idx > len(values) * 0.75:
        return None
    if values[0] < values[high_idx] * 0.95 and values[-1] < values[high_idx] * 0.95:
        if left < values[high_idx] * 0.98 and right < values[high_idx] * 0.98:
            return {'name': 'Rounding Top', 'start_idx': 0, 'end_idx': len(values) - 1}
    return None

def _detect_cup_handle(values):
    if len(values) < 12:
        return None
    mid = len(values) // 2
    low_idx = int(np.argmin(values[:mid]))
    if low_idx < len(values) * 0.2:
        return None
    left_high = np.max(values[:low_idx])
    right_high = np.max(values[mid:])
    if left_high <= values[low_idx] * 1.05 or right_high <= values[low_idx] * 1.05:
        return None
    handle_start = mid
    handle_end = len(values) - 1
    handle_low = np.min(values[handle_start:handle_end + 1])
    if handle_low < right_high * 0.9:
        return None
    return {'name': 'Cup and Handle', 'start_idx': 0, 'end_idx': len(values) - 1}

def _detect_flags_pennants(values):
    if len(values) < 8:
        return None
    n = len(values)
    pole_end = max(2, n // 4)
    pole_move = values[pole_end] - values[0]
    if abs(pole_move) < np.std(values) * 0.5:
        return None
    consolidation = values[pole_end:]
    if np.ptp(consolidation) < abs(pole_move) * 0.5:
        name = 'Bull Flag' if pole_move > 0 else 'Bear Flag'
        return {'name': name, 'start_idx': 0, 'end_idx': n - 1}
    return None

def _detect_triangle(values):
    if len(values) < 8:
        return None
    n = len(values)
    left = values[: n // 2]
    right = values[n // 2 :]
    left_range = np.ptp(left)
    right_range = np.ptp(right)
    if right_range < left_range * 0.7:
        return {'name': 'Triangle', 'start_idx': 0, 'end_idx': n - 1}
    return None

def detect_pattern(values):
    """Detect simple reversal formations in a recent window."""
    pattern = _detect_head_shoulders(values)
    if pattern:
        return pattern
    pattern = _detect_inverse_head_shoulders(values)
    if pattern:
        return pattern
    pattern = _detect_double_top(values)
    if pattern:
        return pattern
    pattern = _detect_double_bottom(values)
    if pattern:
        return pattern
    pattern = _detect_rounding_bottom(values)
    if pattern:
        return pattern
    pattern = _detect_rounding_top(values)
    if pattern:
        return pattern
    pattern = _detect_cup_handle(values)
    if pattern:
        return pattern
    pattern = _detect_flags_pennants(values)
    if pattern:
        return pattern
    return _detect_triangle(values)

def _trend_hint_for_pattern(name):
    trend_map = {
        'Head and Shoulders': {'text': 'Bearish', 'color': 'red', 'ay': -40, 'arrowcolor': 'red'},
        'Inverse Head and Shoulders': {'text': 'Bullish', 'color': 'green', 'ay': 40, 'arrowcolor': 'green'},
        'Double Top': {'text': 'Bearish', 'color': 'red', 'ay': -40, 'arrowcolor': 'red'},
        'Double Bottom': {'text': 'Bullish', 'color': 'green', 'ay': 40, 'arrowcolor': 'green'},
        'Rounding Top': {'text': 'Bearish', 'color': 'red', 'ay': -40, 'arrowcolor': 'red'},
        'Rounding Bottom': {'text': 'Bullish', 'color': 'green', 'ay': 40, 'arrowcolor': 'green'},
        'Cup and Handle': {'text': 'Bullish', 'color': 'green', 'ay': 40, 'arrowcolor': 'green'},
        'Bull Flag': {'text': 'Bullish', 'color': 'green', 'ay': 40, 'arrowcolor': 'green'},
        'Bear Flag': {'text': 'Bearish', 'color': 'red', 'ay': -40, 'arrowcolor': 'red'},
        'Triangle': {'text': 'Neutral', 'color': 'gray', 'ay': 0, 'arrowcolor': 'gray'}
    }
    return trend_map.get(name)

def _build_pattern_chart_from_series(label, dates, close_values, open_values=None, high_values=None, low_values=None):
    if not dates or not close_values:
        return None
    window = 22
    dates = dates[-window:] if len(dates) > window else dates
    close_values = close_values[-window:] if len(close_values) > window else close_values
    if open_values is not None and high_values is not None and low_values is not None:
        open_values = open_values[-window:] if len(open_values) > window else open_values
        high_values = high_values[-window:] if len(high_values) > window else high_values
        low_values = low_values[-window:] if len(low_values) > window else low_values
    use_ohlc = (
        open_values is not None and high_values is not None and low_values is not None
        and len(open_values) == len(close_values)
        and len(high_values) == len(close_values)
        and len(low_values) == len(close_values)
    )
    if use_ohlc:
        fig = go.Figure(data=[
            go.Candlestick(
                x=dates,
                open=open_values,
                high=high_values,
                low=low_values,
                close=close_values,
                name='OHLC (Last 1 Month)'
            )
        ])
    else:
        fig = go.Figure(data=[
            go.Scatter(x=dates, y=close_values, mode='lines', name='Close (Last 1 Month)')
        ])
    pattern = detect_pattern(np.asarray(close_values, dtype=float))
    if pattern:
        start_idx = pattern['start_idx']
        end_idx = pattern['end_idx']
        fig.update_layout(
        title=f"{label} {pattern['name']} Pattern (Last 1 Month)",
            xaxis_title='Date',
            yaxis_title='Price'
        )
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=dates[start_idx],
            x1=dates[end_idx],
            y0=0,
            y1=1,
            fillcolor="rgba(255, 193, 7, 0.2)",
            line_width=0
        )
        trend_hint = _trend_hint_for_pattern(pattern['name'])
        if trend_hint and trend_hint['text'] != 'Neutral':
            y_val = float(close_values[end_idx])
            fig.add_annotation(
                x=dates[end_idx],
                y=y_val,
                text=trend_hint['text'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=trend_hint['arrowcolor'],
                ax=0,
                ay=trend_hint['ay'],
                font={'color': trend_hint['color']}
            )
    else:
        fig.update_layout(
        title=f'{label} No Clear Pattern Detected (Last 1 Month)',
        xaxis_title='Date',
        yaxis_title='Price'
    )
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def _detect_rsi_divergences(dates, close_values, rsi_values, lookback=5):
    """
    Detect classic RSI divergences over a short window.
    Returns list of dicts with: x, y, text, color for annotations.
    """
    annotations = []
    n = len(rsi_values)
    if n < lookback + 2:
        return annotations

    # Find local price highs/lows and RSI highs/lows
    for i in range(lookback, n - 1):
        window_close = close_values[i - lookback:i + 1]
        window_rsi   = rsi_values[i - lookback:i + 1]

        # Bearish divergence: price makes higher high, RSI makes lower high
        if (close_values[i] == max(window_close) and
                rsi_values[i] < max(window_rsi) and
                rsi_values[i] > 55):
            annotations.append({
                'x': dates[i], 'y': rsi_values[i],
                'text': '⚠ Bearish Div', 'color': '#ff8a8a',
                'ay': -35
            })

        # Bullish divergence: price makes lower low, RSI makes higher low
        if (close_values[i] == min(window_close) and
                rsi_values[i] > min(window_rsi) and
                rsi_values[i] < 45):
            annotations.append({
                'x': dates[i], 'y': rsi_values[i],
                'text': '↑ Bullish Div', 'color': '#6ee7a8',
                'ay': 35
            })

    # Deduplicate: keep only first occurrence per divergence cluster
    seen = set()
    deduped = []
    for a in annotations:
        key = a['text']
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    return deduped


def _build_rsi_chart_from_series(label, dates, close_values):
    if not dates or not close_values:
        return None

    # Use 45 days so divergence detection has enough history while showing ~1 month
    window = 45
    dates_full  = dates[-window:]       if len(dates) > window        else dates
    close_full  = close_values[-window:] if len(close_values) > window else close_values

    rsi_full   = compute_rsi(pd.Series(close_full, dtype=float), window=14).tolist()

    # Display only last 22 trading days (~1 month) but detect divergences on full window
    display_n   = min(22, len(dates_full))
    dates_disp  = dates_full[-display_n:]
    rsi_disp    = rsi_full[-display_n:]
    close_disp  = close_full[-display_n:]

    last_rsi = rsi_disp[-1] if rsi_disp else 50.0

    # Badge text for current RSI
    if last_rsi >= 70:
        badge = f'RSI {last_rsi:.1f} — Overbought ⚠'
        badge_color = '#ff8a8a'
        signal_text = 'Overbought'
    elif last_rsi <= 30:
        badge = f'RSI {last_rsi:.1f} — Oversold ✓'
        badge_color = '#6ee7a8'
        signal_text = 'Oversold'
    elif last_rsi >= 60:
        badge = f'RSI {last_rsi:.1f} — Approaching Overbought'
        badge_color = '#f7c04f'
        signal_text = 'Elevated'
    elif last_rsi <= 40:
        badge = f'RSI {last_rsi:.1f} — Approaching Oversold'
        badge_color = '#f7c04f'
        signal_text = 'Depressed'
    else:
        badge = f'RSI {last_rsi:.1f} — Neutral'
        badge_color = '#9ec3ff'
        signal_text = 'Neutral'

    fig = go.Figure()

    # Zone shading — overbought (70-100) in red, oversold (0-30) in green
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,100,100,0.10)',
                  line_width=0, name='Overbought zone')
    fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(100,220,140,0.10)',
                  line_width=0, name='Oversold zone')

    # Reference lines
    fig.add_hline(y=70, line_dash='dash', line_color='rgba(255,100,100,0.6)',
                  line_width=1, annotation_text='70', annotation_position='left')
    fig.add_hline(y=50, line_dash='dot',  line_color='rgba(255,255,255,0.2)',
                  line_width=1, annotation_text='50', annotation_position='left')
    fig.add_hline(y=30, line_dash='dash', line_color='rgba(100,220,140,0.6)',
                  line_width=1, annotation_text='30', annotation_position='left')

    # RSI line — colour shifts based on zone
    fig.add_trace(go.Scatter(
        x=dates_disp, y=rsi_disp,
        mode='lines',
        name='RSI (14)',
        line=dict(color='#9ec3ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(79,142,247,0.06)',
    ))

    # Current RSI endpoint marker
    fig.add_trace(go.Scatter(
        x=[dates_disp[-1]],
        y=[last_rsi],
        mode='markers+text',
        name='Current RSI',
        marker=dict(color=badge_color, size=9, symbol='circle',
                    line=dict(color='white', width=1)),
        text=[f'{last_rsi:.1f}'],
        textposition='top right',
        textfont=dict(color=badge_color, size=11),
        showlegend=False,
    ))

    # Divergence markers (run on full 45-day window)
    divergences = _detect_rsi_divergences(dates_full, close_full, rsi_full)
    # Only annotate those that fall inside the display window
    display_set = set(dates_disp)
    for div in divergences:
        if div['x'] not in display_set:
            continue
        fig.add_annotation(
            x=div['x'], y=div['y'],
            text=div['text'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=div['color'],
            ax=0, ay=div['ay'],
            font=dict(color=div['color'], size=10),
            bgcolor='rgba(27,42,87,0.8)',
            bordercolor=div['color'],
            borderwidth=1,
            borderpad=3,
        )

    # Badge annotation in top-left corner
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.97,
        text=f'<b>{badge}</b>',
        showarrow=False,
        font=dict(color=badge_color, size=12),
        bgcolor='rgba(27,42,87,0.85)',
        bordercolor=badge_color,
        borderwidth=1,
        borderpad=4,
        xanchor='left', yanchor='top',
    )

    fig.update_layout(
        title=f'{label} — RSI Analysis (Last 1 Month) · {signal_text}',
        xaxis_title='Date',
        yaxis=dict(title='RSI', range=[0, 100]),
        legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5),
        margin=dict(b=80),
    )

    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def _build_pattern_chart_48h_from_df(symbol: str, df_48h: pd.DataFrame):
    if df_48h is None or df_48h.empty:
        return None
    dates = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in df_48h.index]
    fig = go.Figure(data=[
        go.Candlestick(
            x=dates,
            open=df_48h['Open'].astype(float).tolist(),
            high=df_48h['High'].astype(float).tolist(),
            low=df_48h['Low'].astype(float).tolist(),
            close=df_48h['Close'].astype(float).tolist(),
            name="OHLC (Last 48 Hours)"
        )
    ])
    pattern = detect_pattern(df_48h['Close'].astype(float).values)
    if pattern:
        s_idx = pattern['start_idx']
        e_idx = pattern['end_idx']
        fig.update_layout(
            title=f"{symbol} {pattern['name']} Pattern (Last 48 Hours)",
            xaxis_title='Date',
            yaxis_title='Price'
        )
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=dates[s_idx],
            x1=dates[e_idx],
            y0=0,
            y1=1,
            fillcolor="rgba(255, 193, 7, 0.2)",
            line_width=0
        )
        trend_hint = _trend_hint_for_pattern(pattern['name'])
        if trend_hint and trend_hint['text'] != 'Neutral':
            y_val = float(df_48h['Close'].iloc[e_idx])
            fig.add_annotation(
                x=dates[e_idx],
                y=y_val,
                text=trend_hint['text'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=trend_hint['arrowcolor'],
                ax=0,
                ay=trend_hint['ay'],
                font={'color': trend_hint['color']}
            )
    else:
        fig.update_layout(
            title=f'{symbol} No Clear Pattern Detected (Last 48 Hours)',
            xaxis_title='Date',
            yaxis_title='Price'
        )
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def _build_pattern_chart_48h_fallback_from_series(symbol: str, dates, closes, opens=None, highs=None, lows=None):
    if not dates or not closes:
        return None
    dates = dates[-48:] if len(dates) > 48 else dates
    closes = closes[-48:] if len(closes) > 48 else closes
    opens = opens[-48:] if opens and len(opens) > 48 else opens
    highs = highs[-48:] if highs and len(highs) > 48 else highs
    lows = lows[-48:] if lows and len(lows) > 48 else lows

    use_ohlc = bool(opens and highs and lows and len(opens) == len(closes) == len(highs) == len(lows))
    if use_ohlc:
        fig = go.Figure(data=[
            go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name="OHLC (Fallback)"
            )
        ])
    else:
        fig = go.Figure(data=[go.Scatter(x=dates, y=closes, mode='lines', name='Close (Fallback)')])

    pattern = detect_pattern(np.asarray(closes, dtype=float))
    if pattern:
        s_idx = pattern['start_idx']
        e_idx = pattern['end_idx']
        fig.update_layout(
            title=f"{symbol} {pattern['name']} Pattern (Last 48 Hours - Fallback)",
            xaxis_title='Date',
            yaxis_title='Price'
        )
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=dates[s_idx],
            x1=dates[e_idx],
            y0=0,
            y1=1,
            fillcolor="rgba(255, 193, 7, 0.2)",
            line_width=0
        )
    else:
        fig.update_layout(
            title=f'{symbol} No Clear Pattern Detected (Last 48 Hours - Fallback)',
            xaxis_title='Date',
            yaxis_title='Price'
        )
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def _normalize_company_info(symbol: str, info: dict | None):
    raw = info or {}
    name = raw.get('name') or raw.get('shortName') or raw.get('longName') or symbol
    sector = raw.get('sector')
    industry = raw.get('industry')
    summary = raw.get('summary') or raw.get('longBusinessSummary')
    if not summary:
        summary = f"Detailed company profile is currently unavailable for {symbol}. You can still use charts, predictions, and news."
    return {
        'name': name,
        'sector': sector,
        'industry': industry,
        'summary': summary
    }

def _upgrade_cached_payload(payload, symbol):
    if 'pattern_chart' not in payload and 'chart_3m' in payload:
        payload['pattern_chart'] = payload['chart_3m']
    dates = payload.get('actual_dates') or []
    closes = payload.get('actual_close') or []
    opens = payload.get('actual_open') or None
    highs = payload.get('actual_high') or None
    lows = payload.get('actual_low') or None
    # Older caches may have OHLC only inside chart trace; extract so 1M pattern can render as candlestick.
    if (not opens or not highs or not lows) and isinstance(payload.get('chart'), dict):
        for trace in payload['chart'].get('data', []):
            if (trace.get('type') or '').lower() == 'candlestick':
                if not dates:
                    dates = trace.get('x') or dates
                    payload['actual_dates'] = dates
                if not closes:
                    closes = trace.get('close') or closes
                    payload['actual_close'] = closes
                opens = trace.get('open') or opens
                highs = trace.get('high') or highs
                lows = trace.get('low') or lows
                payload['actual_open'] = opens
                payload['actual_high'] = highs
                payload['actual_low'] = lows
                break
    company_info = payload.get('company_info') or {}
    payload['company_info'] = _normalize_company_info(symbol, company_info)
    label = (payload.get('company_info') or {}).get('name') or symbol
    news = payload.get('news')
    if isinstance(news, list) and news and isinstance(news[0], dict) and 'content' in news[0]:
        normalized = []
        for item in news:
            content = item.get('content') or {}
            title = content.get('title')
            link = None
            canonical = content.get('canonicalUrl') or {}
            click = content.get('clickThroughUrl') or {}
            if isinstance(canonical, dict):
                link = canonical.get('url')
            if not link and isinstance(click, dict):
                link = click.get('url')
            provider = (content.get('provider') or {}).get('displayName')
            published = content.get('pubDate') or content.get('displayTime')
            summary = content.get('summary') or content.get('description') or ""
            sent = _sentiment_score(title, summary)
            if title and link:
                normalized.append({
                    'title': title,
                    'link': link,
                    'publisher': provider or '',
                    'published': published,
                    'sentiment': sent
                })
        payload['news'] = normalized
    elif not isinstance(news, list):
        payload['news'] = []
    # If cached news is empty, attempt a live refresh now (fast, no ML re-run).
    if not payload.get('news'):
        try:
            t = yf.Ticker(symbol)
            fresh = fetch_news(t, symbol, 5)
            if fresh:
                payload['news'] = fresh
        except Exception:
            pass
    # Backfill sentiment for already-normalized cached news items.
    for item in (payload.get('news') or []):
        if not isinstance(item, dict):
            continue
        if item.get('sentiment') is None:
            sent = _sentiment_score(
                item.get('title') or '',
                item.get('summary') or item.get('description') or ''
            )
            item['sentiment'] = sent
    if 'pattern_chart_48h' not in payload:
        payload['pattern_chart_48h'] = None
    if 'analyst_insights' not in payload:
        payload['analyst_insights'] = {
            'top_analysts': [],
            'recommendations': [],
            'top_analysts_source': 'Unavailable',
            'recommendations_source': 'Unavailable'
        }
    if 'predictions' not in payload or not isinstance(payload.get('predictions'), dict):
        payload['predictions'] = {}
    if dates and closes:
        need_pattern_rebuild = 'pattern_chart' not in payload
        if not need_pattern_rebuild and isinstance(payload.get('pattern_chart'), dict):
            traces = payload['pattern_chart'].get('data') or []
            has_candlestick = any((t.get('type') or '').lower() == 'candlestick' for t in traces if isinstance(t, dict))
            if not has_candlestick and opens and highs and lows:
                need_pattern_rebuild = True
        if need_pattern_rebuild:
            payload['pattern_chart'] = _build_pattern_chart_from_series(
                label,
                dates,
                closes,
                open_values=opens,
                high_values=highs,
                low_values=lows
            )
        if 'rsi_chart' not in payload:
            payload['rsi_chart'] = _build_rsi_chart_from_series(label, dates, closes)
    if payload.get('pattern_chart_48h') is None:
        try:
            df_1h = yf.Ticker(symbol).history(period='7d', interval='1h')
            if df_1h is not None and not df_1h.empty:
                payload['pattern_chart_48h'] = _build_pattern_chart_48h_from_df(symbol, df_1h.tail(48))
        except Exception:
            pass
    if payload.get('pattern_chart_48h') is None and dates and closes:
        payload['pattern_chart_48h'] = _build_pattern_chart_48h_fallback_from_series(
            symbol, dates, closes, opens=opens, highs=highs, lows=lows
        )
    insights = payload.get('analyst_insights') or {}
    if not (insights.get('top_analysts') or insights.get('recommendations')):
        try:
            payload['analyst_insights'] = fetch_analyst_insights(symbol)
        except Exception:
            pass
    if not payload.get('company_info_source'):
        payload['company_info_source'] = "Cache"
    info = payload.get('company_info') or {}
    needs_profile_refresh = (
        not info.get('summary')
        or info.get('name') == symbol
        or "currently unavailable" in str(info.get('summary', '')).lower()
    )
    if needs_profile_refresh:
        fetched_info, fetched_source = fetch_company_profile(symbol, stock_obj=None)
        if fetched_info:
            existing = payload.get('company_info') or {}
            merged = {
                'name': fetched_info.get('name') or existing.get('name') or symbol,
                'sector': fetched_info.get('sector') or existing.get('sector'),
                'industry': fetched_info.get('industry') or existing.get('industry'),
                'summary': fetched_info.get('summary') or existing.get('summary')
            }
            payload['company_info'] = _normalize_company_info(symbol, merged)
            payload['company_info_source'] = fetched_source
    payload['company_info'] = _normalize_company_info(symbol, payload.get('company_info'))
    ai = payload.get('analyst_insights') or {}
    ai.setdefault('top_analysts_source', 'Unavailable')
    ai.setdefault('recommendations_source', 'Unavailable')
    # Legacy cache reconciliation: if rows exist but source is missing/unavailable,
    # mark as cached Yahoo-derived data instead of showing "Unavailable".
    if (ai.get('top_analysts') or []) and str(ai.get('top_analysts_source', '')).strip().lower() in ("", "unavailable", "n/a"):
        ai['top_analysts_source'] = "Yahoo Finance (cached legacy)"
    if (ai.get('recommendations') or []) and str(ai.get('recommendations_source', '')).strip().lower() in ("", "unavailable", "n/a"):
        ai['recommendations_source'] = "Yahoo Finance (cached legacy)"
    payload['analyst_insights'] = ai
    # Same reconciliation for company profile source.
    info_summary = str((payload.get('company_info') or {}).get('summary') or "")
    if payload.get('company_info_source') in (None, "", "Unavailable", "Cache"):
        if info_summary and "currently unavailable" not in info_summary.lower():
            payload['company_info_source'] = "Yahoo Finance (cached legacy)"
        elif "currently unavailable" in info_summary.lower():
            payload['company_info_source'] = "Unavailable"
    return payload

def validate_stock_data(df):
    """Validate and preprocess stock data for the last year"""
    if df is None or df.empty:
        raise ValueError("No stock data available")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    cutoff = df.index.max() - pd.Timedelta(days=365)
    df = df.loc[df.index >= cutoff].copy()
    
    if len(df) < 250:  # Minimum trading days in a year
        raise ValueError(f"Insufficient data points. Required: 250, Available: {len(df)}")
    
    # Remove rows with zero or negative prices
    df = df[df['Close'] > 0]
    
    # Fill any remaining NaN values
    columns_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in columns_to_fill:
        df[col] = df[col].ffill()
    
    return df

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI on a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50)


def build_feature_frame(close_values: np.ndarray) -> pd.DataFrame:
    """Generate a richer feature set from closing prices only."""
    series = pd.Series(close_values, dtype=float)
    returns = series.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    log_returns = np.log1p(returns).replace([np.inf, -np.inf], 0).fillna(0)
    momentum = series.diff().fillna(0)
    sma_5 = series.rolling(window=5).mean()
    sma_10 = series.rolling(window=10).mean()
    sma_20 = series.rolling(window=20).mean()
    rsi_14 = compute_rsi(series, window=14)
    volatility_10 = returns.rolling(window=10).std()
    feature_df = pd.DataFrame({
        'close': series,
        'log_return': log_returns,
        'momentum': momentum,
        'sma_5': sma_5,
        'sma_10': sma_10,
        'sma_20': sma_20,
        'rsi_14': rsi_14,
        'volatility_10': volatility_10
    })
    return feature_df.ffill().bfill()


def prepare_feature_windows(close_values: np.ndarray, time_steps: int = 30):
    """Prepare windowed feature matrices and scalers for the models."""
    feature_df = build_feature_frame(close_values)
    if len(feature_df) <= time_steps:
        raise ValueError(f"Insufficient data for feature windows. Need > {time_steps} points.")

    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(feature_df)

    close_scaler = MinMaxScaler()
    scaled_close = close_scaler.fit_transform(feature_df[['close']])

    X, y = [], []
    for i in range(time_steps, len(feature_df)):
        window = scaled_features[i-time_steps:i]
        X.append(window.flatten())
        y.append(scaled_close[i][0])

    return np.array(X), np.array(y), feature_scaler, close_scaler


def _forecast_tree_recursive(model, close_history, feature_scaler, close_scaler, time_steps, steps):
    """Recursive multi-step forecast using engineered features and scaled targets."""
    history = list(close_history)
    feature_frame = build_feature_frame(history)
    window = feature_scaler.transform(feature_frame.iloc[-time_steps:])
    preds = []

    for _ in range(steps):
        next_scaled = model.predict(window.reshape(1, -1))
        val = float(np.clip(next_scaled.ravel()[0], 0.0, 1.0))
        next_close = close_scaler.inverse_transform([[val]])[0][0]
        preds.append(next_close)

        history.append(next_close)
        feature_frame = build_feature_frame(history)
        window = feature_scaler.transform(feature_frame.iloc[-time_steps:])

    return np.array(preds).reshape(-1, 1)


def train_prediction_models(close_values, forecast_horizon=30, time_steps=30):
    """Train multiple prediction models using engineered price features."""
    X, y, feature_scaler, close_scaler = prepare_feature_windows(close_values, time_steps)
    flat_X = X.reshape(X.shape[0], -1)
    predictions = {}

    # XGBoost Model (more estimators, lower learning rate for stability)
    xgb_model = xgb.XGBRegressor(
        n_estimators=320,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    xgb_model.fit(flat_X, y)
    xgb_forecast = _forecast_tree_recursive(
        xgb_model, close_values, feature_scaler, close_scaler, time_steps, forecast_horizon
    )
    predictions['XGBoost'] = xgb_forecast

    # ExtraTrees Model (controls to reduce overfitting on small windows)
    try:
        et_model = ExtraTreesRegressor(
            n_estimators=260,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=2
        )
        et_model.fit(flat_X, y)
        et_forecast = _forecast_tree_recursive(
            et_model, close_values, feature_scaler, close_scaler, time_steps, forecast_horizon
        )
        predictions['ExtraTrees'] = et_forecast
    except Exception as e:
        logger.error(f"ExtraTrees model error: {e}")

    # RandomForest Model
    try:
        rf_model = RandomForestRegressor(
            n_estimators=260,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=2
        )
        rf_model.fit(flat_X, y)
        rf_forecast = _forecast_tree_recursive(
            rf_model, close_values, feature_scaler, close_scaler, time_steps, forecast_horizon
        )
        predictions['RandomForest'] = rf_forecast
    except Exception as e:
        logger.error(f"RandomForest model error: {e}")

    return predictions

_DIVIDEND_REFRESH_LOCK = threading.Lock()
_DIVIDEND_REFRESH_IN_PROGRESS = False

def fetch_top_dividend_stocks(max_results: int = 5) -> list:
    """Return top dividend stocks sorted by yield DESC, cached daily.
    On first load with no cache, triggers a background fetch and returns [].
    """
    global _DIVIDEND_MEM, _DIVIDEND_MEM_DAY
    today_key = datetime.now(timezone.utc).date().isoformat()

    if _DIVIDEND_MEM is not None and _DIVIDEND_MEM_DAY == today_key:
        return _DIVIDEND_MEM

    try:
        if os.path.exists(DIVIDEND_CACHE_PATH):
            with open(DIVIDEND_CACHE_PATH, "r", encoding="utf-8") as fh:
                fc = json.load(fh)
            cached_stocks = fc.get("stocks", [])
            if fc.get("date") == today_key and cached_stocks:
                _DIVIDEND_MEM = cached_stocks
                _DIVIDEND_MEM_DAY = today_key
                return _DIVIDEND_MEM
    except Exception:
        pass

    # No valid cache — run fetch in background, return stale or empty for now
    _start_dividend_refresh_if_needed(max_results)
    return _DIVIDEND_MEM or []


def _start_dividend_refresh_if_needed(max_results: int = 5):
    global _DIVIDEND_REFRESH_IN_PROGRESS
    with _DIVIDEND_REFRESH_LOCK:
        if _DIVIDEND_REFRESH_IN_PROGRESS:
            return
        _DIVIDEND_REFRESH_IN_PROGRESS = True
    t = threading.Thread(target=_dividend_refresh_worker, args=(max_results,), daemon=True)
    t.start()


def _dividend_refresh_worker(max_results: int = 5):
    global _DIVIDEND_MEM, _DIVIDEND_MEM_DAY, _DIVIDEND_REFRESH_IN_PROGRESS
    try:
        today_key = datetime.now(timezone.utc).date().isoformat()
        stocks = _fetch_dividend_stocks_live(max_results)
        if stocks:
            try:
                os.makedirs(os.path.dirname(DIVIDEND_CACHE_PATH), exist_ok=True)
                with open(DIVIDEND_CACHE_PATH, "w", encoding="utf-8") as fh:
                    json.dump({"date": today_key, "stocks": stocks}, fh)
            except Exception as exc:
                logger.warning(f"Could not write dividend cache: {exc}")
            _DIVIDEND_MEM = stocks
            _DIVIDEND_MEM_DAY = today_key
    except Exception as exc:
        logger.warning(f"Dividend background refresh failed: {exc}")
    finally:
        with _DIVIDEND_REFRESH_LOCK:
            _DIVIDEND_REFRESH_IN_PROGRESS = False


_DIVIDEND_CANDIDATES = [
    "MO", "T", "VZ", "IBM", "CVX", "XOM", "PFE", "MMM", "KO", "PG",
    "JNJ", "ABBV", "PM", "MCD", "TGT", "WMT", "HD", "LOW", "CAT",
    "GPC", "SYY", "NEE", "SO", "DUK", "D", "ED", "O", "WBA", "HRL",
    "CLX", "KMB", "CL", "MDT", "ABT", "BDX", "AFL", "ADP", "ITW",
    "EMR", "DOV", "NUE", "SWK", "PPG", "SHW", "APD", "CINF", "CB",
    "GWW", "CTAS", "ATO", "BEN", "SPGI",
]

def _get_yahoo_crumb():
    """Return (session, crumb) using the same pattern as fetch_analyst_insights."""
    session_req = requests.Session()
    session_req.get("https://fc.yahoo.com", headers={"User-Agent": _USER_AGENT}, timeout=8)
    crumb_resp = session_req.get(
        "https://query2.finance.yahoo.com/v1/test/getcrumb",
        headers={"User-Agent": _USER_AGENT},
        timeout=8,
    )
    if not crumb_resp.ok or not crumb_resp.text.strip():
        return None, None
    return session_req, crumb_resp.text.strip()


def _fetch_dividend_stocks_live(max_results: int = 5) -> list:
    """Fetch summaryDetail for each candidate via quoteSummary (crumb-auth),
    sort by dividendYield DESC, return top max_results."""
    session_req, crumb = _get_yahoo_crumb()
    if not crumb:
        logger.warning("Dividend fetch: could not obtain Yahoo crumb")
        return []

    def _fetch_one(sym):
        try:
            url = (
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
                f"?modules=summaryDetail,price&crumb={crumb}"
            )
            resp = session_req.get(url, headers={"User-Agent": _USER_AGENT}, timeout=8)
            if not resp.ok:
                return None
            result = (resp.json().get("quoteSummary", {}).get("result") or [{}])[0]
            sd = result.get("summaryDetail", {})
            pr = result.get("price", {})

            div_yield = float((sd.get("dividendYield") or {}).get("raw", 0) or 0)
            div_rate  = float((sd.get("dividendRate")  or {}).get("raw", 0) or 0)
            price     = float((pr.get("regularMarketPrice") or {}).get("raw", 0) or 0)
            chg       = float((pr.get("regularMarketChangePercent") or {}).get("raw", 0) or 0)
            name      = (pr.get("longName") or pr.get("shortName") or sym)

            if div_yield <= 0:
                return None
            return {
                "symbol":     sym,
                "name":       name,
                "yield_pct":  round(div_yield * 100, 2),
                "annual_div": round(div_rate, 2),
                "price":      round(price, 2) if price else None,
                "change_pct": round(chg * 100, 2),
            }
        except Exception:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in _DIVIDEND_CANDIDATES}
        for fut in concurrent.futures.as_completed(futures):
            data = fut.result()
            if data:
                results.append(data)

    results.sort(key=lambda x: x["yield_pct"], reverse=True)
    return results[:max_results]


# ---------------------------------------------------------------------------
# Top Undervalued Stocks (by Free Cash Flow Yield)
# ---------------------------------------------------------------------------
_UNDERVALUED_CANDIDATES = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "BRK-B", "JPM", "BAC",
    "WFC", "C", "GS", "MS", "CVX", "XOM", "COP", "EOG", "MPC", "PSX",
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "CVS", "UNH",
    "HD", "LOW", "WMT", "TGT", "COST", "MCD", "SBUX", "NKE", "DIS",
    "CSCO", "INTC", "QCOM", "TXN", "AMAT", "MU", "HPQ", "IBM", "ORCL",
    "CAT", "DE", "MMM", "HON", "GE", "RTX", "LMT", "NOC", "BA",
    "F", "GM", "TSLA", "VZ", "T", "CMCSA", "CHTR", "NFLX",
]

def fetch_top_undervalued_stocks(max_results: int = 5) -> list:
    """Return top undervalued stocks by FCF yield, cached daily."""
    global _UNDERVALUED_MEM, _UNDERVALUED_MEM_DAY
    today_key = datetime.now(timezone.utc).date().isoformat()

    if _UNDERVALUED_MEM is not None and _UNDERVALUED_MEM_DAY == today_key:
        return _UNDERVALUED_MEM

    try:
        if os.path.exists(UNDERVALUED_CACHE_PATH):
            with open(UNDERVALUED_CACHE_PATH, "r", encoding="utf-8") as fh:
                fc = json.load(fh)
            cached_stocks = fc.get("stocks", [])
            if fc.get("date") == today_key and cached_stocks:
                _UNDERVALUED_MEM = cached_stocks
                _UNDERVALUED_MEM_DAY = today_key
                return _UNDERVALUED_MEM
    except Exception:
        pass

    _start_undervalued_refresh_if_needed(max_results)
    return _UNDERVALUED_MEM or []


def _start_undervalued_refresh_if_needed(max_results: int = 5):
    global _UNDERVALUED_REFRESH_IN_PROGRESS
    with _UNDERVALUED_REFRESH_LOCK:
        if _UNDERVALUED_REFRESH_IN_PROGRESS:
            return
        _UNDERVALUED_REFRESH_IN_PROGRESS = True
    t = threading.Thread(target=_undervalued_refresh_worker, args=(max_results,), daemon=True)
    t.start()


def _undervalued_refresh_worker(max_results: int = 5):
    global _UNDERVALUED_MEM, _UNDERVALUED_MEM_DAY, _UNDERVALUED_REFRESH_IN_PROGRESS
    try:
        today_key = datetime.now(timezone.utc).date().isoformat()
        stocks = _fetch_undervalued_stocks_live(max_results)
        if stocks:
            try:
                os.makedirs(os.path.dirname(UNDERVALUED_CACHE_PATH), exist_ok=True)
                with open(UNDERVALUED_CACHE_PATH, "w", encoding="utf-8") as fh:
                    json.dump({"date": today_key, "stocks": stocks}, fh)
            except Exception as exc:
                logger.warning(f"Could not write undervalued cache: {exc}")
            _UNDERVALUED_MEM = stocks
            _UNDERVALUED_MEM_DAY = today_key
    except Exception as exc:
        logger.warning(f"Undervalued background refresh failed: {exc}")
    finally:
        with _UNDERVALUED_REFRESH_LOCK:
            _UNDERVALUED_REFRESH_IN_PROGRESS = False


def _fetch_undervalued_stocks_live(max_results: int = 5) -> list:
    """Fetch financialData + price via quoteSummary for each candidate.
    Rank by FCF yield (freeCashflow / marketCap) DESC — positive FCF only."""
    session_req, crumb = _get_yahoo_crumb()
    if not crumb:
        logger.warning("Undervalued fetch: could not obtain Yahoo crumb")
        return []

    def _fetch_one(sym):
        try:
            url = (
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
                f"?modules=financialData,price,defaultKeyStatistics&crumb={crumb}"
            )
            resp = session_req.get(url, headers={"User-Agent": _USER_AGENT}, timeout=8)
            if not resp.ok:
                return None
            result = (resp.json().get("quoteSummary", {}).get("result") or [{}])[0]
            fd = result.get("financialData", {})
            pr = result.get("price", {})
            ks = result.get("defaultKeyStatistics", {})

            fcf      = float((fd.get("freeCashflow")        or {}).get("raw", 0) or 0)
            mcap     = float((pr.get("marketCap")           or {}).get("raw", 0) or 0)
            price    = float((pr.get("regularMarketPrice")  or {}).get("raw", 0) or 0)
            chg      = float((pr.get("regularMarketChangePercent") or {}).get("raw", 0) or 0)
            pfcf     = float((ks.get("priceToFreeCashflows") or {}).get("raw", 0) or 0)
            name     = pr.get("longName") or pr.get("shortName") or sym

            if fcf <= 0 or mcap <= 0:
                return None

            fcf_yield = round(fcf / mcap * 100, 2)
            if fcf_yield < 1.0:   # skip negligible yields
                return None

            return {
                "symbol":    sym,
                "name":      name,
                "fcf_yield": fcf_yield,
                "pfcf":      round(pfcf, 1) if pfcf > 0 else None,
                "fcf":       fcf,
                "mcap":      mcap,
                "price":     round(price, 2) if price else None,
                "change_pct": round(chg * 100, 2),
            }
        except Exception:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in _UNDERVALUED_CANDIDATES}
        for fut in concurrent.futures.as_completed(futures):
            data = fut.result()
            if data:
                results.append(data)

    results.sort(key=lambda x: x["fcf_yield"], reverse=True)
    return results[:max_results]


@app.route('/')
def home():
    stocks = fetch_sp500_stocks()
    remarkables = get_remarkables(force_refresh=False)
    new_listings = fetch_top_dividend_stocks()
    undervalued_stocks = fetch_top_undervalued_stocks()
    user = None
    if google_client_id and google_client_secret and google.authorized:
        try:
            resp = google.get("/oauth2/v2/userinfo")
            if resp.ok:
                user = resp.json()
                if user.get("email"):
                    session["user_email"] = user.get("email")
        except TokenExpiredError:
            if google_bp and google_bp.token:
                del google_bp.token
            session.pop("user_email", None)
    user_email = session.get("user_email")
    if user_email and user is None:
        user = {"email": user_email}
    user_tier = _get_user_tier(user_email) if user_email else "free"
    subscribed = request.args.get("subscribed") == "1"
    return render_template(
        'index.html',
        stocks=stocks,
        user=user,
        remarkables=remarkables,
        dividend_stocks=new_listings,
        undervalued_stocks=undervalued_stocks,
        user_tier=user_tier,
        user_email=user_email or "",
        subscribed=subscribed,
        stripe_enabled=bool(stripe.api_key),
        free_daily_limit=FREE_DAILY_LIMIT,
    )

@app.route('/api/remarkables', methods=['GET'])
def remarkables_api():
    refresh = request.args.get('refresh') == '1'
    payload = get_remarkables(force_refresh=refresh)
    return jsonify(payload)

@app.route("/logout")
def logout():
    if google_bp and google_bp.token:
        del google_bp.token
    session.clear()
    return redirect(url_for("home"))

@app.route("/login/email", methods=["POST"])
def login_email():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400
    users = _load_users()
    info = users.get(email)
    if not info:
        return jsonify({"error": "Invalid email or password."}), 401
    hashed = info["password_hash"] if isinstance(info, dict) else info
    if not check_password_hash(hashed, password):
        return jsonify({"error": "Invalid email or password."}), 401
    session["user_email"] = email
    tier = info.get("tier", "free") if isinstance(info, dict) else "free"
    return jsonify({"ok": True, "tier": tier})

@app.route("/register/email", methods=["POST"])
def register_email():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"error": "Invalid email address."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    users = _load_users()
    if email in users:
        return jsonify({"error": "An account with this email already exists."}), 409
    users[email] = {"password_hash": generate_password_hash(password), "tier": "free"}
    _save_users(users)
    session["user_email"] = email
    return jsonify({"ok": True, "tier": "free"})

@app.route("/api/watchlist", methods=["GET", "POST"])
def watchlist_api():
    email = _get_current_user_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    tier = _get_user_tier(email)
    if TIER_RANK.get(tier, 0) < TIER_RANK["pro"]:
        return jsonify({"error": "upgrade_required", "required_tier": "pro"}), 403
    path = _get_watchlist_path(email)
    if request.method == "GET":
        if not os.path.exists(path):
            return jsonify({"symbols": []})
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            symbols = payload.get("symbols") if isinstance(payload, dict) else []
            return jsonify({"symbols": symbols if isinstance(symbols, list) else []})
        except Exception as exc:
            logger.warning(f"Failed to read watchlist for {email}: {exc}")
            return jsonify({"symbols": []})
    data = request.get_json(silent=True) or {}
    symbols = data.get("symbols")
    if not isinstance(symbols, list):
        return jsonify({"error": "symbols must be a list"}), 400
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"symbols": symbols}, handle)
        return jsonify({"symbols": symbols})
    except Exception as exc:
        logger.error(f"Failed to write watchlist for {email}: {exc}")
        return jsonify({"error": "failed to save watchlist"}), 500

# ---------------------------------------------------------------------------
# AI Trade Thesis
# ---------------------------------------------------------------------------
_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

def generate_trade_thesis(symbol: str, company_info: dict, current_price: float,
                          predictions: dict, analyst_payload: dict, news: list,
                          sentiment_avg: float | None) -> dict:
    """Call Claude Haiku to generate a bull/bear/verdict trade thesis."""
    if not _ANTHROPIC_API_KEY:
        return {"error": "AI thesis not configured (missing ANTHROPIC_API_KEY)"}

    # Build a compact context string — keep tokens low
    avg_pred = None
    pred_changes = []
    for preds in predictions.values():
        if preds:
            chg = (preds[-1] - current_price) / current_price * 100
            pred_changes.append(round(chg, 2))
    if pred_changes:
        avg_pred = round(sum(pred_changes) / len(pred_changes), 2)

    reco = analyst_payload.get("recommendations") or []
    latest_reco = reco[0] if reco else {}
    analyst_summary = ""
    if latest_reco:
        sb = latest_reco.get("strongBuy", 0)
        b  = latest_reco.get("buy", 0)
        h  = latest_reco.get("hold", 0)
        s  = latest_reco.get("sell", 0) + latest_reco.get("strongSell", 0)
        analyst_summary = f"Analyst consensus: {sb} strong buy, {b} buy, {h} hold, {s} sell."

    news_titles = "; ".join([n["title"] for n in (news or [])[:3] if n.get("title")])
    sector = company_info.get("sector") or "Unknown"
    industry = company_info.get("industry") or "Unknown"

    data_lines = [f"- Current price: ${current_price:.2f}"]
    if avg_pred is not None:
        data_lines.append(f"- ML 30-day avg forecast change: {avg_pred:+.2f}%")
    if analyst_summary:
        data_lines.append(f"- {analyst_summary}")
    data_lines.append(f"- News sentiment score: {sentiment_avg if sentiment_avg is not None else 'N/A'}")
    data_lines.append(f"- Recent headlines: {news_titles or 'none'}")

    prompt = (
        f"You are a concise equity analyst. Write a trade thesis for {symbol} "
        f"({sector} / {industry}) in exactly three short sections:\n"
        f"1. BULL CASE (2 sentences max)\n"
        f"2. BEAR CASE (2 sentences max)\n"
        f"3. VERDICT (1 sentence — buy / hold / sell and why)\n\n"
        f"Data:\n"
        + "\n".join(data_lines)
        + "\n\nBe direct. No disclaimers. No markdown headers — just the plain section labels."
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": _ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if not resp.ok:
            logger.warning(f"Trade thesis API error for {symbol}: {resp.status_code} {resp.text}")
            return {"error": "Could not generate thesis. Please try again."}
        text = resp.json()["content"][0]["text"].strip()
        return {"thesis": text}
    except Exception as e:
        logger.warning(f"Trade thesis API error for {symbol}: {e}")
        return {"error": "Could not generate thesis. Please try again."}


# ---------------------------------------------------------------------------
# Peer Comparison
# ---------------------------------------------------------------------------

def _fetch_ticker_fundamentals(sym: str) -> dict | None:
    """Fetch full fundamentals for one ticker. Returns None on failure."""
    try:
        t = yf.Ticker(sym)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        mkt_cap = info.get("marketCap")
        if not price or not mkt_cap:
            return None

        hist = t.history(period="1y")
        if hist.empty:
            return None

        close = hist["Close"]
        price_now   = float(close.iloc[-1])
        price_1w    = float(close.iloc[-5])  if len(close) >= 5  else None
        price_1m    = float(close.iloc[-21]) if len(close) >= 21 else None
        price_3m    = float(close.iloc[-63]) if len(close) >= 63 else None
        price_start = float(close.iloc[0])

        def pct(a, b):
            return round((a - b) / b * 100, 2) if b else None

        week_52_high = info.get("fiftyTwoWeekHigh")
        week_52_low  = info.get("fiftyTwoWeekLow")
        off_high = pct(price_now, week_52_high) if week_52_high else None

        return {
            "symbol":          sym,
            "name":            info.get("shortName") or sym,
            "sector":          info.get("sector"),
            "industry":        info.get("industry"),
            "market_cap":      int(mkt_cap),
            "price":           round(price_now, 2),
            "chg_1w":          pct(price_now, price_1w),
            "chg_1m":          pct(price_now, price_1m),
            "chg_3m":          pct(price_now, price_3m),
            "chg_ytd":         pct(price_now, price_start),
            "pe_trailing":     round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            "pe_forward":      round(info.get("forwardPE"), 2)  if info.get("forwardPE")  else None,
            "revenue_growth":  round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") is not None else None,
            "profit_margin":   round(info.get("profitMargins", 0) * 100, 1) if info.get("profitMargins") is not None else None,
            "roe":             round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") is not None else None,
            "week_52_high":    week_52_high,
            "week_52_low":     week_52_low,
            "off_52w_high":    off_high,
        }
    except Exception:
        return None


def fetch_peers(symbol: str, sector: str | None) -> dict:
    """
    Find genuine peers by matching industry + market-cap proximity from the
    S&P 500 universe. Returns target fundamentals + up to 4 peer fundamentals.
    """
    # Step 1 — fetch target fundamentals
    target = _fetch_ticker_fundamentals(symbol)
    if not target:
        return {"target": None, "peers": []}

    target_industry = target.get("industry")
    target_mktcap   = target.get("market_cap") or 0

    # Step 2 — build candidate list from S&P 500 + NASDAQ pool
    all_symbols = fetch_sp500_stocks()
    candidates = [s for s in all_symbols if s != symbol]

    # Step 3 — score candidates: fetch info in parallel, filter by industry,
    #           rank by log market-cap distance to target
    import math

    def _score(sym):
        try:
            t = yf.Ticker(sym)
            info = t.info
            ind = info.get("industry")
            sec = info.get("sector")
            mkt = info.get("marketCap") or 0
            if not mkt:
                return None
            # Industry match scores better than sector-only match
            if ind and ind == target_industry:
                match = "industry"
            elif sec and sec == target.get("sector"):
                match = "sector"
            else:
                return None  # different sector — skip
            # Log-scale market cap distance (smaller = closer in size)
            dist = abs(math.log10(mkt + 1) - math.log10(target_mktcap + 1))
            return (sym, match, dist, mkt)
        except Exception:
            return None

    scored = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_score, s): s for s in candidates[:120]}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result:
                scored.append(result)

    # Sort: industry matches first, then by market-cap proximity
    scored.sort(key=lambda x: (0 if x[1] == "industry" else 1, x[2]))

    # Step 4 — fetch full fundamentals for top candidates until we have 4 peers
    peers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_fetch_ticker_fundamentals, s[0]) for s in scored[:12]]
        for fut in concurrent.futures.as_completed(futures):
            data = fut.result()
            if data and len(peers) < 4:
                peers.append(data)

    # Re-sort peers by market cap proximity to target for display
    peers.sort(key=lambda p: abs(
        math.log10(p["market_cap"] + 1) - math.log10(target_mktcap + 1)
    ))

    return {"target": target, "peers": peers[:4]}


def _avg_sentiment(news: list) -> float | None:
    scores = [n["sentiment"] for n in (news or []) if n.get("sentiment") is not None]
    return round(sum(scores) / len(scores), 1) if scores else None


def _sentiment_score(*parts: str):
    text = " ".join([p for p in parts if p]).strip()
    if not text:
        return None
    try:
        compound = SENTIMENT_ANALYZER.polarity_scores(text).get('compound', 0.0)
        return int(max(-100, min(100, compound * 100)))
    except Exception:
        return None

def _parse_yf_news_item(n: dict) -> dict | None:
    """Parse a single yfinance news item regardless of old vs new API shape."""
    # New shape (yfinance >= 0.2.50): all data under n['content']
    content = n.get('content') or {}
    if content:
        title = content.get('title')
        summary = content.get('summary') or content.get('description') or ''
        publisher = (content.get('provider') or {}).get('displayName') or ''
        pub_date = content.get('pubDate') or content.get('displayTime') or ''
        click = content.get('clickThroughUrl') or content.get('canonicalUrl') or {}
        link = click.get('url') if isinstance(click, dict) else None
    else:
        # Legacy shape: fields at top level
        title = n.get('title')
        summary = n.get('summary') or n.get('description') or ''
        publisher = n.get('publisher') or (n.get('provider') or {}).get('displayName') or ''
        pub_date = n.get('providerPublishTime') or ''
        link = n.get('link') or n.get('url')

    if not title or not link:
        return None
    return {
        'title': title,
        'link': link,
        'publisher': publisher,
        'published': pub_date,
        'sentiment': _sentiment_score(title, summary),
    }


def fetch_news(stock_obj, symbol, limit=5):
    items = []

    # --- Primary: yfinance ---
    if stock_obj is not None:
        try:
            raw = stock_obj.news or []
            for n in raw:
                parsed = _parse_yf_news_item(n)
                if parsed:
                    items.append(parsed)
                if len(items) >= limit:
                    break
        except Exception as e:
            logger.warning(f"YF news fetch failed for {symbol}: {e}")

    # --- Fallback: Yahoo Finance search API ---
    if len(items) < limit:
        try:
            url = (
                f"https://query2.finance.yahoo.com/v1/finance/search"
                f"?q={symbol}&newsCount={limit}&enableFuzzyQuery=false"
            )
            resp = requests.get(url, timeout=8, headers={"User-Agent": _USER_AGENT})
            if resp.ok:
                for n in (resp.json().get('news') or []):
                    parsed = _parse_yf_news_item(n)
                    if parsed:
                        items.append(parsed)
                    if len(items) >= limit:
                        break
        except Exception as e:
            logger.warning(f"YF search news fetch failed for {symbol}: {e}")

    # --- Dedupe by title ---
    seen: set = set()
    deduped = []
    for it in items:
        t = it.get('title')
        if t and t not in seen:
            seen.add(t)
            deduped.append(it)
        if len(deduped) >= limit:
            break
    return deduped

def run_prediction(symbol: str):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {'error': 'Please provide a symbol'}, 400
    try:
        stock = yf.Ticker(symbol)
        data_source = "yahoo"
        try:
            df = fetch_with_retry(symbol, period='1y', attempts=3, delay=2)
        except Exception as exc:
            cached = load_cached_response(symbol)
            if cached:
                logger.info(f"Serving cached response for {symbol}")
                return cached, 200
            if _is_rate_limit_error(exc):
                try:
                    df = fetch_stooq_history(symbol)
                    stock = None
                    data_source = "stooq"
                    logger.info(f"Using Stooq fallback for {symbol} due to rate limit")
                except Exception as fallback_exc:
                    logger.warning(f"Stooq fallback failed for {symbol}: {fallback_exc}")
                    raise exc
            else:
                raise exc
        df = validate_stock_data(df)

        forecast_horizon = 30
        predictions = train_prediction_models(
            df['Close'].values,
            forecast_horizon=forecast_horizon,
            time_steps=30
        )

        # Build dates
        actual_dates = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in df.index]
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
        future_dates_iso = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in future_dates]

        # Normalize prediction lengths
        normalized_predictions = {}
        for model_name, pred in predictions.items():
            arr = np.asarray(pred).reshape(-1)
            if len(arr) != len(future_dates):
                if len(arr) < len(future_dates):
                    pad = np.full(len(future_dates) - len(arr), arr[-1] if len(arr) else np.nan)
                    arr = np.hstack([arr, pad])
                else:
                    arr = arr[:len(future_dates)]
            normalized_predictions[model_name] = arr.tolist()

        # Plotly chart serialized safely (1y and 3mo)
        candlestick = go.Candlestick(
            x=actual_dates,
            open=df['Open'].astype(float).tolist(),
            high=df['High'].astype(float).tolist(),
            low=df['Low'].astype(float).tolist(),
            close=df['Close'].astype(float).tolist(),
            name="OHLC (1Y)"
        )
        prediction_traces = [
            go.Scatter(
                x=future_dates_iso,
                y=vals,
                mode='lines',
                name=f'{model_name} Prediction'
            ) for model_name, vals in normalized_predictions.items()
        ]
        fig = go.Figure(data=[candlestick] + prediction_traces)
        fig.update_layout(
            title=f'{symbol} Stock Price Prediction (Last 1 Year)',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        chart_json = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

        # Pattern detection chart (last ~1 month)
        df_1m = df.tail(22) if len(df) > 22 else df.copy()
        last1_dates = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in df_1m.index]
        pattern_chart_json = _build_pattern_chart_from_series(
            symbol, last1_dates,
            df_1m['Close'].astype(float).tolist(),
            open_values=df_1m['Open'].astype(float).tolist(),
            high_values=df_1m['High'].astype(float).tolist(),
            low_values=df_1m['Low'].astype(float).tolist()
        )

        # Pattern detection chart (last 48 hours, hourly)
        pattern_chart_48h_json = None
        try:
            if stock is not None:
                df_1h = stock.history(period='7d', interval='1h')
                if df_1h is not None and not df_1h.empty:
                    pattern_chart_48h_json = _build_pattern_chart_48h_from_df(symbol, df_1h.tail(48))
        except Exception as exc:
            logger.warning(f"48h pattern chart failed for {symbol}: {exc}")
        if pattern_chart_48h_json is None:
            pattern_chart_48h_json = _build_pattern_chart_48h_fallback_from_series(
                symbol,
                actual_dates,
                df['Close'].astype(float).tolist(),
                opens=df['Open'].astype(float).tolist(),
                highs=df['High'].astype(float).tolist(),
                lows=df['Low'].astype(float).tolist()
            )

        # RSI chart (last ~1 month)
        rsi_chart_json = _build_rsi_chart_from_series(
            symbol, last1_dates, df_1m['Close'].astype(float).tolist()
        )

        # 3M predictions removed in favor of pattern detection chart
        preds_3m = {}
        future_dates_3m_iso = []

        # Fetch news, company info, and analyst insights in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            news_future = pool.submit(fetch_news, stock, symbol, 5)
            profile_future = pool.submit(fetch_company_profile, symbol, stock)
            analyst_future = pool.submit(fetch_analyst_insights, symbol)
        news = news_future.result()
        company_info, company_info_source = profile_future.result()
        analyst_payload = analyst_future.result()
        response = {
            'chart': chart_json,
            'pattern_chart': pattern_chart_json,
            'pattern_chart_48h': pattern_chart_48h_json,
            'rsi_chart': rsi_chart_json,
            'data_source': data_source,
            'analyst_insights': analyst_payload,
            'predictions': normalized_predictions,
            'predictions_3m': preds_3m,
            'current_price': float(df['Close'].iloc[-1]),
            'news': news,
            'company_info': company_info,
            'company_info_source': company_info_source,
            'actual_dates': actual_dates,
            'actual_close': df['Close'].astype(float).tolist(),
            'actual_open': df['Open'].astype(float).tolist(),
            'actual_high': df['High'].astype(float).tolist(),
            'actual_low': df['Low'].astype(float).tolist(),
            'future_dates': future_dates_iso,
            'future_dates_3m': future_dates_3m_iso
        }
        save_cached_response(symbol, response)
        return response, 200
    except HTTPError as he:
        status = getattr(getattr(he, "response", None), "status_code", None)
        if status == 429:
            logger.warning(f"Rate limited when fetching {symbol}: {he}")
            return {'error': 'Data provider rate limit hit. Please retry in a minute.'}, 503
        logger.error(f"HTTP error fetching data for {symbol}: {he}")
        return {'error': 'Unable to fetch data from provider.'}, 502
    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return {'error': str(ve)}, 400
    except Exception as e:
        if "Too Many Requests" in str(e):
            logger.warning(f"Rate limited when fetching {symbol}: {e}")
            return {'error': 'Data provider rate limit hit. Please retry in a minute.'}, 503
        logger.error(f"Unexpected error: {e}")
        return {'error': 'An unexpected error occurred'}, 500

@app.route('/predict', methods=['POST'])
def predict():
    email = _get_current_user_email()
    if email:
        allowed, used, limit = _check_and_increment_daily_usage(email)
        if not allowed:
            return jsonify({
                "error": "You've used all 5 free analyses for today. Upgrade to Pro for unlimited access.",
                "upgrade_required": True,
                "used_today": used,
                "daily_limit": limit,
            }), 429
    else:
        anon_used = session.get("anon_analyses", 0)
        if anon_used >= ANON_DAILY_LIMIT:
            return jsonify({
                "error": f"You've used your {ANON_DAILY_LIMIT} free analyses. Sign in for more, or upgrade to Pro.",
                "upgrade_required": True,
                "sign_in_required": True,
                "used_today": anon_used,
                "daily_limit": ANON_DAILY_LIMIT,
            }), 429
        session["anon_analyses"] = anon_used + 1

    resp, status = run_prediction(request.form.get('symbol'))
    if status == 200:
        tier = _get_user_tier(email) if email else "free"
        today = datetime.now(timezone.utc).date().isoformat()
        if email:
            users = _load_users()
            info = users.get(email, {})
            used_now = info.get("daily_usage", {}).get(today, 0) if isinstance(info, dict) else 0
        else:
            used_now = session.get("anon_analyses", 0)
        resp["_meta"] = {
            "tier": tier,
            "used_today": used_now,
            "daily_limit": FREE_DAILY_LIMIT if tier == "free" else -1,
        }
        return jsonify(resp)
    return jsonify(resp), status


@app.route("/api/trade-thesis", methods=["POST"])
@subscription_required("pro")
def api_trade_thesis():
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    cached = load_cached_response(symbol)
    if not cached:
        return jsonify({"error": "Run a prediction for this symbol first."}), 400

    company_info = cached.get("company_info") or {}
    current_price = cached.get("current_price") or 0.0
    predictions   = cached.get("predictions") or {}
    analyst       = cached.get("analyst_insights") or {}
    news          = cached.get("news") or []
    sentiment_avg = _avg_sentiment(news)

    result = generate_trade_thesis(
        symbol, company_info, current_price, predictions, analyst, news, sentiment_avg
    )
    return jsonify(result)


@app.route("/api/earnings-summary", methods=["POST"])
@subscription_required("premium")
def api_earnings_summary():
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    result = generate_earnings_summary(symbol)
    return jsonify(result)


def generate_earnings_summary(symbol: str) -> dict:
    """Fetch earnings data via yfinance and summarise with Claude Haiku."""
    if not _ANTHROPIC_API_KEY:
        return {"error": "AI features not configured (missing ANTHROPIC_API_KEY)"}

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # ── Upcoming / most-recent earnings date ──
        earnings_date = None
        cal = ticker.calendar
        if cal is not None:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    earnings_date = str(ed[0]) if isinstance(ed, (list, tuple)) else str(ed)
            elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                earnings_date = str(cal["Earnings Date"].iloc[0])

        # ── Historical EPS (last 4 quarters) ──
        eps_lines = []
        try:
            eh = ticker.earnings_history
            if eh is not None and not eh.empty:
                for _, row in eh.head(4).iterrows():
                    date_lbl = str(row.name)[:10] if hasattr(row.name, "__str__") else ""
                    est = row.get("epsEstimate") if "epsEstimate" in row else None
                    act = row.get("epsActual")   if "epsActual"   in row else None
                    surprise = row.get("epsDifference") if "epsDifference" in row else None
                    parts = [f"Q ending {date_lbl}"]
                    if est  is not None: parts.append(f"est ${est:.2f}")
                    if act  is not None: parts.append(f"actual ${act:.2f}")
                    if surprise is not None:
                        parts.append(f"surprise {'+' if surprise >= 0 else ''}{surprise:.2f}")
                    eps_lines.append(", ".join(parts))
        except Exception:
            pass

        # ── Annual revenue & earnings trend ──
        rev_lines = []
        try:
            fin = ticker.financials
            if fin is not None and not fin.empty:
                for col in list(fin.columns)[:3]:
                    year = str(col)[:4]
                    rev = fin.loc["Total Revenue", col] if "Total Revenue" in fin.index else None
                    net = fin.loc["Net Income", col]    if "Net Income"    in fin.index else None
                    parts = [year]
                    if rev is not None: parts.append(f"rev ${rev/1e9:.1f}B")
                    if net is not None: parts.append(f"net ${net/1e9:.1f}B")
                    rev_lines.append(", ".join(parts))
        except Exception:
            pass

        # ── Forward estimates from info dict ──
        fwd_eps    = info.get("forwardEps")
        fwd_pe     = info.get("forwardPE")
        rev_growth = info.get("revenueGrowth")
        earn_growth= info.get("earningsGrowth")
        sector     = info.get("sector") or "Unknown"
        industry   = info.get("industry") or "Unknown"
        name       = info.get("longName") or symbol

        data_lines = [f"Company: {name} ({sector} / {industry})"]
        if earnings_date:
            data_lines.append(f"Next earnings date: {earnings_date}")
        if fwd_eps  is not None: data_lines.append(f"Forward EPS estimate: ${fwd_eps:.2f}")
        if fwd_pe   is not None: data_lines.append(f"Forward P/E: {fwd_pe:.1f}x")
        if rev_growth   is not None: data_lines.append(f"Revenue growth (YoY): {rev_growth*100:.1f}%")
        if earn_growth  is not None: data_lines.append(f"Earnings growth (YoY): {earn_growth*100:.1f}%")
        if eps_lines:
            data_lines.append("Recent EPS history:")
            data_lines.extend(f"  {l}" for l in eps_lines)
        if rev_lines:
            data_lines.append("Annual financials:")
            data_lines.extend(f"  {l}" for l in rev_lines)

    except Exception as exc:
        logger.warning(f"Earnings data fetch failed for {symbol}: {exc}")
        return {"error": "Could not fetch earnings data. Please try again."}

    prompt = (
        f"You are a financial analyst. Based on the earnings data below, write a concise "
        f"earnings summary for {symbol} in exactly three sections:\n"
        f"1. EARNINGS TREND — 2 sentences on recent EPS/revenue trajectory\n"
        f"2. UPCOMING OUTLOOK — 2 sentences on forward estimates and what to watch\n"
        f"3. INVESTOR TAKEAWAY — 1 sentence verdict on whether earnings support the current valuation\n\n"
        f"Data:\n" + "\n".join(data_lines) +
        "\n\nBe direct and specific. No disclaimers. No markdown. Just plain section labels."
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": _ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 350,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if not resp.ok:
            logger.warning(f"Earnings summary API error for {symbol}: {resp.status_code} {resp.text}")
            return {"error": "Could not generate summary. Please try again."}
        text = resp.json()["content"][0]["text"].strip()
        return {"summary": text}
    except Exception as e:
        logger.warning(f"Earnings summary error for {symbol}: {e}")
        return {"error": "Could not generate summary. Please try again."}


@app.route("/api/portfolio-advisor", methods=["POST"])
@subscription_required("premium")
def api_portfolio_advisor():
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    cached = load_cached_response(symbol)
    if not cached:
        return jsonify({"error": "Run a prediction for this symbol first."}), 400

    company_info  = cached.get("company_info") or {}
    current_price = cached.get("current_price") or 0.0
    predictions   = cached.get("predictions") or {}
    analyst       = cached.get("analyst_insights") or {}
    news          = cached.get("news") or []
    sentiment_avg = _avg_sentiment(news)

    result = generate_portfolio_advice(
        symbol, company_info, current_price, predictions, analyst, news, sentiment_avg
    )
    return jsonify(result)


def generate_portfolio_advice(symbol: str, company_info: dict, current_price: float,
                               predictions: dict, analyst_payload: dict, news: list,
                               sentiment_avg: float | None) -> dict:
    """Call Claude Haiku to generate portfolio sizing and risk advice."""
    if not _ANTHROPIC_API_KEY:
        return {"error": "AI advisor not configured (missing ANTHROPIC_API_KEY)"}

    pred_changes = []
    for preds in predictions.values():
        if preds:
            chg = (preds[-1] - current_price) / current_price * 100
            pred_changes.append(round(chg, 2))
    avg_pred = round(sum(pred_changes) / len(pred_changes), 2) if pred_changes else None

    reco = analyst_payload.get("recommendations") or []
    latest_reco = reco[0] if reco else {}
    analyst_summary = ""
    if latest_reco:
        sb = latest_reco.get("strongBuy", 0)
        b  = latest_reco.get("buy", 0)
        h  = latest_reco.get("hold", 0)
        s  = latest_reco.get("sell", 0) + latest_reco.get("strongSell", 0)
        analyst_summary = f"{sb} strong buy, {b} buy, {h} hold, {s} sell"

    sector   = company_info.get("sector") or "Unknown"
    industry = company_info.get("industry") or "Unknown"
    beta     = company_info.get("beta")
    pe_ratio = company_info.get("pe_ratio") or company_info.get("trailingPE")
    mktcap   = company_info.get("market_cap") or company_info.get("marketCap")

    data_lines = [f"- Symbol: {symbol} ({sector} / {industry})"]
    data_lines.append(f"- Current price: ${current_price:.2f}")
    if avg_pred is not None:
        data_lines.append(f"- ML 30-day avg forecast change: {avg_pred:+.2f}%")
    if beta is not None:
        data_lines.append(f"- Beta: {beta}")
    if pe_ratio:
        data_lines.append(f"- P/E ratio: {pe_ratio}")
    if mktcap:
        data_lines.append(f"- Market cap: ${mktcap:,}")
    if analyst_summary:
        data_lines.append(f"- Analyst consensus: {analyst_summary}")
    data_lines.append(f"- News sentiment: {sentiment_avg if sentiment_avg is not None else 'N/A'}")

    prompt = (
        f"You are a portfolio advisor. Based on the data below, give practical portfolio guidance "
        f"for {symbol} in exactly four short sections:\n"
        f"1. POSITION SIZE — recommended portfolio allocation % and why (1-2 sentences)\n"
        f"2. RISK PROFILE — key risks and volatility outlook (2 sentences)\n"
        f"3. ENTRY STRATEGY — when/how to build a position (1-2 sentences)\n"
        f"4. EXIT STRATEGY — target price or conditions to sell/trim (1-2 sentences)\n\n"
        f"Data:\n"
        + "\n".join(data_lines)
        + "\n\nBe direct and specific. No disclaimers. No markdown. Just plain section labels."
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": _ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if not resp.ok:
            logger.warning(f"Portfolio advisor API error for {symbol}: {resp.status_code} {resp.text}")
            return {"error": "Could not generate advice. Please try again."}
        text = resp.json()["content"][0]["text"].strip()
        return {"advice": text}
    except Exception as e:
        logger.warning(f"Portfolio advisor error for {symbol}: {e}")
        return {"error": "Could not generate advice. Please try again."}


@app.route("/api/peers", methods=["POST"])
@subscription_required("pro")
def api_peers():
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    cached = load_cached_response(symbol)
    sector = (cached.get("company_info") or {}).get("sector") if cached else None

    result = fetch_peers(symbol, sector)
    return jsonify(result)


@app.route("/api/me")
def api_me():
    email = _get_current_user_email()
    if not email:
        return jsonify({"logged_in": False, "tier": "free"})
    tier = _get_user_tier(email)
    users = _load_users()
    info = users.get(email, {})
    today = datetime.now(timezone.utc).date().isoformat()
    used_today = info.get("daily_usage", {}).get(today, 0) if isinstance(info, dict) else 0
    return jsonify({
        "logged_in": True,
        "email": email,
        "tier": tier,
        "used_today": used_today,
        "daily_limit": FREE_DAILY_LIMIT if tier == "free" else -1,
    })

@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not stripe.api_key:
        return jsonify({"error": "Payments not configured on this server."}), 503
    email = _get_current_user_email()
    if not email:
        return jsonify({"error": "Login required"}), 401
    data = request.get_json(silent=True) or {}
    tier = data.get("tier", "pro")
    price_id = STRIPE_PREMIUM_PRICE_ID if tier == "premium" else STRIPE_PRO_PRICE_ID
    if not price_id:
        return jsonify({"error": "Price not configured"}), 503
    try:
        checkout = stripe.checkout.Session.create(
            customer_email=email,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=url_for("home", _external=True) + "?subscribed=1",
            cancel_url=url_for("home", _external=True),
            metadata={"email": email},
        )
        return jsonify({"url": checkout.url})
    except Exception as e:
        logger.error(f"Stripe checkout error: {e}")
        return jsonify({"error": "Failed to create checkout session"}), 500

@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        return "", 400
    sig = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(request.data, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.warning(f"Stripe webhook signature error: {e}")
        return "", 400

    event_type = event["type"]
    sub = event["data"]["object"]
    customer_id = sub.get("customer")
    users = _load_users()

    def _apply_sub(info, sub_obj):
        items = (sub_obj.get("items") or {}).get("data") or [{}]
        price_id = (items[0].get("price") or {}).get("id", "")
        info["tier"] = "premium" if price_id == STRIPE_PREMIUM_PRICE_ID else "pro"
        info["subscription_status"] = sub_obj.get("status", "active")
        info["current_period_end"] = sub_obj.get("current_period_end", 0)
        info["stripe_subscription_id"] = sub_obj.get("id")
        info["stripe_customer_id"] = customer_id

    matched = False
    for email, info in users.items():
        if isinstance(info, dict) and info.get("stripe_customer_id") == customer_id:
            if event_type in ("customer.subscription.created", "customer.subscription.updated"):
                _apply_sub(info, sub)
            elif event_type == "customer.subscription.deleted":
                info["tier"] = "free"
                info["subscription_status"] = "canceled"
            elif event_type == "invoice.payment_failed":
                info["subscription_status"] = "past_due"
            matched = True
            break

    if not matched and event_type in ("customer.subscription.created", "customer.subscription.updated"):
        # First-time subscriber: look up email from Stripe customer object
        try:
            customer = stripe.Customer.retrieve(customer_id)
            customer_email = (customer.get("email") or "").lower()
        except Exception:
            customer_email = ""
        if customer_email and customer_email in users:
            info = users[customer_email]
            if not isinstance(info, dict):
                info = {"password_hash": info, "tier": "free"}
            _apply_sub(info, sub)
            users[customer_email] = info

    _save_users(users)
    return "", 200

@app.route("/billing-portal")
def billing_portal():
    if not stripe.api_key:
        return redirect(url_for("home"))
    email = _get_current_user_email()
    if not email:
        return redirect(url_for("home"))
    users = _load_users()
    info = users.get(email, {})
    customer_id = info.get("stripe_customer_id") if isinstance(info, dict) else None
    if not customer_id:
        return redirect(url_for("home"))
    try:
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=url_for("home", _external=True),
        )
        return redirect(portal.url)
    except Exception as e:
        logger.error(f"Stripe billing portal error: {e}")
        return redirect(url_for("home"))

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8080'))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)