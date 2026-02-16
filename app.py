import os
import logging
import json
import time
import re
import threading
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
from io import BytesIO, StringIO

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from flask_dance.contrib.google import make_google_blueprint, google
from oauthlib.oauth2 import TokenExpiredError

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

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
    empty_payload = {"top_analysts": [], "recommendations": []}
    def _has_rows(payload):
        return bool((payload.get("top_analysts") or []) or (payload.get("recommendations") or []))

    def _fallback_from_yfinance(sym: str):
        out = {"top_analysts": [], "recommendations": []}
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
        except Exception:
            pass

        return out

    try:
        session_req = requests.Session()
        session_req.get("https://fc.yahoo.com", headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        crumb_resp = session_req.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            headers={"User-Agent": "Mozilla/5.0"},
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
        resp = session_req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
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
                date_str = datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d")
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

        direct_payload = {"top_analysts": top_analysts, "recommendations": recommendations}
        if _has_rows(direct_payload):
            return direct_payload
        fallback = _fallback_from_yfinance(symbol)
        return fallback if _has_rows(fallback) else empty_payload
    except Exception as exc:
        logger.warning(f"Analyst insights fetch failed for {symbol}: {exc}")
        fallback = _fallback_from_yfinance(symbol)
        return fallback if _has_rows(fallback) else empty_payload

WATCHLIST_DIR = os.path.join(os.path.dirname(__file__), "data", "watchlists")
REMARKABLES_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "remarkables_nasdaq.json")
REMARKABLES_CACHE_TTL_SECONDS = 12 * 60 * 60
REMARKABLES_BATCH_SIZE = 25
REMARKABLES_RULE_VERSION = "2026-02-13-v6-risk-near-fill-up1.5"
REMARKABLES_REFRESH_LOCK = threading.Lock()
REMARKABLES_REFRESH_IN_PROGRESS = False
REMARKABLES_MEMORY_CACHE = None
REMARKABLES_MEMORY_DAY = None

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
    resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
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
    resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
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

def _remarkables_cache_is_fresh(payload: dict) -> bool:
    if (payload or {}).get("rule_version") != REMARKABLES_RULE_VERSION:
        return False
    ts = (payload or {}).get("updated_at")
    if not ts:
        return False
    try:
        parsed = datetime.fromisoformat(ts)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
        return age.total_seconds() < REMARKABLES_CACHE_TTL_SECONDS
    except Exception:
        return False

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

            # No Pain But Gain: +10% in 6M, and no daily drop >9%
            if trend_6m >= 10.0 and (daily_loss >= -0.09).all():
                steady_candidates.append({
                    "symbol": sym,
                    "trend_percent": round(trend_6m, 2),
                    "max_daily_drop_percent": round(max_daily_drop, 2),
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
            up_hits, down_hits = _prevday_highlow_hits(close_3m, high_3m, low_3m, up_mult=1.5, down_mult=0.75)
            # For Risk Lovers:
            # - trend >= +15% (3M)
            # - >=2 days where today's LOW <= 75% of previous day's CLOSE
            # - >=3 days where today's HIGH >= 150% of previous day's CLOSE
            if trend_3m >= 15.0 and down_hits >= 2 and up_hits >= 3:
                risk_candidates.append({
                    "symbol": sym,
                    "trend_percent": round(trend_3m, 2),
                    "loss_hits": down_hits,
                    "gain_hits": up_hits,
                    "down_hits": down_hits,
                    "up_hits": up_hits,
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
    steady_sorted = steady_sorted[:5]
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
            steady_sorted.extend([x for x in local_steady if x["symbol"] not in used])
            used = {x["symbol"] for x in steady_sorted}
            if len(steady_sorted) < 5:
                steady_sorted.extend([x for x in local_steady_near if x["symbol"] not in used])
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
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
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
            if trend_6m >= 10.0 and (daily >= -0.09).all():
                steady_strict.append({
                    "symbol": symbol,
                    "trend_percent": round(trend_6m, 2),
                    "max_daily_drop_percent": round(max_drop, 2),
                    "score": trend_6m - max_drop,
                    "match_type": "cache_strict"
                })
            elif trend_6m >= 8.0 and (daily >= -0.12).all():
                steady_near.append({
                    "symbol": symbol,
                    "trend_percent": round(trend_6m, 2),
                    "max_daily_drop_percent": round(max_drop, 2),
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
            up_hits, down_hits = _prevday_highlow_hits(s3, hs3, ls3, up_mult=1.5, down_mult=0.75)
            if trend_3m >= 15.0 and down_hits >= 2 and up_hits >= 3:
                risk_strict.append({
                    "symbol": symbol,
                    "trend_percent": round(trend_3m, 2),
                    "loss_hits": down_hits,
                    "gain_hits": up_hits,
                    "down_hits": down_hits,
                    "up_hits": up_hits,
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
        if len(risk_list) < 5 or len(steady_list) < 5:
            local_risk, local_steady, local_risk_near, local_steady_near = _compute_remarkables_from_local_cache()
            if len(risk_list) < 5 and (local_risk or local_risk_near):
                used = {x.get("symbol") for x in risk_list}
                risk_list.extend([x for x in local_risk if x.get("symbol") not in used])
                used = {x.get("symbol") for x in risk_list}
                if len(risk_list) < 5:
                    risk_list.extend([x for x in local_risk_near if x.get("symbol") not in used])
                cached["for_risk_lovers"] = risk_list[:5]
            if len(steady_list) < 5 and (local_steady or local_steady_near):
                used = {x.get("symbol") for x in steady_list}
                steady_list.extend([x for x in local_steady if x.get("symbol") not in used])
                used = {x.get("symbol") for x in steady_list}
                if len(steady_list) < 5:
                    steady_list.extend([x for x in local_steady_near if x.get("symbol") not in used])
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
    payload["no_pain_but_gain"] = (local_steady + local_steady_near)[:5]
    payload["source"] = "warming_up_daily_refresh"
    return payload

def load_cached_response(symbol: str):
    """Load cached API-style response if available."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
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
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
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

def _build_rsi_chart_from_series(label, dates, close_values):
    if not dates or not close_values:
        return None
    window = 22
    dates = dates[-window:] if len(dates) > window else dates
    close_values = close_values[-window:] if len(close_values) > window else close_values
    rsi_series = compute_rsi(pd.Series(close_values, dtype=float), window=14)
    rsi_values = rsi_series.astype(float).tolist()
    fig = go.Figure(data=[
        go.Scatter(x=dates, y=rsi_values, mode='lines', name='RSI (14)')
    ])
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_signal = None
    last_rsi = rsi_values[-1] if rsi_values else None
    if last_rsi is not None:
        if last_rsi >= 70:
            rsi_signal = {'text': 'Sell', 'color': 'red', 'ay': -40, 'arrowcolor': 'red'}
        elif last_rsi <= 30:
            rsi_signal = {'text': 'Buy', 'color': 'green', 'ay': 40, 'arrowcolor': 'green'}
    rsi_title_suffix = 'Neutral' if rsi_signal is None else rsi_signal['text']
    fig.update_layout(
        title=f'{label} RSI (Last 1 Month) - {rsi_title_suffix}',
        xaxis_title='Date',
        yaxis_title='RSI'
    )
    if rsi_signal and dates:
        fig.add_annotation(
            x=dates[-1],
            y=last_rsi,
            text=rsi_signal['text'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor=rsi_signal['arrowcolor'],
            ax=0,
            ay=rsi_signal['ay'],
            font={'color': rsi_signal['color']}
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
    label = company_info.get('name') or symbol
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
            if title and link:
                normalized.append({
                    'title': title,
                    'link': link,
                    'publisher': provider or '',
                    'published': published,
                    'sentiment': None
                })
        payload['news'] = normalized
    elif not isinstance(news, list):
        payload['news'] = []
    if 'pattern_chart_48h' not in payload:
        payload['pattern_chart_48h'] = None
    if 'analyst_insights' not in payload:
        payload['analyst_insights'] = {'top_analysts': [], 'recommendations': []}
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
    info = payload.get('company_info') or {}
    if not info.get('summary') or info.get('name') == symbol:
        try:
            yf_info = yf.Ticker(symbol).info
            if isinstance(yf_info, dict):
                merged = {
                    'name': yf_info.get('shortName') or yf_info.get('longName') or info.get('name') or symbol,
                    'sector': yf_info.get('sector') or info.get('sector'),
                    'industry': yf_info.get('industry') or info.get('industry'),
                    'summary': yf_info.get('longBusinessSummary') or info.get('summary')
                }
                payload['company_info'] = _normalize_company_info(symbol, merged)
        except Exception:
            pass
    payload['company_info'] = _normalize_company_info(symbol, payload.get('company_info'))
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

def create_lstm_model(input_shape):
    """Create LSTM model with functional API"""
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

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

@app.route('/')
def home():
    stocks = fetch_sp500_stocks()
    remarkables = get_remarkables(force_refresh=False)
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
    return render_template('index.html', stocks=stocks, user=user, remarkables=remarkables)

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

@app.route("/api/watchlist", methods=["GET", "POST"])
def watchlist_api():
    email = _get_current_user_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
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

def _sentiment_score(*parts: str):
    text = " ".join([p for p in parts if p]).strip()
    if not text:
        return None
    try:
        compound = SENTIMENT_ANALYZER.polarity_scores(text).get('compound', 0.0)
        return int(max(-100, min(100, compound * 100)))
    except Exception:
        return None

def fetch_news(stock_obj, symbol, limit=5):
    items = []
    try:
        raw = stock_obj.news or []
        for n in raw[:limit]:
            title = n.get('title')
            link = n.get('link') or n.get('url')
            publisher = n.get('publisher') or n.get('provider', {}).get('displayName')
            published = n.get('providerPublishTime')
            summary = n.get('summary') or n.get('description') or ''
            sent = _sentiment_score(title, summary)
            if title and link:
                items.append({
                    'title': title,
                    'link': link,
                    'publisher': publisher or '',
                    'published': published,
                    'sentiment': sent
                })
    except Exception as e:
        logger.warning(f"YF news fetch failed for {symbol}: {e}")

    if len(items) < limit:
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            resp = requests.get(rss_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok and resp.text:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.text)
                for item in root.findall('.//item'):
                    title = (item.findtext('title') or '').strip()
                    link = (item.findtext('link') or '').strip()
                    pubdate = (item.findtext('pubDate') or '').strip()
                    description = (item.findtext('description') or '').strip()
                    sent = _sentiment_score(title, description)
                    if title and link:
                        items.append({
                            'title': title,
                            'link': link,
                            'publisher': 'Yahoo Finance',
                            'published': pubdate,
                            'sentiment': sent
                        })
                    if len(items) >= limit:
                        break
        except Exception as e:
            logger.warning(f"RSS news fetch failed for {symbol}: {e}")

    # dedupe by title
    seen = set()
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
        fig_pattern = go.Figure(data=[
            go.Candlestick(
                x=last1_dates,
                open=df_1m['Open'].astype(float).tolist(),
                high=df_1m['High'].astype(float).tolist(),
                low=df_1m['Low'].astype(float).tolist(),
                close=df_1m['Close'].astype(float).tolist(),
                name="OHLC (Last 1 Month)"
            )
        ])
        pattern = detect_pattern(df_1m['Close'].astype(float).values)
        if pattern:
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            trend_hint = _trend_hint_for_pattern(pattern['name'])
            fig_pattern.update_layout(
                title=f"{symbol} {pattern['name']} Pattern (Last 1 Month)",
                xaxis_title='Date',
                yaxis_title='Price'
            )
            fig_pattern.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=last1_dates[start_idx],
                x1=last1_dates[end_idx],
                y0=0,
                y1=1,
                fillcolor="rgba(255, 193, 7, 0.2)",
                line_width=0
            )
            if trend_hint and trend_hint['text'] != 'Neutral':
                y_val = float(df_1m['Close'].iloc[end_idx])
                fig_pattern.add_annotation(
                    x=last1_dates[end_idx],
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
            fig_pattern.update_layout(
                title=f'{symbol} No Clear Pattern Detected (Last 1 Month)',
                xaxis_title='Date',
                yaxis_title='Price'
            )
        pattern_chart_json = json.loads(json.dumps(fig_pattern, cls=PlotlyJSONEncoder))

        # Pattern detection chart (last 48 hours, hourly)
        pattern_chart_48h_json = None
        try:
            if stock is not None:
                df_1h = stock.history(period='7d', interval='1h')
            else:
                df_1h = None
            if df_1h is not None and not df_1h.empty:
                df_48h = df_1h.tail(48)
                last48_dates = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in df_48h.index]
                fig_pattern_48h = go.Figure(data=[
                    go.Candlestick(
                        x=last48_dates,
                        open=df_48h['Open'].astype(float).tolist(),
                        high=df_48h['High'].astype(float).tolist(),
                        low=df_48h['Low'].astype(float).tolist(),
                        close=df_48h['Close'].astype(float).tolist(),
                        name="OHLC (Last 48 Hours)"
                    )
                ])
                pattern_48h = detect_pattern(df_48h['Close'].astype(float).values)
                if pattern_48h:
                    s_idx = pattern_48h['start_idx']
                    e_idx = pattern_48h['end_idx']
                    fig_pattern_48h.update_layout(
                        title=f"{symbol} {pattern_48h['name']} Pattern (Last 48 Hours)",
                        xaxis_title='Date',
                        yaxis_title='Price'
                    )
                    fig_pattern_48h.add_shape(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=last48_dates[s_idx],
                        x1=last48_dates[e_idx],
                        y0=0,
                        y1=1,
                        fillcolor="rgba(255, 193, 7, 0.2)",
                        line_width=0
                    )
                    trend_hint_48h = _trend_hint_for_pattern(pattern_48h['name'])
                    if trend_hint_48h and trend_hint_48h['text'] != 'Neutral':
                        y_val = float(df_48h['Close'].iloc[e_idx])
                        fig_pattern_48h.add_annotation(
                            x=last48_dates[e_idx],
                            y=y_val,
                            text=trend_hint_48h['text'],
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.2,
                            arrowwidth=2,
                            arrowcolor=trend_hint_48h['arrowcolor'],
                            ax=0,
                            ay=trend_hint_48h['ay'],
                            font={'color': trend_hint_48h['color']}
                        )
                else:
                    fig_pattern_48h.update_layout(
                        title=f'{symbol} No Clear Pattern Detected (Last 48 Hours)',
                        xaxis_title='Date',
                        yaxis_title='Price'
                    )
                pattern_chart_48h_json = json.loads(json.dumps(fig_pattern_48h, cls=PlotlyJSONEncoder))
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
        rsi_series = compute_rsi(df_1m['Close'].astype(float), window=14)
        rsi_values = rsi_series.astype(float).tolist()
        rsi_dates = last1_dates
        rsi_fig = go.Figure(data=[
            go.Scatter(
                x=rsi_dates,
                y=rsi_values,
                mode='lines',
                name='RSI (14)'
            )
        ])
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")

        rsi_signal = None
        last_rsi = rsi_values[-1] if rsi_values else None
        if last_rsi is not None:
            if last_rsi >= 70:
                rsi_signal = {'text': 'Sell', 'color': 'red', 'ay': 40, 'arrowcolor': 'red'}
            elif last_rsi <= 30:
                rsi_signal = {'text': 'Buy', 'color': 'green', 'ay': -40, 'arrowcolor': 'green'}

        rsi_title_suffix = 'Neutral' if rsi_signal is None else rsi_signal['text']
        rsi_fig.update_layout(
            title=f'{symbol} RSI (Last 1 Month) - {rsi_title_suffix}',
            xaxis_title='Date',
            yaxis_title='RSI'
        )
        if rsi_signal and rsi_dates:
            rsi_fig.add_annotation(
                x=rsi_dates[-1],
                y=last_rsi,
                text=rsi_signal['text'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=rsi_signal['arrowcolor'],
                ax=0,
                ay=rsi_signal['ay'],
                font={'color': rsi_signal['color']}
            )
        rsi_chart_json = json.loads(json.dumps(rsi_fig, cls=PlotlyJSONEncoder))

        # 3M predictions removed in favor of pattern detection chart
        preds_3m = {}
        future_dates_3m_iso = []

        # Fetch news with sentiment
        news = fetch_news(stock, symbol, limit=5)

        # Fetch company info
        info = stock.info if hasattr(stock, "info") else {}
        company_info = _normalize_company_info(symbol, {
            'name': info.get('shortName') or info.get('longName') or symbol,
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'summary': info.get('longBusinessSummary')
        })

        analyst_payload = fetch_analyst_insights(symbol)
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
    resp, status = run_prediction(request.form.get('symbol'))
    if status == 200:
        return jsonify(resp)
    return jsonify(resp), status

def _build_excel(symbol: str, payload: dict) -> BytesIO:
    buffer = BytesIO()
    future_dates = payload.get('future_dates') or []
    pred_rows = []
    for model, values in (payload.get('predictions') or {}).items():
        for idx, val in enumerate(values):
            date_val = future_dates[idx] if idx < len(future_dates) else None
            pred_rows.append({
                'Model': model,
                'Date': date_val,
                'Predicted Price': val
            })
    pred_df = pd.DataFrame(pred_rows)

    actual_df = pd.DataFrame({
        'Date': payload.get('actual_dates') or [],
        'Close': payload.get('actual_close') or []
    })

    summary_df = pd.DataFrame([
        {'Metric': 'Symbol', 'Value': symbol},
        {'Metric': 'Current Price', 'Value': payload.get('current_price')}
    ])

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        actual_df.to_excel(writer, index=False, sheet_name='Actual_Close')
        pred_df.to_excel(writer, index=False, sheet_name='Predictions')

    buffer.seek(0)
    return buffer

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    symbol = (request.form.get('symbol') or '').strip().upper()
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400
    resp, status = run_prediction(symbol)
    if status != 200:
        return jsonify(resp), status
    excel_stream = _build_excel(symbol, resp)
    filename = f"{symbol}_forecast.xlsx"
    return send_file(
        excel_stream,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8080'))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
