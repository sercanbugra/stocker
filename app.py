import os
import logging
import json
import time
import re
import urllib.parse
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import requests
from requests.exceptions import HTTPError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from plotly.utils import PlotlyJSONEncoder
from io import BytesIO

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from flask_dance.contrib.google import make_google_blueprint, google

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

WATCHLIST_DIR = os.path.join(os.path.dirname(__file__), "data", "watchlists")

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

def _upgrade_cached_payload(payload, symbol):
    if 'pattern_chart' not in payload and 'chart_3m' in payload:
        payload['pattern_chart'] = payload['chart_3m']
    dates = payload.get('actual_dates') or []
    closes = payload.get('actual_close') or []
    opens = payload.get('actual_open') or None
    highs = payload.get('actual_high') or None
    lows = payload.get('actual_low') or None
    company_info = payload.get('company_info') or {}
    if not company_info:
        company_info = {'name': symbol, 'sector': None, 'industry': None, 'summary': None}
    payload['company_info'] = company_info
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
    if dates and closes:
        if 'pattern_chart' not in payload:
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
    return rsi.fillna(method='bfill').fillna(50)


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
    user = None
    if google_client_id and google_client_secret and google.authorized:
        resp = google.get("/oauth2/v2/userinfo")
        if resp.ok:
            user = resp.json()
            if user.get("email"):
                session["user_email"] = user.get("email")
    return render_template('index.html', stocks=stocks, user=user)

@app.route("/logout")
def logout():
    if google_bp and google_bp.token:
        del google_bp.token
    session.clear()
    home_url = request.url_root.rstrip("/") + url_for("home")
    continue_url = "https://appengine.google.com/_ah/logout?continue=" + urllib.parse.quote(home_url, safe="")
    logout_url = "https://accounts.google.com/Logout?continue=" + urllib.parse.quote(continue_url, safe="")
    return redirect(logout_url)

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
        try:
            df = fetch_with_retry(symbol, period='1y', attempts=3, delay=2)
        except Exception as exc:
            cached = load_cached_response(symbol)
            if cached:
                logger.info(f"Serving cached response for {symbol}")
                return cached, 200
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
            df_1h = stock.history(period='7d', interval='1h')
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
        company_info = {
            'name': info.get('shortName') or info.get('longName') or symbol,
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'summary': info.get('longBusinessSummary')
        }

        response = {
            'chart': chart_json,
            'pattern_chart': pattern_chart_json,
            'pattern_chart_48h': pattern_chart_48h_json,
            'rsi_chart': rsi_chart_json,
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
    app.run(debug=True) 
