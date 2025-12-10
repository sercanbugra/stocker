import os
import logging
import time
import json
import numpy as np
import random as pyrandom
import pandas as pd
import yfinance as yf
import xgboost as xgb
import requests
from plotly.utils import PlotlyJSONEncoder
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file
from pandas.tseries.offsets import BDay
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Simple in-memory cache for predictions to avoid heavy retraining on each request
# Keys are stock symbols; values include timestamp and prepared response parts
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes
PREDICTION_CACHE = {}
# Optional disk cache for last successful responses (per symbol)
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
# bump this to invalidate old/buggy cache formats
CACHE_VERSION = "4"

# Per-symbol cooldown after rate limiting (epoch seconds)
RATE_LIMIT_COOLDOWN = {}

def fetch_sp500_stocks():
    """Fetch S&P 500 stocks with robust error handling and local fallback.

    Priority:
    1) Local pinned list at data/sp500_symbols.txt
    2) Wikipedia scrape
    3) Small hardcoded fallback list
    """
    # 1) Local pinned list
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

    # 2) Wikipedia
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        stocks = df['Symbol'].astype(str).str.strip().tolist()[:1500]
        logger.info(f"Successfully fetched {len(stocks)} stocks from Wikipedia")
        return stocks
    except Exception as e:
        logger.error(f"Error fetching stocks from Wikipedia: {e}")

    # 3) Fallback
    return [
        'TGTX', 'EXEL', 'POET', 'PNR', 'PTGX', 'LNTH', 'BWA'
    ]

def validate_stock_data(df):
    """Validate and preprocess stock data for the last 6 months.

    - Restrict to last 180 days
    - Require a reasonable minimum history (>= 100 points)
    - Enforce positive close prices
    - Forward-fill remaining NaNs safely
    """
    if df is None or df.empty:
        raise ValueError("No stock data available")

    # Ensure data is for the last ~6 months using an explicit date mask to avoid tz issues/warnings
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    cutoff = df.index.max() - pd.Timedelta(days=180)
    df = df.loc[df.index >= cutoff].copy()

    if len(df) < 100:
        raise ValueError(f"Insufficient data points. Required: 100, Available: {len(df)}")

    # Remove rows with zero or negative prices
    df = df.loc[df['Close'] > 0].copy()

    # Fill any remaining NaN values
    columns_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in columns_to_fill:
        if col in df.columns:
            df.loc[:, col] = df[col].ffill()

    return df


def select_price_series(df: pd.DataFrame) -> pd.Series:
    """Select the best price series for modeling (Adj Close if available)."""
    if 'Adj Close' in df.columns and df['Adj Close'].notna().any():
        return df['Adj Close']
    return df['Close']


def fetch_history_with_retry(symbol, period='1y', retries=3, backoff=2.0):
    """Fetch price history via yfinance with basic retry/backoff.

    Retries on transient errors and 429 Too Many Requests.
    """
    last_exc = None
    import random
    for attempt in range(retries):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
            # Fallback to yf.download which uses a different internal path
            df2 = yf.download(symbol, period=period, progress=False, threads=False, auto_adjust=True)
            if df2 is not None and not df2.empty:
                return df2
            last_exc = ValueError("Empty history returned")
        except Exception as e:
            last_exc = e

        msg = str(last_exc)
        # Exponential backoff on rate limits; small backoff otherwise
        if 'Too Many Requests' in msg or '429' in msg or 'rate limit' in msg.lower():
            jitter = random.uniform(0.0, 0.5)
            sleep_s = max(1.0, backoff ** attempt) + jitter
            logger.warning(f"Rate limited fetching {symbol}; retrying in {sleep_s:.1f}s (attempt {attempt+1}/{retries})")
            time.sleep(sleep_s)
        else:
            time.sleep(min(1.0, 0.3 * (attempt + 1)))

    raise last_exc


def _load_disk_cache(symbol):
    try:
        path = os.path.join(CACHE_DIR, f"{symbol.upper()}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as jde:
                logger.warning(f"Corrupted disk cache for {symbol}; removing file. Error: {jde}")
                try:
                    os.remove(path)
                except Exception:
                    pass
                return None
    except Exception as e:
        logger.warning(f"Failed to load disk cache for {symbol}: {e}")
    return None


def _save_disk_cache(symbol, response):
    try:
        path = os.path.join(CACHE_DIR, f"{symbol.upper()}.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        logger.warning(f"Failed to save disk cache for {symbol}: {e}")


def fetch_news(symbol, limit=5):
    """Fetch latest news with yfinance, fall back to Yahoo Finance RSS."""
    items = []
    # Primary: yfinance news payload
    try:
        stock_obj = yf.Ticker(symbol)
        raw = stock_obj.news or []
        for n in raw[:limit]:
            title = n.get('title')
            link = n.get('link') or n.get('url')
            publisher = n.get('publisher') or n.get('provider', {}).get('displayName')
            ts = n.get('providerPublishTime')
            if title and link:
                items.append({
                    'title': title,
                    'link': link,
                    'publisher': publisher or '',
                    'published': ts,
                })
    except Exception as ne:
        logger.warning(f"YF news fetch failed for {symbol}: {ne}")

    # Fallback: Yahoo Finance RSS
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
                    if title and link:
                        items.append({
                            'title': title,
                            'link': link,
                            'publisher': 'Yahoo Finance',
                            'published': pubdate
                        })
                    if len(items) >= limit:
                        break
        except Exception as re:
            logger.warning(f"RSS news fetch failed for {symbol}: {re}")

    # Deduplicate by title
    seen = set()
    deduped = []
    for it in items:
        title = it.get('title')
        if title and title not in seen:
            seen.add(title)
            deduped.append(it)
        if len(deduped) >= limit:
            break
    return deduped

def prepare_time_series_data(data, time_steps=30):
    """Prepare scaled sliding-window data for time series models."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.asarray(data).reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps])

    return np.array(X), np.array(y), scaler, scaled_data

def _forecast_xgb_recursive(model, last_window_scaled, steps):
    """Recursive multi-step forecast for XGBoost using the last scaled window."""
    window = last_window_scaled.copy().ravel()
    preds = []
    for _ in range(steps):
        next_scaled = model.predict(window.reshape(1, -1))
        val = float(next_scaled.ravel()[0])
        val = max(0.0, min(1.0, val))
        preds.append(val)
        window = np.hstack([window[1:], [val]])
    return np.array(preds).reshape(-1, 1)


def _forecast_tree_recursive(model, last_window_scaled, steps):
    """Recursive multi-step forecast for tree-based regressors using the last scaled window."""
    window = last_window_scaled.copy().ravel()
    preds = []
    for _ in range(steps):
        next_scaled = model.predict(window.reshape(1, -1))
        val = float(next_scaled.ravel()[0])
        val = max(0.0, min(1.0, val))
        preds.append(val)
        window = np.hstack([window[1:], [val]])
    return np.array(preds).reshape(-1, 1)

def train_prediction_models(data, dates, steps_ahead=30, time_steps=30):
    """Train multiple prediction models and produce consistent future forecasts.

    - XGBoost, ExtraTrees, and RandomForest use recursive multi-step forecasting from the last observed window.
    Returns dict of model -> (steps_ahead, 1) array of predicted prices.
    """
    X, y, scaler, scaled_data = prepare_time_series_data(data, time_steps)

    predictions = {}

    # Set seeds for reproducibility across runs
    np.random.seed(42)
    pyrandom.seed(42)

    # XGBoost Model with recursive forecast (tuned a bit for stability)
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.0,
            objective='reg:squarederror'
        )
        xgb_model.fit(X.reshape(X.shape[0], -1), y.ravel())
        last_window = scaled_data[-time_steps:]
        xgb_scaled = _forecast_xgb_recursive(xgb_model, last_window, steps_ahead)
        predictions['XGBoost'] = scaler.inverse_transform(xgb_scaled)
    except Exception as e:
        logger.error(f"XGBoost model error: {e}")

    # ExtraTrees Regressor with recursive forecast (fast ensemble baseline)
    try:
        et_model = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(X.reshape(X.shape[0], -1), y.ravel())
        last_window = scaled_data[-time_steps:]
        et_scaled = _forecast_tree_recursive(et_model, last_window, steps_ahead)
        predictions['ExtraTrees'] = scaler.inverse_transform(et_scaled)
    except Exception as e:
        logger.error(f"ExtraTrees model error: {e}")

    # RandomForest Regressor with recursive forecast
    try:
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X.reshape(X.shape[0], -1), y.ravel())
        last_window = scaled_data[-time_steps:]
        rf_scaled = _forecast_tree_recursive(rf_model, last_window, steps_ahead)
        predictions['RandomForest'] = scaler.inverse_transform(rf_scaled)
    except Exception as e:
        logger.error(f"RandomForest model error: {e}")

    return predictions


def run_prediction(symbol: str):
    """Core prediction pipeline shared by JSON and Excel endpoints."""
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return {'error': 'No symbol provided'}, 400

    try:
        # Cache check
        cached = PREDICTION_CACHE.get(symbol)
        now = time.time()
        # Respect cooldown if set due to recent rate limiting
        cooldown_until = RATE_LIMIT_COOLDOWN.get(symbol)
        if cooldown_until and now < cooldown_until:
            if cached:
                resp = {k: v for k, v in cached.items() if k not in ('timestamp', 'version')}
                resp['stale'] = True
                logger.info(f"Cooldown active for {symbol}; returning stale memory cache")
                return resp, 200
            disk_cached = _load_disk_cache(symbol)
            if disk_cached and disk_cached.get('version') == CACHE_VERSION:
                disk_cached['stale'] = True
                logger.info(f"Cooldown active for {symbol}; returning stale disk cache")
                return disk_cached, 200
            retry_after = int(cooldown_until - now)
            return {'error': f'Rate limited. Try again in {retry_after}s'}, 429

        if cached and cached.get('version') == CACHE_VERSION and (now - cached.get('timestamp', 0) < CACHE_TTL_SECONDS):
            logger.info(f"Cache hit for {symbol}")
            return {k: v for k, v in cached.items() if k not in ('timestamp', 'version')}, 200

        # Fetch and validate stock data for the last 6 months (with retries)
        df = fetch_history_with_retry(symbol, period='6mo')
        df = validate_stock_data(df)
        price_series = select_price_series(df)

        # Predict using multiple models
        predictions = train_prediction_models(
            price_series.values,
            df.index,
            steps_ahead=30,
            time_steps=30
        )

        # Fetch news (best-effort; tolerate rate limits)
        news = fetch_news(symbol, limit=5)
        # Normalize dates to naive ISO strings for Excel/JSON friendliness
        def _to_naive_iso(ts):
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert(None)
            return stamp.isoformat()

        actual_dates = [_to_naive_iso(ts) for ts in df.index]
        future_dates = pd.bdate_range(df.index[-1] + BDay(1), periods=30)
        future_dates_iso = [_to_naive_iso(ts) for ts in future_dates]

        # Pad/trim predictions to match future_dates length
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

        # Build chart JSON manually to avoid Plotly binary formats
        chart_json = {
            "data": [
                {
                    "type": "candlestick",
                    "x": actual_dates,
                    "open": df['Open'].astype(float).tolist(),
                    "high": df['High'].astype(float).tolist(),
                    "low": df['Low'].astype(float).tolist(),
                    "close": df['Close'].astype(float).tolist(),
                    "name": "OHLC",
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Actual Close (Last 6M)",
                    "x": actual_dates,
                    "y": price_series.astype(float).tolist(),
                    "line": {"color": "royalblue", "width": 2},
                },
            ] + [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"{model_name} Prediction",
                    "x": future_dates_iso,
                    "y": normalized_predictions[model_name],
                }
                for model_name in normalized_predictions
            ],
            "layout": {
                "title": f"{symbol} Stock Price Prediction (Last 6 Months)",
                "xaxis": {"title": "Date", "type": "date"},
                "yaxis": {"title": "Price"},
                "shapes": [
                    {
                        "type": "line",
                        "xref": "x",
                        "yref": "paper",
                        "x0": future_dates_iso[0] if future_dates_iso else None,
                        "x1": future_dates_iso[0] if future_dates_iso else None,
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": "gray", "dash": "dash", "width": 2},
                    }
                ] if future_dates_iso else [],
            },
        }

        response = {
            'chart': chart_json,
            'predictions': normalized_predictions,
            'current_price': float(price_series.iloc[-1]),
            'news': news,
            'actual_dates': actual_dates,
            'actual_close': np.asarray(price_series).reshape(-1).tolist(),
            'future_dates': future_dates_iso,
        }

        # Update caches
        PREDICTION_CACHE[symbol] = {
            'timestamp': now,
            'version': CACHE_VERSION,
            **response
        }
        _save_disk_cache(symbol, response)

        return response, 200

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return {'error': str(ve)}, 400
    except Exception as e:
        msg = str(e)
        logger.error(f"Unexpected error: {e}")
        # If rate limited and we have a cached response (even if stale), return it
        if ('Too Many Requests' in msg) or ('429' in msg) or ('rate limit' in msg.lower()):
            RATE_LIMIT_COOLDOWN[symbol] = time.time() + 60  # 60 seconds
            cached = PREDICTION_CACHE.get(symbol)
            if cached:
                logger.warning(f"Returning stale cached response for {symbol} due to rate limiting")
                resp = {k: v for k, v in cached.items() if k != 'timestamp'}
                resp['stale'] = True
                return resp, 200
            disk_cached = _load_disk_cache(symbol)
            if disk_cached:
                logger.warning(f"Returning stale disk-cached response for {symbol} due to rate limiting")
                disk_cached['stale'] = True
                return disk_cached, 200
            return {'error': 'Rate limited by data provider. Please try again later.'}, 429
        return {'error': 'An unexpected error occurred'}, 500

@app.route('/')
def home():
    # Render page without preloading a symbol list; user will type a symbol
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    resp, status = run_prediction(request.form.get('symbol'))
    return jsonify(resp), status


def _build_excel(symbol: str, payload: dict) -> BytesIO:
    """Create an Excel file in-memory from the prediction payload."""
    buffer = BytesIO()
    # Future predictions flattened per model/date
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
