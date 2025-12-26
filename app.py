import os
import logging
import json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from plotly.utils import PlotlyJSONEncoder
from io import BytesIO

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

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
    return render_template('index.html', stocks=stocks)

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
        df = stock.history(period='1y')
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

        # Last 3 months chart
        df_3m = df.last('90D') if hasattr(df, "last") else df.loc[df.index >= (df.index.max() - pd.Timedelta(days=90))]
        last3_dates = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in df_3m.index]
        fig_3m = go.Figure(data=[
            go.Candlestick(
                x=last3_dates,
                open=df_3m['Open'].astype(float).tolist(),
                high=df_3m['High'].astype(float).tolist(),
                low=df_3m['Low'].astype(float).tolist(),
                close=df_3m['Close'].astype(float).tolist(),
                name="OHLC (3M)"
            )
        ])
        fig_3m.update_layout(
            title=f'{symbol} Stock Price (Last 3 Months)',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        chart_3m_json = json.loads(json.dumps(fig_3m, cls=PlotlyJSONEncoder))

        # 3M predictions: train on 3M closes, forecast 5 business days
        preds_3m = {}
        try:
            preds_3m = train_prediction_models(
                df_3m['Close'].values,
                forecast_horizon=5,
                time_steps=20
            )
            for k, v in list(preds_3m.items()):
                arr = np.asarray(v).reshape(-1)
                preds_3m[k] = arr[:5].tolist()
        except Exception as e:
            logger.error(f"3M prediction error for {symbol}: {e}")
        future_dates_3m = pd.date_range(start=df_3m.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
        future_dates_3m_iso = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in future_dates_3m]

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
            'chart_3m': chart_3m_json,
            'predictions': normalized_predictions,
            'predictions_3m': preds_3m,
            'current_price': float(df['Close'].iloc[-1]),
            'news': news,
            'company_info': company_info,
            'actual_dates': actual_dates,
            'actual_close': df['Close'].astype(float).tolist(),
            'future_dates': future_dates_iso,
            'future_dates_3m': future_dates_3m_iso
        }
        return response, 200
    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return {'error': str(ve)}, 400
    except Exception as e:
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
