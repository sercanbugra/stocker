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

def prepare_time_series_data(data, time_steps=30):
    """Prepare data for time series models using last year's data"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps])
    
    return np.array(X), np.array(y), scaler, scaled_data


def _forecast_tree_recursive(model, last_window_scaled, steps):
    """Recursive multi-step forecast for tree-based models using scaled window."""
    window = last_window_scaled.copy().ravel()
    preds = []
    for _ in range(steps):
        next_scaled = model.predict(window.reshape(1, -1))
        val = float(next_scaled.ravel()[0])
        val = max(0.0, min(1.0, val))
        preds.append(val)
        window = np.hstack([window[1:], [val]])
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

def train_prediction_models(data, dates):
    """Train multiple prediction models using last year's data"""
    time_steps = 30
    X, y, scaler, scaled_data = prepare_time_series_data(data, time_steps)

    flat_X = X.reshape(X.shape[0], -1)
    last_window_scaled = scaled_data[-time_steps:].copy()
    predictions = {}
    
    # XGBoost Model
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(flat_X, y)
    xgb_scaled = _forecast_tree_recursive(xgb_model, last_window_scaled, 30)
    predictions['XGBoost'] = scaler.inverse_transform(xgb_scaled)
    
    # ExtraTrees Model
    try:
        et_model = ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(flat_X, y)
        et_scaled = _forecast_tree_recursive(et_model, last_window_scaled, 30)
        predictions['ExtraTrees'] = scaler.inverse_transform(et_scaled)
    except Exception as e:
        logger.error(f"ExtraTrees model error: {e}")

    # RandomForest Model
    try:
        rf_model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(flat_X, y)
        rf_scaled = _forecast_tree_recursive(rf_model, last_window_scaled, 30)
        predictions['RandomForest'] = scaler.inverse_transform(rf_scaled)
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

        predictions = train_prediction_models(df['Close'].values, df.index)

        # Build dates
        actual_dates = [pd.Timestamp(ts).tz_localize(None).isoformat() for ts in df.index]
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
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

        # Plotly chart serialized safely
        candlestick = go.Candlestick(
            x=actual_dates,
            open=df['Open'].astype(float).tolist(),
            high=df['High'].astype(float).tolist(),
            low=df['Low'].astype(float).tolist(),
            close=df['Close'].astype(float).tolist(),
            name="OHLC"
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
            'predictions': normalized_predictions,
            'current_price': float(df['Close'].iloc[-1]),
            'news': news,
            'company_info': company_info,
            'actual_dates': actual_dates,
            'actual_close': df['Close'].astype(float).tolist(),
            'future_dates': future_dates_iso
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
