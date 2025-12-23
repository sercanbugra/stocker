import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Flask imports
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel('INFO')

app = Flask(__name__)

def fetch_sp500_stocks():
    """Fetch S&P 500 stocks with robust error handling"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        stocks = df['Symbol'].tolist()[:500]
        logger.info(f"Successfully fetched {len(stocks)} stocks")
        return stocks
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 
            'NVDA', 'JPM', 'V', 'JNJ', 'WMT', 'MA', 'UNH', 'DIS', 'BAC'
        ]

def validate_stock_data(df):
    """Validate and preprocess stock data for the last year"""
    if df is None or df.empty:
        raise ValueError("No stock data available")
    
    # Ensure data is for the last year
    df = df.last('365D')
    
    if len(df) < 250:  # Minimum trading days in a year
        raise ValueError(f"Insufficient data points. Required: 250, Available: {len(df)}")
    
    # Remove rows with zero or negative prices
    df = df[df['Close'] > 0]
    
    # Fill any remaining NaN values
    columns_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in columns_to_fill:
        df[col].fillna(method='ffill', inplace=True)
    
    return df

def prepare_time_series_data(data, time_steps=30):
    """Prepare data for time series models using last year's data"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps])
    
    return np.array(X), np.array(y), scaler

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
    X, y, scaler = prepare_time_series_data(data, time_steps)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    predictions = {}
    
    # LSTM Model
    lstm_model = create_lstm_model((time_steps, 1))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    lstm_pred = lstm_model.predict(X_test[-30:].reshape(30, time_steps, 1))
    predictions['LSTM'] = scaler.inverse_transform(lstm_pred)
    
    # XGBoost Model
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    xgb_pred = xgb_model.predict(X_test[-30:].reshape(30, -1))
    predictions['XGBoost'] = scaler.inverse_transform(xgb_pred.reshape(-1, 1))
    
    # Prophet Model
    try:
        df_prophet = pd.DataFrame({
            'ds': dates[-len(data):],
            'y': data
        })
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(df_prophet)
        future = prophet_model.make_future_dataframe(periods=30)
        prophet_forecast = prophet_model.predict(future)
        predictions['Prophet'] = prophet_forecast['yhat'][-30:].values.reshape(-1, 1)
    except Exception as e:
        logger.error(f"Prophet model error: {e}")
        predictions['Prophet'] = predictions['LSTM']
    
    # ARIMA Model
    try:
        arima_model = ARIMA(data, order=(5,1,2))
        arima_results = arima_model.fit()
        arima_pred = arima_results.forecast(steps=30)
        predictions['ARIMA'] = arima_pred.reshape(-1, 1)
    except Exception as e:
        logger.error(f"ARIMA model error: {e}")
        predictions['ARIMA'] = predictions['XGBoost']
    
    return predictions

@app.route('/')
def home():
    stocks = fetch_sp500_stocks()
    return render_template('index.html', stocks=stocks)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.form['symbol']
        
        # Fetch and validate stock data for the last year
        stock = yf.Ticker(symbol)
        df = stock.history(period='1y')
        df = validate_stock_data(df)
        
        # Predict using multiple models
        predictions = train_prediction_models(
            df['Close'].values, 
            df.index
        )
        
        # Create candlestick chart
        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
        
        # Add prediction lines
        prediction_traces = [
            go.Scatter(
                x=pd.date_range(start=df.index[-1], periods=30),
                y=pred.flatten(),
                mode='lines',
                name=f'{model_name} Prediction'
            )
            for model_name, pred in predictions.items()
        ]
        
        # Combine traces
        fig = go.Figure(data=[candlestick] + prediction_traces)
        fig.update_layout(
            title=f'{symbol} Stock Price Prediction (Last 1 Year)', 
            xaxis_title='Date', 
            yaxis_title='Price'
        )
        
        # Fetch news
        news = stock.news[:5] if stock.news else []
        
        return jsonify({
            'chart': fig.to_json(),
            'predictions': {k: v.tolist() for k, v in predictions.items()},
            'current_price': df['Close'][-1],
            'news': news
        })
    
    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)