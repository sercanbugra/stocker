import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
from flask import Flask, render_template, request

app = Flask(__name__)

# Fetch all NASDAQ and NYSE tickers
all_tickers = yf.Ticker("SPY").history(period="1d").index.tolist()

def fetch_stock_data(ticker, start_date):
    """Fetch stock data with fallback handling."""
    try:
        df = yf.download(ticker, start=start_date, end=datetime.now())
        return df
    except Exception as e:
        print("Error fetching data:", e)
        return None

def prepare_data(df):
    """Prepares training and testing datasets."""
    data = df[['Close']].copy()
    dataset = data.values
    train_size = int(len(dataset) * 0.9)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    return data, dataset, train_size, scaler, scaled_data

def create_sequences(data, window_size):
    """Creates sequences for LSTM input."""
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    selected_stock = None
    if request.method == 'POST':
        selected_stock = request.form['stock']
        df = fetch_stock_data(selected_stock, start_date='2023-02-01')
        if df is not None and not df.empty:
            data, dataset, train_size, scaler, scaled_data = prepare_data(df)
            
            # Training data preparation
            window_size = 160
            train_data = scaled_data[:train_size]
            x_train, y_train = create_sequences(train_data, window_size)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            # Define LSTM model
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                Dropout(0.3),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                Dense(25),
                Dense(1)
            ])
            
            # Compile & train model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)
            
            # Predict next 10 days
            last_days = scaled_data[-window_size:]
            future_predictions = []
            for i in range(10):
                X_test = np.array([last_days])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                pred_price = model.predict(X_test)
                pred_price_unscaled = scaler.inverse_transform(pred_price)
                future_predictions.append(pred_price_unscaled[0, 0])
                last_days = np.append(last_days, pred_price, axis=0)[-window_size:]
            
            predictions = future_predictions
    
    return render_template('index.html', stocks=all_tickers, selected_stock=selected_stock, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
