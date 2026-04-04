import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

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

# Fetch stock data
ticker = 'SNPS'
df = fetch_stock_data(ticker, start_date='2024-01-01')
if df is None or df.empty:
    raise ValueError("No data fetched. Check stock ticker or network connection.")

# Prepare data
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
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

# Testing data preparation
test_data = scaled_data[train_size - window_size:]
x_test, y_test = create_sequences(test_data, window_size)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict stock prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')

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

# Format future predictions
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 11)]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions}).set_index('Date')

# Save predictions to Excel
try:
    with pd.ExcelWriter('stock_predictions_optimized.xlsx') as writer:
        data[:train_size].to_excel(writer, sheet_name='Train')
        valid = data[train_size:].copy()
        valid['Predictions'] = predictions
        valid.to_excel(writer, sheet_name='Validation')
        future_df.to_excel(writer, sheet_name='Future')
except Exception as e:
    print("Error saving Excel file:", e)

# Plot results
plt.figure(figsize=(16, 6))
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price (USD)', fontsize=14)
plt.plot(data[:train_size]['Close'], label='Train')
plt.plot(valid[['Close']], label='Validation')
plt.plot(valid[['Predictions']], label='Predictions')
plt.plot(future_df['Predicted Close'], marker='o', linestyle='dashed', color='red', label='Future')
plt.legend()
plt.show()
