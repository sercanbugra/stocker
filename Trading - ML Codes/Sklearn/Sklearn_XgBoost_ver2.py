import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Kullanıcıdan giriş al
symbol = input("Tahmin yapmak istediğiniz hisse senedi sembolünü girin (örn: AAPL, TSLA, MSFT): ")
past_days = int(input("Kaç günlük geçmiş veriyi kullanarak tahmin yapmak istiyorsunuz? (örn: 180, 365): "))

# Veri çekme
end_date = datetime.today()
start_date = end_date - timedelta(days=past_days)
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    raise ValueError("Hisse senedi verisi çekilemedi. Lütfen sembolü ve internet bağlantınızı kontrol edin.")

# Prophet için veri hazırlama
df = df.reset_index()
df = df[['Date', 'Close']]
df.columns = ['ds', 'y']  # Prophet için kolon isimleri

# Model oluşturma ve eğitme
model = Prophet()
model.fit(df)

# Gelecek 30 gün için tahmin yapma
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))
model.plot(forecast)
plt.title(f'{symbol} Hisse Senedi Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Kapanış Fiyatı (USD)')
plt.grid()
plt.show()

# Excel'e kaydetme
excel_filename = f'{symbol}_stock_predictions.xlsx'
with pd.ExcelWriter(excel_filename) as writer:
    df.to_excel(writer, sheet_name='Geçmiş Veriler', index=False)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(writer, sheet_name='Tahminler', index=False)

print(f"Tahminler başarıyla '{excel_filename}' dosyasına kaydedildi!")
