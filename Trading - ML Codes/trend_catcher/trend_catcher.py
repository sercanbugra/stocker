import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.stats import linregress
import time 
import requests # Yeni eklenen
from bs4 import BeautifulSoup # Yeni eklenen

## ----------------------------------------------------
## 1. AYARLAR VE PARAMETRELER
## ----------------------------------------------------

# Veri periyodu: Son 6 ay (günlük)
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=6 * 30) 

# Trend Filtresi
MIN_R_SQUARED = 0.60  
MIN_SLOPE = 0.001     

# Tahmin Ayarları
K_SIGMA = 1.5         
FUTURE_DAYS = 30      

# Veri çekme gecikmesi (Sunucu yavaşlatması için)
DELAY_SECONDS = 0.2 

## ----------------------------------------------------
## 2. NASDAQ SEMBOL LİSTESİNİ WEB'DEN ÇEKME
## ----------------------------------------------------

def get_all_nasdaq_tickers_from_web():
    """
    Bir web sitesinden tüm NASDAQ sembollerini çekmeye çalışır.
    """
    # Nasdaq sembol listesinin bulunduğu bir sayfa (Örnek Kaynak - değişebilir!)
    URL = "https://www.nasdaq.com/market-activity/stocks/screener?exchange=NASDAQ&render=download"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print("## 🌐 Tüm NASDAQ Sembolleri Web'den Çekiliyor...")
    
    try:
        # Bu kaynak genellikle bir CSV dosyası indirme linki olduğu için,
        # Pandas ile doğrudan okumayı deneyebiliriz, bu web kazımadan daha güvenilirdir:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            # Yanıt metnini bir pandas DataFrame'e dönüştür
            from io import StringIO
            data = StringIO(response.text)
            df = pd.read_csv(data)
            
            # Sembol sütununu bul ve listeye dönüştür
            # Sütun adı sitedeki güncellemelerle değişebilir (örneğin 'Symbol' veya 'Ticker')
            if 'Symbol' in df.columns:
                 tickers = df['Symbol'].tolist()
            elif 'Symbol ' in df.columns: # Bazen boşluk olabilir
                 tickers = df['Symbol '].tolist()
            else:
                 print("  ❌ Sembol sütunu bulunamadı. Yapı değişmiş olabilir.")
                 return []
                 
            # Boş veya hatalı sembolleri temizle
            tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and len(t.strip()) > 1]
            print(f"  ✅ {len(tickers)} adet sembol Web'den yüklendi.")
            return tickers
        else:
            print(f"  ❌ Web isteği başarısız oldu. Durum Kodu: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"  ❌ Sembol listesi çekilemedi: {e}")
        return []

TICKERS = get_all_nasdaq_tickers_from_web()

if not TICKERS:
    # Eğer liste boşsa, programı sonlandır.
    print("\nProgram sonlandırılıyor. Sembol listesi yüklenemedi.")
    exit()

## ----------------------------------------------------
## 3. VERİ ÇEKME VE ANALİZİ (TÜM LİSTE İÇİN TEK TEK İŞLEME)
## ----------------------------------------------------

print("\n## 🚀 Tüm Hisseler Tek Tek Çekiliyor ve Analiz Ediliyor...")
prediction_table = []
successful_analyses = 0

for i, ticker in enumerate(TICKERS):
    
    time.sleep(DELAY_SECONDS) # Gecikme
    print(f"  [#{i+1}/{len(TICKERS)}] {ticker} işleniyor...", end='\r')
    
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, show_errors=False)
        current_data = data['Adj Close'].dropna()
        
        if len(current_data) < 30: 
            continue
            
        x = np.arange(len(current_data))
        y = current_data.values
        
        # Lineer Regresyon
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2
        
        # Filtreleme
        if slope > MIN_SLOPE and r_squared >= MIN_R_SQUARED:
            
            trend_price = slope * x + intercept
            deviations = y - trend_price
            std_dev = np.std(deviations) 
            
            last_day_index = len(x) - 1
            last_price = y[-1]
            last_trend_price = trend_price[last_day_index]
            
            # Tahmin
            today_min_target = last_trend_price - K_SIGMA * std_dev
            future_day_index = last_day_index + FUTURE_DAYS
            future_trend_price = slope * future_day_index + intercept
            future_max_target = future_trend_price + K_SIGMA * std_dev 
            
            prediction_table.append({
                'Hisse': ticker,
                'Son Kapanış ($)': f"{last_price:.2f}",
                'Min Alım (Destek $)': f"{today_min_target:.2f}",
                f'Satış Hedefi (+{FUTURE_DAYS} Gün $)': f"{future_max_target:.2f}",
                'R-Kare Uyumu': f"{r_squared:.4f}",
                'Eğim': f"{slope:.4f}"
            })
            successful_analyses += 1
            print(f"  [#{i+1}/{len(TICKERS)}] ✅ {ticker}: Analiz Başarılı (Eğim: {slope:.4f}, R-Kare: {r_squared:.4f})")
            
        else:
            pass # Trende uymayanları sessizce geç
            
    except Exception as e:
        pass # Hata veren hisseleri atla


## ----------------------------------------------------
## 4. SONUÇLARIN GÖSTERİLMESİ
## ----------------------------------------------------

print("\n" + "="*90)
print(f"## 📊 FİNAL ANALİZ RAPORU ({successful_analyses} Hissede Lineer Artış Trendi Tespit Edildi)")
print("="*90)

if prediction_table:
    results_df = pd.DataFrame(prediction_table)
    results_df = results_df.sort_values(by='R-Kare Uyumu', ascending=False)
    
    print(results_df.to_markdown(index=False))
    print("\n*Min Alım (Destek): Trend çizgisine göre K_SIGMA Standart Sapma altındaki değer.")
    print(f"*Satış Hedefi (+{FUTURE_DAYS} Gün): {FUTURE_DAYS} gün sonraki trendin K_SIGMA Standart Sapma üzerindeki değeri.")
    
else:
    print("\n⚠️ Belirtilen kriterlere (Eğim > 0, R-Kare > 0.6) uyan lineer artış trendine sahip hisse senedi bulunamadı.")