# NASDAQ Trend ve Salınım Tahmini

Bu küçük Python projesi, NASDAQ hisse senetlerinin son 6 aylık (günlük) kapanış fiyatlarını indirip yükselen lineer trende uyan senetlerde basit salınım tahmini yapar. Her sembol için:

- Lineer regresyon eğimi (`slope`) ve uyum (`R^2`) hesaplanır.
- Yerel min/max noktaları tespit edilir, ortalama salınım periyodu ve genliği bulunur.
- Bir sonraki min ve max tarihlerine denk gelen fiyatlar tahmin edilir.
- Alım için önerilen fiyat, beklenen min ve max arasındaki orta nokta olarak verilir. Örnek: min/max dizisi 1,3,2,4,3,5,4 ise bir sonraki max ≈6, önerilen alım ≈5 olarak raporlanır.

## Kurulum

```
pip install -r requirements.txt
```

## Çalıştırma

Tam NASDAQ listesini indirip 6 aylık veriyle çalışmak (paket olarak):

```
python -m nasdaq_trend_forecaster.cli --limit 0 --months 6 --out tahminler.csv
```

Doğrudan script olarak çalıştırmak isterseniz (relative import hatası alırsanız):

```
python nasdaq_trend_forecaster/cli.py --symbols AAPL MSFT NVDA --out ornek.csv
```

Belirli sembollerle hızlı deneme:

```
python -m nasdaq_trend_forecaster.cli --symbols AAPL MSFT NVDA --out ornek.csv
```

### Daha fazla ihtimal için filtreleri gevşetme

- `--min-r2 0.02` gibi daha düşük R² eşiği, daha fazla (ama daha zayıf) trendi geçirir.
- `--min-slope -0.05` küçük aşağı/yatay trendleri de dahil eder.
- `--min-points 10` gibi bir değer, daha kısa geçmişle tahmin üretir.

Örnek:

```
python -m nasdaq_trend_forecaster.cli --limit 0 --months 6 --min-r2 0.02 --min-slope -0.05 --out tahminler_genis.csv
```

### Ağ kısıtlı ortamlarda

`--offline` bayrağı NASDAQ listesini indirmeye çalışmaz; önce cache varsa onu, yoksa dahili küçük bir sembol listesini (AAPL, MSFT, NVDA, AMZN, GOOGL, META) kullanır. Örnek:

```
python -m nasdaq_trend_forecaster.cli --offline --out tahminler.csv
```

Komut tamamlandığında CSV yolu ve en yüksek güvenli 10 tahmin ekrana yazılır.

## Nasıl Çalışıyor?

- **Veri**: `nasdaq_trend_forecaster/data.py` NASDAQ sembol listesini (ftp.nasdaqtrader.com) indirir ve `yfinance` ile fiyatları çeker.
- **Trend**: `analysis.fit_linear` lineer eğim/R² hesaplar, `is_uptrend` ile minimum eğim ve uyum şartı uygulanır.
- **Salınım**: Yerel min/max noktaları kısa bir pencereyle belirlenir, ortalama periyot (gün) ve genlik çıkarılır.
- **Tahmin**: `forecast.make_forecast` min/max dizilerine göre bir sonraki tarih ve fiyatı lineer olarak projekte eder; alım noktası min/max ortalaması olarak raporlanır.

## Notlar ve Sınırlamalar

- Veri indirme için internet gerekir; sınırlandırılmış ortamlarda cache dosyası kullanılır.
- Lineer trend + basit salınım modeli gerçek piyasa davranışının yalnızca kaba bir yaklaşımıdır. Daha hassas sonuçlar için ek filtreler (hacim, volatilite, makine öğrenimi modelleri vb.) eklenmelidir.
- `--limit` parametresi, çok sayıda sembolde uzun süren indirmeleri yönetmek için vardır; 0 veya negatif değer verirseniz tüm listeyi işler.
