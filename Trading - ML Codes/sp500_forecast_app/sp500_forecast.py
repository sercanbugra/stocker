import warnings
warnings.filterwarnings("ignore")

import sys
import math
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Optional


def _cache_dir() -> Path:
    d = Path("sp500_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path_for(ticker: str) -> Path:
    return _cache_dir() / f"{ticker}.csv"


def _load_cache(ticker: str, max_age_days: int = 1) -> Optional[pd.Series]:
    p = _cache_path_for(ticker)
    if not p.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        s = pd.read_csv(p, parse_dates=[0], index_col=0).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        if age_days <= max_age_days and not s.empty:
            return s
        return s if not s.empty else None
    except Exception:
        return None


def _save_cache(ticker: str, series: pd.Series) -> None:
    try:
        series.to_csv(_cache_path_for(ticker))
    except Exception:
        pass


def download_prices(ticker: str, years: int = 5, max_retries: int = 6, use_cache: bool = True, cache_max_age_days: int = 1) -> pd.Series:
    """Yahoo Finance'tan fiyat indir (rate limit'e dayanıklı).

    Boş dönerse veya hata olursa üssel geri çekilme ile tekrar dener.
    """
    # Önce önbellekten dene
    if use_cache:
        cached = _load_cache(ticker, max_age_days=cache_max_age_days)
        if cached is not None and (datetime.now() - datetime.fromtimestamp(_cache_path_for(ticker).stat().st_mtime)).days <= cache_max_age_days:
            return cached

    wait = 2.0
    last_exc = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                s = df["Close"].dropna()
                s = s[~s.index.duplicated(keep="last")]
                if use_cache:
                    _save_cache(ticker, s)
                return s
        except Exception as e:
            last_exc = e

        # Boş döndüyse veya hata aldıysak bekleyip tekrar dene
        if attempt < max_retries - 1:
            time.sleep(wait)
            wait = min(wait * 2.0, 60.0)

    # İndirme başarısız olduysa ve önbellek varsa onu kullan
    if use_cache:
        cached = _load_cache(ticker, max_age_days=10_000)  # son çare: her yaştaki cache
        if cached is not None:
            print("Uyarı: Ağ hatası/rate limit nedeniyle önbellekteki veriler kullanıldı (güncel olmayabilir).")
            return cached

    msg = (
        "Veri indirilemedi. Olası nedenler: hatalı sembol veya Yahoo Finance oran sınırı (rate limit). "
        "Bir süre sonra yeniden deneyin ya da farklı bir sembol deneyin."
    )
    if last_exc is not None:
        msg += f" Hata: {last_exc}"
    raise ValueError(msg)


def validate_symbol_format(ticker: str) -> bool:
    # Harf, rakam, nokta ve tireye izin ver (yfinance sınıf hisselerinde '-')
    return bool(re.fullmatch(r"[A-Za-z0-9\.\-]{1,10}", ticker))


def warn_if_not_in_local_sp500(ticker: str) -> None:
    # Eğer yerel S&P listesi varsa ve sembol listede değilse kullanıcıyı uyar
    # Beklenen dosya: sp500_cache/sp500_symbols.csv (kolon: Symbol)
    p = _cache_dir() / "sp500_symbols.csv"
    if not p.exists():
        return
    try:
        df = pd.read_csv(p)
        if "Symbol" in df.columns:
            syms = set(df["Symbol"].astype(str).str.upper())
            if ticker.upper() not in syms:
                print("Uyarı: Sembol, yerel S&P 500 listesindeki semboller arasında görünmüyor.")
    except Exception:
        pass


def add_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    # Takvim tabanlı zaman özellikleri
    dow = index.weekday  # 0=Mon..4=Fri
    doy = index.dayofyear
    # Sinüs/kosinüs dönüşümleri
    dow_sin = np.sin(2 * np.pi * dow / 5.0)
    dow_cos = np.cos(2 * np.pi * dow / 5.0)
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame(
        {
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
        },
        index=index,
    )


def make_supervised_features(y: pd.Series) -> pd.DataFrame:
    # Sızıntıyı önlemek için tüm rolling/lag özelliklerini hedefe göre 1 gün geriden hesaplarız.
    df = pd.DataFrame({"y": y})
    df["lag_1"] = y.shift(1)
    df["lag_2"] = y.shift(2)
    df["lag_5"] = y.shift(5)
    df["lag_10"] = y.shift(10)
    df["lag_21"] = y.shift(21)
    df["lag_63"] = y.shift(63)

    df["roll_mean_5"] = y.shift(1).rolling(5).mean()
    df["roll_mean_21"] = y.shift(1).rolling(21).mean()
    df["roll_std_21"] = y.shift(1).rolling(21).std()
    df["roll_min_21"] = y.shift(1).rolling(21).min()
    df["roll_max_21"] = y.shift(1).rolling(21).max()

    df["ret_1"] = y.pct_change(1).shift(1)
    df["ret_5"] = y.pct_change(5).shift(1)

    # EMA ve MACD benzeri özellikler (gecikmeli hesap)
    ema12 = y.shift(1).ewm(span=12, adjust=False).mean()
    ema26 = y.shift(1).ewm(span=26, adjust=False).mean()
    df["ema_12"] = ema12
    df["ema_26"] = ema26
    df["macd"] = ema12 - ema26
    # Z-skoru (21 gün)
    df["zscore_21"] = (y.shift(1) - df["roll_mean_21"]) / (df["roll_std_21"] + 1e-8)

    time_feats = add_time_features(y.index)
    df = df.join(time_feats)

    df = df.dropna()
    return df


def build_next_feature_row(history: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    # history: mevcut son değer dahil (t zamanı)
    values = {}

    def get_lag(k):
        if len(history) >= k:
            return history.iloc[-k]
        return history.iloc[-1]

    values["lag_1"] = get_lag(1)
    values["lag_2"] = get_lag(2)
    values["lag_5"] = get_lag(5)
    values["lag_10"] = get_lag(10)
    values["lag_21"] = get_lag(21)
    values["lag_63"] = get_lag(63)

    last5 = history.iloc[-5:] if len(history) >= 5 else history
    last21 = history.iloc[-21:] if len(history) >= 21 else history

    values["roll_mean_5"] = last5.mean()
    values["roll_mean_21"] = last21.mean()
    values["roll_std_21"] = last21.std(ddof=0) if len(last21) > 1 else 0.0
    values["roll_min_21"] = last21.min()
    values["roll_max_21"] = last21.max()

    # get last pct changes
    if len(history) >= 2:
        values["ret_1"] = history.iloc[-1] / history.iloc[-2] - 1.0
    else:
        values["ret_1"] = 0.0

    if len(history) >= 6:
        values["ret_5"] = history.iloc[-1] / history.iloc[-6] - 1.0
    else:
        values["ret_5"] = 0.0

    # EMA'lar (son örnek üzerine hesaplanmış)
    ema12 = history.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = history.ewm(span=26, adjust=False).mean().iloc[-1]
    values["ema_12"] = float(ema12)
    values["ema_26"] = float(ema26)
    values["macd"] = float(ema12 - ema26)
    # Z-skoru
    if values["roll_std_21"] > 0:
        values["zscore_21"] = (history.iloc[-1] - values["roll_mean_21"]) / values["roll_std_21"]
    else:
        values["zscore_21"] = 0.0

    # time features for next_date
    dow = next_date.weekday()
    doy = next_date.timetuple().tm_yday
    values["dow_sin"] = math.sin(2 * math.pi * dow / 5.0)
    values["dow_cos"] = math.cos(2 * math.pi * dow / 5.0)
    values["doy_sin"] = math.sin(2 * math.pi * doy / 365.25)
    values["doy_cos"] = math.cos(2 * math.pi * doy / 365.25)

    # Tek satırlık güvenli DataFrame oluşturma (skaler sözlük -> kayıt)
    return pd.DataFrame([values], index=[next_date])


def iterative_forecast(estimator, y_train: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    # Eğitim için gözetimli özellik seti
    train_df = make_supervised_features(y_train)
    feature_cols = [c for c in train_df.columns if c != "y"]
    X_train = train_df[feature_cols].values
    y_train_arr = train_df["y"].values

    estimator.fit(X_train, y_train_arr)

    history = y_train.copy()
    preds = []
    for dt in future_index:
        x_next = build_next_feature_row(history, dt)
        y_hat = float(estimator.predict(x_next.values)[0])
        preds.append(y_hat)
        # Kendini-beslemeli: tahmini geçmişe ekle
        history.loc[dt] = y_hat

    return pd.Series(preds, index=future_index)


def arima_forecast(y_fit: pd.Series, steps_index: pd.DatetimeIndex) -> pd.Series:
    # Basit ama güçlü bir SARIMAX yapılandırması
    # order=(5,1,0), seasonal haftalık 5 iş günü
    y_fit = y_fit.sort_index()
    # Düzenli iş günü frekansına oturt ve boşlukları ileri doldur
    y_fit = y_fit.asfreq('B').ffill()
    # Küçük bir grid ile AIC minimizasyonu
    best_aic = np.inf
    best_cfg = None
    best_res = None
    seasonal_period = 5
    orders = [(p, d, q) for p in range(0, 4) for d in (0, 1) for q in range(0, 4)]
    seasonal_orders = [(P, D, Q, seasonal_period) for P in (0, 1) for D in (0, 1) for Q in (0, 1)]
    for order in orders:
        for sorder in seasonal_orders:
            try:
                model = SARIMAX(y_fit, order=order, seasonal_order=sorder, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                aic = getattr(res, 'aic', np.inf)
                if aic < best_aic:
                    best_aic = aic
                    best_cfg = (order, sorder)
                    best_res = res
            except Exception:
                continue
    if best_res is None:
        # Yedek: sabit konfigürasyon
        model = SARIMAX(y_fit, order=(5, 1, 0), seasonal_order=(1, 0, 1, seasonal_period), enforce_stationarity=False, enforce_invertibility=False)
        best_res = model.fit(disp=False)
    fc = best_res.get_forecast(steps=len(steps_index))
    yhat = np.asarray(fc.predicted_mean)
    return pd.Series(yhat, index=steps_index)


def holt_winters_forecast(y_fit: pd.Series, steps_index: pd.DatetimeIndex) -> pd.Series:
    # Küçük bir konfigürasyon taraması (add/mul, damped True/False)
    y_fit = y_fit.sort_index().asfreq('B').ffill()
    seasonal_periods = 5
    configs = [
        ("add", True), ("add", False), ("mul", True), ("mul", False)
    ]
    best_sse = np.inf
    best_res = None
    for seasonal, damped in configs:
        try:
            model = ExponentialSmoothing(
                y_fit,
                trend="add",
                damped_trend=damped,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            res = model.fit(optimized=True, use_brute=True)
            sse = np.sum((res.fittedvalues - y_fit) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_res = res
        except Exception:
            continue
    if best_res is None:
        # Yedek
        model = ExponentialSmoothing(
            y_fit,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        best_res = model.fit(optimized=True, use_brute=True)
    yhat = pd.Series(best_res.forecast(len(steps_index)).values, index=steps_index)
    return yhat


def tune_estimator(model_name: str, X: np.ndarray, y: np.ndarray):
    tss = TimeSeriesSplit(n_splits=3)
    if model_name == "RandomForest":
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = {
            "n_estimators": [400, 800],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        }
        search = RandomizedSearchCV(base, grid, n_iter=12, cv=tss, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42)
    elif model_name == "GradientBoosting":
        base = HistGradientBoostingRegressor(random_state=42)
        grid = {
            "learning_rate": [0.03, 0.06, 0.1],
            "max_depth": [None, 6, 12],
            "max_leaf_nodes": [31, 63, 127],
            "l2_regularization": [0.0, 0.01, 0.1],
        }
        search = RandomizedSearchCV(base, grid, n_iter=12, cv=tss, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42)
    elif model_name == "SVR":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf")),
        ])
        grid = {
            "svr__C": [1.0, 5.0, 10.0, 20.0],
            "svr__epsilon": [0.005, 0.01, 0.02],
            "svr__gamma": ["scale", 0.05, 0.1],
        }
        search = GridSearchCV(base, grid, cv=tss, scoring="neg_mean_absolute_error", n_jobs=-1)
    else:
        raise ValueError("Bilinmeyen model adı")

    search.fit(X, y)
    return search.best_estimator_


def iterative_forecast_tuned(model_name: str, y_train: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    train_df = make_supervised_features(y_train)
    feature_cols = [c for c in train_df.columns if c != "y"]
    X_train = train_df[feature_cols].values
    y_train_arr = train_df["y"].values

    estimator = tune_estimator(model_name, X_train, y_train_arr)

    history = y_train.copy()
    preds = []
    for dt in future_index:
        x_next = build_next_feature_row(history, dt)
        # Reindex to feature order to avoid column misalignment
        x_next = x_next.reindex(columns=feature_cols, fill_value=0.0)
        y_hat = float(estimator.predict(x_next.values)[0])
        preds.append(y_hat)
        history.loc[dt] = y_hat
    return pd.Series(preds, index=future_index)


def plot_results(ticker: str,
                 actual_past: pd.Series,
                 preds_backtest: dict,
                 preds_future: dict,
                 future_index: pd.DatetimeIndex,
                 show: bool = True):
    fig, ax = plt.subplots(figsize=(13, 6))
    # Gerçek son 6 ay
    ax.plot(actual_past.index, actual_past.values, color="black", linewidth=2.0, label="Gerçek (Son 1 Yıl)")

    colors = {
        "ARIMA": "#1f77b4",
        "Holt-Winters": "#ff7f0e",
        "RandomForest": "#2ca02c",
        "GradientBoosting": "#d62728",
        "SVR": "#9467bd",
    }

    # Geriye dönük tahminleri kesikli çizgi ile
    for name, series in preds_backtest.items():
        ax.plot(series.index, series.values, linestyle="--", color=colors.get(name, None), alpha=0.9, label=f"{name} (Backtest)")

    # İleri tahminleri düz çizgi ile
    for name, series in preds_future.items():
        ax.plot(series.index, series.values, linestyle="-", color=colors.get(name, None), alpha=0.9, label=f"{name} (İleri 6 Ay)")

    # Ayırıcı çizgi (bugün)
    cutoff = actual_past.index[-1]
    ax.axvline(cutoff, color="gray", linestyle=":", alpha=0.7)
    ax.text(cutoff, ax.get_ylim()[1], " Bugün", va="top", ha="left", color="gray")

    ax.set_title(f"{ticker} | Son 1 Yıl Gerçek ve 5 Model ile 1 Ay Tahmin")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat (Kapanış)")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def main_cli():
    try:
        ticker = input("S&P 500 hissesi/sembolü girin (örn: AAPL, MSFT, NVDA, SPY): ").strip().upper()
        if not ticker:
            print("Geçerli bir sembol giriniz.")
            sys.exit(1)
        if not validate_symbol_format(ticker):
            print("Geçersiz sembol biçimi. Sadece harf, rakam, '.' ve '-' kullanılabilir.")
            sys.exit(1)

        warn_if_not_in_local_sp500(ticker)

        backtest_horizon = 252  # ~1 yıl iş günü
        forecast_horizon = 21   # ~1 ay iş günü

        prices = download_prices(ticker, years=5, use_cache=True, cache_max_age_days=1)
        if len(prices) < backtest_horizon + 10:
            raise ValueError("Yeterli veri yok.")

        last_idx = prices.index[-backtest_horizon:]
        actual_last = prices[-backtest_horizon:]
        train_bt = prices.iloc[:-backtest_horizon]

        last_date = prices.index[-1]
        future_index = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)

        arima_bt = arima_forecast(train_bt, last_idx)
        arima_fut = arima_forecast(prices, future_index)

        hw_bt = holt_winters_forecast(train_bt, last_idx)
        hw_fut = holt_winters_forecast(prices, future_index)

        rf_bt = iterative_forecast_tuned("RandomForest", train_bt, last_idx)
        rf_fut = iterative_forecast_tuned("RandomForest", prices, future_index)

        gbr_bt = iterative_forecast_tuned("GradientBoosting", train_bt, last_idx)
        gbr_fut = iterative_forecast_tuned("GradientBoosting", prices, future_index)

        svr_bt = iterative_forecast_tuned("SVR", train_bt, last_idx)
        svr_fut = iterative_forecast_tuned("SVR", prices, future_index)

        preds_last = {
            "ARIMA": arima_bt,
            "Holt-Winters": hw_bt,
            "RandomForest": rf_bt,
            "GradientBoosting": gbr_bt,
            "SVR": svr_bt,
        }
        preds_next = {
            "ARIMA": arima_fut,
            "Holt-Winters": hw_fut,
            "RandomForest": rf_fut,
            "GradientBoosting": gbr_fut,
            "SVR": svr_fut,
        }

        plot_results(ticker, actual_last, preds_last, preds_next, future_index, show=True)

    except Exception as e:
        print(f"Hata: {e}")
        sys.exit(1)


def main():
    try:
        ticker = input("S&P 500 hissesi/sembolü girin (örn: AAPL, MSFT, NVDA, SPY): ").strip().upper()
        if not ticker:
            print("Geçerli bir sembol giriniz.")
            sys.exit(1)
        if not validate_symbol_format(ticker):
            print("Geçersiz sembol biçimi. Sadece harf, rakam, '.' ve '-' kullanılabilir.")
            sys.exit(1)

        # Yerel S&P listesi varsa uyarı ver (bloklamaz)
        warn_if_not_in_local_sp500(ticker)

        horizon = 126  # ~6 ay işlem günü
        prices = download_prices(ticker, years=5, use_cache=True, cache_max_age_days=1)
        if len(prices) < horizon * 2 + 100:
            print("Uyarı: Veri kısa, modeller sınırlı performans gösterebilir.")

        # Son 6 ay aralığı
        if len(prices) < horizon + 10:
            raise ValueError("Yeterli veri yok.")

        last6m_idx = prices.index[-horizon:]
        actual_last6m = prices[-horizon:]

        # Backtest eğitimi için kesim: son 6 aydan hemen önce
        train_bt = prices.iloc[: -horizon]

        # Gelecek 6 ay iş günleri
        last_date = prices.index[-1]
        future_index = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

        # 1) ARIMA/SARIMAX (otomatik konfigürasyon seçimi)
        arima_bt = arima_forecast(train_bt, last6m_idx)
        arima_fut = arima_forecast(prices, future_index)

        # 2) Holt-Winters (küçük grid seçimi)
        hw_bt = holt_winters_forecast(train_bt, last6m_idx)
        hw_fut = holt_winters_forecast(prices, future_index)

        # ML modelleri (tuning + iteratif tahmin)
        rf_bt = iterative_forecast_tuned("RandomForest", train_bt, last6m_idx)
        rf_fut = iterative_forecast_tuned("RandomForest", prices, future_index)

        gbr_bt = iterative_forecast_tuned("GradientBoosting", train_bt, last6m_idx)
        gbr_fut = iterative_forecast_tuned("GradientBoosting", prices, future_index)

        svr_bt = iterative_forecast_tuned("SVR", train_bt, last6m_idx)
        svr_fut = iterative_forecast_tuned("SVR", prices, future_index)

        preds_last6m = {
            "ARIMA": arima_bt,
            "Holt-Winters": hw_bt,
            "RandomForest": rf_bt,
            "GradientBoosting": gbr_bt,
            "SVR": svr_bt,
        }
        preds_next6m = {
            "ARIMA": arima_fut,
            "Holt-Winters": hw_fut,
            "RandomForest": rf_fut,
            "GradientBoosting": gbr_fut,
            "SVR": svr_fut,
        }

        plot_results(ticker, actual_last, preds_last, preds_next, future_index, show=True)

    except Exception as e:
        print(f"Hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
