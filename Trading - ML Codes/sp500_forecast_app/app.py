import io
import base64
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from flask import Flask, request, render_template_string, Response, jsonify

import sp500_forecast as f


app = Flask(__name__)


PAGE_TMPL = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>S&P 500 Forecast</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
      .wrap { max-width: 980px; margin: auto; }
      form { margin-bottom: 16px; }
      input[type=text] { padding: 8px 10px; font-size: 16px; width: 220px; }
      button { padding: 8px 14px; font-size: 16px; cursor: pointer; }
      .msg { color: #b00; margin: 8px 0; }
      img { max-width: 100%; border: 1px solid #ddd; }
      .hint { color: #666; font-size: 14px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h2>S&P 500 Forecast (Son 1 Yıl Backtest + İleri 1 Ay)</h2>
      <form method="get">
        <label>Sembol:&nbsp;</label>
        <input name="ticker" type="text" value="{{ ticker or '' }}" placeholder="AAPL, MSFT, SPY" />
        <button type="submit">Tahmin Et</button>
      </form>
      {% if message %}
        <div class="msg">{{ message }}</div>
      {% endif %}
      {% if img_data %}
        <img src="data:image/png;base64,{{ img_data }}" alt="Forecast" />
      {% else %}
        <div class="hint">Bir sembol girin ve Tahmin Et'e tıklayın.</div>
      {% endif %}
    </div>
  </body>
  </html>
"""


def _compute_forecasts(ticker: str):
    # Tek yerde hesapla: 1 yıl backtest, 1 ay ileri
    backtest_horizon = 252
    forecast_horizon = 21
    prices = f.download_prices(ticker, years=5, use_cache=True, cache_max_age_days=1)
    if len(prices) < backtest_horizon + 10:
        raise ValueError("Yeterli veri yok.")

    last_idx = prices.index[-backtest_horizon:]
    actual_last = prices[-backtest_horizon:]
    train_bt = prices.iloc[:-backtest_horizon]
    last_date = prices.index[-1]
    future_index = f.pd.bdate_range(last_date + f.pd.Timedelta(days=1), periods=forecast_horizon)

    arima_bt = f.arima_forecast(train_bt, last_idx)
    arima_fut = f.arima_forecast(prices, future_index)

    hw_bt = f.holt_winters_forecast(train_bt, last_idx)
    hw_fut = f.holt_winters_forecast(prices, future_index)

    rf_bt = f.iterative_forecast_tuned("RandomForest", train_bt, last_idx)
    rf_fut = f.iterative_forecast_tuned("RandomForest", prices, future_index)

    gbr_bt = f.iterative_forecast_tuned("GradientBoosting", train_bt, last_idx)
    gbr_fut = f.iterative_forecast_tuned("GradientBoosting", prices, future_index)

    svr_bt = f.iterative_forecast_tuned("SVR", train_bt, last_idx)
    svr_fut = f.iterative_forecast_tuned("SVR", prices, future_index)

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

    return {
        "ticker": ticker,
        "backtest_horizon": backtest_horizon,
        "forecast_horizon": forecast_horizon,
        "prices": prices,
        "actual_last": actual_last,
        "future_index": future_index,
        "preds_last": preds_last,
        "preds_next": preds_next,
    }


def _generate_plot_png(ticker: str) -> bytes:
    d = _compute_forecasts(ticker)
    fig = f.plot_results(
        d["ticker"], d["actual_last"], d["preds_last"], d["preds_next"], d["future_index"], show=False
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.route("/", methods=["GET"])
def index():
    ticker = request.args.get("ticker", default="", type=str).strip().upper()
    img_b64: Optional[str] = None
    message: Optional[str] = None

    if ticker:
        if not f.validate_symbol_format(ticker):
            message = "Geçersiz sembol biçimi. Sadece harf, rakam, '.' ve '-' kullanılabilir."
        else:
            try:
                f.warn_if_not_in_local_sp500(ticker)
            except Exception:
                pass
            try:
                png = _generate_plot_png(ticker)
                img_b64 = base64.b64encode(png).decode("ascii")
            except Exception as e:
                message = str(e)

    return render_template_string(PAGE_TMPL, ticker=ticker, img_data=img_b64, message=message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


@app.get("/plot")
def plot_endpoint():
    ticker = request.args.get("ticker", type=str, default="").strip().upper()
    if not ticker:
        return Response("ticker parametresi gerekli", status=400)
    if not f.validate_symbol_format(ticker):
        return Response("Geçersiz sembol biçimi", status=400)
    try:
        png = _generate_plot_png(ticker)
    except Exception as e:
        return Response(str(e), status=400)
    return Response(png, mimetype="image/png")


def _series_to_json_dict(s: f.pd.Series):
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": [float(v) for v in s.values],
    }


@app.get("/api/forecast")
def api_forecast():
    ticker = request.args.get("ticker", type=str, default="").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker parametresi gerekli"}), 400
    if not f.validate_symbol_format(ticker):
        return jsonify({"error": "Geçersiz sembol biçimi"}), 400
    try:
        d = _compute_forecasts(ticker)
        last_json = _series_to_json_dict(d["actual_last"])
        backtest_json = {k: _series_to_json_dict(v) for k, v in d["preds_last"].items()}
        future_json = {k: _series_to_json_dict(v) for k, v in d["preds_next"].items()}
        resp = {
            "ticker": d["ticker"],
            "backtest_horizon": d["backtest_horizon"],
            "forecast_horizon": d["forecast_horizon"],
            "last": last_json,
            "backtest": backtest_json,
            "future": future_json,
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

