import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Allow running as a script: python nasdaq_trend_forecaster/cli.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from nasdaq_trend_forecaster.data import download_bulk_history, fetch_nasdaq_tickers
    from nasdaq_trend_forecaster.forecast import ForecastResult, make_forecast
else:
    from .data import download_bulk_history, fetch_nasdaq_tickers
    from .forecast import ForecastResult, make_forecast


def _iter_symbols(args) -> Iterable[str]:
    if args.symbols:
        for s in args.symbols:
            yield s.upper().strip()
        return
    listing = fetch_nasdaq_tickers(allow_network=not args.offline)
    symbols = listing["symbol"].tolist()
    if args.limit:
        symbols = symbols[: args.limit]
    for s in symbols:
        yield s


def run_forecasts(args) -> List[ForecastResult]:
    results: List[ForecastResult] = []
    for symbol, history in download_bulk_history(_iter_symbols(args), months=args.months):
        forecast = make_forecast(
            symbol,
            history,
            min_points=args.min_points,
            min_slope=args.min_slope,
            min_r2=args.min_r2,
        )
        if forecast:
            results.append(forecast)
    return results


def _results_to_dataframe(results: List[ForecastResult]) -> pd.DataFrame:
    data = []
    for r in results:
        data.append(
            {
                "symbol": r.symbol,
                "slope": r.slope,
                "r2": r.r2,
                "next_min_date": r.next_min_date,
                "predicted_min_price": r.predicted_min_price,
                "next_max_date": r.next_max_date,
                "predicted_max_price": r.predicted_max_price,
                "suggested_buy_price": r.suggested_buy_price,
                "note": r.note,
            }
        )
    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NASDAQ hisselerinde yükselen trend ve salınım tahmini üretir."
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Kaç aylık geçmiş veri kullanılacak (varsayılan 6).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="İşlenecek sembol sayısı (tam liste için 0 veya negatif). Varsayılan 200.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Belirli sembollerle çalış (boş bırakılırsa tüm NASDAQ listesi çekilir).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("nasdaq_forecasts.csv"),
        help="Çıktı CSV dosya yolu.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Ağ erişimi olmadan çalış (hazır cache veya dahili sembol listesi kullanılır).",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.05,
        help="Lineer trend için minimum R² eşiği (daha düşük = daha fazla ihtimal). Varsayılan 0.05.",
    )
    parser.add_argument(
        "--min-slope",
        type=float,
        default=0.0,
        help="Lineer trend için minimum eğim eşiği (fiyat birimi/gün). Varsayılan 0.0.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=20,
        help="Bir sembolün değerlendirilmesi için minimum veri noktası sayısı. Varsayılan 20.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit and args.limit < 1:
        args.limit = None

    results = run_forecasts(args)
    df = _results_to_dataframe(results)
    df.to_csv(args.out, index=False)

    print(f"Oluşturulan tahmin sayısı: {len(results)}")
    print(f"Dosya: {args.out.resolve()}")

    if not results:
        print("Hiçbir sembol kriterlere uymadı.")
        return

    top = df.sort_values(by=["r2", "slope"], ascending=False).head(10)
    print("\nEn yüksek güvene sahip 10 tahmin:")
    print(top[["symbol", "r2", "slope", "predicted_min_price", "predicted_max_price", "suggested_buy_price"]])


if __name__ == "__main__":
    main()
