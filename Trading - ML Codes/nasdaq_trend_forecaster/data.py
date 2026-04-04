import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests
import yfinance as yf

NASDAQ_LISTING_URL = (
    "https://ftp.nasdaqtrader.com/dynamic/SymbolDirectory/nasdaqtraded.txt"
)
DEFAULT_CACHE_DIR = Path("data_cache")
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_nasdaq_tickers(
    cache_path: Path = DEFAULT_CACHE_DIR / "nasdaq_tickers.csv",
    allow_network: bool = True,
    fallback_symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Download the latest NASDAQ traded symbols.

    The result is cached to avoid repeated network calls. The returned dataframe
    has columns: Symbol, SecurityName, MarketCategory, ETF.
    """
    if cache_path.exists():
        return pd.read_csv(cache_path)

    if allow_network:
        try:
            response = requests.get(NASDAQ_LISTING_URL, timeout=15)
            response.raise_for_status()
        except Exception:
            # Fall back to offline path if available below.
            response = None
    else:
        response = None

    if response is not None:
        # The NASDAQ file is pipe-delimited with a footer line that starts with "File"
        raw = io.StringIO(response.text)
        df = pd.read_csv(raw, sep="|")
        df = df[df["ETF"] == "N"]  # keep equities only
        df = df[df["Test Issue"] == "N"]
        df = df[["Symbol", "Security Name", "Market Category", "ETF"]].rename(
            columns={
                "Symbol": "symbol",
                "Security Name": "security_name",
                "Market Category": "market_category",
                "ETF": "is_etf",
            }
        )
        df.to_csv(cache_path, index=False)
        return df

    # Offline fallback
    if fallback_symbols is None:
        fallback_symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
    return pd.DataFrame(
        {
            "symbol": list(fallback_symbols),
            "security_name": ["offline-fallback"] * len(list(fallback_symbols)),
            "market_category": ["Q"] * len(list(fallback_symbols)),
            "is_etf": ["N"] * len(list(fallback_symbols)),
        }
    )


def download_price_history(
    symbol: str, months: int = 6, interval: str = "1d"
) -> pd.DataFrame:
    """
    Grab recent history for a symbol using yfinance.

    Returns a dataframe with a DatetimeIndex and columns: Open, High, Low, Close, Adj Close, Volume.
    """
    ticker = yf.Ticker(symbol)
    period = f"{months}mo"
    history = ticker.history(period=period, interval=interval, auto_adjust=True)
    if history.empty:
        return history

    history = history.reset_index().rename(columns={"Date": "date"}).set_index("date")
    history.index = history.index.tz_localize(None)
    return history


def download_bulk_history(
    symbols: Iterable[str], months: int = 6, interval: str = "1d"
) -> List[tuple[str, pd.DataFrame]]:
    """
    Download historical data for a batch of symbols.

    Returns a list of (symbol, dataframe) tuples so callers can keep errors isolated.
    """
    results: List[tuple[str, pd.DataFrame]] = []
    for symbol in symbols:
        try:
            df = download_price_history(symbol, months=months, interval=interval)
            if df.empty:
                continue
            results.append((symbol, df))
        except Exception:
            # Ignore symbols that fail to download; caller can log if desired.
            continue
    return results


def recent_window(df: pd.DataFrame, days: int = 180) -> pd.DataFrame:
    """Trim dataframe to the last N days based on index."""
    if df.empty:
        return df
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    return df[df.index.date >= cutoff]
