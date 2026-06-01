#!/usr/bin/env python3
"""
Build comprehensive autocomplete database for the ticker search input.

Sources:
  US  : Nasdaq Trader public files (nasdaqlisted.txt + otherlisted.txt) — no API key needed
  TR  : data/bist_symbols.txt  + names from Yahoo Finance API
  UK  : data/ftse100_symbols.txt + names from Yahoo Finance API
  All : Enriched/overridden with names from data/cache/industry_db.json

Output: data/autocomplete_db.json
  [{"s": "AAPL", "n": "Apple Inc.", "m": "US"}, ...]

Usage:
  python build_autocomplete_db.py
"""

import json
import os
import re
import sys
import time
import concurrent.futures
from datetime import datetime

import requests

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"
_SYM_RE_US = re.compile(r"^[A-Z][A-Z0-9]{0,5}$")   # pure alpha-numeric, starts with letter

# Verbose suffixes to strip from Nasdaq Trader security names
_NAME_SUFFIXES = [
    " - Class A Common Stock", " - Class B Common Stock", " - Class C Common Stock",
    " Class A Common Stock", " Class B Common Stock", " Class C Common Stock",
    " - Common Stock", " Common Stock",
    " - Ordinary Shares", " Ordinary Shares",
    " - American Depositary Shares", " American Depositary Shares",
    " - American Depositary Receipts", " American Depositary Receipts",
    " - American Depositary Share", " American Depositary Share",
]


def _clean_name(name: str, max_len: int = 55) -> str:
    name = name.strip().rstrip(".")
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)].rstrip(",").rstrip()
            break
    return name[:max_len]


# ---------------------------------------------------------------------------
# US: Nasdaq Trader
# ---------------------------------------------------------------------------

def _fetch_nasdaq_listed() -> dict[str, str]:
    """Return {symbol: name} for NASDAQ-listed stocks."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
        resp.raise_for_status()
        result = {}
        for line in resp.text.splitlines()[1:]:           # skip header
            if line.startswith("File Creation"):
                continue
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym, name, _, test, status = parts[0], parts[1], parts[2], parts[3], parts[4]
            if test == "Y" or status != "N":              # skip test/delisted
                continue
            if not _SYM_RE_US.match(sym):
                continue
            result[sym] = _clean_name(name)
        print(f"  NASDAQ listed  : {len(result):,} symbols")
        return result
    except Exception as e:
        print(f"  NASDAQ listed failed: {e}")
        return {}


def _fetch_other_listed() -> dict[str, str]:
    """Return {symbol: name} for NYSE/AMEX/Arca listed stocks."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
        resp.raise_for_status()
        result = {}
        for line in resp.text.splitlines()[1:]:           # skip header
            if line.startswith("File Creation"):
                continue
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym, name, exch, _, _, _, test = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
            if test == "Y":
                continue
            if exch not in ("N", "A", "P", "Z"):          # NYSE, AMEX, NYSE Arca, BATS
                continue
            if not _SYM_RE_US.match(sym):
                continue
            result[sym] = _clean_name(name)
        print(f"  Other (NYSE/AMEX): {len(result):,} symbols")
        return result
    except Exception as e:
        print(f"  Other listed failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# TR/UK: Yahoo Finance chart API (no library needed)
# ---------------------------------------------------------------------------

def _yf_name(symbol: str) -> str | None:
    """Fetch shortName from Yahoo Finance chart API. Returns None on failure."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=8)
        if not r.ok:
            return None
        meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
        return meta.get("shortName") or meta.get("longName") or None
    except Exception:
        return None


def _fetch_names_batch(symbols: list[str], market: str, workers: int = 12) -> dict[str, str]:
    """Parallel-fetch company names from Yahoo Finance for a list of symbols."""
    result: dict[str, str] = {}
    done = 0
    total = len(symbols)

    def _fetch_one(sym):
        return sym, _yf_name(sym)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in concurrent.futures.as_completed(futures):
            sym, name = fut.result()
            result[sym] = name or sym
            done += 1
            if done % 50 == 0 or done == total:
                print(f"    {market}: {done}/{total} names fetched")
    return result


def _fetch_bist_tradingview() -> dict[str, str]:
    """Fetch all BIST symbols + company names from TradingView scanner (no auth needed).
    Returns {SYMBOL.IS: name}.  Also updates data/bist_symbols.txt."""
    try:
        resp = requests.post(
            "https://scanner.tradingview.com/turkey/scan",
            headers={"User-Agent": _UA, "Content-Type": "application/json"},
            json={
                "filter": [], "options": {"lang": "en"},
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name", "description"],
                "sort": {"sortBy": "name", "sortOrder": "asc"},
                "range": [0, 1000],
            },
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("data", [])
        names: dict[str, str] = {}
        for row in results:
            if len(row.get("d", [])) >= 2:
                sym = row["d"][0] + ".IS"
                names[sym] = row["d"][1][:55]
        if len(names) >= 100:
            # Update bist_symbols.txt with the fresh list
            base_dir = os.path.dirname(os.path.abspath(__file__))
            sym_path = os.path.join(base_dir, "data", "bist_symbols.txt")
            header = "# BIST listed stocks — Yahoo Finance tickers (.IS suffix)\n"
            header += f"# Last updated: {datetime.utcnow().isoformat()[:10]} (TradingView source, ~{len(names)} symbols)\n"
            content = header + "\n".join(sorted(names.keys())) + "\n"
            with open(sym_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            print(f"  BIST (TradingView): {len(names)} symbols  →  updated {sym_path}")
            return names
    except Exception as e:
        print(f"  TradingView BIST fetch failed: {e}")
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path  = os.path.join(base_dir, "data", "autocomplete_db.json")

    # ── Load existing industry_db for curated names ──────────────────────────
    industry_db_path = os.path.join(base_dir, "data", "cache", "industry_db.json")
    industry_names: dict[str, str] = {}
    industry_market: dict[str, str] = {}
    if os.path.exists(industry_db_path):
        with open(industry_db_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        syms = raw.get("symbols", raw)
        for sym, meta in syms.items():
            if meta.get("name") and meta["name"] != sym:
                industry_names[sym] = meta["name"]
            if meta.get("market"):
                industry_market[sym] = meta["market"]
        print(f"Industry DB: {len(industry_names)} named symbols loaded\n")

    # ── US stocks ─────────────────────────────────────────────────────────────
    print("Fetching US stock lists from Nasdaq Trader...")
    nasdaq  = _fetch_nasdaq_listed()
    other   = _fetch_other_listed()

    us_stocks: dict[str, str] = {**other, **nasdaq}   # nasdaq wins on conflict
    # Override with curated industry_db names
    for sym, name in industry_names.items():
        if industry_market.get(sym) == "US":
            us_stocks[sym] = name   # prefer curated name

    print(f"  Total unique US symbols: {len(us_stocks):,}\n")

    # ── BIST: TradingView (full ~600 symbols + names), falls back to file + Yahoo ─
    print("Fetching BIST from TradingView...")
    bist_names: dict[str, str] = _fetch_bist_tradingview()

    if not bist_names:
        bist_sym_path = os.path.join(base_dir, "data", "bist_symbols.txt")
        bist_syms_fallback: list[str] = []
        if os.path.exists(bist_sym_path):
            with open(bist_sym_path, encoding="utf-8") as fh:
                bist_syms_fallback = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        print(f"  Fallback: {len(bist_syms_fallback)} symbols from file")
        bist_names = {s: industry_names[s] for s in bist_syms_fallback if s in industry_names}
        bist_missing = [s for s in bist_syms_fallback if s not in bist_names]
        if bist_missing:
            print(f"  Fetching names for {len(bist_missing)} BIST symbols from Yahoo Finance...")
            bist_names.update(_fetch_names_batch(bist_missing, "BIST"))

    # Override with curated industry_db names
    for sym, name in industry_names.items():
        if sym.endswith(".IS") and sym in bist_names:
            bist_names[sym] = name

    # ── FTSE ──────────────────────────────────────────────────────────────────
    ftse_sym_path = os.path.join(base_dir, "data", "ftse100_symbols.txt")
    ftse_syms: list[str] = []
    if os.path.exists(ftse_sym_path):
        with open(ftse_sym_path, encoding="utf-8") as fh:
            ftse_syms = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    print(f"FTSE symbols: {len(ftse_syms)}")

    ftse_names: dict[str, str] = {s: industry_names[s] for s in ftse_syms if s in industry_names}
    ftse_missing = [s for s in ftse_syms if s not in ftse_names]
    if ftse_missing:
        print(f"Fetching names for {len(ftse_missing)} FTSE symbols from Yahoo Finance...")
        ftse_names.update(_fetch_names_batch(ftse_missing, "FTSE"))

    # ── Assemble output ───────────────────────────────────────────────────────
    result: list[dict] = []
    seen: set[str] = set()

    for sym, name in sorted(us_stocks.items()):
        seen.add(sym)
        result.append({"s": sym, "n": name, "m": "US"})

    for sym in sorted(bist_names):
        if sym not in seen:
            seen.add(sym)
            result.append({"s": sym, "n": bist_names[sym], "m": "TR"})

    for sym in ftse_syms:
        if sym not in seen:
            seen.add(sym)
            result.append({"s": sym, "n": ftse_names.get(sym, sym), "m": "UK"})

    # ── Stats ─────────────────────────────────────────────────────────────────
    by_market: dict[str, int] = {}
    for item in result:
        by_market[item["m"]] = by_market.get(item["m"], 0) + 1
    print(f"\nTotal autocomplete entries: {len(result):,}")
    for m, cnt in sorted(by_market.items()):
        print(f"  {m:4s}: {cnt:,}")

    # ── Write ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, out_path)
    size = os.path.getsize(out_path)
    print(f"\nSaved → {out_path}  ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
