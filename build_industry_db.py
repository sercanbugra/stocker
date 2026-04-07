#!/usr/bin/env python3
"""
One-shot script to build the static industry/sector database for all tracked symbols.

Covers:
  - US: S&P 500 (from Wikipedia)
  - UK: FTSE 100 (from Wikipedia / hardcoded fallback)
  - TR: BIST 100  (from Wikipedia / hardcoded fallback)

Output: data/cache/industry_db.json
  {
    "generated_at": "YYYY-MM-DD",
    "count": N,
    "symbols": {
      "AAPL":  {"name": "...", "sector": "...", "industry": "...", "market": "US", "currency": "USD"},
      "BP.L":  {"name": "...", "sector": "...", "industry": "...", "market": "UK", "currency": "GBp"},
      ...
    }
  }

Usage:
  python build_industry_db.py            # full build (~5-10 min)
  python build_industry_db.py --update   # re-fetch only symbols missing sector/industry
"""

import json
import os
import re
import sys
import time
import concurrent.futures
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Symbol lists
# ---------------------------------------------------------------------------

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"

def _get_sp500():
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": _UA}, timeout=12)
        resp.raise_for_status()
        df = pd.read_html(StringIO(resp.text))[0]
        syms = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
        syms = [s for s in syms if re.fullmatch(r"[A-Z0-9\-]{1,6}", s)]
        print(f"  S&P 500: {len(syms)} symbols from Wikipedia")
        return syms
    except Exception as e:
        print(f"  S&P 500 Wikipedia failed: {e} — using minimal fallback")
        return [
            "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","V","JNJ",
            "WMT","MA","UNH","DIS","BAC","XOM","CVX","PFE","KO","PEP","ABBV",
            "MRK","LLY","TMO","AVGO","COST","CSCO","ABT","DHR","ACN","NKE",
            "ADBE","CRM","TXN","NEE","LIN","HON","IBM","QCOM","AMGN","SBUX",
            "GE","CAT","RTX","BA","MMM","GS","MS","BLK","AXP","SPGI","ICE",
            "GILD","VRTX","REGN","BIIB","ISRG","MDT","BSX","SYK","ZTS","MCK",
            "MO","PM","MDLZ","KHC","GIS","K","CPB","SJM","CAG","TSN","HRL",
            "HD","LOW","TGT","ROST","TJX","NKE","VFC","RL","PVH","HBI",
            "AMT","CCI","PLD","SPG","O","WELL","VTR","EQR","AVB","ESS",
            "F","GM","TM","HMC","STLA","HOG","PCAR","CMI","ALSN",
        ]


def _get_ftse100():
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/FTSE_100_Index",
            headers={"User-Agent": _UA}, timeout=12)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("ticker" in c or "epic" in c or "symbol" in c for c in cols):
                col = next(c for c in df.columns
                           if any(k in str(c).lower() for k in ("ticker", "epic", "symbol")))
                syms = [str(s).strip().upper() + ".L" for s in df[col].dropna()
                        if str(s).strip() and re.fullmatch(r"[A-Z0-9]{1,5}", str(s).strip().upper())]
                if len(syms) >= 50:
                    print(f"  FTSE 100: {len(syms)} symbols from Wikipedia")
                    return syms
    except Exception as e:
        print(f"  FTSE 100 Wikipedia failed: {e} — using hardcoded fallback")
    syms = [
        "AHT.L","AAL.L","ABDN.L","ABF.L","ADM.L","AV.L","AZN.L","AUTO.L",
        "BA.L","BARC.L","BATS.L","BEZ.L","BKG.L","BME.L","BNZL.L","BP.L",
        "BRBY.L","BHP.L","CCH.L","CNA.L","CPG.L","CRDA.L","DCC.L","DGE.L",
        "DPLM.L","EZJ.L","ENT.L","EXPN.L","FCIT.L","FRAS.L","FRES.L","GAW.L",
        "GLEN.L","GSK.L","HALEON.L","HLMA.L","HLN.L","HSBA.L","HSX.L","HIK.L",
        "HWDN.L","IAG.L","IHG.L","III.L","IMB.L","INF.L","ITRK.L","JD.L",
        "KGF.L","LAND.L","LGEN.L","LLOY.L","LSEG.L","MKS.L","MNG.L","MNDI.L",
        "MRO.L","NG.L","NWG.L","NXT.L","OCDO.L","PCG.L","PHNX.L","PRU.L",
        "PSH.L","PSN.L","PSON.L","REL.L","RIO.L","RKT.L","RMV.L","RR.L",
        "RS1.L","SGE.L","SBRY.L","SDR.L","SGRO.L","SHEL.L","SKG.L","SKY.L",
        "SLA.L","SMDS.L","SMIN.L","SMT.L","SN.L","SPX.L","SSE.L","STAN.L",
        "STJ.L","SVT.L","TSCO.L","TW.L","ULVR.L","UU.L","VOD.L","WPP.L",
        "WTB.L","WDS.L",
    ]
    print(f"  FTSE 100: {len(syms)} symbols (hardcoded)")
    return syms


def _get_bist():
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/BIST_100",
            headers={"User-Agent": _UA}, timeout=12)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("ticker" in c or "symbol" in c or "code" in c for c in cols):
                col = next(c for c in df.columns
                           if any(k in str(c).lower() for k in ("ticker", "symbol", "code")))
                syms = [str(s).strip().upper() + ".IS" for s in df[col].dropna()
                        if str(s).strip() and not str(s).strip().endswith(".IS")
                        and re.fullmatch(r"[A-Z0-9]{2,6}", str(s).strip().upper())]
                if len(syms) >= 50:
                    print(f"  BIST: {len(syms)} symbols from Wikipedia")
                    return syms
    except Exception as e:
        print(f"  BIST Wikipedia failed: {e} — using hardcoded fallback")
    syms = [
        "ACSEL.IS","ADEL.IS","AEFES.IS","AGESA.IS","AGHOL.IS","AHGAZ.IS",
        "AKBNK.IS","AKCNS.IS","AKFEN.IS","AKGRT.IS","AKSA.IS","AKSEN.IS",
        "ALARK.IS","ALBRK.IS","ALCAR.IS","ALFAS.IS","ALKIM.IS","ANSGR.IS",
        "ARCLK.IS","ARDYZ.IS","ARSAN.IS","ASELS.IS","ASUZU.IS","AYGAZ.IS",
        "BAGFS.IS","BANVT.IS","BIMAS.IS","BIOEN.IS","BRISA.IS","BRSAN.IS",
        "BRYAT.IS","BUCIM.IS","CCOLA.IS","CIMSA.IS","CLEBI.IS","CMENT.IS",
        "DOAS.IS","DOHOL.IS","ECILC.IS","EGEEN.IS","EKGYO.IS","ENKAI.IS",
        "EREGL.IS","EUPWR.IS","FENER.IS","FROTO.IS","GARAN.IS","GESAN.IS",
        "GUBRF.IS","GWIND.IS","HALKB.IS","HEKTS.IS","HURGZ.IS","INDES.IS",
        "IPEKE.IS","ISCTR.IS","ISGYO.IS","ISMEN.IS","KARSN.IS","KCHOL.IS",
        "KLGYO.IS","KLNMA.IS","KONTR.IS","KONYA.IS","KOZAA.IS","KOZAL.IS",
        "KRDMD.IS","LOGO.IS","MAVI.IS","MGROS.IS","MIATK.IS","NETAS.IS",
        "ODAS.IS","OTKAR.IS","OYAKC.IS","PETKM.IS","PGSUS.IS","POLHO.IS",
        "SAHOL.IS","SASA.IS","SISE.IS","SKBNK.IS","SNGYO.IS","SOKM.IS",
        "TABGD.IS","TAVHL.IS","TCELL.IS","THYAO.IS","TKFEN.IS","TOASO.IS",
        "TSKB.IS","TTKOM.IS","TTRAK.IS","TUPRS.IS","TURSG.IS","ULKER.IS",
        "VAKBN.IS","VESTL.IS","YKBNK.IS","YYLGD.IS",
    ]
    print(f"  BIST: {len(syms)} symbols (hardcoded)")
    return syms


# ---------------------------------------------------------------------------
# Fetch metadata for a single symbol
# ---------------------------------------------------------------------------

def _fetch_meta(symbol: str, market: str) -> dict:
    """Return a metadata dict for one symbol. Never raises."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "name":     info.get("shortName") or info.get("longName") or symbol,
            "sector":   info.get("sector"),
            "industry": info.get("industry"),
            "market":   market,
            "currency": info.get("currency"),
        }
    except Exception:
        return {"name": symbol, "sector": None, "industry": None, "market": market, "currency": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(update_only: bool = False):
    OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "cache", "industry_db.json")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load existing DB if updating
    existing: dict = {}
    if update_only and os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as fh:
            existing = json.load(fh).get("symbols", {})
        print(f"Loaded {len(existing)} existing entries for incremental update")

    print("Collecting symbol lists…")
    tasks: list[tuple[str, str]] = []
    for sym in _get_sp500():
        tasks.append((sym, "US"))
    for sym in _get_ftse100():
        tasks.append((sym, "UK"))
    for sym in _get_bist():
        tasks.append((sym, "TR"))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tasks = []
    for sym, mkt in tasks:
        if sym not in seen:
            seen.add(sym)
            unique_tasks.append((sym, mkt))

    if update_only:
        # Only re-fetch symbols that lack sector/industry
        unique_tasks = [(s, m) for s, m in unique_tasks
                        if s not in existing
                        or not existing[s].get("sector")
                        or not existing[s].get("industry")]
        print(f"  {len(unique_tasks)} symbols need updating")
    else:
        print(f"  Total unique symbols to fetch: {len(unique_tasks)}")

    db: dict = dict(existing)  # start with existing, overwrite below

    WORKERS = 12
    done = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_to_sym = {pool.submit(_fetch_meta, sym, mkt): sym for sym, mkt in unique_tasks}
        for fut in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[fut]
            try:
                db[sym] = fut.result()
            except Exception as e:
                db[sym] = {"name": sym, "sector": None, "industry": None, "market": "US", "currency": None}
            done += 1
            if done % 50 == 0 or done == len(unique_tasks):
                elapsed = time.time() - t0
                eta = (elapsed / done) * (len(unique_tasks) - done)
                print(f"  {done}/{len(unique_tasks)}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    # Summary stats
    with_sector   = sum(1 for v in db.values() if v.get("sector"))
    with_industry = sum(1 for v in db.values() if v.get("industry"))
    by_market = {}
    for v in db.values():
        by_market[v.get("market", "?")] = by_market.get(v.get("market", "?"), 0) + 1

    print(f"\nResults:")
    print(f"  Total symbols : {len(db)}")
    print(f"  Has sector    : {with_sector}")
    print(f"  Has industry  : {with_industry}")
    for mkt, cnt in sorted(by_market.items()):
        print(f"  {mkt:4s}          : {cnt}")

    payload = {
        "generated_at": datetime.utcnow().isoformat()[:10],
        "count": len(db),
        "symbols": db,
    }

    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}  ({os.path.getsize(OUT_PATH):,} bytes)")


if __name__ == "__main__":
    update_only = "--update" in sys.argv
    build(update_only=update_only)
