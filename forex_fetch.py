"""
forex_fetch.py — Daily/weekly OHLC for FX, commodities, indices and shares. (v1.1)

Source: a market-data API (key read from Streamlit secrets or the
TWELVEDATA_KEY environment variable). Four groups, scanned one at a time:

    fx          major + cross currency pairs
    commodities metals, energy, softs
    indices     global stock indices
    shares      US-listed shares

Chart links are built elsewhere from BROKER_SYMBOL so the app can send the
user to their own broker's chart (OANDA / ICMARKETS / CFI) even though the
candles come from the data provider — prices differ only by spread, which
is immaterial at daily resolution for zone work.

v1.1: batched requests. The provider accepts a comma-separated symbol
list and returns all of them in ONE call, so a 36-symbol group costs ~2
credits instead of 36 — essential on plans metered per minute (e.g. 8/min).
Batches are prefetched into the disk cache before the scan loop runs; the
per-symbol path then serves from cache. Falls back to single requests if a
batch response is unusable.

Provider name never surfaces in the UI; failures land in LAST_ERRORS
tagged FX-API for the operator diagnostics panel.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime

import pandas as pd
import requests

API_HOST = "https://api.twelvedata.com"
CACHE_DIR = "/tmp/fibgp_cache"
CANDLE_TTL = 6 * 3600          # these markets close daily; 6h cache is ample

LAST_ERRORS: dict[str, str] = {}
_local = threading.local()

# global pacing — plans meter requests per minute (Basic 8 = 8/min)
MIN_GAP = 8.0                  # ~7.5/min, just inside the cap
BATCH_SIZE = 20                # symbols per batched request
_pace_lock = threading.Lock()
_last_call = [0.0]

# breaker: stop hammering when the plan's limit is hit
_BREAK_AFTER = 4
_COOLDOWN = 120.0
_thr = {"fails": 0, "down_until": 0.0}
_thr_lock = threading.Lock()


# ============================== SYMBOL TABLE ==================================
# (data_symbol, display, broker_symbol_for_charts)
FX_PAIRS = [
    ("EUR/USD", "EURUSD", "OANDA:EURUSD"), ("GBP/USD", "GBPUSD", "OANDA:GBPUSD"),
    ("USD/JPY", "USDJPY", "OANDA:USDJPY"), ("USD/CHF", "USDCHF", "OANDA:USDCHF"),
    ("USD/CAD", "USDCAD", "OANDA:USDCAD"), ("AUD/USD", "AUDUSD", "OANDA:AUDUSD"),
    ("NZD/USD", "NZDUSD", "OANDA:NZDUSD"), ("EUR/GBP", "EURGBP", "OANDA:EURGBP"),
    ("EUR/JPY", "EURJPY", "OANDA:EURJPY"), ("EUR/CHF", "EURCHF", "OANDA:EURCHF"),
    ("EUR/AUD", "EURAUD", "OANDA:EURAUD"), ("EUR/CAD", "EURCAD", "OANDA:EURCAD"),
    ("EUR/NZD", "EURNZD", "OANDA:EURNZD"), ("GBP/JPY", "GBPJPY", "OANDA:GBPJPY"),
    ("GBP/CHF", "GBPCHF", "OANDA:GBPCHF"), ("GBP/AUD", "GBPAUD", "OANDA:GBPAUD"),
    ("GBP/CAD", "GBPCAD", "OANDA:GBPCAD"), ("GBP/NZD", "GBPNZD", "OANDA:GBPNZD"),
    ("AUD/JPY", "AUDJPY", "OANDA:AUDJPY"), ("AUD/CHF", "AUDCHF", "OANDA:AUDCHF"),
    ("AUD/CAD", "AUDCAD", "OANDA:AUDCAD"), ("AUD/NZD", "AUDNZD", "OANDA:AUDNZD"),
    ("CAD/JPY", "CADJPY", "OANDA:CADJPY"), ("CAD/CHF", "CADCHF", "OANDA:CADCHF"),
    ("CHF/JPY", "CHFJPY", "OANDA:CHFJPY"), ("NZD/JPY", "NZDJPY", "OANDA:NZDJPY"),
    ("NZD/CHF", "NZDCHF", "OANDA:NZDCHF"), ("NZD/CAD", "NZDCAD", "OANDA:NZDCAD"),
    ("USD/SGD", "USDSGD", "OANDA:USDSGD"), ("USD/HKD", "USDHKD", "OANDA:USDHKD"),
    ("USD/MXN", "USDMXN", "OANDA:USDMXN"), ("USD/ZAR", "USDZAR", "OANDA:USDZAR"),
    ("USD/TRY", "USDTRY", "OANDA:USDTRY"), ("USD/SEK", "USDSEK", "OANDA:USDSEK"),
    ("USD/NOK", "USDNOK", "OANDA:USDNOK"), ("USD/PLN", "USDPLN", "OANDA:USDPLN"),
]

COMMODITIES = [
    ("XAU/USD", "XAU", "OANDA:XAUUSD"), ("XAG/USD", "XAG", "OANDA:XAGUSD"),
    ("XPT/USD", "XPTUSD", "OANDA:XPTUSD"), ("XPD/USD", "XPDUSD", "OANDA:XPDUSD"),
    ("WTI/USD", "WTI", "CFI:WTI"), ("BRENT/USD", "BRENT", "OANDA:BCOUSD"),
    ("NG/USD", "NAT.GAS", "OANDA:NATGASUSD"),
    ("XCU/USD", "COPPER", "OANDA:XCUUSD"),
    ("ALI/USD", "ALUMINIUM", "OANDA:ALUMINIUMUSD"),
    ("NI/USD", "NICKEL", "OANDA:NICKELUSD"),
    ("ZNC/USD", "ZINC", "OANDA:ZINCUSD"),
    ("LEAD/USD", "LEAD", "OANDA:LEADUSD"),
    ("GASOIL/USD", "GASOIL", "ICMARKETS:GASOIL"),
    ("COCOA/USD", "COCOA", "ICMARKETS:COCOA"),
    ("KC/USD", "COFFEE.AR", "ICMARKETS:COFFEE.AR"),
    ("RC/USD", "COFFEE.ROB", "ICMARKETS:COFFEE.ROB"),
    ("CT/USD", "COTTON", "ICMARKETS:COTTON"),
    ("OJ/USD", "ORANGE.JUICE", "ICMARKETS:ORANGE.JUICE"),
    ("SB/USD", "SUGAR.RAW", "ICMARKETS:SUGAR.RAW"),
    ("SW/USD", "SUGAR.WHITE", "ICMARKETS:SUGAR.WHITE"),
]

INDICES = [
    ("SPX", "US500", "ICMARKETS:US500"), ("NDX", "USTEC", "ICMARKETS:USTEC"),
    ("DJI", "US30", "ICMARKETS:US30"), ("UKX", "UK100", "ICMARKETS:UK100"),
    ("DAX", "GER40", "ICMARKETS:GER40"), ("MDAX", "GERMID50", "ICMARKETS:GERMID50"),
    ("TECDAX", "GERTEC", "ICMARKETS:GERTEC"), ("CAC", "FRANCE", "ICMARKETS:FRANCE40"),
    ("SX5E", "EUR50", "ICMARKETS:EUR50"), ("AEX", "NED25", "ICMARKETS:NED25"),
    ("IBEX", "SPAIN", "ICMARKETS:SPAIN35"), ("SSMI", "SWISS", "ICMARKETS:SWISS20"),
    ("OSEAX", "NOR25", "ICMARKETS:NOR25"), ("N225", "JAPAN", "ICMARKETS:JAPAN225"),
    ("HSI", "HK-HSI", "ICMARKETS:HK-HSI"), ("AXJO", "AUS200", "ICMARKETS:AUS200"),
    ("GSPTSE", "CAN60", "ICMARKETS:CAN60"), ("HSCEI", "CHINAH", "ICMARKETS:CHINAH"),
    ("TOP40", "SA40", "ICMARKETS:SA40"),
]

SHARES = [(t, t, f"ICMARKETS:{t}") for t in
          ("AAPL", "AMD", "AMZN", "BA", "COIN", "GOOG", "INTC", "META",
           "MSTR", "NFLX", "NVDA", "PLTR", "SHOP", "SMCI", "TSLA", "UBER")]

GROUPS = {"fx": FX_PAIRS, "commodities": COMMODITIES,
          "indices": INDICES, "shares": SHARES}

# display -> (data_symbol, broker_symbol)
_LOOKUP = {disp: (data, broker)
           for rows in GROUPS.values() for data, disp, broker in rows}


def list_symbols_fx(group: str = "all") -> tuple[list[str], str]:
    """Symbols for one group, or every group when called with 'all'."""
    if group in ("all", "", None):
        rows = [r for g in ("fx", "commodities", "indices", "shares")
                for r in GROUPS[g]]
    else:
        rows = GROUPS.get(group, [])
    return [disp for _, disp, _ in rows], f"{group} list"


# backwards-compatible alias
list_symbols = list_symbols_fx


def tv_symbol_fx(display: str) -> str:
    """TradingView symbol at the user's own broker (OANDA / ICMARKETS / CFI)."""
    return _LOOKUP.get(display, (None, display))[1]


# ============================== PLUMBING ======================================
def _api_key() -> str | None:
    key = os.environ.get("TWELVEDATA_KEY")
    if key:
        return key
    try:                                   # Streamlit secrets, if present
        import streamlit as st
        return st.secrets.get("TWELVEDATA_KEY")
    except Exception:
        return None


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def _pace():
    with _pace_lock:
        wait = MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def _breaker_open() -> bool:
    with _thr_lock:
        return time.time() < _thr["down_until"]


def _breaker_report(ok: bool):
    with _thr_lock:
        if ok:
            _thr["fails"] = 0
        else:
            _thr["fails"] += 1
            if _thr["fails"] >= _BREAK_AFTER:
                _thr["down_until"] = time.time() + _COOLDOWN
                _thr["fails"] = 0


def _cache_path(display: str, interval: str) -> str:
    safe = display.replace("/", "_").replace(".", "-")
    return os.path.join(CACHE_DIR, f"fx_{interval}_{safe}.csv")


def _cache_read(display: str, interval: str) -> pd.DataFrame | None:
    p = _cache_path(display, interval)
    try:
        if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < CANDLE_TTL:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if {"Open", "High", "Low", "Close"}.issubset(df.columns) and len(df):
                return df
    except Exception:
        pass
    return None


def _cache_write(display: str, interval: str, df: pd.DataFrame):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(_cache_path(display, interval))
    except Exception:
        pass


# ============================== BATCH PREFETCH ================================
def _rows_to_frame(values) -> pd.DataFrame | None:
    recs = []
    for v in values or []:
        try:
            recs.append({"Date": pd.Timestamp(v["datetime"][:10]),
                         "Open": float(v["open"]), "High": float(v["high"]),
                         "Low": float(v["low"]), "Close": float(v["close"]),
                         "Volume": float(v.get("volume") or 0)})
        except (KeyError, ValueError, TypeError):
            continue
    if not recs:
        return None
    df = pd.DataFrame(recs).set_index("Date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def prefetch_group(displays: list[str], interval: str = "1d") -> tuple[int, int]:
    """Fetch a whole group in batched calls and fill the disk cache.
    Returns (cached_now, requested). Safe to call before every scan — symbols
    already fresh in cache are skipped."""
    key = _api_key()
    todo = [d for d in displays if _cache_read(d, interval) is None]
    if not todo or not key:
        return len(displays) - len(todo), len(displays)

    filled = 0
    for i in range(0, len(todo), BATCH_SIZE):
        if _breaker_open():
            break
        chunk = todo[i:i + BATCH_SIZE]
        data_syms = [_LOOKUP.get(d, (d, None))[0] for d in chunk]
        params = {"symbol": ",".join(data_syms),
                  "interval": "1week" if interval == "1w" else "1day",
                  "outputsize": 5000, "order": "ASC",
                  "format": "JSON", "apikey": key}
        _pace()
        try:
            r = _session().get(API_HOST + "/time_series", params=params, timeout=60)
            payload = r.json() if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            payload = None
        if not isinstance(payload, dict):
            _breaker_report(False)
            continue

        # single-symbol responses come back flat; multi-symbol keyed by symbol
        if "values" in payload:
            payload = {data_syms[0]: payload}

        got = False
        for disp, dsym in zip(chunk, data_syms):
            entry = payload.get(dsym)
            if not isinstance(entry, dict):
                continue
            if entry.get("status") == "error":
                LAST_ERRORS[disp] = f"FX-API {str(entry.get('message',''))[:70]}"
                continue
            df = _rows_to_frame(entry.get("values"))
            if df is not None:
                _cache_write(disp, interval, df)
                LAST_ERRORS.pop(disp, None)
                filled += 1
                got = True
        _breaker_report(got)
        if not got and isinstance(payload.get("message"), str):
            msg = payload["message"].lower()
            if "credit" in msg or "limit" in msg or "quota" in msg:
                with _thr_lock:
                    _thr["down_until"] = time.time() + _COOLDOWN
                break
    return filled + (len(displays) - len(todo)), len(displays)


# ============================== FETCH =========================================
def fetch_daily_fx(display: str, start: date,
                   interval: str = "1d") -> pd.DataFrame | None:
    """Daily/weekly OHLC for one symbol. 6h disk cache."""
    cached = _cache_read(display, interval)
    if cached is not None:
        cut = cached.loc[cached.index.date >= start]
        if len(cut):
            LAST_ERRORS.pop(display, None)
            return cut

    key = _api_key()
    if not key:
        LAST_ERRORS[display] = "FX-API key missing (add TWELVEDATA_KEY to secrets)"
        return None
    if _breaker_open():
        LAST_ERRORS[display] = "FX-API cooling down (limit reached)"
        return None

    data_sym = _LOOKUP.get(display, (display, None))[0]
    params = {"symbol": data_sym,
              "interval": "1week" if interval == "1w" else "1day",
              "outputsize": 5000, "order": "ASC",
              "format": "JSON", "apikey": key}
    _pace()
    try:
        r = _session().get(API_HOST + "/time_series", params=params, timeout=30)
        if r.status_code != 200:
            LAST_ERRORS[display] = f"FX-API HTTP {r.status_code}"
            _breaker_report(False)
            return None
        payload = r.json()
    except requests.RequestException as e:
        LAST_ERRORS[display] = f"FX-API {type(e).__name__}: {str(e)[:70]}"
        _breaker_report(False)
        return None
    except ValueError:
        LAST_ERRORS[display] = "FX-API bad JSON"
        _breaker_report(False)
        return None

    if isinstance(payload, dict) and payload.get("status") == "error":
        msg = str(payload.get("message", ""))[:120]
        LAST_ERRORS[display] = f"FX-API {msg}"
        # credit/limit exhaustion is precisely when we must stop hammering
        _breaker_report(False)
        low = msg.lower()
        if "per minute" in low or "minutely" in low:
            with _thr_lock:                     # short pause for a minute cap
                _thr["down_until"] = time.time() + 65
                _thr["fails"] = 0
            return None
        if "credit" in low or "limit" in low or "quota" in low:
            with _thr_lock:                     # trip immediately, don't wait
                _thr["down_until"] = time.time() + _COOLDOWN
                _thr["fails"] = 0
        return None

    values = payload.get("values") if isinstance(payload, dict) else None
    if not values:
        LAST_ERRORS[display] = "FX-API no data returned"
        _breaker_report(False)
        return None

    recs = []
    for v in values:
        try:
            recs.append({"Date": pd.Timestamp(v["datetime"][:10]),
                         "Open": float(v["open"]), "High": float(v["high"]),
                         "Low": float(v["low"]), "Close": float(v["close"]),
                         "Volume": float(v.get("volume") or 0)})
        except (KeyError, ValueError, TypeError):
            continue
    if not recs:
        LAST_ERRORS[display] = "FX-API unparseable rows"
        _breaker_report(False)
        return None

    df = pd.DataFrame(recs).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    _cache_write(display, interval, df)
    _breaker_report(True)
    LAST_ERRORS.pop(display, None)
    out = df.loc[df.index.date >= start]
    return out if len(out) else df
