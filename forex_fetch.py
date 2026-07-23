"""
forex_fetch.py — Daily/weekly OHLC for FX, commodities, indices and shares. (v1.5)

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

v1.5: adds a hard per-scan request budget (MAX_REQUESTS_PER_SCAN) and a
REQUEST_LOG the operator panel displays, so credit use is bounded by
construction and visible after every scan.

v1.4: credit-aware pacing. The provider meters CREDITS (one per symbol,
even inside a batch), 8/minute on the basic plan — so batches are cut to 8
symbols and spaced ~61s apart. A group scan takes ceil(n/8) minutes once,
then serves from the 6h cache. Per-minute limit responses wait and retry
the same chunk; daily-quota responses stop the run. After a group prefetch,
uncached symbols report their error instead of stampeding into per-symbol
calls.

v1.2: self-batching. fetch_daily_fx() itself pulls the whole group in one
batched call the first time it is asked for any symbol in that group, so
batching cannot be bypassed by an older call path. A scan of any group
costs 1-2 credits total.

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

# ---- hard budget: the module cannot exceed this many requests per scan ----
MAX_REQUESTS_PER_SCAN = 10
REQUEST_LOG: list[str] = []          # what was actually sent, for diagnostics
_budget_lock = threading.Lock()


def reset_budget():
    """Called by the app at the start of each scan."""
    with _budget_lock:
        REQUEST_LOG.clear()
        _batched_groups.clear()


def _spend_request(symbol_list: str) -> bool:
    """Record a request; return False if the budget is exhausted."""
    with _budget_lock:
        if len(REQUEST_LOG) >= MAX_REQUESTS_PER_SCAN:
            return False
        REQUEST_LOG.append(symbol_list[:120])
        return True

# global pacing — plans meter requests per minute (Basic 8 = 8/min)
MIN_GAP = 8.0                  # single-call pacing (custom symbols)
BATCH_SIZE = 8                 # credits per batch == symbols; 8 = the minute cap
BATCH_WAIT = 61.0              # seconds between batch calls (credit window reset)
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

# US shares: TradingView resolves bare tickers to the primary listing.
# (Broker CFD prefixes are not published for individual shares.)
SHARES = [(t, t, t) for t in
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


_GROUP_OF = {disp: g for g, rows in GROUPS.items() for _, disp, _ in rows}


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
    chunks = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    for ci, chunk in enumerate(chunks):
        if _breaker_open():
            break
        if ci:
            time.sleep(BATCH_WAIT)             # let the credit window reset
        data_syms = [_LOOKUP.get(d, (d, None))[0] for d in chunk]
        if not _spend_request(f"BATCH[{len(data_syms)}] {','.join(data_syms)}"):
            break
        params = {"symbol": ",".join(data_syms),
                  "interval": "1week" if interval == "1w" else "1day",
                  "outputsize": 5000, "order": "ASC",
                  "format": "JSON", "apikey": key}

        payload = None
        for attempt in range(3):               # minute-limit -> wait, retry chunk
            try:
                r = _session().get(API_HOST + "/time_series", params=params,
                                   timeout=60)
                payload = r.json() if r.status_code == 200 else None
            except (requests.RequestException, ValueError):
                payload = None
            if isinstance(payload, dict) and payload.get("status") == "error" \
                    and "values" not in payload:
                msg = str(payload.get("message", "")).lower()
                if "current minute" in msg or "per minute" in msg:
                    time.sleep(BATCH_WAIT)
                    continue
                if "current day" in msg or "daily" in msg:
                    with _thr_lock:            # day quota gone: stop the run
                        _thr["down_until"] = time.time() + 3600
                    for d in chunk:
                        LAST_ERRORS[d] = "FX-API daily quota exhausted"
                    return filled + (len(displays) - len(todo)), len(displays)
                for d in chunk:
                    LAST_ERRORS[d] = f"FX-API {str(payload.get('message',''))[:80]}"
                payload = None
            break
        if not isinstance(payload, dict):
            _breaker_report(False)
            continue

        # Response shapes:
        #   single symbol -> {"meta":{...},"values":[...],"status":"ok"}
        #   multi symbol  -> {"AAPL":{"meta":..,"values":[..],"status":"ok"},
        #                     "MSFT":{...}, "status":"ok"}
        # NOTE: the outer dict carries its own "status" key alongside the
        # symbol keys, so presence of "values" (not absence of "status") is
        # what distinguishes the flat form.
        if "values" in payload:
            payload = {data_syms[0]: payload}

        got = False
        for disp, dsym in zip(chunk, data_syms):
            entry = payload.get(dsym)
            if not isinstance(entry, dict):
                # provider may key by the symbol without the slash, or upper-case
                for alt in (dsym.replace("/", ""), dsym.upper(),
                            dsym.replace("/", "").upper()):
                    if isinstance(payload.get(alt), dict):
                        entry = payload[alt]
                        break
            if not isinstance(entry, dict):
                LAST_ERRORS[disp] = "FX-API symbol missing from batch response"
                continue
            if entry.get("status") == "error" or "values" not in entry:
                LAST_ERRORS[disp] = f"FX-API {str(entry.get('message', 'no values'))[:70]}"
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
_batched_groups: set[tuple[str, str]] = set()


def fetch_daily_fx(display: str, start: date,
                   interval: str = "1d") -> pd.DataFrame | None:
    """Daily/weekly OHLC for one symbol. 6h disk cache.

    The first call for any symbol in a group triggers ONE batched request
    that caches the entire group — so a whole scan costs 1-2 credits even
    when callers ask symbol by symbol."""
    cached = _cache_read(display, interval)
    if cached is not None:
        cut = cached.loc[cached.index.date >= start]
        if len(cut):
            LAST_ERRORS.pop(display, None)
            return cut

    grp = _GROUP_OF.get(display)
    if grp and (grp, interval) not in _batched_groups:
        _batched_groups.add((grp, interval))
        try:
            prefetch_group([d for _, d, _ in GROUPS[grp]], interval)
        except Exception:
            pass
        cached = _cache_read(display, interval)
        if cached is not None:
            cut = cached.loc[cached.index.date >= start]
            if len(cut):
                LAST_ERRORS.pop(display, None)
                return cut
        # group prefetch ran and this symbol still has no data: report,
        # never fall through to a per-symbol call (credit stampede)
        LAST_ERRORS.setdefault(display, "FX-API no data after group fetch")
        return None

    key = _api_key()
    if not key:
        LAST_ERRORS[display] = "FX-API key missing (add TWELVEDATA_KEY to secrets)"
        return None
    if _breaker_open():
        LAST_ERRORS[display] = "FX-API cooling down (limit reached)"
        return None

    data_sym = _LOOKUP.get(display, (display, None))[0]
    if not _spend_request(f"SINGLE {data_sym}"):
        LAST_ERRORS[display] = ("FX-API request budget reached — batch should "
                                "have covered this symbol")
        return None
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
