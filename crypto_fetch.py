"""
crypto_fetch.py — Daily OHLCV fetcher for crypto (spot + USDT perpetuals). (v1.0)

Primary and fallback market-data providers, no API keys required. Universe
listings are pulled live from the exchange (all active USDT pairs, leveraged
tokens and stable-vs-stable pairs excluded) and cached for the day. Candle
data includes the current forming daily candle (matches chart platforms) and
is disk-cached with a 15-minute TTL since crypto trades around the clock.

Provider names never surface in the UI — failure reasons per symbol land in
LAST_ERRORS (server-side diagnostics only), tagged C1/C2.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

# ------------------------------- shared ---------------------------------
_local = threading.local()
LAST_ERRORS: dict[str, str] = {}

CACHE_DIR = "/tmp/fibgp_cache"
CANDLE_TTL = 15 * 60          # seconds — forming candle keeps moving
LISTING_TTL = 24 * 3600       # universe listings refresh daily

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")

# C1 = primary provider, C2 = fallback provider
C1_SPOT_HOSTS = ("https://data-api.binance.vision", "https://api.binance.com")
C1_PERP_HOSTS = ("https://fapi.binance.com",)
C2_HOSTS = ("https://api.bybit.com", "https://api.bytick.com")

# bases that make a "coin" pair uninteresting for zone scanning
EXCLUDE_BASES = {"USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "GBP",
                 "AEUR", "TRY", "BRL", "ARS", "COP", "UAH", "PLN", "RON",
                 "ZAR", "MXN", "CZK", "JPY", "XUSD", "USD1", "USDE", "PAXG"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "2L", "2S",
                      "4L", "4S", "5L", "5S")

# static emergency universe if all listing endpoints fail
FALLBACK_MAJORS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT",
    "TONUSDT", "SHIBUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "ICPUSDT", "XLMUSDT", "ETCUSDT", "FILUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SUIUSDT", "ATOMUSDT", "HBARUSDT", "VETUSDT",
    "RNDRUSDT", "GRTUSDT", "AAVEUSDT", "ALGOUSDT", "FTMUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "THETAUSDT", "EGLDUSDT", "EOSUSDT", "XTZUSDT",
    "FLOWUSDT", "CHZUSDT", "CRVUSDT", "MKRUSDT", "LDOUSDT", "SNXUSDT",
    "COMPUSDT", "ENJUSDT", "1INCHUSDT", "ZILUSDT", "KAVAUSDT", "RUNEUSDT",
    "DYDXUSDT", "GMXUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT", "SEIUSDT",
]


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": BROWSER_UA})
        _local.session = s
    return _local.session


def _get_json(url: str, params: dict, timeout: float = 20.0):
    """GET returning (json, None) or (None, error_string)."""
    try:
        r = _session().get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return r.json(), None
    except requests.RequestException as e:
        detail = str(e.args[0]) if e.args else str(e)
        return None, f"{type(e).__name__}: {detail[:90]}"
    except ValueError:
        return None, "bad JSON"


# ============================== DISK CACHE (TTL) ===============================
def _cache_path(kind: str, key: str) -> str:
    return os.path.join(CACHE_DIR, f"crypto_{kind}_{key}")


def _cache_fresh(path: str, ttl: int) -> bool:
    try:
        return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl
    except OSError:
        return False


def _candles_read(market: str, symbol: str) -> pd.DataFrame | None:
    p = _cache_path("ohlc", f"{market}_{symbol}.csv")
    if not _cache_fresh(p, CANDLE_TTL):
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df if {"Open", "High", "Low", "Close"}.issubset(df.columns) and len(df) else None
    except Exception:
        return None


def _candles_write(market: str, symbol: str, df: pd.DataFrame):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(_cache_path("ohlc", f"{market}_{symbol}.csv"))
    except Exception:
        pass


# ============================== CANDLES ========================================
def _frame_from_rows(recs: list[dict], start: date) -> pd.DataFrame | None:
    if not recs:
        return None
    df = pd.DataFrame(recs).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[df.index.date >= start]
    return df if len(df) else None


def _c1_klines(symbol: str, market: str, start: date,
               interval: str = "1d") -> tuple[pd.DataFrame | None, str | None]:
    """Primary provider klines. Spot and perps use different hosts/paths.
    Response rows: [openTime, open, high, low, close, volume, ...] oldest-first,
    the last row being the forming candle."""
    if market == "spot":
        hosts, path = C1_SPOT_HOSTS, "/api/v3/klines"
    else:
        hosts, path = C1_PERP_HOSTS, "/fapi/v1/klines"
    err = None
    for host in hosts:
        data, err = _get_json(host + path,
                              {"symbol": symbol, "interval": interval, "limit": 1000})
        if data is None:
            continue
        if not isinstance(data, list):
            err = f"unexpected payload: {str(data)[:80]}"
            continue
        recs = []
        for row in data:
            try:
                d = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
                recs.append({"Date": pd.Timestamp(d.date()),
                             "Open": float(row[1]), "High": float(row[2]),
                             "Low": float(row[3]), "Close": float(row[4]),
                             "Volume": float(row[5])})
            except (ValueError, IndexError, TypeError):
                continue
        df = _frame_from_rows(recs, start)
        return (df, None) if df is not None else (None, "no rows in window")
    return None, f"C1 {err}"


def _c2_klines(symbol: str, market: str, start: date,
               interval: str = "1d") -> tuple[pd.DataFrame | None, str | None]:
    """Fallback provider klines. category spot|linear; rows newest-first:
    [startTime(ms str), open, high, low, close, volume, turnover]."""
    category = "spot" if market == "spot" else "linear"
    err = None
    for host in C2_HOSTS:
        data, err = _get_json(host + "/v5/market/kline",
                              {"category": category, "symbol": symbol,
                               "interval": ("W" if interval == "1w" else "D"), "limit": 1000})
        if data is None:
            continue
        rows = (data.get("result") or {}).get("list") or []
        if not rows:
            err = f"retCode {data.get('retCode')}: {str(data.get('retMsg'))[:60]}"
            continue
        recs = []
        for row in rows:
            try:
                d = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
                recs.append({"Date": pd.Timestamp(d.date()),
                             "Open": float(row[1]), "High": float(row[2]),
                             "Low": float(row[3]), "Close": float(row[4]),
                             "Volume": float(row[5])})
            except (ValueError, IndexError, TypeError):
                continue
        df = _frame_from_rows(recs, start)
        return (df, None) if df is not None else (None, "no rows in window")
    return None, f"C2 {err}"


def fetch_daily_crypto(symbol: str, start: date, market: str = "spot",
                       interval: str = "1d") -> pd.DataFrame | None:
    """15-min disk cache → primary → fallback. interval '1d' or '1w' —
    weekly candles are native exchange candles (Monday-start, UTC), incl.
    the forming candle."""
    ckey = f"{market}_{interval}"
    cached = _candles_read(ckey, symbol)
    if cached is not None:
        cut = cached.loc[cached.index.date >= start]
        if len(cut):
            LAST_ERRORS.pop(symbol, None)
            return cut
    df, e1 = _c1_klines(symbol, market, start, interval)
    if df is not None:
        _candles_write(ckey, symbol, df)
        LAST_ERRORS.pop(symbol, None)
        return df
    df, e2 = _c2_klines(symbol, market, start, interval)
    if df is not None:
        _candles_write(ckey, symbol, df)
        LAST_ERRORS.pop(symbol, None)
        return df
    LAST_ERRORS[symbol] = f"{e1} | {e2}"
    return None


# ============================== UNIVERSE LISTINGS ==============================
def _clean_bases(pairs: list[tuple[str, str]]) -> list[str]:
    """pairs = [(symbol, baseAsset)] → filtered sorted symbol list."""
    out = []
    for sym, base in pairs:
        if base in EXCLUDE_BASES:
            continue
        if any(base.endswith(sfx) for sfx in LEVERAGED_SUFFIXES) and len(base) > 3:
            continue
        out.append(sym)
    return sorted(set(out))


def list_symbols(market: str = "spot") -> tuple[list[str], str]:
    """All active USDT pairs for the market ('spot' | 'perp').
    Returns (symbols, source_note). Daily disk cache; static fallback list
    if every listing endpoint fails."""
    cache_p = _cache_path("universe", f"{market}.json")
    if _cache_fresh(cache_p, LISTING_TTL):
        try:
            syms = json.load(open(cache_p))
            if isinstance(syms, list) and len(syms) > 20:
                return syms, "cached"
        except Exception:
            pass

    pairs: list[tuple[str, str]] = []
    # --- primary provider ---
    if market == "spot":
        for host in C1_SPOT_HOSTS:
            data, _ = _get_json(host + "/api/v3/exchangeInfo", {})
            if data and "symbols" in data:
                pairs = [(s["symbol"], s.get("baseAsset", ""))
                         for s in data["symbols"]
                         if s.get("quoteAsset") == "USDT"
                         and s.get("status") == "TRADING"]
                break
    else:
        for host in C1_PERP_HOSTS:
            data, _ = _get_json(host + "/fapi/v1/exchangeInfo", {})
            if data and "symbols" in data:
                pairs = [(s["symbol"], s.get("baseAsset", ""))
                         for s in data["symbols"]
                         if s.get("quoteAsset") == "USDT"
                         and s.get("status") == "TRADING"
                         and s.get("contractType") == "PERPETUAL"]
                break
    # --- fallback provider ---
    if not pairs:
        category = "spot" if market == "spot" else "linear"
        for host in C2_HOSTS:
            data, _ = _get_json(host + "/v5/market/instruments-info",
                                {"category": category, "limit": 1000})
            lst = ((data or {}).get("result") or {}).get("list") or []
            if lst:
                pairs = [(s["symbol"], s.get("baseCoin", ""))
                         for s in lst
                         if s.get("quoteCoin") == "USDT"
                         and s.get("status") == "Trading"]
                break

    if pairs:
        syms = _clean_bases(pairs)
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            json.dump(syms, open(cache_p, "w"))
        except Exception:
            pass
        return syms, "live"
    return list(FALLBACK_MAJORS), "fallback list (listing endpoints unreachable)"
