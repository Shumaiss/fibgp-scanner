"""
crypto_fetch.py — Daily OHLCV fetcher for crypto (spot + USDT perpetuals). (v2.0)

Primary and fallback market-data providers, no API keys required. Universe
listings are pulled live from the exchange (all active USDT pairs, leveraged
tokens and stable-vs-stable pairs excluded) and cached for the day. Candle
data includes the current forming daily candle (matches chart platforms) and
is disk-cached with a 15-minute TTL since crypto trades around the clock.

Provider names never surface in the UI — failure reasons per symbol land in
LAST_ERRORS (server-side diagnostics only), tagged C1/C2/C3.

v2.0: third provider (C3) added for breadth — its spot catalog is one of
the largest anywhere and its v3 API mirrors C1's format. Universe = union
of C1 + C3 listings; each symbol fetches from the exchange that lists it.
C3's futures API is also wired in as a chance to serve the Perps universe
from hosting where C1/C2 futures are geo-blocked.
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
C3_SPOT_HOST = "https://api.mexc.com"
C3_PERP_HOST = "https://contract.mexc.com"

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


# per-market listing membership: {"spot": {"c1": set, "c3": set}, "perp": {...}}
_listed: dict[str, dict[str, set]] = {"spot": {"c1": set(), "c3": set()},
                                      "perp": {"c1": set(), "c3": set()}}


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


def _binance_rows_to_frame(data, start: date) -> pd.DataFrame | None:
    """Shared parser for C1/C3 spot klines (identical row format)."""
    if not isinstance(data, list):
        return None
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
    return _frame_from_rows(recs, start)


def _c3_spot_klines(symbol: str, start: date,
                    interval: str = "1d") -> tuple[pd.DataFrame | None, str | None]:
    """C3 spot klines — v3 API, C1-compatible rows. Weekly is '1W' there."""
    iv = "1W" if interval == "1w" else interval
    data, err = _get_json(C3_SPOT_HOST + "/api/v3/klines",
                          {"symbol": symbol, "interval": iv, "limit": 1000})
    if data is None:
        return None, f"C3 {err}"
    df = _binance_rows_to_frame(data, start)
    if df is None:
        return None, f"C3 unexpected payload: {str(data)[:70]}"
    return df, None


def _c3_perp_klines(symbol: str, start: date,
                    interval: str = "1d") -> tuple[pd.DataFrame | None, str | None]:
    """C3 futures klines — contract API: BTC_USDT symbols, columnar payload."""
    c_sym = symbol[:-4] + "_USDT" if symbol.endswith("USDT") else symbol
    iv = "Week1" if interval == "1w" else "Day1"
    data, err = _get_json(f"{C3_PERP_HOST}/api/v1/contract/kline/{c_sym}",
                          {"interval": iv})
    if data is None:
        return None, f"C3 {err}"
    d = data.get("data") if isinstance(data, dict) else None
    if not isinstance(d, dict) or not d.get("time"):
        return None, f"C3 contract payload: {str(data)[:70]}"
    try:
        recs = []
        for t, o, h, l, c, v in zip(d["time"], d["open"], d["high"],
                                    d["low"], d["close"],
                                    d.get("vol", [0] * len(d["time"]))):
            ts = datetime.fromtimestamp(int(t), tz=timezone.utc)
            recs.append({"Date": pd.Timestamp(ts.date()),
                         "Open": float(o), "High": float(h),
                         "Low": float(l), "Close": float(c),
                         "Volume": float(v)})
    except (ValueError, TypeError, KeyError) as e:
        return None, f"C3 contract parse: {str(e)[:60]}"
    df = _frame_from_rows(recs, start)
    return (df, None) if df is not None else (None, "C3 no rows in window")


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

    def try_c1():
        return _c1_klines(symbol, market, start, interval)

    def try_c3():
        if market == "spot":
            return _c3_spot_klines(symbol, start, interval)
        return _c3_perp_klines(symbol, start, interval)

    def try_c2():
        return _c2_klines(symbol, market, start, interval)

    # route to the exchange that lists the symbol first; fall through the rest
    on_c1 = symbol in _listed[market]["c1"]
    on_c3 = symbol in _listed[market]["c3"]
    if on_c3 and not on_c1:
        order = (try_c3, try_c1, try_c2)
    else:
        order = (try_c1, try_c3, try_c2)

    errs = []
    for attempt in order:
        df, err = attempt()
        if df is not None:
            _candles_write(ckey, symbol, df)
            LAST_ERRORS.pop(symbol, None)
            return df
        errs.append(err or "?")
    LAST_ERRORS[symbol] = " | ".join(errs)
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
            payload = json.load(open(cache_p))
            if isinstance(payload, dict) and len(payload.get("all", [])) > 20:
                _listed[market]["c1"] = set(payload.get("c1", []))
                _listed[market]["c3"] = set(payload.get("c3", []))
                return payload["all"], "cached"
        except Exception:
            pass

    c1_pairs: list[tuple[str, str]] = []
    c3_pairs: list[tuple[str, str]] = []
    # --- provider C1 ---
    if market == "spot":
        for host in C1_SPOT_HOSTS:
            data, _ = _get_json(host + "/api/v3/exchangeInfo", {})
            if data and "symbols" in data:
                c1_pairs = [(s["symbol"], s.get("baseAsset", ""))
                            for s in data["symbols"]
                            if s.get("quoteAsset") == "USDT"
                            and s.get("status") == "TRADING"]
                break
    else:
        for host in C1_PERP_HOSTS:
            data, _ = _get_json(host + "/fapi/v1/exchangeInfo", {})
            if data and "symbols" in data:
                c1_pairs = [(s["symbol"], s.get("baseAsset", ""))
                            for s in data["symbols"]
                            if s.get("quoteAsset") == "USDT"
                            and s.get("status") == "TRADING"
                            and s.get("contractType") == "PERPETUAL"]
                break
    # --- provider C3 (breadth) ---
    if market == "spot":
        data, _ = _get_json(C3_SPOT_HOST + "/api/v3/exchangeInfo", {})
        if data and "symbols" in data:
            c3_pairs = [(s["symbol"], s.get("baseAsset", ""))
                        for s in data["symbols"]
                        if s.get("quoteAsset") == "USDT"
                        and (s.get("status") in ("TRADING", "1", "ENABLED")
                             or s.get("isSpotTradingAllowed"))]
    else:
        data, _ = _get_json(C3_PERP_HOST + "/api/v1/contract/detail", {})
        lst = (data or {}).get("data") if isinstance(data, dict) else None
        if isinstance(lst, list):
            for s in lst:
                raw = str(s.get("symbol", ""))
                if s.get("quoteCoin") == "USDT" and raw.endswith("_USDT") \
                        and s.get("state", 0) in (0, "0"):
                    c3_pairs.append((raw.replace("_", ""), raw.split("_")[0]))
    # --- provider C2 (only if both above came up empty) ---
    if not c1_pairs and not c3_pairs:
        category = "spot" if market == "spot" else "linear"
        for host in C2_HOSTS:
            data, _ = _get_json(host + "/v5/market/instruments-info",
                                {"category": category, "limit": 1000})
            lst = ((data or {}).get("result") or {}).get("list") or []
            if lst:
                c1_pairs = [(s["symbol"], s.get("baseCoin", ""))
                            for s in lst
                            if s.get("quoteCoin") == "USDT"
                            and s.get("status") == "Trading"]
                break

    if c1_pairs or c3_pairs:
        _listed[market]["c1"] = set(_clean_bases(c1_pairs))
        _listed[market]["c3"] = set(_clean_bases(c3_pairs))
        syms = sorted(_listed[market]["c1"] | _listed[market]["c3"])
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            json.dump({"all": syms,
                       "c1": sorted(_listed[market]["c1"]),
                       "c3": sorted(_listed[market]["c3"])},
                      open(cache_p, "w"))
        except Exception:
            pass
        return syms, "live"
    return list(FALLBACK_MAJORS), "fallback list (listing endpoints unreachable)"
