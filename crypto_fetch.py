"""
crypto_fetch.py — Crypto OHLCV fetcher, single-exchange edition. (v3.0)

One exchange only. PERPETUAL futures are the primary source (contract API);
the same exchange's spot API is the automatic fallback so a missing or
unreachable contract never blanks the scanner. Universe = all active USDT
perpetual contracts (spot listing as fallback), leveraged tokens and
stable-vs-stable pairs excluded.

Candles include the current forming bar (matches chart platforms); disk
cache TTL 15 min (crypto trades around the clock), listings cached daily.
Provider names never surface in the UI — failure reasons land in
LAST_ERRORS (server diagnostics only), tagged C-FUT / C-SPOT.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

# ------------------------------- config ---------------------------------
PERP_HOST = "https://contract.mexc.com"
SPOT_HOST = "https://api.mexc.com"

_local = threading.local()
LAST_ERRORS: dict[str, str] = {}

CACHE_DIR = "/tmp/fibgp_cache"
CANDLE_TTL = 15 * 60
LISTING_TTL = 24 * 3600

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")

EXCLUDE_BASES = {"USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "GBP",
                 "AEUR", "TRY", "BRL", "ARS", "COP", "UAH", "PLN", "RON",
                 "ZAR", "MXN", "CZK", "JPY", "XUSD", "USD1", "USDE", "PAXG"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "2L", "2S",
                      "4L", "4S", "5L", "5S")

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


def _candles_read(kind: str, symbol: str) -> pd.DataFrame | None:
    p = _cache_path("ohlc", f"{kind}_{symbol}.csv")
    if not _cache_fresh(p, CANDLE_TTL):
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df if {"Open", "High", "Low", "Close"}.issubset(df.columns) and len(df) else None
    except Exception:
        return None


def _candles_write(kind: str, symbol: str, df: pd.DataFrame):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(_cache_path("ohlc", f"{kind}_{symbol}.csv"))
    except Exception:
        pass


# ============================== PARSERS ========================================
def _frame_from_rows(recs: list[dict], start: date) -> pd.DataFrame | None:
    if not recs:
        return None
    df = pd.DataFrame(recs).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[df.index.date >= start]
    return df if len(df) else None


def _perp_klines(symbol: str, start: date,
                 interval: str = "1d") -> tuple[pd.DataFrame | None, str | None]:
    """Futures candles — contract API: BTC_USDT symbols, columnar payload.
    Explicit start/end so we always get the full window (default count is small)."""
    c_sym = symbol[:-4] + "_USDT" if symbol.endswith("USDT") else symbol
    iv = "Week1" if interval == "1w" else "Day1"
    t_start = int(datetime(start.year, start.month, start.day,
                           tzinfo=timezone.utc).timestamp())
    t_end = int(time.time())
    data, err = _get_json(f"{PERP_HOST}/api/v1/contract/kline/{c_sym}",
                          {"interval": iv, "start": t_start, "end": t_end})
    if data is None:
        return None, f"C-FUT {err}"
    d = data.get("data") if isinstance(data, dict) else None
    if not isinstance(d, dict) or not d.get("time"):
        return None, f"C-FUT payload: {str(data)[:70]}"
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
        return None, f"C-FUT parse: {str(e)[:60]}"
    df = _frame_from_rows(recs, start)
    return (df, None) if df is not None else (None, "C-FUT no rows in window")


def _spot_klines(symbol: str, start: date,
                 interval: str = "1d") -> tuple[pd.DataFrame | None, str | None]:
    """Spot candles fallback — v3 API rows [openTime, o, h, l, c, vol, ...]."""
    iv = "1W" if interval == "1w" else interval
    data, err = _get_json(SPOT_HOST + "/api/v3/klines",
                          {"symbol": symbol, "interval": iv, "limit": 1000})
    if data is None:
        return None, f"C-SPOT {err}"
    if not isinstance(data, list):
        return None, f"C-SPOT payload: {str(data)[:70]}"
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
    return (df, None) if df is not None else (None, "C-SPOT no rows in window")


# ============================== PUBLIC: CANDLES ================================
def fetch_daily_crypto(symbol: str, start: date, market: str = "perp",
                       interval: str = "1d") -> pd.DataFrame | None:
    """15-min disk cache → perpetual futures → spot fallback."""
    ckey = f"{market}_{interval}"
    cached = _candles_read(ckey, symbol)
    if cached is not None:
        cut = cached.loc[cached.index.date >= start]
        if len(cut):
            LAST_ERRORS.pop(symbol, None)
            return cut
    df, e1 = _perp_klines(symbol, start, interval)
    if df is not None:
        _candles_write(ckey, symbol, df)
        LAST_ERRORS.pop(symbol, None)
        return df
    df, e2 = _spot_klines(symbol, start, interval)
    if df is not None:
        _candles_write(ckey, symbol, df)
        LAST_ERRORS.pop(symbol, None)
        return df
    LAST_ERRORS[symbol] = f"{e1} | {e2}"
    return None


# ============================== PUBLIC: UNIVERSE ===============================
def _clean_bases(pairs: list[tuple[str, str]]) -> list[str]:
    out = []
    for sym, base in pairs:
        if base in EXCLUDE_BASES:
            continue
        if any(base.endswith(sfx) for sfx in LEVERAGED_SUFFIXES) and len(base) > 3:
            continue
        out.append(sym)
    return sorted(set(out))


def list_symbols(market: str = "perp") -> tuple[list[str], str]:
    """All active USDT perpetual contracts; spot listing as fallback;
    static majors if the exchange is unreachable. Daily disk cache."""
    cache_p = _cache_path("universe", f"{market}.json")
    if _cache_fresh(cache_p, LISTING_TTL):
        try:
            syms = json.load(open(cache_p))
            if isinstance(syms, list) and len(syms) > 20:
                return syms, "cached"
        except Exception:
            pass

    pairs: list[tuple[str, str]] = []
    src = "live"
    data, _ = _get_json(PERP_HOST + "/api/v1/contract/detail", {})
    lst = (data or {}).get("data") if isinstance(data, dict) else None
    if isinstance(lst, list):
        for s in lst:
            raw = str(s.get("symbol", ""))
            if s.get("quoteCoin") == "USDT" and raw.endswith("_USDT") \
                    and s.get("state", 0) in (0, "0"):
                pairs.append((raw.replace("_", ""), raw.split("_")[0]))

    if not pairs:                       # spot listing fallback (same exchange)
        data, _ = _get_json(SPOT_HOST + "/api/v3/exchangeInfo", {})
        if data and "symbols" in data:
            pairs = [(s["symbol"], s.get("baseAsset", ""))
                     for s in data["symbols"]
                     if s.get("quoteAsset") == "USDT"
                     and (s.get("status") in ("TRADING", "1", "ENABLED")
                          or s.get("isSpotTradingAllowed"))]
            src = "live (spot listing)"

    if pairs:
        syms = _clean_bases(pairs)
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            json.dump(syms, open(cache_p, "w"))
        except Exception:
            pass
        return syms, src
    return list(FALLBACK_MAJORS), "fallback list (listing endpoints unreachable)"
