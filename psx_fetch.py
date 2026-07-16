"""
psx_fetch.py — Daily OHLCV fetcher for Pakistan Stock Exchange. (v3.0)

v2.0: SCSTrade (scstrade.com) is now the PRIMARY source — one JSON request
per symbol for the full history, and it doesn't block cloud hosts the way
dps.psx.com.pk does. The PSX monthly-page fetcher is kept as a FALLBACK.
Same proven route as the PSX Whale Screener v2.0 fetcher.

Returns identical DataFrames either way: Date-indexed Open/High/Low/Close/
Volume floats. Failure reasons per symbol land in LAST_ERRORS.

v2.1: dual primary hosts + same-day disk cache.
v2.2: circuit breakers + polite pacing.
v3.0: GitHub data repository becomes the PRIMARY source — a daily updater
running on the operator's PC publishes official EOD data for every symbol
to a public data repo; the app reads that one file (fast, never blocked,
never rate-limited). Live providers remain as fallbacks for symbols the
repo doesn't cover yet. Large scans previously hammered the
primary source with thousands of rapid requests when it started failing,
which can earn the host IP a temporary ban. Now: a small delay before every
primary request, and after several consecutive full-symbol failures the
source is marked down for a cooling-off period — remaining symbols fail
fast instead of deepening the ban.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import date

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ------------------------------- shared ---------------------------------
_local = threading.local()
LAST_ERRORS: dict[str, str] = {}
REQUEST_DELAY = 0.25          # politeness gap for the PSX fallback path
P1_DELAY = 0.30               # primary-provider pacing (per symbol request)

# ---- circuit breakers: stop hammering a source that is refusing us ----
_BREAK_AFTER = 5              # consecutive full-symbol failures to trip
_COOLDOWN = 180.0             # seconds a tripped source stays closed
_state = {"p1_fails": 0, "p1_down_until": 0.0,
          "p2_fails": 0, "p2_down_until": 0.0}
_state_lock = threading.Lock()


def _breaker_open(key: str) -> bool:
    with _state_lock:
        return time.time() < _state[f"{key}_down_until"]


def _breaker_report(key: str, ok: bool):
    with _state_lock:
        if ok:
            _state[f"{key}_fails"] = 0
        else:
            _state[f"{key}_fails"] += 1
            if _state[f"{key}_fails"] >= _BREAK_AFTER:
                _state[f"{key}_down_until"] = time.time() + _COOLDOWN
                _state[f"{key}_fails"] = 0

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": BROWSER_UA,
                          "Accept-Language": "en-US,en;q=0.9"})
        _local.session = s
    return _local.session


def _finalize(df: pd.DataFrame, start: date, end: date,
              symbol: str) -> pd.DataFrame | None:
    """Common cleanup: sort, dedupe, window, type-check."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    if not len(df):
        LAST_ERRORS[symbol] = "no bars in requested window"
        return None
    LAST_ERRORS.pop(symbol, None)
    return df


# ============================ PRIMARY: SCSTrade ===========================
SCS_HOSTS = ("https://scstrade.com", "https://www.scstrade.com")
SCS_PATH = "/stockscreening/SS_HistoricalCharts.aspx/chart"
SCS_REF_PATH = "/stockscreening/SS_HistoricalCharts.aspx"

_MS_DATE = re.compile(r"/Date\((\-?\d+)")


def _scs_parse_date(v) -> pd.Timestamp | None:
    """SCSTrade dates arrive as '/Date(1610000000000)/' (ms epoch) or plain
    strings depending on server version — handle both."""
    if isinstance(v, str):
        m = _MS_DATE.search(v)
        if m:
            return pd.Timestamp(int(m.group(1)), unit="ms")
        t = pd.to_datetime(v, errors="coerce")
        return None if pd.isna(t) else t
    if isinstance(v, (int, float)):
        return pd.Timestamp(int(v), unit="ms")
    return None


def _pick_key(keys, *needles) -> str | None:
    """Find a dict key containing all needles (case-insensitive)."""
    for k in keys:
        kl = k.lower()
        if all(n in kl for n in needles):
            return k
    return None


def fetch_scstrade(symbol: str, start: date, end: date,
                   timeout: float = 30.0, retries: int = 2) -> pd.DataFrame | None:
    if _breaker_open("p1"):
        LAST_ERRORS[symbol] = "P1 cooling down (breaker open)"
        return None
    payload = {"par": symbol,
               "date1": start.strftime("%m/%d/%Y"),
               "date2": end.strftime("%m/%d/%Y")}
    err = None
    time.sleep(P1_DELAY)
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(1.5 * attempt)
        host = SCS_HOSTS[attempt % len(SCS_HOSTS)]
        headers = {"Content-Type": "application/json; charset=UTF-8",
                   "Referer": host + SCS_REF_PATH,
                   "X-Requested-With": "XMLHttpRequest",
                   "Origin": host}
        try:
            r = _session().post(host + SCS_PATH, json=payload, headers=headers,
                                timeout=timeout)
            if r.status_code == 404:
                err = "SCS HTTP 404 (endpoint down — likely maintenance)"
                continue
            if r.status_code != 200:
                err = f"SCS HTTP {r.status_code}"
                continue
            data = r.json()
            rows = data.get("d", data) if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                err = "SCS empty response"
                continue
            keys = list(rows[0].keys())
            k_date = _pick_key(keys, "date")
            k_open = _pick_key(keys, "open")
            k_high = _pick_key(keys, "high")
            k_low = _pick_key(keys, "low")
            k_close = _pick_key(keys, "close")
            k_vol = _pick_key(keys, "vol")
            if not all((k_date, k_open, k_high, k_low, k_close)):
                err = f"SCS unexpected keys: {keys[:8]}"
                break
            recs = []
            for row in rows:
                d = _scs_parse_date(row.get(k_date))
                if d is None:
                    continue
                recs.append({
                    "Date": d.normalize(),
                    "Open": float(row[k_open]),
                    "High": float(row[k_high]),
                    "Low": float(row[k_low]),
                    "Close": float(row[k_close]),
                    "Volume": float(row.get(k_vol) or 0),
                })
            if not recs:
                err = "SCS no parseable rows"
                continue
            df = pd.DataFrame(recs).set_index("Date")
            out = _finalize(df, start, end, symbol)
            _breaker_report("p1", out is not None)
            return out
        except requests.RequestException as e:
            detail = str(e.args[0]) if e.args else str(e)
            err = f"SCS {type(e).__name__}: {detail[:100]}"
        except ValueError as e:            # JSON decode / float cast
            err = f"SCS parse error: {str(e)[:100]}"
    LAST_ERRORS[symbol] = err or "SCS unknown failure"
    _breaker_report("p1", False)
    return None


# ====================== FALLBACK: dps.psx.com.pk ==========================
PSX_BASE = "https://dps.psx.com.pk"
PSX_HIST = f"{PSX_BASE}/historical"


def _psx_month(symbol: str, d: date, timeout: float = 20.0):
    payload = {"month": d.month, "year": d.year, "symbol": symbol}
    headers = {"Origin": PSX_BASE, "Referer": PSX_HIST,
               "X-Requested-With": "XMLHttpRequest"}
    try:
        r = _session().post(PSX_HIST, data=payload, headers=headers,
                            timeout=timeout)
        if r.status_code != 200:
            return None, f"PSX HTTP {r.status_code}"
        soup = BeautifulSoup(r.text, "html.parser")
        hdrs = [th.get_text(strip=True) for th in soup.select("th")]
        if not hdrs:
            return None, "PSX empty month"
        rows = [[td.get_text(strip=True) for td in tr.select("td")]
                for tr in soup.select("tr")]
        rows = [x for x in rows if len(x) == len(hdrs)]
        if not rows:
            return None, "PSX empty month"
        return pd.DataFrame(rows, columns=hdrs), None
    except requests.RequestException as e:
        detail = str(e.args[0]) if e.args else str(e)
        return None, f"PSX {type(e).__name__}: {detail[:100]}"


def fetch_psx(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    if _breaker_open("p2"):
        LAST_ERRORS[symbol] = "P2 cooling down (breaker open)"
        return None
    cur = date(start.year, start.month, 1)
    months = [cur]
    while True:
        cur = cur + relativedelta(months=1)
        if cur > end:
            break
        months.append(cur)

    frames, errs = [], []
    for m in months:
        df, err = _psx_month(symbol, m)
        if df is not None:
            frames.append(df)
        elif err and err != "PSX empty month":
            errs.append(err)
            if len(errs) >= 3 and not frames:
                break
        time.sleep(REQUEST_DELAY)

    if not frames:
        LAST_ERRORS[symbol] = errs[0] if errs else "PSX no rows returned"
        _breaker_report("p2", False)
        return None
    df = pd.concat(frames, ignore_index=True)
    date_col = next((c for c in df.columns
                     if c.strip().upper() in ("TIME", "DATE")), None)
    if date_col is None:
        LAST_ERRORS[symbol] = f"PSX unexpected columns: {list(df.columns)[:6]}"
        return None
    parsed = pd.to_datetime(df[date_col], format="%b %d, %Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df["Date"] = parsed
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df.rename(columns=str.title)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if len(keep) < 4:
        LAST_ERRORS[symbol] = f"PSX missing OHLC columns: have {keep}"
        return None
    df = df[keep]
    for col in keep:
        df[col] = (df[col].astype(str).str.replace(",", "", regex=False)
                   .replace({"": np.nan, "-": np.nan}).astype(float))
    out = _finalize(df, start, end, symbol)
    _breaker_report("p2", out is not None)
    return out


# ============================== DISK CACHE ================================
import os
CACHE_DIR = "/tmp/fibgp_cache"


def _cache_path(symbol: str, start: date, end: date) -> str:
    return os.path.join(CACHE_DIR,
                        f"{symbol}_{start.isoformat()}_{end.isoformat()}.csv")


def _cache_read(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        p = _cache_path(symbol, start, end)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        need = {"Open", "High", "Low", "Close"}
        return df if need.issubset(df.columns) and len(df) else None
    except Exception:
        return None


def _cache_write(symbol: str, start: date, end: date, df: pd.DataFrame):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(_cache_path(symbol, start, end))
    except Exception:
        pass


# ====================== GITHUB DATA REPO (PRIMARY) =========================
import io
import gzip
import json

DATA_REPO_RAW = "https://raw.githubusercontent.com/Shumaiss/psx-whale-data/main"
_REPO_TTL = 1800.0            # re-download at most every 30 min
_repo_state = {"data": None, "meta": None, "symbols": None, "at": 0.0}
_repo_lock = threading.Lock()


def _repo_refresh():
    """Download and parse the published data file (once per TTL)."""
    with _repo_lock:
        if time.time() - _repo_state["at"] < _REPO_TTL and _repo_state["data"] is not None:
            return
        _repo_state["at"] = time.time()
        try:
            r = _session().get(f"{DATA_REPO_RAW}/psx_data.csv.gz", timeout=60)
            if r.status_code != 200:
                return
            raw = gzip.decompress(r.content)
            df = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"])
            need = {"Symbol", "Date", "Open", "High", "Low", "Close"}
            if not need.issubset(df.columns):
                return
            _repo_state["data"] = {sym: g.set_index("Date")
                                        .drop(columns=["Symbol"]).sort_index()
                                   for sym, g in df.groupby("Symbol")}
            try:
                m = _session().get(f"{DATA_REPO_RAW}/psx_meta.json", timeout=20)
                _repo_state["meta"] = m.json() if m.status_code == 200 else None
            except (requests.RequestException, ValueError):
                _repo_state["meta"] = None
            try:
                s = _session().get(f"{DATA_REPO_RAW}/psx_symbols.json", timeout=20)
                _repo_state["symbols"] = s.json() if s.status_code == 200 else None
            except (requests.RequestException, ValueError):
                _repo_state["symbols"] = None
        except (requests.RequestException, OSError, ValueError):
            return


def fetch_repo(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    _repo_refresh()
    data = _repo_state["data"]
    if not data or symbol not in data:
        return None
    df = data[symbol]
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    if not len(df):
        return None
    LAST_ERRORS.pop(symbol, None)
    return df.copy()


def repo_symbols() -> list | None:
    """Published full symbol list, if the data repo is live."""
    _repo_refresh()
    syms = _repo_state["symbols"]
    return list(syms) if isinstance(syms, list) and len(syms) > 20 else None


def repo_meta() -> dict | None:
    _repo_refresh()
    return _repo_state["meta"]


# ============================ PUBLIC ENTRYPOINT ============================
def fetch_daily(symbol: str, start: date, end: date | None = None) -> pd.DataFrame | None:
    """Same-day disk cache first, then SCSTrade, then dps.psx.com.pk.
    On total failure, LAST_ERRORS[symbol] carries both sources' reasons.
    Cache is keyed by (symbol, window incl. today) so it naturally expires
    when the calendar day rolls over."""
    end = end or date.today()
    cached = _cache_read(symbol, start, end)
    if cached is not None:
        LAST_ERRORS.pop(symbol, None)
        return cached
    df = fetch_repo(symbol, start, end)
    if df is not None:
        return df
    df = fetch_scstrade(symbol, start, end)
    if df is not None:
        _cache_write(symbol, start, end, df)
        return df
    scs_err = LAST_ERRORS.get(symbol, "SCS failed")
    df = fetch_psx(symbol, start, end)
    if df is not None:
        _cache_write(symbol, start, end, df)
        return df
    psx_err = LAST_ERRORS.get(symbol, "PSX failed")
    LAST_ERRORS[symbol] = f"{scs_err} | {psx_err}"
    return None
