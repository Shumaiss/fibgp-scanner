"""
psx_fetch.py — Daily OHLCV fetcher for Pakistan Stock Exchange. (v2.1)

v2.0: SCSTrade (scstrade.com) is now the PRIMARY source — one JSON request
per symbol for the full history, and it doesn't block cloud hosts the way
dps.psx.com.pk does. The PSX monthly-page fetcher is kept as a FALLBACK.
Same proven route as the PSX Whale Screener v2.0 fetcher.

Returns identical DataFrames either way: Date-indexed Open/High/Low/Close/
Volume floats. Failure reasons per symbol land in LAST_ERRORS.

v2.1: tries both scstrade.com and www.scstrade.com (endpoint 404s during
their maintenance windows), and adds a same-day disk cache — once a scan
succeeds, that day's data survives source outages and app reboots.
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
    payload = {"par": symbol,
               "date1": start.strftime("%m/%d/%Y"),
               "date2": end.strftime("%m/%d/%Y")}
    err = None
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
            return _finalize(df, start, end, symbol)
        except requests.RequestException as e:
            detail = str(e.args[0]) if e.args else str(e)
            err = f"SCS {type(e).__name__}: {detail[:100]}"
        except ValueError as e:            # JSON decode / float cast
            err = f"SCS parse error: {str(e)[:100]}"
    LAST_ERRORS[symbol] = err or "SCS unknown failure"
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
    return _finalize(df, start, end, symbol)


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
