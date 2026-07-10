"""
psx_fetch.py — Daily OHLCV fetcher for Pakistan Stock Exchange. (v1.3)

v1.1: browser-grade headers + homepage warm-up (cookie collection), plus
per-symbol error capture (LAST_ERRORS) for diagnostics.
v1.2: PSX renamed the history table's date column TIME -> DATE; accept both
with flexible date parsing.
v1.3: polite mode — months fetched sequentially per symbol with an
inter-request delay and exponential retry backoff, so scans don't trip
PSX's rate limiting / temporary IP bans. Detailed connection errors.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://dps.psx.com.pk"
HIST_URL = f"{BASE}/historical"
SYMBOLS_URL = f"{BASE}/symbols"

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": BASE,
    "Referer": f"{BASE}/historical",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
}

_local = threading.local()

# symbol -> human-readable reason for the most recent failure
LAST_ERRORS: dict[str, str] = {}

# gap between consecutive requests (seconds) — keeps us under rate limits
REQUEST_DELAY = 0.25


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update(BROWSER_HEADERS)
        # Warm-up: visit the site once like a browser would, collecting any
        # cookies the protection layer sets before we start POSTing.
        try:
            s.get(BASE, timeout=15.0)
        except requests.RequestException:
            pass
        _local.session = s
    return _local.session


def _month_starts(start: date, end: date) -> list[date]:
    cur = date(start.year, start.month, 1)
    out = [cur]
    while True:
        cur = cur + relativedelta(months=1)
        if cur > end:
            break
        out.append(cur)
    return out


def _download_month(symbol: str, d: date, timeout: float = 20.0,
                    retries: int = 3):
    """Returns (DataFrame|None, error_string|None)."""
    payload = {"month": d.month, "year": d.year, "symbol": symbol}
    err = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(1.5 * (2 ** (attempt - 1)))   # 1.5s, 3s, 6s backoff
        try:
            r = _session().post(HIST_URL, data=payload, timeout=timeout)
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            headers = [th.get_text(strip=True) for th in soup.select("th")]
            if not headers:
                # 200 but no table — protection page or genuinely no data
                low = r.text[:400].lower()
                if "cloudflare" in low or "captcha" in low or "just a moment" in low:
                    err = "blocked by site protection"
                else:
                    err = "empty month"
                return None, err
            rows = []
            for tr in soup.select("tr"):
                cols = [td.get_text(strip=True) for td in tr.select("td")]
                if len(cols) == len(headers):
                    rows.append(cols)
            if not rows:
                return None, "empty month"
            return pd.DataFrame(rows, columns=headers), None
        except requests.RequestException as e:
            detail = str(e.args[0]) if e.args else str(e)
            err = f"{type(e).__name__}: {detail[:120]}"
    return None, err


def fetch_daily(symbol: str, start: date, end: date | None = None) -> pd.DataFrame | None:
    """Fetch daily OHLCV for one symbol. Returns a DataFrame indexed by Date
    with float columns Open, High, Low, Close, Volume — or None on failure
    (reason recorded in LAST_ERRORS[symbol])."""
    end = end or date.today()
    months = _month_starts(start, end)

    frames, errs = [], []
    for m in months:
        df, err = _download_month(symbol, m)
        if df is not None and not df.empty:
            frames.append(df)
        elif err and err != "empty month":
            errs.append(err)
            if len(errs) >= 3 and not frames:
                break               # host clearly unreachable — stop early
        time.sleep(REQUEST_DELAY)   # politeness gap between requests

    if not frames:
        LAST_ERRORS[symbol] = errs[0] if errs else "no rows returned"
        return None

    df = pd.concat(frames, ignore_index=True)
    date_col = next((c for c in df.columns
                     if c.strip().upper() in ("TIME", "DATE")), None)
    if date_col is None:
        LAST_ERRORS[symbol] = f"unexpected columns: {list(df.columns)[:6]}"
        return None
    # PSX has used "Jul 10, 2026" historically; parse that first, then fall
    # back to generic parsing if the site changes format again.
    parsed = pd.to_datetime(df[date_col], format="%b %d, %Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    df["Date"] = parsed
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.rename(columns=str.title)

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if len(keep) < 4:
        LAST_ERRORS[symbol] = f"missing OHLC columns: have {keep}"
        return None
    df = df[keep]
    for col in keep:
        df[col] = (df[col].astype(str).str.replace(",", "", regex=False)
                   .replace({"": np.nan, "-": np.nan}).astype(float))
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    if not len(df):
        LAST_ERRORS[symbol] = "no bars in requested window"
        return None
    LAST_ERRORS.pop(symbol, None)
    return df


def fetch_tickers() -> pd.DataFrame | None:
    try:
        r = _session().get(SYMBOLS_URL, timeout=15.0)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except (requests.RequestException, ValueError):
        return None
