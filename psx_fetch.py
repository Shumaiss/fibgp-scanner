"""
psx_fetch.py — Daily OHLCV fetcher for Pakistan Stock Exchange.

Uses the same public endpoint as the psxdata / psx-data-reader package
(POST https://dps.psx.com.pk/historical, one request per symbol-month) but
without per-symbol progress bars, with thread-local sessions, retries, and
a callback hook so the Streamlit app can drive a single scan-level progress
bar. Data is end-of-day official PSX history (effectively delayed, which
suits a daily-timeframe scanner).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

HIST_URL = "https://dps.psx.com.pk/historical"
SYMBOLS_URL = "https://dps.psx.com.pk/symbols"

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (FibGP-Scanner)"})
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


def _download_month(symbol: str, d: date, timeout: float = 15.0,
                    retries: int = 2) -> pd.DataFrame | None:
    payload = {"month": d.month, "year": d.year, "symbol": symbol}
    for attempt in range(retries + 1):
        try:
            r = _session().post(HIST_URL, data=payload, timeout=timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            headers = [th.get_text(strip=True) for th in soup.select("th")]
            if not headers:
                return None
            rows = []
            for tr in soup.select("tr"):
                cols = [td.get_text(strip=True) for td in tr.select("td")]
                if len(cols) == len(headers):
                    rows.append(cols)
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=headers)
            return df
        except requests.RequestException:
            if attempt == retries:
                return None
    return None


def fetch_daily(symbol: str, start: date, end: date | None = None) -> pd.DataFrame | None:
    """Fetch daily OHLCV for one symbol. Returns a DataFrame indexed by Date
    with float columns Open, High, Low, Close, Volume — or None on failure."""
    end = end or date.today()
    months = _month_starts(start, end)

    frames = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(_download_month, symbol, m) for m in months]
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    if "TIME" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["TIME"], format="%b %d, %Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.rename(columns=str.title)

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if len(keep) < 4:
        return None
    df = df[keep]
    for col in keep:
        df[col] = (df[col].astype(str).str.replace(",", "", regex=False)
                   .replace({"": np.nan, "-": np.nan}).astype(float))
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    return df if len(df) else None


def fetch_tickers() -> pd.DataFrame | None:
    """Full PSX symbol directory from the official symbols endpoint."""
    try:
        r = _session().get(SYMBOLS_URL, timeout=15.0)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except (requests.RequestException, ValueError):
        return None
