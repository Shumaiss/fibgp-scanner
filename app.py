"""
PSX Whale Screener — stocks & crypto trading in or near key levels. Daily.

Terminal-style dashboard: stat cards, status-badge table with sparklines,
summary donut, top pick card. Full-PSX universe support.

Run:  streamlit run app.py
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st

from fibgp_engine import FibGPEngine, EngineResult
from psx_fetch import fetch_daily, LAST_ERRORS, CACHE_DIR
from crypto_fetch import (fetch_daily_crypto, list_symbols,
                          LAST_ERRORS as CRYPTO_ERRORS)
from symbols import ALL_PSX, KSE100, QUICK25

# ============================== PALETTE ========================================
BG     = "#0A1216"
PANEL  = "#0E1B21"
PANEL2 = "#122430"
LINE   = "#1B3340"
MINT   = "#00E5B0"
MINTD  = "#0E4D42"
RED    = "#FF5A66"
AMBER  = "#FFC24D"
TXT    = "#D8E4E4"
MUTE   = "#6E8891"

PAGE_SIZE = 15

st.set_page_config(page_title="PSX Whale Screener", page_icon="🐋",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Rajdhani:wght@500;600;700&display=swap');
.stApp {{ background: {BG}; }}
html, body, [class*="css"] {{ font-family:'JetBrains Mono',monospace; color:{TXT}; }}
section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
h1,h2,h3 {{ font-family:'Rajdhani',sans-serif; }}

.hd-title {{ font-family:'Rajdhani',sans-serif; font-size:1.9rem; font-weight:700;
  letter-spacing:.10em; line-height:1.05; }}
.hd-title .m {{ color:{MINT}; }}
.hd-sub {{ color:{MUTE}; font-size:.74rem; letter-spacing:.06em; }}

.statrow {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }}
.stat {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px; padding:10px 14px; }}
.stat .k {{ font-size:.66rem; letter-spacing:.14em; color:{MUTE}; }}
.stat .v {{ font-family:'Rajdhani',sans-serif; font-size:1.7rem; font-weight:700; line-height:1.1; }}
.stat .p {{ font-size:.68rem; color:{MUTE}; }}

.panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px; padding:14px 16px; }}
.panel-hd {{ font-family:'Rajdhani',sans-serif; font-weight:700; letter-spacing:.14em;
  font-size:.95rem; margin-bottom:8px; }}

table.scan {{ width:100%; border-collapse:collapse; font-size:.76rem; }}
table.scan th {{ color:{MUTE}; font-weight:500; font-size:.64rem; letter-spacing:.10em;
  text-align:left; padding:6px 8px; border-bottom:1px solid {LINE}; white-space:nowrap; }}
table.scan td {{ padding:8px 8px; border-bottom:1px solid {LINE}; white-space:nowrap; vertical-align:middle; }}
table.scan tr:last-child td {{ border-bottom:none; }}
.tick {{ color:{MINT}; font-weight:700; }}
.badge {{ display:inline-block; padding:2px 9px; border-radius:5px; font-size:.62rem;
  letter-spacing:.08em; font-weight:700; }}
.b-insup  {{ background:rgba(0,229,176,.15); color:{MINT}; border:1px solid rgba(0,229,176,.5); }}
.b-nearsup{{ background:transparent; color:{MINT}; border:1px solid rgba(0,229,176,.35); }}
.b-inres  {{ background:rgba(255,90,102,.15); color:{RED}; border:1px solid rgba(255,90,102,.5); }}
.b-nearres{{ background:transparent; color:{RED}; border:1px solid rgba(255,90,102,.35); }}
.b-watch  {{ background:transparent; color:{MUTE}; border:1px solid {LINE}; }}
.b-nozone {{ background:transparent; color:{MUTE}; border:1px dashed {LINE}; }}
.stars {{ color:{AMBER}; letter-spacing:.05em; }}
.mut {{ color:{MUTE}; }}
.dgreen {{ color:{MINT}; }} .dred {{ color:{RED}; }} .damber {{ color:{AMBER}; }}

.legend {{ font-size:.72rem; line-height:1.9; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:7px; }}

.toppick .sym {{ font-family:'Rajdhani',sans-serif; font-size:1.7rem; font-weight:700; color:{MINT}; }}
.toppick .px {{ font-family:'Rajdhani',sans-serif; font-size:1.5rem; font-weight:700; }}

.stButton>button {{ border:1px solid {LINE}; background:{PANEL2}; color:{TXT}; }}
.stButton>button:hover {{ border-color:{MINT}; color:{MINT}; }}
div[data-testid="stTextInput"] input {{ background:{PANEL2}; border:1px solid {LINE}; color:{TXT}; }}
</style>
""", unsafe_allow_html=True)


# ============================== ROW MODEL ======================================
@dataclass
class Row:
    symbol: str
    close: float
    status: str            # IN_SUPPORT / NEAR_SUPPORT / IN_RESISTANCE / NEAR_RESISTANCE / WATCHING / NO_ZONE
    side: str              # sup / res / ""
    zone_bot: float
    zone_top: float
    dist: float            # % away (0 inside, inf when NO_ZONE)
    stars: int
    stars_str: str
    conf: str
    rsi: float
    stoch_sig: str
    stoch_dots: str
    zone_age: int
    entered_today: bool


BADGE = {
    "IN_SUPPORT":      ("IN SUPPORT", "b-insup"),
    "NEAR_SUPPORT":    ("NEAR SUPPORT", "b-nearsup"),
    "IN_RESISTANCE":   ("IN RESIST", "b-inres"),
    "NEAR_RESISTANCE": ("NEAR RESIST", "b-nearres"),
    "WATCHING":        ("WATCHING", "b-watch"),
    "NO_ZONE":         ("NO ZONE", "b-nozone"),
}
STATUS_RANK = {"IN_SUPPORT": 0, "IN_RESISTANCE": 0, "NEAR_SUPPORT": 1,
               "NEAR_RESISTANCE": 1, "WATCHING": 2, "NO_ZONE": 3}


def build_row(sym: str, r: EngineResult, thr: float) -> Row:
    """Categorize a symbol against its active zones for the dashboard table."""
    c = r.close
    sup, res = r.support, r.resistance

    cand = []   # (dist, side, zone)
    if sup is not None:
        if sup.bot <= c <= sup.top:
            cand.append((0.0, "sup", sup))
        elif c > sup.top:
            cand.append(((c - sup.top) / sup.top * 100.0, "sup", sup))
    if res is not None:
        if res.bot <= c <= res.top:
            cand.append((0.0, "res", res))
        elif c < res.bot:
            cand.append(((res.bot - c) / c * 100.0, "res", res))

    if not cand:
        return Row(sym, c, "NO_ZONE", "", math.nan, math.nan, math.inf, 0, "",
                   "", r.rsi, r.stoch.signal, r.stoch.dots, 0, False)

    dist, side, z = min(cand, key=lambda x: x[0])
    if dist == 0.0:
        status = "IN_SUPPORT" if side == "sup" else "IN_RESISTANCE"
    elif dist <= thr:
        status = "NEAR_SUPPORT" if side == "sup" else "NEAR_RESISTANCE"
    else:
        status = "WATCHING"
    entered = r.entered_sup_today if side == "sup" else r.entered_res_today
    return Row(sym, c, status, side, z.bot, z.top, dist, z.star_count, z.stars,
               r.conf_tags(side), r.rsi, r.stoch.signal, r.stoch.dots,
               r.n_bars - 1 - z.anchor_idx, entered)


# ============================== SVG HELPERS ====================================
def sparkline(closes: np.ndarray, w: int = 92, h: int = 26) -> str:
    if closes is None or len(closes) < 2:
        return ""
    v = closes[-30:]
    lo, hi = float(np.min(v)), float(np.max(v))
    rng = (hi - lo) or 1.0
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 2 - (x - lo) / rng * (h - 4):.1f}"
                   for i, x in enumerate(v))
    col = MINT if v[-1] >= v[0] else RED
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<polyline points='{pts}' fill='none' stroke='{col}' "
            f"stroke-width='1.5' stroke-linejoin='round'/></svg>")


def donut(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    R, CX, CY, SW = 40, 55, 55, 15
    circ = 2 * math.pi * R
    colors = {"In zone": MINT, "Near zone": RED, "Watching": AMBER, "No zone / n.a.": MUTE}
    segs, off = "", 0.0
    for k, v in counts.items():
        frac = v / total
        segs += (f"<circle r='{R}' cx='{CX}' cy='{CY}' fill='transparent' "
                 f"stroke='{colors[k]}' stroke-width='{SW}' "
                 f"stroke-dasharray='{frac * circ:.2f} {circ:.2f}' "
                 f"stroke-dashoffset='{-off:.2f}' "
                 f"transform='rotate(-90 {CX} {CY})'/>")
        off += frac * circ
    return (f"<svg width='110' height='110' viewBox='0 0 110 110'>{segs}"
            f"<text x='{CX}' y='{CY - 2}' text-anchor='middle' fill='{TXT}' "
            f"font-size='17' font-weight='700' font-family='Rajdhani'>{total}</text>"
            f"<text x='{CX}' y='{CY + 13}' text-anchor='middle' fill='{MUTE}' "
            f"font-size='8'>SCANNED</text></svg>")


def fmt_px(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    a = abs(v)
    if a >= 100:
        return f"{v:,.2f}"
    if a >= 1:
        s = f"{v:,.3f}"
    elif a >= 0.01:
        s = f"{v:.5f}"
    else:
        s = f"{v:.8f}"
    return s.rstrip("0").rstrip(".")


# ============================== SIDEBAR ========================================
with st.sidebar:
    st.markdown(f"<div class='hd-title'>🐋 PSX <span class='m'>WHALE</span></div>"
                f"<div class='hd-sub'>SCREENER · KEY LEVELS · DAILY</div>",
                unsafe_allow_html=True)
    st.write("")
    universe_choice = st.radio("Universe",
                               ["All PSX", "KSE-100", "Quick 25",
                                "Crypto Spot", "Crypto Perps", "Custom"])
    market = "psx"
    symbols: list | None = None
    if universe_choice == "Custom":
        market_lbl = st.selectbox("Market", ["PSX", "Crypto Spot", "Crypto Perps"])
        market = {"PSX": "psx", "Crypto Spot": "spot", "Crypto Perps": "perp"}[market_lbl]
        default_syms = "HBL, UBL, OGDC" if market == "psx" else "BTCUSDT, ETHUSDT, SOLUSDT"
        custom_txt = st.text_area("Symbols", value=default_syms, height=80)
        symbols = sorted({s.strip().upper() for s in
                          custom_txt.replace("\n", ",").split(",") if s.strip()})
    elif universe_choice == "All PSX":
        symbols = ALL_PSX
    elif universe_choice == "KSE-100":
        symbols = KSE100
    elif universe_choice == "Quick 25":
        symbols = QUICK25
    elif universe_choice == "Crypto Spot":
        market = "spot"       # full pair list resolved at scan time
    else:
        market = "perp"

    timeframe = st.radio("Timeframe", ["Daily", "Weekly"], horizontal=True)
    tf = "1w" if timeframe == "Weekly" else "1d"
    near_pct = st.slider("Near threshold (%)", 0.5, 5.0, 2.0, 0.25)
    _min_m = 15 if tf == "1w" else 8
    lookback_m = st.slider("History (months)", _min_m, 24, max(18, _min_m))

    with st.expander("Engine parameters"):
        piv_l = st.number_input("Pivot left bars", 1, 20, 5)
        piv_r = st.number_input("Pivot right bars", 1, 20, 5)
        use_live = st.checkbox("Use live (unconfirmed) endpoint", True)
        use_ema = st.checkbox("EMA confluence", True)
        use_fvg = st.checkbox("FVG confluence", True)
        fvg_lb = st.number_input("FVG lookback (bars)", 10, 200, 50)

    run_scan = st.button("SCAN", type="primary", use_container_width=True)

    # cache status panel
    try:
        n_cached = len([f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")])
    except OSError:
        n_cached = 0
    _sc = st.session_state.get("scan") or (None, None, None, "—")
    stamp = _sc[3]
    st.markdown(f"""<div class='panel' style='margin-top:8px'>
      <div style='font-size:.66rem;letter-spacing:.12em;color:{MINT}'>● CACHE STATUS</div>
      <div style='font-size:.72rem;margin-top:4px'>{n_cached} symbols cached today<br>
      <span class='mut'>Last scan: {stamp}</span></div></div>""",
        unsafe_allow_html=True)
    n_lbl = f"{len(symbols)} symbols" if symbols is not None else "all USDT pairs"
    tf_lbl = timeframe
    ttl_lbl = "EOD · cached" if market == "psx" else "24/7 · 15m cache"
    st.caption(f"{n_lbl} · {tf_lbl} · {ttl_lbl}")


# ============================== SCAN ===========================================
if "ohlc_cache" not in st.session_state:
    st.session_state.ohlc_cache = {}
if "scan" not in st.session_state:
    st.session_state.scan = None
if "page" not in st.session_state:
    st.session_state.page = 1


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Daily -> weekly OHLC. Monday-keyed calendar weeks (chart convention);
    the current forming week is included."""
    wk = df.index - pd.to_timedelta(df.index.weekday, unit="D")
    g = df.groupby(wk)
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(),
                        "Low": g["Low"].min(), "Close": g["Close"].last(),
                        "Volume": g["Volume"].sum()})
    return out.sort_index()


def run_full_scan(syms, start, engine, thr, market, tf):
    rows, results, failed = [], {}, []
    prog = st.progress(0.0, text="Fetching market data…")
    key = (market, tf, start.isoformat())

    def _fetch(s):
        if market == "psx":
            df = fetch_daily(s, start)
            return to_weekly(df) if (tf == "1w" and df is not None) else df
        # crypto: native weekly candles; pull deeper history for weekly depth
        c_start = start if tf == "1d" else date.today() - relativedelta(months=60)
        return fetch_daily_crypto(s, c_start, market, tf)

    to_fetch = [s for s in syms if (s, key) not in st.session_state.ohlc_cache]
    done = 0
    if to_fetch:
        workers = 4 if market == "psx" else 8
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch, s): s for s in to_fetch}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    st.session_state.ohlc_cache[(s, key)] = fut.result()
                except Exception:
                    st.session_state.ohlc_cache[(s, key)] = None
                done += 1
                prog.progress(0.85 * done / len(to_fetch),
                              text=f"Fetching… {s} ({done}/{len(to_fetch)})")

    for i, s in enumerate(syms):
        df = st.session_state.ohlc_cache.get((s, key))
        prog.progress(0.85 + 0.15 * (i + 1) / len(syms), text=f"Engine… {s}")
        if df is None or len(df) < 60:
            failed.append(s)
            continue
        try:
            res = engine.run(df["Open"].to_numpy(), df["High"].to_numpy(),
                             df["Low"].to_numpy(), df["Close"].to_numpy())
            results[s] = res
            rows.append(build_row(s, res, thr))
        except Exception:
            failed.append(s)
    prog.empty()
    return rows, results, failed


if run_scan:
    start = date.today() - relativedelta(months=int(lookback_m))
    if symbols is None:                      # full crypto universe
        with st.spinner("Resolving pair list…"):
            symbols, src_note = list_symbols(market)
        if "fallback" in src_note:
            st.warning(f"Using built-in majors list ({len(symbols)} pairs) — "
                       f"full listing temporarily unavailable.")
    engine = FibGPEngine(piv_left=int(piv_l), piv_right=int(piv_r),
                         use_live=use_live, fvg_lookback=int(fvg_lb),
                         use_ema_conf=use_ema, use_fvg_conf=use_fvg)
    rows, results, failed = run_full_scan(symbols, start, engine, near_pct,
                                          market, tf)
    st.session_state.scan = (rows, results, failed,
                             pd.Timestamp.now().strftime("%d %b %Y %H:%M"))
    st.session_state.page = 1
    st.session_state.scan_start = start.isoformat()
    st.session_state.scan_market = market
    st.session_state.scan_tf = tf


# ============================== HEADER =========================================
hd_l, hd_m, hd_r = st.columns([2.6, 1.6, 1.2])
with hd_l:
    st.markdown(f"<div class='hd-title'>PSX WHALE <span class='m'>SCREENER</span></div>"
                f"<div class='hd-sub'>Find stocks & crypto trading in or near key levels · Daily / Weekly</div>",
                unsafe_allow_html=True)
with hd_m:
    search = st.text_input("Search ticker…", label_visibility="collapsed",
                           placeholder="Search ticker…")
with hd_r:
    if st.session_state.scan:
        st.markdown(f"<div style='text-align:right;font-size:.72rem;padding-top:8px'>"
                    f"<span class='mut'>Last updated:</span> "
                    f"<span class='dgreen'>{st.session_state.scan[3]}</span> ●</div>",
                    unsafe_allow_html=True)
st.write("")

if st.session_state.scan is None:
    st.markdown(f"<div class='panel'><span class='mut'>Pick a universe and press "
                f"<b>SCAN</b>. Every symbol is categorized: in zone, near zone "
                f"(≤ threshold), watching (zone active but farther), or no active "
                f"zone.</span></div>", unsafe_allow_html=True)
    st.stop()

rows, results, failed, stamp = st.session_state.scan
rows_sorted = sorted(rows, key=lambda r: (STATUS_RANK[r.status], r.dist, -r.stars))

# ============================== STAT CARDS =====================================
n_in = sum(1 for r in rows if r.status.startswith("IN_"))
n_near = sum(1 for r in rows if r.status.startswith("NEAR_"))
n_watch = sum(1 for r in rows if r.status == "WATCHING")
n_none = sum(1 for r in rows if r.status == "NO_ZONE") + len(failed)
tot = len(rows) + len(failed)


def pct(n): return f"({100 * n / tot:.1f}%)" if tot else ""


st.markdown(f"""<div class='statrow'>
 <div class='stat'><div class='k'>UNIVERSE</div><div class='v'>{tot}</div><div class='p'>tickers</div></div>
 <div class='stat'><div class='k' style='color:{MINT}'>IN ZONE</div><div class='v' style='color:{MINT}'>{n_in}</div><div class='p'>{pct(n_in)}</div></div>
 <div class='stat'><div class='k' style='color:{RED}'>NEAR ZONE</div><div class='v' style='color:{RED}'>{n_near}</div><div class='p'>≤ {near_pct:g}% {pct(n_near)}</div></div>
 <div class='stat'><div class='k' style='color:{AMBER}'>WATCHING</div><div class='v' style='color:{AMBER}'>{n_watch}</div><div class='p'>{pct(n_watch)}</div></div>
 <div class='stat'><div class='k'>NO ZONE / N.A.</div><div class='v'>{n_none}</div><div class='p'>{pct(n_none)}</div></div>
</div>""", unsafe_allow_html=True)
st.write("")

# ============================== FILTER CHIPS ===================================
flt = st.radio("Show", ["All", "In zone", "Near zone", "Watching", "Support side", "Resistance side"],
               horizontal=True, label_visibility="collapsed")
if st.session_state.get("last_filter") != (flt, search):
    st.session_state.page = 1
    st.session_state.last_filter = (flt, search)

view = rows_sorted
if flt == "In zone":
    view = [r for r in view if r.status.startswith("IN_")]
elif flt == "Near zone":
    view = [r for r in view if r.status.startswith("NEAR_")]
elif flt == "Watching":
    view = [r for r in view if r.status == "WATCHING"]
elif flt == "Support side":
    view = [r for r in view if r.side == "sup"]
elif flt == "Resistance side":
    view = [r for r in view if r.side == "res"]
if search.strip():
    q = search.strip().upper()
    view = [r for r in view if q in r.symbol]

# ============================== MAIN GRID ======================================
left, right = st.columns([2.35, 1.0])

with left:
    npages = max(1, math.ceil(len(view) / PAGE_SIZE))
    st.session_state.page = min(st.session_state.page, npages)
    pg = st.session_state.page
    page_rows = view[(pg - 1) * PAGE_SIZE: pg * PAGE_SIZE]

    key = (st.session_state.get("scan_market", "psx"),
           st.session_state.get("scan_tf", "1d"),
           st.session_state.get("scan_start", ""))
    body = ""
    for i, r in enumerate(page_rows, start=(pg - 1) * PAGE_SIZE + 1):
        label, cls = BADGE[r.status]
        dcls = "dgreen" if r.dist <= near_pct else ("damber" if r.dist < math.inf else "mut")
        dist_txt = "—" if r.dist == math.inf else f"{r.dist:.2f}%"
        star_txt = f"<span class='stars'>{'★' * r.stars}</span><span class='mut'>{'☆' * (5 - r.stars)}</span>" if r.status != "NO_ZONE" else "<span class='mut'>—</span>"
        df = st.session_state.ohlc_cache.get((r.symbol, key))
        spark = sparkline(df["Close"].to_numpy()) if df is not None else ""
        new = " <span class='dgreen'>●</span>" if r.entered_today else ""
        body += (f"<tr><td class='mut'>{i}</td>"
                 f"<td class='tick'>{r.symbol}{new}</td>"
                 f"<td>{fmt_px(r.close)}</td>"
                 f"<td class='mut'>{fmt_px(r.zone_bot)}</td>"
                 f"<td class='mut'>{fmt_px(r.zone_top)}</td>"
                 f"<td class='{dcls}'>{dist_txt}</td>"
                 f"<td><span class='badge {cls}'>{label}</span></td>"
                 f"<td>{star_txt}</td>"
                 f"<td>{spark}</td></tr>")
    st.markdown(f"""<div class='panel'>
      <table class='scan'>
       <tr><th>#</th><th>TICKER</th><th>PRICE</th><th>ZONE BOTTOM</th><th>ZONE TOP</th>
       <th>DISTANCE %</th><th>STATUS</th><th>STARS</th><th>TREND</th></tr>
       {body if body else "<tr><td colspan='9' class='mut'>No rows match this filter.</td></tr>"}
      </table></div>""", unsafe_allow_html=True)

    pcols = st.columns([5, 1, 1.4, 1])
    with pcols[0]:
        st.markdown(f"<span class='mut' style='font-size:.72rem'>Showing "
                    f"{(pg-1)*PAGE_SIZE+1 if view else 0}–{min(pg*PAGE_SIZE, len(view))} "
                    f"of {len(view)} results (filtered from {tot})</span>",
                    unsafe_allow_html=True)
    with pcols[1]:
        if st.button("‹ Prev", disabled=pg <= 1):
            st.session_state.page -= 1; st.rerun()
    with pcols[2]:
        st.markdown(f"<div style='text-align:center;font-size:.78rem;padding-top:6px'>"
                    f"page {pg} / {npages}</div>", unsafe_allow_html=True)
    with pcols[3]:
        if st.button("Next ›", disabled=pg >= npages):
            st.session_state.page += 1; st.rerun()

    if rows:
        export = pd.DataFrame([{
            "Symbol": r.symbol, "Close": r.close, "Status": r.status, "Side": r.side,
            "ZoneBot": r.zone_bot, "ZoneTop": r.zone_top,
            "DistancePct": None if r.dist == math.inf else round(r.dist, 3),
            "Stars": r.stars, "Confluence": r.conf, "RSI7": round(r.rsi, 1),
            "Stoch": f"{r.stoch_sig} {r.stoch_dots}".strip(),
            "ZoneAgeBars": r.zone_age, "EnteredToday": r.entered_today,
        } for r in rows_sorted])
        st.download_button("Export CSV", export.to_csv(index=False),
                           file_name=f"whale_scan_{date.today().isoformat()}.csv",
                           mime="text/csv")

with right:
    # ---- market momentum ----
    buys = sum(1 for r in rows if r.stoch_sig == "BUY")
    ratio = buys / len(rows) if rows else 0.5
    mood = ("Bullish", MINT) if ratio > 0.55 else (("Bearish", RED) if ratio < 0.45 else ("Neutral", AMBER))
    st.markdown(f"""<div class='panel'>
      <div class='panel-hd mut'>MARKET MOMENTUM</div>
      <div style='font-family:Rajdhani;font-size:1.4rem;font-weight:700;color:{mood[1]}'>{mood[0]}</div>
      <div class='mut' style='font-size:.72rem'>{buys}/{len(rows)} symbols on stochastic BUY</div>
    </div>""", unsafe_allow_html=True)
    st.write("")

    # ---- summary donut ----
    counts = {"In zone": n_in, "Near zone": n_near, "Watching": n_watch,
              "No zone / n.a.": n_none}
    leg = "".join(
        f"<div><span class='dot' style='background:{c}'></span>{k} "
        f"<span class='mut'>{v} {pct(v)}</span></div>"
        for (k, v), c in zip(counts.items(), (MINT, RED, AMBER, MUTE)))
    st.markdown(f"""<div class='panel'>
      <div class='panel-hd'>SUMMARY</div>
      <div style='display:flex;gap:14px;align-items:center'>
        {donut(counts)}<div class='legend'>{leg}</div>
      </div></div>""", unsafe_allow_html=True)
    st.write("")

    # ---- top pick ----
    actionable = [r for r in rows_sorted if r.status != "NO_ZONE" and r.dist < math.inf]
    if actionable:
        tp = actionable[0]
        df = st.session_state.ohlc_cache.get(
            (tp.symbol, (st.session_state.get("scan_market", "psx"),
                         st.session_state.get("scan_tf", "1d"),
                         st.session_state.get("scan_start", ""))))
        spark = sparkline(df["Close"].to_numpy(), w=210, h=56) if df is not None else ""
        side_lbl = "support" if tp.side == "sup" else "resistance"
        conf = f" · {tp.conf}" if tp.conf else ""
        st.markdown(f"""<div class='panel toppick'>
          <div class='panel-hd'><span class='damber'>★</span> TOP PICK</div>
          <div class='sym'>{tp.symbol}</div>
          <div class='mut' style='font-size:.72rem'>{"in" if tp.dist == 0 else f"{tp.dist:.2f}% from"} {side_lbl} zone{conf}</div>
          <div class='px'>{fmt_px(tp.close)} <span class='mut' style='font-size:.8rem'>{"PKR" if st.session_state.get("scan_market","psx")=="psx" else "USDT"}</span></div>
          <div style='margin:6px 0'>{spark}</div>
          <div style='font-size:.72rem'>ZONE <span class='mut'>{fmt_px(tp.zone_bot)} – {fmt_px(tp.zone_top)}</span>
          &nbsp; STRENGTH <span class='stars'>{"★" * tp.stars}</span><span class='mut'>{"☆" * (5 - tp.stars)}</span><br>
          RSI(7) <span class='mut'>{tp.rsi:.1f}</span> &nbsp; STOCH
          <span class='{ "dgreen" if tp.stoch_sig == "BUY" else "dred"}'>{tp.stoch_sig} {tp.stoch_dots}</span></div>
        </div>""", unsafe_allow_html=True)
        st.write("")

    st.markdown(f"""<div class='panel'>
      <div class='panel-hd mut'>MARKET INSIGHT</div>
      <div style='font-size:.74rem;line-height:1.7' class='mut'>
      Smart money flows where price meets precision.
      Track the zones. Trade with the whales.</div>
    </div>""", unsafe_allow_html=True)

# ============================== FOOTER =========================================
_tf_name = "Weekly" if st.session_state.get("scan_tf") == "1w" else "Daily"
foot = (f"Universe: {universe_choice} · Timeframe: {_tf_name} · "
        f"Data: EOD (cached) · Scanned {len(results)}/{tot}")
if failed:
    foot += f" · skipped {len(failed)}"
st.caption(foot)
if failed and not results:
    _all_err = {**LAST_ERRORS, **CRYPTO_ERRORS}
    reasons = {_all_err.get(s, "unknown") for s in failed}
    print("FETCH DIAGNOSTICS:", "; ".join(sorted(reasons)))  # server logs only
    st.error("Data sources are unreachable right now — please try again in a "
             "few minutes. (Details logged for the operator.)")
with st.expander(f"Skipped symbols ({len(failed)})" if failed else "Skipped symbols (0)"):
    if failed:
        st.markdown(f"<span class='mut' style='font-size:.72rem'>{', '.join(failed)}</span>",
                    unsafe_allow_html=True)
