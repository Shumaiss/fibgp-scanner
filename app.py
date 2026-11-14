"""
Whale Scanner — stocks & crypto trading in or near key levels.

Terminal-style dashboard: stat cards, status-badge table with sparklines,
top pick card. Full-PSX universe support.

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
from ob_engine import OrderBlockEngine
from liq_engine import LiquidityEngine
from psx_fetch import (fetch_daily, LAST_ERRORS, CACHE_DIR,
                       repo_symbols, repo_meta, STALE_NOTES)
from forex_fetch import (fetch_daily_fx, list_symbols_fx, tv_symbol_fx,
                         prefetch_group, reset_budget, REQUEST_LOG,
                         LAST_ERRORS as FX_ERRORS)
from crypto_fetch import (fetch_daily_crypto, list_symbols,
                          probe_contract_fields,
                          LAST_ERRORS as CRYPTO_ERRORS)
from symbols import ALL_PSX, KSE100, QUICK25

# ============================== PALETTE ========================================
BG     = "#04080E"
PANEL  = "#06121C"
PANEL2 = "#081824"
LINE   = "#16445C"
ROWLN  = "#0D2A3A"
ACC    = "#35E0FF"          # the one accent: glow, numbers, buttons
ON_ACC = "#021016"          # text on an accent fill
# Status colours keep their meaning:
GREEN   = "#2EE6A6"         # support side · bullish trend
GREEN_L = "#8CF2CF"         # near support
RED_C   = "#FF5A6E"         # resistance side · bearish trend
RED_L   = "#FFA3AE"         # near resistance
TXT    = "#DDF6FF"
MUTE   = "#7FA9BD"
DIM    = "#4E7C92"
F_HEAD = "'Orbitron',sans-serif"
F_BODY = "'Rajdhani',sans-serif"
F_MONO = "'Share Tech Mono',monospace"

PAGE_SIZE = 25

st.set_page_config(page_title="Whale Scanner", page_icon="🐋",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&display=swap');
.stApp {{ background-color:{BG};
  background-image:linear-gradient(rgba(53,224,255,.05) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(53,224,255,.05) 1px, transparent 1px);
  background-size:40px 40px; }}
html, body, [class*="css"], .stApp p, .stApp label {{ font-family:{F_BODY}; color:{TXT};
  font-variant-numeric:tabular-nums; }}
header[data-testid="stHeader"] {{ background:transparent; }}
section[data-testid="stSidebar"] {{ background:rgba(6,18,28,.92);
  border-right:1px solid {LINE}; }}
section[data-testid="stSidebar"] label p {{ font-size:1rem; letter-spacing:.06em; }}
div[data-testid="stWidgetLabel"] p {{ font-family:{F_MONO}; font-size:.68rem !important;
  letter-spacing:.24em; text-transform:uppercase; color:{MUTE}; }}
h1,h2,h3 {{ font-family:{F_HEAD}; }}
::-webkit-scrollbar {{ width:10px; height:10px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{LINE}; border-radius:2px; }}

.hd-title {{ font-family:{F_HEAD}; font-size:1.75rem; font-weight:700;
  letter-spacing:.2em; line-height:1.1; color:{TXT};
  text-shadow:0 0 18px rgba(53,224,255,.55); }}
.hd-rule {{ height:1px; margin:10px 0 6px 0;
  background:linear-gradient(90deg, {ACC} 0, {ACC} 120px, {LINE} 120px, {LINE} 100%); }}
.hd-sub {{ color:{MUTE}; font-family:{F_MONO}; font-size:.68rem; letter-spacing:.22em; }}
.sb-logo {{ display:flex; align-items:center; gap:12px; }}
.sb-mark {{ display:inline-flex; align-items:center; justify-content:center; flex:0 0 52px;
  width:52px; height:52px; border:1px solid {ACC}; border-radius:50%;
  box-shadow:0 0 14px rgba(53,224,255,.45); }}
.sb-name {{ font-family:{F_HEAD}; font-size:1.05rem; font-weight:700; letter-spacing:.2em;
  line-height:1.2; color:{TXT}; }}

.hud {{ background-color:rgba(8,24,36,.72); border:1px solid {LINE};
  background-image:linear-gradient({ACC},{ACC}),linear-gradient({ACC},{ACC}),linear-gradient({ACC},{ACC}),linear-gradient({ACC},{ACC}),linear-gradient({ACC},{ACC}),linear-gradient({ACC},{ACC}),linear-gradient({ACC},{ACC}),linear-gradient({ACC},{ACC});
  background-repeat:no-repeat;
  background-size:14px 1px,1px 14px,14px 1px,1px 14px,14px 1px,1px 14px,14px 1px,1px 14px;
  background-position:0 0,0 0,100% 0,100% 0,0 100%,0 100%,100% 100%,100% 100%; }}

.statrow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }}
.stat {{ padding:14px 18px 12px; }}
.stat .k {{ font-family:{F_MONO}; font-size:.62rem; letter-spacing:.24em; color:{MUTE}; }}
.stat .v {{ font-family:{F_HEAD}; font-size:2.2rem; font-weight:700; line-height:1.2; color:{TXT}; }}
.stat .p {{ font-family:{F_MONO}; font-size:.68rem; color:{MUTE}; }}
.stat.hero {{ box-shadow:0 0 24px rgba(53,224,255,.22), inset 0 0 30px rgba(53,224,255,.08); }}
.stat.hero .k {{ color:{ACC}; }}
.stat.hero .v {{ color:{ACC}; text-shadow:0 0 16px rgba(53,224,255,.7); }}

.panel {{ padding:14px 16px; }}
.panel-hd {{ font-family:{F_HEAD}; font-weight:700; letter-spacing:.18em;
  font-size:.9rem; margin-bottom:10px; }}

.whale {{ padding:0; }}
.whale .main {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px 28px; padding:6px 24px; }}
.whale .art {{ flex:0 1 480px; max-width:100%; line-height:0; }}
.whale .art svg {{ width:100%; height:auto; }}
.whale .copy {{ flex:1 1 300px; display:flex; flex-direction:column; gap:8px; padding:10px 0; }}
.whale .eyebrow {{ font-family:{F_MONO}; font-size:.68rem; letter-spacing:.3em; color:{ACC}; }}
.whale .big {{ font-family:{F_HEAD}; font-size:1.35rem; font-weight:700; letter-spacing:.08em;
  line-height:1.25; color:{TXT}; }}
.whale .small {{ font-size:.95rem; line-height:1.4; color:{MUTE}; max-width:520px; }}
.whale .chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
.whale .chip {{ padding:5px 12px; border:1px solid {LINE}; font-family:{F_MONO};
  font-size:.68rem; letter-spacing:.16em; color:{TXT}; }}
.whale .chip.on {{ border-color:{ACC}; color:{ACC}; box-shadow:0 0 10px rgba(53,224,255,.35); }}
.tapebox {{ overflow:hidden; border-top:1px solid {LINE}; padding:8px 0;
  font-family:{F_MONO}; font-size:.74rem; letter-spacing:.08em; white-space:nowrap; }}
.tape {{ display:inline-flex; animation:tapeMove var(--dur,60s) linear infinite; }}
.tape:hover {{ animation-play-state:paused; }}
.tape .it {{ margin-right:34px; }}
.tape .n {{ color:{TXT}; }} .tape .p {{ color:{MUTE}; }}
@keyframes tapeMove {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{ .tape {{ animation:none; }} }}

.tablebox {{ overflow-x:auto; }}
table.scan {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
table.scan th {{ color:{MUTE}; font-family:{F_MONO}; font-weight:400; font-size:.62rem;
  letter-spacing:.2em; text-align:left; padding:9px 11px; border-bottom:1px solid {LINE};
  white-space:nowrap; }}
table.scan th.num, table.scan td.num {{ text-align:right; font-family:{F_MONO}; }}
table.scan td {{ padding:10px 11px; border-bottom:1px solid {ROWLN};
  white-space:nowrap; vertical-align:middle; }}
table.scan tr:hover td {{ background:rgba(53,224,255,.06); }}
table.scan tr:last-child td {{ border-bottom:none; }}
.tick {{ font-family:{F_HEAD}; color:{TXT}; font-weight:700; font-size:.82rem;
  letter-spacing:.08em; text-decoration:none; }}
a.tick:hover {{ color:{ACC}; text-shadow:0 0 10px rgba(53,224,255,.6); }}
.goto {{ opacity:0; font-size:.7em; margin-left:3px; transition:opacity .12s; }}
a:hover .goto {{ opacity:.85; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:2px; font-size:.68rem;
  letter-spacing:.12em; font-weight:700; }}
.b-insup  {{ background:{GREEN}; color:#04140E; border:1px solid {GREEN}; }}
.b-nearsup{{ background:transparent; color:{GREEN_L}; border:1px solid {GREEN_L}; }}
.b-inres  {{ background:{RED_C}; color:#19060A; border:1px solid {RED_C}; }}
.b-nearres{{ background:transparent; color:{RED_L}; border:1px solid {RED_L}; }}
.b-watch  {{ background:transparent; color:{MUTE}; border:1px solid {LINE}; }}
.b-nozone {{ background:transparent; color:{MUTE}; border:1px dashed {LINE}; }}
.sbars {{ display:inline-flex; align-items:center; gap:3px; }}
.sbar {{ display:inline-block; width:14px; height:4px; background:{LINE}; }}
.sbar.on {{ background:{ACC}; box-shadow:0 0 6px rgba(53,224,255,.8); }}
.st-bull {{ color:{GREEN}; font-weight:700; font-size:.7rem; letter-spacing:.12em; }}
.st-bear {{ color:{RED_C}; font-weight:700; font-size:.7rem; letter-spacing:.12em; }}
.st-mix  {{ color:{MUTE}; font-weight:600; font-size:.7rem; letter-spacing:.12em; }}
.pill {{ display:inline-block; padding:1px 7px; border-radius:2px; font-size:.66rem;
  margin-left:5px; vertical-align:1px; background:{PANEL2}; color:{TXT};
  border:1px solid {LINE}; }}
.mut {{ color:{MUTE}; }}
.dgreen {{ color:{GREEN}; }} .dred {{ color:{RED_C}; }} .damber {{ color:{MUTE}; }}
.dacc {{ color:{ACC}; }}

.stButton>button, .stDownloadButton>button {{ border:1px solid {LINE}; background:transparent;
  color:{TXT}; border-radius:2px; font-family:{F_BODY}; letter-spacing:.1em; transition:all .12s; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:{ACC}; color:{ON_ACC};
  border-color:{ACC}; box-shadow:0 0 14px rgba(53,224,255,.5); }}
button[kind="primary"] {{ background:{ACC} !important; color:{ON_ACC} !important;
  border-color:{ACC} !important; font-family:{F_HEAD} !important; font-weight:700;
  letter-spacing:.3em !important; box-shadow:0 0 14px rgba(53,224,255,.5); }}
button[kind="primary"] p {{ font-family:{F_HEAD} !important; color:{ON_ACC} !important; }}
div[data-testid="stTextInput"] input {{ background:{PANEL2}; border:1px solid {LINE};
  color:{TXT}; border-radius:2px; font-family:{F_MONO}; }}
div[data-testid="stExpander"] details {{ border:1px solid {LINE}; border-radius:2px;
  background:rgba(8,24,36,.5); }}
</style>
""", unsafe_allow_html=True)


# ============================== WHALE ART ======================================
FLUKE = ("M296 156 C302 132 309 112 313 98 C290 94 262 84 234 62 "
         "C262 72 296 80 320 82 C344 80 378 72 406 62 C378 84 350 94 327 98 "
         "C331 112 338 132 344 156 Z")


def whale_mark() -> str:
    """Whale tail for the sidebar logo."""
    return (f"<svg width='34' height='26' viewBox='226 54 188 108' aria-hidden='true'>"
            f"<path d='{FLUKE}' fill='none' stroke='{ACC}' stroke-width='9' stroke-linejoin='round'/></svg>")


def _fluke_art() -> str:
    """A whale's tail breaking the surface of a price line, between zones."""
    import numpy as _np
    rng = _np.random.default_rng(11)
    xs = _np.linspace(0, 640, 70)
    ys = 150 + _np.cumsum(rng.normal(0, 2.2, 70))
    ys = ys - ys.mean() + 150
    ys[30:40] = _np.linspace(ys[30], 150, 10)
    price = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    mono = "Share Tech Mono, monospace"
    return (
        f"<svg viewBox='0 30 640 200' role='img' "
        f"aria-label='A whale tail rising out of a price line, between a resistance band and a support band'>"
        f"<rect x='0' y='162' width='640' height='16' fill='{GREEN}' opacity='0.10'/>"
        f"<line x1='0' y1='162' x2='640' y2='162' stroke='{GREEN}' stroke-width='1' opacity='0.6'/>"
        f"<text x='8' y='176' font-family='{mono}' font-size='11' fill='{GREEN}'>SUPPORT ZONE</text>"
        f"<rect x='0' y='52' width='640' height='12' fill='{RED_C}' opacity='0.08'/>"
        f"<line x1='0' y1='64' x2='640' y2='64' stroke='{RED_C}' stroke-width='1' opacity='0.5'/>"
        f"<text x='8' y='61' font-family='{mono}' font-size='11' fill='{RED_C}'>RESISTANCE</text>"
        f"<polyline points='{price}' fill='none' stroke='{ACC}' stroke-width='1.6' opacity='0.9'/>"
        f"<path d='{FLUKE}' fill='{BG}' stroke='{ACC}' stroke-width='2' stroke-linejoin='round' "
        f"style='filter:drop-shadow(0 0 8px {ACC})'/>"
        f"<path d='M270 158 Q320 146 370 158 M250 166 Q320 152 390 166' fill='none' stroke='{ACC}' "
        f"stroke-width='1' opacity='0.6'/>"
        f"<g fill='{ACC}' opacity='0.7'><circle cx='284' cy='138' r='2'/><circle cx='358' cy='132' r='1.6'/>"
        f"<circle cx='268' cy='146' r='1.4'/><circle cx='372' cy='144' r='2'/></g></svg>")


def ticker_tape() -> str:
    """Price and last-bar change for symbols from the last scan. Empty before a scan,
    so the banner never shows invented numbers."""
    scan = st.session_state.get("scan")
    if not scan:
        return ""
    rows = scan[0]
    key = (st.session_state.get("scan_market", "psx"), st.session_state.get("scan_tf", "1d"),
           st.session_state.get("scan_start", ""))
    rank = {"IN_SUPPORT": 0, "IN_RESISTANCE": 0, "NEAR_SUPPORT": 1, "NEAR_RESISTANCE": 1}
    items = ""
    for r in sorted(rows, key=lambda r: rank.get(r.status, 2))[:30]:
        df = st.session_state.ohlc_cache.get((r.symbol, key))
        try:
            prev, last = float(df["Close"].iloc[-2]), float(df["Close"].iloc[-1])
            ch = (last / prev - 1.0) * 100.0
        except Exception:
            continue
        col = GREEN if ch >= 0 else RED_C
        items += (f"<span class='it'><span class='n'>{disp_symbol(r.symbol)}</span> "
                  f"<span class='p'>{fmt_px(last)}</span> "
                  f"<span style='color:{col}'>{ch:+.1f}%</span></span>")
    if not items:
        return ""
    dur = max(30, 3 * items.count("class='it'"))
    return (f"<div class='tapebox'><div class='tape' style='--dur:{dur}s'>{items}{items}</div></div>")


def whale_banner(active: str) -> str:
    chips = "".join(f"<span class='chip{' on' if s == active else ''}'>{s.upper()}</span>"
                    for s in ("Zones", "Order Blocks", "Liquidity"))
    return (f"<div class='hud whale'><div class='main'><div class='art'>{_fluke_art()}</div>"
            f"<div class='copy'>"
            f"<div class='eyebrow'>WHALE RADAR · KEY LEVELS</div>"
            f"<div class='big'>FOLLOW THE BIG MONEY</div>"
            f"<div class='small'>Find the price levels where large players are positioned, "
            f"across every market you trade.</div>"
            f"<div class='chips'>{chips}</div></div></div>{ticker_tape()}</div>")


def strength_bars(n: int) -> str:
    bars = "".join(f"<span class='sbar{' on' if i < n else ''}'></span>" for i in range(5))
    return f"<span class='sbars' role='img' aria-label='{n} of 5 strength'>{bars}</span>"


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
    structure: str = "—"     # Bullish / Bearish / Mixed


BADGE = {
    "IN_SUPPORT":      ("▲ IN SUPPORT", "b-insup"),
    "NEAR_SUPPORT":    ("▲ NEAR SUPPORT", "b-nearsup"),
    "IN_RESISTANCE":   ("▼ IN RESIST", "b-inres"),
    "NEAR_RESISTANCE": ("▼ NEAR RESIST", "b-nearres"),
    "WATCHING":        ("WATCHING", "b-watch"),
    "NO_ZONE":         ("NO ZONE", "b-nozone"),
}
BADGE_LIQ = {
    "IN_SUPPORT":      ("▲ AT SSL", "b-insup"),
    "NEAR_SUPPORT":    ("▲ NEAR SSL", "b-nearsup"),
    "IN_RESISTANCE":   ("▼ AT BSL", "b-inres"),
    "NEAR_RESISTANCE": ("▼ NEAR BSL", "b-nearres"),
    "WATCHING":        ("WATCHING", "b-watch"),
    "NO_ZONE":         ("NO LEVEL", "b-nozone"),
}
STATUS_RANK = {"IN_SUPPORT": 0, "IN_RESISTANCE": 0, "NEAR_SUPPORT": 1,
               "NEAR_RESISTANCE": 1, "WATCHING": 2, "NO_ZONE": 3}


def build_row(sym: str, r: EngineResult, thr: float) -> Row:
    """Categorize a symbol against its active zones for the dashboard table."""
    c = r.close
    sup, res = r.support, r.resistance

    if getattr(r, "is_lines", False):
        tol = getattr(r, "line_tol", 0.0)
        cand = []   # (dist, side, level, at_level)
        if sup is not None and sup.top > 0:
            cand.append((max(0.0, c - sup.top) / sup.top * 100.0, "sup", sup,
                         c - sup.top <= tol))
        if res is not None and c > 0:
            cand.append((max(0.0, res.bot - c) / c * 100.0, "res", res,
                         res.bot - c <= tol))
        if not cand:
            return Row(sym, c, "NO_ZONE", "", math.nan, math.nan, math.inf, 0,
                       "", "", r.rsi, r.stoch.signal, r.stoch.dots, 0, False,
                       r.structure)
        dist, side, z, at = min(cand, key=lambda x: x[0])
        if at:
            status = "IN_SUPPORT" if side == "sup" else "IN_RESISTANCE"
        elif dist <= thr:
            status = "NEAR_SUPPORT" if side == "sup" else "NEAR_RESISTANCE"
        else:
            status = "WATCHING"
        return Row(sym, c, status, side, z.bot, z.top, dist, z.star_count,
                   z.stars, r.conf_tags(side), r.rsi, r.stoch.signal,
                   r.stoch.dots, r.n_bars - 1 - z.anchor_idx, False,
                   r.structure)

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
                   "", r.rsi, r.stoch.signal, r.stoch.dots, 0, False,
                   r.structure)

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
               r.n_bars - 1 - z.anchor_idx, entered, r.structure)


# ============================== SVG HELPERS ====================================
def sparkline(closes: np.ndarray, w: int = 92, h: int = 26) -> str:
    if closes is None or len(closes) < 2:
        return ""
    v = closes[-30:]
    lo, hi = float(np.min(v)), float(np.max(v))
    rng = (hi - lo) or 1.0
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 2 - (x - lo) / rng * (h - 4):.1f}"
                   for i, x in enumerate(v))
    col = ACC if v[-1] >= v[0] else DIM
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<polyline points='{pts}' fill='none' stroke='{col}' "
            f"stroke-width='1.5' stroke-linejoin='round'/></svg>")


def disp_symbol(symbol: str) -> str:
    """Human-readable ticker: the exchange's API uses e.g. MUSTOCK_USDT for
    tokenized equities while displaying MU — mirror that."""
    if st.session_state.get("scan_market", "psx") == "fx":
        return symbol
    _mk = st.session_state.get("scan_market", "psx")
    if _mk.startswith("fx:"):
        return symbol
    if _mk == "stocks":
        base = symbol[:-4] if symbol.endswith("USDT") else symbol
        if base.endswith("STOCK") and len(base) > 5:
            return base[:-5]
        return base
    return symbol


def tv_link(symbol: str) -> str:
    """Chart URL for a ticker: crypto perps get the .P futures symbol,
    PSX tickers the PSX: prefix. Weekly scans open weekly charts."""
    market = st.session_state.get("scan_market", "psx")
    tf = st.session_state.get("scan_tf", "1d")
    iv = {"1w": "1W", "1mo": "1M", "4h": "240"}.get(tf, "1D")
    if market == "psx":
        tv_sym = f"PSX:{symbol}"
    elif market == "fx":
        tv_sym = tv_symbol_fx(symbol)
    else:
        # TradingView chart of the exchange's contract itself — crypto and
        # tokenized stocks/commodities alike (MEXC:TESLASTOCKUSDT.P etc.)
        tv_sym = f"MEXC:{symbol}.P"
    return (f"https://www.tradingview.com/chart/?symbol={tv_sym}"
            f"&interval={iv}")


def struct_cell(structure: str) -> str:
    """Trend badge for the table — same wording as the Trader Pro chart."""
    if structure == "Bullish":
        return "<span class='st-bull'>&#9650; BULLISH</span>"
    if structure == "Bearish":
        return "<span class='st-bear'>&#9660; BEARISH</span>"
    return "<span class='st-mix'>RANGING</span>"


def fmt_vol(v: float) -> str:
    """Compact notional volume: 1.2B / 45M / 320K."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0:
        return "—"
    for cut, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= cut:
            return f"{v / cut:.1f}{suf}"
    return f"{v:.0f}"


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
    st.markdown(f"<div class='sb-logo'><span class='sb-mark'>{whale_mark()}</span>"
                f"<div><div class='sb-name'>WHALE<br>SCANNER</div></div></div>"
                f"<div class='hd-sub' style='margin-top:10px'>KEY LEVEL INTELLIGENCE</div>",
                unsafe_allow_html=True)
    st.write("")
    strategy = st.radio("Strategy", ["Zones", "Order Blocks", "Liquidity"],
                        help="Zones: support and resistance zones. "
                             "Order Blocks: the last opposing candle before a "
                             "displacement that breaks structure, tracked until "
                             "price trades through it. "
                             "Liquidity: the nearest untaken swing high above "
                             "price (BSL) and swing low below it (SSL).")
    st.write("")
    universe_choice = st.radio("Universe",
                               ["All PSX", "KSE-100", "Quick 25",
                                "Crypto", "Stocks & Commodities",
                                "Forex", "Custom"])
    fx_group = "fx"
    if universe_choice == "Forex":
        fx_group = {"FX Pairs": "fx", "Commodities": "commodities",
                    "Indices": "indices", "Shares": "shares"}[
            st.radio("Group", ["FX Pairs", "Commodities", "Indices", "Shares"])]
    market = "psx"
    symbols: list | None = None
    if universe_choice == "Custom":
        market_lbl = st.selectbox("Market", ["PSX", "Crypto",
                                             "Stocks & Commodities", "Forex"])
        market = {"PSX": "psx", "Crypto": "perp",
                  "Stocks & Commodities": "stocks", "Forex": "fx"}[market_lbl]
        default_syms = "HBL, UBL, OGDC" if market == "psx" else "BTCUSDT, ETHUSDT, SOLUSDT"
        custom_txt = st.text_area("Symbols", value=default_syms, height=80)
        symbols = sorted({s.strip().upper() for s in
                          custom_txt.replace("\n", ",").split(",") if s.strip()})
    elif universe_choice == "All PSX":
        symbols = repo_symbols() or ALL_PSX
    elif universe_choice == "KSE-100":
        symbols = KSE100
    elif universe_choice == "Quick 25":
        symbols = QUICK25
    elif universe_choice == "Crypto":
        market = "perp"       # full contract list resolved at scan time
    elif universe_choice == "Stocks & Commodities":
        market = "stocks"     # tokenized equities, metals, oil, indices
    elif universe_choice == "Forex":
        market = "fx"         # real forex, commodities, indices, shares
        symbols, _ = list_symbols_fx(fx_group)

    timeframe = st.radio("Timeframe", ["4H", "Daily", "Weekly", "1M"],
                         index=1, horizontal=True)
    tf = {"4H": "4h", "Weekly": "1w", "1M": "1mo"}.get(timeframe, "1d")
    near_pct = st.slider("Near threshold (%)", 0.5, 5.0, 2.0, 0.25)
    min_vol_lbl = st.selectbox("Min volume (last bar)",
                               ["Off", "100K", "1M", "10M", "100M"], index=0,
                               help="Filters thin pairs by last-bar traded value.")
    MIN_VOL = {"Off": 0, "100K": 1e5, "1M": 1e6, "10M": 1e7, "100M": 1e8}[min_vol_lbl]
    if tf == "1mo":
        # one bar per month: 60 months = 60 bars, comfortably above the guard
        _min_m, _def_m, _max_m = 36, 60, 120
    elif tf == "1w":
        _min_m, _def_m, _max_m = 15, 18, 24
    elif tf == "4h":
        _min_m, _def_m, _max_m = 2, 6, 24   # ~6 months of 4H is already ~1,000 bars
    else:
        _min_m, _def_m, _max_m = 8, 18, 24
    lookback_m = st.slider("History (months)", _min_m, _max_m, _def_m)

    # Engine settings are fixed (not shown in the UI)
    piv_l, piv_r = 5, 3
    use_live = use_ema = use_fvg = True
    fvg_lb = 50
    ob_bos, ob_look, ob_max, ob_atr = 5, 8, 6, 0.25
    liq_len, liq_tol, liq_max = 2, 0.15, 8

    run_scan = st.button("SCAN", type="primary", use_container_width=True)

    # cache status panel
    try:
        n_cached = len([f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")])
    except OSError:
        n_cached = 0
    _sc = st.session_state.get("scan") or (None, None, None, "—")
    stamp = _sc[3]
    st.markdown(f"""<div class='hud panel' style='margin-top:8px'>
      <div style='font-family:{F_MONO};font-size:.66rem;letter-spacing:.2em;color:{GREEN}'>● SYSTEM ONLINE</div>
      <div style='font-size:.72rem;margin-top:4px'>{n_cached} symbols cached today<br>
      <span class='mut'>Last scan: {stamp}</span></div></div>""",
        unsafe_allow_html=True)
    n_lbl = (f"{len(symbols)} symbols" if symbols is not None
             else "all contracts" if market == "stocks"
             else "75 instruments" if market == "fx" else "all USDT pairs")
    tf_lbl = timeframe
    ttl_lbl = ("EOD · cached" if market == "psx"
               else "market hours · 15m cache" if market in ("stocks", "fx")
               else "24/7 · 15m cache")
    if tf == "4h" and market != "psx":
        ttl_lbl = "intraday · 1h cache"
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


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Daily -> monthly OHLC, keyed to the first of each month (chart
    convention); the current forming month is included."""
    mk = df.index.to_period("M").to_timestamp()
    g = df.groupby(mk)
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
            if df is None:
                return df
            if tf == "1w":
                return to_weekly(df)
            if tf == "1mo":
                return to_monthly(df)
            return df
        if market == "fx":
            return fetch_daily_fx(s, start, tf)   # provider serves weekly natively
        # crypto: native weekly candles; pull deeper history for weekly depth
        c_start = (date.today() - relativedelta(months=60)
                   if tf in ("1w", "1mo") else start)
        return fetch_daily_crypto(s, c_start, market, tf)

    if market == "fx":
        reset_budget()
    if market == "fx" and syms:
        # one batched call per 20 symbols instead of one call per symbol
        _est = max(1, -(-len(syms) // 8))     # ceil: 8 symbols per minute
        prog.progress(0.02, text=f"Fetching market data… (~{_est} min on the "
                                 f"current data plan)")
        prefetch_group(list(syms), tf)
    to_fetch = [s for s in syms if (s, key) not in st.session_state.ohlc_cache]
    done = 0
    if to_fetch:
        workers = 2 if market == "psx" else (2 if market == "fx" else 3)
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
        min_bars = (30 if tf == "1mo"                 # 30 monthly bars = 2.5 years
                    else 35 if market == "stocks"     # young tokenized listings
                    else 60)
        if df is None or len(df) < min_bars:
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


if run_scan and market == "psx" and tf == "4h":
    st.warning("4H isn't available for PSX — that data source provides "
               "end-of-day candles only. Use Daily or Weekly.")
    run_scan = False

if run_scan:
    start = date.today() - relativedelta(months=int(lookback_m))
    if symbols is None:                      # full crypto universe
        with st.spinner("Resolving pair list…"):
            symbols, src_note = list_symbols(market)
        if "fallback" in src_note:
            st.warning(f"Using built-in majors list ({len(symbols)} pairs) — "
                       f"full listing temporarily unavailable.")
    if strategy == "Liquidity":
        engine = LiquidityEngine(liq_l=int(liq_len), liq_r=int(liq_len),
                                 tol_m=float(liq_tol), liq_max=int(liq_max))
    elif strategy == "Order Blocks":
        engine = OrderBlockEngine(bos_len=int(ob_bos), ob_look=int(ob_look),
                                  ob_max=int(ob_max), min_atr_x=float(ob_atr))
    else:
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
    st.session_state.scan_strategy = strategy


# ============================== HEADER =========================================
hd_l, hd_m, hd_r = st.columns([2.6, 1.6, 1.2])
with hd_l:
    st.markdown(f"<div class='hd-title'>MARKET SCAN</div>"
                f"<div class='hd-rule'></div>"
                f"<div class='hd-sub'>STRATEGY · {strategy.upper()} &nbsp;|&nbsp; {universe_choice.upper()} "
                f"&nbsp;|&nbsp; {timeframe.upper()}</div>",
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
st.markdown(whale_banner(strategy), unsafe_allow_html=True)
st.write("")

# ---- PSX source test: open the app URL with ?probe=psx ----
if st.query_params.get("probe") == "psx":
    with st.expander("PSX SOURCE TEST", expanded=True):
        st.caption("Tries several free data sources for one PSX ticker (HBL) "
                   "from this server. Screenshot the result.")
        if st.button("Run source test"):
            import requests as _rq
            from forex_fetch import _api_key as _td_key
            _ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/126.0.0.0 Safari/537.36",
                   "Accept": "application/json,text/html,*/*"}
            _tests = [
                ("Y1", "https://query1.finance.yahoo.com/v8/finance/chart/HBL.KA",
                 {"range": "1mo", "interval": "1d"}),
                ("Y2", "https://query2.finance.yahoo.com/v8/finance/chart/HBL.KA",
                 {"range": "1mo", "interval": "1d"}),
                ("Y3", "https://query1.finance.yahoo.com/v8/finance/chart/OGDC.KA",
                 {"range": "2y", "interval": "1d"}),
                ("T1", "https://psxterminal.com/api/klines/HBL/1d", {"limit": 30}),
                ("T2", "https://psxterminal.com/api/klines/HBL/4h", {"limit": 30}),
                ("D1", "https://dps.psx.com.pk/timeseries/eod/HBL", {}),
                ("D2", "https://dps.psx.com.pk/timeseries/int/HBL", {}),
                ("S1", "https://stockanalysis.com/quote/psx/HBL/history/", {}),
                ("Q1", "https://stooq.com/q/d/l/", {"s": "hbl.pk", "i": "d"}),
            ]
            _k = _td_key()
            if _k:
                _tests.append(("X1", "https://api.twelvedata.com/time_series",
                               {"symbol": "HBL", "exchange": "XKAR", "interval": "1day",
                                "outputsize": 5, "apikey": _k}))
            for _tag, _url, _prm in _tests:
                try:
                    _r = _rq.get(_url, params=_prm, headers=_ua, timeout=15)
                    _body = _r.text[:170].replace("\n", " ").replace("`", "'")
                    st.markdown(f"`{_tag}` **HTTP {_r.status_code}** · {len(_r.text):,} chars")
                    st.code(_body or "(empty)", language=None)
                except Exception as _e:
                    st.markdown(f"`{_tag}` **FAILED** · {type(_e).__name__}")
                    st.code(str(_e)[:170], language=None)

if st.session_state.scan is None:
    st.markdown(f"<div class='hud panel'><span class='mut'>Pick a universe and press "
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


_liq_view = st.session_state.get("scan_strategy", "Zones") == "Liquidity"
_lbl_in, _lbl_near, _lbl_none = (("AT LEVEL", "NEAR LEVEL", "NO LEVEL / N.A.")
                                 if _liq_view else
                                 ("IN ZONE", "NEAR ZONE", "NO ZONE / N.A."))
st.markdown(f"""<div class='statrow'>
 <div class='hud stat'><div class='k'>UNIVERSE</div><div class='v'>{tot}</div><div class='p'>tickers</div></div>
 <div class='hud stat hero'><div class='k'>{_lbl_in}</div><div class='v'>{n_in}</div><div class='p'>{pct(n_in)}</div></div>
 <div class='hud stat'><div class='k'>{_lbl_near}</div><div class='v'>{n_near}</div><div class='p'>≤ {near_pct:g}% {pct(n_near)}</div></div>
 <div class='hud stat'><div class='k'>WATCHING</div><div class='v'>{n_watch}</div><div class='p'>{pct(n_watch)}</div></div>
 <div class='hud stat'><div class='k'>{_lbl_none}</div><div class='v'>{n_none}</div><div class='p'>{pct(n_none)}</div></div>
</div>""", unsafe_allow_html=True)
st.write("")

# ============================== FILTER CHIPS ===================================
_chips = (["All", "At level", "Near level", "Watching", "SSL side", "BSL side"]
          if _liq_view else
          ["All", "In zone", "Near zone", "Watching", "Support side", "Resistance side"])
_flt_lbl = st.radio("Show", _chips, horizontal=True, label_visibility="collapsed")
flt = ["All", "In zone", "Near zone", "Watching", "Support side",
       "Resistance side"][_chips.index(_flt_lbl)]
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
if MIN_VOL > 0:
    _key = (st.session_state.get("scan_market", "psx"),
            st.session_state.get("scan_tf", "1d"),
            st.session_state.get("scan_start", ""))
    def _notional(sym):
        df = st.session_state.ohlc_cache.get((sym, _key))
        try:
            return float(df["Volume"].iloc[-1] * df["Close"].iloc[-1])
        except Exception:
            return 0.0
    view = [r for r in view if _notional(r.symbol) >= MIN_VOL]

# ============================== MAIN GRID ======================================
left = st.container()          # boards span the full width

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
        label, cls = (BADGE_LIQ if _liq_view else BADGE)[r.status]
        dcls = "dacc" if r.dist < math.inf else "mut"
        dist_txt = "—" if r.dist == math.inf else f"{r.dist:.2f}%"
        star_txt = strength_bars(r.stars) if r.status != "NO_ZONE" else "<span class='mut'>—</span>"
        df = st.session_state.ohlc_cache.get((r.symbol, key))
        spark = sparkline(df["Close"].to_numpy()) if df is not None else ""
        notional = float(df["Volume"].iloc[-1] * df["Close"].iloc[-1]) if df is not None else float("nan")
        new = " <span class='dgreen'>●</span>" if r.entered_today else ""
        body += (f"<tr><td class='mut'>{i}</td>"
                 f"<td><a class='tick' href='{tv_link(r.symbol)}' "
                 f"target='_blank' rel='noopener'>{disp_symbol(r.symbol)}<span class='goto'>↗</span></a>{new}</td>"
                 f"<td class='num'>{fmt_px(r.close)}</td>"
                 f"<td class='num mut'>{fmt_vol(notional)}</td>"
                 f"<td class='{dcls} num'>{dist_txt}</td>"
                 f"<td><span class='badge {cls}'>{label}</span></td>"
                 f"<td>{struct_cell(r.structure)}</td>"
                 f"<td>{star_txt}</td>"
                 f"<td>{spark}</td></tr>")
    st.markdown(f"""<div class='hud panel tablebox'>
      <table class='scan'>
       <tr><th>#</th><th>TICKER</th><th class='num'>PRICE</th><th class='num'>VOL</th>
       <th class='num'>DISTANCE %</th><th>STATUS</th><th>TREND</th>
       <th>STRENGTH</th><th>CHART</th></tr>
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
            "Trend": r.structure,
        } for r in rows_sorted])
        st.download_button("Export CSV", export.to_csv(index=False),
                           file_name=f"whale_scan_{date.today().isoformat()}.csv",
                           mime="text/csv")

# ============================== FOOTER =========================================
_tf_name = {"1w": "Weekly", "4h": "4H", "1mo": "Monthly"}.get(
    st.session_state.get("scan_tf"), "Daily")
_meta = repo_meta() if st.session_state.get("scan_market", "psx") == "psx" else None
_fresh = f" · PSX data as of {_meta['last_trading_date']}" if _meta else ""
_strat = st.session_state.get("scan_strategy", "Zones")
foot = (f"Strategy: {_strat} · Universe: {universe_choice} · "
        f"Timeframe: {_tf_name} · "
        f"Data: EOD (cached){_fresh} · Scanned {len(results)}/{tot}")
if failed:
    foot += f" · skipped {len(failed)}"
st.caption(foot)
_stale = {s: d for s, d in STALE_NOTES.items() if s in results}
if _stale:
    _oldest = min(_stale.values())
    st.info(f"Live sources are temporarily unavailable — showing the last "
            f"available data (as of {_oldest}) for {len(_stale)} symbols. "
            f"Rescan later for fresh prices.")
if failed:
    _all_err = {**LAST_ERRORS, **CRYPTO_ERRORS, **FX_ERRORS}
    reasons = {_all_err.get(s, "unknown") for s in failed}
    print("FETCH DIAGNOSTICS:", "; ".join(sorted(reasons)), flush=True)  # server logs
    if not results:
        st.error("Data sources are unreachable right now — please try again in a "
                 "few minutes. (Details logged for the operator.)")
    elif len(failed) > max(3, len(results) // 4):
        # Reason differs per market — don't guess one explanation for all.
        if market == "stocks":
            why = ("most are recently listed contracts without enough price "
                   "history yet; they join the boards as history builds.")
        elif market == "psx":
            why = ("the data source stopped responding partway through the "
                   "scan. Fetched symbols are cached — rescan in a few "
                   "minutes to fill the gaps.")
        else:
            why = ("the data source limited or dropped the remaining requests. "
                   "Fetched symbols are cached — rescan in a few minutes to "
                   "fill the gaps.")
        st.warning(f"{len(results)} of {len(results) + len(failed)} symbols "
                   f"scanned — {why} (Details logged for the operator.)")
with st.expander(f"Skipped symbols ({len(failed)})" if failed else "Skipped symbols (0)"):
    if failed:
        st.markdown(f"<span class='mut' style='font-size:.72rem'>{', '.join(failed)}</span>",
                    unsafe_allow_html=True)

# ---- universe probe: open the app URL with ?probe=universe ----
if st.query_params.get("probe") == "universe":
    with st.expander("STOCKS & COMMODITIES UNIVERSE", expanded=True):
        with st.spinner("Reading classified universe…"):
            _syms, _note = list_symbols("stocks")
        def _d(s):
            b = s[:-4] if s.endswith("USDT") else s
            return b[:-5] if b.endswith("STOCK") and len(b) > 5 else b
        names = sorted(_d(s) for s in _syms)
        st.markdown(f"**{len(names)} contracts** ({_note})")
        st.markdown("<span class='mut' style='font-size:.74rem'>"
                    + ", ".join(names) + "</span>", unsafe_allow_html=True)
        _oil = [n for n in names if "OIL" in n]
        _idx = [n for n in names if n in ("SPY","QQQ","SPX","NDX","DJI","US500",
                                          "US30","NAS100","JP225","DE40","UK100")]
        st.caption(f"Oil products found: {_oil or 'NONE'} · "
                   f"Index products found: {_idx or 'NONE'}")

# ---- contract-field probe: open the app URL with ?probe=fields ----
if st.query_params.get("probe") == "fields":
    with st.expander("CONTRACT FIELD PROBE", expanded=True):
        with st.spinner("Reading contract listing…"):
            info = probe_contract_fields()
        st.json(info, expanded=True)
        st.caption("Screenshot or copy this block.")

# ---- operator diagnostics: open the app URL with ?ops=wh4le-ops ----
if st.query_params.get("ops") == "wh4le-ops":
    with st.expander("OPERATOR DIAGNOSTICS", expanded=True):
        _all_err = {**LAST_ERRORS, **CRYPTO_ERRORS, **FX_ERRORS}
        if failed:
            _summary = {}
            for s in failed:
                r = _all_err.get(s, "unknown")
                _summary[r] = _summary.get(r, 0) + 1
            for r, n in sorted(_summary.items(), key=lambda x: -x[1]):
                st.markdown(f"`{n}x` {r}")
            st.caption("Raw provider messages appear above verbatim; "
                       "'cooling down' means the app stopped itself.")
            if REQUEST_LOG:
                st.markdown(f"**Provider requests this scan: {len(REQUEST_LOG)}**")
                for line in REQUEST_LOG[:14]:
                    st.markdown(f"<span class='mut' style='font-size:.7rem'>"
                                f"{line}</span>", unsafe_allow_html=True)
        else:
            st.markdown("No failures in the last scan.")
