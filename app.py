"""
FibGP Scanner — PSX daily golden-pocket proximity scanner.

Companion web app to the "Fib Golden Pocket — Premium + Confluences (v11.5.3)"
TradingView indicator. Runs a faithful Python port of the v11.5.3 zone engine
over daily PSX data and buckets symbols into Near/In Support and Near/In
Resistance boards.

Run:  streamlit run app.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st

from fibgp_engine import FibGPEngine, EngineResult, classify, ScanRow
from psx_fetch import fetch_daily, LAST_ERRORS
from symbols import KSE100, QUICK25

# ============================== PALETTE (FibGP suite) ==========================
TEAL = "#4DB6AC"
TEAL_LT = "#80CBC4"
ROSE = "#F47174"
PEACH = "#FFAB91"
BG = "#12121C"
PANEL = "#1E1E2E"
HDR = "#2D2D44"
TXT = "#E0E0E0"
MUTE = "#8888A0"

st.set_page_config(page_title="FibGP Scanner — PSX", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Rajdhani:wght@500;600;700&display=swap');

.stApp {{ background: {BG}; }}
html, body, [class*="css"] {{ font-family: 'JetBrains Mono', monospace; color: {TXT}; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {HDR}; }}
h1, h2, h3 {{ font-family: 'Rajdhani', sans-serif; letter-spacing: 0.04em; }}

.fib-title {{ font-family:'Rajdhani',sans-serif; font-size:2rem; font-weight:700;
  letter-spacing:.12em; margin-bottom:0; }}
.fib-title .t {{ color:{TEAL}; }} .fib-title .r {{ color:{ROSE}; }}
.fib-sub {{ color:{MUTE}; font-size:.78rem; letter-spacing:.08em; margin-top:2px; }}

.board {{ background:{PANEL}; border:1px solid {HDR}; border-radius:10px;
  padding:14px 16px 10px 16px; }}
.board-hd {{ font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.05rem;
  letter-spacing:.14em; padding-bottom:8px; border-bottom:1px solid {HDR};
  margin-bottom:8px; display:flex; justify-content:space-between; align-items:baseline; }}
.board-hd .n {{ color:{MUTE}; font-family:'JetBrains Mono',monospace; font-size:.75rem; }}

.zrow {{ display:grid; grid-template-columns: 92px 1fr 118px; gap:10px;
  padding:9px 6px; border-bottom:1px solid {HDR}; align-items:center; }}
.zrow:last-child {{ border-bottom:none; }}
.zsym {{ font-weight:700; font-size:1.0rem; }}
.zsub {{ color:{MUTE}; font-size:.68rem; margin-top:1px; }}
.zmid {{ font-size:.78rem; line-height:1.55; }}
.zright {{ text-align:right; font-size:.78rem; line-height:1.5; }}
.stars {{ letter-spacing:.08em; }}
.pill {{ display:inline-block; padding:1px 7px; border-radius:9px; font-size:.66rem;
  margin-left:5px; vertical-align:1px; }}
.pill-sup {{ background:rgba(77,182,172,.16); color:{TEAL_LT}; border:1px solid rgba(77,182,172,.35); }}
.pill-res {{ background:rgba(244,113,116,.16); color:{PEACH}; border:1px solid rgba(244,113,116,.35); }}
.pill-new {{ background:rgba(255,255,255,.10); color:#fff; border:1px solid rgba(255,255,255,.3); }}
.badge-in {{ font-weight:700; letter-spacing:.06em; }}
.dist-track {{ height:4px; background:{HDR}; border-radius:2px; margin-top:5px; }}
.dist-fill {{ height:4px; border-radius:2px; }}
.mut {{ color:{MUTE}; }}
</style>
""", unsafe_allow_html=True)


# ============================== SIDEBAR ========================================
with st.sidebar:
    st.markdown(f"<div class='fib-title'>Fib<span class='t'>G</span><span class='r'>P</span></div>"
                f"<div class='fib-sub'>v11.5.3 ENGINE · PSX · DAILY</div>",
                unsafe_allow_html=True)
    st.write("")

    universe_choice = st.radio("Universe", ["KSE-100", "Quick 25", "Custom"],
                               horizontal=True)
    if universe_choice == "Custom":
        custom_txt = st.text_area("Symbols (comma / newline separated)",
                                  value="HBL, UBL, OGDC, LUCK, SYS",
                                  height=90)
        symbols = sorted({s.strip().upper() for s in
                          custom_txt.replace("\n", ",").split(",") if s.strip()})
    elif universe_choice == "Quick 25":
        symbols = QUICK25
    else:
        symbols = KSE100

    near_pct = st.slider("Near threshold (%)", 0.5, 5.0, 2.0, 0.25,
                         help="Max distance from the zone edge to count as 'near'.")
    lookback_m = st.slider("History (months)", 8, 24, 18,
                           help="~18 months ≈ 370 daily bars: EMA200 warm-up + zone history.")

    with st.expander("Engine parameters"):
        piv_l = st.number_input("Pivot left bars", 1, 20, 5)
        piv_r = st.number_input("Pivot right bars", 1, 20, 5)
        use_live = st.checkbox("Use live (unconfirmed) endpoint", True)
        use_ema = st.checkbox("EMA confluence", True)
        use_fvg = st.checkbox("FVG confluence", True)
        fvg_lb = st.number_input("FVG lookback (bars)", 10, 200, 50)

    run_scan = st.button("Scan", type="primary", use_container_width=True)
    st.caption(f"{len(symbols)} symbols · PSX EOD data · SCSTrade + DPS")


# ============================== DATA + SCAN ====================================
if "ohlc_cache" not in st.session_state:
    st.session_state.ohlc_cache = {}      # (symbol, start_iso) -> DataFrame|None
if "scan" not in st.session_state:
    st.session_state.scan = None          # (rows, results_by_symbol, failed, stamp)


def get_data(symbol: str, start: date) -> pd.DataFrame | None:
    key = (symbol, start.isoformat())
    cache = st.session_state.ohlc_cache
    if key not in cache:
        cache[key] = fetch_daily(symbol, start)
    return cache[key]


def run_full_scan(syms: list[str], start: date, engine: FibGPEngine,
                  threshold: float):
    rows: list[ScanRow] = []
    results: dict[str, EngineResult] = {}
    failed: list[str] = []

    prog = st.progress(0.0, text="Fetching PSX data…")
    done = 0
    total = len(syms)

    # fetch (I/O-bound) in a small pool; engine runs on the main thread
    to_fetch = [s for s in syms
                if (s, start.isoformat()) not in st.session_state.ohlc_cache]
    fetched: dict[str, pd.DataFrame | None] = {}
    if to_fetch:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(fetch_daily, s, start): s for s in to_fetch}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    fetched[s] = fut.result()
                except Exception:
                    fetched[s] = None
                done += 1
                prog.progress(done / (total + len(syms)),
                              text=f"Fetching PSX data… {s} ({done}/{len(to_fetch)})")
        for s, df in fetched.items():
            st.session_state.ohlc_cache[(s, start.isoformat())] = df

    done = len(to_fetch)
    for s in syms:
        df = st.session_state.ohlc_cache.get((s, start.isoformat()))
        done += 1
        prog.progress(min(1.0, done / (total + len(to_fetch))),
                      text=f"Running FibGP engine… {s}")
        if df is None or len(df) < 60:
            failed.append(s)
            continue
        try:
            res = engine.run(df["Open"].to_numpy(), df["High"].to_numpy(),
                             df["Low"].to_numpy(), df["Close"].to_numpy())
            results[s] = res
            rows.extend(classify(s, res, threshold))
        except Exception:
            failed.append(s)
    prog.empty()
    return rows, results, failed


if run_scan:
    start = date.today() - relativedelta(months=int(lookback_m))
    engine = FibGPEngine(piv_left=int(piv_l), piv_right=int(piv_r),
                         use_live=use_live, fvg_lookback=int(fvg_lb),
                         use_ema_conf=use_ema, use_fvg_conf=use_fvg)
    rows, results, failed = run_full_scan(symbols, start, engine, near_pct)
    st.session_state.scan = (rows, results, failed,
                             pd.Timestamp.now().strftime("%d %b %Y %H:%M"))


# ============================== RENDER =========================================
st.markdown(f"<div class='fib-title'>GOLDEN POCKET <span class='t'>SCANNER</span></div>"
            f"<div class='fib-sub'>PSX · DAILY TIMEFRAME · FibGP v11.5.3 ZONE ENGINE · "
            f"0.618 – 0.786 RETRACEMENT</div>", unsafe_allow_html=True)
st.write("")

if st.session_state.scan is None:
    st.markdown(f"<span class='mut'>Pick a universe and press <b>Scan</b>. "
                f"Zones follow the chart indicator exactly: one active support and one "
                f"active resistance zone per symbol, with break, pass-through and "
                f"stale-leg invalidation.</span>", unsafe_allow_html=True)
    st.stop()

rows, results, failed, stamp = st.session_state.scan

sup_rows = sorted([r for r in rows if r.status in ("NEAR_SUPPORT", "IN_SUPPORT")],
                  key=lambda r: r.distance_pct)
res_rows = sorted([r for r in rows if r.status in ("NEAR_RESISTANCE", "IN_RESISTANCE")],
                  key=lambda r: r.distance_pct)


def fmt_px(v: float) -> str:
    return f"{v:,.2f}"


def render_row(r: ScanRow, side: str) -> str:
    accent = TEAL if side == "sup" else ROSE
    lt = TEAL_LT if side == "sup" else PEACH
    pill_cls = "pill-sup" if side == "sup" else "pill-res"
    in_zone = r.status.startswith("IN_")

    pills = ""
    for tag in r.conf.split():
        pills += f"<span class='pill {pill_cls}'>{tag}</span>"
    if r.entered_today:
        pills += "<span class='pill pill-new'>NEW</span>"

    if in_zone:
        right_top = f"<span class='badge-in' style='color:{accent}'>◆ IN ZONE</span>"
        fill_w = 100
    else:
        right_top = f"<span style='color:{accent}'>{r.distance_pct:.2f}%</span> <span class='mut'>away</span>"
        fill_w = max(6, int(100 * (1 - r.distance_pct / 5.0)))

    sig = r.stoch_signal
    sig_col = TEAL_LT if sig == "BUY" else (PEACH if sig == "SELL" else MUTE)
    warn = f" <span style='color:{TEAL if r.stoch_warn == '⚠↑' else ROSE}'>{r.stoch_warn}</span>" if r.stoch_warn else ""

    return f"""
    <div class='zrow'>
      <div>
        <div class='zsym' style='color:{accent}'>{r.symbol}</div>
        <div class='zsub'>{fmt_px(r.close)}</div>
      </div>
      <div class='zmid'>
        <span style='color:{lt}'>{fmt_px(r.zone_bot)} – {fmt_px(r.zone_top)}</span>
        <span class='stars' style='color:{accent}'> {r.stars}</span>{pills}<br>
        <span class='mut'>RSI(7) {r.rsi:.1f} · Stoch </span><span style='color:{sig_col}'>{sig} {r.stoch_dots}</span>{warn}
        <span class='mut'> · zone age {r.zone_age}d</span>
      </div>
      <div class='zright'>
        {right_top}
        <div class='dist-track'><div class='dist-fill' style='width:{fill_w}%; background:{accent}'></div></div>
      </div>
    </div>"""


def render_board(title: str, board_rows: list[ScanRow], side: str):
    accent = TEAL if side == "sup" else ROSE
    body = "".join(render_row(r, side) for r in board_rows) if board_rows \
        else f"<div class='zrow'><span class='mut'>No symbols within threshold.</span></div>"
    st.markdown(f"""
    <div class='board'>
      <div class='board-hd'><span style='color:{accent}'>{title}</span>
      <span class='n'>{len(board_rows)}</span></div>
      {body}
    </div>""", unsafe_allow_html=True)


c1, c2 = st.columns(2)
with c1:
    render_board("▲ NEAR / IN SUPPORT", sup_rows, "sup")
with c2:
    render_board("▼ NEAR / IN RESISTANCE", res_rows, "res")

st.write("")
meta = f"Scanned {len(results)} symbols · {stamp}"
if failed:
    meta += f" · no data / insufficient bars: {', '.join(failed)}"
st.caption(meta)
if failed and not results:
    reasons = {LAST_ERRORS.get(s, "unknown") for s in failed}
    st.error("All fetches failed — reason(s): " + "; ".join(sorted(reasons))
             + ". The PSX data server may be blocking this host.")

# ---- CSV export ----
if rows:
    export = pd.DataFrame([{
        "Symbol": r.symbol, "Close": r.close, "Status": r.status,
        "ZoneBot": r.zone_bot, "ZoneTop": r.zone_top,
        "DistancePct": round(r.distance_pct, 3), "Stars": r.star_count,
        "Confluence": r.conf, "RSI7": round(r.rsi, 1),
        "Stoch": f"{r.stoch_signal} {r.stoch_dots}".strip(),
        "ZoneAgeBars": r.zone_age, "EnteredToday": r.entered_today,
    } for r in rows]).sort_values(["Status", "DistancePct"])
    st.download_button("Export CSV", export.to_csv(index=False),
                       file_name=f"fibgp_scan_{date.today().isoformat()}.csv",
                       mime="text/csv")

# ---- Symbol inspector ----
with st.expander("Symbol inspector — full engine output"):
    pick = st.selectbox("Symbol", sorted(results.keys()))
    if pick:
        r = results[pick]
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Close** {fmt_px(r.close)} · **RSI(7)** {r.rsi:.1f} · "
                        f"**Stoch** K {r.stoch.k:.1f} / D {r.stoch.d:.1f} → "
                        f"{r.stoch.signal} {r.stoch.dots}"
                        + (" ⚠" if (r.stoch.near_bull or r.stoch.near_bear) else ""))
            if r.support:
                z = r.support
                st.markdown(f"<span style='color:{TEAL}'>**Support** "
                            f"{fmt_px(z.bot)} – {fmt_px(z.top)} {z.stars}</span> "
                            f"<span class='mut'>leg {fmt_px(z.leg_lo)}→{fmt_px(z.leg_hi)} · "
                            f"hits {z.hits} · {r.conf_tags('sup') or 'no confluence'}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<span class='mut'>No active support zone.</span>",
                            unsafe_allow_html=True)
            if r.resistance:
                z = r.resistance
                st.markdown(f"<span style='color:{ROSE}'>**Resistance** "
                            f"{fmt_px(z.bot)} – {fmt_px(z.top)} {z.stars}</span> "
                            f"<span class='mut'>leg {fmt_px(z.leg_hi)}→{fmt_px(z.leg_lo)} · "
                            f"hits {z.hits} · {r.conf_tags('res') or 'no confluence'}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<span class='mut'>No active resistance zone.</span>",
                            unsafe_allow_html=True)
        with right:
            if r.upside_targets:
                st.markdown("**Upside targets**")
                for lv, px in r.upside_targets:
                    st.markdown(f"<span style='color:{TEAL_LT}'>{lv}</span> "
                                f"→ {fmt_px(px)}", unsafe_allow_html=True)
            else:
                st.markdown("<span class='mut'>No upside targets (filtered).</span>",
                            unsafe_allow_html=True)
