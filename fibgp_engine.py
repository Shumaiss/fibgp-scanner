"""
fibgp_engine.py — Faithful Python port of the Trader Pro zone engine
(FibGP v11.5.3 base + Trader Pro strategy updates).

Ported line-for-line from the Pine Script v6 source. Trader Pro changes vs
the v11.5.3 base:
  • Asymmetric pivots: left=5, right=3 (faster zone confirmation)
  • Minimum-leg filter: a leg qualifies only if span >= MIN_LEG_ATR × ATR(14)
  • Cleared-anchor freshness: once a zone is broken/passed-through, only
    legs anchored strictly AFTER it may supply the next zone on that side
  • Leg extension: a found pivot pair extends forward through newer pivots
    to the full swing extreme (stops when the structure breaks) — stair-step
    trends form one full-structure leg
  • Upside targets removed (1.4/1.7/2.4/3.4 extensions dropped)
  • Stochastic health score is frozen at the cross (marker formula:
    extreme zone, momentum, K-velocity); live score and ⚠ near-cross
    warning removed. RSI(7) retained as scanner-side analytics. Execution model mirrors Pine exactly: the engine is a
bar-by-bar state machine fed one OHLCV bar at a time, in chronological order.

Ported components:
  • Pivot detection        — ta.pivothigh / ta.pivotlow (left=5, right=5)
  • Pivot registry         — add_pivot() with same-kind replacement, cap 80
  • Live endpoint          — runHi/runLo unconfirmed-swing tracking (useLive)
  • Candidate zones        — golden pocket 0.618–0.786 off the newest valid leg,
                             with chronology guard and stale-zone rejection
                             (f_supStale / f_resStale, 480-bar cap)
  • Zone lifecycle         — break detection, pass-through invalidation,
                             anchor-change replacement, entry (hit) counting
  • Strength stars         — inverted rating: 0 entries = ★★★★★, 5+ = ☆☆☆☆☆
  • Confluences            — EMA-in-zone (200>100>50>34>8 priority),
                             FVG-in-zone (3-candle, ≥50% unfilled, lookback 50)
  • Indicators             — RSI(7) Wilder, Stoch(5,3,3) + persistent BUY/SELL
                             signal, 3-point health score, near-cross warning
  • Upside targets         — 1.4 / 1.7 / 2.4 / 3.4 extensions of the active leg

Bar "time" is represented by integer bar index (data is daily & monotonic, so
index comparisons are equivalent to Pine's ms-timestamp comparisons).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np


# ============================== DEFAULT INPUTS ================================
# Mirrors the Pine input() defaults exactly.
PIV_LEFT = 5
PIV_RIGHT = 3
USE_LIVE = True
MAX_PIVOTS = 80

FVG_LOOKBACK = 50
STALE_CAP = 480          # f_supStale / f_resStale search cap
MIN_LEG_ATR = 2.5        # min leg span in ATRs to qualify as a zone

RSI_LEN = 7
STOCH_K = 5
STOCH_K_SMOOTH = 3
STOCH_D = 3
STOCH_NEAR = 5.0


EMA_LENS = (8, 34, 50, 100, 200)


# ============================== INDICATOR HELPERS ==============================
def pine_ema(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.ema: NaN for the first (length-1) bars, seeded with SMA of the
    first `length` values, then recursive with alpha = 2/(length+1)."""
    n = len(src)
    out = np.full(n, np.nan)
    if n < length:
        return out
    alpha = 2.0 / (length + 1.0)
    seed = np.nanmean(src[:length])
    out[length - 1] = seed
    for i in range(length, n):
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
    return out


def pine_rma(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.rma (Wilder smoothing): seeded with SMA, alpha = 1/length."""
    n = len(src)
    out = np.full(n, np.nan)
    if n < length:
        return out
    alpha = 1.0 / length
    seed = np.nanmean(src[:length])
    out[length - 1] = seed
    for i in range(length, n):
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
    return out


def pine_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             length: int = 14) -> np.ndarray:
    """Pine ta.atr: RMA of true range; TR on the first bar is high-low."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    return pine_rma(tr, length)


def pine_rsi(close: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.rsi: 100 - 100/(1 + rma(gains)/rma(losses))."""
    n = len(close)
    diff = np.diff(close, prepend=close[0])
    diff[0] = 0.0
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    # Pine's rma over change series effectively starts from bar 1
    rma_up = pine_rma(up[1:], length)
    rma_dn = pine_rma(dn[1:], length)
    out = np.full(n, np.nan)
    for i in range(len(rma_up)):
        u, d = rma_up[i], rma_dn[i]
        if not (math.isnan(u) or math.isnan(d)):
            if d == 0:
                out[i + 1] = 100.0
            else:
                out[i + 1] = 100.0 - 100.0 / (1.0 + u / d)
    return out


def pine_sma(src: np.ndarray, length: int) -> np.ndarray:
    n = len(src)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = src[i - length + 1 : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = float(np.mean(window))
    return out


def pine_stoch(close: np.ndarray, high: np.ndarray, low: np.ndarray,
               length: int) -> np.ndarray:
    """Pine ta.stoch: 100 * (close - lowest(low,len)) / (highest(high,len) - lowest(low,len))."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        hh = np.max(high[i - length + 1 : i + 1])
        ll = np.min(low[i - length + 1 : i + 1])
        rng = hh - ll
        out[i] = 100.0 * (close[i] - ll) / rng if rng != 0 else 0.0
    return out


# ============================== PIVOT DETECTION ================================
def pivot_high_at(high: np.ndarray, i: int, left: int, right: int) -> Optional[float]:
    """ta.pivothigh evaluated at bar i (confirmation bar). The pivot candidate
    is the bar at i - right. Returns the pivot value or None.

    TradingView semantics: the center value must not be strictly exceeded by
    any bar in the [i-right-left, i] window."""
    c = i - right
    if c - left < 0:
        return None
    center = high[c]
    window = high[c - left : i + 1]
    if np.any(window > center):
        return None
    return float(center)


def pivot_low_at(low: np.ndarray, i: int, left: int, right: int) -> Optional[float]:
    c = i - right
    if c - left < 0:
        return None
    center = low[c]
    window = low[c - left : i + 1]
    if np.any(window < center):
        return None
    return float(center)


# ============================== RESULT CONTAINERS ==============================
@dataclass
class Zone:
    top: float
    bot: float
    anchor_idx: int          # bar index of the leg endpoint (aSupAnc / aResAnc)
    leg_lo: float
    leg_hi: float
    hits: int                # entry count (aSupHits / aResHits)

    @property
    def stars(self) -> str:
        """f_stars(): inverted — fewer entries = more stars."""
        n = max(0, min(5, 5 - self.hits))
        return "★" * n + "☆" * (5 - n)

    @property
    def star_count(self) -> int:
        return max(0, min(5, 5 - self.hits))

    @property
    def mid(self) -> float:
        return (self.top + self.bot) / 2.0


@dataclass
class StochState:
    k: float = float("nan")
    d: float = float("nan")
    signal: str = "—"          # persistent BUY / SELL held until next cross
    score: int = 0             # 3-point live health score
    near_bull: bool = False    # ⚠ bullish cross incoming
    near_bear: bool = False    # ⚠ bearish cross incoming

    @property
    def dots(self) -> str:
        return "".join("●" if i <= self.score else "○" for i in range(1, 4))


@dataclass
class EngineResult:
    support: Optional[Zone]
    resistance: Optional[Zone]
    close: float
    rsi: float
    stoch: StochState
    sup_ema_tag: str = ""
    sup_fvg: bool = False
    res_ema_tag: str = ""
    res_fvg: bool = False
    upside_targets: list = field(default_factory=list)   # [(level, price), ...]
    entered_sup_today: bool = False
    entered_res_today: bool = False
    n_bars: int = 0

    def conf_tags(self, side: str) -> str:
        """f_confTag() equivalent — space-joined EMA / FVG tags."""
        parts = []
        if side == "sup":
            if self.sup_ema_tag:
                parts.append(self.sup_ema_tag)
            if self.sup_fvg:
                parts.append("FVG")
        else:
            if self.res_ema_tag:
                parts.append(self.res_ema_tag)
            if self.res_fvg:
                parts.append("FVG")
        return " ".join(parts)


# ============================== THE ENGINE =====================================
class FibGPEngine:
    """Bar-by-bar replication of the v11.5.3 zone state machine."""

    def __init__(self,
                 piv_left: int = PIV_LEFT,
                 piv_right: int = PIV_RIGHT,
                 use_live: bool = USE_LIVE,
                 fvg_lookback: int = FVG_LOOKBACK,
                 use_ema_conf: bool = True,
                 use_fvg_conf: bool = True,
                 ema_picks: dict | None = None,
                 min_leg_atr: float = MIN_LEG_ATR):
        self.piv_left = piv_left
        self.piv_right = piv_right
        self.use_live = use_live
        self.fvg_lookback = fvg_lookback
        self.use_ema_conf = use_ema_conf
        self.use_fvg_conf = use_fvg_conf
        self.min_leg_atr = min_leg_atr
        # Pine defaults: all five EMAs enabled
        self.ema_picks = ema_picks or {l: True for l in EMA_LENS}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _add_pivot(s_val: list, s_time: list, s_kind: list,
                   val: float, tm: int, kind: int, max_len: int = MAX_PIVOTS):
        """add_pivot(): alternation with same-kind replacement if more extreme."""
        n = len(s_val)
        if n == 0:
            s_val.append(val); s_time.append(tm); s_kind.append(kind)
        else:
            last_kind = s_kind[-1]
            last_val = s_val[-1]
            if last_kind == kind:
                replace = (kind == 1 and val > last_val) or (kind == -1 and val < last_val)
                if replace:
                    s_val[-1] = val
                    s_time[-1] = tm
            else:
                s_val.append(val); s_time.append(tm); s_kind.append(kind)
        if len(s_val) > max_len:
            s_val.pop(0); s_time.pop(0); s_kind.pop(0)

    @staticmethod
    def _sup_stale(close: np.ndarray, cur: int, anc_idx: int, zone_bot: float) -> bool:
        """f_supStale(): support candidate dead if any close since the leg
        formed went below its bottom. Pine caps the anchor search at 480 bars —
        if the anchor is older than that, no check is performed (barsBack=0)."""
        bars_back = cur - anc_idx
        if bars_back > STALE_CAP or bars_back <= 0:
            return False
        # Pine: for j = 1 to barsBack → closes from the anchor bar up to cur-1
        for j in range(1, bars_back + 1):
            if close[cur - j] < zone_bot:
                return True
        return False

    @staticmethod
    def _res_stale(close: np.ndarray, cur: int, anc_idx: int, zone_top: float) -> bool:
        bars_back = cur - anc_idx
        if bars_back > STALE_CAP or bars_back <= 0:
            return False
        for j in range(1, bars_back + 1):
            if close[cur - j] > zone_top:
                return True
        return False

    def _ema_in_zone(self, emas: dict, zone_top: float, zone_bot: float) -> str:
        """f_emaInZone(): highest-priority tracked EMA inside the zone."""
        for length in (200, 100, 50, 34, 8):
            if not self.ema_picks.get(length, True):
                continue
            v = emas[length]
            if not math.isnan(v) and zone_bot <= v <= zone_top:
                return f"EMA{length}"
        return ""

    def _fvg_in_zone(self, high: np.ndarray, low: np.ndarray,
                     cur: int, zone_top: float, zone_bot: float) -> bool:
        """f_fvgInZone(): scan last `lookback` bars for a ≥50%-unfilled FVG
        overlapping the zone. Direct port including loop bounds."""
        lookback = self.fvg_lookback
        # Pine: for k = 2 to lookback - 1
        for k in range(2, lookback):
            if cur - (k + 1) < 0:
                break
            hi_plus = high[cur - (k + 1)]
            lo_plus = low[cur - (k + 1)]
            hi_minus = high[cur - (k - 1)]
            lo_minus = low[cur - (k - 1)]

            # ---- Bullish FVG: low[k-1] > high[k+1] ----
            if lo_minus > hi_plus:
                gap_bot, gap_top = hi_plus, lo_minus
                gap_size = gap_top - gap_bot
                min_low_after = math.nan
                if k - 1 >= 1:
                    for j in range(0, k - 1):        # Pine: j = 0 to k-2
                        lj = low[cur - j]
                        if math.isnan(min_low_after) or lj < min_low_after:
                            min_low_after = lj
                if math.isnan(min_low_after):
                    min_low_after = gap_top
                rem_top = min(min_low_after, gap_top)
                if rem_top > gap_bot and gap_size > 0:
                    rem_size = rem_top - gap_bot
                    if rem_size / gap_size >= 0.5:
                        overlap_lo = max(gap_bot, zone_bot)
                        overlap_hi = min(rem_top, zone_top)
                        if overlap_hi > overlap_lo:
                            return True

            # ---- Bearish FVG: low[k+1] > high[k-1] ----
            if lo_plus > hi_minus:
                gap_bot, gap_top = hi_minus, lo_plus
                gap_size = gap_top - gap_bot
                max_high_after = math.nan
                if k - 1 >= 1:
                    for j in range(0, k - 1):
                        hj = high[cur - j]
                        if math.isnan(max_high_after) or hj > max_high_after:
                            max_high_after = hj
                if math.isnan(max_high_after):
                    max_high_after = gap_bot
                rem_bot = max(max_high_after, gap_bot)
                if gap_top > rem_bot and gap_size > 0:
                    rem_size = gap_top - rem_bot
                    if rem_size / gap_size >= 0.5:
                        overlap_lo = max(rem_bot, zone_bot)
                        overlap_hi = min(gap_top, zone_top)
                        if overlap_hi > overlap_lo:
                            return True
        return False

    # ------------------------------------------------------------------ main
    def run(self, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
            close: np.ndarray) -> EngineResult:
        n = len(close)
        L, R = self.piv_left, self.piv_right

        # ---- indicator series (vectorised precompute) ----
        emas = {length: pine_ema(close, length) for length in EMA_LENS}
        atr14 = pine_atr(high, low, close, 14)
        rsi_series = pine_rsi(close, RSI_LEN)
        stoch_raw = pine_stoch(close, high, low, STOCH_K)
        k_series = pine_sma(stoch_raw, STOCH_K_SMOOTH)
        d_series = pine_sma(k_series, STOCH_D)

        # ---- pivot registry state ----
        s_val: list[float] = []
        s_time: list[int] = []
        s_kind: list[int] = []

        run_hi = math.nan; run_hi_t = -1
        run_lo = math.nan; run_lo_t = -1

        # ---- active zone state (support) ----
        a_sup_top = math.nan; a_sup_bot = math.nan; a_sup_anc = -1
        a_sup_leg_lo = math.nan; a_sup_leg_hi = math.nan
        a_sup_hits = 0; a_sup_inside = False
        a_sup_closed_above = False; a_sup_closed_below = False
        sup_cleared_anc = None   # anchor of last support cleared downward

        # ---- active zone state (resistance) ----
        a_res_top = math.nan; a_res_bot = math.nan; a_res_anc = -1
        a_res_leg_lo = math.nan; a_res_leg_hi = math.nan
        a_res_hits = 0; a_res_inside = False
        a_res_closed_above = False; a_res_closed_below = False

        # ---- cleared-anchor freshness state ("no reaching backward") ----
        sup_cleared_anc: int | None = None
        res_cleared_anc: int | None = None

        # ---- stochastic persistent signal state ----
        stoch_signal = "—"
        stoch_frozen_score = 0

        entered_sup_today = False
        entered_res_today = False
        prev_in_sup = False
        prev_in_res = False

        for i in range(n):
            c = close[i]

            # ===== 1. pivot confirmation (anchor = bar i - R, Pine's time[pivRight])
            ph = pivot_high_at(high, i, L, R)
            pl = pivot_low_at(low, i, L, R)
            anc_t = i - R

            if ph is not None:
                self._add_pivot(s_val, s_time, s_kind, ph, anc_t, 1)
            if pl is not None:
                self._add_pivot(s_val, s_time, s_kind, pl, anc_t, -1)

            # ===== 2. running high/low (live endpoint tracking)
            if pl is not None:
                run_hi = high[i]; run_hi_t = i
            elif math.isnan(run_hi) or high[i] > run_hi:
                run_hi = high[i]; run_hi_t = i

            if ph is not None:
                run_lo = low[i]; run_lo_t = i
            elif math.isnan(run_lo) or low[i] < run_lo:
                run_lo = low[i]; run_lo_t = i

            # ===== 3. working copy + unconfirmed endpoint
            w_val = list(s_val); w_time = list(s_time); w_kind = list(s_kind)
            if self.use_live and len(w_kind) > 0:
                last_k = w_kind[-1]; last_t = w_time[-1]; last_v = w_val[-1]
                if last_k == -1 and not math.isnan(run_hi) and run_hi_t > last_t and run_hi > last_v:
                    w_val.append(run_hi); w_time.append(run_hi_t); w_kind.append(1)
                elif last_k == 1 and not math.isnan(run_lo) and run_lo_t > last_t and run_lo < last_v:
                    w_val.append(run_lo); w_time.append(run_lo_t); w_kind.append(-1)

            m = len(w_val)

            # ===== 4. candidate SUPPORT (newest low→high up-leg)
            cand_sup_top = math.nan; cand_sup_bot = math.nan
            cand_sup_leg_lo = math.nan; cand_sup_leg_hi = math.nan
            cand_sup_anc = -1
            if m >= 2:
                for j in range(m - 1, 0, -1):
                    if w_kind[j] == 1 and w_kind[j - 1] == -1:
                        hi_v = w_val[j]; lo_v = w_val[j - 1]; hi_t = w_time[j]
                        # leg extension: walk newer pivots; extend the high
                        # until a newer pivot low undercuts the leg low
                        if j < m - 1:
                            for k in range(j + 1, m):
                                if w_kind[k] == -1 and w_val[k] < lo_v:
                                    break
                                if w_kind[k] == 1 and w_val[k] > hi_v:
                                    hi_v = w_val[k]; hi_t = w_time[k]
                        chrono_ok = hi_t > w_time[j - 1]
                        fresh_ok = sup_cleared_anc is None or hi_t > sup_cleared_anc
                        rng = hi_v - lo_v
                        leg_ok = math.isnan(atr14[i]) or rng >= self.min_leg_atr * atr14[i]
                        if rng > 0 and chrono_ok and leg_ok and fresh_ok:
                            c_top = hi_v - 0.618 * rng
                            c_bot = hi_v - 0.786 * rng
                            if c_bot <= c and not self._sup_stale(close, i, hi_t, c_bot):
                                cand_sup_top = c_top; cand_sup_bot = c_bot
                                cand_sup_leg_lo = lo_v; cand_sup_leg_hi = hi_v
                                cand_sup_anc = hi_t
                                break

            # ===== 5. candidate RESISTANCE (newest high→low down-leg)
            cand_res_top = math.nan; cand_res_bot = math.nan
            cand_res_leg_lo = math.nan; cand_res_leg_hi = math.nan
            cand_res_anc = -1
            if m >= 2:
                for j in range(m - 1, 0, -1):
                    if w_kind[j] == -1 and w_kind[j - 1] == 1:
                        lo_v = w_val[j]; hi_v = w_val[j - 1]; lo_t = w_time[j]
                        # leg extension: walk newer pivots; extend the low
                        # until a newer pivot high exceeds the leg high
                        if j < m - 1:
                            for k in range(j + 1, m):
                                if w_kind[k] == 1 and w_val[k] > hi_v:
                                    break
                                if w_kind[k] == -1 and w_val[k] < lo_v:
                                    lo_v = w_val[k]; lo_t = w_time[k]
                        chrono_ok = lo_t > w_time[j - 1]
                        fresh_ok = res_cleared_anc is None or lo_t > res_cleared_anc
                        rng = hi_v - lo_v
                        leg_ok = math.isnan(atr14[i]) or rng >= self.min_leg_atr * atr14[i]
                        if rng > 0 and chrono_ok and leg_ok and fresh_ok:
                            c_bot = lo_v + 0.618 * rng
                            c_top = lo_v + 0.786 * rng
                            if c_top >= c and not self._res_stale(close, i, lo_t, c_top):
                                cand_res_top = c_top; cand_res_bot = c_bot
                                cand_res_leg_lo = lo_v; cand_res_leg_hi = hi_v
                                cand_res_anc = lo_t
                                break

            # ===== 6. SUPPORT state update
            if not math.isnan(a_sup_top):
                if c > a_sup_top:
                    a_sup_closed_above = True
                if c < a_sup_bot:
                    a_sup_closed_below = True

            sup_passed = (not math.isnan(a_sup_top)) and a_sup_closed_above and a_sup_closed_below
            sup_broken = (not math.isnan(a_sup_bot)) and c < a_sup_bot
            if sup_broken or sup_passed:
                sup_cleared_anc = a_sup_anc   # no reaching backward
                a_sup_top = math.nan; a_sup_bot = math.nan; a_sup_anc = -1
                a_sup_leg_lo = math.nan; a_sup_leg_hi = math.nan
                a_sup_hits = 0; a_sup_inside = False
                a_sup_closed_above = False; a_sup_closed_below = False

            if math.isnan(a_sup_top) and not math.isnan(cand_sup_top):
                a_sup_top = cand_sup_top; a_sup_bot = cand_sup_bot; a_sup_anc = cand_sup_anc
                a_sup_leg_lo = cand_sup_leg_lo; a_sup_leg_hi = cand_sup_leg_hi
                a_sup_hits = 0; a_sup_inside = False
                a_sup_closed_above = False; a_sup_closed_below = False

            if (not math.isnan(a_sup_top)) and (not math.isnan(cand_sup_top)) and cand_sup_anc != a_sup_anc:
                a_sup_top = cand_sup_top; a_sup_bot = cand_sup_bot; a_sup_anc = cand_sup_anc
                a_sup_leg_lo = cand_sup_leg_lo; a_sup_leg_hi = cand_sup_leg_hi
                a_sup_hits = 0; a_sup_inside = False
                a_sup_closed_above = False; a_sup_closed_below = False

            if not math.isnan(a_sup_top):
                now_inside = low[i] <= a_sup_top and high[i] >= a_sup_bot
                if now_inside and not a_sup_inside:
                    a_sup_hits += 1
                a_sup_inside = now_inside

            # ===== 7. RESISTANCE state update
            if not math.isnan(a_res_top):
                if c > a_res_top:
                    a_res_closed_above = True
                if c < a_res_bot:
                    a_res_closed_below = True

            res_passed = (not math.isnan(a_res_top)) and a_res_closed_above and a_res_closed_below
            res_broken = (not math.isnan(a_res_top)) and c > a_res_top
            if res_broken or res_passed:
                res_cleared_anc = a_res_anc   # no reaching backward
                a_res_top = math.nan; a_res_bot = math.nan; a_res_anc = -1
                a_res_leg_lo = math.nan; a_res_leg_hi = math.nan
                a_res_hits = 0; a_res_inside = False
                a_res_closed_above = False; a_res_closed_below = False

            if math.isnan(a_res_top) and not math.isnan(cand_res_top):
                a_res_top = cand_res_top; a_res_bot = cand_res_bot; a_res_anc = cand_res_anc
                a_res_leg_lo = cand_res_leg_lo; a_res_leg_hi = cand_res_leg_hi
                a_res_hits = 0; a_res_inside = False
                a_res_closed_above = False; a_res_closed_below = False

            if (not math.isnan(a_res_top)) and (not math.isnan(cand_res_top)) and cand_res_anc != a_res_anc:
                a_res_top = cand_res_top; a_res_bot = cand_res_bot; a_res_anc = cand_res_anc
                a_res_leg_lo = cand_res_leg_lo; a_res_leg_hi = cand_res_leg_hi
                a_res_hits = 0; a_res_inside = False
                a_res_closed_above = False; a_res_closed_below = False

            if not math.isnan(a_res_top):
                now_inside = low[i] <= a_res_top and high[i] >= a_res_bot
                if now_inside and not a_res_inside:
                    a_res_hits += 1
                a_res_inside = now_inside

            # ===== 8. zone-entry alert equivalents (last bar only matters)
            in_sup_now = (not math.isnan(a_sup_top)) and high[i] >= a_sup_bot and low[i] <= a_sup_top
            in_res_now = (not math.isnan(a_res_top)) and high[i] >= a_res_bot and low[i] <= a_res_top
            if i == n - 1:
                entered_sup_today = in_sup_now and not prev_in_sup
                entered_res_today = in_res_now and not prev_in_res
            prev_in_sup = in_sup_now
            prev_in_res = in_res_now

            # ===== 9. stochastic persistent signal
            k_now, k_prev = k_series[i], k_series[i - 1] if i > 0 else math.nan
            d_now, d_prev = d_series[i], d_series[i - 1] if i > 0 else math.nan
            if not any(math.isnan(x) for x in (k_now, k_prev, d_now, d_prev)):
                cross_up = k_prev <= d_prev and k_now > d_now
                cross_dn = k_prev >= d_prev and k_now < d_now
                if cross_up or cross_dn:
                    is_buy = cross_up
                    stoch_signal = "BUY" if is_buy else "SELL"
                    # frozen-at-cross health score (chart marker formula)
                    sc = 0
                    if (is_buy and k_now < 20) or (not is_buy and k_now > 80):
                        sc += 1
                    if (is_buy and close[i] > close[i - 1]) or \
                       (not is_buy and close[i] < close[i - 1]):
                        sc += 1
                    if abs(k_now - k_prev) > 5:
                        sc += 1
                    stoch_frozen_score = sc

        # =================== FINAL-BAR OUTPUT (render equivalent) ===============
        last = n - 1
        support = None
        resistance = None
        if not math.isnan(a_sup_top):
            support = Zone(a_sup_top, a_sup_bot, a_sup_anc,
                           a_sup_leg_lo, a_sup_leg_hi, a_sup_hits)
        if not math.isnan(a_res_top):
            resistance = Zone(a_res_top, a_res_bot, a_res_anc,
                              a_res_leg_lo, a_res_leg_hi, a_res_hits)

        emas_last = {length: emas[length][last] for length in EMA_LENS}

        sup_ema_tag = ""; sup_fvg = False
        res_ema_tag = ""; res_fvg = False
        if support is not None:
            if self.use_ema_conf:
                sup_ema_tag = self._ema_in_zone(emas_last, support.top, support.bot)
            if self.use_fvg_conf:
                sup_fvg = self._fvg_in_zone(high, low, last, support.top, support.bot)
        if resistance is not None:
            if self.use_ema_conf:
                res_ema_tag = self._ema_in_zone(emas_last, resistance.top, resistance.bot)
            if self.use_fvg_conf:
                res_fvg = self._fvg_in_zone(high, low, last, resistance.top, resistance.bot)

        # ---- stochastic final state (Trader Pro: score frozen at cross,
        #      no live score, no near-cross warning) ----
        st = StochState()
        st.k = k_series[last]
        st.d = d_series[last]
        st.signal = stoch_signal
        st.score = stoch_frozen_score if stoch_signal != "—" else 0

        return EngineResult(
            support=support,
            resistance=resistance,
            close=float(close[last]),
            rsi=float(rsi_series[last]),
            stoch=st,
            sup_ema_tag=sup_ema_tag,
            sup_fvg=sup_fvg,
            res_ema_tag=res_ema_tag,
            res_fvg=res_fvg,
            entered_sup_today=entered_sup_today,
            entered_res_today=entered_res_today,
            n_bars=n,
        )


# ============================== SCANNER CLASSIFIER =============================
@dataclass
class ScanRow:
    symbol: str
    close: float
    status: str            # "NEAR_SUPPORT" | "IN_SUPPORT" | "NEAR_RESISTANCE" | "IN_RESISTANCE"
    zone_bot: float
    zone_top: float
    distance_pct: float    # 0 when inside the zone
    stars: str
    star_count: int
    conf: str
    rsi: float
    stoch_signal: str
    stoch_dots: str
    stoch_warn: str        # "", "⚠↑", "⚠↓"
    zone_age: int          # bars since anchor
    entered_today: bool


def classify(symbol: str, result: EngineResult, near_pct: float) -> list[ScanRow]:
    """Bucket a symbol against its active zones. One active zone per side —
    matching the Pine engine — so a symbol can yield up to two rows
    (sandwiched between support below and resistance above)."""
    rows: list[ScanRow] = []
    c = result.close
    warn = "⚠↑" if result.stoch.near_bull else ("⚠↓" if result.stoch.near_bear else "")

    if result.support is not None:
        z = result.support
        if z.bot <= c <= z.top:
            status, dist = "IN_SUPPORT", 0.0
        elif c > z.top:
            dist = (c - z.top) / z.top * 100.0
            status = "NEAR_SUPPORT" if dist <= near_pct else ""
        else:
            status, dist = "", math.inf
        if status:
            rows.append(ScanRow(symbol, c, status, z.bot, z.top, dist,
                                z.stars, z.star_count, result.conf_tags("sup"),
                                result.rsi, result.stoch.signal, result.stoch.dots,
                                warn, result.n_bars - 1 - z.anchor_idx,
                                result.entered_sup_today))

    if result.resistance is not None:
        z = result.resistance
        if z.bot <= c <= z.top:
            status, dist = "IN_RESISTANCE", 0.0
        elif c < z.bot:
            dist = (z.bot - c) / c * 100.0
            status = "NEAR_RESISTANCE" if dist <= near_pct else ""
        else:
            status, dist = "", math.inf
        if status:
            rows.append(ScanRow(symbol, c, status, z.bot, z.top, dist,
                                z.stars, z.star_count, result.conf_tags("res"),
                                result.rsi, result.stoch.signal, result.stoch.dots,
                                warn, result.n_bars - 1 - z.anchor_idx,
                                result.entered_res_today))
    return rows
