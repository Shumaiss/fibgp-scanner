"""
liq_engine.py — Liquidity engine: buy-side / sell-side levels (v1.0)

A faithful Python port of the liquidity block of the chart indicator.

  BSL (buy-side)  — swing highs resting ABOVE price
  SSL (sell-side) — swing lows resting BELOW price

Rules, in the same order the Pine applies them on every bar:

  1. ADD    a swing high / low confirmed with LIQ_L bars left and LIQ_R bars
            right becomes a level. If it lands within LIQ_TOL_M x ATR(14) of
            a live level on the same side the two MERGE: the level keeps the
            more extreme price and its touch count goes up (equal highs /
            equal lows). Otherwise it is a new level; past LIQ_MAX levels per
            side the oldest is dropped.
  2. SCAN   a level is removed the moment price trades through it — whether
            the bar closed back (a sweep) or closed beyond (a break).

Because a taken level is removed at once, every live BSL is above price and
every live SSL is below it. The scanner shows the nearest of each.

The result has the same shape as the other engines, so the boards render it
unchanged: the nearest SSL fills the "support" slot and the nearest BSL fills
"resistance". A level is a single price, so its zone has top == bot, and
`line_tol` (the merge tolerance on the last bar) tells the scanner how close
counts as price sitting AT the level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fibgp_engine import (
    PIV_LEFT, PIV_RIGHT, RSI_LEN,
    STOCH_K, STOCH_K_SMOOTH, STOCH_D,
    pine_atr, pine_rsi, pine_sma, pine_stoch,
    pivot_high_at, pivot_low_at,
    Zone, StochState, EngineResult,
)

# ============================== FIXED CONFIG ==================================
LIQ_L = 2            # swing bars to the left
LIQ_R = 2            # swing bars to the right
LIQ_TOL_M = 0.15     # merge tolerance, in ATRs
LIQ_MAX = 8          # levels kept per side
ATR_LEN = 14


@dataclass
class _Level:
    px: float
    anchor: int      # bar index of the swing
    cnt: int = 1     # touches merged into this level


@dataclass
class LiqResult(EngineResult):
    """EngineResult plus the 'price is AT the level' tolerance."""
    is_lines: bool = True
    line_tol: float = 0.0
    sup_touches: int = 0
    res_touches: int = 0


class LiquidityEngine:
    """Bar-by-bar replay producing the live BSL / SSL levels for a series."""

    def __init__(self, liq_l: int = LIQ_L, liq_r: int = LIQ_R,
                 tol_m: float = LIQ_TOL_M, liq_max: int = LIQ_MAX,
                 atr_len: int = ATR_LEN):
        self.liq_l = liq_l
        self.liq_r = liq_r
        self.tol_m = tol_m
        self.liq_max = liq_max
        self.atr_len = atr_len

    # ------------------------------------------------------------------ #
    def _add(self, arr: list, px: float, anchor: int, is_high: bool,
             tol: float):
        """f_liqAdd(): merge into the first live level within tol, else push."""
        for lv in arr:
            if abs(lv.px - px) <= tol:
                lv.px = max(lv.px, px) if is_high else min(lv.px, px)
                lv.cnt += 1
                return
        arr.append(_Level(px, anchor))
        if len(arr) > self.liq_max:
            arr.pop(0)                    # oldest goes first

    @staticmethod
    def _scan(arr: list, ext: float, is_high: bool) -> list:
        """f_liqScan(): drop every level price traded through on this bar."""
        if is_high:
            return [lv for lv in arr if not ext > lv.px]
        return [lv for lv in arr if not ext < lv.px]

    # ------------------------------------------------------------------ #
    def run(self, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
            close: np.ndarray) -> LiqResult:
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = len(c)

        atr = pine_atr(h, l, c, self.atr_len)
        rsi_arr = pine_rsi(c, RSI_LEN)

        bsl: list[_Level] = []
        ssl: list[_Level] = []

        # trend tags — identical rule to the chart's trend badge
        last_swing_hi = math.nan
        last_swing_lo = math.nan
        last_hi_anc = None
        last_lo_anc = None
        struct_hi = ""
        struct_lo = ""

        tol = 0.0
        for i in range(n):
            a = atr[i]
            tol = 0.0 if math.isnan(a) else a * self.tol_m      # nz(atr14)

            # ---------- 1. add newly confirmed swings ----------
            ph = pivot_high_at(h, i, self.liq_l, self.liq_r)
            pl = pivot_low_at(l, i, self.liq_l, self.liq_r)
            if ph is not None:
                self._add(bsl, ph, i - self.liq_r, True, tol)
            if pl is not None:
                self._add(ssl, pl, i - self.liq_r, False, tol)

            # ---------- 2. remove anything price traded through ----------
            bsl = self._scan(bsl, h[i], True)
            ssl = self._scan(ssl, l[i], False)

            # ---------- 3. trend tags (5/1 pivots) ----------
            tph = pivot_high_at(h, i, PIV_LEFT, PIV_RIGHT)
            tpl = pivot_low_at(l, i, PIV_LEFT, PIV_RIGHT)
            anc_t = i - PIV_RIGHT
            if tph is not None:
                dup = (last_hi_anc is not None and tph == last_swing_hi
                       and anc_t - last_hi_anc <= 1)
                if not dup:
                    struct_hi = ("H" if math.isnan(last_swing_hi)
                                 else "HH" if tph > last_swing_hi else "LH")
                    last_swing_hi = tph
                last_hi_anc = anc_t
            if tpl is not None:
                dup = (last_lo_anc is not None and tpl == last_swing_lo
                       and anc_t - last_lo_anc <= 1)
                if not dup:
                    struct_lo = ("L" if math.isnan(last_swing_lo)
                                 else "HL" if tpl > last_swing_lo else "LL")
                    last_swing_lo = tpl
                last_lo_anc = anc_t

        # ---------- 4. nearest level each side ----------
        px = float(c[-1]) if n else math.nan
        near_ssl = max(ssl, key=lambda lv: lv.px) if ssl else None
        near_bsl = min(bsl, key=lambda lv: lv.px) if bsl else None

        if not struct_hi or not struct_lo:
            structure = "Ranging"
        elif struct_hi == "HH" and struct_lo == "HL":
            structure = "Bullish"
        elif struct_hi == "LH" and struct_lo == "LL":
            structure = "Bearish"
        else:
            structure = "Ranging"

        k_arr = pine_sma(pine_stoch(c, h, l, STOCH_K), STOCH_K_SMOOTH)
        d_arr = pine_sma(k_arr, STOCH_D)
        stoch = StochState(k=float(k_arr[-1]) if n else math.nan,
                           d=float(d_arr[-1]) if n else math.nan)
        if n and not math.isnan(k_arr[-1]) and not math.isnan(d_arr[-1]):
            stoch.signal = "BUY" if k_arr[-1] >= d_arr[-1] else "SELL"

        return LiqResult(
            support=self._to_zone(near_ssl),
            resistance=self._to_zone(near_bsl),
            close=px,
            rsi=float(rsi_arr[-1]) if n else math.nan,
            stoch=stoch,
            sup_ema_tag=self._tag(near_ssl, "SSL", "EQL"),
            res_ema_tag=self._tag(near_bsl, "BSL", "EQH"),
            struct_hi=struct_hi, struct_lo=struct_lo, structure=structure,
            n_bars=n,
            is_lines=True,
            line_tol=float(tol),
            sup_touches=near_ssl.cnt if near_ssl else 0,
            res_touches=near_bsl.cnt if near_bsl else 0,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _tag(lv, base: str, eq: str) -> str:
        if lv is None:
            return ""
        return f"{eq} x{lv.cnt}" if lv.cnt > 1 else base

    @staticmethod
    def _to_zone(lv):
        """A level is one price: top == bot. Stars = touch count (max 5)."""
        if lv is None:
            return None
        stars = max(1, min(5, lv.cnt))
        return Zone(top=float(lv.px), bot=float(lv.px), anchor_idx=lv.anchor,
                    leg_lo=float(lv.px), leg_hi=float(lv.px),
                    hits=5 - stars)          # Zone.stars is 5 - hits
