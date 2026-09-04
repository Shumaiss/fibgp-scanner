"""
ob_engine.py — Order Block engine (v1.0)

A faithful Python port of the order-block half of the Ghost indicator. Only
order blocks are ported: gaps and FVGs are deliberately left out.

An order block is the last opposing candle before a displacement move that
also breaks structure:

  bullish OB — low[0] > high[2]  (3-bar upward displacement)
               AND high[0] > last confirmed swing high   (break of structure)
               -> zone = the most recent bearish candle within the last
                  OB_LOOK bars, taken wick to wick

  bearish OB — high[0] < low[2]  (3-bar downward displacement)
               AND low[0] < last confirmed swing low
               -> zone = the most recent bullish candle, wick to wick

Zones live until price trades through them. Every bar, a zone that price has
entered SHRINKS to the untouched remainder, and dies once fully traversed or
once the remainder falls below the minimum size (ATR * MIN_ATR_X). At most
OB_MAX zones are kept per side; the oldest is dropped first.

Ordering matches the Pine exactly: mitigation runs on every bar (live bar
included), detection only on closed bars, and the swing structure rolls
forward only after the current bar's blocks have been judged.

The result object is the same shape the FibGP engine returns, so the scanner
renders either strategy without changes: the nearest live zone below price
fills the "support" slot and the nearest live zone above fills "resistance".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fibgp_engine import (
    PIV_LEFT, PIV_RIGHT, RSI_LEN,
    STOCH_K, STOCH_K_SMOOTH, STOCH_D,
    pine_atr, pine_rsi, pine_sma, pine_stoch,
    pivot_high_at, pivot_low_at,
    Zone, StochState, EngineResult,
)

# ============================== FIXED CONFIG ==================================
# Mirrors the Ghost script's fixed configuration block.
BOS_LEN = 5          # swing pivot length used for the break-of-structure test
OB_LOOK = 8          # how far back to hunt for the opposing candle
OB_MAX = 6           # zones kept per side
ATR_LEN = 14
MIN_ATR_X = 0.25     # a zone must span at least this many ATRs
DEPTH = 40           # rolling bar buffer, as in the Pine


@dataclass
class _Block:
    """A live order block."""
    top: float
    bot: float
    bull: bool
    anchor: int          # bar index of the originating candle
    tapped: bool = False


class OrderBlockEngine:
    """Bar-by-bar replay producing the live order blocks for a series."""

    def __init__(self, bos_len: int = BOS_LEN, ob_look: int = OB_LOOK,
                 ob_max: int = OB_MAX, atr_len: int = ATR_LEN,
                 min_atr_x: float = MIN_ATR_X,
                 detect_on_last: bool = False):
        self.bos_len = bos_len
        self.ob_look = ob_look
        self.ob_max = ob_max
        self.atr_len = atr_len
        self.min_atr_x = min_atr_x
        # the final bar may still be forming; the Pine only detects on
        # confirmed bars, so by default we do the same
        self.detect_on_last = detect_on_last

    # ------------------------------------------------------------------ #
    def _size_ok(self, top: float, bot: float, atr: float) -> bool:
        """sizeOk() in ATR mode."""
        if math.isnan(atr) or atr <= 0:
            return True          # no ATR yet -> no size opinion
        return (top - bot) >= atr * self.min_atr_x

    def _trim_side(self, blocks: list, bull: bool):
        """trimSide(): drop the OLDEST zone on this side past the cap."""
        idxs = [i for i, z in enumerate(blocks) if z.bull == bull]
        while len(idxs) > self.ob_max:
            blocks.pop(idxs[0])
            idxs = [i for i, z in enumerate(blocks) if z.bull == bull]

    def _ob_offset(self, o: np.ndarray, c: np.ndarray, i: int,
                   want_down: bool) -> int:
        """obOffset(): first opposing candle at 2..OB_LOOK+2 bars back."""
        for j in range(2, self.ob_look + 3):
            k = i - j
            if k < 0:
                break
            oj, cj = o[k], c[k]
            if math.isnan(oj) or math.isnan(cj):
                continue
            opposing = (cj < oj) if want_down else (cj > oj)
            if opposing:
                return j
        return -1

    def _is_pivot_high(self, high: np.ndarray, i: int) -> bool:
        """isPivotHigh() over the rolling window: centre BOS_LEN bars back."""
        b = self.bos_len
        if i < 2 * b:
            return False
        p = high[i - b]
        for j in range(0, 2 * b + 1):
            if j != b and high[i - j] > p:
                return False
        return True

    def _is_pivot_low(self, low: np.ndarray, i: int) -> bool:
        b = self.bos_len
        if i < 2 * b:
            return False
        p = low[i - b]
        for j in range(0, 2 * b + 1):
            if j != b and low[i - j] < p:
                return False
        return True

    # ------------------------------------------------------------------ #
    def run(self, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
            close: np.ndarray) -> EngineResult:
        o = np.asarray(open_, dtype=float)
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = len(c)

        atr = pine_atr(h, l, c, self.atr_len)
        rsi_arr = pine_rsi(c, RSI_LEN)

        blocks: list[_Block] = []
        sw_high = math.nan
        sw_low = math.nan

        # trend tags, identical rule to the Trader Pro badge
        last_swing_hi = math.nan
        last_swing_lo = math.nan
        last_hi_anc = None
        last_lo_anc = None
        struct_hi = ""
        struct_lo = ""

        entered_sup_today = False
        entered_res_today = False

        for i in range(n):
            a = atr[i] if i < len(atr) else math.nan

            # ---------- 1. mitigation: every bar, live included ----------
            survivors = []
            for z in blocks:
                filled = False
                if z.bull:
                    if l[i] <= z.bot:                 # traded fully through
                        filled = True
                    elif l[i] <= z.top:               # entered -> shrink
                        if not z.tapped:
                            z.tapped = True
                            if i == n - 1:
                                entered_sup_today = True
                        z.top = max(l[i], z.bot)
                else:
                    if h[i] >= z.top:
                        filled = True
                    elif h[i] >= z.bot:
                        if not z.tapped:
                            z.tapped = True
                            if i == n - 1:
                                entered_res_today = True
                        z.bot = min(h[i], z.top)
                if not filled and not self._size_ok(z.top, z.bot, a):
                    filled = True                     # shrank below minimum
                if not filled:
                    survivors.append(z)
            blocks = survivors

            # ---------- 2. detection: closed bars only ----------
            confirmed = self.detect_on_last or i < n - 1
            if confirmed and i >= self.ob_look + 4:
                # bullish: upward displacement + break of the last swing high
                if l[i] > h[i - 2] and not math.isnan(sw_high) and h[i] > sw_high:
                    off = self._ob_offset(o, c, i, want_down=True)
                    if off > 0:
                        k = i - off
                        top, bot = h[k], l[k]
                        if top > bot and self._size_ok(top, bot, a):
                            blocks.append(_Block(top, bot, True, k))
                            self._trim_side(blocks, True)
                # bearish: downward displacement + break of the last swing low
                if h[i] < l[i - 2] and not math.isnan(sw_low) and l[i] < sw_low:
                    off = self._ob_offset(o, c, i, want_down=False)
                    if off > 0:
                        k = i - off
                        top, bot = h[k], l[k]
                        if top > bot and self._size_ok(top, bot, a):
                            blocks.append(_Block(top, bot, False, k))
                            self._trim_side(blocks, False)

                # ---------- 3. structure rolls forward AFTER detection ----
                if self._is_pivot_high(h, i):
                    sw_high = h[i - self.bos_len]
                if self._is_pivot_low(l, i):
                    sw_low = l[i - self.bos_len]

            # ---------- 4. trend tags (Trader Pro rule, 5/1 pivots) ----------
            ph = pivot_high_at(h, i, PIV_LEFT, PIV_RIGHT)
            pl = pivot_low_at(l, i, PIV_LEFT, PIV_RIGHT)
            anc_t = i - PIV_RIGHT
            if ph is not None:
                dup = (last_hi_anc is not None and ph == last_swing_hi
                       and anc_t - last_hi_anc <= 1)
                if not dup:
                    struct_hi = ("H" if math.isnan(last_swing_hi)
                                 else "HH" if ph > last_swing_hi else "LH")
                    last_swing_hi = ph
                last_hi_anc = anc_t
            if pl is not None:
                dup = (last_lo_anc is not None and pl == last_swing_lo
                       and anc_t - last_lo_anc <= 1)
                if not dup:
                    struct_lo = ("L" if math.isnan(last_swing_lo)
                                 else "HL" if pl > last_swing_lo else "LL")
                    last_swing_lo = pl
                last_lo_anc = anc_t

        # ---------- 5. pick the two zones the scanner displays ----------
        px = c[-1] if n else math.nan
        support = resistance = None

        inside = [z for z in blocks if z.bot <= px <= z.top]
        if inside:
            # price is standing in a block — that one is the active level
            z = min(inside, key=lambda z: abs((z.top + z.bot) / 2 - px))
            zone = self._to_zone(z, n)
            if z.bull:
                support = zone
            else:
                resistance = zone

        below = [z for z in blocks if z.top < px]
        above = [z for z in blocks if z.bot > px]
        if support is None and below:
            support = self._to_zone(max(below, key=lambda z: z.top), n)
        if resistance is None and above:
            resistance = self._to_zone(min(above, key=lambda z: z.bot), n)

        # ---------- 6. trend badge ----------
        if not struct_hi or not struct_lo:
            structure = "Ranging"
        elif struct_hi == "HH" and struct_lo == "HL":
            structure = "Bullish"
        elif struct_hi == "LH" and struct_lo == "LL":
            structure = "Bearish"
        else:
            structure = "Ranging"

        # same K/D construction as the FibGP engine, so the CSV columns match
        k_arr = pine_sma(pine_stoch(c, h, l, STOCH_K), STOCH_K_SMOOTH)
        d_arr = pine_sma(k_arr, STOCH_D)
        stoch = StochState(k=float(k_arr[-1]) if n else math.nan,
                           d=float(d_arr[-1]) if n else math.nan)
        if n and not math.isnan(k_arr[-1]) and not math.isnan(d_arr[-1]):
            stoch.signal = "BUY" if k_arr[-1] >= d_arr[-1] else "SELL"

        return EngineResult(
            support=support,
            resistance=resistance,
            close=float(px) if n else math.nan,
            rsi=float(rsi_arr[-1]) if n else math.nan,
            stoch=stoch,
            sup_ema_tag="OB", sup_fvg=False,
            res_ema_tag="OB", res_fvg=False,
            struct_hi=struct_hi, struct_lo=struct_lo, structure=structure,
            entered_sup_today=entered_sup_today,
            entered_res_today=entered_res_today,
            n_bars=n,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_zone(z: _Block, n: int) -> Zone:
        """Map a live block onto the Zone shape the scanner renders.

        `hits` drives the star rating. A block that price has already tapped
        is worth less than an untouched one, and a block that has survived
        longer is worth more, so: fresh + old = 5 stars, tapped + new = fewer.
        """
        age = max(0, n - 1 - z.anchor)
        if age >= 120:
            quality = 5
        elif age >= 60:
            quality = 4
        elif age >= 25:
            quality = 3
        elif age >= 10:
            quality = 2
        else:
            quality = 1
        if z.tapped:
            quality = max(1, quality - 1)
        # Zone.stars is inverted (5 - hits), so store the complement
        return Zone(top=float(z.top), bot=float(z.bot), anchor_idx=z.anchor,
                    leg_lo=float(z.bot), leg_hi=float(z.top),
                    hits=5 - quality)
