# FibGP Scanner — PSX Golden Pocket Proximity Scanner (v1.0)

Companion web app to the **Fib Golden Pocket — Premium + Confluences (v11.5.3)**
TradingView indicator. Runs a faithful Python port of the v11.5.3 zone engine
over daily PSX data and buckets symbols into **Near/In Support** and
**Near/In Resistance** boards.

## Quick start
```
pip install -r requirements.txt
streamlit run app.py
```

## What "faithful port" means
`fibgp_engine.py` replicates the Pine engine bar-by-bar, in Pine's execution
order, including:

- Pivot detection (`ta.pivothigh/low`, left=5 right=5) with same-kind
  replacement and the 80-pivot cap
- Live (unconfirmed) endpoint tracking — `useLive` on by default, matching
  the chart default
- Candidate selection: **newest** valid leg, chronology guard, stale-leg
  rejection (`f_supStale`/`f_resStale`, 480-bar cap, including the Pine
  behavior of skipping the check when the anchor is older than 480 bars)
- Zone lifecycle: break on close beyond the far edge, pass-through
  invalidation (closed both above and below), anchor-change replacement
  with hit reset
- Entry counting → inverted strength stars (untapped = ★★★★★)
- Confluences at the last bar: EMA-in-zone (priority 200>100>50>34>8) and
  FVG-in-zone (3-candle, ≥50% unfilled, 50-bar lookback, exact loop bounds)
- RSI(7) Wilder, Stoch(5,3,3) persistent BUY/SELL signal + 3-point health
  score + near-cross ⚠, upside targets 1.4/1.7/2.4/3.4 with the
  above-resistance filter

Pine's `ta.ema`/`ta.rsi` seeding (SMA seed, NaN warm-up) is replicated so
EMA200 confluence matches the chart rather than pandas `ewm` defaults.

## Scanner semantics
- **One active zone per side per symbol** (as in Pine) — a symbol can appear
  on both boards if sandwiched.
- **Near** = close within the threshold (%) of the relevant zone edge:
  above the support top / below the resistance bottom. **In zone** = close
  inside the pocket (distance 0, ranked first).
- `NEW` pill = price entered the zone on the latest bar (the alert condition).
- Zone age = bars since the leg anchor.

## Data
EOD official history from `dps.psx.com.pk/historical` (same endpoint as the
psxdata package), fetched quietly with retries + threading, cached per
session. ~18 months default ≈ 370 daily bars (EMA200 warm-up + zone history).

## Files
- `app.py` — Streamlit UI (FibGP suite palette: teal #4DB6AC / rose #F47174)
- `fibgp_engine.py` — the v11.5.3 engine port + scanner classifier
- `psx_fetch.py` — PSX EOD fetcher
- `symbols.py` — KSE-100 + Quick-25 universes (editable; custom list in-app)
- `test_engine.py` — synthetic-data verification suite (`python test_engine.py`)

## Verification status
All 8 engine tests pass (geometry, break, stale rejection, entry counting,
classifier buckets, indicators). Recommended before trusting a live scan:
spot-check 2–3 symbols against the chart indicator on TradingView — pivot
ties on equal highs/lows are the one place TV's undocumented tie-breaking
could differ marginally.
