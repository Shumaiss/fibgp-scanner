"""Synthetic-data verification of the FibGP v11.5.3 Python port.

Paths avoid flat plateaus (equal-value bars create degenerate micro-pivots
that never occur in real OHLC data) and are shaped so the intended leg is
the newest valid pair — matching how the Pine engine selects zones.
"""
import numpy as np
from fibgp_engine import FibGPEngine, classify


def make_bars(path, spread=0.2):
    close = np.array(path, dtype=float)
    high = close + spread
    low = close - spread
    open_ = np.roll(close, 1); open_[0] = close[0]
    return open_, high, low, close


def ramp(a, b, n):
    return list(np.linspace(a, b, n))


def test_support_zone_geometry():
    """Rally 100->200, end mid-retracement: pocket = hi - 0.618/0.786 * rng."""
    path = ramp(106, 100, 8) + ramp(100, 200, 25) + ramp(200, 152, 14)
    o, h, l, c = make_bars(path)
    res = FibGPEngine().run(o, h, l, c)
    assert res.support is not None
    z = res.support
    assert abs(z.leg_hi - 200.2) < 0.5 and abs(z.leg_lo - 99.8) < 0.5, (z.leg_lo, z.leg_hi)
    rng = z.leg_hi - z.leg_lo
    assert abs(z.top - (z.leg_hi - 0.618 * rng)) < 1e-9
    assert abs(z.bot - (z.leg_hi - 0.786 * rng)) < 1e-9
    print(f"  support geometry OK: leg {z.leg_lo:.1f}->{z.leg_hi:.1f}, zone {z.bot:.2f}-{z.top:.2f}")


def test_resistance_zone_geometry():
    """Decline 200->100, end mid-bounce: pocket = lo + 0.618/0.786 * rng."""
    path = ramp(194, 200, 8) + ramp(200, 100, 25) + ramp(100, 128, 12)
    o, h, l, c = make_bars(path)
    res = FibGPEngine().run(o, h, l, c)
    assert res.resistance is not None
    z = res.resistance
    rng = z.leg_hi - z.leg_lo
    assert abs(z.bot - (z.leg_lo + 0.618 * rng)) < 1e-9
    assert abs(z.top - (z.leg_lo + 0.786 * rng)) < 1e-9
    print(f"  resistance geometry OK: leg {z.leg_hi:.1f}->{z.leg_lo:.1f}, zone {z.bot:.2f}-{z.top:.2f}")


def test_support_break_clears_zone():
    """Decisive close below the pocket bottom clears the old zone; it must
    never resurrect (stale rejection backs up the break logic)."""
    path = ramp(106, 100, 8) + ramp(100, 200, 25) + ramp(200, 95, 28) + ramp(95, 89, 10)
    o, h, l, c = make_bars(path)
    res = FibGPEngine().run(o, h, l, c)
    if res.support is not None:
        z = res.support
        assert not (120 < z.bot < 123 and 137 < z.top < 140), "broken zone resurrected"
        assert z.bot <= res.close + 1e-9
    print(f"  break handling OK (support: "
          f"{None if res.support is None else (round(res.support.bot,1), round(res.support.top,1))})")


def test_entry_counting_and_stars():
    """Discrete zone entries increment hits (stars inverted). Successive dips
    go slightly lower so no new pivot low confirms and the anchor stays put."""
    path = (ramp(106, 100, 8) + ramp(100, 200, 25)
            + ramp(200, 136, 16)                      # entry 1 (into 121.3-138.2)
            + [139.5, 139.6, 139.5]                   # exit above top
            + [135.5, 135.4]                          # entry 2
            + [139.5, 139.6, 139.5]                   # exit
            + [135.0, 134.9]                          # entry 3
            + [139.5, 139.4])                         # exit
    o, h, l, c = make_bars(path)
    res = FibGPEngine().run(o, h, l, c)
    assert res.support is not None, "support zone lost"
    z = res.support
    assert 120 < z.bot < 123 and 137 < z.top < 140, f"unexpected zone {z.bot}-{z.top}"
    assert z.hits == 3, f"expected 3 entries, got {z.hits}"
    assert z.star_count == 2 and z.stars == "★★☆☆☆"
    print(f"  entry counting OK: hits={z.hits}, stars={z.stars}")


def test_stale_rejection():
    """A pocket price already closed through must not resurrect. After the
    crash, the active zone must come from the NEW leg off the 110 low."""
    path = (ramp(106, 100, 8) + ramp(100, 200, 25)    # leg A: pocket 121.3-138.2
            + ramp(200, 110, 22) + ramp(110, 108, 8)  # closes below 121.3 -> stale
            + ramp(108, 145, 14))                     # recovery, end mid-rally
    o, h, l, c = make_bars(path)
    res = FibGPEngine().run(o, h, l, c)
    if res.support is not None:
        assert res.support.leg_lo < 115, f"stale leg-A zone resurrected: {res.support.leg_lo}"
    print(f"  stale rejection OK (support: "
          f"{None if res.support is None else (round(res.support.bot,1), round(res.support.top,1))})")


def test_classifier_buckets():
    """2% proximity honored on the correct zone edge; tighter threshold drops it."""
    path = ramp(106, 100, 8) + ramp(100, 200, 25) + ramp(200, 140, 14)
    o, h, l, c = make_bars(path)
    res = FibGPEngine().run(o, h, l, c)
    assert res.support is not None
    rows = classify("TEST", res, near_pct=2.0)
    sup = [r for r in rows if "SUPPORT" in r.status]
    assert len(sup) == 1 and sup[0].status == "NEAR_SUPPORT"
    assert 1.0 < sup[0].distance_pct < 1.7, sup[0].distance_pct
    assert not any("SUPPORT" in r.status for r in classify("TEST", res, near_pct=1.0))
    print(f"  classifier OK: dist={sup[0].distance_pct:.2f}% at 2% threshold")


def test_inside_zone_bucket():
    """Close inside the pocket -> IN_SUPPORT with distance 0."""
    path = ramp(106, 100, 8) + ramp(100, 200, 25) + ramp(200, 130, 16)
    o, h, l, c = make_bars(path)
    res = FibGPEngine().run(o, h, l, c)
    assert res.support is not None
    rows = classify("TEST", res, near_pct=2.0)
    sup = [r for r in rows if "SUPPORT" in r.status]
    assert len(sup) == 1 and sup[0].status == "IN_SUPPORT" and sup[0].distance_pct == 0.0
    print(f"  inside-zone OK: close={res.close:.1f} in {sup[0].zone_bot:.1f}-{sup[0].zone_top:.1f}")


def test_rsi_stoch_sane():
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 1.0, 300).cumsum() + 100
    o, h, l, c = make_bars(list(steps), spread=0.8)
    res = FibGPEngine().run(o, h, l, c)
    assert 0 <= res.rsi <= 100
    assert 0 <= res.stoch.k <= 100 and 0 <= res.stoch.d <= 100
    assert res.stoch.signal in ("BUY", "SELL")
    assert 0 <= res.stoch.score <= 3
    print(f"  indicators OK: RSI(7)={res.rsi:.1f}, K={res.stoch.k:.1f} D={res.stoch.d:.1f} "
          f"{res.stoch.signal} {res.stoch.dots}")


if __name__ == "__main__":
    for fn in [test_support_zone_geometry, test_resistance_zone_geometry,
               test_support_break_clears_zone, test_entry_counting_and_stars,
               test_stale_rejection, test_classifier_buckets,
               test_inside_zone_bucket, test_rsi_stoch_sane]:
        print(f"[{fn.__name__}]")
        fn()
    print("\nAll engine tests passed.")
