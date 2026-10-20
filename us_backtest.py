"""
us_backtest.py - the real-market backtest: the algorithm is dropped into each start date and may only buy
stocks that were in the S&P 500 ON THAT DATE (members list since 1996, incl. companies that later failed
or were bought, where Yahoo still has their prices). The foreign 20% holds an international index fund,
because no free list of past foreign index members exists.

    python3 us_backtest.py phases growth pre_gift_y5      # each phase on its own, every start year
    python3 us_backtest.py lifecycle                        # whole 2027-2042 plan, every start year 1996-2009
Decisions are cached in us_cache_*.pkl so re-runs and the lifecycle replays reuse them.
"""
import glob, os, pickle, sys, time
import numpy as np
import pandas as pd

from funnel_v2 import MarketData, select
from laura_strategy import Plan, _asset_returns, lifecycle_backtest
from build_us_universe import universe_fn
from phase_backtest import LENGTH, passive, run_window

DATA, FIRST, LAST = "data_us", "1996-12-31", "2026-09-30"


def make_plan(**kw):
    return Plan(universe_at=universe_fn(DATA), foreign_index_only=True, **kw)


def load_cache():
    c = {}
    for f in glob.glob("us_cache_*.pkl"):
        try:
            c.update(pickle.load(open(f, "rb")))
        except Exception:
            pass
    return c


def phases(names):
    data = MarketData(DATA); rets = _asset_returns(data); plan = make_plan()
    cache, tag = load_cache(), "_".join(names)
    rows = []
    for ph in names:
        p = plan.phases[ph]
        starts = [d for d in pd.date_range(FIRST, LAST, freq="YE") if d + pd.DateOffset(years=LENGTH[ph]) <= pd.Timestamp(LAST)]
        print(f"\n== {p.name}: {LENGTH[ph]} yr windows, {len(starts)} starts ({starts[0].year+1}-{starts[-1].year+1})", flush=True)
        t0 = time.time()
        for s in starts:
            f = run_window(data, rets, plan, ph, s, select, cache)
            q = run_window(data, rets, plan, ph, s, passive, {})
            spy = rets["SPY"].loc[s:].iloc[1:12 * LENGTH[ph] + 1]
            rows.append(dict(phase=ph, start=s.date(), first_year=s.year + 1, years=LENGTH[ph],
                             funnel_cagr=f["cagr"], index_cagr=q["cagr"], sp500_cagr=(1 + spy).prod() ** (1 / LENGTH[ph]) - 1,
                             funnel_max_dd=f["max_dd"], index_max_dd=q["max_dd"], funnel_worst_12m=f["worst_12m"],
                             funnel_vol=f["vol"], first_picks=f["first_picks"]))
            x = rows[-1]
            print(f"  {x['first_year']}-{x['first_year']+LENGTH[ph]-1}: Funnel {x['funnel_cagr']:6.1%}/yr (drop {x['funnel_max_dd']:4.0%})  "
                  f"same mix in index funds {x['index_cagr']:6.1%}  S&P 500 {x['sp500_cagr']:6.1%}   [{time.time()-t0:.0f}s]", flush=True)
            pickle.dump(cache, open(f"us_cache_{tag}.pkl", "wb"))
        pd.DataFrame(rows).to_csv(f"us_phase_results_{tag}.csv", index=False)


def lifecycle(starts=None):
    data = MarketData(DATA); cache = load_cache()
    starts = starts or [f"{y}-12-31" for y in range(1996, 2010)]
    out = {}
    for s in starts:
        for mode in ("funnel", "passive"):
            t0 = time.time()
            if mode == "funnel":
                import laura_strategy as ls
                bt = _lifecycle_cached(data, s, make_plan(), cache)
            else:
                bt = lifecycle_backtest(data, s, make_plan(), selector=passive, verbose=False)
            sm = bt["summary"]
            out[(s, mode)] = dict(summary=sm, series=bt["series"], events=bt["events"])
            print(f"{s} {mode:8s} 2031 ${sm['value_2031']:>9,.0f}  gift ${sm['gift']:>9,.0f}  paid {sm['payments_made']}/10  "
                  f"reserve left ${sm['reserve_left_2043']:>9,.0f}  funding left ${sm['funding_left_2043']:>9,.0f}  [{time.time()-t0:.0f}s]", flush=True)
        tag = f"{starts[0][:4]}_{starts[-1][:4]}"
        pickle.dump(out, open(f"us_lifecycle_{tag}.pkl", "wb"))


def _lifecycle_cached(data, start, plan, cache):
    """lifecycle_backtest with the shared decision cache (same decisions as the phase runs)."""
    import laura_strategy as ls
    orig = ls.decide
    def dec(*a, **k):
        k["cache"] = cache
        return orig(*a, **k)
    ls.decide = dec
    try:
        return ls.lifecycle_backtest(data, start, plan, verbose=False)
    finally:
        ls.decide = orig


if __name__ == "__main__":
    if sys.argv[1] == "phases":
        phases(sys.argv[2:] or list(LENGTH))
    else:
        lifecycle(sys.argv[2:] or None)
