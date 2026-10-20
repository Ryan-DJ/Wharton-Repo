"""
phase_backtest.py - backtest each of the five phases ON ITS OWN, for exactly as long as that phase runs
in Laura's plan, starting from every December in the old data.

    Phase           Account     Runs for    In Laura's plan
    growth          Growth      4 years     Jan 2027 - Jan 2031
    pre_gift_y5     Funding     1 year      Jan 2031 - Jan 2032
    pre_gift        Funding     1 year      Jan 2032 - Jan 2033
    post_gift       Funding     10 years    Jan 2033 - Jan 2043
    reserve         Reserve     12 years    Jan 2031 - Jan 2043

For each start date the phase gets $100,000, picks its holdings with the Funnel (data up to that date only),
rebalances on its schedule, and is compared with (a) its own bear / base / bull targets and drawdown limit
and (b) the same mix held in index funds (SPY / EFA instead of Funnel stocks).

    python3 phase_backtest.py                 # all five phases
    python3 phase_backtest.py growth reserve  # just some
Results: phase_results.csv (one row per window) and phase_summary.csv.
"""
import sys, time
import numpy as np
import pandas as pd
from types import SimpleNamespace

from funnel_v2 import CASH, MarketData, max_drawdown, select
from laura_strategy import Plan, _asset_returns, decide, market_up

LENGTH = {"growth": 4, "pre_gift_y5": 1, "pre_gift": 1, "post_gift": 10, "reserve": 12}
FIRST_YEAR = {"growth": 1, "pre_gift_y5": 5, "pre_gift": 6, "post_gift": 7, "reserve": 5}   # year number in the plan
FIRST_START, LAST_DATA = "2004-12-31", "2026-09-30"


def passive(data, asof, cfg, universe):
    us = all(data.meta.loc[t, "country"] == "United States" for t in universe)
    return SimpleNamespace(weights={"SPY" if us else "EFA": 1.0}, status="PASSIVE INDEX", mc={})


def run_window(data, rets, plan, phase, start, selector, cache, cost_bps=25.0):
    years = LENGTH[phase]
    months = rets.index[rets.index >= pd.Timestamp(start)][: 12 * years + 1]
    if len(months) < 12 * years + 1:
        return None
    ph = plan.phases[phase]
    hold, nav, last, picks, last_up = {}, [100_000.0], -10 ** 9, [], None
    for k, d in enumerate(months[:-1]):
        flip = False
        if ph.trend_switch:                                   # v3 gate M1, checked monthly
            now = market_up(data, d, plan); flip = last_up is not None and now != last_up; last_up = now
        if k - last >= ph.rebalance_months or k == 0 or flip:
            dec = decide(data, d, plan, phase, FIRST_YEAR[phase] + k / 12, backtest_mode=True, selector=selector, cache=cache)
            v = sum(hold.values()) if hold else nav[-1]
            tgt = {t: v * x for t, x in dec["weights"].items()}
            traded = sum(abs(tgt.get(t, 0) - hold.get(t, 0)) for t in set(tgt) | set(hold)) / 2 if hold else v
            hold = {t: x * (1 - traded * cost_bps / 1e4 / v) for t, x in tgt.items()}
            last = k
            picks.append((d.date(), {t: round(x, 3) for t, x in sorted(dec["weights"].items(), key=lambda z: -z[1])[:6]}))
        row = rets.loc[months[k + 1]]
        hold = {t: x * (1 + (0.0 if pd.isna(row.get(t, np.nan)) else float(row[t]))) for t, x in hold.items()}
        nav.append(sum(hold.values()))
    nav = np.array(nav)
    r = nav[1:] / nav[:-1] - 1
    return dict(cagr=(nav[-1] / nav[0]) ** (1 / years) - 1, end_value=nav[-1], max_dd=max_drawdown(nav),
                worst_12m=float(pd.Series(1 + r).rolling(12).apply(np.prod, raw=True).min() - 1) if len(r) >= 12 else float(nav[-1] / nav[0] - 1),
                vol=float(r.std() * np.sqrt(12)), first_picks=picks[0][1])


def main(phases):
    data = MarketData("data"); rets = _asset_returns(data); plan = Plan()
    rows = []
    for phase in phases:
        ph, cache, cache_p = plan.phases[phase], {}, {}
        starts = [d for d in pd.date_range(FIRST_START, LAST_DATA, freq="YE")
                  if d + pd.DateOffset(years=LENGTH[phase]) <= pd.Timestamp(LAST_DATA) + pd.offsets.MonthEnd(0)]
        print(f"\n== {ph.name}: {LENGTH[phase]} year(s), {len(starts)} start dates "
              f"({starts[0].year + 1}-{starts[-1].year + 1}); targets {ph.bear:.1%} / {ph.target:.1%} / {ph.bull:.1%}; "
              f"drawdown limit {ph.max_dd:.0%}", flush=True)
        t0 = time.time()
        for s in starts:
            f = run_window(data, rets, plan, phase, s, select, cache)
            p = run_window(data, rets, plan, phase, s, passive, cache_p)
            if f is None:
                continue
            rows.append(dict(phase=phase, start=s.date(), years=LENGTH[phase], funnel_cagr=f["cagr"], index_cagr=p["cagr"],
                             funnel_max_dd=f["max_dd"], index_max_dd=p["max_dd"], funnel_worst_12m=f["worst_12m"],
                             index_worst_12m=p["worst_12m"], funnel_vol=f["vol"], index_vol=p["vol"],
                             hit_bear=f["cagr"] >= ph.bear, hit_base=f["cagr"] >= ph.target, hit_bull=f["cagr"] >= ph.bull,
                             dd_ok=f["max_dd"] >= ph.max_dd, beat_index=f["cagr"] > p["cagr"], first_picks=f["first_picks"]))
            x = rows[-1]
            print(f"  {s.year + 1}-{s.year + LENGTH[phase]}:  Funnel {x['funnel_cagr']:6.1%}/yr (worst drop {x['funnel_max_dd']:5.0%})   "
                  f"index mix {x['index_cagr']:6.1%}/yr ({x['index_max_dd']:5.0%})", flush=True)
        print(f"  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv("phase_results.csv" if len(phases) == 5 else f"phase_results_{'_'.join(phases)}.csv", index=False)
    summ = df.groupby("phase", sort=False).agg(windows=("start", "size"), funnel_median=("funnel_cagr", "median"),
             funnel_worst=("funnel_cagr", "min"), funnel_best=("funnel_cagr", "max"), index_median=("index_cagr", "median"),
             hit_bear=("hit_bear", "mean"), hit_base=("hit_base", "mean"), hit_bull=("hit_bull", "mean"),
             dd_within_limit=("dd_ok", "mean"), worst_drop=("funnel_max_dd", "min"), index_worst_drop=("index_max_dd", "min"),
             beat_index=("beat_index", "mean"))
    summ.to_csv("phase_summary.csv" if len(phases) == 5 else f"phase_summary_{'_'.join(phases)}.csv")
    with pd.option_context("display.float_format", "{:.1%}".format, "display.width", 200):
        print("\n", summ)


if __name__ == "__main__":
    main(sys.argv[1:] or list(LENGTH))
