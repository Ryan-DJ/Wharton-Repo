"""
run_today.py - what each phase would buy today, choosing from the real S&P 500 (data_us/).

    python run_today.py                 # growth phase, latest month-end in the data
    python run_today.py post_gift       # any phase: growth, pre_gift_y5, pre_gift, post_gift, reserve
    python run_today.py growth 2015-12-31   # any past month-end (only data up to that date is used)

Note: data_us/ has no company statements, so gates A3-A5 (valuation, debt, earnings quality)
let every stock through; check those by hand before trading.
"""
import sys
from funnel_v2 import MarketData
from laura_strategy import Plan, decide, market_up
from build_us_universe import universe_fn

phase = sys.argv[1] if len(sys.argv) > 1 else "growth"
data = MarketData("data_us")
asof = sys.argv[2] if len(sys.argv) > 2 else str(data.price_usd.dropna(how="all").index[-1].date())
year = {"growth": 1, "pre_gift_y5": 5, "pre_gift": 6, "post_gift": 7, "reserve": 5}[phase]
plan = Plan(universe_at=universe_fn("data_us"), foreign_index_only=True)

print(f"Phase: {phase}   Date: {asof}   Market above 10-month average: {market_up(data, asof, plan)}")
dec = decide(data, asof, plan, phase, year, backtest_mode=True)
if "bonds_chosen" in dec["info"]:
    print(f"Bond ladder chose {dec['info']['bonds_chosen']:.0%} bonds")
for sleeve, i in dec["info"]["sleeves"].items():
    p = "" if i["p_hit"] is None else f" (chance of hitting target: {i['p_hit']:.0%})"
    print(f"  {sleeve} stocks: {i['status']}{p}")
print("\nHoldings:")
for t, w in sorted(dec["weights"].items(), key=lambda x: -x[1]):
    label = {"IEF": "Treasury bonds (IEF)", "EFA": "International index fund (EFA)", "SPY": "S&P 500 index fund (SPY)",
             "CASH": "T-bills"}.get(t, t)
    print(f"  {w:6.1%}  {label}")
