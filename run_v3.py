"""Full 2027-2042 plan on the real S&P 500, v2 (current) vs v3 (revised growth + post-gift)."""
import sys, pickle, time
from funnel_v2 import MarketData
from laura_strategy import v3_phases, default_phases
from us_backtest import make_plan, load_cache, _lifecycle_cached, DATA
version, start = sys.argv[1], sys.argv[2]
data = MarketData(DATA); cache = load_cache()
plan = make_plan(phases=v3_phases() if version.startswith("v3") else default_phases())
t = time.time(); bt = _lifecycle_cached(data, start, plan, cache)
pickle.dump(bt, open(f"run_{version}_{start[:4]}.pkl", "wb"))
pickle.dump(cache, open(f"us_cache_{version}_{start[:4]}.pkl", "wb"))
sm = bt["summary"]
print(f"{version} {start}: 2031 ${sm['value_2031']:,.0f} | gift ${sm['gift']:,.0f} | paid {sm['payments_made']}/10 | "
      f"reserve left ${sm['reserve_left_2043']:,.0f} | funding left ${sm['funding_left_2043']:,.0f} | growth {sm['accounts']['growth']['cagr']:.1%}/yr "
      f"dd {sm['accounts']['growth']['max_dd']:.0%} | post-gift {sm['accounts'].get('funding',{}).get('cagr',float('nan')):.1%} [{time.time()-t:.0f}s]", flush=True)
