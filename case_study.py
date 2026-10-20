"""Replay one real v3 decision from the backtest (growth phase, 31 Dec 2003, start-2002 run) gate by gate."""
import json
from dataclasses import replace
import numpy as np, pandas as pd
from funnel_v2 import CASH, MarketData, select, max_drawdown
from laura_strategy import Plan, decide, market_up, sleeve_configs, _asset_returns
from build_us_universe import universe_fn
ASOF, YEAR = pd.Timestamp("2003-12-31"), 3.0
data = MarketData("data_us"); plan = Plan(universe_at=universe_fn("data_us"), foreign_index_only=True)
up = market_up(data, ASOF, plan)
dec = decide(data, ASOF, plan, "growth", YEAR, backtest_mode=True)
ph = plan.phases["growth"]; rf = float(data.tbill.loc[:ASOF].dropna().iloc[-1])
# bond ladder: what each bond share asked of the stocks, and whether it hit with room
ladder = []
eq0 = ph.mix["domestic"] + ph.mix["foreign"]
for b in sorted(ph.bond_choices):
    eq = 1 - ph.mix["cash"] - b
    mix = dict(cash=ph.mix["cash"], bonds=b, domestic=ph.mix["domestic"] / eq0 * eq, foreign=ph.mix["foreign"] / eq0 * eq)
    cfg, eq_t = sleeve_configs(plan, "growth", YEAR, rf, True, None, replace(ph, mix=mix))
    c = replace(cfg, name_cap=min(1.0, cfg.name_cap / mix["domestic"]), sector_cap=min(1.0, cfg.sector_cap / eq), country_cap=1.0)
    uni = [t for t in plan.universe_at(ASOF) if t in data.price_usd.columns]
    res = select(data, ASOF, c, uni)
    ladder.append(dict(bonds=b, stock_target=eq_t, status=res.status, p_hit=res.mc.get("p_hit"), res=res, mix=mix, cfg=c))
chosen = next(l for l in ladder if l["bonds"] == dec["info"].get("bonds_chosen", 0.0))
res, c = chosen["res"], chosen["cfg"]
T = res.table
held = sorted([t for t in res.weights if t != CASH], key=lambda t: -res.weights[t])
# A7 bar actually used: top third of the stocks alive at A7, floor 0.50
a7_in = set(T.index)
for g in res.gates:
    if g["gate"][:2] in ("A1", "A2", "A3", "A4", "A5", "A6"):
        a7_in -= {t for t, _ in g["cut"]}
psr_bar = max(c.psr_floor, float(np.quantile(T.loc[list(a7_in), "psr"], 1 - c.psr_top_share)))
B = data.bench.loc[:ASOF].dropna().iloc[-c.lookback_months - 1:]
mkt_dd = max_drawdown(B.values)
ranked = T.dropna(subset=["score"]).sort_values("score", ascending=False)
names = {"SPG": "Simon Property Group", "UNH": "UnitedHealth Group", "SYY": "Sysco", "PAYX": "Paychex", "WEN": "Wendy's",
         "MCO": "Moody's", "AZO": "AutoZone", "CCU": "Clear Channel"}
out = dict(asof=str(ASOF.date()), market_up=up, rf=rf, universe=len([t for t in plan.universe_at(ASOF)]),
           priced=len([t for t in plan.universe_at(ASOF) if t in data.price_usd.columns]),
           ladder=[dict(bonds=l["bonds"], stock_target=l["stock_target"], status=l["status"], p_hit=l["p_hit"]) for l in ladder],
           bonds_chosen=chosen["bonds"], mix=chosen["mix"], target=c.target_cagr, horizon=c.horizon_years, name_cap=c.name_cap,
           sector_cap=c.sector_cap, psr_bar=psr_bar, market_dd=mkt_dd,
           gates=[dict(gate=g["gate"], n_in=g["n_in"], n_out=g["n_out"], cuts=[list(x) for x in g["cut"][:5]],
                       info={k: v for k, v in (g.get("info") or {}).items() if k != "grid"}) for g in res.gates],
           mc={k: v for k, v in res.mc.items() if k != "cagr_samples"}, status=res.status,
           sleeve_weights=res.weights, account_weights=dec["weights"], held=held,
           detail={t: {k: (None if (k not in T.columns or pd.isna(T.loc[t, k])) else float(T.loc[t, k]) if k not in ("sector", "country", "above_10m") else str(T.loc[t, k]))
                       for k in ["sector", "psr", "sortino", "cvar95", "max_dd", "beta", "vol", "mom_12_1", "above_10m", "hist_mean",
                                 "mu_hist", "mu_fund", "mu_capm", "js_shrink", "exp_return", "p_beat_cash", "score"]} for t in held},
           rank={t: int(i + 1) for i, t in enumerate(ranked.index)}, n_ranked=len(ranked), names=names,
           universe_psr_median=float(T.loc[list(a7_in), "psr"].median()))
# what happened next: each held stock's price over the following 3 months (next rebalance)
r = _asset_returns(data)
out["next_3m"] = {t: float((1 + r[t].loc["2004-01-31":"2004-03-31"].fillna(0)).prod() - 1) for t in held}
out["next_3m_spy"] = float((1 + r["SPY"].loc["2004-01-31":"2004-03-31"]).prod() - 1)
json.dump(out, open("case_study.json", "w"), indent=1, default=str)
print("match backtest:", {t: round(x, 3) for t, x in sorted(dec["weights"].items(), key=lambda z: -z[1])[:6]})
print("bonds", chosen["bonds"], "ladder", [(l["bonds"], round(l["stock_target"], 3), l["status"][:12], l["p_hit"]) for l in ladder])
print("M1", up, "psr_bar", round(psr_bar, 3), "mkt_dd", round(mkt_dd, 3))
for g in out["gates"]: print(f"{g['gate']:<34}{g['n_in']:>4} -> {g['n_out']:<4}", g["cuts"][:2])
for t in held[:4]: print(t, out["rank"].get(t), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["detail"][t].items()}, round(res.weights[t], 3))
