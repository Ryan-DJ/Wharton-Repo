"""Adds ranking pillars, contrast stocks and C2 info to case_study.json (run after case_study.py)."""
import json, numpy as np, pandas as pd
from funnel_v2 import pct_rank
from case_study import res, T, c
cs = json.load(open('case_study.json'))
cand = T.dropna(subset=['score'])
col = lambda k: cand[k].astype(float).values
q = pct_rank(np.full(len(cand), np.nan))
P = pd.DataFrame({"momentum": pct_rank(col("mom_12_1")),
                  "risk": (pct_rank(col("psr")) + pct_rank(col("sortino")) + pct_rank(col("cvar95"))) / 3,
                  "exp_return": pct_rank(col("exp_return")), "quality": q, "value": q, "growth": q}, index=cand.index)
cs['pillars'] = {t: P.loc[t].to_dict() for t in cs['held'] if t in P.index}
cs['pillar_weights'] = c.pillar_weights
cs['contrast'] = {t: {k: float(T.loc[t, k]) for k in ['psr', 'sortino', 'cvar95', 'max_dd', 'beta', 'mom_12_1']} for t in ['AAPL', 'ADBE', 'MSFT', 'GE'] if t in T.index}
cs['cut_reason'] = {t: w for g in res.gates for t, w in g['cut'] if t in cs['contrast']}
cs['c2'] = {k: float(v) for k, v in next(g for g in res.gates if g['gate'].startswith('C2'))['info'].items() if k != 'grid'}
json.dump(cs, open('case_study.json', 'w'), indent=1, default=str)
print('enriched', list(cs['pillars'])[:4])
