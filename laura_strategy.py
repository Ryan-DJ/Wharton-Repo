"""
laura_strategy.py - Laura Gao's full plan as code (Part 2 of the strategy).

The plan has three accounts, each run by the Funnel (funnel_v2.py) with settings for its phase:

    Year (2026 = 0)      Growth account        Withdrawal account (reserve)      Funding account (facility gift)
    1-4  (2027-2030)     15%/yr, risk-on       -                                 -
    5    (2031)          split ->              $450k carved out, 3%/yr, safe     the rest, 5%/yr, pre-gift (calm)
    6    (2032)                                                                   5%/yr, more bonds
    7    (2033)                                first $50k payment                 75% given as the facility gift
    8-16 (2034-2042)                           $50k a year, ten in total          what's left, 18.5%/yr, risk-on

Asset mixes and return targets come from the team's spreadsheet (base case). Inside each account,
"Domestic equities" and "Foreign equities" are picked by the Funnel (U.S. universe and non-U.S. universe),
"Bonds" are U.S. Treasuries (IEF), "Cash" is T-bills. The equity sleeves get the return they must earn for
the whole account to hit its target:

    equity target = (account target - cash% x T-bill - bonds% x bond yield) / equity%

Run the live plan:        python3 laura_strategy.py
Run a lifecycle backtest: python3 laura_strategy.py backtest 2007-12-31
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from funnel_v2 import CASH, Config, MarketData, max_drawdown, select

HERE = os.path.dirname(os.path.abspath(__file__))

# Stricter stock gates for the two "security" phases (pre-gift funding account, operating reserve)
TIGHT_GATES = dict(
    max_beta=1.20, min_cvar95_monthly=-0.12, min_max_drawdown=-0.40, min_psr=0.85, min_sortino=0.50,
    pillar_weights={"quality": 0.25, "value": 0.15, "growth": 0.10, "momentum": 0.10, "risk": 0.25, "exp_return": 0.15})


# =============================================================================== plan definition
@dataclass
class Phase:
    name: str
    target: float                     # account return the spreadsheet assumes (base case)
    bear: float
    bull: float
    mix: dict                         # shares of the account: cash, bonds, domestic, foreign (sum to 1)
    confidence: float                 # required P(account hits target)
    max_dd: float                     # worst account drawdown we accept ...
    max_dd_prob: float                # ... in at most this share of simulations
    horizon: object = "phase_end"     # years the target is measured over: int, or "phase_end" (years left)
    gates: dict = field(default_factory=dict)   # Config overrides for stock gates
    bond_proxy: str = "IEF"
    allow_cash: object = "auto"
    rebalance_months: int = 6
    bond_choices: tuple = ()          # v3: bond shares to try, fewest first; more bonds only while the target is
    bond_margin: float = 0.10         #     still hit with this much room (P(hit) >= confidence + margin)
    trend_switch: bool = False        # v3 gate M1: a sleeve goes to T-bills while its market is below its 10-month average


def default_phases() -> dict:
    return {
        # 2027-2030: the whole portfolio grows (sheet rows 3-6)
        "growth": Phase("Growth", 0.15, 0.13, 0.17, dict(cash=0.0, bonds=0.30, domestic=0.50, foreign=0.20),
                        confidence=0.50, max_dd=-0.40, max_dd_prob=0.35),
        # 2031-2042: operating reserve, ten $50k payments with high certainty (sheet rows 10-21)
        "reserve": Phase("Operating reserve", 0.03, 0.02, 0.04, dict(cash=0.0, bonds=0.30, domestic=0.50, foreign=0.20),
                         confidence=0.95, max_dd=-0.05, max_dd_prob=0.05, horizon=2, gates=TIGHT_GATES,
                         allow_cash=True, rebalance_months=12),
        # 2031: facility (funding) account right after the split (sheet row 25)
        "pre_gift_y5": Phase("Pre-gift (2031)", 0.05, 0.03, 0.07, dict(cash=0.10, bonds=0.25, domestic=0.475, foreign=0.175),
                             confidence=0.50, max_dd=-0.15, max_dd_prob=0.10, gates=TIGHT_GATES, allow_cash=True),
        # 2032: funding account, one year before the gift (sheet row 26)
        "pre_gift": Phase("Pre-gift (2032)", 0.05, 0.03, 0.07, dict(cash=0.05, bonds=0.35, domestic=0.45, foreign=0.15),
                          confidence=0.50, max_dd=-0.15, max_dd_prob=0.10, gates=TIGHT_GATES, allow_cash=True),
        # 2034-2042: what is left after the gift (sheet rows 28-36)
        "post_gift": Phase("Post-gift", 0.185, 0.15, 0.21, dict(cash=0.05, bonds=0.35, domestic=0.45, foreign=0.15),
                           confidence=0.50, max_dd=-0.45, max_dd_prob=0.40),
    }


V3_GATES = dict(psr_mode="rank", psr_top_share=1 / 3, psr_floor=0.50, min_sortino=0.0,
                drawdown_mode="relative", max_dd_vs_market=-0.25,
                pillar_weights={"quality": 0.20, "value": 0.15, "growth": 0.15, "momentum": 0.25, "risk": 0.10, "exp_return": 0.15})


def v3_phases() -> dict:
    """v3: growth and post-gift revised after the real-market backtest; reserve and pre-gift unchanged."""
    ph = default_phases()
    # allow_cash=False: no T-bills inside the stock sleeves; only gate M1 moves stocks to cash
    ph["growth"] = replace(ph["growth"], bond_choices=(0.0, 0.10, 0.20, 0.30), trend_switch=True,
                           gates=V3_GATES, rebalance_months=3, allow_cash=False)
    ph["post_gift"] = replace(ph["post_gift"], target=0.09, bear=0.07, bull=0.11, allow_cash=False,
                              bond_choices=(0.0, 0.10, 0.20, 0.35), trend_switch=True, gates=V3_GATES, rebalance_months=3)
    return ph


@dataclass
class Plan:
    contributions: dict = field(default_factory=lambda: {1: 300_000.0, 2: 150_000.0})
    reserve_year: int = 5              # 2031: carve out the operating reserve
    reserve_amount: float = 450_000.0
    reserve_rule: str = "fixed"        # "fixed": the sheet's $450k. "pv": price the ten payments at the bond yield
    reserve_safety: float = 0.01       #   on the split date, minus this safety margin (floored at 0) ...
    reserve_buffer: float = 0.02       #   ... plus this cushion. Never less than reserve_amount if "pv_floor".
    gift_year: int = 7                 # 2033: facility contribution
    gift_share: float = 0.75           # share of the funding account given
    payment: float = 50_000.0
    payment_years: tuple = tuple(range(7, 17))   # 2033-2042
    end_year: int = 17                 # start of 2043: everything has been paid
    bond_term_premium: float = 0.005   # bond yield assumed = T-bill + this
    cost_bps: float = 25.0
    domestic_country: str = "United States"
    # if the Funnel can't fill an equity sleeve (too few stocks pass), a return-seeking phase holds that
    # market's index fund instead of drifting into cash; security phases (allow_cash=True) use T-bills
    index_fallback: dict = field(default_factory=lambda: {"domestic": "SPY", "foreign": "EFA"})
    phases: dict = field(default_factory=lambda: v3_phases())   # v3 (final). v2 = Plan(phases=default_phases())
    # point-in-time universe: function(asof) -> list of tickers that were really in the market on that date
    # (e.g. the S&P 500 members that day). None = every stock in the data folder.
    universe_at: object = None
    foreign_index_only: bool = False   # backtests without historical foreign membership: hold the foreign index fund

    def phase_for(self, account: str, year: int) -> str | None:
        if account == "growth":
            return "growth" if year < self.reserve_year else None
        if account == "reserve":
            return "reserve" if self.reserve_year <= year < self.end_year else None
        if account == "funding":
            if year == self.reserve_year:
                return "pre_gift_y5"
            if self.reserve_year < year < self.gift_year:
                return "pre_gift"
            if self.gift_year <= year < self.end_year:
                return "post_gift"
        return None

    def reserve_needed(self, rf: float) -> float:
        """Dollar reserve to carve out at the split, given the T-bill rate on that date."""
        if self.reserve_rule == "fixed":
            return self.reserve_amount
        r = max(0.0, rf + self.bond_term_premium - self.reserve_safety)
        pv = sum(self.payment / (1 + r) ** (y - self.reserve_year) for y in self.payment_years) * (1 + self.reserve_buffer)
        return max(pv, self.reserve_amount) if self.reserve_rule == "pv_floor" else pv

    def phase_end(self, phase: str) -> int:
        return {"growth": self.reserve_year, "pre_gift_y5": self.gift_year, "pre_gift": self.gift_year,
                "post_gift": self.end_year, "reserve": self.end_year}[phase]


# =============================================================================== deterministic sheet projection
def project(plan: Plan | None = None, scenario: str = "base") -> dict:
    """Re-creates the team's spreadsheet with fixed returns (values in $). Used to check the plan's arithmetic."""
    plan = plan or Plan()
    P = plan.phases
    r = lambda ph: getattr(P[ph], "target" if scenario == "base" else scenario)
    v, rows = 0.0, {}
    for y in range(1, plan.reserve_year):
        v = v * (1 + r("growth")) + plan.contributions.get(y, 0.0) if y > 1 else plan.contributions.get(1, 0.0)
        rows[y] = dict(growth=v)
    total = v * (1 + r("growth"))
    res, fund = min(plan.reserve_amount, total), max(0.0, total - plan.reserve_amount)
    rows[plan.reserve_year] = dict(total=total, reserve=res, funding=fund)
    gift = None
    for y in range(plan.reserve_year + 1, plan.end_year):
        res = res * (1 + r("reserve")) - (plan.payment if y in plan.payment_years else 0.0)
        fund_ph = plan.phase_for("funding", y - 1)
        fund = fund * (1 + r(fund_ph))                      # return of the phase the year started in
        if y == plan.gift_year:
            gift = plan.gift_share * fund
            rows[y] = dict(funding_before_gift=fund)
            fund -= gift
        rows.setdefault(y, {}).update(reserve=res, funding=fund)
    return dict(rows=rows, value_2031=total, gift=gift, reserve_end=res, funding_end=fund)


# =============================================================================== market trend (gate M1)
_FUNDS = {}


def _funds(data):
    f = os.path.join(data_folder(data), "funds.csv")
    if f not in _FUNDS:
        _FUNDS[f] = pd.read_csv(f, index_col=0, parse_dates=True)
    return _FUNDS[f]


def market_up(data, asof, plan) -> dict:
    """Gate M1: is each sleeve's market above its 10-month average at `asof`? (month-end prices up to asof only)"""
    f = _funds(data).loc[:pd.Timestamp(asof)]
    out = {}
    for sleeve, sym in plan.index_fallback.items():
        p = f[sym].dropna()
        out[sleeve] = True if len(p) < 10 else bool(p.iloc[-1] >= p.iloc[-10:].mean())
    return out


# =============================================================================== one rebalance decision
def sleeve_configs(plan: Plan, phase: str, year: float, rf: float, backtest_mode: bool, sim: dict | None = None, ph=None):
    """Config for the domestic and foreign equity sleeves of one phase, at one date."""
    ph = ph or plan.phases[phase]
    eq = ph.mix["domestic"] + ph.mix["foreign"]
    if eq <= 0:
        return None, 0.0
    bond_y = rf + plan.bond_term_premium
    fixed = ph.mix["cash"] * rf + ph.mix["bonds"] * bond_y
    to_eq = lambda t: (t - fixed) / eq
    if ph.horizon == "phase_end":
        h = max(1, int(math.ceil(plan.phase_end(phase) - year)))
    else:
        h = int(ph.horizon)
    kw = dict(target_cagr=to_eq(ph.target), horizon_years=h, confidence=ph.confidence,
              scenario_targets={"bear": to_eq(ph.bear), "base": to_eq(ph.target), "bull": to_eq(ph.bull)},
              max_drawdown_tolerance=max(-0.95, ph.max_dd / eq), max_drawdown_prob=ph.max_dd_prob,
              allow_cash=ph.allow_cash, **ph.gates)
    kw.update(sim or {})
    cfg = Config.for_backtest(**kw) if backtest_mode else Config(**kw)
    return cfg, to_eq(ph.target)


def decide(data: MarketData, asof, plan: Plan, phase: str, year: float, backtest_mode: bool = True,
           sim: dict | None = None, selector=select, cache: dict | None = None) -> dict:
    """Target weights for a whole account (keys: tickers, CASH, bond proxy) plus the Funnel results.
    v3 phases with bond_choices try each bond share (safest first) and keep the first that hits the target."""
    ph = plan.phases[phase]
    up = market_up(data, asof, plan) if ph.trend_switch else {}
    if not ph.bond_choices:
        return _decide_mix(data, asof, plan, phase, ph, year, backtest_mode, sim, selector, cache, up)
    eq0 = ph.mix["domestic"] + ph.mix["foreign"]
    if up and not any(up.values()):                  # every market switched off: stocks to T-bills, planned bond share
        return _decide_mix(data, asof, plan, phase, ph, year, backtest_mode, sim, selector, cache, up)
    best = None
    for b in sorted(ph.bond_choices):                # fewest bonds (most stocks) first
        eq = 1 - ph.mix["cash"] - b
        mix = dict(cash=ph.mix["cash"], bonds=b, domestic=ph.mix["domestic"] / eq0 * eq, foreign=ph.mix["foreign"] / eq0 * eq)
        d = _decide_mix(data, asof, plan, phase, replace(ph, mix=mix), year, backtest_mode, sim, selector, cache, up)
        d["info"]["bonds_chosen"] = b
        if best is None:
            best = d                                 # the all-stock-as-possible mix is the default
            continue
        sl = [i for i in d["info"]["sleeves"].values() if i["p_hit"] is not None]
        if sl and all(i["status"].startswith("TARGET MET") and i["p_hit"] >= ph.confidence + ph.bond_margin for i in sl):
            best = d                                 # more bonds still hits the target with room to spare
        else:
            break
    return best


def _decide_mix(data, asof, plan, phase, ph, year, backtest_mode, sim, selector, cache, up):
    asof = pd.Timestamp(asof)
    rf = float(data.tbill.loc[:asof].dropna().iloc[-1])
    w = {CASH: ph.mix["cash"], ph.bond_proxy: ph.mix["bonds"]}
    cfg, eq_target = sleeve_configs(plan, phase, year, rf, backtest_mode, sim, ph)
    info = dict(phase=phase, asof=asof, equity_target=eq_target, sleeves={})
    if cfg is not None:
        countries = data.meta["country"]
        pool = plan.universe_at(asof) if plan.universe_at is not None else list(data.price_usd.columns)
        pool = [t for t in pool if t in data.price_usd.columns]
        dom_u = [t for t in pool if countries.get(t) == plan.domestic_country]
        for_u = [] if plan.foreign_index_only else [t for t in pool if countries.get(t) != plan.domestic_country]
        eq = ph.mix["domestic"] + ph.mix["foreign"]
        for sleeve, uni, share in (("domestic", dom_u, ph.mix["domestic"]), ("foreign", for_u, ph.mix["foreign"])):
            if share <= 0:
                continue
            if up and not up.get(sleeve, True):          # gate M1: this market is in a downtrend -> T-bills
                w[CASH] = w.get(CASH, 0.0) + share
                info["sleeves"][sleeve] = dict(status="M1 MARKET BELOW 10-MONTH AVERAGE -> T-bills", weights={CASH: 1.0}, p_hit=None)
                continue
            # account limits -> sleeve limits: a stock may be 15% of the ACCOUNT (= 15%/share of its sleeve);
            # a sector 35% of the account's equities. Domestic is one country by definition; the foreign
            # sleeve keeps the 40%-per-country cap so it stays spread across at least three markets.
            c = replace(cfg, name_cap=min(1.0, cfg.name_cap / share), sector_cap=min(1.0, cfg.sector_cap / eq),
                        country_cap=1.0 if sleeve == "domestic" else cfg.country_cap)
            key = (asof, sleeve, repr(c), hash(tuple(uni)), getattr(selector, "__name__", "sel"))
            if cache is not None and key in cache:
                res = cache[key]
            else:
                res = selector(data, asof, c, uni) if uni else None
                if res is not None and cache is not None:      # keep only what later steps need (small, picklable)
                    res = SimpleNamespace(weights=dict(res.weights), status=res.status, mc={"p_hit": (res.mc or {}).get("p_hit")})
                if cache is not None:
                    cache[key] = res
            unfilled = res is None or not res.weights or res.status.startswith(("NO STOCKS", "NOT ENOUGH", "SOLVER"))
            if unfilled:
                fb = plan.index_fallback.get(sleeve) if ph.allow_cash is not True else None
                sw = {fb: 1.0} if fb else {CASH: 1.0}
            else:
                sw = dict(res.weights)
            for t, x in sw.items():
                w[t] = w.get(t, 0.0) + share * x
            info["sleeves"][sleeve] = dict(status=(res.status if res else "EMPTY UNIVERSE") + (" -> index fund" if unfilled and CASH not in sw else ""), weights=sw,
                                           p_hit=(res.mc or {}).get("p_hit") if res else None)
    w = {t: x for t, x in w.items() if x > 1e-9}
    s = sum(w.values())
    return dict(weights={t: x / s for t, x in w.items()}, info=info)


def plan_today(data: MarketData, asof, plan: Plan | None = None, backtest_mode: bool = False) -> dict:
    """What every phase would hold if it started today (live picks for the IPS / trading notes)."""
    plan = plan or Plan()
    start = {"growth": 1, "pre_gift_y5": 5, "pre_gift": 6, "reserve": 5, "post_gift": 7}
    return {ph: decide(data, asof, plan, ph, start[ph], backtest_mode=backtest_mode) for ph in plan.phases}


# =============================================================================== lifecycle backtest
def _asset_returns(data: MarketData):
    px = data.price_usd.copy()
    bonds = pd.read_csv(os.path.join(data_folder(data), "funds.csv"), index_col=0, parse_dates=True)
    px = px.join(bonds, how="left")
    rets = px.pct_change()
    rets[CASH] = data.tbill.reindex(rets.index).ffill().shift(1) / 12     # T-bill rate known at start of month
    return rets


def data_folder(data: MarketData) -> str:
    return getattr(data, "folder", os.path.join(HERE, "data"))


def lifecycle_backtest(data: MarketData, start, plan: Plan | None = None, sim: dict | None = None,
                       selector=select, verbose: bool = True) -> dict:
    """Run Laura's whole timeline on history. `start` is the month-end that plays the role of Jan 2027
    (year 1). Year k begins 12*(k-1) months later. Every decision uses only data up to its date."""
    plan = plan or Plan()
    rets = _asset_returns(data)
    months = rets.index[rets.index >= pd.Timestamp(start)]
    n = 12 * (plan.end_year - 1)
    if len(months) <= n:
        raise ValueError(f"need {n} months of history after {start}; only {len(months) - 1} available")
    months = months[: n + 1]
    hold = {a: {} for a in ("growth", "reserve", "funding")}
    val = lambda a: sum(hold[a].values())
    events, decisions, series, cache = [], [], [], {}
    last_reb = {a: -10 ** 9 for a in hold}
    last_up = {a: None for a in hold}

    def move_out(a, amount):
        v = val(a)
        if v <= 0:
            return 0.0
        amount = min(amount, v)
        hold[a] = {t: x * (1 - amount / v) for t, x in hold[a].items()}
        return amount

    for k, d in enumerate(months):
        year_float = 1 + k / 12
        boundary = k % 12 == 0
        y = k // 12 + 1
        if boundary:
            if y in plan.contributions:
                hold["growth"][CASH] = hold["growth"].get(CASH, 0.0) + plan.contributions[y]
                events.append(dict(date=d, year=y, event="contribution", amount=plan.contributions[y]))
            if y == plan.reserve_year:
                total = val("growth")
                rf_d = float(data.tbill.loc[:d].dropna().iloc[-1])
                r_amt = min(plan.reserve_needed(rf_d), total)
                hold["reserve"] = {t: x * r_amt / total for t, x in hold["growth"].items()}
                hold["funding"] = {t: x * (total - r_amt) / total for t, x in hold["growth"].items()}
                hold["growth"] = {}
                events.append(dict(date=d, year=y, event="split", amount=total, reserve=r_amt, funding=total - r_amt))
            if y == plan.gift_year:
                before = val("funding")
                g = move_out("funding", plan.gift_share * before)
                events.append(dict(date=d, year=y, event="gift", amount=g, funding_before=before))
            if y in plan.payment_years:
                paid = move_out("reserve", plan.payment)
                events.append(dict(date=d, year=y, event="payment", amount=paid, shortfall=plan.payment - paid))
            if y == plan.end_year:
                break
        # rebalance active accounts
        for a in hold:
            ph = plan.phase_for(a, y)
            if ph is None or val(a) <= 0:
                continue
            flip = False
            if plan.phases[ph].trend_switch:              # gate M1 is checked every month
                now_up = market_up(data, d, plan)
                flip = last_up[a] is not None and now_up != last_up[a]
                last_up[a] = now_up
            if not (boundary or flip or k - last_reb[a] >= plan.phases[ph].rebalance_months):
                continue
            dec = decide(data, d, plan, ph, year_float, backtest_mode=True, sim=sim, selector=selector, cache=cache)
            v = val(a)
            target = {t: v * x for t, x in dec["weights"].items()}
            traded = sum(abs(target.get(t, 0.0) - hold[a].get(t, 0.0)) for t in set(target) | set(hold[a])) / 2
            cost = traded * plan.cost_bps / 1e4
            hold[a] = {t: x * (1 - cost / v) for t, x in target.items()}
            last_reb[a] = k
            decisions.append(dict(date=d, year=y, account=a, phase=ph, value=v, traded=traded, cost=cost,
                                  weights=dec["weights"], equity_target=dec["info"]["equity_target"],
                                  sleeves={s: dict(status=i["status"], p_hit=i["p_hit"]) for s, i in dec["info"]["sleeves"].items()}))
            if verbose:
                st = "; ".join(f"{s} {i['status'][:16]}" for s, i in dec["info"]["sleeves"].items())
                print(f"{d.date()}  y{y:<2d} {a:8s} {ph:12s} ${v:>10,.0f}  eq-target {dec['info']['equity_target']:6.1%}  {st}")
        # one month of returns
        if k + 1 < len(months):
            m = months[k + 1]
            row = rets.loc[m]
            rec = dict(date=m)
            for a in hold:
                v0 = val(a)
                hold[a] = {t: x * (1 + (0.0 if pd.isna(row.get(t, np.nan)) else float(row[t]))) for t, x in hold[a].items()}
                rec[a] = val(a)
                rec[a + "_ret"] = rec[a] / v0 - 1 if v0 > 0 else np.nan     # return before any cash flows
            series.append(rec)

    s = pd.DataFrame(series).set_index("date")
    ev = pd.DataFrame(events)
    pay = ev[ev.event == "payment"]
    split = ev[ev.event == "split"].iloc[0]
    gift = ev[ev.event == "gift"].iloc[0]
    sheet = {sc: project(plan, sc) for sc in ("bear", "base", "bull")}
    twr = lambda a: (1 + s[a + "_ret"].dropna()).cumprod()
    acct_stats = {}
    for a in ("growth", "reserve", "funding"):
        r_ = s[a + "_ret"].dropna()
        if len(r_):
            idx = twr(a)
            acct_stats[a] = dict(cagr=float(idx.iloc[-1] ** (12 / len(r_)) - 1), vol=float(r_.std() * math.sqrt(12)),
                                 max_dd=max_drawdown(np.r_[1.0, idx.values]), worst_12m=float((1 + r_).rolling(12).apply(np.prod, raw=True).min() - 1)) 
    summary = dict(
        start=pd.Timestamp(start).date(), value_2031=float(split.amount),
        growth_cagr_vs_contributions=_irr_like(plan, float(split.amount)),
        accounts=acct_stats,
        funding_2033_before_gift=float(gift.funding_before), gift=float(gift.amount),
        payments_made=int((pay.shortfall < 0.01).sum()), payments_total=len(plan.payment_years),
        total_shortfall=float(pay.shortfall.sum()), reserve_left_2043=float(s["reserve"].iloc[-1]),
        funding_left_2043=float(s["funding"].iloc[-1]),
        sheet={sc: dict(value_2031=v["value_2031"], gift=v["gift"], reserve_end=v["reserve_end"]) for sc, v in sheet.items()},
        trading_costs=float(sum(x["cost"] for x in decisions)))
    return dict(series=s, events=ev, decisions=decisions, summary=summary)


def _irr_like(plan: Plan, value_at_split: float) -> float:
    """Constant annual return that turns the two contributions into the 2031 value (money-weighted)."""
    c = plan.contributions
    f = lambda r: sum(a * (1 + r) ** (plan.reserve_year - y) for y, a in c.items()) - value_at_split
    lo, hi = -0.99, 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def print_summary(sm: dict):
    sh = sm["sheet"]
    print(f"\nLifecycle backtest starting {sm['start']}  (plays the role of Jan 2027)")
    print(f"  2031 value        ${sm['value_2031']:>10,.0f}   sheet bear/base/bull ${sh['bear']['value_2031']:,.0f} / ${sh['base']['value_2031']:,.0f} / ${sh['bull']['value_2031']:,.0f}")
    print(f"  growth return     {sm['growth_cagr_vs_contributions']:.1%}/yr (money-weighted)")
    for a, st in sm["accounts"].items():
        print(f"    {a:8s} {st['cagr']:6.1%}/yr  vol {st['vol']:5.1%}  max drawdown {st['max_dd']:5.0%}  worst 12 months {st['worst_12m']:5.0%}")
    print(f"  facility gift     ${sm['gift']:>10,.0f}   sheet ${sh['bear']['gift']:,.0f} / ${sh['base']['gift']:,.0f} / ${sh['bull']['gift']:,.0f}")
    print(f"  payments          {sm['payments_made']}/{sm['payments_total']} paid in full   shortfall ${sm['total_shortfall']:,.0f}")
    print(f"  reserve left 2043 ${sm['reserve_left_2043']:>10,.0f}   sheet ${sh['bear']['reserve_end']:,.0f} / ${sh['base']['reserve_end']:,.0f} / ${sh['bull']['reserve_end']:,.0f}")
    print(f"  funding left 2043 ${sm['funding_left_2043']:>10,.0f}   trading costs ${sm['trading_costs']:,.0f}")


if __name__ == "__main__":
    import sys
    data = MarketData(os.path.join(HERE, "data"))
    if len(sys.argv) > 2 and sys.argv[1] == "backtest":
        bt = lifecycle_backtest(data, sys.argv[2])
        print_summary(bt["summary"])
    else:
        for ph, dec in plan_today(data, "2026-09-30").items():
            top = sorted(dec["weights"].items(), key=lambda x: -x[1])[:8]
            print(f"{ph:12s} equity target {dec['info']['equity_target']:6.1%}  " +
                  ", ".join(f"{t} {x:.0%}" for t, x in top))
