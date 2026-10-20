"""Tests for laura_strategy. Run:  python3 -m pytest -q test_laura_strategy.py"""
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from funnel_v2 import CASH, MarketData
from laura_strategy import Plan as _Plan, decide, default_phases, lifecycle_backtest, project, sleeve_configs, v3_phases


def Plan(**kw):
    """Tests of the spreadsheet plan and the v2 settings use v2 phases explicitly (Plan() itself defaults to v3)."""
    kw.setdefault("phases", default_phases())
    return _Plan(**kw)

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def data():
    return MarketData(os.path.join(HERE, "data"))


# ---------------------------------------------------------------- the plan reproduces the team's spreadsheet
@pytest.mark.parametrize("sc, v2031, fund2031, gift, fund2034, res2042, fund2043", [
    ("bear", 705.58, 255.58, 203.36, 77.95, 12.03, 238.46),
    ("base", 752.83, 302.83, 250.41, 98.91, 49.71, 384.58),
    ("bull", 802.41, 352.41, 302.60, 122.05, 92.45, 560.82)])
def test_projection_matches_spreadsheet(sc, v2031, fund2031, gift, fund2034, res2042, fund2043):
    p = project(Plan(), sc)
    k = lambda x: round(x / 1000, 2)
    assert k(p["value_2031"]) == pytest.approx(v2031, abs=0.02)
    assert k(p["rows"][5]["funding"]) == pytest.approx(fund2031, abs=0.02)
    assert k(p["gift"]) == pytest.approx(gift, abs=0.02)
    assert k(p["rows"][8]["funding"]) == pytest.approx(fund2034, abs=0.02)
    assert k(p["reserve_end"]) == pytest.approx(res2042, abs=0.02)
    assert k(p["funding_end"]) == pytest.approx(fund2043, abs=0.02)


def test_phase_schedule():
    p = Plan()
    assert [p.phase_for("growth", y) for y in (1, 4, 5)] == ["growth", "growth", None]
    assert [p.phase_for("funding", y) for y in (4, 5, 6, 7, 16, 17)] == [None, "pre_gift_y5", "pre_gift", "post_gift", "post_gift", None]
    assert [p.phase_for("reserve", y) for y in (4, 5, 16, 17)] == [None, "reserve", "reserve", None]
    for ph in p.phases.values():
        assert sum(ph.mix.values()) == pytest.approx(1.0)


def test_equity_sleeve_target_makes_account_hit_its_target():
    p, rf = Plan(), 0.04
    for name, ph in p.phases.items():
        cfg, eq_t = sleeve_configs(p, name, 1, rf, True)
        bond = rf + p.bond_term_premium
        account = ph.mix["cash"] * rf + ph.mix["bonds"] * bond + (ph.mix["domestic"] + ph.mix["foreign"]) * eq_t
        assert account == pytest.approx(ph.target)
        assert cfg.target_cagr == pytest.approx(eq_t)


def test_horizon_counts_down_to_phase_end():
    p = Plan()
    assert sleeve_configs(p, "growth", 1.0, 0.04, True)[0].horizon_years == 4
    assert sleeve_configs(p, "growth", 3.5, 0.04, True)[0].horizon_years == 2
    assert sleeve_configs(p, "pre_gift", 6.0, 0.04, True)[0].horizon_years == 1


# ---------------------------------------------------------------- lifecycle accounting (stub selector, fast)
def cash_only(data, asof, cfg, universe):
    return SimpleNamespace(weights={CASH: 1.0}, status="STUB", mc={})


@pytest.fixture(scope="module")
def stub_run(data):
    return lifecycle_backtest(data, "2007-12-31", Plan(), sim=None, selector=cash_only, verbose=False)


def test_cash_flows_happen_on_schedule(stub_run):
    ev = stub_run["events"]
    start = pd.Timestamp("2007-12-31")
    yr = lambda y: start + pd.offsets.MonthEnd(12 * (y - 1))
    c = ev[ev.event == "contribution"]
    assert list(c.amount) == [300_000, 150_000] and list(c.date) == [yr(1), yr(2)]
    split = ev[ev.event == "split"].iloc[0]
    assert split.date == yr(5) and split.reserve == pytest.approx(450_000)
    gift = ev[ev.event == "gift"].iloc[0]
    assert gift.date == yr(7) and gift.amount == pytest.approx(0.75 * gift.funding_before)
    pay = ev[ev.event == "payment"]
    assert len(pay) == 10 and list(pay.date) == [yr(y) for y in range(7, 17)]


def test_shortfalls_are_reported_and_rate_based_reserve_fixes_them(stub_run, data):
    # cash-only stress, reserve years 2012-2022 (near-zero rates): the fixed $450k falls short and says so
    sm, ev = stub_run["summary"], stub_run["events"]
    pay = ev[ev.event == "payment"]
    assert sm["total_shortfall"] == pytest.approx((50_000 - pay.amount).sum())
    assert sm["payments_made"] == (pay.shortfall < 0.01).sum()
    # sizing the reserve from the yield on the split date funds every payment in the same stress
    bt = lifecycle_backtest(data, "2007-12-31", Plan(reserve_rule="pv"), selector=cash_only, verbose=False)
    assert bt["summary"]["payments_made"] == 10 and bt["summary"]["total_shortfall"] == pytest.approx(0)


def test_reserve_rule():
    assert Plan().reserve_needed(0.04) == 450_000
    pv = Plan(reserve_rule="pv").reserve_needed(0.04)          # priced at 3.5%
    assert pv == pytest.approx(sum(50_000 / 1.035 ** k for k in range(2, 12)) * 1.02)
    assert Plan(reserve_rule="pv_floor").reserve_needed(0.08) == 450_000


def test_mix_respected(stub_run):
    p = Plan()
    for d in stub_run["decisions"]:
        mix = p.phases[d["phase"]].mix
        assert d["weights"].get(p.phases[d["phase"]].bond_proxy, 0) == pytest.approx(mix["bonds"])
        assert sum(d["weights"].values()) == pytest.approx(1.0)


def test_rebalances_only_use_past_dates(stub_run):
    """Every decision date is a month-end inside the timeline; the selector never sees a later date."""
    seen = []
    def spy(data, asof, cfg, universe):
        seen.append(pd.Timestamp(asof))
        return cash_only(data, asof, cfg, universe)
    data = MarketData(os.path.join(HERE, "data"))
    bt = lifecycle_backtest(data, "2007-12-31", Plan(), selector=spy, verbose=False)
    months = list(bt["series"].index)
    assert seen and max(seen) < months[-1] and all(s in set(months) | {pd.Timestamp("2007-12-31")} for s in seen)


def test_value_is_conserved_apart_from_returns_flows_and_costs(stub_run):
    s, ev = stub_run["series"], stub_run["events"]
    total = s[["growth", "reserve", "funding"]].sum(axis=1)
    flows_in = ev[ev.event == "contribution"].amount.sum()
    flows_out = ev[ev.event.isin(["gift", "payment"])].amount.sum()
    # rebuild ending wealth from monthly account returns + flows + costs (all money accounted for)
    end_total = total.iloc[-1] + flows_out + stub_run["summary"]["trading_costs"]
    assert end_total > flows_in * 0.99                     # cash/bonds over 16 years: never lose the money


# ---------------------------------------------------------------- live decision, real Funnel
def test_live_decision_reserve_is_safe(data):
    dec = decide(data, "2026-09-30", Plan(), "reserve", 5, backtest_mode=True,
                 sim=dict(screen_paths=400, mc_paths=800, resamples=4, stock_bootstrap_paths=300, grid_points=6))
    w = dec["weights"]
    assert sum(w.values()) == pytest.approx(1.0)
    assert w.get("IEF", 0) == pytest.approx(0.30)
    assert w.get(CASH, 0) >= 0.60                          # 2-4% target: the Funnel parks the equity sleeves in T-bills


def test_unfilled_growth_sleeve_holds_index_not_cash(data):
    empty = lambda data, asof, cfg, universe: SimpleNamespace(weights={}, status="NO STOCKS PASSED", mc={})
    g = decide(data, "2012-12-31", Plan(), "growth", 1, selector=empty)["weights"]
    assert g.get("SPY") == pytest.approx(0.50) and g.get("EFA") == pytest.approx(0.20) and CASH not in g
    r = decide(data, "2012-12-31", Plan(), "reserve", 5, selector=empty)["weights"]
    assert r.get(CASH) == pytest.approx(0.70) and "SPY" not in r      # safety phases still go to T-bills


# ---------------------------------------------------------------- v3 revisions (growth and post-gift only)
def test_v3_changes_only_growth_and_post_gift():
    v2, v3 = default_phases(), v3_phases()
    for k in ("reserve", "pre_gift_y5", "pre_gift"):
        assert v3[k] == v2[k]
    assert v3["growth"].trend_switch and v3["growth"].bond_choices == (0.0, 0.10, 0.20, 0.30)
    assert v3["post_gift"].target == 0.09


def test_market_switch_uses_only_past_prices_and_moves_sleeve_to_tbills(data):
    from laura_strategy import market_up
    p = Plan(phases=v3_phases())
    assert market_up(data, "2008-10-31", p)["domestic"] is False          # S&P 500 deep below its 10-month average
    assert market_up(data, "2013-12-31", p)["domestic"] is True
    stub = lambda data, asof, cfg, universe: SimpleNamespace(weights={"AAPL": 1.0}, status="TARGET MET", mc={"p_hit": 0.9})
    w = decide(data, "2008-10-31", p, "growth", 1, selector=stub)["weights"]
    assert "AAPL" not in w and w[CASH] >= 0.69                              # both markets down -> stocks to T-bills
    w2 = decide(data, "2013-12-31", p, "growth", 1, selector=stub)["weights"]
    assert w2.get("IEF", 0) == pytest.approx(0.30)                          # met with room (P 0.9) at every share -> most bonds
    assert w2.get(CASH, 0) == 0                                             # no T-bills inside the stock sleeves


def test_bond_choice_falls_to_fewer_bonds_when_target_is_missed(data):
    p = Plan(phases=v3_phases())
    def stub(data, asof, cfg, universe):                                    # only an all-stock mix "hits"
        ok = cfg.target_cagr < 0.155
        return SimpleNamespace(weights={"AAPL": 1.0}, status="TARGET MET" if ok else "TARGET NOT MET", mc={"p_hit": 0.7 if ok else 0.3})
    w = decide(data, "2013-12-31", p, "growth", 1, selector=stub)["weights"]
    assert w.get("IEF", 0) == 0.0


def test_bonds_added_only_with_room_to_spare(data):
    p = Plan(phases=v3_phases())
    stub = lambda data, asof, cfg, universe: SimpleNamespace(weights={"AAPL": 1.0}, status="TARGET MET", mc={"p_hit": 0.55})
    w = decide(data, "2013-12-31", p, "growth", 1, selector=stub)["weights"]
    assert w.get("IEF", 0) == 0.0                                           # met, but not by 10 points -> stay all-stock


def test_default_plan_is_v3():
    assert _Plan().phases == v3_phases()
