"""Tests for funnel_v2. Run:  python3 -m pytest -q test_funnel_v2.py"""
import copy, math, os
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from funnel_v2 import (Config, MarketData, select, backtest, psr, ledoit_wolf, stationary_bootstrap_idx,
                       simulate_portfolio, max_drawdown, CASH)

HERE = os.path.dirname(os.path.abspath(__file__))
ASOF = "2026-09-30"
FAST = dict(screen_paths=600, mc_paths=1500, resamples=6, stock_bootstrap_paths=400, grid_points=8)


@pytest.fixture(scope="module")
def data():
    return MarketData(os.path.join(HERE, "data"))


@pytest.fixture(scope="module")
def base_result(data):
    return select(data, ASOF, Config(**FAST))


# ---------------------------------------------------------------- statistics
def test_psr_normal_matches_closed_form():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.05, 240)
    sr = x.mean() / x.std(ddof=1)
    expected = stats.norm.cdf(sr * math.sqrt(239) / math.sqrt(1 - stats.skew(x) * sr + (stats.kurtosis(x, fisher=False) - 1) / 4 * sr ** 2))
    assert abs(psr(x) - expected) < 1e-12
    assert psr(-x) < 0.5 < psr(x)


def test_ledoit_wolf_matches_sklearn():
    from sklearn.covariance import LedoitWolf
    X = np.random.default_rng(1).normal(size=(60, 12))
    assert np.allclose(ledoit_wolf(X), LedoitWolf().fit(X).covariance_, atol=1e-10)
    assert np.linalg.eigvalsh(ledoit_wolf(X)).min() > 0


def test_stationary_bootstrap_mean_block_length():
    idx = stationary_bootstrap_idx(100, 2000, 60, 6.0, np.random.default_rng(2))
    breaks = (np.diff(idx, axis=1) != 1) & ~((idx[:, :-1] == 99) & (idx[:, 1:] == 0))
    assert 5.0 < 1 / breaks.mean() < 7.0                     # mean run length ~ block length
    assert idx.min() >= 0 and idx.max() < 100


def test_monte_carlo_matches_lognormal_theory():
    rng = np.random.default_rng(3)
    mu_a, vol, H = 0.10, 0.20, 48                               # annual arithmetic mean, annual vol, months
    R = rng.normal(mu_a / 12, vol / math.sqrt(12), (600, 1))
    cfg = Config(t_dof=200, block_length=1.0)
    sims = simulate_portfolio(R, np.array([mu_a / 12]), np.array([[vol ** 2 / 12]]), np.array([1.0]), H, 20000, cfg, rng)
    thr = 1.08 ** 4
    m = H * (math.log(1 + mu_a / 12) - (vol ** 2 / 12) / 2)     # log-normal approximation
    p_theory = 1 - stats.norm.cdf((math.log(thr) - m) / (vol * math.sqrt(H / 12)))
    for s in sims.values():
        assert abs(np.mean(s["growth"] >= thr) - p_theory) < 0.03


def test_max_drawdown():
    assert max_drawdown(np.array([1, 2, 1, 3, 1.5])) == pytest.approx(-0.5)


# ---------------------------------------------------------------- algorithm behaviour
def test_weights_valid_and_caps_respected(base_result):
    cfg, w, t = base_result.cfg, base_result.weights, base_result.table
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in w.values())
    stocks = {k: v for k, v in w.items() if k != CASH}
    assert max(stocks.values()) <= cfg.name_cap + 1e-6
    assert all(v >= cfg.min_weight - 1e-9 for v in stocks.values())
    for col, cap in (("sector", cfg.sector_cap), ("country", cfg.country_cap)):
        assert pd.Series(stocks).groupby(t.loc[list(stocks), col]).sum().max() <= cap + 1e-6


def test_every_holding_passed_every_stock_gate(base_result):
    cut = {tk for g in base_result.gates if g["gate"][0] == "A" for tk, _ in g["cut"]}
    assert not (set(base_result.weights) - {CASH}) & cut


def test_no_lookahead(data):
    """Scrambling everything after `asof` must not change the selection."""
    asof = pd.Timestamp("2024-06-30")
    d2 = copy.copy(data)
    rng = np.random.default_rng(4)
    after = d2.price_usd.index > asof
    d2.price_usd = data.price_usd.copy(); d2.price_usd.loc[after] *= rng.uniform(0.2, 5.0, d2.price_usd.loc[after].shape)
    d2.price_local = data.price_local.copy(); d2.price_local.loc[after] *= 3.0
    d2.bench = data.bench.copy(); d2.bench.loc[d2.bench.index > asof] *= 0.1
    d2.fund = data.fund.copy()
    late = d2.fund.period_end + pd.Timedelta(days=90) > asof
    d2.fund.loc[late, "value"] *= -7.0
    cfg = Config(**FAST)
    a, b = select(data, asof, cfg), select(d2, asof, cfg)
    assert a.weights.keys() == b.weights.keys()
    assert all(a.weights[k] == pytest.approx(b.weights[k], abs=1e-12) for k in a.weights)


def test_deterministic(data, base_result):
    again = select(data, ASOF, Config(**FAST))
    assert again.weights == base_result.weights


def test_low_target_is_met_with_cash(data):
    r = select(data, ASOF, Config(target_cagr=0.05, **FAST))
    assert r.status == "TARGET MET"
    assert r.weights.get(CASH, 0) > 0.2
    assert r.mc["p_hit"] >= 0.5


def test_unreachable_target_is_flagged(data):
    r = select(data, ASOF, Config(target_cagr=0.40, **FAST))
    assert r.status.startswith("TARGET NOT MET")
    assert r.mc["p_hit"] < 0.5


def test_higher_target_takes_more_risk(data):
    lo = select(data, ASOF, Config(target_cagr=0.08, **FAST))
    hi = select(data, ASOF, Config(target_cagr=0.13, **FAST))
    assert hi.mc["exp_return"] > lo.mc["exp_return"]
    assert hi.mc["vol"] > lo.mc["vol"]


def test_backtest_smoke(data):
    bt = backtest(data, "2025-06-30", "2026-09-30", Config.for_backtest(**FAST), rebalance_months=6, verbose=False)
    s = bt["series"]
    assert len(s) >= 12 and (s["nav"] > 0).all()
    assert s.index.min() > pd.Timestamp("2025-06-30")         # returns start the month AFTER the first decision
    assert set(bt["stats"]) >= {"cagr", "vol", "sharpe", "max_dd", "bench_cagr"}


def test_despike_repairs_glitches_but_keeps_real_crashes(data):
    from funnel_v2 import despike
    assert 25 < data.fx["TWD"].loc["2014-12-31"] < 35                      # was 3.67 in the raw file
    assert data.price_local["HDFCBANK.NS"].loc["2006-04-30"] < 40          # was 139 (4x spike)
    assert data.price_local["NVDA"].loc["2001-09-30"] == pytest.approx(0.21, abs=0.01)   # real post-9/11 low stays
    s = pd.DataFrame({"x": [10, 10, 40, 10, 10, 5, 5]})
    assert despike(s)["x"].tolist() == [10, 10, 10, 10, 10, 5, 5]          # spike fixed, genuine level shift kept
