"""
THE FUNNEL v2 - target-return stock selection for any market
=============================================================
Plug in the return you need -> get the stocks, the weights, and the odds of hitting it.

    from funnel_v2 import MarketData, Config, select
    data = MarketData("data")                       # CSVs written by fetch_data.py
    res  = select(data, "2026-09-30", Config(target_cagr=0.15, horizon_years=4))
    res.print_report()

Pipeline (each gate shaves the list; every rule is a ratio, rank or probability, so it is
market-agnostic - Part 1 decides the countries, this decides the stocks):

  STAGE A - stock gates
    A1 Data sufficiency        enough clean monthly history + point-in-time fundamentals
    A2 Size & liquidity        market cap and traded value in USD
    A3 Valuation sanity        P/E > 0, <= 60, <= 2x peer median (P/B for financials)
    A4 Solvency                net debt / EBITDA, interest cover (financials exempt)
    A5 Earnings quality        Piotroski F-score, Sloan accruals
    A6 Trend                   price above 10-month average OR positive 12-1 momentum
    A7 Skill, not luck         Probabilistic Sharpe Ratio (Bailey & Lopez de Prado), Sortino
    A8 Tail risk               CVaR(95%), max drawdown, beta
    A9 Monte Carlo edge        expected return > T-bills AND bootstrap P(beat T-bills over horizon)
  STAGE B - rank
    B1 Composite score         quality, value, growth, momentum, risk-adjusted, expected return
  STAGE C - portfolio gates
    C1 Redundancy              hierarchical clustering on correlation
    C2 Target solver           min-variance weights that deliver the target (Ledoit-Wolf covariance)
    C3 Resampled stability     Michaud resampling of inputs; average weights; prune tiny positions
    C4 Monte Carlo confidence  P(CAGR >= target) from block-bootstrap AND fat-tailed (Student-t) engines
    C5 Drawdown budget         P(max drawdown worse than tolerance) <= limit
    C6 Concentration           name / sector / country caps, minimum number of names

No look-ahead: select() first cuts every input to what was knowable on `asof`
(prices <= asof, fundamentals only after period end + reporting lag). backtest() walks forward.
"""
from __future__ import annotations
import math, os
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

CASH = "CASH"


# =============================================================================== config
@dataclass
class Config:
    # ---- what you need (plug these in)
    target_cagr: float = 0.15              # required compound annual return
    horizon_years: int = 4                 # years the money must grow
    confidence: float = 0.50               # required P(CAGR >= target); 0.5 = median outcome hits it
    scenario_targets: dict = field(default_factory=lambda: {"bear": 0.13, "base": 0.15, "bull": 0.17})
    max_drawdown_tolerance: float = -0.40  # C4: a fall worse than this is "too much"
    max_drawdown_prob: float = 0.35        # C4: ...allowed in at most this share of simulations
    allow_cash: object = "auto"            # "auto": T-bills only when the target is below the safest stock mix
    # ---- data
    lookback_months: int = 60
    min_months: int = 60
    max_zero_return_share: float = 0.10    # stale-price filter
    fundamental_lag_days: int = 90         # annual report usable 90 days after fiscal year end
    missing_fundamentals: str = "fail"     # "fail" (live) or "skip" (backtests before statements exist)
    # ---- A2
    min_mcap_usd_b: float = 2.0
    min_median_monthly_value_usd_m: float = 50.0
    # ---- A3
    pe_hard_cap: float = 60.0
    pe_relative_cap: float = 2.0
    pb_hard_cap_financials: float = 4.0
    financial_sectors: tuple = ("Financial Services",)
    min_peers_for_country_group: int = 8
    # ---- A4
    max_net_debt_to_ebitda: float = 3.5
    min_interest_cover: float = 3.0
    # ---- A5
    min_piotroski: float = 5.0
    max_accruals: float = 0.10
    # ---- A7
    min_psr: float = 0.80
    min_sortino: float = 0.30
    psr_mode: str = "absolute"             # "rank": keep the top psr_top_share of survivors by PSR (v3) ...
    psr_top_share: float = 1 / 3
    psr_floor: float = 0.50                # ... but never a PSR below this
    # ---- A8
    min_cvar95_monthly: float = -0.20
    min_max_drawdown: float = -0.60
    max_beta: float = 1.80
    drawdown_mode: str = "absolute"        # "relative": the stock's max drawdown minus the market's, same months (v3)
    max_dd_vs_market: float = -0.25        # relative mode: cut if more than 25 points worse than the market
    # ---- A9
    min_p_beat_cash: float = 0.55
    stock_bootstrap_paths: int = 2000
    block_length: int = 6
    # ---- expected-return model (annual, arithmetic)
    equity_risk_premium: float = 0.05
    w_hist: float = 0.40
    w_fund: float = 0.30
    w_capm: float = 0.30
    growth_cap: float = 0.10
    # ---- B1
    pillar_weights: dict = field(default_factory=lambda: {
        "quality": 0.20, "value": 0.15, "growth": 0.15, "momentum": 0.15, "risk": 0.15, "exp_return": 0.20})
    top_n: int = 25
    # ---- C
    max_pair_corr: float = 0.85
    name_cap: float = 0.15
    sector_cap: float = 0.35
    country_cap: float = 0.40
    min_names: int = 7
    min_weight: float = 0.02
    grid_points: int = 15
    screen_paths: int = 3000
    mc_paths: int = 10000
    t_dof: int = 5
    resamples: int = 30
    seed: int = 42


    @classmethod
    def for_backtest(cls, **kw):
        """Same rules, lighter simulation counts so a walk-forward test runs in minutes, and fundamentals
        gates pass when no report existed yet (early years). Results are seeded, so runs are repeatable."""
        base = dict(screen_paths=1000, mc_paths=3000, resamples=12, stock_bootstrap_paths=800, grid_points=10,
                    missing_fundamentals="skip")
        base.update(kw)
        return cls(**base)


# =============================================================================== data
# Yahoo and S&P (GICS) name the same sectors differently; map GICS names onto Yahoo's
SECTOR_NAMES = {"Financials": "Financial Services", "Consumer Discretionary": "Consumer Cyclical",
                "Consumer Staples": "Consumer Defensive", "Health Care": "Healthcare",
                "Information Technology": "Technology", "Materials": "Basic Materials"}


def despike(df: pd.DataFrame, hi: float = 2.0, lo: float = 0.5, agree: float = 0.15) -> pd.DataFrame:
    """Repair one-month data errors: a value at least 2x (or at most half) BOTH neighbours, while the
    neighbours agree within 15%, is a vendor glitch (it never traded there) and is replaced by the
    average of the neighbours. Real crashes and rallies don't snap back within one month, so they stay."""
    out = df.copy()
    for c in out.columns:
        x = out[c]
        prev, nxt = x.ffill().shift(1), x.bfill().shift(-1)
        mid = (prev + nxt) / 2
        ratio = x / mid
        bad = ((ratio > hi) | (ratio < lo)) & ((nxt / prev - 1).abs() < agree) & x.notna()
        out.loc[bad, c] = mid[bad]
    return out


class MarketData:
    """Loads the CSVs from fetch_data.py. All prices converted to USD."""

    def __init__(self, folder: str):
        rd = lambda f, **k: pd.read_csv(os.path.join(folder, f), **k)
        self.meta = rd("meta.csv").set_index("ticker")
        self.meta["sector"] = self.meta["sector"].replace(SECTOR_NAMES)   # one naming system for sector caps
        local = rd("prices_local.csv", index_col=0, parse_dates=True)
        self.volume = rd("volume.csv", index_col=0, parse_dates=True)
        fx = rd("fx_per_usd.csv", index_col=0, parse_dates=True)
        self.fund = rd("fundamentals.csv", parse_dates=["period_end"])
        self.fund["period_end"] = pd.to_datetime(self.fund["period_end"])     # also when the file is empty
        self.bench = rd("benchmark.csv", index_col=0, parse_dates=True).iloc[:, 0]
        self.tbill = rd("tbill.csv", index_col=0, parse_dates=True).iloc[:, 0] / 100.0
        local, fx = despike(local), despike(fx)          # vendor glitches (e.g. TWD Dec-2014 at 3.67 instead of ~31)
        self.price_local = local
        self.fx = fx
        usd = {}
        for t in local.columns:
            ccy = str(self.meta.loc[t, "currency"])
            scale = 0.01 if ccy == "GBp" else 1.0           # London quotes in pence
            p = local[t] * scale
            if ccy.upper() != "USD":
                p = p / fx[ccy.upper()].reindex(p.index).ffill()
            usd[t] = p
        self.price_usd = pd.DataFrame(usd)
        self.folder = folder
        self.unit_scale = {t: (0.01 if str(self.meta.loc[t, "currency"]) == "GBp" else 1.0) for t in local.columns}

    def fundamentals_asof(self, ticker: str, asof: pd.Timestamp, lag_days: int) -> pd.DataFrame:
        """Fiscal years whose reports were public on `asof` (period_end + lag <= asof). Newest first."""
        f = self.fund[(self.fund.ticker == ticker) & (self.fund.period_end + pd.Timedelta(days=lag_days) <= asof)]
        if f.empty:
            return pd.DataFrame()
        return f.pivot_table(index="period_end", columns="field", values="value").sort_index(ascending=False)


# =============================================================================== statistics
def psr(excess: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true Sharpe > benchmark) allowing for skew and fat tails."""
    n = len(excess)
    sd = excess.std(ddof=1)
    if n < 3 or sd == 0:
        return 0.0
    sr = excess.mean() / sd
    g3 = stats.skew(excess)
    g4 = stats.kurtosis(excess, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def sortino(excess: np.ndarray) -> float:
    dd = math.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    return float(excess.mean() * 12 / (dd * math.sqrt(12))) if dd > 0 else float("inf")


def cvar(r: np.ndarray, level: float = 0.95) -> float:
    q = np.quantile(r, 1 - level)
    return float(r[r <= q].mean())


def max_drawdown(prices: np.ndarray) -> float:
    return float((prices / np.maximum.accumulate(prices) - 1).min())


def stationary_bootstrap_idx(n_obs: int, n_paths: int, n_steps: int, block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: random blocks with mean length `block` (keeps autocorrelation)."""
    idx = np.empty((n_paths, n_steps), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_paths)
    jump = rng.random((n_paths, n_steps)) < 1.0 / block
    fresh = rng.integers(0, n_obs, (n_paths, n_steps))
    for t in range(1, n_steps):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n_obs)
    return idx


def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (shrinks noisy sample covariance toward a scaled identity)."""
    X = X - X.mean(0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / p / n ** 2
    b2 = min(b2, d2)
    shrink = b2 / d2 if d2 > 0 else 1.0
    return shrink * mu * np.eye(p) + (1 - shrink) * S


def pct_rank(v) -> np.ndarray:
    """Percentile rank 0..1 among peers; missing values get the lowest rank."""
    v = np.asarray(v, float)
    if len(v) <= 1:
        return np.ones(len(v))
    fin = np.isfinite(v)
    v = np.where(fin, v, (v[fin].min() - 1) if fin.any() else 0.0)
    return (stats.rankdata(v, method="average") - 1) / (len(v) - 1)


def simulate_portfolio(R_hist: np.ndarray, mu_m: np.ndarray, cov_m: np.ndarray, w: np.ndarray, months: int,
                       paths: int, cfg: Config, rng, cash_m: float = 0.0, w_cash: float = 0.0):
    """Two Monte Carlo engines on the same weights. Returns dict of terminal growth and max drawdown per engine.
    R_hist: (n, k) historical monthly returns of the k stocks; mu_m: forward monthly means; cov_m: monthly cov."""
    out = {}
    # engine 1: stationary block bootstrap of joint history, re-centred on forward expected returns
    Rc = R_hist - R_hist.mean(0) + mu_m
    idx = stationary_bootstrap_idx(len(Rc), paths, months, cfg.block_length, rng)
    rp = Rc[idx] @ w + w_cash * cash_m
    out["bootstrap"] = rp
    # engine 2: multivariate Student-t (fat tails, same covariance)
    k = len(w)
    L = np.linalg.cholesky(cov_m + 1e-12 * np.eye(k))
    z = rng.standard_normal((paths, months, k)) @ L.T
    chi = rng.chisquare(cfg.t_dof, (paths, months, 1))
    z = z * np.sqrt((cfg.t_dof - 2) / chi)
    rp2 = (z + mu_m) @ w + w_cash * cash_m
    out["student_t"] = rp2
    res = {}
    for name, r in out.items():
        r = np.maximum(r, -0.99)
        wealth = np.cumprod(1 + r, axis=1)
        res[name] = dict(growth=wealth[:, -1],
                         max_dd=(wealth / np.maximum(np.maximum.accumulate(wealth, axis=1), 1.0) - 1).min(1))
    return res


# =============================================================================== result
@dataclass
class Result:
    asof: pd.Timestamp
    cfg: Config
    gates: list
    table: pd.DataFrame
    weights: dict
    mc: dict
    status: str
    notes: list

    def print_report(self):
        print(f"\nFUNNEL v2  as of {self.asof.date()}  target {self.cfg.target_cagr:.1%}/yr over "
              f"{self.cfg.horizon_years}y at {self.cfg.confidence:.0%} confidence")
        for g in self.gates:
            print(f"  {g['gate']:<34s} in {g['n_in']:>3d}  out {g['n_out']:>3d}")
            for t, why in g["cut"][:12]:
                print(f"      x {t:<14s} {why}")
        print(f"\n  STATUS: {self.status}")
        for t, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            print(f"    {t:<14s} {w:6.1%}")
        if self.mc:
            m = self.mc
            print(f"  P(CAGR >= target) {m['p_hit']:.1%}   median CAGR {m['cagr_p50']:.1%}   "
                  f"10th-90th {m['cagr_p10']:.1%} to {m['cagr_p90']:.1%}   P(maxDD<{self.cfg.max_drawdown_tolerance:.0%}) {m['p_dd']:.1%}")
            for k, v in m["scenarios"].items():
                print(f"    P(CAGR >= {k} {self.cfg.scenario_targets[k]:.0%}) = {v:.1%}")
        for n in self.notes:
            print("  note:", n)


# =============================================================================== the funnel
def select(data: MarketData, asof, cfg: Config | None = None, universe: list | None = None) -> Result:
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    asof = pd.Timestamp(asof)
    # ---------- point-in-time slice (the only place data enters)
    P = data.price_usd.loc[:asof]
    PL = data.price_local.loc[:asof]
    V = data.volume.loc[:asof]
    B = data.bench.loc[:asof]
    rf_series = data.tbill.loc[:asof]
    rf_now = float(rf_series.dropna().iloc[-1])
    tickers = list(universe or P.columns)
    gates, notes = [], []
    rows = {t: dict(ticker=t, country=data.meta.loc[t, "country"], sector=data.meta.loc[t, "sector"]) for t in tickers}
    alive = list(tickers)

    def run_gate(name, fn):
        nonlocal alive
        keep, cut = [], []
        for t in alive:
            ok, why = fn(t)
            (keep if ok else cut).append((t, why))
        gates.append(dict(gate=name, n_in=len(alive), n_out=len(keep), cut=[(t, w) for t, w in cut]))
        alive = [t for t, _ in keep]

    missing_ok = cfg.missing_fundamentals == "skip"
    FUND = {t: data.fundamentals_asof(t, asof, cfg.fundamental_lag_days) for t in tickers}

    def fval(t, field_, i=0):
        f = FUND[t]
        if f.empty or field_ not in f.columns or len(f) <= i:
            return None
        v = f[field_].iloc[i]
        return None if pd.isna(v) else float(v)

    # ---------- A1 data sufficiency
    def a1(t):
        p = P[t].dropna()
        if len(p) < cfg.min_months + 1:
            return False, f"only {len(p)} months of prices"
        r = p.pct_change().dropna().iloc[-cfg.lookback_months:]
        if (r == 0).mean() > cfg.max_zero_return_share:
            return False, "stale prices"
        if FUND[t].empty and not missing_ok:
            return False, "no published annual report yet"
        return True, ""
    run_gate("A1 Data sufficiency", a1)

    # market value from point-in-time share count x local price (unit-free ratios need no FX)
    for t in alive:
        sh = fval(t, "Ordinary Shares Number") or fval(t, "Diluted Average Shares")
        px_usd = float(P[t].dropna().iloc[-1])
        sccy = str(data.meta.loc[t].get("financial_currency", data.meta.loc[t, "currency"])).upper()
        fx_s = 1.0 if sccy == "USD" else float(data.fx[sccy].loc[:asof].dropna().iloc[-1])
        # market value expressed in the currency the accounts are reported in, so ratios are like-for-like
        rows[t].update(mcap_local=sh * px_usd * fx_s if sh else None, mcap_usd_b=sh * px_usd / 1e9 if sh else None)
        ni, eq, rev = fval(t, "Net Income"), fval(t, "Stockholders Equity"), fval(t, "Total Revenue")
        mc = rows[t]["mcap_local"]
        rows[t]["pe"] = mc / ni if (mc and ni and ni > 0) else (-1.0 if ni is not None and ni <= 0 else None)
        rows[t]["pb"] = mc / eq if (mc and eq and eq > 0) else None
        rows[t]["roe"] = ni / eq if (ni is not None and eq) else None
        op = fval(t, "Operating Income")
        rows[t]["op_margin"] = (op if op is not None else ni) / rev if (rev and (op is not None or ni is not None)) else None
        fcf = fval(t, "Free Cash Flow")
        rows[t]["fcf_yield"] = fcf / mc if (fcf is not None and mc) else None
        revs = [fval(t, "Total Revenue", i) for i in range(4)]
        revs = [x for x in revs if x]
        rows[t]["rev_cagr"] = (revs[0] / revs[-1]) ** (1 / (len(revs) - 1)) - 1 if len(revs) >= 2 and revs[-1] > 0 else None

    # ---------- A2 size & liquidity
    def a2(t):
        m = rows[t]["mcap_usd_b"]
        if m is None:
            return (True, "") if missing_ok else (False, "share count unknown")
        if m < cfg.min_mcap_usd_b:
            return False, f"market cap ${m:.2f}B < ${cfg.min_mcap_usd_b:.0f}B"
        val = (V[t] * P[t]).dropna().iloc[-12:].median() / 1e6
        if val < cfg.min_median_monthly_value_usd_m:
            return False, f"traded value ${val:.0f}M/month"
        return True, ""
    run_gate("A2 Size & liquidity", a2)

    # ---------- A3 valuation sanity
    def is_fin(t):
        return rows[t]["sector"] in cfg.financial_sectors
    def peer_key(t):
        same = [u for u in alive if rows[u]["sector"] == rows[t]["sector"] and rows[u]["country"] == rows[t]["country"]]
        return (rows[t]["sector"], rows[t]["country"]) if len(same) >= cfg.min_peers_for_country_group else (rows[t]["sector"],)
    def peers(t):
        k = peer_key(t)
        return [u for u in alive if (rows[u]["sector"],) + ((rows[u]["country"],) if len(k) == 2 else ()) == k]
    def a3(t):
        metric = "pb" if is_fin(t) else "pe"
        v = rows[t][metric]
        if v is None:
            return (True, "") if missing_ok else (False, f"{metric.upper()} unavailable")
        if v <= 0:
            return False, "no profits (P/E not meaningful)" if metric == "pe" else "negative book value"
        cap = cfg.pb_hard_cap_financials if metric == "pb" else cfg.pe_hard_cap
        if v > cap:
            return False, f"{metric.upper()} {v:.1f} > cap {cap:.0f}"
        pv = [rows[u][metric] for u in peers(t) if rows[u][metric] and rows[u][metric] > 0]
        med = float(np.median(pv)) if pv else v
        if v > cfg.pe_relative_cap * med:
            return False, f"{metric.upper()} {v:.1f} > {cfg.pe_relative_cap:.0f}x peer median {med:.1f}"
        return True, ""
    run_gate("A3 Valuation sanity", a3)

    # ---------- A4 solvency
    def a4(t):
        if is_fin(t):
            return True, ""
        debt = fval(t, "Total Debt") or 0.0
        cash = fval(t, "Cash Cash Equivalents And Short Term Investments") or fval(t, "Cash And Cash Equivalents") or 0.0
        ebitda = fval(t, "EBITDA") or fval(t, "Operating Cash Flow")
        nd = debt - cash
        rows[t]["nd_ebitda"] = nd / ebitda if ebitda else None
        if ebitda is None or ebitda <= 0:
            return (True, "") if missing_ok else (False, "no EBITDA")
        if nd / ebitda > cfg.max_net_debt_to_ebitda:
            return False, f"net debt {nd / ebitda:.1f}x EBITDA"
        op, ie = fval(t, "Operating Income"), fval(t, "Interest Expense")
        if op is not None and ie and abs(ie) > 0 and nd > 0:
            cover = op / abs(ie)
            if cover < cfg.min_interest_cover:
                return False, f"interest cover {cover:.1f}x"
        return True, ""
    run_gate("A4 Solvency", a4)

    # ---------- A5 earnings quality (Piotroski F-score, scaled to available signals; Sloan accruals)
    def piotroski(t):
        g = lambda f, i: fval(t, f, i)
        pts, n = 0, 0
        def add(cond):
            nonlocal pts, n
            if cond is not None:
                n += 1
                pts += int(bool(cond))
        ni0, ni1, ta0, ta1 = g("Net Income", 0), g("Net Income", 1), g("Total Assets", 0), g("Total Assets", 1)
        cfo0 = g("Operating Cash Flow", 0)
        add(ni0 > 0 if ni0 is not None else None)
        add(cfo0 > 0 if cfo0 is not None else None)
        add((ni0 / ta0 > ni1 / ta1) if None not in (ni0, ni1, ta0, ta1) else None)
        add(cfo0 > ni0 if None not in (cfo0, ni0) else None)
        l0, l1 = g("Long Term Debt", 0), g("Long Term Debt", 1)
        add((l0 / ta0 <= l1 / ta1) if None not in (l0, l1, ta0, ta1) else None)
        ca0, cl0, ca1, cl1 = g("Current Assets", 0), g("Current Liabilities", 0), g("Current Assets", 1), g("Current Liabilities", 1)
        add((ca0 / cl0 > ca1 / cl1) if None not in (ca0, cl0, ca1, cl1) and cl0 and cl1 else None)
        s0, s1 = g("Ordinary Shares Number", 0), g("Ordinary Shares Number", 1)
        add(s0 <= s1 * 1.01 if None not in (s0, s1) else None)
        gp0, gp1, r0, r1 = g("Gross Profit", 0), g("Gross Profit", 1), g("Total Revenue", 0), g("Total Revenue", 1)
        add((gp0 / r0 > gp1 / r1) if None not in (gp0, gp1, r0, r1) else None)
        add((r0 / ta0 > r1 / ta1) if None not in (r0, r1, ta0, ta1) else None)
        return (pts / n * 9 if n >= 5 else None), n
    def a5(t):
        f, n = piotroski(t)
        rows[t]["piotroski"] = f
        if f is None:
            return (True, "") if missing_ok else (False, "not enough statement history for F-score")
        if f < cfg.min_piotroski:
            return False, f"Piotroski F-score {f:.1f}/9"
        ni, cfo, ta = fval(t, "Net Income"), fval(t, "Operating Cash Flow"), fval(t, "Total Assets")
        if not is_fin(t) and None not in (ni, cfo, ta) and ta:
            acc = (ni - cfo) / ta
            if acc > cfg.max_accruals:
                return False, f"accruals {acc:.0%} of assets"
        return True, ""
    run_gate("A5 Earnings quality", a5)

    # ---------- price statistics (lookback window, USD, monthly)
    rf_m = rf_series.reindex(P.index).ffill().shift(1) / 12           # T-bill known at start of month
    rb = B.pct_change()
    R = {}
    for t in alive:
        p = P[t].dropna()
        r = p.pct_change().dropna().iloc[-cfg.lookback_months:]
        ex = (r - rf_m.reindex(r.index).fillna(rf_now / 12)).values
        bb = rb.reindex(r.index)
        ok = bb.notna().values
        beta = float(np.cov(r.values[ok], bb.values[ok])[0, 1] / np.var(bb.values[ok], ddof=1)) if ok.sum() > 24 else 1.0
        pw = p.iloc[-cfg.lookback_months - 1:].values
        rows[t].update(psr=psr(ex), sortino=sortino(ex), cvar95=cvar(r.values), max_dd=max_drawdown(pw),
                       beta=beta, vol=float(r.std(ddof=1) * math.sqrt(12)),
                       mom_12_1=float(p.iloc[-2] / p.iloc[-13] - 1),
                       above_10m=bool(p.iloc[-1] > p.iloc[-10:].mean()),
                       hist_mean=float(r.mean() * 12), hist_mean_se2=float(r.var(ddof=1) / len(r) * 144))
        R[t] = r

    # ---------- A6 trend
    def a6(t):
        if rows[t]["above_10m"] or rows[t]["mom_12_1"] > 0:
            return True, ""
        return False, f"below 10-month average and 12-1 momentum {rows[t]['mom_12_1']:.0%}"
    run_gate("A6 Trend", a6)

    # ---------- A7 skill not luck
    psr_bar = cfg.min_psr
    if cfg.psr_mode == "rank" and alive:              # v3: the best third always gets through, even after a crash
        psr_bar = max(cfg.psr_floor, float(np.quantile([rows[t]["psr"] for t in alive], 1 - cfg.psr_top_share)))
    def a7(t):
        if rows[t]["psr"] < psr_bar:
            return False, f"PSR {rows[t]['psr']:.2f} (only {rows[t]['psr']:.0%} sure Sharpe > 0)"
        if rows[t]["sortino"] < cfg.min_sortino:
            return False, f"Sortino {rows[t]['sortino']:.2f}"
        return True, ""
    run_gate("A7 Skill, not luck (PSR, Sortino)", a7)

    # ---------- A8 tail risk
    bw = B.dropna().iloc[-cfg.lookback_months - 1:]
    bench_dd = max_drawdown(bw.values) if len(bw) > cfg.lookback_months * 0.9 else None
    def a8(t):
        r_ = rows[t]
        if r_["cvar95"] < cfg.min_cvar95_monthly:
            return False, f"CVaR95 {r_['cvar95']:.0%} a month"
        if cfg.drawdown_mode == "relative" and bench_dd is not None:   # v3: judged against the market's own fall
            if r_["max_dd"] - bench_dd < cfg.max_dd_vs_market:
                return False, f"max drawdown {r_['max_dd']:.0%} vs market {bench_dd:.0%}"
        elif r_["max_dd"] < cfg.min_max_drawdown:
            return False, f"max drawdown {r_['max_dd']:.0%}"
        if r_["beta"] > cfg.max_beta:
            return False, f"beta {r_['beta']:.2f}"
        return True, ""
    run_gate("A8 Tail risk", a8)

    # ---------- expected returns: James-Stein shrunk history + fundamental implied + CAPM
    def expected_returns(ts):
        if not ts:
            return
        m = np.array([rows[t]["hist_mean"] for t in ts])
        se2 = np.array([rows[t]["hist_mean_se2"] for t in ts])
        grand = m.mean()
        spread = ((m - grand) ** 2).sum()
        c = max(0.0, 1 - (len(ts) - 3) * se2.mean() / spread) if len(ts) > 3 and spread > 0 else 0.0
        for t, mi in zip(ts, m):
            hist = grand + c * (mi - grand)
            g = min(max(rows[t]["rev_cagr"] or 0.0, 0.0), cfg.growth_cap)
            y = rows[t]["fcf_yield"] if (rows[t]["fcf_yield"] is not None and not is_fin(t)) else \
                (1 / rows[t]["pe"] if rows[t].get("pe") and rows[t]["pe"] > 0 else None)
            fund = (y + g) if y is not None else hist
            capm = rf_now + rows[t]["beta"] * cfg.equity_risk_premium
            rows[t].update(mu_hist=hist, mu_fund=fund, mu_capm=capm, js_shrink=c,
                           exp_return=cfg.w_hist * hist + cfg.w_fund * fund + cfg.w_capm * capm)
    expected_returns(alive)

    # ---------- A9 Monte Carlo edge
    H = cfg.horizon_years * 12
    def a9(t):
        e = rows[t]["exp_return"]
        if e <= rf_now:
            return False, f"expected return {e:.1%} <= T-bill {rf_now:.1%}"
        r = R[t].values
        rc = r - r.mean() + e / 12
        idx = stationary_bootstrap_idx(len(rc), cfg.stock_bootstrap_paths, H, cfg.block_length, rng)
        g = np.prod(1 + np.maximum(rc[idx], -0.99), axis=1)
        p = float(np.mean(g > (1 + rf_now) ** cfg.horizon_years))
        rows[t]["p_beat_cash"] = p
        if p < cfg.min_p_beat_cash:
            return False, f"P(beat T-bills over {cfg.horizon_years}y) {p:.0%}"
        return True, ""
    run_gate("A9 Monte Carlo edge", a9)

    table = pd.DataFrame(rows.values()).set_index("ticker")
    if not alive:
        return Result(asof, cfg, gates, table, {CASH: 1.0} if cfg.allow_cash else {}, {}, "NO STOCKS PASSED", notes)

    # ---------- B1 composite score
    T = alive
    col = lambda k: np.array([rows[t].get(k) if rows[t].get(k) is not None else np.nan for t in T], float)
    pillars = {
        "quality": (pct_rank(col("roe")) + pct_rank(col("op_margin")) + pct_rank(col("piotroski"))) / 3,
        "value": (pct_rank(1 / col("pe")) + pct_rank(col("fcf_yield"))) / 2,
        "growth": pct_rank(col("rev_cagr")),
        "momentum": pct_rank(col("mom_12_1")),
        "risk": (pct_rank(col("psr")) + pct_rank(col("sortino")) + pct_rank(col("cvar95"))) / 3,
        "exp_return": pct_rank(col("exp_return")),
    }
    score = 100 * sum(cfg.pillar_weights[k] * pillars[k] for k in cfg.pillar_weights)
    for t, s in zip(T, score):
        rows[t]["score"] = float(s)
    ranked = sorted(T, key=lambda t: -rows[t]["score"])
    kept = ranked[: cfg.top_n]
    gates.append(dict(gate="B1 Composite rank", n_in=len(T), n_out=len(kept),
                      cut=[(t, f"rank {ranked.index(t) + 1} (score {rows[t]['score']:.0f})") for t in ranked[cfg.top_n:]]))
    alive = kept

    # ---------- C1 redundancy via hierarchical clustering
    Rm = pd.DataFrame({t: R[t] for t in alive}).dropna()
    if len(alive) > 1:
        corr = Rm.corr().values
        dist = squareform(np.clip(1 - corr, 0, 2), checks=False)
        lab = hierarchy.fcluster(hierarchy.linkage(dist, "average"), 1 - cfg.max_pair_corr, "distance")
        best = {}
        for t, l in zip(alive, lab):
            if l not in best or rows[t]["score"] > rows[best[l]]["score"]:
                best[l] = t
        survivors = [t for t in alive if t in best.values()]
        gates.append(dict(gate="C1 Redundancy (clusters)", n_in=len(alive), n_out=len(survivors),
                          cut=[(t, f"moves with {best[l]} (corr > {cfg.max_pair_corr})") for t, l in zip(alive, lab) if t not in survivors]))
        alive = survivors

    # ---------- C2-C4 target solver + Monte Carlo
    Rm = pd.DataFrame({t: R[t] for t in alive}).dropna()
    X = Rm.values
    mu = np.array([rows[t]["exp_return"] for t in alive])
    cov = ledoit_wolf(X) * 12
    sect = [rows[t]["sector"] for t in alive]
    ctry = [rows[t]["country"] for t in alive]
    # cash (T-bills) is only allowed when the target is BELOW what the least-risky all-stock mix earns;
    # above that, cash just dilutes a return-seeking sleeve ("auto" mode)
    if cfg.allow_cash == "auto":
        ones = np.ones(len(alive))
        try:
            ic = np.linalg.solve(cov, ones); w_mv = np.clip(ic / ic.sum(), 0, None); w_mv /= w_mv.sum()
            g_mv = w_mv @ mu - 0.5 * w_mv @ cov @ w_mv
        except np.linalg.LinAlgError:
            g_mv = float(mu.min())
        use_cash = bool(cfg.target_cagr < g_mv)
    else:
        use_cash = bool(cfg.allow_cash)

    def solve(m_req, mu_, cov_):
        k = len(alive)
        nvar = k + (1 if use_cash else 0)
        mu_full = np.append(mu_, rf_now) if use_cash else mu_
        cov_full = np.zeros((nvar, nvar))
        cov_full[:k, :k] = cov_
        ones = np.ones(nvar)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: ones},
                {"type": "ineq", "fun": lambda w: w @ mu_full - m_req, "jac": lambda w: mu_full}]
        for grp, cap in ((sect, cfg.sector_cap), (ctry, cfg.country_cap)):
            for gname in sorted(set(grp)):
                ix = [i for i, x in enumerate(grp) if x == gname]
                g = np.zeros(nvar); g[ix] = -1.0
                cons.append({"type": "ineq", "fun": lambda w, ix=ix, cap=cap: cap - w[ix].sum(), "jac": lambda w, g=g: g})
        bounds = [(0, cfg.name_cap)] * k + ([(0, 1)] if use_cash else [])
        w0 = np.full(nvar, min(1 / nvar, cfg.name_cap))          # start inside the bounds ...
        if use_cash:
            w0[k] = 1 - w0[:k].sum()                             # ... and summing to one
        sol = optimize.minimize(lambda w: w @ cov_full @ w, w0, jac=lambda w: 2 * cov_full @ w, method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if not sol.success or sol.x @ mu_full < m_req - 1e-4:
            return None
        w = np.clip(sol.x, 0, None)
        return w / w.sum()

    def max_mu(mu_):
        k = len(alive)
        nvar = k + (1 if use_cash else 0)
        c = -(np.append(mu_, rf_now) if use_cash else mu_)
        A, b = [], []
        for grp, cap in ((sect, cfg.sector_cap), (ctry, cfg.country_cap)):
            for gname in set(grp):
                row = np.zeros(nvar)
                row[[i for i, x in enumerate(grp) if x == gname]] = 1
                A.append(row); b.append(cap)
        bounds = [(0, cfg.name_cap)] * k + ([(0, 1)] if use_cash else [])
        lp = optimize.linprog(c, A_ub=np.array(A), b_ub=b, A_eq=np.ones((1, nvar)), b_eq=[1], bounds=bounds)
        return -lp.fun if lp.success else None

    def evaluate(w, paths, seed):
        r_ = np.random.default_rng(seed)
        k = len(alive)
        ws, wc = w[:k], (w[k] if use_cash else 0.0)
        sims = simulate_portfolio(X, mu / 12, cov / 12, ws, H, paths, cfg, r_, rf_now / 12, wc)
        thr = (1 + cfg.target_cagr) ** cfg.horizon_years
        p_hit = min(float(np.mean(s["growth"] >= thr)) for s in sims.values())
        p_dd = max(float(np.mean(s["max_dd"] < cfg.max_drawdown_tolerance)) for s in sims.values())
        g_all = np.concatenate([s["growth"] for s in sims.values()])
        return p_hit, p_dd, g_all, sims

    hi = max_mu(mu)
    feasible = hi is not None and len(alive) * cfg.name_cap + (1 if use_cash else 0) >= 1
    if not feasible:
        return Result(asof, cfg, gates, pd.DataFrame(rows.values()).set_index("ticker"), {CASH: 1.0} if use_cash else {},
                      {}, "NOT ENOUGH STOCKS TO SATISFY CAPS", notes)
    lo = rf_now if use_cash else float(mu.min())
    candidates = []
    for m_req in np.linspace(lo, hi, cfg.grid_points):
        w = solve(m_req, mu, cov)
        if w is None:
            continue
        p_hit, p_dd, _, _ = evaluate(w, cfg.screen_paths, cfg.seed)
        candidates.append(dict(m=m_req, w=w, p_hit=p_hit, p_dd=p_dd))
    if not candidates:
        return Result(asof, cfg, gates, pd.DataFrame(rows.values()).set_index("ticker"), {CASH: 1.0} if use_cash else {},
                      {}, "SOLVER FAILED", notes)
    k = len(alive)

    def enforce_caps(w):
        """Project weights back inside the name / sector / country caps after pruning (excess goes to
        names with room, pro rata; to cash only if nothing else has room)."""
        w = w.copy()
        groups = [(np.array([i]), cfg.name_cap) for i in range(k)]
        for grp, cap in ((sect, cfg.sector_cap), (ctry, cfg.country_cap)):
            for gname in set(grp):
                groups.append((np.array([i for i, x in enumerate(grp) if x == gname]), cap))
        for _ in range(200):
            over = [(ix, cap) for ix, cap in groups if w[ix].sum() > cap + 1e-9]
            if not over:
                break
            ix, cap = over[0]
            excess = w[ix].sum() - cap
            w[ix] *= cap / w[ix].sum()
            room = np.zeros(len(w), bool)
            for i in range(k):
                if w[i] > 0 and i not in ix and all(w[g].sum() < c_ - 1e-9 for g, c_ in groups if i in g):
                    room[i] = True
            if use_cash:
                room[k] = True if not room[:k].any() else room[k] or w[k] > 0
            base = np.where(room, np.maximum(w, 1e-12 * room), 0)
            if base.sum() <= 0:
                break
            w = w + excess * base / base.sum()
        return w / w.sum()

    def resample(c, seed):
        """C3 Michaud resampling: re-estimate inputs on block-bootstrapped history, re-solve at the same
        required return, average the weights, prune tiny positions, re-apply caps, re-run the full Monte Carlo."""
        r_ = np.random.default_rng(seed)
        acc, ok_n = np.zeros(len(c["w"])), 0
        for _ in range(cfg.resamples):
            idx = stationary_bootstrap_idx(len(X), 1, len(X), cfg.block_length, r_)[0]
            Xb = X[idx]
            mu_b = mu + (Xb.mean(0) - X.mean(0)) * 12 * cfg.w_hist       # only the history-based part is resampled
            w = solve(c["m"], mu_b, ledoit_wolf(Xb) * 12)
            if w is not None:
                acc += w; ok_n += 1
        w_res = acc / ok_n if ok_n else c["w"].copy()
        w_res[:k] = np.where(w_res[:k] < cfg.min_weight, 0, w_res[:k])
        w_res = w_res / w_res.sum()
        w_res = enforce_caps(w_res)
        p_hit, p_dd, g_all, sims = evaluate(w_res, cfg.mc_paths, cfg.seed + 1)
        return dict(c=c, w=w_res, p_hit=p_hit, p_dd=p_dd, g=g_all, sims=sims, ok_n=ok_n,
                    turnover=float(np.abs(w_res - c["w"]).sum() / 2))

    # C2: walk up the efficient frontier; the first (lowest-risk) portfolio that still clears C3 and C4
    # AFTER resampling wins. If none does, keep the one with the best odds of hitting the target.
    passing = [c for c in candidates if c["p_hit"] >= cfg.confidence and c["p_dd"] <= cfg.max_drawdown_prob]
    final, status = None, ""
    for c in passing:
        res_c = resample(c, cfg.seed + 7)
        if res_c["p_hit"] >= cfg.confidence and res_c["p_dd"] <= cfg.max_drawdown_prob:
            final, status = res_c, "TARGET MET"
            break
        if final is None or res_c["p_hit"] > final["p_hit"]:
            final = res_c
    if status != "TARGET MET":
        dd_ok = [c for c in candidates if c["p_dd"] <= cfg.max_drawdown_prob] or candidates
        best_any = None
        for c in sorted(dd_ok, key=lambda c: -c["p_hit"])[:3]:
            res_c = resample(c, cfg.seed + 7)
            if res_c["p_dd"] <= cfg.max_drawdown_prob and (final is None or res_c["p_hit"] > final["p_hit"]):
                final = res_c
            if best_any is None or res_c["p_dd"] < best_any["p_dd"]:
                best_any = res_c
        if final is None:                       # nothing clears the drawdown budget even after resampling:
            final = best_any                    # show the lowest-drawdown option (C5 will report the fail)
        status = "TARGET NOT MET AT REQUIRED CONFIDENCE - best available shown"
    choice, w_res = final["c"], final["w"]
    p_hit, p_dd, g_all, sims, turnover = final["p_hit"], final["p_dd"], final["g"], final["sims"], final["turnover"]
    gates.append(dict(gate="C2 Target solver", n_in=len(alive), n_out=len(alive), cut=[],
                      info=dict(required_arith_return=choice["m"], max_feasible=hi,
                                grid=[(round(c["m"], 4), round(c["p_hit"], 3), round(c["p_dd"], 3)) for c in candidates])))
    dropped = [alive[i] for i in range(k) if w_res[i] == 0]
    added = [alive[i] for i in range(k) if choice["w"][i] < cfg.min_weight and w_res[i] > 0]
    gates.append(dict(gate="C3 Resampled stability", n_in=len(alive), n_out=int((w_res[:k] > 0).sum()),
                      cut=[(t, f"average weight < {cfg.min_weight:.0%} across {final['ok_n']} resamples") for t in dropped],
                      info=dict(turnover_vs_raw=turnover, resamples_ok=final["ok_n"], added=added, dropped=dropped)))

    # ---------- C3 / C4 / C6 confirm on final weights
    names = int((w_res[:k] > 0).sum())
    sec_w = pd.Series(w_res[:k], index=sect).groupby(level=0).sum()
    cty_w = pd.Series(w_res[:k], index=ctry).groupby(level=0).sum()
    conc_ok = names >= cfg.min_names and sec_w.max() <= cfg.sector_cap + 1e-6 and cty_w.max() <= cfg.country_cap + 1e-6
    gates.append(dict(gate="C4 Monte Carlo confidence", n_in=names, n_out=names if p_hit >= cfg.confidence else 0, cut=[],
                      info=dict(p_hit=p_hit, need=cfg.confidence)))
    gates.append(dict(gate="C5 Drawdown budget", n_in=names, n_out=names if p_dd <= cfg.max_drawdown_prob else 0, cut=[],
                      info=dict(p_dd=p_dd, limit=cfg.max_drawdown_prob)))
    gates.append(dict(gate="C6 Concentration", n_in=names, n_out=names if conc_ok else 0, cut=[],
                      info=dict(names=names, max_sector=float(sec_w.max()), max_country=float(cty_w.max()))))
    if not conc_ok:
        notes.append(f"concentration check: {names} names, max sector {sec_w.max():.0%}, max country {cty_w.max():.0%}")

    cagr = g_all ** (1 / cfg.horizon_years) - 1
    mc = dict(p_hit=p_hit, p_dd=p_dd, cagr_p10=float(np.percentile(cagr, 10)), cagr_p50=float(np.median(cagr)),
              cagr_p90=float(np.percentile(cagr, 90)), p_loss=float(np.mean(g_all < 1)),
              scenarios={kk: min(float(np.mean(s["growth"] >= (1 + v) ** cfg.horizon_years)) for s in sims.values())
                         for kk, v in cfg.scenario_targets.items()},
              exp_return=float(w_res[:k] @ mu + (w_res[k] * rf_now if use_cash else 0)),
              vol=float(math.sqrt(w_res[:k] @ cov @ w_res[:k])),
              engines={n_: float(np.mean(s["growth"] >= (1 + cfg.target_cagr) ** cfg.horizon_years)) for n_, s in sims.items()},
              cagr_samples=cagr[:: max(1, len(cagr) // 4000)].tolist())
    weights = {alive[i]: float(w_res[i]) for i in range(k) if w_res[i] > 0}
    if use_cash and w_res[k] > 1e-4:
        weights[CASH] = float(w_res[k])
    return Result(asof, cfg, gates, pd.DataFrame(rows.values()).set_index("ticker"), weights, mc, status, notes)


# =============================================================================== backtest (walk-forward)
def backtest(data: MarketData, start, end, cfg: Config | None = None, rebalance_months: int = 3,
             cost_bps: float = 25.0, verbose: bool = True) -> dict:
    """Walk forward: at each rebalance date choose weights with select() using only data up to that date,
    hold them (drifting) until the next rebalance, charge `cost_bps` on turnover. Returns monthly series."""
    cfg = cfg or Config()
    months = data.price_usd.loc[start:end].index
    rets = data.price_usd.pct_change()
    rf_m = data.tbill.reindex(data.price_usd.index).ffill().shift(1) / 12
    nav, out, hist = 1.0, [], []
    w_prev = {}
    for i in range(0, len(months) - 1, rebalance_months):
        asof = months[i]
        res = select(data, asof, cfg)
        w = dict(res.weights)
        turn = sum(abs(w.get(t, 0) - w_prev.get(t, 0)) for t in set(w) | set(w_prev)) / 2
        nav *= 1 - turn * cost_bps / 1e4
        hist.append(dict(asof=asof, status=res.status, weights=w, p_hit=res.mc.get("p_hit"), turnover=turn))
        if verbose:
            print(f"{asof.date()}  {res.status[:28]:<28s}  names {len([t for t in w if t != CASH]):2d}  "
                  f"cash {w.get(CASH, 0):5.1%}  P(hit) {res.mc.get('p_hit', float('nan')):.0%}  turnover {turn:.0%}")
        cur = dict(w)
        for m in months[i + 1: i + 1 + rebalance_months]:
            r_t = {t: (rf_m.loc[m] if t == CASH else rets.loc[m, t]) for t in cur}
            r_t = {t: (0.0 if pd.isna(v) else float(v)) for t, v in r_t.items()}
            port = sum(cur[t] * r_t[t] for t in cur)
            nav *= 1 + port
            out.append(dict(date=m, ret=port, nav=nav))
            tot = sum(cur[t] * (1 + r_t[t]) for t in cur) or 1.0
            cur = {t: cur[t] * (1 + r_t[t]) / tot for t in cur}            # weights drift between rebalances
        w_prev = cur
    s = pd.DataFrame(out).set_index("date")
    yrs = len(s) / 12
    bench = data.bench.pct_change().reindex(s.index)
    ex = s["ret"] - rf_m.reindex(s.index)
    stats_ = dict(cagr=s["nav"].iloc[-1] ** (1 / yrs) - 1, vol=s["ret"].std() * math.sqrt(12),
                  sharpe=ex.mean() / ex.std() * math.sqrt(12), max_dd=max_drawdown(np.r_[1.0, s["nav"].values]),
                  bench_cagr=(1 + bench).prod() ** (1 / yrs) - 1)
    return dict(series=s, rebalances=hist, stats=stats_)


if __name__ == "__main__":
    import sys, json
    here = os.path.dirname(os.path.abspath(__file__))
    data = MarketData(os.path.join(here, "data"))
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    res = select(data, "2026-09-30", Config(target_cagr=target))
    res.print_report()
