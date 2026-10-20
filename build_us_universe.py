"""
build_us_universe.py - turn the downloaded S&P 500 history into a data folder the Funnel can read.

Input : ../sp500/repo/S&P 500 Historical Components & Changes (Updated).csv  (members on every date since 1996)
        ../sp500/dl/close.csv, volume.csv                                      (monthly prices from Yahoo)
Output: data_us/  prices_local.csv, volume.csv, meta.csv, membership.csv, + T-bill, bond/index funds,
        benchmark (SPY) copied from data/. No company statements: in backtests the statement-based gates
        (A3-A5) are skipped, exactly as before 2022 in the demo data.
"""
import os, shutil
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "sp500")
OUT = os.path.join(HERE, "data_us")


def main():
    os.makedirs(OUT, exist_ok=True)
    mem = pd.read_csv(os.path.join(SRC, "repo", "S&P 500 Historical Components & Changes (Updated).csv"), parse_dates=["date"])
    C = pd.read_csv(os.path.join(SRC, "dl", "close.csv"), index_col=0, parse_dates=True)
    V = pd.read_csv(os.path.join(SRC, "dl", "volume.csv"), index_col=0, parse_dates=True)
    for df in (C, V):
        df.index = df.index.to_period("M").to_timestamp("M")
    C, V = C[~C.index.duplicated(keep="last")], V[~V.index.duplicated(keep="last")]
    C.columns = [c.replace("-", ".") for c in C.columns]; V.columns = [c.replace("-", ".") for c in V.columns]
    C = C.loc[:, C.notna().sum() >= 12]
    V = V.reindex(columns=C.columns)
    # the last row is a partial month; drop anything after the last full month-end
    C, V = C.loc[:"2026-09-30"], V.loc[:"2026-09-30"]
    C.to_csv(os.path.join(OUT, "prices_local.csv")); V.to_csv(os.path.join(OUT, "volume.csv"))
    # month-end membership: members on the last trading day of each month
    mem = mem.set_index("date").sort_index()
    me = mem.groupby(mem.index.to_period("M")).last()
    me.index = me.index.to_timestamp("M")
    me.to_csv(os.path.join(OUT, "membership.csv"))
    # sectors: current members from the Wikipedia list; others from Yahoo if fetched (sectors.csv), else Unknown
    sec = pd.read_csv(os.path.join(SRC, "repo", "sp500.csv")).set_index("Symbol")["GICS Sector"].to_dict()
    extra = os.path.join(SRC, "sectors.csv")
    if os.path.exists(extra):
        sec.update({k: v for k, v in pd.read_csv(extra).set_index("ticker")["sector"].dropna().to_dict().items() if k not in sec})
    meta = pd.DataFrame(dict(ticker=C.columns, name=C.columns, country="United States",
                             sector=[sec.get(t, "Unknown") for t in C.columns], currency="USD", financial_currency="USD"))
    meta.to_csv(os.path.join(OUT, "meta.csv"), index=False)
    pd.DataFrame(columns=["ticker", "period_end", "field", "value"]).to_csv(os.path.join(OUT, "fundamentals.csv"), index=False)
    for f in ("tbill.csv", "fx_per_usd.csv", "funds.csv"):
        shutil.copy(os.path.join(HERE, "data", f), os.path.join(OUT, f))
    spy = pd.read_csv(os.path.join(HERE, "data", "funds.csv"), index_col=0, parse_dates=True)["SPY"].dropna()
    spy.rename("Close").to_csv(os.path.join(OUT, "benchmark.csv"))          # beta measured against the S&P 500
    all_members = sorted({t for x in mem.tickers for t in str(x).split(",")})
    print(f"members ever: {len(all_members)}  with prices: {len(C.columns)}  "
          f"sectors known: {(meta.sector != 'Unknown').sum()}")


def universe_fn(folder=OUT):
    """asof -> list of S&P 500 members on that date (latest month-end membership at or before asof)."""
    me = pd.read_csv(os.path.join(folder, "membership.csv"), index_col=0, parse_dates=True)["tickers"]
    lists = {d: str(x).split(",") for d, x in me.items()}
    dates = sorted(lists)
    def at(asof):
        asof = pd.Timestamp(asof)
        d = max([x for x in dates if x <= asof], default=dates[0])
        return lists[d]
    return at


if __name__ == "__main__":
    main()
