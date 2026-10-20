"""
fetch_data.py - download prices, FX and annual fundamentals for ANY list of tickers (Yahoo symbols)
and save them as plain CSVs that funnel_v2.py reads. Run once; the algorithm and the backtest then
work offline from ./data.

    python3 fetch_data.py                 # uses UNIVERSE below
    python3 fetch_data.py my_list.csv     # csv with columns: ticker,country

Yahoo suffixes: .T Tokyo, .TW Taiwan, .DE Germany, .NS India, .L London, .PA Paris, .HK Hong Kong,
.KS Korea, .AX Australia, .TO Toronto, .SA Brazil, .MX Mexico, none = U.S.
"""
import os, sys, time, csv
import pandas as pd
import yfinance as yf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Demo universe: large caps across sectors in six markets (Part 1 will supply the real country list)
UNIVERSE = {
    "United States": ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "CAT", "UNH", "HD", "COST"],
    "Japan":         ["7203.T", "6758.T", "8306.T", "9984.T", "6861.T", "9432.T", "4502.T", "8035.T"],
    "Taiwan":        ["2330.TW", "2317.TW", "2454.TW", "2412.TW", "2881.TW"],
    "Germany":       ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MBG.DE", "BAS.DE"],
    "India":         ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "HINDUNILVR.NS"],
    "United Kingdom": ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L"],
}
# FX symbols: units of local currency per 1 USD
FX = {"JPY": "JPY=X", "TWD": "TWD=X", "EUR": "EUR=X", "INR": "INR=X", "GBP": "GBP=X",
      "KRW": "KRW=X", "HKD": "HKD=X", "CNY": "CNY=X", "AUD": "AUD=X", "CAD": "CAD=X",
      "BRL": "BRL=X", "MXN": "MXN=X", "CHF": "CHF=X", "SGD": "SGD=X"}
FIELDS = {  # statement -> fields kept
    "income_stmt": ["Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income",
                    "Interest Expense", "Diluted Average Shares"],
    "balance_sheet": ["Total Assets", "Current Assets", "Current Liabilities", "Long Term Debt",
                      "Total Debt", "Cash Cash Equivalents And Short Term Investments",
                      "Cash And Cash Equivalents", "Stockholders Equity", "Ordinary Shares Number"],
    "cashflow": ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
}


def retry(fn, tries=4, wait=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # yahoo rate limits intermittently
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def main(universe):
    os.makedirs(OUT, exist_ok=True)
    prices, volumes, meta, fund = {}, {}, [], []
    for country, tickers in universe.items():
        for t in tickers:
            tk = yf.Ticker(t)
            h = retry(lambda: tk.history(period="max", interval="1mo", auto_adjust=True))
            h = h[["Close", "Volume"]].dropna()
            h.index = h.index.tz_localize(None).to_period("M").to_timestamp("M")
            h = h[~h.index.duplicated(keep="last")]
            prices[t], volumes[t] = h["Close"], h["Volume"]
            info = retry(lambda: tk.info) or {}
            meta.append(dict(ticker=t, name=info.get("shortName", t), country=country,
                             sector=info.get("sector", "Unknown"), currency=info.get("currency", "USD"),
                             financial_currency=info.get("financialCurrency", info.get("currency", "USD"))))
            for stmt, fields in FIELDS.items():
                df = retry(lambda: getattr(tk, stmt))
                if df is None or df.empty:
                    continue
                for f in fields:
                    if f in df.index:
                        for period, v in df.loc[f].items():
                            if pd.notna(v):
                                fund.append(dict(ticker=t, period_end=pd.Timestamp(period).date(), field=f, value=float(v)))
            print(f"{t:14s} {country:15s} {len(h):4d} months  sector={meta[-1]['sector']}  ccy={meta[-1]['currency']}")
            time.sleep(1.0)
    pd.DataFrame(prices).sort_index().to_csv(os.path.join(OUT, "prices_local.csv"))
    pd.DataFrame(volumes).sort_index().to_csv(os.path.join(OUT, "volume.csv"))
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "meta.csv"), index=False)
    pd.DataFrame(fund).to_csv(os.path.join(OUT, "fundamentals.csv"), index=False)
    ccys = sorted(({m["currency"].upper() for m in meta} | {m["financial_currency"].upper() for m in meta}) - {"USD"})
    fx = {}
    for c in ccys:
        sym = FX["GBP" if c == "GBP" else c]
        h = retry(lambda: yf.Ticker(sym).history(period="max", interval="1mo")["Close"])
        h.index = h.index.tz_localize(None).to_period("M").to_timestamp("M")
        fx[c] = h[~h.index.duplicated(keep="last")]
    pd.DataFrame(fx).sort_index().to_csv(os.path.join(OUT, "fx_per_usd.csv"))
    for sym, name in [("ACWI", "benchmark"), ("^IRX", "tbill")]:
        h = retry(lambda: yf.Ticker(sym).history(period="max", interval="1mo", auto_adjust=True)["Close"])
        h.index = h.index.tz_localize(None).to_period("M").to_timestamp("M")
        h[~h.index.duplicated(keep="last")].to_csv(os.path.join(OUT, f"{name}.csv"))
    bonds = {}
    for sym in ("IEF", "SHY", "SPY", "EFA"):   # Treasuries (bond sleeve) + index funds (fallback when a sleeve can't be filled)
        h = retry(lambda: yf.Ticker(sym).history(period="max", interval="1mo", auto_adjust=True)["Close"])
        h.index = h.index.tz_localize(None).to_period("M").to_timestamp("M")
        bonds[sym] = h[~h.index.duplicated(keep="last")]
    f = pd.DataFrame(bonds).sort_index()
    for etf, proxy in (("IEF", "VFITX"), ("EFA", "VGTSX")):   # extend back before the ETFs existed (1991 / 1996)
        h = retry(lambda: yf.Ticker(proxy).history(period="max", interval="1mo", auto_adjust=True)["Close"])
        h.index = h.index.tz_localize(None).to_period("M").to_timestamp("M")
        h = h[~h.index.duplicated(keep="last")]
        f = f.reindex(f.index.union(h.index))
        first = f[etf].first_valid_index()
        f.loc[f.index < first, etf] = (h / h.loc[first] * f.loc[first, etf])[h.index < first]
    f.to_csv(os.path.join(OUT, "funds.csv"))
    print("saved to", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rows = list(csv.DictReader(open(sys.argv[1])))
        u = {}
        for r in rows:
            u.setdefault(r["country"], []).append(r["ticker"])
        main(u)
    else:
        main(UNIVERSE)
