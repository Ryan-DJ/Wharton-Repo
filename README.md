# Laura Gao strategy: Funnel v3 code

v3 is the default. `Plan()` uses the v3 phase settings; `Plan(phases=default_phases())` gives the old v2 settings.

## Setup (once)
```
pip install yfinance pandas numpy scipy scikit-learn matplotlib reportlab openpyxl pytest
python -m pytest -q          # 33 tests, all should pass
```

## Files
| File | What it is |
|---|---|
| `funnel_v2.py` | The stock-picking algorithm: gates A1-A9, rank B1, portfolio gates C1-C6, Monte Carlo, data loading. (The file name is historical; it holds the v3 options.) |
| `laura_strategy.py` | Laura's plan: the five phases, gate M1 (market switch), the bond ladder, the 2031 split / 2033 gift / $50k payments, and the full-plan backtest. `v3_phases()` = final settings |
| `us_backtest.py` | Real-market backtests on the S&P 500 as it stood on each date (`data_us/`) |
| `run_v3.py` | One full-plan replay: `python run_v3.py v3 2001-12-31` (or `v2`) |
| `phase_backtest.py` | Each phase on its own over its real length |
| `build_us_universe.py` | Builds `data_us/` from the S&P 500 membership history and Yahoo prices |
| `fetch_data.py` | Downloads prices, statements, bonds and index funds for any ticker list (`python fetch_data.py my_universe.csv`) |
| `case_study.py`, `enrich_case.py` | Replays the 31 Dec 2003 decision used in the PDF's case studies |
| `build_results_xlsx.py` | Rebuilds the results spreadsheet from saved runs |
| `test_*.py` | Tests: no look-ahead, caps hold, the spreadsheet plan reproduced to the cent, M1 and bond ladder behave as described |
| `data/` | 41-stock demo universe with company statements (live demo) |
| `data_us/` | S&P 500 point-in-time universe, 1990-2026 monthly prices (backtests) |

## Common runs
```
python run_v3.py v3 2006-12-31                    # whole plan, history from 2007 playing 2027-2042
python us_backtest.py phases growth post_gift     # each phase on its own, every start year (slow: ~1 hr)
python laura_strategy.py                          # what each phase would hold today (demo universe)
```
A start date is the month-end that plays the role of Jan 2027; it needs 16 years of data after it (latest: 2009-12-31).

## Limits
- Free data: Yahoo lacks prices for some failed companies. The members we have beat the real equal-weight S&P 500 by about 1.5% a year.
- Company statements exist only from 2021, so in backtests gates A3-A5 let every stock through.
- The foreign part holds an international index fund in backtests (no free list of past foreign index members).
