# Prediction Markets: Favorite-Longshot Bias

This project tests whether prices on Kalshi and Polymarket behave like calibrated probabilities, whether low-probability contracts are systematically overpriced, and whether the resulting signal survives an out-of-sample, spread-aware Kelly backtest.

## What is included

- 84,767 resolved binary contracts priced 48 hours before the event
- Brier scores overall, by venue, and by category
- 10-bin calibration and favorite-longshot bias estimates with 95% confidence intervals
- Venue and category comparisons
- A purged chronological train/test split that only learns from outcomes known before testing begins
- Cash-constrained fractional-Kelly simulations with and without a 2-cent spread
- Reproducible CSV tables and publication-ready PNG figures

The original research notebook is preserved in `notebooks/`. The cleaned implementation lives in `src/prediction_market_analysis/`, where validation, statistics, backtesting, and reporting are separated into reusable functions.

## Reproduce the results

```powershell
python -m pip install -e .
prediction-market-analysis
```

Or run without installing the console command:

```powershell
$env:PYTHONPATH = "src"
python -m prediction_market_analysis
```

The command reads `data/prediction_markets_clean.csv` and rebuilds all result tables and figures under `outputs/`.

## Methodology

1. Interpret each YES price as an implied probability and compare it with the binary outcome.
2. Measure accuracy with the Brier score and compare it with a constant base-rate forecast.
3. Bin contracts into 10-percentage-point probability bands. Bias is mean price minus the observed YES frequency; positive values indicate overpricing.
4. Create a 70/30 chronological split. The training set excludes contracts that had not resolved by the first test entry, preventing outcome leakage.
5. Estimate the test-period model probability from the corresponding training-period calibration bin.
6. Size YES or NO trades with fractional Kelly, enforce available-cash constraints, and compare zero-cost execution with a 2-cent spread.

## Main result

The data show statistically significant overpricing in the lowest-probability bins and underpricing in several high-probability bins, consistent with a favorite-longshot pattern. That descriptive pattern did not become a profitable trading rule in the held-out period: the quarter-Kelly simulations lost money both before and after the spread assumption. The practical conclusion is that calibration evidence is not the same thing as a durable, tradable edge.

## Repository map

```text
article/      Substack-ready write-up
data/         Raw, intermediate, and final datasets
notebooks/    Original exploratory analysis
outputs/      Rebuilt tables and figures
src/          Reusable analysis package
tests/        Unit tests for calibration, splitting, and backtesting
```

## Caveats

- Polymarket categories are inferred from market text, so category-level conclusions depend on a keyword classifier.
- The data are contract observations, not independent experimental draws; related markets can share common information and outcomes.
- The spread is modeled as a fixed execution penalty and does not include fees, slippage, order-book depth, or rejected fills.
- The analysis uses a single historical split. Rolling or event-grouped validation would provide a stronger test of stability.

