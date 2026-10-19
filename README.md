# Prediction-Market Calibration and Favorite–Longshot Bias

This repository studies calibration in a historical sample of 84,767 resolved
binary contracts from Kalshi and Polymarket. It asks three distinct questions:

1. **Prediction quality:** do market probabilities beat a constant base-rate forecast?
2. **Calibration:** do some price ranges differ systematically from realized YES rates?
3. **Trading:** does a simple, historical-bin calibration signal produce positive
   out-of-sample returns under stylized execution assumptions?

The answer is deliberately mixed. In this sample, market probabilities improve
on a constant base-rate benchmark and pooled prices show calibration asymmetries
consistent with favorite–longshot bias under contract-level IID inference. The
effect varies substantially across venues and probability ranges. A simple
out-of-sample fractional-Kelly simulation does not produce positive returns.

## Dataset

The committed analysis dataset is
[`data/prediction_markets_clean.csv`](data/prediction_markets_clean.csv):

| Venue | Contracts |
|---|---:|
| Kalshi | 22,762 |
| Polymarket | 62,005 |
| **Total** | **84,767** |

Each observation is anchored **approximately 48 hours before the scheduled
event/end time**. It is not uniformly 48 hours before final resolution: actual
time to final settlement or closure varies because it can differ from the
scheduled event time. See [`data/README.md`](data/README.md) for definitions,
coverage, provenance, and reproduction boundaries.

## Canonical reproducible analysis

The package under [`src/prediction_market_analysis/`](src/prediction_market_analysis/)
is the canonical analysis. It uses only the committed clean dataset and rebuilds
the CSV tables and PNG figures in `outputs/`.

```powershell
python -m pip install -e .
prediction-market-analysis
```

Or run from the source layout without installing the console command:

```powershell
$env:PYTHONPATH = "src"
python -m prediction_market_analysis
```

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```

The historical notebook is retained as acquisition/exploration history. It
depends on untracked local intermediates and live APIs, so it is **not** the
canonical reproducible path and should not be expected to recreate the original
historical acquisition from a fresh clone.

The package has no HTTP dependency. The retained acquisition notebook uses
`requests`; install it only when working with that non-canonical notebook:

```powershell
python -m pip install ".[acquisition]"
```

## Method

1. Treat each saved YES price as an implied probability.
2. Compare market Brier score with each segment's constant unconditional
   base-rate forecast.
3. Group prices into ten right-closed, ten-percentage-point bins. Bias is mean
   price minus observed YES frequency.
4. Report contract-level IID one-sample t-tests and 95% intervals for each bin.
5. Form a 70/30 chronological split. Training contracts must resolve before the
   first test entry; later contracts are held out.
6. Map each test price bin to its training-period observed YES frequency.
7. Simulate cash-constrained fractional-Kelly positions at zero spread and at a
   fixed 2-cent total entry-spread assumption.

For a YES contract with model probability `p` and execution price `c`, full
Kelly is `(p - c) / (1 - c)`. For NO, it is
`((1 - p) - c_no) / (1 - c_no)`. Negative edges are not traded. At a common
entry timestamp, desired stakes are scaled to available cash before positions
are opened; each position settles once at its recorded resolution.

## Results

The overall market Brier score is **0.156539**, versus **0.221304** for the
overall constant-base-rate forecast (YES rate 33.06%). This demonstrates useful
prediction quality in this historical sample; it does not establish that one
venue is universally superior to another.

| Pooled price band | N | Mean price | YES rate | Bias |
|---|---:|---:|---:|---:|
| 20–30% | 9,079 | 25.12% | 19.13% | +5.99 pp |
| 60–70% | 2,635 | 65.03% | 66.53% | −1.50 pp |
| 80–90% | 2,008 | 86.06% | 91.73% | −5.68 pp |

The pooled sample therefore has a favorite–longshot-style calibration asymmetry,
but it is not perfectly monotonic. Venue heterogeneity matters: 20–30% pricing
is overpriced on both venues but is much stronger on Kalshi; the pooled 80–90%
favorite result is substantially stronger on Kalshi and weak/non-significant on
Polymarket. Category and subgroup tables are exploratory.

### Chronological out-of-sample simulation

The purged split places 58,949 contracts in training and 25,468 in test, with a
split at 2025-11-26 21:00 UTC. The backtest counts a trade only when its scaled
stake is at least $0.01. `eligible_signals` records positive mathematical Kelly
edges before the cash constraint; it is not an execution count.

| Scenario | Final bankroll | Return | Book-value Sharpe | Book-value max drawdown | Executed trades |
|---|---:|---:|---:|---:|---:|
| No transaction-cost assumption | $5,383.40 | −46.17% | −0.34 | 60.69% | 5,272 |
| Fixed 2-cent entry spread | $6,113.39 | −38.87% | −1.11 | 47.44% | 4,546 |

Every fixed-spread Kelly multiplier from 0.1 through 1.0 finishes below the
starting bankroll. The lower loss under the fixed-spread run does not mean costs
help: the spread removes marginal signals. Observed calibration bias is not
necessarily executable trading alpha.

## Outputs

- `outputs/brier_scores.csv` — accuracy against segment-specific base-rate forecasts.
- `outputs/calibration.csv` and `outputs/favorite_longshot_bias.csv` — pooled calibration.
- `outputs/venue_bias.csv` and `outputs/category_bias.csv` — descriptive subgroup results.
- `outputs/backtest_comparison.csv` — scenarios, eligible signals, skipped signals, and executed trades.
- `outputs/kelly_sensitivity.csv` — fixed-spread fractional-Kelly sensitivity.
- `outputs/figures/` — regenerated figures; the equity and risk plots explicitly use book-value/cost-basis equity.

## Limitations

- **Timing and point-in-time reconstruction.** Observation times are based on
  scheduled event/end metadata. Actual resolution timing varies, and historical
  metadata may have changed. This is retrospective analysis, not a perfect
  reconstruction of every live market state.
- **Retrospective sample construction.** Post-close metadata can be involved in
  filtering the historical sample. That is a retrospective sample-selection and
  point-in-time limitation, distinct from direct test-outcome leakage in the
  chronological calibration model.
- **Price definitions.** Kalshi and Polymarket fields are not necessarily the
  same microstructure object. The data should not be read as verified historical
  executable bid/ask quotes.
- **Liquidity and execution.** Liquidity, depth, fill probability, volume,
  historical executable bid/ask conditions, fees, slippage, queue position,
  capacity, and fill uncertainty are not fully modeled.
- **Transaction costs.** The 2-cent spread is a fixed stylized entry sensitivity,
  not a contract-level reconstruction of historical costs.
- **Inference.** P-values and confidence intervals are contract-level IID
  estimates. Related events and contracts can be correlated, so uncertainty may
  be understated. Many venue/category/bin comparisons are exploratory and raw
  subgroup p-values are not multiple-testing adjusted.
- **Categories.** Kalshi labels come from venue metadata; Polymarket labels may
  rely on heuristic text rules. Category-specific conclusions require caution.
- **Risk metrics.** Open positions are carried as cash plus locked capital at
  cost basis, not fully marked to market. Sharpe and maximum drawdown are thus
  book-value metrics.
- **Coverage.** This is a selected historical sample, not proven complete
  coverage of all historical prediction markets.

## Repository safety

Raw API payloads, intermediates, checkpoints, credentials, and environment
files are ignored. The clean analysis CSV remains tracked because it is required
to reproduce the canonical analysis. No license has been chosen for this
repository; reuse terms remain a maintainer decision.
