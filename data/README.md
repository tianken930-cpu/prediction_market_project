# Data inventory and provenance

## Versioned analysis data

`prediction_markets_clean.csv` is the tracked, contract-level dataset used by
the canonical package analysis. It contains 84,767 resolved binary contracts:

| Venue | Rows |
|---|---:|
| Kalshi | 22,762 |
| Polymarket | 62,005 |

The repository reproduces all tables and figures from this committed file. It
does not claim to reproduce the original historical API collection perfectly.

| Field | Meaning |
|---|---|
| `market` | Venue-specific contract identifier. |
| `category` | Market category. Kalshi uses venue metadata; Polymarket may use heuristic text classification. |
| `resolution_date` | Recorded final close or settlement timestamp used for outcome timing and backtest settlement. |
| `price` | Saved YES implied probability at the analysis observation time. It is not documented here as a verified executable bid, ask, or fill price. |
| `outcome` | Resolved YES indicator: 0 or 1. |
| `venue` | `Kalshi` or `Polymarket`. |
| `entry_time` | Observation timestamp anchored approximately 48 hours before the scheduled event/end time. |

## Archived price and category construction

The archived notebook is the only retained evidence for the historical
construction. It shows the following process, but it cannot substitute for the
omitted raw payloads.

- **Kalshi:** the observation target was `occurrence_datetime - 48 hours`.
  The notebook requested one-minute candlesticks and selected the latest candle
  at or before that target. Its saved YES price was the midpoint of that
  candle's `yes_bid.close_dollars` and `yes_ask.close_dollars`. The final
  `resolution_date` was later mapped to the raw market's `settlement_ts`.
- **Polymarket:** the target was `endDate - 48 hours`, after retaining markets
  whose recorded `closedTime` was within one day of `endDate`. The notebook
  queried `batch-prices-history` over the preceding hour and saved the latest
  returned point's `p` at or before the target. Its raw payloads are omitted,
  so this repository does **not** establish whether that endpoint's `p` was a
  midpoint, trade, quote, or executable fill. The final `resolution_date` was
  later mapped to `closedTime`.
- **Categories:** Kalshi categories came from event metadata and a small set of
  labels was normalized. Polymarket categories came from an ordered keyword
  heuristic over question and event text. After combining the venues, only
  categories with at least 100 contracts were retained.

These venue-specific constructions are not interchangeable market
microstructure measures. In particular, neither should be described as a
verified historical fill price.

## Timing convention

The observation target was constructed relative to scheduled event/end metadata,
not a uniform interval before final resolution. Actual settlement or closure can
occur earlier or later than the scheduled event time; consequently, time from
`entry_time` to `resolution_date` varies across contracts. The dataset must not
be described as observations taken exactly 48 hours before actual resolution.

## Historical construction boundaries

Large raw API responses, progress files, and intermediate clean files are kept
locally and excluded from version control. The retained exploratory notebook
documents historical acquisition work, but it uses live APIs, untracked local
files, and post-close metadata for filtering. Those materials are useful
research history, not a deterministic public-clone reconstruction.

In particular, retrospective filtering can use post-close fields such as actual
closure/settlement metadata. This is a point-in-time and sample-selection
limitation; it is distinct from direct use of held-out outcomes in the canonical
chronological calibration model.

## What a public clone can reproduce

A public clone can install the package, read `prediction_markets_clean.csv`, run
the tests, and regenerate `outputs/` exactly from the committed dataset.

A public clone cannot, from the files distributed here alone, guarantee the
exact original API payloads, historical metadata state, retrieval environment,
or acquisition-time price availability. No retrieval dates, API snapshots,
payload hashes, or reconstructed historical source archives are fabricated in
this repository.
