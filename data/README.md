# Data inventory

`prediction_markets_clean.csv` is the versioned analysis dataset used by the reproducible package. It contains one resolved binary contract per row, with the price recorded approximately 48 hours before resolution.

Large raw API downloads, derived payloads, and progress checkpoints remain local and are excluded by `.gitignore`. Recreate them from the Kalshi and Polymarket public APIs if needed; they are not required to rerun the statistics, figures, or article in this repository.

| Field | Meaning |
|---|---|
| `market` | Venue-specific contract identifier |
| `category` | Market category; Polymarket values are inferred from market text |
| `resolution_date` | Actual close or settlement timestamp |
| `price` | YES implied probability at the analysis observation time |
| `outcome` | Resolved YES indicator (0 or 1) |
| `venue` | Kalshi or Polymarket |
| `entry_time` | Observation timestamp, approximately 48 hours before resolution |

