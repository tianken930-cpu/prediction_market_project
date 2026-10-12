from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prediction_market_analysis.analysis import (  # noqa: E402
    add_price_bins,
    calibration_table,
    prepare_purged_split,
    run_kelly_backtest,
)


class AnalysisTests(unittest.TestCase):
    def test_calibration_bias_is_price_minus_frequency(self) -> None:
        data = pd.DataFrame({"price": [0.1, 0.1, 0.8, 0.8], "outcome": [0, 1, 1, 1]})
        table = calibration_table(data)
        low_bin = table.iloc[0]
        self.assertAlmostEqual(low_bin["avg_predicted"], 0.1)
        self.assertAlmostEqual(low_bin["actual_frequency"], 0.5)
        self.assertAlmostEqual(low_bin["bias"], -0.4)

    def test_purged_split_uses_only_known_training_outcomes(self) -> None:
        entries = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
        data = pd.DataFrame(
            {
                "entry_time": entries,
                "resolution_date": entries + pd.Timedelta(hours=12),
                "price": [0.25, 0.75] * 5,
                "outcome": [0, 1] * 5,
            }
        )
        train, test, split_time = prepare_purged_split(data, 0.8)
        self.assertTrue((train["resolution_date"] < split_time).all())
        self.assertTrue((test["entry_time"] >= split_time).all())
        self.assertFalse(test["p_model"].isna().any())

    def test_kelly_backtest_does_not_trade_without_an_edge(self) -> None:
        data = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(["2025-01-01"], utc=True),
                "resolution_date": pd.to_datetime(["2025-01-02"], utc=True),
                "price": [0.5],
                "p_model": [0.5],
                "outcome": [1],
            }
        )
        result = run_kelly_backtest(
            data, kelly_fraction=0.25, spread=0.02, starting_bankroll=10_000
        )
        self.assertEqual(result.metrics["trades"], 0)
        self.assertAlmostEqual(result.metrics["final_bankroll"], 10_000)


if __name__ == "__main__":
    unittest.main()

