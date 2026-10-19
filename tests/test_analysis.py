from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prediction_market_analysis.analysis import (  # noqa: E402
    calibration_table,
    load_dataset,
    prepare_purged_split,
    run_kelly_backtest,
)


def backtest_frame(
    *,
    price: list[float],
    p_model: list[float],
    outcome: list[int],
    entry_time: list[str] | None = None,
    resolution_date: list[str] | None = None,
) -> pd.DataFrame:
    count = len(price)
    if entry_time is None:
        entry_time = ["2025-01-01"] * count
    if resolution_date is None:
        resolution_date = ["2025-01-02"] * count
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(entry_time, utc=True),
            "resolution_date": pd.to_datetime(resolution_date, utc=True),
            "price": price,
            "p_model": p_model,
            "outcome": outcome,
        }
    )


class DatasetValidationTests(unittest.TestCase):
    def _valid_dataset(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "market": ["m1"],
                "category": ["Other"],
                "resolution_date": ["2025-01-02T00:00:00Z"],
                "price": [0.5],
                "outcome": [1],
                "venue": ["Kalshi"],
                "entry_time": ["2025-01-01T00:00:00Z"],
            }
        )

    def _load(self, data: pd.DataFrame) -> pd.DataFrame:
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            path = Path(directory) / "data.csv"
            data.to_csv(path, index=False)
            return load_dataset(path)

    def test_load_dataset_rejects_missing_required_column(self) -> None:
        data = self._valid_dataset().drop(columns="price")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self._load(data)

    def test_load_dataset_rejects_invalid_price(self) -> None:
        data = self._valid_dataset()
        data.loc[0, "price"] = 1.01
        with self.assertRaisesRegex(ValueError, "prices"):
            self._load(data)

    def test_load_dataset_rejects_invalid_outcome(self) -> None:
        data = self._valid_dataset()
        data.loc[0, "outcome"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            self._load(data)

    def test_load_dataset_rejects_entry_at_or_after_resolution(self) -> None:
        data = self._valid_dataset()
        data.loc[0, "entry_time"] = "2025-01-02T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "precede"):
            self._load(data)


class CalibrationAndSplitTests(unittest.TestCase):
    def test_calibration_bias_is_price_minus_frequency(self) -> None:
        data = pd.DataFrame(
            {"price": [0.1, 0.1, 0.8, 0.8], "outcome": [0, 1, 1, 1]}
        )
        table = calibration_table(data)
        low_bin = table.iloc[0]
        self.assertAlmostEqual(low_bin["avg_predicted"], 0.1)
        self.assertAlmostEqual(low_bin["actual_frequency"], 0.5)
        self.assertAlmostEqual(low_bin["bias"], -0.4)

    def test_calibration_uses_right_closed_bin_boundaries(self) -> None:
        data = pd.DataFrame({"price": [0.1, 0.2], "outcome": [0, 1]})
        table = calibration_table(data)
        self.assertEqual(str(table.iloc[0]["price_bin"]), "(-0.001, 0.1]")
        self.assertEqual(str(table.iloc[1]["price_bin"]), "(0.1, 0.2]")

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

    def test_p_model_is_fitted_only_from_training_calibration(self) -> None:
        entries = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
        data = pd.DataFrame(
            {
                "entry_time": entries,
                "resolution_date": entries + pd.Timedelta(hours=1),
                "price": [0.25, 0.25, 0.75, 0.75, 0.25, 0.75],
                "outcome": [0, 0, 1, 1, 1, 0],
            }
        )
        _, test, _ = prepare_purged_split(data, 0.7)
        self.assertEqual(test["p_model"].tolist(), [0.0, 1.0])

    def test_purged_split_rejects_test_bin_missing_from_training(self) -> None:
        entries = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
        data = pd.DataFrame(
            {
                "entry_time": entries,
                "resolution_date": entries + pd.Timedelta(hours=1),
                "price": [0.25, 0.25, 0.75, 0.75],
                "outcome": [0, 0, 1, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "did not cover test bins"):
            prepare_purged_split(data, 0.5)


class KellyBacktestTests(unittest.TestCase):
    def _run(self, data: pd.DataFrame, **kwargs: object):
        return run_kelly_backtest(
            data,
            kelly_fraction=kwargs.pop("kelly_fraction", 1.0),
            spread=kwargs.pop("spread", 0.0),
            starting_bankroll=kwargs.pop("starting_bankroll", 100.0),
            **kwargs,
        )

    def test_no_edge_produces_no_signal_or_trade(self) -> None:
        result = self._run(backtest_frame(price=[0.5], p_model=[0.5], outcome=[1]))
        self.assertEqual(result.metrics["eligible_signals"], 0)
        self.assertEqual(result.metrics["trades"], 0)
        self.assertAlmostEqual(result.metrics["final_bankroll"], 100.0)

    def test_yes_win_and_loss_have_expected_payouts(self) -> None:
        win = self._run(backtest_frame(price=[0.5], p_model=[0.6], outcome=[1]))
        loss = self._run(backtest_frame(price=[0.5], p_model=[0.6], outcome=[0]))
        self.assertAlmostEqual(win.metrics["final_bankroll"], 120.0)
        self.assertAlmostEqual(loss.metrics["final_bankroll"], 80.0)
        self.assertEqual(win.trades.loc[win.trades.index[0], "backtest_side"], "YES")

    def test_no_win_and_loss_have_expected_payouts(self) -> None:
        win = self._run(backtest_frame(price=[0.5], p_model=[0.4], outcome=[0]))
        loss = self._run(backtest_frame(price=[0.5], p_model=[0.4], outcome=[1]))
        self.assertAlmostEqual(win.metrics["final_bankroll"], 120.0)
        self.assertAlmostEqual(loss.metrics["final_bankroll"], 80.0)
        self.assertEqual(win.trades.loc[win.trades.index[0], "backtest_side"], "NO")

    def test_simultaneous_positions_are_scaled_to_available_cash(self) -> None:
        result = self._run(
            backtest_frame(price=[0.5, 0.5], p_model=[0.9, 0.9], outcome=[1, 1])
        )
        self.assertAlmostEqual(result.metrics["final_bankroll"], 200.0)
        self.assertEqual(result.metrics["trades"], 2)
        self.assertTrue(np.allclose(result.trades["bet_fraction"], [0.5, 0.5]))

    def test_final_settlement_closes_positions_once(self) -> None:
        result = self._run(
            backtest_frame(
                price=[0.5],
                p_model=[0.6],
                outcome=[1],
                resolution_date=["2025-02-01"],
            )
        )
        self.assertAlmostEqual(result.metrics["final_bankroll"], 120.0)
        self.assertAlmostEqual(result.trades["pnl"].iloc[0], 20.0)

    def test_micro_stakes_are_signals_not_executed_trades(self) -> None:
        result = self._run(
            backtest_frame(price=[0.5], p_model=[0.50001], outcome=[1]),
            minimum_executed_stake=0.01,
        )
        self.assertEqual(result.metrics["eligible_signals"], 1)
        self.assertEqual(result.metrics["skipped_below_minimum_stake"], 1)
        self.assertEqual(result.metrics["trades"], 0)
        self.assertAlmostEqual(result.metrics["final_bankroll"], 100.0)

    def test_backtest_rejects_invalid_contract_data(self) -> None:
        invalid_cases = [
            (backtest_frame(price=[0.5], p_model=[np.nan], outcome=[1]), "finite"),
            (backtest_frame(price=[0.5], p_model=[np.inf], outcome=[1]), "finite"),
            (backtest_frame(price=[1.1], p_model=[0.5], outcome=[1]), "prices"),
            (backtest_frame(price=[0.5], p_model=[1.1], outcome=[1]), "probabilities"),
            (backtest_frame(price=[0.5], p_model=[0.5], outcome=[2]), "binary"),
        ]
        for data, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._run(data)

    def test_backtest_rejects_invalid_timestamps_and_settings(self) -> None:
        invalid_time = backtest_frame(
            price=[0.5],
            p_model=[0.6],
            outcome=[1],
            entry_time=["2025-01-02"],
            resolution_date=["2025-01-01"],
        )
        with self.assertRaisesRegex(ValueError, "precede"):
            self._run(invalid_time)
        invalid_timestamp = pd.DataFrame(
            {
                "entry_time": ["not-a-timestamp"],
                "resolution_date": ["2025-01-02"],
                "price": [0.5],
                "p_model": [0.6],
                "outcome": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self._run(invalid_timestamp)
        with self.assertRaisesRegex(ValueError, "kelly_fraction"):
            self._run(backtest_frame(price=[0.5], p_model=[0.6], outcome=[1]), kelly_fraction=0)
        with self.assertRaisesRegex(ValueError, "spread"):
            self._run(backtest_frame(price=[0.5], p_model=[0.6], outcome=[1]), spread=1.0)
        with self.assertRaisesRegex(ValueError, "starting_bankroll"):
            self._run(
                backtest_frame(price=[0.5], p_model=[0.6], outcome=[1]),
                starting_bankroll=0,
            )

    def test_backtest_rejects_empty_or_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            self._run(pd.DataFrame())
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self._run(pd.DataFrame({"price": [0.5]}))


if __name__ == "__main__":
    unittest.main()
