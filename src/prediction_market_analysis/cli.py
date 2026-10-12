"""Command-line interface for the prediction-market analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import AnalysisConfig, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild prediction-market result tables and figures."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/prediction_markets_clean.csv"),
        help="Path to the contract-level clean CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated tables and figures.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_analysis(AnalysisConfig(data_path=args.data, output_dir=args.output_dir))
    comparison = result["comparison"]
    print(f"Analyzed {result['rows']:,} contracts.")
    print(
        f"Purged split: {result['train_rows']:,} train / "
        f"{result['test_rows']:,} test at {result['split_time']}."
    )
    print("\nOut-of-sample backtest:")
    print(comparison[["final_bankroll", "total_return", "sharpe", "max_drawdown", "trades"]])


if __name__ == "__main__":
    main()

