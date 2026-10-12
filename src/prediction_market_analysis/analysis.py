"""Core statistics and reporting for the prediction-market project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss


PROBABILITY_BINS = np.linspace(0.0, 1.0, 11)
REQUIRED_COLUMNS = {
    "market",
    "category",
    "resolution_date",
    "price",
    "outcome",
    "venue",
    "entry_time",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime settings for the reproducible analysis."""

    data_path: Path = Path("data/prediction_markets_clean.csv")
    output_dir: Path = Path("outputs")
    train_fraction: float = 0.70
    starting_bankroll: float = 10_000.0
    baseline_kelly_fraction: float = 0.25
    spread: float = 0.02


@dataclass
class BacktestResult:
    metrics: dict[str, float | int]
    equity_curve: pd.Series
    trades: pd.DataFrame


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the final contract-level analysis dataset."""

    data = pd.read_csv(path, dtype={"market": "string"}, low_memory=False)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    data = data.copy()
    data["entry_time"] = pd.to_datetime(
        data["entry_time"], format="mixed", utc=True, errors="coerce"
    )
    data["resolution_date"] = pd.to_datetime(
        data["resolution_date"], format="mixed", utc=True, errors="coerce"
    )
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["outcome"] = pd.to_numeric(data["outcome"], errors="coerce")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        null_counts = data[list(REQUIRED_COLUMNS)].isna().sum()
        raise ValueError(f"Required fields contain nulls: {null_counts[null_counts > 0].to_dict()}")
    if not data["price"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("All market prices must be between 0 and 1.")
    if not data["outcome"].isin([0, 1]).all():
        raise ValueError("Outcomes must be binary (0 or 1).")
    if (data["entry_time"] >= data["resolution_date"]).any():
        raise ValueError("Every trade entry must precede its market resolution.")
    if data.duplicated(["venue", "market"]).any():
        raise ValueError("Market identifiers must be unique within each venue.")

    data["outcome"] = data["outcome"].astype(int)
    return data


def add_price_bins(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["price_bin"] = pd.cut(
        result["price"], bins=PROBABILITY_BINS, include_lowest=True
    )
    return result


def _brier_row(data: pd.DataFrame) -> dict[str, float | int]:
    base_rate = float(data["outcome"].mean())
    market_brier = float(brier_score_loss(data["outcome"], data["price"]))
    naive_brier = float(
        brier_score_loss(data["outcome"], np.full(len(data), base_rate))
    )
    return {
        "n_markets": int(len(data)),
        "base_rate": base_rate,
        "market_brier": market_brier,
        "naive_brier": naive_brier,
        "improvement_vs_naive": naive_brier - market_brier,
    }


def brier_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Return Brier scores overall and by venue and category."""

    rows = [{"segment_type": "overall", "segment": "All", **_brier_row(data)}]
    for segment_type, column in (("venue", "venue"), ("category", "category")):
        for segment, group in data.groupby(column, sort=True):
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    **_brier_row(group),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate market probabilities and outcomes into 10-point bins."""

    binned = add_price_bins(data)
    table = (
        binned.groupby("price_bin", observed=True)
        .agg(
            n_markets=("outcome", "size"),
            avg_predicted=("price", "mean"),
            actual_frequency=("outcome", "mean"),
        )
        .reset_index()
    )
    table["bias"] = table["avg_predicted"] - table["actual_frequency"]
    return table


def _bias_statistics(group: pd.DataFrame) -> pd.Series:
    errors = group["price"] - group["outcome"]
    n_markets = len(errors)
    mean_bias = float(errors.mean())
    if n_markets < 2 or float(errors.std(ddof=1)) == 0.0:
        t_stat = np.nan
        p_value = np.nan
        ci_low = mean_bias
        ci_high = mean_bias
    else:
        test = stats.ttest_1samp(errors, popmean=0.0)
        interval = stats.t.interval(
            confidence=0.95,
            df=n_markets - 1,
            loc=mean_bias,
            scale=stats.sem(errors),
        )
        t_stat = float(test.statistic)
        p_value = float(test.pvalue)
        ci_low = float(interval[0])
        ci_high = float(interval[1])

    return pd.Series(
        {
            "n_markets": n_markets,
            "avg_predicted": float(group["price"].mean()),
            "actual_frequency": float(group["outcome"].mean()),
            "bias": mean_bias,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "t_stat": t_stat,
            "p_value": p_value,
        }
    )


def bias_table(data: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    """Estimate bias, confidence intervals, and t-tests by probability bin."""

    binned = add_price_bins(data)
    group_columns = [*(groups or []), "price_bin"]
    return (
        binned.groupby(group_columns, observed=True)
        .apply(_bias_statistics, include_groups=False)
        .reset_index()
    )


def summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["venue", "category"], sort=True)
        .agg(
            n_contracts=("outcome", "size"),
            resolution_rate=("outcome", "mean"),
            mean_implied_probability=("price", "mean"),
        )
        .reset_index()
    )


def prepare_purged_split(
    data: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Create a chronological split with no unresolved training outcomes at test start."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    ordered = data.sort_values("entry_time").reset_index(drop=True)
    split_index = min(int(len(ordered) * train_fraction), len(ordered) - 1)
    split_time = ordered.iloc[split_index]["entry_time"]
    train = ordered.loc[ordered["resolution_date"] < split_time].copy()
    test = ordered.loc[ordered["entry_time"] >= split_time].copy()
    if train.empty or test.empty:
        raise ValueError("The chronological split produced an empty train or test set.")

    train = add_price_bins(train)
    test = add_price_bins(test)
    calibration_map = train.groupby("price_bin", observed=True)["outcome"].mean()
    test["p_model"] = test["price_bin"].map(calibration_map).astype(float)
    if test["p_model"].isna().any():
        missing_bins = test.loc[test["p_model"].isna(), "price_bin"].unique()
        raise ValueError(f"Training data did not cover test bins: {list(missing_bins)}")
    return train, test, split_time


def run_kelly_backtest(
    data: pd.DataFrame,
    *,
    kelly_fraction: float,
    spread: float,
    starting_bankroll: float,
) -> BacktestResult:
    """Backtest simultaneous, cash-constrained binary-contract positions."""

    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must be in (0, 1].")
    if not 0.0 <= spread < 1.0:
        raise ValueError("spread must be in [0, 1).")
    if starting_bankroll <= 0:
        raise ValueError("starting_bankroll must be positive.")

    ordered = data.sort_values("entry_time").copy()
    half_spread = spread / 2.0
    cash = float(starting_bankroll)
    locked_capital = 0.0
    open_positions: list[dict[str, object]] = []
    sides = pd.Series("NONE", index=ordered.index, dtype="string")
    fractions = pd.Series(0.0, index=ordered.index, dtype=float)
    pnl = pd.Series(0.0, index=ordered.index, dtype=float)
    equity_times: list[pd.Timestamp] = []
    equity_values: list[float] = []

    def settle_positions(up_to_time: pd.Timestamp) -> None:
        nonlocal cash, locked_capital, open_positions
        still_open: list[dict[str, object]] = []
        for position in open_positions:
            if position["resolution_date"] <= up_to_time:
                stake = float(position["stake"])
                locked_capital -= stake
                if bool(position["won"]):
                    payout = stake / float(position["execution_price"])
                    cash += payout
                    pnl.loc[position["idx"]] = payout - stake
                else:
                    pnl.loc[position["idx"]] = -stake
            else:
                still_open.append(position)
        open_positions = still_open

    for entry_time, group in ordered.groupby("entry_time", sort=True):
        settle_positions(entry_time)
        book_equity = cash + locked_capital
        proposals: list[dict[str, object]] = []

        for idx, row in group.iterrows():
            market_price = float(row["price"])
            model_probability = float(row["p_model"])
            yes_price = market_price + half_spread
            no_price = 1.0 - market_price + half_spread
            yes_kelly = (
                max((model_probability - yes_price) / (1.0 - yes_price), 0.0)
                if 0.0 < yes_price < 1.0
                else 0.0
            )
            no_kelly = (
                max(((1.0 - model_probability) - no_price) / (1.0 - no_price), 0.0)
                if 0.0 < no_price < 1.0
                else 0.0
            )

            if yes_kelly > no_kelly and yes_kelly > 0.0:
                side = "YES"
                full_kelly = yes_kelly
                execution_price = yes_price
                won = int(row["outcome"]) == 1
            elif no_kelly > 0.0:
                side = "NO"
                full_kelly = no_kelly
                execution_price = no_price
                won = int(row["outcome"]) == 0
            else:
                side = "NONE"
                full_kelly = 0.0
                execution_price = np.nan
                won = False

            fraction = kelly_fraction * full_kelly
            proposals.append(
                {
                    "idx": idx,
                    "side": side,
                    "fraction": fraction,
                    "desired_stake": book_equity * fraction,
                    "execution_price": execution_price,
                    "won": won,
                    "resolution_date": row["resolution_date"],
                }
            )

        total_desired = sum(float(item["desired_stake"]) for item in proposals)
        scale = min(cash / total_desired, 1.0) if total_desired > 0.0 else 1.0

        for proposal in proposals:
            stake = float(proposal["desired_stake"]) * scale
            if proposal["side"] == "NONE" or stake <= 0.0:
                continue
            idx = proposal["idx"]
            sides.loc[idx] = str(proposal["side"])
            fractions.loc[idx] = float(proposal["fraction"]) * scale
            cash -= stake
            locked_capital += stake
            open_positions.append({**proposal, "stake": stake})

        equity_times.append(entry_time)
        equity_values.append(cash + locked_capital)

    final_time = ordered["resolution_date"].max() + pd.Timedelta(seconds=1)
    settle_positions(final_time)
    equity_times.append(final_time)
    equity_values.append(cash)

    equity_curve = pd.Series(
        equity_values,
        index=pd.DatetimeIndex(equity_times),
        name="equity",
        dtype=float,
    )
    daily_equity = equity_curve.resample("1D").last().ffill()
    daily_returns = daily_equity.pct_change(fill_method=None).dropna()
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(365))
        if len(daily_returns) > 1 and daily_returns.std() > 0
        else np.nan
    )
    max_drawdown = float(-(equity_curve / equity_curve.cummax() - 1.0).min())

    ordered["backtest_side"] = sides
    ordered["bet_fraction"] = fractions
    ordered["pnl"] = pnl
    metrics: dict[str, float | int] = {
        "k": kelly_fraction,
        "spread": spread,
        "starting_bankroll": starting_bankroll,
        "final_bankroll": cash,
        "total_return": cash / starting_bankroll - 1.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": int((ordered["backtest_side"] != "NONE").sum()),
    }
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, trades=ordered)


def _save_figures(
    output_dir: Path,
    calibration: pd.DataFrame,
    bias: pd.DataFrame,
    venue_bias: pd.DataFrame,
    no_cost: BacktestResult,
    with_cost: BacktestResult,
    sensitivity: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.plot(calibration["avg_predicted"], calibration["actual_frequency"], marker="o")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(xlabel="Mean implied probability", ylabel="Observed YES rate", title="Prediction-market calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_curve.png", dpi=180)
    plt.close(fig)

    midpoint = bias["price_bin"].map(lambda interval: interval.mid).astype(float)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.errorbar(
        midpoint,
        bias["bias"],
        yerr=[bias["bias"] - bias["ci_low"], bias["ci_high"] - bias["bias"]],
        marker="o",
        capsize=4,
    )
    ax.axhline(0, linestyle="--", color="gray")
    ax.set(xlabel="Implied-probability bin midpoint", ylabel="Price minus outcome frequency", title="Favorite-longshot bias (95% confidence intervals)")
    fig.tight_layout()
    fig.savefig(figure_dir / "favorite_longshot_bias.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for venue, group in venue_bias.groupby("venue"):
        venue_midpoint = group["price_bin"].map(lambda interval: interval.mid).astype(float)
        ax.plot(venue_midpoint, group["bias"], marker="o", label=venue)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set(xlabel="Implied-probability bin midpoint", ylabel="Bias (price - frequency)", title="Bias by venue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "venue_bias.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(no_cost.equity_curve.index, no_cost.equity_curve.values, label="No transaction costs")
    ax.plot(with_cost.equity_curve.index, with_cost.equity_curve.values, label="2-cent spread")
    ax.axhline(10_000, linestyle="--", color="gray", label="Starting bankroll")
    ax.set(xlabel="Date", ylabel="Book-value bankroll ($)", title="Quarter-Kelly out-of-sample equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "equity_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(sensitivity["k"], sensitivity["total_return"], marker="o")
    axes[0].axhline(0, linestyle="--", color="gray")
    axes[0].set(xlabel="Kelly multiplier", ylabel="Total return", title="Return sensitivity")
    axes[1].plot(sensitivity["k"], sensitivity["max_drawdown"], marker="o", color="#b45309")
    axes[1].set(xlabel="Kelly multiplier", ylabel="Maximum drawdown", title="Drawdown sensitivity")
    fig.tight_layout()
    fig.savefig(figure_dir / "kelly_sensitivity.png", dpi=180)
    plt.close(fig)


def run_analysis(config: AnalysisConfig) -> dict[str, object]:
    """Run the full saved-data analysis and write tables and figures."""

    data = load_dataset(config.data_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary = summary_statistics(data)
    brier = brier_scores(data)
    calibration = calibration_table(data)
    overall_bias = bias_table(data)
    venue = bias_table(data, ["venue"])
    category = bias_table(data, ["category"])
    train, test, split_time = prepare_purged_split(data, config.train_fraction)

    no_cost = run_kelly_backtest(
        test,
        kelly_fraction=config.baseline_kelly_fraction,
        spread=0.0,
        starting_bankroll=config.starting_bankroll,
    )
    with_cost = run_kelly_backtest(
        test,
        kelly_fraction=config.baseline_kelly_fraction,
        spread=config.spread,
        starting_bankroll=config.starting_bankroll,
    )
    comparison = pd.DataFrame(
        [no_cost.metrics, with_cost.metrics],
        index=["No transaction costs", "2-cent spread"],
    )

    sensitivity_rows = []
    for kelly_fraction in np.round(np.arange(0.1, 1.01, 0.1), 1):
        result = run_kelly_backtest(
            test,
            kelly_fraction=float(kelly_fraction),
            spread=config.spread,
            starting_bankroll=config.starting_bankroll,
        )
        sensitivity_rows.append(result.metrics)
    sensitivity = pd.DataFrame(sensitivity_rows)

    tables = {
        "summary_statistics.csv": summary,
        "brier_scores.csv": brier,
        "calibration.csv": calibration,
        "favorite_longshot_bias.csv": overall_bias,
        "venue_bias.csv": venue,
        "category_bias.csv": category,
        "backtest_comparison.csv": comparison,
        "kelly_sensitivity.csv": sensitivity,
    }
    for filename, table in tables.items():
        table.to_csv(
            config.output_dir / filename,
            index=filename == "backtest_comparison.csv",
        )

    _save_figures(
        config.output_dir,
        calibration,
        overall_bias,
        venue,
        no_cost,
        with_cost,
        sensitivity,
    )

    return {
        "rows": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "split_time": split_time,
        "brier": brier,
        "comparison": comparison,
        "sensitivity": sensitivity,
    }

