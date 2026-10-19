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
BACKTEST_REQUIRED_COLUMNS = {
    "entry_time",
    "resolution_date",
    "price",
    "p_model",
    "outcome",
}
# A cash stake below one cent cannot be meaningfully represented as a trade.  This
# prevents floating-point dust from being counted as an execution after a bankroll
# has become very small.
MINIMUM_EXECUTED_STAKE = 0.01


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime settings for the reproducible analysis."""

    data_path: Path = Path("data/prediction_markets_clean.csv")
    output_dir: Path = Path("outputs")
    train_fraction: float = 0.70
    starting_bankroll: float = 10_000.0
    baseline_kelly_fraction: float = 0.25
    spread: float = 0.02
    minimum_executed_stake: float = MINIMUM_EXECUTED_STAKE


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


def _finite_float(value: object, name: str) -> float:
    """Convert a scalar setting to a finite float with a clear error."""

    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number.")
    return converted


def _validate_backtest_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return validated backtest inputs without mutating the caller's frame."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Backtest data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Backtest data must contain at least one contract.")

    missing_columns = BACKTEST_REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Backtest data is missing columns: {sorted(missing_columns)}"
        )

    validated = data.copy()
    for column in ("entry_time", "resolution_date"):
        validated[column] = pd.to_datetime(
            validated[column], format="mixed", utc=True, errors="coerce"
        )
    for column in ("price", "p_model", "outcome"):
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    if validated[["entry_time", "resolution_date"]].isna().any().any():
        raise ValueError("Backtest timestamps must be valid and non-missing.")
    numeric_columns = ["price", "p_model", "outcome"]
    if not np.isfinite(validated[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Backtest price, model probability, and outcome must be finite.")
    if not validated["price"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Backtest market prices must be between 0 and 1.")
    if not validated["p_model"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Backtest model probabilities must be between 0 and 1.")
    if not validated["outcome"].isin([0, 1]).all():
        raise ValueError("Backtest outcomes must be binary (0 or 1).")
    if (validated["entry_time"] >= validated["resolution_date"]).any():
        raise ValueError("Every backtest entry must precede its market resolution.")

    validated["outcome"] = validated["outcome"].astype(int)
    return validated


def run_kelly_backtest(
    data: pd.DataFrame,
    *,
    kelly_fraction: float,
    spread: float,
    starting_bankroll: float,
    minimum_executed_stake: float = MINIMUM_EXECUTED_STAKE,
) -> BacktestResult:
    """Backtest simultaneous, cash-constrained binary-contract positions.

    ``eligible_signals`` counts positive mathematical Kelly edges. ``trades``
    counts only positions at or above ``minimum_executed_stake``; this avoids
    reporting floating-point dust as an execution.
    """

    kelly_fraction = _finite_float(kelly_fraction, "kelly_fraction")
    spread = _finite_float(spread, "spread")
    starting_bankroll = _finite_float(starting_bankroll, "starting_bankroll")
    minimum_executed_stake = _finite_float(
        minimum_executed_stake, "minimum_executed_stake"
    )

    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must be in (0, 1].")
    if not 0.0 <= spread < 1.0:
        raise ValueError("spread must be in [0, 1).")
    if starting_bankroll <= 0:
        raise ValueError("starting_bankroll must be positive.")
    if minimum_executed_stake < 0.0:
        raise ValueError("minimum_executed_stake must be non-negative.")

    ordered = _validate_backtest_data(data).sort_values("entry_time").copy()
    half_spread = spread / 2.0
    cash = float(starting_bankroll)
    locked_capital = 0.0
    open_positions: list[dict[str, object]] = []
    sides = pd.Series("NONE", index=ordered.index, dtype="string")
    fractions = pd.Series(0.0, index=ordered.index, dtype=float)
    pnl = pd.Series(0.0, index=ordered.index, dtype=float)
    equity_times: list[pd.Timestamp] = []
    equity_values: list[float] = []
    eligible_signals = 0
    skipped_below_minimum_stake = 0
    skipped_due_to_cash_constraint = 0

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

            if side != "NONE":
                eligible_signals += 1

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
            if proposal["side"] == "NONE":
                continue
            if stake <= 0.0:
                skipped_due_to_cash_constraint += 1
                continue
            if stake < minimum_executed_stake:
                skipped_below_minimum_stake += 1
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
        "minimum_executed_stake": minimum_executed_stake,
        "eligible_signals": eligible_signals,
        "skipped_due_to_cash_constraint": skipped_due_to_cash_constraint,
        "skipped_below_minimum_stake": skipped_below_minimum_stake,
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
    ax.set(
        xlabel="Implied-probability bin midpoint",
        ylabel="Price minus observed YES rate",
        title="Pooled calibration bias (contract-level IID 95% intervals)",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "favorite_longshot_bias.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for venue, group in venue_bias.groupby("venue"):
        venue_midpoint = group["price_bin"].map(lambda interval: interval.mid).astype(float)
        ax.plot(venue_midpoint, group["bias"], marker="o", label=venue)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set(
        xlabel="Implied-probability bin midpoint",
        ylabel="Bias (price - observed YES rate)",
        title="Calibration bias by venue (descriptive)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "venue_bias.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(no_cost.equity_curve.index, no_cost.equity_curve.values, label="No transaction costs")
    ax.plot(with_cost.equity_curve.index, with_cost.equity_curve.values, label="2-cent spread")
    ax.axhline(10_000, linestyle="--", color="gray", label="Starting bankroll")
    ax.set(
        xlabel="Date",
        ylabel="Book-value bankroll ($)",
        title="Quarter-Kelly out-of-sample equity (cost-basis positions)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "equity_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(sensitivity["k"], sensitivity["total_return"], marker="o")
    axes[0].axhline(0, linestyle="--", color="gray")
    axes[0].set(
        xlabel="Kelly multiplier",
        ylabel="Total return",
        title="Return sensitivity",
    )
    axes[1].plot(sensitivity["k"], sensitivity["max_drawdown"], marker="o", color="#b45309")
    axes[1].set(
        xlabel="Kelly multiplier",
        ylabel="Book-value maximum drawdown",
        title="Drawdown sensitivity",
    )
    fig.suptitle("Fixed 2-cent entry-spread sensitivity (stylized execution)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
        minimum_executed_stake=config.minimum_executed_stake,
    )
    with_cost = run_kelly_backtest(
        test,
        kelly_fraction=config.baseline_kelly_fraction,
        spread=config.spread,
        starting_bankroll=config.starting_bankroll,
        minimum_executed_stake=config.minimum_executed_stake,
    )
    comparison = pd.DataFrame(
        [
            {"scenario": "No transaction costs", **no_cost.metrics},
            {"scenario": "2-cent spread", **with_cost.metrics},
        ]
    )

    sensitivity_rows = []
    for kelly_fraction in np.round(np.arange(0.1, 1.01, 0.1), 1):
        result = run_kelly_backtest(
            test,
            kelly_fraction=float(kelly_fraction),
            spread=config.spread,
            starting_bankroll=config.starting_bankroll,
            minimum_executed_stake=config.minimum_executed_stake,
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
            index=False,
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

