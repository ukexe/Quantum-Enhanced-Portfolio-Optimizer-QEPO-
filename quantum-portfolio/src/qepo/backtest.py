import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from qepo.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)

# Cursor Task: Backtesting engine with walk-forward framework


class BacktestResult:
    """Container for backtest results and performance metrics."""

    def __init__(self):
        self.portfolio_returns: pd.Series = pd.Series(dtype=float)
        self.portfolio_weights: pd.DataFrame = pd.DataFrame()
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.transaction_costs: pd.Series = pd.Series(dtype=float)
        self.turnover: pd.Series = pd.Series(dtype=float)
        self.exposures: pd.DataFrame = pd.DataFrame()
        self.metrics: Dict[str, float] = {}
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.drawdown: pd.Series = pd.Series(dtype=float)
        self.rebalance_dates: List[datetime] = []


def walk_forward(
    config: Dict,
    strategy_fn: Callable,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints,
    benchmark_ticker: str = "SPY",
    output_dir: Optional[Path] = None,
) -> BacktestResult:
    """
    Walk-forward backtest framework with transaction costs and rebalancing.

    Parameters
    ----------
    config : Dict
        Backtest configuration containing rebalancing frequency, costs, etc.
    strategy_fn : Callable
        Strategy function that takes (returns, covariance, constraints) and returns weights.
    prices : pd.DataFrame
        Historical price data with columns ['date', 'ticker', 'adj_close'].
    returns : pd.DataFrame
        Historical returns data with columns ['date', 'ticker', 'ret_d'].
    covariance : pd.DataFrame
        Rolling covariance data with columns ['date', 'ticker1', 'ticker2', 'covariance'].
    constraints : PortfolioConstraints
        Portfolio constraints object.
    benchmark_ticker : str, default="SPY"
        Benchmark ticker for comparison.
    output_dir : Path, optional
        Directory to save outputs. Defaults to 'data/backtest'.

    Returns
    -------
    BacktestResult
        Complete backtest results with performance metrics.
    """
    logger.info("Starting walk-forward backtest")

    if output_dir is None:
        output_dir = Path("data/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize result container
    result = BacktestResult()

    # Extract configuration
    rebalance_freq = config.get("rebalance_freq", "monthly")
    transaction_cost_bps = config.get("transaction_cost_bps", 5.0)
    turnover_cap = config.get("turnover_cap", 0.5)
    train_window = config.get("train_window", 252)  # 1 year
    test_window = config.get("test_window", 21)  # 1 month

    # Get unique dates and tickers
    all_dates = sorted(returns["date"].unique())
    all_tickers = sorted(returns["ticker"].unique())

    # Get benchmark data
    benchmark_data = _get_benchmark_data(prices, benchmark_ticker)

    # Generate rebalancing dates
    rebalance_dates = _generate_rebalance_dates(all_dates, rebalance_freq)
    result.rebalance_dates = rebalance_dates

    logger.info(f"Generated {len(rebalance_dates)} rebalancing dates")

    # Initialize portfolio tracking
    current_weights = np.zeros(len(all_tickers))
    portfolio_value = 1.0  # Start with $1
    equity_curve = [1.0]
    drawdown_curve = [0.0]
    peak_value = 1.0

    # Storage for results
    portfolio_returns_list = []
    portfolio_weights_list = []
    transaction_costs_list = []
    turnover_list = []
    exposure_list = []

    # Walk-forward loop
    for i, rebalance_date in enumerate(rebalance_dates):
        logger.info(
            f"Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}"
        )

        # Determine training period - use data before rebalance date
        rebalance_idx = all_dates.index(rebalance_date)
        train_start_idx = max(0, rebalance_idx - train_window)
        train_end_idx = rebalance_idx

        if (
            train_end_idx - train_start_idx < train_window // 2
        ):  # Need at least half the training window
            logger.info(f"Insufficient training data for rebalance {i+1}")
            continue

        train_start_date = all_dates[train_start_idx]
        train_end_date = all_dates[train_end_idx - 1]

        # Determine test period - use data after rebalance date
        test_start_date = rebalance_date
        test_end_idx = min(len(all_dates), rebalance_idx + test_window)
        test_end_date = all_dates[test_end_idx - 1]

        logger.debug(f"Train: {train_start_date} to {train_end_date}")
        logger.debug(f"Test: {test_start_date} to {test_end_date}")

        # Get training data
        train_returns = returns[
            (returns["date"] >= train_start_date) & (returns["date"] <= train_end_date)
        ]
        train_covariance = covariance[
            (covariance["date"] >= train_start_date)
            & (covariance["date"] <= train_end_date)
        ]

        # Get test period data
        test_returns = returns[
            (returns["date"] >= test_start_date) & (returns["date"] <= test_end_date)
        ]

        if train_returns.empty or test_returns.empty:
            logger.warning(f"Skipping rebalance {i+1}: insufficient data")
            continue

        # Generate portfolio weights using strategy
        try:
            new_weights = strategy_fn(train_returns, train_covariance, constraints)

            # Ensure weights are valid
            if new_weights is None or len(new_weights) != len(all_tickers):
                logger.warning(f"Invalid weights from strategy, using equal weights")
                new_weights = np.ones(len(all_tickers)) / len(all_tickers)

        except Exception as e:
            logger.error(f"Strategy failed: {e}, using equal weights")
            new_weights = np.ones(len(all_tickers)) / len(all_tickers)

        # Calculate transaction costs and turnover
        turnover = np.sum(np.abs(new_weights - current_weights)) / 2.0
        transaction_cost = turnover * transaction_cost_bps / 10000.0

        # Apply turnover cap if specified
        if turnover_cap is not None and turnover > turnover_cap:
            logger.warning(
                f"Turnover {turnover:.3f} exceeds cap {turnover_cap}, scaling down"
            )
            # Scale down the change to respect turnover cap
            scale_factor = turnover_cap / turnover
            current_weights = current_weights + scale_factor * (
                new_weights - current_weights
            )
            turnover = turnover_cap
            transaction_cost = turnover * transaction_cost_bps / 10000.0
        else:
            current_weights = new_weights.copy()

        # Calculate portfolio returns for test period
        test_dates = sorted(test_returns["date"].unique())
        period_returns = []

        for test_date in test_dates:
            # Get returns for this date
            date_returns = test_returns[test_returns["date"] == test_date]
            if date_returns.empty:
                continue

            # Calculate portfolio return
            portfolio_return = 0.0
            for _, row in date_returns.iterrows():
                ticker_idx = all_tickers.index(row["ticker"])
                portfolio_return += current_weights[ticker_idx] * row["ret_d"]

            period_returns.append(portfolio_return)

        # Store results
        if period_returns:
            portfolio_returns_list.extend(period_returns)
            portfolio_weights_list.append(current_weights.copy())
            transaction_costs_list.append(transaction_cost)
            turnover_list.append(turnover)

            # Calculate exposures (sector/industry if available)
            exposure_dict = {"date": test_end_date}
            for j, ticker in enumerate(all_tickers):
                exposure_dict[ticker] = current_weights[j]
            exposure_list.append(exposure_dict)

            # Update equity curve
            for ret in period_returns:
                portfolio_value *= 1 + ret - transaction_cost / len(period_returns)
                equity_curve.append(portfolio_value)

                # Update drawdown
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                drawdown = (
                    portfolio_value - peak_value
                ) / peak_value  # Should be negative
                drawdown_curve.append(drawdown)

    # Compile results
    if portfolio_returns_list:
        result.portfolio_returns = pd.Series(portfolio_returns_list)
        result.equity_curve = pd.Series(equity_curve)
        result.drawdown = pd.Series(drawdown_curve)

        # Portfolio weights DataFrame
        if portfolio_weights_list:
            weights_df = pd.DataFrame(portfolio_weights_list, columns=all_tickers)
            weights_df["date"] = [
                rebalance_dates[i] for i in range(len(portfolio_weights_list))
            ]
            result.portfolio_weights = weights_df

        # Transaction costs and turnover
        result.transaction_costs = pd.Series(transaction_costs_list)
        result.turnover = pd.Series(turnover_list)

        # Exposures
        if exposure_list:
            result.exposures = pd.DataFrame(exposure_list)

        # Benchmark returns
        result.benchmark_returns = (
            benchmark_data["ret_d"] if not benchmark_data.empty else pd.Series()
        )

        # Calculate performance metrics
        result.metrics = _calculate_performance_metrics(
            result.portfolio_returns,
            result.benchmark_returns,
            result.transaction_costs,
            result.turnover,
        )

        # Save outputs
        _save_backtest_outputs(result, output_dir)

        # Log to MLflow
        _log_backtest_to_mlflow(result, config)

        logger.info("Backtest completed successfully")
    else:
        logger.error("No portfolio returns generated")

    return result


def _get_benchmark_data(prices: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    """Get benchmark returns data."""
    benchmark_prices = prices[prices["ticker"] == benchmark_ticker].copy()
    if benchmark_prices.empty:
        logger.warning(f"Benchmark ticker {benchmark_ticker} not found in price data")
        return pd.DataFrame()

    benchmark_prices = benchmark_prices.sort_values("date")
    benchmark_prices["ret_d"] = benchmark_prices["adj_close"].pct_change()
    return benchmark_prices[["date", "ret_d"]].dropna()


def _generate_rebalance_dates(
    all_dates: List[datetime], frequency: str
) -> List[datetime]:
    """Generate rebalancing dates based on frequency."""
    if frequency == "monthly":
        # Last trading day of each month
        rebalance_dates = []
        current_month = None
        for date in all_dates:
            if current_month != date.month:
                if current_month is not None:
                    rebalance_dates.append(last_date)
                current_month = date.month
            last_date = date
        if current_month is not None:
            rebalance_dates.append(last_date)
    elif frequency == "weekly":
        # Every 5 trading days
        rebalance_dates = all_dates[4::5]
    else:
        # Daily
        rebalance_dates = all_dates

    return rebalance_dates


def _calculate_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    transaction_costs: pd.Series,
    turnover: pd.Series,
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    metrics = {}

    if portfolio_returns.empty:
        return metrics

    # Basic return metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)

    metrics["total_return"] = total_return
    metrics["annualized_return"] = annualized_return
    metrics["volatility"] = volatility

    # Risk-adjusted metrics
    if volatility > 0:
        sharpe_ratio = annualized_return / volatility
        metrics["sharpe_ratio"] = sharpe_ratio

    # Sortino ratio (downside deviation)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if len(downside_returns) > 0:
        downside_volatility = downside_returns.std() * np.sqrt(252)
        if downside_volatility > 0:
            sortino_ratio = annualized_return / downside_volatility
            metrics["sortino_ratio"] = sortino_ratio

    # Maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    metrics["max_drawdown"] = max_drawdown

    # Transaction costs and turnover
    if not transaction_costs.empty:
        metrics["avg_transaction_cost"] = transaction_costs.mean()
        metrics["total_transaction_cost"] = transaction_costs.sum()

    if not turnover.empty:
        metrics["avg_turnover"] = turnover.mean()
        metrics["max_turnover"] = turnover.max()

    # Benchmark comparison
    if not benchmark_returns.empty:
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) > 0:
            portfolio_aligned = portfolio_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]

            # Alpha and Beta
            excess_returns = portfolio_aligned - benchmark_aligned
            metrics["excess_return"] = excess_returns.mean() * 252

            # Information ratio
            if excess_returns.std() > 0:
                information_ratio = (
                    excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                )
                metrics["information_ratio"] = information_ratio

    return metrics


def _save_backtest_outputs(result: BacktestResult, output_dir: Path) -> None:
    """Save backtest results to CSV files."""
    logger.info(f"Saving backtest outputs to {output_dir}")

    # Performance metrics
    perf_df = pd.DataFrame([result.metrics])
    perf_df.to_csv(output_dir / "backtest_perf.csv", index=False)

    # Portfolio allocations
    if not result.portfolio_weights.empty:
        result.portfolio_weights.to_csv(output_dir / "portfolio_alloc.csv", index=False)

    # Exposures
    if not result.exposures.empty:
        result.exposures.to_csv(output_dir / "exposures.csv", index=False)

    # Equity curve
    if not result.equity_curve.empty:
        # Ensure dates and equity curve have same length
        num_dates = min(len(result.rebalance_dates), len(result.equity_curve))
        equity_df = pd.DataFrame(
            {
                "date": result.rebalance_dates[:num_dates],
                "portfolio_value": result.equity_curve.values[:num_dates],
            }
        )
        equity_df.to_csv(output_dir / "equity_curve.csv", index=False)

    logger.info("Backtest outputs saved successfully")


def _log_backtest_to_mlflow(result: BacktestResult, config: Dict) -> None:
    """Log backtest results to MLflow."""
    try:
        # Log configuration
        mlflow.log_params(config)

        # Log performance metrics
        for metric_name, metric_value in result.metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log summary statistics
        if not result.portfolio_returns.empty:
            mlflow.log_metric("num_trading_days", len(result.portfolio_returns))
            mlflow.log_metric("num_rebalances", len(result.rebalance_dates))

        logger.info("Backtest results logged to MLflow")

    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")


def plot_equity_curve(
    result: BacktestResult, output_dir: Optional[Path] = None
) -> None:
    """Create equity curve and drawdown plots."""
    if result.equity_curve.empty:
        logger.warning("No equity curve data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Equity curve
    dates = result.rebalance_dates[: len(result.equity_curve)]
    ax1.plot(dates, result.equity_curve.values, label="Portfolio", linewidth=2)

    # Add benchmark if available
    if not result.benchmark_returns.empty:
        benchmark_cumulative = (1 + result.benchmark_returns).cumprod()
        # Ensure benchmark data aligns with dates
        if len(benchmark_cumulative) == len(dates):
            ax1.plot(
                dates,
                benchmark_cumulative.values,
                label="Benchmark",
                linewidth=2,
                alpha=0.7,
            )
        else:
            # Use available benchmark dates
            benchmark_dates = result.benchmark_returns.index[
                : len(benchmark_cumulative)
            ]
            ax1.plot(
                benchmark_dates,
                benchmark_cumulative.values,
                label="Benchmark",
                linewidth=2,
                alpha=0.7,
            )

    ax1.set_ylabel("Portfolio Value")
    ax1.set_title("Equity Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(dates, result.drawdown.values, 0, alpha=0.3, color="red")
    ax2.plot(dates, result.drawdown.values, color="red", linewidth=1)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "equity_curve.png", dpi=300, bbox_inches="tight")
        logger.info(f"Equity curve plot saved to {output_dir / 'equity_curve.png'}")

    plt.show()


def plot_portfolio_weights(
    result: BacktestResult, output_dir: Optional[Path] = None
) -> None:
    """Create portfolio weights heatmap."""
    if result.portfolio_weights.empty:
        logger.warning("No portfolio weights data to plot")
        return

    # Prepare data for heatmap
    weights_data = result.portfolio_weights.set_index("date")

    # Select top holdings for visualization
    avg_weights = weights_data.mean().sort_values(ascending=False)
    top_holdings = avg_weights.head(20).index  # Top 20 holdings
    weights_subset = weights_data[top_holdings]

    plt.figure(figsize=(15, 8))
    sns.heatmap(weights_subset.T, cmap="YlOrRd", cbar_kws={"label": "Weight"})
    plt.title("Portfolio Weights Over Time (Top 20 Holdings)")
    plt.xlabel("Date")
    plt.ylabel("Ticker")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if output_dir:
        plt.savefig(output_dir / "portfolio_weights.png", dpi=300, bbox_inches="tight")
        logger.info(
            f"Portfolio weights plot saved to {output_dir / 'portfolio_weights.png'}"
        )

    plt.show()
