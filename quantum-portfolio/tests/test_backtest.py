from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from qepo.backtest import (
    BacktestResult,
    _calculate_performance_metrics,
    _generate_rebalance_dates,
    _get_benchmark_data,
    _save_backtest_outputs,
    plot_equity_curve,
    plot_portfolio_weights,
    walk_forward,
)
from qepo.constraints import PortfolioConstraints


class TestBacktestResult:
    """Test BacktestResult container class."""

    def test_backtest_result_initialization(self):
        """Test BacktestResult initializes with empty containers."""
        result = BacktestResult()

        assert isinstance(result.portfolio_returns, pd.Series)
        assert isinstance(result.portfolio_weights, pd.DataFrame)
        assert isinstance(result.benchmark_returns, pd.Series)
        assert isinstance(result.transaction_costs, pd.Series)
        assert isinstance(result.turnover, pd.Series)
        assert isinstance(result.exposures, pd.DataFrame)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.drawdown, pd.Series)
        assert isinstance(result.rebalance_dates, list)

        # All should be empty initially
        assert result.portfolio_returns.empty
        assert result.portfolio_weights.empty
        assert result.benchmark_returns.empty
        assert result.transaction_costs.empty
        assert result.turnover.empty
        assert result.exposures.empty
        assert result.metrics == {}
        assert result.equity_curve.empty
        assert result.drawdown.empty
        assert result.rebalance_dates == []


class TestBenchmarkData:
    """Test benchmark data extraction."""

    def test_get_benchmark_data_success(self):
        """Test successful benchmark data extraction."""
        # Create sample price data
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices_data = []

        for date in dates:
            prices_data.extend(
                [
                    {
                        "date": date,
                        "ticker": "AAPL",
                        "adj_close": 100.0 + len(prices_data),
                    },
                    {
                        "date": date,
                        "ticker": "SPY",
                        "adj_close": 200.0 + len(prices_data),
                    },
                ]
            )

        prices_df = pd.DataFrame(prices_data)

        # Test benchmark extraction
        benchmark_data = _get_benchmark_data(prices_df, "SPY")

        assert not benchmark_data.empty
        assert "date" in benchmark_data.columns
        assert "ret_d" in benchmark_data.columns
        assert len(benchmark_data) == 9  # 10 dates - 1 (first has no return)
        assert benchmark_data["ret_d"].notna().all()

    def test_get_benchmark_data_missing_ticker(self):
        """Test benchmark data extraction with missing ticker."""
        # Create sample price data without SPY
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        prices_data = []

        for date in dates:
            prices_data.append({"date": date, "ticker": "AAPL", "adj_close": 100.0})

        prices_df = pd.DataFrame(prices_data)

        # Test benchmark extraction
        benchmark_data = _get_benchmark_data(prices_df, "SPY")

        assert benchmark_data.empty


class TestRebalanceDates:
    """Test rebalancing date generation."""

    def test_generate_rebalance_dates_monthly(self):
        """Test monthly rebalancing date generation."""
        # Create dates spanning multiple months
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        dates = [d.to_pydatetime() for d in dates]

        rebalance_dates = _generate_rebalance_dates(dates, "monthly")

        # Should have last trading day of each month
        assert len(rebalance_dates) >= 2  # At least Jan and Feb
        assert all(isinstance(d, datetime) for d in rebalance_dates)

    def test_generate_rebalance_dates_weekly(self):
        """Test weekly rebalancing date generation."""
        # Create dates spanning multiple weeks
        dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")
        dates = [d.to_pydatetime() for d in dates]

        rebalance_dates = _generate_rebalance_dates(dates, "weekly")

        # Should have every 5th trading day
        assert len(rebalance_dates) > 0
        assert all(isinstance(d, datetime) for d in rebalance_dates)

    def test_generate_rebalance_dates_daily(self):
        """Test daily rebalancing date generation."""
        dates = pd.date_range("2020-01-01", "2020-01-05", freq="D")
        dates = [d.to_pydatetime() for d in dates]

        rebalance_dates = _generate_rebalance_dates(dates, "daily")

        # Should have all dates
        assert len(rebalance_dates) == len(dates)
        assert rebalance_dates == dates


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_calculate_performance_metrics_basic(self):
        """Test basic performance metrics calculation."""
        # Create sample returns
        returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        benchmark_returns = pd.Series([0.005, -0.015, 0.025, 0.008, -0.008])
        transaction_costs = pd.Series([0.001, 0.001, 0.001, 0.001, 0.001])
        turnover = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1])

        metrics = _calculate_performance_metrics(
            returns, benchmark_returns, transaction_costs, turnover
        )

        # Check that key metrics are calculated
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "avg_transaction_cost" in metrics
        assert "avg_turnover" in metrics

        # Check that metrics are reasonable
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["annualized_return"], float)
        assert metrics["volatility"] >= 0
        assert metrics["max_drawdown"] <= 0  # Drawdown should be negative

    def test_calculate_performance_metrics_empty_returns(self):
        """Test performance metrics with empty returns."""
        empty_returns = pd.Series(dtype=float)
        empty_benchmark = pd.Series(dtype=float)
        empty_costs = pd.Series(dtype=float)
        empty_turnover = pd.Series(dtype=float)

        metrics = _calculate_performance_metrics(
            empty_returns, empty_benchmark, empty_costs, empty_turnover
        )

        assert metrics == {}

    def test_calculate_performance_metrics_with_benchmark(self):
        """Test performance metrics with benchmark comparison."""
        returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        benchmark_returns = pd.Series([0.005, -0.015, 0.025, 0.008, -0.008])
        transaction_costs = pd.Series([0.001, 0.001, 0.001, 0.001, 0.001])
        turnover = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1])

        metrics = _calculate_performance_metrics(
            returns, benchmark_returns, transaction_costs, turnover
        )

        # Check benchmark-related metrics
        assert "excess_return" in metrics
        assert "information_ratio" in metrics

        assert isinstance(metrics["excess_return"], float)
        assert isinstance(metrics["information_ratio"], float)


class TestSaveOutputs:
    """Test backtest output saving."""

    def test_save_backtest_outputs(self, tmp_path):
        """Test saving backtest outputs to files."""
        # Create sample result
        result = BacktestResult()
        result.portfolio_returns = pd.Series([0.01, -0.02, 0.03])
        result.equity_curve = pd.Series([1.0, 1.01, 0.99, 1.02])
        result.drawdown = pd.Series([0.0, 0.0, 0.02, 0.0])
        result.metrics = {"total_return": 0.05, "sharpe_ratio": 1.2}
        result.rebalance_dates = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
        ]

        # Create sample portfolio weights
        result.portfolio_weights = pd.DataFrame(
            {
                "AAPL": [0.5, 0.4, 0.6],
                "MSFT": [0.3, 0.4, 0.2],
                "GOOGL": [0.2, 0.2, 0.2],
                "date": result.rebalance_dates,
            }
        )

        # Create sample exposures
        result.exposures = pd.DataFrame(
            {
                "date": result.rebalance_dates,
                "AAPL": [0.5, 0.4, 0.6],
                "MSFT": [0.3, 0.4, 0.2],
            }
        )

        result.transaction_costs = pd.Series([0.001, 0.001, 0.001])
        result.turnover = pd.Series([0.1, 0.1, 0.1])

        # Save outputs
        _save_backtest_outputs(result, tmp_path)

        # Check that files were created
        assert (tmp_path / "backtest_perf.csv").exists()
        assert (tmp_path / "portfolio_alloc.csv").exists()
        assert (tmp_path / "exposures.csv").exists()
        assert (tmp_path / "equity_curve.csv").exists()

        # Check that files contain expected data
        perf_df = pd.read_csv(tmp_path / "backtest_perf.csv")
        assert "total_return" in perf_df.columns
        assert "sharpe_ratio" in perf_df.columns

        weights_df = pd.read_csv(tmp_path / "portfolio_alloc.csv")
        assert "AAPL" in weights_df.columns
        assert "MSFT" in weights_df.columns
        assert "date" in weights_df.columns


class TestWalkForward:
    """Test walk-forward backtest framework."""

    def test_walk_forward_basic(self, tmp_path):
        """Test basic walk-forward backtest functionality."""
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]

        # Create prices data
        prices_data = []
        for date in dates:
            for ticker in tickers:
                prices_data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "adj_close": 100.0 + np.random.randn() * 10,
                    }
                )
        prices_df = pd.DataFrame(prices_data)

        # Create returns data
        returns_data = []
        for date in dates[1:]:  # Skip first date (no return)
            for ticker in tickers:
                returns_data.append(
                    {"date": date, "ticker": ticker, "ret_d": np.random.randn() * 0.02}
                )
        returns_df = pd.DataFrame(returns_data)

        # Create covariance data (simplified)
        covariance_data = []
        for date in dates[1:]:
            for i, ticker1 in enumerate(tickers):
                for j, ticker2 in enumerate(tickers):
                    covariance_data.append(
                        {
                            "date": date,
                            "ticker1": ticker1,
                            "ticker2": ticker2,
                            "covariance": (
                                0.0004 if i == j else 0.0001
                            ),  # Diagonal higher
                        }
                    )
        covariance_df = pd.DataFrame(covariance_data)

        # Create constraints
        constraints = PortfolioConstraints(cardinality_k=2, weight_bounds=(0.0, 0.6))

        # Create config
        config = {
            "rebalance_freq": "monthly",
            "transaction_cost_bps": 5.0,
            "train_window": 30,
            "test_window": 10,
            "strategy": "mvo",
        }

        # Mock strategy function
        def mock_strategy(returns, cov, constraints):
            # Return equal weights for selected assets
            weights = np.zeros(len(tickers))
            weights[: constraints.cardinality_k] = 1.0 / constraints.cardinality_k
            return weights

        # Run backtest
        result = walk_forward(
            config=config,
            strategy_fn=mock_strategy,
            prices=prices_df,
            returns=returns_df,
            covariance=covariance_df,
            constraints=constraints,
            benchmark_ticker="AAPL",  # Use AAPL as benchmark
            output_dir=tmp_path,
        )

        # Check results
        assert isinstance(result, BacktestResult)
        assert not result.portfolio_returns.empty
        assert not result.equity_curve.empty
        assert not result.drawdown.empty
        assert len(result.metrics) > 0
        assert len(result.rebalance_dates) > 0

        # Check that output files were created
        assert (tmp_path / "backtest_perf.csv").exists()
        assert (tmp_path / "equity_curve.csv").exists()

    def test_walk_forward_insufficient_data(self):
        """Test walk-forward with insufficient data."""
        # Create minimal data
        dates = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        tickers = ["AAPL"]

        prices_df = pd.DataFrame(
            [{"date": dates[0], "ticker": tickers[0], "adj_close": 100.0}]
        )

        returns_df = pd.DataFrame(
            [{"date": dates[1], "ticker": tickers[0], "ret_d": 0.01}]
        )

        covariance_df = pd.DataFrame(
            [
                {
                    "date": dates[1],
                    "ticker1": tickers[0],
                    "ticker2": tickers[0],
                    "covariance": 0.0004,
                }
            ]
        )

        constraints = PortfolioConstraints(cardinality_k=1)
        config = {"rebalance_freq": "monthly", "train_window": 30, "test_window": 10}

        def mock_strategy(returns, cov, constraints):
            return np.array([1.0])

        result = walk_forward(
            config=config,
            strategy_fn=mock_strategy,
            prices=prices_df,
            returns=returns_df,
            covariance=covariance_df,
            constraints=constraints,
        )

        # Should handle insufficient data gracefully
        assert isinstance(result, BacktestResult)

    def test_walk_forward_strategy_failure(self):
        """Test walk-forward with strategy function failure."""
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        tickers = ["AAPL", "MSFT"]

        prices_data = []
        for date in dates:
            for ticker in tickers:
                prices_data.append({"date": date, "ticker": ticker, "adj_close": 100.0})
        prices_df = pd.DataFrame(prices_data)

        returns_data = []
        for date in dates[1:]:
            for ticker in tickers:
                returns_data.append({"date": date, "ticker": ticker, "ret_d": 0.01})
        returns_df = pd.DataFrame(returns_data)

        covariance_data = []
        for date in dates[1:]:
            for ticker1 in tickers:
                for ticker2 in tickers:
                    covariance_data.append(
                        {
                            "date": date,
                            "ticker1": ticker1,
                            "ticker2": ticker2,
                            "covariance": 0.0004 if ticker1 == ticker2 else 0.0001,
                        }
                    )
        covariance_df = pd.DataFrame(covariance_data)

        constraints = PortfolioConstraints(cardinality_k=2)
        config = {"rebalance_freq": "monthly", "train_window": 20, "test_window": 5}

        # Strategy that fails
        def failing_strategy(returns, cov, constraints):
            raise ValueError("Strategy failed")

        result = walk_forward(
            config=config,
            strategy_fn=failing_strategy,
            prices=prices_df,
            returns=returns_df,
            covariance=covariance_df,
            constraints=constraints,
        )

        # Should handle strategy failure gracefully
        assert isinstance(result, BacktestResult)


class TestPlotting:
    """Test plotting functions."""

    def test_plot_equity_curve_empty_data(self):
        """Test equity curve plotting with empty data."""
        result = BacktestResult()

        # Should handle empty data gracefully
        with patch("matplotlib.pyplot.show"):
            plot_equity_curve(result)

    def test_plot_portfolio_weights_empty_data(self):
        """Test portfolio weights plotting with empty data."""
        result = BacktestResult()

        # Should handle empty data gracefully
        with patch("matplotlib.pyplot.show"):
            plot_portfolio_weights(result)

    def test_plot_equity_curve_with_data(self, tmp_path):
        """Test equity curve plotting with data."""
        result = BacktestResult()
        result.equity_curve = pd.Series([1.0, 1.01, 0.99, 1.02, 1.05])
        result.drawdown = pd.Series([0.0, 0.0, 0.02, 0.0, 0.0])
        result.rebalance_dates = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
        ]
        result.benchmark_returns = pd.Series([0.01, -0.01, 0.02, 0.01])

        with patch("matplotlib.pyplot.show"):
            plot_equity_curve(result, output_dir=tmp_path)

        # Check that plot was saved
        assert (tmp_path / "equity_curve.png").exists()

    def test_plot_portfolio_weights_with_data(self, tmp_path):
        """Test portfolio weights plotting with data."""
        result = BacktestResult()
        result.portfolio_weights = pd.DataFrame(
            {
                "AAPL": [0.5, 0.4, 0.6],
                "MSFT": [0.3, 0.4, 0.2],
                "GOOGL": [0.2, 0.2, 0.2],
                "date": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                ],
            }
        )

        with patch("matplotlib.pyplot.show"):
            plot_portfolio_weights(result, output_dir=tmp_path)

        # Check that plot was saved
        assert (tmp_path / "portfolio_weights.png").exists()


class TestIntegration:
    """Integration tests for backtest module."""

    def test_full_backtest_workflow(self, tmp_path):
        """Test complete backtest workflow."""
        # Create realistic sample data
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        # Generate realistic price data
        np.random.seed(42)
        prices_data = []
        base_prices = {
            "AAPL": 150,
            "MSFT": 200,
            "GOOGL": 2500,
            "AMZN": 3000,
            "TSLA": 800,
        }

        for i, date in enumerate(dates):
            for ticker in tickers:
                # Add some trend and volatility
                trend = 0.0005 * i  # Slight upward trend
                noise = np.random.randn() * 0.02  # 2% daily volatility
                price = base_prices[ticker] * (1 + trend + noise)
                prices_data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "adj_close": max(price, 1.0),  # Ensure positive prices
                    }
                )

        prices_df = pd.DataFrame(prices_data)

        # Calculate returns
        returns_data = []
        for ticker in tickers:
            ticker_prices = prices_df[prices_df["ticker"] == ticker].sort_values("date")
            ticker_prices["ret_d"] = ticker_prices["adj_close"].pct_change()

            for _, row in ticker_prices.dropna().iterrows():
                returns_data.append(
                    {"date": row["date"], "ticker": ticker, "ret_d": row["ret_d"]}
                )

        returns_df = pd.DataFrame(returns_data)

        # Create covariance data
        covariance_data = []
        for date in dates[1:]:
            # Get returns for this date
            date_returns = returns_df[returns_df["date"] == date]
            if len(date_returns) == len(tickers):
                returns_vector = date_returns["ret_d"].values
                cov_matrix = (
                    np.outer(returns_vector, returns_vector) * 0.1
                )  # Simplified covariance

                for i, ticker1 in enumerate(tickers):
                    for j, ticker2 in enumerate(tickers):
                        covariance_data.append(
                            {
                                "date": date,
                                "ticker1": ticker1,
                                "ticker2": ticker2,
                                "covariance": cov_matrix[i, j],
                            }
                        )

        covariance_df = pd.DataFrame(covariance_data)

        # Create constraints
        constraints = PortfolioConstraints(
            cardinality_k=3, weight_bounds=(0.1, 0.4), no_short=True
        )

        # Create config
        config = {
            "rebalance_freq": "monthly",
            "transaction_cost_bps": 10.0,
            "turnover_cap": 0.3,
            "train_window": 60,
            "test_window": 20,
            "strategy": "mvo",
        }

        # Simple strategy: equal weights for top 3 assets by recent return
        def simple_strategy(returns, cov, constraints):
            # Get latest returns
            latest_date = returns["date"].max()
            latest_returns = returns[returns["date"] == latest_date]

            if latest_returns.empty:
                return np.ones(len(tickers)) / len(tickers)

            # Sort by return and select top K
            sorted_returns = latest_returns.sort_values("ret_d", ascending=False)
            selected_tickers = sorted_returns.head(constraints.cardinality_k)[
                "ticker"
            ].tolist()

            weights = np.zeros(len(tickers))
            for i, ticker in enumerate(tickers):
                if ticker in selected_tickers:
                    weights[i] = 1.0 / constraints.cardinality_k

            return weights

        # Run backtest
        result = walk_forward(
            config=config,
            strategy_fn=simple_strategy,
            prices=prices_df,
            returns=returns_df,
            covariance=covariance_df,
            constraints=constraints,
            benchmark_ticker="AAPL",
            output_dir=tmp_path,
        )

        # Verify results
        assert isinstance(result, BacktestResult)
        assert not result.portfolio_returns.empty
        assert not result.equity_curve.empty
        assert len(result.metrics) > 0

        # Check key metrics
        assert "total_return" in result.metrics
        assert "annualized_return" in result.metrics
        assert "volatility" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics

        # Check output files
        assert (tmp_path / "backtest_perf.csv").exists()
        assert (tmp_path / "equity_curve.csv").exists()

        # Verify equity curve is reasonable
        assert result.equity_curve.iloc[0] == 1.0  # Should start at 1.0
        assert result.equity_curve.min() > 0  # Should never go negative

        # Verify drawdown is reasonable
        assert result.drawdown.max() <= 0  # Drawdown should be non-positive
        assert result.drawdown.min() >= -1.0  # Drawdown should not exceed -100%
