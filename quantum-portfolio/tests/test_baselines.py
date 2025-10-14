from unittest.mock import Mock, patch

import numpy as np
import pytest

from qepo import baselines, constraints


class TestClassicalBaselines:
    """Test ClassicalBaselines class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.expected_returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])
        self.covariance_matrix = np.array(
            [
                [0.01, 0.005, 0.003, 0.002, 0.004],
                [0.005, 0.01, 0.004, 0.001, 0.003],
                [0.003, 0.004, 0.01, 0.002, 0.002],
                [0.002, 0.001, 0.002, 0.01, 0.001],
                [0.004, 0.003, 0.002, 0.001, 0.01],
            ]
        )
        self.constraints = constraints.PortfolioConstraints(
            cardinality_k=3,
            weight_bounds=(0.0, 0.5),
            budget_sum_to_one=True,
            no_short=True,
        )
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    @patch("qepo.baselines.EfficientFrontier")
    @patch("time.time", side_effect=[0.0, 1.5])  # Mock time to simulate 1.5s solve time
    def test_mvo_solve_success(self, mock_time, mock_ef_class):
        """Test successful MVO solve with PyPortfolioOpt."""
        # Mock EfficientFrontier
        mock_ef = Mock()
        mock_ef.max_sharpe.return_value = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        mock_ef.clean_weights.return_value = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        mock_ef.portfolio_performance.return_value = (
            0.12,
            0.08,
            1.5,
        )  # return, vol, sharpe
        mock_ef_class.return_value = mock_ef

        baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)
        result = baselines_obj.mvo_solve(
            self.expected_returns,
            self.covariance_matrix,
            self.constraints,
            tickers=self.tickers,
        )

        assert result["method"] == "MVO_PyPortfolioOpt"
        assert result["status"] == "success"
        assert len(result["weights"]) == 5
        assert result["expected_return"] == 0.12
        assert result["volatility"] == 0.08
        assert result["sharpe_ratio"] == 1.5
        assert result["solve_time"] > 0

    @patch("qepo.baselines.EfficientFrontier")
    def test_mvo_solve_failure(self, mock_ef_class):
        """Test MVO solve failure handling."""
        # Mock EfficientFrontier to raise exception
        mock_ef_class.side_effect = Exception("Optimization failed")

        baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)
        result = baselines_obj.mvo_solve(
            self.expected_returns,
            self.covariance_matrix,
            self.constraints,
            tickers=self.tickers,
        )

        assert result["method"] == "MVO_PyPortfolioOpt"
        assert result["status"] == "failed"
        assert np.all(result["weights"] == 0)
        assert result["expected_return"] == 0.0
        assert "Optimization failed" in result["error"]

    @patch("qepo.baselines.cp.Problem")
    @pytest.mark.xfail(reason="Complex CVXPY mocking issues")
    def test_mvo_cvxpy_solve_success(self, mock_problem_class):
        """Test successful MVO solve with CVXPY."""
        # Mock CVXPY problem
        mock_problem = Mock()
        mock_problem.solve.return_value = None
        mock_problem.status = "optimal"

        # Mock variable values
        mock_w = Mock()
        mock_w.value = np.array([0.4, 0.3, 0.3, 0.0, 0.0])

        # Mock the problem creation
        mock_problem_class.return_value = mock_problem

        with patch("qepo.baselines.cp.Variable", return_value=mock_w):
            baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)
            result = baselines_obj.mvo_cvxpy_solve(
                self.expected_returns,
                self.covariance_matrix,
                self.constraints,
                tickers=self.tickers,
            )

        assert result["method"] == "MVO_CVXPY"
        assert result["status"] == "success"
        assert len(result["weights"]) == 5
        assert result["solve_time"] > 0

    @patch("qepo.baselines.cp.Problem")
    def test_mvo_cvxpy_solve_failure(self, mock_problem_class):
        """Test CVXPY MVO solve failure handling."""
        # Mock CVXPY problem to fail
        mock_problem = Mock()
        mock_problem.solve.return_value = None
        mock_problem.status = "infeasible"
        mock_problem_class.return_value = mock_problem

        with patch("qepo.baselines.cp.Variable"):
            baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)
            result = baselines_obj.mvo_cvxpy_solve(
                self.expected_returns,
                self.covariance_matrix,
                self.constraints,
                tickers=self.tickers,
            )

        assert result["method"] == "MVO_CVXPY"
        assert result["status"] == "failed"
        assert np.all(result["weights"] == 0)

    @patch("qepo.baselines.cp.Problem")
    @pytest.mark.xfail(reason="Complex CVXPY mocking issues")
    def test_greedy_k_select_success(self, mock_problem_class):
        """Test successful greedy K-select."""
        # Mock CVXPY problem for greedy optimization
        mock_problem = Mock()
        mock_problem.solve.return_value = None
        mock_problem.status = "optimal"

        # Mock variable values for selected assets
        mock_w_selected = Mock()
        mock_w_selected.value = np.array([0.5, 0.3, 0.2])  # 3 selected assets

        mock_problem_class.return_value = mock_problem

        with patch("qepo.baselines.cp.Variable", return_value=mock_w_selected):
            baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)
            result = baselines_obj.greedy_k_select(
                self.expected_returns,
                self.covariance_matrix,
                self.constraints,
                tickers=self.tickers,
            )

        assert result["method"] == "Greedy_K_Select"
        assert result["status"] == "success"
        assert len(result["weights"]) == 5
        assert result["K"] == 3
        assert "selected_assets" in result
        assert len(result["selected_assets"]) == 3

    @patch("qepo.baselines.cp.Problem")
    def test_greedy_k_select_failure(self, mock_problem_class):
        """Test greedy K-select failure handling."""
        # Mock CVXPY problem to fail
        mock_problem = Mock()
        mock_problem.solve.return_value = None
        mock_problem.status = "infeasible"
        mock_problem_class.return_value = mock_problem

        with patch("qepo.baselines.cp.Variable"):
            baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)
            result = baselines_obj.greedy_k_select(
                self.expected_returns,
                self.covariance_matrix,
                self.constraints,
                tickers=self.tickers,
            )

        assert result["method"] == "Greedy_K_Select"
        assert result["status"] == "failed"
        assert np.all(result["weights"] == 0)

    def test_add_pypfopt_constraints(self):
        """Test adding constraints to PyPortfolioOpt EfficientFrontier."""
        baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)

        # Mock EfficientFrontier
        mock_ef = Mock()

        # Test constraint addition
        baselines_obj._add_pypfopt_constraints(mock_ef, self.constraints)

        # Verify constraints were added
        assert mock_ef.add_constraint.call_count >= 2  # At least weight bounds

    @patch("qepo.baselines.mlflow.start_run")
    @patch("qepo.baselines.mlflow.set_tag")
    @patch("qepo.baselines.mlflow.log_params")
    @patch("qepo.baselines.mlflow.log_metrics")
    @patch("qepo.baselines.mlflow.log_table")
    def test_log_mvo_results(
        self,
        mock_log_table,
        mock_log_metrics,
        mock_log_params,
        mock_set_tag,
        mock_start_run,
    ):
        """Test MLflow logging of MVO results."""
        # Mock MLflow context manager
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None

        result = {
            "method": "MVO_PyPortfolioOpt",
            "weights": np.array([0.4, 0.3, 0.3, 0.0, 0.0]),
            "expected_return": 0.12,
            "volatility": 0.08,
            "sharpe_ratio": 1.5,
            "solve_time": 0.5,
            "num_assets": 5,
            "risk_aversion": 1.0,
        }

        baselines_obj = baselines.ClassicalBaselines(mlflow_logging=True)
        baselines_obj._log_mvo_results(result, self.constraints)

        # Verify MLflow methods were called
        mock_start_run.assert_called_once()
        mock_set_tag.assert_called()
        mock_log_params.assert_called()
        mock_log_metrics.assert_called()
        mock_log_table.assert_called()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.expected_returns = np.array([0.1, 0.15, 0.12])
        self.covariance_matrix = np.eye(3) * 0.01
        self.constraints = constraints.PortfolioConstraints(cardinality_k=2)

    @patch("qepo.baselines.ClassicalBaselines")
    def test_mvo_solve_function_pypfopt(self, mock_baselines_class):
        """Test mvo_solve convenience function with PyPortfolioOpt."""
        mock_baselines = Mock()
        mock_baselines.mvo_solve.return_value = {
            "method": "MVO_PyPortfolioOpt",
            "status": "success",
        }
        mock_baselines_class.return_value = mock_baselines

        result = baselines.mvo_solve(
            self.expected_returns,
            self.covariance_matrix,
            self.constraints,
            method="pypfopt",
        )

        mock_baselines_class.assert_called_once()
        mock_baselines.mvo_solve.assert_called_once()
        assert result["method"] == "MVO_PyPortfolioOpt"

    @patch("qepo.baselines.ClassicalBaselines")
    def test_mvo_solve_function_cvxpy(self, mock_baselines_class):
        """Test mvo_solve convenience function with CVXPY."""
        mock_baselines = Mock()
        mock_baselines.mvo_cvxpy_solve.return_value = {
            "method": "MVO_CVXPY",
            "status": "success",
        }
        mock_baselines_class.return_value = mock_baselines

        result = baselines.mvo_solve(
            self.expected_returns,
            self.covariance_matrix,
            self.constraints,
            method="cvxpy",
        )

        mock_baselines_class.assert_called_once()
        mock_baselines.mvo_cvxpy_solve.assert_called_once()
        assert result["method"] == "MVO_CVXPY"

    def test_mvo_solve_function_invalid_method(self):
        """Test mvo_solve with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            baselines.mvo_solve(
                self.expected_returns,
                self.covariance_matrix,
                self.constraints,
                method="invalid",
            )

    @patch("qepo.baselines.ClassicalBaselines")
    def test_greedy_k_select_function(self, mock_baselines_class):
        """Test greedy_k_select convenience function."""
        mock_baselines = Mock()
        mock_baselines.greedy_k_select.return_value = {
            "method": "Greedy_K_Select",
            "status": "success",
        }
        mock_baselines_class.return_value = mock_baselines

        result = baselines.greedy_k_select(
            self.expected_returns, self.covariance_matrix, self.constraints
        )

        mock_baselines_class.assert_called_once()
        mock_baselines.greedy_k_select.assert_called_once()
        assert result["method"] == "Greedy_K_Select"


class TestPerformanceTracking:
    """Test performance tracking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.expected_returns = np.array([0.1, 0.15, 0.12])
        self.covariance_matrix = np.eye(3) * 0.01
        self.constraints = constraints.PortfolioConstraints(cardinality_k=2)

    def test_solve_time_tracking(self):
        """Test that solve times are properly tracked."""
        baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)

        # Mock a successful solve
        with patch("qepo.baselines.EfficientFrontier") as mock_ef_class, patch(
            "time.time", side_effect=[0.0, 1.5]
        ) as mock_time:
            mock_ef = Mock()
            mock_ef.max_sharpe.return_value = {"Asset_0": 0.5, "Asset_1": 0.5}
            mock_ef.clean_weights.return_value = {"Asset_0": 0.5, "Asset_1": 0.5}
            mock_ef.portfolio_performance.return_value = (0.12, 0.08, 1.5)
            mock_ef_class.return_value = mock_ef

            result = baselines_obj.mvo_solve(
                self.expected_returns, self.covariance_matrix, self.constraints
            )

        assert "solve_time" in result
        assert result["solve_time"] > 0
        assert isinstance(result["solve_time"], float)

    def test_performance_metrics_calculation(self):
        """Test that performance metrics are calculated correctly."""
        baselines_obj = baselines.ClassicalBaselines(mlflow_logging=False)

        # Mock a successful solve
        with patch("qepo.baselines.EfficientFrontier") as mock_ef_class:
            mock_ef = Mock()
            mock_ef.max_sharpe.return_value = {"Asset_0": 0.5, "Asset_1": 0.5}
            mock_ef.clean_weights.return_value = {"Asset_0": 0.5, "Asset_1": 0.5}
            mock_ef.portfolio_performance.return_value = (0.12, 0.08, 1.5)
            mock_ef_class.return_value = mock_ef

            result = baselines_obj.mvo_solve(
                self.expected_returns, self.covariance_matrix, self.constraints
            )

        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert result["expected_return"] == 0.12
        assert result["volatility"] == 0.08
        assert result["sharpe_ratio"] == 1.5
