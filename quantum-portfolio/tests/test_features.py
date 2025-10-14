from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from qepo import features


class TestComputeReturns:
    """Test compute_returns function."""

    def test_compute_returns_basic(self):
        """Test basic log returns calculation."""
        # Create sample price data
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.DataFrame(
            {
                "date": dates.repeat(2),
                "ticker": ["AAPL", "MSFT"] * 5,
                "adj_close": [100, 200, 101, 201, 102, 202, 103, 203, 104, 204],
            }
        )

        returns = features.compute_returns(prices)

        # Check structure
        assert list(returns.columns) == ["date", "ticker", "ret_d"]
        assert len(returns) == 8  # 4 returns per ticker (first observation removed)

        # Check AAPL returns
        aapl_returns = returns[returns["ticker"] == "AAPL"]["ret_d"].values
        expected_aapl = np.log(
            np.array([101, 102, 103, 104]) / np.array([100, 101, 102, 103])
        )
        np.testing.assert_array_almost_equal(aapl_returns, expected_aapl)

    def test_compute_returns_empty_dataframe(self):
        """Test compute_returns with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["date", "ticker", "adj_close"])

        with pytest.raises(ValueError, match="Prices DataFrame is empty"):
            features.compute_returns(empty_df)

    def test_compute_returns_missing_columns(self):
        """Test compute_returns with missing columns."""
        prices = pd.DataFrame(
            {"date": [1, 2], "ticker": ["A", "B"]}
        )  # Missing adj_close

        with pytest.raises(ValueError, match="missing required columns"):
            features.compute_returns(prices)

    def test_compute_returns_single_ticker(self):
        """Test compute_returns with single ticker."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        prices = pd.DataFrame(
            {"date": dates, "ticker": ["AAPL"] * 3, "adj_close": [100, 101, 102]}
        )

        returns = features.compute_returns(prices)

        assert len(returns) == 2  # 2 returns (first observation removed)
        assert returns["ticker"].nunique() == 1
        assert returns["ticker"].iloc[0] == "AAPL"

    def test_compute_returns_unsorted_data(self):
        """Test compute_returns handles unsorted data correctly."""
        # Create unsorted data
        prices = pd.DataFrame(
            {
                "date": ["2020-01-03", "2020-01-01", "2020-01-02"],
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "adj_close": [102, 100, 101],
            }
        )

        returns = features.compute_returns(prices)

        # Should be sorted internally and return correct results
        assert len(returns) == 2
        expected_returns = np.log(np.array([101, 102]) / np.array([100, 101]))
        np.testing.assert_array_almost_equal(returns["ret_d"].values, expected_returns)


class TestComputeCovariance:
    """Test compute_covariance function."""

    def test_compute_covariance_basic(self):
        """Test basic covariance computation."""
        # Create sample returns data
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.DataFrame(
            {
                "date": dates.repeat(2),
                "ticker": ["AAPL", "MSFT"] * 10,
                "ret_d": np.random.normal(0, 0.01, 20),
            }
        )

        cov_matrices, mean_returns = features.compute_covariance(returns, window=5)

        # Check structure
        assert isinstance(cov_matrices, list)
        assert isinstance(mean_returns, pd.DataFrame)
        assert len(cov_matrices) > 0
        assert len(mean_returns) == len(cov_matrices)

        # Check covariance matrix structure
        cov_matrix = cov_matrices[0]
        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape[0] == cov_matrix.shape[1]  # Square matrix
        assert list(cov_matrix.index) == list(cov_matrix.columns)  # Same tickers

    def test_compute_covariance_empty_data(self):
        """Test compute_covariance with empty data."""
        empty_df = pd.DataFrame(columns=["date", "ticker", "ret_d"])

        with pytest.raises(ValueError, match="Returns DataFrame is empty"):
            features.compute_covariance(empty_df)

    def test_compute_covariance_insufficient_data(self):
        """Test compute_covariance with insufficient data."""
        # Create data with only 2 observations
        returns = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "ticker": ["AAPL", "AAPL"],
                "ret_d": [0.01, 0.02],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            features.compute_covariance(returns, window=252, min_periods=100)

    def test_compute_covariance_no_shrinkage(self):
        """Test covariance computation without shrinkage."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        returns = pd.DataFrame(
            {
                "date": dates.repeat(2),
                "ticker": ["AAPL", "MSFT"] * 20,
                "ret_d": np.random.normal(0, 0.01, 40),
            }
        )

        cov_matrices, _ = features.compute_covariance(
            returns, window=10, shrinkage=False
        )

        assert len(cov_matrices) > 0
        # Without shrinkage, should use sample covariance
        assert isinstance(cov_matrices[0], pd.DataFrame)

    @patch("qepo.features.LedoitWolf")
    def test_compute_covariance_shrinkage_fallback(self, mock_lw):
        """Test covariance computation with shrinkage fallback."""
        # Mock LedoitWolf to raise exception
        mock_lw.side_effect = Exception("Shrinkage failed")

        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        returns = pd.DataFrame(
            {
                "date": dates.repeat(2),
                "ticker": ["AAPL", "MSFT"] * 20,
                "ret_d": np.random.normal(0, 0.01, 40),
            }
        )

        # Should fallback to sample covariance
        cov_matrices, _ = features.compute_covariance(
            returns, window=10, shrinkage=True
        )

        assert len(cov_matrices) > 0
        assert isinstance(cov_matrices[0], pd.DataFrame)

    def test_compute_covariance_single_ticker(self):
        """Test covariance computation with single ticker."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * 10,
                "ret_d": np.random.normal(0, 0.01, 10),
            }
        )

        # Single ticker should still work (1x1 covariance matrix)
        cov_matrices, mean_returns = features.compute_covariance(returns, window=5)

        assert len(cov_matrices) > 0
        assert cov_matrices[0].shape == (1, 1)  # 1x1 matrix
        assert "AAPL" in cov_matrices[0].index
        assert "AAPL" in cov_matrices[0].columns


class TestValidateCovarianceStability:
    """Test validate_covariance_stability function."""

    def test_validate_stability_stable_matrices(self):
        """Test validation with stable covariance matrices."""
        # Create well-conditioned matrices
        cov_matrices = [
            pd.DataFrame(
                [[0.01, 0.005], [0.005, 0.01]], index=["A", "B"], columns=["A", "B"]
            ),
            pd.DataFrame(
                [[0.02, 0.01], [0.01, 0.02]], index=["A", "B"], columns=["A", "B"]
            ),
        ]

        results = features.validate_covariance_stability(cov_matrices)

        assert results["num_matrices"] == 2
        assert results["is_stable"] is True
        assert len(results["condition_numbers"]) == 2
        assert all(results["is_positive_definite"])

    def test_validate_stability_unstable_matrices(self):
        """Test validation with unstable covariance matrices."""
        # Create ill-conditioned matrix
        cov_matrices = [
            pd.DataFrame([[1e-10, 0], [0, 1e10]], index=["A", "B"], columns=["A", "B"])
        ]

        results = features.validate_covariance_stability(
            cov_matrices, condition_threshold=1e6
        )

        assert results["num_matrices"] == 1
        assert results["is_stable"] is False
        assert results["max_condition"] > 1e6

    def test_validate_stability_non_positive_definite(self):
        """Test validation with non-positive definite matrix."""
        # Create non-positive definite matrix
        cov_matrices = [
            pd.DataFrame(
                [[1, 2], [2, 1]], index=["A", "B"], columns=["A", "B"]
            )  # det < 0
        ]

        results = features.validate_covariance_stability(cov_matrices)

        assert results["num_matrices"] == 1
        assert results["is_stable"] is False
        assert not results["is_positive_definite"][0]

    def test_validate_stability_empty_list(self):
        """Test validation with empty list."""
        results = features.validate_covariance_stability([])

        assert "error" in results
        assert results["error"] == "No covariance matrices provided"

    def test_validate_stability_error_handling(self):
        """Test validation handles matrix computation errors."""
        # Create invalid matrix that will cause computation error
        cov_matrices = [
            pd.DataFrame([[np.nan, 0], [0, 1]], index=["A", "B"], columns=["A", "B"])
        ]

        results = features.validate_covariance_stability(cov_matrices)

        assert results["num_matrices"] == 1
        assert results["is_stable"] is False
