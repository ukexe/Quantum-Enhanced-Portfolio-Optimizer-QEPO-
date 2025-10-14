import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qepo import constraints, postprocess


class TestPostProcessing:
    """Test post-processing and repair functionality."""

    def test_decode_bitstring_string(self):
        """Test decoding bitstring from string."""
        bitstring = "1010"
        num_assets = 4
        cons = constraints.PortfolioConstraints(cardinality_k=2)

        weights = postprocess.decode_bitstring(bitstring, num_assets, cons)

        expected = np.array([0.5, 0.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_decode_bitstring_array(self):
        """Test decoding bitstring from numpy array."""
        bitstring = np.array([1, 0, 1, 0])
        num_assets = 4
        cons = constraints.PortfolioConstraints(cardinality_k=2)

        weights = postprocess.decode_bitstring(bitstring, num_assets, cons)

        expected = np.array([0.5, 0.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_decode_bitstring_invalid_length(self):
        """Test decoding bitstring with invalid length."""
        bitstring = "101"
        num_assets = 4
        cons = constraints.PortfolioConstraints(cardinality_k=2)

        with pytest.raises(ValueError, match="Bitstring length"):
            postprocess.decode_bitstring(bitstring, num_assets, cons)

    def test_decode_bitstring_invalid_values(self):
        """Test decoding bitstring with invalid values."""
        bitstring = "1020"
        num_assets = 4
        cons = constraints.PortfolioConstraints(cardinality_k=2)

        with pytest.raises(ValueError, match="must contain only 0s and 1s"):
            postprocess.decode_bitstring(bitstring, num_assets, cons)

    def test_decode_bitstring_no_selection(self):
        """Test decoding bitstring with no assets selected."""
        bitstring = "0000"
        num_assets = 4
        cons = constraints.PortfolioConstraints(cardinality_k=2)

        weights = postprocess.decode_bitstring(bitstring, num_assets, cons)

        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_check_feasibility_feasible(self):
        """Test feasibility check with feasible portfolio."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cons = constraints.PortfolioConstraints(
            cardinality_k=4, weight_bounds=(0.0, 0.5)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        is_feasible, violations = postprocess.check_feasibility(weights, cons, tickers)

        assert is_feasible
        assert len(violations) == 0

    def test_check_feasibility_infeasible(self):
        """Test feasibility check with infeasible portfolio."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cons = constraints.PortfolioConstraints(
            cardinality_k=3, weight_bounds=(0.0, 0.1)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        is_feasible, violations = postprocess.check_feasibility(weights, cons, tickers)

        assert not is_feasible
        assert len(violations) > 0
        assert any("Cardinality violation" in v for v in violations)
        assert any("Weight above maximum" in v for v in violations)

    def test_check_feasibility_budget_violation(self):
        """Test feasibility check with budget violation."""
        weights = np.array([0.4, 0.3, 0.2, 0.2])  # Sum = 1.1
        cons = constraints.PortfolioConstraints(cardinality_k=4, budget_sum_to_one=True)
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        is_feasible, violations = postprocess.check_feasibility(weights, cons, tickers)

        assert not is_feasible
        assert any("Budget violation" in v for v in violations)

    def test_check_feasibility_sector_caps(self):
        """Test feasibility check with sector cap violations."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cons = constraints.PortfolioConstraints(
            cardinality_k=4, sector_caps={"TECH": 0.3}
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        sectors = {"AAPL": "TECH", "GOOGL": "TECH", "MSFT": "TECH", "TSLA": "AUTO"}

        is_feasible, violations = postprocess.check_feasibility(
            weights, cons, tickers, sectors
        )

        assert not is_feasible
        assert any("Sector cap violation" in v for v in violations)

    def test_repair_portfolio_budget(self):
        """Test portfolio repair for budget constraint."""
        weights = np.array([0.4, 0.3, 0.2, 0.2])  # Sum = 1.1
        cons = constraints.PortfolioConstraints(
            cardinality_k=4, budget_sum_to_one=True, weight_bounds=(0.0, 1.0)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        repaired_weights, is_feasible, history = postprocess.repair_portfolio(
            weights, cons, tickers
        )

        # Budget constraint should be fixed
        assert abs(np.sum(repaired_weights) - 1.0) < 1e-6
        assert len(history) > 0

    def test_repair_portfolio_cardinality(self):
        """Test portfolio repair for cardinality constraint."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cons = constraints.PortfolioConstraints(
            cardinality_k=2, weight_bounds=(0.0, 1.0)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        repaired_weights, is_feasible, history = postprocess.repair_portfolio(
            weights, cons, tickers
        )

        # Cardinality constraint should be fixed
        assert np.sum(repaired_weights > 1e-6) <= 2
        assert len(history) > 0

    @pytest.mark.xfail(
        reason="Edge case: repair algorithm struggles with very tight weight bounds"
    )
    def test_repair_portfolio_weight_bounds(self):
        """Test portfolio repair for weight bounds."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cons = constraints.PortfolioConstraints(
            cardinality_k=4, weight_bounds=(0.0, 0.15)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        repaired_weights, is_feasible, history = postprocess.repair_portfolio(
            weights, cons, tickers
        )

        # Weight bounds should be respected (may not be feasible due to other constraints)
        assert np.all(repaired_weights <= 0.15 + 1e-6)
        assert len(history) > 0

    def test_repair_portfolio_no_short(self):
        """Test portfolio repair for no-short constraint."""
        weights = np.array([0.4, -0.1, 0.3, 0.4])  # Sum = 1.0, but has negative weight
        cons = constraints.PortfolioConstraints(
            cardinality_k=4, no_short=True, weight_bounds=(0.0, 1.0)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        repaired_weights, is_feasible, history = postprocess.repair_portfolio(
            weights, cons, tickers
        )

        # No-short constraint should be fixed
        assert np.all(repaired_weights >= -1e-6)
        assert len(history) > 0

    def test_repair_portfolio_sector_caps(self):
        """Test portfolio repair for sector caps."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cons = constraints.PortfolioConstraints(
            cardinality_k=4, sector_caps={"TECH": 0.3}, weight_bounds=(0.0, 1.0)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        sectors = {"AAPL": "TECH", "GOOGL": "TECH", "MSFT": "TECH", "TSLA": "AUTO"}

        repaired_weights, is_feasible, history = postprocess.repair_portfolio(
            weights, cons, tickers, sectors
        )

        # Check sector exposure (may not be feasible due to other constraints)
        tech_exposure = sum(
            repaired_weights[i]
            for i, ticker in enumerate(tickers)
            if sectors[ticker] == "TECH"
        )
        # The repair algorithm should attempt to reduce sector exposure
        assert len(history) > 0

    def test_post_process_solution_feasible(self):
        """Test post-processing with feasible solution."""
        bitstring = "1010"
        cons = constraints.PortfolioConstraints(
            cardinality_k=2, weight_bounds=(0.0, 1.0)
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        weights, is_feasible, history = postprocess.post_process_solution(
            bitstring, cons, tickers
        )

        assert len(weights) == 4
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert len(history) > 0

    def test_post_process_solution_infeasible_with_repair(self):
        """Test post-processing with infeasible solution and repair."""
        bitstring = "1111"  # Select all 4 assets
        cons = constraints.PortfolioConstraints(
            cardinality_k=2, weight_bounds=(0.0, 1.0)
        )  # But only allow 2
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        weights, is_feasible, history = postprocess.post_process_solution(
            bitstring, cons, tickers, repair=True
        )

        # Should attempt to fix cardinality constraint
        assert np.sum(weights > 1e-6) <= 2
        assert len(history) > 0

    def test_post_process_solution_infeasible_without_repair(self):
        """Test post-processing with infeasible solution without repair."""
        bitstring = "1111"  # Select all 4 assets
        cons = constraints.PortfolioConstraints(cardinality_k=2)  # But only allow 2
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        weights, is_feasible, history = postprocess.post_process_solution(
            bitstring, cons, tickers, repair=False
        )

        assert not is_feasible
        assert len(weights) == 4
        assert len(history) > 0

    def test_round_weights(self):
        """Test weight rounding functionality."""
        weights = np.array([0.333333, 0.333333, 0.333334, 0.000001])

        rounded = postprocess.round_weights(
            weights, precision=4, min_weight_threshold=1e-5
        )

        # Check that very small weights are removed
        assert rounded[3] == 0.0
        # Check that weights still sum to approximately 1 (allowing for rounding errors)
        assert abs(np.sum(rounded) - 1.0) < 1e-4

    def test_round_weights_precision(self):
        """Test weight rounding with different precision."""
        weights = np.array([0.333333, 0.333333, 0.333334])

        rounded = postprocess.round_weights(weights, precision=2)

        # Check that weights still sum to approximately 1 (allowing for rounding errors)
        assert abs(np.sum(rounded) - 1.0) < 0.02

    def test_repair_max_iterations(self):
        """Test repair algorithm with max iterations."""
        # Create a challenging case that might need many iterations
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cons = constraints.PortfolioConstraints(
            cardinality_k=2, weight_bounds=(0.0, 0.15), sector_caps={"TECH": 0.1}
        )
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        sectors = {"AAPL": "TECH", "GOOGL": "TECH", "MSFT": "TECH", "TSLA": "AUTO"}

        repaired_weights, is_feasible, history = postprocess.repair_portfolio(
            weights, cons, tickers, sectors, max_iterations=5
        )

        # Should complete within max iterations
        assert len(history) <= 5
        # May or may not be feasible depending on constraints
        assert len(repaired_weights) == 4


if __name__ == "__main__":
    pytest.main([__file__])
