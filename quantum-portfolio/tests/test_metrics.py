"""Tests for metrics module."""

import numpy as np
import pytest

from qepo import metrics


class TestSharpeRatio:
    """Test sharpe_ratio function."""

    def test_sharpe_ratio_placeholder(self):
        """Test that sharpe_ratio returns 0.0 (placeholder implementation)."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        result = metrics.sharpe_ratio(returns)
        assert result == 0.0

    def test_sharpe_ratio_empty_sequence(self):
        """Test sharpe_ratio with empty sequence."""
        result = metrics.sharpe_ratio([])
        assert result == 0.0

    def test_sharpe_ratio_single_value(self):
        """Test sharpe_ratio with single value."""
        result = metrics.sharpe_ratio([0.05])
        assert result == 0.0

    def test_sharpe_ratio_numpy_array(self):
        """Test sharpe_ratio with numpy array."""
        returns = np.array([0.01, 0.02, -0.01, 0.03])
        result = metrics.sharpe_ratio(returns)
        assert result == 0.0
