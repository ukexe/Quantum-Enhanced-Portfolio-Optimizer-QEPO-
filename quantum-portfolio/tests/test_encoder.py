import numpy as np
import pytest

from qepo import constraints, encoder


class TestQUBOEncoder:
    """Test QUBOEncoder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_assets = 5
        self.constraints = constraints.PortfolioConstraints(
            cardinality_k=3, weight_bounds=(0.0, 0.5), budget_sum_to_one=True
        )
        self.encoder = encoder.QUBOEncoder(self.num_assets, self.constraints)

    def test_encoder_initialization(self):
        """Test QUBOEncoder initialization."""
        assert self.encoder.num_assets == 5
        assert self.encoder.num_qubits == 5
        assert self.encoder.constraints.cardinality_k == 3

    def test_calculate_num_qubits(self):
        """Test qubit calculation."""
        # For K-sparse portfolio, we need N qubits (one per asset)
        assert self.encoder._calculate_num_qubits() == 5

    def test_build_qubo_basic(self):
        """Test basic QUBO construction."""
        expected_returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])
        covariance_matrix = np.eye(5) * 0.01  # Identity matrix

        qubo = self.encoder.build_qubo(expected_returns, covariance_matrix)

        # Check structure
        assert isinstance(qubo, dict)
        assert len(qubo) > 0

        # Check linear terms (diagonal)
        for i in range(self.num_assets):
            assert (i, i) in qubo
            # Should be negative expected return
            assert qubo[(i, i)] < 0

    def test_build_qubo_with_risk_aversion(self):
        """Test QUBO construction with different risk aversion."""
        expected_returns = np.array([0.1, 0.15, 0.12, 0.08, 0.14])
        covariance_matrix = np.array(
            [
                [0.01, 0.005, 0.003, 0.002, 0.004],
                [0.005, 0.01, 0.004, 0.003, 0.005],
                [0.003, 0.004, 0.01, 0.002, 0.003],
                [0.002, 0.003, 0.002, 0.01, 0.002],
                [0.004, 0.005, 0.003, 0.002, 0.01],
            ]
        )

        qubo_low_risk = self.encoder.build_qubo(
            expected_returns, covariance_matrix, risk_aversion=0.5
        )
        qubo_high_risk = self.encoder.build_qubo(
            expected_returns, covariance_matrix, risk_aversion=2.0
        )

        # Higher risk aversion should increase quadratic terms
        for i, j in qubo_low_risk:
            if i != j:  # Quadratic terms
                assert qubo_high_risk[(i, j)] > qubo_low_risk[(i, j)]

    def test_build_qubo_invalid_inputs(self):
        """Test QUBO construction with invalid inputs."""
        expected_returns = np.array([0.1, 0.15])  # Wrong length
        covariance_matrix = np.eye(5)

        with pytest.raises(ValueError, match="Expected returns length"):
            self.encoder.build_qubo(expected_returns, covariance_matrix)

        expected_returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])
        covariance_matrix = np.eye(3)  # Wrong shape

        with pytest.raises(ValueError, match="Covariance matrix shape"):
            self.encoder.build_qubo(expected_returns, covariance_matrix)

    def test_decode_solution_basic(self):
        """Test solution decoding."""
        binary_solution = [1, 0, 1, 0, 1]  # Select assets 0, 2, 4

        weights = self.encoder.decode_solution(binary_solution)

        assert len(weights) == 5
        assert np.sum(weights) == 1.0  # Should sum to 1
        assert weights[0] == 1 / 3  # Equal weight among selected
        assert weights[1] == 0.0  # Not selected
        assert weights[2] == 1 / 3
        assert weights[3] == 0.0
        assert weights[4] == 1 / 3

    def test_decode_solution_no_selection(self):
        """Test solution decoding with no assets selected."""
        binary_solution = [0, 0, 0, 0, 0]

        weights = self.encoder.decode_solution(binary_solution)

        assert np.all(weights == 0.0)

    def test_decode_solution_invalid_length(self):
        """Test solution decoding with invalid length."""
        binary_solution = [1, 0, 1]  # Wrong length

        with pytest.raises(ValueError, match="Solution length"):
            self.encoder.decode_solution(binary_solution)

    def test_calculate_energy(self):
        """Test QUBO energy calculation."""
        # Simple QUBO: minimize -x0 - x1 + x0*x1
        qubo = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 1.0}

        # Solution [1, 0]: energy = -1*1*1 + -1*0*0 + 1*1*0 = -1
        energy_1_0 = self.encoder.calculate_energy(qubo, [1, 0])
        assert energy_1_0 == -1.0

        # Solution [0, 1]: energy = -1*0*0 + -1*1*1 + 1*0*1 = -1
        energy_0_1 = self.encoder.calculate_energy(qubo, [0, 1])
        assert energy_0_1 == -1.0

        # Solution [1, 1]: energy = -1*1*1 + -1*1*1 + 1*1*1 = -1
        energy_1_1 = self.encoder.calculate_energy(qubo, [1, 1])
        assert energy_1_1 == -1.0

        # Solution [0, 0]: energy = -1*0*0 + -1*0*0 + 1*0*0 = 0
        energy_0_0 = self.encoder.calculate_energy(qubo, [0, 0])
        assert energy_0_0 == 0.0

    def test_validate_solution_valid(self):
        """Test solution validation with valid solution."""
        binary_solution = [1, 0, 1, 0, 1]  # 3 assets selected (within K=3)

        is_valid, violations = self.encoder.validate_solution(binary_solution)

        assert is_valid
        assert len(violations) == 0

    def test_validate_solution_cardinality_violation(self):
        """Test solution validation with cardinality violation."""
        binary_solution = [1, 1, 1, 1, 0]  # 4 assets selected (exceeds K=3)

        is_valid, violations = self.encoder.validate_solution(binary_solution)

        assert not is_valid
        assert any("Cardinality violation" in v for v in violations)

    def test_validate_solution_non_binary(self):
        """Test solution validation with non-binary values."""
        binary_solution = [1, 0.5, 1, 0, 1]  # Contains 0.5

        is_valid, violations = self.encoder.validate_solution(binary_solution)

        assert not is_valid
        assert any("non-binary values" in v for v in violations)


class TestBuildQUBOFunction:
    """Test build_qubo convenience function."""

    def test_build_qubo_function(self):
        """Test build_qubo convenience function."""
        expected_returns = np.array([0.1, 0.15, 0.12])
        covariance_matrix = np.eye(3) * 0.01
        portfolio_constraints = constraints.PortfolioConstraints(cardinality_k=2)

        qubo = encoder.build_qubo(
            expected_returns, covariance_matrix, portfolio_constraints
        )

        assert isinstance(qubo, dict)
        assert len(qubo) > 0

    def test_build_qubo_with_parameters(self):
        """Test build_qubo with custom parameters."""
        expected_returns = np.array([0.1, 0.15])
        covariance_matrix = np.eye(2) * 0.01
        portfolio_constraints = constraints.PortfolioConstraints(cardinality_k=1)

        qubo = encoder.build_qubo(
            expected_returns,
            covariance_matrix,
            portfolio_constraints,
            risk_aversion=2.0,
            penalty_scale=5.0,
        )

        assert isinstance(qubo, dict)


class TestAdaptivePenaltyScaling:
    """Test adaptive penalty scaling function."""

    def test_adaptive_penalty_scaling_basic(self):
        """Test basic adaptive penalty scaling."""
        qubo = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 0.5}

        scale = encoder.adaptive_penalty_scaling(qubo)

        assert isinstance(scale, float)
        assert scale > 0

    def test_adaptive_penalty_scaling_weak_penalties(self):
        """Test scaling with weak penalties."""
        # QUBO with very small penalty terms
        qubo = {(0, 0): -0.001, (1, 1): -0.001, (0, 1): 0.0005}

        scale = encoder.adaptive_penalty_scaling(qubo)

        # Should recommend higher scaling for weak penalties
        assert scale >= 1.0

    def test_adaptive_penalty_scaling_custom_parameters(self):
        """Test adaptive scaling with custom parameters."""
        qubo = {(0, 0): -1.0, (1, 1): -1.0}

        scale = encoder.adaptive_penalty_scaling(
            qubo, target_violations=0.05, max_iterations=5
        )

        assert isinstance(scale, float)
        assert scale > 0


class TestConstraintPenalties:
    """Test constraint penalty construction."""

    def test_budget_constraint_penalty(self):
        """Test budget constraint penalty terms."""
        portfolio_constraints = constraints.PortfolioConstraints(
            budget_sum_to_one=True, cardinality_k=5
        )
        encoder_obj = encoder.QUBOEncoder(3, portfolio_constraints)

        penalties = encoder_obj._build_constraint_penalties(penalty_scale=1.0)

        # Should have penalty terms for budget constraint
        assert len(penalties) > 0

        # Check that linear terms are negative (encourage selection)
        for i in range(3):
            if (i, i) in penalties:
                assert penalties[(i, i)] < 0

    def test_cardinality_constraint_penalty(self):
        """Test cardinality constraint penalty terms."""
        portfolio_constraints = constraints.PortfolioConstraints(cardinality_k=2)
        encoder_obj = encoder.QUBOEncoder(4, portfolio_constraints)

        penalties = encoder_obj._build_constraint_penalties(penalty_scale=1.0)

        # Should have penalty terms for cardinality constraint
        assert len(penalties) > 0

        # Check structure of penalty terms
        linear_terms = sum(1 for (i, j) in penalties.keys() if i == j)
        quadratic_terms = sum(1 for (i, j) in penalties.keys() if i != j)

        assert linear_terms > 0  # Should have linear terms
        assert quadratic_terms > 0  # Should have quadratic terms

    def test_penalty_scaling(self):
        """Test penalty scaling affects magnitude."""
        portfolio_constraints = constraints.PortfolioConstraints(cardinality_k=2)
        encoder_obj = encoder.QUBOEncoder(3, portfolio_constraints)

        penalties_low = encoder_obj._build_constraint_penalties(penalty_scale=1.0)
        penalties_high = encoder_obj._build_constraint_penalties(penalty_scale=5.0)

        # Higher penalty scale should increase magnitude
        for i, j in penalties_low:
            if (i, j) in penalties_high:
                assert abs(penalties_high[(i, j)]) >= abs(penalties_low[(i, j)])
