import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qepo.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)

# Cursor Task: QUBO/Ising encoder for portfolio optimization


class QUBOEncoder:
    """Encode portfolio optimization problem as QUBO/Ising model.

    This class converts the portfolio optimization problem into a Quadratic
    Unconstrained Binary Optimization (QUBO) form suitable for quantum
    annealing or QAOA algorithms.
    """

    def __init__(self, num_assets: int, constraints: PortfolioConstraints):
        """Initialize QUBO encoder.

        Parameters
        ----------
        num_assets : int
            Number of assets in the universe
        constraints : PortfolioConstraints
            Portfolio constraints specification
        """
        self.num_assets = num_assets
        self.constraints = constraints
        self.num_qubits = self._calculate_num_qubits()

        logger.info(
            f"Initialized QUBO encoder: {num_assets} assets, {self.num_qubits} qubits"
        )

    def _calculate_num_qubits(self) -> int:
        """Calculate number of qubits needed for binary encoding.

        For K-sparse portfolio with N assets, we need N qubits (one per asset).
        Each qubit represents whether an asset is selected (1) or not (0).

        Returns
        -------
        int
            Number of qubits required
        """
        return self.num_assets

    def build_qubo(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        penalty_scale: float = 1.0,
    ) -> Dict[Tuple[int, int], float]:
        """Build QUBO matrix for portfolio optimization.

        The QUBO objective is:
        minimize: -sum(r_i * x_i) + lambda * sum(sum(Sigma_ij * x_i * x_j)) + penalties

        where:
        - r_i is expected return of asset i
        - x_i is binary variable (1 if asset i selected, 0 otherwise)
        - Sigma_ij is covariance between assets i and j
        - lambda is risk aversion parameter
        - penalties enforce constraints

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns for each asset (length = num_assets)
        covariance_matrix : np.ndarray
            Covariance matrix (num_assets x num_assets)
        risk_aversion : float, default=1.0
            Risk aversion parameter (lambda)
        penalty_scale : float, default=1.0
            Scaling factor for constraint penalties

        Returns
        -------
        Dict[Tuple[int, int], float]
            QUBO matrix as dictionary of (i,j) -> coefficient
        """
        if len(expected_returns) != self.num_assets:
            raise ValueError(
                f"Expected returns length {len(expected_returns)} != num_assets {self.num_assets}"
            )

        if covariance_matrix.shape != (self.num_assets, self.num_assets):
            raise ValueError(
                f"Covariance matrix shape {covariance_matrix.shape} != ({self.num_assets}, {self.num_assets})"
            )

        logger.info(
            f"Building QUBO with risk_aversion={risk_aversion}, penalty_scale={penalty_scale}"
        )

        # Initialize QUBO matrix
        qubo = {}

        # Linear terms (diagonal): -r_i * x_i
        for i in range(self.num_assets):
            qubo[(i, i)] = -expected_returns[i]

        # Quadratic terms: lambda * Sigma_ij * x_i * x_j
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                if i != j:
                    qubo[(i, j)] = risk_aversion * covariance_matrix[i, j]
                else:
                    # Add to diagonal term
                    qubo[(i, i)] += risk_aversion * covariance_matrix[i, i]

        # Add constraint penalties
        penalty_terms = self._build_constraint_penalties(penalty_scale)
        for (i, j), penalty in penalty_terms.items():
            if (i, j) in qubo:
                qubo[(i, j)] += penalty
            else:
                qubo[(i, j)] = penalty

        logger.info(f"Built QUBO with {len(qubo)} terms")
        return qubo

    def _build_constraint_penalties(
        self, penalty_scale: float
    ) -> Dict[Tuple[int, int], float]:
        """Build penalty terms for constraints.

        Parameters
        ----------
        penalty_scale : float
            Scaling factor for penalties

        Returns
        -------
        Dict[Tuple[int, int], float]
            Penalty terms as (i,j) -> coefficient
        """
        penalties = {}

        # Budget constraint: (sum(x_i) - 1)^2
        if self.constraints.budget_sum_to_one:
            budget_penalty = penalty_scale * 10.0

            # Linear terms: -2 * sum(x_i)
            for i in range(self.num_assets):
                if (i, i) in penalties:
                    penalties[(i, i)] -= 2 * budget_penalty
                else:
                    penalties[(i, i)] = -2 * budget_penalty

            # Quadratic terms: sum(x_i * x_j)
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    if i != j:
                        if (i, j) in penalties:
                            penalties[(i, j)] += budget_penalty
                        else:
                            penalties[(i, j)] = budget_penalty

            # Constant term: +1 (handled in energy calculation)

        # Cardinality constraint: (sum(x_i) - K)^2
        cardinality_penalty = penalty_scale * 5.0
        K = self.constraints.cardinality_k

        # Linear terms: -2K * sum(x_i)
        for i in range(self.num_assets):
            if (i, i) in penalties:
                penalties[(i, i)] -= 2 * K * cardinality_penalty
            else:
                penalties[(i, i)] = -2 * K * cardinality_penalty

        # Quadratic terms: sum(x_i * x_j)
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                if i != j:
                    if (i, j) in penalties:
                        penalties[(i, j)] += cardinality_penalty
                    else:
                        penalties[(i, j)] = cardinality_penalty

        # Constant term: +K^2 (handled in energy calculation)

        return penalties

    def decode_solution(
        self, binary_solution: Union[List[int], np.ndarray]
    ) -> np.ndarray:
        """Decode binary solution to portfolio weights.

        Parameters
        ----------
        binary_solution : Union[List[int], np.ndarray]
            Binary solution (0/1 for each asset)

        Returns
        -------
        np.ndarray
            Portfolio weights (normalized to sum to 1)
        """
        if len(binary_solution) != self.num_assets:
            raise ValueError(
                f"Solution length {len(binary_solution)} != num_assets {self.num_assets}"
            )

        # Convert to numpy array
        solution = np.array(binary_solution, dtype=float)

        # Count selected assets
        num_selected = np.sum(solution)

        if num_selected == 0:
            logger.warning("No assets selected in solution")
            return np.zeros(self.num_assets)

        # Equal weight among selected assets
        weights = solution / num_selected

        logger.info(f"Decoded solution: {num_selected} assets selected")
        return weights

    def calculate_energy(
        self,
        qubo: Dict[Tuple[int, int], float],
        binary_solution: Union[List[int], np.ndarray],
    ) -> float:
        """Calculate QUBO energy for a given solution.

        Energy = sum(Q_ij * x_i * x_j) for all (i,j) in QUBO

        Parameters
        ----------
        qubo : Dict[Tuple[int, int], float]
            QUBO matrix
        binary_solution : Union[List[int], np.ndarray]
            Binary solution

        Returns
        -------
        float
            QUBO energy
        """
        solution = np.array(binary_solution, dtype=float)
        energy = 0.0

        for (i, j), coefficient in qubo.items():
            energy += coefficient * solution[i] * solution[j]

        return energy

    def validate_solution(
        self, binary_solution: Union[List[int], np.ndarray]
    ) -> Tuple[bool, List[str]]:
        """Validate binary solution against constraints.

        Parameters
        ----------
        binary_solution : Union[List[int], np.ndarray]
            Binary solution

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_violations)
        """
        solution = np.array(binary_solution)
        violations = []

        # Check binary values (before converting to int)
        if not np.all((solution == 0) | (solution == 1)):
            violations.append("Solution contains non-binary values")

        # Convert to int for further processing
        solution = solution.astype(int)

        # Check cardinality
        num_selected = np.sum(solution)
        if num_selected > self.constraints.cardinality_k:
            violations.append(
                f"Cardinality violation: {num_selected} > {self.constraints.cardinality_k}"
            )

        # Check budget (should sum to 1 after normalization)
        if num_selected > 0:
            weights = solution / num_selected
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > 1e-6:
                violations.append(f"Budget violation: weights sum to {weight_sum}")

        is_valid = len(violations) == 0
        return is_valid, violations


def build_qubo(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    constraints: PortfolioConstraints,
    risk_aversion: float = 1.0,
    penalty_scale: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    """Convenience function to build QUBO matrix.

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset
    covariance_matrix : np.ndarray
        Covariance matrix
    constraints : PortfolioConstraints
        Portfolio constraints
    risk_aversion : float, default=1.0
        Risk aversion parameter
    penalty_scale : float, default=1.0
        Penalty scaling factor

    Returns
    -------
    Dict[Tuple[int, int], float]
        QUBO matrix
    """
    encoder = QUBOEncoder(len(expected_returns), constraints)
    return encoder.build_qubo(
        expected_returns, covariance_matrix, risk_aversion, penalty_scale
    )


def adaptive_penalty_scaling(
    qubo: Dict[Tuple[int, int], float],
    target_violations: float = 0.1,
    max_iterations: int = 10,
) -> float:
    """Adaptively scale penalty terms to achieve target constraint violations.

    This is a simplified version - in practice, you'd run multiple
    optimization attempts and adjust penalties based on results.

    Parameters
    ----------
    qubo : Dict[Tuple[int, int], float]
        Current QUBO matrix
    target_violations : float, default=0.1
        Target fraction of constraint violations
    max_iterations : int, default=10
        Maximum scaling iterations

    Returns
    -------
    float
        Recommended penalty scale factor
    """
    # This is a placeholder implementation
    # In practice, you would:
    # 1. Run optimization with current penalties
    # 2. Measure constraint violations
    # 3. Adjust penalties based on violation rates
    # 4. Repeat until target violations achieved

    logger.info(f"Adaptive penalty scaling: target_violations={target_violations}")

    # Simple heuristic: start with moderate scaling
    base_scale = 1.0

    # If penalties seem too weak (based on QUBO magnitude), increase
    penalty_magnitude = sum(abs(v) for v in qubo.values() if v < 0)
    if penalty_magnitude < 1.0:
        base_scale = 2.0

    return base_scale
