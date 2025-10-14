from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Cursor Task: Portfolio constraint modeling and validation


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints.

    Attributes
    ----------
    budget_sum_to_one : bool
            If True, weights must sum to 1.0 (fully invested).
    cardinality_k : int
            Maximum number of assets to select (K-sparsity).
    weight_bounds : Tuple[float, float]
            Min and max weight for each asset [w_min, w_max].
    no_short : bool
            If True, all weights must be non-negative.
    sector_caps : Dict[str, float], optional
            Maximum exposure per sector, e.g., {"TECH": 0.35, "FIN": 0.25}.
    single_name_max : float, optional
            Maximum weight for any single asset.
    turnover_penalty : float, optional
            Soft penalty coefficient for portfolio turnover.
    transaction_cost_bps : float
            Transaction cost in basis points per unit turnover.
    """

    budget_sum_to_one: bool = True
    cardinality_k: int = 25
    weight_bounds: Tuple[float, float] = (0.0, 0.1)
    no_short: bool = True
    sector_caps: Optional[Dict[str, float]] = None
    single_name_max: Optional[float] = None
    turnover_penalty: Optional[float] = None
    transaction_cost_bps: float = 5.0

    def __post_init__(self):
        """Validate constraint parameters."""
        if self.cardinality_k <= 0:
            raise ValueError(
                f"cardinality_k must be positive, got {self.cardinality_k}"
            )

        if self.weight_bounds[0] < 0 and self.no_short:
            raise ValueError("no_short=True conflicts with negative weight_bounds")

        if self.weight_bounds[0] > self.weight_bounds[1]:
            raise ValueError(
                f"Invalid weight_bounds: min={self.weight_bounds[0]} > max={self.weight_bounds[1]}"
            )

        if self.single_name_max is not None and self.single_name_max > 1.0:
            raise ValueError(
                f"single_name_max must be <= 1.0, got {self.single_name_max}"
            )

        if self.sector_caps:
            for sector, cap in self.sector_caps.items():
                if cap < 0 or cap > 1.0:
                    raise ValueError(
                        f"Sector cap for {sector} must be in [0, 1], got {cap}"
                    )


def validate_portfolio(
    weights: np.ndarray,
    constraints: PortfolioConstraints,
    tickers: List[str],
    sectors: Optional[Dict[str, str]] = None,
    epsilon: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """Validate portfolio against constraints.

    Parameters
    ----------
    weights : np.ndarray
            Portfolio weights (length = number of assets).
    constraints : PortfolioConstraints
            Constraint specification.
    tickers : List[str]
            Asset ticker symbols corresponding to weights.
    sectors : Dict[str, str], optional
            Mapping of ticker -> sector.
    epsilon : float, default=1e-6
            Tolerance for floating point comparisons.

    Returns
    -------
    Tuple[bool, List[str]]
            (is_valid, list_of_violations)
    """
    violations = []

    # Budget constraint
    if constraints.budget_sum_to_one:
        weight_sum = np.sum(weights)
        if abs(weight_sum - 1.0) > epsilon:
            violations.append(
                f"Budget violation: weights sum to {weight_sum:.6f}, expected 1.0"
            )

    # Cardinality constraint
    num_selected = np.sum(np.abs(weights) > epsilon)
    if num_selected > constraints.cardinality_k:
        violations.append(
            f"Cardinality violation: {num_selected} assets selected, max is {constraints.cardinality_k}"
        )

    # Weight bounds
    w_min, w_max = constraints.weight_bounds
    if np.any(weights < w_min - epsilon):
        violations.append(
            f"Weight below minimum: min weight = {np.min(weights):.6f}, bound = {w_min}"
        )
    if np.any(weights > w_max + epsilon):
        violations.append(
            f"Weight above maximum: max weight = {np.max(weights):.6f}, bound = {w_max}"
        )

    # No-short constraint
    if constraints.no_short and np.any(weights < -epsilon):
        violations.append(
            f"Short selling violation: min weight = {np.min(weights):.6f}"
        )

    # Single-name max
    if constraints.single_name_max is not None:
        if np.any(weights > constraints.single_name_max + epsilon):
            violations.append(
                f"Single-name max violation: max weight = {np.max(weights):.6f}, limit = {constraints.single_name_max}"
            )

    # Sector caps
    if constraints.sector_caps and sectors:
        sector_exposures = {}
        for ticker, weight in zip(tickers, weights):
            sector = sectors.get(ticker)
            if sector:
                sector_exposures[sector] = sector_exposures.get(sector, 0.0) + weight

        for sector, cap in constraints.sector_caps.items():
            exposure = sector_exposures.get(sector, 0.0)
            if exposure > cap + epsilon:
                violations.append(
                    f"Sector cap violation for {sector}: {exposure:.4f} > {cap}"
                )

    is_valid = len(violations) == 0
    return is_valid, violations


def check_constraint_feasibility(
    constraints: PortfolioConstraints, num_assets: int
) -> Tuple[bool, Optional[str]]:
    """Check if constraints are mathematically feasible.

    Parameters
    ----------
    constraints : PortfolioConstraints
            Constraint specification.
    num_assets : int
            Total number of assets in universe.

    Returns
    -------
    Tuple[bool, Optional[str]]
            (is_feasible, error_message)
    """
    w_min, w_max = constraints.weight_bounds

    # Check if cardinality K allows budget to be met
    if constraints.budget_sum_to_one:
        min_budget = constraints.cardinality_k * w_min
        max_budget = constraints.cardinality_k * w_max

        if min_budget > 1.0 + 1e-6:
            return (
                False,
                f"Infeasible: K={constraints.cardinality_k} with w_min={w_min} requires budget > 1.0",
            )

        if max_budget < 1.0 - 1e-6:
            return (
                False,
                f"Infeasible: K={constraints.cardinality_k} with w_max={w_max} cannot reach budget of 1.0",
            )

    # Check cardinality vs universe size
    if constraints.cardinality_k > num_assets:
        return (
            False,
            f"Infeasible: cardinality K={constraints.cardinality_k} > num_assets={num_assets}",
        )

    return True, None
