import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qepo.constraints import PortfolioConstraints, validate_portfolio

logger = logging.getLogger(__name__)

# Cursor Task: Post-process and repair quantum solutions


def decode_bitstring(
    bitstring: Union[str, np.ndarray, List[int]],
    num_assets: int,
    constraints: PortfolioConstraints,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Decode a bitstring solution into portfolio weights.

    Parameters
    ----------
    bitstring : Union[str, np.ndarray, List[int]]
        Binary solution from quantum solver. Can be:
        - String of '0's and '1's (e.g., "1010")
        - Numpy array of 0s and 1s
        - List of 0s and 1s
    num_assets : int
        Total number of assets in the universe.
    constraints : PortfolioConstraints
        Portfolio constraints object.
    epsilon : float, default=1e-6
        Tolerance for floating point comparisons.

    Returns
    -------
    np.ndarray
        Portfolio weights array of length num_assets.

    Raises
    ------
    ValueError
        If bitstring length doesn't match num_assets or contains invalid values.
    """
    logger.info(f"Decoding bitstring for {num_assets} assets")

    # Convert bitstring to numpy array
    if isinstance(bitstring, str):
        if len(bitstring) != num_assets:
            raise ValueError(
                f"Bitstring length ({len(bitstring)}) doesn't match num_assets ({num_assets})"
            )
        bit_array = np.array([int(b) for b in bitstring])
    elif isinstance(bitstring, (list, np.ndarray)):
        bit_array = np.array(bitstring)
        if len(bit_array) != num_assets:
            raise ValueError(
                f"Bitstring length ({len(bit_array)}) doesn't match num_assets ({num_assets})"
            )
    else:
        raise ValueError(f"Unsupported bitstring type: {type(bitstring)}")

    # Validate bitstring values
    if not np.all(np.isin(bit_array, [0, 1])):
        raise ValueError("Bitstring must contain only 0s and 1s")

    # Count selected assets
    selected_indices = np.where(bit_array > epsilon)[0]
    num_selected = len(selected_indices)

    if num_selected == 0:
        logger.warning("No assets selected in bitstring. Returning zero weights.")
        return np.zeros(num_assets)

    # Apply cardinality constraint if violated
    if num_selected > constraints.cardinality_k:
        logger.warning(
            f"Bitstring selected {num_selected} assets, exceeding cardinality K={constraints.cardinality_k}. "
            "Selecting first K assets."
        )
        selected_indices = selected_indices[: constraints.cardinality_k]
        num_selected = len(selected_indices)

    # Initialize weights
    weights = np.zeros(num_assets)

    if num_selected > 0:
        # Equal weight distribution among selected assets
        equal_weight = 1.0 / num_selected
        weights[selected_indices] = equal_weight

        # Apply individual weight bounds
        w_min, w_max = constraints.weight_bounds

        # Cap weights at w_max
        if np.any(weights[selected_indices] > w_max + epsilon):
            logger.warning(
                f"Some weights exceed w_max={w_max}. Capping and re-normalizing."
            )
            weights[selected_indices] = np.minimum(weights[selected_indices], w_max)
            # Re-normalize if capping occurred
            current_sum = np.sum(weights)
            if current_sum > epsilon:
                weights = weights / current_sum

        # Ensure weights are not below w_min (after potential re-normalization)
        if np.any(weights[selected_indices] < w_min - epsilon):
            logger.warning(
                f"Some weights are below w_min={w_min}. This might indicate an infeasible solution."
            )

    logger.info(f"Decoded weights: {weights}")
    return weights


def check_feasibility(
    weights: np.ndarray,
    constraints: PortfolioConstraints,
    tickers: List[str],
    sectors: Optional[Dict[str, str]] = None,
    epsilon: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """
    Check if portfolio weights satisfy all constraints.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights array.
    constraints : PortfolioConstraints
        Portfolio constraints object.
    tickers : List[str]
        Asset ticker symbols corresponding to weights.
    sectors : Dict[str, str], optional
        Mapping of ticker -> sector.
    epsilon : float, default=1e-6
        Tolerance for floating point comparisons.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_feasible, list_of_violations)
    """
    logger.info("Checking portfolio feasibility")

    is_feasible, violations = validate_portfolio(
        weights, constraints, tickers, sectors, epsilon
    )

    if is_feasible:
        logger.info("Portfolio is feasible")
    else:
        logger.warning(f"Portfolio has {len(violations)} violations: {violations}")

    return is_feasible, violations


def repair_portfolio(
    weights: np.ndarray,
    constraints: PortfolioConstraints,
    tickers: List[str],
    sectors: Optional[Dict[str, str]] = None,
    max_iterations: int = 100,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, bool, List[str]]:
    """
    Repair a portfolio to satisfy all constraints.

    Uses iterative repair algorithm to fix constraint violations:
    1. Fix budget constraint (normalize weights)
    2. Fix cardinality constraint (select top K assets)
    3. Fix weight bounds (clip and re-normalize)
    4. Fix sector caps (reduce overweight sectors)
    5. Fix single-name max (cap individual weights)

    Parameters
    ----------
    weights : np.ndarray
        Initial portfolio weights array.
    constraints : PortfolioConstraints
        Portfolio constraints object.
    tickers : List[str]
        Asset ticker symbols corresponding to weights.
    sectors : Dict[str, str], optional
        Mapping of ticker -> sector.
    max_iterations : int, default=100
        Maximum number of repair iterations.
    epsilon : float, default=1e-6
        Tolerance for floating point comparisons.

    Returns
    -------
    Tuple[np.ndarray, bool, List[str]]
        (repaired_weights, is_feasible, repair_history)
    """
    logger.info("Starting portfolio repair process")

    repair_history = []
    current_weights = weights.copy()

    for iteration in range(max_iterations):
        logger.debug(f"Repair iteration {iteration + 1}")

        # Check current feasibility
        is_feasible, violations = check_feasibility(
            current_weights, constraints, tickers, sectors, epsilon
        )

        if is_feasible:
            logger.info(
                f"Portfolio repaired successfully in {iteration + 1} iterations"
            )
            return current_weights, True, repair_history

        repair_history.append(f"Iteration {iteration + 1}: {violations}")

        # Apply repair steps
        previous_weights = current_weights.copy()

        # 1. Fix budget constraint
        if constraints.budget_sum_to_one:
            weight_sum = np.sum(current_weights)
            if abs(weight_sum - 1.0) > epsilon:
                if weight_sum > epsilon:
                    current_weights = current_weights / weight_sum
                    logger.debug("Fixed budget constraint by normalization")

        # 2. Fix cardinality constraint
        num_selected = np.sum(np.abs(current_weights) > epsilon)
        if num_selected > constraints.cardinality_k:
            # Select top K assets by weight
            top_k_indices = np.argsort(current_weights)[::-1][
                : constraints.cardinality_k
            ]
            cardinality_weights = np.zeros_like(current_weights)
            cardinality_weights[top_k_indices] = current_weights[top_k_indices]
            current_weights = cardinality_weights
            logger.debug(
                f"Fixed cardinality constraint by selecting top {constraints.cardinality_k} assets"
            )

        # 3. Fix weight bounds
        w_min, w_max = constraints.weight_bounds
        if np.any(current_weights < w_min - epsilon) or np.any(
            current_weights > w_max + epsilon
        ):
            # Clip weights but preserve non-zero weights
            non_zero_mask = current_weights > epsilon
            current_weights = np.clip(current_weights, w_min, w_max)
            # If clipping made weights too small, set them to zero
            current_weights[current_weights < w_min + epsilon] = 0.0
            logger.debug("Fixed weight bounds by clipping")

        # 4. Fix no-short constraint
        if constraints.no_short and np.any(current_weights < -epsilon):
            current_weights = np.maximum(current_weights, 0.0)
            logger.debug(
                "Fixed no-short constraint by setting negative weights to zero"
            )

        # 5. Fix single-name max
        if constraints.single_name_max is not None:
            if np.any(current_weights > constraints.single_name_max + epsilon):
                current_weights = np.minimum(
                    current_weights, constraints.single_name_max
                )
                logger.debug("Fixed single-name max constraint by capping weights")

        # 6. Fix sector caps
        if constraints.sector_caps and sectors:
            sector_exposures = {}
            for ticker, weight in zip(tickers, current_weights):
                sector = sectors.get(ticker)
                if sector:
                    sector_exposures[sector] = (
                        sector_exposures.get(sector, 0.0) + weight
                    )

            for sector, cap in constraints.sector_caps.items():
                exposure = sector_exposures.get(sector, 0.0)
                if exposure > cap + epsilon:
                    # Reduce weights in this sector proportionally
                    sector_indices = [
                        i
                        for i, ticker in enumerate(tickers)
                        if sectors.get(ticker) == sector
                    ]
                    if sector_indices:
                        reduction_factor = cap / exposure
                        current_weights[sector_indices] *= reduction_factor
                        logger.debug(
                            f"Fixed sector cap for {sector} by reducing exposure by factor {reduction_factor:.3f}"
                        )

        # Re-normalize after all repairs
        if constraints.budget_sum_to_one:
            weight_sum = np.sum(current_weights)
            if weight_sum > epsilon:
                current_weights = current_weights / weight_sum

        # Check if weights changed significantly
        if np.allclose(current_weights, previous_weights, atol=epsilon):
            logger.warning("Repair algorithm converged without fixing all violations")
            break

    # Final feasibility check
    is_feasible, final_violations = check_feasibility(
        current_weights, constraints, tickers, sectors, epsilon
    )

    if not is_feasible:
        logger.warning(
            f"Portfolio repair failed. Remaining violations: {final_violations}"
        )
        repair_history.append(f"Final: {final_violations}")

    return current_weights, is_feasible, repair_history


def post_process_solution(
    bitstring: Union[str, np.ndarray, List[int]],
    constraints: PortfolioConstraints,
    tickers: List[str],
    sectors: Optional[Dict[str, str]] = None,
    repair: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, bool, List[str]]:
    """
    Complete post-processing pipeline for quantum solution.

    Combines decoding, feasibility checking, and repair into a single function.

    Parameters
    ----------
    bitstring : Union[str, np.ndarray, List[int]]
        Binary solution from quantum solver.
    constraints : PortfolioConstraints
        Portfolio constraints object.
    tickers : List[str]
        Asset ticker symbols.
    sectors : Dict[str, str], optional
        Mapping of ticker -> sector.
    repair : bool, default=True
        Whether to attempt repair if solution is infeasible.
    epsilon : float, default=1e-6
        Tolerance for floating point comparisons.

    Returns
    -------
    Tuple[np.ndarray, bool, List[str]]
        (final_weights, is_feasible, processing_history)
    """
    logger.info("Starting post-processing pipeline")

    processing_history = []
    num_assets = len(tickers)

    # Step 1: Decode bitstring
    try:
        weights = decode_bitstring(bitstring, num_assets, constraints, epsilon)
        processing_history.append("Successfully decoded bitstring")
    except Exception as e:
        logger.error(f"Bitstring decoding failed: {e}")
        processing_history.append(f"Decoding failed: {e}")
        return np.zeros(num_assets), False, processing_history

    # Step 2: Check feasibility
    is_feasible, violations = check_feasibility(
        weights, constraints, tickers, sectors, epsilon
    )
    if is_feasible:
        processing_history.append("Solution is feasible")
        return weights, True, processing_history

    processing_history.append(f"Initial violations: {violations}")

    # Step 3: Repair if requested
    if repair:
        logger.info("Attempting to repair infeasible solution")
        repaired_weights, repair_success, repair_history = repair_portfolio(
            weights, constraints, tickers, sectors, epsilon=epsilon
        )
        processing_history.extend(repair_history)

        if repair_success:
            processing_history.append("Repair successful")
            return repaired_weights, True, processing_history
        else:
            processing_history.append("Repair failed")
            return repaired_weights, False, processing_history
    else:
        processing_history.append("Repair not requested")
        return weights, False, processing_history


def round_weights(
    weights: np.ndarray,
    precision: int = 4,
    min_weight_threshold: float = 1e-6,
) -> np.ndarray:
    """
    Round portfolio weights to specified precision and remove very small weights.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights array.
    precision : int, default=4
        Number of decimal places to round to.
    min_weight_threshold : float, default=1e-6
        Weights below this threshold are set to zero.

    Returns
    -------
    np.ndarray
        Rounded weights array.
    """
    logger.info(f"Rounding weights to {precision} decimal places")

    # Remove very small weights
    rounded_weights = weights.copy()
    rounded_weights[rounded_weights < min_weight_threshold] = 0.0

    # Round to specified precision
    rounded_weights = np.round(rounded_weights, precision)

    # Re-normalize if any weights were removed
    weight_sum = np.sum(rounded_weights)
    if weight_sum > 0:
        rounded_weights = rounded_weights / weight_sum
        # Round again after normalization
        rounded_weights = np.round(rounded_weights, precision)

    logger.info(f"Rounded weights: {rounded_weights}")
    return rounded_weights
