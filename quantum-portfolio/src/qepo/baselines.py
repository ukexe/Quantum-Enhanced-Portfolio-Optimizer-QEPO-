import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import cvxpy as cp
import mlflow
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.risk_models import sample_cov

from qepo.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)

# Cursor Task: Classical portfolio optimization baselines


class ClassicalBaselines:
    """Classical portfolio optimization methods for comparison with quantum solver.

    This class implements traditional portfolio optimization methods including
    Mean-Variance Optimization (MVO) and greedy heuristics.
    """

    def __init__(self, mlflow_logging: bool = True):
        """Initialize classical baselines.

        Parameters
        ----------
        mlflow_logging : bool, default=True
            Whether to log results to MLflow
        """
        self.mlflow_logging = mlflow_logging
        logger.info("Initialized classical baselines")

    def mvo_solve(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        risk_aversion: float = 1.0,
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """Solve portfolio optimization using Mean-Variance Optimization.

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
        tickers : List[str], optional
            Asset ticker symbols

        Returns
        -------
        Dict
            Solution results including weights, performance metrics, and timing
        """
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(len(expected_returns))]

        logger.info(
            f"Solving MVO for {len(expected_returns)} assets with risk_aversion={risk_aversion}"
        )

        start_time = time.time()

        try:
            # Create PyPortfolioOpt EfficientFrontier
            ef = EfficientFrontier(expected_returns, covariance_matrix)

            # Add constraints
            self._add_pypfopt_constraints(ef, constraints)

            # Optimize for maximum Sharpe ratio (equivalent to risk-adjusted return)
            weights = ef.max_sharpe()

            # Clean weights (remove near-zero weights)
            cleaned_weights = ef.clean_weights()

            # Convert to numpy array
            weight_array = np.array(
                [cleaned_weights.get(ticker, 0.0) for ticker in tickers]
            )

            # Calculate performance metrics
            performance = ef.portfolio_performance(verbose=False)
            expected_return, volatility, sharpe_ratio = performance

            solve_time = time.time() - start_time

            result = {
                "method": "MVO_PyPortfolioOpt",
                "weights": weight_array,
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "solve_time": solve_time,
                "status": "success",
                "num_assets": len(expected_returns),
                "risk_aversion": risk_aversion,
            }

            # Log to MLflow if enabled
            if self.mlflow_logging:
                self._log_mvo_results(result, constraints)

            logger.info(
                f"MVO solved in {solve_time:.3f}s: return={expected_return:.4f}, vol={volatility:.4f}, sharpe={sharpe_ratio:.4f}"
            )

        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"MVO optimization failed: {e}")

            result = {
                "method": "MVO_PyPortfolioOpt",
                "weights": np.zeros(len(expected_returns)),
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "solve_time": solve_time,
                "status": "failed",
                "error": str(e),
                "num_assets": len(expected_returns),
                "risk_aversion": risk_aversion,
            }

        return result

    def mvo_cvxpy_solve(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        risk_aversion: float = 1.0,
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """Solve portfolio optimization using CVXPY for more control.

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
        tickers : List[str], optional
            Asset ticker symbols

        Returns
        -------
        Dict
            Solution results
        """
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(len(expected_returns))]

        logger.info(f"Solving MVO with CVXPY for {len(expected_returns)} assets")

        start_time = time.time()

        try:
            n = len(expected_returns)

            # Define variables
            w = cp.Variable(n)

            # Objective: maximize return - risk_aversion * risk
            expected_return = expected_returns.T @ w
            risk = cp.quad_form(w, covariance_matrix)
            objective = cp.Maximize(expected_return - risk_aversion * risk)

            # Constraints
            constraint_list = []

            # Budget constraint
            if constraints.budget_sum_to_one:
                constraint_list.append(cp.sum(w) == 1.0)

            # Weight bounds
            w_min, w_max = constraints.weight_bounds
            constraint_list.append(w >= w_min)
            constraint_list.append(w <= w_max)

            # No short selling
            if constraints.no_short:
                constraint_list.append(w >= 0)

            # Single name max
            if constraints.single_name_max is not None:
                constraint_list.append(w <= constraints.single_name_max)

            # Solve problem
            problem = cp.Problem(objective, constraint_list)
            problem.solve(verbose=False)

            if problem.status == cp.OPTIMAL:
                weights = w.value

                # Calculate performance metrics
                portfolio_return = expected_returns.T @ weights
                portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
                sharpe_ratio = (
                    portfolio_return / portfolio_risk if portfolio_risk > 0 else 0.0
                )

                solve_time = time.time() - start_time

                result = {
                    "method": "MVO_CVXPY",
                    "weights": weights,
                    "expected_return": portfolio_return,
                    "volatility": portfolio_risk,
                    "sharpe_ratio": sharpe_ratio,
                    "solve_time": solve_time,
                    "status": "success",
                    "num_assets": n,
                    "risk_aversion": risk_aversion,
                }

                logger.info(
                    f"CVXPY MVO solved in {solve_time:.3f}s: return={portfolio_return:.4f}, vol={portfolio_risk:.4f}"
                )

            else:
                raise ValueError(
                    f"CVXPY optimization failed with status: {problem.status}"
                )

        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"CVXPY MVO optimization failed: {e}")

            result = {
                "method": "MVO_CVXPY",
                "weights": np.zeros(len(expected_returns)),
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "solve_time": solve_time,
                "status": "failed",
                "error": str(e),
                "num_assets": len(expected_returns),
                "risk_aversion": risk_aversion,
            }

        return result

    def greedy_k_select(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """Greedy K-select heuristic for portfolio optimization.

        Selects K assets with highest risk-adjusted returns, then optimizes weights.

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns for each asset
        covariance_matrix : np.ndarray
            Covariance matrix
        constraints : PortfolioConstraints
            Portfolio constraints
        tickers : List[str], optional
            Asset ticker symbols

        Returns
        -------
        Dict
            Solution results
        """
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(len(expected_returns))]

        logger.info(f"Running greedy K-select for K={constraints.cardinality_k}")

        start_time = time.time()

        try:
            n = len(expected_returns)
            K = constraints.cardinality_k

            # Calculate risk-adjusted returns (Sharpe ratios)
            volatilities = np.sqrt(np.diag(covariance_matrix))
            risk_adjusted_returns = expected_returns / volatilities

            # Select top K assets
            top_k_indices = np.argsort(risk_adjusted_returns)[-K:]

            # Create reduced problem
            selected_returns = expected_returns[top_k_indices]
            selected_cov = covariance_matrix[np.ix_(top_k_indices, top_k_indices)]

            # Solve MVO for selected assets
            selected_constraints = PortfolioConstraints(
                cardinality_k=K,
                weight_bounds=constraints.weight_bounds,
                budget_sum_to_one=constraints.budget_sum_to_one,
                no_short=constraints.no_short,
            )

            # Use CVXPY for the reduced problem
            w_selected = cp.Variable(K)

            # Objective
            expected_return = selected_returns.T @ w_selected
            risk = cp.quad_form(w_selected, selected_cov)
            objective = cp.Maximize(
                expected_return - 0.5 * risk
            )  # Moderate risk aversion

            # Constraints
            constraint_list = [cp.sum(w_selected) == 1.0, w_selected >= 0]

            # Weight bounds
            w_min, w_max = constraints.weight_bounds
            constraint_list.append(w_selected >= w_min)
            constraint_list.append(w_selected <= w_max)

            # Solve
            problem = cp.Problem(objective, constraint_list)
            problem.solve(verbose=False)

            if problem.status == cp.OPTIMAL:
                selected_weights = w_selected.value

                # Map back to full universe
                full_weights = np.zeros(n)
                full_weights[top_k_indices] = selected_weights

                # Calculate performance
                portfolio_return = expected_returns.T @ full_weights
                portfolio_risk = np.sqrt(
                    full_weights.T @ covariance_matrix @ full_weights
                )
                sharpe_ratio = (
                    portfolio_return / portfolio_risk if portfolio_risk > 0 else 0.0
                )

                solve_time = time.time() - start_time

                result = {
                    "method": "Greedy_K_Select",
                    "weights": full_weights,
                    "expected_return": portfolio_return,
                    "volatility": portfolio_risk,
                    "sharpe_ratio": sharpe_ratio,
                    "solve_time": solve_time,
                    "status": "success",
                    "num_assets": n,
                    "selected_assets": top_k_indices.tolist(),
                    "K": K,
                }

                logger.info(
                    f"Greedy K-select solved in {solve_time:.3f}s: return={portfolio_return:.4f}, vol={portfolio_risk:.4f}"
                )

            else:
                raise ValueError(
                    f"Greedy optimization failed with status: {problem.status}"
                )

        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"Greedy K-select failed: {e}")

            result = {
                "method": "Greedy_K_Select",
                "weights": np.zeros(len(expected_returns)),
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "solve_time": solve_time,
                "status": "failed",
                "error": str(e),
                "num_assets": len(expected_returns),
                "K": constraints.cardinality_k,
            }

        return result

    def _add_pypfopt_constraints(
        self, ef: EfficientFrontier, constraints: PortfolioConstraints
    ) -> None:
        """Add constraints to PyPortfolioOpt EfficientFrontier.

        Parameters
        ----------
        ef : EfficientFrontier
            PyPortfolioOpt EfficientFrontier object
        constraints : PortfolioConstraints
            Portfolio constraints
        """
        # Weight bounds
        w_min, w_max = constraints.weight_bounds
        ef.add_constraint(lambda w: w >= w_min)
        ef.add_constraint(lambda w: w <= w_max)

        # Single name max
        if constraints.single_name_max is not None:
            ef.add_constraint(lambda w: w <= constraints.single_name_max)

        # Note: PyPortfolioOpt doesn't directly support cardinality constraints
        # This would need to be handled separately or with custom optimization

    def _log_mvo_results(self, result: Dict, constraints: PortfolioConstraints) -> None:
        """Log MVO results to MLflow.

        Parameters
        ----------
        result : Dict
            MVO solution results
        constraints : PortfolioConstraints
            Portfolio constraints
        """
        try:
            with mlflow.start_run(nested=True) as run:
                mlflow.set_tag("method", "classical_baseline")
                mlflow.set_tag("baseline_type", result["method"])

                # Log parameters
                mlflow.log_params(
                    {
                        "num_assets": result["num_assets"],
                        "risk_aversion": result["risk_aversion"],
                        "cardinality_k": constraints.cardinality_k,
                        "weight_bounds_min": constraints.weight_bounds[0],
                        "weight_bounds_max": constraints.weight_bounds[1],
                        "budget_sum_to_one": constraints.budget_sum_to_one,
                        "no_short": constraints.no_short,
                    }
                )

                # Log metrics
                mlflow.log_metrics(
                    {
                        "expected_return": result["expected_return"],
                        "volatility": result["volatility"],
                        "sharpe_ratio": result["sharpe_ratio"],
                        "solve_time": result["solve_time"],
                    }
                )

                # Log weights as artifact
                weights_df = pd.DataFrame(
                    {
                        "asset": [f"Asset_{i}" for i in range(len(result["weights"]))],
                        "weight": result["weights"],
                    }
                )
                mlflow.log_table(weights_df, "weights.json")

        except Exception as e:
            logger.warning(f"Failed to log MVO results to MLflow: {e}")


def mvo_solve(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    constraints: PortfolioConstraints,
    risk_aversion: float = 1.0,
    tickers: Optional[List[str]] = None,
    method: str = "pypfopt",
) -> Dict:
    """Convenience function to solve MVO.

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns
    covariance_matrix : np.ndarray
        Covariance matrix
    constraints : PortfolioConstraints
        Portfolio constraints
    risk_aversion : float, default=1.0
        Risk aversion parameter
    tickers : List[str], optional
        Asset ticker symbols
    method : str, default="pypfopt"
        Method to use: "pypfopt" or "cvxpy"

    Returns
    -------
    Dict
        Solution results
    """
    baselines = ClassicalBaselines()

    if method == "pypfopt":
        return baselines.mvo_solve(
            expected_returns, covariance_matrix, constraints, risk_aversion, tickers
        )
    elif method == "cvxpy":
        return baselines.mvo_cvxpy_solve(
            expected_returns, covariance_matrix, constraints, risk_aversion, tickers
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pypfopt' or 'cvxpy'")


def greedy_k_select(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    constraints: PortfolioConstraints,
    tickers: Optional[List[str]] = None,
) -> Dict:
    """Convenience function for greedy K-select.

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns
    covariance_matrix : np.ndarray
        Covariance matrix
    constraints : PortfolioConstraints
        Portfolio constraints
    tickers : List[str], optional
        Asset ticker symbols

    Returns
    -------
    Dict
        Solution results
    """
    baselines = ClassicalBaselines()
    return baselines.greedy_k_select(
        expected_returns, covariance_matrix, constraints, tickers
    )
