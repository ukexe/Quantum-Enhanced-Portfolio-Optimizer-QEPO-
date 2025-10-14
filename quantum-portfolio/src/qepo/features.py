import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

# Cursor Task: Feature and risk model computations


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with columns: date, ticker, adj_close

    Returns
    -------
    pd.DataFrame
        Returns data with columns: date, ticker, ret_d (daily log returns)

    Raises
    ------
    ValueError
        If prices DataFrame is empty or missing required columns.
    """
    if prices.empty:
        raise ValueError("Prices DataFrame is empty")

    required_cols = ["date", "ticker", "adj_close"]
    if not all(col in prices.columns for col in required_cols):
        raise ValueError(f"Prices DataFrame missing required columns: {required_cols}")

    logger.info(f"Computing log returns for {prices['ticker'].nunique()} tickers")

    # Sort by ticker and date for proper calculation
    prices_sorted = prices.sort_values(["ticker", "date"]).copy()

    # Compute log returns: log(P_t / P_{t-1})
    def compute_log_returns(group):
        return np.log(group / group.shift(1))

    prices_sorted["ret_d"] = prices_sorted.groupby("ticker")["adj_close"].transform(
        compute_log_returns
    )

    # Remove first observation for each ticker (NaN from shift)
    returns_df = prices_sorted.dropna().reset_index(drop=True)

    # Keep only required columns
    returns_df = returns_df[["date", "ticker", "ret_d"]]

    logger.info(f"Computed {len(returns_df)} return observations")
    return returns_df


def compute_covariance(
    returns: pd.DataFrame,
    window: int = 252,
    shrinkage: bool = True,
    min_periods: Optional[int] = None,
) -> Tuple[list, pd.DataFrame]:
    """Compute rolling covariance matrices with optional Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns data with columns: date, ticker, ret_d
    window : int, default=252
        Rolling window size in days (252 = 1 trading year)
    shrinkage : bool, default=True
        If True, apply Ledoit-Wolf shrinkage to covariance matrix
    min_periods : int, optional
        Minimum number of observations in window. Defaults to window//2

    Returns
    -------
    Tuple[list, pd.DataFrame]
        (covariance_matrices, mean_returns)
        - covariance_matrices: List of DataFrames (one per date)
        - mean_returns: DataFrame with date index and ticker columns

    Raises
    ------
    ValueError
        If returns DataFrame is empty or insufficient data for window.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")

    if min_periods is None:
        min_periods = window // 2

    logger.info(
        f"Computing rolling covariance with window={window}, shrinkage={shrinkage}"
    )

    # Pivot to wide format: date x ticker
    returns_wide = returns.pivot(index="date", columns="ticker", values="ret_d")

    # Check if we have enough data
    if len(returns_wide) < min_periods:
        raise ValueError(
            f"Insufficient data: {len(returns_wide)} observations < {min_periods} minimum"
        )

    # Compute rolling statistics
    cov_matrices = []
    mean_returns = []

    for i in range(window - 1, len(returns_wide)):
        # Get window data
        window_data = returns_wide.iloc[i - window + 1 : i + 1]

        # Remove columns with insufficient data
        valid_cols = window_data.columns[window_data.count() >= min_periods]
        if len(valid_cols) == 0:
            logger.warning(f"No valid assets at date {returns_wide.index[i]}")
            continue

        window_clean = window_data[valid_cols].dropna()

        if len(window_clean) < min_periods:
            logger.warning(f"Insufficient clean data at date {returns_wide.index[i]}")
            continue

        # Compute mean returns
        mean_ret = window_clean.mean()

        # Compute covariance matrix
        if shrinkage and len(window_clean) > 10:  # Need sufficient data for shrinkage
            try:
                lw = LedoitWolf()
                cov_matrix = lw.fit(window_clean.values).covariance_
                cov_df = pd.DataFrame(cov_matrix, index=valid_cols, columns=valid_cols)
            except Exception as e:
                logger.warning(
                    f"Ledoit-Wolf shrinkage failed at {returns_wide.index[i]}: {e}"
                )
                # Fallback to sample covariance
                cov_df = window_clean.cov()
        else:
            # Sample covariance
            cov_df = window_clean.cov()

        # Store results
        cov_matrices.append(cov_df)
        mean_returns.append(mean_ret)

    if not cov_matrices:
        raise ValueError("No valid covariance matrices computed")

    # Combine results
    # For covariance matrices, we'll store them as a list since they're 2D
    # For mean returns, we can create a DataFrame
    mean_returns_df = pd.DataFrame(mean_returns)

    logger.info(f"Computed {len(cov_matrices)} covariance matrices")
    return cov_matrices, mean_returns_df


def validate_covariance_stability(
    cov_matrices: list, condition_threshold: float = 1e12
) -> dict:
    """Validate numerical stability of covariance matrices.

    Parameters
    ----------
    cov_matrices : list
        List of covariance matrices (DataFrames)
    condition_threshold : float, default=1e12
        Maximum condition number for stability

    Returns
    -------
    dict
        Validation results with statistics
    """
    if not cov_matrices:
        return {"error": "No covariance matrices provided"}

    results = {
        "num_matrices": len(cov_matrices),
        "condition_numbers": [],
        "eigenvalues": [],
        "is_positive_definite": [],
        "is_stable": True,
    }

    for i, cov_matrix in enumerate(cov_matrices):
        try:
            # Convert to numpy array
            cov_array = cov_matrix.values

            # Check if positive definite
            eigenvals = np.linalg.eigvals(cov_array)
            is_pd = np.all(eigenvals > 0)

            # Compute condition number
            condition_num = np.linalg.cond(cov_array)

            results["condition_numbers"].append(condition_num)
            results["eigenvalues"].append(eigenvals)
            results["is_positive_definite"].append(is_pd)

            # Check stability
            if condition_num > condition_threshold or not is_pd:
                results["is_stable"] = False
                logger.warning(
                    f"Unstable covariance matrix {i}: condition={condition_num:.2e}, pd={is_pd}"
                )

        except Exception as e:
            logger.error(f"Error validating covariance matrix {i}: {e}")
            results["is_stable"] = False

    # Summary statistics
    if results["condition_numbers"]:
        results["max_condition"] = max(results["condition_numbers"])
        results["min_condition"] = min(results["condition_numbers"])
        results["avg_condition"] = np.mean(results["condition_numbers"])

    return results
