#!/usr/bin/env python3
"""Create mock data for testing the optimize command."""

import pandas as pd
import numpy as np
from pathlib import Path

# Create mock data
np.random.seed(42)

# Mock tickers (reduced size for QAOA testing)
tickers = [f"STOCK_{i:03d}" for i in range(10)]

# Mock dates
dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')

# Mock prices (random walk)
prices_data = {}
for ticker in tickers:
    # Start with random price between 10-100
    initial_price = np.random.uniform(10, 100)
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    prices_data[ticker] = prices

prices_df = pd.DataFrame(prices_data, index=dates)

# Calculate returns
returns_df = prices_df.pct_change().dropna()

# Calculate covariance matrix
covariance_df = returns_df.cov()

# Save to parquet files
data_dir = Path("data/interim")
data_dir.mkdir(parents=True, exist_ok=True)

prices_df.to_parquet(data_dir / "prices.parquet")
returns_df.to_parquet(data_dir / "returns.parquet")
covariance_df.to_parquet(data_dir / "covariance.parquet")

print(f"✓ Created mock data:")
print(f"  - {len(prices_df)} price records for {len(tickers)} tickers")
print(f"  - {len(returns_df)} return records")
print(f"  - {covariance_df.shape[0]}x{covariance_df.shape[1]} covariance matrix")
print(f"  - Saved to {data_dir}")
