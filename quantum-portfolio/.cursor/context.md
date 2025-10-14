### Architecture & Data Flow

- Ingest S&P 500 universe and prices into `data/interim` parquet files.
- Build features (returns, covariance) and encode constraints to QUBO.
- Solve via QAOA (simulator-first), compare with classical baselines.
- Backtest, compute metrics, and log everything to MLflow with artifacts.

Modules in `src/qepo/` map 1:1 to these steps.
