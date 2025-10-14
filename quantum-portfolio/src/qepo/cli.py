from pathlib import Path
from typing import Optional

import mlflow
import typer
import yaml

from qepo import data as data_module
from qepo.utils import configure_logging

app = typer.Typer(help="QEPO CLI - data, optimize, backtest, report")


@app.command("data")
def data_command(
    config: Optional[Path] = typer.Option(
        Path("config/data.yml"), help="Path to data config"
    )
) -> None:
    """Run data ingestion pipeline."""
    configure_logging()

    # Load config
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_experiment("qepo")
    with mlflow.start_run() as run:
        mlflow.set_tag("command", "data")
        mlflow.log_param("config_path", str(config))
        mlflow.log_params(cfg)

        try:
            # Fetch universe
            typer.echo("Fetching S&P 500 universe from Wikipedia...")
            output_dir = Path(cfg.get("cache", {}).get("path", "data/interim"))
            tickers = data_module.fetch_universe(
                save_metadata=True, output_dir=output_dir
            )
            mlflow.log_metric("num_tickers", len(tickers))
            typer.echo(f"✓ Fetched {len(tickers)} tickers")

            # Download prices
            typer.echo(
                f"Downloading historical prices from {cfg['history']['start']} to {cfg['history']['end']}..."
            )
            prices_df = data_module.download_prices(
                tickers,
                cfg["history"]["start"],
                cfg["history"]["end"],
                output_dir=output_dir,
            )
            mlflow.log_metric("num_price_records", len(prices_df))
            typer.echo(f"✓ Downloaded {len(prices_df)} price records")

            typer.echo(f"\n✅ Data ingestion complete! Run ID: {run.info.run_id}")

        except Exception as e:
            typer.echo(f"❌ Error during data ingestion: {e}", err=True)
            mlflow.set_tag("status", "failed")
            raise typer.Exit(1)


@app.command("optimize")
def optimize_command(
    config: Optional[Path] = typer.Option(
        Path("config/optimizer.yml"), help="Path to optimizer config"
    )
) -> None:
    """Run quantum portfolio optimization."""
    configure_logging()

    # Load config
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_experiment("qepo")
    with mlflow.start_run() as run:
        mlflow.set_tag("command", "optimize")
        mlflow.log_param("config_path", str(config))
        mlflow.log_params(cfg)

        try:
            # Import required modules
            import numpy as np
            import pandas as pd

            from qepo import constraints
            from qepo import data as data_module
            from qepo import encoder, features, hardware, postprocess, quantum_qaoa

            # Load data
            typer.echo("Loading historical data...")
            data_dir = Path(cfg.get("data_dir", "data/interim"))

            prices_path = data_dir / "prices.parquet"
            returns_path = data_dir / "returns.parquet"
            covariance_path = data_dir / "covariance.parquet"

            if not all(
                p.exists() for p in [prices_path, returns_path, covariance_path]
            ):
                typer.echo(
                    "Error: Required data files not found. Run 'data' command first.",
                    err=True,
                )
                raise typer.Exit(1)

            prices_df = pd.read_parquet(prices_path)
            returns_df = pd.read_parquet(returns_path)
            covariance_df = pd.read_parquet(covariance_path)

            typer.echo(
                f"✓ Loaded data: {len(prices_df)} price records, {len(returns_df)} return records"
            )

            # Create constraints
            constraints_config = cfg.get("constraints", {})
            portfolio_constraints = constraints.PortfolioConstraints(
                budget_sum_to_one=constraints_config.get("budget_sum_to_one", True),
                cardinality_k=constraints_config.get("cardinality_k", 25),
                weight_bounds=tuple(
                    constraints_config.get("weight_bounds", [0.0, 0.1])
                ),
                no_short=constraints_config.get("no_short", True),
                sector_caps=constraints_config.get("sector_caps"),
                single_name_max=constraints_config.get("single_name_max"),
                transaction_cost_bps=constraints_config.get(
                    "transaction_cost_bps", 5.0
                ),
            )

            # Get latest returns and covariance for optimization
            latest_returns = returns_df.iloc[-1].values  # Most recent returns
            latest_covariance = covariance_df.values

            typer.echo(f"✓ Using latest returns and covariance for optimization")
            typer.echo(f"  Expected return: {np.mean(latest_returns):.4f}")
            typer.echo(
                f"  Portfolio volatility: {np.sqrt(np.mean(np.diag(latest_covariance))):.4f}"
            )

            # Build QUBO
            typer.echo("Building QUBO problem...")
            qubo_config = cfg.get("qubo", {})
            qubo_matrix = encoder.build_qubo(
                expected_returns=latest_returns,
                covariance_matrix=latest_covariance,
                constraints=portfolio_constraints,
                risk_aversion=qubo_config.get("risk_aversion", 1.0),
                penalty_scale=qubo_config.get("penalty_scale", 1.0),
            )

            typer.echo(f"✓ Built QUBO matrix with {len(qubo_matrix)} terms")

            # Initialize QAOA solver
            typer.echo("Initializing QAOA solver...")
            solver_config = cfg.get("solver", {})
            qaoa_solver = quantum_qaoa.QAOASolver(
                depth=solver_config.get("depth", 2),
                shots=solver_config.get("shots", 1000),
                max_iterations=solver_config.get("max_iterations", 100),
            )

            # Check if hardware is enabled
            hardware_config = cfg.get("hardware", {})
            use_hardware = hardware_config.get("enabled", False)

            if use_hardware:
                typer.echo(
                    "⚠️ Hardware mode not yet implemented in CLI, using simulator"
                )
                use_hardware = False

            if not use_hardware:
                typer.echo("Using Qiskit Aer simulator...")

            # Solve QUBO
            typer.echo("Solving QUBO with QAOA...")
            solution = qaoa_solver.solve_qubo(
                qubo_matrix, num_qubits=len(latest_returns)
            )

            typer.echo(f"✓ QAOA solution found with energy: {solution['energy']:.4f}")

            # Post-process solution
            typer.echo("Post-processing solution...")
            portfolio_weights = postprocess.post_process_solution(
                solution["solution"], portfolio_constraints, latest_returns
            )

            # Validate portfolio
            is_valid, violations = constraints.validate_portfolio(
                portfolio_weights, portfolio_constraints
            )
            if not is_valid:
                typer.echo(f"⚠️ Portfolio validation failed: {violations}")
            else:
                typer.echo("✓ Portfolio satisfies all constraints")

            # Calculate portfolio metrics
            portfolio_return = np.dot(portfolio_weights, latest_returns)
            portfolio_volatility = np.sqrt(
                np.dot(portfolio_weights, np.dot(latest_covariance, portfolio_weights))
            )
            sharpe_ratio = (
                portfolio_return / portfolio_volatility
                if portfolio_volatility > 0
                else 0
            )

            # Log metrics to MLflow
            mlflow.log_metric("portfolio_return", portfolio_return)
            mlflow.log_metric("portfolio_volatility", portfolio_volatility)
            mlflow.log_metric("sharpe_ratio", sharpe_ratio)
            mlflow.log_metric("solution_energy", solution["energy"])
            mlflow.log_metric("num_selected_assets", np.sum(portfolio_weights > 0))
            mlflow.log_metric("max_weight", np.max(portfolio_weights))
            mlflow.log_metric(
                "min_weight", np.min(portfolio_weights[portfolio_weights > 0])
            )

            # Save portfolio weights
            output_dir = Path(cfg.get("output_dir", "data/optimization"))
            output_dir.mkdir(parents=True, exist_ok=True)

            portfolio_df = pd.DataFrame(
                {"ticker": returns_df.columns, "weight": portfolio_weights}
            ).sort_values("weight", ascending=False)

            portfolio_path = output_dir / f"portfolio_{run.info.run_id}.csv"
            portfolio_df.to_csv(portfolio_path, index=False)
            mlflow.log_artifact(str(portfolio_path))

            # Display results
            typer.echo("\n📊 Optimization Results:")
            typer.echo(f"Portfolio Return: {portfolio_return:.4f}")
            typer.echo(f"Portfolio Volatility: {portfolio_volatility:.4f}")
            typer.echo(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            typer.echo(f"Selected Assets: {np.sum(portfolio_weights > 0)}")
            typer.echo(f"Max Weight: {np.max(portfolio_weights):.4f}")

            typer.echo(f"\n🏆 Top 10 Holdings:")
            top_holdings = portfolio_df.head(10)
            for _, row in top_holdings.iterrows():
                if row["weight"] > 0:
                    typer.echo(f"  {row['ticker']}: {row['weight']:.4f}")

            typer.echo(
                f"\n✅ Optimization complete! Portfolio saved to {portfolio_path}"
            )
            typer.echo(f"Run ID: {run.info.run_id}")

        except Exception as e:
            typer.echo(f"❌ Error during optimization: {e}", err=True)
            mlflow.set_tag("status", "failed")
            raise typer.Exit(1)


@app.command("backtest")
def backtest_command(
    config: Optional[Path] = typer.Option(
        Path("config/backtest.yml"), help="Path to backtest config"
    )
) -> None:
    """Run backtesting engine."""
    configure_logging()

    # Load config
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_experiment("qepo")
    with mlflow.start_run() as run:
        mlflow.set_tag("command", "backtest")
        mlflow.log_param("config_path", str(config))
        mlflow.log_params(cfg)

        try:
            # Import backtest module
            import pandas as pd

            from qepo import backtest, baselines, constraints
            from qepo import data as data_module
            from qepo import features

            # Load data
            typer.echo("Loading historical data...")
            data_dir = Path(cfg.get("data_dir", "data/interim"))

            prices_path = data_dir / "prices.parquet"
            returns_path = data_dir / "returns.parquet"
            covariance_path = data_dir / "covariance.parquet"

            if not all(
                p.exists() for p in [prices_path, returns_path, covariance_path]
            ):
                typer.echo(
                    "Error: Required data files not found. Run 'data' command first.",
                    err=True,
                )
                raise typer.Exit(1)

            prices_df = pd.read_parquet(prices_path)
            returns_df = pd.read_parquet(returns_path)
            covariance_df = pd.read_parquet(covariance_path)

            typer.echo(
                f"✓ Loaded data: {len(prices_df)} price records, {len(returns_df)} return records"
            )

            # Create constraints
            constraints_config = cfg.get("constraints", {})
            portfolio_constraints = constraints.PortfolioConstraints(
                budget_sum_to_one=constraints_config.get("budget_sum_to_one", True),
                cardinality_k=constraints_config.get("cardinality_k", 25),
                weight_bounds=tuple(
                    constraints_config.get("weight_bounds", [0.0, 0.1])
                ),
                no_short=constraints_config.get("no_short", True),
                sector_caps=constraints_config.get("sector_caps"),
                single_name_max=constraints_config.get("single_name_max"),
                transaction_cost_bps=constraints_config.get(
                    "transaction_cost_bps", 5.0
                ),
            )

            # Define strategy function
            strategy_name = cfg.get("strategy", "mvo")
            if strategy_name == "mvo":

                def strategy_fn(returns, cov, constraints):
                    return baselines.mvo_solve(returns, cov, constraints)[0]

            elif strategy_name == "greedy":

                def strategy_fn(returns, cov, constraints):
                    return baselines.greedy_k_select(returns, cov, constraints)[0]

            else:
                typer.echo(f"Error: Unknown strategy '{strategy_name}'", err=True)
                raise typer.Exit(1)

            # Run backtest
            typer.echo(f"Running backtest with {strategy_name} strategy...")
            output_dir = Path(cfg.get("output_dir", "data/backtest"))

            result = backtest.walk_forward(
                config=cfg,
                strategy_fn=strategy_fn,
                prices=prices_df,
                returns=returns_df,
                covariance=covariance_df,
                constraints=portfolio_constraints,
                benchmark_ticker=cfg.get("benchmark", "SPY"),
                output_dir=output_dir,
            )

            # Display results
            typer.echo("\n📊 Backtest Results:")
            typer.echo(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
            typer.echo(
                f"Annualized Return: {result.metrics.get('annualized_return', 0):.2%}"
            )
            typer.echo(f"Volatility: {result.metrics.get('volatility', 0):.2%}")
            typer.echo(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.3f}")
            typer.echo(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
            typer.echo(f"Avg Turnover: {result.metrics.get('avg_turnover', 0):.2%}")

            typer.echo(f"\n✅ Backtest complete! Results saved to {output_dir}")
            typer.echo(f"Run ID: {run.info.run_id}")

        except Exception as e:
            typer.echo(f"❌ Error during backtest: {e}", err=True)
            mlflow.set_tag("status", "failed")
            raise typer.Exit(1)


@app.command("report")
def report_command(
    run_id: str = typer.Option(..., help="MLflow run id"),
    output_dir: Optional[Path] = typer.Option(
        Path("reports"), help="Output directory for report"
    ),
    format: str = typer.Option("markdown", help="Report format: markdown or html"),
    include_charts: bool = typer.Option(True, help="Include charts in report"),
) -> None:
    """Generate comprehensive report from MLflow run."""
    configure_logging()

    try:
        # Import report module
        from qepo import report

        typer.echo(f"Generating {format} report for run {run_id}...")

        # Generate report
        report_path = report.generate_report(
            run_id=run_id,
            output_dir=output_dir,
            include_charts=include_charts,
            format=format,
        )

        typer.echo(f"✅ Report generated successfully!")
        typer.echo(f"📄 Report file: {report_path}")

        if include_charts:
            typer.echo(f"📊 Charts saved to: {output_dir}")

    except Exception as e:
        typer.echo(f"❌ Error generating report: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
