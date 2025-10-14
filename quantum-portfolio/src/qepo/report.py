import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

# Cursor Task: Comprehensive reporting with metrics, charts, and allocations


def generate_report(
    run_id: str,
    output_dir: Optional[Path] = None,
    include_charts: bool = True,
    format: str = "markdown",
) -> Path:
    """
    Generate comprehensive report from MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID to generate report for.
    output_dir : Path, optional
        Directory to save report. Defaults to 'reports/'.
    include_charts : bool, default=True
        Whether to include charts in the report.
    format : str, default="markdown"
        Report format: 'markdown' or 'html'.

    Returns
    -------
    Path
        Path to the generated report file.
    """
    logger.info(f"Generating report for run {run_id}")

    if output_dir is None:
        output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get MLflow run data
    run_data = _get_mlflow_run_data(run_id)
    if not run_data:
        raise ValueError(f"Run {run_id} not found or has no data")

    # Load backtest results if available
    backtest_data = _load_backtest_data(run_data)

    # Generate report content
    if format == "markdown":
        report_path = _generate_markdown_report(
            run_data, backtest_data, output_dir, include_charts
        )
    elif format == "html":
        report_path = _generate_html_report(
            run_data, backtest_data, output_dir, include_charts
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Report generated: {report_path}")
    return report_path


def _get_mlflow_run_data(run_id: str) -> Optional[Dict]:
    """Get MLflow run data including parameters, metrics, and artifacts."""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        # Get parameters
        params = run.data.params

        # Get metrics
        metrics = run.data.metrics

        # Get tags
        tags = run.data.tags

        # Get artifacts
        artifacts = []
        try:
            artifacts = [artifact.path for artifact in client.list_artifacts(run_id)]
        except Exception as e:
            logger.warning(f"Could not list artifacts: {e}")

        return {
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
            "tags": tags,
            "artifacts": artifacts,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "status": run.info.status,
        }

    except Exception as e:
        logger.error(f"Failed to get MLflow run data: {e}")
        return None


def _load_backtest_data(run_data: Dict) -> Optional[Dict]:
    """Load backtest data from artifacts if available."""
    backtest_data = {}

    # Look for backtest CSV files in artifacts
    csv_files = [a for a in run_data["artifacts"] if a.endswith(".csv")]

    for csv_file in csv_files:
        try:
            # Try to load CSV from MLflow artifacts
            df = pd.read_csv(f"mlruns/0/{run_data['run_id']}/artifacts/{csv_file}")

            if "backtest_perf" in csv_file:
                backtest_data["performance"] = df
            elif "portfolio_alloc" in csv_file:
                backtest_data["allocations"] = df
            elif "exposures" in csv_file:
                backtest_data["exposures"] = df
            elif "equity_curve" in csv_file:
                backtest_data["equity_curve"] = df

        except Exception as e:
            logger.warning(f"Could not load {csv_file}: {e}")

    return backtest_data if backtest_data else None


def _generate_markdown_report(
    run_data: Dict,
    backtest_data: Optional[Dict],
    output_dir: Path,
    include_charts: bool,
) -> Path:
    """Generate markdown report."""
    report_path = output_dir / f"report_{run_data['run_id']}.md"

    with open(report_path, "w") as f:
        # Header
        f.write(f"# QEPO Portfolio Optimization Report\n\n")
        f.write(f"**Run ID:** {run_data['run_id']}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Status:** {run_data['status']}\n\n")

        # Configuration Summary
        f.write("## Configuration Summary\n\n")
        f.write("### Parameters\n")
        for key, value in run_data["params"].items():
            f.write(f"- **{key}:** {value}\n")
        f.write("\n")

        # Tags
        if run_data["tags"]:
            f.write("### Tags\n")
            for key, value in run_data["tags"].items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")

        # Performance Metrics
        if run_data["metrics"]:
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")

            # Group metrics by category
            return_metrics = {
                k: v
                for k, v in run_data["metrics"].items()
                if any(x in k.lower() for x in ["return", "sharpe", "sortino"])
            }
            risk_metrics = {
                k: v
                for k, v in run_data["metrics"].items()
                if any(x in k.lower() for x in ["volatility", "drawdown", "var"])
            }
            cost_metrics = {
                k: v
                for k, v in run_data["metrics"].items()
                if any(x in k.lower() for x in ["cost", "turnover", "transaction"])
            }
            other_metrics = {
                k: v
                for k, v in run_data["metrics"].items()
                if k not in return_metrics
                and k not in risk_metrics
                and k not in cost_metrics
            }

            # Write metrics by category
            if return_metrics:
                f.write("### Return Metrics\n")
                for key, value in return_metrics.items():
                    if "return" in key.lower() or "ratio" in key.lower():
                        f.write(f"| {key} | {value:.4f} |\n")
                    else:
                        f.write(f"| {key} | {value} |\n")
                f.write("\n")

            if risk_metrics:
                f.write("### Risk Metrics\n")
                for key, value in risk_metrics.items():
                    f.write(f"| {key} | {value:.4f} |\n")
                f.write("\n")

            if cost_metrics:
                f.write("### Cost Metrics\n")
                for key, value in cost_metrics.items():
                    f.write(f"| {key} | {value:.4f} |\n")
                f.write("\n")

            if other_metrics:
                f.write("### Other Metrics\n")
                for key, value in other_metrics.items():
                    f.write(f"| {key} | {value} |\n")
                f.write("\n")

        # Backtest Results
        if backtest_data:
            f.write("## Backtest Results\n\n")

            # Performance summary
            if "performance" in backtest_data:
                perf_df = backtest_data["performance"]
                f.write("### Performance Summary\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for col in perf_df.columns:
                    value = perf_df[col].iloc[0]
                    if isinstance(value, (int, float)):
                        if "return" in col.lower() or "ratio" in col.lower():
                            f.write(f"| {col} | {value:.4f} |\n")
                        else:
                            f.write(f"| {col} | {value:.6f} |\n")
                    else:
                        f.write(f"| {col} | {value} |\n")
                f.write("\n")

            # Portfolio allocations
            if "allocations" in backtest_data:
                alloc_df = backtest_data["allocations"]
                f.write("### Portfolio Allocations\n\n")

                # Show latest allocation
                if "date" in alloc_df.columns:
                    latest_date = alloc_df["date"].max()
                    latest_alloc = alloc_df[alloc_df["date"] == latest_date]
                    if not latest_alloc.empty:
                        f.write(f"**Latest Allocation ({latest_date}):**\n\n")
                        f.write("| Asset | Weight |\n")
                        f.write("|-------|--------|\n")

                        # Sort by weight descending
                        weight_cols = [
                            col for col in latest_alloc.columns if col != "date"
                        ]
                        latest_weights = latest_alloc[weight_cols].iloc[0]
                        sorted_weights = latest_weights.sort_values(ascending=False)

                        for asset, weight in sorted_weights.items():
                            if weight > 0.001:  # Only show significant weights
                                f.write(f"| {asset} | {weight:.3f} |\n")
                        f.write("\n")

            # Charts section
            if include_charts:
                f.write("## Charts\n\n")
                f.write(
                    "The following charts are available in the report directory:\n\n"
                )
                f.write("- `equity_curve.png` - Portfolio performance over time\n")
                f.write("- `portfolio_weights.png` - Portfolio weight heatmap\n")
                f.write("- `drawdown_chart.png` - Drawdown analysis\n")
                f.write(
                    "- `sector_exposure.png` - Sector exposure analysis (if available)\n\n"
                )

        # Artifacts
        if run_data["artifacts"]:
            f.write("## Artifacts\n\n")
            f.write("The following files were generated during this run:\n\n")
            for artifact in run_data["artifacts"]:
                f.write(f"- `{artifact}`\n")
            f.write("\n")

        # Footer
        f.write("---\n")
        f.write(
            f"*Report generated by QEPO on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        )

    # Generate charts if requested
    if include_charts and backtest_data:
        _generate_report_charts(backtest_data, output_dir)

    return report_path


def _generate_html_report(
    run_data: Dict,
    backtest_data: Optional[Dict],
    output_dir: Path,
    include_charts: bool,
) -> Path:
    """Generate HTML report."""
    report_path = output_dir / f"report_{run_data['run_id']}.html"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QEPO Portfolio Optimization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
            h3 {{ color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric-value {{ font-weight: bold; color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .positive {{ color: #27ae60; }}
        </style>
    </head>
    <body>
        <h1>QEPO Portfolio Optimization Report</h1>
        <p><strong>Run ID:</strong> {run_data['run_id']}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Status:</strong> {run_data['status']}</p>
    """

    # Add configuration section
    html_content += "<h2>Configuration Summary</h2>"
    html_content += "<h3>Parameters</h3><ul>"
    for key, value in run_data["params"].items():
        html_content += f"<li><strong>{key}:</strong> {value}</li>"
    html_content += "</ul>"

    # Add metrics section
    if run_data["metrics"]:
        html_content += "<h2>Performance Metrics</h2>"
        html_content += "<table><tr><th>Metric</th><th>Value</th></tr>"

        for key, value in run_data["metrics"].items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.4f}"
                css_class = "positive" if value > 0 else "negative" if value < 0 else ""
            else:
                formatted_value = str(value)
                css_class = ""

            html_content += f"<tr><td>{key}</td><td class='metric-value {css_class}'>{formatted_value}</td></tr>"

        html_content += "</table>"

    # Add backtest results
    if backtest_data:
        html_content += "<h2>Backtest Results</h2>"

        if "performance" in backtest_data:
            html_content += "<h3>Performance Summary</h3>"
            html_content += "<table><tr><th>Metric</th><th>Value</th></tr>"

            perf_df = backtest_data["performance"]
            for col in perf_df.columns:
                value = perf_df[col].iloc[0]
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                html_content += f"<tr><td>{col}</td><td class='metric-value'>{formatted_value}</td></tr>"

            html_content += "</table>"

    html_content += """
        <h2>Charts</h2>
        <p>Charts are available in the report directory as PNG files.</p>
        
        <hr>
        <p><em>Report generated by QEPO</em></p>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(html_content)

    # Generate charts if requested
    if include_charts and backtest_data:
        _generate_report_charts(backtest_data, output_dir)

    return report_path


def _generate_report_charts(backtest_data: Dict, output_dir: Path) -> None:
    """Generate charts for the report."""
    logger.info("Generating report charts")

    # Set style
    plt.style.use("seaborn-v0_8")

    # Equity curve chart
    if "equity_curve" in backtest_data:
        _plot_equity_curve(backtest_data["equity_curve"], output_dir)

    # Portfolio weights heatmap
    if "allocations" in backtest_data:
        _plot_portfolio_weights(backtest_data["allocations"], output_dir)

    # Drawdown chart
    if "equity_curve" in backtest_data:
        _plot_drawdown(backtest_data["equity_curve"], output_dir)

    # Sector exposure (if available)
    if "exposures" in backtest_data:
        _plot_sector_exposure(backtest_data["exposures"], output_dir)


def _plot_equity_curve(equity_curve_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot equity curve."""
    if equity_curve_df.empty or "portfolio_value" not in equity_curve_df.columns:
        return

    plt.figure(figsize=(12, 6))

    if "date" in equity_curve_df.columns:
        dates = pd.to_datetime(equity_curve_df["date"])
        plt.plot(
            dates, equity_curve_df["portfolio_value"], linewidth=2, label="Portfolio"
        )
    else:
        plt.plot(equity_curve_df["portfolio_value"], linewidth=2, label="Portfolio")

    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_dir / "equity_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_portfolio_weights(allocations_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot portfolio weights heatmap."""
    if allocations_df.empty or "date" not in allocations_df.columns:
        return

    # Prepare data for heatmap
    weight_cols = [col for col in allocations_df.columns if col != "date"]
    if not weight_cols:
        return

    # Select top holdings for visualization
    avg_weights = allocations_df[weight_cols].mean().sort_values(ascending=False)
    top_holdings = avg_weights.head(20).index  # Top 20 holdings

    weights_subset = allocations_df.set_index("date")[top_holdings]

    plt.figure(figsize=(15, 8))
    sns.heatmap(weights_subset.T, cmap="YlOrRd", cbar_kws={"label": "Weight"})
    plt.title("Portfolio Weights Over Time (Top 20 Holdings)")
    plt.xlabel("Date")
    plt.ylabel("Asset")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_dir / "portfolio_weights.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_drawdown(equity_curve_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot drawdown chart."""
    if equity_curve_df.empty or "portfolio_value" not in equity_curve_df.columns:
        return

    # Calculate drawdown
    portfolio_values = equity_curve_df["portfolio_value"]
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max

    plt.figure(figsize=(12, 6))

    if "date" in equity_curve_df.columns:
        dates = pd.to_datetime(equity_curve_df["date"])
        plt.fill_between(dates, drawdown, 0, alpha=0.3, color="red")
        plt.plot(dates, drawdown, color="red", linewidth=1)
    else:
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color="red")
        plt.plot(drawdown, color="red", linewidth=1)

    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_dir / "drawdown_chart.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_sector_exposure(exposures_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot sector exposure analysis."""
    if exposures_df.empty or "date" not in exposures_df.columns:
        return

    # This is a placeholder - would need sector mapping data
    # For now, just create a simple exposure plot
    weight_cols = [col for col in exposures_df.columns if col != "date"]
    if not weight_cols:
        return

    # Calculate average exposure
    avg_exposure = exposures_df[weight_cols].mean().sort_values(ascending=False)
    top_exposures = avg_exposure.head(15)

    plt.figure(figsize=(12, 8))
    top_exposures.plot(kind="bar")
    plt.title("Average Portfolio Exposure by Asset")
    plt.xlabel("Asset")
    plt.ylabel("Average Weight")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_dir / "sector_exposure.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(
    run_ids: List[str], output_dir: Optional[Path] = None, format: str = "markdown"
) -> Path:
    """
    Generate summary report comparing multiple runs.

    Parameters
    ----------
    run_ids : List[str]
        List of MLflow run IDs to compare.
    output_dir : Path, optional
        Directory to save report. Defaults to 'reports/'.
    format : str, default="markdown"
        Report format: 'markdown' or 'html'.

    Returns
    -------
    Path
        Path to the generated summary report file.
    """
    logger.info(f"Generating summary report for {len(run_ids)} runs")

    if output_dir is None:
        output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data for all runs
    runs_data = []
    for run_id in run_ids:
        run_data = _get_mlflow_run_data(run_id)
        if run_data:
            runs_data.append(run_data)

    if not runs_data:
        raise ValueError("No valid runs found")

    # Generate summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if format == "markdown":
        report_path = output_dir / f"summary_report_{timestamp}.md"
        _generate_summary_markdown(runs_data, report_path)
    elif format == "html":
        report_path = output_dir / f"summary_report_{timestamp}.html"
        _generate_summary_html(runs_data, report_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Summary report generated: {report_path}")
    return report_path


def _generate_summary_markdown(runs_data: List[Dict], report_path: Path) -> None:
    """Generate markdown summary report."""
    with open(report_path, "w") as f:
        f.write("# QEPO Portfolio Optimization Summary Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Number of Runs:** {len(runs_data)}\n\n")

        # Summary table
        f.write("## Run Summary\n\n")
        f.write(
            "| Run ID | Status | Strategy | Total Return | Sharpe Ratio | Max Drawdown |\n"
        )
        f.write(
            "|--------|--------|----------|--------------|--------------|--------------|\n"
        )

        for run_data in runs_data:
            run_id = run_data["run_id"]
            status = run_data["status"]
            strategy = run_data["tags"].get("strategy", "unknown")

            # Get key metrics
            total_return = run_data["metrics"].get("total_return", 0)
            sharpe_ratio = run_data["metrics"].get("sharpe_ratio", 0)
            max_drawdown = run_data["metrics"].get("max_drawdown", 0)

            f.write(
                f"| {run_id[:8]}... | {status} | {strategy} | {total_return:.4f} | {sharpe_ratio:.4f} | {max_drawdown:.4f} |\n"
            )

        f.write("\n")

        # Detailed comparison
        f.write("## Detailed Comparison\n\n")
        for i, run_data in enumerate(runs_data):
            f.write(f"### Run {i+1}: {run_data['run_id'][:8]}...\n\n")
            f.write(f"**Strategy:** {run_data['tags'].get('strategy', 'unknown')}\n")
            f.write(f"**Status:** {run_data['status']}\n\n")

            # Key metrics
            f.write("**Key Metrics:**\n")
            key_metrics = [
                "total_return",
                "annualized_return",
                "volatility",
                "sharpe_ratio",
                "max_drawdown",
            ]
            for metric in key_metrics:
                if metric in run_data["metrics"]:
                    value = run_data["metrics"][metric]
                    f.write(f"- {metric}: {value:.4f}\n")
            f.write("\n")


def _generate_summary_html(runs_data: List[Dict], report_path: Path) -> None:
    """Generate HTML summary report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QEPO Portfolio Optimization Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .best {{ background-color: #d5f4e6; }}
            .worst {{ background-color: #fadbd8; }}
        </style>
    </head>
    <body>
        <h1>QEPO Portfolio Optimization Summary Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Number of Runs:</strong> {len(runs_data)}</p>
        
        <h2>Run Summary</h2>
        <table>
            <tr>
                <th>Run ID</th>
                <th>Status</th>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
            </tr>
    """

    # Find best and worst performers
    total_returns = [
        run_data["metrics"].get("total_return", 0) for run_data in runs_data
    ]
    best_return = max(total_returns) if total_returns else 0
    worst_return = min(total_returns) if total_returns else 0

    for run_data in runs_data:
        run_id = run_data["run_id"]
        status = run_data["status"]
        strategy = run_data["tags"].get("strategy", "unknown")

        total_return = run_data["metrics"].get("total_return", 0)
        sharpe_ratio = run_data["metrics"].get("sharpe_ratio", 0)
        max_drawdown = run_data["metrics"].get("max_drawdown", 0)

        # Add CSS class for best/worst
        css_class = ""
        if total_return == best_return and best_return != worst_return:
            css_class = "best"
        elif total_return == worst_return and best_return != worst_return:
            css_class = "worst"

        html_content += f"""
            <tr class="{css_class}">
                <td>{run_id[:8]}...</td>
                <td>{status}</td>
                <td>{strategy}</td>
                <td>{total_return:.4f}</td>
                <td>{sharpe_ratio:.4f}</td>
                <td>{max_drawdown:.4f}</td>
            </tr>
        """

    html_content += """
        </table>
        
        <hr>
        <p><em>Report generated by QEPO</em></p>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(html_content)
