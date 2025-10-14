from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests

from qepo.report import (
    _generate_html_report,
    _generate_markdown_report,
    _generate_report_charts,
    _generate_summary_html,
    _generate_summary_markdown,
    _get_mlflow_run_data,
    _load_backtest_data,
    _plot_drawdown,
    _plot_equity_curve,
    _plot_portfolio_weights,
    _plot_sector_exposure,
    generate_report,
    generate_summary_report,
)


class TestMLflowRunData:
    """Test MLflow run data retrieval."""

    @patch("qepo.report.mlflow.tracking.MlflowClient")
    def test_get_mlflow_run_data_success(self, mock_client_class):
        """Test successful MLflow run data retrieval."""
        # Mock MLflow client and run
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_run = Mock()
        mock_run.data.params = {"param1": "value1", "param2": "value2"}
        mock_run.data.metrics = {"metric1": 1.5, "metric2": 2.3}
        mock_run.data.tags = {"tag1": "value1", "tag2": "value2"}
        mock_run.info.start_time = 1640995200000  # 2022-01-01
        mock_run.info.end_time = 1640995800000  # 2022-01-01 + 10 minutes
        mock_run.info.status = "FINISHED"

        mock_client.get_run.return_value = mock_run
        mock_client.list_artifacts.return_value = [
            Mock(path="backtest_perf.csv"),
            Mock(path="portfolio_alloc.csv"),
        ]

        result = _get_mlflow_run_data("test_run_id")

        assert result is not None
        assert result["run_id"] == "test_run_id"
        assert result["params"] == {"param1": "value1", "param2": "value2"}
        assert result["metrics"] == {"metric1": 1.5, "metric2": 2.3}
        assert result["tags"] == {"tag1": "value1", "tag2": "value2"}
        assert result["status"] == "FINISHED"
        assert "backtest_perf.csv" in result["artifacts"]
        assert "portfolio_alloc.csv" in result["artifacts"]

    @patch("qepo.report.mlflow.tracking.MlflowClient")
    def test_get_mlflow_run_data_failure(self, mock_client_class):
        """Test MLflow run data retrieval failure."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_run.side_effect = Exception("Run not found")

        result = _get_mlflow_run_data("nonexistent_run_id")

        assert result is None


class TestBacktestDataLoading:
    """Test backtest data loading from artifacts."""

    def test_load_backtest_data_success(self, tmp_path):
        """Test successful backtest data loading."""
        # Create sample CSV files
        perf_data = pd.DataFrame(
            {"total_return": [0.15], "sharpe_ratio": [1.2], "max_drawdown": [-0.05]}
        )
        perf_data.to_csv(tmp_path / "backtest_perf.csv", index=False)

        alloc_data = pd.DataFrame(
            {
                "date": ["2022-01-01", "2022-01-02"],
                "AAPL": [0.5, 0.4],
                "MSFT": [0.3, 0.4],
                "GOOGL": [0.2, 0.2],
            }
        )
        alloc_data.to_csv(tmp_path / "portfolio_alloc.csv", index=False)

        run_data = {
            "run_id": "test_run",
            "artifacts": ["backtest_perf.csv", "portfolio_alloc.csv"],
        }

        # Mock the file reading
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = [perf_data, alloc_data]

            result = _load_backtest_data(run_data)

        assert result is not None
        assert "performance" in result
        assert "allocations" in result
        assert result["performance"].equals(perf_data)
        assert result["allocations"].equals(alloc_data)

    def test_load_backtest_data_no_artifacts(self):
        """Test backtest data loading with no artifacts."""
        run_data = {"run_id": "test_run", "artifacts": []}

        result = _load_backtest_data(run_data)

        assert result is None


class TestMarkdownReport:
    """Test markdown report generation."""

    def test_generate_markdown_report_basic(self, tmp_path):
        """Test basic markdown report generation."""
        run_data = {
            "run_id": "test_run_123",
            "params": {"strategy": "mvo", "cardinality_k": 25},
            "metrics": {"total_return": 0.15, "sharpe_ratio": 1.2},
            "tags": {"command": "backtest"},
            "artifacts": ["backtest_perf.csv"],
            "start_time": 1640995200000,
            "end_time": 1640995800000,
            "status": "FINISHED",
        }

        backtest_data = {
            "performance": pd.DataFrame(
                {"total_return": [0.15], "sharpe_ratio": [1.2]}
            ),
            "allocations": pd.DataFrame(
                {"date": ["2022-01-01"], "AAPL": [0.5], "MSFT": [0.3], "GOOGL": [0.2]}
            ),
        }

        report_path = _generate_markdown_report(
            run_data, backtest_data, tmp_path, False
        )

        assert report_path.exists()

        content = report_path.read_text()
        assert "QEPO Portfolio Optimization Report" in content
        assert "test_run_123" in content
        assert "Configuration Summary" in content
        assert "Performance Metrics" in content
        assert "Backtest Results" in content
        assert "strategy" in content
        assert "mvo" in content
        assert "cardinality_k" in content
        assert "25" in content
        assert "0.1500" in content  # total_return formatted
        assert "1.2000" in content  # sharpe_ratio formatted

    def test_generate_markdown_report_no_backtest_data(self, tmp_path):
        """Test markdown report generation without backtest data."""
        run_data = {
            "run_id": "test_run_123",
            "params": {"strategy": "mvo"},
            "metrics": {"total_return": 0.15},
            "tags": {},
            "artifacts": [],
            "start_time": 1640995200000,
            "end_time": 1640995800000,
            "status": "FINISHED",
        }

        report_path = _generate_markdown_report(run_data, None, tmp_path, False)

        assert report_path.exists()

        content = report_path.read_text()
        assert "QEPO Portfolio Optimization Report" in content
        assert "Configuration Summary" in content
        assert "Performance Metrics" in content
        assert "Backtest Results" not in content


class TestHTMLReport:
    """Test HTML report generation."""

    def test_generate_html_report_basic(self, tmp_path):
        """Test basic HTML report generation."""
        run_data = {
            "run_id": "test_run_123",
            "params": {"strategy": "mvo", "cardinality_k": 25},
            "metrics": {"total_return": 0.15, "sharpe_ratio": 1.2},
            "tags": {"command": "backtest"},
            "artifacts": ["backtest_perf.csv"],
            "start_time": 1640995200000,
            "end_time": 1640995800000,
            "status": "FINISHED",
        }

        backtest_data = {
            "performance": pd.DataFrame({"total_return": [0.15], "sharpe_ratio": [1.2]})
        }

        report_path = _generate_html_report(run_data, backtest_data, tmp_path, False)

        assert report_path.exists()

        content = report_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<title>QEPO Portfolio Optimization Report</title>" in content
        assert "test_run_123" in content
        assert "Configuration Summary" in content
        assert "Performance Metrics" in content
        assert "strategy" in content
        assert "mvo" in content
        assert "0.1500" in content


class TestChartGeneration:
    """Test chart generation functions."""

    def test_plot_equity_curve(self, tmp_path):
        """Test equity curve plotting."""
        equity_data = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=10, freq="D"),
                "portfolio_value": [
                    1.0,
                    1.01,
                    1.02,
                    0.99,
                    1.03,
                    1.05,
                    1.04,
                    1.06,
                    1.08,
                    1.07,
                ],
            }
        )

        with patch("matplotlib.pyplot.show"):
            _plot_equity_curve(equity_data, tmp_path)

        assert (tmp_path / "equity_curve.png").exists()

    def test_plot_equity_curve_empty_data(self, tmp_path):
        """Test equity curve plotting with empty data."""
        empty_data = pd.DataFrame()

        with patch("matplotlib.pyplot.show"):
            _plot_equity_curve(empty_data, tmp_path)

        # Should not create file for empty data
        assert not (tmp_path / "equity_curve.png").exists()

    def test_plot_portfolio_weights(self, tmp_path):
        """Test portfolio weights plotting."""
        weights_data = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=5, freq="D"),
                "AAPL": [0.5, 0.4, 0.6, 0.5, 0.4],
                "MSFT": [0.3, 0.4, 0.2, 0.3, 0.4],
                "GOOGL": [0.2, 0.2, 0.2, 0.2, 0.2],
            }
        )

        with patch("matplotlib.pyplot.show"):
            _plot_portfolio_weights(weights_data, tmp_path)

        assert (tmp_path / "portfolio_weights.png").exists()

    def test_plot_drawdown(self, tmp_path):
        """Test drawdown plotting."""
        equity_data = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=10, freq="D"),
                "portfolio_value": [
                    1.0,
                    1.01,
                    1.02,
                    0.99,
                    1.03,
                    1.05,
                    1.04,
                    1.06,
                    1.08,
                    1.07,
                ],
            }
        )

        with patch("matplotlib.pyplot.show"):
            _plot_drawdown(equity_data, tmp_path)

        assert (tmp_path / "drawdown_chart.png").exists()

    def test_plot_sector_exposure(self, tmp_path):
        """Test sector exposure plotting."""
        exposure_data = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=3, freq="D"),
                "AAPL": [0.5, 0.4, 0.6],
                "MSFT": [0.3, 0.4, 0.2],
                "GOOGL": [0.2, 0.2, 0.2],
            }
        )

        with patch("matplotlib.pyplot.show"):
            _plot_sector_exposure(exposure_data, tmp_path)

        assert (tmp_path / "sector_exposure.png").exists()

    def test_generate_report_charts(self, tmp_path):
        """Test report chart generation."""
        backtest_data = {
            "equity_curve": pd.DataFrame(
                {
                    "date": pd.date_range("2022-01-01", periods=5, freq="D"),
                    "portfolio_value": [1.0, 1.01, 1.02, 0.99, 1.03],
                }
            ),
            "allocations": pd.DataFrame(
                {
                    "date": pd.date_range("2022-01-01", periods=3, freq="D"),
                    "AAPL": [0.5, 0.4, 0.6],
                    "MSFT": [0.3, 0.4, 0.2],
                }
            ),
            "exposures": pd.DataFrame(
                {
                    "date": pd.date_range("2022-01-01", periods=3, freq="D"),
                    "AAPL": [0.5, 0.4, 0.6],
                    "MSFT": [0.3, 0.4, 0.2],
                }
            ),
        }

        with patch("matplotlib.pyplot.show"):
            _generate_report_charts(backtest_data, tmp_path)

        # Check that chart files were created
        assert (tmp_path / "equity_curve.png").exists()
        assert (tmp_path / "portfolio_weights.png").exists()
        assert (tmp_path / "drawdown_chart.png").exists()
        assert (tmp_path / "sector_exposure.png").exists()


class TestSummaryReport:
    """Test summary report generation."""

    def test_generate_summary_markdown(self, tmp_path):
        """Test summary markdown report generation."""
        runs_data = [
            {
                "run_id": "run_1",
                "status": "FINISHED",
                "tags": {"strategy": "mvo"},
                "metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.05,
                },
            },
            {
                "run_id": "run_2",
                "status": "FINISHED",
                "tags": {"strategy": "greedy"},
                "metrics": {
                    "total_return": 0.12,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.08,
                },
            },
        ]

        report_path = tmp_path / "summary.md"
        _generate_summary_markdown(runs_data, report_path)

        assert report_path.exists()

        content = report_path.read_text()
        assert "QEPO Portfolio Optimization Summary Report" in content
        assert "Number of Runs:" in content
        assert "2" in content
        assert "Run Summary" in content
        assert "run_1" in content
        assert "run_2" in content
        assert "mvo" in content
        assert "greedy" in content
        assert "0.1500" in content
        assert "0.1200" in content

    def test_generate_summary_html(self, tmp_path):
        """Test summary HTML report generation."""
        runs_data = [
            {
                "run_id": "run_1",
                "status": "FINISHED",
                "tags": {"strategy": "mvo"},
                "metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.05,
                },
            },
            {
                "run_id": "run_2",
                "status": "FINISHED",
                "tags": {"strategy": "greedy"},
                "metrics": {
                    "total_return": 0.12,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.08,
                },
            },
        ]

        report_path = tmp_path / "summary.html"
        _generate_summary_html(runs_data, report_path)

        assert report_path.exists()

        content = report_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<title>QEPO Portfolio Optimization Summary Report</title>" in content
        assert "Number of Runs:" in content
        assert "2" in content
        assert "run_1" in content
        assert "run_2" in content
        assert "0.1500" in content
        assert "0.1200" in content


class TestMainReportGeneration:
    """Test main report generation function."""

    @patch("qepo.report._get_mlflow_run_data")
    @patch("qepo.report._load_backtest_data")
    @patch("qepo.report._generate_markdown_report")
    def test_generate_report_success(
        self, mock_markdown, mock_load_data, mock_get_data, tmp_path
    ):
        """Test successful report generation."""
        # Mock the dependencies
        mock_get_data.return_value = {"run_id": "test_run", "params": {}, "metrics": {}}
        mock_load_data.return_value = {"performance": pd.DataFrame()}
        mock_markdown.return_value = tmp_path / "report.md"

        result = generate_report("test_run", tmp_path, False, "markdown")

        assert result == tmp_path / "report.md"
        mock_get_data.assert_called_once_with("test_run")
        mock_load_data.assert_called_once()
        mock_markdown.assert_called_once()

    @patch("qepo.report._get_mlflow_run_data")
    def test_generate_report_no_run_data(self, mock_get_data):
        """Test report generation with no run data."""
        mock_get_data.return_value = None

        with pytest.raises(ValueError, match="Run .* not found or has no data"):
            generate_report("nonexistent_run")

    @patch("qepo.report._get_mlflow_run_data")
    @patch("qepo.report._load_backtest_data")
    @patch("qepo.report._generate_html_report")
    def test_generate_report_html_format(
        self, mock_html, mock_load_data, mock_get_data, tmp_path
    ):
        """Test HTML format report generation."""
        mock_get_data.return_value = {"run_id": "test_run", "params": {}, "metrics": {}}
        mock_load_data.return_value = None
        mock_html.return_value = tmp_path / "report.html"

        result = generate_report("test_run", tmp_path, False, "html")

        assert result == tmp_path / "report.html"
        mock_html.assert_called_once()

    @patch("qepo.report._get_mlflow_run_data")
    def test_generate_report_invalid_format(self, mock_get_data):
        """Test report generation with invalid format."""
        mock_get_data.return_value = {
            "run_id": "test_run",
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

        with pytest.raises(ValueError, match="Unsupported format"):
            generate_report("test_run", format="invalid")


class TestSummaryReportGeneration:
    """Test summary report generation function."""

    @patch("qepo.report._get_mlflow_run_data")
    @patch("qepo.report._generate_summary_markdown")
    def test_generate_summary_report_success(
        self, mock_markdown, mock_get_data, tmp_path
    ):
        """Test successful summary report generation."""
        # Mock run data
        run_data = {
            "run_id": "test_run",
            "status": "FINISHED",
            "tags": {"strategy": "mvo"},
            "metrics": {"total_return": 0.15},
        }
        mock_get_data.return_value = run_data
        mock_markdown.return_value = tmp_path / "summary.md"

        result = generate_summary_report(["test_run"], tmp_path, "markdown")

        assert result.name.startswith("summary_report_")
        assert result.suffix == ".md"
        mock_get_data.assert_called_once_with("test_run")
        mock_markdown.assert_called_once()

    @patch("qepo.report._get_mlflow_run_data")
    def test_generate_summary_report_no_valid_runs(self, mock_get_data):
        """Test summary report generation with no valid runs."""
        mock_get_data.return_value = None

        with pytest.raises(ValueError, match="No valid runs found"):
            generate_summary_report(["invalid_run"])

    @patch("qepo.report._get_mlflow_run_data")
    @patch("qepo.report._generate_summary_html")
    def test_generate_summary_report_html_format(
        self, mock_html, mock_get_data, tmp_path
    ):
        """Test HTML format summary report generation."""
        run_data = {
            "run_id": "test_run",
            "status": "FINISHED",
            "tags": {"strategy": "mvo"},
            "metrics": {"total_return": 0.15},
        }
        mock_get_data.return_value = run_data
        mock_html.return_value = tmp_path / "summary.html"

        result = generate_summary_report(["test_run"], tmp_path, "html")

        assert result.name.startswith("summary_report_")
        assert result.suffix == ".html"
        mock_html.assert_called_once()


class TestIntegration:
    """Integration tests for report module."""

    def test_full_report_workflow(self, tmp_path):
        """Test complete report generation workflow."""
        # Create sample data files
        perf_data = pd.DataFrame(
            {"total_return": [0.15], "sharpe_ratio": [1.2], "max_drawdown": [-0.05]}
        )
        perf_data.to_csv(tmp_path / "backtest_perf.csv", index=False)

        alloc_data = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=5, freq="D"),
                "AAPL": [0.5, 0.4, 0.6, 0.5, 0.4],
                "MSFT": [0.3, 0.4, 0.2, 0.3, 0.4],
                "GOOGL": [0.2, 0.2, 0.2, 0.2, 0.2],
            }
        )
        alloc_data.to_csv(tmp_path / "portfolio_alloc.csv", index=False)

        equity_data = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=5, freq="D"),
                "portfolio_value": [1.0, 1.01, 1.02, 0.99, 1.03],
            }
        )
        equity_data.to_csv(tmp_path / "equity_curve.csv", index=False)

        # Mock MLflow data
        run_data = {
            "run_id": "integration_test_run",
            "params": {"strategy": "mvo", "cardinality_k": 25},
            "metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.05,
            },
            "tags": {"command": "backtest", "strategy": "mvo"},
            "artifacts": [
                "backtest_perf.csv",
                "portfolio_alloc.csv",
                "equity_curve.csv",
            ],
            "start_time": 1640995200000,
            "end_time": 1640995800000,
            "status": "FINISHED",
        }

        # Mock the MLflow and file operations
        with patch("qepo.report._get_mlflow_run_data", return_value=run_data), patch(
            "pandas.read_csv"
        ) as mock_read_csv, patch("matplotlib.pyplot.show"):

            # Set up mock CSV reading
            mock_read_csv.side_effect = [perf_data, alloc_data, equity_data]

            # Generate report
            report_path = generate_report(
                "integration_test_run", tmp_path, include_charts=True, format="markdown"
            )

        # Verify report was generated
        assert report_path.exists()

        # Check report content
        content = report_path.read_text()
        assert "QEPO Portfolio Optimization Report" in content
        assert "integration_test_run" in content
        assert "Configuration Summary" in content
        assert "Performance Metrics" in content
        assert "Backtest Results" in content
        assert "Charts" in content

        # Check that chart files were created
        assert (tmp_path / "equity_curve.png").exists()
        assert (tmp_path / "portfolio_weights.png").exists()
        assert (tmp_path / "drawdown_chart.png").exists()
        # Note: sector_exposure.png may not be created if data structure doesn't match expectations
