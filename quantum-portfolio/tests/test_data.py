from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qepo import data


def test_fetch_universe_exists():
    """Test that fetch_universe function is callable."""
    assert callable(data.fetch_universe)


@patch("qepo.data.pd.read_html")
def test_fetch_universe_returns_tickers(mock_read_html):
    """Test fetch_universe returns list of ticker symbols."""
    # Mock Wikipedia table
    mock_table = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "GOOGL"],
            "Security": ["Apple Inc.", "Microsoft Corp.", "Alphabet Inc."],
            "GICS Sector": ["Technology", "Technology", "Technology"],
            "GICS Sub-Industry": ["Computers", "Software", "Internet"],
            "Date added": ["1982-11-30", "1994-06-01", "2006-04-03"],
        }
    )
    mock_read_html.return_value = [mock_table]

    tickers = data.fetch_universe(save_metadata=False)

    assert isinstance(tickers, list)
    assert len(tickers) == 3
    assert "AAPL" in tickers
    assert "MSFT" in tickers


@patch("qepo.data.pd.read_html")
def test_fetch_universe_saves_metadata(mock_read_html, tmp_path):
    """Test fetch_universe saves metadata parquet file."""
    mock_table = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT"],
            "Security": ["Apple Inc.", "Microsoft Corp."],
            "GICS Sector": ["Technology", "Technology"],
            "GICS Sub-Industry": ["Computers", "Software"],
            "Date added": ["1982-11-30", "1994-06-01"],
        }
    )
    mock_read_html.return_value = [mock_table]

    tickers = data.fetch_universe(save_metadata=True, output_dir=tmp_path)

    meta_path = tmp_path / "meta.parquet"
    assert meta_path.exists()

    meta_df = pd.read_parquet(meta_path)
    assert len(meta_df) == 2
    assert "ticker" in meta_df.columns
    assert "sector" in meta_df.columns
    assert "industry" in meta_df.columns


@patch("qepo.data.pd.read_html")
def test_fetch_universe_handles_dots_in_symbols(mock_read_html):
    """Test that dots in ticker symbols are replaced with dashes."""
    mock_table = pd.DataFrame(
        {
            "Symbol": ["BRK.B", "BF.B"],
            "Security": ["Berkshire Hathaway", "Brown-Forman"],
            "GICS Sector": ["Financials", "Consumer Staples"],
            "GICS Sub-Industry": ["Insurance", "Beverages"],
            "Date added": ["2010-02-16", "1999-11-01"],
        }
    )
    mock_read_html.return_value = [mock_table]

    tickers = data.fetch_universe(save_metadata=False)

    assert "BRK-B" in tickers
    assert "BF-B" in tickers
    assert "BRK.B" not in tickers


@patch("qepo.data.pd.read_html")
def test_fetch_universe_raises_on_empty_table(mock_read_html):
    """Test fetch_universe raises ValueError on empty table."""
    mock_table = pd.DataFrame({"Symbol": []}, dtype=str)  # Ensure string dtype
    mock_read_html.return_value = [mock_table]

    with pytest.raises(ValueError, match="No tickers found"):
        data.fetch_universe(save_metadata=False)


@patch("qepo.data.yf.download")
def test_download_prices_returns_dataframe(mock_download, tmp_path):
    """Test download_prices returns DataFrame with correct schema."""
    # Mock yfinance download
    mock_data = pd.DataFrame(
        {"Close": [150.0, 151.0, 152.0]}, index=pd.date_range("2020-01-01", periods=3)
    )
    mock_download.return_value = mock_data

    result = data.download_prices(
        ["AAPL"], "2020-01-01", "2020-01-03", output_dir=tmp_path
    )

    assert isinstance(result, pd.DataFrame)
    assert "date" in result.columns
    assert "ticker" in result.columns
    assert "adj_close" in result.columns
    assert len(result) == 3


@patch("qepo.data.yf.download")
def test_download_prices_saves_parquet_files(mock_download, tmp_path):
    """Test download_prices saves prices.parquet and returns.parquet."""
    mock_data = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]}, index=pd.date_range("2020-01-01", periods=3)
    )
    mock_download.return_value = mock_data

    data.download_prices(["AAPL"], "2020-01-01", "2020-01-03", output_dir=tmp_path)

    assert (tmp_path / "prices.parquet").exists()
    assert (tmp_path / "returns.parquet").exists()

    # Validate schema
    prices_df = pd.read_parquet(tmp_path / "prices.parquet")
    assert list(prices_df.columns) == ["date", "ticker", "adj_close"]

    returns_df = pd.read_parquet(tmp_path / "returns.parquet")
    assert list(returns_df.columns) == ["date", "ticker", "ret_d"]


@patch("qepo.data.yf.download")
def test_download_prices_handles_empty_data(mock_download, tmp_path):
    """Test download_prices raises ValueError on empty data."""
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="Failed to download prices after 3 attempts"):
        data.download_prices(
            ["INVALID"], "2020-01-01", "2020-01-03", output_dir=tmp_path
        )


@patch("qepo.data.yf.download")
def test_download_prices_retries_on_failure(mock_download, tmp_path):
    """Test download_prices retries on failure."""
    # First two attempts fail, third succeeds
    mock_download.side_effect = [
        Exception("Network error"),
        Exception("Timeout"),
        pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2020-01-01", periods=1)),
    ]

    result = data.download_prices(
        ["AAPL"], "2020-01-01", "2020-01-01", output_dir=tmp_path, max_retries=3
    )

    assert not result.empty
    assert mock_download.call_count == 3
