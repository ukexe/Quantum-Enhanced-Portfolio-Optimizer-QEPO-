import logging
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Cursor Task: Data ingestion module for S&P 500 universe and historical prices


def fetch_universe(
    save_metadata: bool = True, output_dir: Optional[Path] = None
) -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia.

    Scrapes the Wikipedia page for S&P 500 companies and extracts ticker symbols,
    sector, and industry information.

    Parameters
    ----------
    save_metadata : bool, default=True
            If True, saves company metadata (sector, industry) to parquet.
    output_dir : Path, optional
            Directory to save metadata. Defaults to 'data/interim'.

    Returns
    -------
    List[str]
            List of ticker symbols.

    Raises
    ------
    ValueError
            If Wikipedia table cannot be parsed or no tickers found.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching S&P 500 constituents from Wikipedia: {url}")

    try:
        # Try the original pd.read_html approach first with headers
        import urllib.request
        
        # Create a request with proper headers
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        # Get the HTML content
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
        
        # Read all tables from the HTML content (fix deprecation warning)
        tables = pd.read_html(StringIO(html_content))
        
        # Find the table with current constituents (should be the first one)
        sp500_table = None
        for i, table in enumerate(tables):
            logger.info(f"Table {i} columns: {list(table.columns)}")
            
            # Check if this table has a Symbol column
            if any('symbol' in col.lower() for col in table.columns):
                sp500_table = table
                logger.info(f"Using table {i} as it contains symbol column")
                break
        
        if sp500_table is None:
            raise ValueError("Could not find a table with symbol column")

        # Extract ticker symbols (handle different column names)
        symbol_col = None
        for col in sp500_table.columns:
            if 'symbol' in col.lower():
                symbol_col = col
                break
        
        if symbol_col is None:
            raise ValueError(f"Could not find symbol column. Available columns: {list(sp500_table.columns)}")
        
        tickers = sp500_table[symbol_col].str.replace(".", "-", regex=False).tolist()

        if not tickers:
            raise ValueError("No tickers found in Wikipedia table")

        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers")

        # Save metadata if requested
        if save_metadata:
            if output_dir is None:
                output_dir = Path("data/interim")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create metadata dataframe (handle different column names)
            meta_df = pd.DataFrame(
                {
                    "ticker": sp500_table[symbol_col].str.replace(".", "-", regex=False),
                    "name": sp500_table.get("Security", sp500_table.get("Company", "")),
                    "sector": sp500_table.get("GICS Sector", sp500_table.get("Sector", "")),
                    "industry": sp500_table.get("GICS Sub-Industry", sp500_table.get("Industry", "")),
                    "included_from": pd.to_datetime(sp500_table.get("Date added", sp500_table.get("Date Added", ""))),
                }
            )

            meta_path = output_dir / "meta.parquet"
            meta_df.to_parquet(meta_path, index=False)
            logger.info(f"Saved metadata to {meta_path}")

        return tickers

    except Exception as e:
        logger.error(f"Failed to fetch universe from Wikipedia: {e}")
        raise


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    output_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download historical prices via yfinance and persist parquet files.

    Parameters
    ----------
    tickers : List[str]
            List of ticker symbols to download.
    start : str
            Start date in 'YYYY-MM-DD' format.
    end : str
            End date in 'YYYY-MM-DD' format.
    output_dir : Path, optional
            Directory to save parquet files. Defaults to 'data/interim'.
    max_retries : int, default=3
            Maximum number of retry attempts for failed downloads.

    Returns
    -------
    pd.DataFrame
            DataFrame with columns: date, ticker, adj_close

    Raises
    ------
    ValueError
            If no price data could be downloaded.
    """
    if output_dir is None:
        output_dir = Path("data/interim")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading prices for {len(tickers)} tickers from {start} to {end}")

    # Download data with retries
    for attempt in range(max_retries):
        try:
            # Download with yfinance (group_by='ticker' for multi-ticker downloads)
            data = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
            )

            if data.empty:
                raise ValueError("Downloaded data is empty")

            break
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError(
                    f"Failed to download prices after {max_retries} attempts"
                )

    # Reshape to long format: date, ticker, adj_close
    prices_list = []

    if len(tickers) == 1:
        # Single ticker case
        df = data.copy()
        df["ticker"] = tickers[0]
        df["date"] = df.index
        df = df[["date", "ticker", "Close"]].rename(columns={"Close": "adj_close"})
        prices_list.append(df)
    else:
        # Multiple tickers
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                ticker_data = data[ticker]["Close"].to_frame()
                ticker_data["ticker"] = ticker
                ticker_data["date"] = ticker_data.index
                ticker_data = ticker_data[["date", "ticker", "Close"]].rename(
                    columns={"Close": "adj_close"}
                )
                prices_list.append(ticker_data)

    prices_df = pd.concat(prices_list, ignore_index=True)
    prices_df = prices_df.dropna()  # Remove rows with missing prices

    if prices_df.empty:
        raise ValueError("No valid price data after processing")

    # Save to parquet
    prices_path = output_dir / "prices.parquet"
    prices_df.to_parquet(prices_path, index=False)
    logger.info(f"Saved {len(prices_df)} price records to {prices_path}")

    # Also compute and save returns
    returns_df = prices_df.copy()
    returns_df = returns_df.sort_values(["ticker", "date"])
    returns_df["ret_d"] = returns_df.groupby("ticker")["adj_close"].pct_change()
    returns_df = returns_df.dropna()

    returns_path = output_dir / "returns.parquet"
    returns_df[["date", "ticker", "ret_d"]].to_parquet(returns_path, index=False)
    logger.info(f"Saved {len(returns_df)} return records to {returns_path}")

    return prices_df
