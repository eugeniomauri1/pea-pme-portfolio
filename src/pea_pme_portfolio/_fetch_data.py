import numpy as np
import requests
import yfinance as yf  # type: ignore
import pandas as pd
import time
import typing
from typing import List
import regex as re
import urllib.request
import os
import sys


def tqdm(*args, **kwargs):
    """
    Environment-aware tqdm wrapper:
    - In Jupyter/IPython kernels, use tqdm.notebook.tqdm
    - Otherwise, use tqdm.tqdm
    Always writes to sys.stdout to avoid 'missing bar' issues in VS Code/devcontainers.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip and "IPKernelApp" in ip.config:  # Jupyter/IPython kernel
            from tqdm.notebook import tqdm as _tqdm
        else:
            from tqdm import tqdm as _tqdm
    except Exception:
        from tqdm import tqdm as _tqdm

    return _tqdm(*args, file=sys.stdout, **kwargs)


euronext_website_config = {
    "base_url": "https://connect2.euronext.com/en/media/169",
    "dataset_name": "liste_pea_pme",
    "header_line": 16,
    "columns_to_use": [3, 4, 5, 6, 7],
    "renaming_columns": {
        "Société/Company": "Company",
        "CodeISIN/ISINCode": "ISIN",
        "Marché/Market": "Market",
        "Compartiment/Compartment": "Compartment",
        "Pays d'incorporation/Country of Incorporation": "Country",
    },
}

yfinance_default_fundamentals = [
    "industry",
    "sector",
    "overallRisk",
    "beta",
    "dividendYield",
    "fiveYearAvgDividendYield",
    "trailingPE",
    "forwardPE",
    "regularMarketVolume",
    "marketCap",
    "currency",
    "enterpriseValue",
    "profitMargins",
    "bookValue",
    "priceToBook",
    "trailingEps",
    "forwardEps",
    "totalCash",
    "totalCashPerShare",
    "totalDebt",
    "totalRevenue",
    "revenuePerShare",
    "returnOnAssets",
    "returnOnEquity",
    "grossProfits",
    "operatingMargins",
]


def load_excel_from_euronext() -> pd.DataFrame:
    """
    Load the Euronext eligible assets from a local Excel file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the eligible assets.
    """
    config = euronext_website_config

    # get configuration details
    query_substr = config["dataset_name"]
    url = str(config["base_url"])
    header = config["header_line"]
    usecols = config["columns_to_use"]
    renaming_columns_dict = config["renaming_columns"]

    with urllib.request.urlopen(url) as fp:
        mybytes = fp.read()
        mystr = mybytes.decode("utf8")

    r = re.compile('(?<=href=").*?(?=")')
    links = re.findall(r, mystr)
    url_to_excel = "".join(s for s in np.unique(links) if query_substr in s)
    df_eligible_asset = pd.read_excel(url_to_excel, header=header, usecols=usecols)
    df_eligible_asset.rename(columns=renaming_columns_dict, inplace=True)
    return df_eligible_asset


def get_tickers_from_isins(
    isins: List[str],
    max_retries: int = 5,
    base_delay: float = 1.0,
    verbose: bool = False,
) -> dict[str, str]:
    results = {}

    iterator = tqdm(isins, desc="Fetching ISINs", disable=not verbose)

    for isin in iterator:
        for attempt in range(max_retries):
            try:
                ticker = yf.utils.get_ticker_by_isin(isin)

                results[isin] = ticker
                break

            except Exception:
                # exponential backoff + jitter
                sleep_time = base_delay * (2**attempt)
                sleep_time += np.random.uniform(0, 1)

                if verbose:
                    print(
                        f"[Retry {attempt + 1}] ISIN {isin} failed. Sleeping {sleep_time:.2f}s"
                    )

                time.sleep(sleep_time)

        else:
            # If all retries exhausted
            results[isin] = ""
            if verbose:
                print(f"[FAILED] {isin}")

        # small delay to reduce rate limiting
        time.sleep(0.5)

    return results


def load_fundamentals_from_yf(
    tickers: List[str],
    fundamentals: typing.Optional[List[str]] = None,
    max_retries: int = 10,
    delay: float = 0.2,
    verbose=False,
) -> dict:
    """
    Load fundamental data from Yahoo Finance for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    fundamentals : list of str, optional
        List of fundamental data fields to fetch. If None, the defaulting list will be loaded from the config.json file.
        Common fields include 'trailingPE', 'forwardPE', 'priceToBook', 'dividendYield', etc.
    max_retries : int
        Number of retries for failed requests (with exponential backoff).
    delay : float
        Initial delay between requests in seconds.
    verbose : bool
        If True, print additional information during the fetching process.

    Returns
    -------
    dict
        Dictionary mapping ticker symbols to their fundamental data.
    """
    if fundamentals is None:
        if verbose:
            print("No fundamentals provided, loading default one.")
        _fundamentals = yfinance_default_fundamentals
    else:
        _fundamentals = fundamentals

    results = {}
    pbar = tqdm(tickers, desc="Fetching fundamentals", disable=not verbose)
    not_found_list = []
    failed_fetching = []
    for ticker in pbar:
        retries = 0
        # manage retries with exponential backoff, but if error 404, do not retry
        while True:
            try:
                # fetch info
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info or {}

                # Sometimes yfinance returns a dict with 'regularMarketPrice' etc. If info is empty, raise to trigger retry logic.
                if not info:
                    raise ValueError(f"No info returned for {ticker}")

                # Extract only requested fundamentals (use .get to avoid KeyError)
                ticker_data = {field: info.get(field) for field in _fundamentals}
                results[ticker] = ticker_data
                break  # success -> break retry loop

            except Exception as e:
                # Try to detect HTTP 404 (non-retriable). Some exceptions expose a response with status_code.
                status_code = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    status_code = getattr(resp, "status_code", None)

                # If the underlying exception is requests.HTTPError or response shows 404, treat as non-retriable
                if (
                    status_code == 404
                    or isinstance(e, requests.HTTPError)
                    or "404" in str(e)
                ):
                    if verbose:
                        not_found_list.append(f"{ticker}")
                    results[ticker] = {}  # record as empty / missing
                    break

                retries += 1
                if retries >= max_retries:
                    if verbose:
                        failed_fetching.append(f"{ticker}")
                    results[ticker] = {}
                    break

                # exponential backoff with jitter
                backoff = delay * (2 ** (retries - 1))
                sleep_time = backoff
                time.sleep(sleep_time)
        # add a small delay to avoid hitting the API rate limit
        time.sleep(delay)
    if verbose:
        if not_found_list:
            print(f"Tickers not found in Yahoo Finance: {', '.join(not_found_list)}")
        if failed_fetching:
            print(
                f"Failed to fetch data for tickers after retries: {', '.join(failed_fetching)}"
            )

    return results


def data_loader(
    verbose: bool = False,
    save_to_csv: bool = False,
    kwargs: dict = {"max_retries": 10, "delay": 0.2},
) -> typing.Union[pd.DataFrame, pd.Series]:
    """
    Load the list of PEA-PME eligible assets available on the Euronext exchanges.

    Parameters
    ----------
    verbose : bool
        If True, print additional information during the fetching process.
    save_to_csv : bool
        If True, save the DataFrame to a CSV file.
    kwargs_ticker_from_isins : dict, optional
        Additional keyword arguments to pass to the `get_tickers_from_isins` function, such as `max_retries`, `base_delay`, etc.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the eligible assets and their fundamentals.
    """
    # get this python file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the full path to the config file
    output_dir = os.path.join(current_dir, "../../output/")

    df_eligible_asset = pd.DataFrame()

    if verbose:
        print("Loading Euronext eligible assets from Excel file...")

    df_eligible_asset = load_excel_from_euronext()

    # get tickers from ISINs
    if verbose:
        print("Fetching tickers from ISINs...")
    isins = df_eligible_asset["ISIN"].tolist()
    tickers = get_tickers_from_isins(
        isins,
        verbose=verbose,
        max_retries=kwargs.get("max_retries", 10),
        base_delay=kwargs.get("delay", 0.2),
    )

    # add tickers to DataFrame
    df_eligible_asset["Ticker"] = df_eligible_asset["ISIN"].map(tickers)

    if save_to_csv:
        if verbose:
            print(f"Saving raw DataFrame at {output_dir + 'peapmea_assets_raw.csv'}...")
        df_eligible_asset.to_csv(output_dir + "peapmea_assets_raw.csv", index=False)

    # drop rowns with None in Ticker
    df_eligible_asset = df_eligible_asset[df_eligible_asset["Ticker"] != ""]

    if verbose:
        print("Fetching fundamentals from Yahoo Finance...")
    fundamentals_from_yf = load_fundamentals_from_yf(
        tickers=df_eligible_asset["Ticker"].tolist(),
        verbose=verbose,
        max_retries=kwargs.get("max_retries", 10),
        delay=kwargs.get("delay", 0.2),
    )

    if verbose:
        print("Adding fundamentals to DataFrame...")
    # add fundamentals to DataFrame
    for i in df_eligible_asset.index:
        ticker = df_eligible_asset.loc[i, "Ticker"]
        if ticker in list(fundamentals_from_yf.keys()):
            for key, value in fundamentals_from_yf[ticker].items():
                df_eligible_asset.loc[i, key] = value
        else:
            if verbose:
                print(f"Ticker {ticker} not found in fundamentals data. Skipping.")
    if save_to_csv:
        if verbose:
            print(
                f"Saving final DataFrame at {output_dir + 'peapme_assets_with_fundamentals.csv'}..."
            )
        df_eligible_asset.to_csv(
            output_dir + "peapme_assets_with_fundamentals.csv", index=False
        )
    return df_eligible_asset
