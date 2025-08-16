import numpy as np
import requests
import yfinance as yf # type: ignore
import pandas as pd
import time
from tqdm import tqdm
import json
import typing
from typing import List
import regex as re
import urllib.request
import os
import logging


logger = logging.getLogger(__name__)

def load_excel_from_euronext(
        config_file: str = './config/config.json',
        verbose: bool = False
)-> pd.DataFrame:
    """
    Load the Euronext eligible assets from a local Excel file.

    Parameters
    ----------
    config_file : str
        Path to the configuration file containing the Excel file path.
    verbose : bool
        If True, print additional information during the fetching process.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the eligible assets.
    """
    # get this python file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the full path to the config file
    config_file_full = os.path.join(current_dir, config_file)
    
    #load config file
    try:
        with open(config_file_full, 'r') as f:
            config = json.load(f)["euronext_website_config"]
    except FileNotFoundError as e:
        if verbose: print(f"Config file not found at {config_file_full} or configuration 'euronext_website_config' missing: {e}")
    
    #get configuration details
    query_substr = config["dataset_name"]
    url = config["base_url"]
    header = config["header_line"]
    usecols = config["columns_to_use"]
    renaming_columns_dict = config["renaming_columns"]

    with urllib.request.urlopen(url) as fp:
        mybytes = fp.read()
        mystr = mybytes.decode("utf8")

    r = re.compile('(?<=href=").*?(?=")')
    links = re.findall(r,mystr)
    url_to_excel = "".join( s for s in np.unique(links) if query_substr in s)
    df_eligible_asset = pd.read_excel(url_to_excel, header=header, usecols=usecols)
    df_eligible_asset.rename(columns=renaming_columns_dict, inplace=True)
    return df_eligible_asset



def get_tickers_from_isins(
        isins: List[str],
        max_retries: int =10,
        batch_size: int =100,
        openfigi_api_key: typing.Optional[str]=None,
        verbose:bool=False)-> dict:
    """
    Query OpenFIGI to get tickers from a list of ISIN codes.

    Parameters
    ----------
    isins : list of str
        List of ISIN codes.
    max_retries : int
        Number of retries for failed requests (with exponential backoff).
    batch_size : int
        Max number of ISINs per request.
    openfigi_api_key : str, optional
        OpenFIGI API key for authentication. It is not required for public access, but recommended for higher rate limits.
    verbose : bool
        If True, print additional information during the fetching process.
    Returns
    -------
    dict
        Mapping {isin: ticker or None}
    """

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Cache-Control': 'private,no-cache,no-store,must-revalidate',
        'Expires': '0',
        'Pragma': 'no-cache'
    }
    # If an OpenFIGI API key is provided, add it to the headers
    if openfigi_api_key:
        headers['X-OPENFIGI-APIKEY'] = openfigi_api_key

    results = {}
    n_batches = (len(isins) + batch_size - 1) // batch_size

    # tqdm progress bar
    for i in tqdm(range(0, len(isins), batch_size), desc="Fetching ISINs", total=n_batches, disable=not verbose):
 
        batch = isins[i:i + batch_size]
        payload = [{"idType": "ID_ISIN", "idValue": isin} for isin in batch]

        delay = 2
        for attempt in range(max_retries):
            response = requests.post("https://api.openfigi.com/v3/mapping", 
                                    json=payload, headers=headers)

            if response.status_code != 200:
                time.sleep(delay)
                delay *= 2

                continue
            
            try:
                data = response.json()
                for isin, item in zip(batch, data):
                    if "data" in item and item["data"]:
                        results[isin] = item["data"][0].get("ticker")
                    else:
                        results[isin] = None
                break  # success → exit retry loop
            except Exception as e:
                if verbose: print(f"Error parsing batch {batch}: {e}")
                break

    return results

def get_suffix(
    markets: List[str],
    config_file: str ='./config/config.json',
    verbose:bool = False
    )-> List[str]:
    """
    Get the Yahoo Finance suffix for a given market.
    Parameters
    ----------
    market : str
        Market name (e.g., 'Dublin', 'Lisbon', etc.).
    config_file : str
        Path to the configuration file containing the mapping of cities to Yahoo Finance suffixes.
    verbose : bool
        If True, print additional information during the fetching process.
    Returns
    -------
    str
        Yahoo Finance suffix for the market.
    """
    # get this python file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the full path to the config file
    config_file_full = os.path.join(current_dir, config_file)

    #recover the mapping saved in config file
    try:
        with open(config_file_full, 'r') as f:
            mapping_dict = json.load(f)['city_to_yf_suffix']
    except FileNotFoundError as e:
        if verbose: print(f"Config file {config_file_full} not found or specific configuration 'city_to_yf_suffix' missing: {e}")

    results = []

    # Iterate through the markets and map them to their corresponding suffixes
    for market in markets:
        try:
            city_ = next((city for city in list(mapping_dict.keys()) if city.lower() in str(market).lower()), None)
            if city_ is None:
                if verbose: print(f"Market {market} not found in mapping. NaN will be returned.")
                results.append('NaN')
                continue
            results.append(mapping_dict[city_])
        except KeyError as e:
            if verbose: print(f"Market {market} not found in mapping. NaN will be returned.")
            results.append('NaN')
    return results

def load_fundamentals_from_yf(
    tickers: List[str],
    fundamentals: typing.Optional[List[str]] = None,
    config_file: str = './config/config.json',
    max_retries: int = 10,
    delay: float = 0.2,
    verbose = False) -> dict:
    """
    Load fundamental data from Yahoo Finance for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    fundamentals : list of str, optional
        List of fundamental data fields to fetch. If None, the defaulting list will be loaded from the config.json file.
        Common fields include 'trailingPE', 'forwardPE', 'priceToBook', 'dividendYield', etc.
    config_file : str
        Path to the configuration file.
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
        if verbose: print("No fundamentals provided, loading default from config file.")
        # get this python file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # construct the full path to the config file
        config_file_full = os.path.join(current_dir, config_file)

        #load config file
        try:
            with open(config_file_full, 'r') as f:
                _fundamentals = json.load(f)["yfinance_default_fundamentals"]
        except FileNotFoundError as e:
            if verbose: print(f"Config file not found at {config_file_full} or configuration 'yfinance_default_fundamentals' missing: {e}")
            return {}
    else:
        _fundamentals = fundamentals
    
    results = {}
    for ticker in tqdm(tickers, desc="Fetching fundamentals", disable=not verbose):
        retries = 0
        #manage retries with exponential backoff, but if error 404, do not retry
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
                if status_code == 404 or isinstance(e, requests.HTTPError) or "404" in str(e):
                    if verbose: logger.warning("Ticker %s returned 404/not found: %s", ticker, e)
                    results[ticker] = {}  # record as empty / missing
                    break

                retries += 1
                if retries >= max_retries:
                    if verbose: logger.error("Failed to fetch fundamentals for %s after %d retries: %s", ticker, retries, e)
                    results[ticker] = {}
                    break

                # exponential backoff with jitter
                backoff = delay * (2 ** (retries - 1))
                sleep_time = backoff 
                if verbose: logger.debug("Error fetching %s: %s. Retrying %d/%d after %.2fs", ticker, e, retries, max_retries, sleep_time)
                time.sleep(sleep_time)
        #add a small delay to avoid hitting the API rate limit
        time.sleep(delay)

    return results





