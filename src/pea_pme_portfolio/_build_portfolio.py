import pandas as pd
import numpy as np
import yfinance as yf
import scipy as sc


def get_value_portfolio(
    all_assets: pd.DataFrame,
    max_pe: float = 20.0,
    min_pe: float = 0.0,
    max_yield: float = 5,
    min_yield: float = 0.0,
    number_of_assets: int = 10,
    diversification_factor: float = 0.95,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Create a value portfolio based on the provided asset DataFrame.

    Parameters:
        all_assets (pd.DataFrame): DataFrame containing asset data.
        max_pe (float): Maximum price-to-earnings ratio for asset selection.
        min_pe (float): Minimum price-to-earnings ratio for asset selection.
        max_yield (float): Maximum yield for asset selection (in percentage points).
        min_yield (float): Minimum yield for asset selection (in percentage points).
        number_of_assets (int): Number of assets to include in the portfolio.
        diversification_factor (float): A number between zero and one to control the diversification of the portfolio (1 for equal weights, 0 for no diversification). Lower values of `diversification_factor` will give more weight to the assets with higher market capitalisation.
        verbose (bool): If True, print additional information.

    Returns:
        pd.DataFrame: DataFrame containing the selected value portfolio assets.
    """

    # assert that the DataFrame has the required columns
    required_columns = [
        "Ticker",
        "trailingPE",
        "marketCap",
        "fiveYearAvgDividendYield",
        "currency",
    ]

    for col in required_columns:
        if col not in all_assets.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter assets based on P/E ratio and yield
    filtered_assets = all_assets[
        (all_assets["trailingPE"] <= max_pe)
        & (all_assets["trailingPE"] >= min_pe)
        & (all_assets["fiveYearAvgDividendYield"] <= max_yield)
        & (all_assets["fiveYearAvgDividendYield"] >= min_yield)
        & (all_assets["marketCap"] > 0)  # Ensure marketCap is positive
        &
        # curency is not nan
        (all_assets["currency"].notna())
    ].copy()
    if verbose:
        print(f"Filtered assets: {len(filtered_assets)}")
    # convert marketCap in foreing currency to EUR
    # load the exchange rates from yfinance

    all_currencies = filtered_assets["currency"].unique()
    fx_tickers = [
        f"EUR{currency}=X" for currency in all_currencies if currency != "EUR"
    ]
    # load last close values for the exchange rates
    if verbose:
        print(f"Fetching exchange rates for: {', '.join(fx_tickers)}")
    if fx_tickers:
        fx_data = yf.download(fx_tickers, period="5d", interval="1d", progress=verbose)[
            "Close"
        ]
        fx_data = fx_data.ffill()  # fill missing
        fx_last = fx_data.iloc[-1]

        # rename columns: keep only the foreign currency (after 'EUR' and before '=X')
        fx_last.index = [
            ticker.replace("EUR", "").replace("=X", "") for ticker in fx_last.index
        ]
        fx_last = fx_last.to_dict()
    else:
        fx_last = {}

    fx_last["EUR"] = 1.0
    # Convert marketCap to EUR
    filtered_assets["marketCapEUR"] = [
        cap / fx_last.get(currency, 1.0)
        for cap, currency in zip(
            filtered_assets["marketCap"], filtered_assets["currency"]
        )
    ]
    # Sort by marketCapEUR and select the top assets
    sorted_assets = filtered_assets.sort_values(by="marketCapEUR", ascending=False)
    selected_assets = sorted_assets.head(number_of_assets).copy()

    n_assets_final = len(selected_assets)
    if n_assets_final == 0:
        raise ValueError(
            "No assets found after filtering. Please adjust the filter criteria."
        )

    # Calculate weights based on marketCapEUR and diversification factor
    def entropy(w):
        return -np.sum(w * np.log(w + 1e-10)) / np.log(len(w))

    def calculate_weights(alpha, market_caps):
        weights = market_caps**alpha
        weights /= np.sum(weights)
        return weights

    # fix value of alpha to match the diversification factor (ie entropy) using scipy.optimize.fsolve

    def objective_function(alpha):
        weights = calculate_weights(alpha, selected_assets["marketCapEUR"])
        return entropy(weights) - diversification_factor

    alpha = sc.optimize.fsolve(objective_function, 0.5)[0]
    weights = calculate_weights(alpha, selected_assets["marketCapEUR"])

    selected_assets["Weight"] = weights

    if verbose:
        print(
            f"Selected {n_assets_final} assets for the value portfolio with a diversification factor of {diversification_factor}."
        )
        print(f"Alpha used for weight calculation: {alpha:.4f}")
        print(
            "Overall PE ratio of the portfolio: {:.2f}".format(
                np.sum(selected_assets["trailingPE"] * selected_assets["Weight"])
                / np.sum(selected_assets["Weight"])
            )
        )
        print(
            "Overall yield of the portfolio: {:.2f}".format(
                np.sum(
                    selected_assets["fiveYearAvgDividendYield"]
                    * selected_assets["Weight"]
                )
                / np.sum(selected_assets["Weight"])
            )
        )

    return selected_assets
