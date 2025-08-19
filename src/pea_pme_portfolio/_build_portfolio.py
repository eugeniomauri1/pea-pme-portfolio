import pandas as pd
import numpy as np
import yfinance as yf
import scipy as sc
import matplotlib.pyplot as plt


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


def get_portfolio_plots(
    portfolio: pd.DataFrame, verbose: bool = True
) -> tuple[plt.Figure, plt.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Create plots for the portfolio.

    Parameters:
        portfolio (pd.DataFrame): DataFrame containing the portfolio assets.
        verbose (bool): If True, print additional information.

    Returns:
        tuple[plt.Figure, plt.Figure]: Tuple containing the pie chart and bar chart figures.
    """
    # assert that the DataFrame has the required columns
    required_columns = ["Ticker", "Weight", "sector", "Country", "currency"]
    for col in required_columns:
        if col not in portfolio.columns:
            raise ValueError(f"Missing required column: {col}")

    # create a subplots with the pie charts of the weights per industry and country (weighted by the weights of the assets)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    # Pie chart for industry distribution
    industry_counts = portfolio.groupby("sector")["Weight"].sum()
    ax[0].pie(
        industry_counts,
        labels=industry_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax[0].set_title("Industry Distribution")
    # Pie chart for country distribution
    country_counts = portfolio.groupby("Country")["Weight"].sum()
    ax[1].pie(
        country_counts,
        labels=country_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax[1].set_title("Country Distribution")
    if verbose:
        print("Loading asset data from yfinance...")
    # load aseset data from yfinance
    tickers = portfolio["Ticker"].tolist()
    if len(tickers) > 0:
        asset_data_full = yf.download(
            tickers, period="max", progress=verbose, auto_adjust=False
        )
        asset_data_no_div = asset_data_full["Close"]
        asset_data = asset_data_full["Adj Close"]
        asset_data = asset_data.ffill().dropna()
        asset_data_no_div = asset_data_no_div.ffill().dropna()
    else:
        raise ValueError("No tickers found in the portfolio.")

    # convert asset performance in EUR

    all_currencies = portfolio["currency"].unique()
    fx_tickers = [
        f"EUR{currency}=X" for currency in all_currencies if currency != "EUR"
    ]
    tickers_not_eur = {
        portfolio.loc[_, "Ticker"]: portfolio.loc[_, "currency"]
        for _ in portfolio.index
        if portfolio.loc[_, "currency"] != "EUR"
    }
    # load last close values for the exchange rates
    if verbose:
        print(f"Fetching exchange rates for: {', '.join(fx_tickers)}")
    if fx_tickers:
        fx_data = yf.download(fx_tickers, period="max", progress=verbose)["Close"]
        fx_data = fx_data.ffill()  # fill missing
        fx_data.rename(
            columns={
                ticker: ticker.replace("EUR", "").replace("=X", "")
                for ticker in fx_data.columns
            },
            inplace=True,
        )
        asset_data_no_div, fx_data_ = asset_data_no_div.align(
            fx_data, join="outer", axis=0
        )
        for ticker, currency in tickers_not_eur.items():
            if currency in fx_data_.columns:
                asset_data_no_div[ticker] = (
                    asset_data_no_div[ticker] / fx_data_[currency]
                )
            else:
                raise ValueError(f"Currency {currency} not found in exchange rates.")
        asset_data, fx_data_ = asset_data.align(fx_data, join="outer", axis=0)
        for ticker, currency in tickers_not_eur.items():
            if currency in fx_data_.columns:
                asset_data[ticker] = asset_data[ticker] / fx_data_[currency]
            else:
                raise ValueError(f"Currency {currency} not found in exchange rates.")
    asset_data = asset_data.ffill().dropna()
    asset_data_no_div = asset_data_no_div.ffill().dropna()
    number_of_stocks = portfolio["Weight"] / asset_data.loc[:, tickers].iloc[-1].values
    portfolio_performance = (
        asset_data.loc[:, tickers] * number_of_stocks.values[np.newaxis, :]
    ).sum(axis=1)
    portfolio_performance = portfolio_performance.ffill().dropna()
    portfolio_performance = portfolio_performance / portfolio_performance.iloc[0] - 1

    portfolio_performance_no_div = (
        asset_data_no_div.loc[:, tickers] * number_of_stocks.values[np.newaxis, :]
    ).sum(axis=1)
    portfolio_performance_no_div = portfolio_performance_no_div.ffill().dropna()
    portfolio_performance_no_div = (
        portfolio_performance_no_div / portfolio_performance_no_div.iloc[0] - 1
    )

    # create a chart with the portfolio performance
    fig2, ax2 = plt.subplots(figsize=(12, 6), tight_layout=True)
    ax2.plot(
        portfolio_performance.index,
        portfolio_performance,
        label="With Dividends",
        color="orange",
    )
    ax2.plot(
        portfolio_performance_no_div.index,
        portfolio_performance_no_div,
        label="Without Dividends",
        color="violet",
    )
    ax2.set_title("Portfolio Performance")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Performance")
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_xlim(portfolio_performance.index.min(), portfolio_performance.index.max())
    # set y tickes as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    return fig, fig2, portfolio_performance, portfolio_performance_no_div
