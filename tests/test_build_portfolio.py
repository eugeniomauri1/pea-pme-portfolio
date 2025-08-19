import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # do not require a display
import pytest

# import the module under test
from pea_pme_portfolio._build_portfolio import (
    get_value_portfolio,
    get_portfolio_plots,
)


class DummyYFinance:
    """
    A small fake for yfinance.download used in tests.
    It returns a dict-like with 'Close' / 'Adj Close' keys as expected by the code.
    """

    def __init__(self, close_df=None, adj_df=None):
        # close_df and adj_df are pandas DataFrames
        self.close_df = close_df
        self.adj_df = adj_df

    def __call__(self, *args, **kwargs):
        # behave like yf.download(...) returning dict-like object
        result = {}
        if self.close_df is not None:
            result["Close"] = self.close_df
        if self.adj_df is not None:
            result["Adj Close"] = self.adj_df
        return result


def test_get_value_portfolio_basic(monkeypatch, capsys):
    # Create sample assets (3 assets)
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "trailingPE": [10.0, 12.0, 8.0],
            "marketCap": [1e9, 2e9, 3e9],
            "fiveYearAvgDividendYield": [2.0, 1.5, 0.5],
            "currency": ["EUR", "USD", "EUR"],
            "sector": ["Tech", "Finance", "Utilities"],
            "Country": ["FR", "FR", "DE"],
        }
    )

    # fake fx download: only for USD -> create a single-row DataFrame with column 'EURUSD=X'
    fx_df = pd.DataFrame({"EURUSD=X": [1.2]}, index=[pd.Timestamp("2020-01-01")])

    # monkeypatch yfinance.download used in get_value_portfolio to return {'Close': fx_df}
    import pea_pme_portfolio._build_portfolio as bp

    monkeypatch.setattr(bp.yf, "download", DummyYFinance(close_df=fx_df))
    # monkeypatch scipy fsolve to a deterministic value
    monkeypatch.setattr(bp.sc.optimize, "fsolve", lambda func, x0: np.array([0.5]))

    # run function
    result = get_value_portfolio(
        all_assets=df,
        max_pe=20,
        min_pe=0,
        max_yield=5,
        min_yield=0,
        number_of_assets=2,
        diversification_factor=0.9,
        verbose=False,
    )

    # assertions
    assert isinstance(result, pd.DataFrame)
    assert "Weight" in result.columns
    assert len(result) == 2
    # weights sum to almost 1
    np.testing.assert_allclose(result["Weight"].sum(), 1.0, atol=1e-8)


def test_get_value_portfolio_missing_columns():
    # missing required column 'Ticker'
    df = pd.DataFrame(
        {
            "trailingPE": [10],
            "marketCap": [1e6],
            "fiveYearAvgDividendYield": [1],
            "currency": ["EUR"],
        }
    )
    with pytest.raises(ValueError, match="Missing required column"):
        get_value_portfolio(df)


def test_get_value_portfolio_no_assets_after_filter(monkeypatch):
    # DataFrame with marketCap non-positive so filtered out
    df = pd.DataFrame(
        {
            "Ticker": ["A"],
            "trailingPE": [100.0],
            "marketCap": [0],  # filtered
            "fiveYearAvgDividendYield": [2.0],
            "currency": ["EUR"],
        }
    )
    import pea_pme_portfolio._build_portfolio as bp

    # no fx tickers -> yf.download shouldn't be called, but provide anyway
    monkeypatch.setattr(bp.yf, "download", DummyYFinance(close_df=pd.DataFrame()))
    # monkeypatch fsolve
    monkeypatch.setattr(bp.sc.optimize, "fsolve", lambda func, x0: np.array([0.5]))

    with pytest.raises(ValueError, match="No assets found after filtering"):
        get_value_portfolio(df, max_pe=10, min_pe=0, verbose=False)


def make_portfolio_df():
    # portfolio with two tickers, all EUR (to avoid FX conversions)
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Weight": [0.6, 0.4],
            "sector": ["Tech", "Finance"],
            "Country": ["FR", "FR"],
            "currency": ["EUR", "EUR"],
        }
    ).set_index(pd.Index(["row1", "row2"]))  # index used in code


def test_get_portfolio_plots_success(monkeypatch):
    # Prepare portfolio
    portfolio = make_portfolio_df()

    # Create fake asset price data: two dates
    dates = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
    close_df = pd.DataFrame({"AAA": [10.0, 12.0], "BBB": [20.0, 22.0]}, index=dates)
    adj_df = pd.DataFrame({"AAA": [10.0, 13.0], "BBB": [20.0, 21.0]}, index=dates)

    # monkeypatch yf.download for asset data and no fx tickers (currencies are all EUR)
    import pea_pme_portfolio._build_portfolio as bp

    monkeypatch.setattr(
        bp.yf, "download", DummyYFinance(close_df=close_df, adj_df=adj_df)
    )

    fig1, fig2, perf_with_div, perf_without_div = get_portfolio_plots(
        portfolio, verbose=False
    )

    # basic checks
    assert hasattr(fig1, "axes")
    assert hasattr(fig2, "axes")
    # performance series are pandas Series (or something indexable)
    assert not perf_with_div.empty
    assert not perf_without_div.empty
    # first value is zero because they normalize by first element
    # use iloc[0] as it's Series
    assert perf_with_div.iloc[0] == 0 or np.isclose(perf_with_div.iloc[0], 0.0)
    assert perf_without_div.iloc[0] == 0 or np.isclose(perf_without_div.iloc[0], 0.0)


def test_get_portfolio_plots_missing_columns():
    # Build a DataFrame that truly *omits* the required 'sector' column
    df = pd.DataFrame(
        {
            "Ticker": ["AAA"],  # present
            "Weight": [1.0],  # present
            "Country": ["FR"],  # present
            "currency": ["EUR"],  # present
            # note: intentionally no "sector" column
        }
    )

    # Expect the function to raise for the missing column (explicit message includes the column name)
    with pytest.raises(ValueError, match=r"Missing required column: sector"):
        get_portfolio_plots(df, verbose=False)


def test_get_portfolio_plots_no_tickers():
    # Ensure checking for no tickers raises the explicit ValueError
    df = pd.DataFrame(
        {"Ticker": [], "Weight": [], "sector": [], "Country": [], "currency": []}
    )
    with pytest.raises(ValueError, match="No tickers found in the portfolio."):
        get_portfolio_plots(df, verbose=False)
