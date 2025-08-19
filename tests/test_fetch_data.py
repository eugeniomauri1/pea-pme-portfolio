import pandas as pd


from pea_pme_portfolio._fetch_data import (
    get_suffix,
    load_excel_from_euronext,
    get_tickers_from_isins,
    load_fundamentals_from_yf,
)


def test_get_suffix_mappings(capfd):
    markets = ["Dublin", "Lisbon", "UnknownCity"]
    result = get_suffix(markets, verbose=True)
    # Dublin -> .IR, Lisbon -> .LS, UnknownCity -> "NaN"
    assert result[0] == ".IR"
    assert result[1] == ".LS"
    assert result[2] == "NaN"


def test_load_excel_from_euronext_monkeypatched(monkeypatch):
    # Build fake HTML content where one link contains the dataset name
    fake_html = '<html><body><a href="https://example.com/liste_pea_pme_latest.xlsx">download</a></body></html>'
    fake_bytes = fake_html.encode("utf8")

    class FakeResponse:
        def __init__(self, b):
            self._b = b

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url: FakeResponse(fake_bytes))

    # monkeypatch pandas.read_excel to return a simple DataFrame with expected columns
    fake_df = pd.DataFrame(
        {
            "Société/Company": ["C1"],
            "CodeISIN/ISINCode": ["ISIN1"],
            "Marché/Market": ["Paris"],
            "Compartiment/Compartment": ["A"],
            "Pays d'incorporation/Country of Incorporation": ["FR"],
        }
    )

    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: fake_df)

    df = load_excel_from_euronext()

    # after the function, columns should be renamed to the mapping in the module
    assert "Company" in df.columns
    assert "ISIN" in df.columns
    assert df.iloc[0]["Company"] == "C1"


def test_get_tickers_from_isins_success(monkeypatch):
    # Prepare a fake successful requests.post returning status_code 200 and useful json
    class FakeResp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    # emulate two ISINs -> results list of dicts, each with "data":[{"ticker":"TICKER"}]
    def fake_post(url, json=None, headers=None):
        payload = [{"data": [{"ticker": "AAA"}]}, {"data": []}]
        return FakeResp(payload)

    monkeypatch.setattr("requests.post", fake_post)

    isins = ["ISIN1", "ISIN2"]
    res = get_tickers_from_isins(isins, max_retries=2, batch_size=2, verbose=False)

    # first returns 'AAA', second returned None
    assert res["ISIN1"] == "AAA"
    assert res["ISIN2"] is None


def test_load_fundamentals_from_yf_success(monkeypatch):
    # Fake yf.Ticker to return objects with .info dict
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker
            # return info dict with keys we will request
            self.info = {
                "trailingPE": 15.0,
                "fiveYearAvgDividendYield": 1.2,
                "marketCap": 1e9,
                "currency": "EUR",
            }

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    tickers = ["AAA", "BBB"]
    res = load_fundamentals_from_yf(
        tickers,
        fundamentals=["trailingPE", "marketCap"],
        max_retries=1,
        delay=0.0,
        verbose=False,
    )

    assert "AAA" in res and "BBB" in res
    assert res["AAA"]["trailingPE"] == 15.0
    assert res["AAA"]["marketCap"] == 1e9


def test_load_fundamentals_from_yf_not_found(monkeypatch):
    # Fake Ticker raising a ValueError containing '404' to simulate non-retriable error path
    class FakeTicker404:
        def __init__(self, ticker):
            self.ticker = ticker
            self.info = {}

        # accessing info will be empty and next the code raises a ValueError -> we simulate it in .info property access by giving empty info and then raising in fetch loop

    def fake_ticker_ctor(ticker):
        # simulate Ticker object whose usage raises ValueError containing 404 on first attempt
        class T:
            def __init__(self):
                self.info = {}

        return T()

    monkeypatch.setattr("yfinance.Ticker", fake_ticker_ctor)

    # Because the code detects "404" in exception text or HTTPError, we simulate raising an exception containing 404 by monkeypatching the internals:
    # However simpler: the code checks "if not info: raise ValueError(f'No info returned for {ticker}')"
    # That ValueError will be raised and then since "404" not in str(e) and not requests.HTTPError, it will retry; after max_retries it will return empty dict.
    # call with max_retries=1 to make it fast
    res = load_fundamentals_from_yf(
        ["NOTFOUND"],
        fundamentals=["trailingPE"],
        max_retries=1,
        delay=0.0,
        verbose=True,
    )

    # should have created an entry but empty dict
    assert "NOTFOUND" in res
    assert res["NOTFOUND"] == {}
