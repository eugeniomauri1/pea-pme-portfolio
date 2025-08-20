# <b> PEA-PME portfolio construction project </b>

## **Introduction**

**What is it?**

A [PEA-PME](https://www.service-public.fr/particuliers/vosdroits/F2385) (Plan d'Épargne en Actions destiné aux Petites et Moyennes Entreprises) is a French investment account designed to encourage investment in small and medium-sized enterprises (PME) and intermediate-sized enterprises (ETI). It offers tax advantages similar to the standard PEA (Plan d'Épargne en Actions) but is specifically focused on smaller businesses.

Key Features of a PEA-PME:
* Eligible Investments: Shares in SMEs and ETIs (both listed and unlisted), as well as certain investment funds (e.g., OPCVMs, FCPs) that allocate at least 75% of their assets to eligible SMEs/ETIs.
* Contribution Limit: The maximum contribution is 225,000 euros. However, if you also have a standard PEA, the combined cap for both accounts is 225,000 euros for the PEA-PME and 150,000 euros for the standard PEA.
* Tax Benefits: After five years, withdrawals are exempt from income tax (but still subject to social security contributions).

Company size conditions to be eligible:
* Being classified as a small or mid-sized enterprise (PME or ETI) as per EU definitions
    * Fewer than 5,000 employees
    * Either annual revenue below €1.5 billion or total assets below €2 billion

**Purpose of the code**

As of today, there isn't an ETF eligible for this kind of account. The goal of this code is to help you fetch data of PEA-PME eligible assets traded in the [EuroNext](https://connect2.euronext.com/en/media/169) exchanges and load fundamentals from [OpenFIGI](https://www.openfigi.com/) and [Yahoo Finance](https://finance.yahoo.com/) (through the dedicated python package [`yfinance`](https://ranaroussi.github.io/yfinance/)). Moreover, we propose a simple function to construct a value and high-yield portfolio.

---

## ⚠️ Disclaimer (please read)

This project is provided **for educational and informational purposes only**. It **is not** financial, investment, legal, or tax advice. Nothing in this repository should be interpreted as a recommendation to buy, sell, or hold any security or as a substitute for individualized professional advice.

By using this code and any outputs derived from it you acknowledge and accept that:
- The code is provided **“as-is”**, without warranties of any kind, express or implied.
- Results from backtests or historical analysis are **not** predictive of future performance.
- Data sources can be incomplete, delayed, or incorrect — you must verify data quality and the correctness of any computations before relying on them.
- You are solely responsible for any investment decisions and for consulting qualified professionals (a licensed financial advisor, tax advisor, or lawyer) about your personal circumstances.

If you intend to use this project for live trading or production systems, perform thorough validation, implement appropriate risk controls, and ensure compliance with data providers’ licensing and regulatory requirements. Use the code at your own risk.

---

## How to run the code
This project consists of a python package `pea_pme_portfolio`. To run it, you need to install the package and its dependencies. The suggested way to do this is to use [`uv`](https://github.com/astral-sh/uv?tab=readme-ov-file).

Assuming nothing is installed, you can follow the instructions below to install `uv` and run the code.

**Linux / MacOS**

Install `uv` using the following command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then use `uv` to install python:
```bash
uv python install 3.12
```

Finally, use the python environment to run the code (`uv` will install the package and its dependencies):
```python
import pea_pme_portfolio
opefigi_key = ... #your free openfigi api key. it will make the fetching faster
#the loader will take around 15 minutes
eligible_assets = data_loader(
    openfigi_api_key= opefigi_key,
    verbose = True,
    save_to_csv= True,
    kwargs = {
        'max_retries': 10,
        'batch_size': 30,
        'delay': 0.2
    })
value_portfolio = get_value_portfolio(eligible_assets,
        max_pe= 10.0,
        min_pe= 0.0,
        max_yield= 5,
        min_yield= 0.0,
        number_of_assets= 10,
        diversification_factor= 0.95,
        verbose= True)
```
**Windows**

The only difference is the command to install `uv`:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Then you can follow the same steps as above to install python and run the code.

## How to Contribute
**Prepare the local environment**
First, install `uv` using the command above. Then, create a new virtual environment using `uv`:
```bash
uv sync
```
(this step is not strictly necessary, it will be done automatically at the first `uv run` command)

**Pre-commit**

To install pre-commit hooks, run the following command:
```bash
uv run pre-commit install
```

**Testing**

To run the tests, use the following command:
```bash
uv run pytest --cov=src
```
