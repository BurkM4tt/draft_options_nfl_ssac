# draft_options_nfl_ssac
# Draft Picks as Options Contracts

Data pipeline and analysis for the paper *"Draft Picks as Derivative Contracts: Applying Options Pricing Theory to NFL Draft Pick Valuation and Trade Analysis"* (Matt Burkett, Ph.D. — UVA Systems Engineering).

The pipeline runs end to end: data acquisition -> row-level rookie contract strikes -> Black-Scholes pricing -> Greeks -> mispricing analysis -> publication figures.

## Data Sources

- nflverse `draft_picks.csv` — full draft history; the pipeline uses the 2011-2020 window (2011-2016 estimation sample, 2017-2020 out-of-sample)
- nflverse `rosters.csv` — seasonal player AV, 2006-2019 (used to build each player's four-year AV)
- nflverse `draft_values.csv` — Johnson, Hill, and Stuart chart values, for comparison against the paper's Black-Scholes valuation
- nflverse-data `draft_picks.csv` (Pro Football Reference release) — career AV, used for the out-of-sample proxy
- Over The Cap `historical_contracts.parquet` (via nflverse-data) — row-level actual signed rookie contracts; this is the strike (K) source for every pick, not a modeled or fitted curve
- U.S. Federal Reserve H.15 — 10-year Treasury annual averages, hardcoded by draft year as the risk-free rate

All nflverse and Over The Cap data is downloaded automatically on first run; an internet connection is required. Nothing about team salary caps is hardcoded or otherwise used; the wage-scale strike comes directly from the signed contracts above.

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt` (pandas, numpy, requests, pyarrow, matplotlib, scipy, statsmodels)

`pyarrow` is required to read the Over The Cap contracts file (`.parquet` format) and is not optional.

## Setup

```bash
git clone <repo-url>
cd <repo-name>
chmod +x setup.sh
./setup.sh
source draft_options/bin/activate
```

`setup.sh` creates a virtual environment named `draft_options` and installs all dependencies. It is safe to re-run.

<details>
<summary>Manual setup (or Windows)</summary>

```bash
python3 -m venv draft_options
source draft_options/bin/activate        # Windows: draft_options\Scripts\activate
pip install -r requirements.txt
```

</details>

## Running the Pipeline

```bash
python run_pipeline_v7.3.py
```

This runs both stages in sequence:

1. `01_data_extraction_v7.3.py` — downloads the raw data above and writes frozen, checksum-stable intermediate CSVs to `data/` (career AV panel, contracted and realized strike grids)
2. `02_analytics_pipeline_v7.3.py` — reads only from `data/`, computes the $/AV calibration, Black-Scholes values and Greeks, the two-tail moneyness structure, round-level surplus, the sign-test validation (in-sample and out-of-sample), the Johnson-chart comparison, and the robustness battery, then writes figures

At the end it prints a target-vs-actual table comparing this run's numbers against the ones reported in the paper (n=1,384, $0.388M/AV, pick-33 ATM crossing, round 1/7 surplus, sign-test accuracy, Johnson correlation) so you can confirm your run reproduced the paper's results.

Outputs:

- `data/` — frozen intermediate datasets (panel, strike grids, extraction log)
- `figures_all/` — 13 individual standalone charts, one topic per file
- `figures_for_paper/` — 6 multi-panel exhibits matching the paper's Figure 1-6 numbering

## Repository Structure

```
├── 01_data_extraction_v7.3.py     # Part 1: downloads data, builds the AV panel + strike grids
├── 02_analytics_pipeline_v7.3.py  # Part 2: pricing, Greeks, validation, figures
├── run_pipeline_v7.3.py           # runs both parts, prints target-vs-actual summary
├── requirements.txt               # Python dependencies
├── setup.sh                       # environment setup script
├── data/                          # created at runtime
├── figures_all/                   # created at runtime
└── figures_for_paper/             # created at runtime
```
