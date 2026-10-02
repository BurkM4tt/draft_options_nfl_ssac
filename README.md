# Draft Picks as Derivative Contracts

**Applying Options Pricing Theory to NFL Draft Pick Valuation and Trade Analysis**
Matt Burkett, Ph.D., University of Virginia (burkett@virginia.edu)
MIT Sloan Sports Analytics Conference 2027 (SSAC27), Football Track

Under the 2011 CBA's slotted rookie wage scale, a draft pick behaves like a call option: the right, but not the obligation, to employ a player's production at a predetermined cost for a fixed period. This repository values that option with an adapted Black-Scholes model and contains all data and code needed to reproduce every number in the SSAC27 abstract.

| Option input | Draft-pick analog |
|---|---|
| Underlying S | Expected four-year Approximate Value (AV), rolling window of +/- 5 picks |
| Strike K | The pick's actual signed four-year rookie contract, converted at $0.388M per AV |
| Time T | Four-year rookie contract |
| Volatility | Cross-sectional variation in four-year outcomes |

Estimation sample: 1,384 picks from the 2011 - 2016 draft classes. Out-of-sample test: 2017 - 2020 classes.

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline_v8.py            # reproduce everything from frozen_inputs/ (no network, ~10 s)
python run_pipeline_v8.py --fresh    # re-download sources and rebuild frozen_inputs/ first
```

Scripts can be run from any folder; paths resolve relative to the repository.

The default run verifies the SHA-256 checksum of every frozen input, recomputes all results, regenerates all figures, and finishes with an abstract verification block. That block asserts all 22 numbers printed in the abstract at their printed precision and fails the run if any does not match.

## Headline results (abstract v8)

| Quantity | Value |
|---|---|
| Calibration rate | $0.388M per four-year AV |
| Pick 1 moneyness | 0.49 |
| At-the-money crossing | Pick 33 |
| Out-of-the-money again | near Pick 170 (pipeline: 172) |
| Realized surplus, Round 1 / Rounds 2 - 3 / Round 7 | -7.7 AV (p < 1e-10) / +4.1 AV / -2.2 AV (p < 1e-8) |
| Robustness | R1 and R7 negative in all 8 specifications |
| Moneyness sign test | 73.0% in-sample, 70.3% out-of-sample |
| Johnson chart mispricing vs. OTM depth | r = -0.93 |

## Repository layout

```
run_pipeline_v8.py             single entry point
01_data_extraction_v8.py       Part 1: live download, sample filter, strike grid, anomaly guard
02_analytics_pipeline_v8.py    Part 2: calibration, Black-Scholes and Greeks, tests, figures, abstract checks
frozen_inputs/                 checksummed inputs to Part 2 (see MANIFEST.md)
outputs/                       result tables and full results_log.json
figures_all/                   13 individual charts
figures_for_paper/             6 multi-panel exhibits (figure5 = abstract Figure 1)
abstract/                      SSAC27 abstract v8: LaTeX source, compiled PDF, figure
CHANGELOG.md                   version history
```

## Data sources

All public, pulled only by Part 1:

- nflverse `nfldata`: `draft_picks.csv`, `rosters.csv` (seasonal AV), `draft_values.csv` (Johnson, Hill, Stuart, OTC, PFF charts)
- nflverse `nflverse-data` releases: Pro Football Reference draft picks (career AV, out-of-sample proxy) and Over The Cap `historical_contracts.parquet` (row-level contracts)

Sample filter: a pick enters the estimation sample if the player recorded at least one NFL roster season in experience years 1 - 4. This keeps 1,384 of 1,526 picks drafted 2011 - 2016; every excluded pick has zero four-year AV (asserted in Part 1).

## Why frozen inputs

Upstream sources are revised. Between the v7.3 snapshot and 2026-09-30, one Over The Cap contract record changed and moved several headline numbers in a live run (see CHANGELOG, v8). Part 2 therefore reads only from `frozen_inputs/`, and `--fresh` is a deliberate, guarded refresh: the strike-anomaly guard rejects incomplete contract histories, and the abstract verification block catches anything else that moves.

## Rebuilding the abstract

```bash
cd abstract
pdflatex S01_Draft_as_Options_Abstract_SSAC_v8.tex
```
