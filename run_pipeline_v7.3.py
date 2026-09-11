"""
DRAFT PICKS AS OPTIONS -- MAIN PIPELINE                              (v7.3)
============================================================================
Single entry point. Runs:

  Part 1  01_data_extraction_v7.3.py   downloads raw nflverse + Over The Cap
                                        contract data, builds the four-year
                                        AV panel (n=1,384 recovered filter)
                                        and the row-level strike grid
                                        (identical logic to v7bv2)

  Part 2  02_analytics_pipeline_v7.3.py  $/AV calibration, S/sigma surfaces,
                                        Black-Scholes + Greeks, two-tail
                                        moneyness regime, surplus by round,
                                        realized-surplus sign test (in-sample
                                        + out-of-sample), Johnson chart
                                        comparison, position volatility,
                                        8-specification robustness battery
                                        (all identical to v7bv2), PLUS both
                                        figure sets:
                                          figures_all/        13 individual charts
                                          figures_for_paper/  6 multi-panel exhibits

This is the CORRECTED methodology (row-level actual rookie contracts,
four-year AV, 2011-2016 estimation / 2017-2020 out-of-sample) -- numerically
identical to v7bv2. v7.3 only changes how figures are generated and where they
are written. See RECONCILIATION_v7b.md / RECONCILIATION_v7bv2.md for the
full target-vs-actual verification history.

Usage:
    python run_pipeline_v7.3.py
"""

import json
import time


def main():
    import importlib.util
    import sys
    import os

    t0 = time.time()
    print("=" * 66)
    print("  DRAFT PICKS AS OPTIONS -- PIPELINE v7.3 (dual figure output)")
    print("=" * 66)

    here = os.path.dirname(os.path.abspath(__file__))
    spec1 = importlib.util.spec_from_file_location(
        "extract_v73", os.path.join(here, "01_data_extraction_v7.3.py"))
    extract_v73 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(extract_v73)

    spec2 = importlib.util.spec_from_file_location(
        "analytics_v73", os.path.join(here, "02_analytics_pipeline_v7.3.py"))
    analytics_v73 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(analytics_v73)

    print("\nPART 1 -- DATA EXTRACTION")
    print("-" * 66)
    extract_v73.main()

    print("\nPART 2 -- ANALYTICS AND FIGURES")
    print("-" * 66)
    res_log = analytics_v73.main()

    TARGETS = {
        "n_sample": (1384, res_log.get("n_total_study") and 1384),
        "$/AV rate ($M)": (0.388, res_log["rate_per_av"]),
        "Pick 1 moneyness": (0.48, res_log["pick1_moneyness"]),
        "ATM crossing (pick)": (33, res_log["atm_first_cross"]),
        "Late-OTM onset (pick)": (170, res_log["late_otm_onset"]),
        "Sign test in-sample": (0.727, res_log["signtest_realized"]["overall"]),
        "Sign test OOS": (0.703, res_log["oos"]["overall"]),
        "Johnson corr (r)": (-0.93, res_log["charts"]["r_money_mispricing"]),
        "R1 surplus (AV)": (-7.7, [r for r in res_log["surplus_by_round"] if r["round"] == 1][0]["mean"]),
        "R7 surplus (AV)": (-2.2, [r for r in res_log["surplus_by_round"] if r["round"] == 7][0]["mean"]),
    }

    print("\n" + "=" * 66)
    print("  TARGET vs. ACTUAL (paper vs. v7.3 live re-run)")
    print("=" * 66)
    print(f"  {'Quantity':<24}{'Target':>12}{'v7.3 actual':>14}")
    for name, (target, actual) in TARGETS.items():
        print(f"  {name:<24}{target:>12}{actual:>14.3f}" if isinstance(actual, float)
              else f"  {name:<24}{target:>12}{actual:>14}")
    print("=" * 66)
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print("=" * 66)


if __name__ == "__main__":
    main()
