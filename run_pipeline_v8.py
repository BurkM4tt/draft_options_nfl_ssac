"""
DRAFT PICKS AS OPTIONS -- MAIN PIPELINE                              (v8)
============================================================================
Single entry point.

  python run_pipeline_v8.py            Part 2 only, on the committed
                                       frozen_inputs/ snapshot (default;
                                       no network, checksum-verified)
  python run_pipeline_v8.py --fresh    Part 1 (live nflverse + Over The Cap
                                       pull, rewrites frozen_inputs/ and
                                       MANIFEST.md), then Part 2

  Part 1  01_data_extraction_v8.py     four-year AV panel (n=1,384), row-level
                                       strike grid, v8 strike-anomaly guard
  Part 2  02_analytics_pipeline_v8.py  $/AV calibration, Black-Scholes + Greeks,
                                       two-tail moneyness, surplus by Round,
                                       sign tests (in-sample + OOS), Johnson
                                       comparison, position volatility,
                                       8-specification robustness battery,
                                       13 figures + 6 paper exhibits, and a
                                       hard check of every abstract number

Either mode fails loudly if any number in the SSAC27 abstract (v8) does not
reproduce at its printed precision.
"""

import argparse
import importlib.util
import os
import sys
import time


def _load(name, fname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Draft Picks as Options pipeline (v8)")
    ap.add_argument("--fresh", action="store_true",
                    help="re-download source data and rewrite frozen_inputs/ before analysis")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    print("=" * 66)
    print(f"  DRAFT PICKS AS OPTIONS -- PIPELINE v8  ({'fresh' if args.fresh else 'frozen'} inputs)")
    print("=" * 66)

    if args.fresh:
        print("\nPART 1 -- DATA EXTRACTION")
        print("-" * 66)
        _load("extract_v8", "01_data_extraction_v8.py").main()

    print("\nPART 2 -- ANALYTICS, FIGURES, ABSTRACT VERIFICATION")
    print("-" * 66)
    _load("analytics_v8", "02_analytics_pipeline_v8.py").main()

    print(f"\n  Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
