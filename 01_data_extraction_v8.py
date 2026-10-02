"""
DRAFT PICKS AS OPTIONS -- PART 1: DATA EXTRACTION                  (v8)
========================================================================
VERSION v8 (2026-09-30). Successor to v7.3. Sample construction and the
four-year AV panel are UNCHANGED. Three changes:

  1. STRIKE-ANOMALY GUARD (root-cause fix for live-data drift).
     A live re-run on 2026-09-30 found that Over The Cap's historical
     contracts no longer carry a clean four-year rookie record for
     2015 Pick 40 (Dorial Green-Beckham). The v7.3 fallback in
     four_year_cost() then summed only the cap charges booked while he
     was on the original team, giving $1.71M against ~$5.6M for his
     slot neighbors. Because 2015 is the reference year, that single
     cell produced a 4.0x moneyness spike at Pick 40, pulled the $/AV
     rate to $0.387M, and (through max-normalization) moved the Johnson
     mispricing correlation from -0.93 to -0.83.

     v8 screens every contract whose four-year cost came from the
     partial-history fallback (years != 4) against a log-interpolated
     reference built ONLY from clean four-year records in the same
     draft class. Anything below ANOMALY_RATIO (0.5) of that reference
     is set to missing and log-interpolated like any other unobserved
     cell. On current data this flags exactly one cell (2015 Pick 40,
     ratio 0.30); the next-lowest of the 59 fallback contracts is 0.69,
     so the threshold has wide margin on both sides. Flagged cells are
     written to frozen_inputs/k_anomalies.csv, and more than
     MAX_ANOMALIES flags is a hard failure (a large upstream change
     should stop the run, not be silently absorbed).

  2. FROZEN INPUTS. Outputs now go to ./frozen_inputs/ with a SHA-256
     MANIFEST.md. Part 2 verifies the manifest before computing
     anything. Raw downloads are cached in ./raw_cache/ (not committed).

  3. Header and file naming updated to v8.

Sole network-touching script. Downloads:
  draft_picks.csv              nflverse/nfldata       full draft history
  rosters.csv                  nflverse/nfldata       seasonal AV, 2006-2019
  draft_values.csv             nflverse/nfldata       Johnson/Hill/Stuart/OTC/PFF
  draft_picks.csv (PFR rel.)   nflverse/nflverse-data career/dr_av, OOS proxy
  historical_contracts.parquet Over The Cap via nflverse-data, row-level contracts

Outputs (./frozen_inputs, consumed only by 02_analytics_pipeline_v8.py):
  panel_4yr.csv            one row per drafted pick 2011-2020, av4, in_sample flag
  kgrid_contracted.csv     pick x year grid of four-year CONTRACTED strike ($M)
  kgrid_realized.csv       pick x year grid of four-year REALIZED cap charges ($M)
  kgrid_observed_mask.csv  True = directly observed contract, False = interpolated
  k_anomalies.csv          contracts rejected by the strike-anomaly guard
  draft_values.csv         pass-through copy for Part 2
  draft_picks_pfr.csv      pass-through copy for Part 2 (OOS proxy source)
  extraction_log.json      sample and grid diagnostics
  MANIFEST.md              SHA-256 checksums of every file above

Usage:
    python 01_data_extraction_v8.py        (normally invoked via run_pipeline_v8.py --fresh)
"""

import os
import io
import json
import hashlib

import numpy as np
import pandas as pd
import requests
import pyarrow.parquet as pq

# Paths anchored to this file's folder so the script works from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "frozen_inputs")   # frozen, checksummed outputs (committed)
CACHE_DIR = os.path.join(HERE, "raw_cache")     # raw downloads (not committed)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Strike-anomaly guard (see header, change 1)
ANOMALY_RATIO = 0.5
MAX_ANOMALIES = 5

FROZEN_FILES = [
    "panel_4yr.csv", "kgrid_contracted.csv", "kgrid_realized.csv",
    "kgrid_observed_mask.csv", "k_anomalies.csv", "draft_values.csv",
    "draft_picks_pfr.csv", "extraction_log.json",
]

STUDY = list(range(2011, 2017))          # estimation sample, 2011-2016
OOS = list(range(2017, 2021))            # out-of-sample, 2017-2020
STUDY_PLUS_OOS = list(range(2011, 2021))

NFLDATA_BASE = "https://github.com/nflverse/nfldata/raw/master/data"
PFR_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
CONTRACTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.parquet"


def _fetch_csv(url):
    r = requests.get(url, timeout=60, headers={"User-Agent": "python-requests"})
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _fetch_binary(url, path):
    r = requests.get(url, timeout=120, headers={"User-Agent": "python-requests"})
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def four_year_cost(years, value, season_history, draft_year):
    """Four-year cost of a rookie contract in $M.

    Clean 4-year rookie deals: use the recorded total value directly.
    Contracts recorded with the exercised 5th-year option folded in (or any
    non-4yr record): sum the year-by-year cap charges for contract years
    1-4 from season_history, recovering the four-year base.
    """
    if years == 4:
        return value
    if season_history is None or len(season_history) == 0:
        return np.nan
    d = pd.DataFrame(list(season_history))
    d = d[pd.to_numeric(d["year"], errors="coerce").notna()].copy()
    d["year"] = d["year"].astype(int)
    # season_history rows are duplicated (repeated verbatim, apparently once
    # per contract-history entry for the player); dedupe on (year, cap_number)
    # before summing or costs are inflated by the duplication factor.
    d = d.drop_duplicates(subset=["year", "cap_number"])
    yrs = d[(d.year >= draft_year) & (d.year <= draft_year + 3)]
    if len(yrs) == 0:
        return np.nan
    return float(yrs["cap_number"].sum())


def realized_cost(season_history, draft_year, present_years):
    """Realized (actually-booked) cap-charge cost: sums cap_number only for
    experience-years the player actually appeared on an NFL roster,
    approximating the abandonment option described in the paper's SS5.3
    ("the team holds a call plus a put, the right to abandon and stop
    paying"). A player cut after year 2, for instance, contributes only
    years 1-2 of cap charges here, not the full four-year contracted cost --
    unlike `four_year_cost()`, which always sums all four years regardless
    of whether the player was still on the team.

    `present_years` is the set of experience-years (1-4) in which the
    player had ANY roster appearance (the same roster-presence data used
    to build av4 in the sample filter), keyed by pfr_id upstream.
    """
    if season_history is None or len(season_history) == 0 or not present_years:
        return np.nan
    d = pd.DataFrame(list(season_history))
    d = d[pd.to_numeric(d["year"], errors="coerce").notna()].copy()
    d["year"] = d["year"].astype(int)
    d = d.drop_duplicates(subset=["year", "cap_number"])
    d["exp_year"] = d.year - draft_year + 1
    yrs = d[d.exp_year.isin(present_years)]
    if len(yrs) == 0:
        return np.nan
    return float(yrs["cap_number"].sum())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    """Checksum every frozen input so Part 2 can refuse drifted data."""
    lines = ["# frozen_inputs MANIFEST (pipeline v8)", "",
             "SHA-256 checksums of the frozen inputs consumed by "
             "02_analytics_pipeline_v8.py. Regenerate only via "
             "`python run_pipeline_v8.py --fresh`.", "",
             "| file | sha256 |", "|---|---|"]
    for fn in FROZEN_FILES:
        lines.append(f"| {fn} | {sha256(os.path.join(RAW_DIR, fn))} |")
    with open(os.path.join(RAW_DIR, "MANIFEST.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    wrote {RAW_DIR}/MANIFEST.md ({len(FROZEN_FILES)} checksums)")


def main():
    log = {}
    print("=" * 66)
    print("  PART 1: DATA EXTRACTION (v8)")
    print("=" * 66)

    # ---------------------------------------------------------- 1. LOAD
    print("\n[1] Loading raw sources")
    dp = _fetch_csv(f"{NFLDATA_BASE}/draft_picks.csv")
    ro = _fetch_csv(f"{NFLDATA_BASE}/rosters.csv")
    dv = _fetch_csv(f"{NFLDATA_BASE}/draft_values.csv")
    pfr = _fetch_csv(PFR_URL)
    con_path = f"{CACHE_DIR}/historical_contracts.parquet"
    _fetch_binary(CONTRACTS_URL, con_path)

    dp = dp.rename(columns={"round": "draft_round"})
    picks = dp[dp.season.isin(STUDY_PLUS_OOS)].copy()
    print(f"    draft_picks 2011-2020: {len(picks):,} picks")
    print(f"    rosters (seasonal AV): {len(ro):,} player-seasons, "
          f"{ro.season.min()}-{ro.season.max()}")
    print(f"    draft_values: {len(dv)} picks x {list(dv.columns)}")
    print(f"    PFR draft_picks (career AV): {len(pfr):,} picks")

    # ------------------------------------------------- 2. FOUR-YEAR AV PANEL
    print("\n[2] Four-year AV panel + recovered n=1,384 sample filter")
    r2 = ro.merge(
        picks[["pfr_id", "season"]].rename(columns={"season": "draft_year", "pfr_id": "playerid"}),
        on="playerid",
    )
    r2["exp"] = r2.season - r2.draft_year + 1
    av4 = r2[r2.exp.between(1, 4)].groupby("playerid")["av"].sum().rename("av4")
    played = set(av4.index)  # >=1 roster season in experience years 1-4

    # Per-player set of experience-years (1-4) with actual roster presence --
    # used below to build genuinely realized (abandonment-adjusted) contract
    # costs, distinct from the always-full-four-years contracted cost.
    present_years_map = (
        r2[r2.exp.between(1, 4)].groupby("playerid")["exp"]
        .apply(lambda s: set(int(x) for x in s.unique())).to_dict()
    )

    panel = picks.merge(av4, left_on="pfr_id", right_index=True, how="left")
    panel["av4"] = panel["av4"].fillna(0)
    panel["in_sample"] = panel.pfr_id.isin(played) & panel.season.isin(STUDY)

    samp = panel[panel.in_sample].copy()
    n_by_yr = samp.groupby("season").size().to_dict()
    log["n_sample"] = int(len(samp))
    log["n_total_study"] = int(len(picks[picks.season.isin(STUDY)]))
    log["n_by_year"] = {int(k): int(v) for k, v in n_by_yr.items()}
    log["total_av4"] = float(samp.av4.sum())
    print(f"    n_sample={len(samp)}  (of {log['n_total_study']} drafted, 2011-2016)")
    print(f"    by-year={n_by_yr}")
    print(f"    sum(av4)={samp.av4.sum():.0f}  (target: n=1384, sum_av4=14568)")

    # HARD GUARDS -- fail loudly rather than silently drift. Small AV
    # swings (+/-2) are tolerated since nflverse's upstream rosters.csv
    # snapshot can itself be revised by a point or two between runs (see
    # SPOT_CHECK_LOG.md's Von Miller 2012 AV finding: nflverse shows 17,
    # PFR's live site currently shows 18 for that one player-season) --
    # but n_sample must be exact and av4 must be within a couple points,
    # not silently different by dozens.
    assert len(samp) == 1384, (
        f"SAMPLE FILTER REGRESSION: n_sample={len(samp)}, expected exactly 1384. "
        f"This is a hard failure, not a warning -- the recovered filter no "
        f"longer reproduces the paper's estimation sample."
    )
    assert abs(samp.av4.sum() - 14568) <= 5, (
        f"SAMPLE FILTER REGRESSION: total av4={samp.av4.sum():.0f}, expected "
        f"~14568 (+/-5 tolerance for upstream nflverse AV revisions). "
        f"A larger gap means the four-year AV construction, not just data "
        f"drift, has changed."
    )
    assert (excl := panel[panel.season.isin(STUDY) & ~panel.in_sample]).av4.eq(0).all(), (
        f"SAMPLE FILTER REGRESSION: {int((excl.av4 != 0).sum())} excluded picks "
        f"have nonzero av4 -- the filter is dropping active players, not just "
        f"never-active ones."
    )

    # --------------------------------------------- 3. ACTUAL ROOKIE CONTRACT K
    print("\n[3] Row-level rookie contracts -> K grid (contracted + realized)")
    cols = ["player", "position", "year_signed", "years", "value", "apy",
            "draft_year", "draft_round", "draft_overall", "season_history"]
    con = pq.ParquetFile(con_path).read(columns=cols).to_pandas()

    rk = con[
        (con.draft_year.between(2011, 2020))
        & (con.year_signed == con.draft_year)
        & con.draft_overall.notna()
    ].copy()
    rk = rk[rk.years.notna()].copy()
    rk["draft_overall"] = rk.draft_overall.astype(int)
    rk["draft_year"] = rk.draft_year.astype(int)
    rk["years"] = rk.years.astype(int)

    rk["k4"] = rk.apply(
        lambda r: four_year_cost(r["years"], r["value"], r["season_history"], r["draft_year"]),
        axis=1,
    )
    rk.loc[rk.k4 <= 1.5, "k4"] = np.nan  # implausibly small -> missing

    # dedupe: prefer clean 4yr rows, then largest k4 (drops futures/PS stubs)
    rk["pref"] = (rk.years == 4).astype(int)
    rk = rk.sort_values(["pref", "k4"], ascending=False).drop_duplicates(
        ["draft_year", "draft_overall"]
    )

    # ---- v8 strike-anomaly guard ------------------------------------------
    # Screen fallback-derived (years != 4) strikes against a reference built
    # only from clean four-year records in the same draft class. Rookie slot
    # values are near-monotone in pick number, so a fallback value far below
    # its clean neighbors means the contract history is incomplete upstream,
    # not that the pick was cheap.
    anomalies = []
    for y, g in rk.groupby("draft_year"):
        clean = g[g.pref == 1].set_index("draft_overall").k4.reindex(range(1, 257))
        ref = np.exp(np.log(clean).interpolate(method="index", limit_direction="both"))
        fb = g[(g.pref == 0) & g.k4.notna()]
        for idx, r in fb.iterrows():
            ratio = r["k4"] / ref.loc[r["draft_overall"]]
            if ratio < ANOMALY_RATIO:
                anomalies.append(dict(draft_year=int(y), pick=int(r["draft_overall"]),
                                      player=r["player"], k4_fallback=round(float(r["k4"]), 3),
                                      k4_reference=round(float(ref.loc[r["draft_overall"]]), 3),
                                      ratio=round(float(ratio), 3)))
                rk.loc[idx, "k4"] = np.nan
    anom_df = pd.DataFrame(anomalies, columns=["draft_year", "pick", "player", "k4_fallback",
                                               "k4_reference", "ratio"])
    anom_df.to_csv(f"{RAW_DIR}/k_anomalies.csv", index=False)
    log["k_anomalies"] = anomalies
    print(f"    strike-anomaly guard: {len(anom_df)} fallback contract(s) below "
          f"{ANOMALY_RATIO:.0%} of clean-neighbor reference -> set to missing")
    for a in anomalies:
        print(f"      {a['draft_year']} Pick {a['pick']} {a['player']}: "
              f"${a['k4_fallback']:.2f}M vs ${a['k4_reference']:.2f}M reference (ratio {a['ratio']:.2f})")
    assert len(anom_df) <= MAX_ANOMALIES, (
        f"STRIKE-ANOMALY GUARD: {len(anom_df)} contracts flagged (max {MAX_ANOMALIES}). "
        f"Upstream contract data has changed substantially; inspect "
        f"frozen_inputs/k_anomalies.csv before accepting this extraction."
    )

    # ---- realized (abandonment-adjusted) cost -----------------------------
    # Join each contract row to the drafted player's pfr_id (via the same
    # draft_year/draft_overall key used everywhere else), then look up which
    # experience-years (1-4) that player actually had roster presence in.
    # realized_cost() sums cap charges only for those years -- a player cut
    # after year 2 contributes only years 1-2, not the full four-year
    # contracted cost. Players not matched to a pfr_id, or with no evidence
    # of early release (present all 4 years), fall back to the contracted
    # cost k4 (i.e., treated as fully paid).
    pid_map = picks.set_index(["season", "pick"])["pfr_id"].to_dict()
    rk["pfr_id"] = [pid_map.get((dy, do)) for dy, do in zip(rk.draft_year, rk.draft_overall)]

    def _realized_row(r):
        py = present_years_map.get(r["pfr_id"], {1, 2, 3, 4})
        rc = realized_cost(r["season_history"], r["draft_year"], py)
        return rc if pd.notna(rc) else r["k4"]

    rk["k4_real"] = rk.apply(_realized_row, axis=1)
    n_abandoned = int((rk.k4_real < rk.k4 - 0.05).sum())
    print(f"    realized cost < contracted cost for {n_abandoned} of {len(rk)} contracts "
          f"(evidence of early release / abandonment)")

    kgrid = rk.pivot_table(index="draft_overall", columns="draft_year", values="k4")
    kgrid = kgrid.reindex(range(1, 257))
    kgrid_real_raw = rk.pivot_table(index="draft_overall", columns="draft_year", values="k4_real")
    kgrid_real_raw = kgrid_real_raw.reindex(range(1, 257))

    n_cells = 256 * len(STUDY)
    n_missing = int(kgrid[STUDY].isna().sum().sum())
    # Save the pre-interpolation observed/missing mask (True = directly
    # observed contract, False = interpolated) for the robustness battery's
    # "directly-observed-contract subset" specification.
    observed_mask = kgrid.notna()
    for y in kgrid.columns:
        s = np.log(kgrid[y])
        kgrid[y] = np.exp(s.interpolate(method="index", limit_direction="both"))
    log["k_cells_interpolated"] = n_missing
    log["k_cells_total"] = n_cells
    print(f"    K grid built; {n_missing} of {n_cells} study-period cells interpolated "
          f"({100*(1-n_missing/n_cells):.0f}% directly observed)")
    print(f"    2015 K: P1=${kgrid.loc[1,2015]:.2f}M P32=${kgrid.loc[32,2015]:.2f}M "
          f"P64=${kgrid.loc[64,2015]:.2f}M P100=${kgrid.loc[100,2015]:.2f}M "
          f"P224=${kgrid.loc[224,2015]:.2f}M")

    # realized grid: interpolate missing cells the same way, then fall back
    # to the contracted grid's value for any cell realized-cost couldn't
    # reach (keeps the two grids on the same footing rather than
    # introducing extra missingness).
    kgrid_real = kgrid_real_raw.copy()
    for y in kgrid_real.columns:
        s = np.log(kgrid_real[y].where(kgrid_real[y] > 0))
        kgrid_real[y] = np.exp(s.interpolate(method="index", limit_direction="both"))
    kgrid_real = kgrid_real.fillna(kgrid)
    pct_of_contracted = float((kgrid_real[STUDY].values / kgrid[STUDY].values).mean())
    print(f"    Realized grid built; averages {pct_of_contracted:.1%} of contracted value "
          f"across study-period cells (<100% = evidence of abandonment)")

    # -------------------------------------------------- 4. WRITE OUTPUTS
    print("\n[4] Writing frozen intermediate outputs")
    panel.to_csv(f"{RAW_DIR}/panel_4yr.csv", index=False)
    kgrid.to_csv(f"{RAW_DIR}/kgrid_contracted.csv")
    kgrid_real.to_csv(f"{RAW_DIR}/kgrid_realized.csv")
    observed_mask.to_csv(f"{RAW_DIR}/kgrid_observed_mask.csv")
    dv.to_csv(f"{RAW_DIR}/draft_values.csv", index=False)
    pfr.to_csv(f"{RAW_DIR}/draft_picks_pfr.csv", index=False)

    with open(f"{RAW_DIR}/extraction_log.json", "w") as f:
        json.dump(log, f, indent=2)

    write_manifest()

    print(f"    wrote panel_4yr.csv, kgrid_contracted.csv, kgrid_realized.csv, "
          f"kgrid_observed_mask.csv, k_anomalies.csv, draft_values.csv, draft_picks_pfr.csv to {RAW_DIR}/")
    return log


if __name__ == "__main__":
    main()
