"""
DRAFT PICKS AS OPTIONS -- PART 2: ANALYTICS AND FIGURES               (v8)
========================================================================
VERSION v8 (2026-09-30). Successor to v7.3. Analysis sections [1]-[14]
and the figure architecture are UNCHANGED from v7.3. Changes:

  1. Reads exclusively from ./frozen_inputs/ (written by
     01_data_extraction_v8.py) and verifies every file against
     frozen_inputs/MANIFEST.md before computing anything. A checksum
     mismatch is a hard failure.

  2. HARD HEADLINE ASSERTS. v7.3 printed targets beside results but
     only asserted loose bounds (e.g. $/AV in [0.35, 0.42]), which is how
     a live run with one drifted contract passed silently at $0.387M /
     r = -0.83. v8 asserts every number that appears in the SSAC27
     abstract at the precision it is printed there (see
     verify_abstract_numbers). Any mismatch fails the run and lists
     every failing quantity.

  3. Writes outputs/abstract_table1.csv: the six
     positions shown in abstract Table 1, rounded exactly as displayed.

  Outputs:
    outputs/                  tables, logs, BS surface, abstract Table 1
    figures_all/              13 individual standalone charts
    figures_for_paper/        6 multi-panel paper exhibits (Figure 1-6)

EXHIBIT GROUPINGS (figures_for_paper/, matching the paper's prose):
  Figure 1  figure1_bs_valuation_vs_charts.png
  Figure 2  figure2_av_distribution_and_risk.png
  Figure 3  figure3_option_greeks.png
  Figure 4  figure4_position_volatility.png
  Figure 5  figure5_valuation_disagreement.png   (= abstract Figure 1)
  Figure 6  figure6_implied_vol_surface.png

Usage:
    python 02_analytics_pipeline_v8.py     (from any folder; normally via run_pipeline_v8.py)
"""

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, pearsonr, spearmanr, ttest_1samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# All paths are anchored to this file's folder, so the script works from any
# working directory (terminal, IDE, or run_pipeline_v8.py).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "frozen_inputs")
OUT = os.path.join(HERE, "outputs")
FIG_ALL = os.path.join(HERE, "figures_all")
FIG_PAPER = os.path.join(HERE, "figures_for_paper")
T = 4.0
REF_YEAR = 2015

# Annual 10-year Treasury yield (Fed H.15), draft year -> percent
RF_RATE = {2011: 2.79, 2012: 1.80, 2013: 2.35, 2014: 2.54, 2015: 2.14,
           2016: 1.84, 2017: 2.33, 2018: 2.91, 2019: 2.14, 2020: 0.89}

STUDY = list(range(2011, 2017))
OOS = list(range(2017, 2021))


def bs_call(S, K, T, r, sig):
    if sig <= 0 or T <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / max(K, 1e-3)) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def greeks(S, K, T, r, sig):
    d1 = (np.log(S / max(K, 1e-3)) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return dict(
        delta=norm.cdf(d1),
        gamma=norm.pdf(d1) / (S * sig * np.sqrt(T)),
        vega=S * norm.pdf(d1) * np.sqrt(T),
        theta=-(S * norm.pdf(d1) * sig) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
    )


def surplus_by_round(df, kcol, rate):
    out = []
    for rd, g in df.groupby("draft_round"):
        s = g.av4 - g[kcol] / rate
        t, p = ttest_1samp(s, 0)
        out.append(dict(round=int(rd), n=len(g), mean=float(s.mean()), sd=float(s.std()),
                         t=float(t), p=float(p), pct_neg=float((s < 0).mean())))
    return pd.DataFrame(out).sort_values("round")


def draw_four_year_av(ax, surf, colors):
    NAVY, ORANGE = colors["NAVY"], colors["ORANGE"]
    ax.fill_between(surf.pick, surf.p10, surf.p90, color=NAVY, alpha=0.15,
                     label="10th-90th percentile range")
    ax.plot(surf.pick, surf.S, color=NAVY, lw=2, label="Mean four-year AV")
    ax.plot(surf.pick, surf.med, color=ORANGE, lw=1.6, ls="--", label="Median four-year AV")
    ax.axvline(32.5, color="gray", ls=":", lw=1)
    ax.set_title("Four-Year AV by Draft Pick (2011-2016 classes)")
    ax.set_xlabel("Overall Draft Pick"); ax.set_ylabel("Four-Year Approximate Value (AV)")
    ax.legend(frameon=False)


def draw_risk_profile(ax, surf, colors):
    ORANGE, TEAL = colors["ORANGE"], colors["TEAL"]
    ax.plot(surf.pick, surf.bust * 100, color=ORANGE, lw=2, label="Bust rate (%, four-year AV <= 3)")
    ax2 = ax.twinx()
    ax2.plot(surf.pick, surf["std"], color=TEAL, lw=2, label="Four-year AV std. dev.")
    ax.axvline(32.5, color="gray", ls=":", lw=1)
    ax.set_title("Draft Risk Profile by Pick")
    ax.set_xlabel("Overall Draft Pick"); ax.set_ylabel("Bust Rate (%)"); ax2.set_ylabel("AV Std. Dev.")
    lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")


def draw_bs_vs_charts(ax, ch, colors):
    NAVY, TEAL, GOLD, ORANGE = colors["NAVY"], colors["TEAL"], colors["GOLD"], colors["ORANGE"]
    ax.plot(ch.pick, ch.bs_norm, color=ORANGE, lw=2.4, label="Black-Scholes (this paper)")
    ax.plot(ch.pick, ch.johnson_n, color=NAVY, lw=1.6, ls="--", label="Jimmy Johnson (1991)")
    ax.plot(ch.pick, ch.hill_n, color=TEAL, lw=1.4, ls="-.", label="Rich Hill (2012)")
    ax.plot(ch.pick, ch.stuart_n, color=GOLD, lw=1.4, ls=":", label="PFR Stuart Chart")
    ax.axvline(32.5, color="gray", ls=":", lw=1)
    ax.set_title("Pick Valuation: Black-Scholes vs. Traditional Charts (normalized 0-1000)")
    ax.set_xlabel("Overall Draft Pick"); ax.set_ylabel("Normalized Pick Value (0-1000)")
    ax.legend(frameon=False)


def draw_mispricing_map(ax, ch, colors):
    ORANGE, TEAL = colors["ORANGE"], colors["TEAL"]
    colors_arr = np.where(ch.mis_j >= 0, ORANGE, TEAL)
    ax.bar(ch.pick, ch.mis_j, color=colors_arr, width=1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Mispricing Map: Johnson Chart vs. Black-Scholes\n"
                 "(positive = Johnson overvalues; negative = Johnson undervalues)")
    ax.set_xlabel("Overall Draft Pick"); ax.set_ylabel("Johnson Norm. - BS Norm.")


def draw_greek(ax, surf, field, color, title, ylabel):
    ax.plot(surf.pick, surf[field], color=color, lw=2)
    ax.axvline(32.5, color="gray", ls=":", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Overall Draft Pick"); ax.set_ylabel(ylabel)


def draw_position_bar(ax, pos_df, colors):
    NAVY = colors["NAVY"]
    pos_sorted = pos_df.sort_values("cv", ascending=True)
    ax.barh(pos_sorted.position, pos_sorted.cv, color=NAVY)
    ax.axvline(pos_df.cv.median(), color="gray", ls="--", lw=1, label=f"Median CV = {pos_df.cv.median():.2f}")
    ax.set_title("Draft Pick Volatility by Position (higher CV = more uncertainty at draft time)")
    ax.set_xlabel("Coefficient of Variation (Std AV / Mean AV)")
    ax.legend(frameon=False)


def draw_position_bubble(fig, ax, pos_df):
    sizes = 300 * (pos_df.n / pos_df.n.max())
    sc = ax.scatter(pos_df["mean"], pos_df.sd, s=sizes, c=pos_df.cv, cmap="RdYlGn_r",
                     edgecolors="black", linewidths=0.6, alpha=0.9)
    for _, r in pos_df.iterrows():
        ax.annotate(r.position, (r["mean"], r.sd), xytext=(5, 3), textcoords="offset points", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label("CV (Implied Vol.)")
    ax.set_title("Mean vs. Volatility of Four-Year AV by Position\n(bubble size = sample size)")
    ax.set_xlabel("Mean Four-Year AV"); ax.set_ylabel("Std. Dev. of Four-Year AV")


def draw_iv_curve(ax, surf, colors):
    NAVY = colors["NAVY"]
    ax.plot(surf.pick, surf.sigma * 100, color=NAVY, lw=2)
    ax.fill_between(surf.pick, 0, surf.sigma * 100, color=NAVY, alpha=0.08)
    ax.axvline(32.5, color="gray", ls=":", lw=1)
    ax.set_title("NFL Draft Implied Volatility Curve\n(analog to the financial 'volatility smile')")
    ax.set_xlabel("Overall Draft Pick"); ax.set_ylabel("Annualized Implied Volatility (%)")


def draw_iv_by_bucket(ax, samp, study_years):
    buckets = [(1, 10), (11, 20), (21, 32), (33, 64), (65, 100), (101, 160), (161, 256)]
    bucket_labels = [f"{lo}-{hi}" for lo, hi in buckets]
    cmap = plt.get_cmap("Blues")
    for i, yr in enumerate(study_years):
        yr_df = samp[samp.season == yr]
        cvs = []
        for lo, hi in buckets:
            sub = yr_df[yr_df["pick"].between(lo, hi)]
            cvs.append(sub.av4.std() / max(sub.av4.mean(), 0.1) if len(sub) >= 3 else np.nan)
        ax.plot(bucket_labels, cvs, marker="o", color=cmap(0.35 + 0.55 * i / (len(study_years) - 1)), label=str(yr))
    ax.set_title("Implied Volatility by Pick Bucket, All Study-Period Draft Years")
    ax.set_xlabel("Pick Range (Draft Bucket)"); ax.set_ylabel("CV (Implied Volatility Proxy)")
    ax.legend(frameon=False, ncol=3, fontsize=9)


def draw_disagreement(ax, ch, colors):
    NAVY, TEAL, ORANGE = colors["NAVY"], colors["TEAL"], colors["ORANGE"]
    ax.plot(ch.pick, ch.bs_norm, color=ORANGE, lw=2.4, label="Black-Scholes Value")
    ax.plot(ch.pick, ch.johnson_n, color=NAVY, lw=1.8, ls="--", label="Johnson Chart")
    ax.fill_between(ch.pick, ch.bs_norm, ch.johnson_n,
                     where=(ch.bs_norm >= ch.johnson_n), color=TEAL, alpha=0.18,
                     label="Johnson undervalues late picks")
    for p in [1, 32, 64, 100]:
        row = ch[ch.pick == p].iloc[0]
        ax.annotate(f"Pick {p}\nBS={row.bs_norm:.0f}\nJJ={row.johnson_n:.0f}",
                    (p, row.bs_norm), xytext=(p + 12, row.bs_norm + 40),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("Black-Scholes vs. Johnson Chart: Where the Disagreement Lives")
    ax.set_xlabel("Overall Draft Pick"); ax.set_ylabel("Normalized Pick Value (0-1000)")
    ax.legend(frameon=False)


def generate_individual_figures(surf, samp, pos_df, ch, round_surplus, RATE, arc, kgrid, FIG):
    """Renders every chart as its own standalone PNG to figures_all/ -- no
    chart here is labeled "Figure N", and no two unrelated charts share a
    canvas. Each drawing routine (draw_*) is the SAME function called by
    generate_paper_exhibits() below for the composite multi-panel
    versions, so the individual and composite outputs can never disagree
    with each other -- there is exactly one place each chart's data,
    title, and styling are defined.
    """
    colors = dict(NAVY="#1B2A4A", TEAL="#1F7A72", GOLD="#C9A227", ORANGE="#D9702E")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
                          "axes.spines.top": False, "axes.spines.right": False,
                          "axes.grid": True, "grid.alpha": 0.25})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"    wrote {FIG}/{name}.png")

    print(f"\n[12] Generating 13 individual figures to {FIG}/ "
          f"(no combined exhibits, no figure numbering)")

    fig, ax = plt.subplots(figsize=(8, 5)); draw_four_year_av(ax, surf, colors)
    save(fig, "four_year_av_by_pick")

    fig, ax = plt.subplots(figsize=(8, 5)); draw_risk_profile(ax, surf, colors)
    save(fig, "draft_risk_profile_by_pick")

    fig, ax = plt.subplots(figsize=(8, 5)); draw_bs_vs_charts(ax, ch, colors)
    save(fig, "pick_valuation_bs_vs_charts")

    fig, ax = plt.subplots(figsize=(8, 5)); draw_mispricing_map(ax, ch, colors)
    save(fig, "mispricing_map_johnson_vs_bs")

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_greek(ax, surf, "delta", colors["NAVY"], "Delta: Sensitivity of Pick Value to Expected Production", "Delta")
    save(fig, "greek_delta_by_pick")

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_greek(ax, surf, "vega", colors["ORANGE"], "Vega: Value Sensitivity to Outcome Uncertainty", "Vega")
    save(fig, "greek_vega_by_pick")

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_greek(ax, surf, "sigma", colors["TEAL"], "Implied Volatility by Draft Position", "Annualized Volatility (sigma)")
    save(fig, "implied_volatility_by_pick")

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_greek(ax, surf, "theta", "#B03A2E", "Theta: Pick Value Decay as Uncertainty Resolves", "Theta")
    save(fig, "greek_theta_by_pick")

    fig, ax = plt.subplots(figsize=(8, 6)); draw_position_bar(ax, pos_df, colors)
    save(fig, "position_volatility_bar")

    fig, ax = plt.subplots(figsize=(7.5, 6)); draw_position_bubble(fig, ax, pos_df)
    save(fig, "position_mean_vs_volatility_bubble")

    fig, ax = plt.subplots(figsize=(8, 5)); draw_iv_curve(ax, surf, colors)
    save(fig, "implied_vol_curve_by_pick")

    fig, ax = plt.subplots(figsize=(8.5, 5.5)); draw_iv_by_bucket(ax, samp, STUDY)
    save(fig, "implied_vol_by_pick_bucket_by_year")

    fig, ax = plt.subplots(figsize=(9, 5.5)); draw_disagreement(ax, ch, colors)
    save(fig, "bs_vs_johnson_disagreement")

    print(f"[12] 13 individual figures written to {FIG}/ (no combined exhibits, no figure numbering)")


def generate_paper_exhibits(surf, samp, pos_df, ch, round_surplus, RATE, arc, kgrid, FIG):
    """Renders the 6 multi-panel exhibits actually cited in the paper's
    prose to figures_for_paper/, restoring the pre-v7b combined-exhibit
    convention. Each panel calls the exact same draw_* function used by
    generate_individual_figures() above -- same data objects, same
    titles, same styling -- so these composites and the standalone
    files in figures_all/ are generated from one code path per chart and
    cannot drift apart the way the old fig04_position_volatility.png did
    (that file predated the OL/K/P/LS/DL position-code exclusion and was
    never regenerated after the exclusion was added).

    Filenames are keyed to the paper's actual figure numbers, fixing the
    old fig01/fig02/.../fig04/fig06/fig08 scheme in which the numbering
    in the filenames did not match the "Figure N" numbering in the
    prose (e.g. the old fig02_bs_vs_charts.png was Figure 1).
    """
    colors = dict(NAVY="#1B2A4A", TEAL="#1F7A72", GOLD="#C9A227", ORANGE="#D9702E")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
                          "axes.spines.top": False, "axes.spines.right": False,
                          "axes.grid": True, "grid.alpha": 0.25})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"    wrote {FIG}/{name}.png")

    print(f"\n[12b] Generating 6 multi-panel paper exhibits to {FIG}/")

    # Figure 1: BS vs traditional charts (left) + mispricing map (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    draw_bs_vs_charts(axes[0], ch, colors)
    draw_mispricing_map(axes[1], ch, colors)
    save(fig, "figure1_bs_valuation_vs_charts")

    # Figure 2: four-year AV distribution (left) + risk profile (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    draw_four_year_av(axes[0], surf, colors)
    draw_risk_profile(axes[1], surf, colors)
    save(fig, "figure2_av_distribution_and_risk")

    # Figure 3: four Greeks, 2x2
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    draw_greek(axes[0, 0], surf, "delta", colors["NAVY"], "Delta: Sensitivity of Pick Value to Expected Production", "Delta")
    draw_greek(axes[0, 1], surf, "vega", colors["ORANGE"], "Vega: Value Sensitivity to Outcome Uncertainty", "Vega")
    draw_greek(axes[1, 0], surf, "sigma", colors["TEAL"], "Implied Volatility by Draft Position", "Annualized Volatility (sigma)")
    draw_greek(axes[1, 1], surf, "theta", "#B03A2E", "Theta: Pick Value Decay as Uncertainty Resolves", "Theta")
    save(fig, "figure3_option_greeks")

    # Figure 4: position volatility bar (left) + mean-vs-volatility bubble (right)
    # pos_df here is the SAME already-filtered (OL/K/P/LS/DL excluded)
    # DataFrame used everywhere else in the script -- no separate filter
    # to go stale.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    draw_position_bar(axes[0], pos_df, colors)
    draw_position_bubble(fig, axes[1], pos_df)
    save(fig, "figure4_position_volatility")

    # Figure 5: BS vs Johnson disagreement (single panel, same as v7bv2's
    # bs_vs_johnson_disagreement.png -- already a single chart, so the
    # "individual" and "for paper" versions are identical here)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    draw_disagreement(ax, ch, colors)
    save(fig, "figure5_valuation_disagreement")

    # Figure 6: implied vol curve (left) + implied vol by pick bucket (right)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    draw_iv_curve(axes[0], surf, colors)
    draw_iv_by_bucket(axes[1], samp, STUDY)
    save(fig, "figure6_implied_vol_surface")

    print(f"[12b] 6 multi-panel exhibits written to {FIG}/")



def verify_manifest():
    """Refuse to run on inputs that differ from the committed snapshot."""
    path = os.path.join(DATA, "MANIFEST.md")
    assert os.path.exists(path), (f"MISSING {path}. frozen_inputs/ should ship with the repository; "
                                  f"if it is absent, run `python run_pipeline_v8.py --fresh` to rebuild it.")
    rows = re.findall(r"^\| (\S+) \| ([0-9a-f]{64}) \|$", open(path).read(), flags=re.M)
    assert rows, f"{path} contains no checksums."
    bad = []
    for fn, digest in rows:
        h = hashlib.sha256(open(os.path.join(DATA, fn), "rb").read()).hexdigest()
        if h != digest:
            bad.append(fn)
    assert not bad, f"FROZEN-INPUT DRIFT: checksum mismatch for {bad}. Inputs changed since MANIFEST.md was written."
    print(f"    MANIFEST verified: {len(rows)} frozen inputs match their checksums")


# Every number printed in the SSAC27 abstract (v8), at the printed precision.
ABSTRACT_TARGETS = {
    "n_sample":                 ("{:d}",   1384),
    "total_av4":                ("{:.0f}", "14568"),
    "rate_per_av ($M)":         ("{:.3f}", "0.388"),
    "Pick 1 moneyness":         ("{:.2f}", "0.49"),
    "ATM crossing (Pick)":      ("{:d}",   33),
    "R1 mean surplus (AV)":     ("{:.1f}", "-7.7"),
    "R1 p < 1e-10":             ("{}",     "True"),
    "R2-3 pooled surplus (AV)": ("{:.1f}", "4.1"),
    "R7 mean surplus (AV)":     ("{:.1f}", "-2.2"),
    "R7 p < 1e-8":              ("{}",     "True"),
    "robustness specs":         ("{:d}",   8),
    "R1 & R7 negative in all":  ("{}",     "True"),
    "sign test in-sample":      ("{:.1%}", "73.0%"),
    "sign test OOS":            ("{:.1%}", "70.3%"),
    "Johnson r":                ("{:.2f}", "-0.93"),
}
LATE_OTM_RANGE = (165, 175)          # abstract: "near Pick 170"
ABSTRACT_TABLE1 = {                  # pos: (n, CV, bust %, mean surplus AV)
    "QB": (57, "1.23", "42", "-2.82"),
    "CB": (57, "1.07", "42", "-4.41"),
    "LB": (179, "0.99", "32", "+1.84"),
    "DT": (114, "0.90", "25", "+1.76"),
    "T":  (97, "0.74", "16", "+0.53"),
    "C":  (31, "0.70", "19", "+6.14"),
}


def verify_abstract_numbers(actual, late_onset, pos_df):
    fails = []
    print("\n" + "=" * 66)
    print("  ABSTRACT VERIFICATION (v8): every printed number, printed precision")
    print("=" * 66)
    for k, (fmt, target) in ABSTRACT_TARGETS.items():
        got = fmt.format(actual[k])
        ok = str(got) == str(target)
        print(f"  {'PASS' if ok else 'FAIL'}  {k:<26} abstract={target!s:<8} pipeline={got}")
        if not ok:
            fails.append(k)
    ok = LATE_OTM_RANGE[0] <= late_onset <= LATE_OTM_RANGE[1]
    print(f"  {'PASS' if ok else 'FAIL'}  {'late-OTM onset (Pick)':<26} abstract=~170    pipeline={late_onset}")
    if not ok:
        fails.append("late-OTM onset")
    tbl = []
    for pz, (n, cv, bust, surp) in ABSTRACT_TABLE1.items():
        r = pos_df[pos_df.position == pz].iloc[0]
        got = (int(r.n), f"{r.cv:.2f}", f"{100 * r.bust:.0f}", f"{r.surp:+.2f}")
        ok = got == (n, cv, bust, surp)
        print(f"  {'PASS' if ok else 'FAIL'}  Table 1 {pz:<18} abstract={(n, cv, bust, surp)} pipeline={got}")
        if not ok:
            fails.append(f"Table 1 {pz}")
        tbl.append(dict(position=pz, n=got[0], cv=got[1], bust_pct=got[2], mean_surplus_av=got[3]))
    pd.DataFrame(tbl).to_csv(f"{OUT}/abstract_table1.csv", index=False)
    print("=" * 66)
    assert not fails, f"ABSTRACT MISMATCH: {len(fails)} quantities differ from the abstract: {fails}"
    print(f"  All {len(ABSTRACT_TARGETS) + 1 + len(ABSTRACT_TABLE1)} abstract checks PASS")
    print("=" * 66)


def main():
    import os
    os.makedirs(FIG_ALL, exist_ok=True)
    os.makedirs(FIG_PAPER, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    verify_manifest()
    with open(f"{DATA}/extraction_log.json") as f:
        ext_log = json.load(f)
    log = {"n_total_study": ext_log["n_total_study"]}
    r_ref = RF_RATE[REF_YEAR] / 100

    panel = pd.read_csv(f"{DATA}/panel_4yr.csv")
    kgrid = pd.read_csv(f"{DATA}/kgrid_contracted.csv", index_col=0)
    kgrid.columns = kgrid.columns.astype(int)
    kreal = pd.read_csv(f"{DATA}/kgrid_realized.csv", index_col=0)
    kreal.columns = kreal.columns.astype(int)
    dv = pd.read_csv(f"{DATA}/draft_values.csv")
    pfr = pd.read_csv(f"{DATA}/draft_picks_pfr.csv")

    samp = panel[panel.in_sample].copy()
    full = panel[panel.season.isin(STUDY)].copy()
    for df in (samp, full):
        df["k_actual"] = df.apply(lambda r: kgrid.loc[r["pick"], r["season"]], axis=1)
        df["k_real"] = df.apply(lambda r: kreal.loc[r["pick"], r["season"]], axis=1)

    print("=" * 66)
    print("  PART 2: ANALYTICS AND FIGURES (v8)")
    print("=" * 66)

    # [1] endogenous $/AV --------------------------------------------------
    RATE = samp.k_actual.sum() / samp.av4.sum()
    log["rate_per_av"] = float(RATE)
    log["total_slot_M"] = float(samp.k_actual.sum())
    log["total_av4"] = float(samp.av4.sum())
    samp["surplus"] = samp.av4 - samp.k_actual / RATE
    full["surplus"] = full.av4 - full.k_actual / RATE
    print(f"\n[1] $/AV = ${RATE:.3f}M  (target: $0.388M)  "
          f"(slot ${samp.k_actual.sum():,.0f}M / {samp.av4.sum():.0f} AV)")
    assert 0.35 <= RATE <= 0.42, (
        f"RATE REGRESSION: $/AV={RATE:.3f}M, expected ~$0.388M (+/-~8% "
        f"tolerance). A larger gap usually means the K-grid construction "
        f"has a bug (e.g. the season_history duplication bug this pipeline "
        f"was built to fix -- see RECONCILIATION_v7b.md)."
    )

    # [2] S(p), sigma(p) surfaces -------------------------------------------
    rows = []
    for p in range(1, 257):
        sub = samp[samp["pick"].between(max(1, p - 5), min(256, p + 5))]
        if len(sub) < 3:
            continue
        mu, sd = sub.av4.mean(), sub.av4.std()
        rows.append(dict(pick=p, S=mu, std=sd, cv=sd / max(mu, .1),
                          med=sub.av4.median(), p10=sub.av4.quantile(.1),
                          p90=sub.av4.quantile(.9), bust=(sub.av4 <= 3).mean(), n=len(sub)))
    surf = pd.DataFrame(rows)
    surf["sigma"] = (surf.cv / np.sqrt(T)).clip(lower=.05)
    print(f"[2] S surface: P1={surf.loc[surf.pick==1,'S'].iat[0]:.1f} "
          f"P32={surf.loc[surf.pick==32,'S'].iat[0]:.1f} "
          f"P100={surf.loc[surf.pick==100,'S'].iat[0]:.1f}  (target: 31.8 / 19.2 / 10.2)")

    # [3] Black-Scholes + Greeks (reference year 2015, actual K) -----------
    surf["K_M"] = surf["pick"].map(kgrid[REF_YEAR])
    surf["K_av"] = surf.K_M / RATE
    surf["money"] = surf.S / surf.K_av
    surf["bs"] = surf.apply(lambda r: bs_call(r.S, r.K_av, T, r_ref, r.sigma), axis=1)
    g = surf.apply(lambda r: pd.Series(greeks(r.S, r.K_av, T, r_ref, r.sigma)), axis=1)
    surf = pd.concat([surf, g], axis=1)
    surf["intrinsic"] = (surf.S - surf.K_av).clip(lower=0)
    surf["timeprem"] = surf.bs - surf.intrinsic
    surf["bs_norm"] = surf.bs / surf.bs.max() * 1000

    atm = int(surf[surf.money >= 1.0].pick.min())
    atm_sustained = int(surf[(surf.money >= 1).astype(int).rolling(5).sum() == 5].pick.min())
    # late-OTM onset: first SUSTAINED run (10 consecutive picks) of money<1
    # after the ATM crossing -- the mid-round moneyness curve is noisy
    # enough that a short window (e.g. 5) catches transient dips around
    # pick 118-122 that reverse; a 10-pick window isolates the genuine
    # regime change and is stable across window widths 10-20 (all -> 172).
    RUN = 10
    late_regime = surf[surf.pick > atm_sustained].copy()
    late_regime["otm_run"] = (late_regime.money < 1).astype(int).rolling(RUN).sum()
    late_hits = late_regime[late_regime.otm_run == RUN]
    late_onset = int(late_hits.pick.min() - RUN + 1) if len(late_hits) else None
    peak = int(surf.loc[surf.bs.idxmax(), "pick"])
    log["atm_first_cross"] = atm
    log["late_otm_onset"] = late_onset
    log["bs_peak_pick"] = peak
    p1 = surf[surf.pick == 1].iloc[0]
    log["pick1_moneyness"] = float(p1.money)
    print(f"[3] Pick 1 moneyness={p1.money:.2f}x (target 0.49x)  |  "
          f"ATM crossing at pick {atm} (target 33)  |  "
          f"late-OTM onset ~pick {late_onset} (target ~170)  |  BS peak pick {peak}")

    # [4] surplus by round ---------------------------------------------------
    round_surplus = surplus_by_round(samp, "k_actual", RATE)
    log["surplus_by_round"] = round_surplus.round(3).to_dict("records")
    r1 = round_surplus[round_surplus["round"] == 1].iloc[0]
    r7 = round_surplus[round_surplus["round"] == 7].iloc[0]
    print(f"[4] R1 surplus={r1['mean']:.2f} AV (t={r1.t:.2f}, p={r1.p:.1e})  "
          f"(target -7.7, p<1e-10)  |  R7={r7['mean']:.2f} AV (t={r7.t:.2f}, p={r7.p:.1e}) "
          f"(target -2.2, p<1e-8)")

    # [5] realized-surplus sign test: in-sample -----------------------------
    pos_surplus = samp.groupby("pick").agg(realized_surplus=("surplus", "mean"),
                                            n=("surplus", "size")).reset_index()
    test = surf[["pick", "money"]].merge(pos_surplus, on="pick", how="inner")
    test["otm"] = test.money < 1
    acc_otm = float((test.loc[test.otm, "realized_surplus"] < 0).mean())
    acc_itm = float((test.loc[~test.otm, "realized_surplus"] > 0).mean())
    acc_all = float(((test.otm & (test.realized_surplus < 0)) |
                      (~test.otm & (test.realized_surplus >= 0))).mean())
    rho, p_rho = spearmanr(test.money, test.realized_surplus)
    r_pear, p_pear = pearsonr(test.money, test.realized_surplus)
    log["signtest_realized"] = dict(overall=acc_all, otm=acc_otm, itm=acc_itm,
                                     n_otm=int(test.otm.sum()), n_itm=int((~test.otm).sum()),
                                     rho_spearman=float(rho), p_rho=float(p_rho),
                                     r_pearson=float(r_pear))
    print(f"[5] In-sample sign test: {acc_all:.1%} overall "
          f"(OTM {acc_otm:.1%}, ITM {acc_itm:.1%})  rho={rho:.2f}  "
          f"(target: 72.7-73.0% overall, rho=0.63)")

    # [6] out-of-sample 2017-2020 via career-AV proxy -----------------------
    pfr_s = pfr[["season", "pick", "pfr_player_id", "dr_av", "w_av", "position"]].copy()
    val = samp.merge(pfr_s[pfr_s.season.isin(STUDY)], on=["season", "pick"], how="left")
    val["dr_av"] = val.dr_av.fillna(0)
    r_proxy, _ = pearsonr(val.dr_av, val.av4)
    b1, b0 = np.polyfit(val.dr_av, val.av4, 1)
    log["oos_proxy_calibration"] = dict(r_in_sample=float(r_proxy), slope=float(b1), intercept=float(b0))

    oos = pfr_s[pfr_s.season.isin(OOS)].copy()
    oos["dr_av"] = oos.dr_av.fillna(0)
    oos["av4_hat"] = b0 + b1 * oos.dr_av
    oos["k_actual"] = oos.apply(
        lambda r: kgrid.loc[r["pick"], r["season"]] if r["pick"] <= 256 and r["season"] in kgrid.columns else np.nan,
        axis=1,
    )
    oos = oos.dropna(subset=["k_actual"])
    oos["surplus_hat"] = oos.av4_hat - oos.k_actual / RATE

    # IMPORTANT (two distinct fixes vs. a naive OOS test):
    #  (1) The classification must come from the pre-existing, in-sample
    #      smoothed S(p) surface -- not from a moneyness ratio built out of
    #      the same OOS av4_hat/k_actual values used to compute
    #      surplus_hat. Using the latter is a tautology (S/K>=1 iff S-K>=0
    #      identically) and is the exact identity pitfall flagged in the
    #      paper's own \S5.4: "if expected surplus is computed from the
    #      same...curve that defines moneyness, agreement...is an identity,
    #      true by construction and evidentially empty."
    #  (2) The classification's STRIKE must reflect the OOS-period years
    #      (2017-2020), not the fixed REF_YEAR=2015 grid used for the
    #      in-sample Table 2 / Figure 3 surface -- rookie-contract slot
    #      cost rose with cap inflation across those years, so using the
    #      2015 K systematically overstates how many OOS picks are ITM.
    #      This matches the paper's own reported OOS ITM count (~59 of 256
    #      positions) and its "ATM crossing near pick 58" -- the 2015-K
    #      classification gives ~130 ITM positions and an ATM crossing at
    #      pick 33, which is the in-sample number, not a transferred one.
    K_oos_avg = kgrid[OOS].mean(axis=1)
    money_oos_class = (surf.set_index("pick")["S"] / (K_oos_avg / RATE)).rename("money").reset_index()

    oos_pos = oos.groupby("pick").agg(surplus_hat=("surplus_hat", "mean"),
                                       n=("surplus_hat", "size")).reset_index()
    oos_test = money_oos_class.merge(oos_pos, on="pick", how="inner")
    oos_test["otm"] = oos_test.money < 1
    oo_acc_otm = float((oos_test.loc[oos_test.otm, "surplus_hat"] < 0).mean())
    oo_acc_itm = float((oos_test.loc[~oos_test.otm, "surplus_hat"] > 0).mean())
    oo_acc_all = float(((oos_test.otm & (oos_test.surplus_hat < 0)) |
                         (~oos_test.otm & (oos_test.surplus_hat >= 0))).mean())
    r_oos, _ = pearsonr(oos_test.money, oos_test.surplus_hat)
    atm_cross_oos = int(oos_test[oos_test.money >= 1].pick.min()) if (oos_test.money >= 1).any() else None
    log["oos"] = dict(overall=oo_acc_all, otm=oo_acc_otm, itm=oo_acc_itm,
                       n_otm=int(oos_test.otm.sum()), n_itm=int((~oos_test.otm).sum()),
                       r_money_surplus=float(r_oos), atm_cross=atm_cross_oos)
    assert 0.4 < oo_acc_all < 0.95, (
        f"OOS TAUTOLOGY REGRESSION: overall accuracy={oo_acc_all:.1%} is "
        f"suspiciously extreme (near 100% or near 0%). This is the exact "
        f"fingerprint of the classification/scoring identity bug fixed in "
        f"this pipeline (see RECONCILIATION_v7b.md) -- check that the OOS "
        f"moneyness classification and the surplus it's scored against use "
        f"genuinely different bases."
    )
    print(f"[6] OOS (2017-2020): {oo_acc_all:.1%} overall (OTM {oo_acc_otm:.1%}, ITM {oo_acc_itm:.1%})  "
          f"proxy r_in_sample={r_proxy:.3f}  (paper (v7bv2): 70.3% overall, averaged 2017-2020 strike basis -- see footnote 4, RECONCILIATION_v7bv2_partial_corr.md)")

    # [7] chart comparison: Johnson mispricing correlation ------------------
    ch = surf[["pick", "bs_norm", "money"]].merge(dv, on="pick", how="left")
    for c in ["johnson", "hill", "stuart", "otc"]:
        ch[c + "_n"] = ch[c] / ch[c].max() * 1000
    ch["mis_j"] = ch.johnson_n - ch.bs_norm

    # The original 1991 Johnson chart only assigned values through pick 224;
    # the source data encodes picks 225-256 as johnson=0 rather than missing.
    # Treating a "no chart value" flag as a literal zero fabricates an
    # extreme mispricing signal at those 32 picks that has nothing to do
    # with the actual Johnson chart. Excluded from every Johnson-specific
    # comparison below (not from the BS/Black-Scholes-only calculations
    # elsewhere in this script, which don't depend on Johnson coverage).
    ch_jv = ch[ch.johnson > 0].copy()
    log["johnson_chart_coverage"] = dict(
        max_valid_pick=int(ch_jv.pick.max()), n_excluded_fabricated_zeros=int((ch.johnson == 0).sum())
    )

    r_mj, p_mj = pearsonr(ch_jv[ch_jv.pick <= 100].money, ch_jv[ch_jv.pick <= 100].mis_j)
    fd_m = ch_jv.sort_values("pick").money.diff().dropna()
    fd_j = ch_jv.sort_values("pick").mis_j.diff().dropna()
    r_fd, p_fd = pearsonr(fd_m, fd_j)
    log["charts"] = dict(
        jj_over_1_32=float(ch_jv[ch_jv.pick <= 32].mis_j.mean()),
        jj_under_65_100=float(-ch_jv[ch_jv.pick.between(65, 100)].mis_j.mean()),
        jj_under_101_200=float(-ch_jv[ch_jv.pick.between(101, 200)].mis_j.mean()),
        r_money_mispricing=float(r_mj), p=float(p_mj),
        r_first_differenced=float(r_fd), p_first_differenced=float(p_fd),
    )
    print(f"[7] Johnson mispricing corr (picks 1-100, valid coverage only): r={r_mj:.3f}  (paper: -0.93)")
    print(f"    First-differenced (picks 1-224, valid coverage only): r={r_fd:.3f}  (paper: -0.79 -- see "
          f"RECONCILIATION_v7bv2_partial_corr.md; corrected to this value)")

    # [8] partial correlation (controlling for pick number) -----------------
    # NOT reported in the paper. An exhaustive search (~50 specifications:
    # linear/log/sqrt/rank/polynomial(2-5) pick controls, OTM-only/ITM-only
    # subsets, Spearman, round fixed-effects, raw bs_norm-vs-johnson_n in
    # place of money-vs-mispricing, and excluding the 38 picks beyond the
    # original 1991 Johnson chart's actual coverage which are encoded as
    # zero rather than missing) never reproduced the earlier draft's stated
    # r=-0.42 -- results ranged from -0.19 to -0.95 depending on
    # specification, with no principled way to prefer one over another.
    # Per the project's verification standard (no number in the paper that
    # can't be verified by reference or analysis), this statistic is
    # therefore omitted from the paper entirely (v7bv2) rather than
    # replaced with a different unverified number. See
    # RECONCILIATION_v7bv2_partial_corr.md for the full search log. Kept
    # here only as an archived exploratory computation, not cited anywhere.
    partial_variants = {}
    for rng_name, rng in [("picks_1_100", ch[ch.pick <= 100]), ("picks_1_256", ch)]:
        res_money = rng.money - np.polyval(np.polyfit(rng.pick, rng.money, 1), rng.pick)
        res_mis = rng.mis_j - np.polyval(np.polyfit(rng.pick, rng.mis_j, 1), rng.pick)
        r_lin, p_lin = pearsonr(res_money, res_mis)
        fd_m = rng.money.diff().dropna()
        fd_j = rng.mis_j.diff().dropna()
        r_fd, p_fd = pearsonr(fd_m, fd_j)
        partial_variants[rng_name] = dict(linear_detrend=float(r_lin), first_diff=float(r_fd))
    log["charts"]["partial_correlation_variants_NOT_USED_IN_PAPER"] = partial_variants
    print(f"[8] Partial correlation (controlling for pick): NOT reported in paper -- "
          f"unresolved after exhaustive search, see RECONCILIATION_v7bv2_partial_corr.md. "
          f"(exploratory values archived in log only: linear-detrend picks1-100="
          f"{partial_variants['picks_1_100']['linear_detrend']:.2f})")

    # [9] position-level implied volatility ----------------------------------
    DROP = {"OL", "K", "P", "LS", "DL"}
    samp_pos = samp[~samp.position.isin(DROP)]
    pos = []
    for pz, g0 in samp_pos.groupby("position"):
        if len(g0) < 20:
            continue
        pos.append(dict(position=pz, n=len(g0), mean=float(g0.av4.mean()), sd=float(g0.av4.std()),
                         cv=float(g0.av4.std() / max(g0.av4.mean(), .1)),
                         bust=float((g0.av4 <= 3).mean()), surp=float(g0.surplus.mean())))
    pos_df = pd.DataFrame(pos).sort_values("cv", ascending=False)
    log["position_table"] = pos_df.round(3).to_dict("records")
    print(f"[9] Position CV range: {pos_df.cv.min():.2f}-{pos_df.cv.max():.2f}  "
          f"(QB CV={pos_df.loc[pos_df.position=='QB','cv'].iat[0]:.2f} if present)")

    # [10] Pick-1 moneyness arc across CBA years -----------------------------
    S1 = surf.loc[surf.pick == 1, "S"].iat[0]
    arc = []
    for y in [2011, 2015, 2019, 2020]:
        if y in kgrid.columns:
            v = kgrid.loc[1, y]
            arc.append(dict(year=y, slot_M=round(float(v), 2), money=round(float(S1 / (v / RATE)), 3)))
    log["pick1_arc"] = arc

    # [11] robustness battery: surplus by round across specifications -------
    specs = {}
    specs["baseline"] = (samp, "k_actual", RATE)
    rate_full = full.k_actual.sum() / full.av4.sum()
    specs["full_sample_1526"] = (full, "k_actual", rate_full)
    specs["rate_x0.8"] = (samp, "k_actual", RATE * 0.8)
    specs["rate_x1.2"] = (samp, "k_actual", RATE * 1.2)
    half1 = samp[samp.season.isin([2011, 2012, 2013])]
    half2 = samp[samp.season.isin([2014, 2015, 2016])]
    specs["2011_2013_half"] = (half1, "k_actual", half1.k_actual.sum() / half1.av4.sum())
    specs["2014_2016_half"] = (half2, "k_actual", half2.k_actual.sum() / half2.av4.sum())
    specs["realized_cost"] = (samp, "k_real", samp.k_real.sum() / samp.av4.sum())

    # 8th specification: directly-observed-contract subset -- restricts to
    # picks whose strike came from an actually-observed OTC contract record,
    # excluding the ~4-5% of study-period cells that were log-linearly
    # interpolated (see 01_data_extraction's observed_mask). This was named
    # explicitly in the paper's methodology text but was missing from the
    # battery entirely prior to this audit.
    obs_mask = pd.read_csv(f"{DATA}/kgrid_observed_mask.csv", index_col=0)
    obs_mask.columns = obs_mask.columns.astype(int)
    obs_mask_bool = obs_mask.astype(bool)
    samp["k_observed"] = samp.apply(
        lambda r: bool(obs_mask_bool.loc[r["pick"], r["season"]]), axis=1
    )
    observed_only = samp[samp.k_observed].copy()
    specs["directly_observed_only"] = (
        observed_only, "k_actual", observed_only.k_actual.sum() / observed_only.av4.sum()
    )
    print(f"    directly-observed-contract subset: n={len(observed_only)} of {len(samp)} "
          f"({len(observed_only)/len(samp):.1%})")

    battery = {}
    for name, (df_s, kcol, rate_s) in specs.items():
        rb = surplus_by_round(df_s, kcol, rate_s)
        battery[name] = dict(rate=float(rate_s), by_round=rb.round(3).to_dict("records"))
    log["robustness_battery"] = battery
    print(f"[11] Robustness battery: {len(specs)} specifications computed "
          f"(baseline + {len(specs)-1} alternates)")

    # check the paper's three stated regularities against the FULL battery
    non_baseline = [k for k in specs if k != "baseline"]
    r1_neg = [battery[k]["by_round"][0]["mean"] < 0 for k in non_baseline]
    r6_neg = [[r for r in battery[k]["by_round"] if r["round"] == 6][0]["mean"] < 0 for k in non_baseline]
    r6_sig = [[r for r in battery[k]["by_round"] if r["round"] == 6][0]["p"] < 0.05 for k in non_baseline]
    r7_neg = [battery[k]["by_round"][-1]["mean"] < 0 for k in non_baseline]
    log["battery_regularities"] = dict(
        r1_negative_in_all=all(r1_neg), n_specs_checked=len(non_baseline),
        r6_negative_count=sum(r6_neg), r6_significant_count=sum(r6_sig),
        r7_negative_in_all=all(r7_neg),
    )
    print(f"    R1 negative in {sum(r1_neg)}/{len(non_baseline)} non-baseline specs "
          f"(paper claims all)")
    print(f"    R6 negative in {sum(r6_neg)}/{len(non_baseline)}, significant in "
          f"{sum(r6_sig)}/{len(non_baseline)} (paper claims 7/7 negative, 5/7 significant)")
    print(f"    R7 negative in {sum(r7_neg)}/{len(non_baseline)} non-baseline specs "
          f"(paper claims all)")

    # [11b] Abandonment option (SS5.3): % of contracted value actually paid,
    # by round, and the realized-cost round-level surplus -- this is the
    # first time this specific computation has actually been run; prior
    # paper drafts stated round-level percentages without a corresponding
    # pipeline computation (kgrid_realized was previously a copy of
    # kgrid_contracted). See RECONCILIATION_v7bv2.md.
    samp["pct_paid"] = samp.k_real / samp.k_actual
    abandonment = samp.groupby("draft_round")["pct_paid"].agg(["mean", "median"]).round(3)
    log["abandonment_pct_paid_by_round"] = abandonment.reset_index().to_dict("records")
    realized_round_surplus = surplus_by_round(samp, "k_real", samp.k_real.sum() / samp.av4.sum())
    log["realized_cost_surplus_by_round"] = realized_round_surplus.round(4).to_dict("records")
    print(f"[11b] Abandonment option: R1 pays {abandonment.loc[1,'mean']:.1%} of contracted value, "
          f"R6={abandonment.loc[6,'mean']:.1%}, R7={abandonment.loc[7,'mean']:.1%}")

    # [13] Table 3: round-fixed-effects position regression -----------------
    # Never implemented in any prior pipeline version -- the paper's Table 3
    # coefficients were carried as text with no reproducing code. Added here;
    # found to closely match the paper's existing numbers (within ~0.1 AV on
    # every coefficient) once actually run.
    DROP_REG = {"OL", "K", "P", "LS", "DL"}
    reg_df = samp[~samp.position.isin(DROP_REG)].copy()
    reg_df["position"] = pd.Categorical(reg_df["position"])
    if "C" in reg_df.position.cat.categories:
        reg_df["position"] = reg_df["position"].cat.reorder_categories(
            ["C"] + [c for c in reg_df.position.cat.categories if c != "C"])
    reg_df["draft_round"] = pd.Categorical(reg_df["draft_round"])
    reg_model = smf.ols(
        'surplus ~ C(position, Treatment(reference="C")) + C(draft_round)', data=reg_df
    ).fit(cov_type="HC1")
    reg_table = (
        reg_model.params.filter(like="position").rename("coef").to_frame()
        .join(reg_model.pvalues.filter(like="position").rename("p"))
        .reset_index().rename(columns={"index": "term"})
    )
    reg_table["position"] = reg_table.term.str.extract(r"\[T\.(\w+)\]")
    reg_table = reg_table[["position", "coef", "p"]].sort_values("coef")
    log["table3_position_regression"] = reg_table.round(4).to_dict("records")
    reg_table.round(4).to_csv(f"{OUT}/table3_position_regression.csv", index=False)
    print(f"[13] Table 3 regression: n={int(reg_model.nobs)}, "
          f"CB coef={reg_table[reg_table.position=='CB'].coef.iat[0]:.2f} "
          f"(paper: -10.28)")

    # [14] SS7 trade case studies: per-trade-year Black-Scholes valuations --
    # Never implemented in any prior pipeline version. Uses the SAME S(p)
    # and sigma(p) from `surf` (computed once from the pooled 2011-2016
    # distribution) but each pick's ACTUAL contract-year K and that year's
    # risk-free rate, rather than the fixed REF_YEAR used for the general
    # Section 5.1 / Figure 1 surface. Found to match the paper's stated
    # case-study numbers exactly once actually computed.
    def bs_at(pick, year):
        S_, sig_ = surf.set_index("pick").loc[pick, ["S", "sigma"]]
        K_av_ = kgrid.loc[pick, year] / RATE
        r_ = RF_RATE.get(year, RF_RATE[REF_YEAR]) / 100
        return bs_call(S_, K_av_, T, r_, sig_), S_ / K_av_

    cases = {
        "goff_2016": dict(
            received=[(1, 2016), (113, 2016), (177, 2016)],
            surrendered=[(15, 2016), (43, 2016), (45, 2016), (76, 2016), (5, 2017), (100, 2017)],
        ),
        "rg3_2012": dict(
            received=[(2, 2012)],
            surrendered=[(6, 2012), (39, 2012), (22, 2013), (2, 2014)],
        ),
        "seahawks_2012": dict(
            received=[(15, 2012), (114, 2012), (172, 2012)],
            surrendered=[(12, 2012)],
        ),
    }
    case_results = {}
    for name, trade in cases.items():
        recv = [(p, y, *bs_at(p, y)) for p, y in trade["received"]]
        surr = [(p, y, *bs_at(p, y)) for p, y in trade["surrendered"]]
        tot_r = sum(x[2] for x in recv)
        tot_s = sum(x[2] for x in surr)
        case_results[name] = dict(
            received=[dict(pick=p, year=y, bs=round(bs, 2), moneyness=round(m, 2)) for p, y, bs, m in recv],
            surrendered=[dict(pick=p, year=y, bs=round(bs, 2), moneyness=round(m, 2)) for p, y, bs, m in surr],
            total_received=round(tot_r, 1), total_surrendered=round(tot_s, 1),
            net=round(tot_r - tot_s, 1),
        )
    log["case_studies"] = case_results
    print(f"[14] Case studies: Goff net={case_results['goff_2016']['net']} AV (paper: -21.6), "
          f"RG3 net={case_results['rg3_2012']['net']} AV (paper: -13.4), "
          f"Seahawks net={case_results['seahawks_2012']['net']} AV (paper: +6.6)")

    # ------------------------------------------------------- SAVE OUTPUTS
    surf.round(4).to_csv(f"{OUT}/bs_surface_actualK.csv", index=False)
    samp.to_csv(f"{OUT}/panel_4yr_actualK.csv", index=False)
    round_surplus.round(4).to_csv(f"{OUT}/table4_surplus_by_round.csv", index=False)
    pos_df.round(4).to_csv(f"{OUT}/table5_position_volatility.csv", index=False)

    with open(f"{OUT}/results_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # -------------------------------------------------- INDIVIDUAL FIGURES
    generate_individual_figures(surf, samp, pos_df, ch, round_surplus, RATE, arc, kgrid, FIG_ALL)
    generate_paper_exhibits(surf, samp, pos_df, ch, round_surplus, RATE, arc, kgrid, FIG_PAPER)

    print("\n" + "=" * 66)
    print("  HEADLINE SUMMARY")
    print("=" * 66)
    print(f"  n (sample)            {len(samp):,} of {log['n_total_study']:,}")
    print(f"  $/AV rate             ${RATE:.3f}M   (target $0.388M)")
    print(f"  Pick 1 moneyness      {p1.money:.2f}x  (target 0.49x)")
    print(f"  ATM crossing          pick {atm}   (target 33)")
    print(f"  Late-OTM onset        ~pick {late_onset}   (target ~170)")
    print(f"  Sign test in-sample   {acc_all:.1%}  (target 72.7-73.0%)")
    print(f"  Sign test OOS         {oo_acc_all:.1%}  (paper: 70.3%)")
    print(f"  Johnson corr (1-100)  r={r_mj:.2f}  (target -0.93)")
    print(f"  R1 / R7 surplus       {r1['mean']:.1f} / {r7['mean']:.1f} AV  (target -7.7 / -2.2)")
    print("=" * 66)

    r23 = samp[samp.draft_round.isin([2, 3])].surplus.mean()
    rb_all = battery
    actual = {
        "n_sample": len(samp),
        "total_av4": samp.av4.sum(),
        "rate_per_av ($M)": RATE,
        "Pick 1 moneyness": p1.money,
        "ATM crossing (Pick)": atm,
        "R1 mean surplus (AV)": r1["mean"],
        "R1 p < 1e-10": bool(r1.p < 1e-10),
        "R2-3 pooled surplus (AV)": r23,
        "R7 mean surplus (AV)": r7["mean"],
        "R7 p < 1e-8": bool(r7.p < 1e-8),
        "robustness specs": len(rb_all),
        "R1 & R7 negative in all": bool(all(r1_neg) and all(r7_neg)),
        "sign test in-sample": acc_all,
        "sign test OOS": oo_acc_all,
        "Johnson r": r_mj,
    }
    log["abstract_verification"] = {k: (float(v) if isinstance(v, (float, np.floating)) else
                                         (bool(v) if isinstance(v, (bool, np.bool_)) else int(v)))
                                     for k, v in actual.items()}
    with open(f"{OUT}/results_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    verify_abstract_numbers(actual, late_onset, pos_df)

    return log


if __name__ == "__main__":
    main()
