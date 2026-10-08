"""Numbers quoted in the text of the paper that the other analysis scripts do not print.

Sections follow the evaluation of the paper; all shares apply the service floor S_min = 0.01 per slot.
The remaining numbers come from analyze_b5.py (Table II, joint vs sequential), validate_model.py
(model validation), analyze_spaceground.py, analyze_weather.py, analyze_weather_outages.py,
satellite_stats.py, and link_lifetime.py (space-ground), and plot_figures.py (figures and tables).

Example:
  .venv/bin/python experiments/paper_numbers.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser import topologies  # noqa: E402
from fiser.heuristic import mincost_disjoint_paths  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.paths import path_metrics  # noqa: E402
from fiser.physics import PhysParams  # noqa: E402
from fiser.report import apply_floor  # noqa: E402
from fiser.satellite import R_E  # noqa: E402
from fiser.spaceground import SatParams, downlink_eta  # noqa: E402

J = ROOT / "results" / "journal"
S_MIN = 0.01
TOPOS = ("nsfnet", "cost239", "usnet")
KEY = ["topology", "F_th", "s", "d"]


def load(run, s_min=S_MIN):
    df = pd.concat([pd.read_csv(J / run / f"{t}.csv") for t in TOPOS])
    df = apply_floor(df, s_min)
    df["F_th"] = df.F_th.round(2)
    return df


def served(df):
    return df.groupby(["topology", "F_th", "method"]).served.sum().unstack("method")


def mean_worst(df):
    return df.groupby(["topology", "F_th", "method"]).worst.mean().unstack("method")


def argmax(s):
    return f"{s.max():.3f} at {s.idxmax()}"


def setup():
    print("=== VI-A Setup")
    P = PhysParams()
    F = []
    for t in TOPOS:
        G = build_network(topologies.load(t)["edges"], P, max_link_km=100)
        F += [(1 + 3 * e["w0"] * math.exp(-e["L"] / P.l_dec_km)) / 4 for *_, e in G.edges(data=True)]
    print(f"link fidelities at level 0: {min(F):.3f}-{max(F):.3f}")
    S = SatParams()

    def slant(el):
        e = math.radians(el)
        return math.sqrt((R_E + S.altitude_km) ** 2 - (R_E * math.cos(e)) ** 2) - R_E * math.sin(e)
    db = [-20 * math.log10(downlink_eta(el, slant(el), S)) for el in (90.0, S.el_min_deg)]
    print(f"two-downlink loss: {db[0]:.1f} dB (both at zenith) to {db[1]:.1f} dB (both at {S.el_min_deg:.0f} deg)")


def single_failure():
    print("\n=== VI-B Single-failure protection")
    g = load("ground")
    sv, mw = served(g), mean_worst(g)
    pairs = g[g.method == "FiSER"].groupby(["topology", "F_th"]).size()
    share = (100 * sv.div(pairs, axis=0)).round(0)
    print("served share [%] at F_th = 0.8 (B1, FiSER):",
          {t: (share.loc[(t, 0.8), "B1"], share.loc[(t, 0.8), "FiSER"]) for t in sv.index.levels[0]})
    print(f"NSFNET at 0.85: B1 leaves {100 * (1 - sv.loc[('NSFNET', 0.85), 'B1'] / pairs['NSFNET', 0.85]):.0f}% "
          "of the requests unusable")
    for F in (0.8, 0.85, 0.9):
        r = (sv.FiSER / sv.B1).xs(F, level="F_th")
        print(f"FiSER/B1 served at {F}: " + ", ".join(f"{t} {v:.1f}x" for t, v in r.items()))
    hi = sv[sv.index.get_level_values("F_th") >= 0.85]
    print(f"B4 - B1 served at F_th >= 0.85: at most {int((hi.B4 - hi.B1).max())}")
    for b in ("B2", "B3"):
        print(f"FiSER/{b} - 1 served: max {argmax(sv.FiSER / sv[b] - 1)}")
    print(f"mean worst-case success FiSER/B3 - 1: max {argmax(mw.FiSER / mw.B3 - 1)}")
    b7 = load("ground_B7")
    sv7, mw7 = served(b7), mean_worst(b7)
    print(f"B7: FiSER/B7 - 1 served max {argmax(sv7.FiSER / sv7.B7 - 1)}; "
          f"mean worst-case 1 - B7/FiSER max {argmax(1 - mw7.B7 / mw7.FiSER)}")
    w = g.pivot_table(index=KEY, columns="method", values="worst")
    for b in ("B2", "B3"):
        both = (w.FiSER > 0) & (w[b] > 0)
        med = (w.FiSER / w[b])[both].groupby(level=["topology", "F_th"]).median()
        at8 = w.xs(0.8, level="F_th")
        beats = (at8.FiSER > 1.01 * at8[b]).groupby(level="topology").sum()
        top = (at8.FiSER / at8[b])[(at8.FiSER > 0) & (at8[b] > 0)].max()
        print(f"vs {b}: median per-pair ratio {med.min():.3f}-{med.max():.3f}; at 0.8 FiSER better by >1% on "
              f"{beats.min()}-{beats.max()} pairs per network (largest ratio {top:.1f}x)")
    for run, F, meths in (("ground_sens_lam0977", 0.8, ("FiSER", "B5", "B3", "B2", "B1")),
                          ("ground_sens_ldec2000", 0.9, ("FiSER", "B5", "B3", "B2", "B1")),
                          ("ground_sens_ldecinf", 0.9, ("B1",))):
        s = served(load(run)).loc[("NSFNET", F)]
        print(f"{run}, NSFNET at {F}: " + ", ".join(f"{m} {int(s[m])}" for m in meths))
    sweep = [("ground", 0.0), ("ground", 0.001), ("ground", 0.01), ("ground", 0.05),
             ("ground_sens_M25", 0.01), ("ground_sens_M100", 0.01),
             ("ground_sens_scale50", 0.01), ("ground_sens_scale150", 0.01)]
    rows = []
    for run, s_min in sweep:
        s = served(load(run, s_min))
        if run == "ground":
            s = s.join(served(load("ground_B7", s_min))[["B7"]])
        rows.append(s[s.index.get_level_values("F_th").isin((0.8, 0.85))])
    cases = pd.concat(rows)
    ratio = cases.FiSER / cases.B1.clip(lower=1)
    print(f"sweep ({len(cases)} cases): FiSER/B1 {ratio.min():.1f}-{ratio.max():.1f}x (median {ratio.median():.1f}x); "
          "FiSER serves more than " + ", ".join(f"{b} by up to {100 * (cases.FiSER / cases[b] - 1).max():.0f}%"
                                               for b in ("B5", "B2", "B7", "B3"))
          + f"; never worse: {bool((cases[['B1', 'B2', 'B3', 'B5', 'B7']].le(cases.FiSER, axis=0)).all().all())}")
    pos = pd.read_csv(J / "ground" / "price_of_survivability.csv")
    r = pos[(pos.topology == "NSFNET") & (pos.F_th.round(2) == 0.85)].iloc[0]
    print(f"price of protection, NSFNET at 0.85: {r.single_feasible} pairs with a single path, {r.survivable} "
          f"protected, worst-case/single-path success {100 * r.geo_worst_over_single:.0f}% (geometric mean)")


def joint_vs_sequential():
    print("\n=== VI-C Joint vs sequential selection (Table II: analyze_b5.py)")
    g = load("ground")
    sv, mw = served(g), mean_worst(g)
    u = sv.loc["USNET"].loc[0.7:0.85]
    print("USNET FiSER - B5 at F_th 0.70-0.85: " + ", ".join(f"{F}: +{int(v)} ({100 * v / b:.0f}%)" for F, v, b in
                                                           zip(u.index, u.FiSER - u.B5, u.B5)))
    print(f"USNET mean worst-case 1 - B5/FiSER at 0.70-0.85: max {100 * (1 - mw.B5 / mw.FiSER).loc['USNET'].loc[0.7:0.85].max():.0f}%")
    m = pd.read_csv(J / "b5_study" / "mesh.csv.gz")
    m = m[m.degree == 3.5]
    agg = m.groupby(["n", "seed", "s", "d"]).agg(f=("fiser_served", "sum"), b=("seq_served", "sum"))
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(10000):
        smp = agg.sample(len(agg), replace=True, random_state=int(rng.integers(1 << 31)))
        boot.append(smp.f.sum() / smp.b.sum() - 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"meshes: FiSER {int(agg.f.sum())} vs B5 {int(agg.b.sum())} (+{100 * (agg.f.sum() / agg.b.sum() - 1):.1f}%; "
          f"95% bootstrap CI over SD pairs +{100 * lo:.1f}% to +{100 * hi:.1f}%)")


def two_failures():
    print("\n=== VI-D Two-failure protection")
    a = load("ground_alpha2")
    pooled = served(a).groupby(level="topology").sum()
    for t, r in pooled.iterrows():
        print(f"{t} (F_th 0.70-0.90): FiSER {int(r.FiSER)}, B2 {int(r.B2)} (+{100 * (r.FiSER / r.B2 - 1):.0f}%), "
              f"B5 {int(r.B5)} (+{100 * (r.FiSER / r.B5 - 1):.0f}%), B1 {int(r.B1)} ({r.FiSER / r.B1:.1f}x)")
    w = a.pivot_table(index=KEY, columns="method", values="worst")
    c = w.xs("COST239", level="topology")
    both = (c.FiSER > 0) & (c.B5 > 0)
    r = (c.B5 / c.FiSER)[both]
    print(f"COST239, both served: {int(both.sum())}; B5's mean worst-case success {100 * (1 - r.mean()):.0f}% lower, "
          f"below half of FiSER's on {int((r < 0.5).sum())}")
    s = pd.read_csv(J / "b5_study" / "alpha2.csv.gz")
    s = s[s.topology == "nsfnet"]
    lost = s.fiser_served & ~s.seq_served
    print(f"NSFNET sequential losses: {int(lost.sum())}, of which no third disjoint path: "
          f"{int((lost & (s.seq_reason == 'trap')).sum())}")


def coherence():
    print("\n=== VI-E Memory coherence (validation: validate_model.py, tests)")
    runs = {"inf": "ground", "1": "ground_T1", "0.1": "ground_T0.1", "0.01": "ground_T0.01"}
    sv = {T: served(load(run)) for T, run in runs.items()}
    base = sv["inf"].FiSER
    d1 = base - sv["1"].FiSER
    print(f"T = 1 s: FiSER loses at most {int(d1.xs(0.8, level='F_th').max())} pairs per network at 0.8 and "
          f"{int(d1.max())} at any threshold")
    rel = 100 * (base - sv["0.1"].FiSER) / base
    for F in (0.8, 0.85):
        print(f"T = 0.1 s at {F}: FiSER loses " + ", ".join(f"{t} {v:.0f}%" for t, v in rel.xs(F, level='F_th').items()))
    s = sv["0.1"].xs(0.8, level="F_th")
    print("T = 0.1 s at 0.8: FiSER/B1 " + ", ".join(f"{t} {v:.1f}x" for t, v in (s.FiSER / s.B1).items())
          + f"; FiSER >= B2, B3 everywhere: {bool((sv['0.1'].FiSER >= sv['0.1'][['B2', 'B3']].max(axis=1)).all())}")


def heuristic():
    print("\n=== VI-F Heuristic and scalability")
    tot_served = tot_nonopt = 0
    for run in ("ground", "ground_alpha2"):
        df = load(run)
        f = df[df.method == "FiSER"].set_index(KEY)
        h = df[df.method == "H"].set_index(KEY).reindex(f.index)
        opt = (h.worst >= f.worst * (1 - 1e-6)) & f.served
        ratios = (h.worst / f.worst)[f.served & ~opt]
        print(f"{run}: H optimal on {int(opt.sum())} of {int(f.served.sum())} served requests"
              + (f" (others at {', '.join(f'{v:.2f}' for v in ratios)} of the optimum)" if len(ratios) else "")
              + f"; runtime H/MILP {h.runtime.sum() / f.runtime.sum():.2f}")
        tot_served += int(f.served.sum())
        tot_nonopt += int((f.served & ~opt).sum())
    for run in ("scaling", "scaling_alpha2", "scaling_fast", "scaling_large"):
        df = pd.read_csv(J / run / "scaling.csv")
        feas = df.milp >= S_MIN
        opt = (df.heur >= df.milp * (1 - 1e-6)) & feas
        speed = df.groupby("n").t_milp.mean() / df.groupby("n").t_heur.mean()
        print(f"{run}: MILP {df.t_milp.mean():.2f} s per request on average, at most {df.t_milp.max():.1f} s; "
              f"heuristic at most {df.t_heur.max():.1f} s; speedup {df.t_milp.mean() / df.t_heur.mean():.1f}x "
              f"(per size {', '.join(f'{n}: {v:.1f}x' for n, v in speed.items())}); "
              f"H optimal on {int(opt.sum())} of {int(feas.sum())} served")
        if run != "scaling_fast":
            tot_served += int(feas.sum())
            tot_nonopt += int((feas & ~opt).sum())
    h = apply_floor(pd.read_csv(J / "spaceground" / "heuristic_nsfnet" / "nsfnet_raw.csv.gz"), S_MIN)
    h = h[(h.ogs == 6) & (h.n_sat_links > 0)]
    f = h[h.method == "FiSER"].set_index(["ti", "s", "d", "F_th"])
    g = h[h.method == "H"].set_index(["ti", "s", "d", "F_th"]).reindex(f.index)
    opt = (g.worst >= f.worst * (1 - 1e-6)) & f.served
    tot_served += int(f.served.sum())
    tot_nonopt += int((f.served & ~opt).sum())
    print(f"satellite snapshots: H optimal on {int(opt.sum())} of {int(f.served.sum())}")
    print(f"all evaluations: H optimal on all but {tot_nonopt} of {tot_served} served requests")

    meta = json.loads((J / "ground" / "params.json").read_text())
    P = PhysParams(**meta["phys"])
    df = load("ground")
    df = df[(df.method == "FiSER") & df.served]
    slack = 0
    for t in TOPOS:
        G = build_network(topologies.load(t)["edges"], P, max_link_km=meta["max_link_km"])
        seeds = {}
        for r in df[df.topology.str.lower() == t].itertuples():
            if (r.s, r.d) not in seeds:
                seeds[r.s, r.d] = mincost_disjoint_paths(G, r.s, r.d, 0.0, 2)
            sp = seeds[r.s, r.d]
            slack += sp is not None and min(path_metrics(G, h, (0,) * len(h))[1] for h in sp) >= r.F_th - 1e-12
    print(f"Theorem 2: the mu = 0 seed is fidelity-slack for {slack} of {len(df)} served single-failure requests")


def space_ground():
    print("\n=== VI-G Space-ground extras (main numbers: analyze_spaceground.py)")
    R = J / "spaceground"
    m = apply_floor(pd.read_csv(R / "main_nsfnet" / "nsfnet_raw.csv.gz"), S_MIN)
    life = apply_floor(pd.read_csv(R / "lifetime60_nsfnet" / "nsfnet_raw.csv.gz"), S_MIN)
    m = m[(m.ogs == 6) & (m.method == "FiSER")]
    win = set(m[m.n_sat_links > 0].ti)
    for F in (0.8, 0.85, 0.9):
        a = m[(m.F_th.round(2) == F) & m.ti.isin(win)]
        b = life[(life.F_th.round(2) == F) & life.ti.isin(win)]
        j = a.merge(b, on=["ti", "s", "d"], suffixes=("_m", "_l"))
        both = j.served_m & j.served_l
        geo = math.exp(np.log(j.worst_l[both] / j.worst_m[both]).mean())
        print(f"links lasting >= 1 min, NSFNET at {F} over the {len(win)} window snapshots: {100 * b.served.mean():.0f}% "
              f"instead of {100 * a.served.mean():.0f}%; worst-case success of requests served either way "
              f"{100 * (geo - 1):+.0f}% (geometric mean)")
    a2 = apply_floor(pd.read_csv(R / "alpha2_nsfnet" / "nsfnet_raw.csv.gz"), S_MIN)
    for label, df in (("two failures", a2), ("one failure", m)):
        w = df[(df.ogs == 6) & (df.method == "FiSER") & (df.n_sat_links > 0) & (df.F_th.round(2) == 0.8) & df.served]
        print(f"NSFNET, six OGS, F_th 0.8, {label}: {100 * w.uses_sat.mean():.0f}% of the served designs use a satellite link")


def main():
    pd.set_option("display.width", 200)
    setup()
    single_failure()
    joint_vs_sequential()
    two_failures()
    coherence()
    heuristic()
    space_ground()


if __name__ == "__main__":
    main()
