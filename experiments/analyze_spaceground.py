"""Numbers for the space-ground subsection, all with the service floor S_min.

"Night windows" are snapshots with at least one satellite link (n_sat_links > 0).

Example:
  .venv/bin/python experiments/analyze_spaceground.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser.report import apply_floor  # noqa: E402

R = ROOT / "results" / "journal" / "spaceground"
S_MIN = 0.01


def load(run, topo):
    return apply_floor(pd.read_csv(R / run / f"{topo}_raw.csv.gz"), S_MIN)


def shares(raw, method="FiSER"):
    raw = raw[raw.method == method]
    g0 = raw[raw.ogs == 0].groupby("F_th").agg(ground=("served", "mean"), w_ground=("worst", "mean"))
    rows = []
    for (const, k), r in raw[raw.ogs > 0].groupby(["constellation", "ogs"]):
        win = r[r.n_sat_links > 0]
        avail = r.groupby("ti").n_sat_links.first().gt(0).mean()
        for F, rf in r.groupby("F_th"):
            wf = win[win.F_th == F]
            rows.append(dict(const=const, ogs=k, F_th=F, avail=avail,
                             ground=g0.loc[F, "ground"], day=rf.served.mean(), night=wf.served.mean(),
                             w_ground=g0.loc[F, "w_ground"], w_night=wf.worst.mean()))
    return pd.DataFrame(rows)


def fmt(df):
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main():
    import argparse
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(R), help="results directory of the space-ground runs")
    R = Path(ap.parse_args().root)
    print(f"service floor S_min = {S_MIN}; runs from {R}\n")
    for topo in ("nsfnet", "cost239", "usnet"):
        print(f"=== MAIN {topo} (FiSER)")
        print(fmt(shares(load(f"main_{topo}", topo))))

    # gain decomposition (m7): pairs served on the ground vs pairs newly served by satellites
    raw = load("main_nsfnet", "nsfnet")
    raw = raw[raw.method == "FiSER"]
    g0 = raw[raw.ogs == 0].set_index(["s", "d", "F_th"]).worst
    for k in (4, 6):
        win = raw[(raw.ogs == k) & (raw.n_sat_links > 0)]
        w = win.groupby(["s", "d", "F_th"]).worst.mean()
        base = g0.reindex(w.index)
        for F in (0.85, 0.90):
            idx = w.index.get_level_values("F_th") == F
            old = base[idx] > 0
            print(f"NSFNET N_o={k} F={F}: pairs served on ground: mean worst {base[idx][old].mean():.3f} -> "
                  f"{w[idx][old].mean():.3f} (+{100 * (w[idx][old].mean() / base[idx][old].mean() - 1):.0f}%); "
                  f"newly servable pairs: {int((~old & (w[idx] > 0)).sum())}")

    print("\n=== BASELINES WITH SATELLITES (NSFNET, N_o = 4), night windows")
    b = load("baselines_nsfnet", "nsfnet")
    win = b[(b.ogs == 4) & (b.n_sat_links > 0)]
    rows = []
    for (F, m), g in win.groupby(["F_th", "method"]):
        sv = g[g.served]
        rows.append(dict(F_th=F, method=m, served=g.served.mean(), worst=g.worst.mean(),
                         sat_use_of_served=sv.uses_sat.mean() if len(sv) else np.nan))
    print(fmt(pd.DataFrame(rows)))
    print("ground (N_o = 0):")
    print(fmt(b[b.ogs == 0].groupby(["F_th", "method"]).served.mean().reset_index()))

    print("\n=== CONSTELLATION SIZE (NSFNET, N_o = 4)")
    c = pd.concat([load("constellation_nsfnet", "nsfnet"),
                   load("main_nsfnet", "nsfnet").query("ogs in (0, 4)")])
    print(fmt(shares(c)))

    print("\n=== SENSITIVITY (NSFNET, N_o = 4)")
    for run, label in (("sens_fsat087", "F_sat = 0.87"), ("sens_etasys006", "eta_sys = 0.06"),
                       ("sens_m5900", "M_sat = 5.9e3 (Micius today)")):
        print(f"--- {label}")
        print(fmt(shares(load(run, "nsfnet"))))

    for run, label, meths in (("alpha2_nsfnet", "TWO-FAILURE PROTECTION (NSFNET, N_o = 6, three paths)", ("FiSER", "B5")),
                              ("T0.1_nsfnet", "MEMORY COHERENCE T = 0.1 s (NSFNET, N_o = 6)", ("FiSER",))):
        if not (R / run / "nsfnet_raw.csv.gz").exists():
            continue
        print(f"\n=== {label}")
        raw = load(run, "nsfnet")
        for m in meths:
            print(f"--- {m}")
            print(fmt(shares(raw, m)))

    if (R / "heuristic_nsfnet" / "nsfnet_raw.csv.gz").exists():
        print("\n=== HEURISTIC ON SATELLITE SNAPSHOTS (NSFNET, N_o = 6, F_th = 0.85)")
        h = load("heuristic_nsfnet", "nsfnet")
        h = h[(h.ogs == 6) & (h.n_sat_links > 0)]
        f = h[h.method == "FiSER"].set_index(["ti", "s", "d", "F_th"])
        g = h[h.method == "H"].set_index(["ti", "s", "d", "F_th"]).reindex(f.index)
        sv = f.worst > 0
        opt = (g.worst >= f.worst * (1 - 1e-6)) & sv
        print(f"served {int(sv.sum())}, H optimal on {int(opt.sum())}; runtime ratio (served) "
              f"{f.runtime[sv].sum() / g.runtime[sv].sum():.1f}x, all {f.runtime.sum() / g.runtime.sum():.1f}x; "
              f"both paths on satellites in {100 * (f[sv].sat_paths >= 2).mean():.1f}% of FiSER's served designs")


if __name__ == "__main__":
    main()
