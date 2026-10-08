"""FiSER vs the sequential (two-step) design B5: numbers for the joint-selection paragraph.

Reads results/journal/b5_study/{hetero,mesh,alpha2}.csv.gz (experiments/run_b5_study.py) and cross-checks
them against results/journal/ground (FiSER and B5 served counts) and results/journal/scaling (Table V).

Example:
  .venv/bin/python experiments/analyze_b5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser.report import apply_floor  # noqa: E402

R = ROOT / "results"
S_MIN = 0.01


def line(g, label):
    lost = g.fiser_served & ~g.seq_served
    both = g.fiser_served & g.seq_served
    r = g.seq[both] / g.fiser[both]
    f, s = int(g.fiser_served.sum()), int(g.seq_served.sum())
    reasons = {k: int((lost & (g.seq_reason == k)).sum()) for k in ("trap", "fidelity", "floor")}
    print(f"{label:30s} FiSER {f:4d}  seq {s:4d}  +{f - s:3d} (+{100 * (f / max(s, 1) - 1):4.1f}%)  "
          f"lost: no path {reasons['trap']}, fidelity {reasons['fidelity']}, floor {reasons['floor']} | "
          f"both {int(both.sum())}: mean ratio {r.mean():.3f}, >1% worse {int((r < 0.99).sum())}, "
          f"<half {int((r < 0.5).sum())}")


def main():
    h = pd.read_csv(R / "journal" / "b5_study" / "hetero.csv.gz")
    m = pd.read_csv(R / "journal" / "b5_study" / "mesh.csv.gz")
    a = pd.read_csv(R / "journal" / "b5_study" / "alpha2.csv.gz")
    for df in (h, m, a):
        assert (df.seq <= df.fiser * (1 + 1e-6) + 1e-12).all(), "sequential beat FiSER"

    print("=== Paper setting (single failure, homogeneous hardware)")
    h0 = h[h.spread == 0]
    ground = apply_floor(pd.read_csv(R / "ground" / "raw.csv.gz"), S_MIN) if (R / "ground" / "raw.csv.gz").exists() else None
    for t in ("nsfnet", "cost239", "usnet"):
        g = h0[h0.topology == t]
        for F, gf in g.groupby("F_th"):
            line(gf, f"{t} F_th={F:.2f}")
        line(g[g.F_th <= 0.85], f"{t} F_th 0.70-0.85")
    if ground is not None:
        print("(cross-check against results/ground skipped: see the summary CSV check below)")
    summ = pd.read_csv(R / "journal" / "ground" / "summary.csv")
    for t in ("nsfnet", "cost239", "usnet"):
        g = h0[h0.topology == t]
        for F, gf in g.groupby("F_th"):
            row = summ[(summ.topology.str.lower() == t) & (summ.F_th.round(2) == round(F, 2))]
            for meth, col in (("FiSER", "fiser_served"), ("B5", "seq_served")):
                ref = row[row.method == meth].served
                if len(ref):
                    assert int(ref.iloc[0]) == int(gf[col].sum()), (t, F, meth, int(ref.iloc[0]), int(gf[col].sum()))
    print("served counts match results/ground/summary.csv for FiSER and B5")

    print("\n=== Random meshes (single failure)")
    sc = pd.read_csv(R / "journal" / "scaling" / "scaling.csv")
    for deg, g in m.groupby("degree"):
        for n, gn in g.groupby("n"):
            line(gn, f"degree {deg} n={n}")
        line(g, f"degree {deg} pooled")
        if abs(deg - 3.5) < 1e-9:
            mine = g.set_index(["n", "seed", "s", "d", "F_th"]).fiser_served.sort_index()
            ref = (sc.set_index(["n", "seed", "s", "d", "F_th"]).milp >= S_MIN).sort_index()
            assert mine.index.equals(ref.index) and (mine == ref).all(), "mesh instances differ from Table I"
            print("degree 3.5 instances and FiSER feasibility match Table I (results/scaling)")

    print("\n=== Heterogeneous hardware (single failure), pooled over F_th 0.70-0.90")
    for (t, sp), g in h.groupby(["topology", "spread"]):
        line(g, f"{t} spread={sp}")

    print("\n=== Two-failure protection (three disjoint paths)")
    for t in ("nsfnet", "cost239", "usnet"):
        g = a[a.topology == t]
        for F, gf in g.groupby("F_th"):
            line(gf, f"{t} F_th={F:.2f}")
        line(g, f"{t} pooled")


if __name__ == "__main__":
    main()
