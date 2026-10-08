"""Summary statistics over per-(pair, F_th, method) result rows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_floor(df, s_min=0.0):
    """Service = fidelity-feasible protected design with worst-case success >= s_min per slot.
    Exact post hoc: every method's design (and FiSER's optimum) is fixed before the floor."""
    out = df.copy()
    ok = out.served.astype(bool) & (out.worst >= s_min) & (out.worst > 0)
    out["served"] = ok
    out["worst"] = out.worst.where(ok, 0.0)
    return out


def summary(df, ref="B2", order=("B1", "B2", "B3", "B4", "B5", "H", "FiSER"), s_min=0.0):
    """Per (topology, F_th, method): served pairs, mean worst-case success (0 if unserved),
    geometric-mean ratio vs ref over pairs both serve, pairs beating ref by >1%, runtime."""
    df = apply_floor(df, s_min)
    rows = []
    for (topo, F_th), g in df.groupby(["topology", "F_th"], sort=True):
        refv = g[g.method == ref].set_index(["s", "d"]).worst
        n_pairs = len(refv)
        for m in [m for m in order if m in set(g.method)]:
            v = g[g.method == m].set_index(["s", "d"]).worst.reindex(refv.index)
            both = (v > 0) & (refv > 0)
            geo = float(np.exp(np.mean(np.log(v[both] / refv[both])))) if both.any() else np.nan
            rows.append(dict(topology=topo, F_th=F_th, method=m,
                             served=int((v > 0).sum()), pairs=n_pairs,
                             mean_worst=float(v.mean()), geo_vs_ref=geo,
                             beats_ref=int((v > 1.01 * refv).sum()),
                             runtime_s=float(g[g.method == m].runtime.sum())))
    return pd.DataFrame(rows)


def price_of_survivability(df, method="FiSER", single="B6", s_min=0.0):
    """Per (topology, F_th): pairs with a fidelity-feasible single path vs a survivable pair (both
    meeting the rate floor), and the geometric-mean ratio of worst-case survivable success to
    failure-free single-path success."""
    rows = []
    for (topo, F_th), g in df.groupby(["topology", "F_th"], sort=True):
        sg = g[g.method == single].set_index(["s", "d"])
        fw = g[g.method == method].set_index(["s", "d"]).worst.reindex(sg.index)
        single_ok, surv_ok = sg.S1.fillna(0) >= max(s_min, 1e-300), fw >= max(s_min, 1e-300)
        both = single_ok & surv_ok
        ratio = float(np.exp(np.mean(np.log(fw[both] / sg.S1[both])))) if both.any() else np.nan
        rows.append(dict(topology=topo, F_th=F_th, pairs=len(sg), single_feasible=int(single_ok.sum()),
                         survivable=int(surv_ok.sum()), geo_worst_over_single=ratio))
    return pd.DataFrame(rows)


def print_summary(tab, ref="B2"):
    for (topo, F_th), g in tab.groupby(["topology", "F_th"], sort=True):
        print(f"\n{topo}  F_th={F_th:.2f}")
        print(f"  {'method':<7}{'served':>10}{'mean worst S':>14}{f'geo vs {ref}':>12}"
              f"{f'>{ref}+1%':>10}{'time[s]':>9}")
        for r in g.itertuples():
            print(f"  {r.method:<7}{r.served:>5}/{r.pairs:<4}{r.mean_worst:>14.4f}"
                  f"{r.geo_vs_ref:>11.3f}x{r.beats_ref:>10}{r.runtime_s:>9.2f}")
