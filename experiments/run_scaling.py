"""Scalability: MILP vs heuristic runtime and optimality on random meshes of growing size.

Example:
  .venv/bin/python experiments/run_scaling.py --sizes 20 40 60 80 100 --seeds 1 2 3 --pairs 30
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fiser import heuristic, milp, topologies  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.physics import PhysParams  # noqa: E402


def _job(args):
    n, seed, s, d, F_ths, hk, n_paths = args
    G = build_network(topologies.random_mesh(n, seed)["edges"], PhysParams(), scale=1.0)
    rows = []
    for F_th in F_ths:
        m = milp.solve(G, s, d, F_th, n_paths=n_paths)
        h = heuristic.disjoint_paths(G, s, d, F_th, n_paths, **hk)
        mv = min(p.S for p in m.paths) if m.ok else 0.0
        hv = min(p.S for p in h.paths) if h.ok else 0.0
        rows.append(dict(n=n, seed=seed, s=s, d=d, F_th=F_th, milp=mv, heur=hv,
                         t_milp=m.runtime, t_heur=h.runtime, milp_status=m.status))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--fth", type=float, nargs="+", default=[0.75, 0.80])
    ap.add_argument("--K", type=int, nargs=2, default=[20, 10], help="heuristic K1 K2")
    ap.add_argument("--mus", type=float, nargs="+", default=[0.0, 2.0, 8.0])
    ap.add_argument("--paths", type=int, default=2, help="disjoint paths per request")
    ap.add_argument("--max-pair-km", type=float, default=None,
                    help="sample only SD pairs at most this far apart (local requests in large networks)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaling"))
    args = ap.parse_args()

    jobs = []
    for n in args.sizes:
        for seed in args.seeds:
            rng = random.Random(f"pairs-{n}-{seed}")
            cand = [(s, d) for s in range(n) for d in range(s + 1, n)]
            if args.max_pair_km is not None:
                import math
                pos = topologies.random_mesh(n, seed)["pos"]
                cand = [(s, d) for s, d in cand if math.dist(pos[s], pos[d]) <= args.max_pair_km]
            pairs = rng.sample(cand, min(args.pairs, len(cand)))
            jobs += [(n, seed, s, d, args.fth, dict(K1=args.K[0], K2=args.K[1], mus=tuple(args.mus)), args.paths)
                     for s, d in pairs]
    with ProcessPoolExecutor(args.workers) as ex:
        df = pd.DataFrame([r for rs in ex.map(_job, jobs) for r in rs])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scaling.csv", index=False)

    rows = []
    for (n, F_th), g in df.groupby(["n", "F_th"]):
        both = (g.milp > 0) & (g.heur > 0)
        rows.append(dict(n=n, F_th=F_th, instances=len(g), milp_served=int((g.milp > 0).sum()),
                         heur_served=int((g.heur > 0).sum()),
                         heur_optimal=int((np.abs(g.heur - g.milp) <= 1e-6 * g.milp.clip(lower=1e-12)).sum()),
                         mean_gap_pct=float(100 * (1 - g.heur[both] / g.milp[both]).mean()) if both.any() else np.nan,
                         t_milp_mean=g.t_milp.mean(), t_milp_max=g.t_milp.max(),
                         t_heur_mean=g.t_heur.mean(), t_heur_max=g.t_heur.max()))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
