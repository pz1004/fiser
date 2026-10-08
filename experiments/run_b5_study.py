"""Where does joint primary-backup selection (FiSER) beat the fidelity-aware sequential design (B5)?

Scenarios (all with the rate floor S_min):
  mesh   random biconnected meshes by size and average degree (the Table I generator)
  hetero the three topologies with heterogeneous link hardware (w0 ~ U[w0 - spread, w0 + spread])
  alpha2 protection against two failures: three disjoint paths, FiSER vs a three-step sequential design

For every request we record FiSER's and the sequential design's worst-case success and, when the
sequential design fails although FiSER serves the request, why: no disjoint path left at all after
the greedy primary(ies) (a pure trap), no fidelity-feasible one, or one below the rate floor.

Examples (the paper's settings):
  .venv/bin/python experiments/run_b5_study.py --scenario hetero --spreads 0 0.01 0.02
  .venv/bin/python experiments/run_b5_study.py --scenario mesh --degrees 3.5 3.0 --fth 0.75 0.80
  .venv/bin/python experiments/run_b5_study.py --scenario alpha2
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fiser import milp, topologies  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.paths import path_nodes  # noqa: E402
from fiser.physics import PhysParams  # noqa: E402

S_MIN = 0.01


def sequential(G, s, d, F_th, n_paths):
    """B5 generalized: best purified path, then the best one in the residual graph, and so on.

    Returns (worst, reason): reason is None if all n_paths were found, else 'trap' (no s-d path at
    all is left), 'fidelity' (paths exist but none meets F_th), or 'primary' (no feasible first path).
    """
    H, chosen = G.copy(), []
    for m in range(n_paths):
        sol = milp.solve(H, s, d, F_th, n_paths=1)
        if not sol.ok:
            if m == 0:
                return 0.0, "primary"
            return 0.0, "fidelity" if nx.has_path(nx.Graph(H), s, d) else "trap"
        p = sol.paths[0]
        chosen.append(p)
        # as B5 (methods._two_step): drop the path's intermediate nodes, its links, and its satellites
        used_sats = {H.edges[h].get("sat_id") for h in p.hops} - {None}
        H.remove_nodes_from(path_nodes(p.hops)[1:-1])
        H.remove_edges_from([(u, v, k) for u, v, k, sid in H.edges(keys=True, data="sat_id")
                             if sid in used_sats or (u, v, k) in p.hops or (v, u, k) in p.hops])
    return min(p.S for p in chosen), None


def _job(args):
    tag, edges, phys, s, d, F_ths, n_paths, meta = args
    G = build_network(edges, PhysParams(**phys), scale=1.0)
    rows = []
    for F_th in F_ths:
        f = milp.solve(G, s, d, F_th, n_paths=n_paths)
        fw = min(p.S for p in f.paths) if f.ok else 0.0
        bw, reason = sequential(G, s, d, F_th, n_paths)
        if bw and bw < S_MIN:
            reason = "floor"
        rows.append(dict(scenario=tag, s=s, d=d, F_th=F_th, n_paths=n_paths, fiser=fw, seq=bw,
                         fiser_served=fw >= S_MIN, seq_served=bw >= S_MIN, seq_reason=reason,
                         t_fiser=f.runtime, **meta))
    return rows


def mesh_jobs(args):
    jobs = []
    for deg in args.degrees:
        for n in args.sizes:
            for seed in args.seeds:
                topo = topologies.random_mesh(n, seed, target_degree=deg)
                rng = random.Random(f"pairs-{n}-{seed}")  # the SD pairs of run_scaling.py (Table I)
                pairs = rng.sample([(s, d) for s in range(n) for d in range(s + 1, n)], args.pairs)
                meta = dict(n=n, degree=deg, seed=seed)
                jobs += [("mesh", topo["edges"], PhysParams().to_dict(), s, d, args.fth, 2, meta)
                         for s, d in pairs]
    return jobs


def topo_jobs(args, n_paths, spreads):
    jobs = []
    for name in args.topologies:
        topo = topologies.load(name)
        G0 = build_network(topo["edges"], PhysParams(), max_link_km=100)
        edges = [(u, v, G0.edges[u, v, k]["L"]) for u, v, k in G0.edges(keys=True)]  # scaled lengths
        for spread in spreads:
            for seed in args.seeds:
                phys = replace(PhysParams(), w0_spread=spread, seed=seed).to_dict()
                meta = dict(topology=name, spread=spread, seed=seed)
                for s, d in itertools.combinations(sorted(G0.nodes), 2):
                    jobs.append((f"{'alpha2' if n_paths == 3 else 'hetero'}", edges, phys, s, d, args.fth,
                                 n_paths, meta))
                if spread == 0:
                    break  # homogeneous: one seed suffices
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["mesh", "hetero", "alpha2"], required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--degrees", type=float, nargs="+", default=[3.0, 3.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--topologies", nargs="+", default=["nsfnet", "cost239", "usnet"])
    ap.add_argument("--spreads", type=float, nargs="+", default=[0.0, 0.01, 0.02])
    ap.add_argument("--fth", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "results" / "b5_study"))
    args = ap.parse_args()

    if args.scenario == "mesh":
        jobs = mesh_jobs(args)
    elif args.scenario == "hetero":
        jobs = topo_jobs(args, 2, args.spreads)
    else:
        jobs = topo_jobs(args, 3, [0.0])
    with ProcessPoolExecutor(args.workers) as ex:
        df = pd.DataFrame([r for rs in ex.map(_job, jobs, chunksize=4) for r in rs])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.scenario}.csv.gz", index=False)
    keys = {"mesh": ["degree", "n", "F_th"], "hetero": ["topology", "spread", "F_th"],
            "alpha2": ["topology", "F_th"]}[args.scenario]
    summary(df, keys)


def summary(df, keys):
    rows = []
    for k, g in df.groupby(keys):
        both = g.fiser_served & g.seq_served
        lost = g.fiser_served & ~g.seq_served
        ratio = (g.seq[both] / g.fiser[both])
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), requests=len(g),
                         fiser=int(g.fiser_served.sum()), seq=int(g.seq_served.sum()),
                         extra_pct=100 * (g.fiser_served.sum() / max(g.seq_served.sum(), 1) - 1),
                         trap=int((lost & (g.seq_reason == "trap")).sum()),
                         fid=int((lost & (g.seq_reason == "fidelity")).sum()),
                         floor=int((lost & (g.seq_reason == "floor")).sum()),
                         worse_1pct=int((ratio < 0.99).sum()), mean_ratio=ratio.mean(), min_ratio=ratio.min()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
