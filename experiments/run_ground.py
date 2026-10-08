"""Ground-network experiments: FiSER vs baselines over all SD pairs of a topology.

Examples:
  .venv/bin/python experiments/run_ground.py --topology nsfnet --fth 0.70 0.75 0.80 0.85 0.90
  .venv/bin/python experiments/run_ground.py --topology nsfnet cost239 usnet --paths 3 \
      --methods B1 B2 B3 B4 B5 H FiSER --out results/ground_alpha2    # two-failure protection
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fiser import methods, report, topologies  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.paths import path_nodes  # noqa: E402
from fiser.physics import PhysParams  # noqa: E402

_G = None


def _init(edges, params, max_link_km, scale):
    global _G
    _G = build_network(edges, PhysParams(**params), max_link_km=max_link_km, scale=scale)


def _work(job):
    s, d, F_ths, meths, K, n_paths = job
    rows = []
    for F_th, res in methods.run_pair(_G, s, d, F_ths, meths, K=K, n_paths=n_paths).items():
        for name, o in res.items():
            row = dict(s=s, d=d, F_th=F_th, method=name, served=o.served, worst=o.worst,
                       runtime=o.runtime)
            for i, pc in enumerate(o.paths, 1):
                row.update({f"S{i}": pc.S, f"F{i}": pc.F,
                            f"nodes{i}": "-".join(map(str, path_nodes(pc.hops))),
                            f"lev{i}": "".join(map(str, pc.levels))})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", nargs="+", default=["nsfnet"])
    ap.add_argument("--fth", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument("--methods", nargs="+", default=list(methods.ALL))
    ap.add_argument("--K", type=int, nargs=2, default=[20, 10], help="heuristic K1 (primary) K2 (backup)")
    ap.add_argument("--max-link-km", type=float, default=100.0)
    ap.add_argument("--scale", type=float, default=None, help="explicit km scale (overrides max-link-km)")
    ap.add_argument("--param", nargs="*", default=[], help="PhysParams overrides, e.g. q_swap=0.95")
    ap.add_argument("--s-min", type=float, default=0.01, help="service floor on worst-case success per slot")
    ap.add_argument("--paths", type=int, default=2, help="disjoint paths per request (failures tolerated + 1)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results" / "ground"))
    args = ap.parse_args()

    P = PhysParams()
    for kv in args.param:
        k, v = kv.split("=")
        P = replace(P, **{k: type(getattr(P, k))(v)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in args.topology:
        topo = topologies.load(name)
        nodes = sorted({u for u, _, _ in topo["edges"]} | {v for _, v, _ in topo["edges"]})
        jobs = [(s, d, args.fth, args.methods, tuple(args.K), args.paths)
                for s, d in itertools.combinations(nodes, 2)]
        t0 = time.time()
        with ProcessPoolExecutor(args.workers, initializer=_init,
                                 initargs=(topo["edges"], P.to_dict(), args.max_link_km, args.scale)) as ex:
            rows = [r for rs in ex.map(_work, jobs, chunksize=2) for r in rs]
        df = pd.DataFrame(rows).assign(topology=topo["name"])
        print(f"{topo['name']}: {len(jobs)} pairs, {len(args.fth)} thresholds, "
              f"{time.time() - t0:.1f}s wall")
        df.to_csv(out / f"{name}.csv", index=False)
        frames.append(df)

    df = pd.concat(frames)
    tab = report.summary(df, s_min=args.s_min)
    tab.to_csv(out / "summary.csv", index=False)
    report.summary(df, s_min=0.0).to_csv(out / "summary_nofloor.csv", index=False)
    if "B6" in args.methods:
        pos = report.price_of_survivability(df, s_min=args.s_min)
        pos.to_csv(out / "price_of_survivability.csv", index=False)
        print("\nPrice of survivability (FiSER worst-case vs best single path):")
        print(pos.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (out / "params.json").write_text(json.dumps(
        {"phys": P.to_dict(), "max_link_km": args.max_link_km, "scale": args.scale, "paths": args.paths,
         "K": args.K, "fth": args.fth, "topologies": args.topology, "s_min": args.s_min}, indent=2))
    report.print_summary(tab)


if __name__ == "__main__":
    main()
