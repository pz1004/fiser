"""Space-ground experiments: satellite dual-downlink links between OGS nodes over 24 h.

For each configuration (number of OGS nodes, constellation), every snapshot's network
is the ground network plus the satellite links available at that time; all SD pairs
are provisioned at each F_th. The ground-only network is the k=0 reference.

Example:
  .venv/bin/python experiments/run_spaceground.py --topology nsfnet --ogs 0 2 4 6 \
      --constellations 72/6 144/12 288/18 --fth 0.80 0.85 0.90 --step-s 600
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fiser import methods, topologies  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.paths import path_nodes  # noqa: E402
from fiser.physics import PhysParams  # noqa: E402
from fiser.spaceground import SatParams, SpaceGround, farthest_point_ogs, place_on_map  # noqa: E402

_CTX = {}


def _context(topo_name, phys, sat, k, t_grid, center, max_link_km):
    topo = topologies.load(topo_name)
    G = build_network(topo["edges"], PhysParams(**phys), max_link_km=max_link_km)
    sg = (SpaceGround(G, farthest_point_ogs(G, k), place_on_map(G, topo, center), SatParams(**sat), t_grid)
          if k > 0 else None)
    return G, sg


def _init(*ctx_args):
    _CTX["G"], _CTX["sg"] = _context(*ctx_args)


def _design(H, paths):
    """Compact record: per path 'nodes:levels:satellite hops' (u-v@sat), paths separated by '|'."""
    out = []
    for p in paths:
        sat = ";".join(f"{u}-{v}@{H.edges[(u, v, k)]['sat_id']}" for u, v, k in p.hops
                       if H.edges[(u, v, k)].get("sat_id") is not None)
        out.append(f"{'-'.join(map(str, path_nodes(p.hops)))}:{''.join(map(str, p.levels))}:{sat}")
    return "|".join(out)


def _work(job):
    ti, F_ths, meths, n_paths = job
    G, sg = _CTX["G"], _CTX["sg"]
    H = sg.snapshot(ti) if sg is not None else G
    n_sat = sum(1 for *_, kind in H.edges(keys=True, data="kind") if kind == "sat")
    rows = []
    for s, d in itertools.combinations(sorted(G.nodes), 2):
        for F_th, res in methods.run_pair(H, s, d, F_ths, meths, n_paths=n_paths).items():
            for name, o in res.items():
                uses_sat = any(H.edges[h]["kind"] == "sat" for p in o.paths for h in p.hops) if o.served else False
                n_sat_paths = sum(any(H.edges[h]["kind"] == "sat" for h in p.hops) for p in o.paths) if o.served else 0
                rows.append(dict(ti=ti, s=s, d=d, F_th=F_th, method=name, served=o.served,
                                 worst=o.worst, uses_sat=uses_sat, sat_paths=n_sat_paths,
                                 n_sat_links=n_sat, runtime=o.runtime,
                                 design=_design(H, o.paths) if o.served else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--ogs", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--constellations", nargs="+", default=["144/12"], help="total/planes")
    ap.add_argument("--fth", type=float, nargs="+", default=[0.80, 0.85, 0.90])
    ap.add_argument("--methods", nargs="+", default=["FiSER"])
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--step-s", type=int, default=300)
    ap.add_argument("--center", type=float, nargs=2, default=[39.0, -98.0],
                    help="lat lon used only for topologies without coordinates")
    ap.add_argument("--max-link-km", type=float, default=100.0)
    ap.add_argument("--sat", nargs="*", default=[], help="SatParams overrides, e.g. F_sat=0.93")
    ap.add_argument("--param", nargs="*", default=[], help="PhysParams overrides")
    ap.add_argument("--paths", type=int, default=2, help="disjoint paths per request (failures tolerated + 1)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "results" / "spaceground"))
    args = ap.parse_args()

    P, S = PhysParams(), SatParams()
    for kv in args.param:
        k, v = kv.split("=")
        P = replace(P, **{k: type(getattr(P, k))(v)})
    for kv in args.sat:
        k, v = kv.split("=")
        cur = getattr(S, k)
        val = (v.lower() in ("1", "true")) if isinstance(cur, bool) else v if isinstance(cur, str) \
            else type(cur)(float(v))
        S = replace(S, **{k: val})
    t_grid = np.arange(0, args.hours * 3600, args.step_s)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def run(Sc, k, snaps):
        ctx = (args.topology, P.to_dict(), Sc.__dict__, k, t_grid, tuple(args.center), args.max_link_km)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=ctx) as ex:
            return [r for rs in ex.map(_work, [(ti, args.fth, args.methods, args.paths) for ti in snaps]) for r in rs]

    ground = pd.DataFrame(run(S, 0, [0]))  # ground-only is static; reused for snapshots without satellites
    frames = []
    for const in args.constellations:
        total, planes = map(int, const.split("/"))
        Sc = replace(S, total=total, planes=planes)
        for k in args.ogs:
            t0 = time.time()
            if k == 0:
                df, snaps = ground.copy(), [0]
            else:
                sg = _context(args.topology, P.to_dict(), Sc.__dict__, k, t_grid, tuple(args.center),
                              args.max_link_km)[1]
                snaps = [ti for ti in range(len(t_grid)) if sg.links(ti)]
                active = set(snaps)
                idle = [ti for ti in range(len(t_grid)) if ti not in active]
                df = pd.concat([pd.DataFrame(run(Sc, k, snaps))]
                               + [ground.assign(ti=ti) for ti in idle], ignore_index=True)
            df = df.assign(constellation=const, ogs=k)
            frames.append(df)
            agg = df.groupby(["F_th", "method"]).agg(served=("served", "mean"), worst=("worst", "mean"))
            print(f"{const} k={k}: {len(snaps)} active snapshots of {len(t_grid) if k else 1} "
                  f"in {time.time() - t0:.0f}s")
            print(agg.round(3).to_string())

    df = pd.concat(frames)
    df.to_csv(out / f"{args.topology}_raw.csv.gz", index=False)
    summ = (df.groupby(["constellation", "ogs", "F_th", "method"])
              .agg(service=("served", "mean"), mean_worst=("worst", "mean"), sat_use=("uses_sat", "mean"))
              .reset_index())
    summ.to_csv(out / f"{args.topology}_summary.csv", index=False)
    (out / "params.json").write_text(json.dumps(
        {"phys": P.to_dict(), "sat": S.__dict__, "args": vars(args)}, indent=2, default=str))


if __name__ == "__main__":
    main()
