"""Residual lifetime of the satellite links offered at each snapshot (both OGS see the satellite above the
elevation mask, at night), on a fine time grid, and per-snapshot re-provisioning time of the heuristic.

Example:
  .venv/bin/python experiments/link_lifetime.py --topology nsfnet --ogs 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser import topologies  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.physics import PhysParams  # noqa: E402
from fiser.spaceground import SatParams, SpaceGround, farthest_point_ogs, place_on_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--ogs", type=int, default=6)
    ap.add_argument("--step-s", type=int, default=300, help="snapshot spacing of the main runs")
    ap.add_argument("--fine-s", type=int, default=10)
    args = ap.parse_args()
    topo = topologies.load(args.topology)
    G = build_network(topo["edges"], PhysParams(), max_link_km=100)
    ll = place_on_map(G, topo, (39.0, -98.0))
    ogs = farthest_point_ogs(G, args.ogs)
    t = np.arange(0, 86400 + 3600, args.fine_s)
    sg = SpaceGround(G, ogs, ll, SatParams(), t)
    up = sg.eta > 0  # (T, n_ogs, n_sats)
    idx = {o: i for i, o in enumerate(sg.ogs)}
    ratio = args.step_s // args.fine_s
    life = []
    for ti in range(0, int(86400 / args.fine_s), ratio):
        for u, v, sat, _ in sg.links(ti):
            ok = up[ti:, idx[u], sat] & up[ti:, idx[v], sat]
            stop = np.flatnonzero(~ok)
            life.append((stop[0] if len(stop) else len(ok)) * args.fine_s)
    life = np.array(life)
    print(f"{args.topology}, {args.ogs} OGS: {len(life)} offered satellite links at {args.step_s}-s snapshots | "
          f"residual lifetime median {np.median(life) / 60:.1f} min, mean {life.mean() / 60:.1f} min | "
          f"lasting >= 1 min {100 * (life >= 60).mean():.0f}%, >= 2 min {100 * (life >= 120).mean():.0f}%, "
          f">= {args.step_s // 60} min {100 * (life >= args.step_s).mean():.0f}%")
    h = ROOT / "results/journal/spaceground/heuristic_nsfnet/nsfnet_raw.csv.gz"
    if args.topology == "nsfnet" and h.exists():
        df = pd.read_csv(h)
        df = df[(df.ogs == args.ogs) & (df.n_sat_links > 0)]
        per = df.groupby(["ti", "method"]).runtime.sum().unstack()
        print(f"re-provisioning all {df.groupby('ti').size().iloc[0] // 2} requests of one threshold per snapshot "
              f"(single core): MILP median {per.FiSER.median():.1f} s, heuristic median {per.H.median():.1f} s")


if __name__ == "__main__":
    main()
