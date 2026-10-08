"""Geometry and load statistics quoted in the space-ground limitations (night windows, 6 OGS).

Prints per topology:
- OGS separations;
- satellites carrying links per window snapshot (network-wide);
- served FiSER designs per window snapshot that rely on a satellite link (F_th = 0.85, rate floor);
- satellite pass length above the minimum elevation, seen from the network centroid.

Example:
  .venv/bin/python experiments/satellite_stats.py
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser import topologies  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.physics import PhysParams  # noqa: E402
from fiser.report import apply_floor  # noqa: E402
from fiser.satellite import ground_ecef, look_angles  # noqa: E402
from fiser.spaceground import SatParams, SpaceGround, farthest_point_ogs, place_on_map  # noqa: E402

N_OGS, F_TH, S_MIN, STEP = 6, 0.85, 0.01, 300


def km(a, b):
    return math.hypot((a[0] - b[0]) * 111.2, (a[1] - b[1]) * 111.2 * math.cos(math.radians(a[0])))


def main():
    S = SatParams()
    for name in ("nsfnet", "cost239", "usnet"):
        topo = topologies.load(name)
        G = build_network(topo["edges"], PhysParams(), max_link_km=100)
        ll = place_on_map(G, topo, (39.0, -98.0))
        ogs = farthest_point_ogs(G, N_OGS)
        sep = [km(ll[a], ll[b]) for a, b in itertools.combinations(ogs, 2)]
        t = np.arange(0, 86400, STEP)
        sg = SpaceGround(G, ogs, ll, S, t)
        win = [ti for ti in range(len(t)) if sg.links(ti)]
        sats = [len({sat for *_, sat, _ in sg.links(ti)}) for ti in win]

        raw = apply_floor(pd.read_csv(ROOT / f"results/journal/spaceground/main_{name}/{name}_raw.csv.gz"), S_MIN)
        r = raw[(raw.method == "FiSER") & (raw.ogs == N_OGS) & (raw.F_th == F_TH) & (raw.n_sat_links > 0)]
        rely = r[r.served & r.uses_sat].groupby("ti").size().reindex(r.ti.unique(), fill_value=0)

        # pass lengths at the centroid: maximal runs above the elevation mask, per satellite (1-min grid)
        lat = np.mean([v[0] for v in ll.values()])
        lon = np.mean([v[1] for v in ll.values()])
        tt = np.arange(0, 86400, 60)
        el, _ = look_angles(ground_ecef([lat], [lon]), S.walker().ecef(tt))
        vis = el[:, 0, :] >= S.el_min_deg
        runs = []
        for k in range(vis.shape[1]):
            edges = np.flatnonzero(np.diff(np.r_[0, vis[:, k].astype(int), 0]))
            runs += list((edges[1::2] - edges[::2]))
        print(f"{name}: OGS separation {min(sep):.1f}-{max(sep):.1f} km | satellites carrying links per window "
              f"snapshot {np.mean(sats):.2f} (max {max(sats)}) | served designs relying on a satellite per "
              f"snapshot {rely.mean():.1f} (max {rely.max()}) | pass length median {np.median(runs):.1f} min")


if __name__ == "__main__":
    main()
