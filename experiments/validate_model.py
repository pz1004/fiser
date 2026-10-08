"""Validation of the analytical link/path model on the designs FiSER actually serves.

1. Recompute: every recorded design's (S, F) is recomputed from the network and its levels.
2. Monte Carlo: the slot protocol is simulated (binomial raw pairs, every BBPSSW round of the
   recurrence tree, swaps) and the empirical success is compared with S by an exact two-sided
   binomial test (z-scores are unreliable when S is within 1e-6 of 1).
3. Link scheduling: the model assumes just-in-time link starts (all links of a path finish
   together). If all links instead start together, the qubits of faster links wait for the
   slowest one; we report the fidelity loss and how many served designs would miss F_th.

Example:
  .venv/bin/python experiments/validate_model.py --runs results/journal/ground_T0.1 results/journal/ground_T0.01
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser import topologies  # noqa: E402
from fiser.network import build_network  # noqa: E402
from fiser.paths import path_metrics  # noqa: E402
from fiser.physics import F_of_w, PhysParams, bbpssw, stored, w_of_F  # noqa: E402

S_MIN = 0.01


def hops_of(nodes):
    seq = [int(x) for x in str(nodes).split("-")]
    return [(seq[i], seq[i + 1], "f") for i in range(len(seq) - 1)]


def simulate(G, hops, levels, n_slots, rng):
    """Empirical end-to-end success per slot under the model's protocol."""
    P = G.graph["params"]
    ok = np.ones(n_slots, dtype=bool)
    for (u, v, k), lev in zip(hops, levels):
        L = G.edges[u, v, k]["L"]
        eta = P.eta_det * 10 ** (-P.att_db_per_km * L / 10)
        ok &= rng.binomial(P.attempts, eta, n_slots) >= 2 ** lev
        F = stored(F_of_w(G.edges[u, v, k]["w0"] * math.exp(-L / P.l_dec_km)), L / P.c_km_s, P.t_coh_s)
        for j in range(lev):  # round j: 2^(lev-1-j) purifications, all must succeed
            g, F_next = bbpssw(F)
            ok &= (rng.random((n_slots, 2 ** (lev - 1 - j))) < g).all(axis=1)
            F = stored(F_next, L / P.c_km_s, P.t_coh_s)
    for _ in hops[1:]:  # swaps at the intermediate nodes
        ok &= rng.random(n_slots) < P.q_swap
    return ok.mean()


def simultaneous_start_F(G, hops, levels, F_jit):
    """Fidelity if all links start together: link e waits R - ready(e) after it is ready."""
    P = G.graph["params"]
    if not math.isfinite(P.t_coh_s):
        return F_jit
    ready = [(1 + lev) * G.edges[h]["L"] / P.c_km_s for h, lev in zip(hops, levels)]
    R = max(ready)
    extra = math.exp(-2.0 * sum(R - r for r in ready) / P.t_coh_s)
    return F_of_w(w_of_F(F_jit) * extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--method", default="FiSER")
    ap.add_argument("--mc-designs", type=int, default=40)
    ap.add_argument("--mc-slots", type=int, default=200_000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    for run in args.runs:
        meta = json.loads((Path(run) / "params.json").read_text())
        P = PhysParams(**meta["phys"])
        print(f"=== {run} (T = {P.t_coh_s} s)")
        for name in meta["topologies"]:
            G = build_network(topologies.load(name)["edges"], P, max_link_km=meta["max_link_km"])
            df = pd.read_csv(Path(run) / f"{name}.csv", dtype={f"lev{i}": str for i in range(1, 6)})
            df = df[(df.method == args.method) & df.served & (df.worst >= S_MIN)]
            n_paths = sum(1 for c in df.columns if c.startswith("nodes"))
            worst_rel, losses, miss, z = 0.0, [], 0, []
            for i, r in enumerate(df.itertuples()):
                Fs = []
                for m in range(1, n_paths + 1):
                    nodes = getattr(r, f"nodes{m}")
                    if not isinstance(nodes, str):
                        continue
                    hops = hops_of(nodes)
                    lev = tuple(int(c) for c in str(getattr(r, f"lev{m}")).zfill(len(hops)))
                    S, F = path_metrics(G, hops, lev)
                    worst_rel = max(worst_rel, abs(S - getattr(r, f"S{m}")) / getattr(r, f"S{m}"),
                                    abs(F - getattr(r, f"F{m}")))
                    Fsim = simultaneous_start_F(G, hops, lev, F)
                    losses.append(F - Fsim)
                    Fs.append(Fsim)
                    if i < args.mc_designs and m == 1:
                        emp = simulate(G, hops, lev, args.mc_slots, rng)
                        z.append(binomtest(round(emp * args.mc_slots), args.mc_slots, min(S, 1.0)).pvalue)
                miss += min(Fs) < r.F_th - 1e-12
            print(f"{name}: {len(df)} served designs | recompute max error {worst_rel:.1e} | "
                  f"MC on {len(z)} paths: min binomial p {min(z) if z else float('nan'):.3f} | "
                  f"simultaneous starts: fidelity loss mean {np.mean(losses):.4f}, max {np.max(losses):.4f}; "
                  f"{miss} designs ({100 * miss / max(len(df), 1):.1f}%) would miss F_th")


if __name__ == "__main__":
    main()
