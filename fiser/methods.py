"""Proposed design and baselines, evaluated on a common footing.

B1  survivable, fidelity-unaware (the EPSPF model of Oki et al.): max-min success, no purification
B2  B1 paths + optimal post-hoc purification per path (two-stage)
B3  min-hop node-disjoint pair (Suurballe-equivalent) + optimal post-hoc purification
B4  fidelity-aware survivable design without purification
B5  sequential: best fidelity-feasible purified primary (exact), then the best purified backup in
    the residual graph without its intermediate nodes, links, and satellites (exact); with
    n_paths > 2, further paths are added the same way
B6  single best fidelity-feasible path with purification (non-survivable reference)
B7  Oki et al.'s polynomial MaxSPA-ND design (min-sum disjoint set on -ln p - ln q, i.e. the
    heuristic's mu = 0 seed) + optimal post-hoc purification per path (two-stage); not in ALL
H   FiSER heuristic (primary-backup search over scalarized-cost candidates + exact purification DP)
FiSER  proposed joint MILP
FiSER-W  FiSER with weather diversity: at most one path uses satellite links (space-ground only)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from . import heuristic, milp
from .milp import PathChoice
from .paths import best_purification, path_metrics, path_nodes

ALL = ("B1", "B2", "B3", "B4", "B5", "B6", "H", "FiSER")


@dataclass
class Outcome:
    served: bool    # a fidelity-feasible design that survives any single intermediate-node failure
    worst: float    # worst-case success probability over single-node failures (0 if not served)
    paths: list
    runtime: float


def _as_is(sol, F_th):
    ok = sol.ok and all(p.F >= F_th - 1e-9 for p in sol.paths)
    return Outcome(ok, min(p.S for p in sol.paths) if ok else 0.0, sol.paths, sol.runtime)


def _post_purify(G, sol, F_th):
    t0 = time.perf_counter()
    if not sol.ok:
        return Outcome(False, 0.0, [], sol.runtime)
    out = []
    for pc in sol.paths:
        bp = best_purification(G, pc.hops, F_th)
        if bp is None:
            return Outcome(False, 0.0, sol.paths, sol.runtime + time.perf_counter() - t0)
        out.append(PathChoice(pc.hops, bp[1], bp[0], path_metrics(G, pc.hops, bp[1])[1]))
    return Outcome(True, min(p.S for p in out), out, sol.runtime + time.perf_counter() - t0)


def _two_step(G, s, d, F_th):
    return _sequential(G, s, d, F_th, 2)


def _sequential(G, s, d, F_th, n_paths):
    t0 = time.perf_counter()
    H, chosen = G, []
    for _ in range(n_paths):
        sol = milp.solve(H, s, d, F_th, n_paths=1)
        if not sol.ok:
            return Outcome(False, 0.0, chosen, time.perf_counter() - t0)
        p = sol.paths[0]
        chosen.append(p)
        used_sats = {H.edges[h].get("sat_id") for h in p.hops} - {None}
        H = H.copy()
        H.remove_nodes_from(path_nodes(p.hops)[1:-1])
        H.remove_edges_from([(u, v, k) for u, v, k, sid in H.edges(keys=True, data="sat_id")
                             if sid in used_sats or (u, v, k) in p.hops or (v, u, k) in p.hops])
    return Outcome(True, min(p.S for p in chosen), chosen, time.perf_counter() - t0)


def run_pair(G, s, d, F_ths, methods=ALL, K=(20, 10), n_paths=2):
    """{F_th: {method: Outcome}} for one SD pair; F_th-independent designs are solved once.

    n_paths disjoint paths protect against n_paths - 1 failures (B6 is always single-path).
    """
    need = set(methods)
    n = n_paths
    b1 = milp.solve(G, s, d, None, n_paths=n, levels=(0,)) if need & {"B1", "B2"} else None
    b3 = milp.solve(G, s, d, None, n_paths=n, levels=(0,), objective="minhop") if "B3" in need else None
    res = {}
    for F_th in F_ths:
        r = {}
        if "B1" in need:
            r["B1"] = _as_is(b1, F_th)
        if "B2" in need:
            r["B2"] = _post_purify(G, b1, F_th)
        if "B3" in need:
            r["B3"] = _post_purify(G, b3, F_th)
        if "B4" in need:
            r["B4"] = _as_is(milp.solve(G, s, d, F_th, n_paths=n, levels=(0,)), F_th)
        if "B5" in need:
            r["B5"] = _sequential(G, s, d, F_th, n)
        if "B7" in need:
            t0 = time.perf_counter()
            hops = heuristic.mincost_disjoint_paths(G, s, d, 0.0, n)
            if hops is None or not heuristic.one_link_per_satellite(G, *hops):
                r["B7"] = Outcome(False, 0.0, [], time.perf_counter() - t0)
            else:
                levels = [tuple(0 for _ in h) for h in hops]
                paths = [PathChoice(h, lv, *path_metrics(G, h, lv)) for h, lv in zip(hops, levels)]
                r["B7"] = _post_purify(G, milp.Solution("Optimal", paths, time.perf_counter() - t0), F_th)
        if "B6" in need:
            sol = milp.solve(G, s, d, F_th, n_paths=1)
            ok = sol.ok and len(sol.paths[0].hops) == 1  # any intermediate-node failure breaks it
            r["B6"] = Outcome(ok, sol.paths[0].S if ok else 0.0, sol.paths, sol.runtime)
        if "H" in need:
            r["H"] = _as_is(heuristic.disjoint_paths(G, s, d, F_th, n, K1=K[0], K2=K[1]), F_th)
        if "FiSER" in need:
            r["FiSER"] = _as_is(milp.solve(G, s, d, F_th, n_paths=n), F_th)
        if "FiSER-W" in need:
            r["FiSER-W"] = _as_is(milp.solve(G, s, d, F_th, n_paths=n, sat_diverse=True), F_th)
        res[F_th] = r
    return res
