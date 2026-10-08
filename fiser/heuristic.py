"""Fast primary-backup heuristic with scalarized (weighted-sum) candidate paths.

1. Candidate paths: K shortest paths under the scalarized link cost
   c_mu(e) = min_k [-log p_e^(k) - mu * log w_e^(k)]  (node cost -log q - mu * log lam)
   for a few multipliers mu; each candidate gets exact purification levels (Pareto DP)
   and is kept if it meets F_th.
2. Seed sets: the min-cost set of n node-disjoint paths (min-cost flow; Suurballe for n = 2)
   under each c_mu.
3. For each first-path candidate (best first), the remaining n - 1 paths are searched the
   same way in the residual graph without its intermediate nodes, links, and satellites
   (for n = 2: the backup is the best feasible candidate there).
4. Return the path set with the largest min_m S_m.

A satellite has one source, so a design uses at most one link per satellite: candidates
that reuse a satellite are dropped, backups avoid the primary's satellites, and Suurballe
seed pairs that reuse a satellite are discarded.
"""
from __future__ import annotations

import itertools
import math
import time

import networkx as nx

from .milp import PathChoice, Solution
from .paths import best_purification, path_metrics, path_nodes


def hop_graph(G, link_cost, node_cost=None, banned_nodes=(), banned_links=(), banned_sats=()):
    """Simple graph with one auxiliary node per link, so parallel links stay distinct paths.

    Each real node's cost is split over its two half-edges, so an intermediate node
    contributes it exactly once (the constant end-node halves do not change rankings).
    """
    node_cost = node_cost or (lambda n: -math.log(G.nodes[n]["q"]))
    H = nx.Graph()
    for u, v, k, data in G.edges(keys=True, data=True):
        if (u in banned_nodes or v in banned_nodes or (min(u, v), max(u, v), k) in banned_links
                or data.get("sat_id") in banned_sats):
            continue
        aux, c = ("link", u, v, k), link_cost(data)
        H.add_edge(u, aux, wt=(c + node_cost(u)) / 2)
        H.add_edge(aux, v, wt=(c + node_cost(v)) / 2)
    return H


def to_hops(hpath):
    """Map a hop-graph path [s, aux, n1, aux, ..., d] to hops (u, v, key)."""
    return tuple((hpath[i - 1], hpath[i + 1], hpath[i][3]) for i in range(1, len(hpath), 2))


def link_ids(hops):
    return {(min(u, v), max(u, v), k) for u, v, k in hops}


def sat_list(G, hops):
    return [sid for sid in (G.edges[h].get("sat_id") for h in hops) if sid is not None]


def sat_ids(G, hops):
    return set(sat_list(G, hops))


def one_link_per_satellite(G, *paths):
    sl = [sid for hops in paths for sid in sat_list(G, hops)]
    return len(sl) == len(set(sl))


def feasible_candidates(G, s, d, F_th, K, mus, banned_nodes=(), banned_links=(), banned_sats=()):
    """Fidelity-feasible candidate paths with optimal purification, best success first."""
    seen, out = set(), []
    for mu in mus:
        H = hop_graph(
            G,
            lambda e: min(-math.log(p) - mu * math.log(w) for p, w in e["opts"].values()),
            lambda n: -math.log(G.nodes[n]["q"]) - mu * math.log(G.nodes[n]["lam"]),
            banned_nodes, banned_links, banned_sats)
        if s not in H or d not in H or not nx.has_path(H, s, d):
            continue
        for hp in itertools.islice(nx.shortest_simple_paths(H, s, d, weight="wt"), K):
            hops = to_hops(hp)
            if hops in seen or not one_link_per_satellite(G, hops):
                continue
            seen.add(hops)
            bp = best_purification(G, hops, F_th)
            if bp is not None:
                out.append(PathChoice(hops, bp[1], bp[0], path_metrics(G, hops, bp[1])[1]))
    return sorted(out, key=lambda c: -c.S)


def mincost_disjoint_pair(G, s, d, mu, scale=1e6):
    """Suurballe-equivalent min-cost node-disjoint pair under the scalarized cost c_mu."""
    return mincost_disjoint_paths(G, s, d, mu, 2, scale)


def mincost_disjoint_paths(G, s, d, mu, n_paths=2, scale=1e6):
    """Min-cost set of n_paths node-disjoint paths under the scalarized cost c_mu.

    Min-cost flow of n_paths units on a node-split digraph: every intermediate node and every
    link is a unit-capacity arc carrying its cost, so the units are node- and link-disjoint.
    Returns n_paths hop tuples, or None if no such set exists.
    """
    D = nx.DiGraph()
    for n in G.nodes:
        cost = 0.0 if n in (s, d) else -math.log(G.nodes[n]["q"]) - mu * math.log(G.nodes[n]["lam"])
        D.add_edge((n, "i"), (n, "o"), capacity=n_paths if n in (s, d) else 1, weight=round(scale * cost))
    for u, v, k, e in G.edges(keys=True, data=True):
        a = ("link", u, v, k)
        c = min(-math.log(p) - mu * math.log(w) for p, w in e["opts"].values())
        D.add_edge((a, "i"), (a, "o"), capacity=1, weight=round(scale * c))
        for x, y in ((u, v), (v, u)):
            D.add_edge((x, "o"), (a, "i"), capacity=1, weight=0)
            D.add_edge((a, "o"), (y, "i"), capacity=1, weight=0)
    D.nodes[(s, "o")]["demand"], D.nodes[(d, "i")]["demand"] = -n_paths, n_paths
    try:
        flow = nx.min_cost_flow(D)
    except nx.NetworkXUnfeasible:
        return None
    paths = []
    for _ in range(n_paths):
        hops, node = [], s
        while node != d:
            a = next(t for t, f in flow[(node, "o")].items() if f > 0)
            flow[(node, "o")][a] -= 1
            nxt = next(t for t, f in flow[(a[0], "o")].items() if f > 0)
            flow[(a[0], "o")][nxt] -= 1
            hops.append((node, nxt[0], a[0][3]))
            node = nxt[0]
        paths.append(tuple(hops))
    return paths


def _pair_value(G, pair, F_th):
    out = []
    for hops in pair:
        bp = best_purification(G, hops, F_th)
        if bp is None:
            return None
        out.append(PathChoice(hops, bp[1], bp[0], path_metrics(G, hops, bp[1])[1]))
    return min(p.S for p in out), out


def primary_backup(G, s, d, F_th, K1=20, K2=10, mus=(0.0, 2.0, 8.0)):
    """Primary-backup pair (two disjoint paths); see disjoint_paths."""
    return disjoint_paths(G, s, d, F_th, 2, K1, K2, mus)


def disjoint_paths(G, s, d, F_th, n_paths=2, K1=20, K2=10, mus=(0.0, 2.0, 8.0)):
    """n_paths node-, link-, and satellite-disjoint fidelity-feasible paths maximizing min_m S_m.

    K1 candidates are tried for the first path and K2 for every later one.
    """
    t0 = time.perf_counter()
    best = None
    # min-cost disjoint sets avoid trap topologies, where the best first path blocks every good rest
    for mu in mus:
        paths = mincost_disjoint_paths(G, s, d, mu, n_paths)
        if paths is None or not one_link_per_satellite(G, *paths):
            continue
        val = _pair_value(G, paths, F_th)
        if val and (best is None or val[0] > best[0]):
            best = val
    found = _search(G, s, d, F_th, n_paths, K1, K2, mus, (frozenset(), frozenset(), frozenset()),
                    best[0] if best else None)
    if found and (best is None or found[0] > best[0]):
        best = found
    runtime = time.perf_counter() - t0
    return Solution("Optimal", best[1], runtime) if best else Solution("Infeasible", runtime=runtime)


def _search(G, s, d, F_th, k, K, K_rest, mus, banned, floor):
    """Best (value, paths) for k disjoint paths by first-path-first search, or None.

    Only results above floor matter: candidates are sorted by success, so the scan stops at
    the first candidate that cannot beat max(floor, incumbent).
    """
    nodes, links, sats = banned
    cands = feasible_candidates(G, s, d, F_th, K, mus, nodes, links, sats)
    if k == 1:
        return (cands[0].S, [cands[0]]) if cands else None
    best = None
    for c1 in cands:
        bar = max(floor or 0.0, best[0] if best else 0.0)
        if c1.S <= bar:
            break  # no later first path can do better
        rest = _search(G, s, d, F_th, k - 1, K_rest, K_rest, mus,
                       (nodes | set(path_nodes(c1.hops)[1:-1]), links | link_ids(c1.hops),
                        sats | sat_ids(G, c1.hops)), bar)
        if rest and min(c1.S, rest[0]) > bar:
            best = (min(c1.S, rest[0]), [c1] + rest[1])
    return best
