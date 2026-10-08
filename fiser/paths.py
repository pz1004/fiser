"""Path representation, end-to-end metrics, and exact per-path purification.

A path is a tuple of hops (u, v, key) on the network MultiGraph; levels[i] is the
purification level used on hop i.
"""
from __future__ import annotations

import math

from .physics import F_of_w, w_of_F


def path_nodes(hops):
    return [hops[0][0]] + [v for _, v, _ in hops]


def end_credit(G, hops):
    """Log-Werner credit for the end qubits' swap-signaling wait (0 without memory decoherence).

    The option tables charge every link exp(-2 tau/T), as if both end qubits waited for the whole
    path's signaling. Exactly, s waits for the farthest swap outcome, from the last intermediate node
    (sum of tau minus the last link's), and d symmetrically; a one-hop path has no swap and no wait.
    The first and last links are thus over-charged by exp(-tau/T) each (one link: exp(-2 tau/T)).
    """
    params = G.graph.get("params")
    T = getattr(params, "t_coh_s", math.inf)
    if not math.isfinite(T):
        return 0.0
    return (G.edges[hops[0]].get("tau", 0.0) + G.edges[hops[-1]].get("tau", 0.0)) / T


def path_metrics(G, hops, levels):
    """(success probability, fidelity) of a path with the given purification levels."""
    S = W = 1.0
    for (u, v, k), lev in zip(hops, levels):
        p, w = G.edges[u, v, k]["opts"][lev]
        S, W = S * p, W * w
    for n in path_nodes(hops)[1:-1]:
        S, W = S * G.nodes[n]["q"], W * G.nodes[n]["lam"]
    return S, F_of_w(W * math.exp(end_credit(G, hops)))


def best_purification(G, hops, F_th, levels=None):
    """Exact max-success purification levels for a FIXED path s.t. fidelity >= F_th.

    Pareto-front DP over hops with state (sum log w, sum log p). Because log w <= 0,
    a state below the fidelity budget can never recover and is dropped.
    Returns (S, levels) or None if no assignment meets F_th.
    """
    inner = path_nodes(hops)[1:-1]
    log_q = sum(math.log(G.nodes[n]["q"]) for n in inner)
    budget = math.log(w_of_F(F_th)) - sum(math.log(G.nodes[n]["lam"]) for n in inner)
    front = [(end_credit(G, hops), 0.0, ())]
    for u, v, k in hops:
        opts = G.edges[u, v, k]["opts"]
        new = [(lw + math.log(w), lp + math.log(p), ks + (lev,))
               for lw, lp, ks in front for lev, (p, w) in opts.items()
               if levels is None or lev in levels]
        new = sorted((st for st in new if st[0] >= budget - 1e-12), key=lambda st: (-st[0], -st[1]))
        front, best_lp = [], -math.inf
        for st in new:
            if st[1] > best_lp + 1e-15:
                front.append(st)
                best_lp = st[1]
        if not front:
            return None
    _, lp, ks = max(front, key=lambda st: st[1])
    return math.exp(lp + log_q), ks


def memory_use(paths):
    """{node: qubits} held by a design: a hop at purification level k holds 2^k qubits at each end."""
    use = {}
    for pc in paths:
        for (u, v, _), lev in zip(pc.hops, pc.levels):
            use[u] = use.get(u, 0) + 2 ** lev
            use[v] = use.get(v, 0) + 2 ** lev
    return use
