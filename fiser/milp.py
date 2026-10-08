"""Arc-based MILP for fidelity-guaranteed survivable entanglement provisioning.

For a pair (s, d), choose n_paths node-disjoint paths and, on every hop, one
purification level. With y[m, a, k] = 1 iff path m uses arc a at level k:

  flow conservation per path; in-degree <= 1 per node and path; no path enters s;
  every physical link is used at most once over all paths and directions;
  node-disjointness at intermediate nodes;
  fidelity:  sum log w + sum_{swaps} log lam >= log W_th          (per path)
  success:   LS_m = sum log p + sum_{swaps} log q
  optional weather diversity: at most one path uses satellite links (by symmetry,
  paths 2..n are restricted to fiber links)
  optional memory: a link used at level k holds 2^k qubits at each end node, so
  sum over the design of 2^k at node v <= mem[v]
  optional fixed paths: path m may use only the arcs of fixed_paths[m] (levels still chosen)

Objectives are solved lexicographically: the primary objective first, then the
total log-success as a tie-break (with the primary fixed at its optimum).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import pulp

from .paths import path_metrics
from .physics import w_of_F

# Objectives are log-success values; scaling them puts near-ties (e.g., two paths whose log-success
# differs by 1e-7) far above HiGHS's LP tolerances, so the true optimum is returned whatever the
# input order. Without it, a greedy baseline (B5) could pick either of two near-tied paths.
OBJ_SCALE = 1e4


@dataclass(frozen=True)
class PathChoice:
    hops: tuple
    levels: tuple
    S: float
    F: float


@dataclass
class Solution:
    status: str
    paths: list = field(default_factory=list)
    runtime: float = 0.0

    @property
    def ok(self):
        return self.status == "Optimal"


def solve(G, s, d, F_th=None, n_paths=2, levels=None, objective="maxmin",
          threads=1, time_limit=None, gap_rel=1e-9, sat_diverse=False, mem=None, fixed_paths=None,
          s_floor=None, node_weight=None):
    """Optimal path set. F_th=None drops the fidelity constraint.

    objective: 'maxmin' maximizes min_m log S_m; 'minhop' minimizes the total hop count;
    'minmem' minimizes the qubits held (sum of 2^(k+1) over hops), then maximizes total log-success;
    with node_weight ({node: weight}), qubits held at node v are weighted by node_weight[v]
    (e.g., scarcity weights for load-balanced provisioning).
    s_floor: every path must reach success s_floor (e.g., the service floor).
    levels restricts the purification levels (e.g. (0,) = no purification).
    sat_diverse: at most one path may use satellite links, so that weather over the OGS
    region (a common-mode risk for all satellite links) cannot cut every path.
    mem: {node: available qubits}; a link at level k holds 2^k qubits at each of its end nodes.
    fixed_paths: one hop tuple per path; path m is restricted to those arcs (best levels only).
    """
    t0 = time.perf_counter()
    lv = tuple(range(G.graph["params"].max_level + 1)) if levels is None else tuple(levels)
    links = list(G.edges(keys=True))
    arcs = links + [(v, u, k) for u, v, k in links]
    aid = {a: i for i, a in enumerate(arcs)}
    in_arcs, out_arcs = {n: [] for n in G}, {n: [] for n in G}
    for a in arcs:
        out_arcs[a[0]].append(a)
        in_arcs[a[1]].append(a)
    M = range(n_paths)
    inner = [n for n in G if n not in (s, d)]
    T = getattr(G.graph.get("params"), "t_coh_s", math.inf)

    # weather diversity: paths are interchangeable, so "at most one path on satellite links"
    # is exactly "every path but the first stays on fiber"
    fiber_only = set(range(1, n_paths)) if sat_diverse else set()
    prob = pulp.LpProblem("fiser", pulp.LpMaximize)
    allowed = None if fixed_paths is None else [set(hops) for hops in fixed_paths]
    y = {(m, a, lev): pulp.LpVariable(f"y_{m}_{aid[a]}_{lev}", cat="Binary")
         for m in M for a in arcs for lev in lv if lev in G.edges[a]["opts"]
         and not (m in fiber_only and G.edges[a].get("sat_id") is not None)
         and (allowed is None or a in allowed[m])}
    x = {(m, a): pulp.lpSum(y[m, a, lev] for lev in lv if (m, a, lev) in y) for m in M for a in arcs}

    def inflow(m, n):
        return pulp.lpSum(x[m, a] for a in in_arcs[n])

    for m in M:
        for n in G:
            out = pulp.lpSum(x[m, a] for a in out_arcs[n])
            prob += out - inflow(m, n) == (1 if n == s else -1 if n == d else 0)
            prob += inflow(m, n) <= 1
        prob += inflow(m, s) == 0
    for u, v, k in links:
        prob += pulp.lpSum(x[m, (u, v, k)] + x[m, (v, u, k)] for m in M) <= 1
    if n_paths > 1:
        for n in inner:
            prob += pulp.lpSum(inflow(m, n) for m in M) <= 1
    # a satellite has one source: at most one of its links in the whole design, which
    # also makes it a shared failure point that primary and backup cannot both rely on
    sats = {}
    for u, v, k, sid in G.edges(keys=True, data="sat_id"):
        if sid is not None:
            sats.setdefault(sid, []).append((u, v, k))
    for sat_links in sats.values():
        if len(sat_links) > 1 or n_paths > 1:
            prob += pulp.lpSum(x[m, (u, v, k)] + x[m, (v, u, k)] for m in M for u, v, k in sat_links) <= 1

    if mem is not None:
        use = {n: [] for n in G}
        for (m, a, lev), var in y.items():
            use[a[0]].append((2 ** lev) * var)
            use[a[1]].append((2 ** lev) * var)
        for n, terms in use.items():
            if n in mem and terms:
                prob += pulp.lpSum(terms) <= mem[n]

    LS = {}
    for m in M:
        terms = [(var, G.edges[a]["opts"][lev]) for (mm, a, lev), var in y.items() if mm == m]
        LS[m] = (pulp.lpSum(var * math.log(p) for var, (p, _) in terms)
                 + pulp.lpSum(inflow(m, n) * math.log(G.nodes[n]["q"]) for n in inner))
        if F_th is not None:
            LW = (pulp.lpSum(var * math.log(w) for var, (_, w) in terms)
                  + pulp.lpSum(inflow(m, n) * math.log(G.nodes[n]["lam"]) for n in inner))
            if math.isfinite(T):  # exact end-qubit wait: credit tau/T on the first and last hop (paths.end_credit)
                LW += pulp.lpSum(x[m, a] * G.edges[a].get("tau", 0.0) / T for a in out_arcs[s] + in_arcs[d])
            prob += LW >= math.log(w_of_F(F_th))

    if s_floor is not None:
        for m in M:
            prob += LS[m] >= math.log(s_floor)
    if objective == "maxmin":
        primary = pulp.LpVariable("t", upBound=0)
        for m in M:
            prob += primary <= LS[m]
    elif objective == "minhop":
        primary = -pulp.lpSum(x.values())
    elif objective == "minmem":
        wt = node_weight or {}
        primary = -pulp.lpSum((2 ** lev) * (wt.get(a[0], 1.0) + wt.get(a[1], 1.0)) * var
                              for (m, a, lev), var in y.items())
    else:
        raise ValueError(objective)

    solver_kw = dict(msg=False, threads=threads, gapRel=gap_rel, timeLimit=time_limit)
    prob.setObjective(OBJ_SCALE * primary)
    _run(prob, solver_kw)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return Solution(status, runtime=time.perf_counter() - t0)
    paths = _extract(G, y, s, d, n_paths)
    if n_paths > 1 or objective != "maxmin":
        best = pulp.value(primary)
        prob += primary >= best - 1e-7 * max(1.0, abs(best))
        prob.setObjective(OBJ_SCALE * pulp.lpSum(LS.values()))
        _run(prob, solver_kw)
        if pulp.LpStatus[prob.status] == "Optimal":
            paths = _extract(G, y, s, d, n_paths)
    return Solution("Optimal", paths, time.perf_counter() - t0)


def _run(prob, solver_kw):
    """Solve with HiGHS; on a rare HiGHS MIP-presolve crash ('vector::reserve', triggered by
    the column order PuLP passes), retry the same exact solve with presolve off."""
    try:
        prob.solve(pulp.HiGHS(**solver_kw))
    except ValueError:
        prob.solve(pulp.HiGHS(**solver_kw, presolve="off"))


def _extract(G, y, s, d, n_paths):
    paths = []
    for m in range(n_paths):
        nxt = {a[0]: (a, lev) for (mm, a, lev), var in y.items()
               if mm == m and (var.value() or 0) > 0.5}
        hops, levels, n = [], [], s
        while n != d:
            if n not in nxt or len(hops) > G.number_of_nodes():
                raise RuntimeError(f"malformed path {m} for ({s}, {d})")
            a, lev = nxt[n]
            hops.append(a)
            levels.append(lev)
            n = a[1]
        S, F = path_metrics(G, hops, levels)
        paths.append(PathChoice(tuple(hops), tuple(levels), S, F))
    return paths
