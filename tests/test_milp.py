"""Exactness tests: DP and MILP against exhaustive enumeration on small graphs."""
import itertools
import math
import random
from dataclasses import replace

import networkx as nx
import pytest

from fiser import heuristic, milp
from fiser.network import add_link, build_network
from fiser.paths import best_purification, path_metrics, path_nodes
from fiser.physics import PhysParams, recurrence_options
from fiser.topologies import NSFNET


def all_paths(G, s, d):
    H = heuristic.hop_graph(G, lambda e: 1.0)
    return [heuristic.to_hops(p) for p in nx.all_simple_paths(H, s, d)]


def brute_value(G, hops, F_th, levels):
    """Best success of a fixed path by exhaustive enumeration of per-hop levels."""
    best = None
    for ks in itertools.product(*[[k for k in G.edges[h]["opts"] if k in levels] for h in hops]):
        S, F = path_metrics(G, hops, ks)
        if (F_th is None or F >= F_th - 1e-12) and (best is None or S > best):
            best = S
    return best


def uses_sat(G, hops):
    return any(G.edges[h].get("sat_id") is not None for h in hops)


def brute_pair(G, s, d, F_th, levels, sat_diverse=False):
    vals = {hops: brute_value(G, hops, F_th, levels) for hops in all_paths(G, s, d)
            if heuristic.one_link_per_satellite(G, hops)}
    vals = {h: v for h, v in vals.items() if v is not None}
    best = None
    for h1, h2 in itertools.combinations(vals, 2):
        if set(path_nodes(h1)[1:-1]) & set(path_nodes(h2)[1:-1]):
            continue
        if heuristic.link_ids(h1) & heuristic.link_ids(h2):
            continue
        if not heuristic.one_link_per_satellite(G, h1, h2):
            continue
        if sat_diverse and uses_sat(G, h1) and uses_sat(G, h2):
            continue
        v = min(vals[h1], vals[h2])
        best = v if best is None else max(best, v)
    return best


def random_network(seed, n=8, P=PhysParams()):
    rng = random.Random(seed)
    while True:
        g = nx.random_regular_graph(3, n, seed=rng.randrange(10 ** 6))
        for _ in range(2):
            u, v = rng.sample(range(n), 2)
            g.add_edge(u, v)
        if nx.is_biconnected(g):
            break
    edges = [(u, v, rng.uniform(10, 100)) for u, v in g.edges()]
    return build_network(edges, P, scale=1.0)


def test_dp_matches_bruteforce_on_nsfnet_paths():
    G = build_network(NSFNET["edges"], PhysParams(), max_link_km=100)
    rng = random.Random(0)
    for s, d in rng.sample(list(itertools.combinations(range(14), 2)), 15):
        for hops in all_paths(G, s, d)[:8]:
            if len(hops) > 7:
                continue
            for F_th in (0.7, 0.8, 0.85, 0.9):
                dp = best_purification(G, hops, F_th)
                bf = brute_value(G, hops, F_th, (0, 1, 2))
                assert (dp is None) == (bf is None)
                if dp:
                    assert dp[0] == pytest.approx(bf, rel=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_milp_matches_bruteforce(seed):
    G = random_network(seed)
    for s, d in itertools.combinations(sorted(G.nodes), 2):
        for F_th, levels in ((None, (0,)), (0.8, (0,)), (0.8, (0, 1, 2)), (0.9, (0, 1, 2))):
            sol = milp.solve(G, s, d, F_th, levels=levels)
            bf = brute_pair(G, s, d, F_th, levels)
            if bf is None:
                assert not sol.ok
                continue
            assert sol.ok
            assert min(p.S for p in sol.paths) == pytest.approx(bf, rel=1e-6)
            p1, p2 = sol.paths
            assert not set(path_nodes(p1.hops)[1:-1]) & set(path_nodes(p2.hops)[1:-1])
            if F_th is not None:
                assert min(p.F for p in sol.paths) >= F_th - 1e-9


@pytest.mark.parametrize("seed", [1, 2])
def test_milp_matches_bruteforce_with_decoherence(seed):
    # At finite T, the fidelity constraint credits the end-qubit wait on the first and last link
    # (a direct s-d link twice); the MILP must match enumeration with the exact path fidelity.
    G = random_network(seed, P=replace(PhysParams(), t_coh_s=0.05))
    direct = 0
    for s, d in itertools.combinations(sorted(G.nodes), 2):
        for F_th in (0.75, 0.8, 0.85):
            sol, bf = milp.solve(G, s, d, F_th), brute_pair(G, s, d, F_th, (0, 1, 2))
            assert (bf is None) == (not sol.ok)
            if bf is not None:
                assert min(p.S for p in sol.paths) == pytest.approx(bf, rel=1e-6)
                assert min(p.F for p in sol.paths) >= F_th - 1e-9
                direct += any(len(p.hops) == 1 for p in sol.paths)
    assert direct > 0  # direct s-d links are exercised


def test_minhop_matches_bruteforce():
    G = random_network(7)
    for s, d in itertools.combinations(sorted(G.nodes), 2):
        sol = milp.solve(G, s, d, None, levels=(0,), objective="minhop")
        best = min(len(h1) + len(h2) for h1, h2 in itertools.combinations(all_paths(G, s, d), 2)
                   if not set(path_nodes(h1)[1:-1]) & set(path_nodes(h2)[1:-1])
                   and not heuristic.link_ids(h1) & heuristic.link_ids(h2))
        assert sum(len(p.hops) for p in sol.paths) == best


def test_parallel_links_are_distinct_resources():
    G = build_network([(0, 1, 40), (0, 2, 30), (2, 1, 30)], PhysParams(), scale=1.0)
    add_link(G, 0, 1, "sat", recurrence_options(0.95, lambda n: 0.3 if n == 1 else 0.05, 2))
    sol = milp.solve(G, 0, 1, 0.85)
    bf = brute_pair(G, 0, 1, 0.85, (0, 1, 2))
    assert sol.ok and min(p.S for p in sol.paths) == pytest.approx(bf, rel=1e-6)
    assert not heuristic.link_ids(sol.paths[0].hops) & heuristic.link_ids(sol.paths[1].hops)


@pytest.mark.parametrize("shared_satellite", [False, True])
def test_one_link_per_satellite(shared_satellite):
    # Ring 0..5 plus satellite links 0-3, 3-5 and 1-4; the first two may come from one satellite,
    # so a path 0-3-5 over two links of the same satellite must be rejected.
    G = build_network([(i, (i + 1) % 6, 60) for i in range(6)], PhysParams(), scale=1.0)
    sat_opts = recurrence_options(0.97, lambda n: 0.4 if n == 1 else 0.1, 2)
    add_link(G, 0, 3, "s:a", sat_opts, sat_id="A")
    add_link(G, 3, 5, "s:b", sat_opts, sat_id="A" if shared_satellite else "B")
    add_link(G, 1, 4, "s:c", sat_opts, sat_id="C")
    for s, d in itertools.combinations(range(6), 2):
        for F_th in (0.8, 0.9):
            sol, bf = milp.solve(G, s, d, F_th), brute_pair(G, s, d, F_th, (0, 1, 2))
            assert (bf is None) == (not sol.ok)
            if bf is not None:
                assert min(p.S for p in sol.paths) == pytest.approx(bf, rel=1e-6)
                assert heuristic.one_link_per_satellite(G, *(p.hops for p in sol.paths))
            h = heuristic.primary_backup(G, s, d, F_th)
            if h.ok:
                assert heuristic.one_link_per_satellite(G, *(p.hops for p in h.paths))


def test_weather_diverse_design():
    # Ring 0..5 with two satellite links from different satellites; without weather diversity the
    # best pair for some requests puts both paths on satellites, with it at most one path may.
    G = build_network([(i, (i + 1) % 6, 60) for i in range(6)], PhysParams(), scale=1.0)
    sat_opts = recurrence_options(0.97, lambda n: 0.4 if n == 1 else 0.1, 2)
    add_link(G, 0, 3, "s:a", sat_opts, sat_id="A")
    add_link(G, 0, 3, "s:b", sat_opts, sat_id="B")
    add_link(G, 1, 4, "s:c", sat_opts, sat_id="C")
    both_on_sat = 0
    for s, d in itertools.combinations(range(6), 2):
        for F_th in (0.8, 0.9):
            free = milp.solve(G, s, d, F_th)
            both_on_sat += free.ok and all(uses_sat(G, p.hops) for p in free.paths)
            sol, bf = milp.solve(G, s, d, F_th, sat_diverse=True), brute_pair(G, s, d, F_th, (0, 1, 2), True)
            assert (bf is None) == (not sol.ok)
            if bf is not None:
                assert min(p.S for p in sol.paths) == pytest.approx(bf, rel=1e-6)
                assert sum(uses_sat(G, p.hops) for p in sol.paths) <= 1
    assert both_on_sat > 0  # the constraint is exercised


def brute_triple(G, s, d, F_th, levels=(0, 1, 2)):
    """Best min-success over three mutually node- and link-disjoint paths (two-failure protection)."""
    vals = {h: brute_value(G, h, F_th, levels) for h in all_paths(G, s, d)}
    hs = [h for h, v in vals.items() if v is not None]

    def disjoint(a, b):
        return not (set(path_nodes(a)[1:-1]) & set(path_nodes(b)[1:-1])) and \
            not (heuristic.link_ids(a) & heuristic.link_ids(b))
    best = None
    for a, b in itertools.combinations(hs, 2):
        if not disjoint(a, b):
            continue
        for c in hs:
            if c in (a, b) or not (disjoint(a, c) and disjoint(b, c)):
                continue
            v = min(vals[a], vals[b], vals[c])
            best = v if best is None else max(best, v)
    return best


@pytest.mark.parametrize("seed", [0, 3])
def test_three_path_milp_matches_bruteforce(seed):
    rng = random.Random(seed)
    while True:
        g = nx.random_regular_graph(3, 8, seed=rng.randrange(10 ** 6))
        for _ in range(3):
            u, v = rng.sample(range(8), 2)
            g.add_edge(u, v)
        if nx.node_connectivity(g) >= 3:
            break
    G = build_network([(u, v, rng.uniform(10, 60)) for u, v in g.edges()], PhysParams(), scale=1.0)
    for s, d in rng.sample(list(itertools.combinations(range(8), 2)), 4):
        for F_th in (0.8, 0.85):
            sol, bf = milp.solve(G, s, d, F_th, n_paths=3), brute_triple(G, s, d, F_th)
            assert (bf is None) == (not sol.ok)
            if bf is not None:
                assert min(p.S for p in sol.paths) == pytest.approx(bf, rel=1e-6)


def brute_pair_mem(G, s, d, F_th, mem, levels, pair=None):
    """Best min-success pair under per-node memory (2^k qubits per hop end), by enumeration."""
    from fiser.milp import PathChoice
    from fiser.paths import memory_use
    opts = {}
    for h in ([pair[0], pair[1]] if pair else all_paths(G, s, d)):
        L = []
        for ks in itertools.product(*[[k for k in G.edges[x]["opts"] if k in levels] for x in h]):
            S, F = path_metrics(G, h, ks)
            if F >= F_th - 1e-12:
                L.append(PathChoice(h, ks, S, F))
        if L:
            opts[h] = L
    best = None
    for h1, h2 in itertools.combinations(opts, 2):
        if set(path_nodes(h1)[1:-1]) & set(path_nodes(h2)[1:-1]) or heuristic.link_ids(h1) & heuristic.link_ids(h2):
            continue
        for a in opts[h1]:
            for b in opts[h2]:
                if best is not None and min(a.S, b.S) <= best:
                    continue
                use = memory_use([a, b])
                if all(q <= mem.get(n, float("inf")) for n, q in use.items()):
                    best = min(a.S, b.S)
    return best


@pytest.mark.parametrize("seed", [1, 2])
def test_memory_constrained_milp_matches_bruteforce(seed):
    from fiser.paths import memory_use
    rng = random.Random(seed)
    G = random_network(seed, n=6)
    for s, d in rng.sample(list(itertools.combinations(sorted(G.nodes), 2)), 6):
        for F_th in (0.8, 0.85):
            mem = {n: rng.choice([2, 3, 4, 6]) for n in G}
            sol = milp.solve(G, s, d, F_th, levels=(0, 1), mem=mem)
            bf = brute_pair_mem(G, s, d, F_th, mem, (0, 1))
            assert (bf is None) == (not sol.ok)
            if bf is not None:
                assert min(p.S for p in sol.paths) == pytest.approx(bf, rel=1e-6)
                assert all(q <= mem[n] for n, q in memory_use(sol.paths).items())
                # fixed paths: the best levels for the MILP's own paths under the same memory
                fixed = milp.solve(G, s, d, F_th, levels=(0, 1), mem=mem,
                                   fixed_paths=[p.hops for p in sol.paths])
                bf2 = brute_pair_mem(G, s, d, F_th, mem, (0, 1), pair=[p.hops for p in sol.paths])
                assert fixed.ok and min(p.S for p in fixed.paths) == pytest.approx(bf2, rel=1e-6)


@pytest.mark.parametrize("seed", [4, 5])
def test_heuristic_is_valid_and_bounded_by_milp(seed):
    G = random_network(seed)
    for s, d in itertools.combinations(sorted(G.nodes), 2):
        for F_th in (0.75, 0.85):
            h, m = heuristic.primary_backup(G, s, d, F_th), milp.solve(G, s, d, F_th)
            if h.ok:
                p1, p2 = h.paths
                assert m.ok and min(p1.S, p2.S) <= min(p.S for p in m.paths) * (1 + 1e-6)
                assert min(p1.F, p2.F) >= F_th - 1e-9
                assert not set(path_nodes(p1.hops)[1:-1]) & set(path_nodes(p2.hops)[1:-1])
                assert not heuristic.link_ids(p1.hops) & heuristic.link_ids(p2.hops)


@pytest.mark.parametrize("seed", [0, 3])
def test_three_path_heuristic_and_sequential_are_valid_and_bounded(seed):
    from fiser import methods
    rng = random.Random(seed)
    while True:
        g = nx.random_regular_graph(3, 8, seed=rng.randrange(10 ** 6))
        for _ in range(3):
            u, v = rng.sample(range(8), 2)
            g.add_edge(u, v)
        if nx.node_connectivity(g) >= 3:
            break
    G = build_network([(u, v, rng.uniform(10, 60)) for u, v in g.edges()], PhysParams(), scale=1.0)
    for s, d in itertools.combinations(range(8), 2):
        res = methods.run_pair(G, s, d, [0.8, 0.85], methods=("H", "B5", "FiSER"), n_paths=3)
        for F_th, r in res.items():
            for name in ("H", "B5"):
                o = r[name]
                if o.served:
                    assert len(o.paths) == 3 and r["FiSER"].served
                    assert o.worst <= r["FiSER"].worst * (1 + 1e-6)
                    assert min(p.F for p in o.paths) >= F_th - 1e-9
                    for a, b in itertools.combinations(o.paths, 2):
                        assert not set(path_nodes(a.hops)[1:-1]) & set(path_nodes(b.hops)[1:-1])
                        assert not heuristic.link_ids(a.hops) & heuristic.link_ids(b.hops)


@pytest.mark.parametrize("seed", [2, 6])
def test_two_step_baseline_is_valid_and_bounded_by_milp(seed):
    from fiser import methods
    G = random_network(seed)
    for s, d in itertools.combinations(sorted(G.nodes), 2):
        res = methods.run_pair(G, s, d, [0.8, 0.9], methods=("B5", "FiSER"))
        for F_th, r in res.items():
            b5, fi = r["B5"], r["FiSER"]
            if b5.served:
                p1, p2 = b5.paths
                assert fi.served and b5.worst <= fi.worst * (1 + 1e-6)
                assert min(p1.F, p2.F) >= F_th - 1e-9
                assert not set(path_nodes(p1.hops)[1:-1]) & set(path_nodes(p2.hops)[1:-1])
                assert not heuristic.link_ids(p1.hops) & heuristic.link_ids(p2.hops)


@pytest.mark.parametrize("seed", [2, 7])
def test_minsum_baseline_b7_is_valid_and_bounded_by_milp(seed):
    from fiser import methods
    G = random_network(seed)
    for s, d in itertools.combinations(sorted(G.nodes), 2):
        for F_th, r in methods.run_pair(G, s, d, [0.75, 0.85], methods=("B7", "FiSER")).items():
            b7, fi = r["B7"], r["FiSER"]
            if b7.served:
                p1, p2 = b7.paths
                assert fi.served and b7.worst <= fi.worst * (1 + 1e-6)
                assert min(p1.F, p2.F) >= F_th - 1e-9
                assert not set(path_nodes(p1.hops)[1:-1]) & set(path_nodes(p2.hops)[1:-1])
