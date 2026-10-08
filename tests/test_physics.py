import math

import pytest

from fiser.physics import F_of_w, PhysParams, bbpssw, fiber_link_options, w_of_F


def test_werner_roundtrip():
    for F in (0.25, 0.5, 0.8, 1.0):
        assert F_of_w(w_of_F(F)) == pytest.approx(F)


def test_bbpssw_known_values():
    p, F1 = bbpssw(0.9)
    assert p == pytest.approx(0.81 + 0.06 + 5 / 900)
    assert F1 == pytest.approx((0.81 + 1 / 900) / p)


def test_bbpssw_fixed_points_and_gain():
    assert bbpssw(1.0) == pytest.approx((1.0, 1.0))
    assert bbpssw(0.5)[1] == pytest.approx(0.5)
    for F in (0.55, 0.7, 0.85, 0.95):
        assert 0.5 < bbpssw(F)[1] > F


def test_link_options_tradeoff():
    P = PhysParams()
    for L in (5, 30, 60, 100):
        opts = fiber_link_options(L, P)
        assert set(opts) == set(range(P.max_level + 1))
        ps = [opts[k][0] for k in sorted(opts)]
        ws = [opts[k][1] for k in sorted(opts)]
        assert all(a > b for a, b in zip(ps, ps[1:]))  # purification costs success
        assert all(a < b for a, b in zip(ws, ws[1:]))  # and buys fidelity
        assert 0 < ps[0] <= 1 and ws[-1] <= 1


def test_link_quality_decreases_with_length():
    P = PhysParams()
    o1, o2 = fiber_link_options(20, P), fiber_link_options(80, P)
    assert o1[0][0] > o2[0][0] and o1[0][1] > o2[0][1]
    assert math.isclose(o1[0][1], P.w0 * math.exp(-20 / P.l_dec_km))


def test_memory_decoherence_in_option_tables():
    import math
    from dataclasses import replace
    from fiser.physics import F_of_w, PhysParams, fiber_link_options, stored, w_of_F
    P = PhysParams()
    # storing both qubits for t multiplies the Werner parameter by exp(-2t/T)
    assert w_of_F(stored(0.9, 0.005, 0.1)) == pytest.approx(w_of_F(0.9) * math.exp(-0.1))
    assert stored(0.9, 0.005, math.inf) == 0.9
    for L in (10.0, 50.0, 100.0):
        base = fiber_link_options(L, P)
        assert fiber_link_options(L, replace(P, t_coh_s=math.inf)) == base   # T = inf: unchanged
        prev = base
        for T in (10.0, 1.0, 0.1, 0.01):
            o = fiber_link_options(L, replace(P, t_coh_s=T))
            assert o.keys() == prev.keys()
            for k in o:  # shorter memories: never higher fidelity, success unchanged at level 0
                assert o[k][1] <= prev[k][1] + 1e-15
            assert o[0][0] == base[0][0]
            prev = o
    # level 0 at L km: heralding (L/c) and swap signaling (L/c) give exp(-4 L / (c T))
    L, T = 100.0, 0.1
    w0 = P.w0 * math.exp(-L / P.l_dec_km)
    assert fiber_link_options(L, replace(P, t_coh_s=T))[0][1] == pytest.approx(w0 * math.exp(-4 * L / (P.c_km_s * T)))


def test_exact_end_qubit_wait_under_decoherence():
    """W(P) equals an independent computation of every qubit's storage time under just-in-time starts."""
    import itertools
    import math
    from dataclasses import replace
    from fiser import milp
    from fiser.network import build_network
    from fiser.paths import best_purification, path_metrics
    from fiser.physics import PhysParams, recurrence_options, w_of_F
    from scipy.stats import binom
    P = replace(PhysParams(), t_coh_s=0.05)
    G = build_network([(0, 1, 40.0), (1, 2, 90.0), (2, 3, 25.0), (0, 3, 70.0), (1, 3, 60.0)], P, scale=1.0)
    for hops in ([(0, 1, "f"), (1, 2, "f"), (2, 3, "f")], [(0, 3, "f")], [(0, 1, "f"), (1, 3, "f")]):
        taus = [G.edges[h]["L"] / P.c_km_s for h in hops]
        for levels in itertools.product(range(3), repeat=len(hops)):
            W = 1.0
            for h, lev, tau in zip(hops, levels, taus):  # link pair after heralding and purification rounds only
                L = G.edges[h]["L"]
                eta = P.eta_det * 10 ** (-P.att_db_per_km * L / 10)
                F0 = (1 + 3 * P.w0 * math.exp(-L / P.l_dec_km)) / 4
                o = recurrence_options(F0, lambda n: binom.sf(n - 1, P.attempts, eta), P.max_level,
                                       round_time=tau, T=P.t_coh_s)
                W *= o[lev][1]
            W *= P.lam_swap ** (len(hops) - 1)
            if len(hops) > 1:  # s waits for the farthest outcome (sum - last), d for (sum - first)
                W *= math.exp(-((sum(taus) - taus[-1]) + (sum(taus) - taus[0])) / P.t_coh_s)
            assert path_metrics(G, hops, levels)[1] == pytest.approx((1 + 3 * W) / 4, abs=1e-12)
    # the MILP and the purification DP use the same corrected fidelity
    for s_, d_ in itertools.combinations(range(4), 2):
        for F_th in (0.75, 0.8):
            sol = milp.solve(G, s_, d_, F_th, n_paths=1)
            if sol.ok:
                p = sol.paths[0]
                assert p.F >= F_th - 1e-9
                bp = best_purification(G, p.hops, F_th)
                assert bp is not None and bp[0] == pytest.approx(p.S, rel=1e-9)
