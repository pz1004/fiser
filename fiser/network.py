"""Quantum network as a networkx MultiGraph.

Edge (u, v, key) carries kind ('fiber' | 'sat'), length L (km, fiber only), and
opts = {level: (p, w)}. Parallel links between the same node pair (e.g. a fiber
link and a satellite dual-downlink) are distinguished by key. Nodes carry their
swap success q and Werner factor lam.
"""
from __future__ import annotations

import random

import networkx as nx

from .physics import PhysParams, fiber_link_options


def build_network(edges, P: PhysParams = PhysParams(), max_link_km=100.0, scale=None):
    """Ground network; link lengths are scaled so the longest equals max_link_km (unless scale)."""
    s = scale if scale is not None else max_link_km / max(L for _, _, L in edges)
    G = nx.MultiGraph(params=P, scale=s)
    rng = random.Random(P.seed)
    for u, v, L in edges:
        w0 = min(1.0, P.w0 + rng.uniform(-P.w0_spread, P.w0_spread)) if P.w0_spread else P.w0
        G.add_edge(u, v, key="f", kind="fiber", L=L * s, w0=w0, tau=L * s / P.c_km_s,
                   opts=fiber_link_options(L * s, P, w0))
    for n in G.nodes:
        G.nodes[n].update(q=P.q_swap, lam=P.lam_swap)
    return G


def add_link(G, u, v, key, opts, kind="sat", **attrs):
    """Add an extra (e.g. satellite) link with a precomputed option table."""
    G.add_edge(u, v, key=key, kind=kind, opts=opts, **attrs)
