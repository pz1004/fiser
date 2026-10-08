"""Reference topologies (link lengths in km, 0-indexed nodes) and random generators."""
from __future__ import annotations

import itertools
import math
import random

import networkx as nx
from scipy.spatial import Delaunay

from .topology_data import COST239, NSFNET, USNET  # noqa: E402  (sources in topology_data.py)

TOPOLOGIES = {"nsfnet": NSFNET, "cost239": COST239, "usnet": USNET}


def load(name):
    return TOPOLOGIES[name.lower()]


def random_mesh(n, seed, spacing_km=40.0, max_link_km=100.0, target_degree=3.5, max_tries=100):
    """Random biconnected geometric topology for scalability studies.

    n nodes are uniform in a square of side sqrt(n) * spacing_km. Starting from their
    Delaunay triangulation (biconnected), links longer than max_link_km are dropped and
    random links are removed while the graph stays biconnected, until the average
    degree reaches target_degree.
    """
    for t in range(max_tries):
        rng = random.Random(seed * 100003 + t)
        pos = [(rng.uniform(0, math.sqrt(n) * spacing_km), rng.uniform(0, math.sqrt(n) * spacing_km))
               for _ in range(n)]
        G = nx.Graph()
        for simplex in Delaunay(pos).simplices:
            for i, j in itertools.combinations(map(int, simplex), 2):
                G.add_edge(i, j, L=math.dist(pos[i], pos[j]))
        G.remove_edges_from([(u, v) for u, v, L in G.edges(data="L") if L > max_link_km])
        if G.number_of_nodes() < n or not nx.is_biconnected(G):
            continue
        edges = list(G.edges(data="L"))
        rng.shuffle(edges)
        for u, v, L in edges:
            if 2 * G.number_of_edges() / n <= target_degree:
                break
            G.remove_edge(u, v)
            if not nx.is_biconnected(G):
                G.add_edge(u, v, L=L)
        return {"name": f"Mesh-{n}-s{seed}", "pos": dict(enumerate(pos)),
                "edges": [(u, v, round(max(L, 1.0), 3)) for u, v, L in G.edges(data="L")]}
    raise RuntimeError(f"no biconnected mesh for n={n} after {max_tries} tries")
