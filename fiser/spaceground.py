"""Space-ground snapshots: satellite dual-downlink links between optical ground stations (OGS).

A satellite source sends one photon of each Bell pair to two OGS (Khatri et al. 2021).
Per downlink, the transmissivity is
    eta = eta_sys * [1 - exp(-2 r^2 / w(L)^2)] * eta_zen^(1 / sin(elevation)),
    w(L) = w0 * sqrt(1 + (L / L_R)^2),  L_R = pi * w0^2 / lambda,
and an OGS pair (u, v) gets Bin(M_s, eta_u * eta_v) raw pairs per slot with fidelity
F_sat. Each OGS pair gets parallel links through its best common satellites above the
minimum elevation, and (night-only baseline) only while the Sun is below sun_limit_deg
at both stations.
Links carry sat_id: a satellite has one source, so a design uses at most one link
per satellite (which also makes it a shared failure point).
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.stats import binom

from .network import add_link
from .physics import recurrence_options, signaling_factor
from .satellite import Walker, ground_ecef, look_angles, sun_elevation


@dataclass(frozen=True)
class SatParams:
    """Satellite-link parameters (values and sources in README.md, "Networks and parameters")."""

    total: int = 144                 # Walker-delta 53: 144/12/1
    planes: int = 12
    phasing: int = 1
    inclination_deg: float = 53.0
    altitude_km: float = 550.0
    el_min_deg: float = 20.0
    wavelength_m: float = 810e-9
    tx_waist_m: float = 0.025        # effective beam waist (Khatri et al. Table 3)
    rx_radius_m: float = 0.6         # 1.2 m OGS telescope (Micius)
    eta_atm_zenith: float = 0.5      # zenith transmittance at 810 nm
    eta_sys: float = 0.25            # per-downlink optics x detection (Micius-calibrated)
    attempts: int = 10 ** 6          # 1e9 pairs/s source x 1 ms slot
    F_sat: float = 0.91              # Micius 2020
    max_level: int = 2
    night_only: bool = True
    sun_limit_deg: float = -12.0
    sats_per_pair: int = 3           # parallel links via the best common satellites of each OGS pair
    epoch: str = "2027-03-20T12:00:00"  # UTC start of the 24 h window (near equinox)
    min_lifetime_s: float = 0.0      # offer only links that stay usable for this long (checked every 15 s)

    def walker(self):
        return Walker(self.total, self.planes, self.phasing, self.inclination_deg, self.altitude_km)

    def epoch_unix(self):
        return dt.datetime.fromisoformat(self.epoch).replace(tzinfo=dt.timezone.utc).timestamp()


def downlink_eta(el_deg, range_km, S: SatParams):
    L = np.asarray(range_km) * 1e3
    L_R = math.pi * S.tx_waist_m ** 2 / S.wavelength_m
    w = S.tx_waist_m * np.sqrt(1 + (L / L_R) ** 2)
    eta_fs = 1 - np.exp(-2 * S.rx_radius_m ** 2 / w ** 2)
    sin_el = np.sin(np.radians(np.clip(el_deg, 0.1, 90)))
    return S.eta_sys * eta_fs * S.eta_atm_zenith ** (1 / sin_el)


def place_on_map(G, topo, center=(39.0, -98.0)):
    """Lat/lon per node at the network's scaled size.

    With source coordinates, each node's offset from the coordinate centroid is shrunk
    by the network's km scale factor (shape and latitude are kept). Without them, a
    2-D layout whose distances follow the fiber lengths is centered at `center`.
    """
    s = G.graph["scale"]
    coords = topo.get("coords")
    if coords:
        lat0 = float(np.mean([c[0] for c in coords.values()]))
        lon0 = float(np.mean([c[1] for c in coords.values()]))
        return {n: (lat0 + s * (coords[n][0] - lat0), lon0 + s * (coords[n][1] - lon0)) for n in G.nodes}
    H = nx.Graph(G)
    pos = nx.kamada_kawai_layout(H, dist=dict(nx.shortest_path_length(H, weight="L")))
    km = np.median([H.edges[u, v]["L"] / math.dist(pos[u], pos[v]) for u, v in H.edges])
    lat0, lon0 = center
    return {n: (lat0 + km * y / 111.2, lon0 + km * x / (111.2 * math.cos(math.radians(lat0))))
            for n, (x, y) in pos.items()}


def farthest_point_ogs(G, k):
    """k spread-out OGS nodes: start at a diameter endpoint, then repeatedly add the node
    farthest (in fiber km) from the chosen set."""
    if k <= 0:
        return []
    dist = dict(nx.shortest_path_length(nx.Graph(G), weight="L"))
    chosen = [max(G.nodes, key=lambda n: (max(dist[n].values()), -n))]
    while len(chosen) < min(k, G.number_of_nodes()):
        chosen.append(max((n for n in G.nodes if n not in chosen),
                          key=lambda n: (min(dist[n][c] for c in chosen), -n)))
    return chosen


def _great_circle_km(a, b, R=6371.0):
    (la1, lo1), (la2, lo2) = (np.radians(a[0]), np.radians(a[1])), (np.radians(b[0]), np.radians(b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(min(1.0, h))))


class SpaceGround:
    """Precomputed satellite geometry for a ground network over a time grid [s from epoch]."""

    def __init__(self, G, ogs, latlon, S: SatParams, t_grid_s):
        self.G, self.ogs, self.S, self.t = G, list(ogs), S, np.asarray(t_grid_s, float)
        lat = [latlon[n][0] for n in self.ogs]
        lon = [latlon[n][1] for n in self.ogs]
        el, rng = look_angles(ground_ecef(lat, lon), S.walker().ecef(self.t))   # (T, n_ogs, n_sats)
        # classical exchanges between two OGS (heralding coincidences, purification) run over the ground
        self.ogs_km = {(a, b): _great_circle_km(latlon[a], latlon[b]) for a in self.ogs for b in self.ogs}
        self.eta = np.where(el >= S.el_min_deg, downlink_eta(el, rng, S), 0.0)
        if S.night_only:
            night = sun_elevation(self.t + S.epoch_unix(), lat, lon) < S.sun_limit_deg  # (T, n_ogs)
            self.eta *= night[:, :, None]
        if S.min_lifetime_s > 0:  # a link must stay usable until the next re-provisioning
            ok = self.eta > 0
            for off in np.arange(15.0, S.min_lifetime_s + 1e-9, 15.0):
                el2, _ = look_angles(ground_ecef(lat, lon), S.walker().ecef(self.t + off))
                vis = el2 >= S.el_min_deg
                if S.night_only:
                    vis &= (sun_elevation(self.t + off + S.epoch_unix(), lat, lon) < S.sun_limit_deg)[:, :, None]
                ok &= vis
            self.eta = np.where(ok, self.eta, 0.0)

    def links(self, ti):
        """[(u, v, sat index, eta_u * eta_v)]: each OGS pair's best usable common satellites."""
        out = []
        for i, j in itertools.combinations(range(len(self.ogs)), 2):
            prod = self.eta[ti, i] * self.eta[ti, j]
            for sat in np.argsort(-prod)[: self.S.sats_per_pair]:
                if prod[sat] > 0:
                    out.append((self.ogs[i], self.ogs[j], int(sat), float(prod[sat])))
        return out

    def snapshot(self, ti):
        """Ground network plus the satellite links available at snapshot ti."""
        H = self.G.copy()
        P = self.G.graph["params"]
        for u, v, sat, eta2 in self.links(ti):
            D = self.ogs_km[u, v]
            opts = recurrence_options(self.S.F_sat, lambda n: binom.sf(n - 1, self.S.attempts, eta2),
                                      self.S.max_level, round_time=D / P.c_km_s, T=P.t_coh_s,
                                      w_extra=signaling_factor(D, P))
            if opts:
                add_link(H, u, v, f"s{sat}", opts, sat_id=sat, eta2=eta2, tau=D / P.c_km_s)
        return H
