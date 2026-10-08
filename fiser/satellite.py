"""LEO Walker-delta constellation geometry (circular orbits, spherical Earth).

Positions are computed in ECI and rotated into ECEF with the Earth's rotation, so
elevation and slant range of every satellite as seen from every ground node are
available per snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_E = 6371.0            # mean Earth radius [km]
MU_E = 398600.4418      # gravitational parameter [km^3/s^2]
OMEGA_E = 7.2921159e-5  # Earth rotation rate [rad/s]


@dataclass(frozen=True)
class Walker:
    """Walker-delta constellation i: T/P/F at a common altitude."""

    total: int
    planes: int
    phasing: int
    inclination_deg: float
    altitude_km: float

    @property
    def radius(self):
        return R_E + self.altitude_km

    @property
    def period_s(self):
        return 2 * np.pi * np.sqrt(self.radius ** 3 / MU_E)

    def elements(self):
        """(RAAN, initial argument of latitude) of every satellite [rad]."""
        per_plane = self.total // self.planes
        p, k = np.divmod(np.arange(self.total), per_plane)
        raan = 2 * np.pi * p / self.planes
        u0 = 2 * np.pi * k / per_plane + 2 * np.pi * self.phasing * p / self.total
        return raan, u0

    def ecef(self, t_s):
        """Satellite ECEF positions [km], shape (len(t), total, 3)."""
        t = np.atleast_1d(np.asarray(t_s, dtype=float))[:, None]
        raan, u0 = self.elements()
        u = u0[None, :] + np.sqrt(MU_E / self.radius ** 3) * t
        inc = np.radians(self.inclination_deg)
        x = np.cos(u) * np.cos(raan) - np.sin(u) * np.cos(inc) * np.sin(raan)
        y = np.cos(u) * np.sin(raan) + np.sin(u) * np.cos(inc) * np.cos(raan)
        z = np.sin(u) * np.sin(inc) * np.ones_like(raan)
        th = OMEGA_E * t
        xe = x * np.cos(th) + y * np.sin(th)
        ye = -x * np.sin(th) + y * np.cos(th)
        return self.radius * np.stack([xe, ye, z], axis=-1)


def ground_ecef(lat_deg, lon_deg):
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    return R_E * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def look_angles(ground, sats):
    """Elevation [deg] and slant range [km] for ground (G, 3) and sats (T, S, 3) -> (T, G, S)."""
    d = sats[:, None, :, :] - ground[None, :, None, :]
    rng = np.linalg.norm(d, axis=-1)
    up = ground / np.linalg.norm(ground, axis=-1, keepdims=True)
    sin_el = np.einsum("tgsk,gk->tgs", d, up) / rng
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1))), rng


def sun_elevation(t_unix_s, lat_deg, lon_deg):
    """Solar elevation [deg] (low-precision almanac formulas, ~0.01 deg for 1950-2050).

    t_unix_s: (T,) UNIX times; lat/lon: (G,) degrees. Returns (T, G).
    """
    n = np.asarray(t_unix_s, float)[:, None] / 86400.0 + 2440587.5 - 2451545.0  # days since J2000
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = np.radians((280.460 + 0.9856474 * n) % 360 + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 4e-7 * n)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    gmst = np.radians((280.46061837 + 360.98564736629 * n) % 360)
    ha = gmst + np.radians(np.asarray(lon_deg, float))[None, :] - ra
    lat = np.radians(np.asarray(lat_deg, float))[None, :]
    return np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)))


def slant_range(elev_deg, altitude_km):
    """Slant range to a satellite at the given elevation (spherical Earth)."""
    el = np.radians(elev_deg)
    r = R_E + altitude_km
    return np.sqrt(r ** 2 - (R_E * np.cos(el)) ** 2) - R_E * np.sin(el)
