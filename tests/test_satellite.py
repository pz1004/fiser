import numpy as np
import pytest

from fiser.satellite import R_E, Walker, ground_ecef, look_angles, slant_range


def test_orbit_radius_and_period():
    w = Walker(total=72, planes=6, phasing=1, inclination_deg=53, altitude_km=550)
    pos = w.ecef(np.linspace(0, 3600, 7))
    assert np.allclose(np.linalg.norm(pos, axis=-1), R_E + 550)
    assert w.period_s / 60 == pytest.approx(95.6, abs=0.3)  # ~95.6 min at 550 km


def test_zenith_pass_geometry():
    # A satellite directly above a ground point: 90 deg elevation, range = altitude.
    g = ground_ecef([0.0], [0.0])
    sat = np.array([[[R_E + 550, 0.0, 0.0]]])
    el, rng = look_angles(g, sat)
    assert el[0, 0, 0] == pytest.approx(90.0)
    assert rng[0, 0, 0] == pytest.approx(550.0)


def test_slant_range_matches_look_angles():
    g = ground_ecef([36.35], [127.38])
    w = Walker(total=72, planes=6, phasing=1, inclination_deg=53, altitude_km=550)
    el, rng = look_angles(g, w.ecef(np.arange(0, 86400, 60)))
    vis = el > 10
    assert vis.any()
    assert np.allclose(rng[vis], slant_range(el[vis], 550), rtol=1e-6)


def test_highs_presolve_crash_instance_is_solved():
    # NSFNET, 2 OGS, snapshot 176 (300 s grid), pair (2, 8), F_th = 0.8 crashed HiGHS presolve
    # through PuLP ('vector::reserve'); the fallback must return the exact optimum.
    from fiser import milp, topologies
    from fiser.network import build_network
    from fiser.physics import PhysParams
    from fiser.spaceground import SatParams, SpaceGround, farthest_point_ogs, place_on_map
    topo = topologies.load("nsfnet")
    G = build_network(topo["edges"], PhysParams(), max_link_km=100)
    sg = SpaceGround(G, farthest_point_ogs(G, 2), place_on_map(G, topo), SatParams(sats_per_pair=1),
                    np.arange(0, 86400, 300))
    sol = milp.solve(sg.snapshot(176), 2, 8, 0.8)
    assert sol.ok and min(p.S for p in sol.paths) == pytest.approx(0.17695, abs=1e-5)


def test_inclination_bounds_latitude():
    w = Walker(total=40, planes=5, phasing=1, inclination_deg=53, altitude_km=550)
    pos = w.ecef(np.arange(0, 6000, 30))
    lat = np.degrees(np.arcsin(pos[..., 2] / np.linalg.norm(pos, axis=-1)))
    assert lat.max() == pytest.approx(53, abs=0.5) and lat.min() == pytest.approx(-53, abs=0.5)
