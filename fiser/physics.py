"""Physical layer: Werner-state fidelity, BBPSSW purification, and per-link option tables.

A link option is (success probability per time slot, Werner parameter) for one
purification level k: level k consumes 2^k raw pairs and 2^k - 1 successful
BBPSSW rounds (symmetric recurrence).

Memory decoherence (optional, coherence time T): every stored qubit depolarizes, so a pair whose two
qubits are stored for t seconds has its Werner parameter multiplied by exp(-2t/T). Links start
just in time, so all links of a path finish together and swaps follow immediately. A link's pairs
are then stored during heralding and during each purification round (one classical exchange over
the link, L/c, each), and the two end qubits wait for the swap outcomes (sum of L/c over the path),
which contributes exp(-2 L/(cT)) per link. All factors are per link, so they live in the option
tables and the MILP stays exact.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.stats import binom


@dataclass(frozen=True)
class PhysParams:
    """Per-time-slot physical parameters (values and sources in README.md, "Networks and parameters")."""

    att_db_per_km: float = 0.2   # fiber attenuation
    eta_det: float = 0.5         # source/coupling/detection efficiency per attempt
    attempts: int = 50           # multiplexed generation attempts per slot
    w0: float = 0.98             # Werner parameter of a freshly generated pair
    l_dec_km: float = 500.0      # L_0: length scale of phenomenological channel noise (w = w0 e^{-L/L_0})
    q_swap: float = 0.9          # entanglement-swapping success probability
    lam_swap: float = 0.99       # Werner factor per swap (imperfect Bell measurement)
    max_level: int = 2           # maximum BBPSSW recurrence rounds per link
    w0_spread: float = 0.0       # per-link hardware heterogeneity: w0_e ~ U[w0 - s, w0 + s]
    seed: int = 0                # seed for per-link heterogeneity
    t_coh_s: float = math.inf    # memory coherence time T [s] (depolarizing per qubit); inf: no storage decay
    c_km_s: float = 2.0e5        # signal speed in fiber [km/s] (classical exchanges)

    def to_dict(self):
        return asdict(self)


def w_of_F(F):
    return (4 * F - 1) / 3


def F_of_w(w):
    return (1 + 3 * w) / 4


def bbpssw(F):
    """One BBPSSW round on two Werner pairs of fidelity F -> (success prob, output fidelity)."""
    p = F ** 2 + 2 * F * (1 - F) / 3 + 5 * ((1 - F) / 3) ** 2
    return p, (F ** 2 + ((1 - F) / 3) ** 2) / p


def stored(F, t, T):
    """Fidelity of a Werner pair after both qubits are stored for t seconds (coherence time T)."""
    if t <= 0 or not math.isfinite(T):
        return F
    return F_of_w(w_of_F(F) * math.exp(-2.0 * t / T))


def recurrence_options(F0, p_at_least, max_level, round_time=0.0, T=math.inf, w_extra=1.0):
    """Option table {k: (p, w)} for a link whose raw pairs have fidelity F0.

    p_at_least(n) is the probability that at least n raw pairs are generated in a slot.
    round_time: duration of one classical exchange over the link (heralding and each BBPSSW
    round); the pairs decay for round_time after heralding and after every round.
    w_extra: factor on the final Werner parameter (the link's share of the swap-signaling wait).
    """
    Fs, ppur = [stored(F0, round_time, T)], []
    for _ in range(max_level):
        p, F_next = bbpssw(Fs[-1])
        ppur.append(p)
        Fs.append(stored(F_next, round_time, T))
    opts = {}
    for k in range(max_level + 1):
        p = p_at_least(2 ** k)
        for r in range(k):
            p *= ppur[r] ** (2 ** (k - 1 - r))  # all 2^(k-1-r) round-r purifications succeed
        if p > 1e-12:
            opts[k] = (float(p), float(w_of_F(Fs[k]) * w_extra))
    return opts


def signaling_factor(L_km, P: PhysParams):
    """The link's share of the end-to-end swap-signaling decay: exp(-2 L / (c T))."""
    return math.exp(-2.0 * L_km / (P.c_km_s * P.t_coh_s)) if math.isfinite(P.t_coh_s) else 1.0


def fiber_link_options(L_km, P: PhysParams, w0=None):
    eta = P.eta_det * 10 ** (-P.att_db_per_km * L_km / 10)
    F0 = F_of_w((P.w0 if w0 is None else w0) * math.exp(-L_km / P.l_dec_km))
    return recurrence_options(F0, lambda n: binom.sf(n - 1, P.attempts, eta), P.max_level,
                              round_time=L_km / P.c_km_s, T=P.t_coh_s, w_extra=signaling_factor(L_km, P))
