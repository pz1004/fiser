"""Independent density-matrix checks of the closed-form physics used by the model.

These tests build explicit 2- and 4-qubit density matrices with numpy and share no code with
fiser.physics: BBPSSW purification (bilateral CNOTs, post-selection, twirled Werner output),
entanglement swapping (Bell measurement and Pauli correction), and depolarizing memories.
"""
import itertools

import numpy as np
import pytest

from fiser.physics import bbpssw

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
Y = 1j * X @ Z
PHI_P = np.array([1, 0, 0, 1]) / np.sqrt(2)
BELL = {  # (Bell state, Pauli on the second qubit that maps it to PHI_P)
    "phi+": (PHI_P, I2),
    "phi-": (np.array([1, 0, 0, -1]) / np.sqrt(2), Z),
    "psi+": (np.array([0, 1, 1, 0]) / np.sqrt(2), X),
    "psi-": (np.array([0, 1, -1, 0]) / np.sqrt(2), Y),
}


def werner(w):
    return w * np.outer(PHI_P, PHI_P) + (1 - w) * np.eye(4) / 4


def fidelity(rho):
    return float(np.real(PHI_P @ rho @ PHI_P))


def kron(*ms):
    out = np.array([[1.0]])
    for m in ms:
        out = np.kron(out, m)
    return out


def cnot(n, control, target):
    """CNOT on an n-qubit register (qubit 0 is the most significant)."""
    dim = 2 ** n
    U = np.zeros((dim, dim))
    for i in range(dim):
        bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[control]:
            bits[target] ^= 1
        U[sum(b << (n - 1 - q) for q, b in enumerate(bits)), i] = 1
    return U


def partial_trace(rho, keep, n):
    """Trace out every qubit not in keep (qubit 0 is the most significant)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    rows = list(letters[:n])
    cols = list(letters[n:2 * n])
    for q in range(n):
        if q not in keep:
            cols[q] = rows[q]          # summed index
    out = "".join(rows[q] for q in keep) + "".join(cols[q] for q in keep)
    t = np.einsum("".join(rows) + "".join(cols) + "->" + out, rho.reshape([2] * (2 * n)))
    d = 2 ** len(keep)
    return t.reshape(d, d)


def depolarize(rho, lam, qubit, n):
    """Single-qubit depolarizing channel rho -> lam rho + (1 - lam) I/2 (x) Tr_qubit(rho)."""
    paulis = [I2, X, Y, Z]
    out = np.zeros_like(rho, dtype=complex)
    p = (1 - lam) / 4
    for k, P in enumerate(paulis):
        ops = [I2] * n
        ops[qubit] = P
        U = kron(*ops)
        out += ((1 - 3 * p) if k == 0 else p) * U @ rho @ U.conj().T
    return out


@pytest.mark.parametrize("F", [0.6, 0.75, 0.85, 0.95])
def test_bbpssw_against_density_matrices(F):
    w = (4 * F - 1) / 3
    rho = np.kron(werner(w), werner(w))           # qubits A1 B1 A2 B2
    U = cnot(4, 0, 2) @ cnot(4, 1, 3)             # bilateral CNOT: A1 -> A2, B1 -> B2
    rho = U @ rho @ U.T
    P = sum(kron(I2, I2, np.diag(np.eye(2)[x]), np.diag(np.eye(2)[x])) for x in (0, 1))  # A2 == B2
    p_succ = float(np.real(np.trace(P @ rho)))
    out = partial_trace(P @ rho @ P / p_succ, keep=[0, 1], n=4)
    g, F_out = bbpssw(F)
    assert p_succ == pytest.approx(g, abs=1e-12)
    assert fidelity(out) == pytest.approx(F_out, abs=1e-12)


@pytest.mark.parametrize("w1,w2", [(0.9, 0.8), (0.7, 0.95), (1.0, 0.6)])
def test_swapping_multiplies_werner_parameters(w1, w2):
    rho = np.kron(werner(w1), werner(w2))         # qubits A B C D; Bell measurement on B, C
    out = np.zeros((4, 4), dtype=complex)
    for b, (state, corr) in BELL.items():
        proj = kron(I2, np.outer(state, state.conj()), I2)
        post = proj @ rho @ proj
        red = partial_trace(post, keep=[0, 3], n=4)
        C = np.kron(I2, corr)                     # Pauli correction on D
        out += C @ red @ C.conj().T
    w_out = (4 * fidelity(out) - 1) / 3
    assert np.real(np.trace(out)) == pytest.approx(1.0, abs=1e-12)
    assert w_out == pytest.approx(w1 * w2, abs=1e-12)


@pytest.mark.parametrize("w,t,T", [(0.95, 0.005, 0.1), (0.8, 0.0005, 0.01), (0.99, 0.2, 1.0)])
def test_depolarizing_memories_scale_werner_parameter(w, t, T):
    lam = np.exp(-t / T)
    rho = depolarize(depolarize(werner(w), lam, 0, 2), lam, 1, 2)
    assert (4 * fidelity(rho) - 1) / 3 == pytest.approx(w * np.exp(-2 * t / T), abs=1e-12)
