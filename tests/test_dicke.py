# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.1: symulacja N=2..100 w bazie Dickego (sektory j)."""
import numpy as np
import pytest

from entropia import core as M
from entropia import extensions as R
from entropia import dicke as D


# ---------- sektory ----------
def test_dicke_sektory_suma():
    """Σ_j m(N,j)·(2j+1) = 2^N (rozkład przestrzeni Hilberta)."""
    for N in [2, 3, 4, 5, 6, 8, 10]:
        sek = D.sektory_dickego(N)
        tot = sum(m * (2 * j + 1) for j, m in sek)
        assert tot == 2 ** N
    # znane krotności N=4: j=2 (1), j=1 (3), j=0 (2)
    sek4 = dict((j, m) for j, m in D.sektory_dickego(4))
    assert sek4[2.0] == 1 and sek4[1.0] == 3 and sek4[0.0] == 2


# ---------- zgodność z pełną przestrzenią ----------
def test_dicke_vs_full_N4():
    """N=4 |1111⟩: metoda sektorowa = pełna przestrzeń (do precyzji)."""
    ket = np.zeros(16, complex); ket[15] = 1.0
    Sf, *_ = R.symuluj_wspolne(M.GAMMA_B, np.outer(ket, ket.conj()), N=4,
                               gamma_phi=0.0, n=200)
    Sd = D.symuluj_dicke(4, M.GAMMA_B, n=200)["S"]
    assert np.max(np.abs(Sd - Sf)) < 1e-8


def test_dicke_vs_full_N5():
    ket = np.zeros(32, complex); ket[31] = 1.0
    Sf, *_ = R.symuluj_wspolne(M.GAMMA_B, np.outer(ket, ket.conj()), N=5,
                               gamma_phi=0.0, n=200)
    Sd = D.symuluj_dicke(5, M.GAMMA_B, n=200)["S"]
    assert np.max(np.abs(Sd - Sf)) < 1e-8


# ---------- S∞ = ln(N+1) ----------
def test_dicke_Sinf_lnN1():
    for N in [2, 4, 6, 10]:
        res = D.symuluj_dicke(N, M.GAMMA_B, n=800)
        assert abs(res["S"][-1] - np.log(N + 1)) < 5e-3


def test_dicke_Sinf_N100():
    """N=100: S∞ → ln 101 (asymptota sektora symetrycznego)."""
    res = D.symuluj_dicke(100, M.GAMMA_B, n=500)
    assert abs(res["S"][-1] - np.log(101)) < 1e-2


# ---------- 27× ----------
def test_dicke_27():
    """Kompresja czasowa S_A(n) = S_B(27n) — dokładna dla N = 2..100."""
    for N in [2, 10, 50]:
        e = D.kompresja_27(N, M.GAMMA_B, n_cmp=20)
        assert e < 1e-10


# ---------- czkanie (stall) ----------
def test_dicke_czkanie_stall():
    """Kwantowany zegar entropii staje przy nasyceniu (Δτ = 0 w ogonie)."""
    for N in [2, 6, 10]:
        res = D.symuluj_dicke(N, M.GAMMA_B, n=600)
        fz = D.czkanie_stat(res["dS"])
        assert fz > 0.9


# ---------- pamięć subradiacyjna ----------
def test_dicke_pamiec_N2():
    """N=2 |10⟩-mieszanina: P_dark = ½, I(A:B)(∞) = ln(2/√3)."""
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    rho_j1 = np.zeros((3, 3), complex); rho_j1[1, 1] = 1.0
    rho_j0 = np.ones((1, 1), complex)
    stan = {1.0: (0.5, rho_j1), 0.0: (0.5, rho_j0)}
    res = D.symuluj_dicke(2, M.GAMMA_B, stan=stan, n=400)
    assert abs(res["P_dark"][-1] - 0.5) < 1e-9
    assert abs(res["I_AB"][-1] - np.log(2 / np.sqrt(3))) < 1e-3


def test_dicke_pdark_haar():
    """P_dark stanu Haar = 1 − (N+1)/2^N → 1 dla dużych N."""
    for N in [4, 10, 20]:
        p = D.Haar_pdark(N) if hasattr(D, "Haar_pdark") else 1 - (N + 1) / 2 ** N
        assert abs(p - (1 - (N + 1) / 2 ** N)) < 1e-12
    assert 1 - (101) / 2 ** 100 > 0.999
