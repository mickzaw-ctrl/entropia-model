# -*- coding: utf-8 -*-
"""Testy zegara kosmicznego: czkanie, czas wstecz, cykl dwustronny."""
import numpy as np
import pytest

from entropia import core as M
from entropia import extensions as R


def test_czkanie_istnieje():
    """Przy niskiej produkcji entropii Δt_n = 0 (czas „czka")."""
    S_B, _, _ = M.symuluj(M.GAMMA_B, n=400)
    dS = M.delta_entropii(S_B)
    _, dt, _ = M.zegar_stochastyczny(dS, seed=11)
    tail = dt[40:200]
    assert np.any(tail == 0)
    # i są też niezerowe kroki
    assert np.any(tail > 0)


def test_zegar_wstecz():
    """Gorący start + zimna kąpiel: T(n) = S(n) − S(0) < 0 (czas wstecz)."""
    S, G, T, S0, Seq = R.symuluj_wielki_wybuch(M.GAMMA_B, 0.95, 0.15, R.FB_STALY, n=300)
    dS = np.zeros_like(S); dS[1:] = S[1:] - S[:-1]
    T_w, _, _ = R.zegar_wstecz(dS, seed=3)
    # średnia po realizacjach ≈ S − S0
    rng = np.random.default_rng(3)
    mean = np.zeros(300)
    for _ in range(200):
        k = rng.poisson(np.maximum(-dS, 0) / M.DELTA_S_Q)
        mean += np.cumsum(-k * M.DELTA_S_Q)
    mean /= 200
    assert np.max(np.abs(mean - (S - S0))) < 0.02
    assert T[-1] < 0


def test_cykl_powtarzalny():
    """Dwa cykle: entropia wraca do ln 2 (czas jako zamknięta pętla)."""
    S2, _, _, _, _ = R.symuluj_cykl(0.05, 0.15, 300, 601)
    assert abs(S2[0] - M.LN2) < 1e-9
    assert abs(S2[-1] - M.LN2) < 0.02                # powrót do ln 2 (pętla)
    # minimum w pierwszym cyklu (ochłodzenie)
    assert S2.min() < M.LN2 - 0.1


def test_kwantowy_zegar_backaction():
    """Kwantowy zegar: silniejsze γ_t ⇒ większy back-action (|S∞ − ln2|)."""
    z1 = R.kwantowy_zegar(0.005, TICKS=200)
    z2 = R.kwantowy_zegar(0.05, TICKS=200)
    dev1 = abs(z1["S_sys"][-1] - M.LN2)
    dev2 = abs(z2["S_sys"][-1] - M.LN2)
    assert dev2 > dev1
    assert dev1 > 1e-4            # back-action realny (nie zero)
