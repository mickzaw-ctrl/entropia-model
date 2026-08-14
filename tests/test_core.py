# -*- coding: utf-8 -*-
"""Testy rdzenia modelu «ENTROPIA» (czas = entropia, równanie Lindblada)."""
import numpy as np
import pytest

from entropia import core as M


def test_S_monotoniczna():
    """S(ρ) rośnie monotonicznie (mapa unitalna, tw. Ando–Lindblada)."""
    for gamma in (M.GAMMA_A, M.GAMMA_B):
        S, _, _ = M.symuluj(gamma, n=300)
        assert np.all(np.diff(S) >= -1e-9)


def test_S_start_zero():
    """Stan czysty: S(0) = 0."""
    S, _, _ = M.symuluj(M.GAMMA_B, n=20)
    assert abs(S[0]) < 1e-12


def test_S_nasycenie_ln2():
    """S(∞) = ln 2: A maszynowo, B w granicach dyskretyzacji."""
    S_A, _, _ = M.symuluj(M.GAMMA_A, n=400)
    S_B, _, _ = M.symuluj(M.GAMMA_B, n=400)
    assert abs(S_A[-1] - M.LN2) < 1e-8
    assert abs(S_B[-1] - M.LN2) < 5e-4


def test_dekoherencja():
    """Tr(ρ²): 1 → 0.5; |r|: 1 → 0."""
    _, P, R = M.symuluj(M.GAMMA_B, n=400)
    assert abs(P[0] - 1.0) < 1e-12
    assert abs(P[-1] - 0.5) < 5e-4
    r_end = np.linalg.norm(R[-1])
    assert r_end < 0.02


def test_kompresja_27():
    """S_A(t) ≡ S_B(27t) — kompresja czasowa."""
    S_A, _, _ = M.symuluj(M.GAMMA_A, n=400)
    S_B, _, _ = M.symuluj(M.GAMMA_B, n=400)
    idx = np.clip((np.arange(400) * 27).astype(int), 0, 399)
    assert np.max(np.abs(S_A - S_B[idx])) < 1e-3


def test_stosunek_tempa_27():
    """Tempo dS/dt przy dopasowanym S* = 27 dokładnie (postać zamknięta).
    dS_A/dt(t) = 27·dS_B/dt(27t), bo S_A(t) = S_B(27t)."""
    r = (M.dSdt_analityczne(M.GAMMA_A, 1.0) /
         M.dSdt_analityczne(M.GAMMA_B, 27.0))
    assert abs(r - 27.0) < 1e-6


def test_zegar_srednia_rown_a_entropia():
    """⟨T(n)⟩ = S(n): czas JEST entropią (w sensie oczekiwanym)."""
    S_B, _, _ = M.symuluj(M.GAMMA_B, n=200)
    dS = M.delta_entropii(S_B)
    rng = np.random.default_rng(0)
    k = rng.poisson(dS / M.DELTA_S_Q, size=(120, 200))
    mean_T = np.cumsum(k, axis=1).mean(0) * M.DELTA_S_Q
    assert np.max(np.abs(mean_T - S_B)) < 0.02
