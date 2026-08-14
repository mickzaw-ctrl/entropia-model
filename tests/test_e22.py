# -*- coding: utf-8 -*-
"""Testy ENTROPIA-3.0: drabina Dickego (dowód C(t)), metryka FRW."""
import numpy as np
import pytest

from entropia import e22 as E


# ---------- R44: drabina Dickego ----------
def test_drabina_gamma():
    """Γ_1 = Nγ; Γ_n = n(N−n+1)γ (symetria: Γ_n = Γ_{N−n}); gap spektralny = γ."""
    for N in [4, 10, 100]:
        ns, G = E.widmo_drabiny(N)
        assert abs(G[0] / E.M.GAMMA_B - N) < 1e-9        # Γ_1 = Nγ
        # Γ_2 = 2(N−1)γ
        assert abs(G[1] / E.M.GAMMA_B - 2 * (N - 1)) < 1e-9
        # wzór dokładny: Γ_n = n(N−n+1)γ dla wszystkich szczebli
        for i, n in enumerate(ns):
            assert abs(G[i] / E.M.GAMMA_B - n * (N - n + 1)) < 1e-9


def test_okno_C():
    """Okno uniwersalności: (1/(Nγ), 1/γ) — rozszerza się z N."""
    t1_2, t2 = E.okno_C(2)
    t1_100, _ = E.okno_C(100)
    assert t2 == 1 / E.M.GAMMA_B
    assert t1_100 < t1_2


def test_F_drabinowe():
    """F_rec(0) = 1; maleje z t i N."""
    assert abs(E.F_rec_drabinowe(2, 0) - 1.0) < 1e-9
    assert E.F_rec_drabinowe(2, 10) > E.F_rec_drabinowe(2, 20)
    assert E.F_rec_drabinowe(100, 10) < E.F_rec_drabinowe(2, 10)


# ---------- R45: FRW ----------
def test_wiek_frw():
    """t(z=0) ≈ 13.8 Gyr; t(z=1) < t(0); t(1100) ≈ 0."""
    assert abs(E.t_wiek(0) - 13.8) < 0.3
    assert E.t_wiek(1) < E.t_wiek(0)
    assert E.t_wiek(1100) < 0.1


def test_s_komobowa():
    """s·a³ = const: s(T)/s₀ = (T/T₀)³ = (1+z)³."""
    assert abs(E.s_komobowa(2) - 27.0) < 1e-9
    assert abs(E.s_komobowa(0) - 1.0) < 1e-12


def test_horyzont():
    """S_BH na horyzoncie rośnie wstecz (S_H(1100) < S_H(0))."""
    S0 = E.S_horyzont(0)
    S1100 = E.S_horyzont(1100)
    assert np.log10(S0) > 100            # ~10¹⁴⁰ k_B (rząd E&L 10¹²²)
    assert S1100 < S0                    # mniejszy horyzont wcześniej
