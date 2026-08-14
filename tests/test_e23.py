# -*- coding: utf-8 -*-
"""Testy R46: dowód wzoru Petza z regularyzacją."""
import numpy as np
import pytest

from entropia import e23 as E


# ---------- Tw.1: dokładny wzór ----------
def test_Tw1_dokladny():
    """F_rec = ½a(2+(1−a)²/(1−½a²)) — Δ < 1e-12."""
    for ns in [5, 20, 40, 160]:
        k = E.krok_1_7(E.M.GAMMA_B, ns)
        assert abs(k["F_rec"] - E.F_an(k["a"])) < 1e-12


def test_Tw1a_stable():
    """F_stable = (1−½a)/(1−½a²) — Δ < 1e-12."""
    k = E.krok_1_7(E.M.GAMMA_B, 40)
    assert abs(k["F_stab"] - E.F_stable_an(k["a"])) < 1e-12


def test_Tw1_granice():
    """F_rec(0)=1; F_rec maleje; F_stable ∈ [F_rec, 1]."""
    assert abs(E.F_an(1.0) - 1.0) < 1e-9
    assert E.F_an(0.5) > E.F_an(0.2)
    assert E.F_stable_an(0.5) > E.F_an(0.5)
    assert E.F_stable_an(0.5) <= 1.0 + 1e-9


# ---------- Tw.2: N≥2 zimna ----------
def test_Tw2_N_element():
    """|⟨0-exc|S₋|1-exc⟩|² = N; populacja 1-exc = e^{−Nγt}."""
    for N in [2, 4, 10]:
        assert E.element_Sminus(N) == N
        a = E.populacja_1exc(N, 40)
        assert abs(a - np.exp(-N * E.M.GAMMA_B * 10)) < 1e-9


def test_Tw2_proj_wzor():
    """Petz rzutowany = F_an z Γ=Nγ."""
    for N in [2, 4]:
        Fp, Fs_ = E.petz_proj_wzor(N, 160)
        assert abs(Fp - E.F_an(np.exp(-N * E.M.GAMMA_B * 40))) < 1e-12
        assert Fs_ > Fp                      # stabilny lepszy niż rozpadający


# ---------- Tw.3: regularyzacja ----------
def test_Tw3_regularizacja():
    """Pełny ε→0 istnieje; zimna: pełny ≈ średnia rzutowana."""
    for N in [2, 4]:
        Fp, Fs_ = E.petz_proj_wzor(N, 160)
        Ffull = E.petz_regularized(N, 160, 1e-6, bath="cold")
        avg = (Fp + Fs_) / 2
        assert abs(Ffull - avg) < 0.02      # przeciek domyka do średniej


def test_Tw3_hot_granica():
    """Gorąca: F_rec > 1/(N+1) (nadwyżka C>0); ε→0 stabilne."""
    for N in [2, 4]:
        F1 = E.petz_regularized(N, 160, 1e-3, bath="hot")
        F2 = E.petz_regularized(N, 160, 1e-6, bath="hot")
        assert abs(F1 - F2) < 0.02          # granica ε→0 istnieje
        assert F2 > 1 / (N + 1)             # C(t) > 0
