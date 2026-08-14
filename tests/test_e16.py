# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.6: suchy bieg, mapa Petza, zimny zegar."""
import numpy as np
import pytest

from entropia import e16 as E


# ---------- R28: suchy bieg ----------
def test_suchy_bieg_sprt():
    """SPRT w suchym biegu: E[N] ≈ 1, błędy 0, czas skończony."""
    det = E.detektor()
    sb = E.suchy_bieg(dict(naz="B", tau_nat=30.5e-9, N=5e3, OD=None, beta=0.15),
                      det, n_real=60)
    assert sb["t_last"] > 0
    assert sb["T_total"] > 0
    for teor in ["T1", "T2"]:
        w = sb["wyn"][teor]
        assert w["E_N"] < 2.0
        assert w["err"] < 0.05


# ---------- R29: mapa Petza ----------
def test_petz_hierarchia():
    """F_rec maleje z j; ciemne sektory odzyskują lepiej."""
    F05 = E.F_rec_sektora(0.5, m=0)
    F1 = E.F_rec_sektora(1.0, m=0)
    F2 = E.F_rec_sektora(2.0, m=0)
    assert F05 > 0.85
    assert F05 > F1 > F2
    assert F2 > 0.5


def test_petz_j0():
    """j=0: F_rec = 1 (kanał identyczności)."""
    assert E.F_rec_sektora(0.0, m=0) == 1.0


# ---------- R30: zimny zegar ----------
def test_Tmax_widmo():
    """Z cutoffem T_max ≫ 3D; T_max(3D) ∝ g^{−2/3}."""
    T3d = E.T_max_widmo("3d", 0.03)[0]
    Tohm = E.T_max_widmo("ohmic", 0.03, wcut=50.0)[0]
    assert Tohm > 100 * T3d
    # skalowanie 3D: g × 10 ⇒ T_max ÷ 10^{2/3}
    T3d2 = E.T_max_widmo("3d", 0.3)[0]
    assert abs(T3d2 / T3d - 10 ** (-2 / 3)) < 0.05


def test_kosmologia():
    """ω_c termiczna rośnie z T; dziś w zakresie mikrofal."""
    rows = E.tabela_kosmologiczna()
    assert rows[0]["wc"] < rows[1]["wc"] < rows[2]["wc"]
    # dziś: ~10¹² rad/s (2π×~10¹¹ Hz — mikrofale)
    assert 1e11 < rows[0]["wc"] < 1e13
