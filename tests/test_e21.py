# -*- coding: utf-8 -*-
"""Testy ENTROPIA-2.1: formalny limit Petza, entrainment faz."""
import numpy as np
import pytest

from entropia import e21 as E


# ---------- R42: formalna asymptotyka Petza ----------
def test_gap_niezalezne_od_N():
    """gap/γ = 1.0000 dla N = 2..100 (utrata pamięci w skali 1/γ)."""
    for N in [2, 4, 10, 50, 100]:
        g = E.gap_sektora(N) / E.M.GAMMA_B
        assert abs(g - 1.0) < 1e-3


def test_F_analityczna_dokladna():
    """F_rec(t) = ½a(2+(1−a)²/(1−½a²)) ≡ numeryka (Δ < 1e-12)."""
    for ns in [10, 40, 160]:
        t = ns * E.M.DELTA_TAU
        an = E.F_rec_analityczna(t, E.M.GAMMA_B, 1)
        nu = E.F_rec_analityczna_Num(1, ns)
        assert abs(an - nu) < 1e-12


def test_F_an_granice():
    """F_rec(0) = 1; F_rec maleje; → 0 dla t→∞."""
    assert abs(E.F_rec_analityczna(0, E.M.GAMMA_B, 1) - 1.0) < 1e-9
    t = np.linspace(0, 200, 100)
    F = [E.F_rec_analityczna(ti, E.M.GAMMA_B, 1) for ti in t]
    assert F[0] > F[10] > F[-1]
    assert F[-1] < 0.1               # F → 0 dla t→∞ (a = e^{−γt} → 0)


# ---------- R43: entrainment faz ----------
def test_entrainment_fazy_lockuja():
    """σ_φ → 0 z g_sync > 0; bez sprzężenia σ_φ stałe."""
    r0 = E.siec_entrainment(10, 300, 3, 0.0, 0.05)
    r1 = E.siec_entrainment(10, 300, 3, 0.05, 0.05)
    assert abs(r0["sigma_phi"][-1] - r0["sigma_phi"][0]) < 0.1
    assert r1["sigma_phi"][-1] < 0.01 * r1["sigma_phi"][0]


def test_entrainment_rosnie_z_g():
    """Silniejsze sprzężenie ⇒ szybszy entrainment (σ_φ niżej wcześniej)."""
    r_s = E.siec_entrainment(10, 300, 3, 0.2, 0.05)
    r_w = E.siec_entrainment(10, 300, 3, 0.01, 0.05)
    n = len(r_s["sigma_phi"])
    # po 1 cyklu silne sprzężenie ma mniejszy σ_φ
    assert r_s["sigma_phi"][300] < r_w["sigma_phi"][300]
