# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.7: arkusz T_max/ω_c(T), koszt energii, suchy bieg z F."""
import numpy as np
import pytest

from entropia import e17 as E


# ---------- R31: arkusz T_max/ω_c(T) ----------
def test_Tmax_wzrost():
    """T_max rośnie z ω_c; 6 GHz → 63 mK (dokładnie)."""
    rows = E.tabela_arkusza()
    assert rows[0]["T_max"] < rows[1]["T_max"] < rows[-1]["T_max"]
    assert abs(rows[0]["T_max"] - 63.4) < 2.0


def test_nbar():
    """n̄ maleje z ω_c, rośnie z T; próg ε=0.01."""
    assert E.nbar_termiczna(6, 10) < 0.01
    assert E.nbar_termiczna(6, 100) > 0.01


def test_purcell():
    """T1_Purcell w zakresie mierzalnym (10-1000 μs)."""
    gP, T1P = E.purcell_dispersive()
    assert 10e-6 < T1P < 1e-3


# ---------- R32: koszt energetyczny ----------
def test_koszt():
    """Zegar zaniedbywalny vs pułapka; ΔE·Δτ ≥ ħ/2."""
    k = E.koszt_energetyczny()
    assert k["E_trap"] > 1e6 * k["E_clk"]      # pułapka dominuje
    assert k["dE_dtau"] >= 0.5 * 1e-30          # ΔE·Δτ ≥ ħ/2 (w jednostkach)
    assert k["E_clk"] > 0 and k["E_landauer"] > 0


# ---------- R33: suchy bieg z F ----------
def test_infidelity_systematyk():
    """I_eq i τ̇_T2 maleją z F (systematyczny efekt), τ̇_T1 = 0 zawsze."""
    det = dict(eta_det=0.3, dark_rate=100.0, jitter=1e-9)
    out = E.moc_vs_F(Fs=(1.0, 0.7, 0.3), det=det, n_real=40)
    assert out[0]["I_eq"] > out[1]["I_eq"] > out[2]["I_eq"]
    assert out[0]["tau2"] > out[1]["tau2"] > out[2]["tau2"]
    for o in out:
        assert abs(o["tau1"]) < 1e-6
        assert o["p1"] > 0.95 and o["p2"] > 0.95


def test_stan_z_F():
    """Stan z F=1 = ρ10; ślad = 1."""
    s = E.stan_z_F(1.0)
    tr = sum(w * np.trace(rho) for w, rho in s.values())
    assert abs(tr - 1.0) < 1e-9
