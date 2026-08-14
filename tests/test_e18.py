# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.8: realizacja Petza, zegar w CMB, sieć zegarów."""
import numpy as np
import pytest

from entropia import e18 as E


# ---------- R34: realizacja mapy Petza ----------
def test_petz_lepszy_od_klasycznego():
    """Petz > klasyczny dla kodów fazowych (wszystkie sektory)."""
    for j in [0.5, 1.0, 1.5, 2.0]:
        r = E.protokoly_odzysku(j)
        assert r["petz"] > r["klasyczny"]
        assert r["klasyczny"] == 0.5            # pomiar fazy losowy


def test_echo_zalezy_od_sektora():
    """Echo pomaga dla j≤1 (dekoherencja czysta), szkodzi dla j≥1.5."""
    r05 = E.protokoly_odzysku(0.5)
    r20 = E.protokoly_odzysku(2.0)
    assert r05["echo"] > r05["klasyczny"]
    assert r20["echo"] < r20["klasyczny"]


# ---------- R35: zegar w CMB ----------
def test_prog_CMB():
    """n̄ < 0.01 ⇒ ω_c/2π ≥ ~261 GHz."""
    wcmin = E.wc_min_CMB()
    assert abs(wcmin / 2 / np.pi / 1e9 - 261) < 10
    assert E.nbar(wcmin, 2.7255) > 0.009       # próg osiągnięty


def test_CMB_bezpieczenstwo():
    """100 GHz: szum (n̄>0.01); 1 THz: bezpieczny."""
    assert E.nbar(2 * np.pi * 100e9, 2.7255) > 0.01
    assert E.nbar(2 * np.pi * 1e12, 2.7255) < 0.01


def test_cutoff_grawitacyjny():
    """J(cutoff)/J(3D) ≈ 1 dla ω_c ≪ ω_Planck; maleje przy ω_Planck."""
    wG = E.W_G
    assert abs(E.J_cutoff(1e12) / 1e36 - 1.0) < 1e-3
    # przy ω = ω_Planck: J = g²ω³/2 (cutoff redukuje o połowę)
    assert abs(E.J_cutoff(wG) / wG ** 3 - 0.5) < 1e-3


# ---------- R36: sieć zegarów ----------
def test_synchronizacja():
    """σ_end maleje z g_sync; τ_net niezależny od sprzężenia."""
    rng = np.random.default_rng(0)
    rates = 0.5 + 0.5 * rng.random(20)
    a0 = E.analiza_sieci(0.0, rates)
    a2 = E.analiza_sieci(0.2, rates)
    assert a2["sigma_end"] < 0.1 * a0["sigma_end"]
    assert abs(a0["tau_net"] - a2["tau_net"]) < 1e-9


def test_jednakowe_T():
    """Jednakowe tempo ⇒ σ ≡ 0 bez sprzężenia."""
    tau = E.siec_zegarow(0.0, np.ones(10))
    assert tau.std(axis=1).max() < 1e-9
