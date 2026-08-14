# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.9: protokół różnicowy, zegar w ewoluującym CMB."""
import numpy as np
import pytest

from entropia import e19 as E


# ---------- R37: protokół różnicowy ----------
def test_roznicowy_T1_T2():
    """Δτ per tyknięcie: T1 ≈ 0 (const), T2 ≈ η·I_eq/σ₀ (dryft liniowy)."""
    r = E.protokol_roznicowy(M_B=4, t_dark=30, n_real=100)
    assert abs(r["T1"]["mean"]) < 0.5           # ~0 (offset tła B)
    assert abs(r["T2"]["mean"] - E.TAU2) < 1.5  # ~7.19 nat/tyk


def test_roznicowy_usrednianie():
    """σ(Δτ̄) maleje z M_A (uśrednianie po sieci)."""
    s1 = E.protokol_roznicowy(M_A=1, M_B=4, t_dark=30, n_real=60)["T2"]["std"]
    s8 = E.protokol_roznicowy(M_A=8, M_B=4, t_dark=30, n_real=60)["T2"]["std"]
    assert s8 < s1


def test_sync_jednakowe():
    """Jednakowe komórki: σ ≡ 0, τ_net = T·rate."""
    s = E.synchronizacja_i_net(M=10, T=100)
    assert s["sigma_end"] < 1e-9
    assert abs(s["tau_net"] - 99.0) < 1e-9


# ---------- R38: zegar w ewoluującym CMB ----------
def test_wiek_wszechswiata():
    """t_age(z=0) ≈ 13.8 Gyr; t_age(1) < t_age(0)."""
    zs, age = E.wiek_wszechswiata()
    iz0 = np.argmin(np.abs(zs - 0.001))
    iz1 = np.argmin(np.abs(zs - 1.0))
    assert abs(age[iz0] - 13.8) < 0.5
    assert age[iz1] < age[iz0]


def test_horyzont_zegarow():
    """6/100 GHz nigdy; 1 THz od z≈2.8; wyższe ω_c wcześniej."""
    k6 = E.kiedy_uzyteczny(6e9 / 1e9)
    k100 = E.kiedy_uzyteczny(100e9 / 1e9)
    k1T = E.kiedy_uzyteczny(1e12 / 1e9)
    k10T = E.kiedy_uzyteczny(1e13 / 1e9)
    assert not k6["usable"] and not k100["usable"]
    assert k1T["usable"] and 2.0 < k1T["z_from"] < 3.5
    assert k10T["z_from"] > k1T["z_from"]       # wyższe ω_c użyteczne wcześniej


def test_nbar_cmb():
    """n̄(6 GHz, dziś) > 0.01; n̄(1 THz, dziś) < 0.01."""
    assert E.nbar(2 * np.pi * 6e9, 2.7255) > 0.01
    assert E.nbar(2 * np.pi * 1e12, 2.7255) < 0.01
