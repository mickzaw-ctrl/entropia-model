# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.2: konkurencja funkcjonałów, odzyskiwalność, fizyczny 27×."""
import numpy as np
import pytest

from entropia import core as M
from entropia import dicke as D
from entropia import e12 as E


# ---------- cztery funkcjonały ----------
def test_funkcjonaly_stall():
    """T0, T1, T3 stają przy równowadze; T2 (absolutna I) nie."""
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=E.stan_10_N2(), n=300)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    dR = np.zeros_like(dS)
    f = E.funkcjonaly_czasu(dS, I, dI, dR)
    tail = slice(200, 300)
    # stall = τ̇ spada o >4 rzędy względem szczytu (nie zero maszynowe)
    for k in ["T0", "T1", "T3"]:
        assert f[k][tail].mean() / max(f[k][:50].max(), 1e-12) < 1e-4
    # T2 NIE staje: τ̇∞ >> τ̇ T1 (pamięć I_eq napędza czas)
    assert f["T2"][tail].mean() > 100 * f["T1"][tail].mean()
    assert f["T2"][tail].mean() > 0.1


def test_funkcjonaly_T1_dynamiczny():
    """T1 wymaga İ ≠ 0: w równowadze (İ=0) staje, choć I_eq > 0."""
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=E.stan_10_N2(), n=300)
    I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    assert I[-1] > 0.1                          # I_eq > 0 (pamięć)
    assert dI[-50:].max() < 1e-5                # ale İ → 0


# ---------- odzyskiwalność ----------
def test_odzyskiwalnosc_ciemny_lepszy():
    """M_dark(j=1) > M_bright dla N=4,10,100; zysk rośnie z N."""
    rows = E.tabela_odzyskiwalnosci(Ns=(4, 10, 100), n=60)
    for row in rows:
        assert row["Md50"] > row["Mb50"]
    r4 = rows[0]; r100 = rows[2]
    assert r100["gain"] > r4["gain"]            # przewaga rośnie z N
    assert r100["gain"] > 5.0


def test_odzyskiwalnosc_j0_doskonala():
    """j=0 (parzyste N): M(t) = 1 (sektor 1-wymiarowy, Γ=0)."""
    # N=2 singlet: stan invariantny — D(t) = D(0)
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    SING = (np.kron(ket1, ket0) - np.kron(ket0, ket1)) / np.sqrt(2)
    res = D.symuluj_dicke(2, M.GAMMA_B,
                          stan={0.0: (1.0, np.ones((1, 1), complex))},
                          n=50)
    assert res["S"][-1] < 1e-10                  # singlet: S = 0 (zegar milczy)


# ---------- fizyczny 27× ----------
def test_27_fizyczny_3d():
    """Kąpiel 3D fotonowa (γ∝T³): R_T ≈ 27 (lim. gorący)."""
    r = E.test_27_fizyczny(TB=30.0)
    assert abs(r["r_3d"] - 27.0) < 1.5          # 27.2 przy TB=30


def test_27_fizyczny_single_mode():
    """Single-mode (γ∝(2n̄+1)): R_T ≈ 3, nie 27."""
    r = E.test_27_fizyczny(TB=30.0)
    assert abs(r["r_single"] - 3.0) < 0.5


def test_27_zbieznosc():
    """R_T(3D) → 27 z gorącym limitem (T_B → ∞)."""
    zb = E.zbieznosc_27(TBs=(10, 100))
    assert zb[1][1] < zb[0][1]                  # maleje do 27
    assert abs(zb[1][1] - 27.0) < 0.2
