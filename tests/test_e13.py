# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.3: coherent information, koszt zegara (S-W), protokół."""
import numpy as np
import pytest

from entropia import core as M
from entropia import dicke as D
from entropia import e12 as E12
from entropia import e13 as E13


# ---------- R20: coherent information ----------
def test_Ic_ujemne_klasyczna_pamiec():
    """I_c < 0 przy równowadze, mimo I(A:B) > 0 — pamięć klasyczna."""
    rhos4 = E13.trajektorie_rhosektora(4, 2.0, n=400)
    Ic4 = np.array([E13.I_c_sym(r, 2) for r in rhos4])
    assert Ic4[-1] < 0
    assert abs(Ic4[-1] - (np.log(3) - np.log(5))) < 5e-3
    # I(A:B) > 0 dla tej samej równowagi
    r10 = D.symuluj_dicke(2, M.GAMMA_B, stan=E12.stan_10_N2(), n=200)
    assert r10["I_AB"][-1] > 0.1
    Ic10 = np.log(2) - np.log(12) / 2
    assert Ic10 < 0


def test_T3c_staje():
    """T3c (z |Ī_c|) staje przy równowadze."""
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=E12.stan_10_N2(), n=300)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    f = E12.funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    assert f["T3"][200:].mean() / max(f["T3"][:50].max(), 1e-12) < 1e-3


# ---------- R21: koszt energetyczny zegara ----------
def test_koszt_trojkat():
    """γ_t↑ ⇒ precyzja↑ (Δn/⟨n⟩↓), koszt↑ (E_clk), back-action↑."""
    gts = [0.002, 0.01, 0.05]
    WC = 50.0
    prec, cost, back = [], [], []
    for gt in gts:
        z = E13.zegar_z_energia(gt, WC, TICKS=120)
        nb, dn, Ss = z["nb"][-1], z["dn"][-1], z["Ss"][-1]
        prec.append(dn / max(nb, 1e-9))
        cost.append(WC * nb)
        back.append(abs(Ss - M.LN2))
    assert prec[2] < prec[0]        # precyzja rośnie
    assert cost[2] > cost[0]        # koszt rośnie
    assert back[2] > back[0]        # entropia rośnie


def test_DE_Dtau_nieoznacznosc():
    """ΔE·Δτ ≥ ħ/2 dla ω_c = 50; ω_c^min > 0 i ≤ 50."""
    z = E13.zegar_z_energia(0.01, 50.0, TICKS=120)
    dE, dtau = 50.0 * z["dn"][-1], z["dn"][-1] * M.DELTA_S_Q
    assert dE * dtau >= 0.5            # ħ/2 (ħ=1)
    wc_min = 0.5 / (z["dn"][-1] ** 2 * M.DELTA_S_Q)
    assert 0 < wc_min <= 50.0


# ---------- R22: protokół T1 vs T2 ----------
def test_protokol_rozstrzyga():
    """Po wygaśnięciu fluorescencji: τ̇(T1) = 0, τ̇(T2) = η·I_eq/σ₀ > 1."""
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=E12.stan_10_N2(), n=300)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    f = E12.funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    fluo = dS / E12.SIGMA0
    assert fluo[200:].mean() < 1e-4                  # fluorescencja wygasła
    # T1: τ̇ w ogonie spada o >3 rzędy względem szczytu (stall)
    assert f["T1"][200:].mean() / max(f["T1"][:50].max(), 1e-12) < 1e-3
    assert f["T2"][200:].mean() > 1.0                # T2: tyka dalej
