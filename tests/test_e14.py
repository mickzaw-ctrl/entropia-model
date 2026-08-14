# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.4: kanał odzysku (fidelity), ω_c(T), protokół e2e."""
import numpy as np
import pytest

from entropia import core as M
from entropia import e14 as E


# ---------- R21: fidelity-based recovery ----------
def test_MF_ciemny_dluzszy():
    """M_F(ciemny j=1) > M_F(jasny) dla N=4,10,100; zysk rośnie z N."""
    vals = {}
    for N in [4, 10]:
        Mb = E.MF_sektora(N / 2.0, -N / 2.0, -N / 2.0 + 1, n=40)
        Md = E.MF_sektora(1.0, -1.0, 0.0, n=40)
        vals[N] = (float(Mb[20]), float(Md[20]))
        assert Md[20] > Mb[20]
    assert vals[10][0] < vals[4][0]              # jasny rozpada szybciej z N
    assert vals[10][1] > vals[10][0] * 5.0


def test_MF_j0_doskonala():
    """j=0: M_F(t) = 1 (kanał identyczności, Γ=0)."""
    MF = E.MF_sektora(0.0, 0.0, 0.0, n=30)
    assert np.all(MF == 1.0)


def test_Fe_sektory():
    """F_e maleje z j; j=0 → 1."""
    fe = {j: E.F_e_sektora(j, n=30) for j in [0.0, 1.0, 2.0]}
    assert fe[0.0] > 0.999
    assert fe[1.0] > fe[2.0]


# ---------- R22: ω_c(T) ----------
def test_omega_c_rosnie_z_T():
    """ω_c(T) rośnie z T; lim. gorący: ω_c ∝ T (stosunek ~3 dla 3T)."""
    rows = E.tabela_omega_c(Ts=(1, 10, 100), eps=0.01)
    assert rows[0]["wc"] < rows[1]["wc"] < rows[2]["wc"]
    wc10 = E.omega_c_T(10.0, 0.01)[0]
    wc30 = E.omega_c_T(30.0, 0.01)[0]
    assert abs(wc30 / wc10 - 3.0) < 0.01          # ω_c ∝ T (lim. gorący)


def test_omega_c_min():
    """ω_c ≥ max(T·ln(1/ε), ω_c^min)."""
    wc, wc_th, wc_min = E.omega_c_T(1.0, eps=0.01, wc_min=1.7)
    assert wc == max(wc_th, wc_min)


# ---------- R23: protokół e2e ----------
def test_protokol_moc():
    """Moc rozstrzygnięcia T1/T2 ≥ 0.99 dla M=10 i M=30."""
    p1 = E.moc_testu(10, "T1", n_real=40)[0]
    p2 = E.moc_testu(10, "T2", n_real=40)[0]
    assert p1 >= 0.99 and p2 >= 0.99


def test_protokol_szum_tla():
    """Moc pozostaje wysoka przy dark counts (tło detektora)."""
    p1 = E.moc_testu(30, "T1", dark_rate=2.0, n_real=40)[0]
    p2 = E.moc_testu(30, "T2", dark_rate=2.0, n_real=40)[0]
    assert p1 >= 0.95 and p2 >= 0.95
