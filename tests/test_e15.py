# -*- coding: utf-8 -*-
"""Testy ENTROPIA-1.5: pamięć operacyjna, samo-spójny ω_c(T), SPRT."""
import numpy as np
import pytest

from entropia import core as M
from entropia import e15 as E


# ---------- R24: operacyjna pamięć ----------
def test_Cmem_ciemny_lepszy():
    """C_mem(ciemny j=1) > C_mem(jasny) przy N=100; j=0 → 1 bit."""
    _, cm4, _ = E.pamiec_operacyjna(2.0, -2, -1, n=40)
    _, cm100, _ = E.pamiec_operacyjna(50.0, -50, -49, n=40)
    _, cmd, _ = E.pamiec_operacyjna(1.0, -1, 0, n=40)
    assert cm100[20] < cm4[20]                # jasny traci pamięć z N
    assert cmd[20] > cm100[20] * 50           # ciemny znacznie lepszy
    assert cmd[20] > 0.1


def test_Cmem_j12_bit():
    """j=1/2 (dim 2 — nośnik 1 bitu): C_mem > 0 długo (pamięć)."""
    _, cm, _ = E.pamiec_operacyjna(0.5, -0.5, 0.5, n=40)
    assert cm[20] > 0.3          # po t=5 wciąż > 0.3 bita


def test_Cmem_j0_zero():
    """j=0 (dim 1): brak pojemności (nie da się zakodować), stan niezmienniczy."""
    pe, cm, _ = E.pamiec_operacyjna(0.0, 0.0, 0.0, n=30)
    assert pe[0] == 0.5 and cm[0] == 0.0


# ---------- R25: samo-spójny ω_c(T) ----------
def test_omega_okno():
    """ω_c ∈ (T·ln(1/ε), ω_c^max); T_max maleje z g."""
    wc_low, wc_high, Tmax, exists = E.omega_c_okno(0.5, 0.03, eps=0.01)
    assert wc_low > 0 and wc_high > wc_low and exists
    rows = E.tabela_T_max(gammas=(0.1, 0.01))
    assert rows[0]["T_max"] < rows[1]["T_max"]     # słabsze g ⇒ wyższa T_max


def test_back_action_monotoniczny():
    """|S∞−ln2| rośnie z γ_t (saturowane)."""
    vals = E.back_action_fit()
    backs = [b for _, b in vals]
    assert all(backs[i] < backs[i + 1] for i in range(len(backs) - 1))


# ---------- R26: SPRT ----------
def test_sprt_model():
    """Przy parametrach modelu E[N] ≈ 1, błędy ≈ 0."""
    e1, _, err1 = E.E_stop_SPRT(0.001, 0.001, 7.19, n_real=100)
    e2, _, err2 = E.E_stop_SPRT(7.19, 0.001, 7.19, n_real=100)
    assert e1 < 2.0 and e2 < 2.0
    assert err1 < 0.05 and err2 < 0.05


def test_sprt_adaptacyjny():
    """Słabsza separacja ⇒ dłuższy E[N] (SPRT adaptuje się)."""
    e_far, _, _ = E.E_stop_SPRT(7.19, 0.001, 7.19, n_real=100)
    e_near, _, _ = E.E_stop_SPRT(0.1, 0.001, 0.1, n_real=100)
    assert e_near > e_far
