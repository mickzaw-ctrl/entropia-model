# -*- coding: utf-8 -*-
"""Testy ENTROPIA-2.0: sieć z η(T), asymptotyka Petza (Dicke)."""
import numpy as np
import pytest

from entropia import e20 as E


# ---------- R40: sieć z dynamiką η(T) ----------
def test_cykl_uplyw_petla():
    """τ_abs rośnie ≈ 3×budżet; T_signed wraca do 0."""
    s = E.siec_cykliczna(5, eta_min=0.15, n_cyc=300, n_cykli=3, phi_max=0.0)
    assert abs(s["tau_abs"][0][-1] - 3 * s["budget"]) < 0.1 * s["budget"]
    assert abs(s["T_signed"][0][-1]) < 0.05          # pętla (≈0)
    assert s["budget"] > 0.3


def test_siec_jednakowe():
    """Jednakowe komórki: σ ≡ 0."""
    s = E.siec_cykliczna(8, phi_max=0.0)
    assert s["sigma"].max() < 1e-10


def test_offsety_fazowe():
    """Offsety fazowe: σ > 0 i maleją do końca (synchronizacja)."""
    s = E.siec_cykliczna(10, phi_max=0.05)
    assert s["sigma"].max() > 1e-4
    assert s["sigma"][-1] < s["sigma"].max()


# ---------- R41: asymptotyka Petza ----------
def test_F_rec_granica():
    """F_rec ≥ 1/(N+1); C = F_rec − 1/(N+1) słabo zależne od N."""
    a = E.asymptotyka_petza(js=(1.0, 2.0, 4.0, 8.0))
    for r in a["rows"]:
        assert r["F"] >= r["F_lim"] - 1e-9
    assert 0.10 < a["C_mean"] < 0.30             # C ~ 0.2


def test_C_t_maleje():
    """C(t) maleje z czasem (wielowykładniczo)."""
    ct = E.petz_F_czas(2.0, n_steps_list=(10, 40, 160))
    C = [F - 1 / 5 for _, F in ct]
    assert C[0] > C[1] > C[2]
    assert C[-1] < 0.1


def test_ciemny_kontrast():
    """Sektor ciemny (j=0): F_rec = 1; j=1/2 lepszy niż jasny."""
    assert E.petz_F(0.0) == 1.0                   # 1-wym.: kanał identyczności
    F05 = E.petz_F(0.5, n_steps=80)
    F20 = E.petz_F(2.0, n_steps=80)
    assert F05 > F20                              # ciemny zachowuje lepiej
