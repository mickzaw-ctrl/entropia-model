# -*- coding: utf-8 -*-
"""Testy karty eksperymentalnej (R27): parametry platform, detekcja, moc."""
import numpy as np
import pytest

from entropia import experyment as E


def test_platformy_fizyka():
    """Subradiancja: t_D ≫ t_B dla wszystkich platform."""
    for p in E.platformy():
        z = E.licz_platforme(p)
        assert z["tD"] >= 99.0 * z["tB"]        # ciemny ≫ jasny (float)
        assert z["tB"] > 0 and z["tD"] > 0


def test_platforma_B_najlepsza():
    """B (nanofiber) ma najdłuższe t_D (najlepsze okno fazy ciemnej)."""
    tds = [E.licz_platforme(p)["tD"] for p in E.platformy()]
    assert tds[1] == max(tds)


def test_mapa_jednostek():
    """T1 → τ̇ = 0; T2 → τ̇ > 0."""
    pB = [p for p in E.platformy() if p["naz"].startswith("B")][0]
    m = E.mapa_jednostek(pB)
    assert m["tau_dot_T1_nat_s"] == 0.0
    assert m["tau_dot_T2_nat_s"] > 0
    assert m["I_eq_nat"] == 0.1438


def test_detekcja_snr():
    """SNR fotonów subradiantnych > 1 dla A i C; B ~2."""
    dA = E.detekcja(E.platformy()[0])
    dB = E.detekcja(E.platformy()[1])
    dC = E.detekcja(E.platformy()[2])
    assert dA["snr"] > 5
    assert dB["snr"] > 1
    assert dC["snr"] > 5


def test_moc_korelacyjna():
    """M potrzebne do σ_I = 0.01 nat jest skończone i rozsądne."""
    mc = E.moc_korelacyjna()
    assert 50 <= mc["M_min"] <= 500
    assert mc["dI_at_M"] <= mc["sigma_I"]       # osiągnięta precyzja


def test_sekwencja_kompletna():
    """Sekwencja ma wszystkie etapy."""
    pB = [p for p in E.platformy() if p["naz"].startswith("B")][0]
    s = E.sekwencja(pB)
    for k in ["prep", "stan", "jasna", "ciemna", "odczyt", "powt"]:
        assert k in s and s[k]["czas"]
