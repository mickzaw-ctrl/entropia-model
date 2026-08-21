# -*- coding: utf-8 -*-
"""
Testy ENTROPIA-6.0 (R51: kosmologiczna kalibracja zegara — Λ jako źródło
jednostki czasu; zapis pomiaru = jeden bit @ temperatura horyzontu de
Sittera). Liczby w asercjach pochodzą z uruchomień (python3 -m entropia.e26)
— wyliczenia deterministyczne ze stałych fundamentalnych (bez RNG).
Uruchomienie: python3 -m pytest tests/test_e26.py -q
"""
import numpy as np
import pytest

from entropia import e26 as E


# =============================================================================
#  R51 — kalibracja kosmologiczna
# =============================================================================
def test_temperatura_gibbons_hawking_rzad_wielkosci():
    """T_dS dla Λ_Planck2018 powinna wypadać ~2.2e-30 K (literatura)."""
    T = E.temperatura_gibbons_hawking(E.LAMBDA_PLANCK2018)
    assert 1e-30 < T < 5e-30


def test_entropia_horyzontu_rzad_wielkosci():
    """S_dS (horyzont de Sittera) ~ 10^122 bit — znana wartość literaturowa
    entropii horyzontu obserwowalnego Wszechświata."""
    S = E.entropia_horyzontu_bity(E.LAMBDA_PLANCK2018)
    assert 1e121 < S < 1e124


def test_czas_hubble_bliski_wiekowi_wszechswiata():
    """t_H (asymptotyczny, dS) powinien być tego samego rzędu co wiek
    Wszechświata (13.8 Gyr), lecz nieco większy (ΩΛ < 1 dzisiaj)."""
    w = E.kalibracja_kosmologiczna(E.LAMBDA_PLANCK2018)
    assert 10.0 < w["t_H_Gyr"] < 25.0
    assert w["t_H_Gyr"] > E.AGE_UNIVERSE_GYR


def test_tau_rec_dluzszy_niz_wiek_wszechswiata():
    """Wynik kluczowy R51: czas zapisu JEDNEGO bitu przy T_dS (Margolus-
    Levitin) jest DŁUŻSZY niż wiek Wszechświata — horyzont de Sittera
    sam z siebie nie zdążyłby jeszcze zarejestrować pełnego bitu."""
    w = E.kalibracja_kosmologiczna(E.LAMBDA_PLANCK2018)
    tau_rec_Gyr = w["tau_ML_Gyr"]
    assert tau_rec_Gyr > E.AGE_UNIVERSE_GYR
    assert 50.0 < tau_rec_Gyr < 500.0                 # ~172 Gyr oczekiwane


def test_landauer_mniejszy_od_margolus_levitin():
    """Granica Landauera (dolna, łagodniejsza) < granica Margolus-Levitin
    (kwantowa granica prędkości) dla tej samej temperatury — stały
    czynnik geometryczny (ln2/π vs π/2)."""
    w = E.kalibracja_kosmologiczna(E.LAMBDA_PLANCK2018)
    assert w["tau_Landauer"] < w["tau_ML"]
    stosunek = w["tau_ML"] / w["tau_Landauer"]
    oczekiwany = (np.pi / 2) / (np.log(2) / np.pi)
    assert abs(stosunek - oczekiwany) / oczekiwany < 1e-9


def test_kappa_cosmo_dodatnia_i_skonczona():
    """κ_cosmo (s/nat) — kalibracja core.py — musi być dodatnia, skończona,
    i dawać Δt=κ·ln2 zgodne z τ_rec dla jednego pełnego bitu core."""
    w = E.kalibracja_kosmologiczna(E.LAMBDA_PLANCK2018)
    assert w["kappa_cosmo"] > 0
    assert np.isfinite(w["kappa_cosmo"])
    assert abs(w["kappa_cosmo"] * np.log(2) - w["tau_ML"]) < 1e-6 * w["tau_ML"]


def test_falsyfikacja_temperatury_cmb_vs_dS():
    """R51 — test falsyfikacji: T_CMB musi być O WIELE WIĘKSZA niż T_dS
    (o >25 rzędów wielkości), inaczej materia nie mogłaby 'tykać' szybciej
    niż horyzont — sprzeczne z obserwowaną strukturą Wszechświata."""
    fals = E.test_falsyfikacji_temperatury(E.LAMBDA_PLANCK2018)
    assert fals["zgodne"] is True
    assert fals["stosunek"] > 1e25


def test_planck_vs_desi_male_odchylenie():
    """Wyniki dla Λ_Planck2018 i Λ_DESI2024 (różne pomiary tej samej
    stałej) powinny się różnić o < 5% — model jest stabilny względem
    niepewności obserwacyjnej Λ."""
    wp = E.kalibracja_kosmologiczna(E.LAMBDA_PLANCK2018)
    wd = E.kalibracja_kosmologiczna(E.LAMBDA_DESI2024)
    for key in ("T_dS", "S_dS_bits", "tau_ML", "t_H"):
        diff = abs(wp[key] - wd[key]) / wp[key]
        assert diff < 0.05


def test_l_p_wartosc_referencyjna():
    """Długość Plancka — wartość literaturowa ~1.616e-35 m."""
    lP = E.dlugosc_plancka()
    assert abs(lP - 1.6163e-35) / 1.6163e-35 < 1e-3


def test_figura_generuje_plik():
    """figura_51() zapisuje plik PNG bez błędów."""
    wp = E.kalibracja_kosmologiczna(E.LAMBDA_PLANCK2018)
    wd = E.kalibracja_kosmologiczna(E.LAMBDA_DESI2024)
    path = E.figura_51(wp, wd)
    import os
    assert os.path.exists(path)
