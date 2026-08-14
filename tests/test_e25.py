# -*- coding: utf-8 -*-
"""
Testy ENTROPIA-5.0 / R50 — pętla pomiarowa na procesorze kwantowym
(IBM Quantum / Google). Liczby z uruchomień (python3 -m entropia.e25).
Uruchomienie: python3 -m pytest tests/test_e25.py -q
"""
import numpy as np
import pytest

from entropia import e25 as E


# =============================================================================
#  Poprawka przygotowania singletu
# =============================================================================
def test_singlet_prep_poprawny():
    """h,cx,z,x → Ψ− (singlet, ciemny); h,cx,x (szkic) → Ψ+ (jasny — BŁĄD)."""
    w = E.weryfikacja_singlet()
    assert w["fid_D"] > 0.999999
    assert w["fid_T0"] < 1e-6
    # szkic użytkownika przygotowuje JASNY tryplet — krytyczny błąd do poprawki
    assert w["fid_T0_bug"] > 0.999999
    assert w["fid_D_bug"] < 1e-6


def test_fizyka_ciemnego():
    """S−|D⟩ = 0 (ciemny); S−|T0⟩ norm² = 2 (superradiancja)."""
    Sm = E.S_minus()
    assert np.linalg.norm(Sm @ E.D) ** 2 < 1e-20
    assert abs(np.linalg.norm(Sm @ E.T0) ** 2 - 2.0) < 1e-12


def test_baza_bella():
    """Ψ− ↔ |11⟩ (witness singletu); Ψ+ ↔ |01⟩."""
    U = E.U_bell()
    pb = np.abs(U @ E.D) ** 2
    assert abs(pb[3] - 1.0) < 1e-12
    pt = np.abs(U @ E.T0) ** 2
    assert abs(pt[1] - 1.0) < 1e-12


# =============================================================================
#  Kanały rozpadu
# =============================================================================
def test_kanały_CPTP():
    """Oba kanały (kolektywny, niezależny) dokładnie CPTP."""
    p = 0.005
    assert E.cp_check(E.kapiel_kolektywna(p)) < 1e-12
    assert E.cp_check(E.kapiel_niezalezna(p)) < 1e-12


def test_ciemny_pod_kolektywna():
    """Kolektywna: P_D(t) = 1 (ciemny — M = 1) przez całą ewolucję."""
    p = 0.02 * 0.25
    M0, M1 = E.kapiel_kolektywna(p)
    rho = np.outer(E.D, E.D.conj())
    for _ in range(100):
        rho = M0 @ rho @ M0.conj().T + M1 @ rho @ M1.conj().T
    assert abs(np.real(E.D.conj() @ rho @ E.D) - 1.0) < 1e-10


def test_jasny_superradiancja():
    """Kolektywna: |T0⟩ rozpada się ≈ e^{−2γt} (superradiancja 2γ)."""
    g, dt, t_max = 0.02, 0.25, 25.0
    p = g * dt
    M0, M1 = E.kapiel_kolektywna(p)
    rho = np.outer(E.T0, E.T0.conj())
    for _ in range(int(t_max / dt)):
        rho = M0 @ rho @ M0.conj().T + M1 @ rho @ M1.conj().T
    P = np.real(E.T0.conj() @ rho @ E.T0)
    assert abs(P - np.exp(-2 * g * t_max)) < 0.01


def test_ciemny_pod_niezalezna():
    """Niezależna (kontrola falsyfikacyjna): P_D ≈ e^{−γt} (nie 2γ — koherencja)."""
    g, dt, t_max = 0.02, 0.25, 25.0
    p = g * dt
    rho = np.outer(E.D, E.D.conj())
    for _ in range(int(t_max / dt)):
        rho = E.krok(rho, p, E.kapiel_niezalezna(p))
    P = np.real(E.D.conj() @ rho @ E.D)
    assert abs(P - np.exp(-g * t_max)) < 0.01
    # rozróżnienie kąpieli: P_D(kolektywna) ≫ P_D(niezależna)
    pr = E.przewidywania()
    assert pr["P_D_kol_end"] - pr["P_D_nz_end"] > 0.3


def test_odblokowanie_rz():
    """rz na q0 łamie ciemność: P_D maleje z Δω (przeciek |D⟩→|T0⟩→rozpad)."""
    pr = E.przewidywania()
    assert pr["P_D_rz_end"] < pr["P_D_kol_end"] - 0.2


def test_obwod_z_ancilla():
    """Osadka unitarna V: Tr_a[V(ρ⊗|0⟩⟨0|)V†] ≡ kanał Krausa (dokładnie)."""
    p = 0.005
    assert E.obwod_vs_kraus(p) < 1e-10


# =============================================================================
#  Tomografia
# =============================================================================
def test_tomografia_bell_szum():
    """Estymata z bazy Bella: P̂_D w 3σ od dokładnego."""
    rng = np.random.default_rng(5)
    U0, _ = E.singlet_prep()
    rho0 = np.outer(U0 @ E.KET00, (U0 @ E.KET00).conj())
    p = 0.02 * 0.25
    rho = rho0.copy()
    for _ in range(50):
        rho = E.krok(rho, p, E.kapiel_niezalezna(p))
    p_hat, sig, _ = E.tomografia_bell(rho, 20000, seed=7)
    p_ex = E.prawd_bell(rho)
    assert abs(p_hat[3] - p_ex[3]) < 3 * sig[3] + 1e-3


def test_pomiar_losowy_rekonstrukcja():
    """Rekonstrukcja ρ z pomiarów losowych (LS): F(ρ, ρ_est) wysoka."""
    U0, _ = E.singlet_prep()
    rho0 = np.outer(U0 @ E.KET00, (U0 @ E.KET00).conj())
    w = E.pomiar_losowy(rho0, 16000, seed=3)
    rho_est = E.rekonstrukcja_LS(w)
    F = float(np.real(np.trace(rho0.conj().T @ rho_est)))
    assert F > 0.9


# =============================================================================
#  Protokół i sprzęt
# =============================================================================
def test_protokol_petla():
    """Pętla użytkownika (suchy bieg): z szumem strzałowym P̂_D zgodne z P_D."""
    z = E.symuluj_protokol(10.0, 0.25, 0.02, "niezalezna", 0.0, shots=4000,
                           seed=11)
    assert np.all(np.abs(z["P_hat"] - z["P_ex"]) < 3 * z["P_sig"] + 1e-2)


def test_hardware_budzet():
    """Budżet sprzętowy: co najmniej kilka kroków na obu platformach."""
    hw = E.hardware_zestawienie()
    for pl in hw["platformy"]:
        assert pl["max_krokow_zakres"][1] >= 3
    assert hw["shots_1pct"] <= 2500


def test_przewidywania_kluczowe():
    """Kluczowe liczby protokołu (kotwice)."""
    pr = E.przewidywania()
    assert abs(pr["P_D_kol_end"] - 1.0) < 1e-6
    assert 0.5 < pr["P_D_nz_end"] < 0.7        # e^{−γ·25} ≈ 0.607
    assert 0.2 < pr["P_T0_kol_end"] < 0.5      # e^{−2γ·25} ≈ 0.368
