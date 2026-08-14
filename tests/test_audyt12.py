# -*- coding: utf-8 -*-
"""
Testy AUDYTU ZAMYKAJĄCEGO ENTROPIA-1.2.
Każdy test opiera się na implementacji-świadku (niezależna konwencja /
integrator) porównanej z projektem; liczby w asercjach pochodzą z uruchomień.
Uruchomienie: python3 -m pytest tests/test_audyt12.py -q
"""
import numpy as np
import pytest

from entropia import core as M
from entropia import dicke as D
from entropia import e12 as E
from entropia import audyt12 as A


# =============================================================================
#  1. AUDYT RÓWNAŃ
# =============================================================================
def test_lindblad_wiersz_vs_kolumna():
    """Konwencja wierszowa (projekt) ≡ kolumnowa (świadek) — dokładnie."""
    sz, sp, sm = M.operatory()
    Hq = (M.OMEGA / 2.0) * sz
    L_row = M.superoperator(0.123, 0.246, M.OMEGA)
    L_col = A.niez_superoperator_kolumnowy(Hq, [sm, sp, sz],
                                           [0.123, 0.123, 0.246])
    d = 2
    P = np.zeros((4, 4))
    for i in range(d):
        for j in range(d):
            P[i * d + j, j * d + i] = 1.0
    assert np.max(np.abs(L_row - P @ L_col @ P.T)) < 1e-14


def test_cp_trace_hermit_pozytywnosc():
    """e^{L·τ} jest CPTP: ślad=1, hermitowskość, dodatniość (20×5 prób)."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        g = float(rng.uniform(0.01, 0.3))
        U = __import__("scipy.linalg", fromlist=["expm"]).expm(
            M.superoperator(g, 2.0 * g, M.OMEGA) * M.DELTA_TAU)
        for _2 in range(5):
            v = rng.normal(size=2) + 1j * rng.normal(size=2)
            v /= np.linalg.norm(v)
            r1 = M._unvec(U @ M._vec(np.outer(v, v.conj())))
            assert abs(np.trace(r1) - 1.0) < 1e-12
            assert np.max(np.abs(r1 - r1.conj().T)) < 1e-12
            ev = np.linalg.eigvalsh((r1 + r1.conj().T) / 2.0)
            assert ev.min() > -1e-12


def test_rk4_vs_analityk():
    """RK4 Blocha (wyprowadzenie ręczne) ≡ postać zamknięta S(γ,t)."""
    ts, S_rk4 = A.niez_rk4_bloch(M.GAMMA_B, 2.0 * M.GAMMA_B, M.OMEGA,
                                 (M.N_TICKS - 1) * M.DELTA_TAU,
                                 M.DELTA_TAU / 16.0)
    S_an = np.array([M.S_analityczne(M.GAMMA_B, t) for t in ts])
    assert np.max(np.abs(S_rk4 - S_an)) < 1e-10


def test_dSdt_formuly_zgodne():
    """dS/dt: dwie niezależne postacie zamknięte — zgodne do 1e-12."""
    tg = np.linspace(0.002, 2.0, 200)
    v1 = np.array([M.dSdt_analityczne(M.GAMMA_B, t) for t in tg])
    v2 = np.array([A.niez_dSdt_an(M.GAMMA_B, t) for t in tg])
    assert np.max(np.abs(v1 - v2)) < 1e-10
    # dodatniość (produkcja entropii ≥ 0 — tw. Spohna)
    assert np.all(v1 >= 0)


def test_sektory_vs_pelna_konwencja():
    """Ten sam schemat krokowy, konwencja kolumnowa vs projekt: ~1e-15."""
    S_dysk, I_dysk = A.niez_n2_dyskretny(n=400)
    proj = D.symuluj_dicke(2, M.GAMMA_B, stan=E.stan_10_N2(), n=400)
    assert np.max(np.abs(S_dysk - proj["S"])) < 1e-12
    assert np.max(np.abs(I_dysk - proj["I_AB"])) < 1e-12


def test_Ieq_analityczne():
    """I_eq = ln(2/√3) — wyprowadzenie analityczne vs numeryka."""
    niez = A.niez_n2_funkcjonaly(n=400)
    assert abs(niez["I"][-1] - np.log(2.0 / np.sqrt(3.0))) < 1e-8
    assert abs(niez["S"][-1] - 0.5 * np.log(12.0)) < 1e-8


# =============================================================================
#  2. AUDYT JEDNOSTEK
# =============================================================================
def test_jednostki_kalibracje():
    """Kalibracje fizyczne: k_BT_CMB/h, próg n̄<0.01, ω_G = m_P c²/ħ."""
    r = A.audyt_jednostki()
    assert abs(r["kBT_h_GHz"] - 56.8) < 0.1          # 56.79 GHz
    assert 255 < r["prog_eps_GHz"] < 265             # 261.5 GHz (R30: 261)
    assert abs(r["omega_Planck_rad_s"] - 1.855e43) / 1.855e43 < 1e-3
    assert r["sigma0_eq_ds"]                          # σ₀ = δs = 0.01
    assert abs(r["gamma_ratio"] - 27.0) < 1e-12       # γ_A/γ_B = 27


# =============================================================================
#  3. NIEZALEŻNA REPLIKACJA
# =============================================================================
def test_replikacja_core():
    """Rdzeń: S(t) ~1e-14; kompresja 27× (porządna) ~1e-9; tempo = 27.000."""
    r = A.replikacja_core()
    assert r["S_B_max_abs_diff"] < 1e-12
    assert r["kompresja_27_max_abs"] < 1e-6          # pełny zakres B
    # projekt: przycięty indeks 27n ⇒ sztuczny błąd ~4e-5 (artefakt)
    assert r["kompresja_27_proj_obcieta"] > 1e-5
    assert all(abs(s - 27.0) < 1e-6 for s in r["stosunek_tempa"])


def test_replikacja_e12():
    """T0–T3: niezależna pełna przestrzeń vs sektory — ogony i nachylenie T2."""
    r = A.replikacja_e12()
    for k in ["T0", "T1", "T3"]:
        assert abs(r[f"tau_{k}_tail_proj"] - r[f"tau_{k}_tail_niez"]) < 1e-9
    assert abs(r["T2_slope_niez"] - r["T2_slope_oczek"]) < 1e-6
    assert abs(r["T2_slope_niez"] - 7.192052) < 1e-3


def test_replikacja_odzyskiwalnosc():
    """M(50): ciemny j=1 i jasny — projekt vs świadek (RK45 na stopach)."""
    r = A.replikacja_odzyskiwalnosc()
    assert abs(r["M_dark_niez"] - r["M_dark_closed"]) < 1e-10
    assert abs(r["M_dark_proj"] - r["M_dark_closed"]) < 1e-10
    for row in r["rows"]:
        assert abs(row["M_bright_niez"] - row["M_bright_proj"]) < 1e-4


def test_replikacja_27_limity():
    """R_T: spójny termiczny → 27 (3D) i 3 (single) w gorącym limicie.
    R47: znalezisko nr 1 ZAMKNIĘTE — poprawiony R_T_fizyczny ≡ świadek."""
    r = A.replikacja_27()
    assert abs(r[100]["spojny_3d"] - 27.0) < 0.3
    assert abs(r[100]["spojny_single"] - 3.0) < 0.05
    # zbieżność: monotonicznie do limitu
    v = [r[t]["spojny_3d"] for t in sorted(r)]
    assert v[-1] < v[0]
    # po R47 projekt (spójna termiczna) zgadza się ze świadkiem do ~1e-10
    for TB in r:
        assert abs(r[TB]["proj_3d"] - r[TB]["spojny_3d"]) < 1e-6
        assert abs(r[TB]["proj_single"] - r[TB]["spojny_single"]) < 1e-6
    # kontrola: TB=10 3D = 27.850, single = 3.092 (wartości po poprawce)
    assert abs(r[10]["proj_3d"] - 27.850) < 0.01
    assert abs(r[10]["proj_single"] - 3.092) < 0.01


# =============================================================================
#  4. TEST T1/T2
# =============================================================================
def test_T0_T1_T3_staja_T2_nie():
    """Przy równowadze T0/T1/T3 stają, T2 tyka z nachyleniem η·I_eq/σ₀."""
    r = A.test_T1_T2()
    for k in ["T0", "T1", "T3"]:
        assert r["tau_dot_tails"][k] < 1e-6
    assert r["tau_dot_tails"]["T2"] > 5.0
    assert abs(r["slope_T2"] - r["slope_oczekiwane"]) < 1e-6
    assert abs(r["slope_T2"] - 7.192052) < 1e-3


def test_funkcjonaly_monotoniczne():
    """Żaden funkcjonał nie cofa czasu: τ̇ ≥ 0 dla wszystkich n."""
    r = A.test_T1_T2()
    assert all(r["monotoniczne"].values())
    assert r["dS_nonneg"]


def test_czkanie_T2_niszczy():
    """W ogonie: T0/T1/T3 → 100% tyknięć zerowych (czas stoi); T2 → 0%."""
    r = A.test_T1_T2()
    assert r["czkanie_zerowe_T0"] > 0.99
    assert r["czkanie_zerowe_T1"] > 0.99
    assert r["czkanie_zerowe_T3"] > 0.99
    assert r["czkanie_zerowe_T2"] < 0.01


# =============================================================================
#  5. TEST ODZYSKIWALNOŚCI DARK-SEKTOR
# =============================================================================
def test_dark_zamknieta_forma():
    """M(j=1, t) = ½(e^{−2γt} + e^{−6γt}) — zamknięta forma (wart. własne 0,−2γ,−6γ)."""
    r = A.test_dark_sektor()["M_dark"]
    assert abs(r["closed"] - r["niez"]) < 1e-12
    assert abs(r["closed"] - r["proj"]) < 1e-12
    assert abs(r["closed"] - 0.414830) < 1e-4


def test_dark_niezaleznosc_od_N():
    """j=1: sektor 3-wymiarowy ⇒ M identyczne dla każdego N (rozrzut 0)."""
    r = A.test_dark_sektor()
    assert r["M_dark_Ns_rozrzut"] < 1e-15


def test_jasny_ciemny_porzadek():
    """M_dark > M_bright dla N=4,10,100; zysk rośnie z N (N=100: ~31.6×)."""
    r = A.test_dark_sektor()
    rows = r["rows"]
    assert rows[0]["M_bright"] > rows[1]["M_bright"] > rows[2]["M_bright"]
    assert rows[2]["zysk"] > 20.0
    assert abs(rows[2]["zysk"] - 31.58) < 0.5


def test_j0_M_jeden():
    """j=0 (N parzyste): sektor 1-wymiarowy, Γ=0 ⇒ M(t) = 1 dokładnie."""
    assert A.test_dark_sektor()["j0_M"] == 1.0


def test_superradiancja_jasnego():
    """Drenaż stanu 'jeden-wzbudzony' jasnego sektora: Γ = Nγ (N=100)."""
    assert A.test_dark_sektor()["drenaz_one_up_Ngamma"] == 100.0
