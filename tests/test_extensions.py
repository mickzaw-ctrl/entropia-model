# -*- coding: utf-8 -*-
"""Testy rozszerzeń R1–R13 (fizyka: temperatury, sektory, feedback, NESS)."""
import numpy as np
import pytest

from entropia import core as M
from entropia import extensions as R


# ---------- R1: skończona temperatura ----------
def test_r1_nasycenie_gibbsa():
    # wartości analityczne: H(1/(1+η))
    assert abs(R.S_eq_termiczna(0.5) - 0.636514168) < 1e-9
    assert abs(R.S_eq_termiczna(0.1) - 0.304636097) < 1e-9
    assert abs(R.S_eq_termiczna(1.0) - M.LN2) < 1e-12


def test_r1_kontrola_krzyzowa():
    """Dyskretny Lindblad vs postać zamknięta (η=0.5)."""
    S, _, _ = R.symuluj_termicznie(M.GAMMA_B, 0.5, n=300)
    t = np.arange(300) * M.DELTA_TAU
    Sa = R.S_termiczna_analitycznie(M.GAMMA_B, 0.5, t)
    assert np.max(np.abs(S - Sa)) < 1e-8


def test_r1_kompresja_27():
    """Kompresja 27× przy skończonej temperaturze."""
    t = np.linspace(0, 40, 2000)
    SA = R.S_termiczna_analitycznie(M.GAMMA_A, 0.5, t)
    SB = R.S_termiczna_analitycznie(M.GAMMA_B, 0.5, 27 * t)
    assert np.max(np.abs(SA - SB)) < 1e-10


# ---------- R2: dwa kubity ----------
def test_r2_ln3():
    ket1 = np.array([0.0, 1.0])
    stan = R.stan_poczatkowy_N([ket1, ket1])
    S, *_ = R.symuluj_wspolne(M.GAMMA_B, stan, N=2, gamma_phi=0.0, n=2000)
    assert abs(S[-1] - np.log(3)) < 5e-3


def test_r2_ciemny_singlet():
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    stan = R.stan_poczatkowy_N([ket1, ket0])
    S, *_ = R.symuluj_wspolne(M.GAMMA_B, stan, N=2, gamma_phi=0.0, n=2000)
    assert abs(S[-1] - np.log(12) / 2) < 5e-3


# ---------- R4: N=3 sektory ----------
def test_r4_ln4_i_ln2():
    b, sector = R.baza_N3()
    S111, *_ = R.symuluj_N3(M.GAMMA_B, b["111"], n=1500)
    S1S, *_ = R.symuluj_N3(M.GAMMA_B, sector[6][2], n=1500)
    assert abs(S111[-1] - np.log(4)) < 5e-3
    assert abs(S1S[-1] - M.LN2) < 5e-3


def test_r4_subradiancja():
    """Populacja na kopii B ≡ 1 (stan |1⟩⊗|S⟩ ciemny dla kanału kolektywnego)."""
    _, sector = R.baza_N3()
    stan = sector[6][2]
    S_z, S_p, S_m, _ = R.macierze_kolektywne(3)
    H = (M.OMEGA / 2.0) * S_z
    from scipy.linalg import expm
    U = expm(R.superoperator_z_jumpami(H, [S_p, S_m], [M.GAMMA_B, M.GAMMA_B]) * M.DELTA_TAU)
    PB = np.outer(sector[6][2], sector[6][2].conj()) + np.outer(sector[7][2], sector[7][2].conj())
    rho = np.outer(stan, stan.conj())
    popmin = 1.0
    for _ in range(100):
        popmin = min(popmin, np.real(np.trace(PB @ rho)))
        rho = R.unvecR(U @ R.vecR(rho), 8)
    assert popmin > 0.99999


# ---------- R5: entropia makro ----------
def test_r5_lnNplus1():
    w2 = R.entropia_makro(2, n=1500)
    w3 = R.entropia_makro(3, n=1500)
    assert abs(w2[1] - np.log(3)) < 5e-3
    assert abs(w3[1] - np.log(4)) < 5e-3
    # ekstensywność niezależnych
    assert abs(w2[0] - 2 * M.LN2) < 1e-6


# ---------- R7: losowe stany, koherencje ----------
def test_r7_blokada_i_odblokowanie():
    b, _ = R.baza_N3()
    ket = b["100"]
    rho = np.outer(ket, ket.conj())
    S0, *_ = R.symuluj_wspolne(M.GAMMA_B, rho, N=3, gamma_phi=0.0, n=1500)
    Sd, *_ = R.symuluj_wspolne(M.GAMMA_B, rho, N=3, gamma_phi=M.GAMMA_B, n=1500)
    assert abs(S0[-1] - np.log(108) / 3) < 5e-3          # blokada
    assert abs(Sd[-1] - 3 * M.LN2) < 5e-3                # odblokowanie


# ---------- R8: cykl ----------
def test_r8_tau_2budget():
    # n_tot = n_cyc+1: ostatni punkt ma η = 1 (pełny powrót do ln 2)
    S, dS, eta, Ts, Ta = R.symuluj_cykl(0.05, 0.15, 300, 301)
    budzet = S[0] - S.min()
    assert abs(Ta[-1] - 2 * budzet) < 0.02          # τ = 2·budżet
    assert abs(Ts[-1] - (S[-1] - S[0])) < 1e-9      # definicja T_signed
    assert abs(S[-1] - S[0]) < 0.02                 # pętla: powrót do ln 2


# ---------- R11: prawo odblokowania ----------
def test_r11_prawo_1_gphi():
    """τ90(γφ) ∝ 1/γφ: dla γφ ×3 → τ90 ÷3 (reżim asymptotyczny)."""
    # głęboki reżim asymptotyczny (z dala od podłogi kolektywnej)
    g1, g2 = 1e-4, 3e-4
    rng = np.random.default_rng(3)
    los = R.stan_haara(rng, 8)
    t1 = R.czas_odblokowania(R.przebieg_gphi(3, los, g1, n=80000), 3 * M.LN2)
    t2 = R.czas_odblokowania(R.przebieg_gphi(3, los, g2, n=30000), 3 * M.LN2)
    ratio = t1 / t2
    assert 2.0 < ratio < 4.5


# ---------- R13: grawitacyjna produkcja entropii ----------
def test_r13_ness_sigma_dodatnie():
    S, sr, sg = R.symuluj_dwie_kapiele(0.05, 0.9, 0.01, 0.1, n=300)
    sig = sr + sg
    assert S[-1] < M.LN2 - 1e-3                       # NESS poniżej ln 2
    assert sig[-1] > 1e-4                             # produkcja entropii trwa
    # stałość w NESS (ostatnie 30 tyknięć)
    assert np.ptp(sig[-30:]) < 1e-5


def test_r13_rownowaga_zero():
    """η_r = η_g ⇒ σ = 0 (równowaga, koniec produkcji)."""
    _, sr, sg = R.symuluj_dwie_kapiele(0.03, 0.5, 0.03, 0.5, n=400)
    assert abs((sr + sg)[-1]) < 1e-8


# ---------- R15: dekoherencja zegara jako strażnik historii ----------
def test_r15_kappa_chroni_historie():
    """κ niszczy rozmycie czasu (koherencje wskazań) w punkcie zwrotnym."""
    GT, G, TICKS = 0.02, 0.2, 200
    z0 = R.kwantowy_zegar_hint(GT, G, 0.0, TICKS=TICKS)
    z5 = R.kwantowy_zegar_hint(GT, G, 0.5, TICKS=TICKS)
    n_t0 = R.punkt_zwrotny(z0["Ss"])
    off0 = z0["offdiag"][n_t0]
    off5 = z5["offdiag"][n_t0]
    assert off5 < off0 / 5.0          # κ suprymuje rozmycie
    assert z5["coh"][n_t0] < z0["coh"][n_t0] / 2.0
    # zapis historii monotoniczny w obu
    assert np.all(np.diff(z0["nb"]) >= -1e-6)
    assert np.all(np.diff(z5["nb"]) >= -1e-6)
    # nieodwracalność: S(zegar) rośnie z κ
    assert z5["Scl"][n_t0] > z0["Scl"][n_t0]


# ---------- R16: formalizm relacyjny (rewizja po recenzji) ----------
def test_r16_zegar_relacyjny_korelacje():
    """η=0 odtwarza zegar entropii; η>0 (korelacje) dodaje czasu."""
    dS = np.zeros(100); dS[10] = 0.05; dS[20] = 0.05
    I = np.linspace(0, 0.1, 100)
    tau0, _ = R.zegar_relacyjny(dS, eta=0.0, seed=1)
    tau1, _ = R.zegar_relacyjny(dS, I=I, eta=0.5, seed=1)
    assert tau1[-1] > tau0[-1]               # korelacje dodają budżetu
    # η=0: zegar = entropia (kwantowana)
    assert abs(tau0[-1] - 0.10 / 0.01) < 1e-6  # ΔS=0.10, ref=0.01 → 10


def test_r16_27_jako_predykcja_warunkowa():
    """Gałąź s daje 27 dokładnie; gałąź Ṡ ≠ 27; γ∝T^p → 3^p."""
    pr = R.stosunek_27_jako_predykcja()
    assert abs(pr["s_branch"] - 27.0) < 1e-6
    assert pr["sdot_tick1"] < 27.0            # gałąź Ṡ nie daje 27
    for p, v in pr["p_scan"].items():
        assert abs(v - 3.0 ** p) < 1e-9


def test_r16_stall_nie_koniec_czasu():
    """Przy równowadze τ̇ → 0, choć ewolucja może trwać."""
    z = R.kwantowy_zegar(0.01, TICKS=200)
    dS = np.concatenate([[0], np.maximum(np.diff(z["S_sys"]), 0)])
    tau, dtau = R.zegar_relacyjny(dS, eta=0.0, seed=1)
    assert dtau[-20:].max() == 0.0            # zegar staje w nasyceniu


# ---------- R17: test laboratoryjny bright↔dark ----------
def test_r17_ciemny_sektor_zamraza_zegar():
    """Singlet: τ̇ = 0 (zegar milczy); frakcja ciemna obniża tempo liniowo."""
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    SING = (np.kron(ket1, ket0) - np.kron(ket0, ket1)) / np.sqrt(2)
    S_s, *_ = R.symuluj_wspolne(M.GAMMA_B, np.outer(SING, SING.conj()), N=2,
                                gamma_phi=0.0, n=200)
    dS_s = np.maximum(np.diff(S_s), 0)
    tau_s = R.zegar_entropowy(dS_s)
    assert tau_s[-1] == 0.0                   # ciemny: zero tyknięć
    # frakcja ciemna → tempo maleje
    T0 = (np.kron(ket1, ket0) + np.kron(ket0, ket1)) / np.sqrt(2)
    rates = []
    for p in [0.0, 0.5, 1.0]:
        rho = (1 - p) * np.outer(T0, T0.conj()) + p * np.outer(SING, SING.conj())
        Sp, *_ = R.symuluj_wspolne(M.GAMMA_B, rho, N=2, gamma_phi=0.0, n=150)
        rates.append(float(np.maximum(np.diff(Sp), 0)[5:40].mean()))
    assert rates[2] < 1e-9                    # p=1 (czysty ciemny): zero
    assert rates[1] < rates[0]                # połowa ciemna: wolniej
