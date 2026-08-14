#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  KOSMOLOGICZNY MODEL „ENTROPIA"
=============================================================================
  Czas JEST entropią.  Tyknięcia są dyskretne.  Czas potrafi „czkać".

  Mikrofizyka:
    • każda „komórka" wszechświata to kubit (dwupoziomowy układ kwantowy);
    • kubit oddziałuje z lokalną kąpielą termiczną (rozkład Lindblada);
    • kąpiel nieskończenie gorąca  ⇒  stan docelowy ρ_eq = ½·𝟙
      ⇒  S(ρ) rośnie monotonicznie: 0 → ln 2   (mapa unitalna — tw. Ando–Lindblada;
         dodatniość produkcji entropii σ = −d/dt S(ρ‖ρ_eq) ≥ 0 — tw. Spohna);
    • temperatura ustawia TEMPO, nie cel: entropia właściwa promieniowania
      s ∝ T³, więc dla T_A = 3·T_B tempo produkcji entropii w A jest 27× większe.

  Makrofizyka (zegar kosmiczny):
    • mikro-tyknięcie τ („planckowski" krok) przesuwa stan: ρ_n = e^{ℒτ} ρ_{n−1};
    • entropia produkowana w tyknięciu: ΔS_n = S(ρ_n) − S(ρ_{n−1});
    • czas kosmologiczny: Δt_n = κ·ΔS_n,  T(n) = Σ_k Δt_k  — czas JEST entropią;
    • entropia jest kwantowana w „bitach" δs: w tyknięciu pada k_n ~ Poisson(ΔS_n/δs)
      kwantów ⇒ Δt_n = k_n·δs, a przy niskiej produkcji entropii Δt_n = 0 —
      czas „czka" (długie zamrożenia, potem skok).

  Wynik (7 cech modelu):
    1. ΔS_n na tyknięcie: A produkuje entropię ~27× szybciej niż B
    2. Skumulowany czas:   T(n) = S(n)  (czas = entropia)
    3. Dyskretne tyknięcia: schodki czasu, nie ciągły przepływ
    4. „Ckanie czasu":      Δt_n → 0 przy niskiej produkcji entropii
    5. T_A/T_B > 1:         czas w gorącym otoczeniu płynie szybciej
    6. Tr(ρ²): 1 → 0.5     (czysty → maksymalnie mieszany)
    7. |r|: 1 → 0          (zanik koherencji kwantowej)

  Uruchomienie:  python3 model_entropia.py
  Wymagania:     numpy, scipy, matplotlib
=============================================================================
"""

import os
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (rejestracja projekcji 3d)

# -----------------------------------------------------------------------------
#  KONFIGURACJA
# -----------------------------------------------------------------------------
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "legend.frameon": False,
})

C_A = "#c0392b"    # gorące otoczenie A
C_B = "#2471a3"    # zimne otoczenie B
C_G = "#7f8c8d"    # szary (asymptoty, oznaczenia)
C_START = "#27ae60"
C_END = "#34495e"

# --- parametry fizyczne ------------------------------------------------------
GAMMA_B   = 0.02      # tempo relaksacji amplitudowej (zimne B), [1/j. czasu]
T_RATIO   = 3.0       # T_A = 3·T_B
GAMMA_A   = GAMMA_B * T_RATIO**3      # 27× — bo s ∝ T³ (promieniowanie)
GAMMA_PHI = 2.0       # dekoherencja czysta: γ_φ = 2·γ
OMEGA     = 0.4       # precesja: H = (Ω/2)·σ_z
DELTA_TAU = 0.25      # mikro-tyknięcie τ („planckowski" krok czasu)
N_TICKS   = 400       # liczba tyknięć
LN2       = np.log(2.0)
DELTA_S_Q = 0.01      # kwant entropii („bit") w natach
KAPPA     = 1.0       # stała „czas = entropia": Δt = κ·ΔS
SEED      = 20260813

THETA0, PHI0 = np.pi / 3.0, np.pi / 4.0   # stan początkowy |ψ⟩ (czysty)

# -----------------------------------------------------------------------------
#  ALGEBRA LINDBLADA
# -----------------------------------------------------------------------------
def operatory():
    """Zwraca macierze Pauliego / kreacji-anihilacji dla kubitu."""
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)   # σ_z
    sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)    # σ_+ = |1⟩⟨0|
    sm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)    # σ_- = |0⟩⟨1|
    return sz, sp, sm


def superoperator(gamma, gamma_phi, omega):
    """
    Generator Lindblada ℒ w bazie Liouville'a: vec(ρ) = [ρ00, ρ01, ρ10, ρ11].

        dρ/dt = −i[H,ρ] + γ·D[σ₋] + γ·D[σ₊] + γ_φ·D[σ_z]
        D[L]ρ = LρL† − ½{L†L, ρ}

    Kąpiel nieskończenie gorąca: równe tempo emisji i absorpcji (γ dla σ₊ i σ₋),
    stan stacjonarny ρ_eq = ½·𝟙 (mapa unitalna ⇒ ℒ(𝟙) = 0).
    """
    L = np.zeros((4, 4), dtype=complex)

    # −i[H,·],  H = (Ω/2)·σ_z  →  precesja koherencji
    L[1, 1] = -1j * omega
    L[2, 2] = +1j * omega

    # D[σ₋] + D[σ₊] (tempo γ): populacje dążą do ½:½, koherencje gasną z tempem γ
    L[0, 0] += -gamma; L[0, 3] +=  gamma      # ρ00' = γ(ρ11 − ρ00)
    L[3, 0] +=  gamma; L[3, 3] += -gamma      # ρ11' = γ(ρ00 − ρ11)
    L[1, 1] += -gamma; L[2, 2] += -gamma      # ρ01' = −γ·ρ01

    # D[σ_z] (dekoherencja czysta, tempo γ_φ): tylko koherencje, ×(−2)
    L[1, 1] += -2.0 * gamma_phi
    L[2, 2] += -2.0 * gamma_phi
    return L


def _vec(rho):
    return np.array([rho[0, 0], rho[0, 1], rho[1, 0], rho[1, 1]], dtype=complex)


def _unvec(v):
    return np.array([[v[0], v[1]], [v[2], v[3]]], dtype=complex)


def entropia(rho):
    """Entropia von Neumanna S(ρ) = −Tr(ρ ln ρ)."""
    ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    ev = np.clip(ev, 1e-300, None)
    return float(-np.sum(ev * np.log(ev)))


def czystosc(rho):
    """Tr(ρ²) — czystość stanu."""
    return float(np.real(np.trace(rho @ rho)))


def bloch(rho):
    """Wektor Blocha r = (Tr(ρσ_x), Tr(ρσ_y), Tr(ρσ_z))."""
    return np.array([2.0 * np.real(rho[0, 1]),
                     2.0 * np.imag(rho[0, 1]),
                     np.real(rho[0, 0] - rho[1, 1])])


def symuluj(gamma, omega=OMEGA, delta_tau=DELTA_TAU, n=N_TICKS):
    """
    Deterministyczna ewolucja Lindblada po n mikro-tyknięciach.

    Zwraca: S[n]  — entropia von Neumanna,
            P[n]  — czystość Tr(ρ²),
            R[n]  — wektor Blocha,
            rho_n — stany.
    """
    L = superoperator(gamma, GAMMA_PHI * gamma, omega)
    U = expm(L * delta_tau)                     # mapa CPTP na jedno tyknięcie

    psi = np.array([np.cos(THETA0 / 2.0),
                    np.exp(1j * PHI0) * np.sin(THETA0 / 2.0)])
    rho = np.outer(psi, psi.conj())

    S = np.zeros(n); P = np.zeros(n); R = np.zeros((n, 3))
    for i in range(n):
        S[i] = entropia(rho); P[i] = czystosc(rho); R[i] = bloch(rho)
        rho = _unvec(U @ _vec(rho))
    return S, P, R


def delta_entropii(S):
    """Produkcja entropii na tyknięcie: ΔS_n = S_n − S_{n−1}."""
    dS = np.zeros_like(S)
    dS[1:] = np.maximum(S[1:] - S[:-1], 0.0)     # unitalność ⇒ ΔS ≥ 0
    return dS


# -----------------------------------------------------------------------------
#  ROZWIĄZANIE ANALITYCZNE (postać zamknięta dynamiki Blocha)
#  |r(t)|² = cos²θ₀·e^(−4γt) + sin²θ₀·e^(−10γt)   (γ_φ = 2γ, precesja obraca
#  składową poprzeczną, nie zmienia jej modułu).  Służy do dokładnych czasów
#  przekroczenia poziomów entropii (brentq) i kontroli krzyżowej symulacji.
# -----------------------------------------------------------------------------
def _eps_analityczne(gamma, t):
    c2, s2 = np.cos(THETA0) ** 2, np.sin(THETA0) ** 2
    r2 = c2 * np.exp(-4.0 * gamma * t) + s2 * np.exp(-10.0 * gamma * t)
    return np.sqrt(np.clip(r2, 0.0, 1.0))


def S_analityczne(gamma, t):
    """Entropia von Neumanna w czasie t (postać zamknięta)."""
    eps = _eps_analityczne(gamma, t)
    lp, lm = (1.0 + eps) / 2.0, (1.0 - eps) / 2.0
    lp = max(lp, 1e-300); lm = max(lm, 1e-300)
    return -(lp * np.log(lp) + lm * np.log(lm))


def dSdt_analityczne(gamma, t):
    """Tempo produkcji entropii dS/dt (postać zamknięta)."""
    eps = _eps_analityczne(gamma, t)
    if eps <= 0 or eps >= 1:
        return 0.0
    artanh = 0.5 * np.log((1.0 + eps) / (1.0 - eps))
    c2, s2 = np.cos(THETA0) ** 2, np.sin(THETA0) ** 2
    d2 = gamma * (4.0 * c2 * np.exp(-4.0 * gamma * t)
                  + 10.0 * s2 * np.exp(-10.0 * gamma * t))
    return artanh * d2 / (2.0 * eps)


def czas_do_poziomu(gamma, poziom):
    """Najwcześniejszy czas t, dla którego S(t) = poziom (brentq)."""
    from scipy.optimize import brentq
    if poziom <= 0:
        return 0.0
    if poziom >= LN2:
        return np.inf
    # górna granica: czas, po którym |r|² < 1e-8
    t_max = 3.0 / gamma
    f = lambda t: S_analityczne(gamma, t) - poziom
    return float(brentq(f, 0.0, t_max, xtol=1e-12))


def zegar_stochastyczny(dS, ds=DELTA_S_Q, seed=None):
    """
    Kwantowy zegar kosmiczny (realizacja):
      k_n ~ Poisson(ΔS_n/δs)  — liczba „bitów" entropii w tyknięciu n;
      Δt_n = k_n·δs,  T(n) = Σ Δt_k  (czas = entropia, κ = 1).
    Gdy ΔS_n jest małe ⇒ k_n = 0 ⇒ Δt_n = 0: czas „czka".
    """
    rng = np.random.default_rng(seed)
    k = rng.poisson(np.maximum(dS, 0.0) / ds)
    dt = k * ds
    T = np.cumsum(dt)
    return T, dt, k


# -----------------------------------------------------------------------------
#  ANALIZA — liczby kluczowe (drukowane, wykorzystywane w raporcie)
# -----------------------------------------------------------------------------
def liczby_kluczowe():
    S_A, P_A, R_A = symuluj(GAMMA_A)
    S_B, P_B, R_B = symuluj(GAMMA_B)
    dS_A, dS_B = delta_entropii(S_A), delta_entropii(S_B)

    # 1) S_A(n) ≡ S_B(27·n) — kompresja czasowa dokładna (identyczna dynamika,
    #    tylko tempo γ różne; S zależy od γ·t). Weryfikacja numeryczna:
    idx = np.clip((np.arange(N_TICKS) * 27).astype(int), 0, N_TICKS - 1)
    blad_kompresji = float(np.max(np.abs(S_A - S_B[idx])))
    # kontrola krzyżowa symulacji vs postać zamknięta (dla B):
    t_grid = np.arange(N_TICKS) * DELTA_TAU
    S_an = np.array([S_analityczne(GAMMA_B, t) for t in t_grid])
    blad_analityka = float(np.max(np.abs(S_B - S_an)))

    # 2) czas do połowy entropii: S = ln2/2  (dokładnie, przez brentq)
    tA_half = czas_do_poziomu(GAMMA_A, LN2 / 2)
    tB_half = czas_do_poziomu(GAMMA_B, LN2 / 2)

    # 3) tempo produkcji entropii dS/dt przy DOPASOWANYCH poziomach entropii:
    #    stosunek tempa = γ_A/γ_B = 27 (dokładnie — ta sama funkcja f(γ·t))
    poziomy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    tA_lvl = [czas_do_poziomu(GAMMA_A, l) for l in poziomy]
    tB_lvl = [czas_do_poziomu(GAMMA_B, l) for l in poziomy]
    dSdt_A_lvl = [dSdt_analityczne(GAMMA_A, t) for t in tA_lvl]
    dSdt_B_lvl = [dSdt_analityczne(GAMMA_B, t) for t in tB_lvl]
    stosunki = [a / b for a, b in zip(dSdt_A_lvl, dSdt_B_lvl)]

    # 4) pierwsze tyknięcie
    dS1_A, dS1_B = dS_A[1], dS_B[1]

    # 5) nasycenie
    S_end_A, S_end_B = S_A[-1], S_B[-1]
    P_end_A, P_end_B = P_A[-1], P_B[-1]
    r_end_A = float(np.linalg.norm(R_A[-1])); r_end_B = float(np.linalg.norm(R_B[-1]))

    print("=" * 64)
    print("LICZBY KLUCZOWE MODELU «ENTROPIA»")
    print("=" * 64)
    print(f"  γ_B = {GAMMA_B:.4f},  γ_A = 27·γ_B = {GAMMA_A:.4f}  (T_A = 3·T_B, s ∝ T³)")
    print(f"  S(∞) = ln 2 = {LN2:.6f}")
    print(f"  S_A(ostatnie) = {S_end_A:.6f},  S_B(ostatnie) = {S_end_B:.6f}")
    print(f"  Tr(ρ²): A: {P_A[0]:.4f} → {P_end_A:.4f}   B: {P_B[0]:.4f} → {P_end_B:.4f}")
    print(f"  |r|:    A: {np.linalg.norm(R_A[0]):.4f} → {r_end_A:.4f}   B: {np.linalg.norm(R_B[0]):.4f} → {r_end_B:.4f}")
    print(f"  S_A(n) ≡ S_B(27n): maks. błąd = {blad_kompresji:.2e}")
    print(f"  symulacja vs postać zamknięta: maks. błąd = {blad_analityka:.2e}")
    print(f"  czas do ½·ln 2:  t_A = {tA_half:.4f},  t_B = {tB_half:.4f},  t_B/t_A = {tB_half/tA_half:.2f}")
    print(f"  ΔS w 1. tyknięciu: A = {dS1_A:.5f},  B = {dS1_B:.5f}  (stos. {dS1_A/dS1_B:.2f})")
    print("  tempo dS/dt przy dopasowanych poziomach entropii (stosunek A/B):")
    for lvl, r in zip(poziomy, stosunki):
        print(f"    S* = {lvl:.1f}:  {r:.3f}")
    print("=" * 64)

    return dict(S_A=S_A, S_B=S_B, P_A=P_A, P_B=P_B, R_A=R_A, R_B=R_B,
                dS_A=dS_A, dS_B=dS_B, tA_half=tA_half, tB_half=tB_half,
                stosunki=stosunki, poziomy=poziomy)


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_entropia(d):
    """Cecha główna: S: 0 → ln 2 (tw. Ando–Lindblada)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    n = np.arange(N_TICKS)
    for ax in (ax1, ax2):
        ax.plot(n, d["S_A"], color=C_A, lw=2, label="A — gorące (T_A = 3·T_B)")
        ax.plot(n, d["S_B"], color=C_B, lw=2, label="B — zimne")
        ax.axhline(LN2, color=C_G, ls="--", lw=1.2)
        ax.text(0.02, 0.96, "ln 2", transform=ax.transAxes, color=C_G, fontsize=10)
    ax1.plot(n, d["S_A"], color=C_A, lw=2)
    ax1.plot(n, d["S_B"], color=C_B, lw=2)
    # czas do połowy entropii
    ax1.axhline(LN2 / 2, color=C_G, ls=":", lw=1)
    ax1.vlines([d["tA_half"]], 0, LN2 / 2, color=C_A, ls=":", lw=1)
    ax1.vlines([d["tB_half"]], 0, LN2 / 2, color=C_B, ls=":", lw=1)
    ax1.annotate(f"t_B/t_A = {d['tB_half'] / d['tA_half']:.1f}",
                 xy=(d["tB_half"], LN2 / 2), xytext=(150, 0.58),
                 fontsize=11, color="#8e44ad",
                 arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2))
    ax1.set_xlabel("tyknięcie n")
    ax1.set_ylabel("S(ρ) [nat]")
    ax1.set_title("S(ρ): 0 → ln 2 — monotonicznie (mapa unitalna, tw. Ando–Lindblada)")
    ax1.legend(loc="center right")
    # S_A(t) ≡ S_B(27t) — wstawka
    axin = ax1.inset_axes([0.045, 0.42, 0.30, 0.30])
    idx = np.clip((np.arange(N_TICKS) * 27).astype(int), 0, N_TICKS - 1)
    axin.plot(d["S_A"], d["S_B"][idx], ".", ms=2, color="#8e44ad", alpha=0.7)
    axin.plot([0, LN2], [0, LN2], color=C_G, ls="--", lw=1)
    axin.set_title("S_A(n) = S_B(27·n)", fontsize=8)
    axin.set_xlabel("S_A(n)", fontsize=8); axin.set_ylabel("S_B(27n)", fontsize=8)
    axin.tick_params(labelsize=7)
    # zoom na A
    ax2.set_xlim(0, 40)
    ax2.set_xlabel("tyknięcie n")
    ax2.set_ylabel("S(ρ) [nat]")
    ax2.set_title("Zoom: gorące A nasyca się ~27× szybciej")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_entropia.png", bbox_inches="tight")
    plt.close(fig)


def figura_produkcja(d):
    """Cecha 1: ΔS_n na tyknięcie — A ~27× szybciej niż B."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    n = np.arange(N_TICKS)

    # (a) ΔS_n per tick
    ax1.semilogy(n, d["dS_A"], color=C_A, lw=2, label="A — gorące")
    ax1.semilogy(n, d["dS_B"], color=C_B, lw=2, label="B — zimne")
    ax1.set_xlabel("tyknięcie n")
    ax1.set_ylabel("ΔS_n  [nat/tyknięcie]")
    ax1.set_title("Produkcja entropii na tyknięcie")
    ax1.legend(loc="upper right")
    ax1.annotate("ta sama krzywa,\n27× ściśnięta w czasie:\nΔS_A(n) = ΔS_B(27n)",
                 xy=(14, d["dS_A"][14]), xytext=(60, 6e-2),
                 fontsize=10, color=C_A,
                 arrowprops=dict(arrowstyle="->", color=C_A, lw=1))
    # (b) tempo produkcji przy dopasowanych poziomach entropii — stosunek = 27
    poziomy, stosunki = d["poziomy"], d["stosunki"]
    tA_lvl = [czas_do_poziomu(GAMMA_A, l) for l in poziomy]
    tB_lvl = [czas_do_poziomu(GAMMA_B, l) for l in poziomy]
    rA = [dSdt_analityczne(GAMMA_A, t) for t in tA_lvl]
    rB = [dSdt_analityczne(GAMMA_B, t) for t in tB_lvl]
    x = np.arange(len(poziomy))
    w = 0.34
    ax2.bar(x - w / 2, rA, w, color=C_A, label="tempo w A")
    ax2.bar(x + w / 2, rB, w, color=C_B, label="tempo w B")
    for xi, a, b in zip(x, rA, rB):
        ax2.text(xi, max(a, b) * 1.03, f"{a / b:.0f}×", ha="center", fontsize=9,
                 color="#8e44ad", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{l:.1f}" for l in poziomy])
    ax2.set_xlabel("poziom entropii S*  [nat]")
    ax2.set_ylabel("dS/dt  [nat/j. czasu]")
    ax2.set_title("Tempo produkcji przy tym samym S*: stosunek = 27")
    ax2.legend()
    ax2.set_yscale("log")
    fig.suptitle("Cecha 1 — A produkuje entropię 27× szybciej niż B", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_produkcja.png", bbox_inches="tight")
    plt.close(fig)


def figura_czas_entropia(d):
    """Cechy 2 i 3: T(n) = S(n); dyskretne schodki czasu."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    n = np.arange(N_TICKS)

    T_A, _, _ = zegar_stochastyczny(d["dS_A"], seed=1)
    T_B, _, _ = zegar_stochastyczny(d["dS_B"], seed=1)

    ax1.plot(n, T_B, color=C_B, lw=1.4, drawstyle="steps-post", label="T(n) — realizacja B")
    ax1.plot(n, d["S_B"], color=C_B, lw=1, ls="--", alpha=0.6,
             label="S(n) — wartość oczekiwana")
    ax1.plot(n, T_A, color=C_A, lw=1.4, drawstyle="steps-post", label="T(n) — realizacja A")
    ax1.plot(n, d["S_A"], color=C_A, lw=1, ls="--", alpha=0.6)
    ax1.axhline(LN2, color=C_G, ls=":", lw=1)
    ax1.text(5, LN2 + 0.015, "ln 2 — koniec czasu (nasycenie)", color=C_G, fontsize=9)
    ax1.set_xlabel("tyknięcie n")
    ax1.set_ylabel("T(n) = S̃(n)  [jedn. czasu = nat]")
    ax1.set_title("Czas JEST entropią: T(n) = Σ ΔS_k — wyraźne schodki tyknięć")
    ax1.legend(loc="center right", fontsize=8)

    # (b) T vs S — zależność liniowa (nachylenie 1)
    ax2.plot(d["S_B"], T_B, ".", ms=2, color=C_B, alpha=0.6, label="B (realizacja)")
    ax2.plot(d["S_A"], T_A, ".", ms=2, color=C_A, alpha=0.6, label="A (realizacja)")
    ax2.plot([0, LN2], [0, LN2], color="#8e44ad", lw=2, label="T = S")
    ax2.set_xlabel("S(ρ_n)  [nat]")
    ax2.set_ylabel("T(n)  [j. czasu]")
    ax2.set_title("Skumulowany czas ∝ entropia (nachylenie = κ = 1)")
    ax2.legend(loc="upper left")
    fig.suptitle("Cechy 2–3 — czas = entropia; dyskretne tyknięcia, nie ciągły przepływ", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_czas_entropia.png", bbox_inches="tight")
    plt.close(fig)


def figura_czkanie(d):
    """Cecha 4: przy niskiej produkcji entropii Δt_n → 0 — czas „czka"."""
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)

    n = np.arange(N_TICKS)
    T_A, dt_A, _ = zegar_stochastyczny(d["dS_A"], seed=11)
    T_B, dt_B, _ = zegar_stochastyczny(d["dS_B"], seed=11)

    # (a) B — ogon: Δt_n
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = 40, 200
    ax.stem(n[lo:hi], dt_B[lo:hi], linefmt=C_B, markerfmt="none", basefmt=" ")
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("tyknięcie n")
    ax.set_ylabel("Δt_n")
    ax.set_title("B (zimne): Δt_n → 0 przy małej produkcji entropii")
    n_zero = int(np.sum(dt_B[lo:hi] == 0))
    ax.text(0.97, 0.93, f"Δt=0 w {n_zero}/{hi - lo} tyknięciach",
            transform=ax.transAxes, ha="right", fontsize=9, color=C_B)

    # (b) B — skumulowany czas w tym oknie: długie zamrożenia
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(n[lo:hi], T_B[lo:hi], drawstyle="steps-post", color=C_B, lw=2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("tyknięcie n")
    ax.set_ylabel("T(n)")
    ax.set_title("B: czas zamrożony… potem skok („czknięcie”)")
    # najdłuższe zamrożenie
    maks = dl = 0
    for i in range(lo, hi):
        if dt_B[i] == 0:
            dl += 1
            maks = max(maks, dl)
        else:
            dl = 0
    ax.text(0.97, 0.9, f"najdłuższe zamrożenie: {maks} tyknięć",
            transform=ax.transAxes, ha="right", fontsize=9, color=C_B)

    # (c) A — ogon po szybkim nasyceniu też „czka”
    ax = fig.add_subplot(gs[1, 0])
    lo2, hi2 = 3, 30
    ax.stem(n[lo2:hi2], dt_A[lo2:hi2], linefmt=C_A, markerfmt="none", basefmt=" ")
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("tyknięcie n")
    ax.set_ylabel("Δt_n")
    ax.set_title("A (gorące): nasyca się w ~10 tyknięciach, potem też „czka”")
    ax.text(0.97, 0.93, f"Δt=0 w {int(np.sum(dt_A[lo2:hi2] == 0))}/{hi2 - lo2} tyknięciach",
            transform=ax.transAxes, ha="right", fontsize=9, color=C_A)

    # (d) A — skumulowany czas
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(n[lo2:hi2], T_A[lo2:hi2], drawstyle="steps-post", color=C_A, lw=2)
    ax.set_xlim(lo2, hi2)
    ax.set_xlabel("tyknięcie n")
    ax.set_ylabel("T(n)")
    ax.set_title("A: schodki, a przy ogonie — czkanie")
    fig.suptitle("Cecha 4 — „czkanie czasu”: przy niskiej produkcji entropii Δt_n → 0",
                 y=1.0, fontsize=13)
    fig.savefig(f"{OUT}/fig4_czkanie.png", bbox_inches="tight")
    plt.close(fig)


def figura_stosunek(d):
    """Cecha 5: T_A/T_B — czas w gorącym otoczeniu płynie szybciej."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    n = np.arange(N_TICKS)

    # (a) skumulowany czas (średnia po realizacjach ± 1σ)
    M = 400
    TA_all = np.zeros((M, N_TICKS)); TB_all = np.zeros((M, N_TICKS))
    for m in range(M):
        TA_all[m], _, _ = zegar_stochastyczny(d["dS_A"], seed=1000 + m)
        TB_all[m], _, _ = zegar_stochastyczny(d["dS_B"], seed=1000 + m)
    mA, sA = TA_all.mean(0), TA_all.std(0)
    mB, sB = TB_all.mean(0), TB_all.std(0)
    ax1.plot(n, mA, color=C_A, lw=2, label="⟨T_A(n)⟩ — gorące")
    ax1.fill_between(n, mA - sA, mA + sA, color=C_A, alpha=0.15)
    ax1.plot(n, mB, color=C_B, lw=2, label="⟨T_B(n)⟩ — zimne")
    ax1.fill_between(n, mB - sB, mB + sB, color=C_B, alpha=0.15)
    ax1.axhline(LN2, color=C_G, ls=":", lw=1)
    ax1.text(5, LN2 + 0.015, "ln 2", color=C_G)
    ax1.set_xlabel("tyknięcie n")
    ax1.set_ylabel("T(n)")
    ax1.set_title("Zegar A zawsze wyprzedza zegar B")
    ax1.legend(loc="center right", fontsize=8)
    ax1.annotate("A osiąga ten sam poziom entropii\n27× szybciej niż B",
                 xy=(4, LN2 * 0.97), xytext=(120, 0.55), fontsize=10,
                 color="#8e44ad", arrowprops=dict(arrowstyle="->", color="#8e44ad"))

    # (b) stosunek T_A/T_B
    det_ratio = d["S_A"] / np.maximum(d["S_B"], 1e-300)
    det_ratio[0] = np.nan                      # 0/0 na starcie
    ax2.plot(n, det_ratio, color="#8e44ad", lw=2,
             label="⟨T_A⟩/⟨T_B⟩ = S_A/S_B")
    r1 = np.where(TB_all[0] > 0, TA_all[0] / np.maximum(TB_all[0], 1e-300), np.nan)
    r1[0] = np.nan
    ax2.plot(n, r1, color=C_G, lw=0.7, alpha=0.6, label="pojedyncza realizacja")
    ax2.axhline(1, color=C_G, ls="--", lw=1)
    ax2.text(5, 1.05, "1 — nasycenie (oba → ln 2)", color=C_G, fontsize=9)
    ax2.annotate("granica ciągła t→0: 27\n(1. tyknięcie: ≈ %.1f)" % (d["dS_A"][1] / d["dS_B"][1]),
                 xy=(0.4, det_ratio[1] * 0.55), xytext=(140, 24),
                 fontsize=10, color="#8e44ad",
                 arrowprops=dict(arrowstyle="->", color="#8e44ad"))
    ax2.set_xlabel("tyknięcie n")
    ax2.set_ylabel("T_A / T_B")
    ax2.set_title("Stosunek czasu: gorące płynie szybciej (T_A > T_B)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 30)
    fig.suptitle("Cecha 5 — czas w gorącym otoczeniu płynie szybciej", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_stosunek.png", bbox_inches="tight")
    plt.close(fig)


def figura_dekoherencja(d):
    """Cechy 6 i 7: Tr(ρ²): 1 → 0.5; |r|: 1 → 0."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    n = np.arange(N_TICKS)
    rA = np.linalg.norm(d["R_A"], axis=1)
    rB = np.linalg.norm(d["R_B"], axis=1)

    ax = axes[0, 0]
    ax.plot(n, d["P_A"], color=C_A, lw=2, label="A — gorące")
    ax.plot(n, d["P_B"], color=C_B, lw=2, label="B — zimne")
    ax.axhline(0.5, color=C_G, ls="--", lw=1)
    ax.text(5, 0.505, "0.5 (stan maksymalnie mieszany ½·𝟙)", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("Tr(ρ²)")
    ax.set_title("Czystość: 1 → 0.5")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(n, rA, color=C_A, lw=2, label="|r_A|")
    ax.plot(n, rB, color=C_B, lw=2, label="|r_B|")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("|r|")
    ax.set_title("Długość wektora Blocha: 1 → 0")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(d["S_B"], d["P_B"], color=C_B, lw=2, label="B")
    ax.plot(d["S_A"], d["P_A"], color=C_A, lw=2, ls="--", label="A")
    ax.set_xlabel("S(ρ)  [nat]"); ax.set_ylabel("Tr(ρ²)")
    ax.set_title("Fundamentalna zależność Tr(ρ²) = ½(1 + |r|²) vs S")
    ax.legend()

    ax = axes[1, 1]
    for i, (naz, col) in enumerate(zip(["r_x", "r_y", "r_z"], ["#e67e22", "#2ecc71", "#34495e"])):
        ax.plot(n, d["R_B"][:, i], color=col, lw=1.5, label=f"{naz} (B)")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("składowa wektora Blocha")
    ax.set_title("Zanikające oscylacje koherencji (precesja + dekoherencja)")
    ax.legend(ncol=3, fontsize=8)
    fig.suptitle("Cechy 6–7 — dekoherencja: Tr(ρ²): 1 → 0.5,  |r|: 1 → 0", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_dekoherencja.png", bbox_inches="tight")
    plt.close(fig)


def figura_bloch(d):
    """Cecha 7 (wizualizacja): trajektoria na sferze Blocha."""
    fig = plt.figure(figsize=(11.5, 5.4))
    for idx, (gamma, naz, col) in enumerate([(GAMMA_A, "A — gorące", C_A),
                                             (GAMMA_B, "B — zimne", C_B)]):
        S, P, R = symuluj(gamma)
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        # sfera
        u = np.linspace(0, 2 * np.pi, 48)
        v = np.linspace(0, np.pi, 24)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="#eaf2f8", alpha=0.55,
                        edgecolor="#b9c9da", linewidth=0.3)
        # osie
        for (x0, y0, z0) in [(1.15, 0, 0), (-1.15, 0, 0), (0, 1.15, 0),
                             (0, -1.15, 0), (0, 0, 1.15), (0, 0, -1.15)]:
            ax.plot([0, x0], [0, y0], [0, z0], color="#9aa5b1", lw=1)
        # trajektoria (gradient koloru = bieg czasu)
        N = len(R)
        for i in range(N - 1):
            ax.plot(R[i:i + 2, 0], R[i:i + 2, 1], R[i:i + 2, 2],
                    color=col, alpha=0.12 + 0.88 * i / N, lw=1.7)
        ax.scatter(*R[0], color=C_START, s=70, depthshade=False, label="start (czysty)")
        ax.scatter(*R[-1], color=C_END, s=55, depthshade=False, label="koniec (|r| = 0)")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel("r_x"); ax.set_ylabel("r_y"); ax.set_zlabel("r_z")
        ax.set_title(f"{naz}: zanik koherencji |r|: 1 → 0", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle("Sfera Blocha — trajektoria stanu: spirala dekoherencji do środka", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_bloch.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    d = liczby_kluczowe()
    figura_entropia(d)
    figura_produkcja(d)
    figura_czas_entropia(d)
    figura_czkanie(d)
    figura_stosunek(d)
    figura_dekoherencja(d)
    figura_bloch(d)
    print(f"Figury zapisano w: {os.path.abspath(OUT)}")


def main_script():
    main()
