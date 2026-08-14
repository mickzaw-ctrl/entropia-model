#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.1 — SYMULACJA N=2..100: CZY PRZEWIDYWANIA WYNIKAJĄ Z MODELU?
=============================================================================
  Rozwiązanie równania Lindblada w bazie Dickego (sektory j, wymiar ≤ N+1)
  dla N = 2..100. Obliczane: S(t), I(A:B)(t), σ(t)=dS/dt, P_dark(t), τ(t)
  (funkcjonał recenzji dτ/dλ = α[Ṡ + η·I]).

  Testowane przewidywania (werdykt na końcu):
    1) „27×”        — kompresja czasowa S_A(t) = S_B(27t) dla każdego N
    2) „czkanie”    — τ̇ → 0 przy Ṡ → 0 (η=0); z η>0: τ̇ → η·I_eq ≠ 0
    3) pamięć subradiacyjna — P_dark, I(A:B) plateau; P_dark(Haar) → 1

  Uruchomienie: python3 entropia_1_1.py   (figury: figury/figE*.png)
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from entropia import core as M
from entropia import dicke as D

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"

NS = [2, 3, 5, 10, 20, 50, 100]


def symuluj_symetryczny(N, gamma, n=400):
    return D.symuluj_dicke(N, gamma, n=n)


def Haar_pdark(N):
    """P_dark stanu Haar = 1 − (N+1)/2^N (waga sektora symetrycznego)."""
    return 1.0 - (N + 1) / 2.0 ** N


def Haar_Sinf_g0(N):
    """S∞ (γ_φ=0) stanu Haar: Σ_j w_j·ln(2j+1) + H(w), w_j = dim_j/2^N."""
    sek = D.sektory_dickego(N)
    dims = [m * (2 * j + 1) for j, m in sek]
    tot = sum(dims)
    w = np.array(dims) / tot
    S = np.sum(w * np.array([np.log(2 * j + 1) for j, _ in sek]))
    S -= np.sum(w * np.log(w))
    return S


def main():
    print("=" * 76)
    print("ENTROPIA-1.1 — N = 2..100 (baza Dickego, sektory j)")
    print("=" * 76)

    # ---------- (1) dynamika i skalowanie ----------
    Sinf = {}
    dyn = {}
    for N in NS:
        res = symuluj_symetryczny(N, M.GAMMA_B, n=500 if N >= 50 else 400)
        Sinf[N] = res["S"][-1]
        dyn[N] = res
    print("\n[1] S(∞) sektora symetrycznego (start |1…1⟩):")
    for N in NS:
        print(f"    N={N:3d}:  S∞ = {Sinf[N]:.4f}   ln(N+1) = {np.log(N+1):.4f}   "
              f"błąd = {abs(Sinf[N]-np.log(N+1)):.2e}")

    # ---------- (2) 27× ----------
    print("\n[2] KOMPRESJA 27×  (S_A(n) = S_B(27n)):")
    err27 = {}
    for N in [2, 5, 10, 20, 50, 100]:
        e = D.kompresja_27(N, M.GAMMA_B, n_cmp=25)
        err27[N] = e
        print(f"    N={N:3d}:  max|S_A − S_B(27n)| (n=0..25) = {e:.2e}")

    # ---------- (3) czkanie ----------
    print("\n[3] CZKANIE (kwantowany zegar entropii, η=0):")
    for N in [2, 4, 6, 10]:
        res = symuluj_symetryczny(N, M.GAMMA_B, n=800)
        dS = res["dS"]
        fz = D.czkanie_stat(dS)
        # budżet = ΣΔS; ile kwantów w ogonie
        k = np.random.default_rng(0).poisson(np.maximum(dS, 0) / M.DELTA_S_Q)
        print(f"    N={N}:  Δτ=0 w ogonie: {100*fz:.0f}%   "
              f"S∞ = {res['S'][-1]:.4f}  (ln{N+1} = {np.log(N+1):.4f})")

    # ---------- (4) pamięć subradiacyjna ----------
    print("\n[4] PAMIĘĆ SUBRADIACYJNA:")
    # (a) P_dark(Haar) — analitycznie
    print("    P_dark(Haar) = 1 − (N+1)/2^N:")
    for N in [4, 10, 20, 50, 100]:
        print(f"      N={N:3d}:  {Haar_pdark(N):.6f}")
    # (b) |10⟩-type (N=2, wagi ½|T0⟩+½|S⟩): P_dark i I(A:B) plateau
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    rho_j1 = np.zeros((3, 3), complex); rho_j1[1, 1] = 1.0
    rho_j0 = np.ones((1, 1), complex)
    stan10 = {1.0: (0.5, rho_j1), 0.0: (0.5, rho_j0)}
    r10 = D.symuluj_dicke(2, M.GAMMA_B, stan=stan10, n=400)
    print(f"    N=2 |10⟩: P_dark = {r10['P_dark'][-1]:.3f}, "
          f"I(A:B)(∞) = {r10['I_AB'][-1]:.4f} (ln(2/√3) = {np.log(2/np.sqrt(3)):.4f})")
    # (c) S∞(Haar, γ_φ=0) vs N·ln2 (pełna termalizacja)
    print("    S∞(Haar, γ_φ=0)/N vs ln2 (pełna termalizacja):")
    for N in [4, 10, 20, 50, 100]:
        print(f"      N={N:3d}:  S∞/N = {Haar_Sinf_g0(N)/N:.4f}  (ln2 = {M.LN2:.4f})")

    # ---------- (5) funkcjonał czasu: czy trzeba modyfikować? ----------
    print("\n[5] FUNKCJONAŁ CZASU — CZY „CZKANIE” I „27×” WYNIKAJĄ Z MODELU?")
    # τ̇ przy równowadze: η=0 (stall) vs η>0 (τ̇ → η·I_eq)
    print(f"    N=2 |10⟩: τ̇_ent(og.200..400) = {np.mean(r10['dtau_ent'][200:]):.4f} "
          f"(stall, η=0)")
    print(f"    N=2 |10⟩: τ̇_rel(og.200..400) = {np.mean(r10['dtau_rel'][200:]):.4f} "
          f"(η=0.5 ⇒ τ̇ → 0.5·I_eq = {0.5*np.log(2/np.sqrt(3)):.4f})")
    print("    Wniosek: „czkanie” (stall) wynika z modelu przy η=0; przy η>0")
    print("    pamięć (I_eq) napędza czas dalej — modyfikacja funkcjonału jest")
    print("    ROZRÓŻNIALNA i testowalna.")

    # ---------- figury ----------
    figura_E1(dyn)
    figura_E2()
    figura_E3(err27)
    figura_E4(r10)
    print(f"\nFigury zapisano w: {os.path.abspath(OUT)}")
    return dict(Sinf=Sinf, err27=err27, r10=r10,
                pdark_haar={N: Haar_pdark(N) for N in NS},
                shaar_g0={N: Haar_Sinf_g0(N) for N in NS})


def figura_E1(dyn):
    """Dynamika S(t) dla N=2,5,20,100 + asymptoty ln(N+1)."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for N, c in [(2, "#2471a3"), (5, "#e67e22"), (20, "#8e44ad"), (100, "#c0392b")]:
        res = dyn[N]
        t = np.arange(len(res["S"])) * M.DELTA_TAU
        ax.plot(t, res["S"], color=c, lw=2, label=f"N = {N}")
        ax.axhline(np.log(N + 1), color=c, ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("ENTROPIA-1.1 — sektor symetryczny: S → ln(N+1)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE1_dynamika.png", bbox_inches="tight")
    plt.close(fig)


def figura_E2():
    """Skalowanie: S∞ vs N (ln(N+1) vs N·ln2); P_dark(Haar) → 1."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    Ns = np.arange(2, 101)
    ax = axs[0]
    ax.semilogy(Ns, np.log(Ns + 1), color="#c0392b", lw=2, label="ln(N+1) (sektor sym.)")
    ax.semilogy(Ns, Ns * M.LN2, color="#8b98a5", lw=2, ls="--", label="N·ln 2 (pełna term.)")
    ax.semilogy(Ns, [Haar_Sinf_g0(N) for N in Ns], color="#2471a3", lw=2,
                label="S∞(Haar, γ_φ=0)")
    ax.set_xlabel("N"); ax.set_ylabel("S(∞) [nat]")
    ax.set_title("Budżet entropii: od ln(N+1) (jasny) do ~N·ln 2 (z dekoherencją)")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.semilogy(Ns, [Haar_pdark(N) for N in Ns], color="#8e44ad", lw=2)
    ax.axhline(0.5, color=C_G, ls=":", lw=1)
    ax.set_xlabel("N"); ax.set_ylabel("P_dark (stan Haar)")
    ax.set_title("Typowe stany są prawie w całości CIEMNE (subradiacyjne)")
    fig.suptitle("ENTROPIA-1.1 — skalowanie N = 2..100", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE2_skalowanie.png", bbox_inches="tight")
    plt.close(fig)


def figura_E3(err27):
    """27×: błąd kompresji vs N; czkanie: Δτ ogon dla N=2,10."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axs[0]
    Ns = list(err27.keys())
    ax.semilogy(Ns, [err27[N] for N in Ns], "o-", color="#27ae60", lw=2, ms=8)
    ax.set_xlabel("N"); ax.set_ylabel("max |S_A(n) − S_B(27n)|")
    ax.set_title("Predykcja 27×: kompresja czasowa dokładna dla N = 2..100")
    ax = axs[1]
    for N, c in [(2, "#2471a3"), (10, "#c0392b")]:
        res = D.symuluj_dicke(N, M.GAMMA_B, n=800)
        dtau = res["dtau_ent"]
        t = np.arange(len(dtau)) * M.DELTA_TAU
        ax.plot(t, dtau, color=c, lw=1.2, alpha=0.85, label=f"N = {N}")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("Δτ_n (kwantowany zegar)")
    ax.set_title("Czkanie: przy nasyceniu Δτ → 0 (zegar staje)")
    ax.legend(fontsize=9)
    fig.suptitle("ENTROPIA-1.1 — testy przewidywań", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE3_testy.png", bbox_inches="tight")
    plt.close(fig)


def figura_E4(r10):
    """Pamięć: P_dark, I(A:B) plateau, τ̇(η=0 vs η>0) — czy funkcjonał trzeba zmienić."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    n = np.arange(len(r10["S"]))
    ax = axs[0]
    ax.plot(n, r10["S"], color="#c0392b", lw=2, label="S(t)")
    ax.plot(n, r10["I_AB"], color="#8e44ad", lw=2, label="I(A:B) — pamięć")
    ax.plot(n, r10["P_dark"], color="#2471a3", lw=2, ls="--", label="P_dark")
    ax.axhline(np.log(2 / np.sqrt(3)), color="#8e44ad", ls=":", lw=1)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("wartość [nat]")
    ax.set_title("N=2 |10⟩: entropia nasyca się, pamięć I = ln(2/√3) trwa")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(n, r10["dtau_ent"], color="#1a5276", lw=1.6, label="τ̇, η = 0 (stall)")
    ax.plot(n, r10["dtau_rel"], color="#e67e22", lw=1.6, label="τ̇, η = 0.5")
    ax.axhline(0.5 * np.log(2 / np.sqrt(3)), color="#e67e22", ls=":", lw=1)
    ax.text(5, 0.5 * np.log(2 / np.sqrt(3)) + 0.01, "0.5·I_eq", color="#e67e22",
            fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("τ̇")
    ax.set_title("Funkcjonał czasu: η=0 ⇒ stall; η>0 ⇒ pamięć napędza czas")
    ax.legend(fontsize=8)
    fig.suptitle("ENTROPIA-1.1 — pamięć subradiacyjna i funkcjonał czasu", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE4_pamiec.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
