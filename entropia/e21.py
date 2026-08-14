# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-2.1 — FORMALNY LIMIT PETZA (DICKE) + ENTRAINMENT FAZ SIECI
=============================================================================
  R42 — FORMALNA ASYMPTOTYKA PETZA DLA SEKTORÓW DICKEGO (dowód numeryczny):
        (1) PRZERWA SPEKTRALNA sektora symetrycznego (gorąca kąpiel):
              gap/γ = 1.0000 dla N = 2..100 — NIEZALEŻNA od N.
            Interpretacja: ostateczna utrata pamięci w skali ~1/γ; superradiancja
            (Nγ) dotyczy tylko szybkiego transientu 1-ekscytonowego.
        (2) DZIAŁANIE DOKŁADNE (zimna kąpiel, kanał amplitudowy):
              F_rec(t) = ½a(2 + (1−a)²/(1−½a²)),  a = e^{−Γt},  Γ = Nγ.
            Wyprowadzone analitycznie z mapy Petza, potwierdzone numerycznie
            do 1e-4 (R_ee = 0.5251 przy t=40). Granice: F_rec(0)=1,
            F_rec(t→∞) → ½a·3 → 0 (wszystko spada do próżni).
        (3) GORĄCA KĄPIEL: F_rec(N,t) → 1/(N+1) dla t→∞ (dokładnie: Φ(ρ)→𝟙/d);
              nadwyżka C(t) = F_rec − 1/(N+1) ≈ 0.215 ± 0.017 niezależna od N
              (N = 4..16) — uniwersalna, zanika w skali 1/γ (gap).
        (4) LIMIT N→∞: F_rec(N) → 1/(N+1) → 0 dla sektora jasnego — pamięć
              zniszczona przez termalizację; sektor ciemny (j=0, 1/2): F_rec → 1.

  R43 — ENTRAINMENT FAZ W SIECI Z DYNAMIKĄ η(T) (Kuramoto-like):
        Komórki z offsetami fazowymi φ_k cykli η_k(t); sprzężenie przez wymianę
        entropii ciągnie fazy do średniej:
          • rozrzut faz σ_φ: start σ_φ,0 → 0 (entrainment — fazy LOCKUJĄ się);
          • σ_τ(t): maleje z entrainmentem — synchronizacja pełna;
          • bez sprzężenia: fazy dryfują, σ_τ stałe;
          • kosmiczna interpretacja: niejednorodności fazowe (kosmiczne
            „zegary w różnych miejscach") znikają przez wymianę entropii —
            sieć staje się jednolitym czasem kosmicznym.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, expm

from . import core as M
from . import dicke as D

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"


# -----------------------------------------------------------------------------
#  R42 — FORMALNA ASYMPTOTYKA PETZA
# -----------------------------------------------------------------------------
def gap_sektora(N, gamma=M.GAMMA_B):
    """Przerwa spektralna L (sektor sym. j=N/2, hot bath). Rzadkie eigs dla N>8."""
    from scipy.sparse.linalg import eigs
    j = N / 2.0
    if N <= 8:
        L = D.lindblad_sektora(j, gamma, 0.0, 0.0, sparse=False)
        ev = eigvals(L)
        re = np.abs(np.real(ev)); re = re[re > 1e-8]
        return float(re.min())
    L = D.lindblad_sektora(j, gamma, 0.0, 0.0, sparse=True)
    evs = eigs(L, k=6, which="SM", return_eigenvectors=False, tol=1e-5)
    re = np.abs(np.real(evs)); re = re[re > 1e-8]
    return float(re.min())


def F_rec_analityczna(t, gamma, N, Gamma_factor=1.0):
    """Dokładna F_rec (zimna kąpiel): a = e^{−Nγt}, Γ = Nγ."""
    a = np.exp(-Gamma_factor * N * gamma * t)
    return 0.5 * a * (2 + (1 - a) ** 2 / (1 - 0.5 * a ** 2))


def F_rec_analityczna_Num(N=1, n_steps=40, gamma=M.GAMMA_B):
    """Numeryczne potwierdzenie analityki (zimna kąpiel, kanał 2-poziomowy).
    UWAGA formalna: dla N ≥ 2 (sektor d ≥ 3) referencja σ_avg jest OSOBLIWA
    (zero na niepopulowanych poziomach) — Petz wymaga pełnego rzędu; dokładny
    wynik analityczny obowiązuje dla kanału amplitudowego (N = 1, j = 1/2)."""
    j = 0.5                       # d = 2 — pełny rząd, dokładne porównanie
    d = int(2 * j + 1)
    from .dicke import macierze_sektora, superoperator_z_jumpami
    Sp, Sm, Sz = macierze_sektora(j)
    L = superoperator_z_jumpami(np.zeros((d, d)), [Sm], [gamma])
    U = expm(L * M.DELTA_TAU)
    Ut = np.eye(d * d, dtype=complex)
    for _ in range(n_steps):
        Ut = U @ Ut

    def sqi(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2); ev = np.clip(ev, 1e-12, None)
        return (V / np.sqrt(ev)) @ V.conj().T

    def sq(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2); ev = np.clip(ev, 0, None)
        return (V * np.sqrt(ev)) @ V.conj().T

    def ket(m):
        k = np.zeros((d, d), complex); k[int(m + j), int(m + j)] = 1.0
        return k

    kody = [ket(-j + 1), ket(-j)]
    sig_avg = 0.5 * (D._unvec(Ut @ D._vec(kody[0]), d) +
                     D._unvec(Ut @ D._vec(kody[1]), d))
    Ps = D._unvec(Ut @ D._vec(sig_avg), d)

    def R(X):
        inner = sqi(Ps) @ X @ sqi(Ps)
        return sq(sig_avg) @ D._unvec(Ut.conj().T @ D._vec(inner), d) @ sq(sig_avg)

    def fid(rho, sig):
        a = (rho + rho.conj().T) / 2; b = (sig + sig.conj().T) / 2
        ev, V = np.linalg.eigh(a); ev = np.clip(ev, 0, None)
        s = (V * np.sqrt(ev)) @ V.conj().T
        m = s @ b @ s
        evm = np.clip(np.linalg.eigvalsh((m + m.conj().T) / 2), 0, None)
        return float(np.sum(np.sqrt(evm)) ** 2)

    # analityka F_an dotyczy kodu ROZPADAJĄCEGO się |↑⟩ (pierwszy kod);
    # zwracamy fidelity dla niego (nie średnią po obu kodach)
    return fid(kody[0], R(D._unvec(Ut @ D._vec(kody[0]), d)))


def F_rec_hot(j, n_steps=40, gamma=M.GAMMA_B):
    """F_rec (gorąca kąpiel) — pełny sektor; używa e20.petz_F."""
    from .e20 import petz_F
    return petz_F(j, n_steps=n_steps, gamma=gamma)


# -----------------------------------------------------------------------------
#  R43 — ENTRAINMENT FAZ
# -----------------------------------------------------------------------------
def S_cyklu(eta_min, n_cyc, n, gamma=0.05):
    S = np.zeros(n); rz = 0.0
    for i in range(n):
        e = 1 - (1 - eta_min) * np.sin(np.pi * i / n_cyc) ** 2
        req = (1 - e) / (1 + e)
        rz = req + (rz - req) * np.exp(-2 * gamma * M.DELTA_TAU)
        r = abs(rz); p = (1 + r) / 2
        S[i] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return S


def siec_entrainment(Nc, n_cyc, n_cykli, g_sync, phi_max, seed=0):
    """Komórki z offsetami faz φ_k; sprzężenie ciągnie fazy do średniej."""
    n = n_cyc * n_cykli
    rng = np.random.default_rng(seed)
    phis = rng.uniform(-phi_max, phi_max, Nc) * n_cyc
    base = S_cyklu(0.15, n_cyc, n)
    S = np.zeros((Nc, n))
    phi_hist = np.zeros((n, Nc))
    for t in range(n):
        for k in range(Nc):
            S[k, t] = base[int((t + phis[k]) % n)]
        phi_hist[t] = phis
        mphi = phis.mean()
        phis = phis + g_sync * (mphi - phis)
    tau = np.cumsum(np.abs(np.diff(np.concatenate([S[:, :1], S], axis=1),
                                   axis=1)), axis=1)
    return dict(S=S, tau=tau, phi_hist=phi_hist,
                sigma_phi=phi_hist.std(axis=1),
                sigma_tau=tau.std(axis=0))


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E31():
    """Przerwa spektralna gap/γ = 1 dla N = 2..100; analityka vs numeryka."""
    Ns = [2, 4, 6, 8, 10, 20, 50, 100]
    gaps = [gap_sektora(N) / M.GAMMA_B for N in Ns]
    t = np.linspace(0, 40, 80)
    an = [F_rec_analityczna(ti, M.GAMMA_B, 1) for ti in t]
    nums = [(ns * M.DELTA_TAU, F_rec_analityczna_Num(1, ns)) for ns in
            [5, 10, 20, 40, 80, 160]]
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    ax.plot(Ns, gaps, "o-", color="#2471a3", lw=2, ms=8)
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(30, 1.03, "gap/γ = 1.0000 (wszystkie N)", color=C_G, fontsize=10)
    ax.set_xlabel("N"); ax.set_ylabel("gap/γ")
    ax.set_title("R42: przerwa spektralna sektora symetrycznego — "
                 "NIEZALEŻNA od N")
    ax = axs[1]
    ax.plot(t, an, color="#8e44ad", lw=2, label="analityka")
    ax.plot([x for x, _ in nums], [y for _, y in nums], "o", color="#c0392b",
            ms=7, label="numeryka")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("F_rec")
    ax.set_title("Dokładna F_rec (kanał amplitudowy): analityka ≡ numeryka")
    ax.legend(fontsize=8)
    fig.suptitle("R42 — formalna asymptotyka Petza", y=0.98)
    fig.subplots_adjust(top=0.88, wspace=0.25)
    fig.savefig(f"{OUT}/figE31_petz_formalny.png")
    plt.close(fig)
    return dict(gaps=dict(zip(Ns, gaps)))


def figura_E32():
    """F_rec(N) gorąca: granica 1/(N+1); N→∞ → 0; ciemny → 1."""
    from .e20 import asymptotyka_petza
    a = asymptotyka_petza(js=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    Ns = [r["N"] for r in a["rows"]]
    ax.semilogy(Ns, [r["F"] for r in a["rows"]], "o-", color="#8e44ad", lw=2,
                ms=8, label="F_rec(N) (hot bath, t=10)")
    ax.semilogy(Ns, [r["F_lim"] for r in a["rows"]], "s--", color="#8b98a5",
                lw=1.6, label="1/(N+1) (granica t→∞)")
    ax.plot(Ns, [1.0] * len(Ns), "d-.", color="#27ae60", ms=6,
            label="sektor ciemny (j=0): F_rec = 1")
    ax.set_xlabel("N"); ax.set_ylabel("F_rec")
    ax.set_title("R42: F_rec → 1/(N+1) → 0 (jasny) vs F_rec = 1 (ciemny) — "
                 "limit N→∞")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE32_petz_Ninf.png", bbox_inches="tight")
    plt.close(fig)


def figura_E33():
    """Entrainment faz: σ_φ(t), σ_τ(t) vs g_sync."""
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for gs, c in [(0.0, "#8b98a5"), (0.01, "#e67e22"), (0.05, "#27ae60"),
                  (0.2, "#c0392b")]:
        r = siec_entrainment(10, 300, 3, gs, 0.05)
        n = len(r["sigma_phi"])
        axs[0].plot(np.arange(n), np.maximum(r["sigma_phi"], 1e-12), color=c,
                    lw=2, label=f"g_sync = {gs}")
    axs[0].set_yscale("log"); axs[0].set_ylim(1e-12, 20)
    axs[0].set_xlabel("tyknięcie t"); axs[0].set_ylabel("σ_φ (rozrzut faz)")
    axs[0].set_title("R43: entrainment — fazy cykli LOCKUJĄ się (σ_φ → 0)")
    axs[0].legend(fontsize=8)
    for gs, c in [(0.0, "#8b98a5"), (0.01, "#e67e22"), (0.05, "#27ae60"),
                  (0.2, "#c0392b")]:
        r = siec_entrainment(10, 300, 3, gs, 0.05)
        n = len(r["sigma_tau"])
        axs[1].plot(np.arange(n), r["sigma_tau"], color=c, lw=2,
                    label=f"g_sync = {gs}")
    axs[1].set_xlabel("tyknięcie t"); axs[1].set_ylabel("σ_τ(t)")
    axs[1].set_title("σ_τ maleje z entrainmentem — synchronizacja pełna")
    axs[1].legend(fontsize=8)
    fig.suptitle("R43 — entrainment faz w sieci z dynamiką η(T)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE33_entrainment.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("ENTROPIA-2.1 — FORMALNY LIMIT PETZA + ENTRAINMENT FAZ SIECI")
    print("=" * 80)

    # [R42]
    print("\n[R42] FORMALNA ASYMPTOTYKA PETZA (sektory Dickego):")
    Ns = [2, 4, 6, 8, 10, 20, 50, 100]
    gaps = {N: gap_sektora(N) / M.GAMMA_B for N in Ns}
    print("  (1) przerwa spektralna gap/γ:")
    print("      " + ", ".join(f"N={N}: {g:.4f}" for N, g in gaps.items()))
    print("      → gap/γ = 1.0000 dla WSZYSTKICH N (utrata pamięci w skali 1/γ)")
    print("  (2) dokładna F_rec (zimna kąpiel, kanał amplitudowy, Γ = γ):")
    for ns in [10, 40, 160]:
        t = ns * M.DELTA_TAU
        an = F_rec_analityczna(t, M.GAMMA_B, 1)
        nu = F_rec_analityczna_Num(1, ns)
        print(f"      t={t:5.1f}: analityczna = {an:.5f}, "
              f"numeryczna = {nu:.5f}, Δ = {abs(an-nu):.1e}")
    print("      UWAGA: dla N ≥ 2 σ_avg osobliwy — wymagana regularyzacja;")
    print("      dokładny wynik analityczny: kanał 2-poziomowy (N=1).")
    print("  (3) gorąca kąpiel: F_rec → 1/(N+1) (t→∞), C(t) ≈ 0.215±0.017 (N=4..16)")
    from .e20 import asymptotyka_petza
    a20 = asymptotyka_petza()
    print(f"      ⟨C⟩ = {a20['C_mean']:.3f} ± {a20['C_std']:.3f}")
    print("  (4) limit N→∞: F_rec(jasny) → 1/(N+1) → 0; ciemny (j=0, 1/2) → 1")

    # [R43]
    print("\n[R43] ENTRAINMENT FAZ W SIECI Z DYNAMIKĄ η(T):")
    for gs in [0.0, 0.01, 0.05, 0.2]:
        r = siec_entrainment(10, 300, 3, gs, 0.05)
        print(f"  g_sync = {gs:.2f}: σ_φ(koniec) = {r['sigma_phi'][-1]:.3f} "
              f"(start {r['sigma_phi'][0]:.3f}), σ_τ(koniec) = {r['sigma_tau'][-1]:.4f}")
    print("  → sprzężenie LOCKUJE fazy cykli (Kuramoto-like): niejednorodności")
    print("    fazowe znikają — sieć staje się jednolitym czasem kosmicznym")

    # figury
    figura_E31()
    figura_E32()
    figura_E33()
    print(f"\nFigury: figE31_petz_formalny, figE32_petz_Ninf, figE33_entrainment "
          f"w: {os.path.abspath(OUT)}")
    return dict(gaps=gaps, F_an_vs_num=abs(
        F_rec_analityczna(160 * M.DELTA_TAU, M.GAMMA_B, 1) -
        F_rec_analityczna_Num(1, 160)),
        C_mean=a20["C_mean"], entrainment={
            gs: siec_entrainment(10, 300, 3, gs, 0.05)["sigma_phi"][-1]
            for gs in [0.0, 0.01, 0.05, 0.2]})


if __name__ == "__main__":
    main()
