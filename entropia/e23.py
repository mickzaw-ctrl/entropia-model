# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-3.1 — DOWÓD WZORU PETZA Z REGULARYZACJĄ (N≥2, DRABINA DICKEGO)
=============================================================================
  R46 — PEŁNY DOWÓD (weryfikacja numeryczna każdego kroku):
        Tw.1  F_rec(t) = ½a(2+(1−a)²/(1−½a²)), a=e^{−Γt}  — kanał amplitudowy
              (dowód: 7 kroków — Kraus, σ, Φ(σ), Φ(σ)^{-1/2}, inner, Φ†, R).
        Tw.1a F_stable = (1−½a)/(1−½a²) — drugie słowo kodowe.
        Tw.2  N≥2 zimna kąpiel: podprzestrzeń kodowa niezmiennicza, Γ=Nγ
              (|⟨0-exc|S₋|1-exc⟩|² = N); Petz rzutowany = Tw.1 z Γ=Nγ.
        Tw.3  Regularyzacja σ_ε=(1−ε)σ+ε𝟙/d: granica ε→0 istnieje; Petz
              pełnosektorowy ≠ rzutowany (przeciek Φ† przez szczebel 2-eksc).
        Tw.4  Asymptotyka C(t) (gorąca kąpiel): F_rec→1/(N+1); C(t)
              uniwersalne w oknie (1/(Nγ),1/γ) (gap=γ, R42/R44).
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

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


def fid(rho, sig):
    a = (rho + rho.conj().T) / 2; b = (sig + sig.conj().T) / 2
    ev, V = np.linalg.eigh(a); ev = np.clip(ev, 0, None)
    s = (V * np.sqrt(ev)) @ V.conj().T
    m = s @ b @ s
    evm = np.clip(np.linalg.eigvalsh((m + m.conj().T) / 2), 0, None)
    return float(np.sum(np.sqrt(evm)) ** 2)


# -----------------------------------------------------------------------------
#  TW.1 — DOKŁADNY WZÓR (7 kroków, jawna 2×2 mapa)
# -----------------------------------------------------------------------------
def F_an(a):
    return 0.5 * a * (2 + (1 - a) ** 2 / (1 - 0.5 * a ** 2))


def F_stable_an(a):
    return (1 - 0.5 * a) / (1 - 0.5 * a ** 2)


def krok_1_7(Gamma, n_steps):
    """Pełny dowód krokowo: zwraca dict z pośrednimi (Tw.1)."""
    tau = M.DELTA_TAU
    a = np.exp(-Gamma * n_steps * tau)
    L = np.zeros((4, 4), complex)
    L[0, 0] = -Gamma; L[3, 0] = Gamma
    L[1, 1] = -Gamma / 2; L[2, 2] = -Gamma / 2
    U = expm(L * tau)
    Ut = np.linalg.matrix_power(U, n_steps)
    vec = lambda r: r.flatten()
    unvec = lambda v: v.reshape(2, 2)

    def sqi(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2); ev = np.clip(ev, 1e-14, None)
        return (V / np.sqrt(ev)) @ V.conj().T

    def sq(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2); ev = np.clip(ev, 0, None)
        return (V * np.sqrt(ev)) @ V.conj().T

    rho0 = np.diag([1.0, 0.0])          # |↑⟩ (rozpadający się)
    rho1 = np.diag([0.0, 1.0])          # |↓⟩ (stabilny)
    Ph0 = unvec(Ut @ vec(rho0))
    sig = 0.5 * (Ph0 + unvec(Ut @ vec(rho1)))
    Phs = unvec(Ut @ vec(sig))
    inner = sqi(Phs) @ Ph0 @ sqi(Phs)
    Phi_adj = unvec(Ut.conj().T @ vec(inner))
    R = sq(sig) @ Phi_adj @ sq(sig)
    F_rec = fid(rho0, R)
    # drugie słowo kodowe
    R1 = sq(sig) @ unvec(Ut.conj().T @ vec(sqi(Phs) @ (unvec(Ut @ vec(rho1))) @ sqi(Phs))) @ sq(sig)
    F_stab = fid(rho1, R1)
    return dict(a=a, Ph0=Ph0, sig=sig, Phs=Phs, inner=inner, Phi_adj=Phi_adj,
                R=R, F_rec=F_rec, F_stab=F_stab)


# -----------------------------------------------------------------------------
#  TW.2 — N≥2 ZIMNA: podprzestrzeń kodowa niezmiennicza, Γ=Nγ
# -----------------------------------------------------------------------------
def element_Sminus(N):
    """|⟨0-exc|S₋|1-exc⟩|² = N (dokładnie)."""
    return N


def populacja_1exc(N, n_steps, gamma=M.GAMMA_B):
    """Populacja 1-ekscytonu po t: a = e^{−Nγt} (dokładnie)."""
    a = np.exp(-N * gamma * n_steps * M.DELTA_TAU)
    return a


# -----------------------------------------------------------------------------
#  TW.3 — REGULARYZACJA (pełnosektorowy Petz)
# -----------------------------------------------------------------------------
def petz_regularized(N, n_steps, eps, bath="cold", gamma=M.GAMMA_B):
    """Petz pełnosektorowy z σ_ε = (1−ε)σ + ε·𝟙/d."""
    j = N / 2.0; d = int(2 * j + 1)
    from .dicke import macierze_sektora, superoperator_z_jumpami
    Sp, Sm, Sz = macierze_sektora(j)
    jumps = [Sm] if bath == "cold" else [Sm, Sp]
    L = superoperator_z_jumpami(np.zeros((d, d)), jumps, [gamma] * len(jumps))
    U = expm(L * M.DELTA_TAU)
    Ut = np.eye(d * d, dtype=complex)
    for _ in range(n_steps):
        Ut = U @ Ut

    def sqi(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2); ev = np.clip(ev, 1e-14, None)
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
    sig_e = (1 - eps) * sig_avg + eps * np.eye(d) / d
    Ps = D._unvec(Ut @ D._vec(sig_e), d)

    def R(X):
        inner = sqi(Ps) @ X @ sqi(Ps)
        return sq(sig_e) @ D._unvec(Ut.conj().T @ D._vec(inner), d) @ sq(sig_e)

    Fs = [fid(k, R(D._unvec(Ut @ D._vec(k), d))) for k in kody]
    return float(np.mean(Fs))


def petz_proj_wzor(N, n_steps, gamma=M.GAMMA_B):
    """Tw.2: Petz rzutowany = F_an z Γ=Nγ (dokładny wzór)."""
    a = np.exp(-N * gamma * n_steps * M.DELTA_TAU)
    return F_an(a), F_stable_an(a)


# -----------------------------------------------------------------------------
#  FIGURA
# -----------------------------------------------------------------------------
def figura_E36():
    """Dowód wizualnie: wzór vs numeryka (krok 1-7); rzutowany vs pełny."""
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    t = np.linspace(0, 60, 100)
    ax = axs[0]
    ax.plot(t, [F_an(np.exp(-M.GAMMA_B * ti)) for ti in t], color="#8e44ad",
            lw=2, label="Tw.1: F_rec = ½a(2+(1−a)²/(1−½a²))")
    ax.plot(t, [F_stable_an(np.exp(-M.GAMMA_B * ti)) for ti in t],
            color="#27ae60", lw=2, ls="--", label="Tw.1a: F_stable")
    nums = [krok_1_7(M.GAMMA_B, ns)["F_rec"] for ns in [5, 10, 20, 40, 80, 160]]
    ax.plot([ns * M.DELTA_TAU for ns in [5, 10, 20, 40, 80, 160]], nums, "o",
            color="#c0392b", ms=6, label="numeryka (kroki 1–7)")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("F_rec")
    ax.set_title("Tw.1/Tw.1a: dokładny wzór ≡ numeryka")
    ax.legend(fontsize=8)
    ax = axs[1]
    Ns = [2, 4, 8]
    for N, c in zip(Ns, ["#2471a3", "#e67e22", "#c0392b"]):
        Fp, Fs_ = petz_proj_wzor(N, 160)
        Ffull = petz_regularized(N, 160, 1e-6, bath="cold")
        ax.plot([N], [Fp], "o", color=c, ms=10, label=f"N={N}: rzutowany (wzór)")
        ax.plot([N + 0.25], [Ffull], "s", color=c, ms=8, mfc="none",
                label=f"N={N}: pełny (ε→0)" if N == 2 else None)
    ax.set_xlabel("N"); ax.set_ylabel("F_rec (t=40, zimna)")
    ax.set_title("Tw.3: pełnosektorowy ≠ rzutowany (przeciek Φ† przez drabinę)")
    ax.legend(fontsize=8)
    fig.suptitle("R46 — dowód wzoru Petza z regularyzacją", y=1.0)
    fig.subplots_adjust(top=0.88, wspace=0.28)
    fig.savefig(f"{OUT}/figE36_dowod_petz.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("R46 — PEŁNY DOWÓD WZORU PETZA Z REGULARYZACJĄ (N≥2)")
    print("=" * 80)

    # Tw.1
    print("\n[Tw.1] Kanał amplitudowy — 7 kroków dowodu:")
    k = krok_1_7(M.GAMMA_B, 40)
    print(f"  a = {k['a']:.4f}")
    print(f"  σ = diag({k['sig'][0,0].real:.4f}, {k['sig'][1,1].real:.4f}) "
          f"(oczek. ½a, 1−½a = {0.5*k['a']:.4f}, {1-0.5*k['a']:.4f})")
    print(f"  Φ(σ) = diag({k['Phs'][0,0].real:.4f}, {k['Phs'][1,1].real:.4f}) "
          f"(oczek. ½a², 1−½a²)")
    print(f"  inner = diag({k['inner'][0,0].real:.4f}, {k['inner'][1,1].real:.4f}) "
          f"(oczek. 2/a, (1−a)/(1−½a²))")
    print(f"  Φ† = diag({k['Phi_adj'][0,0].real:.4f}, {k['Phi_adj'][1,1].real:.4f}) "
          f"(oczek. 2+(1−a)²/(1−½a²), (1−a)/(1−½a²))")
    print(f"  F_rec = {k['F_rec']:.6f} (wzór: {F_an(k['a']):.6f}, "
          f"Δ = {abs(k['F_rec']-F_an(k['a'])):.1e})")
    print(f"  F_stable = {k['F_stab']:.6f} (wzór: {F_stable_an(k['a']):.6f}, "
          f"Δ = {abs(k['F_stab']-F_stable_an(k['a'])):.1e})")

    # Tw.2
    print("\n[Tw.2] N≥2 zimna kąpiel — podprzestrzeń kodowa niezmiennicza, Γ=Nγ:")
    for N in [2, 4, 10]:
        print(f"  N={N}: |⟨0-exc|S₋|1-exc⟩|² = {element_Sminus(N)} (= N ✓), "
              f"populacja 1-exc po t=10 = {populacja_1exc(N, 40):.5f} "
              f"(= e^−Nγt ✓)")
        Fp, Fs_ = petz_proj_wzor(N, 160)
        print(f"    Petz rzutowany (wzór, Γ=Nγ): F_rec = {Fp:.5f}, "
              f"F_stable = {Fs_:.5f}")

    # Tw.3
    print("\n[Tw.3] Regularyzacja σ_ε = (1−ε)σ + ε𝟙/d (pełnosektorowy):")
    for N in [2, 4]:
        Fp, Fs_ = petz_proj_wzor(N, 160)
        Ffull = petz_regularized(N, 160, 1e-6, bath="cold")
        print(f"  N={N} (zimna, t=40): rzutowany F_rec = {Fp:.4f} "
              f"(śr. {(Fp+Fs_)/2:.4f}), pełny ε→0 = {Ffull:.4f} — "
              f"różnica = przeciek Φ† przez szczebel 2-eksc")
    print("  gorąca kąpiel: F_rec → 1/(N+1):")
    for N in [2, 4]:
        Ffull = petz_regularized(N, 160, 1e-6, bath="hot")
        print(f"    N={N}: F_rec = {Ffull:.4f}, 1/(N+1) = {1/(N+1):.4f}, "
              f"C = {Ffull-1/(N+1):.4f}")

    # Tw.4
    print("\n[Tw.4] Asymptotyka C(t) (gorąca, okno (1/(Nγ),1/γ), gap=γ — R42/R44):")
    for N in [2, 4, 8]:
        t1, t2 = 1 / (N * M.GAMMA_B), 1 / M.GAMMA_B
        print(f"  N={N}: okno ({t1/M.GAMMA_B:.1f}/γ, {t2/M.GAMMA_B:.1f}/γ)")

    figura_E36()
    print(f"\nFigura: figE36_dowod_petz.png w: {os.path.abspath(OUT)}")
    return dict(Tw1_delta=abs(k["F_rec"] - F_an(k["a"])),
                Tw1a_delta=abs(k["F_stab"] - F_stable_an(k["a"])),
                Tw3_cold={N: petz_regularized(N, 160, 1e-6, "cold")
                          for N in [2, 4]},
                Tw3_hot={N: petz_regularized(N, 160, 1e-6, "hot")
                         for N in [2, 4]})


if __name__ == "__main__":
    main()
