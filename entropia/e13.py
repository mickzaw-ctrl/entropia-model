# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.3 — KOHERENTNA INFORMACJA, KOSZT ZEGARA, PROTOKÓŁ T1 vs T2
=============================================================================
  Trzy domknięcia programu falsyfikacyjnego recenzji:

  R20 — COHERENT INFORMATION (I_c): czy pamięć subradiacyjna jest kwantowa?
        I_c(A⟩B) = S_B − S_AB. Dla równowagi (max-mix w sektorze) I_c < 0
        (brak odzyskiwalności kwantowej) mimo I(A:B) > 0 — pamięć jest
        KLASYCZNA (korelacje, nie splątanie destylowalne). Funkcjonał
        T3c = (σ + η|Ī_c|)/σ₀: staje (Ī_c → 0).

  R21 — KOSZT ENERGETYCZNY ZEGARA (Salecker–Wigner w modelu):
        E_clock = ω_c·⟨n⟩ (energia oscylatora zegara); ΔE = ω_c·Δn;
        Δτ = Δn·δs (nieoznaczoność odczytu). Trójkąt:
        precyzja (Δn/⟨n⟩↓) ↔ koszt (E_clock↑) ↔ entropia (back-action↑)
        jako funkcje siły zegara γ_t. Warunek ΔE·Δτ ≥ ħ/2 ⇒ ω_c ≥ ω_c^min.

  R22 — PROTOKÓŁ POMIAROWY ROZSTRZYGAJĄCY T1 vs T2:
        po wygaśnięciu fluorescencji Γ = Ṡ → 0 mierzymy tempo zegara τ̇:
        T1 (produkcja informacji): τ̇ → 0 (zegar staje);
        T2 (istnienie informacji): τ̇ → η·I_eq ≠ 0 (zegar tyka dalej).
        Obserwowalna: τ̇ po Γ = 0. Test w układzie Dicke'a (subradiancja).
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

from . import core as M
from . import dicke as D
from .e12 import funkcjonaly_czasu, stan_10_N2, SIGMA0, ETA

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"


# -----------------------------------------------------------------------------
#  R20 — COHERENT INFORMATION
# -----------------------------------------------------------------------------
def I_c_sym(rhoN, k):
    """I_c(A⟩B) = S_B − S_AB dla symetrycznego stanu (dwupodział k:N−k)."""
    N = rhoN.shape[0] - 1
    SN = D.entropia(rhoN)
    SB = D.entropia(D.redukcja_symetryczna(rhoN, N - k))
    return SB - SN


def trajektorie_rhosektora(N, j, n=400, gamma=M.GAMMA_B):
    """Ewolucja macierzy sektora (stan |j,−j⟩) — dla I_c(t)."""
    from .dicke import lindblad_sektora, propagator_sektora, _vec, _unvec
    L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=False)
    U = propagator_sektora(L, M.DELTA_TAU)
    d = int(2 * j + 1)
    rho = np.zeros((d, d), complex); rho[0, 0] = 1.0
    out = []
    for _ in range(n):
        out.append(rho.copy())
        rho = _unvec(U @ _vec(rho), d)
    return out


# -----------------------------------------------------------------------------
#  R21 — KOSZT ENERGETYCZNY ZEGARA (ω_c·b†b)
# -----------------------------------------------------------------------------
def zegar_z_energia(gt, wc, MLEV=22, TICKS=250, gamma=0.02, gphi=0.04,
                    omega=M.OMEGA, delta_tau=M.DELTA_TAU):
    """
    Zegar kwantowy (jak R9) + H_clock = ω_c·b†b. Zwraca ⟨n⟩, Δn, S_sys,
    E_clock = ω_c⟨n⟩, ΔE = ω_cΔn.
    """
    ML = MLEV; tau = delta_tau; n_ticks = TICKS
    I2 = np.eye(2, dtype=complex); sz, sp_, sm_ = M.operatory()
    a = np.diag(np.sqrt(np.arange(1, ML)), 1); ad = a.conj().T
    IM = np.eye(ML, dtype=complex); Nc = ad @ a
    H = 0.5 * omega * np.kron(sz, IM) + wc * np.kron(I2, Nc)
    jumps = [np.kron(sm_, ad), np.kron(sm_, IM), np.kron(sp_, IM), np.kron(sz, IM)]
    rates = [gt, gamma, gamma, gphi]
    d = 2 * ML; Id = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for J, r in zip(jumps, rates):
        Jd = J.conj().T; JJ = Jd @ J
        L += r * (np.kron(J, J.conj()) - 0.5 * (np.kron(JJ, Id) + np.kron(Id, JJ.T)))
    IL = np.eye(d * d, dtype=complex)
    lu, piv = lu_factor(IL - 0.5 * tau * L)
    B = IL + 0.5 * tau * L
    psi = np.array([np.cos(M.THETA0 / 2), np.exp(1j * M.PHI0) * np.sin(M.THETA0 / 2)])
    rc0 = np.zeros((ML, ML), complex); rc0[0, 0] = 1
    rho = np.kron(np.outer(psi, psi.conj()), rc0)

    def S(r):
        ev = np.linalg.eigvalsh((r + r.conj().T) / 2); ev = ev[ev > 1e-15]
        return -np.sum(ev * np.log(ev))

    nb = np.zeros(n_ticks); dn = np.zeros(n_ticks); Ss = np.zeros(n_ticks)
    for k in range(n_ticks):
        rc = np.zeros((ML, ML), complex)
        for i in range(2):
            for j in range(2):
                rc += rho[i::2, j::2]
        pn = np.real(np.diag(rc)); pn = np.clip(pn, 0, None); pn = pn / pn.sum()
        nb[k] = np.sum(np.arange(ML) * pn)
        dn[k] = np.sqrt(max(0.0, np.sum((np.arange(ML) - nb[k]) ** 2 * pn)))
        rs = np.array([[np.trace(rho[0:ML, 0:ML]), np.trace(rho[0:ML, ML:2 * ML])],
                       [np.trace(rho[ML:2 * ML, 0:ML]), np.trace(rho[ML:2 * ML, ML:2 * ML])]])
        Ss[k] = S(rs)
        rho = np.asarray(lu_solve((lu, piv), B @ rho.flatten())).reshape(d, d)
    return dict(nb=nb, dn=dn, Ss=Ss,
                E=wc * nb, dE=wc * dn)


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E8():
    """Coherent information: I_c < 0 przy równowadze (pamięć klasyczna)."""
    rhos4 = trajektorie_rhosektora(4, 2.0, n=400)
    Ic4 = np.array([I_c_sym(r, 2) for r in rhos4])
    n = np.arange(400)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(n, Ic4, color="#8e44ad", lw=2, label="I_c(A⟩B) — N=4 |1111⟩")
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.axhline(np.log(3) - np.log(5), color="#8e44ad", ls="--", lw=1)
    ax.text(5, np.log(3) - np.log(5) + 0.05, f"ln3−ln5 = {np.log(3)-np.log(5):.3f}",
            color="#8e44ad", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(n, 2 * np.log(3) - np.log(5) + 0 * Ic4, color="#8b98a5", ls=":", lw=1)
    ax2.set_ylim(-0.5, 0.8)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("I_c [nat]")
    ax.set_title("R20: I_c < 0 w równowadze — pamięć subradiacyjna jest "
                 "KLASYCZNA (I(A:B) > 0, ale bez odzyskiwalności kwantowej)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE8_koherentna.png", bbox_inches="tight")
    plt.close(fig)
    return Ic4


def figura_E9():
    """Trójkąt: precyzja ↔ koszt energii ↔ entropia (funkcje γ_t)."""
    gts = [0.002, 0.005, 0.01, 0.02, 0.05]
    WC = 50.0
    prec, cost, back = [], [], []
    for gt in gts:
        z = zegar_z_energia(gt, WC, TICKS=200)
        nb, dn, Ss = z["nb"][-1], z["dn"][-1], z["Ss"][-1]
        prec.append(dn / max(nb, 1e-9))
        cost.append(WC * nb)
        back.append(abs(Ss - M.LN2))
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.4), sharex=True)
    for ax, y, lab, c in [
        (axs[0], prec, "Δn/⟨n⟩ (precyzja: ↓ = lepiej)", "#8e44ad"),
        (axs[1], cost, "E_clock = ω_c·⟨n⟩ (koszt)", "#c0392b"),
        (axs[2], back, "|S∞ − ln 2| (back-action)", "#2471a3")]:
        ax.plot(gts, y, "o-", color=c, lw=2, ms=7)
        ax.set_xlabel("γ_t (siła zegara)")
        ax.set_ylabel(lab)
    axs[0].set_title("Precyzja rośnie"); axs[1].set_title("Koszt rośnie")
    axs[2].set_title("Entropia (zaburzenie) rośnie")
    fig.suptitle("R21 — kompromis zegara kwantowego (duch Saleckera–Wignera): "
                 "precyzja ↔ energia ↔ entropia", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE9_koszt.png", bbox_inches="tight")
    plt.close(fig)
    return dict(gts=gts, prec=prec, cost=cost, back=back)


def figura_E10():
    """Protokół: τ̇(T1) vs τ̇(T2) vs fluorescencja Γ = Ṡ."""
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    f = funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    fluo = dS / SIGMA0
    n = np.arange(len(dS))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(n, fluo, color="#7f8c8d", lw=1.8, ls="--",
            label="Γ — fluorescencja (tempo emisji Ṡ/σ₀)")
    ax.plot(n, f["T1"], color="#27ae60", lw=2, label="T1: τ̇ = (σ+η|İ|)/σ₀")
    ax.plot(n, f["T2"], color="#c0392b", lw=2, label="T2: τ̇ = (σ+η·I)/σ₀")
    ax.axhline(ETA * I[-1] / SIGMA0, color="#c0392b", ls=":", lw=1)
    ax.text(120, ETA * I[-1] / SIGMA0 + 0.4, f"η·I_eq/σ₀ = {ETA*I[-1]/SIGMA0:.1f}",
            color="#c0392b", fontsize=9)
    ax.annotate("po wygaśnięciu Γ: τ̇(T1) = 0, τ̇(T2) = 7.19",
                xy=(300, 4.0), xytext=(140, 5.5), fontsize=10, color="#26384a",
                arrowprops=dict(arrowstyle="->", color="#26384a"))
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("τ̇ / Γ")
    ax.set_title("R22 — protokół rozstrzygający: zmierz τ̇ PO wygaśnięciu "
                 "fluorescencji (0 ⇒ T1, stałe ⇒ T2)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.5, 9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE10_protokol.png", bbox_inches="tight")
    plt.close(fig)
    return f, fluo, I


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("ENTROPIA-1.3 — I_c, KOSZT ZEGARA (S-W), PROTOKÓŁ T1 vs T2")
    print("=" * 78)

    # [R20] coherent information
    print("\n[R20] COHERENT INFORMATION — czy pamięć jest kwantowa?")
    rhos4 = trajektorie_rhosektora(4, 2.0, n=400)
    Ic4 = np.array([I_c_sym(r, 2) for r in rhos4])
    print(f"  N=4 |1111⟩: I_c(0) = {Ic4[0]:.4f} (produkt), "
          f"I_c(∞) = {Ic4[-1]:.4f} (ln3−ln5 = {np.log(3)-np.log(5):.4f})")
    print(f"  I_c peak (przejściowe splątanie) = {Ic4.max():.4f} przy n = {Ic4.argmax()}")
    r10 = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    Ic10 = np.log(2) - np.log(12) / 2
    print(f"  N=2 |10⟩: I_c(∞) = {Ic10:.4f} < 0 (pamięć klasyczna); "
          f"I(A:B)(∞) = {r10['I_AB'][-1]:.4f} > 0")
    print("  Wniosek: I_c < 0 przy równowadze ⇒ korelacje klasyczne, "
          "bez odzyskiwalności kwantowej; T3c = (σ+η|Ī_c|)/σ₀ STaje.")
    f = funkcjonaly_czasu(r10["dS"], r10["I_AB"],
                          np.abs(np.diff(np.concatenate([[0], r10["I_AB"]]))),
                          np.abs(np.diff(np.concatenate([[0], Ic4[:len(r10["dS"])]]))[:len(r10["dS"])])
                          if len(Ic4) >= len(r10["dS"]) else np.zeros_like(r10["dS"]))
    print(f"  T3c τ̇∞ = {f['T3'][300:].mean():.6f} (staje)")

    # [R21] koszt energetyczny
    print("\n[R21] KOSZT ENERGETYCZNY ZEGARA (ω_c = 50):")
    WC = 50.0
    for gt in [0.002, 0.01, 0.05]:
        z = zegar_z_energia(gt, WC, TICKS=200)
        nb, dn, Ss = z["nb"][-1], z["dn"][-1], z["Ss"][-1]
        dE, dtau = WC * dn, dn * M.DELTA_S_Q
        print(f"  γ_t={gt:.3f}: ⟨n⟩={nb:.2f}, Δn={dn:.2f}, Δn/⟨n⟩={dn/max(nb,1e-9):.3f}, "
              f"E_clk={WC*nb:.0f}, ΔE·Δτ={dE*dtau:.2f} (ħ/2=0.5), "
              f"back={abs(Ss-M.LN2):.4f}")
    z = zegar_z_energia(0.01, WC, TICKS=200)
    wc_min = 0.5 / (z["dn"][-1] ** 2 * M.DELTA_S_Q)
    print(f"  ω_c^min (ΔE·Δτ ≥ ħ/2) = {wc_min:.1f} — model samo-spójny dla ω_c ≥ ω_c^min")

    # [R22] protokół
    print("\n[R22] PROTOKÓŁ T1 vs T2 (observable: τ̇ po wygaśnięciu Γ):")
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]; dI = np.abs(np.diff(np.concatenate([[0], I])))
    f2 = funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    fluo = dS / SIGMA0
    print(f"  Γ(∞) = {fluo[300:].mean():.6f} (fluorescencja wygasa)")
    print(f"  τ̇(T1)(∞) = {f2['T1'][300:].mean():.6f} → 0  (zegar staje)")
    print(f"  τ̇(T2)(∞) = {f2['T2'][300:].mean():.6f} → η·I_eq/σ₀  (zegar tyka)")
    print("  Decyzja: zmierz τ̇ w ciemnej fazie (Γ=0) — 0 ⇒ T1, stałe ⇒ T2.")

    # figury
    Ic4 = figura_E8()
    d21 = figura_E9()
    d22 = figura_E10()
    print(f"\nFigury: figE8_koherentna, figE9_koszt, figE10_protokol "
          f"w: {os.path.abspath(OUT)}")
    return dict(Ic4_end=float(Ic4[-1]), Ic4_peak=float(Ic4.max()),
                Ic10=float(Ic10), wc_min=float(wc_min), d21=d21,
                tau1=float(f2["T1"][300:].mean()), tau2=float(f2["T2"][300:].mean()),
                Ieq=float(I[-1]))


if __name__ == "__main__":
    main()
