# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.4 — KANAŁ ODZYSKIWANIA (FIDELITY), ω_c(T), PROTOKÓŁ E2E
=============================================================================
  Trzy kolejne domknięcia programu falsyfikacyjnego:

  R21 — FIDELITY-BASED RECOVERY:
        M_F(t) = 1 − F(ρ₀(t), ρ₁(t)) (fidelity dwóch ortogonalnych stanów)
        + F_e (entanglement fidelity kanału sektora). Jasny j=N/2 traci
        rozróżnialność (M_F → 0); ciemny j=1 wolniej; j=0: M_F = 1, F_e = 1
        (kanał identyczności — doskonały odzysk). Fuchs–van de Graaf wiąże
        M_F z trace distance (R19.1).

  R22 — PEŁNY RACHUNEK ω_c(T) Z FIZYCZNEJ KĄPIELI:
        Częstość oscylatora-zegara z ograniczeń fizycznych:
          (a) obsada termiczna: n̄(ω_c,T) < ε  ⇒  ω_c > T·ln(1/ε)
          (b) energia-czas:     ΔE·Δτ ≥ ħ/2    ⇒  ω_c ≥ ω_c^min (R20: 1.7)
          (c) pojemność:        MLEV ≥ ln2/δs  ⇒  69 (δs = 0.01)
        Lim. gorący: ω_c ∝ T (rozdzielczość czasu ~ T), podczas gdy tempo
        produkcji entropii ∝ T³ (27×). Zegar w gorętszym otoczeniu musi tykać
        z wyższą częstością, by uniknąć szumu termicznego.

  R23 — PROTOKÓŁ R20.3 END-TO-END Z DETEKCJĄ FOTONÓW:
        Pełna symulacja Monte Carlo: stany |10⟩-typ, kąpiel kolektywna,
        detekcja fotonów (Poisson o tempie Ṡ/δs z wydajnością η_det),
        odczyty zegara (kwantowane). Faza ciemna: po ostatnim fotonie mierzymy
        τ̇; decyzja T1 (τ̇ ≈ 0) vs T2 (τ̇ = η·I_eq/σ₀). Moc testu vs
        długość integracji M i wydajność detekcji η_det.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, eigh

from . import core as M
from . import dicke as D
from .e12 import stan_10_N2, funkcjonaly_czasu, SIGMA0, ETA

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"


# -----------------------------------------------------------------------------
#  R21 — FIDELITY-BASED RECOVERY
# -----------------------------------------------------------------------------
def fidelity(rho, sigma):
    """F(ρ,σ) = (Tr√(√ρ σ √ρ))² — stabilnie przez eigh."""
    a = (rho + rho.conj().T) / 2.0
    b = (sigma + sigma.conj().T) / 2.0
    ev, V = eigh(a)
    ev = np.clip(ev, 0.0, None)
    sq = (V * np.sqrt(ev)) @ V.conj().T
    m = sq @ b @ sq
    evm = np.clip(np.linalg.eigvalsh((m + m.conj().T) / 2.0), 0.0, None)
    return float(np.sum(np.sqrt(evm)) ** 2)


def MF_sektora(j, m1, m2, n=100, gamma=M.GAMMA_B, delta_tau=M.DELTA_TAU):
    """M_F(t) = 1 − F(ρ₁(t), ρ₂(t)) — fidelity distinguishability."""
    from .dicke import lindblad_sektora, _vec, _unvec
    from scipy.sparse.linalg import expm_multiply
    if j < 1e-9:
        return np.ones(n)                      # j=0: Γ=0, M_F=1 zawsze
    d = int(2 * j + 1)
    if d <= 32:
        from .dicke import propagator_sektora
        L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=False)
        U = propagator_sektora(L, delta_tau)
        step = lambda r: _unvec(U @ _vec(r), d)
    else:
        L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=True)
        step = lambda r: _unvec(expm_multiply(L * delta_tau, _vec(r)), d)
    def ket(m):
        k = np.zeros((d, d), complex); k[int(m + j), int(m + j)] = 1.0
        return k
    r1, r2 = ket(m1), ket(m2)
    out = np.zeros(n)
    for k in range(n):
        out[k] = 1.0 - fidelity(r1, r2)
        r1 = step(r1)
        r2 = step(r2)
    return out


def F_e_sektora(j, n=60, gamma=M.GAMMA_B, delta_tau=M.DELTA_TAU):
    """Entanglement fidelity kanału sektora: F_e = (1/d²)Σ_ij ⟨j|Φ(|i⟩⟨j|)|i⟩."""
    from .dicke import lindblad_sektora, propagator_sektora, _vec, _unvec
    if j < 1e-9:
        return 1.0
    L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=False)
    U = propagator_sektora(L, delta_tau)
    d = int(2 * j + 1)
    Fe = 0.0
    for i in range(d):
        for jj in range(d):
            rho = np.zeros((d, d), complex); rho[i, jj] = 1.0
            for _ in range(n):
                rho = _unvec(U @ _vec(rho), d)
            Fe += np.real(rho[jj, i])
    return Fe / d ** 2


# -----------------------------------------------------------------------------
#  R22 — ω_c(T) Z FIZYCZNEJ KĄPIELI
# -----------------------------------------------------------------------------
def omega_c_T(T, eps=0.01, wc_min=1.7):
    """
    Częstość zegara z ograniczeń:
      (a) termiczne: n̄(ω_c,T) < ε ⇒ ω_c ≥ T·ln(1/ε)
      (b) energia-czas: ω_c ≥ ω_c^min (ΔE·Δτ ≥ ħ/2, z R20)
    Zwraca (ω_c, ω_c_th, ω_c_min).
    """
    wc_th = T * np.log(1.0 / eps)
    return max(wc_th, wc_min), wc_th, wc_min


def tabela_omega_c(Ts=(1, 3, 10, 30, 100), eps=0.01):
    rows = []
    for T in Ts:
        wc, wc_th, wc_min = omega_c_T(T, eps)
        rows.append(dict(T=T, wc=wc, wc_th=wc_th, wc_min=wc_min))
    return rows


# -----------------------------------------------------------------------------
#  R23 — PROTOKÓŁ END-TO-END Z DETEKCJĄ FOTONÓW
# -----------------------------------------------------------------------------
def fotony_i_zegar(teoria, dS, I, dI, eta_det=1.0, dark_rate=0.0, seed=0):
    """
    Jedna realizacja protokołu:
      • fotony: Poisson(dS/δs), obserwowane z wydajnością η_det (binomial)
        + tło detektora (dark counts): Poisson(dark_rate) na tyknięcie;
      • zegar: τ̇ = funkcjonał (T1 lub T2), odczyt kwantowany (Poisson na δs).
    Zwraca clicks, n_last, tau (skumulowany), dtau.
    """
    rng = np.random.default_rng(seed)
    f = funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))[teoria]
    real_clicks = rng.poisson(np.maximum(dS, 0) / M.DELTA_S_Q)
    obs_clicks = rng.binomial(real_clicks.astype(int), eta_det)
    if dark_rate > 0:
        obs_clicks = obs_clicks + rng.poisson(dark_rate, size=len(obs_clicks))
    nz = np.nonzero(obs_clicks)[0]
    n_last = int(nz[-1]) if len(nz) else 0
    k = rng.poisson(np.maximum(f, 0) / SIGMA0)
    dtau = k * SIGMA0
    tau = np.cumsum(dtau)
    return obs_clicks, n_last, tau, dtau


def moc_testu(M_obs, teor, eta_det=1.0, dark_rate=0.0, n_real=200, seed0=0,
              prog=None):
    """
    Moc rozstrzygnięcia: P(decyzja poprawna) dla teorii `teor` przy integracji
    M_obs tyknięć po ostatnim fotonie. Próg: τ̇ < prog ⇒ T1.
    """
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    if prog is None:
        prog = ETA * I[-1] / (2.0 * SIGMA0)        # połowa plateau T2
    dec_T1 = []
    for seed in range(n_real):
        _, n_last, _, dtau = fotony_i_zegar(teor, dS, I, dI, eta_det, dark_rate,
                                            seed)
        seg = dtau[n_last + 1: n_last + 1 + M_obs]
        if len(seg) == 0:
            continue
        slope = seg.mean()
        dec_T1.append(slope < prog)
    dec_T1 = np.array(dec_T1)
    correct = np.mean(dec_T1) if teor == "T1" else 1.0 - np.mean(dec_T1)
    return float(correct), float(np.mean(dec_T1))


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E11():
    """Fidelity recovery: M_F(t) jasny vs ciemny; F_e per sektor."""
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    for N, c in [(4, "#2471a3"), (10, "#e67e22"), (100, "#c0392b")]:
        Mb = MF_sektora(N / 2.0, -N / 2.0, -N / 2.0 + 1, n=60)
        Md = MF_sektora(1.0, -1.0, 0.0, n=60)
        t = np.arange(60) * M.DELTA_TAU
        ax.semilogy(t, Mb, color=c, ls="--", lw=1.8, label=f"N={N}: jasny j=N/2")
        ax.semilogy(t, Md, color=c, lw=2, label=f"N={N}: ciemny j=1")
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(0.5, 1.05, "j=0: M_F(t) = 1 (doskonały odzysk)", color=C_G, fontsize=9)
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("M_F(t) = 1 − F(ρ₀,ρ₁)")
    ax.set_title("Fidelity-based recovery: rozróżnialność w sektorze ciemnym "
                 "przeżywa (j=1: ~7× przy N=100)")
    ax.legend(fontsize=7.5, ncol=2)
    ax = axs[1]
    js = [0.0, 0.5, 1.0, 1.5, 2.0]
    Fe = [F_e_sektora(j, n=60) for j in js]
    ax.plot(js, Fe, "o-", color="#8e44ad", lw=2, ms=9)
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(1.7, 1.02, "1 = kanał identyczności", color=C_G, fontsize=9)
    ax.set_xlabel("sektor j"); ax.set_ylabel("F_e (entanglement fidelity, t=15)")
    ax.set_title("Kanał ciemnych sektorów blisko identyczności (j=0: dokładnie 1)")
    fig.suptitle("R21 — fidelity-based quantum recovery channel", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE11_fidelity.png", bbox_inches="tight")
    plt.close(fig)
    return dict(Fe=dict(zip(js, Fe)))


def figura_E12():
    """ω_c(T): ograniczenia termiczne i energia-czas; lim. gorący ∝ T."""
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    for eps, c in [(0.1, "#e67e22"), (0.01, "#c0392b")]:
        Ts = np.logspace(0, 2.5, 30)
        wcs = [omega_c_T(T, eps)[0] for T in Ts]
        ax.loglog(Ts, wcs, color=c, lw=2, label=f"ε = {eps}")
    ax.loglog(Ts, 1.7 * np.ones_like(Ts), color=C_G, ls=":", lw=1,
              label="ω_c^min (ΔE·Δτ ≥ ħ/2)")
    ax.set_xlabel("T (temperatura kąpieli)"); ax.set_ylabel("ω_c (częstość zegara)")
    ax.set_title("ω_c(T) z fizycznej kąpieli: n̄(ω_c,T) < ε ⇒ ω_c ∝ T (lim. gorący)")
    ax.legend(fontsize=8)
    ax = axs[1]
    eps = 0.01
    Ts = np.array([1, 3, 10, 30, 100])
    wc = np.array([omega_c_T(T, eps)[0] for T in Ts])
    ratio = wc[1] / wc[0]
    ax.plot(Ts, wc, "o-", color="#c0392b", lw=2, ms=8)
    ax.set_xscale("log")
    ax.set_xlabel("T"); ax.set_ylabel("ω_c")
    ax.set_title(f"T_A = 3·T_B ⇒ ω_c(A)/ω_c(B) = {ratio:.1f} (rozdzielczość ~ T),\n"
                 f"a tempo produkcji entropii ∝ T³ ⇒ 27× (rozdzielenie dwóch skal)")
    fig.suptitle("R22 — pełny rachunek ω_c(T)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE12_omega_T.png", bbox_inches="tight")
    plt.close(fig)
    return dict(ratio3=float(ratio))


def figura_E13():
    """Protokół end-to-end: fotony, zegar T1/T2, moc testu vs M i η_det."""
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    f = funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    n = np.arange(len(dS))
    clicks, n_last, tau1, _ = fotony_i_zegar("T1", dS, I, dI, eta_det=1.0, seed=1)
    _, _, tau2, _ = fotony_i_zegar("T2", dS, I, dI, eta_det=1.0, seed=1)

    fig, axs = plt.subplots(3, 1, figsize=(10.0, 11.0), sharex=True)
    ax = axs[0]
    ax.bar(n, clicks, width=0.8, color="#7f8c8d", alpha=0.85)
    ax.axvline(n_last, color="#c0392b", ls="--", lw=1.2)
    ax.text(n_last + 3, clicks.max() * 0.8, f"ostatni foton n={n_last}",
            color="#c0392b", fontsize=9)
    ax.set_ylabel("fotony/tyknięcie")
    ax.set_title("R23 — protokół end-to-end: detekcja fotonów (faza jasna) i zegar")

    ax = axs[1]
    ax.plot(n, tau1, color="#27ae60", lw=2, label="T1: τ(n) — po ostatnim fotonie staje")
    ax.plot(n, tau2, color="#c0392b", lw=2, label="T2: τ(n) — tyka dalej (η·I_eq/σ₀)")
    ax.axvline(n_last, color="#c0392b", ls="--", lw=1.2)
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.set_ylabel("τ (odczyt zegara)")
    ax.legend(fontsize=8)
    ax.text(n_last + 5, tau2[-1] * 0.55, "faza ciemna: τ̇(T1) = 0, τ̇(T2) = 7.2",
            fontsize=9, color="#26384a")

    ax = axs[2]
    for teor, c in [("T1", "#27ae60"), ("T2", "#c0392b")]:
        for etad, mstyle in [(1.0, "o-"), (0.5, "s--"), (0.1, "d:")]:
            Ms = [10, 30, 60, 100]
            pw = [moc_testu(M, teor, eta_det=etad, n_real=60)[0] for M in Ms]
            ax.plot(Ms, pw, mstyle, color=c, lw=1.6, ms=6,
                    label=f"{teor}, η_det = {etad}")
    ax.axhline(0.5, color=C_G, ls=":", lw=1)
    ax.text(10, 0.53, "przypadek", color=C_G, fontsize=8)
    ax.set_xlabel("M — długość integracji w fazie ciemnej (tyknięcia)")
    ax.set_ylabel("moc testu (P poprawna decyzja)")
    ax.set_title("Moc rozstrzygnięcia T1 vs T2: > 0.99 już przy M = 10, "
                 "nawet przy η_det = 0.1")
    ax.legend(fontsize=7.5, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE13_protokol_e2e.png", bbox_inches="tight")
    plt.close(fig)
    return dict(n_last=n_last, tau1_end=float(tau1[-1]), tau2_end=float(tau2[-1]))


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("ENTROPIA-1.4 — KANAŁ ODZYSKU (FIDELITY), ω_c(T), PROTOKÓŁ E2E")
    print("=" * 78)

    # [R21]
    print("\n[R21] FIDELITY-BASED RECOVERY:")
    for N in [4, 10, 100]:
        Mb = MF_sektora(N / 2.0, -N / 2.0, -N / 2.0 + 1, n=60)
        Md = MF_sektora(1.0, -1.0, 0.0, n=60)
        print(f"  N={N:3d}: M_F(bright,30) = {Mb[30]:.4f}, "
              f"M_F(dark j=1,30) = {Md[30]:.4f}, zysk = {Md[30]/Mb[30]:.1f}×")
    print("  j=0: M_F = 1, F_e = 1 (kanał identyczności — doskonały odzysk)")
    Fe = {j: F_e_sektora(j, n=60) for j in [0.0, 0.5, 1.0, 1.5, 2.0]}
    print("  F_e (entanglement fidelity, t=15): " +
          ", ".join(f"j={j}: {v:.3f}" for j, v in Fe.items()))
    print("  → kanał ciemnych sektorów blisko identyczności (Fuchs–van de Graaf "
          "wiąże z R19.1)")

    # [R22]
    print("\n[R22] PEŁNY RACHUNEK ω_c(T) Z FIZYCZNEJ KĄPIELI (ε = 0.01):")
    for row in tabela_omega_c(eps=0.01):
        print(f"  T={row['T']:4d}: ω_c^th = {row['wc_th']:6.1f}, ω_c = {row['wc']:6.1f}")
    wc3 = omega_c_T(30.0, 0.01)[0] / omega_c_T(10.0, 0.01)[0]
    print(f"  lim. gorący: ω_c ∝ T ⇒ ω_c(3T)/ω_c(T) = {wc3:.2f} "
          f"(rozdzielczość ~ T) vs produkcja entropii ∝ T³ (27×)")
    print(f"  pojemność zegara: MLEV ≥ ln2/δs = {M.LN2/M.DELTA_S_Q:.0f} (δs = {M.DELTA_S_Q})")

    # [R23]
    print("\n[R23] PROTOKÓŁ END-TO-END (MC, detekcja fotonów):")
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    clicks, n_last, _, _ = fotony_i_zegar("T1", dS, I, dI, eta_det=1.0, seed=1)
    print(f"  ostatni foton: n = {n_last} (t ≈ {n_last*M.DELTA_TAU:.1f}) — początek fazy ciemnej")
    for teor in ["T1", "T2"]:
        for M_obs in [10, 30, 60]:
            pw, dec = moc_testu(M_obs, teor, n_real=100)
            print(f"  {teor}, M={M_obs:3d}: moc = {pw:.3f}")
    print("  Moc vs wydajność detekcji (M = 30):")
    for etad in [1.0, 0.5, 0.1]:
        p1 = moc_testu(30, "T1", eta_det=etad, n_real=100)[0]
        p2 = moc_testu(30, "T2", eta_det=etad, n_real=100)[0]
        print(f"    η_det = {etad}: P(T1) = {p1:.3f}, P(T2) = {p2:.3f}")
    print("  Moc vs szum tła detektora (dark counts, M = 30, η_det = 1):")
    for dr in [0.0, 0.5, 2.0]:
        p1 = moc_testu(30, "T1", dark_rate=dr, n_real=100)[0]
        p2 = moc_testu(30, "T2", dark_rate=dr, n_real=100)[0]
        print(f"    dark_rate = {dr}: P(T1) = {p1:.3f}, P(T2) = {p2:.3f}")

    # figury
    d21 = figura_E11()
    d22 = figura_E12()
    d23 = figura_E13()
    print(f"\nFigury: figE11_fidelity, figE12_omega_T, figE13_protokol_e2e "
          f"w: {os.path.abspath(OUT)}")
    return dict(Fe=Fe, ratio3=float(wc3), n_last=int(n_last),
                moc_T1_10=float(moc_testu(10, "T1", n_real=100)[0]),
                moc_T2_10=float(moc_testu(10, "T2", n_real=100)[0]),
                moc_T1_010=float(moc_testu(30, "T1", eta_det=0.1, n_real=100)[0]),
                moc_T2_010=float(moc_testu(30, "T2", eta_det=0.1, n_real=100)[0]))


if __name__ == "__main__":
    main()
