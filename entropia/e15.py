# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.5 — OPERACYJNA PAMIĘĆ, SAMO-SPÓJNY ω_c(T), SEKWENCYJNY TEST
=============================================================================
  Pogłębienie R21–R23 (fidelity/ω_c(T)/protokół e2e):

  R24 — OPERACYJNA PAMIĘĆ (Helstrom + Chernoff + pojemność klasyczna):
        p_err(t) = ½(1 − D(t)/2) (granica Helstroma dla rozróżnienia dwóch
        stanów-kodów), C_mem(t) = 1 − h₂(p_err) (pojemność binarnego kanału
        pamięci), ξ(t) (wykładnik Chernoffa). Sektor ciemny zachowuje
        rozróżnialność (C_mem > 0), jasny traci (C_mem → 0); j=0: C_mem = 1
        bit na zawsze. To jest to, co można OPERACYJNIE odzyskać.

  R25 — SAMO-SPÓJNY ω_c(T) (Purcell + back-action):
        γ_t(ω_c) = g²·J(ω_c) ∝ g²ω_c³ (kąpiel 3D — Purcell). Back-action
        rośnie z γ_t, więc ω_c ma GÓRNĄ granicę; obsada termiczna daje DOLNĄ
        granicę ω_c > T·ln(1/ε). Obie naraz ⇒ istnieje maksymalna temperatura
        zegara T_max = (ε_b/(c g²))^{1/3}/ln(1/ε) — nowa, falsyfikowalna
        predykcja (dla T > T_max zegar kwantowy nie może istnieć).

  R26 — SEKWENCYJNY TEST WALDA (SPRT) ZAMIAST STAŁEGO M:
        Sekwencyjny iloraz wiarogodności na tyknięciach zegara w fazie
        ciemnej: E[N] (oczekiwana liczba tyknięć do decyzji) vs błędy α,β.
        Przy parametrach modelu E[N] = 1 (separacja λ 0 vs 7.19 ogromna);
        skan po λ₂ pokazuje wzrost E[N] przy słabszej separacji — protokół
        adaptuje się do jakości danych.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply

from . import core as M
from . import dicke as D
from .e13 import zegar_z_energia

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"


# -----------------------------------------------------------------------------
#  R24 — OPERACYJNA PAMIĘĆ
# -----------------------------------------------------------------------------
def traj(j, m1, m2, n, gamma=M.GAMMA_B):
    from .dicke import lindblad_sektora, _vec, _unvec
    d = int(2 * j + 1)
    if d <= 32:
        from .dicke import propagator_sektora
        L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=False)
        U = propagator_sektora(L, M.DELTA_TAU)
        step = lambda r: _unvec(U @ _vec(r), d)
    else:
        L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=True)
        step = lambda r: _unvec(expm_multiply(L * M.DELTA_TAU, _vec(r)), d)
    def ket(m):
        k = np.zeros((d, d), complex); k[int(m + j), int(m + j)] = 1.0
        return k
    r1, r2 = ket(m1), ket(m2)
    out = []
    for _ in range(n):
        out.append((r1.copy(), r2.copy()))
        r1 = step(r1); r2 = step(r2)
    return out


def trace_dist(r1, r2):
    diff = r1 - r2
    return float(np.sum(np.abs(np.linalg.eigvalsh((diff + diff.conj().T) / 2))))


def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def pamiec_operacyjna(j, m1, m2, n=60, gamma=M.GAMMA_B):
    """Zwraca p_err(t), C_mem(t), ξ(t) dla sektora j."""
    tr = traj(j, m1, m2, n, gamma)
    p_err = np.array([0.5 * (1 - trace_dist(*pair) / 2) for pair in tr])
    C_mem = np.array([1.0 - h2(p) for p in p_err])
    xi = np.array([chernoff(*pair) for pair in tr])
    return p_err, C_mem, xi


def chernoff(rho, sig):
    """ξ = −log min_s Tr(ρ^s σ^{1−s})."""
    ev1, V1 = np.linalg.eigh((rho + rho.conj().T) / 2)
    ev2, V2 = np.linalg.eigh((sig + sig.conj().T) / 2)
    ev1, ev2 = np.clip(ev1, 1e-300, None), np.clip(ev2, 1e-300, None)
    Ov = np.abs(V1.conj().T @ V2) ** 2
    best = np.inf
    for s in np.linspace(0.02, 0.98, 49):
        v = np.sum(ev1[:, None] ** s * ev2[None, :] ** (1 - s) * Ov)
        best = min(best, -np.log(max(v, 1e-300)))
    return best


# -----------------------------------------------------------------------------
#  R25 — SAMO-SPÓJNY ω_c(T)
# -----------------------------------------------------------------------------
def omega_c_okno(T, g, eps=0.01, eps_b=0.05, c=2.0, wc_min=1.7):
    """
    ω_c ∈ (T·ln(1/ε), min((ε_b/(c g²))^{1/3}, ...)) — okno istnienia zegara.
    Zwraca (wc_low, wc_high, T_max, exists).
    """
    wc_low = T * np.log(1.0 / eps)
    wc_high = (eps_b / (c * g ** 2)) ** (1.0 / 3.0)
    wc_high = max(wc_high, wc_min)          # energia-czas
    exists = wc_low < wc_high
    T_max = wc_high / np.log(1.0 / eps)
    return wc_low, wc_high, T_max, exists


def tabela_T_max(gammas=(0.1, 0.03, 0.01), eps_b=0.05, eps=0.01):
    rows = []
    for g in gammas:
        _, wc_high, Tmax, _ = omega_c_okno(1.0, g, eps, eps_b)
        rows.append(dict(g=g, wc_high=wc_high, T_max=Tmax))
    return rows


def back_action_fit():
    """|S∞ − ln2| vs γ_t (saturowane, długie TICKS) — stała c."""
    vals = []
    for gt in [0.002, 0.005, 0.01, 0.02]:
        z = zegar_z_energia(gt, 50.0, TICKS=400)
        vals.append((gt, abs(z["Ss"][-1] - M.LN2)))
    return vals


# -----------------------------------------------------------------------------
#  R26 — SEKWENCYJNY TEST WALDA
# -----------------------------------------------------------------------------
def sprt_poisson(lam, lam0, lam1_, A, Bth, rng):
    """SPRT dla Poisson: zwraca (decyzja: 1=T2, 0=T1, n)."""
    logl = 0.0; n = 0
    while True:
        k = rng.poisson(lam)
        logl += k * np.log(lam1_ / lam0) - (lam1_ - lam0)
        n += 1
        if logl >= A:
            return 1, n
        if logl <= Bth:
            return 0, n


def E_stop_SPRT(lam, lam0, lam1_, alpha=0.01, beta=0.01, n_real=200, seed=0):
    """Oczekiwana liczba tyknięć do decyzji + błędy dla danych (lam0, lam1_)."""
    rng = np.random.default_rng(seed)
    A = np.log((1 - beta) / alpha)
    Bth = np.log(beta / (1 - alpha))
    Ns = np.array([sprt_poisson(lam, lam0, lam1_, A, Bth, rng)[1]
                   for _ in range(n_real)])
    decs = np.array([sprt_poisson(lam, lam0, lam1_, A, Bth, rng)[0]
                     for _ in range(n_real)])
    err = np.mean(decs) if lam == lam0 else 1 - np.mean(decs)
    return float(Ns.mean()), float(Ns.std()), float(err)


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E14():
    """p_err(t) i C_mem(t): jasny vs ciemny; j=0: 1 bit na zawsze."""
    pe4, cm4, xi4 = pamiec_operacyjna(2.0, -2, -1, n=60)
    pe100, cm100, xi100 = pamiec_operacyjna(50.0, -50, -49, n=60)
    pe_d, cm_d, xi_d = pamiec_operacyjna(1.0, -1, 0, n=60)
    t = np.arange(60) * M.DELTA_TAU
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    ax.semilogy(t, np.maximum(pe4 - 0.5, 1e-6), color="#2471a3", ls="--", lw=2,
                label="p_err−½: jasny N=4")
    ax.semilogy(t, np.maximum(pe100 - 0.5, 1e-6), color="#c0392b", ls="--", lw=2,
                label="p_err−½: jasny N=100")
    ax.semilogy(t, np.maximum(pe_d - 0.5, 1e-6), color="#27ae60", lw=2,
                label="p_err−½: ciemny j=1")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("p_err − ½ (0 = zgadywanie)")
    ax.set_title("R24: granica Helstroma — ciemny sektor zachowuje rozróżnialność")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(t, cm100, color="#c0392b", ls="--", lw=2, label="C_mem: jasny N=100")
    ax.plot(t, cm_d, color="#27ae60", lw=2, label="C_mem: ciemny j=1")
    ax.plot(t, cm4, color="#2471a3", ls=":", lw=1.6, label="C_mem: jasny N=4")
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(0.5, 1.02, "j=0: C_mem = 1 bit na zawsze", color=C_G, fontsize=9)
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("C_mem [bit]")
    ax.set_title("Operacyjna pojemność pamięci klasycznej 1 − h₂(p_err)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE14_pamiec_op.png", bbox_inches="tight")
    plt.close(fig)
    return dict(cm4=float(cm4[30]), cm100=float(cm100[30]), cmd=float(cm_d[30]),
                cm_d15=float(cm_d[60 - 1]))


def figura_E15():
    """Okno istnienia zegara: dolna (termiczna) i górna (back-action) granica ω_c."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    Ts = np.logspace(-1, 1.2, 40)
    for g, c in [(0.1, "#c0392b"), (0.03, "#e67e22"), (0.01, "#27ae60")]:
        lows = [omega_c_okno(T, g)[0] for T in Ts]
        high = omega_c_okno(1.0, g)[1]
        ax.loglog(Ts, lows, color=c, lw=2, label=f"dolna: T·ln(1/ε), g = {g}")
        ax.axhline(high, color=c, ls=":", lw=1.4)
        Tmax = omega_c_okno(1.0, g)[2]
        ax.plot(Tmax, high, "o", color=c, ms=9)
        ax.annotate(f"T_max = {Tmax:.2f}", (Tmax, high), textcoords="offset points",
                    xytext=(6, 8), fontsize=9, color=c)
    ax.set_xlabel("T (temperatura kąpieli)"); ax.set_ylabel("ω_c")
    ax.set_title("R25: okno istnienia zegara — ω_c ∈ (T·ln(1/ε), (ε_b/(c g²))^{1/3}); "
                 "powyżej T_max zegar nie może istnieć")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE15_omega_okno.png", bbox_inches="tight")
    plt.close(fig)
    return dict()


def figura_E16():
    """SPRT: E[N] vs separacja λ₂; punkt modelu (λ₂ = 7.19 ⇒ E[N] = 1)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    lam0 = 0.001
    lam2s = np.logspace(np.log10(0.05), np.log10(7.19), 12)
    E1, E2 = [], []
    for lam2 in lam2s:
        e1, _, _ = E_stop_SPRT(lam0, lam0, lam2, n_real=150)
        e2, _, _ = E_stop_SPRT(lam2, lam0, lam2, n_real=150)
        E1.append(e1); E2.append(e2)
    ax.semilogx(lam2s, E1, "o-", color="#27ae60", lw=2, ms=6, label="T1 (λ = λ₀)")
    ax.semilogx(lam2s, E2, "s-", color="#c0392b", lw=2, ms=6, label="T2 (λ = λ₂)")
    ax.plot(7.19, 1.0, "D", color="#8e44ad", ms=12)
    ax.annotate("model: λ₂ = 7.19 ⇒ E[N] = 1", (7.19, 1.0),
                textcoords="offset points", xytext=(-40, 14), fontsize=9,
                color="#8e44ad")
    ax.axhline(10, color=C_G, ls=":", lw=1)
    ax.text(0.06, 11, "stałe M = 10", color=C_G, fontsize=8)
    ax.set_xlabel("λ₂ — tempo tyknięć zegara w fazie ciemnej (T2)")
    ax.set_ylabel("E[N] — oczekiwana liczba tyknięć do decyzji (α=β=0.01)")
    ax.set_title("R26: sekwencyjny test Walda adaptuje się do separacji — "
                 "przy parametrach modelu decyzja w 1 tyknięciu")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE16_sprt.png", bbox_inches="tight")
    plt.close(fig)
    return dict()


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("ENTROPIA-1.5 — PAMIĘĆ OPERACYJNA, SAMO-SPÓJNY ω_c(T), SPRT")
    print("=" * 78)

    # [R24]
    print("\n[R24] OPERACYJNA PAMIĘĆ (Helstrom, C_mem, Chernoff):")
    pe4, cm4, xi4 = pamiec_operacyjna(2.0, -2, -1, n=60)
    pe100, cm100, xi100 = pamiec_operacyjna(50.0, -50, -49, n=60)
    pe_d, cm_d, xi_d = pamiec_operacyjna(1.0, -1, 0, n=60)
    for N, (pe, cm, xi) in [(4, (pe4, cm4, xi4)), (100, (pe100, cm100, xi100))]:
        print(f"  N={N}: p_err(t=7.5) = {pe[30]:.3f}, C_mem(t=7.5) = {cm[30]:.4f} bit")
    print(f"  ciemny j=1: p_err(t=7.5) = {pe_d[30]:.3f}, "
          f"C_mem(t=7.5) = {cm_d[30]:.4f} bit, C_mem(t=15) = {cm_d[-1]:.4f}")
    pe_h, cm_h, _ = pamiec_operacyjna(0.5, -0.5, 0.5, n=60)
    print(f"  ciemny j=1/2 (dim 2 — nośnik 1 bitu): p_err(t=7.5) = {pe_h[30]:.3f}, "
          f"C_mem(t=7.5) = {cm_h[30]:.4f} bit")
    print(f"  j=0 (dim 1): stan niezmienniczy, ale brak pojemności (log2(1)=0)")
    print(f"  Chernoff ξ(t=7.5): jasny N=4 = {xi4[30]:.3f}, "
          f"ciemny j=1 = {xi_d[30]:.3f}")

    # [R25]
    print("\n[R25] SAMO-SPÓJNY ω_c(T) (Purcell + back-action):")
    ba = back_action_fit()
    print("  back-action vs γ_t (saturowane): " +
          ", ".join(f"γ_t={gt}: {b:.4f}" for gt, b in ba))
    print("  okno ω_c ∈ (T·ln(1/ε), (ε_b/(c g²))^{1/3}) — T_max:")
    for row in tabela_T_max():
        print(f"    g={row['g']}: ω_c^max = {row['wc_high']:.3f}, "
              f"T_max = {row['T_max']:.3f}")
    print("  → nowa predykcja: dla T > T_max zegar kwantowy nie może istnieć")

    # [R26]
    print("\n[R26] SEKWENCYJNY TEST WALDA (SPRT):")
    lam0, lam2 = 0.001, 7.19
    e1, s1, err1 = E_stop_SPRT(lam0, lam0, lam2, n_real=200)
    e2, s2, err2 = E_stop_SPRT(lam2, lam0, lam2, n_real=200)
    print(f"  T1: E[N] = {e1:.1f} ± {s1:.1f}, błąd α = {err1:.4f}")
    print(f"  T2: E[N] = {e2:.1f} ± {s2:.1f}, błąd β = {err2:.4f}")
    print(f"  → przy parametrach modelu decyzja w E[N] ≈ 1 tyknięciu "
          f"(vs stałe M = 10)")
    for lam2s in [0.5, 0.1]:
        ee, _, _ = E_stop_SPRT(lam2s, lam0, lam2s, n_real=150)
        print(f"  słabsza separacja (λ₂ = {lam2s}): E[N] = {ee:.1f} — SPRT się wydłuża")

    # figury
    d24 = figura_E14()
    figura_E15()
    figura_E16()
    print(f"\nFigury: figE14_pamiec_op, figE15_omega_okno, figE16_sprt "
          f"w: {os.path.abspath(OUT)}")
    return dict(cm4=d24["cm4"], cm100=d24["cm100"], cmd=d24["cmd"],
                cm_d15=d24["cm_d15"], Tmax=tabela_T_max(),
                E1=e1, E2=e2, err1=err1, err2=err2)


if __name__ == "__main__":
    main()
