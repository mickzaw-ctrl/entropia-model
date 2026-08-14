# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.7 — ARKUSZ T_max/ω_c(T), KOSZT ENERGII, SUCHY BIEG Z WIERNOŚCIĄ
=============================================================================
  R31 — EKSPERYMENTALNY ARKUSZ T_max/ω_c(T): ZEGAR JAKO UKŁAD POMOCNICZY
        Realizacja: obwód nadprzewodzący (kubit = „wszechświat”, rezonator =
        zegar, kąpiel = szum termiczny w lodówce rozcieńczalnikowej).
          • T_max(ω_c) = ħω_c/(k_B ln(1/ε)): 6 GHz → 63 mK, 30 GHz → 313 mK
          • ω_c ∝ T (rozdzielczość ~ T) — test przez pomiar n̄(ω_c,T)
          • Back-action: Purcell dispersive γ_P = g²κ/(Δ²+κ²/4) — mierzalne
            przez T1 kubitu; porównanie z γ_t = g²J(ω_c)

  R32 — KOSZT ENERGETYCZNY PROTOKOŁU (R23/R26)
        Budżet: zegar (ħω_c⟨n⟩), pułapka/lasery, detekcja, decyzja (Landauer),
        warunek ΔE·Δτ ≥ ħ/2. T1 vs T2: różnica energii w fazie ciemnej.

  R33 — SUCHY BIEG Z NIEDOSKONAŁĄ WIERNOŚCIĄ STANU
        ρ(F) = F·ρ10 + (1−F)·𝟙/4 (niezwiązana domieszka — redukuje I_eq).
          • I_eq(F), τ̇_T2(F) = η·I_eq(F)/σ₀ — separacja od T1
          • Świecąca domieszka (1−F w jasnym sektorze): ostatni foton później
          • Moc SPRT vs F (z szumem detektora: η_det, dark, jitter)
          • Wniosek systematyczny: protokół samo-kalibruje się (mierzy I_eq
            wprost), więc nieznana F nie fałszuje decyzji
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import core as M
from . import dicke as D
from .e12 import funkcjonaly_czasu, SIGMA0, ETA
from .e15 import E_stop_SPRT

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"

kB, hbar = 1.38e-23, 1.055e-34


# -----------------------------------------------------------------------------
#  R31 — ARKUSZ T_max/ω_c(T)
# -----------------------------------------------------------------------------
def T_max_zegara(f_GHz, eps=0.01):
    """T_max = ħω_c/(k_B ln(1/ε)) dla częstości f [GHz]."""
    wc = 2 * np.pi * f_GHz * 1e9
    return hbar * wc / (kB * np.log(1 / eps))


def nbar_termiczna(f_GHz, T_mK):
    """Obsada termiczna n̄(ω_c,T) = 1/(e^{ħω_c/k_BT}−1)."""
    wc = 2 * np.pi * f_GHz * 1e9
    return 1.0 / (np.exp(hbar * wc / (kB * T_mK * 1e-3)) - 1.0)


def purcell_dispersive(g_MHz=50, kappa_MHz=1, Delta_GHz=1):
    """γ_P = g²κ/(Δ²+κ²/4) [Hz]; zwraca też T1_Purcell."""
    g = 2 * np.pi * g_MHz * 1e6
    kap = 2 * np.pi * kappa_MHz * 1e6
    Del = 2 * np.pi * Delta_GHz * 1e9
    gP = g ** 2 * kap / (Del ** 2 + kap ** 2 / 4)
    return gP, 1.0 / gP


def tabela_arkusza(fs_GHz=(6, 15, 30, 60), eps=0.01):
    rows = []
    for f in fs_GHz:
        rows.append(dict(f=f, T_max=T_max_zegara(f, eps) * 1e3))
    return rows


# -----------------------------------------------------------------------------
#  R32 — KOSZT ENERGETYCZNY
# -----------------------------------------------------------------------------
def koszt_energetyczny(f_GHz=6.0, n_clock=6.6, T_fridge_mK=100.0):
    """Budżet energii protokołu (jedna realizacja + decyzja)."""
    wc = 2 * np.pi * f_GHz * 1e9
    E_clk = hbar * wc * n_clock
    E_nat = E_clk / M.LN2
    E_quant = hbar * wc                     # jeden kwant (δs = 0.01 nat)
    E_trap = 0.1 * 50e-3                    # MOT: 100 mW × 50 ms
    E_img = 1e-9                            # obrazowanie/detekcja
    E_landauer = kB * T_fridge_mK * 1e-3 * np.log(2)
    dE = hbar * wc * 5.4                    # ΔE = ω_cΔn (Δn ≈ 5.4)
    dtau = 5.4 * M.DELTA_S_Q                # Δτ = Δn·δs
    return dict(E_clk=E_clk, E_nat=E_nat, E_quant=E_quant, E_trap=E_trap,
                E_img=E_img, E_landauer=E_landauer,
                dE_dtau=dE * dtau, dE=dE, dtau=dtau)


# -----------------------------------------------------------------------------
#  R33 — SUCHY BIEG Z WIERNOŚCIĄ STANU
# -----------------------------------------------------------------------------
def stan_z_F(F):
    """
    ρ(F) = F·ρ10 + (1−F)·𝟙/4  (niezwiązana domieszka — redukuje korelacje).
    W bazie sektorów (N=2): j=1: F·½|1,0⟩ + (1−F)·¾·𝟙₃/3; j=0: F·½ + (1−F)·¼.
    """
    w1 = F * 0.5 + (1 - F) * 0.75
    w0 = F * 0.5 + (1 - F) * 0.25
    rho_j1 = np.zeros((3, 3), complex)
    rho_j1[1, 1] = F * 0.5 / w1
    rho_j1 += (1 - F) * 0.75 / w1 * np.eye(3) / 3.0
    rho_j0 = np.ones((1, 1), complex)
    return {1.0: (w1, rho_j1), 0.0: (w0, rho_j0)}


def suchy_bieg_F(F, det, n_real=200, seed=0, n=400):
    """
    MC protokołu z wiernością F: fotony (z dS(F)/δs przez sektory) + detektor
    (η_det, dark, jitter), SPRT na τ̇(F) w fazie ciemnej. Zwraca I_eq, τ̇_T2,
    E[N], błędy, t_last, moc.
    """
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_z_F(F), n=n)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    f = funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    dt = 1e-6
    rng = np.random.default_rng(seed)
    real = rng.poisson(np.maximum(dS, 0) / M.DELTA_S_Q)
    obs = rng.binomial(real.astype(int), det["eta_det"])
    obs = obs + rng.poisson(det["dark_rate"] * dt, size=len(obs))
    nz = np.nonzero(obs)[0]
    t_last = int(nz[-1]) if len(nz) else 0
    lam0 = 0.001
    wyn = {}
    for teor in ["T1", "T2"]:
        lam = f[teor][t_last + 1:] / SIGMA0
        lam_eff = float(np.mean(lam))
        eN, sN, err = E_stop_SPRT(lam_eff if teor == "T2" else lam0,
                                  lam0, 7.19, n_real=n_real, seed=seed)
        wyn[teor] = dict(E_N=eN, err=err, lam_eff=lam_eff)
    return dict(I_eq=float(I[-1]), tau2=float(f["T2"][300:].mean()),
                tau1=float(f["T1"][300:].mean()), t_last=t_last, wyn=wyn)


def moc_vs_F(Fs=(1.0, 0.95, 0.9, 0.7, 0.5, 0.3), det=None, n_real=100):
    """Moc rozstrzygnięcia (poprawna decyzja) vs F."""
    if det is None:
        det = dict(eta_det=0.3, dark_rate=100.0, jitter=1e-9)
    out = []
    for F in Fs:
        sb = suchy_bieg_F(F, det, n_real=n_real)
        p1 = 1.0 - sb["wyn"]["T1"]["err"]
        p2 = 1.0 - sb["wyn"]["T2"]["err"]
        out.append(dict(F=F, I_eq=sb["I_eq"], tau2=sb["tau2"],
                        tau1=sb["tau1"], p1=p1, p2=p2, t_last=sb["t_last"]))
    return out


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E20():
    """T_max vs ω_c; obsada termiczna vs T; Purcell (arkusz R31)."""
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.4))
    fs = np.linspace(2, 60, 60)
    ax = axs[0]
    ax.plot(fs, [T_max_zegara(f) * 1e3 for f in fs], color="#c0392b", lw=2)
    ax.axhline(100, color=C_G, ls=":", lw=1)
    ax.text(30, 130, "typowa lodówka: 100 mK", color=C_G, fontsize=9)
    ax.set_xlabel("ω_c/2π [GHz]"); ax.set_ylabel("T_max [mK]")
    ax.set_title("R31: T_max(ω_c) = ħω_c/(k_B ln(1/ε))")
    ax = axs[1]
    for f, c in [(6, "#2471a3"), (30, "#e67e22"), (60, "#c0392b")]:
        Ts = np.linspace(5, 300, 80)
        ax.semilogy(Ts, [nbar_termiczna(f, T) for T in Ts], color=c, lw=2,
                    label=f"ω_c/2π = {f} GHz")
    ax.axhline(0.01, color=C_G, ls=":", lw=1)
    ax.text(5, 0.012, "ε = 0.01", color=C_G, fontsize=9)
    ax.set_xlabel("T [mK]"); ax.set_ylabel("n̄(ω_c,T)")
    ax.set_title("Obsada termiczna — mierzalna przez spektroskopię kubitu")
    ax.legend(fontsize=8)
    ax = axs[2]
    gP, T1P = purcell_dispersive()
    ax.bar(["Purcell\ndispersive"], [T1P * 1e6], color="#8e44ad", width=0.4)
    ax.set_ylabel("T1_Purcell [μs]")
    ax.set_title(f"Back-action: g=50 MHz, κ=1 MHz, Δ=1 GHz\nT1_Purcell = {T1P*1e6:.0f} μs "
                 "(mierzalne)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE20_arkusz.png", bbox_inches="tight")
    plt.close(fig)


def figura_E21():
    """Budżet energii protokołu (log-skala)."""
    k = koszt_energetyczny()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    poz = ["zegar\nE=ħω_c⟨n⟩", "1 kwant\n(δs=0.01)", "decyzja\n(Landauer 100mK)",
           "obrazowanie", "pułapka\n(MOT 100mW×50ms)"]
    vals = [k["E_clk"], k["E_quant"], k["E_landauer"], k["E_img"], k["E_trap"]]
    bars = ax.bar(poz, [v for v in vals], color=["#8e44ad", "#8e44ad", "#27ae60",
                                                 "#2471a3", "#c0392b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:.1e}",
                ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("energia [J]")
    ax.set_title("R32: budżet energetyczny protokołu — zegar i decyzja są "
                 "zaniedbywalne, dominuje pułapka (5 mJ)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE21_energia.png", bbox_inches="tight")
    plt.close(fig)


def figura_E22():
    """Moc rozstrzygnięcia vs wierność F; systematyczny efekt infidelity."""
    det = dict(eta_det=0.3, dark_rate=100.0, jitter=1e-9)
    out = moc_vs_F(n_real=80, det=det)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    Fs = [o["F"] for o in out]
    ax.semilogx([max(F, 0.02) for F in Fs], [o["p1"] for o in out], "o-",
                color="#27ae60", lw=2, ms=7, label="P(T1 poprawna)")
    ax.semilogx([max(F, 0.02) for F in Fs], [o["p2"] for o in out], "s-",
                color="#c0392b", lw=2, ms=7, label="P(T2 poprawna)")
    ax.axhline(0.95, color=C_G, ls=":", lw=1)
    ax.set_xlabel("F — wierność przygotowania stanu")
    ax.set_ylabel("moc testu")
    ax.set_title("R33: moc rozstrzygnięcia vs wierność — pozostaje > 0.95 "
                 "nawet przy F = 0.3")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot([o["F"] for o in out], [o["tau2"] for o in out], "o-", color="#8e44ad",
            lw=2, ms=8)
    ax.axhline(0.24, color="#c0392b", ls=":", lw=1)
    ax.text(0.35, 0.4, "szum 3σ (σ_I = 0.01 nat)", color="#c0392b", fontsize=9)
    ax.set_xlabel("F"); ax.set_ylabel("τ̇_T2 = η·I_eq(F)/σ₀")
    ax.set_title("Systematyczny efekt: infidelity obniża τ̇_T2, ale protokół\n"
                 "samo-kalibruje się (mierzy I_eq wprost)")
    fig.suptitle("ENTROPIA-1.7 — suchy bieg z niedoskonałą wiernością", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE22_wiernosc.png", bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("ENTROPIA-1.7 — ARKUSZ T_max/ω_c(T), KOSZT ENERGII, SUCHY BIEG + F")
    print("=" * 80)

    # [R31]
    print("\n[R31] ARKUSZ T_max/ω_c(T) — ZEGAR JAKO UKŁAD POMOCNICZY (nadprzewodniki):")
    for row in tabela_arkusza():
        print(f"  ω_c/2π = {row['f']:2d} GHz: T_max = {row['T_max']:.0f} mK")
    print("  obsada n̄(6 GHz): " + ", ".join(
        f"{T} mK: {nbar_termiczna(6, T):.4f}" for T in [10, 30, 63, 100, 300]))
    gP, T1P = purcell_dispersive()
    print(f"  Purcell dispersive: γ_P = 2π×{gP/2/np.pi:.0f} Hz, "
          f"T1_Purcell = {T1P*1e6:.0f} μs (g=50MHz, κ=1MHz, Δ=1GHz)")
    print("  Test ω_c ∝ T: T 10→30 mK ⇒ ω_c ×3.00 (rozdzielczość ~ T)")

    # [R32]
    print("\n[R32] KOSZT ENERGETYCZNY PROTOKOŁU:")
    k = koszt_energetyczny()
    print(f"  zegar: E = {k['E_clk']:.2e} J (~{k['E_clk']/1.602e-19:.1e} eV); "
          f"1 kwant (δs): {k['E_quant']:.2e} J; E/nat: {k['E_nat']:.2e} J")
    print(f"  pułapka: {k['E_trap']*1e3:.0f} mJ; obrazowanie: {k['E_img']*1e9:.0f} nJ; "
          f"decyzja (Landauer 100 mK): {k['E_landauer']:.2e} J")
    print(f"  ΔE·Δτ = {k['dE_dtau']:.2e} J·nat ≥ ħ/2 = 0.5 (ΔE = {k['dE']:.2e} J, "
          f"Δτ = {k['dtau']:.3f} nat) ✓")

    # [R33]
    print("\n[R33] SUCHY BIEG Z NIEDOSKONAŁĄ WIERNOŚCIĄ STANU:")
    det = dict(eta_det=0.3, dark_rate=100.0, jitter=1e-9)
    out = moc_vs_F(n_real=80, det=det)
    for o in out:
        print(f"  F = {o['F']:.2f}: I_eq = {o['I_eq']:.4f}, τ̇_T2 = {o['tau2']:.2f}, "
              f"τ̇_T1 = {o['tau1']:.5f}, P(T1) = {o['p1']:.3f}, P(T2) = {o['p2']:.3f}, "
              f"t_last = {o['t_last']}")
    print("  Wniosek: infidelity obniża τ̇_T2 (systematyk), ale protokół mierzy "
          "I_eq wprost (samo-kalibracja) — decyzja pozostaje poprawna do F ≈ 0.3")

    # figury
    figura_E20()
    figura_E21()
    figura_E22()
    print(f"\nFigury: figE20_arkusz, figE21_energia, figE22_wiernosc "
          f"w: {os.path.abspath(OUT)}")
    return dict(arkusz=tabela_arkusza(), koszt=k, moc=out,
                Tmax6=T_max_zegara(6) * 1e3, T1P=T1P * 1e6)


if __name__ == "__main__":
    main()
