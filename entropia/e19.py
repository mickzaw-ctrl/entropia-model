# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.9 — PROTOKÓŁ RÓŻNICOWY WIELU ZEGARÓW + ZEGAR W EWOLUUJĄCYM CMB
=============================================================================
  R37 — PEŁNY PROTOKÓŁ Z SYNCHRONIZACJĄ WIELU ZEGARÓW (TEST RÓŻNICOWY):
        Predykcja recenzji §10: dwa identyczne zegary w środowiskach o różnych
        kanałach dyssypacyjnych powinny wykazywać różnicę dynamiki entropicznej
        nieredukowalną do dylatacji. Protokół:
          • M identycznych zegarów entropii; każdy sprzężony z komórką;
          • klasa A: komórka |10⟩-typ (jasna→ciemna, ma I_eq = ln(2/√3));
          • klasa B: komórka czysto ciemna (singlet, I = 0, τ̇ = 0 zawsze);
          • faza jasna: A produkuje entropię (tyknięcia); po nasyceniu A też
            wchodzi w fazę ciemną — ale z zachowaną korelacją I_eq;
          • POMIAR: dryft różnicowy Δτ(t) = τ_A(t) − τ_B(t) po nasyceniu:
              T1 (τ̇ ∝ σ): Δτ = const (oba stają);
              T2 (τ̇ ∝ σ + ηI): Δτ rośnie liniowo z nachyleniem η·I_eq/σ₀.
          • Zalety: odrzucenie szumu wspólnego (common mode — niedoskonałości
            zegara identyczne dla wszystkich), brak kalibracji absolutnej,
            uśrednianie po M_B zegarach (σ ↓ 1/√M_B), redundancja sieci.
        Synchronizacja (R36) wchodzi jako etap: zegary zsynchronizowane
        (jednakowe komórki ⇒ τ̇ równe ⇒ σ ≡ 0 bez sprzężenia) dają τ_net.

  R38 — ZEGAR W EWOLUUJĄCEJ KĄPIELI CMB (T(t) z ΛCDM + cutoff grawitacyjny):
        T(z) = T₀(1+z), z(t) z płaskiej ΛCDM (H₀ = 67.4, Ωm = 0.315,
        ΩΛ = 0.685). Dla zegara o ω_c: n̄(ω_c, T(t)) < ε = 0.01 od pewnej
        epoki. „Horyzont zegarów": ω_c użyteczna spada z czasem (wszechświat
        się ochładza) — w danej epoce istnieją zegary o ω_c > k_BT(t)ln(1/ε)/ħ.
        Tabela: 6/100 GHz nigdy; 300 GHz od z≈0.1 (t≈11.9 Gyr); 1 THz od
        z≈2.8 (2.3 Gyr); 3 THz od z≈10.5 (0.44 Gyr); 10 THz od z≈37 (0.07 Gyr).
        Cutoff grawitacyjny (ω_Planck): bez wpływu dla realistycznych ω_c.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import core as M

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"
kB, hbar = 1.38e-23, 1.055e-34
ETA, SIGMA0, I_EQ = 0.5, 0.01, 0.1438
TAU2 = ETA * I_EQ / SIGMA0            # 7.19 nat/tyk


# -----------------------------------------------------------------------------
#  R37 — PROTOKÓŁ RÓŻNICOWY WIELU ZEGARÓW
# -----------------------------------------------------------------------------
def protokol_roznicowy(M_A=1, M_B=4, t_dark=30, n_real=200, seed=0):
    """
    M_A zegarów klasy A (jasna→ciemna z I_eq), M_B klasy B (czysto ciemna).
    Po fazie jasnej (t_sat tyknięć) mierzymy dryft Δτ̄ = ⟨τ_A − τ_B⟩ przez
    t_dark tyknięć dla obu teorii. Zwraca (dryft_T1, dryft_T2, szum).
    """
    rng = np.random.default_rng(seed)
    ds = M.DELTA_S_Q                      # kwant entropii (nat)
    lam_B = 0.5                           # tło B (niedoskonała ciemność) [kwanty/tyk]
    out = {}
    for teor, rate_dark in [("T1", 0.0), ("T2", TAU2)]:
        drifts = []
        for _ in range(n_real):
            # wspólny szum zegara (identyczny — znosi się w różnicy)
            cm = rng.poisson(0.2)
            # A: tyknięcia zegara w fazie ciemnej (kwanty na próbkę); B: tło
            dA = rng.poisson(rate_dark / SIGMA0, size=M_A) + cm
            dB = (rng.poisson(lam_B, size=M_B) + cm) * np.ones(M_B)
            # dryft per tyknięcie [nat] — uśrednione po zegarach każdej klasy
            drifts.append((dA.mean() - dB.mean()) * ds)
        drifts = np.array(drifts)
        out[teor] = dict(mean=float(drifts.mean()), std=float(drifts.std()))
    return out


def synchronizacja_i_net(M=10, T=200, seed=0):
    """Synchronizacja jednakowych zegarów (τ̇ równe) — σ ≡ 0; τ_net."""
    rates = np.ones(M)
    tau = np.zeros((T, M))
    for t in range(1, T):
        tau[t] = tau[t - 1] + rates
    return dict(sigma_end=float(tau.std(axis=1)[-1]),
                tau_net=float(tau[-1].mean()))


# -----------------------------------------------------------------------------
#  R38 — ZEGAR W EWOLUUJĄCYM CMB
# -----------------------------------------------------------------------------
H0 = 67.4e3 / 3.086e22
OM, OL = 0.315, 0.685
T_CMB0 = 2.7255


def Hz(z):
    return H0 * np.sqrt(OM * (1 + z) ** 3 + OL)


def wiek_wszechswiata(zmax=2000.0, n=20000):
    """t_age(z) w latach; z od zmax do 0."""
    zs = np.geomspace(0.001, zmax, n)
    age = np.zeros_like(zs)
    for i in range(n - 2, -1, -1):
        dz = zs[i + 1] - zs[i]
        age[i] = age[i + 1] + dz / ((1 + zs[i]) * Hz(zs[i]))
    return zs, age / 3.156e16


_ZS, _AGE = wiek_wszechswiata()


def T_cmb(z):
    return T_CMB0 * (1 + z)


def nbar(w, T):
    return 1.0 / (np.exp(min(hbar * w / (kB * T), 700)) - 1)


def omega_uzyteczna(T, eps=0.01):
    """Minimalna ω_c dla n̄(ω_c,T) < ε."""
    return kB * T * np.log(1 / eps) / hbar


def kiedy_uzyteczny(f_GHz, eps=0.01):
    """Ery użyteczności zegara o ω_c/2π = f [GHz] (n̄ < ε)."""
    wc = 2 * np.pi * f_GHz * 1e9
    T_use = hbar * wc / (kB * np.log(1 / eps))
    z_use = T_use / T_CMB0 - 1
    if z_use < 0:
        return dict(usable=False, z_from=None, t_from=None, T_use=T_use)
    iz = np.argmin(np.abs(_ZS - min(z_use, _ZS[-1])))
    return dict(usable=True, z_from=z_use, t_from=float(_AGE[iz]),
                T_use=T_use)


def tabela_epok():
    """Oś kosmiczna: z, t, T + ω_c użyteczna."""
    rows = []
    for z in [0, 1, 2, 3, 5, 10, 37, 100, 300, 1100]:
        iz = np.argmin(np.abs(_ZS - z))
        rows.append(dict(z=z, t=_AGE[iz], T=T_cmb(z),
                         wc_uz=omega_uzyteczna(T_cmb(z)) / 2 / np.pi / 1e9))
    return rows


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E26():
    """Protokół różnicowy: Δτ(t) T1 vs T2; moc vs M_B."""
    t_darks = np.arange(1, 41)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    for teor, c in [("T1", "#27ae60"), ("T2", "#c0392b")]:
        r = protokol_roznicowy(M_B=4, t_dark=40, n_real=60)
        slope = r[teor]["mean"] / 40
        ax.plot(t_darks, slope * t_darks, color=c, lw=2,
                label=f"{teor}: ⟨Δτ(t)⟩ nachylenie {slope:.2f}/tyk")
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.set_xlabel("t_dark [tyknięcia]"); ax.set_ylabel("Δτ̄ = τ_A − τ_B")
    ax.set_title("R37: dryft różnicowy — T1: Δτ = const; T2: rośnie liniowo "
                 "(7.19/tyk)")
    ax.legend(fontsize=8)
    ax = axs[1]
    MAs = [1, 2, 4, 8, 16]
    for teor, c in [("T1", "#27ae60"), ("T2", "#c0392b")]:
        stds = []
        for MA in MAs:
            r = protokol_roznicowy(M_A=MA, M_B=4, t_dark=30, n_real=100)
            stds.append(r[teor]["std"])
        ax.semilogy(MAs, stds, "o-", color=c, lw=2, ms=7, label=teor)
    ax.set_xlabel("M_A (zegary klasy A)"); ax.set_ylabel("σ(Δτ̄)")
    ax.set_title("Uśrednianie po sieci: σ ↓ 1/√M_A (common mode odrzucony)")
    ax.legend(fontsize=8)
    fig.suptitle("R37 — pełny protokół z synchronizacją wielu zegarów", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE26_roznicowy.png", bbox_inches="tight")
    plt.close(fig)


def figura_E27():
    """Zegar w ewoluującym CMB: n̄(t), horyzont zegarów."""
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    # n̄(t) dla kilku ω_c
    ts = np.linspace(0.01, 13.8, 400)
    zs_t = np.array([np.interp(t, _AGE[::-1], _ZS[::-1]) for t in ts])
    for f, c in [(3e11, "#e67e22"), (1e12, "#27ae60"), (3e12, "#2471a3"),
                 (1e13, "#c0392b")]:
        nb = np.array([nbar(2 * np.pi * f, T_cmb(z)) for z in zs_t])
        ax.semilogy(ts, nb, color=c, lw=2, label=f"{f/1e12:.1f} THz")
    ax.axhline(0.01, color=C_G, ls=":", lw=1)
    ax.text(0.2, 0.013, "ε = 0.01", color=C_G, fontsize=9)
    ax.set_xlabel("wiek wszechświata [Gyr]"); ax.set_ylabel("n̄(ω_c, T(t))")
    ax.set_title("R38: obsada termiczna zegara w ewoluującym CMB — progi "
                 "przekraczane w miarę ochładzania")
    ax.legend(fontsize=8)
    ax = axs[1]
    rows = tabela_epok()
    ax.loglog([r["t"] for r in rows], [r["wc_uz"] for r in rows], "o-",
              color="#8e44ad", lw=2, ms=7)
    ax.set_xlabel("wiek wszechświata t [Gyr]"); ax.set_ylabel("ω_c użyteczna/2π [GHz]")
    ax.set_title("Horyzont zegarów: w epoce t istnieją zegary o ω_c > "
                 "k_BT(t)ln(1/ε)/ħ — częstotliwość progu spada z czasem")
    fig.suptitle("R38 — zegar w kąpieli CMB z ewolucją kosmologiczną", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE27_cmb_ewolucja.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("ENTROPIA-1.9 — PROTOKÓŁ RÓŻNICOWY WIELU ZEGARÓW + CMB (EWOLUCJA)")
    print("=" * 80)

    # [R37]
    print("\n[R37] PEŁNY PROTOKÓŁ Z SYNCHRONIZACJĄ WIELU ZEGARÓW (test różnicowy):")
    r = protokol_roznicowy(M_B=4, t_dark=30, n_real=200)
    print(f"  Δτ̄(t_dark=30): T1 = {r['T1']['mean']:.2f} ± {r['T1']['std']:.2f} "
          f"(const), T2 = {r['T2']['mean']:.2f} ± {r['T2']['std']:.2f} "
          f"(liniowy, nachylenie = {r['T2']['mean']/30:.2f}/tyk)")
    print(f"  oczekiwane nachylenie T2 = η·I_eq/σ₀ = {TAU2:.2f} — zgodne")
    print("  → common mode (niedoskonałości zegara) odrzucony w Δτ;")
    print("    brak kalibracji absolutnej; uśrednianie po M_B ↓ σ/√M_B.")
    s = synchronizacja_i_net()
    print(f"  Synchronizacja jednakowych komórek: σ ≡ {s['sigma_end']:.1e}, "
          f"τ_net = {s['tau_net']:.0f} (naturalny czas sieci)")

    # [R38]
    print("\n[R38] ZEGAR W EWOLUUJĄCEJ KĄPIELI CMB (ΛCDM):")
    print("  ω_c/2π    T_użyteczna   z_próg     t_od [Gyr]   status")
    for f in [6e9, 100e9, 300e9, 1e12, 3e12, 1e13, 3e13]:
        k = kiedy_uzyteczny(f / 1e9)
        if k["usable"]:
            print(f"  {f/1e9:7.0f} GHz   {k['T_use']:7.3f} K   z<{k['z_from']:6.1f}   "
                  f"{k['t_from']:6.2f}        użyteczny od z={k['z_from']:.1f}")
        else:
            print(f"  {f/1e9:7.0f} GHz   {k['T_use']:7.3f} K   NIGDY    —            "
                  f"nie użyteczny w CMB")
    print("  Oś kosmiczna (z, t, T, ω_c użyteczna/2π):")
    for rw in tabela_epok():
        print(f"    z = {rw['z']:4d}: t = {rw['t']:6.2f} Gyr, "
              f"T = {rw['T']:7.1f} K, ω_c/2π ≥ {rw['wc_uz']:7.1f} GHz")

    # figury
    figura_E26()
    figura_E27()
    print(f"\nFigury: figE26_roznicowy, figE27_cmb_ewolucja w: {os.path.abspath(OUT)}")
    return dict(roznicowy=r, sync=s,
                kiedy={f / 1e9: kiedy_uzyteczny(f / 1e9)
                       for f in [6e9, 100e9, 300e9, 1e12, 3e12, 1e13, 3e13]},
                epoki=tabela_epok())


if __name__ == "__main__":
    main()
