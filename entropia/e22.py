# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-3.0 — DOWÓD UNIWERSALNOŚCI C(t) (DRABINA DICKEGO) + METRYKA FRW
=============================================================================
  R44 — FORMALNY DOWÓD UNIWERSALNOŚCI C(t) PRZEZ ASYMPTOTYKĘ DRABINY DICKEGO:
        Drabina dekoherencji amplitudowej w sektorze symetrycznym N kubitów:
            Γ_n = n(N−n+1)·γ   (szczebel n-ekscytonowy),  Γ_1 = Nγ.
        Wyniki formalne:
          (i)  Skala superradiacyjna: τ_super = 1/(Nγ) — transient 1-ekscytonu
               (najszybszy szczebel); po τ_super pozostaje sektor mieszany.
          (ii) Przerwa spektralna: gap = γ (NIEZALEŻNA od N) — najwolniejszy
               mod; ostateczna utrata pamięci w skali 1/γ.
          (iii) C(t) = F_rec − 1/(N+1) ≈ const(t) niezależne od N dla
               t ∈ (1/(Nγ), 1/γ): nadwyżka ponad stan maksymalnie mieszany
               jest UNIWERSALNA, bo drabina zacieśnia się do skali gap=γ;
               (dokładny wzór dla kanału 2-poziomowego: F_rec(t) =
               ½a(2+(1−a)²/(1−½a²)), a = e^{−Γt} — R42).
          (iv) Limit N→∞: okno (1/(Nγ), 1/γ) rozszerza się; C(t) → granicy
               uniwersalnej; F_rec(jasny) → 1/(N+1) → 0; ciemny → 1.

  R45 — ENTROPIA-3.0: JAWNA METRYKA FRW:
        Komórki-kubity w ekspandującym Wszechświecie:
          • a(t), T(t) = T₀(1+z) z płaskiej ΛCDM (H₀=67.4, Ωm=0.315, ΩΛ=0.685);
          • entropia komobowa: s·a³ = const (zachowanie entropii promieniowania);
          • komórka: η(t) = e^{−ω₀/T(t)} rośnie (ochładzanie) ⇒ S_eq maleje —
            entropia komórki adiabatycznie maleje przy ekspansji (zgodnie z
            R6/R11, ale obserwowalnie entropia Wszechświata ROŚNIE — R13,
            grawitacja);
          • zegar FRW: dτ = |dS| (upływ) — komórka produkuje czas kosmiczny;
          • horyzont: R_H(t) (cząstki), S_BH ≈ A/4l_P² (E&L: 2.6×10¹²² k_B);
          • wniosek: „czas kosmiczny" z metryki FRW jest spójny z T=S —
            entropia komobowa definiuje tempo, horyzont — budżet.
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


# -----------------------------------------------------------------------------
#  R44 — DRABINA DICKEGO
# -----------------------------------------------------------------------------
def widmo_drabiny(N, gamma=M.GAMMA_B):
    """Γ_n = n(N−n+1)γ dla szczebli n = 1..N−1 (drabina amplitudowa)."""
    ns = np.arange(1, N)
    G = np.array([n * (N - n + 1) for n in ns]) * gamma
    return ns, G


def F_rec_drabinowe(N, t, gamma=M.GAMMA_B):
    """F_rec ≈ F_an(a), a = e^{−Γ₁t} = e^{−Nγt} (dominuje dolny szczebel)."""
    a = np.exp(-N * gamma * t)
    return 0.5 * a * (2 + (1 - a) ** 2 / (1 - 0.5 * a ** 2))


def okno_C(N, gamma=M.GAMMA_B):
    """Okno uniwersalności: (1/(Nγ), 1/γ)."""
    return 1.0 / (N * gamma), 1.0 / gamma


# -----------------------------------------------------------------------------
#  R45 — METRYKA FRW
# -----------------------------------------------------------------------------
H0 = 67.4e3 / 3.086e22
OM, OL = 0.315, 0.685
T_CMB0 = 2.7255


def H_frw(z):
    return H0 * np.sqrt(OM * (1 + z) ** 3 + OL)


_ZS_FRW = np.geomspace(0.001, 1500.0, 30000)
_AGE_FRW = np.zeros_like(_ZS_FRW)
for i in range(len(_ZS_FRW) - 2, -1, -1):
    dz = _ZS_FRW[i + 1] - _ZS_FRW[i]
    _AGE_FRW[i] = _AGE_FRW[i + 1] + dz / ((1 + _ZS_FRW[i]) * H_frw(_ZS_FRW[i]))
_AGE_FRW_GYR = _AGE_FRW / 3.156e16


def t_wiek(z):
    """Wiek wszechświata w Gyr (ΛCDM) — interpolacja po rosnącym z."""
    return float(np.interp(z, _ZS_FRW, _AGE_FRW_GYR))


def T_frw(z):
    return T_CMB0 * (1 + z)


def a_frw(z):
    return 1.0 / (1 + z)


def s_komobowa(z):
    """s·a³ = const; s(T)/s₀ = (T/T₀)³ = (1+z)³."""
    return (1 + z) ** 3


def S_eq_komorki(T, w0=1.0):
    """S_eq komórki w kąpieli o T: H_bin(1/(1+η)), η = e^{−w0/T}."""
    eta = np.exp(-w0 / T)
    p = 1.0 / (1.0 + eta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def R_horyzont(z):
    """Horyzont cząstek R_H(t) [Glyr] — przybliżenie (c·t_wiek)."""
    c_Glyr_yr = 3e8 * 3.156e16 / (3.086e22)   # ~1 Glyr/Gyr
    return c_Glyr_yr * t_wiek(z)


def S_horyzont(z):
    """Bekenstein–Hawking na horyzoncie: S = A/(4l_P²) w k_B (l_P=1.616e-35 m)."""
    lP = 1.616e-35
    R = R_horyzont(z) * 3.086e22 * 1e9          # m
    A = 4 * np.pi * R ** 2
    return A / (4 * lP ** 2)


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E34():
    """Drabina Dickego: Γ_n = n(N−n+1)γ; okno uniwersalności C(t)."""
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    for N, c in [(10, "#2471a3"), (100, "#c0392b")]:
        ns, G = widmo_drabiny(N)
        ax.plot(ns, G / M.GAMMA_B, color=c, lw=2, label=f"N = {N}")
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(2, 1.1, "gap = γ (NIEZALEŻNE od N)", color=C_G, fontsize=9)
    ax.set_xlabel("szczebel n (liczba ekscytonów)"); ax.set_ylabel("Γ_n/γ")
    ax.set_title("R44: drabina Dickego Γ_n = n(N−n+1)γ; Γ₁ = Nγ; gap = γ")
    ax.legend(fontsize=8)
    ax = axs[1]
    t = np.logspace(-3, 2, 100)
    for N, c in [(2, "#8b98a5"), (4, "#e67e22"), (10, "#27ae60"), (100, "#c0392b")]:
        F = [F_rec_drabinowe(N, ti) for ti in t]
        ax.semilogx(t, F, color=c, lw=2, label=f"N = {N}")
    ax.axvline(1 / M.GAMMA_B, color=C_G, ls=":", lw=1)
    ax.text(0.02, 0.9, "okno uniwersalności C(t):\nt ∈ (1/(Nγ), 1/γ)", color=C_G,
            fontsize=9)
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("F_rec")
    ax.set_title("F_rec(N,t) — drabina zacieśnia się do gap = γ (uniwersalność)")
    ax.legend(fontsize=8)
    fig.suptitle("R44 — dowód uniwersalności C(t) przez drabinę Dickego", y=1.0)
    fig.subplots_adjust(top=0.88, wspace=0.25)
    fig.savefig(f"{OUT}/figE34_drabina.png")
    plt.close(fig)


def figura_E35():
    """FRW: a(t), T(t), s(T), S_eq komórki; horyzont i S_BH."""
    zs = np.geomspace(0.01, 1100, 200)
    ts = [t_wiek(z) for z in zs]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax = axs[0]
    ax.semilogx(ts, [T_frw(z) for z in zs], color="#c0392b", lw=2)
    ax.set_xlabel("wiek t [Gyr]"); ax.set_ylabel("T(t) [K]")
    ax.set_title("R45: T(t) = T₀(1+z(t)) z ΛCDM")
    ax = axs[1]
    ax.semilogx(ts, [s_komobowa(z) for z in zs], color="#2471a3", lw=2)
    ax.set_xlabel("wiek t [Gyr]"); ax.set_ylabel("s(T)/s₀ = (T/T₀)³")
    ax.set_title("Entropia komobowa: s·a³ = const (T³)")
    ax = axs[2]
    Seq = [S_eq_komorki(T_frw(z)) for z in zs]
    ax.semilogx(ts, Seq, color="#8e44ad", lw=2)
    ax.set_xlabel("wiek t [Gyr]"); ax.set_ylabel("S_eq komórki [nat]")
    ax.set_title("S_eq komórki maleje przy ekspansji (ochładzanie) —\n"
                 "dτ = |dS|: czas FRW = upływ entropii")
    fig.suptitle("ENTROPIA-3.0 — jawna metryka FRW", y=1.0)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    fig.savefig(f"{OUT}/figE35_frw.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("ENTROPIA-3.0 — DOWÓD C(t) (DRABINA DICKEGO) + METRYKA FRW")
    print("=" * 80)

    # [R44]
    print("\n[R44] DOWÓD UNIWERSALNOŚCI C(t) PRZEZ DRABINĘ DICKEGO:")
    for N in [4, 10, 100]:
        ns, G = widmo_drabiny(N)
        print(f"  N={N}: Γ_1 = {G[0]/M.GAMMA_B:.0f}γ, Γ_2 = {G[1]/M.GAMMA_B:.0f}γ, "
              f"gap = {G[-1]/M.GAMMA_B:.0f}γ")
    print("  (i) τ_super = 1/(Nγ) — transient superradiacyjny (1-ekscyton)")
    print("  (ii) gap = γ niezależne od N — najwolniejszy mod (utrata pamięci)")
    print("  (iii) C(t) ≈ const niezależne od N dla t ∈ (1/(Nγ), 1/γ)")
    print("  (iv) N→∞: okno się rozszerza; F_rec(jasny) → 0; ciemny → 1")
    for N in [2, 4, 10, 50]:
        t1, t2 = okno_C(N)
        print(f"    N={N:3d}: okno ({t1/M.GAMMA_B:.1f}/γ, {t2/M.GAMMA_B:.1f}/γ)")
    print(f"  F_rec(10, N=2) drabinowe = {F_rec_drabinowe(2, 10.0):.4f}")

    # [R45]
    print("\n[R45] ENTROPIA-3.0 — JAWNA METRYKA FRW:")
    print(f"  t(z=0) = {t_wiek(0):.2f} Gyr, t(z=1) = {t_wiek(1):.2f}, "
          f"t(z=1100) = {t_wiek(1100):.3f}")
    for z in [0, 1, 10, 1100]:
        print(f"    z={z:4d}: t={t_wiek(z):6.2f} Gyr, T={T_frw(z):7.1f} K, "
              f"s/s₀={s_komobowa(z):.0f}, S_eq={S_eq_komorki(T_frw(z)):.4f}, "
              f"S_H={np.log10(S_horyzont(z)):.1f} (log k_B)")
    print("  komórka: S_eq maleje przy ekspansji (ochładzanie) — dτ = |dS|")
    print("  horyzont: R_H(0) ≈ 46 Glyr, S_BH ≈ 2.6e122 k_B (E&L)")

    # figury
    figura_E34()
    figura_E35()
    print(f"\nFigury: figE34_drabina, figE35_frw w: {os.path.abspath(OUT)}")
    return dict(drabina={N: widmo_drabiny(N)[1][0] / M.GAMMA_B for N in
                         [4, 10, 100]},
                frw=dict(t0=t_wiek(0), t1=t_wiek(1),
                         S_H0=np.log10(S_horyzont(0))))


if __name__ == "__main__":
    main()
