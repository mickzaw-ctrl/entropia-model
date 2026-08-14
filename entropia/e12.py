# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.2 — KONKURENCJA FUNKCJONAŁÓW CZASU, ODZYSKIWALNOŚĆ, FIZYCZNY 27×
=============================================================================
  Wg recenzji: rozdzielić, co jest falsyfikacją, a co konsekwencją definicji.
  Uruchamiamy KONKURENCYJNE definicje czasu (nie ratujemy jednej):

    T0:  dτ = σ/σ₀                  czysto entropiczny (pierwotny)
    T1:  dτ = (σ + η|İ|)/σ₀         informacyjny DYNAMICZNY
    T2:  dτ = (σ + η·I)/σ₀          informacyjny ABSOLUTNY  ← to był faktycznie
                                    użyty w ENTROPIA-1.1 (τ̇∞ = η·I_eq ≠ 0)
    T3:  dτ = (σ + η|Ṙ|)/σ₀         oparty na ODZYSKIWALNOŚCI R = D(t)/D(0)
                                    (trace distance w sektorze ciemnym)

  Dwa dodatkowe testy wg recenzji:
    (A) Odzyskiwalność: D(t) = ½‖ρ₀(t) − ρ₁(t)‖₁ dla dwóch ortogonalnych
        stanów w sektorze jasnym vs subradiacyjnym; M(t) = D(t)/D(0).
        „P_dark→1” nie wystarcza — mierzymy faktyczną pamięć.
    (B) Fizyczny test 27×: γ(T), η(T)=e^{−ω₀/T} wyprowadzone z konkretnej
        kąpieli (3D fotonowa J(ω)∝ω³ ⇒ γ∝T³; single-mode ⇒ γ∝(2n̄+1));
        R_T = τ̇(T_A)/τ̇(T_B) przy T_A = 3T_B — NIE przez skalowanie L.
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
from .extensions import czas_do_poziomu_T, dSdt_termiczne_analitycznie

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"

SIGMA0 = 0.01
ETA = 0.5


# -----------------------------------------------------------------------------
#  CZTERY FUNKCJONAŁY CZASU
# -----------------------------------------------------------------------------
def funkcjonaly_czasu(dS, I, dI, dR, eta=ETA, sigma0=SIGMA0):
    """Zwraca dict z τ̇ dla T0–T3 (te same historie, różne źródła tempa)."""
    dS = np.asarray(dS, float); I = np.asarray(I, float)
    dI = np.asarray(dI, float); dR = np.asarray(dR, float)
    return dict(
        T0=dS / sigma0,
        T1=(dS + eta * dI) / sigma0,
        T2=(dS + eta * I) / sigma0,
        T3=(dS + eta * np.abs(dR)) / sigma0,
    )


def stan_10_N2():
    """N=2 mieszanina sektorów: j=1 (|1,0⟩) waga ½ + j=0 (singlet) waga ½."""
    rho_j1 = np.zeros((3, 3), complex); rho_j1[1, 1] = 1.0
    rho_j0 = np.ones((1, 1), complex)
    return {1.0: (0.5, rho_j1), 0.0: (0.5, rho_j0)}


# -----------------------------------------------------------------------------
#  ODZYSKIWALNOŚĆ (trace distance w sektorze)
# -----------------------------------------------------------------------------
def trace_dist(r1, r2):
    diff = r1 - r2
    return float(np.sum(np.abs(np.linalg.eigvalsh((diff + diff.conj().T) / 2))))


def M_sektora(j, m1, m2, n=120, gamma=M.GAMMA_B, delta_tau=M.DELTA_TAU):
    """M(t) = D(t)/D(0) dla dwóch ortogonalnych stanów |j,m1⟩, |j,m2⟩."""
    from .dicke import lindblad_sektora
    L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=True)
    d = int(2 * j + 1)
    def ket(m):
        k = np.zeros((d, d), complex); k[int(m + j), int(m + j)] = 1.0
        return k
    r1, r2 = ket(m1), ket(m2)
    D0 = trace_dist(r1, r2)
    Ms = np.ones(n)
    for k in range(1, n):
        r1 = D._unvec(expm_multiply(L * delta_tau, D._vec(r1)), d)
        r2 = D._unvec(expm_multiply(L * delta_tau, D._vec(r2)), d)
        Ms[k] = trace_dist(r1, r2) / D0
    return Ms


def tabela_odzyskiwalnosci(Ns=(4, 10, 100), n=100):
    """M_bright vs M_dark (j=1) dla kilku N; j=0: M=1 analitycznie."""
    rows = []
    for N in Ns:
        Mb = M_sektora(N / 2.0, -N / 2.0, -N / 2.0 + 1, n=n)
        Md = M_sektora(1.0, -1.0, 0.0, n=n)
        rows.append(dict(N=N, Mb50=float(Mb[50]), Md50=float(Md[50]),
                         Md_end=float(Md[-1]), gain=float(Md[50] / Mb[50])))
    return rows


# -----------------------------------------------------------------------------
#  FIZYCZNY TEST 27× (z modelu kąpieli, nie ze skalowania L)
# -----------------------------------------------------------------------------
def nbar(x):
    return 1.0 / (np.exp(x) - 1.0)


def gamma_3d_foton(T, T_ref, g0=0.02):
    """Kąpiel 3D fotonowa: J(ω) ∝ ω³ ⇒ γ(T) ∝ T³ (Debye)."""
    return g0 * (T / T_ref) ** 3


def gamma_single_mode(T, w0, g0=0.02, T_ref=10.0 * 1.0):
    """Kąpiel single-mode: γ(T) ∝ (2n̄(ω₀,T)+1) ≈ T dla T ≫ ω₀."""
    return g0 * (2 * nbar(w0 / T) + 1) / (2 * nbar(w0 / T_ref) + 1)


def R_T_fizyczny(gA, etaA, gB, etaB, Sstar=0.5):
    """
    Stosunek tempa produkcji entropii przy dopasowanym S* (dwie kąpiele).
    R47 (audyt ENTROPIA-1.2): czasy przejścia S* ORAZ tempo dS/dt brane z
    termicznej kąpieli Gibbsa (dSdt_termiczne_analitycznie) — wcześniej tempo
    brano z formuły ∞-gorącej, co dawało wartości o 0.2–0.8% za niskie.
    """
    tA = czas_do_poziomu_T(gA, etaA, Sstar)
    tB = czas_do_poziomu_T(gB, etaB, Sstar)
    return float(dSdt_termiczne_analitycznie(gA, etaA, tA)
                 / dSdt_termiczne_analitycznie(gB, etaB, tB))


def test_27_fizyczny(TB=10.0, w0=1.0, Sstar=0.5):
    """T_A = 3·T_B: R_T dla 3D fotonowej i single-mode (η z kąpieli)."""
    TA = 3.0 * TB
    etaB, etaA = np.exp(-w0 / TB), np.exp(-w0 / TA)
    gB3, gA3 = gamma_3d_foton(TB, TB), gamma_3d_foton(TA, TB)
    gB1, gA1 = gamma_single_mode(TB, w0, T_ref=TB), gamma_single_mode(TA, w0, T_ref=TB)
    r3 = R_T_fizyczny(gA3, etaA, gB3, etaB, Sstar)
    r1 = R_T_fizyczny(gA1, etaA, gB1, etaB, Sstar)
    return dict(r_3d=r3, r_single=r1, TB=TB, TA=TA, etaA=etaA, etaB=etaB)


def zbieznosc_27(TBs=(3, 5, 10, 30, 100), w0=1.0, Sstar=0.5):
    """R_T(3D) → 27 w gorącym limicie (T_B/ω₀ → ∞)."""
    out = []
    for TB in TBs:
        r = test_27_fizyczny(TB, w0, Sstar)
        out.append((TB, r["r_3d"]))
    return out


# -----------------------------------------------------------------------------
#  R47 — ZAMKNIĘCIE ZNAJDZISKA AUDYTU nr 1
#  (R_T_fizyczny używa teraz SPÓJNEJ termicznej pochodnej; poniżej tabela
#  przed/po oraz figura porównawcza dla dokumentacji raportu.)
# -----------------------------------------------------------------------------
# Wartości PRZED poprawką (mieszana formuła, z audytu 2026-08-13):
R47_PRZED = {3: (29.420, 3.242), 5: (28.322, 3.138), 10: (27.618, 3.066),
             30: (27.197, 3.022), 100: (27.058, 3.006)}


def tabela_R47(TBs=(3, 5, 10, 30, 100), w0=1.0, Sstar=0.5):
    """Tabela przed/po dla R47: (TB, R_3D_przed, R_3D_po, R_single_przed, R_single_po)."""
    rows = []
    for TB in TBs:
        r = test_27_fizyczny(TB, w0, Sstar)
        przed = R47_PRZED.get(TB)
        rows.append(dict(TB=TB, przed3=przed[0] if przed else np.nan,
                         po3=r["r_3d"], przed1=przed[1] if przed else np.nan,
                         po1=r["r_single"]))
    return rows


def figura_R47():
    """R47: R_T przed vs po poprawce; limity 27 (3D) i 3 (single-mode)."""
    rows = tabela_R47()
    TBs = [rw["TB"] for rw in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(TBs, [rw["po3"] for rw in rows], "o-", color="#27ae60", lw=2.4,
            ms=8, label="3D fotonowa — po poprawce R47 (pełna spójna termiczna)")
    ax.plot(TBs, [rw["przed3"] for rw in rows], "s--", color="#c0392b", lw=2,
            ms=7, label="3D fotonowa — przed R47 (mieszana formuła)")
    ax.plot(TBs, [rw["po1"] for rw in rows], "o--", color="#8e44ad", lw=1.8,
            ms=6, label="single-mode — po poprawce R47")
    ax.plot(TBs, [rw["przed1"] for rw in rows], "s:", color="#1a5276", lw=1.6,
            ms=6, label="single-mode — przed R47")
    ax.axhline(27, color=C_G, ls=":", lw=1)
    ax.text(32, 27.3, "27 (3D, lim. gorący)", color=C_G, fontsize=9)
    ax.axhline(3, color=C_G, ls=":", lw=1)
    ax.text(32, 3.3, "3 (single-mode)", color=C_G, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("T_B/ω₀"); ax.set_ylabel("R_T = τ̇(T_A)/τ̇(T_B),  T_A = 3·T_B")
    ax.set_title("R47: R_T z PEŁNEJ spójnej termicznej (znalezisko audytu nr 1 "
                 "zamknięte) — różnica 0.2–0.8% przed poprawką")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figA3_27_poprawka.png", bbox_inches="tight")
    plt.close(fig)
    return rows


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E5():
    """Cztery funkcjonały: T0,T1,T3 stają; T2 plateau (N=2 |10⟩-typ)."""
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    dR = np.zeros_like(dS)                    # N=2: ciemna pamięć doskonała (Ṙ=0)
    f = funkcjonaly_czasu(dS, I, dI, dR)
    n = np.arange(len(dS))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for naz, c in [("T0", "#2471a3"), ("T1", "#27ae60"),
                   ("T2", "#c0392b"), ("T3", "#8e44ad")]:
        ax.plot(n, f[naz], color=c, lw=1.8, label={
            "T0": "T0 — σ (entropiczny)",
            "T1": "T1 — σ+η|İ| (dynamiczny)",
            "T2": "T2 — σ+η·I (absolutny)",
            "T3": "T3 — σ+η|Ṙ| (odzyskiwalność)"}[naz])
    ax.axhline(ETA * I[-1] / SIGMA0, color="#c0392b", ls=":", lw=1)
    ax.text(5, ETA * I[-1] / SIGMA0 + 0.3, f"η·I_eq/σ₀ = {ETA*I[-1]/SIGMA0:.1f}",
            color="#c0392b", fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("τ̇ [j. czasu/tyknięcie]")
    ax.set_title("ENTROPIA-1.2: cztery funkcjonały czasu — T0, T1, T3 STAJĄ, "
                 "T2 nie (τ̇ → η·I_eq ≠ 0)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.5, 8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE5_funkcjonaly.png", bbox_inches="tight")
    plt.close(fig)
    return f, I


def figura_E6():
    """Odzyskiwalność: M(t) jasny vs ciemny (j=1); j=0: M=1."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for N, c in [(4, "#2471a3"), (10, "#e67e22"), (100, "#c0392b")]:
        Mb = M_sektora(N / 2.0, -N / 2.0, -N / 2.0 + 1, n=100)
        Md = M_sektora(1.0, -1.0, 0.0, n=100)
        t = np.arange(100) * M.DELTA_TAU
        ax.semilogy(t, Mb, color=c, ls="--", lw=1.8, label=f"N={N}: jasny j=N/2")
        ax.semilogy(t, Md, color=c, lw=2, label=f"N={N}: ciemny j=1")
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(0.3, 1.02, "j=0 (parzyste N): M(t) = 1 dokładnie", color=C_G, fontsize=9)
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("M(t) = D(t)/D(0)")
    ax.set_title("Odzyskiwalność: pamięć w sektorze subradiacyjnym przeżywa "
                 "znacznie dłużej (j=1: ~31× przy N=100, t=12.5)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE6_odzyskiwalnosc.png", bbox_inches="tight")
    plt.close(fig)


def figura_E7():
    """Fizyczny 27×: R_T z modelu kąpieli (3D vs single-mode) + zbieżność."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0))
    r = test_27_fizyczny()
    ax = axs[0]
    ax.bar(["3D fotonowa\n(J(ω)∝ω³ ⇒ γ∝T³)", "single-mode\n(γ∝(2n̄+1))"],
           [r["r_3d"], r["r_single"]], color=["#27ae60", "#c0392b"], width=0.5)
    ax.axhline(27, color="#27ae60", ls=":", lw=1)
    ax.text(1.35, 27.5, "27", color="#27ae60", fontsize=9)
    ax.axhline(3, color="#c0392b", ls=":", lw=1)
    ax.text(1.35, 3.4, "3", color="#c0392b", fontsize=9)
    for i, v in enumerate([r["r_3d"], r["r_single"]]):
        ax.text(i, v + 0.6, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("R_T = τ̇(T_A)/τ̇(T_B),  T_A = 3·T_B")
    ax.set_title(f"Fizyczny test 27× (T_B = {r['TB']:.0f}·ω₀, S* = 0.5): "
                 "wynik zależy od widma kąpieli")
    ax = axs[1]
    zb = zbieznosc_27()
    TBs = [t for t, _ in zb]; Rs = [v for _, v in zb]
    ax.plot(TBs, Rs, "o-", color="#27ae60", lw=2, ms=8)
    ax.axhline(27, color=C_G, ls=":", lw=1)
    ax.text(30, 27.4, "27 (lim. gorący)", color=C_G, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("T_B/ω₀"); ax.set_ylabel("R_T (3D fotonowa)")
    ax.set_title("Zbieżność R_T → 27 w gorącym limicie (poprawki η skończonej T)")
    fig.suptitle("ENTROPIA-1.2 — 27× jako PREDYKCJA modelu kąpieli, nie symetrii L",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE7_27fizyczny.png", bbox_inches="tight")
    plt.close(fig)
    return r, zb


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("ENTROPIA-1.2 — KONKURENCJA FUNKCJONAŁÓW CZASU (T0–T3)")
    print("=" * 78)

    # [1] funkcjonały — N=2 |10⟩-typ
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    dR = np.zeros_like(dS)
    f = funkcjonaly_czasu(dS, I, dI, dR)
    print("\n[1] τ̇∞ (300..400), N=2 |10⟩-typ:")
    for naz in ["T0", "T1", "T2", "T3"]:
        v = f[naz][300:].mean()
        st = "STAJE" if v < 1e-6 else "NIE STAJE"
        print(f"    {naz}: τ̇∞ = {v:.6f}  → {st}")
    print(f"    T2: τ̇∞ = η·I_eq/σ₀ = {ETA*I[-1]/SIGMA0:.4f} "
          f"(I_eq = {I[-1]:.4f} = ln(2/√3))")
    print("    Diagnoza: ENTROPIA-1.1 używała T2 (absolutna I), nie T1 (|İ|).")

    # [2] odzyskiwalność
    print("\n[2] ODZYSKIWALNOŚĆ M(t) = D(t)/D(0):")
    rows = tabela_odzyskiwalnosci()
    for row in rows:
        print(f"    N={row['N']:3d}:  M_bright(50) = {row['Mb50']:.4f}   "
              f"M_dark j=1 (50) = {row['Md50']:.4f}   zysk = {row['gain']:.1f}×")
    print("    j=0 (N parzyste): M(t) = 1 dokładnie (Γ=0) — pamięć doskonała")
    print("    → „P_dark→1” + faktyczna pamięć: niska entropia NIE znaczy braku pamięci")

    # [3] fizyczny 27×
    print("\n[3] FIZYCZNY TEST 27× (z modelu kąpieli):")
    rT = test_27_fizyczny()
    print(f"    T_B = {rT['TB']:.0f}·ω₀, T_A = 3T_B, η_A = {rT['etaA']:.4f}, "
          f"η_B = {rT['etaB']:.4f}")
    print(f"    3D fotonowa (γ∝T³):   R_T = {rT['r_3d']:.3f}   (lim. 27)")
    print(f"    single-mode (γ∝(2n̄+1)): R_T = {rT['r_single']:.3f}   (lim. 3)")
    print("    Zbieżność R_T → 27 z gorącym limitem:")
    for TB, v in zbieznosc_27():
        print(f"      T_B/ω₀ = {TB:4d}:  R_T = {v:.3f}")

    # [3b] R47 — zamknięcie znaleziska audytu nr 1
    print("\n[3b] R47 — PEŁNA SPÓJNA TERMICZNA (znalezisko audytu nr 1 zamknięte):")
    for rw in tabela_R47():
        print(f"    T_B/ω₀ = {rw['TB']:3d}: 3D przed {rw['przed3']:.3f} → po "
              f"{rw['po3']:.3f}   single przed {rw['przed1']:.3f} → po "
              f"{rw['po1']:.3f}")

    # [4] werdykt
    print("\n[4] WERDYKT (co jest falsyfikacją, a co konsekwencją definicji):")
    print("    • 27× jako tożsamość S_B(t)=S_A(27t) — konsekwencja skalowania L")
    print("      (test solvera); 27× jako R_T z kąpieli 3D — PREDYKCJA FIZYCZNA.")
    print("    • S∞=ln(N+1) — konsekwencja wymiaru przestrzeni Dickego (geometria).")
    print("    • P_dark→1 — geometryczny podział Hilberta; odzyskiwalność M(t)")
    print("      mierzy faktyczną pamięć (j=1: 31× przy N=100; j=0: dokładnie 1).")
    print("    • „Czkanie” — prawdziwe dla T0, T1, T3; T2 (użyta w 1.1) je niszczy.")
    print("      Wybór funkcjonału = wybór teorii czasu (produkcja vs istnienie")
    print("      informacji) — rozstrzygalny eksperymentalnie (R17).")

    # figury
    figura_E5()
    figura_E6()
    figura_E7()
    rows47 = figura_R47()
    print(f"\nFigury: figE5_funkcjonaly, figE6_odzyskiwalnosc, figE7_27fizyczny, "
          f"figA3_27_poprawka (R47)")
    print(f"  w: {os.path.abspath(OUT)}")
    return dict(f=f, I=I, rows=rows, rT=rT,
                zb=zbieznosc_27(), Ieq=float(I[-1]), R47=rows47)


if __name__ == "__main__":
    main()
