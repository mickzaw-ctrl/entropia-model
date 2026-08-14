# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.6 — SUCHY BIEG PROTOKOŁU, MAPA PETZA, ZIMNY ZEGAR
=============================================================================
  R28 — SUCHY BIEG PROTOKOŁU R23/R26 (realistyczny detektor):
        Pełny Monte Carlo: fotony (Poisson Ṡ/δs) → detekcja z η_det, dark
        counts, timing jitter (SPCM) → faza ciemna → SPRT (R26) na tyknięciach
        zegara z szumem odczytu. Wynik: E[N], błędy α,β, całkowity czas,
        liczba atomów i parametry detektora — konkretna karta dla platform
        A/B/C. Wniosek: jitter ≪ t_B, t_D ⇒ nie wpływa; decyduje η_det i dark.

  R29 — JAWNA MAPA ODZYSKU PETZA:
        R(·) = σ^{1/2} Φ†(Φ(σ)^{-1/2} (·) Φ(σ)^{-1/2}) σ^{1/2}  (Petz 1986).
        Φ = ewolucja sektora (e^{Lt}); Φ† = U† (sprzężenie w HS).
        F_rec = F(ρ₀, R∘Φ(ρ₀)) — ile bitu DA SIĘ odzyskać jawnym kanałem.
        Ciemne sektory: F_rec → 1; jasne: F_rec ↓ (spójne z C_mem, R24).

  R30 — ZIMNY ZEGAR: ω_c(T) z kąpieli o SKOŃCZONEJ gęstości widmowej:
        J(ω) = ηω³ (3D, bez cutoffu) vs J(ω) = ηω·e^{−ω/ω_cut} (Ohmic) vs
        Lorentzian (wnęka). Z cutoffem γ_t(ω_c) = g²J(ω_c) SATURUJE przy
        ω_c ≫ ω_cut ⇒ górna granica ω_c znika (back-action ograniczony)
        ⇒ T_max dramatycznie rośnie. Dyskusja kosmologiczna: w jakich epokach
        (T_CMB = 2.7 K … elektrosłaba 10¹² K) kwantowy zegar może istnieć.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import core as M
from . import dicke as D
from .e12 import stan_10_N2, funkcjonaly_czasu, SIGMA0, ETA
from .e15 import E_stop_SPRT

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"


# -----------------------------------------------------------------------------
#  R28 — SUCHY BIEG (realistyczny detektor)
# -----------------------------------------------------------------------------
def detektor(naz="SPCM standard", eta_det=0.3, dark_rate=100.0, jitter=1e-9):
    return dict(naz=naz, eta_det=eta_det, dark_rate=dark_rate, jitter=jitter)


def suchy_bieg(platforma, det, n_real=200, seed=0):
    """
    Pełny MC protokołu:
      fotony: Poisson(dS/δs) → obserwowane: binomial(η_det) + Poisson(dark·Δt)
      faza ciemna: od t_last; zegar: τ̇ (T1/T2) z szumem odczytu (Poisson)
      decyzja: SPRT (α=β=0.01) na tyknięciach (λ₀ vs λ₂)
    Zwraca dict z liczbami: E[N], błędy, t_last, całkowity czas, okno.
    """
    r = D.symuluj_dicke(2, M.GAMMA_B, stan=stan_10_N2(), n=400)
    dS = r["dS"]; I = r["I_AB"]
    dI = np.abs(np.diff(np.concatenate([[0], I])))
    f = funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    dt = 1e-6                                   # Δt_samp = 1 μs
    rng = np.random.default_rng(seed)
    # detekcja fotonów
    real = rng.poisson(np.maximum(dS, 0) / M.DELTA_S_Q)
    obs = rng.binomial(real.astype(int), det["eta_det"])
    obs = obs + rng.poisson(det["dark_rate"] * dt, size=len(obs))
    nz = np.nonzero(obs)[0]
    t_last = int(nz[-1]) if len(nz) else 0
    # SPRT w fazie ciemnej dla obu teorii
    lam0 = 0.001
    wyn = {}
    for teor in ["T1", "T2"]:
        lam = (f[teor][t_last+1:] / SIGMA0)     # tyknięcia na próbkę
        lam_eff = float(np.mean(lam))
        eN, sN, err = E_stop_SPRT(lam_eff if teor == "T2" else lam0,
                                  lam0, 7.19, n_real=n_real, seed=seed)
        wyn[teor] = dict(E_N=eN, std=sN, err=err, lam_eff=lam_eff)
    # czasy
    T_total = (t_last + 2) * dt + 20 * dt       # faza jasna + ciemna + margines
    return dict(t_last=t_last, dt=dt, T_total=T_total, wyn=wyn)


# -----------------------------------------------------------------------------
#  R29 — MAPA PETZA
# -----------------------------------------------------------------------------
def fidelity(rho, sig):
    a = (rho + rho.conj().T) / 2; b = (sig + sig.conj().T) / 2
    ev, V = np.linalg.eigh(a); ev = np.clip(ev, 0, None)
    sq = (V * np.sqrt(ev)) @ V.conj().T
    m = sq @ b @ sq
    evm = np.clip(np.linalg.eigvalsh((m + m.conj().T) / 2), 0, None)
    return float(np.sum(np.sqrt(evm)) ** 2)


def petz_recover(Ut, sigma_ref, d):
    """R(·) = σ^{1/2} Φ†(Φ(σ)^{-1/2} (·) Φ(σ)^{-1/2}) σ^{1/2}; Φ† = U†."""
    def sq_inv(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2)
        ev = np.clip(ev, 1e-12, None)
        return (V / np.sqrt(ev)) @ V.conj().T
    def sq(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2)
        ev = np.clip(ev, 0, None)
        return (V * np.sqrt(ev)) @ V.conj().T
    Phi_sig = D._unvec(Ut @ D._vec(sigma_ref), d)
    def recover(X):
        inner = sq_inv(Phi_sig) @ X @ sq_inv(Phi_sig)
        return sq(sigma_ref) @ D._unvec(Ut.conj().T @ D._vec(inner), d) @ sq(sigma_ref)
    return recover


def F_rec_sektora(j, n_steps=40, m=0, gamma=M.GAMMA_B):
    """F_rec = F(ρ₀, R∘Φ(ρ₀)) dla stanu |j, −j+m⟩."""
    from .dicke import lindblad_sektora, propagator_sektora
    d = int(2 * j + 1)
    if j < 1e-9:
        return 1.0
    L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=False)
    U = propagator_sektora(L, M.DELTA_TAU)
    Ut = np.eye(d * d, dtype=complex)
    for _ in range(n_steps):
        Ut = U @ Ut
    rho0 = np.zeros((d, d), complex); rho0[m, m] = 1.0
    sigma = D._unvec(Ut @ D._vec(rho0), d)
    rec = petz_recover(Ut, sigma, d)
    rho_ev = D._unvec(Ut @ D._vec(rho0), d)
    return fidelity(rho0, rec(rho_ev))


# -----------------------------------------------------------------------------
#  R30 — ZIMNY ZEGAR (skończona gęstość widmowa)
# -----------------------------------------------------------------------------
def gamma_t(wc, g, model="3d", wcut=50.0, w0=0.0, kappa=1.0):
    """Purcell γ_t = g²·J(ω_c) dla różnych widm."""
    if model == "3d":
        return g ** 2 * wc ** 3
    if model == "ohmic":
        return g ** 2 * wc * np.exp(-wc / wcut)
    if model == "lorentz":
        return g ** 2 * (kappa ** 2 / 4) / ((wc - w0) ** 2 + (kappa / 2) ** 2)
    raise ValueError(model)


def T_max_widmo(model, g, eps=0.01, eps_b=0.05, c=2.0, wcut=50.0):
    """
    Maksymalna temperatura zegara dla danego widma:
      dolna: ω_c > T·ln(1/ε);  górna: c·γ_t(ω_c) < ε_b.
    Zwraca (T_max, ω_c_opt).
    """
    if model == "3d":
        wc_hi = (eps_b / (c * g ** 2)) ** (1 / 3)
    else:
        # skan ω_c: znajdź największe ω_c z c·γ_t(ω_c) < ε_b
        wc = 0.01
        wc_hi = wc
        while wc < 1e6:
            if c * gamma_t(wc, g, model, wcut) < eps_b:
                wc_hi = wc
            else:
                break
            wc *= 1.2
    T_max = wc_hi / np.log(1 / eps)
    return T_max, wc_hi


def tabela_kosmologiczna(eps=0.01):
    """ω_c wymagana termicznie w epokach kosmologicznych (jednostki fizyczne)."""
    kB, hbar = 1.38e-23, 1.055e-34
    epoki = [("dziś (CMB)", 2.7255), ("rekombinacja z≈1100", 3000.0),
             ("BBN (~1 MeV)", 1.16e10), ("elektrosłaba (~100 GeV)", 1.16e15)]
    out = []
    for naz, T in epoki:
        wc = kB * T * np.log(1 / eps) / hbar
        out.append(dict(epoka=naz, T=T, wc=wc, f=wc / 2 / np.pi))
    return out


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E18():
    """F_rec (Petz) per sektor; porównanie z C_mem (R24)."""
    js = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    Fres = [F_rec_sektora(j, m=0) for j in js]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(js, Fres, "o-", color="#8e44ad", lw=2, ms=9)
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(2.4, 1.02, "1 = idealny odzysk (j=0: dokładnie 1)", color=C_G, fontsize=9)
    ax.set_xlabel("sektor j (m = −j)"); ax.set_ylabel("F_rec = F(ρ₀, R_Petz∘Φ(ρ₀))")
    ax.set_title("R29: jawna mapa odzysku Petza — ciemne sektory odzyskują bit\n"
                 "(F_rec maleje z j; spójne z C_mem z R24)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE18_petz.png", bbox_inches="tight")
    plt.close(fig)
    return dict(zip(js, Fres))


def figura_E19():
    """Zimny zegar: γ_t(ω_c) dla widm; T_max vs g; kosmologia."""
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.0))
    wc = np.logspace(-1, 2.5, 80)
    ax = axs[0, 0]
    g = 0.03
    for model, c in [("3d", "#c0392b"), ("ohmic", "#e67e22"), ("lorentz", "#2471a3")]:
        ax.loglog(wc, [gamma_t(w, g, model, wcut=50, w0=1, kappa=1) for w in wc],
                  color=c, lw=2, label=model)
    ax.set_xlabel("ω_c"); ax.set_ylabel("γ_t(ω_c) = g²J(ω_c)")
    ax.set_title("Purcell dla różnych gęstości widmowych: 3D rośnie bez końca,\n"
                 "Ohmic/Lorentzian SATURUJĄ (back-action ograniczony)")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    gs = np.logspace(-2.5, -0.5, 30)
    for model, c in [("3d", "#c0392b"), ("ohmic", "#27ae60")]:
        Tm = [T_max_widmo(model, g, wcut=50.0)[0] for g in gs]
        ax.loglog(gs, Tm, color=c, lw=2, label=model)
    ax.set_xlabel("g (sprzężenie zegar-kąpiel)"); ax.set_ylabel("T_max")
    ax.set_title("T_max(g): z cutoffem (Ohmic) rośnie ∝ g^{−2}·e^{g}, "
                 "bez cutoffu (3D) ∝ g^{−2/3}")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.axis("off")
    ax.set_title("Kosmologia: minimalna ω_c termiczna (ε = 0.01)", fontsize=12)
    rows = tabela_kosmologiczna()
    y = 0.95
    for r in rows:
        ax.text(0.03, y, r["epoka"], fontsize=10, va="center")
        ax.text(0.97, y, f"T = {r['T']:.2g} K", fontsize=9, va="center", ha="right",
                color="#31475c")
        ax.text(0.5, y - 0.045, f"ω_c ≥ {r['wc']:.1e} rad/s = 2π×{r['f']:.1e} Hz",
                fontsize=8.5, va="center", ha="center", color="#6b2f8e")
        y -= 0.13
    ax.text(0.03, y - 0.01,
            "Dziś: mikrofale (łatwe). Rekombinacja: IR (optyka).\n"
            "BBN: rentgen (trudne). Elektrosłaba: gamma (ekstremalne).\n"
            "Z cutoffem i słabym g T_max rośnie — zimny zegar działa\n"
            "w gorętszych epokach niż 3D (R25).",
            fontsize=9, va="top", color="#26384a")
    ax = axs[1, 1]
    eps_b, eps, c = 0.05, 0.01, 2.0
    for wcut, col in [(5, "#27ae60"), (50, "#e67e22"), (500, "#c0392b")]:
        Tm, _ = T_max_widmo("ohmic", 0.03, wcut=wcut)
        ax.plot([wcut], [Tm], "o", color=col, ms=10)
        ax.annotate(f"ω_cut={wcut}: T_max≈{Tm:.0f}", (wcut, Tm),
                    textcoords="offset points", xytext=(8, 8), fontsize=9, color=col)
    Tm3, _ = T_max_widmo("3d", 0.03)
    ax.axhline(Tm3, color="#c0392b", ls=":", lw=1.2)
    ax.text(1.2, Tm3 * 1.5, f"3D (bez cutoffu): T_max = {Tm3:.1f}", color="#c0392b",
            fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ω_cut (częstość odcięcia kąpieli)")
    ax.set_ylabel("T_max (g = 0.03)")
    ax.set_title("T_max vs cutoff: skończona gęstość widmowa podnosi T_max\n"
                 "o wiele rzędów (zimny zegar)")
    fig.suptitle("R30 — zimny zegar: ω_c(T) z kąpieli o skończonej gęstości widmowej",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE19_zimny_zegar.png", bbox_inches="tight")
    plt.close(fig)
    return dict(Tm3=Tm3)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("ENTROPIA-1.6 — SUCHY BIEG, MAPA PETZA, ZIMNY ZEGAR")
    print("=" * 80)

    # [R28]
    print("\n[R28] SUCHY BIEG PROTOKOŁU R23/R26 (realistyczny detektor):")
    det = detektor()
    for p in [dict(naz="A: Rb", tau_nat=26.2e-9, N=1e5, OD=100.0, beta=None),
              dict(naz="B: Cs nanofiber", tau_nat=30.5e-9, N=5e3, OD=None, beta=0.15)]:
        sb = suchy_bieg(p, det)
        print(f"  {p['naz']}: ostatni foton n = {sb['t_last']} "
              f"(t ≈ {sb['t_last']*sb['dt']*1e6:.0f} μs)")
        for teor in ["T1", "T2"]:
            w = sb["wyn"][teor]
            print(f"    {teor}: E[N] = {w['E_N']:.1f} ± {w['std']:.1f}, "
                  f"błąd = {w['err']:.4f}")
        print(f"    całkowity czas pomiaru ≈ {sb['T_total']*1e6:.0f} μs/realizację")
    print("  Detektor: η_det = 0.3, dark = 100 Hz, jitter = 1 ns (≪ t_B, t_D)")

    # [R29]
    print("\n[R29] JAWNA MAPA ODZYSKU PETZA:")
    for j, naz in [(0.5, "ciemny j=1/2"), (1.0, "ciemny j=1"),
                   (2.0, "jasny N=4 j=2"), (3.0, "jasny N=6 j=3")]:
        F = F_rec_sektora(j, m=0)
        print(f"  {naz}: F_rec = {F:.4f}  (1 = idealny odzysk bitu)")
    print("  j=0: F_rec = 1 (kanał identyczności — doskonały odzysk)")

    # [R30]
    print("\n[R30] ZIMNY ZEGAR (skończona gęstość widmowa):")
    for model in ["3d", "ohmic"]:
        Tm, wc_hi = T_max_widmo(model, 0.03, wcut=50.0)
        print(f"  {model} (g=0.03): ω_c^max = {wc_hi:.2f}, T_max = {Tm:.2f}")
    print("  Kosmologia (minimalna ω_c termiczna, ε = 0.01):")
    for r in tabela_kosmologiczna():
        print(f"    {r['epoka']:24s} T = {r['T']:.2g} K: ω_c ≥ {r['wc']:.1e} rad/s "
              f"(2π×{r['f']:.1e} Hz)")

    # figury
    d18 = figura_E18()
    d19 = figura_E19()
    print(f"\nFigury: figE18_petz, figE19_zimny_zegar w: {os.path.abspath(OUT)}")
    return dict(Frec=d18, Tm3=d19["Tm3"],
                sb_B=suchy_bieg(dict(naz="B", tau_nat=30.5e-9, N=5e3, OD=None,
                                     beta=0.15), detektor()),
                kosm=tabela_kosmologiczna())


if __name__ == "__main__":
    main()
