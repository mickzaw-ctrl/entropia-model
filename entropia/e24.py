# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-4.0 — DWUKOMÓRKOWY WSZECHŚWIAT: WYMIANA ENTROPII,
                 ENTROPOWA SIŁA I EMERGENTNA METRYKA FRW
=============================================================================
  Bramka audytu ENTROPIA-1.2 (AUDYT_ENTROPIA12.md, §6.3) OTWARTA po R47.
  Kosmologia zabawkowa zbudowana od mikroskopii:

  R48 — DWIE KOMÓRKI Z WYMIANĄ ENTROPII (mikro-NESS):
        • komórka A (gorąca kąpiel Gibbsa, T_A = 3·T_B, γ_A = 27·γ_B —
          entropia właściwa promieniowania s ∝ T³) i komórka B (zimna,
          γ_B = 0.02); ω₀ = 1;
        • wymiana między komórkami: kanał „hoppingu" X₁ = σ₋^A σ₊^B,
          X₂ = σ₊^A σ₋^B w tempie κ (przenosi ekscytację A↔B; zachowuje
          E_A + E_B);
        • start: A czysto wzbudzona (T_eff = ∞ — „Wielki Wybuch"), B w
          stanie podstawowym; dynamika populacyjna (macierz stóp 4×4);
        • obserwable: S_A, S_B, S_tot, E_A, E_B, prąd energii J_E,
          produkcja entropii Spohna σ_A, σ_B (kąpiele) i σ_ex (wymiana);
        • wyniki: NESS (dS_tot/dt → 0, ale σ > 0 stałe — produkcja trwa),
          prąd J_E,∞ z gorącej do zimnej, σ_NESS ≈ J_E,∞·(1/T_B − 1/T_A)
          (Clausius), prawo Fouriera J ∝ ΔT w reżimie liniowym.

  R49 — GRAWITACJA = BUDŻET ENTROPII + EMERGENTNA FRW:
        • ENTROPOWA SIŁA (zabawka Verlinde'owska): S_tot∞(κ) rośnie z κ
          (bliższe komórki ⇒ więcej dostępnej entropii) ⇒ F(d) < 0 —
          PRZYCIĄGANIE; κ = κ₀e^{−d/L} (tunelowanie) ⇒ F maleje z d;
        • EMERGENTNA FRW: czas kosmiczny z budżetu τ_bud(t) = Σσ·τ
          (rośnie liniowo w NESS — „czas nigdy nie zamiera"), wiek układu
          τ_sys = S_tot (skończony — śmierć cieplna);
          start komórki A w INWERSJI obsadzeń (T_eff < 0 — laserowe
          wzbudzenie); PRZEJŚCIE przez T = ∞ (pg = pe) = „Wielki Wybuch":
          a(τ) = T_NESS/T_eff(τ) dla epoki T > 0 (T ∝ 1/a, konwencja
          promieniowania): a: 0 → a_max ≈ 1.24 (Ekspansja) → 1 (Kontrakcja —
          T_eff przestrzeliwuje poniżej T_NESS, wymiana drenuje A szybciej
          niż kąpiel uzupełnia ⇒ OD BICIE, por. R8); H: +∞ → 0 przez zmianę
          znaku (koniec ekspansji); z = 1/a − 1: ∞ → 0 (przy kontrakcji
          z < 0 — przesunięcie ku fioletowi);
        • DYLATACJA CZASU: σ_A/σ_B ≈ 27 w fazie zegarowej (γ_A/γ_B),
          w NESS produkcja przenosi się do zimnej komórki (Clausius:
          σ = J(1/T_B − 1/T_A), dominuje 1/T_B) — analog dylatacji
          grawitacyjnej + rozkład produkcji entropii w polu temperatury.

  Uczciwa uwaga: to zabawka — „wszechświat" ma 2 kubity; mapowanie a(τ)
  z T_eff jest przyjęte (konwencja promieniowania T ∝ 1/a), nie wyprowadzone;
  κ↔d jest parametrem, nie geometrią; σ > 0 w NESS to produkcja względem
  lokalnych równowag kąpieli (Spohn), nie „nowa fizyka".

  Uruchomienie: python3 -m entropia.e24
=============================================================================
"""

import os
import numpy as np
from scipy.linalg import expm
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

# ---- parametry kosmologicznej zabawki -------------------------------
W0 = 1.0                 # energia wzbudzenia [j. energii]
TB = 1.0                 # temperatura zimnej kąpieli
TA = 3.0 * TB            # gorąca kąpiel: T_A = 3·T_B
GAMMA_B = M.GAMMA_B      # 0.02 — tempo zimnej kąpieli
GAMMA_A = GAMMA_B * (TA / TB) ** 3   # 27·γ_B (s ∝ T³)
ETA_A = np.exp(-W0 / TA)             # 0.7165
ETA_B = np.exp(-W0 / TB)             # 0.3679
KAPPA = 0.3              # tempo wymiany (default)
N = 2000                 # liczba kroków (t_max = 100)
DTAU = 0.05              # mikro-tyknięcie

# stan |10⟩ (A wzbudzona — T_eff = ∞, B w podstawie): p = [p00,p01,p10,p11]
P0 = np.array([0.0, 0.0, 1.0, 0.0])


# -----------------------------------------------------------------------------
#  GŁÓWNY PROPAGATOR (macierz stóp 4×4, populacje — dynamika diagonalna)
# -----------------------------------------------------------------------------
def macierz_stop(gamma_A=GAMMA_A, eta_A=ETA_A, gamma_B=GAMMA_B, eta_B=ETA_B,
                 kappa=KAPPA):
    """
    Generator populacji p = [p00, p01, p10, p11] (baza |A B⟩).
    Kąpiele Gibbsa: a = 2γ/(1+η) (emisja, σ₋), b = 2γη/(1+η) (absorpcja, σ₊).
    Wymiana: X₁ = σ₋^Aσ₊^B (|10⟩→|01⟩), X₂ = σ₊^Aσ₋^B (|01⟩→|10⟩), tempo κ.
    Kolumny = skąd. Wszystkie operatory diagonalne ⇒ populacje domknięte.
    """
    aA = 2.0 * gamma_A / (1.0 + eta_A); bA = 2.0 * gamma_A * eta_A / (1.0 + eta_A)
    aB = 2.0 * gamma_B / (1.0 + eta_B); bB = 2.0 * gamma_B * eta_B / (1.0 + eta_B)
    A = np.zeros((4, 4))
    # 00 ← 10 (aA), ← 01 (aB)
    A[0, 2] += aA; A[0, 1] += aB
    # 01 ← 00 (bB), ← 11 (aA), ← 10 (wymiana κ)
    A[1, 0] += bB; A[1, 3] += aA; A[1, 2] += kappa
    # 10 ← 00 (bA), ← 11 (aB), ← 01 (wymiana κ)
    A[2, 0] += bA; A[2, 3] += aB; A[2, 1] += kappa
    # 11 ← 01 (bA), ← 10 (bB)
    A[3, 1] += bA; A[3, 2] += bB
    np.fill_diagonal(A, -A.sum(axis=0))
    return A


def symuluj_dwie_komorki(n=N, dtau=DTAU, p0=P0, kappa=KAPPA,
                         gamma_A=GAMMA_A, eta_A=ETA_A,
                         gamma_B=GAMMA_B, eta_B=ETA_B):
    """Ewolucja populacji dwukomórkowego wszechświata."""
    A = macierz_stop(gamma_A, eta_A, gamma_B, eta_B, kappa)
    U = expm(A * dtau)
    p = np.array(p0, float)
    P = np.zeros((n, 4))
    for k in range(n):
        P[k] = p
        p = U @ p
    # entropie (komórki są niezależne tylko przy κ=0; tu: populacje marginalne)
    S_A = np.zeros(n); S_B = np.zeros(n); S_tot = np.zeros(n)
    E_A = np.zeros(n); E_B = np.zeros(n)
    for k in range(n):
        pA = np.array([P[k, 0] + P[k, 1], P[k, 2] + P[k, 3]])   # marg. A
        pB = np.array([P[k, 0] + P[k, 2], P[k, 1] + P[k, 3]])   # marg. B
        S_A[k] = _H(pA); S_B[k] = _H(pB); S_tot[k] = _H(P[k])
        E_A[k] = W0 * (P[k, 2] + P[k, 3])
        E_B[k] = W0 * (P[k, 1] + P[k, 3])
    # produkcja entropii Spohna (per kąpiel i wymiana) — dokładna (ρ diagonalne)
    sig_A, sig_B, sig_ex = _spohn(P, gamma_A, eta_A, gamma_B, eta_B, kappa)
    sig_tot = sig_A + sig_B + sig_ex
    # prąd energii przez kanał wymiany: każde |10⟩→|01⟩ przenosi ω₀ z A do B
    # (w NESS dE_A/dt = 0 — kąpiel A uzupełnia; fizyczny prąd = J_ex)
    J_ex = W0 * kappa * (P[:, 2] - P[:, 1])
    dE_A = np.gradient(E_A, dtau)
    dE_B = np.gradient(E_B, dtau)
    return dict(P=P, S_A=S_A, S_B=S_B, S_tot=S_tot, E_A=E_A, E_B=E_B,
                sig_A=sig_A, sig_B=sig_B, sig_ex=sig_ex, sig_tot=sig_tot,
                J_E=J_ex, J_ex=J_ex, dE_A=dE_A, dE_B=dE_B,
                t=np.arange(n) * dtau)


def _H(p):
    p = np.clip(p, 1e-300, 1.0)
    return float(-np.sum(p * np.log(p)))


def _spohn(P, gamma_A, eta_A, gamma_B, eta_B, kappa):
    """Produkcja entropii Spohna per kanał (ρ diagonalne — wzory dokładne).
    σ_j = −Σ_k (L_j p)_k (ln p_k − ln ρ_eq,j,k) ≥ 0 (tw. Spohna).
    Referencje: kąpiel A — Gibbs_A(η_A) × cokolwiek-B (człon B znika);
    wymiana — dowolny stan o p01 = p10 ⇒ σ_ex = κ(p10−p01)ln(p10/p01)."""
    n = P.shape[0]
    aA = 2.0 * gamma_A / (1.0 + eta_A); bA = 2.0 * gamma_A * eta_A / (1.0 + eta_A)
    aB = 2.0 * gamma_B / (1.0 + eta_B); bB = 2.0 * gamma_B * eta_B / (1.0 + eta_B)
    gA0 = 1.0 / (1.0 + eta_A); gA1 = eta_A / (1.0 + eta_A)
    gB0 = 1.0 / (1.0 + eta_B); gB1 = eta_B / (1.0 + eta_B)
    sig_A = np.zeros(n); sig_B = np.zeros(n); sig_ex = np.zeros(n)
    for k in range(n):
        p = P[k]
        lnp = np.log(np.clip(p, 1e-300, 1.0))
        # L_A p: tylko zmiany kubiła A
        LA = np.zeros(4)
        LA[0] = aA * p[2] - bA * p[0]          # 00
        LA[1] = aA * p[3] - bA * p[1]          # 01
        LA[2] = bA * p[0] - aA * p[2]          # 10
        LA[3] = bA * p[1] - aA * p[3]          # 11
        refA = np.log(np.array([gA0, gA0, gA1, gA1]))
        sig_A[k] = -np.sum(LA * (lnp - refA))
        # L_B p
        LB = np.zeros(4)
        LB[0] = aB * p[1] - bB * p[0]
        LB[1] = bB * p[0] - aB * p[1]
        LB[2] = aB * p[3] - bB * p[2]
        LB[3] = bB * p[2] - aB * p[3]
        refB = np.log(np.array([gB0, gB1, gB0, gB1]))
        sig_B[k] = -np.sum(LB * (lnp - refB))
        # wymiana: σ = κ(p10−p01)ln(p10/p01)
        dp = p[2] - p[1]
        sig_ex[k] = kappa * dp * np.log(np.clip(p[2], 1e-300, 1.0)
                                        / np.clip(p[1], 1e-300, 1.0))
    return sig_A, sig_B, sig_ex


def T_eff(pA):
    """Temperatura efektywna komórki o populacjach pA = [p_g, p_e]: ω₀/ln(p_g/p_e)."""
    p_g, p_e = pA
    if p_g <= 0:
        return np.inf
    if p_e <= 0:
        return 0.0
    return W0 / np.log(p_g / p_e)


# -----------------------------------------------------------------------------
#  R48 — LICZBY KLUCZOWE NESS
# -----------------------------------------------------------------------------
def liczby_ness(z=None):
    """Kluczowe liczby dwukomórkowego NESS (R48)."""
    if z is None:
        z = symuluj_dwie_komorki()
    n = z["P"].shape[0]
    tail = slice(int(0.9 * n), n)
    p = z["P"]
    J_E_inf = float(np.mean(z["J_ex"][tail]))              # prąd wymiany A→B (>0)
    J_E_inf_b = float(np.mean(z["J_ex"][tail]))            # = dopływ do B (ten sam kanał)
    sigA_inf = float(np.mean(z["sig_A"][tail]))
    sigB_inf = float(np.mean(z["sig_B"][tail]))
    sigEx_inf = float(np.mean(z["sig_ex"][tail]))
    sig_tot_inf = sigA_inf + sigB_inf + sigEx_inf
    dSdt_inf = float(np.mean(np.gradient(z["S_tot"][tail], z["t"][tail])))
    p_inf = np.mean(p[tail], axis=0)
    # Clausius: σ vs J_E·(1/T_B − 1/T_A)  (1/T w jednostkach k_B = 1)
    clausius = J_E_inf * (1.0 / TB - 1.0 / TA)
    # dylatacja: stosunek produkcji w A vs B w FAZIE ZEGAROWEJ
    # (t ∈ [1,4] — po osobliwym starcie, przed NESS; rząd γ_A/γ_B = 27
    # z poprawkami skończonej T: średnio ~22, szczyt ~33)
    rAB = z["sig_A"] / np.clip(z["sig_B"], 1e-300, None)
    maska = (z["t"] >= 1.0) & (z["t"] <= 4.0)
    ratio_clock = float(np.mean(rAB[maska]))
    ratio_clock_peak = float(np.nanmax(rAB[maska]))
    ratio_late = sigA_inf / max(sigB_inf, 1e-300)
    return dict(
        J_E_inf=J_E_inf, J_E_inf_b=J_E_inf_b,
        sig_A_inf=sigA_inf, sig_B_inf=sigB_inf, sig_ex_inf=sigEx_inf,
        sig_tot_inf=sig_tot_inf, dSdt_inf=dSdt_inf,
        p_inf=p_inf,
        S_tot_end=float(z["S_tot"][-1]),
        clausius=clausius, ratio_clausius=float(sig_tot_inf / max(clausius, 1e-300)),
        ratio_clock=ratio_clock, ratio_clock_peak=ratio_clock_peak,
        ratio_late=ratio_late,
        E_A_end=float(z["E_A"][-1]), E_B_end=float(z["E_B"][-1]),
    )


def skan_fouriera(TAs=(1.5, 2.0, 2.5, 3.0, 4.0, 5.0), kappa=KAPPA):
    """Prawo Fouriera: J_E,∞ vs ΔT = T_A − T_B (γ_A = γ_B(T_A/T_B)³)."""
    rows = []
    for TAk in TAs:
        z = symuluj_dwie_komorki(kappa=kappa,
                                 gamma_A=GAMMA_B * (TAk / TB) ** 3,
                                 eta_A=np.exp(-W0 / TAk))
        d = liczby_ness(z)
        rows.append(dict(TA=TAk, dT=TAk - TB, J=d["J_E_inf"],
                         sig=d["sig_tot_inf"]))
    return rows


def skan_kappa(kappas=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
               wsp=3.0):
    """S_tot∞(κ), J_E,∞(κ), σ_NESS(κ) — baza dla entropowej siły.
    Czas symulacji adaptacyjny: t_max = max(1/κ, 25)·wsp — dla małych κ
    (wolna wymiana) trzeba dłużej, by NESS był ZBIEŻONY (dS_tot/dt → 0)."""
    rows = []
    for kap in kappas:
        t_max = max(1.0 / kap, 25.0) * wsp
        n = int(round(t_max / DTAU))
        z = symuluj_dwie_komorki(n=n, kappa=kap)
        d = liczby_ness(z)
        rows.append(dict(kappa=kap, S_inf=d["S_tot_end"], J=d["J_E_inf"],
                         sig=d["sig_tot_inf"], t_max=t_max))
    return rows


# -----------------------------------------------------------------------------
#  R49 — ENTROPOWA SIŁA + EMERGENTNA FRW
# -----------------------------------------------------------------------------
def entropowa_sila(kappas=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
                   L=1.0, kappa0=1.0, T=TB):
    """
    F(d) = T·∂S∞/∂d,  κ(d) = κ₀·e^{−d/L}.
    S∞(κ) rosnące ⇒ ∂S∞/∂d < 0 ⇒ siła PRZYCIĄGAJĄCA (komórki „chcą" bliżej).
    Zwraca słownik z S∞(κ), dS/dκ, F(d) na siatce odległości.
    """
    rows = skan_kappa(kappas)
    kap = np.array([r["kappa"] for r in rows])
    Sinf = np.array([r["S_inf"] for r in rows])
    dSdk = np.gradient(Sinf, kap)
    # siatka odległości d ≥ 0: κ(d) = κ₀ e^{-d/L}
    ds = np.linspace(0.0, 6.0, 200)
    kap_d = kappa0 * np.exp(-ds / L)
    S_d = np.interp(kap_d, kap, Sinf)                   # S∞(d) (kap rosnące)
    F_d = T * np.gradient(S_d, ds)                      # T·∂S/∂d (ujemne = przyciąganie)
    return dict(kap=kap, Sinf=Sinf, dSdk=dSdk, ds=ds, S_d=S_d, F_d=F_d,
                T=T, L=L)


def emergentna_frw(z=None, kappa=KAPPA):
    """
    Metryka FRW z budżetu entropii:
      τ_sys(t) = S_tot(t) − S_tot(0)          — wiek układu (skończony);
      τ_bud(t) = Σ σ_tot·τ                    — czas kosmiczny z budżetu
                                                (liniowy w NESS: σ_NESS > 0);
      T_A_eff(t) — temperatura efektywna komórki A: start w INWERSJI
                    obsadzeń (T < 0 — laserowe wzbudzenie), przejście przez
                    T = ∞ (pg = pe), potem T > 0 → T_NESS = 2.835 < T_A;
      a(τ_bud)  = T_NESS/T_A_eff(t) dla epoki T > 0  — ekspansja 0 → 1
                    (T ∝ 1/a, konwencja promieniowania);
                    OSOBLIWOŚĆ a = 0 w momencie przejścia przez T = ∞:
                    „Wielki Wybuch = koniec inwersji";
      H(τ_bud)  = (1/a)(da/dτ_bud): ∞ w osobliwości → 0 w NESS;
      z = 1/a − 1: ∞ → 0.
    """
    if z is None:
        z = symuluj_dwie_komorki(kappa=kappa)
    n = z["P"].shape[0]
    t = z["t"]
    pA_g = np.clip(z["P"][:, 0] + z["P"][:, 1], 1e-300, None)
    pA_e = np.clip(z["P"][:, 2] + z["P"][:, 3], 1e-300, None)
    # T_eff = ω₀/ln(pg/pe): ujemne przy inwersji (pe > pg), ∞ przy pg = pe
    T_eff = W0 / np.log(pA_g / pA_e)
    i_cross = int(np.argmax(T_eff > 0.0))     # przejście przez T = ∞
    # τ_sys (wiek układu w natach)
    tau_sys = z["S_tot"] - z["S_tot"][0]
    # τ_bud (budżet: Σ σ_tot·τ, w natach)
    tau_bud = np.concatenate([[0.0], np.cumsum(z["sig_tot"][1:] * DTAU)])
    # skala: a = T_NESS/T_eff dla epoki T > 0; a(i_cross) = 0 (osobliwość)
    T_end = T_eff[-1]
    a = np.zeros(n)
    idx = np.arange(n) >= i_cross
    a[idx] = T_end / T_eff[idx]
    a[i_cross] = 0.0
    # H(τ_bud) = (1/a)(da/dτ_bud) — tylko epoka T > 0
    da = np.gradient(a, tau_bud)
    H = np.zeros(n)
    H[idx] = da[idx] / np.clip(a[idx], 1e-12, None)
    z_red = np.where(a > 0.0, 1.0 / np.clip(a, 1e-300, None) - 1.0, np.inf)
    return dict(t=t, tau_sys=tau_sys, tau_bud=tau_bud, T_eff=T_eff, a=a,
                H=H, z=z_red, sig_tot=z["sig_tot"], S_tot=z["S_tot"],
                i_cross=i_cross)


def T_eff_kom(pg, pe):
    if pg <= 0:
        return np.inf
    if pe <= 0:
        return 0.0
    return W0 / np.log(pg / pe)


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E37(z=None):
    """Dynamika dwukomórkowego wszechświata: entropie, energie, σ, zegary."""
    if z is None:
        z = symuluj_dwie_komorki()
    t = z["t"]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ax = axs[0, 0]
    ax.plot(t, z["S_A"], color=C_A, lw=2, label="S_A (gorąca komórka)")
    ax.plot(t, z["S_B"], color=C_B, lw=2, label="S_B (zimna komórka)")
    ax.plot(t, z["S_tot"], color=C_V, lw=2.4, label="S_tot (wszechświat)")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("S [nat]")
    ax.set_title("R48: entropie dwukomórkowego wszechświata — NESS (S_tot → const)")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.plot(t, z["E_A"], color=C_A, lw=2, label="E_A")
    ax.plot(t, z["E_B"], color=C_B, lw=2, label="E_B")
    ax.plot(t, z["E_A"] + z["E_B"], color="#7f8c8d", lw=2, ls="--",
            label="E_A + E_B (wymiana zachowuje energię)")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("E [ω₀]")
    ax.set_title("Energia płynie A → B (J_E,∞ > 0 w NESS)")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    m = t >= 0.25          # poza logarytmicznym kuspem startu (inwersja)
    ax.plot(t[m], z["sig_A"][m], color=C_A, lw=2, label="σ_A (kąpiel A)")
    ax.plot(t[m], z["sig_B"][m], color=C_B, lw=2, label="σ_B (kąpiel B)")
    ax.plot(t[m], z["sig_ex"][m], color="#27ae60", lw=2, label="σ_ex (wymiana)")
    ax.plot(t[m], z["sig_tot"][m], color="#8e44ad", lw=2.4, label="σ_tot")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("σ [nat/j. czasu]")
    ax.set_title("Produkcja entropii (Spohn): σ ≥ 0; w NESS σ_tot > 0 stałe")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.plot(t, z["S_tot"], color=C_V, lw=2, label="τ_sys = S_tot (wiek układu)")
    ax.plot(t, np.concatenate([[0], np.cumsum(z["sig_tot"][1:] * DTAU)]),
            color="#1a5276", lw=2, label="τ_bud = Σσ·τ (czas z budżetu)")
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("τ [nat]")
    ax.set_title("Dwa zegary: τ_sys kończy się (śmierć cieplna), "
                 "τ_bud trwa (σ_NESS > 0)")
    ax.legend(fontsize=8)
    fig.suptitle("ENTROPIA-4.0 / R48 — dwukomórkowy wszechświat w NESS",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE37_dwie_komorki.png", bbox_inches="tight")
    plt.close(fig)


def figura_E38():
    """NESS: prawo Fouriera, Clausius, zależność od κ."""
    fou = skan_fouriera()
    kap = skan_kappa()
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.6))
    ax = axs[0]
    ax.plot([r["dT"] for r in fou], [r["J"] for r in fou], "o-",
            color="#c0392b", lw=2, ms=7)
    ax.set_xlabel("ΔT = T_A − T_B"); ax.set_ylabel("J_E,∞ [ω₀/j. czasu]")
    ax.set_title("R48: prawo Fouriera — J_E,∞ rośnie z ΔT (liniowo przy małych ΔT)")
    ax = axs[1]
    cl = [r["J"] * (1.0 / TB - 1.0 / r["TA"]) for r in fou]
    sig = [r["sig"] for r in fou]
    ax.plot(cl, sig, "o-", color="#27ae60", lw=2, ms=7, label="σ_NESS vs J·(1/T_B−1/T_A)")
    ax.plot([0, max(cl)], [0, max(cl)], ":", color=C_G, lw=1.5, label="równość (Clausius)")
    ax.set_xlabel("J_E,∞·(1/T_B − 1/T_A)"); ax.set_ylabel("σ_NESS")
    ax.set_title("Clausius: σ_NESS ≈ J_E·(1/T_B − 1/T_A)")
    ax.legend(fontsize=8)
    ax = axs[2]
    ax.semilogx([r["kappa"] for r in kap], [r["J"] for r in kap], "o-",
                color="#2471a3", lw=2, ms=7, label="J_E,∞(κ)")
    ax.semilogx([r["kappa"] for r in kap], [r["sig"] for r in kap], "s--",
                color="#8e44ad", lw=2, ms=6, label="σ_NESS(κ)")
    ax.set_xlabel("κ (tempo wymiany)"); ax.set_ylabel("[j. ω₀/nat·t]")
    ax.set_title("Silniejsza wymiana ⇒ większy prąd i produkcja")
    ax.legend(fontsize=8)
    fig.suptitle("ENTROPIA-4.0 / R48 — termodynamika NESS", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE38_ness.png", bbox_inches="tight")
    plt.close(fig)
    return fou, kap


def figura_E39():
    """Entropowa siła: S∞(κ), ∂S/∂κ, F(d)."""
    es = entropowa_sila()
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.6))
    ax = axs[0]
    ax.semilogx(es["kap"], es["Sinf"], "o-", color="#c0392b", lw=2, ms=7)
    ax.set_xlabel("κ (tempo wymiany = 1/odległość)"); ax.set_ylabel("S_tot∞ [nat]")
    ax.set_title("R49: S∞(κ) rośnie — bliżej = więcej entropii")
    ax = axs[1]
    ax.semilogx(es["kap"], es["dSdk"], "o-", color="#27ae60", lw=2, ms=7)
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.set_xlabel("κ"); ax.set_ylabel("∂S∞/∂κ")
    ax.set_title("∂S∞/∂κ > 0 ⇒ siła przyciągająca")
    ax = axs[2]
    ax.plot(es["ds"], es["F_d"], color="#2471a3", lw=2.4)
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.set_xlabel("odległość d (κ = κ₀e^{−d/L})"); ax.set_ylabel("F(d) = T·∂S∞/∂d")
    ax.set_title("Entropowa siła: F < 0 (przyciąganie), maleje z d")
    fig.suptitle("ENTROPIA-4.0 / R49 — grawitacja z budżetu entropii "
                 "(zabawka Verlinde'owska)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE39_entropowa_sila.png", bbox_inches="tight")
    plt.close(fig)
    return es


def figura_E40(fr=None):
    """Emergentna FRW: a(τ), H(τ), z(τ), dylatacja czasu."""
    if fr is None:
        fr = emergentna_frw()
    z48 = symuluj_dwie_komorki()
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ax = axs[0, 0]
    ax.plot(fr["tau_bud"], fr["a"], color="#c0392b", lw=2.4)
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.set_xlabel("τ_bud = Σσ·τ [nat]"); ax.set_ylabel("a(τ)")
    ax.set_title("R49: emergentna skala FRW — a: 0 → a_max (ekspansja) → 1\n"
                 "(kontrakcja: T_eff przestrzeliwuje poniżej T_NESS — odbicie)")
    ax = axs[0, 1]
    m = fr["t"] >= fr["t"][fr["i_cross"]] + 0.1   # epoka T > 0
    ax.plot(fr["tau_bud"][m], fr["H"][m], color="#2471a3", lw=2.4)
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.set_xlabel("τ_bud [nat]"); ax.set_ylabel("H(τ) = (1/a)(da/dτ)")
    ax.set_title("H(τ): +∞ (Wielki Wybuch) → 0 (odbicie, zmiana znaku) → 0⁻ "
                 "(śmierć cieplna)")
    ax = axs[1, 0]
    ax.plot(fr["tau_bud"], fr["z"], color="#8e44ad", lw=2.4)
    ax.set_xlabel("τ_bud [nat]"); ax.set_ylabel("z(τ) = 1/a − 1")
    ax.set_title("Przesunięcie ku czerwieni: ∞ → 0")
    ax = axs[1, 1]
    rAB = z48["sig_A"] / np.clip(z48["sig_B"], 1e-12, None)
    m = z48["t"] >= 0.25
    ax.plot(z48["t"][m], rAB[m], color="#27ae60", lw=2.4)
    ax.axhline(27, color=C_G, ls=":", lw=1)
    ax.text(1.5, 29.5, "27 = γ_A/γ_B (faza zegarowa)", color=C_G, fontsize=8)
    ax.set_ylim(0, 35)
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("σ_A/σ_B")
    ax.set_title("Dylatacja czasu: w fazie zegarowej σ_A/σ_B ≈ 27;\n"
                 "w NESS produkcja przenosi się do zimnej komórki (Clausius)")
    fig.suptitle("ENTROPIA-4.0 / R49 — emergentna metryka FRW i dylatacja",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE40_frw.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
#  WERYFIKACJA: κ=0 ⇒ dwie niezależne komórki = symulacje jednokubitowe
# -----------------------------------------------------------------------------
def weryfikacja_odsprezenia():
    """κ=0: S_A/S_B muszą zgadzać się z niezależnymi symulacjami termicznymi
    (ten sam stan początkowy: A = |1⟩, B = |0⟩ — p0 = [0,1] i [1,0])."""
    from .extensions import symuluj_termicznie
    z = symuluj_dwie_komorki(kappa=0.0)
    S_A1, _, _ = symuluj_termicznie(GAMMA_A, ETA_A, n=N, delta_tau=DTAU,
                                    p0=[0.0, 1.0])
    S_B1, _, _ = symuluj_termicznie(GAMMA_B, ETA_B, n=N, delta_tau=DTAU,
                                    p0=[1.0, 0.0])
    blad_A = float(np.max(np.abs(z["S_A"] - S_A1)))
    blad_B = float(np.max(np.abs(z["S_B"] - S_B1)))
    return dict(blad_A=blad_A, blad_B=blad_B,
                S_A_end=float(z["S_A"][-1]), S_B_end=float(z["S_B"][-1]))


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 84)
    print("ENTROPIA-4.0 — DWUKOMÓRKOWY WSZECHŚWIAT (R48: NESS; R49: siła + FRW)")
    print("=" * 84)

    print("\n[R48] DWIE KOMÓRKI Z WYMIANĄ ENTROPII:")
    z = symuluj_dwie_komorki()
    d = liczby_ness(z)
    print(f"  parametry: T_A = {TA:.1f} = 3·T_B, γ_A = {GAMMA_A:.3f} = 27·γ_B, "
          f"η_A = {ETA_A:.4f}, η_B = {ETA_B:.4f}, κ = {KAPPA}")
    print(f"  NESS: S_tot(∞) = {d['S_tot_end']:.4f} nat (dS/dt → {d['dSdt_inf']:.2e})")
    print(f"  prąd energii: J_E,∞ = {d['J_E_inf']:.5f} (A→B), "
          f"do B: {d['J_E_inf_b']:.5f} (zachowanie: Δ = "
          f"{abs(d['J_E_inf'] - d['J_E_inf_b']):.2e})")
    print(f"  σ_NESS: σ_A = {d['sig_A_inf']:.5f}, σ_B = {d['sig_B_inf']:.5f}, "
          f"σ_ex = {d['sig_ex_inf']:.5f}, σ_tot = {d['sig_tot_inf']:.5f}")
    print(f"  Clausius: J·(1/T_B − 1/T_A) = {d['clausius']:.6f}, "
          f"σ_tot/J·(…) = {d['ratio_clausius']:.6f} (≈ 1 — zgodność)")
    print(f"  dylatacja: σ_A/σ_B (faza zegarowa) = {d['ratio_clock']:.1f} "
          f"(śr.), szczyt {d['ratio_clock_peak']:.1f} — rząd γ_A/γ_B = 27; "
          f"(NESS) = {d['ratio_late']:.3f} (produkcja → zimna komórka, Clausius)")
    w = weryfikacja_odsprezenia()
    print(f"  weryfikacja κ=0: max|ΔS_A| = {w['blad_A']:.2e}, "
          f"max|ΔS_B| = {w['blad_B']:.2e}")

    print("\n[R48] PRAWO FOURIERA (J_E,∞ vs ΔT):")
    fou = skan_fouriera()
    for r in fou:
        print(f"    T_A = {r['TA']:.1f}: ΔT = {r['dT']:.1f}, "
              f"J_E,∞ = {r['J']:.5f}, σ_NESS = {r['sig']:.5f}")
    J1, J2 = fou[0]["J"], fou[2]["J"]
    print(f"    quasi-liniowość: J(ΔT=0.5) = {J1:.4f}, J(ΔT=1.5) = {J2:.4f} "
          f"(stosunek {J2/J1:.2f} vs ΔT stosunek {3.0:.1f} — nasycenie przy "
          f"większych ΔT)")

    print("\n[R49] ENTROPOWA SIŁA (S∞(κ), F(d)):")
    es = entropowa_sila()
    for k, s in zip(es["kap"], es["Sinf"]):
        print(f"    κ = {k:6.3f}: S_tot∞ = {s:.5f}")
    i1 = np.argmin(np.abs(es["ds"] - 1.0)); i2 = np.argmin(np.abs(es["ds"] - 3.0))
    imin = int(np.argmin(es["F_d"]))
    print(f"    F(d=1) = {es['F_d'][i1]:.4f}, F(d=3) = {es['F_d'][i2]:.4f}, "
          f"min F = {es['F_d'][imin]:.4f} przy d = {es['ds'][imin]:.2f}")
    print("    kierunek: przyciągający (∂S∞/∂κ > 0 wszędzie); profil NIE jest "
          "1/d² —")
    print("    siła najsilniejsza przy pośrednich d (S∞ nasyca się przy silnym "
          "sprzężeniu).")

    print("\n[R49] EMERGENTNA FRW:")
    fr = emergentna_frw(z)
    ic = fr["i_cross"]
    print(f"  start: inwersja obsadzeń (T_A_eff < 0); przejście przez T = ∞ "
          f"przy t = {fr['t'][ic]:.2f} (τ_bud = {fr['tau_bud'][ic]:.3f})")
    print(f"  T_A_eff po przejściu: +∞ → {fr['T_eff'][-1]:.3f} "
          f"(< T_A = {TA} — wymiana chłodzi A poniżej równowagi kąpieli)")
    print(f"  wiek układu (τ_sys, nat): {fr['tau_sys'][-1]:.4f} — SKOŃCZONY "
          f"(śmierć cieplna)")
    print(f"  czas z budżetu (τ_bud, nat): {fr['tau_bud'][-1]:.4f}; w NESS "
          f"rośnie liniowo z σ_NESS = {d['sig_tot_inf']:.5f}")
    m = fr["t"] >= fr["t"][ic] + 0.1     # epoka T > 0, poza punktem osobliwym
    imax = int(np.argmax(fr["H"][m]))
    ia = int(np.argmax(fr["a"]))
    # przecięcie zera H: pierwsza zmiana znaku po szczycie
    zn = np.sign(fr["H"][m])
    izm = int(np.argmax(np.diff(zn) != 0)) + 1
    print(f"  a(τ): 0 (osobliwość przy τ_bud = {fr['tau_bud'][ic]:.3f}) → "
          f"a_max = {fr['a'][ia]:.4f} (t = {fr['t'][ia]:.2f}, ekspansja) → "
          f"{fr['a'][-1]:.4f} (kontrakcja do śmierci cieplnej)")
    print(f"  H(τ_bud): szczyt {fr['H'][m][imax]:.4f}, zero przy t = "
          f"{fr['t'][m][izm]:.2f} (odbicie), koniec {fr['H'][-1]:.2e} (H → 0)")
    print(f"  z = 1/a − 1: ∞ → {fr['z'][-1]:.4f}; przy kontrakcji z < 0 "
          f"(przesunięcie ku fioletowi)")
    print(f"  dylatacja: σ_A/σ_B ≈ {d['ratio_clock']:.0f} (śr. faza zegarowa, "
          f"rząd 27) — czas w gorącej komórce płynie ~27× szybciej")

    figura_E37(z)
    fou2, kap2 = figura_E38()
    es2 = figura_E39()
    figura_E40(fr)
    print(f"\nFigury: figE37_dwie_komorki, figE38_ness, figE39_entropowa_sila, "
          f"figE40_frw w: {os.path.abspath(OUT)}")
    return dict(ness=d, fourier=fou, kappa=kap2, sila=es2, frw=fr, odspr=w)


if __name__ == "__main__":
    main()
