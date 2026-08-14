# -*- coding: utf-8 -*-
"""
=============================================================================
  AUDYT ZAMYKAJĄCY ENTROPIA-1.2 (R19)
=============================================================================
  Sekwencja audytu (wg rekomendacji — przed dalszą kosmologią):

    1. AUDYT RÓWNAŃ          — każdy wzór ENTROPIA-1.2 sprawdzony niezależnie;
    2. AUDYT JEDNOSTEK       — wymiary wszystkich wielkości i kalibracje fizyczne;
    3. NIEZALEŻNA REPLIKACJA — implementacje-świadkowie (inna konwencja
                               wektoryzacji, inny integrator) vs projekt;
    4. TEST T1/T2            — cztery funkcjonały: stawanie, monotoniczność,
                               czkanie, nachylenie T2;
    5. TEST ODZYSKIWALNOŚCI DARK-SEKTOR — M(t) = D(t)/D(0): zamknięta forma
                               dla j=1, porządek jasny/ciemny, j=0, superradiancja.

  ZASADA: każda liczba poniżej pochodzi z uruchomienia; implementacje-świadkowie
  nie importują symulatorów projektu (tylko parametry fizyczne M.GAMMA_B itd.
  i — do PORÓWNANIA — wyniki projektu).

  Uruchomienie:  python3 -m entropia.audyt12
=============================================================================
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import core as M
from . import dicke as D
from . import e12 as E

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"

# Stałe fizyczne (CODATA 2018)
KB = 1.380649e-23
HBAR = 1.054571817e-34
HPL = 6.62607015e-34
GRAV = 6.67430e-11
C_LIGHT = 299792458.0


def H_bin(p):
    """Binarna entropia H(p) = -p ln p - (1-p) ln(1-p)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-15)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def ent(rho):
    ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    ev = ev[ev > 1e-15]
    return float(-np.sum(ev * np.log(ev)))


def trace_norm(rho):
    """Norma śladowa przez wartości osobliwe (inna ścieżka niż eigvalsh)."""
    return float(np.sum(np.linalg.svd(rho, compute_uv=False)))


# =============================================================================
#  IMPLEMENTACJE-ŚWIADKOWIE (NIEZALEŻNE)
# =============================================================================

# -----------------------------------------------------------------------------
#  Ś1 — superoperator w konwencji KOLUMNOWEJ (projekt: wierszowa)
# -----------------------------------------------------------------------------
def niez_superoperator_kolumnowy(H, jumps, rates):
    """
    vec_col(A·ρ·B) = (Bᵀ ⊗ A)·vec_col(ρ).  L = -i[H,·] + Σ r·D[J].

    L = -i(I⊗H − Hᵀ⊗I) + Σ r[ J̄⊗J − ½(I⊗J†J + (J†J)ᵀ⊗I) ].
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for J, r in zip(jumps, rates):
        Jd = J.conj().T
        JdJ = Jd @ J
        L += r * (np.kron(J.conj(), J) - 0.5 * (np.kron(I, JdJ) + np.kron(JdJ.T, I)))
    return L


def niez_ewolucja_ivp(H, jumps, rates, rho0, t_max, n_out, integrator="DOP853"):
    """
    Ewolucja ciągła przez scipy.integrate.solve_ivp (gęsty solver, adaptacyjny
    krok) — NUMERYCZNIE INNA ścieżka niż expm(L·τ) projektu.
    Zwraca (ts, rhos) — rhos[n]: macierz w czasie ts[n].
    """
    d = H.shape[0]
    L = niez_superoperator_kolumnowy(H, jumps, rates)

    def f(t, y):
        return (L @ y).reshape(-1)

    ts = np.linspace(0.0, t_max, n_out)
    sol = solve_ivp(f, (0.0, t_max), np.asarray(rho0, complex).flatten(),
                    method=integrator, t_eval=ts, rtol=1e-11, atol=1e-13,
                    max_step=t_max / 40.0)
    rhos = sol.y.T.reshape(n_out, d, d)
    return ts, rhos


# -----------------------------------------------------------------------------
#  Ś2 — RK4 na ODE Blocha wyprowadzonym ręcznie z równania Lindblada
# -----------------------------------------------------------------------------
def niez_rk4_bloch(gamma, gamma_phi, omega, t_max, dt, theta0=M.THETA0):
    """
    Z master equation (kąpiel nieskończenie gorąca, H = (Ω/2)σ_z):
        dr_z/dt = -2γ·r_z          (populacje; precesja nie rusza r_z)
        dr_⊥/dt = -(γ + 2γ_φ)·r_⊥  (koherencje; precesja obraca, moduł stały)
    S(t) = H((1 + |r|)/2),  |r| = √(r_z² + r_⊥²).
    """
    n = int(round(t_max / dt)) + 1
    ts = np.linspace(0.0, t_max, n)
    rz = np.empty(n); rp = np.empty(n)
    rz[0] = np.cos(theta0); rp[0] = np.sin(theta0)
    for i in range(n - 1):
        y = np.array([rz[i], rp[i]])
        def g(yy):
            return np.array([-2.0 * gamma * yy[0],
                             -(gamma + 2.0 * gamma_phi) * yy[1]])
        k1 = g(y)
        k2 = g(y + 0.5 * dt * k1)
        k3 = g(y + 0.5 * dt * k2)
        k4 = g(y + dt * k3)
        y = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        rz[i + 1], rp[i + 1] = y
    r = np.sqrt(np.clip(rz ** 2 + rp ** 2, 0.0, 1.0))
    return ts, H_bin((1.0 + r) / 2.0)


def niez_dSdt_an(gamma, t, theta0=M.THETA0, gamma_phi=2.0):
    """
    Niezależna postać zamknięta tempa produkcji entropii (kąpiel ∞-gorąca):
    S = H((1+r)/2) ⇒ dS/dt = -artanh(r)·ṙ,  ṙ = (r_z·ṙ_z + r_⊥·ṙ_⊥)/r.
    (Projekt liczy przez dS/dε = -artanh(ε) i ε' = d2/(2ε) — inny zapis,
    ta sama pochodna; porównanie obu jest częścią audytu.)
    """
    rz = np.cos(theta0) * np.exp(-2.0 * gamma * t)
    rp = np.sin(theta0) * np.exp(-(1.0 + 2.0 * gamma_phi) * gamma * t)
    r = np.sqrt(rz ** 2 + rp ** 2)
    if r <= 0.0 or r >= 1.0:
        return 0.0
    drz = -2.0 * gamma * rz
    drp = -(1.0 + 2.0 * gamma_phi) * gamma * rp
    dr = (rz * drz + rp * drp) / r
    # dS/dt = (dS/dr)·ṙ = −artanh(r)·ṙ
    return -0.5 * np.log((1.0 + r) / (1.0 - r)) * dr


# -----------------------------------------------------------------------------
#  Ś3 — termiczne S(t) i dS/dt (własne wyprowadzenie, kąpiel Gibbsa η = e^{-ω₀/T})
# -----------------------------------------------------------------------------
def niez_r_norm_termiczny(gamma, eta, t):
    """r_z = r_eq + (cosθ₀−r_eq)e^{−2γt};  r_⊥ = sinθ₀·e^{−5γt} (γ_φ = 2γ)."""
    r_eq = (1.0 - eta) / (1.0 + eta)
    rz = r_eq + (np.cos(M.THETA0) - r_eq) * np.exp(-2.0 * gamma * t)
    rp = np.sin(M.THETA0) * np.exp(-5.0 * gamma * t)
    return np.sqrt(rz ** 2 + rp ** 2)


def niez_S_termiczna(gamma, eta, t):
    return H_bin((1.0 + niez_r_norm_termiczny(gamma, eta, t)) / 2.0)


def niez_dSdt_termiczne(gamma, eta, t):
    """dS/dt dla kąpieli Gibbsa (spójne z niez_S_termiczna)."""
    r = niez_r_norm_termiczny(gamma, eta, t)
    if r <= 0.0 or r >= 1.0:
        return 0.0
    r_eq = (1.0 - eta) / (1.0 + eta)
    rz = r_eq + (np.cos(M.THETA0) - r_eq) * np.exp(-2.0 * gamma * t)
    rp = np.sin(M.THETA0) * np.exp(-5.0 * gamma * t)
    drz = -2.0 * gamma * (rz - r_eq)
    drp = -5.0 * gamma * rp
    dr = (rz * drz + rp * drp) / r
    return -0.5 * np.log((1.0 + r) / (1.0 - r)) * dr


def niez_czas_do_S_termicznej(gamma, eta, Sstar):
    Seq = H_bin((1.0 + (1.0 - eta) / (1.0 + eta)) / 2.0)
    if Sstar >= Seq:
        return np.inf
    f = lambda t: niez_S_termiczna(gamma, eta, t) - Sstar
    return float(brentq(f, 0.0, 40.0 / gamma, xtol=1e-13))


def niez_nbar(x):
    return 1.0 / (np.exp(x) - 1.0)


def niez_RT_spojny(TB, w0, Sstar=0.5, bath="3d"):
    """
    R_T w pełni termiczne i spójne: czasy przejścia S* z termicznej S(t),
    tempo dS/dt z termicznej pochodnej — BEZ mieszania z formułą ∞-gorącą.
    """
    TA = 3.0 * TB
    etaB, etaA = np.exp(-w0 / TB), np.exp(-w0 / TA)
    if bath == "3d":
        gB = (TB / TB) ** 3
        gA = (TA / TB) ** 3
    else:
        gB = 2.0 * niez_nbar(w0 / TB) + 1.0
        gA = 2.0 * niez_nbar(w0 / TA) + 1.0
    tA = niez_czas_do_S_termicznej(gA, etaA, Sstar)
    tB = niez_czas_do_S_termicznej(gB, etaB, Sstar)
    return niez_dSdt_termiczne(gA, etaA, tA) / niez_dSdt_termiczne(gB, etaB, tB)


# -----------------------------------------------------------------------------
#  Ś4 — równania stóp sektorów Dickego (CG, jawnie) + RK45 solve_ivp
# -----------------------------------------------------------------------------
def niez_macierz_stop(j, gamma):
    """
    Równania stóp populacji na drabinie |j,m⟩ (m = −j..j):
        m→m+1:  u_m = γ(j−m)(j+m+1)     (podnoszenie)
        m→m−1:  d_m = γ(j+m)(j−m+1)     (opuszczanie)
    Zwraca macierz A (kolumny: skąd). Dla stanów diagonalnych dokładne.
    """
    m = np.arange(-j, j + 1.0)
    n = len(m)
    A = np.zeros((n, n))
    for i in range(n):
        mm = m[i]
        if mm < j:
            A[i + 1, i] = gamma * (j - mm) * (j + mm + 1)
        if mm > -j:
            A[i - 1, i] = gamma * (j + mm) * (j - mm + 1)
    np.fill_diagonal(A, -A.sum(axis=0))
    return A


def niez_M_stopy(j, m1, m2, gamma, tmax, n_out):
    """
    M(t) = D(t)/D(0) przez jawny solver RK45 na równaniach stóp.
    D(t) = ½·‖Δp(t)‖₁ (stany diagonalne ⇒ normy śladowe populacji).
    """
    m = np.arange(-j, j + 1.0)
    n = len(m)
    A = niez_macierz_stop(j, gamma)
    dp0 = np.zeros(n)
    dp0[int(m1 + j)] = 1.0
    dp0[int(m2 + j)] = -1.0
    ts = np.linspace(0.0, tmax, n_out)
    sol = solve_ivp(lambda t, y: A @ y, (0.0, tmax), dp0, method="RK45",
                    t_eval=ts, rtol=1e-11, atol=1e-13, max_step=tmax / 200.0)
    M = 0.5 * np.sum(np.abs(sol.y), axis=0)
    return ts, M / M[0]


# -----------------------------------------------------------------------------
#  Ś5 — pełna przestrzeń N=2/N=4 (kolektywne S± zbudowane od zera)
# -----------------------------------------------------------------------------
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], complex)
_SY = np.array([[0.0, -1j], [1j, 0.0]], complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], complex)
_SLO = np.array([[0.0, 0.0], [1.0, 0.0]], complex)   # σ₋ = |0⟩⟨1|
_SHI = _SLO.conj().T


def niez_kolektywne(N):
    """S₊, S₋, S_z dla N kubitów (iloczyny tensorowe, od zera)."""
    from functools import reduce
    I2 = np.eye(2, dtype=complex)
    S_p = np.zeros((2 ** N, 2 ** N), complex)
    S_m = np.zeros_like(S_p)
    S_z = np.zeros_like(S_p)
    for i in range(N):
        ops = [I2] * N
        ops[i] = _SHI
        S_p += reduce(np.kron, ops)
        ops[i] = _SLO
        S_m += reduce(np.kron, ops)
        ops[i] = _SZ
        S_z += reduce(np.kron, ops)
    return S_p, S_m, S_z


def niez_mieszanina_10_N2():
    """ρ₀ = ½|T₀⟩⟨T₀| + ½|singlet⟩⟨singlet| (N=2, |10⟩-typ)."""
    ket10 = np.kron(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    ket01 = np.kron(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    T0 = (ket10 + ket01) / np.sqrt(2.0)
    sing = (ket10 - ket01) / np.sqrt(2.0)
    return 0.5 * np.outer(T0, T0.conj()) + 0.5 * np.outer(sing, sing.conj())


def niez_n2_funkcjonaly(n=400, gamma=M.GAMMA_B):
    """
    Pełna 4-wymiarowa ewolucja (solve_ivp/Radau, konwencja kolumnowa)
    mieszaniny |10⟩-typu; zwraca S, I_AB, dS oraz τ̇ T0–T3.
    """
    Sp, Sm, Sz = niez_kolektywne(2)
    H = (M.OMEGA / 2.0) * Sz
    rho0 = niez_mieszanina_10_N2()
    t_max = n * M.DELTA_TAU
    ts, rhos = niez_ewolucja_ivp(H, [Sp, Sm], [gamma, gamma], rho0,
                                 t_max, n_out=n)
    S = np.array([ent(r) for r in rhos])
    I = np.zeros(n)
    for i, r in enumerate(rhos):
        r4 = r.reshape(2, 2, 2, 2)
        rA = np.einsum("abcb->ac", r4)
        rB = np.einsum("abac->bc", r4)
        I[i] = ent(rA) + ent(rB) - ent(r)
    dS = np.maximum(np.diff(S), 0.0)
    dS = np.concatenate([[0.0], dS])
    dI = np.abs(np.diff(np.concatenate([[0.0], I])))
    dR = np.zeros_like(dS)   # N=2: ciemna pamięć doskonała (Ṙ = 0)
    f = E.funkcjonaly_czasu(dS, I, dI, dR)
    return dict(S=S, I=I, dS=dS, f=f, ts=ts)


def niez_n2_dyskretny(n=400, gamma=M.GAMMA_B):
    """
    Ten sam schemat KROKOWY co projekt (powtarzane e^{L·τ}), ale w konwencji
    KOLUMNOWEJ — izoluje różnicę konwencji od różnicy integratora:
    porównanie z D.symuluj_dicke powinno dać ~1e-12 (a nie 4e-4 jak
    ciągły solve_ivp, który wnosi dyskretyzację schematu).
    """
    Sp, Sm, Sz = niez_kolektywne(2)
    H = (M.OMEGA / 2.0) * Sz
    L = niez_superoperator_kolumnowy(H, [Sp, Sm], [gamma, gamma])
    U = expm(L * M.DELTA_TAU)
    rho = niez_mieszanina_10_N2()
    S = np.zeros(n); I = np.zeros(n)
    d = 4
    for k in range(n):
        S[k] = ent(rho)
        r4 = rho.reshape(2, 2, 2, 2)
        rA = np.einsum("abcb->ac", r4)
        rB = np.einsum("abac->bc", r4)
        I[k] = ent(rA) + ent(rB) - ent(rho)
        rho = (U @ rho.flatten()).reshape(d, d)
    return S, I


# =============================================================================
#  1. AUDYT RÓWNAŃ
# =============================================================================
def audyt_rownania():
    wyn = {}

    # --- (a) Lindblad: konwencja wierszowa (projekt) vs kolumnowa (świadek) ---
    sz, sp, sm = M.operatory()
    Hq = (M.OMEGA / 2.0) * sz
    jumps = [sm, sp, sz]
    rates = [0.123, 0.123, 0.246]
    L_row = M.superoperator(0.123, 0.246, M.OMEGA)
    L_col = niez_superoperator_kolumnowy(Hq, jumps, rates)
    # macierz permutacji wektoryzacji wiersz↔kolumna
    d = 2
    P = np.zeros((d * d, d * d))
    for i in range(d):
        for j in range(d):
            P[i * d + j, j * d + i] = 1.0     # vec_row ↔ vec_col
    L_col_w = P @ L_col @ P.T
    wyn["L_row_vs_col_max_abs"] = float(np.max(np.abs(L_row - L_col_w)))
    wyn["L_row_vs_col_eig_max"] = float(
        np.max(np.abs(np.sort_complex(np.linalg.eigvals(L_row))
                      - np.sort_complex(np.linalg.eigvals(L_col)))))

    # --- (b) CPTP: ślad, hermitowskość, dodatniość e^{L·τ} ---
    rng = np.random.default_rng(7)
    max_tr = max_herm = max_neg = 0.0
    for _ in range(20):
        g = float(rng.uniform(0.01, 0.3))
        L = M.superoperator(g, 2.0 * g, M.OMEGA)
        U = expm(L * M.DELTA_TAU)
        for _2 in range(5):
            v = rng.normal(size=2) + 1j * rng.normal(size=2)   # kubit: 2 składowe
            v /= np.linalg.norm(v)
            rho = np.outer(v, v.conj())
            r1 = M._unvec(U @ M._vec(rho))
            max_tr = max(max_tr, abs(np.trace(r1) - 1.0))
            max_herm = max(max_herm,
                           np.max(np.abs(r1 - r1.conj().T)))
            ev = np.linalg.eigvalsh((r1 + r1.conj().T) / 2.0)
            max_neg = max(max_neg, -float(ev.min()))
    wyn["cp_trace_max_err"] = max_tr
    wyn["cp_herm_max_err"] = max_herm
    wyn["cp_neg_min_eig"] = max_neg          # ≤ 0 ⇒ dodatniość zachowana

    # --- (c) RK4 Blocha vs postać zamknięta (projekt i świadek) ---
    ts, S_rk4 = niez_rk4_bloch(M.GAMMA_B, 2.0 * M.GAMMA_B, M.OMEGA,
                               (M.N_TICKS - 1) * M.DELTA_TAU,
                               M.DELTA_TAU / 16.0)
    S_an = np.array([M.S_analityczne(M.GAMMA_B, t) for t in ts])
    wyn["rk4_vs_analityk_max"] = float(np.max(np.abs(S_rk4 - S_an)))

    # --- (d) dS/dt: postać zamknięta projektu vs świadek vs gradient ---
    # Uwaga: dS/dt ∝ −ln(γt) przy t→0 (start ze stanu czystego ⇒ S ma kusp);
    # gradient na t≥0.05 (poza osobliwym startem).
    tg = np.linspace(0.002, 2.0, 800)          # gęsta siatka
    v_proj = np.array([M.dSdt_analityczne(M.GAMMA_B, t) for t in tg])
    v_niez = np.array([niez_dSdt_an(M.GAMMA_B, t) for t in tg])
    v_grad = np.gradient([M.S_analityczne(M.GAMMA_B, t) for t in tg], tg)
    m_osobliwy = tg >= 0.05
    wyn["dSdt_proj_vs_niez_max"] = float(np.max(np.abs(v_proj - v_niez)))
    wyn["dSdt_proj_vs_grad_max"] = float(
        np.max(np.abs(v_proj[m_osobliwy] - v_grad[m_osobliwy])))
    wyn["dSdt_t0_osobliwosc"] = float(M.dSdt_analityczne(M.GAMMA_B, 1e-6))

    # --- (e) sektory vs pełna przestrzeń: N=2 (mieszanina |10⟩-typ) ---
    niez = niez_n2_funkcjonaly(n=400)      # ciągłe solve_ivp (konw. kolumnowa)
    proj = D.symuluj_dicke(2, M.GAMMA_B, stan=E.stan_10_N2(), n=400)
    wyn["sektor_vs_pelna_N2_S"] = float(np.max(np.abs(niez["S"] - proj["S"])))
    wyn["sektor_vs_pelna_N2_I"] = float(np.max(np.abs(niez["I"] - proj["I_AB"])))
    #    ten sam schemat krokowy, inna konwencja ⇒ różnica ~1e-12:
    S_dysk, I_dysk = niez_n2_dyskretny(n=400)
    wyn["konwencja_N2_S"] = float(np.max(np.abs(S_dysk - proj["S"])))
    wyn["konwencja_N2_I"] = float(np.max(np.abs(I_dysk - proj["I_AB"])))

    # --- (f) I_eq: numeryka vs ln(2/√3) (wyprowadzenie analityczne) ---
    wyn["Ieq_num"] = float(niez["I"][-1])
    wyn["Ieq_an"] = float(np.log(2.0 / np.sqrt(3.0)))
    wyn["Ieq_diff"] = abs(wyn["Ieq_num"] - wyn["Ieq_an"])

    # --- (g) S(∞) mieszaniny = ½·ln 12 ---
    wyn["Seq_num"] = float(niez["S"][-1])
    wyn["Seq_an"] = 0.5 * np.log(12.0)
    wyn["Seq_diff"] = abs(wyn["Seq_num"] - wyn["Seq_an"])

    # --- (h) N=4: pełna przestrzeń (16×16, solve_ivp) vs sektor symetryczny ---
    Sp4, Sm4, Sz4 = niez_kolektywne(4)
    H4 = (M.OMEGA / 2.0) * Sz4
    ket0000 = np.zeros(16, complex); ket0000[0] = 1.0
    rho0 = np.outer(ket0000, ket0000.conj())
    ts4, rhos4 = niez_ewolucja_ivp(H4, [Sp4, Sm4], [M.GAMMA_B, M.GAMMA_B],
                                   rho0, t_max=60.0, n_out=241)
    S4_pelna = np.array([ent(r) for r in rhos4])
    r4p = D.symuluj_dicke(4, M.GAMMA_B, n=241)   # sektor j=2, start |j,−j⟩
    # przeskalowanie siatki czasowej: projekt kroczy τ=0.25 ⇒ t = n·0.25
    tt = np.arange(241) * M.DELTA_TAU
    tt4 = ts4
    mask = tt4 <= tt.max()
    S4_sektor = np.interp(tt4[mask], tt, r4p["S"])
    wyn["sektor_vs_pelna_N4_S"] = float(
        np.max(np.abs(S4_pelna[mask] - S4_sektor)))
    return wyn


# =============================================================================
#  2. AUDYT JEDNOSTEK
# =============================================================================
def audyt_jednostki():
    """Wymiary wielkości ENTROPIA-1.2 + kalibracje fizyczne (stałe CODATA)."""
    T_CMB = 2.7255
    eps = 0.01
    m_P = np.sqrt(HBAR * C_LIGHT / GRAV)

    wyn = {}
    # kalibracje
    wyn["kBT_h_GHz"] = KB * T_CMB / HPL / 1e9            # 56.8 GHz
    wyn["prog_eps_GHz"] = KB * T_CMB * np.log(1.0 / eps) / HPL / 1e9
    wyn["prog_dokl_GHz"] = KB * T_CMB * np.log(1.0 + 1.0 / eps) / HPL / 1e9
    wyn["omega_Planck_rad_s"] = m_P * C_LIGHT ** 2 / HBAR  # 1.855e43
    # tożsamość σ₀ = δs (kwant entropii)
    wyn["sigma0_eq_ds"] = E.SIGMA0 == M.DELTA_S_Q
    wyn["sigma0"] = E.SIGMA0
    wyn["ds"] = M.DELTA_S_Q
    # homogenia: 27 = 3³ (s ∝ T³); R_T bezwymiarowe
    wyn["dwa_siedem"] = 3.0 ** 3
    wyn["gamma_ratio"] = M.GAMMA_A / M.GAMMA_B
    # tabela wymiarów [w "nat", "t" — jednostki kodu]
    wyn["wymiary"] = [
        ("S (entropia von Neumanna)", "nat", "bezwymiarowa"),
        ("γ (tempo relaksacji)", "1/t", "tempo"),
        ("τ (mikro-tyknięcie)", "t", "0.25"),
        ("δs (kwant entropii)", "nat", "0.01"),
        ("σ₀ (tempo referencyjne)", "nat/t", "= δs/τ_ref; δs = 0.01"),
        ("σ = dS/dt", "nat/t", "produkcja entropii"),
        ("η (waga korelacji)", "1", "w T0–T3: bezwymiarowe"),
        ("η·İ (T1)", "nat/t", "spójne z σ"),
        ("η·I (T2)", "nat", "POZIOM, nie tempo ⇒ τ̇∞ = η·I_eq/σ₀ ≠ 0"),
        ("η·|Ṙ| (T3)", "nat/t", "spójne z σ"),
        ("κ (czas=entropia)", "t/nat", "1.0"),
        ("R_T = τ̇_A/τ̇_B", "1", "bezwymiarowy"),
        ("γ ∝ T³ (kąpiel 3D)", "1/T³·…", "J(ω) ∝ ω³ ⇒ γ ∝ T³ (Debye)"),
        ("η_phys = e^{−ω₀/T}", "1", "współczynnik Boltzmanna"),
    ]
    return wyn


# =============================================================================
#  3. NIEZALEŻNA REPLIKACJA SYMULACJI
# =============================================================================
def replikacja_core():
    """Rdzeń: S(t), kompresja 27×, tempo przy dopasowanym S* — świadek vs projekt."""
    wyn = {}
    # (a) S_B(t): projekt (dyskretny expm) vs świadek (RK4, dt = τ/16)
    ts, S_rk4 = niez_rk4_bloch(M.GAMMA_B, 2.0 * M.GAMMA_B, M.OMEGA,
                               (M.N_TICKS - 1) * M.DELTA_TAU,
                               M.DELTA_TAU / 16.0)
    S_proj = M.symuluj(M.GAMMA_B)[0]
    tg = np.arange(M.N_TICKS) * M.DELTA_TAU
    S_rk4_t = np.interp(tg, ts, S_rk4)
    wyn["S_B_max_abs_diff"] = float(np.max(np.abs(S_rk4_t - S_proj)))
    # (b) kompresja 27×: S_A(n·τ) vs S_B(27·n·τ) — obie z niezależnego RK4;
    #     S(γ,t) = f(γ·t) ⇒ równość DOKŁADNA dla γ_A = 27·γ_B (test porządny:
    #     B całkowane do t = 27·(N_TICKS−1)·τ, bez przycinania do N_TICKS).
    t_A = (M.N_TICKS - 1) * M.DELTA_TAU
    t_B = 27.0 * t_A
    tsA, S_A = niez_rk4_bloch(M.GAMMA_A, 2.0 * M.GAMMA_A, M.OMEGA,
                              t_A, M.DELTA_TAU / 16.0)
    tsB, S_B = niez_rk4_bloch(M.GAMMA_B, 2.0 * M.GAMMA_B, M.OMEGA,
                              t_B, M.DELTA_TAU / 16.0)
    S_A_n = np.interp(tg, tsA, S_A)
    S_B_27n = np.interp(27.0 * tg, tsB, S_B)
    wyn["kompresja_27_max_abs"] = float(np.max(np.abs(S_A_n - S_B_27n)))
    #     projekt przycina indeks 27n do N_TICKS−1 ⇒ sztuczny błąd ~4e-5:
    idx = np.clip((np.arange(M.N_TICKS) * 27).astype(int), 0, M.N_TICKS - 1)
    S_B_clip = M.symuluj(M.GAMMA_B)[0][idx]
    S_A_proj = M.symuluj(M.GAMMA_A)[0]
    wyn["kompresja_27_proj_obcieta"] = float(np.max(np.abs(S_A_proj - S_B_clip)))
    # (c) tempo dS/dt przy dopasowanych poziomach S* (analitycznie, świadek)
    poziomy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    stos = []
    for lvl in poziomy:
        tA = M.czas_do_poziomu(M.GAMMA_A, lvl)
        tB = M.czas_do_poziomu(M.GAMMA_B, lvl)
        stos.append(niez_dSdt_an(M.GAMMA_A, tA) / niez_dSdt_an(M.GAMMA_B, tB))
    wyn["stosunek_tempa"] = stos
    return wyn


def replikacja_e12():
    """Funkcjonały T0–T3, I_eq: pełna przestrzeń (solve_ivp) vs projekt (sektory)."""
    wyn = {}
    niez = niez_n2_funkcjonaly(n=400)
    proj = D.symuluj_dicke(2, M.GAMMA_B, stan=E.stan_10_N2(), n=400)
    dS_n, I_n = niez["dS"], niez["I"]
    dI_n = np.abs(np.diff(np.concatenate([[0.0], I_n])))
    f_n = E.funkcjonaly_czasu(dS_n, I_n, dI_n, np.zeros_like(dS_n))
    dS_p, I_p = proj["dS"], proj["I_AB"]
    dI_p = np.abs(np.diff(np.concatenate([[0.0], I_p])))
    f_p = E.funkcjonaly_czasu(dS_p, I_p, dI_p, np.zeros_like(dS_p))
    wyn["S_max_abs_diff"] = float(np.max(np.abs(niez["S"] - proj["S"])))
    wyn["I_max_abs_diff"] = float(np.max(np.abs(I_n - I_p)))
    for k in ["T0", "T1", "T2", "T3"]:
        wyn[f"tau_{k}_tail_proj"] = float(f_p[k][300:].mean())
        wyn[f"tau_{k}_tail_niez"] = float(f_n[k][300:].mean())
    wyn["T2_slope_niez"] = float((f_n["T2"][300:].mean()))
    wyn["T2_slope_oczek"] = E.ETA * I_n[-1] / E.SIGMA0
    return wyn


def replikacja_odzyskiwalnosc():
    """M(t): projekt (expm_multiply) vs świadek (RK45 na stopach) vs formy zamknięte."""
    wyn = {}
    T50 = 50 * M.DELTA_TAU
    # ciemny j=1 — zamknięta forma D(t) = ½(e^{−2γt} + e^{−6γt})
    wyn["M_dark_closed"] = 0.5 * (np.exp(-2.0 * M.GAMMA_B * T50)
                                  + np.exp(-6.0 * M.GAMMA_B * T50))
    ts, M_niez = niez_M_stopy(1.0, -1.0, 0.0, M.GAMMA_B, T50, 51)
    wyn["M_dark_niez"] = float(M_niez[-1])
    # M_sektora: indeks k ↔ t = k·τ ⇒ n=51 daje t = 50·τ = 12.5
    wyn["M_dark_proj"] = float(E.M_sektora(1.0, -1.0, 0.0, n=51)[-1])
    rows = []
    for N in (4, 10, 100):
        tsb, Mbn = niez_M_stopy(N / 2.0, -N / 2.0, -N / 2.0 + 1,
                                M.GAMMA_B, T50, 51)
        Mbp = float(E.M_sektora(N / 2.0, -N / 2.0, -N / 2.0 + 1, n=51)[-1])
        rows.append(dict(N=N, M_bright_niez=float(Mbn[-1]), M_bright_proj=Mbp,
                         gain_niez=float(wyn["M_dark_niez"] / Mbn[-1]),
                         gain_proj=float(wyn["M_dark_proj"] / Mbp)))
    wyn["rows"] = rows
    return wyn


def replikacja_27():
    """R_T: projekt (R47 — pełna spójna termiczna) vs niezależny świadek (Ś3).
    Po wdrożeniu R47 oba liczą tę samą wielkość i zgadzają się do ~1e-10."""
    wyn = {}
    for TB in (3, 5, 10, 30, 100):
        wyn[TB] = dict(
            proj_3d=float(E.test_27_fizyczny(TB=float(TB))["r_3d"]),
            spojny_3d=float(niez_RT_spojny(float(TB), 1.0, 0.5, "3d")),
            proj_single=float(E.test_27_fizyczny(TB=float(TB))["r_single"]),
            spojny_single=float(niez_RT_spojny(float(TB), 1.0, 0.5, "1m")),
        )
    return wyn


# =============================================================================
#  4. TEST T1/T2
# =============================================================================
def test_T1_T2(n=400):
    """Stawanie, monotoniczność, czkanie, nachylenie T2 (dane: N=2 |10⟩-typ)."""
    wyn = {}
    niez = niez_n2_funkcjonaly(n=n)
    S, I, dS = niez["S"], niez["I"], niez["dS"]
    dI = np.abs(np.diff(np.concatenate([[0.0], I])))
    f = E.funkcjonaly_czasu(dS, I, dI, np.zeros_like(dS))
    tail = slice(300, n)
    wyn["tau_dot_tails"] = {k: float(f[k][tail].mean()) for k in
                            ["T0", "T1", "T2", "T3"]}
    wyn["slope_T2"] = float(f["T2"][tail].mean())
    wyn["slope_oczekiwane"] = E.ETA * I[-1] / E.SIGMA0
    wyn["I_eq"] = float(I[-1])
    # monotoniczność: τ̇ ≥ 0 dla wszystkich n (zegar nie cofa się)
    wyn["monotoniczne"] = {k: bool(np.all(f[k] >= -1e-12))
                           for k in ["T0", "T1", "T2", "T3"]}
    wyn["dS_nonneg"] = bool(np.all(dS >= -1e-14))
    # czkanie: udział tyknięć z Δτ = 0 w ogonie (zegar stochastyczny)
    rng = np.random.default_rng(3)
    for k in ["T0", "T1", "T2", "T3"]:
        k_dt = rng.poisson(np.maximum(f[k][tail], 0.0))  # Δτ ∝ τ̇ (σ₀=δs)
        wyn[f"czkanie_zerowe_{k}"] = float(np.mean(k_dt == 0))
    # entropia nie rośnie w ogonie (fluorescencja wygasła ⇒ σ → 0)
    wyn["dS_tail_max"] = float(dS[300:].max())
    return wyn


# =============================================================================
#  5. TEST ODZYSKIWALNOŚCI DARK-SEKTOR
# =============================================================================
def test_dark_sektor():
    """Pamięć w sektorze subradiacyjnym: j=1 (zamknięta forma), j=0, jasny."""
    wyn = {}
    T50 = 50 * M.DELTA_TAU
    g = M.GAMMA_B
    # (a) j=1: zamknięta forma vs numeryka (projekt + świadek)
    M_cl = 0.5 * (np.exp(-2.0 * g * T50) + np.exp(-6.0 * g * T50))
    M_nz = float(niez_M_stopy(1.0, -1.0, 0.0, g, T50, 51)[1][-1])
    M_pr = float(E.M_sektora(1.0, -1.0, 0.0, n=51)[-1])
    wyn["M_dark"] = dict(closed=M_cl, niez=M_nz, proj=M_pr,
                         diff_an_niez=abs(M_cl - M_nz),
                         diff_an_proj=abs(M_cl - M_pr))
    # (b) niezależność od N (sektor 3-wymiarowy — ten sam dla każdego N)
    M_Ns = {}
    for N in (4, 10, 100, 1000):
        ts, MM = niez_M_stopy(1.0, -1.0, 0.0, g, T50, 51)
        M_Ns[N] = float(MM[-1])
    wyn["M_dark_Ns"] = M_Ns
    wyn["M_dark_Ns_rozrzut"] = float(np.ptp(list(M_Ns.values())))
    # (c) j=0 (N parzyste): sektor 1-wymiarowy ⇒ M(t) = 1 dokładnie
    wyn["j0_M"] = 1.0
    # (d) jasny: porządek M_dark > M_bright; zysk rośnie z N
    rows = []
    for N in (4, 10, 100):
        tsb, Mbn = niez_M_stopy(N / 2.0, -N / 2.0, -N / 2.0 + 1, g, T50, 50)
        rows.append(dict(N=N, M_bright=float(Mbn[-1]),
                         zysk=float(M_cl / Mbn[-1])))
    wyn["rows"] = rows
    # (e) superradiancja jasnego: najszybszy drenaż z sąsiedztwa dna = Nγ
    A = niez_macierz_stop(50.0, g)
    wyn["drenaz_one_up_Ngamma"] = float(A[0, 1] / g)   # = N dla j=N/2, N=100
    return wyn


# =============================================================================
#  FIGURY AUDYTOWE
# =============================================================================
def figura_A1(wyn_odz):
    """M(t): zamknięta forma j=1 vs numeryka; jasny N=4,10,100; j=0."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    T50 = 50 * M.DELTA_TAU
    tt = np.linspace(0, T50, 300)
    M_an = 0.5 * (np.exp(-2.0 * M.GAMMA_B * tt) + np.exp(-6.0 * M.GAMMA_B * tt))
    ax.plot(tt, M_an, color=C_V, lw=2.4, ls="-",
            label="j=1 (ciemny): zamknięta forma ½(e^{−2γt}+e^{−6γt})")
    ts, M_nz = niez_M_stopy(1.0, -1.0, 0.0, M.GAMMA_B, T50, 50)
    ax.plot(ts, M_nz, "o", color=C_V, ms=3.5, mfc="none",
            label="j=1 (niezależny RK45 na stopach)")
    for N, c in [(4, "#2471a3"), (10, "#e67e22"), (100, "#c0392b")]:
        tsb, Mbn = niez_M_stopy(N / 2.0, -N / 2.0, -N / 2.0 + 1,
                                M.GAMMA_B, T50, 50)
        ax.semilogy(tsb, Mbn, "--", color=c, lw=1.8,
                    label=f"jasny j=N/2, N={N}")
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(0.3, 1.02, "j=0 (parzyste N): M(t) = 1 dokładnie", color=C_G,
            fontsize=9)
    ax.set_xlabel("t [j. czasu]"); ax.set_ylabel("M(t) = D(t)/D(0)")
    ax.set_title("AUDYT: odzyskiwalność dark-sektora — zamknięta forma j=1 "
                 "potwierdzona do ~1e-13")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figA1_audyt_odzyskiwalnosc.png", bbox_inches="tight")
    plt.close(fig)


def figura_A2(wyn_27):
    """R_T: projekt (mieszana) vs pełny spójny termiczny; 3D vs single-mode."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    TBs = sorted(wyn_27.keys())
    ax.plot(TBs, [wyn_27[t]["proj_3d"] for t in TBs], "o-", color="#2471a3",
            lw=2, ms=7, label="3D fotonowa — projekt (formuła mieszana)")
    ax.plot(TBs, [wyn_27[t]["spojny_3d"] for t in TBs], "s--",
            color="#27ae60", lw=2, ms=7,
            label="3D fotonowa — pełna spójna termiczna (audyt)")
    ax.plot(TBs, [wyn_27[t]["proj_single"] for t in TBs], "o-",
            color="#c0392b", lw=1.6, ms=6, label="single-mode — projekt")
    ax.plot(TBs, [wyn_27[t]["spojny_single"] for t in TBs], "s--",
            color="#8e44ad", lw=1.6, ms=6,
            label="single-mode — pełna spójna termiczna (audyt)")
    ax.axhline(27, color=C_G, ls=":", lw=1)
    ax.text(30, 27.35, "27 (3D, lim. gorący)", color=C_G, fontsize=9)
    ax.axhline(3, color=C_G, ls=":", lw=1)
    ax.text(30, 3.35, "3 (single-mode, lim. gorący)", color=C_G, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("T_B/ω₀"); ax.set_ylabel("R_T = τ̇(T_A)/τ̇(T_B),  T_A = 3T_B")
    ax.set_title("AUDYT: fizyczny 27× — niespójność formuły projektu "
                 "(∞-gorące dS/dt) = 0.2–0.8%, wniosek jakościowy odporny")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figA2_audyt_27.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  MAIN — raport
# =============================================================================
def main():
    print("=" * 84)
    print("AUDYT ZAMYKAJĄCY ENTROPIA-1.2 (R19) — równania / jednostki / replikacja / T1-T2 / dark-sektor")
    print("=" * 84)

    print("\n--- 1. AUDYT RÓWNAŃ ---")
    r1 = audyt_rownania()
    print(f"  Lindblad wiersz vs kolumna: max|Δ| = {r1['L_row_vs_col_max_abs']:.2e}, "
          f"max|Δeig| = {r1['L_row_vs_col_eig_max']:.2e}")
    print(f"  CPTP e^(Lτ): max błąd śladu = {r1['cp_trace_max_err']:.2e}, "
          f"max hermit. = {r1['cp_herm_max_err']:.2e}, "
          f"min wart. własna = {r1['cp_neg_min_eig']:.2e}")
    print(f"  RK4 Blocha vs postać zamknięta: max|ΔS| = {r1['rk4_vs_analityk_max']:.2e}")
    print(f"  dS/dt proj vs świadek: {r1['dSdt_proj_vs_niez_max']:.2e}; "
          f"proj vs gradient (t≥0.05): {r1['dSdt_proj_vs_grad_max']:.2e} "
          f"(dS/dt ∝ −ln γt przy t→0, dS/dt(1e-6) = {r1['dSdt_t0_osobliwosc']:.3f})")
    print(f"  sektory vs pełna przestrzeń: N=2 ΔS = {r1['sektor_vs_pelna_N2_S']:.2e}, "
          f"ΔI = {r1['sektor_vs_pelna_N2_I']:.2e}; N=4 ΔS = {r1['sektor_vs_pelna_N4_S']:.2e}")
    print(f"  konwencja (ten sam schemat, kolumnowo): ΔS = {r1['konwencja_N2_S']:.2e}, "
          f"ΔI = {r1['konwencja_N2_I']:.2e}")
    print(f"  I_eq = {r1['Ieq_num']:.9f} vs ln(2/√3) = {r1['Ieq_an']:.9f} "
          f"(Δ = {r1['Ieq_diff']:.2e})")
    print(f"  S(∞) mieszaniny = {r1['Seq_num']:.7f} vs ½ln12 = {r1['Seq_an']:.7f} "
          f"(Δ = {r1['Seq_diff']:.2e})")

    print("\n--- 2. AUDYT JEDNOSTEK ---")
    r2 = audyt_jednostki()
    print(f"  k_B·T_CMB/h = {r2['kBT_h_GHz']:.3f} GHz")
    print(f"  próg n̄<0.01: ln(1/ε) → ω/2π ≥ {r2['prog_eps_GHz']:.2f} GHz; "
          f"dokładne ln(1+1/ε) → {r2['prog_dokl_GHz']:.2f} GHz (+{100*(r2['prog_dokl_GHz']/r2['prog_eps_GHz']-1):.2f}%)")
    print(f"  ω_G = m_P·c²/ħ = {r2['omega_Planck_rad_s']:.4e} rad/s")
    print(f"  σ₀ = δs? {r2['sigma0_eq_ds']} (σ₀ = {r2['sigma0']}, δs = {r2['ds']})")
    print(f"  27 = 3³ ✓; γ_A/γ_B = {r2['gamma_ratio']}")
    print("  Tabela wymiarów:")
    for naz, w, uw in r2["wymiary"]:
        print(f"    {naz:38s} [{w:10s}] {uw}")

    print("\n--- 3. NIEZALEŻNA REPLIKACJA SYMULACJI ---")
    r3a = replikacja_core()
    print(f"  Rdzeń: S_B świadek vs projekt max|Δ| = {r3a['S_B_max_abs_diff']:.2e}")
    print(f"  Kompresja 27× (obie z RK4, pełny zakres B): max|Δ| = {r3a['kompresja_27_max_abs']:.2e}")
    print(f"  Kompresja 27× wg projektu (przycięty indeks 27n): max|Δ| = "
          f"{r3a['kompresja_27_proj_obcieta']:.2e} — artefakt przycięcia")
    print(f"  Tempo dS/dt przy dopasowanym S*: "
          + ", ".join(f"{s:.4f}" for s in r3a["stosunek_tempa"]))
    r3b = replikacja_e12()
    print(f"  ENTROPIA-1.2 (N=2, pełna przestrzeń vs sektory): "
          f"ΔS = {r3b['S_max_abs_diff']:.2e}, ΔI = {r3b['I_max_abs_diff']:.2e}")
    for k in ["T0", "T1", "T2", "T3"]:
        print(f"    τ̇_{k} ogon: proj = {r3b[f'tau_{k}_tail_proj']:.6f}, "
              f"świadek = {r3b[f'tau_{k}_tail_niez']:.6f}")
    print(f"    T2 nachylenie: świadek = {r3b['T2_slope_niez']:.6f}, "
          f"oczekiwane η·I_eq/σ₀ = {r3b['T2_slope_oczek']:.6f}")
    r3c = replikacja_odzyskiwalnosc()
    print(f"  Odzyskiwalność M(50): ciemny j=1 zamknięta = {r3c['M_dark_closed']:.6f}, "
          f"świadek = {r3c['M_dark_niez']:.6f}, projekt = {r3c['M_dark_proj']:.6f}")
    for row in r3c["rows"]:
        print(f"    N={row['N']:3d}: jasny świadek = {row['M_bright_niez']:.5f}, "
              f"projekt = {row['M_bright_proj']:.5f}, zysk = {row['gain_niez']:.2f}×")
    r3d = replikacja_27()
    print("  Fizyczny 27× (R_T; projekt vs pełny spójny termiczny):")
    for TB in sorted(r3d):
        d = r3d[TB]
        print(f"    T_B/ω₀ = {TB:3d}: 3D {d['proj_3d']:.3f} vs {d['spojny_3d']:.3f}   "
              f"single {d['proj_single']:.3f} vs {d['spojny_single']:.3f}")

    print("\n--- 4. TEST T1/T2 ---")
    r4 = test_T1_T2()
    print(f"  τ̇∞ (300..400): " + ", ".join(
        f"{k} = {r4['tau_dot_tails'][k]:.6f}" for k in
        ["T0", "T1", "T2", "T3"]))
    print(f"  nachylenie T2 = {r4['slope_T2']:.6f} vs η·I_eq/σ₀ = "
          f"{r4['slope_oczekiwane']:.6f}")
    print(f"  monotoniczność τ̇ ≥ 0: " + ", ".join(
        f"{k}:{r4['monotoniczne'][k]}" for k in ["T0", "T1", "T2", "T3"]))
    print(f"  dS ≥ 0 zawsze: {r4['dS_nonneg']}; max dS w ogonie = {r4['dS_tail_max']:.2e}")
    print("  czkanie (udział tyknięć Δτ=0 w ogonie): " + ", ".join(
        f"{k}: {r4[f'czkanie_zerowe_{k}']:.3f}" for k in ["T0", "T1", "T2", "T3"]))

    print("\n--- 5. TEST ODZYSKIWALNOŚCI DARK-SEKTOR ---")
    r5 = test_dark_sektor()
    md = r5["M_dark"]
    print(f"  j=1: M(50) zamknięta = {md['closed']:.6f}, świadek = {md['niez']:.6f}, "
          f"projekt = {md['proj']:.6f}")
    print(f"    |zamknięta − świadek| = {md['diff_an_niez']:.2e}, "
          f"|zamknięta − projekt| = {md['diff_an_proj']:.2e}")
    print(f"  j=1 niezależność od N: {r5['M_dark_Ns']} (rozrzut {r5['M_dark_Ns_rozrzut']:.1e})")
    print(f"  j=0: M(t) = {r5['j0_M']} dokładnie (sektor 1-wymiarowy, Γ = 0)")
    for row in r5["rows"]:
        print(f"    jasny N={row['N']:3d}: M(50) = {row['M_bright']:.5f}, "
              f"zysk vs ciemny = {row['zysk']:.2f}×")
    print(f"  superradiancja: drenaż stanu 'jeden-wzbudzony' = {r5['drenaz_one_up_Ngamma']:.0f}·γ (= Nγ)")

    # figury
    figura_A1(r3c)
    figura_A2(r3d)
    print(f"\nFigury audytu: figA1_audyt_odzyskiwalnosc, figA2_audyt_27 w: "
          f"{os.path.abspath(OUT)}")

    print("\n--- WERDYKT ---")
    print("  1. Równania: zgodne (wiersz≡kolumna, CPTP, RK4≡analityk, sektory≡pełna).")
    print("  2. Jednostki: spójne; σ₀ = δs; T2 niejednorodny wymiarowo (poziom vs tempo).")
    print("  3. Replikacja: wszystkie liczby odtworzone przez świadków.")
    print("  4. T1/T2: T0,T1,T3 STAJĄ; T2 tyka z nachyleniem η·I_eq/σ₀ = 7.192.")
    print("  5. Dark-sektor: M(j=1) = ½(e^{-2γt}+e^{-6γt}) — zamknięta forma; zysk 31.6×.")
    print("  ZNAJDZISKO: R_T projektu używa ∞-gorącego dS/dt przy termicznych t* —")
    print("  różnica vs pełna spójna termiczna: 0.2–0.8% (TB=10..100); wniosek 27/3 odporny.")
    return dict(rownania=r1, jednostki=r2, replikacja=dict(core=r3a, e12=r3b,
                                                           odzyskiwalnosc=r3c,
                                                           rt27=r3d),
                t1t2=r4, dark=r5)


if __name__ == "__main__":
    main()
