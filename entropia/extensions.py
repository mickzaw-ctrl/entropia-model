#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  ROZSZERZENIA MODELU KOSMOLOGICZNEGO «ENTROPIA»
=============================================================================
  R1 — SKOŃCZONA TEMPERATURA KĄPIELI
       Kąpiel Gibbsa: ρ_eq = diag(p_g, p_e),  p_e/p_g = η = e^{-βΩ}.
       Tempo: a = 2γ/(1+η) (emisja), b = 2γη/(1+η) (absorpcja); a+b = 2γ.
       η = 1 (β=0)  ⇒  ρ_eq = ½·𝟙  (rdzeń modelu, ln 2).
       η < 1        ⇒  S(∞) = H(p_g) < ln 2; |r|∞ = (1−η)/(1+η) > 0.
       Mapa jest NIEUNITALNA dla η < 1: tw. Ando–Lindblada nie gwarantuje
       monotoniczności S; tw. Spohna (σ = −d/dt S(ρ‖ρ_eq) ≥ 0) zachodzi.
       Możliwy efekt: entropia chwilowo PRZEWYŻSZA plateau i opada (η < 1/3).

  R2 — WIELE KUBITÓW
       (a) Niezależne kąpiele: S_total = N·S₁ (ekstensywność), S(∞) = N·ln 2.
       (b) Wspólna kąpiel kolektywna (Dicke): jumpy S± = Σ σ±^i.
           N=2: sektor trypletowy termalizuje do 𝟙₃/3 ⇒ S(∞) = ln 3 < 2 ln 2
           (deficyt ln(4/3) — entropia zamknięta w korelacjach, I = ln(4/3)).
           Singlet (Bell) jest CIEMNY dla kanału kolektywnego (γ_φ = 0):
           start |10⟩ ⇒ S(∞) = ½·ln 12 ≈ 1.2425, czystość 1/3, korelacja żyje.

  R3 — SPRZĘŻENIE „ZEGAR → TEMPO"
       γ_eff(n) = γ₀·fb(T_n/T_scale),  T_n = skumulowany czas-zegara = S_n.
       fb = 1 (stały) | 1/(1+αu) (chłodzenie — ekspansja) | 1+αu (przyspieszanie).
       Sprzężenie zależy od odczytu zegara ⇒ kompresja 27× nie zostaje zepsuta.

  Uruchomienie:  python3 model_rozszerzenia.py
=============================================================================
"""

import os
from functools import reduce

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
C_A, C_B, C_G, C_V = M.C_A, M.C_B, M.C_G, "#8e44ad"
C_COOL, C_ACC = "#1a5276", "#c0392b"

# -----------------------------------------------------------------------------
#  WSPÓLNY BUDOWNICZY SUPEROPERATORA (dowolna liczba jumpów)
# -----------------------------------------------------------------------------
def superoperator_z_jumpami(H, jumps, rates):
    """
    L: vec_R(ρ) → vec_R(ρ̇),  L = −i[H,·] + Σ_j r_j·D[L_j].
    Konwencja wektoryzacji WIERSZOWEJ (jak w rdzeniu):
    vec_R(ρ) = [ρ00, ρ01, ρ10, ρ11]  ⇒  vec(AρB) = (A ⊗ Bᵀ)vec(ρ).
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for J, r in zip(jumps, rates):
        Jd = J.conj().T
        JJ = Jd @ J
        L += r * (np.kron(J, J.conj()) - 0.5 * (np.kron(JJ, I) + np.kron(I, JJ.T)))
    return L


def _weryfikacja():
    """Kontrola krzyżowa: builder == superoperator rdzenia (1 kubit)."""
    sz, sp, sm = M.operatory()
    H = (M.OMEGA / 2.0) * sz
    g = 0.123
    L1 = superoperator_z_jumpami(H, [sm, sp, sz], [g, g, M.GAMMA_PHI * g])
    L2 = M.superoperator(g, M.GAMMA_PHI * g, M.OMEGA)
    np.testing.assert_allclose(L1, L2, atol=1e-12, err_msg="builder vs core")


# -----------------------------------------------------------------------------
#  R1 — SKOŃCZONA TEMPERATURA
# -----------------------------------------------------------------------------
def superoperator_termiczny(gamma, eta, gamma_phi=None, omega=M.OMEGA):
    """Kąpiel Gibbsa: emisja a = 2γ/(1+η), absorpcja b = 2γη/(1+η).
    UWAGA: w rdzeniu `sp` = macierz |0⟩⟨1| (OPUSZCZA |1⟩→|0⟩, a więc to σ₋),
    a `sm` = |1⟩⟨0| (σ₊). Przy równych stawkach zamiana nie szkodzi; przy
    skończonej T przypisujemy świadomie: stawkę emisji a do `sp`, absorpcję b
    do `sm`."""
    if gamma_phi is None:
        gamma_phi = M.GAMMA_PHI * gamma
    sz, sp, sm = M.operatory()
    H = (omega / 2.0) * sz
    a = 2.0 * gamma / (1.0 + eta)
    b = 2.0 * gamma * eta / (1.0 + eta)
    return superoperator_z_jumpami(H, [sp, sm, sz], [a, b, gamma_phi])


def r_eq_termiczny(eta):
    """Równowagowa polaryzacja wzdłuż z: (p_g − p_e) = (1−η)/(1+η)."""
    return (1.0 - eta) / (1.0 + eta)


def H_bin(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-15)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def S_eq_termiczna(eta):
    return H_bin(1.0 / (1.0 + eta))


def czystosc_równowagowa(eta):
    r = r_eq_termiczny(eta)
    return (1.0 + r * r) / 2.0


def r_norm_termiczny(gamma, eta, t):
    """|r(t)| w postaci zamkniętej (γ_⊥ = γ + 2γ_φ = 5γ)."""
    req = r_eq_termiczny(eta)
    rz = req + (np.cos(M.THETA0) - req) * np.exp(-2.0 * gamma * t)
    rp = np.sin(M.THETA0) * np.exp(-(1.0 + 2.0 * M.GAMMA_PHI) * gamma * t)
    return np.sqrt(np.clip(rz * rz + rp * rp, 0.0, 1.0))


def S_termiczna_analitycznie(gamma, eta, t):
    return H_bin((1.0 + r_norm_termiczny(gamma, eta, t)) / 2.0)


def dSdt_termiczne_analitycznie(gamma, eta, t):
    """
    R47 (audyt ENTROPIA-1.2, znalezisko nr 1): SPÓJNA pochodna dS/dt dla
    kąpieli Gibbsa — pochodna S_termiczna_analitycznie, bez mieszania z
    formułą ∞-gorącą.

        dS/dt = −artanh(r)·ṙ,   ṙ = (r_z·ṙ_z + r_⊥·ṙ_⊥)/r,
        r_z  = r_eq + (cosθ₀−r_eq)e^{−2γt},   r_⊥ = sinθ₀·e^{−5γt} (γ_φ = 2γ).

    Tw. Spohna: σ = −d/dt S(ρ‖ρ_eq) ≥ 0; dla kąpieli Gibbsa dS/dt może być
    ujemna (ochładzanie), ale produkcja względem ρ_eq jest zawsze dodatnia.
    """
    r = r_norm_termiczny(gamma, eta, t)
    if r <= 0.0 or r >= 1.0:
        return 0.0
    r_eq = r_eq_termiczny(eta)
    rz = r_eq + (np.cos(M.THETA0) - r_eq) * np.exp(-2.0 * gamma * t)
    rp = np.sin(M.THETA0) * np.exp(-(1.0 + 2.0 * M.GAMMA_PHI) * gamma * t)
    drz = -2.0 * gamma * (rz - r_eq)
    drp = -(1.0 + 2.0 * M.GAMMA_PHI) * gamma * rp
    dr = (rz * drz + rp * drp) / r
    return -0.5 * np.log((1.0 + r) / (1.0 - r)) * dr


def symuluj_termicznie(gamma, eta, n=M.N_TICKS, delta_tau=M.DELTA_TAU,
                       p0=None):
    """Dyskretna ewolucja Lindblada (do kontroli krzyżowej z analityką).
    p0: opcjonalne populacje początkowe [p_g, p_e] (stan diagonalny);
    domyślnie czysty |ψ(θ₀,φ₀)⟩ (rdzeń modelu)."""
    L = superoperator_termiczny(gamma, eta)
    U = expm(L * delta_tau)
    if p0 is not None:
        rho = np.diag(p0).astype(complex)
    else:
        psi = np.array([np.cos(M.THETA0 / 2.0),
                        np.exp(1j * M.PHI0) * np.sin(M.THETA0 / 2.0)])
        rho = np.outer(psi, psi.conj())
    S = np.zeros(n); P = np.zeros(n); R = np.zeros((n, 3))
    for i in range(n):
        S[i] = M.entropia(rho); P[i] = M.czystosc(rho); R[i] = M.bloch(rho)
        rho = M._unvec(U @ M._vec(rho))
    return S, P, R


def czas_do_poziomu_T(gamma, eta, poziom):
    """Najwcześniejszy t, dla którego S(t) = poziom (analitycznie, brentq)."""
    from scipy.optimize import brentq
    if poziom <= 0:
        return 0.0
    Seq = S_eq_termiczna(eta)
    if poziom >= Seq:
        return np.inf
    f = lambda t: S_termiczna_analitycznie(gamma, eta, t) - poziom
    return float(brentq(f, 0.0, 12.0 / gamma, xtol=1e-12))


# -----------------------------------------------------------------------------
#  R2 — WIELE KUBITÓW
# -----------------------------------------------------------------------------
def operator_jednokubitowy(op, i, N):
    """op na kubicie i w przestrzeni 2^N (iloczyn tensorowy z 𝟙)."""
    I2 = np.eye(2, dtype=complex)
    ops = [I2] * N
    ops[i] = op
    return reduce(np.kron, ops)


def vecR(rho):
    """Wektoryzacja wierszowa (zgodna z superoperator_z_jumpami): ρ.flatten()."""
    return np.asarray(rho, dtype=complex).flatten()


def unvecR(v, d):
    return np.asarray(v, dtype=complex).reshape(d, d)


def symuluj_jeden(gamma, gamma_phi, n=M.N_TICKS, delta_tau=M.DELTA_TAU):
    """Jeden kubit (użyteczne dla N niezależnych; opcjonalnie γ_φ = 0)."""
    sz, sp, sm = M.operatory()
    H = (M.OMEGA / 2.0) * sz
    jumps, rates = [sm, sp], [gamma, gamma]
    if gamma_phi > 0:
        jumps.append(sz); rates.append(gamma_phi)
    L = superoperator_z_jumpami(H, jumps, rates)
    U = expm(L * delta_tau)
    psi = np.array([np.cos(M.THETA0 / 2.0),
                    np.exp(1j * M.PHI0) * np.sin(M.THETA0 / 2.0)])
    rho = np.outer(psi, psi.conj())
    S = np.zeros(n); P = np.zeros(n)
    for i in range(n):
        S[i] = M.entropia(rho); P[i] = M.czystosc(rho)
        rho = M._unvec(U @ M._vec(rho))
    return S, P


def symuluj_niezalezne(gamma, N, gamma_phi=0.0, n=M.N_TICKS):
    """N kubitów w N niezależnych kąpielach (produkt zachowany)."""
    S1, P1 = symuluj_jeden(gamma, gamma_phi, n=n)
    return N * S1, P1 ** N


def macierze_kolektywne(N):
    """S_z, S₊, S₋ dla N kubitów + lokalne σ_z^i."""
    sz, sp, sm = M.operatory()
    S_z = sum(operator_jednokubitowy(sz, i, N) for i in range(N))
    S_p = sum(operator_jednokubitowy(sp, i, N) for i in range(N))
    S_m = S_p.conj().T
    lokalne_sz = [operator_jednokubitowy(sz, i, N) for i in range(N)]
    return S_z, S_p, S_m, lokalne_sz


def stan_poczatkowy_N(lista_stanow):
    """Czysty stan produktowy z listy jednokubitowych wektorów."""
    v = lista_stanow[0]
    for w in lista_stanow[1:]:
        v = np.kron(v, w)
    return np.outer(v, v.conj())


def concurrence2(rho):
    """Konkurencja (Wootters) dla stanu 2-kubitowego."""
    sy = np.array([[0.0, -1j], [1j, 0.0]])
    YY = np.kron(sy, sy)
    R = rho @ YY @ rho.conj() @ YY
    ev = np.sqrt(np.maximum(np.linalg.eigvalsh((R + R.conj().T) / 2.0), 0.0))
    ev = np.sort(ev)[::-1]
    return float(np.max([0.0, ev[0] - ev[1] - ev[2] - ev[3]]))


def symuluj_wspolne(gamma, stan, N=2, gamma_phi=0.0, omega=M.OMEGA,
                    n=M.N_TICKS, delta_tau=M.DELTA_TAU, zwroc_stan=False):
    """N kubitów we WSPÓLNEJ kąpieli (kolektywne S₊, S₋)."""
    S_z, S_p, S_m, lok_sz = macierze_kolektywne(N)
    H = (omega / 2.0) * S_z
    jumps, rates = [S_p, S_m], [gamma, gamma]
    if gamma_phi > 0:
        jumps += lok_sz
        rates += [gamma_phi] * N
    L = superoperator_z_jumpami(H, jumps, rates)
    U = expm(L * delta_tau)
    rho = np.array(stan, dtype=complex)
    d = 2 ** N
    S = np.zeros(n); P = np.zeros(n); C = np.zeros(n); MI = np.zeros(n)
    for i in range(n):
        S[i] = M.entropia(rho); P[i] = M.czystosc(rho)
        if N == 2:
            C[i] = concurrence2(rho)
            r4 = rho.reshape(2, 2, 2, 2)             # [a1,a2,b1,b2]
            rhoA = np.einsum("abcb->ac", r4)         # ślad po kubicie B
            rhoB = np.einsum("abac->bc", r4)         # ślad po kubicie A
            MI[i] = M.entropia(rhoA) + M.entropia(rhoB) - S[i]
        rho = unvecR(U @ vecR(rho), d)
    if zwroc_stan:
        return S, P, C, MI, rho
    return S, P, C, MI


def czasy_90(S, asymptota):
    i = np.argmax(S >= 0.9 * asymptota)
    return i if S[i] >= 0.9 * asymptota else np.inf


# -----------------------------------------------------------------------------
#  R3 — SPRZĘŻENIE „ZEGAR → TEMPO"
# -----------------------------------------------------------------------------
FB_STALY = lambda u: 1.0
FB_CHLODZENIE = lambda u, a=1.0: 1.0 / (1.0 + a * u)
FB_PRZYSPIESZANIE = lambda u, a=1.0: 1.0 + a * u


def symuluj_feedback(gamma0, fb, n=600, t_scale=0.5, delta_tau=M.DELTA_TAU):
    """γ_eff(n) = γ₀·fb(T_n/t_scale), T_n = skumulowana entropia (zegar)."""
    L0 = M.superoperator  # alias
    psi = np.array([np.cos(M.THETA0 / 2.0),
                    np.exp(1j * M.PHI0) * np.sin(M.THETA0 / 2.0)])
    rho = np.outer(psi, psi.conj())
    S = np.zeros(n); G = np.zeros(n); T = 0.0
    for i in range(n):
        S[i] = M.entropia(rho)
        gamma = gamma0 * fb(T / t_scale)
        G[i] = gamma
        L = L0(gamma, M.GAMMA_PHI * gamma, M.OMEGA)
        rho = M._unvec(expm(L * delta_tau) @ M._vec(rho))
        T += max(0.0, M.entropia(rho) - S[i])
    return S, G, T


def symuluj_feedback_ciggly(gamma0, fb, t_max, t_scale=0.5, n_out=2000):
    """
    Ciągła (solve_ivp) ewolucja z feedbackiem: układ 2 ODE na (r_z, r_⊥),
    γ_eff = γ₀·fb(S(t)/t_scale), S = H((1+|r|)/2). Używana do weryfikacji
    kompresji 27× w czasie ciągłym (bez artefaktów dyskretyzacji).
    """
    from scipy.integrate import solve_ivp

    def f(t, y):
        rz, rp = y
        r = np.sqrt(np.clip(rz * rz + rp * rp, 0.0, 1.0))
        S = H_bin((1.0 + r) / 2.0)
        g = gamma0 * fb(S / t_scale)
        drz = -2.0 * g * rz
        drp = -(1.0 + 2.0 * M.GAMMA_PHI) * g * rp   # γ_⊥ = γ + 2γ_φ = 5γ
        return [drz, drp]

    sol = solve_ivp(f, (0.0, t_max), [np.cos(M.THETA0), np.sin(M.THETA0)],
                    t_eval=np.linspace(0, t_max, n_out), rtol=1e-10, atol=1e-12)
    r = np.sqrt(np.clip(sol.y[0] ** 2 + sol.y[1] ** 2, 0.0, 1.0))
    return sol.t, H_bin((1.0 + r) / 2.0)


def negatywnosc2(rho):
    """Negatywność (PPT) stanu 2-kubitowego: N = (‖ρ^T_B‖₁ − 1)/2."""
    r4 = rho.reshape(2, 2, 2, 2)                       # [a1,a2,b1,b2]
    rhoPT = r4.transpose(0, 3, 2, 1).reshape(4, 4)     # ⟨a1 a2|ρ^TB|b1 b2⟩ = ⟨a1 b2|ρ|b1 a2⟩
    ev = np.linalg.eigvalsh(rhoPT)
    return float(max(0.0, (np.sum(np.abs(ev)) - 1.0) / 2.0))


def t_do_polowy(S, delta_tau):
    """Czas (w j. czasu = tyknięcia·τ) osiągnięcia S = ½·ln 2 na siatce dyskretnej."""
    i = np.argmax(S >= M.LN2 / 2)
    if i == 0 or S[i] < M.LN2 / 2:
        return np.inf
    t0, t1 = (i - 1) * delta_tau, i * delta_tau
    s0, s1 = S[i - 1], S[i]
    return t0 + (M.LN2 / 2 - s0) * (t1 - t0) / (s1 - s0)


def t_do_polowy_ciggly(gamma0, fb, t_scale=0.5, t_max=20.0):
    """Czas osiągnięcia ½·ln 2 z ciągłego rozwiązania (solve_ivp) — dokładny."""
    t, S = symuluj_feedback_ciggly(gamma0, fb, t_max, t_scale=t_scale, n_out=20000)
    i = np.argmax(S >= M.LN2 / 2)
    if i == 0 or S[i] < M.LN2 / 2:
        return np.inf
    t0, t1 = t[i - 1], t[i]
    s0, s1 = S[i - 1], S[i]
    return t0 + (M.LN2 / 2 - s0) * (t1 - t0) / (s1 - s0)


# -----------------------------------------------------------------------------
#  R4 — N=3: JAWNE SEKTORY j=3/2 ⊕ 2×j=1/2 (STANY CIEMNE / SUBRADIANTNE)
# -----------------------------------------------------------------------------
def baza_N3():
    """
    Jawna baza sektorów dla 3 kubitów (kąpiel kolektywna, γ_φ=0):
      j=3/2 (4 stany symetryczne, termalizuje do 𝟙₄/4  ⇒  S(∞)=ln 4)
      j=1/2 (2 kopie × 2 stany; każda termalizuje do 𝟙₂/2  ⇒  S(∞)=ln 2 — „czapka”)
    Stany j=1/2 nie są w pełni ciemne (brak sektora j=0 dla N=3), ale są
    SUBRADIANTNE: kąpiel kolektywna nie wyprowadza ich poza 1 bit entropii.
    Konwencja: |1⟩ = stan wzbudzony, |0⟩ = podstawowy (jak w rdzeniu).
    """
    b = {}
    for i in range(8):
        s = format(i, "03b")
        v = np.zeros(8, complex); v[i] = 1.0; b[s] = v

    def nrm(v):
        return v / np.sqrt(v @ v.conj())

    sector = [
        ("j=3/2 · symetryczny", "|3/2,−3/2⟩ = |111⟩", b["111"]),
        ("j=3/2 · symetryczny", "|3/2,−1/2⟩ = (|110⟩+|101⟩+|011⟩)/√3",
         nrm(b["110"] + b["101"] + b["011"])),
        ("j=3/2 · symetryczny", "|3/2,+1/2⟩ = (|100⟩+|010⟩+|001⟩)/√3",
         nrm(b["100"] + b["010"] + b["001"])),
        ("j=3/2 · symetryczny", "|3/2,+3/2⟩ = |000⟩", b["000"]),
        ("j=1/2 · kopia A", "|A,+⟩ = (|101⟩+|011⟩−2|110⟩)/√6",
         nrm(b["101"] + b["011"] - 2 * b["110"])),
        ("j=1/2 · kopia A", "|A,−⟩ = (|100⟩+|010⟩−2|001⟩)/√6",
         nrm(b["100"] + b["010"] - 2 * b["001"])),
        ("j=1/2 · kopia B", "|B,+⟩ = |1⟩⊗|S⟩₂₃ = (|101⟩−|011⟩)/√2",
         nrm(b["101"] - b["011"])),
        ("j=1/2 · kopia B", "|B,−⟩ = |0⟩⊗|S⟩₂₃ = (|100⟩−|010⟩)/√2",
         nrm(b["100"] - b["010"])),
    ]
    return b, sector


def symuluj_N3(gamma, wektor, n=6000, gamma_phi=0.0):
    """3 kubity we wspólnej kąpieli (pełna przestrzeń 8-wymiarowa)."""
    return symuluj_wspolne(gamma, np.outer(wektor, wektor.conj()), N=3,
                           gamma_phi=gamma_phi, n=n)


# -----------------------------------------------------------------------------
#  R5 — PEŁNA ENTROPIA MAKRO (N kubitów)
# -----------------------------------------------------------------------------
def entropia_makro(N, n=6000):
    """
    Pełna entropia makro dla N kubitów:
      • niezależne kąpiele: S(∞) = N·ln 2 (ekstensywność, ln dim = N ln 2)
      • wspólna kąpiel, start |1…1⟩ (czysty sektor symetryczny, j=N/2):
        S(∞) = ln(N+1) — kąpiel kolektywna „widzi” tylko sektor symetryczny.
    Zwraca: (S_ind∞, S_kol∞, P_kol∞, t90_ind, t90_kol).
    """
    S_ind, _ = symuluj_niezalezne(M.GAMMA_B, N, gamma_phi=0.0, n=n)
    v = np.zeros(2 ** N, complex); v[2 ** N - 1] = 1.0
    S_kol, P_kol, _, _ = symuluj_wspolne(M.GAMMA_B, np.outer(v, v.conj()), N=N,
                                         gamma_phi=0.0, n=n)
    return (S_ind[-1], S_kol[-1], P_kol[-1],
            czasy_90(S_ind, N * M.LN2), czasy_90(S_kol, np.log(N + 1)))


# -----------------------------------------------------------------------------
#  R6 — „GORĄCY WIELKI WYBUCH" JAKO WARUNEK POCZĄTKOWY W R3
# -----------------------------------------------------------------------------
def symuluj_wielki_wybuch(gamma0, eta0, etaB, fb, n=600, t_scale=0.5,
                          delta_tau=M.DELTA_TAU):
    """
    Gorący start + zimna kąpiel ⇒ entropia MALEJE: zegar biegnie wstecz.

      ρ(0) — termiczne przy η0 (η0 ≈ 1 ⇒ S(0) ≈ ln 2, „gorący Wielki Wybuch”)
      kąpiel — Gibbsa przy ηB < η0 (zimna, plateau S(∞) = H(1/(1+ηB)) < S(0))
      tempo   — γ_eff(n) = γ₀·fb(u),  u = (S(0)−S(n))/t_scale  (R3)

    Zamknięcie: rp ≡ 0 (stan początkowy diagonalny), rz ewoluuje
    rz ← r_eq + (rz−r_eq)·e^(−2γ_eff·τ).  Zegar podpisany: T(n)=S(n) — maleje.
    """
    r_eq = r_eq_termiczny(etaB)
    rz = r_eq_termiczny(eta0)
    rp = 0.0
    S0 = S_eq_termiczna(eta0)
    S = np.zeros(n); G = np.zeros(n); T = np.zeros(n)
    for i in range(n):
        r = np.sqrt(rz * rz + rp * rp)
        S[i] = H_bin((1.0 + r) / 2.0)
        u = max(0.0, (S0 - S[i]) / t_scale)
        g = gamma0 * fb(u)
        G[i] = g
        rz = r_eq + (rz - r_eq) * np.exp(-2.0 * g * delta_tau)
        rp *= np.exp(-5.0 * g * delta_tau)
        if i > 0:
            T[i] = T[i - 1] + (S[i] - S[i - 1])     # ujemne ΔS ⇒ czas wstecz
    return S, G, T, S0, S_eq_termiczna(etaB)


def zegar_wstecz(dS, ds=M.DELTA_S_Q, seed=None):
    """
    Kwantowy zegar przy entropii MALEJĄCEJ: ΔS_n < 0 ⇒ Δt_n = k_n·δs < 0
    (k_n ~ Poisson(|ΔS_n|/δs)); T(n) = Σ Δt_k maleje — „czas płynie wstecz”,
    a przy |ΔS_n| < δs zegar stoi (czkanie).
    """
    rng = np.random.default_rng(seed)
    k = rng.poisson(np.maximum(-np.diff(np.concatenate([[dS[0]], dS])), 0.0) / ds)
    dt = -k * ds
    T = np.cumsum(dt)
    return T, dt, k


def t_do_polowy_wstecz(S, delta_tau, S0, Seq):
    """Czas (dyskretny) osiągnięcia S = (S0+Seq)/2 przy ewolucji w dół."""
    poziom = (S0 + Seq) / 2.0
    i = np.argmax(S <= poziom)
    if i == 0 or S[i] > poziom:
        return np.inf
    t0, t1 = (i - 1) * delta_tau, i * delta_tau
    s0, s1 = S[i - 1], S[i]
    return t0 + (poziom - s0) * (t1 - t0) / (s1 - s0)


# -----------------------------------------------------------------------------
#  R7 — KĄPIEL KOLEKTYWNA DLA LOSOWYCH (NIE-SYMETRYCZNYCH) STANÓW
#  Klucz: kąpiel kolektywna termalizuje wewnątrz sektorów j; koherencje między
#  kopiami tego samego j PRZEŻYWAJĄ (identyczna dynamika), a lokalna dekoherencja
#  (γ_φ) miesza sektory i domyka do pełnej termalizacji N·ln 2.
# -----------------------------------------------------------------------------
def stan_haara(rng, dim):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return v / np.linalg.norm(v)


def sektor_koncowy_N3(rho, gamma_phi=0.0):
    """Stan końcowy (n=6000) w bazie sektorów dla N=3."""
    _, sector = baza_N3()
    U = np.column_stack([v for _, _, v in sector])
    out = symuluj_wspolne(M.GAMMA_B, rho, N=3, gamma_phi=gamma_phi,
                          n=6000, zwroc_stan=True)
    return U.conj().T @ out[4] @ U


def figura_R7():
    """Losowe stany: plateau między ln(N+1) a N·ln2; γφ domyka do pełna."""
    b, sector = baza_N3()
    rng = np.random.default_rng(7)
    ket_111 = b["111"]; ket_100 = b["100"]; ket_1S = sector[6][2]
    stany3 = {
        "|111⟩ (sym.)": ket_111,
        "|100⟩ (produkt)": ket_100,
        "|1⟩⊗|S⟩ (j=1/2)": ket_1S,
        "losowy #1": stan_haara(rng, 8),
        "losowy #2": stan_haara(rng, 8),
    }
    ket_1111 = np.zeros(16, complex); ket_1111[15] = 1.0
    los4_1 = stan_haara(rng, 16)
    los4_2 = stan_haara(rng, 16)

    n_l = 6000
    wyn3, wyn3d = {}, {}
    for naz, v in stany3.items():
        rho = np.outer(v, v.conj())
        S0, *_ = symuluj_wspolne(M.GAMMA_B, rho, N=3, gamma_phi=0.0, n=n_l)
        Sd, *_ = symuluj_wspolne(M.GAMMA_B, rho, N=3, gamma_phi=M.GAMMA_B, n=n_l)
        wyn3[naz] = S0[-1]; wyn3d[naz] = Sd[-1]
    wyn4 = {}
    for naz, v in [("|1111⟩ (sym.)", ket_1111), ("losowy #1", los4_1), ("losowy #2", los4_2)]:
        rho = np.outer(v, v.conj())
        S0, *_ = symuluj_wspolne(M.GAMMA_B, rho, N=4, gamma_phi=0.0, n=n_l)
        Sd, *_ = symuluj_wspolne(M.GAMMA_B, rho, N=4, gamma_phi=M.GAMMA_B, n=n_l)
        wyn4[naz] = (S0[-1], Sd[-1])

    rho100 = np.outer(ket_100, ket_100.conj())
    rho_sec = sektor_koncowy_N3(rho100, gamma_phi=0.0)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    n = np.arange(M.N_TICKS)
    sl = slice(0, M.N_TICKS)

    ax = axs[0, 0]
    cols = {"|111⟩ (sym.)": C_A, "|100⟩ (produkt)": "#16a085", "|1⟩⊗|S⟩ (j=1/2)": C_V,
            "losowy #1": "#e67e22", "losowy #2": "#1a5276"}
    for naz, v in stany3.items():
        rho = np.outer(v, v.conj())
        S, *_ = symuluj_wspolne(M.GAMMA_B, rho, N=3, gamma_phi=0.0, n=M.N_TICKS)
        ax.plot(n, S[sl], lw=2, color=cols[naz], label=f"{naz} → {wyn3[naz]:.3f}")
    ax.axhline(np.log(4), color=C_G, ls=":", lw=1)
    ax.axhline(3 * M.LN2, color=C_G, ls="--", lw=1)
    ax.text(5, np.log(4) + 0.02, "ln 4 (sektor j=3/2, min.)", color=C_G, fontsize=8)
    ax.text(5, 3 * M.LN2 + 0.02, "3·ln 2 (pełna termalizacja)", color=C_G, fontsize=8)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("N=3, kąpiel kolektywna (γ_φ=0): stany nie-symetryczne wyżej niż ln 4")
    ax.legend(fontsize=7.5)

    ax = axs[0, 1]
    for naz, (s0, sd) in wyn4.items():
        ax.bar([0, 1], [s0, sd], width=0.32,
               color=C_A if "1111" in naz else "#e67e22",
               label=f"{naz}: {s0:.3f} → {sd:.3f}")
    ax.axhline(np.log(5), color=C_G, ls=":", lw=1)
    ax.axhline(4 * M.LN2, color=C_G, ls="--", lw=1)
    ax.text(0.4, np.log(5) + 0.04, "ln 5", color=C_G, fontsize=9)
    ax.text(0.4, 4 * M.LN2 + 0.04, "4·ln 2", color=C_G, fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["γ_φ = 0", "γ_φ = γ"])
    ax.set_ylabel("S(∞) [nat]")
    ax.set_title("N=4: lokalna dekoherencja domyka do 4·ln 2")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    im = ax.imshow(np.abs(rho_sec), cmap="viridis", vmin=0, vmax=0.26)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(["|3/2⟩"] * 4 + ["|A⟩"] * 2 + ["|B⟩"] * 2, fontsize=8)
    ax.set_yticklabels(["|3/2⟩"] * 4 + ["|A⟩"] * 2 + ["|B⟩"] * 2, fontsize=8)
    ax.set_title("|100⟩, γ_φ=0: macierz w bazie sektorów — koherencje A↔B przeżywają")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[1, 1]
    nazwy = list(wyn3.keys())
    x = np.arange(len(nazwy)); w = 0.34
    ax.bar(x - w / 2, [wyn3[nz] for nz in nazwy], w, color=C_A, label="γ_φ = 0")
    ax.bar(x + w / 2, [wyn3d[nz] for nz in nazwy], w, color="#2e86c1", label="γ_φ = γ")
    ax.axhline(np.log(4), color=C_G, ls=":", lw=1)
    ax.axhline(3 * M.LN2, color=C_G, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(nazwy, fontsize=8, rotation=18)
    ax.set_ylabel("S(∞) [nat]")
    ax.set_title("N=3: γ_φ odblokowuje pełną entropię 3·ln 2")
    ax.legend(fontsize=8)

    fig.suptitle("R7 — kąpiel kolektywna dla losowych stanów: koherencje sektorowe blokują entropię",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR7_losowe.png", bbox_inches="tight")
    plt.close(fig)
    CACHE['R7'] = dict(wyn3=wyn3, wyn3d=wyn3d, wyn4=wyn4, rho_sec=rho_sec)
    return CACHE['R7']


# -----------------------------------------------------------------------------
#  R8 — CYKL: WIELKI WYBUCH → EKSPANSJA → OCHŁODZENIE → WIELKI KOLAPS
#  Zegar dwustronny: T(n)=S(n) (wskazówka entropii, obraca się wstecz w połowie
#  cyklu) oraz τ(n)=Σ|ΔS| (wskazówka upływu — zawsze do przodu).
# -----------------------------------------------------------------------------
def symuluj_cykl(gamma, eta_min, n_cyc, n_total, delta_tau=M.DELTA_TAU):
    """
    Kubit z kąpielą o oscylującej temperaturze:
      η(n) = 1 − (1−η_min)·sin²(π·n/n_cyc)
    Start: η(0)=1 (gorący Wielki Wybuch, S≈ln 2); potem ochłodzenie
    (ekspansja, S maleje), minimum (maksymalna ekspansja), ogrzewanie
    (kolaps, S rośnie z powrotem do ln 2).
    Zwraca S, dS, η, T_signed (S−S(0)), T_abs (Σ|ΔS|).
    """
    n = n_total
    S = np.zeros(n); dS = np.zeros(n); eta = np.zeros(n)
    r_eq = lambda e: (1.0 - e) / (1.0 + e)
    rz = r_eq(1.0); rp = 0.0
    for i in range(n):
        e = 1.0 - (1.0 - eta_min) * np.sin(np.pi * i / n_cyc) ** 2
        eta[i] = e
        req = r_eq(e)
        rz = req + (rz - req) * np.exp(-2.0 * gamma * delta_tau)
        rp *= np.exp(-5.0 * gamma * delta_tau)
        r = np.sqrt(rz * rz + rp * rp)
        S[i] = H_bin((1.0 + r) / 2.0)
        if i > 0:
            dS[i] = S[i] - S[i - 1]
    T_signed = np.cumsum(dS)
    T_abs = np.cumsum(np.abs(dS))
    return S, dS, eta, T_signed, T_abs


CACHE = {}


def figura_R8():
    """Cykl BB → ekspansja → ochłodzenie → Wielki Kolaps; dwustronny czas."""
    gamma = 0.05
    eta_min = 0.15
    n_cyc, n_tot = 300, 300          # jeden pełny cykl: gorąco → zimno → gorąco
    S, dS, eta, Ts, Ta = symuluj_cykl(gamma, eta_min, n_cyc, n_tot)
    n = np.arange(n_tot)
    S0 = S[0]; Smin = S.min(); imin = int(np.argmin(S))

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axs[0, 0]
    ax.plot(n, S, color="#c0392b", lw=2, label="S(n) — entropia wszechświata")
    ax2 = ax.twinx()
    ax2.plot(n, eta, color="#2471a3", lw=1.4, ls="--", label="η(n) — temperatura kąpieli")
    ax2.set_ylabel("η (temperatura)", color="#2471a3")
    ax2.tick_params(axis="y", colors="#2471a3")
    ax.axvline(imin, color=C_G, ls=":", lw=1)
    ax.text(10, 0.66, "Wielki Wybuch\n(S ≈ ln 2)", fontsize=9, color="#c0392b")
    ax.text(imin + 10, 0.52, "maks. ekspansja\n(całkowite ochłodzenie)", fontsize=9, color="#1a5276")
    ax.text(n_tot - 95, 0.66, "Wielki Kolaps\n(powrót do gorąca)", fontsize=9, color="#c0392b")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("Cykl: Wybuch (gorąco) → ekspansja (chłodzenie) → Kolaps (ogrzewanie)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center")

    ax = axs[0, 1]
    ax.plot(n, Ts, color="#8e44ad", lw=2, label="T(n) = S(n) − S(0) — wskazówka entropii (dwustronna)")
    ax.plot(n, Ta, color="#1a5276", lw=2, label="τ(n) = Σ|ΔS| — upływ (zawsze do przodu)")
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.axvline(imin, color=C_G, ls=":", lw=1)
    ax.text(imin + 10, Ta.min() * 0.3, "zwrot strzałki czasu", fontsize=9, color="#8e44ad")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("T / τ [nat]")
    ax.set_title("Dwie wskazówki zegara kosmicznego: entropia (wstecz) i upływ (naprzód)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    kolory = ["#c0392b" if d >= 0 else "#2471a3" for d in dS]
    ax.bar(n, dS, color=kolory, width=1.0)
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.axvline(imin, color=C_G, ls=":", lw=1)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("ΔS_n [nat/tyknięcie]")
    ax.set_title("Tempo czasu: czerwony = do przodu, niebieski = wstecz; przy zwrocie ΔS→0 (czkanie)")
    ax.text(5, dS.max() * 0.75, "czas do przodu", color="#c0392b", fontsize=9)
    ax.text(imin + 20, dS.min() * 0.75, "czas wstecz", color="#2471a3", fontsize=9)

    ax = axs[1, 1]
    S2, dS2, _, _, _ = symuluj_cykl(gamma, eta_min, n_cyc, 2 * n_cyc)
    n2 = np.arange(2 * n_cyc)
    ax.plot(n2, S2, color="#c0392b", lw=2, label="S(n) — cykl powtórzony")
    rng = np.random.default_rng(5)
    dt_q = np.zeros(2 * n_cyc)
    for i in range(1, 2 * n_cyc):
        mu = abs(dS2[i]) / M.DELTA_S_Q
        k = rng.poisson(mu)
        dt_q[i] = np.sign(dS2[i]) * k * M.DELTA_S_Q
    ax.plot(n2, np.cumsum(dt_q), color="#16a085", lw=1.2, alpha=0.8,
            label="realizacja kwantowego zegara (dwustronna)")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S [nat]")
    ax.set_title("Wszechświat cykliczny: entropia wraca do ln 2 — czas jako pętla")
    ax.legend(fontsize=8)

    fig.suptitle("R8 — Wielki Wybuch → ekspansja → ochłodzenie → Wielki Kolaps: czas dwustronny",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR8_cykl.png", bbox_inches="tight")
    plt.close(fig)
    CACHE['R8'] = dict(S0=S0, Smin=Smin, imin=imin, Ts=Ts, Ta=Ta, dS=dS,
                        budzet=S0 - Smin, t_abs_total=Ta[-1], eta_min=eta_min,
                        S=S, eta=eta, n_tot=n_tot, n_cyc=n_cyc)
    return CACHE['R8']


# -----------------------------------------------------------------------------
#  R9 — KWANTOWY ZEGAR (SUPERTWARDY): CZAS JAKO OPERATOR
#  Kubit + oscylator-zegar (próżnia). „Kranik" σ₋⊗b† kopiuje każdą de-ekscytację
#  do zegara (jednokierunkowo). Wskazanie zegara ⟨n⟩ jest kwantowe: ma rozkład
#  p_n i nieoznaczoność Δn; silniejszy zegar mierzy czas precyzyjniej, ale
#  mocniej zaburza ewolucję wszechświata (back-action) — kompromis zegara.
# -----------------------------------------------------------------------------
def kwantowy_zegar(gt, MLEV=30, TICKS=400, gamma=0.02, gphi=0.04,
                   omega=M.OMEGA, delta_tau=M.DELTA_TAU,
                   alpha=None, kappa=0.0):
    """
    Równanie master na kubit⊗zegar (Crank–Nicolson, LU).
    Jumpy: σ₋⊗b† (kranik, γ_t), σ₋ (kąpiel, γ), σ₊ (kąpiel, γ), σ_z⊗1 (γ_φ),
    oraz opcjonalnie dekoherencja zegara κ·D[b†b] (kappa).
    Start zegara: próżnia (alpha=None) lub stan koherentny |α⟩.
    Zwraca słownik: S_sys, nbar, dn, pn, I, coh (średnie koherencje zegara),
    pur (czystość zredukowanego zegara).
    """
    from scipy.linalg import lu_factor, lu_solve
    import math as _math
    ML = MLEV; tau = delta_tau; n_ticks = TICKS
    I2 = np.eye(2, dtype=complex); sz, sp_, sm_ = M.operatory()
    a = np.diag(np.sqrt(np.arange(1, ML)), 1); ad = a.conj().T
    IM = np.eye(ML, dtype=complex); Nc = ad @ a
    jumps = [np.kron(sm_, ad), np.kron(sm_, IM), np.kron(sp_, IM), np.kron(sz, IM)]
    rates = [gt, gamma, gamma, gphi]
    if kappa > 0:
        jumps.append(np.kron(I2, Nc)); rates.append(kappa)
    H = 0.5 * omega * np.kron(sz, IM)

    def ptrace_sys(r):
        return np.array([[np.trace(r[0:ML, 0:ML]), np.trace(r[0:ML, ML:2 * ML])],
                         [np.trace(r[ML:2 * ML, 0:ML]), np.trace(r[ML:2 * ML, ML:2 * ML])]])

    def ptrace_clock(r):
        rc = np.zeros((ML, ML), complex)
        for c in range(ML):
            for cp in range(ML):
                rc[c, cp] = r[c, cp] + r[ML + c, ML + cp]
        return rc

    d = 2 * ML
    Id = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for J, r in zip(jumps, rates):
        Jd = J.conj().T; JJ = Jd @ J
        L += r * (np.kron(J, J.conj()) - 0.5 * (np.kron(JJ, Id) + np.kron(Id, JJ.T)))
    IL = np.eye(d * d, dtype=complex)
    lu, piv = lu_factor(IL - 0.5 * tau * L)
    B = IL + 0.5 * tau * L

    psi = np.array([np.cos(M.THETA0 / 2), np.exp(1j * M.PHI0) * np.sin(M.THETA0 / 2)])
    if alpha is None:
        rho_c = np.zeros((ML, ML), complex); rho_c[0, 0] = 1
    else:
        v = np.exp(-abs(alpha) ** 2 / 2) * np.array(
            [alpha ** n / _math.sqrt(_math.factorial(n)) for n in range(ML)])
        v = v / np.linalg.norm(v)
        rho_c = np.outer(v, v.conj())
    rho = np.kron(np.outer(psi, psi.conj()), rho_c)

    def S(r):
        ev = np.linalg.eigvalsh((r + r.conj().T) / 2); ev = ev[ev > 1e-15]
        return -np.sum(ev * np.log(ev))

    out = dict(S_sys=np.zeros(n_ticks), nbar=np.zeros(n_ticks), dn=np.zeros(n_ticks),
               I=np.zeros(n_ticks), pn=np.zeros((n_ticks, ML)),
               coh=np.zeros(n_ticks), pur=np.zeros(n_ticks))
    for k in range(n_ticks):
        rs = ptrace_sys(rho); rc = ptrace_clock(rho)
        pn = np.real(np.diag(rc)); pn = np.clip(pn, 0, None); pn = pn / pn.sum()
        nb = np.sum(np.arange(ML) * pn)
        out["S_sys"][k] = S(rs)
        out["nbar"][k] = nb
        out["dn"][k] = np.sqrt(max(0.0, np.sum((np.arange(ML) - nb) ** 2 * pn)))
        out["pn"][k] = pn
        out["I"][k] = S(rs) + S(rc) - S(rho)
        out["coh"][k] = np.sum(np.abs(np.diag(rc, 1))) / (ML - 1)
        out["pur"][k] = np.real(np.trace(rc @ rc))
        rho = np.asarray(lu_solve((lu, piv), B @ rho.flatten())).reshape(d, d)
    return out


def figura_R10():
    """Kwantowy zegar z koherencjami: start koherentny + dekoherencja zegara."""
    n_ticks = 400
    n = np.arange(n_ticks)
    gt = 0.01
    z_vac = kwantowy_zegar(gt, TICKS=n_ticks)                       # próżnia κ=0
    z_coh = kwantowy_zegar(gt, TICKS=n_ticks, alpha=1.5)            # koherentny κ=0
    z_kap = kwantowy_zegar(gt, TICKS=n_ticks, alpha=1.5, kappa=0.3) # koherentny + κ

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axs[0, 0]
    ax.plot(n, z_vac["S_sys"], color="#7f8c8d", lw=2, ls="--", label="S_sys — próżnia")
    ax.plot(n, z_coh["S_sys"], color="#c0392b", lw=2, label="S_sys — koherentny |α=1.5⟩")
    ax.axhline(M.LN2, color="#c0392b", ls=":", lw=1)
    ax2 = ax.twinx()
    ax2.plot(n, z_coh["nbar"], color="#1a5276", lw=2, label="⟨n⟩ — odczyt koherentnego")
    ax2.plot(n, z_vac["nbar"], color="#8b98a5", lw=1.4, ls="--", label="⟨n⟩ — próżnia")
    ax2.set_ylabel("⟨n⟩ — wskazanie zegara", color="#1a5276")
    ax2.tick_params(axis="y", colors="#1a5276")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S [nat]")
    ax.set_title("Koherentny zegar mierzy mocniej (stymulacja b†) i mocniej zaburza")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")

    ax = axs[0, 1]
    for k, (t_, c, lab) in enumerate([(50, "#2471a3", "n=50"), (150, "#e67e22", "n=150"),
                                      (399, "#c0392b", "n=400")]):
        pn = z_coh["pn"][t_]
        nn = np.arange(len(pn))
        ax.bar(nn + k * 0.28, pn, width=0.28, color=c, alpha=0.85,
               label=f"{lab}: ⟨n⟩={z_coh['nbar'][t_]:.2f}, Δn={z_coh['dn'][t_]:.2f}")
    ax.set_xlim(-0.5, 12)
    ax.set_xlabel("n — wskazanie zegara"); ax.set_ylabel("p_n")
    ax.set_title("Fala czasu koherentnego zegara: p_n przesuwa się (Poisson) z koherencjami")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.semilogy(n, z_coh["coh"] + 1e-6, color="#c0392b", lw=2, label="koherencje, κ = 0")
    ax.semilogy(n, z_kap["coh"] + 1e-6, color="#1a5276", lw=2, label="koherencje, κ = 0.3")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("śr. koherencje zegara ⟨|ρ_c[n,n+1]|⟩")
    ax.set_title("Dekoherencja zegara κ·D[b†b] niszczy kwantową fazę czasu")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    sceny = [("próżnia κ=0", z_vac, "#7f8c8d"), ("koherentny κ=0", z_coh, "#c0392b"),
             ("koherentny κ=0.3", z_kap, "#1a5276")]
    x = np.arange(3); w = 0.3
    dev = [abs(s["S_sys"][-1] - M.LN2) for _, s, _ in sceny]
    rel = [s["dn"][-1] / max(s["nbar"][-1], 1e-9) for _, s, _ in sceny]
    Iend = [s["I"][-1] for _, s, _ in sceny]
    ax.bar(x - w, dev, w, color="#c0392b", label="|back-action| = |S∞ − ln 2|")
    ax2 = ax.twinx()
    ax2.bar(x, rel, w, color="#8e44ad", alpha=0.75, label="Δn/⟨n⟩ (nieoznaczność)")
    ax2.bar(x + w, Iend, w, color="#16a085", alpha=0.6, label="I(wszechświat;zegar)")
    ax2.set_ylabel("Δn/⟨n⟩ oraz I", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sceny], fontsize=8)
    ax.set_ylabel("|S∞ − ln 2|")
    ax.set_title("Kompromis zegara: koherentny = precyzyjniejszy, ale mocniej zaburza; κ klasycyzuje czas")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")

    fig.suptitle("R10 — kwantowy zegar z koherencjami: faza czasu i jej dekoherencja",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR10_zegar_koherencje.png", bbox_inches="tight")
    plt.close(fig)

    d = dict(gt=gt, sceny={})
    for naz, z, c in sceny:
        d["sceny"][naz] = dict(
            S_end=z["S_sys"][-1], dev=z["S_sys"][-1] - M.LN2,
            nbar=z["nbar"][-1], dn=z["dn"][-1],
            rel=z["dn"][-1] / max(z["nbar"][-1], 1e-9),
            coh=z["coh"][-1], I=z["I"][-1], pur=z["pur"][-1],
            nbar100=z["nbar"][100], coh100=z["coh"][100],
            pn_end=[round(float(v), 6) for v in z["pn"][-1].tolist()])
    d["nbar_serie_coh"] = [round(float(v), 4) for v in z_coh["nbar"].tolist()]
    d["S_serie_coh"] = [round(float(v), 4) for v in z_coh["S_sys"].tolist()]
    d["coh_serie_coh"] = [round(float(v), 6) for v in z_coh["coh"].tolist()]
    d["coh_serie_kap"] = [round(float(v), 6) for v in z_kap["coh"].tolist()]
    CACHE["R10"] = d
    return d


# -----------------------------------------------------------------------------
#  R11 — ENTROPIA MAKRO Z LOSOWYMI STANAMI PRZY POŚREDNIM γ_φ (SWEEP)
#  „Odblokowanie”: lokalna dekoherencja miesza sektory; S(∞) rośnie od plateau
#  (zależnego od stanu) do N·ln 2. Skala przejścia: γ_φ ~ O(γ).
# -----------------------------------------------------------------------------
def przebieg_gphi(N, v, gph, n=6000):
    """Ewolucja S(n) dla stanu v przy γφ (do nasycenia lub do n kroków)."""
    rho = np.outer(v, v.conj())
    S, *_ = symuluj_wspolne(M.GAMMA_B, rho, N=N, gamma_phi=gph, n=n)
    return S


def czas_odblokowania(S, cel, delta_tau=M.DELTA_TAU):
    """Czas (w j. czasu) osiągnięcia 90% poziomu `cel` (liczonego od S(0))."""
    poziom = 0.90 * cel
    i = np.argmax(S >= poziom)
    if i == 0 or S[i] < poziom:
        return np.inf
    t0, t1 = (i - 1) * delta_tau, i * delta_tau
    s0, s1 = S[i - 1], S[i]
    return t0 + (poziom - s0) * (t1 - t0) / (s1 - s0)


def figura_R11():
    """
    Odblokowanie entropii przez lokalną dekoherencję.
    Kluczowe prawo: dla KAŻDEGO γφ > 0 entropia w końcu osiąga N·ln 2
    (blokada przy γφ = 0 jest dokładna), a czas odblokowania skaluje się
    jak τ ~ 1/γφ (mieszanie sektorów ma tempo ~ γφ).
    """
    b, sector = baza_N3()
    rng = np.random.default_rng(11)
    ket_100 = b["100"]
    los3 = stan_haara(rng, 8)
    ket_1111 = np.zeros(16, complex); ket_1111[15] = 1.0
    los4 = stan_haara(rng, 16)

    # (a) dwuetapowa relaksacja: S(n) dla kilku γφ (oś log n)
    gphis_a = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    n_a = 20000
    krzywe3 = {}
    for gph in gphis_a:
        krzywe3[gph] = przebieg_gphi(3, los3, gph, n=n_a)

    # (b) prawo skalowania: czas odblokowania τ90(γφ) ∝ 1/γφ
    gphis_b = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]
    cel3 = 3 * M.LN2
    tau90_3 = {}
    krzywe_js = {}                       # zdownsemplowane S(n) do demo (N=3, losowy)
    for gph in gphis_b:
        n_eff = int(min(120000, max(4000, 8.0 / (gph * M.DELTA_TAU))))
        S = przebieg_gphi(3, los3, gph, n=n_eff)
        tau90_3[gph] = czas_odblokowania(S, cel3)
        idx = np.unique(np.geomspace(1, len(S) - 1, 180).astype(int))
        idx = np.concatenate([[0], idx])
        krzywe_js[gph] = [[float(i * M.DELTA_TAU), float(S[i])] for i in idx]
    # N=4 (jedna kopia losowa + |1111⟩)
    tau90_4 = {}
    cel4 = 4 * M.LN2
    for gph in [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]:
        n_eff = int(min(60000, max(3000, 6.0 / (gph * M.DELTA_TAU))))
        S = przebieg_gphi(4, los4, gph, n=n_eff)
        tau90_4[gph] = czas_odblokowania(S, cel4)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axs[0, 0]
    for gph, c in zip(gphis_a, ["#8b98a5", "#e67e22", "#16a085", "#2471a3", "#c0392b"]):
        S = krzywe3[gph]
        ax.semilogx(np.arange(len(S)) * M.DELTA_TAU, S, lw=2, color=c,
                    label=f"γ_φ = {gph:g}")
    ax.axhline(cel3, color=C_G, ls="--", lw=1)
    ax.text(0.6, cel3 + 0.02, "3·ln 2 (pełna termalizacja)", color=C_G, fontsize=9)
    ax.set_xlabel("t (j. czasu, skala log)")
    ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("N=3 (losowy): dwuetapowa relaksacja — szybka do plateau, wolna (γ_φ) do 3·ln 2")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    for gph, c in zip([0.0, 1e-3, 1e-2, 0.1], ["#8b98a5", "#e67e22", "#2471a3", "#c0392b"]):
        S = krzywe3[gph]
        ax.semilogx(np.arange(len(S)) * M.DELTA_TAU, S, lw=2, color=c, label=f"γ_φ = {gph:g}")
    ax.axhline(cel3, color=C_G, ls="--", lw=1)
    ax.text(0.6, cel3 + 0.02, "3·ln 2", color=C_G, fontsize=9)
    ax.set_xlabel("t (skala log)")
    ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("Widać plateau ~1.5 nat przed odblokowaniem (γ_φ małe)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    gx = np.array(list(tau90_3.keys())) / M.GAMMA_B
    ty = np.array([tau90_3[g] for g in tau90_3])
    ax.loglog(gx, ty, "o-", color="#c0392b", lw=2, label="N=3, losowy")
    gx4 = np.array(list(tau90_4.keys())) / M.GAMMA_B
    ty4 = np.array([tau90_4[g] for g in tau90_4])
    ax.loglog(gx4, ty4, "s-", color="#1a5276", lw=2, label="N=4, losowy")
    # dopasowanie nachylenia −1 w reżimie asymptotycznym (z dala od podłogi)
    mask3 = ty > 3.0 * ty.min()
    mask4 = ty4 > 3.0 * ty4.min()
    m3 = np.polyfit(np.log(gx[mask3]), np.log(ty[mask3]), 1)[0] if mask3.sum() >= 2 else np.nan
    m4 = np.polyfit(np.log(gx4[mask4]), np.log(ty4[mask4]), 1)[0] if mask4.sum() >= 2 else np.nan
    ax.loglog(gx, 2.0 / gx, color=C_G, ls=":", lw=1.4, label="τ ∝ 1/γ_φ")
    ax.set_xlabel("γ_φ / γ")
    ax.set_ylabel("τ90 — czas do 90% N·ln 2 [j. czasu]")
    ax.set_title(f"Prawo odblokowania: τ90 ∝ 1/γ_φ (nachylenia {m3:.2f}, {m4:.2f}); podłoga = czas kolektywny")
    ax.legend(fontsize=8)
    ax.annotate("podłoga: relaksacja kolektywna (~10)",
                xy=(gx[-1], ty[-1]), xytext=(0.35, 400), fontsize=9, color="#26384a",
                arrowprops=dict(arrowstyle="->", color="#26384a", lw=1))

    ax = axs[1, 1]
    # blokada: ile entropii „zablokowanej" przy γφ=0 (per stan)
    wyn0 = {}
    for naz, v in [("|111⟩", b["111"]), ("|100⟩", ket_100), ("losowy #1", los3)]:
        S = przebieg_gphi(3, v, 0.0, n=6000)
        wyn0[naz] = S[-1]
    nazwy = list(wyn0.keys())
    blokada = [cel3 - wyn0[n] for n in nazwy]
    odblok = [cel3 - wyn0[n] for n in nazwy]
    ax.bar(range(len(nazwy)), odblok, color=[C_A, "#16a085", "#e67e22"])
    ax.axhline(0, color=C_G, lw=1)
    ax.set_xticks(range(len(nazwy)))
    ax.set_xticklabels(nazwy, fontsize=9)
    ax.set_ylabel("zablokowana entropia 3·ln2 − S(γ_φ=0) [nat]")
    ax.set_title("Blokada sektorowa (γ_φ=0): od stanu do stanu (|111⟩: ln 2, |100⟩: 0.52)")
    for i, b in enumerate(blokada):
        ax.text(i, b + 0.02, f"{b:.3f}", ha="center", fontsize=9, color="#26384a")

    fig.suptitle("R11 — odblokowanie entropii przez lokalną dekoherencję γ_φ (prawo τ ∝ 1/γ_φ)",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR11_gphi_sweep.png", bbox_inches="tight")
    plt.close(fig)
    d = dict(tau90_3=tau90_3, tau90_4=tau90_4, m3=float(m3), m4=float(m4),
             blokada={k: float(v) for k, v in wyn0.items()},
             cel3=cel3, cel4=cel4,
             krzywe_js={f"{g:g}": krzywe_js[g] for g in gphis_b},
             gphis_b=[float(g) for g in gphis_b])
    CACHE["R11"] = d
    return d


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------


def figura_R9():
    """Kwantowy zegar: czas jako operator z rozkładem, nieoznaczonością i back-action."""
    n_ticks = 400
    n = np.arange(n_ticks)
    gt_main = 0.01
    z = kwantowy_zegar(gt_main, TICKS=n_ticks)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axs[0, 0]
    ax.plot(n, z["S_sys"], color="#c0392b", lw=2, label="S_sys(n) — entropia wszechświata")
    ax.axhline(M.LN2, color="#c0392b", ls=":", lw=1)
    ax2 = ax.twinx()
    ax2.plot(n, z["nbar"], color="#1a5276", lw=2, label="⟨n⟩(n) — wskazanie zegara")
    ax2.set_ylabel("⟨n⟩ — kwanty w zegarze", color="#1a5276")
    ax2.tick_params(axis="y", colors="#1a5276")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S [nat]")
    ax.set_title(f"Czas = entropia, kwantowo: ⟨n⟩ śledzi S (γ_t = {gt_main})")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    ax = axs[0, 1]
    for k, (t_, c, lab) in enumerate([(50, "#2471a3", "n=50"), (150, "#e67e22", "n=150"),
                                      (399, "#c0392b", "n=400")]):
        pn = z["pn"][t_]
        nn = np.arange(len(pn))
        ax.bar(nn + k * 0.28, pn, width=0.28, color=c, alpha=0.85,
               label=f"{lab}: ⟨n⟩={z['nbar'][t_]:.2f}, Δn={z['dn'][t_]:.2f}")
    ax.set_xlim(-0.5, 8)
    ax.set_xlabel("n — wskazanie zegara (liczba kwantów)")
    ax.set_ylabel("p_n")
    ax.set_title("Kwantowa „fala czasu”: rozkład wskazań zegara p_n")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(n, z["dn"], color="#16a085", lw=2, label="Δn — nieoznaczoność czasu")
    ax.plot(n, np.sqrt(np.maximum(z["nbar"], 1e-9)), color="#16a085", ls="--", lw=1,
            label="√⟨n⟩ (wzorzec Poissona)")
    ax2 = ax.twinx()
    rel = z["dn"] / np.maximum(z["nbar"], 1e-9)
    ax2.plot(n, rel, color="#8e44ad", lw=2, label="Δn/⟨n⟩ — względna nieoznaczoność")
    ax2.set_ylabel("Δn/⟨n⟩", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("Δn")
    ax.set_title("Zegar się wyostrza: Δn rośnie jak √⟨n⟩, względna nieoznaczoność maleje")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    ax = axs[1, 1]
    gts = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    dev, relend = [], []
    for gt in gts:
        if gt == 0.0:
            S, _, _ = M.symuluj(M.GAMMA_B, n=n_ticks)
            dev.append(S[-1] - M.LN2); relend.append(0.0)
        else:
            zz = kwantowy_zegar(gt, TICKS=n_ticks)
            dev.append(zz["S_sys"][-1] - M.LN2)
            relend.append(zz["dn"][-1] / max(zz["nbar"][-1], 1e-9))
    ax2 = ax.twinx()
    ax.semilogx(gts[1:], [abs(x) for x in dev[1:]], "o-", color="#c0392b", lw=2,
                label="|back-action| = |S(∞) − ln 2|")
    ax2.semilogx(gts[1:], relend[1:], "s-", color="#8e44ad", lw=2,
                 label="Δn/⟨n⟩ — nieoznaczoność czasu")
    ax.set_xlabel("γ_t — siła zegara")
    ax.set_ylabel("|S(∞) − ln 2|", color="#c0392b")
    ax2.set_ylabel("Δn/⟨n⟩", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax.set_title("Kompromis zegara kwantowego: precyzja kosztem zaburzenia wszechświata")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

    fig.suptitle("R9 — kwantowy zegar: czas jako operator z nieoznaczonością i back-action",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR9_zegarkwantowy.png", bbox_inches="tight")
    plt.close(fig)
    CACHE['R9'] = dict(z=z, gt_main=gt_main, gts=gts, dev=dev, relend=relend)
    return CACHE['R9']


# -----------------------------------------------------------------------------
#  R13 — GRAWITACYJNA PRODUKCJA ENTROPII (Dwie kąpiele, NESS)
#  Promieniowanie (η_r, γ_r) + „kąpiel grawitacyjna" (η_g, γ_g), η_g < η_r:
#  energia płynie gorące promieniowanie → kubit → zimna grawitacja.
#  Stan stacjonarny jest NIE-równowagowy (NESS): σ_NESS = σ_r + σ_g > 0 stałe,
#  więc „zegar grawitacyjny" T_g(n) = Σ σ_k·τ rośnie liniowo w nieskończoność
#  (czas nigdy nie zamiera) — w przeciwieństwie do zegara S(n), który nasyca się.
# -----------------------------------------------------------------------------
def kapiel_gibbsa(gamma, eta):
    """Generator Lindblada kąpieli Gibbsa (a = 2γ/(1+η) emisja, b = 2γη/(1+η) absorpcja)."""
    sz, sp, sm = M.operatory()
    a = 2.0 * gamma / (1.0 + eta)
    b = 2.0 * gamma * eta / (1.0 + eta)
    return superoperator_z_jumpami(np.zeros((2, 2)), [sp, sm], [a, b])


def sigma_spohna(L, rho, rho_eq):
    """
    Produkcja entropii wg Spohna dla pojedynczej kąpieli:
    σ = −Tr[L(ρ)·(ln ρ − ln ρ_eq)] ≥ 0  (tw. Spohna, każda kąpiel osobno).
    Stany diagonalne ⇒ log z populacji.
    """
    Lrho = M._unvec(L @ M._vec(rho))
    X00 = np.log(max(rho[0, 0].real, 1e-300)) - np.log(max(rho_eq[0, 0].real, 1e-300))
    X11 = np.log(max(rho[1, 1].real, 1e-300)) - np.log(max(rho_eq[1, 1].real, 1e-300))
    return float(-(X00 * Lrho[0, 0].real + X11 * Lrho[1, 1].real))


def symuluj_dwie_kapiele(gamma_r, eta_r, gamma_g, eta_g, n=400,
                         delta_tau=M.DELTA_TAU):
    """Kubit w dwóch kąpielach → NESS. Zwraca S, σ_r, σ_g (per tyknięcie)."""
    Lr = kapiel_gibbsa(gamma_r, eta_r)
    Lg = kapiel_gibbsa(gamma_g, eta_g)
    L = Lr + Lg
    rho_eq_r = np.diag([1.0 / (1.0 + eta_r), eta_r / (1.0 + eta_r)])
    rho_eq_g = np.diag([1.0 / (1.0 + eta_g), eta_g / (1.0 + eta_g)])
    U = expm(L * delta_tau)
    rho = np.diag([1.0, 0.0]).astype(complex)
    S = np.zeros(n); sig_r = np.zeros(n); sig_g = np.zeros(n)
    for i in range(n):
        S[i] = M.entropia(rho)
        sig_r[i] = sigma_spohna(Lr, rho, rho_eq_r)
        sig_g[i] = sigma_spohna(Lg, rho, rho_eq_g)
        rho = M._unvec(U @ M._vec(rho))
    return S, sig_r, sig_g


def figura_R13():
    """Grawitacyjna produkcja entropii: NESS, stałe σ, niekończący się zegar."""
    G_R, ETA_R = 0.05, 0.9      # promieniowanie (gorące)
    G_G, ETA_G = 0.01, 0.1      # grawitacja (zimna kąpiel)
    n_l = 400
    S, sig_r, sig_g = symuluj_dwie_kapiele(G_R, ETA_R, G_G, ETA_G, n=n_l)
    sig = sig_r + sig_g
    n = np.arange(n_l)
    tau = M.DELTA_TAU

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axs[0, 0]
    S1, _, _ = M.symuluj(M.GAMMA_B, n=n_l)
    ax.plot(n, S1, color="#8b98a5", lw=1.6, ls="--", label="jedna kąpiel (∞): S → ln 2")
    ax.plot(n, S, color="#c0392b", lw=2.2, label="dwie kąpiele (grawitacja): NESS")
    ax.axhline(M.LN2, color=C_G, ls=":", lw=1)
    ax.axhline(S[-1], color="#c0392b", ls=":", lw=1)
    ax.text(5, M.LN2 + 0.012, "ln 2", color=C_G, fontsize=9)
    ax.text(5, S[-1] + 0.012, f"NESS S* = {S[-1]:.4f} < ln 2", color="#c0392b", fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("Dwie kąpiele: stan stacjonarny NIE-równowagowy (NESS)")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(n, sig_r, color="#2471a3", lw=2, label="σ_prom (gorąca kąpiel)")
    ax.plot(n, sig_g, color="#1a5276", lw=2, label="σ_graw (zimna kąpiel)")
    ax.plot(n, sig, color="#c0392b", lw=2.4, label="σ_tot")
    ax.axhline(sig[-1], color="#c0392b", ls=":", lw=1)
    ax.text(5, sig[-1] * 1.02, f"σ_NESS = {sig[-1]:.5f} (stałe)", color="#c0392b", fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("σ [nat/j. czasu]")
    ax.set_title("Produkcja entropii Spohna — w NESS stała i dodatnia (tw. Spohna)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    T_grav = np.cumsum(sig) * tau
    ax.plot(n, T_grav, color="#c0392b", lw=2.2, label="T_graw(n) = Σ σ_k·τ (zegar grawitacyjny)")
    ax.plot(n, S, color="#8b98a5", lw=1.8, ls="--", label="T = S(n) (zegar entropii — nasyca się)")
    ax.axhline(M.LN2, color=C_G, ls=":", lw=1)
    ax.text(5, M.LN2 + 0.1, "ln 2 — koniec zwykłego czasu", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("skumulowany czas [nat]")
    ax.set_title("Grawitacja daje czas bez końca: T_graw rośnie liniowo")
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1, 1]
    gammas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    sigma_ness = []
    for gg in gammas:
        _, sr, sg = symuluj_dwie_kapiele(G_R, ETA_R, gg, ETA_G, n=250)
        sigma_ness.append((sr + sg)[-1])
    ax.plot(gammas, sigma_ness, "o-", color="#c0392b", lw=2)
    ax.set_xlabel("γ_graw (siła kąpieli grawitacyjnej)")
    ax.set_ylabel("σ_NESS [nat/j. czasu]")
    ax.set_title("Silniejsza grawitacja ⇒ więcej entropii na tyknięcie (czas płynie szybciej)")
    ax.text(0.02, sigma_ness[-1] * 0.5, "σ_NESS = 0 przy γ_graw = 0\n(pojedyncza kąpiel: koniec czasu)",
            fontsize=9, color="#26384a")

    fig.suptitle("R13 — grawitacyjna produkcja entropii: NESS i zegar, który nigdy nie staje",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR13_grawitacja.png", bbox_inches="tight")
    plt.close(fig)

    d = dict(S_ness=float(S[-1]), sigma=float(sig[-1]), sigma_r=float(sig_r[-1]),
             sigma_g=float(sig_g[-1]), sum_sig_tau=float(T_grav[-1]),
             sigma_tick=float(sig[-1] * tau),
             ln2=M.LN2, ratio=float(T_grav[-1] / M.LN2),
             S=[round(float(x), 6) for x in S.tolist()],
             T_grav=[round(float(x), 6) for x in T_grav.tolist()],
             gamma_r=G_R, eta_r=ETA_R, gamma_g=G_G, eta_g=ETA_G)
    CACHE["R13"] = d
    return d


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#  R15 — DEKOHERENCJA ZEGARA JAKO STRAŻNIK HISTORII (κ w punkcie zwrotnym)
#  Teza (demonstrowana numerycznie w modelu):
#    Hamiltonian splątujący H_int (Jaynes–Cummings) próbuje rozmyć czas —
#    tworzy koherencje wskazań zegara (superpozycję „która godzina?”).
#    Człon κ·D[b†b] nieustannie rzutuje stan zegara na oś liczbową (baza
#    Focka — einselected pointer basis), zamieniając kwantowe „czkanie”
#    (dyskretne zdarzenia kranika σ₋⊗b†) w mierzalny, NIEODWRACALNY przyrost
#    drogi czasu τ = ⟨n⟩·δs. W punkcie zwrotnym (nasycenie, ΔS→0) to κ
#    decyduje, czy zapis historii pozostaje klasyczny i jednoznaczny.
# -----------------------------------------------------------------------------
def kwantowy_zegar_hint(gt, g, kappa, MLEV=24, TICKS=320, gamma=0.02,
                        gphi=0.04, omega=M.OMEGA, delta_tau=M.DELTA_TAU,
                        capture=None):
    """
    Kubit⊗zegar z jawnym H_int = g(σ₋⊗b† + σ₊⊗b) (splątanie) + κ·D[b†b]
    (dekoherencja zegara). Zwraca historie: Ss, nb (zapis), coh, offdiag
    (rozmycie), pur, Scl, I, pn; opcjonalnie macierz ρ_c w tyknięciu capture.
    """
    from scipy.linalg import lu_factor, lu_solve
    ML = MLEV; tau = delta_tau; n_ticks = TICKS
    I2 = np.eye(2, dtype=complex); sz, sp_, sm_ = M.operatory()
    a = np.diag(np.sqrt(np.arange(1, ML)), 1); ad = a.conj().T
    IM = np.eye(ML, dtype=complex); Nc = ad @ a
    H = 0.5 * omega * np.kron(sz, IM) + g * (np.kron(sp_, ad) + np.kron(sm_, a))
    jumps = [np.kron(sm_, ad), np.kron(sm_, IM), np.kron(sp_, IM), np.kron(sz, IM)]
    rates = [gt, gamma, gamma, gphi]
    if kappa > 0:
        jumps.append(np.kron(I2, Nc)); rates.append(kappa)

    def ptrace_sys(r):
        return np.array([[np.trace(r[0:ML, 0:ML]), np.trace(r[0:ML, ML:2 * ML])],
                         [np.trace(r[ML:2 * ML, 0:ML]), np.trace(r[ML:2 * ML, ML:2 * ML])]])

    def ptrace_clock(r):
        rc = np.zeros((ML, ML), complex)
        for c in range(ML):
            for cp in range(ML):
                rc[c, cp] = r[c, cp] + r[ML + c, ML + cp]
        return rc

    d = 2 * ML; Id = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for J, r in zip(jumps, rates):
        Jd = J.conj().T; JJ = Jd @ J
        L += r * (np.kron(J, J.conj()) - 0.5 * (np.kron(JJ, Id) + np.kron(Id, JJ.T)))
    IL = np.eye(d * d, dtype=complex)
    lu, piv = lu_factor(IL - 0.5 * tau * L)
    B = IL + 0.5 * tau * L

    psi = np.array([np.cos(M.THETA0 / 2), np.exp(1j * M.PHI0) * np.sin(M.THETA0 / 2)])
    rho_c0 = np.zeros((ML, ML), complex); rho_c0[0, 0] = 1
    rho = np.kron(np.outer(psi, psi.conj()), rho_c0)

    def S(r):
        ev = np.linalg.eigvalsh((r + r.conj().T) / 2); ev = ev[ev > 1e-15]
        return -np.sum(ev * np.log(ev))

    out = dict(Ss=np.zeros(n_ticks), nb=np.zeros(n_ticks), coh=np.zeros(n_ticks),
               offdiag=np.zeros(n_ticks), pur=np.zeros(n_ticks),
               Scl=np.zeros(n_ticks), I=np.zeros(n_ticks),
               pn=np.zeros((n_ticks, ML)), rc_capture=None)
    for k in range(n_ticks):
        rs = ptrace_sys(rho); rc = ptrace_clock(rho)
        pn = np.real(np.diag(rc)); pn = np.clip(pn, 0, None); pn = pn / pn.sum()
        out["Ss"][k] = S(rs)
        out["nb"][k] = np.sum(np.arange(ML) * pn)
        out["coh"][k] = np.sum(np.abs(np.diag(rc, 1))) / (ML - 1)
        out["offdiag"][k] = float(np.sum(np.abs(rc) ** 2) - np.sum(np.abs(np.diag(rc)) ** 2))
        out["pur"][k] = np.real(np.trace(rc @ rc))
        out["Scl"][k] = S(rc)
        out["I"][k] = S(rs) + S(rc) - S(rho)
        out["pn"][k] = pn
        if capture is not None and k == capture:
            out["rc_capture"] = rc.copy()
        rho = np.asarray(lu_solve((lu, piv), B @ rho.flatten())).reshape(d, d)
    return out


def punkt_zwrotny(Ss, prog=1e-3):
    """Pierwsze tyknięcie, po którym ΔS < prog (nasycenie — zwrot strzałki)."""
    dS = np.maximum(np.diff(np.asarray(Ss)), 0.0)
    ix = np.where(dS < prog)[0]
    return int(ix[0]) if len(ix) else len(Ss) - 1


def figura_R15():
    """κ jako strażnik historii: rzutowanie na oś liczbową w punkcie zwrotnym."""
    GT, G, TICKS = 0.02, 0.2, 320
    kappas = [0.0, 0.1, 0.5]
    wyn = {}
    for kap in kappas:
        z = kwantowy_zegar_hint(GT, G, kap, TICKS=TICKS,
                                capture=punkt_zwrotny(
                                    kwantowy_zegar_hint(GT, G, kap, TICKS=TICKS)["Ss"]))
        wyn[kap] = z
    n = np.arange(TICKS)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    # (a) koherencje zegara: κ rzutuje na oś liczbową
    ax = axs[0, 0]
    for kap, c in zip(kappas, ["#8b98a5", "#e67e22", "#c0392b"]):
        ax.semilogy(n, wyn[kap]["coh"] + 1e-6, color=c, lw=2,
                    label=f"κ = {kap}")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("koherencje ⟨|ρ_c[n,n+1]|⟩")
    ax.set_title("H_int splątuje i rozmywa czas; κ nieustannie rzutuje na oś liczbową")
    ax.legend(fontsize=9)

    # (b) rozmycie czasu w punkcie zwrotnym vs κ
    ax = axs[1, 0]
    off_at_turn = [wyn[k]["offdiag"][punkt_zwrotny(wyn[k]["Ss"])] for k in kappas]
    ax.loglog(kappas, off_at_turn, "o-", color="#c0392b", lw=2, ms=9)
    ax.set_xlabel("κ (dekoherencja zegara)")
    ax.set_ylabel("rozmycie czasu Σ|ρ_c[n≠m]|² (punkt zwrotny)")
    ax.set_title("Kwantowe rozmycie czasu ginie z κ — zapis staje się klasyczny")
    for kap, v in zip(kappas, off_at_turn):
        ax.annotate(f"{v:.2e}", (kap, v), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)

    # (c) macierz zegara w punkcie zwrotnym: kwantowa vs klasyczna
    ax = axs[0, 1]
    rho0 = np.abs(wyn[0.0]["rc_capture"])
    rho5 = np.abs(wyn[0.5]["rc_capture"])
    vmax = max(rho0.max(), rho5.max())
    ax.imshow(rho0, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(f"κ = 0 (punkt zwrotny): czas KWANTOWY — koherencje poza przekątną")
    ax.set_xlabel("n (wskazanie)"); ax.set_ylabel("n'")
    ax = axs[1, 1]
    ax.imshow(rho5, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(f"κ = 0.5 (punkt zwrotny): czas KLASYCZNY — oś liczbowa (przekątna)")
    ax.set_xlabel("n (wskazanie)"); ax.set_ylabel("n'")

    fig.suptitle("R15 — dekoherencja zegara jako strażnik historii Wszechświata",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR15_straznik.png", bbox_inches="tight")
    plt.close(fig)

    d = dict(gt=GT, g=G, kappas=kappas)
    for kap in kappas:
        z = wyn[kap]
        n_t = punkt_zwrotny(z["Ss"])
        d[f"k{kap}"] = dict(
            coh_turn=float(z["coh"][n_t]), offdiag_turn=float(z["offdiag"][n_t]),
            pur_turn=float(z["pur"][n_t]), Scl_turn=float(z["Scl"][n_t]),
            I_turn=float(z["I"][n_t]), n_turn=n_t,
            nb_end=float(z["nb"][-1]), Ss_end=float(z["Ss"][-1]),
            mono=bool(np.all(np.diff(z["nb"]) >= -1e-6)),
            offdiag_end=float(z["offdiag"][-1]),
            coh_series=[round(float(v), 6) for v in z["coh"].tolist()])
    CACHE["R15"] = d
    return d



# -----------------------------------------------------------------------------
#  R16 — FORMALIZM RELACYJNY (rewizja po recenzji)
#  Zamiast T≡S: trzypoziomowy schemat  λ → S → τ  oraz funkcjonał czasu
#  recenzenta:
#      dτ_A/dλ = α·[Ṡ_A^prod + η·I(A:E)]
#  • η = 0: zegar entropii (stary model jako przypadek szczególny);
#  • η > 0: korelacje (informacja wzajemna) też napędzają zegar;
#  • „27×” staje się PREDYKCJĄ WARUNKOWĄ gałęzi dτ∝s (γ∝T³), nie aksjomatem;
#    gałąź dτ∝Ṡ daje inne liczby — rozróżnienie falsyfikowalne (test γ∝T^p).
#  • Zatrzymanie zegara entropowego ≠ koniec czasu fizycznego (recenzja §2).
# -----------------------------------------------------------------------------
def zegar_relacyjny(dS, I=None, alpha=1.0, eta=0.0, ds=None, ref=0.01, seed=0):
    """
    Δτ_n = α·[τ₀·ΔS_n/ΔS_ref + η·I_n]   (formuła recenzji + człon korelacyjny).
    Część entropowa kwantowana (k_n ~ Poisson(ΔS/δs)); część korelacyjna ciągła.
    Zwraca (tau, dtau).
    """
    ds = M.DELTA_S_Q if ds is None else ds
    dS = np.asarray(dS, dtype=float)
    n = len(dS)
    I = np.zeros(n) if I is None else np.asarray(I, dtype=float)
    rng = np.random.default_rng(seed)
    k = rng.poisson(np.maximum(dS, 0.0) / ds)
    dtau_ent = (k * ds) / ref
    dtau = alpha * (dtau_ent + eta * I)
    return np.cumsum(dtau), dtau


def stosunek_27_jako_predykcja():
    """
    Gałąź s (dτ ∝ s ∝ T³, γ ∝ T³): τ_A/τ_B = 27 DOKŁADNIE (dopasowane S*).
    Gałąź Ṡ (dτ ∝ Ṡ): stosunek = rzeczywiste tempo — ≠ 27 (pierwsze tyknięcie,
    nasycenie). Zależność od wykładnika p w γ ∝ T^p: stosunek = 3^p.
    """
    d = {}
    t = 1.0
    d["s_branch"] = float(M.dSdt_analityczne(M.GAMMA_A, t) /
                          M.dSdt_analityczne(M.GAMMA_B, 27.0 * t))          # 27.0
    S_A, _, _ = M.symuluj(M.GAMMA_A, n=M.N_TICKS)
    S_B, _, _ = M.symuluj(M.GAMMA_B, n=M.N_TICKS)
    dS_A = M.delta_entropii(S_A); dS_B = M.delta_entropii(S_B)
    d["sdot_tick1"] = float(dS_A[1] / dS_B[1])                               # ≈ 8.8
    # nasycenie: ostatni niezerowy krok A vs B (oba → 0) — stosunek → 1
    d["sdot_sat"] = 1.0
    d["p_scan"] = {p: float(3.0 ** p) for p in [0, 1, 2, 3]}
    return d


def figura_R16():
    """Formalizm relacyjny: funkcjonał τ(Ṡ,I), 27 jako predykcja warunkowa."""
    # dane zegara kwantowego (R9): S_sys, I(system;clock)
    z = CACHE.get("R9", {}).get("z") or kwantowy_zegar(0.01, TICKS=300)
    n = np.arange(len(z["S_sys"]))
    dS = np.maximum(np.diff(z["S_sys"]), 0.0)
    dS = np.concatenate([[0.0], dS])
    tau_ent, _ = zegar_relacyjny(dS, eta=0.0, seed=1)
    tau_rel, _ = zegar_relacyjny(dS, I=z["I"], eta=0.5, seed=1)
    tau_rel2, _ = zegar_relacyjny(dS, I=z["I"], eta=2.0, seed=1)

    pr = stosunek_27_jako_predykcja()

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axs[0, 0]
    ax.plot(n, z["S_sys"], color="#8b98a5", lw=2, ls="--", label="S_sys (wszechświat)")
    ax.plot(n, tau_ent, color="#1a5276", lw=2, label="τ, η = 0 (zegar entropii)")
    ax.plot(n, tau_rel, color="#8e44ad", lw=2, label="τ, η = 0.5 (z korelacjami I)")
    ax.plot(n, tau_rel2, color="#c0392b", lw=1.6, label="τ, η = 2")
    ax.axhline(M.LN2, color=C_G, ls=":", lw=1)
    ax.text(5, M.LN2 + 0.02, "ln 2", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("τ [jedn. czasu]")
    ax.set_title("dτ/dλ = α[Ṡ + η·I(A:E)]: korelacje dodają czasu (recenzja §9)")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(n, np.diff(np.concatenate([[0], tau_ent])), color="#1a5276", lw=2,
            label="τ̇ (η = 0)")
    ax.plot(n, np.diff(np.concatenate([[0], tau_rel])), color="#8e44ad", lw=1.6,
            label="τ̇ (η = 0.5)")
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("τ̇ — tempo zegara")
    ax.set_title("Przy równowadze τ̇ → 0: zegar entropowy STOJE\n"
                 "(ale mikroskopowa ewolucja może trwać — recenzja §2)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    x = np.arange(3); w = 0.5
    vals = [pr["s_branch"], pr["sdot_tick1"], pr["sdot_sat"]]
    b = ax.bar(x, vals, w, color=["#27ae60", "#e67e22", "#c0392b"])
    ax.axhline(27, color=C_G, ls=":", lw=1)
    ax.text(2.3, 27.3, "27", color=C_G, fontsize=9)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.03, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["gałąź s\n(dτ ∝ s ∝ T³)", "gałąź Ṡ\n(1. tyknięcie)",
                        "gałąź Ṡ\n(nasycenie)"], fontsize=8)
    ax.set_ylabel("τ_A / τ_B (dopasowany poziom S*)")
    ax.set_title("„27” to predykcja WARUNKOWA gałęzi dτ∝s; gałąź dτ∝Ṡ daje inne "
                 "liczby (recenzja §3)")

    ax = axs[1, 1]
    ps = sorted(pr["p_scan"])
    ax.plot(ps, [pr["p_scan"][p] for p in ps], "o-", color="#c0392b", lw=2, ms=8)
    ax.set_xticks(ps)
    ax.set_xticklabels([f"T^{p}" for p in ps])
    ax.set_xlabel("skalowanie tempa γ ∝ T^p")
    ax.set_ylabel("τ_A/τ_B (dopasowany S*)")
    ax.set_title("Test falsyfikacyjny: stosunek = 3^p — tylko p = 3 daje 27")
    ax.text(0.05, 0.9, "zmierz stosunek dwóch zegarów\nw kąpielach A i B\n"
                       "⇒ wyznacz p z modelu", transform=ax.transAxes,
            fontsize=9, color="#26384a")

    fig.suptitle("R16 — formalizm relacyjny po recenzji: τ = F(S, Ṡ, I, Γ)",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR16_relacyjny.png", bbox_inches="tight")
    plt.close(fig)
    d = dict(pr=pr, tau_ent=tau_ent[-1], tau_rel=tau_rel[-1], tau_rel2=tau_rel2[-1],
             ln2=M.LN2)
    CACHE["R16"] = d
    return d


# -----------------------------------------------------------------------------
#  R17 — LABORATORYJNY TEST: JASNY ↔ CIEMNY ZEGAR ENTROPOWY
#  Formuła recenzji: Δτ_n = τ₀·ΔS_n/ΔS_ref. Dla N=2 kąpieli kolektywnej:
#    • jasny tryplet  ⇒ Ṡ > 0, zegar tyka;
#    • ciemny singlet ⇒ Ṡ = 0, zegar MILCZY (Γ_dark ≪ Γ_bright);
#    • |10⟩ = jasny+ciemny: zegar zwalnia z zanikiem jasnej części, a entropia
#      zostaje zamrożona w ciemnym sektorze (pamięć).
#  Falsyfikowalne przewidywanie: przejście do sektora subradiacyjnego spowalnia
#  entropiczny zegar — testowalne w zimnych atomach (PRL 116, 083601).
# -----------------------------------------------------------------------------
def zegar_entropowy(dS, ref=0.01, seed=0):
    """Δτ_n = τ₀·ΔS_n/ΔS_ref (formuła recenzji); τ = ΣΔτ."""
    dS = np.asarray(dS, dtype=float)
    rng = np.random.default_rng(seed)
    k = rng.poisson(np.maximum(dS, 0.0) / M.DELTA_S_Q)
    return np.cumsum((k * M.DELTA_S_Q) / ref)


def figura_R17():
    """Jasny↔ciemny zegar entropowy: test laboratoryjny kluczowej hipotezy."""
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    SING = (np.kron(ket1, ket0) - np.kron(ket0, ket1)) / np.sqrt(2)
    T0 = (np.kron(ket1, ket0) + np.kron(ket0, ket1)) / np.sqrt(2)
    n_l = 800
    S_s, *_ = symuluj_wspolne(M.GAMMA_B, np.outer(SING, SING.conj()), N=2,
                              gamma_phi=0.0, n=n_l)
    S_t, *_ = symuluj_wspolne(M.GAMMA_B, np.outer(T0, T0.conj()), N=2,
                              gamma_phi=0.0, n=n_l)
    S_10, *_ = symuluj_wspolne(M.GAMMA_B, stan_poczatkowy_N([ket1, ket0]), N=2,
                               gamma_phi=0.0, n=n_l)
    S_11, *_ = symuluj_wspolne(M.GAMMA_B, stan_poczatkowy_N([ket1, ket1]), N=2,
                               gamma_phi=0.0, n=n_l)
    dS = lambda S: np.maximum(np.diff(S), 0.0)
    dS_s, dS_t, dS_10, dS_11 = dS(S_s), dS(S_t), dS(S_10), dS(S_11)
    tau_s = zegar_entropowy(dS_s); tau_t = zegar_entropowy(dS_t)
    tau_10 = zegar_entropowy(dS_10); tau_11 = zegar_entropowy(dS_11)

    # (c) tempo vs frakcja ciemna: ρ(p) = (1-p)|T0⟩⟨T0| + p|S⟩⟨S|
    ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rate0 = []
    for p in ps:
        rho = (1 - p) * np.outer(T0, T0.conj()) + p * np.outer(SING, SING.conj())
        S_p, *_ = symuluj_wspolne(M.GAMMA_B, rho, N=2, gamma_phi=0.0, n=300)
        rate0.append(float(np.maximum(np.diff(S_p), 0)[5:40].mean()))

    # (d) singlet z precesją unitarną: S = 0 (zegar milczy), stan ewoluuje
    OM = 0.5
    sz, sp, sm = M.operatory()
    I2 = np.eye(2, dtype=complex)
    H2 = OM / 2 * (np.kron(sz, I2) + np.kron(I2, sz))
    # ewolucja unitarna singleta (bez kąpieli)
    U2 = expm(-1j * H2 * M.DELTA_TAU)
    psi = SING
    fid = np.zeros(200); Sent = np.zeros(200)
    for i in range(200):
        rho = np.outer(psi, psi.conj())
        Sent[i] = M.entropia(rho)
        fid[i] = np.abs(psi.conj() @ SING) ** 2
        psi = U2 @ psi

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axs[0, 0]
    nn = np.arange(n_l - 1)
    ax.semilogy(nn, dS_t + 1e-8, color="#c0392b", lw=2, label="|11⟩ / tryplet (jasny)")
    ax.semilogy(nn, dS_10 + 1e-8, color="#e67e22", lw=2, label="|10⟩ (jasny + ciemny)")
    ax.semilogy(nn, dS_s + 1e-8, color="#2471a3", lw=2, label="singlet (ciemny)")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("Ṡ [nat/tyknięcie]")
    ax.set_title("Tempo produkcji entropii wg sektora: Γ_dark ≪ Γ_bright")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    # informacja wzajemna (pamięć) dla |10⟩ — z R2
    _, _, _, MI10 = symuluj_wspolne(M.GAMMA_B, stan_poczatkowy_N([ket1, ket0]),
                                    N=2, gamma_phi=0.0, n=n_l)
    ax.plot(np.arange(n_l), S_t, color="#c0392b", lw=2, label="|11⟩: S → ln 3 (termalizacja)")
    ax.plot(np.arange(n_l), S_10, color="#e67e22", lw=2, label="|10⟩: S → ½·ln 12")
    ax.plot(np.arange(n_l), S_s, color="#2471a3", lw=2, label="singlet: S = 0 (ciemny)")
    ax2 = ax.twinx()
    ax2.plot(np.arange(n_l), MI10, color="#8e44ad", lw=1.6, ls="--",
             label="I(A;B) — pamięć w ciemnym sektorze")
    ax2.set_ylabel("I(A;B) [nat]", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S [nat]")
    ax.set_title("Zegar staje (τ̇ → 0), a pamięć trwa: I(A;B) = ln(2/√3) w |10⟩")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")

    ax = axs[1, 0]
    ax.plot(ps, rate0, "o-", color="#8e44ad", lw=2, ms=8)
    ax.set_xlabel("p — frakcja ciemna (singlet) w stanie początkowym")
    ax.set_ylabel("⟨Ṡ⟩ — tempo zegara (wczesne)")
    ax.set_title("Test: frakcja ciemna ↑ ⇒ tempo zegara ↓ (liniowo)")
    ax.text(0.05, 0.9, "Γ_dark ≈ 0 ⇒ τ̇ ∝ (1−p)·Γ_bright",
            transform=ax.transAxes, fontsize=9, color="#26384a")

    ax = axs[1, 1]
    ax.plot(np.arange(200), fid, color="#1a5276", lw=2, label="|⟨ψ(t)|S⟩|² (ewolucja)")
    ax.plot(np.arange(200), Sent, color="#c0392b", lw=2, label="S(t) — zegar milczy")
    ax.set_xlabel("tyknięcie n (ewolucja unitarna)"); ax.set_ylabel("wartość")
    ax.set_title("Zegar stoi, fizyka trwa: singlet precesuje (fidel. spada),\n"
                 "S = 0 — zatrzymanie zegara ≠ zatrzymanie czasu (recenzja §2)")
    ax.legend(fontsize=8)

    fig.suptitle("R17 — laboratoryjny test ENTROPII: jasny ↔ ciemny zegar entropowy",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR17_test_lab.png", bbox_inches="tight")
    plt.close(fig)
    d = dict(tau_11=float(tau_11[-1]), tau_10=float(tau_10[-1]), tau_s=float(tau_s[-1]),
             rate11=float(dS_t[5:40].mean()), rate10=float(dS_10[5:40].mean()),
             rate_s=float(dS_s[5:40].mean()), ps=ps, rates=rate0,
             s_inf_singlet=float(S_s[-1]), fid_end=float(fid[-1]))
    CACHE["R17"] = d
    return d


#  FIGURY
# -----------------------------------------------------------------------------
def figura_R1():
    """Skończona T: plateau < ln 2, zależność S_eq(η), overshoot, czystość."""
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    etas = [1.0, 0.5, 0.25, 0.1, 0.01]
    cols = [C_G, "#2471a3", "#1a5276", C_V, "#6c3483"]
    t = np.linspace(0, 100, 1200)

    ax = axs[0, 0]
    for eta, c in zip(etas, cols):
        Seq = S_eq_termiczna(eta)
        ax.plot(t, S_termiczna_analitycznie(M.GAMMA_B, eta, t), color=c, lw=2,
                label=f"η={eta:.2g}  (βΩ={-np.log(eta):.2f})")
        ax.axhline(Seq, color=c, ls=":", lw=1)
    ax.axhline(M.LN2, color=C_G, ls="--", lw=1.3)
    ax.text(2, M.LN2 + 0.015, "ln 2 (kąpiel ∞)", color=C_G, fontsize=9)
    ax.set_xlabel("t (j. czasu, otoczenie B)")
    ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("Skończona T: monotoniczny wzrost do plateau < ln 2")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    eta_grid = np.logspace(-3, 0, 400)
    ax.plot(eta_grid, [S_eq_termiczna(e) for e in eta_grid], color=C_V, lw=2)
    ax.axhline(M.LN2, color=C_G, ls="--", lw=1)
    ax.text(2e-3, M.LN2 + 0.015, "ln 2", color=C_G)
    for eta, c in zip(etas, cols):
        ax.plot(eta, S_eq_termiczna(eta), "o", color=c, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("η = e^{−βΩ}  (1 = kąpiel nieskończenie gorąca)")
    ax.set_ylabel("S(∞) [nat]")
    ax.set_title("Nasycenie entropii vs temperatura kąpieli")

    ax = axs[1, 0]
    for eta, c in [(0.5, "#2471a3"), (0.2, "#c0392b")]:
        St = S_termiczna_analitycznie(M.GAMMA_B, eta, t)
        ax.plot(t, St, color=c, lw=2, label=f"η={eta:.1f}, S(∞)={S_eq_termiczna(eta):.4f}")
        ax.axhline(S_eq_termiczna(eta), color=c, ls=":", lw=1)
    # overshoot dla η=0.2
    eta = 0.2
    St = S_termiczna_analitycznie(M.GAMMA_B, eta, t)
    imax = int(np.argmax(St))
    ax.plot(t[imax], St[imax], "o", color="#c0392b", ms=7)
    ax.annotate(f"przewyższenie: S_max = {St[imax]:.4f} > S(∞) = {S_eq_termiczna(eta):.4f}",
                xy=(t[imax], St[imax]), xytext=(18, 0.58), fontsize=10,
                color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.set_xlim(0, 60)
    ax.set_xlabel("t")
    ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("Zimna kąpiel (η < 1/3): entropia przewyższa plateau i opada")
    ax.legend(fontsize=9)

    ax = axs[1, 1]
    for eta, c in zip(etas, cols):
        ax.plot(t, 1 - (1 - czystosc_równowagowa(eta)) * (1 - np.exp(-0.0 * t)),
                color=c, lw=0, label="_nolegend_")
    for eta, c in zip(etas, cols):
        P0 = 1.0
        Pe = czystosc_równowagowa(eta)
        # czystość z postaci zamkniętej |r|(t)
        r = r_norm_termiczny(M.GAMMA_B, eta, t)
        ax.plot(t, (1 + r * r) / 2, color=c, lw=2,
                label=f"η={eta:.2g}: Tr(ρ²)→{Pe:.3f}")
    ax.axhline(0.5, color=C_G, ls="--", lw=1)
    ax.text(2, 0.505, "0.5 (tylko η = 1)", color=C_G, fontsize=8)
    ax.set_xlabel("t")
    ax.set_ylabel("Tr(ρ²)")
    ax.set_title("Czystość: 1 → (1+r_eq²)/2  (dla η < 1: > 0.5)")
    ax.legend(fontsize=8)

    fig.suptitle("R1 — kąpiel o skończonej temperaturze: nasycenie poniżej ln 2",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR1_temperatura.png", bbox_inches="tight")
    plt.close(fig)


def figura_R2():
    """Wiele kubitów: ekstensywność (niezależne) + kolektywność (wspólne)."""
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    n = np.arange(M.N_TICKS)
    t = n * M.DELTA_TAU

    # (a) N niezależnych — ekstensywność
    ax = axs[0, 0]
    for N in [1, 2, 4, 8]:
        S_N, _ = symuluj_niezalezne(M.GAMMA_B, N, gamma_phi=0.0)
        ax.plot(n, S_N / N, lw=2, label=f"N = {N}")
    ax.axhline(M.LN2, color=C_G, ls="--", lw=1)
    ax.text(5, M.LN2 + 0.015, "ln 2 (na kubit)", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(n)/N [nat/kubit]")
    ax.set_title("Niezależne kąpiele: S = N·S₁ — ekstensywność (krzywe = 1)")
    ax.legend(ncol=2, fontsize=9)

    # (b) N=2 z |11⟩: wspólna vs niezależne
    ax = axs[0, 1]
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    stan11 = stan_poczatkowy_N([ket1, ket1])
    S_ws, P_ws, _, _ = symuluj_wspolne(M.GAMMA_B, stan11, N=2, gamma_phi=0.0)
    S_nz, _ = symuluj_niezalezne(M.GAMMA_B, 2, gamma_phi=0.0)
    ax.plot(n, S_nz, color=C_B, lw=2, label="2 niezależne: S(∞)=2·ln 2")
    ax.plot(n, S_ws, color=C_A, lw=2, label="wspólna kąpiel: S(∞)=ln 3")
    ax.axhline(2 * M.LN2, color=C_B, ls=":", lw=1)
    ax.axhline(np.log(3), color=C_A, ls=":", lw=1)
    ax.annotate(f"deficyt ln(4/3) ≈ {np.log(4/3):.3f}\n= entropia w korelacjach\n(I = {2*M.LN2-np.log(3):.3f} nat)",
                xy=(150, 1.21), xytext=(250, 1.28), fontsize=10, color=C_V,
                arrowprops=dict(arrowstyle="->", color=C_V))
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("N=2 start |11⟩: kąpiel kolektywna termalizuje tylko tryplet")
    ax.legend(fontsize=9)

    # (c) N=2 z |10⟩: ciemny singlet
    ax = axs[1, 0]
    stan10 = stan_poczatkowy_N([ket1, ket0])
    S_ws2, P_ws2, C_ws2, MI_ws2 = symuluj_wspolne(M.GAMMA_B, stan10, N=2, gamma_phi=0.0)
    ax.plot(n, S_nz, color=C_B, lw=1.4, ls="--", label="2 niezależne (2·ln 2)")
    ax.plot(n, S_ws2, color=C_V, lw=2, label="wspólna: S(∞)=½·ln 12")
    ax.axhline(np.log(12) / 2, color=C_V, ls=":", lw=1)
    ax.axhline(2 * M.LN2, color=C_B, ls=":", lw=1)
    ax.annotate("singlet (Bell) ciemny:\nprzeżywa w nieskończoność",
                xy=(200, 0.9), xytext=(260, 0.75), fontsize=10, color=C_V,
                arrowprops=dict(arrowstyle="->", color=C_V))
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("N=2 start |10⟩: połowa stanu (singlet) nie termalizuje")
    ax.legend(fontsize=9)

    # (d) czystość i informacja wzajemna dla |10⟩ wspólna (γ_φ = 0)
    ax = axs[1, 1]
    ax.plot(n, P_ws2, color="#1a5276", lw=2, label="Tr(ρ²) → 1/3")
    ax.axhline(1 / 3, color="#1a5276", ls=":", lw=1)
    ax.set_ylabel("Tr(ρ²)")
    ax2 = ax.twinx()
    ax2.plot(n, MI_ws2, color="#c0392b", lw=2, label="I = S(A)+S(B)−S(AB)")
    ax2.axhline(np.log(2 / np.sqrt(3)), color="#c0392b", ls=":", lw=1)
    ax2.set_ylabel("informacja wzajemna I [nat]", color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")
    ax.set_xlabel("tyknięcie n")
    ax.set_title(f"|10⟩: czystość → 1/3; korelacja I → {np.log(2/np.sqrt(3)):.3f} nat (splątanie 0)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")

    fig.suptitle("R2 — wiele kubitów: ekstensywność (niezależne) vs korelacje (wspólne)",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR2_kubity.png", bbox_inches="tight")
    plt.close(fig)
    return S_ws, S_ws2, P_ws2, C_ws2, MI_ws2


def figura_R3():
    """Feedback zegar→tempo: chłodzenie/przyspieszanie vs stały."""
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    n_fb = 600
    nn = np.arange(n_fb)

    scen = {
        "stały": (FB_STALY, C_G, 1.0),
        "chłodzenie (α=2)": (lambda u: FB_CHLODZENIE(u, 2.0), "#1a5276", 2.0),
        "przyspieszanie (α=1)": (lambda u: FB_PRZYSPIESZANIE(u, 1.0), "#c0392b", 1.0),
    }
    wyniki = {}
    for naz, (fb, c, a) in scen.items():
        S, G, T = symuluj_feedback(M.GAMMA_B, fb, n=n_fb)
        wyniki[naz] = (S, G, T)

    ax = axs[0, 0]
    for naz, (fb, c, a) in scen.items():
        S, G, T = wyniki[naz]
        ax.plot(nn, S, color=c, lw=2, label=naz)
    ax.axhline(M.LN2, color=C_G, ls="--", lw=1)
    ax.text(5, M.LN2 + 0.012, "ln 2", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("Ewolucja z samouzależnionym tempem γ_eff(T)")
    ax.legend(fontsize=9)

    ax = axs[0, 1]
    for naz, (fb, c, a) in scen.items():
        S, G, T = wyniki[naz]
        ax.semilogy(nn, G / M.GAMMA_B, color=c, lw=2, label=naz)
    ax.axhline(1, color=C_G, ls=":", lw=1)
    ax.text(5, 1.02, "γ₀", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("γ_eff / γ₀")
    ax.set_title("Historia tempa: chłodzenie wyhamowuje, przyspieszanie rozpędza")
    ax.legend(fontsize=9)

    ax = axs[1, 0]
    for naz, (fb, c, a) in scen.items():
        S, G, T = wyniki[naz]
        dS = np.zeros_like(S); dS[1:] = np.maximum(S[1:] - S[:-1], 0)
        T_cl, dt, k = M.zegar_stochastyczny(dS, seed=11)
        ax.semilogy(nn, dt + 1e-4, color=c, lw=1.2, alpha=0.85, label=naz)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("Δt_n (+1e-4)")
    ax.set_title("Czkanie: chłodzenie wzmacnia zamrożenia czasu")
    ax.legend(fontsize=9)

    ax = axs[1, 1]
    alphas = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    th_c = [t_do_polowy_ciggly(M.GAMMA_B, lambda u, a=a: FB_CHLODZENIE(u, a)) for a in alphas]
    th_a = [t_do_polowy_ciggly(M.GAMMA_B, lambda u, a=a: FB_PRZYSPIESZANIE(u, a)) for a in alphas]
    ax.plot(alphas, th_c, "o-", color="#1a5276", lw=2, label="chłodzenie")
    ax.plot(alphas, th_a, "o-", color="#c0392b", lw=2, label="przyspieszanie")
    ax.axhline(th_c[0], color=C_G, ls="--", lw=1)
    ax.text(2.1, th_c[0] + 0.25, "bez sprzężenia (α=0)", color=C_G, fontsize=9)
    ax.set_xlabel("siła sprzężenia α")
    ax.set_ylabel("czas do ½·ln 2 [j. czasu]")
    ax.set_title("Siła sprzężenia: chłodzenie wydłuża, przyspieszanie skraca")
    ax.legend(fontsize=9)

    fig.suptitle("R3 — sprzężenie „zegar → tempo”: tempo samo-zależy od odczytu zegara",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR3_feedback.png", bbox_inches="tight")
    plt.close(fig)
    return wyniki


def figura_R4():
    """N=3: sektory j=3/2 ⊕ 2×j=1/2 — jawne stany (sub)radiantne."""
    b, sector = baza_N3()
    stan_111 = b["111"]
    stan_100 = b["100"]
    stan_1S = sector[6][2]                    # |1⟩⊗|S⟩₂₃ — czysty j=1/2
    n_l = 6000
    S111, P111, *_ = symuluj_N3(M.GAMMA_B, stan_111, n=n_l)
    S100, P100, *_ = symuluj_N3(M.GAMMA_B, stan_100, n=n_l)
    # krótki przebieg z populacją na kopii B (subradiancja) w locie
    from scipy.linalg import expm as _expm
    S_z, S_p, S_m, _ = macierze_kolektywne(3)
    H = (M.OMEGA / 2.0) * S_z
    Lj = superoperator_z_jumpami(H, [S_p, S_m], [M.GAMMA_B, M.GAMMA_B])
    Uj = _expm(Lj * M.DELTA_TAU)
    PB = np.outer(sector[6][2], sector[6][2].conj()) + np.outer(sector[7][2], sector[7][2].conj())
    rho = np.outer(stan_1S, stan_1S.conj())
    popB = np.zeros(M.N_TICKS)
    S1S_s = np.zeros(M.N_TICKS); P1S_s = np.zeros(M.N_TICKS)
    for i in range(M.N_TICKS):
        S1S_s[i] = M.entropia(rho); P1S_s[i] = M.czystosc(rho)
        popB[i] = np.real(np.trace(PB @ rho))
        rho = unvecR(Uj @ vecR(rho), 8)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    n = np.arange(M.N_TICKS)
    sl = slice(0, M.N_TICKS)

    ax = axs[0, 0]
    ax.plot(n, S111[sl], color=C_A, lw=2, label="|111⟩ (sektor j=3/2)")
    ax.plot(n, S100[sl], color="#16a085", lw=2, label="|100⟩ (mieszany sektory)")
    ax.plot(n, S1S_s[sl], color=C_V, lw=2, label="|1⟩⊗|S⟩₂₃ (sektor j=1/2)")
    ax.axhline(np.log(4), color=C_A, ls=":", lw=1)
    ax.axhline(np.log(12) / 2 * 0 + np.log(2), color=C_V, ls=":", lw=1)
    ax.axhline(3 * M.LN2, color=C_B, ls="--", lw=1)
    ax.text(5, 3 * M.LN2 + 0.03, "3·ln 2 (niezależne)", color=C_B, fontsize=8)
    ax.text(5, np.log(4) + 0.02, "ln 4", color=C_A, fontsize=9)
    ax.text(5, np.log(2) + 0.02, "ln 2 — czapka subradiantna", color=C_V, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("N=3, kąpiel kolektywna: sektor j=3/2 → ln 4, j=1/2 → ln 2")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(n, P111[sl], color=C_A, lw=2, label="|111⟩")
    ax.plot(n, P100[sl], color="#16a085", lw=2, label="|100⟩")
    ax.plot(n, P1S_s[sl], color=C_V, lw=2, label="|1⟩⊗|S⟩₂₃")
    ax.axhline(1 / 4, color=C_G, ls=":", lw=1)
    ax.axhline(1 / 2, color=C_G, ls=":", lw=1)
    ax.text(5, 0.26, "1/4 (sektor 4-wym.)", color=C_G, fontsize=9)
    ax.text(5, 0.52, "1/2 (sektor 2-wym.)", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("Tr(ρ²)")
    ax.set_title("Czystość: |111⟩,|100⟩ → 1/4; stan j=1/2 → 1/2")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.axis("off")
    ax.set_title("Jawny rozkład na sektory (baza 3 kubitów)", fontsize=12)
    ax.text(0.02, 0.98, "j = 3/2  (sektor symetryczny, 4 stany):", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="top", color=C_A)
    for k in range(4):
        ax.text(0.04, 0.92 - 0.075 * k, sector[k][1], transform=ax.transAxes,
                fontsize=10, va="top", family="monospace", color="#26384a")
    ax.text(0.02, 0.60, "j = 1/2  (kopia A):", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="top", color="#16a085")
    for k in [4, 5]:
        ax.text(0.04, 0.54 - 0.075 * (k - 4), sector[k][1], transform=ax.transAxes,
                fontsize=10, va="top", family="monospace", color="#26384a")
    ax.text(0.02, 0.40, "j = 1/2  (kopia B — „ciemna”, singlet⊗kubit):",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top", color=C_V)
    for k in [6, 7]:
        ax.text(0.04, 0.34 - 0.075 * (k - 6), sector[k][1], transform=ax.transAxes,
                fontsize=10, va="top", family="monospace", color="#26384a")
    ax.text(0.02, 0.18, "Kąpiel kolektywna termalizuje każdy sektor do 𝟙/ dim wewnątrz\n"
                        "sektora; stany j=1/2 nie uciekają poza 1 bit (subradiancja).",
            transform=ax.transAxes, fontsize=9.5, va="top", color="#5b6b7b", style="italic")

    ax = axs[1, 1]
    ax.plot(n, popB, color=C_V, lw=2)
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.set_ylim(0.9, 1.01)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("Tr(P_B·ρ)")
    ax.set_title("Subradiancja: stan |1⟩⊗|S⟩₂₃ nigdy nie opuszcza kopii B")
    ax.text(5, 0.995, "populacja na kopii B ≡ 1", color=C_G, fontsize=9)

    fig.suptitle("R4 — N=3: jawne ciemne sektory j=1/2 (subradiantne) obok j=3/2",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR4_sektory.png", bbox_inches="tight")
    plt.close(fig)
    return dict(S111=S111, S100=S100, S1S=S1S_s, P111=P111, P100=P100, P1S=P1S_s,
                popB=popB, t90_111=czasy_90(S111, np.log(4)),
                t90_100=czasy_90(S100, np.log(108) / 3),
                t90_1S=czasy_90(S1S_s, np.log(2)))


def figura_R5():
    """Pełna entropia makro: N niezależnych (N·ln 2) vs kolektywna (ln(N+1))."""
    Ns = [1, 2, 3, 4]
    wyn = {}
    for N in Ns:
        wyn[N] = entropia_makro(N)
    # krzywe dla N=1,2,4,8 (niezależne)
    n_l = 6000
    t = np.arange(M.N_TICKS) * M.DELTA_TAU
    S1, _ = symuluj_niezalezne(M.GAMMA_B, 1, gamma_phi=0.0, n=M.N_TICKS)
    S8, _ = symuluj_niezalezne(M.GAMMA_B, 8, gamma_phi=0.0, n=M.N_TICKS)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    n = np.arange(M.N_TICKS)

    ax = axs[0, 0]
    for N in [1, 2, 4, 8]:
        S_N, _ = symuluj_niezalezne(M.GAMMA_B, N, gamma_phi=0.0, n=M.N_TICKS)
        ax.plot(n, S_N, lw=2, label=f"N = {N} → {N * M.LN2:.2f} nat")
    ax.axhline(M.LN2, color=C_G, ls="--", lw=1)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S_total(n) [nat]")
    ax.set_title("Niezależne kąpiele: pełna entropia makro → N·ln 2 = ln(2^N)")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    for N in [1, 2, 4, 8]:
        S_N, _ = symuluj_niezalezne(M.GAMMA_B, N, gamma_phi=0.0, n=M.N_TICKS)
        ax.plot(n, S_N / N, lw=2, label=f"N = {N}")
    ax.axhline(M.LN2, color=C_G, ls="--", lw=1)
    ax.text(5, M.LN2 + 0.012, "ln 2 / komórkę", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S/N [nat/kubit]")
    ax.set_title("Entropia na komórkę — krzywe się składają (ekstensywność)")

    ax = axs[1, 0]
    ind_vals = [N * M.LN2 for N in Ns]
    kol_vals = [np.log(N + 1) for N in Ns]
    sim_ind = [wyn[N][0] for N in Ns]
    sim_kol = [wyn[N][1] for N in Ns]
    ax.plot(Ns, ind_vals, "o--", color=C_B, lw=1.8, label="N·ln 2 (niezależne)")
    ax.plot(Ns, kol_vals, "o-", color=C_A, lw=2, label="ln(N+1) (kolektywna)")
    ax.plot(Ns, sim_ind, "s", color=C_B, ms=5, mfc="none", label="symulacja (niez.)")
    ax.plot(Ns, sim_kol, "s", color=C_A, ms=5, mfc="none", label="symulacja (kolekt.)")
    ax.set_xticks(Ns); ax.set_xticklabels([f"N={N}" for N in Ns])
    ax.set_xlabel("liczba komórek N"); ax.set_ylabel("S(∞) [nat]")
    ax.set_title("Nasycenie entropii: ekstensywne N·ln 2 vs kolektywne ln(N+1)")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    deficit = [N * M.LN2 - np.log(N + 1) for N in Ns]
    t90_ind = [wyn[N][3] for N in Ns]
    t90_kol = [wyn[N][4] for N in Ns]
    x = np.arange(len(Ns)); w = 0.34
    ax.bar(x - w / 2, t90_ind, w, color=C_B, label="t90% niezależne")
    ax.bar(x + w / 2, t90_kol, w, color=C_A, label="t90% kolektywne")
    ax.set_xticks(x); ax.set_xticklabels([f"N={N}" for N in Ns])
    ax.set_ylabel("t90% [tyknięcia]")
    ax.set_title("Czas nasycenia: kąpiel kolektywna działa szybciej (superradiancja)")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, deficit, "o-", color="#8e44ad", lw=2, label="deficyt N·ln2 − ln(N+1)")
    ax2.set_ylabel("deficyt [nat]", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")

    fig.suptitle("R5 — pełna entropia makro: ekstensywność (niezależne) vs korelacje (kolektywne)",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR5_makro.png", bbox_inches="tight")
    plt.close(fig)
    return wyn


def figura_R6():
    """Gorący Wielki Wybuch: entropia maleje ⇒ zegar biegnie wstecz."""
    ETA0, ETAB = 0.95, 0.15
    n_l = 600
    S_c, G_c, T_c, S0, Seq = symuluj_wielki_wybuch(
        M.GAMMA_B, ETA0, ETAB, FB_STALY, n=n_l)
    S_f, G_f, T_f, _, _ = symuluj_wielki_wybuch(
        M.GAMMA_B, ETA0, ETAB, lambda u: FB_CHLODZENIE(u, 2.0), n=n_l)

    # kontrola krzyżowa: zamknięcie vs pełna mapa Lindblada (η0 start, ηB kąpiel)
    from scipy.linalg import expm as _expm
    rho0 = np.diag([1.0 / (1.0 + ETA0), ETA0 / (1.0 + ETA0)]).astype(complex)
    L = superoperator_termiczny(M.GAMMA_B, ETAB)
    U = _expm(L * M.DELTA_TAU)
    rho = rho0.copy()
    S_l = np.zeros(M.N_TICKS)
    for i in range(M.N_TICKS):
        S_l[i] = M.entropia(rho)
        rho = M._unvec(U @ M._vec(rho))
    t = np.arange(M.N_TICKS) * M.DELTA_TAU
    blad = np.max(np.abs(S_l - S_c[:M.N_TICKS]))

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    n = np.arange(n_l)

    ax = axs[0, 0]
    ax.plot(n, S_c, color="#c0392b", lw=2, label="stałe γ (gorący start)")
    ax.plot(n, S_f, color="#1a5276", lw=2, label="chłodzenie (R3, α=2)")
    ax.axhline(S0, color=C_G, ls="--", lw=1)
    ax.axhline(Seq, color=C_G, ls="--", lw=1)
    ax.text(5, S0 + 0.006, f"S(0) = {S0:.4f} ≈ ln 2 („gorący Wielki Wybuch”)", color=C_G, fontsize=9)
    ax.text(300, Seq + 0.006, f"S(∞) = {Seq:.4f} (zimna kąpiel)", color=C_G, fontsize=9)
    ax.annotate("entropia MALEJE", xy=(150, 0.55), xytext=(250, 0.62), fontsize=11,
                color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("S(ρ) [nat]")
    ax.set_title("Gorący start (S ≈ ln 2) + zimna kąpiel: ochładzanie")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(n, T_c, drawstyle="steps-post", color="#c0392b", lw=2, label="T(n) — realizacja")
    ax.plot(n, S_c - S0, color="#c0392b", lw=1, ls="--", alpha=0.6, label="S(n) − S(0) (oczek.)")
    ax.axhline(0, color=C_G, ls=":", lw=1)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("T(n) [nat]")
    ax.set_title("Zegar kosmiczny biegnie WSTECZ (ujemne Δt)")
    ax.legend(fontsize=8)
    ax.text(0.97, 0.06, "T(n) = S(n) − S(0) < 0 — czas wstecz", transform=ax.transAxes,
            ha="right", fontsize=9, color="#c0392b")

    ax = axs[1, 0]
    ax.semilogy(n, G_c / M.GAMMA_B, color="#c0392b", lw=2, label="stałe γ")
    ax.semilogy(n, G_f / M.GAMMA_B, color="#1a5276", lw=2, label="chłodzenie (R3)")
    ax.axhline(1, color=C_G, ls=":", lw=1)
    ax.text(5, 1.02, "γ₀", color=C_G, fontsize=9)
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("γ_eff / γ₀")
    ax.set_title("Tempo: chłodzenie wyhamowuje ochładzanie (asymptota)")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(n, np.abs(np.diff(np.concatenate([[S_c[0]], S_c]))), color="#c0392b",
            lw=1.4, label="|ΔS_n| (stałe γ)")
    ax.axhline(M.DELTA_S_Q, color=C_G, ls="--", lw=1)
    ax.text(5, M.DELTA_S_Q * 1.4, "kwant δs — poniżej tego zegar stoi (czkanie)",
            color=C_G, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("tyknięcie n"); ax.set_ylabel("|ΔS_n| [nat/tyknięcie]")
    ax.set_title("Produkcja ujemnej entropii — podstawa „czkania wstecz”")
    ax.legend(fontsize=8)

    fig.suptitle("R6 — gorący Wielki Wybuch w R3: ochładzanie i czas wstecz (Wielki Zaciąg?)",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figR6_wielkiwybuch.png", bbox_inches="tight")
    plt.close(fig)
    return dict(S0=S0, Seq=Seq, S_c=S_c, S_f=S_f, G_c=G_c, G_f=G_f,
                T_c=T_c, blad=blad,
                t_half_c=t_do_polowy_wstecz(S_c, M.DELTA_TAU, S0, Seq),
                t_half_f=t_do_polowy_wstecz(S_f, M.DELTA_TAU, S0, Seq))


# -----------------------------------------------------------------------------
#  MAIN — liczby kluczowe rozszerzeń
# -----------------------------------------------------------------------------
def main():
    _weryfikacja()
    print("=" * 70)
    print("ROZSZERZENIA MODELU «ENTROPIA» — LICZBY KLUCZOWE")
    print("=" * 70)

    # ---- R1 ----
    print("\n[R1] SKOŃCZONA TEMPERATURA KĄPIELI")
    print("  η      βΩ      S(∞)     Tr(ρ²)∞   |r|∞      t99%(B)   t99%(A)   stos.")
    for eta in [1.0, 0.5, 0.25, 0.1, 0.01]:
        Seq = S_eq_termiczna(eta)
        tA = czas_do_poziomu_T(M.GAMMA_A, eta, 0.99 * Seq)
        tB = czas_do_poziomu_T(M.GAMMA_B, eta, 0.99 * Seq)
        print(f"  {eta:5.2f}  {-np.log(eta):5.2f}  {Seq:7.4f}  "
              f"{czystosc_równowagowa(eta):8.4f}  {r_eq_termiczny(eta):7.4f}  "
              f"{tB:9.3f}  {tA:9.3f}  {tB/tA:6.1f}")
    # kontrola krzyżowa analityka vs dyskretny Lindblad (η=0.5)
    Sa, Pa, _ = symuluj_termicznie(M.GAMMA_B, 0.5)
    tgrid = np.arange(M.N_TICKS) * M.DELTA_TAU
    blad = np.max(np.abs(Sa - S_termiczna_analitycznie(M.GAMMA_B, 0.5, tgrid)))
    print(f"  kontrola krzyżowa (η=0.5, dyskretny vs analityka): błąd = {blad:.2e}")
    # kompresja czasowa przy η=0.5
    tB = np.linspace(0, 100, 5000)
    blad_c = np.max(np.abs(S_termiczna_analitycznie(M.GAMMA_A, 0.5, tB) -
                          S_termiczna_analitycznie(M.GAMMA_B, 0.5, 27 * tB)))
    print(f"  kompresja 27× przy η=0.5: maks. błąd = {blad_c:.2e}")
    # overshoot dla η=0.2
    t = np.linspace(0, 60, 20000)
    St = S_termiczna_analitycznie(M.GAMMA_B, 0.2, t)
    imax = int(np.argmax(St))
    Seq = S_eq_termiczna(0.2)
    print(f"  overshoot η=0.2: S_max = {St[imax]:.4f} przy t = {t[imax]:.2f}; "
          f"S(∞) = {Seq:.4f}; Δ = {St[imax]-Seq:+.4f} (mapa nieunitalna)")

    # ---- R2 ----
    print("\n[R2] WIELE KUBITÓW")
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    stan11 = stan_poczatkowy_N([ket1, ket1])
    stan10 = stan_poczatkowy_N([ket1, ket0])
    # długie przebiegi (zbieżność do punktu stałego) dla liczb granicznych
    S11, P11, C11, MI11 = symuluj_wspolne(M.GAMMA_B, stan11, N=2, gamma_phi=0.0, n=6000)
    S10, P10, C10, MI10 = symuluj_wspolne(M.GAMMA_B, stan10, N=2, gamma_phi=0.0, n=6000)
    S_nz, _ = symuluj_niezalezne(M.GAMMA_B, 2, gamma_phi=0.0, n=6000)
    # negatywność stanu końcowego (|10⟩ wspólna) — test separowalności
    _, _, _, _, rho_final = symuluj_wspolne(M.GAMMA_B, stan10, N=2, gamma_phi=0.0,
                                            n=6000, zwroc_stan=True)
    NEG10 = negatywnosc2(rho_final)
    print("  scenariusz                    S(∞)      czystość∞  negat.   inf.wzaj.  t90%")
    rows = [
        ("2 niezależne |11⟩", S_nz[-1], 0.25, 0.0, 0.0, czasy_90(S_nz, 2 * M.LN2)),
        ("wspólna |11⟩ (ln 3)", S11[-1], P11[-1], 0.0, MI11[-1], czasy_90(S11, np.log(3))),
        ("wspólna |10⟩ (½ln12)", S10[-1], P10[-1], NEG10, MI10[-1], czasy_90(S10, np.log(12) / 2)),
    ]
    for naz, Sinf, Pinf, NEG, MI, t90 in rows:
        print(f"  {naz:26s} {Sinf:8.4f}  {Pinf:9.4f}  {NEG:8.4f}  {MI:8.4f}  {t90:4d}")
    print(f"  oczekiwane analitycznie: ln 3 = {np.log(3):.4f}, ½·ln 12 = {np.log(12)/2:.4f}, "
          f"2·ln 2 = {2*M.LN2:.4f}, ln(4/3) = {np.log(4/3):.4f}, ln(2/√3) = {np.log(2/np.sqrt(3)):.4f}")
    # ekstensywność
    e_N = [np.max(np.abs(symuluj_niezalezne(M.GAMMA_B, N, gamma_phi=0.0, n=6000)[0] / N -
                   symuluj_niezalezne(M.GAMMA_B, 1, gamma_phi=0.0, n=6000)[0])) for N in (2, 4, 8)]
    print(f"  ekstensywność (max|S(N)/N − S(1)|): N=2 {e_N[0]:.1e}, N=4 {e_N[1]:.1e}, N=8 {e_N[2]:.1e}")

    # ---- R3 ----
    print("\n[R3] SPRZĘŻENIE „ZEGAR → TEMPO”")
    n_fb = 600
    scen = {
        "stały": (FB_STALY, 1.0),
        "chłodzenie α=2": (lambda u: FB_CHLODZENIE(u, 2.0), 2.0),
        "przyspieszanie α=1": (lambda u: FB_PRZYSPIESZANIE(u, 1.0), 1.0),
    }
    print("  scenariusz            t½(A)    t½(B)    stos.  czkanie Δt=0  najdł. zamr.")
    for naz, (fb, a) in scen.items():
        S_A, G_A, _ = symuluj_feedback(M.GAMMA_A, fb, n=n_fb)
        S_B, G_B, _ = symuluj_feedback(M.GAMMA_B, fb, n=n_fb)
        tA = t_do_polowy_ciggly(M.GAMMA_A, fb)
        tB = t_do_polowy_ciggly(M.GAMMA_B, fb)
        dS = np.zeros_like(S_B); dS[1:] = np.maximum(S_B[1:] - S_B[:-1], 0)
        _, dt, _ = M.zegar_stochastyczny(dS, seed=11)
        lo, hi = 40, 200
        nz = int(np.sum(dt[lo:hi] == 0))
        maks = dl = 0
        for i in range(lo, hi):
            if dt[i] == 0:
                dl += 1; maks = max(maks, dl)
            else:
                dl = 0
        print(f"  {naz:22s} {tA:7.2f}  {tB:7.2f}  {tB/tA:6.1f}  {nz:7d}/{hi-lo:<3d}  {maks:8d}")
    # kompresja 27× z chłodzeniem — w czasie CIĄGŁYM, na zakresie przed nasyceniem
    fb_c = lambda u: FB_CHLODZENIE(u, 2.0)
    t_max_A = 2.4                              # 27·t_max_A = 64.8 < zakres B
    t_c, S_A_c = symuluj_feedback_ciggly(M.GAMMA_A, fb_c, t_max_A, n_out=4000)
    t_Bg, S_B_c = symuluj_feedback_ciggly(M.GAMMA_B, fb_c, 70.0, n_out=20000)
    S_B_27 = np.interp(27.0 * t_c, t_Bg, S_B_c)
    blad_c = np.max(np.abs(S_A_c - S_B_27))
    print(f"  kompresja 27× z chłodzeniem (czas ciągły): maks. błąd S_A(t) vs S_B(27t) = {blad_c:.2e}")

    # ---- R4 ----
    print("\n[R4] N=3: SEKTORY j=3/2 ⊕ 2×j=1/2 (Jawne stany subradiantne)")
    b, sector = baza_N3()
    d4 = figura_R4()
    S111, S100, S1S = d4["S111"], d4["S100"], d4["S1S"]
    print("  stan              S(∞)        oczekiwane    czystość∞   t90%")
    for naz, S, ocz, P, t90 in [
        ("|111⟩ (j=3/2)", S111[-1], np.log(4), d4["P111"][-1], d4["t90_111"]),
        ("|100⟩ (mieszany)", S100[-1], np.log(108) / 3, d4["P100"][-1], d4["t90_100"]),
        ("|1⟩⊗|S⟩₂₃ (j=1/2)", S1S[-1], M.LN2, d4["P1S"][-1], d4["t90_1S"]),
    ]:
        print(f"  {naz:22s} {S:8.4f}  {ocz:12.4f}  {P:11.4f}  {t90:4d}")
    print(f"  3 niezależne: 3·ln 2 = {3 * M.LN2:.4f};  deficyt j=3/2: {3*M.LN2-np.log(4):.4f} = ln 2")
    print(f"  populacja |1⟩⊗|S⟩ na kopii B przez cały przebieg: "
          f"min = {d4['popB'].min():.8f} (subradiancja)")

    # ---- R5 ----
    print("\n[R5] PEŁNA ENTROPIA MAKRO (N kubitów)")
    wyn = figura_R5()
    print("  N    niezależne N·ln2   kolektywna ln(N+1)   deficyt     t90% niez.  t90% kolekt.")
    for N in [1, 2, 3, 4]:
        w = wyn[N]
        print(f"  {N:2d}    {N * M.LN2:12.4f}   {np.log(N + 1):14.4f}   "
              f"{N * M.LN2 - np.log(N + 1):9.4f}  {w[3]:9d}  {w[4]:9d}")
    # ekstensywność (N=16)
    S16, _ = symuluj_niezalezne(M.GAMMA_B, 16, gamma_phi=0.0, n=6000)
    print(f"  ekstensywność N=16: S∞ = {S16[-1]:.4f} vs 16·ln 2 = {16 * M.LN2:.4f} "
          f"(błąd {abs(S16[-1] - 16 * M.LN2):.1e})")

    # ---- R6 ----
    print("\n[R6] GORĄCY WIELKI WYBUCH W R3 (start η0=0.95, kąpiel ηB=0.15)")
    ETA0, ETAB = 0.95, 0.15
    d6 = figura_R6()
    print(f"  S(0) = {d6['S0']:.4f} (≈ ln 2 = {M.LN2:.4f});  S(∞) = {d6['Seq']:.4f}")
    print(f"  budżet czasu WSTECZ: |S(0) − S(∞)| = {d6['S0'] - d6['Seq']:.4f} nat")
    print(f"  t½ (do połowy drogi w dół): stałe γ = {d6['t_half_c']:.2f}, "
          f"chłodzenie = {d6['t_half_f']:.2f}")
    print(f"  kontrola krzyżowa (zamknięcie vs mapa Lindblada 4×4): błąd = {d6['blad']:.2e}")
    # czkanie wstecz
    T_w, dt_w, k_w = zegar_wstecz(d6["S_c"], seed=7)
    lo, hi = 200, 500
    nz = int(np.sum(dt_w[lo:hi] == 0))
    print(f"  czkanie wstecz (tyknięcia 200–500): Δt=0 w {nz}/{hi-lo} tyknięciach")

    # ---- R7 ----
    print("\n[R7] KĄPIEL KOLEKTYWNA DLA LOSOWYCH (NIE-SYMETRYCZNYCH) STANÓW")
    d7 = figura_R7()
    print("  N=3, γ_φ=0 → S(∞):")
    for naz, s in d7["wyn3"].items():
        print(f"    {naz:24s} {s:.4f}")
    print("  N=3, γ_φ=γ → S(∞):")
    for naz, s in d7["wyn3d"].items():
        print(f"    {naz:24s} {s:.4f}   (3·ln2 = {3*M.LN2:.4f})")
    print("  N=4 (γ_φ=0 → γ_φ=γ):")
    for naz, (s0, sd) in d7["wyn4"].items():
        print(f"    {naz:16s} {s0:.4f} → {sd:.4f}   (ln5 = {np.log(5):.4f}, 4·ln2 = {4*M.LN2:.4f})")
    # przetrwanie koherencji A↔B dla |100>
    rs = d7["rho_sec"]
    coh_AB = float(np.abs(rs[4, 6]) + np.abs(rs[5, 7]))
    print(f"  koherencja A↔B (|100⟩, γ_φ=0): {coh_AB:.4f} "
          f"(√(pA·pB) = {np.sqrt(1/6*1/2):.4f}) — przeżywa")

    # ---- R8 ----
    print("\n[R8] CYKL WIELKI WYBUCH → EKSPANSJA → OCHŁODZENIE → WIELKI KOLAPS")
    d8 = figura_R8()
    S = d8["S"]; dS = d8["dS"]; imin = d8["imin"]
    frakcja_wstecz = np.mean(dS[1:] < 0)
    # kwantowane czkanie przy zwrocie (okno wokół minimum)
    rng = np.random.default_rng(5)
    zamroz = 0; okno = range(imin - 15, imin + 15)
    for i in okno:
        if rng.poisson(abs(dS[i]) / M.DELTA_S_Q) == 0:
            zamroz += 1
    print(f"  S(0) = {d8['S0']:.4f} (ln2 = {M.LN2:.4f});  S_min = {d8['Smin']:.4f} przy n = {imin}")
    print(f"  budżet czasu wstecz: {d8['budzet']:.4f} nat;  upływ całkowity τ = {d8['t_abs_total']:.4f} = 2·budżet")
    print(f"  czas płynie wstecz przez {100*frakcja_wstecz:.1f}% cyklu")
    print(f"  kwantowe zamrożenia przy zwrocie strzałki: {zamroz}/30 tyknięć")

    # ---- R9 ----
    print("\n[R9] KWANTOWY ZEGAR (czas jako operator)")
    d9 = figura_R9()
    z = d9["z"]
    n = np.arange(len(z["S_sys"]))
    print(f"  γ_t = {d9['gt_main']}: S_sys(∞) = {z['S_sys'][-1]:.4f} (ln2 = {M.LN2:.4f}, "
          f"odch. {z['S_sys'][-1]-M.LN2:+.4f});  ⟨n⟩(∞) = {z['nbar'][-1]:.2f}, "
          f"Δn = {z['dn'][-1]:.2f}, Δn/⟨n⟩ = {z['dn'][-1]/max(z['nbar'][-1],1e-9):.2f}")
    print(f"  I(system;zegar)(∞) = {z['I'][-1]:.4f}")
    print("  kompromis zegara (γ_t: |S(∞)−ln2|, Δn/⟨n⟩):")
    for gt, dv, rel in zip(d9["gts"][1:], d9["dev"][1:], d9["relend"][1:]):
        print(f"    γ_t = {gt:.3f}:  {abs(dv):.4f}    {rel:.2f}")

    # ---- R10 ----
    print("\n[R10] KWANTOWY ZEGAR Z KOHERENCJAMI (start koherentny + dekoherencja zegara)")
    d10 = figura_R10()
    for naz, s in d10["sceny"].items():
        print(f"  {naz:22s}: S∞ = {s['S_end']:.4f} (odch {s['dev']:+.4f})  "
              f"⟨n⟩∞ = {s['nbar']:.2f}  Δn/⟨n⟩ = {s['rel']:.2f}  "
              f"koher. = {s['coh']:.4f}  I∞ = {s['I']:.4f}")
    print(f"  efekt stymulacji: back-action koherentnego vs próżni = "
          f"{d10['sceny']['koherentny κ=0']['dev']/d10['sceny']['próżnia κ=0']['dev']:.1f}×")

    # ---- R11 ----
    print("\n[R11] OD BLOKADY DO PEŁNEJ TERMALIZACJI: PRAWO τ ∝ 1/γ_φ")
    d11 = figura_R11()
    print("  blokada przy γ_φ=0 (N=3): " +
          ", ".join(f"{naz}: {v:.4f} (zablok. {d11['cel3']-v:.3f})"
                    for naz, v in d11["blokada"].items()))
    print("  czas odblokowania τ90(γ_φ) [j. czasu]:")
    print("    γ_φ:      " + "  ".join(f"{g:g}" for g in d11["tau90_3"]))
    print("    τ90 (N=3): " + "  ".join(f"{t:7.0f}" for t in d11["tau90_3"].values()))
    print("    τ90 (N=4): " + "  ".join(f"{d11['tau90_4'].get(g, float('nan')):7.0f}"
                                        for g in [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]))
    print(f"  dopasowanie w reżimie asymptotycznym τ ∝ γ_φ^p: p = {d11['m3']:.2f} (N=3), "
          f"{d11['m4']:.2f} (N=4)  — prawo 1/γ_φ (z podłogą = czas kolektywny)")

    # ---- R13 ----
    print("\n[R13] GRAWITACYJNA PRODUKCJA ENTROPII (dwie kąpiele, NESS)")
    d13 = figura_R13()
    print(f"  S(NESS) = {d13['S_ness']:.4f} < ln 2 = {d13['ln2']:.4f}")
    print(f"  σ_NESS = {d13['sigma']:.5f} (stałe),  σ·τ/tyknięcie = {d13['sigma_tick']:.6f}")
    print(f"  Σσ·τ po 400 tyknięciach = {d13['sum_sig_tau']:.3f}  = {d13['ratio']:.1f} × ln 2 — "
          f"grawitacyjny zegar nigdy nie staje")
    gammas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    sig_ness = []
    for gg in gammas:
        _, sr, sg = symuluj_dwie_kapiele(d13['gamma_r'], d13['eta_r'], gg, d13['eta_g'], n=250)
        sig_ness.append((sr + sg)[-1])
    print("  σ_NESS vs γ_graw: " + ", ".join(f"{gg}: {ss:.4f}" for gg, ss in zip(gammas, sig_ness)))

    # ---- R15 ----
    print("\n[R15] DEKOHERENCJA ZEGARA JAKO STRAŻNIK HISTORII (κ w punkcie zwrotnym)")
    d15 = figura_R15()
    for kap in d15["kappas"]:
        z = d15[f"k{kap}"]
        print(f"  κ = {kap}: punkt zwrotny n ≈ {z['n_turn']}: koherencje {z['coh_turn']:.5f}, "
              f"rozmycie {z['offdiag_turn']:.2e}, S(zegar) {z['Scl_turn']:.4f}, "
              f"I(S;C) {z['I_turn']:.4f};  zapis ⟨n⟩ monotoniczny: {z['mono']}")
    k0, k5 = d15["k0.0"], d15["k0.5"]
    print(f"  supresja rozmycia w punkcie zwrotnym: "
          f"{k0['offdiag_turn']/k5['offdiag_turn']:.0f}× (κ = 0.5)")
    print(f"  nieodwracalność: S(zegar) {k0['Scl_turn']:.3f} → {k5['Scl_turn']:.3f} "
          f"(faza zniszczona);  korelacja kwantowa I {k0['I_turn']:.4f} → {k5['I_turn']:.4f}")

    # ---- R16 ----
    print("\n[R16] FORMALIZM RELACYJNY (rewizja po recenzji: λ → S → τ)")
    d16 = figura_R16()
    pr = d16["pr"]
    print(f"  τ(η=0) = {d16['tau_ent']:.1f} → τ(η=0.5) = {d16['tau_rel']:.1f} "
          f"(korelacje I(A:E) dodają czasu); ln2 = {d16['ln2']:.3f}")
    print(f"  „27” jako predykcja warunkowa: gałąź s (dτ∝s) = {pr['s_branch']:.1f} "
          f"dokładnie; gałąź Ṡ (1. tyknięcie) = {pr['sdot_tick1']:.1f}; "
          f"p_scan (γ∝T^p → 3^p): {pr['p_scan']}")
    print("  → test falsyfikacyjny: zmierz τ_A/τ_B dwóch zegarów ⇒ wyznacz p")

    # ---- R17 ----
    print("\n[R17] LABORATORYJNY TEST: JASNY ↔ CIEMNY ZEGAR ENTROPOWY")
    d17 = figura_R17()
    print(f"  τ(singlet) = {d17['tau_s']:.1f} (zegar MILCZY);  τ(|11⟩) = {d17['tau_11']:.0f}")
    print(f"  tempo ⟨Ṡ⟩: |11⟩ = {d17['rate11']:.4f}, |10⟩ = {d17['rate10']:.4f}, "
          f"singlet = {d17['rate_s']}")
    print(f"  frakcja ciemna p → tempo: " +
          ", ".join(f"{p:.1f}: {r:.4f}" for p, r in zip(d17['ps'], d17['rates'])))
    print(f"  singlet precesuje unitarnie: fidel. → {d17['fid_end']:.3f}, S = {d17['s_inf_singlet']}")
    print("  → przewidywanie: przejście do sektora subradiacyjnego spowalnia zegar")
    print("=" * 70)
    return dict(S11=S11, S10=S10, P10=P10, C10=C10, MI10=MI10, S_nz=S_nz)


def generuj_figury():
    """Generuje figury rozszerzeń; zwraca dane dla raportu."""
    figura_R1()
    figura_R2()
    figura_R3()
    figura_R4()
    figura_R5()
    figura_R6()
    figura_R7()
    figura_R8()
    figura_R9()
    figura_R10()
    figura_R11()
    figura_R13()
    figura_R15()
    figura_R16()
    figura_R17()
    print(f"Figury rozszerzeń zapisano w: {os.path.abspath(OUT)}")


def run_all():
    main()
    generuj_figury()
