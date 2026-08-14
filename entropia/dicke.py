# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.1 — SYMULACJA N=2..100 W BAZIE DICKEGO
=============================================================================
  Pełna dynamika N kubitów we WSPÓLNEJ (kolektywnej) kąpieli przez rozkład
  na sektory spinu całkowitego j (kolektywne jumpy S± komutują z S², więc
  każdy sektor ewoluuje niezależnie — wymiar sektora = 2j+1 ≤ N+1).

  Dla każdego sektora rozwiązujemy równanie Lindblada w bazie |j,m⟩:
      dρ/dt = −i[H,ρ] + γ·D[S₊] + γ·D[S₋] + γ_φ·D[S_z],   H = (Ω/2)S_z
  (kąpiel nieskończenie gorąca ⇒ unitalna ⇒ każdy sektor dąży do 𝟙/(2j+1),
  S(∞) = ln(2j+1) per sektor; pełny stan to suma blokowa po sektorach).

  Obliczane wielkości (dla N=2..100):
      S(t)      — entropia von Neumanna (suma po sektorach),
      P_dark(t) — populacja w sektorach subradiacyjnych (j < N/2),
      σ(t)      — produkcja entropii (dS/dt; mapa unitalna ⇒ σ ≥ 0),
      I(A:B)    — informacja wzajemna (dwupodział; przez rekurencyjną
                  redukcję symetrycznego stanu),
      τ(t)      — czas relacyjny wg funkcjonału recenzji:
                  dτ/dλ = α·[Ṡ_prod + η·I(A:B)]   (η=0: czysty zegar
                  entropii; η>0: korelacje też napędzają czas).

  Testowane przewidywania:  27×  (kompresja czasowa S_A(t)=S_B(27t)),
  „czkanie” (τ̇ → 0 przy Ṡ → 0), pamięć subradiacyjna (P_dark, I plateau).
=============================================================================
"""

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

from . import core as M


# -----------------------------------------------------------------------------
#  SEKTORY DICKEGO
# -----------------------------------------------------------------------------
def sektory_dickego(N):
    """
    Wszystkie sektory spinu całkowitego dla N kubitów z krotnościami:
      j = N/2, N/2−1, …, (0 lub 1/2);  m(N,j) = (2j+1)·N!/((N/2+j+1)!(N/2−j)!).
    Zwraca listę (j, krotność).
    """
    from math import factorial
    sektory = []
    j = N / 2.0
    while j >= 0:
        jj = int(2 * j)
        mult = (jj + 1) * factorial(N) // (factorial((N + jj) // 2 + 1) *
                                           factorial((N - jj) // 2))
        sektory.append((j, mult))
        j -= 1.0
    return sektory


def macierze_sektora(j):
    """S₊, S₋, S_z w bazie |j,m⟩ (m = −j..j). S₊ PODNOSI m."""
    d = int(2 * j + 1)
    m = np.arange(-j, j + 1.0)
    Sp = np.zeros((d, d))
    Sm = np.zeros((d, d))
    for i in range(d):
        mi = m[i]
        if mi < j:
            Sp[i + 1, i] = np.sqrt(j * (j + 1) - mi * (mi + 1))
        if mi > -j:
            Sm[i - 1, i] = np.sqrt(j * (j + 1) - mi * (mi - 1))
    Sz = np.diag(m)
    return Sp, Sm, Sz


def lindblad_sektora(j, gamma, gamma_phi=0.0, omega=0.0, sparse=True):
    """Superoperator Lindblada (d²×d²) dla sektora j (rzadki dla dużych d)."""
    from scipy.sparse import diags, kron as skron, eye as seye
    d = int(2 * j + 1)
    m = np.arange(-j, j + 1.0)
    # macierze sektora jako rzadkie
    c_plus = np.sqrt(np.maximum(j * (j + 1) - m * (m + 1), 0))[:-1]   # S₊
    c_minus = np.sqrt(np.maximum(j * (j + 1) - m * (m - 1), 0))[1:]   # S₋
    Sp = diags(c_plus, 1, shape=(d, d))
    Sm = diags(c_minus, -1, shape=(d, d))
    Sz = diags(m, 0, shape=(d, d))
    H = (omega / 2.0) * Sz
    I = seye(d)
    L = -1j * (skron(H, I, format="csc") - skron(I, H.T, format="csc"))
    for J, r in [(Sp, gamma), (Sm, gamma), (Sz, gamma_phi)]:
        if r == 0 or J.nnz == 0:
            continue
        Jd = J.conj().T
        JJ = Jd @ J
        term = (skron(J, J.conj(), format="csc")
                - 0.5 * (skron(JJ, I, format="csc")
                         + skron(I, JJ.T, format="csc")))
        L = L + r * term
    if sparse:
        return L.tocsc()
    return L.toarray()


def superoperator_z_jumpami(H, jumps, rates):
    """vec_R(ρ̇) = L vec_R(ρ); konwencja wierszowa (jak core)."""
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for J, r in zip(jumps, rates):
        Jd = J.conj().T
        JJ = Jd @ J
        L += r * (np.kron(J, J.conj()) - 0.5 * (np.kron(JJ, I) + np.kron(I, JJ.T)))
    return L


def propagator_sektora(L, tau):
    """Macierz kroku U = e^{L·τ} (gęsta dla małych sektorów)."""
    return expm(L * tau)


# -----------------------------------------------------------------------------
#  REDUKCJE (dwupodział dla I(A:B))
# -----------------------------------------------------------------------------
def redukcja_symetryczna(rho_N, k):
    """
    Stan zredukowany do k kubitów stanu symetrycznego N-kubitowego
    (sektor j=N/2). Rekurencja po jednym kubicie:
      |j,m⟩ = α|j−½,m−½⟩|0⟩ + β|j+½,m+½⟩|1⟩,  α_m=√((j+m)/2j), β_m=√((j−m)/2j)
    Zwraca (k+1)×(k+1) macierz w bazie symetrycznej k kubitów.
    """
    rho = np.asarray(rho_N, dtype=complex)
    for n in range(rho.shape[0] - 1, k, -1):
        d = n + 1
        alpha = np.sqrt(np.arange(d) / n)
        beta = np.sqrt((n - np.arange(d)) / n)
        # wektoryzacja: ρ' = AρA† + BρB†,  A[q=p−1,p]=α_p, B[q=p,p]=β_p
        A = np.zeros((n, d), dtype=complex)
        A[np.arange(n), np.arange(1, d)] = alpha[1:]
        B = np.zeros((n, d), dtype=complex)
        B[np.arange(n), np.arange(n)] = beta[:n]
        rho = A @ rho @ A.conj().T + B @ rho @ B.conj().T
    return rho


def entropia(rho):
    ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    ev = ev[ev > 1e-15]
    return float(-np.sum(ev * np.log(ev)))


def I_bipartyjny_symetryczny(rho_N, k):
    """I(A:B) = S_k + S_{N−k} − S_N dla symetrycznego stanu (dwupodział k:N−k)."""
    N = rho_N.shape[0] - 1
    Sk = entropia(redukcja_symetryczna(rho_N, k))
    S_Nk = entropia(redukcja_symetryczna(rho_N, N - k))
    SN = entropia(rho_N)
    return Sk + S_Nk - SN


# -----------------------------------------------------------------------------
#  SYMULACJA
# -----------------------------------------------------------------------------
def symuluj_dicke(N, gamma, stan=None, gamma_phi=0.0, omega=M.OMEGA,
                  n=400, delta_tau=M.DELTA_TAU, seed=0):
    """
    Ewolucja N kubitów we wspólnej kąpieli (rozkład na sektory).

    stan: dict {j: (waga, rho_j0)} — stany początkowe sektorów; domyślnie
          |1…1⟩ (czysty sektor symetryczny j=N/2).
    Zwraca słownik historii:
      S, P_dark, sigma (dS/dt), dS, I_AB (dwupodział k=N//2),
      tau_ent (η=0, kwantowany), dtau_ent, tau_rel (η=0.5, z I_AB).
    """
    if stan is None:
        d = N + 1
        rho0 = np.zeros((d, d), complex); rho0[0, 0] = 1.0   # |j=N/2, m=−N/2⟩
        stan = {N / 2.0: (1.0, rho0)}

    # znormalizuj wagi
    tot = sum(w for w, _ in stan.values())
    stan = {j: (w / tot, rho) for j, (w, rho) in stan.items()}

    # przygotuj propagatory per sektor (gęste dla małych d, rzadkie dla dużych)
    prop = {}
    for j, (w, rho0j) in stan.items():
        d = int(2 * j + 1)
        if d <= 32:
            Ld = lindblad_sektora(j, gamma, gamma_phi, omega, sparse=False)
            U = propagator_sektora(Ld, delta_tau)
            prop[j] = ("dense", U)
        else:
            Ls = lindblad_sektora(j, gamma, gamma_phi, omega, sparse=True)
            prop[j] = ("sparse", Ls)

    rho_j = {j: rho0.copy() for j, (w, rho0) in stan.items()}
    jmax = max(stan.keys())

    S = np.zeros(n); P_dark = np.zeros(n); dS = np.zeros(n)
    I_AB = np.zeros(n); tau_ent = np.zeros(n); dtau_ent = np.zeros(n)
    tau_rel = np.zeros(n); dtau_rel = np.zeros(n)
    rng = np.random.default_rng(seed)
    ds_q = M.DELTA_S_Q

    for k in range(n):
        S[k] = _entropia_blokowa(stan, rho_j)
        P_dark[k] = sum(w for j, (w, _) in stan.items() if j < jmax - 1e-9)
        if k > 0:
            dS[k] = max(0.0, S[k] - S[k - 1])
        # I(A:B): symetryczny sektor → dwupodział; inaczej 1:rest (S_1 + S_{N-1} − S_N)
        if len(stan) == 1 and jmax == N / 2.0:
            rhoN = rho_j[jmax]
            I_AB[k] = I_bipartyjny_symetryczny(rhoN, N // 2)
        else:
            I_AB[k] = _I_1rest(stan, rho_j, N)
        # zegar: Δτ = τ₀·ΔS/ΔS_ref (kwantowany, η=0) + η·I (ciągły)
        dtau_ent[k] = rng.poisson(dS[k] / ds_q) * ds_q / 0.01
        tau_ent[k] = tau_ent[k - 1] + dtau_ent[k] if k > 0 else dtau_ent[0]
        dtau_rel[k] = dtau_ent[k] + 0.5 * I_AB[k]
        tau_rel[k] = tau_rel[k - 1] + dtau_rel[k] if k > 0 else dtau_rel[0]
        # krok ewolucji
        for j, rho in rho_j.items():
            kind, U = prop[j]
            d = int(2 * j + 1)
            if kind == "dense":
                rho_j[j] = _unvec(U @ _vec(rho), d)
            else:
                v = _vec(rho)
                rho_j[j] = _unvec(expm_multiply(U * delta_tau, v), d)

    return dict(S=S, P_dark=P_dark, dS=dS, sigma=dS / delta_tau, I_AB=I_AB,
                tau_ent=tau_ent, dtau_ent=dtau_ent, tau_rel=tau_rel,
                dtau_rel=dtau_rel, N=N)


def _vec(rho):
    return rho.flatten()


def _unvec(v, d):
    return np.asarray(v).reshape(d, d)


def _entropia_blokowa(stan, rho_j):
    """Entropia pełnego stanu = Σ_j w_j·S(ρ_j) + H(w) (mieszanina sektorów)."""
    Ss = 0.0
    for j, (w, _) in stan.items():
        Ss += w * entropia(rho_j[j])
    ws = [max(w, 1e-300) for w, _ in stan.values()]
    Ss -= np.sum(ws * np.log(ws))
    return Ss


def _I_1rest(stan, rho_j, N):
    """
    I(1:rest) = S₁ + S_{N−1} − S_N dla stanu będącego mieszaniną sektorów.
    S₁ — z wektora Blocha; S_{N−1} — przez rekurencję sektorową Tr₁.
    """
    # S₁: ⟨σ_z⟩, ⟨σ_x⟩, ⟨σ_y⟩ z każdego sektora
    rz = ry = rx = 0.0
    for j, (w, _) in stan.items():
        rho = rho_j[j]
        Sp, Sm, Sz = macierze_sektora(j)
        Sx = (Sp + Sm) / 2.0
        Sy = (Sp - Sm) / (2j)
        rz += w * np.real(np.trace(Sz @ rho))
        rx += w * np.real(np.trace(Sx @ rho))
        ry += w * np.real(np.trace(Sy @ rho))
    r1 = np.array([2 * rx / N, 2 * ry / N, 2 * rz / N])
    r1 = np.clip(np.linalg.norm(r1), 0, 1)
    S1 = M.entropia(np.diag([(1 + r1) / 2, (1 - r1) / 2]))

    # S_{N−1}: rekurencja Tr₁ po sektorach
    blocks = {}   # j' -> macierz (2j'+1)²
    for j, (w, _) in stan.items():
        rho = rho_j[j]
        d = int(2 * j + 1)
        if j < 1e-9:
            # sektor j=0: Tr₁[|0,0⟩⟨0,0|] = ½·𝟙₂ (blok j'=1/2)
            bl = 0.5 * w * np.eye(2, dtype=complex)
            blocks.setdefault(0.5, np.zeros((2, 2), complex))
            blocks[0.5] += bl
            continue
        if abs(j - N / 2.0) < 1e-9:
            # sektor GÓRNY (j=N/2): pozostaje stan symetryczny (N−1)-kubitowy,
            # blok j'=(N−1)/2 — β-składowe składają się z indeksem m+1
            jl = (N - 1) / 2.0
            dl = int(2 * jl + 1)
            bl = np.zeros((dl, dl), complex)
            alpha = np.sqrt((np.arange(d)) / N)          # √(p/N), p = m + N/2
            beta = np.sqrt((N - np.arange(d)) / N)
            for i in range(dl):
                for ip in range(dl):
                    bl[i, ip] = (alpha[i + 1] * alpha[ip + 1] * rho[i + 1, ip + 1]
                                 + beta[i] * beta[ip] * rho[i, ip])
            blocks.setdefault(jl, np.zeros((dl, dl), complex))
            blocks[jl] += w * bl
            continue
        m_idx = np.arange(d)                      # i_m = m + j
        alpha = np.sqrt(m_idx / (2 * j))          # √((j+m)/(2j))
        beta = np.sqrt((2 * j - m_idx) / (2 * j))
        # blok j−1/2: i = i_m − 1
        if j >= 1.0:
            jl = j - 0.5
            dl = int(2 * jl + 1)
            bl = np.zeros((dl, dl), complex)
            for i in range(dl):
                for ip in range(dl):
                    bl[i, ip] = alpha[i + 1] * alpha[ip + 1] * rho[i + 1, ip + 1]
            blocks.setdefault(jl, np.zeros((dl, dl), complex))
            blocks[jl] += w * bl
        # blok j+1/2: i = i_m + 1
        jh = j + 0.5
        dh = int(2 * jh + 1)
        bh = np.zeros((dh, dh), complex)
        for i in range(dh):
            for ip in range(dh):
                bh[i, ip] = beta[i - 1] * beta[ip - 1] * rho[i - 1, ip - 1]
        blocks.setdefault(jh, np.zeros((dh, dh), complex))
        blocks[jh] += w * bh

    S_Nm1 = sum(entropia(b) for b in blocks.values())
    SN = _entropia_blokowa(stan, rho_j)
    return S1 + S_Nm1 - SN


# -----------------------------------------------------------------------------
#  NARZĘDZIA ANALIZY PRZEWIDYWAŃ
# -----------------------------------------------------------------------------
def kompresja_27(N, gamma, n_cmp=25, stan=None):
    """
    Maks. błąd |S_A(n) − S_B(27n)| w oknie n = 0..n_cmp (przed nasyceniem,
    gdzie obie ewolucje są dynamiczne; B jest 27× wolniejsza, więc porównanie
    w całym przebiegu wymagałoby B o długości 27·n).
    """
    nA = n_cmp
    nB = 27 * n_cmp + 1
    dA = symuluj_dicke(N, gamma * 27.0, stan=stan, n=nA)
    dB = symuluj_dicke(N, gamma, stan=stan, n=nB)
    idx = (np.arange(nA) * 27).astype(int)
    return float(np.max(np.abs(dA["S"] - dB["S"][idx])))


def czkanie_stat(dS, ds=M.DELTA_S_Q, seed=0):
    """Frakcja tyknięć z Δτ = 0 w ogonie (kwantowany zegar entropii)."""
    rng = np.random.default_rng(seed)
    k = rng.poisson(np.maximum(dS, 0) / ds)
    dtau = k * ds
    lo = max(int(0.5 * len(dS)), 1)
    tail = dtau[lo:]
    return float(np.mean(tail == 0.0)) if len(tail) else 0.0


def Haar_pdark(N):
    """P_dark stanu Haar = 1 − (N+1)/2^N (waga sektora symetrycznego)."""
    return 1.0 - (N + 1) / 2.0 ** N


def Haar_Sinf_g0(N):
    """S∞ (γ_φ=0) stanu Haar: Σ_j w_j·ln(2j+1) + H(w), w_j = dim_j/2^N."""
    sek = sektory_dickego(N)
    dims = [m * (2 * j + 1) for j, m in sek]
    tot = sum(dims)
    w = np.array(dims) / tot
    S = np.sum(w * np.array([np.log(2 * j + 1) for j, _ in sek]))
    S -= np.sum(w * np.log(w))
    return S
