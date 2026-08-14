# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-5.0 — PĘTLA POMIAROWA NA PROCESORZE KWANTOWYM (R50)
  IBM QUANTUM / GOOGLE SYCAMORE(-WILLOW): TEST CIEMNEGO SEKTORA
=============================================================================
  Cyfrowy bliźniak protokołu sprzętowego (szkic użytkownika, poprawiony):

    for t in 0..t_max (krok dt):
        # 1. Przygotowanie singletu |D⟩ = (|01⟩−|10⟩)/√2
        #    UWAGA (poprawka): h, cx, x daje Ψ+ (JASNY tryplet)!
        #    Potrzebna bramka z:  h(q0), cx(q0,q1), z(q1), x(q1)  → Ψ−
        # 2. Cykl ewolucji: n = t/dt kroków
        #      rz(Δω·dt) na q0            — zaburzenie (sprzęga |D⟩↔|T0⟩)
        #      kolektywny rozpad S− = σ₋^A+σ₋^B (przez ancilla + reset)
        # 3. Tomografia w bazie Bella: cx(q0,q1), h(q0), pomiar obu
        #      P(|11⟩) = populacja singletu (Ψ− ↔ |11⟩)

  FIZYKA (zweryfikowana numerycznie w tym module):
    • S−|D⟩ = 0  ⇒  singlet CIEMNY dla kąpieli kolektywnej: P_D(t) = 1
    • S−|T0⟩ = √2|00⟩  ⇒  tryplet SUPERROBIETY: P_T0(t) ≈ e^{−2γt}
    • kąpiel NIEZALEŻNA (kontrola falsyfikacyjna): P_D(t) ≈ e^{−γt}
      (koherencja |01⟩⟨10| zanika w tempie γ) — rozróżnienie kąpieli:
      kolektywna: P_D = 1 płasko; niezależna: P_D spada z e^{−γt}.
    • rz na jednym kubicie łamie symetrię ciemną: |D⟩ przecieka do |T0⟩,
      który rozpada się superradiacyjnie ⇒ odblokowanie (analog R11).

  Uczciwe uwagi:
    • „kolektywny rozpad" NIE jest natywną bramką IBM/Sycamore — w tym
      module implementujemy kanał Krausa (cyfrowy bliźniak) oraz unitarną
      osadkę V z ancillą (obwod_kolektywny), którą walidujemy; do bramek
      bazowych (CX/ECR, fSim) trzeba transpilacji (szkic w dokumentacji).
    • Liczby sprzętowe (T1/T2, czasy bramek) to szacunki z publicznych
      specyfikacji 2024–2026 — przed startem należy je zweryfikować.
    • Jest to suchy bieg (digital twin): `run_quantum_processor` = nasz
      symulator; podmiana na prawdziwy backend — patrz dokumentacja.

  Uruchomienie: python3 -m entropia.e25
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figury")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "legend.frameon": False,
})
C_A, C_B, C_G, C_V = "#c0392b", "#2471a3", "#7f8c8d", "#8e44ad"

# -----------------------------------------------------------------------------
#  KONWENCJE (jawnie, bez niejednoznaczności)
#  σ₊ = |1⟩⟨0| = [[0,0],[1,0]]  (podnosi: |0⟩→|1⟩)
#  σ₋ = |0⟩⟨1| = [[0,1],[0,0]]  (opuszcza: |1⟩→|0⟩)
#  UWAGA: core.py projektu ma sp/sm PRZEMIANOWANE (sp=σ₋, sm=σ₊) — tu używamy
#  standardowych operatorów fizycznych.
# -----------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
SIGMA_P = np.array([[0.0, 0.0], [1.0, 0.0]], complex)   # σ₊
SIGMA_M = np.array([[0.0, 1.0], [0.0, 0.0]], complex)   # σ₋
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], complex)

KET00 = np.array([1.0, 0, 0, 0], complex)
KET01 = np.array([0.0, 1, 0, 0], complex)
KET10 = np.array([0.0, 0, 1, 0], complex)
KET11 = np.array([0.0, 0, 0, 1], complex)

# stany Bella (baza obliczeniowa |00⟩,|01⟩,|10⟩,|11⟩)
D   = (KET01 - KET10) / np.sqrt(2.0)     # Ψ− — SINGLET (ciemny)
T0  = (KET01 + KET10) / np.sqrt(2.0)     # Ψ+ — tryplet m=0 (jasny)
PHI_P = (KET00 + KET11) / np.sqrt(2.0)   # Φ+
PHI_M = (KET00 - KET11) / np.sqrt(2.0)   # Φ−


def S_minus():
    """Kolektywny operator opuszczający: S− = σ₋^A + σ₋^B (jawnie)."""
    S = np.zeros((4, 4), complex)
    # S−: |10⟩→|00⟩, |01⟩→|00⟩, |11⟩→|10⟩+|01⟩, |00⟩→0
    S[0, 2] = 1.0     # |10⟩ → |00⟩
    S[0, 1] = 1.0     # |01⟩ → |00⟩
    S[2, 3] = 1.0     # |11⟩ → |10⟩
    S[1, 3] = 1.0     # |11⟩ → |01⟩
    return S


def S_plus():
    return S_minus().conj().T


# -----------------------------------------------------------------------------
#  1. PRZYGOTOWANIE SINGLETU (poprawka błędu użytkownika)
# -----------------------------------------------------------------------------
def bramka_h():
    return np.array([[1.0, 1.0], [1.0, -1.0]], complex) / np.sqrt(2.0)


def bramka_cx():
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]], complex)


def singlet_prep(bug=False):
    """
    U_prep|00⟩ = |D⟩. Sekwencja: h(q0), cx(q0,q1), [z(q1)], x(q1).
    bug=True odtwarza szkic użytkownika (h,cx,x) — daje Ψ+ (JASNY tryplet).
    Zwraca (U, etykieta).
    """
    H = np.kron(bramka_h(), I2)
    CX = bramka_cx()
    Z1 = np.kron(I2, SIGMA_Z)          # z na q1 (drugi kubit)
    X1 = np.kron(I2, np.array([[0, 1], [1, 0]], complex))
    if bug:
        U = X1 @ CX @ H
        return U, "h,cx,x  → Ψ+ (JASNY — BŁĄD w szkicu!)"
    U = X1 @ Z1 @ CX @ H
    return U, "h,cx,z,x → Ψ− (SINGLET — poprawnie)"


def weryfikacja_singlet():
    U, _ = singlet_prep()
    Ub, _ = singlet_prep(bug=True)
    psi = U @ KET00
    psib = Ub @ KET00
    return dict(
        fid_D=float(abs(psi.conj() @ D) ** 2),
        fid_T0=float(abs(psi.conj() @ T0) ** 2),
        fid_D_bug=float(abs(psib.conj() @ D) ** 2),
        fid_T0_bug=float(abs(psib.conj() @ T0) ** 2),
    )


# -----------------------------------------------------------------------------
#  2. KANAŁY ROZPADU (kroki Lindblada, first-order — dokładne Krausy CPTP)
# -----------------------------------------------------------------------------
def kapiel_kolektywna(p):
    """
    Φ_coll(ρ) = M0ρM0† + M1ρM1†,  M1 = √p·S−,  M0 = √(I − p·S+S−).
    S+S− w bazie {|00⟩,|T0⟩,|D⟩,|11⟩} = diag{2,2,0,2}: |D⟩ nietknięty,
    |T0⟩,|00⟩,|11⟩ z √(1−2p). M0†M0 + M1†M1 = I DOKŁADNIE (CPTP).
    """
    Sm = S_minus(); Sp = S_plus()
    B = np.column_stack([KET00, T0, D, KET11])          # baza Dickego
    wl = np.real(np.diag(B.conj().T @ (Sp @ Sm) @ B))   # 2,2,0,2
    M0 = B @ np.diag(np.sqrt(np.clip(1.0 - p * wl, 0, None))) @ B.conj().T
    M1 = np.sqrt(p) * Sm
    return M0, M1


def kapiel_niezalezna(p):
    """Niezależne kąpiele: E0,E1 na każdym kubicie (4 Krausy)."""
    E0 = np.array([[1.0, 0], [0, np.sqrt(1 - p)]], complex)
    E1 = np.array([[0.0, np.sqrt(p)], [0, 0]], complex)
    return [np.kron(e1, e2) for e1 in (E0, E1) for e2 in (E0, E1)]


def cp_check(kap):
    if isinstance(kap, tuple):
        S = sum(M.conj().T @ M for M in kap)
    else:
        S = sum(M.conj().T @ M for M in kap)
    return float(np.max(np.abs(S - np.eye(4))))


def krok(rho, p, kap, dphi=0.0):
    """Jeden krok: rz(Δω·dt) na q0 (jeśli dphi≠0) + kanał rozpadu."""
    if dphi:
        R = np.kron(np.diag([np.exp(-1j * dphi / 2), np.exp(1j * dphi / 2)]), I2)
        rho = R @ rho @ R.conj().T
    out = np.zeros_like(rho, complex)
    for M in kap:
        out += M @ rho @ M.conj().T
    return out


# -----------------------------------------------------------------------------
#  3. OBWÓD Z ANCILLĄ (unitarna osadka kanału kolektywnego)
# -----------------------------------------------------------------------------
def obwod_kolektywny(p):
    """
    V (8×8): |ψ⟩_sys|0⟩_anc → M0|ψ⟩|0⟩_anc + M1|ψ⟩|1⟩_anc (osadka izometryczna).
    Po ancilli: Φ_coll(ρ) = Tr_anc[V(ρ⊗|0⟩⟨0|)V†].
    Do bramek bazowych (CX/ECR): transpilacja — szkic w dokumentacji:
      (i)  rotacja Dickego W: |10⟩↔|01⟩ (√iSWAP-typ, ~2 CX),
      (ii) rotacje Givensa |T0⟩↔|00⟩ i |T0⟩↔|11⟩ sterowane ancillą (~4–6 CX),
      (iii) W†, (iv) reset ancilli. Razem ~8–14 CX/krok.
    """
    M0, M1 = kapiel_kolektywna(p)
    iso = np.zeros((8, 4), complex)
    for i in range(4):
        e = np.zeros(4, complex); e[i] = 1.0
        iso[:4, i] = M0 @ e          # ancilla |0⟩
        iso[4:, i] = M1 @ e          # ancilla |1⟩
    # dopełnienie ORTOGONALNE bez ruszania iso (QR sam dokonuje obrotów/signów
    # w podprzestrzeni osadki, co psuje kanał przy koherencjach — patrz testy):
    B0 = np.zeros((8, 4), complex); B0[4:8, :] = np.eye(4)
    B0 = B0 - iso @ (iso.conj().T @ B0)          # ortogonalizacja wzgl. iso
    Qb, _ = np.linalg.qr(B0)
    V = np.hstack([iso, Qb])
    return V


def obwod_vs_kraus(p, n_prob=25, seed=0):
    """Tr_anc[V(ρ⊗|0⟩⟨0|)V†] ≡ Φ_coll(ρ) dla losowych ρ."""
    rng = np.random.default_rng(seed)
    M0, M1 = kapiel_kolektywna(p)
    V = obwod_kolektywny(p)
    maks = 0.0
    for _ in range(n_prob):
        a = rng.normal(size=4) + 1j * rng.normal(size=4)
        rho = np.outer(a, a.conj()) / np.real(a.conj() @ a)
        rho = (rho + rho.conj().T) / 2
        rho = rho / np.trace(rho)
        rho_k = M0 @ rho @ M0.conj().T + M1 @ rho @ M1.conj().T
        rho_anc = rho.reshape(4, 4)
        rho8 = np.zeros((8, 8), complex)
        rho8[:4, :4] = rho_anc
        rhoV = V @ rho8 @ V.conj().T
        rho_v = rhoV[:4, :4] + rhoV[4:, 4:]
        maks = max(maks, float(np.max(np.abs(rho_k - rho_v))))
    return maks


# -----------------------------------------------------------------------------
#  4. TOMOGRAFIA (baza Bella + pomiar losowy)
# -----------------------------------------------------------------------------
def U_bell():
    """cx(q0,q1), h(q0): Ψ−→|11⟩, Ψ+→|01⟩, Φ−→|10⟩, Φ+→|00⟩."""
    return np.kron(bramka_h(), I2) @ bramka_cx()


def prawd_bell(rho):
    """Prawdopodobieństwa 4 wyników w bazie Bella."""
    U = U_bell()
    r = U @ rho @ U.conj().T
    return np.real(np.diag(r))


def tomografia_bell(rho, shots, seed=0):
    """
    Pomiar w bazie Bella z próbkowaniem multinomialnym.
    Zwraca (p_hat, sigma, P11=witness singletu).
    """
    p = np.clip(prawd_bell(rho), 0.0, 1.0)
    s = p.sum()
    if s > 0:
        p = p / s
    rng = np.random.default_rng(seed)
    k = rng.multinomial(shots, p) if shots > 0 else np.zeros(4, int)
    p_hat = k / shots if shots > 0 else p
    sigma = np.sqrt(p * (1 - p) / shots) if shots > 0 else np.zeros(4)
    return p_hat, sigma, float(p_hat[3])   # |11⟩ = Ψ− = singlet


def pomiar_losowy(rho, shots, seed=0):
    """
    Pomiar losowy (randomized measurements, klasyczny shadow-lite):
    dla każdego strzału losowa baza {X,Y,Z} per kubit; zwraca słownik
    {(bA,bB): counts[4]} — agregat do rekonstrukcji ρ (LS).
    """
    rng = np.random.default_rng(seed)
    bazy = ["X", "Y", "Z"]
    mac = {"X": np.array([[0, 1], [1, 0]], complex),
           "Y": np.array([[0, -1j], [1j, 0]], complex),
           "Z": np.diag([1.0, -1.0]).astype(complex)}
    wyniki = {}
    for _ in range(shots):
        bA, bB = rng.choice(bazy), rng.choice(bazy)
        U = np.kron(_diag_unit(bA, mac), _diag_unit(bB, mac))
        r = U @ rho @ U.conj().T
        pp = np.clip(np.real(np.diag(r)), 0, 1)
        s = pp.sum()
        if s > 0:
            pp = pp / s
        k = rng.multinomial(1, pp)
        wyniki.setdefault((bA, bB), np.zeros(4, int))[int(np.argmax(k))] += 1
    return wyniki


def rekonstrukcja_LS(wyniki, seed=0):
    """
    Najmniejsze kwadraty na częstotliwościach pomiarów losowych:
    dla każdej (baza, wynik) Tr[Pr_{b,o}·ρ] = n_{b,o}/N_b.
    ρ w bazie Pauliego {I,X,Y,Z}⊗2 (16 nieznanych współczynników).
    """
    mac = {"X": np.array([[0, 1], [1, 0]], complex),
           "Y": np.array([[0, -1j], [1j, 0]], complex),
           "Z": np.diag([1.0, -1.0]).astype(complex)}
    ops = [np.eye(2, dtype=complex), mac["X"], mac["Y"], mac["Z"]]
    B = [np.kron(ops[i], ops[j]) for i in range(4) for j in range(4)]
    A = []
    y = []
    for (bA, bB), counts in wyniki.items():
        U = np.kron(_diag_unit(bA, mac), _diag_unit(bB, mac))
        N_b = int(counts.sum())
        if N_b == 0:
            continue
        for o in range(4):
            e = np.zeros(4, complex); e[o] = 1.0
            st = U.conj().T @ e                    # stan własny wyniku o
            Pr = np.outer(st, st.conj())
            A.append([np.real(np.trace(Pr.conj().T @ b)) for b in B])
            y.append(counts[o] / N_b)
    A = np.array(A); y = np.array(y)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    rho = sum(ci * b for ci, b in zip(c, B))
    rho = (rho + rho.conj().T) / 2.0
    ev = np.linalg.eigvalsh(rho)
    if ev.min() < 0:
        rho = rho - ev.min() * np.eye(4)
    return rho / np.trace(rho)


def _diag_unit(baza, mac):
    """Unitaria przenosząca bazę Pauliego do obliczeniowej (X,Y,Z)."""
    if baza == "Z":
        return np.eye(2, dtype=complex)
    if baza == "X":
        return bramka_h()
    # Y: H·S (S = diag(1, i))
    S = np.diag([1.0, 1j])
    return bramka_h() @ S


# -----------------------------------------------------------------------------
#  5. PROTOKÓŁ (pętla użytkownika, poprawna)
# -----------------------------------------------------------------------------
def symuluj_protokol(t_max, dt=0.25, gamma=0.02, mode="kolektywna",
                     delta_omega=0.0, shots=4000, seed=0, zwroc_surowe=False):
    """
    Pętla pomiarowa (suchy bieg):
      for t in 0..t_max krok dt:
        przygotuj |D⟩
        n = t/dt kroków: rz(Δω·dt) na q0 + kanał rozpadu (kolektywny/niezależny)
        tomografia Bella (finite shots)
    Zwraca (ts, P_D_hat, P_D_sigma, P_D_exact).
    """
    p = gamma * dt
    kap = kapiel_kolektywna(p) if mode == "kolektywna" else kapiel_niezalezna(p)
    U0, _ = singlet_prep()
    rho0 = np.outer(U0 @ KET00, (U0 @ KET00).conj())
    ts = np.arange(0.0, t_max + 1e-9, dt)
    P_hat = np.zeros(len(ts)); P_sig = np.zeros(len(ts)); P_ex = np.zeros(len(ts))
    rng = np.random.default_rng(seed)
    surowe = []
    for i, t in enumerate(ts):
        rho = rho0.copy()
        n_st = int(round(t / dt))
        for _ in range(n_st):
            rho = krok(rho, p, kap, dphi=delta_omega * dt)
        p_hat, sig, w = tomografia_bell(rho, shots, seed=int(rng.integers(1 << 30)))
        P_hat[i], P_sig[i] = p_hat[3], sig[3]
        P_ex[i] = prawd_bell(rho)[3]
        surowe.append(rho)
    out = dict(ts=ts, P_hat=P_hat, P_sig=P_sig, P_ex=P_ex, mode=mode,
               delta_omega=delta_omega, shots=shots)
    if zwroc_surowe:
        out["surowe"] = surowe
    return out


def przewidywania(t_max=25.0, dt=0.25, gamma=0.02):
    """Kluczowe predykcje protokołu (liczby do testów i raportu)."""
    z_kol = symuluj_protokol(t_max, dt, gamma, "kolektywna", 0.0, shots=20000)
    z_nz = symuluj_protokol(t_max, dt, gamma, "niezalezna", 0.0, shots=20000)
    z_rz = symuluj_protokol(t_max, dt, gamma, "kolektywna", 0.05, shots=20000)
    # tempo zaniku T0 pod kąpielą kolektywną (superradiancja 2γ) — dokładne
    rhoT = np.outer(T0, T0.conj())
    p = gamma * dt
    M0, M1 = kapiel_kolektywna(p)
    for _ in range(int(t_max / dt)):
        rhoT = M0 @ rhoT @ M0.conj().T + M1 @ rhoT @ M1.conj().T
    P_T0_end = float(np.real(T0.conj() @ rhoT @ T0))
    return dict(
        P_D_kol_end=float(z_kol["P_ex"][-1]),
        P_D_nz_end=float(z_nz["P_ex"][-1]),
        P_D_rz_end=float(z_rz["P_ex"][-1]),
        P_T0_kol_end=P_T0_end,
        exp_m2gt=np.exp(-2.0 * gamma * t_max),
        exp_mgt=np.exp(-gamma * t_max),
        t_max=t_max, gamma=gamma, shots=20000,
        P_D_kol_hat=float(z_kol["P_hat"][-1]),
        P_D_nz_hat=float(z_nz["P_hat"][-1]),
    )


# -----------------------------------------------------------------------------
#  6. BUDŻET SPRZĘTOWY (szacunki publicznych specyfikacji 2024–2026)
# -----------------------------------------------------------------------------
def hardware_zestawienie():
    """
    Szacunki: IBM Quantum Heron r2 (133q) / Google Willow (105q).
    Jeden krok ≈ rz (wirtualne) + kanał kolektywny przez ancillę
    (rotacja Dickego W ≈ 2 CX + 2 rotacje sterowane ≈ 2×3 CX + W†,
    razem ~8–14 CX; na IBM ECR ≈ 2×CX) + reset ancilli ≈ 4–10 μs.
    Budżet: T₂_eff = T₂/4 (dekoherencja w trakcie bramek).
    WAŻNE: γ i dt są wolnymi parametrami modelu (skala niezmiennicza —
    liczy się γ·t). Przy małej liczbie kroków wybieramy γ·dt tak, by
    γ·t_max ≈ 1–2 zmieściło się w budżecie; pomiar RÓŻNICOWY (kolektywna
    vs niezależna na tej samej głębokości) nie wymaga absolutnej kalibracji.
    """
    platforms = [
        dict(nazwa="IBM Heron r2", qubity=133,
             t_krok_us_min=4.0, t_krok_us_max=10.0, T2_us=150.0,
             readout_err=0.005, uwagi="basis ECR/RZ/SX; reset w środku obwodu"),
        dict(nazwa="Google Willow", qubity=105,
             t_krok_us_min=3.0, t_krok_us_max=6.0, T2_us=60.0,
             readout_err=0.01, uwagi="basis fSim/√iSWAP; pomiar środkowy (QEC)"),
    ]
    rows = []
    for pl in platforms:
        T2_eff = pl["T2_us"] / 4.0
        rng_st = (int(T2_eff / pl["t_krok_us_max"]),
                  int(T2_eff / pl["t_krok_us_min"]))
        rows.append(dict(pl, T2_eff_us=T2_eff,
                         max_krokow_zakres=rng_st,
                         t_max_us_zakres=tuple(v * t for v, t in
                                               zip(rng_st, (pl["t_krok_us_max"],
                                                            pl["t_krok_us_min"])))))
    shots_1pct = int(np.ceil(0.25 / (0.01 ** 2)))
    shots_05pct = int(np.ceil(0.25 / (0.005 ** 2)))
    return dict(platformy=rows, shots_1pct=shots_1pct, shots_05pct=shots_05pct)


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E41():
    """Przetrwanie singletu: kolektywna (P_D=1) vs niezależna (e^{-γt});
    tryplet superradiacyjny 2γ; odblokowanie rz."""
    t_max, dt, g = 25.0, 0.25, 0.02
    z_kol = symuluj_protokol(t_max, dt, g, "kolektywna", 0.0, shots=0)
    z_nz = symuluj_protokol(t_max, dt, g, "niezalezna", 0.0, shots=0)
    z_rz1 = symuluj_protokol(t_max, dt, g, "kolektywna", 0.02, shots=0)
    z_rz2 = symuluj_protokol(t_max, dt, g, "kolektywna", 0.05, shots=0)
    # T0 pod kolektywną
    p = g * dt; M0, M1 = kapiel_kolektywna(p)
    rhoT = np.outer(T0, T0.conj()); PT0 = [1.0]
    for _ in range(int(t_max / dt)):
        rhoT = M0 @ rhoT @ M0.conj().T + M1 @ rhoT @ M1.conj().T
        PT0.append(float(np.real(T0.conj() @ rhoT @ T0)))
    PT0 = np.array(PT0)

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.plot(z_kol["ts"], z_kol["P_ex"], color="#27ae60", lw=2.6,
            label="|D⟩ kolektywna: P_D = 1 (ciemny, M = 1)")
    ax.plot(z_nz["ts"], z_nz["P_ex"], color="#c0392b", lw=2.4, ls="--",
            label="|D⟩ niezależna: P_D = e^{−γt} (kontrola falsyfikacyjna)")
    ax.plot(z_kol["ts"], PT0, color="#2471a3", lw=2.2, ls=":",
            label="|T0⟩ kolektywna: P = e^{−2γt} (superradiancja)")
    ax.plot(z_rz1["ts"], z_rz1["P_ex"], color="#8e44ad", lw=1.8,
            label="|D⟩ + rz(Δω=0.02) kolektywna (odblokowanie)")
    ax.plot(z_rz2["ts"], z_rz2["P_ex"], color="#8e44ad", lw=2.2, ls="-.",
            label="|D⟩ + rz(Δω=0.05) kolektywna")
    ax.set_xlabel("t [j. czasu = kroki·dt]"); ax.set_ylabel("P_D(t) (witness Bella |11⟩)")
    ax.set_title("R50: singlet pod kąpielą KOLEKTYWNĄ przetrwa (P_D = 1), "
                 "pod NIEZALEŻNĄ rozpada się e^{−γt};\nrz łamie ciemność "
                 "(przeciek |D⟩→|T0⟩ → rozpad 2γ)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE41_przetrwanie.png", bbox_inches="tight")
    plt.close(fig)
    return dict(kol=z_kol, nz=z_nz, rz1=z_rz1, rz2=z_rz2, PT0=PT0)


def figura_E42():
    """Tomografia Bella z szumem strzałowym: estymaty vs dokładne."""
    t_max, dt, g = 25.0, 0.25, 0.02
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for shots, c in [(400, "#c0392b"), (4000, "#e67e22"), (40000, "#27ae60")]:
        z = symuluj_protokol(t_max, dt, g, "niezalezna", 0.0, shots=shots,
                             seed=1)
        axs[0].errorbar(z["ts"][::4], z["P_hat"][::4], yerr=3 * z["P_sig"][::4],
                        fmt="o", ms=3, color=c, lw=1, capsize=2,
                        label=f"{shots} strzałów (3σ)")
    z0 = symuluj_protokol(t_max, dt, g, "niezalezna", 0.0, shots=0)
    axs[0].plot(z0["ts"], z0["P_ex"], color="#1a5276", lw=2.4,
                label="P_D dokładne = e^{−γt}")
    axs[0].set_xlabel("t"); axs[0].set_ylabel("P̂_D (witness |11⟩)")
    axs[0].set_title("Szum strzałowy: estymata z bazy Bella (3σ)")
    axs[0].legend(fontsize=8)
    # pomiar losowy → rekonstrukcja ρ
    rng = np.random.default_rng(2)
    U0, _ = singlet_prep()
    rho0 = np.outer(U0 @ KET00, (U0 @ KET00).conj())
    p = g * dt; kap = kapiel_niezalezna(p)
    rho = rho0.copy()
    for _ in range(50):
        rho = krok(rho, p, kap)
    for shots, c in [(2000, "#8e44ad"), (16000, "#c0392b")]:
        w = pomiar_losowy(rho, shots, seed=3)
        rho_est = rekonstrukcja_LS(w)
        F = float(np.real(np.trace(rho.conj().T @ rho_est)))
        axs[1].bar([f"{shots}"], [F], color=c, width=0.5,
                   label=f"{shots} strzałów")
    axs[1].set_xlabel("liczba strzałów (pomiar losowy)"); axs[1].set_ylabel("wierność rekonstrukcji F")
    axs[1].set_ylim(0, 1.05)
    axs[1].set_title("Rekonstrukcja ρ z pomiarów losowych (shadow-lite, LS)")
    axs[1].legend(fontsize=8)
    fig.suptitle("R50 — tomografia / pomiar losowy (suchy bieg)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE42_tomografia.png", bbox_inches="tight")
    plt.close(fig)


def figura_E43():
    """Budżet sprzętowy: max kroków vs platforma; strzały vs σ."""
    hw = hardware_zestawienie()
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6))
    nazwy = [pl["nazwa"] for pl in hw["platformy"]]
    los = [pl["max_krokow_zakres"][0] for pl in hw["platformy"]]
    his = [pl["max_krokow_zakres"][1] for pl in hw["platformy"]]
    x = np.arange(len(nazwy))
    axs[0].bar(x, his, 0.5, color="#2471a3", label="max kroków (zakres, T₂/4)")
    axs[0].bar(x, los, 0.5, color="#8e44ad",
               label="min kroków (wolniejszy krok)")
    for xi, (lo, hi) in zip(x, zip(los, his)):
        axs[0].text(xi, hi + 0.3, f"{lo}–{hi}", ha="center", fontsize=10)
    axs[0].set_xticks(x); axs[0].set_xticklabels(nazwy, fontsize=9)
    axs[0].set_title("Budżet koherencji: ile kroków ewolucji zdążymy wykonać")
    axs[0].legend(fontsize=8)
    shots = np.logspace(2, 5, 60)
    sigma = np.sqrt(0.25 / shots)
    axs[1].loglog(shots, sigma, color="#27ae60", lw=2.4)
    axs[1].axhline(0.01, color=C_G, ls=":", lw=1)
    axs[1].text(3e3, 0.011, "σ = 1% (≈2500 strzałów)", color=C_G, fontsize=9)
    axs[1].set_xlabel("strzały"); axs[1].set_ylabel("σ(P̂) dla P = 0.5")
    axs[1].set_title("Budżet strzałów: σ = √(p(1−p)/N)")
    fig.suptitle("R50 — koszt sprzętowy protokołu (szacunki)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE43_hardware.png", bbox_inches="tight")
    plt.close(fig)
    return hw


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 84)
    print("ENTROPIA-5.0 / R50 — PĘTLA POMIAROWA NA PROCESORZE KWANTOWYM (IBM/Sycamore)")
    print("=" * 84)

    print("\n[0] KONWENCJE I POPRAWKA SINGLETU:")
    w = weryfikacja_singlet()
    print(f"  h,cx,z,x → |D⟩: fid = {w['fid_D']:.6f}, |T0⟩: {w['fid_T0']:.1e}")
    print(f"  h,cx,x (szkic użytkownika) → |D⟩: {w['fid_D_bug']:.1e}, "
          f"|T0⟩: {w['fid_T0_bug']:.6f}  ← BŁĄD: to jasny tryplet!")
    print("  Fizyka: S−|D⟩ = 0 (ciemny); S−|T0⟩ norm² = 2 (superradiancja 2γ)")

    print("\n[1] KANAŁY ROZPADU (dokładne Krausy CPTP):")
    p = 0.02 * 0.25
    print(f"  kolektywna: błąd CPTP = {cp_check(kapiel_kolektywna(p)):.2e}")
    print(f"  niezależna: błąd CPTP = {cp_check(kapiel_niezalezna(p)):.2e}")
    blad_circ = obwod_vs_kraus(p)
    print(f"  obwód z ancillą ≡ kanał Krausa: max|Δ| = {blad_circ:.2e}")

    print("\n[2] PRZEWIDYWANIA PROTOKOŁU (t_max = 25, dt = 0.25, γ = 0.02):")
    pr = przewidywania()
    print(f"  |D⟩ kolektywna: P_D = {pr['P_D_kol_end']:.6f}  (ciemny — M = 1)")
    print(f"  |D⟩ niezależna: P_D = {pr['P_D_nz_end']:.6f}  "
          f"(oczek. e^{{−γt}} = {pr['exp_mgt']:.4f})")
    print(f"  |T0⟩ kolektywna: P = {pr['P_T0_kol_end']:.6f}  "
          f"(oczek. e^{{−2γt}} = {pr['exp_m2gt']:.4f}, superradiancja)")
    print(f"  |D⟩ + rz(Δω=0.05): P_D = {pr['P_D_rz_end']:.6f}  (odblokowanie)")
    print(f"  z szumem strzałowym (20k): P̂_D = {pr['P_D_kol_hat']:.4f} (kol.), "
          f"{pr['P_D_nz_hat']:.4f} (niez.)")

    print("\n[3] TOMOGRAFIA:")
    U0, _ = singlet_prep()
    rho0 = np.outer(U0 @ KET00, (U0 @ KET00).conj())
    print(f"  baza Bella: Ψ− ↔ |11⟩ (P = {prawd_bell(rho0)[3]:.4f}); "
          f"Ψ+ ↔ |01⟩")
    wlos = pomiar_losowy(rho0, 16000, seed=3)
    rho_est = rekonstrukcja_LS(wlos)
    F = float(np.real(np.trace(rho0.conj().T @ rho_est)))
    print(f"  pomiar losowy 16k: wierność rekonstrukcji |D⟩ = {F:.4f}")

    print("\n[4] BUDŻET SPRZĘTOWY (szacunki 2024–2026, przed startem zweryfikować):")
    hw = hardware_zestawienie()
    for pl in hw["platformy"]:
        lo, hi = pl["max_krokow_zakres"]
        print(f"  {pl['nazwa']:14s}: {pl['qubity']} q, krok ≈ {pl['t_krok_us_min']:.0f}–"
              f"{pl['t_krok_us_max']:.0f} μs, max kroków ≈ {lo}–{hi}, "
              f"błąd odczytu ≈ {pl['readout_err']:.1%}")
    print(f"  strzały: σ=1% → {hw['shots_1pct']}, σ=0.5% → {hw['shots_05pct']} "
          f"(minuty na IBM)")
    print("  WAŻNE: γ i dt wolne (skala niezmiennicza, liczy się γ·t) — przy "
          "małej liczbie kroków")
    print("  dobrać γ·dt tak, by γ·t_max ≈ 1–2; pomiar różnicowy (kolektywna "
          "vs niezależna,")
    print("  ta sama głębokość) nie wymaga kalibracji absolutnej.")

    print("\n[5] PROTOKÓŁ KOMPLETNY (pętla użytkownika, poprawna):")
    print("  for t in 0..t_max: h,cx,z,x → |D⟩; n=t/dt × (rz(Δωdt) na q0 +")
    print("    kolektywny rozpad przez ancilla+reset); cx,h,pomiar → P(|11⟩)")
    print("  Rozstrzygnięcie: P_D płasko przy 1 ⇔ kąpiel KOLEKTYWNA (model);")
    print("  P_D ≈ e^{−γt} ⇔ kąpiel NIEZALEŻNA (kontrola falsyfikacyjna).")

    figura_E41()
    figura_E42()
    hw2 = figura_E43()
    print(f"\nFigury: figE41_przetrwanie, figE42_tomografia, figE43_hardware "
          f"w: {os.path.abspath(OUT)}")
    return dict(singlet=w, cp=dict(kol=cp_check(kapiel_kolektywna(p)),
                                   nz=cp_check(kapiel_niezalezna(p))),
                circuit_blad=blad_circ, przewidywania=pr, hardware=hw2,
                rekon_F=F)


if __name__ == "__main__":
    main()
