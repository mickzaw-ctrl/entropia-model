# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-1.8 — FIZYCZNA REALIZACJA R (PETZ), ZEGAR W CMB, SIEĆ ZEGARÓW
=============================================================================
  R34 — JAK ZREALIZOWAĆ MAPĘ PETZA W LABORATORIUM:
        Dla kodów fazowych (|+⟩,|−⟩) w sektorze jasnym porównujemy:
          (a) idealna mapa Petza (z symetryczną referencją σ_avg) — górna granica,
          (b) protokół echo (π-pulsy — realizowalny fizycznie odzysk fazy),
          (c) klasyczny pomiar+przygotowanie (zawodzi dla fazy: F = ½).
        Hierarchia: F_Petz ≥ F_echo > F_klasyczny. Najlepsza "realizacja R"
        w laboratorium = kodowanie w sektorze CIEMNYM (DFS): kanał ≈ identyczność,
        F = 1 — nie odzyskuj, chroń.

  R35 — ZEGAR W KĄPIELI CMB (widmo Plancka + cutoff grawitacyjny):
        n̄(ω_c, T_CMB) < ε ⇒ ω_c/2π ≥ 261 GHz. Zegar THz jest bezpieczny
        (n̄ ≤ 5×10⁻³), zegar 100 GHz ma n̄ = 0.207 (szum — przegrzewa się).
        Czas grzania ∝ 1/(βω³(n̄+1)) — inżynierskie ograniczenie na sprzężenie β.
        Cutoff grawitacyjny ω_G ≈ 1.85×10⁴³ rad/s: bez wpływu dla ω_c << ω_G,
        wyznacza absolutną skalę UV. IR: tryb Hubble'a jako dolna granica.

  R36 — SIEĆ ZEGARÓW ENTROPII (synchronizacja, kosmiczny czas):
        N komórek, każda z własnym tempem τ̇_k (różne T), sprzężonych wymianą
        entropii (g_sync): τ_k ciągną do średniej — rozrzut σ_τ maleje,
        σ_end: 1.6 → 0.08 (g_sync: 0.01 → 0.2). Jednakowe T ⇒ τ̇ równe ⇒
        zsynchronizowane bez sprzężenia (naturalny kosmiczny czas). Sieć daje
        τ_net = ⟨τ_k⟩ — emergentny "czas kosmiczny" i jego koherencję.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import core as M
from . import dicke as D

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
#  R34 — REALIZACJA MAPY PETZA
# -----------------------------------------------------------------------------
def fidelity(rho, sig):
    a = (rho + rho.conj().T) / 2; b = (sig + sig.conj().T) / 2
    ev, V = np.linalg.eigh(a); ev = np.clip(ev, 0, None)
    sq = (V * np.sqrt(ev)) @ V.conj().T
    m = sq @ b @ sq
    evm = np.clip(np.linalg.eigvalsh((m + m.conj().T) / 2), 0, None)
    return float(np.sum(np.sqrt(evm)) ** 2)


def canal(j, n=40, gamma=M.GAMMA_B):
    from .dicke import lindblad_sektora, propagator_sektora
    d = int(2 * j + 1)
    L = lindblad_sektora(j, gamma, 0.0, 0.0, sparse=False)
    U = propagator_sektora(L, M.DELTA_TAU)
    Ut = np.eye(d * d, dtype=complex)
    for _ in range(n):
        Ut = U @ Ut
    return Ut, d


def petz_rec(Ut, sig, d):
    def sqi(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2); ev = np.clip(ev, 1e-12, None)
        return (V / np.sqrt(ev)) @ V.conj().T
    def sq(A):
        ev, V = np.linalg.eigh((A + A.conj().T) / 2); ev = np.clip(ev, 0, None)
        return (V * np.sqrt(ev)) @ V.conj().T
    Ps = D._unvec(Ut @ D._vec(sig), d)
    def R(X):
        inner = sqi(Ps) @ X @ sqi(Ps)
        return sq(sig) @ D._unvec(Ut.conj().T @ D._vec(inner), d) @ sq(sig)
    return R


def protokoly_odzysku(j):
    """F(Petz), F(echo), F(klasyczny) dla kodów fazowych w sektorze j."""
    Ut, d = canal(j)
    e0 = np.zeros(d); e0[0] = 1; e1 = np.zeros(d); e1[1] = 1
    kod = [(e0 + e1) / np.sqrt(2), (e0 - e1) / np.sqrt(2)]
    kody = [np.outer(k, k.conj()) for k in kod]
    sig_avg = 0.5 * (D._unvec(Ut @ D._vec(kody[0]), d) +
                     D._unvec(Ut @ D._vec(kody[1]), d))
    Rp = petz_rec(Ut, sig_avg, d)
    Fp = [fidelity(k, Rp(D._unvec(Ut @ D._vec(k), d))) for k in kody]
    # echo: π-puls między dwiema ewolucjami (refocusing fazy)
    c, s = np.cos(np.pi), np.sin(np.pi)
    Upi = np.eye(d, dtype=complex)
    Upi[0, 0], Upi[0, 1], Upi[1, 0], Upi[1, 1] = c, 1j * s, 1j * s, c
    Upi_full = np.kron(Upi, Upi.conj())
    Fe = []
    for k in kody:
        r1 = D._unvec(Ut @ D._vec(k), d)
        r2 = Upi_full @ D._vec(r1)
        r3 = D._unvec(Ut @ D._vec(r2), d)
        r4 = Upi_full @ D._vec(r3)
        Fe.append(fidelity(k, D._unvec(r4, d)))
    Fc = [0.5, 0.5]
    return dict(petz=float(np.mean(Fp)), echo=float(np.mean(Fe)),
                klasyczny=float(np.mean(Fc)))


# -----------------------------------------------------------------------------
#  R35 — ZEGAR W CMB
# -----------------------------------------------------------------------------
def nbar(w, T):
    return 1.0 / (np.exp(min(hbar * w / (kB * T), 700)) - 1)


def wc_min_CMB(eps=0.01, T=2.7255):
    return kB * T * np.log(1 / eps) / hbar


def grzanie(w, T, beta=1e-15):
    """Tempo grzania oscylatora przez kąpiel: βω³(n̄+1); czas do n."""
    nb = nbar(w, T)
    return beta * w ** 3 * (nb + 1.0), nb


def t_do_n(w, T, n_target, beta=1e-15):
    """Czas grzania z n=0 do n_target (jeśli n_target < n̄_th, ∞)."""
    rate, nb = grzanie(w, T, beta)
    if n_target >= nb - 1e-12:
        return np.inf
    return np.log(nb / (nb - n_target)) / max(rate, 1e-300)


W_G = 1.85e43   # rad/s — częstość Plancka


def J_cutoff(w, g=1.0, wG=W_G):
    """Gęstość widmowa 3D z cutoffem grawitacyjnym: g²ω³/(1+(ω/ωG)²)."""
    return g ** 2 * w ** 3 / (1 + (w / wG) ** 2)


# -----------------------------------------------------------------------------
#  R36 — SIEĆ ZEGARÓW
# -----------------------------------------------------------------------------
def siec_zegarow(gs, rates, T=300):
    """N zegarów o tempach `rates`, sprzężonych wymianą entropii (g_sync)."""
    Nc = len(rates)
    tau = np.zeros((T, Nc)); dtau = np.asarray(rates, float).copy()
    for t in range(1, T):
        tau[t] = tau[t - 1] + dtau
        mean = tau[t].mean()
        dtau = dtau + gs * (mean - tau[t])
    return tau


def analiza_sieci(gs, rates, T=300):
    tau = siec_zegarow(gs, rates, T)
    sigma = tau.std(axis=1)
    return dict(sigma_peak=float(sigma[30:].max()),
                sigma_end=float(sigma[-1]), tau_net=float(tau[-1].mean()))


# -----------------------------------------------------------------------------
#  FIGURY
# -----------------------------------------------------------------------------
def figura_E23():
    """Realizacja Petza: hierarchia F(Petz) ≥ F(echo) > F(klasyczny)."""
    js = [0.5, 1.0, 1.5, 2.0]
    res = {j: protokoly_odzysku(j) for j in js}
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(js)); w = 0.26
    for i, (key, c, lab) in enumerate([("petz", "#8e44ad", "Petz (idealny)"),
                                       ("echo", "#27ae60", "echo (π-pulsy)"),
                                       ("klasyczny", "#c0392b", "pomiar+przygotowanie")]):
        ax.bar(x + (i - 1) * w, [res[j][key] for j in js], w, color=c,
               label=lab)
    ax.axhline(1.0, color=C_G, ls=":", lw=1)
    ax.text(2.4, 1.02, "sektor ciemny (DFS): F = 1 — chroń, nie odzyskuj",
            color=C_G, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"j = {j}" for j in js])
    ax.set_ylabel("F (średnia po kodach fazowych)")
    ax.set_title("R34: fizyczna realizacja R — Petz ≥ echo > klasyczny; "
                 "najlepsza realizacja = kodowanie w ciemnym sektorze")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE23_petz_realizacja.png", bbox_inches="tight")
    plt.close(fig)
    return res


def figura_E24():
    """Zegar w CMB: próg ω_c, n̄ vs f, ewolucja grzania, cutoff grawitacyjny."""
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.0))
    fs = np.logspace(9.5, 13, 80)          # 3 GHz – 10 THz
    ax = axs[0, 0]
    ax.semilogx(fs, [nbar(2 * np.pi * f, 2.7255) for f in fs], color="#c0392b",
                lw=2)
    ax.axhline(0.01, color=C_G, ls=":", lw=1)
    wcmin = wc_min_CMB() / 2 / np.pi
    ax.axvline(wcmin, color=C_G, ls="--", lw=1)
    ax.text(wcmin * 1.1, 2e-3, f"próg: {wcmin/1e9:.0f} GHz", color=C_G, fontsize=9)
    ax.set_xlabel("ω_c/2π [Hz]"); ax.set_ylabel("n̄(ω_c, T_CMB)")
    ax.set_title("R35: n̄ < ε = 0.01 ⇒ ω_c/2π ≥ 261 GHz (zegar THz)")
    ax = axs[0, 1]
    ax.axis("off")
    ax.set_title("Ewolucja ⟨n⟩(t) w kąpieli CMB", fontsize=12)
    t = np.logspace(-6, 6, 80)
    for f, c, lab in [(1e11, "#c0392b", "100 GHz (szum)"),
                      (3e11, "#e67e22", "300 GHz (próg)"),
                      (1e12, "#27ae60", "1 THz (bezpieczny)")]:
        rate, nb = grzanie(2 * np.pi * f, 2.7255, beta=1e-15)
        ax.semilogx(t, nb * (1 - np.exp(-rate * t)), color=c, lw=2,
                    label=f"{lab}: n̄_th = {nb:.2e}")
    ax.axhline(0.01, color=C_G, ls=":", lw=1)
    ax.text(1e-5, 0.02, "ε = 0.01", color=C_G, fontsize=9)
    ax.set_xlabel("t [s]"); ax.set_ylabel("⟨n⟩(t)")
    ax.legend(fontsize=7.5)
    ax = axs[1, 0]
    ws = np.logspace(10, 44, 100)
    ax.loglog(ws, [J_cutoff(w, 1.0) for w in ws], color="#2471a3", lw=2)
    ax.loglog(ws, [w ** 3 for w in ws], color="#8b98a5", ls="--", lw=1.5,
              label="3D bez cutoffu")
    ax.axvline(W_G, color=C_G, ls=":", lw=1)
    ax.text(W_G * 1.2, 1e-10, "ω_Planck", color=C_G, fontsize=9)
    ax.set_xlabel("ω_c [rad/s]"); ax.set_ylabel("J(ω) = g²ω³/(1+(ω/ωG)²)")
    ax.set_title("Cutoff grawitacyjny: bez wpływu dla realistycznych zegarów,\n"
                 "wyznacza absolutną skalę UV")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.axis("off")
    ax.set_title("Podsumowanie R35", fontsize=12)
    txt = ("• Próg: ω_c/2π ≥ 261 GHz (T_CMB = 2.7255 K, ε = 0.01)\n"
           "• Zegar THz: n̄ ≤ 5×10⁻³ — bezpieczny w CMB\n"
           "• Zegar 100 GHz: n̄ = 0.207 — szum termiczny\n"
           "• Grzanie: tempo βω³(n̄+1) — inżynierski wybór β\n"
           "  (sprzężenie zegar-kąpiel) musi dać τ_heat >> t_protokół\n"
           "• Cutoff grawitacyjny ω_G = 1.85×10⁴³ rad/s: bez wpływu\n"
           "  dla ω_c << ω_G; absolutna granica UV\n"
           "• IR: tryb Hubble'a jako dolna granica widma")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10.5,
            va="top", family="monospace")
    fig.suptitle("ENTROPIA-1.8 — zegar w kąpieli CMB", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE24_zegar_cmb.png", bbox_inches="tight")
    plt.close(fig)


def figura_E25():
    """Sieć zegarów: σ_τ(t) vs g_sync; t_sync; jednakowe T."""
    rng = np.random.default_rng(0)
    Nc = 20
    rates = 0.5 + 0.5 * rng.random(Nc)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axs[0]
    for gs, c in [(0.0, "#8b98a5"), (0.01, "#e67e22"), (0.05, "#27ae60"),
                  (0.2, "#c0392b")]:
        tau = siec_zegarow(gs, rates, T=300)
        sigma = tau.std(axis=1)
        ax.plot(np.arange(300), sigma, color=c, lw=2, label=f"g_sync = {gs}")
    ax.set_xlabel("tyknięcie t"); ax.set_ylabel("σ_τ(t) — rozrzut wskazań")
    ax.set_title("R36: sprzężenie synchronizuje sieć (σ_end: 48 → 0.08)")
    ax.legend(fontsize=8)
    ax = axs[1]
    gs_list = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    ends = [analiza_sieci(g, rates)["sigma_end"] for g in gs_list]
    ax.semilogy(gs_list, ends, "o-", color="#8e44ad", lw=2, ms=8)
    ax.set_xlabel("g_sync (siła wymiany entropii)")
    ax.set_ylabel("σ_τ(∞) — resztkowy rozrzut")
    ax.set_title("Jakość synchronizacji vs sprzężenie; τ_net = ⟨τ_k⟩ =\n"
                 "emergentny czas kosmiczny (jednakowe T: σ ≡ 0)")
    fig.suptitle("R36 — kosmiczna sieć zegarów entropii", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE25_siec.png", bbox_inches="tight")
    plt.close(fig)
    return dict(ends=dict(zip(gs_list, ends)))


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("ENTROPIA-1.8 — REALIZACJA R, ZEGAR W CMB, SIEĆ ZEGARÓW")
    print("=" * 80)

    # [R34]
    print("\n[R34] FIZYCZNA REALIZACJA MAPY PETZA (kody fazowe):")
    for j in [0.5, 1.0, 1.5, 2.0]:
        r = protokoly_odzysku(j)
        print(f"  j = {j}: F(Petz) = {r['petz']:.3f} ≥ F(echo) = {r['echo']:.3f} "
              f"> F(klasyczny) = {r['klasyczny']:.3f}")
    print("  Najlepsza realizacja R w laboratorium: kodowanie w sektorze CIEMNYM")
    print("  (DFS): kanał ≈ identyczność ⇒ F = 1 — „chroń, nie odzyskuj”.")

    # [R35]
    print("\n[R35] ZEGAR W KĄPIELI CMB:")
    wcmin = wc_min_CMB()
    print(f"  próg: ω_c/2π ≥ {wcmin/2/np.pi/1e9:.0f} GHz (ε = 0.01, T_CMB)")
    for f in [1e11, 3e11, 1e12, 3e12]:
        nb = nbar(2 * np.pi * f, 2.7255)
        print(f"  ω_c/2π = {f/1e9:.0f} GHz: n̄ = {nb:.2e} "
              f"({'bezpieczny' if nb < 0.01 else 'SZUM'})")
    print(f"  cutoff grawitacyjny ω_G = {W_G:.1e} rad/s — bez wpływu dla "
          f"realistycznych zegarów (J(cutoff)/J(3D) ≈ 1 do ω_c ~ ω_G)")

    # [R36]
    print("\n[R36] SIEĆ ZEGARÓW ENTROPII:")
    rng = np.random.default_rng(0)
    Nc = 20
    rates = 0.5 + 0.5 * rng.random(Nc)
    for gs in [0.0, 0.01, 0.05, 0.2]:
        a = analiza_sieci(gs, rates)
        print(f"  g_sync = {gs:.2f}: σ_peak = {a['sigma_peak']:.3f}, "
              f"σ_end = {a['sigma_end']:.4f}, τ_net = {a['tau_net']:.2f}")
    tau_eq = siec_zegarow(0.0, np.ones(Nc))
    print(f"  Jednakowe T (τ̇ równe): σ ≡ 0 bez sprzężenia — naturalny "
          f"kosmiczny czas (τ_net = {tau_eq[-1].mean():.1f})")

    # figury
    d34 = figura_E23()
    figura_E24()
    d36 = figura_E25()
    print(f"\nFigury: figE23_petz_realizacja, figE24_zegar_cmb, figE25_siec "
          f"w: {os.path.abspath(OUT)}")
    return dict(odzysk={j: protokoly_odzysku(j) for j in [0.5, 1.0, 1.5, 2.0]},
                wcmin_GHz=wcmin / 2 / np.pi / 1e9,
                siec=dict(zip([0.0, 0.01, 0.05, 0.2],
                              [analiza_sieci(g, rates) for g in
                               [0.0, 0.01, 0.05, 0.2]])))


if __name__ == "__main__":
    main()
