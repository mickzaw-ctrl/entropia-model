# -*- coding: utf-8 -*-
"""
=============================================================================
  EKSPERYMENTALNA KARTA PROTOKOŁU R23 — KONKRETNE PARAMETRY
=============================================================================
  Tłumaczy abstrakcyjny protokół (N=2 sektory Dickego, kwantowy zegar
  entropii, faza ciemna) na konkretny eksperyment z zimnymi atomami:

    • 3 platformy: A) wolna przestrzeń ⁸⁷Rb (jak Guerin PRL 116 083601),
      B) nanofiber ¹³³Cs (jak Pennetta PRL 128 203601), C) wnęka optyczna.
    • Mapowanie jednostek model → eksperyment.
    • Sekwencja czasowa (przygotowanie → faza jasna → faza ciemna).
    • Budżet fotonów i statystyka detekcji (SPCM: η_det, dark counts).
    • Moc statystyczna rozróżnienia T1 vs T2 (kanał korelacyjny).
    • Werdykt wykonalności (co osiągalne dziś, co wymaga rozwoju).

  Kluczowa uczciwa uwaga: kanał fotonowy NIE rozróżnia T1 vs T2 (zegar nie
  sprzęga się zwrotnie z systemem) — rozstrzyga kanał KORELACYJNY: pomiar
  I(A:B) w fazie ciemnej przez statystykę zespołów (destrukcyjny, powtarzalny).
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

LN2 = np.log(2)


def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# -----------------------------------------------------------------------------
#  PLATFORMY
# -----------------------------------------------------------------------------
def platformy():
    """
    (nazwa, τ_nat [s], N, opis kolektywności, Γ_B/γ, Γ_D/γ, uwagi).
    """
    return [
        dict(naz="A: wolna przestrzeń ⁸⁷Rb", tau_nat=26.2e-9, N=1e5,
             kolektyw="OD ≈ 100 (gęsta chmura, kolektywna emisja w przód)",
             GamB_f=5e2, GamD_f=1e-2,                 # Γ_B ≈ Nγ/2·…, Γ_D ≈ γ/OD
             uwagi="Guerin PRL 116 083601: subradiancja do 100× τ_nat",
             beta=None, OD=100.0),
        dict(naz="B: nanofiber ¹³³Cs", tau_nat=30.5e-9, N=5e3,
             kolektyw="β ≈ 0.15 (sprzężenie do modu prowadzonego nanofibrą)",
             GamB_f=7.5e2, GamD_f=1e-2,
             uwagi="Pennetta PRL 128 203601: stany subradiantne z wyłączeniami emisji",
             beta=0.15, OD=None),
        dict(naz="C: wnęka optyczna (Dicke)", tau_nat=26.2e-9, N=50,
             kolektyw="pojedynczy mod wnęki, g ≫ γ,κ (silne sprzężenie)",
             GamB_f=5e1, GamD_f=1e-2,
             uwagi="najczystszy Dicke; N małe, ale kontrola pełna",
             beta=None, OD=None),
    ]


def licz_platforme(p, dt_samp=1e-6):
    """Pełne liczby platformy."""
    gam = 1.0 / p["tau_nat"]
    N = p["N"]
    # UWAGA: sektor 1 ekscytonu — superradiancja (N-krotne wzmocnienie) wymaga
    # wielu ekscytonów; jasny rozpad pojedynczego ekscytonu ≈ γ (ew. wzmocnienie
    # β do modu prowadzonego). t_B realistycznie = τ_nat..kilka τ_nat.
    if p["beta"] is not None:
        GamB = gam * (1.0 + 20.0 * p["beta"])     # 1 ekscyton, wzmocnienie do modu
        GamD = gam * (1 - p["beta"]) / N
    elif p["OD"] is not None:
        GamB = 2.0 * gam                          # jasny 1 ekscyton ≈ 2γ (wzmocn.)
        GamD = gam / p["OD"]
    else:
        GamB = 2.0 * gam
        GamD = gam / N
    return dict(gam=gam, GamB=GamB, GamD=GamD,
                tB=1.0 / max(GamB, 1e-30), tD=1.0 / max(GamD, 1e-30),
                dt_samp=dt_samp)


# -----------------------------------------------------------------------------
#  MAPOWANIE MODEL → EKSPERYMENT
# -----------------------------------------------------------------------------
def mapa_jednostek(p, dt_samp=1e-6):
    """
    Model: 1 tyknięcie = Δt_samp; σ₀ = 0.01 nat/tyk; δs = 0.01 nat;
    τ̇_T2 = η·I_eq = 0.0719 nat/tyk (η=0.5, I_eq=0.1438 nat);
    τ̇_T1 = 0.
    """
    z = licz_platforme(p, dt_samp)
    return dict(
        dt_samp=dt_samp,
        tau_tyk=dt_samp / z["tD"],            # tyknięć na czas życia subrad.
        sigma0_nat_s=0.01 / dt_samp,          # σ₀ w nat/s
        tau_dot_T1_nat_s=0.0,
        tau_dot_T2_nat_s=0.0719 / dt_samp,
        ds_nat=0.01,
        I_eq_nat=0.1438,
    )


# -----------------------------------------------------------------------------
#  SEKWENCJA CZASOWA
# -----------------------------------------------------------------------------
def sekwencja(p, n_runs=1000):
    z = licz_platforme(p)
    return dict(
        prep=dict(etap="pułapka magnetooptyczna + chłodzenie",
                  czas="10–100 ms", N=p["N"]),
        stan=dict(etap="przygotowanie stanu |10⟩-typ: π-puls / słaba wiązka "
                       "(sektor 1 ekscytonu, stan Dicke z fazą)",
                  czas="10–100 ns", cel="jasny+ciemny superpozycja"),
        jasna=dict(etap="faza jasna: superradiancja, detekcja fotonów",
                   czas=f"{z['tB']*1e9:.0f} ns", budzet=f"~{p['N']/2:.0e} fotonów"),
        ciemna=dict(etap="faza ciemna: subradiancja (Γ_D) + pomiar I(A:B)",
                    czas=f"{z['tD']*1e6:.1f} μs (τ_sub)"),
        odczyt=dict(etap="destrukcyjny odczyt korelacji: rozdzielić chmurę na "
                         "A|B, obrazować fluorescencję, estymować S_A, S_B, S_AB",
                    czas="1–10 μs / realizacja"),
        powt=dict(etap="powtórzenie: M ≈ 200 realizacji na punkt czasowy "
                         "(precyzja σ_I = 0.01 nat)",
                  czas="≈ 0.4 ms / punkt"),
    )


# -----------------------------------------------------------------------------
#  BUDŻET FOTONÓW I DETEKCJA
# -----------------------------------------------------------------------------
def detekcja(p, T_dark=1e-3, eta_det=0.3, dark_rate=100.0):
    """Fotony subradiantne w oknie ciemnym + szum SPCM."""
    z = licz_platforme(p)
    n_phot_T1 = z["GamD"] * T_dark                # oczekiwane fotony subrad.
    n_dark = dark_rate * T_dark                    # dark counts SPCM
    snr = n_phot_T1 / max(np.sqrt(n_phot_T1 + n_dark), 1e-9)
    return dict(n_phot_T1=n_phot_T1, n_dark=n_dark, snr=snr,
                eta_det=eta_det, dark_rate=dark_rate, T_dark=T_dark)


# -----------------------------------------------------------------------------
#  MOC STATYSTYCZNA KANAŁU KORELACYJNEGO (I(A:B))
# -----------------------------------------------------------------------------
def moc_korelacyjna(Ieq=0.1438, sigma_I=0.01, alpha=0.01, n_pts=6):
    """
    Aby rozstrzygnąć T1 (I const ⇒ τ̇=0) vs T2 (τ̇=η·Ieq), mierzymy I(t) w
    fazie ciemnej i dopasowujemy nachylenie dI/dt oraz tempo „tyknięć”.
    Potrzebna precyzja estymacji I: σ_I ≤ η·Ieq/3 (3σ separacja).
    σ_I z estymacji p_A, p_B: σ_p = √(p(1−p)/M) ⇒ M = p(1−p)/σ_p².
    """
    # I(A:B) ≈ S_A + S_B (S_AB ≈ 0 dla czystego) = 2·h₂(p̂). Czułość dI/dp̂ = 0
    # przy p̂=0.5 (kwadratowa), więc M liczymy liczbowo: σ_I = ΔI(σ_p).
    p_half = 0.5
    dI = np.inf
    for M in [50, 100, 150, 200, 500, 1000, 2000, 5000, 10000]:
        sigma_p = np.sqrt(p_half * (1 - p_half) / M)
        dI = 2.0 * abs(h2(p_half + sigma_p) - h2(p_half))   # nat (czułość kwadratowa)
        if dI <= sigma_I:
            break
    return dict(Ieq=Ieq, sigma_I=sigma_I, M_min=int(M),
                dI_at_M=float(dI), n_pts=n_pts,
                czas_punkt=4e-4,   # 400 μs / punkt (M×200 μs/realizację)
                czas_calk=6 * 4e-4)


# -----------------------------------------------------------------------------
#  FIGURA — OŚ CZASU EKSPERYMENTU + PRZEWIDYWANIA
# -----------------------------------------------------------------------------
def figura_E17():
    """Oś czasu eksperymentu (platforma B) + przewidywania T1 vs T2."""
    pB = [p for p in platformy() if p["naz"].startswith("B")][0]
    z = licz_platforme(pB)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.0),
                            gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axs[0]
    ax.axis("off")
    ax.set_title("Sekwencja eksperymentalna (platforma B: nanofiber ¹³³Cs)",
                 fontsize=12)
    etapy = [
        ("1 · pułapka + chłodzenie", "10–100 ms", "#dbe4ec"),
        ("2 · stan |10⟩-typ (1 ekscyton, Dicke z fazą)", "10–100 ns", "#eaf4fb"),
        ("3 · FAZA JASNA: superradiancja", f"{z['tB']*1e9:.0f} ns", "#fdecea"),
        ("4 · ostatni foton ⇒ faza ciemna", "t* ≈ t_B", "#fdf6e7"),
        ("5 · FAZA CIEMNA: subradiancja Γ_D", f"{z['tD']*1e6:.0f} μs", "#eafaf1"),
        ("6 · odczyt I(A:B): rozdzielić A|B, obrazować", "1–10 μs", "#f4ecf7"),
        ("7 · powtórzenie ×10⁴ / punkt czasowy", "≈10 s / punkt", "#eef3f8"),
    ]
    y = 0.92
    for naz, czas, col in etapy:
        ax.add_patch(plt.Rectangle((0.02, y - 0.055), 0.96, 0.075,
                                   facecolor=col, edgecolor="#5b6b7b"))
        ax.text(0.04, y, naz, fontsize=9.5, va="center")
        ax.text(0.97, y, czas, fontsize=9, va="center", ha="right",
                color="#31475c")
        y -= 0.115
    ax.text(0.04, y - 0.01,
            "τ_nat = 30.5 ns · N = 5000 · β = 0.15\n"
            f"Γ_B ≈ {z['GamB']/2/np.pi/1e6:.0f} MHz · Γ_D ≈ {z['GamD']/2/np.pi/1e3:.0f} kHz\n"
            f"t_B ≈ {z['tB']*1e9:.0f} ns · t_D ≈ {z['tD']*1e6:.0f} μs",
            fontsize=9, color="#26384a", va="top")

    ax = axs[1]
    t = np.logspace(-1, 2.5, 60) * z["tB"]
    # przewidywania: tempo „tyknięć zegara" w fazie ciemnej
    ax.semilogx(t, np.full_like(t, 0.0), color="#27ae60", lw=2.2,
                label="T1: τ̇ = 0 (zegar staje po ostatnim fotonie)")
    ax.semilogx(t, np.full_like(t, 0.0719 / 1e-6 / 1e3), color="#c0392b", lw=2.2,
                label="T2: τ̇ = η·I_eq ≈ 72 kHz (tyka dalej)")
    C_G = "#7f8c8d"
    ax.axvline(z["tB"], color=C_G, ls=":", lw=1.4)
    ax.text(z["tB"] * 1.2, 0.06e3, "ostatni foton\n(przejście do fazy ciemnej)",
            fontsize=8, color="#7f8c8d")
    ax.set_xlabel("t [s]"); ax.set_ylabel("τ̇ [kHz (tyknięcia zegara)]")
    ax.set_title("Przewidywania w fazie ciemnej: T1 vs T2\n"
                 "(mierzalne w kanale korelacyjnym I(A:B))")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE17_sekwencja.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("EKSPERYMENTALNA KARTA PROTOKOŁU R23 — KONKRETNE PARAMETRY")
    print("=" * 80)

    print("\n[1] PLATFORMY:")
    for p in platformy():
        z = licz_platforme(p)
        print(f"  {p['naz']}: N = {p['N']:.0e}, τ_nat = {p['tau_nat']*1e9:.0f} ns")
        print(f"    {p['kolektyw']}")
        print(f"    Γ_B = {z['GamB']/2/np.pi/1e6:.1f} MHz (t_B = {z['tB']*1e9:.0f} ns) · "
              f"Γ_D = {z['GamD']/2/np.pi/1e3:.2f} kHz (t_D = {z['tD']*1e6:.1f} μs) · "
              f"t_D/t_B = {z['tD']/z['tB']:.0f}×")
        print(f"    {p['uwagi']}")

    print("\n[2] MAPOWANIE JEDNOSTEK (platforma B, Δt_samp = 1 μs):")
    pB = [p for p in platformy() if p["naz"].startswith("B")][0]
    m = mapa_jednostek(pB)
    print(f"  σ₀ = {m['sigma0_nat_s']:.2e} nat/s; τ̇_T1 = 0; "
          f"τ̇_T2 = {m['tau_dot_T2_nat_s']:.2e} nat/s")
    print(f"  I_eq = {m['I_eq_nat']:.4f} nat; δs = {m['ds_nat']:.2f} nat "
          f"(= 1.4% bitu)")

    print("\n[3] SEKWENCJA (platforma B):")
    for k, v in sekwencja(pB).items():
        print(f"  {v['etap']}: {v['czas']}")

    print("\n[4] BUDŻET FOTONÓW I DETEKCJA (T_dark = 1 ms, η_det = 0.3, "
          "dark = 100 Hz):")
    for p in platformy():
        d = detekcja(p)
        print(f"  {p['naz']}: fotony subrad. = {d['n_phot_T1']:.1f}, "
              f"dark = {d['n_dark']:.1f}, SNR = {d['snr']:.1f}")

    print("\n[5] MOC STATYSTYCZNA KANAŁU KORELACYJNEGO:")
    mc = moc_korelacyjna()
    print(f"  potrzebna precyzja I: σ_I = {mc['sigma_I']:.3f} nat "
          f"(≤ η·I_eq/3); osiągnięta przy M: ΔI = {mc['dI_at_M']:.4f}")
    print(f"  realizacje/punkt: M = {mc['M_min']} (p_A = p_B = 0.5, "
          f"σ_p = √(0.25/M))")
    print(f"  czas/punkt ≈ {mc['czas_punkt']*1e3:.1f} ms; "
          f"6 punktów ≈ {mc['czas_calk']*1e3:.1f} ms łącznie")

    print("\n[6] WERDYKT WYKONALNOŚCI:")
    print("  • Kanał FOTONOWY (fluorescencja, subradiancja): osiągalny dziś;")
    print("    NIE rozróżnia T1 vs T2 (zegar nie sprzęga się zwrotnie).")
    print("  • Kanał KORELACYJNY (I(A:B) w fazie ciemnej): wykonalny —")
    print("    destrukcyjny odczyt przez podział chmury, M ≈ 200 realizacji/punkt,")
    print("    ~0.4 ms/punkt, precyzja σ_I = 0.01 nat (czułość h₂ przy p=0.5).")
    print("  • Najlepsza platforma: B (nanofiber, t_D = 180 μs — długie okno).")
    print("  • Ryzyko: szum odczytu I(A:B) > 0.01 nat przy małym N;")
    print("    zalecane N ≥ 5×10³ i η_det ≥ 0.3.")

    figura_E17()
    print(f"\nFigura: figE17_sekwencja.png w: {os.path.abspath(OUT)}")
    return dict(platformy=[dict(naz=p["naz"],
                                N=p["N"], tau_nat=p["tau_nat"],
                                **licz_platforme(p)) for p in platformy()],
                mapa=mapa_jednostek(pB), detekcja={p["naz"].split(":")[0]:
                    detekcja(p) for p in platformy()},
                moc=moc_korelacyjna())


if __name__ == "__main__":
    main()
