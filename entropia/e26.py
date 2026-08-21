# -*- coding: utf-8 -*-
"""
=============================================================================
  ENTROPIA-6.0 — KOSMOLOGICZNA KALIBRACJA ZEGARA (R51):
                 ZAPIS POMIARU JAKO JEDNOSTKA CZASU
=============================================================================
  Rdzeń modelu (core.py) definiuje: Δt_n = κ·ΔS_n — czas JEST entropią,
  ale κ (nat → sekunda) był dotąd DOWOLNY (jednostki wewnętrzne modelu).
  R51 usuwa tę dowolność: κ jest WYPROWADZONE ze stałych kosmologicznych,
  nie dopasowane.

  IDEA: jednostką czasu jest ZAPIS POMIARU — jedno tyknięcie zegara =
  jeden bit (ln 2 nat) entropii NIEODWRACALNIE zapisany w rejestrze
  (dekoherencja / pomiar). Pytanie: jak długo trwa fizycznie zapisanie
  JEDNEGO bitu, jeśli jedyną dostępną „kąpielą” jest sam Wszechświat
  (horyzont de Sittera wyznaczony przez Λ)?

  ŁAŃCUCH WYPROWADZENIA (wyłącznie stałe fundamentalne: ħ, k_B, c, G, Λ):

    1. Horyzont de Sittera:      R_dS = √(3/Λ)
    2. Parametr Hubble'a (dS):   H_dS = c·√(Λ/3)
    3. Temperatura Gibbonsa-Hawkinga horyzontu:
                                  T_dS = ħ·H_dS / (2π k_B)
    4. Długość Plancka:          l_P = √(ħG/c³)
    5. Entropia horyzontu (Bekenstein-Hawking), w bitach:
                                  S_dS = π R_dS² / (l_P² ln 2)   [bity]
    6. CZAS ZAPISU JEDNEGO BITU (Margolus–Levitin, granica prędkości
       kwantowej dla energii termicznej k_B·T_dS):
                                  τ_rec = π ħ / (2 k_B T_dS)  = π² / H_dS
       (dolna granica Landauera daje τ_L = ħ ln2/(π k_B T_dS), różni się
       stałym czynnikiem — obie podane, τ_rec = ML jest fizycznym „tick”).
    7. Kalibracja κ modelu core.py:   κ_cosmo = τ_rec / ln 2   [s/nat]

  WYNIK FALSYFIKOWALNY (R51):
    Dla Λ = 1.1056×10⁻⁵² m⁻² (Planck 2018):
      T_dS  ≈ 2.21×10⁻³⁰ K         (najzimniejsza możliwa kąpiel)
      S_dS  ≈ 4.71×10¹²² bit        (maks. pojemność rejestru horyzontu)
      τ_rec ≈ 5.42×10¹⁸ s ≈ 1.72×10¹¹ lat  — DŁUŻEJ niż wiek Wszechświata
              (13.8 mld lat)!

    ⇒ Przy CZYSTO kosmologicznym tempie zapisu (T_dS) horyzont de Sittera
      nie zdążyłby zarejestrować JEDNEGO bitu od Wielkiego Wybuchu.
      Rzeczywisty Wszechświat „tyka” dużo szybciej, bo lokalna temperatura
      materii/promieniowania (R1: s ∝ T³) jest o dziesiątki rzędów
      wielkości wyższa niż T_dS — dokładnie zgodne z R6 (gorący Wielki
      Wybuch jako warunek początkowy: entropia startuje wysoko, DALEKO
      od równowagi horyzontu, nie w niej).

    Test falsyfikacji: jeśli zmierzone tempo produkcji entropii
    Wszechświata (obecnie S_obs ~ 10¹⁰⁴ bit w CMB+materii, wiek 13.8 Gyr)
    dawałoby τ_eff < τ_rec(T_dS) — model wymaga LOKALNEJ temperatury
    T_eff > T_dS wielu rzędów wielkości (co jest prawdą: T_CMB ≈ 2.725 K
    ≫ T_dS ≈ 2.2×10⁻³⁰ K — czynnik ~10³⁰, zgodny z przewidywaniem).

  Uczciwa uwaga: τ_rec to DOLNA granica fizyczna (Margolus-Levitin), nie
  pomiar; S_dS to entropia MAKSYMALNA horyzontu (górna granica pojemności
  rejestru), nie entropia obecna Wszechświata. Model NIE twierdzi, że
  zegar kosmiczny faktycznie tyka w tym tempie — wyznacza tylko dwa
  fundamentalne, niepodważalne ograniczenia (najwolniejszy możliwy zapis,
  największy możliwy rejestr), między którymi musi się zmieścić każda
  fizyczna realizacja „czasu jako zapisu pomiaru”.

  Uruchomienie:  python3 -m entropia.e26
  Wymagania:     numpy, matplotlib
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

# ── Stałe fundamentalne (CODATA 2018 / SI) ─────────────────────────────────
HBAR = 1.054571817e-34     # J*s
KB   = 1.380649e-23        # J/K
C    = 2.99792458e8        # m/s
G    = 6.67430e-11         # m^3 kg^-1 s^-2
LN2  = np.log(2.0)

# Konwersje
S_PER_YEAR = 3.15576e7
S_PER_GYR  = 3.15576e16
M_PER_GPC  = 3.0857e25

# Stała kosmologiczna Λ [m^-2] — wartości literaturowe
LAMBDA_PLANCK2018 = 1.1056e-52   # Planck 2018 TT,TE,EE+lowE+lensing
LAMBDA_DESI2024   = 1.0900e-52   # DESI DR2 2024 (LCDM, okolica centralna)

# Referencyjne dane obserwacyjne do testu falsyfikacji
T_CMB      = 2.72548        # K, temperatura CMB dziś
AGE_UNIVERSE_GYR = 13.8      # mld lat, wiek Wszechświata


def dlugosc_plancka():
    """l_P = sqrt(hbar*G/c^3)  [m]"""
    return np.sqrt(HBAR * G / C**3)


def horyzont_desittera(Lam):
    """R_dS = sqrt(3/Lambda)  [m]"""
    return np.sqrt(3.0 / Lam)


def hubble_desitter(Lam):
    """H_dS = c*sqrt(Lambda/3)  [1/s] — asymptotyczny parametr Hubble'a"""
    return C * np.sqrt(Lam / 3.0)


def temperatura_gibbons_hawking(Lam):
    """T_dS = hbar*H_dS / (2*pi*kB)  [K]"""
    H = hubble_desitter(Lam)
    return HBAR * H / (2 * np.pi * KB)


def entropia_horyzontu_bity(Lam):
    """S_dS = pi*R_dS^2 / (l_P^2 * ln2)  [bity] — Bekenstein-Hawking"""
    R = horyzont_desittera(Lam)
    lP = dlugosc_plancka()
    A = 4 * np.pi * R**2
    S_nat = A / (4 * lP**2)      # w jednostkach k_B (nats)
    return S_nat / LN2


def czas_zapisu_bitu_ml(T):
    """tau_ML = pi*hbar / (2*kB*T)  [s] — granica Margolus-Levitin
    (min. czas zmiany stanu kwantowego przy śr. energii k_B*T)."""
    return np.pi * HBAR / (2 * KB * T)


def czas_zapisu_bitu_landauer(T):
    """tau_L = hbar*ln2 / (pi*kB*T)  [s] — dolna granica Landauera
    dla zapisu jednego bitu przy temperaturze T."""
    return HBAR * LN2 / (np.pi * KB * T)


def czas_hubble(Lam):
    """t_H = 1/H_dS  [s]"""
    return 1.0 / hubble_desitter(Lam)


def kalibracja_kosmologiczna(Lam=LAMBDA_PLANCK2018):
    """Pełny łańcuch R51: Λ -> {R_dS, H_dS, T_dS, S_dS, tau_rec, kappa_cosmo}.

    kappa_cosmo [s/nat] kalibruje core.py: Δt_n = kappa_cosmo * ΔS_n
    zamiast dowolnej jednostki wewnętrznej."""
    R_dS = horyzont_desittera(Lam)
    H_dS = hubble_desitter(Lam)
    T_dS = temperatura_gibbons_hawking(Lam)
    S_dS = entropia_horyzontu_bity(Lam)
    t_H = czas_hubble(Lam)
    tau_ml = czas_zapisu_bitu_ml(T_dS)
    tau_l = czas_zapisu_bitu_landauer(T_dS)
    kappa_cosmo = tau_ml / LN2   # sekundy na nat entropii core.py

    return dict(Lambda=Lam, R_dS=R_dS, H_dS=H_dS, T_dS=T_dS, S_dS_bits=S_dS,
                t_H=t_H, tau_ML=tau_ml, tau_Landauer=tau_l,
                kappa_cosmo=kappa_cosmo,
                t_H_Gyr=t_H / S_PER_GYR, tau_ML_Gyr=tau_ml / S_PER_GYR,
                R_dS_Gpc=R_dS / M_PER_GPC)


def test_falsyfikacji_temperatury(Lam=LAMBDA_PLANCK2018):
    """R51 — test falsyfikacji: porównanie T_dS (kosmologiczna, z Λ) z
    T_CMB (obserwowana). Model przewiduje T_CMB >> T_dS o wiele rzędów
    wielkości — inaczej lokalny zegar materii nie mógłby 'tykać' szybciej
    niż zegar horyzontu, co jest sprzeczne z obserwacją (Wszechświat MA
    strukturę, nie jest w martwej równowadze de Sittera)."""
    T_dS = temperatura_gibbons_hawking(Lam)
    stosunek = T_CMB / T_dS
    # przewidywanie: stosunek >> 1 (rzędu 10^30), zgodne z obserwacją
    return dict(T_dS=T_dS, T_CMB=T_CMB, stosunek=stosunek,
                zgodne=bool(stosunek > 1e25))


def figura_51(wynik_planck, wynik_desi):
    """Wykres: porównanie kluczowych skal czasowych R51 (log)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = ["wiek\nWszechświata", "t_H\n(dS, Planck18)", "τ_rec (ML)\nprzy T_dS"]
    vals_yr = [AGE_UNIVERSE_GYR * 1e9,
               wynik_planck["t_H_Gyr"] * 1e9,
               wynik_planck["tau_ML_Gyr"] * 1e9]
    bars = ax.bar(labels, vals_yr, color=["#5a7a9a", "#7b2fff", "#00d4ff"])
    ax.set_yscale("log")
    ax.set_ylabel("czas [lata], skala log")
    ax.set_title("R51 — Skale czasowe: zegar horyzontu de Sittera (Λ) vs wiek Wszechświata")
    for b, v in zip(bars, vals_yr):
        ax.annotate(f"{v:.2e} lat", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "figE44_kalibracja_kosmologiczna.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    print("=" * 79)
    print("  ENTROPIA-6.0 (R51) — Kosmologiczna kalibracja zegara: Λ")
    print("=" * 79)

    for label, Lam in [("Planck 2018", LAMBDA_PLANCK2018), ("DESI 2024", LAMBDA_DESI2024)]:
        w = kalibracja_kosmologiczna(Lam)
        print(f"\n[Λ = {label}: {Lam:.4e} m⁻²]")
        print(f"  R_dS (horyzont)     = {w['R_dS_Gpc']:.3f} Gpc")
        print(f"  H_dS (Hubble, dS)   = {w['H_dS']:.4e} 1/s")
        print(f"  t_H = 1/H_dS        = {w['t_H_Gyr']:.3f} Gyr")
        print(f"  T_dS (Gibbons-Hawking) = {w['T_dS']:.4e} K")
        print(f"  S_dS (horyzont)     = {w['S_dS_bits']:.4e} bit")
        print(f"  τ_rec (Margolus-Levitin, jeden bit @ T_dS) = "
              f"{w['tau_ML']:.4e} s = {w['tau_ML_Gyr']:.4e} Gyr")
        print(f"  τ_Landauer (jeden bit @ T_dS)               = "
              f"{w['tau_Landauer']:.4e} s")
        print(f"  κ_cosmo (s/nat, kalibracja core.py)          = "
              f"{w['kappa_cosmo']:.4e} s/nat")
        print(f"  τ_rec / wiek Wszechświata (13.8 Gyr)         = "
              f"{w['tau_ML_Gyr'] / AGE_UNIVERSE_GYR:.4e}×")

    fals = test_falsyfikacji_temperatury(LAMBDA_PLANCK2018)
    print(f"\n[R51 — test falsyfikacji T_dS vs T_CMB]")
    print(f"  T_dS  = {fals['T_dS']:.4e} K")
    print(f"  T_CMB = {fals['T_CMB']:.5f} K")
    print(f"  T_CMB / T_dS = {fals['stosunek']:.4e}×  "
          f"({'ZGODNE' if fals['zgodne'] else 'NIEZGODNE'} z przewidywaniem R51: ≫10²⁵)")

    w_planck = kalibracja_kosmologiczna(LAMBDA_PLANCK2018)
    w_desi = kalibracja_kosmologiczna(LAMBDA_DESI2024)
    path = figura_51(w_planck, w_desi)
    print(f"\nFigura: {path}")
    return dict(planck=w_planck, desi=w_desi, falsyfikacja=fals)


if __name__ == "__main__":
    main()
