# -*- coding: utf-8 -*-
"""
Testy ENTROPIA-4.0 (R48: dwie komórki z wymianą entropii; R49: entropowa
siła + emergentna FRW). Liczby w asercjach pochodzą z uruchomień
(python3 -m entropia.e24) — dynamika deterministyczna (bez RNG).
Uruchomienie: python3 -m pytest tests/test_e24.py -q
"""
import numpy as np
import pytest

from entropia import e24 as E


# =============================================================================
#  R48 — NESS dwukomórkowy
# =============================================================================
def _z():
    return E.symuluj_dwie_komorki()


def test_ness_stan_stacjonarny():
    """S_tot → const (dS/dt → 0), ale σ_tot_NESS > 0 (produkcja trwa)."""
    z = _z()
    d = E.liczby_ness(z)
    assert abs(d["dSdt_inf"]) < 1e-6
    assert d["sig_tot_inf"] > 1e-4
    assert abs(d["S_tot_end"] - 1.3491) < 1e-3        # S_tot(∞) ≈ 1.349 nat
    assert d["S_tot_end"] < np.log(4.0)               # NESS < pełne wymieszanie


def test_clausius():
    """σ_NESS = J_E,∞·(1/T_B − 1/T_A) — zgodność ~1e-6 (Clausius/Onsager)."""
    d = E.liczby_ness(_z())
    assert d["J_E_inf"] > 0                            # energia płynie A→B
    assert abs(d["ratio_clausius"] - 1.0) < 1e-3


def test_produkcja_przy_zimnej():
    """W NESS produkcja dominuje w zimnej komórce (σ_B ≫ σ_A, Clausius)."""
    d = E.liczby_ness(_z())
    assert d["sig_B_inf"] > 10 * d["sig_A_inf"]
    assert d["sig_ex_inf"] > 0


def test_wymiana_zachowuje_energie():
    """Kanał wymiany sam zachowuje E_A + E_B (każdy krok |10⟩↔|01⟩)."""
    z = E.symuluj_dwie_komorki(gamma_A=0.0, gamma_B=0.0, kappa=0.3)
    Etot = z["E_A"] + z["E_B"]
    assert np.max(np.abs(Etot - Etot[0])) < 1e-12
    assert z["E_A"][-1] < z["E_A"][0]                 # A traci, B zyskuje


def test_fourier():
    """Prawo Fouriera: J_E,∞ rośnie z ΔT; quasi-liniowe przy małych ΔT."""
    fou = E.skan_fouriera()
    Js = [r["J"] for r in fou]
    assert all(Js[i] < Js[i + 1] for i in range(len(Js) - 1))   # monotonicznie
    # subliniowość: J(ΔT=1.5)/J(ΔT=0.5) < stosunek ΔT = 3 (nasycenie)
    assert 1.5 < Js[2] / Js[0] < 3.0


def test_spohn_nieujemne():
    """σ_A, σ_B, σ_ex ≥ 0 w każdej chwili (tw. Spohna, każdy kanał osobno)."""
    z = _z()
    assert np.all(z["sig_A"] >= -1e-12)
    assert np.all(z["sig_B"] >= -1e-12)
    assert np.all(z["sig_ex"] >= -1e-12)


def test_odsprezenie():
    """κ=0: komórki niezależne — S_A/S_B ≡ jednokubitowe symulacje termiczne."""
    w = E.weryfikacja_odsprezenia()
    assert w["blad_A"] < 1e-10
    assert w["blad_B"] < 1e-10


# =============================================================================
#  R49 — entropowa siła
# =============================================================================
def test_entropowa_sila_przyciaga():
    """S∞(κ) ściśle rośnie (zbieżny NESS) ⇒ ∂S/∂κ > 0 ⇒ F(d) < 0."""
    es = E.entropowa_sila()
    assert all(es["Sinf"][i] < es["Sinf"][i + 1] for i in
               range(len(es["Sinf"]) - 1))
    assert np.all(es["dSdk"] > 0)
    assert np.all(es["F_d"] < 0)                       # przyciągające w każdym d
    # znika w dużych d (κ → 0, brak sprzężenia)
    imin = int(np.argmin(es["F_d"]))
    assert abs(es["F_d"][-1]) < 0.5 * abs(es["F_d"][imin])


def test_Sinf_zakres():
    """Zakres S∞(κ): od ~1.265 (słabe sprzężenie) do ~1.355 (silne);
    limit κ→0 = niezależne Gibbsy ≈ 1.2617 (poniżej skanu)."""
    es = E.entropowa_sila()
    assert es["Sinf"][0] > 1.2 and es["Sinf"][0] < 1.30
    assert es["Sinf"][-1] > 1.35 and es["Sinf"][-1] < 1.36
    # limit κ→0: suma entropii Gibbsa komórek (niezależne)
    S_A_G = E._H(np.array([1 / (1 + E.ETA_A), E.ETA_A / (1 + E.ETA_A)]))
    S_B_G = E._H(np.array([1 / (1 + E.ETA_B), E.ETA_B / (1 + E.ETA_B)]))
    assert abs((S_A_G + S_B_G) - 1.26165) < 1e-3
    assert es["Sinf"][0] > (S_A_G + S_B_G)            # zbieżny NESS > limit


# =============================================================================
#  R49 — emergentna FRW
# =============================================================================
def test_frw_inwersja_i_osobliwosc():
    """Start w inwersji (T_eff < 0); przejście przez T = ∞ = osobliwość a = 0."""
    fr = E.emergentna_frw()
    ic = fr["i_cross"]
    assert 0 < ic < len(fr["t"]) // 2
    assert fr["T_eff"][0] < 0                          # inwersja (T < 0)
    assert fr["T_eff"][ic] > 5                         # po przejściu (T duże)
    # maksimum |T_eff| w okolicy przejścia (skok przez ±∞ między krokami)
    okno = slice(max(0, ic - 3), min(len(fr["t"]), ic + 3))
    assert np.max(np.abs(fr["T_eff"][okno])) > 50
    assert fr["T_eff"][-1] > 0 and fr["T_eff"][-1] < E.TA   # T_NESS < T_A
    assert fr["a"][ic] == 0.0                          # osobliwość (Wielki Wybuch)
    assert fr["z"][ic] == np.inf
    # ekspansja: a rośnie po osobliwości; koniec: a → 1 (śmierć cieplna)
    assert fr["a"][-1] > 0.99
    assert abs(fr["z"][-1]) < 1e-3


def test_frw_odbicie():
    """T_eff przestrzeliwuje poniżej T_NESS ⇒ a: 0 → a_max > 1 → 1 (odbicie)."""
    fr = E.emergentna_frw()
    ic = fr["i_cross"]
    ia = int(np.argmax(fr["a"]))
    assert fr["a"][ia] > 1.1                           # ekspansja ponad 1
    assert fr["t"][ia] > fr["t"][ic]                   # po Wielkim Wybuchu
    # po maksimum: kontrakcja (a maleje do 1)
    assert fr["a"][-1] < fr["a"][ia]
    # T_eff: przestrzeliwuje poniżej wartości NESS
    assert fr["T_eff"][ia] < fr["T_eff"][-1]


def test_frw_H_koniec():
    """H: dodatnie (ekspansja), zmiana znaku (odbicie), H → 0 (śmierć cieplna)."""
    fr = E.emergentna_frw()
    m = fr["t"] >= fr["t"][fr["i_cross"]] + 0.1
    hm = fr["H"][m]
    assert np.any(hm[20:200] > 1.0)                    # szczyt ekspansji
    # dokładnie jedna zmiana znaku + → − (odbicie)
    znaki = np.sign(hm[20:-200])
    zmiany = np.sum(np.diff(znaki) != 0)
    assert zmiany == 1
    # ogon: H ≈ 0 (szum gradientu ~1e-10, ekspansja zamiera)
    assert np.max(np.abs(hm[-200:])) < 1e-6
    assert abs(fr["H"][-1]) < 1e-5                     # koniec: H → 0


def test_czasy_sys_bud():
    """τ_sys skończony (śmierć cieplna), τ_bud rośnie liniowo w NESS."""
    fr = E.emergentna_frw()
    d = E.liczby_ness(_z())
    assert 1.2 < fr["tau_sys"][-1] < 1.5              # skończony wiek układu
    assert fr["tau_bud"][-1] > fr["tau_sys"][-1]
    # nachylenie ogona τ_bud ≈ σ_NESS (czas z budżetu nigdy nie zamiera)
    n = len(fr["tau_bud"])
    tail = slice(int(0.9 * n), n)
    slope = np.polyfit(fr["t"][tail], fr["tau_bud"][tail], 1)[0]
    assert abs(slope - d["sig_tot_inf"]) / d["sig_tot_inf"] < 0.05


def test_dylatacja():
    """Faza zegarowa: σ_A/σ_B rzędu γ_A/γ_B = 27; NESS: produkcja → zimna."""
    d = E.liczby_ness(_z())
    assert 10 < d["ratio_clock"] < 40
    assert d["ratio_clock_peak"] > d["ratio_clock"]
    assert d["ratio_late"] < 0.1


def test_27_gamma():
    """γ_A = 27·γ_B (s ∝ T³, T_A = 3T_B) — kotwica 27×."""
    assert abs(E.GAMMA_A / E.GAMMA_B - 27.0) < 1e-12
