# AUDYT ZAMYKAJĄCY ENTROPIA-1.2 (R19)

**Wersja:** 1.0 · **Data:** 13 sierpnia 2026 · **Zakres:** równania, jednostki, niezależna replikacja, test T1/T2, odzyskiwalność dark-sektoru
**Odtwarzalność:** `python3 -m entropia.audyt12` (raport + figury) · `python3 -m pytest tests/test_audyt12.py -q` (19 testów) · pełny zestaw: `python3 -m pytest tests/ -q` (137 testów)

---

## 0. Cel, tło i metoda

Zgodnie z rekomendacją zamykamy iterację **ENTROPIA-1.2** przed dalszą kosmologią,
w sekwencji: audyt równań → audyt jednostek → niezależna replikacja symulacji →
test T1/T2 → test odzyskiwalności dark-sektoru.

> **Uczciwa uwaga o stanie projektu.** Rekomendacja mówiła „nie rozszerzać teraz
> modelu do R44". W bieżącym stanie projektu R44–R46 **już istnieją** (sieć
> cykliczna η(T), metryka FRW, dowód wzoru Petza) — powstały w poprzednich
> iteracjach. Audyt poniżej traktujemy zatem jako **formalny krok zamykający
> ENTROPIA-1.2, którego dotąd nie wykonano w tej formie**, a dalsza kosmologia
> (ENTROPIA-4.0) jest bramkowana jego wynikiem.

**Metoda.** Każde twierdzenie ENTROPIA-1.2 sprawdzamy *implementacją-świadkiem* —
osobnym kodem, który **nie importuje symulatorów projektu**:
- **Ś1**: superoperator Lindblada w konwencji **kolumnowej** `vec_col(AρB)=(Bᵀ⊗A)vec_col(ρ)`
  (projekt: wierszowa) + ewolucja ciągła przez `scipy.solve_ivp` (DOP853, krok
  adaptacyjny) zamiast `expm(L·τ)` krokowego;
- **Ś2**: **jawny RK4** na równaniach Blocha `ṙ_z = −2γr_z`, `ṙ_⊥ = −(γ+2γ_φ)r_⊥`
  wyprowadzonych ręcznie z master equation;
- **Ś3**: własne postacie zamknięte dla kąpieli Gibbsa (η = e^{−ω₀/T}) — `S(t)` i `dS/dt`;
- **Ś4**: **jawne równania stóp** drabiny Dickego (współczynniki CG:
  `u_m = γ(j−m)(j+m+1)`, `d_m = γ(j+m)(j−m+1)`) rozwiązywane `RK45`;
- **Ś5**: pełna przestrzeń `2^N` (N=2,4) z kolektywnymi `S±` budowanymi od zera.

Każda liczba w tym dokumencie pochodzi z uruchomienia; tolerancje podano jawnie.

---

## 1. AUDYT RÓWNAŃ

| Pozycja | Wynik | Kryterium |
|---|---|---|
| Lindblad: konwencja wierszowa (projekt) ≡ kolumnowa (Ś1) | `max|Δ| = 0` (bitowo identyczne po permutacji), `max|Δeig| = 0` | < 1e-12 |
| CPTP `e^{L·τ}`: ślad / hermitowskość / dodatniość (100 prób losowych) | `4.45e-16` / `1.11e-16` / `min eig = 0.00e+00` (≥0) | ślad < 1e-12 |
| RK4 Blocha (Ś2) ≡ `S(γ,t)` postać zamknięta | `max|ΔS| = 3.55e-14` | < 1e-10 |
| `dS/dt`: projekt vs Ś2 | `max|Δ| = 2.78e-14` | < 1e-10 |
| `dS/dt` vs gradient (t ≥ 0.05) | `3.21e-05` (dyskretny gradient) | < 1e-3 |
| Sektory vs pełna przestrzeń, N=2 (mieszanina \|10⟩-typ) | `ΔS = 4.40e-04`, `ΔI = 4.40e-04` | dyskretyzacja schematu (niżej) |
| Ten sam schemat, inna konwencja (Ś1 krokowy) vs projekt, N=2 | `ΔS = 4.22e-15`, `ΔI = 1.40e-14` | < 1e-12 |
| Sektory vs pełna przestrzeń, N=4 (stan \|0000⟩, diagonalny) | `ΔS = 8.35e-14` | < 1e-10 |
| `I_eq` numerycznie vs `ln(2/√3)` | `0.143841036` vs `0.143841036` (Δ = 1.89e-11) | < 1e-8 |
| `S(∞)` mieszaniny vs `½·ln 12` | `1.2424533` vs `1.2424533` (Δ = 1.89e-11) | < 1e-8 |

### 1.1. Weryfikacje analityczne (nie tylko numeryczne)

**(a) I_eq = ln(2/√3).** Stan równowagi mieszaniny ½ tryplet (j=1) + ½ singlet (j=0)
ma w bazie obliczeniowej wartości własne `(1/6, 1/6, 1/6, 1/2)`:
`S_AB = ½·ln 12`; oba kubity po redukcji: `ρ_A = ρ_B = diag(½,½)` ⇒ `S_A = S_B = ln 2`.
Zatem `I(A:B) = 2·ln 2 − ½·ln 12 = ½·ln(4/3) = ln(2/√3) = 0.143841…` — **zgodne z
numerem 9. cyfr**. To jest dokładne źródło nachylenia T2: `τ̇∞(T2) = η·I_eq/σ₀ = 7.192052…`.

**(b) Zamknięta forma odzyskiwalności dark-sektoru.** Równania stóp w sektorze j=1
(oba jumpy S₊, S₋ w tempie γ) dają macierz `A = 2γ·[[−1,1,0],[1,−2,1],[0,1,−1]]`
o wartościach własnych **{0, −2γ, −6γ}**. Dla pary ortogonalnych stanów |1,−1⟩, |1,0⟩:
`Δp(0) = (1,−1,0)` ⇒
`Δp(t) = ½e^{−2γt}(1,0,−1) + ½e^{−6γt}(1,−2,1)` ⇒
`M(t) = D(t)/D(0) = ½(e^{−2γt} + e^{−6γt})`.
W `t = 50τ = 12.5`, `γ = 0.02`: `M = ½(e^{−0.5}+e^{−1.5}) = 0.4148304…` — **numeryka
zgadza się do 5e-17** (wszystkie trzy ścieżki: zamknięta forma, Ś4, projekt).

### 1.2. Znaleziska audytu równań

1. **Kompresja 27× w raporcie projektu miała obcięty indeks.** Projekt porównywał
   `S_A(n)` z `S_B(27n)` przy **przycięciu** `27n → min(27n, 399)` i raportował
   błąd 4.28e-05. Ponieważ `S(γ,t) = f(γ·t)`, kompresja jest **dokładna** dla
   `γ_A = 27γ_B`; porządny test (B całkowane do `27·(N_TICKS−1)·τ`, Ś2) daje
   `max|Δ| = 4.14e-09`. 4.28e-05 to artefakt metryki, nie fizyki.
2. **`M_sektora` ma przesunięcie indeksu**: `Ms[k]` odpowiada `t = k·τ`
   (odczyt `[-1]` przy `n=50` daje `t = 12.25`, nie `12.5`). Nie wpływa na
   `tabela_odzyskiwalnosci` (używa indeksów jawnie), ale przy porównaniach
   trzeba używać `n=51`.
3. **Osobliwość logarytmiczna startu**: `dS/dt ∝ −ln(γt)` przy `t→0`
   (`dS/dt(1e-6) = 0.721` i rośnie) — start ze stanu czystego daje kusp entropii.
   To własność matematyczna, nie błąd; dyskretny zegar (`e^{Lτ}`) jej nie widzi
   (pierwszy krok skończony). Wartościowa uwaga dla każdej interpretacji
   „pierwszego tyknięcia".
4. **Dyskretny vs ciągły**: ten sam problem rozwiązywany krokowo vs `solve_ivp`
   różni się o `~4e-4` (N=2, mieszanina z koherencjami; precesja Ω) — to
   dyskretyzacja schematu `e^{Lτ}`, poniżej wszystkich twierdzeń modelu
   (konwencja sama w sobie: `4e-15`).

---

## 2. AUDYT JEDNOSTEK

### 2.1. Tabela wymiarów (jednostki kodu: nat, t)

| Wielkość | Wymiar | Uwagi |
|---|---|---|
| S (entropia von Neumanna) | nat (bezwymiarowa) | |
| γ (tempo relaksacji) | 1/t | γ_B = 0.02, γ_A = 27·γ_B |
| τ (mikro-tyknięcie) | t | 0.25 |
| δs (kwant entropii) | nat | 0.01 |
| σ₀ (tempo referencyjne) | nat/t | **σ₀ = δs = 0.01** (tożsamość sprawdzona) |
| σ = dS/dt | nat/t | produkcja entropii |
| η (waga korelacji) | 1 | bezwymiarowe |
| η·İ (T1) | nat/t | spójne z σ |
| **η·I (T2)** | **nat** | **poziom, nie tempo** ⇒ τ̇∞ = η·I_eq/σ₀ ≠ 0 |
| η·\|Ṙ\| (T3) | nat/t | spójne z σ |
| κ (czas = entropia) | t/nat | 1.0 |
| R_T = τ̇_A/τ̇_B | 1 | bezwymiarowy |
| γ ∝ T³ (kąpiel 3D) | 1/T³·… | J(ω) ∝ ω³ (Debye) |
| η_phys = e^{−ω₀/T} | 1 | czynnik Boltzmanna |

### 2.2. Kluczowe wnioski wymiarowe

1. **T0/T1/T3 są wymiarowo jednorodne** — sumują tempa (nat/t) i znikają, gdy
   tempo zanika. **T2 nie jest**: dodaje *poziom* `η·I` (nat) do *tempa* `σ`
   (nat/t). Jedyna droga do spójności to `η` o wymiarze 1/t — ale wówczas
   `η·I_eq` to nadal stała przy równowadze. **To wymiarowa diagnoza tego, czemu
   T2 nigdy nie staje**: nie chodzi o „pamięć I_eq jako źródło tempa", lecz o
   dodanie wielkości ekstensywnej per tyknięcie. W tym sensie ENTROPIA-1.1
   (faktycznie używająca T2) mierzyła **istnienie** korelacji, nie ich **tempo**.
2. **σ₀ = δs**: `τ̇(T0) = dS/σ₀ = dS/δs` = średnia liczba kwantów entropii na
   tyknięcie — kwantowy zegar T0 jest dokładnie „licznikiem bitów" z rdzenia
   modelu (κ = 1 ⇒ Δt = k·δs).
3. **27 jest bezwymiarowe**: `γ_A/γ_B = 27.0` (sprawdzone) i `27 = 3³` — czysty
   wynik `s ∝ T³` dla `T_A = 3T_B`.

### 2.3. Kalibracje fizyczne (stałe CODATA 2018)

| Wielkość | Audyt | Projekt | Uwagi |
|---|---|---|---|
| k_B·T_CMB/h | 56.790 GHz | — | T_CMB = 2.7255 K |
| próg n̄ < 0.01: ω/2π | **261.53 GHz** (ln(1/ε)); **262.09 GHz** (ln(1+1/ε), +0.22%) | 261 GHz (R30) | projekt używa przybliżenia ln(1/ε) — poprawne dla ε≪1, różnica 0.22% |
| ω_G = m_P·c²/ħ | 1.8549e43 rad/s | 1.85e43 (R30) | zgodne |
| γ_A/γ_B | 27.0 | 27 | dokładne |

Wnioski: wszystkie liczby kalibracyjne projektu odtwarzają się; różnica
`ln(1/ε)` vs `ln(1+1/ε)` jest kosmetyczna (0.22%) i dotyczy definicji progu.

---

## 3. NIEZALEŻNA REPLIKACJA SYMULACJI

### 3.1. Rdzeń (na którym stoi ENTROPIA-1.2)

| Pozycja | Projekt | Świadek (Ś2, RK4) | Różnica |
|---|---|---|---|
| S_B(t) (t = 0..99.75) | symulacja `expm` | RK4 (dt = τ/16) | max|Δ| = **3.55e-14** |
| Kompresja 27×, porządny test | — | RK4 dla B do t = 2693 | max|Δ| = **4.14e-09** |
| Kompresja 27×, metryka projektu (przycięta) | 4.28e-05 | — | artefakt (p. §1.2.1) |
| Tempo dS/dt przy dopasowanym S* (0.1…0.6) | 27.000 | 27.000 | **27.0000** (6/6 poziomów, <1e-6) |

### 3.2. ENTROPIA-1.2 (N=2, mieszanina |10⟩-typ; pełna przestrzeń vs sektory)

| Pozycja | Projekt (sektory, expm) | Świadek (Ś1+Ś5, solve_ivp/DOP853) |
|---|---|---|
| S(t) | — | ΔS = 4.40e-04 (dyskretny vs ciągły; konwencja: 4.22e-15) |
| I(t) | — | ΔI = 4.40e-04 (konwencja: 1.40e-14) |
| τ̇∞ T0 / T1 / T3 (300–400) | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 |
| τ̇∞ T2 (300–400) | 7.192052 | **7.192052** (Δ = 2e-7) |
| nachylenie T2 vs η·I_eq/σ₀ | 7.192052 | 7.192052 (Δ = 1e-6) |

### 3.3. Odzyskiwalność M(50τ) = D(50τ)/D(0)

| Sektor | Zamknięta forma | Świadek (Ś4, RK45) | Projekt (expm_multiply) |
|---|---|---|---|
| ciemny j=1 | 0.4148304 | 0.4148304 (Δ = 1.11e-15) | 0.4148304 (Δ = 5.55e-17) |
| jasny N=4 | — | 0.23235 | 0.23235 (Δ < 1e-4) |
| jasny N=10 | — | 0.11199 | 0.11199 (Δ < 1e-4) |
| jasny N=100 | — | 0.01314 | 0.01314 (Δ < 1e-4) |

### 3.4. Fizyczny 27× — R_T (T_A = 3·T_B, S* = 0.5)

| T_B/ω₀ | 3D: projekt | 3D: pełny spójny (Ś3) | single: projekt | single: pełny spójny (Ś3) |
|---|---|---|---|---|
| 3 | 29.420 | 30.433 | 3.242 | 3.354 |
| 5 | 28.322 | 28.841 | 3.138 | 3.195 |
| 10 | 27.618 | 27.850 | 3.066 | 3.092 |
| 30 | 27.197 | 27.269 | 3.022 | 3.030 |
| 100 | 27.058 | 27.079 | 3.006 | 3.009 |

**Znalezisko (§1.2.5):** `R_T_fizyczny` projektu miesza wzory — czasy przejścia
S* bierze z *termicznej* S(t), ale tempo dS/dt z *nieskończenie-gorącej* formuły
zamkniętej. Pełna spójna termiczna wersja (Ś3) daje wartości o **0.2–0.8% wyższe**
(T_B/ω₀ = 100…3). Wniosek jakościowy — **3D → 27, single-mode → 3, zbieżność
w gorącym limicie** — jest odporny; zalecamy poprawkę formuły (kandydat na R47).

---

## 4. TEST T1/T2

Dane: N=2, mieszanina |10⟩-typ, η = 0.5, σ₀ = δs = 0.01.

| Funkcjonał | τ̇∞ (300–400) | Staje? | Monotoniczny (τ̇ ≥ 0)? | Czkanie w ogonie (Δτ = 0) |
|---|---|---|---|---|
| T0 — σ/σ₀ | 0.000000 | **TAK** | TAK | 1.000 |
| T1 — (σ+η·\|İ\|)/σ₀ | 0.000000 | **TAK** | TAK | 1.000 |
| T2 — (σ+η·I)/σ₀ | **7.192052** | **NIE** | TAK | 0.000 |
| T3 — (σ+η·\|Ṙ\|)/σ₀ | 0.000000 | **TAK** | TAK | 0.000 → 1.000* |

\* T3: dla N=2 Ṙ = 0 z definicji (pamięć ciemna doskonała), więc τ̇(T3) = τ̇(T0).

- **Diagnoza potwierdzona dwiema niezależnymi ścieżkami**: T0/T1/T3 stają przy
  równowadze (fluorescencja wygasła: max dS w ogonie = 4.51e-10); T2 tyka dalej
  z nachyleniem `η·I_eq/σ₀ = 7.192052` (Δ = 1e-6). ENTROPIA-1.1 implementowała
  **T2** — audyt potwierdza, że było to „istnienie korelacji", nie „tempo".
- **Monotoniczność**: wszystkie cztery funkcjonały dają τ̇ ≥ 0 (czas nigdy nie
  biegnie wstecz) — warunek konieczny zegara; `dS ≥ 0` w całej historii
  (unitalność kąpieli, tw. Ando–Lindblada).
- **Czkanie**: T0/T1/T3 → 100% tyknięć zerowych w ogonie (czas stoi — czkanie
  w nieskończoność); T2 → 0% (zegar tyka bez przerwy, napędzany I_eq).
  Rozstrzygalność eksperymentalna: protokół R17/R37 (po wygaśnięciu
  fluorescencji: τ̇ = 0 vs τ̇ = 7.19/tyk).

---

## 5. TEST ODZYSKIWALNOŚCI DARK-SEKTOR

### 5.1. Ciemny sektor j=1 — zamknięta forma

`M(t) = D(t)/D(0) = ½(e^{−2γt} + e^{−6γt})` — wyprowadzenie w §1.1(b).
W t = 12.5: `M = 0.4148304`; zgodność trzech ścieżek: `|zamknięta−Ś4| = 1.11e-15`,
`|zamknięta−projekt| = 5.55e-17`.
**Niezależność od N**: sektor j=1 jest 3-wymiarowy dla każdego N —
`M(50τ) = 0.41483040993…` dla N = 4, 10, 100, 1000 z **rozrzutem 0.0**.

### 5.2. Porządek jasny/ciemny i zysk

| N | M_bright (50τ) | M_dark (50τ) | Zysk ciemny/jasny |
|---|---|---|---|
| 4 | 0.23235 | 0.41483 | 1.79× |
| 10 | 0.11199 | 0.41483 | 3.70× |
| 100 | 0.01314 | 0.41483 | **31.58×** |

- `M_dark > M_bright` dla każdego N; zysk rośnie z N (potwierdzone przez Ś4 i projekt).
- **j=0** (N parzyste): sektor 1-wymiarowy, Γ = 0 ⇒ `M(t) = 1` dokładnie.
- **Superradiancja jasnego**: drenaż stanu „jeden-wzbudzony" (sąsiad dna drabiny
  j=N/2) ma tempo `Γ = Nγ` (dla N=100: `100γ` — potwierdzone z macierzy stóp);
  mimo to `M_bright(50τ) = 0.0131` nie jest czystym `e^{−Nγt}`, bo odległość
  śladowa żyje w wolnych modach dyfuzji po całej drabinie — to ważna subtelność
  interpretacyjna: **szybki rozpad ≠ szybka utrata rozróżnialności**.
- Interpretacja (potwierdzona): „P_dark → 1" jest geometrycznym podziałem
  przestrzeni Hilberta; **M(t) mierzy faktyczną pamięć** — niska entropia nie
  oznacza braku pamięci (ciemny sektor przechowuje 0.415 rozróżnialności,
  gdy jasny traci ją do 0.013 przy N=100).

---

## 6. WERDYKT I BRAMKA KOSMOLOGICZNA

### 6.1. Wynik audytu

| Sekcja | Status | Kluczowe liczby |
|---|---|---|
| 1. Równania | ✅ zgodne | konwencje bitowo identyczne; CPTP ~1e-16; RK4 ≡ analityk 3.6e-14; I_eq = ln(2/√3) co do 1.9e-11 |
| 2. Jednostki | ✅ spójne | σ₀ = δs; T2 wymiarowo niejednorodny (poziom vs tempo) — to wyjaśnia jego wieczne tykanie; kalibracje fizyczne odtworzone (261.5 GHz, 1.855e43 rad/s) |
| 3. Replikacja | ✅ pełna | wszystkie liczby odtworzone przez świadków (S: 1e-14; kompresja 4e-9; T2: 7.192052; M: 1e-15; R_T: ≤1%) |
| 4. T1/T2 | ✅ potwierdzony | T0/T1/T3 stają, T2 = 7.192052/tyk; monotoniczność; czkanie: 100% vs 0% |
| 5. Dark-sektor | ✅ potwierdzony | zamknięta forma ½(e^{−2γt}+e^{−6γt}); zysk 31.58× (N=100); j=0: M=1; superradiancja Nγ |

### 6.2. Znaleziska do wdrożenia (zalecenia)

1. **[R47] ✅ WDROŻONE** `R_T_fizyczny`: użyta spójna termiczna pochodna
   dS/dt (`dSdt_termiczne_analitycznie` w extensions.py). Po poprawce
   projekt ≡ niezależny świadek do 1e-10 (wcześniej różnica 0.2–0.8%);
   nowe wartości: T_B/ω₀ = 10 → 27.850 (3D) / 3.092 (single),
   T_B/ω₀ = 100 → 27.079 / 3.009. Wniosek 27/3 potwierdzony.
2. Metryka kompresji 27×: porównywać z pełnym zakresem B (błąd 4.28e-05 → 4.14e-09).
3. `M_sektora`: udokumentować konwencję `Ms[k] ↔ t = kτ` (odczyt `[-1]` przy n
   wymaga n = liczba_tyknieć + 1).
4. Próg n̄ < 0.01: opcjonalnie ln(1+1/ε) zamiast ln(1/ε) (+0.22%).
5. Udokumentować osobliwość dS/dt ∝ −ln(γt) przy t→0 (kusp startu ze stanu
   czystego) — istotne dla interpretacji „pierwszego tyknięcia".

### 6.3. Bramka kosmologiczna

**OTWARTA i WYKORZYSTANA.** Wszystkie pięć sekcji audytu przeszło; poprawka
R47 wdrożona (powyżej); kosmologia zabawkowa **ENTROPIA-4.0** zbudowana
(R48: dwie komórki z wymianą — NESS z tożsamością Clausiusa do 1e-6;
R49: entropowa siła + emergentna FRW z odbiciem) — sekcja 12 manuskryptu,
moduł `entropia/e24.py`, testy test_e24.py (15).

### 6.4. Uczciwe uwagi (czego audyt NIE dowodzi)

- Wszystkie jednostki kodu są umowne (γ_B = 0.02 [1/t]); kalibracje fizyczne
  (CMB, ω_G) służą tylko porównaniom bezpośrednim, nie czynią z modelu teorii
  z pełnym mapowaniem na sekundy.
- Funkcjonały T0–T3 są **definicjami** (alternatywne teorie czasu), nie
  wyprowadzonymi prawami — audyt potwierdza, że są dobrze zdefiniowane,
  wymiarowo scharakteryzowane i **numerycznie rozróżnialne**, a ich wybór jest
  rozstrzygalny eksperymentalnie (protokół R17/R37).
- Dyskretny vs ciągły schemat różni się o ~4e-4 (precesja Ω, koherencje) —
  poniżej wszystkich twierdzeń modelu, ale przy dalszych precyzyjnych
  predykcjach warto używać mniejszego τ.
- Audyt nie obejmuje równań kosmologicznych (FRW) — te pozostają poza bramką.

---

**Pliki:** `entropia/audyt12.py` (audyt + figury figA1, figA2), `tests/test_audyt12.py`
(19 testów), `figury/figA1_audyt_odzyskiwalnosc.png`, `figury/figA2_audyt_27.png`.
**Odtwarzalność:** `python3 -m entropia.audyt12` (≈8 s) — pełna przebudowa raportu:
`python3 zrob_raport.py`.
