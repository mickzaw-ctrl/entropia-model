# ENTROPIA — czas jako entropia: model kosmologiczny i program falsyfikacyjny

**Manuskrypt zbiorczy · wersja 1.3 · 13 sierpnia 2026**
*Model «ENTROPIA», rozszerzenia R1–R50, ENTROPIA-1.1…5.0, protokół laboratoryjny,
audyt zamykający ENTROPIA-1.2 (Dodatek C) — wersja 1.3*

---

## Streszczenie

Przedstawiamy model kosmologiczny, w którym **czas jest entropią**: zegar
kosmiczny nie mierzy upływu zewnętrznego czasu, lecz kumuluje produkowaną
entropię. Mikrofizykę stanowi kubit („komórka Wszechświata”) w kąpieli
termicznej — równanie Lindblada; temperatura nadaje tempo (entropia właściwa
promieniowania ∝ T³, stąd „27×” dla T_A = 3·T_B), a unitalność kąpieli gwarantuje
monotoniczny wzrost S: 0 → ln 2 (tw. Ando–Lindblada; produkcja entropii σ ≥ 0
— tw. Spohna). Model rozszerzamy w dziewięciu iteracjach (ENTROPIA-1.1…1.9):
od dokładnej symulacji N = 2…100 w bazie Dickego, przez konkurencyjne definicje
czasu (funkcjonały T0–T3), miary odzyskiwalności (trace distance, coherent
information, fidelity, mapa Petza), koszt fizyczny zegara (energia,
ΔE·Δτ ≥ ħ/2, maksymalna temperatura T_max), aż po kompletny program
eksperymentalny (zimne atomy, nadprzewodniki, suchy bieg Monte Carlo) i
porównanie predykcji z danymi obserwacyjnymi (BBN/CMB, budżet entropii
Wszechświata, subradiancja, ograniczenia dyskretności czasu). Program
falsyfikacyjny rozstrzyga, które elementy wynikają z modelu, a które są
konsekwencją definicji; najsilniejsza testowalna predykcja: **zegar entropowy
staje po wygaśnięciu fluorescencji (T1) albo tyka dalej, napędzany korelacją
subradiacyjną (T2)** — rozstrzygalne w istniejącej technologii.

---

## 1. Wprowadzenie

### 1.1. Czas relacyjny

W mechanice kwantowej czas nie jest obserwablą (Pauli); relacyjne podejście
Page'a–Woottersa traktuje czas jako korelację między układem a zegarem.
Nowe prace rozwijają „entropię jako zegar" (czas emergentny parametryzowany
produkcją entropii). Niniejszy model idzie dalej i czyni z tego **definicję**:
T(n) = S(n), z kwantyzacją entropii w „bitach" δs i stochastycznym
poissonowskim zegarem. Od razu zaznaczamy granicę: to nie jest dosłowna
tożsamość fizyczna (czas i entropia mają różne wymiary i strukturę), lecz
**funkcjonał czasu relacyjnego**, który można testować.

### 1.2. Czego dotyczy praca

1. Sformułowanie rdzenia modelu (Lindblad, T = S, 27×, czkanie).
2. Rozszerzenia: termodynamika, struktura sektorowa, kosmologia, zegar kwantowy.
3. Program falsyfikacyjny: konkurencyjne definicje czasu, odzyskiwalność, koszt.
4. Protokół laboratoryjny i porównanie z obserwacjami.
5. Uczciwa dyskusja ograniczeń.

---

## 2. Rdzeń modelu

### 2.1. Równanie Lindblada

Komórka = kubit w kąpieli termicznej (nieskończenie gorąca):

```
dρ/dt = −i[H,ρ] + γ·D[σ₋] + γ·D[σ₊] + γ_φ·D[σ_z],  D[L]ρ = LρL† − ½{L†L,ρ}
ρ_eq = ½·𝟙  ⇒  S(∞) = ln 2
```

Generator jest **unitalny** ⇒ S monotonicznie rośnie (tw. Ando–Lindblada);
σ = −d/dt·S(ρ‖ρ_eq) = dS/dt ≥ 0 (tw. Spohna).

### 2.2. Temperatura jako tempo: „27×”

Entropia właściwa promieniowania s ∝ T³; dla T_A = 3·T_B wszystkie tempa
dyssypacji są w stosunku γ_A/γ_B = 27. **Uczciwie**: z prawa s ∝ T³ samo nie
wynika τ_A/τ_B = 27 — potrzebny jest postulat łączący tempo zegara z entropią.
Model rozróżnia gałęzie dτ ∝ s (⇒ 27 dokładnie) i dτ ∝ Ṡ (⇒ inne liczby);
uogólnienie γ ∝ T^p ⇒ stosunek 3^p jest testem falsyfikacyjnym (R16).

### 2.3. Zegar i „czkanie”

ΔS_n = S(ρ_n) − S(ρ_{n−1}); Δτ_n = κ·ΔS_n (kwantyzacja: k_n ~ Poisson(ΔS_n/δs)).
Przy ΔS_n → 0 zegar staje (Δt = 0) — **czkanie**. Rozróżnienie kluczowe:
**zatrzymanie zegara entropowego ≠ zatrzymanie czasu fizycznego** (R17:
singlet precesuje unitarnie przy S = 0).

### 2.4. Dekoherencja

Tr(ρ²): 1 → 0.5, |r|: 1 → 0 — pełna dekoherencja do ½·𝟙.

---

## 3. Wiele ciał i struktura sektorowa (R2, R4, R5, R7, R11, R18)

Kolektywne jumpy S± komutują z S² ⇒ przestrzeń N kubitów rozkłada się na
sektory spinu j; każdy termalizuje wewnątrz siebie (wymiar ≤ N+1 ⇒ **N = 100
osiągalne dokładnie** w bazie Dickego, ENTROPIA-1.1):

| obiekt | wynik |
|---|---|
| S(∞), sektor symetryczny | ln(N+1) (N=100: 4.6151, błąd 7×10⁻⁵) |
| N=2, |10⟩ | ciemny singlet: S(∞) = ½·ln 12, I(A:B) = ln(2/√3) |
| N=3, sektory | j=3/2 ⊕ 2×j=1/2; subradiancja (populacja kopii B ≡ 1) |
| stany Haar, γ_φ=0 | koherencje A↔B = √(p_A·p_B) blokują entropię |
| odblokowanie γ_φ | τ90 ∝ 1/γ_φ (nachylenie −0.96) — pełna termalizacja |
| kompresja 27× | S_A(n) = S_B(27n), błąd ≤ 3×10⁻¹² (N = 2..100) |

---

## 4. Kosmologia (R3, R6, R8)

- **R3**: tempo samo-zależy od odczytu zegara (γ_eff(T)); kompresja 27× nietknięta.
- **R6**: gorący start (S(0) ≈ ln 2) + zimna kąpiel ⇒ entropia maleje ⇒
  **czas wstecz** (budżet 0.31 nat). Uczciwie: realny Wszechświat ochładza się
  adiabatycznie bez zewnętrznej kąpieli — to hipoteza modelu, nie kosmologia.
- **R8**: cykl BB → ekspansja → ochłodzenie → kolaps; dwie wskazówki czasu
  (T = S − S₀ wraca do zera — pętla; τ = Σ|ΔS| — upływ). ⚠️ literalna pętla
  przeczy przyspieszającej ekspansji ΛCDM.

---

## 5. Kwantowy zegar (R9, R10, R15)

- **R9**: czas jako operator — oscylator-zegar, ⟨n⟩ śledzi S, Δn ≈ √⟨n⟩,
  back-action (kompromis precyzja↔koszt↔entropia; duch Saleckera–Wignera).
- **R10**: start koherentny — stymulowane kopiowanie wzmacnia back-action 4.1×;
  dekoherencja zegara κ·D[b†b] → czas klasyczny (oś liczbowa), fizyka bez zmian.
- **R15**: jawny H_int (Jaynes–Cummings) rozmywa czas; κ nieustannie rzutuje
  na oś liczbową — **dekoherencja zegara jako strażnik historii** (einselection,
  Zurek); rozmycie czasu spada 56× przy κ = 0.5.

---

## 6. Program falsyfikacyjny (R16, R19–R26)

### 6.1. Konkurencyjne funkcjonały czasu (R19)

| funkcjonał | definicja | τ̇∞ (równowaga) |
|---|---|---|
| T0 | σ/σ₀ | 0 — STAJE |
| T1 | (σ + η·|İ|)/σ₀ | 0 — STAJE (İ → 0) |
| T2 | (σ + η·I)/σ₀ | **7.19** — NIE STAJE (I_eq > 0) |
| T3 | (σ + η·|Ṙ|)/σ₀ | 0 — STAJE |

Diagnoza: ENTROPIA-1.1 implementowała T2 (absolutna I), nie T1 (|İ|). Wybór
teorii czasu (produkcja vs istnienie informacji) jest **rozstrzygalny
eksperymentalnie**.

### 6.2. Odzyskiwalność i natura pamięci (R20–R24, R29, R34)

| miara | wynik |
|---|---|
| M(t) = D(t)/D(0) (trace) | ciemny j=1: 31× dłużej niż jasny (N=100); j=0: 1 |
| I_c (coherent information) | < 0 w równowadze — pamięć KLASYCZNA |
| M_F = 1 − F | j=1: 382× dłużej (N=100); F_e(j=0) = 1 |
| C_mem (Helstrom) | j=1/2: 0.44 bitu vs jasny N=100: 0.0004 |
| **mapa Petza** F_rec | 0.886 (j=1/2) → 0.523 (j=2); j=0: 1 |
| **asymptotyka Petza (R41)** | F_rec → 1/(N+1) (t→∞); C(t) = F_rec − 1/(N+1) ≈ 0.215 ± 0.017 niezależne od N (N=4..16); Γ₁ = Nγ niszczy pamięć |
| realizacja Petza (R34) | echo odzyskuje fazę (j≤1); DFS: F=1 — chroń, nie odzyskuj |

### 6.3. Koszt fizyczny (R20, R22, R25, R31, R32)

- ΔE·Δτ ≥ ħ/2 ⇒ ω_c ≥ 1.7; Purcell: górna granica ω_c < (ε_b/(cg²))^{1/3}
  ⇒ **T_max** (3D: 0.66; Ohmic z cutoffem: 215804 — zimny zegar).
- Arkusz nadprzewodzący (R31): T_max(6 GHz) = 63 mK — testowalny.
- Energia protokołu: zegar i decyzja zaniedbywalne (10⁻²³–10⁻²⁵ J); pułapka 5 mJ.

---

## 7. Protokół laboratoryjny (R23, R26–R28, R33)

**Test T1 vs T2** w zimnych atomach (nanofiber ¹³³Cs, N = 5×10³, β = 0.15,
t_B = 8 ns, t_D = 179 μs): po ostatnim fotografie (faza ciemna) mierzymy
I(A:B) i tempo zegara. **Uczciwa granica**: kanał fotonowy nie rozróżnia T1/T2
(zegar bez sprzężenia zwrotnego) — rozstrzyga kanał korelacyjny (σ_I = 0.01 nat,
M = 150 realizacji/punkt). Suchy bieg (R28, R33): SPRT E[N] = 1, błędy 0,
odporność na η_det = 0.1, szum tła i **niedoskonałą wierność F ≥ 0.3**
(samo-kalibracja przez pomiar I(A:B)).

**Protokół różnicowy wielu zegarów (R37)**: M identycznych zegarów w
komórkach jasna→ciemna (A, I_eq) i czysto ciemnych (B, I = 0); dryft
Δτ̄ = τ_A − τ_B po nasyceniu: T1 — const, T2 — liniowy (nachylenie 7.19/tyk).
Zalety: odrzucenie szumu wspólnego, brak kalibracji absolutnej, σ ↓ 1/√M_B.
To wprost predykcja recenzji: identyczne zegary w różnych środowiskach
dyssypacyjnych wykazują różnicę dynamiki entropicznej.

---

## 8. Skala kosmiczna (R35, R36, R38)

- **Zegar w kąpieli CMB**: próg ω_c/2π ≥ 261 GHz (ε = 0.01); 6/100 GHz nigdy
  nie użyteczne w CMB; **horyzont zegarów** — częstotliwość progu spada z
  wiekiem Wszechświata (T(t) z ΛCDM): 300 GHz od z ≈ 0.1, 1 THz od z ≈ 2.8,
  3 THz od z ≈ 10.5, 10 THz od z ≈ 37. Cutoff grawitacyjny (ω_Planck) bez
  wpływu dla realistycznych ω_c.
- **Sieć zegarów (R36)**: synchronizacja przez wymianę entropii (σ_end:
  48 → 0.08); jednakowe T ⇒ σ ≡ 0 bez sprzężenia; τ_net — emergentny czas
  kosmiczny.
- **Sieć z dynamiką η(T) (R40, R8 × R36)**: cykliczna kąpiel — upływ
  τ_abs = Σ|ΔS| rośnie monotonicznie przez cykle (3×budżet = 1.24 nat), choć
  entropia wraca do ln 2 (pętla); jednakowe komórki: σ ≡ 0; offsety fazowe:
  synchronizacja modulowana cyklem. Czas sieci przetrwa cykl kosmiczny.

---

## 9. Predykcje a dane obserwacyjne (R14)

| predykcja | dane | ocena |
|---|---|---|
| s ∝ T³ ⇒ T_ν/T_γ = (4/11)^{1/3} | N_eff = 2.99 ± 0.17 (Planck 2018) | ✅ ilościowo |
| strzałka czasu = wzrost S | druga zasada | ✅ |
| stany ciemne nie termalizują | subradiancja w zimnych atomach (PRL 116, 083601; PRL 128, 203601) | ✅ lab |
| dekoherencja 1 → 0.5 | mechanizm kwantowy | ✅ |
| zegar entropii (S_obs/S_CEH ≈ 10⁻¹⁸) | Egan & Lineweaver 2010 | 🟡 jakościowo |
| kosmiczne tempo produkcji entropii maleje | SFRD szczyt z≈2, spadek ~10× | 🟡 |
| SMBH = niekończący się zegar | budżet entropii (3.1×10¹⁰⁴ k_B) | 🟡 |
| czas wstecz przy chłodzeniu | brak odpowiednika (ekspansja adiabatyczna) | ⚠️ |
| cykl BB→Kolaps | przecząca ekspansja przyspieszająca (ΛCDM) | ⚠️ |
| dyskretność czasu | GRB: E_QG,1 > 7.6·E_Pl (zgodne, τ wolny) | ❓ |

---

## 10. Ograniczenia i uczciwe uwagi

1. **To nie jest teoria fizyczna, lecz laboratorium pojęć** z jednym prawdziwym
   prawem (s ∝ T³) i spójnym formalizmem (Lindblad + funkcjonał czasu).
2. **T ≡ S** dosłownie — odrzucone; obowiązuje funkcjonał τ = F(S, Ṡ, I, Γ, ρ).
3. **27×** wymaga postulatu łączącego tempo z entropią; podane jako predykcja
   warunkowa z testem 3^p.
4. **Czkanie** zachowują T0, T1, T3; T2 (absolutna I) je niszczy — wybór
   teorii czasu rozstrzygalny eksperymentalnie.
5. **Big Crunch ⇒ Ṡ < 0** — nieuzasadnione; w modelu Ṡ < 0 jest napędzane
   kąpielą, nie kontrakcją.
6. **P_dark → 1** to geometria; pamięć dowodzą miary operacyjne (C_mem, Petz).
7. **Realizacja**: subradiancja zmierzona; protokół T1/T2 wykonalny; zegar w
   CMB musi być THz; granice kosmologiczne (T_max) dyskutowane.

---

## 11. Wnioski

Model «ENTROPIA» przechodzi od deklaracji filozoficznej do **konkretnego modelu
matematycznego** z: równaniem ewolucji (Lindblad), strukturą sektorową (Dicke,
N = 2…100), funkcjonałem czasu (konkurencja T0–T3), miarami odzyskiwalności
(trace, I_c, fidelity, Petz), kosztem fizycznym (energia, T_max, ω_c(T)) i
**zestawem przewidywań falsyfikowalnych**: τ̇ ∝ Ṡ + η|İ|, stosunek 3^p, zegar
jasny↔ciemny (τ̇ = 0 vs 7.19), horyzont zegarów w CMB, synchronizacja sieci.
Najsilniejszy wynik: mechanizm modelu (s ∝ T³) jest mechanizmem realnej
kosmologii (BBN/CMB — zgodność ilościowa), a protokół rozstrzygający T1 vs T2
jest wykonalny na istniejącej technologii zimnych atomów i nadprzewodników.

---

## 12. ENTROPIA-4.0 — kosmologia zabawkowa: dwukomórkowy wszechświat (R48, R49)

Bramka audytu (§D.6.3) otwarta; budujemy kosmologię od mikroskopii — bez
nakładania ΛCDM z zewnątrz (inaczej niż R45). Wszechświat-zabawka: **dwie
komórki-kubity** A (gorąca kąpiel Gibbsa, T_A = 3·T_B, γ_A = 27·γ_B — entropia
właściwa promieniowania s ∝ T³) i B (zimna, γ_B = 0.02; ω₀ = 1), połączone
**kanałem wymiany** X₁ = σ₋^Aσ₊^B, X₂ = σ₊^Aσ₋^B w tempie κ = 0.3 (przenosi
ekscytację A↔B i **zachowuje E_A + E_B** — sprawdzone do 1e-12). Start: A
czysto wzbudzona (inwersja obsadzeń), B w stanie podstawowym.

**R48 — mikroskopijny NESS.** Układ dąży do stacjonarnego stanu
nierównowagowego: `S_tot(∞) = 1.3491 nat` (dS/dt → 7.6e-13), ale produkcja
entropii (Spohn, per kanał) pozostaje `σ_NESS = 0.00338 nat/t` — **czas
z budżetu nigdy nie zamiera**. Prąd energii płynie z gorącej do zimnej:
`J_E,∞ = 0.00507`. Najsilniejszy wynik: **tożsamość Clausiusa/Onsagera**
`σ_NESS = J_E,∞·(1/T_B − 1/T_A)` spełniona do 1e-6 (numerycznie 1.000000).
Prawo Fouriera: `J_E,∞` rośnie z ΔT (0.00196 → 0.00635 dla ΔT 0.5 → 4),
quasi-liniowo przy małych ΔT (nasycenie przy dużych). Produkcja w NESS
dominuje w **zimnej** komórce (σ_B ≫ σ_A — tam, gdzie 1/T jest duże; Clausius).

**R49 — grawitacja = budżet entropii.** (i) *Entropowa siła* (zabawka
Verlinde'owska): `S_tot∞(κ)` rośnie ściśle z tempem wymiany (1.2617 przy
κ→0 — niezależne Gibbsy — do 1.3546 przy κ→∞; zbieżny NESS, t_max ∝ 1/κ);
więc `F(d) = T·∂S∞/∂d < 0` — **przyciąganie** (komórki „chcą" bliżej, bo
bliżej = więcej dostępnej entropii). Uczciwie: profil NIE jest 1/d² — siła
najsilniejsza przy pośrednich d (S∞ nasyca się przy silnym sprzężeniu, znika
przy d→∞ gdy κ→0). (ii) *Emergentna FRW*: start w **inwersji obsadzeń**
(T_eff < 0 — laserowe wzbudzenie); **przejście przez T = ∞** (pg = pe) przy
t = 1.10 = „Wielki Wybuch" — osobliwość `a = 0`; potem T_eff: +∞ → 2.835
(< T_A = 3 — wymiana chłodzi A poniżej równowagi kąpieli). Skala
`a(τ) = T_NESS/T_eff` (konwencja promieniowania T ∝ 1/a): **0 → 1.2374
(ekspansja, t = 3.40) → 1.0000 (kontrakcja do śmierci cieplnej)** — T_eff
przestrzeliwuje poniżej T_NESS, wymiana drenuje A szybciej niż kąpiel
uzupełnia ⇒ **odbicie** (cykliczny wszechświat, por. R8). `H = (1/a)(da/dτ)`
przechodzi przez zero przy t = 3.45, `z = 1/a − 1: ∞ → 0` (przy kontrakcji
z < 0 — przesunięcie ku fioletowi). Dwa zegary: `τ_sys = S_tot` kończy się
przy śmierci cieplnej (1.3491 nat — **skończony wiek wszechświata**), `τ_bud =
Σσ·τ` rośnie liniowo w NESS (czas trwa). (iii) *Dylatacja czasu*: w fazie
zegarowej `σ_A/σ_B ≈ 22` (średnio; szczyt 33) — rząd γ_A/γ_B = 27 — zegar
w gorącej komórce tyka ~27× szybciej; w NESS produkcja przenosi się do zimnej.

**Uczciwe uwagi (czego ENTROPIA-4.0 NIE robi).** To zabawka o 2 kubitach;
mapowanie a(τ) z T_eff jest przyjęte (konwencja promieniowania), nie
wyprowadzone z metryki; κ↔odległość jest parametrem, nie geometrią; σ > 0
w NESS to produkcja względem lokalnych równowag kąpieli (Spohn), nie nowa
fizyka; „Wielki Wybuch" to cecha mapowania T_eff (inwersja = ujemna
temperatura — realne zjawisko fizyczne, ale w tej zabawce czysto formalne).
Weryfikacja: κ=0 odtwarza niezależne jednokubitowe kąpiele do 1e-14; kanał
wymiany zachowuje energię dokładnie.

**Liczby kluczowe (z uruchomień, `python3 -m entropia.e24`):**
`S_tot∞ = 1.3491`, `J_E,∞ = 0.00507`, `σ_NESS = 0.00338`,
`σ_NESS = J·(1/T_B−1/T_A) ± 1e-6`, `S∞(κ): 1.2617→1.3546`,
`a: 0 → 1.2374 → 1`, `H: +∞ → 0 (t=3.45) → 0`, `τ_sys = 1.3491`,
`σ_A/σ_B ≈ 22 (faza zegarowa) → 0.034 (NESS)`.
Moduł `entropia/e24.py`, figury figE37–E40, testy test_e24.py (15).

---

## 13. ENTROPIA-5.0 — pętla pomiarowa na procesorze kwantowym (R50)

Konkretyzacja testu dark-sektoru (R17/R23/R27) na sprzęcie **IBM Quantum /
Google**: protokół, który może wykonać pojedynczy użytkownik procesora
kwantowego. Szkic pętli (od użytkownika) — **z poprawkami**:

```
for t in 0..t_max (krok dt):
    h(q0); cx(q0,q1); z(q1); x(q1)        # |D⟩ = (|01⟩−|10⟩)/√2
    n = t/dt × ( rz(Δω·dt) na q0          # zaburzenie (sprzęga |D⟩↔|T0⟩)
                 + kolektywny rozpad S−   # przez ancilla + reset )
    cx(q0,q1); h(q0); pomiar              # baza Bella: P(|11⟩) = P_D
```

**Poprawka krytyczna (błąd w szkicu).** `h, cx, x` przygotowuje **Ψ+ (jasny
tryplet)**, nie singlet: |⟨ψ|T0⟩|² = 1.000, |⟨ψ|D⟩|² ≈ 5e-34. Wymagana bramka
`z` na q1: `h, cx, z, x` → Ψ− (|⟨ψ|D⟩|² = 1.000). Bez niej cały eksperyment
testowałby stan jasny (rozpad 2γ), nie ochronę ciemnego.

**Fizyka rozstrzygająca (liczby z uruchomień, dt = 0.25, γ = 0.02, t = 25):**

| Start | Kąpiel kolektywna | Kąpiel niezależna (kontrola) |
|---|---|---|
| |D⟩ (singlet) | **P_D = 1.000000** (ciemny — M = 1) | P_D = 0.6058 ≈ e^{−γt} = 0.6065 |
| |T0⟩ (tryplet) | P = 0.3660 ≈ e^{−2γt} = 0.3679 (superradiancja) | ≈ e^{−γt} |
| |D⟩ + rz(Δω = 0.05) | P_D = 0.7036 (odblokowanie) | — |

- **S−|D⟩ = 0, S−|T0⟩ = √2|00⟩** (superradiancja 2γ) — zweryfikowane.
- **Kryterium falsyfikacji**: P_D płasko przy 1 ⇔ kąpiel KOLEKTYWNA (model);
  P_D ≈ e^{−γt} ⇔ kąpiel NIEZALEŻNA. Rozróżnienie jest odporne na
  niepewności kalibracji (pomiar różnicowy na tej samej głębokości).
- **rz na jednym kubicie łamie symetrię ciemną** (|D⟩ ↔ |T0⟩, które rozpada
  się 2γ) — sprzętowy analog odblokowania R11.

**Realizacja „kolektywnego rozpadu" na sprzęcie.** To nie jest natywna bramka:
kanał S− implementujemy przez ancillę (reset co krok). W module walidujemy
**unitarną osadkę V** (8×8): Tr_anc[V(ρ⊗|0⟩⟨0|)V†] ≡ kanał Krausa —
**dokładnie (Δ = 0.00e+00)**. Dekompozycja na bramki: rotacja Dickego W
(|10⟩↔|01⟩, √iSWAP-typ ≈ 2 CX) + dwie rotacje Givensa sterowane ancillą
(|T0⟩↔|00⟩, |T0⟩↔|11⟩, ≈ 2×3 CX) + W† + reset ≈ **8–14 CX/krok**.

**Tomografia.** Baza Bella (cx, h, pomiar): Ψ− ↔ |11⟩ — bezpośredni witness
singletu. Pomiar losowy (randomized measurements, bazy X/Y/Z) + rekonstrukcja
LS: F(ρ, ρ_est) = **0.9669** dla |D⟩ przy 16k strzałów. Szum strzałowy:
σ = √(p(1−p)/N) — σ = 1% przy ≈ 2500 strzałów (minuty na IBM).

**Budżet sprzętowy (szacunki 2024–2026, przed startem zweryfikować).**
IBM Heron r2: krok ≈ 4–10 μs, **3–9 kroków** w budżecie T₂ (T₂ ≈ 150 μs,
T₂_eff = T₂/4), błąd odczytu ≈ 0.5%; Google Willow: krok ≈ 3–6 μs,
**2–5 kroków**, błąd odczytu ≈ 1%. Uczciwie: NISQ pozwala na ~5–20 kroków —
dlatego **γ i dt są wolnymi parametrami** (skala niezmiennicza, liczy się
γ·t): dobieramy γ·dt tak, by γ·t_max ≈ 1–2 zmieściło się w budżecie (np.
10 kroków × γdt = 0.1 → γt = 1: T0 → e^{−2} = 0.135, singlet zostaje 1).
Pomiar różnicowy (kolektywna vs niezależna, ta sama głębokość) nie wymaga
kalibracji absolutnej.

**Liczby kluczowe:** P_D(kol) = 1.000000, P_D(niez) = 0.6058 ≈ e^{−γt},
P_T0(kol) = 0.3660 ≈ e^{−2γt}, P_D(rz, 0.05) = 0.7036, obwód ≡ Kraus
Δ = 0.00e+00, F(rekonstr.) = 0.9669, Heron 3–9 / Willow 2–5 kroków.
Moduł `entropia/e25.py`, figury figE41–E43, testy test_e25.py (14).

---

## Dodatek A — Synteza rozszerzeń R1–R50

| # | rozszerzenie | kluczowy wynik |
|---|---|---|
| R1 | Skończona T | S(∞) = H(1/(1+η)); overshoot η<1/3; kompresja 27× trwa |
| R2 | N=2 | ciemny singlet; ln 3 vs 2·ln 2; I = ln(4/3) |
| R3 | Zegar → tempo | γ_eff(T); kompresja nietknięta |
| R4 | N=3 sektory | j=3/2 ⊕ 2×j=1/2; subradiancja |
| R5 | Entropia makro | N·ln 2 vs ln(N+1) |
| R6 | Gorący WB | czas wstecz (budżet 0.31) |
| R7 | Losowe stany | koherencje A↔B = √(p_Ap_B) blokują |
| R8 | Cykl | czas dwustronny; τ = 2·budżet; pętla |
| R9 | Kwantowy zegar | ⟨n⟩, Δn, back-action |
| R10 | Koherencje zegara | stymulacja 4.1×; κ → czas klasyczny |
| R11 | Odblokowanie γ_φ | τ90 ∝ 1/γ_φ (−0.96) |
| R13 | Grawitacja (NESS) | σ stałe; T_graw liniowy |
| R14 | Predykcje vs obserwacje | s∝T³ → N_eff zgodne; subradiancja zmierzona |
| R15 | Strażnik historii | κ rzutuje na oś liczbową (56×) |
| R16 | Formalizm relacyjny | λ→S→τ; 27 jako predykcja warunkowa |
| R17 | Test bright↔dark | singlet: zegar milczy, pamięć trwa |
| R18 | ENTROPIA-1.1 | N=2..100; 27× dokładne; P_dark(Haar)→1 |
| R19 | Funkcjonały T0–T3 | T0,T1,T3 stają; T2 nie (absolutna I) |
| R20 | I_c, koszt, protokół | I_c<0 (pamięć klasyczna); ω_c^min=1.7; ΔE·Δτ≥ħ/2 |
| R21 | Kanał odzysku (fidelity) | M_F: 382× (N=100); F_e(j=0)=1 |
| R22 | ω_c(T) | ω_c ∝ T (rozdzielczość ~T), produkcja ∝T³ |
| R23 | Protokół e2e | moc 1.000; τ̇ po ostatnim fotonie rozstrzyga |
| R24 | Pamięć operacyjna | C_mem: 0.44 (j=1/2) vs 0.0004 (jasny N=100) |
| R25 | Samo-spójny ω_c(T) | T_max ∝ g^{−2/3}; okno istnienia zegara |
| R26 | SPRT | E[N]=1; adaptacja (λ₂=0.1 ⇒ 20) |
| R27 | Karta eksperymentalna | 3 platformy; kanał korelacyjny; wykonalne dziś |
| R28 | Suchy bieg | SPRT E[N]=1 z detektorem (η=0.3, dark, jitter) |
| R29 | Mapa Petza | F_rec 0.89 → 0.54; j=0: 1 |
| R30 | Zimny zegar | Ohmic T_max 215k vs 3D 0.66; kosmologia ω_c |
| R31 | Arkusz T_max (nadprzewodniki) | T_max(6 GHz)=63 mK; n̄ mierzalne; T1_Purcell 64 μs |
| R32 | Koszt energii | zegar/decyzja zaniedbywalne; pułapka 5 mJ |
| R33 | Suchy bieg + F | moc 1.000 do F=0.3 (samo-kalibracja) |
| R34 | Realizacja Petza | echo (j≤1); DFS: F=1 — chroń, nie odzyskuj |
| R35 | Zegar w CMB | próg 261 GHz; cutoff grawitacyjny bez wpływu |
| R36 | Sieć zegarów | σ_end 48→0.08; jednakowe T ⇒ σ≡0 |
| R37 | Protokół różnicowy | Δτ: T1 const vs T2 liniowy (7.19/tyk); common mode zniesiony |
| R38 | Zegar w ewoluującym CMB | horyzont zegarów: 300 GHz od z≈0.1 … 10 THz od z≈37 |
| R39 | Pakiet publikacyjny | MANUSKRYPT.md: spójny opis modelu (R1–R38) |
| R40 | Sieć z dynamiką η(T) | upływ przetrwa cykl; T_signed wraca; σ≡0 (jednakowe); offsety: σ modulowane |
| R41 | Asymptotyka Petza (Dicke) | F_rec → 1/(N+1); C(t) ≈ 0.215±0.017 niezależne od N; Γ₁=Nγ; ciemny: F=1 |
| R42 | Formalny limit Petza | gap/γ = 1.0000 (N=2..100); F_rec(t) = ½a(2+(1−a)²/(1−½a²)) Δ=1e-16; N→∞: jasny→0, ciemny→1 |
| R43 | Entrainment faz sieci | σ_φ: 9.56 → 0.000 (g_sync=0.2) — fazy cykli lockują się |
| R44 | Dowód uniwersalności C(t) | drabina Dickego Γ_n=n(N−n+1)γ, gap=γ; okno (1/(Nγ),1/γ); F_rec dokładny Δ=9e-16 |
| R45 | ENTROPIA-3.0 — metryka FRW | s·a³=const; S_eq maleje (dτ=|dS|); S_BH horyzontu 10¹⁴⁰→10¹³⁰ k_B |
| R46 | Dowód wzoru Petza z regularyzacją | Tw.1: F_rec dokładny (Δ=3e-16); Tw.2: Γ=Nγ; Tw.3: pełny=średnia rzutowana; Tw.4: C(t) uniwersalne (gap=γ) |
| R47 | Poprawka R_T (audyt 1.2) | R_T_fizyczny: pełna spójna termiczna (dSdt_termiczne) — projekt ≡ świadek do 1e-10; TB=10: 27.850 (3D), 3.092 (single) |
| R48 | ENTROPIA-4.0 — dwie komórki (NESS) | S_tot∞=1.3491; J_E,∞=0.00507; σ_NESS=0.00338 = J·(1/T_B−1/T_A) (Clausius, Δ=1e-6); Fouriera: J↑ΔT (nasycenie); produkcja → zimna komórka |
| R49 | ENTROPIA-4.0 — siła + FRW | S∞(κ): 1.2617→1.3546 ⇒ F(d)<0 (przyciąganie, nie 1/d²); inwersja→T=∞ (Wielki Wybuch): a: 0→1.2374→1 (odbicie); H: +∞→0; τ_sys=1.3491 (skończony), τ_bud liniowy; σ_A/σ_B≈22→0.034 (dylatacja) |
| R50 | ENTROPIA-5.0 — pętla pomiarowa (IBM/Sycamore) | POPRAWKA: h,cx,x daje Ψ+ (jasny!), trzeba h,cx,z,x → Ψ−; P_D: kolektywna 1.000000 vs niezależna e^{−γt} (falsyfikacja); |T0⟩: e^{−2γt} (superradiancja); rz odblokowuje; obwód z ancillą ≡ Kraus (Δ=0); F(rekonstr.)=0.9669; Heron 3–9, Willow 2–5 kroków |

## Dodatek B — Bibliografia (wybór)

1. Page D.N., Wootters W.K., *Phys. Rev. D* 27, 2885 (1983) — czas relacyjny.
2. Lindblad G., *Commun. Math. Phys.* 48, 119 (1976); Gorini K. et al., *J. Math. Phys.* 17, 821 (1976).
3. Ando T., *Linear Algebra Appl.* 118, 163 (1989); Lindblad G., *Commun. Math. Phys.* 39, 111 (1974) — entropia map unitalnych.
4. Spohn H., *J. Math. Phys.* 19, 1227 (1978) — produkcja entropii.
5. Dicke R.H., *Phys. Rev.* 93, 99 (1954) — superradiancja; subradiancja: Guerin W. et al., *PRL* 116, 083601 (2016); Pennetta R. et al., *PRL* 128, 203601 (2022).
6. Zurek W.H., *Phys. Rev. D* 24, 1516 (1981) — einselection/dekoherencja.
7. Salecker H., Wigner E.P., *Phys. Rev.* 109, 571 (1958) — ograniczenia pomiaru czasu.
8. Petz D., *Commun. Math. Phys.* 105, 123 (1986) — mapa odzysku.
9. Helstrom C.W., *Quantum Detection and Estimation Theory* (1976) — granica Helstroma.
10. Egan C.A., Lineweaver C.H., *ApJ* 710, 1825 (2010) — budżet entropii Wszechświata.
11. Planck Collaboration, *A&A* 641, A6 (2020) — N_eff = 2.99 ± 0.17; T₀ = 2.7255 K.
12. Abdo A.A. et al., *Nature* 462, 331 (2009); Vasileiou V. et al., *PRD* 87, 122001 (2013) — GRB/LIV.
13. Hopkins P.F., Beacom J.F., *ApJ* 651, 142 (2006); Madau P., Dickinson M., *ARAA* 52, 415 (2014) — kosmiczne SFRD.
14. Planck Collaboration (2020), *A&A* 641, A6 — parametry ΛCDM (H₀, Ωm, ΩΛ) użyte w R38.

---

## Dodatek C — Audyt zamykający ENTROPIA-1.2 (R19)

Pełny raport: **AUDYT_ENTROPIA12.md**; kod: `entropia/audyt12.py`; testy:
`tests/test_audyt12.py` (19); figury: `figA1_audyt_odzyskiwalnosc`,
`figA2_audyt_27`. Sekwencja wg rekomendacji: równania → jednostki → niezależna
replikacja → T1/T2 → dark-sektor. Metoda: implementacje-świadkowie (konwencja
kolumnowa + `solve_ivp`, jawny RK4 na równaniach Blocha, jawne równania stóp
drabiny Dickego z RK45, własne formy zamknięte) — nie importują symulatorów
projektu. Najważniejsze liczby (wszystkie z uruchomień):

| Pozycja | Wynik |
|---|---|
| Lindblad wiersz vs kolumna | bitowo identyczne (max\|Δ\| = 0) |
| CPTP e^{Lτ} (100 prób) | ślad 4.45e-16, hermit. 1.11e-16, dodatniość ✓ |
| RK4 Blocha vs S(γ,t) | 3.55e-14 |
| dS/dt: dwie postacie zamknięte | 2.78e-14 (osobliwość dS/dt ∝ −ln γt przy t→0) |
| Sektory vs pełna przestrzeń (konwencja) | 4.22e-15 (N=2); 8.35e-14 (N=4) |
| I_eq = ln(2/√3) = 0.143841036 | Δ = 1.89e-11; S(∞) = ½ln12 = 1.2424533 |
| Kompresja 27× (porządna, pełny zakres B) | 4.14e-09 (metryka projektu: 4.28e-05 = artefakt przycięcia) |
| Tempo dS/dt przy S* (6 poziomów) | 27.0000 |
| T0/T1/T3 τ̇∞, T2 τ̇∞ | 0 / 0 / 0 ; 7.192052 (= η·I_eq/σ₀, Δ=1e-6) |
| Czkanie w ogonie (Δτ=0) | T0/T1/T3: 1.000; T2: 0.000 |
| M(j=1, 50τ) = ½(e^{−2γt}+e^{−6γt}) | 0.4148304; 3 ścieżki zgodne do 5e-17; niezależne od N (rozrzut 0) |
| M_bright (N=4/10/100) | 0.23235 / 0.11199 / 0.01314; zysk 1.79×/3.70×/31.58× |
| j=0 | M(t) = 1 dokładnie; superradiancja jasnego: Γ = Nγ |
| R_T (3D → 27, single → 3) | pełny spójny termiczny: TB=10 → 27.850/3.092; TB=100 → 27.079/3.009 |
| Kalibracje fizyczne | k_BT_CMB/h = 56.790 GHz; próg n̄<0.01: 261.5 GHz (ln(1+1/ε): 262.1); ω_G = 1.8549e43 rad/s |

**Znaleziska (do wdrożenia):** (1) *R47* — `R_T_fizyczny` miesza ∞-gorące
dS/dt z termicznymi czasami przejścia; pełna spójna wersja daje 0.2–0.8% wyżej;
wniosek 27/3 odporny; (2) metryka kompresji 27× — przycięty indeks 27n daje
sztuczne 4.28e-05 zamiast 4.14e-09; (3) `M_sektora`: `Ms[k] ↔ t = kτ`;
(4) próg n̄: ln(1+1/ε) zamiast ln(1/ε) (+0.22%); (5) kusp startu dS/dt.

**Bramka kosmologiczna: OTWARTA i WYKORZYSTANA** — wszystkie 5 sekcji audytu
przeszły; poprawka R47 wdrożona (znalezisko nr 1 zamknięte, projekt ≡ świadek
do 1e-10); kosmologia zabawkowa ENTROPIA-4.0 zbudowana (R48, R49 — sekcja 12).
