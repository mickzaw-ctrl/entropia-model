# Kosmologiczny model «ENTROPIA»

**Czas jest entropią. Tyknięcia są dyskretne. Czas potrafi „czkać”.**

Model łączy mechanikę kwantową układów otwartych (równanie Lindblada dla kubitu
zanurzonego w kąpieli termicznej) z relacyjną koncepcją czasu: zegar kosmiczny nie
mierzy upływu zewnętrznego czasu, lecz **kumuluje produkowaną entropię**.

## Siedem cech modelu (rdzeń)

| # | Cecha | Wynik |
|---|-------|-------|
| 0 | Entropia | S(ρ): **0 → ln 2** monotonicznie (mapa unitalna — tw. Ando–Lindblada; σ = dS/dt ≥ 0 — tw. Spohna) |
| 1 | ΔS_n na tyknięcie | A produkuje entropię **27× szybciej** niż B (T_A = 3·T_B, s ∝ T³; tempo przy dopasowanym S* = dokładnie 27.000) |
| 2 | Skumulowany czas | **T(n) = S(n)** — czas JEST entropią (κ = 1) |
| 3 | Dyskretne tyknięcia | wyraźne **schodki czasu**, nie ciągły przepływ |
| 4 | „Czkanie czasu” | przy niskiej produkcji entropii **Δt_n → 0** (B: 151/160 zer w ogonie, najdł. zamrożenie 51 tyknięć) |
| 5 | T_A/T_B | czas w gorącym otoczeniu **płynie szybciej** (T_A > T_B; granica ciągła t→0: 27) |
| 6 | Dekoherencja | **Tr(ρ²): 1 → 0.5** (czysty → maksymalnie mieszany ½·𝟙) |
| 7 | Wektor Blocha | **|r|: 1 → 0** (zanik koherencji kwantowej) |

## Rozszerzenia (część II)

### R1 — skończona temperatura kąpieli (nasycenie poniżej ln 2)
Kąpiel Gibbsa: η = e^(−βΩ); emisja a = 2γ/(1+η), absorpcja b = 2γη/(1+η).
- S(∞) = H(1/(1+η)) < ln 2; |r|∞ = (1−η)/(1+η); Tr(ρ²)∞ = (1+|r|∞²)/2.
- η=1 ⇔ rdzeń (ln 2). Dla η=0.5: S(∞) = 0.6365; η=0.1: 0.3046; η=0.01: 0.0555.
- Kompresja 27× przetrwa: S_η(t;γ_A) = S_η(27t;γ_B), błąd 2·10⁻¹⁶.
- Mapa nieunitalna: dla η < 1/3 entropia **przewyższa plateau i opada**
  (η=0.2: S_max = 0.4985 > S(∞) = 0.4506, Δ = +0.048) — tw. Spohna (σ ≥ 0) nadal działa.
- Kontrola krzyżowa dyskretny Lindblad vs postać zamknięta: błąd 3.6·10⁻¹⁴.

### R2 — wiele kubitów (ekstensywność vs korelacje)
- **Niezależne kąpiele**: S_total = N·S₁, S(∞) = N·ln 2 (błąd ekstensywności 0.0).
- **Wspólna kąpiel kolektywna (Dicke, N=2, γ_φ=0)**:
  - start |11⟩: termalizuje tylko tryplet ⇒ S(∞) = **ln 3** < 2 ln 2;
    deficyt **ln(4/3) ≈ 0.288 nat** = informacja wzajemna I (dokładnie 0.2877).
  - start |10⟩: **ciemny singlet Bell** przeżywa wiecznie ⇒ S(∞) = ½·ln 12 = 1.2425,
    czystość 1/3, I → ln(2/√3) ≈ 0.144 nat, ale **negatywność = 0** (separowalny).
  - kolektywna termalizuje szybciej: t90% ≈ 98 vs 177 tyknięć (niezależne).

### R3 — sprzężenie „zegar → tempo”
γ_eff(n) = γ₀·fb(T_n/T_scale); fb ∈ {1 (stały), 1/(1+αu) (chłodzenie), 1+αu (przyspieszanie)}.
- Skoro fb zależy od S (odczytu zegara), kompresja 27× **nie jest psuta**:
  t½_A/t½_B = 27.0 dokładnie w każdym scenariuszu; S_A(t) ≡ S_B(27t) (błąd 4·10⁻⁶).
- t½(B): stały 3.00 → chłodzenie 5.52 (czas się rozrzedza), przyspieszanie 2.15.
- Czkanie: przy nasyceniu każdy scenariusz zamraża czas; najdł. zamrożenie
  rośnie dla przyspieszania (147 tyknięć — wchodzi w nasycenie najwcześniej).

### R4 — N=3: jawne sektory j=3/2 ⊕ 2×j=1/2 (stany subradiantne)
Przestrzeń 3 kubitów: j=3/2 (4 stany symetryczne) + dwie kopie j=1/2.
Kolektywne jumpy S± nie mieszają sektorów ⇒ każdy termalizuje wewnątrz siebie.
- |111⟩ (j=3/2): S(∞) = **ln 4** (deficyt vs 3·ln 2: dokładnie **ln 2**).
- |1⟩⊗|S⟩₂₃ (j=1/2, kopia B): S(∞) = **ln 2** — „czapka” subradiantna;
  populacja na kopii B ≡ 1 (min 1-10⁻⁸) przez całą ewolucję.
- |100⟩ (mieszanina): S(∞) = ln108/3 = 1.5607.
- Widmo S² zweryfikowane: 3.75×4 + 0.75×4; jawne stany j=1/2: (|101⟩−|011⟩)/√2 itd.

### R5 — pełna entropia makro
- Niezależne kąpiele: S(∞) = **N·ln 2 = ln(2^N)** (ekstensywność; N=16: błąd 2·10⁻¹³);
  czas nasycenia niezależny od N (t90% = 177 zawsze).
- Wspólna kąpiel, start |1…1⟩: S(∞) = **ln(N+1)** (sektor symetryczny, j=N/2);
  deficyt N=2: ln(4/3), N=3: ln 2, N=4: ln(16/5).
- Kolektywna jest szybsza (superradiancja): t90% = 100, 98, 96, 94 dla N=1..4.

### R6 — gorący Wielki Wybuch jako warunek początkowy (w R3)
Start „gorący”: ρ(0) termiczne przy η₀=0.95 ⇒ S(0) ≈ ln 2; kąpiel zimna
(η_B=0.15, S(∞)=0.3872). Entropia **maleje** ⇒ zegar T(n)=S(n) **biegnie wstecz**
(Δt_n < 0; budżet czasu wstecz = 0.3056 nat).
- t½ w dół: 31.75 (stałe γ) vs 40.44 (chłodzenie R3, fb=1/(1+2u)).
- Zegar wstecz „czka”: 293/300 tyknięć z Δt=0 w oknie 200–500.
- Kontrola krzyżowa zamknięcia vs pełna mapa Lindblada 4×4: błąd 9·10⁻¹⁵.

### R7 — kąpiel kolektywna dla losowych (nie-symetrycznych) stanów
Kąpiel kolektywna termalizuje wewnątrz sektorów j; koherencje między kopiami
tego samego j **przeżywają** (identyczna dynamika), a lokalna dekoherencja γ_φ
miesza sektory i domyka do pełnej termalizacji N·ln 2.
- N=3 (γ_φ=0): |111⟩ → ln 4 = 1.3863; |100⟩ → 1.5607; losowe Haar → 1.61–1.99
  (między ln 4 a 3·ln 2); |1⟩⊗|S⟩ → ln 2. Przy γ_φ=γ: **wszystkie → 3·ln 2 = 2.0794**.
- N=4: |1111⟩ → ln 5 = 1.6094; losowe → 2.30–2.34; przy γ_φ=γ → **4·ln 2 = 2.7726**.
- Koherencja A↔B dla |100⟩ (γ_φ=0): 0.2887 = √(p_A·p_B) — maksymalna, przeżywa.
- Wniosek: sama kąpiel kolektywna nigdy nie osiąga pełnej entropii N·ln 2
  (entropia „zablokowana” w koherencjach sektorowych) — potrzebna lokalna dekoherencja.

### R8 — cykl Wielki Wybuch → ekspansja → ochłodzenie → Wielki Kolaps (czas dwustronny)
Kąpiel o oscylującej temperaturze η(n) = 1 − (1−η_min)·sin²(πn/n_cyc):
S(0) = ln 2 (gorący Wybuch) → ochłodzenie (ekspansja, S_min = 0.4850 przy
η_min=0.15) → ogrzewanie (kolaps, powrót do ln 2).
- Czas płynie **wstecz przez ~50–60% cyklu** (ujemne Δt); budżet wstecz = 0.2081 nat.
- Dwie wskazówki zegara: T(n)=S(n)−S(0) (dwustronna, wraca do zera — czas jako
  **pętla**) oraz τ(n)=Σ|ΔS| (upływ, zawsze rośnie; τ = 0.4092 = 2·budżet).
- Przy zwrocie strzałki ΔS→0: kwantowe zamrożenia (27/30 tyknięć w oknie zwrotu).
- Wszechświat cykliczny: entropia wraca do ln 2, cykl można powtarzać.

### R9 — kwantowy zegar (super-twardy): czas jako operator
Kubit + oscylator-zegar (próżnia); „kranik” σ₋⊗b† kopiuje każdą de-ekscytację
do zegara. Wskazanie ⟨n⟩ to **operator** z rozkładem p_n i nieoznaczonością Δn.
- ⟨n⟩(t) śledzi S_sys(t) (czas = entropia, kwantowo); Δn ≈ √⟨n⟩ (wzorzec Poissona);
  Δn/⟨n⟩ maleje — zegar się wyostrza.
- Back-action: γ_t=0.01 ⇒ S_sys(∞)=0.6642 (odch. −0.029), ⟨n⟩=0.52, Δn=0.74,
  I(wszechświat;zegar)=0.0149 (czas relacyjny à la Page–Wootters).
- **Kompromis zegara kwantowego**: |S(∞)−ln2| rośnie z γ_t (0.0009→0.29),
  a Δn/⟨n⟩ maleje (3.02→0.73) — precyzja kosztem zaburzenia wszechświata
  (duch Saleckera–Wignera). Granica γ_t→0: zegar nieperturbujący = czas klasyczny.

### R10 — kwantowy zegar z koherencjami (start koherentny + dekoherencja zegara)
Start zegara w stanie koherentnym |α=1.5⟩ zamiast próżni:
- **Faza czasu**: koherencje ⟨n|ρ_c|n+1⟩ = 0.031 (przeżywają), lepsza precyzja
  (Δn/⟨n⟩ = 0.60 vs 1.40 dla próżni).
- **Stymulowane kopiowanie**: kwanty w zegarze wzmacniają kranik (czynnik √(n+1))
  ⇒ back-action rośnie: |S∞−ln2| = 0.119 vs 0.029 — **4.1× mocniej**.
  Wewnętrzny stan zegara ustawia jego pozycję na kompromisie R9.
- **Dekoherencja zegara** κ·D[b†b] (κ=0.3): koherencje → 0, I → 0.016 — **czas
  klasyczny** (diagonalny); fizyka wszechświata (S∞, ⟨n⟩, Δn) bez zmian.
- Mechanizm „kwantowy czas → klasyczny czas”: pomiar/otoczenie niszczy koherencje
  wskazań, wskazówka zegara pozostaje ta sama.

### R11 — od blokady do pełnej termalizacji: prawo τ ∝ 1/γ_φ
Sweep pośredniego γ_φ dla losowych stanów (kąpiel kolektywna):
- Dla **każdego γ_φ > 0** entropia w końcu osiąga N·ln 2 (blokada przy γ_φ = 0
  jest dokładna); czas odblokowania **τ90 ∝ 1/γ_φ** (nachylenie −0.96 dla N=3):
  γ_φ=1e-4 → 1621; 1e-3 → 164; 1e-2 → 28; 0.3 → 10 (podłoga = czas kolektywny).
- Blokada przy γ_φ=0 (N=3): |111⟩: 1.3863 (zablokowane ln 2), |100⟩: 1.5607,
  losowy: 1.4817.
- N=4: dopasowanie płytsze (−0.49), bo losowy start jest blisko celu
  (plateau ~2.3 vs 4·ln2) — τ90 dominuje czas kolektywny.
- Wniosek: **czas potrzebuje dekoherencji, by dotrzeć do pełnej entropii**.

### R12 — wielka synteza (diagram powiązań R1–R12)
Jeden model, trzy osie rozwoju, jeden zegar T(n)=S(n):
- **Termodynamika** (R1): temperatura = cel i tempo (S(∞)=H(1/(1+η)), 27× trwa).
- **Wiele ciał** (R2→R4→R5→R7→R11): struktura sektorowa — koherencje blokują
  entropię (R7), dekoherencja odblokowuje z prawem 1/γ_φ (R11); ciemne stany (R4),
  ln(N+1) vs N·ln 2 (R5).
- **Kosmologia** (R3→R6→R8): tempo z zegara (R3), gorący start cofa czas (R6),
  cykl zamyka czas w pętlę (R8).
- **Czas kwantowy** (R9→R10): czas jako operator (R9), koherencje i dekoherencja
  zegara (R10).
Raport zawiera diagram SVG (inline) i tabelę syntezy wszystkich 12 rozszerzeń.

### R13 — grawitacyjna produkcja entropii (dwie kąpiele, NESS)
Kubit + **dwie kąpiele**: gorące promieniowanie (η_r=0.9, γ_r=0.05) i zimna
„kąpiel grawitacyjna" (η_g=0.1, γ_g=0.01). Energia płynie promieniowanie → kubit
→ grawitacja; stan stacjonarny jest **nie-równowagowy (NESS)**:
- S(NESS) = 0.6768 < ln 2; **σ_NESS = 0.01402** (stałe; tw. Spohna, obie σ_i ≥ 0).
- Zegar grawitacyjny **T_graw(n) = Σ σ_k·τ rośnie liniowo bez końca** — po 400
  tyknięciach 10.34 nat ≈ **14.9 × ln 2** (cały budżet zwykłego zegara!). Grawitacja
  utrzymuje produkcję entropii ⇒ **czas nigdy nie zamiera** (zabawkowy odpowiednik
  czarnych dziur/kondensacji jako wiecznych źródeł entropii).
- σ_NESS rośnie z γ_graw (0.0032 → 0.0561): silniejsza grawitacja = szybszy czas.
- Kontrola: η_r = η_g ⇒ σ = 0 (równowaga, koniec produkcji).

### R14 — predykcje modelu a dane obserwacyjne
Zestawienie 12 predykcji z realnymi danymi (szczegóły: `PREDYKCJE.md`, sekcja
R14 w raporcie):
- ✅ **P1** s ∝ T³: T_ν/T_γ = (4/11)^{1/3} = 0.7138, N_eff = 2.99±0.17 (Planck
  2018) — zgodność ilościowa (mechanizm BBN/CMB = mechanizm modelu).
- ✅ **P7** subradiancja: zmierzona w zimnych atomach (PRL 116, 083601; PRL 128,
  203601); analogia: neutrina odsprzężone niosą 49% entropii promieniowania
  (model R2 |10⟩: 50% „ciemnej" entropii — zbieżność).
- 🟡 **P3/P4/P5/P6/P9**: jakościowo — zegar entropii (S_obs/S_CEH ≈ 10⁻¹⁸),
  heat death = koniec czasu, kosmiczne tempo produkcji entropii maleje (SFRD
  szczyt z≈2, spadek ~10×), SMBH dominują budżet entropii, dekoherencja ∝ T.
- ⚠️ **P11/P12**: napięcia — czas wstecz przy chłodzeniu nie ma odpowiednika
  (realna ekspansja adiabatyczna, entropia rośnie); cykl przeczy przyspieszającej
  ekspansji (ΛCDM).
- ❓ **P10**: dyskretność czasu — brak obserwacji; GRB 090510: E_QG,1 > 7.6·E_Pl
  (zgodne, τ wolny).

### R15 — dekoherencja zegara jako strażnik historii (κ w punkcie zwrotnym)
**Teza (demonstrowana numerycznie).** W stanie ρ_SC jawny Hamiltonian
H_int = g(σ₋b† + σ₊b) (Jaynes–Cummings) splątuje układ z zegarem i rozmywa czas
(tworzy koherencje wskazań — superpozycję „która godzina?"); człon κ·D[b†b]
nieustannie rzutuje zegar na oś liczbową (baza Focka — einselection), zamieniając
kwantowe „czkanie" (zdarzenia kranika σ₋⊗b†) w mierzalny, nieodwracalny przyrost
drogi czasu τ = ⟨n⟩·δs.
- Punkt zwrotny (nasycenie, ΔS→0): κ=0 → rozmycie czasu 1.9×10⁻² (koherencje
  wskazań, czas kwantowy); κ=0.5 → 3.4×10⁻⁴ (**56× mniej**, oś klasyczna).
- Nieodwracalność: S(zegar) rośnie (0.96 → 1.19 — faza zniszczona); korelacja
  kwantowa I(S;C) spada (0.078 → 0.021) — zostaje klasyczny zapis; ⟨n⟩
  monotoniczny (historia jednoznaczna).
- To standardowa fizyka pomiaru (Zurek 1981; ciągły pomiar w granicy dużego κ),
  zilustrowana pełną ewolucją Lindblada ρ_SC modelu; liczby z symulacji.

### R16 — formalizm relacyjny po recenzji (λ → S → τ)
Rewizja założeń wg krytyki: zamiast T ≡ S — schemat trzypoziomowy λ → S → τ
z funkcjonałem dτ_A/dλ = α[Ṡ_A^prod + η·I(A:E)].
- η = 0: stary zegar entropii (przypadek szczególny); η > 0: korelacje
  (I system-zegar) dodają czasu (τ: 79.0 → 83.7 przy η=0.5).
- **„27” jako predykcja warunkowa**: gałąź dτ∝s (γ∝T³) daje 27 dokładnie;
  gałąź dτ∝Ṡ daje ≠27 (8.8 w 1. tyknięciu); γ∝T^p ⇒ stosunek 3^p — test
  falsyfikacyjny (zmierz τ_A/τ_B ⇒ wyznacz p).
- Zatrzymanie zegara entropowego ≠ koniec czasu fizycznego (τ̇→0 w równowadze,
  ewolucja może trwać).

### R17 — laboratoryjny test: jasny ↔ ciemny zegar entropowy
Formuła recenzji Δτ_n = τ₀·ΔS_n/ΔS_ref w układzie N=2 kąpieli kolektywnej:
- Singlet (ciemny): τ = 0 — zegar MILCZY (Γ_dark = 0); precesuje unitarnie
  przy S = 0 (zegar stoi, fizyka trwa).
- Tempo ⟨Ṡ⟩ spada liniowo z frakcją ciemną p: 0.0183 → 0.
- |10⟩: τ̇ → 0 po wygaśnięciu jasnej części, a pamięć I(A;B) = ln(2/√3) =
  0.1438 nat trwa w ciemnym sektorze (informacja → sektor ciemny → pamięć).
- **Przewidywanie falsyfikowalne**: przejście układu do sektora subradiacyjnego
  spowalnia entropiczny zegar — testowalne w zimnych atomach (PRL 116, 083601).

### R18 — ENTROPIA-1.1: pełna symulacja N = 2..100 (baza Dickego)
Rozwiązanie Lindblada przez rozkład na sektory spinu j (wymiar ≤ N+1):
- **S∞ → ln(N+1)** dla sektora symetrycznego (N=100: 4.6151, błąd 7e-5).
- **27× dokładne dla wszystkich N**: max|S_A(n)−S_B(27n)| ≤ 3e-12 (tożsamość
  wynikająca z liniowego skalowania tempa).
- **Czkanie**: kwantowany zegar entropii staje przy nasyceniu (η=0);
  z η>0: τ̇ → η·I_eq = 0.0719 ≠ 0 — pamięć napędza czas — ROZRÓŻNIALNE.
- **Pamięć**: P_dark(Haar) = 1−(N+1)/2^N → 1 (typowe stany prawie w całości
  ciemne); I(A:B) = ln(2/√3) plateau; S∞(Haar,γφ=0)/N: 0.52 → 0.05.
- Werdykt: 27× i pamięć wynikają z modelu; „pełne czkanie" wymaga η=0 —
  wybór funkcjonału jest falsyfikowalny (test bright↔dark).
- Moduł `entropia/dicke.py` (sektory, Lindblad rzadki dla N=100, redukcje dla
  I(A:B)), runner `entropia_1_1.py` / `entropia.e11`, figury figE1–E4.

### R19 — ENTROPIA-1.2: konkurencja funkcjonałów, odzyskiwalność, fizyczny 27×
Wg recenzji rozdzielamy falsyfikację od konsekwencji definicji:
- **Cztery funkcjonały czasu**: T0 (σ), T1 (σ+η|İ|), T2 (σ+η·I), T3 (σ+η|Ṙ|).
  T0, T1, T3 STAJĄ przy równowadze (czkanie); **T2 nie** (τ̇ → η·I_eq/σ₀ = 7.19).
  ENTROPIA-1.1 implementowała T2 (absolutna I), nie T1 (|İ|) — diagnoza recenzji
  potwierdzona; wybór teorii czasu (produkcja vs istnienie informacji) testowalny.
- **Odzyskiwalność** M(t)=D(t)/D(0) (trace distance): sektor subradiacyjny j=1
  ma M=0.41 niezależnie od N; jasny j=N/2 spada 0.23→0.013 — przewaga 31× przy
  N=100; **j=0: M=1 dokładnie** (Γ=0). Niska entropia ≠ brak pamięci (recenzja §7).
- **Fizyczny 27×**: γ(T), η(T)=e^{−ω₀/T} z konkretnej kąpieli; R_T przy
  T_A=3T_B: kąpiel 3D fotonowa (J∝ω³) ⇒ R_T=27.6→27 (lim. gorący); single-mode
  ⇒ R_T=3.07. „27" to predykcja widma kąpieli, nie symetrii L.
- Moduł `entropia/e12.py`, figury figE5–E7, testy test_e12.py (7 testów).

### R20 — ENTROPIA-1.3: coherent information, koszt zegara, protokół T1 vs T2
- **I_c < 0 przy równowadze** (N=4: −0.51, N=2: −0.55) mimo I(A:B) > 0 —
  pamięć subradiacyjna jest KLASYCZNA (korelacje, nie splątanie destylowalne);
  T3c (z |Ī_c|) staje.
- **Koszt zegara (Salecker–Wigner w modelu)**: E_clock = ω_c⟨n⟩, ΔE = ω_cΔn,
  Δτ = Δn·δs; ΔE·Δτ ≥ ħ/2 ⇒ **ω_c ≥ ω_c^min = 1.7**; trójkąt
  precyzja (Δn/⟨n⟩: 1.02→0.47) ↔ koszt (E: 270→462) ↔ entropia
  (back-action: 0.0002→0.22) jako funkcje γ_t.
- **Protokół rozstrzygający T1 vs T2**: po wygaśnięciu fluorescencji Γ=0
  zmierz τ̇: T1 → 0 (zegar staje), T2 → η·I_eq/σ₀ = 7.19 (tyka dalej).
  Test w układzie Dicke'a (subradiancja zmierzona: PRL 116, 083601).
- Moduł `entropia/e13.py`, figury figE8–E10, testy test_e13.py (5 testów).

### R21–R23 — ENTROPIA-1.4: kanał odzysku (fidelity), ω_c(T), protokół e2e
- **Fidelity-based recovery**: M_F(t) = 1−F(ρ₀,ρ₁): ciemny j=1 zachowuje
  rozróżnialność niezależnie od N (0.337), jasny rozpada (N=100: 0.0009) —
  zysk 382×; F_e(j=0) = 1 (kanał identyczności); Fuchs–van de Graaf łączy z M(t).
- **ω_c(T) z fizycznej kąpieli**: n̄(ω_c,T)<ε ⇒ ω_c > T·ln(1/ε); ΔE·Δτ≥ħ/2 ⇒
  ω_c≥1.7; pojemność ln2/δs=69. Lim. gorący: **ω_c ∝ T** (rozdzielczość ~ T),
  produkcja entropii ∝ T³ (27×) — dwie skale rozdzielone.
- **Protokół e2e z detekcją fotonów**: MC z η_det i dark counts; po ostatnim
  fotonie τ̇ rozstrzyga T1 (0) vs T2 (7.19); **moc 1.000** nawet przy
  η_det=0.1 i szumie tła (separacja zbyt duża dla szumu Poissona).
- Moduł `entropia/e14.py`, figury figE11–E13, testy test_e14.py (7 testów).

### R24–R26 — ENTROPIA-1.5: pamięć operacyjna, samo-spójny ω_c(T), SPRT
- **Pamięć operacyjna (Helstrom/Chernoff/C_mem)**: p_err = ½(1−D/2),
  C_mem = 1−h₂(p_err). j=1/2: p_err=0.13, **C_mem=0.44 bitu**; jasny N=100:
  p_err=0.49, C_mem=0.0004 (t=7.5); j=0: niezmienniczy, ale 0 bitów (dim 1).
- **Samo-spójny ω_c(T)**: Purcell γ_t∝g²ω_c³ ⇒ górna granica ω_c <
  (ε_b/(cg²))^{1/3}; termiczna dolna ω_c > T·ln(1/ε) ⇒ **okno istnienia zegara
  i T_max ∝ g^{−2/3}** (nowa predykcja: zegary nie istnieją w zbyt gorących
  kąpielach).
- **SPRT (Wald)**: sekwencyjny iloraz wiarogodności — E[N]=1 tyknięcie przy
  λ₂=7.19 (błędy 0), adaptacja: λ₂=0.1 ⇒ E[N]=20. Optymalny minimalny koszt
  przy zadanych α=β=0.01.
- Moduł `entropia/e15.py`, figury figE14–E16, testy test_e15.py (7 testów).

### R27 — Eksperymentalna karta protokołu R23 (konkretne parametry)
Przeliczenie protokołu na wykonalny eksperyment (szczegóły: `EKSPERYMENT.md`):
- **3 platformy**: A) wolna przestrzeń ⁸⁷Rb (OD=100, t_D=2.6 μs), B) nanofiber
  ¹³³Cs (β=0.15, **t_D=179 μs** — najlepsze okno), C) wnęka (t_D=1.3 μs).
- **Sekwencja** (minuty): pułapka → stan |10⟩-typ (1 ekscyton) → faza jasna
  (SPCM) → faza ciemna → pomiar I(A:B) w 6 punktach (M=150 realizacji/punkt,
  σ_I=0.01 nat) → rozstrzygnięcie T1 vs T2.
- **Uczciwa granica**: kanał fotonowy nie rozróżnia T1/T2 (zegar bez sprzężenia
  zwrotnego) — rozstrzyga kanał korelacyjny I(A:B).
- **Werdykt**: wykonalne na istniejącej technologii (nanofiber+SPCM+obrazowanie);
  cały pomiar w ~minutę. Moduł `entropia/experyment.py`, figura figE17.

### R28–R30 — ENTROPIA-1.6: suchy bieg, mapa Petza, zimny zegar
- **Suchy bieg protokołu (realistyczny detektor)**: MC z η=0.3, dark=100 Hz,
  jitter=1 ns — SPRT E[N]=1, błędy 0, pomiar ~65 μs/realizację; jitter ≪ t_B,t_D
  nie wpływa; karta eksperymentalna potwierdzona.
- **Mapa Petza (jawny kanał odzysku)**: R(·)=σ^{1/2}Φ†(Φ(σ)^{−1/2}(·)Φ(σ)^{−1/2})σ^{1/2};
  F_rec: j=1/2 → 0.886, j=1 → 0.790, j=2 → 0.645, j=3 → 0.543, j=0 → 1.
  Najsilniejsza miara pamięci — jawny protokół odzysku, nie tylko miara.
- **Zimny zegar (skończona gęstość widmowa)**: Ohmic/Lorentzian saturują
  γ_t(ω_c) ⇒ T_max = 215804 vs 0.66 (3D) — o 5 rzędów; kosmologia:
  ω_c ≥ 2π×2.6e11 Hz (dziś), 2π×3e14 (rekombinacja), 2π×1e21 (BBN),
  2π×1e26 (elektrosłaba). Dziś/rekombinacja łatwe, BBN trudne.
- Moduł `entropia/e16.py`, figury figE18–E19, testy test_e16.py (5 testów).

### R31–R33 — ENTROPIA-1.7: arkusz T_max, koszt energii, suchy bieg z F
- **Arkusz T_max/ω_c(T) (nadprzewodniki)**: zegar = rezonator, kąpiel = szum
  termiczny; T_max(6 GHz) = 63 mK (lodówka 10–30 mK OK), n̄(ω_c,T) mierzalne
  spektroskopią, Purcell T1 = 64 μs; ω_c ∝ T (×3.00). Tani test R25/R30.
- **Koszt energetyczny**: zegar 2.6×10⁻²³ J, decyzja (Landauer 100 mK)
  9.6×10⁻²⁵ J — zaniedbywalne; pułapka 5 mJ dominuje (technika); ΔE·Δτ ≥ ħ/2 ✓.
- **Suchy bieg z niedoskonałą wiernością** ρ(F)=F·ρ10+(1−F)·𝟙/4: I_eq
  0.144→0.014, τ̇_T2 7.19→0.71 (F: 1→0.3), ale moc 1.000 — protokół
  samo-kalibruje się przez pomiar I(A:B); decyzja odporna na systematykę.
- Moduł `entropia/e17.py`, figury figE20–E22, testy test_e17.py (6 testów).

### R34–R36 — ENTROPIA-1.8: realizacja Petza, zegar w CMB, sieć zegarów
- **Realizacja mapy Petza**: kody fazowe — F(Petz) > F(klasyczny) zawsze; echo
  (π-pulsy) odzyskuje fazę przy dekoherencji czystej (j≤1), szkodzi przy zaniku
  amplitudowym (j≥1.5); najlepsza realizacja R = kodowanie w ciemnym sektorze
  (DFS): F=1 — „chroń, nie odzyskuj".
- **Zegar w kąpieli CMB**: próg ω_c/2π ≥ 261 GHz (ε=0.01); 100 GHz n̄=0.207
  (szum), 1 THz bezpieczny; grzanie ∝ βω³(n̄+1) — ograniczenie na sprzężenie;
  cutoff grawitacyjny (ω_Planck) bez wpływu dla realistycznych częstości.
- **Sieć zegarów**: synchronizacja przez wymianę entropii — σ_end 48→0.08
  (g_sync 0→0.2); jednakowe T ⇒ σ≡0 bez sprzężenia (naturalny kosmiczny czas);
  τ_net = emergentny czas sieci.
- Moduł `entropia/e18.py`, figury figE23–E25, testy test_e18.py (7 testów).

### R37–R39 — ENTROPIA-1.9: protokół różnicowy, CMB (ewolucja), manuskrypt
- **Protokół różnicowy wielu zegarów**: Δτ̄ = τ_A−τ_B per tyk fazy ciemnej —
  T1 ≈ 0 vs T2 ≈ 7.19 nat/tyk; common mode odrzucony, brak kalibracji
  absolutnej, σ ↓ 1/√M_A (0.25→0.09). To wprost predykcja recenzji §10.
- **Zegar w ewoluującym CMB (ΛCDM)**: horyzont zegarów — 6/100 GHz nigdy,
  300 GHz od z≈0.1, 1 THz od z≈2.8, 3 THz od z≈10.5, 10 THz od z≈37;
  t(z=0) = 13.79 Gyr (weryfikacja); cutoff grawitacyjny bez wpływu.
- **Manuskrypt zbiorczy** `MANUSKRYPT.md`: pełny opis modelu R1–R38 —
  streszczenie, rdzeń, sektory, kosmologia, zegar kwantowy, program
  falsyfikacyjny, eksperyment, predykcje vs obserwacje, 10 uczciwych uwag,
  bibliografia. Pakiet publikacyjny.
- Moduł `entropia/e19.py`, figury figE26–E27, testy test_e19.py (6 testów).

### R40–R41 — ENTROPIA-2.0: sieć z η(T), asymptotyka Petza
- **Sieć z dynamiką η(T) (R8×R36)**: cykliczna kąpiel — upływ τ_abs rośnie
  przez cykle (3×budżet = 1.24 nat), T_signed wraca (pętla); jednakowe komórki
  σ ≡ 0; offsety fazy: synchronizacja modulowana cyklem. Czas sieci przetrwa
  cykl kosmiczny.
- **Asymptotyka Petza (R41)**: F_rec → 1/(N+1) dla t→∞ (dokładnie);
  C(t) = F_rec − 1/(N+1) ≈ 0.215 ± 0.017 niezależne od N (N=4..16);
  superradiacyjny dolny szczebel Γ₁ = Nγ niszczy pamięć; ciemny: F=1.
- Moduł `entropia/e20.py`, figury figE28–E30, testy test_e20.py (6 testów).

### R42–R43 — ENTROPIA-2.1: formalny limit Petza, entrainment faz
- **Formalny limit Petza**: gap/γ = 1.0000 dla N=2..100 (utrata pamięci w
  skali 1/γ, niezależna od N); dokładny wzór F_rec(t) = ½a(2+(1−a)²/(1−½a²)),
  a=e^{−Γt}, potwierdzony do Δ=1e-16; F_rec → 1/(N+1) → 0 (N→∞, jasny),
  ciemny → 1.
- **Entrainment faz (R8×R36)**: σ_φ: 9.56 → 0.000 (g_sync=0.2) — fazy cykli
  η_k(t) LOCKUJĄ się przez wymianę entropii; jednolity kosmiczny czas
  emergentny z niejednorodnych komórek.
- Moduł `entropia/e21.py`, figury figE31–E33, testy test_e21.py (5 testów).

### R44–R45 — ENTROPIA-3.0: dowód C(t) (drabina Dickego), metryka FRW
- **Dowód uniwersalności C(t)**: drabina Dickego Γ_n = n(N−n+1)γ (Γ₁=Nγ,
  gap=γ niezależne od N); okno uniwersalności t∈(1/(Nγ), 1/γ); dokładny wzór
  F_rec=½a(2+(1−a)²/(1−½a²)) (Δ=9e-16).
- **Metryka FRW**: entropia komobowa s·a³=const (T³); S_eq komórki maleje przy
  ekspansji (dτ=|dS|); horyzont: S_BH 10¹⁴⁰→10¹³⁰ k_B (z: 0→1100); czas FRW =
  upływ entropii komobowej, grawitacja = budżet.
- Moduł `entropia/e22.py`, figury figE34–E35, testy test_e22.py (6 testów).

### R46 — Pełny dowód wzoru Petza z regularyzacją (PETZ_DOWOD.md)
Cztery twierdzenia (każde zweryfikowane numerycznie):
- **Tw.1**: F_rec = ½a(2+(1−a)²/(1−½a²)), a=e^{−Γt} (7 kroków, Δ=3e-16);
  F_stable = (1−½a)/(1−½a²).
- **Tw.2**: N≥2 zimna — podprzestrzeń kodowa niezmiennicza, Γ=Nγ
  (|⟨S₋⟩|²=N); Petz rzutowany = Tw.1 z Γ=Nγ.
- **Tw.3**: regularyzacja σ_ε=(1−ε)σ+ε𝟙/d ma granicę; pełnosektorowy ≠
  rzutowany (przeciek Φ† przez szczebel 2-eksc) — pełny = średnia rzutowana.
- **Tw.4**: C(t) uniwersalne w oknie (1/(Nγ),1/γ); gap=γ; N→∞: jasny→0, ciemny→1.
- Moduł `entropia/e23.py`, figura figE36, testy test_e23.py (7 testów),
  dokument `PETZ_DOWOD.md`.

### AUDYT ENTROPIA-1.2 (zamykający iterację 1.2 przed dalszą kosmologią)
Audyt wg sekwencji: równania → jednostki → niezależna replikacja → T1/T2 →
dark-sektor. Implementacje-świadkowie (konwencja kolumnowa + solve_ivp, RK4 na
równaniach Blocha, jawne równania stóp drabiny Dickego, własne formy zamknięte)
odtwarzają wszystkie liczby projektu:
- **Równania**: Lindblad wiersz ≡ kolumna (bitowo); CPTP ~1e-16; RK4 ≡ analityk
  3.6e-14; I_eq = ln(2/√3) co do 1.9e-11.
- **Jednostki**: σ₀ = δs; T2 wymiarowo niejednorodny (poziom vs tempo) — to
  wyjaśnia, czemu T2 nie staje; kalibracje: k_BT_CMB/h = 56.790 GHz, próg
  n̄<0.01: 261.5 GHz, ω_G = 1.855e43 rad/s.
- **Replikacja**: kompresja 27× (porządna) 4.14e-09 (metryka projektu 4.28e-05
  = artefakt przycięcia); tempo przy S* = 27.0000; T2 = 7.192052; M(j=1,50τ)
  = ½(e^{−2γt}+e^{−6γt}) = 0.4148304 (3 ścieżki, Δ ~ 1e-15); zysk 31.58×.
- **Znaleziska**: (1) R_T_fizyczny mieszał ∞-gorące dS/dt z termicznymi t*
  (różnica 0.2–0.8%, wniosek 27/3 odporny) — **WDROŻONE jako R47** (projekt ≡
  świadek do 1e-10); (2) metryka kompresji przycięta; (3) M_sektora:
  Ms[k]↔t=kτ; (4) próg ln(1+1/ε); (5) osobliwość dS/dt ∝ −ln γt przy t→0.
- **Bramka kosmologiczna: OTWARTA i WYKORZYSTANA** (R47 wdrożony;
  ENTROPIA-4.0 zbudowana).
- Moduł `entropia/audyt12.py`, figury figA1–A3, testy test_audyt12.py (19),
  dokument `AUDYT_ENTROPIA12.md`, dodatek C w `MANUSKRYPT.md`.

### R47 — Poprawka R_T_fizyczny (zamknięcie znaleziska audytu nr 1)
- `R_T_fizyczny` używa teraz **pełnej spójnej termicznej** pochodnej dS/dt
  (`dSdt_termiczne_analitycznie`), zamiast mieszanej (termiczne t*, ∞-gorące
  tempo). Po poprawce projekt ≡ niezależny świadek audytu do ~1e-10.
- Nowe wartości: T_B/ω₀ = 10 → R_T = 27.850 (3D) / 3.092 (single);
  T_B/ω₀ = 100 → 27.079 / 3.009. Wniosek 27/3 odporny.
- Figura figA3_27_poprawka; moduł `entropia/e12.py` (R47), testy
  test_audyt12.py (test_replikacja_27_limity: |proj−świadek| < 1e-6).

### R48–R49 — ENTROPIA-4.0: dwukomórkowy wszechświat (kosmologia zabawkowa)
- **R48 (dwie komórki z wymianą entropii)**: kubit A w gorącej kąpieli
  (T_A = 3·T_B, γ_A = 27·γ_B), B w zimnej; kanał wymiany σ₋^Aσ₊^B/σ₊^Aσ₋^B
  (κ=0.3, zachowuje E_A+E_B). NESS: S_tot∞ = 1.3491 nat, prąd energii
  J_E,∞ = 0.00507, produkcja σ_NESS = 0.00338. **Clausius/Onsager:
  σ_NESS = J_E,∞·(1/T_B − 1/T_A) do 1e-6.** Prawo Fouriera: J↑ΔT
  (quasi-liniowo). Produkcja dominuje w zimnej komórce.
- **R49 (grawitacja = budżet entropii)**: entropowa siła — S∞(κ) rośnie
  (1.2617→1.3546) ⇒ F(d) < 0 (przyciąganie; uczciwie: nie 1/d²).
  Emergentna FRW: start w inwersji (T_eff < 0), przejście przez T = ∞
  (t=1.10) = Wielki Wybuch (a=0); a(τ): 0 → 1.2374 → 1 (ekspansja +
  kontrakcja = **odbicie**, T_eff przestrzeliwuje); H: +∞ → 0 (t=3.45) → 0;
  z: ∞ → 0 (przy kontrakcji z < 0 — ku fioletowi). Dwa zegary: τ_sys
  skończony (1.3491 nat — śmierć cieplna), τ_bud liniowy (σ_NESS > 0).
  Dylatacja: σ_A/σ_B ≈ 22 (faza zegarowa, rząd 27) → 0.034 (NESS).
- Moduł `entropia/e24.py`, figury figE37–E40, testy test_e24.py (15),
  sekcja 12 manuskryptu.

### R50 — ENTROPIA-5.0: pętla pomiarowa na procesorze kwantowym (IBM/Sycamore)
Protokół testu dark-sektoru do wykonania na realnym sprzęcie (szkic użytkownika,
poprawiony):
- **KRYTYCZNA POPRAWKA**: `h, cx, x` przygotowuje **Ψ+ (jasny tryplet)**, nie
  singlet — |⟨ψ|T0⟩|² = 1.000, |⟨ψ|D⟩|² ≈ 5e-34. Trzeba `h, cx, z, x` →
  Ψ− (|⟨ψ|D⟩|² = 1.000). Bez z cały eksperyment testowałby stan jasny.
- **Fizyka**: S−|D⟩ = 0 (ciemny), S−|T0⟩ = √2|00⟩ (superradiancja 2γ).
  P_D(t): kąpiel kolektywna = 1.000000 (ochrona); niezależna = e^{−γt}
  (0.6058 przy γt=0.5) — **kryterium falsyfikacji kąpieli**. |T0⟩: e^{−2γt}.
  rz na q0 odblokowuje: P_D(Δω=0.05) = 0.7036 (analog R11).
- **Kolektywny rozpad przez ancillę**: unitarna osadka V ≡ kanał Krausa
  dokładnie (Δ = 0.00e+00); dekompozycja ~8–14 CX/krok (rotacja Dickego W +
  rotacje Givensa + reset).
- **Tomografia**: baza Bella (Ψ− ↔ |11⟩) + pomiar losowy (X/Y/Z) z
  rekonstrukcją LS (F = 0.9669 przy 16k). Szum: σ=1% przy ~2500 strzałów.
- **Budżet NISQ**: Heron 3–9 kroków (krok 4–10 μs), Willow 2–5 — dlatego
  γ·dt dobieramy tak, by γ·t_max ≈ 1–2; pomiar różnicowy (kolektywna vs
  niezależna) bez kalibracji absolutnej.
- Moduł `entropia/e25.py`, figury figE41–E43, testy test_e25.py (14),
  sekcja 13 manuskryptu.

## Struktura kodu (paczka z testami)

```
entropia/
├── pyproject.toml          # pakiet "model-entropia" (numpy, scipy, matplotlib)
├── model_entropia.py       # shim → entropia.core
├── model_rozszerzenia.py   # shim → entropia.extensions
├── zrob_raport.py          # shim → entropia.report
├── entropia/
│   ├── __init__.py         # API paczki (wersja 5.0.0)
│   ├── core.py             # rdzeń: Lindblad, zegar, figury 1–7
│   ├── extensions.py       # R1–R13 + dSdt_termiczne (R47): fizyka + figury
│   ├── dicke.py            # baza Dickego N≤100, superoperatory (rzadkie >32)
│   ├── experyment.py       # karta eksperymentalna R27
│   ├── e11.py…e23.py       # ENTROPIA-1.1…3.1 (R18–R46), figury figE1–E36
│   ├── e24.py              # ENTROPIA-4.0 (R48–R49): dwie komórki, siła, FRW
│   ├── e25.py              # ENTROPIA-5.0 (R50): pętla pomiarowa IBM/Sycamore
│   ├── audyt12.py          # audyt zamykający ENTROPIA-1.2 (świadkowie, figA1–A2)
│   └── report.py           # budowa raportu (build(), R1–R50)
├── tests/                  # pytest (166 testów)
│   ├── test_core.py        # monotoniczność, ln 2, 27×, dekoherencja, zegar
│   ├── test_extensions.py  # R1–R13: wartości analityczne, kontrole krzyżowe
│   ├── test_clock.py       # czkanie, czas wstecz, cykl, back-action
│   ├── test_e11.py…test_e23.py  # ENTROPIA-1.1…3.1 (118)
│   ├── test_audyt12.py     # audyt ENTROPIA-1.2 (19)
│   ├── test_e24.py         # ENTROPIA-4.0 (15)
│   └── test_e25.py         # ENTROPIA-5.0 / R50 (14)
├── figury/                 # 70+ wykresów PNG (fig1–7, figR*, figE*, figA*)
└── raport.html             # raport (R1–R50, 5 demo, diagram SVG)
```

Uruchomienie:
```bash
pip install numpy scipy matplotlib pytest
python3 -m pytest tests/ -q          # 166 testów
python3 zrob_raport.py               # pełna regeneracja raport.html (~8 min)
python3 -m entropia.e25              # ENTROPIA-5.0 / R50 (protokół sprzętowy)
python3 -m entropia.e24              # ENTROPIA-4.0 (R48–R49)
python3 -m entropia.audyt12          # audyt ENTROPIA-1.2
```

## Fizyka (rdzeń)

```
dρ/dt = −i[H,ρ] + γ·D[σ₋] + γ·D[σ₊] + γ_φ·D[σ_z],   D[L]ρ = LρL† − ½{L†L,ρ}
ρ_eq = ½·𝟙  (kąpiel nieskończenie gorąca)  ⇒  S(∞) = ln 2
ΔS_n = S(ρ_n) − S(ρ_{n−1}),   Δt_n = κ·ΔS_n,   T(n) = Σ ΔS_k
k_n ~ Poisson(ΔS_n/δs)  — entropia kwantowana w „bitach” δs
```

- Mikro-tyknięcie τ = 0.25, N = 400 tyknięć, γ_B = 0.02, γ_A = 27·γ_B = 0.54,
  γ_φ = 2γ, Ω = 0.4, stan początkowy |ψ⟩ (θ = 60°, φ = 45°).
- Dokładna tożsamość kompresji czasowej: **S_A(t) ≡ S_B(27·t)** — porządny test
  (pełny zakres B) daje błąd 4.14·10⁻⁹; w raporcie podawane 4·10⁻⁵ było
  artefaktem przycięcia indeksu 27n do N_TICKS (patrz AUDYT_ENTROPIA12.md).

## Pliki

| Plik | Opis |
|------|------|
| `model_entropia.py` | rdzeń: Lindblad 4×4, zegar stochastyczny, 7 figur, liczby kluczowe |
| `model_rozszerzenia.py` | rozszerzenia R1–R12: temperatura Gibbsa, kubity kolektywne (N=2,3,4), feedback, sektory j=1/2, entropia makro, gorący WB, losowe stany, cykl BB→Kolaps, kwantowy zegar (+koherencje), odblokowanie γ_φ; figury figR*.png |
| `zrob_raport.py` | buduje samodzielny `raport.html` (figury base64 + 5 interaktywnych demo JS + diagram syntezy SVG) |
| `figury/` | wykresy PNG (fig1…fig7 — rdzeń, figR1…figR6 — rozszerzenia) |
| `raport.html` | raport końcowy (polski), samodzielny — R1–R27, 40 figur, 5 demo, diagram SVG |
| `PREDYKCJE.md` | predykcje modelu vs dane obserwacyjne (12 predykcji, oceny, źródła) |
| `PETZ_DOWOD.md` | pełny dowód wzoru Petza z regularyzacją (Tw.1–Tw.4) |
| `AUDYT_ENTROPIA12.md` | audyt zamykający ENTROPIA-1.2: równania, jednostki, replikacja, T1/T2, dark-sektor |
| `EKSPERYMENT.md` | eksperymentalna karta protokołu R23: platformy, liczby, sekwencja, moc |
| `MANUSKRYPT.md` | pakiet publikacyjny: spójny manuskrypt całego modelu (R1–R38) |
| `predykcje.py` | obliczenia predykcji + figura figR14 |

## Uruchomienie

```bash
pip install numpy scipy matplotlib
python3 model_entropia.py        # rdzeń: figury + liczby kluczowe
python3 model_rozszerzenia.py    # rozszerzenia: figR*.png + liczby
python3 zrob_raport.py           # regeneracja raport.html (robi wszystko powyżej)
```

## Uwagi

- Model jest fenomenologiczną zabawką (jedna komórka-kubit); „27” wynika z przyjętej
  interpretacji T_A = 3·T_B przy entropii właściwej promieniowania s ∝ T³ — łatwo
  zmienić stosunek temperatur w parametrach.
- W pierwszym tyknięciu stosunek ΔS_A/ΔS_B ≈ 8.8 (a nie 27) — efekt logarytmicznej
  osobliwości dS/dt przy |r| → 1; stosunek 27 odnosi się do tempa przy dopasowanym
  poziomie entropii (granica ciągła) i do kompresji czasowej.
- W rdzeniu zmienne `sp`/`sm` są przemianowane (sp = σ₋, sm = σ₊); przy równych
  tempach nie szkodzi, ale w R1 stawki przypisano świadomie (patrz komentarz w kodzie).
