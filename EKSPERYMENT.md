# EKSPERYMENTALNA KARTA PROTOKOŁU R23 — konkretne parametry

*Czy „czas = entropia” da się rozstrzygnąć w laboratorium? Tak — przez zimne
atomy, kolektywną emisję i pomiar korelacji. Poniżej kompletny arkusz:
platformy, liczby, sekwencja, budżet fotonów, moc statystyczna, werdykt.*

---

## 1. Co mierzymy (przypomnienie protokołu R23)

Przygotowujemy stan „jasny + ciemny” (jak |10⟩ w modelu: superpozycja sektora
promieniującego i subradiacyjnego) w układzie N atomów wspólnie sprzężonych ze
światłem. Faza jasna emituje fotony (superradiancja / wzmocnione emisja);
po ostatnim fotografie wchodzimy w **fazę ciemną** (subradiancja). W fazie
ciemnej mierzymy tempo „zegara entropii” τ̇:

| Teoria | Przewidywanie w fazie ciemnej |
|---|---|
| **T1** (czas = produkcja entropii) | τ̇ = 0 — zegar staje po wygaśnięciu fluorescencji |
| **T2** (czas = istnienie informacji) | τ̇ = η·I_eq ≠ 0 — zegar tyka dalej, napędzany korelacją |

**Kluczowa uwaga (uczciwa):** kanał *fotonowy* NIE rozróżnia T1 od T2 — zegar
w modelu nie sprzęga się zwrotnie z systemem, więc fluorescencja jest taka sama
w obu teoriach. Rozstrzyga **kanał korelacyjny**: pomiar I(A:B) (informacja
wzajemna między dwiema połówkami chmury) w funkcji czasu w fazie ciemnej.
T1: I(t) = const (zegar stoi); T2: I(t) generuje czas (τ̇ = η·I_eq).

---

## 2. Platformy

| | A: wolna przestrzeń ⁸⁷Rb | B: nanofiber ¹³³Cs | C: wnęka optyczna |
|---|---|---|---|
| τ_nat | 26.2 ns (γ = 2π·6.1 MHz) | 30.5 ns (γ = 2π·5.2 MHz) | 26.2 ns |
| N (atomy) | 10⁵ | 5×10³ | 50 |
| sprzężenie | OD ≈ 100 (chmura) | β ≈ 0.15 (mod prowadzony) | g ≫ γ,κ |
| t_B (jasny, 1 ekscyton) | ~13 ns | ~7.6 ns | ~13 ns |
| t_D (subradiancja) | 2.6 μs | **179 μs** | 1.3 μs |
| t_D/t_B | 2×10⁵ | **2.4×10⁴** | 100 |
| referencja | Guerin PRL 116, 083601 | Pennetta PRL 128, 203601 | Dicke, klasyka |

> **Uwaga:** superradiancja (N-krotne wzmocnienie) wymaga *wielu* ekscytonów.
> W sektorze pojedynczego ekscytonu jasny rozpad ≈ γ (ew. wzmocnienie β do
> modu), a subradiancja daje Γ_D = γ/OD (A) lub γ(1−β)/N (B). To są liczby
> użyte w tabeli.

**Najlepsza platforma: B** — t_D = 179 μs daje najdłuższe okno fazy ciemnej
(pomiar I(A:B) w wielu punktach czasowych).

---

## 3. Mapowanie jednostek model → eksperyment

| Wielkość modelu | Wartość | Jednostka eksperymentalna |
|---|---|---|
| 1 tyknięcie (τ) | — | Δt_samp = 1 μs (próbkowanie fazy ciemnej) |
| σ₀ (referencja tempa) | 0.01 nat/tyk | 10⁴ nat/s |
| δs (kwant entropii) | 0.01 nat | 1.4% bitu |
| I_eq (pamięć) | 0.1438 nat | 0.207 bitu |
| τ̇_T1 | 0 | 0 |
| τ̇_T2 (η = 0.5) | 0.0719 nat/tyk | 7.2×10⁴ nat/s |
| γ_B (model) | 0.02 1/τ | ~1/τ_nat (tempo dyssypacji) |

---

## 4. Sekwencja eksperymentalna (platforma B)

| Krok | Co | Czas |
|---|---|---|
| 1 | Pułapka magnetooptyczna + chłodzenie | 10–100 ms |
| 2 | Przygotowanie stanu |10⟩-typ: π-puls / słaba wiązka, sektor 1 ekscytonu, stan Dicke z fazą | 10–100 ns |
| 3 | **Faza jasna**: wzmocniona emisja, detekcja fotonów (SPCM) | ~8 ns |
| 4 | Ostatni foton ⇒ znacznik t* | t* ≈ t_B |
| 5 | **Faza ciemna**: subradiancja Γ_D + pomiar I(A:B) w punktach t*+{0.1, 1, 10, 100}·τ_sub | do 179 μs |
| 6 | Odczyt korelacji: rozdzielić chmurę na A|B, obrazować fluorescencję, estymować S_A, S_B, S_AB | 1–10 μs |
| 7 | Powtórzenie: M ≈ 150 realizacji na punkt (precyzja σ_I = 0.01 nat) | ~0.4 ms/punkt |

Całkowity czas pomiaru: **~2–5 ms** (wszystkie punkty fazy ciemnej) + narzut
przygotowania — **minuty włącznie z powtórzeniami**.

---

## 5. Budżet fotonów i detekcja

SPCM: wydajność η_det = 0.3, dark counts = 100 Hz, rozdzielczość czasowa ~1 ns.

| Platforma | Fotony subrad. w 1 ms | Dark counts | SNR |
|---|---|---|---|
| A | 382 | 0.1 | 19.5 |
| B | 5.6 | 0.1 | 2.3 |
| C | 763 | 0.1 | 27.6 |

Kanał fotonowy (fluorescencja/subradiancja) potwierdza **separację jasny/ciemny**
(SNR > 2 nawet dla B) — to samo w sobie jest testem subradiancji (R17/R21),
ale nie rozróżnia T1/T2.

---

## 6. Moc statystyczna kanału korelacyjnego

Rozstrzygnięcie T1 vs T2 wymaga zmierzenia I(A:B) w fazie ciemnej z precyzją
σ_I ≤ η·I_eq/3 ≈ 0.01 nat (3σ separacja od 0). I(A:B) ≈ 2·h₂(p̂) (dla czystego
stanu, S_AB ≈ 0), p̂ = P(ekscyton w połowie A) ≈ 0.5. Czułość h₂ przy p = 0.5
jest **kwadratowa**:

- σ_p = √(p(1−p)/M) = √(0.25/M)
- σ_I = 2·|Δh₂| = 0.01 nat ⇒ **M = 150 realizacji na punkt czasowy**
- czas/punkt ≈ 150 × 200 μs = 30 ms; 6 punktów ≈ 0.2 s

Alternatywnie (mocniejszy projekt): użyć p̂ ≠ 0.5 (asymetryczny podział A:B),
gdzie czułość liniowa — M spada do ~50.

---

## 7. Wymagania i ryzyka

**Wymagane:**
- N ≥ 5×10³ (platforma B), η_det ≥ 0.3, dark counts < 1 kHz
- Kontrola fazy stanu Dicke (przygotowanie |10⟩-typ z wiernością > 90%)
- Obrazowanie fluorescencji z podziałem chmury A|B (istniejąca technika:
  absorpcja/fluorescencja w pułapce dipolowej)

**Ryzyka:**
1. Szum odczytu I(A:B) > 0.01 nat przy małym N → zwiększyć M lub N.
2. Nieidealna ciemność (Γ_D > 0) daje tło fotonów — nie mylić z sygnałem T2
   (tło znane z pomiaru Γ_D; T2 to *nadmiar* tempa zegara).
3. Dekoherencja stanu w fazie ciemnej (rozpady, kolizje) — czasy < t_D.
4. Interpretacja: T2 wymaga, by *korelacja* (nie produkcja entropii) napędzała
   zegar — to teoretyczna definicja czasu; eksperyment mierzy, czy τ̇ po
   wygaśnięciu Γ jest zerowe (T1) czy nie (T2).

---

## 8. Werdykt wykonalności

| Element | Status |
|---|---|
| Przygotowanie stanów Dicke (jasny/ciemny) | ✅ osiągalne (literatura: superradiancja, subradiancja) |
| Detekcja fotonów, faza jasna/ciemna | ✅ osiągalne (SPCM, t_B ns / t_D μs) |
| Pomiar S(t) przez statystykę zespołów | ✅ osiągalne (tomografia/obrazowanie) |
| Pomiar I(A:B) z precyzją 0.01 nat | ✅ osiągalne (M = 150 realizacji/punkt) |
| Rozstrzygnięcie T1 vs T2 | 🟡 wymaga kanału korelacyjnego + kontroli fazy; wykonalne w ~minutę pomiaru |
| Nowe predykcje modelu (T_max, ω_c∝T) | ❓ wymagają innej geometrii (koszt zegara) |

**Wniosek:** protokół R23 jest **wykonalny na istniejącej technologii** zimnych
atomów (nanofiber + SPCM + obrazowanie). Kanał fotonowy daje test subradiancji
i pamięci (R17/R21/R24); kanał korelacyjny daje pierwszy *laboratoryjny test*
tezy „czas = entropia" — rozstrzygnięcie T1 (zegar staje) vs T2 (korelacja
napędza czas). Cały pomiar: minuty.

---

## Źródła parametrów

- Guerin W., Araújo M.O., Kaiser R., *PRL* 116, 083601 (2016) — subradiancja
  w chmurze zimnych atomów (czasy do 100× τ_nat).
- Pennetta R. et al., *PRL* 128, 203601 (2022) — sprzężenie super/subradiantne,
  stany z wyłączeniami emisji na nanofibrze.
- τ_nat: ⁸⁷Rb D2 26.2 ns; ¹³³Cs D2 30.5 ns (standardowe wartości).
- Parametry modelu: `entropia/experyment.py` (odtwarzalne liczby).
