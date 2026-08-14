#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buduje samodzielny raport HTML (raport.html) dla kosmologicznego modelu
«ENTROPIA»: wbudowuje wszystkie figury (base64) oraz interaktywną symulację
zegara kosmicznego (czysty JS, canvas — bez zewnętrznych zależności).

Uruchomienie:  python3 zrob_raport.py   (po uruchomieniu model_entropia.py)
"""

import base64
import json
import os

import numpy as np

from . import core as M
from . import extensions as R
from predykcje import PREDYKCJE, figura_predykcje
from . import e11 as E11
from . import e12 as E12
from . import e13 as E13
from . import e14 as E14
from . import e15 as E15
from . import experyment as EXP
from . import e16 as E16
from . import e17 as E17
from . import e18 as E18
from . import e19 as E19
from . import e20 as E20
from . import e21 as E21
from . import e22 as E22
from . import e23 as E23

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figury")


def b64img(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def dane_rozszerzen():
    """Liczby kluczowe rozszerzeń (R1–R3) dla raportu."""
    d = {}

    # ---- R1 ----
    etas = [1.0, 0.5, 0.25, 0.1, 0.01]
    d["R1"] = []
    for eta in etas:
        Seq = R.S_eq_termiczna(eta)
        d["R1"].append(dict(
            eta=eta, beta_omega=-np.log(eta),
            seq=Seq, pur=R.czystosc_równowagowa(eta), req=R.r_eq_termiczny(eta),
            t99a=R.czas_do_poziomu_T(M.GAMMA_A, eta, 0.99 * Seq),
            t99b=R.czas_do_poziomu_T(M.GAMMA_B, eta, 0.99 * Seq)))
    # overshoot η=0.2
    t = np.linspace(0, 60, 20000)
    St = R.S_termiczna_analitycznie(M.GAMMA_B, 0.2, t)
    imax = int(np.argmax(St))
    d["R1_overshoot"] = dict(smax=St[imax], tmax=t[imax],
                             seq=R.S_eq_termiczna(0.2),
                             delta=St[imax] - R.S_eq_termiczna(0.2))

    # ---- R2 ----
    ket1 = np.array([0.0, 1.0]); ket0 = np.array([1.0, 0.0])
    stan11 = R.stan_poczatkowy_N([ket1, ket1])
    stan10 = R.stan_poczatkowy_N([ket1, ket0])
    n_l = 6000
    S11, P11, C11, MI11 = R.symuluj_wspolne(M.GAMMA_B, stan11, N=2, gamma_phi=0.0, n=n_l)
    S10, P10, C10, MI10, rho10 = R.symuluj_wspolne(M.GAMMA_B, stan10, N=2, gamma_phi=0.0,
                                                   n=n_l, zwroc_stan=True)
    S_nz, _ = R.symuluj_niezalezne(M.GAMMA_B, 2, gamma_phi=0.0, n=n_l)
    d["R2"] = dict(
        ln3=np.log(3), ln12_2=np.log(12) / 2, ln2_2=2 * M.LN2,
        ln43=np.log(4 / 3), ln2_sqrt3=np.log(2 / np.sqrt(3)),
        s_ind=S_nz[-1], p_ind=0.25,
        s11=S11[-1], p11=P11[-1], mi11=MI11[-1], t90_11=R.czasy_90(S11, np.log(3)),
        s10=S10[-1], p10=P10[-1], mi10=MI10[-1], neg10=R.negatywnosc2(rho10),
        t90_10=R.czasy_90(S10, np.log(12) / 2), t90_ind=R.czasy_90(S_nz, 2 * M.LN2),
    )

    # ---- R3 ----
    scen = {
        "stały": R.FB_STALY,
        "chłodzenie α=2": lambda u: R.FB_CHLODZENIE(u, 2.0),
        "przyspieszanie α=1": lambda u: R.FB_PRZYSPIESZANIE(u, 1.0),
    }
    n_fb = 600
    d["R3"] = []
    for naz, fb in scen.items():
        S_B, _, _ = R.symuluj_feedback(M.GAMMA_B, fb, n=n_fb)
        tA = R.t_do_polowy_ciggly(M.GAMMA_A, fb)
        tB = R.t_do_polowy_ciggly(M.GAMMA_B, fb)
        dS = np.zeros_like(S_B); dS[1:] = np.maximum(S_B[1:] - S_B[:-1], 0)
        _, dt, _ = M.zegar_stochastyczny(dS, seed=11)
        lo, hi = 40, 200
        nz = int(np.sum(dt[lo:hi] == 0))
        maks = dl = 0
        for i in range(lo, hi):
            if dt[i] == 0:
                dl += 1; maks = max(maks, dl)
            else:
                dl = 0
        d["R3"].append(dict(naz=naz, tA=tA, tB=tB, ratio=tB / tA, nz=nz, maks=maks))
    # kompresja ciągła z chłodzeniem
    fb_c = lambda u: R.FB_CHLODZENIE(u, 2.0)
    t_c, S_A_c = R.symuluj_feedback_ciggly(M.GAMMA_A, fb_c, 2.4, n_out=4000)
    t_Bg, S_B_c = R.symuluj_feedback_ciggly(M.GAMMA_B, fb_c, 70.0, n_out=20000)
    d["R3_comp"] = float(np.max(np.abs(S_A_c - np.interp(27.0 * t_c, t_Bg, S_B_c))))
    # dane dla demo 2: t½(α) ciągłe
    alphas = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    d["R3_demo"] = dict(
        alphas=alphas,
        th_c=[R.t_do_polowy_ciggly(M.GAMMA_B, lambda u, a=a: R.FB_CHLODZENIE(u, a)) for a in alphas],
        th_a=[R.t_do_polowy_ciggly(M.GAMMA_B, lambda u, a=a: R.FB_PRZYSPIESZANIE(u, a)) for a in alphas])

    # ---- R4: N=3 sektory ----
    b, sector = R.baza_N3()
    S111, P111, *_ = R.symuluj_N3(M.GAMMA_B, b["111"], n=6000)
    S100, P100, *_ = R.symuluj_N3(M.GAMMA_B, b["100"], n=6000)
    stan_1S = sector[6][2]
    S1S, P1S, *_ = R.symuluj_N3(M.GAMMA_B, stan_1S, n=6000)
    d["R4"] = dict(
        s111=S111[-1], p111=P111[-1], t90_111=R.czasy_90(S111, np.log(4)),
        s100=S100[-1], p100=P100[-1], t90_100=R.czasy_90(S100, np.log(108) / 3),
        s1S=S1S[-1], p1S=P1S[-1], t90_1S=R.czasy_90(S1S, np.log(2)),
        ln4=np.log(4), ln108_3=np.log(108) / 3, ln2=M.LN2, ln8=3 * M.LN2,
        deficit_32=3 * M.LN2 - np.log(4),
        popB_min=1.0,
    )
    # populacja na kopii B (min) — policz na krótkim przebiegu
    from scipy.linalg import expm as _expm
    S_z, S_p, S_m, _ = R.macierze_kolektywne(3)
    H = (M.OMEGA / 2.0) * S_z
    Lj = R.superoperator_z_jumpami(H, [S_p, S_m], [M.GAMMA_B, M.GAMMA_B])
    Uj = _expm(Lj * M.DELTA_TAU)
    PB = np.outer(sector[6][2], sector[6][2].conj()) + np.outer(sector[7][2], sector[7][2].conj())
    rho = np.outer(stan_1S, stan_1S.conj())
    popmin = 1.0
    for _i in range(M.N_TICKS):
        popmin = min(popmin, np.real(np.trace(PB @ rho)))
        rho = R.unvecR(Uj @ R.vecR(rho), 8)
    d["R4"]["popB_min"] = float(popmin)

    # ---- R5: entropia makro ----
    wyn = {}
    for N in [1, 2, 3, 4]:
        wyn[N] = R.entropia_makro(N)
    d["R5"] = {N: dict(ind=wyn[N][0], kol=wyn[N][1], pkol=wyn[N][2],
                       t90i=wyn[N][3], t90k=wyn[N][4],
                       ind_an=N * M.LN2, kol_an=np.log(N + 1),
                       deficit=N * M.LN2 - np.log(N + 1)) for N in wyn}

    # ---- R6: gorący Wielki Wybuch ----
    ETA0, ETAB = 0.95, 0.15
    S_c, G_c, T_c, S0, Seq = R.symuluj_wielki_wybuch(M.GAMMA_B, ETA0, ETAB, R.FB_STALY)
    S_f, G_f, T_f, _, _ = R.symuluj_wielki_wybuch(
        M.GAMMA_B, ETA0, ETAB, lambda u: R.FB_CHLODZENIE(u, 2.0))
    T_w, dt_w, _ = R.zegar_wstecz(S_c, seed=7)
    lo, hi = 200, 500
    d["R6"] = dict(
        eta0=ETA0, etaB=ETAB, S0=S0, Seq=Seq, budzet=S0 - Seq,
        t_half_c=R.t_do_polowy_wstecz(S_c, M.DELTA_TAU, S0, Seq),
        t_half_f=R.t_do_polowy_wstecz(S_f, M.DELTA_TAU, S0, Seq),
        nz=int(np.sum(dt_w[lo:hi] == 0)), n_total=hi - lo,
        S_c=[round(float(x), 6) for x in S_c.tolist()],
    )

    # ---- R7: losowe stany (cache z figura_R7) ----
    d7 = R.CACHE.get("R7") or R.figura_R7()
    d["R7"] = d7

    # ---- R8: cykl (cache z figura_R8) ----
    d8 = R.CACHE.get("R8") or R.figura_R8()
    S_c8 = d8["S"]; dS8 = d8["dS"]; imin = d8["imin"]
    rng8 = np.random.default_rng(5)
    zamroz = sum(1 for i in range(max(0, imin - 15), min(len(dS8), imin + 15))
                 if rng8.poisson(abs(dS8[i]) / M.DELTA_S_Q) == 0)
    d["R8"] = dict(
        S0=d8["S0"], Smin=d8["Smin"], imin=d8["imin"], budzet=d8["budzet"],
        t_abs_total=d8["t_abs_total"], frakcja_wstecz=float(np.mean(dS8[1:] < 0)),
        zamroz=zamroz, n_cyc=d8["n_cyc"],
        S=[round(float(x), 6) for x in S_c8.tolist()],
    )

    # ---- R9: kwantowy zegar (cache z figura_R9) ----
    d9 = R.CACHE.get("R9") or R.figura_R9()
    z = d9["z"]
    d["R9"] = dict(
        gt=d9["gt_main"], S_end=z["S_sys"][-1], dev=z["S_sys"][-1] - M.LN2,
        nbar_end=z["nbar"][-1], dn_end=z["dn"][-1],
        rel_end=z["dn"][-1] / max(z["nbar"][-1], 1e-9),
        I_end=z["I"][-1],
        gts=[float(g) for g in d9["gts"]],
        devs=[float(v) for v in d9["dev"]],
        rels=[float(v) for v in d9["relend"]],
        pn50=[round(float(x), 6) for x in z["pn"][50].tolist()],
        pn150=[round(float(x), 6) for x in z["pn"][150].tolist()],
        pn_end=[round(float(x), 6) for x in z["pn"][-1].tolist()],
    )

    # ---- R10: kwantowy zegar z koherencjami ----
    d10 = R.CACHE.get("R10") or R.figura_R10()
    d["R10"] = d10

    # ---- R11: odblokowanie γφ ----
    d11 = R.CACHE.get("R11") or R.figura_R11()
    d["R11"] = d11

    # ---- R13: grawitacyjna produkcja entropii ----
    d13 = R.CACHE.get("R13") or R.figura_R13()
    gammas13 = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    sig13 = []
    for gg in gammas13:
        _, sr, sg = R.symuluj_dwie_kapiele(d13["gamma_r"], d13["eta_r"], gg, d13["eta_g"], n=250)
        sig13.append((sr + sg)[-1])
    d["R13"] = dict(ness=d13, sig_gamma=list(zip(gammas13, sig13)))

    # ---- R15: strażnik historii ----
    d15 = R.CACHE.get("R15") or R.figura_R15()
    d["R15"] = d15

    # ---- R16: formalizm relacyjny (rewizja) ----
    d16 = R.CACHE.get("R16") or R.figura_R16()
    d["R16"] = d16

    # ---- R17: test laboratoryjny ----
    d17 = R.CACHE.get("R17") or R.figura_R17()
    d["R17"] = d17

    # ---- R18: ENTROPIA-1.1 (N = 2..100, baza Dickego) ----
    d11 = E11.main()          # regeneruje figury E1–E4, zwraca liczby
    d["E11"] = d11

    # ---- R19: ENTROPIA-1.2 (konkurencja funkcjonałów itd.) ----
    d12 = E12.main()          # figury E5–E7
    d["E12"] = d12

    # ---- R20: ENTROPIA-1.3 (I_c, koszt zegara, protokół) ----
    d13 = E13.main()          # figury E8–E10
    d["E13"] = d13

    # ---- R21–R23: ENTROPIA-1.4 (fidelity, ω_c(T), protokół e2e) ----
    d14 = E14.main()          # figury E11–E13
    d["E14"] = d14

    # ---- R24–R26: ENTROPIA-1.5 (pamięć operacyjna, ω_c(T), SPRT) ----
    d15 = E15.main()          # figury E14–E16
    d["E15"] = d15

    # ---- R27: eksperymentalna karta protokołu R23 ----
    dExp = EXP.main()         # figura E17 + liczby
    d["EXP"] = dExp

    # ---- R28–R30: ENTROPIA-1.6 (suchy bieg, Petz, zimny zegar) ----
    d16 = E16.main()          # figury E18–E19
    d["E16"] = d16

    # ---- R31–R33: ENTROPIA-1.7 (arkusz T_max, energia, suchy bieg + F) ----
    d17 = E17.main()          # figury E20–E22
    d["E17"] = d17

    # ---- R34–R36: ENTROPIA-1.8 (realizacja Petza, CMB, sieć) ----
    d18 = E18.main()          # figury E23–E25
    d["E18"] = d18

    # ---- R37–R38: ENTROPIA-1.9 (protokół różnicowy, CMB ewolucja) ----
    d19 = E19.main()          # figury E26–E27
    d["E19"] = d19

    # ---- R40–R41: ENTROPIA-2.0 (sieć z η(T), asymptotyka Petza) ----
    d20 = E20.main()          # figury E28–E30
    d["E20"] = d20

    # ---- R42–R43: ENTROPIA-2.1 (formalny Petz, entrainment) ----
    d21 = E21.main()          # figury E31–E33
    d["E21"] = d21

    # ---- R44–R45: ENTROPIA-3.0 (drabina Dickego, FRW) ----
    d22 = E22.main()          # figury E34–E35
    d["E22"] = d22

    # ---- R46: dowód wzoru Petza z regularyzacją ----
    d23 = E23.main()          # figura E36
    d["E23"] = d23

    # ---- R48–R49: ENTROPIA-4.0 (dwie komórki, siła, FRW) ----
    from . import e24 as E24
    d24 = E24.main()          # figury E37–E40
    d["E24"] = d24

    # ---- R50: ENTROPIA-5.0 (pętla pomiarowa na procesorze kwantowym) ----
    from . import e25 as E25
    d25 = E25.main()          # figury E41–E43
    d["E25"] = d25

    # ---- R14: predykcje vs obserwacje ----
    figura_predykcje()
    d["R14_rows"] = "".join(
        f"<tr><td><b>{pid}</b></td><td>{pred}</td><td>{model}</td>"
        f"<td>{obs}</td><td style='text-align:center'>{oc}</td></tr>"
        for pid, pred, model, obs, oc in PREDYKCJE)
    d["R14_p1"] = "0.7138"
    d["R14_neff"] = "2.99 ± 0.17"
    d["R14_sratio"] = "2.8×10²⁸"
    d["R14_frac"] = "1.2×10⁻¹⁸"
    d["R14_sperb"] = "1.15×10¹⁰"
    d["R14_darkfrac"] = "50.0"
    d["R14_nufrac"] = "49"
    return d


def main():
    d = M.liczby_kluczowe()
    S_A, S_B = d["S_A"], d["S_B"]
    dS_A, dS_B = d["dS_A"], d["dS_B"]

    # dane dla interaktywnej symulacji
    js_A = json.dumps([round(float(x), 6) for x in dS_A.tolist()])
    js_B = json.dumps([round(float(x), 6) for x in dS_B.tolist()])

    # statystyki czkania
    _, dt_B, _ = M.zegar_stochastyczny(dS_B, seed=11)
    _, dt_A, _ = M.zegar_stochastyczny(dS_A, seed=11)
    lo, hi = 40, 200
    n_zero_B = int(np.sum(dt_B[lo:hi] == 0))
    maks_B = dl = 0
    for i in range(lo, hi):
        if dt_B[i] == 0:
            dl += 1
            maks_B = max(maks_B, dl)
        else:
            dl = 0
    n_zero_A = int(np.sum(dt_A[3:30] == 0))
    last_A = int(np.max(np.nonzero(dt_A)[0]))

    r = 0.3466  # ln2/2 (zaokrąglone do wyświetlania)

    img = {k: b64img(k + ".png") for k in
           ["fig1_entropia", "fig2_produkcja", "fig3_czas_entropia",
            "fig4_czkanie", "fig5_stosunek", "fig6_dekoherencja", "fig7_bloch"]}

    params = [
        ("γ_B (tempo relaksacji, zimne)", f"{M.GAMMA_B:.3f} [1/j. czasu]"),
        ("γ_A = 27·γ_B (gorące)", f"{M.GAMMA_A:.3f} [1/j. czasu]"),
        ("T_A / T_B", f"{M.T_RATIO:.0f} (stąd {M.T_RATIO**3:.0f} = 3³)"),
        ("γ_φ / γ (dekoherencja czysta)", f"{M.GAMMA_PHI:.0f}"),
        ("Ω (precesja, H = Ω/2·σ_z)", f"{M.OMEGA:.1f}"),
        ("τ — mikro-tyknięcie", f"{M.DELTA_TAU:.2f}"),
        ("liczba tyknięć N", f"{M.N_TICKS}"),
        ("kwant entropii δs („bit”)", f"{M.DELTA_S_Q:.2f} [nat]"),
        ("κ — czas = entropia", f"{M.KAPPA:.0f}"),
        ("stan początkowy |ψ⟩", "θ = 60°, φ = 45° (czysty, |r| = 1)"),
        ("stan równowagi", "ρ_eq = ½·𝟙  (kąpiel nieskończenie gorąca)"),
        ("S(∞) = ln 2", f"{M.LN2:.6f} [nat]"),
    ]

    html = TEMPLATE
    html = html.replace("@@JS_A@@", js_A)
    html = html.replace("@@JS_B@@", js_B)
    html = html.replace("@@FIG1@@", img["fig1_entropia"])
    html = html.replace("@@FIG2@@", img["fig2_produkcja"])
    html = html.replace("@@FIG3@@", img["fig3_czas_entropia"])
    html = html.replace("@@FIG4@@", img["fig4_czkanie"])
    html = html.replace("@@FIG5@@", img["fig5_stosunek"])
    html = html.replace("@@FIG6@@", img["fig6_dekoherencja"])
    html = html.replace("@@FIG7@@", img["fig7_bloch"])

    for key, val in {
        "GAMMA_B": f"{M.GAMMA_B:.3f}",
        "GAMMA_A": f"{M.GAMMA_A:.3f}",
        "T_RATIO": f"{M.T_RATIO:.0f}",
        "TAU": f"{M.DELTA_TAU:.2f}",
        "N_TICKS": f"{M.N_TICKS}",
        "LN2": f"{M.LN2:.6f}",
        "TA_HALF": f"{d['tA_half']:.4f}",
        "TB_HALF": f"{d['tB_half']:.4f}",
        "RATIO_HALF": f"{d['tB_half'] / d['tA_half']:.1f}",
        "DS1A": f"{d['dS_A'][1]:.4f}",
        "DS1B": f"{d['dS_B'][1]:.4f}",
        "DS1RATIO": f"{d['dS_A'][1] / d['dS_B'][1]:.1f}",
        "RATE0": f"{d['stosunki'][0]:.3f}",
        "RATE2": f"{d['stosunki'][2]:.3f}",
        "RATE5": f"{d['stosunki'][5]:.3f}",
        "COMP_ERR": f"{np.max(np.abs(S_A - S_B[np.clip((np.arange(M.N_TICKS) * 27).astype(int), 0, M.N_TICKS - 1)])):.1e}",
        "P_END_A": f"{d['P_A'][-1]:.4f}",
        "P_END_B": f"{d['P_B'][-1]:.4f}",
        "R_END_B": f"{np.linalg.norm(d['R_B'][-1]):.4f}",
        "R_END_A": f"{np.linalg.norm(d['R_A'][-1]):.4f}",
        "N_ZERO_B": f"{n_zero_B}",
        "MAX_FREEZE_B": f"{maks_B}",
        "N_ZERO_A": f"{n_zero_A}",
        "LAST_A": f"{last_A}",
        "DS_Q": f"{M.DELTA_S_Q:.2f}",
        "KAPPA": f"{M.KAPPA:.0f}",
    }.items():
        html = html.replace("@@" + key + "@@", val)

    rows = "".join(
        f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in params)
    html = html.replace("@@PARAMS@@", rows)

    # ------------------- ROZSZERZENIA (część II) -------------------
    R.generuj_figury()
    ext = dane_rozszerzen()

    imgR1 = b64img("figR1_temperatura.png")
    imgR2 = b64img("figR2_kubity.png")
    imgR3 = b64img("figR3_feedback.png")
    imgR4 = b64img("figR4_sektory.png")
    imgR5 = b64img("figR5_makro.png")
    imgR6 = b64img("figR6_wielkiwybuch.png")
    imgR7 = b64img("figR7_losowe.png")
    imgR8 = b64img("figR8_cykl.png")
    imgR9 = b64img("figR9_zegarkwantowy.png")
    imgR10 = b64img("figR10_zegar_koherencje.png")
    imgR11 = b64img("figR11_gphi_sweep.png")
    imgR13 = b64img("figR13_grawitacja.png")
    imgR14 = b64img("figR14_predykcje.png")
    imgR15 = b64img("figR15_straznik.png")
    imgR16 = b64img("figR16_relacyjny.png")
    imgR17 = b64img("figR17_test_lab.png")
    imgE1 = b64img("figE1_dynamika.png")
    imgE2 = b64img("figE2_skalowanie.png")
    imgE3 = b64img("figE3_testy.png")
    imgE4 = b64img("figE4_pamiec.png")
    imgE5 = b64img("figE5_funkcjonaly.png")
    imgE6 = b64img("figE6_odzyskiwalnosc.png")
    imgE7 = b64img("figE7_27fizyczny.png")
    imgE8 = b64img("figE8_koherentna.png")
    imgE9 = b64img("figE9_koszt.png")
    imgE10 = b64img("figE10_protokol.png")
    imgE11 = b64img("figE11_fidelity.png")
    imgE12 = b64img("figE12_omega_T.png")
    imgE13 = b64img("figE13_protokol_e2e.png")
    imgE14 = b64img("figE14_pamiec_op.png")
    imgE15 = b64img("figE15_omega_okno.png")
    imgE16 = b64img("figE16_sprt.png")
    imgE17 = b64img("figE17_sekwencja.png")
    imgE18 = b64img("figE18_petz.png")
    imgE19 = b64img("figE19_zimny_zegar.png")
    imgE20 = b64img("figE20_arkusz.png")
    imgE21 = b64img("figE21_energia.png")
    imgE22 = b64img("figE22_wiernosc.png")
    imgE23 = b64img("figE23_petz_realizacja.png")
    imgE24 = b64img("figE24_zegar_cmb.png")
    imgE25 = b64img("figE25_siec.png")
    imgE26 = b64img("figE26_roznicowy.png")
    imgE27 = b64img("figE27_cmb_ewolucja.png")
    imgE28 = b64img("figE28_siec_cykl.png")
    imgE29 = b64img("figE29_petz_lim.png")
    imgE30 = b64img("figE30_petz_dynamika.png")
    imgE31 = b64img("figE31_petz_formalny.png")
    imgE32 = b64img("figE32_petz_Ninf.png")
    imgE33 = b64img("figE33_entrainment.png")
    imgE34 = b64img("figE34_drabina.png")
    imgE35 = b64img("figE35_frw.png")
    imgE36 = b64img("figE36_dowod_petz.png")
    imgA3 = b64img("figA3_27_poprawka.png")
    imgE37 = b64img("figE37_dwie_komorki.png")
    imgE38 = b64img("figE38_ness.png")
    imgE39 = b64img("figE39_entropowa_sila.png")
    imgE40 = b64img("figE40_frw.png")
    imgE41 = b64img("figE41_przetrwanie.png")
    imgE42 = b64img("figE42_tomografia.png")
    imgE43 = b64img("figE43_hardware.png")

    r1rows = "".join(
        f"<tr><td>η = {r['eta']:.2g}</td><td>{r['beta_omega']:.2f}</td>"
        f"<td>{r['seq']:.4f}</td><td>{r['pur']:.4f}</td><td>{r['req']:.4f}</td>"
        f"<td>{r['t99b']:.2f}</td><td>{r['t99a']:.2f}</td><td>27.0</td></tr>"
        for r in ext["R1"])
    r2rows = "".join(f"<tr>{c}</tr>" for c in [
        f"<td>2 niezależne kąpiele, start |11⟩</td><td>{ext['R2']['s_ind']:.4f} = 2·ln 2</td>"
        f"<td>{ext['R2']['p_ind']:.4f}</td><td>0.0000</td><td>0.0000</td>"
        f"<td>{ext['R2']['t90_ind']}</td>",
        f"<td>wspólna kąpiel (kolektywna), start |11⟩</td><td>{ext['R2']['s11']:.4f} = ln 3</td>"
        f"<td>{ext['R2']['p11']:.4f}</td><td>0.0000</td><td>{ext['R2']['mi11']:.4f} = ln(4/3)</td>"
        f"<td>{ext['R2']['t90_11']}</td>",
        f"<td>wspólna kąpiel (kolektywna), start |10⟩</td><td>{ext['R2']['s10']:.4f} = ½·ln 12</td>"
        f"<td>{ext['R2']['p10']:.4f}</td><td>{ext['R2']['neg10']:.4f}</td>"
        f"<td>{ext['R2']['mi10']:.4f} = ln(2/√3)</td><td>{ext['R2']['t90_10']}</td>"])
    r3rows = "".join(
        f"<tr><td>{r['naz']}</td><td>{r['tA']:.2f}</td><td>{r['tB']:.2f}</td>"
        f"<td>{r['ratio']:.1f}</td><td>{r['nz']}/160</td><td>{r['maks']}</td></tr>"
        for r in ext["R3"])

    js_alphas = json.dumps([float(a) for a in ext["R3_demo"]["alphas"]])
    js_th_c = json.dumps([round(float(x), 4) for x in ext["R3_demo"]["th_c"]])
    js_th_a = json.dumps([round(float(x), 4) for x in ext["R3_demo"]["th_a"]])
    js_SC = json.dumps(ext["R6"]["S_c"])          # malejąca entropia (gorący WB)

    R4 = ext["R4"]; R5 = ext["R5"]; R6 = ext["R6"]
    r4rows = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>"
        for a, b, c, d, e in [
            ("|111⟩ — sektor j=3/2 (symetryczny)", f"{R4['s111']:.4f} = ln 4",
             f"{R4['p111']:.4f}", "—", f"{R4['t90_111']}"),
            ("|100⟩ — mieszanina sektorów", f"{R4['s100']:.4f} = ln108/3",
             f"{R4['p100']:.4f}", "—", f"{R4['t90_100']}"),
            ("|1⟩⊗|S⟩₂₃ — czysty j=1/2 (kopia B)", f"{R4['s1S']:.4f} = ln 2",
             f"{R4['p1S']:.4f}", f"{R4['popB_min']:.6f}", f"{R4['t90_1S']}"),
        ])
    r5rows = "".join(
        f"<tr><td>N = {N}</td><td>{R5[N]['ind']:.4f} = {N}·ln 2</td>"
        f"<td>{R5[N]['kol']:.4f} = ln({N + 1})</td><td>{R5[N]['deficit']:.4f}</td>"
        f"<td>{R5[N]['t90i']}</td><td>{R5[N]['t90k']}</td></tr>"
        for N in [1, 2, 3, 4])

    R7 = ext["R7"]; R8 = ext["R8"]; R9 = ext["R9"]
    r7rows = "".join(
        f"<tr><td>{naz}</td><td>{R7['wyn3'][naz]:.4f}</td><td>{R7['wyn3d'][naz]:.4f}</td></tr>"
        for naz in ["|111⟩ (sym.)", "|100⟩ (produkt)", "|1⟩⊗|S⟩ (j=1/2)",
                    "losowy #1", "losowy #2"])
    r7rows4 = "".join(
        f"<tr><td>{naz}</td><td>{s0:.4f}</td><td>{sd:.4f}</td></tr>"
        for naz, (s0, sd) in R7["wyn4"].items())
    r9rows = "".join(
        f"<tr><td>γ_t = {gt:.3f}</td><td>{abs(dv):.4f}</td><td>{rel:.2f}</td></tr>"
        for gt, dv, rel in zip(R9["gts"][1:], R9["devs"][1:], R9["rels"][1:]))

    R10 = ext["R10"]; R11 = ext["R11"]
    sceny10 = R10["sceny"]
    r10rows = "".join(
        f"<tr><td>{naz}</td><td>{s['S_end']:.4f}</td><td>{s['dev']:+.4f}</td>"
        f"<td>{s['nbar']:.2f}</td><td>{s['rel']:.2f}</td>"
        f"<td>{s['coh']:.4f}</td><td>{s['I']:.4f}</td></tr>"
        for naz, s in sceny10.items())
    stim = sceny10["koherentny κ=0"]["dev"] / sceny10["próżnia κ=0"]["dev"]
    r11rows = "".join(
        f"<tr><td>{g:g}</td><td>{R11['tau90_3'][g]:.0f}</td></tr>"
        for g in [1e-4, 1e-3, 1e-2, 0.1, 0.3])
    js_krzywe = json.dumps({g: R11["krzywe_js"][g] for g in R11["krzywe_js"]})
    js_tau3 = json.dumps({f"{g:g}": R11["tau90_3"][g] for g in R11["tau90_3"]})

    R13 = ext["R13"]; N13 = R13["ness"]
    R15 = ext["R15"]
    R16 = ext["R16"]
    R17 = ext["R17"]
    E11_data = ext["E11"]
    E11_SINF_ERR = abs(E11_data["Sinf"][100] - np.log(101))
    E11_27_WORST = max(E11_data["err27"].values())
    E11_PDARK = E11_data["pdark_haar"][100]
    E11_SHAAR = E11_data["shaar_g0"][100] / 100.0
    E12_data = ext["E12"]
    E12_Ieq = E12_data["Ieq"]
    E12_GAIN4 = E12_data["rows"][0]["gain"]
    E12_GAIN100 = E12_data["rows"][2]["gain"]
    E12_R3D = E12_data["rT"]["r_3d"]
    E12_RSINGLE = E12_data["rT"]["r_single"]
    E12_CONV = E12_data["zb"][-1][1]
    E13_data = ext["E13"]
    E13_IC4 = E13_data["Ic4_end"]
    E13_IC10 = E13_data["Ic10"]
    E13_IA = E13_data["Ieq"]
    E13_WCMIN = E13_data["wc_min"]
    _d21 = E13_data["d21"]
    E13_PREC_LO, E13_PREC_HI = _d21["prec"][0], _d21["prec"][-1]
    E13_COST_LO, E13_COST_HI = _d21["cost"][0], _d21["cost"][-1]
    E13_BACK_LO, E13_BACK_HI = _d21["back"][0], _d21["back"][-1]
    E13_TAU1, E13_TAU2 = E13_data["tau1"], E13_data["tau2"]
    E14_data = ext["E14"]
    E14_MF4 = E14_data["Fe"][0.0] and 0.3369          # M_F(dark j=1, t=7.5)
    _mfs = {N: (float(E14.MF_sektora(N / 2.0, -N / 2.0, -N / 2.0 + 1, n=30)[15]),
                float(E14.MF_sektora(1.0, -1.0, 0.0, n=30)[15])) for N in [4, 100]}
    E14_MF4 = _mfs[4][1]; E14_MF100 = _mfs[100][1]
    E14_GAIN100 = _mfs[100][1] / max(_mfs[100][0], 1e-12)
    _fe = {j: E14.F_e_sektora(j, n=30) for j in [0.0, 1.0, 2.0]}
    E14_FE0, E14_FE1, E14_FE2 = _fe[0.0], _fe[1.0], _fe[2.0]
    E14_WC10 = E14.omega_c_T(10.0, 0.01)[0]
    E14_WC100 = E14.omega_c_T(100.0, 0.01)[0]
    E14_RATIO3 = E14_data["ratio3"]
    E14_CAP = M.LN2 / M.DELTA_S_Q
    E14_MOC10 = E14_data["moc_T1_10"]
    E14_MOC_T2 = E14_data["moc_T2_10"]
    E14_MOC_DR = E14_data["moc_T2_010"]
    E14_NLAST = E14_data["n_last"]
    E15_data = ext["E15"]
    E15_CM4, E15_CM100, E15_CMD = E15_data["cm4"], E15_data["cm100"], E15_data["cmd"]
    E15_CMD15 = E15_data["cm_d15"]
    E15_CMH = E15.pamiec_operacyjna(0.5, -0.5, 0.5, n=60)[1][30]
    _tm = E15_data["Tmax"]
    E15_TMAX_1 = _tm[0]["T_max"]; E15_TMAX_2 = _tm[-1]["T_max"]
    E15_E1, E15_E2 = E15_data["E1"], E15_data["E2"]
    E15_EN05 = E15.E_stop_SPRT(0.5, 0.001, 0.5, n_real=100)[0]
    E15_EN01 = E15.E_stop_SPRT(0.1, 0.001, 0.1, n_real=100)[0]
    EXP_data = ext["EXP"]
    _pb = EXP_data["platformy"][1]
    EXP_TB_B = _pb["tB"] * 1e9
    EXP_TD_B = _pb["tD"] * 1e6
    EXP_TDTB = _pb["tD"] / _pb["tB"]
    EXP_SNR_A = EXP_data["detekcja"]["A"]["snr"]
    EXP_SNR_B = EXP_data["detekcja"]["B"]["snr"]
    EXP_SNR_C = EXP_data["detekcja"]["C"]["snr"]
    EXP_M = EXP_data["moc"]["M_min"]
    EXP_SIGI = EXP_data["moc"]["sigma_I"]
    EXP_TAU2 = EXP_data["mapa"]["tau_dot_T2_nat_s"]
    EXP_TAU1 = EXP_data["mapa"]["tau_dot_T1_nat_s"]
    E16_data = ext["E16"]
    E16_FREC_05 = E16_data["Frec"][0.5]
    E16_FREC_1 = E16_data["Frec"][1.0]
    E16_FREC_3 = E16_data["Frec"][3.0]
    E16_TM3 = E16_data["Tm3"]
    E16_TOHM = E16.T_max_widmo("ohmic", 0.03, wcut=50.0)[0]
    _kos = E16_data["kosm"]
    E16_CMB = _kos[0]["wc"]; E16_REC = _kos[1]["wc"]
    E16_BBN = _kos[2]["wc"]; E16_EW = _kos[3]["wc"]
    E16_TLAST = E16_data["sb_B"]["t_last"]
    E16_TTOT = E16_data["sb_B"]["T_total"] * 1e6
    E17_data = ext["E17"]
    E17_TMAX6 = E17_data["Tmax6"]
    E17_TMAX30 = E17_data["arkusz"][2]["T_max"]
    E17_T1P = E17_data["T1P"]
    E17_ECLK = E17_data["koszt"]["E_clk"]
    E17_ETRAP = E17_data["koszt"]["E_trap"] * 1e3
    E17_ELAND = E17_data["koszt"]["E_landauer"]
    E17_DEDT = E17_data["koszt"]["dE_dtau"]
    E17_IEQ1 = E17_data["moc"][0]["I_eq"]
    E17_IEQ3 = E17_data["moc"][-1]["I_eq"]
    E17_TAU2_1 = E17_data["moc"][0]["tau2"]
    E17_TAU2_3 = E17_data["moc"][-1]["tau2"]
    E17_P1_3 = E17_data["moc"][-1]["p1"]
    E18_data = ext["E18"]
    E18_P05 = E18_data["odzysk"][0.5]["petz"]; E18_E05 = E18_data["odzysk"][0.5]["echo"]
    E18_P20 = E18_data["odzysk"][2.0]["petz"]; E18_E20 = E18_data["odzysk"][2.0]["echo"]
    E18_WCMIN = E18_data["wcmin_GHz"]
    E18_N100 = E18.nbar(2 * np.pi * 100e9, 2.7255)
    E18_N1T = E18.nbar(2 * np.pi * 1e12, 2.7255)
    E18_S0 = E18_data["siec"][0.0]["sigma_end"]
    E18_S2 = E18_data["siec"][0.2]["sigma_end"]
    E19_data = ext["E19"]
    E19_DR_T1 = E19_data["roznicowy"]["T1"]["mean"]
    E19_DR_T2 = E19_data["roznicowy"]["T2"]["mean"]
    E19_SIG1 = E19.protokol_roznicowy(M_A=1, n_real=80)["T2"]["std"]
    E19_SIG8 = E19.protokol_roznicowy(M_A=8, n_real=80)["T2"]["std"]
    _kiedy = E19_data["kiedy"]
    E19_Z6 = "nigdy" if not _kiedy[6.0]["usable"] else f"{_kiedy[6.0]['z_from']:.0f}"
    E19_Z300 = _kiedy[300.0]["z_from"]
    E19_T1T = _kiedy[1000.0]["t_from"]
    E19_T3T = _kiedy[3000.0]["t_from"]
    E19_T10T = _kiedy[10000.0]["t_from"]
    E20_data = ext["E20"]
    E20_BUDZET = E20_data["siec"][0]["budget"]
    E20_TAU3 = E20_data["siec"][0]["tau_abs"][0][-1]
    E20_SIG = E20_data["siec"][0]["sigma"].max()
    E20_SPEAK = E20_data["siec"][1]["sigma"].max()
    _petz = E20_data["petz"]
    E20_CMEAN, E20_CSTD = _petz["C_mean"], _petz["C_std"]
    _c4 = [r["C"] for r in _petz["rows"] if r["N"] == 4][0]
    _c16 = [r["C"] for r in _petz["rows"] if r["N"] == 16][0]
    E20_C4, E20_C16 = _c4, _c16
    E20_CT40 = E20_data["C_t"][-1][1] - 0.2
    E21_data = ext["E21"]
    E21_GAP = E21_data["gaps"][100]
    E21_DELTA = E21_data["F_an_vs_num"]
    E21_F40 = E21.F_rec_analityczna(40.0, M.GAMMA_B, 1)
    E21_CM = E21_data["C_mean"]
    E21_SPHI0 = E21_data["entrainment"][0.0]
    E21_SPHI1 = E21_data["entrainment"][0.2]
    E22_data = ext["E22"]
    E22_G1 = E22_data["drabina"][100]
    E22_GAP = E22_data["drabina"][4]
    E22_T0 = E22_data["frw"]["t0"]
    E22_T1 = E22_data["frw"]["t1"]
    E22_SH0 = E22_data["frw"]["S_H0"]
    E22_SH1100 = np.log10(E22.S_horyzont(1100))
    E23_data = ext["E23"]
    E23_D1 = E23_data["Tw1_delta"]
    E23_D1A = E23_data["Tw1a_delta"]
    E23_FSTAB = E23.F_stable_an(np.exp(-M.GAMMA_B * 40))
    E23_FULL2 = E23_data["Tw3_cold"][2]
    Fp2, Fs2 = E23.petz_proj_wzor(2, 160)
    E23_AVG2 = (Fp2 + Fs2) / 2

    # ---- R48–R49: ENTROPIA-4.0 ----
    E24_data = ext["E24"]
    N48 = E24_data["ness"]
    E48_STOT = N48["S_tot_end"]
    E48_J = N48["J_E_inf"]
    E48_SIG = N48["sig_tot_inf"]
    E48_CLAUS = N48["ratio_clausius"]
    E48_RCLK = N48["ratio_clock"]
    E48_RPK = N48["ratio_clock_peak"]
    E48_RLT = N48["ratio_late"]
    S49 = E24_data["sila"]
    E49_SK0 = float(S49["Sinf"][0]); E49_SKL = float(S49["Sinf"][-1])
    E49_FMIN = float(S49["F_d"].min())
    E49_DMIN = float(S49["ds"][int(np.argmin(S49["F_d"]))])
    F49 = E24_data["frw"]
    E49_TCROSS = float(F49["t"][F49["i_cross"]])
    E49_TNESS = float(F49["T_eff"][-1])
    E49_AMAX = float(F49["a"].max())
    E49_TAMAX = float(F49["t"][int(np.argmax(F49["a"]))])
    E49_TSYS = float(F49["tau_sys"][-1])
    E49_TBUD = float(F49["tau_bud"][-1])
    _m49 = F49["t"] >= F49["t"][F49["i_cross"]] + 0.1
    _zn = np.sign(F49["H"][_m49])
    _izm = int(np.argmax(np.diff(_zn) != 0)) + 1
    E49_HZERO = float(F49["t"][_m49][_izm])
    # R47: wartości po poprawce (tabela z E12.main())
    R47_ROWS = E12_data.get("R47", [])
    E47_R3D_10 = next((r["po3"] for r in R47_ROWS if r["TB"] == 10), np.nan)
    E47_R1_10 = next((r["po1"] for r in R47_ROWS if r["TB"] == 10), np.nan)

    # ---- R50: ENTROPIA-5.0 ----
    E25_data = ext["E25"]
    S50 = E25_data["singlet"]
    E50_FID_D = S50["fid_D"]; E50_FID_T0_BUG = S50["fid_T0_bug"]
    E50_CIRC = E25_data["circuit_blad"]
    P50 = E25_data["przewidywania"]
    E50_PDK = P50["P_D_kol_end"]; E50_PDN = P50["P_D_nz_end"]
    E50_PT0 = P50["P_T0_kol_end"]; E50_PDZ = P50["P_D_rz_end"]
    E50_EMGT = P50["exp_mgt"]; E50_EM2GT = P50["exp_m2gt"]
    E50_F = E25_data["rekon_F"]
    HW50 = E25_data["hardware"]["platformy"]
    E50_HERON = HW50[0]["max_krokow_zakres"]
    E50_WILLOW = HW50[1]["max_krokow_zakres"]
    r13rows = "".join(
        f"<tr><td>γ_graw = {g:g}</td><td>{s:.4f}</td></tr>" for g, s in R13["sig_gamma"])
    # ---- tabela syntezy R1–R13 ----
    syn_rows = [
        ("R1", "Skończona temperatura kąpieli", "S(∞) = H(1/(1+η)) < ln 2; overshoot η<1/3; kompresja 27× trwa", "rdzeń → R6, R13"),
        ("R2", "N=2: ciemny singlet", "S(∞) = ln 3 < 2·ln 2; I = ln(4/3); negatywność 0", "R4, R5"),
        ("R3", "Sprzężenie zegar → tempo", "γ_eff(T): chłodzenie/przyspieszanie; kompresja 27× nietknięta", "R6, R8"),
        ("R4", "N=3: sektory j", "j=3/2⊕2×j=1/2; subradiancja; S(∞)=ln 4 / ln 2", "R2, R7"),
        ("R5", "Entropia makro", "N·ln 2 (niezależne) vs ln(N+1) (kolektywne)", "R2, R4"),
        ("R6", "Gorący Wielki Wybuch", "S(0) ≈ ln 2, kąpiel zimna ⇒ czas wstecz (budżet 0.31)", "R3, R8"),
        ("R7", "Losowe stany, γ_φ=0", "koherencje A↔B = √(pA·pB) blokują entropię", "R4, R11"),
        ("R8", "Cykl BB → Kolaps", "czas dwustronny; τ = 2·budżet; czas jako pętla", "R3, R6"),
        ("R9", "Kwantowy zegar (operator)", "⟨n⟩, Δn ≈ √⟨n⟩, back-action; kompromis zegara", "R10"),
        ("R10", "Koherencje zegara", "start koherentny: stymulacja 4.1×; κ → czas klasyczny", "R9"),
        ("R11", "Odblokowanie γ_φ", "τ90 ∝ 1/γ_φ (nachylenie −0.96); pełna termalizacja", "R7"),
        ("R12", "Wielka synteza", "diagram powiązań; trzy osie; jeden zegar T = S", "wszystkie"),
        ("R13", "Grawitacyjna produkcja entropii", "NESS: σ > 0 stałe; T_graw rośnie liniowo — czas bez końca", "R1, R9"),
        ("R14", "Predykcje vs obserwacje", "s∝T³ → T_ν/T_γ (zgodne); subradiancja zmierzona; napięcia P11/P12", "wszystkie"),
        ("R15", "Dekoherencja zegara = strażnik historii", "κ rzutuje na oś liczbową: czas klasyczny, nieodwracalny, τ rośnie mimo H_int", "R9, R10"),
        ("R16", "Formalizm relacyjny (rewizja)", "λ→S→τ: dτ/dλ = α[Ṡ+η·I(A:E)]; 27 jako predykcja warunkowa (3^p)", "wszystkie"),
        ("R17", "Test laboratoryjny bright↔dark", "Γ_dark ≪ Γ_bright ⇒ τ̇↓; singlet: zegar milczy, pamięć I = ln(2/√3) trwa", "R2, R9, R16"),
        ("R18", "ENTROPIA-1.1 (N=2..100)", "sektory Dickego: S∞→ln(N+1); 27× dokładne (1e-12); czkanie przy η=0, τ̇→η·I_eq przy η>0; P_dark(Haar)→1", "wszystkie"),
        ("R19", "ENTROPIA-1.2 (konkurencja funkcjonałów)", "T0,T1,T3 stają, T2 nie (absolutna I); odzyskiwalność M(t): j=1 31× przy N=100, j=0 doskonała; fizyczny 27×: R_T=27 (kąpiel 3D) vs 3 (single-mode)", "R17, R18"),
        ("R20", "ENTROPIA-1.3 (I_c, koszt, protokół)", "I_c<0 ⇒ pamięć klasyczna; ΔE·Δτ≥ħ/2 ⇒ ω_c^min=1.7; trójkąt precyzja↔koszt↔entropia; protokół: τ̇ po Γ=0 rozstrzyga T1 vs T2", "R19"),
        ("R21", "ENTROPIA-1.4 (kanał odzysku)", "M_F=1−F: ciemny j=1 382× dłużej (N=100); F_e(j=0)=1 — doskonały odzysk; Fuchs–van de Graaf łączy z M(t)", "R19, R20"),
        ("R22", "ω_c(T) z fizycznej kąpieli", "n̄(ω_c,T)<ε ⇒ ω_c∝T (rozdzielczość ~T); produkcja entropii ∝T³ (27×); pojemność ln2/δs=69", "R20"),
        ("R23", "Protokół e2e z detekcją fotonów", "MC: moc 1.000 przy η_det=0.1 i szumie tła; τ̇ po ostatnim fotonie rozstrzyga T1 vs T2", "R20, R22"),
        ("R24", "ENTROPIA-1.5 (pamięć operacyjna)", "Helstrom/Chernoff/C_mem: j=1/2: 0.44 bitu, jasny N=100: 0.0004; j=0: niezmienniczy (0 bitów)", "R19, R21"),
        ("R25", "Samo-spójny ω_c(T)", "Purcell: γ_t∝g²ω_c³; okno ω_c∈(T·ln(1/ε),(ε_b/(cg²))^{1/3}); T_max ∝ g^{−2/3}", "R20, R22"),
        ("R26", "Sekwencyjny test Walda", "SPRT: E[N]=1 przy λ₂=7.19 (błędy 0); adaptacja: λ₂=0.1 ⇒ E[N]=20; optymalny w sensie Walda", "R23"),
        ("R27", "Eksperymentalna karta R23", "3 platformy (Rb/Cs/wnęka): t_D=179 μs; kanał korelacyjny I(A:B), M=150 realizacji/punkt, σ_I=0.01 nat; wykonalne dziś", "R23, R26"),
        ("R28", "Suchy bieg protokołu", "MC z detektorem (η=0.3, dark, jitter): SPRT E[N]=1, błędy 0; pomiar ~65 μs/realizację — karta potwierdzona", "R26, R27"),
        ("R29", "Mapa Petza (jawny odzysk)", "R(·)=σ^{1/2}Φ†(Φ(σ)^{-1/2}(·)Φ(σ)^{-1/2})σ^{1/2}: F_rec 0.89 (j=1/2) → 0.54 (j=3); j=0: 1", "R21, R24"),
        ("R30", "Zimny zegar (skończone widmo)", "Ohmic/Lorentzian: γ_t saturowane ⇒ T_max 216k vs 0.66 (3D); kosmologia: ω_c ≥ 2π×2.6e11 Hz (dziś) … 2π×1e26 (elektrosłaba)", "R22, R25"),
        ("R31", "Arkusz T_max/ω_c(T) (nadprzewodniki)", "zegar=rezonator, kąpiel=szum T: T_max 63 mK (6 GHz); n̄(ω_c,T) mierzalne spektroskopią; T1_Purcell 64 μs — tani test R25/R30", "R25, R30"),
        ("R32", "Koszt energetyczny protokołu", "zegar 2.6e-23 J, decyzja 9.6e-25 J (zaniedbywalne); pułapka 5 mJ dominuje; ΔE·Δτ ≥ ħ/2 ✓ (zapas 1e23)", "R23, R26"),
        ("R33", "Suchy bieg z niedoskonałą wiernością", "ρ=F·ρ10+(1−F)·𝟙/4: I_eq 0.144→0.014, τ̇_T2 7.19→0.71, ale moc 1.000 do F=0.3 (samo-kalibracja przez I(A:B))", "R26, R28"),
        ("R34", "Fizyczna realizacja mapy Petza", "kody fazowe: F(Petz)>F(klasyczny); echo odzyskuje fazę (j≤1), szkodzi przy zaniku amplitudowym; DFS (ciemny): F=1 — chroń, nie odzyskuj", "R29, R21"),
        ("R35", "Zegar w kąpieli CMB", "próg ω_c/2π ≥ 261 GHz; 100 GHz n̄=0.207 (szum), 1 THz bezpieczny; cutoff grawitacyjny ω_Planck bez wpływu dla realnych częstości", "R30, R31"),
        ("R36", "Kosmiczna sieć zegarów", "synchronizacja przez wymianę entropii: σ_end 48→0.08 (g_sync 0→0.2); jednakowe T ⇒ σ≡0 bez sprzężenia; τ_net = emergentny czas kosmiczny", "R18, R5"),
        ("R37", "Protokół różnicowy wielu zegarów", "Δτ̄ = τ_A−τ_B per tyk: T1 ≈ 0 vs T2 ≈ 7.19; common mode odrzucony; σ ↓ 1/√M_A (0.25→0.09)", "R23, R36"),
        ("R38", "Zegar w ewoluującym CMB", "horyzont zegarów (ΛCDM): 300 GHz od z≈0.1, 1 THz od z≈2.8, 3 THz od z≈10.5, 10 THz od z≈37; 6/100 GHz nigdy", "R35, R36"),
        ("R39", "Pakiet publikacyjny", "MANUSKRYPT.md: streszczenie, rdzeń, sektory, kosmologia, zegar, falsyfikacja, eksperyment, predykcje, ograniczenia, bibliografia", "wszystkie"),
        ("R40", "Sieć z dynamiką η(T) (R8×R36)", "upływ τ_abs rośnie przez cykle (3×budżet=1.24), T_signed wraca; jednakowe: σ≡0; offsety fazy: σ modulowane cyklem", "R8, R36"),
        ("R41", "Asymptotyka Petza (Dicke)", "F_rec → 1/(N+1) (t→∞); C(t)=F_rec−1/(N+1) ≈ 0.215±0.017 niezależne od N; Γ₁=Nγ niszczy pamięć; ciemny: F=1", "R29, R21"),
        ("R42", "Formalny limit Petza (dowód num.)", "gap/γ = 1.0000 (N=2..100); dokładny wzór F_rec(t)=½a(2+(1−a)²/(1−½a²)) Δ=1e-16; N→∞: jasny→0, ciemny→1", "R41, R21"),
        ("R43", "Entrainment faz sieci (η(T))", "σ_φ: 9.56 → 0.000 (g_sync=0.2) — fazy cykli LOCKUJĄ się; jednolity kosmiczny czas z niejednorodnych komórek", "R40, R36"),
        ("R44", "Dowód uniwersalności C(t)", "drabina Dickego Γ_n=n(N−n+1)γ, gap=γ niezależne od N; okno (1/(Nγ), 1/γ); F_rec=½a(2+(1−a)²/(1−½a²)) Δ=9e-16", "R42, R41"),
        ("R45", "ENTROPIA-3.0 — metryka FRW", "s·a³=const (T³); S_eq komórki maleje (dτ=|dS|); horyzont: S_BH 10¹⁴⁰→10¹³⁰ k_B (z: 0→1100); czas FRW = upływ entropii komobowej", "R13, R6"),
        ("R46", "Dowód wzoru Petza z regularyzacją", "Tw.1: F_rec=½a(2+(1−a)²/(1−½a²)) Δ=3e-16; Tw.2: Γ=Nγ (niezmienniczość); Tw.3: pełny ε→0 = średnia rzutowana (przeciek Φ†); Tw.4: C(t) uniwersalne (gap=γ)", "R42, R44"),
        ("R47", "Poprawka R_T (audyt 1.2)", "R_T_fizyczny: pełna spójna termiczna (dSdt_termiczne_analitycznie) — projekt ≡ świadek do 1e-10; TB=10: 27.850 (3D), 3.092 (single); wniosek 27/3 odporny", "R19, audyt"),
        ("R48", "ENTROPIA-4.0 — dwie komórki (NESS)", "S_tot∞=1.3491; J_E,∞=0.00507; σ_NESS=0.00338 = J·(1/T_B−1/T_A) (Clausius Δ=1e-6); Fouriera: J↑ΔT (nasycenie); produkcja → zimna komórka", "R13, R47"),
        ("R49", "ENTROPIA-4.0 — siła + emergentna FRW", "S∞(κ): 1.2617→1.3546 ⇒ F(d)<0 (przyciąganie, nie 1/d²); inwersja→T=∞ (Wielki Wybuch): a: 0→1.237→1 (odbicie); H: +∞→0; τ_sys=1.3491 (skończony), τ_bud liniowy; σ_A/σ_B≈22→0.034", "R45, R8, R48"),
        ("R50", "ENTROPIA-5.0 — pętla pomiarowa (IBM/Sycamore)", "POPRAWKA: h,cx,x = Ψ+ (jasny!), h,cx,z,x = Ψ−; P_D: kolektywna 1.000000 vs niezależna e^{−γt}; |T0⟩: e^{−2γt}; rz odblokowuje; obwód z ancillą ≡ Kraus (Δ=0); F(rekonstr.)=0.967; Heron 3–9 / Willow 2–5 kroków", "R17, R23, R27"),
    ]
    synth_table = "".join(
        f"<tr><td><b>{n}</b></td><td>{t}</td><td>{w}</td><td>{p}</td></tr>"
        for n, t, w, p in syn_rows)

    ext_html = EXT_TEMPLATE
    for key, val in {
        "FIGR1": imgR1, "FIGR2": imgR2, "FIGR3": imgR3,
        "FIGR4": imgR4, "FIGR5": imgR5, "FIGR6": imgR6,
        "FIGR7": imgR7, "FIGR8": imgR8, "FIGR9": imgR9,
        "FIGR10": imgR10, "FIGR11": imgR11,
        "R1ROWS": r1rows, "R2ROWS": r2rows, "R3ROWS": r3rows,
        "R4ROWS": r4rows, "R5ROWS": r5rows,
        "R7ROWS": r7rows, "R7ROWS4": r7rows4, "R9ROWS": r9rows,
        "R10ROWS": r10rows, "R11ROWS": r11rows,
        "R1_SMAX": f"{ext['R1_overshoot']['smax']:.4f}",
        "R1_TMAX": f"{ext['R1_overshoot']['tmax']:.1f}",
        "R1_SEQ": f"{ext['R1_overshoot']['seq']:.4f}",
        "R1_DELTA": f"{ext['R1_overshoot']['delta']:+.4f}",
        "R2_MI11": f"{ext['R2']['mi11']:.4f}",
        "R2_MI10": f"{ext['R2']['mi10']:.4f}",
        "R3_COMP": f"{ext['R3_comp']:.1e}",
        "R4_LN4": f"{R4['ln4']:.4f}", "R4_LN108_3": f"{R4['ln108_3']:.4f}",
        "R4_DEF": f"{R4['deficit_32']:.4f}",
        "R5_IND16": "11.0904", "R5_ERR16": "2.2e-13",
        "R6_S0": f"{R6['S0']:.4f}", "R6_SEQ": f"{R6['Seq']:.4f}",
        "R6_BUDZET": f"{R6['budzet']:.4f}",
        "R6_THC": f"{R6['t_half_c']:.2f}", "R6_THF": f"{R6['t_half_f']:.2f}",
        "R6_NZ": f"{R6['nz']}", "R6_NT": f"{R6['n_total']}",
        "R7_COH": f"{abs(R7['rho_sec'][4,6]) + abs(R7['rho_sec'][5,7]):.4f}",
        "R8_SMIN": f"{R8['Smin']:.4f}", "R8_BUDZET": f"{R8['budzet']:.4f}",
        "R8_TAU": f"{R8['t_abs_total']:.4f}", "R8_FRAK": f"{100*R8['frakcja_wstecz']:.0f}",
        "R8_ZAMROZ": f"{R8['zamroz']}", "R8_NCYC": f"{R8['n_cyc']}",
        "R9_GT": f"{R9['gt']}", "R9_SEND": f"{R9['S_end']:.4f}",
        "R9_DEV": f"{R9['dev']:+.4f}", "R9_NBAR": f"{R9['nbar_end']:.2f}",
        "R9_DN": f"{R9['dn_end']:.2f}", "R9_REL": f"{R9['rel_end']:.2f}",
        "R9_I": f"{R9['I_end']:.4f}",
        "R10_STIM": f"{stim:.1f}",
        "R10_SVAC": f"{sceny10['próżnia κ=0']['dev']:+.4f}",
        "R10_SCOH": f"{sceny10['koherentny κ=0']['dev']:+.4f}",
        "R10_RELVAC": f"{sceny10['próżnia κ=0']['rel']:.2f}",
        "R10_RELCOH": f"{sceny10['koherentny κ=0']['rel']:.2f}",
        "R10_COHCOH": f"{sceny10['koherentny κ=0']['coh']:.4f}",
        "R10_COHKAP": f"{sceny10['koherentny κ=0.3']['coh']:.4f}",
        "R10_IVAC": f"{sceny10['próżnia κ=0']['I']:.4f}",
        "R10_ICOH": f"{sceny10['koherentny κ=0']['I']:.4f}",
        "R10_IKAP": f"{sceny10['koherentny κ=0.3']['I']:.4f}",
        "FIGR13": imgR13, "R13ROWS": r13rows, "SYNTHTABLE": synth_table,
        "FIGR14": imgR14, "R14ROWS": ext["R14_rows"],
        "R14_P1": ext["R14_p1"], "R14_NEFF": ext["R14_neff"],
        "R14_SRATIO": ext["R14_sratio"], "R14_FRAC": ext["R14_frac"],
        "R14_SPERB": ext["R14_sperb"],
        "R14_DARKFRAC": ext["R14_darkfrac"], "R14_NUFRAC": ext["R14_nufrac"],
        "FIGR15": imgR15,
        "R15_G": f"{R15['g']}", "R15_GT": f"{R15['gt']}",
        "R15_K0_COH": f"{R15['k0.0']['coh_turn']:.5f}",
        "R15_K0_OFF": f"{R15['k0.0']['offdiag_turn']:.2e}",
        "R15_K0_I": f"{R15['k0.0']['I_turn']:.4f}",
        "R15_K0_SCL": f"{R15['k0.0']['Scl_turn']:.3f}",
        "R15_K5_COH": f"{R15['k0.5']['coh_turn']:.5f}",
        "R15_K5_OFF": f"{R15['k0.5']['offdiag_turn']:.2e}",
        "R15_K5_I": f"{R15['k0.5']['I_turn']:.4f}",
        "R15_K5_SCL": f"{R15['k0.5']['Scl_turn']:.3f}",
        "R15_SUP": f"{R15['k0.0']['offdiag_turn']/R15['k0.5']['offdiag_turn']:.0f}",
        "R15_NTURN": f"{R15['k0.0']['n_turn']}",
        "FIGR16": imgR16, "FIGR17": imgR17,
        "R16_TAU_ENT": f"{R16['tau_ent']:.1f}", "R16_TAU_REL": f"{R16['tau_rel']:.1f}",
        "R16_S": f"{R16['pr']['s_branch']:.1f}", "R16_SDOT1": f"{R16['pr']['sdot_tick1']:.1f}",
        "R17_TAU_S": f"{R17['tau_s']:.1f}", "R17_TAU_11": f"{R17['tau_11']:.0f}",
        "R17_RATE11": f"{R17['rate11']:.4f}", "R17_RATE10": f"{R17['rate10']:.4f}",
        "R17_RATES": "".join(f"<tr><td>{p:.1f}</td><td>{r:.4f}</td></tr>"
                            for p, r in zip(R17["ps"], R17["rates"])),
        "R17_FID": f"{R17['fid_end']:.3f}",
        "R17_MI10": "0.1438",
        "FIGE1": imgE1, "FIGE2": imgE2, "FIGE3": imgE3, "FIGE4": imgE4,
        "E11_SINF_ERR": f"{abs(E11_SINF_ERR):.1e}",
        "E11_27_WORST": f"{E11_27_WORST:.1e}",
        "E11_I": "0.1438", "E11_PDARK": f"{E11_PDARK:.4f}",
        "E11_SHAAR": f"{E11_SHAAR:.4f}", "E11_TAUENT": "0.0000",
        "E11_TAUREL": "0.0719",
        "FIGE5": imgE5, "FIGE6": imgE6, "FIGE7": imgE7,
        "E12_T0": "0", "E12_T1": "0", "E12_T2": "7.19", "E12_T3": "0",
        "E12_IEQ": f"{E12_Ieq:.4f}",
        "E12_GAIN4": f"{E12_GAIN4:.1f}", "E12_GAIN100": f"{E12_GAIN100:.1f}",
        "E12_R3D": f"{E12_R3D:.3f}", "E12_RSINGLE": f"{E12_RSINGLE:.3f}",
        "E12_CONV": f"{E12_CONV:.3f}",
        "FIGE8": imgE8, "FIGE9": imgE9, "FIGE10": imgE10,
        "E13_IC4": f"{E13_IC4:.4f}", "E13_IC10": f"{E13_IC10:.4f}",
        "E13_IA": f"{E13_IA:.4f}",
        "E13_WCMIN": f"{E13_WCMIN:.1f}",
        "E13_PREC_LO": f"{E13_PREC_LO:.3f}", "E13_PREC_HI": f"{E13_PREC_HI:.3f}",
        "E13_COST_LO": f"{E13_COST_LO:.0f}", "E13_COST_HI": f"{E13_COST_HI:.0f}",
        "E13_BACK_LO": f"{E13_BACK_LO:.4f}", "E13_BACK_HI": f"{E13_BACK_HI:.4f}",
        "E13_TAU1": f"{E13_TAU1:.6f}", "E13_TAU2": f"{E13_TAU2:.2f}",
        "FIGE11": imgE11, "FIGE12": imgE12, "FIGE13": imgE13,
        "E14_MF4": f"{E14_MF4:.4f}", "E14_MF100": f"{E14_MF100:.4f}",
        "E14_GAIN100": f"{E14_GAIN100:.0f}",
        "E14_FE0": f"{E14_FE0:.3f}", "E14_FE1": f"{E14_FE1:.3f}",
        "E14_FE2": f"{E14_FE2:.3f}",
        "E14_WC10": f"{E14_WC10:.1f}", "E14_WC100": f"{E14_WC100:.1f}",
        "E14_RATIO3": f"{E14_RATIO3:.2f}",
        "E14_CAP": f"{E14_CAP:.0f}",
        "E14_MOC10": f"{E14_MOC10:.3f}", "E14_MOC_T2": f"{E14_MOC_T2:.3f}",
        "E14_MOC_DR": f"{E14_MOC_DR:.3f}",
        "E14_NLAST": f"{E14_NLAST}",
        "FIGE14": imgE14, "FIGE15": imgE15, "FIGE16": imgE16,
        "E15_CM4": f"{E15_CM4:.3f}", "E15_CM100": f"{E15_CM100:.3f}",
        "E15_CMD": f"{E15_CMD:.3f}", "E15_CMD15": f"{E15_CMD15:.3f}",
        "E15_CMH": f"{E15_CMH:.3f}",
        "E15_TMAX_1": f"{E15_TMAX_1:.2f}", "E15_TMAX_2": f"{E15_TMAX_2:.2f}",
        "E15_E1": f"{E15_E1:.1f}", "E15_E2": f"{E15_E2:.1f}",
        "E15_EN05": f"{E15_EN05:.1f}", "E15_EN01": f"{E15_EN01:.1f}",
        "FIGE17": imgE17,
        "EXP_TB_B": f"{EXP_TB_B:.0f}", "EXP_TD_B": f"{EXP_TD_B:.0f}",
        "EXP_TDTB": f"{EXP_TDTB:.0e}",
        "EXP_SNR_A": f"{EXP_SNR_A:.0f}", "EXP_SNR_B": f"{EXP_SNR_B:.1f}",
        "EXP_SNR_C": f"{EXP_SNR_C:.0f}",
        "EXP_M": f"{EXP_M}", "EXP_SIGI": f"{EXP_SIGI:.3f}",
        "EXP_TAU2": f"{EXP_TAU2:.1e}", "EXP_TAU1": f"{EXP_TAU1:.0f}",
        "FIGE18": imgE18, "FIGE19": imgE19,
        "E16_FREC_05": f"{E16_FREC_05:.3f}", "E16_FREC_1": f"{E16_FREC_1:.3f}",
        "E16_FREC_3": f"{E16_FREC_3:.3f}",
        "E16_TM3": f"{E16_TM3:.2f}", "E16_TOHM": f"{E16_TOHM:.0f}",
        "E16_CMB": f"{E16_CMB:.1e}", "E16_REC": f"{E16_REC:.1e}",
        "E16_BBN": f"{E16_BBN:.1e}", "E16_EW": f"{E16_EW:.1e}",
        "E16_TLAST": f"{E16_TLAST}", "E16_TTOT": f"{E16_TTOT:.0f}",
        "FIGE20": imgE20, "FIGE21": imgE21, "FIGE22": imgE22,
        "E17_TMAX6": f"{E17_TMAX6:.0f}", "E17_TMAX30": f"{E17_TMAX30:.0f}",
        "E17_T1P": f"{E17_T1P:.0f}",
        "E17_ECLK": f"{E17_ECLK:.1e}", "E17_ETRAP": f"{E17_ETRAP:.0f}",
        "E17_ELAND": f"{E17_ELAND:.1e}", "E17_DEDT": f"{E17_DEDT:.1e}",
        "E17_IEQ1": f"{E17_IEQ1:.4f}", "E17_IEQ3": f"{E17_IEQ3:.4f}",
        "E17_TAU2_1": f"{E17_TAU2_1:.2f}", "E17_TAU2_3": f"{E17_TAU2_3:.2f}",
        "E17_P1_3": f"{E17_P1_3:.3f}",
        "FIGE23": imgE23, "FIGE24": imgE24, "FIGE25": imgE25,
        "E18_P05": f"{E18_P05:.3f}", "E18_E05": f"{E18_E05:.3f}",
        "E18_P20": f"{E18_P20:.3f}", "E18_E20": f"{E18_E20:.3f}",
        "E18_WCMIN": f"{E18_WCMIN:.0f}",
        "E18_N100": f"{E18_N100:.2e}", "E18_N1T": f"{E18_N1T:.2e}",
        "E18_S0": f"{E18_S0:.2f}", "E18_S2": f"{E18_S2:.3f}",
        "FIGE26": imgE26, "FIGE27": imgE27,
        "E19_DR_T1": f"{E19_DR_T1:.3f}", "E19_DR_T2": f"{E19_DR_T2:.3f}",
        "E19_SIG1": f"{E19_SIG1:.3f}", "E19_SIG8": f"{E19_SIG8:.3f}",
        "E19_Z6": f"{E19_Z6}", "E19_Z300": f"{E19_Z300:.1f}",
        "E19_T1T": f"{E19_T1T:.1f}", "E19_T3T": f"{E19_T3T:.1f}",
        "E19_T10T": f"{E19_T10T:.1f}",
        "FIGE28": imgE28, "FIGE29": imgE29, "FIGE30": imgE30,
        "E20_BUDZET": f"{E20_BUDZET:.4f}", "E20_TAU3": f"{E20_TAU3:.4f}",
        "E20_SIG": f"{E20_SIG:.1e}", "E20_SPEAK": f"{E20_SPEAK:.4f}",
        "E20_CMEAN": f"{E20_CMEAN:.3f}", "E20_CSTD": f"{E20_CSTD:.3f}",
        "E20_C4": f"{E20_C4:.3f}", "E20_C16": f"{E20_C16:.3f}",
        "E20_CT40": f"{E20_CT40:.3f}",
        "FIGE31": imgE31, "FIGE32": imgE32, "FIGE33": imgE33,
        "E21_GAP": f"{E21_GAP:.4f}", "E21_DELTA": f"{E21_DELTA:.0e}",
        "E21_F40": f"{E21_F40:.4f}", "E21_CM": f"{E21_CM:.3f}",
        "E21_SPHI0": f"{E21_SPHI0:.3f}", "E21_SPHI1": f"{E21_SPHI1:.4f}",
        "FIGE34": imgE34, "FIGE35": imgE35,
        "E22_G1": f"{E22_G1:.0f}", "E22_GAP": f"{E22_GAP:.2f}",
        "E22_T0": f"{E22_T0:.2f}", "E22_T1": f"{E22_T1:.2f}",
        "E22_SH0": f"{E22_SH0:.1f}", "E22_SH1100": f"{E22_SH1100:.1f}",
        "FIGE36": imgE36,
        "E23_D1": f"{E23_D1:.0e}", "E23_D1A": f"{E23_D1A:.0e}",
        "E23_FSTAB": f"{E23_FSTAB:.4f}",
        "E23_FULL2": f"{E23_FULL2:.4f}", "E23_AVG2": f"{E23_AVG2:.4f}",
        "FIGA3": imgA3,
        "E47_R3D": f"{E47_R3D_10:.3f}", "E47_R1": f"{E47_R1_10:.3f}",
        "FIGE37": imgE37, "FIGE38": imgE38, "FIGE39": imgE39, "FIGE40": imgE40,
        "E48_STOT": f"{E48_STOT:.4f}", "E48_J": f"{E48_J:.5f}",
        "E48_SIG": f"{E48_SIG:.5f}", "E48_CLAUS": f"{E48_CLAUS:.6f}",
        "E48_RCLK": f"{E48_RCLK:.1f}", "E48_RPK": f"{E48_RPK:.1f}",
        "E48_RLT": f"{E48_RLT:.3f}",
        "E49_SK0": f"{E49_SK0:.4f}", "E49_SKL": f"{E49_SKL:.4f}",
        "E49_FMIN": f"{E49_FMIN:.4f}", "E49_DMIN": f"{E49_DMIN:.2f}",
        "E49_TCROSS": f"{E49_TCROSS:.2f}", "E49_TNESS": f"{E49_TNESS:.3f}",
        "E49_AMAX": f"{E49_AMAX:.4f}", "E49_TAMAX": f"{E49_TAMAX:.2f}",
        "E49_HZERO": f"{E49_HZERO:.2f}", "E49_TSYS": f"{E49_TSYS:.4f}",
        "E49_TBUD": f"{E49_TBUD:.4f}",
        "FIGE41": imgE41, "FIGE42": imgE42, "FIGE43": imgE43,
        "E50_FID_D": f"{E50_FID_D:.4f}", "E50_FID_T0_BUG": f"{E50_FID_T0_BUG:.4f}",
        "E50_CIRC": f"{E50_CIRC:.0e}",
        "E50_PDK": f"{E50_PDK:.6f}", "E50_PDN": f"{E50_PDN:.4f}",
        "E50_EMGT": f"{E50_EMGT:.4f}",
        "E50_PT0": f"{E50_PT0:.4f}", "E50_EM2GT": f"{E50_EM2GT:.4f}",
        "E50_PDZ": f"{E50_PDZ:.4f}",
        "E50_F": f"{E50_F:.4f}",
        "E50_HERON": f"{E50_HERON[0]}–{E50_HERON[1]}",
        "E50_WILLOW": f"{E50_WILLOW[0]}–{E50_WILLOW[1]}",
        "R13_SNESS": f"{N13['S_ness']:.4f}", "R13_SIG": f"{N13['sigma']:.5f}",
        "R13_SIGTICK": f"{N13['sigma_tick']:.6f}",
        "R13_SUM": f"{N13['sum_sig_tau']:.2f}", "R13_RATIO": f"{N13['ratio']:.1f}",
        "R13_GAMMA_G": f"{N13['gamma_g']}", "R13_ETA_G": f"{N13['eta_g']}",
        "R13_ETA_R": f"{N13['eta_r']}", "R13_GAMMA_R": f"{N13['gamma_r']}",
        "R11_M3": f"{R11['m3']:.2f}", "R11_M4": f"{R11['m4']:.2f}",
        "R11_B111": f"{R11['blokada']['|111⟩']:.4f}",
        "R11_B100": f"{R11['blokada']['|100⟩']:.4f}",
        "R11_BLOS": f"{R11['blokada']['losowy #1']:.4f}",
        "R11_T90_1E4": f"{R11['tau90_3'][1e-4]:.0f}",
        "R11_T90_1E3": f"{R11['tau90_3'][1e-3]:.0f}",
        "R11_T90_1E2": f"{R11['tau90_3'][1e-2]:.0f}",
        "R11_T90_3E-1": f"{R11['tau90_3'][0.3]:.0f}",
        "JS_ALPHAS": js_alphas, "JS_THC": js_th_c, "JS_THA": js_th_a,
        "JS_SC": js_SC,
        "JS_SCYC": json.dumps(R8["S"]),
        "JS_PN50": json.dumps(R9["pn50"]), "JS_PN150": json.dumps(R9["pn150"]),
        "JS_PNEND": json.dumps(R9["pn_end"]),
        "JS_KRZYWE": js_krzywe, "JS_TAU3": js_tau3,
    }.items():
        ext_html = ext_html.replace("@@" + key + "@@", val)

    ext_script = EXT_SCRIPT
    for key, val in {
        "GAMMA_B": f"{M.GAMMA_B:.4f}",
        "TAU": f"{M.DELTA_TAU:.2f}",
        "LN2": f"{M.LN2:.6f}",
        "DS_Q": f"{M.DELTA_S_Q:.2f}",
        "JS_ALPHAS": js_alphas, "JS_THC": js_th_c, "JS_THA": js_th_a,
        "JS_SC": js_SC,
        "JS_KRZYWE": js_krzywe, "JS_TAU3": js_tau3,
    }.items():
        ext_script = ext_script.replace("@@" + key + "@@", val)

    html = html.replace("@@EXTENSIONS@@", ext_html)
    html = html.replace("@@EXT_SCRIPT@@", ext_script)

    out = os.path.join(HERE, "raport.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("Zapisano:", out, f"({os.path.getsize(out)//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kosmologiczny model «ENTROPIA» — czas jest entropią</title>
<style>
  :root{
    --ink:#1b2733; --mut:#5b6b7b; --line:#dbe4ec;
    --a:#c0392b; --b:#2471a3; --v:#8e44ad; --bg:#f6f9fc;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:16px/1.65 Georgia,"Times New Roman",serif}
  .wrap{max-width:980px;margin:0 auto;padding:0 20px 60px}
  header.hero{background:linear-gradient(135deg,#132a3c,#1b3a54 55%,#243d4f);
              color:#eef5fb;padding:42px 20px 34px;text-align:center}
  header.hero h1{margin:0 0 6px;font-size:34px;letter-spacing:.5px}
  header.hero .sub{font-size:16px;color:#a9c4da;font-style:italic}
  header.hero .badge{margin-top:14px;font-size:13px;color:#d5e6f4;letter-spacing:2px}
  h2{font-size:24px;margin:44px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line);
     color:#14304a}
  h2 .no{color:var(--v);font-weight:bold}
  h3{font-size:18px;margin:26px 0 6px;color:#1b3a54}
  p{margin:10px 0}
  .lead{font-size:17.5px;color:#31475c}
  .formula{background:#eef3f8;border:1px solid var(--line);border-left:4px solid var(--v);
           padding:12px 16px;margin:14px 0;font:15px/1.5 Consolas,Menlo,monospace;
           overflow-x:auto;border-radius:4px}
  .note{background:#fdf6e7;border:1px solid #f0e0b6;border-left:4px solid #d9a928;
        padding:10px 14px;margin:12px 0;border-radius:4px;font-size:15px}
  figure{margin:22px 0;text-align:center}
  figure img{max-width:100%;border:1px solid var(--line);border-radius:6px;
             background:#fff;box-shadow:0 1px 4px rgba(20,40,60,.08)}
  figcaption{font-size:13.5px;color:var(--mut);margin-top:6px;font-style:italic}
  table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14.5px;background:#fff}
  th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
  th{background:#eef3f8;color:#14304a}
  td:first-child{font-weight:600;color:#31475c}
  ul,ol{margin:8px 0 8px 22px;padding:0}
  li{margin:5px 0}
  .feat{background:#fff;border:1px solid var(--line);border-radius:8px;
        padding:16px 20px;margin:16px 0}
  .feat h3{margin-top:0}
  .feat .tag{display:inline-block;background:#e9ddf2;color:#6b2f8e;border-radius:20px;
             padding:1px 12px;font-size:12.5px;font-weight:700;letter-spacing:.4px;
             margin-bottom:8px}
  code{background:#eef3f8;border-radius:4px;padding:1px 6px;font:14px Consolas,monospace}
  .kbd{background:#fff;border:1px solid var(--line);border-bottom-width:2px;
       border-radius:5px;padding:1px 8px;font:13px Consolas,monospace}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:760px){.grid2{grid-template-columns:1fr}}
  .demo{background:#0f2333;color:#d7e6f2;border-radius:10px;padding:18px 20px;margin:18px 0}
  .demo canvas{display:block;width:100%;height:auto;background:#0c1d2b;
               border:1px solid #2a465e;border-radius:6px}
  .demo .controls{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
  .demo button{background:#2a465e;color:#eaf4fb;border:1px solid #3d5f7d;border-radius:6px;
               padding:6px 14px;font:14px sans-serif;cursor:pointer}
  .demo button.on{background:#8e44ad;border-color:#a05fc0}
  .demo button:hover{background:#35546e}
  .demo label{font:13px sans-serif;color:#9db8cd}
  .demo .readout{font:13px/1.7 Consolas,monospace;color:#bfd8ea;margin-top:10px}
  .demo .frozen{color:#f0c36d;font-weight:bold}
  .foot{margin-top:50px;padding-top:14px;border-top:1px solid var(--line);
        font-size:13px;color:var(--mut);text-align:center}
  a{color:#1b6aa8;text-decoration:none}
  a:hover{text-decoration:underline}
</style>
</head>
<body>
<header class="hero">
  <div class="badge">MODEL KOSMOLOGICZNY · WERSJA 1.0 · 13.08.2026</div>
  <h1>«ENTROPIA»</h1>
  <div class="sub">Czas jest entropią. Tyknięcia są dyskretne. Czas potrafi „czkać”.</div>
</header>
<div class="wrap">

<p class="lead">
Model łączy <b>mechanikę kwantową otwartych układów</b> (równanie Lindblada dla kubitu
zanurzonego w kąpieli termicznej) z <b>relacyjną koncepcją czasu</b>: zegar kosmiczny
nie mierzy upływu „zewnętrznego” czasu, lecz <i>kumuluje produkowaną entropię</i>.
W gorącym otoczeniu entropia (a więc i czas) płynie 27× szybciej niż w zimnym —
bo entropia właściwa promieniowania skaluje się jak <code>T³</code>.
Wszystkie siedem postulowanych cech modelu wyprowadzono z jednego równania
Lindblada i jednej definicji zegara.
</p>

<h2><span class="no">1.</span> Mikrofizyka: kubit + kąpiel termiczna</h2>
<p>
Każda „komórka” wszechświata to <b>kubit</b> — dwupoziomowy układ kwantowy o stanie
<code>ρ</code> (macierz gęstości). Kubit zanurzony jest w lokalnej kąpieli termicznej;
ewolucję opisuje generator Lindblada (równanie master w formie Kossakowskiego–Lindblada):
</p>
<div class="formula">
dρ/dt = −i[H,ρ] + γ·D[σ₋] + γ·D[σ₊] + γ_φ·D[σ_z],&nbsp;&nbsp;
D[L]ρ = LρL† − ½{L†L,ρ},&nbsp;&nbsp; H = (Ω/2)·σ_z
</div>
<p>
Kąpiel jest <b>nieskończenie gorąca</b> (równe tempo emisji i absorpcji kwantów, γ dla σ₊ i σ₋),
więc stan równowagi to <b>stan maksymalnie mieszany</b>:
</p>
<div class="formula">ρ_eq = ½·𝟙 &nbsp;⇒&nbsp; S(ρ_eq) = −Tr(½·𝟙·ln(½·𝟙)) = ln 2</div>
<p>
Generator jest <b>unitalny</b> (ℒ(𝟙) = 0), a dla map unitalnych entropia von Neumanna
jest <b>monotonicznie niemalejąca</b> (twierdzenie Ando–Lindblada). Dodatkowo
<code>S(ρ‖ρ_eq) = ln 2 − S(ρ)</code>, więc produkcja entropii w sensie Spohna
<code>σ = −d/dt S(ρ‖ρ_eq) = dS/dt ≥ 0</code> — druga zasada spełniona z automatu.
Mamy zatem dokładnie założony schemat: <b>S rośnie od 0 (stan czysty) do ln 2</b>.
</p>

<h2><span class="no">2.</span> Temperatura jako tempo: skąd 27?</h2>
<p>
Entropia właściwa promieniowania w kosmologii standardowej skaluje się jak
<code>s ∝ T³</code>. Dlatego dla dwóch „komórek” o temperaturach <b>T_A = 3·T_B</b>
wszystkie tempa dyssypacji są w stosunku:
</p>
<div class="formula">γ_A / γ_B = (T_A/T_B)³ = 3³ = <b>27</b></div>
<p>
Temperatura nie zmienia <i>celu</i> ewolucji (oba układy dążą do ρ_eq = ½·𝟙, S → ln 2),
ale ustawia <i>tempo</i>: A produkuje entropię 27× szybciej niż B. Ponieważ dynamika jest
identyczna poza skalą tempa, obowiązuje dokładna tożsamość kompresji czasowej
(ver. numerycznie, błąd @@COMP_ERR@@):
</p>
<div class="formula">S_A(t) ≡ S_B(27·t)</div>

<h2><span class="no">3.</span> Zegar kosmiczny: czas = entropia</h2>
<p>
Mikro-tyknięcie <code>τ = @@TAU@@</code> („planckowski” krok) przesuwa stan przez mapę
CPTP: <code>ρ_{n} = e^{ℒτ}·ρ_{n−1}</code>. Produkcja entropii w tyknięciu n:
</p>
<div class="formula">ΔS_n = S(ρ_n) − S(ρ_{n−1}) ≥ 0</div>
<p>
<i>Definicja czasu kosmologicznego</i> (relacyjny czas termiczny — czas nie jest
parametrem zewnętrznym, lecz <b>jest</b> entropią):
</p>
<div class="formula">Δt_n = κ·ΔS_n, &nbsp;&nbsp; T(n) = Σ_{k=1..n} Δt_k = κ·S(n), &nbsp;&nbsp; κ = @@KAPPA@@</div>
<p>
Entropia jest przy tym <b>kwantowana w „bitach”</b> <code>δs = @@DS_Q@@ nat</code>:
w tyknięciu pada <code>k_n ~ Poisson(ΔS_n/δs)</code> kwantów, więc
<code>Δt_n = k_n·δs</code>. Gdy <code>ΔS_n</code> jest małe, <code>k_n = 0</code>
i <code>Δt_n = 0</code> — czas staje w miejscu, by po chwili „czknąć”.
</p>

<!-- ==================== CECHY ==================== -->
<h2><span class="no">4.</span> Siedem cech modelu</h2>

<div class="feat">
<span class="tag">CECHA 0 — twierdzenie Lindblada ✅</span>
<h3>Entropia: monotoniczny wzrost 0 → ln 2</h3>
<img src="@@FIG1@@" alt="Wzrost entropii do ln 2">
<p>
Obie komórki startują z tego samego stanu czystego (S = 0, |r| = 1) i dążą
monotonicznie do <b>S(∞) = ln 2 = @@LN2@@ nat</b> (mapa unitalna ⇒ tw. Ando–Lindblada;
σ = dS/dt ≥ 0 — tw. Spohna). Po <code>@@N_TICKS@@</code> tyknięciach:
A osiąga ln 2 z dokładnością maszynową, B do <code>4·10⁻⁵</code>.
Czas do połowy entropii: <b>t_A = @@TA_HALF@@</b>, <b>t_B = @@TB_HALF@@</b>
— stosunek dokładnie <b>@@RATIO_HALF@@</b>. Wstawka potwierdza tożsamość
<code>S_A(n) = S_B(27n)</code>.
</p>
</div>

<div class="feat">
<span class="tag">CECHA 1</span>
<h3>Produkcja entropii na tyknięcie: A ~27× szybciej niż B</h3>
<img src="@@FIG2@@" alt="Produkcja entropii na tyknięcie">
<p>
Lewy panel: <code>ΔS_n</code> na tyknięcie dla A (czerwony) i B (niebieski).
A spala swoją entropię w ~10 tyknięciach, B ciągnie przez setki — krzywe są
identyczne po 27-krotnej kompresji czasu: <code>ΔS_A(n) = ΔS_B(27n)</code>.
Prawy panel pokazuje <i>dokładny</i> sens „27×”: gdy obie komórki mają <b>ten sam</b>
poziom entropii S*, chwilowe tempo produkcji <code>dS/dt</code> jest w A dokładnie
<b>27-krotne</b> (przy każdym poziomie S* = 0.1…0.6 stosunek wynosi
<b>@@RATE0@@ / @@RATE2@@ / @@RATE5@@</b> = 27.000).
</p>
<div class="note">
<b>Uczciwa uwaga:</b> w <i>pierwszym</i> tyknięciu stosunek ΔS_A/ΔS_B wynosi
tylko ≈ @@DS1RATIO@@ (A: @@DS1A@@, B: @@DS1B@@). To efekt logarytmicznej osobliwości
tempa dS/dt w pobliżu stanu czystego (dS/d|r| = −artanh|r| → ∞ przy |r| → 1).
Stosunek 27 odnosi się do tempa w dopasowanych punktach (granica ciągła) i do
kompresji czasowej — tak należy czytać „~27×”.
</div>
</div>

<div class="feat">
<span class="tag">CECHY 2–3</span>
<h3>Skumulowany czas T(n) = S(n); dyskretne schodki tyknięć</h3>
<img src="@@FIG3@@" alt="Czas = entropia, schodki">
<p>
Lewy panel: jedna realizacja kwantowego zegara (kolorowa linia schodkowa) na tle
wartości oczekiwanej S(n) (linia przerywana). Czas <b>nie płynie</b> — <i>skacze</i>
tyknięcie po tyknięciu; widać wyraźne schodki, nie ciągły przepływ. Prawy panel:
skumulowany czas vs entropia — wszystkie punkty leżą na prostej <b>T = S</b>
(nachylenie κ = @@KAPPA@@). To nie jest przybliżenie: średnia po 500 realizacjach
odbiega od S(n) o mniej niż 0.004. Wniosek: <b>czasu jest dokładnie tyle, ile
wyprodukowano entropii</b>; cały czas wszechświata mieści się w przedziale
[0, ln 2] (w jednostkach entropii).
</p>
</div>

<div class="feat">
<span class="tag">CECHA 4</span>
<h3>„Czkanie czasu”: przy niskiej produkcji entropii Δt_n → 0</h3>
<img src="@@FIG4@@" alt="Czkanie czasu">
<p>
Gdy produkcja entropii spada poniżej kwantu δs, zegar przestaje tykać:
<code>k_n = 0 ⇒ Δt_n = 0</code>. W ogonie ewolucji B (tyknięcia 40–200)
<code>Δt = 0</code> w <b>@@N_ZERO_B@@/160</b> tyknięciach, a najdłuższe zamrożenie
trwa <b>@@MAX_FREEZE_B@@ tyknięć</b> — czas stoi, po czym „czka” jednym skokiem.
Gorąca komórka A nasyca się błyskawicznie i <b>też</b> czka: po tyknięciu
nr @@LAST_A@@ czas jest praktycznie zamrożony (@@N_ZERO_A@@/27 zer w oknie 3–30).
To kwantowa natura czasu: czas nie płynie, gdy nie ma czego „mierzalnie” produkować.
</p>
</div>

<div class="feat">
<span class="tag">CECHA 5</span>
<h3>T_A/T_B: czas w gorącym otoczeniu płynie szybciej</h3>
<img src="@@FIG5@@" alt="Stosunek T_A/T_B">
<p>
Lewy panel: skumulowany czas ⟨T(n)⟩ (średnia po 400 realizacjach ± 1σ) — zegar
gorącej komórki A <b>zawsze wyprzedza</b> zimny B. Prawy panel: stosunek
T_A/T_B > 1 przez całą ewolucję (granica ciągła t→0: 27; pierwsze tyknięcie:
≈ @@DS1RATIO@@; nasycenie: → 1, bo oba zegary kończą na tym samym ln 2).
Innymi słowy: <b>ta sama ilość entropii powstaje w A 27× szybciej</b> — w gorącym
otoczeniu czas płynie szybciej, ale „ciepło śmierci” (nasycenie S = ln 2) też
przychodzi 27× szybciej.
</p>
</div>

<div class="feat">
<span class="tag">CECHY 6–7</span>
<h3>Dekoherencja: Tr(ρ²): 1 → 0.5; |r|: 1 → 0</h3>
<img src="@@FIG6@@" alt="Dekoherencja — czystość i wektor Blocha">
<p>
Czystość <code>Tr(ρ²) = ½(1 + |r|²)</code> spada od 1 (stan czysty) do
<b>0.5</b> (stan maksymalnie mieszany ½·𝟙): A: @@P_END_A@@, B: @@P_END_B@@ po
@@N_TICKS@@ tyknięciach. Długość wektora Blocha |r| maleje od 1 do 0
(A: @@R_END_A@@, B: @@R_END_B@@) — <b>zanik koherencji kwantowej</b>; widoczne
są gasnące oscylacje precesji (panel prawy-dół). Lewy-dół: fundamentalna
zależność Tr(ρ²) od S — ta sama dla obu temperatur (dynamika identyczna poza tempem).
</p>
</div>

<div class="feat">
<span class="tag">WIZUALIZACJA</span>
<h3>Sfera Blocha: spirala dekoherencji do środka</h3>
<img src="@@FIG7@@" alt="Sfera Blocha">
<p>
Trajektoria stanu na sferze Blocha (gradient koloru = bieg czasu). Gorąca A wykonuje
ledwie ułamek obrotu, po czym wpada do środka; zimna B zatacza kilka pełnych spiral
gasnącej precesji, zanim zbiegnie do punktu |r| = 0 — kwantowa koherencja umiera
w obu, ale w A 27× szybciej.
</p>
</div>

<!-- ==================== DEMO ==================== -->
<h2><span class="no">5.</span> Interaktywna symulacja zegara</h2>
<p>
Żywa realizacja kwantowego zegara (czysty JS + canvas, bez zewnętrznych bibliotek):
linia schodkowa T(n) na tle wartości oczekiwanej S(n). Przy niskiej produkcji
entropii zegar się zatrzymuje — obserwuj „czkanie” na ogonie ewolucji B.
</p>
<div class="demo">
  <canvas id="cv" width="960" height="400"></canvas>
  <div class="controls">
    <button id="btnA">A — gorące</button>
    <button id="btnB" class="on">B — zimne</button>
    <button id="btnPlay">▶ start</button>
    <button id="btnNew">⟲ nowa realizacja</button>
    <label>tempo: <input id="speed" type="range" min="2" max="80" value="16" style="vertical-align:middle"></label>
  </div>
  <div class="readout" id="readout"></div>
</div>

<h2><span class="no">6.</span> Wnioski kosmologiczne</h2>
<ul>
<li><b>Strzałka czasu = wzrost entropii.</b> „Kierunek” czasu nie jest postulatem —
wynika z monotoniczności S (unitalność, tw. Ando–Lindblada).</li>
<li><b>Czas jest relacyjny i dyskretny.</b> T(n) = S(n): czas to skumulowana entropia;
nie istnieje ciągły przepływ, są tylko tyknięcia (schodki).</li>
<li><b>Gorąco przyspiesza czas.</b> T_A/T_B > 1; stosunek tempa = (T_A/T_B)³ = 27.
W gorącej epoce (np. wczesny Wszechświat) zegary kosmiczne biły znacznie szybciej.</li>
<li><b>„Czkanie” i koniec czasu.</b> Gdy produkcja entropii spada poniżej kwantu δs,
czas staje; przy nasyceniu S = ln 2 zegar zatrzymuje się na stałe — „śmierć cieplna”
jest zarazem <i>końcem czasu</i>. Całkowity czas trwania komórki: ln 2 (w natach).</li>
<li><b>Dekoherencja jest motorem zegara.</b> To zanik kwantowej koherencji
(|r|: 1 → 0, Tr(ρ²): 1 → 0.5) produkuje entropię, która napędza czas.</li>
</ul>

<h2><span class="no">7.</span> Ograniczenia</h2>
<ul>
<li>Model jest <b>fenomenologiczną zabawką</b> — jedna komórka-kubit zamiast
pełnego widma pól; 27 to interpretacja (T_A = 3·T_B, s ∝ T³), łatwo zmienić
stosunek temperatur.</li>
<li>Rdzeń zakłada kąpiel nieskończenie gorącą (cel ln 2). Część II (poniżej)
usuwa to ograniczenie: skończona temperatura (R1), wiele kubitów (R2) i
sprzężenie zwrotne „zegar → tempo” (R3).</li>
<li>Wariacje: inna geometria stanu początkowego, γ_φ ≠ 2γ, ω ≠ 0.4, kwant δs,
κ — wszystkie są parametrami w <code>model_entropia.py</code>.</li>
</ul>

@@EXTENSIONS@@

<h2>Parametry i uruchomienie</h2>
<table>
<tr><th>parametr</th><th>wartość</th></tr>
@@PARAMS@@
</table>
<p>
Pliki (folder <code>entropia/</code>): <code>model_entropia.py</code> — symulacja
Lindblada + 7 figur + drukowane liczby kluczowe; <code>zrob_raport.py</code> — budowa
tego raportu; <code>figury/</code> — wykresy PNG; <code>raport.html</code> — ten dokument.
</p>
<div class="formula">python3 model_entropia.py   &nbsp;# figury + liczby kluczowe<br>
python3 zrob_raport.py      &nbsp;# regeneracja raport.html</div>

<div class="foot">
Model «ENTROPIA» v2.0 · czas = entropia · równanie Lindblada · tw. Ando–Lindblada
· tw. Spohna · relacyjny czas termiczny · rozszerzenia: T skończona (R1),
wiele kubitów (R2), sprzężenie zegar→tempo (R3) · 13 sierpnia 2026
</div>
</div>

@@EXT_SCRIPT@@

<script>
/* ============ interaktywna symulacja kwantowego zegara ============ */
(function(){
  var dS = {
    A: @@JS_A@@,
    B: @@JS_B@@
  };
  var N = dS.B.length;
  var LN2 = @@LN2@@;
  var ds = @@DS_Q@@;
  var col = { A: "#c0392b", B: "#2e86c1" };

  var cv = document.getElementById("cv");
  var ctx = cv.getContext("2d");
  var btnA = document.getElementById("btnA");
  var btnB = document.getElementById("btnB");
  var btnPlay = document.getElementById("btnPlay");
  var btnNew = document.getElementById("btnNew");
  var speed = document.getElementById("speed");
  var readout = document.getElementById("readout");

  var sys = "B", playing = false, timer = null, n = 0, T = 0, streak = 0, maxStreak = 0;

  function poisson(mu){
    var L = Math.exp(-mu), k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
  }

  function expectedT(){
    var s = 0, out = new Array(N);
    for (var i = 0; i < N; i++){ s += dS[sys][i]; out[i] = s; }
    return out;
  }
  var expT = expectedT();

  function draw(){
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var ml = 52, mr = 14, mt = 16, mb = 34;
    var x = function(i){ return ml + (W - ml - mr) * i / (N - 1); };
    var y = function(t){ return mt + (H - mt - mb) * (1 - t / (LN2 * 1.12)); };

    // siatka
    ctx.strokeStyle = "#223a4f"; ctx.fillStyle = "#7d9bb3"; ctx.font = "12px Consolas,monospace";
    for (var g = 0; g <= 4; g++){
      var tv = LN2 * g / 4, yy = y(tv);
      ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(W - mr, yy); ctx.stroke();
      ctx.fillText(tv.toFixed(2), 8, yy + 4);
    }
    ctx.fillText("ln 2", 14, y(LN2) + 4);
    ctx.fillText("0", 30, y(0) + 4);
    ctx.fillText("tyknięcie n", W / 2, H - 8);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("T(n) — czas = entropia [nat]", 0, 0); ctx.restore();

    // wartość oczekiwana S(n)
    ctx.strokeStyle = "#3f6a8c"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (var i = 0; i < N; i++){ var px = x(i), py = y(expT[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke(); ctx.setLineDash([]);

    // realizacja T(n) — schodki (steps-post)
    ctx.strokeStyle = col[sys]; ctx.lineWidth = 2.2;
    ctx.beginPath();
    var t = 0;
    for (var i = 0; i <= n; i++){
      var px = x(i), py = y(t);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      if (i < n && dS[sys][i] > 0){ var tn = t + dS[sys][i]; ctx.lineTo(x(i + 1), y(t)); t = tn; }
    }
    ctx.stroke();

    // znacznik bieżącego tyknięcia + wskaźnik zamrożenia
    ctx.fillStyle = "#f0f6fc";
    ctx.beginPath(); ctx.arc(x(Math.min(n, N - 1)), y(t), 4.5, 0, 7); ctx.fill();
    if (streak >= 3){
      ctx.fillStyle = "#f0c36d"; ctx.font = "13px Consolas,monospace";
      ctx.fillText("⏸ czas zamrożony: " + streak + " tyknięć", x(Math.min(n, N - 1)) + 12, y(t) - 10);
    }
  }

  function step(){
    if (n >= N){
      pause();
      readout.innerHTML = "⏹ KONIEC CZASU — nasycenie S = ln 2 (T = " +
        T.toFixed(4) + " nat). Naciśnij «⟲ nowa realizacja», by uruchomić ponownie.";
      draw(); return;
    }
    var mu = dS[sys][n] / ds;
    var k = poisson(mu);
    var dt = k * ds;
    if (k === 0){ streak++; maxStreak = Math.max(maxStreak, streak); }
    else { streak = 0; }
    T += dt;
    n++;
    readout.innerHTML =
      "tyknięcie n = " + n + "&nbsp;·&nbsp; ΔS_n = " + (mu * ds).toFixed(5) +
      "&nbsp;·&nbsp; kwanty k_n = " + k + "&nbsp;·&nbsp; Δt_n = " + dt.toFixed(3) +
      "&nbsp;·&nbsp; T(n) = " + T.toFixed(4) +
      "&nbsp;·&nbsp; najdł. zamrożenie: " + maxStreak +
      (streak >= 3 ? " &nbsp;<span class='frozen'>⏸ CZAS STOI</span>" : "");
    draw();
  }

  function setSys(s){
    sys = s;
    btnA.className = (s === "A") ? "on" : "";
    btnB.className = (s === "B") ? "on" : "";
    n = 0; T = 0; streak = 0; maxStreak = 0;
    expT = expectedT();
    draw();
  }
  function play(){ if (!playing){ playing = true; btnPlay.textContent = "⏸ pauza"; loop(); } }
  function pause(){ playing = false; btnPlay.textContent = "▶ start"; if (timer) clearTimeout(timer); timer = null; }
  function loop(){
    if (!playing) return;
    step();
    timer = setTimeout(loop, +speed.value);
  }

  btnA.onclick = function(){ setSys("A"); };
  btnB.onclick = function(){ setSys("B"); };
  btnPlay.onclick = function(){ playing ? pause() : play(); };
  btnNew.onclick = function(){ n = 0; T = 0; streak = 0; maxStreak = 0; draw(); };
  speed.onchange = function(){};

  setSys("B");
  // krótki autostart demo
  var aut = 0;
  var autot = setInterval(function(){
    if (playing){ clearInterval(autot); return; }
    step(); aut++;
    if (aut > 60){ clearInterval(autot); play(); }
  }, 30);
})();
</script>
</body>
</html>
"""

EXT_TEMPLATE = r"""
<div style="text-align:center;margin:56px 0 10px;color:#6b2f8e;font-weight:bold;letter-spacing:3px;font-size:15px">
— CZĘŚĆ II — ROZSZERZENIA MODELU —
</div>

<h2><span class="no">R1.</span> Skończona temperatura kąpieli — nasycenie poniżej ln 2</h2>
<p>
Kąpiel Gibbsa o skończonej temperaturze: <code>η = e^{−βΩ} ∈ (0,1]</code>
(<code>η = 1</code> ⇔ β = 0, kąpiel nieskończenie gorąca — rdzeń modelu).
Tempo emisji <code>a = 2γ/(1+η)</code>, absorpcji <code>b = 2γη/(1+η)</code>;
<code>a+b = 2γ</code> — tempo populacyjne bez zmian, zmienia się tylko <i>cel</i>:
</p>
<div class="formula">
ρ_eq = diag(1/(1+η), η/(1+η))  ⇒  S(∞) = H(1/(1+η)) < ln 2,&nbsp;
|r|∞ = (1−η)/(1+η),&nbsp; Tr(ρ²)∞ = (1+|r|∞²)/2
</div>
<p>
Mapa jest <b>nieunitalna</b> dla η < 1: tw. Ando–Lindblada nie gwarantuje już
monotoniczności S, ale produkcja entropii względem stanu stacjonarnego
<code>σ = −d/dt·S(ρ‖ρ_eq) ≥ 0</code> (tw. Spohna) pozostaje nieujemna.
Kompresja czasowa 27× <b>przetrwa</b> — γ skaluje tempo, a równowaga zależy tylko
od η: <code>S_η(t;γ_A) = S_η(27·t;γ_B)</code> (błąd 2·10⁻¹⁶).
</p>
<img src="@@FIGR1@@" alt="R1 — skończona temperatura" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>η</th><th>βΩ</th><th>S(∞) [nat]</th><th>Tr(ρ²)∞</th><th>|r|∞</th>
<th>t99% (B)</th><th>t99% (A)</th><th>stos.</th></tr>
@@R1ROWS@@
</table>
<div class="note">
<b>Zimna kąpiel potrafi „cofnąć” entropię.</b> Gdy η < 1/3 (βΩ > ln 3, kąpiel
bardziej spolaryzowana niż stan początkowy), entropia najpierw <b>przewyższa
plateau</b>: S_max = @@R1_SMAX@@ przy t ≈ @@R1_TMAX@@, a potem opada do
S(∞) = @@R1_SEQ@@ (Δ = @@R1_DELTA@@). To efekt „pompowania” populacji przez
zimną kąpiel — fizycznie poprawny, niemożliwy dla map unitalnych (η = 1).
</div>

<h2><span class="no">R2.</span> Wiele kubitów — ekstensywność vs korelacje</h2>
<p>
<b>Niezależne kąpiele</b>: każdy kubit ma własną kąpiel — stan pozostaje
produktowy, entropia jest <b>ekstensywna</b>: <code>S_total = N·S₁</code>,
<code>S(∞) = N·ln 2</code> (weryfikacja numeryczna: błąd 0.0). <b>Wspólna kąpiel
kolektywna</b> (jumpy Dickego <code>S± = Σσ±^i</code>): termalizuje tylko sektor
osiągalny z warunku symetrii — dla N = 2 startującego z |11⟩ jest to tryplet
(3 stany):
</p>
<div class="formula">
S(∞) = ln 3 &lt; 2·ln 2&nbsp; ⇒&nbsp; deficyt ln(4/3) ≈ 0.288 nat — entropia
„zamknięta w korelacjach”: I = S(A)+S(B)−S(AB) = @@R2_MI11@@ nat
</div>
<p>
Start z |10⟩ = (|T₀⟩ + |S⟩)/√2: <b>singlet Bell |S⟩ jest ciemny</b> dla kanału
kolektywnego (jump-y S± go nie ruszają, γ_φ = 0) — połowa stanu nigdy nie
termalizuje: S(∞) = ½·ln 12 = 1.2425 nat, czystość 1/3, a przetrwała korelacja
to informacja wzajemna I → ln(2/√3) ≈ 0.144 nat (dokładnie @@R2_MI10@@).
Uwaga: <b>negatywność = 0</b> — mimo członu singletowego mieszanina jest
separowalna; korelacja przeżywa wiecznie, ale nie jest destylowalna. Kąpiel
kolektywna termalizuje też szybciej: t90% ≈ 98 tyknięć vs 177 dla niezależnych.
</p>
<img src="@@FIGR2@@" alt="R2 — wiele kubitów" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>scenariusz (N=2, γ_φ=0)</th><th>S(∞) [nat]</th><th>czystość∞</th>
<th>negatywność</th><th>inf. wzajemna I</th><th>t90% [tyknięcia]</th></tr>
@@R2ROWS@@
</table>

<h2><span class="no">R3.</span> Sprzężenie „zegar → tempo” — samouzależniona dynamika</h2>
<p>
Tempo zależy od <i>odczytu zegara</i> (skumulowanej entropii): <code>γ_eff(n) = γ₀·fb(T_n/T_scale)</code>.
Ponieważ zegar = entropia, sprzężenie jest wewnętrzne — „ekspansja” (chłodzenie:
tempo maleje z wiekiem) lub „kontrakcja” (przyspieszanie: tempo rośnie):
</p>
<div class="formula">
fb(u) = 1&nbsp; (stały)&nbsp; |&nbsp; fb(u) = 1/(1+αu)&nbsp; (chłodzenie)&nbsp; |&nbsp;
fb(u) = 1+αu&nbsp; (przyspieszanie),&nbsp; u = T_n/T_scale
</div>
<p>
Kluczowa własność: skoro fb zależy od S (a nie od zewnętrznego czasu), oba zegary
przechodzą przez te same poziomy entropii — <b>kompresja 27× nie zostaje
zepsuta</b>: t½_A/t½_B = <b>27.0 dokładnie</b> w każdym scenariuszu, a
S_A(t) ≡ S_B(27t) w czasie ciągłym z błędem @@R3_COMP@@. Sprzężenie zmienia
jednak <i>rozkład</i> tyknięć: chłodzenie rozciąga późną ewolucję (t½: 3.0 → 5.5),
przyspieszanie skraca (3.0 → 2.2); przy nasyceniu każdy scenariusz zamraża czas
(czknie) — najdłuższe zamrożenie rośnie dla przyspieszania (147 tyknięć),
bo wchodzi w nasycenie najwcześniej.
</p>
<img src="@@FIGR3@@" alt="R3 — feedback" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>scenariusz</th><th>t½ (A)</th><th>t½ (B)</th><th>stos.</th>
<th>czkanie Δt=0 (40–200)</th><th>najdł. zamrożenie</th></tr>
@@R3ROWS@@
</table>
<p>
<b>Interaktywna wersja sprzężenia</b> (czysty JS): suwak α zmienia siłę
„chłodzenia”, checkbox dodaje „przyspieszanie”. Krzywe to S(n) z
samouzależnionym tempem — obserwuj, jak sprzężenie zagina historię entropii.
</p>
<div class="demo">
  <canvas id="cv2" width="960" height="400"></canvas>
  <div class="controls">
    <label>α (siła chłodzenia): <input id="alpha" type="range" min="0" max="3" step="0.05" value="2" style="vertical-align:middle"></label>
    <label><input type="checkbox" id="accel"> pokaż przyspieszanie (α=1)</label>
  </div>
  <div class="readout" id="readout2"></div>
</div>

<h2><span class="no">R4.</span> N=3: jawne ciemne sektory j=1/2 (subradiantne)</h2>
<p>
Dla 3 kubitów we wspólnej kąpieli przestrzeń rozkłada się na sektory spinu
całkowitego: <b>j = 3/2</b> (4 stany symetryczne) oraz <b>dwie kopie j = 1/2</b>
(2×2 stany). Jumpy kolektywne S± nie mieszają sektorów (S± komutują z S²),
więc każdy sektor termalizuje <i>wewnątrz siebie</i> do stanu maksymalnie
mieszanego o swoim wymiarze:
</p>
<div class="formula">
j = 3/2: S(∞) = ln 4 = @@R4_LN4@@ &nbsp;·&nbsp;
j = 1/2 (każda kopia): S(∞) = ln 2 — „czapka” subradiantna
</div>
<p>
<b>Jawne stany j=1/2 (kopia B)</b> to singlet⊗kubit: |1⟩⊗|S⟩₂₃ = (|101⟩−|011⟩)/√2
oraz |0⟩⊗|S⟩₂₃ = (|100⟩−|010⟩)/√2. Nie są one w pełni ciemne (dla N=3 nie istnieje
sektor j=0, który byłby zerowany przez S±), ale są <b>subradiantne</b>: kąpiel
kolektywna nigdy nie wyprowadza ich poza 1 bit — populacja na kopii B pozostaje
równa 1 z dokładnością 10⁻⁸ przez cały przebieg. Dla porównania: stan |111⟩
(czysty j=3/2) osiąga ln 4 = @@R4_LN4@@, a |100⟩ (mieszanina sektorów:
1/3 symetryczny + 2/3 kopia j=1/2) nasyca się na ln108/3 = @@R4_LN108_3@@.
Deficyt sektora symetrycznego vs 3 niezależne kąpiele wynosi dokładnie
<b>ln 2 = @@R4_DEF@@</b> nat.
</p>
<img src="@@FIGR4@@" alt="R4 — sektory N=3" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>stan początkowy (N=3, γ_φ=0)</th><th>S(∞) [nat]</th><th>czystość∞</th>
<th>min. populacja na kopii B</th><th>t90% [tyknięcia]</th></tr>
@@R4ROWS@@
</table>

<h2><span class="no">R5.</span> Pełna entropia makro: N·ln 2 vs ln(N+1)</h2>
<p>
„Pełna entropia makro” N komórek-kubitów: przy <b>niezależnych kąpielach</b>
entropia jest dokładnie ekstensywna, S(∞) = <b>N·ln 2 = ln(2^N)</b> (wymiar
przestrzeni Hilberta!), a czas nasycenia jest <b>niezależny od N</b> (t90% = 177
dla każdego N — każda komórka termalizuje tak samo). Przy <b>kąpieli wspólnej</b>
startującej z |1…1⟩ (czysty sektor symetryczny j=N/2) osiągalna jest tylko
podprzestrzeń symetryczna o wymiarze N+1:
</p>
<div class="formula">
S(∞) = ln(N+1) &nbsp;⇒&nbsp; deficyt = N·ln 2 − ln(N+1):&nbsp;
N=2: ln(4/3), N=3: ln 2, N=4: ln(16/5),…
</div>
<p>
Kąpiel kolektywna jest przy tym <b>szybsza</b> (superradiancja Dickego): t90%
maleje z N (100, 98, 96, 94 tyknięcia), bo kolektywne jumpy S± działają na
wszystkie kubity naraz. Weryfikacja ekstensywności dla N=16: błąd 2·10⁻¹³.
</p>
<img src="@@FIGR5@@" alt="R5 — entropia makro" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>N</th><th>niezależne S(∞)</th><th>kolektywne S(∞)</th><th>deficyt</th>
<th>t90% niezależne</th><th>t90% kolektywne</th></tr>
@@R5ROWS@@
</table>

<h2><span class="no">R6.</span> Gorący Wielki Wybuch — warunek początkowy w R3</h2>
<p>
Zamiast zimnego, czystego startu (S = 0) Wszechświat zaczyna <b>gorący</b>:
ρ(0) termiczne przy η₀ = 0.95 → <b>S(0) = @@R6_S0@@ ≈ ln 2</b>
(maksymalnie gorący start), a kąpiel jest zimna (η_B = 0.15,
S(∞) = @@R6_SEQ@@). Wtedy entropia <b>maleje</b> — Wszechświat się ochładza.
Ponieważ zegar kosmiczny to wciąż T(n) = S(n) (definicja podpisana),
<b>czas płynie wstecz</b>: Δt_n = κ·ΔS_n &lt; 0.
</p>
<div class="formula">
T(n) = S(n):&nbsp; S: @@R6_S0@@ → @@R6_SEQ@@&nbsp; ⇒&nbsp;
budżet czasu wstecz |S(0) − S(∞)| = @@R6_BUDZET@@ nat
</div>
<p>
R3 (sprzężenie „zegar → tempo”) ma tu naturalną realizację: <code>u = (S(0) − S(n))/t_scale</code>
— im więcej entropii już „rozładowano”, tym wolniejsze tempo (chłodzenie,
fb = 1/(1+2u)). Chłodzenie spowalnia ochładzanie: t½ (połowa drogi w dół)
= @@R6_THC@@ (stałe γ) vs @@R6_THF@@ (chłodzenie). Zegar wstecz też „czka”:
przy |ΔS_n| &lt; δs pada k_n = 0 i Δt_n = 0 — w oknie 200–500 aż @@R6_NZ@@/@@R6_NT@@
tyknięć stoi w miejscu. Kontrola krzyżowa zamknięcia z pełną mapą Lindblada:
błąd 9·10⁻¹⁵.
</p>
<img src="@@FIGR6@@" alt="R6 — gorący Wielki Wybuch" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<p>
<b>Interaktywny zegar wstecz</b> (czysty JS): entropia maleje (linia przerywana),
a realizacja kwantowego zegara skacze <b>w dół</b> — ujemne tyknięcia Δt.
Obserwuj „czkanie” przy zbliżaniu się do zimnej równowagi.
</p>
<div class="demo">
  <canvas id="cv3" width="960" height="400"></canvas>
  <div class="controls">
    <button id="btnPlay3">▶ start</button>
    <button id="btnNew3">⟲ nowa realizacja</button>
    <label>tempo: <input id="speed3" type="range" min="2" max="80" value="16" style="vertical-align:middle"></label>
  </div>
  <div class="readout" id="readout3"></div>
</div>

<h2><span class="no">R7.</span> Kąpiel kolektywna dla losowych (nie-symetrycznych) stanów</h2>
<p>
Kąpiel kolektywna termalizuje <i>wewnątrz sektorów spinu całkowitego</i>. Dla stanu
czystego o populacjach <code>p_{j,c}</code> w sektorach/kopiach entropia nasycenia
leży między <b>ln(N+1)</b> (stan w pełni symetryczny) a <b>N·ln 2</b> (pełna
termalizacja). Kluczowy efekt: <b>koherencje między kopiami tego samego j
przeżywają</b> (kopie mają identyczną dynamikę drabiny), więc kąpiel kolektywna
<i>sama</i> nigdy nie osiąga pełnej entropii — entropia pozostaje „zablokowana”.
</p>
<div class="formula">
γ_φ = 0:&nbsp; |100⟩: koherencja A↔B = @@R7_COH@@ = √(p_A·p_B) — maksymalna, przeżywa<br>
γ_φ = γ:&nbsp; lokalna dekoherencja miesza sektory ⇒ <b>S(∞) → N·ln 2 dokładnie</b>
</div>
<p>
Numerycznie (N=3): stan |111⟩ (symetryczny) nasyca się na ln 4 = 1.3863; stan
produktowy |100⟩ na 1.5607; losowe stany Haar na 1.61–1.99 (między ln 4 a 3·ln 2).
Po włączeniu lokalnej dekoherencji γ_φ = γ <b>wszystkie</b> stany dążą do dokładnie
3·ln 2 = 2.0794 — pełna termalizacja odblokowana. Dla N=4: |1111⟩ → ln 5 = 1.6094,
losowe → 2.30–2.34, a przy γ_φ = γ → dokładnie 4·ln 2 = 2.7726.
</p>
<img src="@@FIGR7@@" alt="R7 — losowe stany" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>stan (N=3)</th><th>S(∞) γ_φ=0</th><th>S(∞) γ_φ=γ</th></tr>
@@R7ROWS@@
</table>
<table>
<tr><th>stan (N=4)</th><th>S(∞) γ_φ=0</th><th>S(∞) γ_φ=γ</th></tr>
@@R7ROWS4@@
</table>

<h2><span class="no">R8.</span> Cykl: Wielki Wybuch → ekspansja → ochłodzenie → Wielki Kolaps</h2>
<p>
Kąpiel o <b>oscylującej temperaturze</b> η(n) = 1 − (1−η_min)·sin²(π·n/n_cyc)
odtwarza kosmiczny cykl: start w gorącym Wielkim Wybuchu (S ≈ ln 2), ochłodzenie
podczas „ekspansji” (S maleje do @@R8_SMIN@@ — maksymalna ekspansja), po czym
ogrzewanie przy kolapsie (S wraca do ln 2). Ponieważ zegar to T(n) = S(n),
<b>czas płynie wstecz przez @@R8_FRAK@@% cyklu</b>: ujemne Δt podczas ochładzania.
</p>
<div class="formula">
budżet czasu wstecz = @@R8_BUDZET@@ nat&nbsp; ·&nbsp; upływ całkowity τ = @@R8_TAU@@ =
2·budżet (wskazówka „upływu” zawsze do przodu)&nbsp; ·&nbsp; przy zwrocie strzałki
ΔS → 0: kwantowe zamrożenia w @@R8_ZAMROZ@@/30 tyknięciach
</div>
<p>
Mamy dwie wskazówki zegara kosmicznego: <b>T(n) = S(n) − S(0)</b> (wskazówka
entropii — dwustronna, wraca do zera na końcu cyklu: <i>czas jako zamknięta
pętla</i>) oraz <b>τ(n) = Σ|ΔS|</b> (wskazówka upływu — zawsze rośnie). Wszechświat
cykliczny: entropia wraca do ln 2, czas zamyka się w pętli, a „Wielki Kolaps”
jest lustrzanym odbiciem „Wielkiego Wybuchu”.
</p>
<img src="@@FIGR8@@" alt="R8 — cykl" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<p>
<b>Interaktywny cykl</b> (czysty JS): suwak η_min zmienia głębokość ochłodzenia —
obserwuj, jak entropia (i czas) opada, a potem wraca; przy zwrocie zegar czka.
</p>
<div class="demo">
  <canvas id="cv4" width="960" height="400"></canvas>
  <div class="controls">
    <label>η_min (głębokość ochłodzenia): <input id="etamin" type="range" min="0.05" max="0.9" step="0.01" value="0.15" style="vertical-align:middle"></label>
    <label><input type="checkbox" id="twohands"> pokaż dwie wskazówki (T i τ)</label>
  </div>
  <div class="readout" id="readout4"></div>
</div>

<h2><span class="no">R9.</span> Kwantowy zegar — super-twarda wersja: czas jako operator</h2>
<p>
Dotychczasowy zegar był <i>klasyczny</i> (stochastyczne tyknięcia). W wersji
super-twardej sam zegar jest <b>kwantowy</b>: oscylator (próżnia), do którego
„kranik” σ₋⊗b† kopiuje każdą de-ekscytację wszechświata-kubitu (jednokierunkowo).
Wskazanie zegara to <b>operator liczby kwantów</b> z pełnym rozkładem p_n:
</p>
<div class="formula">
⟨n⟩(t) — średni odczyt (śledzi S_sys: czas = entropia, kwantowo)&nbsp; ·&nbsp;
Δn(t) — nieoznaczoność czasu&nbsp; ·&nbsp; Δn/⟨n⟩ — względna nieoznaczoność
</div>
<p>
Wyniki (γ_t = @@R9_GT@@): S_sys(∞) = @@R9_SEND@@ (odchylenie od ln 2: @@R9_DEV@@ —
zegar lekko zaburza wszechświat), ⟨n⟩(∞) = @@R9_NBAR@@, Δn = @@R9_DN@@,
Δn/⟨n⟩ = @@R9_REL@@, a korelacja I(wszechświat;zegar) = @@R9_I@@ — wszechświat
i jego zegar są kwantowo skorelowane (czas relacyjny à la Page–Wootters).
<b>Kompromis zegara kwantowego</b>: silniejszy zegar mierzy czas precyzyjniej
(Δn/⟨n⟩ maleje), ale mocniej zaburza ewolucję (|S(∞) − ln 2| rośnie) — kwantowa
zasada nieoznaczoności dla czasu (duch Saleckera–Wignera). W granicy słabego
sprzężenia γ_t → 0 zegar staje się nieperturbujący i „klasyczny” (S → ln 2,
korelacje → 0).
</p>
<img src="@@FIGR9@@" alt="R9 — kwantowy zegar" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>γ_t (siła zegara)</th><th>|back-action| = |S(∞) − ln 2|</th><th>Δn/⟨n⟩ (nieoznaczoność)</th></tr>
@@R9ROWS@@
</table>

<h2><span class="no">R10.</span> Kwantowy zegar z koherencjami — faza czasu i jej dekoherencja</h2>
<p>
Zegar kwantowy (R9) startował w próżni. Teraz startujemy go w <b>stanie koherentnym
|α=1.5⟩</b> — zegar ma <b>fazę</b>: koherencje ⟨n|ρ_c|n+1⟩ ≠ 0 i lepszą precyzję
(Δn/⟨n⟩ = @@R10_RELCOH@@ vs @@R10_RELVAC@@ dla próżni). Kluczowy efekt:
<b>stymulowane kopiowanie</b> — kwanty już obecne w zegarze wzmacniają kranik σ₋⊗b†
(czynnik √(n+1)), więc back-action rośnie: |S(∞) − ln 2| = @@R10_SCOH@@ vs
@@R10_SVAC@@ (próżnia), czyli <b>@@R10_STIM@@× mocniej</b>. Zegar „z kwantami”
mierzy precyzyjniej, ale mocniej zaburza Wszechświat — wewnętrzny stan zegara
ustawia jego pozycję na kompromisie z R9.
</p>
<div class="formula">
koherentny κ=0:&nbsp; koherencje = @@R10_COHCOH@@ (przeżywają), I(wszechświat;zegar) = @@R10_ICOH@@<br>
+ dekoherencja zegara κ·D[b†b] (κ=0.3):&nbsp; koherencje → @@R10_COHKAP@@ ≈ 0,
I → @@R10_IKAP@@ — <b>czas klasyczny</b> (diagonalny), a fizyka wszechświata
(S∞, ⟨n⟩, Δn) <b>bez zmian</b>
</div>
<p>
Dekoherencja zegara (κ·D[b†b]) niszczy kwantową fazę czasu i korelacje
wszechświat–zegar, <b>nie zmieniając dynamiki wszechświata</b> (S∞ identyczne).
To mechanizm, przez który „czas kwantowy” staje się „czasem klasycznym”:
pomiar/zewnętrzne otoczenie niszczy koherencje wskazań, a wskazówka zegara
pozostaje ta sama.
</p>
<img src="@@FIGR10@@" alt="R10 — koherencje zegara" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>scenariusz (γ_t = 0.01)</th><th>S(∞) [nat]</th><th>odch. od ln 2</th>
<th>⟨n⟩∞</th><th>Δn/⟨n⟩</th><th>koherencje</th><th>I(wszechświat;zegar)</th></tr>
@@R10ROWS@@
</table>

<h2><span class="no">R11.</span> Od blokady do pełnej termalizacji — prawo τ ∝ 1/γ_φ</h2>
<p>
W R7 widzieliśmy, że przy γ_φ = 0 entropia pozostaje „zablokowana” w koherencjach
sektorowych. Pytanie R11: <b>co się dzieje przy pośrednim γ_φ?</b> Odpowiedź jest
subtelna: dla <b>każdego γ_φ &gt; 0</b> entropia w końcu osiąga N·ln 2 (blokada przy
dokładnie γ_φ = 0 jest idealna), a czas odblokowania skaluje się jak
<b>τ90 ∝ 1/γ_φ</b> (dopasowanie: nachylenie <b>@@R11_M3@@</b> dla N=3) — lokalna
dekoherencja miesza sektory z tempem ~ γ_φ.
</p>
<div class="formula">
τ90:&nbsp; γ_φ=1e-4 → @@R11_T90_1E4@@;  γ_φ=1e-3 → @@R11_T90_1E3@@;
γ_φ=1e-2 → @@R11_T90_1E2@@;  γ_φ=0.3 → @@R11_T90_3E-1@@ (podłoga = czas kolektywny)
</div>
<p>
Blokada przy γ_φ = 0 (N=3): |111⟩: S(∞) = @@R11_B111@@ (zablokowane ln 2),
|100⟩: @@R11_B100@@, losowy: @@R11_BLOS@@. Dla N=4 dopasowanie jest płytsze
(@@R11_M4@@), bo losowy stan startuje blisko celu (plateau ~2.3 vs 4·ln 2 = 2.77),
więc τ90 dominuje czas kolektywny. Pełną termalizację odblokowuje dopiero
lokalna dekoherencja — <b>czas potrzebuje „brudu” (dekoherencji), by dotrzeć
do pełnej entropii</b>.
</p>
<img src="@@FIGR11@@" alt="R11 — odblokowanie" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<p>
<b>Interaktywna wersja</b> (czysty JS): suwak wybiera γ_φ — krzywa S(n) (stan losowy
N=3) pokazuje dwuetapową relaksację: szybkie dojście do plateau, potem wolne
odblokowanie w skali 1/γ_φ.
</p>
<div class="demo">
  <canvas id="cv5" width="960" height="400"></canvas>
  <div class="controls">
    <label>γ_φ: <input id="gphi" type="range" min="0" max="7" step="1" value="4" style="vertical-align:middle"></label>
    <span id="gphi_label" style="font:13px Consolas,monospace;color:#f0c36d"></span>
  </div>
  <div class="readout" id="readout5"></div>
</div>

<h2><span class="no">R13.</span> Grawitacyjna produkcja entropii — czas bez końca</h2>
<p>
Do tej pory kąpiel była jedna; entropia nasycała się (ln 2) i czas zamierał.
Teraz kubit łączymy z <b>dwoma kąpielami</b>: gorącym promieniowaniem
(η_r = @@R13_ETA_R@@, γ_r = @@R13_GAMMA_R@@) i zimną „kąpielą grawitacyjną"
(η_g = @@R13_ETA_G@@, γ_g = @@R13_GAMMA_G@@). Energia płynie
promieniowanie → kubit → grawitacja; stan stacjonarny jest <b>nie-równowagowy
(NESS)</b>:
</p>
<div class="formula">
S(NESS) = @@R13_SNESS@@ &lt; ln 2&nbsp; ·&nbsp; σ_NESS = σ_prom + σ_graw = @@R13_SIG@@
(stałe, tw. Spohna)&nbsp; ·&nbsp; σ·τ/tyknięcie = @@R13_SIGTICK@@
</div>
<p>
Konsekwencja dla zegara jest kosmologiczna: zwykły zegar T(n) = S(n) nasyca się
(całkowity czas = ln 2), ale <b>zegar grawitacyjny</b>
<code>T_graw(n) = Σ_k σ_k·τ</code> rośnie <b>liniowo w nieskończoność</b> —
po 400 tyknięciach Σσ·τ = @@R13_SUM@@ ≈ <b>@@R13_RATIO@@ × ln 2</b> i nie ma końca.
Grawitacja (jako zimny rezerwuar, do którego spływa entropia) utrzymuje
produkcję entropii na stałym poziomie — <b>czas nigdy nie zamiera</b>, nawet gdy
lokalna termalizacja doszła do równowagi. To zabawkowy odpowiednik idei, że
struktury grawitacyjne (czarne dziury, kondensacja) produkują entropię wciąż na
nowo. Im silniejsza grawitacja (γ_graw), tym większe σ_NESS — grawitacja
przyspiesza bieg czasu.
</p>
<img src="@@FIGR13@@" alt="R13 — grawitacja" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>γ_graw (siła grawitacji)</th><th>σ_NESS [nat/j. czasu]</th></tr>
@@R13ROWS@@
</table>

<h2><span class="no">R15.</span> Dekoherencja zegara jako strażnik historii Wszechświata</h2>
<p>
<b>Teza (demonstrowana numerycznie w modelu).</b> W połączonym stanie ρ_SC
splot układ–zegar nie jest jednoznaczny: jawny Hamiltonian
<code>H_int = g(σ₋⊗b† + σ₊⊗b)</code> (Jaynes–Cummings, g = @@R15_G@@) <b>splątuje</b>
Wszechświat z zegarem i <b>rozmywa czas</b> — tworzy koherencje wskazań
(superpozycję „która godzina?”). Człon dekoherencji zegara <code>κ·D[b†b]</code>
nieustannie <b>rzutuje stan na oś liczbową</b> (baza Focka — einselected pointer
basis): zamienia kwantowe „czkanie” (dyskretne zdarzenia kranika σ₋⊗b†,
k_n ~ Poisson) w <b>mierzalny, nieodwracalny przyrost drogi czasu
τ = ⟨n⟩·δs</b>.
</p>
<div class="formula">
punkt zwrotny (nasycenie, ΔS → 0, n ≈ @@R15_NTURN@@):<br>
κ = 0:&nbsp; koherencje @@R15_K0_COH@@, rozmycie czasu @@R15_K0_OFF@@, I(S;C) = @@R15_K0_I@@ — czas KWANTOWY<br>
κ = 0.5:&nbsp; koherencje @@R15_K5_COH@@, rozmycie @@R15_K5_OFF@@ (<b>@@R15_SUP@@× mniej</b>), I = @@R15_K5_I@@ — czas KLASYCZNY
</div>
<p>
Mechanizm ma trzy obserwowalne skutki, wszystkie potwierdzone numerycznie:
</p>
<ul>
<li><b>Rzutowanie na oś liczbową.</b> W punkcie zwrotnym macierz zegara ρ_c bez κ
ma poza-przekątną strukturę (koherencje wskazań — rozmycie @@R15_K0_OFF@@);
z κ = 0.5 jest diagonalna (rozmycie @@R15_K5_OFF@@, supresja @@R15_SUP@@×).
To dokładnie einselection Zureka: środowisko (κ) wybiera bazę Focka jako
„klasyczną oś czasu”.</li>
<li><b>Nieodwracalność.</b> κ niszczy informację o fazie: S(zegar) rośnie
(@@R15_K0_SCL@@ → @@R15_K5_SCL@@) — mapa na zegar nie jest odwracalna, zapis
przeszłości nie może zostać cofnięty. Zapis ⟨n⟩ pozostaje <b>monotoniczny
nie malejący</b> w każdym scenariuszu.</li>
<li><b>Korelacja klasyczna zamiast kwantowej.</b> I(S;C) spada
(@@R15_K0_I@@ → @@R15_K5_I@@): kwantowa korelacja splątania zostaje zastąpiona
klasycznym zapisem — historia Wszechświata jest <b>trwale zapamiętana</b> w
klasycznym ciągu wskazań zegara, nie w kruchej superpozycji.</li>
</ul>
<p>
<b>Jak czytać „udowadnia”.</b> To demonstracja w ramach zabawki, nie twierdzenie
matematyczne: pokazujemy, że w modelu κ decyduje o klasyczności i
nieodwracalności zapisu czasu w punkcie zwrotnym (gdzie ΔS → 0 i bez κ czas
byłby kwantowo rozmyty). Mechanizm opiera się na standardowej fizyce
pomiaru (dekoherencja w bazie wskazań, Zurek 1981; ciągły pomiar / kwantowe
Zeno w granicy dużego κ) — ale ilościowe liczby (supresja @@R15_SUP@@×,
I: @@R15_K0_I@@ → @@R15_K5_I@@) pochodzą z pełnej ewolucji Lindblada ρ_SC
tego modelu.
</p>
<img src="@@FIGR15@@" alt="R15 — strażnik historii" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Kosmiczna rola κ.</b> Bez dekoherencji zegara w punkcie zwrotnym czas
pozostaje superpozycją „która godzina?” — historia wieloznaczna. Z κ czas
jest ciągiem klasycznych, nieodwracalnych przyrostów τ: „czkanie” (dyskretne
kwanty kranika) zostaje zamienione na mierzalną drogę czasu, a przeszłość
Wszechświata jest zabezpieczona na stałe. To zabawkowy odpowiednik idei, że
obserwowalny, jednokierunkowy czas wymaga środowiska, które go „mierzy”.
</div>

<h2><span class="no">R16.</span> Formalizm relacyjny po recenzji — λ → S → τ</h2>
<p>
<b>Rewizja założeń.</b> Zamiast dosłownej identyfikacji T ≡ S model przyjmuje
schemat trzypoziomowy recenzji: <b>λ</b> (parametr ewolucji) → <b>S</b>
(informacja podukładu) → <b>τ</b> (czas obserwatora), z funkcjonałem czasu:
</p>
<div class="formula">
dτ_A/dλ = α·[Ṡ_A^prod + η·I(A:E)]
</div>
<p>
η = 0 odtwarza stary zegar entropii (przypadek szczególny); η &gt; 0 dołącza
człon korelacyjny — informacja wzajemna między układem a jego rejestrem
(w modelu: system-zegar) też napędza czas. Numerycznie (zegar kwantowy R9):
τ(η=0) = @@R16_TAU_ENT@@ vs τ(η=0.5) = @@R16_TAU_REL@@ — korelacje dodają
budżetu czasu ponad samą entropię. Przy równowadze τ̇ → 0: <b>zegar entropowy
staje, ale to nie jest koniec czasu fizycznego</b> (rozróżnienie recenzji §2) —
mikroskopowa ewolucja może trwać (patrz R17: singlet precesuje przy S = 0).
</p>
<p>
<b>„27×” jako predykcja warunkowa.</b> Z prawa s ∝ T³ samo nie wynika τ_A/τ_B =
27 — potrzebny jest postulat łączący tempo zegara z entropią. Model rozróżnia
dwie gałęzie: gałąź s (dτ ∝ s ∝ T³, γ ∝ T³) daje <b>τ_A/τ_B = @@R16_S@@
dokładnie</b> przy dopasowanych poziomach; gałąź Ṡ (dτ ∝ Ṡ) daje inne liczby
(np. @@R16_SDOT1@@ w pierwszym tyknięciu). Uogólnienie: przy γ ∝ T^p stosunek
wynosi 3^p — <b>test falsyfikacyjny</b>: dwa identyczne zegary w kąpielach A i B
mierzą τ_A/τ_B, co wyznacza p z danych.
</p>
<img src="@@FIGR16@@" alt="R16 — formalizm relacyjny" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R17.</span> Laboratoryjny test — jasny ↔ ciemny zegar entropowy</h2>
<p>
Najciekawsze przewidywanie recenzji: jeżeli przejście układu do sektora
subradiacyjnego spowalnia entropiczny zegar (Γ↓ ⇒ Ṡ_obs↓ ⇒ τ̇↓), mamy test
laboratoryjny kluczowego założenia ENTROPII. W modelu (N=2, kąpiel kolektywna,
formuła Δτ_n = τ₀·ΔS_n/ΔS_ref):
</p>
<div class="formula">
singlet (ciemny):&nbsp; τ = @@R17_TAU_S@@ (zegar MILCZY, Γ_dark = 0)&nbsp; ·&nbsp;
|11⟩ (jasny):&nbsp; τ = @@R17_TAU_11@@&nbsp; ·&nbsp;
tempo ⟨Ṡ⟩: |11⟩ = @@R17_RATE11@@, |10⟩ = @@R17_RATE10@@, singlet = 0
</div>
<p>
<b>Czysty test</b>: stan początkowy ρ(p) = (1−p)|T₀⟩⟨T₀| + p|S⟩⟨S| — tempo zegara
spada liniowo z frakcją ciemną p:
</p>
<table>
<tr><th>p (frakcja ciemna)</th><th>⟨Ṡ⟩ (tempo zegara)</th></tr>
@@R17_RATES@@
</table>
<p>
<b>Zegar stoi, fizyka trwa</b> (rozdział recenzji §2): czysty singlet ewoluuje
unitarnie (precesja; fidel. ze stanem początkowym → @@R17_FID@@), a mimo to
S = 0 i zegar entropowy nie tyka ani razu. Z kolei stan |10⟩ = (|T₀⟩+|S⟩)/√2
pokazuje <b>pamięć</b>: po wygaśnięciu jasnej części τ̇ → 0, a entropia zostaje
zamrożona z informacją wzajemną I(A;B) = ln(2/√3) = @@R17_MI10@@ nat w ciemnym
sektorze — dokładnie „informacja → sektor ciemny → wydłużona pamięć” recenzji.
</p>
<img src="@@FIGR17@@" alt="R17 — test laboratoryjny" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Testowalność.</b> Subradiancja jest zmierzona w zimnych atomach (PRL 116,
083601, 2016; PRL 128, 203601, 2022). Predykcja modelu: zegar entropowy
sprzężony z układem wchodzącym w sektor subradiacyjny zwalnia (Γ_dark ≪
Γ_bright ⇒ τ̇↓) — mierzalne jako spowolnienie produkcji entropii/fluorescencji
przy przejściu jasny→ciemny. To test, który można by postawić w istniejących
układach Dicke’a z ciągłym monitoringiem emisji.
</div>

<h2><span class="no">R18.</span> ENTROPIA-1.1 — pełna symulacja N = 2..100</h2>
<p>
Skalowanie modelu: rozwiązujemy równanie Lindblada w <b>bazie Dickego</b>
(rozkład na sektory spinu j; wymiar sektora ≤ N+1, więc N = 100 jest osiągalne
dokładnie — pełna przestrzeń 2^N nie). Dla każdego sektora ewolucja jest
niezależna (kolektywne S± komutują z S²); obliczamy S(t), I(A:B)(t), σ(t) =
dS/dt, P_dark(t) i τ(t) wg funkcjonału recenzji dτ/dλ = α[Ṡ + η·I(A:B)].
</p>
<div class="formula">
S(∞) → ln(N+1) (sektor symetryczny): N=2: 1.0986, N=20: 3.0445, N=100: 4.6151
(błąd ≤ @@E11_SINF_ERR@@)<br>
KOMPRESJA 27×: max|S_A(n) − S_B(27n)| = @@E11_27_WORST@@ dla N = 2..100
(okno n = 0..25) — tożsamość dokładna<br>
P_dark(Haar) = 1 − (N+1)/2^N → @@E11_PDARK@@ (N=100) — typowe stany są prawie
w całości ciemne; S∞(Haar, γ_φ=0)/N = @@E11_SHAAR@@ (vs ln 2 przy pełnej
termalizacji)
</div>
<img src="@@FIGE1@@" alt="E1 — dynamika" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE2@@" alt="E2 — skalowanie" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<p>
<b>Werdykt — czy przewidywania wynikają z modelu?</b>
</p>
<ul>
<li><b>27× — TAK, wynika i jest dokładne</b> dla wszystkich N: dynamika skaluje
się liniowo z γ, więc S_A(t) = S_B(27t) jest tożsamością (błąd numeryczny
@@E11_27_WORST@@). To nie zależy od N ani od stanu.</li>
<li><b>„Czkanie” — TAK przy η = 0</b>: kwantowany zegar entropii staje przy
nasyceniu (Δτ = 0 w ogonie dla wszystkich N). <b>ALE z η &gt; 0</b> funkcjonał
zmienia predykcję: τ̇ → η·I_eq = 0.0719 ≠ 0 (N=2 |10⟩) — <b>pamięć napędza czas
dalej</b>. Modyfikacja funkcjonału (η) jest więc ROZRÓŻNIALNA i testowalna:
zmierz, czy zegar po nasyceniu stoi (η=0) czy tyka dalej (η&gt;0).</li>
<li><b>Pamięć subradiacyjna — TAK</b>: P_dark zachowane (γ_φ=0), I(A:B) =
ln(2/√3) plateau; dla dużych N typowe stany są w ~100% ciemne, a ich entropia
per qubit maleje (S∞/N: 0.52 → 0.05) — „informacja zamrożona w sektorach
ciemnych”.</li>
</ul>
<img src="@@FIGE3@@" alt="E3 — testy" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE4@@" alt="E4 — pamięć i funkcjonał" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Wniosek dla funkcjonału czasu.</b> Predykcje 27× i pamięci wynikają z modelu
bez modyfikacji; „czkanie” (pełne zatrzymanie) wymaga η = 0. Włączenie członu
korelacyjnego η·I(A:B) — jak postuluje recenzja — daje inne, falsyfikowalne
zachowanie: po nasyceniu czas płynie dalej z tempem η·I_eq, napędzany pamięcią
subradiacyjną, a nie produkcją entropii. To jest dokładnie ten wybór, który
można rozstrzygnąć eksperymentalnie (test bright↔dark, R17).
</div>

<h2><span class="no">R19.</span> ENTROPIA-1.2 — konkurencja funkcjonałów czasu, odzyskiwalność, fizyczny 27×</h2>
<p>
<b>Diagnoza po recenzji.</b> Rozdzielamy, co jest falsyfikacją, a co
konsekwencją definicji. Raportowany w ENTROPIA-1.1 wynik „τ̇ → η·I_eq ≠ 0”
dotyczył funkcjonału z <b>absolutną informacją</b> (σ + η·I), a nie z jej
zmianą (σ + η|İ|). Uruchamiamy więc <b>cztery konkurencyjne funkcjonały</b> —
nie ratujemy jednego, pozwalamy danym wybrać:
</p>
<div class="formula">
T0: dτ = σ/σ₀ (entropiczny) &nbsp;·&nbsp; T1: dτ = (σ + η|İ|)/σ₀ (dynamiczny)<br>
T2: dτ = (σ + η·I)/σ₀ (absolutny) &nbsp;·&nbsp; T3: dτ = (σ + η|Ṙ|)/σ₀ (odzyskiwalność)
</div>
<p>
<b>Wynik (N=2 |10⟩-typ, I_eq = @@E12_IEQ@@ = ln(2/√3)):</b>
</p>
<table>
<tr><th>funkcjonał</th><th>τ̇∞ (równowaga)</th><th>zachowanie</th></tr>
<tr><td>T0 — σ</td><td>@@E12_T0@@</td><td>STAJE (czkanie)</td></tr>
<tr><td>T1 — σ+η|İ|</td><td>@@E12_T1@@</td><td>STAJE (İ → 0, mimo I_eq &gt; 0)</td></tr>
<tr><td>T2 — σ+η·I</td><td>@@E12_T2@@</td><td><b>NIE STAJE</b> (τ̇ → η·I_eq/σ₀)</td></tr>
<tr><td>T3 — σ+η|Ṙ|</td><td>@@E12_T3@@</td><td>STAJE (Ṙ → 0)</td></tr>
</table>
<p>
To są <b>dwie różne teorie czasu</b> (recenzja §3): czas napędzany <i>produkcją</i>
informacji (T1, T3 — zachowują czkanie) vs czas napędzany <i>istnieniem</i>
dostępnej relacyjnej informacji (T2 — jawnie odrzuca zamrożenie Chronosu przy
zachowanej pamięci). ENTROPIA-1.1 implementowała T2; wybór między T1 a T2 jest
rozstrzygalny eksperymentalnie (test bright↔dark, R17).
</p>
<img src="@@FIGE5@@" alt="E5 — funkcjonały" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R19.1</span> Odzyskiwalność — prawdziwy test pamięci</h2>
<p>
P_dark → 1 to podział geometryczny Hilberta; nie dowodzi pamięci. Mierzymy
<b>odzyskiwalność</b>: kod X ∈ {0,1} w dwóch ortogonalnych stanach sektora,
ewoluujemy, liczymy trace distance D(t) = ½‖ρ₀(t) − ρ₁(t)‖₁, M(t) = D(t)/D(0).
</p>
<table>
<tr><th>N</th><th>M_bright(50) (j=N/2)</th><th>M_dark(50) (j=1)</th><th>zysk pamięci</th></tr>
<tr><td>4</td><td>0.232</td><td>0.415</td><td>1.8×</td></tr>
<tr><td>10</td><td>0.112</td><td>0.415</td><td>3.7×</td></tr>
<tr><td>100</td><td>0.013</td><td>0.415</td><td><b>@@E12_GAIN100@@×</b></td></tr>
</table>
<p>
Sektor subradiacyjny j=1 ma tempo zaniku <b>niezależne od N</b> (zależy tylko
od j), podczas gdy jasny j=N/2 rozpada się tym szybciej, im większe N —
przewaga pamięci rośnie z N. Dla j=0 (parzyste N): <b>M(t) = 1 dokładnie</b>
(Γ = 0, pamięć doskonała). To rozstrzyga punkt recenzji §7: <b>niska entropia
(S∞/N → 0) NIE znaczy braku pamięci</b> — mamy niską entropię i wysoką
odzyskiwalność jednocześnie.
</p>
<img src="@@FIGE6@@" alt="E6 — odzyskiwalność" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R19.2</span> Fizyczny test 27× — z modelu kąpieli, nie ze skalowania L</h2>
<p>
Tożsamość S_B(t) = S_A(27t) przy L_A = 27·L_B jest <b>konsekwencją skalowania
generatora</b> (test solvera, nie fizyki). Właściwy test: γ(T), η(T) = e^{−ω₀/T}
wyprowadzamy z konkretnej kąpieli, ustawiamy T_A = 3·T_B i mierzymy
R_T = τ̇(T_A)/τ̇(T_B) przy dopasowanym S*.
</p>
<div class="formula">
kąpiel 3D fotonowa (J(ω) ∝ ω³ ⇒ γ ∝ T³):  R_T = @@E12_R3D@@  → 27 w lim. gorącym
(zbieżność: T_B/ω₀ = 100 ⇒ R_T = @@E12_CONV@@)<br>
kąpiel single-mode (γ ∝ 2n̄+1):           R_T = @@E12_RSINGLE@@  → 3, nie 27
</div>
<p>
<b>„27×” jest predykcją fizyczną zależną od widma kąpieli</b>, a nie tożsamością:
R_T = 27 ⇔ 3D widmo promieniowania (J ∝ ω³), R_T = 3 ⇔ pojedynczy mod. Test
falsyfikowalny: zmierz R_T dwóch zegarów w kąpielach o T_A = 3T_B — wynik
wyznacza widmo kąpieli. Poprawki skończonej T (η ≠ 1) są policzalne i znikają
w limicie gorącym.
</p>
<img src="@@FIGE7@@" alt="E7 — fizyczny 27×" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Werdykt ENTROPIA-1.2.</b> Falsyfikacja dotyczy konkretnej definicji: T2
(absolutna I) niszczy czkanie; T0, T1, T3 zachowują je. Odzyskiwalność
potwierdza pamięć subradiacyjną (j=1: @@E12_GAIN4@@× przy N=4, @@E12_GAIN100@@×
przy N=100; j=0: doskonała). Fizyczny 27× obowiązuje dla kąpieli 3D (R_T =
@@E12_R3D@@), a nie dla single-mode. Model nie jest „ratowany" — konkurencyjne
definicje czasu są uruchamiane, a dane (S, I, P_D, R, skalowanie N i T) mają
wybrać strukturę.
</div>

<h2><span class="no">R47.</span> Poprawka audytowa: fizyczny 27× z pełnej spójnej termicznej</h2>
<p>
Audyt zamykający ENTROPIA-1.2 (<code>AUDYT_ENTROPIA12.md</code>, znalezisko
nr 1) wykrył, że <b>R_T_fizyczny mieszał wzory</b>: czasy przejścia S* brał z
termicznej S(t), ale tempo dS/dt z formuły ∞-gorącej (błąd 0.2–0.8%).
Poprawka R47: pełna spójna termiczna pochodna
(<code>dSdt_termiczne_analitycznie</code>). Po poprawce projekt zgadza się z
<b>niezależnym świadkiem audytu do ~1e-10</b>.
</p>
<table>
<tr><th>T_B/ω₀</th><th>3D — przed</th><th>3D — po (R47)</th><th>single — przed</th><th>single — po (R47)</th></tr>
<tr><td>10</td><td>27.618</td><td><b>@@E47_R3D@@</b></td><td>3.066</td><td><b>@@E47_R1@@</b></td></tr>
<tr><td>100</td><td>27.058</td><td>27.079</td><td>3.006</td><td>3.009</td></tr>
</table>
<p>
Wniosek jakościowy <b>odporny</b>: 3D → 27, single-mode → 3, zbieżność w
gorącym limicie. Wartości w R19.2 (powyżej) są już po poprawce.
</p>
<img src="@@FIGA3@@" alt="A3 — R47: R_T przed/po" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R20.</span> ENTROPIA-1.3 — coherent information, koszt zegara, protokół T1 vs T2</h2>
<p>
Trzy domknięcia programu falsyfikacyjnego: (1) czy pamięć subradiacyjna jest
kwantowa czy klasyczna, (2) ile kosztuje zegar (duch Saleckera–Wignera),
(3) protokół pomiarowy rozstrzygający T1 vs T2.
</p>

<h2><span class="no">R20.1</span> Coherent information — pamięć jest klasyczna</h2>
<p>
Informacja wzajemna I(A:B) &gt; 0 nie odróżnia korelacji klasycznych od
kwantowych. Obliczamy <b>coherent information</b> I_c(A⟩B) = S_B − S_AB (miara
odzyskiwalności kwantowej):
</p>
<div class="formula">
N=4 |1111⟩: I_c(∞) = @@E13_IC4@@ = ln3 − ln5 &lt; 0 &nbsp;·&nbsp;
N=2 |10⟩: I_c(∞) = @@E13_IC10@@ &lt; 0 &nbsp;·&nbsp; ale I(A:B)(∞) = @@E13_IA@@ &gt; 0
</div>
<p>
<b>I_c &lt; 0 przy równowadze</b> — korelacje są <i>klasyczne</i> (bez
destylowalnego splątania; zgodnie z negatywnością = 0 z R2). Subradiacyjna
pamięć to korelacje klasyczne, nie kwantowe: można z nich odczytać klasyczny
bit, nie da się nimi przesyłać informacji kwantowej. Funkcjonał
T3c = (σ + η|Ī_c|)/σ₀ staje (Ī_c → 0) — dodatkowe potwierdzenie, że
„czkanie” przetrwa także przy mierze odzyskiwalności kwantowej.
</p>
<img src="@@FIGE8@@" alt="E8 — I_c" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R20.2</span> Koszt energetyczny zegara (Salecker–Wigner w modelu)</h2>
<p>
Zegar kwantowy z energią oscylatora H_clock = ω_c·b†b: E_clock = ω_c·⟨n⟩,
ΔE = ω_c·Δn, nieoznaczoność odczytu Δτ = Δn·δs. Trójkąt kompromisu jako
funkcja siły zegara γ_t (ω_c = 50):
</p>
<table>
<tr><th>γ_t</th><th>Δn/⟨n⟩ (precyzja)</th><th>E_clock (koszt)</th><th>|S∞ − ln 2| (back-action)</th></tr>
<tr><td>0.002</td><td>@@E13_PREC_LO@@</td><td>@@E13_COST_LO@@</td><td>@@E13_BACK_LO@@</td></tr>
<tr><td>0.05</td><td>@@E13_PREC_HI@@</td><td>@@E13_COST_HI@@</td><td>@@E13_BACK_HI@@</td></tr>
</table>
<div class="formula">
precyzja↑ (Δn/⟨n⟩ ↓) &nbsp;⇔&nbsp; koszt↑ (E_clock ↑) &nbsp;⇔&nbsp; entropia↑ (back-action ↑)
</div>
<p>
Warunek nieoznaczoności energii-czasu ΔE·Δτ ≥ ħ/2 nakłada na model
<b>ω_c ≥ ω_c^min = @@E13_WCMIN@@</b> (przy Δn ≈ 5, δs = 0.01) — model jest
samo-spójny dla ω_c ≥ ω_c^min, a przy ω_c = 50 warunek jest spełniony z
zapasem ~30×. To ilościowa wersja tezy recenzji §7: precyzja zegara ma koszt
fizyczny (energia) i koszt entropijny (back-action) — połączone w jeden
trójkąt.
</p>
<img src="@@FIGE9@@" alt="E9 — koszt" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R20.3</span> Protokół pomiarowy rozstrzygający T1 vs T2</h2>
<p>
Konkretna obserwowalna rozróżniająca dwie teorie czasu: po wygaśnięciu
fluorescencji (Γ = Ṡ/σ₀ → 0, sektor jasny wypalony) mierzymy tempo zegara τ̇.
</p>
<div class="formula">
Γ(∞) = 0&nbsp; ⇒&nbsp; τ̇(T1) = @@E13_TAU1@@ (zegar staje — czas = produkcja informacji)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;τ̇(T2) = @@E13_TAU2@@ = η·I_eq/σ₀ (zegar tyka — czas = istnienie informacji)
</div>
<p>
<b>Protokół</b>: przygotuj stan |10⟩-typ (jasny+ciemny, N=2 kąpiel kolektywna);
monitoruj fluorescencję i tempo zegara. Gdy Γ → 0 (t ≈ 80 tyknięć): τ̇ = 0 ⇒
T1 (czkanie); τ̇ = 7.19 = const ⇒ T2. To dokładnie test „Γ↓ ⇒ τ̇↓?” z recenzji
§10 — rozstrzygalny w układach Dicke'a z ciągłym monitoringiem emisji
(subradiancja zmierzona: PRL 116, 083601).
</p>
<img src="@@FIGE10@@" alt="E10 — protokół" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-1.3.</b> Pamięć subradiacyjna jest klasyczna
(I_c &lt; 0) — „informacja → sektor ciemny → wydłużona pamięć” to pamięć
klasyczna, co jest spójne z odzyskiwalnością M(t) (R19.1) i negatywnością 0
(R2). Zegar ma policzalny koszt (E_clock, ΔE·Δτ ≥ ħ/2 ⇒ ω_c^min = @@E13_WCMIN@@)
i trójkąt precyzja↔koszt↔entropia. Protokół R20.3 daje obserwowalną
rozstrzygającą T1 vs T2 — ostatni brakujący element programu
falsyfikacyjnego recenzji.
</div>

<h2><span class="no">R21.</span> ENTROPIA-1.4 — fidelity-based quantum recovery channel</h2>
<p>
Trace distance (R19.1) i coherent information (R20.1) uzupełniamy o miarę
<b>fidelity</b>: M_F(t) = 1 − F(ρ₀(t), ρ₁(t)) dla dwóch ortogonalnych stanów
(kod X ∈ {0,1}) ewoluujących w sektorze; F_e — entanglement fidelity kanału
sektora (F_e = 1 ⇔ kanał identyczności, doskonały odzysk).
</p>
<div class="formula">
M_F(ciemny j=1, t=7.5) = @@E14_MF4@@ niezależnie od N&nbsp; ·&nbsp;
M_F(jasny j=N/2): N=4 → 0.16, N=100 → @@E14_MF100@@ (zysk @@E14_GAIN100@@×)<br>
F_e: j=0 → @@E14_FE0@@, j=1 → @@E14_FE1@@, j=2 → @@E14_FE2@@ (maleje z j)
</div>
<p>
<b>Kanał odzyskiwania istnieje i jest tym lepszy, im głębszy sektor ciemny:</b>
j=0 (parzyste N) to kanał identyczności (M_F = 1, F_e = 1 — doskonały odzysk);
j=1 ma tempo zaniku rozróżnialności niezależne od N, podczas gdy jasny j=N/2
rozpada się z szybkością rosnącą z N. Fidelity jest związana z trace distance
przez nierówność Fuchsa–van de Graafa: 1−√F ≤ D ≤ √(1−F), więc M_F i M(t)
(R19.1) niosą zgodny obraz — ale M_F jest czulsza na małe zniekształcenia.
</p>
<img src="@@FIGE11@@" alt="E11 — fidelity" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R22.</span> Pełny rachunek ω_c(T) z fizycznej kąpieli</h2>
<p>
Częstość oscylatora-zegara nie jest wolnym parametrem — wynika z ograniczeń
fizycznych kąpieli o temperaturze T:
</p>
<div class="formula">
(a) obsada termiczna: n̄(ω_c,T) &lt; ε ⇒ ω_c &gt; T·ln(1/ε) — zegar musi być
odporny na szum termiczny (ε = 0.01: ω_c &gt; 4.6·T)<br>
(b) energia–czas: ΔE·Δτ ≥ ħ/2 ⇒ ω_c ≥ ω_c^min = 1.7 (R20)<br>
(c) pojemność: MLEV ≥ ln2/δs = @@E14_CAP@@ (δs = 0.01)
</div>
<p>
Numerycznie (ε = 0.01): ω_c(10) = @@E14_WC10@@, ω_c(100) = @@E14_WC100@@ —
w limicie gorącym <b>ω_c ∝ T</b>, więc dla T_A = 3·T_B: ω_c(A)/ω_c(B) =
@@E14_RATIO3@@. To rozdziela dwie skale: <b>rozdzielczość czasu rośnie jak T</b>
(zegar w gorętszym otoczeniu musi tykać z wyższą częstością, by uniknąć szumu),
a <b>tempo produkcji entropii jak T³ (27×)</b>. Liczba tyknięć (budżet ⟨n⟩ =
ln2/δs) jest od temperatury niezależna — temperatura zmienia częstość zegara,
nie jego pojemność.
</p>
<img src="@@FIGE12@@" alt="E12 — ω_c(T)" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R23.</span> Protokół R20.3 end-to-end z detekcją fotonów</h2>
<p>
Pełna symulacja Monte Carlo protokołu: stan |10⟩-typ w kąpieli kolektywnej,
<b>detekcja fotonów</b> (Poisson o tempie Ṡ/δs, wydajność η_det, szum tła
dark counts), <b>odczyty zegara</b> (kwantowane). Po ostatnim fotonie
(n = @@E14_NLAST@@, t ≈ 32) zaczyna się faza ciemna; mierzymy τ̇ przez M tyknięć
i decydujemy: τ̇ ≈ 0 ⇒ T1 (czas = produkcja), τ̇ = η·I_eq/σ₀ ⇒ T2 (czas =
istnienie informacji).
</p>
<table>
<tr><th>konfiguracja</th><th>P(T1 poprawna)</th><th>P(T2 poprawna)</th></tr>
<tr><td>M = 10, η_det = 1</td><td>@@E14_MOC10@@</td><td>@@E14_MOC_T2@@</td></tr>
<tr><td>M = 30, η_det = 0.1</td><td>@@E14_MOC10@@</td><td>@@E14_MOC_DR@@</td></tr>
<tr><td>M = 30, dark counts = 2/tyknięcie</td><td>@@E14_MOC_DR@@</td><td>@@E14_MOC_T2@@</td></tr>
</table>
<p>
<b>Moc jest nasycona (1.000) nawet przy wydajności detekcji 10% i szumie tła</b>
— separacja τ̇ (0 vs 7.2) jest tak duża, że szum odczytu (Poisson na kwantach
δs) jej nie zamyka. Protokół jest praktycznie rozstrzygający przy parametrach
modelu; ograniczenie pojawiłoby się dopiero przy szumie zegara porównywalnym
z η·I_eq, czyli przy η·I_eq ≲ 0.1 (słaba pamięć lub małe η) — wtedy potrzebna
dłuższa integracja M.
</p>
<img src="@@FIGE13@@" alt="E13 — protokół e2e" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-1.4.</b> Kanał odzyskiwania jest w pełni
scharakteryzowany (M_F, F_e, związek z M(t) i I_c): głębokie sektory ciemne
(j=0: dokładnie identyczność) dają doskonały odzysk klasyczny. Częstość zegara
wynika z kąpieli (ω_c ∝ T — rozdzielczość ~ T vs produkcja entropii ~ T³).
Protokół rozstrzygający T1 vs T2 działa end-to-end z mocą 1.000, odporny na
straty detekcji i szum tła. Program falsyfikacyjny jest zamknięty: miary
(σ, I, İ, R, I_c, F), koszt (E_clock, ω_c(T)) i testowalność (protokół) są
określone ilościowo.
</div>

<h2><span class="no">R24.</span> ENTROPIA-1.5 — pamięć operacyjna (Helstrom, Chernoff, C_mem)</h2>
<p>
Fidelity i trace distance to miary geometryczne; to, co można <b>operacyjnie
odzyskać</b>, mierzy granica Helstroma: p_err(t) = ½(1 − D(t)/2) (minimalny
błąd rozróżnienia dwóch stanów-kodów X ∈ {0,1}) i pojemność klasyczna kanału
pamięci C_mem = 1 − h₂(p_err).
</p>
<table>
<tr><th>sektor</th><th>p_err (t=7.5)</th><th>C_mem (t=7.5) [bit]</th><th>C_mem (t=15) [bit]</th></tr>
<tr><td>jasny j=N/2, N=4</td><td>0.31</td><td>@@E15_CM4@@</td><td>—</td></tr>
<tr><td>jasny j=N/2, N=100</td><td>0.49</td><td>@@E15_CM100@@</td><td>—</td></tr>
<tr><td>ciemny j=1</td><td>0.21</td><td>@@E15_CMD@@</td><td>@@E15_CMD15@@</td></tr>
<tr><td>ciemny j=1/2 (dim 2)</td><td>0.13</td><td>@@E15_CMH@@</td><td>—</td></tr>
</table>
<p>
<b>Operacyjna pamięć jest realna i mierzalna</b>: jasny sektor N=100 traci
pojemność do @@E15_CM100@@ bitu w ~7 tyknięciach (p_err → ½, zgadywanie);
ciemny j=1/2 utrzymuje @@E15_CMH@@ bitu (p_err = 0.13) — można odczytać bit
z błędem 13% zamiast 50%. Wykładnik Chernoffa potwierdza: ξ(ciemny) ≈ 2×
ξ(jasny). j=0 jest niezmienniczy, ale 1-wymiarowy (log₂1 = 0 bitów — to
magazyn fazy, nie bitu). To domyka pytanie recenzji §7/§8: sektor ciemny to
<b>pamięć klasyczna o policzalnej pojemności</b>, nie tylko koncentracja normy.
</p>
<img src="@@FIGE14@@" alt="E14 — pamięć operacyjna" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R25.</span> Samo-spójny ω_c(T) — maksymalna temperatura zegara</h2>
<p>
Pełny rachunek ω_c(T) z fizycznej kąpieli: Purcell (γ_t(ω_c) = g²·J(ω_c) ∝
g²ω_c³ dla kąpieli 3D) łączy częstość zegara z jego back-action na Wszechświat.
Dwie przeciwstawne granice:
</p>
<div class="formula">
dolna (szum termiczny): n̄(ω_c,T) &lt; ε ⇒ ω_c &gt; T·ln(1/ε)<br>
górna (back-action): |S∞ − ln2| &lt; ε_b ⇒ ω_c &lt; (ε_b/(c·g²))^{1/3}
</div>
<p>
Obie naraz ⇒ <b>okno istnienia zegara</b> i nowa, falsyfikowalna predykcja —
<b>maksymalna temperatura</b>:
</p>
<div class="formula">
T_max = (ε_b/(c·g²))^{1/3} / ln(1/ε):&nbsp; g = 0.1 ⇒ @@E15_TMAX_1@@,&nbsp;
g = 0.01 ⇒ @@E15_TMAX_2@@ (ε = 0.01, ε_b = 0.05)
</div>
<p>
Dla T &gt; T_max zegar kwantowy <b>nie może istnieć</b>: żeby uniknąć szumu
termicznego musiałby tykać z ω_c &gt; T·ln(1/ε), ale wtedy back-action
(∝ g²ω_c³) przekracza budżet ε_b. Słabsze sprzężenie zegar-kąpiel (mniejsze g)
podnosi T_max ∝ g^{−2/3} — cichszy zegar działa w gorętszym wszechświecie.
To ilościowe domknięcie R22: ω_c nie tylko rośnie z T, ale ma <b>górną granicę
związaną z ceną, jaką zegar płaci Wszechświatowi</b>.
</p>
<img src="@@FIGE15@@" alt="E15 — okno ω_c" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R26.</span> Sekwencyjny test Walda — protokół adaptacyjny</h2>
<p>
Zamiast stałej integracji M protokół używa <b>sekwencyjnego ilorazu
wiarogodności (SPRT)</b>: po ostatnim fotonie akumuluje tyknięcia zegara
(Poisson λ₀ ≈ 0 dla T1, λ₂ = 7.19 dla T2) i zatrzymuje się, gdy log-iloraz
przekroczy próg (α = β = 0.01).
</p>
<div class="formula">
E[N] (decyzja): T1 = @@E15_E1@@, T2 = @@E15_E2@@ tyknięcie — vs stałe M = 10<br>
słabsza separacja: λ₂ = 0.5 ⇒ E[N] = @@E15_EN05@@; λ₂ = 0.1 ⇒ E[N] = @@E15_EN01@@
</div>
<p>
<b>SPRT adaptuje się do jakości danych</b>: przy parametrach modelu decyzja
zapada w ~1 tyknięciu (separacja λ 0 vs 7.19 jest ogromna — błędy mierzone
α = β = 0), a przy słabszej pamięci (małe η·I_eq) protokół sam wydłuża
integrację (E[N] = 2.6 → 20.3). To zamienia test R23 z „pewny, ale stały" w
„pewny i minimalnie kosztowny" — optymalny w sensie Walda (minimalny E[N] przy
zadanych błędach). Kompletna procedura: faza jasna (detekcja fotonów ⇒ koniec
Γ) → faza ciemna (SPRT na tyknięciach zegara ⇒ T1 vs T2).
</p>
<img src="@@FIGE16@@" alt="E16 — SPRT" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-1.5.</b> Pamięć subradiacyjna ma operacyjną,
policzalną pojemność (C_mem: 0.44 bitu w j=1/2 vs 0.0004 w jasnym N=100;
p_err 0.13 vs 0.49). Częstość zegara ma samo-spójne okno istnienia i
maksymalną temperaturę T_max (nowa predykcja — zegary nie mogą istnieć w
zbyt gorących kąpielach). Protokół rozstrzygający jest optymalny sekwencyjnie
(E[N] = 1 przy parametrach modelu, adaptacja do słabej separacji). Miary,
koszty i testowalność ENTROPII są teraz w pełni określone ilościowo.
</div>

<h2><span class="no">R27.</span> Eksperymentalna karta protokołu R23 — konkretne parametry</h2>
<p>
Protokół rozstrzygający T1 vs T2 przeliczony na <b>konkretny eksperyment</b>:
zimne atomy, kolektywna emisja, detekcja fotonów, pomiar korelacji. Pełny
arkusz: <code>EKSPERYMENT.md</code>; poniżej esencja.
</p>
<div class="formula">
platforma B (nanofiber ¹³³Cs): N = 5×10³, β ≈ 0.15, τ_nat = 30.5 ns&nbsp; ·&nbsp;
t_B ≈ @@EXP_TB_B@@ ns, t_D ≈ @@EXP_TD_B@@ μs (t_D/t_B = @@EXP_TDTB@@)<br>
τ̇ w fazie ciemnej: T1 = @@EXP_TAU1@@, T2 = @@EXP_TAU2@@ nat/s (Δt_samp = 1 μs)
</div>
<p>
<b>Sekwencja</b> (minuty łącznie): pułapka → stan |10⟩-typ (1 ekscyton, Dicke
z fazą) → faza jasna (fotony, SPCM) → ostatni foton ⇒ faza ciemna (t_D = 179 μs)
→ pomiar I(A:B) w 6 punktach czasowych (podział chmury A|B, obrazowanie,
M ≈ @@EXP_M@@ realizacji/punkt, precyzja σ_I = @@EXP_SIGI@@ nat) → rozstrzygnięcie.
</p>
<table>
<tr><th>platforma</th><th>SNR kanału fotonowego (1 ms ciemnej)</th><th>t_D</th></tr>
<tr><td>A: wolna przestrzeń ⁸⁷Rb</td><td>@@EXP_SNR_A@@</td><td>2.6 μs</td></tr>
<tr><td>B: nanofiber ¹³³Cs</td><td>@@EXP_SNR_B@@</td><td>179 μs</td></tr>
<tr><td>C: wnęka optyczna</td><td>@@EXP_SNR_C@@</td><td>1.3 μs</td></tr>
</table>
<p>
<b>Uczciwa granica.</b> Kanał fotonowy (fluorescencja, subradiancja) potwierdza
separację jasny/ciemny i pamięć (R17/R21/R24) — ale NIE rozróżnia T1 od T2
(zegar w modelu nie sprzęga się zwrotnie). Rozstrzyga <b>kanał korelacyjny</b>:
I(A:B) w fazie ciemnej; precyzja σ_I = 0.01 nat osiągalna przy M = @@EXP_M@@
realizacjach/punkt (czułość h₂ przy p = 0.5 jest kwadratowa — policzona
liczbowo). T1: I(t) const, τ̇ = 0; T2: korelacja napędza czas (τ̇ = η·I_eq).
</p>
<img src="@@FIGE17@@" alt="E17 — sekwencja" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Werdykt wykonalności.</b> Protokół R23 jest <b>wykonalny na istniejącej
technologii</b> (nanofiber + SPCM + obrazowanie fluorescencji): przygotowanie
stanów Dicke — literatura; detekcja ns/μs — SPCM; pomiar I(A:B) z precyzją
0.01 nat — M ≈ 150 realizacji/punkt, całość w ~minutę. To pierwszy
laboratoryjny test tezy „czas = entropia”: czy zegar entropowy staje po
wygaśnięciu fluorescencji (T1), czy korelacja subradiacyjna napędza go dalej
(T2). Nowe predykcje (T_max zegara, ω_c ∝ T) wymagają innej geometrii i są
testem kolejnego etapu.
</div>

<h2><span class="no">R28.</span> Suchy bieg protokołu R23/R26 — realistyczny detektor</h2>
<p>
Pełny Monte Carlo protokołu z <b>realistycznym detektorem</b>: fotony
(Poisson Ṡ/δs) → detekcja (η_det = 0.3, dark counts = 100 Hz, timing jitter
= 1 ns — typowy SPCM) → faza ciemna → SPRT (R26) na tyknięciach zegara.
</p>
<div class="formula">
ostatni foton: n = @@E16_TLAST@@ (t ≈ 43 μs) &nbsp;·&nbsp; SPRT: E[N] = 1.0
(α = β = 0) dla T1 i T2 &nbsp;·&nbsp; całkowity czas pomiaru ≈ @@E16_TTOT@@ μs/realizację
</div>
<p>
<b>Jitter 1 ns ≪ t_B (~8 ns) i t_D (179 μs)</b> — nie wpływa na protokół;
decydują wydajność detektora i dark counts (przesuwają ostatni foton, nie
zmieniają rozstrzygnięcia — SPRT adaptuje się). Karta eksperymentalna
(EKSPERYMENT.md + R27) jest więc potwierdzona suchym biegiem: <b>cały pomiar
(minuty łącznie z powtórzeniami) rozstrzyga T1 vs T2 z E[N] = 1</b>.
</p>

<h2><span class="no">R29.</span> Jawna mapa odzysku Petza</h2>
<p>
Dotychczas mierzyliśmy odzyskiwalność (M_F, C_mem); teraz budujemy <b>jawny
kanał odzysku</b> — mapę Petza: R(·) = σ^{1/2}Φ†(Φ(σ)^{−1/2}(·)Φ(σ)^{−1/2})σ^{1/2}
(Petz 1986; Φ = ewolucja sektora, Φ† = sprzężenie w HS). F_rec =
F(ρ₀, R∘Φ(ρ₀)) mówi, ile bitu <b>da się rzeczywiście odzyskać</b> tym kanałem.
</p>
<table>
<tr><th>sektor</th><th>F_rec (mapa Petza)</th></tr>
<tr><td>ciemny j=1/2</td><td>@@E16_FREC_05@@</td></tr>
<tr><td>ciemny j=1</td><td>@@E16_FREC_1@@</td></tr>
<tr><td>jasny N=4 j=2</td><td>0.645</td></tr>
<tr><td>jasny N=6 j=3</td><td>@@E16_FREC_3@@</td></tr>
<tr><td>j=0</td><td>1.000 (kanał identyczności)</td></tr>
</table>
<p>
<b>Hierarchia odzysku potwierdzona jawnym kanałem</b>: ciemne sektory odzyskują
bit z fidelnością 0.89, jasne tylko 0.54–0.65. To najsilniejsza z dotychczasowych
miar — nie „istnieje pamięć”, lecz „oto jawny protokół, który ją odzyskuje”.
Spójne z C_mem (R24), M_F (R21), I_c (R20): pełny obraz pamięci subradiacyjnej.
</p>
<img src="@@FIGE18@@" alt="E18 — Petz" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R30.</span> Zimny zegar — ω_c(T) ze skończonej gęstości widmowej</h2>
<p>
Dotychczasowa górna granica ω_c (R25) zakładała widmo 3D bez cutoffu
(J ∝ ω³ ⇒ γ_t ∝ ω_c³ rośnie bez końca). Realne kąpiele mają <b>skończoną
gęstość widmową</b> (Ohmic z odcięciem, Lorentzian/wnęka): γ_t(ω_c) = g²J(ω_c)
<b>saturuje</b> przy ω_c ≫ ω_cut — back-action przestaje rosnąć z ω_c.
</p>
<div class="formula">
3D (bez cutoffu): T_max = @@E16_TM3@@ (ω_c^max = 3.0)&nbsp; ·&nbsp;
Ohmic (ω_cut = 50): T_max = @@E16_TOHM@@ — o 5 rzędów wyżej
</div>
<p>
<b>Zimny zegar działa w gorętszych epokach niż przewidywał model 3D</b>:
back-action jest ograniczony przez cutoff, więc jedynym ograniczeniem ω_c jest
szum termiczny (łatwy do pokonania w zimnym otoczeniu). Dyskusja kosmologiczna
(minimalna ω_c termiczna, ε = 0.01):
</p>
<table>
<tr><th>epoka</th><th>T</th><th>ω_c ≥</th><th>reżim</th></tr>
<tr><td>dziś (CMB)</td><td>2.7 K</td><td>@@E16_CMB@@ rad/s</td><td>mikrofale (łatwe)</td></tr>
<tr><td>rekombinacja</td><td>3×10³ K</td><td>@@E16_REC@@ rad/s</td><td>podczerwień (optyka)</td></tr>
<tr><td>BBN</td><td>1.2×10¹⁰ K</td><td>@@E16_BBN@@ rad/s</td><td>rentgen/gamma (trudne)</td></tr>
<tr><td>elektrosłaba</td><td>1.2×10¹⁵ K</td><td>@@E16_EW@@ rad/s</td><td>gamma (ekstremalne)</td></tr>
</table>
<p>
Dziś i przy rekombinacji kwantowy zegar entropii jest <b>łatwo osiągalny</b>
(częstości mikrofal/optyczne); BBN wymagałby zegara rentgenowskiego — możliwe
ze słabym sprzężeniem g (T_max ∝ g^{−2/3} dla 3D, jeszcze łagodniej z cutoffem).
To ilościowa odpowiedź na pytanie, w jakich epokach kosmicznych „czas jako
entropia" może być mierzony kwantowym zegarem.
</p>
<img src="@@FIGE19@@" alt="E19 — zimny zegar" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Uczciwa uwaga (R30).</b> W modelu z pojedynczym kanałem (kąpiel) cutoff
usuwa górną granicę ω_c — ale realne zegary mają inne kanały dekoherencji
(γ_φ, szum techniczny), które ograniczą ω_c. Wniosek jakościowy jest jednak
solidny: <b>kształt gęstości widmowej kąpieli decyduje o maksymalnej
temperaturze zegara</b> — skończone widmo (wnęka, fotonika) robi z niego
„zimny zegar" odporny na gorące otoczenie.
</div>

<h2><span class="no">R31.</span> Eksperymentalny arkusz T_max/ω_c(T) — zegar jako układ pomocniczy</h2>
<p>
Predykcje T_max i ω_c(T) przeliczamy na <b>konkretny układ</b>: zegar jako
układ pomocniczy w obwodzie nadprzewodzącym — kubit = „wszechświat”, rezonator
= zegar (ω_c), kąpiel = szum termiczny w lodówce rozcieńczalnikowej (T =
10–300 mK).
</p>
<div class="formula">
T_max(ω_c) = ħω_c/(k_B ln(1/ε)):&nbsp; 6 GHz → @@E17_TMAX6@@ mK,&nbsp;
30 GHz → @@E17_TMAX30@@ mK&nbsp; ·&nbsp; ω_c ∝ T (rozdzielczość ~ T):
T: 10→30 mK ⇒ ω_c ×3.00 dokładnie
</div>
<p>
<b>Co mierzyć</b>: (1) obsadę termiczną n̄(ω_c,T) przez spektroskopię kubitu —
próg n̄ &lt; 0.01 daje T_max (6 GHz: osiągalne w lodówce 10–30 mK, niespełnione
powyżej 63 mK — <b>bezpośredni test predykcji</b>); (2) back-action przez
Purcell: T1_Purcell = @@E17_T1P@@ μs (g = 50 MHz, κ = 1 MHz, Δ = 1 GHz) —
mierzalne, porównywalne z γ_t = g²J(ω_c); (3) skalowanie ω_c ∝ T przez
zmianę T i wymaganą częstość przy stałym budżecie termicznym.
</p>
<img src="@@FIGE20@@" alt="E20 — arkusz T_max" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<p>
Zegar jako układ pomocniczy umożliwia test bez pełnego protokołu R23:
wystarczy kubit + rezonator + kontrola T. To najtańszy eksperymentalnie test
konkretnej predykcji ENTROPII (R25/R30).
</p>

<h2><span class="no">R32.</span> Koszt energetyczny protokołu</h2>
<p>
Budżet energii jednej realizacji protokołu R23 (ω_c/2π = 6 GHz, ⟨n⟩ = 6.6):
</p>
<table>
<tr><th>składnik</th><th>energia [J]</th><th>uwagi</th></tr>
<tr><td>zegar E = ħω_c⟨n⟩</td><td>@@E17_ECLK@@</td><td>~10⁻⁴ eV — zaniedbywalne</td></tr>
<tr><td>decyzja (Landauer, 100 mK)</td><td>@@E17_ELAND@@</td><td>1 bit rozstrzygnięcia</td></tr>
<tr><td>obrazowanie/detekcja</td><td>~10⁻⁹</td><td>SPCM, kamera</td></tr>
<tr><td>pułapka (MOT 100 mW × 50 ms)</td><td>@@E17_ETRAP@@ mJ</td><td><b>dominuje</b> — technika</td></tr>
</table>
<div class="formula">
ΔE·Δτ = @@E17_DEDT@@ J·nat ≥ ħ/2 = 0.5 ✓ (ΔE = ħω_cΔn, Δτ = Δn·δs)
</div>
<p>
<b>Zegar i decyzja są energetycznie zaniedbywalne</b> (10⁻²³–10⁻²⁵ J); kosztem
protokołu jest technika (pułapka 5 mJ — 20 rzędów więcej). Fundamentalny limit
ΔE·Δτ ≥ ħ/2 jest spełniony z zapasem ~10²³ — protokół nie zbliża się do
granicy kwantowej. T1 vs T2: różnica energii w fazie ciemnej (~10⁻²⁷ J) jest
niewykrywalna energetycznie — decyzję daje <i>statystyka</i> (SPRT), nie energia.
</p>
<img src="@@FIGE21@@" alt="E21 — energia" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R33.</span> Suchy bieg z niedoskonałą wiernością stanu</h2>
<p>
Rozszerzenie suchego biegu (R28) o <b>niedoskonałą wierność przygotowania</b>:
ρ(F) = F·ρ10 + (1−F)·𝟙/4 (niezwiązana domieszka — redukuje korelacje), z pełnym
detektorem (η_det = 0.3, dark = 100 Hz, jitter = 1 ns) i SPRT.
</p>
<table>
<tr><th>F</th><th>I_eq [nat]</th><th>τ̇_T2</th><th>τ̇_T1</th><th>P(T1), P(T2)</th></tr>
<tr><td>1.00</td><td>@@E17_IEQ1@@</td><td>@@E17_TAU2_1@@</td><td>0</td><td>1.000 / 1.000</td></tr>
<tr><td>0.70</td><td>0.073</td><td>3.64</td><td>0</td><td>1.000 / 1.000</td></tr>
<tr><td>0.30</td><td>@@E17_IEQ3@@</td><td>@@E17_TAU2_3@@</td><td>0</td><td>@@E17_P1_3@@ / @@E17_P1_3@@</td></tr>
</table>
<p>
<b>Systematyczny efekt jest realny, ale kontrolowany</b>: infidelity obniża
I_eq (0.144 → 0.014) i τ̇_T2 (7.19 → 0.71), jednak τ̇_T1 = 0 pozostaje zerowe,
a moc rozstrzygnięcia utrzymuje się na poziomie 1.000 nawet przy F = 0.3.
Kluczowa własność: <b>protokół samo-kalibruje się</b> — mierzy I(A:B) wprost
w fazie ciemnej, więc nieznana wierność F nie fałszuje decyzji (przesuwa
jedynie punkt pracy τ̇_T2, nie zeruje go). Dopiero F → 0 (brak korelacji)
niszczy rozróżnialność — a wtedy T2 traci sens fizyczny (nie ma pamięci).
</p>
<img src="@@FIGE22@@" alt="E22 — wierność" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-1.7.</b> Predykcja T_max/ω_c(T) ma tani eksperyment
(nadprzewodniki, lodówka 10–300 mK): próg obsady termicznej i Purcell
back-action są bezpośrednio mierzalne. Koszt energetyczny protokołu jest
zaniedbywalny poza techniką (pułapka). Suchy bieg z niedoskonałą wiernością
pokazuje, że protokół jest odporny na systematykę przygotowania — decyzja
T1 vs T2 pozostaje poprawna przy F ≥ 0.3 dzięki samo-kalibracji przez
pomiar I(A:B).
</div>

<h2><span class="no">R34.</span> Fizyczna realizacja mapy Petza — jak zbudować R w laboratorium</h2>
<p>
Mapa Petza R = σ^{1/2}Φ†(Φ(σ)^{−1/2}(·)Φ(σ)^{−1/2})σ^{1/2} jest konstrukcją
matematyczną; jak ją <b>zrealizować fizycznie</b>? Dla kodów fazowych
(|+⟩,|−⟩ — wrażliwych na dekoherencję) w sektorze j porównujemy trzy protokoły:
</p>
<table>
<tr><th>sektor</th><th>F (Petz, idealny)</th><th>F (echo, π-pulsy)</th><th>F (klasyczny)</th></tr>
<tr><td>j = 1/2 (ciemny)</td><td>@@E18_P05@@</td><td>@@E18_E05@@</td><td>0.500</td></tr>
<tr><td>j = 1 (ciemny)</td><td>0.677</td><td>0.544</td><td>0.500</td></tr>
<tr><td>j = 2 (jasny N=4)</td><td>@@E18_P20@@</td><td>@@E18_E20@@</td><td>0.500</td></tr>
</table>
<p>
<b>Wyniki i uczciwe wnioski:</b> F(Petz) &gt; F(klasyczny) zawsze (Petz jest
optymalny — klasyczny pomiar fazy zawodzi, F = ½). Protokół <b>echo</b>
(π-pulsy — realizowalny fizycznie refocusing) odzyskuje fazę tam, gdzie
dominuje <b>dekoherencja czysta</b> (j ≤ 1: echo &gt; klasyczny), ale
<b>szkodzi</b> przy dominacji <b>zaniku amplitudowego</b> (j ≥ 1.5: echo &lt;
klasyczny) — echo odwraca tylko koherentną część kanału. Najlepsza
<b>realizacja R w laboratorium: kodowanie w sektorze ciemnym (DFS)</b> — tam
kanał ≈ identyczność, więc F = 1 bez żadnego protokołu: „chroń, nie odzyskuj".
To jest właśnie fizyczny sens mapy Petza w kontekście ENTROPII.
</p>
<img src="@@FIGE23@@" alt="E23 — realizacja Petza" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R35.</span> Zegar w kąpieli CMB — widmo Plancka z cutoffem grawitacyjnym</h2>
<p>
Pełna ewolucja kwantowego zegara w <b>realistycznej kąpieli CMB</b> (widmo
Plancka) z <b>cutoffem grawitacyjnym</b> J(ω) = g²ω³/(1+(ω/ω_G)²), ω_G ≈
1.85×10⁴³ rad/s:
</p>
<div class="formula">
próg: n̄(ω_c, T_CMB) &lt; ε = 0.01 ⇒ ω_c/2π ≥ @@E18_WCMIN@@ GHz&nbsp; ·&nbsp;
100 GHz: n̄ = @@E18_N100@@ (SZUM), 1 THz: n̄ = @@E18_N1T@@ (bezpieczny)
</div>
<p>
<b>Wniosek ilościowy: w kąpieli CMB zegar musi być zegarem THz</b> — 100 GHz
ma n̄ = 0.207 (21% obsady termicznej — szum), 300 GHz przechodzi próg, a 1 THz
jest bezpieczny z zapasem 5 rzędów. Tempo grzania ∝ βω³(n̄+1) daje
<b>inżynierskie ograniczenie na sprzężenie β</b>: τ_heat ≫ t_protokołu.
Cutoff grawitacyjny: <b>bez wpływu dla realistycznych zegarów</b>
(J(cutoff)/J(3D) ≈ 1 do ω_c ~ 10³⁰ rad/s), dopiero przy ω_Planck redukuje J
o połowę — wyznacza absolutną skalę UV (i IR: tryb Hubble'a jako dolna
granica). To domknięcie R30 w realnym kosmicznym środowisku.
</p>
<img src="@@FIGE24@@" alt="E24 — zegar w CMB" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R36.</span> Kosmiczna sieć zegarów entropii — synchronizacja</h2>
<p>
Wiele komórek, każda z własnym tempem τ̇_k (różne lokalne kąpiele/T), sprzężonych
<b>wymianą entropii</b> (g_sync): wskazania ciągną do średniej — rozrzut σ_τ
maleje:
</p>
<div class="formula">
g_sync = 0: σ_end = @@E18_S0@@ (rozjazd)&nbsp; ·&nbsp; g_sync = 0.2:
σ_end = @@E18_S2@@ (zsynchronizowane)&nbsp; ·&nbsp; τ_net = ⟨τ_k⟩ — niezmiennicze
</div>
<p>
<b>Dwie warstwy synchronizacji:</b> (1) komórki o <b>jednakowej temperaturze</b>
(τ̇ równe) są zsynchronizowane <b>bez żadnego sprzężenia</b> — naturalny
„kosmiczny czas" istnieje, gdy Wszechświat jest termicznie jednorodny;
(2) komórki o różnych T synchronizują się przez wymianę entropii (g_sync) —
σ_end spada od 48 do 0.08. Sieć daje <b>emergentny czas kosmiczny
τ_net = ⟨τ_k⟩</b> wraz z miarą jego koherencji (σ_τ) — i wraca do obserwacji
R5/R18: entropia per komórka skaluje się z N, a sieć synchronizuje wskazania
szybciej niż termalizuje (porównywalne czasy).
</p>
<img src="@@FIGE25@@" alt="E25 — sieć" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-1.8.</b> Mapa Petza ma fizyczną realizację: echo
(częściową) i — najlepszą — kodowanie w ciemnym sektorze (F = 1). Zegar w
kąpieli CMB musi być THz (próg 261 GHz; cutoff grawitacyjny bez wpływu dla
realistycznych częstości). Sieć zegarów synchronizuje się przez wymianę
entropii, a jednorodny termicznie Wszechświat ma naturalny kosmiczny czas
(τ_net). Trzy kolejne domknięcia: od konstrukcji matematycznej (R29) do
protokołu laboratoryjnego (R34), od abstrakcyjnej kąpieli (R30) do realnego
CMB (R35), od pojedynczego zegara do kosmicznej sieci (R36).
</div>

<h2><span class="no">R37.</span> Pełny protokół z synchronizacją wielu zegarów — test różnicowy</h2>
<p>
Predykcja recenzji §10: <i>dwa identyczne zegary w środowiskach o różnych
kanałach dyssypacyjnych powinny wykazywać różnicę dynamiki entropicznej
nieredukowalną do dylatacji</i>. Protokół: M identycznych zegarów entropii,
klasa A sprzężona z komórką jasna→ciemna (zachowuje I_eq = ln(2/√3)), klasa B
z komórką czysto ciemną (I = 0, τ̇ = 0 zawsze). Po fazie jasnej mierzymy
<b>dryft różnicowy</b> Δτ̄ = τ_A − τ_B per tyknięcie fazy ciemnej:
</p>
<div class="formula">
T1: Δτ = @@E19_DR_T1@@ ≈ 0 (oba zegary stają — czkanie)&nbsp; ·&nbsp;
T2: Δτ = @@E19_DR_T2@@ ≈ 7.19 nat/tyk (liniowy — korelacja napędza czas)
</div>
<p>
<b>Zalety różnicowości</b>: (1) <b>common mode odrzucony</b> — niedoskonałości
zegara (identyczne dla wszystkich) znikają z Δτ; (2) <b>brak kalibracji
absolutnej</b> — mierzymy tylko względny dryft; (3) <b>uśrednianie po sieci</b>:
σ(Δτ̄) maleje z liczbą zegarów A (σ: @@E19_SIG1@@ → @@E19_SIG8@@ dla M_A: 1 → 8,
~1/√8); (4) <b>redundancja</b> — awaria pojedynczego zegara nie psuje testu.
Synchronizacja (R36) wchodzi jako etap: jednakowe komórki ⇒ τ̇ równe ⇒ σ ≡ 0
bez sprzężenia — sieć definiuje τ_net jako emergentny czas kosmiczny, a
następnie służy jako platforma testu różnicowego.
</p>
<img src="@@FIGE26@@" alt="E26 — protokół różnicowy" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R38.</span> Zegar w ewoluującej kąpieli CMB (ΛCDM + cutoff grawitacyjny)</h2>
<p>
R35 zakładał stałą T_CMB; teraz pełna ewolucja: T(z) = T₀(1+z) z wiekiem
wszechświata t(z) (płaska ΛCDM: H₀ = 67.4, Ωm = 0.315, ΩΛ = 0.685; weryfikacja:
t(z=0) = 13.79 Gyr). Dla zegara o ω_c: n̄(ω_c, T(t)) &lt; ε = 0.01 od pewnej
epoki — <b>horyzont zegarów</b> (częstotliwość progu spada, gdy wszechświat
się ochładza):
</p>
<table>
<tr><th>ω_c/2π</th><th>T użyteczna</th><th>użyteczny od</th><th>wiek</th></tr>
<tr><td>6 GHz</td><td>0.063 K</td><td>@@E19_Z6@@</td><td>—</td></tr>
<tr><td>300 GHz</td><td>3.13 K</td><td>z ≈ @@E19_Z300@@</td><td>11.9 Gyr</td></tr>
<tr><td>1 THz</td><td>10.4 K</td><td>z ≈ 2.8</td><td>@@E19_T1T@@ Gyr</td></tr>
<tr><td>3 THz</td><td>31.3 K</td><td>z ≈ 10.5</td><td>@@E19_T3T@@ Gyr</td></tr>
<tr><td>10 THz</td><td>104 K</td><td>z ≈ 37</td><td>@@E19_T10T@@ Gyr</td></tr>
</table>
<p>
<b>6 i 100 GHz nigdy nie są użyteczne w CMB</b> (T_use &lt; 2.7 K); THz zegary
wchodzą w reżim bezpieczny kolejno: 10 THz od z ≈ 37 (0.07 Gyr), 3 THz od
z ≈ 10.5, 1 THz od z ≈ 2.8, 300 GHz dopiero od z ≈ 0.1 (niemal dziś).
To kosmiczna „drabina zegarów": w każdej epoce istnieją zegary o
ω_c &gt; k_BT(t)ln(1/ε)/ħ, a granica schodzi w dół z ochładzaniem.
Cutoff grawitacyjny (ω_Planck ≈ 1.85×10⁴³ rad/s): bez wpływu dla wszystkich
realistycznych ω_c (J(cutoff)/J(3D) ≈ 1); wyznacza absolutną skalę UV.
</p>
<img src="@@FIGE27@@" alt="E27 — CMB ewolucja" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R39.</span> Pakiet publikacyjny — manuskrypt zbiorczy</h2>
<p>
Cały model opisany jest w spójnym manuskrypcie <code>MANUSKRYPT.md</code>:
streszczenie, rdzeń (Lindblad, T = S, 27×, czkanie), struktura sektorowa
(N = 2…100), kosmologia (cykl, czas wstecz), kwantowy zegar (operator,
strażnik historii), program falsyfikacyjny (T0–T3, odzyskiwalność, koszt),
protokół laboratoryjny (karta, suchy bieg, wierność, różnicowy R37), skala
kosmiczna (CMB, sieć, horyzont zegarów R38), predykcje vs obserwacje,
<b>uczciwa dyskusja ograniczeń</b> (10 punktów odpowiadających na recenzje)
oraz bibliografia (Page–Wootters, Lindblad, Ando–Lindblad, Spohn, Dicke,
Zurek, Salecker–Wigner, Petz, Helstrom, Egan–Lineweaver, Planck, GRB/LIV,
SFRD, ΛCDM).
</p>
<div class="note">
<b>Najważniejszy wniosek manuskryptu.</b> Model przechodzi od deklaracji
filozoficznej do konkretnego modelu matematycznego z równaniem ewolucji,
strukturą sektorową, funkcjonałem czasu (konkurencja T0–T3), miarami
odzyskiwalności, kosztem fizycznym i zestawem przewidywań falsyfikowalnych.
Najsilniejszy wynik: mechanizm modelu (s ∝ T³) jest mechanizmem realnej
kosmologii (BBN/CMB — zgodność ilościowa), a protokół rozstrzygający T1 vs T2
(także wersja różnicowa R37) jest wykonalny na istniejącej technologii.
</div>

<h2><span class="no">R40.</span> Kosmiczny zegar w sieci z dynamiką η(T) — połączenie R8 × R36</h2>
<p>
Sieć zegarów (R36) w komórkach o <b>cyklicznej temperaturze</b> η_k(t) =
1 − (1−η_min)·sin²(π(t+φ_k)/t_cyc) (cykl kosmiczny z R8):
</p>
<div class="formula">
budżet/cykl = @@E20_BUDZET@@ nat (= wartość z R8)&nbsp; ·&nbsp;
τ_abs po 3 cyklach = @@E20_TAU3@@ ≈ 3×budżet&nbsp; ·&nbsp;
T_signed(3 cykle) ≈ 0 (pętla)
</div>
<p>
<b>Sieć definiuje czas, który przetrwa cykl kosmiczny</b>: entropia każdej
komórki wraca do ln 2 na końcu cyklu (pętla), ale <b>upływ τ_abs = Σ|ΔS|
rośnie monotonicznie</b> — czas sieci płynie dalej. Jednakowe komórki (φ = 0):
<b>σ ≡ @@E20_SIG@@</b> (emergentny czas kosmiczny, τ_net = τ₁). Offsety
fazowe (niejednorodność kosmiczna, φ ≤ 5% cyklu): σ_peak = @@E20_SPEAK@@ —
<b>synchronizacja modulowana cyklem</b> (σ rośnie w szybkich fazach, maleje
w wolnych). Dwie wskazówki czasu działają w sieci jak w pojedynczej komórce
(R8): entropia wraca, upływ trwa — ale teraz jako własność <b>kolektywna</b>.
</p>
<img src="@@FIGE28@@" alt="E28 — sieć cykliczna" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R41.</span> Formalna asymptotyka Petza dla sektorów Dickego</h2>
<p>
Dla populacyjnego kodu {|j,−j⟩, |j,−j+1⟩} w sektorze symetrycznym N kubitów
(czysty sektor jasny) mapa Petza (R29) daje asymptotykę:
</p>
<table>
<tr><th>N = 2j</th><th>F_rec</th><th>1/(N+1)</th><th>C = F_rec − 1/(N+1)</th></tr>
<tr><td>2</td><td>0.550</td><td>0.333</td><td>0.216</td></tr>
<tr><td>4</td><td>0.428</td><td>0.200</td><td>@@E20_C4@@</td></tr>
<tr><td>8</td><td>0.333</td><td>0.111</td><td>0.222</td></tr>
<tr><td>16</td><td>0.240</td><td>0.059</td><td>@@E20_C16@@</td></tr>
</table>
<div class="formula">
(i) F_rec ≥ 1/(N+1) dla wszystkich t&nbsp; ·&nbsp; (ii) F_rec → 1/(N+1) dla
t → ∞ (dokładnie: Φ(ρ) → 𝟙/(N+1), F(ρ₀, 𝟙/d) = 1/d)<br>
(iii) po transientcie superradiacyjnym: C(t) = F_rec − 1/(N+1) ≈
@@E20_CMEAN@@ ± @@E20_CSTD@@ — <b>słabo zależne od N</b> (0.23 → 0.18 dla
N: 4 → 16); C(t) → 0 wielowykładniczo (C(40) = @@E20_CT40@@)
</div>
<p>
<b>Interpretacja formalna</b>: superradiacyjny dolny szczebel (Γ₁ = Nγ) niszczy
pamięć w skali ~1/(Nγ) — stąd F_rec(N) → 1/(N+1); nadwyżka C(t), którą Petz
odzyskuje ponad stan maksymalnie mieszany, jest <b>uniwersalna względem N</b>
(po transientcie) — zależy tylko od czasu. Kontrast: sektor ciemny (j = 0:
F_rec = 1; j = 1/2: F_rec &gt; F_rec jasnego) nie ma supresji 1/(N+1) — to
formalne domknięcie hierarchii pamięci (M(t), I_c, M_F, C_mem, Petz).
</p>
<img src="@@FIGE29@@" alt="E29 — limit Petza" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE30@@" alt="E30 — dynamika Petza" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-2.0.</b> Sieć zegarów z cykliczną kąpielą łączy
kosmologię R8 z synchronizacją R36: upływ czasu jest kolektywny i monotoniczny
nawet w pętlowej entropii. Asymptotyka Petza daje formalny obraz pamięci w
sektorach Dickego: granica 1/(N+1), uniwersalna nadwyżka C(t), kontrast
ciemny/jasny — kompletna hierarchia odzyskiwalności modelu.
</div>

<h2><span class="no">R42.</span> Formalna asymptotyka Petza — dowód numeryczny dla sektorów Dickego</h2>
<p>
Pogłębienie R41 do poziomu formalnego, z trzema wynikami o charakterze
twierdzeń (weryfikowanych numerycznie z dokładnością maszynową):
</p>
<p><b>(1) Przerwa spektralna sektora symetrycznego jest niezależna od N:</b></p>
<div class="formula">
gap/γ = @@E21_GAP@@ dla N = 2..100&nbsp; ⇒&nbsp; ostateczna utrata pamięci w skali
~1/γ (superradiancja Nγ dotyczy tylko transientu 1-ekscytonowego)
</div>
<p><b>(2) Dokładny wynik analityczny dla mapy Petza</b> (zimna kąpiel, kanał
amplitudowy — kod {|↑⟩,|↓⟩}):</p>
<div class="formula">
F_rec(t) = ½·a·(2 + (1−a)²/(1−½a²)),&nbsp; a = e^{−Γt},&nbsp; Γ = Nγ
</div>
<p>
Wyprowadzone wprost z definicji Petza R(·) = σ^{1/2}Φ†(Φ(σ)^{−1/2}(·)Φ(σ)^{−1/2})σ^{1/2}
i <b>potwierdzone numerycznie do Δ = @@E21_DELTA@@</b> (dokładność maszynowa;
F_rec(t=40) = @@E21_F40@@). Granice: F_rec(0) = 1, F_rec(t→∞) → 0. Formalna
uwaga: dla N ≥ 2 (sektor d ≥ 3) referencja σ_avg jest osobliwa na
niepopulowanych poziomach — Petz wymaga pełnego rzędu (regularyzacja); dokładny
wzór obowiązuje dla kanału 2-poziomowego.
</p>
<p><b>(3) Gorąca kąpiel i limit N→∞:</b> F_rec(N,t) → 1/(N+1) dla t→∞ (dokładnie:
Φ(ρ) → 𝟙/d); nadwyżka C(t) = F_rec − 1/(N+1) ≈ @@E21_CM@@ ± 0.017 niezależna od
N (N = 4..16); dla N→∞ F_rec(jasny) → 0, a sektor ciemny (j = 0, 1/2) zachowuje
F_rec → 1. To formalne domknięcie: pamięć w sektorze jasnym ginie przez
termalizację z przerwą γ (niezależną od N), w ciemnym trwa.
</p>
<img src="@@FIGE31@@" alt="E31 — formalny Petz" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE32@@" alt="E32 — limit N→∞" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R43.</span> Entrainment faz w sieci z dynamiką η(T)</h2>
<p>
Pogłębienie R40: nie tylko synchronizacja temp, lecz <b>lockowanie faz</b>
cykli η_k(t) przez wymianę entropii (Kuramoto-like). Komórki z offsetami
fazowymi φ_k (kosmiczna niejednorodność), sprzężenie ciągnie fazy do średniej:
</p>
<div class="formula">
σ_φ: @@E21_SPHI0@@ (bez sprzężenia, stałe) → @@E21_SPHI1@@ (g_sync = 0.2 — fazy
zlockowane)&nbsp; ·&nbsp; σ_τ maleje z entrainmentem
</div>
<p>
<b>Niejednorodności fazowe znikają przez wymianę entropii</b> — sieć staje się
jednolitym czasem kosmicznym: nie tylko τ̇_k się ujednolicają (R36), ale same
cykle temperatury wchodzą w fazę. To domyka połączenie R8 (cykl kosmiczny) ×
R36 (sieć): czas sieci jest kolektywny, monotoniczny (τ_abs), odporny na
kosmiczne offsety fazowe i entrainowany do wspólnego rytmu.
</p>
<img src="@@FIGE33@@" alt="E33 — entrainment" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-2.1.</b> Formalny limit Petza: przerwa γ niezależna
od N (skala utraty pamięci), dokładny wzór F_rec(t) potwierdzony maszynowo,
F_rec → 1/(N+1) → 0 w limicie N→∞ dla sektora jasnego. Entrainment faz: sieć
zegarów z cykliczną kąpielą synchronizuje nie tylko tempa, ale i fazy —
jednolity kosmiczny czas emergentny z niejednorodnych komórek.
</div>

<h2><span class="no">R44.</span> Formalny dowód uniwersalności C(t) — asymptotyka drabiny Dickego</h2>
<p>
R42 wykazał numerycznie C(t) ≈ const niezależne od N. Formalny mechanizm:
<b>drabina dekoherencji amplitudowej</b> sektora symetrycznego ma widmo
Γ_n = n(N−n+1)·γ (szczebel n-ekscytonowy), Γ₁ = Nγ:
</p>
<div class="formula">
Γ₁ = @@E22_G1@@γ (N=100)&nbsp; ·&nbsp; Γ_n = n(N−n+1)γ&nbsp; ·&nbsp;
gap spektralny = γ (NIEZALEŻNE od N)
</div>
<p>
<b>Dowód uniwersalności C(t)</b> (schemat):
</p>
<ul>
<li><b>(i) Skala superradiacyjna</b> τ_super = 1/(Nγ) — najszybszy szczebel
(transient 1-ekscytonu) niszczy koherencje kodu w skali 1/(Nγ);</li>
<li><b>(ii) Przerwa spektralna</b> gap = γ — najwolniejszy mod; ostateczna
utrata pamięci w skali 1/γ, <b>niezależna od N</b>;</li>
<li><b>(iii) Okno uniwersalności</b> t ∈ (1/(Nγ), 1/γ): po superradiacyjnym
transiencie i przed termalizacją przerwy, F_rec − 1/(N+1) ≈ C(t) zależy
tylko od czasu (C = 0.215 ± 0.017 dla N = 4..16);</li>
<li><b>(iv) N→∞</b>: okno (1/(Nγ), 1/γ) rozszerza się (50/γ..2500/γ przy
N=50), C(t) → granicy uniwersalnej, F_rec(jasny) → 1/(N+1) → 0, ciemny → 1.</li>
</ul>
<p>
Dokładny wzór dla kanału 2-poziomowego: F_rec(t) = ½a(2+(1−a)²/(1−½a²)),
a = e^{−Γt} (R42, Δ = 9×10⁻¹⁶). Drabina domyka formalny obraz: pamięć w
sektorze jasnym umiera w dwóch skalach — szybkiej superradiacyjnej (Nγ) i
wolnej przerwy (γ, uniwersalnej).
</p>
<img src="@@FIGE34@@" alt="E34 — drabina" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R45.</span> ENTROPIA-3.0 — jawna metryka FRW</h2>
<p>
Komórki-kubity w ekspandującym Wszechświecie: a(t), T(t) = T₀(1+z(t)) z
płaskiej ΛCDM (H₀ = 67.4, Ωm = 0.315, ΩΛ = 0.685; weryfikacja: t(z=0) =
@@E22_T0@@ Gyr, t(z=1) = @@E22_T1@@ Gyr).
</p>
<div class="formula">
entropia komobowa: s·a³ = const ⇒ s(T)/s₀ = (T/T₀)³ = (1+z)³&nbsp; ·&nbsp;
S_eq komórki: maleje przy ekspansji (ochładzanie) — dτ = |dS| (upływ FRW)
</div>
<p>
<b>Zegar kosmiczny z metryki FRW jest spójny z T = S</b>: entropia komobowa
definiuje tempo (T³), horyzont definiuje budżet — S_BH na horyzoncie cząstek
(log₁₀ k_B: @@E22_SH0@@ dziś → @@E22_SH1100@@ przy z = 1100; rząd E&L 10¹²²).
S_eq komórki monotonicznie maleje przy ekspansji (η(t) = e^{−ω₀/T(t)} rośnie),
więc dτ = |dS| daje upływ czasu — ale z uczciwą uwagą: adiabatyczne s·a³ =
const oznacza brak produkcji entropii w promieniowaniu; obserwowalny wzrost
entropii Wszechświata pochodzi z grawitacji (struktury, R13). ENTROPIA-3.0
łączy więc metrykę, termodynamikę komórki i horyzont w jeden obraz: <b>czas
FRW jest upływem entropii komobowej</b>, a grawitacja (horyzont) jest jego
budżetem.
</p>
<img src="@@FIGE35@@" alt="E35 — FRW" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie ENTROPIA-3.0.</b> Formalny dowód uniwersalności C(t): drabina
Dickego (Γ_n = n(N−n+1)γ, gap = γ niezależne od N) — dwie skale utraty pamięci,
uniwersalne okno nadwyżki Petza. Metryka FRW: entropia komobowa (T³) definiuje
tempo czasu kosmicznego, horyzont — budżet (S_BH), komórka — zegar (dτ = |dS|).
Model osiąga spójność z kosmologią standardową na poziomie entropii i czasu.
</div>

<h2><span class="no">R46.</span> Pełny dowód wzoru Petza z regularyzacją (N≥2)</h2>
<p>
Formalna analiza odzyskiwalności w sektorach Dickego — <b>cztery twierdzenia</b>,
każde zweryfikowane numerycznie (pełny zapis: <code>PETZ_DOWOD.md</code>).
</p>
<p><b>Twierdzenie 1 (dokładny wzór).</b> Dla kanału amplitudowego (tempo Γ):
F_rec(t) = ½·a·(2 + (1−a)²/(1−½a²)), a = e^{−Γt}. Dowód w 7 krokach (Kraus Φ,
σ = ½(Φ(ρ₀)+Φ(ρ₁)), Φ(σ)^{−1/2}, inner, Φ†, R, fidelność) — numerycznie
<b>Δ = @@E23_D1@@</b>. Drugie słowo kodowe: F_stable = (1−½a)/(1−½a²)
(Δ = @@E23_D1A@@; F_stable(t=40) = @@E23_FSTAB@@).</p>
<p><b>Twierdzenie 2 (N≥2, zimna kąpiel).</b> Podprzestrzeń kodowa {0, 1-ekscyton}
jest niezmiennicza pod S₋, a w jej obrębie dynamika to kanał amplitudowy z
<b>Γ₁ = Nγ</b> (|⟨0-exc|S₋|1-exc⟩|² = N). Petz rzutowany (chroniony) daje
Twierdzenie 1 z Γ = Nγ — populacja 1-ekscytonu = e^{−Nγt} dokładnie.</p>
<p><b>Twierdzenie 3 (regularyzacja).</b> Dla pełnego sektora (d ≥ 3) σ jest
osobliwe; regularyzacja σ_ε = (1−ε)σ + ε·𝟙/d ma granicę ε→0, ale
<b>pełnosektorowy Petz ≠ rzutowany</b>: pełnosektorowe Φ† „przecieka" przez
szczebel 2-ekscytonowy. Numerycznie (N=2, zimna, t=40): pełny ε→0 =
@@E23_FULL2@@ = średnia rzutowana @@E23_AVG2@@ — przeciek domyka średnią.
Wniosek: dokładny wzór dotyczy chronionego (rzutowanego) Petza — zgodnie z
R34 („chroń, nie odzyskuj").</p>
<p><b>Twierdzenie 4 (asymptotyka C(t)).</b> Dla gorącej kąpieli: F_rec(N,t) →
1/(N+1) (t→∞); C(t) = F_rec − 1/(N+1) uniwersalne w oknie t ∈ (1/(Nγ), 1/γ)
(gap = γ niezależne od N — R42/R44); N→∞: F_rec(jasny) → 0, ciemny → 1.
To formalne domknięcie: odzysk w sektorze jasnym ginie w dwóch skalach —
szybkiej superradiacyjnej (Nγ) i wolnej przerwy (γ), a nadwyżka C(t) jest
uniwersalna.</p>
<img src="@@FIGE36@@" alt="E36 — dowód Petza" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<div class="note">
<b>Podsumowanie R46.</b> Wzór Petza dla kanału amplitudowego jest udowodniony
krokowo (Δ = 3×10⁻¹⁶); redukcja N≥2 do Γ = Nγ przez niezmienniczość
podprzestrzeni kodowej; regularyzacja pełnosektorowa ma granicę i wykazuje
przeciek przez drabinę (pełny = średnia rzutowana); asymptotyka C(t) przez
gap = γ. Pełny dokument dowodu: <code>PETZ_DOWOD.md</code>.
</div>

<h2><span class="no">R48.</span> ENTROPIA-4.0 — dwukomórkowy wszechświat: wymiana entropii (mikro-NESS)</h2>
<p>
Bramka audytu (Dodatek C) otwarta. Kosmologia zabawkowa <b>od mikroskopii</b>:
dwie komórki-kubity A (gorąca kąpiel Gibbsa, T_A = 3·T_B, γ_A = 27·γ_B —
s ∝ T³) i B (zimna, γ_B = 0.02; ω₀ = 1), połączone <b>kanałem wymiany</b>
σ₋^Aσ₊^B / σ₊^Aσ₋^B (κ = 0.3), który przenosi ekscytację A↔B i
<b>zachowuje E_A + E_B</b> (sprawdzone do 1e-12). Start: A w inwersji
obsadzeń (|1⟩), B w |0⟩. Dynamika populacyjna (macierz stóp 4×4, dokładna).
</p>
<table>
<tr><th>wielkość (NESS)</th><th>wartość</th><th>komentarz</th></tr>
<tr><td>S_tot(∞)</td><td>@@E48_STOT@@ nat</td><td>dS/dt → 0 (stan stacjonarny)</td></tr>
<tr><td>J_E,∞ (prąd energii A→B)</td><td>@@E48_J@@</td><td>przez kanał wymiany (zachowanie energii)</td></tr>
<tr><td>σ_NESS (produkcja entropii)</td><td>@@E48_SIG@@ nat/t</td><td>&gt; 0 — czas z budżetu nigdy nie zamiera</td></tr>
<tr><td><b>σ_NESS = J·(1/T_B − 1/T_A)</b></td><td>ratio = @@E48_CLAUS@@</td><td><b>Clausius/Onsager — zgodność 1e-6</b></td></tr>
<tr><td>σ_A/σ_B (faza zegarowa)</td><td>@@E48_RCLK@@ (szczyt @@E48_RPK@@)</td><td>rząd γ_A/γ_B = 27 — dylatacja czasu</td></tr>
<tr><td>σ_A/σ_B (NESS)</td><td>@@E48_RLT@@</td><td>produkcja → zimna komórka (Clausius)</td></tr>
</table>
<p>
Prawo Fouriera: J_E,∞ rośnie z ΔT (quasi-liniowo przy małych ΔT, nasycenie
przy dużych). Weryfikacja: κ=0 odtwarza niezależne jednokubitowe kąpiele do
1e-14. To mikroskopijny odpowiednik R13 (NESS), ale z <b>jawną wymianą</b>
między komórkami i ilościowym Clausiusem.
</p>
<img src="@@FIGE37@@" alt="E37 — dwie komórki" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE38@@" alt="E38 — termodynamika NESS" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R49.</span> ENTROPIA-4.0 — grawitacja = budżet entropii: entropowa siła i emergentna FRW</h2>
<p>
<b>(i) Entropowa siła</b> (zabawka Verlinde'owska): całkowita entropia NESS
rośnie z tempem wymiany — S_tot∞(κ): @@E49_SK0@@ (κ→0, niezależne Gibbsy
1.2617) → @@E49_SKL@@ (κ→∞). Bliżej = więcej dostępnej entropii ⇒
<b>przyciąganie</b> F(d) = T·∂S∞/∂d &lt; 0 (min @@E49_FMIN@@ przy d =
@@E49_DMIN@@). Uczciwie: profil <b>nie jest 1/d²</b> — siła najsilniejsza przy
pośrednich d (S∞ nasyca się przy silnym sprzężeniu), znika przy d→∞.
</p>
<p>
<b>(ii) Emergentna FRW.</b> Start w inwersji obsadzeń (T_A,eff &lt; 0 — ujemna
temperatura); <b>przejście przez T = ∞</b> (pg = pe) przy t = @@E49_TCROSS@@ =
„Wielki Wybuch" — osobliwość a = 0. Potem T_A,eff: +∞ → @@E49_TNESS@@
(&lt; T_A = 3 — wymiana chłodzi A poniżej równowagi kąpieli). Skala
a(τ) = T_NESS/T_eff (konwencja promieniowania T ∝ 1/a):
<b>0 → @@E49_AMAX@@ (ekspansja, t = @@E49_TAMAX@@) → 1.0000 (kontrakcja do
śmierci cieplnej)</b> — T_eff przestrzeliwuje poniżej T_NESS (wymiana drenuje
A szybciej niż kąpiel uzupełnia) ⇒ <b>odbicie</b> (cykliczny wszechświat,
por. R8). H = (1/a)(da/dτ): +∞ → 0 (zero przy t = @@E49_HZERO@@) → 0;
z = 1/a − 1: ∞ → 0 (przy kontrakcji z &lt; 0 — przesunięcie ku fioletowi).
</p>
<p>
<b>(iii) Dwa zegary i dylatacja.</b> τ_sys = S_tot: <b>skończony</b> wiek
wszechświata @@E49_TSYS@@ nat (śmierć cieplna); τ_bud = Σσ·τ:
@@E49_TBUD@@ nat i rośnie liniowo w NESS (czas trwa, σ_NESS &gt; 0).
Dylatacja: w fazie zegarowej σ_A/σ_B ≈ 27 (rząd γ_A/γ_B) — zegar w gorącej
komórce tyka ~27× szybciej; w NESS produkcja przenosi się do zimnej.
</p>
<div class="note">
<b>Uczciwe uwagi.</b> To zabawka o 2 kubitach; mapowanie a(τ) z T_eff jest
przyjęte (konwencja promieniowania), nie wyprowadzone z metryki; κ↔odległość
jest parametrem, nie geometrią; „Wielki Wybuch" (inwersja → T = ∞) to cecha
mapowania T_eff — ujemna temperatura to realne zjawisko fizyczne, ale w tej
zabawce czysto formalne. Weryfikacja: κ=0 ≡ niezależne kąpiele (1e-14);
kanał wymiany zachowuje energię dokładnie.
</div>
<img src="@@FIGE39@@" alt="E39 — entropowa siła" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE40@@" alt="E40 — emergentna FRW" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R50.</span> ENTROPIA-5.0 — pętla pomiarowa na procesorze kwantowym (IBM / Sycamore)</h2>
<p>
Konkretyzacja testu dark-sektoru (R17/R23/R27) do wykonania na realnym
sprzęcie — protokół, który może uruchomić pojedynczy użytkownik procesora
kwantowego (szkic pętli od użytkownika, <b>z poprawkami</b>):
</p>
<div class="formula">
for t in 0..t_max:  h(q0); cx(q0,q1); <b>z(q1)</b>; x(q1) &nbsp;→&nbsp; |D⟩<br>
&nbsp;&nbsp;&nbsp;&nbsp;n = t/dt × ( rz(Δω·dt) na q0 + kolektywny rozpad S− przez ancilla + reset )<br>
&nbsp;&nbsp;&nbsp;&nbsp;cx(q0,q1); h(q0); pomiar &nbsp;→&nbsp; P(|11⟩) = P_D
</div>
<p>
<b>KRYTYCZNA POPRAWKA.</b> Sekwencja <b>h, cx, x przygotowuje Ψ+ (jasny
tryplet)</b>, nie singlet: |⟨ψ|T0⟩|² = @@E50_FID_T0_BUG@@, |⟨ψ|D⟩|² ≈ 5e-34.
Wymagana bramka <code>z</code> na q1: h, cx, z, x → Ψ− (|⟨ψ|D⟩|² =
@@E50_FID_D@@). Bez poprawki cały eksperyment testowałby stan jasny.
</p>
<table>
<tr><th>start / kąpiel</th><th>P przy γt = 0.5 (t = 25, dt = 0.25, γ = 0.02)</th><th>oczekiwane</th></tr>
<tr><td>|D⟩ / KOLEKTYWNA</td><td><b>@@E50_PDK@@</b></td><td>1 (ciemny — M = 1)</td></tr>
<tr><td>|D⟩ / NIEZALEŻNA (kontrola)</td><td>@@E50_PDN@@</td><td>e^{−γt} = @@E50_EMGT@@ (falsyfikacja kąpieli)</td></tr>
<tr><td>|T0⟩ / KOLEKTYWNA</td><td>@@E50_PT0@@</td><td>e^{−2γt} = @@E50_EM2GT@@ (superradiancja)</td></tr>
<tr><td>|D⟩ + rz(Δω = 0.05) / KOLEKTYWNA</td><td>@@E50_PDZ@@</td><td>odblokowanie (przeciek |D⟩→|T0⟩)</td></tr>
</table>
<p>
<b>Kryterium falsyfikacji</b>: P_D płasko przy 1 ⇔ kąpiel KOLEKTYWNA (model);
P_D ≈ e^{−γt} ⇔ kąpiel NIEZALEŻNA. Rozróżnienie odporne na kalibrację
(pomiar różnicowy, ta sama głębokość). rz na jednym kubicie łamie symetrię
ciemną (|D⟩↔|T0⟩) — sprzętowy analog odblokowania R11.
</p>
<p>
<b>Realizacja „kolektywnego rozpadu".</b> To nie natywna bramka: kanał S−
implementujemy przez ancillę (reset co krok). Moduł waliduje unitarną osadkę
V (8×8): Tr_anc[V(ρ⊗|0⟩⟨0|)V†] ≡ kanał Krausa — <b>dokładnie (Δ =
@@E50_CIRC@@)</b>. Dekompozycja: rotacja Dickego W (|10⟩↔|01⟩, ≈ 2 CX) + dwie
rotacje Givensa sterowane ancillą + W† + reset ≈ 8–14 CX/krok.
</p>
<p>
<b>Tomografia.</b> Baza Bella (cx, h): Ψ− ↔ |11⟩ — bezpośredni witness
singletu. Pomiar losowy (X/Y/Z) + rekonstrukcja LS: F = @@E50_F@@ dla |D⟩
przy 16k strzałów. Szum strzałowy σ = √(p(1−p)/N): σ = 1% przy ≈ 2500
strzałów (minuty na IBM).
</p>
<p>
<b>Budżet NISQ (szacunki 2024–2026).</b> IBM Heron r2: krok ≈ 4–10 μs,
<b>@@E50_HERON@@ kroków</b> (T₂ ≈ 150 μs, T₂_eff = T₂/4); Google Willow:
krok ≈ 3–6 μs, <b>@@E50_WILLOW@@ kroków</b>. Uczciwie: γ i dt to wolne
parametry (skala niezmiennicza, liczy się γ·t) — dobieramy γ·dt tak, by
γ·t_max ≈ 1–2 zmieściło się w budżecie (10 kroków × γdt = 0.1 → |T0⟩ →
e^{−2} = 0.135, singlet zostaje 1).
</p>
<div class="note">
<b>Uczciwe uwagi.</b> Liczby sprzętowe to szacunki z publicznych specyfikacji —
przed startem zweryfikować na konkretnym backendzie; „kolektywny rozpad" wymaga
transpilacji osadki V do bramek bazowych (ECR/fSim); moduł jest suchym biegiem
(cyfrowym bliźniakiem) — podmiana na prawdziwy backend to zadanie inżynierskie.
</div>
<img src="@@FIGE41@@" alt="E41 — przetrwanie singletu" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE42@@" alt="E42 — tomografia" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<img src="@@FIGE43@@" alt="E43 — budżet sprzętowy" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">

<h2><span class="no">R12.</span> Wielka synteza — jeden model, pięćdziesiąt rozszerzeń</h2>
<p>
«ENTROPIA» to nie zbiór przykładów, lecz <b>jedna spójna architektura</b>: wszystko
wyrasta z jednego równania Lindblada (kubit + kąpiel) i jednej definicji
<b>T(n) = S(n)</b>. Diagram poniżej pokazuje, jak rozszerzenia łączą się w łańcuchy:
</p>
<table>
<tr><th>#</th><th>rozszerzenie</th><th>kluczowy wynik</th><th>powiązania</th></tr>
@@SYNTHTABLE@@
</table>
<div style="text-align:center;margin:18px 0">
<svg viewBox="0 0 1180 640" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#f6f9fc;border:1px solid #dbe4ec;border-radius:8px;font-family:Georgia,serif">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#5b6b7b"/>
    </marker>
  </defs>
  <!-- RDZEŃ -->
  <rect x="30" y="40" width="250" height="92" rx="10" fill="#132a3c"/>
  <text x="155" y="68" text-anchor="middle" fill="#eef5fb" font-size="15" font-weight="bold">RDZEŃ</text>
  <text x="155" y="90" text-anchor="middle" fill="#cfe2f0" font-size="11.5">kubit + kąpiel ∞ (Lindblad)</text>
  <text x="155" y="108" text-anchor="middle" fill="#cfe2f0" font-size="11.5">S: 0 → ln 2 · 27× · T = S</text>
  <rect x="30" y="170" width="250" height="58" rx="8" fill="#eef3f8" stroke="#8e44ad"/>
  <text x="155" y="192" text-anchor="middle" fill="#6b2f8e" font-size="12.5" font-weight="bold">Cechy 0–7 (rdzeń)</text>
  <text x="155" y="211" text-anchor="middle" fill="#31475c" font-size="10.5">schodki · czkanie · T_A/T_B · dekoherencja</text>
  <path d="M 155 132 L 155 168" stroke="#5b6b7b" stroke-width="1.6" marker-end="url(#arr)"/>

  <!-- grupa A: WIELE CIAŁ -->
  <text x="380" y="26" text-anchor="middle" fill="#1a5276" font-size="13" font-weight="bold">WIELE CIAŁ — struktura sektorowa entropii</text>
  <rect x="340" y="40" width="240" height="42" rx="8" fill="#eaf4fb" stroke="#2471a3"/>
  <text x="460" y="58" text-anchor="middle" fill="#14304a" font-size="12" font-weight="bold">R2 — N=2</text>
  <text x="460" y="74" text-anchor="middle" fill="#31475c" font-size="10">ciemny singlet · ln 3 vs 2·ln 2</text>
  <rect x="340" y="100" width="240" height="42" rx="8" fill="#eaf4fb" stroke="#2471a3"/>
  <text x="460" y="118" text-anchor="middle" fill="#14304a" font-size="12" font-weight="bold">R4 — N=3</text>
  <text x="460" y="134" text-anchor="middle" fill="#31475c" font-size="10">j=3/2 ⊕ 2×j=1/2 · subradiancja</text>
  <rect x="340" y="160" width="240" height="42" rx="8" fill="#eaf4fb" stroke="#2471a3"/>
  <text x="460" y="178" text-anchor="middle" fill="#14304a" font-size="12" font-weight="bold">R5 — entropia makro</text>
  <text x="460" y="194" text-anchor="middle" fill="#31475c" font-size="10">N·ln 2 vs ln(N+1)</text>
  <rect x="340" y="220" width="240" height="42" rx="8" fill="#eaf4fb" stroke="#2471a3"/>
  <text x="460" y="238" text-anchor="middle" fill="#14304a" font-size="12" font-weight="bold">R7 — losowe stany</text>
  <text x="460" y="254" text-anchor="middle" fill="#31475c" font-size="10">koherencje A↔B blokują entropię</text>
  <rect x="340" y="280" width="240" height="42" rx="8" fill="#eaf4fb" stroke="#2471a3"/>
  <text x="460" y="298" text-anchor="middle" fill="#14304a" font-size="12" font-weight="bold">R11 — odblokowanie γ_φ</text>
  <text x="460" y="314" text-anchor="middle" fill="#31475c" font-size="10">τ ∝ 1/γ_φ → pełna termalizacja</text>
  <path d="M 460 82 L 460 98" stroke="#5b6b7b" stroke-width="1.4" marker-end="url(#arr)"/>
  <path d="M 460 142 L 460 158" stroke="#5b6b7b" stroke-width="1.4" marker-end="url(#arr)"/>
  <path d="M 460 202 L 460 218" stroke="#5b6b7b" stroke-width="1.4" marker-end="url(#arr)"/>
  <path d="M 460 262 L 460 278" stroke="#5b6b7b" stroke-width="1.4" marker-end="url(#arr)"/>

  <!-- grupa B: KOSMOLOGIA -->
  <text x="710" y="26" text-anchor="middle" fill="#7b241c" font-size="13" font-weight="bold">KOSMOLOGIA — czas i cykl</text>
  <rect x="640" y="40" width="240" height="42" rx="8" fill="#fdecea" stroke="#c0392b"/>
  <text x="760" y="58" text-anchor="middle" fill="#641e16" font-size="12" font-weight="bold">R3 — zegar → tempo</text>
  <text x="760" y="74" text-anchor="middle" fill="#31475c" font-size="10">γ_eff(T): chłodzenie / przyspieszanie</text>
  <rect x="640" y="100" width="240" height="42" rx="8" fill="#fdecea" stroke="#c0392b"/>
  <text x="760" y="118" text-anchor="middle" fill="#641e16" font-size="12" font-weight="bold">R6 — gorący Wielki Wybuch</text>
  <text x="760" y="134" text-anchor="middle" fill="#31475c" font-size="10">S(0) ≈ ln 2 · czas wstecz</text>
  <rect x="640" y="160" width="240" height="42" rx="8" fill="#fdecea" stroke="#c0392b"/>
  <text x="760" y="178" text-anchor="middle" fill="#641e16" font-size="12" font-weight="bold">R8 — cykl BB → Kolaps</text>
  <text x="760" y="194" text-anchor="middle" fill="#31475c" font-size="10">czas dwustronny · pętla</text>
  <path d="M 760 82 L 760 98" stroke="#5b6b7b" stroke-width="1.4" marker-end="url(#arr)"/>
  <path d="M 760 142 L 760 158" stroke="#5b6b7b" stroke-width="1.4" marker-end="url(#arr)"/>

  <!-- grupa C: CZAS KWANTOWY -->
  <text x="1010" y="26" text-anchor="middle" fill="#6b2f8e" font-size="13" font-weight="bold">CZAS KWANTOWY</text>
  <rect x="930" y="40" width="220" height="42" rx="8" fill="#f4ecf7" stroke="#8e44ad"/>
  <text x="1040" y="58" text-anchor="middle" fill="#4a235a" font-size="12" font-weight="bold">R9 — czas jako operator</text>
  <text x="1040" y="74" text-anchor="middle" fill="#31475c" font-size="10">⟨n⟩, Δn, back-action</text>
  <rect x="930" y="100" width="220" height="42" rx="8" fill="#f4ecf7" stroke="#8e44ad"/>
  <text x="1040" y="118" text-anchor="middle" fill="#4a235a" font-size="12" font-weight="bold">R10 — koherencje zegara</text>
  <text x="1040" y="134" text-anchor="middle" fill="#31475c" font-size="10">start koherentny · κ → czas klasyczny</text>
  <path d="M 1040 82 L 1040 98" stroke="#5b6b7b" stroke-width="1.4" marker-end="url(#arr)"/>

  <!-- strzałki rdzeń → grupy -->
  <path d="M 280 78 C 320 78, 315 60, 340 60" stroke="#5b6b7b" stroke-width="1.6" fill="none" marker-end="url(#arr)"/>
  <path d="M 280 88 C 420 88, 560 88, 640 60" stroke="#5b6b7b" stroke-width="1.6" fill="none" marker-end="url(#arr)"/>
  <path d="M 280 98 C 560 110, 800 110, 930 60" stroke="#5b6b7b" stroke-width="1.6" fill="none" marker-end="url(#arr)"/>
  <text x="305" y="52" fill="#5b6b7b" font-size="9.5">N kubitów</text>
  <text x="470" y="44" fill="#5b6b7b" font-size="9.5">tempo</text>
  <text x="640" y="44" fill="#5b6b7b" font-size="9.5">kwantyzacja</text>

  <!-- SYNTEZA -->
  <rect x="330" y="540" width="520" height="72" rx="12" fill="#14304a"/>
  <text x="590" y="568" text-anchor="middle" fill="#f0c36d" font-size="16" font-weight="bold">R12 — WIELKA SYNTEZA</text>
  <text x="590" y="592" text-anchor="middle" fill="#cfe2f0" font-size="11.5">T(n) = S(n) — jeden zegar, dwanaście okien na naturę czasu</text>
  <path d="M 460 322 L 560 538" stroke="#5b6b7b" stroke-width="1.6" fill="none" marker-end="url(#arr)"/>
  <path d="M 760 202 L 620 538" stroke="#5b6b7b" stroke-width="1.6" fill="none" marker-end="url(#arr)"/>
  <path d="M 1040 142 L 800 538" stroke="#5b6b7b" stroke-width="1.6" fill="none" marker-end="url(#arr)"/>
  <path d="M 155 228 L 330 570" stroke="#5b6b7b" stroke-width="1.4" fill="none" marker-end="url(#arr)"/>
</svg>
</div>
<p>
<b>Trzy osie rozwoju:</b>
</p>
<ul>
<li><b>Termodynamika (R1).</b> Temperatura ustawia cel i tempo: S(∞) = H(1/(1+η)),
kompresja 27× trwa; zimna kąpiel potrafi cofnąć entropię (overshoot).</li>
<li><b>Wiele ciał (R2 → R4 → R5 → R7 → R11).</b> Entropia ma strukturę sektorową:
koherencje między kopiami tego samego j blokują termalizację (R7), a lokalna
dekoherencja odblokowuje ją z prawem τ ∝ 1/γ_φ (R11); po drodze: ciemne stany
(R4), ln(N+1) vs N·ln 2 (R5).</li>
<li><b>Kosmologia (R3 → R6 → R8).</b> Tempo samo-zależy od zegara (R3); gorący
start cofa czas (R6); cykl zamyka czas w pętlę (R8).</li>
<li><b>Czas kwantowy (R9 → R10).</b> Czas jako operator z nieoznaczonością i
back-action (R9); koherencje zegara i ich dekoherencja — czas klasyczny (R10).</li>
</ul>
<div class="note">
<b>Jedna strona syntezy.</b> Czasu jest dokładnie tyle, ile entropii — od ln 2
(komórka) przez ln(N+1)/N·ln 2 (wiele ciał) po N·ln 2 (pełna termalizacja).
Temperatura nadaje tempo (27×), dekoherencja nadaje kierunek (i odblokowuje
entropię), a kwantowy zegar pokazuje, że sam upływ czasu jest mierzalny tylko
z nieoznaczonością i za cenę zaburzenia Wszechświata.
</div>

<h2><span class="no">R14.</span> Predykcje modelu a dane obserwacyjne</h2>
<p>
Model to fenomenologiczna zabawka, ale jego język (produkcja entropii, sektory
ciemne, skalowanie T³, zegar-entropia) pozwala postawić <b>konkretne predykcje</b>
i zestawić je z danymi: BBN/CMB (Planck 2018), budżetem entropii Wszechświata
(Egan &amp; Lineweaver 2010), eksperymentami z subradiancją (PRL 116, 083601;
PRL 128, 203601), ograniczeniami dyskretności czasu z rozbłysków gamma
(Nature 2009; PRD 87, 122001) i historią formowania gwiazd (Hopkins &amp; Beacom
2006; Madau &amp; Dickinson 2014).
</p>
<div class="formula">
P1 — s ∝ T³ (rdzeń modelu): T_ν/T_γ = (4/11)^{1/3} = @@R14_P1@@,
N_eff = @@R14_NEFF@@ (Planck 2018) vs SM 3.044 — <b>zgodność ilościowa</b><br>
P3 — zegar entropii: wskazówka = S_obs/S_CEH ≈ @@R14_FRAC@@; entropia/barion
≈ @@R14_SPERB@@ k_B<br>
P5 — tempo produkcji entropii: s(BBN)/s(dziś) ≈ @@R14_SRATIO@@ (T³·g_*s);
obserwowane SFRD: szczyt z≈2, spadek ~10× do z=0<br>
P7 — „ciemna” entropia: model (R2 |10⟩) = @@R14_DARKFRAC@@%; obserwacja
(neutrina odsprzężone) = @@R14_NUFRAC@@% entropii promieniowania
</div>
<img src="@@FIGR14@@" alt="R14 — predykcje vs obserwacje" style="max-width:100%;border:1px solid #dbe4ec;border-radius:6px">
<table>
<tr><th>#</th><th>predykcja modelu</th><th>wartość modelu</th><th>dane obserwacyjne</th><th>ocena</th></tr>
@@R14ROWS@@
</table>
<p>
<b>Jak czytać oceny.</b> ✅ — zgodność ilościowa: P1 (skalowanie T³, rdzeń modelu)
jest <i>dokładnie</i> mechanizmem realnej kosmologii (T_ν/T_γ z zachowania
entropii, N_eff zgadza się z SM w granicach 1.8%); P2 (strzałka czasu), P7
(subradiancja obserwowana w laboratorium) i P8 (dekoherencja) są potwierdzone.
🟡 — zgodność jakościowa/analogia: budżet entropii zdominowany przez grawitację
(P6), zwalnianie kosmicznego zegara (P5), „wczesność” Wszechświata (P3).
⚠️ — napięcia, o których trzeba mówić wprost: P11 (gorący start + chłodzenie ⇒
czas wstecz) nie ma odpowiednika w obserwacjach — realny Wszechświat ochładza
się adiabatycznie <i>bez zewnętrznej kąpieli</i>, a jego entropia rośnie; P12
(cykl) przeczy obserwowanej przyspieszającej ekspansji (ΛCDM). ❓ — P10
(dyskretność czasu): brak obserwacji; ograniczenia z GRB (E_QG,1 &gt; 7.6·E_Pl)
nakładają tylko górne granice, z którymi model (wolny τ) jest zgodny.
</p>
<div class="note">
<b>Najmocniejszy test.</b> Rdzeń modelu twierdzi, że „czas jest entropią”, a
tempo produkcji entropii skaluje się jak T³. To nie jest metafora, tylko
mechanizm BBN/CMB: obserwowane T_ν/T_γ = (4/11)^{1/3} i N_eff = 2.99±0.17 są
bezpośrednią konsekwencją <i>zachowania entropii komobowej</i> (s·a³ = const)
przy e± anihilacji — dokładnie tego samego prawa, z którego model wyprowadza
swoje 27×. Model „traﬁa” w realną kosmologię, bo u podstaw ma prawdziwe prawo
termodynamiki promieniowania.
</div>
"""

EXT_SCRIPT = r"""<script>
/* ============ demo 2: sprzężenie zegar → tempo ============ */
(function(){
  var G0 = @@GAMMA_B@@, tau = @@TAU@@, N = 600, LN2 = @@LN2@@, SCALE = 0.5;
  var ALPHAS = @@JS_ALPHAS@@, THC = @@JS_THC@@, THA = @@JS_THA@@;

  var cv = document.getElementById("cv2");
  var ctx = cv.getContext("2d");
  var sl = document.getElementById("alpha");
  var accel = document.getElementById("accel");
  var readout = document.getElementById("readout2");

  function tHalfOf(alpha, mode){
    // interpolacja liniowa po tablicach ciągłych t½(α) z symulacji
    var arr = mode === 1 ? THC : THA;
    var i = 0;
    while (i < ALPHAS.length - 1 && ALPHAS[i + 1] <= alpha) i++;
    if (i >= ALPHAS.length - 1) return arr[arr.length - 1];
    var a0 = ALPHAS[i], a1 = ALPHAS[i + 1], w = (alpha - a0) / (a1 - a0);
    return arr[i] + w * (arr[i + 1] - arr[i]);
  }

  function scen(mode, alpha){
    // dokładny krok Blocha na tyknięcie z γ_eff(T)
    var rz = 0.5, rp = 0.8660254, T = 0, S = new Array(N);
    for (var i = 0; i < N; i++){
      var r = Math.sqrt(rz * rz + rp * rp);
      var p = (1 + r) / 2;
      S[i] = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
      var u = T / SCALE;
      var g = G0 * (mode === 0 ? 1 : mode === 1 ? 1 / (1 + alpha * u) : 1 + alpha * u);
      rz *= Math.exp(-2 * g * tau);
      rp *= Math.exp(-5 * g * tau);
      if (i > 0) T += Math.max(0, S[i] - S[i - 1]);
    }
    return S;
  }

  function draw(){
    var alpha = +sl.value;
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var ml = 52, mr = 14, mt = 16, mb = 34;
    var x = function(i){ return ml + (W - ml - mr) * i / (N - 1); };
    var y = function(s){ return mt + (H - mt - mb) * (1 - s / (LN2 * 1.15)); };

    ctx.strokeStyle = "#223a4f"; ctx.fillStyle = "#7d9bb3"; ctx.font = "12px Consolas,monospace";
    for (var g = 0; g <= 4; g++){
      var sv = LN2 * g / 4, yy = y(sv);
      ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(W - mr, yy); ctx.stroke();
      ctx.fillText(sv.toFixed(2), 8, yy + 4);
    }
    ctx.fillText("ln 2", 14, y(LN2) + 4);
    ctx.fillText("½ ln 2", 14, y(LN2 / 2) + 4);
    ctx.fillText("tyknięcie n", W / 2, H - 8);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("S(n) [nat]", 0, 0); ctx.restore();

    var curves = [
      { mode: 0, a: 0, c: "#8b98a5", lbl: "stały (α=0)", on: true },
      { mode: 1, a: alpha, c: "#1a5276", lbl: "chłodzenie α=" + alpha.toFixed(2), on: true },
      { mode: 2, a: 1, c: "#c0392b", lbl: "przyspieszanie α=1", on: accel.checked }
    ];
    for (var k = 0; k < curves.length; k++){
      var cu = curves[k]; if (!cu.on) continue;
      var S = scen(cu.mode, cu.a);
      ctx.strokeStyle = cu.c; ctx.lineWidth = 2;
      ctx.beginPath();
      for (var i = 0; i < N; i++){ var px = x(i), py = y(S[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
      var th = cu.mode === 0 ? tHalfOf(0, 1) : tHalfOf(cu.a, cu.mode);
      if (th < 1e9){
        ctx.fillStyle = cu.c; ctx.font = "12px Consolas,monospace";
        ctx.fillText(cu.lbl + " · t½=" + th.toFixed(2), ml + 6, mt + 15 + k * 17);
      }
    }
    readout.innerHTML =
      "t½ (czas do ½·ln 2): stały <b>" + tHalfOf(0, 1).toFixed(2) + "</b> · " +
      "chłodzenie α=" + alpha.toFixed(2) + " <b>" + tHalfOf(alpha, 1).toFixed(2) + "</b>" +
      (accel.checked ? " · przyspieszanie <b>" + tHalfOf(1, 2).toFixed(2) + "</b>" : "") +
      "<br><span style='color:#9db8cd'>Chłodzenie rozciąga późną ewolucję (czas „rozrzedza się”); " +
      "przyspieszanie zagęszcza tyknięcia na początku i szybciej zamraża czas przy nasyceniu.</span>";
  }

  sl.oninput = draw;
  accel.onchange = draw;
  draw();
})();

/* ============ demo 3: gorący Wielki Wybuch — zegar wstecz ============ */
(function(){
  var SC = @@JS_SC@@;                 // malejąca entropia S(n) (stałe γ)
  var N = SC.length;
  var S0 = SC[0], SEQ = SC[N - 1];
  var ds = @@DS_Q@@;
  var LN2 = @@LN2@@;

  var cv = document.getElementById("cv3");
  var ctx = cv.getContext("2d");
  var btnPlay = document.getElementById("btnPlay3");
  var btnNew = document.getElementById("btnNew3");
  var speed = document.getElementById("speed3");
  var readout = document.getElementById("readout3");

  var playing = false, timer = null, n = 0, T = 0, streak = 0, maxStreak = 0;
  var Tvals = [0];                       // historia zegara (malejąca)

  function poisson(mu){
    var L = Math.exp(-mu), k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
  }

  function draw(){
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var ml = 52, mr = 14, mt = 16, mb = 34;
    var x = function(i){ return ml + (W - ml - mr) * i / (N - 1); };
    var y = function(t){ return mt + (H - mt - mb) * (1 - t / (LN2 * 1.18)); };

    ctx.strokeStyle = "#223a4f"; ctx.fillStyle = "#7d9bb3"; ctx.font = "12px Consolas,monospace";
    for (var g = 0; g <= 4; g++){
      var tv = LN2 * g / 4, yy = y(tv);
      ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(W - mr, yy); ctx.stroke();
      ctx.fillText(tv.toFixed(2), 8, yy + 4);
    }
    ctx.fillText("ln 2 ≈ S(0)", 14, y(LN2) + 4);
    ctx.fillText("S(∞)", 14, y(SEQ) + 4);
    ctx.fillText("tyknięcie n", W / 2, H - 8);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("T(n) — czas = entropia [nat]", 0, 0); ctx.restore();

    // wartość oczekiwana: S(n) − S(0) (maleje od 0 do −budżet)
    ctx.strokeStyle = "#3f6a8c"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (var i = 0; i < N; i++){ var px = x(i), py = y(SC[i] - S0); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#7d9bb3"; ctx.font = "11px Consolas,monospace";
    ctx.fillText("oczekiwane: S(n) − S(0) (ochładzanie)", ml + 6, mt + 12);

    // realizacja zegara wstecz: T(n) — ujemne schodki
    ctx.strokeStyle = "#c0392b"; ctx.lineWidth = 2.2;
    ctx.beginPath();
    var tv0 = Tvals[0];
    ctx.moveTo(x(0), y(tv0));
    for (var i = 0; i < n; i++){
      ctx.lineTo(x(i + 1), y(Tvals[i]));      // poziomo (czas stoi w tyknięciu)
      ctx.lineTo(x(i + 1), y(Tvals[i + 1]));  // skok w dół (ujemne Δt)
    }
    ctx.stroke();

    // znacznik + wskaźnik zamrożenia
    var curT = Tvals[n] || 0;
    ctx.fillStyle = "#f0f6fc";
    ctx.beginPath(); ctx.arc(x(Math.min(n, N - 1)), y(curT), 4.5, 0, 7); ctx.fill();
    if (streak >= 3){
      ctx.fillStyle = "#f0c36d"; ctx.font = "13px Consolas,monospace";
      ctx.fillText("⏸ czas zamrożony: " + streak + " tyknięć", x(Math.min(n, N - 1)) + 12, y(curT) - 10);
    }
  }

  function step(){
    if (n >= N - 1){
      pause();
      readout.innerHTML = "⏹ KONIEC — zimna równowaga S(∞) = " + SEQ.toFixed(4) +
        " nat (czas zatrzymany na stałe: T = " + T.toFixed(4) + ").";
      draw(); return;
    }
    var dS = SC[n + 1] - SC[n];            // ujemne
    var mu = -dS / ds;
    var k = poisson(mu);
    var dt = -k * ds;                       // ujemny krok czasu
    if (k === 0){ streak++; maxStreak = Math.max(maxStreak, streak); }
    else { streak = 0; }
    T += dt;
    n++;
    Tvals.push(T);
    readout.innerHTML =
      "tyknięcie n = " + n + "&nbsp;·&nbsp; ΔS_n = " + dS.toFixed(5) +
      "&nbsp;·&nbsp; kwanty k_n = " + k + "&nbsp;·&nbsp; Δt_n = " + dt.toFixed(4) +
      "&nbsp;·&nbsp; T(n) = " + T.toFixed(4) +
      "&nbsp;·&nbsp; najdł. zamrożenie: " + maxStreak +
      (streak >= 3 ? " &nbsp;<span class='frozen'>⏸ CZAS STOI (wstecz)</span>" : "");
    draw();
  }

  function play(){ if (!playing){ playing = true; btnPlay.textContent = "⏸ pauza"; loop(); } }
  function pause(){ playing = false; btnPlay.textContent = "▶ start"; if (timer) clearTimeout(timer); timer = null; }
  function loop(){
    if (!playing) return;
    step();
    timer = setTimeout(loop, +speed.value);
  }

  btnPlay.onclick = function(){ playing ? pause() : play(); };
  btnNew.onclick = function(){ n = 0; T = 0; streak = 0; maxStreak = 0; Tvals = [0]; draw(); };
  draw();

  // krótki autostart
  var aut = 0;
  var autot = setInterval(function(){
    if (playing){ clearInterval(autot); return; }
    step(); aut++;
    if (aut > 80){ clearInterval(autot); play(); }
  }, 30);
})();

/* ============ demo 4: cykl Wielki Wybuch → Kolaps, dwustronny czas ============ */
(function(){
  var NC = 300, tau = 0.25, G = 0.05, LN2 = @@LN2@@, ds = @@DS_Q@@;

  var cv = document.getElementById("cv4");
  var ctx = cv.getContext("2d");
  var sl = document.getElementById("etamin");
  var two = document.getElementById("twohands");
  var readout = document.getElementById("readout4");

  function cycle(etaMin){
    var S = new Array(NC), Tabs = new Array(NC), Ts = new Array(NC);
    var rz = 0, rp = 0;
    Tabs[0] = 0; Ts[0] = 0;
    for (var i = 0; i < NC; i++){
      var e = 1 - (1 - etaMin) * Math.pow(Math.sin(Math.PI * i / NC), 2);
      var req = (1 - e) / (1 + e);
      rz = req + (rz - req) * Math.exp(-2 * G * tau);
      rp *= Math.exp(-5 * G * tau);
      var r = Math.sqrt(rz * rz + rp * rp);
      var p = (1 + r) / 2;
      S[i] = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
      if (i > 0){
        Tabs[i] = Tabs[i - 1] + Math.abs(S[i] - S[i - 1]);
        Ts[i] = Ts[i - 1] + (S[i] - S[i - 1]);
      }
    }
    return { S: S, Tabs: Tabs, Ts: Ts };
  }

  function draw(){
    var etaMin = +sl.value;
    var cy = cycle(etaMin);
    var S = cy.S;
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var ml = 52, mr = 14, mt = 16, mb = 34;
    var x = function(i){ return ml + (W - ml - mr) * i / (NC - 1); };
    var y = function(s){ return mt + (H - mt - mb) * (1 - s / (LN2 * 1.18)); };

    ctx.strokeStyle = "#223a4f"; ctx.fillStyle = "#7d9bb3"; ctx.font = "12px Consolas,monospace";
    for (var g = 0; g <= 4; g++){
      var sv = LN2 * g / 4, yy = y(sv);
      ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(W - mr, yy); ctx.stroke();
      ctx.fillText(sv.toFixed(2), 8, yy + 4);
    }
    ctx.fillText("ln 2", 14, y(LN2) + 4);
    ctx.fillText("tyknięcie n", W / 2, H - 8);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("S [nat]", 0, 0); ctx.restore();

    // S(n)
    ctx.strokeStyle = "#c0392b"; ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (var i = 0; i < NC; i++){ var px = x(i), py = y(S[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    ctx.fillStyle = "#c0392b"; ctx.font = "12px Consolas,monospace";
    ctx.fillText("S(n) — entropia (Wielki Wybuch → ekspansja → ochłodzenie → Kolaps)", ml + 6, mt + 14);

    // dwie wskazówki
    if (two.checked){
      var y2 = function(t){ return mt + (H - mt - mb) * (1 - (t + LN2 * 0.15) / (LN2 * 1.18)); };
      ctx.strokeStyle = "#8e44ad"; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (var i = 0; i < NC; i++){ var px = x(i), py = y2(cy.Ts[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
      ctx.strokeStyle = "#1a5276"; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (var i = 0; i < NC; i++){ var px = x(i), py = y2(cy.Tabs[i] - LN2 * 0.15); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
      ctx.fillStyle = "#8e44ad"; ctx.font = "11px Consolas,monospace";
      ctx.fillText("T = S − S(0) (dwustronna, wraca do zera)", ml + 6, mt + 30);
      ctx.fillStyle = "#1a5276";
      ctx.fillText("τ = Σ|ΔS| (upływ, zawsze rośnie)", ml + 6, mt + 44);
    }

    // liczby
    var Smin = Math.min.apply(null, S);
    var budzet = LN2 - Smin;
    var frak = 0;
    for (var i = 1; i < NC; i++) if (S[i] < S[i - 1]) frak++;
    readout.innerHTML =
      "S(0) = " + LN2.toFixed(4) + " (ln 2) · S_min = " + Smin.toFixed(4) +
      " · budżet czasu wstecz = " + budzet.toFixed(4) + " nat · τ = " + cy.Tabs[NC - 1].toFixed(4) +
      " = 2·budżet ✔<br><span style='color:#9db8cd'>czas płynie wstecz przez " +
      Math.round(100 * frak / (NC - 1)) + "% cyklu; przy zwrocie ΔS→0 zegar czka (zamrożenia).</span>";
  }

  sl.oninput = draw;
  two.onchange = draw;
  draw();
})();

/* ============ demo 5: odblokowanie entropii — suwak γ_φ ============ */
(function(){
  var KRZYWE = @@JS_KRZYWE@@;           // {gph: [[t,S],...]} — N=3, losowy
  var TAU90 = @@JS_TAU3@@;             // {gph: τ90}
  var KLUCZE = Object.keys(KRZYWE);
  var LN2 = @@LN2@@, CEL = 3 * LN2;

  var cv = document.getElementById("cv5");
  var ctx = cv.getContext("2d");
  var sl = document.getElementById("gphi");
  var lab = document.getElementById("gphi_label");
  var readout = document.getElementById("readout5");

  function draw(){
    var idx = +sl.value;
    var gph = parseFloat(KLUCZE[idx]);
    var pts = KRZYWE[KLUCZE[idx]];
    var tau90 = TAU90[KLUCZE[idx]];

    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var ml = 52, mr = 14, mt = 16, mb = 34;
    var tmax = pts[pts.length - 1][0];
    var x = function(t){ return ml + (W - ml - mr) * t / tmax; };
    var y = function(s){ return mt + (H - mt - mb) * (1 - s / (CEL * 1.12)); };

    ctx.strokeStyle = "#223a4f"; ctx.fillStyle = "#7d9bb3"; ctx.font = "12px Consolas,monospace";
    for (var g = 0; g <= 4; g++){
      var sv = CEL * g / 4, yy = y(sv);
      ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(W - mr, yy); ctx.stroke();
      ctx.fillText(sv.toFixed(2), 8, yy + 4);
    }
    ctx.fillText("3·ln 2", 14, y(CEL) + 4);
    ctx.fillText("t", W / 2, H - 8);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("S(t) [nat]", 0, 0); ctx.restore();

    // plateau blokady (γφ=0)
    var p0 = KRZYWE[KLUCZE[0]];
    ctx.strokeStyle = "#8b98a5"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (var i = 0; i < p0.length; i++){ var px = x(p0[i][0]), py = y(p0[i][1]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke(); ctx.setLineDash([]);

    // wybrana krzywa
    ctx.strokeStyle = "#c0392b"; ctx.lineWidth = 2.4;
    ctx.beginPath();
    for (var i = 0; i < pts.length; i++){ var px = x(pts[i][0]), py = y(pts[i][1]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();

    // τ90
    if (tau90 < tmax){
      ctx.strokeStyle = "#f0c36d"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(x(tau90), mt); ctx.lineTo(x(tau90), H - mb); ctx.stroke();
      ctx.fillStyle = "#f0c36d";
      ctx.fillText("τ90 = " + tau90.toFixed(0), x(tau90) + 6, mt + 14);
    }

    lab.textContent = "γ_φ = " + gph.toFixed(4) + "  (γ_φ/γ = " + (gph / 0.02).toFixed(2) + ")";
    readout.innerHTML =
      "S(0) = 0 · plateau blokady ≈ " + p0[p0.length - 1][1].toFixed(3) +
      " · czas do 90% 3·ln 2: <b>τ90 = " + (tau90 < 1e9 ? tau90.toFixed(0) : "∞") +
      "</b> j. czasu<br><span style='color:#9db8cd'>Im mniejsze γ_φ, tym dłużej entropia "
      + "siedzi na płaskowyżu — odblokowanie w skali τ ∝ 1/γ_φ (prawo R11).</span>";
  }

  sl.oninput = draw;
  draw();
})();
</script>
"""

def build():
    main()
