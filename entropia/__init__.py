"""
«ENTROPIA» — kosmologiczny model czasu = entropii.

Rdzeń (core) + rozszerzenia R1–R13 (extensions) + budowa raportu (report).
Odtwarzalne: python3 zrob_raport.py
"""
from . import core, extensions, report
from .core import (operatory, superoperator, symuluj, entropia, czystosc,
                   bloch, delta_entropii, zegar_stochastyczny, liczby_kluczowe,
                   LN2, GAMMA_A, GAMMA_B, DELTA_TAU, DELTA_S_Q)
from .extensions import run_all

__version__ = "5.0.0"
__all__ = ["core", "extensions", "report", "run_all", "operatory",
           "superoperator", "symuluj", "entropia", "czystosc", "bloch",
           "delta_entropii", "zegar_stochastyczny", "liczby_kluczowe",
           "LN2", "GAMMA_A", "GAMMA_B", "DELTA_TAU", "DELTA_S_Q"]
