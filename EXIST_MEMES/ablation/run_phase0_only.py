"""Roda só a FASE 0 (baseline x 3 seeds, 30 épocas cada) -- para medir tempo real
por run antes de disparar a suite de ablação inteira (FASE 1, ~33 runs a mais).

Uso: python ablation/run_phase0_only.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import run_phase_0  # noqa: E402


if __name__ == "__main__":
    start = time.time()
    run_phase_0()
    elapsed_hours = (time.time() - start) / 3600
    print(f"\nFASE 0 completa em {elapsed_hours:.2f}h para 3 runs "
          f"({elapsed_hours / 3:.2f}h/run em média).")
    print("Ajuste MEMES_HOURS_PER_TRAINING com esse valor antes de rodar runner.py "
          "para a suite completa (FASE 1).")
