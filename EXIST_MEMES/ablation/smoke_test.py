"""Smoke test: 1 época da baseline, só pra confirmar que o pipeline inteiro
(dataset -> modelo -> treino -> PyEvALL -> métricas) fecha sem erro, e que o
checkpoint temporário é apagado ao final (sem estourar disco).

Uso: python ablation/smoke_test.py   (rodar de dentro de EXIST_MEMES/)
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_baseline_config  # noqa: E402
from train_one import run_single  # noqa: E402


if __name__ == "__main__":
    cfg = replace(get_baseline_config(name="smoke_test"), num_epochs=1)
    result = run_single(cfg)
    print(result)
