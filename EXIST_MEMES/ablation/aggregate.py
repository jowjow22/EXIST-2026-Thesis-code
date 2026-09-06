"""ablation/aggregate.py

Script standalone: lê `results/raw_results.csv` e regenera todas as
`results/summary_<ablation_id>.csv`, sem retreinar nada. Útil para reagregar
depois de editar KEY_METRICS ou depois de rodar mais seeds manualmente.

Uso: python aggregate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ABLATION_SUITE  # noqa: E402
from runner import summarize_ablation, RAW_CSV_PATH  # noqa: E402


def main():
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Nada para agregar -- {RAW_CSV_PATH} não existe ainda.")
        return

    for ablation_id, _ in ABLATION_SUITE:
        summarize_ablation(ablation_id)


if __name__ == "__main__":
    main()
