"""Recalcula os diagnósticos das matrizes do braço clr=1e-2 (Fase 1) na época E=3.
viés = C[NO->YES] − C[YES->NO]; competência = C[YES,YES] + C[NO,NO] − 1;
fracção de variância na direcção do viés = var((c10 − c01)/√2) / (var c01 + var c10).
Uso (da raiz do repositório): python3 reliability/relatorio/verificacao/verify_confusion_matrices.py ."""
import json, sys
import numpy as np, pandas as pd
B = sys.argv[1]
prof = pd.read_csv(f"{B}/reliability/outputs/behavioral_profiles.csv").set_index("annotator_id")
for fn in ("20260907_234357_conf_M1_lam0.1_clr0.01_seed42", "20260908_192123_conf_M1_lam0.1_clr0.01_seed7",
           "20260909_225545_conf_M1_lam0.1_clr0.01_seed2025"):
    d = json.load(open(f"{B}/EXIST_MEMES/metrics_history/{fn}_confusion.json"))
    C = np.array([s for s in d["snapshots"] if s["epoch"] == 3.0][0]["matrices"], float)
    yr = prof.loc[d["annotator_ids"], "yes_rate"].values
    a, b = C[:, 0, 1], C[:, 1, 0]
    bias, comp = b - a, C[:, 0, 0] + C[:, 1, 1] - 1
    sl, ic = np.polyfit(yr, bias, 1)
    print(fn.split("_")[-1], f"viés = {sl:.3f}·yes_rate {ic:+.3f}  R² = {np.corrcoef(yr, bias)[0,1]**2:.3f}",
          f"| fracção var. viés {np.var((b - a) / np.sqrt(2)) / (np.var(a) + np.var(b)):.3f}",
          f"| sd competência {comp.std(ddof=1):.3f} | r(comp, |yes_rate − 0,55|) {np.corrcoef(comp, np.abs(yr - 0.55))[0,1]:+.3f}",
          f"| r(comp, yes_rate) {np.corrcoef(comp, yr)[0,1]:+.3f}")
