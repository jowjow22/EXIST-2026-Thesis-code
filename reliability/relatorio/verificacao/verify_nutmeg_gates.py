"""Reexecuta os portões sintéticos do NUTMEG (validate.py) com o estimador LEGADO
(responsabilidade 'legacy', n_iter=200, como a 2026-08-26) e com o CORRIGIDO
('fixed', n_iter=5000). Uso: python3 verify_nutmeg_gates.py <pasta_pkg> <saida> legacy|fixed"""
import sys, os, json, functools
sys.path.insert(0, sys.argv[1])
from reliability import validate as V
from reliability import nutmeg_em as NE
mode, out = sys.argv[3], sys.argv[2]
os.makedirs(out, exist_ok=True)
NITER = 200 if mode == "legacy" else 5000

class EM(NE.NutmegEM):
    def __init__(self, *a, **k):
        k.setdefault("responsibility", mode)
        super().__init__(*a, **k)
    def fit(self, annotations, subpop, n_iter=NITER, **k):
        return super().fit(annotations, subpop, n_iter=n_iter, **k)

V.NutmegEM = EM
V.OUTPUT_DIR = out
V.MINORITY_JSON = os.path.join(out, "minority_preservation.json")
V.run_recovery_grid(out_csv=os.path.join(out, "recovery_grid.csv"), verbose=False)
V.run_recovery_grid(n_items=3984, n_annotators=887, out_csv=os.path.join(out, "recovery_grid_existdensity.csv"), verbose=False)
V.test_minority_preservation(verbose=False)
V.test_constant_bias_spammer_detection(biased_spammer_label=1, verbose=False)
V.test_constant_bias_spammer_detection(biased_spammer_label=0, verbose=False)
sw = V.sweep_bias_spammer_fraction(verbose=False)
json.dump(sw, open(os.path.join(out, "sweep_bias.json"), "w"), indent=1)
print("ok", mode)
