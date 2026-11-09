"""Stress-sweep do spammer enviesado (modo adversarial) em 2×2: responsabilidade {legacy, fixed} × n_iter {200, 5000}."""
import sys, json
sys.path.insert(0, sys.argv[1])
from reliability import validate as V
from reliability import nutmeg_em as NE
out = {}
for resp in ("legacy", "fixed"):
    for nit in (200, 5000):
        class EM(NE.NutmegEM):
            def __init__(self, *a, **k):
                k.setdefault("responsibility", resp); super().__init__(*a, **k)
            def fit(self, a, s, n_iter=nit, **k):
                return super().fit(a, s, n_iter=n_iter, **k)
        V.NutmegEM = EM
        sw = V.sweep_bias_spammer_fraction(verbose=False)
        out[f"{resp}_{nit}"] = [(r["biased_frac"], round(r["theta_biased_mean"], 3)) for r in sw]
        print(resp, nit, out[f"{resp}_{nit}"], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
