"""Compara os portões NUTMEG reexecutados (legacy/fixed, nesta pasta) com os artefactos de 2026-08-26.
Uso (da raiz do repositório): python3 reliability/relatorio/verificacao/verify_gates_compare.py ."""
import json, os, sys
import pandas as pd
B = sys.argv[1]
G = os.path.dirname(os.path.abspath(__file__))
DIRS = {"json": f"{B}/reliability/outputs", "gates_legacy": f"{G}/gates_legacy", "gates_fixed": f"{G}/gates_fixed"}
for fn in ("recovery_grid.csv", "recovery_grid_existdensity.csv"):
    o = pd.read_csv(f"{DIRS['json']}/{fn}"); L = pd.read_csv(f"{G}/gates_legacy/{fn}"); F = pd.read_csv(f"{G}/gates_fixed/{fn}")
    k = ["divisiveness", "spam_rate", "seed", "estimator"]
    m = o.merge(L, on=k, suffixes=("_o", "_l"))
    print(f"{fn}: legado reexecutado vs arquivo: máx|Δr| = {(m.pearson_r_o - m.pearson_r_l).abs().max():.2e}, n={len(m)}")
    for nm, g in (("arquivo", o), ("legado", L), ("corrigido", F)):
        t = g.groupby(["divisiveness", "estimator"]).pearson_r.mean().unstack()
        print(f"  {nm:9s}", " ".join(f"div{d}: sub {t.loc[d,'subpop_aware']:.3f} one {t.loc[d,'single_truth']:.3f}" for d in t.index),
              f"| global sub {g[g.estimator=='subpop_aware'].pearson_r.mean():.3f} one {g[g.estimator=='single_truth'].pearson_r.mean():.3f}",
              f"| convergiu sub {g[g.estimator=='subpop_aware'].converged.mean():.2f} one {g[g.estimator=='single_truth'].converged.mean():.2f}")
for nm in ("json", "gates_legacy", "gates_fixed"):
    mp = json.load(open(f"{DIRS[nm]}/minority_preservation.json"))
    print(f"minoria [{nm}]", [(r["seed"], round(r["theta_minority"], 3), round(r["theta_reliable_majority"], 3), round(r["theta_spammers"], 3), r["pass"]) for r in mp["per_seed"]], mp["overall_pass"])
for lab in (1, 0):
    for nm in ("json", "gates_legacy", "gates_fixed"):
        d = json.load(open(f"{DIRS[nm]}/bias_spammer_test_label{lab}.json"))
        rows = [(r["seed"], r["mode"][:9], round(r["theta_biased_spammers"], 3), round(r["gap_biased_vs_uniform"], 3)) for r in d["rows"]]
        print(f"viés label{lab} [{nm}] {d['overall_pass_by_mode']}")
        print("   ", rows)
for nm in ("gates_legacy", "gates_fixed"):
    sw = json.load(open(f"{DIRS[nm]}/sweep_bias.json"))
    print(f"sweep [{nm}]", [(r["biased_frac"], round(r["theta_biased_mean"], 3), [round(x, 3) for x in r["theta_biased_per_seed"]]) for r in sw])
