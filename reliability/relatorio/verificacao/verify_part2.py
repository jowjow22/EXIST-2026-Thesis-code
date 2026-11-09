"""Verificação — parte 2: NUTMEG sintético, por-anotador, divisividade, ablação, votos (Fase 2).
Uso (da raiz do repositório): python3 reliability/relatorio/verificacao/verify_part2.py ."""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

B = sys.argv[1]
J = f"{B}/reliability/outputs"
FAILS = []


def check(name, got, exp, tol):
    ok = abs(got - exp) <= tol
    print(f"[{'OK ' if ok else 'XX '}] {name}: calc={got:.6g} relatório={exp:.6g}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------ NUTMEG: grade de recuperação
for fn, exp in (("recovery_grid.csv", {0.0: (0.763, 0.883), 0.15: (0.768, 0.723), 0.3: (0.757, 0.510), 0.5: (0.765, 0.317), 0.7: (0.754, 0.287), "G": (0.761, 0.544)}),
                ("recovery_grid_existdensity.csv", {0.0: (0.602, 0.692), 0.15: (0.603, 0.596), 0.3: (0.602, 0.458), 0.5: (0.595, 0.334), 0.7: (0.594, 0.287), "G": (0.599, 0.473)})):
    g = pd.read_csv(f"{J}/{fn}")
    print(fn, "linhas", len(g), "seeds", sorted(g.seed.unique()), "spam", sorted(g.spam_rate.unique()), "conv(sub)", g[g.estimator == "subpop_aware"].converged.mean())
    m = g.groupby(["divisiveness", "estimator"]).pearson_r.mean().unstack()
    for dv, (sa, stt) in exp.items():
        if dv == "G":
            check(f"{fn} global subpop", g[g.estimator == "subpop_aware"].pearson_r.mean(), sa, 6e-4)
            check(f"{fn} global single", g[g.estimator == "single_truth"].pearson_r.mean(), stt, 6e-4)
        else:
            check(f"{fn} div={dv} subpop", m.loc[dv, "subpop_aware"], sa, 6e-4)
            check(f"{fn} div={dv} single", m.loc[dv, "single_truth"], stt, 6e-4)

mp = json.load(open(f"{J}/minority_preservation.json"))
for r in mp["per_seed"]:
    print(f"minoria seed {r['seed']}: θ̂ min {r['theta_minority']:.3f} maioria {r['theta_reliable_majority']:.3f} spam {r['theta_spammers']:.3f} gap {r['gap_majority_minority']:+.3f} sep {r['separation_vs_spammers']:+.3f} n={r['n_minority']}/{r['n_reliable_majority']}/{r['n_spammers']} pass={r['pass']}")
print("minoria overall", mp["overall_pass"], mp.get("config"), {k: v for k, v in mp.items() if k not in ("per_seed", "thresholds", "config")})

for lab in (1, 0):
    d = json.load(open(f"{J}/bias_spammer_test_label{lab}.json"))
    print(f"\n== spammer viés constante label={lab} ({'sempre YES' if lab else 'sempre NO'}) overall={d['overall_pass_by_mode']}")
    for r in d["rows"]:
        print(f"  seed {r['seed']} {r['mode']:20s} enviesado {r['theta_biased_spammers']:.3f} uniforme {r['theta_uniform_spammers']:.3f} fiável {r['theta_reliable']:.3f} gap {r['gap_reliable_biased']:+.3f} |Δ| {r['gap_biased_vs_uniform']:.3f} c1={r['c1_ceiling']} c2={r['c2_gap_reliable']} c3={r.get('c3_gap_uniform', r.get('c3'))}")
    for k in d:
        if k not in ("rows", "thresholds", "config", "overall_pass_by_mode"):
            print("  chave extra:", k, json.dumps(d[k])[:1500])

# ------------------------------------------------ por-anotador
pg = json.load(open(f"{J}/perann_synthetic_gate.json"))
print("\nperann seeds", pg["seeds"])
for r in pg["recovery_grid"]:
    print("  ", r)
print("  recovery_verdicts", pg["recovery_verdicts"])
print("  minority", json.dumps(pg["minority"])[:2000])
print("  minority_verdicts", pg["minority_verdicts"])
print("  bias", json.dumps(pg["bias_spammer"])[:1500])
for k in pg:
    if k not in ("thresholds", "seeds", "recovery_grid", "recovery_verdicts", "minority", "minority_verdicts", "bias_spammer"):
        print("  chave extra", k, json.dumps(pg[k])[:3000])

# ------------------------------------------------ divisividade
dv = json.load(open(f"{J}/divisiveness_metrics.json"))
print("\ndivisividade chaves:", list(dv)[:20])
print(json.dumps(dv)[:2500])

# ------------------------------------------------ ablação: recalcular R1/R2 e histograma de densidades
data = json.load(open(f"{B}/EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json", encoding="utf-8"))
ds = pd.DataFrame(list(data.values()))
votos = ds[["id_EXIST", "lang", "annotators", "labels_task2_1"]].explode(["annotators", "labels_task2_1"]).rename(
    columns={"annotators": "annotator", "labels_task2_1": "label_a"})
perfil = votos.assign(yes=(votos.label_a == "YES").astype(int)).groupby("annotator").agg(n_votes=("yes", "size"), n_yes=("yes", "sum"))
perfil["yes_rate"] = perfil.n_yes / perfil.n_votes
perfil["extremo"] = (perfil.yes_rate > 0.90) | (perfil.yes_rate < 0.10)
perfil["lado"] = np.where(perfil.yes_rate > 0.9, "alto", np.where(perfil.yes_rate < 0.1, "baixo", "-"))
perfil["lang"] = votos.groupby("annotator")["lang"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "ambas")
print("\nextremos", perfil.extremo.sum(), perfil.lado.value_counts().to_dict())
print(pd.crosstab(perfil.lang, perfil.lado))


def sortear_controle(seed):
    rng_local = np.random.default_rng(seed)
    alvo = perfil.loc[perfil["extremo"], ["lang", "n_votes"]]
    pool = perfil.loc[~perfil["extremo"]]
    escolhidos = []
    for (lg, nv), grupo in alvo.groupby(["lang", "n_votes"]):
        cand = pool.index[(pool["lang"] == lg) & (pool["n_votes"] == nv) & (~pool.index.isin(escolhidos))].to_numpy()
        k = len(grupo)
        if len(cand) < k:
            cand = pool.index[(pool["lang"] == lg) & (pool["n_votes"].between(nv - 3, nv + 3)) & (~pool.index.isin(escolhidos))].to_numpy()
        escolhidos.extend(rng_local.choice(cand, size=k, replace=False))
    return pd.Index(escolhidos)


vb = votos.assign(yes=(votos.label_a == "YES").astype(int))
ext = perfil.index[perfil.extremo]
ctrl = sortear_controle(42)
print("controle seed 42: n", len(ctrl), "yes_rate média", round(perfil.loc[ctrl, "yes_rate"].mean(), 3), "min", round(perfil.loc[ctrl, "yes_rate"].min(), 3), "max", round(perfil.loc[ctrl, "yes_rate"].max(), 3), perfil.loc[ctrl, "lang"].value_counts().to_dict())
print("extremos por língua", perfil.loc[ext, "lang"].value_counts().to_dict(), "n_votes extremos", perfil.loc[ext, "n_votes"].describe().round(2).to_dict())
for nm, rem in (("R1_rem_extremos", ext), ("R2_rem_controle", ctrl)):
    sub = vb[~vb.annotator.isin(rem)]
    g = sub.groupby("id_EXIST")["yes"]
    lab, n = g.mean(), g.size()
    stored = json.load(open(f"{J}/soft_labels_taskA_{nm}.json"))
    mx = max(abs(stored[str(k)][0] - lab[k]) for k in lab.index)
    check(f"{nm} recalculado == arquivo (máx |Δ|)", mx, 0, 1e-12)
    check(f"{nm} nº itens", len(lab), 3984, 0)
    print(f"  {nm}: histograma de votos/item {dict(sorted(Counter(n).items()))} | votos removidos {len(vb) - len(sub)}")
    OUT_hist = dict(sorted(Counter(n).items()))
# densidade cruzada
n1 = vb[~vb.annotator.isin(ext)].groupby("id_EXIST").size()
n2 = vb[~vb.annotator.isin(ctrl)].groupby("id_EXIST").size()
print("tabela cruzada votos/item R1 × R2:")
print(pd.crosstab(n1.rename("R1"), n2.rename("R2")))
print("itens com densidade diferente entre R1 e R2:", int((n1 != n2).sum()))

# ------------------------------------------------ Fase 2 run A
v = json.load(open(f"{J}/experiment_votes_report.json"))
for r in v["runs"]:
    mtr = r["votes"]["record"]["metrics"]
    print(f"\nFase 2 {r['arm']} seed {r['seed']} E={r['E']}:", {k: round(x['nll'], 4) for k, x in mtr.items()})
    check("b − b_perm", mtr["b"]["nll"] - mtr["b_perm"]["nll"], -0.0271, 5e-5)
    check("b − d4_train", mtr["b"]["nll"] - mtr["d4_train"]["nll"], 0.0147, 5e-5)
    check("b − a", mtr["b"]["nll"] - mtr["a"]["nll"], -0.0244, 5e-5)
    print("  diag:", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r["diagnostics"].items()})
print("chaves top do votes report:", list(v))
print("\nFALHAS:", FAILS if FAILS else "nenhuma")
