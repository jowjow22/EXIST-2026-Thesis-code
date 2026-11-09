"""Verificação independente — parte 1: dados, soft labels, θ, runs treinadas.

Recalcula a partir dos artefactos primários (dataset, θ, soft labels, logs de
metrics_history) e compara com os números escritos nos relatórios.
Uso (da raiz do repositório): python3 reliability/relatorio/verificacao/verify_part1.py .
"""
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
from scipy import stats as st

B = sys.argv[1]
OUT = {}
FAILS = []


def check(name, got, exp, tol):
    ok = abs(got - exp) <= tol
    print(f"[{'OK ' if ok else 'XX '}] {name}: calc={got:.6g} relatório={exp:.6g} (tol {tol})")
    if not ok:
        FAILS.append(name)
    OUT[name] = got


# ---------------------------------------------------------------- dados
data = json.load(open(f"{B}/EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json", encoding="utf-8"))
items = list(data.values())
votes = [(it["id_EXIST"], a, l, it["lang"]) for it in items
         for a, l in zip(it["annotators"], it["labels_task2_1"])]
anns = {a for _, a, _, _ in votes}
check("n_itens", len(items), 3984, 0)
check("n_anotadores", len(anns), 887, 0)
check("n_votos", len(votes), 23904, 0)
print("rótulos Task A:", Counter(l for _, _, l, _ in votes))
yes_frac = np.mean([l == "YES" for _, _, l, _ in votes])
check("fração votos YES (todos)", yes_frac, 0.5558, 5e-5)
nyes = Counter(sum(l == "YES" for l in it["labels_task2_1"]) for it in items)
print("n_YES/6 (3984 itens):", dict(sorted(nyes.items())))
for k, v in {0: 373, 1: 459, 2: 535, 3: 614, 4: 672, 5: 759, 6: 572}.items():
    check(f"itens com {k} YES", nyes[k], v, 0)
check("itens maioria estrita YES", sum(nyes[k] for k in (4, 5, 6)) / 3984, 0.5028, 5e-5)
per_ann = Counter(a for _, a, _, _ in votes)
pa = np.array(list(per_ann.values()))
print(f"votos/anotador: média {pa.mean():.3f} mediana {np.median(pa)} min {pa.min()} max {pa.max()}")
check("votos/anotador média", pa.mean(), 26.95, 0.005)
check("anotadores <15 votos", int((pa < 15).sum()), 11, 0)
langs = {}
for _, a, _, lg in votes:
    langs.setdefault(a, set()).add(lg)
strata = Counter("both" if len(s) > 1 else s.pop() for s in langs.values())
print("estratos de língua:", strata)
check("anotadores es_only", strata["es"], 437, 0)
check("anotadores en_only", strata["en"], 431, 0)
check("anotadores bilingues", strata["both"], 19, 0)
for lg, n_it, n_an, n_v, un, yr in (("es", 1979, 456, 11874, 474, 0.573), ("en", 2005, 450, 12030, 471, 0.539)):
    sel = [it for it in items if it["lang"] == lg]
    a_ = {a for it in sel for a in it["annotators"]}
    v_ = [l for it in sel for l in it["labels_task2_1"]]
    u_ = sum(len(set(it["labels_task2_1"])) == 1 for it in sel)
    check(f"painel {lg} itens", len(sel), n_it, 0)
    check(f"painel {lg} anotadores", len(a_), n_an, 0)
    check(f"painel {lg} votos", len(v_), n_v, 0)
    check(f"painel {lg} unânimes", u_, un, 0)
    check(f"painel {lg} taxa YES", np.mean([x == "YES" for x in v_]), yr, 5e-4)
check("esperado unânime se intercambiáveis (p=0,5558)", yes_frac**6 + (1 - yes_frac)**6, 0.036, 0.002)
# painéis fixos
pan = Counter(frozenset(it["annotators"]) for it in items)
check("painéis distintos", len(pan), 151, 0)
ann_pan = Counter()
for p in pan:
    for a in p:
        ann_pan[a] += 1
check("anotadores num único painel", sum(v == 1 for v in ann_pan.values()), 868, 0)


# Krippendorff alfa nominal (binário), dados completos por item (6 votos)
def kripp_alpha(items):
    o = np.zeros((2, 2))
    for it in items:
        v = [1 if l == "YES" else 0 for l in it["labels_task2_1"]]
        m = len(v)
        c = Counter(v)
        for a in (0, 1):
            for b in (0, 1):
                o[a, b] += (c[a] * (c[b] - (a == b))) / (m - 1)
    n = o.sum()
    nc = o.sum(axis=1)
    do = (o[0, 1] + o[1, 0])
    de = (nc[0] * nc[1] * 2) / (n - 1)
    return 1 - do / de


check("Krippendorff α global", kripp_alpha(items), 0.2791, 5e-5)

# ---------------------------------------------------------------- split e soft labels
split = json.load(open(f"{B}/reliability/outputs/split_taskA_frozen.json"))
tr, va = set(split["train_ids"]), set(split["val_ids"])
check("split treino", len(tr), 3386, 0)
check("split validação", len(va), 598, 0)
check("split interseção", len(tr & va), 0, 0)
raw = json.load(open(f"{B}/reliability/outputs/soft_labels_taskA_raw.json"))
calc_raw = {it["id_EXIST"]: np.mean([l == "YES" for l in it["labels_task2_1"]]) for it in items}
check("raw == n_yes/6 (máx |Δ|)", max(abs(raw[k][0] - calc_raw[k]) for k in raw), 0.0, 1e-12)
val_votes = sum(6 for k in va)
check("votos de validação", val_votes, 3588, 0)
tr_yes = np.mean([l == "YES" for it in items if it["id_EXIST"] in tr for l in it["labels_task2_1"]])
check("taxa YES só treino", tr_yes, 0.5552, 5e-5)
nyes_tr = Counter(sum(l == "YES" for l in it["labels_task2_1"]) for it in items if it["id_EXIST"] in tr)
print("n_YES/6 treino:", dict(sorted(nyes_tr.items())))


def load_theta(fn):
    d = json.load(open(f"{B}/reliability/outputs/{fn}"))
    th = d.get("theta", d)
    if isinstance(th, dict) and "theta" not in th:
        return {k: (v if isinstance(v, (int, float)) else v.get("theta")) for k, v in th.items()}
    return th


def theta_map(fn):
    d = json.load(open(f"{B}/reliability/outputs/{fn}"))
    for key in ("theta", "thetas", "annotators"):
        if key in d:
            t = d[key]
            if isinstance(t, dict):
                return {k: float(v if not isinstance(v, dict) else v["theta"]) for k, v in t.items()}
            if isinstance(t, list):
                return {r["annotator"]: float(r["theta"]) for r in t}
    raise KeyError(list(d.keys())[:10])


def weighted(theta):
    out = {}
    for it in items:
        w = np.array([theta[a] for a in it["annotators"]])
        v = np.array([l == "YES" for l in it["labels_task2_1"]], float)
        out[it["id_EXIST"]] = float((w * v).sum() / w.sum())
    return out


def shift(a, b):
    ks = list(raw)
    d = np.array([abs(a[k] - b[k]) for k in ks])
    cross = sum((a[k] > 0.5) != (b[k] > 0.5) for k in ks)
    return d, cross


try:
    th_fix = theta_map("theta_taskA_single_truth_K1_fixed.json")
    th_leg = theta_map("theta_taskA_single_truth_K1.json")
except KeyError as e:
    print("estrutura θ:", e)
    raise
wf_file = json.load(open(f"{B}/reliability/outputs/soft_labels_taskA_weighted_fixed.json"))
wl_file = json.load(open(f"{B}/reliability/outputs/soft_labels_taskA_weighted.json"))
wf_calc = weighted(th_fix)
wl_calc = weighted(th_leg)
check("weighted_fixed arquivo == Σθv/Σθ com θ FIXED (máx |Δ|)", max(abs(wf_file[k][0] - wf_calc[k]) for k in raw), 0.0, 1e-9)
check("weighted (legado) arquivo == Σθv/Σθ com θ LEGADO (máx |Δ|)", max(abs(wl_file[k][0] - wl_calc[k]) for k in raw), 0.0, 1e-9)
rawp = {k: raw[k][0] for k in raw}
d, c = shift({k: wf_file[k][0] for k in raw}, rawp)
check("fixed: média |w−raw|", d.mean(), 0.0617, 5e-5)
check("fixed: sd |w−raw|", d.std(), 0.0656, 1e-4)
check("fixed: máx |w−raw|", d.max(), 0.3525, 5e-5)
check("fixed: itens |Δ|>0,05", int((d > 0.05).sum()), 1861, 0)
check("fixed: itens |Δ|>0,10", int((d > 0.10).sum()), 934, 0)
check("fixed: cruzam 0,5 (>=0,5, definição do código)", sum((rawp[k] >= 0.5) != (wf_file[k][0] >= 0.5) for k in raw), 421, 0)
check("fixed: Pearson raw×w", np.corrcoef([rawp[k] for k in raw], [wf_file[k][0] for k in raw])[0, 1], 0.9661, 5e-5)
d, c = shift({k: wl_file[k][0] for k in raw}, rawp)
check("legado: média |w−raw|", d.mean(), 0.0720, 5e-5)
check("legado: cruzam 0,5 (>=)", sum((rawp[k] >= 0.5) != (wl_file[k][0] >= 0.5) for k in raw), 443, 0)
d, c = shift({k: wf_file[k][0] for k in raw}, {k: wl_file[k][0] for k in raw})
check("fixed vs legado: média |Δ|", d.mean(), 0.0246, 5e-5)
check("fixed vs legado: cruzam 0,5", c, 134, 2)
tf = np.array([th_fix[a] for a in sorted(th_fix)])
tl = np.array([th_leg[a] for a in sorted(th_fix)])
check("θ legado×fixed Pearson", np.corrcoef(tf, tl)[0, 1], 0.8923, 5e-5)
check("θ legado×fixed média|Δ|", np.abs(tf - tl).mean(), 0.0794, 5e-5)
check("θ fixed média", tf.mean(), 0.4826, 5e-5)
check("θ legado média", tl.mean(), 0.5352, 5e-5)
check("θ cruzam 0,5", int(((tf > 0.5) != (tl > 0.5)).sum()), 95, 0)
check("θ fixed <0,5", int((tf < 0.5).sum()), 458, 0)
check("θ fixed <0,3", int((tf < 0.3).sum()), 208, 0)
print(f"θ fixed: min {tf.min():.4f} mediana {np.median(tf):.4f} max {tf.max():.4f}")
# rem: relatório soft_labels_taskA_report_fixed cita θ legado no §2 -> verificar qual bate
print("θ fixed range ↔ relatório [0.0294, 0.9521], mediana 0.4861")

# permutações
perm_means = []
for s in (101, 202, 303, 404, 505):
    p = json.load(open(f"{B}/reliability/outputs/soft_labels_taskA_permuted_seed{s}_fixed.json"))
    dd = np.mean([abs(p[k][0] - rawp[k]) for k in raw])
    perm_means.append(dd)
    print(f"perm seed {s}: média|Δ| {dd:.6f}")
check("permutado média", np.mean(perm_means), 0.051351, 5e-6)
check("razão permutado/real", np.mean(perm_means) / 0.061732, 0.832, 5e-4)

# ---------------------------------------------------------------- runs treinadas
PRIM = "eval_task_a_icmsoftnorm_soft"
METRICS = ["eval_task_a_icmnorm_hard", PRIM, "eval_task_b_icmnorm_hard",
           "eval_task_b_icmsoftnorm_soft", "eval_task_c_icmnorm_hard", "eval_task_c_icmsoftnorm_soft"]


def run(fn):
    h = json.load(open(f"{B}/EXIST_MEMES/metrics_history/{fn}.json"))
    ev = [r for r in h if "eval_loss" in r]
    if not ev:
        return {"ev": [], "E": None, "Eep": None, "n": 0, "end": "train_runtime" in h[-1]}
    i = int(np.argmin([r["eval_loss"] for r in ev]))
    return {"ev": ev, "E": i, "Eep": ev[i]["epoch"], "n": len(ev), "end": "train_runtime" in h[-1]}


def paired(a_files, b_files, key=PRIM, final=False):
    d = []
    for fa, fb in zip(a_files, b_files):
        ra, rb = run(fa), run(fb)
        ia = ra["n"] - 1 if final else ra["E"]
        ib = rb["n"] - 1 if final else rb["E"]
        d.append(rb["ev"][ib][key] - ra["ev"][ia][key])
    d = np.array(d)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / math.sqrt(len(d)))
    p = 2 * st.t.sf(abs(t), len(d) - 1)
    return d, d.mean(), sd, t, p, d.mean() / sd


A0 = ["20260831_065521_raw_seed42", "20260831_075933_raw_seed7", "20260830_225133_raw_seed2025"]
A1 = ["20260830_205756_weighted_fixed_seed42", "20260830_215815_weighted_fixed_seed7", "20260830_194907_weighted_fixed_seed2025"]
R1 = ["20260905_112128_R1_rem_extremos_seed42", "20260905_122152_R1_rem_extremos_seed7", "20260905_140756_R1_rem_extremos_seed2025"]
R2 = ["20260905_150442_R2_rem_controle_seed42", "20260905_155802_R2_rem_controle_seed7", "20260905_222414_R2_rem_controle_seed2025"]
for f in A0 + A1 + R1 + R2:
    r = run(f)
    print(f"{f}: n_eval={r['n']} E=ép{r['Eep']} fim={r['end']} eval_loss(E)={r['ev'][r['E']]['eval_loss']:.5f} prim(E)={r['ev'][r['E']][PRIM]:.4f}")
d, m, sd, t, p, dz = paired(A0, A1)
print("15ep A1−A0 por seed:", np.round(d, 4))
check("15ep Δ médio", m, -0.0025, 5e-5)
check("15ep sd", sd, 0.0020, 5e-5)
check("15ep t", t, -2.16, 0.01)
check("15ep p", p, 0.164, 0.001)
check("15ep d_z", dz, -1.25, 0.01)
for key in METRICS:
    for fin in (False, True):
        d, m, sd, t, p, dz = paired(A0, A1, key, fin)
        OUT[f"15ep {key} {'final' if fin else 'E'}"] = [list(d), m, sd, t, p, dz]
        print(f"  15ep {key:32s} {'final' if fin else 'E    '} Δ={np.round(d,4)} m={m:+.4f} sd={sd:.4f} t={t:+.2f} p={p:.3f} dz={dz:+.2f}")
d, m, sd, t, p, dz = paired(R2, R1)
print("abl R1−R2:", np.round(d, 4))
check("abl R1−R2 Δ", m, -0.0077, 5e-5)
check("abl R1−R2 p", p, 0.206, 0.001)
check("abl R1−R2 d_z", dz, -1.06, 0.01)
d, m, sd, t, p, dz = paired(A0, R1)
check("abl R1−A0 Δ", m, -0.0064, 5e-5)
check("abl R1−A0 p", p, 0.048, 0.001)
d, m, sd, t, p, dz = paired(A0, R2)
check("abl R2−A0 Δ", m, +0.0013, 5e-5)
check("abl R2−A0 p", p, 0.799, 0.001)
for key in METRICS:
    for nm, (a, b) in {"R1−R2": (R2, R1), "R1−A0": (A0, R1), "R2−A0": (A0, R2)}.items():
        for fin in (False, True):
            d, m, sd, t, p, dz = paired(a, b, key, fin)
            print(f"  abl {nm} {key:32s} {'final' if fin else 'E    '} m={m:+.4f} sd={sd:.4f} p={p:.3f} dz={dz:+.2f}")

# 2 épocas (θ legado)
TWO_RAW = ["20260830_030421", "20260830_154057_raw_seed7", "20260830_163503_raw_seed2025"]
TWO_W = ["20260830_151242", "20260830_155444_weighted_seed7", "20260830_162416_weighted_seed2025"]
d = []
for fa, fb in zip(TWO_RAW, TWO_W):
    ra, rb = run(fa), run(fb)
    print(f"2ep {fa} n={ra['n']} | {fb} n={rb['n']}")
    d.append(rb["ev"][-1][PRIM] - ra["ev"][-1][PRIM])
d = np.array(d)
print("2ep Δ (época 2):", np.round(d, 4))
check("2ep Δ médio (época final)", d.mean(), 0.0045, 5e-5)
t = d.mean() / (d.std(ddof=1) / math.sqrt(3))
check("2ep t", t, 1.604, 0.005)
check("2ep p", 2 * st.t.sf(abs(t), 2), 0.2499, 0.001)
sp = np.std([run(f)["ev"][-1][PRIM] for f in TWO_RAW], ddof=1)
print(f"2ep spread baseline sd(ddof=1)={sp:.4f}; amplitude={np.ptp([run(f)['ev'][-1][PRIM] for f in TWO_RAW]):.4f}")

# confusão
for f in sorted(glob.glob(f"{B}/EXIST_MEMES/metrics_history/*conf_*.json")):
    if f.endswith(("_confusion.json", "_votes.json")):
        continue
    fn = os.path.basename(f)[:-5]
    r = run(fn)
    if not r["n"]:
        print(f"conf {fn}: sem avaliações (teste curto)"); continue
    s = int(re.search(r"seed(\d+)", fn).group(1))
    a0 = run(A0[[42, 7, 2025].index(s)])
    pe = r["ev"][r["E"]][PRIM] if r["n"] else float("nan")
    pa_ = a0["ev"][a0["E"]][PRIM]
    print(f"conf {fn}: n_eval={r['n']} E=ép{r['Eep'] if r['n'] else '-'} prim(E)={pe:.4f} A0={pa_:.4f} Δ={pe-pa_:+.4f} eval_loss(E)={r['ev'][r['E']]['eval_loss'] if r['n'] else float('nan'):.5f}")
CLR = ["20260907_234357_conf_M1_lam0.1_clr0.01_seed42", "20260908_192123_conf_M1_lam0.1_clr0.01_seed7", "20260909_225545_conf_M1_lam0.1_clr0.01_seed2025"]
d, m, sd, t, p, dz = paired(A0, CLR)
print("clr Δ:", np.round(d, 4))
check("clr Δ médio", m, 0.0009, 5e-5)
check("clr sd", sd, 0.0011, 5e-5)
check("clr t", t, 1.380, 0.01)
check("clr p", p, 0.30, 0.01)
a0v = [run(f)["ev"][run(f)["E"]][PRIM] for f in A0]
check("sd A0 entre seeds (E)", np.std(a0v, ddof=1), 0.0059, 5e-5)
print("A0 em E por seed:", np.round(a0v, 4))

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_part1_out.json"), "w"), indent=1, default=float)
print("\nFALHAS:", FAILS if FAILS else "nenhuma")
