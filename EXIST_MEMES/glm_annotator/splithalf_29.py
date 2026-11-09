"""Teste offline do doc 29 (SPEC_29 §1): o GLM ajuda a prever a outra metade dos humanos?

Protocolo: memes de treino com 6 votos YES/NO e referência do GLM. As 20 divisões 3 vs 3 são
usadas nos dois sentidos (a metade A prevê a metade B). O mapa de quantis q e a constante c
são ajustados uma vez, sobre o p_humano de 6 votos (só 4 números + 1, desprezível para vazamento).
A entropia cruzada usa p cortado em [1e-3, 1 − 1e-3]. IC 95% por bootstrap de memes,
2 000 reamostragens, default_rng(2026).

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/splithalf_29.py
"""
import csv
import itertools

import numpy as np

from common import REF, REPORTS, SOFT_RAW, TRAIN_IDS, load_json, load_memes
from doc29 import quantile_map
from weights import glm_soft, load_train_panels

EPS = 1e-3


def ce(p, t):
    p = np.clip(p, EPS, 1 - EPS)
    return -(t * np.log(p) + (1 - t) * np.log(1 - p))


def main():
    raw, train = load_json(SOFT_RAW), load_json(TRAIN_IDS)
    panels, _ = load_train_panels(load_memes(), train)
    with open(REF / "glm_votes.csv", encoding="utf-8") as f:
        g = {r["meme_id"]: glm_soft([r["label_p1"], r["label_p2"], r["label_p3"]]) for r in csv.DictReader(f)}
    ids = [i for i in train if g.get(i) is not None and len(panels[i]) == 6]
    ph = {i: raw[i][0] for i in ids}
    q = quantile_map(ph, {i: g[i] for i in ids})
    c = float(np.mean(list(ph.values())))
    preds = {"só a metade (humano puro)": lambda a, i: a,
             "0,5·metade + 0,5·c (= smooth50)": lambda a, i: .5 * a + .5 * c,
             "0,5·metade + 0,5·GLM sem correção": lambda a, i: .5 * a + .5 * g[i],
             "0,5·metade + 0,5·q(GLM) (= glmq50)": lambda a, i: .5 * a + .5 * q[g[i]],
             "só q(GLM)": lambda a, i: q[g[i]]}
    splits = [(list(A), [j for j in range(6) if j not in A]) for A in itertools.combinations(range(6), 3)]
    mse = {k: np.zeros(len(ids)) for k in preds}
    cre = {k: np.zeros(len(ids)) for k in preds}
    for n, i in enumerate(ids):
        v = np.array([x == "YES" for x in panels[i].values()], float)
        ab = [(v[A].mean(), v[B].mean()) for A, B in splits]
        for k, f in preds.items():
            mse[k][n] = np.mean([(f(a, i) - b) ** 2 for a, b in ab])
            cre[k][n] = np.mean([ce(f(a, i), b) for a, b in ab])
    L = ["# Doc 29 — teste offline split-half (3 anotadores preveem os outros 3)", "",
         f"{len(ids)} memes de treino · 20 divisões por meme · q = "
         + ", ".join(f"{k:.3f}→{v:.3f}" for k, v in q.items()) + f" · c = {c:.4f}", "",
         "| previsor da outra metade | MSE ↓ | entropia cruzada ↓ |", "|---|---|---|"]
    L += [f"| {k} | {mse[k].mean():.4f} | {cre[k].mean():.4f} |" for k in preds]
    rng = np.random.default_rng(2026)
    t, s = "0,5·metade + 0,5·q(GLM) (= glmq50)", "0,5·metade + 0,5·c (= smooth50)"
    for name, m in (("MSE", mse), ("entropia cruzada", cre)):
        d = m[t] - m[s]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        L.append(f"\nglmq50 − smooth50, {name}: {d.mean():+.4f} (IC 95% [{np.percentile(bs, 2.5):+.4f}; "
                 f"{np.percentile(bs, 97.5):+.4f}])")
    (REPORTS / "splithalf_29.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
