"""Casos extremos de concordância humana no treino e o que o GLM faz neles.

Grupos (6 anotadores por meme, Task 2.1):
    1Y5N: 1 YES e 5 NO      3Y3N: empate 3-3      5Y1N: 5 YES e 1 NO

Só lê arquivos já gravados (nenhuma chamada ao servidor). Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/select_extremes.py

Grava em glm_reference/extremes/:
    extremes.csv  — um meme por linha: grupo, votos humanos, labels do GLM off/on
    extremes.json — o mesmo + imagem, texto, demografia e raciocínio do GLM (usado pelo viewer.py)
e o relatório reports/extremes.md. A ordem dentro de cada grupo é embaralhada com
default_rng(2026): avaliar só os primeiros N de cada grupo continua sendo uma amostra aleatória.
"""
import csv

import numpy as np

from client import load_records
from common import REF, REPORTS, TRAIN_IDS, load_json, load_memes, write_json
from weights import glm_soft

OUT = REF / "extremes"
GROUPS = {1: "1Y5N", 3: "3Y3N", 5: "5Y1N"}        # nº de YES humanos → grupo
ARMS = {"think_off": REF / "glm_votes.csv", "think_on": REF / "glm_votes_think_on.csv"}
SEED = 42


def read_votes(path):
    with open(path, encoding="utf-8") as f:
        return {r["meme_id"]: r for r in csv.DictReader(f)}


def main():
    memes, train = load_memes(), load_json(TRAIN_IDS)
    votes = {a: read_votes(p) for a, p in ARMS.items()}
    recs = {a: load_records(a) for a in ARMS}

    rng = np.random.default_rng(2026)
    sel = []
    for k, g in GROUPS.items():
        ids = sorted(i for i in train if memes[i]["labels_task2_1"].count("YES") == k)
        sel += [(g, ids[j]) for j in rng.permutation(len(ids))]

    rows, items = [], []
    for g, i in sel:
        m = memes[i]
        row = {"meme_id": i, "group": g, "lang": m["lang"],
               "human_yes": m["labels_task2_1"].count("YES"),
               "human_labels": "|".join(m["labels_task2_1"])}
        glm = {}
        for a in ARMS:
            labs = [votes[a][i][f"label_p{p}"] for p in (1, 2, 3)]
            for p, l in zip((1, 2, 3), labs):
                row[f"{a}_p{p}"] = l
            row[f"{a}_vote"] = votes[a][i]["glm_vote"]
            glm[a] = [{"label": l, "reasoning": (recs[a].get((i, a, p, SEED)) or {}).get("reasoning_content")}
                      for p, l in zip((1, 2, 3), labs)]
        rows.append(row)
        items.append({"id": i, "group": g, "lang": m["lang"], "text": m["text"], "img": m["path_memes"],
                      "humans": [{"label": l, "gender": s, "age": a} for l, s, a in
                                 zip(m["labels_task2_1"], m["gender_annotators"], m["age_annotators"])],
                      "glm": glm})

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "extremes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(OUT / "extremes.json", items)

    # Relatório: taxa de YES do GLM (todas as 3 paráfrases) e voto da maioria das 3, por grupo
    L = ["# Casos extremos de concordância humana — comportamento do GLM (treino)", "",
         "Taxa de YES = fração de YES nas 3 paráfrases (P1–P3). Voto = maioria das 3. "
         "Humanos: fração de YES dos 6 anotadores.", "",
         "| grupo | n | humanos | YES think_off | YES think_on | voto YES off | voto YES on | "
         "p_GLM off (0/⅓/⅔/1) | p_GLM on (0/⅓/⅔/1) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, g in GROUPS.items():
        r = [x for x in rows if x["group"] == g]
        for lang in ("total", "en", "es"):
            s = [x for x in r if lang == "total" or x["lang"] == lang]
            yes = {a: np.mean([x[f"{a}_p{p}"] == "YES" for x in s for p in (1, 2, 3)]) for a in ARMS}
            vote = {a: np.mean([x[f"{a}_vote"] == "YES" for x in s]) for a in ARMS}
            dist = {a: "/".join(str(sum(glm_soft([x[f"{a}_p{p}"] for p in (1, 2, 3)]) == v for x in s))
                                for v in (0, 1 / 3, 2 / 3, 1)) for a in ARMS}
            name = g if lang == "total" else f"↳ {lang}"
            L.append(f"| {name} | {len(s)} | {k / 6:.3f} | {yes['think_off']:.3f} | {yes['think_on']:.3f} | "
                     f"{vote['think_off']:.3f} | {vote['think_on']:.3f} | {dist['think_off']} | "
                     f"{dist['think_on']} |")
    L += ["", "Leitura: em 1Y5N o esperado de um anotador típico é ~17% de YES; em 5Y1N, ~83%; "
          "em 3Y3N, ~50%. p_GLM não soma n quando alguma paráfrase não foi parseada.",
          "", "Arquivos: `glm_reference/extremes/extremes.csv` e `extremes.json`. "
          "Visualizador: `python3 EXIST_MEMES/glm_annotator/viewer.py`."]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "extremes.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
