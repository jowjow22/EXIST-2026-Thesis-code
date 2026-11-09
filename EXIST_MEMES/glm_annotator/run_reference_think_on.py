"""Doc 30 (SPEC_30.md): referência do GLM no treino com o thinking LIGADO, para ver se o viés
de YES do `think_off` (doc 25, F2) se mantém.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/run_reference_think_on.py

Retomável (mesmo comando de novo): as 720 chamadas `think_on` do piloto são reaproveitadas.
Não toca em `glm_votes.csv`, `chosen_arm` nem nos portões: grava só
`glm_votes_think_on.csv` e `reports/think_on_vs_off.md`.
"""
import asyncio
import csv
import statistics as st

from client import assert_complete, confident, majority, make_jobs, run_calls, server_info
from common import (FIXED, LABELS, REF, REPORTS, SOFT_RAW, TRAIN_IDS, VOTES_CSV, assert_config,
                    assert_server_version, clean_texts, load_json, load_memes, require_gate,
                    update_runtime)
from weights import glm_soft

SEED = FIXED["seed_reference"]
OUT_CSV = REF / "glm_votes_think_on.csv"
PERSISTS = 0.10   # SPEC_30 §3: viés persiste se a taxa de YES do think_on ≥ humanos + 0,10


def yes_rate(labels) -> float:
    v = [l for l in labels if l in LABELS]
    return sum(l == "YES" for l in v) / len(v)


def main():
    cfg = assert_config()
    require_gate(cfg, "G1")
    assert_server_version(cfg, server_info()[1])
    memes, train = load_memes(), load_json(TRAIN_IDS)
    jobs = make_jobs(memes, train, clean_texts(memes, train), (1, 2, 3), (SEED,))
    asyncio.run(run_calls(jobs, "think_on", cfg["runtime"]["concurrency"]))
    recs = assert_complete("think_on", jobs)

    rows = []
    for i in train:
        labels = [recs[(i, "think_on", p, SEED)]["label"] for p in (1, 2, 3)]
        rows.append({"meme_id": i, "lang": memes[i]["lang"],
                     "label_p1": labels[0], "label_p2": labels[1], "label_p3": labels[2],
                     "glm_vote": majority(labels) or "", "glm_confident": confident(labels)})
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(VOTES_CSV, encoding="utf-8") as f:
        off = {r["meme_id"]: r for r in csv.DictReader(f)}
    raw = load_json(SOFT_RAW)
    arms = {"think_on": {r["meme_id"]: r for r in rows}, "think_off": off}
    lab = lambda r: [r[f"label_p{p}"] for p in (1, 2, 3)]

    L = ["# Doc 30 — GLM com thinking ligado vs desligado (treino)", "",
         f"P1–P3, seed {SEED}, {len(train)} memes de treino, mesmos parâmetros do doc 25.", "",
         "| escopo | humanos (média p_YES) | YES think_off | YES think_on | Δ (on − off) | "
         "parse think_on | confiantes think_on | voto igual on/off |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for scope in ("total", "en", "es"):
        ids = [i for i in train if scope == "total" or memes[i]["lang"] == scope]
        y = {a: yes_rate([l for i in ids for l in lab(arms[a][i])]) for a in arms}
        on_labels = [l for i in ids for l in lab(arms["think_on"][i])]
        both = [i for i in ids if arms["think_on"][i]["glm_vote"] and off[i]["glm_vote"]]
        res[scope] = {"human": st.mean(raw[i][0] for i in ids), "off": y["think_off"], "on": y["think_on"],
                      "parse_on": sum(l in LABELS for l in on_labels) / len(on_labels),
                      "confident_on": sum(arms["think_on"][i]["glm_confident"] for i in ids) / len(ids),
                      "vote_agree": sum(arms["think_on"][i]["glm_vote"] == off[i]["glm_vote"]
                                        for i in both) / len(both)}
        r = res[scope]
        L.append(f"| {scope} | {r['human']:.3f} | {r['off']:.3f} | {r['on']:.3f} | "
                 f"{r['on'] - r['off']:+.3f} | {r['parse_on']:.4f} | {r['confident_on']:.3f} | "
                 f"{r['vote_agree']:.3f} |")

    levels = (0, 1 / 3, 2 / 3, 1)
    L += ["", "Distribuição de p_GLM (fração de YES nas 3 paráfrases):", "",
          "| braço | 0 | 1/3 | 2/3 | 1 |", "|---|---|---|---|---|"]
    for a in arms:
        p = [glm_soft(lab(arms[a][i])) for i in train]
        L.append(f"| {a} | " + " | ".join(str(sum(x == k for x in p)) for k in levels) + " |")

    t = res["total"]
    persists = t["on"] >= t["human"] + PERSISTS
    L += ["", f"**Decisão (SPEC_30 §3): " + (
        f"o viés de YES PERSISTE com thinking ligado ({t['on']:.3f} ≥ {t['human']:.3f} + {PERSISTS}). "
        "Próximo: testar outros modelos.**" if persists else
        f"o viés de YES NÃO persiste com thinking ligado ({t['on']:.3f} < {t['human']:.3f} + {PERSISTS}). "
        "Próximo: discutir com o orientador antes de testar outros modelos.**")]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "think_on_vs_off.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    update_runtime(doc30={"results": res, "bias_persists": persists})
    print("\n".join(L))


if __name__ == "__main__":
    main()
