"""F2 (§7): referência do GLM nos memes de TREINO. Não olha voto humano.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/run_reference.py

Retomável: as chamadas do piloto no braço escolhido são reaproveitadas.
"""
import asyncio
import csv
import sys

from client import assert_complete, confident, majority, make_jobs, run_calls, server_info
from common import (FIXED, LABELS, REPORTS, T, TRAIN_IDS, VOTES_CSV, assert_config,
                    assert_server_version, clean_texts, load_json, load_memes, require_gate,
                    set_gate)

SEED = FIXED["seed_reference"]


def main():
    cfg = assert_config()
    require_gate(cfg, "G1")
    assert_server_version(cfg, server_info()[1])
    arm = cfg["runtime"]["chosen_arm"]
    memes = load_memes()
    train = load_json(TRAIN_IDS)
    texts = clean_texts(memes, train)
    jobs = make_jobs(memes, train, texts, (1, 2, 3), (SEED,))
    asyncio.run(run_calls(jobs, arm, cfg["runtime"]["concurrency"]))
    recs = assert_complete(arm, jobs)

    rows = []
    for i in train:
        labels = [recs[(i, arm, p, SEED)]["label"] for p in (1, 2, 3)]
        rows.append({"meme_id": i, "lang": memes[i]["lang"],
                     "label_p1": labels[0], "label_p2": labels[1], "label_p3": labels[2],
                     "glm_vote": majority(labels) or "", "glm_confident": confident(labels)})
    with open(VOTES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    stats = {}
    for scope in ("total", "en", "es"):
        rs = [r for r in rows if scope == "total" or r["lang"] == scope]
        labels = [r[f"label_p{p}"] for r in rs for p in (1, 2, 3)]
        stats[scope] = {"n_memes": len(rs),
                        "parse_rate": sum(l in LABELS for l in labels) / len(labels),
                        "confident_frac": sum(r["glm_confident"] for r in rs) / len(rs),
                        "no_vote_frac": sum(not r["glm_vote"] for r in rs) / len(rs)}
    ok = all(stats[l]["parse_rate"] >= T["parse"] and stats[l]["confident_frac"] >= T["confident"]
             for l in ("en", "es"))

    L = ["# F2 — Referência do GLM (treino)", "",
         f"Braço `{arm}`, P1–P3, seed {SEED}, {len(train)} memes de treino. "
         "Nenhum voto humano foi lido.", "",
         "| escopo | n_memes | parse_rate | confident_frac | no_vote_frac |", "|---|---|---|---|---|",
         *[f"| {s} | {v['n_memes']} | {v['parse_rate']:.4f} | {v['confident_frac']:.4f} | "
           f"{v['no_vote_frac']:.4f} |" for s, v in stats.items()], "",
         f"**G2 {'APROVADO' if ok else 'FALHOU'}** (por idioma: parse ≥ {T['parse']} e "
         f"confiantes ≥ {T['confident']}). " + ("Próximo: `weights.py`." if ok else "Parar (§11).")]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "reference_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    set_gate("G2", ok, stats=stats, arm=arm)
    print("\n".join(L))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
