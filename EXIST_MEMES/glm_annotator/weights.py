"""F3′ (SPEC_27.md §8): pesos dos anotadores pela distância ao soft label do GLM, e novos
soft labels da Task 2.1. Só cálculo.

As funções de cima são puras (sem I/O) e reproduzem o Apêndice A do SPEC_27.md. O doc 25
(voto duro do GLM) é o caso particular em que todo `p_GLM` ∈ {0, 1}. `main()` faz o I/O,
avalia o G2′, roda os testes do Apêndice A (portão 12.4) e os asserts do portão 12.5.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/weights.py
"""
import csv
import json
import shutil
import sys
from collections import Counter
from math import log2
from pathlib import Path

from common import LABELS

METHOD = "doc27_soft_distance"


def binary_entropy(p: float) -> float:
    return 0.0 if p in (0.0, 1.0) else -(p * log2(p) + (1 - p) * log2(1 - p))


def n_yes(votes: dict) -> int:
    return sum(v == "YES" for v in votes.values())


def glm_soft(labels) -> float | None:
    """Fração de YES nas 3 paráfrases; None se alguma não tiver rótulo."""
    return None if any(l not in LABELS for l in labels) else sum(l == "YES" for l in labels) / len(labels)


def deviation_terms(memes: dict, p_glm: dict, peer: bool = False) -> list:
    """[(annotator, meme_id, w_ai, d_ai)] sobre os memes incluídos (SPEC_27 §8): fora os
    unânimes entre os humanos e os sem referência do GLM (p_glm None).

    d_ai = |1[voto = YES] − ref|, com ref = p_GLM; com peer=True (checagem a), ref = fração
    de YES entre os OUTROS votos do meme. w_ai = 1 − H(fração de YES entre os outros).
    """
    out = []
    for mid, votes in memes.items():
        p = p_glm.get(mid)
        if p is None or n_yes(votes) in (0, len(votes)):
            continue
        for a in votes:
            others = [v for b, v in votes.items() if b != a]
            k = sum(o == "YES" for o in others) / len(others)
            ref = k if peer else p
            out.append((a, mid, 1 - binary_entropy(k), abs((votes[a] == "YES") - ref)))
    return out


def deviation_from_terms(terms: list, annotators) -> dict:
    """{annotator: D_a ou None}. None = nenhum meme incluído (ou Σw = 0)."""
    num, den = {}, {}
    for a, _, w, d in terms:
        num[a] = num.get(a, 0.0) + w * d
        den[a] = den.get(a, 0.0) + w
    return {a: (num[a] / den[a] if den.get(a, 0) > 0 else None) for a in annotators}


def annotator_deviation(memes: dict, p_glm: dict, peer: bool = False) -> dict:
    return deviation_from_terms(deviation_terms(memes, p_glm, peer),
                                {a for v in memes.values() for a in v})


def vote_weights(D: dict) -> dict:
    return {a: (1.0 if d is None else 1.0 - d) for a, d in D.items()}


def weighted_soft_label(votes: dict, r: dict) -> float:
    return sum(r[a] * (v == "YES") for a, v in votes.items()) / sum(r[a] for a in votes)


# ------------------------------------------------------------------ I/O (fora das puras)

def load_train_panels(memes_json: dict, train_ids: list) -> tuple[dict, int]:
    """{meme_id: {annotator: vote}} só do treino. Votos fora de YES/NO saem e são contados."""
    panels, dropped = {}, 0
    for i in train_ids:
        votes, anns = memes_json[i]["labels_task2_1"], memes_json[i]["annotators"]
        assert len(votes) == len(anns), f"{i}: {len(votes)} votos vs {len(anns)} anotadores"
        assert len(set(anns)) == len(anns), f"{i}: anotador repetido no painel"
        panels[i] = {a: v for a, v in zip(anns, votes) if v in LABELS}
        dropped += len(votes) - len(panels[i])
    return panels, dropped


def load_glm_votes(path) -> dict:
    """{meme_id: linha do glm_votes.csv}."""
    with open(path, encoding="utf-8") as f:
        return {r["meme_id"]: r for r in csv.DictReader(f)}


def glm_labels(row: dict) -> list:
    return [row[f"label_p{p}"] for p in (1, 2, 3)]


def soft_votes(glm_rows: dict) -> dict:
    """{meme_id: p_GLM ou None}."""
    return {i: glm_soft(glm_labels(r)) for i, r in glm_rows.items()}


def g2_prime(glm_rows: dict, threshold: float) -> dict:
    """G2′ (SPEC_27 §2): parse ≥ limiar em cada idioma."""
    out = {}
    for lang in ("en", "es"):
        labels = [l for r in glm_rows.values() if r["lang"] == lang for l in glm_labels(r)]
        out[lang] = sum(l in LABELS for l in labels) / len(labels)
    return {"parse_rate": out, "pass": all(v >= threshold for v in out.values())}


def annotator_demographics(memes_json: dict, ids: list) -> tuple[dict, int]:
    """{annotator: (gênero, idade)} e nº de anotadores em mais de um painel."""
    demo, panels_of = {}, {}
    for i in ids:
        it = memes_json[i]
        key = tuple(it["annotators"])
        for a, g, age in zip(it["annotators"], it["gender_annotators"], it["age_annotators"]):
            assert demo.setdefault(a, (g, age)) == (g, age), f"{a}: demografia inconsistente"
            panels_of.setdefault(a, set()).add(key)
    return demo, sum(len(p) > 1 for p in panels_of.values())


def annotator_rows(panels: dict, p_glm: dict, demo: dict) -> list:
    terms = deviation_terms(panels, p_glm)
    anns = {a for v in panels.values() for a in v}
    D_glm = deviation_from_terms(terms, anns)
    D_peer = annotator_deviation(panels, p_glm, peer=True)
    r_glm, r_peer = vote_weights(D_glm), vote_weights(D_peer)
    n_inc, mass = Counter(), Counter()
    for a, _, w, _ in terms:
        n_inc[a] += 1
        mass[a] += w
    n_memes, n_y = Counter(), Counter()
    for votes in panels.values():
        for a, v in votes.items():
            n_memes[a] += 1
            n_y[a] += v == "YES"
    return [{"annotator_id": a, "gender": demo[a][0], "age": demo[a][1],
             "n_memes": n_memes[a], "n_included": n_inc[a], "mass_included": mass[a],
             "D_glm": D_glm[a], "r_glm": r_glm[a], "D_peer": D_peer[a], "r_peer": r_peer[a],
             "yes_rate": n_y[a] / n_memes[a], "no_included_memes": D_glm[a] is None}
            for a in sorted(anns)]


def main():
    from common import (REF, SOFT_GLMBASE, SOFT_GLMW, SOFT_RAW, T, TRAIN_IDS, VAL_IDS, VOTES_CSV,
                        assert_config, load_json, load_memes, set_gate, sha256_file,
                        update_runtime, write_json)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))  # outro `tests` pode fazer sombra
    from test_weights import run_all

    assert_config()
    assert run_all() == 0, "testes do Apêndice A falharam (portão 12.4)"

    memes = load_memes()
    train, val = load_json(TRAIN_IDS), load_json(VAL_IDS)
    assert not set(train) & set(val), "treino ∩ validação ≠ ∅ (portão 12.3)"
    glm_rows = load_glm_votes(VOTES_CSV)
    assert set(glm_rows) == set(train), "glm_votes.csv não cobre exatamente os memes de treino"
    g2 = g2_prime(glm_rows, T["parse"])
    set_gate("G2prime", g2["pass"], **g2, method=METHOD)
    assert g2["pass"], f"G2′ falhou: {g2}"

    panels, dropped = load_train_panels(memes, train)          # só treino (portão 12.3)
    p_glm = soft_votes(glm_rows)
    demo, multi = annotator_demographics(memes, train)

    rows = annotator_rows(panels, p_glm, demo)
    r = {x["annotator_id"]: x["r_glm"] for x in rows}
    raw = load_json(SOFT_RAW)
    soft = {}
    for i in train:
        p_old, p_new = raw[i][0], weighted_soft_label(panels[i], r)
        assert abs(p_old - n_yes(panels[i]) / len(panels[i])) < 1e-12, f"{i}: raw ≠ fração de votos"
        soft[i] = {"old": raw[i], "new": [p_new, 1 - p_new]}

    # portão 12.5
    for i, s in soft.items():
        p, q = s["new"]
        assert 0.0 <= p <= 1.0 and abs(p + q - 1) < 1e-12, f"{i}: soft label inválido {s['new']}"
        if n_yes(panels[i]) in (0, len(panels[i])):
            assert p == s["old"][0], f"{i}: meme unânime mudou ({s['old'][0]} → {p})"
    glmw = {i: (soft[i]["new"] if i in soft else raw[i]) for i in raw}
    assert len(glmw) == 3984 and all(glmw[i] == raw[i] for i in val)
    assert json.dumps({i: glmw[i] for i in val}) == json.dumps({i: raw[i] for i in val}), \
        "labels de validação diferem dos originais (portão 12.5)"

    with open(REF / "annotator_weights.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(REF / "soft_labels_weighted.json", soft)
    write_json(SOFT_GLMW, glmw)
    shutil.copyfile(SOFT_RAW, SOFT_GLMBASE)
    assert sha256_file(SOFT_GLMBASE) == sha256_file(SOFT_RAW)

    n_no = sum(x["no_included_memes"] for x in rows)
    n_ref = sum(p is not None for p in p_glm.values())
    update_runtime(method=METHOD,
                   soft_labels_sha256={"raw": sha256_file(SOFT_RAW), "glmw": sha256_file(SOFT_GLMW),
                                       "glmbase": sha256_file(SOFT_GLMBASE)},
                   f3={"method": METHOD, "dropped_votes": dropped, "n_annotators": len(rows),
                       "no_included_memes": n_no, "multi_panel_annotators": multi,
                       "memes_with_glm_reference": n_ref})
    set_gate("F3", True, method=METHOD)
    print(f"F3′ ok ({METHOD}) · G2′ {g2['parse_rate']} · {len(rows)} anotadores "
          f"({n_no} sem meme incluído, r = 1) · {len(soft)} soft labels de treino · "
          f"{len(train) - n_ref} memes sem referência do GLM · votos removidos: {dropped}\n"
          f"Gravados: annotator_weights.csv, soft_labels_weighted.json, "
          f"{SOFT_GLMW.name}, {SOFT_GLMBASE.name}. Próximo: checks.py")


if __name__ == "__main__":
    main()
