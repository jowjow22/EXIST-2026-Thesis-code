"""Doc 26 (SPEC_26.md): fatorial 2×2 da estrutura do prompt, nos 240 memes do piloto.

Pergunta: a divergência da P3 (doc 25) vem da ORDEM (texto do meme antes da instrução)
ou do ENQUADRAMENTO ("Task: decide whether ...")? Não lê nenhum voto humano.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/factorial_26.py

Retomável. Saídas: glm_reference/raw/factorial26.jsonl,
reports/factorial26_summary.md e reports/factorial26.json.
"""
import asyncio
import sys
from math import comb

from client import assert_complete, load_records, make_jobs, run_calls, server_info
from common import (FIXED, PILOT_IDS, REPORTS, assert_config, assert_server_version, clean_texts,
                    load_json, load_memes, write_json)
from prompts_26 import CELL_SHA256, CELLS

ARM, STEM, SEED = "think_off", "factorial26", FIXED["seed_reference"]
NOISE_MAX, R_MOST, R_PART = 0.02, 0.5, 0.2          # pré-registrados (SPEC_26 §4)
TESTED = {"direct_before": "só ORDEM", "task_after": "só ENQUADRAMENTO", "task_before": "ORDEM + ENQUADRAMENTO"}


def disagree(a: dict, b: dict, ids) -> float:
    """Fração de memes com rótulos diferentes; `?` em qualquer lado conta como diferente."""
    return sum(a[i] != b[i] or "?" in (a[i], b[i]) for i in ids) / len(ids)


def yes_rate(a: dict, ids) -> float:
    parsed = [i for i in ids if a[i] != "?"]
    return sum(a[i] == "YES" for i in parsed) / len(parsed) if parsed else float("nan")


def mcnemar(ref: dict, x: dict, ids) -> tuple:
    """McNemar exato bilateral: b = ref YES→x NO, c = ref NO→x YES."""
    b = sum(ref[i] == "YES" and x[i] == "NO" for i in ids)
    c = sum(ref[i] == "NO" and x[i] == "YES" for i in ids)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p


def reading(r: float) -> str:
    return "explica a maior parte" if r >= R_MOST else "explica parte" if r >= R_PART else "não explica"


def main():
    cfg = assert_config()
    assert cfg["runtime"]["chosen_arm"] == ARM
    assert_server_version(cfg, server_info()[1])
    memes = load_memes()
    pilot = load_json(PILOT_IDS)
    jobs = make_jobs(memes, pilot, clean_texts(memes, pilot), list(CELLS), (SEED,))
    asyncio.run(run_calls(jobs, ARM, cfg["runtime"]["concurrency"], templates=CELLS, stem=STEM))
    recs = assert_complete(ARM, jobs, STEM)
    assert all(r["prompt_sha256"] == CELL_SHA256[r["paraphrase"]] for r in recs.values())

    orig = load_records(ARM)                                  # doc 25, piloto, seed 42
    lab = {c: {i: recs[(i, ARM, c, SEED)]["label"] for i in pilot} for c in CELLS}
    p1 = {i: orig[(i, ARM, 1, SEED)]["label"] for i in pilot}
    p3 = {i: orig[(i, ARM, 3, SEED)]["label"] for i in pilot}
    scopes = {"total": pilot, **{l: [i for i in pilot if memes[i]["lang"] == l] for l in ("en", "es")}}

    noise = disagree(lab["direct_after"], p1, pilot)
    valid = noise <= NOISE_MAX
    d_p3 = disagree(p3, p1, pilot)
    res = {"noise_P1_rerun_vs_P1": noise, "valid": valid, "D_P3_vs_P1": d_p3,
           "yes_rate": {"P1": {s: yes_rate(p1, ids) for s, ids in scopes.items()},
                        "P3": {s: yes_rate(p3, ids) for s, ids in scopes.items()}},
           "cells": {}}
    for c in CELLS:
        b, cc, p = mcnemar(p1, lab[c], pilot)
        d = disagree(lab[c], p1, pilot)
        res["cells"][c] = {
            "parse_rate": sum(v != "?" for v in lab[c].values()) / len(pilot),
            "yes_rate": {s: yes_rate(lab[c], ids) for s, ids in scopes.items()},
            "D_vs_P1": {s: disagree(lab[c], p1, ids) for s, ids in scopes.items()},
            "D_vs_P3": disagree(lab[c], p3, pilot),
            "R": d / d_p3 if d_p3 else float("nan"),
            "mcnemar_vs_P1": {"yes_to_no": b, "no_to_yes": cc, "p": p}}
    y = {c: res["cells"][c]["yes_rate"]["total"] for c in CELLS}
    res["effects_yes_rate"] = {
        "order": (y["direct_before"] + y["task_before"] - y["direct_after"] - y["task_after"]) / 2,
        "framing": (y["task_after"] + y["task_before"] - y["direct_after"] - y["direct_before"]) / 2,
        "interaction": (y["task_before"] - y["task_after"]) - (y["direct_before"] - y["direct_after"])}

    f = lambda v: f"{v:.4f}"                                   # noqa: E731
    L = ["# Doc 26 — Fatorial 2×2 da estrutura do prompt (piloto, 240 memes)", "",
         f"Braço `{ARM}`, seed {SEED}. Referência: P1 e P3 do doc 25 (mesmos memes, mesma seed). "
         "Nenhum voto humano foi lido.", "",
         f"**Controle de ruído** (célula direta+depois = P1 rodado de novo): discordância com o P1 "
         f"original = **{f(noise)}** (limite {NOISE_MAX}) → "
         + ("válido." if valid else "**INVÁLIDO: o servidor mudou; nenhuma leitura abaixo vale.**"), "",
         f"**Referência:** discordância P3 × P1 = **{f(d_p3)}** · YES: P1 {f(res['yes_rate']['P1']['total'])}"
         f" · P3 {f(res['yes_rate']['P3']['total'])}", "",
         "| célula | muda | parse | YES (total · en · es) | discord. vs P1 (total · en · es) | discord. vs P3 "
         "| R | McNemar vs P1 (YES→NO / NO→YES, p) |", "|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        x = res["cells"][c]
        L.append(f"| `{c}` | {TESTED.get(c, 'controle (= P1)')} | {f(x['parse_rate'])} | "
                 + " · ".join(f(x["yes_rate"][s]) for s in scopes) + " | "
                 + " · ".join(f(x["D_vs_P1"][s]) for s in scopes)
                 + f" | {f(x['D_vs_P3'])} | {f(x['R'])} | {x['mcnemar_vs_P1']['yes_to_no']} / "
                 f"{x['mcnemar_vs_P1']['no_to_yes']}, p = {x['mcnemar_vs_P1']['p']:.4f} |")
    e = res["effects_yes_rate"]
    L += ["", f"Efeitos na taxa de YES: ordem {e['order']:+.4f} · enquadramento {e['framing']:+.4f} · "
              f"interação {e['interaction']:+.4f}", "",
          f"## Leitura pré-registrada (R = discordância(célula, P1) ÷ discordância(P3, P1); "
          f"≥ {R_MOST} maior parte · ≥ {R_PART} parte · < {R_PART} não explica)", ""]
    if valid:
        for c, what in TESTED.items():
            L.append(f"- {what} (`{c}`): R = {f(res['cells'][c]['R'])} → **{reading(res['cells'][c]['R'])}** "
                     "da divergência da P3")
        if res["cells"]["task_before"]["R"] < R_MOST:
            L.append("- Ordem + enquadramento juntos reproduzem menos da metade: a maior parte da "
                     "divergência da P3 vem das diferenças de redação não testadas (doc 26 §2).")
    else:
        L.append("Sem leitura: controle de ruído falhou.")
    res["readings"] = {c: reading(res["cells"][c]["R"]) for c in TESTED} if valid else None
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "factorial26_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(REPORTS / "factorial26.json", res)
    print("\n".join(L))
    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
