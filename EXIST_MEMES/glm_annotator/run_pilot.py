"""F1 (§6): piloto (escolha do thinking) e F1b (estabilidade). Não olha voto humano.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/run_pilot.py --stage pilot
    python3 EXIST_MEMES/glm_annotator/run_pilot.py --stage stability

Retomável: se cair no meio, rode o mesmo comando de novo.
"""
import argparse
import asyncio
import re
import statistics
import sys

from client import assert_complete, confident, make_jobs, run_calls, server_info
from common import (FIXED, LABELS, PILOT_IDS, REPORTS, T, assert_config, assert_server_version,
                    clean_texts, load_json, load_memes, require_gate, set_gate, update_runtime)

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
SCOPES = ("total", "en", "es")
SEED = FIXED["seed_reference"]


def _med(xs):
    return statistics.median(xs) if xs else float("nan")


def pilot_metrics(recs: dict, arm: str, ids: list) -> dict:
    calls = [recs[(i, arm, p, SEED)] for i in ids for p in (1, 2, 3)]
    parsed = [c for c in calls if c["label"] in LABELS]
    reasoned = [c for c in calls if c["reasoning_content"]]
    agree = [confident([recs[(i, arm, p, SEED)]["label"] for p in (1, 2, 3)]) for i in ids]
    return {
        "n_memes": len(ids),
        "parse_rate": len(parsed) / len(calls),
        "length_rate": sum(c["finish_reason"] == "length" for c in calls) / len(calls),
        "yes_rate": (sum(c["label"] == "YES" for c in parsed) / len(parsed)) if parsed else float("nan"),
        "paraphrase_agreement": sum(agree) / len(ids),
        "refusals_other": sum(c["parse_method"] == "other" for c in calls),
        "api_errors": sum(c["parse_method"] == "error" for c in calls),
        "cjk_reasoning_frac": (sum(bool(CJK.search(c["reasoning_content"])) for c in reasoned)
                               / len(reasoned)) if reasoned else None,
        "median_completion_tokens": _med([c["usage"]["completion_tokens"] for c in calls if c["usage"]]),
        "median_latency_s": _med([c["latency_s"] for c in calls if c["latency_s"] is not None]),
    }


def eligible(m: dict) -> bool:
    """Regra 1 da §6."""
    per_lang = all(m[l]["parse_rate"] >= T["parse"] and m[l]["paraphrase_agreement"] >= T["confident"]
                   for l in ("en", "es"))
    return per_lang and T["yes_lo"] <= m["total"]["yes_rate"] <= T["yes_hi"]


def choose(M: dict):
    """Regras 2 e 3 da §6. Devolve o braço escolhido ou None."""
    el = [a for a in M if eligible(M[a])]
    if len(el) == 2:
        gain = M["think_on"]["total"]["paraphrase_agreement"] - M["think_off"]["total"]["paraphrase_agreement"]
        return "think_on" if gain > T["think_gain"] else "think_off"
    return el[0] if el else None


def md_table(M: dict) -> list:
    keys = list(next(iter(next(iter(M.values())).values())))
    L = ["| braço | escopo | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for arm, by in M.items():
        for s in SCOPES:
            vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in by[s].values()]
            L.append(f"| {arm} | {s} | " + " | ".join(vals) + " |")
    return L


def setup():
    cfg = assert_config()
    require_gate(cfg, "G0.1")
    require_gate(cfg, "G0.3")
    assert_server_version(cfg, server_info()[1])
    memes = load_memes()
    pilot = load_json(PILOT_IDS)
    by_lang = {"total": pilot, **{l: [i for i in pilot if memes[i]["lang"] == l] for l in ("en", "es")}}
    return cfg, memes, pilot, by_lang


def stage_pilot():
    cfg, memes, pilot, by_lang = setup()
    texts = clean_texts(memes, pilot)
    M = {}
    for arm in cfg["runtime"]["arms_enabled"]:
        jobs = make_jobs(memes, pilot, texts, (1, 2, 3), (SEED,))
        asyncio.run(run_calls(jobs, arm, cfg["runtime"]["concurrency"]))
        recs = assert_complete(arm, jobs)
        M[arm] = {s: pilot_metrics(recs, arm, by_lang[s]) for s in SCOPES}
    chosen = choose(M)

    L = ["# F1 — Piloto (escolha do thinking)", "",
         f"Gerado por `run_pilot.py --stage pilot`. {len(pilot)} memes, P1–P3, seed {SEED}. "
         "Nenhum voto humano foi lido.", "", *md_table(M), "",
         "## Regra de escolha (§6)", ""]
    for arm in M:
        L.append(f"- `{arm}` elegível: **{eligible(M[arm])}**")
    if not chosen:
        L += ["", "**G1 FALHOU: nenhum braço elegível. Parar (§11).**"]
    else:
        L += ["", f"Braço escolhido: **`{chosen}`**. Próximo: `run_pilot.py --stage stability`."]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "pilot_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    set_gate("G1.pilot", chosen is not None, metrics=M, eligible={a: eligible(M[a]) for a in M})
    update_runtime(chosen_arm=chosen)
    print("\n".join(L))
    if not chosen:
        sys.exit(1)


def stage_stability():
    cfg, memes, pilot, by_lang = setup()
    require_gate(cfg, "G1.pilot")
    arm = cfg["runtime"]["chosen_arm"]
    texts = clean_texts(memes, pilot)
    jobs = make_jobs(memes, pilot, texts, (1,), FIXED["seeds_stability"])
    asyncio.run(run_calls(jobs, arm, cfg["runtime"]["concurrency"]))
    recs = assert_complete(arm, jobs + make_jobs(memes, pilot, texts, (1,), (SEED,)))
    seeds = [SEED, *FIXED["seeds_stability"]]

    def flip(i):
        labels = [recs[(i, arm, 1, s)]["label"] for s in seeds]
        return "?" in labels or len(set(labels)) > 1

    flips = {s: sum(map(flip, by_lang[s])) / len(by_lang[s]) for s in SCOPES}
    ok = flips["total"] <= T["flip"]
    L = ["", "## F1b — Estabilidade", "",
         f"Braço `{arm}`, P1, seeds {seeds}. `?` conta como flip.", "",
         "| escopo | flip_rate |", "|---|---|",
         *[f"| {s} | {flips[s]:.4f} |" for s in SCOPES], "",
         f"**G1 {'APROVADO' if ok else 'FALHOU'}** (flip_rate total {flips['total']:.4f}, limite "
         f"{T['flip']}). " + ("Próximo: `run_reference.py`." if ok else "Parar (§11).")]
    path = REPORTS / "pilot_summary.md"
    head = path.read_text(encoding="utf-8").split("\n## F1b")[0].rstrip("\n")
    path.write_text(head + "\n" + "\n".join(L) + "\n", encoding="utf-8")
    set_gate("G1", ok, flip_rate=flips, arm=arm)
    print("\n".join(L))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "stability"], required=True)
    {"pilot": stage_pilot, "stability": stage_stability}[ap.parse_args().stage]()
