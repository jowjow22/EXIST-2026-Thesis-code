"""Doc 33 (SPEC_33.md): a Persona (Claude Sonnet 5.5, system prompts S1–S3, think_off) como referência
para ponderar os anotadores pelo método do doc 27 (r = 1 − D, SPEC_27 §8). Só cálculo, nenhuma chamada.

Os votos vêm de claude_reference_32.py: os 1 551 extremos (doc 32) e os 1 835 memes de treino restantes
(--scope train). As fórmulas são as funções puras de weights.py, sem cópia; as checagens reaproveitam os
auxiliares de checks.py. Nada do GLM é reescrito: as saídas vão para claude_reference_32/persona_ref/,
reliability/outputs/soft_labels_taskA_personaw.json e reports/persona33_*.md; no config.json só entram
as chaves novas runtime.doc33 e gates.doc33_*.

Uso (da raiz do repositório), nesta ordem:
    python3 EXIST_MEMES/glm_annotator/persona_ref.py votes    # junta os votos, portão G2, calibração
    python3 EXIST_MEMES/glm_annotator/persona_ref.py build    # F3: pesos, personaw e o controle personashift
    python3 EXIST_MEMES/glm_annotator/persona_ref.py checks   # F4: checagens (a)–(k) e portão G4
    python3 EXIST_MEMES/glm_annotator/persona_ref.py drift    # ruído entre rodadas (depois do --scope repeat)

Cada fase confere o que a anterior congelou (SHA-256 em runtime.doc33) e grava config.json numa escrita só.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from checks import (B, SCOPES, bootstrap_mean_ci, crosstab, dissident, f, glm_pattern, hard, human_split,
                    quantiles, spearman)
from client import majority
from common import (CONFIG_PATH, LABELS, MEMES_DIR, REF, REL_OUT, REPORTS, SOFT_RAW, T, TRAIN_IDS, VAL_IDS,
                    assert_config, load_config, load_json, load_memes, sha256_file, write_json)
from control_glmshift import shifted
from prompts_32 import SYSTEM_SHA256
from weights import (annotator_demographics, annotator_rows, deviation_from_terms, deviation_terms, glm_labels,
                     glm_soft, load_glm_votes, load_train_panels, n_yes, soft_votes, vote_weights,
                     weighted_soft_label)

MODEL, ARM = "claude-sonnet-5-5", "think_off"
RAW_DIR = MEMES_DIR / "claude_reference_32" / "raw"
RAWS = (RAW_DIR / f"{MODEL}_{ARM}.jsonl", RAW_DIR / f"{MODEL}_{ARM}_train.jsonl")   # extremos, resto do treino
REPEAT = RAW_DIR / f"{MODEL}_{ARM}_repeat.jsonl"                                   # piloto repetido (drift)
OUT = MEMES_DIR / "claude_reference_32" / "persona_ref"
VOTES = OUT / "persona_votes.csv"
WEIGHTS = OUT / "annotator_weights_persona.csv"
SOFT_W = OUT / "soft_labels_weighted_persona.json"
PARCELS = OUT / "annotator_deviation_parcels_persona.csv"
SOFT_PERSONAW = REL_OUT / "soft_labels_taskA_personaw.json"
SOFT_SHIFT = REL_OUT / "soft_labels_taskA_personashift.json"     # controle (SPEC_33 §7)
IMAGE_ERROR = "Could not process image"
N_PERM = 200
EXTREMES = REF / "extremes" / "extremes.csv"
METHOD = "doc33_persona_soft_distance"
FIELDS = ["meme_id", "lang", "label_p1", "label_p2", "label_p3", "vote", "confident"]


def record(gate: str | None = None, passed: bool = False, gate_info: dict | None = None, **doc33_kv) -> None:
    """Uma escrita só no config.json: acrescenta a runtime.doc33 e, se houver, o portão gates.<gate>.
    Nenhuma outra chave (GLM, docs 25–29) é tocada."""
    cfg = load_config()
    rt = cfg["runtime"]
    rt["doc33"] = {**rt.get("doc33", {}), **doc33_kv}
    if gate:
        rt.setdefault("gates", {})[gate] = {"pass": bool(passed), **(gate_info or {})}
    write_json(CONFIG_PATH, cfg)


# ------------------------------------------------------------------ votes

def read_records(paths) -> dict:
    """{(meme_id, prompt): registro}; o último registro de cada chave vale (reenvio PNG).
    Leitura sem read_jsonl, que pode reescrever o arquivo."""
    recs = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(r["meme_id"], r["paraphrase"])] = r
    return recs


def vote_rows(recs: dict, train: list, memes: dict) -> list:
    pend = sorted({k[0] for k, r in recs.items() if IMAGE_ERROR in (r.get("error") or "") and not r.get("image_reencoded")})
    assert not pend, (f"{len(pend)} memes aguardam o reenvio PNG: rode claude_reference_32.py --scope train "
                      "de novo antes do `votes`")
    keys = {(i, p) for i in train for p in (1, 2, 3)}
    assert set(recs) == keys, f"votos ≠ treino × S1–S3: faltam {len(keys - set(recs))}, sobram {len(set(recs) - keys)}"
    bad = [k for k, r in recs.items() if r["prompt_sha256"] != SYSTEM_SHA256[f"S{k[1]}"]]
    assert not bad, f"{len(bad)} registros com system prompt diferente de S1–S3: {bad[:3]}"
    rows = []
    for i in train:
        labels = [recs[(i, p)]["label"] for p in (1, 2, 3)]
        rows.append({"meme_id": i, "lang": memes[i]["lang"], **{f"label_p{p}": l for p, l in zip((1, 2, 3), labels)},
                     "vote": majority(labels) or "",
                     "confident": str("?" not in labels and len(set(labels)) == 1)})
    return rows


def gate_g2(rows: list) -> dict:
    """G2 (definição: SPEC_25 §7; SPEC_33 §3): parse ≥ 0,98 e fração confiante ≥ 0,70, em cada idioma.
    Basta um idioma falhar para o portão falhar."""
    out = {}
    for lang in ("en", "es"):
        sel = [r for r in rows if r["lang"] == lang]
        labels = [l for r in sel for l in glm_labels(r)]
        out[lang] = {"n": len(sel), "parse_rate": sum(l in LABELS for l in labels) / len(labels),
                     "confident": sum(r["confident"] == "True" for r in sel) / len(sel)}
    ok = all(v["parse_rate"] >= T["parse"] and v["confident"] >= T["confident"] for v in out.values())
    return {"by_lang": out, "pass": ok}


def calibration(rows: list, recs: dict, memes: dict, extremes: set, glm: dict | None) -> list:
    """Tabela de calibração por nº de YES humanos: Persona (por prompt e voto) e GLM off (contexto)."""
    L = ["| conjunto | YES humanos | n | humanos | S1 | S2 | S3 | Persona (3 respostas) | voto Persona | "
         "GLM off (3 respostas) | parse | recusas |", "|" + "---|" * 12]
    by = {r["meme_id"]: r for r in rows}
    for name, keep in (("novos (fora dos extremos)", lambda i: i not in extremes),
                       ("extremos (doc 32)", lambda i: i in extremes), ("treino inteiro", lambda i: True)):
        for k in list(range(7)) + ["todos"]:
            ids = [i for i in by if keep(i) and (k == "todos" or n_yes_raw(memes[i]) == k)]
            if not ids:
                continue
            hum = np.mean([n_yes_raw(memes[i]) / len(memes[i]["labels_task2_1"]) for i in ids])
            per = [np.mean([by[i][f"label_p{p}"] == "YES" for i in ids if by[i][f"label_p{p}"] in LABELS])
                   for p in (1, 2, 3)]
            lab = [l for i in ids for l in glm_labels(by[i])]
            g = "—"
            if glm:
                gl = [l for i in ids for l in glm_labels(glm[i]) if l in LABELS]
                g = f"{np.mean([l == 'YES' for l in gl]):.3f}"
            L.append(f"| {name} | {k} | {len(ids)} | {hum:.3f} | " + " | ".join(f"{v:.3f}" for v in per)
                     + f" | {np.mean([l == 'YES' for l in lab if l in LABELS]):.3f} | "
                     f"{np.mean([by[i]['vote'] == 'YES' for i in ids]):.3f} | {g} | "
                     f"{np.mean([l in LABELS for l in lab]):.3f} | "
                     f"{sum(recs[(i, p)]['parse_method'] == 'refusal' for i in ids for p in (1, 2, 3))} |")
    return L


def n_yes_raw(item) -> int:
    return sum(v == "YES" for v in item["labels_task2_1"])


def votes(raws=RAWS, out=VOTES, save=True) -> dict:
    if save:
        cfg = assert_config()
        assert "build" not in cfg["runtime"].get("doc33", {}), "o `build` já rodou: os votos estão congelados"
    memes, train = load_memes(), load_json(TRAIN_IDS)
    recs = read_records(raws)
    rows = vote_rows(recs, train, memes)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    g2 = gate_g2(rows)
    ext = {r["meme_id"] for r in csv.DictReader(open(EXTREMES, encoding="utf-8"))}
    glm = load_glm_votes(REF / "glm_votes.csv")
    errs = Counter(r["parse_method"] for r in recs.values() if r["label"] not in LABELS)
    reenc = sorted({k[0] for k, r in recs.items() if r.get("image_reencoded")})
    tin = sum(r["usage"]["input_tokens"] for r in recs.values() if r.get("usage"))
    tout = sum(r["usage"]["output_tokens"] for r in recs.values() if r.get("usage"))
    L = ["# Doc 33 — votos da Persona no treino (S1–S3, think_off)", "",
         f"Gerado por `persona_ref.py votes`. {len(rows)} memes de treino × 3 prompts = {len(recs)} registros "
         f"(o último de cada chave vale). Sem rótulo: {dict(errs) or 'nenhum'}. Imagens reenviadas em PNG: "
         f"{len(reenc)} memes. Tokens: {tin} de entrada, {tout} de saída (≈ US$ {(tin * 2 + tout * 10) / 2e6:.2f} "
         "no preço batch, extremos incluídos).", "",
         "## Portão G2 (SPEC_33 §3: parse ≥ 0,98 e confiante ≥ 0,70 em cada idioma)", "",
         "| idioma | n | parse | confiante (3-0) |", "|---|---|---|---|"]
    L += [f"| {lang} | {v['n']} | {v['parse_rate']:.4f} | {v['confident']:.4f} |" for lang, v in g2["by_lang"].items()]
    new = gate_g2([r for r in rows if r["meme_id"] not in ext])
    L += ["", f"**G2: {'aprovado' if g2['pass'] else 'FALHOU — parar (SPEC_33 §3)'}**", "",
          "Descritivo: os 1 551 extremos já eram conhecidos do doc 32; só nos 1 835 memes novos: " +
          " · ".join(f"{lang} parse {v['parse_rate']:.4f}, confiante {v['confident']:.4f}"
                     for lang, v in new["by_lang"].items()) + ".", "",
          "## Calibração por nº de YES humanos (descritivo; a linha dos novos é o teste fora dos extremos, "
          "SPEC_33 §2)", "", "Fração de YES sobre as respostas parseadas; humanos = fração de YES dos 6 votos.", "",
          *calibration(rows, recs, memes, ext, glm), ""]
    if save:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "persona33_votes.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        record("doc33_G2", g2["pass"], {**g2, "new_only": new["by_lang"]}, votes_sha256=sha256_file(out),
               raw_sha256={p.name: sha256_file(p) for p in map(Path, raws)}, image_reencoded=reenc,
               tokens={"input": tin, "output": tout})
    print("\n".join(L))
    return g2


# ------------------------------------------------------------------ build (F3)

def build_core(votes_csv, memes, train, val, raw) -> tuple[list, dict, dict]:
    """(linhas por anotador, soft labels de treino {old, new}, soft labels personaw completos)."""
    ref_rows = load_glm_votes(votes_csv)
    assert set(ref_rows) == set(train), "o arquivo de votos não cobre exatamente os memes de treino"
    assert not set(train) & set(val), "treino ∩ validação ≠ ∅ (portão 12.3)"
    panels, _ = load_train_panels(memes, train)
    p_ref = soft_votes(ref_rows)
    demo, _ = annotator_demographics(memes, train)
    rows = annotator_rows(panels, p_ref, demo)
    r = {x["annotator_id"]: x["r_glm"] for x in rows}
    soft = {}
    for i in train:
        p_new = weighted_soft_label(panels[i], r)
        assert abs(raw[i][0] - n_yes(panels[i]) / len(panels[i])) < 1e-12, f"{i}: raw ≠ fração de votos"
        soft[i] = {"old": raw[i], "new": [p_new, 1 - p_new]}
    for i, s in soft.items():                                     # portão 12.5
        p, q = s["new"]
        assert 0.0 <= p <= 1.0 and abs(p + q - 1) < 1e-12, f"{i}: soft label inválido {s['new']}"
        if n_yes(panels[i]) in (0, len(panels[i])):
            assert p == s["old"][0], f"{i}: meme unânime mudou ({s['old'][0]} → {p})"
    full = {i: (soft[i]["new"] if i in soft else raw[i]) for i in raw}
    assert len(full) == 3984 and json.dumps({i: full[i] for i in val}) == json.dumps({i: raw[i] for i in val}), \
        "labels de validação diferem dos originais (portão 12.5)"
    return rows, soft, full


def rename(row: dict) -> dict:
    """Colunas do weights.py (D_glm, r_glm) com o nome da referência desta onda."""
    return {k.replace("_glm", "_persona"): v for k, v in row.items()}


def build():
    cfg = assert_config()
    d33 = cfg["runtime"].get("doc33", {})
    if "build" in d33:                      # congelado: só confere (o congelamento vale pelo config.json)
        assert sha256_file(SOFT_PERSONAW) == d33["sha256"], f"{SOFT_PERSONAW.name} ≠ o congelado: não sobrescrever"
        print(f"F3 já congelado em {d33['build']['date']} (SHA-256 confere); nada a fazer")
        return
    gate = cfg["runtime"].get("gates", {}).get("doc33_G2", {})
    assert gate.get("pass"), "portão G2 do doc 33 não aprovado: rode `votes` (SPEC_33 §3)"
    assert sha256_file(VOTES) == d33["votes_sha256"], "persona_votes.csv mudou depois do `votes`"
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    from test_weights import run_all
    assert run_all() == 0, "testes do Apêndice A falharam (portão 12.4)"
    memes, train, val, raw = load_memes(), load_json(TRAIN_IDS), load_json(VAL_IDS), load_json(SOFT_RAW)
    rows, soft, full = build_core(VOTES, memes, train, val, raw)
    with open(WEIGHTS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rename(rows[0])))
        w.writeheader()
        w.writerows(rename(x) for x in rows)
    write_json(SOFT_W, soft)
    write_json(SOFT_PERSONAW, full)
    ctrl, delta = shifted(raw, full, train)      # controle congelado antes de qualquer run (SPEC_27 §16C)
    assert all(ctrl[i] == raw[i] for i in val) and len(ctrl) == len(raw)
    write_json(SOFT_SHIFT, ctrl)
    d = [soft[i]["new"][0] - soft[i]["old"][0] for i in train]
    info = {"method": METHOD, "date": str(__import__("datetime").date.today()),
            "votes_sha256": sha256_file(VOTES), "weights_sha256": sha256_file(WEIGHTS),
            "soft_w_sha256": sha256_file(SOFT_W),
            "mean_shift": sum(d) / len(d), "mean_abs_shift": sum(map(abs, d)) / len(d),
            "share_moved": sum(abs(x) >= T["half_vote"] - T["tol"] for x in d) / len(d),
            "no_included_memes": sum(x["no_included_memes"] for x in rows), "n_annotators": len(rows)}
    control = {"sha256": sha256_file(SOFT_SHIFT), "mean_shift": delta, "runs": abs(delta) >= 1 / 24}
    record("doc33_F3", True, {"method": METHOD}, build=info, sha256=sha256_file(SOFT_PERSONAW), control=control)
    print(f"F3 ok ({METHOD}) · {len(rows)} anotadores ({info['no_included_memes']} sem meme incluído, r = 1) · "
          f"deslocamento médio {info['mean_shift']:+.4f} · share_moved {info['share_moved']:.3f}\n"
          f"controle personashift: Δ̄ = {delta:+.4f} ({'roda se o critério for satisfeito' if control['runs'] else '|Δ̄| < 1/24: não roda (SPEC_33 §7)'})\n"
          f"Gravados: {WEIGHTS.name}, {SOFT_W.name}, {SOFT_PERSONAW.name}, {SOFT_SHIFT.name}. Próximo: persona_ref.py checks")


# ------------------------------------------------------------------ checks (F4)

def null_band(panels, ref_rows, train, lang, yes_rate, r_peer, raw, n_perm, rng) -> dict:
    """(k) Faixa nula: as triplas de rótulos da referência embaralhadas entre memes, dentro de cada idioma.
    Mantém a taxa de YES da referência e destrói a informação por meme. Devolve {estatística: (p2,5, p97,5)}."""
    anns = sorted(r_peer)
    ids = {L: [i for i in train if lang[i] == L] for L in ("en", "es")}
    trip = {L: [glm_labels(ref_rows[i]) for i in ids[L]] for L in ids}
    out = defaultdict(list)
    for _ in range(n_perm):
        p = {}
        for L in ids:
            for i, j in zip(ids[L], rng.permutation(len(ids[L]))):
                p[i] = glm_soft(trip[L][j])
        terms = deviation_terms(panels, p)
        D = deviation_from_terms(terms, anns)
        r = vote_weights(D)
        by = defaultdict(list)
        for a, m, _, _ in terms:
            by[a].append(m)
        A = [a for a in anns if a in by]
        Bd = [a for a in A if D[a] is not None]
        gap = {a: abs(np.mean([panels[m][a] == "YES" for m in ms]) - np.mean([p[m] for m in ms])) for a, ms in by.items()}
        out["rho_a"].append(spearman([r[a] for a in A], [r_peer[a] for a in A]))
        out["rho_b"].append(spearman([D[a] for a in Bd], [yes_rate[a] for a in Bd]))
        out["rho_gap"].append(spearman([D[a] for a in Bd], [gap[a] for a in Bd]))
        out["share_moved"].append(np.mean([abs(weighted_soft_label(panels[i], r) - raw[i][0]) >= T["half_vote"] - T["tol"]
                                           for i in train]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def reading_b(obs: dict, band: dict | None) -> str:
    """Leitura pré-registrada da (b) com a faixa nula (SPEC_33 §5)."""
    if band is None:
        return "sem faixa nula (n_perm = 0)"
    if obs["rho_a"] <= band["rho_a"][1]:
        return "ρ(a) dentro da faixa nula: o peso não carrega informação por meme além da taxa de YES da referência"
    if abs(obs["rho_b"]) < 0.40 and obs["rho_gap"] < 0.40:
        return "o peso deixou de medir sobretudo o limiar de YES e carrega informação por meme"
    if abs(obs["rho_b"]) > 0.60:
        return "o peso continua medindo sobretudo o limiar de YES, como no GLM"
    return "parcial"


def checks_core(votes_csv, weights_csv, soft, memes, train, glm_weights_csv=None, raw=None,
                n_perm=0) -> tuple[list, dict]:
    """Checagens (a)–(i) do SPEC_27 §9 e §16A com a Persona como referência, (j) Persona × GLM e (k) faixa
    nula por permutação (n_perm > 0). Devolve (linhas do relatório, resultados)."""
    panels, _ = load_train_panels(memes, train)
    ref_rows = load_glm_votes(votes_csv)
    p_ref = soft_votes(ref_rows)
    aw = pd.read_csv(weights_csv, dtype={"annotator_id": str})
    r_col, d_col = ("r_persona", "D_persona") if "r_persona" in aw else ("r_glm", "D_glm")
    lang = {i: memes[i]["lang"] for i in train}
    inc = aw[aw.n_included >= 1]
    terms = deviation_terms(panels, p_ref)
    by_ann = defaultdict(list)
    for a, mid, w, d in terms:
        by_ann[a].append((mid, w, d))
    L = ["# Doc 33 — F4: checagens (a)–(j) com a Persona como referência", "",
         f"Gerado por `persona_ref.py checks`. Treino: {len(train)} memes, {len(aw)} anotadores, {len(inc)} com "
         f"≥ 1 meme incluído. Método: `{METHOD}` (fórmulas do doc 27, SPEC_27 §8).", "",
         "> Escala (SPEC_27 §16A, P1): num meme em que a Persona fica 2-1, todo anotador leva pelo menos 1/3 "
         "de desvio.", ""]

    rho_a = spearman(inc[r_col], inc.r_peer)
    L += ["## (a) Persona vs consenso dos pares", "", f"ρ de Spearman(r_persona, r_peer) = **{f(rho_a)}** "
          f"(n = {len(inc)}).", ""]

    gap = {a: abs(np.mean([panels[m][a] == "YES" for m, _, _ in t]) - np.mean([p_ref[m] for m, _, _ in t]))
           for a, t in by_ann.items()}
    b_ok = inc[inc[d_col].notna()]
    rho_b1, rho_b2 = spearman(b_ok[d_col], b_ok.yes_rate), spearman(b_ok[d_col], b_ok.annotator_id.map(gap))
    L += ["## (b) O peso só mede o limiar?", "",
          f"- ρ(D_persona, taxa de YES do anotador em todos os seus memes de treino) = **{f(rho_b1)}**",
          f"- ρ(D_persona, |taxa de YES do anotador − média de p_Persona|, nos memes incluídos) = **{f(rho_b2)}**", ""]

    rng = np.random.default_rng(B["rng"])
    L += ["## (c) r por gênero × idade (IC 95% bootstrap de anotadores)", "",
          "| gênero | idade | n | r_persona | IC 95% | r_peer | IC 95% |", "|---|---|---|---|---|---|---|"]
    for (g, age), grp in inc.groupby(["gender", "age"], sort=True):
        cg, cp = bootstrap_mean_ci(grp[r_col], rng), bootstrap_mean_ci(grp.r_peer, rng)
        L.append(f"| {g} | {age} | {len(grp)} | {f(grp[r_col].mean())} | [{f(cg[0])}, {f(cg[1])}] | "
                 f"{f(grp.r_peer.mean())} | [{f(cp[0])}, {f(cp[1])}] |")
    _, multi = annotator_demographics(memes, train)
    L += ["", f"Anotadores em mais de um painel (demografia conferida como consistente): {multi}.", ""]

    moved = {i: abs(s["new"][0] - s["old"][0]) >= T["half_vote"] - T["tol"] for i, s in soft.items()}
    changes = {i: (hard(s["old"][0]), hard(s["new"][0])) for i, s in soft.items()
               if hard(s["old"][0]) != hard(s["new"][0])}
    share_moved = sum(moved.values()) / len(moved)
    L += ["## (d) Portão — quanto os soft labels se movem", "",
          "| escopo | share_moved (|Δp| ≥ 1/12) | mudanças de rótulo duro | Δp médio |", "|---|---|---|---|"]
    for scope in SCOPES:
        ids = [i for i in soft if scope == "total" or lang[i] == scope]
        L.append(f"| {scope} | {f(sum(moved[i] for i in ids) / len(ids))} | {sum(i in changes for i in ids)} | "
                 f"{np.mean([soft[i]['new'][0] - soft[i]['old'][0] for i in ids]):+.4f} |")
    kinds = Counter(f"{a}→{b}" for a, b in changes.values())
    L += ["", "Tipos de mudança: " + (", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) or "nenhuma"), ""]

    conf = {i: ref_rows[i]["confident"] == "True" for i in train}
    unan = [i for i in train if n_yes(panels[i]) in (0, len(panels[i])) and conf[i]]
    contra = [i for i in unan if ref_rows[i]["vote"] != ("YES" if n_yes(panels[i]) else "NO")]
    L += ["## (e) Persona confiante (3-0) contrariando meme unânime", "",
          "| escopo | unânimes com Persona 3-0 | contrariados | taxa |", "|---|---|---|---|"]
    for scope in SCOPES:
        u = [i for i in unan if scope == "total" or lang[i] == scope]
        c = [i for i in contra if scope == "total" or lang[i] == scope]
        L.append(f"| {scope} | {len(u)} | {len(c)} | {f(len(c) / len(u) if u else float('nan'))} |")
    sample = sorted(np.random.default_rng(B["rng"]).choice(sorted(contra), size=min(30, len(contra)),
                                                           replace=False).tolist()) if contra else []
    L += ["", f"{len(sample)} memes sorteados para revisão manual (`persona_ref/manual_review_unanimous.csv`).", ""]

    L += ["## (f) Acordo do voto da Persona com o rótulo duro humano (empates 3-3 fora)", "",
          "| escopo | divisão | n | acordo |", "|---|---|---|---|"]
    for scope in SCOPES:
        for split in ("todas", "6-0", "5-1", "4-2"):
            ok = []
            for i in train:
                k, n = n_yes(panels[i]), len(panels[i])
                v = ref_rows[i]["vote"]
                if 2 * k == n or not v or (scope != "total" and lang[i] != scope):
                    continue
                if split != "todas" and human_split(panels[i]) != split:
                    continue
                ok.append(v == ("YES" if 2 * k > n else "NO"))
            L.append(f"| {scope} | {split} | {len(ok)} | {f(np.mean(ok) if ok else float('nan'))} |")
    L.append("")

    rng = np.random.default_rng(B["rng"])
    widths = []
    for a in sorted(by_ann):
        wd = np.array([(w, d) for _, w, d in by_ann[a]], dtype=float)
        idx = rng.integers(0, len(wd), (B["n"], len(wd)))
        W, Dd = wd[idx, 0], wd[idx, 1]
        sw = W.sum(axis=1)
        Ds = (W * Dd).sum(axis=1)[sw > 0] / sw[sw > 0]
        if len(Ds):
            widths.append(float(np.percentile(Ds, 97.5) - np.percentile(Ds, 2.5)))
    anns = list(aw.annotator_id)
    is30 = lambda m: p_ref[m] in (0.0, 1.0)                     # noqa: E731
    D30 = deviation_from_terms([t for t in terms if is30(t[1])], anns)
    D21 = deviation_from_terms([t for t in terms if not is30(t[1])], anns)
    m30, m21 = Counter(), Counter()
    for a, mid, w, _ in terms:
        (m30 if is30(mid) else m21)[a] += w
    parcels = [{"annotator_id": a, "D_persona": "" if pd.isna(d) else d, "D_persona_30": "" if D30[a] is None else D30[a],
                "D_persona_21": "" if D21[a] is None else D21[a], "mass_30": m30[a], "mass_21": m21[a]}
               for a, d in zip(aw.annotator_id, aw[d_col])]
    q = aw.n_included.quantile([0, .25, .5, .75, 1]).tolist()
    L += ["## (g) Precisão por anotador", "",
          f"- n_included: mín {q[0]:.0f} · p25 {q[1]:.0f} · mediana {q[2]:.0f} · p75 {q[3]:.0f} · máx {q[4]:.0f}",
          f"- mass_included (Σ w por anotador): {quantiles(aw.mass_included.tolist())}",
          f"- fração com < 5 memes incluídos: **{f((aw.n_included < 5).mean())}**",
          f"- largura mediana do IC 95% de D_persona ({len(widths)} anotadores): "
          f"**{f(float(np.median(widths)) if widths else float('nan'))}**",
          f"- D_persona total: {quantiles([None if pd.isna(x) else x for x in aw[d_col]])}",
          f"- só memes Persona 3-0: {quantiles(D30.values())}",
          f"- só memes Persona 2-1: {quantiles(D21.values())} (piso de 1/3 por construção)", ""]

    L += ["## (h) Fração da massa Σ w que vem de memes Persona 2-1", "", "| escopo | massa 2-1 / total |", "|---|---|"]
    for scope in SCOPES:
        sel = [(mid, w) for _, mid, w, _ in terms if scope == "total" or lang[mid] == scope]
        tot = sum(w for _, w in sel)
        L.append(f"| {scope} | {f(sum(w for mid, w in sel if not is30(mid)) / tot if tot else float('nan'))} |")
    L.append("")

    pat = {i: glm_pattern(glm_labels(ref_rows[i])) for i in train}
    ids = [i for i in train if pat[i] is not None]
    ids21 = [i for i in ids if pat[i] == "2-1"]
    dis = {i: dissident(glm_labels(ref_rows[i])).replace("P", "S") for i in ids21}
    L += ["## (i) Hesitação da Persona × divisão humana (descritiva)", "",
          *crosstab(pat, ids, lang, ("3-0", "2-1"), panels), "",
          "**Memes Persona 2-1, pelo prompt que discordou:**", "",
          *crosstab(dis, ids21, lang, ("S1", "S2", "S3"), panels), ""]
    for scope in SCOPES:
        amb, n = {}, {}
        for row in ("3-0", "2-1"):
            sel = [i for i in ids if pat[i] == row and (scope == "total" or lang[i] == scope)]
            n[row] = len(sel)
            amb[row] = sum(human_split(panels[i]) in ("4-2", "3-3") for i in sel) / len(sel) if sel else float("nan")
        diff = amb["2-1"] - amb["3-0"]
        se = np.sqrt(sum(amb[r] * (1 - amb[r]) / n[r] for r in amb)) if all(n.values()) else float("nan")
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        verdict = ("têm **mais** divisões humanas fortes" if lo > 0 else "têm **menos** divisões humanas fortes"
                   if hi < 0 else "**não diferem** dos 3-0 em divisões humanas fortes (IC inclui zero)")
        L.append(f"- {scope}: 4-2 + 3-3 = {f(amb['2-1'], 3)} nos 2-1 vs {f(amb['3-0'], 3)} nos 3-0; diferença "
                 f"{diff:+.3f} (IC 95% [{lo:+.3f}, {hi:+.3f}]) → os memes em que a Persona hesita {verdict}.")
    L += ["", "Leitura pré-registrada (SPEC_27 §16B): \"mais\" só quando o IC 95% da diferença exclui zero. "
              "Aqui S1–S3 não são paráfrases: um 2-1 é discordância entre procedimentos (SPEC_33 §9).", ""]

    if glm_weights_csv is not None:                               # (j) comparação com o GLM (doc 27)
        g = pd.read_csv(glm_weights_csv, dtype={"annotator_id": str}).set_index("annotator_id")
        both = inc[inc.annotator_id.map(lambda a: a in g.index and g.loc[a, "n_included"] >= 1)]
        gi = g[g.n_included >= 1]
        gb = gi[gi.D_glm.notna()]
        L += ["## (j) Persona × GLM (doc 27), mesmos anotadores", "",
              f"- ρ(r_persona, r_glm) = **{f(spearman(both[r_col], both.annotator_id.map(g.r_glm)))}** "
              f"(n = {len(both)})",
              f"- (b) lado a lado — ρ(D, taxa de YES do anotador): Persona **{f(rho_b1)}** · GLM "
              f"**{f(spearman(gb.D_glm, gb.yes_rate))}**",
              f"- r médio: Persona {f(inc[r_col].mean())} · GLM {f(gi.r_glm.mean())}", ""]

    obs = {"rho_a": rho_a, "rho_b": rho_b1, "rho_gap": rho_b2, "share_moved": share_moved}
    band = None
    if n_perm:
        yes_rate = dict(zip(aw.annotator_id, aw.yes_rate))
        r_peer = dict(zip(inc.annotator_id, inc.r_peer))
        band = null_band(panels, ref_rows, train, lang, yes_rate, r_peer, raw, n_perm, np.random.default_rng(B["rng"]))
        L += [f"## (k) Faixa nula por permutação ({n_perm} permutações das triplas S1–S3 entre memes, dentro de cada "
              "idioma; default_rng(2026))", "",
              "Mantém a taxa de YES da Persona e destrói a informação por meme. Uma referência calibrada derruba a (b) "
              "sozinha; o que separa informação de taxa é a (a) acima da faixa (SPEC_33 §5).", "",
              "| estatística | observado | faixa nula p2,5–p97,5 |", "|---|---|---|"]
        L += [f"| {k} | {f(obs[k])} | [{f(band[k][0])}, {f(band[k][1])}] |" for k in ("rho_a", "rho_b", "rho_gap", "share_moved")]
    verdict = reading_b(obs, band)
    L += ["", f"**Leitura pré-registrada da (b) (SPEC_33 §5): {verdict}.**", ""]
    ok = share_moved >= T["share_moved"] and rho_a < T["rho"]
    return L, {"share_moved": share_moved, "rho_a": rho_a, "rho_b_yes_rate": rho_b1, "rho_b_gap": rho_b2,
               "pass": ok, "sample": sample, "parcels": parcels, "band": band, "reading_b": verdict}


def checks():
    cfg = assert_config()
    assert cfg["runtime"].get("gates", {}).get("doc33_F3", {}).get("pass"), "rode `build` antes (F3)"
    d33 = cfg["runtime"]["doc33"]
    for path, key in ((VOTES, "votes_sha256"), (WEIGHTS, "weights_sha256"), (SOFT_W, "soft_w_sha256")):
        assert sha256_file(path) == d33["build"][key], f"{path.name} mudou depois do `build`"
    assert sha256_file(SOFT_PERSONAW) == d33["sha256"], "personaw ≠ o congelado no `build`"
    memes, train, raw = load_memes(), load_json(TRAIN_IDS), load_json(SOFT_RAW)
    soft = load_json(SOFT_W)
    L, res = checks_core(VOTES, WEIGHTS, soft, memes, train, glm_weights_csv=REF / "annotator_weights.csv",
                         raw=raw, n_perm=N_PERM)
    # a mesma faixa nula para o GLM do doc 27, como contexto (descritivo)
    panels, _ = load_train_panels(memes, train)
    gw = pd.read_csv(REF / "annotator_weights.csv", dtype={"annotator_id": str})
    gi = gw[gw.n_included >= 1]
    gband = null_band(panels, load_glm_votes(REF / "glm_votes.csv"), train, {i: memes[i]["lang"] for i in train},
                      dict(zip(gw.annotator_id, gw.yes_rate)), dict(zip(gi.annotator_id, gi.r_peer)), raw, N_PERM,
                      np.random.default_rng(B["rng"]))
    L += ["**Contexto — a mesma faixa nula para o GLM (doc 27):** " +
          " · ".join(f"{k} [{f(v[0])}, {f(v[1])}]" for k, v in gband.items()) +
          " (observados no doc 27: ρ(a) 0,5008, ρ(b) −0,676, ρ(gap) 0,710, share_moved 0,3461).", ""]
    with open(PARCELS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["parcels"][0]))
        w.writeheader()
        w.writerows(res["parcels"])
    ref_rows = load_glm_votes(VOTES)
    with open(OUT / "manual_review_unanimous.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["meme_id", "lang", "path_memes", "text", "votes", "S1", "S2", "S3"])
        for i in res["sample"]:
            w.writerow([i, memes[i]["lang"], memes[i]["path_memes"], memes[i]["text"],
                        ";".join(memes[i]["labels_task2_1"]), *glm_labels(ref_rows[i])])
    G = ["# Portão G4 — retreino (doc 33)", "",
         f"- share_moved = **{f(res['share_moved'])}** (exige ≥ {T['share_moved']}) → "
         f"{'ok' if res['share_moved'] >= T['share_moved'] else 'FALHA'}",
         f"- ρ(r_persona, r_peer) = **{f(res['rho_a'])}** (exige < {T['rho']}) → "
         f"{'ok' if res['rho_a'] < T['rho'] else 'FALHA'}", "",
         f"**Decisão: {'RETREINO AUTORIZADO' if res['pass'] else 'RETREINO NÃO AUTORIZADO — encerrar (SPEC_33 §5)'}**"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "persona33_checks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (REPORTS / "persona33_gate.md").write_text("\n".join(G) + "\n", encoding="utf-8")
    record("doc33_G4", res["pass"], {"share_moved": res["share_moved"], "rho_r_persona_r_peer": res["rho_a"],
                                     "rho_b_yes_rate": res["rho_b_yes_rate"], "rho_b_gap": res["rho_b_gap"],
                                     "null_band": res["band"], "reading_b": res["reading_b"], "method": METHOD})
    print("\n".join(L + [""] + G))
    if not res["pass"]:
        sys.exit(1)


# ------------------------------------------------------------------ drift (ruído entre rodadas)

def drift():
    """Compara o piloto repetido (--scope repeat) com o piloto original do doc 32, chave a chave.
    Descritivo (SPEC_33 §2): mede quanto um rótulo muda entre duas rodadas idênticas."""
    a, b = read_records(RAWS[:1]), read_records([REPEAT])
    assert b and set(b) <= set(a), "o JSONL do repeat tem chaves fora dos extremos"
    ext = {r["meme_id"]: r for r in csv.DictReader(open(EXTREMES, encoding="utf-8"))}
    ids = sorted({k[0] for k in b})
    L = ["# Doc 33 — ruído entre rodadas: piloto do doc 32 repetido", "",
         f"{len(ids)} memes × S1–S3 = {len(b)} pedidos; mesma configuração, rodadas em datas diferentes.", "",
         "| grupo | prompt | n | rótulos que mudaram | YES rodada 1 | YES rodada 2 |", "|---|---|---|---|---|---|"]
    for g in ("1Y5N", "3Y3N", "5Y1N", "todos"):
        for p in (1, 2, 3, "todos"):
            ks = [(i, q) for i in ids for q in ((1, 2, 3) if p == "todos" else (p,)) if g == "todos" or ext[i]["group"] == g]
            if not ks:
                continue
            flips = sum(a[k]["label"] != b[k]["label"] for k in ks)
            L.append(f"| {g} | {'S' + str(p) if p != 'todos' else 'todos'} | {len(ks)} | {flips} ({flips / len(ks):.3f}) | "
                     f"{np.mean([a[k]['label'] == 'YES' for k in ks]):.3f} | {np.mean([b[k]['label'] == 'YES' for k in ks]):.3f} |")
    vflip = sum(majority([a[(i, p)]["label"] for p in (1, 2, 3)]) != majority([b[(i, p)]["label"] for p in (1, 2, 3)])
                for i in ids)
    L += ["", f"Votos (maioria de 3) que mudaram: {vflip} de {len(ids)}.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "persona33_drift.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    record(drift={"n_keys": len(b), "label_flips": sum(a[k]["label"] != b[k]["label"] for k in b),
                  "vote_flips": vflip, "raw_sha256": sha256_file(REPEAT)})
    print("\n".join(L))


if __name__ == "__main__":
    cmds = {"votes": votes, "build": build, "checks": checks, "drift": drift}
    assert len(sys.argv) == 2 and sys.argv[1] in cmds, f"uso: persona_ref.py {{{'|'.join(cmds)}}}"
    out = cmds[sys.argv[1]]()
    if sys.argv[1] == "votes" and not out["pass"]:
        sys.exit(1)                                  # G2 falhou: a onda para (SPEC_33 §3)
