"""Análise da FASE 2 da camada de confusão: régua de votos, controlo positivo
e braço com viés congelado.

Lê os logs de `EXIST_MEMES/metrics_history/` escritos pelo `MetricsHistoryLogger`,
`ConfusionMatrixLogger` e `VoteMetricLogger` do `EXIST_MEMES/memes_confusion.ipynb`.
Não treina, não regenera nada.

Porque é um módulo separado de `analyze_confusion_experiment.py`: aquele é o
relatório CONFIRMATÓRIO da Fase 1 (grelha método x lambda x seed, t pareado,
Bonferroni) e não deve mudar de forma. Este é a Fase 2, com uma régua diferente
e um critério diferente. Declarado como desvio em `CONFUSION_EXPERIMENT.md` §13.

Desenho (ver `CONFUSION_EXPERIMENT.md` §§10-18)
-----------------------------------------------
- Régua: NLL dos 3588 votos individuais dos 598 itens de VALIDAÇÃO, na época
  `E = argmin(eval_loss)` -- **nunca** escolhida pela métrica reportada.
- Critério: `NLL(b) < NLL(d4_train)`. Ver os vereditos pré-declarados na §12.
- ⚠ `b` contra `a` é DESCRITIVO: `b` já bate `a` no init por encolhimento
  (calibração global, não sinal de anotador -- §14). O contraste limpo do sinal
  individual é `b` contra `b_perm`.

Uso:
    python3 -m reliability.analyze_votes_experiment
"""

import glob
import json
import os
import re
import sys

import numpy as np

METRICS_DIR = "EXIST_MEMES/metrics_history"
OUT_DIR = "reliability/outputs"
DATA_JSON = "EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json"
SPLIT_PATH = "reliability/outputs/split_taskA_frozen.json"
REPORT_PATH = os.path.join(OUT_DIR, "experiment_votes_report.md")
JSON_PATH = os.path.join(OUT_DIR, "experiment_votes_report.json")

N_EPOCHS = 15
PRIMARY = "eval_task_a_icmsoftnorm_soft"
PREDICTORS = ["a", "b", "b_perm", "c", "d3_train", "d4_train", "d3_val", "d4_val"]
TAG_RE = re.compile(
    r"_conf_(M[123]sd(?:frz)?(?:crp)?)_lam([0-9.]+)"
    r"(?:_clr([0-9.e+-]+))?_seed(\d+)\.json$")


# ---------------------------------------------------------------------------
# descoberta
# ---------------------------------------------------------------------------
def discover():
    runs = {}
    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "*.json"))):
        base = os.path.basename(path)
        if base.endswith(("_confusion.json", "_votes.json")):
            continue
        m = TAG_RE.search(base)
        if not m:
            continue
        hist = json.load(open(path, encoding="utf-8"))
        evals = [r for r in hist if "eval_loss" in r]
        if len(evals) != N_EPOCHS:
            continue                      # testes curtos não são runs do estudo
        arm, lam, clr, seed = m.group(1), float(m.group(2)), m.group(3), int(m.group(4))
        runs[(arm, lam, seed)] = {
            "path": path, "arm": arm, "lam": lam, "seed": seed,
            "conf_lr": None if clr is None else float(clr),
            "evals": evals,
            "tag": base[:-len(".json")],
        }
    return runs


def best_epoch(evals):
    """E = argmin(eval_loss). NUNCA pela métrica reportada -- isso é leakage."""
    losses = [r["eval_loss"] for r in evals]
    i = int(np.argmin(losses))
    return i, evals[i]["epoch"], int(evals[i].get("step", 0)), losses[i]


# ---------------------------------------------------------------------------
# estatísticas de anotador, recalculadas SÓ do treino
# ---------------------------------------------------------------------------
def annotator_reference(manifest=None):
    """`(stats, strata, ids)` calculados SÓ do TREINO.

    ⚠ Não usar `behavioral_profiles.csv`: a `yes_rate` de lá é sobre os 3984
    itens, logo inclui os votos de validação -- os que a régua tenta prever.

    ⚠ `manifest` (run corrompida): as estatísticas TÊM de vir da tabela
    CORROMPIDA, que é o mundo em que as fichas foram treinadas. Com a tabela
    limpa, os 120 spammers apareceriam com a `yes_rate` que nunca tiveram e a
    leitura "a competência absorveu |yes − base|?" ficaria sem sentido.
    """
    sys.path.insert(0, "EXIST_MEMES")
    from confusion_core import (build_vote_table, build_annotator_stats,
                                build_annotator_strata, corrupt_vote_table)
    table, ids, _ = build_vote_table(DATA_JSON, verbose=False)
    strata = build_annotator_strata(DATA_JSON, ids, verbose=False)
    if manifest is not None:
        table, _m = corrupt_vote_table(
            table, ids, strata, levels=tuple(manifest["levels"]),
            n_per_level=manifest["n_per_level"], seed=manifest["seed"],
            verbose=False)
        assert _m["n_votes_flipped"] == manifest["n_votes_flipped"], (
            "a corrupção reconstruída não bate com o manifesto -- a análise "
            "estaria a ler um mundo diferente do que foi treinado")
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    stats = build_annotator_stats(table, split["train_ids"], len(ids), verbose=False)
    return stats, strata, ids


def find_manifest(run):
    """Manifesto de corrupção da run, ou `(None, None)`."""
    # O tag do ficheiro é `AAAAMMDD_HHMMSS_<RUN_TAG>`; o manifesto é gravado
    # pelo notebook como `corruption_manifest_<RUN_TAG>.json`. Cortar por
    # `conf_` e não por contagem de underscores (a data tem dois).
    t = run["tag"]
    stem = t[t.index("conf_"):] if "conf_" in t else t
    exact = os.path.join(OUT_DIR, f"corruption_manifest_{stem}.json")
    cands = [exact] + sorted(glob.glob(
        os.path.join(OUT_DIR, "corruption_manifest_*crp*.json")))
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        return None, None
    return json.load(open(path, encoding="utf-8")), path


# ---------------------------------------------------------------------------
# diagnóstico das matrizes na parametrização `sd`
# ---------------------------------------------------------------------------
def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sd_diagnostics(run, epoch_index, stats):
    """Competência e viés na época E, e com que é que a competência correlaciona.

    Recuperados das PRÓPRIAS matrizes (`u = logit C[0,0]`, `v = logit C[1,1]`),
    não dos parâmetros -- funciona mesmo em ficheiros antigos sem `s_common`.

    ⚠ O viés reporta-se em ESPAÇO DE LOGIT (`u - v`). Em probabilidade,
    `sigma(u) - sigma(v)` varia com a competência mesmo com o viés individual
    congelado a zero, por não-linearidade da sigmóide -- reportar ali faria o
    braço congelado parecer que continua a aprender viés (§13.4).
    """
    cpath = run["path"].replace(".json", "_confusion.json")
    if not os.path.exists(cpath):
        return None
    data = json.load(open(cpath, encoding="utf-8"))
    snaps = data["snapshots"][:N_EPOCHS]
    if epoch_index >= len(snaps):
        return None
    snap = snaps[epoch_index]
    C = np.asarray(snap["matrices"], dtype=float)                # [M, 2, 2]
    eps = 1e-12
    u = np.log(np.clip(C[:, 0, 0], eps, 1 - eps) / np.clip(C[:, 0, 1], eps, 1 - eps))
    v = np.log(np.clip(C[:, 1, 1], eps, 1 - eps) / np.clip(C[:, 1, 0], eps, 1 - eps))
    comp = 2.0 / (1.0 + np.exp(-(u + v) / 2.0)) - 1.0
    bias_logit = u - v

    yes = stats["yes_rate"][:len(comp)]
    agr = stats["agree"][:len(comp)]
    base = stats["panel_yes_rate"]
    out = {
        "epoch": snap["epoch"],
        "param": data.get("param"), "freeze_bias": data.get("freeze_bias"),
        "s_common": snap.get("s_common", data.get("s_common")),
        "d_common": snap.get("d_common", data.get("d_common")),
        "competence_mean": float(comp.mean()), "competence_sd": float(comp.std(ddof=1)),
        "competence_min": float(comp.min()), "competence_max": float(comp.max()),
        "bias_logit_mean": float(bias_logit.mean()),
        "bias_logit_sd": float(bias_logit.std(ddof=1)),
        "bias_prob_sd": float((C[:, 1, 0] - C[:, 0, 1]).std(ddof=1)),
        # A previsão registada: bloqueado o viés, a competência absorve o
        # DESVIO à taxa base -- forma em U. A correlação LINEAR com a yes_rate
        # esconde-a (era -0,15 na Fase 1 contra -0,74 no desvio).
        "r_comp_yesrate_linear": _pearson(comp, yes),
        "r_comp_abs_dev_base": _pearson(comp, np.abs(yes - base)),
        "r_comp_agreement": _pearson(comp, agr),
        "r_bias_yesrate": _pearson(bias_logit, yes),
        "n_annotators": int(len(comp)),
    }
    if out["d_common"] is not None and out.get("freeze_bias"):
        # Só no braço CONGELADO. No braço livre `u - v = 2*(d_common + d_ind)`
        # com `d_ind` treinado, logo um desvio grande é o comportamento
        # correcto e reportá-lo como "FAIL" seria um erro de leitura.
        dev = float(np.abs(bias_logit - 2.0 * out["d_common"]).max())
        out["frozen_invariant_max_dev"] = dev
        out["frozen_invariant_ok"] = bool(dev < 1e-5)
    out["_competence"] = comp
    return out


# ---------------------------------------------------------------------------
# curva de detecção (run corrompida)
# ---------------------------------------------------------------------------
def _auc(pos, neg):
    """Área sob a ROC por estatística de ranks (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)).astype(float) + 1.0
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def detection_curve(run, comp, man=None, path=None):
    """Competência por nível de corrupção + AUC. `comp` vem de `sd_diagnostics`."""
    if man is None or comp is None:
        return None
    sel = {e["annotator_idx"]: e for e in man["entries"]}
    by_level = {}
    for idx, e in sel.items():
        if idx < len(comp):
            by_level.setdefault(e["level"], []).append(float(comp[idx]))
    clean = [float(c) for i, c in enumerate(comp) if i not in sel]
    levels = sorted(by_level)
    rows = []
    ctrl = np.asarray(by_level.get(0.0, []), float)
    for f in levels:
        v = np.asarray(by_level[f], float)
        rows.append({
            "level": f, "n": int(len(v)),
            "competence_mean": float(v.mean()), "competence_sd": float(v.std(ddof=1)),
            # AUC contra o GRUPO DE CONTROLO INTERNO (nível 0%), escolhido do
            # mesmo modo. Comparar com "todos os limpos" confundiria o efeito do
            # nível com o da selecção.
            "auc_vs_level0": (None if f == 0.0 or len(ctrl) == 0
                              else _auc(-v, -ctrl)),
        })
    heavy = np.concatenate([by_level[f] for f in levels if f >= 0.3]) if any(
        f >= 0.3 for f in levels) else np.array([])
    # ⚠ Monotonia ESTRITA entre pares adjacentes NÃO é critério: com 20
    # anotadores por nível e sd da competência ~0,03, o par 0%->10% inverte-se
    # por ruído mesmo com o mecanismo a funcionar (10% de 27 votos ~ 3 votos,
    # abaixo do ruído binomial da própria taxa). A tendência mede-se por
    # Spearman sobre os 120, e a Porta 1 decide-se pelo AUC dos níveis >=40%.
    lv_all = [e["level"] for i, e in sel.items() if i < len(comp)]
    cp_all = [float(comp[i]) for i in sel if i < len(comp)]
    return {
        "manifest": os.path.basename(path),
        "spearman_level_vs_competence": _spearman(lv_all, cp_all),
        "n_votes_flipped": man["n_votes_flipped"],
        "frac_votes_flipped": man["frac_votes_flipped"],
        "rows": rows,
        "n_clean": len(clean),
        "auc_heavy_vs_level0": (None if len(heavy) == 0 or len(ctrl) == 0
                                else _auc(-heavy, -ctrl)),
        "auc_50_vs_level0": next(
            (r["auc_vs_level0"] for r in rows
             if abs(r["level"] - 0.5) < 1e-9), None),
        # DESCRITIVO, não pré-registado: o limiar de 0,75 foi escolhido depois
        # de ver a curva. A Porta 1 decide-se pelo AUC dos níveis >=40% (§12).
        "detection_threshold_descriptive": next(
            (r["level"] for r in rows
             if r["auc_vs_level0"] is not None and r["auc_vs_level0"] >= 0.75),
            None),
    }


# ---------------------------------------------------------------------------
# régua de votos
# ---------------------------------------------------------------------------
def vote_metrics(run, step_at_E, epoch_index):
    """Métricas dos preditores na época E, casadas por `global_step`.

    ⚠ `on_train_end` repete a última avaliação, logo `records` tem N_EPOCHS+1
    entradas. Casar por posição daria a época errada.
    """
    vpath = run["path"].replace(".json", "_votes.json")
    if not os.path.exists(vpath):
        return None
    data = json.load(open(vpath, encoding="utf-8"))
    recs = data["records"]
    rec = next((r for r in recs if int(r["global_step"]) == int(step_at_E)), None)
    if rec is None:
        rec = recs[epoch_index] if epoch_index < len(recs) else recs[-1]
        rec = dict(rec, _matched_by="posição (fallback)")
    else:
        rec = dict(rec, _matched_by="global_step")
    return {"record": rec, "meta": {k: v for k, v in data.items() if k != "records"},
            "n_records": len(recs), "path": os.path.basename(vpath)}


def trajectory(run, vm):
    """`b − b_perm` e `b − d4_train` por época, contra o `eval_loss`.

    Descritivo. Serve para ver se a régua de votos e o selector de época
    (`eval_loss`) apontam para épocas diferentes -- e, se apontarem, para o
    dizer explicitamente em vez de escolher a que dá jeito.
    """
    if vm is None:
        return None
    vpath = run["path"].replace(".json", "_votes.json")
    recs = json.load(open(vpath, encoding="utf-8"))["records"][:N_EPOCHS]
    out = []
    for rec, ev in zip(recs, run["evals"]):
        m = rec["metrics"]
        out.append({"epoch": rec["epoch"], "eval_loss": float(ev["eval_loss"]),
                    "b_minus_bperm": m["b"]["nll"] - m["b_perm"]["nll"],
                    "b_minus_d4": m["b"]["nll"] - m["d4_train"]["nll"]})
    return out


def verdict(m):
    """Vereditos pré-declarados da §12. Nenhum dos negativos é passagem."""
    if m is None:
        return {"status": "SEM RÉGUA", "why": "ficheiro `_votes.json` ausente"}
    n = {k: m[k]["nll"] for k in PREDICTORS if k in m}
    notes = []
    if n["b"] >= n["d4_train"]:
        if n["b"] < n.get("d3_train", np.inf):
            return {"status": "NULO (atribuído)", "why":
                    "b bate d3 mas NÃO d4 -> a matriz aprendeu o U na taxa base",
                    "nlls": n}
        return {"status": "NULO", "why": "b não bate d4_train", "nlls": n}
    if n["b"] >= n.get("d4_val", np.inf):
        return {"status": "NULO", "why":
                "b bate d4_train mas não o oráculo d4_val -> má calibração do "
                "ajuste, não sinal em C_a", "nlls": n}
    if abs(n["b"] - n["b_perm"]) < 1e-4:
        return {"status": "NULO", "why":
                "b == b_perm -> a vantagem não vem de conhecer AQUELA pessoa",
                "nlls": n}
    notes.append(f"b - b_perm = {n['b'] - n['b_perm']:+.5f}")
    notes.append(f"b - d4_train = {n['b'] - n['d4_train']:+.5f}")
    return {"status": "SINAL", "why": "; ".join(notes), "nlls": n}


# ---------------------------------------------------------------------------
# relatório
# ---------------------------------------------------------------------------
ARM_NAMES = {
    "M1sd": "viés livre (run A -- termo de comparação)",
    "M1sdcrp": "CONTROLO POSITIVO, anotadores corrompidos (run B -- Porta 1)",
    "M1sdfrz": "viés individual CONGELADO (runs C/D/E -- Porta 2)",
    "M1sdfrzcrp": "congelado + corrompido",
}


def analyse():
    runs = discover()
    if not runs:
        return {"runs": [], "note": "nenhuma run da Fase 2 (`*_conf_M1sd*`) encontrada"}
    stats_clean, strata, ids = annotator_reference()
    out = []
    for key, run in sorted(runs.items()):
        i, ep, step, lo = best_epoch(run["evals"])
        # A run corrompida lê-se com as estatísticas do mundo CORROMPIDO -- é
        # nesse mundo que as fichas foram treinadas.
        man, mpath = (find_manifest(run) if "crp" in run["arm"] else (None, None))
        stats = annotator_reference(man)[0] if man is not None else stats_clean
        diag = sd_diagnostics(run, i, stats)
        vm = vote_metrics(run, step, i)
        row = {
            "arm": run["arm"], "lam": run["lam"], "seed": run["seed"],
            "conf_lr": run["conf_lr"], "tag": run["tag"],
            "E": ep, "E_index": i, "eval_loss_E": lo,
            "primary_E": float(run["evals"][i].get(PRIMARY, float("nan"))),
            "diagnostics": diag,
            "votes": vm,
            "trajectory": trajectory(run, vm),
            "verdict": verdict(vm["record"]["metrics"] if vm else None),
        }
        if "crp" in run["arm"]:
            row["detection"] = detection_curve(
                run, diag["_competence"] if diag else None, man, mpath)
        if diag:
            diag.pop("_competence", None)
        out.append(row)
    return {"runs": out,
            "panel_yes_rate": stats_clean["panel_yes_rate"],
            "n_unseen_annotators": stats_clean["n_unseen_annotators"]}


def build_report(res):
    L = ["# Fase 2 — régua de votos, controlo positivo e viés congelado", "",
         "Gerado por `reliability/analyze_votes_experiment.py`. Não treina nada.",
         "Protocolo e critério: `reliability/CONFUSION_EXPERIMENT.md` §§10-18.",
         "", "Época reportada: **E = argmin(eval_loss)** — nunca a métrica "
         "reportada (isso seria leakage).", ""]
    if not res.get("runs"):
        L += ["> " + res.get("note", "sem runs"), ""]
        return "\n".join(L)

    L += [f"Taxa base do painel (só treino): **{res['panel_yes_rate']:.4f}** · "
          f"anotadores sem votos de treino: {res['n_unseen_annotators']}", ""]

    L += ["## 1. Régua — surpresa (NLL) nos votos retidos, em E", "",
          "⚠ **`b` contra `a` é descritivo, não prova.** `b` já bate `a` no init "
          "por encolhimento da previsão (calibração global, não sinal de "
          "anotador — §14). O contraste limpo do sinal individual é "
          "**`b` contra `b_perm`**, e o critério é **`b` contra `d4_train`**.", ""]
    hdr = "| braço | seed | E | " + " | ".join(PREDICTORS) + " |"
    L += [hdr, "|" + "---|" * (len(PREDICTORS) + 3)]
    for r in res["runs"]:
        v = r["votes"]
        if not v:
            L.append(f"| {r['arm']} | {r['seed']} | {r['E']:.0f} | "
                     + " | ".join(["—"] * len(PREDICTORS)) + " |")
            continue
        m = v["record"]["metrics"]
        L.append(f"| {r['arm']} | {r['seed']} | {r['E']:.0f} | "
                 + " | ".join(f"{m[k]['nll']:.4f}" if k in m else "—"
                              for k in PREDICTORS) + " |")
    L += ["", "Acerto (%) nos mesmos votos:", "",
          "| braço | seed | " + " | ".join(PREDICTORS) + " |",
          "|" + "---|" * (len(PREDICTORS) + 2)]
    for r in res["runs"]:
        v = r["votes"]
        if not v:
            continue
        m = v["record"]["metrics"]
        L.append(f"| {r['arm']} | {r['seed']} | "
                 + " | ".join(f"{100 * m[k]['acc']:.1f}" if k in m else "—"
                              for k in PREDICTORS) + " |")

    L += ["", "### As duas diferenças que decidem", "",
          "`b − b_perm` isola o sinal INDIVIDUAL (mesma calibração global dos "
          "dois lados; muda só cada pessoa receber a sua ficha ou a de outra). "
          "`b − d4_train` é o critério. **Negativo = a ficha ganha.**", "",
          "| braço | seed | E | b − a | **b − b_perm** | **b − d4_train** | b − d4_val |",
          "|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        v = r["votes"]
        if not v:
            continue
        m = v["record"]["metrics"]
        g = lambda k: m[k]["nll"]
        L.append(f"| {r['arm']} | {r['seed']} | {r['E']:.0f} | "
                 f"{g('b') - g('a'):+.4f} | **{g('b') - g('b_perm'):+.4f}** | "
                 f"**{g('b') - g('d4_train'):+.4f}** | "
                 f"{g('b') - g('d4_val'):+.4f} |")

    L += ["", "### Trajectória ao longo das épocas", "",
          "⚠ A época reportada é **E = argmin(eval_loss)**, que é o selector do "
          "SISTEMA. O modelo de anotador é outro objecto e pode ter outro óptimo. "
          "A trajectória vai aqui como **descritivo**; mudar o selector para a "
          "régua de votos seria escolher a época pela métrica reportada, que o "
          "protocolo proíbe.", ""]
    for r in res["runs"]:
        v = r["votes"]
        if not v:
            continue
        recs = v["record"]
        traj = r.get("trajectory")
        if not traj:
            continue
        L += [f"`{r['arm']} seed {r['seed']}`  (E = {r['E']:.0f})", "",
              "| ép | eval_loss | b − b_perm | b − d4_train |", "|---|---|---|---|"]
        for t in traj:
            mark = "  ← **E**" if abs(t["epoch"] - r["E"]) < 1e-9 else ""
            L += [f"| {t['epoch']:.0f} | {t['eval_loss']:.4f} | "
                  f"{t['b_minus_bperm']:+.4f} | {t['b_minus_d4']:+.4f}{mark} |"]
        L.append("")

    L += ["", "## 2. Veredito por braço", ""]
    for r in res["runs"]:
        vd = r["verdict"]
        L += [f"- **{r['arm']} seed {r['seed']}** ({ARM_NAMES.get(r['arm'], '?')}) "
              f"→ **{vd['status']}** — {vd['why']}"]

    L += ["", "## 3. O que as matrizes aprenderam (competência e viés em E)", "",
          "Viés em **espaço de logit** (`u − v`). Em probabilidade ele varia com "
          "a competência mesmo com o viés individual congelado a zero, por "
          "não-linearidade da sigmóide — reportar ali faria o braço congelado "
          "parecer que continua a aprender viés (§13.4).", "",
          "| braço | seed | comp. média | comp. sd | viés logit sd | "
          "r(comp, yes_rate) | **r(comp, \\|yes−base\\|)** | r(comp, concord.) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        d = r["diagnostics"]
        if not d:
            continue
        f = lambda x: "—" if x is None else f"{x:+.3f}"
        L.append(f"| {r['arm']} | {r['seed']} | {d['competence_mean']:.4f} | "
                 f"{d['competence_sd']:.4f} | {d['bias_logit_sd']:.4f} | "
                 f"{f(d['r_comp_yesrate_linear'])} | "
                 f"**{f(d['r_comp_abs_dev_base'])}** | "
                 f"{f(d['r_comp_agreement'])} |")
    L += ["", "Referência da Fase 1 (viés livre, `delta`): competência sd "
          "**0,066**; `r(comp, |yes_rate − base|)` = −0,758 / −0,731 / −0,747; "
          "`r(comp, concordância)` = +0,217 / +0,231 / +0,248.", ""]

    for r in res["runs"]:
        d = r["diagnostics"]
        if d and d.get("frozen_invariant_ok") is not None and d.get("freeze_bias"):
            ok = "PASS" if d["frozen_invariant_ok"] else "**FAIL**"
            L.append(f"- Invariante do congelado ({r['arm']} seed {r['seed']}): "
                     f"{ok} — `max|(u−v) − 2·d_common|` = "
                     f"{d['frozen_invariant_max_dev']:.2e} "
                     f"(d_common = {d['d_common']:+.4f})")

    det = [r for r in res["runs"] if r.get("detection")]
    if det:
        L += ["", "## 4. Curva de detecção — controlo positivo (Porta 1)", "",
              "Anotadores com uma fracção conhecida dos votos invertida. A "
              "comparação é contra o **grupo de controlo interno (nível 0 %)**, "
              "escolhido do mesmo modo — comparar com 'todos os limpos' "
              "confundiria o efeito do nível com o da selecção.", ""]
        for r in det:
            dd = r["detection"]
            L += [f"`{r['tag']}` — {dd['n_votes_flipped']} votos invertidos "
                  f"({100 * dd['frac_votes_flipped']:.2f}% do total), "
                  f"manifesto `{dd['manifest']}`", "",
                  "| nível | n | competência média | sd | AUC vs nível 0% |",
                  "|---|---|---|---|---|"]
            for row in dd["rows"]:
                auc = "—" if row["auc_vs_level0"] is None else f"{row['auc_vs_level0']:.3f}"
                L.append(f"| {row['level']:.0%} | {row['n']} | "
                         f"{row['competence_mean']:.4f} | "
                         f"{row['competence_sd']:.4f} | {auc} |")
            thr = dd["detection_threshold_descriptive"]
            L += ["", f"- Spearman(nível, competência) sobre os 120 "
                  f"anotadores: "
                  + ("—" if dd["spearman_level_vs_competence"] is None
                     else f"**{dd['spearman_level_vs_competence']:+.3f}**"),
                  "- AUC do nível 50% contra o 0%: "
                  + ("—" if dd["auc_50_vs_level0"] is None
                     else f"**{dd['auc_50_vs_level0']:.3f}**"),
                  f"- AUC dos níveis ≥30 % contra o nível 0 %: "
                  + ("—" if dd["auc_heavy_vs_level0"] is None
                     else f"**{dd['auc_heavy_vs_level0']:.3f}**"),
                  f"- limiar de detecção *descritivo* (1.º nível com AUC ≥ 0,75, "
                  f"escolhido post-hoc): "
                  + ("**nenhum**" if thr is None else f"**{thr:.0%}**"), "",
                  "**PORTA 1** — se o mecanismo não separa o nível de 50 % do "
                  "grupo de controlo, nenhum nulo dos braços congelados é "
                  "interpretável e essas runs não devem ser feitas.", ""]

    L += ["", "## 5. ICM em E (contexto, não critério)", "",
          "O ICM compara dois **treinos**, logo tem por baixo o ruído de seed: "
          "sd do próprio A0 entre seeds = **0,0059**, contra um efeito medido de "
          "+0,0009 na Fase 1. Fica aqui por simetria e honestidade.", "",
          "⚠ **O ICM de uma run corrompida (`*crp`) não se compara com nada** — "
          "os dados de treino foram estragados de propósito.", "",
          "| braço | seed | E | eval_loss | ICM-Soft-Norm (A, soft) |",
          "|---|---|---|---|---|"]
    for r in res["runs"]:
        L.append(f"| {r['arm']} | {r['seed']} | {r['E']:.0f} | "
                 f"{r['eval_loss_E']:.6f} | {r['primary_E']:.4f} |")
    L.append("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = analyse()
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    report = build_report(res)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n-> {REPORT_PATH}\n-> {JSON_PATH}")


if __name__ == "__main__":
    main()
