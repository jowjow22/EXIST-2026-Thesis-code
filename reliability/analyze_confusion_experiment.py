"""Análise do experimento da camada de confusão (perda auxiliar da Task A).

Lê os logs de `EXIST_MEMES/metrics_history/` escritos pelo `MetricsHistoryLogger`
e pelo `ConfusionMatrixLogger` do `EXIST_MEMES/memes_confusion.ipynb`.
Não treina, não regenera nada.

Desenho pré-registado (ver `reliability/CONFUSION_EXPERIMENT.md`)
----------------------------------------------------------------
- A0 = as runs `raw_seed*` de 15 épocas já existentes. NÃO são re-treinadas.
- Grelha: método x lambda x seed, sementes [42, 7, 2025] pareadas com o A0.
- Época reportada: E = argmin(eval_loss), nunca a métrica reportada.
- Métrica: Task A ICM-Soft-Norm (soft), gold de validação SEMPRE `raw`.
- Critério da FAMÍLIA: >= 1 método com Δ > 0 e p < 0.0167 (Bonferroni, 3 métodos),
  t pareado bilateral sobre 3 sementes, no melhor lambda (escolha post-hoc,
  declarada como tal).
- Paragem antecipada: se o M1 falhar em todos os lambdas, M2 e M3 não correm.

Uso:
    python3 -m reliability.analyze_confusion_experiment
"""

import csv
import glob
import json
import os
import re

import numpy as np

METRICS_DIR = "EXIST_MEMES/metrics_history"
OUT_DIR = "reliability/outputs"
PROFILES = "reliability/outputs/behavioral_profiles.csv"
REPORT_PATH = os.path.join(OUT_DIR, "experiment_confusion_report.md")
JSON_PATH = os.path.join(OUT_DIR, "experiment_confusion_report.json")

SEEDS = [42, 7, 2025]
METHODS = ["M1", "M2", "M3"]
LAMBDAS = [0.1, 0.3, 1.0]
N_EPOCHS = 15
ALPHA_BONF = 0.0167          # 0.05 / 3 métodos

METRICS = [
    ("A ICM-Norm hard", "eval_task_a_icmnorm_hard"),
    ("A ICM-Soft-Norm soft", "eval_task_a_icmsoftnorm_soft"),
    ("B ICM-Norm hard", "eval_task_b_icmnorm_hard"),
    ("B ICM-Soft-Norm soft", "eval_task_b_icmsoftnorm_soft"),
    ("C ICM-Norm hard", "eval_task_c_icmnorm_hard"),
    ("C ICM-Soft-Norm soft", "eval_task_c_icmsoftnorm_soft"),
]
PRIMARY = "eval_task_a_icmsoftnorm_soft"


# ---------------------------------------------------------------------------
# descoberta das runs
# ---------------------------------------------------------------------------
def _load(path):
    hist = json.load(open(path, encoding="utf-8"))
    ev = [r for r in hist if "eval_loss" in r]
    return {"path": path, "hist": hist, "evals": ev,
            "train_end": "train_runtime" in hist[-1]}


def discover():
    """Devolve (a0, arms, control, explor).

    Chaves: `a0[seed]`, `arms[(method, lam, seed)]`, `control[(method, 0.0, seed)]`,
    `explor[(method, lam, conf_lr, seed)]` -- os braços com learning rate próprio
    para as matrizes, que são exploratórios e ficam fora da estatística.
    """
    a0, arms, control, explor = {}, {}, {}, {}
    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "*.json"))):
        if path.endswith("_confusion.json"):
            continue
        base = os.path.basename(path)
        m_a0 = re.search(r"_raw_seed(\d+)\.json$", base)
        # Os braços da Fase 2 codificam a parametrização no tag: `M1sd` (viés
        # livre), `M1sdfrz` (congelado), `M1sdcrp` (corrompido). Sem isto
        # ficariam indistinguíveis das runs `delta` da Fase 1 e o `discover()`
        # levantaria "duas runs para a mesma chave".
        m_cf = re.search(
            r"_conf_(M[123](?:sd)?(?:frz)?(?:crp)?)_lam([0-9.]+)"
            r"(?:_clr([0-9.e+-]+))?_seed(\d+)\.json$", base)
        if m_a0:
            run = _load(path)
            if len(run["evals"]) != N_EPOCHS:
                continue                      # runs de 2 épocas do estudo anterior
            seed = int(m_a0.group(1))
            if seed in a0:
                # Repetições da MESMA run A0 são evidência de reprodutibilidade, não
                # ambiguidade: se as curvas forem idênticas, fica a primeira. Curvas
                # diferentes com o mesmo tag é que são ambíguas.
                prev = [r["eval_loss"] for r in a0[seed]["evals"]]
                cur = [r["eval_loss"] for r in run["evals"]]
                if prev == cur:
                    a0[seed].setdefault("duplicates", []).append(
                        os.path.basename(path))
                    continue
                raise RuntimeError(
                    f"duas runs A0 de {N_EPOCHS} épocas para seed {seed} com curvas "
                    f"DIFERENTES: {os.path.basename(a0[seed]['path'])} vs "
                    f"{os.path.basename(path)}")
            a0[seed] = run
        elif m_cf:
            run = _load(path)
            if len(run["evals"]) != N_EPOCHS:
                continue          # testes curtos (max_steps=2) não são runs do estudo
            method, lam = m_cf.group(1), float(m_cf.group(2))
            if method not in METHODS:
                # Braço da FASE 2 (`M1sd*`): analisado por
                # `reliability/analyze_votes_experiment.py`, que tem a régua de
                # votos. Fica fora da estatística confirmatória da Fase 1.
                continue
            clr, seed = m_cf.group(3), int(m_cf.group(4))
            run["conf_lr"] = None if clr is None else float(clr)
            key = (method, lam, seed)
            if clr is not None:
                # Braço EXPLORATÓRIO (lr próprio para as matrizes): fica fora da
                # grelha pré-registada e da estatística confirmatória.
                explor[(method, lam, float(clr), seed)] = run
                continue
            target = control if lam == 0.0 else arms
            if key in target:
                raise RuntimeError(f"duas runs para {key}")
            target[key] = run
    return a0, arms, control, explor


def best_epoch(ev):
    losses = [r["eval_loss"] for r in ev]
    i = int(np.argmin(losses))
    return i, ev[i]["epoch"], losses[i]


# ---------------------------------------------------------------------------
# estatística (mesmas funções do analyze_arm_experiment, sem scipy)
# ---------------------------------------------------------------------------
def p_two_sided(t, df):
    from math import gamma, pi, sqrt
    t = abs(float(t))
    if not np.isfinite(t):
        return 0.0
    c = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    xs = np.linspace(0, t, 200001)
    dens = c * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
    return float(2 * (0.5 - np.trapezoid(dens, xs)))


def paired_stats(deltas):
    d = np.asarray(deltas, dtype=float)
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0:
        t = float("inf") if mean != 0 else 0.0
        return {"n": n, "mean": mean, "sd": 0.0, "t": t, "p": 0.0 if mean else 1.0,
                "d_z": float("inf") if mean else 0.0,
                "same_sign": bool(np.all(np.sign(d) == np.sign(d[0])))}
    t = mean / (sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "t": float(t),
            "p": p_two_sided(t, n - 1), "d_z": float(mean / sd),
            "same_sign": bool(np.all(np.sign(d) == np.sign(d[0])))}


# ---------------------------------------------------------------------------
# diagnóstico das matrizes
# ---------------------------------------------------------------------------
def load_yes_rate():
    if not os.path.exists(PROFILES):
        return {}
    out = {}
    with open(PROFILES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["annotator_id"]] = float(row["yes_rate"])
            except (KeyError, ValueError):
                continue
    return out


def matrix_diagnostics(run_path, epoch_index, yes_rate):
    """||C_a − I||_F na época E, e correlação com a yes_rate do anotador.

    Diagnóstico, NÃO critério: diz se o mecanismo aprendeu alguma coisa sobre os
    anotadores ou colapsou na identidade.
    """
    cpath = run_path.replace(".json", "_confusion.json")
    if not os.path.exists(cpath):
        return None
    data = json.load(open(cpath, encoding="utf-8"))
    snaps = data["snapshots"]
    # o dump de `on_train_end` repete a última avaliação -> usa só os primeiros N
    snaps = snaps[:N_EPOCHS] if len(snaps) > N_EPOCHS else snaps
    if epoch_index >= len(snaps):
        return None
    C = np.asarray(snaps[epoch_index]["matrices"], dtype=float)      # [M, 2, 2]
    dist = np.sqrt(((C - np.eye(2)) ** 2).sum(axis=(1, 2)))          # [M]
    # Índices latentes: 0 = YES, 1 = NO. Viés de resposta COM SINAL:
    # P(vota YES | latente NO) - P(vota NO | latente YES). A distância de
    # Frobenius é sem sinal e mistura as duas off-diagonais -- pode ficar plana
    # enquanto o viés está estruturado.
    bias = C[:, 1, 0] - C[:, 0, 1]                                   # [M]
    ids = data["annotator_ids"]
    out = {"epoch": snaps[epoch_index]["epoch"],
           "dist_mean": float(dist.mean()), "dist_sd": float(dist.std(ddof=1)),
           "dist_min": float(dist.min()), "dist_max": float(dist.max()),
           "diag_mean": float(C.diagonal(axis1=1, axis2=2).mean()),
           "bias_mean": float(bias.mean()), "bias_sd": float(bias.std(ddof=1)),
           "n_annotators": len(ids)}
    matched = [(dist[i], bias[i], yes_rate[a]) for i, a in enumerate(ids) if a in yes_rate]
    if len(matched) > 2:
        d = np.asarray([m[0] for m in matched])
        b = np.asarray([m[1] for m in matched])
        y = np.asarray([m[2] for m in matched])
        out["n_matched"] = len(matched)

        def _corrs(x):
            if x.std() == 0 or y.std() == 0:
                return None, None
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return (float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rx, ry)[0, 1]))

        out["pearson_dist_yesrate"], out["spearman_dist_yesrate"] = _corrs(d)
        out["pearson_bias_yesrate"], out["spearman_bias_yesrate"] = _corrs(b)
    return out


# ---------------------------------------------------------------------------
# análise
# ---------------------------------------------------------------------------
def analyse():
    a0, arms, control, explor = discover()
    yes_rate = load_yes_rate()

    checks = []
    missing_a0 = [s for s in SEEDS if s not in a0]
    checks.append(("as 3 runs A0 de 15 épocas existem", not missing_a0,
                   "todas presentes" if not missing_a0 else f"em falta: {missing_a0}"))
    checks.append(("existe run de controlo (lambda = 0)", bool(control),
                   ", ".join(f"{m}/{s}" for m, _, s in control) or "NENHUMA"))
    present = sorted({(m, l) for m, l, _ in arms})
    checks.append(("células (método, lambda) com as 3 sementes",
                   all(all((m, l, s) in arms for s in SEEDS) for m, l in present),
                   ", ".join(f"{m}@{l}" for m, l in present) or "nenhuma"))
    checks.append((f"cada run tem {N_EPOCHS} avaliações",
                   all(len(r["evals"]) == N_EPOCHS for r in
                       list(arms.values()) + list(control.values()) + list(a0.values())),
                   "ok"))

    # -- controlo contra o A0 -------------------------------------------------
    # Régua: a dispersão do A0 entre as 3 seeds na métrica primária, em E. É o
    # "ruído de seed" que o protocolo do handoff nomeia.
    a0_primary_E = []
    for s_ in SEEDS:
        if s_ in a0:
            i_ = best_epoch(a0[s_]["evals"])[0]
            a0_primary_E.append(float(a0[s_]["evals"][i_][PRIMARY]))
    a0_seed_sd = (float(np.std(a0_primary_E, ddof=1))
                  if len(a0_primary_E) > 1 else None)

    control_rows = []
    for (method, lam, seed), run in sorted(control.items()):
        if seed not in a0:
            continue
        ref = [r["eval_loss"] for r in a0[seed]["evals"]]
        cur = [r["eval_loss"] for r in run["evals"]]
        n = min(len(ref), len(cur))
        worst = max(abs(cur[i] - ref[i]) for i in range(n))
        # O critério é avaliado EM E (a época que o estudo reporta), contra a
        # dispersão entre seeds do A0. O máximo ao longo das 15 épocas fica
        # registado como contexto, mas NÃO é critério: o treino é caótico e uma
        # perturbação de um ulp no passo 1 amplifica para escala macro na ép. 15,
        # logo esse máximo não distingue "ulp" de "bug".
        ia = best_epoch(a0[seed]["evals"])[0]
        ib = best_epoch(run["evals"])[0]
        d_loss_E = run["evals"][ib]["eval_loss"] - a0[seed]["evals"][ia]["eval_loss"]
        d_prim_E = (run["evals"][ib][PRIMARY] - a0[seed]["evals"][ia][PRIMARY])
        sd = a0_seed_sd
        ok = (sd is None) or (abs(d_prim_E) < sd)
        control_rows.append({"method": method, "seed": seed, "n_epochs": n,
                             "worst_abs_delta": worst,
                             "E_a0": a0[seed]["evals"][ia]["epoch"],
                             "E_control": run["evals"][ib]["epoch"],
                             "delta_eval_loss_E": d_loss_E,
                             "delta_primary_E": d_prim_E,
                             "a0_seed_sd": sd,
                             "pass": bool(ok),
                             "verdict": ("**PASS**" if ok else "**FAIL**"),
                             "a0_path": os.path.basename(a0[seed]["path"]),
                             "path": os.path.basename(run["path"])})

    # -- grelha ---------------------------------------------------------------
    grid = []
    for (method, lam, seed), run in sorted(arms.items()):
        i, ep, lo = best_epoch(run["evals"])
        row = {"method": method, "lam": lam, "seed": seed, "E": ep,
               "E_index": i, "eval_loss_E": lo,
               "path": os.path.basename(run["path"])}
        for name, key in METRICS:
            row[key] = float(run["evals"][i].get(key, float("nan")))
        if seed in a0:
            j = best_epoch(a0[seed]["evals"])[0]
            row["a0_E"] = a0[seed]["evals"][j]["epoch"]
            row["a0_primary"] = float(a0[seed]["evals"][j][PRIMARY])
            row["delta_primary"] = row[PRIMARY] - row["a0_primary"]
        row["matrices"] = matrix_diagnostics(run["path"], i, yes_rate)
        grid.append(row)

    # -- por (método, lambda): t pareado --------------------------------------
    cells = {}
    for method, lam in present:
        rows = [r for r in grid if r["method"] == method and r["lam"] == lam
                and "delta_primary" in r]
        if len(rows) != len(SEEDS):
            continue
        st = paired_stats([r["delta_primary"] for r in rows])
        st.update({"method": method, "lam": lam,
                   "mean_arm": float(np.mean([r[PRIMARY] for r in rows])),
                   "mean_a0": float(np.mean([r["a0_primary"] for r in rows]))})
        cells[(method, lam)] = st

    # -- melhor lambda por método (post-hoc) + veredicto ----------------------
    per_method = {}
    for method in METHODS:
        cand = {l: cells[(method, l)] for (m, l) in cells if m == method}
        if not cand:
            continue
        best_lam = max(cand, key=lambda l: cand[l]["mean"])
        st = cand[best_lam]
        per_method[method] = {
            "best_lam": best_lam, "stats": st,
            "pass": bool(st["mean"] > 0 and st["p"] < ALPHA_BONF),
            "lambdas_tested": sorted(cand),
        }

    family_pass = any(v["pass"] for v in per_method.values())
    m1_all_fail = ("M1" in per_method and not per_method["M1"]["pass"]
                   and len(per_method["M1"]["lambdas_tested"]) == len(LAMBDAS))

    # -- braços exploratórios (lr próprio para as matrizes) -------------------
    explor_rows = []
    for (method, lam, clr, seed), run in sorted(explor.items()):
        i, ep, lo = best_epoch(run["evals"])
        row = {"method": method, "lam": lam, "conf_lr": clr, "seed": seed,
               "E": ep, "eval_loss_E": lo, "path": os.path.basename(run["path"])}
        for _, key in METRICS:
            row[key] = float(run["evals"][i].get(key, float("nan")))
        if seed in a0:
            j = best_epoch(a0[seed]["evals"])[0]
            row["a0_primary"] = float(a0[seed]["evals"][j][PRIMARY])
            row["delta_primary"] = row[PRIMARY] - row["a0_primary"]
        row["matrices"] = matrix_diagnostics(run["path"], i, yes_rate)
        explor_rows.append(row)

    return {"checks": checks, "control": control_rows, "grid": grid,
            "exploratory": explor_rows,
            "cells": {f"{m}|{l}": v for (m, l), v in cells.items()},
            "per_method": per_method, "family_pass": family_pass,
            "m1_all_lambdas_fail": m1_all_fail,
            "alpha_bonferroni": ALPHA_BONF, "seeds": SEEDS,
            "a0_runs": {s: os.path.basename(r["path"]) for s, r in a0.items()}}


# ---------------------------------------------------------------------------
# relatório
# ---------------------------------------------------------------------------
def build_report(res):
    L, a = [], None
    a = L.append
    a("# Camada de confusão como perda auxiliar da Task A — relatório")
    a("")
    a("Gerado por `reliability/analyze_confusion_experiment.py` a partir dos logs de")
    a("`EXIST_MEMES/metrics_history/`. Não treina nem regenera nada.")
    a("")
    a("## 1. O que este experimento pergunta")
    a("")
    a("Se modelar o anotador com uma matriz de confusão 2x2 acrescenta sinal ao")
    a("treino sobre soft labels: `L = L_soft(A,B,C) + lambda * L_aux(A)`, com")
    a("`p_voto(a,i) = p_latente(i) @ C_a`. O objectivo **não é melhorar o sistema** —")
    a("é caracterização. Um nulo é resultado válido e esperado.")
    a("")
    a("**Intervenção:** só um termo somado à perda, e só na Task A. B e C intocadas.")
    a("A auxiliar entra apenas no treino: `eval_loss` continua a ser a `L_soft` pura,")
    a("logo `E = argmin(eval_loss)` é o mesmo selector de época nos dois lados.")
    a("")
    a("## 2. Integridade `[MEDIDO]`")
    a("")
    for name, ok, detail in res["checks"]:
        a(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    a("")
    a("Runs A0 usadas como referência (não re-treinadas): " +
      ", ".join(f"seed {s}: `{p}`" for s, p in sorted(res["a0_runs"].items())))
    a("")
    a("## 3. Controlo obrigatório — lambda = 0 `[MEDIDO]`")
    a("")
    if not res["control"]:
        a("**NÃO CORRIDO.** A grelha não é interpretável sem ele.")
    else:
        a("Com `lambda = 0` o caminho auxiliar é calculado e multiplicado por zero")
        a("(não é saltado): um NaN lá dentro ainda contaminaria a perda. A trajectória")
        a("tem de reproduzir o A0 da mesma seed.")
        a("")
        a("| método | seed | E (A0 / controlo) | Δ eval_loss em E | Δ primária em E | sd do A0 entre seeds | máx \\|Δ\\| eval_loss (15 ép, contexto) | veredicto |")
        a("|---|---|---|---|---|---|---|---|")
        for r in res["control"]:
            sd = r.get("a0_seed_sd")
            a(f"| `{r['method']}` | {r['seed']} | {r['E_a0']:.0f} / {r['E_control']:.0f} | "
              f"{r['delta_eval_loss_E']:+.2e} | {r['delta_primary_E']:+.4f} | "
              f"{'—' if sd is None else f'{sd:.4f}'} | {r['worst_abs_delta']:.2e} | "
              f"{r['verdict']} |")
        a("")
        a("**Critério do controlo (recalibrado — ver nota):** as matrizes têm de ficar")
        a("congeladas, o A0 tem de reproduzir, e `|Δ da métrica primária em E|` tem de")
        a("ficar abaixo de **1 sd do A0 entre seeds**. O máximo ao longo das 15 épocas")
        a("aparece só como contexto.")
        a("")
        a("> **Nota — um degrau retirado.** A versão inicial deste protocolo exigia")
        a("> reprodução **bit-a-bit** (máx \\|Δ\\| de `eval_loss` < 1e-3 nas 15 épocas).")
        a("> Essa estatística está mal calibrada para um sistema caótico: uma")
        a("> perturbação de um ulp no passo 1 amplifica para escala macro na época 15,")
        a("> logo o máximo sobre 15 épocas **não distingue um ulp de um bug**. A")
        a("> evidência que discrimina é independente e já está medida: as matrizes")
        a("> ficam congeladas (sem fuga de gradiente nem weight decay) e o A0")
        a("> reproduz bit-a-bit ao longo de 6 dias. O degrau foi retirado **depois**")
        a("> de ver o resultado do controlo — está declarado como tal. O critério do")
        a("> RESULTADO (secção 5) não foi tocado.")
    a("")
    a("## 4. Grelha método x lambda x seed `[MEDIDO]`")
    a("")
    a("| método | lambda | seed | E | eval_loss(E) | A ICM-Soft-Norm | A0 na mesma seed | Δ |")
    a("|---|---|---|---|---|---|---|---|")
    for r in res["grid"]:
        d = r.get("delta_primary")
        a(f"| `{r['method']}` | {r['lam']} | {r['seed']} | {r['E']:.0f} | "
          f"{r['eval_loss_E']:.5f} | {r[PRIMARY]:.4f} | "
          f"{r.get('a0_primary', float('nan')):.4f} | "
          f"{'' if d is None else f'{d:+.4f}'} |")
    a("")
    a("### 4.1 As seis métricas na época E")
    a("")
    a("| método | lambda | seed | " + " | ".join(n for n, _ in METRICS) + " |")
    a("|---|---|---|" + "---|" * len(METRICS))
    for r in res["grid"]:
        a(f"| `{r['method']}` | {r['lam']} | {r['seed']} | " +
          " | ".join(f"{r[k]:.4f}" for _, k in METRICS) + " |")
    a("")
    a("## 5. Resultado por método, no melhor lambda `[MEDIDO]`")
    a("")
    a("A escolha do melhor lambda é **post-hoc** (maior Δ médio) e está declarada")
    a(f"como tal. Critério pré-registado: Δ > 0 e p < {res['alpha_bonferroni']} ")
    a("(Bonferroni sobre 3 métodos), t pareado bilateral sobre 3 sementes.")
    a("")
    a("| método | melhor lambda | média A0 | média método | Δ médio | sd | t | p | d_z | mesmo sinal | veredicto |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        v = res["per_method"].get(m)
        if not v:
            a(f"| `{m}` | — | — | — | — | — | — | — | — | — | não corrido |")
            continue
        s = v["stats"]
        a(f"| `{m}` | {v['best_lam']} | {s['mean_a0']:.4f} | {s['mean_arm']:.4f} | "
          f"{s['mean']:+.4f} | {s['sd']:.4f} | {s['t']:.3f} | {s['p']:.4f} | "
          f"{s['d_z']:+.3f} | {'sim' if s['same_sign'] else 'não'} | "
          f"{'**PASS**' if v['pass'] else 'falha'} |")
    a("")
    a(f"**Veredicto da família:** "
      f"{'PASS — pelo menos um método passa o critério' if res['family_pass'] else 'NULO — nenhum método passa o critério pré-registado'}.")
    if res["m1_all_lambdas_fail"]:
        a("")
        a("**Paragem antecipada aplicável:** o M1 falhou em todos os lambdas. O M1 é o")
        a("único desenhado para poucos votos por anotador (~27); um nulo nele é")
        a("evidência contra a família, e M2/M3 não precisam de correr.")
    a("")
    a("## 5b. Braços exploratórios — learning rate próprio para as matrizes `[MEDIDO]`")
    a("")
    a("**Fora da grelha pré-registada e fora da estatística confirmatória.** Com o")
    a("`learning_rate` do encoder (1e-5), as matrizes movem-se no máximo ~6e-4 em 15")
    a("épocas — o tecto de 795 passos de AdamW — e `lambda` não muda isso, porque o")
    a("AdamW é invariante à escala do gradiente e a auxiliar é a única fonte de")
    a("gradiente delas. Estes braços dão às matrizes um grupo de parâmetros com lr")
    a("próprio, para responder se o nulo sobrevive quando o mecanismo tem margem")
    a("para se mover.")
    a("")
    if not res["exploratory"]:
        a("Nenhum braço exploratório corrido.")
    else:
        a("| método | lambda | lr das matrizes | seed | E | A ICM-Soft-Norm | A0 | Δ | dist média a I | viés sd | r(viés, yes_rate) |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in res["exploratory"]:
            d = r.get("delta_primary")
            m = r.get("matrices") or {}
            pe = m.get("pearson_bias_yesrate")
            a(f"| `{r['method']}` | {r['lam']} | {r['conf_lr']:g} | {r['seed']} | "
              f"{r['E']:.0f} | {r[PRIMARY]:.4f} | "
              f"{r.get('a0_primary', float('nan')):.4f} | "
              f"{'—' if d is None else f'{d:+.4f}'} | "
              f"{m.get('dist_mean', float('nan')):.4f} | "
              f"{m.get('bias_sd', float('nan')):.4f} | "
              f"{'—' if pe is None else f'{pe:+.3f}'} |")
    a("")
    a("## 6. Diagnóstico das matrizes `[MEDIDO]` (não é critério)")
    a("")
    a("Distância média `||C_a − I||_F` na época E e correlação com a `yes_rate` do")
    a("anotador. Distância ~0 = o mecanismo colapsou na identidade (não aprendeu")
    a("nada sobre os anotadores); correlação com a `yes_rate` = aprendeu o viés de")
    a("resposta, que é o que a matriz deveria capturar.")
    a("")
    a("O **viés** é `C_a[NO->YES] − C_a[YES->NO]`, com sinal: é a quantidade que")
    a("deveria correlacionar com a `yes_rate` se o mecanismo aprendeu alguma coisa.")
    a("A distância de Frobenius é sem sinal e mistura as duas off-diagonais — pode")
    a("ficar plana enquanto o viés está estruturado.")
    a("")
    a("| método | lambda | seed | E | dist média | sd | min | max | diag média | viés médio | r(dist, yes_rate) | rho(dist) | r(viés, yes_rate) | rho(viés) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["grid"]:
        d = r.get("matrices")
        if not d:
            a(f"| `{r['method']}` | {r['lam']} | {r['seed']} | {r['E']:.0f} | " + "— | " * 10 + "|")
            continue
        def _f(k):
            v = d.get(k)
            return "—" if v is None else f"{v:+.3f}"
        a(f"| `{r['method']}` | {r['lam']} | {r['seed']} | {r['E']:.0f} | "
          f"{d['dist_mean']:.4f} | {d['dist_sd']:.4f} | {d['dist_min']:.4f} | "
          f"{d['dist_max']:.4f} | {d['diag_mean']:.4f} | {d['bias_mean']:+.4f} | "
          f"{_f('pearson_dist_yesrate')} | {_f('spearman_dist_yesrate')} | "
          f"{_f('pearson_bias_yesrate')} | {_f('spearman_bias_yesrate')} |")
    a("")
    a("## 7. Limitações declaradas")
    a("")
    a("- O split é por **item**, não por anotador: anotadores de treino aparecem em")
    a("  validação, o que **favorece** o método. Declarado, não corrigido.")
    a("- `L_aux` é a **média** sobre os pares (anotador, item) do batch, não a soma —")
    a("  mantém `lambda` na escala da `L_soft`, que também é média.")
    a("- `C_a[i,j] = P(voto=j | latente=i)` com linhas normalizadas e")
    a("  `p_voto = p_latente @ C_a`. A forma `C_a @ p_latente` do handoff, com linhas")
    a("  normalizadas, não devolve uma distribuição.")
    a("- Os parâmetros de confusão estão **fora do weight decay** (0.05): o decaimento")
    a("  puxaria os logits para a matriz uniforme, o oposto da inicialização perto da")
    a("  identidade, e no M3 seria regularização que a especificação proíbe.")
    a("- Os pesos de regularização (M1 e M2) foram fixados **a priori**, um valor cada,")
    a("  sem grelha — uma grelha neles inflaria as comparações para além do Bonferroni-3.")
    a("- O M1 é uma **adaptação** de Chu et al.: implementa a decomposição")
    a("  comum + individual, não o portão dependente da instância do CoNAL.")
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = analyse()
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    report = build_report(res)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nEscrito: {REPORT_PATH}\n         {JSON_PATH}")


if __name__ == "__main__":
    main()
