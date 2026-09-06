"""
Análise pareada do experimento de ABLAÇÃO de anotadores na Task A, 15 épocas.

Braços (RUN_TAG no nome do ficheiro de log):
  A0_baseline      -> `raw`               (reutiliza as runs `raw` de 15 épocas)
  R1_rem_extremos  -> `R1_rem_extremos`   (remove os anotadores extremos)
  R2_rem_controle  -> `R2_rem_controle`   (remove um controlo emparelhado, sorteio 42)

Lê os logs de `EXIST_MEMES/metrics_history/`. Não treina, não regenera labels,
não ajusta nada.

CRITÉRIO PRÉ-REGISTADO
----------------------
Fixado na especificação da tarefa em 2026-09-06, **depois** de as runs terminarem
(2026-09-05, pelos timestamps dos ficheiros) mas **antes** de qualquer valor de
métrica de R1/R2 ser inspeccionado.

- A comparação **decisiva** é **R1 − R2** na **Task A ICM-Soft-Norm (soft)**, na
  época E = argmin(eval_loss).
- **R1 − A0** e **R2 − A0** são **contexto, NÃO evidência de efeito de critério**:
  ambas incluem a remoção de densidade (menos votos por item), que é exactamente
  aquilo que o controlo emparelhado R2 existe para isolar. Só R1 vs R2 tem a
  remoção de densidade dos dois lados.

Regras herdadas do estudo A0/A1, mantidas sem alteração:
- Split congelado em disco, idêntico em todos os braços (`split_taskA_frozen.json`).
- Gold de validação SEMPRE `raw`, em todos os braços — a régua não muda com o alvo.
- Época reportada: E = argmin(eval_loss), NUNCA a métrica reportada (evita leakage).
- Regra de simetria: o estado FINAL também é reportado, para todos os braços.

Uso:
    python3 -m reliability.analyze_ablation_experiment
"""

import glob
import json
import os
import re

import numpy as np

from reliability.analyze_arm_experiment import (
    METRICS, N_EPOCHS, PRIMARY, SEEDS, best_epoch, p_two_sided, stats,
)

METRICS_DIR = "EXIST_MEMES/metrics_history"
LABELS_DIR = "reliability/outputs"
OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "experiment_taskA_ablation_report.md")
JSON_PATH = os.path.join(OUT_DIR, "experiment_taskA_ablation_report.json")

# tag do log -> (nome do braço, ficheiro de soft labels que o treino consumiu)
ARMS = {
    "raw":             ("A0_baseline",     "soft_labels_taskA_raw.json"),
    "R1_rem_extremos": ("R1_rem_extremos", "soft_labels_taskA_R1_rem_extremos.json"),
    "R2_rem_controle": ("R2_rem_controle", "soft_labels_taskA_R2_rem_controle.json"),
}
ARM_ORDER = ["raw", "R1_rem_extremos", "R2_rem_controle"]

# (rótulo, braço A, braço B, papel). Δ = A − B.
COMPARISONS = [
    ("R1 − R2", "R1_rem_extremos", "R2_rem_controle", "DECISIVA"),
    ("R1 − A0", "R1_rem_extremos", "raw",             "contexto"),
    ("R2 − A0", "R2_rem_controle", "raw",             "contexto"),
]
DECISIVE = "R1 − R2"

# ficheiro de labels do A0 gerado pela ablação; deve ser equivalente ao `raw` que
# as runs reutilizadas de facto consumiram — verificado, não assumido.
A0_ABLATION_LABELS = "soft_labels_taskA_A0_baseline.json"


def discover():
    """Localiza a run de 15 épocas de cada (braço, seed).

    O `N_EPOCHS` filtra as runs de 2 épocas do estudo anterior, que partilham o
    tag `raw_seed*`.
    """
    found = {}
    pat = re.compile(r"_(" + "|".join(map(re.escape, ARMS)) + r")_seed(\d+)\.json$")
    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "*.json"))):
        m = pat.search(path)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        hist = json.load(open(path, encoding="utf-8"))
        ev = [r for r in hist if "eval_loss" in r]
        if len(ev) != N_EPOCHS:
            continue
        key = (arm, seed)
        if key in found:
            raise RuntimeError(f"duas runs de {N_EPOCHS} épocas para {key}")
        found[key] = {"path": path, "hist": hist, "evals": ev,
                      "train_end": "train_runtime" in hist[-1]}
    return found


def _load_labels(name):
    with open(os.path.join(LABELS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def a0_reuse_check():
    """As runs `raw` só podem servir de A0 se o alvo que elas consumiram for o
    mesmo que a ablação chama de A0_baseline.

    O notebook faz `torch.tensor(label, dtype=torch.float32)`, logo a igualdade
    que importa é em float32, não em float64: diferenças de 1 ulp em float64
    (1−p vs divisão directa) desaparecem na entrada real do treino.
    """
    raw = _load_labels(ARMS["raw"][1])
    a0 = _load_labels(A0_ABLATION_LABELS)
    same_keys = set(raw) == set(a0)
    ks = sorted(set(raw) & set(a0))
    r64 = np.array([raw[k] for k in ks], dtype=np.float64)
    a64 = np.array([a0[k] for k in ks], dtype=np.float64)
    d64 = np.abs(r64 - a64)
    same32 = bool(np.array_equal(r64.astype(np.float32), a64.astype(np.float32)))
    return {
        "same_keys": bool(same_keys), "n_items": len(ks),
        "n_diff_float64": int((d64 > 0).any(axis=1).sum()),
        "max_abs_diff_float64": float(d64.max()),
        "max_abs_diff_pyes_float64": float(d64[:, 0].max()),
        "identical_float32": same32,
    }


def target_context():
    """Deslocamento do alvo ANTES do treino, medido nos ficheiros de labels."""
    lab = {arm: _load_labels(f) for arm, (_n, f) in ARMS.items()}
    ks = sorted(set(lab["raw"]))
    p = {arm: np.array([lab[arm][k][0] for k in ks], dtype=float) for arm in ARMS}

    grid = {ARMS[a][0]: int(len(np.unique(np.round(p[a], 6)))) for a in ARM_ORDER}

    vs_a0 = {}
    for arm in ("R1_rem_extremos", "R2_rem_controle"):
        d = p[arm] - p["raw"]
        aff = np.abs(d) > 1e-9
        vs_a0[ARMS[arm][0]] = {
            "n_affected": int(aff.sum()),
            "mean_abs_delta_affected": float(np.abs(d[aff]).mean()) if aff.any() else 0.0,
            "mean_abs_delta_all": float(np.abs(d).mean()),
            "crossings_0_5": int((((p["raw"] >= 0.5) != (p[arm] >= 0.5))).sum()),
        }

    d12 = p["R1_rem_extremos"] - p["R2_rem_controle"]
    aff12 = np.abs(d12) > 1e-9
    pair = {
        "n_affected": int(aff12.sum()),
        "mean_abs_delta_all": float(np.abs(d12).mean()),
        "mean_abs_delta_affected": float(np.abs(d12[aff12]).mean()),
        "crossings_0_5": int(((p["R1_rem_extremos"] >= 0.5)
                              != (p["R2_rem_controle"] >= 0.5)).sum()),
    }
    diff_of_means = abs(vs_a0["R1_rem_extremos"]["mean_abs_delta_affected"]
                        - vs_a0["R2_rem_controle"]["mean_abs_delta_affected"])
    return {"grid": grid, "vs_a0": vs_a0, "r1_vs_r2": pair,
            "diff_of_mean_displacements": float(diff_of_means),
            "n_items": len(ks)}


def integrity(found, a0chk):
    checks = []
    missing = [(a, s) for a in ARMS for s in SEEDS if (a, s) not in found]
    checks.append((f"as 9 runs (3 braços × {len(SEEDS)} seeds) existem", not missing,
                   "todas presentes" if not missing else f"em falta: {missing}"))
    ok_ep = all(len(v["evals"]) == N_EPOCHS for v in found.values())
    checks.append((f"cada run tem {N_EPOCHS} avaliações", ok_ep,
                   ", ".join(f"{a}/{s}={len(found[(a,s)]['evals'])}"
                             for a in ARM_ORDER for s in SEEDS if (a, s) in found)))
    ok_end = all(v["train_end"] for v in found.values())
    checks.append(("cada run terminou (`train_runtime` no log)", ok_end,
                   "todas com on_train_end" if ok_end else "run truncada"))
    ok_keys = all(all(k in v["evals"][0] for _, k in METRICS)
                  for v in found.values())
    checks.append(("as 6 métricas ICM estão em todas as runs", ok_keys, "ok"))
    ok_a0 = a0chk["same_keys"] and a0chk["identical_float32"]
    checks.append((
        "A0_baseline == alvo consumido pelas runs `raw` (em float32, que é o dtype "
        "que entra no treino)", ok_a0,
        f"{a0chk['n_items']} itens; {a0chk['n_diff_float64']} diferem em float64 "
        f"(max |Δ| = {a0chk['max_abs_diff_float64']:.2e}, max |Δ p_YES| = "
        f"{a0chk['max_abs_diff_pyes_float64']:.2e}); idênticos em float32: "
        f"{a0chk['identical_float32']}"))
    return checks


def paired(found, key, arm_a, arm_b, point="best"):
    """Δ = arm_a − arm_b por semente, na métrica `key`."""
    out = {}
    for s in SEEDS:
        vals = {}
        for arm in (arm_a, arm_b):
            ev = found[(arm, s)]["evals"]
            i = best_epoch(ev)[0] if point == "best" else len(ev) - 1
            vals[arm] = float(ev[i][key])
        out[s] = {"a": vals[arm_a], "b": vals[arm_b],
                  "delta": vals[arm_a] - vals[arm_b]}
    return out


def _stats_from(ps):
    d = np.asarray([ps[s]["delta"] for s in SEEDS], dtype=float)
    st = stats({s: {"delta": float(d[i])} for i, s in enumerate(SEEDS)})
    return st


def compute(found):
    res = {}
    for point in ("best", "final"):
        res[point] = {}
        for lbl, a_, b_, _role in COMPARISONS:
            res[point][lbl] = {}
            for _mlbl, key in METRICS:
                ps = paired(found, key, a_, b_, point)
                st = _stats_from(ps)
                res[point][lbl][key] = {
                    "per_seed": ps, "stats": st,
                    "p": p_two_sided(st["t"], len(SEEDS) - 1)}
    return res


def _metric_table(a, res_point, lbl):
    a("| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |")
    a("|---|---|---|---|---|---|---|---|")
    for mlbl, key in METRICS:
        r = res_point[lbl][key]
        st = r["stats"]
        a(f"| {mlbl} | {r['per_seed'][42]['delta']:+.4f} | "
          f"{r['per_seed'][7]['delta']:+.4f} | {r['per_seed'][2025]['delta']:+.4f} | "
          f"**{st['mean']:+.4f}** | {st['sd']:.4f} | {st['d_z']:+.2f} | "
          f"{r['p']:.3f} |")
    a("")


def build_report(found, checks, res, curves, ctx, a0chk):
    L = []
    a = L.append
    a("# Ablação de anotadores na Task A — A0 / R1 / R2, 15 épocas, 3 sementes pareadas")
    a("")
    a("Gerado por `reliability/analyze_ablation_experiment.py` a partir dos logs de")
    a("`EXIST_MEMES/metrics_history/`. Não treina, não regenera labels.")
    a("")
    a("## 0. Proveniência e ordem dos factos")
    a("")
    a("- Runs concluídas em **2026-09-05** (timestamps nos nomes dos ficheiros de log).")
    a("- Critério decisivo fixado em **2026-09-06**, na especificação da tarefa, e")
    a("  codificado neste script **antes** de qualquer valor de métrica de R1/R2 ser")
    a("  lido. As verificações feitas antes disso foram: existência e integridade das")
    a("  runs, e o deslocamento do alvo nos ficheiros de labels — nenhuma métrica ICM.")
    a("- Braços e ficheiros de alvo consumidos pelo treino:")
    for arm in ARM_ORDER:
        name, f = ARMS[arm]
        a(f"  - **{name}** — RUN_TAG `{arm}`, alvo `{f}`")
    a("")
    a("## 1. Critério pré-registado")
    a("")
    a("**Decisiva: R1 − R2, Task A ICM-Soft-Norm (soft), na época E = argmin(eval_loss).**")
    a("")
    a("`R1 − A0` e `R2 − A0` são **contexto, não evidência de efeito de critério**.")
    a("Ambas misturam duas coisas: o critério pelo qual os anotadores foram removidos")
    a("**e** a remoção de densidade (o item passa a ter menos votos). O controlo")
    a("emparelhado R2 existe precisamente para tirar a densidade da equação: em")
    a("`R1 − R2` a remoção de densidade está presente **dos dois lados** e cancela-se")
    a("no pareamento; o que sobra é o critério de selecção.")
    a("")
    a("**Intervenção:** apenas o alvo de treino da Task A. Os anotadores removidos")
    a("saem **só** do alvo da Task A — continuam a contar em B e C. O que se observa")
    a("em B e C é **spillover do encoder partilhado**, não efeito directo.")
    a("")
    a("## 2. Integridade das runs `[MEDIDO]`")
    a("")
    for name, ok, detail in checks:
        a(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    a("")
    a("As runs `raw` reutilizadas como A0 são as mesmas do estudo A0/A1")
    a("(`experiment_taskA_15ep_report.md`) — não houve retreino do baseline.")
    a("")
    a("| braço | seed | ficheiro | E = argmin(eval_loss) | eval_loss em E |")
    a("|---|---|---|---|---|")
    for arm in ARM_ORDER:
        for s in SEEDS:
            f = found[(arm, s)]
            i, ep, lo = best_epoch(f["evals"])
            a(f"| `{ARMS[arm][0]}` | {s} | `{os.path.basename(f['path'])}` | "
              f"**{ep:.0f}** | {lo:.5f} |")
    a("")
    eps = sorted({int(best_epoch(found[(a_, s)]["evals"])[1])
                  for a_ in ARM_ORDER for s in SEEDS})
    if len(eps) > 1:
        odd = [f"`{ARMS[a_][0]}`/{s}" for a_ in ARM_ORDER for s in SEEDS
               if int(best_epoch(found[(a_, s)]["evals"])[1]) != eps[0]]
        a(f"`[MEDIDO]` E não é o mesmo em todas as runs: {', '.join(odd)} tem "
          f"E = {eps[-1]}, as restantes E = {eps[0]}. Cada braço é lido no seu")
        a("próprio E, como o critério pré-registado manda — a alternativa (forçar a")
        a("mesma época nos dois lados) escolheria a época por conveniência de")
        a("comparação, que é a porta de entrada do leakage que a regra fecha.")
        a("")
    a("## 3. O alvo, antes do treino `[MEDIDO]`")
    a("")
    a("### 3.1 A ablação fragmenta a grade do alvo")
    a("")
    a("| braço | valores distintos possíveis no alvo |")
    a("|---|---|")
    for arm in ARM_ORDER:
        a(f"| {ARMS[arm][0]} | {ctx['grid'][ARMS[arm][0]]} |")
    a("")
    a(f"A0 tem **{ctx['grid']['A0_baseline']}** valores distintos (grade de 6 votos:")
    a(f"0, 1/6, …, 1). R1 tem **{ctx['grid']['R1_rem_extremos']}** e R2 tem "
      f"**{ctx['grid']['R2_rem_controle']}**, porque os itens ficam com densidades")
    a("**mistas** — 6, 5 ou 4 votos, conforme quantos dos anotadores removidos")
    a("tocavam aquele item — e cada densidade traz a sua própria grade (1/5, 1/4, …).")
    a("")
    a("Isto **contamina qualquer leitura contra o A0**: parte do Δ vem de o alvo ter")
    a("passado a viver numa grade diferente, não do critério de remoção. É mais uma")
    a("razão para a comparação decisiva ser R1 vs R2, onde a grade é a mesma dos dois")
    a("lados.")
    a("")
    same_grid = ctx["grid"]["R1_rem_extremos"] == ctx["grid"]["R2_rem_controle"]
    a("`[LIMITAÇÃO]` O que está verificado é a **grade de valores** "
      + (f"(as duas iguais, {ctx['grid']['R1_rem_extremos']} valores)."
         if same_grid else "(as duas DIFEREM — ver tabela acima).")
      + " O **histograma de densidades** (quantos itens ficam com 6, 5 ou 4")
    a("votos em cada braço) **não foi verificado** — o gerador dos ficheiros de")
    a("ablação vive no notebook `EXIST_MEMES/reliability_tests.ipynb` e não deixou")
    a("manifesto em disco. Se essa distribuição diferir entre R1 e R2, sobra um")
    a("resíduo de densidade na comparação decisiva. Fica como item a fechar.")
    a("")
    a("### 3.2 Deslocamento do alvo face ao A0")
    a("")
    a("| braço | itens afectados | \\|Δ\\| médio (afectados) | \\|Δ\\| médio (todos) | cruzam 0,5 |")
    a("|---|---|---|---|---|")
    for arm in ("R1_rem_extremos", "R2_rem_controle"):
        v = ctx["vs_a0"][arm]
        a(f"| {arm} | {v['n_affected']} | {v['mean_abs_delta_affected']:.4f} | "
          f"{v['mean_abs_delta_all']:.4f} | {v['crossings_0_5']} |")
    a("")
    a(f"Diferença entre os dois deslocamentos médios: "
      f"**{ctx['diff_of_mean_displacements']:.4f}**.")
    a("")
    a("### 3.3 Distância directa R1 ↔ R2 no alvo")
    a("")
    a("A diferença dos deslocamentos médios (3.2) **não** é a distância entre os dois")
    a("alvos: dois braços podem afastar-se do A0 na mesma magnitude e em direcções")
    a("diferentes. A quantidade que limita o que `R1 − R2` pode mostrar é a distância")
    a("directa entre eles:")
    a("")
    p12 = ctx["r1_vs_r2"]
    a(f"- itens em que R1 e R2 diferem: **{p12['n_affected']}** de {ctx['n_items']}")
    a(f"- \\|Δ\\| médio sobre todos os itens: **{p12['mean_abs_delta_all']:.4f}**")
    a(f"- \\|Δ\\| médio sobre os afectados: **{p12['mean_abs_delta_affected']:.4f}**")
    a(f"- itens que cruzam 0,5 entre R1 e R2: **{p12['crossings_0_5']}**")
    a("")
    a("## 4. Resultado decisivo — R1 − R2, época E `[MEDIDO]`")
    a("")
    a("Métrica pré-registada: **Task A ICM-Soft-Norm (soft)**.")
    a("")
    a("| seed | R1 | R2 | Δ = R1 − R2 |")
    a("|---|---|---|---|")
    pr = res["best"][DECISIVE][PRIMARY]
    for s in SEEDS:
        d = pr["per_seed"][s]
        a(f"| {s} | {d['a']:.4f} | {d['b']:.4f} | **{d['delta']:+.4f}** |")
    st = pr["stats"]
    a(f"| **média** | | | **{st['mean']:+.4f}** |")
    a("")
    a(f"sd(Δ) = {st['sd']:.4f} · t({st['n']-1}) = {st['t']:.2f} · "
      f"p = {pr['p']:.3f} · d_z = {st['d_z']:.2f} · "
      f"mesmo sinal em 3/3: {'sim' if st['same_sign'] else 'não'}")
    a("")
    verdict = ("NULO — não sustentado" if pr["p"] >= 0.05
               else "EFEITO SUSTENTADO ao nível 0,05")
    a(f"**Veredito (critério pré-registado): {verdict}.**")
    a("")
    a("## 5. As três comparações, na época E `[MEDIDO]`")
    a("")
    for lbl, _a, _b, role in COMPARISONS:
        tag = "**DECISIVA**" if role == "DECISIVA" else "contexto — não é evidência de efeito de critério"
        a(f"### 5.{COMPARISONS.index((lbl, _a, _b, role))+1} {lbl} ({tag})")
        a("")
        _metric_table(a, res["best"], lbl)
    a("`[LIMITAÇÃO]` **Multiplicidade.** A tabela completa deste relatório tem")
    a(f"3 comparações × {len(METRICS)} métricas × 2 pontos de leitura = "
      f"{3*len(METRICS)*2} células. **Uma** é decisiva (R1 − R2, Task A")
    a("ICM-Soft-Norm, época E). Qualquer p < 0,05 avulso nas outras 35 é o que se")
    a("espera por acaso e não deve ser lido como achado.")
    a("")
    a("## 6. Regra de simetria — estado FINAL (época 15) `[MEDIDO]`")
    a("")
    a("O mesmo critério aplicado a todos os braços. Reportado porque a definição de")
    a("baseline adoptada na Task 3 é o estado final da run, e misturar «fim de um")
    a("lado, pico do outro» é o erro que essa regra existe para impedir.")
    a("")
    for lbl, _a, _b, role in COMPARISONS:
        a(f"### 6.{COMPARISONS.index((lbl, _a, _b, role))+1} {lbl}")
        a("")
        _metric_table(a, res["final"], lbl)
    mins = {k: int(np.argmin(v)) + 1 for k, v in curves.items()}
    span = sorted(set(mins.values()))
    a("`[LIMITAÇÃO]` **Nada nesta secção é um achado.** O mínimo de `eval_loss` cai")
    a(f"na época {span[0] if len(span)==1 else '/'.join(map(str, span))} nas nove "
      f"runs; a época 15 está em sobreajuste profundo. A secção existe para cumprir")
    a("a regra de simetria, não para inferência.")
    sig = [f"{lbl} / {mlbl} (p = {res['final'][lbl][key]['p']:.3f})"
           for lbl, _x, _y, _r in COMPARISONS
           for mlbl, key in METRICS
           if res["final"][lbl][key]["p"] < 0.05]
    if sig:
        a((f"A célula com p < 0,05 aqui — {sig[0]} — é" if len(sig) == 1
           else f"As {len(sig)} células com p < 0,05 aqui — " + "; ".join(sig)
                + " — são")
          + " exactamente o tipo de célula que a multiplicidade explica (§5).")
    a("")
    a("## 7. Curvas de `eval_loss` `[MEDIDO]`")
    a("")
    a("| braço | seed | " + " | ".join(f"ép {i+1}" for i in range(N_EPOCHS)) + " |")
    a("|---|---|" + "---|" * N_EPOCHS)
    for arm in ARM_ORDER:
        for s in SEEDS:
            a(f"| `{ARMS[arm][0]}` | {s} | " +
              " | ".join(f"{v:.4f}" for v in curves[(arm, s)]) + " |")
    a("")
    a("`[MEDIDO]` Mínimo de `eval_loss` por run: " +
      ", ".join(f"`{ARMS[a_][0]}`/{s}=ép {mins[(a_, s)]}"
                for a_ in ARM_ORDER for s in SEEDS) + ".")
    a("")
    a("## 8. Dimensionamento `[MEDIDO]`")
    a("")
    a("Sementes necessárias para 80% de poder (α = 0,05 bilateral, aproximação")
    a("normal), a partir do d_z observado na época E, na comparação **decisiva**:")
    a("")
    a("| métrica | d_z | sementes para 80% |")
    a("|---|---|---|")
    for mlbl, key in METRICS:
        st_ = res["best"][DECISIVE][key]["stats"]
        n80 = st_["n_for_80"]
        a(f"| {mlbl} | {st_['d_z']:+.2f} | {n80 if n80 else '—'} |")
    a("")
    a("`[LIMITAÇÃO]` Com n = 3 o sd é ele próprio muito ruidoso; estes números são")
    a("ordens de grandeza, não estimativas. A leitura útil é binária: «cabe em ~10")
    a("sementes» ou «não cabe em nenhum orçamento razoável».")
    a("")
    a("## 9. Interpretação")
    a("")
    dec = res["best"][DECISIVE][PRIMARY]
    dst = dec["stats"]
    a("### 9.1 O critério de remoção não se distingue do controlo emparelhado")
    a("")
    a(f"Na comparação decisiva, na métrica e na época pré-registadas: "
      f"**Δ = {dst['mean']:+.4f}, p = {dec['p']:.3f}, d_z = {dst['d_z']:.2f}**, "
      f"mesmo sinal em "
      f"{'3/3' if dst['same_sign'] else 'menos de 3/3'} sementes.")
    a("")
    if dec["p"] >= 0.05:
        a("**Veredito: nulo.** Remover os anotadores extremos **não produz um alvo")
        a("mensuravelmente melhor** do que remover o mesmo número de anotadores")
        a("sorteados ao acaso. O que a ablação mexe nas métricas é compatível com a")
        a("perda de votos, não com o critério pelo qual os votos foram escolhidos.")
    else:
        a("**Veredito: efeito sustentado ao nível 0,05** na comparação decisiva — o")
        a("critério de remoção distingue-se do controlo emparelhado. Ver §4.")
    a("")
    a("### 9.2 O nulo não vem de os dois alvos serem quase iguais")
    a("")
    p12 = ctx["r1_vs_r2"]
    r1a = ctx["vs_a0"]["R1_rem_extremos"]
    r2a = ctx["vs_a0"]["R2_rem_controle"]
    a("Objecção óbvia a um nulo entre dois braços de ablação: «os alvos eram")
    a("praticamente o mesmo, claro que não deu diferença». **Os números dizem o")
    a("contrário.**")
    a("")
    a(f"R1 e R2 discordam em **{p12['n_affected']}** dos {ctx['n_items']} itens, com")
    a(f"\\|Δ\\| médio de **{p12['mean_abs_delta_affected']:.4f}** nos afectados e")
    a(f"**{p12['crossings_0_5']}** itens a mudar de lado dos 0,5 — **mais**")
    a(f"cruzamentos do que R1 gera contra o A0 ({r1a['crossings_0_5']}) ou R2 contra")
    a(f"o A0 ({r2a['crossings_0_5']}). A intervenção era grande à entrada; não")
    a("chegou à saída.")
    a("")
    a("### 9.3 O `p` nominal em R1 − A0 não é um achado")
    a("")
    ctxr = res["best"]["R1 − A0"][PRIMARY]
    ctxr2 = res["best"]["R2 − A0"][PRIMARY]
    a(f"`R1 − A0` na métrica primária dá Δ = {ctxr['stats']['mean']:+.4f}, "
      f"p = {ctxr['p']:.3f} — nominalmente abaixo de 0,05. **Não é evidência de")
    a("efeito de critério**, por três razões, e a primeira estava fixada antes de")
    a("qualquer número ser visto:")
    a("")
    a("1. **Pré-registo.** `R1 − A0` foi declarado contexto. Confunde critério com")
    a("   remoção de densidade e fragmentação da grade (§3.1).")
    a(f"2. **Multiplicidade.** É uma célula entre {3*len(METRICS)*2}. Uma abaixo de")
    a("   0,05 é o que se espera por acaso.")
    a("3. **O teste decisivo, que isola o critério, não a sustenta**")
    a(f"   (p = {dec['p']:.3f}).")
    a("")
    a("**O que separa os dois `p`, mecanicamente.** As médias são quase iguais "
      f"({ctxr['stats']['mean']:+.4f} contra {dst['mean']:+.4f}); o que muda é o")
    a("denominador. O `t` pareado divide pelo **sd das diferenças por semente**, e")
    a("essa é a quantidade que difere:")
    a("")
    a("| comparação | Δ por semente (42 / 7 / 2025) | sd(Δ) | t | p |")
    a("|---|---|---|---|---|")
    for lbl, _x, _y, _r in COMPARISONS:
        rr = res["best"][lbl][PRIMARY]
        a(f"| {lbl} | " + " / ".join(f"{rr['per_seed'][s]['delta']:+.4f}" for s in SEEDS)
          + f" | {rr['stats']['sd']:.4f} | {rr['stats']['t']:.2f} | {rr['p']:.3f} |")
    a("")
    sd_by_arm = {}
    for arm in ARM_ORDER:
        vals = [float(found[(arm, s)]["evals"][
            best_epoch(found[(arm, s)]["evals"])[0]][PRIMARY]) for s in SEEDS]
        sd_by_arm[ARMS[arm][0]] = (vals, float(np.std(vals, ddof=1)))
    a("E **não** é que o R2 seja um braço mais ruidoso em nível absoluto — a")
    a("dispersão entre sementes é comparável nos três:")
    a("")
    a("| braço | valor por semente (42 / 7 / 2025) | sd entre sementes |")
    a("|---|---|---|")
    for arm in ARM_ORDER:
        vals, sd_ = sd_by_arm[ARMS[arm][0]]
        a(f"| {ARMS[arm][0]} | " + " / ".join(f"{v:.4f}" for v in vals)
          + f" | {sd_:.4f} |")
    a("")
    sd_ctx = res["best"]["R1 − A0"][PRIMARY]["stats"]["sd"]
    sd_dec = dst["sd"]
    ratio = sd_dec / sd_ctx if sd_ctx else float("inf")
    a(f"O que difere é o **co-movimento entre sementes**: R1 e A0 sobem e descem")
    a(f"juntos (sd(Δ) = {sd_ctx:.4f}), enquanto o R2 não acompanha nenhum dos dois")
    a(f"(sd(Δ) = {sd_dec:.4f}, **{ratio:.1f}×** maior). Mesmo Δ médio, denominador")
    a(f"{ratio:.1f}× maior, `t` proporcionalmente menor — é isso, e não uma")
    a(f"diferença de efeito, que separa p = {ctxr['p']:.3f} de p = {dec['p']:.3f}.")
    a("")
    a("E a inferência que isto autoriza é mais forte do que a objecção de")
    a("multiplicidade sozinha: **a instabilidade entre sementes introduzida por")
    a("remover anotadores AO ACASO é da mesma ordem do deslocamento causado por")
    a("remover os extremos**. O «efeito» do critério cabe dentro do piso de ruído da")
    a("própria ablação. Isso fecha também a leitura inversa — «o controlo é que é")
    a("errático, logo `R1 − A0` é o sinal bom»: essa erraticidade **é** a medida de")
    a("quanto uma remoção arbitrária mexe no resultado, e é a escala contra a qual o")
    a("critério tem de ser julgado.")
    a("")
    a("Dito isto, há um padrão coerente que vale a pena registar **como hipótese, não")
    a("como resultado**: `R2 − A0` na mesma métrica é "
      f"{ctxr2['stats']['mean']:+.4f} (p = {ctxr2['p']:.3f}), ou seja **a remoção de")
    a("densidade sozinha não custa nada**, enquanto R1 fica abaixo do A0 e abaixo do")
    a("R2. As três leituras apontam na mesma direcção: remover os extremos custa um")
    a("pouco, em vez de ganhar. `[HIPÓTESE]` — a magnitude está dentro do ruído a")
    a("n = 3 e o teste que a decidiria não a sustenta.")
    a("")
    a("### 9.4 A régua favorece o A0 por construção — e é por isso que a decisiva é R1 vs R2")
    a("")
    a("O gold de validação é o `raw` em todos os braços, por desenho. R1 e R2 treinam")
    a("ambos para um alvo com menos votos que o da régua, logo **os dois** partem em")
    a("desvantagem contra o A0. Essa desvantagem **cancela-se no pareamento R1 − R2**,")
    a("que é exactamente a razão de ser o teste decisivo. Ler qualquer um dos dois")
    a("contra o A0 como «a ablação piorou o modelo» é ler o desencontro alvo-régua.")
    a("")
    a("### 9.5 O que faria falta para fechar")
    a("")
    n80 = dst["n_for_80"]
    a(f"Com d_z = {dst['d_z']:.2f} na métrica decisiva, seriam precisas "
      f"**≈{n80} sementes** para 80% de poder — mesma ordem de grandeza do estudo")
    a("A0/A1 (≈6), e dentro de um orçamento razoável. Mas a direcção que essas")
    a("sementes iriam confirmar é")
    a("**negativa**: no melhor dos casos estabelece-se que remover os extremos tem um")
    a("**custo** pequeno, nunca um ganho. Correr mais sementes não pode virar isto num")
    a("resultado positivo.")
    a("")
    a("`[LIMITAÇÃO]` Fica por fechar o histograma de densidades por braço (§3.1). Se")
    a("R1 e R2 não removerem votos na mesma distribuição de itens, sobra um resíduo de")
    a("densidade dentro da própria comparação decisiva.")
    a("")
    a("### 9.6 Onde isto encaixa no capítulo")
    a("")
    a("Duas linhas independentes, o mesmo desfecho:")
    a("")
    a("1. **Ponderar** os votos pela competência estimada (θ do MACE corrigido) — nulo")
    a("   (`experiment_taskA_15ep_report.md`).")
    a("2. **Remover** os anotadores identificados como problemáticos, contra controlo")
    a("   emparelhado — nulo, aqui.")
    a("")
    a("Ponderar e remover são as duas formas canónicas de usar uma estimativa de")
    a("confiabilidade para construir um alvo. Nenhuma delas rende neste dataset,")
    a("**medido no regime real de treino e com o confound de densidade controlado**.")
    a("Isso é mais forte do que um nulo só de ponderação: mostra que o limite não está")
    a("na forma de usar o θ, está na informação disponível.")
    a("")
    return "\n".join(L) + "\n"


def main():
    found = discover()
    a0chk = a0_reuse_check()
    checks = integrity(found, a0chk)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if any(not ok for _, ok, _ in checks):
        raise SystemExit("ABORTADO — verificação de integridade falhou.")

    ctx = target_context()
    print()
    print("alvo antes do treino:")
    print("  grade de valores distintos:",
          ", ".join(f"{ARMS[a_][0]}={ctx['grid'][ARMS[a_][0]]}" for a_ in ARM_ORDER))
    for arm in ("R1_rem_extremos", "R2_rem_controle"):
        v = ctx["vs_a0"][arm]
        print(f"  {arm} vs A0: afetados={v['n_affected']} "
              f"|Δ|medio={v['mean_abs_delta_affected']:.4f} "
              f"cruzam0.5={v['crossings_0_5']}")
    p12 = ctx["r1_vs_r2"]
    print(f"  R1 vs R2 (directo): afetados={p12['n_affected']} "
          f"|Δ|medio_todos={p12['mean_abs_delta_all']:.4f} "
          f"|Δ|medio_afetados={p12['mean_abs_delta_affected']:.4f} "
          f"cruzam0.5={p12['crossings_0_5']}")

    curves = {k: [r["eval_loss"] for r in v["evals"]] for k, v in found.items()}
    res = compute(found)

    print()
    for arm in ARM_ORDER:
        for s in SEEDS:
            i, ep, lo = best_epoch(found[(arm, s)]["evals"])
            print(f"  {ARMS[arm][0]:16s} seed {s:>4}: E={ep:.0f} eval_loss={lo:.5f}")
    print()
    print(f"DECISIVA — {DECISIVE}, Task A ICM-Soft-Norm soft, época E:")
    pr = res["best"][DECISIVE][PRIMARY]
    for s in SEEDS:
        d = pr["per_seed"][s]
        print(f"  seed {s:>4}: R1={d['a']:.4f} R2={d['b']:.4f} Δ={d['delta']:+.4f}")
    st = pr["stats"]
    print(f"  média Δ={st['mean']:+.4f} sd={st['sd']:.4f} t={st['t']:.2f} "
          f"p={pr['p']:.3f} d_z={st['d_z']:.2f} n80={st['n_for_80']}")
    print()
    for lbl, _a, _b, role in COMPARISONS:
        print(f"{lbl} ({role}) — 6 metricas na epoca E:")
        for mlbl, key in METRICS:
            r = res["best"][lbl][key]
            print(f"  {mlbl:24s} media={r['stats']['mean']:+.4f} "
                  f"sd={r['stats']['sd']:.4f} d_z={r['stats']['d_z']:+.2f} "
                  f"p={r['p']:.3f}")
        print()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(found, checks, res, curves, ctx, a0chk))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "criterion": {
                "decisive": f"{DECISIVE} @ {PRIMARY} @ argmin(eval_loss)",
                "context_only": [c[0] for c in COMPARISONS if c[3] != "DECISIVA"],
                "runs_finished": "2026-09-05", "criterion_fixed": "2026-09-06",
            },
            "runs": {f"{ARMS[a_][0]}_seed{s}": os.path.basename(found[(a_, s)]["path"])
                     for a_ in ARM_ORDER for s in SEEDS},
            "best_epoch": {f"{ARMS[a_][0]}_seed{s}": best_epoch(found[(a_, s)]["evals"])[1]
                           for a_ in ARM_ORDER for s in SEEDS},
            "eval_loss_curves": {f"{ARMS[a_][0]}_seed{s}": curves[(a_, s)]
                                 for a_ in ARM_ORDER for s in SEEDS},
            "a0_reuse_check": a0chk,
            "target_context": ctx,
            "results": res,
        }, f, indent=2, ensure_ascii=False)
    print(f"escrito: {REPORT_PATH}\n         {JSON_PATH}")


if __name__ == "__main__":
    main()
