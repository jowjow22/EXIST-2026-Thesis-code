"""ablation/report_inj.py

Gera `ablation/results/summary_INJ.md` a partir de:
  - `ablation/results/raw_results.csv`      (métricas por config × seed)
  - `ablation/diagnostics/task_c_collapse_<config>_seed<seed>.json`

Não treina nada e não recalcula métrica nenhuma -- só lê, agrega e responde às
duas perguntas da ABL-INJ. Pode rodar quantas vezes quiser.

Uso:
    python report_inj.py
"""

import os
import sys
import json
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEEDS, ABL_INJ_INJECTION  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV_PATH = os.path.join(HERE, "results", "raw_results.csv")
DIAGNOSTICS_DIR = os.path.join(HERE, "diagnostics")
OUT_PATH = os.path.join(HERE, "results", "summary_INJ.md")

CONFIG_ORDER = [c.name for c in ABL_INJ_INJECTION]

# Task C tem duas métricas hard: threshold fixo 0.5 (_hard_default) e threshold
# ajustado (_hard). As duas entram na tabela -- uma head colapsada pode parecer
# aceitável no ajustado e péssima no default.
REPORT_METRICS = [
    "task_a_icmsoftnorm_soft",
    "task_b_icmsoftnorm_soft",
    "task_c_icmsoftnorm_soft",
    "task_a_icmnorm_hard",
    "task_b_icmnorm_hard",
    "task_c_icmnorm_hard",
    "task_c_icmnorm_hard_default",
]

# Referência de cada feature_set: o modo "current" é a âncora de comparação.
REFERENCE_OF = {
    "base": "inj_current_base",
    "extended": "inj_current_extended",
}


def _feature_set_of(config_name):
    return "extended" if config_name.endswith("_extended") else "base"


def _load_all_rows():
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} não existe -- rode `python runner.py` antes."
        )
    return pd.read_csv(RAW_CSV_PATH)


def _fingerprint_check_section(all_rows):
    """REGRA da PARTE 1: `demo_injection_mode="current"` tem de reproduzir a
    baseline bit-a-bit.

    A verificação é uma comparação de `fingerprint_init` entre a linha
    `inj_current_base` e a linha `baseline` da MESMA seed -- as duas configs
    diferem só em `name` e nos campos que não tocam a arquitetura
    (`run_collapse_diagnostics`), então os pesos iniciais têm de ser idênticos.
    Só vale se as duas linhas vieram da MESMA GPU/versões de lib; fingerprints
    de máquinas diferentes não são comparáveis (ver CLAUDE.md).

    `renorm` e `head_inject` NÃO são checados: o primeiro acrescenta duas chaves
    ao state_dict, o segundo muda o shape das heads -- fingerprint diferente é
    esperado por construção, não regressão.
    """
    if "fingerprint_init" not in all_rows.columns:
        return "_Coluna `fingerprint_init` ausente no CSV -- verificação impossível._"

    lines = ["| seed | `baseline` | `inj_current_base` | resultado |",
             "|---|---|---|---|"]
    verdicts = []
    for seed in SEEDS:
        baseline_rows = all_rows[(all_rows["config_name"] == "baseline") & (all_rows["seed"] == seed)]
        current_rows = all_rows[(all_rows["config_name"] == "inj_current_base") & (all_rows["seed"] == seed)]
        if baseline_rows.empty or current_rows.empty:
            lines.append(f"| {seed} | {'—' if baseline_rows.empty else 'ok'} | "
                         f"{'—' if current_rows.empty else 'ok'} | n/d (falta uma das runs) |")
            verdicts.append(None)
            continue
        baseline_fp = str(baseline_rows.iloc[0]["fingerprint_init"])
        current_fp = str(current_rows.iloc[0]["fingerprint_init"])
        match = baseline_fp == current_fp
        verdicts.append(match)
        lines.append(f"| {seed} | `{baseline_fp}` | `{current_fp}` | "
                     f"{'**PASS**' if match else '**FAIL**'} |")

    measured = [v for v in verdicts if v is not None]
    if not measured:
        conclusion = ("\n**Inconclusivo:** nenhuma seed tem as duas linhas no CSV.")
    elif all(measured):
        conclusion = (f"\n**PASS ({len(measured)}/{len(SEEDS)} seeds):** o modo `current` nasce "
                      "bit-a-bit igual à baseline -- é uma âncora de comparação válida.")
    else:
        conclusion = ("\n**FAIL:** o modo `current` NÃO reproduz a baseline. Algo mudou a "
                      "ordem de criação dos módulos (e portanto os draws do RNG). Investigar "
                      "antes de interpretar qualquer delta desta ablação.")
    return "\n".join(lines) + "\n" + conclusion


def _load_rows():
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} não existe -- rode `python runner.py` antes."
        )
    df = pd.read_csv(RAW_CSV_PATH)
    df = df[df["config_name"].isin(CONFIG_ORDER)].copy()
    if df.empty:
        raise ValueError(
            "nenhuma linha de ABL-INJ em raw_results.csv -- a suite ainda não rodou."
        )
    return df


def _fmt(value, decimals=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/d"
    return f"{value:.{decimals}f}"


def _mean_std_table(df, metrics):
    lines = ["| config | " + " | ".join(metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for name in CONFIG_ORDER:
        rows = df[df["config_name"] == name]
        if rows.empty:
            lines.append(f"| `{name}` | " + " | ".join(["n/d"] * len(metrics)) + " |")
            continue
        cells = []
        for metric in metrics:
            if metric not in rows.columns:
                cells.append("n/d")
                continue
            values = rows[metric].astype(float)
            cells.append(f"{_fmt(values.mean())} ± {_fmt(values.std(ddof=1), 4)}")
        lines.append(f"| `{name}` (n={len(rows)}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _paired_deltas(df, metric):
    """Delta por seed de cada config contra a referência ("current") do MESMO
    feature_set. As seeds são pareadas (VAL_SPLIT_SEED fixo, mesma seed de init),
    então a comparação seed-a-seed é legítima -- e o "sinal consistente 3/3" é o
    critério de efeito real, não a diferença das médias.
    """
    results = {}
    for name in CONFIG_ORDER:
        reference_name = REFERENCE_OF[_feature_set_of(name)]
        if name == reference_name:
            continue
        deltas = {}
        for seed in SEEDS:
            variant = df[(df["config_name"] == name) & (df["seed"] == seed)]
            reference = df[(df["config_name"] == reference_name) & (df["seed"] == seed)]
            if variant.empty or reference.empty or metric not in df.columns:
                continue
            deltas[seed] = float(variant.iloc[0][metric]) - float(reference.iloc[0][metric])
        results[name] = (reference_name, deltas)
    return results


def _delta_table(df, metric):
    lines = [f"**{metric}** — delta contra o `current` do mesmo feature_set (seeds pareadas)",
             "",
             "| config | vs | " + " | ".join(f"seed {s}" for s in SEEDS) + " | média | sinal consistente |",
             "|---|---|" + "---|" * (len(SEEDS) + 2)]
    for name, (reference_name, deltas) in _paired_deltas(df, metric).items():
        if not deltas:
            continue
        values = [deltas.get(s) for s in SEEDS]
        present = [v for v in values if v is not None]
        signs = {np.sign(v) for v in present if v != 0.0}
        consistent = (len(signs) == 1 and len(present) == len(SEEDS))
        if consistent:
            mark = "sim (%d/%d %s)" % (len(present), len(SEEDS),
                                       "+" if present[0] > 0 else "−")
        elif len(present) < len(SEEDS):
            mark = f"não ({len(present)}/{len(SEEDS)} seeds medidas)"
        else:
            n_positive = sum(1 for v in present if v > 0)
            mark = f"não (sinais mistos: {n_positive}+ / {len(present) - n_positive}−)"
        cells = " | ".join("n/d" if v is None else f"{v:+.4f}" for v in values)
        lines.append(f"| `{name}` | `{reference_name}` | {cells} | "
                     f"{_fmt(float(np.mean(present)) if present else float('nan'))} | {mark} |")
    return "\n".join(lines)


def _consolidated_delta_table(df):
    """Matriz delta média × métrica para TODAS as configs (base e extended),
    com marca de sinal consistente 3/3. É a visão que responde (a) sem ficar
    restrita ao feature_set "base"."""
    def short(metric):
        return (metric.replace("task_", "")
                      .replace("icmsoftnorm_", "")
                      .replace("icmnorm_", ""))

    lines = ["| config | " + " | ".join(short(m) for m in REPORT_METRICS) + " | métricas 3/3 + |",
             "|---|" + "---|" * (len(REPORT_METRICS) + 1)]
    for name in CONFIG_ORDER:
        cells, n_consistent_positive = [], 0
        for metric in REPORT_METRICS:
            _, deltas = _paired_deltas(df, metric).get(name, (None, {}))
            if not deltas:
                cells.append("—")
                continue
            values = [deltas[s] for s in SEEDS if s in deltas]
            consistent = len({np.sign(v) for v in values}) == 1 and len(values) == len(SEEDS)
            if consistent and values[0] > 0:
                n_consistent_positive += 1
            cells.append(f"{np.mean(values):+.4f}" + ("\\*" if consistent else ""))
        anchor = all(c == "—" for c in cells)
        lines.append(f"| `{name}` | " + " | ".join(cells) + " | "
                     + ("âncora" if anchor else f"**{n_consistent_positive}/{len(REPORT_METRICS)}**") + " |")
    lines.append("")
    lines.append("`\\*` = mesmo sinal nas 3 seeds pareadas. Linhas `inj_current_*` são as "
                 "âncoras (delta contra si mesmas).")
    return "\n".join(lines)


def _caveats_section():
    return "\n".join([
        "1. **`head_inject` não é uma realocação neutra do sinal demográfico.** As heads "
        "passam de 1024 para 1536 `in_features` -- são 512 dimensões aprendidas a mais, "
        "ou seja, mais capacidade, não só demografia noutro lugar. Esta suite **não tem** "
        "o braço de controle que separa as duas coisas (mesma concatenação com um slot de "
        "zeros, ou com o `demo_vec` embaralhado). Enquanto esse controle não existir, o "
        "ganho do `head_inject` **não pode** ser atribuído à demografia na tese.",
        "",
        "2. **A premissa de causa raiz única não se sustentou.** A tarefa supunha que a "
        "distorção de escala (injeção depois do LayerNorm) explicava os dois sintomas. O "
        "`renorm` corrige exatamente essa distorção e **não move nada** -- logo a escala "
        "não é o mecanismo que suprime a demografia.",
        "",
        "3. **O colapso da Task C não reproduziu.** Duas leituras continuam vivas e esta "
        "suite não as separa: (i) o colapso era artefato do framework antigo (braços com "
        "pesos partilhados, ARCH-20; o diretório de predições foi apagado e não dá para "
        "reconferir), ou (ii) o colapso é real mas não aparece sob o regime de treino "
        "corrigido. Reportar como observação não replicada, nunca como refutação.",
    ])


def _load_diagnostics():
    """{config_name: {seed: dict}} a partir dos JSONs persistidos."""
    diagnostics = {}
    for path in sorted(glob.glob(os.path.join(DIAGNOSTICS_DIR, "task_c_collapse_*.json"))):
        with open(path) as f:
            payload = json.load(f)
        diagnostics.setdefault(payload.get("config_name"), {})[payload.get("seed")] = payload
    return diagnostics


def _collapse_table(diagnostics):
    lines = ["| config | var inter-item média | \\|média − taxa base\\| | cosseno médio | seeds colapsadas | veredito |",
             "|---|---|---|---|---|---|"]
    for name in CONFIG_ORDER:
        per_seed = diagnostics.get(name, {})
        if not per_seed:
            lines.append(f"| `{name}` | n/d | n/d | n/d | n/d | diagnóstico ausente |")
            continue
        variances = [d["mean_inter_item_variance"] for d in per_seed.values()]
        diffs = [d["mean_abs_diff_vs_baserate"] for d in per_seed.values()]
        cosines = [d["mean_pairwise_cosine_between_items"] for d in per_seed.values()]
        collapsed = [d["collapsed"] for d in per_seed.values()]
        n_collapsed = sum(1 for c in collapsed if c is True)
        if n_collapsed == len(collapsed):
            verdict = "[COLAPSO CONFIRMADO] prevê a marginal, ignora o input"
        elif n_collapsed == 0:
            verdict = "[OK] discrimina entre itens"
        else:
            verdict = "[MISTO] colapsa em parte das seeds"
        lines.append(
            f"| `{name}` | {np.mean(variances):.3e} | {np.mean(diffs):.4f} | "
            f"{np.mean(cosines):.4f} | {n_collapsed}/{len(collapsed)} | {verdict} |"
        )
    return "\n".join(lines)


def _answer_question_a(df):
    """(a) A correção da injeção faz a demografia ter efeito mensurável vs current?
    Comparação restrita ao feature_set "base"."""
    lines = []
    for metric in REPORT_METRICS:
        deltas = _paired_deltas(df, metric)
        for name in ("inj_renorm_base", "inj_headinject_base"):
            reference_name, per_seed = deltas.get(name, (None, {}))
            if not per_seed:
                continue
            values = [per_seed[s] for s in SEEDS if s in per_seed]
            signs = {np.sign(v) for v in values if v != 0.0}
            consistent = len(signs) == 1 and len(values) == len(SEEDS)
            lines.append(
                f"- `{name}` vs `{reference_name}` em **{metric}**: "
                f"média {np.mean(values):+.4f}, "
                f"{'sinal consistente nas 3 seeds' if consistent else 'sinal NÃO consistente'}"
            )
    if not lines:
        return "_Sem dados suficientes para responder (faltam runs de `*_base`)._"
    header = (
        "Critério: um efeito só conta como mensurável se o delta contra "
        "`inj_current_base` tiver o **mesmo sinal nas 3 seeds pareadas**. Diferença "
        "de médias sozinha não basta -- é o mesmo erro que a tabela de ablação "
        "original cometeu (SIMP-01/OPT-06).\n"
    )
    return header + "\n".join(lines)


def _answer_question_b(diagnostics):
    """(b) O colapso da Task C no extended some com a injeção corrigida?"""
    def summarize(name):
        per_seed = diagnostics.get(name, {})
        if not per_seed:
            return f"- `{name}`: diagnóstico ausente."
        n_collapsed = sum(1 for d in per_seed.values() if d["collapsed"] is True)
        variance = np.mean([d["mean_inter_item_variance"] for d in per_seed.values()])
        diff = np.mean([d["mean_abs_diff_vs_baserate"] for d in per_seed.values()])
        return (f"- `{name}`: colapsou em {n_collapsed}/{len(per_seed)} seeds "
                f"(var inter-item {variance:.3e}, |média − taxa base| {diff:.4f}).")

    lines = [summarize(n) for n in
             ("inj_current_extended", "inj_renorm_extended", "inj_headinject_extended")]

    current = diagnostics.get("inj_current_extended", {})
    if current:
        current_collapsed = sum(1 for d in current.values() if d["collapsed"] is True)
        fixed_collapsed = sum(
            1 for name in ("inj_renorm_extended", "inj_headinject_extended")
            for d in diagnostics.get(name, {}).values() if d["collapsed"] is True
        )
        if current_collapsed == 0:
            lines.append("\n**Conclusão:** o colapso NÃO se reproduziu nem no `current` do "
                         "extended -- a premissa da tarefa não se sustenta nesta suite, e a "
                         "comparação com os modos corrigidos fica sem base.")
        elif fixed_collapsed == 0:
            lines.append("\n**Conclusão:** o colapso aparece no `current` e desaparece nos dois "
                         "modos corrigidos -- consistente com a hipótese de que a causa é o "
                         "ponto de injeção, não o demo_vec expandido.")
        else:
            lines.append("\n**Conclusão:** o colapso sobrevive a pelo menos um modo corrigido -- "
                         "a injeção não é a única causa.")
    return "\n".join(lines)


def main():
    all_rows = _load_all_rows()
    df = _load_rows()
    diagnostics = _load_diagnostics()

    parts = [
        "# ABL-INJ — mecanismo de injeção da demografia × demo_feature_set",
        "",
        "Gerado por `ablation/report_inj.py` a partir de `results/raw_results.csv` e dos "
        "JSONs de `ablation/diagnostics/`. Todas as configs: `use_vision=True`, "
        f"`use_demographics=True`, seeds {SEEDS}.",
        "",
        "## 0. Verificação do modo-âncora (`current` == baseline, bit-a-bit)",
        "",
        _fingerprint_check_section(all_rows),
        "",
        "## 1. Métricas por config (média ± desvio-padrão sobre seeds)",
        "",
        _mean_std_table(df, REPORT_METRICS),
        "",
    ]
    parts += [
        "### 1.1 Visão consolidada — delta médio contra o `current` do mesmo feature_set",
        "",
        _consolidated_delta_table(df),
        "",
        "### 1.2 Deltas pareados, seed a seed",
        "",
    ]
    for metric in REPORT_METRICS:
        parts += [_delta_table(df, metric), ""]

    parts += [
        "## 2. Diagnóstico de colapso da head de Task C",
        "",
        "Colapso = variância inter-item média < 1e-04 **e** "
        "|média − taxa base do treino| média < 0.02.",
        "",
        _collapse_table(diagnostics),
        "",
        "## 3. Respostas",
        "",
        "### (a) A correção da injeção faz a demografia ter efeito mensurável vs `current`?",
        "",
        _answer_question_a(df),
        "",
        "### (b) O colapso da Task C no `extended` desaparece com a injeção corrigida?",
        "",
        _answer_question_b(diagnostics),
        "",
        "## 4. Ressalvas — o que estes números NÃO autorizam a afirmar",
        "",
        _caveats_section(),
        "",
    ]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(parts))
    print(f"relatório escrito em {OUT_PATH}")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
