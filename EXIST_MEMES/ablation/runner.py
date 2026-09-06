"""ablation/runner.py

Orquestra FASE 0 (baseline × 3 seeds) e FASE 1 (ABLATION_SUITE), gravando
resultados incrementalmente em `results/raw_results.csv` e agregando por
ablação em `results/summary_<ablation_id>.csv`.

Uso:
    python runner.py            # pede confirmação se a estimativa de tempo for alta
    python runner.py --yes      # não pede confirmação (para rodar sem supervisão na VM)
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    SEEDS, BASELINE, ABLATION_SUITE, ABLATION_NAME_PREFIXES,
    get_baseline_config, config_is_baseline_equivalent,
)
from train_one import run_single  # noqa: E402
from dataclasses import replace

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW_CSV_PATH = os.path.join(RESULTS_DIR, "raw_results.csv")
BASELINE_JSON_PATH = os.path.join(RESULTS_DIR, "..", "baseline_config.json")

KEY_METRICS = [
    "task_a_icmsoftnorm_soft",
    "task_b_icmsoftnorm_soft",
    "task_c_icmsoftnorm_soft",
    "task_a_icmnorm_hard",
    "task_b_icmnorm_hard",
    "task_c_icmnorm_hard",
    # Task C tem DUAS métricas hard (memes_core._metrics_task_c): "_hard_default"
    # é threshold fixo 0.5, "_hard" é o threshold ajustado pela busca gulosa.
    # O relatório da ABL-INJ compara as duas -- uma head colapsada pode parecer
    # decente no threshold ajustado e péssima no default.
    "task_c_icmnorm_hard_default",
]

# Estimativa grosseira de horas por treino de 30 épocas nesta VM -- ajustar
# com base na primeira run real observada (imprime timing real ao final de
# cada run; use esse número para corrigir esta constante em runs futuras).
ESTIMATED_HOURS_PER_TRAINING = float(os.environ.get("MEMES_HOURS_PER_TRAINING", "3.0"))
HOURS_WARNING_THRESHOLD = 72.0  # acima disso, pede confirmação explícita


def append_result_row(row, csv_path=RAW_CSV_PATH):
    """Acrescenta 1 linha ao CSV, reescrevendo o arquivo inteiro via pandas para
    manter o schema consistente (colunas novas viram NaN nas linhas antigas, e
    vice-versa) -- métricas ausentes (PyEvALL falhou / task sem voto válido)
    já chegam como chaves faltando no dict e viram NaN automaticamente aqui."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)


def _read_raw_results():
    if not os.path.exists(RAW_CSV_PATH):
        return pd.DataFrame()
    return pd.read_csv(RAW_CSV_PATH)


def _find_reusable_baseline_row(config, seed):
    """Se `config` (ignorando name/seed) é idêntico à BASELINE e já existe uma
    linha para essa seed no CSV com config_name == 'baseline', devolve essa
    linha (como dict) para reaproveitar em vez de retreinar.

    Não recomputa o fingerprint aqui: como os hiperparâmetros são bit-a-bit
    idênticos à baseline (config_is_baseline_equivalent), o fingerprint_init
    já registrado para essa seed É o fingerprint correto -- recarregar
    XLM-R+CLIP só para reconferir seria puro custo sem novo sinal. Isso só
    vale para variantes "config_A" (arquitetura idêntica); ABL-2/4/5/6
    mudam a arquitetura e por isso NUNCA reaproveitam (ver
    config_is_baseline_equivalent).
    """
    if not config_is_baseline_equivalent(config):
        return None
    df = _read_raw_results()
    if df.empty:
        return None
    match = df[(df["config_name"] == "baseline") & (df["seed"] == seed)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def run_phase_0():
    print("\n" + "#" * 70)
    print("# FASE 0 -- baseline (3 seeds)")
    print("#" * 70)

    baseline = get_baseline_config(name="baseline")
    with open(BASELINE_JSON_PATH, "w") as f:
        json.dump(
            {k: v for k, v in baseline.__dict__.items()}, f, indent=2,
        )
    print(f"BASELINE_CONFIG salva em {BASELINE_JSON_PATH}:")
    print(json.dumps(baseline.__dict__, indent=2))

    df_existing = _read_raw_results()
    for seed in SEEDS:
        already = (not df_existing.empty and
                   ((df_existing["config_name"] == "baseline") & (df_existing["seed"] == seed)).any())
        if already:
            print(f"[FASE 0] baseline seed={seed} já existe em {RAW_CSV_PATH}, pulando.")
            continue
        cfg = replace(baseline, seed=seed)
        row = run_single(cfg)
        append_result_row(row)


def run_phase_1():
    print("\n" + "#" * 70)
    print("# FASE 1 -- ablações")
    print("#" * 70)

    for ablation_id, variants in ABLATION_SUITE:
        print(f"\n{'-' * 70}")
        print(f"-- {ablation_id} --")
        print(f"{'-' * 70}")

        for variant in variants:
            for seed in SEEDS:
                cfg = replace(variant, seed=seed)

                df_existing = _read_raw_results()
                already = (not df_existing.empty and
                           ((df_existing["config_name"] == cfg.name) & (df_existing["seed"] == seed)).any())
                if already:
                    print(f"[{ablation_id}] {cfg.name} seed={seed} já existe, pulando.")
                    continue

                reused = _find_reusable_baseline_row(variant, seed)
                if reused is not None:
                    print(f"[{ablation_id}] {cfg.name} seed={seed}: config == baseline, "
                          f"reaproveitando resultado da FASE 0 (fingerprint_init={reused.get('fingerprint_init')}).")
                    row = dict(reused)
                    row["config_name"] = cfg.name
                    row["seed"] = seed
                    append_result_row(row)
                    continue

                row = run_single(cfg)
                append_result_row(row)

        summarize_ablation(ablation_id)


def summarize_ablation(ablation_id):
    df = _read_raw_results()
    # ABL-1..6 usam "abl<N>_config_"; ABL-INJ usa "inj_" (ver ABLATION_NAME_PREFIXES).
    prefix = ABLATION_NAME_PREFIXES.get(ablation_id)
    if prefix is None:
        raise KeyError(
            f"{ablation_id} não tem prefixo em ABLATION_NAME_PREFIXES -- "
            f"acrescente-o em config.py antes de agregar."
        )
    variant_rows = df[df["config_name"].str.startswith(prefix)]
    if variant_rows.empty:
        print(f"[{ablation_id}] sem resultados para agregar ainda.")
        return

    available_metrics = [m for m in KEY_METRICS if m in variant_rows.columns]
    agg = variant_rows.groupby("config_name")[available_metrics].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("config_name")

    summary_path = os.path.join(RESULTS_DIR, f"summary_{ablation_id}.csv")
    agg.to_csv(summary_path, index=False)
    print(f"[{ablation_id}] resumo salvo em {summary_path}")
    print(agg.to_string(index=False))

    selection_metric = "task_a_icmsoftnorm_soft"
    if f"{selection_metric}_mean" not in agg.columns:
        print(f"[{ablation_id}] métrica de seleção '{selection_metric}' ausente, sem winner.")
        return

    ranked = agg.sort_values(f"{selection_metric}_mean", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    print(f"[{ablation_id}] vencedor em {selection_metric}: {best['config_name']} "
          f"({best[f'{selection_metric}_mean']:.4f} ± {best[f'{selection_metric}_std']:.4f})")

    if len(ranked) > 1:
        runner_up = ranked.iloc[1]
        gap = best[f"{selection_metric}_mean"] - runner_up[f"{selection_metric}_mean"]
        combined_std = np.nanmax([best[f"{selection_metric}_std"], runner_up[f"{selection_metric}_std"], 1e-9])
        if gap > combined_std:
            print(f"[{ablation_id}] diferença ({gap:.4f}) excede 1 std ({combined_std:.4f}) "
                  f"-- sinal de efeito real, não só ruído de seed.")
        else:
            print(f"[{ablation_id}] diferença ({gap:.4f}) NÃO excede 1 std ({combined_std:.4f}) "
                  f"-- pode ser ruído de seed, não conclusivo.")


def estimate_total_runs():
    phase_0_runs = len(SEEDS)
    phase_1_new_runs = 0
    phase_1_reused_runs = 0
    for _, variants in ABLATION_SUITE:
        for variant in variants:
            for _ in SEEDS:
                if config_is_baseline_equivalent(variant):
                    phase_1_reused_runs += 1
                else:
                    phase_1_new_runs += 1
    return phase_0_runs, phase_1_new_runs, phase_1_reused_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="não pede confirmação mesmo se a estimativa de tempo for alta")
    args = parser.parse_args()

    phase_0_runs, phase_1_new_runs, phase_1_reused_runs = estimate_total_runs()
    total_new_trainings = phase_0_runs + phase_1_new_runs
    estimated_hours = total_new_trainings * ESTIMATED_HOURS_PER_TRAINING

    print("=" * 70)
    print("ESTIMATIVA DE EXECUÇÃO")
    print("=" * 70)
    print(f"FASE 0 (baseline):        {phase_0_runs} treinos")
    print(f"FASE 1 (novos):           {phase_1_new_runs} treinos")
    print(f"FASE 1 (reaproveitados):  {phase_1_reused_runs} (config_A == baseline, sem retreino)")
    print(f"Total de treinos novos:   {total_new_trainings}")
    print(f"Estimativa: {total_new_trainings} × {ESTIMATED_HOURS_PER_TRAINING}h/treino "
          f"= {estimated_hours:.1f}h (ajuste MEMES_HOURS_PER_TRAINING com o tempo real observado)")
    print("=" * 70)

    if estimated_hours > HOURS_WARNING_THRESHOLD and not args.yes:
        answer = input(f"Estimativa acima de {HOURS_WARNING_THRESHOLD}h. Prosseguir? [s/N] ")
        if answer.strip().lower() not in ("s", "sim", "y", "yes"):
            print("Abortado pelo usuário.")
            return

    run_phase_0()
    run_phase_1()

    print("\n" + "=" * 70)
    print("Suite de ablação concluída.")
    print(f"Resultados brutos: {RAW_CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
