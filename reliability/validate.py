"""Harness de validação sintética do estimador `NutmegEM`.

Régua desta etapa. Regra do projeto: nenhum componente corre nos dados reais do
EXIST antes de passar aqui.

Três coisas:
 1. `evaluate_recovery`  — correlação entre theta verdadeiro e estimado.
 2. `run_recovery_grid`  — grade (divisiveness x spam x seed) com DUAS variantes:
    subpop-aware e single-truth (baseline conceptual tipo-MACE = a MESMA classe
    com K=1). O contraste entre as duas curvas é o argumento de tese.
 3. `test_minority_preservation` — o teste que diz se o método rompe o LeWiDi:
    um subgrupo minoritário CONSISTENTE mas discordante tem de receber theta ALTO.

CLI:
    python -m reliability.validate --grid
    python -m reliability.validate --minority
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .nutmeg_em import NutmegEM
from .perann_reliability import compute_competence
from .synthetic import generate_difficulty_synthetic, generate_synthetic

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
GRID_CSV = os.path.join(OUTPUT_DIR, "recovery_grid.csv")
MINORITY_JSON = os.path.join(OUTPUT_DIR, "minority_preservation.json")

DEFAULT_SEEDS = [42, 7, 2025]
DEFAULT_DIVISIVENESS = [0.0, 0.15, 0.30, 0.50, 0.70]
DEFAULT_SPAM = [0.15, 0.30, 0.45]

# --- Limiares PRÉ-REGISTADOS do teste do SPAMMER DE VIÉS CONSTANTE ----------
# Fixados antes de qualquer execução. Não ajustar depois de ver o resultado.
BIAS_SPAMMER_THETA_CEILING = 0.60   # theta_hat médio dos spammers enviesados tem de ficar ABAIXO
BIAS_MIN_GAP_VS_RELIABLE = 0.20     # theta_hat(fiáveis) - theta_hat(enviesados)
BIAS_MAX_GAP_VS_UNIFORM = 0.15      # |theta_hat(enviesados) - theta_hat(spam uniforme)|
                                    # (os dois tipos de ruído devem ser tratados de forma parecida)

# --- Limiares PRÉ-REGISTADOS do teste de preservação de minoria --------------
# Fixados antes de qualquer execução. Não ajustar depois de ver o resultado.
MIN_PRESERVATION_THETA_FLOOR = 0.70   # theta_hat médio da minoria consistente
MAX_MAJORITY_MINORITY_GAP = 0.10      # theta_hat(maioria fiável) - theta_hat(minoria)
MIN_SPAMMER_SEPARATION = 0.25         # theta_hat(minoria) - theta_hat(spammers)


def evaluate_recovery(true_theta, estimated_theta) -> dict:
    """Correlação entre theta verdadeiro e estimado.

    Parameters
    ----------
    true_theta, estimated_theta : array-like (M,)

    Returns
    -------
    dict com pearson_r, pearson_p, spearman_rho, spearman_p.
    """
    t = np.asarray(true_theta, dtype=float)
    e = np.asarray(estimated_theta, dtype=float)
    assert t.shape == e.shape, f"{t.shape} != {e.shape}"
    r, rp = pearsonr(t, e)
    rho, sp = spearmanr(t, e)
    return {
        "pearson_r": float(r),
        "pearson_p": float(rp),
        "spearman_rho": float(rho),
        "spearman_p": float(sp),
    }


def _fit_both(annotations, subpop, n_labels=2):
    """Ajusta as duas variantes e devolve (theta_subpop_aware, theta_single_truth)."""
    aware = NutmegEM(n_labels=n_labels, monotonic_check=False).fit(annotations, subpop)
    single = NutmegEM(n_labels=n_labels, monotonic_check=False).fit(annotations, np.zeros_like(subpop))
    return aware, single


def run_recovery_grid(
    divisiveness_rates=None,
    spam_rates=None,
    seeds=None,
    n_items: int = 800,
    n_annotators: int = 60,
    minority_frac: float = 0.25,
    annotations_per_item: int = 6,
    out_csv: str = GRID_CSV,
    verbose: bool = True,
):
    """Corre a grade sintética e grava `recovery_grid.csv`.

    Uma linha por (divisiveness, spam, seed, estimador). Devolve a lista de linhas.
    """
    divisiveness_rates = divisiveness_rates or DEFAULT_DIVISIVENESS
    spam_rates = spam_rates or DEFAULT_SPAM
    seeds = seeds or DEFAULT_SEEDS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for div in divisiveness_rates:
        for spam in spam_rates:
            for seed in seeds:
                ann, true_theta, subpop, _, _g = generate_synthetic(
                    n_items=n_items,
                    n_annotators=n_annotators,
                    minority_frac=minority_frac,
                    divisiveness_rate=div,
                    mean_spam_rate=spam,
                    annotations_per_item=annotations_per_item,
                    seed=seed,
                )
                aware, single = _fit_both(ann, subpop)
                for name, model in (("subpop_aware", aware), ("single_truth", single)):
                    m = evaluate_recovery(true_theta, model.theta_)
                    rows.append(
                        {
                            "divisiveness": div,
                            "spam_rate": spam,
                            "seed": seed,
                            "estimator": name,
                            "n_iter": model.n_iter_run_,
                            "converged": int(model.converged_),
                            **m,
                        }
                    )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if verbose:
        _print_grid_table(rows)
        print(f"\n[grid] {len(rows)} linhas -> {out_csv}")
    return rows


def _print_grid_table(rows):
    """Imprime Pearson r médio por divisiveness, lado a lado, e o delta."""
    divs = sorted({r["divisiveness"] for r in rows})
    print("\nPearson r médio (média sobre spam_rates x seeds)")
    print(f"{'divisiveness':>13} | {'subpop_aware':>12} | {'single_truth':>12} | {'delta':>7}")
    print("-" * 54)
    for d in divs:
        a = np.mean([r["pearson_r"] for r in rows
                     if r["divisiveness"] == d and r["estimator"] == "subpop_aware"])
        s = np.mean([r["pearson_r"] for r in rows
                     if r["divisiveness"] == d and r["estimator"] == "single_truth"])
        print(f"{d:>13.2f} | {a:>12.3f} | {s:>12.3f} | {a - s:>+7.3f}")
    ga = np.mean([r["pearson_r"] for r in rows if r["estimator"] == "subpop_aware"])
    gs = np.mean([r["pearson_r"] for r in rows if r["estimator"] == "single_truth"])
    print("-" * 54)
    print(f"{'GLOBAL':>13} | {ga:>12.3f} | {gs:>12.3f} | {ga - gs:>+7.3f}")
    print("(referência do paper NUTMEG r~0.81 vs MACE r~0.58 — ordem de grandeza)")


def test_minority_preservation(
    seeds=None,
    n_items: int = 800,
    n_annotators: int = 60,
    minority_frac: float = 0.25,
    divisiveness_rate: float = 0.70,
    mean_spam_rate: float = 0.35,
    minority_theta: float = 0.90,
    annotations_per_item: int = 6,
    verbose: bool = True,
):
    """Minoria CONSISTENTE mas discordante tem de receber theta ALTO.

    Constrói um subgrupo minoritário com theta fixo alto (`minority_theta`) que
    vota sempre a verdade da SUA subpop, discordante da maioria em 70% dos itens.
    Se o EM lhe der theta baixo, está a confundir discordância legítima com
    ruído — ou seja, rompe o LeWiDi.

    Critério PASS/FAIL pré-registado (constantes do módulo, fixadas antes da
    execução): floor absoluto, gap máximo vs maioria fiável, separação mínima
    vs spammers reais.
    """
    seeds = seeds or DEFAULT_SEEDS
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for seed in seeds:
        ann, true_theta, subpop, _, _g = generate_synthetic(
            n_items=n_items,
            n_annotators=n_annotators,
            minority_frac=minority_frac,
            divisiveness_rate=divisiveness_rate,
            mean_spam_rate=mean_spam_rate,
            annotations_per_item=annotations_per_item,
            minority_theta=minority_theta,
            seed=seed,
        )
        model = NutmegEM(monotonic_check=False).fit(ann, subpop)
        th = model.theta_
        is_min = subpop == 1
        rel_maj = (~is_min) & (true_theta >= 0.80)   # maioria fiável
        spammers = (~is_min) & (true_theta <= 0.50)  # ruído real

        rec = {
            "seed": seed,
            "theta_minority": float(th[is_min].mean()),
            "theta_reliable_majority": float(th[rel_maj].mean()) if rel_maj.any() else float("nan"),
            "theta_spammers": float(th[spammers].mean()) if spammers.any() else float("nan"),
            "n_minority": int(is_min.sum()),
            "n_reliable_majority": int(rel_maj.sum()),
            "n_spammers": int(spammers.sum()),
        }
        rec["gap_majority_minority"] = rec["theta_reliable_majority"] - rec["theta_minority"]
        rec["separation_vs_spammers"] = rec["theta_minority"] - rec["theta_spammers"]
        rec["pass"] = bool(
            rec["theta_minority"] >= MIN_PRESERVATION_THETA_FLOOR
            and rec["gap_majority_minority"] <= MAX_MAJORITY_MINORITY_GAP
            and rec["separation_vs_spammers"] >= MIN_SPAMMER_SEPARATION
        )
        results.append(rec)

    overall = all(r["pass"] for r in results)
    if verbose:
        print("\nTeste de preservação de minoria (limiares pré-registados: "
              f"floor>={MIN_PRESERVATION_THETA_FLOOR}, gap<={MAX_MAJORITY_MINORITY_GAP}, "
              f"sep>={MIN_SPAMMER_SEPARATION})")
        print(f"{'seed':>6} | {'θ minoria':>9} | {'θ maioria':>9} | {'θ spam':>7} | "
              f"{'gap':>6} | {'sep':>6} | veredito")
        print("-" * 72)
        for r in results:
            print(
                f"{r['seed']:>6} | {r['theta_minority']:>9.3f} | "
                f"{r['theta_reliable_majority']:>9.3f} | {r['theta_spammers']:>7.3f} | "
                f"{r['gap_majority_minority']:>+6.3f} | {r['separation_vs_spammers']:>+6.3f} | "
                f"{'PASS' if r['pass'] else 'FAIL'}"
            )
        print("-" * 72)
        print(f"VEREDITO GLOBAL: {'PASS' if overall else 'FAIL'}")

    import json
    with open(MINORITY_JSON, "w") as f:
        json.dump(
            {
                "thresholds": {
                    "MIN_PRESERVATION_THETA_FLOOR": MIN_PRESERVATION_THETA_FLOOR,
                    "MAX_MAJORITY_MINORITY_GAP": MAX_MAJORITY_MINORITY_GAP,
                    "MIN_SPAMMER_SEPARATION": MIN_SPAMMER_SEPARATION,
                },
                "config": {
                    "divisiveness_rate": divisiveness_rate,
                    "mean_spam_rate": mean_spam_rate,
                    "minority_theta": minority_theta,
                    "minority_frac": minority_frac,
                },
                "per_seed": results,
                "overall_pass": overall,
            },
            f,
            indent=2,
        )
    return results, overall


def test_constant_bias_spammer_detection(
    seeds=None,
    n_items: int = 800,
    n_annotators: int = 60,
    minority_frac: float = 0.25,
    divisiveness_rate: float = 0.30,
    mean_spam_rate: float = 0.30,
    uniform_spammer_frac: float = 0.20,
    uniform_spammer_theta: float = 0.25,
    biased_spammer_frac: float = 0.20,
    biased_spammer_label: int = 1,
    biased_spammer_theta: float = 0.25,
    annotations_per_item: int = 6,
    out_name: str | None = None,
    verbose: bool = True,
):
    """Fecha o ponto cego do Stage 2: spammer de VIÉS CONSTANTE (vota sempre YES).

    Três populações no mesmo dataset:
      (i)   fiáveis            — theta alto, votam a verdade da sua subpop
      (ii)  spam uniforme      — theta baixo, voto aleatório quando spammam
      (iii) spam de viés const — theta baixo, votam SEMPRE a mesma classe

    O teste de preservação de minoria não cobre (iii): o spam que injeta é
    uniforme. A suspeita é que o modo **subpop-aware** seja MAIS vulnerável, por
    o spammer enviesado poder formar a sua própria "subpopulação" e ganhar
    verdade própria. Por isso corre os DOIS modos e compara.

    PASS/FAIL por limiares pré-registados no topo do módulo.
    """
    seeds = seeds or DEFAULT_SEEDS
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for seed in seeds:
        ann, true_theta, subpop, _, groups = generate_synthetic(
            n_items=n_items,
            n_annotators=n_annotators,
            minority_frac=minority_frac,
            divisiveness_rate=divisiveness_rate,
            mean_spam_rate=mean_spam_rate,
            annotations_per_item=annotations_per_item,
            uniform_spammer_frac=uniform_spammer_frac,
            uniform_spammer_theta=uniform_spammer_theta,
            biased_spammer_frac=biased_spammer_frac,
            biased_spammer_label=biased_spammer_label,
            biased_spammer_theta=biased_spammer_theta,
            seed=seed,
        )
        is_bias = groups["biased_spammer"]
        uniform_spam = groups["uniform_spammer"]
        reliable = (~is_bias) & (~uniform_spam) & (true_theta >= 0.80)
        # `subpop_adversarial`: os spammers enviesados ficam TODOS numa subpop só
        # deles — é o que um clustering COMPORTAMENTAL produziria, já que votam
        # todos igual. É aqui que o ponto cego do Stage 2 pode aparecer: com
        # verdade própria, o cluster pode "ter razão" e receber theta alto.
        sp_adv = subpop.copy()
        sp_adv[is_bias] = int(subpop.max()) + 1
        for mode, sp in (
            ("subpop_aware", subpop),
            ("subpop_adversarial", sp_adv),
            ("single_truth", np.zeros_like(subpop)),
        ):
            th = NutmegEM(monotonic_check=False).fit(ann, sp).theta_
            rec = {
                "seed": seed,
                "mode": mode,
                "theta_biased_spammers": float(th[is_bias].mean()),
                "theta_uniform_spammers": float(th[uniform_spam].mean()) if uniform_spam.any() else float("nan"),
                "theta_reliable": float(th[reliable].mean()) if reliable.any() else float("nan"),
                "n_biased": int(is_bias.sum()),
                "n_uniform": int(uniform_spam.sum()),
                "n_reliable": int(reliable.sum()),
            }
            rec["gap_reliable_biased"] = rec["theta_reliable"] - rec["theta_biased_spammers"]
            rec["gap_biased_vs_uniform"] = abs(
                rec["theta_biased_spammers"] - rec["theta_uniform_spammers"]
            )
            # critérios reportados SEPARADAMENTE: se falhar, tem de se ver qual
            rec["c1_ceiling"] = bool(rec["theta_biased_spammers"] <= BIAS_SPAMMER_THETA_CEILING)
            rec["c2_gap_reliable"] = bool(rec["gap_reliable_biased"] >= BIAS_MIN_GAP_VS_RELIABLE)
            rec["c3_similar_to_uniform"] = bool(rec["gap_biased_vs_uniform"] <= BIAS_MAX_GAP_VS_UNIFORM)
            rec["pass"] = bool(rec["c1_ceiling"] and rec["c2_gap_reliable"] and rec["c3_similar_to_uniform"])
            rec["posthoc_theta_uniform_strict"] = rec["theta_uniform_spammers"]
            rec["posthoc_n_uniform_strict"] = int(uniform_spam.sum())
            rows.append(rec)

    by_mode = {
        m: all(r["pass"] for r in rows if r["mode"] == m)
        for m in ("subpop_aware", "subpop_adversarial", "single_truth")
    }
    if verbose:
        print(
            "\nTeste do spammer de viés constante (limiares pré-registados: "
            f"θ_enviesado<={BIAS_SPAMMER_THETA_CEILING}, "
            f"gap_fiável>={BIAS_MIN_GAP_VS_RELIABLE}, "
            f"|θ_env-θ_unif|<={BIAS_MAX_GAP_VS_UNIFORM})"
        )
        print(f"{'seed':>6} | {'modo':>18} | {'θ enviesado':>11} | {'θ unif':>7} | "
              f"{'θ fiável':>8} | {'gap':>6} | {'|Δ|':>5} | veredito")
        print("-" * 90)
        for r in rows:
            print(
                f"{r['seed']:>6} | {r['mode']:>18} | {r['theta_biased_spammers']:>11.3f} | "
                f"{r['theta_uniform_spammers']:>7.3f} | {r['theta_reliable']:>8.3f} | "
                f"{r['gap_reliable_biased']:>+6.3f} | {r['gap_biased_vs_uniform']:>5.3f} | "
                f"{'PASS' if r['pass'] else 'FAIL'}"
                + ("" if r["pass"] else "  (falhou: "
                   + ",".join(c for c, k in (("c1", "c1_ceiling"), ("c2", "c2_gap_reliable"),
                                             ("c3", "c3_similar_to_uniform")) if not r[k]) + ")")
            )
        print("-" * 90)
        print("c1 = θ_enviesado abaixo do teto · c2 = separado dos fiáveis · "
              "c3 = tratado como o spam uniforme")
        for m, ok in by_mode.items():
            print(f"VEREDITO {m}: {'PASS' if ok else 'FAIL'}")

    import json
    fname = out_name or f"bias_spammer_test_label{biased_spammer_label}.json"
    with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
        json.dump(
            {
                "thresholds": {
                    "BIAS_SPAMMER_THETA_CEILING": BIAS_SPAMMER_THETA_CEILING,
                    "BIAS_MIN_GAP_VS_RELIABLE": BIAS_MIN_GAP_VS_RELIABLE,
                    "BIAS_MAX_GAP_VS_UNIFORM": BIAS_MAX_GAP_VS_UNIFORM,
                },
                "config": {
                    "divisiveness_rate": divisiveness_rate,
                    "mean_spam_rate": mean_spam_rate,
                    "biased_spammer_frac": biased_spammer_frac,
                    "biased_spammer_label": biased_spammer_label,
                    "biased_spammer_theta": biased_spammer_theta,
                },
                "rows": rows,
                "overall_pass_by_mode": by_mode,
            },
            f,
            indent=2,
        )
    return rows, by_mode


def sweep_bias_spammer_fraction(fracs=(0.03, 0.05, 0.10, 0.20, 0.35, 0.50), seeds=None, verbose=True):
    """Stress-sweep: até que fração de spammers enviesados o método aguenta?

    Só o modo `subpop_adversarial` (o pior caso: os enviesados têm subpop própria).
    Diagnóstico secundário — NÃO altera o veredito pré-registado de
    `test_constant_bias_spammer_detection`.
    """
    seeds = seeds or DEFAULT_SEEDS
    out = []
    for frac in fracs:
        vals = []
        for seed in seeds:
            ann, true_theta, subpop, _, groups = generate_synthetic(
                n_items=800, n_annotators=60, minority_frac=0.25,
                divisiveness_rate=0.30, mean_spam_rate=0.30,
                uniform_spammer_frac=0.20, biased_spammer_frac=frac,
                biased_spammer_theta=0.25, seed=seed,
            )
            is_bias = groups["biased_spammer"]
            sp = subpop.copy()
            sp[is_bias] = int(subpop.max()) + 1
            th = NutmegEM(monotonic_check=False).fit(ann, sp).theta_
            vals.append(float(th[is_bias].mean()))
        rec = {
            "biased_frac": frac,
            "theta_biased_mean": float(np.mean(vals)),
            "theta_biased_per_seed": vals,
            "below_ceiling": bool(np.mean(vals) <= BIAS_SPAMMER_THETA_CEILING),
        }
        out.append(rec)
        if verbose:
            print(f"  frac={frac:.2f} -> θ̂ enviesados = {rec['theta_biased_mean']:.3f} "
                  f"({', '.join(f'{v:.3f}' for v in vals)}) "
                  f"{'ok' if rec['below_ceiling'] else '** ACIMA DO TETO **'}")
    return out


def test_determinism(seed: int = 42) -> bool:
    """Mesma seed -> mesmos theta (bit a bit via np.array_equal)."""
    ann, _, subpop, _, _g = generate_synthetic(seed=seed)
    a = NutmegEM(monotonic_check=False).fit(ann, subpop).theta_
    b = NutmegEM(monotonic_check=False).fit(ann, subpop).theta_
    ok = np.array_equal(a, b)
    print(f"[determinismo] seed={seed}: {'OK' if ok else 'FALHOU'} "
          f"(max|Δ|={np.abs(a - b).max():.3e})")
    return ok


# ===========================================================================
# ESTIMADOR POR-ANOTADOR COM DESCONTO DE DIFICULDADE (perann_reliability.py)
# Testes independentes dos do NutmegEM, que ficam intocados.
# ===========================================================================

PERANN_JSON = os.path.join(OUTPUT_DIR, "perann_synthetic_gate.json")

# --- Limiares PRÉ-REGISTADOS (fixados ANTES de qualquer execução) -----------
PERANN_RECOVERY_MIN_R = 0.60          # Pearson r (competência vs theta real), hard_frac=0.4
PERANN_DISCOUNT_MIN_ADVANTAGE = 0.05  # r(com desconto) - r(MACE ingénuo) em hard_frac >= 0.4
PERANN_MINORITY_HARD_FLOOR = 0.70     # braço "hard": competência da minoria consistente
PERANN_MINORITY_HARD_MAX_GAP = 0.10   # maioria fiável - minoria, no braço "hard"
PERANN_MINORITY_EASY_CEILING = 0.60   # braço "easy": competência TEM de cair abaixo disto
PERANN_MINORITY_EASY_MIN_GAP = 0.20   # maioria fiável - minoria, no braço "easy"
PERANN_BIAS_CEILING = 0.60            # spammer de viés constante
PERANN_BIAS_MIN_GAP = 0.20            # maioria fiável - spammer enviesado

# (discount, leave_one_out) -> nome
PERANN_CONFIGS = {
    "mace_naive": dict(discount=False, leave_one_out=False),
    "discount_raw": dict(discount=True, leave_one_out=False),   # literal da spec (primária)
    "discount_loo": dict(discount=True, leave_one_out=True),    # variante
}
PERANN_DISCOUNTED = ["discount_raw", "discount_loo"]

# Densidades: "confortável" (80 votos/anotador) e a REAL do EXIST (~27).
PERANN_DENSITIES = {
    "confortavel": dict(n_items=800, n_annotators=60, annotations_per_item=6),
    "exist": dict(n_items=4000, n_annotators=887, annotations_per_item=6),
}


def _perann_all_configs(annotations, M, N):
    return {name: compute_competence(annotations, n_annotators=M, n_items=N, **cfg)
            for name, cfg in PERANN_CONFIGS.items()}


def perann_recovery_grid(hard_fracs=(0.0, 0.2, 0.4, 0.6), seeds=None, verbose=True):
    """2.1 — recuperação de competência vs theta real, com e sem desconto."""
    seeds = seeds or DEFAULT_SEEDS
    rows = []
    for dens_name, dens in PERANN_DENSITIES.items():
        for hf in hard_fracs:
            acc = {k: [] for k in PERANN_CONFIGS}
            for seed in seeds:
                ann, th, is_hard, groups, info = generate_difficulty_synthetic(
                    hard_frac=hf, seed=seed, **dens)
                res = _perann_all_configs(ann, dens["n_annotators"], dens["n_items"])
                for k, r in res.items():
                    acc[k].append(evaluate_recovery(th, r["competence"])["pearson_r"])
            row = {"densidade": dens_name, "hard_frac": hf,
                   "votos_por_anotador": round(float(info["votes_per_annotator"].mean()), 1)}
            for k in PERANN_CONFIGS:
                row[f"r_{k}"] = float(np.mean(acc[k]))
            rows.append(row)
    if verbose:
        print("\n--- 2.1 Recuperação de competência (Pearson r, média de "
              f"{len(seeds)} seeds) ---")
        hdr = f"{'densidade':<12}{'hard':>6}{'votos/an':>10}"
        for k in PERANN_CONFIGS:
            hdr += f"{k:>15}"
        print(hdr)
        for r in rows:
            line = (f"{r['densidade']:<12}{r['hard_frac']:>6.2f}"
                    f"{r['votos_por_anotador']:>10.1f}")
            for k in PERANN_CONFIGS:
                line += f"{r[f'r_{k}']:>15.3f}"
            print(line)
    # --- portão -----------------------------------------------------------
    ref = [r for r in rows if r["densidade"] == "confortavel" and r["hard_frac"] == 0.4][0]
    verdicts = {}
    for k in PERANN_DISCOUNTED:
        c1 = ref[f"r_{k}"] >= PERANN_RECOVERY_MIN_R
        adv = [r[f"r_{k}"] - r["r_mace_naive"] for r in rows if r["hard_frac"] >= 0.4]
        c2 = min(adv) >= PERANN_DISCOUNT_MIN_ADVANTAGE
        verdicts[k] = {"pass": bool(c1 and c2), "r_ref": ref[f"r_{k}"],
                       "min_advantage": float(min(adv)),
                       "c1_r_min": bool(c1), "c2_advantage": bool(c2)}
        if verbose:
            print(f"[2.1 {k}] r(hard=0.4, confortável)={ref[f'r_{k}']:.3f} "
                  f"(>= {PERANN_RECOVERY_MIN_R}) -> {'ok' if c1 else 'FALHA'} | "
                  f"vantagem mínima sobre MACE em hard>=0.4: {min(adv):+.3f} "
                  f"(>= {PERANN_DISCOUNT_MIN_ADVANTAGE}) -> {'ok' if c2 else 'FALHA'} "
                  f"=> {'PASS' if c1 and c2 else 'FAIL'}")
    return rows, verdicts


def perann_minority_arms(seeds=None, minority_frac=0.25, minority_theta=0.95,
                         disagree_frac=0.30, hard_frac=0.4, density="confortavel",
                         verbose=True):
    """2.2 — os dois braços MATCHED da preservação de minoria.

    Braço "hard": minoria consistente discorda em terreno GENUINAMENTE ambíguo ->
        o desconto DEVE protegê-la (competência alta).
    Braço "easy": minoria consistente discorda em terreno de CONSENSO -> é ruído,
        o desconto NÃO a deve proteger (competência baixa).
    Tudo igual entre braços (fração, theta, nº de itens de discordância); só muda
    o terreno.
    """
    seeds = seeds or DEFAULT_SEEDS
    dens = PERANN_DENSITIES[density]
    out = {}
    for arm in ("hard", "easy"):
        per_cfg = {k: {"minority": [], "reliable": []} for k in PERANN_CONFIGS}
        for seed in seeds:
            ann, th, is_hard, groups, info = generate_difficulty_synthetic(
                hard_frac=hard_frac, minority_frac=minority_frac,
                minority_theta=minority_theta, minority_mode=arm,
                disagree_frac=disagree_frac, seed=seed, **dens)
            res = _perann_all_configs(ann, dens["n_annotators"], dens["n_items"])
            for k, r in res.items():
                c = r["competence"]
                per_cfg[k]["minority"].append(float(c[groups["minority"]].mean()))
                per_cfg[k]["reliable"].append(float(c[groups["reliable_majority"]].mean()))
        out[arm] = {k: {"minority_mean": float(np.mean(v["minority"])),
                        "reliable_mean": float(np.mean(v["reliable"])),
                        "minority_per_seed": v["minority"]}
                    for k, v in per_cfg.items()}
    # --- portão -----------------------------------------------------------
    verdicts = {}
    for k in PERANN_DISCOUNTED:
        h, e = out["hard"][k], out["easy"][k]
        gap_h = h["reliable_mean"] - h["minority_mean"]
        gap_e = e["reliable_mean"] - e["minority_mean"]
        c1 = h["minority_mean"] >= PERANN_MINORITY_HARD_FLOOR
        c2 = gap_h <= PERANN_MINORITY_HARD_MAX_GAP
        c3 = e["minority_mean"] <= PERANN_MINORITY_EASY_CEILING
        c4 = gap_e >= PERANN_MINORITY_EASY_MIN_GAP
        verdicts[k] = {"pass": bool(c1 and c2 and c3 and c4),
                       "hard_minority": h["minority_mean"], "hard_gap": gap_h,
                       "easy_minority": e["minority_mean"], "easy_gap": gap_e,
                       "c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "c4": bool(c4)}
    if verbose:
        print("\n--- 2.2 Preservação de minoria: dois braços MATCHED ---")
        print(f"{'config':<15}{'hard:minoria':>14}{'hard:fiáveis':>14}"
              f"{'easy:minoria':>14}{'easy:fiáveis':>14}")
        for k in PERANN_CONFIGS:
            print(f"{k:<15}{out['hard'][k]['minority_mean']:>14.3f}"
                  f"{out['hard'][k]['reliable_mean']:>14.3f}"
                  f"{out['easy'][k]['minority_mean']:>14.3f}"
                  f"{out['easy'][k]['reliable_mean']:>14.3f}")
        for k in PERANN_DISCOUNTED:
            v = verdicts[k]
            print(f"[2.2 {k}] hard: minoria={v['hard_minority']:.3f} "
                  f"(>= {PERANN_MINORITY_HARD_FLOOR}) {'ok' if v['c1'] else 'FALHA'}, "
                  f"gap={v['hard_gap']:+.3f} (<= {PERANN_MINORITY_HARD_MAX_GAP}) "
                  f"{'ok' if v['c2'] else 'FALHA'} | easy: minoria={v['easy_minority']:.3f} "
                  f"(<= {PERANN_MINORITY_EASY_CEILING}) {'ok' if v['c3'] else 'FALHA'}, "
                  f"gap={v['easy_gap']:+.3f} (>= {PERANN_MINORITY_EASY_MIN_GAP}) "
                  f"{'ok' if v['c4'] else 'FALHA'} => "
                  f"{'PASS' if v['pass'] else 'FAIL'}")
    return out, verdicts


def perann_bias_spammer(seeds=None, frac=0.20, label=1, theta=0.25,
                        hard_frac=0.4, density="confortavel", verbose=True):
    """2.3 — spammer de viés constante (sempre `label`).

    Os erros dele caem sobretudo em itens FÁCEIS (onde o pool concorda no oposto),
    logo o desconto não o deve proteger.
    """
    seeds = seeds or DEFAULT_SEEDS
    dens = PERANN_DENSITIES[density]
    per_cfg = {k: {"biased": [], "reliable": []} for k in PERANN_CONFIGS}
    for seed in seeds:
        ann, th, is_hard, groups, info = generate_difficulty_synthetic(
            hard_frac=hard_frac, biased_spammer_frac=frac, biased_spammer_label=label,
            biased_spammer_theta=theta, seed=seed, **dens)
        res = _perann_all_configs(ann, dens["n_annotators"], dens["n_items"])
        for k, r in res.items():
            c = r["competence"]
            per_cfg[k]["biased"].append(float(c[groups["biased_spammer"]].mean()))
            per_cfg[k]["reliable"].append(float(c[groups["reliable_majority"]].mean()))
    out = {k: {"biased_mean": float(np.mean(v["biased"])),
               "reliable_mean": float(np.mean(v["reliable"])),
               "biased_per_seed": v["biased"]} for k, v in per_cfg.items()}
    verdicts = {}
    for k in PERANN_DISCOUNTED:
        gap = out[k]["reliable_mean"] - out[k]["biased_mean"]
        c1 = out[k]["biased_mean"] <= PERANN_BIAS_CEILING
        c2 = gap >= PERANN_BIAS_MIN_GAP
        verdicts[k] = {"pass": bool(c1 and c2), "biased": out[k]["biased_mean"],
                       "gap": gap, "c1": bool(c1), "c2": bool(c2)}
    if verbose:
        print(f"\n--- 2.3 Spammer de viés constante (label={label}, frac={frac}) ---")
        print(f"{'config':<15}{'enviesados':>13}{'fiáveis':>13}")
        for k in PERANN_CONFIGS:
            print(f"{k:<15}{out[k]['biased_mean']:>13.3f}{out[k]['reliable_mean']:>13.3f}")
        for k in PERANN_DISCOUNTED:
            v = verdicts[k]
            print(f"[2.3 {k}] enviesados={v['biased']:.3f} (<= {PERANN_BIAS_CEILING}) "
                  f"{'ok' if v['c1'] else 'FALHA'}, gap={v['gap']:+.3f} "
                  f"(>= {PERANN_BIAS_MIN_GAP}) {'ok' if v['c2'] else 'FALHA'} => "
                  f"{'PASS' if v['pass'] else 'FAIL'}")
    return out, verdicts


def perann_dissenter_density_sweep(fracs=(0.03, 0.08, 0.15, 0.25, 0.35), seeds=None,
                                   hard_frac=0.4, density="confortavel", verbose=True):
    """DIAGNÓSTICO (não é critério de portão): como a discriminação varia com a
    DENSIDADE do bloco dissidente.

    A dificuldade empírica sai dos votos. Um bloco de dissidentes suficientemente
    grande FABRICA a dificuldade dos itens em que discorda, e o desconto passa a
    protegê-lo mesmo em terreno de consenso. Este sweep mede onde isso começa,
    nos dois braços em que importa: minoria em terreno FÁCIL (deve ser punida) e
    spammer de viés constante (deve ser punido).
    """
    seeds = seeds or DEFAULT_SEEDS
    dens = PERANN_DENSITIES[density]
    rows = []
    for frac in fracs:
        rec = {"frac": frac}
        # braço A: minoria dissidente em terreno FÁCIL (é ruído; deve ser punida)
        acc = {k: {"d": [], "r": []} for k in PERANN_CONFIGS}
        for seed in seeds:
            ann, th, ih, gr, info = generate_difficulty_synthetic(
                hard_frac=hard_frac, minority_frac=frac, minority_theta=0.95,
                minority_mode="easy", seed=seed, **dens)
            res = _perann_all_configs(ann, dens["n_annotators"], dens["n_items"])
            for k, r in res.items():
                c = r["competence"]
                acc[k]["d"].append(float(c[gr["minority"]].mean()))
                acc[k]["r"].append(float(c[gr["reliable_majority"]].mean()))
        for k in PERANN_CONFIGS:
            rec[f"easy_{k}"] = float(np.mean(acc[k]["d"]))
            rec[f"easy_gap_{k}"] = float(np.mean(acc[k]["r"]) - np.mean(acc[k]["d"]))
        # braço B: spammer de viés constante
        acc = {k: {"d": [], "r": []} for k in PERANN_CONFIGS}
        for seed in seeds:
            ann, th, ih, gr, info = generate_difficulty_synthetic(
                hard_frac=hard_frac, biased_spammer_frac=frac, biased_spammer_label=1,
                biased_spammer_theta=0.25, seed=seed, **dens)
            res = _perann_all_configs(ann, dens["n_annotators"], dens["n_items"])
            for k, r in res.items():
                c = r["competence"]
                acc[k]["d"].append(float(c[gr["biased_spammer"]].mean()))
                acc[k]["r"].append(float(c[gr["reliable_majority"]].mean()))
        for k in PERANN_CONFIGS:
            rec[f"bias_{k}"] = float(np.mean(acc[k]["d"]))
            rec[f"bias_gap_{k}"] = float(np.mean(acc[k]["r"]) - np.mean(acc[k]["d"]))
        rows.append(rec)
    if verbose:
        print("\n--- DIAGNÓSTICO: discriminação vs densidade do bloco dissidente ---")
        print("(gap = competência dos fiáveis - competência do bloco; maior = melhor)")
        for tag, name in (("easy", "minoria dissidente em terreno FÁCIL"),
                          ("bias", "spammer de viés constante")):
            print(f"\n  {name}")
            print(f"  {'frac':>6}" + "".join(f"{k:>16}" for k in PERANN_CONFIGS))
            for r in rows:
                print(f"  {r['frac']:>6.2f}" + "".join(
                    f"{r[f'{tag}_{k}']:>8.3f}{r[f'{tag}_gap_{k}']:>+8.3f}"
                    for k in PERANN_CONFIGS))
    return rows


def perann_determinism(seed=42):
    ann, _, _, _, _ = generate_difficulty_synthetic(seed=seed)
    a = compute_competence(ann, n_annotators=60, n_items=800)["competence"]
    b = compute_competence(ann, n_annotators=60, n_items=800)["competence"]
    ok = np.array_equal(a, b)
    print(f"[perann determinismo] seed={seed}: {'OK' if ok else 'FALHOU'} "
          f"(max|Δ|={np.abs(a - b).max():.3e})")
    return ok


def run_perann_gate(seeds=None, verbose=True):
    """Corre 2.1 + 2.2 + 2.3 e grava o portão em JSON."""
    seeds = seeds or DEFAULT_SEEDS
    print("=" * 78)
    print("PORTÃO SINTÉTICO — estimador por-anotador com desconto de dificuldade")
    print(f"limiares pré-registados: r>={PERANN_RECOVERY_MIN_R} "
          f"vantagem>={PERANN_DISCOUNT_MIN_ADVANTAGE} | minoria-hard>="
          f"{PERANN_MINORITY_HARD_FLOOR} gap<={PERANN_MINORITY_HARD_MAX_GAP} | "
          f"minoria-easy<={PERANN_MINORITY_EASY_CEILING} gap>="
          f"{PERANN_MINORITY_EASY_MIN_GAP} | viés<={PERANN_BIAS_CEILING} "
          f"gap>={PERANN_BIAS_MIN_GAP}")
    print("=" * 78)
    grid, v_grid = perann_recovery_grid(seeds=seeds, verbose=verbose)
    mino, v_mino = perann_minority_arms(seeds=seeds, verbose=verbose)
    bias, v_bias = perann_bias_spammer(seeds=seeds, verbose=verbose)
    sweep = perann_dissenter_density_sweep(seeds=seeds, verbose=verbose)
    det = perann_determinism()
    print("\n--- VEREDITO DO PORTÃO ---")
    for k in PERANN_DISCOUNTED:
        allp = v_grid[k]["pass"] and v_mino[k]["pass"] and v_bias[k]["pass"]
        print(f"  {k:<14} 2.1 {'PASS' if v_grid[k]['pass'] else 'FAIL'} | "
              f"2.2 {'PASS' if v_mino[k]['pass'] else 'FAIL'} | "
              f"2.3 {'PASS' if v_bias[k]['pass'] else 'FAIL'} "
              f"=> {'PASS' if allp else 'FAIL'}")
    payload = {"thresholds": {
                   "PERANN_RECOVERY_MIN_R": PERANN_RECOVERY_MIN_R,
                   "PERANN_DISCOUNT_MIN_ADVANTAGE": PERANN_DISCOUNT_MIN_ADVANTAGE,
                   "PERANN_MINORITY_HARD_FLOOR": PERANN_MINORITY_HARD_FLOOR,
                   "PERANN_MINORITY_HARD_MAX_GAP": PERANN_MINORITY_HARD_MAX_GAP,
                   "PERANN_MINORITY_EASY_CEILING": PERANN_MINORITY_EASY_CEILING,
                   "PERANN_MINORITY_EASY_MIN_GAP": PERANN_MINORITY_EASY_MIN_GAP,
                   "PERANN_BIAS_CEILING": PERANN_BIAS_CEILING,
                   "PERANN_BIAS_MIN_GAP": PERANN_BIAS_MIN_GAP},
               "seeds": list(seeds), "recovery_grid": grid,
               "recovery_verdicts": v_grid, "minority": mino,
               "minority_verdicts": v_mino, "bias_spammer": bias,
               "bias_verdicts": v_bias, "density_sweep": sweep,
               "determinism": bool(det)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PERANN_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[perann] portão gravado em {PERANN_JSON}")
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", action="store_true", help="corre a grade de recuperação")
    p.add_argument("--minority", action="store_true", help="teste de preservação de minoria")
    p.add_argument("--bias-spammer", action="store_true")
    p.add_argument("--determinism", action="store_true")
    p.add_argument("--perann", action="store_true",
                   help="portão sintético do estimador por-anotador com desconto")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.all or not (args.grid or args.minority or args.bias_spammer
                        or args.determinism or args.perann):
        args.grid = args.minority = args.bias_spammer = args.determinism = True
        args.perann = args.all
    if args.grid:
        run_recovery_grid()
    if args.minority:
        test_minority_preservation()
    if args.bias_spammer:
        print("\n=== viés constante = sempre YES (label 1) ===")
        test_constant_bias_spammer_detection(biased_spammer_label=1)
        print("\n=== viés constante = sempre NO (label 0) ===")
        test_constant_bias_spammer_detection(biased_spammer_label=0)
        print("\nStress-sweep da fração de spammers enviesados (modo adversarial):")
        sweep_bias_spammer_fraction()
    if args.determinism:
        test_determinism()
    if args.perann:
        run_perann_gate()


if __name__ == "__main__":
    main()
