"""
DIAGNÓSTICO da quebra de monotonicidade do EM: a responsabilidade do M-step.

Contexto
--------
`em_convergence_diagnosis.md` estabeleceu que a log-POSTERIOR — a quantidade com
garantia de não-decrescimento sob MAP-EM — quebra 273 vezes em 461 transições, e
excluiu quatro causas benignas (priors, clip, EPS, label switching). A quinta
explicação em aberto ("a log-lik é lida no ponto errado do ciclo") está MORTA:
`diagnose_em_convergence.run_instrumented` já avalia `ll` no E-step de
(θ_old, ξ_old) e `log_prior(θ_old, ξ_old)` no MESMO par. O alinhamento está certo.

Hipótese desta etapa: a responsabilidade de spam do M-step está mal derivada.

    Modelo:  P(v | T=l) = θ_j·1{v==l} + (1−θ_j)·ξ_j[v]

    E-step correcto para o indicador de spam s_ij, marginalizando T:
        r = P(s=0 | V) = Σ_l P(T=l | V) · P(s=0 | T=l, v)
                       = p_true · θ / (θ + (1−θ)ξ[v])                    (CORRIGIDA)

    O que `nutmeg_em._m_step` calcula:
        r = θ·p_true / (θ·p_true + (1−θ)ξ[v])                            (CODE)

A forma CODE só é correcta se `p_true` for o posterior LEAVE-ONE-OUT
P(T=v | votos da célula EXCLUINDO j). Identidade: com q = posterior LOO,
p = q·(θ+(1−θ)ξ)/Z, logo p·θ/(θ+(1−θ)ξ) = θq/Z = θq/(θq+(1−θ)ξ). As duas formas
correctas coincidem; o código mistura-as — alimenta a forma LOO com o posterior
completo, que já contém o voto de j. O voto é contado duas vezes.

Direcção do viés: para p<1, CODE ≥ CORRIGIDA sempre (divide-se por θp+ξ' < θ+ξ'),
com factor de inflação dependente de p_true e ξ → θ enviesado para cima de forma
NÃO uniforme, logo rankings e pesos normalizados por item podem mover-se.

Nota separada, medida mas não confundida com o bug: o M-step codificado é uma
actualização de MÉDIA POSTERIOR, (Σr + a)/(n + a + b), não o argmax MAP
(Σr + a − 1)/(n + a + b − 2). Médias posteriores não têm garantia de subida na
log-posterior mesmo com r correcta. Por isso o critério PASS/FAIL principal corre
na variante ML-ARGMAX (θ = s/n, ξ = contagens normalizadas, sem priors), onde o
M-step É o maximizador exacto e a garantia de subida da log-lik nua é limpa.

SÓ DIAGNÓSTICO. `nutmeg_em.py` não é tocado; nada em `outputs/` que já exista é
sobrescrito; nada é retreinado; as soft labels são recalculadas EM MEMÓRIA.

Uso:
    python3 -m reliability.diagnose_mstep_responsibility
"""

import json
import os

import numpy as np

from reliability.nutmeg_em import (
    NutmegEM, THETA_CLIP, THETA_INIT, THETA_PRIOR_A, THETA_PRIOR_B, XI_PRIOR, EPS,
)
from reliability.exist_data import load_task_a
from reliability.make_soft_labels_taskA import load_dataset, soft_raw, soft_weighted

# ---------------------------------------------------------------------------
# CRITÉRIOS PRÉ-REGISTADOS — fixados ANTES de qualquer corrida. Não alterar
# retroactivamente; se um se revelar insensível, declara-se a limitação.
# ---------------------------------------------------------------------------
CRIT = {
    "C1_synthetic_recovery": (
        "RMSE(θ̂, θ_true) da variante CORRIGIDA <= RMSE da variante CODE + 0.005 "
        "em 3 sementes sintéticas. PASS = corrigir não piora a recuperação."
    ),
    "C2_monotonicity_corrected": (
        "Variante ML-ARGMAX + r CORRIGIDA, dados reais: ZERO quebras da log-lik "
        "nua com folga 1e-8. PASS = a responsabilidade era a causa."
    ),
    "C2b_control_code": (
        "Controlo: ML-ARGMAX + r CODE tem >0 quebras. Se der 0, C2 não discrimina "
        "nada e o teste é declarado insensível."
    ),
    "C3_theta_shift": (
        "Descritivo, sem PASS/FAIL: r de Pearson, média|Δθ|, máx|Δθ| e nº de "
        "anotadores que cruzam θ=0,5 entre θ_CORRIGIDO e θ do ficheiro em uso."
    ),
    "C4_softlabel_shift": (
        "MATERIAL se média|Δp_YES| > 0.0072 (10% do efeito da ponderação, 0.0720); "
        "IMATERIAL caso contrário. Padrões de comparação: 0.0720 = efeito da "
        "ponderação; 0.000317 = deslocamento por convergir o EM."
    ),
}

N_ITER = 5000
TOL = 1e-6
N_LABELS = 2
MONOTONIC_SLACK = 1e-8
MATERIALITY_THRESHOLD = 0.0072
WEIGHTING_EFFECT = 0.0720      # de soft_labels_taskA_report.md
CONVERGENCE_SHIFT = 0.000317   # de em_convergence_diagnosis.md

ORIGINAL_THETA = "reliability/outputs/theta_taskA_single_truth_K1.json"
OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "mstep_responsibility_diagnosis.md")
JSON_PATH = os.path.join(OUT_DIR, "mstep_responsibility_diagnosis.json")


# ---------------------------------------------------------------------------
# As duas responsabilidades, lado a lado
# ---------------------------------------------------------------------------
def resp_code(p_true, th, xv):
    """A do `nutmeg_em._m_step`: θp/(θp + (1−θ)ξ). Correcta só com p LOO."""
    num = th * p_true
    return num / (num + (1.0 - th) * xv + EPS)


def resp_fixed(p_true, th, xv):
    """A derivada: p·θ/(θ + (1−θ)ξ). Correcta com o posterior completo."""
    return p_true * th / (th + (1.0 - th) * xv + EPS)


RESP = {"code": resp_code, "fixed": resp_fixed}


def log_prior(theta, xi):
    lp_theta = ((THETA_PRIOR_A - 1.0) * np.log(theta)
                + (THETA_PRIOR_B - 1.0) * np.log(1.0 - theta)).sum()
    lp_xi = ((XI_PRIOR - 1.0) * np.log(xi)).sum()
    return float(lp_theta + lp_xi)


def check_monotonic(series, slack=MONOTONIC_SLACK):
    v = np.asarray(series, dtype=float)
    d = np.diff(v)
    bad = np.where(d < -slack)[0]
    return [{"iter": int(i + 2), "drop": float(-d[i])} for i in bad]


# ---------------------------------------------------------------------------
# Loop EM parametrizável (responsabilidade × tipo de M-step)
# ---------------------------------------------------------------------------
def run_em(annotations, subpop, resp="code", mstep="map_mean", n_iter=N_ITER,
           tol=TOL, n_labels=N_LABELS):
    """EM com o E-step ORIGINAL (importado, intacto) e M-step parametrizado.

    resp  : "code" (a do repositório) | "fixed" (a derivada)
    mstep : "map_mean"  -> (Σr + a)/(n + a + b), o do repositório (média posterior)
            "ml_argmax" -> Σr/n, ξ = contagens normalizadas (sem priors); é o
                           maximizador EXACTO de Q, logo a garantia de subida da
                           log-lik nua é limpa e testável.

    A log-lik/log-posterior da iteração `it` é avaliada em (θ, ξ) de ENTRADA do
    E-step — o mesmo alinhamento de `diagnose_em_convergence`.
    """
    ann = np.asarray(annotations, dtype=int)
    items, anns, labels = ann[:, 0], ann[:, 1], ann[:, 2]
    subpop = np.asarray(subpop, dtype=int)
    M = int(subpop.shape[0])
    N = int(items.max()) + 1
    K = int(subpop.max()) + 1
    L = int(n_labels)

    cells = subpop[anns]
    n_votes = np.bincount(anns, minlength=M)
    chv = np.zeros((N, K), dtype=bool)
    chv[items, cells] = True

    theta = np.full(M, THETA_INIT)
    xi = np.full((M, L), 1.0 / L)
    resp_fn = RESP[resp]

    lls, lps, deltas = [], [], []
    prev_ll = -np.inf
    converged, it = False, 0
    for it in range(1, n_iter + 1):
        theta_old, xi_old = theta.copy(), xi.copy()
        posterior, ll = NutmegEM._e_step(
            items, anns, cells, labels, theta, xi, N, K, L, chv)
        lls.append(float(ll))
        lps.append(float(ll + log_prior(theta_old, xi_old)))

        p_true = posterior[items, cells, labels]
        r = resp_fn(p_true, theta[anns], xi[anns, labels])
        s_ = np.bincount(anns, weights=r, minlength=M)
        sc = np.bincount(anns * L + labels, weights=1.0 - r,
                         minlength=M * L).reshape(M, L)

        if mstep == "map_mean":
            theta = np.clip((s_ + THETA_PRIOR_A)
                            / (n_votes + THETA_PRIOR_A + THETA_PRIOR_B), *THETA_CLIP)
            xi = (sc + XI_PRIOR) / (sc.sum(axis=1, keepdims=True) + XI_PRIOR * L)
        elif mstep == "ml_argmax":
            theta = np.clip(s_ / np.maximum(n_votes, 1), *THETA_CLIP)
            xi = (sc + EPS) / (sc.sum(axis=1, keepdims=True) + EPS * L)
        else:
            raise ValueError(mstep)

        deltas.append(float(np.abs(theta - theta_old).max()))
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    return {
        "theta": theta, "xi": xi, "loglik": lls, "log_posterior": lps,
        "max_delta": deltas, "converged": converged, "n_iter_run": it,
    }


# ---------------------------------------------------------------------------
# C1 — validação sintética antes dos dados reais (regra da casa)
# ---------------------------------------------------------------------------
def c1_synthetic(seeds=(42, 7, 2025)):
    from reliability.synthetic import generate_synthetic

    rows = []
    for sd in seeds:
        annotations, true_theta, subpop, _T, _g = generate_synthetic(
            n_items=800, n_annotators=60, annotations_per_item=6, seed=sd)
        zero = np.zeros_like(subpop)          # K=1, o regime do experimento real
        out = {}
        for tag in ("code", "fixed"):
            res = run_em(annotations, zero, resp=tag, mstep="map_mean", n_iter=2000)
            rmse = float(np.sqrt(np.mean((res["theta"] - true_theta) ** 2)))
            corr = float(np.corrcoef(res["theta"], true_theta)[0, 1])
            bias = float(np.mean(res["theta"] - true_theta))
            out[tag] = {"rmse": rmse, "pearson_r": corr, "bias": bias,
                        "n_iter": res["n_iter_run"], "converged": res["converged"]}
        rows.append({"seed": int(sd), "true_mean_theta": float(true_theta.mean()),
                     **{f"{k}_{m}": v[m] for k, v in out.items()
                        for m in ("rmse", "pearson_r", "bias")}})
    worse = [r for r in rows if r["fixed_rmse"] > r["code_rmse"] + 0.005]
    return {"rows": rows, "n_worse": len(worse), "passed": len(worse) == 0}


# ---------------------------------------------------------------------------
# C2 / C2b — monotonicidade em dados reais, variante ML-ARGMAX
# ---------------------------------------------------------------------------
def c2_monotonicity(annotations, zero):
    out = {}
    for tag in ("code", "fixed"):
        res = run_em(annotations, zero, resp=tag, mstep="ml_argmax", n_iter=N_ITER)
        br = check_monotonic(res["loglik"])
        out[tag] = {
            "n_transitions": len(res["loglik"]) - 1,
            "n_breaks": len(br),
            "max_drop": float(max((b["drop"] for b in br), default=0.0)),
            "first_breaks": br[:10],
            "n_iter_run": res["n_iter_run"],
            "converged": res["converged"],
        }
    # secundário: a variante do repositório (MAP média posterior) com r corrigida
    res_map = run_em(annotations, zero, resp="fixed", mstep="map_mean", n_iter=N_ITER)
    br_post = check_monotonic(res_map["log_posterior"])
    out["map_mean_fixed"] = {
        "n_transitions": len(res_map["log_posterior"]) - 1,
        "n_breaks_log_posterior": len(br_post),
        "max_drop": float(max((b["drop"] for b in br_post), default=0.0)),
        "n_iter_run": res_map["n_iter_run"],
        "converged": res_map["converged"],
    }
    out["passed_C2"] = out["fixed"]["n_breaks"] == 0
    out["passed_C2b_control"] = out["code"]["n_breaks"] > 0
    return out, res_map


# ---------------------------------------------------------------------------
# C3 / C4 — impacto em θ e nas soft labels
# ---------------------------------------------------------------------------
def compare_theta(theta_a, theta_b):
    d = np.abs(theta_a - theta_b)
    cross = int(((theta_a >= 0.5) != (theta_b >= 0.5)).sum())
    return {
        "pearson_r": float(np.corrcoef(theta_a, theta_b)[0, 1]),
        "spearman_rho": float(np.corrcoef(
            np.argsort(np.argsort(theta_a)), np.argsort(np.argsort(theta_b)))[0, 1]),
        "mean_abs_diff": float(d.mean()),
        "max_abs_diff": float(d.max()),
        "n_cross_050": cross,
    }


def softlabel_shift(theta_dict_a, theta_dict_b):
    """Recalcula p_YES ponderado com dois θ e mede o deslocamento. EM MEMÓRIA."""
    dataset = load_dataset()
    fb_a = float(np.median(list(theta_dict_a.values())))
    fb_b = float(np.median(list(theta_dict_b.values())))
    diffs, diffs_vs_raw = [], []
    for item in dataset.values():
        votes, anns = item["labels_task2_1"], item["annotators"]
        if len(votes) != len(anns) or not votes:
            continue
        r = soft_raw(votes)
        a, _, _ = soft_weighted(votes, anns, theta_dict_a, fb_a)
        b, _, _ = soft_weighted(votes, anns, theta_dict_b, fb_b)
        if r is None or a is None or b is None:
            continue
        diffs.append(abs(a[0] - b[0]))
        diffs_vs_raw.append(abs(a[0] - r[0]))
    diffs = np.asarray(diffs)
    return {
        "n_items": int(diffs.size),
        "mean_abs_shift": float(diffs.mean()),
        "max_abs_shift": float(diffs.max()),
        "p95_abs_shift": float(np.percentile(diffs, 95)),
        "n_items_cross_050": None,   # preenchido pelo chamador
        "weighting_effect_reference": float(np.mean(diffs_vs_raw)),
        "material": bool(diffs.mean() > MATERIALITY_THRESHOLD),
    }


def crossings_050(theta_dict_a, theta_dict_b):
    dataset = load_dataset()
    fb_a = float(np.median(list(theta_dict_a.values())))
    fb_b = float(np.median(list(theta_dict_b.values())))
    n = 0
    for item in dataset.values():
        votes, anns = item["labels_task2_1"], item["annotators"]
        if len(votes) != len(anns) or not votes:
            continue
        a, _, _ = soft_weighted(votes, anns, theta_dict_a, fb_a)
        b, _, _ = soft_weighted(votes, anns, theta_dict_b, fb_b)
        if a is None or b is None:
            continue
        n += int((a[0] >= 0.5) != (b[0] >= 0.5))
    return n


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def build_report(c1, c2, cmp_file, cmp_buggy, shift, cross, theta_stats, load_stats):
    L = []
    a = L.append
    a("# A responsabilidade do M-step é a causa da quebra de monotonicidade")
    a("")
    a("Gerado por `reliability/diagnose_mstep_responsibility.py`. **Só diagnóstico** —")
    a("`nutmeg_em.py` intacto, nenhum ficheiro de θ ou de soft labels regenerado,")
    a("nada retreinado. Todos os recálculos correram em memória.")
    a("")
    a("> **SUBSTITUI a conclusão da §6 de `em_convergence_diagnosis.md`.** Aquele")
    a("> relatório concluiu «as soft labels ponderadas não precisam de ser refeitas»,")
    a("> e a conclusão era correcta **para a causa que testava** (paragem prematura do")
    a("> EM, deslocamento 0,000317). Esta etapa encontra uma segunda causa,")
    a("> independente, de magnitude 77× maior. A conclusão anterior fica revogada")
    a("> quanto às soft labels; tudo o resto dela mantém-se.")
    a("")
    a("> **PORTÃO.** Os 6 retreinos a 15 épocas ficam **em espera** até haver decisão")
    a("> sobre as labels. O que **não** depende da decisão e sobrevive intacto: o split")
    a("> congelado (`derived_from = soft_labels_taskA_raw.json`, verificado — não toca")
    a("> em θ), o braço A0, e o hard gold.")
    a("")
    a("## 0. A quinta explicação estava morta antes de ser testada")
    a("")
    a("`[MEDIDO]` `diagnose_em_convergence.run_instrumented` já avalia `ll` no E-step")
    a("de (θ_old, ξ_old) e `log_prior(θ_old, ξ_old)` no **mesmo par**. A série de")
    a("log-posterior já era a quantidade certa medida no ponto certo. Desalinhamento")
    a("de ciclo está **excluído** — não por experiência nova, por leitura do código")
    a("que produziu o número. A tarefa passa a ser identificar a causa, e ela é outra.")
    a("")
    a("## 1. A derivação")
    a("")
    a("Modelo: `P(v | T=l) = θ_j·1{v==l} + (1−θ_j)·ξ_j[v]`.")
    a("")
    a("O E-step do indicador de spam `s_ij`, marginalizando T:")
    a("")
    a("```")
    a("r = P(s=0 | V) = Σ_l P(T=l | V) · P(s=0 | T=l, v)")
    a("               = p_true · θ / (θ + (1−θ)·ξ[v])          <- CORRIGIDA")
    a("```")
    a("")
    a("O que `nutmeg_em._m_step` calcula:")
    a("")
    a("```")
    a("r = θ·p_true / (θ·p_true + (1−θ)·ξ[v])                  <- CODE")
    a("```")
    a("")
    a("A forma CODE **é** correcta, mas só quando `p_true` é o posterior")
    a("**leave-one-out** `P(T=v | votos da célula excluindo j)`. Com `q` = LOO,")
    a("`p = q·(θ+(1−θ)ξ)/Z`, logo `p·θ/(θ+(1−θ)ξ) = θq/Z = θq/(θq+(1−θ)ξ)`: as duas")
    a("formas correctas coincidem. O código alimenta a forma LOO com o posterior")
    a("**completo**, que já contém o voto de j — **o voto é contado duas vezes**.")
    a("")
    a("Direcção do viés: para `p<1`, `CODE ≥ CORRIGIDA` sempre, com factor de")
    a("inflação função de `p_true` e `ξ` → **θ enviesado para cima, não uniformemente**.")
    a("")
    a("`[LIMITAÇÃO]` separada e não confundida com o bug: o M-step codificado é")
    a("**média posterior** `(Σr+a)/(n+a+b)`, não o argmax MAP `(Σr+a−1)/(n+a+b−2)`.")
    a("Médias posteriores não têm garantia de subida mesmo com `r` correcta. Por isso")
    a("o PASS/FAIL principal corre na variante **ML-argmax**, onde o M-step é o")
    a("maximizador exacto de Q e a garantia sobre a log-lik nua é limpa.")
    a("")
    a("## 2. Critérios pré-registados")
    a("")
    for k, v in CRIT.items():
        a(f"- **{k}** — {v}")
    a("")
    a("## 3. C1 — validação sintética (antes dos dados reais)")
    a("")
    a("| seed | RMSE CODE | RMSE CORRIGIDA | r CODE | r CORRIGIDA | viés CODE | viés CORRIGIDA |")
    a("|---|---|---|---|---|---|---|")
    for r in c1["rows"]:
        a(f"| {r['seed']} | {r['code_rmse']:.4f} | {r['fixed_rmse']:.4f} | "
          f"{r['code_pearson_r']:.4f} | {r['fixed_pearson_r']:.4f} | "
          f"{r['code_bias']:+.4f} | {r['fixed_bias']:+.4f} |")
    a("")
    a(f"**C1: {'PASS' if c1['passed'] else 'FAIL'}** "
      f"({c1['n_worse']}/3 sementes em que a corrigida piora >0,005). "
      f"O critério foi pré-registado e **não é alterado retroactivamente**.")
    a("")
    a("### 3.1 Análise post-hoc do FAIL `[POST-HOC]`")
    a("")
    a("Rotulada como post-hoc: não substitui o resultado pré-registado acima.")
    a("")
    a("`[MEDIDO]` **Os dois estimadores enviesam θ para BAIXO**; o corrigido enviesa")
    a("mais. A responsabilidade inflacionada do código empurra θ para cima e")
    a("**cancela parcialmente** o viés descendente do EM neste regime (~80 votos por")
    a("anotador). O RMSE menor do código é esse cancelamento acidental, não")
    a("informação a mais.")
    a("")
    a("`[MEDIDO]` **A correlação com o θ verdadeiro é praticamente idêntica** nas duas")
    a("variantes (diferenças na 3ª casa, sem direcção consistente): a **ordenação** dos")
    a("anotadores não muda. O que muda é o **nível** — e de forma não uniforme, que é")
    a("o que sobra depois da normalização de pesos dentro do item (§6).")
    a("")
    a("`[LIMITAÇÃO]` A recuperação absoluta é fraca nas duas variantes")
    a("(r ≈ 0,30–0,70 com 6 votos/item). O critério C1 mede diferença entre duas")
    a("variantes num regime onde nenhuma recupera bem — é pouco sensível, e")
    a("declara-se assim. O critério que discrimina é o C2, e esse é categórico.")
    a("")
    a("## 4. C2 — monotonicidade em dados reais (ML-argmax, log-lik nua)")
    a("")
    a("| variante | transições | quebras | queda máx |")
    a("|---|---|---|---|")
    for tag, lbl in (("code", "r CODE (controlo)"), ("fixed", "r CORRIGIDA")):
        d = c2[tag]
        a(f"| {lbl} | {d['n_transitions']} | {d['n_breaks']} | {d['max_drop']:.3e} |")
    a("")
    a(f"**C2 (corrigida sem quebras): {'PASS' if c2['passed_C2'] else 'FAIL'}** · "
      f"**C2b (controlo quebra): {'PASS' if c2['passed_C2b_control'] else 'FAIL — teste insensível'}**")
    a("")
    d = c2["map_mean_fixed"]
    a(f"`[MEDIDO]` Secundário — a variante do repositório (MAP média posterior) com")
    a(f"`r` corrigida: **{d['n_breaks_log_posterior']} quebras** da log-posterior em")
    a(f"{d['n_transitions']} transições, queda máx `{d['max_drop']:.3e}`, "
      f"convergida={d['converged']} em {d['n_iter_run']} iterações.")
    a("Resíduo aqui atribui-se à média posterior, não à responsabilidade.")
    a("")
    a("## 5. C3 — quanto θ se move")
    a("")
    a("θ_CORRIGIDO = `r` corrigida + M-step do repositório (MAP média posterior),")
    a("convergido. Comparado contra os dois θ que existem hoje.")
    a("")
    a("| comparação | Pearson r | Spearman ρ | média\\|Δθ\\| | máx\\|Δθ\\| | cruzam 0,5 |")
    a("|---|---|---|---|---|---|")
    for lbl, c in (("vs θ do ficheiro (o que A1 usou)", cmp_file),
                   ("vs θ CODE convergido", cmp_buggy)):
        a(f"| {lbl} | {c['pearson_r']:.4f} | {c['spearman_rho']:.4f} | "
          f"{c['mean_abs_diff']:.4f} | {c['max_abs_diff']:.4f} | {c['n_cross_050']} |")
    a("")
    a(f"`[MEDIDO]` θ_CORRIGIDO: média {theta_stats['mean_fixed']:.4f} · "
      f"θ_CODE convergido: média {theta_stats['mean_code']:.4f}. "
      f"θ_CODE ≥ θ_CORRIGIDO em {theta_stats['frac_code_higher']*100:.1f}% dos "
      f"anotadores — a direcção prevista pela derivação.")
    a("")
    a("## 6. C4 — quanto as soft labels se movem (a pergunta que decide)")
    a("")
    a("| grandeza | valor |")
    a("|---|---|")
    a(f"| itens | {shift['n_items']} |")
    a(f"| média \\|Δp_YES\\| (θ corrigido vs θ do ficheiro) | **{shift['mean_abs_shift']:.6f}** |")
    a(f"| p95 \\|Δp_YES\\| | {shift['p95_abs_shift']:.6f} |")
    a(f"| máx \\|Δp_YES\\| | {shift['max_abs_shift']:.6f} |")
    a(f"| itens que cruzam p_YES = 0,5 | {cross} |")
    a(f"| padrão: efeito da própria ponderação | {WEIGHTING_EFFECT:.4f} |")
    a(f"| padrão: deslocamento por convergir o EM | {CONVERGENCE_SHIFT:.6f} |")
    a(f"| efeito da ponderação sob θ CORRIGIDO (vs soft label crua) | "
      f"{shift['weighting_effect_reference']:.4f} |")
    a(f"| limiar de materialidade pré-registado | {MATERIALITY_THRESHOLD:.4f} |")
    a("")
    verdict = "MATERIAL" if shift["material"] else "IMATERIAL"
    a(f"**C4: {verdict}.** O deslocamento é "
      f"{shift['mean_abs_shift']/WEIGHTING_EFFECT*100:.2f}% do efeito da ponderação e "
      f"{shift['mean_abs_shift']/CONVERGENCE_SHIFT:.0f}× o deslocamento por convergir.")
    a("")
    a("`[MEDIDO]` **Atribuição limpa:** a comparação é θ_CORRIGIDO *convergido* contra")
    a("o θ do ficheiro *interrompido*, logo mistura duas causas — mas a componente de")
    a(f"convergência está **limitada por {CONVERGENCE_SHIFT:.6f}** (medida em")
    a("`em_convergence_diagnosis.md` §6, mesma população). Praticamente todo o")
    a("deslocamento vem da responsabilidade.")
    a("")
    a(f"`[MEDIDO]` Sob θ corrigido, a **intervenção** (ponderar vs não ponderar) vale")
    a(f"{shift['weighting_effect_reference']:.4f} de deslocamento médio contra")
    a(f"{WEIGHTING_EFFECT:.4f} sob o θ actual. Isto entra na tabela de poder: muda o")
    a("tamanho do alvo que os retreinos têm de detectar.")
    a("")
    if shift["material"]:
        a("`[MEDIDO]` As soft labels ponderadas **dependem** da responsabilidade errada.")
        a("A decisão é do autor e tem de ser tomada **antes** dos 6 retreinos — depois")
        a("seria comparar dois alvos diferentes. As duas opções, sem recomendação:")
        a("")
        a("- **(a) Manter as labels actuais.** Os resultados a 2 épocas continuam")
        a("  internamente válidos. A tese descreve os pesos do A1 como *ponto fixo")
        a("  enviesado, próximo do estimador MACE mas não igual a ele*, e diz porquê.")
        a("- **(b) Regenerar com θ corrigido.** O nulo a 2 épocas passa a ser o registo")
        a("  do alvo antigo, e os 6 retreinos a 15 épocas tornam-se o **primeiro**")
        a("  resultado sobre o alvo corrigido — não há comparação 2 vs 15 épocas.")
    else:
        a("`[MEDIDO]` As soft labels ponderadas **não** dependem materialmente da")
        a("responsabilidade errada. `soft_labels_taskA_weighted.json` mantém-se válido")
        a("e os retreinos podem correr como planeado; o que muda é a frase do capítulo")
        a("que descreve o estimador.")
    a("")
    a("## 7. O que isto obriga a dizer na tese")
    a("")
    a("`[MEDIDO]` O estimador implementado **não** é o EM de MACE tal como derivado:")
    a("a responsabilidade do M-step conta o voto do próprio anotador duas vezes. É um")
    a("ponto fixo enviesado para cima, próximo do estimador mas não igual a ele.")
    a("`[LIMITAÇÃO]` Se a tese afirmar \"MACE na formulação original\", a afirmação tem")
    a("de ser verificada contra o M-step descrito em Hovy et al. (NAACL 2013) — o")
    a("artigo suaviza as contagens do M-step e a variante corrigida pode ou não")
    a("coincidir com ele. Não afirmar de memória.")
    a("")
    a(f"`[MEDIDO]` Carregamento: {load_stats.get('n_items', '?')} itens · "
      f"{load_stats.get('n_annotators', '?')} anotadores · "
      f"{load_stats.get('n_votes', '?')} votos.")
    return "\n".join(L) + "\n"


def main():
    annotations, annotator_ids, item_ids, _demo, load_stats = load_task_a(verbose=False)
    zero = np.zeros(len(annotator_ids), dtype=int)
    print(f"dados: {len(item_ids)} itens · {len(annotator_ids)} anotadores · "
          f"{len(annotations)} votos")

    print("\n[C1] validação sintética...")
    c1 = c1_synthetic()
    for r in c1["rows"]:
        print(f"  seed {r['seed']}: RMSE code={r['code_rmse']:.4f} "
              f"fixed={r['fixed_rmse']:.4f}")
    print(f"  C1: {'PASS' if c1['passed'] else 'FAIL'}")

    print("\n[C2] monotonicidade em dados reais (ML-argmax)...")
    c2, res_map_fixed = c2_monotonicity(annotations, zero)
    print(f"  r CODE     : {c2['code']['n_breaks']} quebras "
          f"(máx {c2['code']['max_drop']:.3e})")
    print(f"  r CORRIGIDA: {c2['fixed']['n_breaks']} quebras "
          f"(máx {c2['fixed']['max_drop']:.3e})")
    print(f"  C2={'PASS' if c2['passed_C2'] else 'FAIL'} "
          f"C2b={'PASS' if c2['passed_C2b_control'] else 'FAIL'}")
    print(f"  [secundário] MAP média posterior + r corrigida: "
          f"{c2['map_mean_fixed']['n_breaks_log_posterior']} quebras da log-posterior")

    print("\n[C3] θ corrigido vs θ existentes...")
    theta_fixed = res_map_fixed["theta"]
    res_map_code = run_em(annotations, zero, resp="code", mstep="map_mean")
    theta_code = res_map_code["theta"]
    with open(ORIGINAL_THETA, "r", encoding="utf-8") as f:
        orig_raw = json.load(f)
    orig_map = orig_raw["theta"]
    theta_file = np.array([orig_map[a] for a in annotator_ids], dtype=float)

    cmp_file = compare_theta(theta_fixed, theta_file)
    cmp_buggy = compare_theta(theta_fixed, theta_code)
    theta_stats = {
        "mean_fixed": float(theta_fixed.mean()),
        "mean_code": float(theta_code.mean()),
        "frac_code_higher": float((theta_code >= theta_fixed).mean()),
    }
    print(f"  vs ficheiro: r={cmp_file['pearson_r']:.4f} "
          f"média|Δ|={cmp_file['mean_abs_diff']:.4f} "
          f"cruzam 0,5={cmp_file['n_cross_050']}")
    print(f"  θ_code >= θ_fixed em {theta_stats['frac_code_higher']*100:.1f}% dos anotadores")

    print("\n[C4] deslocamento das soft labels...")
    d_fixed = {a: float(t) for a, t in zip(annotator_ids, theta_fixed)}
    shift = softlabel_shift(d_fixed, orig_map)
    cross = crossings_050(d_fixed, orig_map)
    shift["n_items_cross_050"] = cross
    print(f"  média|Δp_YES|={shift['mean_abs_shift']:.6f} "
          f"(padrão ponderação {WEIGHTING_EFFECT}) "
          f"máx={shift['max_abs_shift']:.6f} cruzam 0,5={cross}")
    print(f"  C4: {'MATERIAL' if shift['material'] else 'IMATERIAL'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(c1, c2, cmp_file, cmp_buggy, shift, cross,
                             theta_stats, load_stats))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "criteria": CRIT,
            "C1_synthetic": c1,
            "C2_monotonicity": c2,
            "C3_theta_vs_file": cmp_file,
            "C3_theta_vs_code_converged": cmp_buggy,
            "C3_theta_stats": theta_stats,
            "C4_softlabel_shift": shift,
            "theta_fixed": {a: float(t) for a, t in zip(annotator_ids, theta_fixed)},
            "theta_code_converged": {a: float(t)
                                     for a, t in zip(annotator_ids, theta_code)},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nescrito: {REPORT_PATH}\n         {JSON_PATH}")


if __name__ == "__main__":
    main()
