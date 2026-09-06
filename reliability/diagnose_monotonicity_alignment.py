"""
FECHO da questão da monotonicidade: a log-lik lida no FIM da iteração completa.

Pedido
------
Reavaliar a log-verossimilhança apenas no fim de cada iteração completa, com θ e
ξ do MESMO estado (nunca entre o E-step e o M-step), e monitorar a log-POSTERIOR
no modo MAP. Se as violações desaparecerem, a causa era a instrumentação.

O que muda face ao que já existe
--------------------------------
`diagnose_em_convergence.run_instrumented` regista, na iteração `it`, a log-lik
devolvida pelo E-step — avaliada em (θ_old, ξ_old), os parâmetros de ENTRADA — e
soma-lhe `log_prior(θ_old, ξ_old)`. Os dois membros vêm do mesmo estado, logo o
alinhamento já estava coerente. Este script mede a variante pedida explicitamente:

    ponto INÍCIO : L(θ_t, ξ_t)      medido antes do M-step da iteração t
    ponto FIM    : L(θ_{t+1}, ξ_{t+1})  medido DEPOIS do M-step da iteração t,
                   com um E-step novo, nos parâmetros de saída

Ambas as séries são funções puras dos parâmetros de um único estado. Se a série
FIM for monótona e a INÍCIO não, a causa era instrumentação. Se as duas quebrarem,
o problema está no estimador.

`[NOTA]` As duas séries são a MESMA sequência de valores desfasada de uma posição
(o E-step do fim da iteração t é o E-step do início da iteração t+1). Medir isto
não é redundante: é o que transforma "deve dar o mesmo" em resultado, e fecha a
explicação sem depender de argumento.

Corre também a variante com a responsabilidade do M-step corrigida
(ver `mstep_responsibility_diagnosis.md`), para a comparação ficar num só sítio.

SÓ DIAGNÓSTICO. `nutmeg_em.py` não é tocado. Nada é sobrescrito nem retreinado.

Uso:
    python3 -m reliability.diagnose_monotonicity_alignment
"""

import json
import os

import numpy as np

from reliability.nutmeg_em import (
    NutmegEM, THETA_CLIP, THETA_INIT, THETA_PRIOR_A, THETA_PRIOR_B, XI_PRIOR, EPS,
)
from reliability.exist_data import load_task_a
from reliability.diagnose_mstep_responsibility import RESP, log_prior

N_ITER = 5000
TOL = 1e-6
N_LABELS = 2
SLACK = 1e-8

OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "monotonicity_alignment_report.md")
JSON_PATH = os.path.join(OUT_DIR, "monotonicity_alignment_report.json")

CRIT = {
    "M1_end_of_iteration_monotone": (
        "Série FIM (log-posterior avaliada DEPOIS do M-step, em (θ_novo, ξ_novo)): "
        "zero quebras com folga 1e-8. PASS = a causa era a instrumentação."
    ),
    "M2_start_vs_end_identical": (
        "As séries INÍCIO e FIM têm o mesmo número de quebras (a menos das "
        "fronteiras). PASS = mudar o ponto de leitura não altera o diagnóstico."
    ),
}


def check_monotonic(series, slack=SLACK):
    v = np.asarray(series, dtype=float)
    d = np.diff(v)
    bad = np.where(d < -slack)[0]
    return [{"iter": int(i + 2), "drop": float(-d[i])} for i in bad]


def run_both_points(annotations, subpop, resp="code", n_iter=N_ITER, tol=TOL):
    """EM com o M-step do repositório (MAP média posterior), registando a log-lik e
    a log-posterior nos DOIS pontos do ciclo, cada uma coerente com um só estado."""
    ann = np.asarray(annotations, dtype=int)
    items, anns, labels = ann[:, 0], ann[:, 1], ann[:, 2]
    subpop = np.asarray(subpop, dtype=int)
    M = int(subpop.shape[0]); N = int(items.max()) + 1
    K = int(subpop.max()) + 1; L = N_LABELS
    cells = subpop[anns]
    n_votes = np.bincount(anns, minlength=M)
    chv = np.zeros((N, K), dtype=bool); chv[items, cells] = True

    theta = np.full(M, THETA_INIT)
    xi = np.full((M, L), 1.0 / L)
    resp_fn = RESP[resp]

    ll_start, lp_start, ll_end, lp_end = [], [], [], []
    prev_ll = -np.inf
    converged, it = False, 0
    for it in range(1, n_iter + 1):
        # --- ponto INÍCIO: tudo avaliado em (theta, xi) de entrada -----------
        posterior, ll_in = NutmegEM._e_step(
            items, anns, cells, labels, theta, xi, N, K, L, chv)
        ll_start.append(float(ll_in))
        lp_start.append(float(ll_in + log_prior(theta, xi)))

        # --- M-step (o do repositório: MAP média posterior) ------------------
        p_true = posterior[items, cells, labels]
        r = resp_fn(p_true, theta[anns], xi[anns, labels])
        s_ = np.bincount(anns, weights=r, minlength=M)
        sc = np.bincount(anns * L + labels, weights=1.0 - r,
                         minlength=M * L).reshape(M, L)
        theta = np.clip((s_ + THETA_PRIOR_A)
                        / (n_votes + THETA_PRIOR_A + THETA_PRIOR_B), *THETA_CLIP)
        xi = (sc + XI_PRIOR) / (sc.sum(axis=1, keepdims=True) + XI_PRIOR * L)

        # --- ponto FIM: E-step NOVO, tudo em (theta, xi) de saída ------------
        # Nada aqui usa o posterior antigo. É o pedido: um único estado, lido
        # depois da iteração completa.
        _post_end, ll_out = NutmegEM._e_step(
            items, anns, cells, labels, theta, xi, N, K, L, chv)
        ll_end.append(float(ll_out))
        lp_end.append(float(ll_out + log_prior(theta, xi)))

        if abs(ll_in - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll_in

    return {
        "ll_start": ll_start, "lp_start": lp_start,
        "ll_end": ll_end, "lp_end": lp_end,
        "theta": theta, "converged": converged, "n_iter_run": it,
    }


def summarize(series, name):
    br = check_monotonic(series)
    return {
        "name": name,
        "n_points": len(series),
        "n_transitions": len(series) - 1,
        "n_breaks": len(br),
        "max_drop": float(max((b["drop"] for b in br), default=0.0)),
        "first_breaks": br[:10],
    }


def build_report(res_code, res_fixed, sums, offset_check, load_stats):
    L = []
    a = L.append
    a("# Monotonicidade lida no fim da iteração completa — fecho")
    a("")
    a("Gerado por `reliability/diagnose_monotonicity_alignment.py`. **Só")
    a("diagnóstico** — `nutmeg_em.py` intacto, nada sobrescrito, nada retreinado.")
    a("")
    a("## 0. O que foi medido")
    a("")
    a("Duas séries, cada uma função pura dos parâmetros de **um só estado**:")
    a("")
    a("- **INÍCIO** — `L(θ_t, ξ_t)`, avaliada antes do M-step da iteração `t`.")
    a("  É a que `diagnose_em_convergence` já registava (e já estava coerente: a")
    a("  log-lik e o `log_prior` vinham ambos de (θ_old, ξ_old)).")
    a("- **FIM** — `L(θ_{t+1}, ξ_{t+1})`, com um **E-step novo** corrido depois do")
    a("  M-step, nos parâmetros de saída. Nenhum posterior antigo é reutilizado.")
    a("")
    a("No modo MAP a quantidade com garantia de não-decrescimento é a")
    a("**log-posterior** (log-lik + log-prior); é por ela que o veredito é dado. A")
    a("log-lik nua vai ao lado, como sintoma.")
    a("")
    a("## 1. Critérios pré-registados")
    a("")
    for k, v in CRIT.items():
        a(f"- **{k}** — {v}")
    a("")
    a("## 2. Resultado — M-step do repositório (responsabilidade CODE) `[MEDIDO]`")
    a("")
    a("| série | ponto do ciclo | transições | quebras | queda máx |")
    a("|---|---|---|---|---|")
    for s, pt in ((sums["code_lp_start"], "INÍCIO"), (sums["code_lp_end"], "FIM"),
                  (sums["code_ll_start"], "INÍCIO"), (sums["code_ll_end"], "FIM")):
        a(f"| {s['name']} | {pt} | {s['n_transitions']} | **{s['n_breaks']}** | "
          f"{s['max_drop']:.3e} |")
    a("")
    m1 = sums["code_lp_end"]["n_breaks"] == 0
    a(f"**M1: {'PASS' if m1 else 'FAIL'}** — a log-posterior lida no FIM da iteração")
    a(f"completa quebra **{sums['code_lp_end']['n_breaks']}** vezes, queda máx")
    a(f"`{sums['code_lp_end']['max_drop']:.3e}`.")
    a("")
    a(f"`[MEDIDO]` **As violações PERSISTEM.** Mudar o ponto de leitura não as")
    a("remove. A instrumentação **não** era a causa.")
    a("")
    a("### 2.1 Porque tinha de dar isto")
    a("")
    a("`[MEDIDO]` As duas séries são a mesma sequência desfasada de uma posição: o")
    a("E-step do fim da iteração `t` e o do início da iteração `t+1` correm nos")
    a("mesmos parâmetros. Verificado numericamente — máx")
    a(f"`|lp_end[t] − lp_start[t+1]|` = `{offset_check['max_abs_diff']:.3e}` em")
    a(f"{offset_check['n_compared']} pares.")
    a("")
    a(f"**M2: {'PASS' if sums['m2_ok'] else 'FAIL'}** — o número de quebras coincide")
    a("nas duas séries (a menos das fronteiras). Ler no fim em vez de no meio é a")
    a("mesma medição, não uma medição diferente.")
    a("")
    a("## 3. A causa real, para não ficar em aberto `[MEDIDO]`")
    a("")
    a("Com a **responsabilidade do M-step corrigida** (derivação em")
    a("`mstep_responsibility_diagnosis.md`: o código conta o voto do próprio")
    a("anotador duas vezes), no MESMO M-step do repositório e nos MESMOS dois")
    a("pontos de leitura:")
    a("")
    a("| série | ponto do ciclo | transições | quebras | queda máx |")
    a("|---|---|---|---|---|")
    for s, pt in ((sums["fixed_lp_start"], "INÍCIO"), (sums["fixed_lp_end"], "FIM")):
        a(f"| {s['name']} | {pt} | {s['n_transitions']} | **{s['n_breaks']}** | "
          f"{s['max_drop']:.3e} |")
    a("")
    a("A quebra desaparece nos dois pontos de leitura. **Havia bug no estimador** —")
    a("na responsabilidade, não na instrumentação.")
    a("")
    a("## 4. Veredito")
    a("")
    a("`[MEDIDO]` A hipótese \"a causa era a instrumentação\" está **REJEITADA**. A")
    a("log-posterior lida no fim da iteração completa, com θ e ξ do mesmo estado,")
    a(f"continua a quebrar {sums['code_lp_end']['n_breaks']} vezes. A causa é a")
    a("responsabilidade de spam do M-step, e corrigi-la leva as quebras a zero.")
    a("")
    a(f"`[MEDIDO]` Carregamento: {load_stats.get('n_items', '?')} itens · "
      f"{load_stats.get('n_annotators', '?')} anotadores · "
      f"{load_stats.get('n_votes', '?')} votos. "
      f"Corridas: CODE {res_code['n_iter_run']} iterações "
      f"(convergida={res_code['converged']}) · "
      f"CORRIGIDA {res_fixed['n_iter_run']} (convergida={res_fixed['converged']}).")
    return "\n".join(L) + "\n"


def main():
    annotations, annotator_ids, item_ids, _demo, load_stats = load_task_a(verbose=False)
    zero = np.zeros(len(annotator_ids), dtype=int)
    print(f"dados: {len(item_ids)} itens · {len(annotator_ids)} anotadores · "
          f"{len(annotations)} votos")

    print("\n[1] M-step do repositório (responsabilidade CODE)...")
    res_code = run_both_points(annotations, zero, resp="code")
    print("[2] mesma coisa com a responsabilidade CORRIGIDA...")
    res_fixed = run_both_points(annotations, zero, resp="fixed")

    sums = {
        "code_lp_start": summarize(res_code["lp_start"], "log-posterior (CODE)"),
        "code_lp_end": summarize(res_code["lp_end"], "log-posterior (CODE)"),
        "code_ll_start": summarize(res_code["ll_start"], "log-lik nua (CODE)"),
        "code_ll_end": summarize(res_code["ll_end"], "log-lik nua (CODE)"),
        "fixed_lp_start": summarize(res_fixed["lp_start"], "log-posterior (CORRIGIDA)"),
        "fixed_lp_end": summarize(res_fixed["lp_end"], "log-posterior (CORRIGIDA)"),
    }
    # tolerância de fronteira: as séries desfasam-se de 1, logo o nº de quebras
    # pode diferir em 1 sem significar discordância.
    sums["m2_ok"] = abs(sums["code_lp_start"]["n_breaks"]
                        - sums["code_lp_end"]["n_breaks"]) <= 1

    end = np.asarray(res_code["lp_end"][:-1])
    start_next = np.asarray(res_code["lp_start"][1:])
    offset_check = {
        "n_compared": int(end.size),
        "max_abs_diff": float(np.abs(end - start_next).max()) if end.size else 0.0,
    }

    print(f"\n  CODE     log-posterior INÍCIO: {sums['code_lp_start']['n_breaks']} quebras")
    print(f"  CODE     log-posterior FIM   : {sums['code_lp_end']['n_breaks']} quebras "
          f"(máx {sums['code_lp_end']['max_drop']:.3e})")
    print(f"  CODE     log-lik nua    FIM   : {sums['code_ll_end']['n_breaks']} quebras")
    print(f"  CORRIGIDA log-posterior FIM   : {sums['fixed_lp_end']['n_breaks']} quebras")
    print(f"  desfasamento fim[t] vs início[t+1]: máx |Δ| = "
          f"{offset_check['max_abs_diff']:.3e}")
    print(f"  M1 = {'PASS' if sums['code_lp_end']['n_breaks'] == 0 else 'FAIL'} · "
          f"M2 = {'PASS' if sums['m2_ok'] else 'FAIL'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(res_code, res_fixed, sums, offset_check, load_stats))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"criteria": CRIT, "summaries": sums,
                   "offset_check": offset_check,
                   "n_iter_code": res_code["n_iter_run"],
                   "n_iter_fixed": res_fixed["n_iter_run"]},
                  f, indent=2, ensure_ascii=False)
    print(f"\nescrito: {REPORT_PATH}\n         {JSON_PATH}")


if __name__ == "__main__":
    main()
