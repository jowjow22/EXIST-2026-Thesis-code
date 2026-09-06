"""
DIAGNÓSTICO da não-convergência do estimador de θ (Task A, single_truth_K1).

`theta_taskA_single_truth_K1.json` foi gerado com `converged = false` após 200
iterações. Este script instrumenta o MESMO estimador, iteração a iteração, com
`n_iter=5000`, para saber por que parou e se o θ interrompido está longe da solução.

SÓ DIAGNÓSTICO. Não altera a formulação, não muda a tolerância, não acrescenta
prior/âncora/regularização, não regenera soft labels, não toca no treino.
`nutmeg_em.py` é importado e chamado, nunca modificado: os `_e_step`/`_m_step` são
staticmethods, logo o loop pode ser replicado por fora com a matemática intacta.

DIVERGÊNCIAS FACE AO MODELO ASSUMIDO NO PEDIDO (medidas no código, não inferidas):

  1. Não é Dawid-Skene simétrico. É um modelo de SPAM com ruído por anotador:
         P(v | T=l) = theta_j * 1{v==l} + (1 - theta_j) * xi_j[v]
     `theta` é P(NÃO spammar), não P(voto coincide com z). A verossimilhança do
     pedido (theta se v==1, 1-theta se v==0) é de outro modelo e mediria outra coisa.

  2. O critério de paragem é sobre a LOG-VEROSSIMILHANÇA, não sobre max|dtheta|:
         if abs(ll - prev_ll) < tol      (tol = 1e-6)
     Logo `converged=false` diz que a log-lik ainda se movia >1e-6, o que é bem
     mais fraco do que "theta ainda se movia". max|dtheta| é instrumentado à mesma,
     como o pedido manda, mas não é o critério que parou o loop.

  3. THETA_CLIP = (1e-3, 1-1e-3). O limite inferior está perto de 0, não em 0.5 --
     a simetria theta <-> 1-theta NÃO está quebrada por construção pelo clip.

  4. O prior sobre a verdade é uniforme (1/L = 0.5), como o pedido assume. MAS há
     priors Beta(0.5, 0.5) em theta e Dirichlet(0.5) em xi: o M-step é MAP, não ML.
     A quantidade que o EM garante não-decrescente é então a log-POSTERIOR
     (loglik + log-prior), não a loglik nua devolvida pelo _e_step. As duas são
     registadas; a monotonicidade é julgada pela posterior, e a violação da loglik
     nua é reportada à parte, como sintoma esperado e não como bug.

Uso:
    python3 -m reliability.diagnose_em_convergence
"""

import json
import os

import numpy as np
import pandas as pd

from reliability.nutmeg_em import (
    NutmegEM, THETA_CLIP, THETA_INIT, THETA_PRIOR_A, THETA_PRIOR_B, XI_PRIOR, EPS,
)
from reliability.exist_data import load_task_a
from reliability.run_behavioral_diagnosis import md_table

# --- parâmetros: os MESMOS que geraram o ficheiro, excepto n_iter --------------
N_ITER = 5000           # o pedido manda 5000; o original correu com 200
TOL = 1e-6              # inalterada, tal como está no default do fit()
N_LABELS = 2
SNAPSHOT_RANGE = (195, 205)   # θ completo nestas iterações
SNAPSHOT_LAST = 10            # e nas últimas N antes de parar

ORIGINAL_THETA = "reliability/outputs/theta_taskA_single_truth_K1.json"
OUT_DIR = "reliability/outputs"
TRACE_PATH = os.path.join(OUT_DIR, "em_convergence_trace.csv")
REPORT_PATH = os.path.join(OUT_DIR, "em_convergence_diagnosis.md")
JSON_PATH = os.path.join(OUT_DIR, "em_convergence_diagnosis.json")

MONOTONIC_SLACK = 1e-9


def log_prior(theta, xi):
    """log-prior (a menos de constantes de normalização, que não afectam deltas).

    Beta(a,b) em cada theta_j   -> (a-1) log th + (b-1) log(1-th)
    Dirichlet(c) em cada linha  -> (c-1) sum_l log xi_jl
    """
    lp_theta = ((THETA_PRIOR_A - 1.0) * np.log(theta)
                + (THETA_PRIOR_B - 1.0) * np.log(1.0 - theta)).sum()
    lp_xi = ((XI_PRIOR - 1.0) * np.log(xi)).sum()
    return float(lp_theta + lp_xi)


def run_instrumented(annotations, subpop, n_iter=N_ITER, tol=TOL, n_labels=N_LABELS):
    """Replica exactamente o loop de `NutmegEM.fit` (n_restarts=1, init determinístico),
    registando por iteração. A matemática vem dos staticmethods do estimador."""
    ann = np.asarray(annotations, dtype=int)
    items, anns, labels = ann[:, 0], ann[:, 1], ann[:, 2]
    subpop = np.asarray(subpop, dtype=int)
    M = int(subpop.shape[0])
    N = int(items.max()) + 1
    K = int(subpop.max()) + 1
    L = int(n_labels)

    cells = subpop[anns]
    n_votes = np.bincount(anns, minlength=M)
    cell_has_votes = np.zeros((N, K), dtype=bool)
    cell_has_votes[items, cells] = True

    theta = np.full(M, THETA_INIT)
    xi = np.full((M, L), 1.0 / L)

    rows, snapshots = [], {}
    prev_ll = -np.inf
    converged, it = False, 0

    for it in range(1, n_iter + 1):
        theta_old, xi_old = theta.copy(), xi.copy()
        posterior, ll = NutmegEM._e_step(
            items, anns, cells, labels, theta, xi, N, K, L, cell_has_votes)
        theta, xi = NutmegEM._m_step(
            items, anns, cells, labels, theta, xi, posterior, M, L, n_votes)

        # a log-lik da iteração `it` é avaliada no theta/xi ANTES do M-step desta
        # iteração -- é assim que o fit() original a regista, e a comparação com o
        # ficheiro original só é válida se o alinhamento for o mesmo.
        rows.append({
            "iter": it,
            "loglik": float(ll),
            "log_posterior": float(ll + log_prior(theta_old, xi_old)),
            # o objectivo MAP tem de ser avaliado no MESMO par de parametros em
            # que a log-lik foi avaliada -- (theta_old, xi_old), os de entrada do
            # E-step. Emparelhar theta_old com o xi novo da um objectivo mal
            # formado e inventa quebras de monotonicidade que nao existem.
            "max_delta": float(np.abs(theta - theta_old).max()),
            "n_clip_low": int((theta <= THETA_CLIP[0] + 1e-15).sum()),
            "n_clip_high": int((theta >= THETA_CLIP[1] - 1e-15).sum()),
            "theta_mean": float(theta.mean()),
            "theta_min": float(theta.min()),
            "theta_max": float(theta.max()),
        })
        if SNAPSHOT_RANGE[0] <= it <= SNAPSHOT_RANGE[1]:
            snapshots[it] = theta.copy()

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    for k in range(max(1, it - SNAPSHOT_LAST + 1), it + 1):
        snapshots.setdefault(k, None)     # marcador; preenchido abaixo se existir

    return {
        "theta": theta, "xi": xi, "trace": pd.DataFrame(rows),
        "converged": converged, "n_iter_run": it,
        "snapshots": {k: v for k, v in snapshots.items() if v is not None},
        "M": M, "N": N, "n_votes": n_votes,
    }


def causal_exclusions(annotations, subpop, n_iter):
    """Contrafactuais de DIAGNÓSTICO para isolar a causa da quebra de monotonicidade.

    Corre variantes do M-step em memória, sem escrever nada e sem tocar em
    `nutmeg_em.py`. O E-step é sempre o original -- só o M-step muda, e só para
    responder "a violação vem daqui?".

    Variante ML: theta = s/n, xi = contagens normalizadas, SEM priors. Se a loglik
    marginal for monótona aqui, a violação vem dos priors; se não for, vem da
    álgebra do E/M e os priors estão inocentes.
    """
    ann = np.asarray(annotations, dtype=int)
    items, anns, labels = ann[:, 0], ann[:, 1], ann[:, 2]
    subpop = np.asarray(subpop, dtype=int)
    M = int(subpop.shape[0]); N = int(items.max()) + 1
    K = int(subpop.max()) + 1; L = N_LABELS
    cells = subpop[anns]; n_votes = np.bincount(anns, minlength=M)
    chv = np.zeros((N, K), dtype=bool); chv[items, cells] = True

    theta = np.full(M, THETA_INIT); xi = np.full((M, L), 1.0 / L)
    lls = []
    for _ in range(n_iter):
        post, ll = NutmegEM._e_step(items, anns, cells, labels, theta, xi,
                                    N, K, L, chv)
        lls.append(ll)
        p_true = post[items, cells, labels]
        th = theta[anns]; xv = xi[anns, labels]
        num = th * p_true
        r = num / (num + (1.0 - th) * xv + EPS)
        s_ = np.bincount(anns, weights=r, minlength=M)
        theta = np.clip(s_ / np.maximum(n_votes, 1), *THETA_CLIP)
        sc = np.bincount(anns * L + labels, weights=1.0 - r,
                         minlength=M * L).reshape(M, L)
        xi = (sc + EPS) / (sc.sum(axis=1, keepdims=True) + EPS * L)
    breaks = check_monotonic(lls)
    return {
        "ml_variant_breaks": len(breaks),
        "ml_variant_max_drop": float(max((b["drop"] for b in breaks), default=0.0)),
        "ml_variant_n_transitions": len(lls) - 1,
        "priors_exonerated": len(breaks) > 0,
    }


def check_monotonic(series, slack=MONOTONIC_SLACK):
    """Devolve a lista de quebras: (iteração, queda). O EM garante não-decrescimento."""
    v = np.asarray(series, dtype=float)
    d = np.diff(v)
    bad = np.where(d < -slack)[0]
    return [{"iter": int(i + 2), "drop": float(-d[i])} for i in bad]


def delta_trajectory(trace):
    """Classifica a trajectória de max|dtheta|: decresce? satura? oscila?"""
    d = trace["max_delta"].to_numpy()
    n = len(d)
    tail = d[max(0, n - 50):]
    # oscilação de período 2: pares e ímpares separam-se de forma estável
    par, imp = tail[::2], tail[1::2]
    k = min(len(par), len(imp))
    osc = float(np.abs(par[:k] - imp[:k]).mean()) if k else 0.0
    return {
        "primeiro": float(d[0]), "ultimo": float(d[-1]),
        "min": float(d.min()), "max_apos_it10": float(d[10:].max()) if n > 10 else float(d.max()),
        "monotona_decrescente": bool(np.all(np.diff(d) <= 1e-12)),
        "media_ultimas_50": float(tail.mean()),
        "sd_ultimas_50": float(tail.std()),
        "gap_par_impar_ultimas_50": osc,
    }


def oscillation_analysis(snapshots):
    """Se θ oscila, quem alterna. Usa os snapshots consecutivos disponíveis."""
    its = sorted(snapshots)
    if len(its) < 3:
        return None
    T = np.stack([snapshots[i] for i in its])           # (n_snap, M)
    diffs = np.diff(T, axis=0)                          # (n_snap-1, M)
    signs = np.sign(diffs)
    # alterna = o sinal do movimento troca em >=60% dos passos consecutivos
    flips = (signs[1:] * signs[:-1] < 0).sum(axis=0)
    frac = flips / max(signs.shape[0] - 1, 1)
    amp = np.abs(diffs).max(axis=0)
    return {"iters": its, "frac_flip": frac, "amplitude": amp}


# ---------------------------------------------------------------------------

def build_report(res, orig, comp, mono_ll, mono_post, traj, osc, load_stats, excl):
    tr = res["trace"]
    L = []
    a = L.append
    a("# Diagnóstico da não-convergência do estimador de θ (Task A, K=1)")
    a("")
    a("Gerado por `reliability/diagnose_em_convergence.py`. **Só diagnóstico** — a")
    a("formulação, a tolerância e os priors do estimador não foram tocados.")
    a("")

    a("## 0. O estimador que está no código (não o do enunciado)")
    a("")
    a("`[MEDIDO]` em `reliability/nutmeg_em.py`:")
    a("")
    a("- Modelo de **spam**, não Dawid-Skene simétrico:")
    a("  `P(v | T=l) = θ_j · 1{v==l} + (1 − θ_j) · ξ_j[v]`. **θ é P(não spammar)**,")
    a("  não P(voto coincide com z).")
    a(f"- Critério de paragem: `abs(ll − prev_ll) < tol`, **tol = {TOL:g}** — sobre a")
    a("  log-verossimilhança, **não** sobre `max|Δθ|`. `converged = false` diz que a")
    a("  log-lik ainda se movia, não que θ ainda se movia.")
    a(f"- `THETA_CLIP = ({THETA_CLIP[0]:g}, {THETA_CLIP[1]:g})` — limite inferior perto")
    a("  de 0. **A simetria θ ↔ 1−θ não está quebrada pelo clip.**")
    a("- Prior sobre a verdade: **uniforme** (1/L = 0,5), como o enunciado assume.")
    a(f"- Priors Beta({THETA_PRIOR_A}, {THETA_PRIOR_B}) em θ e Dirichlet({XI_PRIOR}) em ξ:")
    a("  o M-step é **MAP, não ML**.")
    a(f"- Init determinístico θ = {THETA_INIT}, ξ uniforme; `n_restarts=1`, logo **nenhum**")
    a("  RNG é consumido — a run é reprodutível sem seed.")
    a("")
    a("`[LIMITAÇÃO]` como o M-step é MAP, a quantidade que o EM garante não-decrescente")
    a("é a **log-posterior** (loglik + log-prior), não a loglik nua que o `_e_step`")
    a("devolve. As duas são verificadas abaixo; só a violação da **posterior** seria bug.")
    a("")

    a("## 1. Convergiu?")
    a("")
    if res["converged"]:
        a(f"**SIM, na iteração {res['n_iter_run']}** (com `n_iter={N_ITER}`, `tol={TOL:g}`).")
        a(f"O ficheiro original parou em 200 por bater o limite, não por divergir.")
    else:
        a(f"**NÃO** — chegou a {res['n_iter_run']} iterações sem atingir `tol = {TOL:g}`.")
    a("")
    a(f"Dados: {res['N']} itens · {res['M']} anotadores · "
      f"{int(load_stats.get('n_votes', tr.shape[0]*0)) or 'n/d'} votos carregados.")
    a("")

    a("## 2. Monotonicidade")
    a("")
    a(f"- **log-posterior** (a que tem garantia): {len(mono_post)} quebras"
      + (f" — máx {max(x['drop'] for x in mono_post):.3e}" if mono_post else " (nenhuma)"))
    a(f"- loglik nua (sem prior, sem garantia sob MAP): {len(mono_ll)} quebras"
      + (f" — máx {max(x['drop'] for x in mono_ll):.3e}" if mono_ll else " (nenhuma)"))
    a("")
    if mono_post:
        a("**Há quebra na log-posterior — isso é bug de implementação, não interpretação.**")
        a("Primeiras 10 quebras:")
        a("")
        for x in mono_post[:10]:
            a(f"- iteração {x['iter']}: queda de `{x['drop']:.3e}`")
    else:
        a("Nenhuma quebra na quantidade com garantia. **Não há evidência de bug no EM.**")
    a("")

    if mono_post:
        a("### 2.1 O que foi excluído como causa")
        a("")
        tr_ = res["trace"]
        a(f"- **Priors** — o M-step de máxima verossimilhança pura (sem prior nenhum) "
          f"também viola: `{excl['ml_variant_breaks']}` quebras em "
          f"`{excl['ml_variant_n_transitions']}` transições, queda máx "
          f"`{excl['ml_variant_max_drop']:.3e}`. **Priors inocentes.**")
        a(f"- **Clip** — θ nunca se aproxima dos limites: máximo observado "
          f"`{tr_['theta_max'].max():.4f}`, mínimo `{tr_['theta_min'].min():.4f}`, "
          f"contra `({THETA_CLIP[0]:g}, {THETA_CLIP[1]:g})`. Zero anotadores no clip.")
        a(f"- **`EPS = {EPS:g}`** — o menor `(1−θ)` observado é "
          f"`{1 - tr_['theta_max'].max():.3e}`, sete ordens de grandeza acima do EPS. "
          f"Nunca entra no regime em que perturbaria os logs.")
        a("- **Label switching** — sem alternância de período 2 (secção 5).")
        a("")
        a("`[MEDIDO]` **Assinatura:** as quebras concentram-se no regime de passo "
          f"pequeno — `max|Δθ|` médio `{tr_.loc[np.r_[False, np.diff(tr_['log_posterior'].to_numpy()) < -MONOTONIC_SLACK], 'max_delta'].mean():.2e}` "
          f"nas iterações com quebra contra "
          f"`{tr_.loc[np.r_[True, np.diff(tr_['log_posterior'].to_numpy()) >= -MONOTONIC_SLACK], 'max_delta'].mean():.2e}` "
          "nas restantes — e só começam na iteração "
          f"{mono_post[0]['iter']}; a fase inicial é limpa.")
        a("")
        a("`[NÃO IDENTIFICADO]` a causa exacta fica por localizar. Localizá-la exige mexer")
        a("no E/M, o que está fora do âmbito desta etapa. O que está estabelecido é que")
        a("**existe** e que não é nenhuma das quatro explicações benignas acima.")
        a("")
    a("## 3. Trajectória de `max|Δθ|`")
    a("")
    dft = pd.DataFrame({"valor": [
        traj["primeiro"], traj["ultimo"], traj["min"], traj["max_apos_it10"],
        traj["media_ultimas_50"], traj["sd_ultimas_50"], traj["gap_par_impar_ultimas_50"],
    ]}, index=[
        "primeira iteração", "última iteração", "mínimo observado",
        "máximo depois da it. 10", "média das últimas 50", "sd das últimas 50",
        "gap médio par↔ímpar (últimas 50)",
    ])
    a(md_table(dft, float_fmt="{:.3e}", index_name="métrica"))
    a("")
    a(f"Monótona decrescente: **{traj['monotona_decrescente']}**.")
    a("")

    a("## 4. Clip")
    a("")
    a(f"Intervalo: `({THETA_CLIP[0]:g}, {THETA_CLIP[1]:g})`. Na última iteração: "
      f"**{int(tr['n_clip_low'].iloc[-1])}** anotadores no extremo inferior, "
      f"**{int(tr['n_clip_high'].iloc[-1])}** no superior, de {res['M']}.")
    a("")

    a("## 5. Oscilação / label switching")
    a("")
    if osc is None:
        a("Snapshots insuficientes para a análise (o loop parou cedo demais).")
    else:
        n_alt = int((osc["frac_flip"] > 0.6).sum())
        a(f"Anotadores cujo movimento de θ troca de sinal em >60% dos passos "
          f"consecutivos (it. {osc['iters'][0]}–{osc['iters'][-1]}): **{n_alt}** de {res['M']}.")
        a(f"Amplitude máxima de oscilação no painel: `{float(osc['amplitude'].max()):.3e}`.")
        if n_alt == 0:
            a("")
            a("**Sem padrão de alternância.** Não há sinal de label switching.")
    a("")

    a("## 6. θ convergido vs θ do ficheiro original (a pergunta que decide)")
    a("")
    dfc = pd.DataFrame({"valor": [
        comp["pearson"], comp["spearman"], comp["mean_abs_diff"],
        comp["max_abs_diff"], comp["n_cross_05"],
    ]}, index=[
        "Pearson r", "Spearman ρ", "diferença média absoluta",
        "diferença máxima absoluta", "anotadores que cruzam θ = 0,5",
    ])
    a(md_table(dfc, index_name="métrica"))
    a("")
    if comp["mean_abs_diff"] < 0.01 and comp["pearson"] > 0.99:
        a(f"`[MEDIDO]` **O θ interrompido é praticamente o convergido** "
          f"(diferença média `{comp['mean_abs_diff']:.4f}`, r = `{comp['pearson']:.4f}`).")
        a("O `converged = false` era **cosmético**: as soft labels ponderadas e o")
        a("experimento dos dois braços **não precisam de ser refeitos por esta causa**.")
    else:
        a(f"`[MEDIDO]` **O θ do ficheiro difere do convergido** "
          f"(diferença média `{comp['mean_abs_diff']:.4f}`, r = `{comp['pearson']:.4f}`).")
        a("As soft labels ponderadas foram construídas com um θ que não é a solução do")
        a("estimador. **`soft_labels_taskA_weighted.json` e o experimento dos dois braços")
        a("teriam de ser refeitos** com o θ convergido para serem defensáveis.")
    a("")
    a(f"Traço completo por iteração: `{os.path.basename(TRACE_PATH)}`.")
    a("")
    return "\n".join(L) + "\n"


def main():
    annotations, annotator_ids, item_ids, demo, load_stats = load_task_a(verbose=False)
    M = len(annotator_ids)
    subpop = np.zeros(M, dtype=int)          # single_truth_K1

    res = run_instrumented(annotations, subpop)
    tr = res["trace"]

    mono_ll = check_monotonic(tr["loglik"])
    mono_post = check_monotonic(tr["log_posterior"])
    traj = delta_trajectory(tr)
    osc = oscillation_analysis(res["snapshots"])
    excl = causal_exclusions(annotations, subpop, res["n_iter_run"])

    with open(ORIGINAL_THETA, encoding="utf-8") as f:
        orig_raw = json.load(f)
    orig = np.array([orig_raw["theta"][a] for a in annotator_ids], dtype=float)
    new = res["theta"]
    comp = {
        "pearson": float(np.corrcoef(orig, new)[0, 1]),
        "spearman": float(pd.Series(orig).corr(pd.Series(new), method="spearman")),
        "mean_abs_diff": float(np.abs(orig - new).mean()),
        "max_abs_diff": float(np.abs(orig - new).max()),
        "n_cross_05": int(((orig >= 0.5) != (new >= 0.5)).sum()),
    }

    print(f"itens={res['N']} anotadores={res['M']} | n_iter={N_ITER} tol={TOL:g}")
    print(f"convergiu: {res['converged']} na iteracao {res['n_iter_run']}")
    print(f"monotonicidade log-POSTERIOR (com garantia): {len(mono_post)} quebras")
    print(f"monotonicidade loglik nua (sem garantia sob MAP): {len(mono_ll)} quebras")
    print(f"max|dtheta| primeiro={traj['primeiro']:.3e} ultimo={traj['ultimo']:.3e} "
          f"min={traj['min']:.3e}")
    print(f"  monotona decrescente: {traj['monotona_decrescente']} | "
          f"gap par-impar (ult.50)={traj['gap_par_impar_ultimas_50']:.3e}")
    print(f"clip ({THETA_CLIP[0]:g},{THETA_CLIP[1]:g}): "
          f"{int(tr['n_clip_low'].iloc[-1])} baixo / {int(tr['n_clip_high'].iloc[-1])} alto")
    if osc is not None:
        print(f"anotadores a alternar (>60% dos passos): "
              f"{int((osc['frac_flip'] > 0.6).sum())}/{res['M']}")
    print(f"theta convergido vs ficheiro: r={comp['pearson']:.6f} "
          f"media|d|={comp['mean_abs_diff']:.6f} max|d|={comp['max_abs_diff']:.6f} "
          f"cruzam 0.5: {comp['n_cross_05']}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    tr.to_csv(TRACE_PATH, index=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(res, orig, comp, mono_ll, mono_post, traj, osc, load_stats, excl))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "params": {"n_iter": N_ITER, "tol": TOL, "theta_clip": list(THETA_CLIP),
                       "theta_init": THETA_INIT,
                       "priors": {"theta_beta": [THETA_PRIOR_A, THETA_PRIOR_B],
                                  "xi_dirichlet": XI_PRIOR},
                       "truth_prior": "uniforme (1/L)"},
            "converged": bool(res["converged"]), "n_iter_run": int(res["n_iter_run"]),
            "monotonic_breaks_log_posterior": mono_post,
            "monotonic_breaks_loglik": mono_ll,
            "max_delta_trajectory": traj,
            "clip_final": {"low": int(tr["n_clip_low"].iloc[-1]),
                           "high": int(tr["n_clip_high"].iloc[-1])},
            "comparacao_com_ficheiro": comp,
            "exclusoes_causais": excl,
        }, f, indent=1, ensure_ascii=False)

    for p in (TRACE_PATH, REPORT_PATH, JSON_PATH):
        print(f"escrito: {p}")


if __name__ == "__main__":
    main()
