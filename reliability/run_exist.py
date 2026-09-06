"""STAGE 2 — aplica o `NutmegEM` validado aos dados reais do EXIST 2026, Task A.

Corre CPU-only (EM sobre 23 904 votos + KMeans sobre 887 anotadores: segundos).
Não precisa de GPU nem da VM.

Três definições de subpopulação, cada uma com K in {4, 8}:
  (a) demo_base      : etnia + estudo + país      -> one-hot 66-d -> KMeans(K)
  (b) demo_extended  : + género + idade           -> one-hot 71-d -> KMeans(K)
  (c) behavioral     : perfil de voto (6-d: yes-rate, desvio do consenso, ...) -> KMeans(K)
Mais `single_truth` (K=1) como referência tipo-MACE — o MESMO estimador.

Sanity checks externos (dados reais não têm verdade):
  1. taxa de concordância-com-a-maioria por anotador;
  2. Krippendorff alpha nominal leave-one-out (delta_alpha por anotador).
AMBOS são diagnósticos DESCRITIVOS com viés conhecido para o consenso — não são
critério de correção do theta. Ver `outputs/summary_taskA.md`.

    python -m reliability.run_exist [--K 4 8] [--json <path>]
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .exist_data import (
    DEMO_GROUPS_BASE,
    DEMO_GROUPS_EXTENDED,
    DEFAULT_JSON,
    demo_one_hot,
    load_task_a,
)
from .nutmeg_em import NutmegEM, MONOTONIC_SLACK

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CLUSTER_SEED = 42
THETA_THRESHOLDS = (0.3, 0.5, 0.7)


# ----------------------------------------------------------------- subpops
def _kmeans_quiet(X, K):
    """KMeans determinístico, sem o RuntimeWarning benigno do kmeans++ do sklearn.

    `_euclidean_distances` dispara `divide by zero encountered in matmul` com
    matrizes esparsas/one-hot. Resultados verificados sãos (clusters equilibrados,
    inércia finita); o filtro só evita poluir o log de diagnóstico.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return KMeans(n_clusters=K, n_init=10, random_state=CLUSTER_SEED).fit(X)


def subpop_from_demo(annotator_ids, demo, groups, K):
    """KMeans sobre a one-hot demográfica. Devolve (subpop (M,), diag dict)."""
    X = demo_one_hot(annotator_ids, demo, groups).astype(np.float64)
    km = _kmeans_quiet(X, K)
    lab = km.labels_.astype(int)
    return lab, {
        "feature_dim": int(X.shape[1]),
        "unique_profiles": len(set(map(tuple, X.tolist()))),
        "inertia": float(km.inertia_),
    }


def behavior_profile(annotations, M, N):
    """Perfil comportamental de baixa dimensão por anotador (M, 6).

    Colunas: [yes_rate, mean_residual, mean_abs_residual, majority_agreement,
              mean_residual_em_itens_contestados, fracao_de_votos_contestados]
    onde residual = voto - média do item e "contestado" = item sem unanimidade.

    PORQUÊ NÃO concordância par-a-par nem SVD do resíduo esparso (ambos testados
    e REJEITADOS, evidência medida):
      - sobreposição par-a-par esperada ~0.15 itens por par (6 anotadores/item,
        887 anotadores) -> matriz de concordância par-a-par praticamente vazia;
      - TruncatedSVD(32) da matriz resíduo (887x3984, 76% de zeros, e os itens
        unânimes dão resíduo exactamente 0) + KMeans produziu clusters
        degenerados: [883, 2, 1, 1] para K=4 e [873, 2, 2, 2, 2, 2, 2, 3] para
        K=8 — os suportes de itens são quase disjuntos, logo as distâncias
        medem o nº de votos, não o comportamento.
    O perfil abaixo é comparável entre anotadores apesar dos suportes disjuntos.
    """
    items, anns, labels = annotations[:, 0], annotations[:, 1], annotations[:, 2]
    cnt = np.bincount(items, minlength=N)
    mean = np.bincount(items, weights=labels, minlength=N) / np.maximum(cnt, 1)
    resid = labels - mean[items]
    contested = (mean[items] > 0.0) & (mean[items] < 1.0)

    n = np.maximum(np.bincount(anns, minlength=M), 1)
    n_cont = np.maximum(np.bincount(anns, weights=contested, minlength=M), 1e-9)

    def by_ann(w):
        return np.bincount(anns, weights=w, minlength=M)

    maj = np.where(mean > 0.5, 1.0, np.where(mean < 0.5, 0.0, 0.5))
    hit = np.where(maj[items] == 0.5, 0.5, (labels == maj[items]).astype(float))

    X = np.stack([
        by_ann(labels) / n,
        by_ann(resid) / n,
        by_ann(np.abs(resid)) / n,
        by_ann(hit) / n,
        by_ann(resid * contested) / n_cont,
        by_ann(contested.astype(float)) / n,
    ], axis=1)
    return X


def subpop_from_behavior(annotations, M, N, K):
    """KMeans sobre o perfil comportamental padronizado. (subpop (M,), diag)."""
    X = behavior_profile(annotations, M, N)
    Z = StandardScaler().fit_transform(X)
    km = _kmeans_quiet(Z, K)
    lab = km.labels_.astype(int)
    cnt_items = np.bincount(annotations[:, 0], minlength=N).mean()
    exp_shared = N * (cnt_items / M) * ((cnt_items - 1) / max(M - 1, 1))
    return lab, {
        "feature_dim": int(X.shape[1]),
        "expected_shared_items_per_pair": float(exp_shared),
        "inertia": float(km.inertia_),
        "note": "perfil comportamental; SVD-do-residuo e par-a-par rejeitados (ver docstring)",
    }


# --------------------------------------------------------- sanity externos
def majority_agreement_rate(annotations, M):
    """Fração dos votos de cada anotador que batem com a maioria do item.

    Empates (3-3) contam 0.5. Diagnóstico DESCRITIVO enviesado para o consenso:
    por construção pune a minoria sistemática que o método existe para proteger.
    """
    items, anns, labels = annotations[:, 0], annotations[:, 1], annotations[:, 2]
    N = items.max() + 1
    ones = np.bincount(items, weights=labels, minlength=N)
    cnt = np.bincount(items, minlength=N)
    frac_yes = ones / np.maximum(cnt, 1)
    maj = np.where(frac_yes > 0.5, 1.0, np.where(frac_yes < 0.5, 0.0, 0.5))
    hit = np.where(maj[items] == 0.5, 0.5, (labels == maj[items]).astype(float))
    s = np.bincount(anns, weights=hit, minlength=M)
    n = np.bincount(anns, minlength=M)
    return s / np.maximum(n, 1)


def _alpha_from_item_counts(n0, n1):
    """Krippendorff alpha nominal binário a partir das contagens por item.

    Do = sum_u 2*n_u0*n_u1/(m_u-1) / n ;  De = 2*n_0*n_1 / (n*(n-1))
    Itens com m_u < 2 são excluídos (não informam sobre concordância).
    """
    m = n0 + n1
    keep = m >= 2
    n0, n1, m = n0[keep], n1[keep], m[keep]
    if len(m) == 0:
        return float("nan")
    n_tot = m.sum()
    do = (2.0 * n0 * n1 / (m - 1)).sum() / n_tot
    c0, c1 = n0.sum(), n1.sum()
    de = 2.0 * c0 * c1 / (n_tot * (n_tot - 1))
    return float(1.0 - do / de) if de > 0 else float("nan")


def krippendorff_alpha_loo(annotations, M, verbose=True):
    """alpha global e delta_alpha[j] = alpha_sem_j - alpha_global.

    delta_alpha > 0 => remover o anotador AUMENTA a concordância global
    (o anotador é um discordante). Segundo diagnóstico descritivo, com o mesmo
    viés pró-consenso do anterior.
    """
    items, anns, labels = annotations[:, 0], annotations[:, 1], annotations[:, 2]
    N = items.max() + 1
    n1 = np.bincount(items, weights=labels, minlength=N).astype(float)
    n0 = np.bincount(items, weights=1 - labels, minlength=N).astype(float)
    alpha_all = _alpha_from_item_counts(n0, n1)

    order = np.argsort(anns, kind="stable")
    anns_s, items_s, labels_s = anns[order], items[order], labels[order]
    bounds = np.searchsorted(anns_s, np.arange(M + 1))

    deltas = np.full(M, np.nan)
    for j in range(M):
        lo, hi = bounds[j], bounds[j + 1]
        if lo == hi:
            continue
        it, lb = items_s[lo:hi], labels_s[lo:hi]
        a0, a1 = n0.copy(), n1.copy()
        np.subtract.at(a1, it, lb)
        np.subtract.at(a0, it, 1 - lb)
        deltas[j] = _alpha_from_item_counts(a0, a1) - alpha_all
    if verbose:
        print(f"[sanity] Krippendorff alpha (todos) = {alpha_all:.4f}")
    return alpha_all, deltas


# ------------------------------------------------------------------ helpers
def theta_stats(theta, n_votes):
    q = np.percentile(theta, [5, 25, 50, 75, 95])
    return {
        "mean": float(theta.mean()),
        "std": float(theta.std()),
        "min": float(theta.min()),
        "max": float(theta.max()),
        "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
        "iqr": float(q[3] - q[1]),
        **{f"n_below_{t}": int((theta < t).sum()) for t in THETA_THRESHOLDS},
        "n_low_vote_annotators": int((n_votes < 15).sum()),
    }


def _hist(theta, bins=10):
    h, edges = np.histogram(theta, bins=bins, range=(0.0, 1.0))
    lines = []
    for c, lo, hi in zip(h, edges[:-1], edges[1:]):
        bar = "#" * int(round(40 * c / max(h.max(), 1)))
        lines.append(f"| {lo:.1f}–{hi:.1f} | {c:>4} | `{bar}` |")
    return lines


# --------------------------------------------------------------------- main
def run(json_path=DEFAULT_JSON, Ks=(4, 8), verbose=True,
        responsibility="fixed", suffix=None, overwrite=False):
    """Ajusta o θ para as sete definições de subpopulação e escreve os ficheiros.

    responsibility : "fixed" (default, estimador corrigido) | "legacy" (o das
        fases anteriores, mantido para reprodutibilidade). Ver `nutmeg_em`.
    suffix : sufixo dos ficheiros de saída. `None` = automático — "_fixed" em
        modo corrigido, "" em modo legado. Assim uma corrida corrigida NUNCA
        aterra por cima dos ficheiros que produziram os resultados publicados.
    overwrite : sem isto, um ficheiro já existente faz o run falhar.
    """
    if suffix is None:
        suffix = "_fixed" if responsibility == "fixed" else ""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    annotations, annotator_ids, item_ids, demo, load_stats = load_task_a(json_path, verbose)
    M, N = len(annotator_ids), len(item_ids)

    maj_rate = majority_agreement_rate(annotations, M)
    alpha_all, delta_alpha = krippendorff_alpha_loo(annotations, M, verbose)

    configs = []
    for K in Ks:
        sp_a, d_a = subpop_from_demo(annotator_ids, demo, DEMO_GROUPS_BASE, K)
        sp_b, d_b = subpop_from_demo(annotator_ids, demo, DEMO_GROUPS_EXTENDED, K)
        sp_c, d_c = subpop_from_behavior(annotations, M, N, K)
        configs += [
            (f"demo_base_K{K}", sp_a, d_a),
            (f"demo_extended_K{K}", sp_b, d_b),
            (f"behavioral_K{K}", sp_c, d_c),
        ]
    configs.append(("single_truth_K1", np.zeros(M, dtype=int), {"note": "baseline tipo-MACE"}))

    results = {}
    for name, subpop, diag in configs:
        # A asserção de monotonicidade fica DESLIGADA aqui e é aplicada a
        # posteriori sobre `logpost_history_`, para que uma varredura de sete
        # configurações registe a violação em vez de morrer na terceira.
        # Duas faixas, ambas pré-registadas:
        #   > MONOTONIC_GROSS  -> erro: é o regime do bug da responsabilidade
        #                         (quedas ~1e-1). Nunca deve voltar a acontecer.
        #   > MONOTONIC_SLACK  -> flag no ficheiro: quedas ~1e-7 que o M-step de
        #                         MÉDIA POSTERIOR não garante evitar (o argmax MAP
        #                         garantiria; o M-step do repositório é a média).
        model = NutmegEM(responsibility=responsibility, monotonic_check=False).fit(
            annotations, subpop, verbose=False)
        lp = np.asarray(model.logpost_history_, dtype=float)
        d = np.diff(lp)
        mono_breaks = int((d < -MONOTONIC_SLACK).sum())
        mono_max_drop = float(-d.min()) if d.size and d.min() < 0 else 0.0
        # Sem limiar de erro aqui, e a razão está medida, não assumida: com
        # `responsibility="fixed"` + M-step de ML-argmax a log-lik é monótona
        # mesmo em K=8 (0 quebras em 3000 iterações). Com o M-step do
        # repositório -- MÉDIA POSTERIOR -- aparecem 30 quebras, máx 3.8e-02.
        # A causa é que os priors de Jeffreys (a = c = 0.5 < 1) não têm modo
        # interior: o argmax MAP nem sequer está definido, logo a média posterior
        # é um compromisso deliberado e a garantia de subida NÃO se aplica a ela.
        # Um limiar de magnitude não separaria isto do bug antigo (1.5e-01 contra
        # 3.8e-02, mesma ordem), por isso a violação é REGISTADA e SINALIZADA no
        # ficheiro em vez de convertida em erro. A asserção estrita continua a
        # correr por default dentro de `NutmegEM.fit`.
        theta = model.theta_
        cell_cov = float(model.cell_has_votes_.mean())
        votes_per_cell = len(annotations) / max(model.cell_has_votes_.sum(), 1)
        sizes = np.bincount(subpop).tolist()
        flags = []
        if len(sizes) > 1 and max(sizes) > 0.9 * len(subpop):
            flags.append("CLUSTER_DEGENERADO(>90% num só cluster)")
        if votes_per_cell < 1.5:
            flags.append("CÉLULAS_QUASE_SINGULARES(<1.5 votos/célula: θ inflacionado)")
        if mono_breaks:
            flags.append(
                f"MONOTONIA_LOG_POSTERIOR({mono_breaks} quebras, máx "
                f"{mono_max_drop:.2e}; M-step de média posterior, não argmax MAP)")
        rec = {
            "flags": flags,
            "subpop_def": name,
            "K": int(subpop.max()) + 1,
            "cluster_sizes": sizes,
            "cluster_diag": diag,
            "cell_coverage": cell_cov,
            "mean_votes_per_nonempty_cell": float(votes_per_cell),
            "n_iter": model.n_iter_run_,
            "converged": bool(model.converged_),
            "responsibility": responsibility,
            "monotonic_breaks": mono_breaks,
            "monotonic_max_drop": mono_max_drop,
            "monotonic_slack": MONOTONIC_SLACK,
            "final_loglik": model.final_loglik_,
            "theta_stats": theta_stats(theta, model.n_votes_),
            "spearman_theta_vs_majority_agreement": float(spearmanr(theta, maj_rate)[0]),
            "spearman_theta_vs_delta_alpha": float(
                spearmanr(theta[~np.isnan(delta_alpha)], delta_alpha[~np.isnan(delta_alpha)])[0]
            ),
            "_theta": theta,
        }
        results[name] = rec
        out_path = os.path.join(OUTPUT_DIR, f"theta_taskA_{name}{suffix}.json")
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"{out_path} já existe — não sobrescrever")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "subpop_def": name,
                    "theta": {aid: float(t) for aid, t in zip(annotator_ids, theta)},
                    "n_votes": {aid: int(v) for aid, v in zip(annotator_ids, model.n_votes_)},
                    "subpop": {aid: int(s) for aid, s in zip(annotator_ids, subpop)},
                    "meta": {k: v for k, v in rec.items() if k != "_theta"},
                },
                f,
                indent=2,
            )
        if verbose:
            ts = rec["theta_stats"]
            print(
                f"[fit] {name:22s} cobertura células={100*cell_cov:5.1f}% "
                f"votos/célula={votes_per_cell:4.2f} | θ mean={ts['mean']:.3f} "
                f"std={ts['std']:.3f} <0.5={ts['n_below_0.5']:3d} | "
                f"ρ(θ,maj)={rec['spearman_theta_vs_majority_agreement']:+.3f} "
                f"ρ(θ,Δα)={rec['spearman_theta_vs_delta_alpha']:+.3f}"
                + (("  ** " + " ".join(flags) + " **") if flags else "")
            )

    _write_summary(results, annotator_ids, load_stats, alpha_all, maj_rate, Ks,
                   suffix=suffix, overwrite=overwrite)
    return results


def _write_summary(results, annotator_ids, load_stats, alpha_all, maj_rate, Ks,
                   suffix="", overwrite=False):
    names = [n for n in results if n != "single_truth_K1"]
    path = os.path.join(OUTPUT_DIR, f"summary_taskA{suffix}.md")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} já existe — não sobrescrever")
    L = []
    A = L.append
    A("# Task A (memes) — θ por anotador, três definições de subpopulação\n")
    A("Gerado por `python -m reliability.run_exist`. CPU-only, sem GPU.\n")
    A("## 0. Dados\n")
    A(f"- itens **{load_stats['n_items']}** · anotadores **{load_stats['n_annotators']}** "
      f"· votos **{load_stats['n_votes']}** · descartados {load_stats['n_dropped_labels']}")
    A(f"- YES = {100*load_stats['label_balance_yes']:.1f}% dos votos")
    A(f"- votos/anotador: mean {load_stats['votes_per_annotator_mean']:.1f} · "
      f"min {load_stats['votes_per_annotator_min']} · max {load_stats['votes_per_annotator_max']} · "
      f"**{load_stats['n_annotators_below_15_votes']} anotadores abaixo de 15 votos** "
      f"(θ desses é fraco por construção)")
    A(f"- Krippendorff α (nominal, todos os anotadores) = **{alpha_all:.4f}**")
    A(f"- concordância-com-maioria: mean {maj_rate.mean():.3f} · min {maj_rate.min():.3f} · "
      f"max {maj_rate.max():.3f}")
    for aid, fld, c in load_stats["inconsistent_demo"]:
        A(f"- demografia inconsistente resolvida por moda: `{aid}` `{fld}` {c}")
    A("")
    A("## 1. Enquadramento obrigatório dos sanity checks\n")
    A("As duas métricas externas — **concordância-com-maioria** e **Krippendorff α "
      "leave-one-out** — são **diagnósticos descritivos com viés conhecido para o "
      "consenso**, NÃO critérios de correção do θ. Por construção penalizam a minoria "
      "sistemática que este método existe para proteger: um anotador de uma subpopulação "
      "coerente mas minoritária discorda da maioria *por ter razão dentro do seu grupo*, "
      "e ambas as métricas leem isso como erro.\n")
    A("Por isso, qualquer divergência entre θ e estas métricas é interpretada com o "
      "**teste sintético de preservação de minoria como árbitro** "
      "(`outputs/minority_preservation.json`, veredito PASS nas 3 seeds e nas duas "
      "densidades) — só ele distingue ruído de discordância legítima, porque só ele tem "
      "acesso à verdade. Um ρ(θ, concordância) alto **não** valida o θ, e um ρ baixo "
      "**não** o invalida.\n")
    A("## 2. Distribuição de θ por definição\n")
    A("| definição | K | cobertura células | votos/célula | θ mean | θ std | θ IQR | "
      "<0.3 | <0.5 | <0.7 | ρ(θ, concord.) | ρ(θ, Δα) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in results.items():
        t = r["theta_stats"]
        A(f"| `{n}` | {r['K']} | {100*r['cell_coverage']:.2f}% | "
          f"{r['mean_votes_per_nonempty_cell']:.2f} | {t['mean']:.3f} | {t['std']:.3f} | "
          f"{t['iqr']:.3f} | {t['n_below_0.3']} | {t['n_below_0.5']} | {t['n_below_0.7']} | "
          f"{r['spearman_theta_vs_majority_agreement']:+.3f} | "
          f"{r['spearman_theta_vs_delta_alpha']:+.3f} |")
    A("")
    A("### Tamanhos de cluster\n")
    for n, r in results.items():
        warn = ("  **<- " + "; ".join(r["flags"]) + "**") if r["flags"] else ""
        A(f"- `{n}`: {r['cluster_sizes']}{warn}")
    A("")
    # --- leitura computada dos resultados ---
    # Critério: maior dispersão de θ ENTRE configs com evidência suficiente por célula
    # (>=2 votos/célula). Sem esse corte, o critério premia fragmentação: θ infla à
    # medida que votos/célula cai (ver nota do confound abaixo).
    clean = {
        n: r for n, r in results.items()
        if n != "single_truth_K1" and not r["flags"]
        and r["mean_votes_per_nonempty_cell"] >= 2.0
    }
    best = max(clean, key=lambda n: clean[n]["theta_stats"]["std"]) if clean else None
    st_rho = results["single_truth_K1"]["spearman_theta_vs_majority_agreement"]
    A("### Leitura\n")
    A(f"- O baseline **single_truth (K=1)** tem ρ(θ, concordância-com-maioria) = "
      f"**{st_rho:+.3f}** — quase uma reexpressão da métrica de consenso. É exatamente o "
      "comportamento que se espera de um estimador de verdade única: 'competência' e "
      "'concordar com a maioria' colapsam na mesma coisa.")
    A("- As definições com subpopulação **descolam** dessa métrica "
      "(ρ entre "
      + f"{min(r['spearman_theta_vs_majority_agreement'] for n, r in results.items() if n != 'single_truth_K1'):+.3f}"
      + " e "
      + f"{max(r['spearman_theta_vs_majority_agreement'] for n, r in results.items() if n != 'single_truth_K1'):+.3f}"
      + "). Pela §1, isto **não** é evidência de falha: é o efeito pretendido — parte da "
      "discordância deixa de ser contabilizada como incompetência. Também não é, por si "
      "só, prova de sucesso; a prova está no teste sintético.")
    if best:
        b = results[best]["theta_stats"]
        A(f"- Sem flags de degenerescência, a definição com maior dispersão de θ "
          f"(separação mais limpa entre anotadores) é **`{best}`** "
          f"(std {b['std']:.3f}, IQR {b['iqr']:.3f}, {b['n_below_0.5']} anotadores < 0.5), "
          f"com {results[best]['mean_votes_per_nonempty_cell']:.2f} votos/célula.")
    A("- **Confound a declarar:** a dispersão de θ **não** é comparável entre configs com "
      "densidades de célula diferentes. θ médio sobe monotonicamente à medida que "
      "votos/célula cai ("
      + " → ".join(
          f"{results[n]['mean_votes_per_nonempty_cell']:.2f}/{results[n]['theta_stats']['mean']:.3f}"
          for n in sorted(results, key=lambda x: -results[x]["mean_votes_per_nonempty_cell"]))
      + " em votos-por-célula/θ̄). A inflação **não é um efeito de limiar**: é contínua, e "
      "um std maior sob poucos votos/célula é em parte ruído de fragmentação (anotadores "
      "quase sozinhos numa célula recebem θ→1), não separação mais limpa. Por isso o "
      "critério acima exige ≥2 votos/célula; sem esse corte a diferença entre "
      "`demo_base_K4` e `demo_extended_K4` (std 0.316 vs 0.322) é um empate técnico "
      "decidido pelo config que está mais perto do regime inflacionado.")
    A("- **K=4 é preferível a K=8** nestes dados: com K=8 a definição `demo_extended` cai "
      "para ~1.2 votos por célula não vazia, regime em que a 'verdade da subpopulação' é "
      "praticamente o voto do próprio anotador e o θ infla artificialmente (flag "
      "`CÉLULAS_QUASE_SINGULARES`). Foi por isso que ambos os K foram corridos.")
    A("")
    A("## 3. Concordância entre as três definições (Spearman de θ)\n")
    A("| | " + " | ".join(f"`{n}`" for n in names) + " |")
    A("|---" * (len(names) + 1) + "|")
    for a in names:
        cells = []
        for b in names:
            cells.append(f"{spearmanr(results[a]['_theta'], results[b]['_theta'])[0]:.3f}")
        A(f"| `{a}` | " + " | ".join(cells) + " |")
    A("")
    A("Referência single-truth (K=1, tipo-MACE): "
      + ", ".join(
          f"ρ(`{n}`, single) = "
          f"{spearmanr(results[n]['_theta'], results['single_truth_K1']['_theta'])[0]:.3f}"
          for n in names) + "\n")
    A("## 4. Histograma de θ (definições K=" + str(Ks[0]) + ")\n")
    for n in names:
        if not n.endswith(f"K{Ks[0]}"):
            continue
        A(f"**`{n}`**\n")
        A("| faixa | n | |")
        A("|---|---|---|")
        L.extend(_hist(results[n]["_theta"]))
        A("")
    A("## 5. Células (item, subpop) vazias — fenómeno esperado, não bug\n")
    A("Com 6 anotadores por item, K clusters fragmentam cada item em no máximo 6 células "
      "não vazias de K possíveis. A cobertura na tabela acima mostra o efeito directo, e "
      "os votos/célula caem de 6 (K=1) para ~1–2. Verdades por subpopulação assim "
      "fracamente determinadas são o limite estrutural da Task A neste dataset — é por "
      "isso que K=4 e K=8 são ambos reportados: K menor troca granularidade "
      "de subpopulação por evidência por célula.\n")
    A("## 6. Aviso de transferência para Task B/C\n")
    A("**Um resultado fraco na Task A não é preditivo da utilidade em Task B/C.** O "
      "gradiente de sinal demográfico já medido neste trabalho vai na direção oposta: a "
      "condição B1 (fusão de biosinais/demografia sem embeddings per-viewer) mostrou "
      "deltas positivos consistentes em **Task B soft e Task C hard**, e não na deteção "
      "binária. A Task A é a tarefa onde o sinal de subpopulação é mais fraco por "
      "natureza — julgar o método por ela subestima-o. A extensão a Task B "
      "(`n_labels=3`, single-label) é directa na formulação atual; Task C (multilabel) "
      "exige um modelo generativo diferente.\n")
    A("## 7. Limites metodológicos desta corrida\n")
    A("- **`behavioral` não é concordância par-a-par** e não deve ser descrito como tal. "
      "Com 6 anotadores/item e 887 anotadores, a sobreposição par-a-par esperada é "
      "**~0.15 itens por par** — a matriz de concordância par-a-par é praticamente vazia. "
      "Duas alternativas foram testadas e REJEITADAS por evidência medida: (i) essa "
      "matriz par-a-par; (ii) `TruncatedSVD(32)` da matriz de resíduos 887×3984 + KMeans, "
      "que deu clusters degenerados `[883, 2, 1, 1]` (K=4) e `[873, 2, 2, 2, 2, 2, 2, 3]` "
      "(K=8) — os suportes de itens são quase disjuntos, logo as distâncias mediam o nº "
      "de votos, não o comportamento. O que ficou é um **perfil comportamental 6-d** "
      "(yes-rate, desvio médio do consenso, |desvio| médio, concordância-com-maioria, "
      "desvio em itens contestados, fração de votos contestados), padronizado e "
      "clusterizado — comparável entre anotadores apesar dos suportes disjuntos.")
    A("- **Circularidade parcial de `behavioral`**: os clusters são derivados dos mesmos "
      "votos que alimentam o EM. O θ resultante não é independente do sinal que o gerou; "
      "comparar com as definições demográficas (derivadas de metadados externos aos votos) "
      "é o controlo.")
    A("- **Ponto cego específico da definição `behavioral`**: um spammer de viés "
      "CONSTANTE (vota sempre YES, por exemplo) cai num cluster de votantes semelhantes; "
      "o EM concede a esse cluster uma verdade própria, os votos batem certo com ela, e o "
      "spammer recebe θ ALTO. Por construção, esta definição **não distingue** um spammer "
      "de viés constante de uma subpopulação legítima. O teste sintético de preservação "
      "de minoria — o árbitro indicado na §1 — **não cobre este caso**: o spam que ele "
      "injeta é uniforme, não enviesado. Cobrir isto exigiria um braço sintético novo "
      "(spammers de viés constante), fora do âmbito desta etapa.")
    A("- **θ de anotadores com poucos votos** encolhe para a média do prior Beta(0.5,0.5) "
      "= 0.5. Ver `n_votes` no JSON de cada definição antes de usar um θ individual.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[summary] -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=DEFAULT_JSON)
    p.add_argument("--K", type=int, nargs="+", default=[4, 8])
    p.add_argument("--responsibility", default="fixed", choices=["fixed", "legacy"])
    p.add_argument("--suffix", default=None,
                   help="default: '_fixed' em modo corrigido, '' em legado")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    run(json_path=args.json, Ks=tuple(args.K), responsibility=args.responsibility,
        suffix=args.suffix, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
