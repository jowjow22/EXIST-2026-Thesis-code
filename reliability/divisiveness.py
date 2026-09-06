"""PARTE A — divisividade empírica dos dados reais do EXIST (Task A).

Pergunta decisiva: a discordância entre anotadores é **estruturada por subgrupos**
(justifica o modelo de subpopulação) ou é **ruído difuso** (single-truth seria
quase equivalente)? O Krippendorff alpha global (0.2791) mede discordância total
mas não a decompõe. Este módulo decompõe-a.

Diagnóstico puro: não toca no `NutmegEM`, não treina nada, CPU-only.
Reutiliza `run_exist.subpop_from_demo/_from_behavior/_alpha_from_item_counts`.

CONTROLO DE PERMUTAÇÃO (essencial, e não estava no pedido original):
com 1–2 votos por célula (item, subpop), qualquer partição — mesmo ALEATÓRIA —
produz "divisividade" e rácio entre-grupos altos só por fragmentação. Por isso
todas as quantidades são também calculadas sob N_PERM partições aleatórias
(mesmas dimensões de cluster) e reportadas como observado vs. nulo (z-score).
Sem esse contraste os números absolutos não significam nada.

    python -m reliability.divisiveness [--K 4 8] [--n-perm 50]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .exist_data import DEMO_GROUPS_BASE, DEMO_GROUPS_EXTENDED, DEFAULT_JSON, load_task_a
from .run_exist import (
    OUTPUT_DIR,
    _alpha_from_item_counts,
    subpop_from_behavior,
    subpop_from_demo,
)

N_PERM = 50
PERM_SEED = 42
SYNTHETIC_CROSSOVER = 0.15  # divisividade a que subpop-aware ultrapassa single-truth (Stage 1)
GRID_CSV = os.path.join(OUTPUT_DIR, "recovery_grid_existdensity.csv")
REPORT = os.path.join(OUTPUT_DIR, "divisiveness_report.md")
JSON_OUT = os.path.join(OUTPUT_DIR, "divisiveness_metrics.json")


# ------------------------------------------------------------------ núcleo
def _cell_counts(annotations, subpop, N, K):
    """Contagens por célula (item, subpop). Devolve (n (N,K), s (N,K)) com
    n = nº de votos e s = nº de votos YES."""
    items, anns, labels = annotations[:, 0], annotations[:, 1], annotations[:, 2]
    cell = subpop[anns]
    flat = items * K + cell
    n = np.bincount(flat, minlength=N * K).reshape(N, K).astype(float)
    s = np.bincount(flat, weights=labels, minlength=N * K).reshape(N, K).astype(float)
    return n, s


def divisive_fraction(annotations, subpop, N, K, min_votes_per_cell=2):
    """A.1 — fração de itens em que duas subpopulações têm MAIORIAS OPOSTAS.

    Critério explícito: existe k1 com taxa-de-YES > 0.5 e k2 com taxa < 0.5.
    Empates exatos (0.5) NÃO contam como maioria de nenhum dos lados.

    Devolve dict com a fração sobre todos os itens e sobre o subconjunto de
    confiança (>=2 subpops com >=`min_votes_per_cell` votos cada).
    """
    n, s = _cell_counts(annotations, subpop, N, K)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    has_yes = np.nansum(rate > 0.5, axis=1) > 0
    has_no = np.nansum(rate < 0.5, axis=1) > 0
    divisive = has_yes & has_no

    reliable = (n >= min_votes_per_cell).sum(axis=1) >= 2
    return {
        "frac_divisive_all_items": float(divisive.mean()),
        "n_items_all": int(N),
        "frac_divisive_reliable": float(divisive[reliable].mean()) if reliable.any() else float("nan"),
        "n_items_reliable": int(reliable.sum()),
        "frac_items_reliable": float(reliable.mean()),
    }


def between_within_ratio(annotations, subpop, N, K):
    """A.2 — decomposição ANOVA da discordância DENTRO de cada item.

    Por item i:  SS_total = sum_j (v_ij - media_i)^2
                 SS_between = sum_k n_ik (media_ik - media_i)^2
                 SS_within  = SS_total - SS_between
    Agregado sobre itens. Rácio = SS_between / SS_total.

    Rácio alto => a discordância separa-se por subgrupo (subpop justifica-se).
    Rácio baixo => discordância vive DENTRO dos grupos (single-truth equivalente).

    ATENÇÃO: células com 1 voto têm SS_within = 0 por construção, logo empurram o
    rácio para cima mecanicamente. Comparar sempre com o nulo de permutação.
    """
    n, s = _cell_counts(annotations, subpop, N, K)
    n_i = n.sum(axis=1)
    s_i = s.sum(axis=1)
    ok = n_i >= 2
    mean_i = np.where(n_i > 0, s_i / np.maximum(n_i, 1), 0.0)

    # votos binários: sum_j (v - m)^2 = n*m*(1-m) quando m é a média do grupo
    ss_total = (n_i * mean_i * (1 - mean_i))[ok].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ik = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    ss_within = (n * mean_ik * (1 - mean_ik))[ok].sum()
    ss_between = ss_total - ss_within
    return {
        "ss_total": float(ss_total),
        "ss_between": float(ss_between),
        "ss_within": float(ss_within),
        "ratio_between": float(ss_between / ss_total) if ss_total > 0 else float("nan"),
    }


def alpha_per_subgroup(annotations, subpop, N, K):
    """A.3 — Krippendorff alpha nominal DENTRO de cada subpopulação.

    Reutiliza `_alpha_from_item_counts` (run_exist). Só itens com >=2 votos do
    subgrupo entram. alpha_k > alpha_global => os anotadores concordam mais
    dentro do seu grupo do que na população toda (evidência de estrutura).
    """
    items, anns, labels = annotations[:, 0], annotations[:, 1], annotations[:, 2]
    cell = subpop[anns]
    out = []
    for k in range(K):
        m = cell == k
        if m.sum() == 0:
            out.append({"k": k, "alpha": float("nan"), "n_votes": 0, "n_items_ge2": 0})
            continue
        n1 = np.bincount(items[m], weights=labels[m], minlength=N).astype(float)
        n0 = np.bincount(items[m], weights=1 - labels[m], minlength=N).astype(float)
        out.append({
            "k": k,
            "alpha": _alpha_from_item_counts(n0, n1),
            "n_votes": int(m.sum()),
            "n_items_ge2": int(((n0 + n1) >= 2).sum()),
        })
    w = np.array([o["n_items_ge2"] for o in out], dtype=float)
    a = np.array([o["alpha"] for o in out], dtype=float)
    valid = ~np.isnan(a) & (w > 0)
    weighted = float((a[valid] * w[valid]).sum() / w[valid].sum()) if valid.any() else float("nan")
    return {"per_subgroup": out, "alpha_weighted_mean": weighted}


def _permutation_null(annotations, subpop, N, K, n_perm=N_PERM, seed=PERM_SEED):
    """Nulo: mesmas dimensões de cluster, atribuição de anotadores ALEATÓRIA.

    Responde a "quanto destes números é só fragmentação?". Devolve mean/std das
    três quantidades sob H0 e o z-score do observado.
    """
    rng = np.random.default_rng(seed)
    divs_all, divs_rel, ratios, alphas = [], [], [], []
    for _ in range(n_perm):
        perm = rng.permutation(subpop)
        d = divisive_fraction(annotations, perm, N, K)
        divs_all.append(d["frac_divisive_all_items"])
        divs_rel.append(d["frac_divisive_reliable"])
        ratios.append(between_within_ratio(annotations, perm, N, K)["ratio_between"])
        alphas.append(alpha_per_subgroup(annotations, perm, N, K)["alpha_weighted_mean"])
    def st(x):
        x = np.asarray(x, dtype=float)
        return {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x))}
    return {
        "n_perm": n_perm,
        "frac_divisive_all_items": st(divs_all),
        "frac_divisive_reliable": st(divs_rel),
        "ratio_between": st(ratios),
        "alpha_weighted_mean": st(alphas),
    }


def _distinct_subpops_per_item(annotations, subpop, N, K):
    """Nº médio de subpopulações distintas representadas em cada item."""
    n, _ = _cell_counts(annotations, subpop, N, K)
    return float((n > 0).sum(axis=1).mean())


def _null_within_item(annotations, subpop, N, K, n_perm=N_PERM, seed=PERM_SEED):
    """NULO CONDICIONADO (o correto): baralha os VOTOS dentro de cada item.

    Porque foi preciso: o nulo por permutação de anotadores (`_permutation_null`)
    baralha quem pertence a que cluster, o que muda também a COMPOSIÇÃO de cada
    item — os itens reais têm em média menos subpops distintas (ex.: 2.77 vs 3.13
    em `demo_base_K4`), porque os pools de anotadores são correlacionados com os
    itens (split ES/EN, clusters dominados por país). Menos células = menos
    fragmentação = menos divisividade espúria, logo o observado sai ABAIXO desse
    nulo por razões geométricas, não por ausência de estrutura. Os z negativos
    desse nulo **não** significam "menos estrutura que o acaso".

    Este nulo mantém a partição e a composição de cada item EXATAMENTE como estão,
    e permuta apenas os rótulos de voto entre os anotadores do mesmo item. Destrói
    só a associação anotador↔voto — precisamente a hipótese em teste.
    """
    rng = np.random.default_rng(seed)
    items, labels = annotations[:, 0], annotations[:, 2]
    order = np.argsort(items, kind="stable")
    bounds = np.searchsorted(items[order], np.arange(N + 1))
    divs_all, divs_rel, ratios, alphas = [], [], [], []
    for _ in range(n_perm):
        lab = labels.copy()
        lab_o = lab[order]
        for i in range(N):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo > 1:
                lab_o[lo:hi] = rng.permutation(lab_o[lo:hi])
        lab[order] = lab_o
        perm_ann = np.column_stack([annotations[:, 0], annotations[:, 1], lab])
        d = divisive_fraction(perm_ann, subpop, N, K)
        divs_all.append(d["frac_divisive_all_items"])
        divs_rel.append(d["frac_divisive_reliable"])
        ratios.append(between_within_ratio(perm_ann, subpop, N, K)["ratio_between"])
        alphas.append(alpha_per_subgroup(perm_ann, subpop, N, K)["alpha_weighted_mean"])
    def st(x):
        x = np.asarray(x, dtype=float)
        return {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x))}
    return {
        "n_perm": n_perm,
        "frac_divisive_all_items": st(divs_all),
        "frac_divisive_reliable": st(divs_rel),
        "ratio_between": st(ratios),
        "alpha_weighted_mean": st(alphas),
    }


def _z(obs, null):
    return (obs - null["mean"]) / null["std"] if null["std"] > 1e-12 else float("nan")


# -------------------------------------------------------------------- main
def run(json_path=DEFAULT_JSON, Ks=(4, 8), n_perm=N_PERM, verbose=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    annotations, annotator_ids, item_ids, demo, load_stats = load_task_a(json_path, verbose)
    M, N = len(annotator_ids), len(item_ids)

    items, labels = annotations[:, 0], annotations[:, 2]
    n1 = np.bincount(items, weights=labels, minlength=N).astype(float)
    n0 = np.bincount(items, weights=1 - labels, minlength=N).astype(float)
    alpha_global = _alpha_from_item_counts(n0, n1)

    defs = []
    for K in Ks:
        defs.append((f"demo_base_K{K}", subpop_from_demo(annotator_ids, demo, DEMO_GROUPS_BASE, K)[0], K))
        defs.append((f"demo_extended_K{K}", subpop_from_demo(annotator_ids, demo, DEMO_GROUPS_EXTENDED, K)[0], K))
        defs.append((f"behavioral_K{K}", subpop_from_behavior(annotations, M, N, K)[0], K))

    results = {}
    for name, subpop, K in defs:
        div = divisive_fraction(annotations, subpop, N, K)
        bw = between_within_ratio(annotations, subpop, N, K)
        al = alpha_per_subgroup(annotations, subpop, N, K)
        null = _permutation_null(annotations, subpop, N, K, n_perm)
        null_wi = _null_within_item(annotations, subpop, N, K, n_perm)
        rec = {
            "K": K,
            "distinct_subpops_per_item_obs": _distinct_subpops_per_item(annotations, subpop, N, K),
            "distinct_subpops_per_item_null": float(np.mean([
                _distinct_subpops_per_item(
                    annotations, np.random.default_rng(PERM_SEED + t).permutation(subpop), N, K)
                for t in range(10)])),
            "null_within_item": null_wi,
            "zwi_frac_divisive_reliable": _z(
                divisive_fraction(annotations, subpop, N, K)["frac_divisive_reliable"],
                null_wi["frac_divisive_reliable"]),
            "zwi_ratio_between": _z(
                between_within_ratio(annotations, subpop, N, K)["ratio_between"],
                null_wi["ratio_between"]),
            "zwi_alpha_weighted": _z(
                alpha_per_subgroup(annotations, subpop, N, K)["alpha_weighted_mean"],
                null_wi["alpha_weighted_mean"]),
            "divisive": div,
            "between_within": bw,
            "alpha": al,
            "alpha_global": alpha_global,
            "null": null,
            "z_frac_divisive_all": _z(div["frac_divisive_all_items"], null["frac_divisive_all_items"]),
            "z_frac_divisive_reliable": _z(div["frac_divisive_reliable"], null["frac_divisive_reliable"]),
            "z_ratio_between": _z(bw["ratio_between"], null["ratio_between"]),
            "z_alpha_weighted": _z(al["alpha_weighted_mean"], null["alpha_weighted_mean"]),
        }
        results[name] = rec
        if verbose:
            print(
                f"[div] {name:20s} divisivo(todos)={div['frac_divisive_all_items']:.3f} "
                f"(nulo {null['frac_divisive_all_items']['mean']:.3f}, z={rec['z_frac_divisive_all']:+.2f}) | "
                f"divisivo(confiável)={div['frac_divisive_reliable']:.3f} "
                f"(z={rec['z_frac_divisive_reliable']:+.2f}) | "
                f"ratio_entre={bw['ratio_between']:.3f} (z={rec['z_ratio_between']:+.2f}) | "
                f"α_intra={al['alpha_weighted_mean']:.3f} vs global {alpha_global:.3f} "
                f"(z={rec['z_alpha_weighted']:+.2f})"
            )
            print(
                f"      nulo-within-item (correto): z_div={rec['zwi_frac_divisive_reliable']:+.2f} "
                f"z_ratio={rec['zwi_ratio_between']:+.2f} z_alpha={rec['zwi_alpha_weighted']:+.2f} | "
                f"subpops distintas/item obs={rec['distinct_subpops_per_item_obs']:.2f} "
                f"vs nulo-anotador {rec['distinct_subpops_per_item_null']:.2f}"
            )

    _write_report(results, load_stats, alpha_global, n_perm)
    with open(JSON_OUT, "w") as f:
        json.dump({"alpha_global": alpha_global, "definitions": results}, f, indent=2)
    return results


def _verdict(rec):
    """Conclusão de UMA frase por definição, a partir dos z-scores vs. nulo."""
    z_div = rec["zwi_frac_divisive_reliable"]
    z_ratio = rec["zwi_ratio_between"]
    z_alpha = rec["zwi_alpha_weighted"]
    strong = sum(z > 2 for z in (z_div, z_ratio, z_alpha) if not np.isnan(z))
    frac_rel = rec["divisive"]["frac_items_reliable"]
    if frac_rel < 0.25:
        return ("**inconclusivo por esparsidade** — só "
                f"{100 * frac_rel:.1f}% dos itens têm duas subpops com ≥2 votos, "
                "logo o subconjunto de confiança não sustenta o veredito")
    if strong >= 2:
        zmax = max(z for z in (z_div, z_ratio, z_alpha) if not np.isnan(z))
        força = "marginal (nenhum z > 3)" if zmax < 3 else "clara"
        return (f"**subpopulação justifica-se, evidência {força}** — {strong} dos 3 "
                "indicadores acima do nulo condicionado")
    if strong == 0:
        return ("**não se justifica** — nenhum indicador excede o nulo condicionado: "
                "a discordância é ruído difuso, não estrutura de subgrupo")
    return "**inconclusivo** — só 1 dos 3 indicadores excede o nulo"


def _write_report(results, load_stats, alpha_global, n_perm):
    L = []
    A = L.append
    A("# Parte A — divisividade empírica real (EXIST Task A, memes)\n")
    A("Gerado por `python -m reliability.divisiveness`. CPU-only, diagnóstico puro "
      "(o `NutmegEM` não é usado nem alterado).\n")
    A(f"Dados: {load_stats['n_items']} itens · {load_stats['n_annotators']} anotadores · "
      f"{load_stats['n_votes']} votos · Krippendorff α global = **{alpha_global:.4f}**\n")
    A("## 0. Dois nulos, e porque o segundo é o que conta\n")
    A("**Nulo A (permutação de anotadores)** — baralha quem pertence a que cluster, "
      "mantendo os tamanhos de cluster. Problema descoberto ao validar: também muda a "
      "COMPOSIÇÃO de cada item. Os itens reais têm em média **menos** subpops distintas do "
      "que os permutados ("
      + "; ".join(
          f"`{n}` {r['distinct_subpops_per_item_obs']:.2f} vs {r['distinct_subpops_per_item_null']:.2f}"
          for n, r in results.items())
      + "), porque os pools de anotadores estão correlacionados com os itens (split ES/EN, "
      "clusters dominados por país). Menos células = menos fragmentação = menos "
      "divisividade espúria. **Os z negativos do Nulo A são geometria, não ausência de "
      "estrutura** — não devem ser lidos como 'menos estruturado que o acaso'.\n")
    A("**Nulo B (condicionado, *within-item*)** — mantém a partição e a composição de cada "
      "item exatamente como estão, e permuta apenas os **votos** entre os anotadores do "
      "mesmo item. Destrói só a associação anotador↔voto, que é a hipótese em teste. "
      "**Os vereditos usam o Nulo B.** O Nulo A fica na tabela como diagnóstico de "
      "composição.\n")
    A("### Nota de método original\n")
    A(f"Com 1–2 votos por célula (item, subpop), **qualquer** partição — inclusive "
      f"aleatória — produz divisividade e rácio entre-grupos altos, só por fragmentação. "
      f"Todas as quantidades são por isso comparadas com {n_perm} partições aleatórias das "
      "mesmas dimensões (mesmos tamanhos de cluster, anotadores baralhados). "
      "**O número que interessa é o z-score, não o valor absoluto.** Um z ≈ 0 significa "
      "'esta partição não explica a discordância melhor do que uma partição ao acaso'.\n")
    A("## A.1 / A.2 / A.3 — NULO B (condicionado) — os números que contam\n")
    A("| definição | divisivo (confiável) | z (Nulo B) | ratio entre/total | z (Nulo B) | "
      "α intra (pond.) | z (Nulo B) |")
    A("|---|---|---|---|---|---|---|")
    for n, r in results.items():
        A(f"| `{n}` | {r['divisive']['frac_divisive_reliable']:.3f} | "
          f"{r['zwi_frac_divisive_reliable']:+.2f} | {r['between_within']['ratio_between']:.3f} | "
          f"{r['zwi_ratio_between']:+.2f} | {r['alpha']['alpha_weighted_mean']:.3f} | "
          f"{r['zwi_alpha_weighted']:+.2f} |")
    A("")
    A("## Tabela completa (inclui o Nulo A, só diagnóstico de composição)\n")
    A("| definição | divisivo (todos) | nulo | z | divisivo (confiável) | nulo | z | "
      "itens confiáveis | ratio entre/total | nulo | z | α intra (pond.) | nulo | z |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in results.items():
        d, bw, al, nu = r["divisive"], r["between_within"], r["alpha"], r["null"]
        A(f"| `{n}` | {d['frac_divisive_all_items']:.3f} | "
          f"{nu['frac_divisive_all_items']['mean']:.3f} | {r['z_frac_divisive_all']:+.2f} | "
          f"{d['frac_divisive_reliable']:.3f} | {nu['frac_divisive_reliable']['mean']:.3f} | "
          f"{r['z_frac_divisive_reliable']:+.2f} | "
          f"{d['n_items_reliable']} ({100*d['frac_items_reliable']:.1f}%) | "
          f"{bw['ratio_between']:.3f} | {nu['ratio_between']['mean']:.3f} | "
          f"{r['z_ratio_between']:+.2f} | {al['alpha_weighted_mean']:.3f} | "
          f"{nu['alpha_weighted_mean']['mean']:.3f} | {r['z_alpha_weighted']:+.2f} |")
    A("")
    A(f"α global (referência para a coluna α intra): **{alpha_global:.4f}**\n")
    A("**Critério de divisividade (A.1), explícito:** um item é divisivo se existe uma "
      "subpop com taxa-de-YES > 0.5 e outra com taxa < 0.5 nesse item. Empates exatos "
      "(0.5) não contam para nenhum lado. 'Confiável' = itens com ≥2 subpops com ≥2 votos "
      "cada — o caveat pedido: com 1 voto por célula a 'maioria da subpop' é o voto de uma "
      "pessoa.\n")
    A("**A.2, decomposição:** por item, `SS_total = Σ_j (v_ij − média_i)²`, "
      "`SS_between = Σ_k n_ik (média_ik − média_i)²`, `SS_within = SS_total − SS_between`, "
      "agregado sobre itens com ≥2 votos. Células de 1 voto têm `SS_within = 0` por "
      "construção e empurram o rácio para cima — daí o nulo.\n")
    A("## A.3 — α por subgrupo (detalhe)\n")
    for n, r in results.items():
        vals = ", ".join(
            f"k{o['k']}: {o['alpha']:.3f} (n_itens≥2={o['n_items_ge2']})"
            for o in r["alpha"]["per_subgroup"]
        )
        A(f"- `{n}` — {vals}")
    A("")
    A("## A.4 — onde os dados caem na curva de contraste do Stage 1\n")
    A(f"No grid sintético (densidade EXIST), subpop-aware ultrapassa single-truth a partir "
      f"de divisividade ≈ **{SYNTHETIC_CROSSOVER}**. Comparação:\n")
    A("| definição | divisividade empírica (confiável) | acima do cruzamento 0.15? | "
      "divisividade ACIMA DO NULO (obs − nulo) | acima de 0.15? |")
    A("|---|---|---|---|---|")
    for n, r in results.items():
        obs = r["divisive"]["frac_divisive_reliable"]
        exc = obs - r["null"]["frac_divisive_reliable"]["mean"]
        A(f"| `{n}` | {obs:.3f} | {'SIM' if obs > SYNTHETIC_CROSSOVER else 'não'} | "
          f"{exc:+.3f} | {'SIM' if exc > SYNTHETIC_CROSSOVER else 'não'} |")
    A("")
    A("⚠ **A comparação com 0.15 é direcional, não pontual.** O cruzamento sintético foi "
      "medido com **K=2** e subpopulações de informação equilibrada; a divisividade "
      "empírica vem de **K ∈ {4,8}** com células fragmentadas. As duas quantidades não são "
      "comensuráveis ponto a ponto — servem para dizer de que lado do regime os dados "
      "caem, não para prever um valor de r.\n")
    A("**Como ler estas duas colunas.** A divisividade sintética do Stage 1 é a fração de "
      "itens em que as VERDADES das subpops diferem — quantidade limpa, sem ruído. A "
      "divisividade empírica é medida sobre votos ruidosos, logo está inflacionada: parte "
      "dela seria observada mesmo com subpopulações aleatórias. A comparação honesta com o "
      "cruzamento de 0.15 é a coluna **obs − nulo**, não a coluna bruta.\n")
    A("## Aviso de circularidade (aplica-se só a `behavioral`)\n")
    A("As definições `behavioral_*` clusterizam anotadores **pelos próprios votos** "
      "(perfil de desvio face ao consenso). Medir depois se esses clusters explicam a "
      "discordância dos mesmos votos é **parcialmente tautológico**: um z alto ali é o "
      "esperado por construção, não evidência de estrutura latente. O nulo de permutação "
      "**não** corrige isto — baralha os rótulos, mas os rótulos originais já foram "
      "ajustados aos dados. Só as definições demográficas (`demo_base`, `demo_extended`) "
      "usam informação externa aos votos e por isso só elas dão um teste honesto da "
      "hipótese de subpopulação.\n")
    A("## Conclusão por definição\n")
    for n, r in results.items():
        extra = (" ⚠ **mas ver o aviso de circularidade acima — este veredito não é "
                 "evidência independente**") if n.startswith("behavioral") else ""
        A(f"- `{n}`: {_verdict(r)}{extra}.")
    A("")
    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] -> {REPORT}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=DEFAULT_JSON)
    p.add_argument("--K", type=int, nargs="+", default=[4, 8])
    p.add_argument("--n-perm", type=int, default=N_PERM)
    args = p.parse_args()
    run(json_path=args.json, Ks=tuple(args.K), n_perm=args.n_perm)


if __name__ == "__main__":
    main()
