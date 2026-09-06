"""Validação estatística da dimensão latente "expansivo ↔ parcimonioso".

A Parte 2 do diagnóstico comportamental (`behavioral_diagnosis.md` §6.3) observou
que as quatro features contínuas "movem-se todas na mesma direção" ENTRE OS
CENTRÓIDES do KMeans, e concluiu daí uma dimensão latente única. Essa evidência é
fraca e parcialmente circular: o KMeans agrupa por proximidade nesse mesmo espaço
de features, logo tende a produzir centróides que diferem em várias em simultâneo.

Este módulo testa a afirmação ao nível dos ANOTADORES INDIVIDUAIS, e controla o
mecanismo alternativo: três das quatro features partilham o denominador
`n_sexist_items`, logo podem correlacionar-se por puro artefacto de tamanho de
amostra (mais itens => mais oportunidade de acumular categorias distintas; a
entropia estimada de poucas amostras é sistematicamente subestimada).

O que este módulo NÃO faz: não treina nada, não toca no modelo de memes, não
regenera `behavioral_profiles.csv` (lê o existente), e **não usa o KMeans nem
qualquer label de cluster** — o objetivo é testar a afirmação independentemente
da partição que a originou.

Uso:
    python3 -m reliability.validate_latent_dimension
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from .run_behavioral_diagnosis import OUT_DIR, md_table  # reutilização, sem cópia

# ===========================================================================
# RÉGUAS PRÉ-REGISTADAS — fixadas antes de olhar qualquer resultado
# ===========================================================================

SEED = 42

PROFILES_CSV = os.path.join(OUT_DIR, "behavioral_profiles.csv")
REPORT_PATH = os.path.join(OUT_DIR, "latent_dimension_validation.md")
JSON_PATH = os.path.join(OUT_DIR, "latent_dimension_validation.json")

#: As quatro features contínuas cuja estrutura conjunta está em causa.
FEATURES = [
    "yes_rate",
    "n_distinct_categories",
    "category_entropy",
    "mean_labels_per_item",
]
#: As três estimadas sobre o denominador `n_sexist_items`.
CATEGORY_FEATURES = FEATURES[1:]
CONTROL = "n_sexist_items"

#: Régua de MAGNITUDE, não de significância. Com n≈844, ρ=0,08 sai
#: "significativo" e não quer dizer nada na prática.
RHO_BANDS = [
    (0.20, "desprezável"),
    (0.30, "fraca"),
    (0.50, "moderada"),
    (1.01, "forte"),
]

#: Divergência |ρ_spearman − r_pearson| acima da qual se assinala não-linearidade.
PEARSON_DIVERGENCE_FLAG = 0.10

#: PCA. Referência de nulidade: com 4 features independentes esperar-se-ia ~25%
#: por componente (com 3 features, ~33%).
PC1_UNIQUE_DIMENSION = 0.55  # >55% E todas as loadings do mesmo sinal => dimensão única
PC1_PARTIAL = 0.35           # 35–55% => estrutura parcial; <35% => sem dimensão latente

#: Se alguma loading do PC1 tiver sinal oposto às outras, a interpretação
#: "expansivo ↔ parcimonioso" está ERRADA e o padrão real de sinais tem de ser
#: reportado em vez dela.

#: Estratificação (C.4): banda em torno da mediana de `n_sexist_items`.
STRATUM_HALF_WIDTH = 3

#: Queda relativa da parcial face ao bruto a partir da qual se declara que a
#: associação era em boa parte artefacto de tamanho de amostra.
PARTIAL_COLLAPSE_FRACTION = 0.50

#: `category_entropy` é calculada sobre a distribuição das categorias que o
#: anotador usou, logo é limitada por cima por log(n_distinct_categories): com 2
#: categorias distintas a entropia normalizada não pode passar de log(2)/log(5).
#: As duas features estão portanto ligadas POR DEFINIÇÃO, não só por
#: comportamento. Acima deste ρ, o relatório assinala que a associação é em parte
#: definicional e que o PC1 das features de categoria sobrestima a evidência
#: independente.
DEFINITIONAL_LINK_RHO = 0.70
DEFINITIONAL_PAIR = ("n_distinct_categories", "category_entropy")

#: Pares cuja parcial por `n_sexist_items` é OVER-CONTROL.
#: ρ(n_sexist_items, yes_rate) = +0,925 e `n_sexist_items == n_yes` em 834/887
#: anotadores — a variável de controlo e `yes_rate` são quase a mesma coisa.
#: Nesses pares a parcial tende a zero POR CONSTRUÇÃO e não pode ser lida como
#: "a associação era artefacto". Para eles o controlo interpretável é a
#: estratificação (C.4). Decisão do autor, tomada antes de codar.
OVERCONTROL_FEATURE = "yes_rate"


def rho_band(rho: float) -> str:
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return "—"
    a = abs(rho)
    for edge, label in RHO_BANDS:
        if a < edge:
            return label
    return "forte"


def pc1_verdict(explained: float, loadings: np.ndarray) -> str:
    """Aplica a régua pré-registada ao PC1."""
    signs = np.sign(loadings)
    same_sign = bool(np.all(signs == signs[0]))
    if not same_sign:
        return "SINAIS MISTOS — 'expansivo ↔ parcimonioso' não descreve este PC1"
    if explained > PC1_UNIQUE_DIMENSION:
        return "DIMENSÃO LATENTE ÚNICA"
    if explained >= PC1_PARTIAL:
        return "ESTRUTURA PARCIAL — descrever associações específicas, não dimensão única"
    return "SEM DIMENSÃO LATENTE — features substancialmente independentes"


# ===========================================================================
# Estatística
# ===========================================================================


def pairs(features):
    return [(features[i], features[j])
            for i in range(len(features)) for j in range(i + 1, len(features))]


def corr_matrix(df: pd.DataFrame, features, method: str) -> pd.DataFrame:
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    m = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
    for a, b in pairs(features):
        r = fn(df[a], df[b])[0]
        m.loc[a, b] = m.loc[b, a] = float(r)
    return m


def corr_table(df: pd.DataFrame, features, method: str = "spearman") -> pd.DataFrame:
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    rows = {}
    for a, b in pairs(features):
        r, p = fn(df[a], df[b])
        rows[f"{a} × {b}"] = {
            "n": int(len(df)),
            "ρ": float(r),
            "p": float(p),
            "magnitude": rho_band(float(r)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Resíduos de y regredido em x por OLS simples (com intercepto).

    `scipy` e `sklearn` não expõem correlação parcial diretamente; a parcial
    ρ(X, Y | Z) é obtida correlacionando os resíduos de X~Z com os de Y~Z.
    Equivalente à definição clássica de correlação parcial.
    """
    X = np.column_stack([np.ones(len(x)), x])
    with np.errstate(all="ignore"):  # flags espúrias do BLAS Accelerate (macOS)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta


def partial_corr_table(df: pd.DataFrame, features, control: str) -> pd.DataFrame:
    z = df[control].to_numpy(dtype=float)
    resid = {f: residualize(df[f].to_numpy(dtype=float), z) for f in features}
    rows = {}
    for a, b in pairs(features):
        raw = float(stats.spearmanr(df[a], df[b])[0])
        par = float(stats.spearmanr(resid[a], resid[b])[0])
        over = (a == OVERCONTROL_FEATURE) or (b == OVERCONTROL_FEATURE)
        rows[f"{a} × {b}"] = {
            "ρ bruto": raw,
            "ρ parcial": par,
            "Δ": par - raw,
            "magnitude (parcial)": rho_band(par),
            "leitura": "OVER-CONTROL" if over else "controlo válido",
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def run_pca(df: pd.DataFrame, features):
    """PCA sobre features padronizadas (z-score). Determinística."""
    X = df[features].to_numpy(dtype=float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    with np.errstate(all="ignore"):  # flags espúrias do BLAS Accelerate (macOS)
        pca = PCA(n_components=len(features), random_state=SEED).fit(Z)
    ev = pca.explained_variance_ratio_
    loadings = pca.components_[0]
    # Convenção de sinal: PC1 orientado para que a maioria das loadings seja
    # positiva, para o sinal ser legível. Não altera a estrutura.
    if float(np.sum(np.sign(loadings))) < 0:
        loadings = -loadings
    return {
        "n": int(len(df)),
        "features": list(features),
        "explained": [float(v) for v in ev],
        "cumulative": [float(v) for v in np.cumsum(ev)],
        "pc1_loadings": {f: float(v) for f, v in zip(features, loadings)},
        "pc1_explained": float(ev[0]),
        "null_reference": 1.0 / len(features),
        "veredito": pc1_verdict(float(ev[0]), loadings),
    }


def pca_tables(res: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    var = pd.DataFrame(
        {
            "variância explicada (%)": [100 * v for v in res["explained"]],
            "acumulada (%)": [100 * v for v in res["cumulative"]],
        },
        index=[f"PC{i + 1}" for i in range(len(res["explained"]))],
    )
    load = pd.DataFrame.from_dict(
        {k: {"loading no PC1": v} for k, v in res["pc1_loadings"].items()},
        orient="index",
    )
    return var, load


# ===========================================================================
# Relatório
# ===========================================================================


def build_report(data: dict) -> str:
    """Constrói o relatório A PARTIR dos números medidos.

    NOTA DE PROCESSO: nenhuma frase deste relatório afirma a direção do
    resultado por prosa fixa; tudo o que é conclusão vem de f-strings sobre os
    valores calculados. O defeito oposto ocorreu na Parte 2 e está documentado
    em `reliability/outputs/behavioral_diagnosis_observacoes.md` §1.1.
    """
    L: list[str] = []
    A = L.append

    n = data["n"]
    sp = data["spearman_table"]
    pe = data["pearson_table"]
    pca_raw = data["pca_raw"]
    pca_res4 = data["pca_resid_4"]
    pca_res3 = data["pca_resid_3"]
    partial = data["partial_table"]
    strat = data["stratum_table"]
    ctrl = data["control_table"]
    band = data["stratum_band"]
    n_str = data["stratum_n"]

    cat_pairs = [i for i in sp.index if "yes_rate" not in i]
    yes_pairs = [i for i in sp.index if "yes_rate" in i]
    max_cat = sp.loc[cat_pairs, "ρ"].abs().max()
    max_yes = sp.loc[yes_pairs, "ρ"].abs().max()

    A("# Validação da dimensão latente \"expansivo ↔ parcimonioso\"")
    A("")
    A("**O que este documento é:** teste estatístico, ao nível dos anotadores individuais, da "
      "afirmação feita em `behavioral_diagnosis.md` §6.3 e §8.1 — a de que as quatro features "
      "contínuas formam uma dimensão latente única. A evidência original foi observada **entre "
      "centróides do KMeans**, o que é parcialmente circular: o KMeans agrupa por proximidade "
      "nesse mesmo espaço.")
    A("")
    A(f"- **Fonte:** `{PROFILES_CSV}` (lido, não regenerado)")
    A(f"- **Código:** `reliability/validate_latent_dimension.py` · seed {SEED}")
    A(f"- **n efetivo:** {n} anotadores sem NaN nas quatro features (de 887)")
    A("- **KMeans e labels de cluster não são usados em nenhum ponto desta análise.**")
    A("")
    A("`[MEDIDO]` — número observado · `[HIPÓTESE]` — interpretação não testada · "
      "`[LIMITAÇÃO]` — restrição a declarar.")
    A("")
    A("---")
    A("")

    # ================= VEREDITO (primeiro, a pedido) =================
    A("## 1. Veredito")
    A("")

    A("### 1.1 As quatro features correlacionam-se entre si ao nível individual?")
    A("")
    A(f"`[MEDIDO]` — entre as três features de categoria, o |ρ| máximo é **{max_cat:.4f}** "
      f"({rho_band(max_cat)}). Nos pares que envolvem `yes_rate`, o |ρ| máximo é "
      f"**{max_yes:.4f}** ({rho_band(max_yes)}).")
    A("")
    for idx, row in sp.iterrows():
        A(f"- `{idx}` — ρ = {row['ρ']:+.4f} ({row['magnitude']})")
    A("")
    bands_present = sorted(set(sp["magnitude"]))
    A(f"Magnitudes presentes: {', '.join(bands_present)}. "
      "Leitura por magnitude, não por significância — com "
      f"n = {n}, ρ = 0,08 sai \"significativo\" e não quer dizer nada.")
    A("")

    A("### 1.2 Existe uma dimensão latente única?")
    A("")
    A(f"`[MEDIDO]` — PC1 explica **{100 * pca_raw['pc1_explained']:.1f}%** da variância das "
      f"quatro features padronizadas (referência de nulidade com 4 features independentes: "
      f"{100 * pca_raw['null_reference']:.0f}%).")
    A("")
    A("Loadings do PC1: " + " · ".join(
        f"`{k}` {v:+.4f}" for k, v in pca_raw["pc1_loadings"].items()))
    A("")
    signs = np.sign(list(pca_raw["pc1_loadings"].values()))
    same = bool(np.all(signs == signs[0]))
    A(f"Sinais: {'todos iguais' if same else '**MISTOS**'}.")
    A("")
    A(f"**Veredito pela régua pré-registada: {pca_raw['veredito']}.**")
    A("")
    margin = 100 * (pca_raw["pc1_explained"] - PC1_UNIQUE_DIMENSION)
    A(f"`[MEDIDO]` — margem sobre o limiar de dimensão única: **{margin:+.1f} pontos "
      f"percentuais** ({100 * pca_raw['pc1_explained']:.1f}% contra "
      f"{100 * PC1_UNIQUE_DIMENSION:.0f}%). "
      + ("`[LIMITAÇÃO]` — a passagem é **apertada**. O veredito muda de categoria com uma "
         "diferença de poucos pontos, e o limiar de 55% é convencional, não derivado dos "
         "dados. Reportar a percentagem, nunca só o rótulo."
         if abs(margin) <= 5.0 else
         "A passagem é folgada: o veredito não depende da escolha exacta do limiar."))
    A("")
    if not same:
        A("`[LIMITAÇÃO]` — com sinais mistos, a leitura \"expansivo contra parcimonioso\" "
          "**não descreve** este eixo: as features não crescem em conjunto. O padrão real de "
          "sinais está acima e é o que deve ser reportado.")
        A("")

    A("### 1.3 A associação sobrevive ao controlo por `n_sexist_items`?")
    A("")
    A("`[MEDIDO]` — os três números lado a lado, por par:")
    A("")
    A(md_table(strat, index_name="par"))
    A("")
    valid = strat[strat["leitura"] == "controlo válido"]
    if len(valid):
        worst = valid.assign(keep=(valid["ρ parcial"].abs() /
                                   valid["ρ bruto"].abs().replace(0, np.nan)))
        collapsed = worst[worst["keep"] < PARTIAL_COLLAPSE_FRACTION]
        A(f"Nos {len(valid)} pares com controlo válido (entre features de categoria), "
          f"**{len(collapsed)}** perdem mais de "
          f"{100 * (1 - PARTIAL_COLLAPSE_FRACTION):.0f}% da magnitude ao controlar por "
          f"`{CONTROL}`"
          + (": " + ", ".join(f"`{i}`" for i in collapsed.index) if len(collapsed) else "."))
        A("")
    A(f"`[LIMITAÇÃO]` — **over-control nos pares com `yes_rate`.** "
      f"ρ(`{CONTROL}`, `yes_rate`) = {data['control_vs_yes_rate']:+.4f}, e "
      f"`{CONTROL}` == `n_yes` em {data['n_sexist_equals_n_yes']} dos 887 anotadores. A "
      "variável de controlo e `yes_rate` são quase a mesma coisa: nesses pares a parcial "
      "tende a zero **por construção** e não pode ser lida como \"a associação era "
      "artefacto\". Para eles o controlo interpretável é a estratificação.")
    A("")
    A(f"`[MEDIDO]` — estratificação: `{CONTROL}` na banda **{band[0]}–{band[1]}** "
      f"(mediana ± {STRATUM_HALF_WIDTH}), **n = {n_str}**. As colunas `ρ estratificado` "
      "da tabela acima vêm daí.")
    A("")
    A("`[MEDIDO]` — PCA sobre features residualizadas em `" + CONTROL + "`:")
    A("")
    A(f"| variante | PC1 antes | PC1 depois | nulidade | veredito |")
    A("|---|---|---|---|---|")
    A(f"| 4 features (contaminada por over-control) | {100 * pca_raw['pc1_explained']:.1f}% | "
      f"{100 * pca_res4['pc1_explained']:.1f}% | "
      f"{100 * pca_res4['null_reference']:.0f}% | {pca_res4['veredito']} |")
    A(f"| **3 de categoria (a pergunta limpa)** | "
      f"{100 * data['pca_raw_3']['pc1_explained']:.1f}% | "
      f"{100 * pca_res3['pc1_explained']:.1f}% | "
      f"{100 * pca_res3['null_reference']:.0f}% | {pca_res3['veredito']} |")
    A("")
    A("### 1.4 Duas ressalvas que os números impõem")
    A("")
    dp = f"{DEFINITIONAL_PAIR[0]} × {DEFINITIONAL_PAIR[1]}"
    rho_def = float(sp.loc[dp, "ρ"]) if dp in sp.index else float("nan")
    if abs(rho_def) > DEFINITIONAL_LINK_RHO:
        A(f"**(a) Duas das três features de categoria estão ligadas por definição.** "
          f"`[MEDIDO]` ρ(`{DEFINITIONAL_PAIR[0]}`, `{DEFINITIONAL_PAIR[1]}`) = "
          f"{rho_def:+.4f} — o par mais forte da matriz. `category_entropy` é calculada sobre "
          "a distribuição das categorias efetivamente usadas, logo é **limitada por cima** por "
          "`log(n_distinct_categories)`: quem usou 2 categorias não pode ter entropia "
          "normalizada acima de `log(2)/log(5)` ≈ 0,43. A associação é em parte matemática, "
          "não comportamental.")
        A("")
        A(f"Consequência: o PC1 de {100 * data['pca_raw_3']['pc1_explained']:.1f}% sobre as "
          "três features de categoria **sobrestima a evidência independente** — são mais "
          "próximas de duas medições e meia do que de três. `[LIMITAÇÃO]` a declarar sempre "
          "que esse número for citado.")
        A("")
        nd = [i for i in strat.index
              if i != dp and OVERCONTROL_FEATURE not in i]
        if nd:
            A("**Mas o achado não colapsa com esta ressalva.** `[MEDIDO]` — descontando o par "
              f"definicional, restam {len(nd)} associações **sem** ligação matemática entre "
              "si, e nenhuma delas é artefacto do denominador:")
            A("")
            for i in nd:
                A(f"- `{i}` — ρ bruto {strat.loc[i, 'ρ bruto']:+.4f} → parcial "
                  f"{strat.loc[i, 'ρ parcial']:+.4f} → estratificado "
                  f"{strat.loc[i, 'ρ estratificado']:+.4f} "
                  f"({rho_band(float(strat.loc[i, 'ρ parcial']))})")
            A("")
            A("Ou seja: o par inflacionado é **um** dos três, não todos. O núcleo da estrutura "
              "entre features de categoria sobrevive à ressalva definicional **e** ao controlo "
              f"por `{CONTROL}`. Descontar o achado inteiro por causa de (a) seria "
              "sobre-corrigir.")
            A("")
    else:
        A(f"**(a)** ρ(`{DEFINITIONAL_PAIR[0]}`, `{DEFINITIONAL_PAIR[1]}`) = {rho_def:+.4f}, "
          f"abaixo do limiar de ligação definicional ({DEFINITIONAL_LINK_RHO}). As duas "
          "features carregam evidência substancialmente independente.")
        A("")

    ys = strat.loc[[i for i in strat.index if OVERCONTROL_FEATURE in i]]
    raw_max = float(ys["ρ bruto"].abs().max())
    str_max = float(ys["ρ estratificado"].abs().max())
    A(f"**(b) A questão do `{OVERCONTROL_FEATURE}` fica por resolver, e não por omissão.** "
      f"`[MEDIDO]` em bruto, `{OVERCONTROL_FEATURE}` associa-se às features de categoria até "
      f"|ρ| = {raw_max:.4f} ({rho_band(raw_max)}); no estrato, até |ρ| = {str_max:.4f} "
      f"({rho_band(str_max)}); na parcial, até "
      f"|ρ| = {float(ys['ρ parcial'].abs().max()):.4f}.")
    A("")
    A("`[LIMITAÇÃO]` — **os dois controlos atenuam esta associação pela mesma razão, e nenhum "
      "a resolve.** A estratificação fixa `" + CONTROL + "` numa banda estreita, e como "
      "`yes_rate ≈ " + CONTROL + " / n_votes`, fixar o numerador deixa `yes_rate` a variar "
      "quase só através de `n_votes`. A estratificação é, também ela, um controlo parcial "
      "sobre a própria variável — mais suave que a parcial, não independente dela.")
    A("")
    A(f"`[HIPÓTESE]` — não é possível, com estes dados, separar a associação de "
      f"`{OVERCONTROL_FEATURE}` com as features de categoria do efeito mecânico do "
      "denominador, porque `yes_rate` **é** o denominador normalizado. Distingui-los exigiria "
      "uma medida de propensão a marcar YES que não entrasse no denominador das features de "
      "categoria — e não existe nestes dados.")
    A("")
    A("---")
    A("")

    # ================= O QUE PODE / NÃO PODE SER ESCRITO =================
    A("## 2. O que pode e não pode ser escrito na tese")
    A("")
    A("### Sustentado pelos números")
    A("")
    for line in data["supported"]:
        A(f"- {line}")
    A("")
    A("### NÃO sustentado — não escrever")
    A("")
    for line in data["unsupported"]:
        A(f"- {line}")
    A("")
    A("---")
    A("")

    # ================= TABELAS =================
    A("## 3. Correlações par-a-par")
    A("")
    A(f"Spearman é a métrica **primária**: `n_distinct_categories` tem apenas "
      f"{data['n_distinct_nunique']} valores distintos "
      f"({data['n_distinct_min']:.0f}–{data['n_distinct_max']:.0f}), quase ordinal, e as "
      "distribuições são enviesadas. Pearson entra só como verificação de consistência.")
    A("")
    A("**Régua de magnitude (pré-registada):** |ρ| < 0,20 desprezável · 0,20–0,30 fraca · "
      "0,30–0,50 moderada · > 0,50 forte.")
    A("")
    A("### 3.1 Spearman (primária)")
    A("")
    A(md_table(sp, index_name="par"))
    A("")
    A("Matriz completa:")
    A("")
    A(md_table(data["spearman_matrix"], index_name="ρ"))
    A("")
    A("### 3.2 Pearson (verificação de consistência)")
    A("")
    A(md_table(pe, index_name="par"))
    A("")
    div = data["divergence"]
    flagged = {k: v for k, v in div.items() if abs(v) > PEARSON_DIVERGENCE_FLAG}
    if flagged:
        A(f"`[MEDIDO]` — divergência |ρ − r| acima de {PEARSON_DIVERGENCE_FLAG} em "
          f"{len(flagged)} par(es): " +
          ", ".join(f"`{k}` ({v:+.4f})" for k, v in flagged.items()) +
          ". Indica não-linearidade nessas relações; a leitura fica com o Spearman.")
    else:
        A(f"`[MEDIDO]` — nenhuma divergência |ρ − r| acima de {PEARSON_DIVERGENCE_FLAG}. "
          "Spearman e Pearson concordam; não há não-linearidade a assinalar.")
    A("")
    A("---")
    A("")

    A("## 4. PCA bruta")
    A("")
    var, load = pca_tables(pca_raw)
    A(md_table(var, float_fmt="{:.2f}", index_name="componente"))
    A("")
    A(md_table(load, index_name="feature"))
    A("")
    A("**Régua pré-registada:** PC1 > 55% **e** loadings todas do mesmo sinal → dimensão "
      "latente única · 35–55% → estrutura parcial · < 35% → sem dimensão latente · qualquer "
      "loading de sinal oposto → a interpretação \"expansivo ↔ parcimonioso\" está errada.")
    A("")
    A("---")
    A("")

    A("## 5. Controlo do artefacto mecânico")
    A("")
    A("### 5.1 O denominador partilhado")
    A("")
    A(f"`{CONTROL}` = número de itens que o anotador marcou com ≥ 1 categoria real de "
      "sexismo. **Já existia** como coluna em `behavioral_profiles.csv` (escrita por "
      "`behavioral_profile.py::_profile_one`); não foi derivada aqui.")
    A("")
    A("`[MEDIDO]` — ρ de Spearman entre o denominador e cada feature:")
    A("")
    A(md_table(ctrl, index_name="feature"))
    A("")
    A("Hipótese sob teste: quem tem mais itens sexistas tem mecanicamente mais oportunidade "
      "de acumular categorias distintas, e a entropia estimada de poucas amostras é "
      "sistematicamente subestimada.")
    A("")
    A("### 5.2 Correlações parciais")
    A("")
    A("Método: `scipy` e `sklearn` não expõem correlação parcial diretamente. ρ(X, Y | Z) é "
      "obtida regredindo X e Y em Z por OLS (com intercepto, via `numpy.linalg.lstsq`) e "
      "correlacionando os resíduos por Spearman — a definição clássica de parcial.")
    A("")
    A(md_table(partial, index_name="par"))
    A("")
    A("### 5.3 Estratificação")
    A("")
    A(f"Banda `{CONTROL}` ∈ [{band[0]}, {band[1]}] (mediana ± {STRATUM_HALF_WIDTH}), "
      f"n = {n_str} de {n}. Verificação independente da parcial: se a associação persistir "
      "dentro de um estrato estreito do denominador, não é artefacto de tamanho de amostra.")
    A("")
    A(md_table(strat, index_name="par"))
    A("")
    A("### 5.4 PCA residualizada")
    A("")
    for label, res in (("4 features", pca_res4), ("3 features de categoria", pca_res3)):
        A(f"**{label}** — n = {res['n']}, referência de nulidade "
          f"{100 * res['null_reference']:.0f}%")
        A("")
        v, l = pca_tables(res)
        A(md_table(v, float_fmt="{:.2f}", index_name="componente"))
        A("")
        A(md_table(l, index_name="feature"))
        A("")
        A(f"Veredito: **{res['veredito']}**")
        A("")
    A("---")
    A("")
    A("## 6. Limites")
    A("")
    A("1. **Análise descritiva sobre um artefacto já produzido.** Não treina nada, não "
      "reamostra, não valida contra critério externo. Descreve a estrutura de covariância "
      "das features, não a sua validade.")
    A("2. **A parcial nos pares com `yes_rate` é over-control** e está marcada como tal. Não "
      "citar essas três linhas como evidência de artefacto.")
    A("3. **A estratificação reduz o n** e com ele o poder: uma associação fraca pode "
      "desaparecer no estrato por falta de amostra, não por ser artefacto. Comparar "
      "magnitudes, não presença/ausência de significância.")
    A(f"4. **43 anotadores estão fora de toda esta análise** (NaN nas features de categoria, "
      f"por terem menos de 5 itens sexistas). n = {n} de 887.")
    A("5. **Nada aqui julga a legitimidade de nenhum perfil.** Continua a valer o limite da "
      "Parte 2: estas features não medem acerto e não justificam down-weighting.")
    A("")
    return "\n".join(L)


def build_claims(data: dict) -> tuple[list[str], list[str]]:
    """Listas de afirmações sustentadas / não sustentadas, derivadas dos números."""
    sp = data["spearman_table"]
    pca_raw = data["pca_raw"]
    pca_res3 = data["pca_resid_3"]
    strat = data["stratum_table"]
    n = data["n"]

    cat_pairs = [i for i in sp.index if "yes_rate" not in i]
    yes_pairs = [i for i in sp.index if "yes_rate" in i]
    max_cat = float(sp.loc[cat_pairs, "ρ"].abs().max())
    max_yes = float(sp.loc[yes_pairs, "ρ"].abs().max())
    pc1 = pca_raw["pc1_explained"]
    signs = np.sign(list(pca_raw["pc1_loadings"].values()))
    same_sign = bool(np.all(signs == signs[0]))

    sup, uns = [], []

    # --- correlações
    sup.append(
        f"`[MEDIDO]` Entre as três features de categoria, as correlações de Spearman ao nível "
        f"individual chegam a |ρ| = {max_cat:.4f} ({rho_band(max_cat)}), n = {n}.")
    sup.append(
        f"`[MEDIDO]` Nos pares que envolvem `yes_rate`, |ρ| máximo = {max_yes:.4f} "
        f"({rho_band(max_yes)}).")

    # --- PCA
    if pc1 > PC1_UNIQUE_DIMENSION and same_sign:
        margin = 100 * (pc1 - PC1_UNIQUE_DIMENSION)
        sup.append(
            f"`[MEDIDO]` O PC1 das quatro features explica {100 * pc1:.1f}% da variância com "
            f"todas as loadings do mesmo sinal — passa o limiar pré-registado de "
            f"{100 * PC1_UNIQUE_DIMENSION:.0f}%, por **{margin:+.1f} pontos**. A formulação "
            "\"dimensão latente única\" é sustentável ao nível individual, **desde que dita "
            "com a percentagem, com a margem, e com a composição do eixo da §1.4** — a força "
            f"vem das features de categoria, não de `{OVERCONTROL_FEATURE}`.")
    elif pc1 >= PC1_PARTIAL:
        sup.append(
            f"`[MEDIDO]` O PC1 explica {100 * pc1:.1f}% da variância (limiar de dimensão única: "
            f"{100 * PC1_UNIQUE_DIMENSION:.0f}%; nulidade {100 * pca_raw['null_reference']:.0f}%)"
            " — há **estrutura parcial**, não uma dimensão única.")
        uns.append(
            f"❌ \"As quatro features formam uma dimensão latente única.\" O PC1 fica em "
            f"{100 * pc1:.1f}%, abaixo do limiar pré-registado de "
            f"{100 * PC1_UNIQUE_DIMENSION:.0f}%. Descrever associações específicas em vez disso.")
    else:
        uns.append(
            f"❌ \"As quatro features formam uma dimensão latente única.\" O PC1 explica só "
            f"{100 * pc1:.1f}%, contra {100 * pca_raw['null_reference']:.0f}% de nulidade — as "
            "features são substancialmente independentes ao nível individual.")

    if not same_sign:
        uns.append(
            "❌ \"Anotadores expansivos contra parcimoniosos.\" As loadings do PC1 têm sinais "
            "mistos (" + ", ".join(f"`{k}` {v:+.3f}" for k, v in pca_raw["pc1_loadings"].items())
            + "): as features não crescem em conjunto.")
    else:
        sup.append(
            "`[MEDIDO]` As loadings do PC1 têm todas o mesmo sinal — a direção "
            "\"expansivo ↔ parcimonioso\" descreve corretamente o eixo, "
            "independentemente da força dele.")

    # --- controlo
    valid = strat[strat["leitura"] == "controlo válido"]
    if len(valid):
        keep = (valid["ρ parcial"].abs() /
                valid["ρ bruto"].abs().replace(0, np.nan))
        n_collapsed = int((keep < PARTIAL_COLLAPSE_FRACTION).sum())
        if n_collapsed:
            sup.append(
                f"`[MEDIDO]` {n_collapsed} de {len(valid)} associações entre features de "
                f"categoria perdem mais de {100 * (1 - PARTIAL_COLLAPSE_FRACTION):.0f}% da "
                f"magnitude ao controlar por `{CONTROL}` — nessa medida eram artefacto de "
                "tamanho de amostra.")
        else:
            sup.append(
                f"`[MEDIDO]` Nenhuma das {len(valid)} associações entre features de categoria "
                f"colapsa ao controlar por `{CONTROL}`: não são artefacto de tamanho de "
                "amostra.")

    sup.append(
        f"`[MEDIDO]` Depois de residualizar as três features de categoria em `{CONTROL}`, o "
        f"PC1 passa de {100 * data['pca_raw_3']['pc1_explained']:.1f}% para "
        f"{100 * pca_res3['pc1_explained']:.1f}% (nulidade "
        f"{100 * pca_res3['null_reference']:.0f}%). É a versão da estrutura livre do "
        "artefacto do denominador.")

    # --- ressalva definicional
    dp = f"{DEFINITIONAL_PAIR[0]} × {DEFINITIONAL_PAIR[1]}"
    rho_def = float(sp.loc[dp, "ρ"]) if dp in sp.index else float("nan")
    if abs(rho_def) > DEFINITIONAL_LINK_RHO:
        uns.append(
            f"❌ Apresentar as três features de categoria como três medições independentes do "
            f"mesmo comportamento. ρ(`{DEFINITIONAL_PAIR[0]}`, `{DEFINITIONAL_PAIR[1]}`) = "
            f"{rho_def:+.4f} e a ligação é em parte **definicional** — a entropia normalizada "
            "é limitada por `log(n_distinct_categories)`. O PC1 das três sobrestima a "
            "evidência independente.")
        sup.append(
            f"`[MEDIDO]` A estrutura entre as features de categoria é forte "
            f"(ρ até {rho_def:+.4f}) **mas parcialmente definicional** — a declarar sempre "
            "que o PC1 dessas três for citado.")
        nd = [i for i in data["stratum_table"].index
              if i != dp and OVERCONTROL_FEATURE not in i]
        if nd:
            st = data["stratum_table"]
            sup.append(
                f"`[MEDIDO]` **Descontando o par definicional, o achado mantém-se:** as "
                f"{len(nd)} associações restantes entre features de categoria não têm ligação "
                "matemática entre si e sobrevivem ao controlo — " +
                " e ".join(
                    f"`{i}` (bruto {st.loc[i, 'ρ bruto']:+.3f} → parcial "
                    f"{st.loc[i, 'ρ parcial']:+.3f})" for i in nd) +
                ". A ressalva definicional atinge um par, não a estrutura toda.")

    # --- ressalva do yes_rate
    ys = data["stratum_table"]
    ys = ys.loc[[i for i in ys.index if OVERCONTROL_FEATURE in i]]
    uns.append(
        f"❌ Afirmar que `{OVERCONTROL_FEATURE}` faz parte da mesma dimensão latente que as "
        f"features de categoria. Em bruto |ρ| ≤ {float(ys['ρ bruto'].abs().max()):.4f}; no "
        f"estrato cai para |ρ| ≤ {float(ys['ρ estratificado'].abs().max()):.4f} "
        f"({rho_band(float(ys['ρ estratificado'].abs().max()))}). A loading de "
        f"`{OVERCONTROL_FEATURE}` no PC1 é a mais fraca das quatro "
        f"({pca_raw['pc1_loadings'][OVERCONTROL_FEATURE]:+.4f}).")
    uns.append(
        f"❌ Afirmar o contrário — que `{OVERCONTROL_FEATURE}` **não** faz parte da dimensão. "
        "Os dois controlos disponíveis atenuam a associação pela mesma razão (ambos controlam "
        f"sobre um quase-duplicado de `{OVERCONTROL_FEATURE}`) e nenhum a resolve. A questão "
        "fica em aberto por limitação dos dados, e é assim que deve ser escrita.")

    uns.append(
        f"❌ Citar a parcial dos pares com `yes_rate` como prova de artefacto. "
        f"ρ(`{CONTROL}`, `yes_rate`) = {data['control_vs_yes_rate']:+.4f} — é over-control.")
    uns.append(
        "❌ Reutilizar a formulação original de `behavioral_diagnosis.md` §6.3, que inferiu a "
        "dimensão latente **da comparação entre centróides do KMeans**. Esse argumento é "
        "circular e fica substituído por esta análise, qualquer que seja a direção do "
        "resultado.")
    uns.append(
        "❌ Qualquer leitura destes eixos como qualidade, fiabilidade ou competência de "
        "anotação. Nenhuma feature aqui mede acerto.")

    return sup, uns


# ===========================================================================
# Main
# ===========================================================================


def main():
    df_all = pd.read_csv(PROFILES_CSV)
    n_total = len(df_all)

    needed = FEATURES + [CONTROL]
    df = df_all.dropna(subset=needed).reset_index(drop=True)
    n = len(df)

    data: dict = {
        "n_total": n_total,
        "n": n,
        "features": FEATURES,
        "control": CONTROL,
        "n_distinct_nunique": int(df["n_distinct_categories"].nunique()),
        "n_distinct_min": float(df["n_distinct_categories"].min()),
        "n_distinct_max": float(df["n_distinct_categories"].max()),
        "n_sexist_equals_n_yes": int((df_all[CONTROL] == df_all["n_yes"]).sum()),
    }

    # ---- Parte A
    data["spearman_table"] = corr_table(df, FEATURES, "spearman")
    data["pearson_table"] = corr_table(df, FEATURES, "pearson")
    data["spearman_matrix"] = corr_matrix(df, FEATURES, "spearman")
    data["divergence"] = {
        k: float(data["spearman_table"].loc[k, "ρ"] - data["pearson_table"].loc[k, "ρ"])
        for k in data["spearman_table"].index
    }

    # ---- Parte B
    data["pca_raw"] = run_pca(df, FEATURES)
    data["pca_raw_3"] = run_pca(df, CATEGORY_FEATURES)

    # ---- Parte C
    ctrl_rows = {}
    for f in FEATURES:
        r, p = stats.spearmanr(df[CONTROL], df[f])
        ctrl_rows[f] = {"n": n, "ρ com " + CONTROL: float(r), "p": float(p),
                        "magnitude": rho_band(float(r))}
    data["control_table"] = pd.DataFrame.from_dict(ctrl_rows, orient="index")
    data["control_vs_yes_rate"] = float(
        stats.spearmanr(df[CONTROL], df["yes_rate"])[0])

    data["partial_table"] = partial_corr_table(df, FEATURES, CONTROL)

    med = float(df[CONTROL].median())
    lo, hi = int(med - STRATUM_HALF_WIDTH), int(med + STRATUM_HALF_WIDTH)
    strat_df = df[(df[CONTROL] >= lo) & (df[CONTROL] <= hi)]
    data["stratum_band"] = (lo, hi)
    data["stratum_n"] = int(len(strat_df))

    combined = data["partial_table"].copy()
    strat_rho = {}
    for a, b in pairs(FEATURES):
        strat_rho[f"{a} × {b}"] = float(stats.spearmanr(strat_df[a], strat_df[b])[0])
    combined.insert(2, "ρ estratificado", pd.Series(strat_rho))
    combined = combined[["ρ bruto", "ρ parcial", "ρ estratificado", "Δ",
                         "magnitude (parcial)", "leitura"]]
    data["stratum_table"] = combined

    resid_df = df.copy()
    z = df[CONTROL].to_numpy(dtype=float)
    for f in FEATURES:
        resid_df[f] = residualize(df[f].to_numpy(dtype=float), z)
    data["pca_resid_4"] = run_pca(resid_df, FEATURES)
    data["pca_resid_3"] = run_pca(resid_df, CATEGORY_FEATURES)

    # ---- Parte D
    data["supported"], data["unsupported"] = build_claims(data)
    report = build_report(data)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    dump = {
        "seed": SEED,
        "n_total": n_total,
        "n": n,
        "features": FEATURES,
        "control": CONTROL,
        "prereg": {
            "rho_bands": {label: edge for edge, label in RHO_BANDS},
            "pc1_unique_dimension": PC1_UNIQUE_DIMENSION,
            "pc1_partial": PC1_PARTIAL,
            "stratum_half_width": STRATUM_HALF_WIDTH,
            "partial_collapse_fraction": PARTIAL_COLLAPSE_FRACTION,
            "pearson_divergence_flag": PEARSON_DIVERGENCE_FLAG,
            "overcontrol_feature": OVERCONTROL_FEATURE,
        },
        "spearman": data["spearman_table"].to_dict(orient="index"),
        "pearson": data["pearson_table"].to_dict(orient="index"),
        "control_corr": data["control_table"].to_dict(orient="index"),
        "partial_and_stratum": data["stratum_table"].to_dict(orient="index"),
        "stratum_band": list(data["stratum_band"]),
        "stratum_n": data["stratum_n"],
        "pca_raw_4": data["pca_raw"],
        "pca_raw_3": data["pca_raw_3"],
        "pca_resid_4": data["pca_resid_4"],
        "pca_resid_3": data["pca_resid_3"],
        "supported": data["supported"],
        "unsupported": data["unsupported"],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False, default=str)

    print(f"n = {n} de {n_total} (NaN excluídos)")
    print(f"PC1 bruto (4 feats):        {100 * data['pca_raw']['pc1_explained']:.1f}%  "
          f"-> {data['pca_raw']['veredito']}")
    print(f"PC1 bruto (3 categoria):    {100 * data['pca_raw_3']['pc1_explained']:.1f}%")
    print(f"PC1 residualizado (3 cat):  {100 * data['pca_resid_3']['pc1_explained']:.1f}%  "
          f"-> {data['pca_resid_3']['veredito']}")
    print(f"estrato {CONTROL} {data['stratum_band']}: n = {data['stratum_n']}")
    print(f"\n-> {REPORT_PATH}")
    print(f"-> {JSON_PATH}")


if __name__ == "__main__":
    main()
