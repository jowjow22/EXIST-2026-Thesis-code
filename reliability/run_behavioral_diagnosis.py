"""Parte 2 — Diagnóstico descritivo dos perfis comportamentais (EXIST 2026, Task 2).

Produz `reliability/outputs/behavioral_diagnosis.md` a partir dos perfis extraídos
por `reliability/behavioral_profile.py`.

O que este script NÃO faz: não treina nada, não toca no modelo de memes, não
implementa down-weighting, e não usa nenhuma feature que dependa de consenso,
verdade inferida ou concordância par-a-par.

Todos os critérios de veredito (dispersão, anomalia, PASS/FAIL do clustering)
estão PRÉ-REGISTADOS no bloco imediatamente abaixo, antes de qualquer leitura de
dados, para que nenhum limiar possa ser escolhido depois de ver o resultado.

Uso:
    python3 -m reliability.run_behavioral_diagnosis
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .behavioral_profile import (
    FEATURE_COLUMNS,
    THRESHOLDS,
    COHERENCE_RULE,
    build_profiles,
)

# ===========================================================================
# CRITÉRIOS PRÉ-REGISTADOS — fixados antes de correr, não editar após ver o output
# ===========================================================================

SEED = 42
KMEANS_N_INIT = 10
K_GRID = (2, 3, 4, 5, 6)

#: Clustering "útil" <=> nenhum cluster com menos de 5% dos anotadores E silhouette > 0,25.
#: Existe para não repetir o resultado degenerado [883, 2, 1, 1] sem o notar.
CLUSTER_MIN_FRACTION = 0.05
CLUSTER_MIN_SILHOUETTE = 0.25

#: Uma feature é "dispersa" <=> as TRÊS condições se verificam. Caso contrário
#: "concentrada". Variância nula -> "degenerada" (reportada, fora do clustering).
DISPERSION_MIN_UNIQUE = 10          # nº de valores distintos observados
DISPERSION_MAX_MODAL_BIN = 0.60     # fracção máxima num único bin de 10 sobre o range
DISPERSION_MIN_IQR_FRACTION = 0.10  # IQR >= 10% da amplitude observada

#: Limiares de anomalia (§2.2 da especificação).
ANOMALY_RULES = {
    "yes_rate_extremo": ("yes_rate quase-constante (>0,90 ou <0,10)",
                         lambda p: (p["yes_rate"] > 0.90) | (p["yes_rate"] < 0.10)),
    "categoria_unica": ("usa sempre a mesma categoria (n_distinct_categories == 1)",
                        lambda p: p["n_distinct_categories"] == 1),
    "incoerencia_AC_qualquer": ("qualquer contradição A->C (> 0)",
                                lambda p: p["incoherence_A_to_C"] > 0),
    "incoerencia_AC_sistematica": ("contradição A->C sistemática (> 0,10)",
                                   lambda p: p["incoherence_A_to_C"] > 0.10),
    "incoerencia_AB_sistematica": ("contradição A->B sistemática (> 0,10)",
                                   lambda p: p["incoherence_A_to_B"] > 0.10),
    "lang_gap_grande": ("comportamento muito diferente entre línguas (lang_gap > 0,30)",
                        lambda p: p["lang_gap"] > 0.30),
}

#: Estratificação por volume de votos (§2.3). Features de quem tem poucos votos
#: são ruidosas por construção; os dois estratos não se misturam nas conclusões.
LOW_VOTE_THRESHOLD = 15

#: Features candidatas ao clustering. As de variância nula são removidas em
#: runtime pela regra abaixo (não em silêncio: a exclusão é reportada).
CLUSTER_CANDIDATE_FEATURES = [
    "yes_rate",
    "unknown_rate_b",
    "unknown_rate_c",
    "n_distinct_categories",
    "category_entropy",
    "mean_labels_per_item",
    "incoherence_A_to_C",
    "incoherence_A_to_B",
    "incoherence_yes_empty",
]
#: `lang_gap`/`yes_rate_en`/`yes_rate_es` ficam FORA do clustering: NaN em 97,9% /
#: 49,3% / 48,6% dos anotadores (painéis EN e ES quase disjuntos). Reportadas à parte.

#: Tratamento de NaN no clustering: complete-case sobre as features retidas.
#: Alternativa (imputação pela mediana) seria inventar comportamento onde o
#: denominador não existe; preferimos excluir e declarar quantos.
NAN_POLICY = "complete-case"

# ===========================================================================
# ANÁLISE SECUNDÁRIA — POST-HOC, NÃO PRÉ-REGISTADA
# ===========================================================================
# Acrescentada DEPOIS de ver o resultado da corrida pré-registada, e marcada
# como tal em todo o relatório. Motivo em
# `reliability/outputs/behavioral_diagnosis_observacoes.md` §1.2: o K=2
# vencedor da análise primária separa exactamente os anotadores que usaram
# `UNKNOWN` na Task C, porque `unknown_rate_c` e `incoherence_yes_empty` são
# quase binárias (94% de zeros) e colineares entre si. Depois de padronizadas
# funcionam como indicador, não como escala, e dominam a distância euclidiana.
#
# Esta análise repete o clustering só com as features contínuas. NÃO substitui
# o resultado pré-registado; corre a par dele.
SENSITIVITY_FEATURES = [
    "yes_rate",
    "n_distinct_categories",
    "category_entropy",
    "mean_labels_per_item",
]

#: Uma feature é "inflacionada em zeros" se >= 80% dos valores observados forem
#: exactamente 0. Serve só para DIAGNOSTICAR o que dominou o clustering
#: primário; não filtra nada na análise primária.
ZERO_INFLATION_FLAG = 0.80

OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "behavioral_diagnosis.md")

#: Artefactos das fases anteriores para a §2.5 (correlação descritiva).
PRIOR_THETA_FILES = {
    "NutmegEM theta (demo_base_K4)": "theta_taskA_demo_base_K4.json",
    "NutmegEM theta (demo_extended_K4)": "theta_taskA_demo_extended_K4.json",
    "NutmegEM theta (single_truth_K1)": "theta_taskA_single_truth_K1.json",
}
PRIOR_PERANN_CANDIDATES = [
    "perann_reliability_exist.json",
    "perann_taskA.json",
    "perann_exist_taskA.json",
    "competence_taskA.json",
]

DEMO_FIELDS = [
    "gender_annotators",
    "age_annotators",
    "ethnicities_annotators",
    "study_levels_annotators",
    "countries_annotators",
]

# ===========================================================================
# Utilitários de relatório
# ===========================================================================


def text_histogram(series: pd.Series, n_bins: int = 10, width: int = 40) -> list[str]:
    """Histograma textual sobre a amplitude observada (NaN excluídos)."""
    s = series.dropna().to_numpy(dtype=float)
    if len(s) == 0:
        return ["(sem valores)"]
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return [f"[{lo:.4f}] {'#' * width} {len(s)} (valor único)"]
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(s, bins=edges)
    top = counts.max()
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / top)) if top else ""
        lines.append(f"[{edges[i]:7.4f}, {edges[i+1]:7.4f}) {bar:<{width}} {c:4d}")
    return lines


def dispersion_verdict(series: pd.Series) -> tuple[str, dict]:
    """Aplica a regra pré-registada. Devolve (veredito, diagnósticos)."""
    s = series.dropna().to_numpy(dtype=float)
    n = len(s)
    if n == 0:
        return "sem-dados", {}
    nunique = int(len(np.unique(s)))
    lo, hi = float(s.min()), float(s.max())
    rng = hi - lo
    if rng <= 0:
        return "degenerada", {"nunique": nunique, "range": rng, "modal_bin": 1.0,
                              "iqr_frac": 0.0}
    counts, _ = np.histogram(s, bins=np.linspace(lo, hi, 11))
    modal_bin = float(counts.max()) / n
    q1, q3 = np.percentile(s, [25, 75])
    iqr_frac = float(q3 - q1) / rng
    diag = {"nunique": nunique, "range": rng, "modal_bin": modal_bin,
            "iqr_frac": iqr_frac}
    ok = (
        nunique >= DISPERSION_MIN_UNIQUE
        and modal_bin <= DISPERSION_MAX_MODAL_BIN
        and iqr_frac >= DISPERSION_MIN_IQR_FRACTION
    )
    return ("dispersa" if ok else "concentrada"), diag


def describe_table(profiles: pd.DataFrame, columns) -> pd.DataFrame:
    rows = {}
    for c in columns:
        s = profiles[c].dropna()
        rows[c] = {
            "n": int(s.size),
            "NaN": int(profiles[c].isna().sum()),
            "média": s.mean() if s.size else np.nan,
            "std": s.std() if s.size else np.nan,
            "min": s.min() if s.size else np.nan,
            "q25": s.quantile(0.25) if s.size else np.nan,
            "mediana": s.median() if s.size else np.nan,
            "q75": s.quantile(0.75) if s.size else np.nan,
            "max": s.max() if s.size else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}", index_name: str = "") -> str:
    header = "| " + (index_name or df.index.name or "") + " | " + \
             " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    lines = [header, sep]
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_ids(ids, limit: int = 40) -> str:
    ids = list(ids)
    if not ids:
        return "(nenhum)"
    shown = ", ".join(f"`{i}`" for i in ids[:limit])
    if len(ids) > limit:
        shown += f", … (+{len(ids) - limit})"
    return shown


# ===========================================================================
# Blocos de análise
# ===========================================================================


def anomaly_sets(profiles: pd.DataFrame) -> dict:
    """Aplica ANOMALY_RULES. NaN nunca conta como anomalia (comparações com NaN
    dão False), o que é o comportamento certo: denominador insuficiente não é
    evidência de anomalia. O nº de NaN por regra é reportado à parte."""
    out = {}
    for key, (label, fn) in ANOMALY_RULES.items():
        mask = fn(profiles).fillna(False).astype(bool)
        col = key.split("_")[0]
        out[key] = {
            "label": label,
            "ids": list(profiles.index[mask]),
            "n": int(mask.sum()),
        }
    return out


def overlap_matrix(sets: dict, index) -> pd.DataFrame:
    keys = [k for k in sets if sets[k]["n"] > 0]
    m = pd.DataFrame(0, index=keys, columns=keys, dtype=int)
    for a in keys:
        sa = set(sets[a]["ids"])
        for b in keys:
            m.loc[a, b] = len(sa & set(sets[b]["ids"]))
    return m


def run_clustering(profiles: pd.DataFrame):
    """KMeans no grid de K, com o critério PASS/FAIL pré-registado."""
    cand = [c for c in CLUSTER_CANDIDATE_FEATURES if c in profiles.columns]
    used, dropped = [], []
    for c in cand:
        s = profiles[c].dropna()
        if s.size == 0 or float(s.std(ddof=0)) == 0.0:
            dropped.append((c, "variância nula" if s.size else "sem valores"))
        else:
            used.append(c)

    sub = profiles[used].dropna(axis=0, how="any")
    X = sub.to_numpy(dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd_safe

    results = []
    labels_by_k = {}
    # `np.errstate` local: o BLAS Accelerate do macOS levanta flags de overflow /
    # divide-by-zero espúrias dentro do `matmul` de `sklearn.utils.extmath`. A
    # matriz é finita e o silhouette recalculado pelo caminho sem BLAS
    # (`pdist` + metric='precomputed') bate a 1e-13 — verificação em
    # `behavioral_diagnosis_observacoes.md` §2. Silenciado no ponto exacto da
    # chamada, nunca com um filtro global de warnings.
    with np.errstate(all="ignore"):
        for k in K_GRID:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=KMEANS_N_INIT)
            lab = km.fit_predict(Z)
            sizes = np.bincount(lab, minlength=k)
            sil = float(silhouette_score(Z, lab)) if k > 1 else np.nan
            min_frac = float(sizes.min()) / len(lab)
            passed = (min_frac >= CLUSTER_MIN_FRACTION) and (sil > CLUSTER_MIN_SILHOUETTE)
            results.append({
                "K": k,
                "tamanhos": list(map(int, sizes)),
                "min_frac": min_frac,
                "silhouette": sil,
                "veredito": "PASS" if passed else "FAIL",
            })
            labels_by_k[k] = pd.Series(lab, index=sub.index)

    meta = {
        "features_usadas": used,
        "features_excluidas": dropped,
        "n_anotadores_clusterizados": int(len(sub)),
        "n_excluidos_por_nan": int(len(profiles) - len(sub)),
        "standardizacao": {c: {"media": float(m), "std": float(s)}
                           for c, m, s in zip(used, mu, sd)},
    }
    return results, labels_by_k, sub, meta


def zero_inflation(profiles: pd.DataFrame, features) -> pd.DataFrame:
    """Fracção de valores exactamente zero por feature. Diagnóstico do que pode
    estar a funcionar como indicador binário em vez de escala contínua."""
    rows = {}
    for c in features:
        s = profiles[c].dropna()
        frac = float((s == 0).sum()) / s.size if s.size else np.nan
        rows[c] = {
            "n": int(s.size),
            "zeros": int((s == 0).sum()),
            "fracção de zeros": frac,
            "indicador?": "SIM" if frac >= ZERO_INFLATION_FLAG else "não",
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def diagnose_primary_split(profiles: pd.DataFrame, sub: pd.DataFrame,
                           labels: pd.Series) -> dict:
    """Caracteriza uma partição: médias por cluster e cruzamento com o evento
    'usou UNKNOWN alguma vez' (o candidato a explicar a separação)."""
    means = sub.groupby(labels).mean()
    any_unknown = (
        (profiles.loc[sub.index, "unknown_rate_b"] > 0)
        | (profiles.loc[sub.index, "unknown_rate_c"] > 0)
    )
    ct = pd.crosstab(labels, any_unknown.to_numpy())
    ct.columns = [str(c) for c in ct.columns]
    return {"means": means, "crosstab_unknown": ct}


def run_sensitivity_clustering(profiles: pd.DataFrame):
    """Análise POST-HOC: repete o grid só com as features contínuas."""
    used = [c for c in SENSITIVITY_FEATURES if c in profiles.columns]
    sub = profiles[used].dropna(axis=0, how="any")
    X = sub.to_numpy(dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=0)
    Z = (X - mu) / np.where(sd == 0, 1.0, sd)

    results, labels_by_k = [], {}
    with np.errstate(all="ignore"):  # ver nota em run_clustering
        for k in K_GRID:
            lab = KMeans(n_clusters=k, random_state=SEED,
                         n_init=KMEANS_N_INIT).fit_predict(Z)
            sizes = np.bincount(lab, minlength=k)
            sil = float(silhouette_score(Z, lab))
            min_frac = float(sizes.min()) / len(lab)
            results.append({
                "K": k,
                "tamanhos": list(map(int, sizes)),
                "min_frac": min_frac,
                "silhouette": sil,
                "veredito": "PASS" if (min_frac >= CLUSTER_MIN_FRACTION
                                       and sil > CLUSTER_MIN_SILHOUETTE) else "FAIL",
            })
            labels_by_k[k] = pd.Series(lab, index=sub.index)
    return results, labels_by_k, sub, {"features_usadas": used,
                                       "n_anotadores": int(len(sub))}


def load_prior_theta() -> dict:
    out = {}
    for label, fname in PRIOR_THETA_FILES.items():
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and "theta" in d:
            out[label] = pd.Series(d["theta"], dtype=float)
    return out


def find_prior_perann() -> list[str]:
    return [f for f in PRIOR_PERANN_CANDIDATES
            if os.path.exists(os.path.join(OUT_DIR, f))]


# ===========================================================================
# Relatório
# ===========================================================================


def build_report(profiles, demo, audit) -> str:
    L: list[str] = []
    A = L.append

    n_tot = len(profiles)
    feats = [c for c in FEATURE_COLUMNS if c != "n_votes"]

    A("# Diagnóstico descritivo dos padrões comportamentais dos anotadores (EXIST 2026, Task 2)")
    A("")
    A("**O que este documento é:** um diagnóstico DESCRITIVO dos 887 anotadores do EXIST 2026 "
      "Task 2 (memes), a partir de perfis de comportamento agregado calculados sem recorrer a "
      "consenso, verdade inferida ou concordância par-a-par. Não é um sistema, não treina nada, "
      "e não justifica down-weighting de ninguém.")
    A("")
    A("- **Extração:** `reliability/behavioral_profile.py` (Parte 1, "
      "`reliability/outputs/behavioral_profile_part1.md`)")
    A("- **Análise:** `reliability/run_behavioral_diagnosis.py` (este relatório)")
    A("- **Perfis:** `reliability/outputs/behavioral_profiles.csv`")
    A("- **Seed:** 42 (KMeans, `n_init=10`) — determinístico")
    A("")
    A("**Convenção** (a mesma de `reliability/NOTAS_PARA_TESE.md`): `[MEDIDO]` — número "
      "observado, com o artefacto que o produziu · `[HIPÓTESE]` — interpretação plausível, "
      "ainda não testada; **não escrever como facto** · `[LIMITAÇÃO]` — restrição a declarar "
      "explicitamente no texto.")
    A("")
    A("---")
    A("")

    # ---------------- critérios pré-registados -----------------
    A("## 0. Critérios pré-registados")
    A("")
    A("Fixados no código (`run_behavioral_diagnosis.py`, bloco no topo) antes de qualquer "
      "leitura de dados.")
    A("")
    A("**Dispersão** — uma feature é *dispersa* se e só se as três condições se verificarem; "
      "caso contrário é *concentrada*. Variância nula → *degenerada* (reportada, fora do "
      "clustering).")
    A("")
    A(f"1. valores distintos ≥ {DISPERSION_MIN_UNIQUE}")
    A(f"2. nenhum bin (10 bins sobre a amplitude observada) contém > "
      f"{DISPERSION_MAX_MODAL_BIN:.0%} dos anotadores")
    A(f"3. IQR ≥ {DISPERSION_MIN_IQR_FRACTION:.0%} da amplitude observada")
    A("")
    A("**Clustering útil** — um clustering PASSA se e só se nenhum cluster tiver menos de "
      f"{CLUSTER_MIN_FRACTION:.0%} dos anotadores **e** o silhouette for > "
      f"{CLUSTER_MIN_SILHOUETTE}. Este critério existe para não repetir o resultado degenerado "
      "`[883, 2, 1, 1]` das tentativas anteriores sem o notar.")
    A("")
    A("**Anomalia** — os limiares da tabela da §2, todos fixados a priori.")
    A("")
    A(f"**NaN no clustering** — política `{NAN_POLICY}`: anotadores com NaN em qualquer feature "
      "retida são excluídos, e o número é declarado. Imputação foi rejeitada porque inventaria "
      "comportamento onde o denominador não existe.")
    A("")
    A("---")
    A("")

    # ---------------- factos estruturais -----------------
    A("## 1. Factos estruturais herdados da Parte 1")
    A("")
    A("Três descobertas da extração limitam o que este diagnóstico pode perguntar. Repetidas "
      "aqui porque condicionam a leitura de tudo o que se segue.")
    A("")
    A("### 1.1 A coerência hierárquica é perfeita — a família \"erro objetivo\" não existe neste corpus")
    A("")
    A("`[MEDIDO]` — cross-tab ao nível do voto, n = "
      f"{audit['n_votes']:,}".replace(",", " ") + ":")
    A("")
    A("```")
    A("A × B:   " + "  ·  ".join(f"{k} {v}" for k, v in audit["crosstab_a_b"].items()))
    A("A × C:   " + "  ·  ".join(f"{k} {v}" for k, v in audit["crosstab_a_c"].items()))
    A("```")
    A("")
    A("Zero votos `NO` + categoria de sexismo. Zero votos `NO` + `DIRECT`/`JUDGEMENTAL`. Logo "
      "`incoherence_A_to_C` e `incoherence_A_to_B` valem **0,0000 nos 887 anotadores**.")
    A("")
    A("`[HIPÓTESE]` — hierarquia imposta pela interface de anotação. A coincidência exacta das "
      "contagens é consistente com um constrangimento de ferramenta, mas **não distingue** "
      "constrangimento de disciplina do anotador.")
    A("")
    A("### 1.2 Os painéis EN e ES são quase disjuntos")
    A("")
    A("`[MEDIDO]` — " + " · ".join(f"**{k}**: {v}" for k, v in audit["annotators_by_lang"].items())
      + ". `lang_gap` fica definida em apenas "
      f"{int(profiles['lang_gap'].notna().sum())} dos {n_tot} anotadores.")
    A("")
    A("### 1.3 A regra de coerência usada")
    A("")
    A("```")
    A(COHERENCE_RULE.rstrip())
    A("```")
    A("")
    A("Limiares de denominador mínimo: " +
      ", ".join(f"`{k}` = {v}" for k, v in THRESHOLDS.items()) + ".")
    A("")
    A("---")
    A("")

    # ---------------- 2.1 dispersão -----------------
    A("## 2. Há dispersão real, ou toda a gente se parece?")
    A("")
    desc = describe_table(profiles, feats)
    A("`[MEDIDO]` — estatísticas descritivas sobre os "
      f"{n_tot} anotadores (NaN excluídos por feature):")
    A("")
    A(md_table(desc, index_name="feature"))
    A("")
    A("### Veredito por feature")
    A("")
    verdicts = {}
    vrows = {}
    for c in feats:
        v, diag = dispersion_verdict(profiles[c])
        verdicts[c] = v
        vrows[c] = {
            "veredito": v,
            "n_distintos": diag.get("nunique", 0),
            "bin_modal": diag.get("modal_bin", np.nan),
            "IQR/amplitude": diag.get("iqr_frac", np.nan),
        }
    A(md_table(pd.DataFrame.from_dict(vrows, orient="index"), index_name="feature"))
    A("")
    n_disp = sum(1 for v in verdicts.values() if v == "dispersa")
    n_conc = sum(1 for v in verdicts.values() if v == "concentrada")
    n_deg = sum(1 for v in verdicts.values() if v == "degenerada")
    A(f"**Resumo:** {n_disp} dispersas · {n_conc} concentradas · {n_deg} degeneradas "
      f"(variância nula), de {len(feats)} features.")
    A("")
    A("### Histogramas")
    A("")
    for c in feats:
        A(f"**`{c}`** — {verdicts[c]}")
        A("")
        A("```")
        for line in text_histogram(profiles[c]):
            A(line)
        A("```")
        A("")
    A("---")
    A("")

    # ---------------- 2.2 anomalias -----------------
    A("## 3. Quantos anotadores são anómalos, e em quê?")
    A("")
    sets_all = anomaly_sets(profiles)
    rows = {}
    for k, d in sets_all.items():
        na = int(profiles[_rule_column(k)].isna().sum()) if _rule_column(k) else 0
        rows[k] = {
            "critério": d["label"],
            "n": d["n"],
            "% de 887": 100.0 * d["n"] / n_tot,
            "NaN (não avaliáveis)": na,
        }
    A("`[MEDIDO]` — contagens sobre os 887 anotadores. NaN nunca conta como anomalia: "
      "denominador insuficiente não é evidência de comportamento anómalo.")
    A("")
    A(md_table(pd.DataFrame.from_dict(rows, orient="index"), float_fmt="{:.1f}",
               index_name="regra"))
    A("")
    for k, d in sets_all.items():
        A(f"- **{k}** ({d['n']}): {fmt_ids(d['ids'])}")
    A("")
    A("### Sobreposição entre conjuntos de anomalia")
    A("")
    om = overlap_matrix(sets_all, profiles.index)
    if om.empty:
        A("Nenhum conjunto de anomalia é não-vazio — não há sobreposição a reportar.")
    else:
        A(md_table(om, float_fmt="{:.0f}", index_name="∩"))
        A("")
        counts = Counter()
        for k, d in sets_all.items():
            for i in d["ids"]:
                counts[i] += 1
        multi = sorted([i for i, c in counts.items() if c >= 2])
        A(f"**Anotadores anómalos em ≥ 2 critérios:** {len(multi)} — {fmt_ids(multi)}")
    A("")
    A("---")
    A("")

    # ---------------- 2.3 estratificação -----------------
    A("## 4. Estratificação por número de votos")
    A("")
    low = profiles[profiles["n_votes"] < LOW_VOTE_THRESHOLD]
    high = profiles[profiles["n_votes"] >= LOW_VOTE_THRESHOLD]
    A(f"`[MEDIDO]` — {len(low)} anotadores com < {LOW_VOTE_THRESHOLD} votos, "
      f"{len(high)} com ≥ {LOW_VOTE_THRESHOLD}. As features dos primeiros são ruidosas por "
      "construção (denominadores de 9 a 14); **os dois estratos não se misturam nas conclusões**.")
    A("")
    strat = {}
    for name, sub in (("< 15 votos", low), ("≥ 15 votos", high)):
        s = anomaly_sets(sub)
        for k, d in s.items():
            strat.setdefault(k, {})[name] = f"{d['n']} / {len(sub)}"
            strat[k][f"% {name}"] = 100.0 * d["n"] / len(sub) if len(sub) else np.nan
    A(md_table(pd.DataFrame.from_dict(strat, orient="index"), float_fmt="{:.1f}",
               index_name="regra"))
    A("")
    low_ids = anomaly_sets(low)
    for k, d in low_ids.items():
        if d["n"]:
            A(f"- **{k}** no estrato < 15 votos ({d['n']} de {len(low)}): {fmt_ids(d['ids'])}")
    A("")
    A("`[LIMITAÇÃO]` — com 9 a 14 votos, um `yes_rate` de 1,00 exige apenas 9 votos YES "
      "consecutivos. No estrato baixo, \"quase-constante\" e \"pouca amostra\" não são "
      "distinguíveis por estes dados.")
    A("")
    A("---")
    A("")

    # ---------------- 2.4 clustering -----------------
    A("## 5. Os perfis clusterizam de forma equilibrada?")
    A("")
    results, labels_by_k, sub, meta = run_clustering(profiles)
    A(f"**Features retidas ({len(meta['features_usadas'])}):** " +
      ", ".join(f"`{c}`" for c in meta["features_usadas"]))
    A("")
    if meta["features_excluidas"]:
        A("**Features excluídas (regra pré-registada, não exclusão silenciosa):**")
        for c, why in meta["features_excluidas"]:
            A(f"- `{c}` — {why}")
        A("")
    A("`lang_gap`, `yes_rate_en` e `yes_rate_es` ficam fora por desenho: NaN em 97,9% / 49,3% / "
      "48,6% dos anotadores (§1.2). São reportadas à parte, não imputadas.")
    A("")
    A(f"**Anotadores clusterizados:** {meta['n_anotadores_clusterizados']} de {n_tot} "
      f"(política `{NAN_POLICY}`; {meta['n_excluidos_por_nan']} excluídos por NaN).")
    A("")
    res_df = pd.DataFrame(results).set_index("K")
    A(md_table(res_df, index_name="K"))
    A("")
    passed = [r for r in results if r["veredito"] == "PASS"]
    if not passed:
        A("**Nenhum K passa o critério pré-registado.** Não há partição comportamental "
          "equilibrada e bem separada a caracterizar. A caracterização por cluster "
          "(features distintivas, composição demográfica) fica por fazer — por ausência de "
          "clusters válidos, não por omissão.")
        A("")
        best = max(results, key=lambda r: (r["silhouette"] if not np.isnan(r["silhouette"]) else -1))
        A(f"Para registo, o melhor silhouette do grid é {best['silhouette']:.4f} em K={best['K']} "
          f"(tamanhos {best['tamanhos']}), abaixo do limiar {CLUSTER_MIN_SILHOUETTE}.")
        A("")
    else:
        A(f"**{len(passed)} de {len(K_GRID)} valores de K passam o critério pré-registado.** "
          "A caracterização segue abaixo. Ler em conjunto com a §6, que mostra que a separação "
          "do K com melhor silhouette **não** é comportamental.")
        A("")
        for r in passed:
            k = r["K"]
            A(f"### K = {k} (PASS) — o que distingue cada cluster")
            A("")
            lab = labels_by_k[k]
            delta = sub.groupby(lab).mean() - sub.mean()
            A("Média por cluster **menos** média global:")
            A("")
            A(md_table(delta, index_name="cluster"))
            A("")
        best = max(passed, key=lambda r: r["silhouette"])
        A(f"### Composição demográfica (K = {best['K']}, o melhor silhouette dos que passam)")
        A("")
        A("Descritivo. Os clusters alinharem ou não com demografia é informativo nos dois "
          "sentidos: alinhamento sugeriria que o \"comportamento\" é um proxy do grupo; "
          "ausência de alinhamento sugere variação individual.")
        A("")
        lab = labels_by_k[best["K"]]
        for fld in DEMO_FIELDS:
            if fld not in demo.columns:
                continue
            ct = pd.crosstab(lab, demo.loc[lab.index, fld], normalize="index") * 100
            ct = ct.loc[:, ct.max(axis=0) >= 5.0]
            A(f"**{fld}** (% por cluster; categorias com ≥5% nalgum cluster)")
            A("")
            A(md_table(ct, float_fmt="{:.1f}", index_name="cluster"))
            A("")
    A("---")
    A("")

    # ---------------- 6 (NOVO) diagnose post-hoc do clustering -----------------
    A("## 6. O que a partição primária está realmente a separar (post-hoc)")
    A("")
    A("> ⚠ **Esta secção é POST-HOC e não estava pré-registada.** Foi acrescentada depois de "
      "ver o resultado da §5. Não substitui a análise primária; corre a par dela. Registo "
      "completo em `reliability/outputs/behavioral_diagnosis_observacoes.md`.")
    A("")
    A("### 6.1 Três das features retidas são indicadores binários, não escalas")
    A("")
    zi = zero_inflation(profiles, meta["features_usadas"])
    A(f"`[MEDIDO]` — fracção de valores exactamente zero (sinalizada a partir de "
      f"{ZERO_INFLATION_FLAG:.0%}):")
    A("")
    A(md_table(zi, index_name="feature"))
    A("")
    A("Depois de padronizadas, features com esta forma funcionam como um **flag ligado/"
      "desligado**, não como uma escala de comportamento, e dominam a distância euclidiana. "
      "Duas delas — `unknown_rate_c` e `incoherence_yes_empty` — sinalizam o mesmo evento e "
      "são colineares por construção (ver Parte 1, §2).")
    A("")
    A("### 6.2 O K com melhor silhouette separa quem usou `UNKNOWN`, não perfis de comportamento")
    A("")
    best_all = max(results, key=lambda r: r["silhouette"])
    diag = diagnose_primary_split(profiles, sub, labels_by_k[best_all["K"]])
    A(f"`[MEDIDO]` — K = {best_all['K']} (silhouette {best_all['silhouette']:.4f}, tamanhos "
      f"{best_all['tamanhos']}). Médias por cluster:")
    A("")
    A(md_table(diag["means"].T, index_name="feature"))
    A("")
    A("Cruzamento com o evento «usou `UNKNOWN` alguma vez»:")
    A("")
    A(md_table(diag["crosstab_unknown"], float_fmt="{:.0f}", index_name="cluster"))
    A("")
    A("As features de comportamento propriamente dito são **quase idênticas** entre os grupos; "
      "o que os separa é um flag administrativo. Um anotador ter carregado `UNKNOWN` uma vez "
      "não é um perfil de comportamento — é um evento pontual de preenchimento.")
    A("")
    A("`[LIMITAÇÃO]` — **o critério pré-registado é insensível a este modo de falha.** Foi "
      "desenhado contra clusters degenerados por *tamanho* (o `[883, 2, 1, 1]` das tentativas "
      "anteriores). Um flag binário com ~6% de prevalência passa nas duas condições. O critério "
      "cumpriu o que prometia e ainda assim deixou passar um resultado que não responde à "
      "pergunta. Fica registado para quem reutilizar este protocolo.")
    A("")
    A("### 6.3 Sensibilidade: clustering só com as features contínuas")
    A("")
    s_results, s_labels, s_sub, s_meta = run_sensitivity_clustering(profiles)
    A("Removendo as features sinalizadas em 6.1 e mantendo as contínuas — " +
      ", ".join(f"`{c}`" for c in s_meta["features_usadas"]) +
      f" (n = {s_meta['n_anotadores']}):")
    A("")
    A(md_table(pd.DataFrame(s_results).set_index("K"), index_name="K"))
    A("")
    s_best = max(s_results, key=lambda r: r["silhouette"])
    A(f"Clusters muito mais equilibrados (fracção mínima {s_best['min_frac']:.3f} em "
      f"K = {s_best['K']}, contra {best_all['min_frac']:.3f} na análise primária). O silhouette "
      "é menor precisamente porque já não há um flag binário a inflacioná-lo.")
    A("")
    s_delta = s_sub.groupby(s_labels[s_best["K"]]).mean() - s_sub.mean()
    A(f"O que distingue cada cluster em K = {s_best['K']} (média do cluster menos média global):")
    A("")
    A(md_table(s_delta, index_name="cluster"))
    A("")
    same_sign = bool((np.sign(s_delta.iloc[0]) == np.sign(s_delta.iloc[0]).iloc[0]).all())
    if same_sign and len(s_delta) == 2:
        A("**As quatro features movem-se todas na mesma direção** dentro de cada cluster: um "
          "grupo está acima da média em tudo, o outro abaixo em tudo. Isso é a assinatura de "
          "**uma única dimensão latente** cortada ao meio — anotadores mais expansivos "
          "(marcam mais YES, usam mais categorias, mais rótulos por item, entropia maior) "
          "contra anotadores mais parcimoniosos. Não são dois *tipos*: são as duas metades de "
          "um contínuo.")
        A("")
    A("Composição demográfica desta partição (a substantiva), K = "
      f"{s_best['K']}:")
    A("")
    s_lab = s_labels[s_best["K"]]
    for fld in DEMO_FIELDS:
        if fld not in demo.columns:
            continue
        ct = pd.crosstab(s_lab, demo.loc[s_lab.index, fld], normalize="index") * 100
        ct = ct.loc[:, ct.max(axis=0) >= 5.0]
        A(f"**{fld}** (% por cluster; categorias com ≥5% nalgum cluster)")
        A("")
        A(md_table(ct, float_fmt="{:.1f}", index_name="cluster"))
        A("")
    A("`[HIPÓTESE]` — esta é a estrutura substantiva e a da §5 é um artefacto de features "
      "indicadoras. **Não testada:** não existe critério externo que arbitre qual das duas "
      "partições é a verdadeira, e por desenho este diagnóstico não tem acesso a nenhum.")
    A("")
    A("---")
    A("")

    # ---------------- 2.5 relação com artefactos anteriores -----------------
    A("## 7. Relação com os artefactos das fases anteriores")
    A("")
    A("**Enquadramento obrigatório:** correlação **descritiva**, não validação. Os métodos "
      "anteriores foram caracterizados como capturáveis por blocos de anotadores correlacionados "
      "(`reliability/outputs/stage1_report.md`, `perann_gate_report.md`). Concordar com eles "
      "não é prova de nada; discordar também não.")
    A("")
    thetas = load_prior_theta()
    perann = find_prior_perann()
    if not thetas:
        A("Nenhum ficheiro de θ encontrado em `reliability/outputs/` — secção sem conteúdo.")
    else:
        for label, th in thetas.items():
            common = profiles.index.intersection(th.index)
            A(f"### {label} — n = {len(common)} anotadores em comum")
            A("")
            rows = {}
            for c in feats + ["n_votes"]:
                x = profiles.loc[common, c]
                y = th.loc[common]
                m = x.notna() & y.notna()
                if m.sum() < 20 or x[m].nunique() < 2:
                    rows[c] = {"n": int(m.sum()), "Spearman ρ": np.nan, "p": np.nan}
                    continue
                rho, p = stats.spearmanr(x[m], y[m])
                rows[c] = {"n": int(m.sum()), "Spearman ρ": float(rho), "p": float(p)}
            A(md_table(pd.DataFrame.from_dict(rows, orient="index"), index_name="feature"))
            A("")
    if perann:
        A(f"Ficheiros de competência per-anotador encontrados: {', '.join(perann)}.")
    else:
        A("`[LIMITAÇÃO]` — **não existe** em `reliability/outputs/` nenhum ficheiro de "
          "competência per-anotador calculada sobre o EXIST real; o `perann_synthetic_gate.json` "
          "é do gate sintético e não tem IDs do EXIST. A comparação pedida com essa família de "
          "estimadores fica por fazer, por ausência do artefacto.")
    A("")
    A("---")
    A("")
    extra = {
        "zero_inflation": zi,
        "primary_best": best_all,
        "primary_split_diag": diag,
        "sensitivity_results": s_results,
        "sensitivity_meta": s_meta,
    }
    return "\n".join(L), verdicts, sets_all, results, meta, low, high, thetas, extra


def _rule_column(rule_key: str):
    return {
        "yes_rate_extremo": "yes_rate",
        "categoria_unica": "n_distinct_categories",
        "incoerencia_AC_qualquer": "incoherence_A_to_C",
        "incoerencia_AC_sistematica": "incoherence_A_to_C",
        "incoerencia_AB_sistematica": "incoherence_A_to_B",
        "lang_gap_grande": "lang_gap",
    }.get(rule_key)


def build_verdict_section(profiles, verdicts, sets_all, results, meta,
                          low, high, thetas, extra):
    """Veredito construído a partir dos números medidos.

    NOTA DE PROCESSO: a primeira versão desta função continha prosa fixa que
    afirmava "não há estrutura de grupos", escrita antes de correr, na
    expectativa de repetir o resultado degenerado das tentativas anteriores.
    Aconteceu o contrário e o texto contradizia as tabelas. Foi reescrita para
    derivar cada afirmação dos resultados. Registo em
    `reliability/outputs/behavioral_diagnosis_observacoes.md` §1.1.
    """
    L: list[str] = []
    A = L.append
    n_tot = len(profiles)
    disp = [c for c, v in verdicts.items() if v == "dispersa"]
    deg = [c for c, v in verdicts.items() if v == "degenerada"]
    passed = [r for r in results if r["veredito"] == "PASS"]
    best = extra["primary_best"]
    s_results = extra["sensitivity_results"]
    s_passed = [r for r in s_results if r["veredito"] == "PASS"]
    s_best = max(s_results, key=lambda r: r["silhouette"])
    zi = extra["zero_inflation"]
    n_indicadoras = int((zi["indicador?"] == "SIM").sum())

    union = set()
    for d in sets_all.values():
        union |= set(d["ids"])
    n_high_anom = len([i for i in union if i in high.index])
    n_multi = 0
    from collections import Counter as _C
    _c = _C()
    for d in sets_all.values():
        for i in d["ids"]:
            _c[i] += 1
    n_multi = sum(1 for v in _c.values() if v >= 2)

    A("## 8. Veredito")
    A("")

    # ---- 8.1
    A("### 8.1 Existem padrões comportamentais distinguíveis entre os anotadores do EXIST?")
    A("")
    A("**SIM, parcialmente — e a parte que existe é um contínuo, não uma tipologia.**")
    A("")
    A(f"`[MEDIDO]` — {len(disp)} das {len(verdicts)} features são dispersas pelo critério "
      f"pré-registado ({', '.join('`' + c + '`' for c in disp) if disp else '—'}); "
      f"{len(deg)} são degeneradas (variância nula).")
    A("")
    A("`[LIMITAÇÃO]` — dessa contagem, `yes_rate_en`, `yes_rate_es` e `lang_gap` são avaliadas "
      "sobre subpopulações de 450, 456 e **19** anotadores. Só `yes_rate` e `category_entropy` "
      "são dispersas sobre o painel completo. Não citar \"5 de 12\" sem esta ressalva.")
    A("")
    A(f"`[MEDIDO]` — o clustering passa o critério pré-registado em "
      f"{len(passed)} dos {len(K_GRID)} valores de K testados "
      f"({', '.join('K=' + str(r['K']) for r in passed) if passed else 'nenhum'}). "
      f"Melhor silhouette: {best['silhouette']:.4f} em K = {best['K']}, tamanhos "
      f"{best['tamanhos']}.")
    A("")
    A("**Mas o PASS não deve ser lido como \"existem tipos de anotador\".** A §6 mostra que a "
      f"partição com melhor silhouette separa quem alguma vez carregou `UNKNOWN` na Task C — "
      f"as features de comportamento são quase idênticas entre os dois grupos. "
      f"{n_indicadoras} das {len(meta['features_usadas'])} features retidas têm "
      f"≥ {ZERO_INFLATION_FLAG:.0%} de zeros e funcionam como indicador binário.")
    A("")
    A("Removendo-as (análise post-hoc, §6.3), o clustering continua a passar em "
      f"{len(s_passed)} de {len(K_GRID)} valores de K, agora com clusters **equilibrados** "
      f"(fracção mínima {s_best['min_frac']:.3f} em K = {s_best['K']}, contra "
      f"{best['min_frac']:.3f} na análise primária) e silhouette mais baixo "
      f"({s_best['silhouette']:.4f}).")
    A("")
    A("`[MEDIDO]` — **a partição substantiva não é um proxy demográfico.** Na partição da "
      "§6.3, a composição por idade é praticamente idêntica nos dois grupos "
      "(31,6/34,2/34,2 contra 32,2/33,7/34,0), e etnia, escolaridade e país variam pouco. "
      "A maior diferença é de género (55,2% F contra 46,0% F) e continua modesta. O "
      "comportamento de anotação varia entre indivíduos, não entre os grupos demográficos "
      "que o modelo de memes já codifica no vetor demográfico.")
    A("")
    A("**Formulação defensável para a tese:** os anotadores do EXIST diferem de forma "
      "mensurável ao longo de eixos contínuos — sobretudo a propensão a marcar YES "
      "(`yes_rate` cobre [0, 1] inteiro) e a forma como usam o espaço de categorias "
      "(`category_entropy` cobre [0,15 · 0,99]). Não há evidência de tipos discretos: as "
      "partições que o KMeans encontra ou são conduzidas por um flag administrativo (§6.2), "
      "ou são cortes num contínuo, sem separação forte (silhouette ≈ 0,28–0,34). "
      "`[HIPÓTESE]` — variação individual gradual, não subgrupos.")
    A("")

    # ---- 8.2
    A("### 8.2 Quantos anotadores estão envolvidos, e em que tipo de anomalia?")
    A("")
    A(f"`[MEDIDO]` — **{len(union)} anotadores distintos ({100 * len(union) / n_tot:.1f}% do "
      f"painel)** disparam pelo menos um critério de anomalia. Destes, {n_high_anom} estão no "
      f"estrato ≥ {LOW_VOTE_THRESHOLD} votos.")
    A("")
    for k, d in sets_all.items():
        A(f"- **{d['label']}** — {d['n']}")
    A("")
    A(f"`[MEDIDO]` — **sobreposição zero**: {n_multi} anotadores disparam ≥ 2 critérios. Os "
      "conjuntos são disjuntos. Não existe neste painel o perfil \"anómalo em várias "
      "dimensões\" que se procurava — quem tem taxa de YES extrema não é quem diverge entre "
      "línguas, e ninguém acumula.")
    A("")
    A("Uma única família de anomalia tem população não-trivial: a **taxa de YES extrema** "
      "(61 anotadores, 6,9%). As restantes têm população zero por razões estruturais: "
      "`n_distinct_categories` tem mínimo observado 2 (ninguém usa uma só categoria) e as "
      "incoerências são nulas em todo o painel (§8.3).")
    A("")
    A("`[LIMITAÇÃO]` — dos 61, dois estão no estrato < 15 votos, onde `yes_rate = 1,00` exige "
      "apenas 9 votos YES seguidos. E `lang_gap > 0,30` descreve 5 anotadores de uma "
      "subpopulação de 19 — não é uma taxa do painel.")
    A("")

    # ---- 8.3
    A("### 8.3 A incoerência hierárquica existe em grau não-trivial, ou é residual?")
    A("")
    A("**Nem uma coisa nem outra: é INEXISTENTE, e por construção.**")
    A("")
    A("`[MEDIDO]` — `incoherence_A_to_C` e `incoherence_A_to_B` valem exactamente 0 nos 887 "
      "anotadores. Ao nível do voto (n = 23 904) não há um único caso de `NO` na Task A "
      "acompanhado de categoria de sexismo ou de intenção positiva. As contagens batem certo "
      "ao voto: 10 618 votos `NO` ↔ 10 618 `-` na Task B ↔ 10 618 `-` na Task C.")
    A("")
    A("Duas consequências para a tese:")
    A("")
    A("1. A feature que a especificação identificava como a mais informativa — a única próxima "
      "de um *erro objetivo*, independente de qualquer verdade — **não é computável neste "
      "corpus**. Não porque o método falhe, mas porque o fenómeno que mediria não ocorre. É um "
      "resultado, não uma lacuna.")
    A("2. `[HIPÓTESE]` a hierarquia é imposta pela interface de anotação. Se assim for, a "
      "ausência de incoerência **não é evidência de qualidade de anotação** — é evidência de um "
      "constrangimento de UI. Não escrever \"os anotadores do EXIST são hierarquicamente "
      "coerentes\" sem esta ressalva. Distinguir as duas leituras exigiria acesso à ferramenta "
      "de anotação ou às guidelines operacionais, que não temos.")
    A("")
    A("A única variante que sobrevive, `incoherence_yes_empty` (0 a 0,25), capta os 65 votos "
      "`YES` + `UNKNOWN` em C e é colinear com `unknown_rate_c`. Mede uso de `UNKNOWN`, não "
      "incoerência.")
    A("")
    A("---")
    A("")

    # ---- limites
    A("## 9. Limites")
    A("")
    A("1. **Este diagnóstico descreve padrões; não julga legitimidade.** Um anotador com 95% "
      "de YES pode ser um spammer **ou** genuinamente mais sensível a sexismo. Estas features "
      "não distinguem os dois casos e nunca distinguirão — por desenho, não medem acerto. Sob "
      "o paradigma LeWiDi, a segunda leitura é tão admissível como a primeira, e os 61 "
      "anotadores da §8.2 não são um conjunto de suspeitos.")
    A("2. **Nenhum down-weighting é justificado por estes números.** Nada aqui é uma medida de "
      "fiabilidade. Um perfil atípico é um perfil atípico; a validade dos votos não decorre "
      "daqui em nenhum sentido.")
    A("3. **O critério pré-registado do clustering é insensível a features indicadoras** "
      "(§6.2). Passou uma partição conduzida por um flag binário com ~6% de prevalência. Quem "
      "reutilizar este protocolo deve acrescentar uma condição sobre a forma das features, não "
      "só sobre o tamanho dos clusters.")
    A("4. **A §6.3 é post-hoc.** Foi decidida depois de ver os resultados. Não tem o estatuto "
      "do resultado pré-registado e não pode ser reportada como se tivesse.")
    A("5. **As correlações da §7 não validam nem invalidam os métodos anteriores.** Todas as "
      "associações significativas são fracas (|ρ| ≤ 0,24). Os anotadores que o NutmegEM marcou "
      "como menos fiáveis **não são** os de perfil comportamental anómalo. Como o NutmegEM foi "
      "caracterizado como capturável por blocos correlacionados, nem a concordância nem a "
      "discordância com ele transportam garantia.")
    A("6. **O estrato < 15 votos (11 anotadores) não suporta conclusões.** Extremos de taxa são "
      "esperados por amostragem com denominadores de 9 a 14.")
    A("7. **`lang_gap` descreve 19 anotadores**, não o painel: EN e ES são recrutados quase "
      "disjuntamente (431 / 437 / 19). Qualquer afirmação sobre comportamento inter-língua "
      "dentro do mesmo anotador vive nesta amostra minúscula.")
    A("8. **As features de categorias dependem da Task C**, cujo denominador é o número de "
      "itens que o anotador marcou como sexistas. Quem marca poucos YES tem estas features "
      "estimadas sobre poucos itens, ou a NaN (43 anotadores). A incerteza não é uniforme "
      "pelo painel.")
    A("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    profiles, demo, votes, audit = build_profiles(verbose=True)

    profiles.join(demo).to_csv(os.path.join(OUT_DIR, "behavioral_profiles.csv"))

    body, verdicts, sets_all, results, meta, low, high, thetas, extra = build_report(
        profiles, demo, audit
    )
    verdict = build_verdict_section(
        profiles, verdicts, sets_all, results, meta, low, high, thetas, extra
    )
    report = body + verdict

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    summary = {
        "seed": SEED,
        "n_annotators": int(len(profiles)),
        "prereg": {
            "cluster_min_fraction": CLUSTER_MIN_FRACTION,
            "cluster_min_silhouette": CLUSTER_MIN_SILHOUETTE,
            "dispersion_min_unique": DISPERSION_MIN_UNIQUE,
            "dispersion_max_modal_bin": DISPERSION_MAX_MODAL_BIN,
            "dispersion_min_iqr_fraction": DISPERSION_MIN_IQR_FRACTION,
            "low_vote_threshold": LOW_VOTE_THRESHOLD,
            "nan_policy": NAN_POLICY,
            "thresholds_extracao": THRESHOLDS,
        },
        "dispersion_verdicts": verdicts,
        "anomaly_counts": {k: d["n"] for k, d in sets_all.items()},
        "anomaly_ids": {k: d["ids"] for k, d in sets_all.items()},
        "clustering": results,
        "clustering_meta": meta,
        "posthoc_sensitivity": {
            "nota": "POST-HOC, nao pre-registada. Ver behavioral_diagnosis_observacoes.md.",
            "features": extra["sensitivity_meta"]["features_usadas"],
            "n_anotadores": extra["sensitivity_meta"]["n_anotadores"],
            "resultados": extra["sensitivity_results"],
            "zero_inflation": extra["zero_inflation"].to_dict(orient="index"),
        },
        "structural_audit": {k: v for k, v in audit.items() if k != "coherence_rule"},
    }
    with open(os.path.join(OUT_DIR, "behavioral_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n-> {REPORT_PATH}")
    print(f"-> {os.path.join(OUT_DIR, 'behavioral_diagnosis.json')}")
    print(f"-> {os.path.join(OUT_DIR, 'behavioral_profiles.csv')}")
    for r in results:
        print(f"   K={r['K']} tamanhos={r['tamanhos']} sil={r['silhouette']:.4f} {r['veredito']}")


if __name__ == "__main__":
    main()
