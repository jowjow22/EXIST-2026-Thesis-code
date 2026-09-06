"""Perfis de comportamento AGREGADO por anotador (EXIST 2026 Task 2 — memes).

O que este módulo é
-------------------
Extração DESCRITIVA. Cada feature calcula-se a partir dos votos do próprio
anotador, sem sobreposição com outros anotadores e sem comparação com qualquer
consenso ou verdade inferida. Nenhuma feature aqui é uma medida de acerto.

Restrição de desenho (verificada feature a feature):
    nenhuma quantidade depende de outro anotador, de uma verdade estimada,
    ou de concordância par-a-par.

Convenção de anotação dos factos (mesma de reliability/NOTAS_PARA_TESE.md):
    [MEDIDO]    -> número observado, com o artefacto que o produziu
    [HIPOTESE]  -> interpretação plausível, ainda não testada
    [LIMITACAO] -> restrição a declarar explicitamente no texto

Proveniência
------------
O tratamento de apóstrofos e a resolução da demografia por moda vêm de
`reliability/exist_data.py` (importados, não copiados). O `load_task_a` desse
módulo descarta as Tasks B/C, por isso este faz uma segunda passagem alinhada
por (item, posição) sobre o JSON cru.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .exist_data import (  # reutilização explícita, sem cópia
    DEFAULT_JSON,
    DEMO_GROUPS_EXTENDED,
    _normalize_apostrophes,
)

# ---------------------------------------------------------------------------
# Vocabulários
# ---------------------------------------------------------------------------

FIELD_A = "labels_task2_1"
FIELD_B = "labels_task2_2"
FIELD_C = "labels_task2_3"

#: As 5 categorias de sexismo da subtask 2.3. `-` e `UNKNOWN` NÃO são categorias.
SEXISM_CATEGORIES = (
    "IDEOLOGICAL-INEQUALITY",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "STEREOTYPING-DOMINANCE",
)
SEXISM_SET = frozenset(SEXISM_CATEGORIES)

#: Rótulos de intenção "positivos" da subtask 2.2 (i.e. não `-`, não `UNKNOWN`).
INTENT_POSITIVE = frozenset({"DIRECT", "JUDGEMENTAL"})

NON_SEXIST_MARK = "-"
UNKNOWN_MARK = "UNKNOWN"

# ---------------------------------------------------------------------------
# Limiares de denominador mínimo (features abaixo do limiar devolvem NaN, não 0)
# ---------------------------------------------------------------------------

MIN_SEXIST_ITEMS = 5    # features de uso do espaço de categorias (5, 6, 7)
MIN_NO_VOTES = 5        # incoerências que dividem pelos votos NO (8, 9)
MIN_YES_VOTES = 5       # incoerência que divide pelos votos YES (10)
MIN_VOTES_PER_LANG = 5  # yes_rate_en / yes_rate_es / lang_gap (11, 12)

THRESHOLDS = {
    "MIN_SEXIST_ITEMS": MIN_SEXIST_ITEMS,
    "MIN_NO_VOTES": MIN_NO_VOTES,
    "MIN_YES_VOTES": MIN_YES_VOTES,
    "MIN_VOTES_PER_LANG": MIN_VOTES_PER_LANG,
}

FEATURE_COLUMNS = [
    "yes_rate",
    "n_votes",
    "unknown_rate_b",
    "unknown_rate_c",
    "n_distinct_categories",
    "category_entropy",
    "mean_labels_per_item",
    "incoherence_A_to_C",
    "incoherence_A_to_B",
    "incoherence_yes_empty",
    "yes_rate_en",
    "yes_rate_es",
    "lang_gap",
]

# ---------------------------------------------------------------------------
# REGRA DE COERÊNCIA HIERÁRQUICA (fixada aqui, citada no relatório)
# ---------------------------------------------------------------------------
COHERENCE_RULE = """\
Segundo as guidelines do EXIST, `-` é o rótulo atribuído a instâncias NÃO-SEXISTAS
nas subtasks 2.2 e 2.3. Logo:

  NO na Task A  +  `-` na B/C                     -> COERENTE (nao conta)
  NO na Task A  +  >=1 categoria de SEXISM_SET    -> incoherence_A_to_C
  NO na Task A  +  DIRECT ou JUDGEMENTAL na B     -> incoherence_A_to_B
  YES na Task A +  nem `-` nem categoria real     -> incoherence_yes_empty
  UNKNOWN                                          -> NUNCA conta como incoerencia
                                                      A->C ou A->B; conta como
                                                      "vazio" no caso YES (e o
                                                      unico valor que satisfaz
                                                      "nem `-` nem categoria").

`incoherence_yes_empty` e, por esta definicao, colinear com `unknown_rate_c`
restrita aos votos YES. [LIMITACAO] -- as duas nao sao evidencia independente.
"""

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------


def _as_label_set(raw_value) -> frozenset:
    """Normaliza o campo da Task C, que é uma lista de rótulos (ou um escalar)."""
    if isinstance(raw_value, (list, tuple, set)):
        vals = [str(v).strip() for v in raw_value]
    elif raw_value is None:
        vals = []
    else:
        vals = [str(raw_value).strip()]
    return frozenset(v for v in vals if v != "")


def load_votes(json_path: str = DEFAULT_JSON):
    """Uma linha por voto (anotador × item), com as três subtasks alinhadas.

    O alinhamento é POSICIONAL: `annotators[k]` corresponde a `labels_task2_*[k]`.
    Itens cujas listas não tenham o mesmo comprimento são contados em
    `meta['n_length_mismatch']` e truncados pelo mínimo (guarda defensiva).

    Returns
    -------
    votes    : DataFrame (n_votos, 8)
    demo_raw : dict[ID, dict[campo, Counter]] — contagens brutas por anotador
    meta     : dict com diagnósticos do carregamento

    NOTA: `demo_raw` é devolvido à parte e NÃO vai para `votes.attrs`. O pandas
    propaga `attrs` copiando-os em cada objecto derivado; com 887 anotadores ×
    5 Counters lá dentro, um `groupby` por anotador custava ~20 s só em cópias.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    demo_raw: dict[str, dict[str, Counter]] = {}
    demo_fields = [g[0] for g in DEMO_GROUPS_EXTENDED]
    n_mismatch = 0

    for item_key, rec in raw.items():
        item_id = str(rec.get("id_EXIST", item_key))
        lang = str(rec.get("lang", "")).strip().lower()
        annotators = rec.get("annotators") or []
        labs_a = rec.get(FIELD_A) or []
        labs_b = rec.get(FIELD_B) or []
        labs_c = rec.get(FIELD_C) or []
        n = min(len(annotators), len(labs_a), len(labs_b), len(labs_c))
        if not (len(annotators) == len(labs_a) == len(labs_b) == len(labs_c)):
            n_mismatch += 1

        for pos in range(n):
            aid = annotators[pos]
            cset = _as_label_set(labs_c[pos])
            rows.append(
                {
                    "item_id": item_id,
                    "annotator_id": aid,
                    "lang": lang,
                    "label_a": str(labs_a[pos]).strip().upper(),
                    "label_b": str(labs_b[pos]).strip().upper(),
                    "cats": cset & SEXISM_SET,
                    "c_has_dash": NON_SEXIST_MARK in cset,
                    "c_has_unknown": UNKNOWN_MARK in cset,
                }
            )
            if aid not in demo_raw:
                demo_raw[aid] = {fld: Counter() for fld in demo_fields}
            for fld in demo_fields:
                vals = rec.get(fld) or []
                if pos < len(vals) and vals[pos] is not None:
                    demo_raw[aid][fld][_normalize_apostrophes(str(vals[pos]).strip())] += 1

    df = pd.DataFrame(rows)
    return df, demo_raw, {"n_length_mismatch": n_mismatch}


def resolve_demographics(demo_raw) -> tuple[pd.DataFrame, list]:
    """Demografia canónica por ID = moda (mesma regra de exist_data.load_task_a).

    2 IDs conhecidos têm valores inconsistentes ao longo dos itens
    (Annotator_125: Romania vs Spain; Annotator_187: Bachelor's vs Master's).
    """
    canon, inconsistent = {}, []
    for aid, fields in demo_raw.items():
        row = {}
        for fld, counter in fields.items():
            if not counter:
                row[fld] = "__MISSING__"
                continue
            if len(counter) > 1:
                inconsistent.append((aid, fld, dict(counter)))
            row[fld] = counter.most_common(1)[0][0]
        canon[aid] = row
    return pd.DataFrame.from_dict(canon, orient="index"), inconsistent


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _normalized_entropy(counts) -> float:
    """Entropia de Shannon normalizada por log(len(SEXISM_CATEGORIES)).

    0.0 = usa sempre a mesma categoria; 1.0 = distribuição uniforme pelas 5.
    """
    total = float(np.sum(counts))
    if total <= 0:
        return np.nan
    p = np.asarray(counts, dtype=float) / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    return h / float(np.log(len(SEXISM_CATEGORIES)))


def _profile_one(g: pd.DataFrame) -> dict:
    n_votes = len(g)
    is_yes = (g["label_a"] == "YES").to_numpy()
    is_no = (g["label_a"] == "NO").to_numpy()
    n_yes, n_no = int(is_yes.sum()), int(is_no.sum())

    cats = g["cats"].to_numpy()
    n_cats = np.array([len(c) for c in cats], dtype=int)

    out = {
        "n_votes": n_votes,
        "n_yes": n_yes,
        "n_no": n_no,
        # 1 — tendência de resposta
        "yes_rate": n_yes / n_votes if n_votes else np.nan,
        # 3, 4 — uso de UNKNOWN
        "unknown_rate_b": float((g["label_b"] == UNKNOWN_MARK).mean()),
        "unknown_rate_c": float(g["c_has_unknown"].mean()),
    }

    # 5, 6, 7 — uso do espaço de categorias.
    # Denominador = itens em que o anotador atribuiu >=1 categoria REAL.
    sexist_mask = n_cats > 0
    n_sexist_items = int(sexist_mask.sum())
    out["n_sexist_items"] = n_sexist_items
    if n_sexist_items >= MIN_SEXIST_ITEMS:
        counter = Counter()
        for c in cats[sexist_mask]:
            counter.update(c)
        out["n_distinct_categories"] = len(counter)
        out["category_entropy"] = _normalized_entropy(
            [counter.get(k, 0) for k in SEXISM_CATEGORIES]
        )
        out["mean_labels_per_item"] = float(n_cats[sexist_mask].mean())
    else:
        # NaN explícito, não 0 — assim `n_distinct_categories == 1` (anomalia)
        # nunca se confunde com "não tem votos sexistas suficientes".
        out["n_distinct_categories"] = np.nan
        out["category_entropy"] = np.nan
        out["mean_labels_per_item"] = np.nan

    # 8, 9, 10 — coerência hierárquica (ver COHERENCE_RULE)
    if n_no >= MIN_NO_VOTES:
        out["incoherence_A_to_C"] = float((n_cats[is_no] > 0).mean())
        out["incoherence_A_to_B"] = float(
            g.loc[is_no, "label_b"].isin(INTENT_POSITIVE).mean()
        )
    else:
        out["incoherence_A_to_C"] = np.nan
        out["incoherence_A_to_B"] = np.nan

    if n_yes >= MIN_YES_VOTES:
        empty = (n_cats[is_yes] == 0) & (~g.loc[is_yes, "c_has_dash"].to_numpy())
        out["incoherence_yes_empty"] = float(empty.mean())
    else:
        out["incoherence_yes_empty"] = np.nan

    # 11, 12 — comportamento por língua
    for lang in ("en", "es"):
        sub = g[g["lang"] == lang]
        out[f"n_votes_{lang}"] = len(sub)
        out[f"yes_rate_{lang}"] = (
            float((sub["label_a"] == "YES").mean())
            if len(sub) >= MIN_VOTES_PER_LANG
            else np.nan
        )
    out["lang_gap"] = abs(out["yes_rate_en"] - out["yes_rate_es"])

    return out


def structural_audit(votes: pd.DataFrame) -> dict:
    """Factos estruturais do dataset, ao nível do voto [MEDIDO].

    Existe para que o relatório possa afirmar que as incoerências são zero
    (ou não) por evidência directa, e não por inferência a partir dos perfis.
    """
    ab = Counter(zip(votes["label_a"], votes["label_b"]))

    def _kind(row):
        if row["cats"]:
            return "CATEGORIA"
        if row["c_has_dash"]:
            return "DASH"
        if row["c_has_unknown"]:
            return "UNKNOWN"
        return "VAZIO"

    ac = Counter(zip(votes["label_a"], votes.apply(_kind, axis=1)))
    lang_per_ann = votes.groupby("annotator_id")["lang"].agg(
        lambda s: "both" if s.nunique() > 1 else s.iloc[0]
    )
    return {
        "n_votes": int(len(votes)),
        "n_items": int(votes["item_id"].nunique()),
        "n_annotators": int(votes["annotator_id"].nunique()),
        "crosstab_a_b": {f"{a}|{b}": int(n) for (a, b), n in sorted(ab.items())},
        "crosstab_a_c": {f"{a}|{c}": int(n) for (a, c), n in sorted(ac.items())},
        "annotators_by_lang": {k: int(v) for k, v in lang_per_ann.value_counts().items()},
        "coherence_rule": COHERENCE_RULE,
    }


def build_profiles(json_path: str = DEFAULT_JSON, verbose: bool = True):
    """Devolve (profiles, demo, votes, audit).

    profiles : DataFrame (887, ...) indexado pelo ID persistente do anotador
    demo     : DataFrame (887, 5) demografia canónica (moda)
    votes    : DataFrame (n_votos, ...) linha por voto
    audit    : dict com os factos estruturais do dataset [MEDIDO]
    """
    votes, demo_raw, meta = load_votes(json_path)
    demo, inconsistent = resolve_demographics(demo_raw)

    rows = {aid: _profile_one(g) for aid, g in votes.groupby("annotator_id", sort=True)}
    profiles = pd.DataFrame.from_dict(rows, orient="index")
    profiles.index.name = "annotator_id"
    demo = demo.reindex(profiles.index)

    audit = structural_audit(votes)
    audit["n_length_mismatch"] = meta["n_length_mismatch"]
    audit["inconsistent_demo"] = inconsistent
    audit["thresholds"] = dict(THRESHOLDS)

    if verbose:
        print(
            f"[behav] anotadores={len(profiles)} votos={len(votes)} "
            f"itens={votes['item_id'].nunique()} "
            f"| listas desalinhadas={audit['n_length_mismatch']}"
        )
        for aid, fld, c in inconsistent:
            print(f"[behav] demografia inconsistente -> moda: {aid} {fld} {c}")
        print(f"[behav] cross-tab A x B: {audit['crosstab_a_b']}")
        print(f"[behav] cross-tab A x C: {audit['crosstab_a_c']}")
        print(f"[behav] anotadores por lingua: {audit['annotators_by_lang']}")

    return profiles, demo, votes, audit


if __name__ == "__main__":
    prof, demo, votes, audit = build_profiles()
    cols = ["n_votes"] + [c for c in FEATURE_COLUMNS if c != "n_votes"]
    pd.set_option("display.width", 250)
    print("\n--- 5 anotadores de exemplo ---")
    print(prof[cols].head(5).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n--- regra de coerencia ---")
    print(COHERENCE_RULE)
