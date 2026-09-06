"""Carregamento dos dados reais do EXIST 2026 (Task 2 — memes) para o módulo de
confiabilidade. Task A (`labels_task2_1`, binária) apenas.

NOTA DE PROVENIÊNCIA — porque não importamos `EXIST_MEMES/memes_core.py`:
esse módulo faz `import torch`, `transformers` e `pyevall` no topo, e o `pyevall`
não está instalado no ambiente LOCAL onde este módulo corre (só na VM de treino).
Como esta etapa é offline e autónoma por especificação, as duas peças puramente
numéricas de que precisamos foram copiadas VERBATIM com a origem anotada:
  - `_normalize_apostrophes`  <- memes_core.py:326
  - `DEMO_*_VALS`             <- memes_core.py:275-310
Se alguma delas mudar em `memes_core.py`, tem de ser reflectida aqui.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

DEFAULT_JSON = "EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json"

TASK_A_FIELD = "labels_task2_1"
TASK_A_LABEL_MAP = {"NO": 0, "YES": 1}
# Ganchos (NÃO implementar nesta fase): Task B = `labels_task2_2` (NONE/JUDGEMENTAL/
# DIRECT, single-label -> `NutmegEM(n_labels=3)` já serviria); Task C =
# `labels_task2_3` (multilabel -> exige um modelo generativo diferente).

# --- VERBATIM de EXIST_MEMES/memes_core.py:275-310 --------------------------
DEMO_ETHNICITY_VALS = sorted(['Asian', 'Black or African American', 'Hispano or Latino',
                              'Middle Eastern', 'Multiracial', 'White or Caucasian', 'other'])
DEMO_STUDY_VALS = sorted(["Bachelor's degree", 'Doctorate',
                          'High school degree or equivalent',
                          'Less than high school diploma', "Master's degree", 'other'])
DEMO_COUNTRY_VALS = sorted(['Afghanistan', 'Argentina', 'Australia', 'Austria', 'Azerbaijan',
                            'Bangladesh', 'Belgium', 'Brazil', 'Bulgaria', 'Canada', 'Chile',
                            'China', 'Colombia', 'Croatia', 'Cuba', 'Czech Republic',
                            'Dominican Republic', 'Ecuador', 'Estonia', 'Finland', 'France',
                            'Germany', 'Greece', 'Hungary', 'India', 'Ireland', 'Israel',
                            'Italy', 'Latvia', 'Mexico', 'Morocco', 'Netherlands',
                            'New Zealand', 'Nicaragua', 'Nigeria', 'Peru', 'Poland',
                            'Portugal', 'Romania', 'Russian Federation', 'Serbia', 'Slovenia',
                            'South Africa', 'Spain', 'Sweden', 'Turkey', 'United Arab Emirates',
                            'United Kingdom', 'United States', 'Uruguay', 'Venezuela',
                            'Viet Nam', 'Zimbabwe'])
DEMO_GENDER_VALS = sorted(['F', 'M'])
DEMO_AGE_VALS = sorted(['18-22', '23-45', '46+'])

DEMO_GROUPS_BASE = [
    ("ethnicities_annotators", DEMO_ETHNICITY_VALS),
    ("study_levels_annotators", DEMO_STUDY_VALS),
    ("countries_annotators", DEMO_COUNTRY_VALS),
]
DEMO_GROUPS_EXTENDED = DEMO_GROUPS_BASE + [
    ("gender_annotators", DEMO_GENDER_VALS),
    ("age_annotators", DEMO_AGE_VALS),
]


def _normalize_apostrophes(text):
    """VERBATIM de memes_core.py:326. Apóstrofo curvo (U+2019) -> reto (U+0027).
    Sem isto, 64,6% dos votos de escolaridade não batiam com o vocabulário."""
    return text.replace("’", "'")


def load_task_a(json_path: str = DEFAULT_JSON, verbose: bool = True):
    """Lê o JSON de treino e devolve a matriz de anotações da Task A.

    Returns
    -------
    annotations : ndarray (n_votes, 3) -> (item_idx, annotator_idx, label 0/1)
    annotator_ids : list[str] (M,)   índice -> ID persistente ("Annotator_193")
    item_ids : list[str] (N,)        índice -> id_EXIST
    demo : dict[str, dict[str, str]] ID -> {campo: valor canónico (moda)}
    stats : dict com diagnósticos do carregamento
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ann_id_to_idx: dict[str, int] = {}
    annotator_ids: list[str] = []
    item_ids: list[str] = []
    rows = []
    demo_raw: dict[str, dict[str, Counter]] = {}
    n_dropped = 0
    demo_fields = [g[0] for g in DEMO_GROUPS_EXTENDED]

    for item_key, rec in raw.items():
        i = len(item_ids)
        item_ids.append(str(rec.get("id_EXIST", item_key)))
        annotators = rec.get("annotators") or []
        labels = rec.get(TASK_A_FIELD) or []
        for pos, (aid, lab) in enumerate(zip(annotators, labels)):
            lab = str(lab).strip().upper()
            if lab not in TASK_A_LABEL_MAP:  # guard: UNKNOWN/'-' (ausentes na Task A)
                n_dropped += 1
                continue
            if aid not in ann_id_to_idx:
                ann_id_to_idx[aid] = len(annotator_ids)
                annotator_ids.append(aid)
                demo_raw[aid] = {fld: Counter() for fld in demo_fields}
            rows.append((i, ann_id_to_idx[aid], TASK_A_LABEL_MAP[lab]))
            for fld in demo_fields:
                vals = rec.get(fld) or []
                if pos < len(vals) and vals[pos] is not None:
                    demo_raw[aid][fld][_normalize_apostrophes(str(vals[pos]).strip())] += 1

    annotations = np.asarray(rows, dtype=int)

    # Demografia canónica por ID = moda. 2 IDs conhecidos com valores inconsistentes
    # (Annotator_125: Romania vs Spain; Annotator_187: Bachelor's vs Master's).
    demo = {}
    inconsistent = []
    for aid, fields in demo_raw.items():
        canon = {}
        for fld, counter in fields.items():
            if not counter:
                canon[fld] = "__MISSING__"
                continue
            if len(counter) > 1:
                inconsistent.append((aid, fld, dict(counter)))
            canon[fld] = counter.most_common(1)[0][0]
        demo[aid] = canon

    counts = np.bincount(annotations[:, 1], minlength=len(annotator_ids))
    stats = {
        "n_items": len(item_ids),
        "n_annotators": len(annotator_ids),
        "n_votes": int(len(annotations)),
        "n_dropped_labels": n_dropped,
        "votes_per_annotator_mean": float(counts.mean()),
        "votes_per_annotator_min": int(counts.min()),
        "votes_per_annotator_max": int(counts.max()),
        "n_annotators_below_15_votes": int((counts < 15).sum()),
        "label_balance_yes": float(annotations[:, 2].mean()),
        "inconsistent_demo": inconsistent,
    }
    if verbose:
        print(
            f"[exist] itens={stats['n_items']} anotadores={stats['n_annotators']} "
            f"votos={stats['n_votes']} descartados={n_dropped} | "
            f"YES={100 * stats['label_balance_yes']:.1f}%"
        )
        print(
            f"[exist] votos/anotador: mean={stats['votes_per_annotator_mean']:.1f} "
            f"min={stats['votes_per_annotator_min']} max={stats['votes_per_annotator_max']} "
            f"| abaixo de 15 votos: {stats['n_annotators_below_15_votes']} "
            f"({100 * stats['n_annotators_below_15_votes'] / len(annotator_ids):.1f}%)"
        )
        for aid, fld, c in inconsistent:
            print(f"[exist] demografia inconsistente -> moda: {aid} {fld} {c}")
    return annotations, annotator_ids, item_ids, demo, stats


def demo_one_hot(annotator_ids, demo, groups):
    """Matriz one-hot (M, D) da demografia canónica por anotador.

    `groups` = DEMO_GROUPS_BASE (66-d) ou DEMO_GROUPS_EXTENDED (71-d).
    Valores fora do vocabulário ficam a zeros nesse bloco (mesmo comportamento
    silencioso do `_count_votes_per_category` original).
    """
    blocks = []
    for fld, vocab in groups:
        idx = {_normalize_apostrophes(v): i for i, v in enumerate(vocab)}
        block = np.zeros((len(annotator_ids), len(vocab)), dtype=np.float32)
        for r, aid in enumerate(annotator_ids):
            j = idx.get(demo[aid].get(fld, ""), None)
            if j is not None:
                block[r, j] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)
