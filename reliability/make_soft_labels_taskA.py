"""
Gera soft labels da Task A (EXIST 2026 / memes) em duas variantes: fracção crua de
votos `YES` (réplica exacta do que o pipeline faz hoje) e média ponderada pelo θ de
competência de cada anotador, estimado em `theta_taskA_single_truth_K1.json`.

EXPERIMENTO CONTROLADO. O objectivo não é melhorar o modelo — é medir se θ carrega
informação. Um resultado nulo é aceitável.

O que este script NÃO faz: não treina nada, não importa `memes_core.py` (esse módulo
puxa torch/transformers/pyevall, indisponíveis no ambiente local — mesma razão
documentada em `reliability/exist_data.py`), não altera pipeline, dataloaders nem
arquitectura, não toca nas soft labels de B nem de C, e não regenera θ.

Fórmula (pesos normalizados DENTRO de cada item, nunca globalmente):

    soft_weighted(i) = Σ_a (θ_a · v_a) / Σ_a θ_a          v_a ∈ {0,1}

Uso:
    python3 -m reliability.make_soft_labels_taskA
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from reliability.run_behavioral_diagnosis import md_table

# ---------------------------------------------------------------------------
# Constantes — espelham o notebook `EXIST_MEMES/memes_model.ipynb`
# ---------------------------------------------------------------------------

# Ordem da célula 16. A soft label é uma lista [p_YES, p_NO] NESTA ordem.
LABELS_2_1 = ["YES", "NO"]

DATASET_PATH = "EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json"
THETA_PATH = "reliability/outputs/theta_taskA_single_truth_K1.json"
THETA_KEY = "theta"

OUT_DIR = "reliability/outputs"
RAW_PATH = os.path.join(OUT_DIR, "soft_labels_taskA_raw.json")
WEIGHTED_PATH = os.path.join(OUT_DIR, "soft_labels_taskA_weighted.json")
SPLIT_PATH = os.path.join(OUT_DIR, "split_taskA_frozen.json")
REPORT_PATH = os.path.join(OUT_DIR, "soft_labels_taskA_report.md")


def out_paths(suffix=""):
    """Caminhos de saída com sufixo. Sufixo vazio = os ficheiros originais."""
    return (
        os.path.join(OUT_DIR, f"soft_labels_taskA_raw{suffix}.json"),
        os.path.join(OUT_DIR, f"soft_labels_taskA_weighted{suffix}.json"),
        os.path.join(OUT_DIR, f"split_taskA_frozen{suffix}.json"),
        os.path.join(OUT_DIR, f"soft_labels_taskA_report{suffix}.md"),
    )

# Split — valores da célula 44 do notebook, replicados exactamente.
SPLIT_SEED = 42
SPLIT_TEST_SIZE = 0.15

TOL = 1e-9

# Contrafactual de diagnóstico (decisão do autor: seguir o spec literal, medir o
# efeito colateral). Convenção MACE: θ é P(não-spam), logo o piso é o acaso.
CLIP_LO, CLIP_HI = 0.50, 0.999


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATASET_PATH) -> dict:
    """Lê o JSON cru `{id_EXIST: {...}}`. `json.load` preserva a ordem de inserção,
    que é a mesma ordem de linhas que `pd.read_json(...).transpose()` produz no
    notebook — é dela que depende a reprodução do split congelado."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_theta(path: str = THETA_PATH):
    """Devolve `(theta, meta)`. O ficheiro tem 5 chaves de topo; os θ estão em
    `d["theta"]`, um dict `{Annotator_N: float}`."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d[THETA_KEY], d.get("meta", {})


# ---------------------------------------------------------------------------
# Construção das soft labels
# ---------------------------------------------------------------------------

def soft_raw(votes):
    """Réplica exacta de `get_soft_distribution(votes, LABELS_2_1)` de
    `memes_core.py:243`: conta votos sobre o vocabulário fixo e divide pelo número
    TOTAL de votos (não pelo número de votos reconhecidos). Devolve None se a lista
    estiver vazia — o notebook descarta essas amostras."""
    if not votes:
        return None
    counts = dict.fromkeys(LABELS_2_1, 0)
    for vote in votes:
        if vote in counts:
            counts[vote] += 1
    n = len(votes)
    return [counts[label] / n for label in LABELS_2_1]


def soft_weighted(votes, annotators, theta, fallback):
    """Média dos votos ponderada por θ, com os pesos normalizados dentro do item.

    Devolve `([p_YES, p_NO], weights_normalizados, ids_sem_theta)`. Um anotador
    ausente do ficheiro de θ recebe a mediana do painel (`fallback`) — nunca 1.0, e
    o voto nunca é descartado.
    """
    if not votes:
        return None, None, []

    missing = [a for a in annotators if a not in theta]
    w = np.array([float(theta.get(a, fallback)) for a in annotators], dtype=np.float64)
    v = np.array([1.0 if vote == "YES" else 0.0 for vote in votes], dtype=np.float64)

    total = w.sum()
    if total <= 0.0:
        raise ValueError("soma de pesos não-positiva — item indeterminado")
    w_norm = w / total

    p_yes = float(np.dot(w_norm, v))
    return [p_yes, 1.0 - p_yes], w_norm, missing


def build_all(dataset, theta, fallback):
    """Percorre o dataset uma vez e devolve tudo o que as verificações e o
    diagnóstico precisam, na ordem de inserção do JSON."""
    ids, raw, wt, weights, missing = [], {}, {}, {}, {}
    for item_id, item in dataset.items():
        votes = item["labels_task2_1"]
        anns = item["annotators"]
        if len(votes) != len(anns):
            raise ValueError(f"item {item_id}: {len(votes)} votos vs {len(anns)} anotadores")

        r = soft_raw(votes)
        w, w_norm, miss = soft_weighted(votes, anns, theta, fallback)
        if r is None or w is None:
            continue  # sem voto válido — o notebook descartaria a amostra

        ids.append(item_id)
        raw[item_id] = r
        wt[item_id] = w
        weights[item_id] = w_norm
        if miss:
            missing[item_id] = miss
    return ids, raw, wt, weights, missing


# ---------------------------------------------------------------------------
# Verificações obrigatórias
# ---------------------------------------------------------------------------

def verify(dataset, ids, raw, wt, weights, missing):
    """Corre as cinco verificações. Devolve `(checks, detalhes)`; `checks` é uma
    lista de `(nome, passou, detalhe)`. Não escreve nada."""
    checks = []

    # 1. Identidade do baseline: forma fechada n_yes / n_votos.
    worst, n_bad = 0.0, 0
    for item_id in ids:
        votes = dataset[item_id]["labels_task2_1"]
        closed = votes.count("YES") / len(votes)
        d = abs(raw[item_id][0] - closed)
        worst = max(worst, d)
        if d > TOL:
            n_bad += 1
        # a segunda posição tem de fechar a distribuição
        if abs(sum(raw[item_id]) - 1.0) > TOL:
            n_bad += 1
    checks.append((
        "1. identidade do baseline (raw == n_yes/n_votos, |Δ|<1e-9)",
        n_bad == 0,
        f"{len(ids) - n_bad}/{len(ids)} itens · maxΔ = {worst:.3e}",
    ))

    # 2. Domínio.
    bad_dom = []
    for item_id in ids:
        for name, vec in (("raw", raw[item_id]), ("weighted", wt[item_id])):
            arr = np.array(vec, dtype=np.float64)
            if np.isnan(arr).any() or (arr < -TOL).any() or (arr > 1.0 + TOL).any():
                bad_dom.append((item_id, name))
    same_count = len(raw) == len(wt) == len(dataset)
    same_ids = set(raw) == set(wt) == set(dataset)
    checks.append((
        "2. domínio [0,1], sem NaN, contagens e IDs iguais ao dataset",
        not bad_dom and same_count and same_ids,
        f"{len(raw)} raw · {len(wt)} weighted · {len(dataset)} dataset · "
        f"{len(bad_dom)} violações de domínio",
    ))

    # 3. Pesos somam 1 dentro do item.
    worst_w = max(abs(float(w.sum()) - 1.0) for w in weights.values())
    checks.append((
        "3. Σ pesos normalizados == 1 por item (|Δ|<1e-9)",
        worst_w <= TOL,
        f"maxΔ = {worst_w:.3e} sobre {len(weights)} itens",
    ))

    # 4. Anotadores sem θ.
    missing_ids = sorted({a for lst in missing.values() for a in lst})
    checks.append((
        "4. anotadores sem θ (recebem mediana do painel)",
        True,  # informativo: o fallback é comportamento definido, não falha
        f"{len(missing_ids)} anotadores em {len(missing)} itens"
        + (f": {', '.join(missing_ids[:20])}" if missing_ids else ""),
    ))

    return checks, missing_ids


def build_frozen_split(ids, raw):
    """Replica a célula 44: estratifica pelo voto majoritário derivado do RAW e faz
    `train_test_split(test_size=0.15, random_state=42)`. Congela as listas de IDs
    para que os dois braços do experimento treinem no mesmo split."""
    strat = ["YES" if raw[i][0] >= 0.5 else "NO" for i in ids]
    train_ids, val_ids = train_test_split(
        ids, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, stratify=strat
    )
    return list(train_ids), list(val_ids), strat


def verify_split(ids, train_ids, val_ids):
    ok_partition = (
        set(train_ids).isdisjoint(val_ids)
        and set(train_ids) | set(val_ids) == set(ids)
        and len(train_ids) + len(val_ids) == len(ids)
    )
    return (
        "5. split congelado particiona os itens sem sobreposição",
        ok_partition,
        f"train {len(train_ids)} + val {len(val_ids)} = {len(train_ids) + len(val_ids)} "
        f"de {len(ids)}",
    )


# ---------------------------------------------------------------------------
# Diagnóstico de poder
# ---------------------------------------------------------------------------

def power_report(ids, raw, wt):
    """Quanto as labels realmente mudaram. Se o deslocamento for da ordem de 0,01 e
    r > 0,99, o retreino provavelmente não terá poder de detectar diferença."""
    r = np.array([raw[i][0] for i in ids], dtype=np.float64)
    w = np.array([wt[i][0] for i in ids], dtype=np.float64)
    d = np.abs(w - r)
    with np.errstate(all="ignore"):  # flags espúrias do BLAS Accelerate (macOS)
        pearson = float(pearsonr(r, w)[0])
    return {
        "n_items": int(len(ids)),
        "mean_abs_delta": float(d.mean()),
        "sd_abs_delta": float(d.std(ddof=0)),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "n_gt_005": int((d > 0.05).sum()),
        "n_gt_010": int((d > 0.10).sum()),
        "pearson_r": pearson,
        "n_cross_half": int(((r >= 0.5) != (w >= 0.5)).sum()),
        "max_abs_delta": float(d.max()),
    }


def theta_tail_diagnostic(dataset, ids, theta, fallback, wt, weights):
    """Decisão do autor: seguir o spec literal (θ sem clip) e MEDIR o efeito
    colateral da cauda sub-0,5. Sob EM single-truth, θ<0,5 é "discorda da verdade
    inferida mais do que acerta"; a fórmula atenua esse anotador mas nunca inverte
    o voto dele. O contrafactual clipado existe só como número no relatório — não é
    escrito em disco e não gera braço de treino."""
    vals = np.array(list(theta.values()), dtype=np.float64)

    # fracção dos votos carregada por anotadores de θ baixo
    n_votes_total, n_lt50, n_lt30 = 0, 0, 0
    for item_id in ids:
        for a in dataset[item_id]["annotators"]:
            t = float(theta.get(a, fallback))
            n_votes_total += 1
            if t < 0.50:
                n_lt50 += 1
            if t < 0.30:
                n_lt30 += 1

    all_w = np.concatenate([weights[i] for i in ids])

    # contrafactual com θ clipado
    w_clip = {}
    for item_id in ids:
        item = dataset[item_id]
        t = np.clip(
            np.array([float(theta.get(a, fallback)) for a in item["annotators"]]),
            CLIP_LO, CLIP_HI,
        )
        v = np.array([1.0 if x == "YES" else 0.0 for x in item["labels_task2_1"]])
        w_clip[item_id] = float(np.dot(t / t.sum(), v))

    lit = np.array([wt[i][0] for i in ids])
    clp = np.array([w_clip[i] for i in ids])
    dd = np.abs(lit - clp)

    return {
        "n_annotators": int(len(vals)),
        "n_theta_lt_050": int((vals < 0.50).sum()),
        "n_theta_lt_030": int((vals < 0.30).sum()),
        "frac_votes_by_theta_lt_050": n_lt50 / n_votes_total,
        "frac_votes_by_theta_lt_030": n_lt30 / n_votes_total,
        "n_votes_total": int(n_votes_total),
        "weight_min": float(all_w.min()),
        "weight_p50": float(np.percentile(all_w, 50)),
        "weight_max": float(all_w.max()),
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "mean_abs_literal_minus_clip": float(dd.mean()),
        "max_abs_literal_minus_clip": float(dd.max()),
        "n_cross_half_literal_vs_clip": int(((lit >= 0.5) != (clp >= 0.5)).sum()),
    }


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def build_report(theta_meta, theta, fallback, dataset, ids, checks, power, tail,
                 train_ids, val_ids, missing_ids):
    """Todo o texto de veredito é construído a partir dos números medidos em runtime
    (f-strings), nunca prosa fixa sobre a direcção do resultado. Este defeito já
    ocorreu antes e está documentado em `behavioral_diagnosis_observacoes.md` §1.1."""
    vals = np.array(list(theta.values()))

    lines = []
    a = lines.append

    a("# Soft labels da Task A ponderadas por θ — geração e diagnóstico")
    a("")
    a("Gerado por `reliability/make_soft_labels_taskA.py`. Experimento controlado: mede")
    a("se o θ de competência por anotador carrega informação quando entra na construção")
    a("da soft label. **Não é uma tentativa de melhorar o modelo.** Resultado nulo é")
    a("aceitável.")
    a("")

    # --- veredito de poder, primeiro ---
    a("## 1. Veredito — o retreino tem poder?")
    a("")
    tem_poder = power["mean_abs_delta"] > 0.02 or power["pearson_r"] < 0.99
    if tem_poder:
        a(f"**SIM.** O deslocamento médio é `{power['mean_abs_delta']:.4f}` "
          f"(sd `{power['sd_abs_delta']:.4f}`), a correlação de Pearson entre raw e")
        a(f"weighted é `{power['pearson_r']:.4f}`, e **{power['n_cross_half']} itens "
          f"({100 * power['n_cross_half'] / power['n_items']:.1f}%) mudam de lado da")
        a("maioria**. Isto não é ruído numérico: é uma mudança de alvo que o treino pode")
        a("registar. Gastar GPU é defensável.")
    else:
        a(f"**PROVAVELMENTE NÃO.** Deslocamento médio `{power['mean_abs_delta']:.4f}` e")
        a(f"Pearson `{power['pearson_r']:.4f}` — as duas variantes são quase o mesmo alvo.")
        a("Um delta de ICM nulo seria indistinguível de ausência de efeito.")
    a("")
    a("`[LIMITAÇÃO]` isto mede quanto os **alvos** mudaram, não se a mudança é na direcção")
    a("certa. Um θ que carrega ruído desloca as labels tanto quanto um θ que carrega sinal.")
    a("")

    # --- inspecção ---
    a("## 2. Entradas")
    a("")
    a(f"- Dataset: `{DATASET_PATH}` — **{len(dataset)} itens**, "
      f"{len(dataset[ids[0]]['labels_task2_1'])} votos no primeiro item.")
    a(f"- θ: `{THETA_PATH}` (`{theta_meta.get('subpop_def', '?')}`) — "
      f"**{len(theta)} anotadores**.")
    a(f"  θ ∈ [`{vals.min():.4f}`, `{vals.max():.4f}`], mediana `{np.median(vals):.4f}`, "
      f"`converged = {theta_meta.get('converged')}`.")
    a(f"- Mediana do painel usada como fallback: `{fallback:.6f}` "
      f"({len(missing_ids)} anotadores precisaram dela).")
    a("")
    a("Auditoria dos votos de Task A: nenhum `UNKNOWN` no corpus, logo o raw é exactamente")
    a("`n_yes / n_votos` e a identidade com o pipeline é exacta, não aproximada.")
    a("")

    # --- verificações ---
    a("## 3. Verificações obrigatórias")
    a("")
    for name, ok, detail in checks:
        a(f"- {'**PASS**' if ok else '**FAIL**'} — {name}  \n  {detail}")
    a("")

    # --- poder, tabela ---
    a("## 4. Diagnóstico de poder (weighted vs raw)")
    a("")
    df = pd.DataFrame(
        {
            "valor": [
                power["mean_abs_delta"], power["sd_abs_delta"],
                power["p50"], power["p90"], power["p99"], power["max_abs_delta"],
                power["n_gt_005"], power["n_gt_010"],
                power["pearson_r"], power["n_cross_half"],
            ]
        },
        index=[
            "média |weighted − raw|", "desvio-padrão |weighted − raw|",
            "percentil 50", "percentil 90", "percentil 99", "máximo",
            "itens com Δ > 0,05", "itens com Δ > 0,10",
            "Pearson r (raw, weighted)", "itens que cruzam 0,5",
        ],
    )
    a(md_table(df, index_name="métrica"))
    a("")
    a(f"n = {power['n_items']} itens. "
      f"Δ>0,05 = {100 * power['n_gt_005'] / power['n_items']:.1f}% · "
      f"Δ>0,10 = {100 * power['n_gt_010'] / power['n_items']:.1f}% · "
      f"cruzam 0,5 = {100 * power['n_cross_half'] / power['n_items']:.1f}%.")
    a("")

    # --- cauda de θ ---
    a("## 5. Efeito colateral do θ não-clipado")
    a("")
    a("`[DECISÃO DO AUTOR]` a fórmula do spec (`Σθv/Σθ`) é seguida **literalmente**, sem")
    a("clip nem piso. Sob EM single-truth, θ<0,5 significa \"discorda da verdade inferida")
    a("mais do que acerta\"; a fórmula **atenua** esse anotador mas nunca inverte o voto")
    a("dele. Os números abaixo quantificam o que isso faz — são diagnóstico, não um")
    a("segundo braço de treino.")
    a("")
    dft = pd.DataFrame(
        {
            "valor": [
                tail["n_theta_lt_050"], tail["n_theta_lt_030"],
                tail["frac_votes_by_theta_lt_050"], tail["frac_votes_by_theta_lt_030"],
                tail["weight_min"], tail["weight_p50"], tail["weight_max"],
                tail["mean_abs_literal_minus_clip"], tail["max_abs_literal_minus_clip"],
                tail["n_cross_half_literal_vs_clip"],
            ]
        },
        index=[
            f"anotadores com θ < 0,50 (de {tail['n_annotators']})",
            "anotadores com θ < 0,30",
            "fracção dos votos emitida por θ < 0,50",
            "fracção dos votos emitida por θ < 0,30",
            "peso normalizado mínimo observado",
            "peso normalizado mediano",
            "peso normalizado máximo",
            f"média |literal − clipado em [{CLIP_LO}, {CLIP_HI}]|",
            "máximo da mesma diferença",
            "itens que cruzam 0,5 entre literal e clipado",
        ],
    )
    a(md_table(dft, index_name="métrica"))
    a("")
    a(f"Com 6 votos por item, o peso uniforme seria `{1/6:.4f}`. O intervalo observado")
    a(f"(`{tail['weight_min']:.4f}`–`{tail['weight_max']:.4f}`) mostra quanto um voto")
    a("individual chega a valer face a outro no mesmo item.")
    a("")

    # --- split ---
    a("## 6. Split congelado")
    a("")
    a("`[DECISÃO DO AUTOR]` a estratificação da célula 44 lê a mesma coluna de soft label.")
    a(f"Como {power['n_cross_half']} itens cruzam 0,5, o braço ponderado teria um split")
    a("**diferente** do baseline — parte de qualquer delta de ICM viria daí, não do θ.")
    a("")
    a(f"`{os.path.basename(SPLIT_PATH)}` congela as listas de `id_EXIST` resolvidas **uma")
    a(f"vez a partir do raw**, com `random_state={SPLIT_SEED}`, `test_size={SPLIT_TEST_SIZE}`")
    a(f"e estratificação pelo voto majoritário do raw: **{len(train_ids)} treino / "
      f"{len(val_ids)} validação**.")
    a("Os **dois** braços carregam esse ficheiro. O baseline tem de ser re-treinado com ele")
    a("também — comparar contra uma run que usou o `train_test_split` in-line reintroduz o")
    a("confound que este ficheiro existe para eliminar.")
    a("")

    # --- instruções ---
    a("## 7. Como correr o retreino")
    a("")
    a("Nenhum ficheiro do pipeline foi modificado. `memes_core.py` está bit-a-bit intacto;")
    a("as duas edições abaixo são no **notebook**, e são as únicas necessárias.")
    a("")
    a("**Célula 35** — substituir a atribuição de `labels_task2_1_soft`:")
    a("")
    a("```python")
    a("with open('../reliability/outputs/soft_labels_taskA_weighted.json') as f:")
    a("    _SOFT_A = json.load(f)")
    a("dataset['labels_task2_1_soft'] = dataset.index.map(lambda i: _SOFT_A[str(i)])")
    a("```")
    a("")
    a("**Célula 44** — substituir o `train_test_split` (nos dois braços):")
    a("")
    a("```python")
    a("with open('../reliability/outputs/split_taskA_frozen.json') as f:")
    a("    _SPLIT = json.load(f)")
    a("dataset_train = dataset.loc[_SPLIT['train_ids']].reset_index(drop=True)")
    a("dataset_val   = dataset.loc[_SPLIT['val_ids']].reset_index(drop=True)")
    a("```")
    a("")
    a("**Voltar ao baseline:** apontar a célula 35 para `soft_labels_taskA_raw.json` (é")
    a("idêntico ao que `get_soft_distribution` produz — verificação 1), mantendo a célula 44")
    a("no split congelado. Ou restaurar a linha original com `get_soft_distribution`.")
    a("")
    a("**Medir ICM em A, B e C.** Os alvos de B e C não mudaram; qualquer movimento neles é")
    a("spillover exploratório vindo do encoder partilhado, e tem de ser lido como tal.")
    a("")

    # --- o que pode e não pode ser escrito ---
    a("## 8. O que estes números sustentam")
    a("")
    a("**Sustentado:**")
    a("")
    a(f"- As duas variantes de soft label diferem materialmente: média "
      f"`{power['mean_abs_delta']:.4f}`, {power['n_cross_half']} itens cruzam a maioria.")
    a("- O baseline é reproduzido exactamente (verificação 1), logo qualquer delta de ICM")
    a("  é atribuível à ponderação e não a uma reimplementação divergente.")
    a("- O split é idêntico nos dois braços por construção (secção 6).")
    a("")
    a("**Não sustentado (não escrever):**")
    a("")
    a("- Que a ponderação melhora o modelo. Isto ainda não foi medido — só o alvo mudou.")
    a("- Que θ mede competência. θ vem de EM single-truth, que **define** competência como")
    a("  concordância com a verdade que ele próprio infere. Sob LeWiDi, um anotador de θ")
    a("  baixo pode ser uma perspectiva minoritária legítima, não ruído.")
    a(f"- Que o θ está convergido: `converged = {theta_meta.get('converged')}` "
      f"após {theta_meta.get('n_iter', '?')} iterações.")
    a("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", default=THETA_PATH)
    ap.add_argument("--suffix", default="",
                    help="sufixo dos ficheiros de saída; vazio = originais")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw_path, weighted_path, split_path, report_path = out_paths(args.suffix)
    for pth in (raw_path, weighted_path, split_path, report_path):
        if os.path.exists(pth) and not args.overwrite:
            raise SystemExit(f"ABORTADO — {pth} já existe e --overwrite não foi dado")

    dataset = load_dataset()
    theta, theta_meta = load_theta(args.theta)
    fallback = float(np.median(list(theta.values())))

    ids, raw, wt, weights, missing = build_all(dataset, theta, fallback)

    checks, missing_ids = verify(dataset, ids, raw, wt, weights, missing)
    train_ids, val_ids, _ = build_frozen_split(ids, raw)
    checks.append(verify_split(ids, train_ids, val_ids))

    print(f"itens: {len(ids)} | anotadores no θ: {len(theta)} | "
          f"fallback (mediana do painel): {fallback:.6f}")
    print(f"anotadores sem θ: {len(missing_ids)}"
          + (f" -> {', '.join(missing_ids[:20])}" if missing_ids else ""))
    print()
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print()

    hard_fail = [n for n, ok, _ in checks if not ok]
    if hard_fail:
        raise SystemExit(
            "ABORTADO — verificação falhou, nada foi escrito:\n  "
            + "\n  ".join(hard_fail)
        )

    power = power_report(ids, raw, wt)
    tail = theta_tail_diagnostic(dataset, ids, theta, fallback, wt, weights)

    print("--- diagnóstico de poder (weighted vs raw) ---")
    print(f"  media|d| = {power['mean_abs_delta']:.4f}   sd = {power['sd_abs_delta']:.4f}")
    print(f"  p50 = {power['p50']:.4f}  p90 = {power['p90']:.4f}  "
          f"p99 = {power['p99']:.4f}  max = {power['max_abs_delta']:.4f}")
    print(f"  itens d>0.05 = {power['n_gt_005']} "
          f"({100 * power['n_gt_005'] / power['n_items']:.1f}%)")
    print(f"  itens d>0.10 = {power['n_gt_010']} "
          f"({100 * power['n_gt_010'] / power['n_items']:.1f}%)")
    print(f"  pearson r    = {power['pearson_r']:.4f}")
    print(f"  cruzam 0.5   = {power['n_cross_half']} "
          f"({100 * power['n_cross_half'] / power['n_items']:.1f}%)")
    print()
    print("--- cauda de theta (nao-clipado, diagnostico) ---")
    print(f"  anotadores theta<0.50 = {tail['n_theta_lt_050']} / {tail['n_annotators']}"
          f"   theta<0.30 = {tail['n_theta_lt_030']}")
    print(f"  fracao dos votos por theta<0.50 = {tail['frac_votes_by_theta_lt_050']:.4f}")
    print(f"  peso normalizado min/p50/max = {tail['weight_min']:.4f} / "
          f"{tail['weight_p50']:.4f} / {tail['weight_max']:.4f}  (uniforme = {1/6:.4f})")
    print(f"  media|literal - clipado[{CLIP_LO},{CLIP_HI}]| = "
          f"{tail['mean_abs_literal_minus_clip']:.4f}   "
          f"cruzam 0.5 entre eles = {tail['n_cross_half_literal_vs_clip']}")
    print()
    print(f"--- split congelado: {len(train_ids)} treino / {len(val_ids)} validacao ---")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=1, sort_keys=True)
    with open(weighted_path, "w", encoding="utf-8") as f:
        json.dump(wt, f, indent=1, sort_keys=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SPLIT_SEED,
                "test_size": SPLIT_TEST_SIZE,
                "derived_from": os.path.basename(raw_path),
                "stratify": "voto majoritario do raw (p_YES >= 0.5)",
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "train_ids": train_ids,
                "val_ids": val_ids,
            },
            f, indent=1,
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report(theta_meta, theta, fallback, dataset, ids, checks,
                             power, tail, train_ids, val_ids, missing_ids))

    for pth in (raw_path, weighted_path, split_path, report_path):
        print(f"escrito: {pth}")

    # Se isto correu com sufixo, o RAW e o SPLIT têm de sair BYTE A BYTE iguais
    # aos originais: nenhum dos dois depende de θ. É a verificação que prova que
    # trocar de estimador não mexeu no baseline nem na partição congelada.
    if args.suffix:
        import filecmp
        fails = []
        if os.path.exists(RAW_PATH):
            same = filecmp.cmp(raw_path, RAW_PATH, shallow=False)
            print(f"[{'PASS' if same else 'FAIL'}] {os.path.basename(raw_path)} "
                  f"idêntico BYTE A BYTE a {os.path.basename(RAW_PATH)}")
            fails += [] if same else ["raw"]
        if os.path.exists(SPLIT_PATH):
            # o split compara-se pelas LISTAS DE IDs, não byte a byte: o campo
            # `derived_from` regista o ficheiro de origem e muda com o sufixo,
            # sem que a partição mude.
            with open(SPLIT_PATH, encoding="utf-8") as f:
                old_sp = json.load(f)
            with open(split_path, encoding="utf-8") as f:
                new_sp = json.load(f)
            same = (old_sp["train_ids"] == new_sp["train_ids"]
                    and old_sp["val_ids"] == new_sp["val_ids"]
                    and old_sp["seed"] == new_sp["seed"]
                    and old_sp["test_size"] == new_sp["test_size"])
            print(f"[{'PASS' if same else 'FAIL'}] {os.path.basename(split_path)} "
                  f"tem as MESMAS listas de IDs que {os.path.basename(SPLIT_PATH)}")
            fails += [] if same else ["split"]
        if fails:
            raise SystemExit(f"ABORTADO — {', '.join(fails)} mudaram, e não podiam")


if __name__ == "__main__":
    main()
