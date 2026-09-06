"""
NULO POR PERMUTAÇÃO para o experimento de reliability (Task A, memes).

Porquê
------
As 3 sementes provam que o Δ observado entre A0 (soft label crua) e A1 (soft label
ponderada por θ) não é ruído de inicialização. **Não** provam que o Δ vem do θ. Se
pesos ALEATÓRIOS com a mesma distribuição produzirem o mesmo Δ, o ganho é de
perturbação do alvo — efeito de regularização — e não de confiabilidade.

O nulo: manter a distribuição de θ exactamente, embaralhar só a ATRIBUIÇÃO
θ → anotador, e reconstruir as soft labels ponderadas.

Estratificação por painel linguístico (obrigatória)
---------------------------------------------------
Os painéis ES e EN são quase disjuntos. Permutar entre línguas destruiria
estrutura que **não é confiabilidade** (que anotador viu que língua), gerando um
nulo artificialmente fácil de bater — o braço permutado ficaria pior por um motivo
que nada tem a ver com θ, e o A1 pareceria bom por comparação.

Regra, em três estratos disjuntos, cada um permutado **dentro de si**:

  1. `es_only`     — anotou apenas itens `lang == es`
  2. `en_only`     — anotou apenas itens `lang == en`
  3. `bilingual`   — anotou itens das duas línguas

Os bilingues formam **estrato próprio** em vez de serem repartidos: é simétrico
(não privilegia nenhuma das línguas), preserva a estrutura de painel exactamente,
e é uma frase a reportar. Alternativas (atribuir à língua maioritária, ou excluir)
enviesariam ou perderiam 19 anotadores.

Estado
------
`[BLOQUEADO POR DECISÃO]` Este script é parametrizado pelo ficheiro de θ. Qual θ
alimenta o A1 está **em aberto** — ver `mstep_responsibility_diagnosis.md`: a
responsabilidade do M-step conta o voto do próprio anotador duas vezes e as soft
labels ponderadas movem-se 0,0246 em média (34% do efeito da própria ponderação).
Gerar os ficheiros antes da decisão obrigaria a gerá-los duas vezes.

Por isso o default é **dry-run**: valida a lógica de permutação e imprime as
estatísticas, sem escrever nada. `--write` gera os ficheiros, e só deve ser usado
depois de o θ estar decidido.

Uso:
    python3 -m reliability.make_permuted_soft_labels                 # validação
    python3 -m reliability.make_permuted_soft_labels --write         # gera
    python3 -m reliability.make_permuted_soft_labels --theta <path.json> [--theta-key theta]
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from reliability.make_soft_labels_taskA import (
    DATASET_PATH, LABELS_2_1, load_dataset, soft_raw, soft_weighted,
)

# ---------------------------------------------------------------------------
# CRITÉRIOS PRÉ-REGISTADOS — fixados antes de correr. Não alterar retroactivamente.
# ---------------------------------------------------------------------------
CRIT = {
    "P1_distribution_preserved": (
        "Dentro de cada estrato, o multiconjunto ordenado de θ é IDÊNTICO antes e "
        "depois (igualdade exacta de floats). FAIL = a permutação não é permutação."
    ),
    "P2_no_cross_language": (
        "Todo o anotador recebe um θ de um anotador do MESMO estrato. "
        "FAIL = houve fuga entre painéis linguísticos."
    ),
    "P3_fixed_points": (
        "Fração de pontos fixos (anotador que mantém o próprio θ) < 0.05 em cada "
        "permutação e estrato com n >= 20. FAIL = a permutação mal mexeu."
    ),
    "P4_strata_sizes": (
        "Os três estratos particionam os anotadores do ficheiro de θ, sem sobras. "
        "Os tamanhos são MEDIDOS e reportados, não assumidos."
    ),
    "P5_perturbation_magnitude": (
        "Descritivo, sem PASS/FAIL: média|Δp_YES| das labels permutadas contra a "
        "crua, comparada com a mesma quantidade para o A1 real. Se as permutadas "
        "perturbarem MUITO menos, o nulo é fraco e declara-se a limitação."
    ),
}

PERM_SEEDS = [101, 202, 303, 404, 505]   # fixas, documentadas
FIXED_POINT_MAX = 0.05
MIN_N_FOR_P3 = 20

DEFAULT_THETA = "reliability/outputs/theta_taskA_single_truth_K1.json"
OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "permutation_null_report.md")
MANIFEST_PATH = os.path.join(OUT_DIR, "soft_labels_taskA_permuted_manifest.json")


# ---------------------------------------------------------------------------
def load_theta(path, key="theta"):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    th = d[key] if key in d else d
    if not isinstance(th, dict):
        raise ValueError(f"{path}[{key}] não é um dict anotador->θ")
    return {str(k): float(v) for k, v in th.items()}, d


def annotator_languages(dataset):
    """anotador -> conjunto de línguas dos itens em que tem voto VÁLIDO na Task A.

    Só votos que a Task A reconhece ("YES"/"NO") contam — os mesmos que entram no
    estimador. Um anotador cujo único voto num idioma seja UNKNOWN não é bilingue
    para efeito deste estudo.
    """
    langs = defaultdict(set)
    for item in dataset.values():
        lang = str(item.get("lang", "?")).lower()
        votes, anns = item["labels_task2_1"], item["annotators"]
        for aid, v in zip(anns, votes):
            if str(v).strip().upper() in ("YES", "NO"):
                langs[str(aid)].add(lang)
    return langs


def build_strata(theta, langs):
    """Três estratos disjuntos. Anotadores do θ sem voto válido caem em `orphan`."""
    strata = {"es_only": [], "en_only": [], "bilingual": [], "orphan": []}
    for aid in theta:
        s = langs.get(aid, set())
        if s == {"es"}:
            strata["es_only"].append(aid)
        elif s == {"en"}:
            strata["en_only"].append(aid)
        elif len(s) > 1:
            strata["bilingual"].append(aid)
        else:
            strata["orphan"].append(aid)
    for k in strata:
        strata[k].sort()
    return strata


def permute_theta(theta, strata, seed):
    """Embaralha a atribuição θ→anotador DENTRO de cada estrato. `orphan` fica
    intacto (não há painel a que pertença; mexer nele inventaria estrutura)."""
    rng = np.random.default_rng(seed)
    out = dict(theta)
    donors = {}
    for name, members in strata.items():
        if name == "orphan" or len(members) < 2:
            for a in members:
                donors[a] = a
            continue
        perm = rng.permutation(len(members))
        for pos, a in enumerate(members):
            src = members[perm[pos]]
            out[a] = theta[src]
            donors[a] = src
    return out, donors


# ---------------------------------------------------------------------------
# Verificações pré-registadas
# ---------------------------------------------------------------------------
def check_permutation(theta, permuted, strata, donors, seed):
    res = {"seed": int(seed), "strata": {}}
    p1 = p2 = p3 = True
    for name, members in strata.items():
        if not members:
            continue
        before = sorted(theta[a] for a in members)
        after = sorted(permuted[a] for a in members)
        ok_dist = before == after
        ok_cross = all(donors[a] in set(members) for a in members)
        n_fixed = sum(1 for a in members if donors[a] == a)
        frac_fixed = n_fixed / len(members)
        ok_fixed = (len(members) < MIN_N_FOR_P3) or (frac_fixed < FIXED_POINT_MAX)
        p1 &= ok_dist
        p2 &= ok_cross
        p3 &= ok_fixed
        res["strata"][name] = {
            "n": len(members), "dist_preserved": bool(ok_dist),
            "no_cross": bool(ok_cross), "n_fixed_points": int(n_fixed),
            "frac_fixed": float(frac_fixed), "p3_ok": bool(ok_fixed),
        }
    res["P1"] = bool(p1)
    res["P2"] = bool(p2)
    res["P3"] = bool(p3)
    return res


def soft_labels_from_theta(dataset, theta):
    fb = float(np.median(list(theta.values())))
    ids, out = [], {}
    for item_id, item in dataset.items():
        votes, anns = item["labels_task2_1"], item["annotators"]
        if len(votes) != len(anns):
            raise ValueError(f"item {item_id}: votos vs anotadores")
        r = soft_raw(votes)
        w, _, _ = soft_weighted(votes, anns, theta, fb)
        if r is None or w is None:
            continue
        ids.append(item_id)
        out[item_id] = w
    return ids, out


def mean_abs_shift(a, b, ids):
    return float(np.mean([abs(a[i][0] - b[i][0]) for i in ids]))


# ---------------------------------------------------------------------------
def build_report(theta_path, theta_meta, strata, checks, mags, wrote, files):
    L = []
    a = L.append
    a("# Nulo por permutação — geração e validação")
    a("")
    a("Gerado por `reliability/make_permuted_soft_labels.py`.")
    a("")
    a(f"- θ de origem: `{theta_path}`")
    a(f"- sementes de permutação (fixas): `{PERM_SEEDS}`")
    a(f"- ficheiros escritos: **{'sim' if wrote else 'NÃO (dry-run)'}**")
    a("")
    a("## 0. Porquê estratificar por painel linguístico")
    a("")
    a("Os painéis ES e EN são quase disjuntos. Permutar θ entre línguas destruiria")
    a("estrutura que **não é confiabilidade** — que anotador viu que língua — e daria")
    a("um nulo artificialmente fácil de bater. Três estratos disjuntos, cada um")
    a("permutado dentro de si. Os **bilingues formam estrato próprio**: simétrico, não")
    a("privilegia nenhuma língua, preserva a estrutura de painel exactamente.")
    a("")
    a("## 1. Estratos `[MEDIDO]`")
    a("")
    a("| estrato | n |")
    a("|---|---|")
    for k in ("es_only", "en_only", "bilingual", "orphan"):
        a(f"| `{k}` | {len(strata[k])} |")
    a(f"| **total** | **{sum(len(v) for v in strata.values())}** |")
    a("")
    if strata["orphan"]:
        a(f"`[LIMITAÇÃO]` {len(strata['orphan'])} anotador(es) no ficheiro de θ sem voto")
        a("válido na Task A — ficam com o próprio θ, sem permutação. Não pertencem a")
        a("painel nenhum; mexer neles inventaria estrutura.")
        a("")
    a("## 2. Critérios pré-registados")
    a("")
    for k, v in CRIT.items():
        a(f"- **{k}** — {v}")
    a("")
    a("## 3. Resultado das verificações `[MEDIDO]`")
    a("")
    a("| seed | P1 distribuição | P2 sem fuga | P3 pontos fixos | máx frac. fixos |")
    a("|---|---|---|---|---|")
    for c in checks:
        mf = max((s["frac_fixed"] for s in c["strata"].values()
                  if s["n"] >= MIN_N_FOR_P3), default=0.0)
        a(f"| {c['seed']} | {'PASS' if c['P1'] else 'FAIL'} | "
          f"{'PASS' if c['P2'] else 'FAIL'} | {'PASS' if c['P3'] else 'FAIL'} | "
          f"{mf:.4f} |")
    a("")
    a("## 4. P5 — magnitude da perturbação `[MEDIDO]`")
    a("")
    a("média|Δp_YES| contra a soft label **crua**. O A1 real é o padrão: se as")
    a("permutadas perturbarem muito menos, o nulo é fraco.")
    a("")
    a("| braço | média\\|Δp_YES\\| vs crua |")
    a("|---|---|")
    a(f"| A1 real (θ na sua atribuição) | **{mags['real']:.6f}** |")
    for sd, m in mags["perm"]:
        a(f"| permutado seed {sd} | {m:.6f} |")
    pm = [m for _, m in mags["perm"]]
    a(f"| permutado: média | {np.mean(pm):.6f} |")
    a("")
    ratio = np.mean(pm) / mags["real"] if mags["real"] else float("nan")
    a(f"Razão permutado/real = **{ratio:.3f}**.")
    if ratio < 0.5:
        a("")
        a("`[LIMITAÇÃO]` As permutadas perturbam o alvo bem menos que o A1 real. Um Δ")
        a("menor no braço permutado passa a ser explicável por perturbação menor, não")
        a("só por ausência de sinal de confiabilidade — o nulo perde poder de")
        a("discriminação e isso tem de ser dito ao lado do resultado.")
    a("")
    a("## 5. Ficheiros")
    a("")
    if wrote:
        for f_ in files:
            a(f"- `{f_}`")
        a(f"- `{os.path.basename(MANIFEST_PATH)}` — metadados, verificações e o mapa")
        a("  doador→receptor de cada permutação. Fica **fora** dos ficheiros de")
        a("  labels para que o formato seja idêntico ao")
        a("  `soft_labels_taskA_weighted.json`: dict plano `id → [p_YES, p_NO]`.")
        a("")
        a("`[LIMITAÇÃO]` Estes ficheiros foram construídos a partir do θ indicado em")
        a("cima. Se a decisão sobre `mstep_responsibility_diagnosis.md` for regenerar")
        a("o θ, estes cinco ficheiros têm de ser regenerados com ele — a permutação é")
        a("do mesmo vector que alimenta o A1, senão o nulo não é o nulo do A1.")
    else:
        a("Nenhum. `[BLOQUEADO POR DECISÃO]` — qual θ alimenta o A1 está em aberto")
        a("(`mstep_responsibility_diagnosis.md`). Correr com `--write` só depois de")
        a("decidido, para não gerar os ficheiros duas vezes.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default=DEFAULT_THETA)
    ap.add_argument("--theta-key", default="theta")
    ap.add_argument("--suffix", default="",
                    help="sufixo dos ficheiros gerados (ex.: _fixed)")
    ap.add_argument("--write", action="store_true",
                    help="escreve os ficheiros de soft labels permutadas")
    args = ap.parse_args()

    theta, meta = load_theta(args.theta, args.theta_key)
    dataset = load_dataset(DATASET_PATH)
    langs = annotator_languages(dataset)
    strata = build_strata(theta, langs)
    print(f"θ: {args.theta} ({len(theta)} anotadores)")
    print("estratos: " + " · ".join(f"{k}={len(v)}" for k, v in strata.items()))

    ids_real, real = soft_labels_from_theta(dataset, theta)
    _ids_raw, raw = None, {}
    for item_id, item in dataset.items():
        r = soft_raw(item["labels_task2_1"])
        if r is not None:
            raw[item_id] = r
    mags = {"real": mean_abs_shift(real, raw, ids_real), "perm": []}

    checks, files, manifests = [], [], []
    os.makedirs(OUT_DIR, exist_ok=True)
    for sd in PERM_SEEDS:
        permuted, donors = permute_theta(theta, strata, sd)
        c = check_permutation(theta, permuted, strata, donors, sd)
        checks.append(c)
        ids_p, labels_p = soft_labels_from_theta(dataset, permuted)
        assert ids_p == ids_real, "conjunto de itens mudou entre braços"
        mags["perm"].append((sd, mean_abs_shift(labels_p, raw, ids_real)))
        print(f"  seed {sd}: P1={c['P1']} P2={c['P2']} P3={c['P3']} "
              f"| perturbação vs crua = {mags['perm'][-1][1]:.6f}")
        if args.write:
            path = os.path.join(
                OUT_DIR, f"soft_labels_taskA_permuted_seed{sd}{args.suffix}.json")
            if os.path.exists(path):
                raise FileExistsError(f"{path} já existe — não sobrescrever")
            # MESMO formato de `soft_labels_taskA_weighted.json`: dict plano
            # id_EXIST -> [p_YES, p_NO], indent=1. Sem chaves extra, para o
            # notebook consumir os ficheiros sem nenhuma ramificação nova.
            # chaves ordenadas, como no ficheiro existente
            with open(path, "w", encoding="utf-8") as f:
                json.dump({k: labels_p[k] for k in sorted(labels_p)}, f,
                          ensure_ascii=False, indent=1)
            files.append(os.path.basename(path))
            manifests.append({
                "file": os.path.basename(path),
                "perm_seed": int(sd),
                "source_theta": args.theta,
                "labels_order": LABELS_2_1,
                "n_items": len(labels_p),
                "strata_sizes": {k: len(v) for k, v in strata.items()},
                "bilingual_rule": "estrato próprio, permutado dentro de si",
                "checks": c,
                "mean_abs_shift_vs_raw": mags["perm"][-1][1],
                "donors": {a: donors[a] for a in sorted(donors)},
            })

    if args.write:
        # metadados fora dos ficheiros de labels, para o formato ficar idêntico
        manifest_path = MANIFEST_PATH.replace(
            ".json", f"{args.suffix}.json") if args.suffix else MANIFEST_PATH
        if os.path.exists(manifest_path) and args.suffix:
            raise FileExistsError(f"{manifest_path} já existe — não sobrescrever")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"criteria": CRIT, "perm_seeds": PERM_SEEDS,
                       "permutations": manifests}, f, indent=2, ensure_ascii=False)
        print(f"manifesto: {manifest_path}")

    report_path = REPORT_PATH.replace(".md", f"{args.suffix}.md") \
        if args.suffix else REPORT_PATH
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report(args.theta, meta, strata, checks, mags,
                             args.write, files))
    print(f"\nescrito: {report_path}")
    if not args.write:
        print("dry-run: nenhum ficheiro de labels gerado (θ ainda por decidir).")


if __name__ == "__main__":
    main()
