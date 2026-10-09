"""F4 (SPEC_27 §9 e §16A): checagens (a)–(i) e portão G4 de retreino. Tudo no split de treino.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/checks.py

Saídas: reports/checks_summary.md, reports/gate_decision.md, manual_review_unanimous.csv,
annotator_deviation_parcels.csv.
"""
import csv
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from client import load_records
from common import (FIXED, REF, REPORTS, ROOT, T, TRAIN_IDS, VOTES_CSV, assert_config, image_path,
                    load_json, load_memes, require_gate, set_gate)
from weights import (METHOD, annotator_demographics, deviation_from_terms, deviation_terms,
                     glm_labels, load_glm_votes, load_train_panels, n_yes, soft_votes)

B = FIXED["bootstrap"]
SCOPES = ("total", "en", "es")
SPLITS = ("6-0", "5-1", "4-2", "3-3")


def spearman(x, y) -> float:
    """Pearson dos ranks (SPEC_25 §1: sem SciPy)."""
    return float(pd.Series(list(x)).rank().corr(pd.Series(list(y)).rank()))


def hard(p: float) -> str:
    return "YES" if p > 0.5 + T["tol"] else "NO" if p < 0.5 - T["tol"] else "TIE"


def human_split(votes: dict) -> str:
    k, n = n_yes(votes), len(votes)
    return f"{max(k, n - k)}-{min(k, n - k)}"


def glm_pattern(labels: list) -> str | None:
    if "?" in labels:
        return None
    return "3-0" if len(set(labels)) == 1 else "2-1"


def dissident(labels: list) -> str:
    """Num meme 2-1, a paráfrase que discordou da maioria."""
    maj = Counter(labels).most_common(1)[0][0]
    return next(f"P{p}" for p, l in zip((1, 2, 3), labels) if l != maj)


def bootstrap_mean_ci(x, rng) -> tuple:
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), (B["n"], len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def f(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def quantiles(xs) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    q = np.percentile(xs, [25, 50, 75])
    return f"n {len(xs)} · p25 {q[0]:.4f} · mediana {q[1]:.4f} · p75 {q[2]:.4f}"


def crosstab(rows_key, ids, lang, rows_order, panels) -> list:
    """Tabela (linha × divisão humana), contagens e proporção por linha."""
    L = ["| escopo | linha | n | " + " | ".join(SPLITS) + " | 4-2 + 3-3 |", "|" + "---|" * (len(SPLITS) + 4)]
    for scope in SCOPES:
        for row in rows_order:
            sel = [i for i in ids if rows_key[i] == row and (scope == "total" or lang[i] == scope)]
            c = Counter(human_split(panels[i]) for i in sel)
            n = len(sel)
            cells = [f"{c[s]} ({c[s] / n:.3f})" if n else "0" for s in SPLITS]
            amb = (c["4-2"] + c["3-3"]) / n if n else float("nan")
            L.append(f"| {scope} | {row} | {n} | " + " | ".join(cells) + f" | **{f(amb, 3)}** |")
    return L


def main():
    cfg = assert_config()
    require_gate(cfg, "F3")
    assert cfg["runtime"].get("method") == METHOD, "pesos não foram calculados pelo método do doc 27"
    memes = load_memes()
    train = load_json(TRAIN_IDS)
    panels, _ = load_train_panels(memes, train)
    glm_rows = load_glm_votes(VOTES_CSV)
    p_glm = soft_votes(glm_rows)
    aw = pd.read_csv(REF / "annotator_weights.csv", dtype={"annotator_id": str})
    soft = load_json(REF / "soft_labels_weighted.json")
    lang = {i: memes[i]["lang"] for i in train}
    inc = aw[aw.n_included >= 1]
    terms = deviation_terms(panels, p_glm)
    by_ann = defaultdict(list)
    for a, mid, w, d in terms:
        by_ann[a].append((mid, w, d))

    L = ["# F4 — Checagens (a)–(i) · doc 27 (distância ao soft label do GLM)", "",
         f"Gerado por `checks.py`. Split de treino: {len(train)} memes, {len(aw)} anotadores, "
         f"{len(inc)} com ≥ 1 meme incluído. Método: `{METHOD}`.", "",
         "> **Escala (SPEC_27 §16A, P1):** num meme em que o GLM fica 2-1, todo anotador leva pelo "
         "menos 1/3 de desvio. Os `D` deste relatório **não são comparáveis** com os do doc 25.", ""]

    # (a) linha de base dos pares (soft)
    rho_a = spearman(inc.r_glm, inc.r_peer)
    L += ["## (a) GLM vs consenso dos pares", "",
          f"ρ de Spearman(r_glm, r_peer) = **{f(rho_a)}** (n = {len(inc)}; `D_peer` com "
          "d = |voto − fração de YES dos outros 5|).", ""]

    # (b) limiar — ambas as taxas nos memes incluídos do anotador (decisão de 2026-10-06)
    gap = {a: abs(np.mean([panels[m][a] == "YES" for m, _, _ in t]) - np.mean([p_glm[m] for m, _, _ in t]))
           for a, t in by_ann.items()}
    b_ok = inc[inc.D_glm.notna()]
    L += ["## (b) O peso só mede o limiar?", "",
          f"- ρ(D_glm, taxa de YES do anotador em todos os seus memes de treino) = "
          f"**{f(spearman(b_ok.D_glm, b_ok.yes_rate))}**",
          f"- ρ(D_glm, |taxa de YES do anotador − média de p_GLM|, ambas nos memes incluídos) = "
          f"**{f(spearman(b_ok.D_glm, b_ok.annotator_id.map(gap)))}**", ""]

    # (c) demografia
    rng = np.random.default_rng(B["rng"])
    L += ["## (c) r por gênero × idade (anotadores com ≥ 1 meme incluído; IC 95% bootstrap de anotadores)", "",
          "| gênero | idade | n | r_glm | IC 95% | r_peer | IC 95% |", "|---|---|---|---|---|---|---|"]
    for (g, age), grp in inc.groupby(["gender", "age"], sort=True):
        cg, cp = bootstrap_mean_ci(grp.r_glm, rng), bootstrap_mean_ci(grp.r_peer, rng)
        L.append(f"| {g} | {age} | {len(grp)} | {f(grp.r_glm.mean())} | [{f(cg[0])}, {f(cg[1])}] | "
                 f"{f(grp.r_peer.mean())} | [{f(cp[0])}, {f(cp[1])}] |")
    _, multi = annotator_demographics(memes, train)
    L += ["", f"Anotadores em mais de um painel (demografia conferida como consistente): {multi}.", ""]

    # (d) portão: quanto os labels se movem
    moved = {i: abs(s["new"][0] - s["old"][0]) >= T["half_vote"] - T["tol"] for i, s in soft.items()}
    changes = {i: (hard(s["old"][0]), hard(s["new"][0])) for i, s in soft.items()
               if hard(s["old"][0]) != hard(s["new"][0])}
    share_moved = sum(moved.values()) / len(moved)
    L += ["## (d) Portão — quanto os soft labels se movem", "",
          "| escopo | share_moved (|Δp| ≥ 1/12) | mudanças de rótulo duro |", "|---|---|---|"]
    for scope in SCOPES:
        ids = [i for i in soft if scope == "total" or lang[i] == scope]
        L.append(f"| {scope} | {f(sum(moved[i] for i in ids) / len(ids))} | {sum(i in changes for i in ids)} |")
    kinds = Counter(f"{a}→{b}" for a, b in changes.values())
    L += ["", "Tipos de mudança: " + (", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) or "nenhuma"), ""]

    # (e) GLM confiante (3-0) contra meme unânime — descritiva
    conf = {i: glm_rows[i]["glm_confident"] == "True" for i in train}
    unan = [i for i in train if n_yes(panels[i]) in (0, len(panels[i])) and conf[i]]
    contra = [i for i in unan if glm_rows[i]["glm_vote"] != ("YES" if n_yes(panels[i]) else "NO")]
    L += ["## (e) GLM confiante (3-0) contrariando meme unânime", "",
          "| escopo | unânimes com GLM 3-0 | contrariados | taxa |", "|---|---|---|---|"]
    for scope in SCOPES:
        u = [i for i in unan if scope == "total" or lang[i] == scope]
        c = [i for i in contra if scope == "total" or lang[i] == scope]
        L.append(f"| {scope} | {len(u)} | {len(c)} | {f(len(c) / len(u) if u else float('nan'))} |")
    sample = sorted(np.random.default_rng(B["rng"]).choice(sorted(contra), size=min(30, len(contra)),
                                                           replace=False).tolist()) if contra else []
    arm = cfg["runtime"]["chosen_arm"]
    recs = load_records(arm)
    with open(REF / "manual_review_unanimous.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["meme_id", "lang", "image_path", "text", "votes", "label_p1", "label_p2",
                    "label_p3", "reasoning"])
        for i in sample:
            reasons = [recs[(i, arm, p, FIXED["seed_reference"])]["reasoning_content"] or "" for p in (1, 2, 3)]
            w.writerow([i, lang[i], str(image_path(memes[i]).relative_to(ROOT)), memes[i]["text"],
                        ";".join(memes[i]["labels_task2_1"]), *glm_labels(glm_rows[i]),
                        "\n---\n".join(r for r in reasons if r)])
    L += ["", f"{len(sample)} memes sorteados em `manual_review_unanimous.csv` para revisão manual.", ""]

    # (f) concordância descritiva com o rótulo duro humano
    L += ["## (f) Acordo de glm_vote (maioria das 3 paráfrases) com o rótulo duro humano (empates 3-3 fora)", "",
          "| escopo | divisão | n | acordo |", "|---|---|---|---|"]
    for scope in SCOPES:
        for split in ("todas", "6-0", "5-1", "4-2"):
            ok = []
            for i in train:
                k, n = n_yes(panels[i]), len(panels[i])
                v = glm_rows[i]["glm_vote"]
                if 2 * k == n or not v or (scope != "total" and lang[i] != scope):
                    continue
                if split != "todas" and human_split(panels[i]) != split:
                    continue
                ok.append(v == ("YES" if 2 * k > n else "NO"))
            L.append(f"| {scope} | {split} | {len(ok)} | {f(np.mean(ok) if ok else float('nan'))} |")
    L.append("")

    # (g) precisão por anotador + parcelas 3-0 / 2-1 (SPEC_27 §16A, P1)
    rng = np.random.default_rng(B["rng"])
    widths = []
    for a in sorted(by_ann):
        wd = np.array([(w, d) for _, w, d in by_ann[a]], dtype=float)
        idx = rng.integers(0, len(wd), (B["n"], len(wd)))
        W, Dd = wd[idx, 0], wd[idx, 1]
        sw = W.sum(axis=1)
        Ds = (W * Dd).sum(axis=1)[sw > 0] / sw[sw > 0]
        if len(Ds):
            widths.append(float(np.percentile(Ds, 97.5) - np.percentile(Ds, 2.5)))
    anns = list(aw.annotator_id)
    is30 = lambda m: p_glm[m] in (0.0, 1.0)                     # noqa: E731
    D30 = deviation_from_terms([t for t in terms if is30(t[1])], anns)
    D21 = deviation_from_terms([t for t in terms if not is30(t[1])], anns)
    m30, m21 = Counter(), Counter()
    for a, mid, w, _ in terms:
        (m30 if is30(mid) else m21)[a] += w
    with open(REF / "annotator_deviation_parcels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["annotator_id", "D_glm", "D_glm_30", "D_glm_21", "mass_30", "mass_21"])
        for a, d in zip(aw.annotator_id, aw.D_glm):
            w.writerow([a, "" if pd.isna(d) else d, "" if D30[a] is None else D30[a],
                        "" if D21[a] is None else D21[a], m30[a], m21[a]])
    q = aw.n_included.quantile([0, .25, .5, .75, 1]).tolist()
    L += ["## (g) Precisão por anotador", "",
          f"- n_included (todos os {len(aw)} anotadores): mín {q[0]:.0f} · p25 {q[1]:.0f} · "
          f"mediana {q[2]:.0f} · p75 {q[3]:.0f} · máx {q[4]:.0f} · média {aw.n_included.mean():.2f}",
          f"- mass_included (Σ w por anotador): {quantiles(aw.mass_included.tolist())}",
          f"- fração com < 5 memes incluídos: **{f((aw.n_included < 5).mean())}**",
          f"- largura mediana do IC 95% de D_glm (bootstrap dos memes de cada anotador, "
          f"{len(widths)} anotadores): **{f(float(np.median(widths)) if widths else float('nan'))}**", "",
          "**Parcelas do D_glm** (mesma fórmula, mesmo w; por anotador em `annotator_deviation_parcels.csv`):", "",
          f"- D_glm total: {quantiles([None if pd.isna(x) else x for x in aw.D_glm])}",
          f"- D_glm só com memes GLM 3-0: {quantiles(D30.values())}",
          f"- D_glm só com memes GLM 2-1: {quantiles(D21.values())} (piso de 1/3 por construção)", ""]

    # (h) quanto da massa vem dos memes 2-1
    L += ["## (h) Fração da massa Σ w que vem de memes GLM 2-1", "", "| escopo | massa 2-1 / massa total |", "|---|---|"]
    for scope in SCOPES:
        sel = [(mid, w) for _, mid, w, _ in terms if scope == "total" or lang[mid] == scope]
        tot = sum(w for _, w in sel)
        L.append(f"| {scope} | {f(sum(w for mid, w in sel if not is30(mid)) / tot if tot else float('nan'))} |")
    L.append("")

    # (i) hesitação do GLM × divisão humana (SPEC_27 §16A, P2)
    pat = {i: glm_pattern(glm_labels(glm_rows[i])) for i in train}
    ids = [i for i in train if pat[i] is not None]
    ids21 = [i for i in ids if pat[i] == "2-1"]
    dis = {i: dissident(glm_labels(glm_rows[i])) for i in ids21}
    L += ["## (i) Hesitação do GLM × divisão humana", "",
          "Descritiva; não decide nada e não entra no G4. Contagens e (proporção na linha).", "",
          "**Padrão do GLM × divisão dos votos humanos:**", "",
          *crosstab(pat, ids, lang, ("3-0", "2-1"), panels), "",
          "**Memes GLM 2-1, pela paráfrase que discordou:**", "",
          *crosstab(dis, ids21, lang, ("P1", "P2", "P3"), panels), ""]
    for scope in SCOPES:
        amb, n = {}, {}
        for row in ("3-0", "2-1"):
            sel = [i for i in ids if pat[i] == row and (scope == "total" or lang[i] == scope)]
            n[row] = len(sel)
            amb[row] = sum(human_split(panels[i]) in ("4-2", "3-3") for i in sel) / len(sel) if sel else float("nan")
        diff = amb["2-1"] - amb["3-0"]
        se = np.sqrt(sum(amb[r] * (1 - amb[r]) / n[r] for r in amb)) if all(n.values()) else float("nan")
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        verdict = ("têm **mais** divisões humanas fortes" if lo > 0 else
                   "têm **menos** divisões humanas fortes" if hi < 0 else
                   "**não diferem** dos 3-0 em divisões humanas fortes (IC inclui zero)")
        L.append(f"- {scope}: 4-2 + 3-3 = {f(amb['2-1'], 3)} nos GLM 2-1 vs {f(amb['3-0'], 3)} nos GLM 3-0; "
                 f"diferença {diff:+.3f} (IC 95% [{lo:+.3f}, {hi:+.3f}]) → os memes em que o GLM hesita {verdict}.")
    L += ["", "Leitura pré-registrada: se os 2-1 **não** tiverem mais 4-2/3-3 que os 3-0, a [HIPÓTESE] "
              "\"os memes em que o GLM hesita são os ambíguos\" não se sustenta. \"Mais\" só quando o IC 95% "
              "da diferença (aproximação normal para duas proporções) exclui zero. Nada muda no cálculo.", ""]

    ok = share_moved >= T["share_moved"] and rho_a < T["rho"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "checks_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    G = ["# Portão G4 — retreino (doc 27)", "",
         f"- share_moved = **{f(share_moved)}** (exige ≥ {T['share_moved']}) → "
         f"{'ok' if share_moved >= T['share_moved'] else 'FALHA'}",
         f"- ρ(r_glm, r_peer) = **{f(rho_a)}** (exige < {T['rho']}) → {'ok' if rho_a < T['rho'] else 'FALHA'}",
         "", f"**Decisão: {'RETREINO AUTORIZADO' if ok else 'RETREINO NÃO AUTORIZADO — encerrar (§11)'}**"]
    (REPORTS / "gate_decision.md").write_text("\n".join(G) + "\n", encoding="utf-8")
    set_gate("G4", ok, share_moved=share_moved, rho_r_glm_r_peer=rho_a, method=METHOD)
    print("\n".join(L + [""] + G))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
