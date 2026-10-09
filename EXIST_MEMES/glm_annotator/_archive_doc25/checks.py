"""F4 (§9): checagens (a)–(g) e portão G4 de retreino. Tudo no split de treino.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/checks.py

Saídas: reports/checks_summary.md, reports/gate_decision.md, manual_review_unanimous.csv.
"""
import csv
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from client import load_records
from common import (FIXED, REF, REPORTS, ROOT, T, TRAIN_IDS, VOTES_CSV, assert_config, image_path,
                    load_json, load_memes, require_gate, set_gate)
from weights import (annotator_demographics, confident_votes, deviation_terms, load_glm_votes,
                     load_train_panels, n_yes)

B = FIXED["bootstrap"]


def spearman(x, y) -> float:
    """Pearson dos ranks (§1: sem SciPy)."""
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def hard(p: float) -> str:
    return "YES" if p > 0.5 + T["tol"] else "NO" if p < 0.5 - T["tol"] else "TIE"


def bootstrap_mean_ci(x, rng) -> tuple:
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), (B["n"], len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def f(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main():
    cfg = assert_config()
    require_gate(cfg, "F3")
    memes = load_memes()
    train = load_json(TRAIN_IDS)
    panels, _ = load_train_panels(memes, train)
    glm_rows = load_glm_votes(VOTES_CSV)
    glm_conf = confident_votes(glm_rows)
    aw = pd.read_csv(REF / "annotator_weights.csv")
    soft = load_json(REF / "soft_labels_weighted.json")
    lang = {i: memes[i]["lang"] for i in train}
    inc = aw[aw.n_included >= 1]
    L = ["# F4 — Checagens (a)–(g)", "",
         f"Gerado por `checks.py`. Split de treino: {len(train)} memes, {len(aw)} anotadores, "
         f"{len(inc)} com ≥ 1 meme incluído.", ""]

    # (a) linha de base dos pares
    rho_a = spearman(inc.r_glm, inc.r_peer)
    L += ["## (a) GLM vs consenso dos pares", "",
          f"ρ de Spearman(r_glm, r_peer) = **{f(rho_a)}** (n = {len(inc)} anotadores com ≥ 1 meme incluído).", ""]

    # (b) limiar — 2ª correlação sobre os memes INCLUÍDOS de cada anotador (decisão 2026-10-06)
    terms = deviation_terms(panels, glm_conf)
    by_ann = defaultdict(list)
    for a, mid, w, d in terms:
        by_ann[a].append((mid, w, d))
    gap = {a: abs(np.mean([panels[m][a] == "YES" for m, _, _ in t])
                  - np.mean([glm_conf[m] == "YES" for m, _, _ in t])) for a, t in by_ann.items()}
    b_ok = inc[inc.D_glm.notna()]
    rho_b1 = spearman(b_ok.D_glm, b_ok.yes_rate)
    rho_b2 = spearman(b_ok.D_glm, b_ok.annotator_id.map(gap))
    L += ["## (b) O peso só mede o limiar?", "",
          f"- ρ(D_glm, taxa de YES do anotador em todos os seus memes de treino) = **{f(rho_b1)}**",
          f"- ρ(D_glm, |taxa de YES do anotador − taxa de YES do GLM|, ambas nos memes incluídos "
          f"do anotador) = **{f(rho_b2)}**", ""]

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
    for scope in ("total", "en", "es"):
        ids = [i for i in soft if scope == "total" or lang[i] == scope]
        L.append(f"| {scope} | {f(sum(moved[i] for i in ids) / len(ids))} | "
                 f"{sum(i in changes for i in ids)} |")
    kinds = Counter(f"{a}→{b}" for a, b in changes.values())
    L += ["", "Tipos de mudança: " + (", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) or "nenhuma"), ""]

    # (e) GLM confiante contra meme unânime
    unan = [i for i in train if n_yes(panels[i]) in (0, len(panels[i])) and glm_conf[i] is not None]
    contra = [i for i in unan if glm_conf[i] != ("YES" if n_yes(panels[i]) else "NO")]
    L += ["## (e) GLM confiante contrariando meme unânime", "",
          "| escopo | unânimes com GLM confiante | contrariados | taxa |", "|---|---|---|---|"]
    for scope in ("total", "en", "es"):
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
                        ";".join(memes[i]["labels_task2_1"]),
                        *(glm_rows[i][f"label_p{p}"] for p in (1, 2, 3)),
                        "\n---\n".join(r for r in reasons if r)])
    L += ["", f"{len(sample)} memes sorteados em `manual_review_unanimous.csv` para revisão manual.", ""]

    # (f) concordância descritiva com o rótulo duro humano
    L += ["## (f) Acordo de glm_vote com o rótulo duro humano (empates 3-3 fora)", "",
          "| escopo | divisão | n | acordo |", "|---|---|---|---|"]
    for scope in ("total", "en", "es"):
        for split in ("todas", "6-0", "5-1", "4-2"):
            ok = []
            for i in train:
                k, n = n_yes(panels[i]), len(panels[i])
                v = glm_rows[i]["glm_vote"]
                if 2 * k == n or not v or (scope != "total" and lang[i] != scope):
                    continue
                if split != "todas" and f"{max(k, n - k)}-{min(k, n - k)}" != split:
                    continue
                ok.append(v == ("YES" if 2 * k > n else "NO"))
            L.append(f"| {scope} | {split} | {len(ok)} | {f(np.mean(ok) if ok else float('nan'))} |")
    L.append("")

    # (g) precisão por anotador
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
    q = aw.n_included.quantile([0, .25, .5, .75, 1]).tolist()
    L += ["## (g) Precisão por anotador", "",
          f"- n_included (todos os {len(aw)} anotadores): mín {q[0]:.0f} · p25 {q[1]:.0f} · "
          f"mediana {q[2]:.0f} · p75 {q[3]:.0f} · máx {q[4]:.0f} · média {aw.n_included.mean():.2f}",
          f"- fração com < 5 memes incluídos: **{f((aw.n_included < 5).mean())}**",
          f"- largura mediana do IC 95% de D_glm (bootstrap dos memes de cada anotador, "
          f"{len(widths)} anotadores): **{f(float(np.median(widths)) if widths else float('nan'))}**", ""]

    ok = share_moved >= T["share_moved"] and rho_a < T["rho"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "checks_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    G = ["# Portão G4 — retreino", "",
         f"- share_moved = **{f(share_moved)}** (exige ≥ {T['share_moved']}) → "
         f"{'ok' if share_moved >= T['share_moved'] else 'FALHA'}",
         f"- ρ(r_glm, r_peer) = **{f(rho_a)}** (exige < {T['rho']}) → {'ok' if rho_a < T['rho'] else 'FALHA'}",
         "", f"**Decisão: {'RETREINO AUTORIZADO' if ok else 'RETREINO NÃO AUTORIZADO — encerrar (§11)'}**"]
    (REPORTS / "gate_decision.md").write_text("\n".join(G) + "\n", encoding="utf-8")
    set_gate("G4", ok, share_moved=share_moved, rho_r_glm_r_peer=rho_a)
    print("\n".join(L + [""] + G))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
