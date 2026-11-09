"""Figura do doc 30: GLM com thinking desligado vs ligado, contra os humanos (treino).

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/plot_doc30.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import REF, REPORTS, SOFT_RAW, TRAIN_IDS, load_json, load_memes
from weights import glm_soft

ARMS = [("think_off", "thinking desligado", "#2a78d6", REF / "glm_votes.csv"),
        ("think_on", "thinking ligado", "#eb6834", REF / "glm_votes_think_on.csv")]
HUMAN = "#a3a29c"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load():
    raw, memes, train = load_json(SOFT_RAW), load_memes(), load_json(TRAIN_IDS)
    votes = {}
    for arm, _, _, path in ARMS:
        with open(path, encoding="utf-8") as f:
            votes[arm] = {r["meme_id"]: r for r in csv.DictReader(f)}
    ids = [i for i in train if all(glm_soft([votes[a][i][f"label_p{p}"] for p in (1, 2, 3)]) is not None
                                   for a in votes)]
    return raw, memes, votes, ids


def labels(r):
    return [r[f"label_p{p}"] for p in (1, 2, 3)]


def bars(ax, groups, series, fmt="{:.2f}", ylim=1.0):
    """series = [(rótulo, cor, [valor por grupo])]; barras finas, 2 px de folga entre elas."""
    x, w = np.arange(len(groups)), 0.8 / len(series)
    for k, (label, color, vals) in enumerate(series):
        pos = x - 0.4 + w * (k + 0.5)
        ax.bar(pos, vals, w * 0.92, color=color, label=label, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + ylim * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ylim)


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)


def main():
    raw, memes, votes, ids = load()
    h = {i: raw[i][0] for i in ids}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), facecolor="#fcfcfb")
    for a in ax.flat:
        a.set_facecolor("#fcfcfb")

    # 1. Taxa de YES por idioma
    scopes = [("total", ids), ("en", [i for i in ids if memes[i]["lang"] == "en"]),
              ("es", [i for i in ids if memes[i]["lang"] == "es"])]
    yes = lambda arm, s: np.mean([l == "YES" for i in s for l in labels(votes[arm][i])])
    bars(ax[0, 0], [n for n, _ in scopes],
         [("humanos (média de p_YES)", HUMAN, [np.mean([h[i] for i in s]) for _, s in scopes])]
         + [(lab, c, [yes(arm, s) for _, s in scopes]) for arm, lab, c, _ in ARMS])
    style(ax[0, 0], "1 · Taxa de YES: o thinking reduz, mas fica acima dos humanos", "fração de YES")

    # 2. Onde o GLM erra em relação à maioria humana (empates 3-3 fora)
    no_h, yes_h = [i for i in ids if h[i] < 0.5], [i for i in ids if h[i] > 0.5]
    vote = lambda arm, i: votes[arm][i]["glm_vote"]
    def metrics(arm):
        g = {i: glm_soft(labels(votes[arm][i])) for i in ids}
        return [np.mean([vote(arm, i) == "YES" for i in no_h]),
                np.mean([vote(arm, i) == "NO" for i in yes_h]),
                np.mean([(vote(arm, i) == "YES") == (h[i] > 0.5) for i in no_h + yes_h]),
                pd.Series([h[i] for i in ids]).corr(pd.Series([g[i] for i in ids]), method="spearman")]
    bars(ax[0, 1], [f"YES quando\nhumanos dizem NO\n(n={len(no_h)})",
                    f"NO quando\nhumanos dizem YES\n(n={len(yes_h)})",
                    "acerto vs\nmaioria humana", "Spearman\n(p_GLM × p_humanos)"],
         [(lab, c, metrics(arm)) for arm, lab, c, _ in ARMS])
    style(ax[0, 1], "2 · Erros: o thinking corta o falso YES; a ordenação não muda", "fração / correlação")

    # 3. Distribuição de p_GLM (YES nas 3 paráfrases)
    levels = [0, 1 / 3, 2 / 3, 1]
    dist = lambda arm: [sum(glm_soft(labels(votes[arm][i])) == k for i in ids) for k in levels]
    bars(ax[1, 0], ["0 de 3", "1 de 3", "2 de 3", "3 de 3"],
         [(lab, c, dist(arm)) for arm, lab, c, _ in ARMS], fmt="{:.0f}", ylim=2500)
    style(ax[1, 0], "3 · Quantas das 3 paráfrases dizem YES, por meme", "nº de memes")

    # 4. Votos que mudaram (off → on), por maioria humana
    groups = [("maioria NO", no_h), ("empate 3-3", [i for i in ids if h[i] == 0.5]), ("maioria YES", yes_h)]
    flip = lambda s, a, b: sum(vote("think_off", i) == a and vote("think_on", i) == b for i in s)
    bars(ax[1, 1], [f"{n}\n(n={len(s)})" for n, s in groups],
         [("YES → NO", "#1baf7a", [flip(s, "YES", "NO") for _, s in groups]),
          ("NO → YES", "#6250d6", [flip(s, "NO", "YES") for _, s in groups])], fmt="{:.0f}", ylim=350)
    style(ax[1, 1], "4 · Votos que mudam ao ligar o thinking (maioria das 3 paráfrases)", "nº de memes")

    ax[1, 1].legend(frameon=False, fontsize=9, loc="upper right", labelcolor=MUTED)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper left", ncol=3, frameon=False,
               fontsize=10, labelcolor=MUTED, bbox_to_anchor=(0.005, 0.965))
    fig.suptitle(f"GLM-4.6V-Flash com thinking desligado vs ligado · {len(ids)} memes de treino · "
                 "P1–P3, seed 42", fontsize=13, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Maioria humana = p_YES dos 6 anotadores > 0,5 (YES) ou < 0,5 (NO). "
             "Spearman = correlação de ranks entre a fração de YES do GLM e a dos humanos.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.935))
    for ext in ("png", "pdf"):
        fig.savefig(REPORTS / f"doc30_think_on_vs_off.{ext}", dpi=200, facecolor=fig.get_facecolor())
    print(f"gravado: {REPORTS / 'doc30_think_on_vs_off.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
