"""Figuras do doc 31: GLM (thinking desligado/ligado) vs Claude Sonnet 5.5 nos 1 551 memes extremos.

Uso (da raiz do repositório, depois de claude_reference.py):
    python3 EXIST_MEMES/glm_annotator/plot_doc31.py
Grava em glm_reference/reports/: doc31_voto_yes.png, doc31_parafrases.png, doc31_acerto.png,
doc31_cruzamento.png (+ .pdf de cada).
"""
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from client import majority
from common import MEMES_DIR, REF, REPORTS
from plot_doc30 import GRID, HUMAN, INK, MUTED, style

CLAUDE_RAW = MEMES_DIR / "claude_reference/raw/claude-sonnet-5-5_think_off.jsonl"
# (chave, rótulo, cor, hachura): a hachura no Claude dá uma segunda codificação além da cor
ARMS = [("think_off", "GLM · thinking desligado", "#2a78d6", ""),
        ("think_on", "GLM · thinking ligado", "#eb6834", ""),
        ("claude", "Claude Sonnet 5.5 · thinking desligado", "#0f8f6a", "//")]
GROUPS = [("1Y5N", 1 / 6), ("3Y3N", 0.5), ("5Y1N", 5 / 6)]
BG = "#fcfcfb"


def load():
    rows = list(csv.DictReader(open(REF / "extremes/extremes.csv", encoding="utf-8")))
    cl = {}
    for line in open(CLAUDE_RAW, encoding="utf-8"):
        r = json.loads(line)
        cl[(r["meme_id"], r["paraphrase"])] = r["label"]          # o último registro vale (reenvio PNG)
    for r in rows:
        r["think_off"] = [r[f"think_off_p{p}"] for p in (1, 2, 3)]
        r["think_on"] = [r[f"think_on_p{p}"] for p in (1, 2, 3)]
        r["claude"] = [cl[(r["meme_id"], p)] for p in (1, 2, 3)]
    return rows


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def grouped(ax, cats, vals, errs=None, fmt="{:.3f}", ylim=1.0):
    """vals[arm] = [valor por categoria]; errs[arm] = [(lo, hi)] opcional."""
    x, w = np.arange(len(cats)), 0.8 / len(ARMS)
    for k, (arm, label, color, hatch) in enumerate(ARMS):
        pos = x - 0.4 + w * (k + 0.5)
        ax.bar(pos, vals[arm], w * 0.92, color=color, hatch=hatch, edgecolor="white", linewidth=0,
               label=label, zorder=3)
        if errs:
            lo = [v - e[0] for v, e in zip(vals[arm], errs[arm])]
            hi = [e[1] - v for v, e in zip(vals[arm], errs[arm])]
            ax.errorbar(pos, vals[arm], yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        for p, v, e in zip(pos, vals[arm], errs[arm] if errs else [None] * len(cats)):
            top = e[1] if e else v
            ax.text(p, top + ylim * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, ylim)


def fig_voto_yes(rows):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    vals, errs = {}, {}
    for arm, *_ in ARMS:
        vals[arm], errs[arm] = [], []
        for g, _ in GROUPS:
            s = [r for r in rows if r["group"] == g]
            k = sum(majority(r[arm]) == "YES" for r in s)
            vals[arm].append(k / len(s))
            errs[arm].append(wilson(k, len(s)))
    grouped(ax, [f"{g}\n(n = {sum(r['group'] == g for r in rows)})" for g, _ in GROUPS], vals, errs, ylim=1.1)
    for i, (_, h) in enumerate(GROUPS):
        ax.hlines(h, i - 0.45, i + 0.45, colors=HUMAN, linestyles="--", linewidth=2, zorder=5,
                  label="humanos (fração de YES dos 6)" if i == 0 else None)
    style(ax, "Voto YES (maioria das 3 paráfrases) nos memes extremos · barras = IC 95% de Wilson",
          "fração de memes com voto YES")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=MUTED)
    save(fig, "doc31_voto_yes")


def fig_parafrases(rows):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
    per_p = {arm: [np.mean([r[arm][i] == "YES" for r in rows]) for i in range(3)] for arm, *_ in ARMS}
    grouped(ax[0], ["P1", "P2", "P3"], per_p)
    style(ax[0], f"1 · Fração de YES por paráfrase ({len(rows)} memes extremos)", "fração de YES")
    agree = {arm: [np.mean([len(set(r[arm])) == 1 and r[arm][0] != "?" for r in rows if r["group"] == g])
                   for g, _ in GROUPS]
             for arm, *_ in ARMS}
    grouped(ax[1], [g for g, _ in GROUPS], agree)
    style(ax[1], "2 · Memes em que as 3 paráfrases concordam (3-0)", "fração de memes")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper left", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=MUTED, bbox_to_anchor=(0.005, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "doc31_parafrases")


def fig_acerto(rows):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    a = [r for r in rows if r["group"] == "1Y5N"]
    b = [r for r in rows if r["group"] == "5Y1N"]
    vals = {}
    for arm, *_ in ARMS:
        spec = np.mean([majority(r[arm]) == "NO" for r in a])
        sens = np.mean([majority(r[arm]) == "YES" for r in b])
        vals[arm] = [spec, sens, (spec + sens) / 2]
    grouped(ax, [f"NO quando 5 de 6\nhumanos dizem NO\n(1Y5N, n = {len(a)})",
                 f"YES quando 5 de 6\nhumanos dizem YES\n(5Y1N, n = {len(b)})",
                 "média das duas\n(acurácia balanceada)"], vals, ylim=1.1)
    style(ax, "Acordo com a maioria humana nos grupos 1Y5N e 5Y1N (descritivo, pós-hoc)", "fração de memes")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=MUTED)
    save(fig, "doc31_acerto")


def fig_cruzamento(rows):
    combos = [("YES", "YES", "#9a9993"), ("YES", "NO", "#2a78d6"), ("NO", "YES", "#0f8f6a"), ("NO", "NO", "#d9d8d3")]
    names = {("YES", "YES"): "os dois YES", ("YES", "NO"): "só o GLM diz YES",
             ("NO", "YES"): "só o Claude diz YES", ("NO", "NO"): "os dois NO"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), facecolor=BG, sharey=True)
    for a, glm, title in ((ax[0], "think_off", "GLM thinking desligado × Claude"),
                          (ax[1], "think_on", "GLM thinking ligado × Claude")):
        a.set_facecolor(BG)
        for y, (g, _) in enumerate(GROUPS):
            s = [r for r in rows if r["group"] == g and majority(r[glm]) and majority(r["claude"])]
            left = 0
            for gv, cv, color in combos:
                n = sum(majority(r[glm]) == gv and majority(r["claude"]) == cv for r in s)
                a.barh(y, n / len(s), left=left, color=color, height=0.6, edgecolor=BG, linewidth=2,
                       label=names[(gv, cv)] if y == 0 else None, zorder=3)
                if n / len(s) > 0.04:
                    a.text(left + n / len(s) / 2, y, str(n), ha="center", va="center", fontsize=8.5,
                           color="white" if color in ("#2a78d6", "#0f8f6a", "#9a9993") else INK)
                left += n / len(s)
        a.set_yticks(range(len(GROUPS)), [g for g, _ in GROUPS])
        a.invert_yaxis()
        a.set_xlim(0, 1)
        style(a, title, "")
        a.grid(axis="x", color=GRID, zorder=0)
        a.grid(axis="y", visible=False)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper left", ncol=4, frameon=False, fontsize=9.5,
               labelcolor=MUTED, bbox_to_anchor=(0.005, 0.99))
    fig.text(0.01, 0.01, "Voto = maioria das 3 paráfrases; números = memes; largura = fração do grupo. "
             "1 meme do 5Y1N sem voto do GLM (imagem recusada) fica fora.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(fig, "doc31_cruzamento")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(REPORTS / f"{name}.{ext}", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"gravado: {REPORTS / name}.png (+ .pdf)")


if __name__ == "__main__":
    rows = load()
    assert len(rows) == 1551 and all(len(r["claude"]) == 3 for r in rows)
    fig_voto_yes(rows)
    fig_parafrases(rows)
    fig_acerto(rows)
    fig_cruzamento(rows)
