"""Figura do doc 33: o desvio de cada anotador com o GLM e com a Persona como referência, e a faixa nula (k).
Só lê arquivos gravados. Os asserts conferem ρ(a) e ρ(b), recalculados dos CSV de pesos, contra
reports/persona33_checks.md (Persona) e reports/checks_summary.md (GLM, doc 27).

Uso (no Mac, da raiz do repositório): python3 EXIST_MEMES/glm_annotator/plot_persona33.py
Grava glm_reference/reports/persona33_pesos.png e .pdf.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd

from common import REF, REPORTS
from persona_ref import WEIGHTS
from plot_doc30 import GRID, INK, MUTED

# mesmas cores e marcadores dos braços em resultados_*.png (paleta validada)
GLM_C, PER_C, BAND_C = "#2a78d6", "#1a9a6e", "#d9d7d0"
STATS = [("rho_a", "ρ(a): r × r dos pares"), ("rho_b", "ρ(b): D × taxa de YES"),
         ("rho_gap", "ρ(gap): D × distância de limiar"), ("share_moved", "memes que mudam ≥ meio voto")]
NUM = r"(-?\d+\.\d+)"

checks = (REPORTS / "persona33_checks.md").read_text(encoding="utf-8").split("## (k)")[1]
obs_p = {k: float(v) for k, v in re.findall(rf"\| (\w+) \| {NUM} \| \[", checks)}
band_p = {k: (float(a), float(b)) for k, _, a, b in re.findall(rf"\| (\w+) \| {NUM} \| \[{NUM}, {NUM}\]", checks)}
band_g = {k: (float(a), float(b)) for k, a, b in re.findall(rf"(\w+) \[{NUM}, {NUM}\]", checks.split("Contexto")[1])}
summ = (REPORTS / "checks_summary.md").read_text(encoding="utf-8")       # doc 27
obs_g = {"rho_a": float(re.search(rf"r_peer\) = \*\*{NUM}", summ).group(1)),
         "rho_b": float(re.search(rf"todos os seus memes de treino\) = \*\*{NUM}", summ).group(1)),
         "rho_gap": float(re.search(rf"nos memes incluídos\) = \*\*{NUM}", summ).group(1)),
         "share_moved": float(re.search(rf"\| total \| {NUM} \|", summ).group(1))}
assert set(band_p) == set(band_g) == set(obs_p) == {k for k, _ in STATS}, (band_p, band_g, obs_p)

glm = pd.read_csv(REF / "annotator_weights.csv")
per = pd.read_csv(WEIGHTS)
assert len(glm) == len(per) == 887 and glm.D_glm.notna().all() and per.D_persona.notna().all()
for df, r, D, obs in ((glm, "r_glm", "D_glm", obs_g), (per, "r_persona", "D_persona", obs_p)):
    assert abs(df[r].corr(df.r_peer, method="spearman") - obs["rho_a"]) < 5e-4, r
    assert abs(df[D].corr(df.yes_rate, method="spearman") - obs["rho_b"]) < 5e-4, D

plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
comma = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−"))


def clean(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    ax.grid(color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


fig = plt.figure(figsize=(9.6, 8.4))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.42, wspace=0.22)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
for ax, df, D, col, mk, name in ((axes[0], glm, "D_glm", GLM_C, "o", "GLM (doc 27)"),
                                 (axes[1], per, "D_persona", PER_C, "^", "Persona (doc 33)")):
    rho = df[D].corr(df.yes_rate, method="spearman")
    ax.scatter(df.yes_rate, df[D], s=11, marker=mk, color=col, alpha=0.5, linewidths=0, zorder=3)
    clean(ax, f"{name}\nρ(D, taxa de YES) = {rho:+.2f}".replace(".", ",").replace("-", "−"))
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))     # marcadores em 0 e 1 inteiros
    ax.xaxis.set_major_formatter(comma)
    ax.yaxis.set_major_formatter(comma)
    ax.set_xlabel("taxa de YES do anotador (todos os seus memes)", fontsize=8.5)
    ax.set_aspect("equal")
axes[0].set_ylabel("desvio D do anotador (0 = sempre com a referência)", fontsize=8.5)

ax = axes[2]
for y, (k, _) in enumerate(STATS):
    for dy, b, o, col, mk in ((0.17, band_g, obs_g, GLM_C, "o"), (-0.17, band_p, obs_p, PER_C, "^")):
        ax.hlines(y + dy, *b[k], color=BAND_C, lw=7, zorder=2)
        ax.scatter(o[k], y + dy, marker=mk, s=60, color=col, edgecolor="white", linewidth=0.8, zorder=4)
ax.axvline(0, color=INK, lw=0.8, zorder=1)
ax.set_yticks(range(len(STATS)), [lab for _, lab in STATS], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-0.85, 0.85)
ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
ax.xaxis.set_major_formatter(comma)
clean(ax, "Observado × faixa nula (200 permutações das triplas de respostas entre memes)")
ax.grid(axis="y", visible=False)
ax.set_xlabel("valor (ρ de Spearman ou fração)", fontsize=8.5)
ax.legend(handles=[Line2D([], [], color=GLM_C, marker="o", ls="none", ms=7, label="GLM observado"),
                   Line2D([], [], color=PER_C, marker="^", ls="none", ms=7, label="Persona observado"),
                   Line2D([], [], color=BAND_C, lw=7, label="faixa nula p2,5–p97,5")],
          frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, labelcolor=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(REPORTS / f"persona33_pesos.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
print("gravado", REPORTS / "persona33_pesos.png")
print({k: (obs_g[k], band_g[k], obs_p[k], band_p[k]) for k, _ in STATS})
