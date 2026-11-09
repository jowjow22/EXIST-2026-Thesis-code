"""Figura dos docs 27 (peso r) e 28 (peso r⁴) contra o BASE (glmbase), 3 seeds pareadas.

Linha 1: Δ = braço − glmbase no estado primário (época de menor eval_loss), por seed e média
com IC 95% (t, df = 2). Linha 2: curvas época a época, média das 3 seeds (faixa = mín–máx).

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/plot_doc27_28.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_f5 import discover
from common import REPORTS, load_json
from reliability.analyze_arm_experiment import SEEDS

PANELS = [("eval_task_a_icmsoftnorm_soft", "Task A · ICM-Soft-Norm (↑ melhor)"),
          ("eval_task_a_icmnorm_hard", "Task A · ICM-Norm hard (↑ melhor)"),
          ("eval_loss", "eval_loss (↓ melhor)")]
ARMS = [("glmbase", "BASE (labels originais)", "#555555"),
        ("glmw", "doc 27 · peso r", "#1f77b4"),
        ("glmw4", "doc 28 · peso r⁴", "#d62728")]
T_CRIT = 4.303   # t(0,975; df = 2)


def main():
    reports = {"glmw": load_json(REPORTS / "f5_report.json"),
               "glmw4": load_json(REPORTS / "f5_report_glmw4.json")}
    runs = {**discover("glmw"), **discover("glmw4")}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    for j, (key, title) in enumerate(PANELS):
        a = ax[0, j]
        for x, (arm, label, color) in enumerate(ARMS[1:]):
            r = reports[arm]["best"][key]
            d = [r["per_seed"][str(s)]["delta"] for s in SEEDS]
            ci = T_CRIT * r["sd"] / np.sqrt(r["n"])
            a.scatter([x] * 3, d, color=color, alpha=0.45, s=40, zorder=3)
            a.errorbar(x, r["mean"], yerr=ci, fmt="D", color=color, ms=9, capsize=8, lw=2, zorder=4)
            a.annotate(f"{r['mean']:+.4f}\np = {r['p']:.3f}", (x + 0.12, r["mean"]),
                       va="center", fontsize=9, color=color)
        a.axhline(0, color="black", lw=0.8, ls="--")
        a.set_xticks([0, 1], [ARMS[1][1], ARMS[2][1]])
        a.set_xlim(-0.5, 1.8)
        a.set_title(f"Δ vs BASE · {title}", fontsize=10)
        a.grid(axis="y", alpha=0.3)

        b = ax[1, j]
        for arm, label, color in ARMS:
            curves = np.array([[e[key] for e in runs[(arm, s)]["evals"]] for s in SEEDS])
            ep = np.arange(1, curves.shape[1] + 1)
            b.plot(ep, curves.mean(0), color=color, lw=2, label=label)
            b.fill_between(ep, curves.min(0), curves.max(0), color=color, alpha=0.12)
        b.axvline(3, color="black", lw=0.8, ls=":")
        b.set_xlabel("época")
        b.set_title(title, fontsize=10)
        b.grid(alpha=0.3)

    ax[0, 0].set_ylabel("Δ no estado primário\n(pontos = seeds; losango = média ± IC 95%)")
    ax[1, 0].set_ylabel("média das 3 seeds (faixa = mín–máx)")
    ax[1, 0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Ponderação de anotadores pelo GLM-4.6V-Flash: doc 27 (r) e doc 28 (r⁴) vs BASE · "
                 "seeds 42, 7, 2025 · validação com os labels originais", fontsize=12)
    fig.text(0.5, 0.005, "Linha pontilhada na época 3: época de menor eval_loss na maioria das runs "
             "(estado primário pré-registrado). t pareado bilateral, df = 2.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(REPORTS / f"doc27_28_resultados.{ext}", dpi=200)
    print(f"gravado: {REPORTS / 'doc27_28_resultados.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
