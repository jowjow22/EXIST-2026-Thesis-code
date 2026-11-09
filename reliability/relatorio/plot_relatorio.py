"""Figuras do relatório de confiabilidade (reliability/relatorio/figs/).

Uso (da raiz do repositório):
    python3 reliability/relatorio/plot_relatorio.py
Lê só artefactos já gravados: reliability/outputs/*, EXIST_MEMES/metrics_history/*,
o dataset de treino. Não treina nem regenera nada. O diagrama de fluxo sai de
figs/fluxo_reliability.dot (graphviz: dot -Tpng -Gdpi=200).
"""
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
OUT = ROOT / "reliability/outputs"
MH = ROOT / "EXIST_MEMES/metrics_history"
DATA = ROOT / "EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json"
FIGS = pathlib.Path(__file__).resolve().parent / "figs"
VER = pathlib.Path(__file__).resolve().parent / "verificacao"
FIGS.mkdir(exist_ok=True)

INK, MUTED, GRID, BG = "#1a1a18", "#6b6962", "#e4e2dc", "#fcfcfb"
BLUE, ORANGE, GREEN, GRAY, RED = "#2a78d6", "#eb6834", "#0f8f6a", "#9a9993", "#c0392b"
PRIM = "eval_task_a_icmsoftnorm_soft"
A0 = ["20260831_065521_raw_seed42", "20260831_075933_raw_seed7", "20260830_225133_raw_seed2025"]
A1 = ["20260830_205756_weighted_fixed_seed42", "20260830_215815_weighted_fixed_seed7", "20260830_194907_weighted_fixed_seed2025"]
R1 = ["20260905_112128_R1_rem_extremos_seed42", "20260905_122152_R1_rem_extremos_seed7", "20260905_140756_R1_rem_extremos_seed2025"]
R2 = ["20260905_150442_R2_rem_controle_seed42", "20260905_155802_R2_rem_controle_seed7", "20260905_222414_R2_rem_controle_seed2025"]
CLR = ["20260907_234357_conf_M1_lam0.1_clr0.01_seed42", "20260908_192123_conf_M1_lam0.1_clr0.01_seed7", "20260909_225545_conf_M1_lam0.1_clr0.01_seed2025"]
SEEDS = [42, 7, 2025]
MARK = {42: "o", 7: "s", 2025: "^"}


def style(ax, title, ylabel, xlabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("gravado", FIGS / f"{name}.png")


def evals(fn):
    h = json.load(open(MH / f"{fn}.json"))
    return [r for r in h if "eval_loss" in r]


def at_E(fn, key=PRIM):
    ev = evals(fn)
    return ev[int(np.argmin([r["eval_loss"] for r in ev]))][key]


# ---------------------------------------------------------------- F1: grade de recuperação
def fig_recuperacao():
    """Grade de recuperação: estimador legado (arquivo de 2026-08-26, linha cheia) e corrigido
    (reexecução de 2026-10-24 em verificacao/gates_fixed, linha tracejada)."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=BG, sharey=True)
    for ax, fn, tit in ((axs[0], "recovery_grid.csv", "Densidade confortável (80 votos/anotador)"),
                        (axs[1], "recovery_grid_existdensity.csv", "Densidade do EXIST (26,9 votos/anotador)")):
        for src, ls, alpha, tag in ((OUT / fn, "-", 1.0, "legado"), (VER / "gates_fixed" / fn, "--", 0.9, "corrigido")):
            m = pd.read_csv(src).groupby(["divisiveness", "estimator"]).pearson_r.mean().unstack()
            for est, col, mk, lab in (("subpop_aware", BLUE, "o", "NUTMEG (K = 2)"), ("single_truth", ORANGE, "s", "verdade única (K = 1)")):
                ax.plot(m.index, m[est], color=col, marker=mk, lw=2, ms=5.5, ls=ls, alpha=alpha,
                        mfc=col if tag == "legado" else "white", label=f"{lab} · {tag}", zorder=3)
                if est == "subpop_aware":
                    x = m.index[-1]
                    ax.annotate(f"{m[est].iloc[-1]:.2f}", (x, m[est].iloc[-1]), textcoords="offset points", xytext=(8, 0),
                                va="center", fontsize=7.5, color=col)
        style(ax, tit, "Pearson r (θ̂, θ verdadeiro)" if ax is axs[0] else "", "divisividade (fracção de itens com verdades diferentes)")
        ax.set_ylim(0.2, 1.0)
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", labelcolor=MUTED, ncol=4,
               bbox_to_anchor=(0.5, -0.06))
    fig.text(0.01, -0.10, "Média de 3 seeds × 3 taxas de spam. Cheio = estimador legado (arquivo de 2026-08-26); tracejado = estimador corrigido "
             "(reexecução de 2026-10-24). Fontes: reliability/outputs/recovery_grid*.csv; reliability/relatorio/verificacao/gates_fixed/",
             fontsize=7.5, color=MUTED)
    save(fig, "f1_recuperacao")


def fig_sweep_nutmeg():
    """Stress-sweep do spammer de viés com subpopulação própria: legado vs corrigido, por seed."""
    fig, ax = plt.subplots(figsize=(9, 4.0), facecolor=BG)
    for tag, col, mk in (("legacy", ORANGE, "s"), ("fixed", BLUE, "o")):
        sw = json.load(open(VER / f"gates_{tag}" / "sweep_bias.json"))
        fr = [r["biased_frac"] for r in sw]
        ax.plot(fr, [r["theta_biased_mean"] for r in sw], color=col, marker=mk, lw=2.2, ms=6, zorder=4,
                label={"legacy": "estimador legado (2026-08-26)", "fixed": "estimador corrigido (2026-10-24)"}[tag])
        for r in sw:
            ax.scatter([r["biased_frac"]] * 3, r["theta_biased_per_seed"], color=col, s=14, alpha=0.45, zorder=3)
        for x, r in zip(fr, sw):
            if x == 0.05:
                continue
            ax.annotate(f"{r['theta_biased_mean']:.3f}", (x, r["theta_biased_mean"]), textcoords="offset points",
                        xytext=(0, 8 if tag == "legacy" else -14), ha="center", fontsize=7.5, color=col)
    ax.axhline(0.60, color=RED, ls="--", lw=1.2)
    ax.text(0.5, 0.615, "teto pré-registado (0,60)", ha="right", fontsize=7.5, color=RED)
    ax.set_ylim(0, 1.05)
    style(ax, "NUTMEG, spammers de viés constante com subpopulação própria: θ̂ médio por fracção do painel",
          "θ̂ dos spammers enviesados (θ real = 0,25)", "fracção de spammers enviesados")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=MUTED)
    fig.text(0.01, -0.03, "Linha = média de 3 seeds; pontos = cada seed. Mesmo código e critério de validate.py; só muda a responsabilidade do M-step (n_iter 200 ou 5000: idêntico). "
             "Fonte: reliability/relatorio/verificacao/gates_*/sweep_bias.json", fontsize=7.5, color=MUTED)
    save(fig, "f1_sweep_bloco")


# ---------------------------------------------------------------- F3: sweep do bloco
def fig_perann_sweep():
    pg = json.load(open(OUT / "perann_synthetic_gate.json"))
    sw = pd.DataFrame(pg["density_sweep"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=BG, sharey=True)
    for ax, arm, tit in ((axs[0], "easy", "Minoria que discorda em terreno de consenso\n(é ruído: deve ser punida)"),
                         (axs[1], "bias", "Spammer de viés constante, sempre YES\n(deve ser punido)")):
        for cfg, col, mk, lab in (("mace_naive", GRAY, "o", "mace_naive (sem desconto)"),
                                  ("discount_raw", ORANGE, "s", "discount_raw (literal)"),
                                  ("discount_loo", BLUE, "^", "discount_loo")):
            ax.plot(sw.frac, sw[f"{arm}_gap_{cfg}"], color=col, marker=mk, lw=2, ms=6, label=lab, zorder=3)
        ax.axhline(0.20, color=RED, ls="--", lw=1.2, zorder=2)
        ax.text(sw.frac.max(), 0.205, "gap mínimo pré-registado (0,20)", ha="right", va="bottom", fontsize=7.5, color=RED)
        ax.axhline(0, color=INK, lw=0.8)
        style(ax, tit, "gap = competência fiáveis − bloco" if ax is axs[0] else "", "fracção do painel no bloco")
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=MUTED)
    fig.text(0.01, -0.02, "Densidade confortável (80 votos/anotador), hard_frac = 0,4, média de 3 seeds. Fonte: reliability/outputs/perann_synthetic_gate.json (density_sweep)",
             fontsize=7.5, color=MUTED)
    save(fig, "f3_sweep_bloco")


# ---------------------------------------------------------------- F4: bloco no IRT (números do notebook, reproduzidos)
def fig_irt_bloco():
    ks = ["k = 20 (9%)", "k = 60 (27%)", "k = 120 (55%)"]
    vals = {"δ = 0 (controlo)": [-0.0989, 0.0393, -0.0200], "δ = 1": [-0.9541, -0.8547, -0.9216], "δ = 2": [-1.7885, -1.6525, -1.7454]}
    sds = {"δ = 0 (controlo)": [0.0425, 0.0490, 0.0776], "δ = 1": [0.0387, 0.0413, 0.1451], "δ = 2": [0.0820, 0.0809, 0.1544]}
    fig, ax = plt.subplots(figsize=(8.5, 4.0), facecolor=BG)
    x, w = np.arange(3), 0.26
    for j, (lab, col) in enumerate(zip(vals, (GRAY, BLUE, ORANGE))):
        pos = x + (j - 1) * w
        ax.bar(pos, vals[lab], w * 0.92, color=col, label=lab, zorder=3)
        ax.errorbar(pos, vals[lab], yerr=sds[lab], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        for p, v, e in zip(pos, vals[lab], sds[lab]):
            ax.text(p, v - e - 0.04 if v < 0 else v + e + 0.03, f"{v:+.2f}", ha="center", va="top" if v < 0 else "bottom", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, ks)
    style(ax, "IRT 1PL: gap de competência estimada (bloco − maioria), competência verdadeira igual", "gap médio (3 seeds; barra = sd)")
    ax.set_ylim(-2.1, 0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=MUTED)
    fig.text(0.01, -0.03, "k = 5 excluído: controlo falhou (|gap| = 0,449) e o seguimento de 30 seeds dá IC95 [−0,030; +0,283]. "
             "Fonte: EXIST_MEMES/dataset_verifications.ipynb, célula 9", fontsize=7.5, color=MUTED)
    save(fig, "f4_irt_bloco")


# ---------------------------------------------------------------- retreinos: Δ por seed
def fig_deltas():
    comps = [("F2 · A1 − A0\n(θ corrigido, 15 ép.)", A0, A1, True),
             ("F5 · R1 − R2\n(decisiva)", R2, R1, True),
             ("F5 · R1 − A0\n(contexto)", A0, R1, False),
             ("F5 · R2 − A0\n(contexto)", A0, R2, False),
             ("F6 · M1 clr = 1e-2 − A0\n(exploratório)", A0, CLR, False)]
    sd_a0 = float(np.std([at_E(f) for f in A0], ddof=1))
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=BG)
    ax.axhspan(-sd_a0, sd_a0, color=GRID, alpha=0.6, zorder=1)
    ax.text(len(comps) - 0.55, sd_a0 + 0.0004, f"±1 sd do A0 entre seeds ({sd_a0:.4f})", fontsize=7.5, color=MUTED, ha="right")
    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    for i, (lab, a, b, decisive) in enumerate(comps):
        d = [at_E(fb) - at_E(fa) for fa, fb in zip(a, b)]
        for s, v in zip(SEEDS, d):
            ax.scatter(i + {42: -0.12, 7: 0, 2025: 0.12}[s], v, marker=MARK[s], s=46, color=BLUE if decisive else GRAY,
                       edgecolor="white", linewidth=0.8, zorder=4)
        ax.hlines(np.mean(d), i - 0.25, i + 0.25, color=ORANGE, lw=2.4, zorder=5)
        ax.text(i + 0.28, np.mean(d), f"{np.mean(d):+.4f}", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(comps)), [c[0] for c in comps], fontsize=8.5)
    style(ax, "Δ da métrica decisiva (Task A ICM-Soft-Norm) na época E = argmin(eval_loss), por seed", "Δ ICM-Soft-Norm (A)")
    ax.set_ylim(-0.0195, 0.0125)
    h = [Line2D([], [], color=BLUE, marker="o", ls="none", ms=7, label="comparação decisiva (pré-registada)"),
         Line2D([], [], color=GRAY, marker="o", ls="none", ms=7, label="contexto ou exploratória"),
         Line2D([], [], color=ORANGE, lw=2.4, label="média das 3 seeds")]
    h += [Line2D([], [], color=MUTED, marker=MARK[s], ls="none", ms=6, label=f"seed {s}") for s in SEEDS]
    ax.legend(handles=h, frameon=False, fontsize=8, loc="upper left", labelcolor=MUTED, ncol=3)
    fig.text(0.01, -0.03, "Cor = estatuto da comparação; marcador = seed. Fonte: EXIST_MEMES/metrics_history/",
             fontsize=7.5, color=MUTED)
    save(fig, "retreinos_deltas")


# ---------------------------------------------------------------- curvas de eval_loss
def legend_arms_seeds(arms):
    """Legenda em duas partes: cor = braço (linha sem marcador), marcador = seed (cinzento).
    Cada braço tem 3 linhas (uma por seed); sem isto a legenda mostraria só a da seed 42."""
    h = [Line2D([], [], color=c, lw=2.2, label=l) for c, l in arms]
    h += [Line2D([], [], color=GRAY, marker=MARK[s], ls="none", ms=6, label=f"seed {s}") for s in SEEDS]
    return h


def fig_curvas():
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=BG)
    arms = ((A0, INK, "A0 (raw)"), (A1, BLUE, "A1 (ponderado por θ)"), (R1, ORANGE, "R1 (sem extremos)"), (R2, GREEN, "R2 (controlo)"))
    for files, col, lab in arms:
        for k, f in enumerate(files):
            ev = evals(f)
            ax.plot([r["epoch"] for r in ev], [r["eval_loss"] for r in ev], color=col, lw=1.4, alpha=0.85,
                    marker=MARK[SEEDS[k]], ms=3.5, zorder=3)
    ax.axvline(3, color=RED, ls="--", lw=1)
    ax.text(3.1, ax.get_ylim()[1], "época 3 = E em 11 das 12 runs", fontsize=7.5, color=RED, va="top")
    style(ax, "eval_loss por época: 4 braços × 3 seeds = 12 runs", "eval_loss", "época")
    fig.legend(handles=legend_arms_seeds([(c, l) for _, c, l in arms]), frameon=False, fontsize=8.5, loc="lower center",
               labelcolor=MUTED, ncol=7, bbox_to_anchor=(0.5, -0.07), handlelength=1.8, columnspacing=1.2)
    save(fig, "retreinos_eval_loss")


# ---------------------------------------------------------------- ablação: densidade do alvo
def controle42(perfil):
    """Sorteio do controlo R2, cópia do `sortear_controle(42)` de EXIST_MEMES/reliability_tests.ipynb (célula 8).
    verificacao/verify_part2.py confirma que reproduz soft_labels_taskA_R2_rem_controle.json exactamente."""
    rng = np.random.default_rng(42)
    alvo = perfil.loc[perfil["extremo"], ["lang", "n_votes"]]
    pool = perfil.loc[~perfil["extremo"]]
    esc = []
    for (lg, nv), grupo in alvo.groupby(["lang", "n_votes"]):
        cand = pool.index[(pool["lang"] == lg) & (pool["n_votes"] == nv) & (~pool.index.isin(esc))].to_numpy()
        if len(cand) < len(grupo):
            cand = pool.index[(pool["lang"] == lg) & pool["n_votes"].between(nv - 3, nv + 3) & (~pool.index.isin(esc))].to_numpy()
        esc.extend(rng.choice(cand, size=len(grupo), replace=False))
    return pd.Index(esc)


def fig_ablacao_densidade():
    data = json.load(open(DATA, encoding="utf-8"))
    ds = pd.DataFrame(list(data.values()))
    v = ds[["id_EXIST", "lang", "annotators", "labels_task2_1"]].explode(["annotators", "labels_task2_1"])
    v["yes"] = (v.labels_task2_1 == "YES").astype(int)
    perfil = v.groupby("annotators").agg(n_votes=("yes", "size"), yes_rate=("yes", "mean"))
    perfil["extremo"] = (perfil.yes_rate > 0.90) | (perfil.yes_rate < 0.10)
    perfil["lang"] = v.groupby("annotators")["lang"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "ambas")
    ext = perfil.index[perfil.extremo]
    ctrl = controle42(perfil)
    r2 = json.load(open(OUT / "soft_labels_taskA_R2_rem_controle.json"))
    lab2 = v[~v.annotators.isin(ctrl)].groupby("id_EXIST").yes.mean()
    assert max(abs(r2[k][0] - lab2[k]) for k in lab2.index) == 0, "sorteio do controlo não reproduz o ficheiro"
    n1 = v[~v.annotators.isin(ext)].groupby("id_EXIST").size()
    n2 = v[~v.annotators.isin(ctrl)].groupby("id_EXIST").size()
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG)
    ks = [3, 4, 5, 6]
    x, w = np.arange(4), 0.38
    for j, (n, col, lab) in enumerate(((n1, ORANGE, "R1 (sem os 61 extremos)"), (n2, GREEN, "R2 (sem 61 sorteados)"))):
        c = [int((n == k).sum()) for k in ks]
        ax.bar(x + (j - 0.5) * w, c, w * 0.92, color=col, label=lab, zorder=3, hatch="//" if j else "", edgecolor="white")
        for xx, cc in zip(x + (j - 0.5) * w, c):
            ax.text(xx, cc + 30, str(cc), ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{k} votos" for k in ks])
    style(ax, "Votos por item no alvo de treino da Task A após a remoção (1 597 votos removidos em cada braço)", "nº de itens")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=MUTED)
    save(fig, "f5_densidade")


# ---------------------------------------------------------------- F6 fase 2: trajectória da régua
def fig_regua():
    d = json.load(open(MH / "20260911_011116_conf_M1sd_lam0.1_clr0.01_seed2025_votes.json"))
    recs = [r for r in d["records"] if float(r["epoch"]).is_integer()][:15]
    ep = [r["epoch"] for r in recs]
    m = lambda r, k: r["metrics"][k]["nll"]
    fig, ax = plt.subplots(figsize=(9.5, 4.0), facecolor=BG)
    ax.plot(ep, [m(r, "b") - m(r, "b_perm") for r in recs], color=BLUE, marker="o", lw=2, label="b − b_perm (sinal individual)")
    ax.plot(ep, [m(r, "b") - m(r, "d4_train") for r in recs], color=ORANGE, marker="s", lw=2, label="b − d4_train (critério)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(3, color=RED, ls="--", lw=1)
    ax.text(3.15, ax.get_ylim()[1] * 0.9, "E = 3 (selector do sistema)", fontsize=7.5, color=RED)
    style(ax, "Fase 2, run A (M1sd, seed 2025): diferença de NLL nos 3 588 votos retidos (negativo = a ficha ganha)", "Δ NLL", "época")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=MUTED)
    fig.text(0.01, -0.03, "A trajectória é descritiva: o protocolo proíbe escolher a época pela régua. Fonte: *_M1sd_*_votes.json",
             fontsize=7.5, color=MUTED)
    save(fig, "f6_regua_votos")


DOT = r'''digraph G {
rankdir=TB; nodesep=0.25; ranksep=0.32;
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#9a9993", fillcolor="#ffffff"];
edge [color="#6b6962", fontname="Helvetica", fontsize=8.5];
F1 [label="F1 · NUTMEG (verdade por subpopulação)\nsintético 26/08: fenómeno reproduzido; bloco pequeno\n(2–3) capturava a célula — artefacto do bug do M-step\n(reexecução corrigida 24/10: não captura)"];
G1 [shape=diamond, label="portão\nsintético", fillcolor="#fbe3df"];
F2 [label="F2 · verdade única K = 1 (tipo MACE)\nbug do M-step corrigido 30/08\nretreino 15 ép.: Δ = −0,0025, p = 0,164", fillcolor="#fbe3df"];
F3 [label="F3 · por anotador + desconto de dificuldade\n27/08: FAIL (bloco grande fabrica a dificuldade)", fillcolor="#fbe3df"];
F4 [label="F4 · IRT 2PL → 1PL\n≈06/09: θ identificável, mas absorve posição;\nb ≈ logit(taxa de YES)", fillcolor="#fbe3df"];
F5 [label="F5 · remover 61 extremos vs 61 sorteados\n05–06/09: Δ R1 − R2 = −0,0077, p = 0,206", fillcolor="#fbe3df"];
F6 [label="F6 · camada de confusão (perda auxiliar)\n06–11/09: Δ = +0,0009, p = 0,30;\nrégua de votos: nulo; controlo positivo não analisado", fillcolor="#fbe3df"];
F7 [label="F7 · referência externa (LLM)\ndocs 25–31 (outro relatório)", style="rounded,dashed"];
F1 -> G1; G1 -> F2 [label=" só K = 1 vai a treino"]; G1 -> F3 [label=" eliminar a célula"];
F3 -> F4 [label=" dificuldade como parâmetro latente"];
F2 -> F5 [label=" ponderar falhou → remover"];
F4 -> F6 [label=" modelar o anotador no próprio treino"];
F5 -> F7 [style=dashed]; F6 -> F7 [style=dashed, label=" referência fora dos votos"];
}'''


def fig_fluxo():
    (FIGS / "fluxo_reliability.dot").write_text(DOT, encoding="utf-8")
    for ext in ("png", "pdf"):
        subprocess.run(["dot", f"-T{ext}", "-Gdpi=200", str(FIGS / "fluxo_reliability.dot"), "-o", str(FIGS / f"fluxo_reliability.{ext}")], check=True)
    print("gravado", FIGS / "fluxo_reliability.png")


if __name__ == "__main__":
    fig_fluxo()
    fig_recuperacao()
    fig_sweep_nutmeg()
    fig_perann_sweep()
    fig_irt_bloco()
    fig_deltas()
    fig_curvas()
    fig_ablacao_densidade()
    fig_regua()
