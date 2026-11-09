"""Relatório curto: resultados dos três métodos de ponderação (docs 27, 28 e 33) e como o peso é calculado.
Só lê arquivos gravados; nenhuma chamada a modelo. Todo número do texto sai dos arquivos; os asserts
conferem os pesos, os soft labels e as tabelas já registradas nos SPEC_28 e SPEC_33.

Uso (no Mac, da raiz do repositório; precisa de matplotlib e do Google Chrome):
    python3 EXIST_MEMES/glm_annotator/relatorio_resultados_pdf.py /tmp/relatorio_resultados.html
Grava as figuras resultados_*.png e o PDF relatorio_resultados.pdf em glm_reference/reports/;
o argumento é onde fica o HTML intermediário (fora do repositório).
"""
import csv
import pathlib
import subprocess
import sys
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from analyze_f5 import discover
from checks import hard
from common import (REF, REL_OUT, REPORTS, SOFT_GLMW, SOFT_RAW, T, TRAIN_IDS, VAL_IDS, VOTES_CSV, load_json,
                    load_memes)
from persona_ref import SOFT_PERSONAW, VOTES as PERSONA_VOTES, WEIGHTS as PERSONA_WEIGHTS
from plot_doc30 import GRID, INK, MUTED
from weights import (binary_entropy, deviation_from_terms, deviation_terms, load_glm_votes,
                     load_train_panels, n_yes, soft_votes, vote_weights, weighted_soft_label)

OUT_HTML = pathlib.Path(sys.argv[1])
PDF = REPORTS / "relatorio_resultados.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SEEDS = (42, 7, 2025)
# (braço, nome curto, doc, cor, marcador): paleta validada (validate_palette.py, light, all-pairs);
# o marcador é a segunda codificação além da cor
ARMS = [("glmw", "GLM · peso r", "doc 27", "#2a78d6", "o"),
        ("glmw4", "GLM · peso r⁴", "doc 28", "#e0791f", "s"),
        ("personaw", "Persona · peso r", "doc 33", "#1a9a6e", "^")]
BASE_C = "#8f8e88"
REPORT = {"glmw": "f5_report.json", "glmw4": "f5_report_glmw4.json", "personaw": "f5_report_personaw.json"}
SOFT = {"glmw": SOFT_GLMW, "glmw4": REL_OUT / "soft_labels_taskA_glmw4.json", "personaw": SOFT_PERSONAW}
EX_ANN, EX_MEME = "Annotator_456", "211327"       # exemplos do diagrama do peso (dados reais)
SOFT_KEY, HARD_KEY, LOSS_KEY = "eval_task_a_icmsoftnorm_soft", "eval_task_a_icmnorm_hard", "eval_loss"


def num(x, n=3, sign=False):
    return (f"{x:+.{n}f}" if sign else f"{x:.{n}f}").replace(".", ",").replace("-", "−")


def pct(x, n=1):
    return f"{100 * x:.{n}f}".replace(".", ",") + " %"


# ------------------------------------------------------------------ dados

memes, train, raw = load_memes(), load_json(TRAIN_IDS), load_json(SOFT_RAW)
N_VAL = len(load_json(VAL_IDS))
panels, _ = load_train_panels(memes, train)
anns = {a for v in panels.values() for a in v}
p_ref = {"glm": soft_votes(load_glm_votes(VOTES_CSV)), "persona": soft_votes(load_glm_votes(PERSONA_VOTES))}
terms = {k: deviation_terms(panels, p) for k, p in p_ref.items()}
D = {k: deviation_from_terms(t, anns) for k, t in terms.items()}
r_glm, r_per = vote_weights(D["glm"]), vote_weights(D["persona"])
csv_r = {"glm": {x["annotator_id"]: float(x["r_glm"]) for x in csv.DictReader(open(REF / "annotator_weights.csv"))},
         "persona": {x["annotator_id"]: float(x["r_persona"]) for x in csv.DictReader(open(PERSONA_WEIGHTS))}}
assert all(r_glm[a] == csv_r["glm"][a] and r_per[a] == csv_r["persona"][a] for a in anns), "pesos ≠ CSV"
R = {"glmw": r_glm, "glmw4": {a: v ** 4 for a, v in r_glm.items()}, "personaw": r_per}

soft = {arm: load_json(path) for arm, path in SOFT.items()}
shift = {}
for arm, *_ in ARMS:
    for i in train:      # o arquivo congelado é exatamente Σ r·[SIM] / Σ r com o peso do braço
        assert abs(soft[arm][i][0] - weighted_soft_label(panels[i], R[arm])) < 1e-12, (arm, i)
    d = [soft[arm][i][0] - raw[i][0] for i in train]
    hh = [(hard(raw[i][0]), hard(soft[arm][i][0])) for i in train]
    shift[arm] = {"d": d, "mean": mean(d), "abs": mean(map(abs, d)),
                  "moved": sum(abs(x) >= T["half_vote"] - T["tol"] for x in d) / len(d),
                  "hard": sum(a != b for a, b in hh),
                  "ties": sum(a == "TIE" for a, _ in hh), "ties_broken": sum(a == "TIE" != b for a, b in hh),
                  "flips": sum({a, b} == {"YES", "NO"} for a, b in hh),
                  "r_mean": mean(R[arm].values())}
# as mesmas contas já registradas (SPEC_28 §1 e SPEC_33 §12; inversões SIM↔NÃO: checagem (d))
for arm, ref in {"glmw": (0.054, 0.065, 0.346, 554, 36), "glmw4": (0.120, 0.154, 0.625, 849, 331),
                 "personaw": (0.0054, 0.053, 0.274, 526, 8)}.items():
    s = shift[arm]
    assert (round(s["mean"], 4 if arm == "personaw" else 3), round(s["abs"], 3), round(s["moved"], 3),
            s["hard"], s["flips"]) == ref, (arm, s)
    assert s["hard"] == s["ties_broken"] + s["flips"]
N_TIES = shift["glmw"]["ties"]

calib = {k: [] for k in ("glm", "persona", "n")}
for k in range(7):
    ids = [i for i in train if len(panels[i]) == 6 and n_yes(panels[i]) == k]
    calib["n"].append(len(ids))
    for ref in ("glm", "persona"):
        calib[ref].append(mean(p_ref[ref][i] for i in ids if p_ref[ref][i] is not None))
assert sum(calib["n"]) == len(train)
assert [round(x, 3) for x in calib["persona"]] == [0.055, 0.153, 0.261, 0.477, 0.662, 0.791, 0.923]
assert [round(x, 3) for x in calib["glm"]] == [0.400, 0.648, 0.737, 0.817, 0.902, 0.929, 0.964]
# grupos 1, 3 e 5 = os extremos do doc 32, onde S1–S3 foram escritos; 0, 2, 4 e 6 são fora da amostra
GAP_OUT = max(abs(calib["persona"][k] - k / 6) for k in (0, 2, 4, 6))

rep = {arm: load_json(REPORTS / f)["best"] for arm, f in REPORT.items()}
crit = {arm: {"up": rep[arm][SOFT_KEY]["mean"] > 0, "sig": rep[arm][SOFT_KEY]["p"] < 0.05,
              "loss": rep[arm][LOSS_KEY]["mean"] <= 0} for arm in rep}
for c in crit.values():
    c["all"] = c["up"] and c["sig"] and c["loss"]
assert not any(c["all"] for c in crit.values()), "o texto abaixo supõe nenhum critério satisfeito"
assert [a for a in rep if rep[a][LOSS_KEY]["p"] < 0.05] == ["glmw4", "personaw"]       # o texto supõe isto
assert [a for a in rep if rep[a][SOFT_KEY]["p"] < 0.05] == ["personaw"]
BASE_AT_E = {k: mean(rep["personaw"][k]["per_seed"][str(s)]["glmbase"] for s in SEEDS) for k in (SOFT_KEY, LOSS_KEY)}
for a in rep:     # o BASE é o mesmo nos três relatórios
    assert all(rep[a][k]["per_seed"][str(s)]["glmbase"] == rep["personaw"][k]["per_seed"][str(s)]["glmbase"]
               for k in (SOFT_KEY, LOSS_KEY) for s in SEEDS)

found = {**discover("glmw"), **discover("glmw4"), **discover("personaw")}
curves = {arm: {k: np.mean([[e[k] for e in found[(arm, s)]["evals"]] for s in SEEDS], axis=0)
                for k in (SOFT_KEY, LOSS_KEY)} for arm in ["glmbase"] + [a for a, *_ in ARMS]}
best_ep = {(arm, s): 1 + int(np.argmin([e[LOSS_KEY] for e in v["evals"]])) for (arm, s), v in found.items()}
odd = [k for k, e in best_ep.items() if e != 3]
assert odd == [("glmw4", 42)] and best_ep[("glmw4", 42)] == 2, best_ep      # o texto supõe isto
E_TXT = "a 3ª época em 11 dos 12 treinos (4 modelos × 3 seeds); no doc 28 com a seed 42, foi a 2ª"

# exemplo A: um anotador, as parcelas dos dois desvios
by = {k: {(a, m): (w, d) for a, m, w, d in t if a == EX_ANN} for k, t in terms.items()}
assert by["glm"].keys() == by["persona"].keys()       # o exemplo tem os mesmos memes nas duas referências
rowsA = []
for (a, m), (w, dg) in by["glm"].items():
    w2, dp = by["persona"][(a, m)]
    assert w2 == w
    others = [v for b, v in panels[m].items() if b != a]
    rowsA.append({"meme": m, "vote": panels[m][a], "k": sum(o == "YES" for o in others), "n": len(others),
                  "w": w, "pg": p_ref["glm"][m], "dg": dg, "pp": p_ref["persona"][m], "dp": dp})
sw = sum(x["w"] for x in rowsA)
DA = {k: sum(x["w"] * x[d] for x in rowsA) / sw for k, d in (("glm", "dg"), ("persona", "dp"))}
assert abs(DA["glm"] - D["glm"][EX_ANN]) < 1e-12 and abs(DA["persona"] - D["persona"][EX_ANN]) < 1e-12
N_MEMES_A = sum(EX_ANN in v for v in panels.values())

# exemplo B: um meme, os 6 votos com os dois pesos
votesB = panels[EX_MEME]
pB = {arm: weighted_soft_label(votesB, R[arm]) for arm in ("glmw", "personaw")}
assert pB["glmw"] == soft["glmw"][EX_MEME][0] and pB["personaw"] == soft["personaw"][EX_MEME][0]


# ------------------------------------------------------------------ figuras

plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"],      # DejaVu cobre as setas
                     "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def clean(ax, title, ylabel=None):
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_zorder(5)          # o eixo por cima das bordas brancas das barras
    ax.tick_params(length=0)


def save(fig, name):
    path = REPORTS / f"resultados_{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")     # vetorial, para a tese
    plt.close(fig)
    return path.as_uri()


def fig_delta():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, key, title in ((axes[0], SOFT_KEY, "Métrica principal: ICM-Soft-Norm (↑ melhor)"),
                           (axes[1], LOSS_KEY, "eval_loss (↓ melhor; acima de 0 = piorou)")):
        ax.axhline(0, color=INK, lw=1, zorder=1)
        top = max(abs(rep[a][key]["per_seed"][str(s)]["delta"]) for a, *_ in ARMS for s in SEEDS)
        for x, (arm, name, doc, color, mk) in enumerate(ARMS):
            r = rep[arm][key]
            d = [r["per_seed"][str(s)]["delta"] for s in SEEDS]
            ax.scatter([x - 0.2, x - 0.12, x - 0.04], d, color=color, marker=mk, s=34, alpha=0.8, zorder=3,
                       edgecolor="white", linewidth=0.8)          # uma coluna por seed: pontos iguais não se escondem
            ax.plot([x + 0.04, x + 0.24], [r["mean"]] * 2, color=color, lw=2.5, zorder=4,
                    solid_capstyle="round")
            ax.annotate(f"média {num(r['mean'], 4, True)}\np-valor {num(r['p'])}", (x + 0.28, r["mean"]),
                        xytext=(0, 2), textcoords="offset points", ha="left", va="bottom", fontsize=7.2, color=INK)
        ax.set_xticks(range(3), [f"{n.replace(' · peso', '')}\n({d})" for _, n, d, *_ in ARMS], fontsize=8)
        ax.set_xlim(-0.45, 2.85)
        # a eval_loss só sobe: escala assimétrica para os pontos perto de zero não sumirem na linha
        ax.set_ylim(*((-1.45 * top, 1.45 * top) if key == SOFT_KEY else (-0.2 * top, 1.3 * top)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))     # ticks redondos: 2 casas não arredondam
        ax.yaxis.set_major_formatter(lambda v, _: num(v, 2, abs(v) > 1e-9))
        clean(ax, title, "Δ = método − BASE" if key == SOFT_KEY else None)
    fig.tight_layout(w_pad=2.5)
    return save(fig, "delta")


def fig_curvas():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ep = np.arange(1, 16)
    lines = [("glmbase", "BASE (rótulos originais)", BASE_C, "D")] + [(a, f"{n} ({d})", c, m) for a, n, d, c, m in ARMS]
    for ax, key, title in ((axes[0], SOFT_KEY, "ICM-Soft-Norm por época (↑ melhor)"),
                           (axes[1], LOSS_KEY, "eval_loss por época (↓ melhor)")):
        for arm, label, color, mk in lines:
            ax.plot(ep, curves[arm][key], color=color, lw=2 if arm != "glmbase" else 2.4, marker=mk, ms=3.6,
                    label=label, zorder=3 if arm != "glmbase" else 2,
                    ls="--" if arm == "glmbase" else "-")
        ax.axvline(3, color=MUTED, lw=1, ls=":", zorder=1)
        ax.text(3.25, ax.get_ylim()[1], "3ª época", fontsize=7.5, color=MUTED, va="top")
        ax.set_xticks([1, 3, 5, 10, 15])
        ax.set_xlabel("época", fontsize=8.5)
        ax.yaxis.set_major_formatter(lambda v, _, k=key: num(v, 3 if k == SOFT_KEY else 2))
        clean(ax, title)
    axes[0].legend(fontsize=7.4, frameon=False, loc="lower right")
    fig.tight_layout(w_pad=2.5)
    return save(fig, "curvas")


def fig_labels():
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.85), sharey="row")
    for j, (arm, name, doc, color, mk) in enumerate(ARMS):
        w = "r⁴" if arm == "glmw4" else "r"
        ax = axes[0, j]
        ax.hist(list(R[arm].values()), bins=np.linspace(0, 1, 21), color=color, edgecolor="white", lw=1.2, zorder=3)
        ax.axvline(shift[arm]["r_mean"], color=INK, lw=1.2, ls="--", zorder=4)
        ax.set_xlabel(f"peso usado ({w}) do anotador", fontsize=8)
        ax.xaxis.set_major_formatter(lambda v, _: num(v, 2))
        clean(ax, f"{name} ({doc})\n{w} médio {num(shift[arm]['r_mean'], 2)}", "nº de anotadores" if j == 0 else None)
        ax = axes[1, j]
        ax.hist(shift[arm]["d"], bins=np.linspace(-0.5, 0.5, 25), color=color, edgecolor="white", lw=1.2, zorder=3)
        ax.axvline(0, color=INK, lw=1, zorder=4)
        ax.set_xlabel("p_new − p_old  (+ = para SIM)", fontsize=8)
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.xaxis.set_major_formatter(lambda v, _: num(v, 1, abs(v) > 1e-9))
        clean(ax, f"média {num(shift[arm]['mean'], 3, True)}\n{pct(shift[arm]['moved'], 0)} mudam ≥ meio voto",
              "nº de memes de treino" if j == 0 else None)
    fig.tight_layout(h_pad=1.2)
    return save(fig, "labels")


def fig_calib():
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    k = np.arange(7)
    ax.plot(k, k / 6, color=BASE_C, lw=2, ls="--", marker="D", ms=4, label="humanos (fração de SIM entre os 6)")
    ax.plot(k, calib["glm"], color=ARMS[0][3], lw=2, marker="o", ms=5, label="GLM-4.6V-Flash (docs 27 e 28)")
    ax.plot(k, calib["persona"], color=ARMS[2][3], lw=2, marker="^", ms=5.5, label="Persona: Claude Sonnet 5.5 (doc 33)")
    ax.annotate(num(calib["glm"][0]), (0, calib["glm"][0]), xytext=(7, 0), textcoords="offset points",
                va="center", fontsize=7.6, color=INK)
    ax.annotate(num(calib["persona"][0]), (0, calib["persona"][0]), xytext=(-2, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.6, color=INK)
    ax.set_xticks(k, [f"{i} de 6" for i in k])
    ax.set_xlim(-0.35, 6.3)
    ax.set_xlabel("quantos dos 6 humanos disseram SIM", fontsize=8.5)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(lambda v, _: num(v, 1))
    clean(ax, "Com que frequência a referência diz SIM", "fração de respostas SIM")
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    fig.tight_layout()
    return save(fig, "calibracao")


FIG = {"delta": fig_delta(), "curvas": fig_curvas(), "labels": fig_labels(), "calib": fig_calib()}


# ------------------------------------------------------------------ HTML

def pv(a, k):
    return f"p-valor = {num(rep[a][k]['p'])}"


def mark(ok, text):
    return f'<span class="{"ok" if ok else "no"}">{"✓" if ok else "✗"} {text}</span>'


verdict_rows = "".join(
    f"<tr><td>{q}</td>" + "".join(f"<td>{mark(crit[a][c], v(a))}</td>" for a, *_ in ARMS) + "</tr>"
    for q, c, v in [
        ("1. A métrica principal subiu?<br><small>Δ médio de ICM-Soft-Norm &gt; 0</small>", "up",
         lambda a: num(rep[a][SOFT_KEY]["mean"], 4, True)),
        ("2. A diferença passa no teste?<br><small>p-valor &lt; 0,05</small>", "sig", lambda a: pv(a, SOFT_KEY)),
        ("3. A eval_loss não piorou?<br><small>Δ médio de eval_loss ≤ 0</small>", "loss",
         lambda a: f"{num(rep[a][LOSS_KEY]['mean'], 4, True)} <small>({pv(a, LOSS_KEY)})</small>"),
    ]) + "<tr class='total'><td><b>Critério (as 3 juntas)</b></td>" + "".join(
    f"<td>{mark(crit[a]['all'], 'satisfeito' if crit[a]['all'] else 'não satisfeito')}</td>" for a, *_ in ARMS) + "</tr>"

cards = "".join(f"""<div class="card" style="border-top-color:{c}"><div class="doc">{d}</div><b>{n}</b>
<p>{t}</p></div>""" for (a, n, d, c, m), t in zip(ARMS, [
    "Referência: o modelo GLM-4.6V-Flash. Peso r = 1 − D (D = quanto o anotador discorda da referência).",
    "Mesma referência e mesmo r do doc 27, elevado à 4ª potência: quem já tinha peso baixo quase some (dose maior).",
    "Referência: o modelo Claude Sonnet 5.5 no papel de anotador (Persona). Peso r = 1 − D, como no doc 27."]))

sp = rep["personaw"][SOFT_KEY]
sp_d = [sp["per_seed"][str(s)]["delta"] for s in SEEDS]
rel = {(a, k): rep[a][k]["mean"] / BASE_AT_E[k] for a in rep for k in (SOFT_KEY, LOSS_KEY)}

glossary = [
    ("Dados e rótulos", [
        ("Task 2.1", "Pergunta do EXIST 2026 (competição de detecção de sexismo): o meme é sexista? Resposta SIM ou NÃO."),
        ("anotador", f"Pessoa que rotulou o meme. Cada meme foi rotulado por 6 pessoas; no treino são {len(anns)} pessoas diferentes."),
        ("treino / validação", f"{len(train)} memes usados para treinar o modelo; {N_VAL} memes separados só para medir o resultado. "
                               "Os rótulos da validação são sempre os originais, em todos os métodos."),
        ("rótulo suave (soft label)", "Fração dos 6 anotadores que disse SIM. Ex.: 4 de 6 → 0,667. Escrevemos <b>p_old</b> (original), "
                                      "<b>p_new</b> (com os pesos) e <b>p_ref</b> (da referência). Não confundir com o p-valor, explicado abaixo."),
        ("rótulo final", "SIM se a fração de SIM passa de 0,5; NÃO se fica abaixo; empate se é exatamente 0,5 (3 × 3)."),
    ]),
    ("Referências (os modelos que servem de régua)", [
        ("LLM", "Modelo de linguagem de grande porte, do tipo do ChatGPT. Os dois usados aqui também leem a imagem do meme."),
        ("prompt", "A instrução em texto enviada ao modelo junto com o meme. Cada referência responde 3 vezes, com 3 instruções "
                   "diferentes; cada resposta conta como um voto (p_ref = 0, 1/3, 2/3 ou 1)."),
        ("GLM", "O modelo GLM-4.6V-Flash, com a mesma pergunta feita de 3 jeitos e sem o modo de \"pensar por escrito\" antes de responder."),
        ("Persona", "O modelo Claude Sonnet 5.5, com 3 instruções de sistema (S1–S3) que o colocam no papel de um anotador humano cuidadoso."),
        ("D e r", "D = quanto o anotador discorda da referência (0 a 1); r = 1 − D é o peso do voto dele: 1 = voto inteiro, "
                  "perto de 0 = quase não conta. A conta passo a passo está na p. 3."),
    ]),
    ("Treino", [
        ("BASE", "O mesmo modelo de memes treinado com os rótulos originais, em que todos os anotadores valem o mesmo. "
                 "É contra ele que cada método é comparado."),
        ("seed", "Número que fixa os sorteios do treino (pesos iniciais, ordem dos memes). Mudar a seed muda o resultado um pouco, "
                 "só por acaso; por isso cada método foi treinado com 3 seeds (42, 7 e 2025) e comparado com o BASE na mesma seed."),
        ("época", f"Uma passada completa pelos {len(train)} memes de treino. Cada treino tem 15 épocas."),
        ("E", "A época em que os resultados são lidos: a de menor eval_loss, escolhida sem olhar a métrica principal, para não "
              f"pegar a época mais favorável. Foi {E_TXT}."),
    ]),
    ("Métricas (medidas na validação)", [
        ("ICM-Soft-Norm", "Métrica oficial do EXIST (ICM = Information Contrast Model), de 0 a 1. Compara a probabilidade de SIM "
                          "que o modelo prevê com a fração de anotadores que disse SIM. <b>Maior = melhor.</b> É a métrica principal."),
        ("ICM-Norm hard", "A mesma métrica usando só o rótulo final (SIM ou NÃO) em vez da probabilidade. Maior = melhor."),
        ("eval_loss", "Medida de erro do modelo na validação, contra os rótulos originais. <b>Menor = melhor.</b> Soma o erro das três "
                      "perguntas que o modelo responde: 2.1 (é sexista?, conta em dobro), 2.2 (qual a intenção de quem postou?) e "
                      "2.3 (que tipo de sexismo?)."),
        ("Δ", "Diferença método − BASE na mesma seed. Δ &gt; 0 na ICM = melhorou; Δ &gt; 0 na eval_loss = piorou."),
    ]),
]
gloss_html = "".join(f"<tr class='grp'><td colspan=2>{g}</td></tr>" + "".join(
    f"<tr><td class='term'>{t}</td><td>{d}</td></tr>" for t, d in items) for g, items in glossary)

flow = "".join(f'<div class="step"><div class="n">{i}</div>{t}</div>' + ('<div class="arrow">→</div>' if i < 5 else "")
               for i, t in enumerate([
                   "<b>6 humanos</b> votam SIM/NÃO em cada meme",
                   "<b>Referência</b> (GLM ou Persona) vota 3 vezes no mesmo meme",
                   "<b>Peso&nbsp;r</b> de cada anotador: quanto ele concorda com a referência",
                   "<b>Novo rótulo</b> p_new: votos somados com o peso&nbsp;r",
                   "<b>Retreino</b> com p_new, 3 seeds, e comparação com o BASE"], 1))


def frac(p):
    return {0: "0", 1: "1"}.get(p, f"{round(p * 3)}/3")


lbl = {"YES": "SIM", "NO": "NÃO"}
rowsA_html = "".join(
    f"<tr><td>{x['meme']}</td><td class='{x['vote'].lower()}'>{lbl[x['vote']]}</td><td>{x['k']} de {x['n']}</td>"
    f"<td>{num(x['w'])}</td><td>{frac(x['pg'])}</td><td>{num(x['dg'])}</td><td>{frac(x['pp'])}</td>"
    f"<td>{num(x['dp'])}</td></tr>" for x in rowsA)
swd = {k: sum(x["w"] * x[d] for x in rowsA) for k, d in (("glm", "dg"), ("persona", "dp"))}
keyA = max(rowsA, key=lambda x: x["w"] * abs(x["dg"] - x["dp"]))     # a parcela que mais separa os dois desvios

rB = {"glmw": r_glm, "personaw": r_per}


def bar(r, vote):
    return (f'<div class="bar"><span class="{vote.lower()}" style="width:{100 * r:.1f}%"></span></div>'
            f'<span class="bv">{num(r)}</span>')


rowsB_html = "".join(f"<tr><td>{a}</td><td class='{v.lower()}'>{lbl[v]}</td><td>{bar(rB['glmw'][a], v)}</td>"
                     f"<td>{bar(rB['personaw'][a], v)}</td></tr>" for a, v in sorted(votesB.items()))
sumB = {arm: (sum(rB[arm][a] for a, v in votesB.items() if v == "YES"), sum(rB[arm][a] for a in votesB))
        for arm in rB}

lab_rows = "".join(
    f"<tr><td><span class='dot' style='background:{c}'></span>{n}<br><small>{d}</small></td>"
    f"<td>{num(shift[a]['r_mean'], 2)}{' (r⁴)' if a == 'glmw4' else ''}</td>"
    f"<td>{num(shift[a]['mean'], 3, True)}</td><td>{num(shift[a]['abs'], 3)}</td><td>{pct(shift[a]['moved'])}</td>"
    f"<td>{shift[a]['ties_broken']} de {N_TIES}</td><td>{shift[a]['flips']}</td></tr>" for a, n, d, c, m in ARMS)

METRICS = [(SOFT_KEY, "Task 2.1 · ICM-Soft-Norm (principal)", "↑", 1), (HARD_KEY, "Task 2.1 · ICM-Norm hard", "↑", 1),
           (LOSS_KEY, "eval_loss", "↓", -1), ("eval_task_b_icmsoftnorm_soft", "Task 2.2 · ICM-Soft-Norm", "↑", 1),
           ("eval_task_c_icmsoftnorm_soft", "Task 2.3 · ICM-Soft-Norm", "↑", 1)]


def cell_class(a, k, good):
    r = rep[a][k]
    return "" if r["p"] >= 0.05 else ("better" if r["mean"] * good > 0 else "worse")


res_rows = "".join(
    f"<tr><td>{lab} <span class='muted'>{d}</span></td>" + "".join(
        f"<td class='{cell_class(a, k, g)}'>{num(rep[a][k]['mean'], 4, True)} "
        f"<small>({pv(a, k)})</small></td>" for a, *_ in ARMS) + "</tr>"
    for k, lab, d, g in METRICS)
assert not any(rep[a][k]["p"] < 0.05 for a in rep for k, *_ in METRICS[3:])          # o texto supõe isto
head_arms = "".join(f"<th><span class='dot' style='background:{c}'></span>{n}<br><small>{d}</small></th>"
                    for a, n, d, c, m in ARMS)

CSS = """
@page { size: A4; margin: 15mm 15mm 15mm 15mm; }
:root { --ink:#1a1a18; --muted:#55534e; --line:#dcdad3; --soft:#f3f2ee; --ok:#17663f; --no:#a3261c; }
html, body { background:#fff; }
body { color:var(--ink); font-family:"Charter","Georgia",serif; font-size:10pt; line-height:1.42; margin:0; }
h1 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:19pt; line-height:1.15; margin:0 0 4pt; }
h2 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:13.5pt; margin:14pt 0 5pt; border-bottom:1px solid var(--line);
     padding-bottom:3pt; break-after:avoid; }
h3 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:11pt; margin:10pt 0 4pt; break-after:avoid; }
p { margin:4pt 0 6pt; }
.sub { color:var(--muted); font-size:10.5pt; margin-bottom:10pt; }
.nw { white-space:nowrap; }
.muted, small { color:var(--muted); }
small { font-size:7.6pt; }
.note { color:var(--muted); font-size:8.6pt; }
table { border-collapse:collapse; width:100%; margin:5pt 0 8pt; font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.4pt; }
th, td { border-bottom:1px solid var(--line); padding:3.5pt 5pt; text-align:left; vertical-align:top; }
th { background:var(--soft); font-weight:600; }
tr { break-inside:avoid; }
table, figure, .box, .flow, .steps { break-inside:avoid; }
figure { margin:6pt 0 8pt; }
figure img { width:100%; display:block; }
figcaption { font-size:8.3pt; color:var(--muted); margin-top:3pt; line-height:1.35; }
.cards { display:flex; gap:8pt; margin:6pt 0 8pt; }
.card { flex:1; border:1px solid var(--line); border-top:4px solid; border-radius:4pt; padding:6pt 8pt;
        font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.6pt; }
.card .doc { color:var(--muted); font-size:7.6pt; text-transform:uppercase; letter-spacing:.04em; }
.card p { margin:3pt 0 0; line-height:1.35; }
.ok { color:var(--ok); font-weight:600; } .no { color:var(--no); font-weight:600; }
tr.total td { background:var(--soft); }
.box { background:var(--soft); border-left:3px solid var(--ink); padding:6pt 10pt; margin:8pt 0; }
.box p { margin:3pt 0; }
.term { width:20%; font-weight:600; }
tr.grp td { font-weight:700; background:#fff; border-bottom:1.5px solid var(--ink); padding-top:8pt; }
.flow { display:flex; align-items:stretch; gap:3pt; margin:6pt 0 10pt; font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.2pt; }
.flow .step { flex:1; border:1px solid var(--line); border-radius:4pt; padding:5pt 6pt; background:#fff; line-height:1.3; }
.n { display:inline-block; width:13pt; height:13pt; border-radius:50%; background:var(--ink); color:#fff;
     text-align:center; font-size:7.5pt; line-height:13pt; margin-right:3pt; font-weight:700; font-family:"Helvetica Neue",Arial,sans-serif; }
.flow .arrow { align-self:center; color:var(--muted); font-size:12pt; }
.steps { display:grid; grid-template-columns:1fr 1fr; gap:6pt; margin:6pt 0 8pt; align-items:start; }
.steps div.s { border:1px solid var(--line); border-radius:4pt; padding:5pt 8pt; font-size:9pt; line-height:1.35; }
.steps div.wide { grid-column:1 / 3; background:var(--soft); }
.f { font-family:"SF Mono",Menlo,monospace; font-size:8.4pt; background:#fff; border:1px solid var(--line);
     border-radius:2pt; padding:0 2pt; white-space:nowrap; }
td.yes { font-weight:700; } td.no { color:var(--muted); }
.bar { display:inline-block; width:62%; height:8pt; background:var(--soft); vertical-align:middle; border-radius:2pt; }
.bar span { display:block; height:100%; border-radius:2pt; }
.bar span.yes { background:#3d3c38; } .bar span.no { background:#b9b7b0; }
.bv { margin-left:5pt; }
.dot { display:inline-block; width:8pt; height:8pt; border-radius:50%; margin-right:4pt; vertical-align:-0.5pt; }
td.better { background:#e5f2ea; } td.worse { background:#f8e4e1; }
td, th { font-variant-numeric:tabular-nums; }
.pagebreak { break-before:page; }
table.gl { font-size:7.9pt; margin-top:2pt; } table.gl td { padding:2.2pt 4pt; line-height:1.3; }
table.gl tr.grp td { padding-top:5pt; }
.box.small { font-size:9pt; line-height:1.36; }
table.res td { padding:2.6pt 5pt; }
"""

r456 = {"glm": 1 - DA["glm"], "persona": 1 - DA["persona"]}
html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Resultados da ponderação</title>
<style>{CSS}</style></head><body>

<h1>Dar pesos aos anotadores usando um modelo de linguagem como referência: resultados</h1>
<div class="sub">EXIST 2026 · Task 2.1 (o meme é sexista? SIM/NÃO) · docs 27, 28 e 33 · versão de 2026-10-29 ·
<span class="nw">Jonathan Santos da Silva</span></div>

<h2>Resumo</h2>
<p class="note">Os termos técnicos (seed, Δ, p-valor, ICM, eval_loss…) estão explicados na p. 2; a conta do peso está na p. 3.</p>
<p><b>A pergunta.</b> Cada meme de treino foi rotulado por 6 pessoas, que muitas vezes discordam. A ideia testada:
dar <b>mais peso</b> às pessoas que concordam com um modelo de linguagem usado como <b>referência</b>, recalcular os rótulos
de treino com esses pesos e treinar o modelo de memes de novo. Ele fica melhor do que o mesmo modelo treinado com os
rótulos originais (o <b>BASE</b>)?</p>

<div class="cards">{cards}</div>

<p><b>O critério</b>, fixado antes de qualquer treino: o método só "funciona" se as três condições abaixo valerem juntas.
Cada método foi treinado 3 vezes (3 seeds) e comparado com o BASE na mesma seed.</p>
<table><tr><th style="width:31%">condição</th>{head_arms}</tr>{verdict_rows}</table>

<div class="box"><p><b>Resposta curta: nenhum método mostrou melhora.</b> Nenhum dos três cumpriu o critério; cumprir duas
condições em três não basta.</p>
<p><b>Persona (doc 33):</b> melhorou a métrica principal ({num(sp['mean'], 4, True)}, {pv('personaw', SOFT_KEY)}), mas também piorou a
eval_loss nas 3 seeds ({num(rep['personaw'][LOSS_KEY]['mean'], 4, True)}, {pv('personaw', LOSS_KEY)}). As duas diferenças passam no teste
e vão em direções opostas. A regra fixada antes classifica esse caso como "efeito não detectado com 3 seeds", o que não é o mesmo
que "efeito zero".</p>
<p><b>Doc 27:</b> nenhuma diferença passa no teste (efeito não detectado). <b>Doc 28:</b> a eval_loss piorou
({num(rep['glmw4'][LOSS_KEY]['mean'], 4, True)}, {pv('glmw4', LOSS_KEY)}): um efeito negativo, registrado como resultado.</p></div>

<p class="note">Para dar escala: na época E, o BASE marca {num(BASE_AT_E[SOFT_KEY])} na ICM-Soft-Norm e {num(BASE_AT_E[LOSS_KEY])} na eval_loss
(média das 3 seeds). O ganho da Persona na ICM equivale a {pct(rel[('personaw', SOFT_KEY)])}; a piora da eval_loss foi de
{pct(rel[('personaw', LOSS_KEY)])} na Persona e de {pct(rel[('glmw4', LOSS_KEY)])} no doc 28. Os três docs são exploratórios: o doc 28
testou uma dose maior do peso sem apostar antes se ia melhorar ou piorar.</p>

<div class="pagebreak"></div>
<h2>Como ler: o que cada termo significa</h2>
<table class="gl">{gloss_html}</table>

<div class="box small">
<p><b>O que é o p-valor (e o "p-valor &lt; 0,05")?</b> Imagine que o método <i>não muda nada</i>. Mesmo assim, as três seeds
dariam diferenças pequenas em relação ao BASE, umas para cima e outras para baixo, só por acaso. O <b>p-valor</b> é a chance
de, nesse mundo "sem efeito", aparecerem três diferenças tão longe de zero, em relação ao quanto variam entre si, quanto as que observamos.</p>
<p><b>p-valor &lt; 0,05</b> (menos de 5 %): seria raro só por acaso, então dizemos que a diferença <b>passa no teste</b>.
<b>p-valor ≥ 0,05</b>: o acaso ainda poderia produzir algo assim com alguma frequência, então não dá para afirmar que existe efeito.
O limite de 0,05 é uma convenção usada em muitas áreas. O p-valor <b>não</b> é a probabilidade de o método funcionar.</p>
<p><b>Como ele é calculado aqui (teste t pareado).</b> "Pareado" porque cada seed do método é comparada com a mesma seed do BASE:
Δ = método − BASE. Com as 3 diferenças, calcula-se <span class="f">t = média(Δ) ÷ (desvio padrão(Δ) ÷ √3)</span>, em que o desvio
padrão mede o quanto as 3 diferenças se espalham em torno da média, e 3 é o número de seeds. Exemplo da Persona na métrica principal:
Δ = {", ".join(num(x, 5, True) for x in sp_d)}; média {num(sp['mean'], 5, True)}, desvio padrão {num(sp['sd'], 5)} → t = {num(sp['t'], 2)}
→ p-valor = {num(sp['p'])} (tirado da distribuição do teste t para 3 medidas). As três seeds subiram quase o mesmo tanto, por isso o
p-valor ficou baixo.</p>
<p><b>Limite honesto:</b> com só 3 seeds, o p-valor só fica abaixo de 0,05 se t passar de 4,30 (ou ficar abaixo de −4,30). O teste só
detecta diferenças grandes e consistentes; uma diferença real, mas pequena, pode passar sem ser detectada. Por isso
"não detectado" ≠ "efeito zero".</p>
</div>

<h2>O que foi feito</h2>
<div class="flow">{flow}</div>

<h2>Como o peso de cada anotador é calculado</h2>
<p>A conta é a mesma nos três métodos; só muda a referência (GLM ou Persona) e, no doc 28, o peso final é elevado à 4ª potência.
Σ quer dizer "soma".</p>
<div class="steps">
<div class="s"><span class="n">1</span><b>Quais memes entram.</b> Só os memes de treino em que os 6 humanos <b>não</b> foram unânimes
(e a referência respondeu). Memes unânimes não dizem nada sobre quem discorda.</div>
<div class="s"><span class="n">2</span><b>Distância no meme:</b> <span class="f">d = |voto − p_ref|</span>, com SIM = 1 e NÃO = 0; as barras
indicam a distância sem sinal (d = 0: concordou totalmente; d = 1: discordou totalmente). Ex.: o anotador disse SIM e a referência
disse SIM em 2 das 3 vezes → d = |1 − 0,667| = 0,333.</div>
<div class="s"><span class="n">3</span><b>Importância do meme (w):</b> mede o quanto os <b>outros 5</b> anotadores concordam entre si,
seja a maioria SIM ou NÃO. 5 a 0 → w = 1; 4 a 1 → w = {num(1 - binary_entropy(0.8))}; 3 a 2 → w = {num(1 - binary_entropy(0.6))}.
Memes em que os próprios humanos estão divididos quase não contam.
<br><small>Tecnicamente, w = 1 − H(k), com k = fração de SIM entre os outros 5 e H a entropia binária, que vale 0 quando todos
concordam e 1 quando o grupo está dividido ao meio.</small></div>
<div class="s"><span class="n">4</span><b>Desvio e peso:</b> <span class="f">D = Σ w·d ÷ Σ w</span>, a média ponderada das distâncias
(memes mais importantes contam mais), e <span class="f">r = 1 − D</span>. r perto de 1 = concorda com a referência; perto de 0 =
discorda. No doc 28, o peso usado é r⁴. Anotador sem nenhum meme incluído: r = 1.</div>
<div class="s wide"><span class="n">5</span><b>Novo rótulo do meme:</b> <span class="f">p_new = (soma dos r de quem votou SIM) ÷ (soma
dos r dos 6)</span>. Se todos tivessem r = 1, isso voltaria a ser a fração original p_old. Os memes unânimes não mudam, e os rótulos
da validação também não.</div>
</div>

<div class="pagebreak"></div>
<h3>Exemplo real A: o peso do {EX_ANN} ({len(rowsA)} dos seus {N_MEMES_A} memes entram)</h3>
<table><tr><th>meme</th><th>voto dele</th><th>SIM entre os outros 5</th><th>importância w</th>
<th>GLM: p_ref</th><th>GLM: d</th><th>Persona: p_ref</th><th>Persona: d</th></tr>
{rowsA_html}
<tr class="total"><td colspan=3><b>Σ w</b> e <b>Σ w·d</b></td><td>{num(sw)}</td><td></td><td>{num(swd['glm'])}</td><td></td><td>{num(swd['persona'])}</td></tr>
<tr class="total"><td colspan=3><b>D = Σ w·d ÷ Σ w</b></td><td></td><td colspan=2>{num(swd['glm'])} ÷ {num(sw)} = <b>{num(DA['glm'])}</b></td>
<td colspan=2>{num(swd['persona'])} ÷ {num(sw)} = <b>{num(DA['persona'])}</b></td></tr>
<tr class="total"><td colspan=3><b>peso r = 1 − D</b></td><td></td><td colspan=2><b>{num(r456['glm'])}</b> (r⁴ = {num(r456['glm'] ** 4)})</td>
<td colspan=2><b>{num(r456['persona'])}</b></td></tr></table>
<p class="note">Valores arredondados; a conta usa os valores exatos. O mesmo anotador recebe pesos diferentes porque as duas referências
discordam em alguns memes. O que mais pesa é o meme {keyA['meme']}, em que os outros 5 concordam entre si (w = {num(keyA['w'])}): ele
votou {lbl[keyA['vote']]}, a GLM deu p_ref = {frac(keyA['pg'])} (d = {num(keyA['dg'])}) e a Persona p_ref = {frac(keyA['pp'])}
(d = {num(keyA['dp'])}).</p>

<h3>Exemplo real B: o meme {EX_MEME}, empatado em 3 SIM × 3 NÃO (p_old = 0,500)</h3>
<p>A GLM disse SIM nas 3 respostas e a Persona disse NÃO nas 3. Mas o meme não recebe o voto da referência: ele recebe os votos dos
6 humanos, cada um com o peso r que o anotador ganhou nos seus memes incluídos (os não unânimes), não só neste. A barra mostra o peso
(escuro = votou SIM).</p>
<table><tr><th>anotador</th><th>voto</th><th style="width:30%">peso r com a GLM (doc 27)</th><th style="width:30%">peso r com a Persona (doc 33)</th></tr>
{rowsB_html}
<tr class="total"><td colspan=2><b>p_new = Σ r dos SIM ÷ Σ r</b></td>
<td>{num(sumB['glmw'][0])} ÷ {num(sumB['glmw'][1])} = <b>{num(pB['glmw'])}</b> → SIM</td>
<td>{num(sumB['personaw'][0])} ÷ {num(sumB['personaw'][1])} = <b>{num(pB['personaw'])}</b> → NÃO</td></tr></table>

<div class="pagebreak"></div>
<h2>O que cada método fez com os rótulos de treino</h2>
<table><tr><th style="width:17%">método</th><th>peso médio usado</th><th>deslocamento médio (+ = rumo ao SIM)</th>
<th>mudança média (sem sinal)</th><th>memes que mudam ≥ meio voto (1 voto = 1/6)</th>
<th>empates 3 × 3 desfeitos</th><th>SIM ↔ NÃO invertidos</th></tr>{lab_rows}</table>
<figure><img src="{FIG['labels']}"><figcaption>Linha de cima: o peso usado para cada um dos {len(anns)} anotadores (r; no doc 28, r⁴),
com a média tracejada. Linha de baixo: quanto o rótulo suave de cada meme de treino mudou (memes unânimes ficam em 0). Com a GLM, os
rótulos se deslocam para o SIM, ainda mais com r⁴; com a Persona, o deslocamento médio é quase zero, mas um quarto dos memes ainda
muda pelo menos meio voto. Todo empate 3 × 3 vira SIM ou NÃO com os pesos; inverter um rótulo que já era SIM ou NÃO é raro, exceto com r⁴.</figcaption></figure>

<div class="pagebreak"></div>
<h2>Resultados do retreino (época E)</h2>
<figure><img src="{FIG['delta']}"><figcaption>Cada símbolo é uma seed (da esquerda para a direita: 42, 7, 2025); o traço é a média das três.
Zero = igual ao BASE. À esquerda, acima de zero é melhor; à direita (eval_loss), acima de zero é pior. Os três métodos pioram a
eval_loss nas 3 seeds.</figcaption></figure>
<table class="res"><tr><th style="width:28%">métrica (Δ médio = método − BASE)</th>{head_arms}</tr>{res_rows}</table>
<p class="note">↑ = quanto maior, melhor; ↓ = quanto menor, melhor. Fundo verde = diferença que passa no teste (p-valor &lt; 0,05) para
melhor; fundo vermelho = para pior. Tasks 2.2 e 2.3: as outras duas perguntas do mesmo modelo, cujos rótulos não foram mexidos; servem
para ver se a mudança na Task 2.1 estragou algo nelas. Nenhuma tem p-valor &lt; 0,05.</p>

<h2>Época a época</h2>
<figure><img src="{FIG['curvas']}"><figcaption>Média das 3 seeds em cada época. A menor eval_loss (a época E) foi {E_TXT}. Depois dela,
a eval_loss sobe, e sobe mais nos métodos com peso, sobretudo no r⁴. As curvas são médias e não substituem o teste feito na época E.</figcaption></figure>

<div class="pagebreak"></div>
<h2>Por que as duas referências dão pesos diferentes</h2>
<figure><img src="{FIG['calib']}"><figcaption>Todos os {len(train)} memes de treino, agrupados por quantos dos 6 humanos disseram SIM.
Linha tracejada = os próprios humanos. Nos memes que <b>nenhum</b> humano achou sexista, {pct(calib['glm'][0], 0)} das respostas da GLM
foram SIM; da Persona, {pct(calib['persona'][0], 1)}. Os grupos 1, 3 e 5 são os memes em que as instruções da Persona foram escritas e
testadas (doc 32); nos grupos 0, 2, 4 e 6, que ela não tinha visto, a diferença para os humanos fica em no máximo {num(GAP_OUT, 2)}.</figcaption></figure>
<p>Com a GLM, quem diz muito SIM concorda mais com a referência e ganha peso, e os rótulos andam para o SIM. A Persona diz SIM
com frequência parecida com a dos humanos e quase não desloca os rótulos na média. Nas checagens do doc 33, o peso da Persona quase não
acompanha a taxa de SIM do anotador, mas ainda acompanha em parte o quanto ele é mais ou menos exigente que a Persona para dizer SIM
(leitura fixada antes: "parcial").</p>

<h2>Conclusão</h2>
<ul>
<li><b>Doc 27 (GLM, r):</b> ganho pequeno na métrica principal ({num(rep['glmw'][SOFT_KEY]['mean'], 4, True)}, {pv('glmw', SOFT_KEY)})
e eval_loss um pouco maior ({num(rep['glmw'][LOSS_KEY]['mean'], 4, True)}, {pv('glmw', LOSS_KEY)}); nenhum dos dois passa no teste.
Efeito não detectado.</li>
<li><b>Doc 28 (GLM, r⁴):</b> a dose maior piorou a eval_loss ({num(rep['glmw4'][LOSS_KEY]['mean'], 4, True)}, {pv('glmw4', LOSS_KEY)}):
efeito negativo. A métrica principal caiu ({num(rep['glmw4'][SOFT_KEY]['mean'], 4, True)}), mas sem passar no teste
({pv('glmw4', SOFT_KEY)}).</li>
<li><b>Doc 33 (Persona, r):</b> melhora que passa no teste na métrica principal ({num(sp['mean'], 4, True)}, {pv('personaw', SOFT_KEY)}),
positiva nas 3 seeds, e piora que também passa no teste na eval_loss ({num(rep['personaw'][LOSS_KEY]['mean'], 4, True)},
{pv('personaw', LOSS_KEY)}; cerca de 1/11 da piora do doc 28). O critério exigia as três condições juntas: não satisfeito.</li>
</ul>
<p>Na métrica principal, a Persona (a referência mais parecida com os humanos) ficou à frente da GLM, mas essa diferença entre as duas
não foi testada: é só uma observação. Nenhum método com peso superou o BASE pelo critério combinado. Pela regra fixada antes, esta
série de experimentos termina aqui, sem testar novas variantes do peso em busca de um resultado melhor.</p>

<h3>Limites</h3>
<ul>
<li>3 seeds: o teste só detecta efeitos grandes e consistentes.</li>
<li>Uma única divisão treino/validação, com {N_VAL} memes de validação. A mesma validação e o mesmo BASE foram usados nos três testes
(docs 27, 28 e 33), sem correção para comparações múltiplas. Com a correção mais simples (Bonferroni: limite 0,05 ÷ 3 ≈ 0,017),
o {pv('personaw', SOFT_KEY)} da Persona não passaria.
Os três docs são exploratórios, não confirmatórios.</li>
<li>A eval_loss é medida contra os rótulos originais. Um método que muda os rótulos de treino tende a se afastar deles; não medimos
quanto da piora vem disso. A condição sobre a eval_loss foi fixada antes, junto com o resto do critério.</li>
</ul>
<p class="note" style="font-size:8pt">Fontes: f5_report*.json (glm_reference/reports), metrics_history/, annotator_weights*.csv,
soft_labels_taskA_*.json, glm_votes.csv e persona_votes.csv. Gerado por glm_annotator/relatorio_resultados_pdf.py.</p>
</body></html>"""

OUT_HTML.write_text(html, encoding="utf-8")
subprocess.run([CHROME, "--headless=new", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", OUT_HTML.resolve().as_uri()],
               check=True, capture_output=True)
print("ok", PDF, PDF.stat().st_size)
