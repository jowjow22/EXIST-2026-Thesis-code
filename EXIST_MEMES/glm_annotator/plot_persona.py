"""Relatório "Persona" (doc 32): GLM off, GLM on, Claude (doc 31, P1–P3) e Claude Persona (doc 32, S1–S3)
nos 1 551 memes extremos. Só lê arquivos gravados; nenhuma chamada a modelo.

Uso (da raiz do repositório, depois de claude_reference_32.py):
    python3 EXIST_MEMES/glm_annotator/plot_persona.py
Imprime as tabelas do RELATORIO_PERSONA.md e grava em glm_reference/reports/: persona_voto_yes,
persona_prompts, persona_acerto, persona_cruzamento, persona_por_prompt (.png + .pdf).
"""
import csv
import json
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from client import majority
from common import MEMES_DIR, REF, REPORTS
from plot_doc30 import GRID, HUMAN, INK, MUTED, style

RAW = {"claude": MEMES_DIR / "claude_reference/raw/claude-sonnet-5-5_think_off.jsonl",
       "persona": MEMES_DIR / "claude_reference_32/raw/claude-sonnet-5-5_think_off.jsonl"}
# (chave, rótulo, cor, hachura): hachura nos dois Claude = segunda codificação além da cor
ARMS = [("think_off", "GLM · thinking desligado", "#2a78d6", ""),
        ("think_on", "GLM · thinking ligado", "#eb6834", ""),
        ("claude", "Claude · P1–P3 (doc 31)", "#0f8f6a", "//"),
        ("persona", "Claude Persona · S1–S3 (doc 32)", "#6250d6", "xx")]
NAME = {a: n for a, n, *_ in ARMS}
GROUPS = [("1Y5N", 1 / 6), ("3Y3N", 0.5), ("5Y1N", 5 / 6)]
BG = "#fcfcfb"
FS = 10.5          # legendas e rótulos: impressos a ~7 pol., ficam ≥ 6 pt


def big(ax):
    ax.title.set_fontsize(13)
    ax.yaxis.label.set_fontsize(11)
    ax.tick_params(labelsize=11)


def load():
    rows = list(csv.DictReader(open(REF / "extremes/extremes.csv", encoding="utf-8")))
    lab = {}
    for arm, path in RAW.items():
        lab[arm] = {}
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                lab[arm][(r["meme_id"], r["paraphrase"])] = r["label"]    # o último registro vale (reenvio PNG)
    for r in rows:
        r["think_off"] = [r[f"think_off_p{p}"] for p in (1, 2, 3)]
        r["think_on"] = [r[f"think_on_p{p}"] for p in (1, 2, 3)]
        for arm in RAW:
            r[arm] = [lab[arm][(r["meme_id"], p)] for p in (1, 2, 3)]
        for arm, *_ in ARMS:
            r[arm + "_vote"] = majority(r[arm])
    return rows


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def sel(rows, g=None, lang=None):
    return [r for r in rows if (g is None or r["group"] == g) and (lang is None or r["lang"] == lang)]


f3 = lambda x: f"{x:.3f}".replace(".", ",")


# ------------------------------------------------------------------ tabelas (texto do relatório)

def tables(rows):
    print("\n## T1 · voto YES (maioria das 3 respostas; denominador = todos os memes do grupo)")
    print("| grupo | idioma | n | humanos | " + " | ".join(NAME[a] for a, *_ in ARMS) + " |")
    for g, h in GROUPS + [("todos", None)]:
        for lang in (None, "en", "es"):
            s = sel(rows, None if g == "todos" else g, lang)
            if g == "todos" and lang:
                continue
            hum = h if h is not None else np.mean([int(r["human_yes"]) / 6 for r in s])
            cells = []
            for a, *_ in ARMS:
                k = sum(r[a + "_vote"] == "YES" for r in s)
                lo, hi = wilson(k, len(s))
                cells.append(f"{f3(k / len(s))} ({k}) [{f3(lo)}; {f3(hi)}]")
            print(f"| {g} | {lang or 'total'} | {len(s)} | {f3(hum)} | " + " | ".join(cells) + " |")
    print("sem voto:", {a: sum(r[a + "_vote"] is None for r in rows) for a, *_ in ARMS})

    print("\n## T2 · fração de YES por prompt (1, 2, 3 = P1–P3 ou S1–S3), sobre respostas parseadas")
    for g in (None, "1Y5N", "3Y3N", "5Y1N"):
        s = sel(rows, g)
        for a, *_ in ARMS:
            per = []
            for i in range(3):
                xs = [r[a][i] for r in s if r[a][i] in ("YES", "NO")]
                per.append(sum(x == "YES" for x in xs) / len(xs))
            xs = [x for r in s for x in r[a] if x in ("YES", "NO")]
            print(f"| {g or '1 551 extremos'} | {NAME[a]} | " + " | ".join(f3(v) for v in per)
                  + f" | {f3(sum(x == 'YES' for x in xs) / len(xs))} |")

    print("\n## T3 · concordância 3-0 (fração dos memes do grupo) e prompt dissidente nos memes 2-1")
    for a, *_ in ARMS:
        agree = [np.mean([len(set(r[a])) == 1 and r[a][0] != "?" for r in sel(rows, g)]) for g, _ in GROUPS]
        tot = np.mean([len(set(r[a])) == 1 and r[a][0] != "?" for r in rows])
        diss = Counter()
        for r in rows:
            if "?" not in r[a] and len(set(r[a])) == 2:
                odd = [i for i in range(3) if r[a].count(r[a][i]) == 1][0]
                diss[odd + 1] += 1
        n21 = sum(diss.values())
        print(f"| {NAME[a]} | " + " | ".join(f3(v) for v in agree) + f" | {f3(tot)} | "
              f"{diss[1]} · {diss[2]} · {diss[3]} (n = {n21}; maior: {max(diss.values()) / n21:.0%}) |")

    print("\n## T4 · acordo com a maioria humana (1Y5N: NO; 5Y1N: YES; sem voto = erro)")
    for a, *_ in ARMS:
        for lang in (None, "en", "es"):
            neg, pos = sel(rows, "1Y5N", lang), sel(rows, "5Y1N", lang)
            spec = np.mean([r[a + "_vote"] == "NO" for r in neg])
            sens = np.mean([r[a + "_vote"] == "YES" for r in pos])
            acc = (sum(r[a + "_vote"] == "NO" for r in neg) + sum(r[a + "_vote"] == "YES" for r in pos)) / (len(neg) + len(pos))
            print(f"| {NAME[a]} | {lang or 'total'} | {f3(spec)} | {f3(sens)} | {f3(acc)} | {f3((spec + sens) / 2)} |")

    print("\n## T5 · AUC 5Y1N × 1Y5N com escore = nº de YES nas 3 respostas (0–3; '?' conta como não-YES)")
    neg, pos = sel(rows, "1Y5N"), sel(rows, "5Y1N")
    for a, *_ in ARMS:
        cn = Counter(r[a].count("YES") for r in neg)
        cp = Counter(r[a].count("YES") for r in pos)
        w = sum(cp[x] * cn[y] * (1.0 if x > y else 0.5 if x == y else 0.0) for x in range(4) for y in range(4))
        print(f"| {NAME[a]} | {f3(w / (len(neg) * len(pos)))} |")

    print("\n## T6 · Fleiss κ entre as 3 respostas (só memes com as 3 parseadas) e fração de YES por resposta")
    for a, *_ in ARMS:
        out = []
        for g in ("1Y5N", "3Y3N", "5Y1N", None):
            s = [r for r in sel(rows, g) if "?" not in r[a]]
            ny = [r[a].count("YES") for r in s]
            Pb = sum((y * (y - 1) + (3 - y) * (2 - y)) / 6 for y in ny) / len(s)
            pY = sum(ny) / (3 * len(s))
            Pe = pY ** 2 + (1 - pY) ** 2
            out.append(f"{f3((Pb - Pe) / (1 - Pe))} (YES {f3(pY)}, n = {len(s)})")
        print(f"| {NAME[a]} | " + " | ".join(out) + " |")

    print("\n## T8 · Persona, prompts pareados por meme (b = só o 1º diz YES; c = só o 2º; McNemar exato)")
    for g in ("1Y5N", "3Y3N", "5Y1N", None):
        cells = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            s = [r for r in sel(rows, g)]
            b = sum(r["persona"][i] == "YES" and r["persona"][j] == "NO" for r in s)
            c = sum(r["persona"][i] == "NO" and r["persona"][j] == "YES" for r in s)
            cells.append(f"S{i + 1}×S{j + 1} {b}/{c} p={mcnemar(b, c):.1e}")
        print(f"| {g or 'todos'} | " + " | ".join(cells) + " |")

    print("\n## T9 · Persona, cada prompt em cada grupo: fração de YES (k) [IC 95% Wilson] e diferença para a taxa humana")
    for g, h in GROUPS:
        for lang in (None, "en", "es"):
            s = sel(rows, g, lang)
            cells = []
            for i in range(3):
                k = sum(r["persona"][i] == "YES" for r in s)
                lo, hi = wilson(k, len(s))
                cells.append(f"{f3(k / len(s))} ({k}) [{f3(lo)}; {f3(hi)}] Δ {k / len(s) - h:+.3f}".replace(".", ","))
            print(f"| {g} | {lang or 'total'} | {len(s)} | {f3(h)} | " + " | ".join(cells) + " |")
    print("acordo com a maioria humana por prompt: NO no 1Y5N | YES no 5Y1N | média")
    for i in range(3):
        for lang in (None, "en", "es"):
            neg, pos = sel(rows, "1Y5N", lang), sel(rows, "5Y1N", lang)
            spec = np.mean([r["persona"][i] == "NO" for r in neg])
            sens = np.mean([r["persona"][i] == "YES" for r in pos])
            print(f"| S{i + 1} | {lang or 'total'} | {f3(spec)} | {f3(sens)} | {f3((spec + sens) / 2)} |")

    print("\n## T7 · cruzamento X × Persona (memes com voto dos dois; b = só X diz YES; c = só Persona diz YES)")
    for x in ("claude", "think_off", "think_on"):
        for g in ("1Y5N", "3Y3N", "5Y1N", None):
            s = [r for r in sel(rows, g) if r[x + "_vote"] and r["persona_vote"]]
            yy = sum(r[x + "_vote"] == "YES" and r["persona_vote"] == "YES" for r in s)
            b = sum(r[x + "_vote"] == "YES" and r["persona_vote"] == "NO" for r in s)
            c = sum(r[x + "_vote"] == "NO" and r["persona_vote"] == "YES" for r in s)
            nn = len(s) - yy - b - c
            po = (yy + nn) / len(s)
            px, pp = (yy + b) / len(s), (yy + c) / len(s)
            pe = px * pp + (1 - px) * (1 - pp)
            print(f"| {g or 'todos'} | {NAME[x]} × Persona | {len(s)} | {f3(po)} | {f3((po - pe) / (1 - pe))} | "
                  f"{yy} | {b} | {c} | {nn} | {mcnemar(b, c):.1e} |")


# ------------------------------------------------------------------ figuras

def grouped(ax, cats, vals, errs=None, fmt="{:.3f}", ylim=1.0):
    """vals[arm] = [valor por categoria]; errs[arm] = [(lo, hi)] opcional."""
    x, w = np.arange(len(cats)), 0.84 / len(ARMS)
    for k, (arm, label, color, hatch) in enumerate(ARMS):
        pos = x - 0.42 + w * (k + 0.5)
        ax.bar(pos, vals[arm], w * 0.9, color=color, hatch=hatch, edgecolor="white", linewidth=0,
               label=label, zorder=3)
        if errs:
            lo = [v - e[0] for v, e in zip(vals[arm], errs[arm])]
            hi = [e[1] - v for v, e in zip(vals[arm], errs[arm])]
            ax.errorbar(pos, vals[arm], yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=4)
        for p, v, e in zip(pos, vals[arm], errs[arm] if errs else [None] * len(cats)):
            top = e[1] if e else v
            ax.text(p, top + ylim * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, ylim)


def fig_voto_yes(rows):
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=BG)
    ax.set_facecolor(BG)
    vals, errs = {}, {}
    for arm, *_ in ARMS:
        vals[arm], errs[arm] = [], []
        for g, _ in GROUPS:
            s = sel(rows, g)
            k = sum(r[arm + "_vote"] == "YES" for r in s)
            vals[arm].append(k / len(s))
            errs[arm].append(wilson(k, len(s)))
    grouped(ax, [f"{g}\n(n = {len(sel(rows, g))})" for g, _ in GROUPS], vals, errs, ylim=1.12)
    for i, (_, h) in enumerate(GROUPS):
        ax.hlines(h, i - 0.46, i + 0.46, colors=HUMAN, linestyles="--", linewidth=2, zorder=5,
                  label="humanos (fração de YES dos 6)" if i == 0 else None)
    style(ax, "Voto YES (maioria das 3 respostas) nos memes extremos · barras = IC 95% de Wilson",
          "fração de memes com voto YES")
    big(ax)
    ax.legend(frameon=False, fontsize=FS, loc="upper left", labelcolor=MUTED)
    save(fig, "persona_voto_yes")


def fig_prompts(rows):
    fig, ax = plt.subplots(2, 1, figsize=(11, 10.5), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
    per_p = {arm: [np.mean([r[arm][i] == "YES" for r in rows if r[arm][i] in ("YES", "NO")]) for i in range(3)]
             for arm, *_ in ARMS}
    grouped(ax[0], ["prompt 1\n(P1 · S1)", "prompt 2\n(P2 · S2)", "prompt 3\n(P3 · S3)"], per_p)
    style(ax[0], f"1 · Fração de YES por prompt ({len(rows)} memes extremos)", "fração de YES")
    big(ax[0])
    agree = {arm: [np.mean([len(set(r[arm])) == 1 and r[arm][0] != "?" for r in sel(rows, g)]) for g, _ in GROUPS]
             for arm, *_ in ARMS}
    grouped(ax[1], [g for g, _ in GROUPS], agree)
    style(ax[1], "2 · Memes em que as 3 respostas concordam (3-0)", "fração de memes")
    big(ax[1])
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper left", ncol=2, frameon=False, fontsize=FS,
               labelcolor=MUTED, bbox_to_anchor=(0.005, 0.995))
    fig.text(0.01, 0.005, "GLM e Claude: paráfrases P1–P3 da mesma definição, na mensagem user.\nPersona: três "
             "system prompts S1–S3 com persona e procedimento diferentes; a posição 1/2/3 não pareia os textos.",
             fontsize=FS, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, "persona_prompts")


def fig_acerto(rows):
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=BG)
    ax.set_facecolor(BG)
    neg, pos = sel(rows, "1Y5N"), sel(rows, "5Y1N")
    vals = {}
    for arm, *_ in ARMS:
        spec = np.mean([r[arm + "_vote"] == "NO" for r in neg])
        sens = np.mean([r[arm + "_vote"] == "YES" for r in pos])
        vals[arm] = [spec, sens, (spec + sens) / 2]
    grouped(ax, [f"NO quando 5 de 6\nhumanos dizem NO\n(1Y5N, n = {len(neg)})",
                 f"YES quando 5 de 6\nhumanos dizem YES\n(5Y1N, n = {len(pos)})",
                 "média das duas"], vals, ylim=1.12)
    style(ax, "Acordo com a maioria humana nos grupos 1Y5N e 5Y1N (descritivo, pós-hoc)", "fração de memes")
    big(ax)
    ax.legend(frameon=False, fontsize=FS, loc="upper left", labelcolor=MUTED)
    save(fig, "persona_acerto")


def fig_cruzamento(rows):
    pc = dict((a, c) for a, _, c, _ in ARMS)
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), facecolor=BG, sharex=True)
    for a, x in zip(ax, ("claude", "think_off", "think_on")):
        a.set_facecolor(BG)
        combos = [("YES", "YES", "#9a9993", "os dois YES"), ("YES", "NO", pc[x], f"só {short(x)} diz YES"),
                  ("NO", "YES", pc["persona"], "só a Persona diz YES"), ("NO", "NO", "#d9d8d3", "os dois NO")]
        for y, (g, _) in enumerate(GROUPS):
            s = [r for r in sel(rows, g) if r[x + "_vote"] and r["persona_vote"]]
            left = 0
            for xv, pv, color, label in combos:
                n = sum(r[x + "_vote"] == xv and r["persona_vote"] == pv for r in s)
                a.barh(y, n / len(s), left=left, color=color, height=0.6, edgecolor=BG, linewidth=2,
                       label=label if y == 0 else None, zorder=3)
                if n / len(s) > 0.035:
                    a.text(left + n / len(s) / 2, y, str(n), ha="center", va="center", fontsize=11,
                           color=INK if color == "#d9d8d3" else "white")
                left += n / len(s)
        a.set_yticks(range(len(GROUPS)), [g for g, _ in GROUPS])
        a.invert_yaxis()
        a.set_xlim(0, 1)
        style(a, f"{NAME[x]} × Persona", "")
        big(a)
        a.grid(axis="x", color=GRID, zorder=0)
        a.grid(axis="y", visible=False)
        a.legend(frameon=False, fontsize=FS, loc="center left", bbox_to_anchor=(1.01, 0.5), labelcolor=MUTED)
    fig.text(0.01, 0.005, "Voto = maioria das 3 respostas; números = memes; largura = fração do grupo.\n"
             "1 meme do 5Y1N sem voto do GLM (imagem recusada) fica fora dos painéis do GLM.",
             fontsize=FS, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "persona_cruzamento")


def fig_por_prompt(rows):
    """Cada prompt da Persona (S1–S3) em cada grupo extremo, com IC de Wilson e a taxa humana."""
    prompts = [("S1", "#6250d6", "xx"), ("S2", "#e87ba4", "//"), ("S3", "#eda100", "..")]
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=BG)
    ax.set_facecolor(BG)
    x, w = np.arange(len(GROUPS)), 0.8 / 3
    for k, (name, color, hatch) in enumerate(prompts):
        pos = x - 0.4 + w * (k + 0.5)
        vals, lo, hi = [], [], []
        for g, _ in GROUPS:
            s = sel(rows, g)
            n = sum(r["persona"][k] == "YES" for r in s)
            a, b = wilson(n, len(s))
            vals.append(n / len(s)); lo.append(n / len(s) - a); hi.append(b - n / len(s))
        ax.bar(pos, vals, w * 0.9, color=color, hatch=hatch, edgecolor="white", linewidth=0, label=name, zorder=3)
        ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        for p_, v, e in zip(pos, vals, hi):
            ax.text(p_, v + e + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK, zorder=6,
                    bbox=dict(facecolor=BG, edgecolor="none", pad=0.5))     # legível sobre o tracejado humano
    for i, (_, h) in enumerate(GROUPS):
        ax.hlines(h, i - 0.46, i + 0.46, colors=HUMAN, linestyles="--", linewidth=2, zorder=5,
                  label="humanos (fração de YES dos 6)" if i == 0 else None)
    ax.set_xticks(x, [f"{g}\n(n = {len(sel(rows, g))})" for g, _ in GROUPS])
    ax.set_ylim(0, 1.05)
    style(ax, "Persona: fração de YES de cada system prompt por grupo · barras = IC 95% de Wilson",
          "fração de YES")
    big(ax)
    ax.legend(frameon=False, fontsize=FS, loc="upper left", labelcolor=MUTED)
    save(fig, "persona_por_prompt")


def short(arm):
    return {"claude": "o Claude", "think_off": "o GLM off", "think_on": "o GLM on"}[arm]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(REPORTS / f"{name}.{ext}", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"gravado: {REPORTS / name}.png (+ .pdf)")


if __name__ == "__main__":
    rows = load()
    assert len(rows) == 1551 and all(len(r[a]) == 3 for r in rows for a in RAW)
    tables(rows)
    fig_voto_yes(rows)
    fig_prompts(rows)
    fig_acerto(rows)
    fig_cruzamento(rows)
    fig_por_prompt(rows)
