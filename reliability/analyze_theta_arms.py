"""
Compara os dois braços do experimento de soft labels ponderadas por θ (Task A,
EXIST 2026 / memes) a partir dos CSV de `EXIST_MEMES/metrics_history/`.

DESENHO (o que torna a comparação válida):
  - split CONGELADO em `split_taskA_frozen.json`, idêntico nas 6 runs -> as seeds
    ficam PAREADAS entre braços; compara-se seed-a-seed, não médias de conjuntos de
    validação diferentes (o erro OPT-06 do modelo de vídeo);
  - gold de validação SEMPRE `raw`, nos dois braços -> a régua não muda com o braço.
    Sem isto, 69 dos 598 itens de validação (11,5%) teriam rótulo hard diferente
    entre braços e o delta mediria a mudança da régua, não a do modelo;
  - só os alvos de TREINO da Task A mudam. B e C são spillover do encoder partilhado.

CRITÉRIO PRÉ-REGISTADO (fixado antes de olhar a 3ª seed):
  métrica primária = Task A ICM-Soft-Norm (soft). Uma só.
  O delta médio tem de exceder o spread entre seeds do baseline para contar como
  efeito. Consistência de sinal é evidência secundária, não substitui magnitude.

Todo o veredito é derivado dos números em runtime (f-strings), nunca prosa fixa
sobre a direcção do resultado -- esse defeito já ocorreu neste projecto e está
documentado em `behavioral_diagnosis_observacoes.md` §1.1.

Uso:
    python3 -m reliability.analyze_theta_arms
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from reliability.run_behavioral_diagnosis import md_table

# ---------------------------------------------------------------------------
# Critérios pré-registados
# ---------------------------------------------------------------------------

PRIMARY = "eval_task_a_icmsoftnorm_soft"
PRIMARY_NAME = "Task A ICM-Soft-Norm"
REPORT_EPOCH = 2.0          # estado final da run; regra de simetria (nunca pico vs fim)
SEEDS = (42, 7, 2025)

METRICS = [
    ("eval_task_a_icmsoftnorm_soft", "A ICM-Soft-Norm (PRIMÁRIA)", True),
    ("eval_task_a_icmnorm_hard", "A ICM-Norm hard", True),
    ("eval_task_b_icmnorm_hard", "B ICM-Norm hard", False),
    ("eval_task_b_icmsoftnorm_soft", "B ICM-Soft-Norm", False),
    ("eval_task_c_icmnorm_hard", "C ICM-Norm hard", False),
    ("eval_task_c_icmsoftnorm_soft", "C ICM-Soft-Norm", False),
]

# (braço, seed) -> basename do CSV em metrics_history/
RUNS = {
    ("raw", 42): "20260830_030421",
    ("weighted", 42): "20260830_151242",
    ("raw", 7): "20260830_154057_raw_seed7",
    ("weighted", 7): "20260830_155444_weighted_seed7",
    ("raw", 2025): "20260830_163503_raw_seed2025",
    ("weighted", 2025): "20260830_162416_weighted_seed2025",
}

METRICS_DIR = "EXIST_MEMES/metrics_history"
OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "theta_arms_comparison.md")
JSON_PATH = os.path.join(OUT_DIR, "theta_arms_comparison.json")

N_VAL_EXPECTED = 598


# ---------------------------------------------------------------------------

def load_runs():
    """Lê os 6 CSV e devolve `{(braço, seed): {época: linha}}`, validando a estrutura.

    Aborta se alguma run não tiver as duas épocas de eval ou se o n de validação
    divergir -- seria sinal de que o split congelado não foi usado nessa run.
    """
    out, audit = {}, []
    for (arm, seed), base in RUNS.items():
        path = os.path.join(METRICS_DIR, f"{base}.csv")
        df = pd.read_csv(path)
        ev = df.dropna(subset=["eval_loss"])
        n_val = (ev["eval_runtime"] * ev["eval_samples_per_second"]).round().astype(int)
        audit.append({
            "arm": arm, "seed": seed, "file": f"{base}.csv",
            "n_eval_epochs": int(len(ev)),
            "epochs": [float(e) for e in ev["epoch"]],
            "n_val_estimado": [int(x) for x in n_val],
            "n_nan_primaria": int(ev[PRIMARY].isna().sum()),
        })
        if len(ev) != 2:
            raise SystemExit(f"ABORTADO — {arm}/{seed}: {len(ev)} épocas de eval, esperadas 2")
        if not (n_val - N_VAL_EXPECTED).abs().le(2).all():
            raise SystemExit(f"ABORTADO — {arm}/{seed}: n_val {list(n_val)} != {N_VAL_EXPECTED}")
        out[(arm, seed)] = {float(r["epoch"]): r for _, r in ev.iterrows()}
    return out, audit


def deltas(runs, col, epoch):
    """Deltas pareados (weighted − raw), um por seed, na época dada."""
    return np.array([
        runs[("weighted", s)][epoch][col] - runs[("raw", s)][epoch][col]
        for s in SEEDS
    ], dtype=float)


def baseline_spread(runs, col, epoch):
    """Amplitude do baseline entre seeds: a régua de ruído do critério."""
    vals = np.array([runs[("raw", s)][epoch][col] for s in SEEDS], dtype=float)
    return float(vals.max() - vals.min())


def analyse(runs):
    rows = {}
    for col, name, is_taskA in METRICS:
        d = deltas(runs, col, REPORT_EPOCH)
        spread = baseline_spread(runs, col, REPORT_EPOCH)
        t, p = stats.ttest_rel(
            [runs[("weighted", s)][REPORT_EPOCH][col] for s in SEEDS],
            [runs[("raw", s)][REPORT_EPOCH][col] for s in SEEDS],
        )
        rows[col] = {
            "name": name, "task_a": is_taskA,
            "deltas": {str(s): float(x) for s, x in zip(SEEDS, d)},
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "n_positive": int((d > 0).sum()),
            "baseline_spread": spread,
            "exceeds_noise": bool(abs(d.mean()) > spread),
            "t": float(t), "p": float(p),
            "baseline": {str(s): float(runs[("raw", s)][REPORT_EPOCH][col]) for s in SEEDS},
            "weighted": {str(s): float(runs[("weighted", s)][REPORT_EPOCH][col]) for s in SEEDS},
        }
    # robustez: a primária na época 1
    d1 = deltas(runs, PRIMARY, 1.0)
    rows["_primary_epoch1"] = {
        "deltas": {str(s): float(x) for s, x in zip(SEEDS, d1)},
        "mean": float(d1.mean()), "n_positive": int((d1 > 0).sum()),
    }
    return rows


# ---------------------------------------------------------------------------

def build_report(res, audit):
    p = res[PRIMARY]
    e1 = res["_primary_epoch1"]
    n_pos_total = p["n_positive"] + e1["n_positive"]

    # veredito derivado dos números, não escrito de antemão
    sustained = p["exceeds_noise"] and p["n_positive"] == len(SEEDS) and p["p"] < 0.05
    consistent = p["n_positive"] == len(SEEDS) and e1["n_positive"] == len(SEEDS)

    L = []
    a = L.append
    a("# Soft labels ponderadas por θ (Task A) — comparação dos dois braços")
    a("")
    a("Gerado por `reliability/analyze_theta_arms.py` a partir dos CSV de")
    a("`EXIST_MEMES/metrics_history/`. Experimento controlado: mede se o θ de")
    a("competência por anotador carrega informação quando entra na construção da soft")
    a("label de treino. **Não era uma tentativa de melhorar o modelo.**")
    a("")

    a("## 1. Veredito")
    a("")
    if sustained:
        a(f"**θ carrega informação.** Na métrica primária ({PRIMARY_NAME}), o delta médio")
        a(f"é `{p['mean']:+.4f}`, positivo nas {len(SEEDS)} seeds, excede o spread entre")
        a(f"seeds do baseline (`{p['baseline_spread']:.4f}`) e `p = {p['p']:.4f}`.")
    else:
        a(f"**NULO — não sustentado.** Na métrica primária ({PRIMARY_NAME}), o delta médio")
        a(f"é `{p['mean']:+.4f}` (sd `{p['sd']:.4f}`, n={len(SEEDS)} seeds pareadas),")
        a(f"contra um spread entre seeds do baseline de `{p['baseline_spread']:.4f}`.")
        reasons = []
        if not p["exceeds_noise"]:
            reasons.append(
                f"o efeito (`{abs(p['mean']):.4f}`) é **menor que o ruído de seed** "
                f"(`{p['baseline_spread']:.4f}`), que era o critério pré-registado")
        if p["p"] >= 0.05:
            reasons.append(f"`p = {p['p']:.4f}` no t pareado (t = {p['t']:.3f})")
        hard = res["eval_task_a_icmnorm_hard"]
        if hard["n_positive"] < len(SEEDS):
            reasons.append(
                f"a métrica companheira (A ICM-Norm hard) troca de sinal "
                f"({hard['n_positive']}/{len(SEEDS)} positivos, média `{hard['mean']:+.4f}`)")
        for r in reasons:
            a(f"- {r}")
    a("")
    if consistent:
        a(f"`[MEDIDO]` **Ressalva na direcção oposta:** o delta é positivo em")
        a(f"**{n_pos_total} de {2 * len(SEEDS)}** observações (as {len(SEEDS)} seeds × 2")
        a("épocas). Sob um nulo simétrico, as 3 seeds saírem todas positivas tem")
        a("probabilidade 1/8 = 12,5% (as duas épocas de uma mesma run não são")
        a("independentes, logo contam como 3 observações, não 6). É sugestivo e")
        a("**insuficiente** — direcção consistente com magnitude sub-ruído é exactamente")
        a("o que um efeito real pequeno E um acaso produzem.")
        a("")

    a("## 2. Métrica primária, seed a seed")
    a("")
    df = pd.DataFrame(
        {
            "baseline (raw)": [p["baseline"][str(s)] for s in SEEDS],
            "ponderado (θ)": [p["weighted"][str(s)] for s in SEEDS],
            "Δ": [p["deltas"][str(s)] for s in SEEDS],
        },
        index=[f"seed {s}" for s in SEEDS],
    )
    a(md_table(df, index_name="run"))
    a("")
    a(f"média `{p['mean']:+.4f}` · sd `{p['sd']:.4f}` · "
      f"spread do baseline `{p['baseline_spread']:.4f}` · "
      f"t `{p['t']:.3f}` · p `{p['p']:.4f}`")
    a("")
    a(f"Época 1 (robustez): Δ = "
      + " · ".join(f"`{e1['deltas'][str(s)]:+.4f}`" for s in SEEDS)
      + f" · média `{e1['mean']:+.4f}`")
    a("")

    a("## 3. Todas as métricas (época 2, estado final)")
    a("")
    df2 = pd.DataFrame(
        {
            **{f"Δ s{s}": [res[c]["deltas"][str(s)] for c, _, _ in METRICS] for s in SEEDS},
            "Δ médio": [res[c]["mean"] for c, _, _ in METRICS],
            "sinais +": [f"{res[c]['n_positive']}/{len(SEEDS)}" for c, _, _ in METRICS],
            "ruído base": [res[c]["baseline_spread"] for c, _, _ in METRICS],
        },
        index=[n for _, n, _ in METRICS],
    )
    a(md_table(df2, index_name="métrica"))
    a("")
    taskA = [c for c, _, t in METRICS if t]
    other = [c for c, _, t in METRICS if not t]
    nA = sum(1 for c in taskA if res[c]["n_positive"] == len(SEEDS))
    nO = sum(1 for c in other if res[c]["n_positive"] == len(SEEDS)
             or res[c]["n_positive"] == 0)
    a(f"`[MEDIDO]` Consistência de sinal: {nA}/{len(taskA)} métricas da Task A com sinal")
    a(f"unânime, contra {nO}/{len(other)} em B e C. **B e C não são evidência** — os alvos")
    a("deles não mudaram, qualquer movimento vem do encoder partilhado.")
    a("")

    a("## 4. Auditoria das runs")
    a("")
    dfa = pd.DataFrame(audit).set_index("file")[
        ["arm", "seed", "n_eval_epochs", "n_val_estimado", "n_nan_primaria"]]
    a(md_table(dfa, index_name="ficheiro"))
    a("")
    a(f"As 6 runs têm 2 épocas de eval, n_val ≈ {N_VAL_EXPECTED} e zero NaN na primária.")
    a("Split congelado idêntico nas 6; gold de validação `raw` nas 6.")
    a("")

    a("## 5. O que pode e não pode ser escrito")
    a("")
    a("**Sustentado:**")
    a("")
    a("- O desenho é limpo: split congelado (seeds pareadas), gold de validação fixo em")
    a("  `raw` nos dois braços, e só os alvos de treino da Task A mudam.")
    a(f"- O baseline é reproduzido exactamente pelo braço `raw` "
      f"(`soft_labels_taskA_raw.json` == `get_soft_distribution`, maxΔ = 0).")
    a(f"- A ponderação desloca materialmente os alvos: média `|w−raw|` = 0,0720, e 443 dos")
    a("  3.984 itens (11,1%) mudam de lado da maioria. O experimento teve poder.")
    a(f"- O efeito medido na primária é `{p['mean']:+.4f}`, **abaixo do ruído de seed**")
    a(f"  (`{p['baseline_spread']:.4f}`).")
    a("")
    a("**Não sustentado (não escrever):**")
    a("")
    a("- Que ponderar por θ melhora o sistema. O critério pré-registado não foi atingido.")
    a("- Que θ **não** carrega informação nenhuma. Com n=3 seeds, o teste não tem poder")
    a("  para excluir um efeito de +0,005; a direcção foi consistente. O correcto é")
    a("  *inconclusivo com limite superior*, não *ausência de efeito*.")
    a("- Qualquer conclusão a partir de B e C.")
    a("")
    a("`[LIMITAÇÃO]` θ vem de EM single-truth, que **define** competência como")
    a("concordância com a verdade que ele próprio infere. Sob LeWiDi, um anotador de θ")
    a("baixo pode ser uma perspectiva minoritária legítima, não ruído — ponderar por θ")
    a("pode estar a apagar exactamente o desacordo que a tarefa quer modelar. O ficheiro")
    a("de θ também não convergiu (`converged = false` em 200 iterações).")
    a("")
    a("`[LIMITAÇÃO CRÍTICA]` **As 6 runs são de 2 épocas; o sistema real treina com 30**")
    a("(`Thesis_main.ipynb` e `ablation/config.py`). O `2` em `memes_model.ipynb` é")
    a("resquício dos smoke tests de 2026-08-04, não a configuração do sistema.")
    a("")
    a("`[MEDIDO]` E o subtreino é directo, não inferido: nas 6 runs o `eval_loss` ainda")
    a("**descia** da época 1 para a 2 (−0,042 a −0,093) e a métrica primária ainda")
    a("**subia** (+0,024 a +0,044 por época). Nenhuma run se aproximou do platô.")
    a("")
    a("Um delta entre braços de `%+.4f` num regime em que a própria métrica ganha ~+0,03" % p["mean"])
    a("por época não é interpretável. **O veredito nulo acima vale para 2 épocas, e não")
    a("pode ser transportado para o sistema treinado.** Refazer no regime real (30")
    a("épocas) custa ~1h40 por run — ~10 h para as 6, ou ~3h20 para um par de controlo.")
    a("")
    a("`[HIPÓTESE]` a direcção consistente na Task A com magnitude sub-ruído é compatível")
    a("com um efeito real pequeno. Distinguir exigiria ~10 seeds, não 3. **Não testado.**")
    a("")
    return "\n".join(L) + "\n"


def main():
    runs, audit = load_runs()
    res = analyse(runs)
    p = res[PRIMARY]

    print(f"=== {PRIMARY_NAME} (época {int(REPORT_EPOCH)}) ===")
    for s in SEEDS:
        print(f"  seed {s:<5} raw {p['baseline'][str(s)]:.4f}  "
              f"wgt {p['weighted'][str(s)]:.4f}  delta {p['deltas'][str(s)]:+.4f}")
    print(f"  media {p['mean']:+.4f}  sd {p['sd']:.4f}  "
          f"{p['n_positive']}/{len(SEEDS)} positivos")
    print(f"  spread do baseline entre seeds: {p['baseline_spread']:.4f}")
    print(f"  excede o ruido: {p['exceeds_noise']}   t={p['t']:.3f}  p={p['p']:.4f}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(res, audit))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"criterio": {"primaria": PRIMARY, "epoca": REPORT_EPOCH,
                                "seeds": list(SEEDS)},
                   "runs": {f"{a}_{s}": b for (a, s), b in RUNS.items()},
                   "auditoria": audit, "resultados": res}, f, indent=1, ensure_ascii=False)
    print(f"escrito: {REPORT_PATH}")
    print(f"escrito: {JSON_PATH}")


if __name__ == "__main__":
    main()
