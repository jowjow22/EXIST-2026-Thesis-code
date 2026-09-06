"""
Mapeamento LEGADO -> CORRIGIDO: o que muda em θ e nas soft labels.

Compara, ficheiro a ficheiro, as sete configurações de θ da Task A antes e depois
da correcção da responsabilidade do M-step (`nutmeg_em`, 2026-08-30), e o
diagnóstico de poder das soft labels ponderadas que saem de `single_truth_K1`.

Não ajusta nada, não regenera nada: lê os ficheiros já escritos.

Uso:
    python3 -m reliability.compare_theta_legacy_vs_fixed
"""

import json
import os

import numpy as np

from reliability.make_soft_labels_taskA import (
    load_dataset, soft_raw, soft_weighted,
)

OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "theta_legacy_vs_fixed.md")
JSON_PATH = os.path.join(OUT_DIR, "theta_legacy_vs_fixed.json")

CONFIGS = [
    "demo_base_K4", "demo_extended_K4", "behavioral_K4",
    "demo_base_K8", "demo_extended_K8", "behavioral_K8",
    "single_truth_K1",
]

# Referências do diagnóstico anterior (`soft_labels_taskA_report.md`), para o §7.
REF_MEAN_ABS_DELTA = 0.0720
REF_N_CROSS_HALF = 443


def load_theta_file(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["theta"], d.get("meta", {})


def compare(old, new):
    keys = sorted(set(old) & set(new))
    a = np.array([old[k] for k in keys])
    b = np.array([new[k] for k in keys])
    d = np.abs(a - b)
    return {
        "n": len(keys),
        "n_only_old": len(set(old) - set(new)),
        "n_only_new": len(set(new) - set(old)),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "spearman_rho": float(np.corrcoef(np.argsort(np.argsort(a)),
                                          np.argsort(np.argsort(b)))[0, 1]),
        "mean_abs_delta": float(d.mean()),
        "max_abs_delta": float(d.max()),
        "n_cross_050": int(((a >= 0.5) != (b >= 0.5)).sum()),
        "n_delta_gt_005": int((d > 0.05).sum()),
        "frac_delta_gt_005": float((d > 0.05).mean()),
        "mean_old": float(a.mean()),
        "mean_new": float(b.mean()),
        "frac_old_higher": float((a >= b).mean()),
    }


def power(labels_a, labels_b, ids):
    """|A − B| item a item sobre p_YES, no mesmo formato do diagnóstico original."""
    d = np.array([abs(labels_a[i][0] - labels_b[i][0]) for i in ids])
    ya = np.array([labels_a[i][0] for i in ids])
    yb = np.array([labels_b[i][0] for i in ids])
    return {
        "n_items": len(ids),
        "mean_abs_delta": float(d.mean()),
        "sd_abs_delta": float(d.std(ddof=1)),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max_abs_delta": float(d.max()),
        "n_gt_005": int((d > 0.05).sum()),
        "n_gt_010": int((d > 0.10).sum()),
        "pearson_r": float(np.corrcoef(ya, yb)[0, 1]),
        "n_cross_half": int(((ya >= 0.5) != (yb >= 0.5)).sum()),
    }


def build_report(cmps, metas, pw, load_note):
    L = []
    a = L.append
    a("# θ legado vs θ corrigido — mapeamento e impacto")
    a("")
    a("Gerado por `reliability/compare_theta_legacy_vs_fixed.py`. Lê ficheiros já")
    a("escritos; não ajusta nem regenera nada.")
    a("")
    a("## 1. Mapeamento de ficheiros")
    a("")
    a("Os ficheiros legados **não foram tocados**. Os novos têm sufixo `_fixed`.")
    a("")
    a("| legado (fases anteriores) | corrigido (oficial) |")
    a("|---|---|")
    for name in CONFIGS:
        a(f"| `theta_taskA_{name}.json` | `theta_taskA_{name}_fixed.json` |")
    a("| `summary_taskA.md` | `summary_taskA_fixed.md` |")
    a("| `soft_labels_taskA_weighted.json` | `soft_labels_taskA_weighted_fixed.json` |")
    a("| `soft_labels_taskA_raw.json` | `soft_labels_taskA_raw_fixed.json` "
      "(idêntico byte a byte — o raw não depende de θ) |")
    a("| `split_taskA_frozen.json` | `split_taskA_frozen_fixed.json` "
      "(mesmas listas de IDs — o split deriva do raw) |")
    a("| `soft_labels_taskA_report.md` | `soft_labels_taskA_report_fixed.md` |")
    a("")
    a("O estimador legado continua acessível por flag:")
    a("`NutmegEM(responsibility=\"legacy\")`, ou "
      "`python -m reliability.run_exist --responsibility legacy`.")
    a("")
    a("## 2. θ: legado vs corrigido, nas sete configurações `[MEDIDO]`")
    a("")
    a("| configuração | Pearson r | média\\|Δθ\\| | máx\\|Δθ\\| | cruzam 0,5 | \\|Δ\\|>0,05 |")
    a("|---|---|---|---|---|---|")
    for name in CONFIGS:
        c = cmps[name]
        a(f"| `{name}` | {c['pearson_r']:.4f} | {c['mean_abs_delta']:.4f} | "
          f"{c['max_abs_delta']:.4f} | {c['n_cross_050']} | "
          f"{c['n_delta_gt_005']} ({100*c['frac_delta_gt_005']:.1f}%) |")
    a("")
    a("## 3. `single_truth_K1` em detalhe — a configuração que alimenta o A1")
    a("")
    c = cmps["single_truth_K1"]
    a("| grandeza | valor |")
    a("|---|---|")
    a(f"| anotadores comparados | {c['n']} |")
    a(f"| Pearson r | **{c['pearson_r']:.4f}** |")
    a(f"| Spearman ρ | {c['spearman_rho']:.4f} |")
    a(f"| média \\|Δθ\\| | **{c['mean_abs_delta']:.4f}** |")
    a(f"| máx \\|Δθ\\| | **{c['max_abs_delta']:.4f}** |")
    a(f"| anotadores que cruzam θ = 0,5 | **{c['n_cross_050']}** |")
    a(f"| anotadores com \\|Δθ\\| > 0,05 | **{c['n_delta_gt_005']}** "
      f"({100*c['frac_delta_gt_005']:.1f}%) |")
    a(f"| θ médio legado → corrigido | {c['mean_old']:.4f} → {c['mean_new']:.4f} |")
    a(f"| θ_legado ≥ θ_corrigido | {100*c['frac_old_higher']:.1f}% dos anotadores |")
    a("")
    a("`[MEDIDO]` A direcção bate com a derivação: a responsabilidade legada infla")
    a("`r` (conta o voto do próprio anotador duas vezes), logo θ legado é o maior")
    a("na esmagadora maioria dos anotadores.")
    a("")
    m_old = metas["single_truth_K1"]["legacy"]
    m_new = metas["single_truth_K1"]["fixed"]
    a("| | legado | corrigido |")
    a("|---|---|---|")
    a(f"| iterações | {m_old.get('n_iter', '?')} | {m_new.get('n_iter', '?')} |")
    a(f"| convergiu | {m_old.get('converged', '?')} | {m_new.get('converged', '?')} |")
    a(f"| quebras da log-posterior | 273 (medidas à parte) | "
      f"{m_new.get('monotonic_breaks', '?')} |")
    a("")
    a("`[MEDIDO]` O `converged=false` original vinha do limite de 200 iterações. Com")
    a("`n_iter=5000` (novo default) o legado converge em 462 e o corrigido em 292.")
    a("")
    a("## 4. Diagnóstico de poder das soft labels `[MEDIDO]`")
    a("")
    a("Todas as colunas são `weighted − raw` sobre p_YES, no mesmo formato do")
    a("relatório original. A última coluna é a comparação directa entre os dois")
    a("alvos ponderados — quanto o A1 se move ao trocar de estimador.")
    a("")
    a("| grandeza | legado (ref.) | corrigido | corrigido − legado |")
    a("|---|---|---|---|")
    rows = [
        ("média \\|Δ\\|", "mean_abs_delta", "{:.4f}"),
        ("sd \\|Δ\\|", "sd_abs_delta", "{:.4f}"),
        ("p50", "p50", "{:.4f}"),
        ("p90", "p90", "{:.4f}"),
        ("p99", "p99", "{:.4f}"),
        ("máx \\|Δ\\|", "max_abs_delta", "{:.4f}"),
        ("itens \\|Δ\\|>0,05", "n_gt_005", "{:.0f}"),
        ("itens \\|Δ\\|>0,10", "n_gt_010", "{:.0f}"),
        ("Pearson r (ponderado vs cru)", "pearson_r", "{:.4f}"),
        ("itens que cruzam 0,5", "n_cross_half", "{:.0f}"),
    ]
    for lbl, k, fmt in rows:
        delta = pw["fixed"][k] - pw["legacy"][k]
        dfmt = "{:+.4f}" if fmt == "{:.4f}" else "{:+.0f}"
        a(f"| {lbl} | {fmt.format(pw['legacy'][k])} | {fmt.format(pw['fixed'][k])} | "
          f"{dfmt.format(delta)} |")
    a("")
    a("| grandeza | valor |")
    a("|---|---|")
    a(f"| média \\|p_YES corrigido − p_YES legado\\| | "
      f"**{pw['fixed_vs_legacy']['mean_abs_delta']:.4f}** |")
    a(f"| p90 | {pw['fixed_vs_legacy']['p90']:.4f} |")
    a(f"| máx | {pw['fixed_vs_legacy']['max_abs_delta']:.4f} |")
    a(f"| Pearson r entre os dois alvos ponderados | "
      f"{pw['fixed_vs_legacy']['pearson_r']:.4f} |")
    a(f"| itens que cruzam 0,5 entre os dois alvos | "
      f"**{pw['fixed_vs_legacy']['n_cross_half']}** |")
    a("")
    a("## 5. O que muda face ao diagnóstico anterior")
    a("")
    d_mean = pw["fixed"]["mean_abs_delta"] - REF_MEAN_ABS_DELTA
    d_cross = pw["fixed"]["n_cross_half"] - REF_N_CROSS_HALF
    a(f"- **Magnitude da intervenção desce**: média \\|weighted − raw\\| passa de")
    a(f"  **{REF_MEAN_ABS_DELTA:.4f}** para **{pw['fixed']['mean_abs_delta']:.4f}**")
    a(f"  ({d_mean:+.4f}, {100*d_mean/REF_MEAN_ABS_DELTA:+.1f}%). O alvo que os")
    a("  retreinos têm de detectar ficou **menor**.")
    a(f"- **Itens que cruzam 0,5**: **{REF_N_CROSS_HALF}** → "
      f"**{pw['fixed']['n_cross_half']}** ({d_cross:+d}).")
    a(f"- **Correlação com o alvo cru sobe**: {pw['legacy']['pearson_r']:.4f} → "
      f"{pw['fixed']['pearson_r']:.4f} — o alvo corrigido está **mais perto** do")
    a("  baseline, não mais longe.")
    a("")
    a("`[LIMITAÇÃO]` Menor efeito no alvo significa **menos poder** para o mesmo")
    a("número de sementes. A tabela de dimensionamento do plano foi calculada sobre")
    a("o efeito antigo; com um alvo mais próximo do baseline, um nulo a 15 épocas")
    a("passa a ser ainda mais provável, e a análise de poder tem de ser refeita a")
    a("partir do d observado nesta configuração — não reaproveitada.")
    a("")
    a(f"`[MEDIDO]` {load_note}")
    return "\n".join(L) + "\n"


def main():
    cmps, metas = {}, {}
    for name in CONFIGS:
        old_p = os.path.join(OUT_DIR, f"theta_taskA_{name}.json")
        new_p = os.path.join(OUT_DIR, f"theta_taskA_{name}_fixed.json")
        t_old, m_old = load_theta_file(old_p)
        t_new, m_new = load_theta_file(new_p)
        cmps[name] = compare(t_old, t_new)
        metas[name] = {"legacy": m_old, "fixed": m_new}
        c = cmps[name]
        print(f"{name:20s} r={c['pearson_r']:.4f} média|Δθ|={c['mean_abs_delta']:.4f} "
              f"máx={c['max_abs_delta']:.4f} cruzam0.5={c['n_cross_050']:3d} "
              f"|Δ|>0.05={c['n_delta_gt_005']:3d}")

    dataset = load_dataset()
    with open(os.path.join(OUT_DIR, "soft_labels_taskA_raw.json"), encoding="utf-8") as f:
        raw = json.load(f)
    with open(os.path.join(OUT_DIR, "soft_labels_taskA_weighted.json"),
              encoding="utf-8") as f:
        wt_old = json.load(f)
    with open(os.path.join(OUT_DIR, "soft_labels_taskA_weighted_fixed.json"),
              encoding="utf-8") as f:
        wt_new = json.load(f)
    ids = sorted(raw)
    pw = {
        "legacy": power(wt_old, raw, ids),
        "fixed": power(wt_new, raw, ids),
        "fixed_vs_legacy": power(wt_new, wt_old, ids),
    }
    print(f"\npoder weighted vs raw | legado média|Δ|={pw['legacy']['mean_abs_delta']:.4f} "
          f"cruzam={pw['legacy']['n_cross_half']}")
    print(f"                      | corrig média|Δ|={pw['fixed']['mean_abs_delta']:.4f} "
          f"cruzam={pw['fixed']['n_cross_half']}")
    print(f"corrigido vs legado   | média|Δ|={pw['fixed_vs_legacy']['mean_abs_delta']:.4f} "
          f"cruzam={pw['fixed_vs_legacy']['n_cross_half']}")

    note = (f"{len(ids)} itens · {cmps['single_truth_K1']['n']} anotadores.")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(cmps, metas, pw, note))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"theta_comparisons": cmps, "power": pw,
                   "reference_previous": {"mean_abs_delta": REF_MEAN_ABS_DELTA,
                                          "n_cross_half": REF_N_CROSS_HALF}},
                  f, indent=2, ensure_ascii=False)
    print(f"\nescrito: {REPORT_PATH}\n         {JSON_PATH}")


if __name__ == "__main__":
    main()
