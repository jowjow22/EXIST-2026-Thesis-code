"""Doc 28 (SPEC_28.md): pesos extremos r⁴. Gera o braço `glmw4` e faz a triagem com 1 seed.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/doc28.py build     # gera e congela os labels (antes das runs)
    python3 EXIST_MEMES/glm_annotator/doc28.py screen    # depois da run glmw4 seed 42
"""
import csv
import sys

from common import (MEMES_DIR, REF, REL_OUT, SOFT_RAW, TRAIN_IDS, VAL_IDS, assert_config, load_json,
                    load_memes, require_gate, sha256_file, update_runtime, write_json)
from weights import load_train_panels, n_yes, weighted_soft_label

K = 4
SOFT_GLMW4 = REL_OUT / "soft_labels_taskA_glmw4.json"
SCREEN = {"eval_task_a_icmnorm_hard": 0.03, "eval_task_a_icmsoftnorm_soft": 0.02, "eval_loss": 0.03}


def build():
    cfg = assert_config()
    require_gate(cfg, "F3")
    assert cfg["runtime"].get("method") == "doc27_soft_distance"
    assert not SOFT_GLMW4.exists(), f"{SOFT_GLMW4.name} já existe: a definição está congelada"
    memes, train, val = load_memes(), load_json(TRAIN_IDS), load_json(VAL_IDS)
    panels, _ = load_train_panels(memes, train)
    with open(REF / "annotator_weights.csv", encoding="utf-8") as f:
        r = {x["annotator_id"]: float(x["r_glm"]) ** K for x in csv.DictReader(f)}
    raw = load_json(SOFT_RAW)
    out = dict(raw)
    for i in train:
        p = weighted_soft_label(panels[i], r)
        if n_yes(panels[i]) in (0, len(panels[i])):
            assert p == raw[i][0], f"{i}: meme unânime mudou"
        out[i] = [p, 1 - p]
    assert all(out[i] == raw[i] for i in val) and len(out) == len(raw)
    assert all(0.0 <= v[0] <= 1.0 and abs(sum(v) - 1) < 1e-12 for v in out.values())
    write_json(SOFT_GLMW4, out)
    d = [out[i][0] - raw[i][0] for i in train]
    info = {"k": K, "sha256": sha256_file(SOFT_GLMW4),
            "annotator_weights_sha256": sha256_file(REF / "annotator_weights.csv"),
            "mean_shift": sum(d) / len(d), "mean_abs_shift": sum(map(abs, d)) / len(d),
            "share_moved": sum(abs(x) >= 1 / 12 - 1e-9 for x in d) / len(d), "spec": "SPEC_28"}
    update_runtime(doc28=info)
    print(f"glmw4 gravado (r^{K}) · deslocamento médio {info['mean_shift']:+.4f} · "
          f"|Δ| médio {info['mean_abs_shift']:.4f} · share_moved {info['share_moved']:.3f}")


def best(arm: str, seed: int) -> tuple:
    runs = []
    for p in sorted((MEMES_DIR / "metrics_history").glob(f"*_{arm}_seed{seed}.json")):
        ev = [r for r in load_json(p) if "eval_loss" in r]
        if len(ev) == 15:
            runs.append((p.name, ev))
    assert runs, f"nenhuma run completa (15 épocas) de {arm} seed {seed}"
    name, ev = runs[0]
    return name, min(ev, key=lambda r: r["eval_loss"])


def screen(seed: int = 42):
    cfg = assert_config()
    assert sha256_file(SOFT_GLMW4) == cfg["runtime"]["doc28"]["sha256"], "glmw4 ≠ o congelado"
    nb, b = best("glmbase", seed)
    nt, t = best("glmw4", seed)
    L = [f"# Doc 28 — triagem (seed {seed}), época de menor eval_loss", "",
         f"BASE `{nb}` (época {b['epoch']:.0f}) · glmw4 `{nt}` (época {t['epoch']:.0f})", "",
         "| métrica | base | glmw4 | Δ | limite | relevante? |", "|---|---|---|---|---|---|"]
    hit = False
    for key, lim in SCREEN.items():
        delta = t[key] - b[key]
        rel = abs(delta) >= lim
        hit |= rel
        L.append(f"| {key} | {b[key]:.4f} | {t[key]:.4f} | {delta:+.4f} | ±{lim} | {'**sim**' if rel else 'não'} |")
    L += ["", "**Decisão: " + ("CONTINUAR para as seeds 7 e 2025.**" if hit else
                              "ENCERRAR o doc 28 (dose maior sem diferença relevante); seguir para a opção A.**"),
          "", "A triagem não é teste de significância: decide só se vale gastar as outras seeds (SPEC_28 §3)."]
    (REF / "reports").mkdir(parents=True, exist_ok=True)
    (REF / "reports" / "doc28_screen.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    update_runtime(doc28={**cfg["runtime"]["doc28"], "screen": {"seed": seed, "continue": hit}})
    print("\n".join(L))


if __name__ == "__main__":
    {"build": build, "screen": screen}[sys.argv[1]]()
