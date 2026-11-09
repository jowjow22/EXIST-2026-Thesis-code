"""Doc 29 (SPEC_29.md): mistura do GLM no soft label, viés de YES corrigido por quantis
(`glmq50`), contra o controle de suavização cega (`smooth50`).

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/doc29.py build     # gera e congela os 2 braços (antes das runs)
    python3 EXIST_MEMES/glm_annotator/doc29.py screen    # depois das runs glmq50 e smooth50 seed 42
"""
import csv
import statistics as st
import sys

from common import (REF, REL_OUT, SOFT_RAW, TRAIN_IDS, VAL_IDS, assert_config, load_json,
                    require_gate, sha256_file, update_runtime, write_json)
from doc28 import SCREEN, best
from weights import glm_soft

ALPHA = 0.5
ARMS = {"glmq50": REL_OUT / "soft_labels_taskA_glmq50.json",
        "smooth50": REL_OUT / "soft_labels_taskA_smooth50.json"}


def quantile_map(p_h: dict, p_glm: dict) -> dict:
    """Nível do GLM -> média dos p_h no bloco da fila humana ordenada com o mesmo tamanho e posição."""
    hs, out, pos = sorted(p_h.values()), {}, 0
    for k in sorted(set(p_glm.values())):
        n = sum(v == k for v in p_glm.values())
        out[k] = st.mean(hs[pos:pos + n])
        pos += n
    assert pos == len(hs)
    return out


def labels(raw: dict, train: list, p_glm: dict) -> tuple[dict, dict, dict, float]:
    ids = [i for i in train if p_glm.get(i) is not None]
    p_h = {i: raw[i][0] for i in ids}
    q = quantile_map(p_h, {i: p_glm[i] for i in ids})
    c = st.mean(p_h.values())
    arms = {}
    for arm, ref in (("glmq50", lambda i: q[p_glm[i]]), ("smooth50", lambda i: c)):
        out = dict(raw)
        for i in ids:
            p = ALPHA * p_h[i] + (1 - ALPHA) * ref(i)
            out[i] = [p, 1 - p]
        arms[arm] = out
    return arms["glmq50"], arms["smooth50"], q, c


def summary(raw: dict, new: dict, train: list) -> dict:
    d = [new[i][0] - raw[i][0] for i in train]
    sgn = lambda p: (p > 0.5) - (p < 0.5)
    return {"mean_shift": sum(d) / len(d), "mean_abs_shift": sum(map(abs, d)) / len(d),
            "share_moved": sum(abs(x) >= 1 / 12 - 1e-9 for x in d) / len(d),
            "hard_up": sum(sgn(new[i][0]) > sgn(raw[i][0]) for i in train),
            "hard_down": sum(sgn(new[i][0]) < sgn(raw[i][0]) for i in train)}


def build():
    cfg = assert_config()
    require_gate(cfg, "F3")
    assert not any(p.exists() for p in ARMS.values()), "braços do doc 29 já existem: definição congelada"
    raw, train, val = load_json(SOFT_RAW), load_json(TRAIN_IDS), load_json(VAL_IDS)
    with open(REF / "glm_votes.csv", encoding="utf-8") as f:
        p_glm = {r["meme_id"]: glm_soft([r["label_p1"], r["label_p2"], r["label_p3"]]) for r in csv.DictReader(f)}
    glmq, smooth, q, c = labels(raw, train, p_glm)
    info = {"alpha": ALPHA, "quantile_map": {f"{k:.4f}": v for k, v in q.items()}, "c": c,
            "glm_votes_sha256": sha256_file(REF / "glm_votes.csv"), "spec": "SPEC_29"}
    for arm, out in (("glmq50", glmq), ("smooth50", smooth)):
        assert len(out) == len(raw) and all(out[i] == raw[i] for i in val), f"{arm}: validação difere"
        assert all(0.0 <= v[0] <= 1.0 and abs(sum(v) - 1) < 1e-12 for v in out.values())
        write_json(ARMS[arm], out)
        info[arm] = {"sha256": sha256_file(ARMS[arm]), **summary(raw, out, train)}
        s = info[arm]
        print(f"{arm}: deslocamento {s['mean_shift']:+.4f} · |Δ| {s['mean_abs_shift']:.4f} · "
              f"share {s['share_moved']:.3f} · duros ↑{s['hard_up']} ↓{s['hard_down']}")
    update_runtime(doc29=info)
    print("mapa q:", {round(k, 3): round(v, 3) for k, v in q.items()}, f"· c = {c:.4f}")


def screen(seed: int = 42):
    cfg = assert_config()
    for arm, path in ARMS.items():
        assert sha256_file(path) == cfg["runtime"]["doc29"][arm]["sha256"], f"{arm} ≠ o congelado"
    runs = {a: best(a, seed) for a in ("glmbase", "glmq50", "smooth50")}
    L = [f"# Doc 29 — triagem (seed {seed}), época de menor eval_loss", "",
         " · ".join(f"{a} `{n}` (época {e['epoch']:.0f})" for a, (n, e) in runs.items()), ""]
    hit = False
    for name, (t, b) in {"T1: glmq50 − glmbase": ("glmq50", "glmbase"),
                         "T2: glmq50 − smooth50": ("glmq50", "smooth50"),
                         "(descritivo) smooth50 − glmbase": ("smooth50", "glmbase")}.items():
        L += [f"## {name}", "", f"| métrica | {b} | {t} | Δ | limite | relevante? |", "|---|---|---|---|---|---|"]
        for key, lim in SCREEN.items():
            vb, vt = runs[b][1][key], runs[t][1][key]
            rel = abs(vt - vb) >= lim
            hit |= rel and not name.startswith("(")
            L.append(f"| {key} | {vb:.4f} | {vt:.4f} | {vt - vb:+.4f} | ±{lim} | {'**sim**' if rel else 'não'} |")
        L.append("")
    L += ["**Decisão: " + ("CONTINUAR os dois braços para as seeds 7 e 2025.**" if hit else
                          "ENCERRAR o doc 29 (sem diferença relevante na seed 42).**"),
          "", "A triagem não é teste de significância (SPEC_29 §3). Só T1 e T2 decidem."]
    (REF / "reports" / "doc29_screen.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    update_runtime(doc29={**cfg["runtime"]["doc29"], "screen": {"seed": seed, "continue": hit}})
    print("\n".join(L))


if __name__ == "__main__":
    {"build": build, "screen": screen}[sys.argv[1]]()
