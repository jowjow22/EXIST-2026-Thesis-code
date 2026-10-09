"""F5 (§10): verificação pré-treino (portão 12.6) e análise pareada GLMW − BASE.

Braços no memes_model.ipynb (célula 6, só trocar ARM e SEED; nada mais muda):
    ARM = 'glmbase'  -> soft_labels_taskA_glmbase.json (cópia byte a byte do raw)
    ARM = 'glmw'     -> soft_labels_taskA_glmw.json   (Task 2.1 de treino ponderada)
    SEED ∈ {42, 7, 2025}, 15 épocas -> 6 runs

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/analyze_f5.py --precheck   # antes de treinar
    python3 EXIST_MEMES/glm_annotator/analyze_f5.py              # depois das 6 runs

No notebook (cwd = EXIST_MEMES), numa célula nova logo depois da que carrega `_SOFT_A`:
    import sys; sys.path.insert(0, 'glm_annotator')
    from analyze_f5 import precheck; precheck(ARM, _SOFT_A)
"""
import argparse
import re

from common import (MEMES_DIR, REPORTS, SOFT_GLMBASE, SOFT_GLMW, SOFT_RAW, TRAIN_IDS, VAL_IDS,
                    assert_config, load_json, require_gate, sha256_file, write_json)
from reliability.analyze_arm_experiment import (METRICS, N_EPOCHS, PRIMARY, SEEDS, best_epoch,
                                                p_two_sided, stats)

ARMS = {"glmbase": SOFT_GLMBASE, "glmw": SOFT_GLMW}
ALL_METRICS = METRICS + [("eval_loss (↓ melhor)", "eval_loss")]


def precheck(arm: str | None = None, loaded: dict | None = None) -> None:
    """Portão 12.6: os arquivos são os gerados na F3 e só a Task 2.1 de treino difere."""
    cfg = assert_config()
    require_gate(cfg, "G4")
    h = cfg["runtime"]["soft_labels_sha256"]
    for name, path in ARMS.items():
        assert sha256_file(path) == h[name], f"{path.name}: SHA-256 ≠ o gravado na F3"
    assert h["glmbase"] == sha256_file(SOFT_RAW), "glmbase deixou de ser cópia do raw"
    base, glmw = load_json(SOFT_GLMBASE), load_json(SOFT_GLMW)
    train, val = set(load_json(TRAIN_IDS)), load_json(VAL_IDS)
    assert set(base) == set(glmw)
    diff = {i for i in base if base[i] != glmw[i]}
    assert diff <= train, f"{len(diff - train)} memes fora do treino diferem"
    assert all(base[i] == glmw[i] for i in val), "validação difere entre braços"
    if arm is not None:
        assert arm in ARMS, f"ARM={arm!r}: use 'glmbase' ou 'glmw'"
        if loaded is not None:
            assert loaded == load_json(ARMS[arm]), f"o dicionário carregado não é {ARMS[arm].name}"
    print(f"portão 12.6 ok · {len(diff)} memes de treino diferem entre glmbase e glmw · "
          f"validação idêntica" + (f" · carregado = {ARMS[arm].name}" if arm else ""))


def discover() -> dict:
    found, dup = {}, []
    for path in sorted((MEMES_DIR / "metrics_history").glob("*.json")):
        m = re.search(r"_(glmbase|glmw)_seed(\d+)\.json$", path.name)
        if not m:
            continue
        ev = [r for r in load_json(path) if "eval_loss" in r]
        if len(ev) != N_EPOCHS:
            print(f"ignorada (≠ {N_EPOCHS} épocas): {path.name}")
            continue
        key = (m[1], int(m[2]))
        if key in found:
            dup.append(path.name)
        else:
            found[key] = {"path": path.name, "evals": ev}
    missing = [k for k in ((a, s) for a in ARMS for s in SEEDS) if k not in found]
    assert not missing, f"faltam runs: {missing}"
    if dup:
        print(f"repetições ignoradas (fica a primeira por ordem de nome): {dup}")
    return found


def paired(found, key, point):
    out = {}
    for s in SEEDS:
        v = {}
        for arm in ARMS:
            ev = found[(arm, s)]["evals"]
            v[arm] = float(ev[best_epoch(ev)[0] if point == "best" else -1][key])
        out[s] = {**v, "delta": v["glmw"] - v["glmbase"]}
    return out


def main():
    precheck()
    found = discover()
    res, L = {}, ["# F5 — GLMW vs BASE (glmbase), 3 seeds pareadas, 15 épocas", "",
                  "Δ = glmw − glmbase. Validação sempre com os labels originais (raw). t pareado "
                  "**bilateral**, df = 2 (p < 0,05 exige |t| > 4,30). **Estado primário: E = "
                  "argmin(eval_loss)**; estado final só como secundário (SPEC_27 §16A, P3/P4).", ""]
    for point, title in (("best", "E = argmin(eval_loss) — PRIMÁRIO"),
                         ("final", "estado final — secundário (simetria)")):
        res[point] = {}
        L += [f"## {title}", "", "| métrica | " + " | ".join(f"Δ seed {s}" for s in SEEDS)
              + " | média | sd | t | p |", "|" + "---|" * (len(SEEDS) + 5)]
        for label, key in ALL_METRICS:
            d = paired(found, key, point)
            st = stats(d)
            st["p"] = p_two_sided(st["t"], st["n"] - 1)
            res[point][key] = {"per_seed": d, **st}
            L.append(f"| {label} | " + " | ".join(f"{d[s]['delta']:+.4f}" for s in SEEDS)
                     + f" | {st['mean']:+.4f} | {st['sd']:.4f} | {st['t']:.2f} | {st['p']:.3f} |")
        if point == "best":                       # o critério só vale no estado primário
            pr, el = res[point][PRIMARY], res[point]["eval_loss"]
            sustained = pr["mean"] > 0 and pr["p"] < 0.05 and el["mean"] <= 0
            res["criterio_11"] = sustained
            L += ["", "Critério (Δ médio de ICM-Soft-Norm > 0, p < 0,05 e eval_loss sem piora), "
                      "no estado primário: " + ("**satisfeito** — evidência exploratória (SPEC_27 §11)."
                                                if sustained else
                                                "**não satisfeito** — efeito não detectado com 3 seeds "
                                                "(não é evidência de efeito zero)."), ""]
        else:
            L += ["", "Estado secundário: só descritivo, não decide nada.", ""]
    L += ["Runs: " + ", ".join(f"`{v['path']}`" for v in found.values())]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "f5_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(REPORTS / "f5_report.json", res)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    precheck() if ap.parse_args().precheck else main()
