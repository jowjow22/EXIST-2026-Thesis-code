"""F5 (§10): verificação pré-treino (portão 12.6) e análise pareada GLMW − BASE.

Braços no memes_model.ipynb (célula 6, só trocar ARM e SEED; nada mais muda):
    ARM = 'glmbase'  -> soft_labels_taskA_glmbase.json (cópia byte a byte do raw)
    ARM = 'glmw'     -> soft_labels_taskA_glmw.json   (Task 2.1 de treino ponderada)
    ARM = 'personaw' -> soft_labels_taskA_personaw.json (doc 33: referência = Persona, persona_ref.py)
    ARM = 'personashift' -> controle do doc 33 (congelado no persona_ref.py build; SPEC_33 §7)
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
SOFT_GLMSHIFT = SOFT_GLMW.with_name("soft_labels_taskA_glmshift.json")   # controle, SPEC_27 §16C
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
    if arm == "glmshift":                                  # controle (SPEC_27 §16C)
        ctrl = cfg["runtime"]["control_glmshift"]
        assert sha256_file(SOFT_GLMSHIFT) == ctrl["sha256"], "glmshift: SHA-256 ≠ o congelado"
        shift = load_json(SOFT_GLMSHIFT)
        assert {i for i in base if base[i] != shift[i]} <= train
        assert all(base[i] == shift[i] for i in val), "validação difere no glmshift"
        if loaded is not None:
            assert loaded == shift, "o dicionário carregado não é o glmshift"
        print(f"controle glmshift ok · Δ̄ = {ctrl['mean_shift']:+.4f}")
        return
    if arm == "glmw4":                                     # doc 28 (SPEC_28)
        path = SOFT_GLMW.with_name("soft_labels_taskA_glmw4.json")
        assert sha256_file(path) == cfg["runtime"]["doc28"]["sha256"], "glmw4: SHA-256 ≠ o congelado"
        w4 = load_json(path)
        assert {i for i in base if base[i] != w4[i]} <= train
        assert all(base[i] == w4[i] for i in val), "validação difere no glmw4"
        if loaded is not None:
            assert loaded == w4, "o dicionário carregado não é o glmw4"
        print(f"glmw4 ok · r^4 · Δ̄ = {cfg['runtime']['doc28']['mean_shift']:+.4f}")
        return
    if arm == "personaw":                                  # doc 33 (SPEC_33)
        assert cfg["runtime"]["gates"].get("doc33_G4", {}).get("pass"), "portão G4 do doc 33 não aprovado"
        path = SOFT_GLMW.with_name("soft_labels_taskA_personaw.json")
        assert sha256_file(path) == cfg["runtime"]["doc33"]["sha256"], "personaw: SHA-256 ≠ o congelado"
        pw = load_json(path)
        assert {i for i in base if base[i] != pw[i]} <= train
        assert all(base[i] == pw[i] for i in val), "validação difere no personaw"
        if loaded is not None:
            assert loaded == pw, "o dicionário carregado não é o personaw"
        print(f"personaw ok · referência Persona · Δ̄ = {cfg['runtime']['doc33']['build']['mean_shift']:+.4f}")
        return
    if arm == "personashift":                              # controle do doc 33 (SPEC_33 §7)
        assert cfg["runtime"]["gates"].get("doc33_G4", {}).get("pass"), "portão G4 do doc 33 não aprovado"
        ctrl = cfg["runtime"]["doc33"]["control"]
        assert ctrl["runs"], f"|Δ̄| = {abs(ctrl['mean_shift']):.4f} < 1/24: o controle não roda (SPEC_33 §7)"
        path = SOFT_GLMW.with_name("soft_labels_taskA_personashift.json")
        assert sha256_file(path) == ctrl["sha256"], "personashift: SHA-256 ≠ o congelado"
        ps = load_json(path)
        assert {i for i in base if base[i] != ps[i]} <= train
        assert all(base[i] == ps[i] for i in val), "validação difere no personashift"
        if loaded is not None:
            assert loaded == ps, "o dicionário carregado não é o personashift"
        print(f"controle personashift ok · Δ̄ = {ctrl['mean_shift']:+.4f}")
        return
    if arm is not None:
        assert arm in ARMS, f"ARM={arm!r}: use 'glmbase', 'glmw' ou 'glmshift'"
        if loaded is not None:
            assert loaded == load_json(ARMS[arm]), f"o dicionário carregado não é {ARMS[arm].name}"
    print(f"portão 12.6 ok · {len(diff)} memes de treino diferem entre glmbase e glmw · "
          f"validação idêntica" + (f" · carregado = {ARMS[arm].name}" if arm else ""))


def discover(treat: str = "glmw", base: str = "glmbase") -> dict:
    found, dup = {}, []
    for path in sorted((MEMES_DIR / "metrics_history").glob("*.json")):
        m = re.search(rf"_({base}|{treat})_seed(\d+)\.json$", path.name)
        if not m:
            continue
        ev = [r for r in load_json(path) if "eval_loss" in r]
        if len(ev) != N_EPOCHS:
            print(f"ignorada (≠ {N_EPOCHS} épocas): {path.name}")
            continue
        key = (m[1], int(m[2]))
        if key in found:                 # repetição: o treino é determinístico, tem de dar a mesma eval_loss
            first = [r["eval_loss"] for r in found[key]["evals"]]
            assert [r["eval_loss"] for r in ev] == first, (
                f"{path.name} não reproduz {found[key]['path']} (eval_loss difere): o ambiente mudou. "
                "Ver SPEC_33 §7: mover as runs antigas desse braço para fora de metrics_history/ e retreinar.")
            dup.append(path.name)
        else:
            found[key] = {"path": path.name, "evals": ev}
    missing = [k for k in ((a, s) for a in (base, treat) for s in SEEDS) if k not in found]
    assert not missing, f"faltam runs: {missing}"
    if dup:
        print(f"repetições com eval_loss idêntica à primeira (ficam de fora): {dup}")
    return found


def paired(found, key, point, treat: str = "glmw", base: str = "glmbase"):
    out = {}
    for s in SEEDS:
        v = {}
        for arm in (base, treat):
            ev = found[(arm, s)]["evals"]
            v[arm] = float(ev[best_epoch(ev)[0] if point == "best" else -1][key])
        out[s] = {**v, "delta": v[treat] - v[base]}
    return out


def main(treat: str = "glmw", base: str = "glmbase"):
    precheck()
    if "persona" in treat + base:               # doc 33: o portão próprio tem de estar aprovado
        assert assert_config()["runtime"]["gates"].get("doc33_G4", {}).get("pass"), "portão G4 do doc 33 não aprovado"
    if treat != "glmw":                         # braços seguintes (doc 28: glmw4)
        cfg = assert_config()
        rt = cfg["runtime"]
        frozen = {"glmw4": rt.get("doc28", {}).get("sha256"), "personaw": rt.get("doc33", {}).get("sha256"),
                  "personashift": rt.get("doc33", {}).get("control", {}).get("sha256"),
                  **{a: rt.get("doc29", {}).get(a, {}).get("sha256") for a in ("glmq50", "smooth50")}}
        for arm in {treat, base} - {"glmbase"}:
            assert frozen.get(arm), f"braço {arm!r} sem registro em config.json"
            path = SOFT_GLMW.with_name(f"soft_labels_taskA_{arm}.json")
            assert sha256_file(path) == frozen[arm], f"{path.name}: SHA-256 ≠ o congelado"
    found = discover(treat, base)
    res, L = {}, [f"# {treat} vs " + ("BASE (glmbase)" if base == "glmbase" else base) + ", 3 seeds pareadas, 15 épocas", "",
                  f"Δ = {treat} − {base}. Validação sempre com os labels originais (raw). t pareado "
                  "**bilateral**, df = 2 (p < 0,05 exige |t| > 4,30). **Estado primário: E = "
                  "argmin(eval_loss)**; estado final só como secundário (SPEC_27 §16A, P3/P4).", ""]
    for point, title in (("best", "E = argmin(eval_loss) — PRIMÁRIO"),
                         ("final", "estado final — secundário (simetria)")):
        res[point] = {}
        L += [f"## {title}", "", "| métrica | " + " | ".join(f"Δ seed {s}" for s in SEEDS)
              + " | média | sd | t | p |", "|" + "---|" * (len(SEEDS) + 5)]
        for label, key in ALL_METRICS:
            d = paired(found, key, point, treat, base)
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
    stem = "f5_report" if treat == "glmw" else f"f5_report_{treat}" + ("" if base == "glmbase" else f"_vs_{base}")
    (REPORTS / f"{stem}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(REPORTS / f"{stem}.json", res)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--treat", default="glmw",
                    help="braço tratado: glmw (doc 27), glmw4 (doc 28), glmq50/smooth50 (doc 29), personaw (doc 33)")
    # controle do doc 33: --treat personaw --base personashift (SPEC_33 §7)
    ap.add_argument("--base", default="glmbase", help="braço de comparação (doc 29: smooth50)")
    a = ap.parse_args()
    precheck() if a.precheck else main(a.treat, a.base)
