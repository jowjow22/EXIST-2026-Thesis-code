"""Braço de controle `glmshift` (SPEC_27 §16C): o mesmo deslocamento médio para o YES que o
GLMW produziu, sem ponderar anotador nenhum.

Gerado ANTES da F5, para congelar a definição (SHA-256 em config.json). Só treina se o GLMW
passar no critério de sucesso no estado primário (§16C).

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/control_glmshift.py
"""
from common import (REF, REL_OUT, SOFT_GLMW, SOFT_RAW, TRAIN_IDS, VAL_IDS, assert_config, load_json,
                    require_gate, sha256_file, update_runtime, write_json)

SOFT_GLMSHIFT = REL_OUT / "soft_labels_taskA_glmshift.json"


def shifted(raw: dict, glmw: dict, train: list) -> tuple[dict, float]:
    """p_ctrl = clip(p_old + Δ̄, 0, 1) nos memes de treino não unânimes; Δ̄ = média de
    (p_glmw − p_old) nesses memes. Unânimes e validação ficam iguais ao raw."""
    mixed = [i for i in train if raw[i][0] not in (0.0, 1.0)]
    delta = sum(glmw[i][0] - raw[i][0] for i in mixed) / len(mixed)
    out = dict(raw)
    for i in mixed:
        p = min(1.0, max(0.0, raw[i][0] + delta))
        out[i] = [p, 1 - p]
    return out, delta


def main():
    cfg = assert_config()
    require_gate(cfg, "G4")
    assert not SOFT_GLMSHIFT.exists(), f"{SOFT_GLMSHIFT.name} já existe: a definição está congelada"
    assert sha256_file(SOFT_GLMW) == cfg["runtime"]["soft_labels_sha256"]["glmw"]
    raw, glmw = load_json(SOFT_RAW), load_json(SOFT_GLMW)
    train, val = load_json(TRAIN_IDS), load_json(VAL_IDS)
    ctrl, delta = shifted(raw, glmw, train)
    assert len(ctrl) == len(raw) and all(ctrl[i] == raw[i] for i in val)
    assert all(ctrl[i] == raw[i] for i in train if raw[i][0] in (0.0, 1.0))
    write_json(SOFT_GLMSHIFT, ctrl)
    n_clip = sum(ctrl[i][0] in (0.0, 1.0) and raw[i][0] not in (0.0, 1.0) for i in train)
    update_runtime(control_glmshift={"sha256": sha256_file(SOFT_GLMSHIFT), "mean_shift": delta,
                                     "clipped": n_clip, "spec": "SPEC_27 §16C"})
    print(f"glmshift gravado · deslocamento médio Δ̄ = {delta:+.4f} nos memes não unânimes · "
          f"cortados em 0/1: {n_clip} · SHA-256 em config.json")


if __name__ == "__main__":
    main()
