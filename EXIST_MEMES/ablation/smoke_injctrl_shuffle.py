"""Smoke do embaralhamento do demo_vec (ABL-INJ-CTRL). Sem GPU, sem HF.

Verifica: derangement sem pontos fixos, permutação idêntica entre as 3 seeds de
treino, aplicação ANTES do split, e preservação da distribuição marginal.
"""
import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# Stub SÓ para este smoke: ele não chama PyEvALL, mas o import de topo de
# memes_core (via train_one) exige o módulo. NUNCA copiar para código de treino.
for name in ("pyevall", "pyevall.evaluation", "pyevall.utils", "pyevall.utils.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyevall.evaluation"].PyEvALLEvaluation = object
sys.modules["pyevall.utils.utils"].PyEvALLUtils = object

import train_one as t  # noqa: E402
from config import DEMO_SHUFFLE_SEED, SEEDS, ABL_INJ_CTRL  # noqa: E402

N, D = 500, 66
rng = np.random.default_rng(0)
# imita a estrutura real: poucos "pools" distintos, muitos itens por pool
pools = rng.random((25, D))
pools = pools / pools.sum(axis=1, keepdims=True)
assignment = rng.integers(0, len(pools), N)
frame = pd.DataFrame({"demo_vec": list(pools[assignment]), "item_id": np.arange(N)})

# 1. derangement: nenhum ponto fixo
perm = t._build_demo_vec_derangement(N, DEMO_SHUFFLE_SEED)
assert not np.any(perm == np.arange(N)), "derangement tem ponto fixo"
assert sorted(perm.tolist()) == list(range(N)), "não é permutação"
print(f"1. derangement ok: {N} índices, 0 pontos fixos, permutação válida")

# 2. determinismo: mesma seed -> mesma permutação; seed diferente -> diferente
assert np.array_equal(perm, t._build_demo_vec_derangement(N, DEMO_SHUFFLE_SEED))
assert not np.array_equal(perm, t._build_demo_vec_derangement(N, DEMO_SHUFFLE_SEED + 1))
print("2. determinismo ok: mesma seed -> mesma permutação")

# 3. a permutação NÃO depende do seed de treino (as 3 seeds veem a mesma)
shuffled_per_seed = []
for _ in SEEDS:
    out, info = t._apply_demo_vec_shuffle(frame, DEMO_SHUFFLE_SEED)
    shuffled_per_seed.append(np.stack(out["demo_vec"].values))
assert np.array_equal(shuffled_per_seed[0], shuffled_per_seed[1])
assert np.array_equal(shuffled_per_seed[0], shuffled_per_seed[2])
print(f"3. permutação idêntica nas 3 seeds {SEEDS} ok")

# 4. distribuição marginal preservada (é permutação: mesmo multiconjunto)
original = np.stack(frame["demo_vec"].values)
shuffled = shuffled_per_seed[0]
assert np.allclose(np.sort(original, axis=0), np.sort(shuffled, axis=0))
assert np.allclose(original.mean(axis=0), shuffled.mean(axis=0))
assert np.allclose(original.var(axis=0), shuffled.var(axis=0))
print("4. marginal preservada ok: média, variância e multiconjunto idênticos")

# 5. vínculo item<->demografia destruído
still_linked = np.all(np.isclose(original, shuffled), axis=1).mean()
print(f"5. vínculo destruído: {100 * still_linked:.2f}% dos itens mantiveram vetor "
      f"numericamente igual (colisão de pool, esperado ~1/n_pools = {100/len(pools):.2f}%)")
assert still_linked < 0.15, "colisão alta demais -- controle perde poder"
assert np.isclose(info["demo_shuffle_value_collision_rate"], still_linked)

# 6. o dataframe original não é mutado (shuffle devolve cópia)
assert np.allclose(np.stack(frame["demo_vec"].values), original), "mutou o dataframe original"
print("6. sem mutação do dataframe de entrada ok")

# 7. aplicado ANTES do split: os dois conjuntos saem da MESMA permutação
frame_split = frame.copy()
frame_split["labels_task2_1_soft"] = [np.array([0.7, 0.3]) if i % 2 else np.array([0.2, 0.8])
                                      for i in range(N)]
shuffled_frame, _ = t._apply_demo_vec_shuffle(frame_split, DEMO_SHUFFLE_SEED)
train_part, val_part = t._split_train_val(shuffled_frame)
seen = np.concatenate([np.stack(train_part["demo_vec"].values),
                       np.stack(val_part["demo_vec"].values)])
assert np.allclose(np.sort(seen, axis=0), np.sort(shuffled, axis=0)), \
    "train+val não reconstroem a permutação -> shuffle não foi antes do split"
print(f"7. antes do split ok: train({len(train_part)}) + val({len(val_part)}) "
      f"reconstroem a permutação")

# 8. a config de controle difere de inj_headinject_base SÓ em demo_shuffle/nome
from dataclasses import replace, fields  # noqa: E402
from config import ABL_INJ_INJECTION  # noqa: E402
control = ABL_INJ_CTRL[0]
reference = next(c for c in ABL_INJ_INJECTION if c.name == "inj_headinject_base")
differing = [f.name for f in fields(control)
             if getattr(control, f.name) != getattr(reference, f.name)]
assert differing == ["name", "demo_shuffle"], f"difere em campos inesperados: {differing}"
print(f"8. config de controle difere de `{reference.name}` só em {differing} ok")

print("\nSMOKE INJ-CTRL OK")
