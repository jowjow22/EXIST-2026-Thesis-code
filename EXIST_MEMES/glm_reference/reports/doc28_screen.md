# Doc 28 — triagem (seed 42), época de menor eval_loss

BASE `20261009_144803_glmbase_seed42.json` (época 3) · glmw4 `20261011_193429_glmw4_seed42.json` (época 2)

| métrica | base | glmw4 | Δ | limite | relevante? |
|---|---|---|---|---|---|
| eval_task_a_icmnorm_hard | 0.5262 | 0.4484 | -0.0778 | ±0.03 | **sim** |
| eval_task_a_icmsoftnorm_soft | 0.4711 | 0.4219 | -0.0492 | ±0.02 | **sim** |
| eval_loss | 2.6540 | 2.7808 | +0.1269 | ±0.03 | **sim** |

**Decisão: CONTINUAR para as seeds 7 e 2025.**

A triagem não é teste de significância: decide só se vale gastar as outras seeds (SPEC_28 §3).
