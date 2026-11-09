# glmw vs BASE (glmbase), 3 seeds pareadas, 15 épocas

Δ = glmw − glmbase. Validação sempre com os labels originais (raw). t pareado **bilateral**, df = 2 (p < 0,05 exige |t| > 4,30). **Estado primário: E = argmin(eval_loss)**; estado final só como secundário (SPEC_27 §16A, P3/P4).

## E = argmin(eval_loss) — PRIMÁRIO

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média | sd | t | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | +0.0160 | -0.0079 | -0.0019 | +0.0021 | 0.0125 | 0.29 | 0.802 |
| A ICM-Soft-Norm soft | +0.0040 | +0.0057 | +0.0126 | +0.0074 | 0.0046 | 2.81 | 0.107 |
| B ICM-Norm hard | +0.0002 | -0.0024 | +0.0033 | +0.0004 | 0.0029 | 0.22 | 0.847 |
| B ICM-Soft-Norm soft | +0.0007 | +0.0013 | -0.0034 | -0.0005 | 0.0025 | -0.32 | 0.781 |
| C ICM-Norm hard | +0.0077 | +0.0095 | -0.0048 | +0.0041 | 0.0078 | 0.92 | 0.456 |
| C ICM-Soft-Norm soft | +0.0014 | +0.0017 | -0.0042 | -0.0004 | 0.0033 | -0.20 | 0.860 |
| eval_loss (↓ melhor) | +0.0148 | +0.0096 | +0.0035 | +0.0093 | 0.0057 | 2.83 | 0.105 |

Critério (Δ médio de ICM-Soft-Norm > 0, p < 0,05 e eval_loss sem piora), no estado primário: **não satisfeito** — efeito não detectado com 3 seeds (não é evidência de efeito zero).

## estado final — secundário (simetria)

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média | sd | t | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | +0.0262 | -0.0152 | -0.0012 | +0.0032 | 0.0211 | 0.27 | 0.815 |
| A ICM-Soft-Norm soft | -0.0072 | -0.0093 | +0.0009 | -0.0052 | 0.0054 | -1.67 | 0.237 |
| B ICM-Norm hard | +0.0037 | -0.0083 | -0.0003 | -0.0016 | 0.0061 | -0.46 | 0.688 |
| B ICM-Soft-Norm soft | -0.0045 | -0.0022 | +0.0018 | -0.0016 | 0.0032 | -0.90 | 0.465 |
| C ICM-Norm hard | -0.0058 | +0.0065 | +0.0112 | +0.0040 | 0.0088 | 0.78 | 0.516 |
| C ICM-Soft-Norm soft | -0.0036 | -0.0037 | +0.0047 | -0.0009 | 0.0048 | -0.32 | 0.781 |
| eval_loss (↓ melhor) | +0.0536 | +0.0500 | +0.0261 | +0.0432 | 0.0149 | 5.01 | 0.038 |

Estado secundário: só descritivo, não decide nada.

Runs: `20261009_144803_glmbase_seed42.json`, `20261009_154608_glmw_seed42.json`, `20261009_164210_glmbase_seed7.json`, `20261011_120419_glmw_seed7.json`, `20261011_130127_glmbase_seed2025.json`, `20261011_140034_glmw_seed2025.json`
