# personaw vs BASE (glmbase), 3 seeds pareadas, 15 épocas

Δ = personaw − glmbase. Validação sempre com os labels originais (raw). t pareado **bilateral**, df = 2 (p < 0,05 exige |t| > 4,30). **Estado primário: E = argmin(eval_loss)**; estado final só como secundário (SPEC_27 §16A, P3/P4).

## E = argmin(eval_loss) — PRIMÁRIO

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média | sd | t | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | +0.0032 | +0.0305 | -0.0215 | +0.0041 | 0.0260 | 0.27 | 0.811 |
| A ICM-Soft-Norm soft | +0.0082 | +0.0131 | +0.0140 | +0.0118 | 0.0031 | 6.49 | 0.023 |
| B ICM-Norm hard | -0.0028 | +0.0007 | +0.0152 | +0.0044 | 0.0095 | 0.79 | 0.511 |
| B ICM-Soft-Norm soft | +0.0012 | -0.0009 | -0.0006 | -0.0001 | 0.0012 | -0.10 | 0.928 |
| C ICM-Norm hard | +0.0144 | -0.0059 | +0.0131 | +0.0072 | 0.0113 | 1.10 | 0.386 |
| C ICM-Soft-Norm soft | -0.0008 | -0.0092 | -0.0037 | -0.0046 | 0.0042 | -1.87 | 0.202 |
| eval_loss (↓ melhor) | +0.0106 | +0.0052 | +0.0106 | +0.0088 | 0.0031 | 4.88 | 0.039 |

Critério (Δ médio de ICM-Soft-Norm > 0, p < 0,05 e eval_loss sem piora), no estado primário: **não satisfeito** — efeito não detectado com 3 seeds (não é evidência de efeito zero).

## estado final — secundário (simetria)

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média | sd | t | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | +0.0032 | +0.0133 | +0.0283 | +0.0149 | 0.0126 | 2.05 | 0.177 |
| A ICM-Soft-Norm soft | -0.0056 | +0.0055 | +0.0085 | +0.0028 | 0.0074 | 0.65 | 0.581 |
| B ICM-Norm hard | -0.0179 | +0.0060 | +0.0097 | -0.0007 | 0.0150 | -0.08 | 0.941 |
| B ICM-Soft-Norm soft | -0.0045 | +0.0016 | +0.0011 | -0.0006 | 0.0034 | -0.31 | 0.787 |
| C ICM-Norm hard | -0.0209 | +0.0056 | +0.0101 | -0.0018 | 0.0167 | -0.18 | 0.872 |
| C ICM-Soft-Norm soft | -0.0114 | -0.0011 | +0.0033 | -0.0031 | 0.0075 | -0.70 | 0.554 |
| eval_loss (↓ melhor) | +0.0686 | +0.0414 | +0.0245 | +0.0448 | 0.0223 | 3.49 | 0.073 |

Estado secundário: só descritivo, não decide nada.

Runs: `20261009_144803_glmbase_seed42.json`, `20261009_164210_glmbase_seed7.json`, `20261011_130127_glmbase_seed2025.json`, `20261029_151046_personaw_seed42.json`, `20261029_161954_personaw_seed7.json`, `20261029_180306_personaw_seed2025.json`
