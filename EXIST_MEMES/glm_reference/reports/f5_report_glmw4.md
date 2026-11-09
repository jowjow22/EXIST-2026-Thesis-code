# glmw4 vs BASE (glmbase), 3 seeds pareadas, 15 épocas

Δ = glmw4 − glmbase. Validação sempre com os labels originais (raw). t pareado **bilateral**, df = 2 (p < 0,05 exige |t| > 4,30). **Estado primário: E = argmin(eval_loss)**; estado final só como secundário (SPEC_27 §16A, P3/P4).

## E = argmin(eval_loss) — PRIMÁRIO

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média | sd | t | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0778 | -0.0350 | -0.0257 | -0.0462 | 0.0278 | -2.88 | 0.102 |
| A ICM-Soft-Norm soft | -0.0492 | -0.0222 | -0.0026 | -0.0247 | 0.0234 | -1.83 | 0.209 |
| B ICM-Norm hard | -0.0470 | +0.0027 | -0.0030 | -0.0158 | 0.0272 | -1.00 | 0.421 |
| B ICM-Soft-Norm soft | -0.0291 | +0.0000 | -0.0048 | -0.0113 | 0.0156 | -1.25 | 0.337 |
| C ICM-Norm hard | -0.0575 | +0.0037 | -0.0175 | -0.0238 | 0.0311 | -1.32 | 0.316 |
| C ICM-Soft-Norm soft | -0.0296 | -0.0020 | -0.0155 | -0.0157 | 0.0138 | -1.97 | 0.187 |
| eval_loss (↓ melhor) | +0.1269 | +0.1014 | +0.0680 | +0.0988 | 0.0295 | 5.79 | 0.029 |

Critério (Δ médio de ICM-Soft-Norm > 0, p < 0,05 e eval_loss sem piora), no estado primário: **não satisfeito** — efeito não detectado com 3 seeds (não é evidência de efeito zero).

## estado final — secundário (simetria)

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média | sd | t | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | +0.0316 | -0.0223 | -0.0265 | -0.0058 | 0.0324 | -0.31 | 0.787 |
| A ICM-Soft-Norm soft | -0.0446 | -0.0510 | -0.0439 | -0.0465 | 0.0039 | -20.49 | 0.002 |
| B ICM-Norm hard | -0.0157 | +0.0017 | -0.0090 | -0.0077 | 0.0088 | -1.51 | 0.270 |
| B ICM-Soft-Norm soft | -0.0115 | -0.0014 | -0.0067 | -0.0066 | 0.0050 | -2.26 | 0.152 |
| C ICM-Norm hard | -0.0226 | +0.0112 | +0.0049 | -0.0022 | 0.0180 | -0.21 | 0.854 |
| C ICM-Soft-Norm soft | -0.0072 | -0.0014 | -0.0064 | -0.0050 | 0.0031 | -2.76 | 0.110 |
| eval_loss (↓ melhor) | +0.2913 | +0.2902 | +0.2838 | +0.2884 | 0.0041 | 123.14 | 0.000 |

Estado secundário: só descritivo, não decide nada.

Runs: `20261009_144803_glmbase_seed42.json`, `20261009_164210_glmbase_seed7.json`, `20261011_130127_glmbase_seed2025.json`, `20261011_193429_glmw4_seed42.json`, `20261011_205755_glmw4_seed2025.json`, `20261011_214926_glmw4_seed7.json`
