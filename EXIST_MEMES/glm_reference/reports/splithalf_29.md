# Doc 29 — teste offline split-half (3 anotadores preveem os outros 3)

3381 memes de treino · 20 divisões por meme · q = 0.000→0.000, 0.333→0.124, 0.667→0.303, 1.000→0.756 · c = 0.5549

| previsor da outra metade | MSE ↓ | entropia cruzada ↓ |
|---|---|---|
| só a metade (humano puro) | 0.1193 | 1.1079 |
| 0,5·metade + 0,5·c (= smooth50) | 0.0916 | 0.6112 |
| 0,5·metade + 0,5·GLM sem correção | 0.1062 | 0.8563 |
| 0,5·metade + 0,5·q(GLM) (= glmq50) | 0.0885 | 0.6242 |
| só q(GLM) | 0.1154 | 0.7240 |

glmq50 − smooth50, MSE: -0.0031 (IC 95% [-0.0043; -0.0019])

glmq50 − smooth50, entropia cruzada: +0.0130 (IC 95% [+0.0077; +0.0182])
