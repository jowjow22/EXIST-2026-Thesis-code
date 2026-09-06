# Soft labels ponderadas por θ (Task A) — comparação dos dois braços

Gerado por `reliability/analyze_theta_arms.py` a partir dos CSV de
`EXIST_MEMES/metrics_history/`. Experimento controlado: mede se o θ de
competência por anotador carrega informação quando entra na construção da soft
label de treino. **Não era uma tentativa de melhorar o modelo.**

## 1. Veredito

**NULO — não sustentado.** Na métrica primária (Task A ICM-Soft-Norm), o delta médio
é `+0.0045` (sd `0.0048`, n=3 seeds pareadas),
contra um spread entre seeds do baseline de `0.0107`.
- o efeito (`0.0045`) é **menor que o ruído de seed** (`0.0107`), que era o critério pré-registado
- `p = 0.2499` no t pareado (t = 1.604)
- a métrica companheira (A ICM-Norm hard) troca de sinal (2/3 positivos, média `-0.0027`)

`[MEDIDO]` **Ressalva na direcção oposta:** o delta é positivo em
**6 de 6** observações (as 3 seeds × 2
épocas). Sob um nulo simétrico, as 3 seeds saírem todas positivas tem
probabilidade 1/8 = 12,5% (as duas épocas de uma mesma run não são
independentes, logo contam como 3 observações, não 6). É sugestivo e
**insuficiente** — direcção consistente com magnitude sub-ruído é exactamente
o que um efeito real pequeno E um acaso produzem.

## 2. Métrica primária, seed a seed

| run | baseline (raw) | ponderado (θ) | Δ |
|---|---|---|---|
| seed 42 | 0.4239 | 0.4269 | 0.0031 |
| seed 7 | 0.4203 | 0.4301 | 0.0099 |
| seed 2025 | 0.4309 | 0.4314 | 0.0005 |

média `+0.0045` · sd `0.0048` · spread do baseline `0.0107` · t `1.604` · p `0.2499`

Época 1 (robustez): Δ = `+0.0011` · `+0.0183` · `+0.0021` · média `+0.0072`

## 3. Todas as métricas (época 2, estado final)

| métrica | Δ s42 | Δ s7 | Δ s2025 | Δ médio | sinais + | ruído base |
|---|---|---|---|---|---|---|
| A ICM-Soft-Norm (PRIMÁRIA) | 0.0031 | 0.0099 | 0.0005 | 0.0045 | 3/3 | 0.0107 |
| A ICM-Norm hard | 0.0042 | 0.0125 | -0.0248 | -0.0027 | 2/3 | 0.0517 |
| B ICM-Norm hard | -0.0146 | 0.0143 | -0.0175 | -0.0059 | 1/3 | 0.0224 |
| B ICM-Soft-Norm | -0.0016 | 0.0030 | -0.0040 | -0.0009 | 1/3 | 0.0087 |
| C ICM-Norm hard | -0.0078 | 0.0094 | 0.0086 | 0.0034 | 2/3 | 0.0128 |
| C ICM-Soft-Norm | -0.0004 | 0.0025 | -0.0025 | -0.0001 | 1/3 | 0.0069 |

`[MEDIDO]` Consistência de sinal: 1/2 métricas da Task A com sinal
unânime, contra 0/4 em B e C. **B e C não são evidência** — os alvos
deles não mudaram, qualquer movimento vem do encoder partilhado.

## 4. Auditoria das runs

| ficheiro | arm | seed | n_eval_epochs | n_val_estimado | n_nan_primaria |
|---|---|---|---|---|---|
| 20260830_030421.csv | raw | 42 | 2 | [598, 598] | 0 |
| 20260830_151242.csv | weighted | 42 | 2 | [598, 598] | 0 |
| 20260830_154057_raw_seed7.csv | raw | 7 | 2 | [598, 598] | 0 |
| 20260830_155444_weighted_seed7.csv | weighted | 7 | 2 | [598, 598] | 0 |
| 20260830_163503_raw_seed2025.csv | raw | 2025 | 2 | [598, 598] | 0 |
| 20260830_162416_weighted_seed2025.csv | weighted | 2025 | 2 | [598, 598] | 0 |

As 6 runs têm 2 épocas de eval, n_val ≈ 598 e zero NaN na primária.
Split congelado idêntico nas 6; gold de validação `raw` nas 6.

## 5. O que pode e não pode ser escrito

**Sustentado:**

- O desenho é limpo: split congelado (seeds pareadas), gold de validação fixo em
  `raw` nos dois braços, e só os alvos de treino da Task A mudam.
- O baseline é reproduzido exactamente pelo braço `raw` (`soft_labels_taskA_raw.json` == `get_soft_distribution`, maxΔ = 0).
- A ponderação desloca materialmente os alvos: média `|w−raw|` = 0,0720, e 443 dos
  3.984 itens (11,1%) mudam de lado da maioria. O experimento teve poder.
- O efeito medido na primária é `+0.0045`, **abaixo do ruído de seed**
  (`0.0107`).

**Não sustentado (não escrever):**

- Que ponderar por θ melhora o sistema. O critério pré-registado não foi atingido.
- Que θ **não** carrega informação nenhuma. Com n=3 seeds, o teste não tem poder
  para excluir um efeito de +0,005; a direcção foi consistente. O correcto é
  *inconclusivo com limite superior*, não *ausência de efeito*.
- Qualquer conclusão a partir de B e C.

`[LIMITAÇÃO]` θ vem de EM single-truth, que **define** competência como
concordância com a verdade que ele próprio infere. Sob LeWiDi, um anotador de θ
baixo pode ser uma perspectiva minoritária legítima, não ruído — ponderar por θ
pode estar a apagar exactamente o desacordo que a tarefa quer modelar. O ficheiro
de θ também não convergiu (`converged = false` em 200 iterações).

`[LIMITAÇÃO CRÍTICA]` **As 6 runs são de 2 épocas; o sistema real treina com 30**
(`Thesis_main.ipynb` e `ablation/config.py`). O `2` em `memes_model.ipynb` é
resquício dos smoke tests de 2026-08-04, não a configuração do sistema.

`[MEDIDO]` E o subtreino é directo, não inferido: nas 6 runs o `eval_loss` ainda
**descia** da época 1 para a 2 (−0,042 a −0,093) e a métrica primária ainda
**subia** (+0,024 a +0,044 por época). Nenhuma run se aproximou do platô.

Um delta entre braços de `+0.0045` num regime em que a própria métrica ganha ~+0,03
por época não é interpretável. **O veredito nulo acima vale para 2 épocas, e não
pode ser transportado para o sistema treinado.** Refazer no regime real (30
épocas) custa ~1h40 por run — ~10 h para as 6, ou ~3h20 para um par de controlo.

`[HIPÓTESE]` a direcção consistente na Task A com magnitude sub-ruído é compatível
com um efeito real pequeno. Distinguir exigiria ~10 seeds, não 3. **Não testado.**

