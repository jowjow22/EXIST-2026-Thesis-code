# Ablação de anotadores na Task A — A0 / R1 / R2, 15 épocas, 3 sementes pareadas

Gerado por `reliability/analyze_ablation_experiment.py` a partir dos logs de
`EXIST_MEMES/metrics_history/`. Não treina, não regenera labels.

## 0. Proveniência e ordem dos factos

- Runs concluídas em **2026-09-05** (timestamps nos nomes dos ficheiros de log).
- Critério decisivo fixado em **2026-09-06**, na especificação da tarefa, e
  codificado neste script **antes** de qualquer valor de métrica de R1/R2 ser
  lido. As verificações feitas antes disso foram: existência e integridade das
  runs, e o deslocamento do alvo nos ficheiros de labels — nenhuma métrica ICM.
- Braços e ficheiros de alvo consumidos pelo treino:
  - **A0_baseline** — RUN_TAG `raw`, alvo `soft_labels_taskA_raw.json`
  - **R1_rem_extremos** — RUN_TAG `R1_rem_extremos`, alvo `soft_labels_taskA_R1_rem_extremos.json`
  - **R2_rem_controle** — RUN_TAG `R2_rem_controle`, alvo `soft_labels_taskA_R2_rem_controle.json`

## 1. Critério pré-registado

**Decisiva: R1 − R2, Task A ICM-Soft-Norm (soft), na época E = argmin(eval_loss).**

`R1 − A0` e `R2 − A0` são **contexto, não evidência de efeito de critério**.
Ambas misturam duas coisas: o critério pelo qual os anotadores foram removidos
**e** a remoção de densidade (o item passa a ter menos votos). O controlo
emparelhado R2 existe precisamente para tirar a densidade da equação: em
`R1 − R2` a remoção de densidade está presente **dos dois lados** e cancela-se
no pareamento; o que sobra é o critério de selecção.

**Intervenção:** apenas o alvo de treino da Task A. Os anotadores removidos
saem **só** do alvo da Task A — continuam a contar em B e C. O que se observa
em B e C é **spillover do encoder partilhado**, não efeito directo.

## 2. Integridade das runs `[MEDIDO]`

- [PASS] as 9 runs (3 braços × 3 seeds) existem — todas presentes
- [PASS] cada run tem 15 avaliações — raw/42=15, raw/7=15, raw/2025=15, R1_rem_extremos/42=15, R1_rem_extremos/7=15, R1_rem_extremos/2025=15, R2_rem_controle/42=15, R2_rem_controle/7=15, R2_rem_controle/2025=15
- [PASS] cada run terminou (`train_runtime` no log) — todas com on_train_end
- [PASS] as 6 métricas ICM estão em todas as runs — ok
- [PASS] A0_baseline == alvo consumido pelas runs `raw` (em float32, que é o dtype que entra no treino) — 3984 itens; 1966 diferem em float64 (max |Δ| = 1.11e-16, max |Δ p_YES| = 0.00e+00); idênticos em float32: True

As runs `raw` reutilizadas como A0 são as mesmas do estudo A0/A1
(`experiment_taskA_15ep_report.md`) — não houve retreino do baseline.

| braço | seed | ficheiro | E = argmin(eval_loss) | eval_loss em E |
|---|---|---|---|---|
| `A0_baseline` | 42 | `20260831_065521_raw_seed42.json` | **3** | 2.65397 |
| `A0_baseline` | 7 | `20260831_075933_raw_seed7.json` | **3** | 2.67339 |
| `A0_baseline` | 2025 | `20260830_225133_raw_seed2025.json` | **3** | 2.67177 |
| `R1_rem_extremos` | 42 | `20260905_112128_R1_rem_extremos_seed42.json` | **3** | 2.66661 |
| `R1_rem_extremos` | 7 | `20260905_122152_R1_rem_extremos_seed7.json` | **3** | 2.68292 |
| `R1_rem_extremos` | 2025 | `20260905_140756_R1_rem_extremos_seed2025.json` | **3** | 2.68698 |
| `R2_rem_controle` | 42 | `20260905_150442_R2_rem_controle_seed42.json` | **3** | 2.66191 |
| `R2_rem_controle` | 7 | `20260905_155802_R2_rem_controle_seed7.json` | **3** | 2.67600 |
| `R2_rem_controle` | 2025 | `20260905_222414_R2_rem_controle_seed2025.json` | **4** | 2.67633 |

`[MEDIDO]` E não é o mesmo em todas as runs: `R2_rem_controle`/2025 tem E = 4, as restantes E = 3. Cada braço é lido no seu
próprio E, como o critério pré-registado manda — a alternativa (forçar a
mesma época nos dois lados) escolheria a época por conveniência de
comparação, que é a porta de entrada do leakage que a regra fecha.

## 3. O alvo, antes do treino `[MEDIDO]`

### 3.1 A ablação fragmenta a grade do alvo

| braço | valores distintos possíveis no alvo |
|---|---|
| A0_baseline | 7 |
| R1_rem_extremos | 13 |
| R2_rem_controle | 13 |

A0 tem **7** valores distintos (grade de 6 votos:
0, 1/6, …, 1). R1 tem **13** e R2 tem **13**, porque os itens ficam com densidades
**mistas** — 6, 5 ou 4 votos, conforme quantos dos anotadores removidos
tocavam aquele item — e cada densidade traz a sua própria grade (1/5, 1/4, …).

Isto **contamina qualquer leitura contra o A0**: parte do Δ vem de o alvo ter
passado a viver numa grade diferente, não do critério de remoção. É mais uma
razão para a comparação decisiva ser R1 vs R2, onde a grade é a mesma dos dois
lados.

`[LIMITAÇÃO]` O que está verificado é a **grade de valores** (as duas iguais, 13 valores). O **histograma de densidades** (quantos itens ficam com 6, 5 ou 4
votos em cada braço) **não foi verificado** — o gerador dos ficheiros de
ablação vive no notebook `EXIST_MEMES/reliability_tests.ipynb` e não deixou
manifesto em disco. Se essa distribuição diferir entre R1 e R2, sobra um
resíduo de densidade na comparação decisiva. Fica como item a fechar.

### 3.2 Deslocamento do alvo face ao A0

| braço | itens afectados | \|Δ\| médio (afectados) | \|Δ\| médio (todos) | cruzam 0,5 |
|---|---|---|---|---|
| R1_rem_extremos | 998 | 0.1023 | 0.0256 | 150 |
| R2_rem_controle | 1054 | 0.0866 | 0.0229 | 114 |

Diferença entre os dois deslocamentos médios: **0.0157**.

### 3.3 Distância directa R1 ↔ R2 no alvo

A diferença dos deslocamentos médios (3.2) **não** é a distância entre os dois
alvos: dois braços podem afastar-se do A0 na mesma magnitude e em direcções
diferentes. A quantidade que limita o que `R1 − R2` pode mostrar é a distância
directa entre eles:

- itens em que R1 e R2 diferem: **1653** de 3984
- \|Δ\| médio sobre todos os itens: **0.0445**
- \|Δ\| médio sobre os afectados: **0.1073**
- itens que cruzam 0,5 entre R1 e R2: **232**

## 4. Resultado decisivo — R1 − R2, época E `[MEDIDO]`

Métrica pré-registada: **Task A ICM-Soft-Norm (soft)**.

| seed | R1 | R2 | Δ = R1 − R2 |
|---|---|---|---|
| 42 | 0.4621 | 0.4662 | **-0.0041** |
| 7 | 0.4570 | 0.4599 | **-0.0029** |
| 2025 | 0.4546 | 0.4706 | **-0.0160** |
| **média** | | | **-0.0077** |

sd(Δ) = 0.0072 · t(2) = -1.84 · p = 0.206 · d_z = -1.06 · mesmo sinal em 3/3: sim

**Veredito (critério pré-registado): NULO — não sustentado.**

## 5. As três comparações, na época E `[MEDIDO]`

### 5.1 R1 − R2 (**DECISIVA**)

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0075 | +0.0017 | -0.0397 | **-0.0152** | 0.0218 | -0.70 | 0.351 |
| A ICM-Soft-Norm soft | -0.0041 | -0.0029 | -0.0160 | **-0.0077** | 0.0072 | -1.06 | 0.206 |
| B ICM-Norm hard | +0.0095 | -0.0093 | +0.0144 | **+0.0049** | 0.0125 | +0.39 | 0.570 |
| B ICM-Soft-Norm soft | -0.0034 | +0.0020 | -0.0036 | **-0.0017** | 0.0032 | -0.53 | 0.456 |
| C ICM-Norm hard | -0.0101 | +0.0234 | -0.0392 | **-0.0086** | 0.0314 | -0.28 | 0.681 |
| C ICM-Soft-Norm soft | -0.0050 | +0.0008 | -0.0345 | **-0.0129** | 0.0190 | -0.68 | 0.360 |

### 5.2 R1 − A0 (contexto — não é evidência de efeito de critério)

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0133 | +0.0025 | -0.0388 | **-0.0166** | 0.0209 | -0.79 | 0.303 |
| A ICM-Soft-Norm soft | -0.0090 | -0.0041 | -0.0061 | **-0.0064** | 0.0025 | -2.55 | 0.048 |
| B ICM-Norm hard | -0.0040 | -0.0169 | +0.0150 | **-0.0020** | 0.0160 | -0.12 | 0.852 |
| B ICM-Soft-Norm soft | -0.0043 | +0.0010 | -0.0014 | **-0.0016** | 0.0027 | -0.59 | 0.417 |
| C ICM-Norm hard | -0.0075 | +0.0171 | +0.0093 | **+0.0063** | 0.0126 | +0.50 | 0.478 |
| C ICM-Soft-Norm soft | -0.0056 | -0.0028 | -0.0083 | **-0.0056** | 0.0028 | -2.00 | 0.074 |

### 5.3 R2 − A0 (contexto — não é evidência de efeito de critério)

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0059 | +0.0008 | +0.0009 | **-0.0014** | 0.0039 | -0.36 | 0.597 |
| A ICM-Soft-Norm soft | -0.0049 | -0.0011 | +0.0099 | **+0.0013** | 0.0077 | +0.17 | 0.799 |
| B ICM-Norm hard | -0.0135 | -0.0076 | +0.0005 | **-0.0068** | 0.0070 | -0.97 | 0.234 |
| B ICM-Soft-Norm soft | -0.0009 | -0.0010 | +0.0021 | **+0.0001** | 0.0018 | +0.06 | 0.926 |
| C ICM-Norm hard | +0.0026 | -0.0064 | +0.0485 | **+0.0149** | 0.0294 | +0.51 | 0.473 |
| C ICM-Soft-Norm soft | -0.0006 | -0.0036 | +0.0262 | **+0.0073** | 0.0164 | +0.45 | 0.520 |

`[LIMITAÇÃO]` **Multiplicidade.** A tabela completa deste relatório tem
3 comparações × 6 métricas × 2 pontos de leitura = 36 células. **Uma** é decisiva (R1 − R2, Task A
ICM-Soft-Norm, época E). Qualquer p < 0,05 avulso nas outras 35 é o que se
espera por acaso e não deve ser lido como achado.

## 6. Regra de simetria — estado FINAL (época 15) `[MEDIDO]`

O mesmo critério aplicado a todos os braços. Reportado porque a definição de
baseline adoptada na Task 3 é o estado final da run, e misturar «fim de um
lado, pico do outro» é o erro que essa regra existe para impedir.

### 6.1 R1 − R2

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0232 | -0.0016 | -0.0041 | **-0.0096** | 0.0118 | -0.82 | 0.293 |
| A ICM-Soft-Norm soft | -0.0061 | -0.0014 | -0.0087 | **-0.0054** | 0.0037 | -1.45 | 0.128 |
| B ICM-Norm hard | +0.0097 | -0.0046 | -0.0006 | **+0.0015** | 0.0074 | +0.20 | 0.761 |
| B ICM-Soft-Norm soft | -0.0009 | -0.0022 | -0.0012 | **-0.0014** | 0.0007 | -2.15 | 0.065 |
| C ICM-Norm hard | -0.0089 | +0.0279 | -0.0194 | **-0.0001** | 0.0248 | -0.00 | 0.994 |
| C ICM-Soft-Norm soft | -0.0007 | +0.0023 | -0.0025 | **-0.0003** | 0.0024 | -0.13 | 0.839 |

### 6.2 R1 − A0

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0191 | -0.0008 | -0.0058 | **-0.0085** | 0.0095 | -0.90 | 0.258 |
| A ICM-Soft-Norm soft | -0.0115 | -0.0060 | -0.0071 | **-0.0082** | 0.0029 | -2.79 | 0.040 |
| B ICM-Norm hard | -0.0055 | +0.0004 | +0.0056 | **+0.0002** | 0.0056 | +0.03 | 0.966 |
| B ICM-Soft-Norm soft | -0.0040 | -0.0047 | +0.0010 | **-0.0026** | 0.0031 | -0.84 | 0.284 |
| C ICM-Norm hard | -0.0160 | +0.0144 | -0.0183 | **-0.0066** | 0.0182 | -0.36 | 0.593 |
| C ICM-Soft-Norm soft | -0.0058 | -0.0020 | +0.0003 | **-0.0025** | 0.0031 | -0.80 | 0.301 |

### 6.3 R2 − A0

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | +0.0041 | +0.0008 | -0.0016 | **+0.0011** | 0.0029 | +0.38 | 0.580 |
| A ICM-Soft-Norm soft | -0.0054 | -0.0046 | +0.0017 | **-0.0028** | 0.0039 | -0.72 | 0.339 |
| B ICM-Norm hard | -0.0152 | +0.0050 | +0.0062 | **-0.0013** | 0.0120 | -0.11 | 0.866 |
| B ICM-Soft-Norm soft | -0.0031 | -0.0025 | +0.0021 | **-0.0012** | 0.0028 | -0.41 | 0.551 |
| C ICM-Norm hard | -0.0071 | -0.0135 | +0.0010 | **-0.0065** | 0.0073 | -0.89 | 0.261 |
| C ICM-Soft-Norm soft | -0.0051 | -0.0042 | +0.0028 | **-0.0022** | 0.0044 | -0.50 | 0.481 |

`[LIMITAÇÃO]` **Nada nesta secção é um achado.** O mínimo de `eval_loss` cai
na época 3/4 nas nove runs; a época 15 está em sobreajuste profundo. A secção existe para cumprir
a regra de simetria, não para inferência.
A célula com p < 0,05 aqui — R1 − A0 / A ICM-Soft-Norm soft (p = 0.040) — é exactamente o tipo de célula que a multiplicidade explica (§5).

## 7. Curvas de `eval_loss` `[MEDIDO]`

| braço | seed | ép 1 | ép 2 | ép 3 | ép 4 | ép 5 | ép 6 | ép 7 | ép 8 | ép 9 | ép 10 | ép 11 | ép 12 | ép 13 | ép 14 | ép 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A0_baseline` | 42 | 2.8221 | 2.6922 | 2.6540 | 2.6685 | 2.6934 | 2.6893 | 2.6933 | 2.6863 | 2.7176 | 2.7200 | 2.7159 | 2.7344 | 2.7356 | 2.7459 | 2.7439 |
| `A0_baseline` | 7 | 2.8704 | 2.7011 | 2.6734 | 2.6923 | 2.7229 | 2.7331 | 2.7312 | 2.7407 | 2.7472 | 2.7642 | 2.7679 | 2.7871 | 2.8042 | 2.8009 | 2.8083 |
| `A0_baseline` | 2025 | 2.8533 | 2.6914 | 2.6718 | 2.6857 | 2.7237 | 2.7360 | 2.7334 | 2.7299 | 2.7508 | 2.7602 | 2.7680 | 2.7906 | 2.7963 | 2.8060 | 2.8129 |
| `R1_rem_extremos` | 42 | 2.8230 | 2.7135 | 2.6666 | 2.6786 | 2.6918 | 2.7106 | 2.7090 | 2.7153 | 2.7258 | 2.7432 | 2.7465 | 2.7663 | 2.7726 | 2.7814 | 2.7852 |
| `R1_rem_extremos` | 7 | 2.8734 | 2.7027 | 2.6829 | 2.7075 | 2.7391 | 2.7511 | 2.7496 | 2.7648 | 2.7622 | 2.7809 | 2.7844 | 2.8074 | 2.8230 | 2.8235 | 2.8302 |
| `R1_rem_extremos` | 2025 | 2.8648 | 2.7011 | 2.6870 | 2.7045 | 2.7380 | 2.7414 | 2.7525 | 2.7470 | 2.7595 | 2.7684 | 2.7825 | 2.8032 | 2.8055 | 2.8222 | 2.8303 |
| `R2_rem_controle` | 42 | 2.8324 | 2.6901 | 2.6619 | 2.6753 | 2.6845 | 2.7060 | 2.7006 | 2.7132 | 2.7182 | 2.7377 | 2.7359 | 2.7538 | 2.7602 | 2.7704 | 2.7706 |
| `R2_rem_controle` | 7 | 2.8572 | 2.7052 | 2.6760 | 2.6985 | 2.7258 | 2.7399 | 2.7434 | 2.7570 | 2.7601 | 2.7770 | 2.7794 | 2.8022 | 2.8171 | 2.8158 | 2.8231 |
| `R2_rem_controle` | 2025 | 2.8724 | 2.7182 | 2.6901 | 2.6763 | 2.7102 | 2.7215 | 2.7240 | 2.7209 | 2.7370 | 2.7536 | 2.7603 | 2.7847 | 2.7898 | 2.7968 | 2.8042 |

`[MEDIDO]` Mínimo de `eval_loss` por run: `A0_baseline`/42=ép 3, `A0_baseline`/7=ép 3, `A0_baseline`/2025=ép 3, `R1_rem_extremos`/42=ép 3, `R1_rem_extremos`/7=ép 3, `R1_rem_extremos`/2025=ép 3, `R2_rem_controle`/42=ép 3, `R2_rem_controle`/7=ép 3, `R2_rem_controle`/2025=ép 4.

## 8. Dimensionamento `[MEDIDO]`

Sementes necessárias para 80% de poder (α = 0,05 bilateral, aproximação
normal), a partir do d_z observado na época E, na comparação **decisiva**:

| métrica | d_z | sementes para 80% |
|---|---|---|
| A ICM-Norm hard | -0.70 | 17 |
| A ICM-Soft-Norm soft | -1.06 | 7 |
| B ICM-Norm hard | +0.39 | 52 |
| B ICM-Soft-Norm soft | -0.53 | 29 |
| C ICM-Norm hard | -0.28 | 104 |
| C ICM-Soft-Norm soft | -0.68 | 17 |

`[LIMITAÇÃO]` Com n = 3 o sd é ele próprio muito ruidoso; estes números são
ordens de grandeza, não estimativas. A leitura útil é binária: «cabe em ~10
sementes» ou «não cabe em nenhum orçamento razoável».

## 9. Interpretação

### 9.1 O critério de remoção não se distingue do controlo emparelhado

Na comparação decisiva, na métrica e na época pré-registadas: **Δ = -0.0077, p = 0.206, d_z = -1.06**, mesmo sinal em 3/3 sementes.

**Veredito: nulo.** Remover os anotadores extremos **não produz um alvo
mensuravelmente melhor** do que remover o mesmo número de anotadores
sorteados ao acaso. O que a ablação mexe nas métricas é compatível com a
perda de votos, não com o critério pelo qual os votos foram escolhidos.

### 9.2 O nulo não vem de os dois alvos serem quase iguais

Objecção óbvia a um nulo entre dois braços de ablação: «os alvos eram
praticamente o mesmo, claro que não deu diferença». **Os números dizem o
contrário.**

R1 e R2 discordam em **1653** dos 3984 itens, com
\|Δ\| médio de **0.1073** nos afectados e
**232** itens a mudar de lado dos 0,5 — **mais**
cruzamentos do que R1 gera contra o A0 (150) ou R2 contra
o A0 (114). A intervenção era grande à entrada; não
chegou à saída.

### 9.3 O `p` nominal em R1 − A0 não é um achado

`R1 − A0` na métrica primária dá Δ = -0.0064, p = 0.048 — nominalmente abaixo de 0,05. **Não é evidência de
efeito de critério**, por três razões, e a primeira estava fixada antes de
qualquer número ser visto:

1. **Pré-registo.** `R1 − A0` foi declarado contexto. Confunde critério com
   remoção de densidade e fragmentação da grade (§3.1).
2. **Multiplicidade.** É uma célula entre 36. Uma abaixo de
   0,05 é o que se espera por acaso.
3. **O teste decisivo, que isola o critério, não a sustenta**
   (p = 0.206).

**O que separa os dois `p`, mecanicamente.** As médias são quase iguais (-0.0064 contra -0.0077); o que muda é o
denominador. O `t` pareado divide pelo **sd das diferenças por semente**, e
essa é a quantidade que difere:

| comparação | Δ por semente (42 / 7 / 2025) | sd(Δ) | t | p |
|---|---|---|---|---|
| R1 − R2 | -0.0041 / -0.0029 / -0.0160 | 0.0072 | -1.84 | 0.206 |
| R1 − A0 | -0.0090 / -0.0041 / -0.0061 | 0.0025 | -4.42 | 0.048 |
| R2 − A0 | -0.0049 / -0.0011 / +0.0099 | 0.0077 | 0.29 | 0.799 |

E **não** é que o R2 seja um braço mais ruidoso em nível absoluto — a
dispersão entre sementes é comparável nos três:

| braço | valor por semente (42 / 7 / 2025) | sd entre sementes |
|---|---|---|
| A0_baseline | 0.4711 / 0.4610 / 0.4608 | 0.0059 |
| R1_rem_extremos | 0.4621 / 0.4570 / 0.4546 | 0.0038 |
| R2_rem_controle | 0.4662 / 0.4599 / 0.4706 | 0.0054 |

O que difere é o **co-movimento entre sementes**: R1 e A0 sobem e descem
juntos (sd(Δ) = 0.0025), enquanto o R2 não acompanha nenhum dos dois
(sd(Δ) = 0.0072, **2.9×** maior). Mesmo Δ médio, denominador
2.9× maior, `t` proporcionalmente menor — é isso, e não uma
diferença de efeito, que separa p = 0.048 de p = 0.206.

E a inferência que isto autoriza é mais forte do que a objecção de
multiplicidade sozinha: **a instabilidade entre sementes introduzida por
remover anotadores AO ACASO é da mesma ordem do deslocamento causado por
remover os extremos**. O «efeito» do critério cabe dentro do piso de ruído da
própria ablação. Isso fecha também a leitura inversa — «o controlo é que é
errático, logo `R1 − A0` é o sinal bom»: essa erraticidade **é** a medida de
quanto uma remoção arbitrária mexe no resultado, e é a escala contra a qual o
critério tem de ser julgado.

Dito isto, há um padrão coerente que vale a pena registar **como hipótese, não
como resultado**: `R2 − A0` na mesma métrica é +0.0013 (p = 0.799), ou seja **a remoção de
densidade sozinha não custa nada**, enquanto R1 fica abaixo do A0 e abaixo do
R2. As três leituras apontam na mesma direcção: remover os extremos custa um
pouco, em vez de ganhar. `[HIPÓTESE]` — a magnitude está dentro do ruído a
n = 3 e o teste que a decidiria não a sustenta.

### 9.4 A régua favorece o A0 por construção — e é por isso que a decisiva é R1 vs R2

O gold de validação é o `raw` em todos os braços, por desenho. R1 e R2 treinam
ambos para um alvo com menos votos que o da régua, logo **os dois** partem em
desvantagem contra o A0. Essa desvantagem **cancela-se no pareamento R1 − R2**,
que é exactamente a razão de ser o teste decisivo. Ler qualquer um dos dois
contra o A0 como «a ablação piorou o modelo» é ler o desencontro alvo-régua.

### 9.5 O que faria falta para fechar

Com d_z = -1.06 na métrica decisiva, seriam precisas **≈7 sementes** para 80% de poder — mesma ordem de grandeza do estudo
A0/A1 (≈6), e dentro de um orçamento razoável. Mas a direcção que essas
sementes iriam confirmar é
**negativa**: no melhor dos casos estabelece-se que remover os extremos tem um
**custo** pequeno, nunca um ganho. Correr mais sementes não pode virar isto num
resultado positivo.

`[LIMITAÇÃO]` Fica por fechar o histograma de densidades por braço (§3.1). Se
R1 e R2 não removerem votos na mesma distribuição de itens, sobra um resíduo de
densidade dentro da própria comparação decisiva.

### 9.6 Onde isto encaixa no capítulo

Duas linhas independentes, o mesmo desfecho:

1. **Ponderar** os votos pela competência estimada (θ do MACE corrigido) — nulo
   (`experiment_taskA_15ep_report.md`).
2. **Remover** os anotadores identificados como problemáticos, contra controlo
   emparelhado — nulo, aqui.

Ponderar e remover são as duas formas canónicas de usar uma estimativa de
confiabilidade para construir um alvo. Nenhuma delas rende neste dataset,
**medido no regime real de treino e com o confound de densidade controlado**.
Isso é mais forte do que um nulo só de ponderação: mostra que o limite não está
na forma de usar o θ, está na informação disponível.

