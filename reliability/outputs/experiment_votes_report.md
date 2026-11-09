# Fase 2 — régua de votos, controlo positivo e viés congelado

Gerado por `reliability/analyze_votes_experiment.py`. Não treina nada.
Protocolo e critério: `reliability/CONFUSION_EXPERIMENT.md` §§10-18.

Época reportada: **E = argmin(eval_loss)** — nunca a métrica reportada (isso seria leakage).

Taxa base do painel (só treino): **0.5552** · anotadores sem votos de treino: 0

## 1. Régua — surpresa (NLL) nos votos retidos, em E

⚠ **`b` contra `a` é descritivo, não prova.** `b` já bate `a` no init por encolhimento da previsão (calibração global, não sinal de anotador — §14). O contraste limpo do sinal individual é **`b` contra `b_perm`**, e o critério é **`b` contra `d4_train`**.

| braço | seed | E | a | b | b_perm | c | d3_train | d4_train | d3_val | d4_val |
|---|---|---|---|---|---|---|---|---|---|---|
| M1sd | 2025 | 3 | 0.6234 | 0.5989 | 0.6260 | 0.6301 | 0.5845 | 0.5843 | 0.5680 | 0.5676 |

Acerto (%) nos mesmos votos:

| braço | seed | a | b | b_perm | c | d3_train | d4_train | d3_val | d4_val |
|---|---|---|---|---|---|---|---|---|---|
| M1sd | 2025 | 65.8 | 68.3 | 65.0 | 63.2 | 68.8 | 69.0 | 69.0 | 68.8 |

### As duas diferenças que decidem

`b − b_perm` isola o sinal INDIVIDUAL (mesma calibração global dos dois lados; muda só cada pessoa receber a sua ficha ou a de outra). `b − d4_train` é o critério. **Negativo = a ficha ganha.**

| braço | seed | E | b − a | **b − b_perm** | **b − d4_train** | b − d4_val |
|---|---|---|---|---|---|---|
| M1sd | 2025 | 3 | -0.0244 | **-0.0271** | **+0.0147** | +0.0313 |

### Trajectória ao longo das épocas

⚠ A época reportada é **E = argmin(eval_loss)**, que é o selector do SISTEMA. O modelo de anotador é outro objecto e pode ter outro óptimo. A trajectória vai aqui como **descritivo**; mudar o selector para a régua de votos seria escolher a época pela métrica reportada, que o protocolo proíbe.

`M1sd seed 2025`  (E = 3)

| ép | eval_loss | b − b_perm | b − d4_train |
|---|---|---|---|
| 1 | 2.8884 | -0.0012 | +0.0525 |
| 2 | 2.7026 | -0.0095 | +0.0396 |
| 3 | 2.6669 | -0.0271 | +0.0147  ← **E** |
| 4 | 2.6877 | -0.0389 | -0.0012 |
| 5 | 2.7148 | -0.0495 | -0.0127 |
| 6 | 2.7201 | -0.0570 | -0.0139 |
| 7 | 2.7149 | -0.0635 | -0.0179 |
| 8 | 2.7223 | -0.0690 | -0.0216 |
| 9 | 2.7367 | -0.0722 | -0.0231 |
| 10 | 2.7445 | -0.0749 | -0.0263 |
| 11 | 2.7514 | -0.0778 | -0.0273 |
| 12 | 2.7759 | -0.0792 | -0.0271 |
| 13 | 2.7804 | -0.0806 | -0.0302 |
| 14 | 2.7874 | -0.0807 | -0.0310 |
| 15 | 2.7955 | -0.0810 | -0.0309 |


## 2. Veredito por braço

- **M1sd seed 2025** (viés livre (run A -- termo de comparação)) → **NULO** — b não bate d4_train

## 3. O que as matrizes aprenderam (competência e viés em E)

Viés em **espaço de logit** (`u − v`). Em probabilidade ele varia com a competência mesmo com o viés individual congelado a zero, por não-linearidade da sigmóide — reportar ali faria o braço congelado parecer que continua a aprender viés (§13.4).

| braço | seed | comp. média | comp. sd | viés logit sd | r(comp, yes_rate) | **r(comp, \|yes−base\|)** | r(comp, concord.) |
|---|---|---|---|---|---|---|---|
| M1sd | 2025 | 0.7685 | 0.0530 | 0.7178 | +0.395 | **-0.519** | +0.574 |

Referência da Fase 1 (viés livre, `delta`): competência sd **0,066**; `r(comp, |yes_rate − base|)` = −0,758 / −0,731 / −0,747; `r(comp, concordância)` = +0,217 / +0,231 / +0,248.


## 5. ICM em E (contexto, não critério)

O ICM compara dois **treinos**, logo tem por baixo o ruído de seed: sd do próprio A0 entre seeds = **0,0059**, contra um efeito medido de +0,0009 na Fase 1. Fica aqui por simetria e honestidade.

⚠ **O ICM de uma run corrompida (`*crp`) não se compara com nada** — os dados de treino foram estragados de propósito.

| braço | seed | E | eval_loss | ICM-Soft-Norm (A, soft) |
|---|---|---|---|---|
| M1sd | 2025 | 3 | 2.666935 | 0.4647 |
