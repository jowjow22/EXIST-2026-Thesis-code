# 09 — Baseline congelado: sistema submetido no split congelado

**Estes são os números contra os quais o sistema corrigido será comparado.**
Congelados em 2026-08-08. Não re-rodar e substituir sem registar aqui o porquê.

- **Notebook:** `Thesis_video_baseline.ipynb`
- **Artefatos:** `runs_baseline/seed_42/`
- **Fonte do número reportável:** `runs_baseline/seed_42/final_metrics.json`

---

## 1. Proveniência da run

| | |
|---|---|
| split | **congelado** — `Dataset/val_split_42.json` (`split_seed=42`, 20 %, estratificado por Task A) |
| treino / val | 2 019 / 505 |
| `SEED` (init + ordem dos dados) | 42 |
| épocas | 15, SWA a partir da 5 |
| `train_runtime` | 5 523,9 s (92,1 min) — treino puro 340,6 s/época, evals 7 % do total |
| fingerprint init | `8a60e7121b63fbb2` |
| fingerprint final | `9bfb83bcc6efd303` |
| reprodutibilidade | **verificada** — segunda run reproduziu os fingerprints |

**O uso do split congelado foi verificado por aritmética, não por confiança no log:**
`train_samples_per_second × train_runtime ÷ epochs` = **2 019,2** amostras/época
(congelado = 2 019; split antigo teria dado 2 271), e
`eval_samples_per_second × eval_runtime` = **505,0** (antigo = 253).

**Tratamento intacto:** `cross_attn`, `bio_ln`, `max_length=128`, SWA, fusão 2688,
`warmup_ratio`, 15 épocas. A única mudança em relação ao `Thesis_video_fixed.ipynb`
foi o split — o instrumento de medida, não o objeto medido.

---

## 2. Decisão: o baseline é o **estado final da run**

Havia duas leituras defensáveis:

| leitura | Task A hard | o que é |
|---|---|---|
| **estado final da run (ESCOLHIDA)** | **0,5483** | o que o código produz quando rodado de ponta a ponta — modelo **SWA** |
| melhor época da curva | 0,5872 (época 6) | o que o sistema antigo conseguiria com a época escolhida bem |

**Escolhido: o estado final.** Justificação do autor: assim o baseline reflete o
que o modelo/código de facto é, sem uma escolha de época feita depois de ver os
resultados. É também o regime em que o sistema foi submetido ao EXIST 2026.

### A regra de simetria que isto cria — obrigatória

Se o baseline é "o estado final de rodar o código", então o número do sistema
corrigido tem de ser **o estado final de rodar o código dele também** — nunca a
sua melhor época. Escolher pelo fim de um lado e pelo pico do outro fabricaria
ganho.

Consequência prática: o `NUM_TRAIN_EPOCHS` do sistema corrigido tem de ser
congelado pelo critério a-priori (`E = argmin(eval_loss)` na run de calibração)
**antes** de qualquer comparação, e depois o número reportado é simplesmente o
fim da run com esse E.

### A atribuição que tem de ser declarada

O sistema corrigido removeu o SWA (ARCH-16). Comparado contra o estado final do
baseline, **parte de qualquer ganho vem de ter removido o SWA**, não das outras
correções. Isto está medido nesta própria run: o SWA é pior que o melhor da curva
em **6/6 métricas**. Não é um problema da comparação — é uma componente dela que
precisa ser nomeada, não escondida.

Se alguém perguntar "quanto do ganho é o SWA?", a resposta está na tabela §4:
o teto do sistema antigo sem SWA e com época bem escolhida era 0,5872 em A hard.

---

## 3. Baseline — números oficiais

De `runs_baseline/seed_42/final_metrics.json` (modelo SWA, val de 505 amostras).

| tarefa | ICM-Norm hard | ICM-Soft-Norm soft |
|---|---|---|
| **A** (binária) | **0,5483** | **0,4972** |
| **B** (intenção) | **0,4579** | **0,3713** |
| **C** (categorias) | **0,3784** | **0,1378** |

Métricas de apoio:

| | valor |
|---|---|
| `eval_loss` | 1,5309 |
| Task A F-measure hard | 0,6989 |
| Task A ICM hard (bruto) | +0,0966 |
| Task B ICM hard (bruto) | −0,1221 |
| Task C ICM hard (bruto) | −0,4887 |
| Task A ICM soft (bruto) | −0,0159 |
| Task B ICM soft (bruto) | −1,1159 |
| Task C ICM soft (bruto) | −6,9380 |

**Task C com limiares ajustados** (`runs_baseline/seed_42/task_c_thresholds.json`):
limiares `[0.30, 0.30, 0.20, 0.25, 0.25, 0.25]` na ordem de `LABELS_2_3` →
**ICMNorm 0,4267** contra 0,3784 sem ajuste (**+0,0483**).

⚠ Este 0,4267 foi ajustado e reportado **no mesmo conjunto de validação** — é o
regime do sistema antigo, e é otimista. O sistema corrigido ajusta numa metade do
val e reporta na outra (ARCH-11 b), o que é mais honesto e provavelmente dará um
número mais baixo pela metodologia, não pelo modelo. **Não comparar 0,4267 com o
número de meia-val do sistema corrigido.** Para Task C hard, comparar os valores
sem ajuste (0,3784) ou re-ajustar o baseline em meia-val.

---

## 4. Curva por época — contexto, não baseline

`runs_baseline/seed_42/epoch_metrics.csv`, 15 linhas.

| métrica | melhor | @ época | ep15 pré-SWA | SWA final | SWA − melhor |
|---|---|---|---|---|---|
| A hard | 0,5872 | 6 | 0,5626 | 0,5483 | −0,0389 |
| A soft | 0,5060 | 15 | 0,5060 | 0,4972 | −0,0088 |
| B hard | 0,4993 | 8 | 0,4442 | 0,4579 | −0,0414 |
| B soft | 0,3763 | 15 | 0,3763 | 0,3713 | −0,0050 |
| C hard | 0,3857 | 10 | 0,3579 | 0,3784 | −0,0072 |
| C soft | 0,1468 | 10 | 0,1433 | 0,1378 | −0,0090 |

**`argmin(eval_loss)` = época 5** (1,3475). A loss sobe monotonicamente da 6 em
diante — as últimas 10 épocas são cauda de overfitting.

**O ARCH-06 reproduziu.** O finding original (seed 2025, split antigo, val de 253)
media Task A ICMNorm hard 0,6199 → 0,5647. Aqui, em condições diferentes:
0,5872 → 0,5483. Mesma direção, magnitude parecida. O achado não era artefato do
split antigo.

Nota sobre `eval_task_c_icmsoftnorm_soft = 0,0000` nas épocas 1–2: é valor real,
não falha silenciosa. O ICM bruto é −10,34 e −9,69, e a normalização tem piso em
zero. O modo de falha do ARCH-14 (métricas voltando zero por exceção engolida)
**não** ocorreu nesta run.

---

## 5. Como comparar quando o sistema corrigido rodar

1. Mesmo split congelado (`val_split_42.json`), mesma `INIT_SEED=42`. Verificar
   por aritmética, como na §1 — não confiar no log.
2. Comparar **estado final contra estado final**. Nunca fim de um lado, pico do outro.
3. Declarar que a remoção do SWA é parte do tratamento e contribui para o delta.
4. Task C hard: comparar sem ajuste de limiar, ou re-ajustar o baseline em
   meia-val. Os dois regimes de ajuste não são comparáveis.
5. Uma seed é uma seed. Um delta em `INIT_SEED=42` não é um resultado até
   replicar em 7 e 2025, com **delta pareado por seed + consistência de sinal**.
6. Fingerprints **nunca** se comparam entre os dois notebooks — arquiteturas
   diferentes consomem RNG de forma diferente.

---

## 6. O que este baseline NÃO é

- **Não é o número do paper.** O paper reporta a tabela de ablação, que é uma seed
  apresentada como agregado (SIMP-01), com braços partilhando pesos (ARCH-20), no
  split antigo. Este baseline é um objeto novo, medido para esta comparação.
- **Não é comparável com `ablation_seed_{42,7,2025}.csv`** nem com
  `predictions/cross_att/` — split diferente, e aqueles eram sondas de zeragem de
  input, não runs treinadas.
- **Não usa features de áudio corrigidas.** Roda sobre as features diluídas
  (ARCH-04, `valid_fraction` 0,655). É de propósito: é o estado do sistema antes
  de qualquer melhoria.
