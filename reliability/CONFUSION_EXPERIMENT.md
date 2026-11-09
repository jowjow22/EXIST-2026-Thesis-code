# Camada de confusão como perda auxiliar da Task A — protocolo

Linha de **caracterização pós-submissão**. O objectivo **não é melhorar o
sistema**: é medir se modelar o anotador através de uma camada de confusão
acrescenta sinal ao treino sobre soft labels. Um resultado nulo é um resultado
válido e esperado — nada aqui é optimizado para obter Δ positivo.

## 1. A intervenção

```
L = L_soft(A, B, C)  +  lambda * L_aux(A)
```

`L_soft` é a perda do A0, **inalterada** (`MultiHeadModel.forward`, `memes_core.py`).
`L_aux` aplica-se só à Task A:

- `p_latente(i)` = `[1 - sigmoid(logit_a), sigmoid(logit_a)]` = `[P(YES), P(NO)]`
  — a convenção `sigmoid(head_a) = P(NO)` é a do baseline (`memes_core.py:837`).
- `C_a[i, j] = P(voto = j | latente = i)`, linhas somando 1 (softmax por linha).
- `p_voto(a, i) = p_latente(i) @ C_a`.
- `L_aux` = cross-entropy entre `p_voto(a,i)` e o voto observado, **média** sobre
  os pares (anotador, item) com voto presente no batch.

**Duas correcções ao handoff, declaradas:**

1. O handoff escreve `p_voto = C_a @ p_latente` com linhas normalizadas — isso não
   devolve uma distribuição. A forma usada é a equivalente correcta (`p @ C`).
2. O handoff diz "somada"; usa-se a **média**, para `lambda` ficar na mesma escala
   da `L_soft` (que também é média). Uma soma sobre ~48 votos por batch mudaria o
   significado numérico de cada valor da grelha de `lambda`.

## 2. Os três métodos

| id | referência | `C_a` | regularização (fixada **a priori**) |
|---|---|---|---|
| **M1** | Chu et al., AAAI 2021 (**adaptado**) | `row_softmax(L_comum + Delta_a)` | L2 sobre `Delta_a`, peso 0.01 |
| **M2** | Tanno et al., CVPR 2019 | individual completa | penalização do traço, peso 0.01 |
| **M3** | Rodrigues & Pereira, AAAI 2018 (crowd layer) | individual completa | **nenhuma**, por desenho |

- **M1 é uma adaptação**: implementa a decomposição comum + individual, não o
  portão dependente da instância do CoNAL. Não reportar como "é CoNAL".
- **M2**: 0.01 é o ponto de partida atribuído a Tanno et al. — **confirmar no paper
  antes de o citar na tese**; o valor efectivamente usado é o que fica reportado.
- Os pesos de regularização **não são grelha**: um valor cada. Uma grelha neles
  inflaria o número de comparações para além do Bonferroni-3.
- A regularização é multiplicada por `lambda` junto com a auxiliar: com
  `lambda = 0` o mecanismo inteiro está desligado, o que é o que torna o controlo
  uma reprodução do A0.

## 3. Ordem de execução

1. **CONTROLO (obrigatório, antes de tudo):** `METHOD='M1'`, `LAMBDA=0.0`,
   `SEED=42`. ~1 h. **FEITO a 2026-09-06 — PASSOU** (secção 5). Corre com **M1** e
   não M3 de propósito: o `common = 3*I` do M1 é um detector vivo da exclusão do
   weight decay, enquanto o `delta` do M3 nasce a zero e o decaimento nele seria
   invisível (0 -> 0).
2. **M1**, `lambda ∈ {0.1, 0.3, 1.0}` x `seed ∈ {42, 7, 2025}` — 9 runs, ~9 h.
   `lambda=0.1`/seed 42 feito a 2026-09-07.
3. **Δ_null (recomendado, 2 h):** `lambda = 0` nas seeds 7 e 2025, para ter a
   banda do offset do invólucro nas três seeds. Pode correr depois da grelha.
4. **Paragem antecipada:** se o M1 falhar o critério em **todos** os lambdas,
   **não correr M2 nem M3**. O M1 é o único desenhado para poucos votos por
   anotador (média medida: **26,95** votos/anotador, min 9, máx 54, 887
   anotadores). Um nulo no M1 é evidência contra a família. Reportar e parar.
5. Só se o M1 passar: **M2** e depois **M3**, mesma grelha (~9 h cada).

Custo total se a família correr inteira: ~28 h de GPU.

## 4. Como correr

Notebook: `EXIST_MEMES/memes_confusion.ipynb` (cópia de `memes_model.ipynb` com
as células 5, 6, 37, 39, 41, 54, 55, 58 alteradas e uma célula de verificação do
controlo depois do treino). Editar **só** a célula de configuração:

```python
SEED   = 42
METHOD = "M1"      # "M1" | "M2" | "M3"
LAMBDA = 0.1       # 0.0 = controlo; grelha: 0.1, 0.3, 1.0
REG_WEIGHT = None  # None = valor a priori do método
CONF_LR = None     # None = pré-registado; um valor = braço exploratório (§5c)
```

Run All. Cada run escreve:

- `EXIST_MEMES/metrics_history/<ts>_conf_<M>_lam<L>[_clr<LR>]_seed<S>.json`
- `..._confusion.json` (as 887 matrizes por época — E é escolhida post-hoc, por
  isso guardam-se todas)

Análise (local, não treina nada):

```bash
python3 -m reliability.analyze_confusion_experiment
```

Escreve `reliability/outputs/experiment_confusion_report.{md,json}`.

Testes de sanidade da implementação (correm em CPU, sem `transformers`):

```bash
cd EXIST_MEMES && python3 confusion_selftest.py
```

## 5. Critério do controlo

Com `lambda = 0` o mecanismo está desligado, mas o caminho auxiliar continua
**vivo** (é calculado e multiplicado por zero, não é saltado) — `0 * NaN = NaN` é
o detector de bug que se quer.

**Critério (recalibrado a 2026-09-07, ver nota):**

1. as matrizes de confusão ficam **congeladas** (sem fuga de gradiente, sem
   weight decay) — medido no dump `*_confusion.json`;
2. o **A0 reproduz** quando re-corrido — prova que o ambiente não derivou;
3. `|Δ da métrica primária em E|` **< 1 sd do A0 entre as 3 seeds**, e o
   `Δ eval_loss em E` dentro da amplitude entre seeds.

`E` é a época que o estudo reporta; o máximo de `|Δ|` ao longo das 15 épocas
fica registado como **contexto, não como critério**.

### Nota — um degrau retirado, declarado

A primeira versão deste protocolo exigia reprodução **bit-a-bit**: máximo de
`|Δ|` de `eval_loss` sobre as 15 épocas < 1e-3. Esse degrau **foi retirado
depois de ver o resultado do controlo** e isso fica declarado.

A razão não é o resultado ter sido inconveniente, é a estatística estar mal
calibrada para um sistema caótico: uma perturbação de um ulp no passo 1
amplifica para escala macro na época 15, logo o máximo sobre 15 épocas **não
distingue um ulp de um bug** — os dois têm a mesma assinatura. A evidência que
efectivamente discrimina é independente e está medida:

- matrizes congeladas: `max|C − C_init| = 7e-9` (arredondamento float32 do
  softmax) nas épocas 1 e 15, variação entre os 887 anotadores = **0,0 exacto**;
- A0 re-corrido a 2026-09-06 contra o log de 2026-08-31: **bit-a-bit igual** —
  15 logs de treino (`loss` e `grad_norm`, Δ = 0,0) e 23 de 25 métricas de
  avaliação idênticas (as duas que diferem são tempos de parede).

O critério do **resultado** (secção 6) não foi tocado.

### Medido no controlo (M1, lambda = 0, seed 42, 2026-09-06)

| quantidade | valor | régua |
|---|---|---|
| Δ métrica primária em E | **−0,0017** | sd do A0 entre seeds = **0,0059** |
| Δ `eval_loss` em E | **+1,8e-3** | amplitude entre seeds = **1,94e-2** |
| E | 3 nos dois | — |
| máx \|Δ\| `eval_loss` (15 ép) | 1,66e-2 | contexto |

**Veredicto: PASS.**

### Mecanismo da perturbação — RESOLVIDO (2026-09-07)

Teste de 2 passos (`max_steps=2`, `logging_steps=1`, sem avaliação nem
checkpoints), notebooks `memes_model_step2.ipynb` e `memes_confusion_step2.ipynb`,
`SEED=42` / `ARM='raw'` contra `M1`, `lambda=0`:

| | `memes_model` (A0) | `memes_confusion` (invólucro) |
|---|---|---|
| passo 1 · `loss` | 3,4732069969 | 3,4732069969 |
| passo 1 · `grad_norm` | 15,6839170456 | 15,683916**0919** |
| passo 2 · `loss` | 3,4032576084 | 3,4032576084 |
| passo 2 · `grad_norm` | inf | inf |

**A `loss` do passo 1 é bit-a-bit igual.** Essa `loss` é a média de 8
micro-batches **antes de qualquer update do optimizador**, logo o forward do
modelo-base não é tocado pelo invólucro: a afirmação "`L_soft` inalterada" está
verificada empiricamente, não só por leitura do código.

**A divergência nasce no `grad_norm`:** 9,54e-7 de diferença absoluta, **6,08e-8
relativa**. É o valor devolvido por `clip_grad_norm_`, que com `foreach` calcula
uma norma por tensor, empilha-as e reduz — dois tensores extra na lista (os
parâmetros de confusão, com gradiente exactamente zero) mudam o comprimento do
vector e portanto a árvore de redução e o arredondamento. Somar zeros é exacto em
aritmética real, não na ordem de redução do `vector_norm`.

Ponto de injecção: `clip_coef = max_grad_norm / total_norm` difere em 6,08e-8
relativo, e **multiplica todos os gradientes**. Cada passo desloca todos os pesos
por ~1e-7 relativo; ao longo de 795 passos e 15 épocas, isso amplifica para a
divergência macro observada (1,66e-2 de `eval_loss` na época 8).

`grad_norm = inf` no passo 2 nos DOIS notebooks é overflow do fp16 no
`GradScaler`, que salta o passo em ambos — daí a `loss` do passo 2 também
coincidir. É comportamento do A0, não do invólucro.

**Consequência para a tese:** o invólucro não altera a perda nem o forward; o que
altera é o arredondamento do recorte de gradiente, com efeito equivalente ao de
uma perturbação de seed. Por isso o `Δ_null` abaixo é a régua honesta para o
tamanho de efeito que uma intervenção nula produz.

### Δ_null — o offset do invólucro como régua empírica

O controlo mede o efeito de uma intervenção **de efeito zero**: `Δ_null =
controlo − A0`, por seed. Correr `lambda = 0` também nas seeds 7 e 2025 (2 h) dá
a banda de `Δ_null` nas três. Se os Δ do M1 não saírem dessa banda, isso é
decisivo independentemente do teste t. O comparador **continua a ser o A0**, como
pré-registado — o `Δ_null` entra no relatório como contexto, não substitui a
referência.

## 5b. Achado da primeira run da grelha (M1, lambda=0,1, seed 42, 2026-09-07)

Δ contra o A0 = **+0,0005** (Δ_null do controlo = −0,0017; sd entre seeds =
0,0059). E moveu de 3 para 4. Mas o diagnóstico das matrizes mostra que elas
praticamente não se moveram:

| época | dist. média a I | sd do viés | `max\|C − C_init\|` | r(viés, `yes_rate`) |
|---|---|---|---|---|
| 1 | 0,0949 | 0,0000 | 0,0000 | +0,944 |
| 4 (= E) | 0,0948 | 0,0001 | 0,0001 | +0,947 |
| 15 | 0,0942 | 0,0002 | 0,0006 | +0,924 |

Não colapsaram na identidade — **nunca saíram do sítio onde nasceram**. E a
direcção está certa: o viés aprendido correlaciona +0,92 a +0,95 com a
`yes_rate` do anotador. O gradiente aponta para o sítio certo; o passo é que é
minúsculo.

## 5c. Braço exploratório — learning rate próprio para as matrizes

**Causa do achado acima: as matrizes herdavam o `learning_rate` do encoder
(1e-5)**, tunado para afinar um XLM-R pré-treinado. O passo do AdamW é ~`lr` por
iteração quando o sinal do gradiente é consistente; 795 passos dão no máximo
7,95e-3 em espaço de logit, e com `dp/dlogit = p(1−p) = 0,045` no ponto de init
isso é **Δp ≈ 3,6e-4**. Medido: 6e-4. As matrizes moveram-se o máximo que o LR
permitia.

**Pior — `lambda` não corrige isto.** O AdamW é invariante à escala do gradiente
(`update = lr·m/(√v+ε)`; um gradiente `λ·g` escala `m` e `√v` juntos, e ε=1e-8 é
irrelevante nesta escala). Para as matrizes, a auxiliar é a **única** fonte de
gradiente, logo `lambda` cancela-se: a grelha `{0,1; 0,3; 1,0}` não mede quanto
pesa modelar o anotador, mede quanto a auxiliar perturba o encoder.

Consequência para a interpretação: um nulo na grelha pré-registada **não** é
evidência contra a família de métodos — é compatível com "o mecanismo nunca teve
hipótese de se exprimir".

**O braço.** `CONF_LR` no notebook põe `confusion.*` num grupo de parâmetros
próprio (`ConfusionTrainer.create_optimizer`), com weight decay a zero como no
caminho pré-registado. O scheduler do HF aplica o mesmo factor multiplicativo a
todos os grupos, logo o grupo novo também tem warmup e decaimento, a partir da
sua própria base. Verificado localmente: `conf_lr=1e-2` move as matrizes 1,2e-1
contra 1,2e-4 com o LR partilhado — factor 1000, exactamente a razão dos LR.

Valor escolhido: **1e-2**. Justificação: para um anotador se afastar em ~0,2 de
probabilidade do ponto de init é preciso `Δlogit ≈ 0,2/0,045 ≈ 4,4`; com 795
passos, só `lr ≥ ~5e-3` dá essa margem. 1e-2 é a definição de *expressividade
máxima*, não um hiperparâmetro tunado — não há grelha sobre ele.

**Estatuto: EXPLORATÓRIO e POST-HOC.** Está fora da grelha pré-registada e fora
da estatística confirmatória (secção 6). Os ficheiros levam `_clr<valor>` no
`RUN_TAG` e o analisador põe-nos numa secção separada do relatório (5b). A grelha
pré-registada mantém-se com `CONF_LR = None`.

## 5c-bis. O que `lambda` faz de facto — e porque a grelha degenera (2026-09-08)

`lambda` multiplica `L_aux` na perda total, e tem dois caminhos de gradiente com
comportamentos diferentes:

**Caminho 1 — para as matrizes `C_a`.** É a única fonte de gradiente delas, e o
AdamW normaliza pelo próprio gradiente, logo `lambda` cancela-se: **não afecta
quanto as matrizes se movem** (§5c).

**Caminho 2 — para o encoder, via `head_a`.** Aqui `lambda` pesa a auxiliar
contra a `L_soft`. Mas com `C_a` perto da identidade, **a auxiliar É a perda da
Task A**, não algo parecido: é o teste T2 (`L_aux(I) = BCE(logits, P(NO) crua)`,
delta 6e-8), porque somar o cross-entropy dos 6 votos de um item é o mesmo que o
cross-entropy contra a frequência desses votos.

Confirmado nos logs, `loss` do passo 50 da seed 42:

| run | `loss` |
|---|---|
| A0 | 3,209908 |
| `lambda=0` | 3,209771 |
| `lambda=0,1` | 3,286565 |

Diferença = 0,0767 = **0,1 x 0,767**, e 0,767 é o BCE inicial. A auxiliar
contribuiu exactamente `lambda x BCE_A`.

O baseline já tem `task_a_weight = 2.0`, logo com as matrizes presas perto da
identidade o efeito de `lambda` é apenas:

| lambda | peso efectivo da Task A | |
|---|---|---|
| 0,1 | 2,1 | +5% |
| 0,3 | 2,3 | +15% |
| 1,0 | 3,0 | +50% |

**Consequência: a grelha pré-registada com `CONF_LR=None` não mede modelação do
anotador — mede uma ablação do peso da Task A.** A informação genuinamente nova
está só no desvio de `C_a` em relação à identidade, e com o LR partilhado esse
desvio é ~1e-4.

Isto muda a leitura dos dois nulos:

- o nulo da **grelha pré-registada** é quase tautológico (uma intervenção de +5%
  a +50% num peso de perda, sem conteúdo novo);
- o nulo do braço **`clr=1e-2`** é o informativo: lá as matrizes desviaram-se a
  sério (viés sd 0,185, amplitude −0,45 a +0,52, r=+0,95 com a `yes_rate`), logo
  `lambda·L_aux` carregava estrutura real de anotador — e o Δ foi na mesma
  +0,0019, dentro do ruído de seed.

Nota de desenho para a tese: uma grelha de `lambda` só é interpretável como
"quanto pesa modelar o anotador" se as matrizes tiverem margem para se afastar da
identidade. Caso contrário mede outra coisa. A grelha pré-registada corre na
mesma, e este achado é ele próprio reportável.

## 5d. Alteração de configuração a meio do estudo (2026-09-07) — declarada

A VM encheu o disco (`No space left on device`, 126 GB a 100%) a meio do primeiro
braço exploratório. Cada checkpoint do HF Trainer pesa ~3,3 GB e
`save_total_limit=2` mantinha dois, mais uma pasta `best_model_<RUN_TAG>` por run.

A partir daqui o `memes_confusion.ipynb` corre com `save_strategy="no"`,
`load_best_model_at_end=False` e `SAVE_MODEL = False` na célula de gravação.

**É neutro para a análise, e é por isso que é aceitável a meio do estudo:**
`E = argmin(eval_loss)` é escolhido post-hoc a partir dos logs por época, que são
escritos pelo `MetricsHistoryLogger` a cada avaliação e não dependem de haver
checkpoints em disco. `load_best_model_at_end` só decide que pesos ficam em
memória **depois** do treino — não toca na trajectória nem em nenhuma métrica
registada. As duas runs já feitas com `save_strategy="epoch"` (controlo e
`lambda=0,1`) continuam comparáveis com as seguintes.

O que se perde: os pesos treinados de cada braço deixam de ficar em disco. Nenhum
passo da análise os usa; re-avaliar um braço exige re-treinar (~1 h).

**Incidente associado:** com o disco a 0 bytes, uma escrita do assistente truncou
`memes_confusion.ipynb` e este ficheiro para 0 bytes (a montagem local é a VM por
sshfs; `open(..., 'w')` trunca antes de escrever e o `write` falhou com ENOSPC).
Ambos foram reconstruídos a 2026-09-07. Nenhum resultado se perdeu —
`metrics_history/` e os dumps `*_confusion.json` ficaram intactos.

## 6. Critério do resultado (pré-registado — não alterar)

- Métrica: **Task A ICM-Soft-Norm (soft)** via PyEvALL, gold de validação sempre
  `raw`, na época **E = argmin(eval_loss)** — nunca pela métrica reportada.
- Por método: Δ = método − A0, **t pareado bilateral** sobre as 3 sementes, no
  melhor `lambda`.
- **PASS da família:** pelo menos um método com Δ > 0 e **p < 0,0167**
  (Bonferroni, 3 métodos).
- A escolha do melhor `lambda` é **post-hoc** e é reportada como tal.
- Com n = 3 o teste é fraco por construção; o `d_z` é a grandeza que interessa.

## 7. O que garante que o A0 não é tocado

- `memes_core.py`, `memes_model.ipynb`, o split e as soft labels ficam **intactos**.
  Tudo o que é novo está em `EXIST_MEMES/confusion_core.py`.
- O `MultiHeadModel` é construído exactamente como no A0 (mesma sequência de draws
  de RNG) e só depois embrulhado por composição.
- A auxiliar entra **só em treino**: a validação nunca recebe votos, logo
  `eval_loss` continua a ser a `L_soft` pura e `E = argmin(eval_loss)` é o mesmo
  selector dos dois lados. Sem isto, a comparação seria confound por construção.
- Os parâmetros de confusão estão **fora do weight decay** (0.05) via
  `ConfusionTrainer.get_decay_parameter_names`: decaimento nos logits das matrizes
  puxa-as para a matriz **uniforme** — o oposto de "perto da identidade" — e no M3
  seria regularização que a especificação proíbe.
- Todos os lookups (anotador, item, classe do voto) são **comparação + matmul**,
  nunca `F.one_hot`, indexação avançada ou `nn.Embedding`: sob
  `torch.use_deterministic_algorithms(True)` o backward de scatter/index_put em
  CUDA aborta ou é não-determinístico.
- A auxiliar corre com o **autocast explicitamente desligado**: com `fp16=True`,
  matmul e einsum executariam em fp16 mesmo depois de um `.float()`, e log-probs
  perto de 0/1 em fp16 são ruído.
- As chaves novas do batch chamam-se `ann_idx`/`ann_vote`: nenhuma contém
  "label", que o `Trainer` usaria para montar `eval_pred.label_ids` e partiria o
  `compute_metrics`.

## 8. Limitação declarada (não corrigir)

O split é **por item**, não por anotador. Anotadores de treino aparecem em
validação, o que **favorece** o método: as matrizes vistas em validação foram
ajustadas com votos desses mesmos anotadores. Está declarado como limitação.

## 9. O que o relatório final tem de trazer

- Tabela método x lambda x seed com ICM-Soft-Norm, `eval_loss` e época E.
- Média e Δ por método no melhor lambda, com p pareado.
- Para M1 e M2 (e M3, se correr): distância média de `C_a` à identidade e se
  correlaciona com a `yes_rate` do anotador (`behavioral_profiles.csv`), e o
  **viés com sinal** `C_a[NO->YES] − C_a[YES->NO]`, que é a quantidade que
  deveria correlacionar se o mecanismo aprendeu alguma coisa (a distância de
  Frobenius é sem sinal e mistura as duas off-diagonais).
  **Diagnóstico, não critério.**

---

# FASE 2 (2026-09-10) — régua de votos, controlo positivo e viés congelado

Tudo o que se segue é **acrescento**. Nada da Fase 1 é alterado: o A0, o split,
a métrica pré-registada e o selector de época `E = argmin(eval_loss)` ficam
exactamente como estavam.

## 10. Porque a Fase 1 não fecha a pergunta

Três factos medidos, todos replicados em 3 seeds:

1. As matrizes, com `CONF_LR=1e-2`, **colapsam no viés marginal** — 88,7 % da
   variância, `viés = 0,827·yes_rate − 0,420`, R² 0,910/0,906/0,908.
2. **Não movem o ICM** — Δ = +0,0009, sd 0,0011, t=1,380, p=0,30.
3. O sd do próprio A0 entre seeds é **0,0059**, ou seja **6× o efeito**.

Faltam três coisas para o resultado ser publicável:

- **O ICM é a régua errada** para modelação de anotador. Compara dois *treinos*,
  logo tem o ruído de seed por baixo. A régua certa compara *preditores dentro
  do mesmo treino*, nos mesmos votos — o sorteio é o mesmo dos dois lados e
  cancela.
- **Não existe controlo positivo.** Sem saber se o mecanismo detecta um anotador
  que *sabidamente* difere, nenhum nulo é interpretável.
- A subsecção já escrita em `cap07.tex` afirma que a competência não correlaciona
  com nada, com base na correlação **linear** (−0,15). **É falso** e tem de ser
  corrigido independentemente de qualquer treino novo (ver §16).

## 11. A régua — votos individuais retidos

Os **3 588 votos** dos **598 itens de validação**. Nunca entraram na `L_soft`
nem na `L_aux`, e os anotadores são quase todos conhecidos do treino: é a
pergunta certa, **anotador conhecido, item novo**.

⚠ Reter votos de itens de **treino** teria fuga: o alvo de treino é a média dos
6 votos do item, logo o voto "retirado" já entrou pela porta de trás.

Implementada como *callback* com loaders próprios, em `on_evaluate`, modelo em
`eval`, sem gradiente. **Nada disto entra na perda do Trainer** e o dataset de
votos **nunca** é passado como `eval_dataset`: `eval_loss` continua a ser
`L_soft` pura, porque é o selector de época.

### Os preditores

| | Preditor | O que usa |
|---|---|---|
| `a` | sem modelo de anotador | só a probabilidade do item (matriz = identidade) |
| `b` | **a ficha aprendida** | probabilidade do item, ajustada por `C_a` |
| `b_perm` | ficha de OUTRA pessoa | igual a `b`, ficha trocada dentro do painel linguístico |
| `c` | só o histórico | `yes_rate_a` constante, ignora o item |
| `d3` | logística de 3 ingredientes | `x1, x2, x3` |
| `d4` | **logística de 4 ingredientes** | `x1, x2, x3, x4` — **é quem decide** |

```
x1 = −logits_a                          (logit P(YES) do item; sigmoid(logits_a) = P(NO))
x2 = logit(yes_rate suavizada)
x3 = logit(concordância suavizada com a maioria leave-one-out)
x4 = |logit(yes_rate) − logit(taxa base do painel)|
```

`d3` e `d4` são ajustadas duas vezes: num subconjunto fixo de 600 itens de
treino (**critério**) e na própria validação (**oráculo**, barra mais alta).

## 12. Critério da Fase 2 — fixado antes da run A

**Sensibilidade (run B, decide a Porta 1).** A competência dos anotadores
corrompidos tem de cair monotonicamente com o nível, e o nível ≥ 40 % tem de
separar do grupo de controlo (nível 0 %). Reportar a **curva de limiar de
detecção**, não um passa/falha.

**Sinal (runs C/D/E).** O braço congelado extrai sinal se e só se
`NLL(b) < NLL(d4_train)` nos votos retidos.

**Vereditos pré-declarados — nenhum destes é passagem:**

- `b > d3` mas `b < d4` → *"a matriz aprendeu o U na taxa base"*. Nulo, agora
  atribuível.
- `b > d_train` mas `b < d_val` → *"má calibração do ajuste de poucos
  parâmetros, não sinal em `C_a`"*.
- `b ≈ b_perm` → a vantagem não vem de conhecer *aquela* pessoa.
- **Se a run B falhar, nenhum nulo das runs C/D/E é interpretável** e essas runs
  não devem ser feitas.

## 13. Desvios ao protocolo — declarados, com razão e data (2026-09-10)

1. **Régua de votos acrescentada.** Razão: o ICM compara treinos (piso de ruído
   0,0059); a régua compara preditores dentro do mesmo treino.
2. **Reparametrização `sd`.** `C_a` passa a ter 2 parâmetros por anotador
   (competência e viés) em vez dos 4 redundantes de `delta[2,2]` — a softmax por
   linha é invariante a somar uma constante à linha. **No init as duas
   parametrizações são numericamente idênticas** (teste T9, maxdiff 0,00e+00),
   mas as runs da Fase 1 **não são comparáveis bit-a-bit** com as novas. É por
   isso que a run A existe.
3. **`reg_weight=0,01` muda de alvo.** Passa a incidir sobre `s_ind`/`d_ind`:
   `reg·(s_ind² + d_ind²).mean()` no braço livre, `reg·(s_ind²).mean()` no
   congelado — mesmo peso por parâmetro **livre** nos dois braços. Em `s_ind`
   puxa a competência para o valor do painel. Não é numericamente igual ao
   `delta.pow(2).mean()` antigo.
4. **O invariante do modo congelado é em espaço de LOGIT.** O que o handoff
   propunha (`C[:,0,1] == C[:,1,0]`) está **errado** sempre que `d_common ≠ 0`.
   O correcto é `u_a − v_a == 2·d_common` para todo `a`. Em espaço de
   probabilidade, `σ(u_a) − σ(v_a)` **varia** com `s_ind[a]` mesmo com
   `d_ind ≡ 0`, por não-linearidade da sigmóide — sem esta distinção o braço
   congelado *pareceria* continuar a aprender viés.
5. **`d3`/`d4` ajustadas num subconjunto fixo de 600 itens de treino**, não nos
   3 386. Razão: 5 parâmetros sobre ~3 600 votos chegam e sobram, e o custo fica
   igual ao de uma passagem de avaliação em vez de 6×. Subconjunto sorteado uma
   vez com `np.random.default_rng(0)`, independente do RNG global.
6. **`x3` e `x4` incluídos desde o início.** `x3` porque o achado 7 já mediu
   `r(comp, concordância)` = +0,217/+0,231/+0,248, logo a condição do handoff
   está cumprida. `x4` porque uma logística **linear** em `logit(yes_rate)` não
   representa a forma em U que a competência deve absorver — sem `x4`, `C_a`
   bateria `d` justamente por **confirmar a previsão nula**, um "sucesso" que é
   o fracasso previsto.
7. **Oráculos (`d3_val`/`d4_val`)** acrescentados como barra superior, com a
   regra de veredito da §12.
8. **`yes_rate` recalculada só do TREINO**, com suavização de Laplace
   `(n_yes+1)/(n+2)`. A de `behavioral_profiles.csv` é sobre os 3 984 itens,
   logo **inclui os votos de validação** — exactamente os que a régua tenta
   prever. Anotadores sem votos de treino recuam para a taxa base do painel; é
   justo, porque em `b` esses mesmos anotadores têm `C_a` = painel.
9. **Controlo negativo por embaralhamento de ficha na avaliação** (`b_perm`),
   custo zero. O embaralhamento **no treino** fica pré-registado como
   **condicional a um positivo** — só faz sentido descartar um falso positivo
   que exista. A permutação é **estratificada por painel linguístico**, pela
   mesma razão de `make_permuted_soft_labels.py`: os painéis ES e EN são quase
   disjuntos e permutar entre eles destruiria estrutura que não é comportamento.
   Estratos verificados iguais aos do nulo já existente: **437 / 431 / 19**.
10. **Controlo positivo por corrupção sintética.** 120 anotadores, 20 por nível,
    níveis 0/10/20/30/40/50 %. Selecção estratificada por painel linguístico **e
    por número de votos** (sem isso o efeito do nível confunde-se com "quem tem
    poucos votos é mais difícil de detectar"); ordem dos níveis **sorteada
    dentro de cada bloco**, senão o nível de 50 % cairia sempre no anotador com
    mais votos. Medido: 828 votos invertidos, **3,46 %** do total, e as médias de
    votos/anotador por nível ficam entre 26,6 e 28,1.
11. **A run B reconstrói as soft labels a partir da tabela corrompida** — alvo
    de treino **e** gold de validação. Sem isto o braço seria inconsistente: a
    `L_soft` treinaria contra um consenso **limpo** e a camada de confusão
    receberia um **latente-oráculo**, tendo apenas de explicar porque é que 120
    pessoas discordam de algo que nunca foi contaminado. Isso sobrestima a
    detecção e parte a identidade do T2 (`L_aux(identidade) ≡ BCE contra a soft
    label`). Verificado que `soft_labels_from_votes` sobre a tabela **limpa**
    reproduz `soft_labels_taskA_raw.json` a 1,1e-16 nos 3 984 itens (T14a).
    ⚠ **O ICM da run B não se compara com nada.**
12. **Desenho escalonado com duas portas de saída**, em vez das 3 seeds fixas.
    Razão: o nulo é o resultado esperado, e confirmar o esperado três vezes é
    caro. É conservador na direcção certa — só se gasta mais quando se vê algo.

13. **Seed 2025 em vez de 42** (2026-09-11). A run A foi executada com a seed
    que estava no notebook (2025, herdada da última run da Fase 1). É uma run A
    válida — a 42 não tem nada de especial. O que o desenho exige é que os
    braços a **comparar** partilhem a seed, logo as runs B e C passam também a
    2025, e as seeds adicionais (se a Porta 2 abrir) serão 42 e 7.

## 14. ⚠ Armadilha medida: `b` bate `a` no init, sem aprender nada

Medido no teste T12 (2026-09-10, CPU): `NLL(a) = 0,8250` contra
`NLL(b) = 0,7997`, com a ficha **no ponto de partida**.

Causa: `C` no init é `softmax([3,0])`, diagonal 0,9526, e isso **encolhe a
previsão na direcção de 0,5** — efeito de **calibração global**, não de
anotador. E `s_common` é um parâmetro livre, logo o treino pode afinar essa
temperatura à vontade.

**Consequência para o critério:** `b` contra `a` mistura calibração global com
sinal individual e **não serve de prova**. O contraste limpo do sinal
*individual* é `b` contra `b_perm` — as duas têm exactamente a mesma calibração
global e o mesmo conjunto de fichas, e diferem só em cada pessoa receber a **sua**
ficha ou a de outra. Confirmado que no init `b == b_perm` até à última casa (T12f).

`b − a` fica como **descritivo**. Decidem `b − b_perm` e `b − d4_train`.

## 15. Taxa base medida — resposta pré-registada à pergunta do `no_rate`

Medido em `soft_labels_taskA_raw.json`, 3 984 itens (2026-09-10):

```
fracção de VOTOS YES          0,5558
itens com maioria estrita YES 0,5028
unânimes YES 572  |  unânimes NO 373
n_YES/6: {0:373, 1:459, 2:535, 3:614, 4:672, 5:759, 6:572}
```

**YES não é a classe minoritária na Task A** — é ligeiramente maioritária.
E `no_rate ≡ 1 − yes_rate` é a mesma variável: em `d3`/`d4` trocaria apenas o
sinal de β₂, deixando NLL, acerto e ajuste **bit-a-bit idênticos**.

A assimetria real, se existisse, estaria nas **linhas** de `C_a` — a linha do
latente YES é estimada dos ~15 votos YES do anotador e a do NO dos ~12 — não na
taxa marginal. Aqui está equilibrada; o problema é o total ser **27**. E a
parametrização `sd` já partilha força entre as duas linhas via `s_ind`.

## 16. Pendências na tese, decorrentes desta fase

1. `\subsection{Annotator Confusion Matrices...}` em `cap07.tex` diz que a
   competência não correlaciona com nada, com base na correlação **linear**
   (−0,15). **Errado.** Correlaciona **−0,74** com o *desvio à taxa base*
   (forma em U). Corrigir independentemente do resultado das runs.
2. A subsecção do IRT diz *"four independent families have now been evaluated"*
   → são **cinco** com a camada de confusão.
3. Verificar em fonte primária `chu2021common`, `tanno2019regularized` e
   `rodrigues2018crowds` em `lib/refs.bib` — escritas sem verificação, e a
   auditoria do paper já apanhou duas referências fabricadas.

## 17. Plano de runs

| # | Braço | `PARAM` | `FREEZE_BIAS` | `CORRUPT` | Seed | Porta |
|---|---|---|---|---|---|---|
| A | viés livre + régua | `sd` | False | — | 42 | — |
| B | **controlo positivo** | `sd` | False | 120 anot. | 42 | **Porta 1** |
| C | congelado | `sd` | True | — | 42 | **Porta 2** |
| D | congelado | `sd` | True | — | 7 | — |
| E | congelado | `sd` | True | — | 2025 | — |

Fixos: `METHOD="M1"`, `LAMBDA=0,1`, `CONF_LR=1e-2`, 15 épocas, `save_strategy="no"`.

**Porta 1** — se o mecanismo não detectar corrupção a 50 %, a linha está morta:
escreve-se em dois parágrafos e nenhuma run congelada acontece.
**Porta 2** — se a seed 42 congelada reproduzir o colapso previsto, reporta-se
uma seed com a previsão registada de antemão e pára-se.

## 18. Testes que bloqueiam a execução

`EXIST_MEMES/confusion_selftest.py`, **41 verificações**, em CPU, sem
`transformers`. Acrescentados na Fase 2:

- **T9** — `sd` e `delta` produzem matrizes idênticas no init; `sd` não consome RNG.
- **T10** — invariante do congelado em espaço de logit; viés em probabilidade
  varia (artefacto declarado); `d_ind` não é `Parameter` e não recebe gradiente.
- **T11** — `l_aux == −vote_logprobs(...).mean()` exacto; `use_matrices=False`
  devolve a probabilidade crua do item; `perm` muda o resultado.
- **T12** — estratos linguísticos batem 437/431/19; permutação fica dentro do
  estrato; **uma passagem do callback não altera o RNG global**; os 8 preditores
  saem finitos; `b` bate `a` no init (armadilha da §14) e `b == b_perm` no init.
- **T13** — corrupção: 20 por nível, fracção invertida exacta, nível 0 % intocado,
  o manifesto bate a tabela, quem votou não muda.
- **T14** — soft labels reconstruídas dos votos reproduzem o ficheiro do A0 a
  1,1e-16, e corromper move 744/3 984 alvos.

Além disso, o callback tem uma **guarda de RNG activa em todas as runs reais**
(CPU e CUDA): se deslocar o gerador, a run aborta com `RuntimeError` em vez de
terminar 15 épocas incomparáveis.
