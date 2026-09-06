# Relatório de Ablação — Task 2 (Memes), EXIST 2026
### Jow @ EXIST 2026 — Anexo de Tese

**Autor:** Jonathan Santos da Silva
**Escopo:** Task 2 (deteção de sexismo baseada em memes), subtasks A (binária),
B (intenção da fonte) e C (categorias finas, multilabel), sob o paradigma
*Learning with Disagreement* (LeWiDi).
**Framework de ablação:** `EXIST_MEMES/ablation/` (`config.py`, `train_one.py`,
`runner.py`, `aggregate.py`), construído sobre a lógica extraída verbatim de
`memes_model.ipynb` para `memes_core.py`.

---

## 1. Introdução e objetivo

Este documento reporta uma suite de 6 estudos de ablação sobre a arquitetura
multimodal do sistema Jow para a Task 2 do EXIST 2026, com o objetivo de: (i)
quantificar a contribuição de cada componente arquitetural e de cada
hiperparâmetro de treino às 6 métricas de avaliação principais; (ii) distinguir
efeitos reais de ruído de seed, através de um desenho experimental pareado; e
(iii) usar os achados para propor mudanças arquiteturais concretas que
otimizem o conjunto de métricas, com foco especial em investigar por que a
modalidade de visão parece dominar o resultado final e se essa dominância pode
ser reduzida.

A suite foi executada inteiramente numa VM remota dedicada (não a máquina de
desenvolvimento local — ver `CLAUDE.md`, seção "Ambiente de Execução"),
totalizando 36 treinos únicos de 30 épocas cada (3 de FASE 0 + 33 de FASE 1)
mais 18 reaproveitamentos de resultados idênticos (ver seção 5), num total de
75 linhas de resultado.

---

## 2. Metodologia experimental

### 2.1 Configuração baseline

Todos os testes de ablação variam **exatamente um** hiperparâmetro em relação
à seguinte configuração baseline (`ablation/config.py`, `AblationConfig`
default):

| hiperparâmetro | valor baseline |
|---|---|
| `num_epochs` | 30 |
| `warmup_steps` | 0.1 (fração dos passos totais — `transformers==5.6.0`, `warmup_ratio` removido) |
| `task_a_weight` | 2.0 |
| `unfrozen_vision_layers` | 4 |
| `use_vision` | True |
| `use_demographics` | True |
| `demo_feature_set` | "base" (66-d: etnia + escolaridade + país) |
| `demo_norm_mode` | "sum" |
| `per_device_train_batch_size` | 8, `gradient_accumulation_steps=8` (batch efetivo 64) |
| `learning_rate` | 1e-5, `weight_decay=0.05` |
| `max_length` | 128 tokens |

### 2.2 Desenho pareado

Cada configuração é treinada com **3 seeds fixas** (42, 7, 2025), aplicadas de
forma idêntica em: inicialização de pesos (`set_full_determinism(seed)`,
chamada imediatamente antes de instanciar `MultiHeadModel`), amostragem do
`Trainer` (`seed`/`data_seed`), e dropout durante o forward. O **split
treino/validação usa uma seed separada e fixa em todas as configs**
(`VAL_SPLIT_SEED=42`, estratificado pelo voto majoritário de Task A) —
garantindo que toda comparação entre configs A e B, para uma dada seed de
treino, é feita sobre exatamente o mesmo conjunto de validação.

Essa fixação faz do desenho um **desenho pareado por seed**: para cada
comparação (config A vs config B), existem 3 pares (seed 42 de A com seed 42
de B, etc.), e o efeito de interesse é o delta dentro de cada par, não a
diferença entre médias não-pareadas. Um efeito com sinal consistente nas 3
seeds (3/3 mesmo sinal) é evidência mais forte de direção real do que a mera
comparação `|Δmédia| > max(std_A, std_B)` usada numa primeira passada de
análise — esta última é o teste mais fraco possível com n=3 porque descarta o
pareamento. A análise em profundidade (seção 6) usa o delta pareado como
critério principal.

### 2.3 Reprodutibilidade determinística

`memes_core.set_full_determinism(seed)` fixa, nesta ordem: `random.seed`,
`np.random.seed`, `torch.manual_seed` (CPU), `torch.cuda.manual_seed_all`
(todas as GPUs), desativa `cudnn.benchmark`, ativa `cudnn.deterministic` e
`torch.use_deterministic_algorithms(True)`; a variável de ambiente
`CUBLAS_WORKSPACE_CONFIG=":4096:8"` é fixada no import do módulo (precisa
existir antes de qualquer operação CUDA). A função é chamada de novo,
imediatamente antes de `MultiHeadModel(...)`, em cada run — isso "re-zera" os
geradores para o mesmo ponto independentemente do que rodou antes no mesmo
processo.

`model_fingerprint(model)` calcula SHA-256 sobre a concatenação ordenada
(por nome de parâmetro) de todos os tensores de peso, truncado a 16 hex.
Dois modelos com o mesmo fingerprint têm pesos bit-a-bit idênticos.

### 2.4 Vieses de protocolo — declarados explicitamente

Dois vieses estão presentes em **todas** as linhas de resultado, de forma
idêntica entre configs comparadas (o que preserva a validade das comparações
*relativas*, mas precisa ser declarado para qualquer leitura *absoluta*):

**(a) Seleção de época com vazamento de validação.** `load_best_model_at_end=True`
com `metric_for_best_model="task_a_icmsoftnorm_soft"` escolhe, ao fim das 30
épocas, os pesos da época com melhor `task_a_icmsoftnorm_soft` — no mesmo
conjunto de validação sobre o qual todas as métricas finais reportadas (as
outras 5 métricas-chave, e as métricas auxiliares) são calculadas. Isso é
padrão em early-stopping/checkpoint-selection e não é, por si, incomum — mas
significa que os números aqui não são estimativas de generalização não
enviesadas, e sim o melhor resultado de validação dentro da grade de épocas,
selecionado por uma das 6 métricas.

**(b) Ajuste de threshold da Task C, por época, no conjunto de validação.**
`_search_thresholds_task_c` (seção 4.5) roda **a cada chamada de avaliação**
(isto é, a cada época) e o resultado sobrescreve `task_c_icmnorm_hard` — o
código documenta isso explicitamente: *"'task_c_..._hard' é escrito primeiro
[threshold 0.5] e depois SOBRESCRITO pelo bloco de threshold ajustado"*
(`memes_core.py`, comentário da função `_metrics_task_c`). Isto é ajuste de
threshold durante o treino, no set de validação — exatamente o que o
`CLAUDE.md` do projeto proíbe para o modelo final ("ajuste de threshold
deve ser feito apenas pós-treino, para evitar leakage de validação"). Para
comparações *relativas* entre configs de ablação, o viés é idêntico dos dois
lados e portanto não invalida a conclusão de direção — mas **para qualquer
leitura absoluta do valor hard de Task C, este relatório usa preferencialmente
`task_c_icmnorm_hard_default`** (threshold fixo 0.5, sem ajuste), que é logado
em paralelo em todas as linhas. Onde os dois valores (tuned e default)
concordam em sinal e magnitude aproximada, a conclusão é robusta a este viés;
isso é verificado explicitamente para cada achado reportado na seção 6.

**(c) Descontinuidade de `eval_batch_size` entre FASE 0 e FASE 1.** Um exame de
`steps_per_second` no CSV bruto revela que as 3 linhas de baseline (e todas as
7 linhas "config_A" reaproveitadas dela — ver seção 5) foram avaliadas com
`steps_per_second≈1.07–1.10` (batch de avaliação 8, valor original antes da
otimização de velocidade aplicada no meio da suite), enquanto **todas** as
linhas B/C/D de FASE 1 mostram `steps_per_second≈0.30–0.32` (batch de
avaliação 32). Isto é: o braço de referência de toda comparação de ablação foi
avaliado sob uma configuração de batch diferente do braço experimental.
`fp16=True` pode introduzir diferenças de arredondamento na ordem de ~1e-4
entre tamanhos de batch distintos (reduções somadas em ordem diferente) — isso
é 2 a 3 ordens de grandeza abaixo dos efeitos observados nesta suite (0.01 a
0.08, seção 6), então **não compromete as conclusões**, mas é uma diferença
sistemática presente em um dos lados de toda comparação e precisa constar aqui.
Evidência de que o efeito é desprezível: o `fingerprint_init` de uma mesma
seed permanece bit-a-bit idêntico antes e depois da mudança de batch de
avaliação (ver seção 5), e um teste de fumaça (smoke test) rodado com a
config baseline sob `eval_batch_size=32` reproduziu métricas numericamente
idênticas (até a 4ª casa decimal reportada) às obtidas sob `eval_batch_size=8`
para a mesma seed, mudando apenas o tempo de execução (71.99s → 60.77s).

---

## 3. Definições formais

### 3.1 Rótulos suaves (soft labels)

Task A e Task B (cada anotador escolhe **um** rótulo por amostra):

```
get_soft_distribution(votos, rótulos_possíveis)[l] = (nº de votos em l) / (nº total de votos)
```
Devolve `None` (amostra descartada do treino) se não houver nenhum voto válido.
Votos `"UNKNOWN"` são removidos antes da contagem; votos `"-"` (não-sexista) são
remapeados para `"NONE"` antes da contagem — não filtrados, para não inflar
artificialmente as proporções de `DIRECT`/`JUDGEMENTAL` em Task B.

Task C (cada anotador pode escolher **múltiplos** rótulos por amostra):
```
annotations_matrix_to_distribution(listas_por_anotador, rótulos)[l] =
    (nº de ANOTADORES cuja lista contém l) / (nº de anotadores)
```
**Nota importante:** esta distribuição, ao contrário da de Task A/B, **não
soma necessariamente 1** — um anotador que marca 2 categorias contribui 1 voto
para cada uma. É uma distribuição de presença por categoria, não uma
distribuição categórica única.

### 3.2 Vetor demográfico dos anotadores

Para cada grupo de característica `g` (etnia, escolaridade, país — e,
opcionalmente, gênero/idade no `demo_feature_set="extended"`, ver ABL-2), com
vocabulário fixo `V_g`:

```
count_g[v] = |{anotadores da amostra cujo valor de g é v}|,  v ∈ V_g
vetor_bruto = concat(count_ethnicity, count_study, count_country [, count_gender, count_age])
```
Dimensão: 66 (base: 7+6+53) ou 71 (extended: +2+3).

Normalização (`compute_annotator_vector`, parâmetro `norm_mode`, alvo do ABL-6):
```
sum:  v_norm = vetor_bruto / Σ(vetor_bruto)         (se Σ > 1e-9, senão inalterado)
l2:   v_norm = vetor_bruto / ‖vetor_bruto‖₂          (se ‖·‖₂ > 1e-9, senão inalterado)
```

Correção de dado aplicada nesta revisão do código (`_normalize_apostrophes`):
os valores brutos de escolaridade usam apóstrofo curvo (`’`, U+2019, ex.
`"Bachelor’s degree"`), o vocabulário fixo usa apóstrofo reto (`'`, U+0027) —
sem tratar os dois como equivalentes, 64,6% dos votos de escolaridade eram
descartados silenciosamente do vetor. Esta correção está presente em todas as
runs deste relatório (ver seção 6.6 para o impacto retroativo dessa correção
na claim histórica sobre normalização L2).

### 3.3 Injeção de demografia no encoder de texto (gated additive injection)

Seja `h_CLS` o embedding do token `[CLS]` de saída do XLM-RoBERTa-large,
projetado para a dimensão compartilhada de 512 via `text_proj` (Linear +
LayerNorm):
```
t = LayerNorm(W_t · h_CLS)                              ∈ ℝ⁵¹²
```
Se `use_demographics=True` e um `demo_vec` `d ∈ ℝ^{66 ou 71}` está disponível:
```
gate = σ(W_g · d)                                        ∈ (0,1)⁵¹²   (elemento a elemento)
p    = LayerNorm(W_p · d)                                ∈ ℝ⁵¹²
t'   = t + gate ⊙ p                                       (soma elemento a elemento, ⊙ = produto de Hadamard)
```
`t'` substitui `t` como a metade "texto" do vetor de fusão. Note que esta
injeção acontece **depois** do `LayerNorm` interno de `text_proj` — a soma
`gate ⊙ p` pode inflar a magnitude de `t'` além da escala normalizada de `t`,
antes da normalização global de fusão (seção 3.4). Ver seção 7, item 4, para a
implicação disso.

### 3.4 Fusão multimodal

```
v = LayerNorm(W_v · h_CLS^vision)                        ∈ ℝ⁵¹²   (se use_vision=True)
v = 0                                                     ∈ ℝ⁵¹²   (se use_vision=False)

fused = LayerNorm(concat(t', v))                          ∈ ℝ¹⁰²⁴
fused = Dropout(fused, p=0.1)
```
Fusão por **concatenação simples** — sem cross-attention (decisão de projeto
documentada: cross-attention colapsou para padrões quase uniformes no dataset
pequeno, ~5.000 memes). Quando `use_vision=False`, o `LayerNorm` de `fused`
é computado sobre 1024 dimensões das quais 512 são zeros constantes — ver
seção 7, item 1, para a implicação disso na interpretação do ABL-5.

### 3.5 Funções de perda por task

**Task A** (BCE ponderada, sigmoid → P(NO)):
```
L_a = task_a_weight · BCEWithLogits(logit_a, label_a)     (só sobre amostras com voto válido)
```
`task_a_weight` é o alvo do ABL-3 (baseline 2.0).

**Task B** (cross-entropy contra soft label, não hard):
```
L_b = -Σ_c (p_true[c] · log(softmax(logits_b)[c]))  , média sobre o batch
```
Isto é entropia cruzada generalizada para distribuição-alvo suave — não é
classificação com rótulo único.

**Task C** (BCE por categoria, multilabel):
```
L_c = mean_categoria( BCEWithLogits(logits_c[categoria], label_c[categoria]) )
```
6 saídas sigmoid independentes, uma por categoria.

**Loss total:**
```
L = Σ_{task ∈ {A,B,C} com voto válido no batch} L_task
```
Uma task inteira é omitida do batch se nenhuma amostra dele tiver voto válido
para ela (sentinela -1.0/-100.0 na coluna de rótulo) — isto acontece porque
Task B e C dependem de anotadores terem respondido "YES" em Task A
primeiro (hierarquia — ver `TASK_2_2_HIERARCHY`/`TASK_2_3_HIERARCHY`).

### 3.6 Métricas de avaliação (PyEvALL)

As métricas ICM / ICM-Soft / ICM-Norm / ICM-Soft-Norm usadas neste relatório
**não estão implementadas neste repositório** — são calculadas externamente
pela biblioteca `PyEvALL` (versão 0.1.78, confirmada na VM), oficialmente
adotada pelo EXIST/CLEF para avaliação sob o paradigma LeWiDi. Este relatório
não reproduz a definição matemática interna do ICM por não tê-la verificado
contra a implementação — o que se pode afirmar com certeza, a partir do código
de integração (`memes_core.py`, seção 14), é:

- Maior valor = melhor (`greater_is_better=True` usado em `metric_for_best_model`).
- `ICM`/`ICMSoft` não são limitadas a `[0,1]`; `ICMNorm`/`ICMSoftNorm` são as
  variantes normalizadas (usadas como métricas-chave neste relatório).
- As variantes "soft" avaliam a predição como distribuição de probabilidade
  (`{label: prob}` por amostra); as "hard" avaliam uma decisão binarizada
  (rótulo único para Task A/B, lista de rótulos para Task C).
- O cálculo é *hierarchy-aware*: `PyEvALLUtils.PARAM_HIERARCHY` recebe
  `TASK_2_2_HIERARCHY = {"YES": ["DIRECT","JUDGEMENTAL"], "NO": []}` para Task
  B, e `TASK_2_3_HIERARCHY` (idem, com as 5 categorias finas) para Task C —
  penalizações incorretas dentro da árvore de rótulos são tratadas de forma
  diferente de confusões fora da hierarquia.
- Task C também reporta `FMeasure` e `CrossEntropy` (Task A/B) como métricas
  auxiliares, não usadas como critério de seleção neste relatório.

Cada chamada ao PyEvALL grava um par de arquivos JSON temporários
(predição/gold) e invoca `PyEvALLEvaluation().evaluate(...)`; falhas são
capturadas e retornam `{}` (métrica ausente, não um crash) — ver seção 8 para
a implicação disso em linhas com métricas faltantes (não ocorreu nesta suite,
mas é um guard ativo no código).

**Nota de rastreabilidade:** os IDs passados ao PyEvALL são índices
posicionais (`str(i) for i in range(len(probs))`), não os `id_EXIST`
originais do dataset — inofensivo porque predição e gold são construídos a
partir do mesmo array na mesma ordem, mas significa que os JSONs
intermediários não são rastreáveis de volta a uma amostra específica sem
reconstrução manual da ordem do batch de avaliação.

### 3.7 Busca de threshold da Task C (coordinate descent guloso)

`_search_thresholds_task_c` ajusta o threshold de decisão de cada uma das 6
categorias de Task C, uma de cada vez, mantendo as demais fixas no melhor
valor encontrado até o momento (busca gulosa por coordenada, não busca
conjunta):

```
threshold[0..5] ← 0.5 (inicial, todas as categorias)
para cada categoria c em [NONE, IDEOLOGICAL-INEQUALITY, MISOGYNY-NON-SEXUAL-VIOLENCE,
                           SEXUAL-VIOLENCE, OBJECTIFICATION, STEREOTYPING-DOMINANCE]:
    melhor_icm ← -∞
    para cada t em THRESHOLD_GRID = {0.05, 0.10, 0.15, ..., 0.60}  (12 valores):
        threshold[c] ← t  (demais fixos)
        icm ← PyEvALL(predições binarizadas com threshold atual, hierarquia Task C)
        se icm > melhor_icm: melhor_icm ← icm; melhor_t ← t
    threshold[c] ← melhor_t
avaliação final com threshold[0..5] travado no melhor de cada categoria
```
Custo: 6 categorias × 12 valores de grade = 72 chamadas PyEvALL + 1 avaliação
final = **73 chamadas por avaliação (por época)**. Como a busca é gulosa e
sequencial (não conjunta), o resultado depende da ordem de iteração das
categorias — thresholds de categorias processadas depois são otimizados
condicionalmente aos já fixados antes, não são um ótimo global conjunto. Este
processo roda a cada época de cada uma das 75 linhas de resultado — é o
principal custo computacional adicional da avaliação de Task C, embora (ver
`CLAUDE.md` / investigação de tempo de execução) não seja o fator dominante do
tempo total de treino, que é dominado pelo forward/backward pass do
XLM-R-large + CLIP ao longo de 30 épocas.

---

## 4. Configuração da suite de ablação

| ID | Hiperparâmetro variado | Variantes | Config_A = baseline? |
|---|---|---|---|
| ABL-1 | `warmup_steps` | 0.1 / 0.0 / 0.2 | sim |
| ABL-2 | `demo_feature_set` | base (66-d) / extended (71-d, +gênero+idade) | sim |
| ABL-3 | `task_a_weight` | 2.0 / 1.0 / 3.0 | sim |
| ABL-4 | `unfrozen_vision_layers` | 4 / 0 / 8 | sim |
| ABL-5 | `use_vision` × `use_demographics` | (T,T) / (T,F) / (F,T) / (F,F) | sim |
| ABL-6 | `demo_norm_mode` | sum / l2 | sim |
| ABL-INJ | `demo_injection_mode` × `demo_feature_set` | (current/renorm/head_inject) × (base/extended) | não* |
| ABL-INJ-CTRL | `demo_shuffle` sobre `head_inject` base | permutado vs real | não |

Cada variante roda 3 seeds (42, 7, 2025). Quando uma variante é
arquiteturalmente idêntica à baseline (`config_A` em toda ablação, por
construção — ver `config_is_baseline_equivalent`), o resultado da FASE 0 é
reaproveitado sem retreino (`runner.py::_find_reusable_baseline_row`),
verificado pelo `fingerprint_init` idêntico (seção 5).

\* ABL-INJ e ABL-INJ-CTRL ativam `run_collapse_diagnostics`, o que as torna
não-equivalentes à baseline por construção — nenhuma variante reaproveita a FASE 0,
todas treinam. Isso é deliberado: sem treino não há modelo em memória para
diagnosticar, e a config-âncora ficaria sem diagnóstico. A equivalência
arquitetural de `inj_current_base` com a baseline é verificada a posteriori pelo
`fingerprint_init` (seção 6.7), não assumida.

Total: 6 ablações × 17 variantes-template × 3 seeds = 51 linhas de ablação +
3 de baseline = **54 linhas**, correspondendo a **36 treinos únicos** (3 FASE
0 + 33 FASE 1 novos) e **18 reaproveitamentos**. As ablações posteriores ABL-INJ
(6 variantes) e ABL-INJ-CTRL (1 variante) acrescentam **21 treinos** e 21 linhas,
totalizando **75 linhas** e **57 treinos únicos**.

---

## 5. Reprodutibilidade: evidência de fingerprints

Toda linha cujo `config_name` termina em `config_A` (arquiteturalmente igual
à baseline) tem `fingerprint_init` **bit-a-bit idêntico** ao da linha
`baseline` da mesma seed:

| seed | `fingerprint_init` (idêntico em: baseline, abl1_A, abl2_A, abl3_A, abl4_A, abl5_A, abl6_A) |
|---|---|
| 42 | `6cd0ef763f16d975` |
| 7 | `e8f051f9614d9927` |
| 2025 | `9502837aaa79b209` |

Isto confirma, empiricamente e não apenas por inspeção de código, que
`set_full_determinism(seed)` produz inicializações de peso idênticas entre
execuções de processo separadas — os 7 conjuntos de pesos-A foram criados em
7 chamadas de `run_single` diferentes (dias diferentes, em alguns casos), e
convergem no mesmo hash. Isto também é a evidência de que o reaproveitamento
de resultados (`_find_reusable_baseline_row`) é seguro: as 18 linhas
reaproveitadas não são apenas "assumidas iguais" — o fingerprint confirma que
o modelo que as gerou é, byte a byte, o mesmo que teria sido instanciado se o
retreino tivesse ocorrido.

Note-se que uma linha com `fingerprint_init` idêntico e métricas idênticas
**não é anomalia nem corrupção de dado** — é o comportamento correto e
esperado do mecanismo de reaproveitamento, exatamente porque as 7 configs
citadas acima diferem da baseline apenas em campos que essa ablação
especificamente varia nas *outras* variantes (B, C, D), não na variante A.

`fingerprint_final` também é idêntico entre as 7 linhas "config_A" para uma
mesma seed — mas isso não é evidência adicional de determinismo: são,
literalmente, a mesma linha copiada. `runner.py::run_phase_1` (via
`_find_reusable_baseline_row`) copia o dicionário de resultado inteiro da
linha `baseline` correspondente e só sobrescreve `config_name`/`seed` antes de
gravar (`row = dict(reused); row["config_name"] = cfg.name; ...`) — nenhum
processo de treino novo roda para essas 6 linhas extra. A única evidência de
determinismo real (execuções de processo independentes convergindo ao mesmo
ponto) está no `fingerprint_init`, comparado entre a run original da FASE 0 e
qualquer run de FASE 1 que **não** tenha sido reaproveitada mas ainda assim
comece de uma config arquiteturalmente igual à baseline — o que, por
construção da suite (toda ablação reaproveita seu próprio `config_A`), não
ocorre neste conjunto de dados. A validação apresentada aqui é, portanto, de
que o mecanismo de reaproveitamento em si é seguro (a linha copiada de fato
corresponde a pesos idênticos, confirmados pelo fingerprint da execução
original), não uma segunda confirmação independente de determinismo.

---

## 6. Resultados por ablação

### 6.1 ABL-1 — Schedule de warmup

**Hipótese testada:** um warmup mais longo (fração maior dos passos totais
com learning rate crescendo linearmente de 0 até o valor alvo) estabiliza o
início do fine-tuning do XLM-R-large + CLIP, hipoteticamente melhorando
métricas finais.

### ABL-1 — 0.1 (baseline) vs 0.0

Deltas por seed (0.0 − 0.1):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | -0.0054 | -0.0092 | +0.0047 | -0.0033 | não |
| B soft | -0.0048 | +0.0036 | +0.0054 | +0.0014 | não |
| C soft | -0.0054 | -0.0225 | +0.0059 | -0.0073 | não |
| A hard | -0.0108 | -0.0034 | -0.0225 | -0.0122 | **sim** |
| B hard | -0.0249 | -0.0014 | +0.0050 | -0.0071 | não |
| C hard (tuned) | -0.0099 | -0.0198 | +0.0195 | -0.0034 | não |
| C hard (thr=0.5) | -0.0189 | -0.0245 | +0.0084 | -0.0117 | não |

### ABL-1 — 0.1 (baseline) vs 0.2

Deltas por seed (0.2 − 0.1):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | -0.0009 | -0.0002 | +0.0099 | +0.0029 | não |
| B soft | +0.0000 | +0.0005 | +0.0084 | +0.0030 | **sim** |
| C soft | -0.0013 | +0.0037 | +0.0079 | +0.0035 | não |
| A hard | +0.0050 | +0.0033 | -0.0033 | +0.0017 | não |
| B hard | -0.0094 | +0.0118 | +0.0027 | +0.0017 | não |
| C hard (tuned) | -0.0214 | +0.0008 | +0.0076 | -0.0043 | não |
| C hard (thr=0.5) | -0.0087 | -0.0043 | +0.0057 | -0.0024 | não |

**Interpretação:** de 14 comparações métrica×par testadas, apenas 2 mostram
sinal consistente (3/3), ambas de magnitude pequena (≤0.012) e sem
confirmação cruzada em métricas relacionadas (task_a hard consistente mas
task_a soft não; task_b soft consistente mas task_b hard não). **Não há
evidência robusta de que o schedule de warmup, na faixa 0–20% dos passos
totais, tenha efeito prático.** Hipótese rejeitada.

### 6.2 ABL-2 — Vetor demográfico expandido

**Hipótese testada:** adicionar gênero e idade dos anotadores ao vetor
demográfico (66-d → 71-d) melhora a capacidade do modelo de capturar
disagreement correlacionado com perfil demográfico.

Deltas por seed (extended − base):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | -0.0103 | +0.0018 | +0.0132 | +0.0016 | não |
| B soft | +0.0054 | +0.0054 | +0.0135 | **+0.0081** | **sim** |
| C soft | -0.0180 | -0.0198 | +0.0012 | -0.0122 | não |
| A hard | -0.0051 | +0.0115 | -0.0333 | -0.0089 | não |
| B hard | -0.0472 | -0.0076 | +0.0079 | -0.0156 | não |
| C hard (tuned) | -0.0511 | -0.0400 | -0.0055 | **-0.0322** | **sim** |
| C hard (thr=0.5) | -0.0435 | -0.0302 | -0.0084 | **-0.0274** | **sim** |

**Interpretação:** dois efeitos reais e robustos, em direções opostas —
task_b soft sobe de forma consistente (a informação de gênero/idade ajuda a
inferir a intenção da fonte quando o modelo já sabe que é sexista), enquanto
task_c hard cai de forma consistente **tanto sob threshold ajustado quanto
sob threshold fixo 0.5** — a queda não é artefato do viés de threshold da
seção 2.4(b), é um efeito real da mudança de features. Task_a e task_c soft
não mostram sinal consistente.

**Ressalva de qualidade de dado:** o desvio-padrão não-pareado de
`abl2_config_B_demo_extended` em task_c soft é 0.0003 (valores por seed:
0.27237 / 0.27213 / 0.27271), ordem de grandeza menor que a variação típica de
~0.01–0.02 observada em outras configs na mesma métrica. Isto é consistente
com — mas não confirma — a cabeça de saída de Task C convergindo para uma
distribuição de probabilidade quase-constante entre amostras sob o demo_vec
extended (um modo de falha diferente de "a feature piora a métrica", mais
próximo de "a feature faz a cabeça parar de discriminar entre amostras"). Não
foi possível confirmar isso examinando as probabilidades por categoria porque
o diretório temporário de predições do PyEvALL (`tempfile.TemporaryDirectory`)
é apagado ao fim de cada avaliação — este relatório registra a hipótese, não
uma confirmação, e recomenda salvar os `logits_c` do checkpoint final numa
run futura antes de reportar a queda de task_c hard como puramente um trade-
off informativo (poderia ser, em parte, um artefato de colapso).

### 6.3 ABL-3 — Peso da loss da Task A

**Hipótese testada:** Task A (binária) é a task mais bem-representada nos
dados; reponderar sua contribuição à loss total (1×, 2× baseline, 3×) deveria
mover o trade-off entre tasks de forma previsível.

Deltas por seed (variante − baseline 2.0):

| comparação | métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|---|
| 1.0 vs 2.0 | A soft | +0.0060 | -0.0003 | +0.0053 | +0.0037 | não |
| 1.0 vs 2.0 | B soft | +0.0009 | -0.0005 | +0.0018 | +0.0007 | não |
| 1.0 vs 2.0 | C soft | -0.0033 | -0.0042 | +0.0093 | +0.0006 | não |
| 1.0 vs 2.0 | A hard | -0.0007 | +0.0034 | -0.0142 | -0.0038 | não |
| 1.0 vs 2.0 | B hard | -0.0047 | +0.0080 | +0.0135 | +0.0056 | não |
| 1.0 vs 2.0 | C hard (tuned) | -0.0049 | +0.0013 | +0.0116 | +0.0027 | não |
| 1.0 vs 2.0 | C hard (thr=0.5) | -0.0048 | -0.0116 | +0.0093 | -0.0024 | não |
| 3.0 vs 2.0 | A soft | -0.0014 | -0.0039 | -0.0042 | **-0.0032** | **sim** |
| 3.0 vs 2.0 | B soft | -0.0020 | -0.0009 | +0.0007 | -0.0007 | não |
| 3.0 vs 2.0 | C soft | +0.0058 | -0.0047 | -0.0038 | -0.0009 | não |
| 3.0 vs 2.0 | A hard | +0.0067 | -0.0058 | -0.0159 | -0.0050 | não |
| 3.0 vs 2.0 | B hard | -0.0181 | +0.0054 | -0.0173 | -0.0100 | não |
| 3.0 vs 2.0 | C hard (tuned) | +0.0194 | -0.0054 | -0.0013 | +0.0042 | não |
| 3.0 vs 2.0 | C hard (thr=0.5) | +0.0036 | -0.0082 | -0.0016 | -0.0021 | não |

**Interpretação:** de 14 comparações, apenas 1 mostra sinal consistente
(peso 3.0 vs 2.0 em task_a soft, -0.0032 — magnitude pequena, e nem 1.0 vs 2.0
mostra sinal na mesma métrica, então não há relação monotônica). **Negative
result:** reponderar a loss de Task A entre 1×/2×/3× não move nenhuma métrica
de forma útil, apesar de a magnitude absoluta da loss total variar
substancialmente entre pesos (≈2.14–2.19 com peso 1.0, ≈2.79–2.83 com peso 2.0,
≈3.44–3.49 com peso 3.0 — coluna `loss` do CSV bruto, 3 seeds cada). Isto sugere que o gargalo de
otimização não está na ponderação escalar entre tasks — todas convergem de
forma semelhante independentemente do peso relativo de Task A.

### 6.4 ABL-4 — Camadas do CLIP destravadas

**Hipótese testada:** o encoder de visão (CLIP ViT-L/14), parcialmente
congelado por padrão (últimas 4 de 24 camadas treináveis), pode estar
subutilizado — destravar mais camadas deveria melhorar a capacidade do modelo
de extrair sinal visual relevante para sexismo.

Deltas por seed (variante − baseline 4 camadas):

| comparação | métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|---|
| 0 vs 4 | A soft | -0.0164 | -0.0036 | -0.0044 | **-0.0081** | **sim** |
| 0 vs 4 | B soft | +0.0069 | +0.0048 | -0.0007 | +0.0037 | não |
| 0 vs 4 | C soft | -0.0334 | -0.0485 | -0.0186 | **-0.0335** | **sim** |
| 0 vs 4 | A hard | +0.0208 | -0.0183 | -0.0050 | -0.0008 | não |
| 0 vs 4 | B hard | -0.0434 | +0.0013 | -0.0403 | -0.0275 | não |
| 0 vs 4 | C hard (tuned) | -0.0675 | -0.0574 | -0.0404 | **-0.0551** | **sim** |
| 0 vs 4 | C hard (thr=0.5) | -0.0614 | -0.0469 | -0.0422 | **-0.0502** | **sim** |
| 8 vs 4 | A soft | +0.0047 | -0.0025 | +0.0184 | +0.0068 | não |
| 8 vs 4 | B soft | +0.0171 | +0.0189 | +0.0089 | **+0.0150** | **sim** |
| 8 vs 4 | C soft | -0.0168 | -0.0097 | +0.0049 | -0.0072 | não |
| 8 vs 4 | A hard | +0.0255 | +0.0213 | +0.0181 | **+0.0216** | **sim** |
| 8 vs 4 | B hard | -0.0095 | -0.0034 | +0.0092 | -0.0012 | não |
| 8 vs 4 | C hard (tuned) | +0.0052 | -0.0122 | +0.0283 | +0.0071 | não |
| 8 vs 4 | C hard (thr=0.5) | +0.0070 | -0.0113 | +0.0250 | +0.0069 | não |

**Interpretação: sinal real forte e robusto.** Congelar o CLIP inteiramente
(0 camadas destravadas) derruba task_a soft, task_c soft e task_c hard **com
sinal consistente e — crucialmente — a queda em task_c hard é confirmada tanto
sob threshold ajustado quanto sob threshold fixo 0.5**, eliminando a
possibilidade de ser artefato do viés (b) da seção 2.4. Destravar mais camadas
(8 vs 4) melhora task_b soft e task_a hard de forma consistente, sem prejudicar
nenhuma métrica de forma consistente. **Conclusão: o encoder de visão precisa
de fine-tuning parcial para funcionar bem; a configuração com 8 camadas
destravadas domina ou empata com a baseline (4 camadas) em toda métrica que
mostra sinal.**

### 6.5 ABL-5 — Ablação de modalidades

**Correção de rotulagem, feita nesta revisão:** o texto (XLM-RoBERTa-large,
todas as 24 camadas sempre treináveis — `self.encoder = XLMRobertaModel.from_pretrained(...)`
sem nenhum `requires_grad=False` aplicado) está presente em **todas as 4
variantes desta ablação**. Não existe, nesta suite, um braço "sem texto". O
que foi de fato testado, e como cada nome de config no CSV deve ser lido:

| nome no CSV | conteúdo real |
|---|---|
| `vision_demo` (baseline) | texto + visão + demografia |
| `vision_only` | texto + visão, sem demografia |
| `demo_only` | texto + demografia, sem visão |
| `text_only` | apenas texto |

**Viés de implementação a declarar (seção 3.4):** quando `use_vision=False`,
o vetor `vision_cls` é `zeros_like(text_cls)`, e o `LayerNorm` de fusão
(`fusion_norm`) é computado sobre os 1024-d completos, dos quais 512 são zeros
constantes. Isto distorce a normalização estatística da metade real (texto) e
faz as 3 cabeças gastarem metade da sua largura de entrada em constantes. As
quedas reportadas abaixo, ao remover visão, são portanto **remoção de
informação somada a um caminho de fusão degenerado** — a direção do efeito
é confiável (removida em toda comparação de forma sistemática e consistente),
mas a magnitude provavelmente está inflada em relação ao que se obteria com
uma fusão que lida corretamente com modalidades ausentes (proposta de correção
na seção 7, item 1).

Deltas por seed (variante − baseline "full"):

| comparação | métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|---|
| vision_only − full | A soft | -0.0024 | +0.0122 | +0.0103 | +0.0067 | não |
| vision_only − full | B soft | +0.0077 | -0.0001 | -0.0098 | -0.0007 | não |
| vision_only − full | C soft | -0.0085 | +0.0046 | +0.0195 | +0.0052 | não |
| vision_only − full | A hard | +0.0040 | +0.0373 | -0.0250 | +0.0054 | não |
| vision_only − full | B hard | -0.0434 | +0.0014 | -0.0260 | -0.0227 | não |
| vision_only − full | C hard (tuned) | -0.0222 | +0.0257 | +0.0275 | +0.0103 | não |
| vision_only − full | C hard (thr=0.5) | -0.0164 | +0.0120 | +0.0196 | +0.0051 | não |
| demo_only − full | A soft | -0.0324 | -0.0142 | -0.0258 | **-0.0242** | **sim** |
| demo_only − full | B soft | -0.0030 | +0.0007 | -0.0208 | -0.0077 | não |
| demo_only − full | C soft | -0.0496 | -0.0378 | -0.0583 | **-0.0486** | **sim** |
| demo_only − full | A hard | +0.0097 | +0.0266 | -0.0469 | -0.0035 | não |
| demo_only − full | B hard | -0.0520 | -0.0267 | -0.0388 | **-0.0392** | **sim** |
| demo_only − full | C hard (tuned) | -0.1137 | -0.0429 | -0.0838 | **-0.0801** | **sim** |
| demo_only − full | C hard (thr=0.5) | -0.1051 | -0.0371 | -0.0878 | **-0.0767** | **sim** |
| text_only − full | A soft | -0.0496 | -0.0196 | -0.0256 | **-0.0316** | **sim** |
| text_only − full | B soft | -0.0171 | -0.0031 | -0.0135 | **-0.0112** | **sim** |
| text_only − full | C soft | -0.0677 | -0.0562 | -0.0489 | **-0.0576** | **sim** |
| text_only − full | A hard | -0.0321 | +0.0114 | -0.0357 | -0.0188 | não |
| text_only − full | B hard | -0.0493 | -0.0431 | -0.0459 | **-0.0461** | **sim** |
| text_only − full | C hard (tuned) | -0.1074 | -0.0920 | -0.0479 | **-0.0824** | **sim** |
| text_only − full | C hard (thr=0.5) | -0.0978 | -0.0928 | -0.0490 | **-0.0799** | **sim** |

#### Contraste isolado da visão: `text_only` → `vision_only`

As comparações da tabela acima usam `full` (texto+visão+demografia) como
referência, o que significa que `full − text_only` mede a adição de **duas**
coisas simultaneamente (visão *e* demografia). Para isolar exclusivamente a
contribuição da modalidade visual, a comparação correta é
`text_only` → `vision_only`: ambos os braços têm `use_demographics=False`,
diferindo *apenas* em `use_vision`.

| métrica | texto só | texto + visão | Δ médio | Δ relativo | seed 42 | seed 7 | seed 2025 | consistente |
|---|---|---|---|---|---|---|---|---|
| A soft | 0.4609 | 0.4992 | **+0.0384** | +8.3% | +0.0473 | +0.0319 | +0.0359 | **sim** |
| B soft | 0.3857 | 0.3962 | **+0.0105** | +2.7% | +0.0248 | +0.0030 | +0.0037 | **sim** |
| C soft | 0.2270 | 0.2898 | **+0.0628** | **+27.7%** | +0.0592 | +0.0608 | +0.0685 | **sim** |
| A hard | 0.5073 | 0.5315 | **+0.0243** | +4.8% | +0.0361 | +0.0260 | +0.0107 | **sim** |
| B hard | 0.3975 | 0.4209 | **+0.0234** | +5.9% | +0.0059 | +0.0445 | +0.0199 | **sim** |
| C hard (tuned) | 0.3233 | 0.4160 | **+0.0928** | **+28.7%** | +0.0852 | +0.1177 | +0.0754 | **sim** |
| C hard (thr=0.5) | 0.3050 | 0.3899 | **+0.0849** | **+27.9%** | +0.0815 | +0.1047 | +0.0687 | **sim** |

**Este é o resultado mais forte de toda a suite de ablação: 7 de 7 métricas
com sinal consistente nas 3 seeds, todas positivas, sem uma única exceção.**
A adição do encoder de visão melhora todas as métricas-chave em todas as
seeds testadas.

Observações sobre a distribuição do ganho entre tasks:

- **Task C tem, de longe, o maior ganho relativo** (+27,7% soft, +28,7% hard
  tuned, +27,9% hard com threshold fixo). Isto é coerente semanticamente: as
  categorias finas de sexismo em memes — notadamente `OBJECTIFICATION` e
  `SEXUAL-VIOLENCE` — são frequentemente expressas pelo conteúdo visual da
  imagem, não pelo texto sobreposto, que pode ser neutro ou até irônico em
  relação à imagem.
- **Task B tem o menor ganho** (+2,7% soft, +5,9% hard) — também coerente: a
  intenção da fonte (JUDGEMENTAL vs DIRECT) é uma distinção primariamente
  pragmático-linguística, menos dependente do canal visual.
- **Task A fica no meio** (+8,3% soft, +4,8% hard): a decisão binária
  "é sexista?" beneficia-se da imagem, mas menos que a atribuição de categoria.
- **O ganho em Task C hard sobrevive ao controle de threshold fixo**
  (+0,0849 com threshold 0.5 vs +0,0928 com threshold ajustado) — a melhoria
  não é artefato do viés de ajuste de threshold descrito na seção 2.4(b).

**Ressalvas para interpretação (obrigatórias antes de citar estes números):**

1. **A magnitude é um teto, não uma estimativa limpa.** O braço `text_only`
   roda com `use_vision=False`, o que aciona o caminho degenerado descrito na
   seção 3.4 (`vision_cls = zeros_like(text_cls)`, e `fusion_norm` computado
   sobre 1024-d dos quais 512 são zeros constantes). Portanto o braço sem
   visão sofre simultaneamente perda de informação visual **e** degradação do
   caminho de fusão. A direção do efeito (positiva, 3/3 em todas as métricas)
   é robusta; os valores de +28% em Task C devem ser lidos como limite
   superior. Separar as duas causas exige a correção proposta na seção 8.1.
2. **Este resultado autoriza dizer que "o modelo extrai informação relevante
   da imagem", mas NÃO autoriza dizer que "a imagem carrega mais informação
   que o texto".** Como o texto está presente em ambos os braços (seção 6.5,
   correção de rotulagem), a suite mede o ganho *incremental* da visão sobre
   uma base textual, não a contribuição absoluta de cada modalidade. Um
   ranking entre modalidades exigiria um braço visão-sem-texto, que não existe
   nesta suite (ver seção 9, limitação 1).

**Interpretação — achado central desta suite, reformulado:**

1. **Remover apenas a demografia (full → vision_only) não produz sinal
   consistente em NENHUMA das 7 métricas.** Todas as 7 comparações têm sinais
   trocando entre seeds. Isto contraria a hipótese "demografia sempre ajuda"
   e é consistente com o padrão observado no ABL-2 (demo só ajuda em métricas
   específicas, com trade-off).

2. **Remover visão (full → demo_only, full → text_only) derruba quase todas
   as métricas de forma robusta e consistente**, com magnitudes muito
   maiores que qualquer outro efeito medido nesta suite: até -0.0801 (task_c
   hard tuned) e -0.0767 (task_c hard threshold fixo — **confirmando que o
   efeito não é artefato do viés de threshold**). `demo_only` e `text_only`
   têm magnitudes de queda muito próximas entre si (ex. task_c hard_default:
   -0.0767 vs -0.0799) — a diferença entre "ter demografia sem visão" e "não
   ter nem demografia nem visão" é pequena perto do efeito de perder a visão.

3. **A demografia sozinha (sem visão) não compensa a ausência de visão, nem
   agrega de forma clara sobre o texto puro** — `demo_only` não é
   consistentemente melhor que `text_only` em nenhuma métrica (comparação
   não tabulada aqui, mas visível pela proximidade dos deltas acima).

**Reformulação do achado (a formulação original, "visão domina o resultado",
não é sustentada por este desenho — não existe braço sem texto):** a leitura
correta e sustentada pelos dados é **"adicionar visão a texto produz ganhos
grandes e consistentes, especialmente em task_c; adicionar demografia a
texto+visão não produz efeito mensurável em nenhuma métrica sob a injeção
atual (gate + soma pós-LayerNorm)."**

### 6.6 ABL-6 — Normalização do vetor demográfico

**Hipótese testada:** o modo de normalização do `demo_vec` (dividir pela soma
total de votos vs. dividir pela norma L2) afeta como o sinal demográfico chega
ao modelo — motivada por uma nota histórica do projeto (`CLAUDE.md`) afirmando
que a normalização L2 "destrói" o sinal.

Deltas por seed (L2 − sum):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | +0.0002 | -0.0044 | +0.0074 | +0.0011 | não |
| B soft | +0.0003 | +0.0050 | +0.0066 | **+0.0040** | **sim** |
| C soft | +0.0018 | -0.0068 | +0.0094 | +0.0014 | não |
| A hard | -0.0075 | +0.0017 | -0.0100 | -0.0053 | não |
| B hard | -0.0195 | +0.0041 | +0.0090 | -0.0021 | não |
| C hard (tuned) | -0.0116 | -0.0229 | -0.0017 | **-0.0121** | **sim** |
| C hard (thr=0.5) | +0.0087 | -0.0161 | +0.0023 | -0.0017 | não |

**Interpretação:** dois sinais consistentes, mas pequenos (≤0.012) e **em
direções opostas** entre métricas relacionadas — L2 "ganha" em task_b soft e
"perde" em task_c hard tuned, mas esta última perda **não se confirma** sob
threshold fixo 0.5 (não consistente), então é plausivelmente, ao menos em
parte, interação com o viés de threshold da seção 2.4(b), não puramente
mudança de sinal informativo. **Não há, nestes dados, uma claim defensável de
que L2 "destrói" o sinal demográfico.**

### Reconciliação com a nota histórica do projeto

`CLAUDE.md` registra: *"Normalização por soma dos vetores one-hot
demográficos preservou o sinal; normalização L2 destruiu-o"*. A origem desta
frase foi localizada em `memes_model.ipynb`, célula markdown da seção 8
("Vetor demográfico dos anotadores"): *"dividimos a lista inteira pela soma
total [...] isso é importante: normalizar dividindo pela soma preserva o
sinal; testamos normalizar por norma L2 e o sinal demográfico foi
destruído"* — escrita **na mesma seção, imediatamente antes** da nota que
documenta a correção de `_normalize_apostrophes` (aplicada em 2026-08-01).

Isto significa que a claim original foi testada sobre uma versão do vetor
demográfico **com um bug de dado conhecido e já corrigido**: 64,6% dos votos
de escolaridade eram descartados silenciosamente por mismatch entre apóstrofo
curvo (dados) e reto (vocabulário), deixando a fatia de escolaridade do vetor
quase sempre zerada (3.957 de 3.984 amostras afetadas). Um teste de
normalização L2 sobre um vetor já majoritariamente esparso (uma fatia inteira
zerada) tende a concentrar a norma nas poucas posições não-zero remanescentes
de forma diferente de um vetor completo — o teste histórico não é comparável
ao teste atual (vetor corrigido, 3 seeds pareadas, protocolo determinístico
documentado).

**Conclusão: a claim histórica não é reproduzida sob o protocolo atual. A
explicação mais provável não é uma contradição a resolver, mas sim que a
claim original foi medida sobre dados diferentes (vetor com bug de
correspondência de apóstrofo ainda não corrigido), muito provavelmente numa
única execução não documentada com seeds. Recomenda-se que versões futuras do
`CLAUDE.md`/paper não repitam esta claim sem qualificá-la como
pré-correção de dado.**

---

### 6.7 ABL-INJ — Mecanismo de injeção da demografia

**Hipótese testada:** a injeção demográfica atual ocorre **depois** do
`LayerNorm` de `text_proj` e antes do `fusion_norm` global (seção 3.3),
inflando a magnitude da metade de texto e fazendo texto e visão entrarem na
fusão em escalas diferentes. A hipótese é que essa distorção de escala explica
por que a demografia não demonstra valor incremental (seção 6.5).

Três modos, cruzados com os dois `demo_feature_set`, todos com visão e
demografia ativas: `current` (baseline), `renorm` (re-aplica `LayerNorm` depois
da injeção, antes da concatenação com visão) e `head_inject` (não injeta no CLS
de texto; concatena o sinal demográfico projetado ao vetor fundido **depois** do
`fusion_norm`, elevando as três cabeças de 1024 para 1536 in_features).

**Validação da âncora.** O modo `current` não cria nenhum módulo novo e preserva
a ordem de consumo do RNG, portanto deve nascer bit-a-bit idêntico à baseline.
Verificado: `fingerprint_init` de `inj_current_base` é igual ao de `baseline` nas
três seeds (`6cd0ef763f16d975`, `e8f051f9614d9927`, `9502837aaa79b209`). Os
deltas abaixo medem, portanto, apenas o mecanismo de injeção.

Deltas por seed, `renorm` − `current` (feature set base):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | +0.0039 | -0.0073 | +0.0084 | +0.0017 | não |
| B soft | +0.0008 | +0.0034 | +0.0076 | **+0.0039** | **sim** |
| C soft | +0.0050 | -0.0128 | +0.0057 | -0.0007 | não |
| A hard | -0.0016 | -0.0092 | +0.0084 | -0.0008 | não |
| B hard | -0.0244 | +0.0042 | +0.0101 | -0.0034 | não |
| C hard (tuned) | -0.0055 | -0.0193 | +0.0182 | -0.0022 | não |
| C hard (thr=0.5) | -0.0025 | -0.0245 | +0.0139 | -0.0044 | não |

Deltas por seed, `head_inject` − `current` (feature set base):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | +0.0061 | +0.0020 | +0.0259 | **+0.0113** | **sim** |
| B soft | +0.0134 | +0.0069 | +0.0044 | **+0.0082** | **sim** |
| C soft | +0.0094 | +0.0029 | +0.0378 | **+0.0167** | **sim** |
| A hard | +0.0017 | +0.0216 | +0.0201 | **+0.0145** | **sim** |
| B hard | -0.0025 | -0.0043 | +0.0032 | -0.0012 | não |
| C hard (tuned) | +0.0163 | +0.0144 | +0.0522 | **+0.0276** | **sim** |
| C hard (thr=0.5) | +0.0216 | +0.0076 | +0.0500 | **+0.0264** | **sim** |

Deltas por seed, `head_inject` − `current` (feature set extended):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | +0.0032 | +0.0134 | +0.0113 | **+0.0093** | **sim** |
| B soft | +0.0020 | +0.0043 | +0.0126 | **+0.0063** | **sim** |
| C soft | +0.0164 | +0.0247 | +0.0116 | **+0.0176** | **sim** |
| A hard | +0.0076 | -0.0015 | +0.0274 | +0.0112 | não |
| B hard | +0.0353 | +0.0299 | +0.0336 | **+0.0329** | **sim** |
| C hard (tuned) | +0.0273 | +0.0474 | +0.0262 | **+0.0337** | **sim** |
| C hard (thr=0.5) | +0.0241 | +0.0255 | +0.0237 | **+0.0244** | **sim** |

**Interpretação.** O `renorm` corrige exatamente a distorção de escala descrita
na hipótese e produz **1 de 7** sinais consistentes no base (e 1 de 7 no
extended, não mostrado), com médias dentro do ruído de seed. Se a inflação da
metade de texto fosse o mecanismo que suprime a demografia, o `renorm` teria
funcionado. **A hipótese de escala está refutada.**

O `head_inject` produz **6 de 7** sinais consistentes nos dois feature sets, com
ganhos de +0,008 a +0,028, incluindo confirmação sob threshold fixo em Task C. O
mecanismo eficaz não é corrigir a escala, é dar às cabeças acesso direto ao
sinal. A única exceção no base é `task_b_icmnorm_hard` (−0,0012), que no extended
inverte para +0,0329 — inconsistente entre feature sets, não generalizar.

**Ressalva essencial:** o `head_inject` também acrescenta 512 dimensões
aprendidas a cada cabeça. O ganho pode ser demografia, capacidade, ou ambos.
Esta ablação não separa as duas contribuições; a seção 6.8 separa.

### 6.7.1 Diagnóstico de colapso da cabeça de Task C

A seção 6.2 registrou a suspeita de colapso da saída da cabeça de Task C sob
`demo_vec` extended, não confirmada à época por falta de acesso às
probabilidades por categoria. Esta suite instrumentou o diagnóstico
explicitamente: para cada config × seed, as probabilidades da Task C em todo o
conjunto de validação são coletadas na avaliação final e persistidas em JSON
(`ablation/diagnostics/`, fora de qualquer diretório temporário — foi o descarte
das predições que impediu a confirmação anterior).

Critério de colapso, fixado antes da medição: variância inter-item média
< 1×10⁻⁴ **e** |média prevista − taxa base do treino| média < 0,02, isto é, a
cabeça prevê a marginal do treino e ignora o item.

**Resultado: 0 de 3 seeds em todas as 6 configs**, incluindo `current` +
`extended`, onde o colapso teria sido observado. A variância inter-item média
ficou em ~2,5×10⁻² (cerca de 250× acima do limiar) e |média − taxa base| em
~0,34 (~17× acima). O cosseno médio entre distribuições de itens distintos ficou
em ~0,68, longe do ~1,0 esperado sob colapso. A cabeça discrimina claramente
entre itens.

**Leitura correta:** duas explicações permanecem compatíveis com a evidência e
esta suite não as distingue — (i) o colapso era artefato do framework anterior,
cujas predições foram descartadas e não podem ser reexaminadas; ou (ii) o
colapso é real mas não se manifesta sob o regime de treino corrigido. Reportar
como **observação não replicada**, nunca como refutação da observação original.
A ressalva de qualidade da seção 6.2 deve ser lida à luz disto.

### 6.8 ABL-INJ-CTRL — Controle de capacidade do `head_inject`

**Motivação.** O `head_inject` altera duas coisas simultaneamente: o acesso das
cabeças ao sinal demográfico e a dimensionalidade de entrada das cabeças (1024 →
1536, isto é, 512 parâmetros aprendidos a mais por cabeça). O ganho da seção 6.7
é, portanto, ambíguo. Sem separar as contribuições, a afirmação "a demografia
passou a ter efeito mensurável" não é sustentável: o mesmo resultado seria
esperado se as dimensões extras estivessem apenas ampliando a capacidade do
classificador, com a demografia ignorada.

**Desenho.** A configuração de controle é idêntica a `inj_headinject_base` em
arquitetura, capacidade e hiperparâmetros — `fingerprint_init` verificado como
idêntico nas três seeds — diferindo apenas em que o `demo_vec` é permutado entre
itens. A permutação é um *derangement* (sem pontos fixos), gerada por semente
dedicada e fixa, independente da semente de treino, de modo que as três seeds
veem a mesma permutação e a comparação pareada isola o embaralhamento. É
aplicada depois da construção da coluna demográfica e **antes** da divisão
treino/validação, evitando assimetria entre os conjuntos.

**Por que permutar e não zerar.** Com `demo_vec` fixado em zero, `demo_proj(0)`
é constante, o portão sigmoide é constante e o bloco concatenado vira um viés
idêntico para todos os itens: as 512 dimensões extras deixam de ser utilizáveis.
Zerar removeria simultaneamente sinal e capacidade, remedindo as duas juntas —
exatamente o confundimento a eliminar. A permutação preserva a distribuição
marginal dos vetores (o multiconjunto é o mesmo) e a capacidade utilizável,
destruindo apenas a correspondência item↔demografia.

**Caracterização do `demo_vec`.** Medida ao desenhar o controle: o vetor assume
apenas **151 valores distintos entre 3.984 itens** (mediana de 27 itens por
valor), funcionando como identificador categórico do grupo de anotadores. Esse
grupo explica **15,1% da variância de P(YES)** na Task A — há sinal real a ser
destruído. A permutação pode, por acaso, atribuir a um item um vetor
numericamente igual ao seu quando ambos partilham o grupo: probabilidade teórica
0,67%, taxa observada 0,60% (24 de 3.984). Medida e registrada, não corrigida —
forçar valores distintos enviesaria a permutação.

Deltas por seed, embaralhado − real (delta negativo = embaralhar piorou, isto é,
a demografia estava contribuindo):

| métrica | seed 42 | seed 7 | seed 2025 | média | consistente |
|---|---|---|---|---|---|
| A soft | -0.0017 | -0.0016 | +0.0012 | -0.0007 | não |
| B soft | -0.0016 | +0.0004 | +0.0011 | -0.0001 | não |
| C soft | -0.0032 | -0.0053 | -0.0018 | **-0.0035** | **sim** |
| A hard | +0.0158 | -0.0150 | -0.0042 | -0.0011 | não |
| B hard | -0.0099 | -0.0067 | -0.0077 | **-0.0081** | **sim** |
| C hard (tuned) | -0.0091 | -0.0172 | +0.0020 | -0.0081 | não |
| C hard (thr=0.5) | -0.0062 | -0.0049 | -0.0076 | **-0.0063** | **sim** |

**Decomposição do ganho.** Sendo o ganho total do `head_inject` sobre `current`
(seção 6.7) e a parcela que desaparece ao embaralhar:

| métrica | ganho total | parcela demográfica | % do ganho |
|---|---|---|---|
| A soft | +0.0113 | +0.0007 | 6,4% |
| B soft | +0.0082 | +0.0001 | 0,7% |
| C soft | +0.0167 | +0.0035 | 20,7% |
| A hard | +0.0145 | +0.0011 | 7,9% |
| B hard | -0.0012 | +0.0081 | n/a (ganho não positivo) |
| C hard (tuned) | +0.0276 | +0.0081 | 29,4% |
| C hard (thr=0.5) | +0.0264 | +0.0063 | 23,8% |

Mediana entre as métricas com ganho positivo: **14,3%** (intervalo 0,7%–29,4%).
O efeito concentra-se na Task C (20,7%–29,4%) e é quase nulo nas Tasks A e B
(0,7%–7,9%).

**Veredito, pelo critério pré-registrado.** O critério de decisão foi fixado e
escrito antes da execução do treino (`ablation/results/ABL-INJ-CTRL.md`):
degradação com sinal consistente 3/3 na maioria das métricas implicaria adotar o
`head_inject` como evidência de contribuição demográfica. Observou-se degradação
consistente em **3 de 7** métricas (C soft, B hard, C hard sob threshold fixo) —
abaixo da maioria exigida. **Resultado parcial e não conclusivo; não adotar com
base neste braço.**

**Observação descritiva**, registrada por transparência e que não altera o
veredito: a média do delta é negativa em **7 de 7** métricas — o embaralhamento
nunca ajudou em média. O critério pré-registrado é a consistência de sinal entre
seeds, não o sinal da média. A direção uniformemente negativa é compatível com
um efeito demográfico real porém pequeno demais para superar o ruído de seed na
maioria das métricas, e deve ser descrita assim, sem ser convertida em evidência
de efeito.

---

## 7. Síntese consolidada e implicações

### 7.1 Achados robustos (sinal 3/3, confirmados sob threshold fixo quando aplicável a Task C)

1. **Adicionar visão melhora TODAS as métricas, em TODAS as seeds** (ABL-5,
   contraste isolado `text_only` → `vision_only`, seção 6.5) — 7 de 7 métricas
   com sinal consistente 3/3, o resultado mais forte e mais limpo da suite
   inteira. Ganho concentrado em Task C (+27,7% soft, +28,7% hard relativo),
   menor em Task B (+2,7% soft), confirmado sob threshold fixo. Simetricamente,
   remover visão a partir de `full` (`demo_only`/`text_only` vs `full`) produz
   as maiores quedas medidas (até -0.08 em task_c hard). Ressalva de
   magnitude: o braço sem visão sofre também o confound de fusão da seção 3.4
   — direção robusta, valores absolutos são teto.
2. **Congelar o CLIP inteiramente prejudica task_a e task_c** (ABL-4, config
   B vs A) — confirmado sob threshold ajustado e fixo. Destravar mais camadas
   (8 vs 4) ajuda task_b soft e task_a hard sem prejudicar nada de forma
   consistente.
3. **Demo_vec extended (gênero+idade) tem trade-off real**: ajuda task_b
   soft, prejudica task_c hard (confirmado sob threshold fixo) — com ressalva
   de possível colapso de saída na cabeça de Task C (seção 6.2, não
   confirmada por falta de acesso às probabilidades por categoria).
4. **Remover só a demografia (mantendo visão) não tem efeito mensurável em
   nenhuma métrica** (ABL-5, `vision_only` vs `full`) — 0 de 7 comparações
   com sinal consistente.
5. **Concatenar a demografia ao vetor fundido, junto às cabeças, melhora 6 de
   7 métricas nos dois feature sets** (ABL-INJ, `head_inject` vs `current`,
   seção 6.7) — confirmado sob threshold fixo em Task C. Porém o controle
   pareado (ABL-INJ-CTRL, seção 6.8) atribui apenas ~14% desse ganho (mediana)
   ao vínculo item↔demografia; o restante é capacidade das 512 dimensões
   adicionais por cabeça. O efeito demográfico concentra-se na Task C
   (20,7%–29,4%) e é quase nulo em A e B (0,7%–7,9%).

### 7.2 Negative results (sem sinal consistente robusto, ou sinal fraco/contraditório)

- Warmup schedule (ABL-1) — 2 de 14 comparações consistentes, magnitude pequena.
- Peso da loss de Task A, 1×/2×/3× (ABL-3) — 1 de 14, magnitude pequena, sem
  relação monotônica entre 1.0→2.0→3.0.
- Normalização sum vs L2 (ABL-6) — 2 de 7, direções conflitantes, um deles
  não sobrevive ao controle de threshold fixo; não reproduz a claim histórica
  do projeto (reconciliada na seção 6.6).
- Re-normalizar depois da injeção demográfica (ABL-INJ, `renorm`, seção 6.7) —
  1 de 7 nos dois feature sets, médias dentro do ruído de seed. **Refuta a
  hipótese de que a distorção de escala pós-`LayerNorm` era a causa de a
  demografia não contribuir.**
- Colapso da cabeça de Task C sob `demo_vec` extended (seção 6.7.1) — 0 de 3
  seeds em todas as 6 configs, com o diagnóstico instrumentado e persistido.
  Observação não replicada; ver a leitura correta na seção 6.7.1.

### 7.3 Implicação prática

A arquitetura atual depende de um encoder de visão capaz de fine-tuning
parcial — visão ausente ou totalmente congelada derruba o modelo de forma
substancial e consistente, especialmente em Task C. O vetor demográfico,
como injetado hoje (contagem normalizada → projeção linear → gate sigmoide →
soma ao CLS de texto pós-LayerNorm), **não demonstra valor incremental
mensurável sobre texto+visão** na configuração base, exceto no caso específico
de features expandidas (ABL-2), que trocam ganho em Task B por perda em Task
C. Isso não implica necessariamente que a informação demográfica seja
inútil — pode indicar que **o mecanismo de injeção atual não deixa esse sinal
chegar de forma eficaz às cabeças de saída**, hipótese explorada na seção 8.

---

## 8. Propostas de melhoria arquitetural

Ordenadas por payoff esperado por hora de GPU. Todas compatíveis com o
framework de ablação existente, parametrizáveis via extensão de
`AblationConfig` com custo de implementação baixo a médio.

### 8.1 Corrigir a fusão para modalidades ausentes (pré-requisito para as demais propostas)

**Problema (seção 3.4/6.5):** quando `use_vision=False`,
`vision_cls = zeros_like(text_cls)` entra no `LayerNorm` global de fusão
junto com o texto real — distorce a normalização estatística da metade viva e
desperdiça metade da largura de entrada das 3 cabeças em constantes. Isso
significa que a ablação de modalidade (ABL-5) mede "remoção de informação +
degradação do caminho de fusão" simultaneamente, não uma remoção limpa.

**Proposta:** montar `fused` apenas a partir dos ramos presentes (dimensão de
entrada das cabeças variável por config, com `nn.Linear` inicializado
conforme a dimensão ativa), ou aplicar `LayerNorm` por-ramo antes da
concatenação, evitando a normalização conjunta sobre dimensões
constantemente zeradas.

**Custo:** baixo–médio (mudança estrutural em `forward` + inicialização
dinâmica das cabeças, ou padding explícito e documentado se a dimensão
precisar ficar fixa por razões de compatibilidade de checkpoint). **Risco:**
médio — muda a dimensão de entrada efetiva das cabeças; qualquer comparação
histórica com resultados anteriores precisa ser refeita. **Pré-requisito
formal** para qualquer ablação futura de modalidade ser interpretável sem a
ressalva da seção 6.5.

### 8.2 Modality dropout no ramo de visão

**Motivação direta do achado 4 (seção 7.1):** o sintoma observado — uma
modalidade (visão) parece "monopolizar" a representação fundida, sem que as
demais (texto, demografia) mostrem ganho incremental mensurável quando
testadas isoladamente contra ela — é o cenário clássico de aplicação de
modality dropout em arquiteturas multimodais.

**Proposta:** durante o treino (após a correção do item 8.1), zerar
`vision_cls` com probabilidade `p ∈ {0.15, 0.30}` por amostra/batch, forçando
o caminho de texto a permanecer preditivo de forma independente da
disponibilidade de visão, em vez de aprender a depender dela quase
exclusivamente.

**Custo:** muito baixo — poucas linhas em `forward`, um novo campo
`vision_dropout_p: float = 0.0` em `AblationConfig`. **Risco:** baixo.
**Payoff esperado:** alto — ataca diretamente o objetivo declarado de tornar
o texto mais influente na predição final, sem exigir uma mudança estrutural
maior.

### 8.3 Supervisão auxiliar só de texto (deep supervision)

**Proposta:** adicionar uma cabeça extra `Linear(512, 1)` sobre `t'` (o CLS de
texto já com demografia injetada, antes da concatenação com visão), com sua
própria loss de Task A, ponderada com peso pequeno (ex. 0.2–0.5) somada à loss
principal:
```
L_text_aux = w_aux · BCEWithLogits(head_text_aux(t'), label_a)
L = L_a + L_b + L_c + L_text_aux
```
Isso força o encoder de texto a manter um sinal preditivo por conta própria,
mesmo quando a fusão com visão poderia "abafar" seu peso relativo nas cabeças
principais.

**Custo:** baixo — uma `Linear` nova, um termo de loss extra, sem alterar
dimensões das cabeças existentes. **Risco:** baixo. Complementar aos itens
8.1/8.2, não competitivo — pode ser testado junto ou isoladamente.

### 8.4 Repensar a injeção de demografia

**Problema (seção 3.3):** a injeção `t' = t + gate ⊙ p` acontece **depois**
do `LayerNorm` interno de `text_proj`, então pode inflar a magnitude da
metade de texto além da escala em que `t` originalmente estava, antes de
`fused` ser normalizado globalmente — texto e visão podem entrar na fusão em
escalas efetivamente diferentes. O ABL-5 mostra que a injeção atual, na
prática, não produz nenhum ganho mensurável (`vision_only` ≈ `full` em todas
as 7 métricas, achado 4 da seção 7.1) — não há o que perder testando uma
alternativa de mecanismo.

**Proposta A:** re-normalizar (`LayerNorm`) depois da injeção de demografia,
antes da concatenação com visão: `t' = LayerNorm(t + gate ⊙ p)`.
**Proposta B (alternativa mais estrutural):** em vez de injetar demografia só
no CLS de texto, re-injetar (ou concatenar) o `demo_vec` projetado também nas
cabeças de saída, dando a cada task acesso direto ao sinal demográfico sem
depender de ele sobreviver ao caminho de fusão inteiro.

**Custo:** muito baixo (proposta A, ~1 linha) a baixo (proposta B, nova
concatenação nas cabeças). **Risco:** baixo em ambas.

### 8.5 Não investir em capacidade de texto ou reponderação de loss

O XLM-R-large já está totalmente destravado (todas as 24 camadas treináveis,
sem `requires_grad=False` em lugar nenhum do código) — não há capacidade
adicional óbvia a "liberar" nesse encoder. A ABL-3 mostrou que reponderar a
loss de Task A (1×/2×/3×) não move nenhuma métrica de forma consistente,
apesar de alterar substancialmente a magnitude absoluta da loss total.
**A alavanca disponível não é escalar (peso de loss) — é estrutural** (itens
8.1–8.4 acima). Recomenda-se não alocar runs futuras testando mais variações
de peso de loss ou de tamanho da projeção de texto (`shared_dim=512`) antes
de esgotar 8.1–8.4.

### 8.6 Ordem de execução recomendada

1. **8.1** (pré-requisito técnico — sem ele, qualquer nova ablação de
   modalidade herda o mesmo confound do ABL-5 atual).
2. **8.2 e 8.4-A** em paralelo — ambos muito baratos, atacam o mesmo sintoma
   (texto subordinado à visão na fusão) por ângulos diferentes (dropout de
   modalidade vs. escala de injeção).
3. **8.3** — complementar, pode ser adicionado a qualquer uma das
   configurações acima sem conflito estrutural.
4. **Nova rodada do ABL-5, com a fusão corrigida (8.1)**, para confirmar se a
   ausência de sinal da demografia isolada (achado 4, seção 7.1) é real
   mesmo com uma fusão limpa, ou se era em parte também um artefato da fusão
   degenerada — esta é a validação decisiva que separa "demografia não ajuda"
   de "o mecanismo atual de usar demografia está quebrado". **Custo:** 4
   variantes × 3 seeds × 30 épocas ≈ 12 treinos completos — à taxa observada
   nesta suite (~1,4h/treino após as otimizações de `eval_batch_size`/
   `dataloader_num_workers`), ≈17h de GPU só para este passo de validação.
   Vale ser explícito sobre esse custo porque a suite inteira já motivou uma
   rodada de otimização de tempo de execução — este item deveria ser
   orçado com a mesma atenção antes de ser disparado.

---

## Apêndice A — Tabela bruta completa (75 linhas)

Valor por seed, sem agregação, para as 6 métricas-chave + loss total final +
tempo de avaliação. Fonte primária de todas as tabelas de delta pareado da
seção 6 — permite a qualquer leitor recalcular ou auditar qualquer delta
citado neste relatório diretamente a partir dos números abaixo (`config_name`,
`seed` identificam univocamente cada linha). "C hard(t)" = threshold ajustado
por busca gulosa (seção 3.7); "C hard(0.5)" = threshold fixo. Linhas
repetidas entre ablações (mesmo `fingerprint_init`/`fingerprint_final`, mesmos
valores) são reaproveitamentos da FASE 0, não reexecuções — ver seção 5.

| config_name                      |   seed |   A soft |   B soft |   C soft |   A hard |   B hard |   C hard(t) |   C hard(0.5) |    loss |   runtime(s) |
|:---------------------------------|-------:|---------:|---------:|---------:|---------:|---------:|------------:|--------------:|--------:|-------------:|
| baseline                         |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.7944  |         70.4 |
| baseline                         |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83017 |         68.5 |
| baseline                         |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.7886  |         68.6 |
| abl1_config_A_warmup_0.1         |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.7944  |         70.4 |
| abl1_config_A_warmup_0.1         |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83017 |         68.5 |
| abl1_config_A_warmup_0.1         |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.7886  |         68.6 |
| abl1_config_B_warmup_0.0         |     42 |   0.4953 |   0.3913 |   0.285  |   0.5079 |   0.4386 |      0.4153 |        0.3746 | 2.8276  |         59.3 |
| abl1_config_B_warmup_0.0         |      7 |   0.4831 |   0.4018 |   0.2694 |   0.5119 |   0.4268 |      0.3897 |        0.3679 | 2.77051 |         61   |
| abl1_config_B_warmup_0.0         |   2025 |   0.4893 |   0.4018 |   0.2774 |   0.5219 |   0.444  |      0.4018 |        0.377  | 2.77641 |         61.6 |
| abl1_config_C_warmup_0.2         |     42 |   0.4998 |   0.3962 |   0.2891 |   0.5237 |   0.4542 |      0.4038 |        0.3849 | 2.80788 |         62.4 |
| abl1_config_C_warmup_0.2         |      7 |   0.4921 |   0.3987 |   0.2956 |   0.5186 |   0.44   |      0.4104 |        0.3882 | 2.82073 |         61   |
| abl1_config_C_warmup_0.2         |   2025 |   0.4945 |   0.4049 |   0.2794 |   0.541  |   0.4417 |      0.39   |        0.3743 | 2.74783 |         61.5 |
| abl2_config_A_demo_base          |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.7944  |         70.4 |
| abl2_config_A_demo_base          |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83017 |         68.5 |
| abl2_config_A_demo_base          |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.7886  |         68.6 |
| abl2_config_B_demo_extended      |     42 |   0.4904 |   0.4016 |   0.2724 |   0.5136 |   0.4164 |      0.3741 |        0.35   | 2.76086 |         60   |
| abl2_config_B_demo_extended      |      7 |   0.4941 |   0.4036 |   0.2721 |   0.5268 |   0.4206 |      0.3696 |        0.3622 | 2.78421 |         61.6 |
| abl2_config_B_demo_extended      |   2025 |   0.4978 |   0.41   |   0.2727 |   0.5111 |   0.4469 |      0.3768 |        0.3602 | 2.73348 |         60.9 |
| abl3_config_A_taskaw_2.0         |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.7944  |         70.4 |
| abl3_config_A_taskaw_2.0         |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83017 |         68.5 |
| abl3_config_A_taskaw_2.0         |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.7886  |         68.6 |
| abl3_config_B_taskaw_1.0         |     42 |   0.5067 |   0.3971 |   0.2871 |   0.5179 |   0.4589 |      0.4203 |        0.3887 | 2.1723  |         61.2 |
| abl3_config_B_taskaw_1.0         |      7 |   0.492  |   0.3977 |   0.2877 |   0.5187 |   0.4362 |      0.4109 |        0.3809 | 2.18508 |         60.4 |
| abl3_config_B_taskaw_1.0         |   2025 |   0.4899 |   0.3982 |   0.2808 |   0.5302 |   0.4525 |      0.394  |        0.378  | 2.14115 |         60.6 |
| abl3_config_C_taskaw_3.0         |     42 |   0.4993 |   0.3942 |   0.2962 |   0.5253 |   0.4455 |      0.4446 |        0.3971 | 3.44739 |         61.6 |
| abl3_config_C_taskaw_3.0         |      7 |   0.4884 |   0.3973 |   0.2872 |   0.5095 |   0.4336 |      0.4042 |        0.3842 | 3.4879  |         61.5 |
| abl3_config_C_taskaw_3.0         |   2025 |   0.4803 |   0.3971 |   0.2677 |   0.5285 |   0.4217 |      0.3811 |        0.367  | 3.43705 |         60.6 |
| abl4_config_A_unfrozen_4         |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.7944  |         70.4 |
| abl4_config_A_unfrozen_4         |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83017 |         68.5 |
| abl4_config_A_unfrozen_4         |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.7886  |         68.6 |
| abl4_config_B_unfrozen_0         |     42 |   0.4843 |   0.4031 |   0.257  |   0.5395 |   0.4202 |      0.3576 |        0.3322 | 2.76489 |         59.9 |
| abl4_config_B_unfrozen_0         |      7 |   0.4887 |   0.403  |   0.2434 |   0.497  |   0.4295 |      0.3522 |        0.3455 | 2.77534 |         60.2 |
| abl4_config_B_unfrozen_0         |   2025 |   0.4802 |   0.3957 |   0.2529 |   0.5394 |   0.3987 |      0.342  |        0.3264 | 2.79397 |         59.8 |
| abl4_config_C_unfrozen_8         |     42 |   0.5054 |   0.4133 |   0.2736 |   0.5442 |   0.4541 |      0.4304 |        0.4005 | 2.70826 |         62   |
| abl4_config_C_unfrozen_8         |      7 |   0.4897 |   0.4171 |   0.2822 |   0.5367 |   0.4249 |      0.3974 |        0.3811 | 2.72766 |         60.1 |
| abl4_config_C_unfrozen_8         |   2025 |   0.503  |   0.4053 |   0.2764 |   0.5624 |   0.4482 |      0.4106 |        0.3937 | 2.67046 |         61.9 |
| abl5_config_A_vision_demo        |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.7944  |         70.4 |
| abl5_config_A_vision_demo        |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83017 |         68.5 |
| abl5_config_A_vision_demo        |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.7886  |         68.6 |
| abl5_config_B_vision_only        |     42 |   0.4983 |   0.4039 |   0.2819 |   0.5227 |   0.4202 |      0.403  |        0.3772 | 2.76099 |         60.7 |
| abl5_config_B_vision_only        |      7 |   0.5045 |   0.3981 |   0.2964 |   0.5526 |   0.4296 |      0.4353 |        0.4044 | 2.82008 |         60.8 |
| abl5_config_B_vision_only        |   2025 |   0.4949 |   0.3866 |   0.291  |   0.5193 |   0.413  |      0.4099 |        0.3883 | 2.79625 |         60.1 |
| abl5_config_C_demo_only          |     42 |   0.4683 |   0.3932 |   0.2408 |   0.5284 |   0.4116 |      0.3115 |        0.2884 | 2.84405 |         60.8 |
| abl5_config_C_demo_only          |      7 |   0.478  |   0.399  |   0.2541 |   0.5419 |   0.4015 |      0.3667 |        0.3553 | 2.80317 |         58.9 |
| abl5_config_C_demo_only          |   2025 |   0.4588 |   0.3757 |   0.2132 |   0.4974 |   0.4002 |      0.2986 |        0.2809 | 2.83813 |         61.2 |
| abl5_config_D_text_only          |     42 |   0.4511 |   0.3791 |   0.2227 |   0.4865 |   0.4143 |      0.3177 |        0.2957 | 2.80527 |         60   |
| abl5_config_D_text_only          |      7 |   0.4726 |   0.3951 |   0.2357 |   0.5267 |   0.3851 |      0.3176 |        0.2997 | 2.77417 |         58.6 |
| abl5_config_D_text_only          |   2025 |   0.4589 |   0.3829 |   0.2226 |   0.5086 |   0.3931 |      0.3345 |        0.3196 | 2.83932 |         61   |
| abl6_config_A_norm_sum           |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.7944  |         70.4 |
| abl6_config_A_norm_sum           |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83017 |         68.5 |
| abl6_config_A_norm_sum           |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.7886  |         68.6 |
| abl6_config_B_norm_l2            |     42 |   0.5009 |   0.3965 |   0.2921 |   0.5112 |   0.4441 |      0.4136 |        0.4023 | 2.79572 |         60.8 |
| abl6_config_B_norm_l2            |      7 |   0.4879 |   0.4032 |   0.2851 |   0.517  |   0.4323 |      0.3866 |        0.3764 | 2.80111 |         60.2 |
| abl6_config_B_norm_l2            |   2025 |   0.492  |   0.403  |   0.2809 |   0.5343 |   0.448  |      0.3807 |        0.371  | 2.7695  |         60.6 |
| inj_current_base                 |     42 |   0.5007 |   0.3962 |   0.2904 |   0.5186 |   0.4636 |      0.4252 |        0.3935 | 2.79443 |         61.8 |
| inj_current_base                 |      7 |   0.4923 |   0.3982 |   0.2919 |   0.5153 |   0.4282 |      0.4096 |        0.3924 | 2.83016 |         62.3 |
| inj_current_base                 |   2025 |   0.4846 |   0.3964 |   0.2715 |   0.5443 |   0.439  |      0.3824 |        0.3687 | 2.78857 |         64.9 |
| inj_renorm_base                  |     42 |   0.5046 |   0.3969 |   0.2954 |   0.517  |   0.4391 |      0.4197 |        0.3911 | 2.80688 |         60.2 |
| inj_renorm_base                  |      7 |   0.485  |   0.4016 |   0.2791 |   0.5061 |   0.4325 |      0.3902 |        0.3679 | 2.80484 |         61   |
| inj_renorm_base                  |   2025 |   0.493  |   0.404  |   0.2772 |   0.5527 |   0.4491 |      0.4006 |        0.3825 | 2.75227 |         60.9 |
| inj_headinject_base              |     42 |   0.5067 |   0.4095 |   0.2998 |   0.5203 |   0.4611 |      0.4415 |        0.4151 | 2.75431 |         61.4 |
| inj_headinject_base              |      7 |   0.4943 |   0.4051 |   0.2948 |   0.5369 |   0.4239 |      0.4239 |        0.4    | 2.74459 |         60.6 |
| inj_headinject_base              |   2025 |   0.5105 |   0.4009 |   0.3093 |   0.5644 |   0.4421 |      0.4346 |        0.4186 | 2.80303 |         61.3 |
| inj_current_extended             |     42 |   0.4904 |   0.4016 |   0.2724 |   0.5136 |   0.4164 |      0.3741 |        0.35   | 2.76086 |         61.2 |
| inj_current_extended             |      7 |   0.4941 |   0.4036 |   0.2721 |   0.5268 |   0.4206 |      0.3696 |        0.3622 | 2.78421 |         62.5 |
| inj_current_extended             |   2025 |   0.4978 |   0.41   |   0.2727 |   0.5111 |   0.4469 |      0.3768 |        0.3602 | 2.73348 |         60.3 |
| inj_renorm_extended              |     42 |   0.4856 |   0.4012 |   0.269  |   0.5093 |   0.4356 |      0.3752 |        0.353  | 2.76727 |         61.5 |
| inj_renorm_extended              |      7 |   0.5062 |   0.4127 |   0.2866 |   0.536  |   0.4349 |      0.4021 |        0.387  | 2.74461 |         61.4 |
| inj_renorm_extended              |   2025 |   0.5018 |   0.4079 |   0.2764 |   0.5144 |   0.4411 |      0.3965 |        0.35   | 2.72661 |         61.4 |
| inj_headinject_extended          |     42 |   0.4935 |   0.4035 |   0.2887 |   0.5212 |   0.4517 |      0.4015 |        0.3741 | 2.8388  |         60.8 |
| inj_headinject_extended          |      7 |   0.5074 |   0.4079 |   0.2968 |   0.5253 |   0.4505 |      0.417  |        0.3876 | 2.78171 |         60.7 |
| inj_headinject_extended          |   2025 |   0.5091 |   0.4226 |   0.2843 |   0.5385 |   0.4805 |      0.403  |        0.3839 | 2.71577 |         60.9 |
| injctrl_headinject_shuffled_base |     42 |   0.505  |   0.4079 |   0.2965 |   0.5361 |   0.4513 |      0.4324 |        0.4089 | 2.7636  |         59.5 |
| injctrl_headinject_shuffled_base |      7 |   0.4927 |   0.4055 |   0.2895 |   0.5219 |   0.4172 |      0.4067 |        0.3951 | 2.75424 |         60.7 |
| injctrl_headinject_shuffled_base |   2025 |   0.5117 |   0.4019 |   0.3074 |   0.5602 |   0.4344 |      0.4365 |        0.411  | 2.81412 |         59.5 |

## 9. Limitações deste relatório

1. **ABL-5 não inclui um braço "sem texto".** A suite testa remoção de visão
   e demografia contra um texto sempre presente e sempre totalmente
   treinável — não é possível, a partir destes dados, quantificar a
   contribuição do texto isoladamente, só a das outras duas modalidades
   condicionadas à presença dele.
2. **Confound de fusão em `use_vision=False`** (seções 3.4, 6.5, 8.1) — a
   magnitude dos efeitos de remover visão está provavelmente inflada por um
   caminho de normalização degenerado, não isolado da remoção de informação
   em si.
3. **Dois vieses de protocolo (seção 2.4) presentes em toda linha de
   resultado** — seleção de época e threshold de Task C ambos ajustados no
   mesmo conjunto de validação reportado. Mitigado nas comparações relativas
   (viés idêntico dos dois lados) e checado explicitamente contra
   `task_c_icmnorm_hard_default` (threshold fixo) para cada achado citado
   como robusto.
4. **Descontinuidade de `eval_batch_size` entre FASE 0 e FASE 1** (seção
   2.4c) — argumentada como desprezível (evidência: fingerprints e métricas
   idênticas num teste controlado), mas não eliminada estruturalmente (as
   linhas de referência de toda ablação vêm da FASE 0).
5. **Hipótese de colapso da cabeça de Task C sob demo_vec extended (ABL-2)
   não confirmada** — falta acesso às probabilidades por categoria, apagadas
   junto com o diretório temporário do PyEvALL ao fim de cada avaliação.
6. **n=3 seeds.** Suficiente para o critério de consistência de sinal usado
   neste relatório (uma seed discordante já classifica o efeito como "não
   consistente"), mas não permite teste de significância estatística formal
   (ex. teste t pareado) com poder adequado — a análise aqui é de consistência
   de direção, não de significância inferencial.
7. **Definição matemática do ICM/ICM-Soft não verificada contra a
   implementação** (seção 3.6) — reportada apenas operacionalmente, a partir
   do que o código de integração permite afirmar com certeza, para evitar
   reproduzir uma fórmula não confirmada contra a fonte primária.
