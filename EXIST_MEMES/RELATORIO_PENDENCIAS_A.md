# Relatório — verificação das pendências A (Capítulo 3, sistema de memes)

**Data:** 2026-10-05 · **Itens:** A1–A15 e A17 (exceto A16 e A18, que ficaram fora do pedido)
**Regra seguida:** o texto da tese acompanha o **código**, e não o paper. Cada divergência listada aqui alimenta a decisão C1 (errata).

---

## 0. Fontes e mapeamento dos runs (resolvido nesta verificação)

| Artefato | O que é | Evidência |
|---|---|---|
| `EXIST_MEMES/Thesis_main.ipynb` | Código **e log de treino do Jow_3** | Log da célula 49: treino de 2026-05-14 09:39 até cerca de 10:51. O código é idêntico ao do commit `ff37b33` e ao `.ipynb_checkpoints/Thesis_main-checkpoint.ipynb` (salvo às 09:34, antes do treino); só difere um `os.chdir`. |
| `EXIST_MEMES/run3.zip` = `EXIST_MEMES/predictions/` | Predições do **Jow_3** | Os md5 dos 6 JSON são idênticos aos de `task2_*_Jow_3.json` do pacote final |
| `EXIST_MEMES/exist2026_Jow.zip` (VM) | Predições do **Jow_2** | Os md5 dos 6 JSON são idênticos aos de `task2_*_Jow_2.json` |
| `~/Downloads/exist2026_Jow (1).zip` (Mac) | **Pacote final submetido** (Task 2 e 3, os 3 runs) | Horários: Jow_1 em 05-14 00:04 · Jow_2 em 05-14 09:12 · Jow_3 em 05-14 11:20 |
| `.ipynb_checkpoints/Untitled-checkpoint.ipynb` (2026-05-13 23:02) | Versão antiga do `inference.py`: **2 camadas, sem gate, injeção aditiva** | É compatível com o Jow_1 (predições às 00:04) |
| `EXIST_MEMES/inference.py` | Versão para o Jow_3 (tem o gate) | Idêntico a `.ipynb_checkpoints/inference-checkpoint.py` (05-14 11:16) |

Com isso fica respondida a dúvida nº 3 de `thesis-checkpoints/cleanup_2026-09-29/reports/C_memes.md` ("qual zip é qual run").

**Níveis de evidência por run.** Eles valem para A3, A4 e A5:
- **Jow_3: verificado.** Há código, log de treino e predições.
- **Jow_1: parcialmente verificado.** Existe o `inference.py` da época (arquitetura sem gate, 2 camadas). O código de treino se perdeu, então w_A não é verificável.
- **Jow_2: só o paper.** Código e log se perderam: o notebook foi sobrescrito pelo treino do Jow_3. Restam apenas as predições.

⚠ `EXIST_MEMES/best_model/` **não** contém os pesos submetidos: foi sobrescrito em 6/set pelo A0 (ver `C_memes.md`). Os pesos dos 3 runs submetidos não existem mais.

---

## A1. Ponto de injeção demográfica

**Resposta:** o vetor entra no **CLS do texto já projetado para 512-d, antes da concatenação**. Não entra no vetor fundido de 1024-d.

`Thesis_main.ipynb`, célula 42 (Jow_3):
```python
text_cls = self.text_proj(text_out.last_hidden_state[:, 0, :])  # [B, 512]
gate = torch.sigmoid(self.demo_gate(demo_vec))                  # [B, 512]
text_cls = text_cls + gate * self.demo_proj(demo_vec)
...
fused = torch.cat([text_cls, vision_cls], dim=-1)               # [B, 1024]
fused = self.fusion_norm(fused)   # LayerNorm(1024)
```
- `demo_proj = Linear(66→512) + LayerNorm(512)`; `demo_gate = Linear(66→512)`.
- Jow_1 (em `Untitled-checkpoint.ipynb`): o mesmo ponto, sem gate: `text_cls = text_cls + self.demo_proj(demo_vec)`.
- Depois do `cat` existe um `LayerNorm(1024)` (`fusion_norm`) e um `Dropout(0.1)`.

**Paper vs código (alimenta C1):**
- O paper, na §4.1/Eq. 2, diz que a injeção é feita *"to the fused representation"*, com `W_d` projetando para **1024-d** depois da concatenação. **Falso.**
- O paper, na §4.4, diz que o gate é um vetor de **1024-d**. **Falso: tem 512-d.**
- A Eq. (injection) da tese, que assume o CLS do texto, **está correta**. A forma exata é `h_text = LN(W_t·CLS_text) + σ(W_g·d + b_g) ⊙ LN(W_d·d + b_d)`.

## A2. Atributos do vetor demográfico

**Resposta:** são **3 atributos** (etnia, escolaridade e país) e **66 dimensões** (7 + 6 + 53). A versão de 71 dimensões não existe no código submetido.

- `DEMO_FEATURE_GROUPS` na célula 10 tem `ethnicities_annotators`, `study_levels_annotators` e `countries_annotators`. A saída impressa é `Annotator vector dim: 66`.
- `compute_annotator_vector` (célula 27) conta os votos por categoria, concatena e **divide pela soma total do vetor**, não pela soma de cada atributo. Isso importa para o A14.
- **Por que gênero e idade ficaram de fora (útil para a §3.3.6):** pela célula 8 (`describe()`), `gender_annotators` tem `unique=1`: todos os 3.984 memes têm `[F,F,F,M,M,M]`. Já `age_annotators` tem `unique=2`, e as duas permutações têm a mesma contagem por faixa. Os dois atributos são **constantes depois da contagem** e não carregam informação. Vale a pena dizer isso explicitamente.

**Paper vs código:** o paper (§4.4) lista 5 atributos ("age group, gender, educational background, ethnicity, and country") e diz "approximately 66-dimensional". **O número de atributos está errado; as 66 dimensões estão certas.** A figura E2 deve ser desenhada com 3 atributos.

## A3. Quais runs usam o gate sigmoide

| Run | Gate | Evidência |
|---|---|---|
| Jow_1 | **Não** | `Untitled-checkpoint.ipynb` (inferência da época): `text_cls + self.demo_proj(demo_vec)` |
| Jow_2 | Não *(só o paper)* | Paper §4.7. Sem artefato de código |
| Jow_3 | **Sim** | `Thesis_main.ipynb` célula 42 + `inference.py` atual (que exige `demo_gate` no `state_dict`) |

## A4. Peso w_A por run

| Run | w_A | Evidência |
|---|---|---|
| Jow_1 | 1 *(só o paper)* | O paper fala em "uniform loss weighting". O código de treino se perdeu; o `inference.py` não contém a loss |
| Jow_2 | 2 *(só o paper)* | Paper §4.5 |
| Jow_3 | **2,0** | Célula 42: `loss = loss_a * 2.0 if loss is None else loss + loss_a * 2.0` |

As perdas de B e C têm peso 1. A perda total é `2·L_A + L_B + L_C`.

## A5. Camadas do CLIP treinadas por run

| Run | Camadas destravadas | Evidência |
|---|---|---|
| Jow_1 | **2** (as últimas) | `Untitled-checkpoint.ipynb`: `layers[-2:]` com o comentário "matches training" |
| Jow_2 | 4 *(só o paper)* | Paper §4.2 |
| Jow_3 | **4** (as últimas) | Célula 42 |

⚠ **Armadilha de nome:** na célula 42 a variável se chama `frozen_layers = 4`, mas ela **destrava** as 4 últimas camadas. Quem escrever a prosa a partir do nome da variável vai inverter o fato.
⚠ O `inference.py` atual tem `layers[-2:]` com gate. Essa combinação nunca foi treinada; é resíduo da versão do Jow_1. Na inferência isso é inofensivo (`requires_grad` não altera a saída), mas **não use o `inference.py` como fonte para o A5**.
- Os embeddings, o pré-LN e o `post_layernorm` do CLIP ficam congelados. O XLM-R-large é **100% treinável**, sem LoRA e sem congelamento.

## A6. `CLIPProcessor`

**Resposta:** há **center crop**. A entrada é **224×224**. Os parâmetros foram lidos do checkpoint `openai/clip-vit-large-patch14` na VM, com `transformers==5.6.0`:

| Passo | Valor |
|---|---|
| Classe | `CLIPImageProcessor` (MRO: `TorchvisionBackend → BaseImageProcessor`), ou seja, o resize é feito via torchvision |
| Resize | `shortest_edge=224`, `resample=3` (**bicúbico**), mantendo a proporção |
| Center crop | `crop_size=224×224` |
| Rescale | `1/255` |
| Normalize | média `(0.4815, 0.4578, 0.4082)`, desvio `(0.2686, 0.2613, 0.2758)` (estatísticas do CLIP, **não** do ImageNet) |
| `do_convert_rgb` | `True` (o notebook também faz `.convert('RGB')`) |

**Consequência medida (útil para a §3.3.2):** o center crop corta as bordas do eixo longo. Nos memes reais, a fração do eixo longo que sobra é:
- treino (n=3.984): mediana **0,81**; **33,9 %** perdem mais de 25 % do eixo longo; 2,6 % perdem mais de 50 %.
- teste (n=1.053): mediana 0,81; 34,9 % perdem mais de 25 %; 2,0 % perdem mais de 50 %.

Em memes altos ou largos, a legenda no topo ou na base pode ficar fora do que o CLIP vê. O texto chega ao modelo de qualquer forma, pelo XLM-R, a partir do campo `text`.

## A7. Convenção do sigmoide da Tarefa A

**Resposta: confirmada.** σ = P(NO), e a saída é montada como `[1 − σ, σ]` = `[P(YES), P(NO)]`.

- **Treino** (células 10 e 42): `LABELS_2_1 = ['YES','NO']`; a loss é `BCEWithLogits(logit_a, labels_a[:, 1])`, ou seja, o alvo é a fração de votos NO.
- **Métricas de validação** (célula 31): `prob_no = sigmoid(logit)`; `probs_a = stack([1 - prob_no, prob_no])`; índice 0 → `"YES"`.
- **`inference.py` atual** (linhas 28–32 e 315–324, 617–622): `p_no = sigmoid(logits_a)`; `p_yes = 1 - p_no`.
- **Nos 3 arquivos submetidos:** YES+NO = 1 em 1.053/1.053 casos; hard = argmax do soft em 1.053/1.053; correlação de P_A(YES) com P_B(NO) = **−0,96** em todos os runs. Nenhum run saiu invertido.
- ⚠ A versão antiga do `inference.py` (`Untitled-checkpoint.ipynb`, 13/mai 23:02) usava `p_yes = sigmoid(logit)`, o que seria **invertido** sob o alvo atual. Mesmo assim, a saída do Jow_1 é coerente. Logo, ou o alvo de treino do Jow_1 era P(YES), ou outra versão da inferência foi usada. **Não é verificável**, e não afeta a tese: os arquivos submetidos são coerentes.

A nota da §3.3.7 pode afirmar a convenção `[1 − σ, σ]` com σ = P(NO) para o sistema descrito, que é o do Jow_3.

## A8. Limiares da Tarefa C no teste — ⚠ achado principal

**Resposta: no Jow_3, os limiares aplicados no teste NÃO são os da época cujos pesos foram usados.**

1. `TrainingArguments` (célula 47): `load_best_model_at_end=True` e `metric_for_best_model="task_a_icmsoftnorm_soft"`. O máximo dessa métrica no log está na **época 19** (0,4907). Os pesos usados são, portanto, os da época 19.
2. Os limiares de **cada época** estão no log da célula 49. Os da época 19 são `[0.35, 0.45, 0.40, 0.35, 0.20, 0.25]` (NONE, IDEO, MISO, SEX-V, OBJ, STER).
3. **Reconstruí os limiares do arquivo submetido** comparando `task2_3_hard` com `task2_3_soft`. Para cada classe, o limiar t tem de ficar acima da maior probabilidade rejeitada e abaixo da menor aceita; exclui os casos de fallback por argmax. Intervalos obtidos:

| Classe | Intervalo inferido para t | Valor do grid | Época 19 | Épocas 28–30 |
|---|---|---|---|---|
| NONE ("NO") | (0,4932 ; 0,5011] | **0,50** | 0,35 | 0,50 |
| IDEOLOGICAL | (0,4385 ; 0,4651] | **0,45** | 0,45 | 0,45 |
| MISOGYNY-NSV | > 0,4064 | 0,45 (compatível) | 0,40 | 0,45 |
| SEXUAL-VIOLENCE | (0,3955 ; 0,4037] | **0,40** | 0,35 | 0,40 |
| OBJECTIFICATION | (0,2490 ; 0,2504] | **0,25** | 0,20 | 0,25 |
| STEREOTYPING | (0,3497 ; 0,3500] | **0,35** | 0,25 | 0,35 |

→ O conjunto submetido é **exatamente o das épocas 28, 29 e 30** (idênticos entre si), ou seja, o último que o log imprimiu. **Os pesos são da época 19 e os limiares foram ajustados nas probabilidades da época 30.** Os limiares ainda vieram da validação, sem leakage de teste, mas de outra época.

| Run | Limiares aplicados no teste | Fonte |
|---|---|---|
| Jow_1 | **0,5 uniforme (nenhum ajuste)** | Todas as cotas inferiores são < 0,5; nenhuma amostra tem ≥2 classes com p ≥ 0,5 (então não dá para distinguir de argmax puro); hard = argmax em 1.053/1.053. Bate com o default `[0.5]*6` do `inference.py` da época |
| Jow_2 | ≈ `[0.30, 0.25, 0.30, 0.35, 0.45, 0.40]` (inferidos) | Sem log de treino, não dá para saber de que época vieram |
| Jow_3 | `[0.50, 0.45, 0.45, 0.40, 0.25, 0.35]` = épocas 28–30 | Ver acima |

**Paper vs código:** o paper (§4.6) diz *"The thresholds from the best validation epoch were then used directly during inference"*. **Isso é falso para o Jow_3**, onde foi usada a última época e não a melhor, **e para o Jow_1**, onde não houve ajuste. Alimenta C1.

**Bônus, sobre o número "0,3948 vs 0,3761"** (paper §4.6 e CLAUDE.md): no log do Jow_3, 0,3948 é o valor **com limiares ajustados na época 19**. O valor 0,3761 **não aparece** no log; o mais próximo é 0,3762, o default 0,5 da época 27. Dentro da própria época 19, a comparação correta seria 0,3948 contra 0,3593, ou seja, **+0,036**, e não +0,019. A origem do 0,3761 não foi encontrada.

## A9. Contagem de parâmetros

Medido na VM em 2026-10-05: a arquitetura foi instanciada em CPU, sem treinar e sem carregar pesos treinados (os pesos submetidos não existem mais). Os números do Jow_3 batem **dígito a dígito** com a saída da célula 44 do notebook.

| Módulo | Total | Treinável (Jow_3) |
|---|---|---|
| XLM-R-large (`encoder`) | 559.890.432 | 559.890.432 (100 %) |
| CLIP ViT-L/14 vision (`vision_encoder`) | 303.179.776 | 50.384.896 (4 × 12.596.224) |
| `text_proj` / `vision_proj` | 525.824 cada | 525.824 cada |
| `demo_proj` | 35.328 | 35.328 |
| `demo_gate` (só Jow_3) | 34.304 | 34.304 |
| `fusion_norm` | 2.048 | 2.048 |
| `head_a` / `head_b` / `head_c` | 1.025 / 3.075 / 6.150 | idem |

| Run | Total | Treináveis | Congelados |
|---|---|---|---|
| Jow_1 (2 camadas, sem gate) | 864.169.482 | **586.182.154** (67,8 %) | 277.987.328 |
| Jow_2 (4 camadas, sem gate) | 864.169.482 | **611.374.602** (70,7 %) | 252.794.880 |
| Jow_3 (4 camadas + gate) | 864.203.786 | **611.408.906** (70,7 %) | 252.794.880 |

As linhas do Jow_1 e do Jow_2 foram medidas instanciando a mesma classe com o gate removido e com 2 ou 4 camadas destravadas, conforme a arquitetura descrita no paper.

**Checagem das frases:**
- "~560M" (XLM-R) ✔ 559,9 M.
- "~300M" (CLIP) ✔ 303,2 M.
- "menos de um bilhão de parâmetros treinados" ✔ 586–611 M. O total do modelo (864 M) também fica abaixo de 1 B.
- "under 700M trainable" (paper) ✔.
- ✘ **"Neither … requires full fine-tuning of large backbones"** (paper, Introdução): **falso para os memes**, porque o XLM-R-large (560 M) é ajustado por completo. Alimenta C1 ("sem fine-tuning completo").

## A10. Painéis fixos de anotadores

**Resposta:** o treino tem 151 painéis de 6 anotadores (confirmado). **O teste NÃO usa os mesmos painéis.**

| | Treino | Teste |
|---|---|---|
| Painéis distintos (tupla de `annotators`) | **151**, todos de 6 | **39**, todos de 6 |
| Memes por painel | mín. 9 · mediana 27 · máx. 27 · **média 26,4** | 27 em todos (39 × 27 = 1.053) |
| Anotadores distintos | 887 | 233 |
| Painéis do teste presentes no treino | — | **0 / 39** |
| Anotadores em comum | — | 14 |
| Multiconjuntos demográficos (etnia, escolaridade, país) do teste vistos no treino | — | **0 / 39** |

Os arquivos estão no repositório: `Dataset/memes/training/EXIST2026_training.json` e `Dataset/memes/test/EXIST2026_test_clean.json`. O teste **tem** os campos demográficos, então o vetor é calculado normalmente na inferência.

**Implicação para a tese:** o teste avalia a injeção demográfica com **painéis inteiramente novos**. Isso **reforça** o argumento da §3.4 do paper de que embeddings demográficos generalizam para anotadores não vistos, ao contrário de embeddings por identidade. Mas também significa que nenhuma composição demográfica do teste foi vista no treino: o modelo precisa interpolar. "~26 memes cada" está correto como média, mas não é uniforme (o mínimo é 9).

## A11. Hiperparâmetros de treino (Jow_3, células 45 e 47)

Todos confirmados no código, com os números derivados:

| Item | Valor | Observação |
|---|---|---|
| Épocas | 30 | Avaliação e checkpoint a cada época |
| Batch | 8 × acúmulo 8 = **64 efetivo** | 424 batches/época → **53 passos de otimização/época** (bate com `~/results/checkpoint-53`) → 1.590 no total |
| LR | 1e-5, **o mesmo para todos os módulos** | Scheduler e otimizador são os defaults do Trainer (linear, AdamW) |
| Weight decay | 0,05 | — |
| Precisão | `fp16=True` | Não é bf16 e não há LoRA nos memes |
| `max_length` | 128 | Treino e inferência |
| Split | 85/15 → **3.386 / 598** | `train_test_split(test_size=0.15, random_state=42, stratify=maioria_A)`. Regra de empate: soft[YES] ≥ 0,5 → YES (empate 3×3 vira YES) |
| Seleção | `task_a_icmsoftnorm_soft`, `load_best_model_at_end=True` | Melhor época = 19. `save_total_limit=2` |
| Dropout | 0,1 (após `fusion_norm`) | — |
| Seed | sem `set_seed` explícito; seed default do Trainer (42) | O modelo é instanciado **antes** do Trainer, logo a init das camadas novas não é semeada. `EarlyStoppingCallback` é importado e **não** é usado |

A informação "veio da documentação" agora está **confirmada no código**.

## A12. Warmup

**Resposta:** a run submetida usou `transformers==5.6.0`. Com ela, `warmup_steps=0.1` é **10 % dos passos**: `ceil(0,1 × 1.590) = 159` passos ≈ 3 épocas.

- `transformers-5.6.0.dist-info` na VM tem data de **2026-04-22 23:01** e não foi reinstalado depois. As runs são de 13 e 14 de maio, logo usaram a mesma versão. Não usei a saída do `pip` na célula 0 como prova: as saídas de célula do notebook não são todas da mesma sessão (a célula 2 mostra datas de 1º de agosto).
- Código-fonte na VM, `transformers/training_args.py:2111–2112`: `int(warmup_steps) if warmup_steps >= 1 else math.ceil(num_training_steps * warmup_steps)`.
- O paper diz "warmup_steps = 0.1" para os memes. **Está correto**, mas convém escrever "10 % dos passos (159)" na tese.

## A13. Bug da segmentação de hashtags

- **Onde:** `normalize_hashtags` (célula 18) é **chamada**: é o primeiro passo de `pre_process_text`, na célula 23. O que nunca foi chamado é `wordsegment.load()`, que carrega o dicionário de frequências.
- **Mecanismo:** sem `load()`, `segment(tag)` levanta `ValueError: max() iterable argument is empty` (verificado na VM). O `except ValueError: words = [tag]` engole o erro **em silêncio**, e a hashtag sai inteira, só sem o `#`. Os ramos com `_` e CamelCase não dependem do dicionário e sempre funcionaram. Só as hashtags **todas em minúsculas** eram afetadas.
- **Alcance (medido):** no treino, 76 textos têm hashtag, há 32 hashtags só em minúsculas e **10 / 3.984 textos mudam** com o `load()`. No teste, 21 textos têm hashtag, 3 são só em minúsculas e **2 / 1.053 mudam**.
- **Onde existe:** o bug está no `Thesis_main.ipynb` **e** no `inference.py` (nenhum dos dois chama `load()`). O sistema submetido era **consistente entre treino e teste**. A correção existe só no `memes_core.py` (linha 40) e no `memes_model.ipynb` (célula 5).
- ⚠ **Efeito colateral da correção:** o `wordsegment` usa um corpus em **inglês**. As 2 hashtags do teste viram lixo: `#enfrijoladas → "en fri jola das"` e `#masculinismo → "masculin is mo"`. Na §3.3.11, não apresentar a correção como melhoria inequívoca. O alcance é ínfimo (12 textos no total), e para hashtags em espanhol a correção piora.

## A14. Bug do apóstrofo curvo

- **64,6 % confirmado:** no treino há 23.904 votos de escolaridade e **15.435 descartados (64,6 %)**: `Bachelor’s degree` 11.508 e `Master’s degree` 3.927. No teste foram descartados 4.023 / 6.318 (**63,7 %**).
- **Amostras afetadas:** 3.957 / 3.984 no treino (bate com o CLAUDE.md) e **1.053 / 1.053** no teste.
- **Mecanismo:** o vocabulário `DEMO_STUDY_VALS` (célula 10) usa o apóstrofo reto `'`, e os dados usam o curvo `’` (U+2019). A célula 33 do próprio notebook imprime `'Bachelor’s degree'`, então o descasamento estava visível na saída. `vocab_index.get(...)` devolve `None` e o voto é descartado sem aviso.
- **Correção:** `_normalize_apostrophes` (`memes_core.py:329–351` e célula 29 do `memes_model.ipynb`) aplica `.replace("’", "'")` dos dois lados da comparação. O vocabulário e a ordem das 66 posições não mudam.
- ⚠ **Efeito de segunda ordem (citar na §3.3.11):** como o vetor é normalizado pela **soma total**, descartar votos de escolaridade reduz o denominador e **infla as fatias de etnia e país**. O bug não só zerava a escolaridade: mudava a escala das 66 posições. A correção altera todas elas.

## A15. Nome do sexto rótulo da Tarefa C

**Resposta: `NO`** é o rótulo oficial.
- Nos golds oficiais (`Dataset/evaluation/golds/EXIST2025_training_task2_3_gold_*.json`), as chaves são `NO`, `IDEOLOGICAL-INEQUALITY`, … O mesmo vale para a Tarefa B (`NO`, `JUDGEMENTAL`, `DIRECT`).
- Nas guidelines (`EXIST_2026_Lab_Guidelines.V0.5.pdf`), a hierarquia é `{"YES": [...], "NO": []}`.
- Na anotação bruta o valor é `-` (10.618 ocorrências na Tarefa C do treino), além de `UNKNOWN` (65).
- `NONE` existe **só dentro do código** (`LABELS_2_3[0]`). Na saída para o PyEvALL ele é mapeado para `"NO"` (`label_map = {"NONE": "NO"}`).

## A17. Fórmula da ICM-Norm no PyEvALL

Fonte: `pyevall==0.1.78`, `site-packages/pyevall/metrics/metrics.py` na VM.

```python
# ICMNorm.evaluate, linhas 613-639 (ICMSoftNorm é idêntico, linhas 937-963)
res_pred = ICM(pred, gold)                 # média por instância
res_gold = ICM(gold, gold)
icm_norm = (res_pred - (-res_gold)) / (res_gold - (-res_gold))
if icm_norm < 0: icm_norm = 0
```

$$\text{ICM-Norm} = \max\!\left(0,\ \frac{\text{ICM}(\hat y, y) + \text{ICM}(y, y)}{2\,\text{ICM}(y, y)}\right)$$

- **ICM por instância** (linhas 527–546): `α₁·IC(pred) + α₂·IC(gold) − β·IC(pred ∪ gold)`, com **α₁ = α₂ = 2, β = 3**.
- **IC(c) = −log₂ P(c)**, com P(c) estimada pela frequência da classe no **gold**. Para conjuntos, a hierarquia entra pelo ancestral comum mais profundo (`information_content`, linha 547).
- **Interpretação:** 1,0 equivale a uma predição tão informativa quanto o gold; **0,5 equivale a ICM = 0**; 0,0 equivale a ICM ≤ −ICM(gold, gold). **Não há corte superior** (não existe `min(1, …)`).
- **A normalização é relativa à autoinformação do gold, não aos outros sistemas.** A redação atual da §3.6.1, que não afirma que 0,0 é o pior sistema submetido, está correta. ICM-Soft usa a mesma normalização sobre `ICMSoft`.

---

## Resumo para a errata (C1): divergências paper × código encontradas aqui

1. Injeção demográfica no **CLS do texto em 512-d**, e não no vetor fundido em 1024-d (A1).
2. Gate de **512-d**, e não de 1024-d (A1).
3. **3 atributos** demográficos, e não 5 (A2).
4. Limiares da Tarefa C: Jow_3 usou os da **época 30** com pesos da época 19, e o Jow_1 **não ajustou** limiares. O paper diz "best validation epoch" (A8).
5. "0,3948 vs 0,3761": a origem do 0,3761 não foi encontrada. Na mesma época, o ganho é +0,036 (A8).
6. "Sem fine-tuning completo de backbones grandes": o XLM-R-large é **totalmente ajustado** (A9).

## Itens que continuam sem verificação possível

- w_A do Jow_1, e a arquitetura e o w_A do Jow_2: o código se perdeu, e só o paper sustenta.
- De qual época vieram os limiares do Jow_2: não há log.
- Por que a versão antiga do `inference.py` usa `p_yes = sigmoid` e mesmo assim o Jow_1 sai coerente (A7).
