# 18 — Braço CLIP para vídeo (item 10): decisões, forma do código, limitações

**Escrito em:** 2026-08-13
**Escopo:** documento de decisão/rastreabilidade para o braço `EXISTVideoModelCLIP`
(`video_core_clip.py` + `Thesis_video_clip.ipynb`). Não é um relatório de resultados — não
há números de treino aqui, só as escolhas de desenho e os riscos pré-declarados. Fonte
autoritativa: o item 10 de `17_RETRAIN_IMPROVEMENT_OPTIONS.md` (o "porquê" desta rodada) e
o plano de design desta sessão (o "como").

**Convenção de marcação** (igual à dos docs 15/16):
- `[DECISÃO DO AUTOR]` — escolha feita pelo autor nesta sessão, não derivada por mim.
- `[VERIFICADO]` — conferido diretamente no código ou por aritmética nesta sessão.
- `[HIPÓTESE]` — interpretação ou expectativa. Não escrever na tese como fato.

---

## 1. Decisões congeladas pelo autor `[DECISÃO DO AUTOR]`

Estas três decisões fecham o espaço de desenho do braço CLIP e tornam os arquivos A-E do
plano paralelizáveis sem coordenação — qualquer implementação que as contradiga está fora
de especificação:

1. **16 frames por vídeo** passam pelo CLIP — não 8, não 4. A "decisão em aberto" que o
   item 10 do doc 17 deixava pendente ("frames-chave, subconjunto menor por custo") foi
   fechada: usar os mesmos 16 índices que já alimentam o VideoMAE hoje, mantendo o braço
   CLIP comparável em quantidade de conteúdo visto, não só em arquitetura.
2. **4 últimas camadas do CLIP destravadas, full-rank, sem LoRA** — replica literalmente o
   padrão já em produção em `EXIST_MEMES/memes_core.py:488-494` (unfreeze das últimas N
   camadas do `CLIPVisionModel`, ~50,4 M parâmetros treináveis). Não é um novo desenho de
   adaptação; é o precedente que já funciona no projeto, no mesmo tipo de dado (imagem) e
   mesma ordem de tamanho de dataset, transplantado sem reinterpretação.
3. **Módulo overlay, `video_core.py` intocado.** Todo o código novo vive em
   `video_core_clip.py`, que importa de `video_core` em vez de duplicar. Isto satisfaz duas
   coisas ao mesmo tempo: o pedido explícito do autor de não tocar no arquivo que já
   sustenta as 5 runs VideoMAE existentes, e a convenção do projeto de árvore nova para
   mudança estrutural (mesmo espírito do `v2/` para re-extração) — sem exigir sequer um
   `.py.bak`, porque o original nunca é editado.

---

## 2. O wrapper `CLIPFrameEncoder` — por que essa forma

`EXISTVideoModel.forward` (herdado, não reescrito) espera de `self.videomae` um objeto com
`.last_hidden_state`, resultado de um único `forward(pixel_values=[B,16,3,224,224])`. O
`CLIPVisionModel` nativo não aceita esse shape — ele processa uma imagem por chamada. A
alternativa óbvia seria copiar o `forward` inteiro de `EXISTVideoModel` para um modelo CLIP
paralelo, mas essa é **o maior anti-padrão já documentado no projeto**: cópias divergentes
do mesmo pipeline (modality dropout, `_mask`, fusão de texto+áudio+bio, três heads,
`compute_loss`) que silenciosamente saem de sincronia quando uma é corrigida e a outra não
— exatamente a categoria de bug que motivou o `video_core.py` inteiro nesta auditoria.

Em vez disso, `CLIPFrameEncoder(nn.Module)` se **apresenta** com a mesma interface que o
forward existente já consome:

```
forward(pixel_values: [B,16,3,224,224]) -> objeto com .last_hidden_state: [B,16,1024]
  internamente: view(B*16,3,224,224) -> CLIPVisionModel -> last_hidden_state[:,0,:]
                (CLS PRÉ-post_layernorm — cópia literal de memes_core.py:573,
                 confirmado nesta sessão, NÃO usar pooler_output)
                -> view(B,16,1024)
gradient_checkpointing_enable(**kwargs) -> delega a self.clip
  (resolve video_core.py:1077-1082 sem precisar de override no modelo)
```

Com isso, `EXISTVideoModelCLIP` herda de `vc.EXISTVideoModel` e só substitui `video_pool`
(dim 1024) e `fusion` (dim nova) no `__init__` — `vc.set_all_seeds()` é re-chamado
imediatamente antes dessa substituição, porque os módulos descartados já consumiram RNG e,
sem re-seed, o init dos módulos novos não seria determinístico (mesmo padrão já usado no
projeto para LoRA A do VideoMAE/XLM-R). **O `forward` inteiro — modality dropout, `_mask`,
texto, áudio, bio, fusão, heads, `compute_loss` — é herdado sem reescrita.** Um bug corrigido
em `video_core.py` continua corrigido automaticamente nos dois braços, porque só existe uma
cópia dele.

---

## 3. Limitação conhecida nº 1 — interpolação bilinear vs. bicúbica

Os frames em disco (`frames/v2/`) foram redimensionados na extração pelo
`VideoMAEImageProcessorFast`/`VideoMAEImageProcessor` (confirmado na VM em 2026-08-09: são
a **mesma classe**), que usa **interpolação bilinear** (`resample=2`), via torchvision.
`CLIPProcessor`/`CLIPImageProcessor` nativo do checkpoint `openai/clip-vit-large-patch14`
usa, por padrão, **interpolação bicúbica**. O braço CLIP desta rodada **não re-extrai
pixels** — ele consome os mesmos `.pt` já em disco e só reaplica a normalização (afim,
`selfcheck_renorm_equivalence`), então herda o resize bilinear já cravado nesses arquivos em
vez do bicúbico que o CLIP "esperaria" nativamente.

`[HIPÓTESE, não verificável a partir do repositório]`: o efeito provável é uma pequena perda
de nitidez de alta frequência em relação ao regime de pré-treino do CLIP, análoga (mas não
idêntica) ao risco residual já registrado no `CLAUDE.md` para o VideoMAE sob `do_rescale=False`
(T4b). Não há como medir a divergência bicúbica vs. bilinear sem rodar os dois pipelines
lado a lado na VM — **isto não bloqueia esta rodada**: é um item de risco documentado, a
revisitar se o braço CLIP performar abaixo do esperado e a causa não for óbvia por outro
lado (ex.: LR, capacidade, ou o próprio teto de sinal do vídeo já medido no doc 16).

---

## 4. Limitação conhecida nº 2 — perda de temporalidade por tubelet

O VideoMAE processa os 16 frames como um clipe único via `tubelet_size=2`, que pareia
frames adjacentes para captar movimento de curta duração (ainda que — doc 16 e `CLAUDE.md`
— esse pareamento já opere num regime degenerado, com gap mediano de 1,21 s entre frames
consecutivos, ~8,6× mais esparso que o pré-treino do VideoMAE). O CLIP processa cada frame
**isoladamente**: não existe pareamento algum entre frames dentro do encoder. A única
temporalidade que sobra no braço CLIP é a que o `TemporalAttentionPool` já existente
(`video_core.py:949-967`, reaproveitado sem reescrita) impõe **depois** do encoder, sobre os
16 vetores já prontos.

**Argumento a favor de que isto não é um custo severo**: o doc 16 (`[VERIFICADO]`, §3.3 e
seção de achados) mediu que o `TemporalAttentionPool` **já não é temporal na prática** —
TESTE 6 mostrou `mean_all`, `mean_temporal` e `top5` empatando dentro da tolerância com o
pool aprendido, e o sinal que o ramo de vídeo carrega hoje se comporta como **propriedade de
frame único** (classe de intenção/formato — como a pessoa está enquadrada, se fala para a
câmera), não como movimento entre frames. Ou seja: a modelagem temporal que o VideoMAE
oferece e o CLIP não oferece já não estava sendo aproveitada pelo checkpoint atual. Isto
favorece a aposta, mas continua sendo leitura do doc 16 aplicada por analogia — `[HIPÓTESE]`
que o mesmo padrão se repete no braço CLIP, não medição direta dele.

---

## 5. Wart de nomenclatura no state_dict

`EXISTVideoModel.__init__` atribui o encoder de vídeo ao atributo `self.videomae`
(herdado, não renomeado — renomear quebraria a herança sem reescrita de `forward`). Como
`EXISTVideoModelCLIP` herda dessa classe e só troca o que `self.videomae` **aponta para**,
as chaves do state_dict nascem como `videomae.clip.*` — um `CLIPVisionModel` morando sob um
atributo chamado `videomae`. É uma inconsistência de nome aceita conscientemente, não um bug.

Duas mitigações, ambas obrigatórias:
- O submódulo interno do encoder CLIP é nomeado `self.clip` (não `self.vision_model` nem
  outro nome genérico), então `videomae.clip.*` é grepável e o padrão salta aos olhos de
  quem inspeciona o checkpoint.
- `meta.json["video_encoder"]` é o **desambiguador oficial**: `save_run_clip` grava
  `"video_encoder": "clip"` (contra `"videomae"` implícito nos runs antigos), e
  `load_run_clip` faz `assert` sobre esse campo antes de carregar. **Nunca inferir o tipo de
  encoder pelo nome das chaves do state_dict** — só pelo `meta.json`.

---

## 6. Custo e risco de OOM

Tabela de riscos reproduzida do plano aprovado (`R1`–`R6`; `R7`-`R9` pertencem ao braço do
item 11, fora do escopo deste documento):

| # | Risco | Sinal | Contingência (nesta ordem) |
|---|---|---|---|
| R1 | **OOM** — 32 imagens ViT-L/forward (16 frames × micro-bs 2) + grad em 4 camadas. O modelo de memes já deu OOM em ~16GB com 1 imagem/amostra | `CUDA OutOfMemoryError` no smoke test | (1) confirmar bf16+grad-checkpoint; (2) `frame_chunk` no `CLIPFrameEncoder` (2×16 ou 4×8 — numericamente exato, LayerNorm não BatchNorm); (3) micro-bs 1 + accum 32; (4) **nunca reduzir os 16 frames sem perguntar ao autor de novo** |
| R2 | LR errado — CLIP cai em `base_decay` 1e-3, destrói pesos pré-treinados, sem erro | métricas ruins sem exceção | os 2 asserts de `get_optimizer_clip` (partição exata + contagem ~40-60M) pegam isso antes do treino |
| R3 | Unfreeze não pega (caminho de atributo mudou de versão do `transformers`) | assert de contagem falha ou grads zerados no smoke test | copiar literalmente `memes_core.py:488-494`; se falhar, imprimir os nomes reais dos módulos e ajustar |
| R4 | Época mais lenta que VideoMAE (CLIP-L × 16 frames é mais caro por forward que um clipe único via tubelet) | timing extrapolado do smoke test | decidir antes das 15 épocas: `frame_chunk` maior sem checkpointing, ou reduzir orçamento de épocas com a perda de comparabilidade declarada |
| R5 | Kernel não-determinístico do CLIP sob `use_deterministic_algorithms(True)` | `RuntimeError` nomeando o operador | já de-riscado — `memes_core.py` roda CLIP-L com a mesma flag em produção (checar antes de assumir novidade) |
| R6 | Checkpoint ~220MB (vs 44MB dos runs VideoMAE) | — | documentado em `save_run_clip`; `save_strategy="no"` evita checkpoints intermediários |

---

## 7. O que NÃO muda — variáveis controladas

O braço CLIP troca **só** o encoder de vídeo e a fusão que depende da sua dimensão. Tudo o
resto é herdado sem alteração, precisamente para que qualquer delta observado seja
atribuível ao encoder e não a uma mudança colateral:

- **Amostragem:** mesmos 16 índices `linspace` sobre os 64 frames em disco (§1, ponto 1).
- **Split:** `SPLIT_SEED` congelado, mesmo `build_frozen_split` do doc 09 (2 019/505).
- **Áudio:** v2 (máscara de padding + média ponderada por janela de 30s), o mesmo que o
  baseline B15 já usa — `AUDIO_PROJ_DIM`, `frames_dir`/`audio_dir` importados de `vc`, não
  reimplementados.
- **Bio:** `BIO_HIDDEN_DIM`, `RobustScaler`, `use_bio`/`use_viewer_emb` — caminho idêntico.
- **Texto + LoRA:** XLM-R com `make_lora_config()` importado de `vc`, sem mudança de `r`,
  `alpha` ou `target_modules` (isso é o item 1 do doc 17, um braço separado, não combinado
  aqui).
- **Loss:** `compute_loss`/`loss_task_a/b/c` importados de `vc`, pesos `LOSS_W_A/B/C`
  inalterados (o item 3 do doc 17 é outro braço separado).
- **Métricas:** `compute_metrics`, `_format_for_pyevall`, `tune_task_c_thresholds` — mesmo
  caminho de avaliação do baseline, para que o número final seja comparável.
- **Orçamento de treino:** 15 épocas, mesma `TrainingArguments` (bs 2, accum 16, bf16,
  gradient checkpointing) da run congelada de referência (B15) — a comparabilidade do
  braço CLIP com o baseline depende de não mudar o orçamento junto com o encoder.

---

## 8. Disciplina de atribuição

O braço CLIP **não é comparável por state_dict** com as runs VideoMAE existentes (dimensão
de fusão diferente: 2944/2816 vs. 2688/2560) e **não substitui** o baseline B15 nem os
braços A′/A/B do doc 09. É reportável apenas no mesmo molde de atribuição em 3 braços que o
`09_BASELINE_RESULTS.md` já estabeleceu:

- A única comparação legítima é **Δ contra B15** (`run_B2_bio_viewer_seed42_B15/`, frames
  v2 + áudio v2 + `restore_biases=True`, 15 épocas, estado final da run — mesma regra de
  simetria que o `CLAUDE.md` já fixa: nunca comparar pico de um lado com estado final do
  outro).
- O braço CLIP muda **duas coisas ao mesmo tempo** em relação ao B15 por construção do item
  10 (encoder e dimensão de fusão) — isso é aceito porque a troca de encoder inteira é a
  hipótese sob teste (H2 do doc 16: capacidade absoluta + objetivo de pré-treino), não uma
  correção pontual. Não combinar, na mesma run, com o item 1 (LoRA no texto) ou qualquer
  outro item do doc 17 — o Combo D do doc 17 já registra essa ordem: item 10 isolado
  primeiro, combinação depois, nunca junto na primeira medição.
- O notebook `Thesis_video_clip.ipynb` declara isso na célula 0 (markdown): braço de
  atribuição isolado, incomparável por state_dict com as runs VideoMAE, divergência
  bilinear/bicúbica declarada (ver §3 deste documento).
- Nenhum número deste braço entra no paper já submetido — CEUR-WS está fechado; este braço,
  se rodar, é material de tese (arco experimental), na mesma categoria dos demais itens do
  doc 17 que ainda não foram executados.

---

## 9. Resultados e histórico de decisões pós-implementação `[VERIFICADO]`

Este documento era só de desenho até aqui (§1-8, escrito antes de qualquer treino). A
seção abaixo registra o que aconteceu depois — 4 treinos, 3 sondas causais, duas hipóteses
de correção testadas e refutadas, e a decisão final de estado. Datas: 2026-08-13 a
2026-08-15.

### 9.1 As quatro runs, em ordem cronológica

**Run 1 — 4 camadas, `save_strategy="no"`, sem seleção de época (1º treino).**
Estado salvo = final da época 15 por padrão (não havia outra opção no código na época).

```
final_metrics: A 0,5741 / 0,5172  ·  B 0,4986 / 0,4122  ·  C 0,3640 / 0,1368
eval_loss final = 2,0885 (vs eval_loss mínimo teórico, não medido nesta run)
```

Sonda causal (`probe_video_contribution_clip.ipynb`) revelou padrão anômalo: o ramo de
texto (XLM-R+LoRA, código idêntico ao B15) ficou **quase constante entre 505 amostras
distintas** — `‖Δ_texto‖ = 0,785` (24× menor que os 18,596 do B15) — enquanto o vídeo
passou a dominar causalmente todas as tasks (`‖Δ_vídeo‖ = 23,658`, Δ de −0,20 a −0,33 nas
seis métricas, do tamanho que o texto causava no B15). `perm_within` de B_hard ≈ 0
(+0,0026) — o slot de vídeo carrega informação de classe de B (formato/registro da
fonte), não conteúdo.

**Hipótese 1 (overtraining) formulada e testada — Run 2.** Adicionado
`SaveTrainableStateCallback` (checkpoint enxuto por época) + seleção
`E = argmin(eval_loss)` pós-treino, mesmo protocolo ARCH-06 do braço VideoMAE. 2º treino,
mesmo código (4 camadas), `eval_loss` mínimo na **época 2** (1,5158), subindo
monotonicamente depois (até 2,0885 na 15 — determinismo confirmado: idêntico byte a byte
ao 1º treino).

```
final_metrics (época 2): A 0,4284 / 0,3736  ·  B 0,3769 / 0,2687  ·  C 0,2674 / 0,0388
```

Sonda no checkpoint época 2: `‖Δ_texto‖ = 3,138` (melhora 4× sobre o colapso, mas ainda
6× abaixo do B15), `‖Δ_vídeo‖ = 17,315` (continua dominando, "MUDOU" em 4-5/6 métricas
contra B15). **Hipótese 1 REFUTADA em forma forte**: overtraining explica só parte do
padrão, não o todo — o colapso já está presente cedo no treino, não é degeneração tardia
acumulada.

**Hipótese 2 (desequilíbrio de capacidade do encoder) formulada e testada — Run 3.**
Raciocínio: CLIP full-rank (~50,4M params treináveis) vs LoRA-texto (~1,57M) — reduzir
`UNFROZEN_VISION_LAYERS` 4→2 (~25,2M) deveria rebalancear a competição de gradiente. 3º
treino (2 camadas), `E = argmin(eval_loss) =` **época 4** (1,4802).

```
final_metrics (época 4, 2 camadas): A 0,5271 / 0,4554  ·  B 0,4571 / 0,3544  ·  C 0,3393 / 0,0524
```

Sonda: `‖Δ_texto‖ = 1,046` (**PIOROU** em relação à Run 2, quase de volta ao nível de
colapso), `‖Δ_vídeo‖ = 18,433` (**subiu**, agora maior que o B15). **Hipótese 2 REFUTADA
— e na direção oposta à prevista**: menos capacidade no encoder tornou o vídeo MAIS
dominante, não menos.

**Mecanismo real (reconciliação com o advisor, não medido diretamente, `[HIPÓTESE]` bem
fundamentada):** `video_pool` e `fusion` são sempre full-rank e treináveis com
`lr=1e-3` (`base_decay`), **independentemente** de `UNFROZEN_VISION_LAYERS` — essa
constante só controla quanto o CLIP *se adapta*, não a força do que ele já entrega
congelado. Features do CLIP-L pré-treinado já são semanticamente fortes (diferente do
VideoMAE); a competição pela fusão é decidida pela **qualidade relativa das features**,
não pela contagem de parâmetros treináveis do encoder. Com 2 camadas, o CLIP fica mais
genérico/estável (menos overfit) — um atalho ainda mais confiável para a fusão, daí a
dominância ter subido. `‖Δ_texto‖` baixo em todas as variantes significa que o XLM-R+LoRA
foi **treinado** até virar quase-constante — o gradiente conjunto ensina o texto a se
calar porque o vídeo já resolve.

**Run 4 — 4 camadas revertido, estado FINAL da época 15 (decisão final, 4º treino).**
Após as duas hipóteses de correção falharem, decisão do autor (confirmada explicitamente
antes da implementação): reverter `UNFROZEN_VISION_LAYERS` 2→4 e abandonar
`E=argmin(eval_loss)` como critério de seleção para este braço — reportar o **estado
final da época 15**, por simetria com o baseline VideoMAE (regra do `CLAUDE.md`: estado
final vs estado final, nunca fim de um lado e pico do outro) e porque o padrão passou a
ser lido como **trade-off real de trocar o encoder**, não bug corrigível por seleção de
época ou capacidade. Esta run sobrescreveu o artefato da Run 2 em `run_..._CLIP` — a
reprodução é determinística (confirmado nas Runs 1↔2), então o estado final é
equivalente ao da Run 1.

Curva completa de 15 épocas desta run (idêntica à da Run 2 até a época 15, já que é o
mesmo treino determinístico):

| época | eval_loss | A hard | A soft | B hard | B soft | C hard | C soft |
|---|---|---|---|---|---|---|---|
| 1 | 1,6648 | 0,4700 | 0,3244 | 0,2943 | 0,2034 | 0,2048 | 0,0000 |
| 2 (argmin global) | 1,5158 | 0,4284 | 0,3736 | 0,3769 | 0,2687 | 0,2674 | 0,0388 |
| 3 | 1,5500 | 0,5130 | 0,4386 | 0,4044 | 0,3208 | 0,2874 | 0,0689 |
| 4 | 1,5742 | 0,5406 | 0,4865 | 0,4817 | 0,3777 | 0,3280 | 0,0834 |
| 5 | 1,7391 | 0,5162 | 0,4804 | 0,4362 | 0,3653 | 0,3139 | 0,1025 |
| 6 | 1,6614 | 0,5360 | 0,4750 | 0,4680 | 0,3727 | 0,3110 | 0,0810 |
| 7 (pico local de A/B) | 1,6121 | **0,5801** | 0,5063 | **0,5046** | 0,3963 | 0,3570 | 0,1217 |
| 8 | 1,6923 | 0,5604 | 0,4937 | 0,4884 | 0,3896 | 0,3379 | 0,1081 |
| 9 | 1,8688 | 0,5742 | 0,5082 | 0,4995 | 0,4010 | 0,3454 | 0,1246 |
| 10 | 1,7995 | 0,5767 | 0,4931 | 0,5091 | 0,3959 | 0,3685 | 0,1172 |
| 11 | 1,8869 | 0,5769 | 0,5034 | 0,4841 | 0,3996 | 0,3684 | 0,1259 |
| 12 | 1,9240 | 0,5773 | 0,5095 | 0,4984 | 0,4053 | 0,3595 | 0,1269 |
| 13 | 2,0840 | 0,5709 | 0,5148 | 0,4800 | 0,4068 | 0,3645 | 0,1342 |
| 14 | 2,0812 | 0,5710 | 0,5161 | 0,4904 | 0,4104 | 0,3631 | 0,1367 |
| **15 (final, oficial)** | 2,0885 | 0,5741 | 0,5172 | 0,4986 | 0,4122 | 0,3640 | **0,1368** |

### 9.2 O debate "época 2 vs época 7 vs época 15" — resolvido, registrado para não repetir

Depois de ver a curva acima, foi levantada a hipótese de que a época 7 seria uma escolha
melhor que a 15: `eval_loss` tem um vale local ali (1,6121, menor que as épocas 5/6/8/9
vizinhas) e a Task A/B atingem seus picos absolutos nessa época (A hard 0,5801, o maior
de todas as 15 épocas; B hard 0,5046, quase o maior).

**Rejeitado, por dois motivos independentes:**

1. **Não é argmin.** O mínimo global de `eval_loss` continua sendo a época 2 (1,5158 <
   1,6121). Época 7 só ganhou consideração depois de olhar que as métricas de A/B eram
   boas ali — é exatamente o mecanismo de leakage que a regra do projeto proíbe (escolher
   a época pela métrica reportada). Teste decisivo: sem olhar nenhuma métrica, só
   `eval_loss`, a época 2 vence sem disputa.
2. **Não existe época de equilíbrio nesta curva.** As quatro (sub)tasks têm picos em
   épocas diferentes: A em 7 (0,5801), B em 10 (0,5091), C hard em 10-11 (0,3685/0,3684),
   C soft em 15 (0,1368, ainda subindo no fim do orçamento). "Época 7 balanceia as
   métricas" é empiricamente falso — é o pico de uma task só.

Decisão final mantida: **estado da época 15**, sem exceção.

### 9.3 Divergência sistemática entre `eval_loss` e Task C soft — abriu novo escopo

Task C soft (`icmsoftnorm_soft`) continua melhorando bem depois do argmin de
`eval_loss` (0,0388 no argmin da época 2 → 0,1368 na 15, ~3,5×), enquanto A/B saturam
mais cedo. Investigado à parte em `20_EPOCH_SELECTION_TASK_C.md` (pesquisa de literatura
sobre seleção de checkpoint em MTL quando tasks divergem) e, por extensão, em
`21_HEAD_PROBING_PREREGISTRATION.md` (probing das 3 heads sobre trunk congelado,
generalizando a opção 4 do doc 20 às três tasks, com condição de rodar também sobre o
B15 por simetria). Nenhuma dessas duas rotas mexe no trunk deste braço — são variações
reportadas separadamente, nunca substituindo os números da seção 9.1 sem decisão
explícita do autor.

### 9.4 Estado atual e pendências

- **Artefato oficial:** `run_B2_bio_viewer_seed42_CLIP/` — 4 camadas, estado final da
  época 15, números da Run 1/Run 4 (§9.1).
- **Pendente:** rodar a sonda causal (`probe_video_contribution_clip.ipynb`) sobre o
  artefato da Run 4 para confirmar que reproduz os números da Run 1 (`‖Δ_texto‖ = 0,785`
  esperado) — determinismo já demonstrado nos `final_metrics.json`, mas a sonda em si
  ainda não foi re-rodada especificamente sobre este artefato regenerado.
- **Achado de robustez para a tese, não corrigido aqui:** o slot de vídeo carrega
  informação de formato/registro da fonte (classe de B), não conteúdo — um modelo que
  troca canal de conteúdo por canal de formato pode generalizar pior no teste real mesmo
  com métricas de validação melhores. Argumento de robustez contra este braço,
  independente do ganho agregado.
- **Não tentar de novo:** `UNFROZEN_VISION_LAYERS` em qualquer valor diferente de 4 (eixo
  refutado, moveu na direção errada) e qualquer critério de seleção de época que não seja
  "estado final" ou "argmin(eval_loss) formal, pré-registrado antes de ver a curva" —
  ver §9.2.

### 9.5 Run 5 — repesagem da Task C (opção 5 do doc 20), decidida 2026-08-15

**Decisão do autor**, tomada depois do probing de heads dar INCONCLUSIVO (doc 21 §6):
aplicar a opção 5 do doc 20 em vez de encerrar o braço. É a única rota que restava e é
correção de causa nos **dois** ramos da bifurcação em aberto ("C sub-treinada" vs
"representação sem sinal de C") — nenhum dos dois precisa ser resolvido antes de agir.

**A mudança, e só ela:** `LOSS_W_C = 0,7 → 4,2`. Valor **principiado, não afinado**:
`loss_task_c` reduz as 6 classes com `.mean(dim=-1)` enquanto `loss_task_b` usa
`.sum(dim=-1)`, deixando o peso efetivo de C em `0,7 × 1/6`; `0,7 × 6 = 4,2` desfaz
exatamente essa assimetria aritmética, equivalente a trocar a redução para `.sum`.
Nenhum grid, nenhuma escolha por métrica de validação.

**Forma da mudança (decidida com o autor antes de tocar em código):**
`vcc.apply_loss_reweight_c()` muta `vc.LOSS_W_C` em runtime, chamada de uma célula
explícita (§2.2 do notebook). `vc.compute_loss` (video_core.py:1358) resolve o global a
cada chamada e o `forward` chama a função de módulo (video_core.py:1156), logo a mutação
surte efeito sem editar `video_core.py` (regra forte do projeto) e sem duplicar a loss.
A mutação **nunca** acontece no import do overlay — as duas sondas e `load_run_clip`
também importam `video_core_clip` e passariam a computar loss repesada em silêncio.
Um `assert vc.LOSS_W_C == 4.2` imediatamente antes de `trainer.train()` protege contra
execução fora de ordem do notebook.

**Congelado em relação à Run 4:** `UNFROZEN_VISION_LAYERS=4`, estado reportado = FINAL da
época 15 (sem reload de argmin), `INIT_SEED=42`, `SPLIT_SEED`, `TrainingArguments`,
áudio v2, frames v2. Um eixo por run.

**Artefato:** `run_B2_bio_viewer_seed42_CLIP_WC42/` — diretório NOVO.
`run_B2_bio_viewer_seed42_CLIP/` (baseline desta comparação) **não é sobrescrito**.
`meta.json` passa a gravar `loss_w_a`/`loss_w_b`/`loss_w_c` — a identidade desta run *é*
a mudança de peso, e até agora nada registrava os pesos no artefato.

⚠ **A `eval_loss` desta run não é comparável com a de nenhuma run anterior** — o termo de
C entra 6× maior. `min_eval_loss_epoch` continua sendo gravado como registro histórico,
mas comparar `eval_loss` entre runs deixa de ser válido aqui. Só as 6 métricas do PyEvALL
comparam.

⚠ **É REALOCAÇÃO, não ganho de graça** — subir C deve custar A e/ou B. O relatório desta
run tem de trazer as seis métricas contra a Run 4 (CLIP: A 0,5741/0,5172 · B 0,4986/0,4122
· C 0,3640/0,1368) e contra o B15 (A 0,5536/0,5058 · B 0,4968/0,3947 · C 0,3436/0,1514),
nunca só a que melhorou.

#### Resultado da Run 5 `[VERIFICADO]` — 2026-08-15, executada na VM

Treino completo, 15 épocas, 6 271 s (o baseline `CLIP` levou 6 220 s — mesmo custo).
`meta.json` confirma `loss_w_c: 4.2`, `epoch_selection: "final"`,
`unfrozen_vision_layers: 4`, `min_eval_loss_epoch: 2` (registro; nenhum reload).

Estado FINAL da época 15 nos dois lados, as seis métricas:

| métrica | `CLIP` (baseline) | **`CLIP_WC42`** | Δ | `B15` |
|---|---|---|---|---|
| A ICMNorm hard | 0,5741 | **0,5353** | **−0,0388** | 0,5536 |
| A ICMSoftNorm soft | 0,5172 | **0,5163** | −0,0009 | 0,5058 |
| B ICMNorm hard | 0,4986 | **0,4733** | **−0,0253** | 0,4968 |
| B ICMSoftNorm soft | 0,4122 | **0,4124** | +0,0002 | 0,3947 |
| C ICMNorm hard | 0,3640 | **0,3318** | **−0,0322** | 0,3436 |
| C ICMSoftNorm soft | 0,1368 | **0,1456** | **+0,0088** | 0,1514 |

**C hard com limiares ajustados** (`task_c_thresholds.json`, metade de ajuste → metade de
reporte, mesmo protocolo dos dois lados): `CLIP` 0,3697 → `WC42` 0,3322 = **−0,0376**.
A queda de C hard **não** é artefato de calibração de limiar — sobrevive ao tuning.

**Veredito: opção 5 REFUTADA por medição.** A realocação aconteceu, mas paga A hard,
B hard **e C hard** por +0,0088 em C soft — o único ganho, de uma seed só (dentro da
variância de seed já documentada no projeto para deltas dessa ordem), e ainda **abaixo do
B15** (0,1514). Não é o trade-off que a opção prevê ("subir C deve custar A e/ou B"):
a própria Task C piora na métrica hard.

**Consequência para a bifurcação do doc 20 (é o achado que vale para a tese):**
multiplicar o gradiente efetivo da Task C por 6 — a correção de causa do ramo
"C sub-treinada dentro do treino conjunto" — **não fez a Task C render mais**. Isso é
evidência direta contra esse ramo e a favor do outro: *a representação do trunk não
carrega mais sinal de C a ser extraído por mais peso de loss*. É a mesma conclusão que o
probing de heads tentou alcançar e não conseguiu (doc 21, INCONCLUSIVO por falha de
controle) — aqui ela vem de uma intervenção no treino, sem o problema de protocolo do
probing. ⚠ **Uma seed, sem replicação** — reportar como evidência convergente, não como
prova.

**Artefato:** `run_B2_bio_viewer_seed42_CLIP_WC42/` completo e válido.
`run_B2_bio_viewer_seed42_CLIP/` intacto — continua sendo o braço CLIP oficial reportado.

**Não tentar de novo:** repesagem de `LOSS_W_C` em qualquer valor. O valor principiado
(4,2 = desfaz a assimetria `.mean` vs `.sum`) foi testado; procurar outro valor por grid
seria afinar hiperparâmetro contra a métrica reportada — leakage, proibido pela regra do
projeto — e a direção do efeito já é conhecida.

### 9.6 Run 6/7 — fase 2 com heads A/B congeladas (hipótese 4), 2026-08-16 `[VERIFICADO]`

Pré-registro e resultado completos em `22_HEAD_FREEZE_PREREGISTRATION.md`. Resumo:
duas runs de 5 épocas a partir do checkpoint `CLIP` (fase 1), únicas diferentes entre si
pelo freeze de `head_a`/`head_b` — `run_..._CLIP_FT5_FREEZEAB` e `run_..._CLIP_FT5_CTRL`,
este último o controle de orçamento sem o qual "freeze ajudou" seria inseparável de
"5 épocas extras ajudaram".

**Δ do freeze (FREEZE−CTRL):** A hard +0,0032 · A soft +0,0093 · B hard +0,0155 ·
B soft +0,0162 · C hard −0,0012 (**−0,0264** com limiares ajustados) · C soft +0,0122.

**Veredito: hipótese 4 não sustentada.** O freeze preserva A/B (quase mecânico) e **não**
move C — os dois sinais de C se contradizem. Terceira via independente confirmando o
§9.5: dar 6× mais gradiente a C e tirar a competição de A/B sobre o trunk são mecanismos
opostos, e nenhum moveu C. **O gargalo da Task C é a representação do trunk, não a
alocação de treino entre tasks.** Linha de investigação de C encerrada.

**Colateral:** os dois braços de fase 2 ficam abaixo da fase 1 em C hard tuned e em A —
treinar além de 15 épocas piora, com ou sem freeze. O orçamento de 15 não é curto.

⚠ n=1 seed; deltas na ordem da variância de seed. O que sustenta a conclusão é a
convergência com o §9.5, não a magnitude.

**Artefatos oficiais do braço CLIP permanecem os do §9.1** (`run_..._CLIP`, época 15).
As runs de fase 2 são braços de atribuição, não substituem nada.

**Pendente (opcional):** re-rodar `probe_video_contribution_clip.ipynb` contra o novo
`RUN_DIR` (o `CACHE_DIR` auto-deriva do nome da run) para medir `‖Δ_texto‖`/`‖Δ_vídeo‖`
sob a loss repesada. Não muda o veredito acima — é caracterização adicional do mecanismo,
não critério de decisão.
