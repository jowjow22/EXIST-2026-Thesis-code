# CLAUDE.md — Jow @ EXIST 2026 (LeWiDi Multimodal Sexism Detection)

Este ficheiro fornece contexto de projeto ao Claude Code. Deve ficar na raiz do repositório.

## Contexto & Propósito

Jonathan Santos da Silva — Mestrado em Inteligência Artificial e Ciência de Dados na
Escola Superior de Tecnologia e Gestão, Instituto Politécnico de Bragança (IPB), Portugal,
em regime de dupla titulação com a Universidade Tecnológica Federal do Paraná (UTFPR), Brasil.

- **Orientadores:** Rui Pedro Lopes e Ahmad Gamal Ali Ali Ibrahim (ambos no IPB)
- **Tese:** deteção multimodal de sexismo sob o paradigma *Learning with Disagreement* (LeWiDi)
- **Competição:** EXIST 2026 (CLEF 2026), equipa **Jow**
- **Paper:** working notes CEUR-WS, classe LaTeX `ceurart`
- **Título atual:** *"Jow at EXIST 2026: Lightweight Multimodal Sexism Detection Across
  Memes and Videos with Demographic-Aware Soft-Label Learning"*

## Tarefas da Competição

- **Task 2** — deteção de sexismo baseada em memes
- **Task 3** — deteção de sexismo baseada em vídeos do TikTok

Cada tarefa tem 3 subtasks:
- **A** — deteção binária de sexismo
- **B** — classificação da intenção da fonte (NONE / JUDGEMENTAL / DIRECT)
- **C** — classificação multilabel de categorias finas (6 categorias)

**Avaliação:** ICM-Norm (variantes soft e hard) via PyEvALL, sob o paradigma LeWiDi.

## Estado Atual

Ambos os sistemas (Task 2 e Task 3) foram construídos, extensivamente depurados e submetidos.
O paper CEUR-WS foi redigido, corrigido iterativamente e finalizado em LaTeX (`ceurart`),
passando por:
- Auditoria completa de citações (2 referências fabricadas corrigidas, 1 nome de autor errado corrigido)
- Diagramas de arquitetura em TikZ para os dois modelos
- Ronda de revisão obrigatória endereçada (correções na lista de autores, título revisto)

**Próximos passos:**
- Revisões finais do paper conforme feedback adicional dos revisores
- Redação da tese, incorporando todo o arco experimental (estudos de ablação,
  validação cross-seed, decisões arquiteturais e respetiva justificação)
- **Re-extração de features da Task 3 (áudio + frames)** — spec em
  `improvements/10_EXTRACTION_FIXES_SPEC.md`, plano executável com código em
  `improvements/11_EXTRACTION_IMPLEMENTATION_PLAN.md` (§13 tem as decisões do autor).
  Decidido 2026-08-09: **64 frames `uint8`** (mesmo disco que 16 float32 hoje),
  árvores `v2/`, arquivos novos `_v2` com os originais intocados. ~45 min de GPU de
  extração + ~3,5 h de re-treino.
  **Atribuição em 3 braços** — o `restore_biases` de `build_model` permite isolar o
  ARCH-01 pelo lado do modelo, sem extração extra:
  `A′` (v2 frames + v1 áudio + `restore_biases=False`) → `A` (idem, True) → `B` (v2+v2).
  `A−A′` = ARCH-01; `B−A` = áudio; `B−baseline` = total. Uma calibração de 15 épocas
  congela `E=argmin(eval_loss)` e **os três braços rodam com esse mesmo E**.
  ⚠ Separar *extração de frames* de *demais correções de código* é **irrecuperável** —
  os frames v1 foram apagados (não regeneráveis, problema V6). Reportar como bloco único.
- ~~`openai-whisper` não tem versão registrada~~ — **colhido na VM em 2026-08-09:
  `20250625`.** Ver tabela de versões. Também colhido: `opencv` **4.13.0**.
  ⚠ **Nenhum dos dois estava registrado quando a extração v1 rodou (maio/2026)** — não há
  como saber se eram estes. É a razão de o teste T4a (drift de ambiente) existir no plano
  de re-extração: ele mede a divergência contra `frames_v1_sample/` sem poder atribuí-la.
- ⚠ **`VideoMAEImageProcessorFast` está depreciado em `transformers==5.6.0`**
  (observado no P0, 2026-08-09): *"The `Fast` suffix for image processors has been
  removed; use `VideoMAEImageProcessor` instead."* Funciona, como o `warmup_ratio` —
  **depreciado ≠ removido**. Usar o nome novo é durabilidade, não correção de bug.
  **CONFIRMADO na VM (2026-08-09): são a MESMA classe** (`is` → `True`), logo o
  resize/crop da v2 é idêntico ao da v1 e a decisão V5-NORM se sustenta.
  Parâmetros lidos do checkpoint (vão para o `_extraction_meta.json`, não hardcodar):
  `shortest_edge=224` · `crop_size=224×224` · `mean=(0.485,0.456,0.406)` ·
  `std=(0.229,0.224,0.225)` · `resample=2` (bilinear) · todos os 4 flags `True`.
  MRO: `VideoMAEImageProcessor → TorchvisionBackend → BaseImageProcessor` — o resize é
  feito por **torchvision**, não PIL.
  ⚠ **Risco residual:** com `do_rescale=False`, o resize bilinear pode acontecer sobre
  `uint8` em vez de `float`, e interpolação bilinear arredonda diferente nos dois dtypes.
  É exatamente o que o teste **T4b** do plano 11 mede, com aborto se `maxΔ > 0,017`.

**Baseline da Task 3 CONGELADO (2026-08-08)** — `improvements/09_BASELINE_RESULTS.md`.
Sistema submetido re-treinado no split congelado (`Thesis_video_baseline.ipynb`,
`SEED=42`, 2 019/505, fingerprints reproduzidos em duas runs). Números oficiais
(ICM-Norm hard / ICM-Soft-Norm soft): **A 0,5483 / 0,4972 · B 0,4579 / 0,3713 ·
C 0,3784 / 0,1378**. Task C com limiares ajustados: 0,4267 (mesmo val — regime
otimista, não comparar com meia-val).
- **Definição de baseline escolhida: o estado FINAL da run** (modelo SWA), não a
  melhor época. **Regra de simetria obrigatória:** o sistema corrigido tem de ser
  reportado pelo seu estado final também — nunca fim de um lado e pico do outro.
- **Declarar na atribuição:** o corrigido remove o SWA, então parte do delta vem
  daí. Medido nesta run: SWA é pior que o melhor da curva em **6/6 métricas**.
- **ARCH-06 reproduziu** em split e seed diferentes (A hard 0,5872 @ ép. 6 →
  0,5483 SWA); `argmin(eval_loss)` = época 5, loss sobe monotonicamente depois.
- **Verificação na VM do sistema de vídeo corrigido** — sequência de 15 itens em
  `EXIST_VIDEOS/improvements/05_IMPLEMENTATION_NOTES.md` §5, começando pelo item 0

## Auditoria do Modelo de Vídeo (Task 3) — 2026-08-07/08

Time de 3 agentes (arquitetura / simplificação / otimização) + reconciliação +
implementação + code review. Artefatos em `EXIST_VIDEOS/improvements/` (01–08) e
`EXIST_VIDEOS/analysis/`. Código novo: `video_core.py` + `Thesis_video_improved.ipynb`
(originais intocados).

**Bugs confirmados por evidência direta (não hipóteses):**

- **ARCH-01 (CRÍTICO)** — `transformers==5.6.0` **não mapeia** `q_bias`/`v_bias` de
  `MCG-NJU/videomae-base`: eles chegam como `UNEXPECTED` e `query.bias`/`value.bias`
  como `MISSING` (output da célula 22 do notebook original, 12 camadas). L2 do
  `q_bias` ≈ 12,9/camada contra ≈1,37 das linhas de `query.weight` — não é um detalhe.
  Candidato nº 1 para a performance ruim. Corrigido via `restore_videomae_qv_biases()`,
  chamada **antes** de `get_peft_model`. **MEDIDO NA VM em 2026-08-08** com
  `transformers==5.6.0` — não inferido do LOAD REPORT, medido em runtime:
  ```
  query.weight carregou?              True     <- o prefixo `videomae.` resolve
  norma do q_bias no checkpoint:     17.541    <- o bias treinado existe
  norma do query.bias no modelo:      0.000    <- e vira ZERO ao carregar
  q_bias carregou?                    False
  ```
  O `query.weight` carregar prova que o problema **não é o prefixo** (senão os 86 M de
  params do encoder estariam todos MISSING) — é o **nome**: o checkpoint traz `q_bias`/
  `v_bias` (resquício da projeção `qkv` fundida do VideoMAE original) e o v5 quer
  `query.bias`/`value.bias` das camadas `Linear` separadas. O mapeamento dos pesos foi
  feito na refatoração; o dos biases não.
  Efeito real: nas 12 camadas, um bias treinado de norma ~12,9 (min 10,5 / máx 17,8) é
  **substituído por zeros**. Item pendente nº 1 de `05_IMPLEMENTATION_NOTES.md` §5
  **RESOLVIDO**.
  Detalhe: `key.bias` também vem MISSING, e a função o **zera de propósito** — o VideoMAE
  não tem `k_bias` por desenho (só `q_bias` e `v_bias`), logo zero é o valor correto,
  não um remendo. As chaves do decoder virem UNEXPECTED é normal: o checkpoint é de
  `VideoMAEForPreTraining` e estamos a carregar só o encoder.
- **`num_frames` é livre no VideoMAE (verificado na VM, 2026-08-08).**
  `VideoMAEModel.from_pretrained(..., num_frames=32)` carrega sem incompatibilidade de
  shape e dá `num_patches=3136`. Motivo: `position_embeddings` é uma tabela sinusoidal
  calculada por fórmula (`get_sinusoid_encoding_table`), **não está no `state_dict`**, e
  o `Conv3d` de patches tem kernel `(tubelet=2, 16, 16)` que desliza — shape idêntico
  para qualquer duração. Nada precisa ser interpolado.
  **Mas:** o checkpoint foi pré-treinado com 16 frames, então as posições 1568–3135
  nunca foram vistas em pré-treino. Custo: tokens 1568→3136, atenção **4×**, ~2,5–3× no
  total (época 340 s → ~850–1000 s). Tratar como **braço de ablação isolado**, nunca
  como mudança de baseline, e só depois de o `probe_zero_video` mostrar que o ramo de
  vídeo contribui alguma coisa.
- **SIMP-01** — `ablation_results.csv` é md5-idêntico a `ablation_seed_2025.csv`
  (`718264812c8df36df087a1484c95b6b2`). A tabela de ablação do paper é **uma seed
  apresentada como agregado de três**. `run_ablations_multi_seed` existe e nunca foi
  chamada.
- **ARCH-20** — os braços de ablação **partilhavam os mesmos pesos**, sempre treinados
  com bio+viewer; as flags apenas zeravam a entrada. `B0_content_only` não era um
  modelo content-only, era uma sonda de zeragem de input.
- **SIMP-01b** — `B2_bio_viewer_no_tau` e `FULL_bio_viewer_tau` tinham dicts de flags
  **idênticos**; não existia parâmetro `use_tau` (era engolido por `**kwargs`). Braço
  fantasma nas linhas do CSV.
- **OPT-06** — `SEED` controlava split **e** init **e** ordem dos dados
  (`train_test_split(random_state=SEED)`, linha 467). As "3 seeds" eram **repeated
  holdout sobre 3 conjuntos de validação diferentes**, não seeds pareadas. Importar a
  análise pareada do modelo de memes literalmente estaria errado. Corrigido separando
  `SPLIT_SEED` (fixo, artefato commitado) de `INIT_SEED`.
- **ARCH-05** — `max_length=128` truncava **42,9 %** das transcrições (mediana 105
  tokens, p90 364, máx 1 122). Corrigido para 256 (cobre 78 % integralmente).
- **ARCH-06** — 15 épocas + SWA a partir da 5 produziam um modelo final **pior** que a
  época 4/6 em todas as seis métricas (Task A ICMNorm hard 0,6199 → 0,5647). SWA
  removido; época a escolher por `E = argmin(eval_loss)` — **nunca** pela métrica
  reportada (é leakage).
- **BUG-I** — a `cross_attn` rodava incondicionalmente, então uma modalidade zerada
  pelo modality dropout ainda era misturada e **re-normalizada** por `cross_norm`.
  Ativo em todos os treinos já feitos. Deixou de existir com o ARCH-03b.
- **Modality dropout usava `torch.rand(1)`** — por batch, não por amostra. Corrigido
  para `torch.rand(B)` com faixas posicionais `band = clamp(floor(r/p), max=4)`, que
  mantêm a taxa por modalidade igual em todos os braços.

**ARCH-04 — features de áudio diluídas por padding (medido na população completa,
2026-08-08).** `whisper.pad_or_trim` força uma janela fixa de 30 s e o `.mean(dim=0)`
promedia os 1500 frames do encoder, silêncio incluído. Números (`EXIST_VIDEOS/analysis/`,
n=2 524 treino + 674 teste, `ffprobe -select_streams a:0`):

| | treino | teste |
|---|---|---|
| mediana | 18,19 s | 15,05 s |
| < 30 s (diluídos) | 66,6 % | 76,0 % |
| > 30 s (truncados) | 33,4 % | 24,0 % |
| `valid_fraction` média | **0,655** | **0,582** |
| do vetor `[1280]` que é silêncio | 34,5 % | 41,8 % |

- No treino: 16 104 s (24,5 %) de áudio real **nunca chegam ao encoder**, e 26 087 s de
  silêncio sintético são injetados.
- **Achado que nenhum agente reportou:** treino e teste têm o confound em intensidades
  diferentes (0,655 vs 0,582) → *dataset shift induzido pelo pipeline de extração*.
- As features **não estão corrompidas** (0/2 524 com norma zero, norma média 14,96):
  estão **diluídas**. Corrompido se detecta; diluído passa em toda checagem de sanidade.
- A amostra n=60 do finding original segurou: todos os números da população caem perto,
  nenhum muda a direção. Única correção: `% < 15 s` é **37,6 %**, não 45 %.

**Restrições do `CLAUDE.md` que se revelaram obsoletas ou insatisfazíveis:**

- ~~`warmup_ratio` foi removido em `transformers==5.6.0`~~ — **falso**. Está
  *depreciado mas funcional*; o stderr da célula 33 imprime `warmup_ratio is deprecated
  and will be removed in v5.2`. Migrar para `warmup_steps` é **durabilidade, não bugfix**.
- ~~fusão da Task 3 = 2592~~ — ver seção "Arquitetura & Fusão".
- **`reset_lora_weights()` / `reset_new_layer_weights()` não existem** em lugar nenhum
  do código (REV-01) — nem no notebook de vídeo original, nem no `memes_core.py`, nem no
  `ablation/train_one.py`. Só na documentação. `bf16=True` continua ativo sem reset, como
  sempre esteve. **Decidir** se a mitigação nunca foi implementada ou foi removida.

**Escopo que ultrapassa código (decisão do autor + orientadores, nada foi tocado):**
SIMP-01, ARCH-20 e OPT-06 atingem a tabela de ablação do paper CEUR-WS **já submetido**.
`predictions/cross_att/` e `predictions/swa/` são submissões de um sistema que deixa de
existir e **não podem ser regeneradas** — na tese têm de aparecer como *"sistema
submetido"*, separadas de *"sistema corrigido"*.

**DECISÃO (2026-08-08) — re-extração de frames grava `uint8`, não `float32`** (OPT-01,
opção B). Normalização ImageNet passa a ser aplicada na GPU. 9,6 MB → 2,4 MB por vídeo
(4×). **Só válido dentro de uma re-extração:** converter os `.pt` atuais para uint8
quantizaria em ~0,017 no espaço normalizado, da mesma ordem do ulp do bf16 (~0,016) — não
é round-trip inócuo. Não é ganho de velocidade hoje (o loader sustenta ~71 MB/s sem
esforço); é **pré-requisito** do OPT-04 (batch 8), que dobraria a demanda para ~142 MB/s.

**ACHADO NOVO (2026-08-08) — a amostragem temporal está fora do regime de pré-treino.**
Cruzando as durações medidas (`analysis/audio_durations.csv`, n=2 524) com `NUM_FRAMES=16`:

| n_frames | fps efetivo (mediana) | gap entre frames consecutivos |
|---|---|---|
| 16 (**atual**) | 0,88 | mediana 1,21 s (p10 0,54 · p90 3,85) |
| 32 | 1,76 | mediana 0,59 s |
| 64 | 3,52 | mediana 0,29 s |

Referência do pré-treino do VideoMAE (Kinetics-400, 16 frames, stride 4 @ ~30 fps →
clipe de ~2,1 s): ~7,6 fps, gap ~0,13 s. **O regime atual é ~8,6× mais esparso.**
⚠ os números do pré-treino vêm da config padrão do paper e **não foram confirmados em
fonte primária nesta sessão** — confirmar antes de irem para a tese.

Duas consequências:
1. O `tubelet_size=2` do VideoMAE pareia frames adjacentes para captar movimento de
   curta duração. Com gap mediano de 1,21 s, o tubelet vê **duas imagens praticamente
   sem relação** — a modelação temporal é degenerada por construção.
2. **O gap varia 7,2× dentro do dataset** (p10 0,54 s → p90 3,85 s) porque os 16 frames
   são espalhados pelo vídeo *inteiro* e as durações variam de 3,3 s a 60,6 s. É o
   **mesmo padrão de confound do ARCH-04**, agora no eixo temporal do vídeo: a taxa de
   amostragem é função da duração. Aumentar `n_frames` divide o gap mas **não remove o
   confound** — a razão p90/p10 continua 7,2× para 16, 32 ou 64.

**Dívida aberta — os notebooks de extração ficaram intocados por regra, não por estarem
corretos:** presos ali estão ARCH-04, ARCH-09 (duplo `cvtColor(BGR2RGB)` no fallback de
frames), OPT-01 (fp16/uint8), BUG-D/E (zeros no disco mascarando vídeos faltantes) e
SIMP-04/10/11. Uma re-extração fecha todos de uma vez.

## Arquitetura & Fusão

- Concatenação simples de tokens CLS superou a fusão por cross-attention na Task 2;
  cross-attention colapsou para padrões quase uniformes no dataset pequeno (~5.000 memes)
- **Task 2 — fusão:** 1024-d (duas projeções de 512-d concatenadas, sem projeção adicional)
- ~~**Task 3 — fusão:** 2592→1024-d (768 texto + 768 vídeo + 1024 áudio + 32 bio)~~
  **ERRADO — corrigido 2026-08-07 (SIMP-16).** O código dizia 2688, com bio=128 e os
  rótulos de texto/áudio trocados. No sistema **corrigido** a largura é função da
  condição de ablação: **2688→1024 (B1/B2, com bio); 2560→1024 (B0, sem bio)**.
  O diagrama TikZ e a seção de arquitetura do paper herdaram o 2592 — **verificar**.
- **Task 3 — sem cross-attention no sistema corrigido (ARCH-03b).** A `cross_attn`
  original atendia sobre uma **única** key (o vídeo já colapsado pelo pooling), logo
  softmax sobre 1 elemento = 1,0 — não era atenção, era um mapa linear fixo, e custava
  4.987.904 params = 45 % dos 11,0 M treináveis. O diagrama não pode mais mostrá-la.
- **Nunca normalizar o vetor concatenado** enquanto qualquer braço de ablação zerar um
  slot. A fusão da Task 3 é `Linear→GELU→Dropout` **sem** LayerNorm após o `cat`: um
  slot zerado contribui 0 e recebe gradiente 0. Um LayerNorm ali reintroduziria o
  confound de modalidade ausente do modelo de memes.
- Injeção aditiva de sinais auxiliares à dimensionalidade partilhada total superou
  consistentemente a concatenação de baixa dimensionalidade como componente separado
- Normalização por soma dos vetores one-hot demográficos preservou o sinal;
  normalização L2 destruiu-o
- A rede-τ (temperatura adaptativa) prejudicou consistentemente o ICM-Soft-Norm da Task A
  e foi removida
- Embeddings por avaliador (per-viewer) para normalização biométrica neutralizaram os
  benefícios dos biosinais, provavelmente por confounding de língua (língua do avaliador
  correlacionada com a língua do vídeo)

## Biosinais

- EEG foi descartado devido a taxa de 70,6% de canais mortos
- Features biométricas usam `RobustScaler` (não z-score)
- Condição B1 (fusão de biosinais sem embeddings per-viewer) mostrou deltas positivos
  consistentes em todas as seeds nas métricas Task B soft, Task A hard e Task C hard
- O ganho de +0,020 em Task C soft observado numa seed não replicou — confirmado como
  variância de seed

## Treino & Avaliação

- Grid search de threshold por classe ({0.05, 0.10, …, 0.60}, 72 chamadas PyEvALL)
  melhorou o ICM hard da Task C de 0,3761 para 0,3948
- O melhor threshold deve ser passado através de `_format_for_pyevall`, não aplicado
  externamente (o formatter tem um threshold hardcoded interno)
- Ajuste de threshold deve ser feito apenas pós-treino, para evitar leakage de validação
- ~~`warmup_steps=0.1` é bug~~ — correto para `transformers==5.6.0` da VM (ver seção
  "Ambiente de Execução"). ~~`warmup_ratio` foi removido nessa versão~~ — **falso,
  corrigido 2026-08-07:** está depreciado mas funcional (`warmup_ratio is deprecated and
  will be removed in v5.2`). As duas formas funcionam; `warmup_steps` é a durável
- `memes_model.ipynb` (reescrita legível de `Thesis_main.ipynb`, 2026-08-01) corrigiu 2 bugs
  de dados encontrados na revisão: (1) `wordsegment.load()` nunca era chamado, então a
  segmentação de hashtags por dicionário nunca funcionou — corrigido, muda o texto de
  10/3.984 amostras; (2) apóstrofo curvo (`’`) nos dados vs reto (`'`) no vocabulário de
  escolaridade descartava 64,6% desses votos do vetor demográfico — corrigido via
  `_normalize_apostrophes`, muda o vetor de 3.957/3.984 amostras. **Consequência:**
  `memes_model.ipynb` não reproduz mais o sistema submetido; é um baseline novo, precisa de
  re-treino do zero antes de qualquer comparação de métricas. `Thesis_main.ipynb` e
  `inference.py` foram deixados intocados — atualizar `inference.py` só quando o re-treino
  a partir de `memes_model.ipynb` virar oficial
- Runs de `memes_model.ipynb` eram irreprodutíveis (2026-08-04, corrigido): (1) o
  `MultiHeadModel` era instanciado antes do `Trainer`, então as camadas novas nasciam de um
  RNG nunca semeado (o `set_seed(42)` interno do Trainer vem tarde demais); (2) kernels CUDA
  não-determinísticos (cuDNN benchmark, somas atômicas) amplificavam diferenças ao longo das
  épocas. Correção: seção 2.1 do notebook (`set_full_determinism(SEED)` — seeds
  python/numpy/torch/cuda + `cudnn.benchmark=False` + `use_deterministic_algorithms(True)` +
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`), re-seed imediatamente antes de instanciar o modelo,
  `seed`/`data_seed` explícitos no `TrainingArguments`, e `model_fingerprint()` (SHA-256 dos
  pesos) impresso no init e pós-treino para verificar bit a bit entre runs. Custo ~10% de
  velocidade; reprodutibilidade vale só com Run All na mesma GPU/versões. Verificado na VM:
  fingerprints iniciais e finais idênticos entre duas runs. Mesmo padrão aplicado em
  `Thesis_video_fixed.ipynb` (2026-08-04): lá `set_all_seeds`/`seed`/`data_seed` já existiam,
  faltavam as flags de determinismo CUDA, o re-seed imediatamente antes de cada init de pesos
  (LoRA A é aleatória: VideoMAE, XLM-R, `EXISTVideoModel`) e os fingerprints
- SWA entra em conflito com `load_best_model_at_end` do HF Trainer; os pesos SWA são
  sobrescritos se o carregamento do melhor checkpoint estiver ativo — requer escolha
  explícita de estratégia
- `bf16=True` em `TrainingArguments` causou corrupção recorrente dos pesos LoRA;
  ~~`reset_lora_weights()` e `reset_new_layer_weights()` devem ser chamados
  pós-instanciação~~ — **essas funções não existem em nenhum lugar do repositório**
  (REV-01, verificado 2026-08-07): nem no notebook de vídeo, nem no `memes_core.py`, nem
  no `ablation/train_one.py`. Só nesta documentação. Todos os treinos rodaram com
  `bf16=True` **sem** reset. Decidir se a mitigação nunca existiu ou foi removida
- `compute_metrics` com PyEvALL pode falhar silenciosamente (retornando zeros); métricas
  proxy baseadas em sklearn são recomendadas como fallback de diagnóstico
- Guards para valores em falta são essenciais antes de chamadas `np.stack`, para evitar
  crashes silenciosos

## Anotação & Tratamento de Labels

- Labels UNKNOWN e REPORTED devem ser excluídas dos targets de treino
- O voto do anotador `-` (não-sexista) na Task B deve ser remapeado para "NONE",
  não filtrado — filtrar infla artificialmente as proporções de DIRECT/JUDGEMENTAL
- Listas de labels duplicadas/conflituantes com ordens de índice diferentes corrompem
  silenciosamente as distribuições soft enviadas ao PyEvALL

## Seleção de Features

- Chi-quadrado é inapropriado para features biométricas contínuas; informação mútua é
  o análogo correto
- Seleção de features deve usar apenas estatísticas do conjunto de treino, para evitar leakage

## Abordagem & Padrões de Trabalho

- Workflow iterativo e diagnóstico: partilhar código → correr diagnósticos → devolver
  output → interpretar e corrigir → correr novamente
- Debugging camada a camada: isolar componente, imprimir flags de NaN e ranges de valores,
  identificar elemento corrompido, aplicar correção pontual, verificar
- Preferência por funções de substituição completas em vez de diffs; decisões de design
  devem ser explicadas antes de escrever código
- Corridas com múltiplas seeds independentes (ex.: 42, 7, 2025) para validar achados de
  ablação e distinguir ganhos reais de variância de seed
- Tabelas de ablação cross-seed usadas para confirmar afirmações publicáveis
- Framing do paper: contributo positivo (fusão de biosinais ajuda tarefas finas) +
  contributo negativo/analítico (normalização per-viewer falha e porquê)

## Ferramentas & Stack

- **Frameworks:** HuggingFace Transformers, PEFT/LoRA, HuggingFace Trainer, PyEvALL, PyTorch
- **Modelos:**
  - Texto: XLM-RoBERTa-large
  - Visão (Task 2): CLIP ViT-Large/14
  - Vídeo (Task 3): VideoMAE-base com LoRA r=16, α=32
  - Áudio (Task 3): Whisper-large-v3 (pré-extraído offline, 1280→768 projetado)
  - Bio-MLP: eye-tracking + heart rate, `RobustScaler`, input 17-d → output 32-d
    (inclui CVPC — cosine similarity inter-avaliador)
- **Paper:** LaTeX com `ceurart.cls`, `pdflatex`, TikZ para diagramas de arquitetura,
  BibTeX (`references.bib`)
- **Ambiente de desenvolvimento:** Jupyter notebooks (`.ipynb`), Google Colab
- **Avaliação:** PyEvALL (ICM-Norm soft e hard), métricas sklearn como proxy de diagnóstico

## Ambiente de Execução

- Notebooks de treino (`Thesis_main.ipynb`, `memes_model.ipynb`, equivalentes da Task 3)
  rodam numa **VM separada** (Jupyter remoto, `/home/a67606/jupyter-env/`), não na máquina
  local de desenvolvimento — as versões de biblioteca podem divergir entre ambientes
- **Nunca assumir** que a versão de uma lib instalada localmente reflete o ambiente de
  treino real; decisões que dependem de versão de API devem ser confirmadas rodando
  `!pip show <lib>` na própria VM antes de decidir
- `transformers` na VM: **5.6.0** (confirmado em 2026-08-01). Nesta versão,
  `warmup_steps` aceita float `[0,1)` como ratio, logo `warmup_steps=0.1` **é a sintaxe
  correta para v5**, não o bug que se pensava sob a semântica de v4.x.
  **Correção 2026-08-07:** a versão anterior desta entrada dizia que `warmup_ratio` "foi
  removido" — **é falso**. Evidência empírica: o stderr da célula 33 de
  `Thesis_video_fixed.ipynb` imprime `[transformers] warmup_ratio is deprecated and will
  be removed in v5.2. Use 'warmup_steps' instead.` Depreciado ≠ removido; o treino de
  vídeo rodou com ele. Migrar é durabilidade, não correção de bug
- Versões das demais libs na VM (confirmado em 2026-08-01):

  | Lib | Versão |
  |---|---|
  | `torch` | 2.6.0+cu124 |
  | `torchvision` | 0.21.0+cu124 |
  | `transformers` | 5.6.0 |
  | `accelerate` | 1.13.0 |
  | `peft` | 0.19.1 |
  | `safetensors` | 0.7.0 |
  | `huggingface_hub` | 1.11.0 |
  | `numpy` | 1.26.4 |
  | `pandas` | 2.2.3 |
  | `scikit-learn` | 1.8.0 |
  | `scipy` | 1.17.1 |
  | `PyEvALL` | 0.1.78 |
  | `ftfy` | 6.3.1 |
  | `regex` | 2026.4.4 |
  | `tqdm` | 4.67.3 |
  | `Pillow` | 12.1.1 |
  | `emoji` | 2.15.0 |
  | `wordsegment` | 1.3.1 |
  | `openai-whisper` | 20250625 *(colhido 2026-08-09)* |
  | `opencv` (`cv2`) | 4.13.0 *(colhido 2026-08-09)* |

  Comando usado para coletar (rodar de novo na VM se precisar revalidar):
  ```bash
  pip freeze | grep -Ei '^(torch|torchvision|transformers|accelerate|peft|safetensors|huggingface[_-]hub|emoji|wordsegment|pandas|scikit-learn|scipy|PyEvALL|ftfy|regex|tqdm|pillow|numpy)=='
  ```

## Convenções para o Claude Code

- Ao editar notebooks/scripts de treino, preferir funções de substituição completas
- Nunca aplicar ajuste de threshold antes do fim do treino (evitar leakage)
- Em `transformers==5.6.0` (versão da VM), usar `warmup_steps` com float `[0,1)` como
  ratio (ex.: `warmup_steps=0.1` = 10% de warmup). `warmup_ratio` **também funciona**
  (depreciado, remoção prevista para v5.2) — não tratar como bug
- ~~Ao mexer em LoRA + bf16, confirmar chamada de `reset_lora_weights()` /
  `reset_new_layer_weights()`~~ — essas funções não existem no repositório (ver
  "Treino & Avaliação")
- **Nunca normalizar o vetor concatenado da fusão** enquanto qualquer braço de ablação
  zerar um slot
- Flags de ablação vão no **construtor** do modelo, não no `forward` — braços que
  partilham pesos não são ablações (ARCH-20)
- Ao usar múltiplas seeds, separar `SPLIT_SEED` (fixo) de `INIT_SEED` — uma seed única
  para split+init transforma "3 seeds" em repeated holdout (OPT-06)
- Ao re-extrair features, escrever em **árvore versionada** (`v2/`) — os scripts são
  resume-safe e re-rodar sobre a árvore antiga não produz nada (OPT-12)
- Ao tocar em normalização de features biométricas, usar `RobustScaler`
- Ao tocar em seleção de features biométricas, usar informação mútua (não chi-quadrado)
- Manter separação estrita entre estatísticas de treino e validação/teste (anti-leakage)