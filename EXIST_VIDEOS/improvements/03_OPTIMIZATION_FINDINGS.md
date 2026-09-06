# 03 — Auditoria de Otimização (Agente 3)

**Escopo:** processo — arquitetura de treino, pipeline de dados, uso de recursos
computacionais, custo/wall-clock e protocolo experimental.
**Interlocutores:** `01_ARCHITECTURE_FINDINGS.md` (Agente 1) e
`02_SIMPLIFICATION_FINDINGS.md` (Agente 2). Este documento **reage** a eles; não
repete a análise deles.

**Fontes dos números.** Tudo abaixo vem de artefatos deste repositório. As
principais:

| Medida | Valor | Fonte |
|---|---|---|
| `train_runtime` (15 épocas, SEED=2025) | **3 267,33 s** (54 min 27 s) | output da célula 52, `TrainOutput(...)` |
| Passos de otimização | 1 065 | mesma linha |
| Throughput | 10,426 amostras/s | mesma linha |
| Tempo por época (independente) | **217,6 s** | intervalo entre inícios de blocos de log PyEvALL na célula 52 (18:25:55,919 → 18:29:33,522) |
| PyEvALL por avaliação de época | **≈ 2,3 s** (35 linhas de log) | 15 clusters de log na célula 52 |
| Busca de limiares Task C (73 chamadas) | **22,9 s**, uma vez | célula 57, span dos 473 log lines |
| Train / Val | 2 271 / 253 | output da célula 19 |
| Params totais / treináveis | 657 144 827 / **11 017 979** | output da célula 51 |
| Tamanho de um tensor de frames | **9 634 903 B** (9,19 MiB) | `ls -l frames/test/320001.pt` |
| Nº de tensores de frames | 2 524 (train) + 674 (test) | `ls | wc -l` |
| Extração de áudio | **7,50 it/s** (674 vídeos em 1 min 29 s) | tqdm salvo em `extract_audio.ipynb` |
| Extração de frames | **1,54 it/s** (674 vídeos em 7 min 17 s) | tqdm salvo em `extract_video.ipynb` |

Onde estimo, mostro a conta. Onde não consigo estimar, digo que não consigo.

---

## Sumário executivo

**1. O custo nunca foi o problema. A tabela de ablação defensável que o ARCH-20
pede custa ~4–5 h de GPU.** A aritmética: 217,6 s/época × 6 épocas = 21,8 min de
treino puro por run; × 9 runs (3 condições × 3 seeds) ≈ **3,3 h**, ≈ 4,1 h com o
overhead por run (carregar backbones do Hub, salvar checkpoint, busca de limiar).
**Ressalva honesta:** isso pressupõe que o ganho não-verificado do OPT-04 absorva
os +~22 % que o `max_length=256` do ARCH-05 acrescenta. Se o OPT-04 não entregar
nada, a época vai a ~265 s e as 9 runs custam **~5 h**. Somando uma run de
calibração de 15 épocas e um braço extra para a cross-attention, o orçamento
completo fica em **~6–8 h de GPU**. O ARCH-20 e o
SIMP-01 foram tratados como caros e nunca foram; foram um problema de desenho de
protocolo travestido de problema de custo.

**2. Nenhuma das re-extrações que o Agente 1 pede é cara.** Áudio: 7,50 it/s
medidos → **3 198 vídeos em ~7 min** (~10 min com as janelas múltiplas do
ARCH-04). Frames: 1,54 it/s → **~35 min**. O ARCH-04 classifica a re-extração de
áudio como custo alto ("exige re-extrair todas as features"); **discordo do
dimensionamento** — a extração é trivial. O que é caro no ARCH-04 é a
*invalidação*: todo checkpoint, toda métrica e todo JSON de submissão existentes
morrem. Isso é uma decisão de sequenciamento, não de orçamento.

**3. PyEvALL não é o gargalo — confirmado com medição, não por analogia.** 2,3 s
de PyEvALL por época contra 217,6 s de época = **1,1 % do wall-clock**. A busca
de limiares inteira são 22,9 s, uma vez. Reduzir o grid de 72 → 60 chamadas
(proposta do ARCH-10) economiza ~4 s: faça pela corretude, **nunca cite custo
como motivo**. O precedente do modelo de memes se repete aqui.

**4. Onde o tempo realmente vai: forward/backward, e o pipeline de frames é o
teto escondido.** Cada época lê **24,3 GB** de tensores de frames (2 524
amostras × 9,63 MB, float32) — 111 MB/s sustentados só de frames, em paralelo
com a GPU. Hoje o loader acompanha a bs=2; **no momento em que se aumentar o
batch, ele vira o gargalo** (OPT-01/OPT-04).

**5. Alocação de capacidade treinável está invertida.** Dos 11,0 M parâmetros
treináveis, **≈ 45 % estão no bloco `cross_attn` que o ARCH-03 prova ser um mapa
linear degenerado**, ≈ 25 % na `Linear` de fusão, e apenas **≈ 20 % em LoRA** —
isto é, só um quinto do orçamento treinável realmente adapta os encoders
pré-treinados, num dataset de 2 271 amostras. É o argumento decisivo a favor da
opção (b) do ARCH-03.

**6. Achado próprio de protocolo, o mais consequente deste documento: `SEED`
controla simultaneamente o split, a init dos pesos e a ordem dos dados**
(célula 16, `train_test_split(..., random_state=SEED)`). As três "seeds"
existentes **não são seeds pareadas** como no precedente dos memes — são
**holdout repetido** sobre três conjuntos de validação diferentes. Importar a
análise pareada do modelo de memes literalmente estaria errado. (OPT-06/OPT-07.)

---

## Diálogo com os findings de arquitetura

Custo é em GPU/wall-clock medido ou estimado com a base indicada; "0" significa
que não consome GPU (edição de código, decisão de protocolo).

| ID | Posição | Justificativa de processo / custo | Custo da correção |
|---|---|---|---|
| **ARCH-01** q_bias VideoMAE | **CONCORDO COM RESSALVA** | Correção certa e é o candidato nº 1 dele com razão. Três ressalvas de processo: (a) antes de remendar biases à mão, testar se outra versão de `transformers` mapeia certo — é mais barato e mais durável que um patch manual que fica órfão; (b) os biases restaurados são **frozen** e portanto **não** entram no conjunto de params treináveis — se se adotar o salvamento enxuto (OPT-03), eles têm de ser restaurados no *load*, senão o modelo servido difere do treinado, silenciosamente; (c) invalida 100 % dos pesos e métricas existentes. | Código ≈ 0. Consequência: entra na onda única de re-treino (Fase 2). |
| **ARCH-02** `model_accepts_loss_kwargs` | **CONCORDO** | Refutação bem feita e o registro da não-hipótese vale mais que o patch. O patch defensivo é gratuito e elimina uma dependência de acidente. | 0 |
| **ARCH-03** `cross_attn` degenerada | **CONCORDO — opção (b), e (a) deve ser rejeitada por custo** | Concordo com o diagnóstico. Reforço a opção (b) com dois argumentos que ele não tem: (1) o bloco consome **≈ 45 % dos params treináveis** (OPT-02) num dataset de 2 271 amostras; (2) a opção (a) exige manter `vid_out` `[B, 1568, 768]` vivo, o que **fecha a porta** ao aumento de batch do OPT-04 e obriga a manter gradient checkpointing. As duas propostas colidem de frente — ver "Conflitos". | (b): 0 GPU, libera ~2,6 GB de ativação e 45 % de capacidade. (a): +memória, −throughput, mais 3 runs de ablação (~1,4 h). |
| **ARCH-04** áudio: padding de 30 s | **CONCORDO NO DEFEITO, DISCORDO DO CUSTO** | O defeito (diluição variável com a duração = variável de confusão) é o achado de dados mais forte do relatório dele. Mas o texto diz "Custo: exige re-extrair todas as features de áudio (2 524 + 674) e re-treinar", enquadrando isso como caro. **Medido: 7,50 it/s → 3 198 vídeos em ~7 min**; com ~1,4 janelas/vídeo em média (duração média 25,3 s medida por ele), ~10 min. É desprezível. O custo real é a invalidação, não a extração. | **~10 min de GPU** para extrair. Re-treino: entra na onda da Fase 2. |
| **ARCH-05** `max_length=128` | **CONCORDO COM RESSALVA — 256, não 384** | Truncamento de 42,9 % é grave. Ressalva de custo, com a conta: contando tokens × camadas × hidden² como proxy de FLOPs, VideoMAE (12 × 1568 × 768²) ≈ 1,11e10 contra XLM-R@128 (24 × 128 × 1024²) ≈ 3,2e9. Ir para 256 acrescenta ~3,2e9 = **+~22 % de compute por passo**; 384 acrescenta ~+49 %. 256 cobre 78 % das amostras integralmente e é o melhor ponto da curva. Nota: o cache de embeddings de texto do OPT-09 **exige que este número seja decidido antes**. | +~22 % no tempo de época (≈ +48 s/época). Compensado com folga pelo OPT-04. |
| **ARCH-06** 15 épocas + SWA | **CONCORDO, COM REFINO IMPORTANTE** | A evidência é sólida e é o maior ganho de graça. **Mas o "6 épocas" não pode ser transportado**: a curva citada é da run SEED=2025 do sistema **pré-correção**. Depois de restaurar `q_bias` (ARCH-01), re-extrair o áudio (ARCH-04), remover a cross-attn (ARCH-03) e dobrar `max_length` (ARCH-05), o ponto de overfitting **se move** — pode ser antes ou depois. É obrigatório uma **run de calibração de 15 épocas pós-correção** para reler a curva. Refino adicional: fixar o orçamento de épocas **numa seed só** e congelá-lo para todas as demais runs e condições, e declarar isso na tese como hiperparâmetro escolhido no dev set (ele já alerta para isso — concordo e reforço que a mitigação é congelar, não repetir a escolha por run). | Calibração: 1 run × 15 épocas = **~55 min + overhead ≈ 1 h**. Depois: −60 % em todas as runs subsequentes (54,5 → 21,8 min). |
| **ARCH-07** modality dropout | **CONCORDO — e é PRÉ-REQUISITO do OPT-04** | Os 3 defeitos estão certos. O ponto de processo que falta: o sorteio único por *forward* significa que **quanto maior o micro-batch, pior o regularizador** — com bs=8, um sorteio derruba a modalidade de 8 amostras de uma vez. Portanto o OPT-04 (aumentar o batch) **não pode entrar antes** da máscara por amostra. Dependência dura de sequenciamento. | 0 GPU. Bloqueia OPT-04 até ser aplicado. |
| **ARCH-08** `bio_ln` | **CONCORDO EM REMOVER, DISCORDO DO `BatchNorm1d`** | O diagnóstico (LayerNorm por amostra desfaz o `RobustScaler` por feature) está certo. Mas ele oferece `BatchNorm1d` como primeira opção e "remover" como a mais segura — **a primeira opção deve sair da mesa**: BatchNorm introduz dependência do batch size, o que colide com o OPT-04, reintroduz `update_bn` no SWA (que quero remover) e torna treino/inferência numericamente divergentes. Remover o LN é a única proposta compatível com o resto do plano. Os itens 2 e 3 (cvpc fora do LN; `rt_norm`/`fix_norm` mortos) — concordo, e concordo com "remover, não conectar". | 0 |
| **ARCH-09** fallbacks da extração de frames | **CONCORDO, COM PREVALÊNCIA MEDIDA** | Ele marcou a prevalência como "hipótese não verificada". **Consegui quantificar parcialmente**, dos outputs salvos do `extract_video.ipynb` (split de test, 674 vídeos): 368 linhas de erro `[mov,mp4] ... partial file` em **29 offsets distintos** e 208 linhas `[h264] Invalid NAL unit size / Error splitting the input into NAL units` em **9 padrões distintos** — ou seja, da ordem de **1–4 % dos vídeos do split de test são arquivos truncados** cujo `cap.read()` falha, disparando exatamente o fallback com dupla conversão BGR2RGB. E como a truncagem afeta o fim do arquivo, os frames corrompidos são sistematicamente os **últimos** da sequência. Ao mesmo tempo, `Done. Skipped 0 videos.` confirma que o caminho catastrófico (vídeo inteiro em zeros) **não** foi disparado nesse split. Ou seja: defeito real, prevalência baixa, custo de correção baixo → **vale corrigir junto com a mudança de formato de armazenamento (OPT-01)**, não isoladamente. | Re-extração: **~35 min** (1,54 it/s × 3 198). |
| **ARCH-10** Task C hard "NO"+categoria | **CONCORDO NA CORREÇÃO, DISCORDO DO ARGUMENTO DE CUSTO** | A correção semântica está certa e o aviso de comparabilidade (muda também o gold) é o ponto mais importante do finding. Mas a frase "reduz de 72 para 60 chamadas ao PyEvALL" não deve aparecer como benefício: **72 chamadas custam 22,9 s medidos**; 60 custam ~19 s. A economia é de 4 segundos. Excluir `NONE` do grid deve ser justificado **só** por eliminar um grau de liberdade contraditório. | 0 |
| **ARCH-11** limiares no mesmo val de 253 | **REFINO** | Ele lê a variação entre seeds (`STEREOTYPING-DOMINANCE` 0,05 / 0,20 / 0,25) como sobreajuste a ruído. **Isso é parcialmente confundido**: como `SEED` também define o split (OPT-06), os três limiares foram ajustados em **três conjuntos de validação diferentes**, com prevalências de classe diferentes. Parte da variação é variação de amostra, não instabilidade do procedimento — e a distinção importa para o que se escreve na tese. Das opções dele, a (b) (dividir o val em duas metades: ajustar numa, reportar na outra) é a única que resolve o vazamento e custa 0 GPU; a (c) (média entre seeds) mistura splits distintos e eu a descartaria. | 0 GPU (é pós-processamento sobre predições já geradas). |
| **ARCH-12** `RUN_DIR` único | **CONCORDO, E ESTENDO** | Concordo integralmente (é o mesmo bug do SIMP-02). Estendo: enquanto se mexe nisso, mudar **o que** se salva — hoje `model.safetensors` tem 657 M params ≈ 2,6 GB em fp32, dos quais 646 M são backbones reconstrutíveis via `from_pretrained`. Salvar só o treinável + adaptadores LoRA = ~44 MB (**60× menor**). Ver OPT-03; atenção à interação com ARCH-01 descrita acima. | 0 GPU; economiza ~2,6 GB por (condição × seed) — ~23 GB na Fase 4. |
| **ARCH-13** B2 ≡ FULL | **CONCORDO** | Fato verificado por ele e independentemente pelo Agente 2 (linhas bit-idênticas). Nenhuma ressalva. | 0 |
| **ARCH-14** métricas somem em silêncio | **CONCORDO, PRIORIDADE MAIOR QUE "BAIXO"** | Ele classifica como BAIXO. Do ponto de vista de processo, num plano que vai rodar **9–12 runs desatendidas**, uma métrica que desaparece sem erro é exatamente o modo de falha que custa um dia de GPU. Elevaria para MÉDIO e trataria como **pré-requisito da Fase 4**, não como limpeza. | 0 |
| **ARCH-15** `warmup_ratio` não é bug | **CONCORDO** | Fato verificado no output da célula 33. Não tratar como bug. Migrar para `warmup_steps=0.1` é durabilidade — e se for feito, tem de entrar **antes** da run de calibração, nunca no meio de uma série. | 0 |
| **ARCH-16** detalhes do SWA | **CONCORDO — remover** | Concordo com remover, e acrescento o argumento de recursos que ele só menciona de passagem: `AveragedModel` duplica os 657 M params **na GPU durante o treino inteiro** ≈ **2,6 GB em fp32**. Removê-lo não é só simplificação: é o espaço de memória que financia o aumento de batch do OPT-04. Um ganho negativo (ARCH-06) pago com 2,6 GB de VRAM é a pior troca do notebook. | 0 GPU; libera ~2,6 GB. |
| **ARCH-17** perdas/cabeças | **CONCORDO NOS ITENS 1/3/4, REFINO O ITEM 2** | Itens 1 (sentinelas), 3 (código morto) e 4 (decay em bias/LayerNorm) são gratuitos e não têm objeção. Item 2 (`mean` vs `sum` na Task C): ele já marca como hipótese, e concordo que **não é bug**. Refino: é exatamente o tipo de coisa que o loop de triagem com features congeladas (OPT-09) consegue varrer em segundos em vez de queimar runs de LoRA. Não deve virar "fix" nem entrar na tabela de ablação sem triagem prévia. | 0 (via OPT-09). |
| **ARCH-18** `--scaler` obrigatório | **CONCORDO** | Custo zero, elimina vazamento e não-reprodutibilidade. Sem ressalvas. | 0 |
| **ARCH-19** determinismo | **CONCORDO** | O protocolo está correto e a lacuna apontada é real. Acrescento que a persistência automática de fingerprints vira **infraestrutura obrigatória** na Fase 4: com 9–12 runs, a verificação manual não escala. | 0 |
| **ARCH-20** braços de ablação partilham pesos | **CONCORDO, E DIMENSIONO** | Concordo integralmente com o diagnóstico e com a proposta (flags no construtor, um checkpoint por condição × seed). A contribuição que me cabe é o número: ele estima "9 treinos de ~55 min cada; ~22 min com as 6 épocas do ARCH-06". Confirmo a base (3 267,3 s / 15 = 217,8 s por época) e completo com o overhead por run (carga dos backbones do Hub, salvamento, busca de limiar ≈ 5 min): **~27 min/run → 9 runs ≈ 4,1 h**. A recomendação nº 2 dele (renomear as linhas para `probe_zero_bio` etc. enquanto o re-treino não existe) é boa como medida provisória, **mas o re-treino é barato demais para justificar publicar sondas renomeadas**. Recomendo pular direto para o re-treino. **Lacuna a fechar antes de agendar: o finding diz onde as flags devem morar, não o que `use_bio=False` deve construir — ver OPT-13.** | **~4,1–5,0 h** (3 condições × 3 seeds, 6 épocas). |

---

## Findings próprios

### OPT-01 — Frames em float32: 24,3 GB lidos por época; é o teto que trava o OPT-04 (ALTA)

**Local:** `extract_video_CODE.py` (célula 1, `torch.save(tensor, out_path)` após
`process_frames`) e `Thesis_video_fixed_CODE.py` célula 6 (`load_frames`).

**Fato medido:** cada tensor de frames ocupa **9 634 903 B** (`[16, 3, 224, 224]`
float32). Uma época lê 2 271 (treino) + 253 (avaliação) = 2 524 amostras =
**24,3 GB**. As 15 épocas da run de 2025 leram ≈ **365 GB**.

**Enquadramento correto (importante — não é um ganho de tempo hoje):** a
217,6 s/época, os 24,3 GB implicam ~**111 MB/s** sustentados de leitura de frames
em paralelo com a GPU, e o loader com `dataloader_num_workers=2` está dando
conta. **O problema não é o presente; é que o OPT-04 exige ~2× de throughput, e
aí a demanda vai para ~200–220 MB/s** — o loader passa a ser o gargalo e o
aumento de batch não entrega o ganho esperado. Por isso este finding é
**pré-requisito** do OPT-04, não um item independente.

**Proposta — duas variantes, com garantias diferentes (não confundir):**

1. **Conversão fp16 dos tensores existentes** (sem re-decodificar vídeo):
   9,63 MB → **4,82 MB** (2×). Seguro: o range medido pelo Agente 1 é
   [−2,10; 2,64]; nessa magnitude o ulp de fp16 é ~0,002 contra ~0,016 do bf16 em
   que o modelo de facto computa — a precisão armazenada continua estritamente
   mais fina que a da aritmética. `load_frames` faz `.float()` no retorno, então
   o resto do pipeline não muda.
2. **uint8 gravado na re-extração** (só válido dentro da Fase 2):
   → **2 408 448 B** (2,41 MB, 4×), com a normalização ImageNet aplicada na GPU.
   **Não fazer isto por conversão dos `.pt` atuais:** inverter
   `round((x·σ+µ)·255)` quantiza em (1/255)/0,225 ≈ **0,017** no espaço
   normalizado, ou seja, da mesma ordem do ulp de bf16 (~0,016) — não é um
   round-trip inócuo. Como escolha de pré-processamento numa extração nova não há
   referência anterior a preservar, e aí é legítimo.

**Custo:** variante 1: uma passada de conversão (~31 GB lidos + ~15 GB escritos),
minutos. Variante 2: zero adicional se embarcada na re-extração do ARCH-09.

---

### OPT-02 — 45 % da capacidade treinável está num módulo degenerado; só 20 % adapta os encoders (ALTA)

**Local:** célula 28 (definição do modelo), corroborado pelo output da célula 51.

**Orçamento de parâmetros treináveis** (contado a partir das dimensões da célula
5; soma reconcilia com os 11 017 979 da célula 51 a menos de ~50 params):

| Bloco | Params | ≈ % |
|---|---|---|
| `cross_attn` + `cross_proj` + `cross_norm` (**degenerado, ARCH-03**) | 4 987 904 | **≈ 45 %** |
| `fusion` (`Linear` 2688×1024 + bias) | 2 753 536 | ≈ 25 % |
| **LoRA** (VideoMAE 589 824 + XLM-R 1 572 864) | 2 162 688 | **≈ 20 %** |
| `audio_proj` (`Linear` 1280×768) | 983 808 | ≈ 9 % |
| `bio_mlp` + `viewer_emb`/`viewer_proj` + `video_pool` + 3 cabeças | ≈ 130 000 | ≈ 1 % |

**Por que importa:** com 2 271 amostras de treino, ~4 850 params treináveis por
amostra, e **apenas um quinto desse orçamento realmente adaptando os encoders
pré-treinados**. Quase metade está num bloco que o ARCH-03 demonstra ser um mapa
linear fixo. Isto é o argumento de alocação de recursos que decide o ARCH-03 a
favor da opção (b), independentemente de qualquer medida de métrica.

**Proposta:** (i) adotar ARCH-03 (b) → orçamento cai para ~6,0 M, dos quais LoRA
passa a ser ~36 %; (ii) **depois disso**, considerar realocar parte do que sobrou
para adaptação real — subir o rank de LoRA ou incluir `key` e as `dense` de saída
nos `target_modules`. **Hipótese não verificada** e não triável pelo OPT-09 (mexe
em adaptação); vale no máximo um braço de ablação, não uma mudança de baseline.

**Custo:** (i) 0 e negativo em memória; (ii) +1 condição × 3 seeds ≈ 1,4 h.

---

### OPT-03 — Checkpoints de 2,6 GB gravados a cada época e nunca usados (ALTA)

**Local:** célula 33, `save_strategy="epoch"`, `save_total_limit=2`, com
`load_best_model_at_end=False`; célula 53 (o modelo final é salvo à mão, noutro
diretório).

**O que está errado:** o `Trainer` serializa o `nn.Module` inteiro — **657 M
params ≈ 2,6 GB em fp32** — a cada época. Em 15 épocas são ~39 GB escritos. E
**nenhum desses checkpoints é lido**: `load_best_model_at_end=False`, o SWA
substitui os pesos em `on_train_end`, e a célula 53 salva o modelo final por
conta própria. É IO puro sem consumidor.

**Não consigo medir quanto do wall-clock isso representa** — tanto o
`train_runtime` (3 267,3 s) quanto o intervalo entre avaliações (217,6 s) já
incluem o salvamento; não há no notebook nenhum registro que o separe. Registro o
volume (39 GB) e a inutilidade, não uma fração de tempo.

**Proposta:** `save_strategy="no"` (aceitando perder recuperação de falha numa run
de ~22 min), ou manter `save_strategy="epoch", save_total_limit=1` com um
`state_dict` enxuto: **só os params com `requires_grad` + adaptadores LoRA ≈
44 MB** (60× menor), reconstruindo os backbones via `from_pretrained` no load.
**Ressalva dura, ligada ao ARCH-01:** se a correção de `q_bias`/`v_bias` for
aplicada, esses tensores são *frozen* e **não estão** no conjunto treinável — o
load enxuto tem de reaplicar a restauração, senão o modelo servido diverge do
treinado sem nenhum aviso.

**Custo:** 0 GPU. Economiza ~39 GB de escrita por run e ~23 GB de disco na Fase 4.

---

### OPT-04 — `batch_size=2` + gradient checkpointing: configuração de memória herdada, provavelmente desnecessária (ALTA)

**Local:** célula 33 (`per_device_train_batch_size=2`,
`gradient_accumulation_steps=16`, `gradient_checkpointing=True`,
`per_device_eval_batch_size=2`).

**O que está errado:** o batch efetivo é 2 × 16 = 32, o que é razoável. O que não
é razoável é chegar lá com micro-batch 2 **e** gradient checkpointing ligado nos
dois backbones. Checkpointing troca ~25–35 % de compute extra por memória; a
micro-batch 2 desperdiça ocupação da GPU. Somando o que os outros findings
liberam — SWA fora (**~2,6 GB**, ARCH-16) e cross-attn fora (ARCH-03 (b)) — há
folga que hoje não é usada.

**Proposta (medir antes de fixar):** `per_device_train_batch_size=8`,
`gradient_accumulation_steps=4` (batch efetivo **32, inalterado**),
`gradient_checkpointing=False`, `per_device_eval_batch_size=16`.

**Três avisos, todos duros:**
1. **Depende do ARCH-07.** Com sorteio de modality dropout único por forward,
   bs=8 derruba a modalidade de 8 amostras de uma vez. A máscara por amostra tem
   de entrar **antes**.
2. **Depende do OPT-01.** A ~2× de throughput a demanda de leitura vai a
   ~200–220 MB/s.
3. **É `[MUDA]`, não `[EQUIV]`.** Muda o consumo de RNG, a granularidade das
   médias mascaradas por micro-batch (o BUG-G do Agente 2) e portanto os pesos
   finais bit a bit. Tem de ser congelado **antes** da run de calibração e não
   pode mudar no meio de uma série de ablação.

**Ganho:** **hipótese não verificada.** Faixa plausível 1,5–2× (25–35 % de
checkpointing + melhor ocupação), o que levaria a época de 217,6 s para
~110–145 s. Protocolo de medição, na VM, ~15 min: rodar 30 passos com
`max_steps=30` em cada uma das 4 combinações {ckpt on/off} × {bs 2, 8} e comparar
`train_samples_per_second` e `torch.cuda.max_memory_allocated()`.

---

### OPT-05 — DataLoader subdimensionado para amostras de 9,6 MB (MÉDIA)

**Local:** célula 33, `dataloader_num_workers=2`; sem `persistent_workers`, sem
`prefetch_factor` explícito.

Com 2 workers, cada um precisa entregar ~5,2 amostras/s = ~50 MB/s de
desserialização de tensores `torch.load`. Além disso, sem
`dataloader_persistent_workers=True` os workers são recriados a cada época e a
cada avaliação — em 15 épocas são ~30 recriações de pool.

**Proposta:** `dataloader_num_workers=8` (ajustar ao nº de vCPUs da VM — não sei
quantas são), `dataloader_persistent_workers=True`, `dataloader_pin_memory=True`
(já é o default do HF, tornar explícito).

**Classificação:** numericamente `[EQUIV]` — o dataset não tem aleatoriedade
(nenhuma augmentação), e os índices são produzidos pelo sampler no processo
principal, então a ordem dos dados não depende do nº de workers. Verificável pelo
`model_fingerprint()` final.

**Custo:** 0.

---

### OPT-06 — `SEED` confunde split, inicialização e ordem dos dados (ALTA — protocolo)

**Local:** célula 16, `train_test_split(train_df, test_size=0.1,
random_state=SEED, ...)`, combinada com `set_all_seeds(SEED)` (células 22/24/31) e
`seed=SEED, data_seed=SEED` (célula 33).

**O que está errado:** um único `SEED` governa três fontes de variação distintas:
qual é o conjunto de validação, como os pesos novos são inicializados, e a ordem
dos batches. Consequências concretas:

1. **As runs 42 / 7 / 2025 não são seeds pareadas.** São **holdout repetido** —
   três modelos treinados em 2 271 amostras diferentes e avaliados em 253
   amostras diferentes. O precedente do modelo de memes (3 seeds pareadas, teste
   de delta por seed + consistência de sinal) **não se transporta literalmente**:
   lá o split é fixo e só a init varia. Aqui, deltas **dentro** de uma seed
   (entre condições de ablação) continuam pareados e válidos; **valores absolutos
   entre seeds não são comparáveis** e a média entre eles é uma média de
   holdouts, com variância maior do que uma média entre seeds.
2. É a explicação parcial da dispersão de limiares do ARCH-11.
3. Impossibilita separar "o modelo é instável à inicialização" de "o dev set de
   253 amostras é pequeno demais" — duas conclusões de tese completamente
   diferentes.

**Proposta:** separar `SPLIT_SEED` (fixo, ex.: 42, gerando **um** conjunto de
validação usado por todas as runs) de `INIT_SEED` ∈ {42, 7, 2025}. Isso restaura
o pareamento do precedente dos memes e torna a tabela de ablação interpretável.
Se se quiser deliberadamente medir variância de split, isso vira uma **segunda**
dimensão declarada, não um efeito colateral.

**Custo:** 0 GPU (mudança de código). Mas **invalida a comparabilidade dos três
CSVs existentes** com qualquer coisa nova — que a Fase 2 já invalida de todo modo.

---

### OPT-07 — `run_ablations_multi_seed` avalia todas as seeds no mesmo dataset, mas o split é dependente da seed (ALTA — protocolo)

**Local:** `ablation_harness.py` linhas 168–195.

```python
def run_ablations_multi_seed(model_fn, dataset, compute_metrics_fn,
                              seeds=(42, 7, 2025), device="cuda"):
    for seed in seeds:
        model = model_fn(seed)
        results = run_all_ablations(model, dataset, compute_metrics_fn, device)
```

O `dataset` é recebido **uma vez** e reutilizado para todas as seeds, enquanto
`model_fn(seed)` carrega um modelo cuja validação é outra (célula 16). Se esta
função fosse usada, dois terços das seeds seriam avaliados **sobre parte do
próprio conjunto de treino** — vazamento total. Hoje o Agente 2 confirmou que ela
**nunca foi chamada** (`ablation_results.csv` é cópia de `ablation_seed_2025.csv`,
gerado condição a condição). Ou seja: é uma armadilha armada, não um dano
consumado — mas é a função que qualquer pessoa chamaria ao tentar consertar o
SIMP-01.

**Segundo defeito, na mesma função:** a agregação é `np.mean` / `np.std` por
condição — exatamente o padrão nº 3 do brief ("análise não-pareada mascarando
ruído"), o mesmo erro que a auditoria dos memes já encontrou.

**Proposta:** (a) assinatura passa a ser `dataset_fn(seed)`, ou — melhor, se o
OPT-06 for adotado — o split fica fixo e o problema desaparece por construção;
(b) substituir a agregação por **delta pareado por seed + consistência de sinal**:

```
Para cada métrica m e cada par (condição X, baseline B):
    δ_s = m(X, seed s) − m(B, seed s)       para s ∈ {42, 7, 2025}
Reportar: δ_42, δ_7, δ_2025, mediana(δ), e "sinal consistente" = todos os δ com o mesmo sinal.
NÃO reportar "|Δmédia| > std" como critério.
```

**Custo:** 0 GPU.

---

### OPT-08 — Não há caminho único de avaliação: `trainer.evaluate()` e o harness dão números diferentes (ALTA)

**Local:** `ablation_harness.py` `run_inference` (linhas 71–127) vs
`Trainer.prediction_step`; sintoma documentado pelo Agente 2 em BUG-H
(`task_c_icmnorm_hard` = 0,3531 na célula 55 vs 0,34764 em
`ablation_seed_2025.csv`, mesmo modelo, mesma seed).

O harness roda `@torch.no_grad()` sem contexto de autocast e com `batch_size=4`;
o `Trainer` avalia sob `bf16=True` com `batch_size=2`. São dois caminhos de
código independentes produzindo dois números que aparecem lado a lado no paper.

**Proposta de processo (é a raiz, o resto é sintoma):** que exista **um** caminho
de avaliação. Com as flags de ablação viradas argumentos de construtor (ARCH-20),
o harness deixa de precisar de um loop próprio: cada condição é um modelo, e a
avaliação é `trainer.predict(val_dataset)` para todas. O `run_inference` do
harness pode ser apagado. Isso resolve BUG-H, DIV-05 e o risco do
`ablation_collate` (SIMP-09) de uma vez.

**Custo:** 0 GPU; remove código em vez de acrescentar.

---

### OPT-09 — Loop de triagem com backbones congelados: experimentos de fusão a custo ~zero (MÉDIA)

**Ideia:** congelar VideoMAE e XLM-R (sem LoRA), pré-computar **uma vez** os
vetores `vid_feat` (768-d) e `txt_feat` (1024-d) de todas as 2 524 amostras, e
treinar apenas o bloco de fusão + cabeças sobre esses vetores em memória.

**Custo da pré-computação:** uma passada de forward sobre 2 524 amostras. Base do
cálculo: se um passo de treino com checkpointing custa ≈ 4 forwards, então
217,6 s ≈ (2 271 × 4 + 253) × t_fwd → t_fwd ≈ 0,0236 s → **~60 s para o dataset
inteiro**. Depois disso, cada experimento de fusão treina sobre matrizes de
2 271 × (768+1024+768+17) floats — segundos.

**O que isto pode triar:** largura e profundidade da fusão, desenho do ramo bio
(incluindo a remoção do `bio_ln`, ARCH-08), pesos de perda e `mean` vs `sum` na
Task C (ARCH-17.2), estrutura das cabeças, política de limiar da Task C.

**O que isto NÃO pode triar — lista de exclusão explícita, senão vira armadilha:**
qualquer coisa que mexa em adaptação ou nas representações de entrada —
**cross-attention (ARCH-03), `max_length` (ARCH-05), modality dropout (ARCH-07),
rank/módulos de LoRA (OPT-02), restauração de `q_bias` (ARCH-01)**. Com LoRA
ativo as representações mudam durante o treino e o proxy deixa de valer.

**Dependência:** o cache de embeddings de texto exige que `max_length` esteja
decidido — o ARCH-05 **gate** este finding.

**Status:** proposta de infraestrutura. O grau de correlação entre a triagem
congelada e o resultado com LoRA é **hipótese não verificada**; recomendo validar
com um único caso (rodar uma decisão nas duas vias e comparar o sinal) antes de
confiar na triagem.

---

### OPT-13 — Como o braço B0 é *construído* decide se a tabela de ablação é honesta (ALTA — protocolo, bloqueia a Fase 4)

**Local:** célula 28 — `EXISTVideoModel.__init__` (`FUSION_INPUT_DIM = 2688`
fixo) e o ramo `else` do bloco bio no `forward`
(`bio_feat = self._zeros(B, BIO_HIDDEN_DIM, device, ref_dtype)`); proposta do
ARCH-20 de mover `use_bio`/`use_viewer_emb` para o construtor.

**O buraco:** o ARCH-20 diz *onde* as flags devem morar (construtor, não
`forward`) mas **não diz o que `use_bio=False` deve construir**. Hoje, a
semântica herdada é inequívoca e é a errada: a largura da fusão continua 2 688 e
o slot bio recebe um vetor de zeros permanente. Se o re-treino por condição for
feito mantendo essa construção, **o ARCH-20 é corrigido no mecanismo e mantido no
viés**: B0 continua não sendo um modelo content-only — é um modelo com 128
entradas constantes atravessando a `Linear` de fusão partilhada. É exatamente o
achado nº 1 do brief (confound de fusão com modalidade ausente, importado da
auditoria dos memes), que o brief mandou verificar aqui. Verifiquei: está
presente.

**Duas construções possíveis e o que cada uma produz:**

| | Largura da fusão | O que B0 mede |
|---|---|---|
| (i) manter 2 688, slot bio zerado | 2 688 | modelo penalizado por construção: 131 072 pesos de entrada da fusão gastos num vetor constante |
| (ii) construir `2560 → 1024`, sem slot bio | 2 560 | modelo content-only honesto; braços deixam de ser pareados em nº de params |

**Recomendação: (ii), com a diferença de parâmetros declarada.** A conta, no
sistema pós-ARCH-03(b) (total treinável 11 017 979 − 4 987 904 ≈ **6,03 M**):

- B0 dispensa 131 072 pesos de entrada da fusão + `bio_mlp` (≈ 18 944) +
  `viewer_emb`/`viewer_proj` (848) ≈ **150 864 params a menos** → ≈ **5,88 M**.
- B1 dispensa só `viewer_emb`/`viewer_proj` (848) → ≈ **6,03 M**.
- B2 ≈ **6,03 M**.

Uma diferença de ~2,5 % de capacidade entre B0 e B2 é uma nota de rodapé
defensável na tese. "B0 recebeu um vetor de zeros através da projeção partilhada"
é o mesmo erro que a auditoria dos memes já apanhou uma vez — e um revisor que
leia o `CLAUDE.md` do projeto vai reconhecê-lo.

**Ação:** confirmar na VM o que o construtor produz depois da mudança do ARCH-20,
e fixar a opção (ii) por escrito antes de agendar a Fase 4. **Esta decisão é de
protocolo (minha), não de arquitetura** — mas toca código do Agente 1.

**Custo:** 0 GPU. Bloqueia a Fase 4.

---

### OPT-10 — Onde o tempo realmente vai (registro para não se re-investigar) (MÉDIA)

Decomposição do wall-clock da run de 15 épocas (3 267,3 s), com o que é medido e o
que não é:

| Componente | Custo | Como sei |
|---|---|---|
| Forward/backward de treino + avaliação | **≈ 98 %** | resíduo |
| PyEvALL na avaliação de época | **2,3 s/época ≈ 35 s total = 1,1 %** | 15 clusters de log na célula 52 |
| Busca de limiares Task C (73 chamadas) | **22,9 s, uma vez = 0,7 %** | célula 57 |
| Salvamento de checkpoints (39 GB) | **não separável** dos números acima | ver OPT-03 |
| Leitura de frames (24,3 GB/época) | sobreposta à GPU; não separável hoje | ver OPT-01 |

**Conclusão a registrar explicitamente:** o precedente do modelo de memes se
confirma — **a busca de threshold via PyEvALL não é o gargalo**, nem de perto. Não
otimizar PyEvALL. Otimizar o passo de treino (OPT-04) e a alimentação de dados
(OPT-01).

---

### OPT-11 — Val de 253 amostras: opções e o que cada uma custa (MÉDIA — protocolo)

O Agente 1 registra o problema (253 amostras, 6 classes esparsas na Task C) nas
"observações fora do escopo". Dimensionando as saídas:

| Opção | Efeito | Custo de GPU |
|---|---|---|
| Manter 90/10 | status quo; Task C hard estimada sobre punhados de positivos | 0 |
| 85/15 ou 80/20 | val de 379–505 amostras; ~5–10 % menos dados de treino | 0 (mesmo nº de runs, épocas ~10 % mais curtas) |
| Val dividido em duas metades (tuning / report) | resolve o vazamento de limiar do ARCH-11 — **mas só é viável com o 80/20** | 0 |
| k-fold (k=5) | melhor estimativa, mas **5× runs**: 3 condições × 3 inits × 5 folds = 45 runs | ~20 h |

**Dependência que precisa ficar explícita: a opção (b) do ARCH-11 (dividir o val
em metades) NÃO é aplicável ao val atual.** Com 253 amostras, cada metade fica com
~126; sobre 6 classes esparsas da Task C, alguns limiares seriam ajustados sobre
contagens de positivos de um dígito — pior que o problema que o ARCH-11
identifica. Com o 80/20 (505 amostras) as metades ficam com ~252 cada, que é o
tamanho do val inteiro de hoje. **Portanto o 80/20 é pré-requisito do ARCH-11(b),
e ambos entram juntos na Fase 2** — não são opções paralelas.

**Recomendação:** 80/20 com split fixo (OPT-06) + divisão interna do val para o
ajuste de limiar. k-fold só se sobrar orçamento no fim; ~20 h não é proibitivo mas
não é onde o retorno está.

---

### OPT-12 — Os scripts de extração são resume-safe e vão ignorar silenciosamente a correção (MÉDIA — operacional)

**Local:** `extract_video_CODE.py` linha 88 e `extract_audio_CODE.py` linha 87:
`if out_path.exists(): continue`.

Depois de corrigir ARCH-04 e ARCH-09, **re-rodar os scripts não faz nada**: todos
os `.pt` já existem. A Fase 2 produziria zero arquivos novos e o treino seguinte
usaria features velhas — com todas as métricas parecendo simplesmente
"não melhorou". É um modo de falha silencioso que custaria uma onda inteira de
re-treino.

**Proposta:** escrever numa árvore versionada — `frames/v2/train/`,
`audio_features/v2/train/` — em vez de apagar, e gravar o
`_extraction_meta.json` que o Agente 2 propõe no SIMP-04 (com `SPLIT`,
`NUM_FRAMES`, versões de lib, dtype de armazenamento, `n_skipped`). Assim as duas
gerações coexistem e a proveniência de qual treino usou qual extração fica no
disco. Custo em disco: com uint8 (OPT-01), a v2 inteira ocupa ~7,7 GB contra os
~31 GB da v1.

---

## Plano de execução em fases

Custos de GPU derivados de: **217,6 s/época** (medido), **+~5 min de overhead por
run** (carga dos backbones do Hub + salvamento + busca de limiar 22,9 s, estimado
a partir dos timestamps das células 52–57), extração de áudio **7,50 it/s** e de
frames **1,54 it/s** (medidos), 3 198 vídeos no total.

### Fase 0 — Verificação na VM (sem treino) — ~1 h de máquina, ~0 de GPU útil

Nenhuma decisão das fases seguintes deve ser tomada antes disto. Sete checagens:

1. `print([n for n,_ in videomae.named_parameters() if 'layer.0.attention' in n])`
   → caminho de atributo do ARCH-01; **e** verificar se uma versão diferente de
   `transformers` já mapeia `q_bias` corretamente (alternativa mais barata ao
   remendo).
2. `print(trainer.model_accepts_loss_kwargs)` e `print("labels" in train_dataset[0])`
   → ARCH-02.
3. `print([n for n,_ in model.named_buffers()])` → ARCH-16.2.
4. Contar tensores de frames todo-zeros (ARCH-09/BUG-D) — comando no relatório do
   Agente 1.
5. `logits_c.dtype` no harness vs no `prediction_step` → BUG-H / OPT-08.
6. **Benchmark do OPT-04:** `max_steps=30` nas 4 combinações {ckpt on/off} ×
   {bs 2, 8}, comparando `train_samples_per_second` e
   `torch.cuda.max_memory_allocated()`.
7. `nproc` / VRAM disponível → dimensionar OPT-05.

**Invalida:** nada.

### Fase 1 — Correções de custo zero + extração de `video_core.py` — 0 GPU

Tudo que não exige re-treino nem re-extração, e que precisa estar pronto **antes**
de qualquer run cujo número entre no paper:

- ARCH-02, ARCH-12 (+ OPT-03), ARCH-13, ARCH-14 (elevado a pré-requisito),
  ARCH-17.1/3/4, ARCH-18, ARCH-19.
- OPT-05, OPT-06 (separar `SPLIT_SEED`/`INIT_SEED`), OPT-07, OPT-08, OPT-12.
- Do Agente 2: SIMP-02, SIMP-03 (`video_core.py`), SIMP-04, SIMP-05, SIMP-06,
  SIMP-07, SIMP-09..SIMP-16 — seguindo o **protocolo de migração** dele
  (copiar verbatim → comparar `compute_metrics` sobre um `eval_pred` salvo →
  comparar `model_fingerprint()` → só então simplificar).
- ARCH-15 / SIMP-08 (`warmup_ratio` → `warmup_steps`): **agora ou nunca** — não
  no meio de uma série.

**Dependências:** nenhuma externa.
**Invalida:** nada dos resultados; muda a organização do código.

### Fase 2 — Correções de semântica + re-extração de dados — ~45 min de GPU/CPU

Esta é a fase que **mata todos os resultados existentes**. Fazer tudo de uma vez,
numa única onda, para não pagar a invalidação duas vezes.

| Item | Custo |
|---|---|
| ARCH-01 (`q_bias`/`v_bias`) | 0 (código) |
| ARCH-03 (b) — remover cross-attn | 0 (código) |
| ARCH-07 — dropout por amostra | 0 (código) |
| ARCH-08 — remover `bio_ln` | 0 (código) |
| ARCH-05 — `max_length=256` | 0 (código); +~22 % por época daqui em diante |
| ARCH-10 — formato hard da Task C (pred **e** gold) | 0 |
| ARCH-11 (b) — val dividido para ajuste de limiar, **junto com o 80/20 do OPT-11** (a metade não é viável a 253 amostras) | 0 |
| ARCH-16 — remover SWA | 0; libera ~2,6 GB |
| **ARCH-04 — re-extrair áudio** (janelas múltiplas, média só do válido) | **~10 min** (3 198 ÷ 7,50 it/s × ~1,4) |
| **ARCH-09 + OPT-01 — re-extrair frames** (decode sequencial, uint8) | **~35 min** (3 198 ÷ 1,54 it/s); pode ser mais com decode sequencial — medir em 20 vídeos e extrapolar |
| OPT-04 — batch/checkpointing, **conforme a Fase 0** | 0; depende de ARCH-07 e OPT-01 |

**Dependências:** Fase 0 (itens 1 e 6); ARCH-07 **antes** de OPT-04;
OPT-01 **junto com** OPT-04.

**O que esta fase invalida — lista completa, para não sobrar nada meio-vivo:**
- Todos os checkpoints (`best_model_seed_*`, `best_model_seed_*_cross_att`,
  `checkpoints_seed_*`).
- `ablation_results.csv`, `ablation_results.tex`, `ablation_seed_{42,7,2025}.csv`.
- `task_c_thresholds_seed_{42,7,2025}.json` (duplamente: pesos novos **e** a
  mudança de formato do ARCH-10 muda o gold hard).
- A tabela de treino da célula 52 e a avaliação da célula 55.
- **`predictions/cross_att/` e `predictions/swa/`** — são as submissões de um
  sistema que deixa de existir. Os resultados oficiais da competição derivados
  delas **não podem ser regenerados** e **não podem partilhar tabela** com
  números pós-correção. Na tese, têm de aparecer como "sistema submetido",
  separado de "sistema corrigido".

### Fase 3 — Run de calibração — ~1 h de GPU

**Uma** run, **15 épocas**, `INIT_SEED=42`, sistema pós-Fase-2, para reler a curva
de overfitting. Custo: 15 × 217,6 s ≈ 54,5 min (± o efeito líquido de +22 % do
ARCH-05 contra o ganho do OPT-04) + ~5 min de overhead.

**Por que esta fase existe:** o "6 épocas" do ARCH-06 vem da curva do sistema
**pré-correção**, seed 2025. Restaurar `q_bias`, re-extrair o áudio, remover a
cross-attn e dobrar `max_length` movem o ponto de overfitting. Carregar as 6
épocas adiante seria transportar um hiperparâmetro medido noutro sistema.

**Saída:** o orçamento de épocas, **congelado** e usado sem alteração em todas as
runs da Fase 4. Declarar na tese: escolhido no dev set, numa seed, uma vez.

**Dependências:** Fase 2 completa e congelada. Nenhuma mudança de configuração
entre esta run e a Fase 4 — qualquer alteração obriga a repeti-la.

### Fase 4 — Tabela de ablação defensável — ~4,1 h a ~5,0 h de GPU (base)

Um checkpoint treinado por (condição × `INIT_SEED`), com `SPLIT_SEED` fixo
(OPT-06), flags no construtor (ARCH-20), **B0 construído sem slot bio (OPT-13)**,
avaliação por um caminho único (OPT-08).

**Duas faixas de preço, porque dependem do OPT-04 (não verificado).** A Fase 2
introduz `max_length=256`, que pelo meu proxy de FLOPs custa +~22 % por época
(217,6 → ~265 s); o OPT-04 deveria compensar, mas não está medido:

| Condições | Runs | Se OPT-04 entregar ≥1,22× (época ≈ 218 s) | Se OPT-04 não entregar nada (época ≈ 265 s) |
|---|---|---|---|
| B0, B1, B2 × 3 inits | 9 | **≈ 4,1 h** | **≈ 5,0 h** |
| + braço `cross_attn` (ARCH-03 (a)) × 3 inits | +3 | +1,4 h | +1,7 h |
| + braço LoRA ampliado (OPT-02) × 3 inits | +3 | +1,4 h | +1,7 h |

Base do cálculo pessimista: 9 × 6 × 265 s = 14 310 s ≈ 4,0 h de treino puro +
9 × ~5 min de overhead ≈ 45 min → **≈ 4,7–5,0 h**. A Fase 0 (item 6) resolve qual
das duas colunas vale, **antes** de agendar.

Se o orçamento de épocas sair da Fase 3 maior que 6, escalar proporcionalmente:
cada época extra custa 218–265 s × nº de runs (a 9 runs, **+33 a +40 min por
época**).

**Análise:** delta pareado por seed + consistência de sinal (OPT-07), **nunca**
`|Δmédia| > std`. Deltas dentro de uma seed são pareados e válidos.

**Dependências:** **OPT-13 decidido por escrito** (como B0 é construído — sem
isso a tabela reproduz o achado nº 1 do brief), Fase 3 (orçamento de épocas
congelado), Fase 1 (ARCH-14, senão
uma métrica pode sumir sem erro no meio de 9 runs desatendidas), ARCH-19
(fingerprints persistidos — com 9–12 runs não dá para conferir à mão).

**Invalida:** nada novo — a Fase 2 já invalidou tudo.

### Fase 5 — Reconciliação de paper/tese — 0 GPU

SIMP-16 (dimensões de fusão no `CLAUDE.md` e no diagrama TikZ), SIMP-01 (linha
fantasma na tabela), DIV-03/BUG-C (`test_case`), e a separação explícita entre
"sistema submetido" e "sistema corrigido" imposta pela Fase 2.

### Orçamento total

| Fase | GPU (OPT-04 entrega) | GPU (OPT-04 não entrega) |
|---|---|---|
| 0 — verificação | ~15 min | ~15 min |
| 1 — custo zero | 0 | 0 |
| 2 — semântica + re-extração | ~45 min | ~45 min |
| 3 — calibração (15 épocas) | ~1 h | ~1,2 h |
| 4 — ablação (9 runs) | ~4,1 h | ~5,0 h |
| **Total (base)** | **≈ 6 h** | **≈ 7,2 h** |
| Fase 4 estendida (12 runs) | +1,4 h | +1,7 h |

Em qualquer dos ramos, o orçamento é de **menos de um dia de GPU** — a ordem de
grandeza que sustenta a afirmação nº 1 do sumário não muda.

---

## Conflitos que o orquestrador precisa resolver

| # | Conflito | Posições | Minha recomendação |
|---|---|---|---|
| 1 | **ARCH-03 (a) vs OPT-04.** A opção (a) do Agente 1 (atenção sobre os 1 568 tokens temporais) mantém `vid_out` `[B, 1568, 768]` vivo no grafo, o que exige gradient checkpointing e batch pequeno — exatamente o que quero remover. As duas propostas não coexistem. | Agente 1 admite que (a) é "caro em memória"; eu quero bs=8 sem checkpointing. | **(b) vence.** Além do argumento de memória, os ≈ 45 % de capacidade treináveis num mapa linear (OPT-02) num dataset de 2 271 amostras decidem sozinhos. (a) pode voltar como braço de ablação da Fase 4, com pooling espacial para `[B, 8, 768]` — **nunca como baseline**. |
| 2 | **ARCH-08: `BatchNorm1d` vs remover o `bio_ln`.** | Agente 1 oferece `BatchNorm1d` como proposta principal e "remover" como a mais segura. | **Remover.** `BatchNorm` acopla o resultado ao micro-batch (colide com OPT-04), reintroduz `update_bn` no SWA e faz treino e inferência divergirem. A opção `BatchNorm` deve sair da mesa, não ficar como alternativa. |
| 3 | **ARCH-20: sondas renomeadas vs re-treino.** O Agente 1 propõe, como medida intermediária, renomear os braços para `probe_zero_bio` etc. e publicar assim. | Agente 1 (pragmático); eu (custo). | **Re-treinar.** 9 runs = ~4,1 h. Publicar uma sonda renomeada como se fosse ablação continua sendo uma tabela que não sustenta a afirmação do paper, só com nome honesto. A medida intermediária dele só se justifica se houver deadline duro. |
| 4 | **ARCH-06 "6 épocas" vs a run de calibração.** O plano do Agente 1 aplica `num_train_epochs=6` diretamente. | Agente 1; eu exijo recalibrar. | **Calibrar.** As 6 épocas vêm da curva do sistema pré-correção (seed 2025). Adotá-las direto é transportar um hiperparâmetro medido noutro modelo. Custo da discordância: 1 h. |
| 5 | **ARCH-10 grid 72 → 60 apresentado como economia.** | Agente 1 cita a redução de chamadas PyEvALL entre os benefícios. | **Fazer a mudança, mudar o argumento.** 72 chamadas = 22,9 s medidos. Justificar só por eliminar o grau de liberdade contraditório; qualquer menção a custo aqui é enganosa e reforça a suposição errada sobre PyEvALL. |
| 6 | **SIMP-06 (`rt_norm`/`fix_norm`): remover vs usar.** O Agente 2 quer remover, e sinaliza conflito se o Agente 1 quiser usá-los como sinal auxiliar aditivo. | Agente 2 (remover); Agente 1 (ARCH-08.3, também remover). | **Remover** — os dois concordam, e eu também. Registro só para fechar o ponto: reconectá-los com escala bruta (~10²–10³) ao lado de features robust-scaled seria um bug novo. Se um dia se quiser esse sinal, é matéria do OPT-09 (triagem barata), não de uma mudança direta no baseline. |
| 7 | **DIV-06 (`build_bio_vector` vs `collect_bio_vectors`): unificar exige escolher uma semântica, e mexer em `collect_bio_vectors` muda o `RobustScaler` → re-treino.** | Agente 2 classifica como `[MUDA]` nível re-treino e pede decisão do Agente 1. | **Unificar dentro da Fase 2**, onde tudo já é re-treinado — o custo marginal é zero. Fora da Fase 2, o custo é uma onda inteira de re-treino por uma divergência que hoje afeta 0 registros (o próprio Agente 2 mediu: 0 valores não-castáveis). |
| 8 | **OPT-13: o que `use_bio=False` constrói.** O ARCH-20 move as flags para o construtor mas não especifica a arquitetura resultante; o default herdado (largura 2 688 com slot bio zerado) reproduz o achado nº 1 do brief. Toca código do Agente 1, mas a consequência é de protocolo. | Agente 1 (dono do `__init__`); eu (dono da validade da tabela). | **Opção (ii): construir `2560 → 1024` sem slot bio, declarando a diferença de ~150 k params (≈ 2,5 %) entre B0 e B2.** Um braço com entrada constante não é content-only, por mais que tenha sido re-treinado. Decidir por escrito antes da Fase 4. |
| 9 | **Célula 33 tem quatro donos** (Agente 1: épocas/SWA; Agente 2: `warmup_steps`; eu: batch/checkpointing/save/workers). | — | **Um único patch da célula 33, escrito na Fase 2, com todas as mudanças juntas**, e depois **congelado** até o fim da Fase 4. É a célula com maior risco de alteração acidental no meio de uma série. |

---

## O que eu NÃO consegui verificar e por quê

1. **O ganho real do OPT-04.** A faixa 1,5–2× é estimativa a partir do custo
   típico de gradient checkpointing e de melhor ocupação; não medi nada. Protocolo
   de medição está no finding (4 combinações × `max_steps=30`, ~15 min na VM).
   Também não sei qual é a GPU nem quanta VRAM está livre — sem isso não dá para
   afirmar que bs=8 cabe.
2. **Quanto do wall-clock é IO de frames.** As 24,3 GB/época e os ~111 MB/s
   implícitos são fato aritmético; **qual fração do tempo de época é espera de
   loader é desconhecida**, porque a leitura está sobreposta ao compute e o
   notebook não registra nada que os separe. Medir com
   `dataloader_num_workers=0` vs 8 em `max_steps=30`. Nota: no *meu* ambiente
   local um `du -sh frames/` estourou 2 min de timeout, o que sugere volume lento
   — mas isso é a máquina de desenvolvimento, **não** a VM, e não permite
   nenhuma conclusão sobre o treino.
3. **O custo do salvamento de checkpoints (OPT-03).** Não é separável: tanto o
   `train_runtime` quanto o intervalo entre avaliações já o incluem. Reporto o
   volume (39 GB) e a inutilidade, não uma fração de tempo — deliberadamente não
   chutei um intervalo.
4. **O custo da re-extração de frames com decode sequencial** (a correção do
   ARCH-09). Os 35 min vêm da taxa medida do método **atual** (seek por frame,
   1,54 it/s). O decode sequencial lê todos os frames de cada vídeo — pode ser
   mais lento (mais decode) ou mais rápido (sem seeks em VFR). Medir em 20 vídeos
   e extrapolar antes de agendar a Fase 2.
5. **A prevalência exata do ARCH-09.** Estimei 1–4 % do split de test contando
   *padrões distintos* de mensagem de erro (29 offsets `partial file`, 9 padrões
   `h264`) sobre 674 vídeos — mas cada vídeo com defeito emite várias linhas, e o
   mapeamento mensagem→vídeo é aproximado. Do split de **treino** não há output
   salvo (SIMP-04: a versão que gerou os tensores de treino não existe no
   repositório) — **não consigo dizer nada sobre a prevalência no treino**.
6. **A taxa de extração de áudio no split de treino.** Os 7,50 it/s são do split
   de **test** (674 vídeos). Se a distribuição de durações do treino for
   diferente, a taxa muda — mas mesmo dobrando, continua em ~15 min.
7. **Se a triagem com features congeladas (OPT-09) correlaciona com o resultado
   com LoRA.** É a premissa do finding e não está validada. Recomendo validar
   numa decisão única antes de confiar.
8. **Qualquer ganho de métrica.** Nada foi treinado. Todos os deltas propostos
   por mim e pelos Agentes 1 e 2 são hipóteses até a Fase 4.
9. **Nº de vCPUs da VM** (dimensionamento do OPT-05) e **se a VM tem uma ou mais
   GPUs** (que abriria a opção de paralelizar as 9 runs da Fase 4 e cortar o
   wall-clock por um fator igual ao nº de GPUs, sem mudar o custo em GPU-hora).
