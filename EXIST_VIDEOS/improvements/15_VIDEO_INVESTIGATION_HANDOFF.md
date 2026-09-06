# 15 — Investigação: por que o ramo de vídeo contribui pouco? (handoff)

**O que este documento é:** o estado completo da investigação sobre a baixa contribuição
do ramo de vídeo na Task 3, para ser retomado numa sessão nova. Consolida as sondas
medidas, o que foi verificado diretamente no código, o que foi refutado, as hipóteses em
aberto e os cinco testes propostos.

**Companion:** `14_VIDEO_CONTRIBUTION_ANALYSIS.md` (464 linhas) tem a análise longa das
hipóteses H1–H6, com `arquivo:linha` em cada afirmação. Este documento é o resumo
acionável; o 14 é a fundamentação.

**Convenção de confiança — respeitar ao ler:**
- `[VERIFICADO]` — conferido diretamente no código ou por aritmética nesta sessão.
- `[MEDIDO]` — número produzido por execução, com o artefato que o contém.
- `[RELATADO]` — produzido por um agente e **não** reconferido de forma independente.
- `[HIPÓTESE]` — interpretação. Não escrever na tese como fato.

---

## 1. A pergunta

O sistema é de deteção de sexismo em vídeos do TikTok. Permutar os frames entre amostras
(destruindo o alinhamento vídeo↔rótulo) derruba a Task B em 0,076, e quase nada em A e C.

**A objeção do autor, que motivou esta investigação:** 0,076 é pouco demais para um
sistema cuja modalidade central é vídeo. No modelo de memes do mesmo autor, uma única
imagem estática contribui de forma grande. Por que a diferença?

A investigação **não** é sobre melhorar a métrica. É sobre descobrir se existe um defeito
(e qual) ou se existe um teto de informação (e onde).

---

## 2. Evidência medida

### 2.1 Sonda de permutação `[MEDIDO — EXIST_VIDEOS/test.ipynb]`

Checkpoint `run_B2_bio_viewer_seed42_B15/` (frames v2 + áudio v2, `restore_biases=True`,
15 épocas), val = 505 amostras do split congelado. O tensor `frames` da amostra *i* é
substituído pelo de outra amostra (derangement — ninguém recebe o próprio). A entrada
continua na distribuição; só o alinhamento com o rótulo é destruído.

| métrica | normal | shuf0 | shuf1 | Δ médio |
|---|---|---|---|---|
| Task A ICMNorm hard | 0,5536 | 0,5414 | 0,5475 | −0,009 |
| Task A ICMSoftNorm soft | 0,5058 | 0,4825 | 0,4909 | −0,019 |
| **Task B ICMNorm hard** | 0,4968 | 0,4106 | 0,4316 | **−0,076** |
| **Task B ICMSoftNorm soft** | 0,3947 | 0,3371 | 0,3411 | **−0,056** |
| Task C ICMNorm hard | 0,3436 | 0,3408 | 0,3528 | +0,003 |
| Task C ICMSoftNorm soft | 0,1514 | 0,1467 | 0,1538 | −0,001 |

Duas permutações independentes concordam na direção e na ordem de grandeza.

### 2.2 Sonda de zerar `vid_feat` pós-pooling `[MEDIDO]`

Via `forward_hook` em `model.video_pool` devolvendo `zeros_like`. Equivale ao estado que
o modality dropout produz no treino. Task B hard: **0,4749** (−0,022).

**Permutar machuca 3,5× mais que remover.** Leitura: vídeo errado é ativamente enganoso,
enquanto ausência é um estado para o qual o modality dropout treinou robustez — logo a
sonda de zerar **subestima por construção** a dependência.

### 2.3 Sonda descartada — registrar para não repetir `[MEDIDO]`

A primeira tentativa zerou o tensor `frames` **depois** da normalização. Isso não apaga o
vídeo: produz uma imagem cinza uniforme (o pixel médio do ImageNet), que o VideoMAE
converte num embedding **constante mas não-nulo**, absorvido pelo modelo como viés fixo.
Resultado: métricas quase não mudam e algumas **melhoram** (Task A hard +0,008, Task C
hard +0,010), sugerindo falsamente que o vídeo não contribui.

**Lição metodológica, aproveitável na tese:** zerar entrada normalizada é uma sonda
enganosa; permutação preserva a distribuição de entrada e mede o que interessa.

---

## 3. Achados verificados diretamente no código

### 3.1 Desequilíbrio de capacidade `[VERIFICADO — aritmética conferida]`

Parâmetros treináveis, com o mesmo `make_lora_config` (r=16, α=32, alvos q/v) aplicado
aos dois backbones:

| bloco | params | conta |
|---|---:|---|
| fusion (2688→1024) | 2 753 536 | 2688·1024 + 1024 |
| xlmr LoRA | 1 572 864 | 24 cam. × 2 mód. × (1024·16 + 16·1024) |
| audio_proj (1280→768 + LN) | 985 344 | 1280·768 + 768 + 2·768 |
| **videomae LoRA** | **589 824** | 12 cam. × 2 mód. × (768·16 + 16·768) |
| video_pool | 98 561 | `[RELATADO]` |
| bio_mlp + viewer | 19 664 | `[RELATADO]` |
| heads a/b/c | 10 250 | `[RELATADO]` |

**O ramo de áudio — que apenas projeta um vetor pré-extraído fixo — tem 1,4× mais
parâmetros treináveis que todo o encoder de vídeo adaptável.** O ramo de texto tem 2,7×.

A assimetria não é decisão de design: é consequência de aplicar a mesma config LoRA a um
backbone de 24 camadas × 1024-d e a outro de 12 × 768-d.

### 3.2 Contraste com o ramo visual dos memes `[RELATADO — estimativa por fórmula]`

O ramo visual dos memes destrava 4 camadas do CLIP ViT-L (hidden 1024, MLP 4096) →
**~50,4 M** parâmetros treináveis, **full-rank**, sobre features de pré-treino
**contrastivo** (400 M pares imagem–texto).

**≈85× a capacidade de adaptação do ramo de vídeo**, sobre features que já nascem
semânticas. Esta é a resposta mais direta a *"por que uma imagem CLIP contribui muito e
16 frames VideoMAE contribuem pouco"*: **não é o número de frames.**

⚠ Os ~50,4 M são estimativa por fórmula a partir de `memes_core.py:487-494`, não medição.
**Reconferir contando no código** antes de ir para a tese.

### 3.3 O `TemporalAttentionPool` não é temporal `[VERIFICADO — video_core.py:949-967]`

```python
def forward(self, x):                           # x = [B, 1568, 768]
    logits = self.attn(x)                       # [B, T, 1]
    weights = torch.softmax(logits, dim=1)      # softmax sobre 1568 tokens
    return (weights * x).sum(dim=1)             # [B, 768]
```

A docstring diz *"pooling por atenção aprendida sobre a dimensão temporal"* e *"cada token
de frame recebe um peso escalar"*. **Ambas as afirmações estão erradas:** os 1 568 tokens
do `last_hidden_state` são espaço-temporais (8 posições temporais × 196 patches de 14×14),
não frames. Espaço e tempo competem no **mesmo softmax**, e o peso é um escalar por token
— nenhuma informação de *onde na tela* ou *quando no vídeo* sobrevive à soma global.

⚠ **A descrição de arquitetura da tese precisa ser corrigida**, independentemente do resto
desta investigação. Isso não é hipótese: é o que o código faz.

Nota: o VideoMAE **não tem token CLS** pré-treinado, então este agregador nasce do zero e
aprende com ~2 019 amostras — ao contrário do CLS do CLIP nos memes.

### 3.4 O que foi refutado `[MEDIDO — sobre trainable.safetensors, CPU]` `[RELATADO]`

**A fusão não ignora o vídeo.** Norma média das colunas de `fusion.0.weight` por slot:

| slot | norma média |
|---|---|
| **vídeo** | **0,749** |
| texto | 0,630 |
| bio | 0,504 |
| áudio | 0,435 |

A hipótese "o modelo aprendeu a descartar o slot de vídeo" morre aqui — o slot de vídeo
tem peso nominal **maior** que o do texto.

⚠ **Caveat que invalida a conclusão forte:** contribuição = peso × **escala da feature**.
As escalas de `vid_feat` (pooling, sem norm), CLS do XLM-R (sem norm) e áudio (com LN) não
são conhecidas sem um forward. **Medição em falta** (teste 3 abaixo): norma ℓ₂ média por
amostra de cada feature na val.

**Outras medições do agente, não reconferidas** `[RELATADO]`:
- LoRA do VideoMAE moveu ~2 % da norma da matriz base (‖ΔW‖_F ≈ 0,80 vs ‖W‖_F ≈ 38),
  confinado a rank ≤ 16 de 768 — o encoder está quase no regime de sonda linear.
- O `video_pool` treinou (pesos 2–3× a escala de init) — não ficou parado no init.

### 3.5 O ARCH-01 não era a explicação `[VERIFICADO — meta.json do B15]`

O checkpoint sondado tem `"restore_biases": true`. O candidato histórico nº 1 para "vídeo
fraco" (biases `q_bias`/`v_bias` do VideoMAE chegando zerados) **estava corrigido**, e o
vídeo continua fraco. Confirmado também pelo braço A−A′, cujo delta isolado ficou abaixo
de 0,01 em cinco das seis métricas.

---

## 4. Hipóteses em aberto

Detalhamento completo em `14_VIDEO_CONTRIBUTION_ANALYSIS.md` §2. Resumo:

| # | hipótese | explica o padrão B-específico? |
|---|---|---|
| **H1** | O conteúdo sexista está na **fala** (já chega por transcrição); o visual só entrega **formato/enquadramento** — e formato é o que separa DIRECT de JUDGEMENTAL | **sim, diretamente** |
| **H2** | Desequilíbrio de capacidade + pré-treino errado (§3.1–3.2): o encoder não consegue extrair o que existe | parcialmente |
| **H3** | *(hipótese nula)* Task B é só a cabeça mais frágil a **qualquer** perturbação do vetor fundido — classes raras, 505 amostras | sim, e é a que precisa morrer primeiro |
| **H4** | Amostragem temporal esparsa (gap mediano 1,21 s vs ~0,13 s do pré-treino): o VideoMAE vira "bag of snapshots" | não sozinha |
| **H5** | O pooling (§3.3) destrói estrutura espacial e temporal | não sozinha — afetaria A, B e C igualmente |
| **H6** | Vídeos truncados (1,0 % treino / 0,6 % teste) | marginal |

**H3 é a prioridade lógica:** se o −0,076 for fragilidade da cabeça B e não sinal de
vídeo, todas as outras hipóteses ficam sem objeto.

---

## 5. Os cinco testes

Todos sobre o checkpoint `run_B2_bio_viewer_seed42_B15/`, **sem treinar**. Total
estimado: **< 1 h de GPU**.

### Teste 1 — Escada de permutação (texto, áudio, bio)
Repetir a sonda de permutação para **cada** modalidade, não só vídeo. Produz a quota de
dependência por modalidade na mesma escala.
**Decide:** H1. Se permutar texto derrubar ~0,3 e vídeo 0,076, a hierarquia fica
explícita e o resultado do vídeo deixa de ser anômalo — vira proporcional.
**Custo:** ~20 GPU-min. **Cuidado:** usar o mesmo derangement (mesma seed) entre
modalidades, para que as quotas sejam comparáveis.

### Teste 2 — Ruído pareado + mais derangements
Injetar em `vid_feat` ruído gaussiano de norma equivalente à da permutação, e rodar 5
derangements independentes.
**Decide:** H3. Se ruído aleatório derrubar Task B tanto quanto a permutação, o −0,076 é
**fragilidade da cabeça**, não informação de vídeo — e a investigação inteira muda de
objeto. Se a permutação derrubar significativamente mais que o ruído, há sinal.
**Custo:** ~10 GPU-min. **Este teste vem primeiro na ordem lógica.**

### Teste 3 — Forward único com cache de features + diagnóstico do pool
Um `forward` sobre a val cacheando `vid_feat`, `aud_feat`, `txt_feat`, `bio` e os pesos
do softmax do pool. Medir: (a) norma ℓ₂ média por amostra de cada feature — fecha o
caveat de §3.4; (b) **entropia do softmax do pool** — distingue "pool ≈ média uniforme"
(entropia ≈ log 1568) de "pool seletivo"; (c) variância de `vid_feat` entre amostras —
se for baixa, o encoder devolve quase o mesmo vetor para todo vídeo.
**Decide:** H2, H5, caveat de §3.4. **Custo:** ~15 GPU-min. Guardar os caches: os
testes 4 e 5 rodam sobre eles em CPU.

### Teste 4 — Sondas lineares sobre os caches
Regressão logística sobre `vid_feat` (congelado) para prever: rótulo A, rótulo B, **língua
do vídeo**, **duração**. Comparar com a mesma sonda sobre `txt_feat`.
**Decide:** o que `vid_feat` realmente codifica. Se prever bem duração ou língua e mal os
rótulos, o encoder está a codificar propriedades de captura, não semântica.
**Custo:** CPU-min. **Anti-leakage:** ajustar em metade da val, reportar na outra.

### Teste 5 — Análise de flips da Task B
Matriz de transição das predições de B sob permutação: quais classes viram quais.
Cruzar com a transcrição dos casos que mudaram.
**Decide:** H1. Se os flips se concentrarem no eixo DIRECT↔JUDGEMENTAL, confirma que o
visual entrega registro/formato — o mecanismo exato que H1 propõe.
**Custo:** CPU-min sobre os caches do teste 3.

**Ordem recomendada:** 2 → 1 → 3 → 4 → 5. O teste 2 pode encerrar a investigação sozinho.

---

## 6. O que NÃO fazer

- **Não re-extrair features.** A re-extração v2 foi concluída e verificada em 2026-08-09
  (custou um dia). Frames: 64 uint8/vídeo, 2 524 + 674, 0 faltantes.
- **Não trocar de encoder como primeira ação.** Só se a análise sustentar. Se sustentar, a
  via indicada é CLIP sobre frames-chave, não VideoMAE com mais frames.
- **Não rodar 32 frames antes dos testes.** Custo 4–5 h; o gap continua sendo função da
  duração (razão p90/p10 = 7,2× para 16, 32 ou 64).
- **Não citar as sondas como evidência de que o vídeo ajuda.** Regra explícita em
  `video_core.py:1069`: a sonda é **gatilho** para gastar runs num braço treinado, nunca
  evidência. Quem prova é o braço.
- **Não usar `n=1` para afirmação de tese.** Este projeto já registrou um ganho de +0,020
  que não replicou entre seeds.

---

## 7. Contexto do sistema (para quem chega agora)

- **Arquitetura:** VideoMAE-base+LoRA → pool → 768 · XLM-R-large+LoRA → CLS 1024 ·
  Whisper-large-v3 pré-extraído [1280] → 768 · bio-MLP 17→128. Concatenação 2688 →
  Linear→GELU→Dropout → 1024 → 3 cabeças (A binária, B 3 classes, C 6 multilabel).
  Sem cross-attention (removida — atendia sobre uma única key).
- **Treino:** batch efetivo 32 (micro 2 × accum 16), bf16, gradient checkpointing,
  2 019 treino / 505 val (split congelado `Dataset/val_split_42.json`).
- **Estado:** re-extração v2 concluída e verificada; três braços de atribuição rodados;
  `B15 − baseline` positivo nas 5 métricas comparáveis. Ver
  `13_NOTAS_PARA_TESE.md` §9.
- **Ambiente:** treino roda numa **VM separada** (Jupyter remoto). O repositório local é
  um mount SSHFS lento — não ler `.pt`, não usar `du`, não fazer `find` recursivo em
  `Dataset/`, `frames/`, `audio_features/`. Versões da VM na tabela do `CLAUDE.md`.

---

*Escrito em 2026-08-09. Fatos com fonte; hipóteses marcadas como tais. Nenhum número
deste documento veio de treino novo — todos vêm de sondas sobre o checkpoint B15 ou de
leitura de código.*
