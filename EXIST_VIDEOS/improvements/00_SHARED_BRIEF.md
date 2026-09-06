# Brief compartilhado — revisão do modelo de vídeo (Task 3, EXIST 2026)

**Leia este arquivo inteiro antes de qualquer análise.** Todos os agentes do
time partem deste mesmo contexto.

## Contexto do projeto

Tese de mestrado (IPB/UTFPR) sobre deteção multimodal de sexismo sob o
paradigma *Learning with Disagreement* (LeWiDi), competição EXIST 2026 (CLEF),
equipa "Jow". Duas tarefas:

- **Task 2 — memes** (já auditada extensivamente; ver precedente abaixo)
- **Task 3 — vídeos do TikTok** ← **É O ALVO DESTA REVISÃO**

Cada task tem 3 subtasks:
- **A** — deteção binária de sexismo (YES/NO)
- **B** — intenção da fonte (NONE / JUDGEMENTAL / DIRECT)
- **C** — categorias finas, multilabel (6 categorias)

**Avaliação:** ICM-Norm (soft e hard) via PyEvALL, sob LeWiDi — os alvos são
distribuições soft de votos de anotadores, não rótulos únicos.

## Por que esta revisão existe

O modelo de vídeo (Task 3) foi **o de pior performance** dos dois, e foi
construído com muito menos cuidado que o de memes. O objetivo é: simplificar o
código, encontrar bugs e inconsistências de arquitetura, e propor melhorias
concretas — tudo documentado para revisitar em sessões futuras.

## Precedente: o que a auditoria do modelo de memes encontrou

Vale ler `EXIST_MEMES/ablation/ABLATION_REPORT.md` (especialmente seções 2.4,
3.4, 6.5, 8 e 9) — a auditoria do modelo de memes encontrou padrões que podem
se repetir aqui. Achados que valem checar no modelo de vídeo:

1. **Confound de fusão com modalidade ausente:** no modelo de memes, quando
   uma modalidade era desligada, o vetor virava zeros e passava por um
   `LayerNorm` global junto com as modalidades vivas — distorcendo as
   estatísticas do ramo vivo e desperdiçando largura de entrada das cabeças.
2. **Vazamento de validação em duas camadas:** (a) `load_best_model_at_end`
   selecionando época pela mesma métrica reportada; (b) busca de threshold da
   Task C rodando a cada época sobre o set de validação.
3. **Análise não-pareada mascarando ruído:** com 3 seeds pareadas, o teste
   correto é delta por seed + consistência de sinal, não `|Δmédia| > std`.
4. **Rotulagem enganosa de ablação:** um braço chamado "text_only" que na
   verdade continha texto+demografia.
5. **Claim histórica não reproduzível:** uma afirmação no `CLAUDE.md` sobre
   normalização L2 vinha de um teste feito antes de um bugfix de dados.

**Não assuma que os mesmos problemas existem aqui — verifique.** Mas conheça
o padrão de erro.

## Ambiente de execução (CRÍTICO)

O treino roda numa **VM remota separada** (`/home/a67606/Thesis/`), não na
máquina local. **Nunca assuma que a versão de uma lib local reflete a VM.**
Versões confirmadas na VM (2026-08-01):

| lib | versão |
|---|---|
| `torch` | 2.6.0+cu124 |
| `transformers` | **5.6.0** |
| `accelerate` | 1.13.0 |
| `peft` | 0.19.1 |
| `numpy` | 1.26.4 |
| `scikit-learn` | 1.8.0 |
| `PyEvALL` | 0.1.78 |

**Consequência de API mais importante — CORRIGIDO 2026-08-07, ler com atenção:**

A versão anterior deste brief (e o `CLAUDE.md` do projeto) afirmava que
`warmup_ratio` **foi removido** em `transformers==5.6.0`. **Isso é
factualmente errado.** Evidência empírica do stderr da célula 33 do próprio
`Thesis_video_fixed.ipynb`:

```
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
```

Ou seja: `warmup_ratio` está **depreciado mas funcional** nesta versão — o
treino do modelo de vídeo rodou com ele. Consequências para o time:

- **NÃO trate `warmup_ratio` no código de vídeo como bug.** Ele funciona.
- Migrar para `warmup_steps` (float `[0,1)` = ratio) é uma melhoria de
  *durabilidade* (evita quebra numa versão futura), não uma correção de bug —
  classifique assim se propuser.
- `warmup_steps=0.1` no código de memes também está correto. Só o *motivo*
  documentado no `CLAUDE.md` estava errado.

Regra geral que continua valendo: se precisar decidir algo dependente de
versão de API, diga explicitamente que precisa ser confirmado rodando
`pip show <lib>` na VM — não decida por memória.

## Arquitetura atual do modelo de vídeo (resumo, verificar no código)

`EXISTVideoModel` combina 4 modalidades:
- **Vídeo:** VideoMAE-base com LoRA (r=16, α=32) → `TemporalAttentionPool` → 768-d
- **Texto:** XLM-RoBERTa-large → CLS → 1024-d
- **Áudio:** Whisper-large-v3 pré-extraído offline (1280-d) → projetado
- **Bio:** eye-tracking + heart rate, `RobustScaler`, → MLP

Há também: `cross_attn` (MultiheadAttention texto←vídeo), embeddings
per-viewer para normalização biométrica, modality dropout, SWA callback,
`MultiTaskTrainer` customizado.

## Restrições conhecidas do projeto (do CLAUDE.md — respeitar)

- Labels `UNKNOWN` e `REPORTED` são excluídas dos targets de treino.
- O voto `-` (não-sexista) na Task B é remapeado para `"NONE"`, **não
  filtrado** — filtrar infla artificialmente DIRECT/JUDGEMENTAL.
- Features biométricas usam `RobustScaler`, **não** z-score.
- Seleção de features biométricas usa **informação mútua**, não chi-quadrado
  (chi² é inapropriado para features contínuas).
- EEG foi descartado (70,6% de canais mortos) — não sugira reintroduzir.
- Ajuste de threshold só pós-treino (evitar leakage).
- SWA conflita com `load_best_model_at_end` do HF Trainer — os pesos SWA são
  sobrescritos se o carregamento do melhor checkpoint estiver ativo. Requer
  escolha explícita de estratégia.
- `bf16=True` causou corrupção recorrente de pesos LoRA; `reset_lora_weights()`
  / `reset_new_layer_weights()` devem ser chamados pós-instanciação.
- Embeddings per-viewer para normalização biométrica **neutralizaram** os
  benefícios dos biosinais no passado (suspeita: confounding de língua —
  língua do avaliador correlacionada com a língua do vídeo).
- Determinismo: `set_all_seeds` + flags CUDA determinísticas + re-seed
  imediatamente antes de cada init de pesos (LoRA A é aleatória) +
  `model_fingerprint()` para verificar bit a bit entre runs.

## Arquivos a analisar

**Versões limpas (código sem outputs de notebook — USE ESTAS para ler):**
```
/private/tmp/claude-501/-Users-jonathansantosdasilva-Documents-projects-VM-THESIS/bb660d18-45cc-4227-932b-e1510d99ab65/scratchpad/video_code/
  Thesis_video_fixed_CODE.py    (1418 linhas — o modelo e treino principal)
  extract_video_CODE.py         (114 linhas — extração de frames)
  extract_audio_CODE.py         (108 linhas — extração de features de áudio)
  inference_video.py            (cópia do original)
  ablation_harness.py           (cópia do original)
```
Cada bloco tem um marcador `# ===== [cell N] CODE =====` que mapeia de volta
para a célula do notebook original — **sempre cite o número da célula** ao
reportar um finding, para ser rastreável.

**Originais (NÃO MODIFICAR NENHUM DELES):**
```
/Users/jonathansantosdasilva/Documents/projects/VM-THESIS/EXIST_VIDEOS/
  Thesis_video_fixed.ipynb
  extract_video.ipynb
  extract_audio.ipynb
  inference_video.py
  ablation_harness.py
  ablation_results.csv, ablation_seed_{42,7,2025}.csv   (resultados existentes)
```

**IMPORTANTE:** a extração de embeddings de vídeo (frames) e áudio é feita nos
notebooks `extract_video.ipynb` / `extract_audio.ipynb` — **os três agentes
precisam analisar esses arquivos também**, não só o modelo. Bugs de
pré-processamento (normalização, sampling de frames, alinhamento
temporal áudio-vídeo, ordem de canais, escala) são candidatos fortes para
explicar performance ruim e são fáceis de passar despercebidos.

## Regras para todos os agentes

1. **NÃO modifique nenhum arquivo original.** Esta fase é de análise. O
   arquivo novo melhorado será escrito depois, numa fase separada, com base
   nos findings reconciliados.
2. **Escreva seus findings no arquivo que seu prompt especificar**, dentro de
   `EXIST_VIDEOS/improvements/`.
3. **Todo finding precisa de:** ID único, severidade, localização exata
   (arquivo + número de célula + trecho de código), o que está errado, por que
   importa (impacto concreto em métrica/corretude/manutenção), e a correção
   proposta. Sem correção proposta, não é um finding utilizável.
4. **Distinga fato de hipótese.** Se você não conseguiu verificar algo
   (ex.: precisa rodar na VM, precisa de dados que não estão disponíveis),
   marque explicitamente como hipótese não verificada — não afirme.
5. **Não invente números de métrica.** Se citar performance, cite a fonte
   (`ablation_results.csv` etc.).
6. **Priorize.** Um finding que muda métrica vale mais que um finding de
   estilo. Ordene por impacto.
7. **Seja específico sobre risco de conflito:** se sua proposta mexe num
   trecho que outro agente do time provavelmente também vai querer mexer,
   sinalize isso explicitamente numa seção "possíveis conflitos" — o time
   precisa reconciliar antes de escrever código.
