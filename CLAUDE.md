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

## Arquitetura & Fusão

- Concatenação simples de tokens CLS superou a fusão por cross-attention na Task 2;
  cross-attention colapsou para padrões quase uniformes no dataset pequeno (~5.000 memes)
- **Task 2 — fusão:** 1024-d (duas projeções de 512-d concatenadas, sem projeção adicional)
- **Task 3 — fusão:** 2592→1024-d (768 texto + 768 vídeo + 1024 áudio + 32 bio)
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
  "Ambiente de Execução"); `warmup_ratio` foi removido nessa versão
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
  `reset_lora_weights()` e `reset_new_layer_weights()` devem ser chamados pós-instanciação
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
- `transformers` na VM: **5.6.0** (confirmado em 2026-08-01). Nesta versão, `warmup_ratio`
  foi removido (combinado em `warmup_steps`, que agora aceita float `[0,1)` como ratio —
  ver `MIGRATION_GUIDE_V5.md`). Logo `warmup_steps=0.1` no código atual **é a sintaxe correta
  para v5**, não o bug que se pensava sob a semântica de v4.x
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

  Comando usado para coletar (rodar de novo na VM se precisar revalidar):
  ```bash
  pip freeze | grep -Ei '^(torch|torchvision|transformers|accelerate|peft|safetensors|huggingface[_-]hub|emoji|wordsegment|pandas|scikit-learn|scipy|PyEvALL|ftfy|regex|tqdm|pillow|numpy)=='
  ```

## Convenções para o Claude Code

- Ao editar notebooks/scripts de treino, preferir funções de substituição completas
- Nunca aplicar ajuste de threshold antes do fim do treino (evitar leakage)
- Em `transformers==5.6.0` (versão da VM), `warmup_ratio` foi removido; usar `warmup_steps`
  com float `[0,1)` como ratio (ex.: `warmup_steps=0.1` = 10% de warmup) — não o contrário
- Ao mexer em LoRA + bf16, confirmar chamada de `reset_lora_weights()` /
  `reset_new_layer_weights()` pós-instanciação
- Ao tocar em normalização de features biométricas, usar `RobustScaler`
- Ao tocar em seleção de features biométricas, usar informação mútua (não chi-quadrado)
- Manter separação estrita entre estatísticas de treino e validação/teste (anti-leakage)