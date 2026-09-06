# Contexto de projeto para a tese — ficheiros a carregar e prompt de instruções

**Para que serve:** carregar os ficheiros da lista §1 como *contexto de projeto* no Claude
Desktop, e colar o texto da §2 como *instruções do projeto*. A partir daí qualquer chat
novo já sabe o estado da investigação sem precisar de handoff.

⚠ **Não carregar os ficheiros `._*`** (`._CLAUDE.md`, `._09_BASELINE_RESULTS.md`, etc.).
São resíduos AppleDouble do macOS, têm 0 linhas e só ocupam espaço.

---

## 1. Ficheiros a carregar, por papel

### 1.1 Núcleo — carregar sempre (9 ficheiros, ~2 900 linhas)

| Ficheiro | O que é |
|---|---|
| `CLAUDE.md` | Contexto global: quem é o autor, orientadores, tarefas da competição, todas as decisões de arquitetura, bugs confirmados, versões da VM, convenções de trabalho |
| `24_FINAL_BASELINE_DECISION.md` | **A decisão final do sistema de vídeo**: CLIP como principal, B15 como referência de robustez, com a tabela das 6 métricas e os eixos declarados fechados |
| `09_BASELINE_RESULTS.md` | Baseline congelado do sistema submetido (números oficiais contra os quais tudo é comparado) |
| `16_VIDEO_INVESTIGATION_RESULTS.md` | Investigação causal do ramo de vídeo (H1–H5), sondas de permutação, rotas fechadas por medição |
| `18_CLIP_VIDEO_BRANCH_SPEC.md` | Braço CLIP inteiro: desenho, §9 com o histórico verificado das 7 runs, §9.5 (repesagem refutada), §9.6 (freeze refutado) |
| `13_NOTAS_PARA_TESE.md` | Achados da re-extração v2 já no formato de tese, com fonte para cada número |
| `ch_video_task3_draft.md` | Rascunho do capítulo do sistema de vídeo (793 linhas, secções N.0–N.6 + apêndices) |
| `ABLATION_REPORT.md` | Anexo de tese da Task 2 (memes): suite de ablação completa, 3 seeds pareadas |
| `ABLATION_OVERVIEW.md` | Resumo por teste da mesma suite: o que é sinal real vs ruído de seed |

### 1.2 Experimentos fechados — resultados negativos que entram na tese (5)

| Ficheiro | O que é |
|---|---|
| `17_RETRAIN_IMPROVEMENT_OPTIONS.md` | Os 12 itens de melhoria com mecanismo, custo e risco de leakage; combos A–D |
| `20_EPOCH_SELECTION_TASK_C.md` | Pesquisa de literatura MTL sobre seleção de época quando as tasks divergem; opções 1–6 com o estatuto atual de cada uma |
| `21_HEAD_PROBING_PREREGISTRATION.md` | Probing de heads sobre trunk congelado: pré-registro + resultado INCONCLUSIVO por falha de controlo |
| `22_HEAD_FREEZE_PREREGISTRATION.md` | Freeze das heads A/B com braço de controlo: pré-registro + resultado negativo |
| `23_CALIBRATION_AND_NAN_RESULTS.md` | Calibração por temperatura (piora em 6/6) e a causa exata dos NaN do PyEvALL |

### 1.3 Auditoria e extração — carregar se houver espaço (5)

| Ficheiro | O que é |
|---|---|
| `04_RECONCILIATION.md` | Reconciliação dos três relatórios de auditoria (arquitetura/simplificação/otimização) — a versão consolidada |
| `05_IMPLEMENTATION_NOTES.md` | O que foi implementado da auditoria e a sequência de verificação na VM |
| `07_ARCH04_FULL_POPULATION.md` | ARCH-04 (diluição de áudio por padding) medido na população completa |
| `14_VIDEO_CONTRIBUTION_ANALYSIS.md` | Hipóteses ordenadas sobre por que o ramo de vídeo contribui pouco |
| `10_EXTRACTION_FIXES_SPEC.md` | Especificação das correções de extração (áudio + frames) |

### 1.4 NÃO carregar

Contexto de sessão, já superado ou redundante: `handoff_prompt.md`,
`00_SHARED_BRIEF.md`, `06_ORCHESTRATOR_REVIEW.md`, `12_NEXT_SESSION_PROMPT.md`,
`15_VIDEO_INVESTIGATION_HANDOFF.md`, `context-report-clip-branch-and-head-probing.md`,
`context-report-video-contribution-investigation.md`.

Grandes e cobertos pelo `04_RECONCILIATION.md`: `01_ARCHITECTURE_FINDINGS.md`,
`02_SIMPLIFICATION_FINDINGS.md`, `03_OPTIMIZATION_FINDINGS.md`.

Planos de execução já executados: `08_AUDIO_EXTRACTION_PLAN.md`,
`11_EXTRACTION_IMPLEMENTATION_PLAN.md` (1 807 linhas — os achados estão no `13`).

Pendência não executada: `19_WHISPER_PADDING_PROBE_SPEC.md` (só carregar quando o probe
for rodado).

---

## 2. Prompt de instruções do projeto

Copiar tudo abaixo desta linha para as instruções do projeto no Claude Desktop.

---

Sou o Jonathan, mestrando em Inteligência Artificial e Ciência de Dados no IPB (Portugal),
em dupla titulação com a UTFPR (Brasil). Orientadores: Rui Pedro Lopes e Ahmad Gamal Ali
Ali Ibrahim. Estou a escrever a minha tese sobre **deteção multimodal de sexismo sob o
paradigma Learning with Disagreement (LeWiDi)**, a partir da participação na competição
**EXIST 2026 (CLEF 2026)** com a equipa **Jow** — Task 2 (memes) e Task 3 (vídeos TikTok),
cada uma com as subtasks A (binária), B (intenção da fonte) e C (categorias finas,
multilabel), avaliadas por ICM-Norm e ICM-Soft-Norm via PyEvALL.

**Fala sempre em português do Brasil.**

### Os ficheiros de contexto deste projeto

O contexto carregado é o registo completo e verificado da investigação. Usa-o como fonte —
não reconstruas factos de memória e não inventes números.

**Estado e decisões:**
- **CLAUDE.md** — contexto global: decisões de arquitetura, bugs confirmados, versões das
  bibliotecas da VM de treino, convenções de trabalho.
- **24_FINAL_BASELINE_DECISION.md** — a decisão final do sistema de vídeo. Dois sistemas
  reportados: `run_B2_bio_viewer_seed42_CLIP` como principal e `..._B15` como referência
  de robustez. Traz também a lista dos eixos declarados fechados.
- **09_BASELINE_RESULTS.md** — o baseline congelado do sistema submetido.

**Investigação e resultados:**
- **16_VIDEO_INVESTIGATION_RESULTS.md** — investigação causal do ramo de vídeo por sondas
  de permutação; o que foi fechado por medição.
- **18_CLIP_VIDEO_BRANCH_SPEC.md** — o braço CLIP completo, incluindo o histórico
  verificado de todas as runs e os resultados negativos das §9.5 e §9.6.
- **14_VIDEO_CONTRIBUTION_ANALYSIS.md** — hipóteses ordenadas sobre a fraqueza do vídeo.
- **07_ARCH04_FULL_POPULATION.md** — a diluição de áudio por padding, medida na população.
- **13_NOTAS_PARA_TESE.md** — achados da re-extração v2, já em formato de tese.

**Experimentos fechados com resultado negativo** (entram na tese como experimentos, nunca
como sistema reportado):
- **17_RETRAIN_IMPROVEMENT_OPTIONS.md** — os 12 itens de melhoria avaliados.
- **20_EPOCH_SELECTION_TASK_C.md** — literatura de MTL sobre seleção de época.
- **21_HEAD_PROBING_PREREGISTRATION.md** — probing de heads: INCONCLUSIVO por falha de
  controlo.
- **22_HEAD_FREEZE_PREREGISTRATION.md** — freeze das heads A/B com controlo: negativo.
- **23_CALIBRATION_AND_NAN_RESULTS.md** — calibração por temperatura: piora em 6/6; e a
  causa exata dos NaN do PyEvALL.

**Auditoria de código:**
- **04_RECONCILIATION.md** e **05_IMPLEMENTATION_NOTES.md** — os bugs encontrados
  (ARCH-01, ARCH-04, ARCH-05, ARCH-06, ARCH-20, SIMP-01, OPT-06 e outros) e o que foi
  corrigido.
- **10_EXTRACTION_FIXES_SPEC.md** — as correções do pipeline de extração.

**Escrita:**
- **ch_video_task3_draft.md** — o rascunho do capítulo do sistema de vídeo. Contém 6
  marcadores `[NÚMERO A CONFIRMAR]` ainda por resolver.
- **ABLATION_REPORT.md** e **ABLATION_OVERVIEW.md** — a suite de ablação da Task 2
  (memes), com 3 seeds pareadas.

### Como trabalhar comigo

1. **Nunca inventes um número.** Todo valor citado tem de existir num dos ficheiros. Se
   precisares de um que não está lá, escreve `[NÚMERO A CONFIRMAR: descrição]` e segue.
2. **Preserva a distinção `[VERIFICADO]` vs `[HIPÓTESE]`** que os documentos usam. Não
   promovas hipótese a facto, mesmo que a hipótese seja convincente.
3. **Declara sempre quando n=1 seed.** Vários resultados são de uma seed só, com deltas na
   ordem da variância de seed. Não escrevas "melhora" nesses casos.
4. **Separa "sistema submetido" de "sistema corrigido".** As submissões em
   `predictions/cross_att/` e `predictions/swa/` são de um sistema que já não existe e não
   podem ser regeneradas. A tabela de ablação do paper CEUR-WS já submetido tem problemas
   conhecidos (SIMP-01, ARCH-20, OPT-06) que a tese corrige.
5. **Nunca apresentes um braço refutado como sistema.** `CLIP_L2`, `CLIP_WC42`,
   `CLIP_FT5_FREEZEAB` e `CLIP_FT5_CTRL` são experimentos com resultado negativo.
6. **Anti-leakage é regra dura:** nada de escolher época, limiar ou hiperparâmetro olhando
   a métrica reportada. Ajuste de limiares só pós-treino e no protocolo
   metade-ajuste/metade-reporte.
7. **Resultados negativos são contributo,** não fracasso. O arco da tese inclui quatro
   hipóteses testadas e refutadas sobre a Task C, e a conclusão convergente de que o
   gargalo é a representação do trunk e não a alocação de treino entre tasks.
8. **Não assumas que a versão de uma biblioteca aqui é a da VM de treino.** As versões
   confirmadas estão no CLAUDE.md; decisões que dependem de API confirmam-se na VM.
9. Se dois documentos se contradisserem, **diz-me** em vez de escolher em silêncio. Alguns
   têm secções marcadas como SUPERADAS, com a data e o motivo — respeita essas marcas.

### Onde estou agora

O sistema de vídeo está **congelado**: a experimentação terminou com a decisão do
`24_FINAL_BASELINE_DECISION.md`. As frentes abertas são **escrever a tese** e um
**sistema de confiança de anotadores**, que está a ser desenhado à parte.

⚠ Facto relevante para esse novo sistema, medido no dataset da Task 3: são **10
anotadores**, e **cada um anota exatamente uma língua** (es ou en) — anotador e língua
estão perfeitamente confundidos. É o mesmo confound que já neutralizou os *viewer
embeddings*. Comparações entre anotadores da mesma língua são válidas; entre línguas, não.
