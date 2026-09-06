# 12 — Prompt para a próxima sessão

Cole o bloco abaixo como primeira mensagem de uma sessão nova, na raiz do repositório.

---

```
Vamos executar a re-extração de features da Task 3 (vídeos). O planeamento já está
feito e validado — esta sessão é de IMPLEMENTAÇÃO e VERIFICAÇÃO, não de replanejamento.

## Leia primeiro, nesta ordem

1. EXIST_VIDEOS/improvements/11_EXTRACTION_IMPLEMENTATION_PLAN.md
   ← ESTE É O PLANO. Tem código integral célula a célula, ordem de execução (P0–P7),
   os 8 testes de verificação como código, modos de falha e checklist. A §13 tem as
   decisões do autor JÁ TOMADAS — não reabra nenhuma.
2. EXIST_VIDEOS/improvements/10_EXTRACTION_FIXES_SPEC.md
   ← a especificação: cada correção (V1–V7 vídeo, A1–A5 áudio, C1–C4 no video_core)
   com o motivo. Se o plano e a spec divergirem, o PLANO ganha (é posterior).
3. EXIST_VIDEOS/improvements/09_BASELINE_RESULTS.md
   ← o baseline congelado e a REGRA DE SIMETRIA (estado final vs estado final).
4. CLAUDE.md (raiz) — convenções do projeto, versões da VM, seção "Auditoria do
   Modelo de Vídeo".
5. EXIST_VIDEOS/video_core.py — em especial load_frames, load_audio_features,
   frames_dir, audio_dir, EXISTVideoDataset, build_model, restore_videomae_qv_biases.
6. EXIST_VIDEOS/extract_video.ipynb e extract_audio.ipynb — os originais, para
   referência. NÃO MODIFICAR.

## Onde estamos

P0 (pré-condições) JÁ PASSOU na VM em 2026-08-09. Confirmado por medição:

  disco livre           49,8 GB  (precisa ~31 GB)
  frames_v1_sample/     20 treino + 5 teste   ← referência do teste T4, NÃO APAGAR
  audio_features/ (v1)  2524 + 674            ← referência do teste 6, NÃO APAGAR
  árvores v2            INEXISTENTES          ← condição do V7 (skip resume-safe)
  torch 2.6.0+cu124 · transformers 5.6.0 · numpy 1.26.4 · cv2 4.13.0
  openai-whisper 20250625 · CUDA disponível

  VideoMAEImageProcessor is VideoMAEImageProcessorFast  →  True  (renomeação pura)
  processor: shortest_edge=224 · crop 224x224 · resample=2 (bilinear)
             mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225) · 4 flags True
             MRO: VideoMAEImageProcessor → TorchvisionBackend → BaseImageProcessor

Próximo passo é o P1 do plano.

## Fatos do ambiente que você não vai inferir sozinho

- O repositório local é um MOUNT SSHFS da VM (a67606@172.31.16.19:/home/a67606/Thesis).
  Escrever local = escrever na VM. Mas IO pesado sobre o mount é lentíssimo (~4,4 s para
  ler um .pt de 9,6 MB; `du -sh` dá timeout). Qualquer comando pesado deve ser escrito
  para o USUÁRIO rodar numa célula do Jupyter NA VM. Para ler muitos .pt você mesmo, use
  leitura parcial via zipfile (um .pt é um zip; os primeiros 256 KB do maior membro
  custam ~0,55 s) — o padrão está em analysis/frames_zero_scan.json.
- VOCÊ NÃO TEM GPU. A extração (~45 min) e os re-treinos (~3,5 h) rodam na VM, lançados
  pelo usuário. Você escreve o código e verifica os artefatos que ele produzir. NUNCA
  afirme ter verificado algo que exige GPU.
- As versões LOCAIS divergem das da VM (local: torch 2.9.1, transformers 4.57.6, sem
  peft, sem pyevall). Nunca decida algo dependente de API a partir do ambiente local.
- EXIST_VIDEOS/frames/ v1 FOI APAGADO. Não escreva nada que dependa dele existir.

## O time de agentes

Crie DOIS agentes, ambos com janela de contexto limpa:

**Agente IMPLEMENTADOR** — escreve o código do plano:
  - extract_video_v2.ipynb e extract_audio_v2.ipynb (§5 e §6 do plano)
  - as alterações C1–C4 em video_core.py (§7 do plano, funções de substituição integrais)
  - segue o plano literalmente; onde o plano deixar algo à sua escolha, escolhe o
    conservador e REGISTA a decisão
  - NÃO pode modificar: extract_video.ipynb, extract_audio.ipynb,
    Thesis_video_fixed.ipynb, Thesis_video_baseline.ipynb, inference_video.py,
    ablation_harness.py, runs_baseline/, os relatórios 0X_*.md, nem apagar
    frames_v1_sample/ ou audio_features/

**Agente VALIDADOR** — verifica a implementação CONTRA O PLANO, item por item:
  - para cada célula do plano, confirma que o código escrito corresponde
  - confirma que os notebooks são nbformat válido, sem outputs, e que todas as células
    de código parseiam com ast
  - confirma que video_core.py compila (py_compile) e não executa nada no import
  - confirma que nenhum arquivo proibido foi tocado (git status + mtime + git diff)
  - confirma que os GATES do plano estão presentes no código, não só documentados
  - reporta divergências com localização exata; não conserta, reporta
  - NÃO aceita nada por relato do implementador — só por leitura do arquivo

Os dois precisam conversar: quando o validador achar divergência, o implementador
corrige e o validador re-verifica. Itere até fechar. Você (orquestrador) medeia.

No fim, VOCÊ faz uma verificação geral independente: não aceite o relatório de nenhum
dos dois sem confirmar você mesmo os itens de maior consequência, rodando comando.

## Gates que NÃO podem ser pulados

1. **T4b (probe de equivalência de normalização)** — célula 3 do extract_video_v2.
   É o gate de maior risco de todo o plano. Aborta se maxΔ > 0,017 contra
   frames_v1_sample/. Rodar ANTES da extração em massa.
   Risco residual conhecido: com do_rescale=False o resize bilinear pode acontecer
   sobre uint8 em vez de float, e a interpolação arredonda diferente. Se o T4b
   acusar, a saída é manter do_rescale=True e multiplicar por 255 no fim.
2. **Árvores v2 têm de nascer vazias** — o skip resume-safe (`if out_path.exists():
   continue`) só é correto assim. Se uma tentativa anterior abortou por BUG, apagar
   antes de re-rodar; se abortou por queda/disco, o resume é legítimo.
3. **Teste 6 (norma do áudio)** — v1 tem norma média 14,96 ± 0,50. Se a v2 der
   praticamente o mesmo número, a correção NÃO pegou e o script pulou tudo em
   silêncio. É o teste que detecta o modo de falha do V7.
4. **Gate de 5% no preenchimento parcial** — se >5% dos vídeos precisarem de frame de
   preenchimento, PARAR e reportar: deixou de ser cauda e virou sintoma.

## Decisões já tomadas — não reabrir

- 64 frames por vídeo, uint8, árvores v2/, arquivos novos _v2 com originais intocados
- V4-MEM: dois passes com grab()/retrieve(), memória O(64)
- V5-NORM: mesmo processor com do_rescale=False, do_normalize=False, nada reimplementado
- C2: desnormalização no Dataset (load_frames), não no forward
- Preenchimento parcial pelo frame real mais próximo, gate 5%
- Índices com .round() (não truncamento)
- N_FRAMES_USED=16 subamostrados UNIFORMEMENTE no primeiro re-treino (uma variável de
  cada vez); 32/64 e janela densa são braços de ablação futuros

## Depois da extração (o usuário roda na VM)

Rodar os 8 testes de verificação da §9 do plano. Só então os re-treinos, em 3 braços:

  A'  frames v2 + áudio v1 + restore_biases=False   → frames+código (bloco único)
  A   frames v2 + áudio v1 + restore_biases=True    → A − A'  = ARCH-01
  B   frames v2 + áudio v2 + restore_biases=True    → B − A   = áudio
                                                      B − baseline = total

  Protocolo de épocas: UMA calibração de 15 épocas na configuração B → congelar
  E = argmin(eval_loss) → os TRÊS braços rodam com esse mesmo E. Estado final contra
  estado final, sempre (regra de simetria do 09_BASELINE_RESULTS.md §2).

## Regras do projeto

- Fale sempre em português do Brasil.
- Distinga fato de hipótese. Todo número precisa de fonte. Se não verificou, diga.
- Nunca invente métricas. Nenhum treino novo rodou ainda.
- Anti-leakage: threshold só pós-treino; o split congelado (Dataset/val_split_42.json,
  2019/505) nunca é recalculado.
- Prefira funções de substituição completas a diffs.
- Explique decisões de design antes de escrever código.
- Se algo ficar ambíguo ou aparecer algo que o plano não previu e que mude o trabalho
  materialmente, PERGUNTE ao autor em vez de decidir sozinho.
```
