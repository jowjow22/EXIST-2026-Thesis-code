# Context Report — Investigação da baixa contribuição do ramo de vídeo (Task 3, EXIST 2026)

**Export date:** 2026-08-12
**Messages in this session:** ~30 trocas substantivas
**Report scope:** Full context — no compression applied
**Idioma do projeto:** português do Brasil (regra do autor)

---

## 1. Project Overview

Tese de mestrado de Jonathan Santos da Silva (IPB/UTFPR) sobre deteção multimodal de
sexismo sob o paradigma *Learning with Disagreement* (LeWiDi), submetida ao EXIST 2026
(CLEF 2026), equipa **Jow**. A **Task 3** é deteção de sexismo em vídeos do TikTok, com 3
subtasks (A binária, B intenção da fonte em 3 classes, C multilabel de 6 categorias),
avaliadas por ICM-Norm (hard e soft) via PyEvALL.

Esta sessão foi de **validação e medição**, não de implementação nem de replanejamento: o
objetivo era descobrir **por que o ramo de vídeo contribui tão pouco** no sistema da Task 3,
executando os cinco testes especificados na §5 de
`EXIST_VIDEOS/improvements/15_VIDEO_INVESTIGATION_HANDOFF.md`.

**A investigação está COMPLETA.** Os 5 testes especificados rodaram, mais 2 testes
adicionais propostos e implementados nesta sessão (§8 comparação de resumos, §8b retorno da
adaptação LoRA). Todas as rotas de correção testáveis sem treino foram fechadas por
medição.

### 1.1 Original Goal

O prompt de abertura da sessão pedia, textualmente:

1. Ler `15_VIDEO_INVESTIGATION_HANDOFF.md`, `14_VIDEO_CONTRIBUTION_ANALYSIS.md`,
   `CLAUDE.md` e `video_core.py`.
2. **Postura exigida:** *"Não trate nada como verdade estabelecida."* Reanalisar o código
   por conta própria e dizer, item a item, o que confirma, o que corrige e o que não
   consegue verificar. As marcações `[VERIFICADO]` / `[RELATADO]` / `[HIPÓTESE]` dos
   documentos dizem o que foi checado; `[RELATADO]` veio de um agente e **não** foi
   reconferido.
3. Dois pontos a checar com ceticismo explícito:
   - A afirmação de que a fusão "não ignora o vídeo" (norma de coluna 0,749 vs 0,630 do
     texto) — tem caveat declarado: contribuição = peso × escala da feature, e as escalas
     nunca foram medidas. **A conclusão pode inverter.**
   - A afirmação de que o `TemporalAttentionPool` aplica softmax sobre 1 568 tokens
     espaço-temporais, não sobre 8 posições temporais. Muda a descrição de arquitetura da
     tese.
4. Produzir `EXIST_VIDEOS/probe_video_contribution.ipynb` implementando os 5 testes.
5. Duas coisas que valem sair independentemente do resultado: **a correção da descrição do
   pooling na tese** e **a lição metodológica das sondas** (zerar entrada normalizada
   engana; permutação mede certo).

### 1.2 Current Objective

Escrever `EXIST_VIDEOS/improvements/16_VIDEO_INVESTIGATION_RESULTS.md` consolidando:

- os 7 testes com todos os números,
- as 8 correções aos documentos 14 e 15 (3 delas correções de afirmações minhas próprias),
- as 4 correções de arquitetura para a tese,
- o achado novo do *nuisance* de duração em 3 modalidades,
- a lição metodológica das sondas.

O usuário confirmou o interesse mas a sessão terminou antes de o documento ser escrito.
**O documento 16 NÃO existe ainda.**

---

## 2. Environment & Stack

| Component | Details |
|-----------|---------|
| OS (dev local) | macOS, Darwin 25.5.0 |
| OS (VM de treino) | Linux (Jupyter remoto) |
| Python | 3.11 (VM) |
| Diretório do projeto (VM) | `/home/a67606/Thesis/EXIST_VIDEOS` |
| Diretório do projeto (local) | `/Users/jonathansantosdasilva/Documents/projects/VM-THESIS` |
| Ambiente Jupyter (VM) | `/home/a67606/jupyter-env/` |
| GPU | **só na VM.** A máquina local NÃO tem GPU |
| Mount | O repositório local é um **mount SSHFS** da mesma árvore da VM |
| `torch` | 2.6.0+cu124 |
| `torchvision` | 0.21.0+cu124 |
| `transformers` | **5.6.0** |
| `accelerate` | 1.13.0 |
| `peft` | 0.19.1 |
| `safetensors` | 0.7.0 |
| `huggingface_hub` | 1.11.0 |
| `numpy` | 1.26.4 |
| `pandas` | 2.2.3 |
| `scikit-learn` | 1.8.0 |
| `scipy` | 1.17.1 |
| `PyEvALL` | 0.1.78 |
| `openai-whisper` | 20250625 |
| `opencv` (`cv2`) | 4.13.0 |
| `Pillow` | 12.1.1 |

### 2.1 Restrições operacionais do ambiente (CRÍTICAS)

- **O mount SSHFS é lento.** NÃO ler arquivos `.pt` grandes, NÃO usar `du`, NÃO fazer
  `find` recursivo em `Dataset/`, `frames/`, `audio_features/`. Ler código e JSONs pequenos
  é barato. Ler CSVs de ~200 KB é barato.
- **`mtime` do mount NÃO é confiável.** Observado nesta sessão: o `.ipynb` reportava
  `mtime = 2026-08-09 21:07` enquanto continha edições feitas em `2026-08-11`. Verificar
  sempre pelo **conteúdo** (grep de marcadores + sha256), nunca pelo `mtime`.
- **Versões locais divergem das da VM.** Nunca decidir nada dependente de API pelo ambiente
  local. Se algo precisa de verificação decisiva, escrever arquivo de teste e pedir ao
  usuário para executar na VM.
- **Nunca afirmar ter verificado algo que exige GPU.**

---

## 3. Architecture & Design

### 3.1 Component Map — sistema da Task 3 (`EXIST_VIDEOS/video_core.py`, 2 056 linhas)

```
frames [16,3,224,224]  →  VideoMAE-base + LoRA(r=16,α=32,q/v)  →  last_hidden_state [B,1568,768]
                                                                        ↓
                                                          TemporalAttentionPool  →  vid_feat [768]

audio [1280] (Whisper-large-v3 pré-extraído)  →  Linear(1280→768)+GELU+LayerNorm  →  aud_feat [768]

texto (transcrição, max_length=256)  →  XLM-R-large + LoRA(r=16,α=32,q/v)  →  CLS  →  txt_feat [1024]

bio [16] + cvpc [1]  →  (−baseline per-viewer)  →  bio_mlp 17→128→128  →  bio [128]

        cat([vid 768, aud 768, txt 1024, bio 128]) = 2688
                        ↓  Linear(2688→1024) → GELU → Dropout(0.1)     ← SEM LayerNorm (regra)
                     fused [1024]
                        ↓
        head_a Linear(1024→1)   head_b Linear(1024→3)   head_c Linear(1024→6)
```

**Offsets dos slots na entrada da fusão** (de `parts = [vid_feat, aud_feat, txt_feat, bio]`,
`video_core.py:1135-1146`):

| slot | fatia |
|---|---|
| vídeo | `0:768` |
| áudio | `768:1536` |
| texto | `1536:2560` |
| bio | `2560:2688` |

**Sem cross-attention** (ARCH-03b). A original atendia sobre **uma única key** (o vídeo já
colapsado pelo pooling): softmax sobre 1 elemento = 1,0. Não era atenção, era mapa linear
fixo, e consumia 4 987 904 params = 45 % dos 11,0 M treináveis de então.

### 3.2 Data Flow — treino e avaliação

- Split congelado `Dataset/val_split_42.json`: **2 019 treino / 505 val**, `SPLIT_SEED=42`,
  estratificação por `task_a`. **NUNCA recalcular.**
- `SPLIT_SEED` (fixo, artefato commitado) separado de `INIT_SEED` (OPT-06).
- Batch efetivo 32 (micro 2 × accum 16), bf16, gradient checkpointing.
- `LOAD_BEST_MODEL_AT_END = False`. Época escolhida por `E = argmin(eval_loss)`, **nunca**
  pela métrica reportada (leakage).
- `MODALITY_DROPOUT_P = 0.05`, por amostra, com faixas posicionais
  `band = clamp(floor(r/p), max=4)`: 0=vídeo, 1=áudio, 2=texto, 3=bio, 4=nada.
- Threshold da Task C ajustado **só pós-treino**, numa metade da val, reportado na outra
  (`tune_task_c_thresholds`, `split_val_halves`).

### 3.3 Key Design Decisions desta sessão

- **Um forward + replay exato (decisão nº 1).** O `forward` do `EXISTVideoModel` em `eval()`
  não tem **nenhuma operação entre amostras** (`keep_* = None`, dropouts off, e é
  **proibido** por regra do projeto ter LayerNorm/BatchNorm depois do `cat` —
  `video_core.py:1042-1046`). Logo cada `part` é função determinística **só** das entradas da
  própria amostra, e **permutar uma modalidade na entrada é idêntico a permutar o slot dela
  no vetor concatenado**. Um único forward cacheando a entrada da fusão permite replay
  *exato* dos testes 1, 2 e 5, rodando só `fusion → heads`. Custo: ~45 GPU-min → 1 forward.
  **Alternativa rejeitada:** rodar cada arm com forward completo (custo proibitivo, sem
  ganho de precisão).

- **O replay fica na GPU, em bf16 (decisão nº 2).** Replay em fp32 na CPU a partir de
  features bf16 introduziria divergência de dtype exatamente na cabeça B, que é a hipótese
  sob teste (H3, fragilidade de fronteira). Só os testes 4/5 (sklearn, matriz de flips) são
  de facto CPU.

- **Captura num único ponto (decisão nº 3).** Um `forward_hook` em `model.fusion[0]` captura
  a **entrada** da `Linear` — o vetor concatenado como o modelo o fundiu, no dtype exato. Os
  quatro slots saem por fatias. **Alternativa rejeitada:** quatro hooks separados (um por
  ramo), porque podem dessincronizar; um não pode.

- **Portões duros, decompostos em três.** A primeira versão tinha um assert cego que
  confundia três causas distintas. Decomposto em:
  - **A** — logits do replay == logits do forward (mesmo cache): **aborta** se falhar. É a
    única condição de validade do replay.
  - **B** — métricas do forward == `final_metrics.json`: **avisa**, não aborta.
  - **C** — replay+derangement == `test.ipynb` (0,4106 / 0,4316): **avisa**.

  Razão: B e C comparam com números de **outra sessão**. Se divergirem, o replay ainda é
  válido e todos os **Δ** continuam decidindo as hipóteses — elas dependem de comparação
  **entre arms da mesma sessão**, não de valor absoluto. Abortar nesse caso perderia a
  sessão por uma diferença que não afeta conclusão nenhuma.

- **Autocalibração do modo de replay.** 4 modos (`bs` ∈ {2, 64} × autocast bf16 on/off)
  testados contra os logits cacheados; ganha o que reproduz as decisões. Razão: `bs=2` casa
  com `per_device_eval_batch_size=2` do `predict`; matmul bf16 reduz em tiles, então mudar o
  batch muda o último bit — e a cabeça B é fragilidade de fronteira por hipótese, logo o
  último bit pode virar um `argmax`. Autocast on/off porque não é dado que o `Trainer` da v5
  autocaste no `predict`.

- **Teste 2 ganhou dois arms que não estavam na spec.** Ruído isotrópico só é decisivo numa
  direção: se ruído ≥ permutação, H3 confirmada; se ruído < permutação, é **ambíguo**
  ("permutação é in-distribution-mas-errada" vs "a cabeça é mais robusta a direções OOD").
  Adicionados: `perm_within` (derangement **dentro da mesma classe gold de B**) e
  `noise_within` (magnitude casada com `perm_within`). Razão do segundo: "machuca menos" não
  vale nada se estiver confundido com "foi perturbado menos".

- **Nulo por permutação nas sondas lineares.** Com n≈252 no report e d=768 (ou 6 144), uma
  sonda decora. Sem o nulo, "duração é previsível" pode ser artefato de dimensionalidade.
  10 repetições com rótulos embaralhados; o número lido é **AUC − AUC_nulo**. É o nulo que
  absorve a vantagem de dimensionalidade do `mean_temporal` (6 144 features).

- **Escala de ruído obrigatória.** Além das metades fixas de `vc.split_val_halves`
  (comparáveis, anti-leakage), 8 divisões 50/50 aleatórias dão média ± desvio do AUC.
  Comparar 0,069 com 0,11 sem barra de erro não significa nada.

- **Regra de leitura PRÉ-REGISTADA, escrita antes de ver qualquer número.** As duas
  permutações já medidas deram Task B hard −0,086 e −0,065 (amplitude 0,021 sobre efeito de
  −0,076). Portanto: (1) todo arm reportado como **média ± amplitude** sobre 5 derangements;
  (2) diferença entre arms só conta se **exceder a amplitude**; (3) abaixo disso a leitura
  honesta é **inconclusivo a n=505**.

- **Só o ramo de vídeo roda no §8/§8b.** Precisa do `last_hidden_state`, não do XLM-R.
  `model.videomae` + `model.video_pool.attn` corta o forward em ~2/3.

- **Os tokens não são guardados.** 505 × 1 568 × 768 × 4 B = 2,4 GB. Guardadas só as 4
  variantes de resumo: ~17 MB por condição.

- **No §8b, só resumos SEM parâmetros contam.** O `pool_learned` foi treinado sobre tokens
  **adaptados**; aplicá-lo a tokens base confundiria "os tokens mudaram" com "o pool está
  descasado com a entrada". A comparação válida usa `mean_all` e `mean_temporal`, operações
  idênticas nas duas condições. O `pool_learned` sai na tabela **rotulado como confundido**.

- **Portão contra no-op silencioso no §8b.** Se `disable_adapter()` não tivesse efeito, tudo
  pareceria "o LoRA não faz diferença" — indistinguível da conclusão verdadeira, e levaria à
  decisão errada. A célula mede a distância entre tokens base e adaptados e aborta se forem
  iguais.

- **Log em arquivo via tee de `sys.stdout`.** O Jupyter do usuário não grava outputs no
  `.ipynb` (causa não determinada). Solução: duplicar `sys.stdout` para
  `analysis/probe_report.txt` na célula 0, fechar na última célula com guarda `isinstance`
  para ser idempotente.

---

## 4. File & Directory Structure

```
/Users/jonathansantosdasilva/Documents/projects/VM-THESIS/      (= /home/a67606/Thesis/ na VM)
├── CLAUDE.md                                  # contexto de projeto, versões da VM, histórico de auditoria
├── EXIST_MEMES/
│   ├── memes_core.py                          # sistema da Task 2 (memes) — referência de comparação
│   └── memes_model.ipynb
└── EXIST_VIDEOS/
    ├── video_core.py                          # 2056 linhas — TODO o sistema da Task 3
    ├── probe_video_contribution.ipynb          # ★ ENTREGÁVEL desta sessão — 34 células, 18 de código
    ├── test.ipynb                              # sondas originais (permutação de frames + zerar vid_feat)
    ├── Thesis_video_baseline.ipynb             # baseline congelado
    ├── Thesis_video_improved.ipynb             # sistema corrigido
    ├── Dataset/
    │   ├── val_split_42.json                   # SPLIT CONGELADO — nunca recalcular
    │   ├── val_seed_2025.json
    │   ├── training/EXIST2026_training.json
    │   └── test/
    ├── run_B2_bio_viewer_seed42_B15/           # ★ CHECKPOINT SONDADO
    │   ├── trainable.safetensors               # 24 144 460 bytes, 167 tensores
    │   ├── meta.json
    │   ├── final_metrics.json
    │   ├── fingerprints.json
    │   ├── scaler.pkl
    │   ├── task_c_thresholds.json
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    ├── run_B2_bio_viewer_seed42_A/             # braços de atribuição
    ├── run_B2_bio_viewer_seed42_A_prime/
    ├── run_B2_bio_viewer_seed42_B/
    ├── run_B2_bio_viewer_seed42_CAL/
    ├── runs_baseline/
    ├── analysis/
    │   ├── probe_report.txt                    # ★ 195 947 bytes — LOG COMPLETO da última run
    │   ├── audio_durations.csv                 # split,id_EXIST,file,status,duration_s
    │   ├── truncation_scan_v2.csv
    │   ├── frames_zero_scan.json
    │   ├── audio_duration_analysis.ipynb
    │   ├── arch04_audio_cdf.png
    │   ├── arch04_audio_durations.png
    │   └── probe_cache_B15/                    # ★ CACHES DAS SONDAS
    │       ├── val_cache.pt                    # 5 511 010 B — fusion_in, logits, labels, pool_stats
    │       ├── pool_variants.pt                # 17 067 008 B — 4 resumos, COM LoRA
    │       ├── pool_variants_base.pt           # 17 067 048 B — 4 resumos, SEM LoRA
    │       └── task_b_flips.csv                # 208 813 B — 574 flips com transcrição
    └── improvements/
        ├── 05_IMPLEMENTATION_NOTES.md
        ├── 09_BASELINE_RESULTS.md
        ├── 10_EXTRACTION_FIXES_SPEC.md
        ├── 11_EXTRACTION_IMPLEMENTATION_PLAN.md
        ├── 13_NOTAS_PARA_TESE.md
        ├── 14_VIDEO_CONTRIBUTION_ANALYSIS.md   # 464 linhas — hipóteses H1–H6
        ├── 15_VIDEO_INVESTIGATION_HANDOFF.md   # 267 linhas — spec dos 5 testes
        ├── 16_VIDEO_INVESTIGATION_RESULTS.md   # ⏳ NÃO EXISTE — próximo passo
        └── context-report-video-contribution-investigation.md   # este arquivo
```

### 4.1 Key Files

| File | Status | Role |
|------|--------|------|
| `EXIST_VIDEOS/probe_video_contribution.ipynb` | ✅ Completo, rodado até o fim | Entregável: 7 testes. 34 células (18 código). bytes 95 451, sha256 `a7b07270b3b22973` |
| `EXIST_VIDEOS/analysis/probe_report.txt` | ✅ 195 947 bytes | Log completo da run final. **Fonte de todos os números deste relatório** |
| `EXIST_VIDEOS/analysis/probe_cache_B15/*.pt` | ✅ | Caches idempotentes; re-rodar salta os forwards |
| `EXIST_VIDEOS/analysis/probe_cache_B15/task_b_flips.csv` | ✅ | 574 flips com `gold_b`/`pred_normal`/`pred_perm`/`donor_gold_b`/`transcript` |
| `EXIST_VIDEOS/improvements/16_VIDEO_INVESTIGATION_RESULTS.md` | ⏳ Pendente | **Próxima ação** |
| `EXIST_VIDEOS/video_core.py` | ✅ Intocado nesta sessão | Sistema. **Nada foi editado aqui** |
| `EXIST_VIDEOS/improvements/14_*.md`, `15_*.md` | ✅ Intocados | Precisam das 8 correções, ainda não aplicadas |

⚠ **O script gerador do notebook (`gen_probe_nb.py`) vivia no scratchpad da sessão e é
efêmero — não existe mais.** O `.ipynb` no disco é a única fonte de verdade do código.
Edições futuras devem ser feitas no notebook diretamente ou via `NotebookEdit`.

---

## 5. Complete Code & Configurations

Todo o código abaixo está no notebook `EXIST_VIDEOS/probe_video_contribution.ipynb`, na
ordem de execução. Reproduzido aqui na íntegra para as partes que definem o método.

### 5.1 Célula 0 — chdir + tee de log

```python
# P1-fix: os caminhos deste notebook são RELATIVOS a EXIST_VIDEOS/ (plano 11 §3).
# Se o kernel nascer noutro cwd, os asserts abaixo falham. Idempotente.
import os
from pathlib import Path

_TARGET = Path("/home/a67606/Thesis/EXIST_VIDEOS")
if Path.cwd() != _TARGET:
    assert _TARGET.is_dir(), f"diretório do projeto não encontrado: {_TARGET}"
    os.chdir(_TARGET)
assert Path("Dataset").is_dir(), f"cwd errado: {Path.cwd()} (sem Dataset/)"
assert Path("video_core.py").is_file(), "video_core.py não está no cwd"
print("cwd:", Path.cwd())

# ── Log em arquivo (independente de o Jupyter gravar outputs ou não) ────────
# O stdout é DUPLICADO: continua aparecendo na célula E vai para o arquivo.
# `_ORIG_STDOUT` guardado em globals() para re-executar esta célula não empilhar Tees.
import sys

_ORIG_STDOUT = globals().get("_ORIG_STDOUT", sys.stdout)


class _Tee:
    def __init__(self, path, term):
        self.f = open(path, "w", buffering=1, encoding="utf-8")
        self.term = term

    def write(self, s):
        self.term.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.term.flush()
        self.f.flush()

    def isatty(self):
        return False


Path("analysis").mkdir(exist_ok=True)
LOG_PATH = "analysis/probe_report.txt"
sys.stdout = _Tee(LOG_PATH, _ORIG_STDOUT)
print("[log] stdout duplicado para", LOG_PATH,
      "\n[log] stderr NAO e capturado — se houver traceback, copiar da tela")
```

### 5.2 Célula 1 — imports, checkpoint, split congelado

```python
import json, math, warnings
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

import video_core as vc
from transformers import XLMRobertaTokenizer, TrainingArguments

warnings.filterwarnings("ignore", category=UserWarning)

RUN_DIR    = Path("run_B2_bio_viewer_seed42_B15")
CACHE_DIR  = Path("analysis/probe_cache_B15")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "val_cache.pt"

N_DERANGEMENTS = 5
DERANGEMENT_SEEDS = list(range(N_DERANGEMENTS))     # 0..4; 0 e 1 = os de test.ipynb

# Offsets dos slots — lidos de video_core.py:1135-1146 (parts = [vid, aud, txt, bio]).
SLOTS = {
    "video": (0, vc.VIDEO_DIM),                                       # 0:768
    "audio": (vc.VIDEO_DIM, vc.VIDEO_DIM + vc.AUDIO_PROJ_DIM),        # 768:1536
    "texto": (vc.VIDEO_DIM + vc.AUDIO_PROJ_DIM,
              vc.VIDEO_DIM + vc.AUDIO_PROJ_DIM + vc.TEXT_DIM),        # 1536:2560
    "bio":   (vc.VIDEO_DIM + vc.AUDIO_PROJ_DIM + vc.TEXT_DIM,
              vc.fusion_input_dim(True)),                             # 2560:2688
}

METRIC_KEYS = [
    "task_a_icmnorm_hard", "task_a_icmsoftnorm_soft",
    "task_b_icmnorm_hard", "task_b_icmsoftnorm_soft",
    "task_c_icmnorm_hard", "task_c_icmsoftnorm_soft",
]

model, scaler = vc.load_run(RUN_DIR, init_seed=42)
with open(RUN_DIR / "meta.json") as f:
    META = json.load(f)
with open(RUN_DIR / "final_metrics.json") as f:
    FINAL = json.load(f)

print("\nmeta do checkpoint sondado:")
for k in ("condition", "arm", "restore_biases", "frames_version", "audio_version",
          "num_train_epochs", "fusion_in", "split_seed"):
    print(f"  {k:18s} {META.get(k)}")
assert META["fusion_in"] == vc.fusion_input_dim(True) == 2688, "fusion_in inesperado"
assert META["restore_biases"] is True, "este checkpoint NÃO tem ARCH-01 corrigido"

tokenizer = XLMRobertaTokenizer.from_pretrained(vc.XLMR_NAME)
full_df = pd.read_json("Dataset/training/EXIST2026_training.json").transpose()
train_df, val_df = vc.build_frozen_split(full_df)
val_ds = vc.EXISTVideoDataset(val_df, tokenizer, scaler,
                              vc.frames_dir("train"), vc.audio_dir("train"))
N_VAL = len(val_ds)
VAL_IDS = [str(i) for i in val_df["id_EXIST"].values]
print(f"\nval = {N_VAL} amostras (split congelado seed {vc.SPLIT_SEED})")
assert N_VAL == len(VAL_IDS) == 505, f"val tem {N_VAL}, esperado 505 (split congelado)"
```

### 5.3 Célula 2 — Stage A: um forward cacheando tudo

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", ("Stage A exige GPU. Este notebook roda na VM, não na máquina "
                          "local de desenvolvimento.")

args = TrainingArguments(output_dir="/tmp/probe_vid", per_device_eval_batch_size=2,
                         bf16=True, remove_unused_columns=False, report_to="none",
                         label_names=["labels_a", "labels_b", "labels_c"])
trainer = vc.MultiTaskTrainer(model=model, args=args,
                              compute_metrics=vc.compute_metrics,
                              data_collator=vc.video_collate)
trainer.label_names = ["labels_a", "labels_b", "labels_c"]

# 16 frames / tubelet 2 = 8 posições temporais × 196 patches 14×14 = 1568 tokens (F6).
N_TEMPORAL = vc.NUM_FRAMES // 2
N_SPATIAL  = (vc.FRAME_H // 16) * (vc.FRAME_W // 16)
assert N_TEMPORAL * N_SPATIAL == 1568, (N_TEMPORAL, N_SPATIAL)


def _run_stage_a():
    buf_fusion, buf_pool = [], []

    def hook_fusion(_m, inp, _out):
        buf_fusion.append(inp[0].detach().float().cpu())

    def hook_pool(_m, _inp, out):
        # out = logits [B, 1568, 1] — o MESMO tensor de video_core.py:965.
        lg = out.detach().float()
        w = torch.softmax(lg, dim=1).squeeze(-1)                 # [B, 1568]
        ent = -(w.clamp_min(1e-12).log() * w).sum(dim=1)         # nats
        top10 = w.topk(10, dim=1).values.sum(dim=1)
        temporal = w.view(w.shape[0], N_TEMPORAL, N_SPATIAL).sum(dim=2)   # [B, 8]
        buf_pool.append((torch.stack([ent, w.max(dim=1).values, top10], dim=1).cpu(),
                         temporal.cpu()))

    h1 = model.fusion[0].register_forward_hook(hook_fusion)
    h2 = model.video_pool.attn.register_forward_hook(hook_pool)
    try:
        out = trainer.predict(val_ds)
    finally:
        h1.remove(); h2.remove()

    logits_a, logits_b, logits_c = vc._unpack_predictions(out.predictions)
    lab_a, lab_b, lab_c = out.label_ids

    cache = {
        "fusion_in": torch.cat(buf_fusion, dim=0),                       # [505, 2688]
        "logits_a": np.asarray(logits_a), "logits_b": np.asarray(logits_b),
        "logits_c": np.asarray(logits_c),
        "labels_a": np.asarray(lab_a), "labels_b": np.asarray(lab_b),
        "labels_c": np.asarray(lab_c),
        "pool_stats": torch.cat([b[0] for b in buf_pool], dim=0),        # [505, 3]
        "pool_temporal": torch.cat([b[1] for b in buf_pool], dim=0),     # [505, 8]
        "ids": VAL_IDS,
    }
    torch.save(cache, CACHE_FILE)
    return cache


if CACHE_FILE.exists():
    cache = torch.load(CACHE_FILE, weights_only=False)
    print(f"[cache] carregado de {CACHE_FILE} (forward saltado)")
else:
    cache = _run_stage_a()
    print(f"[cache] gravado em {CACHE_FILE}")

FUSION_IN = cache["fusion_in"]
LAB_A, LAB_B, LAB_C = cache["labels_a"], cache["labels_b"], cache["labels_c"]
assert FUSION_IN.shape == (N_VAL, 2688), FUSION_IN.shape
assert cache["pool_stats"].shape == (N_VAL, 3)
assert cache["ids"] == VAL_IDS, ("ordem do cache divergiu do val_df — cache antigo? "
                                 "apagar analysis/probe_cache_B15/ e re-rodar")
print("shapes OK:", {k: tuple(v.shape) for k, v in cache.items()
                     if hasattr(v, "shape")})
```

### 5.4 Célula 3 — PORTÃO + autocalibração do replay

```python
EvalPred = namedtuple("EvalPred", ["predictions", "label_ids"])

model.eval()
print("model.training =", model.training, "(tem de ser False)")
assert model.training is False

_FUSION_GPU = FUSION_IN.to(device)
REPLAY_BS, REPLAY_AUTOCAST = 2, True          # sobrescritos pela autocalibração abaixo


def replay(fusion_in_gpu, bs=None, use_autocast=None):
    """fusion -> heads. Modo (bs, autocast) vem da autocalibração desta célula."""
    bs = REPLAY_BS if bs is None else bs
    ac = REPLAY_AUTOCAST if use_autocast is None else use_autocast
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if ac
           else torch.autocast("cuda", enabled=False))
    outs = ([], [], [])
    with torch.no_grad(), ctx:
        for i in range(0, fusion_in_gpu.shape[0], bs):
            fused = model.fusion(fusion_in_gpu[i:i + bs])
            outs[0].append(model.head_a(fused).squeeze(-1).float().cpu())
            outs[1].append(model.head_b(fused).float().cpu())
            outs[2].append(model.head_c(fused).float().cpu())
    return tuple(torch.cat(o).numpy() for o in outs)


def metrics_of(logits3):
    m = vc.compute_metrics(EvalPred(logits3, (LAB_A, LAB_B, LAB_C)))
    return {k: float(m[k]) for k in METRIC_KEYS}


def derangement(n, seed):
    """MESMA construção de ShuffledVideo em test.ipynb: permutação sem ponto fixo."""
    rs = np.random.RandomState(seed)
    while True:
        p = rs.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def perturb(slot, index=None, replacement=None):
    """Cópia GPU de FUSION_IN com o slot substituído (permutação de linhas ou tensor)."""
    a, b = SLOTS[slot]
    x = FUSION_IN.clone()
    if index is not None:
        x[:, a:b] = FUSION_IN[np.asarray(index), a:b]
    if replacement is not None:
        x[:, a:b] = replacement.to(FUSION_IN.dtype)
    return x.to(device)


def delta_norm(slot, index=None, replacement=None):
    """||Dslot||2 médio por amostra — magnitude da perturbação realmente aplicada."""
    a, b = SLOTS[slot]
    base = FUSION_IN[:, a:b].float()
    new = base[np.asarray(index)] if index is not None else replacement.float()
    return float((new - base).norm(dim=1).mean())


# ================= autocalibração: qual modo reproduz o forward? =================
REF = (cache["logits_a"], cache["logits_b"], cache["logits_c"])
MODES = {"bs2+bf16": (2, True), "bs2+fp32": (2, False),
         "bs64+bf16": (64, True), "bs64+fp32": (64, False)}
cands = {}
print(f"\n{'modo':>10s}{'maxD logit':>14s}{'flips argmax B':>16s}{'flips A@0.5':>13s}")
for name, (bs, ac) in MODES.items():
    r = replay(_FUSION_GPU, bs=bs, use_autocast=ac)
    mx = max(float(np.abs(r[i] - REF[i]).max()) for i in range(3))
    fb = int((r[1].argmax(1) != REF[1].argmax(1)).sum())
    fa = int(((r[0] > 0) != (REF[0] > 0)).sum())
    cands[name] = (mx, fb, fa, bs, ac)
    print(f"{name:>10s}{mx:14.3e}{fb:16d}{fa:13d}")

best = min(cands, key=lambda k: (cands[k][1] + cands[k][2], cands[k][0]))
mx, fb, fa, REPLAY_BS, REPLAY_AUTOCAST = cands[best]
print(f"\nmodo escolhido: {best}  (bs={REPLAY_BS}, autocast_bf16={REPLAY_AUTOCAST})")

assert fb == 0 and fa == 0, (
    f"PORTAO A FALHOU: nenhum modo de replay reproduz as DECISOES do forward completo "
    f"(melhor: {best}, {fb} flips em B, {fa} em A, maxD logit {mx:.3e}). A equivalencia "
    f"fusion-in -> logits nao se sustenta nesta GPU/versao. Abandonar o replay: rodar os "
    f"testes 1/2/5 com forwards completos (custo ~45 GPU-min) e manter 3/4 sobre o cache.")
if mx > 1e-3:
    print(f"[AVISO] maxD logit = {mx:.3e} sem flip de decisao. Replay valido para metricas "
          f"hard; tratar as metricas SOFT com reserva.")
print(">>> PORTAO A OK: o replay reproduz as decisoes do forward completo.")

# ================= B: esta sessao reproduz final_metrics.json? =================
m_forward = metrics_of(REF)          # métricas pelo MESMO código, sobre os logits reais
m_normal = metrics_of(replay(_FUSION_GPU))

print(f"\n{'metrica':30s}{'forward':>10s}{'replay':>10s}{'final_metrics':>15s}"
      f"{'D repl-fwd':>12s}{'D fwd-final':>13s}")
gate_replay, gate_session = True, True
for k in METRIC_KEYS:
    ref = float(FINAL[f"eval_{k}"])
    d1, d2 = m_normal[k] - m_forward[k], m_forward[k] - ref
    gate_replay &= abs(d1) < 1e-3
    gate_session &= abs(d2) < 1e-3
    print(f"{k:30s}{m_forward[k]:10.4f}{m_normal[k]:10.4f}{ref:15.4f}"
          f"{d1:+12.5f}{d2:+13.5f}")

assert gate_replay, (
    "PORTAO A FALHOU no nivel das METRICAS: os logits do replay reproduzem as decisoes "
    "mas as metricas divergem >1e-3. Isso so pode vir de nao-determinismo dentro do "
    "PyEvALL (mesma entrada, saida diferente) — investigar antes de seguir.")

# ================= C: e os numeros de test.ipynb? =================
REF_SHUF_B_HARD = {0: 0.4106, 1: 0.4316}
print()
gate_shuf = True
for seed, ref in REF_SHUF_B_HARD.items():
    got = metrics_of(replay(perturb("video", index=derangement(N_VAL, seed))))["task_b_icmnorm_hard"]
    gate_shuf &= abs(got - ref) < 1e-3
    print(f"shuf{seed}  task_b_icmnorm_hard  replay={got:.4f}  test.ipynb={ref:.4f}  "
          f"D={got - ref:+.5f}")

SESSION_REPRODUCES = bool(gate_session and gate_shuf)
print()
if SESSION_REPRODUCES:
    print(">>> PORTOES B e C OK: esta sessao reproduz a run de origem. Valores ABSOLUTOS "
          "comparaveis com o doc 15.")
else:
    print("=" * 78)
    print("[AVISO ALTO] Esta sessao NAO reproduz os numeros da run de origem:")
    if not gate_session:
        print("  - metricas do forward != final_metrics.json (coluna 'D fwd-final' acima)")
    if not gate_shuf:
        print("  - permutacao != test.ipynb (0,4106 / 0,4316)")
    print("  O replay esta VALIDO (portao A passou), logo todos os DELTAS deste notebook")
    print("  sao internamente consistentes e as hipoteses H1/H2/H3/H5 continuam decidiveis")
    print("  — elas dependem de comparacao ENTRE arms da mesma sessao, nao de absolutos.")
    print("  MAS: nao colar os valores absolutos ao lado dos do doc 15 sem esta ressalva,")
    print("  e investigar a divergencia (versao de PyEvALL? scaler? split?) antes de")
    print("  qualquer numero deste notebook ir para a tese.")
    print("=" * 78)
```

### 5.5 Célula 4 — matriz de confusão de B em condição normal

```python
LABELS_B = list(vc.LABELS_2_2)
mask_b = LAB_B[:, 0] != vc.SENTINEL_BC
gold_b = LAB_B[mask_b].argmax(axis=1)
pred_b_normal = cache["logits_b"][mask_b].argmax(axis=1)

print(f"amostras com voto válido em B: {int(mask_b.sum())} de {N_VAL}\n")
print("ordem das classes:", LABELS_B, "\n")

cm = np.zeros((3, 3), dtype=int)
for g, p in zip(gold_b, pred_b_normal):
    cm[g, p] += 1
print(f"{'gold|pred':>16s}" + "".join(f"{c:>16s}" for c in LABELS_B) + f"{'total':>8s}")
for i, c in enumerate(LABELS_B):
    print(f"{c:>16s}" + "".join(f"{cm[i, j]:16d}" for j in range(3)) + f"{cm[i].sum():8d}")
print(f"{'total':>16s}" + "".join(f"{cm[:, j].sum():16d}" for j in range(3)) + f"{cm.sum():8d}")

never_pred = [LABELS_B[j] for j in range(3) if cm[:, j].sum() == 0]
print("\nLEITURA:")
if never_pred:
    print(f"  CONFIRMA: a cabeça B NUNCA prediz {never_pred} na val em condição normal.")
    print("  REFUTA o mecanismo declarado da H1 como causa do −0,076 se DIRECT estiver")
    print("  nessa lista: o delta só pode vir de movimento entre as classes restantes.")
    print("  Também explica o fmeasure_hard = NaN sem precisar de hipótese nova.")
else:
    print("  As 3 classes aparecem no argmax — o eixo DIRECT↔JUDGEMENTAL É alcançável,")
    print("  logo o mecanismo da H1 permanece na mesa. O NaN do fmeasure vem de outro")
    print("  motivo (interno ao PyEvALL), não de classe nunca predita.")
print(f"\n  (referência: eval_proxy_task_b_f1 = {FINAL['eval_proxy_task_b_f1']:.4f}, "
      f"eval_task_b_fmeasure_hard = {FINAL['eval_task_b_fmeasure_hard']})")
```

### 5.6 Célula 5 — TESTE 2: ruído pareado, 5 derangements, within-class

```python
rng = np.random.RandomState(20260809)
A_V, B_V = SLOTS["video"]
vid_base = FUSION_IN[:, A_V:B_V].float()

# ---- gold hard de B para o arm within-class (sentinela = grupo próprio) ----
group = np.full(N_VAL, -1, dtype=int)
group[mask_b] = gold_b
groups_desc = {int(g): int((group == g).sum()) for g in sorted(set(group.tolist()))}
print("grupos para permutação within-class (gold hard de B; -1 = sem voto válido):")
for g, n in groups_desc.items():
    name = "SEM_VOTO" if g == -1 else LABELS_B[g]
    print(f"  {name:>16s}  n={n}")


def derangement_within(seed):
    """Derangement dentro de cada grupo. Grupos de tamanho 1 ficam fixos (contados)."""
    rs = np.random.RandomState(1000 + seed)
    idx = np.arange(N_VAL)
    fixed = 0
    for g in set(group.tolist()):
        pos = np.where(group == g)[0]
        if len(pos) < 2:
            fixed += len(pos)
            continue
        for _ in range(500):
            p = rs.permutation(len(pos))
            if not (p == np.arange(len(pos))).any():
                break
        idx[pos] = pos[p]
        fixed += int((idx[pos] == pos).sum())
    return idx, fixed


def gaussian_at(target_norm, seed):
    """Ruído gaussiano com ||e||2 = target_norm em TODA amostra (média casada)."""
    g = torch.from_numpy(np.random.RandomState(2000 + seed).randn(N_VAL, B_V - A_V).astype("float32"))
    g = g / g.norm(dim=1, keepdim=True) * target_norm
    return vid_base + g


arms = {}          # nome -> lista de (metrics, delta)
for s in DERANGEMENT_SEEDS:
    p = derangement(N_VAL, s)
    arms.setdefault("perm_across", []).append(
        (metrics_of(replay(perturb("video", index=p))), delta_norm("video", index=p)))
fixed_total = None
for s in DERANGEMENT_SEEDS:
    p, fixed = derangement_within(s)
    fixed_total = fixed
    arms.setdefault("perm_within", []).append(
        (metrics_of(replay(perturb("video", index=p))), delta_norm("video", index=p)))

d_across = float(np.mean([d for _, d in arms["perm_across"]]))
d_within = float(np.mean([d for _, d in arms["perm_within"]]))
print(f"\namostras sem par no grupo (ficaram fixas no arm within): {fixed_total}")
print(f"||Dvid_feat|| médio — across={d_across:.3f}  within={d_within:.3f}")

for s in DERANGEMENT_SEEDS:
    r = gaussian_at(d_across, s)
    arms.setdefault("noise_across", []).append(
        (metrics_of(replay(perturb("video", replacement=r))), delta_norm("video", replacement=r)))
    r = gaussian_at(d_within, 100 + s)
    arms.setdefault("noise_within", []).append(
        (metrics_of(replay(perturb("video", replacement=r))), delta_norm("video", replacement=r)))


def summarize(arm_name, key):
    vals = np.array([m[key] for m, _ in arms[arm_name]])
    base = m_normal[key]
    return vals.mean() - base, vals.max() - vals.min(), vals.mean()
```

(A célula continua com a impressão da tabela e a árvore de leitura corrigida — ver §5.11.)

### 5.7 Célula 6 — TESTE 1: escada de permutação por modalidade

```python
ladder = {"video": arms["perm_across"]}     # já calculado no teste 2 — não recomputar
for slot in ("audio", "texto", "bio"):
    ladder[slot] = []
    for s in DERANGEMENT_SEEDS:
        p = derangement(N_VAL, s)
        ladder[slot].append((metrics_of(replay(perturb(slot, index=p))),
                             delta_norm(slot, index=p)))
```

### 5.8 Célula 7 — TESTE 3a: escalas e contribuição na fusão

```python
W = model.fusion[0].weight.detach().float().cpu()        # [1024, 2688]
b_fus = model.fusion[0].bias.detach().float().cpu()

contribs = {}
for slot in ("video", "audio", "texto", "bio"):
    a, b = SLOTS[slot]
    f = FUSION_IN[:, a:b].float()
    contrib = (f @ W[:, a:b].T).norm(dim=1)
    contribs[slot] = float(contrib.mean())
tot = sum(contribs.values())
```

### 5.9 Célula 9 — TESTE 3c: entropia do pool

```python
ps = cache["pool_stats"]                      # [505, 3] = (entropia nats, max w, top10)
ent, wmax, top10 = ps[:, 0], ps[:, 1], ps[:, 2]
eff = ent.exp()
H_UNIF = math.log(1568)
tprof = cache["pool_temporal"]                # [505, 8]
```

### 5.10 Célula 10 — TESTE 3d: checagem empírica do reshape 1568→(8,196)

```python
it0, it1 = val_ds[0], val_ds[1]
f0 = it0["frames"].unsqueeze(0).to(device)
f_mix = f0.clone()
f_mix[0, 0:2] = it1["frames"][0:2].to(device)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    h_ref = model.videomae(pixel_values=f0).last_hidden_state.float()
    h_mix = model.videomae(pixel_values=f_mix).last_hidden_state.float()
assert h_ref.shape[1] == 1568, f"esperado 1568 tokens, veio {h_ref.shape[1]}"

d = (h_mix - h_ref).norm(dim=-1).squeeze(0).view(N_TEMPORAL, N_SPATIAL).mean(dim=1).cpu()
```

### 5.11 Célula 5 (continuação) — árvore de leitura CORRIGIDA do teste 2

⚠ A primeira versão desta árvore só comparava `noise_across` vs `perm_across` e retornava
antes de consultar os arms `within`. Ela imprimiu **"CONFIRMA H3"**, veredicto insustentável.
Substituída por:

```python
print("\nLEITURA (Task B ICMNorm hard) — arvore CORRIGIDA em 2026-08-09.")
print("A versao anterior desta celula so comparava noise_across vs perm_across e")
print("retornava antes de consultar os arms within. Rotulava 'indistinguivel' como")
print("'CONFIRMA H3' e ignorava os dois arms que existiam justamente para discriminar.")
print("Nao e violacao de pre-registro: a regra estava incompleta e assimetrica.\n")
for arm, (dm, amp) in (("perm_across", (dp, ap)), ("noise_across", (dn_, an)),
                       ("perm_within", (dw, aw)), ("noise_within", (dnw, anw))):
    print(f"  {arm:14s} D={dm:+.4f} +-{amp:.4f}   ||Dvid_feat||="
          f"{float(np.mean([d for _, d in arms[arm]])):.3f}")
print(f"  tolerancia pre-registada = {tol:.4f}\n")

# --- discriminador PRIMARIO: within vs across, a magnitude praticamente casada ---
print(f"  [1] perm_within vs perm_across: |{dw:+.4f} - {dp:+.4f}| = {abs(dw - dp):.4f} "
      f"{'>' if abs(dw - dp) > tol else '<='} tol")
if abs(dw - dp) > tol and dw > dp:
    print("      REFUTA H3 na forma forte. As duas permutacoes tem ||D|| praticamente")
    print("      igual e a unica coisa que difere entre elas e a CLASSE GOLD DE B DO")
    print("      DOADOR — que o modelo nunca ve. Uma cabeca genericamente fragil nao")
    print("      pode dar dano ~zero sob uma perturbacao dessa magnitude.")
    print("      CONFIRMA o mecanismo da H1: o que o slot de video carrega e informacao")
    print("      de classe de B (registro/formato da fonte).")
    if abs(dw) < 0.01:
        print("      E MAIS: dano ~zero dentro da classe significa que o slot de video NAO")
        print("      carrega nada ALEM da classe de B. Se carregasse intensidade, categoria")
        print("      de C ou detalhe de enquadramento, trocar dentro da classe custaria")
        print("      algo. O ramo de video funciona como um classificador fraco de")
        print("      intencao da fonte, e nada mais. E uma afirmacao de TETO.")
# ... (arms [2] e [3], caveat oracular, correcao do -0,076)
```

### 5.12 Célula 11 — TESTE 4: sondas lineares

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

IDX_FIT, IDX_REP = vc.split_val_halves(N_VAL)
print(f"metades (vc.split_val_halves): fit={len(IDX_FIT)}  report={len(IDX_REP)}")

# ---- alvos ----
dur = pd.read_csv("analysis/audio_durations.csv")
dur = dur[dur["split"] == "training"].copy()
dur["id_EXIST"] = dur["id_EXIST"].astype(str)
dmap = dict(zip(dur["id_EXIST"], dur["duration_s"]))
dur_v = np.array([dmap.get(i, np.nan) for i in VAL_IDS], dtype=float)
assert np.isfinite(dur_v).sum() == N_VAL, (
    f"join de duração incompleto: {int(np.isfinite(dur_v).sum())} de {N_VAL}. "
    "Verificar dtype de id_EXIST e o filtro split=='training'.")

lang_v = val_df["lang"].values
targets = {
    "A_hard (sexista?)": (LAB_A > 0.5).astype(int),
    "B_hard (3 classes)": group,                                    # -1 = sem voto
    "língua (es vs en)": (lang_v == "es").astype(int),
    "duração > mediana": (dur_v > np.median(dur_v)).astype(int),
}


def probe(X, y, seed_shuffle=None):
    """AUC (macro-ovr se multiclasse) na metade de report. Fit só na metade de fit."""
    m = y >= 0 if y.min() < 0 else np.ones(len(y), bool)
    if seed_shuffle is not None:
        y = y.copy()
        y[m] = np.random.RandomState(seed_shuffle).permutation(y[m])
    fit = np.array([i for i in IDX_FIT if m[i]])
    rep = np.array([i for i in IDX_REP if m[i]])
    if len(np.unique(y[fit])) < 2 or len(np.unique(y[rep])) < 2:
        return float("nan")
    sc = StandardScaler().fit(X[fit])
    clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(X[fit]), y[fit])
    p = clf.predict_proba(sc.transform(X[rep]))
    if p.shape[1] == 2:
        return float(roc_auc_score(y[rep], p[:, 1]))
    return float(roc_auc_score(y[rep], p, multi_class="ovr", average="macro",
                              labels=clf.classes_))


feats = {s: FUSION_IN[:, a:b].float().numpy() for s, (a, b) in SLOTS.items()}
probe_out = {}
for name, y in targets.items():
    for s in ("video", "texto", "audio", "bio"):
        auc = probe(feats[s], y)
        null = np.array([probe(feats[s], y, seed_shuffle=1000 + r) for r in range(10)])
        gain = auc - np.nanmean(null)
        probe_out[(name, s)] = (auc, float(np.nanmean(null)), gain)
```

### 5.13 Célula 12 — TESTE 5: matriz de flips da Task B

```python
flip = np.zeros((3, 3), dtype=int)     # pred normal -> pred permutado
flip_rows = []
texts = val_df["text"].values
for s in DERANGEMENT_SEEDS:
    p = derangement(N_VAL, s)
    lb = replay(perturb("video", index=p))[1]
    pn = cache["logits_b"].argmax(axis=1)
    pp = lb.argmax(axis=1)
    for i in range(N_VAL):
        if not mask_b[i]:
            continue
        flip[pn[i], pp[i]] += 1
        if pn[i] != pp[i]:
            flip_rows.append({
                "seed": s, "id_EXIST": VAL_IDS[i], "lang": lang_v[i],
                "gold_b": LABELS_B[int(LAB_B[i].argmax())],
                "pred_normal": LABELS_B[int(pn[i])], "pred_perm": LABELS_B[int(pp[i])],
                "donor_id": VAL_IDS[int(p[i])],
                "donor_gold_b": (LABELS_B[int(LAB_B[p[i]].argmax())]
                                 if mask_b[p[i]] else "SEM_VOTO"),
                "transcript": str(texts[i])[:400],
            })

out_csv = CACHE_DIR / "task_b_flips.csv"
pd.DataFrame(flip_rows).to_csv(out_csv, index=False)
```

### 5.14 Célula 13 — TESTE 6: variantes de resumo dos 1568 tokens

```python
import contextlib

from torch.utils.data import DataLoader

VARIANTS_FILE = CACHE_DIR / "pool_variants.pt"
N_TOP = 5


def _compute_variants(out_path, disable_lora=False):
    """Forward SO do ramo de video, devolvendo as 4 variantes de resumo.

    `disable_lora=True` roda dentro de `model.videomae.disable_adapter()`: ZERA o delta
    LoRA e deixa os pesos base. Os q_bias/v_bias restaurados pelo ARCH-01 sao params BASE
    congelados (nao LoRA) e permanecem — e o que torna a comparacao limpa.
    """
    dl = DataLoader(val_ds, batch_size=2, shuffle=False,
                    collate_fn=vc.video_collate, num_workers=2)   # 0 se der problema
    acc = {k: [] for k in ("pool_learned", "mean_all", "mean_temporal", "top5")}
    lora_ctx = (model.videomae.disable_adapter() if disable_lora
                else contextlib.nullcontext())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), lora_ctx:
        for bi, batch in enumerate(dl):
            h = model.videomae(pixel_values=batch["frames"].to(device)).last_hidden_state
            B = h.shape[0]
            assert h.shape[1] == 1568, h.shape
            lg = model.video_pool.attn(h).squeeze(-1)                  # [B, 1568]
            w = torch.softmax(lg, dim=1)
            acc["pool_learned"].append((w.unsqueeze(-1) * h).sum(1).float().cpu())
            acc["mean_all"].append(h.mean(1).float().cpu())
            acc["mean_temporal"].append(
                h.view(B, N_TEMPORAL, N_SPATIAL, vc.VIDEO_DIM).mean(2)
                 .reshape(B, N_TEMPORAL * vc.VIDEO_DIM).float().cpu())
            vals, idx = w.topk(N_TOP, dim=1)                          # [B, 5]
            tok = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, vc.VIDEO_DIM))
            wn = vals / vals.sum(1, keepdim=True)
            acc["top5"].append((wn.unsqueeze(-1) * tok).sum(1).float().cpu())
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(dl)}", flush=True)
    out = {k: torch.cat(v, dim=0) for k, v in acc.items()}
    torch.save(out, out_path)
    return out


if VARIANTS_FILE.exists():
    VAR = torch.load(VARIANTS_FILE, weights_only=False)
    print(f"[cache] variantes carregadas de {VARIANTS_FILE} (forward saltado)")
else:
    print("[forward] rodando SO o ramo de video sobre a val...")
    VAR = _compute_variants(VARIANTS_FILE)
    print(f"[cache] gravado em {VARIANTS_FILE}")

# ---- PORTAO: pool_learned recomputado == slot de video do cache do Stage A ----
_a, _b = SLOTS["video"]
ref_slot = FUSION_IN[:, _a:_b].float()
got = VAR["pool_learned"].float()
cos = torch.nn.functional.cosine_similarity(got, ref_slot, dim=1)
mx = float((got - ref_slot).abs().max())
print("\nPORTAO da secao 8: cosseno(pool_learned recomputado, slot cacheado)")
print(f"  media={cos.mean():.6f}  min={cos.min():.6f}  maxD absoluto={mx:.4f}")
assert float(cos.mean()) > 0.9995 and float(cos.min()) > 0.995, (
    "PORTAO DA SECAO 8 FALHOU: o pool recomputado NAO reproduz o slot de video do "
    "Stage A. O forward so-video divergiu do forward completo (modo train? dtype?). "
    "Nada desta secao pode ser lido.")
print(">>> PORTAO OK: o forward so-video reproduz o ramo de video do forward completo.")
```

### 5.15 Célula 14 — sondas sobre as variantes (COM LoRA)

```python
NULL_REPS, SPLIT_REPS = 10, 8


def probe_split(X, y, fit, rep, seed_shuffle=None):
    """AUC na metade `rep`, ajustado so na metade `fit`. Mesma config da secao 6."""
    m = y >= 0 if y.min() < 0 else np.ones(len(y), bool)
    if seed_shuffle is not None:
        y = y.copy()
        y[m] = np.random.RandomState(seed_shuffle).permutation(y[m])
    f = np.array([i for i in fit if m[i]])
    r = np.array([i for i in rep if m[i]])
    if len(np.unique(y[f])) < 2 or len(np.unique(y[r])) < 2:
        return float("nan")
    sc = StandardScaler().fit(X[f])
    clf = LogisticRegression(max_iter=1000, C=0.05).fit(sc.transform(X[f]), y[f])
    pr = clf.predict_proba(sc.transform(X[r]))
    if pr.shape[1] == 2:
        return float(roc_auc_score(y[r], pr[:, 1]))
    return float(roc_auc_score(y[r], pr, multi_class="ovr", average="macro",
                               labels=clf.classes_))


RAND_SPLITS = []
for _r in range(SPLIT_REPS):
    _perm = np.random.RandomState(7000 + _r).permutation(N_VAL)
    RAND_SPLITS.append((_perm[:N_VAL // 2], _perm[N_VAL // 2:]))

VNAMES = ["pool_learned", "mean_all", "mean_temporal", "top5"]
Xs = {k: VAR[k].float().numpy() for k in VNAMES}
gains, sds, aucs = {}, {}, {}
for t, y in targets.items():
    for v in VNAMES:
        auc = probe_split(Xs[v], y, IDX_FIT, IDX_REP)
        null = np.array([probe_split(Xs[v], y, IDX_FIT, IDX_REP, seed_shuffle=1000 + k)
                         for k in range(NULL_REPS)])
        rand = np.array([probe_split(Xs[v], y, f, r) for f, r in RAND_SPLITS])
        aucs[(v, t)] = auc
        gains[(v, t)] = auc - float(np.nanmean(null))
        sds[(v, t)] = float(np.nanstd(rand))
```

### 5.16 Células 15-16 — TESTE 7: retorno da adaptação LoRA

```python
BASE_FILE = CACHE_DIR / "pool_variants_base.pt"

print("tipo do encoder de video:", type(model.videomae).__name__)
assert hasattr(model.videomae, "disable_adapter"), (
    "model.videomae nao expoe disable_adapter() — nao e um PeftModel nesta versao do "
    "peft. Sem isso este teste nao pode rodar.")

if BASE_FILE.exists():
    VAR_BASE = torch.load(BASE_FILE, weights_only=False)
    print(f"[cache] variantes SEM LoRA carregadas de {BASE_FILE} (forward saltado)")
else:
    print("[forward] rodando o ramo de video com o adaptador LoRA DESLIGADO...")
    VAR_BASE = _compute_variants(BASE_FILE, disable_lora=True)
    print(f"[cache] gravado em {BASE_FILE}")

# ---- PORTAO: disable_adapter() teve efeito? ----
_ad = VAR["mean_all"].float()
_bs = VAR_BASE["mean_all"].float()
_cos = torch.nn.functional.cosine_similarity(_ad, _bs, dim=1)
_rel = (_ad - _bs).norm(dim=1) / _ad.norm(dim=1).clamp_min(1e-9)
assert float(_rel.mean()) > 1e-3, (
    "PORTAO FALHOU: os tokens com e sem LoRA sao praticamente IDENTICOS "
    f"(desvio relativo medio {float(_rel.mean()):.2e}). disable_adapter() foi no-op — "
    "provavelmente o adaptador nao esta ativo no caminho medido. Um no-op faria tudo "
    "parecer 'o LoRA nao faz diferenca', que e indistinguivel da conclusao verdadeira. "
    "Nada desta secao pode ser lido.")

PARAM_FREE = ["mean_all", "mean_temporal"]
CONFOUNDED = ["pool_learned"]
Xs_base = {k: VAR_BASE[k].float().numpy() for k in PARAM_FREE + CONFOUNDED}

gains_base, sds_base = {}, {}
for t, y in targets.items():
    for v in PARAM_FREE + CONFOUNDED:
        auc = probe_split(Xs_base[v], y, IDX_FIT, IDX_REP)
        null = np.array([probe_split(Xs_base[v], y, IDX_FIT, IDX_REP, seed_shuffle=1000 + k)
                         for k in range(NULL_REPS)])
        rand = np.array([probe_split(Xs_base[v], y, f, r) for f, r in RAND_SPLITS])
        gains_base[(v, t)] = auc - float(np.nanmean(null))
        sds_base[(v, t)] = float(np.nanstd(rand))
```

### 5.17 Célula 17 — fecho do log (IDEMPOTENTE)

```python
# ── fecha o log (IDEMPOTENTE: re-executar esta celula nao levanta) ─────────
sys.stdout.flush()
if isinstance(sys.stdout, _Tee):
    _fh = sys.stdout.f
    sys.stdout = sys.stdout.term
    _fh.close()
    print(f"\n[log] relatorio completo gravado em {LOG_PATH} "
          f"({Path(LOG_PATH).stat().st_size} bytes)")
else:
    print(f"\n[log] tee ja estava fechado (celula re-executada). O log de {LOG_PATH} "
          f"e da PRIMEIRA execucao; para regravar, rodar a celula 0 de novo.")
```

### 5.18 Script de análise do CSV de flips (rodado localmente, sem GPU)

```python
import pandas as pd, numpy as np
d = pd.read_csv("analysis/probe_cache_B15/task_b_flips.csv")
L = ['NONE','JUDGEMENTAL','DIRECT']
print("linhas (flips):", len(d), "| seeds:", sorted(d.seed.unique()),
      "| ids únicos:", d.id_EXIST.nunique())

ct = pd.crosstab(pd.Categorical(d.pred_normal, L), pd.Categorical(d.pred_perm, L),
                 dropna=False).reindex(index=L, columns=L, fill_value=0)
tot = ct.values.sum()
dj = ct.loc['DIRECT','JUDGEMENTAL'] + ct.loc['JUDGEMENTAL','DIRECT']

m = d.donor_gold_b.isin(L)
seg = (d.loc[m,'pred_perm'] == d.loc[m,'donor_gold_b']).mean()
base = d.loc[m,'donor_gold_b'].value_counts(normalize=True)
exp = sum(base.get(c,0)*(d.loc[m,'pred_perm']==c).mean() for c in L)
gold_own = (d.loc[m,'pred_perm'] == d.loc[m,'gold_b']).mean()
gold_norm = (d.loc[m,'pred_normal'] == d.loc[m,'gold_b']).mean()
```

---

## 6. Commands & Execution

```bash
# --- NA VM (Jupyter) ---
# Rodar o notebook: abrir EXIST_VIDEOS/probe_video_contribution.ipynb
# FECHAR A ABA E REABRIR antes de rodar (o Jupyter mantém a versão antiga em memória)
# Restart & Run All

# --- Alternativa de terminal na VM (garante log de texto puro, sem depender do Jupyter) ---
cd /home/a67606/Thesis/EXIST_VIDEOS
jupyter nbconvert --to script probe_video_contribution.ipynb
python probe_video_contribution.py > analysis/probe_run.log 2>&1
tail -5 analysis/probe_run.log

# --- Alternativa mantendo formato notebook (timeout OBRIGATÓRIO: PyEvALL passa de 30 s) ---
cd /home/a67606/Thesis/EXIST_VIDEOS
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output probe_video_contribution_run.ipynb \
  probe_video_contribution.ipynb

# --- LOCAL (leitura pelo mount SSHFS; barato) ---
# Verificar integridade/conteúdo do notebook (NUNCA confiar no mtime)
python3 -c "
import json,hashlib
raw=open('probe_video_contribution.ipynb','rb').read(); nb=json.loads(raw)
src='\n'.join(''.join(c['source']) for c in nb['cells'])
print('bytes:',len(raw),'sha256:',hashlib.sha256(raw).hexdigest()[:16],'células:',len(nb['cells']))
for m in ['## 8. TESTE 6','## 8b. TESTE 7','disable_adapter()','probe_report.txt']:
    print(' ',('PRESENTE' if m in src else 'AUSENTE '),m)"

# Ler seções do log
grep -n "TESTE 7\|PORTAO\|AUC-nulo\|RESUMO —" analysis/probe_report.txt
sed -n '1650,1725p' analysis/probe_report.txt
tail -40 analysis/probe_report.txt

# Validar sintaxe de todas as células de código
python3 -c "
import json,ast
nb=json.load(open('probe_video_contribution.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        try: ast.parse(''.join(c['source']))
        except SyntaxError as e: print(i,e)
print('ok')"
```

---

## 7. Errors, Failures & Resolutions

### 7.1 PORTÃO 1 FALHOU — AssertionError no primeiro run

**Error message:**
```
AssertionError: PORTÃO 1 FALHOU: o replay fusion→heads não reproduz final_metrics.json.
Causa provável: dtype/autocast divergente ou cache de outra run. TUDO abaixo seria
artefato de replay — não ler.
```

**Root cause:** duas causas somadas, nenhuma diagnosticável pelo assert original:
1. **`model.training == True`.** No caminho "cache carregado do disco", o
   `trainer.predict` é saltado, o modelo fica no `train()` de nascença
   (`nn.Module` nasce em `train()`) e o `Dropout(0.1)` da fusão (`video_core.py:1049`)
   rodava no replay.
2. **`bs=64` no replay vs `per_device_eval_batch_size=2` no `predict`.** Matmul bf16
   reduz em tiles diferentes; a cabeça B é fragilidade de fronteira por hipótese, então o
   último bit pode virar um `argmax`.

**Resolution:** ✅ Resolvido. `model.eval()` explícito + `assert model.training is False`,
autocalibração de 4 modos (`bs` × autocast) contra os logits cacheados, e decomposição do
assert único em 3 portões (A aborta; B e C avisam).

### 7.2 AttributeError no fecho do log

**Error message:**
```
AttributeError: 'OutStream' object has no attribute 'f'
```
Linha: `_log.f.close()`

**Root cause:** a última célula foi executada **duas vezes** (`In[17]` com apenas 16 células
de código na versão de então). A primeira execução restaurou `sys.stdout = _ORIG_STDOUT`
(um `OutStream` do ipykernel) e fechou o arquivo; a segunda encontrou `sys.stdout` já
restaurado e tentou acessar `.f`. **Não é bug de fluxo** — a run tinha completado e todo o
output já estava no log.

**Resolution:** ✅ Resolvido. Guarda `isinstance(sys.stdout, _Tee)` antes de fechar; caminho
`else` imprime aviso em vez de levantar.

### 7.3 Jupyter não grava outputs no `.ipynb`

**Sintoma:** após duas runs completas, `probe_video_contribution.ipynb` tem
`execution_count = None` e `outputs = []` em todas as células de código, apesar de os
artefatos da run existirem no disco (`task_b_flips.csv` com mtime posterior ao do notebook).

**Root cause:** ⚠️ **NÃO DETERMINADA.** Candidatos não verificados: `nbstripout` como git
hook, `jupytext`, `ContentsManager` customizado, ou o usuário salvando antes do fim da run.
Não investigado por decisão — contornado.

**Resolution:** ⚠️ **CONTORNADO, não resolvido.** O notebook duplica `sys.stdout` para
`analysis/probe_report.txt` (célula 0). O log funcionou: 195 947 bytes com o output completo
de todas as seções.

### 7.4 Falsos positivos e armadilhas de ferramenta (registrar para não repetir)

- **`f"{'gold \\ pred':>16s}"` levanta SyntaxError em Python < 3.12** — backslash dentro de
  expressão de f-string. Trocado por `'gold|pred'`.
- **Marcador de grep sem acento não encontra string com acento.** Buscar
  `"autocalibracao"` num arquivo que contém `"autocalibração"` dá falso negativo. Sempre
  confirmar com a forma acentuada.
- **`new = """..."""` no script gerador quebra se o conteúdo tiver docstring `"""`.** A
  solução foi escrever o conteúdo das novas células num arquivo separado e ler com
  `Path(...).read_text()`.
- **Sonda de zerar o tensor `frames` PÓS-normalização é enganosa** (registrado no doc 15,
  §2.3, confirmado): zero pós-normalização é o **pixel médio do ImageNet** — uma imagem
  cinza plausível, que o VideoMAE converte num embedding **constante mas não-nulo**,
  absorvido como viés fixo. Métricas quase não mudam e algumas **melhoram** (Task A hard
  +0,008, Task C hard +0,010), sugerindo falsamente que o vídeo não contribui.

---

## 8. What Works Right Now

### 8.1 Infraestrutura

- ✅ `EXIST_VIDEOS/probe_video_contribution.ipynb` — 34 células, 18 de código, sintaxe
  validada, **rodou do início ao fim na VM**. bytes 95 451, sha256 `a7b07270b3b22973`.
- ✅ **Portão A (validade do replay):** o replay `fusion → heads` reproduz as **decisões**
  do forward completo com 0 flips em A e B.
- ✅ **Portão B/C:** esta sessão reproduz `final_metrics.json` e os dois derangements de
  `test.ipynb` (0,4106 / 0,4316).
- ✅ **Portão da §8:** `pool_learned` recomputado no forward só-vídeo casa com o slot
  cacheado — cosseno médio **1,000000**, mínimo **0,999999**, maxΔ absoluto **0,0000**.
- ✅ **Cross-check §6 ↔ §8:** as sondas sobre `pool_learned` reproduzem o §6 com
  **Δ = +0,0000** nos 4 alvos.
- ✅ **Portão da §8b (no-op):** desligar o adaptador move os tokens **7,24 %** em norma
  (cos médio 0,996492, mín 0,927426; rel p05 0,0347 / p95 0,1591).
- ✅ Caches idempotentes: `val_cache.pt` (5 511 010 B), `pool_variants.pt` (17 067 008 B),
  `pool_variants_base.pt` (17 067 048 B), `task_b_flips.csv` (208 813 B).
- ✅ Log completo em `analysis/probe_report.txt` (195 947 bytes).

### 8.2 Baseline do checkpoint sondado (`final_metrics.json`)

| métrica | valor |
|---|---:|
| `eval_loss` | 1,7531743049621582 |
| `task_a_icmnorm_hard` | 0,5536400497959829 |
| `task_a_icmsoftnorm_soft` | 0,5058336902253192 |
| `task_b_icmnorm_hard` | 0,4968406607263488 |
| `task_b_icmsoftnorm_soft` | 0,3947158991795177 |
| `task_c_icmnorm_hard` | 0,343594298083231 |
| `task_c_icmsoftnorm_soft` | 0,15140392884508733 |
| `task_a_fmeasure_hard` | 0,7016964904542877 |
| `task_b_fmeasure_hard` | **NaN** |
| `task_b_crossentropy_soft` | **NaN** |
| `task_c_fmeasure_hard` | **NaN** |
| `proxy_task_a_auc` | 0,7809871971884021 |
| `proxy_task_b_f1` | 0,610426630697786 |
| `proxy_task_c_f1` | 0,33314283140848094 |

`meta.json` do checkpoint: `condition=B2_bio_viewer`, `arm=B15`, `restore_biases=true`,
`frames_version=v2`, `audio_version=v2`, `num_train_epochs=15`, `fusion_in=2688`,
`split_seed=42`, `init_seed=42`, `train_runtime_s=5594.2768`.

### 8.3 TESTE 2 — H3 (Task B ICMNorm hard, Δ vs normal, 5 seeds)

| arm | Δ | amplitude | ‖Δvid_feat‖ |
|---|---:|---:|---:|
| `perm_across` | −0,0631 | ±0,0501 | 13,154 |
| `noise_across` | −0,0353 | ±0,0484 | 13,154 |
| `perm_within` | **−0,0002** | ±0,0207 | 13,119 |
| `noise_within` | −0,0423 | ±0,0465 | 13,119 |

Tolerância pré-registada: **0,0501**.

| comparação | diferença | vs tolerância | licencia |
|---|---:|---|---|
| `perm_within` vs `perm_across` | **0,0629** | **>** | real |
| `perm_within` vs `noise_within` | 0,0421 | < | mesma direção, não sustenta só |
| `perm_across` vs `noise_across` | 0,0278 | < | **sem poder** |

### 8.4 TESTE 1 — escada de permutação (Δ médio, 5 seeds)

| slot | A hard | A soft | B hard | B soft | C hard | C soft |
|---|---:|---:|---:|---:|---:|---:|
| vídeo | −0,0079 | −0,0225 | −0,0631 | −0,0527 | +0,0032 | −0,0086 |
| áudio | −0,0120 | −0,0143 | −0,0574 | −0,0417 | −0,0098 | −0,0207 |
| **texto** | **−0,2313** | **−0,1985** | **−0,2541** | **−0,1921** | **−0,2160** | **−0,1514** |
| bio | −0,0078 | −0,0051 | −0,0207 | −0,0092 | −0,0022 | +0,0027 |

### 8.5 TESTE 3 — escalas, contribuição, pool

| slot | ‖feat‖ média | ‖W·feat‖ média | fatia | norma de coluna (doc 15) |
|---|---:|---:|---:|---:|
| texto | 31,126 | **53,686** | **52,2 %** | 0,630 |
| áudio | 23,174 | 27,242 | 26,5 % | 0,435 |
| vídeo | 15,632 | 18,840 | 18,3 % | **0,749** |
| bio | 1,888 | 3,091 | 3,0 % | 0,504 |

`vid_feat` std/média entre amostras = **0,757**.

Entropia do pool (`ln(1568) = 7,358` nats ⇒ 1 568 tokens efetivos se uniforme):

| estatística | média | mediana | p05 | p95 |
|---|---:|---:|---:|---:|
| entropia (nats) | 1,591 | 1,599 | 0,352 | 2,941 |
| tokens efetivos `exp(H)` | 6,794 | **4,950** | 1,422 | 18,938 |
| **peso do token máximo** | 0,531 | **0,502** | 0,207 | 0,935 |
| massa do top-10 | 0,941 | **0,970** | 0,775 | 0,999 |

Perfil temporal (massa de atenção por posição; uniforme = 0,125):

| t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,095 | **0,232** | 0,127 | **0,032** | 0,051 | 0,164 | 0,057 | **0,243** |

Desvio-padrão entre amostras do perfil: **0,2480**. `t7 + t1 = 47,5 %`; `t3 + t4 = 8,3 %`.

Checagem do reshape `1568 → (8,196)` (troca dos frames 0-1, ‖Δhidden‖ por bloco de 196):

| bloco | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ‖Δ‖ | **16,4362** | 2,4150 | 1,6511 | 1,2542 | 1,2099 | 1,5133 | 1,1803 | 1,3509 |

Razão bloco0 / média dos outros = **10,88×**. Reshape **sustentado**; perfil temporal
reportável.

### 8.6 TESTE 4 — sondas lineares (AUC − nulo, metade de report, n≈252)

| alvo | vídeo | texto | áudio | bio |
|---|---:|---:|---:|---:|
| A_hard | +0,050 | **+0,212** | +0,083 | +0,043 |
| B_hard | +0,069 | **+0,189** | +0,133 | +0,066 |
| língua (es vs en) | **−0,076** | +0,466 | +0,424 | **+0,411** |
| duração > mediana | +0,117 | +0,323 | **+0,494** | **+0,361** |

### 8.7 TESTE 5 — matriz de flips da Task B (5 seeds, 574 flips de 2 525 decisões, 231 ids únicos)

| normal → | NONE | JUDGEMENTAL | DIRECT |
|---|---:|---:|---:|
| **NONE** | — | 48 | **134** |
| **JUDGEMENTAL** | 84 | — | 57 |
| **DIRECT** | **168** | 83 | — |

| eixo | flips | % | acaso (uniforme) |
|---|---:|---:|---:|
| **DIRECT ↔ NONE** | **302** | **52,6 %** | 33,3 % |
| DIRECT ↔ JUDGEMENTAL | 140 | 24,4 % | 33,3 % |
| JUDGEMENTAL ↔ NONE | 132 | 23,0 % | 33,3 % |

Nos 574 casos que mudaram:

| | acerto vs gold próprio |
|---|---:|
| antes de permutar (`pred_normal`) | **51,9 %** |
| depois de permutar (`pred_perm`) | **34,3 %** |

`pred_perm == gold do doador` = **43,0 %** contra **36,6 %** esperado se independente
(+6,4 pontos; n efetivo ≈ 231 ⇒ ~2σ — **marginal, não usar**).

Distribuições no CSV de flips: `gold_b` {NONE 236, DIRECT 222, JUDGEMENTAL 116};
`pred_normal` {DIRECT 251, NONE 182, JUDGEMENTAL 141}; `pred_perm` {NONE 252, DIRECT 191,
JUDGEMENTAL 131}; `donor_gold_b` {NONE 289, DIRECT 175, JUDGEMENTAL 110}; `lang` {es 346,
en 228}.

**Todas as 3 classes de B aparecem no `argmax` em condição normal** — o `fmeasure_hard = NaN`
NÃO vem de classe nunca predita.

### 8.8 TESTE 6 — comparação de resumos dos 1 568 tokens (AUC − nulo ± sd)

| alvo | `pool_learned` | `mean_all` | `mean_temporal` | `top5` |
|---|---:|---:|---:|---:|
| A_hard | +0,050 ±0,020 | +0,057 ±0,028 | **+0,096** ±0,022 | +0,040 ±0,016 |
| B_hard | +0,069 ±0,024 | +0,088 ±0,018 | **+0,093** ±0,014 | +0,068 ±0,021 |
| língua | −0,076 ±0,033 | +0,055 ±0,029 | +0,011 ±0,031 | −0,089 ±0,029 |
| duração | +0,117 ±0,016 | **+0,297** ±0,022 | **+0,304** ±0,028 | +0,114 ±0,016 |

`mean_all` − `pool_learned`: **+0,007** (A, tol 0,056) e **+0,019** (B, tol 0,050) — ambos
abaixo. `mean_temporal` − `pool_learned`: +0,046 (A, tol 0,056) e +0,024 (B, tol 0,050) —
ambos abaixo.

### 8.9 TESTE 7 — retorno da adaptação LoRA (resumos SEM parâmetros)

| resumo | alvo | COM LoRA | SEM LoRA | ganho | tolerância |
|---|---|---:|---:|---:|---:|
| `mean_all` | A_hard | +0,057 ±0,028 | +0,053 ±0,027 | **+0,004** | 0,056 |
| `mean_all` | B_hard | +0,088 ±0,018 | +0,083 ±0,017 | **+0,005** | 0,050 |
| `mean_all` | língua | +0,055 ±0,029 | +0,060 ±0,027 | −0,005 | — |
| `mean_all` | duração | +0,297 ±0,022 | +0,297 ±0,019 | +0,001 | — |
| `mean_temporal` | A_hard | +0,096 ±0,022 | +0,075 ±0,024 | **+0,021** | 0,050 |
| `mean_temporal` | B_hard | +0,093 ±0,014 | +0,076 ±0,017 | **+0,017** | 0,050 |
| `mean_temporal` | língua | +0,011 ±0,031 | −0,004 ±0,033 | +0,015 | — |
| `mean_temporal` | duração | +0,304 ±0,028 | +0,306 ±0,030 | −0,002 | — |

`pool_learned` (**CONFUNDIDO** — treinado sobre tokens adaptados): A +0,050/+0,036
(+0,014); B +0,069/+0,075 (−0,005); língua −0,076/−0,079 (+0,003); duração +0,117/+0,142
(−0,025).

**Todos os ganhos abaixo da tolerância.**

### 8.10 Contagens de parâmetros — TODAS reconferidas por aritmética

| bloco | conta | params |
|---|---|---:|
| fusion (2688→1024) | 2688·1024+1024 | 2 753 536 |
| xlmr LoRA | 24·2·(1024·16+16·1024) | 1 572 864 |
| audio_proj | 1280·768+768+2·768 | 985 344 |
| **videomae LoRA** | 12·2·(768·16+16·768) | **589 824** |
| `video_pool` | (768·128+128)+(128·1+1) | 98 561 |
| bio + viewer | 18 816 + 10·32 + 528 | 19 664 |
| heads a/b/c | 1025+3075+6150 | 10 250 |
| **total** | | **6 030 043** |

Ramo visual dos memes: 4 camadas de CLIP ViT-L destreinadas
(`memes_core.py:489-494`). Por camada: atenção 4·(1024²+1024) = 4 198 400; MLP
(1024·4096+4096) + (4096·1024+1024) = 8 393 728; 2 LayerNorms = 4 096. Total
**12 596 224 × 4 = 50 384 896**. ⚠ Estimativa por fórmula a partir do código, **não**
contagem em runtime.

---

## 9. What's Broken or Incomplete

- ⏳ **`improvements/16_VIDEO_INVESTIGATION_RESULTS.md` NÃO EXISTE.** É a próxima ação.
- ⏳ **As 8 correções aos docs 14 e 15 NÃO foram aplicadas nesses arquivos.** Os documentos
  ainda contêm as afirmações erradas listadas na §10.
- ⏳ **A tese e o paper CEUR-WS ainda descrevem o pooling errado** (3 erros, §10.4).
- ⚠ **Confound não resolvido no perfil temporal.** Não sei como a extração v2 tratou vídeos
  com menos de 64 frames decodificáveis. Se houver repetição de frame no fim, o pico em
  `t7` (0,243) pode ser artefato. **Checar antes de o perfil temporal ir para a tese.**
- ⚠ **Jupyter não grava outputs no `.ipynb`** (causa não determinada, contornada pelo tee).
- ❌ **O script gerador do notebook era efêmero (scratchpad) e não existe mais.** Edições
  futuras têm de ser feitas no `.ipynb` diretamente.

---

## 10. Direction Changes, Correções e Abandoned Approaches

### 10.1 Correções aos documentos 14 e 15 (8 itens)

1. **§3.4 do doc 15 INVERTE.** A afirmação *"a fusão não ignora o vídeo — norma de coluna
   0,749 (vídeo) > 0,630 (texto)"* é enganosa. Em **peso × escala**: texto **52,2 %**,
   vídeo **18,3 %** — o texto contribui **2,85×**. A norma de coluna maior do vídeo mede
   aproximadamente o **inverso da escala da feature** (`vid_feat` tem metade da norma de
   `txt_feat`: 15,6 vs 31,1) e a `Linear` compensa por unidade. **NÃO escrever na tese que
   a fusão dá ao vídeo peso maior que ao texto.**
   ⚠ A métrica `‖W_slot·feat_slot‖` é **descritiva, não causal** — há bias e GELU a
   jusante, o pré-ativação não se decompõe linearmente na saída. Quem mede causalmente é a
   escada de permutação. **Convergência notável:** a ordem das fatias (texto ≫ áudio ≈
   vídeo > bio) é a mesma da escada causal.

2. **O enquadramento de "desequilíbrio de capacidade" da H2 (§3.1/§4 do doc 14) não
   sustenta.** `memes_core.py` mexe em `requires_grad` **só** na visão (linhas 489-494); o
   XLM-R dos memes (linha 473) é treinado **inteiro**, sem LoRA e sem freeze. Logo:
   memes = texto ~355 M vs visão 50,4 M ≈ **7×**; vídeo = texto 1,57 M vs vídeo 0,59 M ≈
   **2,7×**. **O modelo de memes é MAIS dominado por texto em termos relativos e a visão
   dele contribui muito.** A razão texto:visão não explica a diferença. A H2 tem de repousar
   em **capacidade absoluta** (0,59 M rank-16 vs 50,4 M full-rank) **+ objetivo de
   pré-treino** (contrastivo vs reconstrução de pixels).

3. **"`fmeasure_hard = NaN` sugere classe nunca predita" (F16/H3) é inferência fraca e
   está REFUTADA.** As 3 classes aparecem no `argmax` em condição normal. `proxy_task_b_f1 =
   0,6104` era compatível com as duas leituras (teto com classe nunca predita = 0,667). O
   NaN vem de outro motivo, interno ao PyEvALL.

4. **−0,076 → −0,063 ± 0,050.** Com 5 derangements, `perm_across` dá −0,0631 com amplitude
   0,0501. A estimativa de 2 seeds do doc 15 era um ponto frágil: **a amplitude é da ordem
   do próprio efeito.** Reportar SEMPRE com a amplitude, nunca como valor pontual.

5. **O eixo do mecanismo da H1 está errado.** A H1 afirma que "formato separa DIRECT de
   JUDGEMENTAL". Os flips dizem **DIRECT ↔ NONE (52,6 %)**; o eixo D↔J tem 24,4 %, **abaixo
   do acaso** (33,3 %). Reescrever: *o visual entrega enquadramento, e enquadramento
   discrimina a enunciação em primeira pessoa (DIRECT) do não-sexista (NONE)*. Mecanicamente
   coerente: "a fonte fala para a câmera" é visualmente distintivo; separar "comenta
   material de terceiros" (JUDGEMENTAL) de NONE pelo visual é muito mais difícil — ambos
   podem ser reação/duet/greenscreen.

6. **[CORREÇÃO DE AFIRMAÇÃO MINHA] Promovi a H5 sem base.** Eu escrevi *"A H5 sai de
   'amplificador' para candidata a gargalo primário"* com base no colapso de 5 tokens.
   **Errado** — inferi custo a partir de mecanismo. O TESTE 6 mede o custo diretamente:
   `mean_all` − `pool_learned` = +0,007 (A) e +0,019 (B), abaixo da tolerância. O colapso é
   real e **não custa** informação de rótulo linearmente decodificável. Volta ao veredicto
   original do doc 14 §3: **amplificador, não causa primária.**

7. **[CORREÇÃO DE AFIRMAÇÃO MINHA] Sobrecorrigi a metáfora dos "5 quadradinhos".** Eu disse
   que "5 tokens ≠ 5 pedaços da imagem, porque depois de 12 camadas de atenção cada token é
   uma mistura global". **Desmentido pela medição:** o TESTE 3d mostra concentração de
   **10,88×** no bloco correspondente. Os tokens do VideoMAE são **fortemente locais** —
   coerente com o pré-treino de reconstruir o próprio patch, ao contrário de um CLS
   contrastivo. A primeira leitura ("um recorte do vídeo") estava mais certa que a minha
   correção.

8. **[CORREÇÃO DE AFIRMAÇÃO MINHA / do notebook] "retorno medido de ZERO" é impreciso.** A
   célula do §8b imprime isso. O correto: *"abaixo da tolerância, com padrão positivo
   consistente apenas sob `mean_temporal`"*. Sob `mean_all` o efeito do LoRA desaparece
   (+0,004 / +0,005); sob `mean_temporal` aparece (+0,021 / +0,017 — 0,6σ e 0,8σ).
   **Hipótese (não medida):** o que o LoRA mudou é dependente de posição — a média global
   lava, a média por fatia preserva. ⚠ Enfraquecida por `língua` também subir +0,015 no
   mesmo resumo: não se pode afirmar que o ganho é semântico.

**Nota adicional sobre o doc 14 §3:** o argumento *"o pool treinou (pesos 2-3× a escala de
init), logo H5 está enfraquecida"* está errado no **mecanismo** — ele treinou **para
colapsar num seletor**; treinar e colapsar não são alternativas. Mas a **conclusão** (H5
não é causa primária) foi restaurada pelo TESTE 6, por outra via.

### 10.2 Achados novos, não previstos em nenhum documento

1. **O `nuisance` de duração atravessa 3 modalidades.** Decodificabilidade de "duração >
   mediana" (AUC − nulo): áudio **+0,494**, bio **+0,361**, texto **+0,323**, tokens crus de
   vídeo (`mean_all`) **+0,297**, `vid_feat` final **+0,117**. É um eixo de confusão do
   dataset inteiro, não de uma modalidade.
2. **O nuisance de duração é PRÉ-TREINADO, não aprendido.** Duração: +0,297 vs +0,297
   (`mean_all`), +0,304 vs +0,306 (`mean_temporal`) — **idêntico com e sem LoRA**. É
   propriedade das features pré-treinadas do VideoMAE + amostragem, não algo que o treino
   introduziu. Isso **exonera o treino** e localiza o confound no encoder/pipeline.
3. **O pool aprendido é um FILTRO DE NUISANCE.** Duração: tokens crus +0,297 → `vid_feat`
   +0,117. Língua: +0,055 → **−0,076**. O seletor descarta a variável de captura. É ponto
   **a favor** do pool, não contra. Também mata a preocupação de que o confound temporal da
   F12 vaze para a fusão: está nos tokens, mas o pool o filtra.
4. **O áudio empata com o vídeo na Task B e ganha na Task A.** Escada: áudio −0,0574 vs
   vídeo −0,0631 em B; áudio −0,0120 vs vídeo −0,0079 em A. Um único vetor de 1 280-d
   pré-extraído e congelado iguala 16 frames por um encoder adaptável. Vai direto no
   argumento de **capacidade absoluta** da H2.
5. **`vid_feat` NÃO codifica língua (−0,076, abaixo do nulo).** O canal visual não traz o
   confound de língua que arruinou a normalização per-viewer. Achado limpo.
6. **A cabeça B é instável:** **231 de 505 itens (45,7 %)** trocam de predição sob
   permutação de vídeo, e a métrica só cai 0,063 — muitos flips se cancelam. Vindicação
   parcial do *espírito* da H3 (fragilidade), apesar de a forma forte estar refutada pelo
   `perm_within`. **As duas coisas convivem:** a cabeça é instável E o vídeo carrega
   informação real de classe.
7. **Nos 574 flips, o acerto cai de 51,9 % para 34,3 %** (−17,6 pontos). O slot de vídeo
   estava entregando classe **correta** nesses itens. Junto com `perm_within` ≈ 0, fecha:
   informação alinhada ao rótulo, não ruído.

### 10.3 Conclusões substantivas — o que a investigação estabeleceu

- **H3 (forma forte) REFUTADA.** `perm_within` (−0,0002) e `perm_across` (−0,0631) têm
  ‖Δ‖ praticamente idêntico (13,119 vs 13,154) e diferem em **0,0629 > 0,0501**. A única
  coisa que difere entre os arms é a **classe gold de B do doador**, que o modelo nunca vê.
  Uma cabeça genericamente frágil não pode dar dano ~zero sob perturbação de norma 13,1.
  ⚠ **Caveat de desenho a declarar:** `perm_within` agrupa pelo **gold** de B, logo o arm é
  oracular na **construção**. É legítimo — o gold escolhe o doador, nunca entra no modelo —
  mas tem de estar escrito antes que alguém levante.
- **AFIRMAÇÃO DE TETO:** dano ~zero dentro da classe significa que o slot de vídeo **não
  carrega nada além da classe de B**. Se carregasse intensidade, categoria de C ou detalhe
  de enquadramento, trocar dentro da classe custaria algo. **O ramo de vídeo funciona como
  um classificador fraco de intenção da fonte, e nada mais.**
- **H1 confirmada na parte 1, corrigida no eixo.** O texto domina (−0,231 / −0,254 /
  −0,216). A contribuição pequena do vídeo é **proporcional, não anômala**.
- **H5 não é causa primária** (TESTE 6).
- **Rotas FECHADAS por medição:** trocar o agregador (não há reservatório de sinal na cauda
  de 1 563 tokens); a fusão ignorar o vídeo (dá 18,3 %); ARCH-01 (já corrigido);
  capacidade-de-adaptação-no-primeiro-incremento (retorno abaixo do ruído).
- **Sobra H2:** capacidade absoluta + objetivo de pré-treino. O melhor que qualquer resumo
  de vídeo decodifica é **+0,096** (A); o texto decodifica **+0,212**. Mesmo com o melhor
  agregador testado, o vídeo fica em ~metade.

### 10.4 Correções de arquitetura para a tese (4 itens, independentes de qualquer resultado)

1. **O `TemporalAttentionPool` NÃO é temporal.** `video_core.py:964-967` faz
   `softmax(logits, dim=1)` sobre `x = last_hidden_state` `[B, 1568, 768]`
   (`video_core.py:1128-1129`). Os 1 568 tokens são **espaço-temporais** (8 posições
   temporais × 196 patches 14×14). **Não depende de nenhuma suposição sobre ordenação de
   tokens:** `dim=1` é o eixo dos 1 568. A docstring (`video_core.py:950-953`) está errada
   nas **duas** afirmações ("sobre a dimensão temporal", "cada token de frame").
2. **NÃO é pooling, é quase um SELETOR.** Peso mediano do token máximo = **0,502**; massa
   mediana do top-10 = **0,970**; 5 % dos vídeos com 1,42 tokens efetivos. Uma média
   ponderada onde um elemento tem peso 0,5 não é agregação. Entropia 1,599 de 7,358
   possíveis = **21,7 % do máximo**.
3. **Os tokens do VideoMAE são LOCAIS (10,88×), não resumos globais.** Ao contrário do CLS
   contrastivo do CLIP, o pré-treino de reconstrução por patch mantém os tokens locais mesmo
   após 12 camadas de self-attention. Logo o token eleito é um **recorte** do vídeo.
4. **A fusão dá ao vídeo 18,3 %, não "mais que ao texto".** Ver §10.1 item 1.

Complemento: **o meio do vídeo é ignorado.** `t7 + t1 = 47,5 %` da massa de atenção;
`t3 + t4 = 8,3 %`. Desvio-padrão entre amostras 0,2480 > valor uniforme 0,125 ⇒ não é viés
posicional fixo, o pool escolhe **posição diferente por vídeo**. ⚠ Sujeito ao confound de
frames repetidos (§9).

### 10.5 Lição metodológica (material publicável por si só)

1. **Zerar entrada normalizada engana.** Zero pós-normalização = pixel médio do ImageNet =
   imagem cinza plausível → embedding **constante mas não-nulo** → absorvido como viés fixo.
   Métricas quase não mudam e algumas **melhoram** (A hard +0,008, C hard +0,010).
2. **Zerar `vid_feat` pós-pooling subestima por construção** (−0,022 vs −0,063 da
   permutação): reproduz exatamente a operação do modality dropout, para a qual o treino
   criou robustez.
3. **Permutação preserva a distribuição de entrada** e mede alinhamento.
4. **Ruído isotrópico é o controle ERRADO — e foi medido como tal.** `perm_across` vs
   `noise_across` = 0,0278, abaixo da tolerância de 0,0501: **sem poder a n=505**.
   "Indistinguível" ≠ "confirmado".
5. **O controle que discrimina é permutação condicional à classe com magnitude casada.**
   ‖Δ‖ 13,119 vs 13,154, dano −0,0002 vs −0,0631.
6. **A regra assimétrica funciona.** `video_core.py:1066-1071`: métricas não caírem fecha a
   opção (achado negativo publicável); métricas caírem é **gatilho**, nunca evidência. Quem
   prova é um braço treinado.

### 10.6 Abordagens abandonadas — NÃO revisitar

- **Assert cego único no portão.** Confundia 3 causas. Substituído por 3 portões
  decompostos. Não voltar.
- **Replay em `bs=64` sem autocalibração.** Causou a falha 7.1. Sempre autocalibrar contra
  os logits cacheados.
- **Cross-attention sobre os 1 568 tokens como primeira ação.** O critério de reabertura de
  `video_core.py:1061-1064` **foi atendido** (`perm_within` provou que o vídeo contribui),
  **mas o TESTE 6 fechou a rota**: não há reservatório de sinal na cauda. Seria uma forma
  mais cara de não achar nada. Custo estimado, se algum dia voltar: query de dimensão 256 →
  Wq 262 144 + Wk 196 608 + Wv 196 608 + Wo 196 608 ≈ **852 k params** — num dataset de
  2 019 amostras, o mesmo risco que matou a versão anterior (45 % dos treináveis).
  **Não revisitar sem evidência nova.**
- **Concatenar os 1 568 tokens direto na fusão.** 1 568 × 768 = 1 204 224 entradas ×
  1 024 = **1 233 125 376 params** na fusão. Impossível.
- **Braço de 32 frames.** `num_frames` é livre no VideoMAE (verificado na VM: carrega sem
  incompatibilidade de shape, `num_patches=3136`, porque `position_embeddings` é tabela
  sinusoidal por fórmula e não está no `state_dict`). Mas custo 4-5 h e o gap continua
  função da duração (razão p90/p10 = 7,2× para 16, 32 ou 64). **Não rodar.**
- **Re-extrair features.** Concluída e verificada em 2026-08-09, custou um dia. Frames: 64
  uint8/vídeo, 2 524 + 674, 0 faltantes. **Não repetir.**

---

## 11. Pending Decisions & Open Questions

- ❓ **Gastar ou não uma run no braço de mais capacidade (LoRA r=64 ou `target_modules` com
  MLP, só no vídeo)?** Minha recomendação registrada: **não**, por 4 razões: (1) retorno
  medido do primeiro incremento abaixo do ruído; (2) teto causal do vídeo é −0,063 numa
  única task contra −0,254 do texto; (3) o único sinal positivo (`mean_temporal`) exigiria
  mudar **duas** coisas ao mesmo tempo (capacidade **e** pooling fatorado), o que é desenho
  experimental ruim e não atribuível; (4) o achado negativo já está medido e é publicável.
  **O usuário não decidiu.**
- ❓ **O pico em `t7` do perfil temporal é real ou artefato de frames repetidos?** Depende de
  como a extração v2 tratou vídeos com < 64 frames decodificáveis. **Não sei.** Checar em
  `improvements/11_EXTRACTION_IMPLEMENTATION_PLAN.md` ou no código de extração
  (`extract_video_v2.ipynb`) antes de o perfil ir para a tese.
- ❓ **Por que o Jupyter não grava outputs no `.ipynb`?** Contornado, não diagnosticado.
- ❓ **`reset_lora_weights()` / `reset_new_layer_weights()`** — REV-01: essas funções não
  existem em nenhum lugar do repositório, só na documentação. Decidir se a mitigação de
  `bf16=True` + LoRA nunca foi implementada ou foi removida. **Pendência antiga, não tocada
  nesta sessão.**
- ❓ **`TEST_CASE = "EXIST2026"`** — confirmar contra o arquivo gold oficial do EXIST 2026
  antes de submeter. Pendência antiga.
- ❓ Os números do pré-treino do VideoMAE (Kinetics-400, 16 frames, stride 4 @ ~30 fps →
  ~7,6 fps, gap ~0,13 s) vêm da config padrão do paper e **não foram confirmados em fonte
  primária**. Confirmar antes de irem para a tese.

---

## 12. Next Steps

- [ ] **Escrever `EXIST_VIDEOS/improvements/16_VIDEO_INVESTIGATION_RESULTS.md`** com: os 7
      testes e todos os números da §8 deste relatório; as 8 correções da §10.1; os 7 achados
      novos da §10.2; as conclusões da §10.3; as 4 correções de arquitetura da §10.4; a lição
      metodológica da §10.5; as rotas fechadas da §10.6. Marcar cada afirmação com
      `[MEDIDO]` / `[VERIFICADO]` / `[HIPÓTESE]`, conforme a convenção do doc 15.
- [ ] **Aplicar as correções nos docs 14 e 15** — ou marcar as seções obsoletas com um
      ponteiro para o doc 16. As afirmações erradas (§3.4 do 15; §3.1/§4 e §3 do 14; F16)
      ainda estão lá.
- [ ] **Corrigir a docstring do `TemporalAttentionPool`** (`video_core.py:950-953`) — está
      errada nas duas afirmações. É a única edição de código indicada, e é documental.
- [ ] **Checar o confound de frames repetidos** antes de o perfil temporal ir para a tese.
- [ ] **Corrigir a descrição de arquitetura na tese e no paper CEUR-WS** (4 itens da §10.4).
- [ ] Decidir sobre o braço treinado de capacidade (§11, item 1).
- [ ] (Opcional, barato) Ler manualmente `analysis/probe_cache_B15/task_b_flips.csv` para
      verificar se os JUDGEMENTAL cuja transcrição "parece DIRECT" são os que mudam — o
      passo humano da H1 que nunca foi feito.

---

## 13. External References

| Resource | URL / Reference | Purpose |
|----------|-----------------|---------|
| VideoMAE | `MCG-NJU/videomae-base` (HuggingFace) | encoder de vídeo, `vc.VIDEOMAE_NAME` |
| XLM-RoBERTa | `xlm-roberta-large` (HuggingFace) | encoder de texto, `vc.XLMR_NAME` |
| CLIP | `openai/clip-vit-large-patch14` (HuggingFace) | encoder visual do modelo de memes |
| Whisper | `openai-whisper` 20250625, large-v3 | features de áudio pré-extraídas |
| PyEvALL | 0.1.78 | métricas ICM-Norm / ICM-Soft-Norm |
| PEFT | 0.19.1, `LoraConfig`, `get_peft_model`, `PeftModel.disable_adapter()` | LoRA |
| EXIST 2026 | CLEF 2026 | competição; equipa **Jow** |
| CEUR-WS | classe LaTeX `ceurart` | working notes do paper |

---

> **Instructions for the next chat:** This is a full context report from a previous session.
> Read it completely before responding. You now have full knowledge of the project state,
> all decisions made, all code written, and all pending work. Do not ask the user to repeat
> information already present in this document. Begin by briefly confirming your
> understanding of the current state and ask where the user wants to continue.
