# 11 — Plano de implementação: re-extração de features v2 (frames + áudio)

**O que este documento é:** o plano de execução mecânica das correções especificadas em
`10_EXTRACTION_FIXES_SPEC.md` (V1–V7, A1–A5, C1–C4, decisões D1–D4). Contém o código
integral dos dois notebooks novos, as funções de substituição integrais do
`video_core.py`, a ordem de execução, a verificação e os modos de falha.

**Quem executa:** um executor sem contexto prévio. Tudo o que é preciso saber está aqui
ou está referenciado com caminho exato. Não inferir nada; quando um passo diz PARAR,
parar.

**Regras invioláveis (do projeto):**
- Os arquivos originais `extract_video.ipynb` e `extract_audio.ipynb` **não são tocados**
  (decisão D3 da spec). Os novos são `extract_video_v2.ipynb` e `extract_audio_v2.ipynb`.
- O split congelado `Dataset/val_split_42.json` **nunca é recalculado**.
- `frames_v1_sample/` e `audio_features/` (v1) **não são apagados** até a §9 (verificação)
  passar inteira — são as referências dos testes 4 e 6.
- Nenhum treino faz parte deste plano. Nenhum número de ICM novo existe ainda.
- Comandos pesados de IO rodam **em célula de Jupyter na VM**, nunca sobre o mount SSHFS
  (ler um `.pt` de 9,6 MB pelo mount custa ~4,4 s; na VM é local).

**Convenção de rótulos de fato:**
- `[MEDIDO]` — número com fonte citada.
- `[NÃO VERIFICADO]` — estimativa ou hipótese; o plano diz como confirmar.
- `[VM]` — só pode ser confirmado rodando na VM; o comando exato está no texto.

---

## 1. Sumário executivo

### O que muda

| área | de | para | correções |
|---|---|---|---|
| frames | 16 frames float32 normalizados, amostrados por `cap.set` (salto impreciso), fallback com dupla conversão BGR, zeros gravados em falha, árvore `frames/{split}` sem proveniência | **64 frames uint8** (resize/crop idênticos, SEM normalizar), leitura sequencial em 2 passes, falha → `_missing.json`, árvore `frames/v2/{training,test}` + `_extraction_meta.json` | V1–V7, D1, D3, D4 |
| áudio | média do encoder Whisper sobre 30 s COM padding, corte em 30 s, zeros em falha, sem proveniência | média mascarada aos frames válidos (A1), janelas múltiplas ponderadas pelos segundos válidos (A2), falha → `_missing.json`, árvore `audio_features/v2/{training,test}` + meta | A1–A5, D4 |
| `video_core.py` | `frames_dir`/`audio_dir` → v1; `load_frames` assume float normalizado; zeros silenciosos em arquivo faltante | default v2 (v1 acessível por parâmetro), desnormalização uint8 no `Dataset` (decisão T3, §4.3), subamostragem 64→`N_FRAMES_USED=16`, `RuntimeError` no build se houver ID em `_missing.json` | C1–C4, D2, D4 |

### Custos

| item | custo | fonte |
|---|---|---|
| disco frames v2 | **30,8 GB** (9 633 792 B × 3 198 — exatamente os mesmos bytes/vídeo da v1: 64·3·224²·1 = 16·3·224²·4) | aritmética, conferida em §0 deste plano |
| disco áudio v2 | ~26 MB | tamanho da árvore v1 (`audio_features/`, mesma forma `[1280]` float32) |
| disco livre na VM | 47 GB `[MEDIDO em 2026-08-08, spec §6]` — **margem ~16 GB. Reconfirmar antes de começar (§3)** | spec §6 |
| GPU áudio | ~10–14 min os dois splits `[NÃO VERIFICADO — v1 fez 674 itens a 7,50 it/s; v2 tem ~4 240 janelas no total: 3 198 itens + 1 042 janelas extra (calculado de `analysis/audio_durations.csv`)]` | output salvo de `extract_audio.ipynb`; CSV |
| CPU frames | **35 min a ~2 h** `[NÃO VERIFICADO]` — a v1 mediu 1,54 it/s (~35 min), mas a leitura em 2 passes decodifica ~2×. O notebook v2 tem uma célula de sondagem de tempo (célula 5) que projeta o total ANTES da run completa | `03_OPTIMIZATION_FINDINGS.md` (1,54 it/s); decisão T1 em §4.1 |
| verificação | ~20–30 min de células na VM | estimativa |

Nota de enquadramento: a extração de frames **não usa GPU** (cv2 + processor em CPU);
"custo de GPU" da spec refere-se ao áudio (Whisper) e à ocupação da máquina.

### Ordem e bloqueios (detalhe em §8)

```
P0  Pré-condições (§3)  ──────────────── se falhar qualquer uma: PARAR
P1  Escrever os 3 arquivos (2 notebooks v2 + edições no video_core.py)
P2  extract_audio_v2: training → test  ── ~10-14 min GPU
P3  Verificação do áudio (testes 1,2,3,6,7)  ── bloqueia P6 se falhar
P4  extract_video_v2: probe de tempo → training → test  ── 35 min-2 h
P5  Verificação dos frames (testes 1,2,3,4,5,8)  ── bloqueia P6 se falhar
P6  Ativação do consumidor: smoke test do Dataset com as árvores v2
P7  (fora deste plano) re-treinos na ordem da spec §7: caminho 1 primeiro
```

P2/P3 (áudio) e P4/P5 (frames) são independentes entre si; a ordem áudio-primeiro é
escolhida por ser 10× mais rápida e validar o mecanismo de manifesto/meta cedo. As
edições do `video_core.py` (P1) podem ser escritas a qualquer momento, mas o smoke test
(P6) só depois de P3+P5.

**O que bloqueia o quê:** o teste 4 (equivalência de normalização) reprovando bloqueia
TUDO que envolva frames v2 — é o único teste cuja reprovação invalida a comparabilidade
com o baseline. O teste 6 reprovando (normas de áudio v2 ≈ v1) significa que a extração
não rodou de verdade (V7) — apagar a árvore de áudio v2 e repetir.

---

## 2. Arquivos deste plano

| arquivo | ação | onde vive |
|---|---|---|
| `EXIST_VIDEOS/extract_video_v2.ipynb` | **criar** (código na §5) | repo (mount = VM) |
| `EXIST_VIDEOS/extract_audio_v2.ipynb` | **criar** (código na §6) | repo (mount = VM) |
| `EXIST_VIDEOS/video_core.py` | **editar** (funções de substituição na §7) | repo (mount = VM) |
| `EXIST_VIDEOS/video_core_pre_v2.py.bak` | **criar** — cópia de segurança antes de editar (o arquivo NÃO está no git; ver §11) | repo |
| `EXIST_VIDEOS/frames/v2/{training,test}/` | criados pela extração | VM |
| `EXIST_VIDEOS/audio_features/v2/{training,test}/` | criados pela extração | VM |

Não tocar: `extract_video.ipynb`, `extract_audio.ipynb`, `Thesis_video_fixed.ipynb`,
`Thesis_video_baseline.ipynb`, `inference_video.py`, `ablation_harness.py`,
`runs_baseline/`, `predictions/`, `Dataset/val_split_42.json`, `frames_v1_sample/`,
`audio_features/{train,test}` (v1).

---

## 3. Pré-condições verificáveis — TODAS antes de qualquer extração

Rodar numa célula de Jupyter **na VM** (não no mount). Se qualquer check imprimir
`FALHOU`, **PARAR e reportar ao autor** — não improvisar.

```python
# ── P0: pré-condições da re-extração v2 ─────────────────────────────────────
# Rodar na VM (kernel do Jupyter em /home/a67606/Thesis/EXIST_VIDEOS ou cd até lá).
import shutil, subprocess, sys
from pathlib import Path

ROOT = Path(".")  # = EXIST_VIDEOS/
ok = True
def check(name, cond, detail=""):
    global ok
    status = "ok" if cond else "FALHOU"
    ok = ok and cond
    print(f"[{status:6s}] {name} {detail}")

# 1. Disco: precisa de ~31 GB para frames v2 + margem. Limiar: 35 GB.
usage = shutil.disk_usage(str(ROOT))
free_gb = usage.free / 1e9
check("disco livre >= 35 GB", free_gb >= 35, f"(livre: {free_gb:.1f} GB)")

# 2. Referências preservadas (testes 4 e 6 dependem delas)
n_v1s_train = len(list((ROOT/"frames_v1_sample/train").glob("*.pt")))
n_v1s_test  = len(list((ROOT/"frames_v1_sample/test").glob("*.pt")))
check("frames_v1_sample/train == 20", n_v1s_train == 20, f"({n_v1s_train})")
check("frames_v1_sample/test == 5",  n_v1s_test == 5,  f"({n_v1s_test})")
n_a1_train = len(list((ROOT/"audio_features/train").glob("*.pt")))
n_a1_test  = len(list((ROOT/"audio_features/test").glob("*.pt")))
check("audio_features/train (v1) == 2524", n_a1_train == 2524, f"({n_a1_train})")
check("audio_features/test (v1) == 674",  n_a1_test == 674,  f"({n_a1_test})")

# 3. Dados de entrada
check("JSON treino", (ROOT/"Dataset/training/EXIST2026_training.json").exists())
check("JSON teste",  (ROOT/"Dataset/test/EXIST2026_test_clean.json").exists())
check("videos treino", (ROOT/"Dataset/training/videos").is_dir())
check("videos teste",  (ROOT/"Dataset/test/videos").is_dir())
check("split congelado", (ROOT/"Dataset/val_split_42.json").exists())

# 4. Versões e imports (a VM é a verdade; local DIVERGE)
import torch, transformers, numpy, cv2
print(f"        torch={torch.__version__} transformers={transformers.__version__} "
      f"numpy={numpy.__version__} cv2={cv2.__version__}")
check("transformers 5.6.0 (CLAUDE.md diz que é a da VM)",
      transformers.__version__.startswith("5.6"),
      f"({transformers.__version__}) — se divergir, REGISTRAR e confirmar V5-NORM com mais cuidado")
try:
    from transformers import VideoMAEImageProcessorFast
    check("import VideoMAEImageProcessorFast", True)
except ImportError as e:
    check("import VideoMAEImageProcessorFast", False, str(e))
try:
    import whisper, torchaudio
    check("import whisper + torchaudio", True)
    print("        openai-whisper:",
          subprocess.run([sys.executable, "-m", "pip", "show", "openai-whisper"],
                         capture_output=True, text=True).stdout.splitlines()[1]
          if True else "?")
except Exception as e:
    check("import whisper + torchaudio", False, str(e))
check("CUDA disponível (áudio precisa)", torch.cuda.is_available())

# 5. Árvores v2 devem estar VAZIAS ou inexistentes (V7: o skip resume-safe só é
#    correto se a árvore nasce vazia). Se existirem com conteúdo de uma tentativa
#    anterior abortada por bug de código, apagar antes; se abortada por falta de
#    disco/queda, o resume é legítimo.
for d in ["frames/v2/training", "frames/v2/test",
          "audio_features/v2/training", "audio_features/v2/test"]:
    p = ROOT/d
    n = len(list(p.glob("*.pt"))) if p.exists() else 0
    print(f"[info  ] {d}: {'inexistente' if not p.exists() else f'{n} .pt'}")

print("\n==>", "TODAS AS PRÉ-CONDIÇÕES OK" if ok else "HÁ FALHAS — PARAR")
```

**Resultado esperado:** todas as linhas `ok`; versões iguais à tabela do `CLAUDE.md`
(torch 2.6.0+cu124, transformers 5.6.0, numpy 1.26.4). A versão do `openai-whisper` não
está registrada em lugar nenhum — **anotá-la** (ela entra no `_extraction_meta.json` do
áudio).

---

## 4. As três decisões técnicas

### 4.1 T1 · V4-MEM — leitura sequencial em **dois passes com `grab()`/`retrieve()`** (opção a, refinada)

**Opções consideradas:**

| opção | memória | custo | risco |
|---|---|---|---|
| (a) dois passes: passe 1 conta, passe 2 guarda só os 64 alvo | O(64 frames em resolução nativa) ≈ ≤400 MB p/ 1080×1920 | decodifica 2× | nenhum estrutural |
| (b) um passe redimensionando na hora | ≤610 MB no pior caso ATUAL (135 s) `[MEDIDO na spec §V4-MEM]` | redimensiona todos os frames | **dois riscos**: (i) o teto de memória depende do dataset — um vídeo mais longo numa edição futura estoura; (ii) o redimensionamento sairia do processor e entraria no loop de leitura, ou seja, encostaria exatamente no ponto de maior risco da spec (V5-NORM) |

**Escolha: (a), com uma otimização que a spec não menciona** — usar `cap.grab()` (que
demultiplexa e decodifica sem converter/copiar o frame para o usuário) nos dois passes, e
`cap.retrieve()` **só nos 64 índices selecionados** do passe 2. Assim o custo do "decodifica
2×" cai: a parte cara de conversão de cor/cópia só acontece 64 vezes por vídeo.

**Justificação:**
1. Memória O(64) independente da duração — a garantia estrutural que (b) não tem.
2. Mantém os frames em **resolução nativa até a única chamada do processor**, que é
   exatamente o fluxo de dados da v1. O resize/crop continua acontecendo num único lugar
   (dentro do processor), o que minimiza a superfície de risco do V5-NORM. Com (b), o
   resize migraria para o loop de leitura e a equivalência com a v1 dependeria de
   reimplementação manual — o erro que a spec manda evitar.
3. VFR-seguro: a contagem do passe 1 é o número real de frames decodificáveis, não
   `CAP_PROP_FRAME_COUNT`.

**Decisão tomada pelo caminho conservador** porque a medição decisiva (velocidade de
decodificação na VM) não pode ser feita localmente. **Como confirmar `[VM]`:** a célula 5
do `extract_video_v2.ipynb` (§5) cronometra 10 vídeos e projeta o total. Critério: se a
projeção para os 3 198 vídeos passar de **3 h**, PARAR e reportar — nesse cenário vale
reavaliar (b) ou paralelizar por processo. Abaixo de 3 h, seguir.

**Efeito colateral aceito:** um `retrieve()` que falhe num frame `grab()`-ado (arquivos
truncados — o Agente 3 mediu da ordem de 1–4 % dos vídeos de teste com fim truncado,
`03_OPTIMIZATION_FINDINGS.md`, reconciliação do ARCH-09) é preenchido com o **frame real
mais próximo já decodificado** — nunca com zeros, nunca com dupla conversão (V1/V3). O
número de preenchimentos por vídeo vai para `_extraction_warnings.json`. Se >5 % dos
vídeos de um split tiverem preenchimento, PARAR (§10).

---

### 4.2 T2 · V5-NORM — **o mesmo processor, com `do_rescale=False, do_normalize=False`**; nada reimplementado à mão

**O que foi investigado (no `transformers` local, 4.57.6 — a classe base
`BaseImageProcessorFast` de onde o `VideoMAEImageProcessorFast` da VM herda):**

1. A ordem interna do `_preprocess` é **resize → center_crop → rescale_and_normalize**,
   e os três estágios são separáveis por flags de chamada (`do_rescale`, `do_normalize`
   são kwargs aceitos na própria chamada, não só no construtor).
2. `rescale_and_normalize` com ambos `False` é um **no-op** — devolve o tensor como saiu
   do crop, sem cast para float (`if do_normalize ... elif do_rescale ... return images`).
3. A preparação do input converte numpy `uint8 [H,W,C]` em tensor `uint8 [C,H,W]`
   **sem cast para float** (`torch.from_numpy(...).contiguous()` + `permute`).
4. O resize é `torchvision.transforms.v2.functional.resize(..., antialias=True)`, que
   **preserva dtype uint8** para entrada uint8 (caminho nativo uint8 do torchvision).
5. Na v1, o par rescale+normalize era **fundido**: `normalize(x.float(), mean·255, std·255)`
   (`_fuse_mean_std_and_rescale_factor`). Logo `(x/255 − mean)/std` aplicado no load do v2
   é **matematicamente idêntico** ao que a v1 gravou — diferença apenas de ordem de
   arredondamento float32, ~1 ulp.

**Consequência:** se a VM se comportar como a classe base local,
`processor(images=frames_list, do_rescale=False, do_normalize=False, return_tensors="pt")`
devolve **uint8 já com o resize/crop bit-idêntico ao da v1**, e a equivalência do teste 4
deve fechar em ~1e−6 — muito abaixo da tolerância de 0,017 da spec.

**`[VM] — obrigatório confirmar antes de extrair**, porque (i) a versão local (4.57.6)
nem sequer TEM o `VideoMAEImageProcessorFast` (só a classe base), e (ii) a VM está em
5.6.0. A confirmação é a **célula 3 do notebook** (§5): um probe sintético que
- chama o processor no modo v1 (default) e no modo v2 (`do_rescale=False, do_normalize=False`)
  sobre a MESMA imagem uint8;
- `assert` de que a saída v2 é `torch.uint8` (se vier float: ativa o caminho de fallback
  `round().clamp(0,255).to(uint8)`, cujo erro máximo é 0,5/255/0,225 ≈ **0,0087** no
  espaço normalizado — ainda dentro da tolerância de 0,017 da spec, mas o fato de ter
  caído no fallback é REGISTRADO no meta);
- reconstrói `(uint8/255 − mean)/std` e compara com a saída v1: **PARAR se
  `max|Δ| > 0,017`** (indicaria resize/crop diferente entre os dois modos — o cenário
  que invalida a comparação com o baseline).

**`size`/`crop_size` NÃO são passados explicitamente** — a v1 também não passou; os dois
modos usam o que veio de `from_pretrained("MCG-NJU/videomae-base")`. Passá-los à mão
seria introduzir uma segunda fonte de verdade. Em vez disso, `processor.to_dict()` inteiro
vai para o `_extraction_meta.json`, e `image_mean`/`image_std` são **lidos do processor**
(nunca hardcodados) — requisito 2 da spec §V5-NORM.

**Ressalva declarada:** o teste 4 (§9) tem duas camadas. T4a compara a v1 *recomputada
hoje* com os `.pt` de `frames_v1_sample/` — se T4a divergir, o ambiente da VM mudou desde
maio (torchvision/transformers) e isso é registrado como fato sobre o v1, não como bug do
v2. T4b (o teste que decide) compara v2+desnormalização contra a v1 recomputada **sobre
os mesmos frames de entrada** — é este que tem de fechar.

---

### 4.3 T3 · C2 — desnormalização **no `Dataset` (CPU, dentro de `load_frames`)**, não no `forward`

**Opções consideradas:**

| opção | prós | contras |
|---|---|---|
| (i) `Dataset`/workers (CPU) | contida num único lugar (`load_frames`); tensor que sai do dataset tem EXATAMENTE a mesma semântica da v1 (float32 normalizado); zero mudança no modelo, no collate, no Trainer, nos probes de ablação | trabalho por amostra em CPU com `dataloader_num_workers=2`; PCIe transporta float32 (4×) como hoje |
| (ii) `forward` (GPU, em lote) | PCIe transporta uint8 (4× menos); normalização em lote é ~grátis na GPU | espalha responsabilidade por dois arquivos; **muda a semântica de qualquer zeragem de frames a jusante**: os probes `probe_zero_video` do `ablation_harness.py` e qualquer máscara sobre `frames` zerariam o tensor uint8, e `(0−mean)/std ≈ −2,1` (preto) ≠ `0` normalizado (pixel médio do ImageNet) que era o comportamento v1 — um confound novo entre gerações de experimento |

**Escolha: (i), Dataset.** Três razões, em ordem de peso:

1. **"Uma variável de cada vez."** O primeiro re-treino pós-extração precisa que a única
   diferença em relação ao baseline seja o conteúdo das features. Com (i), tudo do
   `EXISTVideoDataset.__getitem__` para a frente é bit-compatível com o contrato v1
   (`frames: float32 [16,3,224,224] normalizado`). Com (ii), o `forward` muda junto.
2. **Semântica de zeragem.** O modality dropout atual mascara `vid_feat` (o embedding
   pós-pooling, `video_core.py` linha ~997), então não seria afetado — mas os probes de
   ablação e qualquer manipulação futura de `frames` seriam. (i) elimina a classe de
   problema.
3. **A aritmética do custo de CPU não assusta.** Por amostra: subamostrar 16 de 64
   (indexação, ~2,4 MB copiados) + `float().div_().sub_().div_()` sobre 2,4 M elementos —
   da ordem de poucos ms por amostra em CPU `[NÃO VERIFICADO]`. No batch efetivo atual
   (micro-batch 2), irrelevante; no cenário OPT-04 (micro-batch 8), 2 workers precisam
   sustentar ~8 amostras/passo — o gargalo esperado continua sendo o DISCO (agora 4×
   menor por byte lido: lê-se 9,6 MB uint8 e usa-se 2,4 MB), não a normalização.

**Interação com OPT-04, declarada:** a spec avisa que (i) "pode virar o novo gargalo".
Aceito o risco porque OPT-04 ainda não entrou (pendente) e a decisão é reversível por
medição. **Como confirmar `[VM]`, quando OPT-04 for ativado:**

```python
# célula de medição de throughput do dataloader (rodar na VM, sem GPU envolvida)
import time
from torch.utils.data import DataLoader
dl = DataLoader(train_dataset, batch_size=8, num_workers=2,
                collate_fn=vc.video_collate, shuffle=True)
it = iter(dl); next(it)                      # aquece workers
t0 = time.perf_counter(); n = 0
for _ in range(50):
    next(it); n += 8
dt = time.perf_counter() - t0
print(f"{n/dt:.1f} amostras/s (o passo de treino do OPT-04 precisa de ~8/passo; "
      f"comparar com a duração medida do passo)")
```
Se o dataloader não sustentar o passo da GPU, mover a normalização para o `forward` é a
otimização seguinte — **com a ressalva da zeragem acima documentada e um braço de
verificação de equivalência** (mesma loss nos primeiros passos com seeds idênticas).

**Detalhe de implementação que a decisão fixa:** a ordem dentro de `load_frames` é
**subamostrar primeiro, desnormalizar depois** — desnormalizar 16 frames custa 1/4 de
desnormalizar 64.

---

## 5. `extract_video_v2.ipynb` — código integral, célula a célula

Criar o notebook em `EXIST_VIDEOS/extract_video_v2.ipynb` com as células abaixo, na
ordem. Executar **na VM** (Kernel → Restart & Run All), primeiro com `SPLIT = "training"`,
depois editar a célula 2 para `SPLIT = "test"` e Restart & Run All de novo.

### Célula 0 (markdown)

```markdown
# extract_video_v2 — extração de frames corrigida (V1–V7 da spec 10)

Substitui `extract_video.ipynb` (v1, PRESERVADO como registro — decisão D3).
Diferenças em relação à v1, cada uma com o porquê:

- **V1** conversão BGR→RGB feita UMA vez, após a seleção dos frames (a v1 convertia
  dentro do loop e re-convertia o frame de fallback, devolvendo-o a BGR).
- **V2/D4** falha NÃO grava tensor de zeros (que, normalizado, era o pixel médio do
  ImageNet — dado plausível, não buraco). Falha → ID em `_missing.json`; o dataset
  falha alto no build (C4 do `video_core.py`).
- **V3** nenhum `except Exception` engolindo bugs de shape: os 64 frames têm o mesmo
  shape por construção (índices escolhidos DEPOIS de contar).
- **V4** leitura sequencial em 2 passes (`grab`/`retrieve`) — sem `cap.set` (salto
  impreciso) nem `CAP_PROP_FRAME_COUNT` (estimado, inválido sob VFR).
- **V5/D1** grava `uint8` [64,3,224,224] SEM rescale/normalize (9,6 MB/vídeo — os
  mesmos bytes da v1). A normalização ImageNet é aplicada no carregamento
  (`video_core.load_frames`). Resize/crop continuam DENTRO do processor → idênticos
  aos da v1 (V5-NORM). A célula 3 prova isso antes de qualquer extração.
- **V6** `SPLIT` é a única linha editável; caminhos derivados; nomes reais dos JSON
  mapeados explicitamente (treino `EXIST2026_training.json`, teste
  `EXIST2026_test_clean.json` — NÃO seguem o mesmo padrão). `_extraction_meta.json`
  gravado ao lado das features.
- **V7** grava em `frames/v2/{split}/` — o skip resume-safe volta a ser correto
  porque a árvore nasce vazia.

Rodar 2×: `SPLIT="training"` e `SPLIT="test"` (Restart & Run All em cada).
```

### Célula 1 (code) — dependências (inofensiva se já instaladas)

```python
%pip install --quiet opencv-python-headless
```

### Célula 2 (code) — configuração (V6: UMA constante, caminhos derivados)

```python
#!/usr/bin/env python
# coding: utf-8
"""
Extração de frames v2 para VideoMAE — ver célula 0 (markdown) para o mapa de
correções V1–V7. Salva [64, 3, 224, 224] uint8 (resize+crop do processor, SEM
rescale/normalize) em frames/v2/{split}/{id_EXIST}.pt.
"""
import json, platform, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import VideoMAEImageProcessorFast
import transformers

# ── Config ───────────────────────────────────────────────────────────────────
SPLIT = "training"        # ← A ÚNICA LINHA A EDITAR: "training" ou "test" (V6)

# V6: os JSON NÃO seguem o mesmo padrão de nome — mapeamento explícito, sem f-string.
DATASET_JSON = {
    "training": Path("./Dataset/training/EXIST2026_training.json"),
    "test":     Path("./Dataset/test/EXIST2026_test_clean.json"),
}
EXPECTED_N = {"training": 2524, "test": 674}   # contagens conhecidas (spec §"Estado")

VIDEO_DIR        = Path(f"./Dataset/{SPLIT}/videos/")
OUTPUT_DIR       = Path(f"./frames/v2/{SPLIT}/")           # V7: árvore versionada
JSON_PATH        = DATASET_JSON[SPLIT]
NUM_FRAMES_SAVED = 64        # D1: 64 uint8 = mesmos bytes de 16 float32 (9 633 792 B)
MODEL_NAME       = "MCG-NJU/videomae-base"                 # NÃO MUDAR (spec §4)
# ─────────────────────────────────────────────────────────────────────────────

assert SPLIT in DATASET_JSON, f"SPLIT inválido: {SPLIT!r}"
assert JSON_PATH.exists(), f"JSON não encontrado: {JSON_PATH}"
assert VIDEO_DIR.is_dir(), f"Diretório de vídeos não encontrado: {VIDEO_DIR}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

processor = VideoMAEImageProcessorFast.from_pretrained(MODEL_NAME)
# V5-NORM req.2: mean/std/size lidos DO PROCESSOR (nunca hardcodados) — vão para o meta
# e para o video_core.load_frames via _extraction_meta.json.
CROP_H = processor.crop_size["height"]
CROP_W = processor.crop_size["width"]
print(f"processor: size={processor.size} crop={processor.crop_size} "
      f"mean={processor.image_mean} std={processor.image_std}")

df = pd.read_json(JSON_PATH).transpose()   # mesmo parse da v1
assert len(df) == EXPECTED_N[SPLIT], \
    f"JSON de {SPLIT} tem {len(df)} itens; esperado {EXPECTED_N[SPLIT]} — PARAR e investigar"
print(f"{SPLIT}: {len(df)} itens → {OUTPUT_DIR}")
```

### Célula 3 (code) — **probe V5-NORM (obrigatório; ABORTA o notebook se falhar)**

```python
# ── Probe V5-NORM: prova, ANTES de extrair, que desligar rescale/normalize não
# altera o resize/crop, e mede o erro de armazenamento uint8.
# Decisão T2 do plano 11: mesma chamada de processor da v1, com
# do_rescale=False, do_normalize=False. Tolerância dura: 0,017 (spec §V5-NORM).
rng = np.random.default_rng(0)
NEEDS_CAST = False          # vira True se o processor devolver float no modo v2
worst = 0.0
for (h, w) in [(360, 640), (1920, 1080), (720, 720)]:   # paisagem, retrato TikTok, quadrado
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    full = processor(images=[img], return_tensors="pt")["pixel_values"]       # modo v1
    raw  = processor(images=[img], do_rescale=False, do_normalize=False,
                     return_tensors="pt")["pixel_values"]                     # modo v2
    if raw.dtype != torch.uint8:
        # Fallback: quantiza explicitamente. Erro máx = 0,5/255/0,225 ≈ 0,0087
        # no espaço normalizado — dentro da tolerância, mas REGISTRADO no meta.
        NEEDS_CAST = True
        raw = raw.round().clamp(0, 255).to(torch.uint8)
    mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    std  = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    recon = (raw.float() / 255.0 - mean) / std     # exatamente o que load_frames fará
    diff = (recon - full).abs().max().item()
    worst = max(worst, diff)
    print(f"  {h}x{w}: shape v2={tuple(raw.shape)} dtype={raw.dtype} max|Δ|={diff:.2e}")

print(f"\nNEEDS_CAST={NEEDS_CAST}  pior max|Δ|={worst:.2e}")
assert worst <= 0.017, (
    "V5-NORM REPROVADO: a reconstrução (uint8/255−mean)/std diverge do modo v1 além "
    "da tolerância de quantização — o resize/crop MUDOU entre os modos. PARAR. "
    "NÃO extrair. Ver plano 11 §10 (modos de falha).")
if NEEDS_CAST:
    print("[AVISO] processor devolveu float no modo v2; usando cast explícito "
          "(erro ≤0,0087, dentro da tolerância). Registrado no meta como needs_cast=true.")
```

### Célula 4 (code) — funções de leitura (V1+V3+V4, decisão T1: dois passes grab/retrieve)

```python
def count_decodable_frames(video_path: Path) -> int:
    """Passe 1 (V4): conta os frames REALMENTE decodificáveis, sequencialmente.

    `cap.grab()` demultiplexa+decodifica sem converter/copiar o frame — é a forma
    barata de contar. NUNCA usar CAP_PROP_FRAME_COUNT (estimado por duração×fps,
    inválido para VFR de TikTok) nem cap.set(POS_FRAMES) (salto aproximado
    para o keyframe mais próximo). Ver spec §V4.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def retrieve_frames_at(video_path: Path, targets_sorted: list) -> dict:
    """Passe 2 (V4/T1): percorre sequencialmente com grab(); retrieve() SÓ nos
    índices alvo. Memória O(len(targets)) em resolução nativa (≤ ~400 MB para
    64 frames 1080×1920) — nunca o vídeo inteiro (V4-MEM).
    Devolve {índice: frame BGR nativo} com o que conseguiu decodificar.
    """
    got = {}
    cap = cv2.VideoCapture(str(video_path))
    pos, ti = 0, 0
    while ti < len(targets_sorted):
        if not cap.grab():
            break                      # fim real (ou truncamento) — sem inventar nada
        if pos == targets_sorted[ti]:
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                got[pos] = frame
            ti += 1                    # alvo consumido mesmo se retrieve falhou
        pos += 1
    cap.release()
    return got


def sample_frames_v2(video_path: Path, n: int):
    """Amostragem uniforme sobre a contagem REAL (V4), sem fallbacks fabricados.

    Devolve (frames_rgb [n,H,W,3] uint8, info) ou (None, info) se o vídeo não
    tem nenhum frame decodificável (o chamador manda o ID para _missing.json — D4).

    - Índices escolhidos DEPOIS de contar → shapes iguais por construção; o
      np.stack não pode falhar por mistura com fallback 224×224 (V3).
    - Se um retrieve falhar (arquivos truncados: ~1–4 % do split de teste,
      medição do Agente 3 em 03_OPTIMIZATION_FINDINGS.md), preenche com o frame
      REAL mais próximo já decodificado — nunca zeros, e a contagem de
      preenchimentos é reportada (info["n_filled"]).
    - Conversão BGR→RGB UMA vez, no fim, só nos selecionados (V1).
    """
    total = count_decodable_frames(video_path)
    if total == 0:
        return None, {"reason": "0 frames decodificáveis no passe 1"}
    wanted = np.linspace(0, total - 1, n).round().astype(int)   # com repetição se total<n
    targets = sorted(set(wanted.tolist()))
    got = retrieve_frames_at(video_path, targets)
    if not got:
        return None, {"reason": f"passe 2 não decodificou nenhum frame (passe 1 contou {total})"}
    avail = sorted(got)
    n_filled = 0
    frames = []
    for idx in wanted:
        if idx in got:
            frames.append(got[idx])
        else:
            nearest = min(avail, key=lambda a: abs(a - idx))
            frames.append(got[nearest])
            n_filled += 1
    frames_bgr = np.stack(frames)                            # [n, H, W, 3] BGR — shapes iguais
    frames_rgb = np.ascontiguousarray(frames_bgr[..., ::-1]) # V1: UMA conversão, no fim
    return frames_rgb, {"total_frames": int(total), "n_filled": int(n_filled)}


def process_frames_v2(frames_rgb: np.ndarray) -> torch.Tensor:
    """V5/T2: MESMO processor da v1, com rescale/normalize desligados.
    Resize (aresta menor) + center-crop idênticos aos da v1; saída uint8.
    A normalização ImageNet acontece no carregamento (video_core.load_frames, C2).
    """
    processed = processor(
        images=[frames_rgb[i] for i in range(frames_rgb.shape[0])],
        do_rescale=False, do_normalize=False,
        return_tensors="pt",
    )
    pv = processed["pixel_values"].squeeze(0)                # [n, 3, H, W]
    if pv.dtype != torch.uint8:                              # fallback do probe (célula 3)
        pv = pv.round().clamp(0, 255).to(torch.uint8)
    assert pv.shape == (NUM_FRAMES_SAVED, 3, CROP_H, CROP_W), \
        f"shape inesperado: {tuple(pv.shape)}"
    return pv
```

### Célula 5 (code) — sondagem de tempo (gate da decisão T1; NÃO grava nada)

```python
# ── Projeção de custo ANTES da run completa (decisão T1 do plano 11).
# Critério: projeção > 3 h para os 3 198 vídeos → PARAR e reportar (reavaliar a
# opção (b) do V4-MEM ou paralelizar). Esta célula não grava nenhum arquivo.
probe_rows = df.head(10)
t0 = time.perf_counter()
for _, row in probe_rows.iterrows():
    vp = VIDEO_DIR / Path(str(row["path_video"])).name
    if not vp.exists():
        continue
    frames_rgb, info = sample_frames_v2(vp, NUM_FRAMES_SAVED)
    if frames_rgb is not None:
        _ = process_frames_v2(frames_rgb)
dt = time.perf_counter() - t0
per_item = dt / len(probe_rows)
proj_min = per_item * 3198 / 60
print(f"{per_item:.2f} s/vídeo → projeção p/ 3 198 vídeos: {proj_min:.0f} min "
      f"({proj_min/60:.1f} h)")
assert proj_min <= 180, (
    f"Projeção {proj_min:.0f} min > 180 min: PARAR e reportar (plano 11 §4.1). "
    "Não iniciar a run completa.")
```

### Célula 6 (code) — loop principal (V2/D4: falha → manifesto, nunca zeros)

```python
missing, warns = [], []
t_start = time.perf_counter()

for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Frames v2 [{SPLIT}]"):
    item_id  = str(row["id_EXIST"])
    out_path = OUTPUT_DIR / f"{item_id}.pt"

    # Resume-safe — CORRETO na v2 porque a árvore nasce vazia (V7). Numa retomada
    # após queda, os itens já gravados são pulados e os que falharam são re-tentados
    # (falha não grava arquivo, então volta a entrar aqui).
    if out_path.exists():
        continue

    # SIMP-10: UMA resolução de caminho (o "fallback" da v1 era um no-op idêntico).
    video_path = VIDEO_DIR / Path(str(row["path_video"])).name
    if not video_path.exists():
        # V2/D4: NÃO gravar zeros (zeros normalizados = pixel médio do ImageNet =
        # dado plausível). Ausência vira manifesto; o dataset falha alto no build.
        missing.append({"id": item_id, "reason": f"vídeo não encontrado: {video_path.name}"})
        continue

    try:
        frames_rgb, info = sample_frames_v2(video_path, NUM_FRAMES_SAVED)
    except Exception as e:
        # Diferente do except da v1 (que devolvia None → zeros no disco, V3):
        # aqui a exceção é IMPRESSA INTEIRA e o item vai para o manifesto.
        traceback.print_exc()
        missing.append({"id": item_id, "reason": f"{type(e).__name__}: {e}"})
        continue

    if frames_rgb is None:
        missing.append({"id": item_id, "reason": info["reason"]})
        continue
    if info["n_filled"] > 0:
        warns.append({"id": item_id, **info})

    torch.save(process_frames_v2(frames_rgb), out_path)

elapsed_min = (time.perf_counter() - t_start) / 60
n_saved = len(list(OUTPUT_DIR.glob("*.pt")))
print(f"\n{SPLIT}: {n_saved} gravados | {len(missing)} faltantes | "
      f"{len(warns)} com preenchimento | {elapsed_min:.1f} min")
```

### Célula 7 (code) — proveniência (V6): `_extraction_meta.json`, `_missing.json`, `_extraction_warnings.json`

```python
def _pip_version(pkg: str) -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "show", pkg],
                             capture_output=True, text=True).stdout
        return next(l.split(":", 1)[1].strip() for l in out.splitlines()
                    if l.startswith("Version:"))
    except Exception:
        return "desconhecida"

meta = {
    # o que define o resultado (spec §V6, item 2) — NADA hardcodado:
    "split": SPLIT,
    "num_frames_saved": NUM_FRAMES_SAVED,
    "dtype": "uint8",
    "storage": "resize+center_crop do processor, SEM rescale e SEM normalize (V5); "
               "normalização ImageNet aplicada no carregamento (video_core.load_frames)",
    "needs_cast_fallback": bool(NEEDS_CAST),   # probe da célula 3
    "model_name": MODEL_NAME,
    "image_mean": list(processor.image_mean),  # lidos DO processor (req. V5-NORM 2)
    "image_std": list(processor.image_std),
    "size": dict(processor.size),
    "crop_size": dict(processor.crop_size),
    "processor_config": processor.to_dict(),
    "sampling": "2 passes sequenciais (grab/retrieve); índices = "
                "linspace(0, total_real-1, 64).round(); BGR→RGB uma vez após seleção",
    "n_items": int(len(df)),
    "n_saved": int(n_saved),
    "n_missing": int(len(missing)),
    "n_with_fill": int(len(warns)),
    "versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "opencv_pip": _pip_version("opencv-python-headless"),
    },
    "date": datetime.now(timezone.utc).isoformat(),
    "notebook": "extract_video_v2.ipynb",
}
with open(OUTPUT_DIR / "_extraction_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# D4: manifesto SEMPRE gravado, mesmo vazio — ausência do arquivo passa a significar
# "extração não terminou", não "não houve falhas".
with open(OUTPUT_DIR / "_missing.json", "w") as f:
    json.dump({"split": SPLIT, "missing": missing, "date": meta["date"]}, f, indent=2)
with open(OUTPUT_DIR / "_extraction_warnings.json", "w") as f:
    json.dump({"split": SPLIT, "filled": warns, "date": meta["date"]}, f, indent=2)

print(f"meta/manifesto gravados em {OUTPUT_DIR}")
print(f"faltantes: {len(missing)}")
if missing:
    print("IDs:", [m["id"] for m in missing][:20])
    print("⚠ A v1 teve 0 falhas nestes dados (spec §V2, medido). QUALQUER faltante "
          "aqui é regressão do script — investigar antes de seguir (plano 11 §10).")
frac_fill = len(warns) / max(1, len(df))
if frac_fill > 0.05:
    print(f"⚠ {frac_fill:.1%} dos vídeos precisaram de preenchimento (>5%): "
          "PARAR e investigar (plano 11 §10).")
```

**Observações para o executor:**
- A célula 6 é retomável após queda (V7 + manifesto re-gerado na retomada). Após uma
  retomada, `_extraction_warnings.json` cobre só os itens da última run — aceitável; o
  `_missing.json` cobre tudo (itens falhados não têm `.pt` e são re-tentados).
- O notebook NUNCA deve ser rodado com a célula 3 pulada. Ela é o portão do V5-NORM.

---

## 6. `extract_audio_v2.ipynb` — código integral, célula a célula

Criar em `EXIST_VIDEOS/extract_audio_v2.ipynb`. Executar **na VM**, primeiro com
`SPLIT = "training"`, depois `SPLIT = "test"` (Restart & Run All em cada).

Nota sobre partilha de código (spec §A4): as funções de manifesto/meta são
**duplicadas de propósito** entre os dois notebooks — a spec veta introduzir um módulo
partilhado novo que complique a execução na VM. Manter as duplicatas idênticas.

### Célula 0 (markdown)

```markdown
# extract_audio_v2 — extração de áudio corrigida (A1–A5 da spec 10)

Substitui `extract_audio.ipynb` (v1, PRESERVADO — decisão D3). Correções:

- **A1 (máscara do padding):** a v1 fazia `pad_or_trim` até 30 s e promediava os
  1500 frames do encoder INCLUINDO o silêncio sintético — 34,5 % do vetor médio de
  treino era encoder de silêncio, e a fração variava com a duração (confound:
  o modelo podia ler a DURAÇÃO, não o áudio). Agora a média usa só os
  `n_valid = round(segundos_válidos × 50)` primeiros frames.
  ⚠ Ressalva a declarar na tese: a self-attention do encoder AINDA viu o padding
  dentro de cada janela; o que se remove é a diluição da MÉDIA, não o efeito do
  padding no encoder (que é o regime de treino do próprio Whisper).
- **A2 (janelas múltiplas):** a v1 CORTAVA em 30 s — 24,5 % do áudio de treino
  nunca chegava ao encoder (33,4 % dos vídeos passam de 30 s; o pior item de teste
  perdia 77,8 %). Agora o áudio é percorrido em janelas de 30 s e a média é
  ponderada pelos frames válidos de cada janela — matematicamente IGUAL à média
  sobre todos os frames válidos do áudio inteiro (não é heurística).
- **A3:** resolvido de graça por A1+A2 (o shift treino 0,655 vs teste 0,582 de
  `valid_fraction` deixa de existir como variável).
- **A4 = V2+V6+V7 do áudio:** falha → `_missing.json` (nunca `torch.zeros(1280)`
  no disco); `SPLIT` único + caminhos derivados + `_extraction_meta.json`;
  árvore `audio_features/v2/{split}/`.
- **A5:** ramo `else` no-op removido do loader mono.

O shape de saída continua [1280] float32 — o consumidor não muda de contrato.
Fontes dos números: analysis/audio_durations.csv, improvements/07, spec §A1–A3.
```

### Célula 1 (code) — configuração

```python
#!/usr/bin/env python
# coding: utf-8
"""
Extração de áudio v2 (Whisper encoder-only, média mascarada e janelada).
Ver célula 0 para o mapa de correções A1–A5. Salva [1280] float32 em
audio_features/v2/{split}/{id_EXIST}.pt.
"""
import json, platform, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import whisper
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
SPLIT = "training"        # ← A ÚNICA LINHA A EDITAR: "training" ou "test" (V6/A4)

DATASET_JSON = {          # V6: nomes reais dos JSON não seguem o mesmo padrão
    "training": Path("./Dataset/training/EXIST2026_training.json"),
    "test":     Path("./Dataset/test/EXIST2026_test_clean.json"),
}
EXPECTED_N = {"training": 2524, "test": 674}

VIDEO_DIR     = Path(f"./Dataset/{SPLIT}/videos/")
OUTPUT_DIR    = Path(f"./audio_features/v2/{SPLIT}/")      # V7/A4: árvore versionada
JSON_PATH     = DATASET_JSON[SPLIT]
WHISPER_MODEL = "large-v3"                                 # NÃO MUDAR (spec §4)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR     = 16000                                      # NÃO MUDAR (spec §4)

# Constantes do encoder do Whisper: janela de 30 s → 1500 frames → 50 frames/s.
CHUNK_SECONDS   = 30
CHUNK_SAMPLES   = CHUNK_SECONDS * TARGET_SR                # 480 000
N_ENC_FRAMES    = 1500
FRAMES_PER_SEC  = N_ENC_FRAMES // CHUNK_SECONDS            # 50
# ─────────────────────────────────────────────────────────────────────────────

assert SPLIT in DATASET_JSON, f"SPLIT inválido: {SPLIT!r}"
assert JSON_PATH.exists(), f"JSON não encontrado: {JSON_PATH}"
assert VIDEO_DIR.is_dir(), f"Diretório de vídeos não encontrado: {VIDEO_DIR}"
assert DEVICE == "cuda", "Sem GPU: a extração de áudio precisa de CUDA (pré-condição P0)"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Carregando Whisper {WHISPER_MODEL} em {DEVICE}...")
model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
model.eval()

df = pd.read_json(JSON_PATH).transpose()
assert len(df) == EXPECTED_N[SPLIT], \
    f"JSON de {SPLIT} tem {len(df)} itens; esperado {EXPECTED_N[SPLIT]} — PARAR"
print(f"{SPLIT}: {len(df)} itens → {OUTPUT_DIR}")

resampler_cache = {}
```

### Célula 2 (code) — funções (A1, A2, A5)

```python
def load_audio_mono_16k(video_path: Path):
    """Waveform [T] float32 mono a 16 kHz, ou None em falha (→ _missing.json, D4).
    A5: o ramo `else: waveform = waveform` (no-op) da v1 foi removido."""
    try:
        waveform, sr = torchaudio.load(str(video_path))        # [C, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)      # [1, T]
        if sr != TARGET_SR:
            if sr not in resampler_cache:
                resampler_cache[sr] = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler_cache[sr](waveform)
        return waveform.squeeze(0)                             # [T]
    except Exception as e:
        print(f"  [ERR] torchaudio.load falhou: {e}")
        return None


@torch.no_grad()
def extract_whisper_features_v2(waveform: torch.Tensor):
    """Encoder do Whisper com média MASCARADA (A1) e JANELADA (A2).

    A1 — dentro de cada janela de 30 s, a média usa só os primeiros
         n_valid = round(segundos_válidos × 50) frames do encoder; o resto é
         encoder de padding (silêncio sintético) e fica FORA da média.
    A2 — o áudio inteiro é percorrido em janelas de 30 s; a agregação final é a
         média das janelas PONDERADA por n_valid. Com pesos (n_valid_1, ...,
         n_valid_k) o resultado é exatamente a média sobre TODOS os frames
         válidos do áudio — o mesmo número que A1 daria se o Whisper não tivesse
         limite de 30 s (justificação matemática, spec §A2).

    Nota sobre o pseudocódigo da spec §A1: lá está `encoder_out[:n_valid]`, que
    fatiaria a dimensão de BATCH ([1,1500,1280]); o índice correto é
    `encoder_out[0, :n_valid]` (registrado no plano 11 §12).

    Devolve (features [1280] float32 em CPU, n_windows) ou (None, 0) se o
    waveform está vazio.
    """
    n_samples = int(waveform.shape[0])
    if n_samples == 0:
        return None, 0
    feats, weights = [], []
    for start in range(0, n_samples, CHUNK_SAMPLES):
        chunk = waveform[start:start + CHUNK_SAMPLES]
        valid_seconds = chunk.shape[0] / TARGET_SR
        n_valid = max(1, min(N_ENC_FRAMES, int(round(valid_seconds * FRAMES_PER_SEC))))
        audio = whisper.pad_or_trim(chunk.numpy())             # [480000]
        mel = whisper.log_mel_spectrogram(
            audio, n_mels=model.dims.n_mels).to(DEVICE)        # [n_mels, 3000]
        enc = model.encoder(mel.unsqueeze(0))                  # [1, 1500, 1280]
        feats.append(enc[0, :n_valid].mean(dim=0))             # A1: só frames válidos
        weights.append(float(n_valid))
    w = torch.tensor(weights, device=feats[0].device)
    features = (torch.stack(feats, dim=0) * w[:, None]).sum(dim=0) / w.sum()  # A2
    return features.cpu().float(), len(feats)
```

### Célula 3 (code) — auto-teste da agregação (CPU, sem Whisper; ABORTA se falhar)

```python
# ── Prova de que "média ponderada das janelas == média plana dos frames válidos".
# Teste puro de aritmética, sem encoder: se isto falhar, a implementação da
# agregação está errada e NADA deve ser extraído.
g = torch.Generator().manual_seed(0)
fake_frames = torch.randn(1500 + 1000, 1280, generator=g)   # 50 s de "frames válidos"
win1, win2 = fake_frames[:1500], fake_frames[1500:]         # janelas de 30 s e 20 s
w = torch.tensor([1500.0, 1000.0])
weighted = (torch.stack([win1.mean(0), win2.mean(0)]) * w[:, None]).sum(0) / w.sum()
flat = fake_frames.mean(0)
assert torch.allclose(weighted, flat, atol=1e-5), \
    "agregação ponderada != média plana — bug na célula 2, PARAR"
print(f"ok: max|Δ| = {(weighted - flat).abs().max():.2e} (esperado ~1e-7, float32)")

# n_valid: casos de borda
for secs, expect in [(30.0, 1500), (0.001, 1), (18.19, 910), (0.649, 32)]:
    nv = max(1, min(N_ENC_FRAMES, int(round(secs * FRAMES_PER_SEC))))
    assert nv == expect, (secs, nv, expect)
print("ok: n_valid nos casos de borda (30 s→1500, mediana treino 18,19 s→910, "
      "resto de 60,649 s→32, mínimo→1)")
```

### Célula 4 (code) — loop principal (A4/D4)

```python
missing = []
window_hist = {}          # nº de janelas → contagem de itens (alimenta o teste 7)

for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Áudio v2 [{SPLIT}]"):
    item_id  = str(row["id_EXIST"])
    out_path = OUTPUT_DIR / f"{item_id}.pt"

    if out_path.exists():        # resume-safe — correto: árvore nasce vazia (V7)
        continue

    video_path = VIDEO_DIR / Path(str(row["path_video"])).name
    if not video_path.exists():
        # A4=V2/D4: NUNCA torch.save(torch.zeros(1280)) — ausência vira manifesto.
        missing.append({"id": item_id, "reason": f"vídeo não encontrado: {video_path.name}"})
        continue

    waveform = load_audio_mono_16k(video_path)
    if waveform is None:
        missing.append({"id": item_id, "reason": "torchaudio.load falhou"})
        continue

    try:
        features, n_windows = extract_whisper_features_v2(waveform)
    except Exception as e:
        traceback.print_exc()
        missing.append({"id": item_id, "reason": f"{type(e).__name__}: {e}"})
        continue

    if features is None:
        missing.append({"id": item_id, "reason": "waveform vazio (0 amostras)"})
        continue

    assert features.shape == (1280,), f"shape inesperado: {tuple(features.shape)}"
    window_hist[n_windows] = window_hist.get(n_windows, 0) + 1
    torch.save(features, out_path)

n_saved = len(list(OUTPUT_DIR.glob("*.pt")))
print(f"\n{SPLIT}: {n_saved} gravados | {len(missing)} faltantes")
print(f"histograma de janelas: {dict(sorted(window_hist.items()))}")
# Referência calculada de analysis/audio_durations.csv (durações ffprobe; a
# fronteira exata pode deslocar ±poucos itens porque aqui conta o waveform real):
#   training: {1: 1682, 2: 821, 3: 21}   → 842 itens com >1 janela
#   test:     {1: 512, 2: 145, 3: 15, 4: 1, 5: 1} → 162 itens com >1 janela
multi = sum(v for k, v in window_hist.items() if k > 1)
print(f"itens com >1 janela: {multi} "
      f"(esperado ~842 em training / ~162 em test, ±5)")
```

### Célula 5 (code) — proveniência (A4/V6): meta + manifesto

```python
def _pip_version(pkg: str) -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "show", pkg],
                             capture_output=True, text=True).stdout
        return next(l.split(":", 1)[1].strip() for l in out.splitlines()
                    if l.startswith("Version:"))
    except Exception:
        return "desconhecida"

meta = {
    "split": SPLIT,
    "feature_shape": [1280],
    "dtype": "float32",
    "whisper_model": WHISPER_MODEL,
    "target_sr": TARGET_SR,
    "n_mels": int(model.dims.n_mels),
    "pooling": "média mascarada aos frames válidos (A1) sobre janelas de 30 s, "
               "ponderada por n_valid (A2); n_valid = round(segundos×50), clamp [1,1500]",
    "declared_caveat": "a self-attention do encoder ainda vê o padding dentro de cada "
                       "janela; removida a diluição da MÉDIA, não o efeito do padding "
                       "no encoder (spec §A1)",
    "n_items": int(len(df)),
    "n_saved": int(n_saved),
    "n_missing": int(len(missing)),
    "window_histogram": {str(k): v for k, v in sorted(window_hist.items())},
    "versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "openai_whisper": _pip_version("openai-whisper"),
        "numpy": np.__version__,
    },
    "date": datetime.now(timezone.utc).isoformat(),
    "notebook": "extract_audio_v2.ipynb",
}
with open(OUTPUT_DIR / "_extraction_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
with open(OUTPUT_DIR / "_missing.json", "w") as f:
    json.dump({"split": SPLIT, "missing": missing, "date": meta["date"]}, f, indent=2)

print(f"meta/manifesto gravados em {OUTPUT_DIR}")
if missing:
    print("⚠ A v1 teve 0 falhas de áudio nestes dados (0 sem stream, 0 ausentes — "
          "analysis/audio_durations.csv). QUALQUER faltante é regressão: investigar.")
    print("IDs:", [m["id"] for m in missing][:20])
```

**Observação:** numa retomada após queda, `window_hist` cobre só os itens da última run
(itens pulados pelo resume não passam pelo encoder). Se houve retomada, o teste 7 da §9
usa a recontagem independente lá descrita, não o histograma desta célula.

---

## 7. Alterações em `video_core.py` (C1–C4) — funções de substituição integrais

**Antes de editar:** fazer a cópia de segurança (o arquivo NÃO está versionado no git —
aparece como untracked; sem esta cópia não há rollback):

```bash
cp EXIST_VIDEOS/video_core.py EXIST_VIDEOS/video_core_pre_v2.py.bak
```

Convenção do projeto: **funções de substituição completas**, não diffs. Cada item abaixo
diz exatamente qual bloco substituir (nome + localização atual) e dá o texto integral
novo. Nenhuma outra parte do arquivo é tocada. Os imports existentes (`json`, `numpy as
np`, `torch`, `Path`) já cobrem tudo o que o código novo usa.

**Efeito nos consumidores:** `Thesis_video_improved.ipynb` (célula 12) chama
`vc.frames_dir("train")` / `vc.audio_dir("train")` — com o default `version="v2"` abaixo,
essas chamadas passam a resolver para as árvores v2 **sem editar o notebook**. É o
comportamento desejado (C1).

### 7.1 Constantes novas (C3) — inserir na seção 3, imediatamente abaixo da linha `NUM_CHANNELS = 3`

```python
# ── Extração v2 (plano 11 / spec 10) ────────────────────────────────────────
# D1: a v2 grava 64 frames uint8 por vídeo (mesmos 9 633 792 B que 16×float32 da
# v1). O modelo continua consumindo NUM_FRAMES=16: load_frames subamostra.
N_FRAMES_STORED_V2 = 64

# C3: quantos frames o modelo consome. DEFAULT 16 = regime do baseline (uma
# variável de cada vez no primeiro re-treino). A subamostragem é UNIFORMEMENTE
# ESPAÇADA sobre os 64 (reproduz o regime v1, gap mediano ~1,21 s); a janela
# densa (16 CONSECUTIVOS ≈ 4,6 s) é braço de ablação FUTURO, não default.
# ⚠ Mudar para 32/64 exige também reconfigurar o VideoMAE (num_frames≠16 muda
# os embeddings posicionais) — fora do escopo da extração v2.
N_FRAMES_USED = NUM_FRAMES     # = 16

# C1: geração de features que os helpers de caminho resolvem por default.
FEATURES_VERSION = "v2"
```

### 7.2 C1 — substituir `frames_dir` e `audio_dir` (seção 4, hoje nas linhas ~343–351)

Substituir as DUAS funções inteiras por:

```python
# v2 usa os nomes de split do Dataset ("training"/"test") — os mesmos que a
# constante SPLIT dos notebooks de extração (V6: um vocabulário só). Os aliases
# aceitam o "train" legado usado pelos notebooks de treino.
_V2_SPLIT_DIRNAME = {"train": "training", "training": "training", "test": "test"}


def frames_dir(split: str, root: str = ".", version: str = FEATURES_VERSION) -> Path:
    """Árvore de frames pré-extraídos.

    v2 (default): frames/v2/{training,test} — 64 frames uint8/vídeo, SEM
    normalização (extract_video_v2.ipynb; plano 11). `load_frames` desnormaliza.
    v1: frames/{train,test} — ⚠ APAGADA em 2026-08-08 (spec 10 §6); resta apenas
    frames_v1_sample/ (20 treino + 5 teste). O parâmetro `version="v1"` existe
    para comparações explícitas entre gerações, não para treino.
    """
    if version == "v1":
        return Path(root) / "frames" / ("train" if split in ("train", "training") else "test")
    return Path(root) / "frames" / version / _V2_SPLIT_DIRNAME[split]


def audio_dir(split: str, root: str = ".", version: str = FEATURES_VERSION) -> Path:
    """Árvore de features de áudio.

    v2 (default): audio_features/v2/{training,test} — média mascarada (A1) e
    janelada (A2) do encoder do Whisper (extract_audio_v2.ipynb; plano 11).
    v1: audio_features/{train,test} — PRESERVADA de propósito: é a referência do
    teste 6 da verificação (norma média 14,96 ± 0,50) e o insumo de qualquer
    ablação v1↔v2 de áudio.
    """
    if version == "v1":
        return Path(root) / "audio_features" / ("train" if split in ("train", "training") else "test")
    return Path(root) / "audio_features" / version / _V2_SPLIT_DIRNAME[split]
```

### 7.3 C2 + D4 — substituir a seção 5 inteira (`load_frames` e `load_audio_features`, hoje nas linhas ~359–400)

Substituir da linha do cabeçalho `# 5. IO de features pré-extraídas` até o fim de
`load_audio_features` por:

```python
# ============================================================================
# 5. IO de features pré-extraídas
# ============================================================================
# D4 (plano 11): ausência de dado NUNCA vira tensor de zeros — os loaders falham
# alto. Os antigos except→zeros eram, na prática, código morto (a extração v1
# gravava zeros ela mesma, BUG-D); agora que a extração v2 NÃO grava nada em
# falha, um arquivo ausente é um erro real e tem de gritar.

_EXTRACTION_META_CACHE: dict = {}


def load_extraction_meta(dir_path) -> dict:
    """Lê (e cacheia por diretório) o `_extraction_meta.json` de uma árvore v2.

    V6: é a proveniência da extração — mean/std/size do processor, versões,
    contagens. Ausência do arquivo significa extração incompleta ou caminho
    errado, e é erro (não warning): sem o meta não há como desnormalizar uint8
    sem hardcodar constantes, que é exatamente o que a spec proíbe (C2).
    """
    key = str(dir_path)
    if key not in _EXTRACTION_META_CACHE:
        mpath = Path(dir_path) / "_extraction_meta.json"
        if not mpath.exists():
            raise RuntimeError(
                f"_extraction_meta.json não encontrado em {dir_path} — árvore de "
                f"features sem proveniência (V6). Extração incompleta, caminho errado, "
                f"ou árvore v1 sendo usada onde se espera v2.")
        with open(mpath) as f:
            _EXTRACTION_META_CACHE[key] = json.load(f)
    return _EXTRACTION_META_CACHE[key]


def _frames_norm_constants(dir_path):
    """(mean, std) [3,1,1] float32, lidos do meta da extração — NUNCA hardcodados (C2)."""
    key = ("norm", str(dir_path))
    if key not in _EXTRACTION_META_CACHE:
        meta = load_extraction_meta(dir_path)
        mean = torch.tensor(meta["image_mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(meta["image_std"], dtype=torch.float32).view(3, 1, 1)
        _EXTRACTION_META_CACHE[key] = (mean, std)
    return _EXTRACTION_META_CACHE[key]


def _subsample_uniform(t: torch.Tensor, n_frames: int) -> torch.Tensor:
    """C3: n_frames uniformemente espaçados sobre os armazenados (índices
    linspace(0, N−1, n).round() — para 64→16: [0,4,8,13,...,59,63]).
    Reproduz o regime de amostragem do baseline; janela densa é ablação futura."""
    if t.shape[0] == n_frames:
        return t
    idx = torch.from_numpy(
        np.linspace(0, t.shape[0] - 1, n_frames).round().astype(np.int64))
    return t[idx]


def load_frames(frames_dir_path, item_id, n_frames: int = None) -> torch.Tensor:
    """Carrega frames pré-extraídos → [n_frames, 3, 224, 224] float32 NORMALIZADO.

    Contrato de saída IDÊNTICO ao da v1 (float32, normalização ImageNet): tudo a
    jusante (collate, forward, probes de ablação) permanece bit-compatível —
    decisão T3 do plano 11 (desnormalização no Dataset, não no forward).

    v2 (uint8, [64,3,224,224]): subamostra PRIMEIRO (1/4 do custo), depois aplica
    (x/255 − mean)/std com mean/std do `_extraction_meta.json`. Isso é idêntico
    (a ~1 ulp float32) ao rescale+normalize fundido que a v1 gravava:
    (x/255 − m)/s == (x − 255m)/(255s).

    v1 (float32 já normalizado): devolvido como está (para frames_v1_sample/ e
    comparações entre gerações).

    D4: arquivo ausente → FileNotFoundError. O tensor de zeros de antes era uma
    entrada PLAUSÍVEL (zero normalizado = pixel médio do ImageNet), não um buraco.
    """
    if n_frames is None:
        n_frames = N_FRAMES_USED
    path = Path(frames_dir_path) / f"{item_id}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Frames não encontrados para {item_id} em {frames_dir_path} — D4: "
            f"ausência não vira zeros. Ver _missing.json da extração e "
            f"assert_no_missing_features().")
    t = torch.load(path, weights_only=True)
    t = _subsample_uniform(t, n_frames)
    if t.dtype == torch.uint8:
        mean, std = _frames_norm_constants(frames_dir_path)
        return t.float().div_(255.0).sub_(mean).div_(std)
    return t.float()


def load_audio_features(audio_dir_path, item_id) -> torch.Tensor:
    """Features do encoder do Whisper ([1280] float32).

    v2: média mascarada ao áudio real (A1) e janelada além de 30 s (A2) — o
    fator de diluição dependente da duração (ARCH-04; valid_fraction treino
    0,655 vs teste 0,582) deixa de existir por construção.

    D4: arquivo ausente → FileNotFoundError (antes: zeros silenciosos).
    """
    path = Path(audio_dir_path) / f"{item_id}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Features de áudio não encontradas para {item_id} em {audio_dir_path} — "
            f"D4: ausência não vira zeros. Ver _missing.json da extração.")
    return torch.load(path, weights_only=True).float()


def assert_no_missing_features(dataframe, frames_path, audio_path) -> None:
    """C4/D4: falha ALTA no build do dataset se algum ID do split estiver no
    `_missing.json` de qualquer das duas árvores — erro em segundos com a lista
    de IDs, em vez de descobrir depois de horas de GPU.

    Se o manifesto não existir (árvore v1, que não tem manifesto), apenas avisa:
    a falha por item de load_frames/load_audio_features continua cobrindo o caso.
    """
    ids = set(str(i) for i in dataframe["id_EXIST"].values)
    for name, p in (("frames", frames_path), ("audio", audio_path)):
        mpath = Path(p) / "_missing.json"
        if not mpath.exists():
            print(f"[WARN] {mpath} não existe — árvore sem manifesto (v1?); "
                  f"a verificação por item nos loaders continua ativa.")
            continue
        with open(mpath) as f:
            manifest = json.load(f)
        missing_ids = {str(m["id"]) for m in manifest.get("missing", [])}
        hit = sorted(ids & missing_ids)
        if hit:
            raise RuntimeError(
                f"D4: {len(hit)} amostras do split não têm features de {name} "
                f"(extração falhou nesses itens): {hit}\n"
                f"Manifesto: {mpath}. A v1 teve 0 falhas nestes dados — qualquer "
                f"faltante é regressão da extração v2; investigar, não contornar.")
```

### 7.4 C4 — substituir `EXISTVideoDataset.__init__` (seção 8, hoje nas linhas ~615–632)

Substituir o método inteiro por:

```python
    def __init__(self, dataframe, tokenizer, scaler, frames_path, audio_path,
                 max_length: int = MAX_LENGTH, has_labels: bool = True):
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.frames_path = frames_path
        self.audio_path = audio_path
        self.max_length = max_length
        self.has_labels = has_labels

        # C4/D4 (plano 11): se a extração falhou em qualquer ID deste split, o
        # build falha AQUI, em segundos, listando os IDs — nunca no meio do treino.
        assert_no_missing_features(dataframe, frames_path, audio_path)

        self.item_ids = dataframe["id_EXIST"].values
        self.langs = dataframe["lang"].values
        self.texts = dataframe["text"].values
        self.sensorials = dataframe["sensorial"].values

        if has_labels:
            self.labels_a = dataframe["labels_task3_1"].values
            self.labels_b = dataframe["labels_task3_2"].values
            self.labels_c = dataframe["labels_task3_3"].values
```

### 7.5 O que NÃO muda (e por quê)

- `__getitem__`, `video_collate`, `forward`, `NUM_FRAMES` — o contrato do tensor
  `frames` na saída do dataset é idêntico ao v1 (decisão T3).
- O comentário `[16, 3, 224, 224]` no `__getitem__` continua verdadeiro
  (`N_FRAMES_USED = 16`).
- `build_frozen_split` — intocado; o split congelado é sagrado.
- Nada de normalização do vetor concatenado da fusão (regra do projeto).

### 7.6 Checagem estática após editar (pode rodar no mount, é leve)

```bash
python3 -m py_compile EXIST_VIDEOS/video_core.py && echo OK
python3 - <<'EOF'
import sys; sys.path.insert(0, "EXIST_VIDEOS")
import numpy as np, torch
import video_core as vc
# helpers de caminho
assert str(vc.frames_dir("train")).endswith("frames/v2/training")
assert str(vc.frames_dir("test")).endswith("frames/v2/test")
assert str(vc.audio_dir("train")).endswith("audio_features/v2/training")
assert str(vc.frames_dir("train", version="v1")).endswith("frames/train")
assert str(vc.audio_dir("test", version="v1")).endswith("audio_features/test")
# subamostragem 64->16
t = torch.arange(64).view(64, 1, 1, 1).expand(64, 3, 2, 2).contiguous()
s = vc._subsample_uniform(t, 16)
assert s.shape[0] == 16 and s[0, 0, 0, 0] == 0 and s[-1, 0, 0, 0] == 63
assert vc.N_FRAMES_USED == 16 and vc.N_FRAMES_STORED_V2 == 64
print("checagens estáticas de video_core: OK")
EOF
```

(⚠ Se o import de `video_core` falhar localmente por dependência ausente — ex.
`pyevall` não instalado localmente — rodar essa célula na VM. O `py_compile` roda em
qualquer lugar.)

---

## 8. Ordem de execução

Legenda de onde roda: **[VM-Jupyter]** = célula no Jupyter remoto (IO pesado, GPU);
**[mount]** = pode ser feito pelo mount SSHFS (edição de texto, arquivos pequenos).

| # | passo | onde | depende de | tempo estimado |
|---|---|---|---|---|
| P0 | Pré-condições (§3) | [VM-Jupyter] | — | 2 min |
| P1a | Backup `video_core_pre_v2.py.bak` + editar `video_core.py` (§7) + checagem estática (§7.6) | [mount] | — | 30 min |
| P1b | Criar `extract_audio_v2.ipynb` (§6) e `extract_video_v2.ipynb` (§5) | [mount] | — | 30 min |
| P2a | `extract_audio_v2` com `SPLIT="training"` (Restart & Run All) | [VM-Jupyter] | P0, P1b | ~8–11 min `[NÃO VERIFICADO — ver §1]` |
| P2b | idem, `SPLIT="test"` | [VM-Jupyter] | P2a | ~2–3 min |
| P3 | Verificação do áudio: testes 1, 2, 3, 6, 7 (§9) | [VM-Jupyter] | P2b | 5 min |
| P4a | `extract_video_v2` com `SPLIT="training"` — células 0–5 rodam o probe V5-NORM e a sondagem de tempo ANTES do loop | [VM-Jupyter] | P0, P1b | probe 2 min; loop **28 min–1,6 h** `[NÃO VERIFICADO — a célula 5 projeta e aborta se >3 h]` |
| P4b | idem, `SPLIT="test"` | [VM-Jupyter] | P4a | ~1/4 do tempo de P4a |
| P5 | Verificação dos frames: testes 1, 2, 3, 4, 5, 8 (§9) | [VM-Jupyter] | P4b | 10–15 min (teste 8 lê ~31 GB do disco local da VM) |
| P6 | Smoke test do consumidor (§9.9) | [VM-Jupyter] | P1a, P3, P5 | 5 min |
| P7 | *(fora deste plano)* re-treinos na ordem da spec §7 — **caminho 1 primeiro** (só `video_core`, sem features novas), depois frames, depois áudio, para o delta ser atribuível | — | P6 | — |

Notas:
- P2/P3 e P4/P5 são independentes; se a GPU estiver ocupada, P4 (que é CPU) pode rodar
  antes/em paralelo de P2.
- P1a pode ser feito a qualquer momento, mas os defaults v2 de `frames_dir`/`audio_dir`
  fazem qualquer notebook de treino falhar alto (meta ausente) até P2/P4 terminarem —
  é comportamento desejado (D4), não um bug. Não rodar treino entre P1a e P6.
- O passo P4a com `SPLIT="training"` deve rodar até o fim **antes** de P4b, para não
  disputar IO de disco.

---

## 9. Verificação obrigatória (spec §8) — código executável

Criar um notebook `verify_extraction_v2.ipynb` (ou rodar como células avulsas) **na VM**,
com as células abaixo. Todos os testes imprimem `PASS`/`FAIL`; qualquer `FAIL` → §10.

### 9.1 Teste 1 — metas de treino e teste idênticos exceto campos por-split

```python
import json
from pathlib import Path

def load_meta(kind, split):
    root = {"frames": "frames/v2", "audio": "audio_features/v2"}[kind]
    with open(Path(root) / split / "_extraction_meta.json") as f:
        return json.load(f)

IGNORE = {
    "frames": {"split", "n_items", "n_saved", "n_missing", "n_with_fill", "date"},
    "audio":  {"split", "n_items", "n_saved", "n_missing", "window_histogram", "date"},
}
for kind in ["frames", "audio"]:
    a, b = load_meta(kind, "training"), load_meta(kind, "test")
    assert set(a) == set(b), f"{kind}: conjuntos de chaves diferem: {set(a)^set(b)}"
    diff = [k for k in set(a) - IGNORE[kind] if a[k] != b[k]]
    print(f"teste 1 [{kind}]:", "PASS" if not diff else f"FAIL — campos divergem: {diff}")
    # Este teste é a prova ANTI-V6: treino e teste extraídos com a MESMA configuração.
    # Na v1 isso era indemonstrável (o script do treino não existe mais).
```

### 9.2 Teste 2 — contagem de arquivos (2 524 / 674 em cada árvore)

```python
from pathlib import Path
EXPECT = {"training": 2524, "test": 674}
ok = True
for root in ["frames/v2", "audio_features/v2"]:
    for split, n_exp in EXPECT.items():
        n = len(list((Path(root) / split).glob("*.pt")))
        ok &= (n == n_exp)
        print(f"{root}/{split}: {n} (esperado {n_exp})", "ok" if n == n_exp else "FAIL")
print("teste 2:", "PASS" if ok else "FAIL")
```

### 9.3 Teste 3 — manifestos de faltantes vazios

```python
import json
from pathlib import Path
ok = True
for root in ["frames/v2", "audio_features/v2"]:
    for split in ["training", "test"]:
        with open(Path(root) / split / "_missing.json") as f:
            m = json.load(f)["missing"]
        ok &= (len(m) == 0)
        print(f"{root}/{split}: {len(m)} faltantes", "" if not m else f"→ {m[:5]}")
print("teste 3:", "PASS" if ok else
      "FAIL — a v1 teve 0 falhas nestes dados (spec §V2, MEDIDO em frames_zero_scan + "
      "audio_durations): qualquer faltante aqui é REGRESSÃO do script v2, não do dado.")
```

### 9.4 Teste 4 — **equivalência de normalização (V5-NORM) — o teste de maior risco**

Duas camadas (decisão T2, §4.2): **T4a** detecta drift do ambiente desde maio (informativo);
**T4b** é o teste que decide (bloqueante).

```python
import numpy as np, pandas as pd, torch, cv2
from pathlib import Path
from transformers import VideoMAEImageProcessorFast

processor = VideoMAEImageProcessorFast.from_pretrained("MCG-NJU/videomae-base")
SAMPLES = {
    "training": (Path("frames_v1_sample/train"), Path("Dataset/training/videos"),
                 Path("Dataset/training/EXIST2026_training.json")),
    "test":     (Path("frames_v1_sample/test"), Path("Dataset/test/videos"),
                 Path("Dataset/test/EXIST2026_test_clean.json")),
}

def v1_sample_frames(video_path, n=16):
    """Réplica EXATA do sample_frames_uniform da v1 (extract_video.ipynb),
    EXCETO o fallback: se algum read falhar, devolve None e o vídeo sai do teste
    (precisamos de vídeos onde a v1 NÃO acionou o fallback, para que os .pt
    armazenados correspondam a frames reais)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release(); return None
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release(); return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)

t4a_max = t4b_max = 0.0
n_ok = 0
for split, (sdir, vdir, jpath) in SAMPLES.items():
    df = pd.read_json(jpath).transpose()
    name_by_id = {str(r["id_EXIST"]): Path(str(r["path_video"])).name
                  for _, r in df.iterrows()}
    for pt in sorted(sdir.glob("*.pt")):
        iid = pt.stem
        raw = v1_sample_frames(vdir / name_by_id[iid])
        if raw is None:
            print(f"  {split}/{iid}: pulado (a v1 acionou fallback neste vídeo)")
            continue
        frames_list = [raw[i] for i in range(raw.shape[0])]
        # T4a: pipeline v1 recomputado HOJE vs .pt armazenado em maio
        v1_now = processor(images=frames_list,
                           return_tensors="pt")["pixel_values"].squeeze(0)
        stored = torch.load(pt, weights_only=True).float()
        d4a = (v1_now - stored).abs().max().item()
        # T4b: caminho v2 (uint8 + desnormalização do load) vs v1 recomputado,
        # sobre OS MESMOS frames de entrada — isola resize/crop+dtype da amostragem
        rawv2 = processor(images=frames_list, do_rescale=False, do_normalize=False,
                          return_tensors="pt")["pixel_values"].squeeze(0)
        if rawv2.dtype != torch.uint8:
            rawv2 = rawv2.round().clamp(0, 255).to(torch.uint8)
        mean = torch.tensor(processor.image_mean).view(3, 1, 1)
        std = torch.tensor(processor.image_std).view(3, 1, 1)
        recon = rawv2.float().div(255).sub(mean).div(std)   # = video_core.load_frames
        d4b = (recon - v1_now).abs().max().item()
        t4a_max, t4b_max = max(t4a_max, d4a), max(t4b_max, d4b)
        n_ok += 1
        print(f"  {split}/{iid}: T4a={d4a:.2e}  T4b={d4b:.2e}")

print(f"\nvídeos comparados: {n_ok} (mínimo exigido pela spec: 5)")
print(f"T4a (drift de ambiente desde maio): max|Δ| = {t4a_max:.2e} — INFORMATIVO; "
      f"se > 1e-4, registrar que torchvision/transformers mudaram desde a v1")
print(f"T4b (V5-NORM):                      max|Δ| = {t4b_max:.2e}")
print("teste 4:", "PASS" if (n_ok >= 5 and t4b_max <= 0.017) else
      "FAIL — NÃO usar os frames v2; ver plano 11 §10")
# Esperado se a VM se comportar como a classe base inspecionada localmente
# (plano 11 §4.2): T4b ~1e-6 (caminho uint8) ou ≤9e-3 (fallback com cast).
```

### 9.5 Testes 5 + 8 — dtype/shape e varredura de zeros na v2 (uma passada só)

```python
# Lê ~31 GB do disco LOCAL da VM (nunca pelo mount). ~5–15 min.
import torch
from pathlib import Path
EXPECT_SHAPE = (64, 3, 224, 224)
all_ok = True
for split in ["training", "test"]:
    d = Path("frames/v2") / split
    bad_zero, bad_meta = [], []
    for p in sorted(d.glob("*.pt")):
        t = torch.load(p, weights_only=True)
        if t.dtype != torch.uint8 or tuple(t.shape) != EXPECT_SHAPE:
            bad_meta.append((p.name, str(t.dtype), tuple(t.shape)))
        elif not t.any():
            bad_zero.append(p.name)
    print(f"{split}: dtype/shape ruins={len(bad_meta)} todo-zeros={len(bad_zero)}")
    if bad_meta[:3]: print("   ex.:", bad_meta[:3])
    if bad_zero[:3]: print("   ex.:", bad_zero[:3])
    all_ok &= not bad_meta and not bad_zero
print("testes 5+8:", "PASS" if all_ok else "FAIL")
# Referência v1: 0/3198 todo-zeros (analysis/frames_zero_scan.json). Na v2 um
# todo-zeros é ainda mais impossível por construção (falha não grava arquivo);
# encontrar um seria bug novo no processor ou no disco.

# Áudio: mesma varredura, barata (26 MB)
for split in ["training", "test"]:
    d = Path("audio_features/v2") / split
    bad = [p.name for p in d.glob("*.pt")
           if not torch.load(p, weights_only=True).any()]
    print(f"áudio {split}: todo-zeros={len(bad)}", "ok" if not bad else "FAIL")
```

### 9.6 Teste 6 — **norma do áudio v1 vs v2 — o detector do modo de falha silencioso (V7)**

A referência v1 está PRESERVADA em `audio_features/{train,test}` de propósito para este
teste. Norma média v1 (treino) = **14,96 ± 0,50** `[MEDIDO, spec §6]`.

```python
import csv, torch
from pathlib import Path
import numpy as np

dur = {r["id_EXIST"]: float(r["duration_s"])
       for r in csv.DictReader(open("analysis/audio_durations.csv"))
       if r["split"] == "training"}

v1d, v2d = Path("audio_features/train"), Path("audio_features/v2/training")
norms1, norms2, deltas = [], [], []
n_ident = 0
n_short = n_short_diff = 0
for p1 in sorted(v1d.glob("*.pt")):
    iid = p1.stem
    a = torch.load(p1, weights_only=True).float()
    b = torch.load(v2d / f"{iid}.pt", weights_only=True).float()
    norms1.append(a.norm().item()); norms2.append(b.norm().item())
    d = (a - b).abs().max().item(); deltas.append(d)
    if d < 1e-6:
        n_ident += 1
    if dur.get(iid, 99) < 25:          # itens claramente diluídos na v1
        n_short += 1
        if d > 1e-3:
            n_short_diff += 1

m1, m2 = float(np.mean(norms1)), float(np.mean(norms2))
print(f"norma média v1 = {m1:.2f} (referência MEDIDA: 14,96 ± 0,50)")
print(f"norma média v2 = {m2:.2f}  [informativo — direção NÃO prevista; reportar]")
print(f"itens bit-idênticos v1==v2: {n_ident} "
      f"(itens com duração ≈30,0 s podem coincidir legitimamente; muitos = V7)")
print(f"itens <25 s com diferença material: {n_short_diff}/{n_short}")
ok = (abs(m1 - 14.96) <= 0.50           # sanity: estamos lendo a v1 certa
      and n_short_diff / max(1, n_short) >= 0.99)
print("teste 6:", "PASS" if ok else
      "FAIL — se v2≈v1 item a item, a correção NÃO pegou: o script pulou tudo "
      "(V7: árvore de saída errada ou resume sobre árvore cheia). "
      "Apagar audio_features/v2/ e repetir P2 verificando OUTPUT_DIR.")
```

### 9.7 Teste 7 — itens > 30 s passaram por > 1 janela

```python
import json, math, csv
ok = True
for split, csv_split in [("training", "training"), ("test", "test")]:
    with open(f"audio_features/v2/{split}/_extraction_meta.json") as f:
        hist = {int(k): v for k, v in json.load(f)["window_histogram"].items()}
    multi = sum(v for k, v in hist.items() if k > 1)
    durs = [float(r["duration_s"]) for r in csv.DictReader(open("analysis/audio_durations.csv"))
            if r["split"] == csv_split]
    expected = sum(1 for d in durs if d > 30.0)   # training: 842 | test: 162 [MEDIDO, CSV]
    print(f"{split}: hist={dict(sorted(hist.items()))} | multi-janela={multi} "
          f"(esperado ~{expected} ±5, pelas durações ffprobe)")
    ok &= abs(multi - expected) <= 5
print("teste 7:", "PASS" if ok else "FAIL")
# ⚠ Se a run foi RETOMADA após queda, o histograma do meta cobre só a última run
# (plano 11 §6, observação final). Nesse caso este teste é inconclusivo pelo meta;
# re-rodar a extração do split numa árvore limpa OU aceitar o teste 6 como evidência.
```

### 9.8 Teste 4-complementar (informativo, não bloqueante) — estatísticas por canal v1 vs v2

A spec sugere comparar estatísticas por canal quando os índices de amostragem não
coincidem (é o caso: v1 usava contagem estimada, v2 usa real). Frames diferentes do
mesmo vídeo têm estatísticas parecidas mas não iguais — por isso é informativo.

```python
import torch
from pathlib import Path
import video_core as vc   # já editado (P1a)
for split, sdir in [("training", "frames_v1_sample/train"),
                    ("test", "frames_v1_sample/test")]:
    for pt in sorted(Path(sdir).glob("*.pt"))[:5]:
        v1 = torch.load(pt, weights_only=True).float()               # [16,...] normalizado
        v2 = vc.load_frames(vc.frames_dir(split), pt.stem)           # [16,...] desnormalizado do uint8
        m1, m2 = v1.mean(dim=(0, 2, 3)), v2.mean(dim=(0, 2, 3))
        print(f"{split}/{pt.stem}: média/canal v1={m1.tolist()} v2={m2.tolist()} "
              f"|Δ|={(m1-m2).abs().max():.3f}  (informativo; diferenças ~0,1 são "
              f"esperadas por serem FRAMES diferentes do mesmo vídeo)")
```

### 9.9 Smoke test do consumidor (P6) — o pipeline inteiro com as árvores v2

```python
import sys; sys.path.insert(0, ".")
import pandas as pd
import torch
import video_core as vc
from sklearn.preprocessing import RobustScaler
from transformers import XLMRobertaTokenizer

full_df = pd.read_json("Dataset/training/EXIST2026_training.json").transpose()
train_df, val_df = vc.build_frozen_split(full_df)      # carrega o artefato congelado
# scaler ajustado SÓ no treino (anti-leakage) — mesmo procedimento do notebook de treino
scaler = RobustScaler().fit(vc.collect_bio_vectors(train_df))
tokenizer = XLMRobertaTokenizer.from_pretrained(vc.XLMR_NAME)
ds = vc.EXISTVideoDataset(train_df, tokenizer, scaler,
                          vc.frames_dir("train"), vc.audio_dir("train"))
item = ds[0]
assert item["frames"].shape == (16, 3, 224, 224) and item["frames"].dtype == torch.float32
assert item["audio"].shape == (1280,)
# a faixa da v1 medida era [−2,10; +2,64] (OPT-01) — a v2 tem de cair na mesma:
lo, hi = item["frames"].min().item(), item["frames"].max().item()
assert -2.5 < lo < 0 and 0 < hi < 3.0, (lo, hi)
print("smoke test do consumidor: PASS", (lo, hi))
```

(O `scaler` deve ser construído exatamente como no notebook de treino; se der trabalho,
substituir por um `RobustScaler` ajustado em `vc.collect_bio_vectors(train_df)` — o
smoke test não mede números, só o contrato dos tensores.)

---

## 10. Modos de falha e o que fazer

| sintoma | causa provável | ação |
|---|---|---|
| **Probe V5-NORM (célula 3 do notebook de frames) reprova (`maxΔ > 0,017`)** | o processor da VM (transformers 5.6.0) muda o resize/crop quando rescale/normalize são desligados, OU ignora os kwargs de chamada | **PARAR. Não extrair.** Diagnóstico: imprimir `raw.shape`, `raw.dtype`, `full.shape` e `processor.to_dict()`; testar passar os flags no construtor (`VideoMAEImageProcessorFast.from_pretrained(MODEL_NAME, do_rescale=False, do_normalize=False)`) em vez de na chamada. Se nada fechar ≤0,017, reportar ao autor com os números — a alternativa (gravar fp16 normalizado, 2× em vez de 4×, spec §V5 parêntese) é decisão do autor, não do executor |
| Probe passa mas com `NEEDS_CAST=True` (saída float no modo v2) | a versão da VM faz cast interno para float no resize | Seguir (o fallback quantiza com erro ≤0,0087, dentro da tolerância). O meta grava `needs_cast_fallback: true` — citar na tese se relevante |
| **Teste 4 (T4b) reprova pós-extração** | o probe sintético passou mas vídeos reais divergem (ex.: resolução ímpar, caminho de interpolação diferente) | **Descartar `frames/v2/` inteira** (não misturar gerações), reportar com os `maxΔ` por vídeo. Não tentar "compensar" no load |
| Teste 4 (T4a) mostra drift (>1e-4) mas T4b passa | torchvision/transformers mudaram desde maio — o v1 armazenado não é mais reproduzível bit a bit | Não bloqueia (T4b é o que decide). REGISTRAR no plano/tese: "o ambiente de extração mudou entre v1 e v2; a equivalência foi estabelecida sobre o pipeline recomputado" |
| Sondagem de tempo (célula 5) projeta > 3 h | decodificação 2× mais cara do que o estimado | PARAR e reportar. Opções (decisão do autor): aceitar o custo; implementar a opção (b) do V4-MEM com o resize do processor aplicado por frame; paralelizar por processo |
| `_missing.json` não-vazio (teste 3) | regressão do script v2 — a v1 teve 0 falhas nestes dados `[MEDIDO]` | Investigar item a item (a razão está no manifesto). NÃO seguir para treino com faltantes: o dataset vai recusar (C4), por design |
| >5 % dos vídeos com preenchimento (`_extraction_warnings.json`) | truncamento mais extenso do que o 1–4 % medido no split de teste, ou bug no passe 2 | PARAR. Comparar `total_frames` do passe 1 com a duração do CSV (`total/fps ≈ duração?`) para 5 casos; se o passe 1 conta bem e só o retrieve falha no fim, é o padrão de arquivo truncado conhecido — decidir com o autor se o preenchimento por vizinho é aceitável acima de 5 % |
| **Teste 6 reprova: v2 ≈ v1 item a item** | V7 — o script gravou na árvore errada ou fez resume sobre árvore cheia; OU o notebook rodou a célula 4 com a função v1 em memória (kernel sujo) | Apagar `audio_features/v2/{split}/` inteira, **Restart & Run All** (nunca rodar células avulsas), conferir o `OUTPUT_DIR` impresso na célula 1 |
| Teste 6: norma média v1 fora de 14,96±0,50 | a referência v1 lida não é a esperada (diretório errado) | Conferir que se está lendo `audio_features/train` (v1), não a v2 |
| Teste 7 fora de ±5 | contagem de janelas divergente das durações ffprobe | Se a run foi retomada, o histograma é parcial (inconclusivo — ver nota do teste). Senão, comparar 3 itens à mão: `duração/30 → ceil` vs janelas |
| Extração morre no meio por disco cheio | margem de 16 GB consumida por outra coisa | `df -h`; liberar espaço; re-rodar o MESMO notebook — o resume (V7) retoma de onde parou corretamente |
| `RuntimeError: _extraction_meta.json não encontrado` ao treinar | treino iniciado entre P1a e P2/P4, ou extração de um dos splits esquecida | Rodar as extrações que faltam. É o D4 funcionando |
| `FileNotFoundError` de frames/áudio num item durante o treino | árvore v2 incompleta apesar dos testes (ex.: arquivo apagado depois) | Recontar (teste 2); re-rodar a extração do split (resume preenche só o que falta) |
| OOM na extração de frames | vídeo fora do perfil (ex.: 4K longo) — 64 frames nativos > RAM | Identificar o vídeo (o traceback aponta), registrar ID; se for um caso isolado, reportar ao autor antes de qualquer tratamento especial |

**Regra geral:** nenhuma falha destas é contornada "gravando alguma coisa mesmo assim".
O desenho inteiro do D4 é que dado ausente = erro visível, nunca dado fabricado.

---

## 11. Rollback — o que é reversível e o que não é

### Reversível

| artefato | como reverter |
|---|---|
| `video_core.py` | `cp video_core_pre_v2.py.bak video_core.py` (o arquivo NÃO está no git — o backup do P1a é o único caminho; se o backup não existir, NÃO editar) |
| `extract_video_v2.ipynb` / `extract_audio_v2.ipynb` | apagar; os originais v1 nunca foram tocados (D3) |
| `frames/v2/`, `audio_features/v2/` | apagar as árvores (liberam ~31 GB); a extração é re-executável a qualquer momento a partir de `Dataset/*/videos/` |
| default `FEATURES_VERSION="v2"` | mudar a constante para `"v1"` devolve os helpers de caminho ao comportamento antigo (⚠ mas frames v1 não existem mais — só faz sentido para o áudio) |

### Irreversível (já perdido ANTES deste plano — declarar, não esconder)

- **`frames/` v1 foi apagado em 2026-08-08** (spec §6). Não é regenerável: o script que
  o produziu foi editado à mão e não existe no repositório nem no git (o próprio
  problema V6). Restam 25 tensores em `frames_v1_sample/`.
- **`runs_baseline/seed_42/` deixou de ser um experimento repetível.** Os números
  continuam congelados e válidos (`final_metrics.json`, `epoch_metrics.csv`,
  fingerprints `8a60e7121b63fbb2`/`9bfb83bcc6efd303`), mas ninguém consegue re-rodar a
  run. Formulação para a tese (da spec §6): *"o baseline continua válido como número;
  deixou de ser válido como experimento repetível"*.
- `inference_video.py` sobre a v1 não funciona mais (já estava dessincronizado —
  `05_IMPLEMENTATION_NOTES.md` §6, item 6).

### O que NUNCA entra em rollback nem limpeza

`frames_v1_sample/` e `audio_features/{train,test}` (v1) só podem ser considerados para
remoção **depois** de a §9 passar inteira E de o autor decidir explicitamente — e mesmo
então, recomenda-se manter (230 MB + 26 MB é barato; são as únicas referências físicas
da geração v1).

---

## 12. Contradições encontradas (spec 10 × relatórios anteriores × este plano)

Regra aplicada: **a spec (doc 10) ganha** dos relatórios 01–03; onde este plano refina a
spec, está dito explicitamente.

1. **ARCH-09 (doc 01) propunha `while True: frames.append(...)`** — acumula o vídeo
   inteiro em resolução nativa: até 25,2 GB no pior item `[MEDIDO na spec §V4-MEM]`.
   A spec já marca a proposta original como errada; este plano implementa a opção (a)
   com `grab()`/`retrieve()` (§4.1).
2. **OPT-12 (doc 03) estimou a árvore v2 em ~7,7 GB** — era sob 16 frames uint8. A
   decisão D1 (64 frames) leva a ~31 GB. A spec §V7/§6 já usa o número correto.
3. **Pseudocódigo do A1 na spec (§A1): `features = encoder_out[:n_valid].mean(dim=0)`**
   fatiaria a dimensão de BATCH de `[1, 1500, 1280]` (o `squeeze(0)` da v1 sumiu no
   snippet). O índice correto, usado neste plano, é `encoder_out[0, :n_valid]`.
   É erro de pseudocódigo, não de conceito.
4. **Docstring atual de `frames_dir` (`video_core.py`)** diz `split ∈ {"train","test"}`;
   a v2 usa os nomes do Dataset (`training`/`test`). Resolvido com o mapa de aliases
   `_V2_SPLIT_DIRNAME` (§7.2) — as chamadas existentes com `"train"` continuam válidas.
5. **Spec §V5-NORM diz "tipicamente `do_rescale=False, do_normalize=False`"** sem
   afirmar que a versão da VM aceita — a inspeção local (classe base do transformers
   4.57.6) confirma que são kwargs de chamada e que o caminho uint8 é preservado
   (§4.2), mas a classe concreta `VideoMAEImageProcessorFast` **nem existe** no 4.57.6
   local; a confirmação final é o probe da célula 3 na VM `[VM]`.
6. **`09_BASELINE_RESULTS.md` (regra de simetria)** não é contradito por nada aqui, mas
   ganha uma dependência nova: o primeiro re-treino comparável ao baseline TEM de usar
   `N_FRAMES_USED=16` com subamostragem uniformemente espaçada (C3) — 32/64 e janela
   densa mudam o objeto medido e são braços de ablação.

---

## 13. Questões para o autor (nenhuma bloqueia P0–P6, salvo indicação)

> ---
> ## ✅ RESPONDIDAS PELO AUTOR — 2026-08-09
>
> **Todas as cinco estão decididas. Executor: siga estas respostas, não reabra.**
>
> **Q1 — política de preenchimento parcial: MEIO-TERMO ACEITO.** Preencher com o frame
> real mais próximo já decodificado, contabilizar em `_extraction_warnings.json`, gate de
> 5 %. Razão decisiva: a alternativa estrita descartaria 1–4 % do split de **teste**, e
> são itens para os quais a submissão ao EXIST 2026 é **obrigatória** — não há a opção de
> não predizer. Manter o gate: se >5 % dos vídeos precisarem de preenchimento, **parar** e
> reportar, porque aí deixou de ser cauda e virou sintoma.
>
> **Q2 — arredondamento: `.round()` CONFIRMADO.** A fidelidade exigida é ao *regime*
> (amostragem uniforme sobre o vídeo), não ao truncamento acidental da v1. O
> deslocamento ≤1 frame por posição é irrelevante perto do gap mediano de 1,21 s entre
> frames consecutivos. Registrar a escolha no `_extraction_meta.json`.
>
> **Q3 — versão do `openai-whisper`:** gravar no meta, como planeado. A adição à tabela do
> `CLAUDE.md` fica com o orquestrador (fora do escopo deste plano). É o problema V6 outra
> vez, num pacote que ninguém tinha olhado — bom apanhado.
>
> **Q4 — nota metodológica do T4a: APROVADA** conforme redigida na §10. Se o ambiente
> tiver mudado desde maio (T4a > 1e-4), a comparação baseline↔v2 carrega essa nota na
> tese. Não muda o critério de decisão, que continua sendo o T4b.
>
> **Q5 — sequenciamento: TRÊS BRAÇOS, não dois.** A alternativa (a) está morta, como você
> mesmo notou. Mas a (b) subestimava o que ainda é separável: **`build_model` já aceita
> `restore_biases` (`video_core.py:1139`)**, então o ARCH-01 isola-se pelo lado do
> **modelo**, sem extração e sem disco. Braços:
>
> | braço | frames | áudio | `restore_biases` | delta isola |
> |---|---|---|---|---|
> | baseline (congelado) | v1 | v1 | — (código antigo) | referência |
> | **A′** | v2 | v1 | **False** | frames + código (bloco único) |
> | **A** | v2 | v1 | True | **ARCH-01** = A − A′ |
> | **B** | v2 | **v2** | True | **áudio** = B − A |
>
> `B − baseline` é o ganho total; os três deltas o decompõem.
>
> **Protocolo de épocas (obrigatório — regra de simetria, `09_BASELINE_RESULTS.md` §2):**
> uma calibração de 15 épocas na configuração final (B) → congelar
> `E = argmin(eval_loss)` → rodar **os três braços com esse mesmo E**. Comparar sempre
> estado final contra estado final, nunca fim de um lado e pico do outro.
> Custo: ~92 min de calibração + ~40 min por braço ≈ **3,5 h**.
>
> ⚠ **A única separação genuinamente perdida** é entre correções de *extração de frames* e
> demais correções de *código* — as duas entram juntas em A′. Exigiria frames v1, que não
> existem nem são regeneráveis (V6). Reportar A′ como bloco único, sem proporção interna.
>
> A spec (`10_EXTRACTION_FIXES_SPEC.md` §7) foi atualizada com esta mesma tabela.
>
> ---

1. **Política de preenchimento no passe 2 (§4.1/§5 célula 4).** A spec diz "nada é
   inventado: se não há frames, devolve None". Este plano adota um meio-termo para o
   caso *parcial* (retrieve falha em ALGUNS índices de um vídeo majoritariamente
   legível — o padrão de arquivo truncado que atinge 1–4 % do teste): preencher com o
   **frame real mais próximo já decodificado**, contabilizado em
   `_extraction_warnings.json`, com gate de 5 %. A alternativa estrita (qualquer
   retrieve falho → `_missing.json`) descartaria ~1–4 % do split de teste — itens para
   os quais será obrigatório submeter predição. Confirmar se o meio-termo está aceito.
   *(Se não: trocar, na função `sample_frames_v2`, o bloco de preenchimento por
   `return None, {"reason": f"retrieve falhou em {n} índices"}`.)*
2. **Arredondamento dos índices de amostragem.** A v1 usava `linspace(..., dtype=int)`
   (truncamento); a v2 usa `.round()` (mais uniforme). A diferença é ≤1 frame de
   deslocamento por posição. Assumi que a fidelidade exigida é ao *regime* (uniforme
   sobre o vídeo), não ao truncamento acidental da v1. Confirmar.
3. **Versão do `openai-whisper` na VM** não está registrada em nenhum documento
   (a tabela do `CLAUDE.md` não a inclui). O P0 e o meta do áudio passam a registrá-la;
   vale adicionar à tabela do `CLAUDE.md` na próxima edição dele (fora do escopo deste
   plano — este plano não edita o `CLAUDE.md`).
4. **T4a com drift**: se o ambiente mudou desde maio (T4a > 1e-4), a comparação
   baseline↔v2 ganha uma nota metodológica na tese. Redigi a formulação na §10; validar.
5. **Sequenciamento dos re-treinos (P7)**: a spec §7 manda rodar o caminho 1 (só
   `video_core`, sem features novas) primeiro. Com C1 defaultando para v2, o caminho 1
   **não é mais executável como descrito** depois deste plano — frames v1 não existem.
   Alternativas: (a) rodar o caminho 1 ANTES de P1a (com `video_core` pré-v2 + árvores
   v1 de áudio... mas frames v1 já não existem — logo o caminho 1 puro já era
   inviável desde o apagamento); (b) aceitar que a atribuição passa a ser
   {v2-frames+core} → +áudio, em dois passos em vez de três. **Isto precisa de decisão
   do autor antes de P7** — registrado aqui porque a spec §7 foi escrita antes de
   `frames/` v1 ser apagado.

---

## 14. Checklist final do executor

Marcar na ordem. Não pular; não marcar sem o output correspondente salvo no notebook.

- [ ] P0: célula de pré-condições rodada na VM — TODAS `ok`; versão do `openai-whisper` anotada
- [ ] P1a: `video_core_pre_v2.py.bak` criado ANTES de editar
- [ ] P1a: `video_core.py` editado (§7.1–7.4) — `py_compile` OK + checagens §7.6 OK
- [ ] P1b: `extract_audio_v2.ipynb` criado com as 6 células da §6
- [ ] P1b: `extract_video_v2.ipynb` criado com as 8 células da §5
- [ ] P2a: áudio `training` — Restart & Run All completo; célula 3 (auto-teste) PASS; `_missing.json` vazio
- [ ] P2b: áudio `test` — idem
- [ ] P3: testes 1, 2, 3 (parte áudio), 6 e 7 — PASS; norma média v2 registrada (valor: ______)
- [ ] P4a: frames `training` — probe V5-NORM (célula 3) PASS (`NEEDS_CAST` = ______; pior `max|Δ|` = ______); sondagem de tempo ≤ 3 h (projeção: ______ min); loop completo
- [ ] P4b: frames `test` — idem
- [ ] P5: testes 1, 2, 3 (parte frames), 4 (T4a = ______; T4b = ______), 5+8 — PASS
- [ ] P6: smoke test do consumidor (§9.9) — PASS, faixa dos frames registrada
- [ ] `_extraction_warnings.json`: fração com preenchimento ≤ 5 % (valor: ______)
- [ ] NADA foi apagado de: `frames_v1_sample/`, `audio_features/{train,test}`, `runs_baseline/`, originais v1
- [ ] Questão 5 da §13 (sequenciamento P7) levada ao autor ANTES de qualquer re-treino

---

*Plano escrito em 2026-08-08 a partir de `10_EXTRACTION_FIXES_SPEC.md` (espec primária),
com inspeção direta de `extract_video.ipynb`, `extract_audio.ipynb`, `video_core.py`,
`09_BASELINE_RESULTS.md`, `01/02/03_*_FINDINGS.md` e `analysis/audio_durations.csv`.
Números sem fonte explícita não existem neste documento; o que não pôde ser verificado
localmente está marcado `[NÃO VERIFICADO]` ou `[VM]` com o comando de confirmação.*
