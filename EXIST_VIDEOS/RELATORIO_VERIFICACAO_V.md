# Relatório — verificação do sistema de vídeo (Task 3), para a Seção 3.4 da tese

**Data:** 2026-10-08 · **Itens:** V1–V11
**Regra seguida:** o texto da tese acompanha o **código**, e não o paper. Nada foi treinado. As únicas medições foram feitas em CPU: instanciar os modelos na VM sem pesos treinados, contar parâmetros e ler JSONs, checkpoints e logs.

**Níveis de evidência usados em cada linha:**
- **[código]**: verificado no código.
- **[log]**: verificado em log ou artefato (saída de célula, JSON, md5, `.bash_history`).
- **[medido]**: medido nesta sessão (CPU, na VM).
- **[inferido]**: dedução a partir de artefatos, sem prova direta.
- **[paper]**: só o paper sustenta.
- **[n/v]**: não verificável.

**Convenções:** as células são numeradas a partir de 0, na ordem do `.ipynb`. "Paper" é o PDF final (`~/Downloads/jow_at_exist_2026_…soft_label_learning.pdf`, 05/07), §5 (Task 3).

---

## 0. Fontes e mapeamento dos runs (resolvido nesta verificação)

| Artefato | O que é | Evidência |
|---|---|---|
| `~/.bash_history` da VM, linhas 253–268 | **Os comandos de inferência dos 3 runs.** Jow_1 usou `--model-dir ./best_model_seed_2025` e saída em `./predictions/`. Jow_2 usou `./best_model_seed_2025_swa` e saída em `./predictions/swa/`. Jow_3 usou `./best_model_seed_2025_cross_att` e saída em `./predictions/cross_att/`. Os três usaram `--thresholds-json ./task_c_thresholds_seed_2025.json` | [log] |
| `~/.bash_history`, linha 285 | `rm -rf best_model_seed_2025 best_model_seed_2025_cross_att/ best_model_seed_2025_swa/`. **Os pesos dos 3 runs submetidos não existem mais.** Na linha 287, `checkpoints_seed_2025/` também foi apagado | [log] |
| `EXIST_VIDEOS/predictions/cross_att/` | Predições do **Jow_3** (VM: 05-14 19:32:43) | md5 dos 6 JSON idênticos aos de `task3_*_Jow_3.json` do pacote final [log] |
| `EXIST_VIDEOS/predictions/swa/` | Predições do **Jow_2** (VM: 05-14 17:15:45) | md5 dos 6 JSON idênticos aos de `task3_*_Jow_2.json` [log] |
| `~/Downloads/exist2026_Jow (1).zip` | **Pacote final submetido.** Horários no zip (hora do Mac): Jow_1 às 15:19, Jow_2 às 19:36, Jow_3 às 19:37 | As predições do Jow_1 **só existem no zip**. A pasta `predictions/` não tem mais os arquivos da raiz [log] |
| `EXIST_VIDEOS/.ipynb_checkpoints/Thesis_video_fixed-checkpoint.ipynb` (05-14 18:02) | **Código exato que treinou o Jow_3**, salvo antes do treino. Tem saídas de uma tentativa abortada às 17:26–18:00 | [log] |
| `EXIST_VIDEOS/Thesis_video_fixed.ipynb` | **Log de treino do Jow_3**: célula 52, de 05-14 18:25 a 19:16. O código em disco = checkpoint das 18:02 **mais** o andaime de determinismo adicionado em 2026-08-04 (`CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`, re-seeds antes de cada init, prints de fingerprint). Essas linhas novas não têm saída. **O arquivo carrega saídas de 05-14 sobre código de 08-04.** A arquitetura e os hiperparâmetros são idênticos; a reprodutibilidade bit a bit, não | [log] (diff do checkpoint × arquivo atual) |
| `EXIST_VIDEOS/inference_video.py` | Inferência do **Jow_3** apenas. É idêntico (md5 `3b58d921…`) a `.ipynb_checkpoints/inference_video-checkpoint.py` (05-14 19:31). **Não pode ter produzido o Jow_1 nem o Jow_2**: a classe tem `cross_attn`, e o `load_state_dict` estrito falharia com aqueles pesos (ver célula 54 abaixo). A versão anterior do script se perdeu | [código]+[log] |
| `Thesis_video_fixed.ipynb`, célula 54 | **Prova de arquitetura do Jow_1.** Às 19:16 a célula tentou carregar `./best_model_seed_2025/model.safetensors` (o diretório do Jow_1) num `EXISTVideoModel` com `cross_attn` e falhou com `Missing key(s): "cross_attn.*", "cross_proj.*", "cross_norm.*"`. Não houve chave inesperada nem *size mismatch*. Logo, o Jow_1 tinha **exatamente a arquitetura do Jow_3 sem a cross-attention**: `video_pool`, `audio_proj` com LayerNorm, `viewer_emb`/`viewer_proj`, `bio_ln`, `bio_mlp` 17→128→128, fusão 2688 | [log]. Ressalva: o diretório pode ter sido sobrescrito entre a inferência do Jow_1 e as 19:16. Não há `mv`/`cp` de modelo no histórico, mas o navegador de arquivos do Jupyter não deixa rastro |
| `EXIST_VIDEOS/task_c_thresholds_seed_2025.json` (05-14 19:19) | Limiares do **Jow_3** apenas. Os do Jow_1 e do Jow_2 eram versões anteriores do mesmo arquivo, sobrescritas | Reconstrução (V10) [log] |
| `EXIST_VIDEOS/Thesis_video.ipynb` | Arquitetura **igual à descrita no paper** (2592, bio 17→64→32, *mean pooling*, `audio_proj` linear, *modality dropout* 0,15, SWA a partir da 8). **Não pode ser ligado a nenhum run submetido.** Não tem célula de salvar modelo, scaler ou limiares. As saídas em disco são de um re-run de **2026-06-03** (pós-submissão, SEED=42, interrompido). O checkpoint de 05-13 (14:41–15:03) também foi interrompido | [log] |

**Níveis de evidência por run** (mesmo critério do relatório dos memes):
- **Jow_3: verificado.** Há código (checkpoint das 18:02), log de treino (célula 52), predições e limiares. Os pesos foram apagados.
- **Jow_1: parcialmente verificado.** A arquitetura vem do artefato da célula 54 e os limiares foram reconstruídos. O código de treino, o log, o SWA e as épocas se perderam.
- **Jow_2: só as predições e o nome do diretório (`_swa`).** Código, log e pesos se perderam. "SWA" e "sem cross-attention" são [inferido]/[paper].

---

## V1. Mapeamento das runs submetidas da Task 3

### V1.1 Artefatos ↔ Jow_1/2/3

| Run | Predições locais | md5 = pacote final | Diretório do modelo (`.bash_history`) |
|---|---|---|---|
| Jow_1 | **nenhuma** (só no zip) | — | `best_model_seed_2025` |
| Jow_2 | `predictions/swa/` | **sim, 6/6** | `best_model_seed_2025_swa` |
| Jow_3 | `predictions/cross_att/` | **sim, 6/6** | `best_model_seed_2025_cross_att` |

`inference_video.py` grava sempre `task3_*_{team}_1.json` (linhas 744–761). Os sufixos `_2`/`_3` foram renomeados à mão.

### V1.2 Configuração de cada run

| | Jow_1 | Jow_2 | Jow_3 |
|---|---|---|---|
| `cross_attn` | **não** [log, célula 54] | não [inferido: o treino com `cross_attn` começou às 17:26, depois das predições do Jow_2 às 17:15] | **sim** [código+log] |
| SWA | [n/v] (paper: não) | sim [inferido pelo nome `_swa`; paper] | **sim, a partir da época 5**: `SWACallback(swa_start_epoch=5)` na célula 50, condição `state.epoch + 1e-6 >= 5`. Log: `Averaged over 11 epochs` (épocas 5–15) [log] |
| Épocas | [n/v] | [paper: 15] | **15** (célula 33; log `global_step=1065` = 71 × 15) [log] |
| Estado final | [n/v] | [n/v] | **média SWA** das épocas 5–15, com `load_best_model_at_end=False` [log] |
| SEED / split | 2025 [log: diretório e `val_seed_2025.json` no `--eval`] | 2025 [log] | 2025 [log] |

- **Paper × código:** o paper diz que o SWA do Jow_2 média as épocas **8–14 (7 snapshots)**. Nenhum código sobrevivente produz isso. O `Thesis_video.ipynb` começa na 8 com `int(state.epoch) >= 8`, o que daria 8 snapshots (8–15). O código do Jow_3 começa na 5 (11 snapshots). O paper também não diz que o **Jow_3 tem SWA**, e ele tem.
- O doc 09 ("SWA a partir da 5") descreve corretamente o código do Jow_3.

### V1.3 O baseline congelado do doc 09

`Thesis_video_baseline.ipynb` é o **código do Jow_3** (`cross_attn` + SWA a partir da 5 + `bio_ln` + `max_length=128` + 2688) [código, diff contra `Thesis_video_fixed.ipynb`]. Muda só o instrumento:
- split congelado `val_split_42.json` (2019/505) no lugar de 2271/253;
- `SEED=42`;
- flags de determinismo e fingerprints;
- curva por época gravada.

**Não corresponde a nenhum run submetido em pesos.** É a reprodução da *arquitetura* do Jow_3 em outro split e outra seed. Tempo de treino: 5524 s (`runs_baseline/seed_42/train_output.json`).

### V1.4 O que sobrevive por run

| | Código de treino | Log de treino | Pesos | Predições | Limiares |
|---|---|---|---|---|---|
| Jow_1 | não (arquitetura via célula 54) | não | **apagados** | só no zip | reconstruídos (V10) |
| Jow_2 | não | não | **apagados** | sim | reconstruídos (V10) |
| Jow_3 | sim (checkpoint das 18:02) | sim (célula 52) | **apagados** | sim | sim (JSON + reconstrução) |

---

## V2. Demografia dos anotadores

### V2.1 A versão submetida injeta demografia?

**Não, em nenhum código sobrevivente.** Não há uso de `annotators`, `gender_annotators` ou de qualquer vetor demográfico em `Thesis_video_fixed.ipynb`, no checkpoint, em `inference_video.py` nem no `Thesis_video.ipynb` [código, grep]. Para o Jow_1, o conjunto de chaves da célula 54 não tem nenhum módulo demográfico [log]. Para o Jow_2: [n/v].

**Além disso, o dataset da Task 3 não permite o mecanismo da Task 2.** Os únicos campos de anotador no JSON são `annotators`, `number_annotators` e `gender_annotators`. Não existem idade, etnia, escolaridade nem país [log, `EXIST2026_training.json` e `EXIST2026_test_clean.json`].

**Paper × código:** a §5.1 diz *"Annotator demographics follow the same mechanism as Task 2"*, a §5.4 lista *"annotator demographics"* no Jow_1 e a legenda da Fig. 2 diz *"annotator demographics are injected additively"*. **Falso para o Jow_1 e o Jow_3.**

O markdown da célula 0 do notebook fala em *"viewer demographics"*. Isso se refere ao `viewer_emb`, que é a **identidade** do participante de laboratório e não demografia (V3.5).

### V2.2 A versão corrigida

**Confirmado: não usa demografia** [código]. Os 2688 da fusão são 768 (vídeo) + 768 (áudio) + 1024 (texto) + **128 (biossinais dos participantes de laboratório)** (`video_core.py:970-987`). O slot de 128 não é de anotadores.

### V2.3 Anotadores na Task 3 [log]

| | Treino (2524) | Teste (674) |
|---|---|---|
| Anotadores distintos | **10** (`Annotator_1`…`Annotator_10`) | os mesmos 10 |
| Votos por vídeo | 2 votos: 1780 · 3 votos: 744 | 2 votos: 495 · 3 votos: 179 |
| Painéis distintos | **8, fixos** | os mesmos 8 |

- **Inglês:** 5 anotadores (1, 2, 5, 6, 10). Pares {1,5} e {2,6}; o 10 entra como terceiro.
- **Espanhol:** 5 anotadores (3, 4, 7, 8, 9). Pares {3,7} e {4,8}; o 9 entra como terceiro.
- Exemplo (treino): {4,8} em 572 vídeos; {4,8,9} em 244.
- "10 anotadores, cada um anota uma única língua" está **correto**: cada anotador aparece em uma só língua, em treino e teste. Mas o desenho é de **2 ou 3 votos por item** (não 6 como nos memes), com painéis fixos por língua.
- Gênero dos painéis, por exemplo: {1,5} = F,M; {2,6} = F,F; {2,6,10} = F,F,F.

---

## V3. Ramo de biossinais e embeddings por participante

### V3.1 Dimensões

| Constante | Jow_3 (`Thesis_video_fixed`, células 5 e 28) | Jow_1 | Corrigida (`video_core.py:254-259`) | Paper / `Thesis_video.ipynb` |
|---|---|---|---|---|
| `BIO_RAW_DIM` | 16 | 16 | 16 | 16 |
| `BIO_INPUT_DIM` | 17 | 17 | 17 | 17 |
| Oculta do MLP | 128 (literal) | 128 | `BIO_MLP_HIDDEN=128` | **64** |
| Saída (`BIO_HIDDEN_DIM`) | **128** | **128** | **128** | **32** |
| Dropout | 0,2 | 0,2 | `BIO_DROPOUT=0.2` | 0,2 |
| Forma | `Linear(17→128)→GELU→Dropout(0.2)→Linear(128→128)→GELU` | idem (célula 54) | idem (`:1031-1035`) | `Linear(17→64)→GELU→Dropout→Linear(64→32)`, sem GELU final |
| Parâmetros | 18.816 [medido] | 18.816 | 18.816 [medido] | 3.232 |

O MLP do Jow_1 tem as mesmas formas do Jow_3, já que a célula 54 não acusou *size mismatch* [log].

**Paper × código:** o MLP 17→64→32 e a fusão 2592 só existem no `Thesis_video.ipynb`, que não está ligado a nenhuma submissão. Jow_1, Jow_3 e a versão corrigida usam **17→128→128**.

### V3.2 As 16 features, clipping e imputação [código; iguais em todas as versões]

Ordem canônica (`video_core.py:566-592`; célula 9 do fixed):

| # | Fonte | Feature |
|---|---|---|
| 0 | ET | `3d_eye_states_pupil diameter left [mm]_mean` |
| 1 | ET | `3d_eye_states_pupil diameter left [mm]_std` |
| 2 | ET | `3d_eye_states_pupil diameter right [mm]_mean` |
| 3 | ET | `3d_eye_states_pupil diameter right [mm]_std` |
| 4 | ET | `blinks_count` |
| 5 | ET | `blinks_duration_mean_ns` |
| 6 | ET | `fixations_count` |
| 7 | ET | `fixations_duration_mean_ns` |
| 8 | ET | `fixations_duration_std_ns` |
| 9 | ET | `saccades_count` |
| 10 | ET | `saccades_duration_mean_ns` |
| 11 | ET | `reaction_time` |
| 12–15 | HR | `garmin_hr_mean`, `garmin_hr_std`, `garmin_hr_max`, `garmin_hr_min` |

- **Tempo de reação:** quando é `None` ou NaN, recebe a imputação por língua `RT_IMPUTE = {"en": 29.0, "es": 17.5}`. Depois vale `min(rt, 120.0)` (`video_core.py:544-563`, `:325-326`). Foram imputadas 97 linhas participante×vídeo no treino (de 5048) e 31 no teste (de 1348) [log].
- **Imputação que o paper não menciona, presente nas duas versões:** as demais features `None` viram **0,0 cru** via `safe_float`, *antes* do `RobustScaler`. No treino: `blinks_duration_mean_ns` em 329 linhas; as 4 de pupila e `saccades_duration_mean_ns` em 71; as 2 de duração de fixação em 129; `fixations_count` em 53. No teste: 99 / 25 / 35 / 8 [log].
- **Escalonamento:** `RobustScaler` ajustado só no treino, sobre as linhas participante×vídeo. No Jow_3 são 2271×2 linhas (célula 17); na corrigida, 2019×2.

### V3.3 CVPC e escolha dos 2 participantes [código + log]

- **CVPC** é o cosseno entre os dois vetores de 16-d **já escalados**: `dot/(‖v1‖‖v2‖ + 1e-8)`. É calculado antes da subtração da *baseline* por participante e entra como 17ª dimensão do MLP (`video_core.py:614-622`; célula 9).
- **Escolha dos 2 participantes: não há escolha.** `sensorial.users` tem **exatamente 2 participantes em 100 % dos vídeos** (treino 2524/2524, teste 674/674). Os `by_user` de ET, HR e EEG também têm sempre 2 [log]. O código usa `users[0]` e `users[1]` na ordem do JSON.
- **Paper × dados:** o "2–4 participantes por estímulo" da §5.2 **não aparece nos dados fornecidos**.

### V3.4 Cobertura [log]

- **100 % dos vídeos de treino e de teste** têm ET e HR para os dois participantes.
- O slot de bio só fica zerado por **ablação** (B0) ou por **modality dropout**. O código não tem ramo para bio ausente (daria `KeyError`).
- **Paper × dados:** *"For instances without biometric data, this component is set to zero"* nunca acontece.

### V3.5 Embeddings por participante (`viewer_emb`)

| | Valor | Fonte |
|---|---|---|
| `NUM_VIEWERS` | 10 | `VIEWER_TO_IDX` = `EN1_2…EN5_2`, `ES1_2…ES5_2` [código] |
| `VIEWER_EMB_DIM` | 32 | [código] |
| `VIEWER_PROJ_DIM` | 16 (= `BIO_RAW_DIM`) | [código] |
| `viewer_proj` | `Linear(32→16) + Tanh` | Jow_3 e corrigida. No `Thesis_video.ipynb` não há Tanh [código] |
| Uso | `bio_vec − (proj(emb(u0)) + proj(emb(u1)))/2` | `video_core.py:1137-1142`; célula 28 |

- **`user_ids`** são os índices dos 2 **participantes de laboratório** (ET/HR). **Não são anotadores.**
- São **10 participantes distintos**. No treino: EN1_2 513, EN2_2 441, EN3_2 508, EN4_2 488, EN5_2 50, ES1_2 759, ES2_2 774, ES3_2 726, ES4_2 767, ES5_2 22.
- O par de participantes tem sempre a língua do vídeo, e existem 18 pares distintos [log].
- **A versão submetida tinha `viewer_emb`:** o Jow_3 sim [código]; o Jow_1 sim (chaves presentes, célula 54) [log]; o Jow_2 [n/v].

### V3.6 `bio_ln` na versão submetida

No Jow_3 (célula 28), `bio_ln = LayerNorm(16)` (32 parâmetros) é aplicado em `bio_centered`, isto é, **depois** da subtração da *baseline* por participante e **antes** da concatenação com o CVPC. O CVPC fica fora da normalização:

```python
bio_centered = bio_vec - (base1 + base2) / 2      # [B,16]
bio_norm  = self.bio_ln(bio_centered)             # [B,16]
bio_input = torch.cat([bio_norm, cvpc], dim=-1)   # [B,17]
bio_feat  = self.bio_mlp(bio_input)               # [B,128]
```

- O Jow_1 também tinha (chave presente na célula 54) [log].
- A versão corrigida removeu (ARCH-08, `video_core.py:1022-1030`).

---

## V4. Texto

1. **LoRA do XLM-R-large** [código + log + medido]: `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=["query","value"]`, 24 camadas. São **1.572.864 parâmetros treináveis** (log da célula 24: `all params: 561,463,296`). É igual na versão corrigida (`video_core.py:312-315`, `:1167-1176`) e no CLIP.
2. **CLS cru de 1024-d:**
   - **Corrigida: sim.** `last_hidden_state[:, 0, :]` vai direto para a fusão, sem projeção nem LayerNorm (`video_core.py:1131-1133`) [código].
   - **Jow_3: não.** O texto que entra na fusão é `cross_norm(txt_feat + cross_out)`, um **LayerNorm(1024)** sobre o CLS somado à saída da cross-attention (célula 28; `inference_video.py:400-403`) [código].
   - **Jow_1:** CLS cru, porque não tinha `cross_norm` (célula 54) [log].
3. **Campo usado:** `text` do JSON (a transcrição fornecida). **Não há pré-processamento**, só tokenização com `padding="max_length"` e `truncation=True`. `emoji` e `wordsegment` são importados na célula 2 do fixed e nunca chamados [código]. Na corrigida a decisão está registrada (`video_core.py:713-726`, BUG-F).
4. **`max_length`:**
   - Jow_3 usa **128**: o default de `EXISTVideoDataset` (célula 15) nunca é sobrescrito; o log da célula 20 mostra `input_ids: [128]`; `inference_video.py:279` também usa 128.
   - A corrigida usa **256** (`video_core.py:281`; `meta.json` do B15 e do CLIP: `"max_length": 256`) [código + log].

---

## V5. Frames e VideoMAE

### V5.1 Extração v1 (submetida) [código; os frames foram apagados]

Fonte: `extract_video.ipynb`, célula 1 (versão de teste) e `.ipynb_checkpoints/extract_video-checkpoint.ipynb` (versão de treino; só os caminhos diferem).
- **16 frames.** Índices `np.linspace(0, total−1, 16, dtype=int)`, ou seja, **truncamento**, sobre `CAP_PROP_FRAME_COUNT` (estimado). Leitura por *seek* com `cap.set(CAP_PROP_POS_FRAMES)`.
- `cvtColor(BGR2RGB)` por frame. Em falha de leitura, duplica o último frame bom **e o reconverte**, devolvendo-o a BGR (ARCH-09); se for o primeiro frame, usa zeros 224×224.
- `VideoMAEImageProcessorFast` com defaults: *resize* da aresta menor para 224 (bilinear), *center crop* 224×224, *rescale* 1/255, normalização ImageNet (0,485/0,456/0,406; 0,229/0,224/0,225).
- Gravado em **float32 `[16,3,224,224]`**. Vídeo ausente vira um tensor de zeros gravado em disco.
- A árvore `frames/` da VM só tem `v2/`; os frames v1 não existem mais.

### V5.2 Extração v2 (corrigida): confirmado, com uma ressalva [código + log]

Fonte: `extract_video_v2.ipynb`, células 3–8; `frames/v2/training/_extraction_meta.json`.
- **64 frames `uint8`**. Decodificação sequencial em 2 passes (`grab`/`retrieve`), sem *seek*. Índices `linspace(0, total_real−1, 64).round()` sobre a contagem **real**. BGR→RGB uma vez só.
- Mesmo *processor* com `do_rescale=False, do_normalize=False`: `shortest_edge=224`, `crop_size=224×224`, sem *cast* (`needs_cast_fallback: false`).
- Normalização ImageNet aplicada no carregamento, `(x/255 − mean)/std`, com mean/std lidos do meta (`video_core.py:452-483`).
- **16 frames por `linspace(0,63,16).round()` = `[0,4,8,13,17,21,25,29,34,38,42,46,50,55,59,63]`: confirmado** (`video_core.py:441-449`).
- 0 vídeos faltantes e 0 preenchidos, em treino e teste.
- **Ressalva:** v1 e v2 são "uniformes", mas **não dão os mesmos índices**. A v1 usa truncamento sobre a contagem estimada; a v2 usa arredondamento sobre a real, com 64 e depois 16.

### V5.3 VideoMAE-base [código + log + medido]

- `config.json` de `MCG-NJU/videomae-base`: `tubelet_size=2`, `num_frames=16`, `image_size=224`, `patch_size=16`. São **(16/2)·(224/16)² = 8·196 = 1568 tokens**, sem CLS.
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`, `["query","value"]` em 12 camadas. São **589.824 treináveis** (log da célula 22 + medido). O resto, 86.236.416, fica congelado.

### V5.4 `TemporalAttentionPool` [código + medido]

- Forma: `Linear(768→128) → Tanh → Linear(128→1)`, *softmax* em `dim=1` sobre os **1568 tokens** e soma ponderada, resultando em 768.
- **98.561 parâmetros** (98.304 + 128 + 128 + 1): **confirmado**.
- É igual no Jow_3 (célula 28), no Jow_1 (célula 54) e na corrigida (`video_core.py:949-967`).
- **Nota para o texto:** o *softmax* é sobre tokens **espaço-temporais** (8 posições de tempo × 196 *patches*), não só sobre o tempo, apesar do nome.
- O `Thesis_video.ipynb` (paper) usa `mean(dim=1)`.

### V5.5 Biases `q_bias`/`v_bias` (ARCH-01)

| | Estado | Evidência |
|---|---|---|
| Jow_3 | **zerados**: o LOAD REPORT da célula 22 lista `q_bias`/`v_bias` como UNEXPECTED e `query/key/value.bias` como MISSING nas 12 camadas. Norma 0,000 medida na VM em 2026-08-08 | [log] (célula 22; medição registrada no CLAUDE.md) |
| Jow_1, Jow_2 | **zerados** | [inferido pelo ambiente: mesmo `transformers` 5.6.0; o checkpoint de 05-13 do `Thesis_video.ipynb` mostra o mesmo LOAD REPORT com MISSING] |
| Corrigida (B15) | **restaurados** antes do `get_peft_model` (`video_core.py:1179-1243`, `:1256-1260`). `meta.json`: `restore_biases: true`. Medido agora: ‖`query.bias`‖ na camada 0 = **17,541** depois da restauração | [código + log + medido] |

Os biases estão congelados, porque só a LoRA treina. Portanto os zeros persistiram durante todo o treino submetido.

---

## V6. Braço CLIP (`video_core_clip.py`, run `run_B2_bio_viewer_seed42_CLIP`)

1. **Encoder: confirmado** [código]. `CLIPFrameEncoder` reformata para `B·16` frames, passa pelo `CLIPVisionModel` (`openai/clip-vit-large-patch14`), toma `last_hidden_state[:, 0, :]` (**CLS antes do `post_layernorm`**) e devolve `[B,16,1024]` (`:307-348`).
2. **Renormalização** [código + log + medido]. É uma afim exata sobre o frame já normalizado em ImageNet: `x_clip = x_vmae·(std_v/std_c) + (mean_v − mean_c)/std_c` (`:210-241`). As constantes do CLIP são lidas do `CLIPProcessor` em tempo de execução:
   - mean = **(0,48145467; 0,45782751; 0,40821072)**;
   - std = **(0,26862955; 0,26130259; 0,27577710)**;
   - estão registradas no `meta.json` da run e foram medidas agora.
3. **Pool** [código + medido]: o mesmo `TemporalAttentionPool` com dimensão 1024 (`Linear(1024→128)→Tanh→Linear(128→1)`), aqui sim só **temporal**, sobre os 16 CLS. São **131.329 parâmetros**.
4. **Fusão: confirmado** [código + log + medido]. **1024 + 768 + 1024 + 128 = 2944 → 1024** (`meta.json`: `fusion_in: 2944`). A fusão tem 3.015.680 parâmetros.
5. **Últimas 4 camadas destravadas**, *full-rank* e sem LoRA [medido]:
   - **50.384.896** treináveis no encoder (= 4 × 12.596.224), de 303.179.776 totais;
   - `post_layernorm` congelado (e sem uso);
   - **sistema inteiro: 56.120.027 treináveis** de 868.805.339 (V11);
   - LR do CLIP 1e-5, *weight decay* 0,01.
6. **Peso da Task C na run oficial: 0,7** [inferido].
   - O `meta.json` da run CLIP **não tem** as chaves `loss_w_*`; o do `CLIP_WC42` tem `loss_w_c: 4.2`.
   - O mínimo de `eval_loss` é 1,516 (contra 2,753 no WC42).
   - ⚠ O `Thesis_video_clip.ipynb` em disco é a **edição posterior para o WC42** (`ARM = "CLIP_WC42"`) e **não tem nenhuma saída**. O código exato que treinou o CLIP oficial não está preservado: o git só tem o commit de 09-06.
7. **Estado reportado: final da época 15.** O md5 de `trainable.safetensors` é igual ao de `epoch_states/epoch_15.safetensors` (`1b3ee93e…`). O *mtime* de 16/08 é uma restauração, não outro estado [log].

---

## V7. Áudio

1. **Extração** [código + log]:
   - **v1** (`extract_audio.ipynb`, célula 0):
     - `torchaudio.load` do mp4; mono pela média dos canais; reamostragem para 16 kHz;
     - Whisper-large-v3 (`openai-whisper`), **só o encoder**;
     - `whisper.pad_or_trim` para **30 s** (480.000 amostras) e log-mel com `n_mels=128`;
     - encoder → `[1500,1280]`, **média sobre os 1500 frames** (silêncio incluído), float32;
     - vídeo sem áudio legível gera zeros gravados em disco. **Confirmado.**
   - **v2** (`extract_audio_v2.ipynb`, célula 3):
     - janelas de 30 s sobre o áudio inteiro;
     - por janela, `n_valid = clamp(round(segundos_válidos × 50), 1, 1500)` e média só dos `n_valid` primeiros frames do encoder;
     - entre janelas, média **ponderada por `n_valid`** (equivale à média sobre todos os frames válidos);
     - meta do treino: 1689 vídeos com 1 janela, 813 com 2, 22 com 3. **Confirmado.**
     - Ressalva registrada no meta: a *self-attention* do encoder ainda vê o *padding* dentro de cada janela.
2. **`audio_proj`** [código + medido]: `Linear(1280→768) → GELU → LayerNorm(768)`, com **985.344 parâmetros**.
   - **É igual no Jow_3** (célula 28) **e na corrigida** (`video_core.py:1016-1020`).
   - O Jow_1 também tem essa forma: as chaves `audio_proj.0`/`audio_proj.2` casaram na célula 54 [log].
   - **Paper × código:** *"A linear projection maps these 1280-d features to 768"* descreve só o `Thesis_video.ipynb` (`nn.Linear` puro).

---

## V8. Fusão, cabeças e perdas

1. **Fusão** `Linear(2688→1024) → GELU → Dropout(0.1)`, com **2.753.536 parâmetros**, **sem normalização depois da concatenação**. Confirmado nas duas versões (célula 28; `video_core.py:1047-1050`) [código + medido].
   - No Jow_3 o slot de texto passa por `cross_norm` *antes* do `cat`. É uma normalização por ramo e não viola a regra.
   - **Paper × código:** *"two-layer fusion MLP"* é falso; a fusão é **uma** `Linear`.
2. **`cross_attn` do Jow_3** [código + medido]:
   ```python
   self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True, dropout=0.1)
   self.cross_proj = nn.Linear(768, 1024)
   self.cross_norm = nn.LayerNorm(1024)
   # forward (depois dos 4 ramos, antes do cat):
   vid_proj = self.cross_proj(vid_feat).unsqueeze(1)              # [B,1,1024]  ← UMA key
   cross_out, _ = self.cross_attn(query=txt_feat.unsqueeze(1), key=vid_proj, value=vid_proj)
   txt_feat = self.cross_norm(txt_feat + cross_out.squeeze(1))    # substitui o slot de texto
   fused = self.fusion(torch.cat([vid_feat, aud_feat, txt_feat, bio_feat], -1))  # vídeo cru continua
   ```
   - **Parâmetros:** 4.198.400 + 787.456 + 2.048 = **4.987.904: confirmado**. São **45,3 %** dos 11.017.979 treináveis.
   - **Comportamento:** com uma única key, o *softmax* vale 1. A atenção vira um mapa linear fixo do vídeo somado ao texto.
   - **Ativação:** a cross-attention roda mesmo quando o *modality dropout* zerou vídeo ou texto (BUG-I).
   - **Paper × código:** *"text attends to video and video attends to text, and the attended representations enter the concatenation in place of the original CLS tokens"*. **Falso.** A atenção é **unidirecional** (texto ← vídeo) sobre 1 token, e só o slot de texto é substituído; o vetor de vídeo de 768-d entra cru.
3. **Cabeças:** `Linear(1024→1)` (1.025 parâmetros), `Linear(1024→3)` (3.075) e `Linear(1024→6)` (6.150). Confirmado [código + medido].
4. **Convenção da Tarefa A: `sigmoid = P(YES)`, eixo `[P(NO), P(YES)]`. Confirmado nas duas versões** [código + log].
   - No Jow_3 o alvo é a proporção de YES (célula 10); `compute_metrics` monta `[1−p, p]` (célula 41); `inference_video.py:717-722`.
   - Na corrigida: `TASK_A_PROB_ORDER = ("NO","YES")` (`video_core.py:181`) e `selfcheck_task_a_convention()`.
   - Nos 3 runs submetidos, o soft soma 1 e o hard bate com `p_yes ≥ p_no` em 674/674 vídeos [log].
5. **Perda total `1,0·L_A + 0,7·L_B + 0,7·L_C`. Confirmado nas duas versões** (célula 30; `video_core.py:1309-1361`) [código].
   - `L_A` é BCE com alvo soft.
   - `L_B` é a entropia cruzada soft com **soma** sobre as 3 classes.
   - `L_C` é BCE por classe com **média** sobre as 6.
   - O braço CLIP oficial usa 0,7 [inferido, V6.6]; o WC42 usa 4,2.
6. **Alvos** [código + log]:
   - votos `UNKNOWN` excluídos (treino: 74 na A, 74 na B, 78 na C);
   - voto `-` mapeado para NONE na B e na C (2988 votos cada); na A, `-` não ocorre;
   - sentinelas: A = −100,0 (escalar); B/C = −1,0 na coluna 0;
   - `REPORTED` não ocorre nos dados da Task 3;
   - o alvo C é `contagem/nº de anotadores válidos` por categoria (multirrótulo, não soma 1).

---

## V9. Treino

| | Jow_3 (submetida) | B15 (corrigida, VideoMAE) | CLIP (corrigida) |
|---|---|---|---|
| Otimizador | AdamW | AdamW | AdamW |
| LoRA (texto **e** vídeo, mesmo grupo) | lr 1e-4, wd 0,01 | lr 1e-4, wd 0,01 | lr 1e-4, wd 0,01 (só texto) |
| `viewer_emb` | lr 1e-3, wd 0 | lr 1e-3, wd 0 | lr 1e-3, wd 0 |
| Camadas novas | lr 1e-3, wd **0,05** (todas, inclusive biases e LN) | lr 1e-3: wd 0,05 nos pesos; wd 0 em `.bias`/`.norm.` ¹ | idem B15 |
| `cross_*` | lr 1e-4, wd 0,01 | — | — |
| Camadas do CLIP | — | — | lr **1e-5**, wd 0,01 |
| Scheduler / warmup | cosine, `warmup_ratio=0.1` | cosine, `warmup_steps=0.1` (95 de ~945 passos, log) | cosine, `warmup_steps=0.1` |
| Épocas | 15 | 15 | 15 |
| Batch | 2 × acum. 16 = 32 | 2 × 16 = 32 | 2 × 16 = 32 |
| Passos | 1065 (71/época) | 960 (64/época) | — |
| Precisão / checkpointing | bf16 / sim | bf16 / sim | bf16 / sim |
| Determinismo | **não** (as flags foram adicionadas depois) | sim + fingerprints | sim + fingerprints |
| Modality dropout | **p = 0,05**, **um sorteio por batch**, no máximo uma modalidade (faixas de 0,05) | **p = 0,05 por modalidade, por amostra** (`torch.rand(B)`) | idem B15 |
| Estado reportado | **média SWA das épocas 5–15** | **final da época 15** (`argmin eval_loss` = época 5, só registro) | **final da época 15** (`argmin` = época 2, só registro) |
| Tempo de treino | **3267 s** (54 min; 05-14 18:25–19:16) | **5594 s** (93 min) | **6220 s** (104 min) |
| Split | 2271/253 | 2019/505 | 2019/505 |
| Fonte | células 32, 33, 50, 52 [código + log] | `video_core.py:1869-1897`; `Thesis_video_improved.ipynb`, células 7, 21 e 27; `meta.json` [código + log] | `video_core_clip.py:466-526`; `meta.json` [código + log]; *TrainingArguments* = B15 segundo o markdown do notebook [inferido] |

¹ O predicado `_no_decay` (`video_core.py:1876-1877`) não pega `audio_proj.2.weight` (o LayerNorm do áudio). Esse peso recebe wd 0,05. É um detalhe para não escrever "LayerNorms sem decay" de forma absoluta.

**Notas:**
- **Paper × código (wd):** o paper diz *"weight decay 10⁻²"* uniforme. O Jow_3 usa 0,05 nas camadas novas, 0 no `viewer_emb` e 0,01 em LoRA/cross.
- **Modality dropout (corrige a premissa do pedido):** o Jow_3 usa **0,05**, o default do `forward` (célula 28), já que o Trainer nunca passa o argumento. As faixas são `[0,.05)` vídeo, `[.05,.10)` áudio, `[.10,.15)` texto, `[.15,.20)` bio, ou seja, 20 % dos passos afetados. O 0,15 do paper e da premissa só existe no `Thesis_video.ipynb`. Jow_1/Jow_2: [n/v].
- **Split da submetida:** `train_test_split(test_size=0.1, random_state=2025, stratify="YES" se houver ≥1 voto YES)` (célula 16), resultando em **2271/253**. O val foi gravado em `Dataset/val_seed_2025.json` (253). Esse arquivo foi o `--test-json` do `--eval` dos **três** modelos no `.bash_history` (linhas 254, 260 e 267). Logo, o mesmo split sustenta Jow_1, Jow_2 e Jow_3 [log].
- **Split da corrigida:** artefato `Dataset/val_split_42.json` com `split_seed=42`, `val_fraction=0.2`, `stratify="task_a"`, **2019/505** [log].
- **Atenção na redação:** "estratificado pela Tarefa A" significa **"tem pelo menos um voto YES"**, e não voto majoritário. Isso vale nas duas versões (célula 16; `video_core.py:818-820`).

---

## V10. Limiares da Tarefa C

### V10.1 Submetida

**Como o Jow_3 ajustou** [código + log]: `tune_task_c_thresholds` (célula 57) rodou **uma vez, depois do treino, sobre o modelo SWA final**.
- Usou os **253 vídeos de val inteiros**, o mesmo conjunto usado para monitorar o treino.
- Busca gulosa por coordenada, grade 0,05–0,60 em passos de 0,05, **com NONE incluído**, maximizando a ICM-Norm hard.
- Regra hard: todo rótulo com `p ≥ limiar`; se nenhum passar, usa o *argmax* **entre os 6, incluindo NO**. NO pode sair junto com categorias.
- Resultado: `[0.45, 0.55, 0.15, 0.30, 0.50, 0.25]`, com val 0,3725 (contra 0,3531 com 0,5).

**Reconstrução a partir dos arquivos submetidos** [log]. Para cada rótulo, o intervalo `(maior p excluído, menor p incluído]` sobre os 674 vídeos, ignorando as linhas de *fallback*:

| Rótulo (ordem `LABELS_2_3`) | Jow_1 | Jow_2 | Jow_3 |
|---|---|---|---|
| NO | (0,2472; 0,2516] → **0,25** | (0,4499; 0,4510] → **0,45** | (0,4456; 0,4501] → **0,45** |
| IDEOLOGICAL-INEQUALITY | (0,3356; 0,3514] → **0,35** | (0,3472; 0,3607] → **0,35** | (0,5440; 0,5708] → **0,55** |
| MISOGYNY-NON-SEXUAL-VIOLENCE | (0,2471; 0,2550] → **0,25** | (0,1417; 0,1539] → **0,15** | (0,1491; 0,1502] → **0,15** |
| SEXUAL-VIOLENCE | (0,3275; 0,4522] → **0,35, 0,40 ou 0,45 (ambíguo)** | (0,1474; 0,1508] → **0,15** | (0,2616; 0,3224] → **0,30** |
| OBJECTIFICATION | (0,4398; 0,4634] → **0,45** | (0,3433; 0,3555] → **0,35** | (0,4983; 0,5042] → **0,50** |
| STEREOTYPING-DOMINANCE | (0,2471; 0,2548] → **0,25** | (0,2483; 0,2520] → **0,25** | (0,2481; 0,2512] → **0,25** |

- **Jow_3:** idêntico ao `task_c_thresholds_seed_2025.json` em disco.
- **Jow_1 e Jow_2:** valores da grade, mas **não batem** com nenhum JSON sobrevivente (`seed_42`, `seed_7`, `seed_2025`). Vieram de versões anteriores do `task_c_thresholds_seed_2025.json`, sobrescritas pelo Jow_3 às 19:19. A época ou estado em que foram ajustados é [n/v]. Ajuste pós-treino sobre o estado final, como no fixed, é [inferido].
- **Linhas com "NO" junto com categorias** nos arquivos hard submetidos: Jow_1 **107**, Jow_2 **84**, Jow_3 **62** (de 674). É a predição autocontraditória que o ARCH-10 corrigiu.

### V10.2 Corrigida: confirmado [código + log]

- `split_val_halves(n, 42)` (`video_core.py:1713-1721`) faz uma permutação com `RandomState(42)`: metade de **ajuste** (252) e de **reporte** (253).
- **NO fica fixo em 0,5**, fora da grade: a busca usa `range(1, 6)` (`:1526`).
- Regra hard (`:1450-1459`):
  1. categorias acima do limiar saem sozinhas;
  2. senão, `["NO"]` se `p_NO ≥ 0,5`;
  3. senão, a **categoria não-NO mais provável**.
- O gold também passa pela regra nova, então os números de C hard não se comparam com os da geração antiga.
- **Resultados na metade de reporte** (o que vai para a tese, tal como está):

| | Limiares (NONE, IDEO, MNSV, SV, OBJ, SD) | Ajustado | Default 0,5 |
|---|---|---|---|
| B15 | 0,50 / 0,15 / 0,20 / 0,20 / 0,50 / 0,10 | **0,3171** | 0,3403 |
| CLIP | 0,50 / 0,35 / 0,25 / 0,15 / 0,35 / 0,40 | **0,3697** | 0,3733 |

**Nos dois braços, os limiares ajustados ficam abaixo do default na metade independente.**

---

## V11. Contagem de parâmetros

Instanciação em CPU na VM (`transformers` 5.6.0, `CUDA_VISIBLE_DEVICES=""`), com pesos pré-treinados dos backbones e **sem** pesos treinados. Script: `EXIST_VIDEOS/analysis/count_params_v11.py` (rodar na VM: `python - submitted|b15|clip < count_params_v11.py`) [medido]. Os totais da submetida batem com o log da célula 51 (`total 657144827, trainable 11017979`). Os treináveis da corrigida batem com os tamanhos dos `trainable.safetensors` (B15: 24,1 MB ≈ 6,03 M × 4 B; CLIP: 224,5 MB ≈ 56,1 M × 4 B).

| Módulo | Submetida (Jow_3) | B15 | CLIP |
|---|---|---|---|
| Encoder de vídeo, congelado | 86.236.416 (VideoMAE) | 86.236.416 | 252.794.880 (CLIP ViT-L/14) |
| Encoder de vídeo, treinável | 589.824 (LoRA) | 589.824 (LoRA) | **50.384.896** (4 camadas *full-rank*) |
| XLM-R-large, congelado | 559.890.432 | 559.890.432 | 559.890.432 |
| XLM-R LoRA, treinável | 1.572.864 | 1.572.864 | 1.572.864 |
| `video_pool` | 98.561 | 98.561 | 131.329 |
| `audio_proj` | 985.344 | 985.344 | 985.344 |
| `viewer_emb` + `viewer_proj` | 320 + 528 | 320 + 528 | 320 + 528 |
| `bio_ln` | 32 | — | — |
| `bio_mlp` | 18.816 | 18.816 | 18.816 |
| `fusion` | 2.753.536 (2688→1024) | 2.753.536 (2688→1024) | 3.015.680 (2944→1024) |
| `head_a` / `head_b` / `head_c` | 1.025 / 3.075 / 6.150 | idem | idem |
| `cross_attn` / `cross_proj` / `cross_norm` | 4.198.400 / 787.456 / 2.048 | — | — |
| **Treinável** | **11.017.979** (1,68 %) | **6.030.043** (0,92 %) | **56.120.027** (6,46 %) |
| **Congelado** | **646.126.848** | **646.126.848** | **812.685.312** |
| **Total** | **657.144.827** | **652.156.891** | **868.805.339** |

- **B15 = submetida − (`cross_*` 4.987.904 + `bio_ln` 32).**
- **Referência:** a variante do paper (`Thesis_video.ipynb`, 2592, sem `bio_ln`, *pool* e cross-attention) teria 5.816.058 treináveis (3.653.370 em camadas novas + 2.162.688 de LoRA). Ela não corresponde a nenhum run submetido.

---

## Resumo para a errata: divergências paper × código

Comparação com o **Jow_3** (verificado). A maioria vale também para o Jow_1 pela prova da célula 54.

1. **Demografia de anotadores:** não existe em nenhum código de vídeo, e o dataset da Task 3 nem fornece os atributos (só gênero). O paper diz que segue o mecanismo da Task 2 (§5.1, §5.4, Fig. 2). (V2)
2. **Fusão de 2688, e não 2592.** O slot de bio tem 128-d, e não 32. (V3.1, V8.1)
3. **Bio-MLP `17→128→128` (+GELU), e não `17→64→32`.** (V3.1)
4. **`bio_ln` (LayerNorm(16)) depois da subtração da *baseline* por participante:** existe e o paper não menciona. (V3.6)
5. **Projeção de áudio `Linear+GELU+LayerNorm`, e não "linear".** (V7.2)
6. **Fusão de uma camada, e não "two-layer".** (V8.1)
7. ***Pooling* por atenção sobre os 1568 tokens:** o paper não descreve. (V5.4)
8. **Cross-attention unidirecional sobre uma única key, que só substitui o slot de texto.** Não é bidirecional e não substitui o vídeo. (V8.2)
9. **O Jow_3 também tem SWA (épocas 5–15, 11 snapshots).** O paper atribui SWA só ao Jow_2, com "8–14 (7 snapshots)", número que nenhum código sobrevivente produz. (V1.2)
10. ***Modality dropout* de 0,05 por batch, e não 0,15.** (V9)
11. ***Weight decay* de 0,05/0/0,01 por grupo, e não 10⁻² uniforme.** (V9)
12. **"2–4 participantes por estímulo":** os dados têm sempre 2. **"Zero quando não há biossinal":** a cobertura é de 100 %. Imputação de features `None` por 0,0 antes do scaler: o paper não menciona. (V3.2–V3.4)
13. **Biases `q_bias`/`v_bias` do VideoMAE zerados nos três runs:** não é declarado. (V5.5)
14. **Limiares da Tarefa C ajustados no mesmo val de 253 usado para monitorar, com NONE na grade e NO coocorrendo com categorias** (62–107 linhas por run). (V10.1)
15. **Texto do Jow_3 passa por `cross_norm`:** não é o CLS cru. (V4.2)

## Itens que continuam sem verificação possível

- **Jow_2:** arquitetura, código, log, épocas, janela do SWA (o paper diz 8–14) e se tinha cross-attention. Restam só as predições, o nome do diretório `_swa` e os limiares reconstruídos.
- **Jow_1:** se usou SWA, número de épocas, época/estado de ajuste dos limiares e o valor exato do limiar de SEXUAL-VIOLENCE (0,35, 0,40 ou 0,45). A arquitetura vem da célula 54, com a ressalva de sobrescrita não registrada.
- **Script de inferência do Jow_1/Jow_2:** a versão sem cross-attention se perdeu. O `inference_video.py` atual só serve ao Jow_3.
- **Pesos dos 3 runs submetidos:** apagados (`.bash_history`, linha 285). Nenhuma métrica dos runs submetidos pode ser recomputada fora do leaderboard.
- **Frames v1:** apagados, então não dá para conferir os índices efetivos nem os casos de *fallback* do ARCH-09.
- **CLIP oficial:** o notebook exato que o treinou foi sobrescrito pela edição WC42 (sem saídas). `LOSS_W_C=0,7` e as *TrainingArguments* são [inferido].
- **Reprodutibilidade bit a bit do Jow_3:** impossível. Rodou sem as flags de determinismo, e o baseline do doc 09 é uma reprodução da arquitetura, não dos pesos.
