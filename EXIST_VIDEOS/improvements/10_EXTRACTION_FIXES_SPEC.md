# 10 — Especificação das correções de extração (vídeo + áudio)

**O que este documento é:** a lista completa do que muda na extração de features da
Task 3, com o motivo de cada item. É a especificação primária para quem for escrever o
código.

**O que este documento NÃO é:** o código. O plano de implementação detalhado sai daqui.

**Custo total de GPU:** ~45 min (áudio ~10 min + frames ~35 min). O caro não é extrair —
é a invalidação. Ver §7.

---

## 0. Decisões já tomadas (não reabrir)

Confirmadas pelo autor em 2026-08-08:

| # | decisão | consequência |
|---|---|---|
| D1 | **64 frames por vídeo, `uint8`** | 9,6 MB/vídeo — **mesmo disco de hoje**. Permite testar 16/32/64 depois só subamostrando, sem nova extração |
| D2 | **Escopo inclui `video_core.py`** | sem adaptar o consumidor, os dados v2 ficam no disco sem ninguém que os leia corretamente |
| D3 | **Arquivos novos `_v2`**, originais intocados | os originais são o único registro de como o v1 foi gerado — apagá-los repetiria o problema V6 |
| D4 | **Faltante → falha alta no build do dataset** | erro em 5 s com a lista de IDs, em vez de descobrir depois de 4 h de GPU |

**D1 em detalhe — por que 64 e não 16:** 64 frames em `uint8` custa exatamente o mesmo
disco que 16 frames em `float32` custam hoje (9,6 MB). Além de habilitar o teste de
contagem de frames, permite o protocolo de **janela densa**: 16 frames *consecutivos*
dos 64 formam uma janela de ~4,6 s na mediana, próxima do regime de pré-treino do
VideoMAE (~2,1 s). Com 16 salvos, nada disso é possível sem re-extrair.

---

## 1. Correções na extração de FRAMES

### V1 — dupla conversão de canais no fallback (ALTA)

**Onde:** `extract_video.ipynb`, `sample_frames_uniform`.

```python
if not ret:
    frame = frames[-1] if frames else np.zeros((224,224,3), dtype=np.uint8)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # ← aplicado TAMBÉM ao fallback
```

**O defeito:** `frames[-1]` já foi convertido para RGB. `cvtColor(BGR2RGB)` troca o canal
1 com o 3 incondicionalmente — não inspeciona a imagem. Aplicá-lo a um frame já
convertido o manda de volta para BGR.

**Por que importa:** produz frames com **vermelho e azul trocados** no meio de uma
sequência correta. O VideoMAE foi pré-treinado em vídeos onde tom de pele é alaranjado e
céu é azulado; um frame invertido é uma imagem nítida, coerente e completamente fora da
distribuição. O modelo processa com confiança total e extrai características sem sentido.

**Prevalência: NÃO MEDIDA.** O script imprimia avisos, mas esse output não ficou salvo no
notebook. Não é detectável pela varredura de zeros — um frame com canais trocados tem
valores perfeitamente normais, só nas posições erradas.

**Correção:** converter **uma vez só**, depois de selecionar os frames finais. Nunca
dentro do loop de leitura.

---

### V2 — zeros gravados no disco mascaram vídeos ausentes (ALTA)

**Onde:** `extract_video.ipynb` (2 sítios) e `extract_audio.ipynb` (2 sítios).

```python
if not video_path.exists():
    torch.save(torch.zeros(NUM_FRAMES, 3, 224, 224), out_path)   # grava e segue
    continue
```

**Dois defeitos independentes:**

1. **Zeros aqui não significam "preto".** Os frames são gravados **já normalizados**.
   Depois da normalização ImageNet, `0` é o **pixel médio** do ImageNet, não preto
   (preto ≈ −2,1; branco ≈ +2,6). Um tensor de zeros é uma entrada plausível e
   familiar ao modelo — não um buraco detectável.
2. **O aviso do treino é código morto.** O `try/except` de `load_frames` só dispara se o
   arquivo não existir. Como a extração sempre grava, o `except` é inalcançável e o
   `[WARN]` nunca aparece. **Ninguém nunca soube quantas amostras eram zeros.**

**Prevalência: 0 / 3 198 — MEDIDO** (`analysis/frames_zero_scan.json`, varredura de
todos os tensores por leitura parcial via `zipfile`). Combinado com o áudio (0 ausentes,
0 sem stream, 0 falhas de probe), a armadilha nunca disparou nestes dados.

**Correção (D4):** não gravar `.pt` nenhum quando falha. Acumular os IDs e gravar
`_missing.json` ao lado das features. O dataset falha alto no build, listando os IDs.

**Regra:** *ausência de dado nunca deve ser representada por um valor que o modelo lê
como dado válido.*

---

### V3 — um frame ruim apaga o vídeo inteiro (ALTA)

**Onde:** `extract_video.ipynb`, mesmo bloco do V1.

**A cadeia:**
```
1 frame falha na leitura
  → fallback cria np.zeros((224,224,3))          ← 224×224, tamanho FINAL
  → np.stack junta com frames em resolução nativa (ex. 1080×1920) → ValueError
  → except Exception genérico captura           ← engole o bug
  → return None
  → o chamador grava torch.zeros(...)
  → O VÍDEO INTEIRO vira zeros
```

**Por que importa:** um único frame ilegível, num vídeo perfeitamente bom, destrói os 16.
E o `except Exception` foi escrito para "vídeo corrompido" mas engole **bugs de
programação** — um erro de shape que deveria gritar vira uma amostra silenciosamente
vazia. O `CLAUDE.md` já registra "guards antes de `np.stack`" como lição de um incidente
anterior.

**Prevalência: 0 / 3 198 — MEDIDO** (mesma varredura do V2).

**Correção:** ver V4 — a mesma reescrita resolve V1, V3 e V4.

---

### V4 — os frames obtidos não são os frames pedidos (MÉDIA)

**Onde:** `extract_video.ipynb`, `sample_frames_uniform`.

```python
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # estimado
indices = np.linspace(0, total - 1, n, dtype=int)
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)          # salto impreciso
```

**Dois defeitos:**

1. **`cap.set(POS_FRAMES)` é aproximado.** Vídeo comprimido guarda keyframes completos e
   frames intermediários que só têm diferenças. Não existe acesso aleatório real — o
   OpenCV frequentemente devolve o keyframe mais próximo, sem aviso.
2. **`CAP_PROP_FRAME_COUNT` é estimado**, tipicamente por `duração × fps`. Vídeos de
   TikTok costumam ter **taxa de quadros variável (VFR)**, o que invalida essa conta. Se
   o total está errado, os índices já nascem errados.

**Por que importa:** amostragem irregular, com desvio que **varia por vídeo** conforme
onde caem os keyframes. O componente temporal do modelo assume espaçamento uniforme.

**Prevalência: NÃO MEDIDA.** O mecanismo é limitação documentada do OpenCV; o impacto
neste dataset é hipótese. Medir exigiria decodificar tudo pelos dois métodos e comparar.

**Correção — leitura sequencial. Resolve V1, V3 e V4 de uma vez:**

- Ler com `cap.read()` sucessivamente (a operação nativa do decodificador, sem estimativa)
- `len(frames)` é a **contagem real**, não `CAP_PROP_FRAME_COUNT`
- Escolher os 64 índices **depois** de saber quantos existem → todos têm o mesmo shape
  por construção, o `np.stack` não pode falhar (V3)
- Converter BGR→RGB **uma vez**, no fim, só nos 64 selecionados (V1)
- Nada é inventado: se não há frames, devolve `None` e o item entra em `_missing.json`

#### ⚠ V4-MEM — a versão ingênua desta correção estoura a memória

**Acumular todos os frames em resolução nativa não é viável.** Medido, com as durações
reais e 30 fps:

| vídeo | frames | full-res na memória | já redimensionado 224×224 |
|---|---|---|---|
| mediana (17,2 s) | 515 | 3,2 GB | 78 MB |
| p90 (57,3 s) | 1 719 | 10,7 GB | 259 MB |
| máx treino (60,6 s) | 1 819 | 11,3 GB | 274 MB |
| **máx test (135,1 s)** | **4 054** | **25,2 GB** | **610 MB** |

Duas saídas, **a escolher no plano de implementação**:

- **(a) Dois passes.** Passe 1 decodifica e conta (descartando os frames); passe 2
  decodifica de novo e guarda só os 64 alvo. Memória O(64). Custo: decodifica 2×.
- **(b) Um passe com redimensionamento imediato.** Redimensiona cada frame para 224×224
  ao lê-lo e acumula (150 KB/frame). Memória ≤ 610 MB no pior caso do dataset atual.
  Custo: redimensiona todos os frames, não só 64.

**Recomendação: (a)**, por ser O(64) em memória independentemente da duração — (b) tem um
teto que depende do dataset e quebraria num vídeo mais longo. Mas (b) pode ser mais
rápido. **Medir antes de fixar**, e registar a escolha.

> Nota honesta: a proposta original do ARCH-09 (`while True: frames_bgr.append(f)`)
> ignora este ponto. Está errada como escrita.

---

### V5 — `float32` no disco: 24,3 GB lidos por época (ALTA — pré-requisito do OPT-04)

**Fato medido:** cada tensor ocupa **9 634 903 B** (`[16,3,224,224]` float32). Uma época
lê **24,3 GB**; as 15 épocas leram ~365 GB.

**Por que float32 é desperdício aqui:** os valores vieram de inteiros 0–255 (256 valores
possíveis por canal) e a faixa medida é `[−2,10; +2,64]`. E o modelo treina em **bf16** —
lê-se 4 bytes para converter imediatamente para 2 bytes com menos dígitos.

**⚠ Não é ganho de velocidade hoje.** 24,3 GB ÷ 340,6 s = **~71 MB/s**, e o loader com 2
workers sustenta isso sem esforço. **Não prometer aceleração.** O valor está em ser
**pré-requisito do OPT-04** (batch 2→8), que dobraria a demanda para ~142 MB/s e faria o
loader virar gargalo.

**Correção (D1):** gravar `uint8`, aplicar a normalização ImageNet no carregamento.

**⚠ Só é válido dentro de uma re-extração.** Converter os `.pt` atuais para uint8 exigiria
inverter `round((x·σ+µ)·255)`, o que quantiza em ~**0,017** no espaço normalizado — da
mesma ordem do ulp do bf16 (~0,016). Não é round-trip inócuo. Numa extração nova parte-se
dos pixels originais, que já eram inteiros: não há nada a preservar.

*(Para contraste: converter para **fp16** os `.pt` atuais seria seguro — ulp ~0,002,
estritamente mais fino que o bf16 da aritmética. Mas é 2× em vez de 4×, e a re-extração
torna a questão irrelevante.)*

#### ⚠ V5-NORM — o ponto de maior risco técnico de toda a especificação

Hoje o `VideoMAEImageProcessorFast` faz três coisas numa chamada:
**redimensionar → reescalar (÷255) → normalizar (ImageNet)**.

Para gravar uint8 é preciso **manter o redimensionamento idêntico** e desligar as outras
duas — tipicamente `do_rescale=False, do_normalize=False`.

**Se o redimensionamento/crop mudar, o v2 difere do v1 em duas variáveis em vez de uma, e
a comparação de métricas perde o sentido.** O processor faz resize da aresta menor
seguido de center crop; reimplementar isso à mão é onde o erro entra.

Requisitos:
1. **Verificar** que o processor desta versão aceita `do_rescale=False, do_normalize=False`
   e devolve uint8 já redimensionado/cropado.
2. **Ler `image_mean`/`image_std`/`size`/`crop_size` do próprio processor** e gravá-los no
   `_extraction_meta.json`. **Não hardcodar** valores ImageNet.
3. **Teste de equivalência obrigatório:** para ≥5 vídeos, comparar
   `normalize(uint8_v2/255)` contra o `.pt` v1 correspondente. Devem coincidir dentro da
   tolerância de quantização (~0,017 em espaço normalizado). Se divergirem mais, o
   redimensionamento mudou — **parar e investigar**.

*(O teste 3 só é comparável nos primeiros 16 dos 64 frames se os índices coincidirem;
como o v2 usa contagem real e o v1 usava estimativa, os índices podem diferir. Fazer o
teste sobre um vídeo cujo `CAP_PROP_FRAME_COUNT` bate com a contagem real, ou comparar
estatísticas por canal em vez de pixel a pixel.)*

---

### V6 — proveniência perdida (ALTA — não tem correção retroativa)

**Onde:** ambos os scripts de extração.

```python
"""... saves → frames/train/{id_EXIST}.pt"""      # docstring diz TRAIN
VIDEO_DIR  = Path("./Dataset/test/videos/")       # código aponta para TEST
OUTPUT_DIR = Path("./frames/test/")
```

**O defeito:** o script commitado só gera os dados de **teste**. Mas `frames/train/` tem
2 524 arquivos — produzidos por uma versão editada à mão que **não existe mais no
repositório nem no histórico**.

**Por que importa:** não há como provar que treino e teste foram extraídos com o mesmo
`NUM_FRAMES`, o mesmo modelo de normalização, o mesmo `TARGET_SR`. Se divergissem, o
treino rodaria sem erro, a validação (que sai do mesmo arquivo do treino) pareceria
ótima, e o modelo desabaria no teste **sem nenhuma mensagem em lugar nenhum**.

Sintoma correlato: o ARCH-09 não conseguiu contar vídeos afetados porque *"o script
imprime `skipped` mas esse output não está no notebook salvo"*. A informação existiu, foi
impressa, e evaporou.

**Correção, duas partes:**

1. **Uma constante, caminhos derivados** — torna a divergência impossível por construção:
   ```python
   SPLIT = "training"                                  # a ÚNICA linha a editar
   VIDEO_DIR  = Path(f"./Dataset/{SPLIT}/videos/")
   OUTPUT_DIR = Path(f"./frames/v2/{SPLIT}/")
   JSON_PATH  = f"./Dataset/{SPLIT}/EXIST2026_{SPLIT}.json"
   ```
   ⚠ Confirmar os nomes reais dos JSON: treino é `EXIST2026_training.json`, teste é
   `EXIST2026_test_clean.json` — **não seguem o mesmo padrão**. A f-string ingênua quebra.

2. **`_extraction_meta.json` ao lado das features**, com tudo que define o resultado:
   `split`, `num_frames`, `dtype`, `model_name`, `image_mean`/`image_std`/`size` lidos do
   processor, estratégia de amostragem, `n_items`, `n_missing`, `missing_ids`, versões de
   `torch`/`transformers`/`opencv`, data.

**Regra:** *dados sem proveniência são dados que você não pode defender.*

---

### V7 — árvore versionada `v2/` (MÉDIA — mas custaria uma onda de re-treino)

**Onde:** ambos os scripts. `if out_path.exists(): continue`.

**O defeito:** os scripts são resume-safe. Depois de aplicar V1–V6, **re-rodar sobre a
árvore antiga não faz nada** — todos os `.pt` já existem. Zero arquivos novos, e o treino
seguinte usa features velhas.

**O modo de falha:** tudo roda sem erro e as métricas parecem simplesmente *"não
melhorou"*. Conclui-se que as correções eram irrelevantes, quando nunca foram aplicadas.

**Correção:** gravar em `frames/v2/{split}/` e `audio_features/v2/{split}/`. Três ganhos:
o skip volta a ser correto (diretório vazio); as duas gerações coexistem para comparação;
a proveniência de qual treino usou qual extração fica em disco.

**Custo em disco:** frames v2 (64 uint8) ≈ 9,6 MB × 3 198 ≈ **31 GB**; áudio v2 ≈ 16 MB.
⚠ **Verificar espaço livre na VM antes de começar** — o último incidente de disco cheio
matou uma run de ablação a meio. Havia 25 GB livres na última medição. **Isto não cabe.**
Ver §6.

---

## 2. Correções na extração de ÁUDIO

### A1 — máscara do padding: promediar só o áudio real (ALTA)

**Onde:** `extract_audio.ipynb`, `extract_whisper_features`.

```python
audio = whisper.pad_or_trim(waveform.numpy())     # SEMPRE 480000 amostras = 30 s
encoder_out = model.encoder(mel)                   # [1, 1500, 1280]
features = encoder_out.squeeze(0).mean(dim=0)      # média sobre TODOS os 1500
```

**O defeito:** `pad_or_trim` preenche com zeros até 30 s. O `.mean(dim=0)` promedia os
1500 frames do encoder — **silêncio incluído**.

**Medido na população completa** (`analysis/audio_durations.csv`, n=3 198,
`ffprobe -select_streams a:0`):

| | treino (n=2 524) | teste (n=674) |
|---|---|---|
| mediana | 18,19 s | 15,05 s |
| < 30 s (diluídos) | 66,6 % | 76,0 % |
| `valid_fraction` média | **0,655** | **0,582** |
| do vetor `[1280]` que é silêncio | 34,5 % | 41,8 % |

No treino: **26 087 s de silêncio sintético injetados**.

**Por que importa — é confound, não ruído.** A diluição é **função da duração do vídeo**.
Se vídeos sexistas tiverem duração média diferente (plausível — formatos de TikTok
diferem por tipo de conteúdo), o modelo pode acertar a Task A **lendo a duração**, não o
áudio. Confound é pior que ruído porque pode **subir** a métrica pelo motivo errado.

As features **não estão corrompidas** (0/2 524 com norma zero, norma média 14,96): estão
**diluídas**. Corrompido se detecta; diluído passa em toda checagem de sanidade.

**Correção:**
```python
enc = model.encoder(mel).squeeze(0)                            # [1500, 1280]
n_valid = max(1, min(1500, int(round(valid_seconds * 50))))    # 1500 frames / 30 s
features = enc[:n_valid].mean(dim=0)                           # [1280]
```

> **Correção 2026-08-08 (apanhada na revisão do plano 11).** A versão anterior deste
> bloco fatiava `encoder_out[:n_valid]` **sem o `.squeeze(0)`** — ou seja, fatiava a
> dimensão de **batch** (tamanho 1), não a do tempo. Com qualquer `n_valid ≥ 1` isso
> devolvia o tensor inteiro `[1, 1500, 1280]` e o `.mean(dim=0)` dava `[1500, 1280]`:
> shape errado, máscara sem efeito nenhum, e a correção do A1 seria um no-op silencioso.
> O pseudocódigo do ARCH-04 original (`01_ARCHITECTURE_FINDINGS.md`) está correto — o
> `.squeeze(0)` perdeu-se ao reescrever aqui. **Usar a versão acima.**

**⚠ Ressalva a declarar:** a self-attention do encoder **ainda viu o padding**. Mascarar
depois corrige a *agregação*, não o *encoder*. A afirmação correta é "removemos a
diluição da média", não "removemos o efeito do padding". O Whisper é treinado com esse
padding, então o encoder está desenhado para ele — mas não afirmar mais do que se fez.

---

### A2 — janelas múltiplas: parar de descartar áudio (ALTA)

**O defeito:** `pad_or_trim` também **corta**. Um vídeo de 60 s tem os primeiros 30 s
codificados e os outros 30 s **descartados sem aviso**.

**Medido:** **33,4 %** dos vídeos de treino passam de 30 s; entre eles perde-se em média
36,1 % do áudio (máx 50,5 %). Agregado: **16 104 s (24,5 %) do áudio de treino nunca
chega ao encoder**. No teste a cauda é pior: máx 135,1 s, e esse item perde 77,8 %.

**Correção:** percorrer em janelas de 30 s e promediar **ponderando pelos segundos
válidos** de cada janela.

**Por que ponderar e não `mean()` simples:** num vídeo de 50 s, a janela 1 tem 30 s e a
janela 2 tem 20 s. Com pesos `(30, 20)` o resultado é **exatamente a média sobre todos os
frames válidos do áudio inteiro** — o mesmo número que A1 daria se o Whisper não tivesse
limite. É justificação matemática, não heurística.

---

### A3 — o deslocamento treino/teste (ALTA — resolvido de graça por A1+A2)

`valid_fraction` média: treino **0,655**, teste **0,582**. O modelo é **calibrado sob um
regime de diluição e avaliado sob outro** — *dataset shift induzido pelo pipeline de
extração*, não propriedade dos dados.

**Nenhum agente reportou isto** — a amostra de 60 vídeos do ARCH-04 era só de treino.
Com a máscara, `valid_fraction` deixa de existir como variável e os dois splits passam a
ser medidos na mesma escala.

**Frase para a tese:** *"a extração de áudio introduzia um fator de diluição dependente da
duração, com médias diferentes entre treino (0,655) e teste (0,582), tornando as duas
distribuições de features não-comparáveis por construção."*

---

### A4 — zeros, proveniência e árvore v2 (idênticos a V2, V6, V7)

Os mesmos três defeitos existem em `extract_audio.ipynb`, nos mesmos moldes:
`torch.save(torch.zeros(1280), out_path)` no caminho de falha; docstring dizendo `train`
com código apontando para `test`; `if out_path.exists(): continue`.

**Aplicar as mesmas correções.** Idealmente com as funções partilhadas entre os dois
scripts (resolução de caminho, escrita do meta, manifesto de faltantes) — mas **sem
introduzir um módulo partilhado novo** se isso complicar a execução na VM; duplicação
explícita e idêntica é preferível a uma dependência frágil.

---

### A5 — ramo `else` vazio (BAIXA)

```python
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
else:
    waveform = waveform          # no-op
```
Apagar o `else`. Sem efeito funcional. Listado por completude.

---

## 3. Alterações em `video_core.py` (D2)

Sem estas, os dados v2 ficam no disco sem consumidor correto.

| # | o quê | detalhe |
|---|---|---|
| C1 | `frames_dir()` / `audio_dir()` → `v2/` | manter a v1 acessível por parâmetro, para poder comparar as duas gerações |
| C2 | `load_frames()` desnormaliza uint8 | `x.float().div(255).sub(mean).div(std)` — `mean`/`std` **lidos do `_extraction_meta.json`**, não hardcodados |
| C3 | subamostragem 64 → `N_FRAMES_USED` | constante nova; **default 16** para o primeiro re-treino (comparação com o baseline com uma variável de cada vez). Documentar que 32/64 são braços de ablação futuros |
| C4 | dataset falha alto se faltar feature (D4) | ler `_missing.json`; se algum ID do split estiver lá, `RuntimeError` no build listando os IDs |

**⚠ C2 × OPT-04:** a desnormalização passa a acontecer por amostra no `Dataset`, em CPU,
dentro dos workers do dataloader. Isso desloca trabalho do disco para a CPU. Com
`dataloader_num_workers=2` pode virar o novo gargalo. **Alternativa a considerar no
plano:** devolver uint8 do dataset e normalizar em lote na GPU, no início do `forward`.
Mais rápido, mas espalha a responsabilidade por dois arquivos. **Decidir e registar.**

**⚠ C3 — como subamostrar 64 → 16 importa.** Pegar 16 *uniformemente espaçados* dos 64
reproduz o regime atual (gap mediano 1,21 s). Pegar 16 *consecutivos* dá a janela densa
(~4,6 s). **Para o primeiro re-treino, usar os uniformemente espaçados** — é o que mantém
a comparação com o baseline honesta. A janela densa é braço de ablação.

---

## 4. O que NÃO pode mudar

A comparabilidade com o baseline congelado (`09_BASELINE_RESULTS.md`) depende disto.

- **O split congelado** (`Dataset/val_split_42.json`, 2 019/505). Nunca recalcular.
- **O redimensionamento/crop dos frames** — ver V5-NORM. Se mudar, o delta deixa de ser
  atribuível.
- **`TARGET_SR = 16000`, Whisper `large-v3`, `MCG-NJU/videomae-base`.**
- **Os arquivos originais** — `extract_video.ipynb`, `extract_audio.ipynb`,
  `Thesis_video_fixed.ipynb`, `inference_video.py`, `ablation_harness.py` (D3).
- **`runs_baseline/seed_42/`** e os artefatos do sistema submetido
  (`ablation_seed_*.csv`, `ablation_results.tex`, `task_c_thresholds_seed_*.json`,
  `predictions/`). São registro não regenerável.
- **A árvore v1** (`frames/`, `audio_features/`) até a v2 ser validada.

---

## 5. Achado em aberto — NÃO corrigir nesta ronda

**A amostragem temporal está fora do regime de pré-treino do VideoMAE.**

| n_frames | fps efetivo (mediana) | gap entre frames consecutivos |
|---|---|---|
| 16 (atual) | 0,88 | 1,21 s (p10 0,54 · p90 3,85) |
| 32 | 1,76 | 0,59 s |
| 64 | 3,52 | 0,29 s |
| pré-treino (Kinetics, stride 4 @30fps) | ~7,6 | ~0,13 s |

Duas consequências:

1. O `tubelet_size=2` pareia frames adjacentes para captar movimento curto. Com gap
   mediano de 1,21 s, **o tubelet vê duas imagens praticamente sem relação** — a
   modelação temporal é degenerada por construção.
2. **O gap varia 7,2× dentro do dataset** (p10 0,54 s → p90 3,85 s), porque os frames são
   espalhados pelo vídeo inteiro e as durações vão de 3,3 s a 60,6 s. **Mesmo padrão de
   confound do ARCH-04**, no eixo temporal. Aumentar `n_frames` divide o gap mas **não
   remove o confound** — a razão p90/p10 continua 7,2× em 16, 32 ou 64.

⚠ Os números do pré-treino vêm da config padrão do paper do VideoMAE e **não foram
confirmados em fonte primária**. A conclusão qualitativa é robusta; o fator "8,6×" precisa
ser verificado antes de ir para a tese.

**Por que não corrigir agora:** é `[MUDA]` de baseline e o ramo de vídeo ainda não foi
mostrado contribuir (o `probe_zero_video` não rodou). Gravar 64 frames **deixa a porta
aberta** sem custo — que é exatamente o ponto da D1.

---

## 6. Disco — RESOLVIDO em 2026-08-08

Era bloqueio duro (18 GB livres contra ~31 GB necessários). **`frames/` v1 foi apagado**
pelo autor, com salvaguardas aplicadas antes.

```
ANTES:   126G  102G   18G  86%
DEPOIS:  126G   73G   47G  62%
```

| item | espaço |
|---|---|
| frames v2 (64 uint8 × 3 198) | ~31 GB |
| áudio v2 | ~26 MB |
| **disponível** | **47 GB** |
| margem | ~16 GB |

**Cabe.** Duas salvaguardas foram aplicadas antes de apagar, e **as duas são
pré-requisitos da verificação da §8**:

1. **`audio_features/` v1 PRESERVADO** (26 MB). Não foi apagado de propósito: é a
   referência do **teste 6**, o único que distingue "a correção pegou" de "o script pulou
   tudo em silêncio" (norma média v1 = 14,96 ± 0,50).
2. **`frames_v1_sample/` criado** (230 MB — 20 tensores de treino + 5 de teste,
   verificados legíveis, shape `[16,3,224,224]` float32). É a referência do **teste 4**
   (V5-NORM), o item de maior risco técnico desta especificação.

### ⚠ O que foi perdido, e é irreversível

Os frames v1 **não são regeneráveis**: o script que os produziu foi editado à mão e não
existe no repositório nem no git — é literalmente o problema V6. Consequências:

- **`runs_baseline/seed_42` deixa de ser reproduzível.** Os números continuam congelados
  em `final_metrics.json` / `epoch_metrics.csv` e os fingerprints estão gravados
  (`8a60e7121b63fbb2` / `9bfb83bcc6efd303`), mas ninguém consegue rodar de novo.
  **O baseline continua válido como número; deixou de ser válido como experimento
  repetível.** Declarar assim na tese.
- `inference_video.py` sobre a v1 deixa de funcionar (já estava dessincronizado —
  `05_IMPLEMENTATION_NOTES.md` §6, item 6).

**Não apagar `frames_v1_sample/` nem `audio_features/` até a verificação da §8 passar.**

---

## 7. O que a re-extração invalida

- Todas as features em `frames/` e `audio_features/` (por isso `v2/`)
- Todo checkpoint treinado — já invalidados por outras mudanças
  (`05_IMPLEMENTATION_NOTES.md` §6)
- **O baseline `runs_baseline/seed_42/` continua válido** — é o sistema v1 completo, e é
  exatamente com ele que a comparação será feita

**O caro não é extrair (~45 min). É a invalidação.** Isso é decisão de sequenciamento,
não de orçamento.

### Atribuição — DOIS caminhos, não três

> **Corrigido 2026-08-08 (apanhado na revisão do plano 11).** A versão anterior desta
> seção listava **três** caminhos, com o caminho 1 sendo "correções do `video_core`
> sozinhas, sem re-extrair". **Esse caminho deixou de ser executável** no momento em que
> `frames/` v1 foi apagado para liberar disco (§6) — não há mais features v1 de frames
> para treinar. A §6 foi atualizada e esta não; o erro é meu. A atribuição real é a
> abaixo.

| passo | o que entra | features usadas | delta isola |
|---|---|---|---|
| **A** | correções do `video_core` (ARCH-01 etc.) **+** extração de frames (V1–V7) | frames **v2** + áudio **v1** | código **e** frames, **conflacionados** |
| **B** | extração de áudio (A1–A5) | frames **v2** + áudio **v2** | **áudio, isolado** |

**O que ainda dá para separar:** o efeito do áudio (A1–A3, a diluição por padding) fica
limpo, porque `audio_features/` v1 **foi preservado** de propósito. O passo A roda com
áudio v1; o passo B só troca o áudio.

**O ARCH-01 também é separável — mas pelo lado do modelo, não dos dados.**

`build_model(..., restore_biases: bool = True)` já é parametrizável (`video_core.py:1139`).
Rodar o mesmo sistema com `restore_biases=False` isola o efeito do ARCH-01 exatamente,
**sem extração nenhuma e sem disco adicional** — é um braço de ablação, não um controle
de dados. Custo: 1 run.

| braço | frames | áudio | `restore_biases` | isola |
|---|---|---|---|---|
| baseline (congelado) | v1 | v1 | — (código antigo) | referência |
| **A′** | v2 | v1 | **False** | correções de frames + código, **sem** ARCH-01 |
| **A** | v2 | v1 | True | **ARCH-01** (delta A − A′) |
| **B** | v2 | **v2** | True | **áudio** (delta B − A) |

Com os três braços, a decomposição fica completa: `B − baseline` é o ganho total, e
`A′−baseline`, `A−A′`, `B−A` atribuem-no a frames+código, ARCH-01 e áudio.

⚠ **A única separação genuinamente perdida** é entre *correções de frames* e *demais
correções de código do `video_core`* — as duas entram juntas em A′. Separá-las exigiria
features v1 de frames, que não existem mais nem são regeneráveis (problema V6). Reportar
A′ como *"correções de extração de frames + correções de código"*, sem proporção.

**Protocolo de épocas (regra de simetria, `09_BASELINE_RESULTS.md` §2):** rodar **uma**
calibração de 15 épocas na configuração final (B), congelar `E = argmin(eval_loss)`, e
rodar os três braços com esse mesmo E. Comparar sempre estado final contra estado final.
Custo: ~92 min de calibração + ~40 min por braço.

---

## 8. Verificação obrigatória pós-extração

| # | teste | resultado esperado |
|---|---|---|
| 1 | `_extraction_meta.json` de treino vs teste | idênticos exceto `split`, `n_items`, `date` |
| 2 | contagem de arquivos | 2 524 treino / 674 teste em cada árvore v2 |
| 3 | `_missing.json` | vazio (v1 teve 0 falhas; uma falha nova indica regressão) |
| 4 | **equivalência de normalização (V5-NORM)** | `normalize(uint8/255)` ≈ `.pt` v1, dentro de ~0,017 |
| 5 | dtype e shape | `uint8`, `[64, 3, 224, 224]` |
| 6 | **áudio: norma média v2 vs v1** | v1 era **14,96 ± 0,50**. Se v2 der praticamente o mesmo, **a correção NÃO pegou** — verificar o skip resume-safe (V7) |
| 7 | áudio: itens > 30 s | têm de ter passado por >1 janela (registar contagem) |
| 8 | varredura de zeros na v2 | 0 tensores todo-zeros (script: `scan_frames.py`) |

O teste 6 é o mais importante do lote: é o que distingue "a correção rodou" de "o script
pulou tudo em silêncio".

---

## 9. Fontes

| afirmação | onde |
|---|---|
| durações, `valid_fraction`, shift treino/teste | `analysis/audio_durations.csv`, `analysis/audio_duration_analysis.ipynb`, `improvements/07_ARCH04_FULL_POPULATION.md` |
| 0/3 198 tensores todo-zeros | `analysis/frames_zero_scan.json` |
| ARCH-04, ARCH-09 (defeitos e correções) | `improvements/01_ARCHITECTURE_FINDINGS.md` |
| OPT-01, OPT-12, custos de extração medidos | `improvements/03_OPTIMIZATION_FINDINGS.md` |
| BUG-D, BUG-E, SIMP-04/10/11 | `improvements/02_SIMPLIFICATION_FINDINGS.md` |
| baseline congelado, regra de simetria | `improvements/09_BASELINE_RESULTS.md` |
| ARCH-01 medido na VM, `num_frames` livre | `CLAUDE.md`, seção da auditoria |
| plano do áudio (versão anterior, mais didática) | `improvements/08_AUDIO_EXTRACTION_PLAN.md` |
