# 01 — Revisão de Arquitetura de IA (Agente 1)

Alvo: `Thesis_video_fixed.ipynb` (Task 3, vídeo) + `extract_video.ipynb` +
`extract_audio.ipynb` + `inference_video.py` + `ablation_harness.py`.

Referências de código: números de célula do notebook original (idênticos aos
marcadores `# ===== [cell N] CODE =====` das cópias limpas em
`.../scratchpad/video_code/`).

Todos os números de métrica citados vêm de: outputs persistidos na célula 52/55
do `Thesis_video_fixed.ipynb` (run SEED=2025), `ablation_seed_{42,7,2025}.csv`,
`task_c_thresholds_seed_*.json`. Nada foi inventado.

---

## Sumário executivo — os 5 achados mais graves

1. **ARCH-01 (CRÍTICO)** — O checkpoint do VideoMAE **não carrega os `q_bias`/`v_bias`
   pré-treinados** em `transformers==5.6.0`: o relatório de load da célula 22 lista
   `q_bias`/`v_bias` como `UNEXPECTED` e `query.bias`/`key.bias`/`value.bias` como
   `MISSING → newly initialized`. Esses biases têm norma L2 ≈ 12,9 por camada
   (vs. norma média das linhas de `query.weight` ≈ 1,37) — o encoder de vídeo roda
   com atenção pré-treinada corrompida em todas as 12 camadas.
2. **ARCH-03 (ALTO)** — A `cross_attn` atende sobre **uma única key** (`vid_proj` com
   `T=1`): `softmax` sobre 1 elemento é identicamente 1,0. Não é atenção — é um mapa
   linear fixo do vetor de vídeo somado ao texto. As 8 cabeças são decorativas, e isso
   consome ~45 % dos parâmetros treináveis do modelo.
3. **ARCH-20 (ALTO)** — **Todos os braços de ablação partilham os mesmos pesos**, treinados
   sempre na configuração bio+viewer. `B0_content_only` não mede um modelo sem biosinais —
   mede o mesmo modelo com o input bio zerado na inferência. A tabela de ablação do paper
   não sustenta a afirmação "fusão de biosinais ajuda".
4. **ARCH-04 (ALTO)** — Features de áudio são a média do encoder Whisper sobre a
   **janela inteira de 30 s, incluindo o padding**. Mediana de duração medida = 16,4 s
   → ~45 % do vetor médio da amostra mediana é silêncio, e o fator de diluição varia
   com a duração (variável de confusão embutida em todo embedding de áudio).
5. **ARCH-06 (ALTO)** — 15 épocas + SWA a partir da época 5 fazem o modelo final ser
   **pior que a época 4/6 em todas as métricas reportadas** (Task A ICMNorm hard:
   0,6199 na época 4 → 0,5647 no modelo SWA final).

Menção honrosa fora do top-5 por incerteza de ganho, mas trivial de corrigir:
**ARCH-05** — `max_length=128` trunca **42,9 % das transcrições** (medido com o
tokenizer XLM-R sobre o JSON de treino).

---

## Findings detalhados

### ARCH-01 — VideoMAE perde `q_bias`/`v_bias` pré-treinados no carregamento (CRÍTICO)

**Local:** `Thesis_video_fixed.ipynb` célula 22 (`Thesis_video_fixed_CODE.py`,
`# ===== [cell 22] CODE =====`); mesmo problema em `inference_video.py` (`run()`,
`VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")`).

```python
videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
```

**Evidência (fato verificado, output da própria célula 22 na VM):**

```
[transformers] VideoMAEModel LOAD REPORT from: MCG-NJU/videomae-base
videomae.encoder.layer.{0...11}.attention.attention.q_bias | UNEXPECTED
videomae.encoder.layer.{0...11}.attention.attention.v_bias | UNEXPECTED
encoder.layer.{0...11}.attention.attention.query.bias      | MISSING
encoder.layer.{0...11}.attention.attention.key.bias        | MISSING
encoder.layer.{0...11}.attention.attention.value.bias      | MISSING
Notes: MISSING: those params were newly initialized because missing from the checkpoint.
```

**O que está errado:** o `transformers` v5 refatorou a self-attention do VideoMAE de
`qkv` + `q_bias`/`v_bias` separados para `query`/`key`/`value` padrão com bias, e o
mapeamento de chaves para este checkpoint falhou. Resultado: os biases de query e
value **treinados** são descartados e substituídos por init novo (zeros).

**Por que importa (quantificado por mim, lendo o checkpoint do Hub):**

| tensor | norma L2 por camada (média das 12) |
|---|---|
| `encoder.*.q_bias` | **12,93** (min 10,47 / max 17,76) |
| `encoder.*.v_bias` | **1,19** |
| linhas de `query.weight` (referência de escala) | 1,37 |
| linhas de `value.weight` (referência de escala) | 1,12 |

O `q_bias` é grande em relação à projeção — zerá-lo altera materialmente o padrão de
atenção de **todas** as camadas do encoder. O ramo de vídeo está rodando sobre um
backbone parcialmente descaracterizado, e apenas LoRA r=16 em query/value (590 k
params) tem de compensar. Isto é o candidato número um para explicar por que o ramo de
vídeo contribui tão pouco.

**Correção proposta** (rodar logo após o `from_pretrained`, antes do `get_peft_model`):

```python
from huggingface_hub import hf_hub_download
import safetensors.torch as _st

videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

# transformers v5 não mapeia q_bias/v_bias -> query.bias/value.bias neste checkpoint.
_sd = _st.load_file(hf_hub_download("MCG-NJU/videomae-base", "model.safetensors"))
_restored = 0
for i, layer in enumerate(videomae.encoder.layer):
    attn = layer.attention.attention          # confirmar o caminho na VM (v5)
    qb = _sd[f"videomae.encoder.layer.{i}.attention.attention.q_bias"]
    vb = _sd[f"videomae.encoder.layer.{i}.attention.attention.v_bias"]
    attn.query.bias.data.copy_(qb.to(attn.query.bias.dtype))
    attn.value.bias.data.copy_(vb.to(attn.value.bias.dtype))
    attn.key.bias.data.zero_()                # VideoMAE não tem k_bias por design
    _restored += 1
assert _restored == len(videomae.encoder.layer)
print(f"[fix] q_bias/v_bias restaurados em {_restored} camadas")
```

**Status:** o *sintoma* é fato verificado (output da célula 22 + magnitudes lidas do
checkpoint oficial). O *caminho de atributo* (`layer.attention.attention.query.bias`)
vem do relatório de MISSING keys da própria v5, mas **precisa ser confirmado rodando na
VM** — em v4.57 (local) a classe é `VideoMAESelfAttention` com `qkv` fundido e não tem
esses atributos. Confirmar com:
`print([n for n,_ in videomae.named_parameters() if 'layer.0.attention' in n])`.

---

### ARCH-02 — `model_accepts_loss_kwargs` fica `True` sem que o modelo use `num_items_in_batch` (BAIXO — hipótese inicial de LR 16× **REFUTADA**)

**Local:** célula 28 (assinatura do `forward`) + célula 48 (`MultiTaskTrainer.compute_loss`)
+ célula 33 (`gradient_accumulation_steps=16`).

```python
# cell 28
def forward(self, frames, audio, ..., modality_dropout_prob=0.05, **kwargs):

# cell 48
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    outputs = model(**inputs)
    loss    = outputs[0]          # num_items_in_batch é aceito e IGNORADO
    ...
    return loss
```

**Mecanismo (verificado lendo o fonte do `transformers` 4.57 instalado localmente):**

```python
# Trainer.__init__
if hasattr(unwrapped_model, "accepts_loss_kwargs"): ...
else:
    forward_params = inspect.signature(unwrapped_model.forward).parameters
    self.model_accepts_loss_kwargs = any(
        k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values())

# Trainer.training_step
if not self.model_accepts_loss_kwargs or num_items_in_batch is None:
    loss = loss / self.current_gradient_accumulation_steps
```

O `**kwargs` do `forward` faz `model_accepts_loss_kwargs = True`. Suspeitei que isso
fizesse o Trainer pular a divisão por 16 (LR efetivo 16×).

**A hipótese está REFUTADA.** O segundo operando da guarda também tem de ser verdadeiro,
e não é. Fonte do 4.57, `Trainer._get_num_items_in_batch`:

```python
count_num_items_in_batch = (
    len(batch_samples) > 0
    and "labels" in batch_samples[0]        # ← exige uma chave literal "labels"
    and (self.model_accepts_loss_kwargs or self.compute_loss_func is not None)
)
```

O dataset (célula 15) devolve `labels_a`, `labels_b`, `labels_c` — **não existe chave
`labels`**. Logo `num_items_in_batch is None`, a guarda dispara e **a divisão por
`gradient_accumulation_steps` acontece normalmente**. Não há inflação de LR. Registro
isto explicitamente para que ninguém "corrija" um não-bug e para que a hipótese não seja
reinvestigada numa sessão futura.

**O que sobra (risco latente, BAIXO):** a corretude depende de um acidente — o nome das
chaves de label. Se alguém renomear qualquer campo para `labels` (por exemplo, ao
unificar o dataset com o do modelo de memes), a divisão passa a ser pulada
silenciosamente e o LR efetivo salta 16× sem nenhum aviso.

**Correção proposta (defensiva)** — a própria docstring do HF recomenda isto:

```python
class MultiTaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # forward() tem **kwargs mas NÃO usa num_items_in_batch. Hoje isto é inócuo
        # porque nenhuma chave do batch se chama "labels"; travar em False torna a
        # escala do gradiente independente do nome das chaves de label.
        self.model_accepts_loss_kwargs = False
```

**Status:** a refutação é fato verificado no fonte do 4.57 local. Como a VM roda 5.6.0,
uma linha confirma que a guarda não mudou de forma:

```python
print(trainer.model_accepts_loss_kwargs)     # True — inócuo, pois...
print("labels" in train_dataset[0])          # ...deve imprimir False
```

---

### ARCH-03 — `cross_attn` é degenerada por construção (ALTO)

**Local:** célula 28, bloco `── Fusion ──`; espelhado em `inference_video.py:400-403`.

```python
vid_proj = self.cross_proj(vid_feat).unsqueeze(1)     # [B, 1, 1024]  ← T=1
txt_q    = txt_feat.unsqueeze(1)                      # [B, 1, 1024]
cross_out, _ = self.cross_attn(query=txt_q, key=vid_proj, value=vid_proj)
txt_feat = self.cross_norm(txt_feat + cross_out.squeeze(1))
```

**O que está errado:** `vid_feat` já foi colapsado no tempo por `TemporalAttentionPool`
antes de entrar na cross-attention. Com uma única key/value, `softmax` sobre a dimensão
de sequência retorna exatamente 1,0 para todas as cabeças e todos os batches. A operação
inteira reduz-se a:

```
txt_feat ← LayerNorm( txt_feat + W_out · W_v · cross_proj(vid_feat) )
```

ou seja, um mapa **linear fixo** vídeo→texto somado ao texto. Não há seleção, não há
dependência do conteúdo da query, e as 8 cabeças + dropout de atenção não fazem nada
além de particionar um produto matricial.

Nota sobre o precedente dos memes: lá a cross-attention **colapsou** para pesos quase
uniformes por escassez de dados; aqui ela é uniforme **por construção** — mais dados não
resolveriam. Também significa que `vid_feat` entra duas vezes na fusão (direto e via
texto), consumindo capacidade da cabeça sem informação nova.

**Por que importa (contagem exata):**

| tensor | parâmetros |
|---|---|
| `cross_attn.in_proj_weight` (3×1024×1024) | 3 145 728 |
| `cross_attn.in_proj_bias` | 3 072 |
| `cross_attn.out_proj` (1024² + 1024) | 1 049 600 |
| `cross_proj` (768×1024 + 1024) | 787 456 |
| `cross_norm` | 2 048 |
| **total** | **4 987 904** |

São **≈ 45 % dos 11 017 979 parâmetros treináveis** do modelo (célula 51) gastos num mapa
linear redundante, num dataset de 2 271 amostras de treino.

**Correção proposta — escolher uma das duas:**

(a) *Fazer a atenção existir* — atender sobre os tokens temporais antes do pooling:

```python
vid_seq  = self.cross_proj(vid_out)                   # [B, T, 1024]  (T = 1568 tokens)
txt_q    = txt_feat.unsqueeze(1)                      # [B, 1, 1024]
cross_out, attn_w = self.cross_attn(query=txt_q, key=vid_seq, value=vid_seq)
txt_feat = self.cross_norm(txt_feat + cross_out.squeeze(1))
```
Requer manter `vid_out` (`[B, 1568, 768]`) vivo — caro em memória com
`per_device_train_batch_size=2` + gradient checkpointing; possivelmente reduzir os tokens
(pooling espacial por frame → `[B, 8, 768]`) antes. Também exige logar `attn_w.std()`
para detetar colapso uniforme (o risco dos memes passa a valer de verdade).

(b) *Remover* `cross_attn`/`cross_proj`/`cross_norm` e confiar na concatenação, como no
modelo de memes (que foi o melhor dos dois). Libera ~38 % dos parâmetros treináveis.
**Recomendo (b) como baseline e (a) como braço de ablação** — as predições
`predictions/cross_att/` vs `predictions/swa/` sugerem que a variante sem cross-attn já
foi submetida, então há base de comparação, mas não encontrei as métricas dessa
comparação nos arquivos deste repositório.

**Status: fato verificado** (semântica de `nn.MultiheadAttention` com `S=1`; contagem de
parâmetros calculada, não medida).

---

### ARCH-04 — Áudio: média sobre a janela de 30 s com padding, e corte em 30 s (ALTO)

**Local:** `extract_audio.ipynb` célula 0 (`extract_audio_CODE.py`,
`extract_whisper_features`).

```python
audio = whisper.pad_or_trim(waveform.numpy())            # sempre 480000 amostras
mel   = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
encoder_out = model.encoder(mel.unsqueeze(0))            # [1, 1500, 1280]
features    = encoder_out.squeeze(0).mean(dim=0)         # média sobre TODOS os 1500
```

**O que está errado:** dois problemas no mesmo `pad_or_trim`:

1. **Diluição por padding.** Medi a duração de 60 vídeos de treino com `ffprobe`:
   mediana **16,4 s**, média 25,3 s, p10 7,8 s; **61,7 % têm < 30 s** e **45 % têm < 15 s**.
   Fração média válida da janela de 30 s ≈ **0,63**. Para a amostra mediana, ~45 % do
   vetor médio é encoder de silêncio; para os 45 % mais curtos, ≥ 50 %. Pior: o fator de
   diluição **varia com a duração**, então cada embedding de áudio carrega um sinal de
   "quão curto é o vídeo" misturado ao conteúdo — variável de confusão gratuita.
2. **Truncamento.** ~38 % dos vídeos passam de 30 s e só têm o primeiro terço da fala
   representado (máx. medido 59,1 s).

**Correção proposta:**

```python
@torch.no_grad()
def extract_whisper_features(waveform: torch.Tensor) -> torch.Tensor:
    n_samples = waveform.shape[-1]
    feats, weights = [], []
    # janelas de 30 s cobrindo o áudio inteiro (resolve o truncamento)
    for start in range(0, max(n_samples, 1), 30 * TARGET_SR):
        chunk = waveform[start:start + 30 * TARGET_SR]
        valid = chunk.shape[-1] / TARGET_SR                      # segundos reais
        if valid < 0.5:
            continue
        audio = whisper.pad_or_trim(chunk.numpy())
        mel   = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(DEVICE)
        out   = model.encoder(mel.unsqueeze(0)).squeeze(0)        # [1500, 1280]
        n_valid = max(1, min(1500, int(round(valid * 50))))       # 1500 frames / 30 s
        feats.append(out[:n_valid].mean(dim=0))                   # média SÓ do válido
        weights.append(valid)
    if not feats:
        return torch.zeros(1280)
    w = torch.tensor(weights, device=feats[0].device)
    return ((torch.stack(feats) * w[:, None]).sum(0) / w.sum()).cpu().float()
```

Se quiser mudança mínima (só mata a diluição, mantém o corte em 30 s), basta a linha
`n_valid` + `out[:n_valid].mean(0)`.

**Custo:** exige **re-extrair todas as features de áudio** (2 524 treino + 674 teste) e
re-treinar. Verifiquei que os `.pt` atuais estão saudáveis (0/2 524 com norma zero,
norma média 14,96, std 0,50) — o problema não é corrupção, é conteúdo diluído.

**Status: fato verificado** (código + durações medidas por `ffprobe` em amostra de 60).
O ganho de métrica é **hipótese não verificada** — só re-treino confirma.

---

### ARCH-05 — Texto truncado em 128 tokens: 42,9 % das amostras perdem conteúdo (ALTO)

**Local:** célula 15, `EXISTVideoDataset.__init__(..., max_length: int = 128)`; usado nas
células 18 (train/val) e em `inference_video.py`.

**Medição (fato verificado, tokenizer XLM-R SentencePiece sobre
`Dataset/training/EXIST2026_training.json`, 2 524 itens):**

| percentil | tokens |
|---|---|
| mediana | 105 |
| p75 | 231 |
| p90 | 364 |
| máx | 1 122 |

Fração truncada: **42,9 % com `max_length=128`**, 21,6 % com 256, 8,4 % com 384,
3,6 % com 512 (limite do XLM-R).

**Por que importa:** o campo `text` é a transcrição + texto em tela — historicamente a
modalidade mais forte nestas tarefas. Quase metade dos itens entrega ao XLM-R menos de
metade do seu conteúdo, e o CLS é extraído desse prefixo truncado.

**Correção proposta:** `max_length=256` (cobre 78 % integralmente) ou 384 (92 %). Custo
de memória do XLM-R-large cresce ~linear na atenção com `per_device_train_batch_size=2`;
com 2 amostras/batch, 256 deve caber. Precisa ser igual no treino e em
`inference_video.py`.

**Status:** truncamento é fato medido; o ganho é **hipótese não verificada**.

---

### ARCH-06 — 15 épocas + SWA a partir da época 5 produzem um modelo final pior (ALTO)

**Local:** célula 33 (`num_train_epochs=15`, `load_best_model_at_end=False`), célula 50
(`SWACallback(swa_start_epoch=5)`), célula 26 (`SWACallback`).

**Evidência (fato verificado, tabela de treino da célula 52 + célula 55, SEED=2025):**

| métrica | melhor época | valor | modelo SWA final | Δ |
|---|---|---|---|---|
| Task A ICMNorm hard | 4 | 0,6199 | 0,5647 | **−0,055** |
| Task A ICMSoftNorm soft | 6 | 0,4814 | 0,4771 | −0,004 |
| Task B ICMNorm hard | 4 | 0,4547 | 0,4533 | −0,001 |
| Task B ICMSoftNorm soft | 6 | 0,3471 | 0,3410 | −0,006 |
| Task C ICMSoftNorm soft | 6 | 0,1350 | 0,1282 | −0,007 |
| Task C ICMNorm hard | 6 | 0,3693 | 0,3531 | −0,016 |

Val loss: 1,3616 (ép. 4) → 1,8783 (ép. 15), monotônica a partir da 5. Train loss:
1,3803 → 0,6317. O SWA média as épocas 5–15 — **exatamente a região de overfitting**.
Nenhuma métrica melhora depois da época 6.

**Por que importa:** o modelo submetido é pior que um checkpoint mais cedo em todas as
seis métricas reportadas. Este é provavelmente o maior ganho "de graça" do relatório,
sem tocar em arquitetura.

**Correção proposta — com cuidado anti-leakage explícito:**

```python
num_train_epochs = 6            # val loss mínimo na ép. 4; todas as métricas planam na 6
swa_callback = SWACallback(swa_start_epoch=3)   # média sobre 3–6, não sobre a cauda overfit
# manter load_best_model_at_end=False (conflita com SWA; ver docstring da SWACallback)
```

**Aviso importante:** *não* resolver isto ligando `load_best_model_at_end=True`. Isso é
literalmente o item 2(a) do precedente do relatório de memes (selecionar época pela mesma
métrica reportada, sobre o mesmo split), agravado aqui porque o val tem só **253
amostras**. Reduzir o número de épocas com base nesta tabela **também** é uma forma
branda de seleção pelo val — ao escrever a tese, isto deve ser declarado como escolha de
hiperparâmetro feita no dev set, não apresentado como ganho limpo.

Nota secundária: `EarlyStoppingCallback` é importado (célula 2) e nunca usado.

**Status: fato verificado.**

---

### ARCH-07 — Modality dropout interage mal com a cross-attention (ALTO)

**Local:** célula 28, blocos de dropout e de fusão.

```python
r = torch.rand(1).item()                # UM sorteio para o batch inteiro
p = modality_dropout_prob               # 0.05
if   r < p:                 drop_video = True
elif r < 2 * p:             drop_audio = True
elif r < 3 * p:             drop_text  = True
elif r < 4 * p and use_bio: drop_bio   = True
...
vid_proj = self.cross_proj(vid_feat).unsqueeze(1)     # roda SEMPRE
cross_out, _ = self.cross_attn(query=txt_q, key=vid_proj, value=vid_proj)
txt_feat = self.cross_norm(txt_feat + cross_out.squeeze(1))
```

**Problemas, em ordem de gravidade:**

1. **`drop_text` não dropa o texto.** `txt_feat` vira zeros, mas logo em seguida
   `cross_norm(0 + cross_out)` **re-preenche o slot de texto** com um vetor derivado do
   vídeo — e o LayerNorm reescala esse vetor para magnitude cheia. A cabeça recebe, no
   slot "texto", uma projeção do vídeo com escala normal. O braço que deveria ensinar
   robustez a texto ausente ensina o oposto: que o slot de texto às vezes contém vídeo.
2. **`drop_video` não zera a contribuição do vídeo ao texto.** Com `vid_feat = 0`,
   `cross_proj(0) = bias`, e um vetor constante (o bias, projetado por `W_v`/`W_out`)
   é somado ao texto. É um vazamento pequeno mas sistemático — o modelo pode aprender a
   detetar esse vetor constante como flag de "vídeo ausente".
3. **Granularidade por batch, não por amostra.** Um sorteio por forward derruba a
   modalidade das 2 amostras do micro-batch juntas. Menos grave do que parece por
   `per_device_train_batch_size=2`, mas reduz a diversidade efetiva do regularizador.
4. As probabilidades são as pretendidas (4 ramos exclusivos, 5 % cada, 20 % dos batches
   com alguma modalidade derrubada) — **isto está correto**, ao contrário do que a
   leitura rápida sugere.

**Correção proposta** (cobre 1, 2 e 3):

```python
# --- dropout por amostra, com máscaras ---
if self.training and modality_dropout_prob > 0:
    r = torch.rand(B, device=device)
    p = modality_dropout_prob
    keep_video = (r >= p).float().unsqueeze(-1)
    keep_audio = ((r < p) | (r >= 2 * p)).float().unsqueeze(-1)
    keep_text  = ((r < 2 * p) | (r >= 3 * p)).float().unsqueeze(-1)
    keep_bio   = ((r < 3 * p) | (r >= 4 * p)).float().unsqueeze(-1)
else:
    keep_video = keep_audio = keep_text = keep_bio = 1.0

vid_feat = self.video_pool(self.videomae(pixel_values=frames).last_hidden_state) * keep_video
aud_feat = self.audio_proj(audio) * keep_audio
txt_feat = self.xlmr(...).last_hidden_state[:, 0, :] * keep_text
...
# se a cross-attention for mantida (ver ARCH-03), ela tem de respeitar as máscaras:
cross_out = cross_out.squeeze(1) * keep_video * keep_text
txt_feat  = self.cross_norm(txt_feat + cross_out) * keep_text   # slot fica zerado de fato
```

Se a recomendação (b) do ARCH-03 for adotada (remover a cross-attention), os problemas
1 e 2 desaparecem sozinhos e só resta trocar a granularidade.

**Status: fato verificado** (leitura de código). O impacto em métrica é **hipótese**;
com 20 % dos batches afetados, é plausível mas não medido.

---

### ARCH-08 — Ramo bio: LayerNorm destrói a escala do RobustScaler; entradas mortas (MÉDIO)

**Local:** células 5, 9, 15, 28.

```python
# cell 28
self.bio_ln  = nn.LayerNorm(BIO_RAW_DIM)          # 16 features heterogéneas
bio_norm  = self.bio_ln(bio_centered)
bio_input = torch.cat([bio_norm, cvpc], dim=-1)   # cvpc NÃO passa pelo LN
```

**Problemas:**

1. **LayerNorm sobre 16 features heterogéneas** (diâmetro de pupila em mm, contagem de
   piscadas, durações em ns, BPM…) subtrai a média *entre features diferentes* e divide
   pelo desvio *entre features diferentes* — uma operação sem significado físico. Pior:
   desfaz exatamente o que o `RobustScaler` (ajustado no treino, por feature, entre
   amostras) construiu, e reintroduz uma normalização *por amostra* que apaga diferenças
   de magnitude absoluta entre sujeitos — que é justamente o sinal biométrico. O
   comentário do cabeçalho ("Bio: LayerNorm added after viewer-baseline subtraction")
   trata isto como fix, mas o efeito é o oposto do pretendido.
2. **`cvpc` entra depois do LN** — inconsistente, ainda que inofensivo (já está em [−1,1];
   medido: média 0,0867, std 0,3854, célula 43).
3. **`rt_norm` e `fix_norm` são entradas mortas.** São calculadas em `build_bio_vector`
   (célula 9), retornadas pelo dataset (célula 15), passadas ao `forward` (célula 28) e
   **nunca usadas**. Nota: seus valores são *brutos* (média de segundos e de contagem de
   fixações, sem escalar) — se alguém "consertar" isso conectando-as, entrarão com escala
   ~10²–10³ ao lado de features robust-scaled. Ambas já estão dentro de `bio_vec`
   (índices 11 e 6, escaladas). **Remover, não conectar.**

**Correção proposta:**

```python
# usar normalização por feature (aprendida entre amostras), não por amostra:
self.bio_ln = nn.BatchNorm1d(BIO_INPUT_DIM)   # ou simplesmente remover o LN
...
bio_input = torch.cat([bio_centered, cvpc], dim=-1)   # [B, 17]
bio_feat  = self.bio_mlp(self.bio_ln(bio_input))
```
`BatchNorm1d` com `per_device_train_batch_size=2` é ruidoso — a opção mais segura é
**remover o `bio_ln`** e confiar no `RobustScaler` (que já é a normalização certa,
per-feature, ajustada só no treino) + primeira `Linear` do MLP. Atenção: `BatchNorm`
também reativaria a necessidade de `update_bn` no SWA (célula 26, hoje comentado) —
mais um motivo para preferir remover.

**Aviso de escala:** `viewer_proj` é `Tanh`-limitado a ±1 e é subtraído de features
robust-scaled (IQR ≈ 1). Ou seja, a "baseline por viewer" só consegue deslocar ~1 IQR.
Isso é coerente com o achado histórico do `CLAUDE.md` (embeddings por viewer neutralizaram
o benefício dos biosinais), mas **não posso usar a tabela de ablação como evidência disso
neste modelo** — ver ARCH-20: os braços B1 e B2 partilham os mesmos pesos, treinados
sempre *com* viewer embeddings, então a diferença entre eles não é uma comparação de
arquiteturas treinadas.

**Status:** itens 1–3 são fatos verificados (leitura de código); o efeito em métrica do
`bio_ln` é **hipótese não verificada**.

---

### ARCH-09 — Bugs de fallback na extração de frames (MÉDIO)

**Local:** `extract_video.ipynb` célula 1 (`extract_video_CODE.py`,
`sample_frames_uniform`).

```python
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # ← aplicado TAMBÉM ao fallback
    frames.append(frame)
```

**Três defeitos:**

1. **Dupla conversão de canais no fallback.** `frames[-1]` já foi convertido para RGB; a
   linha seguinte aplica `BGR2RGB` de novo, **trocando R e B** desse frame duplicado. O
   vídeo passa a ter frames com canais invertidos no meio da sequência — entrada fora da
   distribuição do VideoMAE.
2. **Shape incompatível no fallback de zeros.** Se a *primeira* leitura falhar,
   `np.zeros((224,224,3))` é empilhado com frames em resolução nativa (ex. 1024×576) →
   `np.stack` levanta exceção → capturada pelo `except` → retorna `None` → **o vídeo
   inteiro é salvo como zeros**. Falha silenciosa que transforma uma leitura ruim numa
   amostra inteiramente vazia.
3. **`cap.set(CAP_PROP_POS_FRAMES)` por frame** é notoriamente impreciso em MP4 com taxa
   de quadros variável (o caso típico do TikTok) e depende de `CAP_PROP_FRAME_COUNT`, que
   também é estimado. Os índices amostrados podem não corresponder aos tempos esperados.

**Correção proposta:**

```python
def sample_frames_uniform(video_path: Path, n: int):
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    while True:                      # leitura sequencial: robusta a VFR
        ret, f = cap.read()
        if not ret:
            break
        frames_bgr.append(f)
    cap.release()
    if not frames_bgr:
        return None
    idxs = np.linspace(0, len(frames_bgr) - 1, n).round().astype(int)
    out  = [cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB) for i in idxs]  # 1 conversão só
    return np.stack(out)
```
(Se ler o vídeo inteiro for caro em memória, guardar só os `n` frames alvo durante a
varredura sequencial, comparando o contador com `idxs`.)

**Nota:** não consegui contar quantos vídeos foram afetados — o script imprime
`skipped` mas esse output não está no notebook salvo. Verifiquei que 0/25 tensores
amostrados são todo-zeros e que a normalização está correta (média por canal
0,277/−0,277/−0,265, range [−2,10; 2,64] — ImageNet, coerente com o processor do
VideoMAE) e que há variação temporal real (|Δ| médio entre frames = 0,89). Ou seja, o
grosso da extração está são; estes são bugs de caminho de erro.

**Status:** defeitos 1 e 2 são fatos verificados por leitura; a *prevalência* é
**hipótese não verificada** (rodar `find frames/ -name '*.pt'` + contar tensores
todo-zeros para quantificar).

---

### ARCH-10 — Formato hard da Task C permite "NO" junto com categorias de sexismo (MÉDIO)

**Local:** célula 37, `_format_for_pyevall`, ramo `task == '2_3'`; espelhado em
`inference_video.py:format_records`.

```python
hard_labels = [lbl for lbl, p in relevant.items() if p >= thresh_map[lbl]]
if not hard_labels:
    hard_labels = [max(relevant, key=relevant.get)]
```

**O que está errado:** `relevant` inclui `"NO"` (mapeado de `NONE`), e nada impede que
`"NO"` e, digamos, `"OBJECTIFICATION"` cruzem seus limiares simultaneamente. Sob a
hierarquia declarada (`TASK_2_3_HIERARCHY`: `"YES"` → 5 categorias, `"NO"` → []), emitir
os dois é uma predição autocontraditória, e o ICM hierárquico penaliza isso.

O risco é amplificado pela busca de limiares: em `task_c_thresholds_seed_7.json` o
limiar de `STEREOTYPING-DOMINANCE` é **0,05** e o de `NONE` é 0,45 — com limiares tão
baixos, coocorrência "NO" + categoria vira comum.

**Prevalência no gold** (medida): apenas 0,7 % dos itens de treino produzem gold hard com
"NO" + categoria (por causa da regra `p >= 0.5` sobre a distribuição gold). Ou seja, o
problema é essencialmente **do lado das predições**, não do gold.

**Correção proposta:**

```python
elif task == '2_3':
    ...
    if evaluation == 'hard':
        threshs = thresholds_c if thresholds_c is not None else [0.5] * len(LABELS_2_3)
        thresh_map = {label_map[lbl]: threshs[idx] for idx, lbl in enumerate(LABELS_2_3)}
        cats = [lbl for lbl, p in relevant.items()
                if lbl != "NO" and p >= thresh_map[lbl]]
        if cats:
            entry["value"] = cats                       # decidiu YES → só categorias
        elif relevant["NO"] >= thresh_map["NO"]:
            entry["value"] = ["NO"]                     # decidiu NO → só NO
        else:
            best = max((l for l in relevant if l != "NO"), key=lambda l: relevant[l])
            entry["value"] = [best]
```

Além disso, a busca de limiares deveria **excluir a classe `NONE`** do grid (o limiar de
`NONE` deixa de ter efeito com a regra acima) — reduz de 72 para 60 chamadas ao PyEvALL e
elimina um grau de liberdade que hoje só serve para criar contradições.

**⚠ Aviso de comparabilidade:** `_run_pyevall` aplica `_format_for_pyevall` **também ao
gold** (`gold = _format_for_pyevall(gold_dist, ...)`). Mudar o ramo hard da Task C muda,
portanto, os rótulos gold (as 0,7 % de linhas contraditórias desaparecem). Todos os
números de **Task C hard já reportados** — `ablation_results.csv`,
`ablation_seed_*.csv`, `ablation_results.tex`, os campos `scores` dos
`task_c_thresholds_seed_*.json` e a coluna da célula 52 — passam a ser **não
comparáveis** com os novos. Não misturar as duas gerações na mesma tabela do paper.

**Status:** o defeito é fato verificado por leitura; o ganho em ICM é **hipótese não
verificada**.

---

### ARCH-11 — Limiares da Task C são ajustados no mesmo val de 253 amostras usado para tudo (MÉDIO)

**Local:** células 35 e 57; artefatos `task_c_thresholds_seed_{42,7,2025}.json`.

O código está correto no ponto que o `CLAUDE.md` exige — a busca roda **uma vez**, após o
treino (`tune_thresholds_c=False` durante o treino), e não a cada época. Isso está bom e
não deve ser alterado.

O que **não** está resolvido: os limiares são ajustados no mesmo split de validação de
253 amostras usado para reportar todas as métricas e para o `ablation_harness`. Evidência
de sobreajuste ao ruído, comparando os três JSONs:

| classe | seed 42 | seed 7 | seed 2025 |
|---|---|---|---|
| IDEOLOGICAL-INEQUALITY | 0,35 | 0,25 | 0,55 |
| SEXUAL-VIOLENCE | 0,10 | 0,10 | 0,30 |
| STEREOTYPING-DOMINANCE | 0,20 | **0,05** | 0,25 |
| OBJECTIFICATION | 0,20 | 0,25 | 0,50 |

Variação de 5× em `STEREOTYPING-DOMINANCE` e de 3× em `SEXUAL-VIOLENCE` entre seeds. Com
253 amostras e 6 classes esparsas, alguns limiares são estimados sobre punhados de
positivos.

**Correção proposta:** ou (a) reportar o número da Task C hard **sem** ajuste de limiar
como resultado principal e o ajustado como análise, ou (b) dividir o val em duas metades
(ajustar numa, reportar na outra), ou (c) usar a média dos limiares entre seeds como
limiar fixo. Também vale reduzir o grid a valores mais grossos ({0,2; 0,3; 0,4; 0,5}) para
diminuir a variância da seleção.

**Status: fato verificado** (os três JSONs estão no repositório).

---

### ARCH-12 — A célula de validação de checkpoint lê o diretório errado e crashou (MÉDIO)

**Local:** células 53 e 54.

```python
# cell 53 — salva aqui:
safetensors.torch.save_file(state_dict, f"./best_model_seed_{SEED}_cross_att/model.safetensors")
# cell 54 — lê daqui:
state_dict_check = safetensors.torch.load_file(f"./best_model_seed_{SEED}/model.safetensors", ...)
```

**Evidência (fato verificado, traceback salvo na célula 54):**

```
RuntimeError: Error(s) in loading state_dict for EXISTVideoModel:
  Missing key(s) in state_dict: "cross_attn.in_proj_weight", "cross_attn.in_proj_bias",
  "cross_attn.out_proj.weight", "cross_attn.out_proj.bias", "cross_proj.weight",
  "cross_proj.bias", "cross_norm.weight", "cross_norm.bias".
```

Ou seja: a célula tentou validar um checkpoint **de uma versão anterior do modelo, sem
cross-attention**, e explodiu. **A validação nunca rodou nesta run**, e o
`model_fingerprint` final (célula 53) foi impresso mas não comparado com nada.

Agravantes menores: (a) a célula 17 salva `scaler.pkl` em `./best_model_seed_{SEED}/`
enquanto o modelo vai para `./best_model_seed_{SEED}_cross_att/` — artefatos do mesmo run
espalhados em dois diretórios; (b) a docstring do `inference_video.py` sugere
`--model-dir ./checkpoints_seed_2025/best`, caminho que o notebook nunca cria (o Trainer
cria `checkpoint-N`).

**O que verifiquei e NÃO é problema:** `inference_video.py` tem `--model-dir` como
argumento obrigatório (sem default) e usa `load_state_dict` estrito, então não há como
carregar silenciosamente o checkpoint velho — ele levantaria o mesmo erro. Os diretórios
`predictions/cross_att/` e `predictions/swa/` contêm dois conjuntos completos de
submissão, coerentes com duas runs distintas. **Não há evidência de predições geradas a
partir de um checkpoint errado.**

**Correção proposta:** um único `RUN_DIR = f"./best_model_seed_{SEED}_{TAG}"` definido na
célula 5 e usado em todas as células (17, 53, 54), e a validação carregando `RUN_DIR`.

**Status: fato verificado.**

---

### ARCH-20 — Todos os braços de ablação partilham os mesmos pesos (ALTO)

**Local:** `ablation_harness.py` (docstring, `ABLATION_CONDITIONS`, `run_inference`,
`run_all_ablations`, `run_ablations_multi_seed`) + célula 59 do notebook + assinatura do
`forward` (célula 28).

```python
# ablation_harness.py, docstring
"Runs the same trained model under different ablation conditions by toggling
 forward() flags and re-running compute_metrics. No retraining required —
 each condition uses the same checkpoint."

# cell 59
model   = trainer.model                      # ← o modelo treinado, uma vez só
results = run_all_ablations(model, val_dataset, compute_metrics, device="cuda")
```

**O que está errado:** `use_bio` e `use_viewer_emb` são kwargs do `forward` com default
`True`, e o Trainer **nunca** os sobrescreve durante o treino. Portanto o único modelo
treinado é sempre o `bio + viewer`. As quatro "condições" são esse mesmo modelo avaliado
com o input bio zerado (`B0`) ou com a subtração de baseline desligada (`B1`).

`B0_content_only` **não** mede um modelo content-only. Mede quanto um modelo
*dependente de biosinais* se degrada quando se zeram os biosinais na inferência — o que
tende a produzir um `B0` artificialmente ruim e, por consequência, um ganho aparente e
inflado para os braços com bio. É exatamente a classe de erro do achado 4 do relatório de
memes (braço com rótulo enganoso), e é mais consequente que o ARCH-13.

`run_ablations_multi_seed(model_fn, ...)` não resolve: ele carrega um modelo treinado por
seed e volta a alternar flags na inferência. Os três CSVs `ablation_seed_*.csv` são três
modelos treinados na configuração FULL, cada um sondado por zeragem de input.

**Por que importa:** o contributo positivo do paper ("fusão de biosinais ajuda tarefas
finas") apoia-se nesta tabela. Como está, a tabela não sustenta essa afirmação.

**Correção proposta:**

1. Braços que diferem em *treino* têm de ser **treinados separadamente**. Passar as flags
   ao modelo na construção, não no `forward`:
   ```python
   model = EXISTVideoModel(videomae, xlmr, use_bio=..., use_viewer_emb=...)
   ```
   e treinar um checkpoint por (condição × seed) — 3 condições × 3 seeds = 9 treinos de
   ~55 min cada (célula 52: 54:23 para 15 épocas; ~22 min com as 6 épocas do ARCH-06).
2. Enquanto isso não existir, **renomear as linhas do CSV e da tabela do paper** para o
   que elas realmente são: sondas de remoção de input em tempo de inferência
   (`probe_zero_bio`, `probe_no_viewer_centering`), não ablações de arquitetura.
3. Manter as flags no `forward` **apenas** para as sondas, deixando claro na docstring que
   sonda ≠ ablação.

**Status: fato verificado** (docstring explícita + defaults do `forward` + célula 59). O
*sinal* e a magnitude da diferença entre sondagem e re-treino são **hipótese não
verificada**.

---

### ARCH-13 — Os braços de ablação B2 e FULL são idênticos (MÉDIO)

**Local:** `ablation_harness.py`, `ABLATION_CONDITIONS`.

```python
("B2_bio_viewer_no_tau",  {"use_bio": True, "use_viewer_emb": True}),
("FULL_bio_viewer_tau",   {"use_bio": True, "use_viewer_emb": True}),   # ← mesmos flags
```

A rede-τ foi removida do modelo (corretamente, conforme `CLAUDE.md`), mas o braço "FULL
+ τ" continua na tabela. Consequência verificável: em **todos** os CSVs
(`ablation_results.csv`, `ablation_seed_{42,7,2025}.csv`) as linhas `B2_*` e `FULL_*` são
byte a byte idênticas. Qualquer leitor da tabela conclui erradamente que "a rede-τ não
teve efeito" — quando na verdade ela não existe. É a mesma classe de erro do achado 4 do
relatório de memes (braço com rótulo enganoso).

**Correção proposta:** remover a entrada `FULL_bio_viewer_tau` e renomear
`B2_bio_viewer_no_tau` → `B2_bio_viewer`. Atualizar `ablation_results.tex`.

**Status: fato verificado.**

---

### ARCH-14 — Métricas podem sumir silenciosamente (BAIXO)

**Local:** células 39 e 40.

```python
except Exception as e:
    print(f"  ! PyEvALL failed for task {task} ({evaluation}): {e}")
    return {}
```
e, em `_extract_metrics_from_report`, métricas com `status != 'OK'` são simplesmente
puladas. O resultado não é zero (o que o `CLAUDE.md` alerta para o modelo de memes) — é
**ausência da chave**, o que faz a coluna desaparecer da tabela do Trainer sem erro. Numa
run longa é fácil não notar que, digamos, a Task C soft parou de ser avaliada.

**Correção proposta:** retornar `{m: float("nan") for m in metric_list}` no `except` e
para métricas não-OK, e adicionar métricas proxy do sklearn (`f1_macro`, `roc_auc`) como
diagnóstico independente — que é a recomendação já registrada no `CLAUDE.md` e que este
notebook não segue (`f1_score` é importado na célula 2 e nunca usado).

**Status: fato verificado.**

---

### ARCH-15 — `warmup_ratio` NÃO é um bug nesta versão (BAIXO — correção ao brief)

**Local:** célula 33, `warmup_ratio=0.1`.

O brief compartilhado afirma que `warmup_ratio` foi **removido** em
`transformers==5.6.0` e que `warmup_steps=0.1` é a sintaxe correta. **A evidência
primária contradiz isso para a versão da VM.** Output de stderr da própria célula 33:

```
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
```

A célula executou (`execution_count=28`) e o treino rodou 1 065 steps. Portanto em 5.6.0
o argumento é **deprecado, não removido**, e continua funcionando.

**Ação para o time:** não "consertar" isto como se fosse bug — a mudança é opcional e
cosmética (migrar para `warmup_steps=0.1` silencia o warning e prepara para v5.2).
**Agente 2 (simplificador): atenção especial aqui.**

**Status: fato verificado** (a mensagem "will be removed in v5.2" com a VM em 5.6.0 é
estranha — provavelmente texto de deprecation desatualizado no upstream — mas o
comportamento observado é o que importa).

---

### ARCH-16 — Detalhes do SWA (BAIXO)

**Local:** célula 26.

1. `swa_lr` é aceito no `__init__` e **nunca usado**; nenhum `SWALR` é criado. O SWA roda
   sob o `lr_scheduler_type="cosine"`, que decai até ~0 na época 15. Média de pesos sob LR
   decrescente é quase equivalente a "pegar os pesos do fim" — perde-se boa parte do
   efeito do SWA. Ou remover o parâmetro, ou usar de facto um LR constante na fase SWA.
2. `AveragedModel` (padrão `use_buffers=False`) **não média buffers**: os buffers vêm da
   cópia feita em `on_train_begin`. Neste modelo não há BatchNorm e não identifiquei
   buffers treináveis relevantes, então o risco é baixo — mas vira um bug real se alguém
   aceitar a sugestão de `BatchNorm1d` no ARCH-08 (aí `update_bn` passa a ser
   obrigatório, e a linha está comentada).
3. `AveragedModel` duplica os 657 M parâmetros na GPU durante todo o treino. Custo real
   de memória para um ganho que, nesta configuração, é negativo (ARCH-06).

**Correção proposta:** dado o ARCH-06, considerar **remover o SWA** e treinar 6 épocas —
mais simples e demonstravelmente melhor no dado disponível. Se mantido, criar `SWALR`
e usar `swa_start_epoch=3`.

**Status:** (1) e (3) fatos verificados; (2) verificado quanto à semântica do
`AveragedModel`, mas o inventário de buffers do modelo é **hipótese não verificada**.

---

### ARCH-17 — Assimetrias e resíduos nas perdas/cabeças (BAIXO)

**Local:** células 28, 30, 32, 41.

1. **Sentinelas inconsistentes:** Task A usa `-100.0`, Tasks B e C usam `-1.0` na coluna 0.
   Funciona (proporções são sempre ≥ 0), mas é uma armadilha para quem editar. Unificar em
   `-100.0` para as três.
2. **Escala relativa das perdas:** `loss_task_c` faz `.mean(dim=-1)` sobre as 6 classes,
   enquanto `loss_task_b` **soma** sobre as 3 classes (cross-entropia soft). Com o peso
   externo de 0,7 igual para ambas, o gradiente efetivo da Task C é ~1/6 por unidade de
   erro em relação ao de B. Se a Task C é a que está pior (ICMSoftNorm 0,128), pode valer
   testar `sum(dim=-1)` em C ou peso maior. **Hipótese não verificada** — é uma mudança de
   ablação, não um bug.
3. **Código morto:** `cross_ids` é calculado em `get_optimizer` (célula 32) e nunca usado;
   `CLIPProcessor`/`CLIPVisionModel`/`whisper`/`torchaudio`/`segment`/`emoji`/`Image`/
   `f1_score`/`DataLoader`/`update_bn`/`SWALR` importados e não usados (célula 2); em
   `compute_metrics`, o ramo `if labels_a.ndim == 2: labels_a = labels_a[:, 1]` é resíduo
   do modelo de memes (aqui `labels_a` é sempre 1-D — confirmado pelo output da célula 20:
   `labels_a: torch.Size([])`).
4. **Grupos do optimizer:** `base_params` recebe `lr=1e-3` **e** `weight_decay=0.05`
   aplicado indiscriminadamente, incluindo parâmetros de `LayerNorm` e biases — prática
   que normalmente se evita. Impacto pequeno, correção trivial (excluir `bias` e
   `LayerNorm.weight` do grupo com decay).

**Status:** 1, 3 e 4 são fatos verificados; 2 é hipótese.

---

### ARCH-18 — `inference_video.py` pode ajustar o scaler no set de teste (MÉDIO)

**Local:** `inference_video.py`, `run()`.

```python
if args.scaler:
    ... pickle.load ...
else:
    # Fit scaler on the provided dataset (useful when no saved scaler exists)
    scaler = RobustScaler().fit(collect_bio_vectors(raw_for_scaler))
```

Se `--scaler` for omitido, o `RobustScaler` é ajustado nas estatísticas biométricas do
**próprio conjunto de avaliação**. Isso viola a regra anti-leakage do projeto
("seleção/normalização de features biométricas deve usar apenas estatísticas do treino") e
torna os resultados não reprodutíveis entre chamadas com subconjuntos diferentes.

**Correção proposta:** tornar `--scaler` obrigatório (`required=True`) e falhar cedo se o
arquivo não existir. O `scaler.pkl` já é salvo pela célula 17 do notebook.

**Status: fato verificado** (leitura de código). Se foi de facto usado assim nas
submissões, não consegui determinar.

---

### ARCH-19 — Determinismo: o protocolo está correto, com uma lacuna (BAIXO)

O protocolo de seeding está bem-feito e cobre o alerta do brief: `set_all_seeds` é chamado
imediatamente antes de cada init de pesos aleatórios (célula 22 VideoMAE+LoRA A, célula 24
XLM-R+LoRA A, célula 31 `EXISTVideoModel`), as flags CUDA determinísticas estão ligadas,
`seed`/`data_seed` estão no `TrainingArguments`, e `CUBLAS_WORKSPACE_CONFIG` é setado antes
de qualquer uso de CUDA. Nada a corrigir aqui.

**Lacuna:** os fingerprints não provam o que aparentam. O inicial (célula 31) é impresso e
o final (célula 53) também, mas **nada os compara com uma run anterior** — a comparação é
manual, e a célula que faria uma verificação automática (54) crashou (ARCH-12). Além
disso, o fingerprint final depende do estado do SWA (ARCH-16), então duas runs só batem
bit a bit se o SWA for determinístico também — o que é, mas convém declarar.

**Correção proposta:** persistir os fingerprints em `RUN_DIR/fingerprints.json` e comparar
automaticamente com o arquivo da run anterior, se existir.

**Status: fato verificado.**

---

## O que verifiquei e está OK (para poupar tempo dos outros agentes)

- **Features de áudio não estão corrompidas.** 0/2 524 tensores com norma zero; norma
  média 14,96 (min 9,70 / máx 22,56); shape [1280], std ≈ 0,50. A hipótese
  "`torchaudio.load` falha em MP4 e tudo virou zeros" está **descartada**.
- **Frames estão corretamente normalizados.** 0/25 amostrados são todo-zeros; médias por
  canal 0,277 / −0,277 / −0,265, range [−2,10; 2,64] (ImageNet, coerente com o processor
  do VideoMAE); |Δ| temporal médio 0,89 (há movimento real). Shape [16, 3, 224, 224] ✓,
  bate com `NUM_FRAMES=16` e com `num_frames=16`/`tubelet_size=2` do config do
  videomae-base.
- **Não há desbalanceamento de escala na fusão.** Suspeitei que `vid_feat` entrasse não
  normalizado ao lado de ramos LayerNormed. Falso: o config do `videomae-base` tem
  `use_mean_pooling=False`, logo `VideoMAEModel.layernorm` existe e **é** aplicado ao
  `last_hidden_state` (e a chave `videomae.layernorm.weight` está no checkpoint). Os
  quatro ramos entram em escala comparável. **Não mexer nisso.**
- **Dimensões batem com os comentários.** 768 + 768 + 1024 + 128 = 2 688 = `FUSION_INPUT_DIM` ✓
  (o comentário `# [B, 2688]` na linha do `fused` está no lugar errado — o `fused` é
  [B, 1024] —, mas as constantes estão certas).
- **Gradient checkpointing + LoRA está correto.** `use_reentrant=False` em ambos os
  backbones evita o problema clássico de gradiente nulo com embeddings congelados.
- **Ordem dos índices de labels é consistente.** `LABELS_2_2`/`LABELS_2_3` são definidos
  uma única vez (célula 1) e usados por `build_soft_label_*`, `_format_for_pyevall` e pelo
  `_search_thresholds_task_c` com a mesma ordem; o `inference_video.py` redefine as listas
  mas com **a mesma ordem** (conferido linha a linha). O bug histórico de listas
  duplicadas com ordens diferentes **não** se repete aqui.
- **Remapeamento de labels está conforme as regras do projeto.** `UNKNOWN` excluído,
  `'-'` → `NONE` (não filtrado) em B e C. ✓
- **Ajuste de limiar não roda durante o treino.** `tune_thresholds_c` default `False`; a
  busca só acontece na célula 57. ✓ (a ressalva do ARCH-11 é sobre o split, não sobre o
  momento).
- **Empates no gold hard da Task A são irrelevantes.** A regra `p_yes >= p_no` favorece
  YES em empates, mas apenas 0,63 % dos itens têm proporção exatamente 0,5.
- **`load_best_model_at_end=False`** está correto e coerente com o SWA — o conflito
  mencionado no brief **não** está presente.
- **`ablation_harness` roda em `model.eval()`**, então o modality dropout não contamina as
  ablações. ✓

---

## Possíveis conflitos com outros agentes

| Trecho | Meu finding | Quem mais deve querer mexer | Como reconciliar |
|---|---|---|---|
| `forward()` da célula 28 (bloco de dropout + fusão) | ARCH-03, ARCH-07 | **Agente 2** vai querer simplificar/achatar esse forward | Minha reescrita muda a *semântica* (máscaras por amostra, cross-attn removida ou sobre sequência). Aplicar a mudança semântica **primeiro**, simplificar depois. |
| `**kwargs` na assinatura do `forward` | ARCH-02 | **Agente 2** pode remover `**kwargs` "porque não é usado" | Remover `**kwargs` é seguro **apenas** se as chaves `rt_norm`/`fix_norm` saírem do dataset junto (ARCH-08.3), já que `remove_unused_columns=False` injeta tudo no forward. Coordenar. |
| `ABLATION_CONDITIONS` + flags `use_bio`/`use_viewer_emb` no `forward` | ARCH-20 | **Agente 3** (é a evidência da tabela do paper) e **Agente 2** (vai querer simplificar o `forward` removendo as flags) | As flags **não podem** simplesmente sumir: viram argumentos de construtor, para permitir treino por condição. Decisão de protocolo do Agente 3. |
| `rt_norm` / `fix_norm` (células 9, 15, 28) | ARCH-08.3 | **Agente 2** (código morto) | Concordamos: remover. Só garantir que ninguém "conserte" conectando-as. |
| `SWACallback` + `num_train_epochs` (células 26, 33, 50) | ARCH-06, ARCH-16 | **Agente 3** (protocolo de treino/seleção de época) e **Agente 2** (SWA é código complexo, candidato a remoção) | Propus reduzir épocas e possivelmente remover o SWA; é decisão de protocolo — o Agente 3 tem a palavra. Não ligar `load_best_model_at_end`. |
| `_format_for_pyevall` (célula 37) e `format_records` (inference) | ARCH-10 | **Agente 2** (duplicação óbvia entre notebook e `inference_video.py`) | A deduplicação é boa, mas a correção de semântica do ARCH-10 tem de entrar nas **duas** cópias — ou na única, se forem unificadas. |
| `_search_thresholds_task_c` / grid | ARCH-10, ARCH-11 | **Agente 3** (leakage/protocolo) | Minha proposta de excluir `NONE` do grid muda o número de chamadas ao PyEvALL (72 → 60), o que o Agente 3 pode também querer mudar por custo. |
| `ABLATION_CONDITIONS` | ARCH-13 | **Agente 3** (é a tabela do paper) | Apenas remover a linha duplicada; sem conflito real, mas o Agente 3 é o dono do artefato. |
| `extract_audio` / `extract_video` | ARCH-04, ARCH-09 | **Agente 3** (custo de re-extração de 3 198 vídeos) | Re-extração de áudio é obrigatória para o ARCH-04. Decidir junto se vale re-extrair frames (ARCH-09 afeta poucos itens). |
| `warmup_ratio` | ARCH-15 | **Agente 2** provavelmente vai "corrigir" | **Não é bug.** Migrar é opcional. |

---

## Observações fora do meu escopo

- **Dados/protocolo (Agente 3):** 73 % dos itens têm apenas **2 anotadores** (1 854 de
  2 524; os outros 670 têm 3). As distribuições soft são portanto quantizadas em
  {0; 0,5; 1} para a maioria — isto limita mecanicamente o teto de ICMSoft e deve ser
  discutido na tese ao comparar com a Task 2. Não é bug.
- **Dados (Agente 3):** o modelo de vídeo **não usa nenhuma informação demográfica dos
  anotadores** (`gender_annotators` existe no JSON e é ignorado), ao contrário do título
  do paper ("Demographic-Aware Soft-Label Learning"), que descreve o modelo de memes.
  Vale checar se a redação do paper generaliza indevidamente para a Task 3.
- **Dados (Agente 3):** o split é 90/10 estratificado só por Task A, gerando um val de
  **253 amostras** — pequeno demais para as 6 classes esparsas da Task C. Considerar
  validação cruzada em k-folds ou pelo menos estratificação conjunta A×C.
- **Custo (Agente 3):** `compute_metrics` roda ~10 avaliações PyEvALL por época (célula
  41), com I/O de JSON em disco em `TemporaryDirectory`. Em 15 épocas são ~150 execuções.
- **Simplificação (Agente 2):** `inference_video.py` reimplementa `build_bio_vector`,
  `collect_bio_vectors`, `build_soft_label_*`, `EXISTVideoDataset`, `EXISTVideoModel`,
  `TemporalAttentionPool` e a formatação PyEvALL — ~400 linhas duplicadas do notebook, com
  divergências já presentes (o `forward` da inference não tem `use_bio`/`use_viewer_emb`).
  Candidato natural a um `video_core.py`, espelhando o `memes_core.py` que já existe no
  repositório de memes.

---

## O que eu NÃO consegui verificar e por quê

1. **Se a guarda de `num_items_in_batch` tem a mesma forma em `transformers==5.6.0`**
   (ARCH-02). O mecanismo foi lido no 4.57 local e a hipótese de LR 16× foi **refutada**
   lá; a VM roda 5.6.0. Verificação: duas linhas, na VM.
2. **O caminho de atributo exato dos biases do VideoMAE em v5** (ARCH-01). O nome vem do
   relatório de MISSING keys, mas a estrutura de módulos precisa ser confirmada na VM com
   `print([n for n,_ in videomae.named_parameters() if 'layer.0.attention' in n])`.
   Também vale checar se uma versão mais nova do `transformers` já corrigiu o mapeamento —
   nesse caso, a correção é atualizar a lib em vez de remendar biases.
3. **Qualquer ganho de métrica.** Nenhuma das correções foi treinada. Todos os deltas
   propostos são hipóteses; só re-treino com as 3 seeds (42, 7, 2025) e teste pareado por
   seed confirma (e, conforme o precedente dos memes, o teste correto é delta por seed +
   consistência de sinal, não `|Δmédia| > std`).
4. **Quantos vídeos foram afetados pelos fallbacks da extração de frames** (ARCH-09). O
   output com a lista `skipped` não está salvo no notebook `extract_video.ipynb`, e
   contar tensores todo-zeros nos 3 198 arquivos `.pt` excede o tempo de execução
   disponível aqui (os arquivos estão num volume lento). Comando para rodar:
   `python -c "import torch,glob; print(sum(torch.load(f,weights_only=True).abs().sum()==0 for f in glob.glob('frames/*/*.pt')))"`.
5. **Se o `EXISTVideoModel` tem buffers relevantes** para o problema do `AveragedModel`
   (ARCH-16.2). Precisa de `print([n for n,_ in model.named_buffers()])` na VM.
6. **Se as submissões finais usaram `--scaler`** (ARCH-18). Os JSONs em `predictions/` não
   registram os argumentos de linha de comando usados.
7. **Quanto os números de ablação mudariam com re-treino por condição** (ARCH-20). Exige
   9 treinos na VM. Sei que a comparação atual é inválida como ablação de arquitetura;
   não sei se o *sinal* ("bio ajuda") sobrevive.
8. **A comparação direta cross-attn vs sem cross-attn.** Existem dois conjuntos de
   predições (`predictions/cross_att/` = `Jow_3`, `predictions/swa/` = `Jow_2`), mas não
   encontrei neste repositório as métricas de validação das duas variantes lado a lado —
   só a run com cross-attn está no notebook salvo. Isso decidiria o ARCH-03 empiricamente.
