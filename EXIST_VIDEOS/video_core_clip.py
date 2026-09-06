"""video_core_clip.py — ramo CLIP para o encoder de vídeo (item 10 do plano
`zippy-shimmying-glacier.md`).

Overlay sobre `video_core.py` (NÃO editado, só importado como `vc`): troca o encoder de
vídeo de VideoMAE-base+LoRA por CLIP ViT-Large/14 com as 4 últimas camadas destravadas
full-rank, sem LoRA — mesmo padrão exato de `EXIST_MEMES/memes_core.py:488-494`. É um
braço de ablação/atribuição isolado (ver doc `improvements/18_CLIP_VIDEO_BRANCH_SPEC.md`),
não um substituto do baseline: incomparável por state_dict com as runs VideoMAE. Reportado
no estado FINAL da época 15 (não `E=argmin(eval_loss)` — ver constante
`UNFROZEN_VISION_LAYERS` para o histórico de por que isso foi decidido assim).

Decisão arquitetural (wrapper "videomae-like", ver plano §"Decisão arquitetural
central"): em vez de duplicar o `forward` de `EXISTVideoModel` — o maior anti-padrão já
documentado no projeto —, `CLIPFrameEncoder` se apresenta com a MESMA interface que o
forward existente já consome (`forward(pixel_values) -> objeto com .last_hidden_state`,
`gradient_checkpointing_enable(**kwargs)`). `EXISTVideoModelCLIP` herda de
`vc.EXISTVideoModel` e só substitui `video_pool` (dim 1024) e `fusion` (dim nova); o
`forward` inteiro (modality dropout, `_mask`, texto, áudio, bio, fusão, heads,
`compute_loss`) é herdado sem reescrita.

Wart aceita e documentada: como `EXISTVideoModel.__init__` atribui
`self.videomae = video_encoder`, as chaves do state_dict deste braço nascem como
`videomae.clip.*` mesmo não havendo VideoMAE nenhum aqui. Mitigado por nomear o submódulo
interno `self.clip` (grepável) e por `meta.json["video_encoder"]` como desambiguador
autoritativo — NUNCA inferir o tipo de encoder pelo nome das chaves do checkpoint.

⚠ VERIFICAÇÃO LOCAL vs VM (não confundir as duas):
  - `transformers` LOCAL (onde este arquivo foi escrito): 4.57.6.
  - `transformers` da VM (onde isto de fato roda, GPU): 5.6.0.
  Só foi possível verificar aqui, localmente, SEM GPU e sem as libs reais instaladas:
  sintaxe (`ast.parse`), aritmética pura (`fusion_input_dim_clip` → 2944/2816) e
  inspeção estática (nenhuma constante de normalização do CLIP hardcoded, nenhum
  `vc.VIDEO_DIM` sobrescrito, `restore_biases`/`videomae_qv_biases` ausentes do caminho
  CLIP, todo símbolo `vc.X` referenciado existe de fato em `video_core.py`). NÃO foi
  possível verificar: o caminho de atributo real de `CLIPVisionModel` na v5.6.0
  (`clip_vision.encoder.layers`, copiado literalmente de `memes_core.py:488-494`), a API
  de `CLIPProcessor.image_processor.image_mean/.image_std` nessa versão, nem
  `gradient_checkpointing_enable` do `CLIPVisionModel` na v5.6.0. Isso é exatamente o que
  os portões da seção "Verificação — LOCAL vs VM" do plano cobrem: `python
  video_core_clip.py` na VM, depois `selfcheck_renorm_equivalence`, depois
  `smoke_test_clip` — cada um bloqueante para o próximo, antes de qualquer treino.
"""

import json
import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn

import video_core as vc

# ============================================================================
# Constantes CONGELADAS (plano §"Assinaturas públicas CONGELADAS")
# ============================================================================

CLIP_NAME = "openai/clip-vit-large-patch14"
CLIP_VIDEO_DIM = 1024          # hidden size do CLIP ViT-L/14 — NÃO sobrescrever vc.VIDEO_DIM
                                # (768, é do VideoMAE; os dois módulos coexistem, não fundem)
UNFROZEN_VISION_LAYERS = 4     # mesmo nº do modelo de memes (ABL-4 lá, baseline 4).
                                # ⚠ Testado 4→2 numa 2ª run (hipótese: desequilíbrio de
                                # capacidade do encoder explicaria o vídeo dominando a
                                # fusão) — REFUTADA pelos dados: ‖Δ_texto‖ da sonda
                                # causal PIOROU (3,14→1,05) e ‖Δ_vídeo‖ SUBIU
                                # (17,3→18,4) com menos capacidade, direção oposta à
                                # prevista. Mecanismo real: `video_pool`/`fusion` são
                                # sempre full-rank e não dependem desta constante — a
                                # dominância é decidida na fusão pela qualidade das
                                # features do CLIP (fortes mesmo congeladas), não pela
                                # contagem de params do encoder. Revertido para 4 —
                                # braço oficial reportado no estado FINAL da época 15
                                # (não E=argmin), ver `improvements/16_VIDEO_INVESTIGATION_RESULTS.md`
                                # e `improvements/18_CLIP_VIDEO_BRANCH_SPEC.md`.
CLIP_LR = 1e-5                 # precedente memes_model.ipynb: LR único 1e-5 p/ o CLIP inteiro

LOSS_W_C_REWEIGHTED = 4.2      # = vc.LOSS_W_C (0.7) * 6, opção 5 do doc 20.


def apply_loss_reweight_c(w: float = LOSS_W_C_REWEIGHTED) -> dict:
    """Repesa a Task C na loss conjunta, mutando `vc.LOSS_W_C` em runtime.

    Por que assim: `vc.compute_loss` (video_core.py:1358) resolve `LOSS_W_C` no
    namespace do módulo A CADA CHAMADA, e o `forward` do modelo chama a função
    de módulo (video_core.py:1156). Logo, mutar o global surte efeito sem editar
    `video_core.py` (regra forte do projeto) e sem duplicar a loss.

    Por que 4.2 e não um valor de busca: `loss_task_c` reduz as 6 classes com
    `.mean(dim=-1)` enquanto `loss_task_b` usa `.sum(dim=-1)`, deixando o peso
    EFETIVO de C em `0.7 * 1/6`. `0.7 * 6 = 4.2` desfaz exatamente essa
    assimetria aritmética — é equivalente a trocar a redução para `.sum`, não é
    hiperparâmetro afinado em validação (isso seria leakage).

    ⚠ NUNCA chamar no import do módulo: as sondas (`probe_video_contribution_clip`,
    `probe_head_training`) e `load_run_clip` também importam este overlay e
    passariam a computar loss repesada silenciosamente. Chamar de uma célula
    explícita do notebook de treino, e verificar com `assert vc.LOSS_W_C == 4.2`
    imediatamente antes de `trainer.train()`.

    ⚠ A eval_loss desta run NÃO é comparável com a das runs anteriores (o termo de
    C entra 6× maior). Comparar só as 6 métricas do PyEvALL.

    Devolve o dict dos três pesos, para gravar em `meta.json`.
    """
    vc.LOSS_W_C = float(w)
    weights = {"loss_w_a": vc.LOSS_W_A, "loss_w_b": vc.LOSS_W_B, "loss_w_c": vc.LOSS_W_C}
    print(f"[loss reweight] LOSS_W_A={weights['loss_w_a']}  "
          f"LOSS_W_B={weights['loss_w_b']}  LOSS_W_C={weights['loss_w_c']} "
          f"(default era 0.7 — opção 5 do doc 20)")
    return weights


FREEZABLE_HEADS = ("head_a", "head_b", "head_c")


def set_heads_trainable(model, frozen: tuple = (), verbose: bool = True) -> dict:
    """Liga/desliga `requires_grad` das heads. `frozen` lista as heads a CONGELAR;
    todas as outras são explicitamente destravadas (a função é idempotente e absoluta,
    não incremental — chamar com `frozen=()` restaura o estado padrão).

    ⚠ **Congelar uma head NÃO corta o gradiente do trunk.** `requires_grad=False` impede
    que os PESOS da head sejam atualizados, mas o gradiente da loss continua fluindo
    através dela até `fusion` e o resto do trunk. É exatamente esse o mecanismo
    pretendido no braço FREEZE_AB: as heads A/B viram probes FIXAS cuja loss ancora o
    trunk perto de representações que elas ainda conseguem ler, enquanto head C e o trunk
    adaptam. Por isso as losses de A e B **continuam ativas** (pesos originais 1,0/0,7) —
    remover as losses A/B é o que de fato soltaria o trunk. Não "otimizar" isso.

    ⚠ **Antes de salvar, restaurar com `set_heads_trainable(model)`.**
    `vc.trainable_state_dict` (video_core.py:1908) filtra por `requires_grad`: uma head
    congelada NÃO entra no `trainable.safetensors`, e `load_run_clip` também não acusa
    (o check `not_loaded` de lá só percorre params treináveis) — o checkpoint recarregaria
    com essa head em init ALEATÓRIO, sem erro nenhum. A célula de validação do notebook
    (`torch.allclose(loaded, fresh)`) é a rede de segurança contra isso.

    Devolve dict {head: n_params_treináveis} para gravar no `meta.json`.
    """
    for h in frozen:
        assert h in FREEZABLE_HEADS, f"set_heads_trainable: head desconhecida {h!r}"

    report = {}
    for h in FREEZABLE_HEADS:
        module = getattr(model, h)
        trainable = h not in frozen
        for p in module.parameters():
            p.requires_grad_(trainable)
        report[h] = sum(p.numel() for p in module.parameters() if p.requires_grad)

    if verbose:
        print(f"[heads] congeladas={list(frozen) or 'nenhuma'} | "
              + " ".join(f"{h}:{n:,}" for h, n in report.items())
              + "  (params treináveis por head)")
    return report


def assert_heads_frozen(model, frozen: tuple) -> None:
    """Portão duro: confirma que exatamente as heads de `frozen` estão congeladas.
    Chamar imediatamente antes de `trainer.train()` — protege contra execução fora de
    ordem do notebook (a célula de freeze não ter rodado, ou o modelo ter sido
    reconstruído depois dela)."""
    for h in FREEZABLE_HEADS:
        grads = {p.requires_grad for p in getattr(model, h).parameters()}
        assert len(grads) == 1, f"assert_heads_frozen: {h} tem requires_grad misto {grads}"
        expected = h not in frozen
        assert grads.pop() is expected, (
            f"assert_heads_frozen: {h}.requires_grad != {expected} — estado de freeze "
            f"não bate com o pré-registrado ({list(frozen) or 'nenhuma congelada'}). "
            f"Rodar a célula de freeze ANTES de treinar.")
    print(f"✓ freeze confere: congeladas={list(frozen) or 'nenhuma'}")


def fusion_input_dim_clip(use_bio: bool) -> int:
    """Largura da entrada da fusão no braço CLIP — mesma lógica de `vc.fusion_input_dim`,
    mas com CLIP_VIDEO_DIM (1024) no lugar de vc.VIDEO_DIM (768):

        1024 (CLIP) + 768 (áudio) + 1024 (texto) = 2816  (B0, sem bio)
        2816 + 128 (bio)                          = 2944  (B1/B2, com bio)

    Mesma regra do original (04_RECONCILIATION §1.1): zeros de um braço sem bio são
    algebricamente inertes para uma `Linear` sem LayerNorm depois do `cat` — a largura
    variável é por auditabilidade, blindagem e contagem de params honesta, não porque os
    zeros distorceriam algo.
    """
    d = CLIP_VIDEO_DIM + vc.AUDIO_PROJ_DIM + vc.TEXT_DIM   # 1024 + 768 + 1024 = 2816
    if use_bio:
        d += vc.BIO_HIDDEN_DIM                             # + 128            = 2944
    return d


def clip_norm_constants(model_name: str = CLIP_NAME):
    """(mean, std) [3,1,1] float32 lidos do `CLIPProcessor` em RUNTIME — regra C2 do
    projeto: nunca hardcodar constantes de normalização. O CLIP-L/14 usa mean/std
    diferentes do ImageNet/VideoMAE (~0.481/0.268 vs ~0.485/0.229 na literatura pública),
    mas o valor de verdade usado aqui é sempre o que o processor devolver nesta VM, nesta
    versão do `transformers` — nunca um número copiado de memória.
    """
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    image_processor = getattr(processor, "image_processor", processor)
    mean = torch.tensor(image_processor.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=torch.float32).view(3, 1, 1)
    return mean, std


# ============================================================================
# Dataset — só a normalização difere
# ============================================================================

class EXISTVideoDatasetCLIP(vc.EXISTVideoDataset):
    """Só a normalização difere de `vc.EXISTVideoDataset`. `load_frames` (chamado dentro
    de `__getitem__` herdado) já devolve frames normalizados em mean/std do
    VideoMAE/ImageNet — em vez de reler o disco com uma normalização diferente,
    remapeia por uma afim composta, algebricamente exata:

        x_clip = x_vmae * (std_v/std_c) + (mean_v - mean_c)/std_c

    Verificado nesta sessão (aritmeticamente, sobre valores representativos): erro
    ~1e-7 em float32 contra normalizar diretamente do uint8 bruto. Ver
    `selfcheck_renorm_equivalence` para a verificação empírica contra frames reais em
    disco (roda na VM).
    """

    def __init__(self, dataframe, tokenizer, scaler, frames_path, audio_path,
                 max_length=vc.MAX_LENGTH, has_labels: bool = True,
                 clip_name: str = CLIP_NAME):
        super().__init__(dataframe, tokenizer, scaler, frames_path, audio_path,
                          max_length=max_length, has_labels=has_labels)
        self.clip_name = clip_name

        # Constantes fixas por instância — computadas uma vez, reaplicadas por item.
        vmae_mean, vmae_std = vc._frames_norm_constants(frames_path)
        clip_mean, clip_std = clip_norm_constants(clip_name)
        self._clip_scale = vmae_std / clip_std                       # [3,1,1]
        self._clip_shift = (vmae_mean - clip_mean) / clip_std        # [3,1,1]

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        # Broadcast [3,1,1] sobre [T,3,H,W] — mesma afim para todos os frames do vídeo.
        item["frames"] = item["frames"] * self._clip_scale + self._clip_shift
        return item


def selfcheck_renorm_equivalence(dataset_clip, item_ids=None, atol: float = 1e-5) -> None:
    """Roda NA VM, sobre frames reais em disco: recomputa a normalização CLIP
    independentemente (direto do `.pt` bruto em disco, sem passar pelo caminho afim de
    `EXISTVideoDatasetCLIP`) e compara com o que o dataset de fato devolve.

    `item_ids`: subconjunto de `id_EXIST` a checar; `None` usa os 5 primeiros itens do
    dataset. Levanta AssertionError se qualquer item divergir mais que `atol`.
    """
    all_ids = [str(i) for i in dataset_clip.item_ids]
    if item_ids is None:
        positions = list(range(min(5, len(all_ids))))
    else:
        wanted = {str(i) for i in item_ids}
        positions = [i for i, iid in enumerate(all_ids) if iid in wanted]
        if not positions:
            raise RuntimeError(
                f"selfcheck_renorm_equivalence: nenhum dos item_ids pedidos foi "
                f"encontrado no dataset (pedidos: {sorted(wanted)[:5]}...).")

    clip_mean, clip_std = clip_norm_constants(dataset_clip.clip_name)
    vmae_mean, vmae_std = vc._frames_norm_constants(dataset_clip.frames_path)

    max_seen = 0.0
    for pos in positions:
        item_id = all_ids[pos]
        raw_path = Path(dataset_clip.frames_path) / f"{item_id}.pt"
        raw = torch.load(raw_path, weights_only=True)
        raw = vc._subsample_uniform(raw, vc.N_FRAMES_USED)

        if raw.dtype == torch.uint8:
            # v2: uint8 bruto, sem normalização nenhuma no disco.
            direct = raw.float().div(255.0).sub(clip_mean).div(clip_std)
        else:
            # v1: float32 já normalizado p/ VideoMAE/ImageNet — desfaz e reaplica p/ CLIP.
            direct = raw.float().mul(vmae_std).add(vmae_mean).sub(clip_mean).div(clip_std)

        via_affine = dataset_clip[pos]["frames"]
        diff = (direct - via_affine).abs().max().item()
        max_seen = max(max_seen, diff)
        assert diff < atol, (
            f"selfcheck_renorm_equivalence: item {item_id} diverge {diff:.2e} "
            f"(atol={atol}) entre normalização direta do disco e o caminho afim de "
            f"EXISTVideoDatasetCLIP — a fórmula T3 pode estar errada, ou os meta.json "
            f"de mean/std divergiram entre extrações.")

    print(f"✓ selfcheck_renorm_equivalence: {len(positions)} itens, "
          f"max|diff|={max_seen:.2e} < atol={atol}")


# ============================================================================
# Encoder de vídeo via CLIP — wrapper "videomae-like"
# ============================================================================

class _CLIPFrameOutput:
    """Objeto mínimo com `.last_hidden_state`, para casar com o que
    `EXISTVideoModel.forward` já consome (`self.videomae(pixel_values=frames)
    .last_hidden_state`) sem precisar reescrever o forward."""
    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class CLIPFrameEncoder(nn.Module):
    """Adapta um `CLIPVisionModel` para a interface que o `forward` de
    `EXISTVideoModel` espera de `self.videomae`:

        forward(pixel_values: [B, 16, 3, 224, 224]) -> objeto com
            .last_hidden_state: [B, 16, 1024]

    Internamente: `view(B*16, 3, 224, 224)` -> `CLIPVisionModel` ->
    `last_hidden_state[:, 0, :]` (CLS PRÉ-post_layernorm — cópia literal de
    `memes_core.py:573`; NÃO usar `pooler_output`, que aplica post_layernorm) ->
    `view(B, 16, 1024)`.

    `frame_chunk` (R1, contingência de OOM): processa os `B*16` frames em fatias de
    `frame_chunk` em vez de tudo de uma vez. Numericamente exato — CLIP usa LayerNorm,
    não BatchNorm, então processar em fatias não muda o resultado, só o pico de memória.
    """

    def __init__(self, clip_vision, frame_chunk: int | None = None):
        super().__init__()
        self.clip = clip_vision           # nome deliberado — grepável, ver wart no topo
        self.frame_chunk = frame_chunk

    def gradient_checkpointing_enable(self, **kwargs):
        self.clip.gradient_checkpointing_enable(**kwargs)

    def _encode_chunk(self, flat_pixels: torch.Tensor) -> torch.Tensor:
        out = self.clip(pixel_values=flat_pixels)
        return out.last_hidden_state[:, 0, :]     # CLS pré-post_layernorm

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> _CLIPFrameOutput:
        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * T, C, H, W)

        if self.frame_chunk is None:
            cls = self._encode_chunk(flat)
        else:
            pieces = []
            for start in range(0, flat.shape[0], self.frame_chunk):
                pieces.append(self._encode_chunk(flat[start:start + self.frame_chunk]))
            cls = torch.cat(pieces, dim=0)

        return _CLIPFrameOutput(cls.view(B, T, -1))


def build_clip_vision(init_seed: int = vc.INIT_SEED,
                       unfrozen_layers: int = UNFROZEN_VISION_LAYERS):
    """Congela tudo, depois destrava as últimas `unfrozen_layers` camadas do encoder do
    CLIP, full-rank, sem LoRA. Caminho de atributo (`clip_vision.encoder.layers`)
    COPIADO LITERALMENTE de `EXIST_MEMES/memes_core.py:488-494` — não re-derivado de
    memória; se a versão do `transformers` da VM tiver mudado a estrutura de módulos do
    `CLIPVisionModel`, isto falha com `AttributeError` na hora (R3), não em silêncio.
    """
    from transformers import CLIPVisionModel

    vc.set_all_seeds(init_seed)
    clip_vision = CLIPVisionModel.from_pretrained(CLIP_NAME)

    for param in clip_vision.parameters():
        param.requires_grad = False
    if unfrozen_layers > 0:
        for layer in clip_vision.encoder.layers[-unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    # ~12,596,224 params/camada, medido: 50.384.896 params p/ 4 camadas destravadas
    # (log da 1ª run, build_clip_vision). Bounds proporcionais a unfrozen_layers, não
    # fixos em [40M,60M] — isso era específico de 4 camadas e quebraria silenciosamente
    # (ou barraria sem motivo) qualquer outra contagem, inclusive as 2 atuais (~25M).
    n_train = sum(p.numel() for p in clip_vision.parameters() if p.requires_grad)
    per_layer_estimate = 12_596_224
    expected = unfrozen_layers * per_layer_estimate
    lo, hi = 0.7 * expected, 1.3 * expected
    assert lo < n_train < hi, (
        f"build_clip_vision: {n_train:,} params treináveis no CLIP, fora do intervalo "
        f"esperado [{lo:,.0f}, {hi:,.0f}] para {unfrozen_layers} camadas destravadas do "
        f"ViT-L/14 — R3: o caminho de atributo pode ter mudado nesta versão do "
        f"transformers. Depurar com: print([n for n, p in clip_vision.named_parameters() "
        f"if p.requires_grad][:5])")

    print(f"[build_clip_vision] {unfrozen_layers} últimas camadas destravadas, "
          f"{n_train:,} params treináveis")
    return clip_vision


def build_backbones_clip(init_seed: int = vc.INIT_SEED,
                          unfrozen_layers: int = UNFROZEN_VISION_LAYERS,
                          frame_chunk: int | None = None):
    """Constrói os dois backbones do braço CLIP: `(CLIPFrameEncoder, xlmr com LoRA)`.

    Mesmo protocolo de re-seed de `vc.build_backbones`: a matriz LoRA A do XLM-R é
    aleatória, então o init não pode depender de quanto RNG o CLIP consumiu antes.
    """
    from transformers import XLMRobertaModel
    from peft import get_peft_model

    clip_vision = build_clip_vision(init_seed, unfrozen_layers=unfrozen_layers)
    clip_encoder = CLIPFrameEncoder(clip_vision, frame_chunk=frame_chunk)

    vc.set_all_seeds(init_seed)
    xlmr = XLMRobertaModel.from_pretrained(vc.XLMR_NAME)
    xlmr = get_peft_model(xlmr, vc.make_lora_config())

    return clip_encoder, xlmr


# ============================================================================
# Modelo
# ============================================================================

class EXISTVideoModelCLIP(vc.EXISTVideoModel):
    """Herda `vc.EXISTVideoModel` inteiro (modality dropout, `_mask`, texto, áudio,
    bio, fusão, heads, `compute_loss`, `gradient_checkpointing_enable`) e só substitui
    `video_pool` (dim CLIP_VIDEO_DIM=1024, era 768) e `fusion`/`fusion_in`
    (fusion_input_dim_clip, era vc.fusion_input_dim). Os heads NÃO mudam de dimensão
    (a entrada deles é sempre FUSION_HIDDEN_DIM) e por isso não são recriados.

    `vc.set_all_seeds(init_seed)` é re-chamado ANTES da substituição: os módulos
    descartados (video_pool/fusion de 768 do `super().__init__`) consumiram RNG; sem
    re-seed o init dos módulos novos não seria determinístico.
    """

    def __init__(self, clip_encoder, xlmr, *, use_bio: bool = True,
                 use_viewer_emb: bool = True, init_seed: int = vc.INIT_SEED):
        super().__init__(clip_encoder, xlmr, use_bio=use_bio, use_viewer_emb=use_viewer_emb)

        vc.set_all_seeds(init_seed)
        self.video_pool = vc.TemporalAttentionPool(CLIP_VIDEO_DIM)
        self.fusion_in = fusion_input_dim_clip(use_bio)
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_in, vc.FUSION_HIDDEN_DIM), nn.GELU(),
            nn.Dropout(vc.FUSION_DROPOUT),
        )

    @property
    def clip_vision(self):
        """Desambiguador legível: `self.videomae` é o `CLIPFrameEncoder` (wart do
        __init__ herdado, ver docstring do módulo); `self.videomae.clip` é o
        `CLIPVisionModel` de fato."""
        return self.videomae.clip


def build_model_clip(init_seed: int = vc.INIT_SEED, *, use_bio: bool = True,
                      use_viewer_emb: bool = True,
                      unfrozen_layers: int = UNFROZEN_VISION_LAYERS,
                      frame_chunk: int | None = None) -> EXISTVideoModelCLIP:
    """Caminho único de construção do braço CLIP: backbones + `EXISTVideoModelCLIP` na
    condição pedida — espelha `vc.build_model`, sem `restore_biases` (não existe
    ARCH-01 aqui: não há `q_bias`/`v_bias` fundido no CLIP)."""
    clip_encoder, xlmr = build_backbones_clip(
        init_seed, unfrozen_layers=unfrozen_layers, frame_chunk=frame_chunk)
    vc.set_all_seeds(init_seed)   # re-seed antes do init de EXISTVideoModelCLIP
    return EXISTVideoModelCLIP(clip_encoder, xlmr, use_bio=use_bio,
                                use_viewer_emb=use_viewer_emb, init_seed=init_seed)


# ============================================================================
# Otimizador
# ============================================================================

def get_optimizer_clip(model, clip_lr: float = CLIP_LR):
    """AdamW com grupos por papel — mesmo esqueleto de `vc.get_optimizer`, com um grupo
    NOVO dedicado ao CLIP.

    R2 (CRÍTICO): o grupo do CLIP (`name.startswith("videomae.")` — é assim que as
    chaves nascem, ver wart no topo do módulo) é verificado PRIMEIRO no predicado, ANTES
    de cair em `base_decay` (lr=1e-3, video_core.py:1889-1895). Se um parâmetro do ViT-L
    pré-treinado caísse em `base_decay` por engano, o LR 100x maior destruiria os pesos
    pré-treinados SEM NENHUM ERRO — só métricas ruins, indistinguível de o CLIP
    simplesmente não ajudar.

    Dois asserts obrigatórios: (1) o grupo do CLIP tem a contagem esperada de params p/
    `UNFROZEN_VISION_LAYERS` (mesmo portão proporcional de `build_clip_vision`,
    reaplicado aqui porque um bug de nomeação poderia fazer os params do CLIP caírem em
    outro grupo mesmo com `build_clip_vision` correto); (2) partição exata — cada
    parâmetro treinável cai em exatamente 1 grupo.
    """
    def _no_decay(name: str) -> bool:
        return name.endswith(".bias") or "LayerNorm.weight" in name or ".norm." in name

    clip_group, lora, viewer, base_decay, base_no_decay = [], [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("videomae."):        # grupo do CLIP — CHECADO PRIMEIRO (R2)
            clip_group.append(param)
        elif "lora" in name:
            lora.append(param)
        elif "viewer_emb" in name:
            viewer.append(param)
        elif _no_decay(name):
            base_no_decay.append(param)
        else:
            base_decay.append(param)

    # Mesma estimativa proporcional de build_clip_vision (~12,596,224 params/camada).
    n_clip = sum(p.numel() for p in clip_group)
    per_layer_estimate = 12_596_224
    expected = UNFROZEN_VISION_LAYERS * per_layer_estimate
    lo, hi = 0.7 * expected, 1.3 * expected
    assert lo < n_clip < hi, (
        f"get_optimizer_clip: grupo do CLIP tem {n_clip:,} params, fora de "
        f"[{lo:,.0f}, {hi:,.0f}] esperado p/ UNFROZEN_VISION_LAYERS={UNFROZEN_VISION_LAYERS} "
        f"(R2) — confirmar unfrozen_layers e o prefixo 'videomae.' nas chaves antes de "
        f"treinar; se isto passar sem erro o CLIP pré-treinado pode estar recebendo "
        f"lr=1e-3 de base_decay em vez de clip_lr.")

    n_grouped = len(clip_group) + len(lora) + len(viewer) + len(base_decay) + len(base_no_decay)
    n_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    assert n_grouped == n_trainable, (
        f"get_optimizer_clip: partição incompleta — {n_grouped} params nos grupos vs "
        f"{n_trainable} parâmetros treináveis no modelo. Algum parâmetro treinável caiu "
        f"fora de todos os grupos (ou os predicados se sobrepõem).")

    return torch.optim.AdamW([
        {"params": clip_group,     "lr": clip_lr, "weight_decay": 0.01},
        {"params": lora,           "lr": 1e-4,    "weight_decay": 0.01},
        {"params": viewer,         "lr": 1e-3,    "weight_decay": 0.0},
        {"params": base_decay,     "lr": 1e-3,    "weight_decay": 0.05},
        {"params": base_no_decay,  "lr": 1e-3,    "weight_decay": 0.0},
    ])


# ============================================================================
# Checkpoint enxuto
# ============================================================================

def save_run_clip(model, run_path, scaler=None, tokenizer=None, meta: dict = None):
    """Espelha `vc.save_run`, com `meta.json` do braço CLIP: `video_encoder`,
    `unfrozen_vision_layers`, `video_hidden_dim`, `n_frames`, `frame_sampling`,
    `clip_image_mean`/`_std`, `clip_lr` — SEM `videomae_qv_biases`/`restore_biases`
    (não existe restauração de bias aqui; não há ARCH-01 no CLIP)."""
    import safetensors.torch as st

    run_path = Path(run_path)
    run_path.mkdir(parents=True, exist_ok=True)

    st.save_file(vc.trainable_state_dict(model), str(run_path / "trainable.safetensors"))
    if scaler is not None:
        with open(run_path / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(run_path))

    clip_mean, clip_std = clip_norm_constants(CLIP_NAME)
    full_meta = {
        "use_bio": bool(model.use_bio),
        "use_viewer_emb": bool(model.use_viewer_emb),
        "fusion_in": int(model.fusion_in),
        "max_length": vc.MAX_LENGTH,
        "test_case": vc.TEST_CASE,
        "split_seed": vc.SPLIT_SEED,
        "video_encoder": "clip",
        "unfrozen_vision_layers": UNFROZEN_VISION_LAYERS,
        "video_hidden_dim": CLIP_VIDEO_DIM,
        "n_frames": vc.N_FRAMES_USED,
        "frame_sampling": "uniform_subsample_from_64_stored",
        "clip_image_mean": clip_mean.flatten().tolist(),
        "clip_image_std": clip_std.flatten().tolist(),
        "clip_lr": CLIP_LR,
    }
    full_meta.update(meta or {})
    with open(run_path / "meta.json", "w") as f:
        json.dump(full_meta, f, indent=2)

    print(f"[save_run_clip] gravado em {run_path} (video_encoder=clip, "
          f"use_bio={full_meta['use_bio']}, use_viewer_emb={full_meta['use_viewer_emb']}, "
          f"fusion_in={full_meta['fusion_in']})")
    return run_path


def load_run_clip(run_path, init_seed: int = vc.INIT_SEED):
    """Espelha `vc.load_run`. R9 (não confundir checkpoint CLIP com VideoMAE): assert
    `meta["video_encoder"] == "clip"` antes de tocar em qualquer peso; assert
    `fusion_in` do checkpoint bate com `fusion_input_dim_clip(use_bio)` recomputado —
    protege contra servir um checkpoint CLIP numa arquitetura VideoMAE ou vice-versa."""
    import safetensors.torch as st

    run_path = Path(run_path)
    with open(run_path / "meta.json") as f:
        meta = json.load(f)

    assert meta.get("video_encoder") == "clip", (
        f"load_run_clip: meta.json diz video_encoder={meta.get('video_encoder')!r}, "
        f"esperado 'clip' (R9) — {run_path} não é um checkpoint do braço CLIP.")

    expected_fusion_in = fusion_input_dim_clip(meta["use_bio"])
    assert meta["fusion_in"] == expected_fusion_in, (
        f"load_run_clip: fusion_in do checkpoint ({meta['fusion_in']}) não bate com "
        f"fusion_input_dim_clip(use_bio={meta['use_bio']}) = {expected_fusion_in} — "
        f"checkpoint de outra condição/arquitetura.")

    model = build_model_clip(
        init_seed,
        use_bio=meta["use_bio"],
        use_viewer_emb=meta["use_viewer_emb"],
        unfrozen_layers=meta.get("unfrozen_vision_layers", UNFROZEN_VISION_LAYERS))

    sd = st.load_file(str(run_path / "trainable.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"load_run_clip: chaves inesperadas no checkpoint ({unexpected[:5]}...). "
            "Provavelmente o checkpoint é de outra condição/arquitetura.")
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    not_loaded = trainable_names - set(sd.keys())
    if not_loaded:
        raise RuntimeError(
            f"load_run_clip: {len(not_loaded)} parâmetros TREINÁVEIS não estavam no "
            f"checkpoint (ex.: {sorted(not_loaded)[:5]}). Checkpoint incompleto.")
    # `missing` contém os pesos congelados do CLIP/XLM-R — esperado, vêm do from_pretrained.

    scaler = None
    scaler_path = run_path / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("[load_run_clip] AVISO: scaler.pkl ausente. NUNCA ajustar um RobustScaler "
              "novo no conjunto avaliado — isso é vazamento (BUG-B/ARCH-18).")

    print(f"[load_run_clip] {run_path} carregado | video_encoder=clip | "
          f"use_bio={meta['use_bio']}, use_viewer_emb={meta['use_viewer_emb']} | "
          f"{len(missing)} pesos congelados vindos dos backbones")
    return model, scaler


# ============================================================================
# Smoke test — portão duro antes de qualquer treino (plano, célula 19b)
# ============================================================================

def smoke_test_clip(model, dataset, device: str = "cuda", micro_bs: int = 2) -> dict:
    """1 forward + backward + step sob bf16 + gradient checkpointing. Mede memória
    pico, extrapola timing de 1 época, e assert que os grads das `UNFROZEN_VISION_LAYERS`
    camadas destravadas do CLIP não são nulos (R3: se o unfreeze não pegou, o treino roda
    sem erro e sem ajustar o CLIP nenhum — este é o portão que pega isso ANTES de gastar GPU).

    PORTÃO DURO (plano §"Notebook B"): nada roda depois desta célula sem passar.
    """
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    optimizer = get_optimizer_clip(model)

    n = min(micro_bs, len(dataset))
    batch = vc.video_collate([dataset[i] for i in range(n)])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    # modality_dropout_prob=0.0: o smoke test verifica o unfreeze/grafo de gradiente do
    # CLIP (R3), não o modality dropout. Com micro_bs pequeno, deixar o dropout padrão
    # (MODALITY_DROPOUT_P=0.05) tornava este portão ALEATÓRIO — cada amostra tem 5% de
    # chance de zerar keep_video; com micro_bs=2 isso já basta pra zerar o vídeo do
    # micro-batch inteiro (~0,25% de chance) e produzir um falso-negativo indistinguível
    # de unfreeze quebrado. Diagnosticado nesta sessão: com dropout=0 os 64 grads das
    # camadas destravadas do CLIP saem OK (não-None, não-zero). O treino de verdade usa
    # MODALITY_DROPOUT_P normalmente — só este portão isola a variável que testa.
    autocast_device = "cuda" if device == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        loss, logits_a, logits_b, logits_c = model(**batch, modality_dropout_prob=0.0)
    loss.backward()

    # PORTÃO: checar os grads AQUI, logo após backward() e ANTES de optimizer.step()/
    # zero_grad(). `optimizer.zero_grad()` (default set_to_none=True) limpa `.grad` para
    # None em todo parâmetro do optimizer — checar depois dele faz este assert falhar
    # SEMPRE, faça o backward o que fizer. Bug real desta sessão: a ordem original
    # (backward -> step -> zero_grad -> checagem) garantia 64/64 "zero ou nulo" mesmo com
    # o unfreeze e o grafo de gradiente corretos. Não mover este bloco para depois de
    # zero_grad() de novo.
    clip_layers = model.clip_vision.encoder.layers[-UNFROZEN_VISION_LAYERS:]
    n_checked, n_nonzero = 0, 0
    for layer in clip_layers:
        for p in layer.parameters():
            if p.requires_grad:
                n_checked += 1
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    n_nonzero += 1
    assert n_checked > 0, (
        "smoke_test_clip: nenhum parâmetro treinável encontrado nas camadas destravadas "
        "do CLIP — o unfreeze não pegou (R3).")
    assert n_nonzero == n_checked, (
        f"smoke_test_clip: {n_checked - n_nonzero}/{n_checked} params das camadas "
        f"destravadas do CLIP com grad nulo ou ausente — o unfreeze pode não ter pegado "
        f"(R3), ou o forward não está de fato usando o encoder de vídeo.")

    optimizer.step()
    optimizer.zero_grad()

    elapsed = time.time() - t0

    peak_mb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    n_batches_per_epoch = max(1, len(dataset) // micro_bs)
    extrapolated_epoch_s = elapsed * n_batches_per_epoch

    print("─── smoke_test_clip ───")
    print(f"  loss             : {loss.item():.4f}")
    print(f"  fusion_in        : {model.fusion_in}")
    print(f"  micro_bs         : {micro_bs}")
    print(f"  tempo (1 step)   : {elapsed:.3f} s")
    if peak_mb is not None:
        print(f"  memória pico     : {peak_mb:.1f} MB")
    else:
        print("  memória pico     : n/d (CPU)")
    print(f"  timing p/ época  : ~{extrapolated_epoch_s:.1f} s ({n_batches_per_epoch} passos, extrapolado)")
    print(f"  grads não-nulos  : {n_nonzero}/{n_checked} params das "
          f"{UNFROZEN_VISION_LAYERS} camadas destravadas do CLIP")

    return {
        "loss": loss.item(),
        "elapsed_s": elapsed,
        "peak_memory_mb": peak_mb,
        "extrapolated_epoch_s": extrapolated_epoch_s,
        "grads_nonzero": n_nonzero,
        "grads_checked": n_checked,
    }


# ============================================================================
# Self-checks locais (baratos, determinísticos, sem GPU)
# ============================================================================

def selfcheck_fusion_widths_clip():
    """A largura da fusão do braço CLIP tem de ser função da condição, com
    CLIP_VIDEO_DIM no lugar de vc.VIDEO_DIM."""
    assert fusion_input_dim_clip(True) == 2944, fusion_input_dim_clip(True)
    assert fusion_input_dim_clip(False) == 2816, fusion_input_dim_clip(False)
    print("✓ fusion_input_dim_clip: 2944 com bio (B1/B2), 2816 sem bio (B0)")


def run_selfchecks_clip():
    """Selfchecks que NÃO tocam a rede (sem GPU, sem download de pesos). Os selfchecks
    que dependem do CLIP/CLIPProcessor reais (`selfcheck_renorm_equivalence`,
    `smoke_test_clip`) rodam separadamente, na VM, com frames em disco — ver plano
    §"Verificação — LOCAL vs VM"."""
    selfcheck_fusion_widths_clip()
    vc.selfcheck_task_a_convention()   # convenção da Task A é do vc.EXISTVideoModel herdado


if __name__ == "__main__":
    print("video_core_clip.py: overlay CLIP sobre video_core.py, sem execução de treino.")
    print(f"CLIP_NAME={CLIP_NAME} CLIP_VIDEO_DIM={CLIP_VIDEO_DIM} "
          f"UNFROZEN_VISION_LAYERS={UNFROZEN_VISION_LAYERS} CLIP_LR={CLIP_LR}")
    run_selfchecks_clip()
