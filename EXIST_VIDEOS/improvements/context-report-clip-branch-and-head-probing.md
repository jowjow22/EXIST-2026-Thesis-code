# Context Report — Braço CLIP (item 10) e Probing de Heads sobre Trunk Congelado

**Export date:** 2026-08-15
**Report scope:** Full context — sessão inteira sobre o braço CLIP do EXIST 2026 Task 3,
desde a investigação original (doc 16) até o resultado inconclusivo do probing de heads
(doc 21). Sem compressão de decisões/raciocínio; código extenso referenciado por caminho
em vez de duplicado (já vive no repo).

---

## 1. Project Overview

Projeto: tese de mestrado (Jonathan, IPB/UTFPR), sistema Jow para EXIST 2026 (CLEF 2026),
deteção multimodal de sexismo. Task 3 = vídeos TikTok (Tasks A binária, B intenção-da-fonte
3-classes, C multilabel 6-categorias), avaliadas por ICM-Norm hard/soft via PyEvALL.

Esta sessão cobriu inteiramente o **item 10 do doc 17** (`improvements/17_RETRAIN_IMPROVEMENT_OPTIONS.md`):
trocar o encoder de vídeo VideoMAE-base+LoRA por CLIP-ViT-L/14 no sistema de vídeo, como
resposta ao achado do doc 16 (o ramo de vídeo é fraco por capacidade absoluta + objetivo de
pré-treino, não por defeito de pooling/adaptação).

### 1.1 Original Goal
Implementar o braço CLIP (item 10 do doc 17), medir se ele realmente aumenta a contribuição
causal do vídeo (não só a métrica agregada), e — depois de um resultado anômalo (colapso do
ramo de texto) — investigar a causa e decidir o estado final reportável do braço.

### 1.2 Current Objective
**Decisão pendente do autor**, apresentada no fim desta sessão: aplicar a opção 5 do doc 20
(repesar `LOSS_W_C` no treino, retreino completo, ~3,5h de GPU) ou aceitar o estado atual
(braço CLIP, 4 camadas, estado final da época 15) como resultado definitivo, sem mais
intervenções.

---

## 2. Environment & Stack

| Component | Details |
|---|---|
| OS (dev local) | macOS (Darwin), sem GPU — mount SSHFS pra VM |
| OS/execução real | VM separada, Jupyter remoto, `/home/a67606/jupyter-env/` |
| Python / libs (VM, confirmado 2026-08-01/09) | `torch 2.6.0+cu124`, `transformers 5.6.0`, `accelerate 1.13.0`, `peft 0.19.1`, `safetensors 0.7.0`, `PyEvALL 0.1.78`, `opencv 4.13.0`, `openai-whisper 20250625` |
| Repo root local | `/Users/jonathansantosdasilva/Documents/projects/VM-THESIS` |
| Diretório de trabalho desta sessão | `EXIST_VIDEOS/` (relativo à raiz do repo, e ao cwd esperado na VM: `/home/a67606/Thesis/EXIST_VIDEOS`) |
| Idioma da conversa | Português do Brasil (preferência registrada em memória de projeto) |

⚠ **Regra do projeto:** nunca assumir que versão de lib local = versão da VM (só a VM
executa treino/GPU real). Mount SSHFS é lento — nunca `du`/`find` recursivo em
`Dataset/`/frames/áudio.

---

## 3. Architecture & Design

### 3.1 Component Map

- `EXIST_VIDEOS/video_core.py` — módulo original (VideoMAE+LoRA), **NUNCA editado** nesta
  sessão (regra do autor + convenção do projeto).
- `EXIST_VIDEOS/video_core_clip.py` — overlay novo, importa de `video_core` (não duplica).
  Contém todo o código específico do encoder CLIP: `CLIPFrameEncoder` (wrapper
  "videomae-like"), `EXISTVideoModelCLIP`, `build_model_clip`, `get_optimizer_clip`,
  `save_run_clip`/`load_run_clip`, `smoke_test_clip`.
- `EXIST_VIDEOS/Thesis_video_clip.ipynb` — notebook de orquestração (treino), clone
  estrutural de `Thesis_video_improved.ipynb`.
- `EXIST_VIDEOS/probe_video_contribution_clip.ipynb` — sonda causal (permutação) do braço
  CLIP, adaptado de `probe_video_contribution.ipynb` (braço B15/VideoMAE original).
- `EXIST_VIDEOS/probe_head_training.ipynb` — notebook novo desta sessão: probing das 3
  heads (A/B/C) sobre trunk congelado, para ambos os trunks CLIP e B15.
- `EXIST_VIDEOS/improvements/16,17,18,20,21_*.md` — documentos de decisão/rastreabilidade
  (ver §4.1).

### 3.2 Data Flow (braço CLIP)

```
16 frames (uint8, v2, já em disco) --renormalização afim--> CLIP-ViT-L/14
  (4 últimas camadas destravadas full-rank, sem LoRA)
  --> CLS pré-post_layernorm por frame --> [B,16,1024]
  --> TemporalAttentionPool (herdado, sem reescrita) --> [B,1024] (vídeo)

texto --> XLM-R-large + LoRA r=16 (herdado, sem mudança) --> [B,1024] (texto)
áudio --> Whisper pré-extraído v2 (máscara+média ponderada) --> [B,768] (áudio)
bio   --> Bio-MLP (herdado) --> [B,128] (bio, opcional)

concat([vídeo, áudio, texto, bio]) --> [B,2944] (fusion_in, PRÉ-fusão)
  --> model.fusion = Linear(2944,1024)->GELU->Dropout --> [B,1024] (fused, PÓS-fusão)
  --> head_a(fused)->[B,1] | head_b(fused)->[B,3] | head_c(fused)->[B,6]
```

**Distinção crítica usada nesta sessão:** "fusion_in" (2944-d, entrada do módulo fusion,
usado nos probes CAUSAIS por slot de modalidade) vs "fused" (1024-d, SAÍDA do módulo
fusion, o que as heads realmente consomem, usado no probing de heads desta sessão). Um
bug real ocorreu por confundir os dois (ver §7.1).

### 3.3 Key Design Decisions

- **Wrapper `CLIPFrameEncoder`** em vez de duplicar `EXISTVideoModel.forward`: se
  apresenta com a mesma interface (`forward(pixel_values)->objeto com .last_hidden_state`,
  `gradient_checkpointing_enable`) que o forward original já consome. `EXISTVideoModelCLIP`
  herda de `vc.EXISTVideoModel`, só substitui `video_pool` e `fusion`. O `forward` inteiro
  (modality dropout, `_mask`, texto, áudio, bio, fusão, heads, `compute_loss`) é herdado
  sem reescrita — evita "o maior anti-padrão já documentado no projeto" (cópias
  divergentes do pipeline).
- **`UNFROZEN_VISION_LAYERS = 4`** (constante em `video_core_clip.py`) — mesmo padrão do
  `EXIST_MEMES/memes_core.py:488-494` (~50,4M params treináveis). **Testado 4→2 e
  REFUTADO** — ver §10.
- **CLS pré-post_layernorm** do CLIP (não `pooler_output`) — cópia literal de
  `memes_core.py:573`.
- **`meta.json["video_encoder"]`** como desambiguador autoritativo — as chaves do
  state_dict nascem `videomae.clip.*` (wart de nomenclatura aceita, herança de
  `self.videomae` do `EXISTVideoModel.__init__`).
- **Estado reportado do braço CLIP: FINAL da época 15**, não `E=argmin(eval_loss)` — ver
  §10 para a história completa dessa decisão (foi revertida de argmin depois de duas
  hipóteses de correção falharem).

---

## 4. File & Directory Structure

```
EXIST_VIDEOS/
├── video_core.py                              # NUNCA editado
├── video_core_clip.py                         # overlay CLIP — UNFROZEN_VISION_LAYERS=4
├── Thesis_video_improved.ipynb                # braço VideoMAE, intocado
├── Thesis_video_clip.ipynb                    # treino do braço CLIP — ver §5.2
├── probe_video_contribution_clip.ipynb        # sonda causal (permutação) do braço CLIP
├── probe_head_training.ipynb                  # NOVO — probing de heads (esta sessão)
├── run_B2_bio_viewer_seed42_CLIP/              # artefato oficial do braço CLIP (4 cam., época 15 final)
├── run_B2_bio_viewer_seed42_CLIP_L2/           # artefato do experimento refutado (2 camadas, argmin época 4)
├── run_B2_bio_viewer_seed42_B15/               # baseline VideoMAE oficial
├── analysis/
│   ├── probe_report_clip.txt                  # log da sonda causal, run oficial
│   ├── probe_cache_run_B2_..._CLIP/            # cache da sonda causal (fusion_in 2944-d)
│   ├── probe_train_fused_run_B2_..._CLIP.pt    # NOVO — cache do probing de heads (1024-d, treino)
│   ├── probe_head_val_fused_run_B2_..._CLIP.pt # NOVO — cache do probing de heads (1024-d, val)
│   └── probe_head_training_report.txt          # log do probing de heads
└── improvements/
    ├── 16_VIDEO_INVESTIGATION_RESULTS.md       # investigação original (H1-H5)
    ├── 17_RETRAIN_IMPROVEMENT_OPTIONS.md       # opções de melhoria, item 10 = CLIP
    ├── 18_CLIP_VIDEO_BRANCH_SPEC.md            # decisão/design do braço CLIP + §9 histórico completo
    ├── 20_EPOCH_SELECTION_TASK_C.md            # pesquisa MTL, opções 1-6 p/ Task C atrasada
    ├── 21_HEAD_PROBING_PREREGISTRATION.md      # pré-registro + resultado do probing (INCONCLUSIVO)
    └── context-report-clip-branch-and-head-probing.md   # ESTE ARQUIVO
```

### 4.1 Key Files

| File | Status | Role |
|---|---|---|
| `video_core_clip.py` | ✅ Completo, `UNFROZEN_VISION_LAYERS=4` (revertido de 2) | Overlay CLIP |
| `Thesis_video_clip.ipynb` | ✅ Completo, reporta estado FINAL (não argmin) | Treino do braço CLIP |
| `probe_video_contribution_clip.ipynb` | ✅ Completo, `CACHE_DIR` auto-derivado de `RUN_DIR` (bug de cache corrigido) | Sonda causal por permutação |
| `probe_head_training.ipynb` | ✅ Completo, rodado, resultado INCONCLUSIVO | Probing de heads sobre trunk congelado |
| `improvements/18_CLIP_VIDEO_BRANCH_SPEC.md` | ✅ §9 com histórico completo das 4 runs | Doc de decisão do braço CLIP |
| `improvements/20_EPOCH_SELECTION_TASK_C.md` | ✅ Completo (pesquisa já feita em sessão anterior) | Opções pra divergência eval_loss/Task C |
| `improvements/21_HEAD_PROBING_PREREGISTRATION.md` | ✅ Pré-registro + seção Resultado escrita nesta sessão | Protocolo + resultado do probing |

---

## 5. Complete Code & Configurations

Código extenso já vive nos arquivos do repo (caminhos acima) — não duplicado aqui por
inteiro. Trechos críticos que mudaram nesta sessão:

### 5.1 `video_core_clip.py` — constante revertida (não mexer de novo)

```python
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
                                # (não E=argmin).
```

Também: os dois asserts de contagem de params (`build_clip_vision`, `get_optimizer_clip`)
foram tornados proporcionais a `unfrozen_layers` (antes hardcoded `[40M,60M]`, só válido
pra 4 camadas):

```python
per_layer_estimate = 12_596_224  # medido: 50.384.896 / 4 camadas
expected = unfrozen_layers * per_layer_estimate
lo, hi = 0.7 * expected, 1.3 * expected
assert lo < n_train < hi, (...)
```

### 5.2 `Thesis_video_clip.ipynb` — protocolo de seleção de estado (célula 12, código)

O notebook **não recarrega** `E=argmin(eval_loss)` — `actual_model` permanece exatamente o
estado que o `Trainer` deixa ao fim das 15 épocas:

```python
evals = [h for h in trainer.state.log_history if "eval_loss" in h]
best = min(evals, key=lambda h: h["eval_loss"])
best_epoch_int = round(best["epoch"])
print(f"min(eval_loss) na época {best_epoch_int} ... — REGISTRO APENAS. "
      f"Este braço reporta o estado FINAL da época {NUM_EPOCHS} ... "
      f"nenhum reload acontece aqui.")

actual_model = extract_model_from_parallel(trainer.model)
```

`SaveTrainableStateCallback` (registrado no `Trainer`) ainda grava `epoch_states/epoch_NN.safetensors`
a cada época — só como **registro diagnóstico**, removido (seção 14) depois que a run
oficial é validada.

`meta.json` salvo com `"epoch_selection": "final"`, `"selected_epoch": NUM_EPOCHS`,
`"min_eval_loss_epoch": best_epoch_int` (registro histórico do argmin, não usado pra
selecionar nada).

### 5.3 `probe_head_training.ipynb` — hook correto (pós-fix)

O bug (§7.1) estava aqui — versão corrigida:

```python
buf = []
def hook(_m, _inp, out):
    # saída de model.fusion INTEIRO (Linear->GELU->Dropout), 1024-d — o que as
    # heads consomem. NÃO a entrada de fusion[0] (2944-d, pré-fusão).
    buf.append(out.detach().float().cpu())
h = model.fusion.register_forward_hook(hook)
```

E o gate de leitura (célula final, corrigido nesta sessão — crítico, não reverter):

```python
controls_ok = True
for head in ("a", "b"):
    ...
    if abs(d_hard) > 0.03 or abs(d_soft) > 0.03:
        controls_ok = False
        print("⚠ controle moveu mais que 0.03 — desconfiar do protocolo ...")

if not controls_ok:
    print("-> INCONCLUSIVO — controles A/B falharam, protocolo de probing não é "
          "confiável para decidir sobre C nesta rodada. NÃO ler como 'C sem sinal' "
          "nem como 'C sub-treinada'.")
elif d_hard > 0.03 or d_soft > 0.03:
    print("-> C_probe >> C_original: head sub-treinada, ganho barato.")
else:
    print("-> C_probe ~= C_original: representação sem sinal extra.")
```

⚠ **Este gate disparou "INCONCLUSIVO" na run real** (controles falharam 4/4) — ver §7 e §9.

### 5.4 Loss reweighting proposta (opção 5 do doc 20, NÃO aplicada ainda)

```python
# video_core.py:1334-1349 aprox. — loss_task_c usa .mean(dim=-1) sobre as 6 classes.
# LOSS_W_A=1.0, LOSS_W_B=0.7, LOSS_W_C=0.7 (video_core.py:272-274) — peso EFETIVO é
# 1.0 / 0.7 / (0.7 * ~1/6) porque a redução .mean() já divide por 6.
# Correção principiada (opção 2/5 do doc 20): LOSS_W_C: 0.7 -> 4.2 (= 0.7 * 6),
# equivalente a trocar a redução de loss_task_c de .mean(dim=-1) para .sum(dim=-1).
# NÃO É valor afinado por busca — desfaz a assimetria aritmética, ponto.
```

---

## 6. Commands & Execution

```bash
# Verificação de artefatos existentes (rodado nesta sessão, via mount SSHFS local)
ls -d run_* 2>&1
for d in run_B2_bio_viewer_seed42_CLIP run_B2_bio_viewer_seed42_B15; do
  for f in meta.json trainable.safetensors final_metrics.json; do
    [ -f "$d/$f" ] && echo "OK $d/$f" || echo "FALTA $d/$f"
  done
done
# -> ambos completos, confirmado

# Checar espaço em disco na VM (usuário rodou)
df -h ~
# -> 126G total, 18G livres (86% usado) — orientou decisão de usar checkpoints
#    enxutos (trainable-only, ~220MB) em vez de checkpoints completos do HF Trainer

# Validação de sintaxe local (sem GPU, sem valor sobre comportamento real na VM)
python3 -c "import ast; ast.parse(open('video_core_clip.py').read())"
python3 -c "import json,ast; nb=json.load(open('X.ipynb')); [ast.parse(''.join(c['source'])) for c in nb['cells'] if c['cell_type']=='code']"
```

**Execuções reais (na VM, GPU, feitas pelo usuário — reportadas via chat):**
1. `Thesis_video_clip.ipynb` — 4 treinos completos (15 épocas cada, ~373-882s/época
   variando por config), ver §10 pra números completos.
2. `probe_video_contribution_clip.ipynb` — 3 sondas causais completas (uma por checkpoint
   relevante).
3. `probe_head_training.ipynb` — 1 execução completa (extração + 6 probes + tabela +
   leitura), resultado em §9.

---

## 7. Errors, Failures & Resolutions

### 7.1 Bug de cache estático na sonda causal

**Sintoma:** `probe_video_contribution_clip.ipynb` reportou "[cache] carregado de
analysis/probe_cache_CLIP/val_cache.pt (forward saltado)" mesmo sondando um checkpoint
**novo** (época 2 vs o checkpoint antigo de época 15 usado na 1ª sonda).

**Root cause:** `CACHE_DIR` era um path fixo (`analysis/probe_cache_CLIP`), sem checagem
de fingerprint contra o checkpoint sendo sondado — reusava silenciosamente o forward de
um checkpoint DIFERENTE.

**Resolução:** `CACHE_DIR` agora deriva de `RUN_DIR.name` automaticamente
(`Path("analysis") / f"probe_cache_{RUN_DIR.name}"`) — cada checkpoint tem seu próprio
cache, corrigido permanentemente, não mais um risco manual.

### 7.2 `smoke_test_clip` — assert de grad falso-negativo (2 iterações)

**Sintoma:** `AssertionError: smoke_test_clip: 64/64 params das camadas destravadas do CLIP
com grad nulo ou ausente`, mesmo com o forward/backward funcionando.

**Tentativa 1 (errada):** hipótese de modality dropout zerando o ramo de vídeo
aleatoriamente. Fix aplicado (`modality_dropout_prob=0.0` no smoke test) não resolveu —
mesmo erro reapareceu.

**Root cause real:** `optimizer.zero_grad()` era chamado ANTES do loop que checava
`p.grad` — `zero_grad()` (default `set_to_none=True`) limpa `.grad` para `None` em todo
parâmetro, então o assert falhava sempre (64/64), independente do backward ter funcionado.

**Resolução:** mover o bloco de checagem de grad para logo após `loss.backward()`, ANTES
de `optimizer.step()`/`zero_grad()`. Confirmado funcionando: 64/64 grads não-nulos,
memória pico 4193MB, timing ~882s/época extrapolado.

### 7.3 `probe_head_training.ipynb` — placeholder de RUN_DIR errado

**Sintoma:** primeira versão do notebook tinha
`Path("EXIST_VIDEOS/checkpoints_B2_bio_viewer_seed42_CLIP/epoch_states/epoch_15.safetensors")`
como `RUN_DIR` do trunk CLIP — um arquivo `.safetensors` solto, não o diretório da run.

**Root cause:** erro de digitação/geração no script que criou o notebook (não do
usuário — confirmado pelo autor da sessão).

**Resolução:** corrigido para `Path("run_B2_bio_viewer_seed42_CLIP")` (diretório completo
com `meta.json`+`trainable.safetensors`+`final_metrics.json`, confirmado existente via
`ls` no mount).

### 7.4 `probe_head_training.ipynb` — `RuntimeError: mat1 and mat2 shapes cannot be
multiplied (2019x2944 and 1024x6)`

**Root cause:** o hook de extração capturava a **entrada** de `model.fusion[0]` (2944-d,
vetor PRÉ-fusão — mesmo ponto usado pela sonda causal por slot de modalidade), não a
**saída** de `model.fusion` inteiro (1024-d, o que `head_a/b/c` realmente recebem). Erro
meu (do agente), não do usuário.

**Resolução:** hook trocado pra `model.fusion.register_forward_hook(hook)` capturando
`out` (não `inp[0]`). Chave do cache renomeada `fusion_in`→`fused` em todo o notebook.
Cache também renomeado (`probe_train_fused_*`/`probe_head_val_fused_*`) pra nunca colidir
com o cache da sonda causal (que tem dimensão diferente, 2944 vs 1024). Assert de shape
adicionado na extração pra pegar esse tipo de erro na hora.

### 7.5 Probing de heads — controles A/B falharam (não é bug de código, é falha do
protocolo experimental)

⚠️ **NÃO É UM ERRO DE CÓDIGO — é um resultado experimental negativo, documentado em
detalhe no §9.** Listado aqui só por completude do audit de "coisas que não deram certo".

---

## 8. What Works Right Now

- ✅ `video_core_clip.py` — overlay CLIP completo, sintaticamente válido, testado na VM
  (smoke test passa, 4 treinos completos rodaram sem erro).
- ✅ `Thesis_video_clip.ipynb` — treina o braço CLIP corretamente, reporta estado FINAL da
  época 15, determinismo confirmado (2 runs idênticas byte-a-byte no log de treino).
- ✅ `probe_video_contribution_clip.ipynb` — sonda causal funcional, bug de cache
  corrigido permanentemente (auto-deriva `CACHE_DIR` de `RUN_DIR`).
- ✅ `probe_head_training.ipynb` — roda de ponta a ponta sem erro, produz tabela de
  comparação original-vs-probe e leitura pré-registada com gate de controle funcionando
  corretamente (disparou "INCONCLUSIVO" quando deveria).
- ✅ `run_B2_bio_viewer_seed42_CLIP/` — artefato oficial válido, `meta.json` +
  `trainable.safetensors` + `final_metrics.json` presentes e consistentes.
- ✅ Toda a cadeia de documentação (`16→17→18→20→21`) rastreável e sem contradições
  internas.

---

## 9. What's Broken or Incomplete

- ❌ **Bifurcação do doc 20 sobre Task C continua em aberto** — "Task C sub-treinada
  dentro do treino conjunto" vs "representação congelada do trunk não carrega sinal
  suficiente de C" — nenhuma das duas rotas tentadas (seleção de época, probing de heads)
  respondeu isso. Ver §10 e §11.
- 🔧 **Resultado do probing de heads (`probe_head_training.ipynb`, rodado 2026-08-15):**

  ```
  trunk variante  A_hard   A_soft   B_hard   B_soft   C_hard   C_soft
  CLIP  original  0.5741   0.5172   0.4986   0.4122   0.3640   0.1368
  CLIP  probe_a   0.4953   0.3890   0.4986   0.4122   0.3640   0.1368
  CLIP  probe_b   0.5741   0.5172   0.4604   0.2780   0.3640   0.1368
  CLIP  probe_c   0.5741   0.5172   0.4986   0.4122   0.3108   0.0000
  B15   original  0.5536   0.5058   0.4968   0.3947   0.3436   0.1514
  B15   probe_a   0.5588   0.4298   0.4968   0.3947   0.3436   0.1514
  B15   probe_b   0.5536   0.5058   0.4493   0.2767   0.3436   0.1514
  B15   probe_c   0.5536   0.5058   0.4986   0.4122   0.3120   0.0000
  ```

  best_epoch das probes (argmin da própria BCE val loss, de 300 épocas full-batch
  disponíveis): CLIP → head_a=5, head_b=3, head_c=11. B15 → head_a=6, head_b=5,
  head_c=15.

  **Veredito: INCONCLUSIVO por falha de controle.** Controles A/B pioraram em TODOS os 4
  casos (CLIP-A, CLIP-B, B15-A, B15-B) — nunca empataram, sempre na direção errada, mais
  forte no soft que no hard. `task_c_icmsoftnorm_soft` caiu para exatamente `0,000000`
  nos dois trunks (consistente com clip do ICM-Soft-Norm em ICM cru negativo — não
  confirmado se é bug ou floor genuíno, ver pendência abaixo). Pelo gate pré-registado no
  doc 21 §1.4, o veredito sobre C que o notebook chegou a calcular ("representação sem
  sinal") **NÃO é válido** e não deve ser citado como conclusão em nenhum lugar.

  **Diagnóstico do mecanismo (não confirmado, `[HIPÓTESE]` bem fundamentada):** os
  `best_epoch` das probes (3-15 de um orçamento de 300) mostram que elas pararam pelo
  argmin da própria loss BCE de validação muito cedo — antes de a superfície de decisão
  calibrar para o que o ICM mede. As heads originais receberam ~960 steps de mini-batch
  com dropout ativo; as probes convergiram o BCE-val em poucos steps full-batch e pararam
  ali. Mesma divergência loss-cedo/métrica-tarde (padrão Guo et al. ICML 2017) já
  documentada nos docs 20/16 para o treino do trunk inteiro, reaparecendo *dentro* do
  próprio instrumento de diagnóstico.

  **⚠️ Pendência não verificada:** confirmar se `C_soft=0,000000` idêntico nos dois
  trunks é o clip do ICM-Soft-Norm (dois valores CRUS `eval_task_c_icmsoft_soft`
  diferentes e negativos → normalizado clipa em 0, leitura esperada) ou um bug de código
  (valores crus idênticos indicariam problema real). ~1 minuto de trabalho: olhar o valor
  cru no log de `probe_head_training_report.txt`.

- ⏳ **`probe_whisper_attention_padding.ipynb`** (item 11, ARCH-04 residual) — pronto
  desde sessão anterior, nunca executado. Fora do escopo desta sessão, mas pendência
  aberta do projeto.
- ⏳ **Célula de sonda causal sobre a Run 4 (artefato atual)** — determinismo já confirmado
  nos `final_metrics.json`, mas a sonda causal (`probe_video_contribution_clip.ipynb`) em
  si não foi re-rodada especificamente sobre o artefato regenerado da 4ª run (deveria
  reproduzir `‖Δ_texto‖≈0,785` da Run 1, por determinismo, mas isso não foi verificado
  diretamente nesta sessão).

---

## 10. Direction Changes & Abandoned Approaches

Histórico completo e definitivo já está em `improvements/18_CLIP_VIDEO_BRANCH_SPEC.md`
§9 (escrito nesta sessão) — resumo aqui:

- **4 runs de treino do braço CLIP**, cronologicamente:
  1. 4 camadas, `save_strategy="no"`, estado final da época 15 (única opção disponível
     então). `final_metrics`: A 0,5741/0,5172, B 0,4986/0,4122, C 0,3640/0,1368.
     Sonda causal revelou colapso do ramo de texto (`‖Δ_texto‖=0,785`, 24× menor que o
     B15).
  2. **Hipótese 1 (overtraining) testada e REFUTADA.** Adicionado
     `SaveTrainableStateCallback` + `E=argmin(eval_loss)`. 2º treino, argmin=época 2
     (eval_loss 1,5158). Sonda: `‖Δ_texto‖=3,138` (melhora parcial, ainda 6× abaixo do
     B15). Overtraining explica só parte, não o todo.
  3. **Hipótese 2 (desequilíbrio de capacidade do encoder) testada e REFUTADA, na direção
     oposta à prevista.** `UNFROZEN_VISION_LAYERS` 4→2. 3º treino, argmin=época 4.
     Sonda: `‖Δ_texto‖=1,046` (PIOROU), `‖Δ_vídeo‖=18,433` (SUBIU, > B15). Reduzir
     capacidade do encoder tornou o vídeo MAIS dominante, não menos — mecanismo real
     (não medido diretamente, `[HIPÓTESE]` bem fundamentada via advisor): `video_pool`/
     `fusion` são sempre full-rank independente dessa constante; a dominância é decidida
     pela qualidade relativa das features na fusão, não pela contagem de params do
     encoder.
  4. **Decisão final: reverter `UNFROZEN_VISION_LAYERS` 4→2→4, abandonar
     `E=argmin(eval_loss)`, reportar estado FINAL da época 15** — por simetria com o
     baseline VideoMAE (regra do `CLAUDE.md`: estado final vs estado final) e porque o
     padrão passou a ser lido como **trade-off real de trocar o encoder**, não bug
     corrigível. 4º treino (determinístico, reproduz a Run 1).

- **Debate "época 2 vs 7 vs 15" — rejeitado.** Depois de ver a curva completa da Run 4,
  época 7 parecia atraente (vale local de eval_loss + pico de Task A/B). Rejeitado por
  dois motivos: (1) não é o argmin global (época 2 é mais fundo); escolher 7 só depois de
  ver métricas boas ali é o mecanismo de leakage que o projeto proíbe. (2) não existe
  "época de equilíbrio" — cada (sub)task pica numa época diferente (A em 7, B em 10, C
  hard em 10-11, C soft em 15, ainda subindo). Decisão mantida: estado final, sem
  exceção.

- **Probing de heads sobre trunk congelado (opção 4 generalizada do doc 20, esta
  sessão) — rota fechada por medição, INCONCLUSIVA, não consertada.** Pré-registrada em
  doc 21 antes de rodar. Rodada, resultado: controles A/B falharam 4/4 (§9). Decisão
  explícita do autor + advisor: não tentar consertar o protocolo (mexer em LR/épocas até
  os controles empatarem seria leakage disfarçado e mesmo bem-sucedido só devolveria
  diagnóstico, não ganho). Rota fechada, mesmo estatuto das rotas fechadas do doc 16.

- **Ideia descartada sem implementar: destravar XLM-R (texto) full-rank em vez de
  LoRA**, como alternativa a encolher o CLIP. Analisada e rejeitada (via advisor) porque
  o texto já dominava o B15 com o mesmo LoRA r=16 — capacidade nunca foi o gargalo do
  texto; misturaria duas variáveis no braço (encoder + estratégia de fine-tune do texto).

- **Ideia descartada sem implementar: "checkpoint Frankenstein"** (trunk de uma época +
  head de outra época). Sem fonte na literatura que autorize (doc 20 §3); `Resurrection`
  do paper MAGPIE é evidência direta CONTRA.

---

## 11. Pending Decisions & Open Questions

- ❓ **A decisão principal, apresentada ao autor no fim desta sessão:**
  1. **Opção 5 do doc 20** — repesar `LOSS_W_C` no treino (0,7→4,2, valor principiado,
     não afinado por busca — desfaz a divisão por 6 da redução `.mean` em
     `loss_task_c`), retreino completo do trunk CLIP, ~3,5h de GPU. É correção de causa
     nos dois cenários da bifurcação em aberto. É REALOCAÇÃO — subir C deve custar A
     e/ou B, relatório final precisa trazer as seis métricas, não só a que melhorou.
  2. **Aceitar o estado atual como resultado final do braço CLIP** — já documentado
     integralmente em `18_CLIP_VIDEO_BRANCH_SPEC.md` §9. Arco experimental completo (H2
     testada, duas hipóteses de correção testadas e refutadas, achado de robustez sobre
     formato-vs-conteúdo registrado). Encerrar aqui é decisão legítima, não desistência.

  **Nenhuma das duas é recomendada sobre a outra** — decisão do autor.

- ❓ Confirmar se `C_soft=0,000000` idêntico nos dois trunks do probing é floor do
  ICM-Soft-Norm (esperado, benigno) ou bug de código (preocupante) — checagem de ~1 min
  no log cru, não feita ainda.

- ❓ Rodar a sonda causal sobre o artefato regenerado da Run 4 (deveria reproduzir
  `‖Δ_texto‖≈0,785` por determinismo — não verificado diretamente).

---

## 12. Next Steps

- [ ] **A sessão que continuar este trabalho deve ABRIR perguntando ao autor a decisão do
      §11** (opção 5 do doc 20 vs encerrar o braço) — não presumir, não decidir sozinha.
- [ ] Se opção 5: implementar `LOSS_W_C = 4.2` em `video_core.py`... **espera, não —
      `video_core.py` nunca é editado.** A mudança precisa ser um overlay em
      `video_core_clip.py` ou um parâmetro passado — decidir a forma exata com o autor
      antes de tocar em código (o projeto tem regra forte contra editar `video_core.py`).
- [ ] Se opção 5: rodar retreino completo (~3,5h GPU), reportar as 6 métricas (não só
      C), aplicar a mesma disciplina de determinismo/checkpointing já estabelecida.
- [ ] Se encerrar: nenhuma ação de código — só fechar a narrativa na tese/paper citando
      os docs 16/18/20/21 como o arco experimental completo do braço CLIP.
- [ ] (Não bloqueante, pode ser feito em paralelo) Verificar `eval_task_c_icmsoft_soft`
      cru no log do probing pra resolver a pendência do §11.
- [ ] (Não bloqueante, pendência antiga do projeto) Rodar
      `probe_whisper_attention_padding.ipynb` (item 11).

---

## 13. External References

| Resource | Reference | Purpose |
|---|---|---|
| Guo et al. | ICML 2017, PMLR v70, arxiv 1706.04599 | Padrão loss-sobe-métrica-sobe (calibração) — usado pra explicar divergência eval_loss/Task C e o mecanismo do probing inconclusivo |
| Harutyunyan et al. | Scientific Data 6:96 (2019), arxiv 1703.07771 | Early stopping por task em MTL |
| Koreeda et al. (Hitachi) | Shared Task MRP@CoNLL 2019, arxiv 1910.01299 | Early stopping por task, protocolo mais limpo (doc 20 opção 3) |
| Horych et al. (MAGPIE) | LREC-COLING 2024, arxiv 2403.07910 | Head-Specific Early Stopping, evidência CONTRA "checkpoint Frankenstein" |
| Alain & Bengio | 2016, arxiv 1610.01644 | Linear probing sobre representações congeladas — base do probing de heads desta sessão |
| Wortsman et al. (model soups) | ICML 2022, arxiv 2203.05482 | Descartado — seleção pela métrica em held-out é leakage sob a regra do projeto |
| `EXIST_MEMES/memes_core.py:488-494,573` | arquivo do repo | Precedente do unfreeze de 4 camadas do CLIP e CLS pré-post_layernorm |
| `CLAUDE.md` | raiz do repo | Regras de projeto — simetria estado-final, anti-leakage, versões de lib da VM |

---

> **Instructions for the next chat:** This is a full context report from a previous
> session. Read it completely before responding. You now have full knowledge of the
> project state, all decisions made, all code written, and all pending work. Do not ask
> the user to repeat information already present in this document. **Begin by asking the
> author the pending decision in §11/§12 (opção 5 do doc 20 vs encerrar o braço CLIP)
> before writing or running any code.** Fale em português do Brasil.
