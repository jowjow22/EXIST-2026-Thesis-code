# 04 — Reconciliação (Agente 1, arquitetura)

Respostas aos 6 itens levantados pelo coordenador após os relatórios
`02_SIMPLIFICATION_FINDINGS.md` (Agente 2) e `03_OPTIMIZATION_FINDINGS.md`
(Agente 3). Este arquivo é **especificação** — onde eu decido algo, decido por
inteiro e com código.

Convenção: **ACEITO** / **REJEITO** / **REVISADO PARA X**. Toda mudança de
posição em relação ao `01_ARCHITECTURE_FINDINGS.md` está também consolidada na
tabela da seção 7.

---

## 1. OPT-13 — o que `use_bio=False` deve construir

**DECISÃO: ACEITO a opção (ii) do OPT-13** — a largura da fusão é função da
condição; um braço sem bio não carrega slot de 128 zeros. **Com uma correção
importante ao diagnóstico do Agente 3, e com duas regras adicionais que ele não
cobriu e sem as quais a tabela continua enviesada.**

### 1.1 Correção ao diagnóstico: aqui os zeros são algebricamente inertes

O OPT-13 afirma que manter 2 688 com o slot zerado reproduz o achado nº 1 do
brief (confound de fusão com modalidade ausente). **Isso não é exato, e a
diferença importa para não escrevermos uma justificativa errada na tese.**

No modelo de memes o vetor de zeros passava por um **`LayerNorm` global depois da
concatenação** — e aí sim os zeros distorcem média e variância do ramo vivo. No
modelo de vídeo a fusão é:

```python
self.fusion = nn.Sequential(nn.Linear(FUSION_INPUT_DIM, FUSION_HIDDEN_DIM),
                            nn.GELU(), nn.Dropout(0.1))
```

**Não há normalização após o `cat`.** Para uma `Linear`, colunas multiplicadas por
entrada zero contribuem exatamente 0 na saída **e recebem gradiente exatamente 0**
(`∂L/∂W[:, j] = grad_out ⊗ x_j = 0`). Ou seja, num B0 **re-treinado** com a
construção (i), as 131 072 colunas do slot bio ficariam congeladas no init e
nunca influenciariam nada: (i) e (ii) produzem modelos funcionalmente
equivalentes, diferindo só (a) na contagem de params reportada e (b) no fator de
init `1/sqrt(fan_in)` (2 688 vs 2 560, ~2,4 % de diferença no bound do Kaiming
uniform) e no consumo de RNG.

**Então por que adoto (ii) mesmo assim?** Por três motivos que continuam válidos:

1. **Auditabilidade.** "B0 é um modelo content-only" passa a ser literalmente
   verdade e verificável por `model.fusion[0].in_features == 2560`. Com (i), a
   afirmação depende de um argumento algébrico que nenhum revisor vai refazer.
2. **Blindagem contra a próxima mudança.** No instante em que alguém acrescentar
   uma normalização à entrada da fusão — e há motivos plausíveis para isso — a
   construção (i) vira **de facto** o confound de memes, silenciosamente. (ii)
   torna isso impossível por construção.
3. **Contagem de params honesta na tabela**, com a diferença declarada.

**Regra que decorre disto e que precisa estar escrita no código:**

> **Enquanto qualquer braço zerar um slot da fusão, é proibido inserir
> `LayerNorm`/`BatchNorm` sobre o vetor concatenado.** Normalizar por ramo
> (antes do `cat`) é permitido; normalizar depois do `cat` não é.

### 1.2 Segunda regra, não coberta pelo OPT-13: a taxa de modality dropout tem de ser invariante entre braços

Com as flags no construtor, o ramo `elif r < 4*p and use_bio` do dropout atual
deixaria B0 com **3 faixas × 5 % = 15 %** de amostras afetadas e B2 com **20 %**.
Isso é uma diferença de **força de regularização entre os braços da ablação** —
um confound novo, introduzido pela própria correção do ARCH-20, e que teria
explicado parte de qualquer delta B0↔B2.

A formulação de máscara do ARCH-07 resolve isso de graça, desde que as faixas
sejam **fixas e posicionais**: `r ∈ [0,p)` vídeo, `[p,2p)` áudio, `[2p,3p)` texto,
`[3p,4p)` bio. Um braço sem bio simplesmente nunca usa a quarta faixa, e as taxas
de vídeo/áudio/texto continuam exatamente `p` em **todos** os braços. É essa a
invariante correta (taxa **por modalidade**, não taxa total).

### 1.3 Código — construtor e forward definitivos

Incorpora: ARCH-03 (b) (sem cross-attention), ARCH-07 (máscara por amostra),
ARCH-08 (sem `bio_ln`), ARCH-20 + OPT-13 (flags no construtor, largura variável),
SIMP-06 (`rt_norm`/`fix_norm` fora), SIMP-14 (constantes nomeadas).

```python
# ── constantes (video_core.py) ────────────────────────────────────────────
BIO_MLP_HIDDEN     = 128     # camada interna do bio_mlp (era o literal 128)
BIO_DROPOUT        = 0.2
FUSION_DROPOUT     = 0.1
MODALITY_DROPOUT_P = 0.05


def fusion_input_dim(use_bio: bool) -> int:
    """Largura da entrada da fusão — é FUNÇÃO da condição de ablação.
    Um braço sem bio não carrega um slot de 128 zeros: ver OPT-13/ARCH-20."""
    d = VIDEO_DIM + AUDIO_PROJ_DIM + TEXT_DIM          # 768 + 768 + 1024 = 2560
    if use_bio:
        d += BIO_HIDDEN_DIM                            # + 128            = 2688
    return d


class EXISTVideoModel(nn.Module):
    """Flags de ablação são de CONSTRUÇÃO, não de forward. Cada condição é um
    modelo diferente, treinado do zero, com o seu próprio checkpoint."""

    def __init__(self, videomae, xlmr, *,
                 use_bio: bool = True, use_viewer_emb: bool = True):
        super().__init__()
        if use_viewer_emb and not use_bio:
            raise ValueError("use_viewer_emb=True exige use_bio=True")
        self.use_bio        = use_bio
        self.use_viewer_emb = use_viewer_emb

        self.videomae   = videomae
        self.xlmr       = xlmr
        self.video_pool = TemporalAttentionPool(VIDEO_DIM)
        self.audio_proj = nn.Sequential(
            nn.Linear(AUDIO_RAW_DIM, AUDIO_PROJ_DIM),
            nn.GELU(),
            nn.LayerNorm(AUDIO_PROJ_DIM),      # normalização POR RAMO: permitida
        )

        if use_bio:
            # Sem bio_ln (ARCH-08): o RobustScaler já normaliza por feature,
            # entre amostras. Um LayerNorm sobre as 16 features heterogéneas
            # normalizava por amostra e desfazia exatamente isso.
            self.bio_mlp = nn.Sequential(
                nn.Linear(BIO_INPUT_DIM, BIO_MLP_HIDDEN), nn.GELU(),
                nn.Dropout(BIO_DROPOUT),
                nn.Linear(BIO_MLP_HIDDEN, BIO_HIDDEN_DIM), nn.GELU(),
            )
            if use_viewer_emb:
                self.viewer_emb  = nn.Embedding(NUM_VIEWERS, VIEWER_EMB_DIM)
                self.viewer_proj = nn.Sequential(
                    nn.Linear(VIEWER_EMB_DIM, VIEWER_PROJ_DIM), nn.Tanh())

        self.fusion_in = fusion_input_dim(use_bio)
        # PROIBIDO acrescentar normalização aqui sobre o vetor concatenado
        # enquanto houver braços que zeram slots — ver 04/§1.1.
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_in, FUSION_HIDDEN_DIM), nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
        )
        self.head_a = nn.Linear(FUSION_HIDDEN_DIM, NUM_TASK_A)
        self.head_b = nn.Linear(FUSION_HIDDEN_DIM, NUM_TASK_B)
        self.head_c = nn.Linear(FUSION_HIDDEN_DIM, NUM_TASK_C)
        # sem cross_attn / cross_proj / cross_norm — ARCH-03 (b)

    def gradient_checkpointing_enable(self, **kwargs):
        self.videomae.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.xlmr.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, frames, audio, input_ids, attention_mask,
                bio_vec, cvpc, user_ids,
                labels_a=None, labels_b=None, labels_c=None,
                modality_dropout_prob: float = MODALITY_DROPOUT_P,
                **kwargs):
        B, device = frames.shape[0], frames.device

        # ── Modality dropout: por AMOSTRA, faixas fixas e posicionais.
        # [0,p) vídeo | [p,2p) áudio | [2p,3p) texto | [3p,4p) bio.
        # Um braço sem bio nunca usa a 4ª faixa, então as taxas de
        # vídeo/áudio/texto são idênticas em TODOS os braços de ablação
        # (invariante exigida em 04/§1.2).
        if self.training and modality_dropout_prob > 0:
            r = torch.rand(B, device=device)           # ∈ [0, 1)
            p = modality_dropout_prob
            # band: 0=vídeo, 1=áudio, 2=texto, 3=bio, 4+=nada derrubado.
            # As 4 faixas são estruturalmente idênticas — é isto que torna a
            # invariante de §1.2 legível no código, e não só afirmada no comentário.
            band = torch.clamp((r / p).floor(), max=4.0)
            keep_video = band != 0
            keep_audio = band != 1
            keep_text  = band != 2
            keep_bio   = band != 3
        else:
            keep_video = keep_audio = keep_text = keep_bio = None

        def _mask(x, keep):
            # cast no ponto de uso: evita promover bf16→fp32 no cat sob autocast
            return x if keep is None else x * keep.to(x.dtype).unsqueeze(-1)

        vid_feat = _mask(self.video_pool(
            self.videomae(pixel_values=frames).last_hidden_state), keep_video)
        aud_feat = _mask(self.audio_proj(audio), keep_audio)
        txt_feat = _mask(self.xlmr(input_ids=input_ids,
                                   attention_mask=attention_mask
                                   ).last_hidden_state[:, 0, :], keep_text)

        parts = [vid_feat, aud_feat, txt_feat]
        if self.use_bio:
            if self.use_viewer_emb:
                base = (self.viewer_proj(self.viewer_emb(user_ids[:, 0])) +
                        self.viewer_proj(self.viewer_emb(user_ids[:, 1]))) / 2
                bio_in = bio_vec - base
            else:
                bio_in = bio_vec
            parts.append(_mask(self.bio_mlp(
                torch.cat([bio_in, cvpc], dim=-1)), keep_bio))

        fused    = self.fusion(torch.cat(parts, dim=-1))
        logits_a = self.head_a(fused).squeeze(-1)
        logits_b = self.head_b(fused)
        logits_c = self.head_c(fused)
        ...
```

Notas de implementação que o escritor do arquivo novo precisa respeitar:

- `self._zeros(...)` e `ref_dtype` **desaparecem por construção** — não há mais
  tensores de preenchimento. Isso resolve, sem tocar no assunto, a ressalva do
  Agente 2 ("`_zeros` interage com autocast bf16, não mexer"): o problema deixa
  de existir porque as máscaras multiplicam tensores vivos, com cast no ponto de
  uso.
- **Fingerprints só são comparáveis dentro do mesmo par (braço, seed).** Braços
  diferentes criam módulos diferentes, em ordem diferente, com `fan_in` diferente
  na fusão → consomem RNG de forma diferente. Isso é esperado; a verificação de
  determinismo do ARCH-19 tem de ser feita por braço.
- `load_state_dict(strict=True)` passa a **detectar** troca de braço (as chaves
  `bio_mlp.*` / `viewer_*` simplesmente não existem em B0/B1). É uma proteção
  gratuita contra servir um checkpoint na configuração errada (DIV-02).
- **Mudança acoplada, obrigatória no mesmo commit (senão fica meio-aplicada e
  ninguém nota):** o `forward` acima deixou de aceitar `rt_norm`/`fix_norm`, mas
  `remove_unused_columns=False` faz o Trainer encaminhar **todas** as chaves do
  dataset e o `**kwargs` (mantido por causa da nota defensiva do ARCH-02) as
  engoliria em silêncio. Portanto, no mesmo commit: (a)
  `build_bio_vector` devolve `(bio_vec, cvpc, user_ids)`; (b)
  `EXISTVideoDataset.__getitem__` **para de emitir** `rt_norm`/`fix_norm`;
  (c) os `pop`/`to(device)` dessas chaves em `ablation_collate` e
  `run_inference` saem; (d) idem em `inference_video.py`.
- **Verificar, não assumir, a contagem de params de §1.4:** depois da reescrita,
  imprimir `count_params(model)` para cada braço e reconciliar com a tabela
  **antes** de qualquer número ir para a nota de rodapé da tese. A tabela abaixo
  é aritmética minha, feita à mão a partir das dimensões.

### 1.4 Contagem de parâmetros por braço (para a nota de rodapé da tese)

Sistema pós-ARCH-03(b), i.e. sem os 4 987 904 params da cross-attention:

| braço | fusão | params treináveis ≈ | Δ vs B2 |
|---|---|---|---|
| B0 (sem bio) | 2560 → 1024 | ≈ 5,88 M | −2,5 % |
| B1 (bio, sem viewer) | 2688 → 1024 | ≈ 6,03 M | −0,014 % |
| B2 (bio + viewer) | 2688 → 1024 | ≈ 6,03 M | — |

Confirmo a aritmética do OPT-13. A diferença B0↔B2 de ~2,5 % de capacidade é uma
nota de rodapé, e é **muito** mais defensável que "B0 recebeu um vetor de zeros
através da projeção partilhada".

---

## 2. ARCH-03 — abandonar a opção (a)?

**DECISÃO: REVISADO PARA — (b) é o baseline e a opção (a) sai do plano da Fase 4.
Aceito a conclusão do Agente 3, mas rejeito uma das premissas dele, e substituo-a
por um critério de reabertura.**

### 2.1 Onde concordo, e o que decide

Concordo com (b), e o argumento decisivo é o **OPT-02**: com 2 271 amostras de
treino, ter ≈ 45 % do orçamento treinável num mapa linear degenerado e apenas
≈ 20 % em LoRA é uma alocação indefensável. Somado ao precedente dos memes
(cross-attention colapsou para pesos quase uniformes com ~5 000 amostras — aqui
temos menos da metade disso), a probabilidade de a versão "de verdade" ajudar é
baixa, e o custo de descobrir isso é um braço inteiro de ablação (3 runs, ~1,4 h)
mais a diluição da narrativa da tese.

### 2.2 Onde discordo — a premissa técnica está errada, e isso precisa ficar registado

O Agente 3 diz que a opção (a) "exige manter `vid_out` `[B, 1568, 768]` vivo, o
que fecha a porta ao aumento de batch do OPT-04". **Esse tensor já está vivo
hoje**: `self.videomae(...).last_hidden_state` é exatamente `[B, 1568, 768]` e é
consumido pelo `TemporalAttentionPool`, que atende sobre os 1 568 tokens. A opção
(a) não introduz esse tensor — introduz apenas a projeção `cross_proj(vid_out)`
(`[B, 1568, 1024]`, ~25,7 MB em bf16 a B=8) e os intermediários de uma camada de
atenção com `S=1` de query. Isso é um incremento modesto sobre os 12 blocos do
VideoMAE que já processam a mesma sequência.

Registro isto porque, se a premissa ficar no arquivo como está, alguém daqui a
seis meses vai descobrir que ela é falsa e reabrir a decisão **pelo motivo
errado**. A opção (a) é rejeitada por **alocação de capacidade e precedente**,
não por impossibilidade de memória.

### 2.3 Critério de reabertura (para não virar decisão eterna)

A opção (a) só volta à mesa se houver evidência de que o vídeo contribui de todo.
Teste barato, a rodar **depois** da Fase 4, sobre checkpoints já treinados, sem
GPU adicional relevante:

```
probe_zero_video: avaliar B2 com keep_video = 0 para todas as amostras.
  - Se as métricas caírem pouco (dentro do ruído entre INIT_SEEDs),
    o ramo de vídeo não está a ser usado e cross-attention sobre tokens de
    vídeo é irrelevante → (a) permanece fechada, e isso vira um achado
    negativo publicável ("o sinal é dominado por texto/áudio").
  - Se caírem de forma consistente nas 3 seeds, então (a) passa a ser um
    braço justificado numa fase futura.
```

Atenção ao rotular: isto é uma **sonda de remoção de input** sobre um modelo
treinado **com** vídeo, não uma ablação — a mesma distinção que o ARCH-20 impõe.
A assimetria do critério é deliberada e tem de ser respeitada:

- resultado **negativo** (métricas não caem) → **fecha** a opção (a) e é, esse
  sim, um achado reportável ("o sinal é dominado por texto/áudio");
- resultado **positivo** (métricas caem consistentemente nas 3 seeds) → **não é
  resultado nenhum**; é apenas um gatilho que justifica gastar 3 runs num braço
  treinado. Nunca citar a sonda como evidência de que o vídeo ajuda.

---

## 3. ARCH-06 — "6 épocas" é provisório?

**DECISÃO: ACEITO integralmente.** O ARCH-06 fica **rebaixado a provisório**: o
*defeito* (SWA sobre a cauda de overfitting; modelo final pior que a época 4/6 em
todas as seis métricas) continua sendo fato verificado; o *número 6* deixa de ser
recomendação e passa a ser "ordem de grandeza esperada, a confirmar na run de
calibração da Fase 3".

Razão: a curva citada é do sistema pré-correção. Restaurar `q_bias` (ARCH-01),
re-extrair áudio (ARCH-04), remover a cross-attention (ARCH-03 b) e dobrar
`max_length` (ARCH-05) mudam a capacidade efetiva e a qualidade do sinal de
entrada — o ponto de overfitting move-se, e não sei em que direção.

### 3.1 O que faltava: o critério de leitura da curva

"Rodar 15 épocas e reler a curva" é ambíguo, e ambiguidade aqui vira leakage.
Especifico o critério, **a fixar antes de rodar**:

> **E = argmin(`eval_loss`)** sobre as 15 épocas da run de calibração, com
> `INIT_SEED=42` e o split fixo do OPT-06. Congelar E e usá-lo, sem alteração,
> em todas as runs da Fase 4.

Por que `eval_loss` e não a ICM: escolher a época pela **mesma métrica que se
reporta** é o vazamento nº 2 do precedente dos memes. `eval_loss` é um escalar
único, independente do PyEvALL, e não é o número que vai para a tabela — é o
critério menos contaminado disponível sem um terceiro split. Na run existente
esse critério daria E = 4 (val loss 1,3616), enquanto os picos de ICM estão entre
4 e 6, ou seja, é um critério **conservador**, o que é a direção certa de erro.

Regras de acompanhamento, todas obrigatórias:

1. A run de calibração tem de ser rodada **com o SWA já removido** (ARCH-16). Com
   SWA ativo a curva por época não descreve o modelo final, e o exercício perde o
   sentido.
2. A run de calibração tem de ter **exatamente** a configuração da Fase 4 —
   `max_length`, batch/accum (OPT-04), split, `warmup_steps`, LR. Qualquer
   alteração posterior obriga a repeti-la (é 1 h, não é desculpa para não
   repetir).
3. Declarar na tese, textualmente: *orçamento de épocas escolhido uma vez, no
   conjunto de desenvolvimento, numa única seed, pelo mínimo da loss de
   validação, e congelado para todas as condições e seeds.*
4. Registar junto o valor das 6 métricas em E e em E±2, para documentar quão
   sensível é a escolha — sem usar isso para mudar E.

---

## 4. Custos medidos pelo Agente 3

**DECISÃO: ACEITO as duas correções, sem ressalvas.** As medições dele vêm de
artefatos deste repositório (tqdm salvo nos notebooks de extração, timestamps de
log da célula 52/57) e são mais confiáveis que a minha impressão qualitativa.

| minha afirmação original | correção aceite | efeito |
|---|---|---|
| ARCH-04: "Custo: exige re-extrair todas as features de áudio (2 524 + 674) e re-treinar" — enquadrado como caro | **~10 min** (7,50 it/s medidos × ~1,4 janelas/vídeo). O custo real é a **invalidação**, não a extração | a frase de custo do ARCH-04 sai; o finding **sobe** de prioridade dentro da Fase 2 |
| ARCH-10: "reduz de 72 para 60 chamadas ao PyEvALL" apresentado como benefício colateral | 72 chamadas = **22,9 s medidos**; a economia é de ~4 s | **retirar a economia como argumento**. Excluir `NONE` do grid justifica-se **só** por eliminar um grau de liberdade que produz predições contraditórias |

### 4.1 Findings que mudam de prioridade por causa disso

- **ARCH-04 — sobe.** Deixa de ter contrapeso de custo. Passa a ser o item de
  melhor relação sinal/custo da Fase 2 depois do ARCH-01: defeito de dados forte,
  10 min de CPU/GPU, zero código de treino afetado.
- **ARCH-09 — sobe de "prevalência não verificada" para "prevalência medida
  parcialmente"** e passa a ser **grátis**: o Agente 3 encontrou, nos outputs
  salvos do `extract_video.ipynb` (split de test), 368 linhas de erro
  `partial file` em 29 offsets distintos e 208 de `Invalid NAL unit size` em 9
  padrões — da ordem de 1–4 % dos vídeos com leitura falhada, disparando o
  fallback com dupla conversão BGR2RGB, e sistematicamente nos **últimos** frames
  (a truncagem afeta o fim do arquivo). O `Done. Skipped 0 videos.` confirma que
  o caminho catastrófico (vídeo inteiro em zeros) não disparou nesse split.
  Como a re-extração de frames já acontece na Fase 2 por causa do OPT-01, o custo
  marginal da correção é zero.
- **ARCH-14 — REVISADO DE BAIXO PARA MÉDIO**, e aceito como **pré-requisito da
  Fase 4**. O argumento do Agente 3 é melhor que o meu: numa série de 9–12 runs
  desatendidas, uma métrica que some sem erro custa uma onda inteira de GPU. A
  correção continua a mesma (retornar `nan` em vez de omitir a chave + métricas
  proxy do sklearn como diagnóstico independente).
- **ARCH-16 — o defeito continua BAIXO, mas a ação sobe.** O argumento de VRAM do
  Agente 3 (`AveragedModel` duplica 657 M params ≈ 2,6 GB durante todo o treino)
  transforma "remover o SWA" de limpeza em pré-condição do OPT-04. A remoção
  passa a estar amarrada à ação do ARCH-06, não a um item isolado.
- **ARCH-08 — REVISADO:** retiro a opção `BatchNorm1d`. O Agente 3 está certo
  (dependência de batch size, colisão com OPT-04, `update_bn` de volta no SWA,
  divergência treino/inferência). **A recomendação passa a ser única: remover o
  `bio_ln`**, sem substituto, confiando no `RobustScaler` por feature + primeira
  `Linear` do MLP. Já está refletido no código de §1.3.

---

## 5. Ordem dos índices da Task A (DIV-07) — como blindar

**DECISÃO: a convenção do vídeo é `sigmoid(logit_a) = P(YES)`, e o eixo de
probabilidades é `[P(NO), P(YES)]` — nesta ordem. Confirmado como fato, por três
pontos independentes do código.**

1. `build_soft_label_a` devolve a **proporção de votos YES** (célula 10).
2. `loss_task_a` treina `sigmoid(logits_a)` contra esse alvo via
   `binary_cross_entropy_with_logits` (célula 30) → `sigmoid = P(YES)`.
3. `compute_metrics` empilha `probs_a = [1 - p_yes, p_yes]` e
   `targets_a = [1 - label, label]`; `_format_for_pyevall` lê
   `p_no = probs[i][0]`, `p_yes = probs[i][1]` (células 41 e 37). Consistente
   ponta a ponta.

A convenção do modelo de memes é a **oposta** (`sigmoid = P(NO)`,
`probs = [p_yes, p_no]`) e também é internamente consistente. Um merge feito na
confiança de que nomes iguais significam comportamento igual inverte a Task A sem
erro, sem shape errado — só métricas ruins. O Agente 2 está certo em chamar isto
de armadilha de merge.

### 5.1 Blindagem — três camadas, todas obrigatórias

**(a) Matar a constante que mente.** `LABELS_2_1 = ['YES', 'NO']` está definida
nos dois arquivos, **nunca é indexada**, e declara a ordem **inversa** da que os
arrays realmente usam. É a raiz do risco. Substituir por uma constante que é a
ordem de verdade e passar a indexar por ela:

```python
# video_core.py
# Convenção da Task A do modelo de VÍDEO (oposta à do modelo de MEMES):
#     sigmoid(logits_a) = P(YES)
# e o eixo -1 de probs_a / targets_a é, NESTA ORDEM:
TASK_A_PROB_ORDER = ("NO", "YES")
IDX_NO, IDX_YES   = TASK_A_PROB_ORDER.index("NO"), TASK_A_PROB_ORDER.index("YES")
# LABELS_2_1 foi REMOVIDA: declarava ('YES','NO'), a ordem inversa da real, e
# nunca era indexada — ver DIV-07.
```
e em `_format_for_pyevall`:
```python
if task == '2_1':
    p_no  = float(probs[i][IDX_NO])
    p_yes = float(probs[i][IDX_YES])
```

**(b) Um self-check executável que amarra loss e formatter.** Barato,
determinístico, roda em milissegundos; deve ficar em `video_core.py` e ser
chamado na célula de sanity checks e no início de `inference_video.py`:

```python
def selfcheck_task_a_convention():
    """Falha se a convenção da Task A for invertida em QUALQUER das duas pontas.
    Um merge com memes_core.py inverteria uma delas silenciosamente."""
    # ponta 1 — a loss: logit muito positivo tem de casar com "todos votaram YES"
    assert loss_task_a(torch.tensor([10.0]), torch.tensor([1.0])).item() < 1e-3
    assert loss_task_a(torch.tensor([-10.0]), torch.tensor([1.0])).item() > 5.0
    # ponta 2 — o formatter: probs [P(NO), P(YES)] = [0, 1] tem de virar "YES"
    probs = np.array([[0.0, 1.0]])
    assert _format_for_pyevall(probs, ["x"], '2_1', 'hard')[0]["value"] == "YES"
    soft = _format_for_pyevall(probs, ["x"], '2_1', 'soft')[0]["value"]
    assert soft["YES"] == 1.0 and soft["NO"] == 0.0
    print("✓ convenção Task A (sigmoid = P(YES), eixo [NO, YES]) verificada")
```

Se alguém copiar a versão dos memes por cima, o primeiro `assert` da ponta 2
falha imediatamente. Se alguém inverter só a loss, falha a ponta 1.

**(c) Nunca fundir `video_core.py` com `memes_core.py`.** Concordo integralmente
com o "Perigo nº 1" do Agente 2. Se algum dia existir um `exist_core.py`, a
convenção da Task A tem de ser **parâmetro explícito e obrigatório** da função
(sem default), e a migração precisa de um teste que compare os 6 JSONs gerados
antes e depois byte a byte.

---

## 6. Seeds não pareadas (OPT-06 / OPT-07)

**DECISÃO: ACEITO a separação `SPLIT_SEED` (fixo) × `INIT_SEED` (varia).**
Manter holdout repetido e ajustar só a análise seria a escolha errada aqui.

### 6.1 Por quê

O que a tabela de ablação precisa responder é **"a condição X ajuda?"**, e a
comparação relevante é **entre condições, dentro de uma seed**. Com o split fixo,
as 9 runs partilham exatamente o mesmo conjunto de validação e o delta por
`INIT_SEED` é pareado — que é o desenho do precedente dos memes e o único
compatível com o teste correto (delta por seed + consistência de sinal, OPT-07 b).
Com holdout repetido, cada delta é medido sobre um val diferente, a variância do
delta incorpora variância de amostra, e o número de seeds (3) é pequeno demais
para separar as duas fontes. Não existe correção de análise que recupere
pareamento perdido no desenho.

### 6.2 Especificação

```python
# video_core.py / célula de configuração
SPLIT_SEED = 42       # FIXO. Define o conjunto de validação. NUNCA muda.
INIT_SEED  = 42       # varia em {42, 7, 2025}: init dos pesos + ordem dos dados
VAL_FRACTION = 0.20   # OPT-11: 80/20 → ~505 amostras de validação
```
- O split usa **literalmente** `train_test_split(..., random_state=SPLIT_SEED)`.
  `train_test_split` com `random_state` inteiro constrói um `RandomState` local,
  portanto **não** consome nem depende do RNG global — o split fica de facto
  desacoplado de `set_all_seeds(INIT_SEED)`. Isso é o que torna a separação
  limpa, e não apenas cosmética.
- `set_all_seeds(INIT_SEED)` continua a ser chamado imediatamente antes de cada
  init de pesos (VideoMAE+LoRA, XLM-R+LoRA, `EXISTVideoModel`) e
  `seed=INIT_SEED, data_seed=INIT_SEED` no `TrainingArguments`. O protocolo de
  determinismo do ARCH-19 fica intacto.
- **O split vira artefato congelado no disco:** gravar
  `Dataset/val_split_{SPLIT_SEED}.json` **e** a lista de `id_EXIST` de treino,
  commitar, e cada run carrega a lista em vez de recalcular. Assim "as 9 runs
  partilham o mesmo val" é verificável, não uma promessa.
- **`SPLIT_SEED` é escolhido antes de olhar qualquer resultado e nunca é
  re-sorteado.** Re-sortear porque os números saíram feios é split-shopping.

### 6.3 Duas consequências que precisam entrar na tese

1. **Todos os resultados passam a ser condicionais a um único split.** É o preço
   do pareamento. Mitigação: o 80/20 do OPT-11 (505 em vez de 253 amostras) torna
   esse único sorteio bem menos volátil, e é **pré-requisito** do ARCH-11 (b)
   (dividir o val em metades para o ajuste de limiar não é viável a 253 amostras
   — concordo com o Agente 3, entram juntos).
2. **A estratificação atual é insuficiente, e isto é um portão, não um conselho.**
   Hoje o split é estratificado só pela Task A. Com 6 classes esparsas na Task C:

   > **Portão (dono: quem escrever o arquivo novo; momento: antes de commitar
   > `Dataset/val_split_42.json`).** Imprimir a contagem de positivos por
   > categoria da Task C no val candidato. Se alguma categoria ficar com contagem
   > de um dígito, refazer o split estratificando por chave conjunta
   > (Task A hard × "tem alguma categoria da Task C") — e só então congelar.
   > Depois de congelado e commitado, o split **não pode mais ser tocado**;
   > qualquer mexida posterior é split-shopping e invalida as 9 runs.

### 6.4 Sobre `run_ablations_multi_seed`

Concordo com o OPT-07: com o split fixo, o defeito de vazamento dessa função
(dataset único + `model_fn(seed)` com splits diferentes) **desaparece por
construção**. E concordo com a segunda parte: substituir a agregação
`np.mean/np.std` por delta pareado + consistência de sinal. Acrescento, do meu
lado, que com o ARCH-20 aplicado essa função deixa de carregar flags — ela passa a
iterar sobre **checkpoints** (condição × seed), e o `run_inference` próprio do
harness pode ser apagado em favor de um caminho único de avaliação (OPT-08), o que
resolve de uma vez o BUG-H, o DIV-05 e o risco do `ablation_collate`.

---

## 7. Mudanças de posição e decisões pendentes que me foram endereçadas

### 7.1 Findings meus que mudam

| finding | mudança |
|---|---|
| **ARCH-03** | (b) confirmado como baseline; **(a) sai do plano**, com critério de reabertura explícito (§2.3). A justificativa "custo de memória" é substituída por "alocação de capacidade + precedente" — a premissa de memória do OPT-02/OPT-03 está errada e ficou registada como tal |
| **ARCH-04** | remover o enquadramento de custo alto; ~10 min. Sobe de prioridade |
| **ARCH-06** | **o número 6 vira provisório**; passa a depender da run de calibração da Fase 3, com critério `E = argmin(eval_loss)` fixado a priori (§3.1). O defeito continua fato |
| **ARCH-08** | **opção `BatchNorm1d` retirada**. Recomendação única: remover `bio_ln` |
| **ARCH-09** | prevalência deixa de ser "não verificada": 1–4 % dos vídeos do split de test com leitura falhada (medição do Agente 3). Correção passa a ser gratuita (embarcada na re-extração) |
| **ARCH-10** | retirar "72 → 60 chamadas" como benefício; justificar só por corretude. O aviso de comparabilidade do gold continua sendo o ponto principal |
| **ARCH-11** | aceito o refino do Agente 3: parte da dispersão de limiares entre seeds é **variância de amostra** (splits diferentes), não só instabilidade do procedimento. Com o split fixo (§6) a dispersão passa a ser interpretável. Opção (c) (média entre seeds) **descartada**; opção (b) adotada, amarrada ao 80/20 |
| **ARCH-14** | **BAIXO → MÉDIO**, e vira pré-requisito da Fase 4 |
| **ARCH-16** | defeito continua BAIXO; a **ação** (remover SWA) sobe e passa a estar amarrada ao ARCH-06 e a ser pré-condição do OPT-04 (libera ~2,6 GB) |
| **ARCH-20** | complementado pelo OPT-13: a especificação de construção está em §1.3. Retiro a recomendação provisória de "renomear as linhas para sondas" — concordo com o Agente 3 que a 4,1–5,0 h o re-treino é barato demais para justificar publicar sondas renomeadas. **As sondas continuam existindo, mas como análise separada** (ex.: `probe_zero_video` de §2.3), nunca na tabela de ablação |

### 7.2 Itens dos outros agentes que exigiam decisão minha — decididos aqui

| item | decisão |
|---|---|
| **SIMP-06** (`rt_norm`/`fix_norm`) | **REMOVER**, não conectar. Não pretendo usá-los como sinal auxiliar: RT já entra no `bio_vec` (índice 11, escalado pelo `RobustScaler`) e `fixations_count` no índice 6. Reconectá-los seria duplicar features, e em escala bruta. `build_bio_vector` passa a devolver `(bio_vec, cvpc, user_ids)` |
| **DIV-06** (semântica de `reaction_time` não-castável) | **UNIFICAR na semântica de `collect_bio_vectors`** (não-castável → NaN → imputa `RT_IMPUTE[lang]`), que é a correta. Verificado pelo Agente 2: 0 ocorrências em 5 048 linhas de treino → **numericamente `[EQUIV]` nos dados atuais, não exige re-treino por si só**. Obrigatório: contar e imprimir o nº de imputações por split, e rodar essa contagem sobre o JSON de **test** antes de submeter |
| **DIV-04** (arredondamento a 6 casas nos JSONs) | **UNIFICAR SEM arredondamento** (versão do treino). O arredondamento existia só para legibilidade |
| **DIV-03 / BUG-C** (`test_case` `"EXIST2025"` vs `"EXIST2026"`) | **Parametrizar, default `"EXIST2026"`, e verificar contra o gold oficial antes de submeter.** Não consigo verificar aqui. Risco real: o PyEvALL casa `test_case` entre pred e gold; se divergir, a avaliação sai vazia ou falha. Procedimento: abrir o arquivo gold oficial do EXIST 2026 e copiar a string literal; se não houver gold disponível, usar a string exigida pelas *guidelines* de submissão. **Não deduzir por analogia com 2025** |
| **BUG-G** (`num_items_in_batch` / médias por micro-batch) | **NENHUMA AÇÃO — impacto exatamente zero, verificado.** Contei os sentinelas no `EXIST2026_training.json`: **0 amostras** com todos os votos `UNKNOWN` nas três tasks (n=2 524). Logo `mask.sum() == B` em todo micro-batch, todas as médias mascaradas são sobre a mesma contagem, e a média não ponderada entre micro-batches é exatamente correta. O maquinário de sentinelas continua necessário para robustez em test não inspecionado, mas não há viés a corrigir. Isto fecha o BUG-G |
| **BUG-A / SIMP-02 / ARCH-12** (dois diretórios de checkpoint) | `RUN_DIR` único derivado de `(INIT_SEED, condição)`; a validação de checkpoint passa a ler o que a gravação escreveu e a rodar de facto. Combinar com OPT-03 (salvar só treinável + LoRA), **com a ressalva do ARCH-01**: os `q_bias`/`v_bias` restaurados são *frozen*, não entram no state_dict enxuto, e portanto a restauração tem de ser reaplicada no **load** — senão o modelo servido diverge do treinado sem aviso |
| **BUG-B / ARCH-18** (`--scaler` opcional) | `required=True`. Sem discussão |
| **BUG-F** (zero pré-processamento de texto) | **NENHUMA AÇÃO nesta ronda.** É uma mudança de representação de entrada, não um bug: o transcript do TikTok não tem a estrutura de hashtags/menções que motivou o pipeline dos memes. Se virar hipótese, é braço de ablação depois da Fase 4, nunca no baseline |
| **SIMP-16** (dimensões no `CLAUDE.md`) | Confirmo que **o código está certo e a documentação errada**: 2688 = 1024 texto + 768 vídeo + 768 áudio + 128 bio. Corrigir o `CLAUDE.md` e **verificar o diagrama TikZ e a seção de arquitetura do paper**. Aviso adicional: depois do ARCH-03 (b) + OPT-13, a linha correta passa a ser **"2688→1024 (B1/B2); 2560→1024 (B0)"**, e o diagrama não pode mais mostrar a cross-attention |
| **SIMP-01** (braço fantasma + tabela de uma seed) | Concordo com a remoção e com o `save_for_paper(csv_path=...)` obrigatório. Redundante com ARCH-13 |
| **ARCH-15 / SIMP-08** (`warmup_ratio`) | Migrar para `warmup_steps=0.1` é opcional e **tem de entrar na Fase 1**, antes da run de calibração — nunca no meio de uma série. Confirmar equivalência imprimindo `training_args.get_warmup_steps(total_steps)` antes e depois |

### 7.3 O que continua fora da minha alçada

- **OPT-04** (batch 8 / sem gradient checkpointing): concordo com o
  sequenciamento (ARCH-07 antes, OPT-01 junto) e com o `[MUDA]`. A escolha final
  do batch é do Agente 3, mas tem de estar **congelada antes da run de
  calibração** — mudá-la depois invalida o orçamento de épocas.
- **OPT-11** (80/20 vs k-fold): decisão de protocolo. Aceito 80/20 como
  pré-requisito do ARCH-11 (b).
- **OPT-09** (triagem com backbones congelados): aceito a lista de exclusão dele
  como está. Acrescento que a triagem **não** pode ser usada para decidir a
  remoção do `bio_ln` (ARCH-08) e depois reportá-la como resultado — é decisão de
  desenho, tomada por argumento, não por busca no dev set.
