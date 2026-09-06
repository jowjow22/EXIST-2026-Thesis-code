# 14 — Por que o ramo de vídeo contribui tão pouco? Hipóteses ordenadas

**Data:** 2026-08-09.
**Checkpoint analisado:** `run_B2_bio_viewer_seed42_B15/` (meta: `restore_biases: true`,
`frames_version: v2`, `audio_version: v2`, 15 épocas, `fusion_in: 2688`). As métricas
"normal" da sonda de permutação batem exatamente com `final_metrics.json` deste run
(A 0,5536/0,5058 · B 0,4968/0,3947 · C 0,3436/0,1514), então as sondas foram medidas
sobre este checkpoint.
**Regra deste documento:** tudo o que é **FATO** tem fonte (`arquivo:linha` ou número
fornecido/medido); o resto está marcado **HIPÓTESE**. Nenhum número foi inventado.

---

## 0. Sumário executivo

O padrão observado — permutar frames derruba **Task B** (−0,076 hard) muito mais que
A (−0,009) e C (~0), e permutar machuca 3,5× mais que zerar `vid_feat` — **não é o
padrão de um ramo morto**. Um ramo morto daria ~0 nas três tasks e permutar ≈ zerar.
O ramo de vídeo carrega *alguma* informação alinhada ao rótulo, concentrada na task de
**intenção da fonte** (B), e quase nada nas tasks de **conteúdo** (A, C).

A leitura mais econômica dos dados (H1 + H2 abaixo): o VideoMAE, com ~0,69 M de
parâmetros adaptáveis e pré-treino de reconstrução de pixels, só consegue entregar
**aparência/formato** (talking-head vs. slideshow vs. reação; cenário; enquadramento)
— e formato é exatamente o que discrimina DIRECT (a fonte fala a mensagem em primeira
pessoa, para a câmera) de JUDGEMENTAL (a fonte comenta/critica material de terceiros).
O **conteúdo** sexista está na fala, que já chega ao modelo pela transcrição
(ramo de texto) — então A e C não precisam do vídeo. Se isso se confirmar, **não há
bug**: há um teto de informação visual, e o resultado é um achado publicável, não um
defeito.

Importante: **o ARCH-01 já está corrigido neste checkpoint** (`meta.json`:
`"videomae_qv_biases": "restored_at_build_not_in_checkpoint"`). O candidato nº 1
histórico para "vídeo fraco" estava ativo aqui e o vídeo continua fraco — o q_bias
não era (sozinho) a explicação.

Medições novas desta sessão (CPU, sobre `trainable.safetensors` — ver §6):
- A fusão **não** suprime o slot de vídeo: norma média de coluna 0,749 (vídeo) vs
  0,630 (texto) vs 0,435 (áudio) vs 0,504 (bio).
- O LoRA do VideoMAE moveu os pesos só **~2 %** em norma relativa
  (‖ΔW‖_F ≈ 0,80 vs ‖W_query‖_F ≈ 38).
- O `TemporalAttentionPool` **treinou** (pesos 2–3× acima da escala de init) — não
  ficou parado no init.

---

## 1. Fatos de base (com fonte)

| # | Fato | Fonte |
|---|---|---|
| F1 | Sonda de permutação: Δ médio A hard −0,009 · A soft −0,019 · **B hard −0,076** · **B soft −0,056** · C hard +0,003 · C soft −0,001 | números fornecidos (2 derangements independentes, val=505) |
| F2 | Zerar `vid_feat` pós-pooling: B hard 0,4749 (−0,022) → permutar machuca 3,5× mais que remover | número fornecido |
| F3 | Zerar o tensor `frames` **pós-normalização** = imagem cinza (pixel médio ImageNet) → embedding constante ≠ 0 → sonda enganosa | análise fornecida; consistente com `load_frames`, `video_core.py:452-483` |
| F4 | O modality dropout de treino zera `vid_feat` (multiplicação por máscara pós-pooling) em ~5 % das amostras — a mesma operação da sonda de zerar | `video_core.py:1108-1129` (band 0 = vídeo, `MODALITY_DROPOUT_P=0.05` em `video_core.py:261`) |
| F5 | Pooling: atenção escalar (softmax) sobre **todos** os tokens do `last_hidden_state` do VideoMAE, saída = combinação convexa 768-d | `TemporalAttentionPool`, `video_core.py:949-967`; chamada em `video_core.py:1128-1129` |
| F6 | 16 frames, 224², tubelet 2 → 8×(224/16)² = **1 568 tokens** | `NUM_FRAMES=16`, `video_core.py:229`; num_patches=3136 p/ 32 frames verificado na VM (CLAUDE.md) ⇒ 1568 p/ 16 |
| F7 | LoRA r=16, α=32, dropout 0,05, `target_modules=["query","value"]`, **o mesmo config para os dois backbones** | `video_core.py:312-315`, `make_lora_config`, `video_core.py:1167-1176` |
| F8 | Backbones congelados fora do LoRA (PEFT); biases q/v restaurados são frozen | `build_backbones`, `video_core.py:1246-1266`; `restore_videomae_qv_biases`, `video_core.py:1179-1243` |
| F9 | Fusão: `cat([vid 768, aud 768, txt 1024, bio 128]) → Linear(2688,1024) → GELU → Dropout`, sem norm pós-cat; 3 cabeças lineares partilham o mesmo `fused` | `video_core.py:1041-1053`, `1135-1151` |
| F10 | Texto = CLS cru do XLM-R-large (1024-d, sem projeção/LN); áudio = Linear+GELU+**LayerNorm** (768-d); vídeo = pooling sem norm | `video_core.py:1128-1133`, `1016-1020` |
| F11 | Memes: CLIP ViT-L/14 com as **4 últimas camadas inteiras destreinadas** (não LoRA), CLS pré-treinado contrastivo, projeção 1024→512+LN | `../EXIST_MEMES/memes_core.py:481-497`, `570-575` |
| F12 | Gap mediano entre frames consecutivos 1,21 s (p10 0,54 / p90 3,85); pré-treino Kinetics ~0,13 s → regime ~8,6× mais esparso; razão p90/p10 = 7,2× dentro do dataset | números fornecidos / CLAUDE.md (⚠ os números do pré-treino não foram confirmados em fonte primária) |
| F13 | 1,0 % dos vídeos de treino fisicamente truncados | número fornecido |
| F14 | Contagem de parâmetros treináveis por bloco (medida no checkpoint, §6): fusion 2 753 536 · xlmr-LoRA 1 572 864 · audio_proj 985 344 · **videomae-LoRA 589 824** · video_pool 98 561 · bio 19 664 · heads 10 250 → total 6 030 043 | medido nesta sessão em `trainable.safetensors` (167 tensores) |
| F15 | Loss: pesos A 1,0 / B 0,7 / C 0,7; Task C usa `.mean(dim=-1)` (gradiente ~1/6 por unidade de erro vs B) | `video_core.py:272-274`, `1334-1349` |
| F16 | Task B soft-label: `'-' → NONE`; classes NONE/JUDGEMENTAL/DIRECT; `eval_task_b_fmeasure_hard = NaN` no checkpoint | `video_core.py:674-687`; `run_B2_bio_viewer_seed42_B15/final_metrics.json:11` |

---

## 2. Hipóteses, ordenadas por (plausibilidade × barateza do teste)

> Convenção de custo: "CPU-min" = roda na máquina local ou VM sem GPU;
> "GPU-min" = forward passes na VM (val 505 ≈ 2–4 min de forward completo do modelo,
> pela referência de época de treino ~340 s para 2 019 amostras com backward);
> "run" = treino completo (~35–60 min para 6 épocas).

---

### H1 — O sinal de sexismo está na FALA; o vídeo só carrega formato/registro — e formato é informação de Task B

**Status: HIPÓTESE (a mais econômica; se confirmada, não há bug).**

**Mecanismo.** A transcrição Whisper entra pelo ramo de texto (`video_core.py:771-773`,
`tokenize_text:713-726`) — ou seja, o *conteúdo falado*, que é onde uma mensagem
sexista mora numa plataforma de vídeo falado, já está disponível numa modalidade que o
XLM-R-large processa com competência nativa. O que sobra para o visual é o que a fala
não diz: **quem** fala, **como** o vídeo é montado (primeira pessoa para a câmara vs.
reação/duet/greenscreen sobre material alheio vs. slideshow com texto), cenário,
apresentação. Essas são precisamente as marcas que separam **DIRECT** (a fonte enuncia
a mensagem) de **JUDGEMENTAL** (a fonte julga/reporta material de terceiros) — a
taxonomia da Task B é sobre a *relação da fonte com o conteúdo*, não sobre o conteúdo.

**Por que explica o padrão B ≫ A ≈ C ≈ 0.** A (é sexista?) e C (que categoria de
sexismo?) são funções do conteúdo → texto/áudio bastam; o vídeo é redundante e o
modelo aprendeu a quase ignorá-lo para essas cabeças. B precisa de um traço que a
transcrição frequentemente não tem (a transcrição de um duet crítico contém a fala
sexista citada!) → é a única cabeça com incentivo a usar o visual. E explica o
**3,5× permutar vs. zerar**: um vídeo permutado entrega um formato *coerente mas
errado* (um talking-head onde havia uma reação), que a cabeça B segue com confiança;
zerar entrega "sem sinal", condição para a qual o treino explicitamente criou robustez
(F4 — a sonda de zerar reproduz o modality dropout, então ela **subestima por
construção** a dependência do vídeo).

**Teste 1 (o mais informativo por GPU-min): escada de permutação nas outras
modalidades.** O harness da sonda já existe; trocar o tensor permutado de `frames`
para `input_ids`+`attention_mask` (e depois `audio`) é edição trivial.
*Custo:* ~10–20 GPU-min (2 derangements × 2 modalidades × val 505).
*Se H1 verdadeira:* permutar TEXTO derruba A hard para perto de 0 (dezenas de pontos,
não milésimos) e derruba B e C também; permutar ÁUDIO fica entre texto e vídeo. O
"share" do vídeo (−0,076 em B, −0,009 em A) fica pequeno comparado ao do texto em
todas as tasks exceto, proporcionalmente, B.
*Se falsa:* texto permutado causa dano comparável ao do vídeo (então o modelo não está
lendo conteúdo de nenhuma modalidade e o problema é outro — fusão/heads).

**Teste 2: para onde vão os flips de B sob permutação.** Guardar `argmax(logits_b)`
normal vs. permutado e cruzar com o gold. *Custo:* já está pago se o Teste 1 rodar
(são os mesmos forwards); análise CPU-min.
*Se H1 verdadeira:* os flips concentram-se no eixo DIRECT↔JUDGEMENTAL (e
NONE→JUDGEMENTAL), não uniformemente.
*Se falsa:* flips difusos, proporcionais às marginais.

**Teste 3 (sem GPU, com humanos): auditoria de formato.** Amostrar 20 vídeos gold
DIRECT e 20 JUDGEMENTAL da val e classificar à mão o formato (fala-para-câmera /
reação-sobre-material / slideshow / outro). *Custo:* ~30–45 min humanos.
*Se H1 verdadeira:* o formato sozinho separa DIRECT de JUDGEMENTAL bem acima do acaso.
*Se falsa:* formatos indistinguíveis entre as duas classes.

---

### H2 — Desequilíbrio de capacidade E de pré-treino: o ramo visual do vídeo tem ~1/70 da capacidade de adaptação do ramo visual dos memes, sobre features que nascem piores para semântica social

**Status: mecanismo é FATO (números abaixo); que ele seja *a causa* do teto é HIPÓTESE.**
Detalhes quantitativos na §4. Resumo do mecanismo:

1. **Adaptação:** vídeo = LoRA r=16 em q/v = **589 824** params treináveis no encoder
   (F14) + 98 561 do pooling ≈ 0,69 M. Memes = 4 camadas inteiras do CLIP ViT-L
   destreinadas ≈ **50,4 M** (estimativa por fórmula, §4) — ~73× mais. E a atualização
   efetiva medida no checkpoint é minúscula: ‖ΔW‖_F ≈ 0,80 contra ‖W‖_F ≈ 38 da matriz
   base (~2 %, rank ≤ 16 de 768).
2. **Pré-treino:** CLIP foi treinado para alinhar imagem↔texto — as features do CLS já
   são semânticas (objetos, pessoas, *texto sobreposto na imagem* — CLIP tem
   sensibilidade OCR conhecida, e um meme É texto sobre imagem). VideoMAE foi treinado
   para **reconstruir pixels mascarados** em Kinetics-400 — features de aparência/
   movimento de baixo nível, notoriamente fracas em linear probing; o próprio paper do
   VideoMAE reporta gap grande entre fine-tuning completo e sonda linear
   (⚠ afirmação de literatura, não verificada em fonte primária nesta sessão).
   Com backbone congelado + 2 % de ΔW, o sistema está muito mais perto do regime de
   "sonda linear" do que do de fine-tuning.
3. Overlays de texto do TikTok (captions queimadas no vídeo) seriam a via visual mais
   direta para conteúdo sexista — CLIP as leria parcialmente; VideoMAE, por pré-treino,
   não tem razão para codificá-las.

**Por que explica o padrão B ≫ A ≈ C.** Features de aparência/cenário são exatamente o
que sobrevive a esse regime — e aparência ≈ formato ≈ informação de B (mesma cadeia da
H1; H1 e H2 são complementares: H1 diz *o que* o dataset oferece ao visual, H2 diz
*o que* este encoder consegue capturar dele).

**Teste 1 (barato): sonda linear sobre `vid_feat` cacheado.** Um forward do checkpoint
sobre train+val guardando `vid_feat` [768] por amostra (~10–15 GPU-min), depois
regressão logística (sklearn, CPU-min): prever (a) A-hard, (b) B-hard, (c) metadados
não-semânticos — língua do vídeo e **duração** (bucketizada).
*Se H2 verdadeira:* AUC(A) ≈ 0,5–0,6 (quase nada), acc(B) modesta mas > marginal, e
os metadados (duração, língua) previsíveis com folga — i.e., o vetor codifica
aparência/estatística global, não semântica.
*Se falsa:* AUC(A) claramente acima de 0,65 — a informação está em `vid_feat` e o
problema é a jusante (fusão/heads), o que apontaria para H5/H3.
*Bônus:* prever duração a partir de `vid_feat` testa também o confound temporal da F12
(o gap de amostragem é função da duração — se `vid_feat` "sabe" a duração, o eixo
temporal do vídeo vaza a duração para o modelo, análogo visual do ARCH-04).

**Teste 2 (médio, 1 run): destravar mais capacidade só no vídeo.** Adicionar
`"dense"`/MLP aos `target_modules` do LoRA do VideoMAE (ou subir r para 64 só nele) e
treinar 1 braço com E congelado.
*Custo:* 1 run (~1 h).
*Se H2 verdadeira:* B sobe e A/C mexem pouco (mais capacidade sobre as mesmas features
de aparência), ou nada muda (teto é das features, não da capacidade — distingue as
sub-hipóteses 1 e 2 acima).
*Se falsa:* nada muda e o Teste 1 mostrou informação em `vid_feat` → olhar H5.
⚠ Não é a primeira opção — só depois da escada de permutação (H1-T1) e da sonda
linear (H2-T1).

---

### H3 — Task B é a cabeça mais frágil a QUALQUER perturbação do vetor fundido (hipótese nula da sonda)

**Status: HIPÓTESE-NULA — precisa ser descartada antes de dar peso demais ao −0,076.**

**Mecanismo.** As 3 cabeças partilham o mesmo `fused` (F9). B tem 3 classes com
DIRECT/JUDGEMENTAL minoritárias, `fmeasure_hard = NaN` no checkpoint (F16 — sugere
classe nunca predita na val), e ICM-Norm é sensível a acertos em classes raras com 505
amostras. Se muitas amostras de B estão coladas à fronteira NONE↔JUDGEMENTAL,
**qualquer** perturbação do `fused` — informativa ou não — derruba B mais que A/C.
Nesse caso o −0,076 mede fragilidade da cabeça, não informação de vídeo alinhada.
As duas permutações deram −0,086 e −0,065 (hard): direção consistente, magnitude com
variância — compatível tanto com sinal real quanto com fragilidade + sorte.

**Teste: perturbação com ruído pareado.** No mesmo harness da sonda: em vez de
permutar, substituir `vid_feat` por `vid_feat + ε`, com ε gaussiano escalado para que
‖Δvid_feat‖ médio seja igual ao da permutação (‖v_i − v_{π(i)}‖ médio, mensurável nos
mesmos forwards). *Custo:* ~5 GPU-min + CPU-min.
*Se H3 verdadeira:* ruído pareado derruba B tanto quanto a permutação → o −0,076 não é
evidência de informação alinhada; o vídeo contribui ainda menos do que parece.
*Se falsa:* ruído derruba muito menos que permutação (permutar entrega um vetor
*em-distribuição mas errado*; ruído entrega um vetor *fora de distribuição*) → a
queda é semântica, reforça H1.
Também vale rodar mais 3–5 derangements para apertar o intervalo do Δ (custo marginal).

---

### H4 — Amostragem temporal esparsa: com gap de 1,21 s, o VideoMAE vira um "bag of snapshots" — movimento não existe, só aparência

**Status: o regime esparso é FATO (F12); que ele limite a contribuição é HIPÓTESE.**

**Mecanismo.** O tubelet 2 pareia frames consecutivos para codificar movimento curto;
com 1,21 s de gap mediano (vs ~0,13 s no pré-treino), cada tubelet vê duas imagens sem
relação de movimento — o eixo temporal do encoder recebe ruído estruturado. O que
sobra utilizável é aparência estática por frame — de novo a mesma assinatura: formato
sim (B), dinâmica/gesto/conteúdo não (A/C). Além disso o gap varia 7,2× com a duração
do vídeo (F12), inserindo o análogo temporal do confound ARCH-04.

**Teste 1 (barato, eval-only): sensibilidade a jitter de subamostragem.** Os arquivos
v2 guardam **64** frames (`N_FRAMES_STORED_V2=64`, `video_core.py:236`); o loader tira
16 por `linspace` (`_subsample_uniform`, `video_core.py:441-449`). Avaliar o mesmo
checkpoint com offsets/deslocamentos diferentes dos 16 índices (ex.: `idx+1`, `idx+2`)
e medir Δ`vid_feat` e Δmétricas. *Custo:* ~10 GPU-min.
*Se H4 verdadeira (forma forte):* trocar quais frames entram quase não muda `vid_feat`
(o vetor é uma estatística de aparência global do vídeo) — a informação temporal é
nula por construção.
*Se falsa:* `vid_feat` varia bastante com o jitter e as métricas de B oscilam — há
sensibilidade temporal real.

**Teste 2 (1 run): braço de janela densa.** 16 frames CONSECUTIVOS dos 64 (~4,6 s de
janela, gap ~0,29 s — perto do regime de pré-treino), já previsto como ablação futura
no código (`video_core.py:239-244`). Exige treinar (checkpoint atual foi treinado no
regime esparso; avaliá-lo com janela densa mediria mismatch, não a hipótese).
*Custo:* 1 run (~1 h) + o E congelado da calibração.
*Se H4 verdadeira:* Δ de permutação do braço denso cresce (o vídeo passa a carregar
mais), possivelmente também em A.
*Se falsa:* nada muda → aparência era mesmo tudo o que havia a extrair (reforça H1/H2).

---

### H5 — O pooling comprime 1 568 tokens → 768-d por combinação convexa escalar, destruindo estrutura espacial e temporal

**Status: a operação é FATO (F5, F6); que ela seja gargalo é HIPÓTESE — enfraquecida
por uma medição desta sessão.** Ver §3 para a anatomia completa.

**Mecanismo.** Um peso escalar por token, softmax global, soma ponderada: a saída vive
no casco convexo dos tokens; toda a estrutura "onde/quando" é descartada; 1 mapa
768→128→1 decide sozinho o que importa, treinado com ~2 019 amostras.
Contraste com os memes: o CLS do CLIP é um agregador **pré-treinado** com objetivo
contrastivo — chega pronto; aqui o agregador nasce do zero (VideoMAE não tem CLS).

**Por que B e não A/C:** não explica sozinho o padrão (a perda de estrutura afetaria
as três tasks igualmente) — por isso está abaixo de H1–H4 apesar de barata de testar.

**Evidência desta sessão (contra a forma forte):** os pesos do pool **treinaram** —
`attn.2.weight` com ‖·‖=1,78 e std 0,157 vs ~0,58/0,051 esperados do init uniforme de
`Linear(128,1)` (≈3×), `attn.0.weight` std 0,043 vs ~0,021 do init (≈2×) (§6). O pool
não ficou parado; se colapsou, colapsou "por escolha" do gradiente.

**Teste: entropia da atenção na val.** Um forward guardando os `weights` do softmax
(`video_core.py:966`) e calculando a entropia média por amostra. *Custo:* ~5 GPU-min.
*Se colapso:* H ≈ ln(1568) = 7,36 (uniforme → pooling = média) ou H ≈ 0 (um token
domina → o "vídeo" é um único patch).
*Se saudável:* H intermediária com variância entre amostras, e os pesos por posição
temporal variando entre vídeos. Também informativo: correlacionar o perfil temporal
dos pesos com o gold de B.

---

### H6 — Vídeos truncados e outros defeitos residuais de dados

**Status: FATO que existem (F13, 1,0 %); HIPÓTESE de impacto — quase certamente marginal.**
20 amostras de treino (~1 %) não movem −0,009 vs −0,076. **Teste:** excluir os IDs
truncados da val (lista em `analysis/truncation_scan_v2.csv`) e re-rodar a sonda;
CPU/GPU-min. *Se relevante:* Δs mudam visivelmente — não se espera.

---

### Nota metodológica (não é hipótese): a sonda de zerar SUBESTIMA por construção

A sonda de zerar `vid_feat` reproduz exatamente a operação do modality dropout de
treino (F4): o modelo foi *explicitamente treinado* para funcionar com `vid_feat=0` em
5 % das amostras. O −0,022 é portanto um piso amortecido; a permutação (−0,076) é a
sonda honesta. E a sonda de frames cinza é enganosa pelo motivo já registrado (F3):
zero *pós-normalização* é o pixel médio do ImageNet, uma entrada plausível, não uma
ausência — o embedding constante resultante é absorvido como bias. Qualquer sonda
futura deve operar em `vid_feat` (permutar/ruído) ou em frames *pré*-normalização.

---

## 3. Anatomia do pooling (1 568 tokens → 768-d)

**FATOS (código):**
- `self.videomae(pixel_values=frames).last_hidden_state` → `[B, 1568, 768]`
  (16 frames / tubelet 2 = 8 posições temporais × 196 patches espaciais 14×14, F6).
- `TemporalAttentionPool` (`video_core.py:949-967`): `Linear(768,128) → Tanh →
  Linear(128,1)` produz UM logit escalar por token; `softmax(dim=1)` sobre os 1 568;
  saída `(weights * x).sum(dim=1)` → `[B, 768]`.
- O nome é enganoso: apesar de "Temporal", o softmax roda sobre os 1 568 tokens
  **espaço-temporais**, não sobre 8 posições temporais. Espaço e tempo competem no
  mesmo softmax.
- Não há CLS: o VideoMAE não tem token de agregação pré-treinado; o agregador inteiro
  (98 561 params, F14) nasce do zero e aprende com ~2 019 amostras.
- Compressão: 1 568×768 ≈ 1,2 M de valores → 768 (fator ~1 568×). A saída é uma
  combinação convexa dos tokens — nenhuma informação de posição (onde na tela, quando
  no vídeo) sobrevive, porque o peso é escalar e a soma é global.

**Contraste com os memes (FATO, `memes_core.py:570-575`):** lá o vetor visual é o CLS
do CLIP — um agregador aprendido durante um pré-treino contrastivo de 400 M de pares
imagem-texto — seguido de projeção treinável 1024→512+LN. O agregador dos memes chega
sabendo o que é semanticamente relevante numa imagem; o dos vídeos tem de descobrir a
partir de 2 019 amostras e de features de reconstrução de pixels.

**Medição desta sessão (§6):** o pool não está no init (pesos 2–3× a escala inicial),
logo a forma forte de "o pooling nunca aprendeu nada" está enfraquecida. O que falta
medir (GPU-min): a **entropia do softmax** na val — distingue "pool ≈ média uniforme"
de "pool seleciona poucos tokens" de "pool saudável" (teste da H5).

**Veredito:** a perda de estrutura é real e quantificável, mas afetaria A, B e C por
igual — não explica o padrão B-específico sozinha. Tratar como amplificador das H1/H2,
não como causa primária.

---

## 4. Desequilíbrio de capacidade entre modalidades

**Parâmetros treináveis por bloco — MEDIDOS no checkpoint (F14, §6):**

| Bloco | Params treináveis | % dos 6,03 M |
|---|---:|---:|
| fusion (2688→1024) | 2 753 536 | 45,7 % |
| xlmr LoRA (24 camadas × q,v) | 1 572 864 | 26,1 % |
| audio_proj (1280→768 + LN) | 985 344 | 16,3 % |
| **videomae LoRA (12 camadas × q,v)** | **589 824** | **9,8 %** |
| video_pool | 98 561 | 1,6 % |
| bio_mlp + viewer | 19 664 | 0,3 % |
| heads a/b/c | 10 250 | 0,2 % |

Ramo de vídeo completo (LoRA + pool): **688 385** (11,4 %). Ramo de texto: 1 572 864
(26,1 %) — 2,3× mais, simplesmente porque o XLM-R-large tem 24 camadas × 1024-d contra
12 × 768-d do VideoMAE-base, com o MESMO `make_lora_config` (F7). O ramo de áudio, que
processa um vetor pré-extraído fixo, tem 1,4× mais parâmetros treináveis que todo o
encoder de vídeo adaptável.

**Magnitude efetiva da adaptação — MEDIDA (§6):** ‖(α/r)·B·A‖_F médio por módulo:
VideoMAE 0,80 (query 0,87 / value 0,73), XLM-R 0,68 (query 0,82 / value 0,55).
Referência de escala: linhas de `query.weight` do VideoMAE têm ‖·‖₂ ≈ 1,37 (CLAUDE.md,
medido) ⇒ ‖W‖_F ≈ 1,37·√768 ≈ 38. **A atualização LoRA é ~2 % da norma da matriz base,
confinada a rank ≤ 16 de 768.** Os dois backbones moveram-se pouco em termos
relativos; a diferença de partida é que o XLM-R já sabia a tarefa (texto multilingue,
semântica social) e o VideoMAE não.

**Comparação com o ramo visual dos memes (estimativa por fórmula, não medida):**
4 camadas de CLIP ViT-L (hidden 1024, MLP 4096): atenção ≈ 4·(1024²+1024) ≈ 4,20 M;
MLP ≈ 2·(1024·4096)+5120 ≈ 8,39 M; +LNs ⇒ ~12,6 M/camada → **~50,4 M treináveis**
(`memes_core.py:487-494`), atualização full-rank. O ramo visual dos memes tem ~73× a
capacidade de adaptação do ramo visual dos vídeos — e sobre features já semânticas.
**Esta é a resposta mais direta ao "por que 1 imagem CLIP contribui muito e 16 frames
VideoMAE contribuem pouco": não é o número de frames, é 50 M full-rank sobre features
contrastivas vs 0,6 M rank-16 sobre features de reconstrução de pixels.**

**Dominância de gradiente — parcialmente REFUTADA nesta sessão:** a hipótese "a fusão
aprendeu a ignorar o slot de vídeo" é testável pelas colunas de `fusion.0.weight`, e o
resultado vai na direção CONTRÁRIA: norma média de coluna do slot de vídeo **0,749** >
texto 0,630 > bio 0,504 > áudio 0,435 (§6). A fusão dá ao vídeo pelo menos tanto peso
nominal quanto ao texto. ⚠ Caveat: peso × escala da feature = contribuição; as escalas
de `vid_feat` (pooling sem norm) vs CLS do XLM-R (sem norm) vs áudio (com LN, F10) não
são conhecidas sem um forward. **Medição em falta:** ‖·‖₂ média por amostra de
`vid_feat`/`aud_feat`/`txt_feat`/`bio` na val (5 GPU-min, cabe no mesmo forward da
sonda linear H2-T1). Se ‖txt_feat‖ ≫ ‖vid_feat‖, a dominância volta à mesa; se forem
comparáveis, o slot de vídeo tem participação nominal saudável na fusão — e o pouco
que ele contribui é porque o *vetor* é pouco informativo, não porque é ignorado
(empurra para H1/H2).

---

## 5. O que o dataset realmente contém (e como saber sem re-extração)

**HIPÓTESE central (H1):** num TikTok, a mensagem sexista é majoritariamente *falada*
(ou escrita em overlay). A fala chega ao modelo duas vezes — transcrição (texto) e
prosódia (Whisper embedding). O visual puro carrega formato, identidade e cenário.
Se isso for verdade, a contribuição pequena do vídeo em A/C **é o resultado esperado
de um sistema correto**, e o achado para a tese é "o sinal de conteúdo é dominado por
texto/áudio; o visual contribui informação de *enquadramento* específica da Task B" —
que é exatamente o tipo de contributo analítico que o paper já pratica.

**Como testar com os artefatos existentes (sem re-extração, sem treinar):**

1. **Escada de permutação completa** (H1-T1): permutar texto, áudio e vídeo (um de
   cada vez) no checkpoint B15. Dá a decomposição de dependência por modalidade nas 6
   métricas com ~20 GPU-min. É o teste com melhor razão informação/custo de toda esta
   análise e deve vir primeiro.
2. **Upper bound textual:** regressão logística (ou o próprio XLM-R via `txt_feat`
   cacheado) prevendo A-hard só com texto. Se um modelo só-texto chega perto do
   sistema completo (A hard ~0,55), o teto multimodal de A é estreito por natureza.
   CPU-min sobre features cacheadas.
3. **Sonda linear de `vid_feat`** (H2-T1) incluindo alvos de metadados (língua,
   duração) para caracterizar O QUE o vetor de vídeo codifica.
4. **Auditoria humana de formato** (H1-T3): 40 vídeos, 30–45 min, zero GPU.
5. **Cruzar flips de B com transcrição:** nos itens em que a permutação muda a
   predição de B, verificar se a transcrição contém a fala citada (casos JUDGEMENTAL
   cuja transcrição "parece DIRECT") — são os casos em que só o visual desambigua.
   CPU-min + leitura manual.

**Medição que NÃO existe e faria diferença:** qualquer quantificação de texto em
overlay (OCR) nos vídeos. Se overlays forem frequentes e carregarem a mensagem, há
sinal visual-textual que o VideoMAE não lê (H2.3) — um argumento para um braço futuro
com features CLIP por frame, mas isso é troca de modelo e fica explicitamente fora do
escopo desta análise (só se a escada de permutação + sonda linear sustentarem).

---

## 6. Medições novas desta sessão (CPU, locais, reproduzíveis)

Script: sonda sobre cópia local de
`run_B2_bio_viewer_seed42_B15/trainable.safetensors` (24 144 460 bytes, 167 tensores).
Números exatos do output:

```
fusion.0.weight [1024, 2688] — norma média/mediana/Frobenius das colunas por slot:
  video (0:768)      mean=0.7492  med=0.7154  fro=21.226
  audio (768:1536)   mean=0.4351  med=0.4243  fro=12.118
  texto (1536:2560)  mean=0.6298  med=0.6182  fro=20.443
  bio  (2560:2688)   mean=0.5036  med=0.4510  fro=5.939

LoRA ||(α/r)·B@A||_F por módulo:
  VideoMAE (24 módulos, [768,768]):  mean=0.803  min=0.628  max=1.015
      query mean=0.872 · value mean=0.733
  XLM-R   (48 módulos, [1024,1024]): mean=0.684  min=0.454  max=1.747
      query mean=0.816 · value mean=0.553

video_pool:
  attn.0.weight [128,768]  ||.||=13.508  std=0.04308   (init esperado std≈0.021)
  attn.2.weight [1,128]    ||.||=1.775   std=0.15738   (init esperado ||.||≈0.58, std≈0.051)

Contagem por bloco (soma = 6 030 043): ver tabela da §4.
```

Leituras: (a) fusão não suprime o vídeo; (b) LoRA do VideoMAE ≈ 2 % de atualização
relativa; (c) pooling treinou (2–3× a escala de init). Todas incorporadas acima.

---

## 7. Ordem de execução recomendada (tudo sobre o checkpoint B15, sem treinar)

| # | Teste | Hipóteses | Custo | Decide |
|---|---|---|---|---|
| 1 | Escada de permutação (texto, áudio) | H1 | ~20 GPU-min | quota de dependência por modalidade |
| 2 | Ruído pareado em `vid_feat` + mais derangements | H3 | ~10 GPU-min | o −0,076 é sinal ou fragilidade |
| 3 | Forward único cacheando `vid/aud/txt_feat` + normas por ramo + entropia do pool | H2, H5, §4-caveat | ~15 GPU-min | escalas, colapso do pool |
| 4 | Sondas lineares sobre os caches (A, B, língua, duração) | H2, H1 | CPU-min | o que `vid_feat` codifica |
| 5 | Flips de B: matriz de transição + cruzamento com transcrição | H1 | CPU-min | onde o visual desambigua |
| 6 | Jitter de subamostragem 16-de-64 | H4 | ~10 GPU-min | sensibilidade temporal |
| 7 | Auditoria humana de formato (40 vídeos) | H1 | 30–45 min humanos | formato ↔ intenção |
| 8 | (só se 1–4 sustentarem) braço denso 16-consecutivos e/ou LoRA maior só no vídeo | H4, H2 | 1 run cada | causa vs teto |

Os testes 1–5 cabem numa única sessão de VM (<1 h de GPU total) e, juntos, separam as
três narrativas possíveis: **(i)** "não há bug — o dataset oferece pouco ao visual e o
que oferece é de Task B" (H1 confirmada: escada mostra texto dominante, flips no eixo
DIRECT↔JUDGEMENTAL, auditoria de formato positiva); **(ii)** "o encoder não extrai o
que existe" (H2: sonda linear fraca até para B, mas auditoria de formato mostra sinal
visual claro); **(iii)** "o número da sonda enganou" (H3: ruído pareado ≈ permutação).
