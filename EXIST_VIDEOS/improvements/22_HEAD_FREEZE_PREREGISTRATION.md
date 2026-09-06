# 22 — Pré-registro: fine-tune de fase 2 com heads A/B congeladas (braço CLIP)

**Escrito em 2026-08-15, ANTES de rodar.** Nada neste documento pode ser alterado depois
de ver a primeira curva. Se a leitura tiver de mudar, a mudança entra como seção nova
datada, com o motivo — nunca por edição do que está aqui.

---

## 0. De onde isto vem

Terceira hipótese de correção testada no braço CLIP, contra a mesma anomalia:

| # | Hipótese | Resultado |
|---|---|---|
| 1 | Overtraining → selecionar `E=argmin(eval_loss)` | REFUTADA (doc 18 §9, Run 2) |
| 2 | Desequilíbrio de capacidade do encoder (4→2 camadas) | REFUTADA, direção oposta (Run 3) |
| — | Probing de heads sobre trunk congelado (diagnóstico) | INCONCLUSIVO por falha de controle (doc 21) |
| 3 | Task C sub-treinada → repesar `LOSS_W_C` 0,7→4,2 | REFUTADA: C hard −0,032 (−0,038 pós-tuning), C soft +0,009 (doc 18 §9.5) |
| **4** | **A/B "puxam" o trunk; congelá-las libera C sem custo em A/B** | **este documento** |

O WC42 produziu evidência de que a representação do trunk não carrega mais sinal de C a
ser extraído por mais peso de loss. **Este braço é um teste contra essa evidência**, não
uma tentativa de contorná-la: se C ficar flat aqui também, a evidência do WC42 é
confirmada por uma terceira via independente — resultado publicável na tese tanto quanto
o ganho seria.

---

## 1. O que exatamente se faz

**Fase 1** = a run `run_B2_bio_viewer_seed42_CLIP` já existente (15 épocas, estado final).
Não é re-treinada. É o ponto de partida dos dois braços de fase 2.

**Fase 2** = 5 épocas de continuação a partir desses pesos, em **dois braços**:

| ARM | `head_a`/`head_b` | `head_c` + trunk | Serve para |
|---|---|---|---|
| `CLIP_FT5_FREEZEAB` | **congeladas** | treinam | braço da hipótese |
| `CLIP_FT5_CTRL` | treinam | treinam | **controle de orçamento** |

A diferença entre os dois braços é o freeze e **só** o freeze — mesmo checkpoint-base,
mesmas 5 épocas, mesmas seeds, mesma `TrainingArguments`, mesmo scaler. `CTRL` existe
porque, sem ele, "freeze ajudou" seria inseparável de "5 épocas extras ajudaram" (dois
eixos numa run só — o que a regra do doc 18 §9.5 proíbe).

### 1.1 A loss de A e B **continua ativa**, com os pesos ORIGINAIS

`LOSS_W_A=1,0 · LOSS_W_B=0,7 · LOSS_W_C=0,7`. A repesagem do WC42 **não** é aplicada aqui
(um eixo por run), e o notebook assert-a explicitamente `vc.LOSS_W_C == 0.7`.

Mecanismo, porque é contraintuitivo e alguém vai querer "consertar":
`requires_grad=False` numa head impede que os **pesos dela** mudem, mas o gradiente da
loss continua fluindo **através** dela até `fusion` e o trunk. As heads A/B congeladas
viram *probes fixas*: a loss delas ancora o trunk perto de representações que elas ainda
conseguem ler. **Remover as losses de A/B é o que soltaria o trunk** — e aí A/B
desabariam, porque as heads não poderiam se readaptar. Congelar head + manter loss é a
combinação deliberada.

### 1.2 O que NÃO é

- **Não é resume de treino.** Só os pesos treináveis foram salvos; o estado do AdamW e o
  scheduler nascem do zero na fase 2 (warmup de novo, cosine de novo, 5 épocas). É
  continuação de **pesos**, não de otimização. Declarado, não escondido.
- **Não é seleção de época.** Estado FINAL da época 5 nos dois braços, mesma regra de
  simetria de todo o projeto.
- **Não é re-fit de scaler.** O `RobustScaler` vem do checkpoint-base (`load_run_clip`),
  nunca re-ajustado (BUG-B/ARCH-18).

---

## 2. Parâmetros fixados A PRIORI

| Parâmetro | Valor | Fixado porque |
|---|---|---|
| Épocas da fase 2 | **5** | decisão do autor antes de qualquer curva. O trunk já viu 15 épocas e a `eval_loss` sobe monotonicamente desde a época 2 — 5 é adaptação curta, baixo risco de overfit adicional |
| Estado reportado | **FINAL da época 5** | simetria com todo o resto do projeto; nenhum reload de argmin |
| Heads congeladas | `head_a`, `head_b` | a hipótese |
| Pesos de loss | 1,0 / 0,7 / 0,7 (originais) | um eixo por run |
| `UNFROZEN_VISION_LAYERS` | 4 | eixo refutado, não mexer (doc 18 §9.2) |
| `INIT_SEED` / `SPLIT_SEED` | 42 / 42 | idem fase 1 |
| Checkpoint-base | `run_B2_bio_viewer_seed42_CLIP` | braço CLIP oficial, **não** o WC42 |

⚠ **Escolher a duração da fase 2 depois de ver a curva seria o leakage que o projeto
proíbe.** 5 épocas está travado aqui, antes da primeira run.

---

## 3. Leitura pré-registada

Comparação primária: **`FREEZE_AB` vs `CTRL`** (freeze isolado, mesmo orçamento).
Comparação secundária: os dois vs `CLIP` (época 15) — mede o efeito de continuar
treinando, não o do freeze.

Todas as seis métricas, sempre. C hard também com limiares ajustados (protocolo
metade-ajuste/metade-reporte de `tune_task_c_thresholds`), porque foi o que revelou que a
queda de C no WC42 era real e não artefato de calibração.

| Desfecho | Leitura |
|---|---|
| C sobe em `FREEZE_AB` vs `CTRL` **e** A/B não caem | Hipótese sustentada: A/B disputavam o trunk; congelá-las libera C. Único desfecho que justifica levar o braço adiante |
| C flat entre `FREEZE_AB` e `CTRL` | Confirma o WC42 por terceira via independente: o gargalo é a representação, não a alocação de treino entre tasks. **Encerra a linha de investigação de C** |
| C sobe mas A/B caem | Não é o que o freeze promete. Mesma realocação do WC42 por outro caminho — sem valor como correção, registrar e encerrar |
| Ambos os braços melhoram tudo igual | O ganho é das 5 épocas extras, não do freeze. Levanta a questão separada de o orçamento de 15 épocas ser curto — **não** responder essa aqui |

**Nenhum desfecho autoriza** ajustar épocas/LR/pesos e rodar de novo mirando um resultado
melhor. Esta é uma medição, não uma busca.

---

## 4. Portões de execução (falha dura, não aviso)

1. `assert vc.LOSS_W_C == 0.7` antes de treinar — a célula de repesagem do WC42 não pode
   ter rodado por hábito.
2. `vcc.assert_heads_frozen(model, FROZEN_HEADS)` imediatamente antes de `trainer.train()`
   — confirma que exatamente as heads pré-registradas estão congeladas.
3. **Restaurar `requires_grad` de todas as heads antes de salvar**
   (`vcc.set_heads_trainable(model)`). `vc.trainable_state_dict` filtra por
   `requires_grad`: uma head congelada não entraria no `trainable.safetensors` e
   `load_run_clip` **não acusaria** — o checkpoint recarregaria com essa head em init
   aleatório, em silêncio. A célula de validação (`allclose(loaded, fresh)`) é a rede de
   segurança e tem de continuar no notebook.
4. `meta.json` grava `phase`, `base_run`, `base_fingerprint`, `frozen_heads`,
   `loss_w_a/b/c` — a identidade destas runs é o freeze + a proveniência.
5. **Restaurar o checkpoint-base logo após o smoke test.** `vcc.smoke_test_clip` dá um
   `optimizer.step()` REAL: ele muta os pesos carregados, e muta **diferente** nos dois
   braços (com A/B congeladas as heads não recebem o passo). Sem restaurar, `FREEZE_AB` e
   `CTRL` partiriam de pontos distintos e o controle deixaria de controlar. O notebook
   recarrega o `trainable.safetensors` do base e assert-a
   `model_fingerprint(model) == base_fingerprint` (o fingerprint percorre o `state_dict`
   inteiro, video_core.py:131 — vale com heads congeladas).
6. **Portão pós-treino, nos PESOS (seção 11.1 do notebook):** no braço `FREEZE_AB`,
   `head_a`/`head_b` têm de estar bit a bit idênticas ao base; no `CTRL`, têm de ter
   mudado; `head_c` tem de ter mudado nos dois. Verifica a afirmação central do
   experimento no artefato, não nas flags `requires_grad` — e pega a falha inversa
   (congelamento acidental no controle).

---

## 5. Custo

~1,2 h de GPU por braço, ~2,4 h os dois (5/15 do custo da fase 1, que levou 6 220 s).

---

## 6. Resultado (2026-08-16) `[VERIFICADO]`

Os dois braços rodaram na VM, 5 épocas cada, 2 068 s (FREEZE_AB) e 2 048 s (CTRL) —
custo idêntico, como esperado. `meta.json` confere nos dois: `phase: 2`,
`base_run: run_B2_bio_viewer_seed42_CLIP`, `loss_w_c: 0.7`, `selected_epoch: 5`,
`frozen_heads: ["head_a","head_b"]` / `[]`. Os portões da §4 passaram (pesos do base
restaurados após o smoke test; freeze verificado nos pesos pós-treino).

**Verificação independente nos artefatos gravados** (não inferida do notebook ter
terminado — comparação direta dos `trainable.safetensors` contra o checkpoint-base):

| | `head_a` | `head_b` | `head_c` |
|---|---|---|---|
| `FT5_FREEZEAB` == base | **True** | **True** | False |
| `FT5_CTRL` == base | False | False | False |

Lido: no braço com freeze as heads A/B estão **bit a bit idênticas** ao base (o freeze
segurou); no controle mudaram (o controle é mesmo um controle, nada congelou por engano);
`head_c` mudou nos dois (a fase 2 treinou). As heads A/B **estarem presentes** no
checkpoint do braço FREEZE também confirma que o portão de restauração de `requires_grad`
antes do save (§4.3) funcionou — sem ele, elas não teriam sido gravadas.

### 6.1 As seis métricas, estado final da época 5

| métrica | `CLIP` (ép. 15, fase 1) | `FT5_FREEZEAB` | `FT5_CTRL` | **Δ freeze (FREEZE−CTRL)** |
|---|---|---|---|---|
| A ICMNorm hard | 0,5741 | 0,5657 | 0,5625 | **+0,0032** |
| A ICMSoftNorm soft | 0,5172 | 0,4976 | 0,4883 | **+0,0093** |
| B ICMNorm hard | 0,4986 | 0,5139 | 0,4984 | **+0,0155** |
| B ICMSoftNorm soft | 0,4122 | 0,4063 | 0,3901 | **+0,0162** |
| C ICMNorm hard | 0,3640 | 0,3486 | 0,3498 | **−0,0012** |
| **C ICMNorm hard, limiares ajustados** | **0,3697** | **0,3126** | **0,3390** | **−0,0264** |
| C ICMSoftNorm soft | 0,1368 | 0,1400 | 0,1278 | **+0,0122** |

`eval_loss` (comparável entre os três — todos com `LOSS_W_C=0,7`): 2,0885 (fase 1) ·
2,0253 (FREEZE_AB) · 2,0874 (CTRL).

### 6.2 Veredito: **desfecho 2 da tabela §3 — C flat**

O freeze **preserva A e B** (+0,003 a +0,016 nas quatro métricas de A/B), o que é quase
mecânico e não era a hipótese. Para a Task C, os dois sinais se **contradizem**:
C soft +0,0122 a favor do freeze, C hard pós-tuning **−0,0264 contra**. Não há ganho de C
atribuível ao freeze — é o desfecho "C flat" do pré-registro, com ruído, não o desfecho 1.

**A hipótese 4 não se sustenta.** Congelar A/B não liberou capacidade para C; liberou
capacidade para... A e B, que já eram as tasks bem servidas.

**Terceira via independente confirmando o WC42.** Duas intervenções de mecanismos
opostos — dar 6× mais gradiente a C (WC42) e tirar a competição de A/B sobre o trunk
(este braço) — falharam em mover C na mesma direção consistente. A leitura convergente é
a do doc 18 §9.5: **o gargalo da Task C é a representação do trunk, não a alocação de
treino entre tasks.** Pelo pré-registro §3, isto **encerra a linha de investigação de C**.

### 6.3 Achado colateral: o orçamento de 15 épocas não é curto

Os **dois** braços de fase 2 ficam abaixo da fase 1 em C hard com limiares ajustados
(0,3697 → 0,3390 CTRL / 0,3126 FREEZE) e em A hard/soft. Treinar além das 15 épocas
piora, com ou sem freeze — coerente com a `eval_loss` subindo monotonicamente desde a
época 2 e com a decisão de reportar o estado final da época 15. Fecha, pela negativa, a
pergunta que o desfecho 4 do §3 deixava em aberto ("e se o orçamento fosse maior?").

### 6.4 Limitações, ditas sem suavizar

- **n = 1 seed.** Os deltas do freeze (0,003–0,026) estão na ordem de grandeza que o
  projeto já documentou como variância de seed em outros braços. Nenhum número aqui
  sustenta afirmação forte isolado; o que sustenta a conclusão é a **convergência** com o
  WC42, não a magnitude.
- **C hard sem tuning (−0,0012) e com tuning (−0,0264) discordam em magnitude.** A
  metade de reporte tem n=253 — o tuning é ruidoso por construção. Reportar os dois,
  como acima, e não escolher o que favorece a narrativa.
- **A fase 2 não é resume de treino** (§1.2): AdamW e scheduler reiniciam. Parte da
  degradação dos dois braços vs fase 1 pode vir do warmup novo, não do treino extra.
  Vale igual nos dois braços, logo **não** afeta a comparação primária — mas afeta a
  leitura secundária do §6.3, que por isso fica como indício, não prova.

### 6.5 Não repetir

Este eixo está fechado. Não tentar outras combinações de heads congeladas, outro N de
épocas, ou freeze + repesagem juntos: seriam buscas contra a métrica reportada, e as duas
rotas de "realocar treino entre tasks" já falharam por mecanismos opostos.
