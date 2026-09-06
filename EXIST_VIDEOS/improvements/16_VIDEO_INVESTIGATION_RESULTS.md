# 16 — Resultados da investigação: por que o ramo de vídeo contribui pouco? (Task 3)

**Escrito em:** 2026-08-12
**Objeto:** fecho da investigação especificada em `15_VIDEO_INVESTIGATION_HANDOFF.md` §5,
fundamentada em `14_VIDEO_CONTRIBUTION_ANALYSIS.md`.
**Estado:** investigação **COMPLETA**. Os 5 testes especificados rodaram, mais 2 propostos e
implementados durante a execução (TESTE 6 — comparação de resumos; TESTE 7 — retorno da
adaptação LoRA). Todas as rotas de correção testáveis **sem treino** foram fechadas por
medição.

**Nada foi treinado.** Nenhum número deste documento vem de run nova. Tudo vem de sondas
sobre o checkpoint `run_B2_bio_viewer_seed42_B15/` (frames v2 + áudio v2,
`restore_biases=True`, 15 épocas), sobre a val congelada de 505 amostras.

---

## 0. Convenção de marcação e fontes

Mesma convenção do doc 15:

- `[MEDIDO]` — número produzido por execução, **com o artefato que o contém**.
- `[VERIFICADO]` — conferido diretamente no código, com `arquivo:linha`, ou por aritmética.
- `[HIPÓTESE]` — interpretação. **Não escrever na tese como fato.**

**Fontes primárias deste documento** (todas reconferidas nesta sessão, não herdadas):

| artefato | papel |
|---|---|
| `EXIST_VIDEOS/analysis/probe_report.txt` (195 947 B) | log completo da run final — fonte da maioria dos números |
| `EXIST_VIDEOS/analysis/probe_cache_B15/task_b_flips.csv` (208 813 B) | 574 flips; fonte dos números do TESTE 5 recomputados aqui |
| `EXIST_VIDEOS/video_core.py` | código do sistema — fonte de todo `[VERIFICADO]` |
| `EXIST_MEMES/memes_core.py` | sistema da Task 2, comparação de capacidade |
| `run_B2_bio_viewer_seed42_B15/{meta,final_metrics}.json` | metadados e baseline do checkpoint |
| `EXIST_VIDEOS/probe_video_contribution.ipynb` | código dos 7 testes (34 células, 18 de código) |

**Números do TESTE 5 recomputados nesta sessão** a partir do CSV (o log imprime só as seis
contagens direcionais; percentagens e taxas de acerto foram derivadas de novo, não copiadas).

> **REGRA QUE NÃO MUDA** (`video_core.py:1067-1071`) `[VERIFICADO]`: nada neste documento é
> evidência de que o vídeo **ajuda**. As sondas são **gatilho** para gastar runs num braço
> treinado. Quem prova é o braço treinado. A leitura é deliberadamente assimétrica: métricas
> **não** caírem fecha a opção (achado negativo publicável); métricas caírem é só gatilho.

---

## 1. Sumário — o que a investigação estabeleceu

1. **A hipótese nula H3 (a Task B é só a cabeça mais frágil) está REFUTADA na forma forte.**
   Permutar o vídeo **dentro da mesma classe gold de B** custa −0,0002; permutar entre
   classes custa −0,0631, com magnitude de perturbação praticamente idêntica.
2. **Afirmação de TETO:** o slot de vídeo não carrega nada **além** da classe de B. Ele
   funciona como um **classificador fraco de intenção da fonte**, e nada mais.
3. **A contribuição pequena do vídeo é proporcional, não anômala.** Permutar texto custa
   −0,231 / −0,254 / −0,216 (A/B/C hard); vídeo custa −0,008 / −0,063 / +0,003.
4. **O eixo do mecanismo da H1 estava errado:** os flips concentram-se em **DIRECT ↔ NONE**
   (52,6 %), não em DIRECT ↔ JUDGEMENTAL (24,4 %, **abaixo do acaso**).
5. **O pooling não é a causa primária** (H5): trocar o agregador não recupera sinal.
6. **A adaptação LoRA do vídeo rende abaixo do ruído** no primeiro incremento.
7. **Achado transversal novo:** existe um *nuisance* de **duração** decodificável em
   **três modalidades**, e ele é **pré-treinado**, não aprendido.
8. **Sobra a H2**, na forma corrigida: **capacidade absoluta + objetivo de pré-treino** —
   não "razão texto:visão", que foi refutada.

---

## 2. Portões de validade (sem isto, nenhum número abaixo vale)

O método é **um forward + replay exato**: um `forward_hook` em `model.fusion[0]` captura o
vetor concatenado de 2 688-d como o modelo o funde; os testes 1, 2 e 5 são replayados
rodando só `fusion → heads`. Isso é válido porque em `eval()` o `forward` não tem **nenhuma
operação entre amostras** — e é **proibido por regra do projeto** haver LayerNorm/BatchNorm
depois do `cat` (`video_core.py:1041-1045`) `[VERIFICADO]`.

| portão | o que testa | resultado |
|---|---|---|
| **A** (aborta se falhar) | replay reproduz as **decisões** do forward completo | ✅ modo `bs2+bf16`: maxΔ logit **0,000e+00**, **0** flips em B, **0** em A `[MEDIDO — probe_report.txt:22-29]` |
| **B** (avisa) | métricas do forward == `final_metrics.json` | ✅ Δ = **+0,00000** nas 6 métricas `[MEDIDO — probe_report.txt]` |
| **C** (avisa) | replay + derangement == `test.ipynb` | ✅ shuf0 0,4106 (Δ −0,00001); shuf1 0,4316 (Δ +0,00003) `[MEDIDO]` |
| **§8** | `pool_learned` recomputado == slot cacheado | ✅ cosseno médio **1,000000**, mín **0,999999**, maxΔ **0,0000** `[MEDIDO]` |
| **§8b** (no-op) | `disable_adapter()` tem efeito? | ✅ move os tokens **7,24 %** da norma (cos médio 0,996492, mín 0,927426) `[MEDIDO]` |

**Portões B e C passaram**, logo os valores **absolutos** deste documento são comparáveis
lado a lado com os do doc 15 sem ressalva de reprodutibilidade.

Modos de replay descartados pela autocalibração `[MEDIDO — probe_report.txt:22-27]`:
`bs2+fp32` (maxΔ 3,8e−02, 1 flip em B), `bs64+bf16` (6,3e−02, 1 flip), `bs64+fp32`
(3,8e−02, 1 flip). **Um único bit de matmul bf16 vira um `argmax` na cabeça B** — o que é,
por si só, uma medida da fragilidade dessa cabeça.

**Baseline do checkpoint sondado** `[MEDIDO — final_metrics.json]`:

| métrica | valor |
|---|---:|
| `eval_loss` | 1,75317 |
| Task A ICMNorm hard / ICMSoftNorm soft | 0,5536 / 0,5058 |
| Task B ICMNorm hard / ICMSoftNorm soft | 0,4968 / 0,3947 |
| Task C ICMNorm hard / ICMSoftNorm soft | 0,3436 / 0,1514 |
| `proxy_task_a_auc` / `proxy_task_b_f1` / `proxy_task_c_f1` | 0,7810 / 0,6104 / 0,3331 |
| `task_b_fmeasure_hard`, `task_b_crossentropy_soft`, `task_c_fmeasure_hard` | **NaN** |

`meta.json`: `condition=B2_bio_viewer`, `arm=B15`, `restore_biases=true`,
`frames_version=v2`, `audio_version=v2`, `num_train_epochs=15`, `fusion_in=2688`,
`split_seed=42`, `init_seed=42`.

**Regra de leitura PRÉ-REGISTADA** (escrita antes de qualquer número ser visto): todo arm
reportado como **média ± amplitude** sobre 5 derangements; diferença entre arms só conta se
**exceder a amplitude**; abaixo disso a leitura honesta é **inconclusivo a n=505**.

---

## 3. Os sete testes

### TESTE 2 — Ruído pareado, 5 derangements, permutação within-class (decide H3)

Este teste vinha primeiro na ordem lógica: se o −0,076 fosse fragilidade genérica da cabeça
B, toda a investigação mudaria de objeto.

Dois arms **não estavam na spec** e foram acrescentados: `perm_within` (derangement **dentro
da mesma classe gold de B**) e `noise_within` (ruído de magnitude casada com ele). Razão:
ruído isotrópico só é decisivo numa direção — "machuca menos" não vale nada se estiver
confundido com "foi perturbado menos".

**Tabela completa, 4 arms × 6 métricas** `[MEDIDO — probe_report.txt:907-912]`:

| arm | ‖Δvid_feat‖ | A hard | A soft | B hard | B soft | C hard | C soft |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal (absoluto) | 0,000 | 0,5536 | 0,5058 | 0,4968 | 0,3947 | 0,3436 | 0,1514 |
| `perm_across` | 13,154 | −0,0079 ±0,009 | −0,0225 ±0,014 | **−0,0631 ±0,050** | −0,0527 ±0,025 | +0,0032 ±0,015 | −0,0086 ±0,019 |
| `noise_across` | 13,154 | −0,0035 ±0,039 | −0,0079 ±0,025 | −0,0353 ±0,048 | −0,0136 ±0,031 | +0,0016 ±0,019 | −0,0071 ±0,019 |
| `perm_within` | 13,119 | +0,0152 ±0,021 | +0,0012 ±0,015 | **−0,0002 ±0,021** | −0,0041 ±0,008 | +0,0210 ±0,014 | −0,0045 ±0,015 |
| `noise_within` | 13,119 | −0,0231 ±0,050 | −0,0138 ±0,025 | −0,0423 ±0,046 | −0,0196 ±0,026 | −0,0071 ±0,041 | −0,0071 ±0,012 |

Tolerância pré-registada: **0,0501**.

| comparação (B hard) | diferença | vs tolerância | licencia |
|---|---:|---|---|
| `perm_within` vs `perm_across` | **0,0629** | **>** | leitura real |
| `perm_within` vs `noise_within` | 0,0421 | ≤ | mesma direção, não sustenta sozinho |
| `perm_across` vs `noise_across` | 0,0278 | ≤ | **sem poder a n=505** |

**Leitura.** As duas permutações têm ‖Δ‖ praticamente idêntico (13,119 vs 13,154) e a única
coisa que difere entre elas é a **classe gold de B do doador** — que o modelo nunca vê. Uma
cabeça genericamente frágil não pode dar dano ~zero sob perturbação de norma 13,1.
**H3 REFUTADA na forma forte** `[MEDIDO]`.

⚠ **`perm_within` é ≈0 em TODAS as seis métricas**, não só em B hard: A hard +0,0152 ±0,021
e C hard +0,0210 ±0,014 são positivos mas com amplitude que abarca o zero. A leitura honesta
é **"≈0 nas seis"**, não "positivo". É exatamente o que a afirmação de teto prediz.

⚠ **Caveat de desenho, a declarar sempre:** `perm_within` agrupa pelo **gold** de B, logo o
arm é **oracular na construção**. É legítimo — o gold escolhe o doador, nunca entra no
modelo — mas tem de estar escrito **antes** que alguém levante.

---

### TESTE 1 — Escada de permutação por modalidade (decide H1)

Mesmo derangement (mesmas seeds) em todas as modalidades, para que as quotas sejam
comparáveis. Δ médio ± amplitude sobre 5 derangements
`[MEDIDO — probe_report.txt:1484-1489]`:

| slot | dim | ‖Δ‖ | A hard | A soft | B hard | B soft | C hard | C soft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vídeo | 768 | 13,154 | −0,0079 | −0,0225 | −0,0631 | −0,0527 | +0,0032 | −0,0086 |
| áudio | 768 | 13,066 | −0,0120 | −0,0143 | −0,0574 | −0,0417 | −0,0098 | −0,0207 |
| **texto** | 1024 | 18,596 | **−0,2313** | **−0,1985** | **−0,2541** | **−0,1921** | **−0,2160** | **−0,1514** |
| bio | 128 | 2,820 | −0,0078 | −0,0051 | −0,0207 | −0,0092 | −0,0022 | +0,0027 |

Hierarquia por task (mais nocivo → menos):
- **A hard:** texto (−0,231) > áudio (−0,012) > vídeo (−0,008) > bio (−0,008)
- **B hard:** texto (−0,254) > **vídeo (−0,063)** > áudio (−0,057) > bio (−0,021)
- **C hard:** texto (−0,216) > áudio (−0,010) > bio (−0,002) > vídeo (+0,003)

**CONFIRMA a H1 parte 1** `[MEDIDO]`: o texto domina por larga margem. A contribuição
pequena do vídeo é **proporcional, não anômala** — o conteúdo sexista chega pela transcrição.

⚠ **Caveat obrigatório:** estes Δ medem **dependência do modelo treinado**, não **teto de
informação da modalidade**. Um ramo que o treino aprendeu a ignorar dá Δ≈0 mesmo que a
modalidade contenha sinal.

---

### TESTE 3 — Escalas, contribuição na fusão, diagnóstico do pool

#### 3a. Escalas e contribuição `[MEDIDO — probe_report.txt:1504-1510]`

| slot | dim | ‖feat‖ méd | ‖feat‖ med | ‖col W‖ méd | **‖W·feat‖ méd** | **% da soma** |
|---|---:|---:|---:|---:|---:|---:|
| vídeo | 768 | 15,632 | 15,579 | **0,7492** | 18,840 | **18,3 %** |
| áudio | 768 | 23,174 | 23,084 | 0,4351 | 27,242 | 26,5 % |
| **texto** | 1024 | 31,126 | 31,127 | 0,6298 | **53,686** | **52,2 %** |
| bio | 128 | 1,888 | 1,598 | 0,5036 | 3,091 | 3,0 % |
| bias | — | — | — | — | 0,480 | — |

**Fecha o caveat da §3.4 do doc 15** (ver correção 1 na §4).

⚠ A métrica `‖W_slot·feat_slot‖` é **descritiva, não causal** — há bias e GELU a jusante, e
o pré-ativação não se decompõe linearmente na saída. Quem mede causalmente é a escada de
permutação. **Convergência notável:** a ordem das fatias (texto ≫ áudio ≈ vídeo > bio) é a
mesma da escada causal.

#### 3b. Variabilidade de `vid_feat` entre amostras `[MEDIDO — probe_report.txt:1516-1521]`

| slot | ‖média‖ | ‖std‖ | std/média | cos médio | cos p05 | cos p95 |
|---|---:|---:|---:|---:|---:|---:|
| vídeo | 12,499 | 9,458 | **0,757** | 0,640 | 0,431 | 0,811 |
| áudio | 21,070 | 9,663 | 0,459 | 0,825 | 0,648 | 0,950 |
| texto | 28,028 | 13,550 | 0,483 | 0,805 | 0,630 | 0,925 |
| bio | 0,926 | 2,326 | 2,512 | 0,101 | −0,456 | 0,815 |

**REFUTA a forma forte da H2** `[MEDIDO]`: `std/média = 0,757` — o `vid_feat` **varia** entre
amostras. O encoder distingue os vídeos; a pergunta que sobra é *o que* distingue (→ TESTE 4).

#### 3c. Entropia do pool `[MEDIDO — probe_report.txt:1526-1532]`

Entropia uniforme `ln(1568) = 7,358` nats ⇒ 1 568 tokens efetivos se uniforme.

| estatística | média | mediana | p05 | p95 |
|---|---:|---:|---:|---:|
| entropia (nats) | 1,591 | 1,599 | 0,352 | 2,941 |
| tokens efetivos `exp(H)` | 6,794 | **4,950** | 1,422 | 18,938 |
| **peso do token máximo** | 0,531 | **0,502** | 0,207 | 0,935 |
| massa do top-10 | 0,941 | **0,970** | 0,775 | 0,999 |

Entropia mediana = **21,7 % do máximo**. 5 % dos vídeos operam com **1,42 tokens efetivos**.

Perfil temporal (massa de atenção por posição; uniforme = 0,125)
`[MEDIDO — probe_report.txt:1534-1537]`:

| t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,095 | **0,232** | 0,127 | **0,032** | 0,051 | 0,164 | 0,057 | **0,243** |

`t7 + t1 = 47,5 %`; `t3 + t4 = 8,3 %`. Desvio-padrão **entre amostras** do perfil = **0,2480**
> 0,125 ⇒ não é viés posicional fixo: o pool escolhe **posição diferente por vídeo**.

⚠ **Sujeito a confound não resolvido** — ver §10.

#### 3d. Checagem empírica do reshape 1568 → (8,196) `[MEDIDO — probe_report.txt:1544-1554]`

Trocando os frames 0-1 de uma amostra pelos de outra, ‖Δhidden‖ média por bloco de 196:

| bloco | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ‖Δ‖ | **16,4362** | 2,4150 | 1,6511 | 1,2542 | 1,2099 | 1,5133 | 1,1803 | 1,3509 |

Razão bloco0 / média dos outros = **10,88×**. A ordem de flatten é `(t, h, w)` com `t` mais
externo — **reshape sustentado**, o perfil temporal é reportável (com o caveat da §10).

---

### TESTE 4 — Sondas lineares sobre os caches

Regressão logística (`C=0.05`, `StandardScaler`) ajustada na metade `fit` de
`vc.split_val_halves` (n=252) e reportada na metade `report` (n=253). **Anti-leakage por
construção.** O número lido é **AUC − AUC_nulo**, com nulo = 10 repetições de rótulos
embaralhados — é o nulo que absorve a vantagem de dimensionalidade.

Distribuição dos alvos: A_hard {0:261, 1:244}; B_hard {0:268, 1:92, 2:145};
língua {0:222, 1:283}; duração {0:254, 1:251} `[MEDIDO]`.

**AUC (AUC − nulo)** `[MEDIDO — probe_report.txt:1566-1572]`:

| alvo | vídeo | texto | áudio | bio |
|---|---:|---:|---:|---:|
| A_hard (sexista?) | 0,546 (**+0,050**) | 0,716 (**+0,212**) | 0,584 (+0,083) | 0,537 (+0,043) |
| B_hard (3 classes) | 0,560 (**+0,069**) | 0,699 (**+0,189**) | 0,629 (+0,133) | 0,560 (+0,066) |
| língua (es vs en) | 0,447 (**−0,076**) | 0,972 (+0,466) | 0,939 (+0,424) | 0,915 (+0,411) |
| duração > mediana | 0,620 (+0,117) | 0,820 (+0,323) | 0,985 (**+0,494**) | 0,863 (+0,361) |

⚠ **INCONCLUSIVO a n≈253 para diferenças abaixo de 0,05.** E o caveat geral das sondas: uma
sonda linear boa **não prova** que o modelo usa essa informação; uma sonda ruim **não prova**
que a informação não está lá em forma não-linear.

---

### TESTE 5 — Matriz de flips da Task B

Agregado sobre 5 derangements × 505 amostras = **2 525 decisões**; **574 flips (22,7 %)**,
concentrados em **231 ids únicos de 505 (45,7 %)**.
`[MEDIDO — task_b_flips.csv, recomputado nesta sessão]`

Matriz completa (linha = predição normal, coluna = predição sob permutação)
`[MEDIDO — probe_report.txt:1583-1587]`:

| normal \ perm | NONE | JUDGEMENTAL | DIRECT | total |
|---|---:|---:|---:|---:|
| **NONE** | 1288 | 48 | **134** | 1470 |
| **JUDGEMENTAL** | 84 | 304 | 57 | 445 |
| **DIRECT** | **168** | 83 | 359 | 610 |

Só os 574 que mudaram, por eixo:

| eixo | flips | % | acaso (uniforme) |
|---|---:|---:|---:|
| **DIRECT ↔ NONE** | **302** | **52,6 %** | 33,3 % |
| DIRECT ↔ JUDGEMENTAL | 140 | 24,4 % | 33,3 % |
| JUDGEMENTAL ↔ NONE | 132 | 23,0 % | 33,3 % |

Nos 574 casos que mudaram `[MEDIDO — recomputado do CSV]`:

| | acerto vs gold próprio |
|---|---:|
| antes de permutar (`pred_normal`) | **51,9 %** |
| depois de permutar (`pred_perm`) | **34,3 %** |

`pred_perm == gold do doador` = **43,0 %** contra **36,6 %** esperado se independente
(+6,4 pontos). ⚠ n efetivo ≈ 231 ⇒ ~2σ — **marginal, não usar como afirmação**.

Distribuições no CSV: `gold_b` {NONE 236, DIRECT 222, JUDGEMENTAL 116}; `pred_normal`
{DIRECT 251, NONE 182, JUDGEMENTAL 141}; `pred_perm` {NONE 252, DIRECT 191,
JUDGEMENTAL 131}; `donor_gold_b` {NONE 289, DIRECT 175, JUDGEMENTAL 110}; `lang`
{es 346, en 228}.

**Matriz de confusão de B em condição normal** (505 de 505 amostras têm voto válido)
`[MEDIDO — probe_report.txt:183-192]`:

| gold \ pred | NONE | JUDGEMENTAL | DIRECT | total |
|---|---:|---:|---:|---:|
| NONE | 210 | 21 | 37 | 268 |
| JUDGEMENTAL | 23 | 52 | 17 | 92 |
| DIRECT | 61 | 16 | 68 | 145 |
| **total** | 294 | 89 | 122 | 505 |

**As 3 classes aparecem no `argmax`** — o `fmeasure_hard = NaN` **não** vem de classe nunca
predita (ver correção 3 na §4).

---

### TESTE 6 — Comparação de resumos dos 1 568 tokens (decide H5)

Quatro maneiras de resumir os mesmos tokens: `pool_learned` (o pool treinado do sistema),
`mean_all` (média simples dos 1 568), `mean_temporal` (média dentro de cada uma das 8
posições temporais → 6 144-d), `top5` (média ponderada dos 5 tokens de maior peso do pool).
Mesmas metades, mesmo nulo, mais **8 divisões 50/50 aleatórias** para dar o ±sd.

**AUC − nulo (±sd)** `[MEDIDO — probe_report.txt:1619-1623]`:

| alvo | `pool_learned` | `mean_all` | `mean_temporal` | `top5` |
|---|---:|---:|---:|---:|
| A_hard | +0,050 ±0,020 | +0,057 ±0,028 | **+0,096** ±0,022 | +0,040 ±0,016 |
| B_hard | +0,069 ±0,024 | +0,088 ±0,018 | **+0,093** ±0,014 | +0,068 ±0,021 |
| língua | −0,076 ±0,033 | +0,055 ±0,029 | +0,011 ±0,031 | −0,089 ±0,029 |
| duração | +0,117 ±0,016 | **+0,297** ±0,022 | **+0,304** ±0,028 | +0,114 ±0,016 |

Diferenças vs `pool_learned`, contra tolerância `max(0,05 ; 2sd)`:
`mean_all` **+0,007** (A, tol 0,056) e **+0,019** (B, tol 0,050); `mean_temporal` +0,046
(A, tol 0,056) e +0,024 (B, tol 0,050). **Todas abaixo da tolerância.**

Cross-check §6 ↔ §8: as sondas sobre `pool_learned` reproduzem o TESTE 4 com **Δ = +0,0000**
nos 4 alvos `[MEDIDO]`.

**Leitura pré-registada:** empate. O seletor **não está perdendo sinal linearmente
decodificável**, e trocar o agregador **não tem alvo óbvio**. `top5 ≈ pool_learned` confirma
que a cauda de 1 563 tokens não adiciona nada. Pela regra assimétrica, isto **FECHA** a rota
de trocar o agregador — achado negativo. **H5 não é causa primária.**

---

### TESTE 7 — Retorno da adaptação LoRA no ramo de vídeo

Comparação COM vs SEM adaptador (`model.videomae.disable_adapter()`). Os `q_bias`/`v_bias`
restaurados pelo ARCH-01 são params **base congelados**, não LoRA, e permanecem nas duas
condições — é o que torna a comparação limpa. **Só resumos SEM parâmetros contam**
(`mean_all`, `mean_temporal`): o `pool_learned` foi treinado sobre tokens **adaptados**, logo
aplicá-lo a tokens base confundiria "os tokens mudaram" com "o pool está descasado".

`[MEDIDO — probe_report.txt:1668-1685]`

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

`pool_learned` (**CONFUNDIDO**, para registro): A +0,050/+0,036 (+0,014); B +0,069/+0,075
(−0,005); língua −0,076/−0,079 (+0,003); duração +0,117/+0,142 (−0,025).

**Todos os ganhos abaixo da tolerância.** Com ‖ΔW‖_F ≈ **2 %** da matriz base (`[RELATADO]`,
não reconferido nesta sessão) e efeito nos **tokens** de **7,24 %**, os tokens são
essencialmente **features de pré-treino** do VideoMAE.

**Diagnóstico de nuisance — em que os 2 % de ΔW foram gastos** `[MEDIDO]`:

| resumo | ganho máx. em RÓTULO | ganho máx. em NUISANCE (duração/língua) |
|---|---:|---:|
| `mean_all` | +0,005 | +0,001 |
| `mean_temporal` | +0,021 | +0,015 |

⚠ Isto **não decide** entre as duas causas: "falta capacidade" e "o objetivo de reconstrução
de pixels não tem o que oferecer" predizem o mesmo aqui. Mas **muda o custo esperado** do
braço de mais capacidade (§11).

---

## 4. Correções aos documentos 14 e 15

**As correções abaixo NÃO foram aplicadas nos arquivos 14 e 15** — eles ainda contêm as
afirmações erradas. Aplicá-las (ou marcar as seções com ponteiro para este doc) é a próxima
ação da §12.

Três itens (**6, 7, 8**) são correções de afirmações feitas **pela sessão anterior sobre o
próprio material dela**, não erros herdados. O item **9** é novo desta sessão e corrige uma
leitura impressa **dentro do próprio `probe_report.txt`**.

---

**1. §3.4 do doc 15 INVERTE.** `[MEDIDO]`
A afirmação *"a fusão não ignora o vídeo — norma de coluna 0,749 (vídeo) > 0,630 (texto)"* é
enganosa. Em **peso × escala**: texto **52,2 %**, vídeo **18,3 %** — o texto contribui
**2,85×**. A norma de coluna maior do vídeo mede aproximadamente o **inverso da escala da
feature** (`vid_feat` tem metade da norma de `txt_feat`: 15,6 vs 31,1) e a `Linear` compensa
por unidade. **NÃO escrever na tese que a fusão dá ao vídeo peso maior que ao texto.**
⚠ Descritivo, não causal — quem mede causalmente é a escada de permutação, que converge na
mesma ordem.

**2. O enquadramento de "desequilíbrio de capacidade" da H2 (§3.1/§4 do doc 14) não
sustenta.** `[VERIFICADO — memes_core.py:473, 489-494]`
`memes_core.py` mexe em `requires_grad` **só** na visão (`:489-494`); o XLM-R dos memes
(`:473`) é treinado **inteiro**, sem LoRA e sem freeze. Logo:

| sistema | texto treinável | visão/vídeo treinável | razão |
|---|---:|---:|---:|
| memes | ~355 M (full) | 50,4 M (4 camadas CLIP, full-rank) | **≈7×** |
| vídeo | 1,57 M (LoRA r=16) | 0,59 M (LoRA r=16) | **≈2,7×** |

**O modelo de memes é MAIS dominado por texto em termos relativos, e a visão dele contribui
muito.** A razão texto:visão **não explica** a diferença. A H2 tem de repousar em
**capacidade absoluta** (0,59 M rank-16 vs 50,4 M full-rank) **+ objetivo de pré-treino**
(contrastivo vs reconstrução de pixels).
⚠ Os 50,4 M são **estimativa por fórmula** a partir do código, não contagem em runtime
(`[RELATADO]`): por camada, atenção 4·(1024²+1024) = 4 198 400; MLP (1024·4096+4096) +
(4096·1024+1024) = 8 393 728; 2 LayerNorms = 4 096 ⇒ 12 596 224 × 4 = **50 384 896**.

**3. "`fmeasure_hard = NaN` sugere classe nunca predita" (F16/H3) está REFUTADA.** `[MEDIDO]`
As 3 classes aparecem no `argmax` em condição normal (294 NONE / 89 JUDGEMENTAL / 122
DIRECT). `proxy_task_b_f1 = 0,6104` era compatível com as duas leituras (o teto com uma
classe nunca predita seria 0,667). O NaN vem de outro motivo, **interno ao PyEvALL**.

**4. −0,076 → −0,063 ± 0,050.** `[MEDIDO]`
Com 5 derangements, `perm_across` dá **−0,0631** com amplitude **0,0501**. A estimativa de
2 seeds do doc 15 era um ponto frágil: **a amplitude é da ordem do próprio efeito.**
Reportar SEMPRE com a amplitude, nunca como valor pontual. (Portões B/C passaram, logo a
comparação lado a lado com o doc 15 é legítima.)

**5. O eixo do mecanismo da H1 está errado.** `[MEDIDO]`
A H1 afirma que "formato separa DIRECT de JUDGEMENTAL". Os flips dizem **DIRECT ↔ NONE
(52,6 %)**; o eixo D↔J tem 24,4 %, **abaixo do acaso** (33,3 %). Reescrever como: *o visual
entrega enquadramento, e enquadramento discrimina a enunciação em primeira pessoa (DIRECT)
do não-sexista (NONE)*.
`[HIPÓTESE]` Mecanismo plausível: "a fonte fala para a câmera" é visualmente distintivo;
separar "comenta material de terceiros" (JUDGEMENTAL) de NONE pelo visual é muito mais
difícil — ambos podem ser reação/duet/greenscreen. **Não escrever na tese como fato.**

**6. [CORREÇÃO DE AFIRMAÇÃO DA SESSÃO ANTERIOR] A H5 foi promovida sem base.** `[MEDIDO]`
Foi escrito que *"a H5 sai de 'amplificador' para candidata a gargalo primário"* com base no
colapso para 5 tokens. **Errado** — inferiu-se custo a partir de mecanismo. O TESTE 6 mede o
custo diretamente: `mean_all − pool_learned` = +0,007 (A) e +0,019 (B), abaixo da tolerância.
O colapso é real e **não custa** informação de rótulo linearmente decodificável. Volta ao
veredicto original do doc 14 §3: **amplificador, não causa primária.**

**7. [CORREÇÃO DE AFIRMAÇÃO DA SESSÃO ANTERIOR] A metáfora dos "5 quadradinhos" foi
sobrecorrigida.** `[MEDIDO]`
Foi dito que "5 tokens ≠ 5 pedaços da imagem, porque depois de 12 camadas de atenção cada
token é uma mistura global". **Desmentido pela medição:** o TESTE 3d mostra concentração de
**10,88×** no bloco correspondente. Os tokens do VideoMAE são **fortemente locais** —
coerente com o pré-treino de reconstruir o próprio patch, ao contrário de um CLS contrastivo.
A primeira leitura ("um recorte do vídeo") estava mais certa que a correção.

**8. [CORREÇÃO DE AFIRMAÇÃO DA SESSÃO ANTERIOR / do notebook] "retorno medido de ZERO" é
impreciso.** `[MEDIDO]`
A célula do §8b imprime isso. O correto é *"abaixo da tolerância, com padrão positivo
consistente apenas sob `mean_temporal`"*. Sob `mean_all` o efeito do LoRA desaparece
(+0,004 / +0,005); sob `mean_temporal` aparece (+0,021 / +0,017 — 0,6σ e 0,8σ).
`[HIPÓTESE]` O que o LoRA mudou pode ser dependente de posição: a média global lava, a média
por fatia preserva. ⚠ Enfraquecida por `língua` também subir +0,015 no mesmo resumo — **não**
se pode afirmar que o ganho é semântico.

**9. [NOVA — corrige o próprio `probe_report.txt`] A LEITURA impressa do TESTE 5 está errada
nas duas metades.** `[MEDIDO]`
O log imprime: *"NÃO CONFIRMA a H1: os flips são difusos / não se concentram no eixo
DIRECT<->JUDGEMENTAL. Compatível com H3 (fragilidade genérica da cabeça)."*
- *"difusos"* é falso: os flips **são** concentrados, só que noutro eixo — DIRECT ↔ NONE
  com 52,6 % contra 33,3 % de acaso. A célula interrogou apenas o eixo D↔J e parou.
- *"compatível com H3"* está refutado pelo `perm_within` = −0,0002 na mesma run.

Registrado aqui porque este documento cita `probe_report.txt` como fonte de todos os números:
sem esta nota, um leitor futuro encontra uma leitura impressa que contradiz a conclusão deste
documento, dentro do próprio artefato de origem. **A tabela de contagens do log está certa; a
LEITURA abaixo dela, não.**

---

## 5. Achados novos, não previstos em nenhum documento

**1. O *nuisance* de duração atravessa TRÊS modalidades.** `[MEDIDO]`
Decodificabilidade de "duração > mediana" (AUC − nulo): áudio **+0,494**, bio **+0,361**,
texto **+0,323**, tokens crus de vídeo (`mean_all`) **+0,297**, `vid_feat` final **+0,117**.
É um **eixo de confusão do dataset inteiro**, não de uma modalidade. O áudio decodifica
duração quase perfeitamente (AUC 0,985) `[MEDIDO]`.
`[HIPÓTESE]` A ligação causal com o ARCH-04 (`pad_or_trim` de 30 s + `.mean` sobre os 1 500
frames do encoder ⇒ fração de silêncio é função da duração) é **inferência**, não medição —
plausível e coerente com os números do doc 07, mas não testada aqui.

**2. O nuisance de duração é PRÉ-TREINADO, não aprendido.** `[MEDIDO]`
Duração: **+0,297 vs +0,297** (`mean_all`), **+0,304 vs +0,306** (`mean_temporal`) —
**idêntico com e sem LoRA**. É propriedade das features pré-treinadas do VideoMAE + do
esquema de amostragem, não algo que o treino introduziu. Isso **exonera o treino** e localiza
o confound no **encoder/pipeline de extração**. É o achado mais forte da §5 para a tese: o
confound temporal documentado no `CLAUDE.md` (gap variando 7,2× entre p10 e p90 porque a taxa
de amostragem é função da duração) tem aqui a sua assinatura medida no espaço de features.

**3. O pool aprendido é um FILTRO DE NUISANCE.** `[MEDIDO]`
Duração: tokens crus +0,297 → `vid_feat` **+0,117**. Língua: +0,055 → **−0,076**. O seletor
**descarta a variável de captura**. É ponto **a favor** do pool, não contra — e mata a
preocupação de que o confound temporal vaze para a fusão: ele está nos tokens, mas o pool o
filtra.

**4. O áudio empata com o vídeo na Task B e ganha na Task A.** `[MEDIDO]`
Escada: áudio −0,0574 vs vídeo −0,0631 em B hard; áudio −0,0120 vs vídeo −0,0079 em A hard.
**Um único vetor de 1 280-d pré-extraído e congelado iguala 16 frames processados por um
encoder adaptável.** Vai direto no argumento de **capacidade absoluta** da H2.

**5. `vid_feat` NÃO codifica língua.** `[MEDIDO]`
AUC 0,447, ou **−0,076 abaixo do nulo**. O canal visual **não traz** o confound de língua que
arruinou a normalização per-viewer (texto +0,466, áudio +0,424, bio +0,411). Achado limpo e
reportável.

**6. A cabeça B é instável — e isso convive com o vídeo carregar informação real.** `[MEDIDO]`
**231 de 505 itens (45,7 %)** trocam de predição sob permutação de vídeo, e a métrica só cai
0,063: muitos flips **se cancelam**. É vindicação parcial do *espírito* da H3 (fragilidade),
apesar de a **forma forte** estar refutada pelo `perm_within`. As duas coisas convivem: a
cabeça é instável **E** o vídeo carrega informação de classe. Reforço independente: na
autocalibração do replay, **um único bit de matmul bf16** basta para virar 1 `argmax` de B.

**7. Nos 574 flips, o acerto cai de 51,9 % para 34,3 %** (−17,6 pontos). `[MEDIDO]`
O slot de vídeo estava entregando classe **correta** nesses itens. Junto com
`perm_within ≈ 0`, fecha o argumento: **informação alinhada ao rótulo, não ruído.**

---

## 6. Conclusões substantivas

- **H3 (forma forte) REFUTADA.** `[MEDIDO]` `perm_within` (−0,0002 ±0,021) e `perm_across`
  (−0,0631 ±0,050) têm ‖Δ‖ praticamente idêntico (13,119 vs 13,154) e diferem em
  **0,0629 > 0,0501**. A única coisa que difere entre os arms é a **classe gold de B do
  doador**, que o modelo nunca vê.
  ⚠ **Caveat oracular, obrigatório declarar:** `perm_within` agrupa pelo **gold** de B, logo
  o arm é **oracular na construção**. É legítimo — o gold escolhe o doador, nunca entra no
  modelo — mas tem de estar escrito antes que alguém levante. Sem esta frase, a refutação da
  H3 fica vulnerável na defesa.

- **AFIRMAÇÃO DE TETO.** `[MEDIDO]` Dano ≈0 dentro da classe, **nas seis métricas**, significa
  que o slot de vídeo **não carrega nada além da classe de B**. Se carregasse intensidade,
  categoria de C ou detalhe de enquadramento, trocar dentro da classe custaria algo.
  **O ramo de vídeo funciona como um classificador fraco de intenção da fonte, e nada mais.**
  Esta é a afirmação mais forte da investigação e a que mais muda a narrativa da tese: o
  problema não é um defeito a corrigir, é um **teto de informação a documentar**.
  ⚠ **Escopo exato do teto, a escrever junto sempre:** é teto do **vetor `vid_feat` deste
  checkpoint**, não teto da **modalidade vídeo**. O próprio log declara a limitação: *"Δ mede
  DEPENDÊNCIA do modelo treinado, não teto de informação da modalidade. Um ramo que o treino
  aprendeu a ignorar dá Δ≈0 mesmo que a modalidade contenha sinal."* Nenhuma sonda sobre
  features congeladas pode fechar isso — só um braço treinado com outro encoder.

- **H1 confirmada na parte 1, corrigida no eixo.** `[MEDIDO]` O texto domina (−0,231 /
  −0,254 / −0,216). A contribuição pequena do vídeo é **proporcional, não anômala**. O eixo do
  mecanismo é DIRECT↔NONE, não DIRECT↔JUDGEMENTAL.

- **H5 não é causa primária.** `[MEDIDO]` TESTE 6: nenhum resumo alternativo bate o pool
  acima da tolerância.

- **Sobra a H2, na forma corrigida:** capacidade **absoluta** + **objetivo de pré-treino**.
  `[MEDIDO]` O melhor que **qualquer** resumo de vídeo decodifica é **+0,096** (A, com
  `mean_temporal`); o texto decodifica **+0,212**. Mesmo com o melhor agregador testado, o
  vídeo fica em **~metade** do texto.
  ⚠ `[HIPÓTESE]` A separação entre "falta capacidade" e "o objetivo de reconstrução de pixels
  não tem o que oferecer" **não foi decidida** — os dois predizem exatamente o mesmo em todos
  os testes rodados. Decidir exigiria trocar de encoder, rota fechada por custo/benefício.

---

## 7. Correções de arquitetura para a tese e para o paper

Estas quatro são **independentes de qualquer resultado da investigação**: valem mesmo que
tudo acima fosse descartado. São descrições do que o código faz.

**1. O `TemporalAttentionPool` NÃO é temporal.** `[VERIFICADO — video_core.py:965-967]`
```python
logits = self.attn(x)                       # x = last_hidden_state [B, 1568, 768]
weights = torch.softmax(logits, dim=1)      # softmax sobre os 1568 tokens
return (weights * x).sum(dim=1)             # [B, 768]
```
A entrada vem de `video_core.py:1128-1129` (`self.videomae(...).last_hidden_state`), logo os
1 568 são **tokens espaço-temporais** (8 posições temporais × 196 patches de 14×14), não
frames. Espaço e tempo competem no **mesmo softmax**, e o peso é um escalar por token —
nenhuma informação de *onde na tela* ou *quando no vídeo* sobrevive à soma global.
**Não depende de nenhuma suposição sobre ordenação de tokens:** `dim=1` é o eixo dos 1 568.
A docstring (`video_core.py:950-953`) está errada nas **duas** afirmações — *"sobre a
dimensão temporal"* e *"cada token de frame"*. Corrigi-la é a **única** edição de código
indicada por esta investigação, e é documental.

**2. NÃO é pooling, é quase um SELETOR.** `[MEDIDO]`
Peso mediano do token máximo = **0,502**; massa mediana do top-10 = **0,970**; **5 tokens
efetivos** de 1 568; 5 % dos vídeos com **1,42**. Entropia mediana = **21,7 %** do máximo.
Uma média ponderada em que um único elemento tem peso 0,5 **não é agregação**. Descrever como
"pooling por atenção" na tese é impreciso; descrever como **seletor esparso** é o que os
números sustentam.

**3. Os tokens do VideoMAE são LOCAIS (10,88×), não resumos globais.** `[MEDIDO]`
Ao contrário do CLS contrastivo do CLIP, o pré-treino de reconstrução por patch mantém os
tokens locais mesmo após 12 camadas de self-attention. Logo o token eleito pelo seletor é um
**recorte** do vídeo — a metáfora do "quadradinho" é defensável.

**4. A fusão dá ao vídeo 18,3 %, não "mais que ao texto".** `[MEDIDO]`
Ver correção 1 da §4. O diagrama TikZ e a seção de arquitetura do paper herdaram a leitura
errada da norma de coluna.

**Complemento — o meio do vídeo é ignorado.** `[MEDIDO]` `t7 + t1 = 47,5 %` da massa de
atenção; `t3 + t4 = 8,3 %`. Desvio-padrão entre amostras 0,2480 > uniforme 0,125 ⇒ **não** é
viés posicional fixo: o pool escolhe posição diferente por vídeo.
⚠ **Sujeito ao confound de frames repetidos — ver §10. Não levar para a tese antes de
resolver.**

---

## 8. Lição metodológica das sondas (material publicável por si só)

1. **Zerar entrada normalizada engana.**
   `[MEDIDO — doc 15 §2.3, origem test.ipynb; NÃO reconferido nesta sessão]`
   Zero pós-normalização = pixel médio do
   ImageNet = imagem cinza plausível → embedding **constante mas não-nulo** → absorvido como
   viés fixo. Métricas quase não mudam e algumas **melhoram** (A hard +0,008, C hard +0,010),
   sugerindo falsamente que o vídeo não contribui.
2. **Zerar `vid_feat` pós-pooling subestima por construção.**
   `[MEDIDO — doc 15 §2.2, origem test.ipynb; NÃO reconferido nesta sessão]`
   Task B hard 0,4749, ou −0,022 contra −0,063
   da permutação: reproduz exatamente a operação do **modality dropout**, para a qual o treino
   criou robustez. **Permutar machuca 3,5× mais que remover.**
3. **Permutação preserva a distribuição de entrada** e mede **alinhamento**, não presença.
4. **Ruído isotrópico é o controle ERRADO — e foi medido como tal.** `[MEDIDO]`
   `perm_across` vs `noise_across` = 0,0278, abaixo da tolerância de 0,0501: **sem poder a
   n=505**. "Indistinguível" **≠** "confirmado". A árvore de leitura original do notebook
   caiu nessa armadilha e imprimiu "CONFIRMA H3"; foi corrigida antes de a conclusão ser
   escrita.
5. **O controle que discrimina é permutação condicional à classe com magnitude casada.**
   `[MEDIDO]` ‖Δ‖ 13,119 vs 13,154; dano −0,0002 vs −0,0631. É este par que decide, e é
   barato: nenhum treino, um forward.
6. **Nulo por permutação é obrigatório em sonda linear com n≈252 e d até 6 144.** Sem ele,
   "duração é previsível" pode ser artefato de dimensionalidade. O número reportado é sempre
   **AUC − AUC_nulo**, com 10 repetições de rótulos embaralhados.
7. **A regra assimétrica funciona** (`video_core.py:1066-1071`) `[VERIFICADO]`: métricas não
   caírem **fecha** a opção (achado negativo publicável); métricas caírem é **gatilho**, nunca
   evidência. Quem prova é um braço treinado.

---

## 9. Rotas FECHADAS, com a razão de cada uma

| rota | estado | razão |
|---|---|---|
| **Trocar o agregador do vídeo** | ❌ fechada | TESTE 6: `mean_all`, `mean_temporal` e `top5` empatam com `pool_learned` dentro da tolerância. `top5 ≈ pool_learned` mostra que **não há reservatório de sinal na cauda de 1 563 tokens**. `[MEDIDO]` |
| **"A fusão ignora o vídeo"** | ❌ morta | TESTE 3a: o slot de vídeo pesa **18,3 %** da soma de `‖W·feat‖`. Não é descarte. `[MEDIDO]` |
| **ARCH-01 (`q_bias`/`v_bias` zerados)** | ❌ já corrigido | `meta.json` do B15 tem `restore_biases=true` e o vídeo continua fraco. `[VERIFICADO]` |
| **Mais capacidade de adaptação (primeiro incremento)** | ⚠ retorno medido abaixo do ruído | TESTE 7: +0,004/+0,005 (`mean_all`), +0,021/+0,017 (`mean_temporal`), todos < tolerância. Decisão do autor em aberto — §11. `[MEDIDO]` |
| **Cross-attention sobre os 1 568 tokens** | ❌ fechada | O critério de reabertura de `video_core.py:1061-1064` **foi atendido** (o `perm_within` provou que o vídeo contribui), **mas o TESTE 6 fechou a rota**: não há sinal na cauda. Seria uma forma mais cara de não achar nada. Custo se algum dia voltar: query 256-d → Wq 262 144 + Wk 196 608 + Wv 196 608 + Wo 196 608 ≈ **852 k params** num dataset de 2 019 amostras — o mesmo risco que matou a versão anterior (45 % dos treináveis). **Não revisitar sem evidência nova.** |
| **Concatenar os 1 568 tokens direto na fusão** | ❌ impossível | 1 568 × 768 = 1 204 224 entradas × 1 024 = **1 233 125 376 params** só na `Linear` da fusão. |
| **Braço de 32 frames** | ❌ descartado por custo/benefício | `num_frames` **é livre** no VideoMAE (verificado na VM: carrega sem incompatibilidade, `num_patches=3136`, porque `position_embeddings` é tabela sinusoidal por fórmula e não está no `state_dict`). Mas custa 4-5 h e **o gap continua função da duração**: razão p90/p10 = 7,2× para 16, 32 **ou** 64 frames. Dobrar os frames divide o gap e **não remove o confound**. |
| **Trocar de encoder (CLIP sobre frames-chave)** | ⚠ **não fechada por medição** — descartada por custo/escopo | Nenhum teste deste documento a refuta; ao contrário, é para onde a H2 corrigida aponta (capacidade absoluta + objetivo de pré-treino). O doc 15 §6 já a registrava como a via indicada **se** a análise sustentasse — e sustenta. Está fora do escopo do que resta da tese, não fora da mesa cientificamente. Registrar assim, e não como refutada. |
| **Re-extrair features** | ❌ concluída, não repetir | v2 concluída e verificada em 2026-08-09, custou um dia. Frames: 64 uint8/vídeo, 2 524 + 674, **0 faltantes**. |
| **Assert cego único no portão** | ❌ abandonado | Confundia 3 causas distintas. Substituído por 3 portões decompostos (A aborta; B e C avisam). |
| **Replay em `bs=64` sem autocalibração** | ❌ abandonado | Causou a falha do primeiro run. Sempre autocalibrar contra os logits cacheados. |

---

## 10. Caveats que GATEIAM o uso na tese

Dois achados limpos estão bloqueados por questões não resolvidas. **Não levar para a tese sem
resolver primeiro.**

1. **O pico em `t7` (0,243) do perfil temporal pode ser artefato.** ❓ **Não sei como a
   extração v2 tratou vídeos com menos de 64 frames decodificáveis.** Se houver repetição do
   último frame no fim, o pico em `t7` é artefato de pipeline, não comportamento do pool.
   Checar em `11_EXTRACTION_IMPLEMENTATION_PLAN.md` ou no código de `extract_video_v2.ipynb`.
   **Afeta:** o complemento da §7 ("o meio do vídeo é ignorado"). **Não afeta** nenhuma outra
   conclusão deste documento.

2. **Os números de pré-treino do VideoMAE nunca foram confirmados em fonte primária.**
   Kinetics-400, 16 frames, stride 4 @ ~30 fps → clipe de ~2,1 s ⇒ ~7,6 fps, gap ~0,13 s.
   Vêm da config padrão citada no paper e **não foram verificados na fonte**. **Afeta:** toda
   afirmação da forma "o regime atual é ~8,6× mais esparso que o do pré-treino", que sustenta
   a H4 e parte do argumento de objetivo de pré-treino da H2.

Caveats menores, já declarados no corpo: `perm_within` é **oracular na construção** (§6);
`‖W·feat‖` é **descritivo, não causal** (§3a); sondas lineares **não provam** uso pelo modelo
nem ausência de informação não-linear (§3, TESTE 4); os 50,4 M do CLIP dos memes são
**estimativa por fórmula**, não contagem em runtime (§4, correção 2); `pred_perm == gold do
doador` (+6,4 pontos) é **~2σ, marginal, não usar** (§3, TESTE 5).

---

## 11. Decisão em aberto — do autor, não minha

❓ **Gastar ou não uma run no braço de mais capacidade** (LoRA r=64, ou `target_modules`
incluindo o MLP, **só no vídeo**)?

**Recomendação registrada: NÃO**, por quatro razões:

1. O **retorno medido do primeiro incremento de adaptação está abaixo do ruído**
   (TESTE 7: +0,004/+0,005 sob `mean_all`, contra tolerância 0,050-0,056). Mais capacidade é
   aposta contra uma medição, não a favor de uma.
2. O **teto causal do vídeo é −0,063 numa única task**, contra −0,254 do texto em todas. Mesmo
   um sucesso completo do braço mexeria numa métrica de seis.
3. O único sinal positivo (`mean_temporal`, +0,021/+0,017) exigiria mudar **duas coisas ao
   mesmo tempo** — capacidade **e** pooling fatorado — o que é **desenho experimental ruim e
   não atribuível**. E esse mesmo sinal está enfraquecido por `língua` subir +0,015 no mesmo
   resumo.
4. **O achado negativo já está medido e é publicável**, com método (permutação condicional à
   classe com magnitude casada) que é contributo metodológico por si só.

**Contra-argumento honesto, para o autor pesar:** as sondas medem **decodificabilidade linear
de vetores congelados**. Uma fusão treinada sobre tokens com mais capacidade de adaptação
poderia extrair de forma não-linear algo que nenhuma sonda deste documento veria. A regra
assimétrica do projeto reconhece exatamente isso: **quem prova é o braço treinado.** Se o
orçamento de GPU permitir uma run que não desloque nada do cronograma da tese, ela é
defensável — mas como **braço isolado de ablação**, nunca como mudança de baseline.

**Não decidido. É decisão do autor.**

---

## 12. Próximos passos

- [ ] **Aplicar as 9 correções da §4 nos docs 14 e 15** — ou marcar as seções obsoletas com
      ponteiro para este documento. As afirmações erradas (§3.4 do 15; §3.1/§4 e §3 do 14;
      F16) **ainda estão lá**.
- [ ] **Corrigir a docstring do `TemporalAttentionPool`** (`video_core.py:950-953`) — errada
      nas duas afirmações. **Única edição de código indicada**, e é documental.
- [ ] **Checar o confound de frames repetidos** (§10 item 1) antes de o perfil temporal ir
      para a tese.
- [ ] **Confirmar em fonte primária os números de pré-treino do VideoMAE** (§10 item 2).
- [ ] **Corrigir a descrição de arquitetura na tese e no paper CEUR-WS** — os 4 itens da §7.
- [ ] **Decidir sobre o braço de capacidade** (§11).
- [ ] (Opcional, barato, sem GPU) Ler manualmente `analysis/probe_cache_B15/task_b_flips.csv`
      e verificar se os JUDGEMENTAL cuja transcrição "parece DIRECT" são os que mudam — é o
      passo humano da H1 que nunca foi feito. O CSV tem 574 linhas com transcrição.

---

*Todo número deste documento tem fonte declarada. Os números dos 7 testes, os portões e as
citações `arquivo:linha` foram reconferidos nesta sessão contra `analysis/probe_report.txt`,
`analysis/probe_cache_B15/task_b_flips.csv` ou o próprio código. **Quatro itens são herdados e
estão marcados como tais no ponto de uso:** as duas sondas antigas da §8 (doc 15 §2.2/§2.3,
origem `test.ipynb`), o `‖ΔW‖_F ≈ 2 %` (`[RELATADO]`) e os 50,4 M do CLIP dos memes
(estimativa por fórmula). Nenhum número veio de treino novo.*
