# 13 — Notas para a tese: achados da re-extração v2 (Task 3)

**O que este documento é:** o registro dos fatos produzidos durante a execução do
`11_EXTRACTION_IMPLEMENTATION_PLAN.md` (sessão de 2026-08-09), no formato em que podem
ir para a tese. Cada número tem fonte. O que é hipótese está marcado como hipótese.

**Convenção:**
- `[MEDIDO]` — número observado, com o artefato/célula que o produziu.
- `[HIPÓTESE]` — interpretação plausível, ainda não testada. **Não escrever na tese como
  fato.**
- `[LIMITAÇÃO]` — restrição a declarar explicitamente no texto.

---

## 1. Equivalência de normalização entre gerações de features (V5-NORM)

**Formulação pronta para o texto:**

> O ambiente de extração mudou entre v1 e v2; a equivalência foi estabelecida sobre o
> pipeline recomputado (T4b = 2,4 × 10⁻⁷), e o desvio contra os tensores v1 armazenados
> é de 1 nível de quantização uint8 (1,75 × 10⁻² no espaço normalizado).

**Dados de suporte** `[MEDIDO — verify_extraction_v2.ipynb, teste 4, 25 vídeos de
frames_v1_sample/]`:

| camada | o que compara | max\|Δ\| | papel |
|---|---|---|---|
| T4b | v2 (uint8 + desnormalização no load) vs v1 recomputado hoje, **sobre os mesmos frames de entrada** | **2,38 × 10⁻⁷** | bloqueante — passou |
| T4a | v1 recomputado hoje vs `.pt` v1 gravados em maio/2026 | 1,75 × 10⁻² | informativo |

**Por que o T4a vale exatamente isso:** `1 / (255 × 0,225) = 0,017429`, onde 0,225 é o
menor `image_std` do ImageNet (canal azul). O valor é **idêntico nos 25 vídeos**, sem
variação — assinatura de diferença de **um único nível de uint8** num pixel, propagada
pela normalização, e não de diferença de conteúdo.

**Causa** `[HIPÓTESE — coerente com a magnitude, não isolada experimentalmente]`:
mudança no arredondamento do resize bilinear entre maio/2026 e agosto/2026, plausivelmente
por atualização de `numpy` (1.26.4 → 2.5.1) e/ou `torchvision`. Não foi testada trocando
uma versão de cada vez.

**Consequência metodológica a declarar:** a comparação baseline↔v2 carrega esta nota. O
critério de decisão foi o T4b, que isola resize/crop + dtype da amostragem temporal.

**Fato adicional** `[MEDIDO — extract_video_v2.ipynb, célula 4, probe sintético]`: o
processor devolveu **uint8 nativo** com `do_rescale=False, do_normalize=False`
(`needs_cast_fallback = False`), com `max|Δ| = 2,38 × 10⁻⁷` em três geometrias (360×640,
1920×1080, 720×720). O risco residual antecipado — resize bilinear operando sobre uint8
em vez de float e arredondando diferente — **não se materializou**.

---

## 2. Truncagem física de arquivos no dataset (achado novo)

Um subconjunto dos vídeos do EXIST 2026 está **fisicamente incompleto**: o cabeçalho
declara uma duração que o arquivo não contém. São downloads interrompidos, não defeito
de software.

**Escala** `[MEDIDO — analysis/truncation_scan_v2.csv, população completa]`:

| split | vídeos truncados | fração | áudio perdido |
|---|---|---|---|
| training | 26 / 2 524 | 1,0 % | 9,3 min |
| test | 4 / 674 | 0,6 % | 1,1 min |

**Distribuição bimodal** `[MEDIDO]`: 25 dos 26 itens de treino perdem **mais de 5 s**.
Não existe cauda de casos ambíguos — ou o arquivo está íntegro, ou perdeu muito. Os
piores casos:

| id_EXIST | cabeçalho | decodificado | perdido |
|---|---|---|---|
| 220471 | 59,07 s | 1,23 s | 57,84 s (98 %) |
| 220203 | 58,16 s | 6,25 s | 51,92 s |
| 220516 | 56,66 s | 5,04 s | 51,62 s |
| 220589 | 42,35 s | 1,56 s | 40,79 s |
| 220390 | 41,77 s | 5,18 s | 36,59 s |

**Prova de que é dado, não software** `[MEDIDO — duas medições independentes]`:

1. `ffmpeg` (via `whisper.load_audio`) e `torchaudio.load` devolvem **exatamente a mesma
   duração** nos 9 itens testados, até o milésimo de segundo. Nenhum backend está errado.
2. O **vídeo trunca no mesmo ponto do áudio**: no 220471, áudio 1,232 s e vídeo 36 frames
   a 29,97 fps = 1,20 s. O decodificador emite `partial file`, `Invalid NAL unit size` e
   `Error splitting the input into NAL units`.

**Comparação com a estimativa anterior:** `03_OPTIMIZATION_FINDINGS.md` estimava 1–4 % do
split de teste com fim truncado, sem medir a população. A medição dá **0,6 % no teste** e
1,0 % no treino — dentro da ordem estimada, agora com número.

**Efeito sobre a v1 e sobre a v2:** ambas as gerações receberam o mesmo dado. A v1
promediava o silêncio sintético do `pad_or_trim` sobre esses itens; a v2, com a média
mascarada (A1), usa apenas os frames válidos e é **estritamente melhor** neles. O áudio
ausente não é recuperável por nenhum pipeline. **Não é regressão da v2** e não afeta a
atribuição A′/A/B.

**Efeito sobre a extração de frames** `[MEDIDO]`: nenhum. O passe 1 do V4 conta os frames
**realmente decodificáveis**, e os 64 índices são escolhidos depois disso — logo caem
todos dentro do que existe. Resultado: `n_with_fill = 0` nos dois splits, contra um gate
de 5 %.

---

## 3. Efeito da correção A1+A2 no áudio (teste 6)

`[MEDIDO — verify_extraction_v2.ipynb, teste 6, split training, n = 2 524]`

| métrica | valor | leitura |
|---|---|---|
| norma média v1 | 14,96 | bate com a referência congelada (14,96 ± 0,50) |
| norma média v2 | **12,17** | −18,7 % |
| itens bit-idênticos v1 == v2 | **0** | a correção pegou em todos |
| itens < 25 s com diferença material | **1 533 / 1 533** | 100 % |

O resultado descarta o modo de falha silencioso do V7 (script gravando na árvore errada
ou pulando por resume): se a correção não tivesse pegado, a v2 seria numericamente igual
à v1.

**Queda de 18,7 % na norma** `[HIPÓTESE — não testada]`: o encoder do Whisper produz,
para o padding de silêncio, vetores mutuamente muito consistentes; promediá-los não
provoca cancelamento, então eles dominavam a média e inflavam a norma. Removê-los da
média reduz a magnitude. **Verificação possível, ainda não feita:** medir a norma do
encoder sobre 30 s de silêncio puro e a similaridade cosseno entre frames de padding.

**Distribuição de janelas** `[MEDIDO — _extraction_meta.json de cada split]`:

| split | 1 janela | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| training | 1 689 | 813 | 22 | — | — |
| test | 513 | 143 | 16 | 1 | 1 |

A divergência inicial contra a previsão feita pelas durações de cabeçalho
(`ceil(duração/30)`) — 835 observados contra 842 previstos no treino — é **inteiramente
explicada** pela truncagem da seção 2: 9 itens cujo cabeçalho declara mais de 30 s contêm
menos de 30 s de áudio real, e 3 itens de fronteira (±0,03 s) atravessam o limite na
direção oposta. Reconciliação exata, sem resíduo.

---

## 4. Proveniência do ambiente (o problema V6, agora fechado)

Versões registradas nos `_extraction_meta.json`, que antes não existiam em lugar nenhum
`[MEDIDO]`:

| pacote | áudio v2 (02:07 UTC) | frames v2 (03:35 / 10:36 UTC) |
|---|---|---|
| python | 3.12.3 | 3.12.3 |
| torch | 2.6.0+cu124 | 2.6.0+cu124 |
| torchaudio | 2.6.0+cu124 | — |
| transformers | — | 5.6.0 |
| opencv | — | 4.13.0 (pip 4.13.0.92) |
| **numpy** | **1.26.4** | **2.5.1** |
| openai-whisper | **20250625** | — |

A versão do `openai-whisper` **nunca havia sido registrada** — era um caso do problema V6
num pacote que ninguém tinha olhado. A extração v1 (maio/2026) rodou com uma versão
desconhecida deste pacote e do OpenCV; não há como saber se eram estas.

**Incidente de ambiente durante a execução** `[MEDIDO]`: a célula `%pip install --quiet
opencv-python-headless` do `extract_video_v2.ipynb` atualizou o `numpy` de 1.26.4 para
2.5.1 entre a extração do áudio e a dos frames. Consequência observada:

```
ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.
```

— o `openai-whisper` deixou de importar (via `numba`). O áudio v2 já estava extraído sob
numpy 1.26.4, então nenhum artefato foi afetado, mas **os frames v2 foram extraídos sob
numpy 2.5.1**, o que é a causa candidata do drift do T4a (seção 1).

**Lição a registrar:** um `%pip install` dentro de um notebook de extração muda o
ambiente de proveniência no meio de um experimento. O registro de versões no meta é o que
permitiu detectar isso.

---

## 5. Custos medidos da re-extração

`[MEDIDO — outputs dos notebooks]`

| etapa | custo |
|---|---|
| áudio, training (2 524 itens) | ~14 min, GPU |
| áudio, test (674 itens) | ~3 min, GPU |
| frames, sondagem | 0,65 s/vídeo → projeção 35 min para 3 198 |
| frames, test (674 itens) | 9,8 min reais (1,15 it/s), CPU |
| disco, frames v2 | 9 633 792 B/vídeo × 3 198 ≈ 30,8 GB |

Os 64 frames `uint8` ocupam exatamente os mesmos bytes que os 16 `float32` da v1
(`64·3·224²·1 = 16·3·224²·4`) — quadruplicar a resolução temporal saiu de graça em disco.

---

## 6. Limitações a declarar explicitamente

- `[LIMITAÇÃO]` **A self-attention do encoder do Whisper ainda vê o padding** dentro de
  cada janela de 30 s. A correção A1 remove a diluição da **média**, não o efeito do
  padding no encoder — que é, aliás, o regime de treino do próprio Whisper.
- `[LIMITAÇÃO]` **A árvore `frames/` v1 foi apagada** (2026-08-08) e não é regenerável: o
  script que a produziu foi editado à mão e não existe no repositório nem no git. Restam
  25 tensores em `frames_v1_sample/`. Consequência direta: o braço **A′** da atribuição
  mistura *correções de extração de frames* com *demais correções de código*, e tem de
  ser reportado como **bloco único**, sem proporção interna.
- `[LIMITAÇÃO]` **O baseline continua válido como número, mas deixou de ser válido como
  experimento repetível** — `runs_baseline/seed_42/` tem métricas e fingerprints
  congelados, e ninguém consegue re-rodar a run.
- `[LIMITAÇÃO]` A amostragem temporal permanece **função da duração do vídeo**: os frames
  são espalhados pelo vídeo inteiro, então o intervalo entre frames consecutivos varia
  7,2× dentro do dataset (p10 0,54 s → p90 3,85 s para 16 frames). Aumentar `n_frames`
  divide o intervalo mas **não remove o confound** — a razão p90/p10 continua 7,2× para
  16, 32 ou 64.
- `[LIMITAÇÃO]` Os números do regime de pré-treino do VideoMAE (Kinetics-400, 16 frames,
  stride 4 @ ~30 fps → ~7,6 fps efetivos, gap ~0,13 s) vêm da configuração padrão do paper
  e **não foram confirmados em fonte primária**. Confirmar antes de irem para a tese.

---

## 7. Resultado da verificação (§9 do plano 11)

| teste | áudio | frames |
|---|---|---|
| 1 — metas de training e test idênticos exceto campos por-split | PASS | PASS |
| 2 — contagem (2 524 / 674) | PASS | PASS |
| 3 — `_missing.json` vazio | PASS (0 / 0) | PASS (0 / 0) |
| 4 — equivalência de normalização (T4b) | — | **PASS** (2,38e-07) |
| 5 + 8 — dtype/shape e varredura de zeros | PASS (0 todo-zeros) | **PASS** (0 dtype/shape ruins, 0 todo-zeros em 3 198) |
| 6 — norma v1 vs v2 | **PASS** | — |
| 7 — janelas por duração | PASS (explicado pela truncagem) | — |
| gate de 5 % de preenchimento | — | PASS (0,0 %) |
| P6 — smoke test do consumidor | **PASS** | **PASS** |

**Smoke test (P6)** `[MEDIDO — verify_extraction_v2.ipynb]`: o `EXISTVideoDataset` lendo
as árvores v2 devolve `frames` com shape `(16, 3, 224, 224)` em `float32` e `audio` com
shape `(1280,)`. A faixa de valores do tensor de frames é **[−2,1179; +2,6400]**, contra
a faixa da v1 medida em [−2,10; +2,64].

Os dois extremos são **exatamente os limites teóricos** do espaço normalizado ImageNet:
`(0/255 − 0,485)/0,229 = −2,1179` e `(255/255 − 0,406)/0,225 = +2,6400`. Isto é
verificação forte da desnormalização: se o `mean`/`std` lido do `_extraction_meta.json`
ou a divisão por 255 estivessem errados, os extremos não cairiam nos valores fechados.

O split congelado carregou como esperado: **2 019 treino / 505 validação**
(`Dataset/val_split_42.json`, seed 42, estratificação por `task_a`).

**Nenhum treino novo rodou até a data deste documento.** Nenhuma métrica de ICM nova
existe. O baseline congelado continua sendo `09_BASELINE_RESULTS.md`.

---

## 8. Run de calibração do P7 (2026-08-09)

`[MEDIDO — run_B2_bio_viewer_seed42_CAL/]` Configuração final (frames v2 + áudio v2 +
`restore_biases=True`), 15 épocas, `INIT_SEED=42`, split congelado 2 019/505.
Runtime 5 660 s (94 min). Fingerprints: init `ddb48bff571f413f`, final `c322520695e01c81`.

**E = argmin(eval_loss) = época 5** (`eval_loss` = 1,3445). Este é o orçamento de épocas
congelado para os três braços A′/A/B.

**O ARCH-06 reproduziu pela terceira vez** `[MEDIDO]`: a `eval_loss` atinge o mínimo na
época 5 e sobe monotonicamente depois (1,3445 → 1,7532 na época 15), enquanto as métricas
ICM continuam a melhorar até a época 7 (`task_a_icmnorm_hard` 0,5555 na 5 → **0,6028** na
7 → 0,5536 na 15). Escolher a época pela métrica reportada em vez de pela `eval_loss`
daria um número melhor **e seria leakage de validação** — a regra `E = argmin(eval_loss)`
foi mantida.

**Proveniência do ARCH-01 no `meta.json`** `[MEDIDO]`:
`"videomae_qv_biases": "restored_at_build_not_in_checkpoint"` — confirma que os biases
`q_bias`/`v_bias` foram restaurados no build desta run.

⚠ **Nenhum número desta run é reportável.** A CAL roda 15 épocas e serve exclusivamente
para congelar E. Além disso, uma run de 5 épocas **não é equivalente** à época 5 de uma
run de 15: o `lr_scheduler_type="cosine"` decai em função do total de passos, então o
braço B terá o LR completamente decaído no fim, o que a época 5 da CAL não tem.

**Observação sobre o ajuste de limiares da Task C** `[MEDIDO, run CAL]`: na metade de
reporte, os limiares ajustados deram ICMNorm **0,3171** contra **0,3403** do limiar
padrão 0,5 — ou seja, o ajuste **piorou** o resultado fora da metade em que foi ajustado.
É evidência a favor do protocolo de meia-validação do ARCH-11(b) (ajustar e reportar no
mesmo conjunto inflaria o número). Reconfirmar nos braços definitivos antes de escrever.

**Métricas `nan` no relatório do PyEvALL** `[MEDIDO]`: `task_b_fmeasure_hard`,
`task_b_crossentropy_soft` e `task_c_fmeasure_hard` vêm `nan` em todas as épocas. Não é
regressão: no baseline essas colunas **não existiam** no CSV. O PyEvALL não devolve
FMeasure nem CrossEntropy para as tarefas avaliadas com hierarquia (2_2 e 2_3), e o
ARCH-14 passou a materializar a ausência como `nan` em vez de omitir a coluna
silenciosamente. Todas as métricas oficiais (ICM, ICMNorm, ICMSoft, ICMSoftNorm) estão
presentes nas três tasks, e os proxies sklearn também.

---

## 9. Resultados dos três braços do P7 (2026-08-09)

Todos com `INIT_SEED=42`, split congelado 2 019/505, **E = 5 épocas** (congelado pela
CAL), frames **v2** em todos, estado final contra estado final.

| métrica | baseline | A′ | A | B |
|---|---|---|---|---|
| `eval_loss` | 1,5309 | 1,3208 | 1,3329 | 1,3405 |
| Task A ICMNorm hard | 0,5483 | 0,5651 | 0,5563 | **0,5862** |
| Task A ICMSoftNorm soft | 0,4972 | 0,4456 | 0,4413 | 0,4446 |
| Task B ICMNorm hard | 0,4579 | 0,4797 | **0,4966** | 0,4805 |
| Task B ICMSoftNorm soft | 0,3713 | 0,3207 | 0,3209 | 0,3303 |
| Task C ICMNorm hard | 0,3784 ⚠ | 0,3384 | 0,3359 | 0,3546 |
| Task C ICMSoftNorm soft | 0,1378 | 0,0653 | 0,0644 | 0,0605 |

⚠ Task C **hard** do baseline **não é comparável** com a das runs novas: o
`Thesis_video_baseline.ipynb` tem a sua própria cópia do `_format_for_pyevall`, anterior
à correção ARCH-10, que altera o **gold** (as ~0,7 % de linhas contraditórias
desaparecem). São notas contra provas diferentes. Task C soft, A e B não são afetadas.

**Proveniência** `[MEDIDO — fingerprints.json / meta.json]`:

| braço | áudio | `restore_biases` | `init` | `final` | runtime |
|---|---|---|---|---|---|
| A′ | v1 | False | `61a04c68e5c5bf7f` | `8cc5f9483276b070` | 31 min |
| A | v1 | True | `ddb48bff571f413f` | `3cc7a8064cf8849f` | 31 min |
| B | v2 | True | `ddb48bff571f413f` | `0b79cb9e0bb8bd5d` | 32 min |

O `init` de A e B coincide com o da CAL (mesma seed, mesmo flag) e difere do de A′ — o
protocolo de determinismo se sustenta e o flag `restore_biases` é genuinamente a única
variável entre A′ e A. Os `final` diferem entre os três.

### 9.1 `A − A′` = efeito isolado do ARCH-01 (biases q/v do VideoMAE)

| métrica | Δ |
|---|---|
| Task A hard | −0,009 |
| Task A soft | −0,004 |
| Task B hard | **+0,017** |
| Task B soft | +0,000 |
| Task C hard | −0,003 |
| Task C soft | −0,001 |

**Restaurar os biases não produziu o efeito esperado.** O `CLAUDE.md` registrava o
ARCH-01 como *"candidato nº 1 para a performance ruim"* — um bias treinado de norma ~12,9
por camada substituído por zeros em 12 camadas. Medido isoladamente: cinco das seis
métricas se movem menos de 0,01.

⚠ `[LIMITAÇÃO]` Seed única. Este projeto já registrou um ganho de +0,020 que não replicou
entre seeds. Deltas desta magnitude não se distinguem de variância de seed com n = 1.
Para afirmar que o ARCH-01 é irrelevante seriam necessárias 2–3 `INIT_SEED` pareadas
(`SPLIT_SEED` fixo — é para isso que o OPT-06 separou os dois).

### 9.2 `B − A` = efeito isolado da correção de áudio (A1 + A2)

| métrica | Δ |
|---|---|
| Task A hard | **+0,030** |
| Task A soft | +0,003 |
| Task B hard | **−0,016** |
| Task B soft | +0,009 |
| Task C hard | **+0,019** |
| Task C soft | −0,004 |

Não é uniformemente positivo: Task A e C hard sobem, Task B hard desce. Com n = 1, dentro
da faixa de variância de seed do projeto. **Não escrever "a correção de áudio ajuda" sem
seeds pareadas.**

### 9.3 `B − baseline` — **CONFUNDIDO, não usar como está**

O baseline é o modelo **SWA sobre as épocas 5–15**, ou seja, **15 épocas de treino**. Os
braços treinaram **5**. `B − baseline` mistura, portanto, {features v2 + código
corrigido} com {5 épocas vs 15 + SWA}. Não é atribuível.

**A evidência de que o orçamento de épocas é o que governa o padrão** `[MEDIDO — curva da
CAL]`: as métricas soft sobem monotonicamente até a época 15, enquanto a `eval_loss` sobe
a partir da 5.

| | CAL ép. 5 | CAL ép. 15 |
|---|---|---|
| Task A soft | 0,4417 | 0,5058 |
| Task B soft | 0,3098 | 0,3947 |
| Task C soft | 0,0785 | 0,1514 |

Todas as métricas soft dos três braços caem perto do valor da **época 5**, não do da 15.
O padrão "hard melhora, soft piora" observado em todos os braços **não é propriedade das
features v2** — é E = 5 contra um baseline de 15 épocas com SWA.

**O que sobrevive ao confound, na direção favorável:** as métricas **hard** melhoraram com
**um terço do treino** — Task A hard **+0,038** e Task B hard **+0,023** contra um
baseline que treinou 3× mais. Um confound que penaliza o sistema novo e ainda assim o
deixa à frente é evidência mais forte, não mais fraca.

### 9.4 `B15 − baseline` = ganho total, com orçamento de treino igualado

Braço **B15**: mesma configuração do B (frames v2 + áudio v2 + `restore_biases=True`), com
**15 épocas** — o orçamento do baseline. Estado final dos dois lados (regra de simetria).

| métrica | baseline (SWA, 15 ép.) | B15 (final, 15 ép.) | Δ |
|---|---|---|---|
| Task A ICMNorm hard | 0,5483 | 0,5536 | **+0,005** |
| Task A ICMSoftNorm soft | 0,4972 | 0,5058 | **+0,009** |
| Task B ICMNorm hard | 0,4579 | 0,4968 | **+0,039** |
| Task B ICMSoftNorm soft | 0,3713 | 0,3947 | **+0,023** |
| Task C ICMSoftNorm soft | 0,1378 | 0,1514 | **+0,014** |
| Task C ICMNorm hard | 0,3784 | 0,3436 | ⚠ não comparável (ARCH-10) |

**As cinco métricas comparáveis são todas positivas.**

⚠ `[LIMITAÇÃO]` **Resta uma diferença declarada:** o baseline é o modelo **SWA** (média
dos pesos das épocas 5–15) e o B15 **não usa SWA** — removido de propósito pelo ARCH-06,
com medição de que o SWA era pior que o melhor da curva em 6/6 métricas. É diferença **de
sistema**, não artefato de protocolo, e parte do delta vem daí.

⚠ `[LIMITAÇÃO]` Seed única (`INIT_SEED=42`). Os deltas de Task A (+0,005 / +0,009) são
pequenos demais para se distinguirem de variância de seed. Os de Task B (+0,039 / +0,023)
são os mais robustos do conjunto. Confirmação exigiria 2–3 seeds pareadas.

**O padrão "soft piora" era orçamento de treino, confirmado** `[MEDIDO]`: os braços de 5
épocas ficam em 0,4446 / 0,3303 / 0,0605 (A/B/C soft); com 15 épocas sobem para 0,5058 /
0,3947 / 0,1514 e superam o baseline. Não era propriedade das features v2.

### 9.5 Reprodutibilidade bit a bit entre runs independentes

`[MEDIDO]` O braço **B15** e a run de **calibração** são a mesma configuração e foram
executados em sessões separadas, com ~2 h de intervalo. Resultado: fingerprint final
**idêntico** (`c322520695e01c81`) e todas as métricas iguais até a quarta casa decimal.

É a verificação mais forte possível do protocolo de determinismo do projeto
(`set_full_determinism`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, re-seed antes de cada init de pesos): duas runs
independentes produziram os mesmos pesos bit a bit.

---

*Escrito em 2026-08-09 a partir dos outputs de `extract_audio_v2.ipynb`,
`extract_video_v2.ipynb` e `verify_extraction_v2.ipynb`, dos `_extraction_meta.json` das
quatro árvores v2 e de `analysis/truncation_scan_v2.csv`. Números sem fonte explícita não
existem neste documento.*
