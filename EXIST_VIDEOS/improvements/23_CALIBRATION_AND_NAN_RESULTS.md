# 23 — Combo C: calibração pós-treino (item 4) e NaN do PyEvALL (item 9)

**Executado em 2026-08-16** sobre `run_B2_bio_viewer_seed42_CLIP` e
`run_B2_bio_viewer_seed42_B15`. Zero GPU de treino. Código: `EXIST_VIDEOS/calibration.py`,
notebook `EXIST_VIDEOS/Thesis_video_calibration.ipynb`. Artefatos por run:
`calibration.json`, `pyevall_nan_diagnosis.json`.

---

## 1. Item 9 — os NaN do PyEvALL: RESOLVIDO `[VERIFICADO]`

**Causa exata**, idêntica nos dois encoders:

```
2_2_hard_FMeasure       status=WARNING  METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC
2_2_soft_CrossEntropy   status=WARNING  METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC
2_3_hard_FMeasure       status=WARNING  METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC
                        description: 'Use parameter: report="embedded"!'
```

F-Measure e Cross-Entropy **não suportam avaliação hierárquica**. As tasks 2_2 e 2_3
passam `PARAM_HIERARCHY` ao PyEvALL (`video_core.py:1496-1498`), e o PyEvALL recusa
calcular essas duas métricas nesse modo, devolvendo `status=WARNING`.
`vc._extract_metrics_from_report` converte `status != 'OK'` em `nan` (ARCH-14, decisão
deliberada para a coluna não sumir da tabela do Trainer).

**Confirmação cruzada:** a Task A **não** passa hierarquia — e é a única cujo
`fmeasure_hard` tem valor em todas as runs. O padrão bate exatamente.

**Não é bug, não há nada a corrigir.** Não é o `compute_metrics` a falhar em silêncio
(o modo de falha que o ARCH-14 existe para tornar visível), não são dados corrompidos, e
não afeta nenhuma métrica reportada — ICM/ICMNorm/ICMSoft/ICMSoftNorm são calculadas
normalmente. Se algum dia se quiser F1 por classe em B/C, é preciso uma avaliação
separada **sem** hierarquia; nunca misturar as duas na mesma tabela.

---

## 2. Item 4 — calibração por temperatura: PIORA, sistematicamente `[VERIFICADO]`

Protocolo: temperatura ajustada por NLL soft na metade de AJUSTE, `ICMSoftNorm` medido
na metade de REPORTE (n=252/253). Escalar em A/B, vector scaling em C — pré-registrado
no item 4 do doc 17, não escolhido pelo resultado.

| run | task | T ajustado | ICMSoftNorm REPORTE | Δ REPORTE | Δ ajuste | NLL ajuste |
|---|---|---|---|---|---|---|
| CLIP | A | 3,905 | 0,5280 → 0,3970 | **−0,1310** | −0,1096 | 0,885 → 0,585 |
| CLIP | B | 3,296 | 0,3908 → 0,2759 | **−0,1148** | −0,1261 | 1,232 → 0,800 |
| CLIP | C | [3,92 · 1,86 · 1,61 · 1,61 · 1,56 · 2,98] | 0,1339 → 0,0359 | **−0,0979** | −0,0718 | 0,457 → 0,348 |
| B15 | A | 2,830 | 0,5034 → 0,4183 | **−0,0851** | −0,0881 | 0,734 → 0,564 |
| B15 | B | 2,449 | 0,3667 → 0,2803 | **−0,0864** | −0,1073 | 1,018 → 0,799 |
| B15 | C | [2,79 · 1,41 · 1,36 · 1,29 · 1,22 · 2,15] | 0,1201 → 0,0615 | **−0,0586** | −0,0495 | 0,383 → 0,333 |

(T escalar da C, só como registro: 2,090 no CLIP · 1,621 no B15.)

### 2.1 Não é overfit da calibração — é sistemático

O modo de falha esperado (temperatura ajusta ruído) daria **ganho no ajuste e perda no
reporte**. Não é o que acontece: piora nas **duas** metades, nos **dois** encoders, nas
**três** tasks — 6/6 casos, 12/12 metades. E a NLL caiu em todos, como tem de cair (é o
objetivo otimizado; a queda da NLL não é evidência de nada).

### 2.2 Mecanismo: NLL e ICM-Soft-Norm são objetivos ANTAGÓNICOS aqui `[HIPÓTESE]`

Todo T ajustado ficou entre 2,4 e 3,9 — a NLL quer **suavizar** substancialmente as
probabilidades (os modelos são over-confident sob NLL). Mas o ICM-Soft é uma métrica de
**informação**: suavizar aumenta a entropia da predição e reduz a informação que ela
carrega sobre o gold. O que a NLL premia, o ICM-Soft penaliza.

Consequência para a tese: **temperature scaling por NLL — o método padrão (Guo et al.,
ICML 2017) — é o objetivo errado sob ICM-Soft-Norm.** Não é uma falha de implementação
nem um resultado sobre estes checkpoints: os dois encoders concordam, e a direção é a
mesma nas três tasks. É um resultado sobre a métrica.

⚠ `[HIPÓTESE]` quanto ao mecanismo: a leitura "ICM-Soft penaliza entropia" explica os
dados mas não foi verificada contra a definição formal do ICM-Soft no PyEvALL. Confirmar
antes de afirmar isto na tese.

### 2.3 Predição testável que isto gera

Se T > 1 piora monotonicamente, **T < 1 (afiar as probabilidades) deve melhorar** o
ICM-Soft-Norm. Não foi testado. A variante legítima é ajustar T maximizando
**ICM-Soft-Norm** (não a NLL) na metade de AJUSTE e reportar na de REPORTE — exatamente o
protocolo que `vc.tune_task_c_thresholds` já usa e que o projeto já aceita como
não-leaky. Custo: minutos, sem GPU de treino. **Decisão do autor**, e exige pré-registro
antes de rodar.

⚠ Se for feito: o número reportável continua a ser o da metade de REPORTE, e é preciso
declarar que a calibração passa a ser ajustada **contra a métrica avaliada** — o que é
defensável sob o protocolo metade/metade, mas tem de ser dito, não escondido.

---

## 3. Estado do Combo C

- Item 9: **fechado**, causa identificada, nada a corrigir.
- Item 4 na forma pré-registada (NLL): **fechado com resultado negativo**, forte e
  replicado nos dois encoders.
- Variante ICM-driven (§2.3): **em aberto**, aguarda decisão do autor.

Nenhuma métrica oficial de nenhuma run muda por causa deste documento. A calibração não
foi aplicada a nenhum artefato reportado.
