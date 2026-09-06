# Diagnóstico descritivo dos padrões comportamentais dos anotadores (EXIST 2026, Task 2)

**O que este documento é:** um diagnóstico DESCRITIVO dos 887 anotadores do EXIST 2026 Task 2 (memes), a partir de perfis de comportamento agregado calculados sem recorrer a consenso, verdade inferida ou concordância par-a-par. Não é um sistema, não treina nada, e não justifica down-weighting de ninguém.

- **Extração:** `reliability/behavioral_profile.py` (Parte 1, `reliability/outputs/behavioral_profile_part1.md`)
- **Análise:** `reliability/run_behavioral_diagnosis.py` (este relatório)
- **Perfis:** `reliability/outputs/behavioral_profiles.csv`
- **Seed:** 42 (KMeans, `n_init=10`) — determinístico

**Convenção** (a mesma de `reliability/NOTAS_PARA_TESE.md`): `[MEDIDO]` — número observado, com o artefacto que o produziu · `[HIPÓTESE]` — interpretação plausível, ainda não testada; **não escrever como facto** · `[LIMITAÇÃO]` — restrição a declarar explicitamente no texto.

---

## 0. Critérios pré-registados

Fixados no código (`run_behavioral_diagnosis.py`, bloco no topo) antes de qualquer leitura de dados.

**Dispersão** — uma feature é *dispersa* se e só se as três condições se verificarem; caso contrário é *concentrada*. Variância nula → *degenerada* (reportada, fora do clustering).

1. valores distintos ≥ 10
2. nenhum bin (10 bins sobre a amplitude observada) contém > 60% dos anotadores
3. IQR ≥ 10% da amplitude observada

**Clustering útil** — um clustering PASSA se e só se nenhum cluster tiver menos de 5% dos anotadores **e** o silhouette for > 0.25. Este critério existe para não repetir o resultado degenerado `[883, 2, 1, 1]` das tentativas anteriores sem o notar.

**Anomalia** — os limiares da tabela da §2, todos fixados a priori.

**NaN no clustering** — política `complete-case`: anotadores com NaN em qualquer feature retida são excluídos, e o número é declarado. Imputação foi rejeitada porque inventaria comportamento onde o denominador não existe.

---

## 1. Factos estruturais herdados da Parte 1

Três descobertas da extração limitam o que este diagnóstico pode perguntar. Repetidas aqui porque condicionam a leitura de tudo o que se segue.

### 1.1 A coerência hierárquica é perfeita — a família "erro objetivo" não existe neste corpus

`[MEDIDO]` — cross-tab ao nível do voto  n = 23 904:

```
A × B:   NO|- 10618  ·  YES|DIRECT 8471  ·  YES|JUDGEMENTAL 4688  ·  YES|UNKNOWN 127
A × C:   NO|DASH 10618  ·  YES|CATEGORIA 13221  ·  YES|UNKNOWN 65
```

Zero votos `NO` + categoria de sexismo. Zero votos `NO` + `DIRECT`/`JUDGEMENTAL`. Logo `incoherence_A_to_C` e `incoherence_A_to_B` valem **0,0000 nos 887 anotadores**.

`[HIPÓTESE]` — hierarquia imposta pela interface de anotação. A coincidência exacta das contagens é consistente com um constrangimento de ferramenta, mas **não distingue** constrangimento de disciplina do anotador.

### 1.2 Os painéis EN e ES são quase disjuntos

`[MEDIDO]` — **es**: 437 · **en**: 431 · **both**: 19. `lang_gap` fica definida em apenas 19 dos 887 anotadores.

### 1.3 A regra de coerência usada

```
Segundo as guidelines do EXIST, `-` é o rótulo atribuído a instâncias NÃO-SEXISTAS
nas subtasks 2.2 e 2.3. Logo:

  NO na Task A  +  `-` na B/C                     -> COERENTE (nao conta)
  NO na Task A  +  >=1 categoria de SEXISM_SET    -> incoherence_A_to_C
  NO na Task A  +  DIRECT ou JUDGEMENTAL na B     -> incoherence_A_to_B
  YES na Task A +  nem `-` nem categoria real     -> incoherence_yes_empty
  UNKNOWN                                          -> NUNCA conta como incoerencia
                                                      A->C ou A->B; conta como
                                                      "vazio" no caso YES (e o
                                                      unico valor que satisfaz
                                                      "nem `-` nem categoria").

`incoherence_yes_empty` e, por esta definicao, colinear com `unknown_rate_c`
restrita aos votos YES. [LIMITACAO] -- as duas nao sao evidencia independente.
```

Limiares de denominador mínimo: `MIN_SEXIST_ITEMS` = 5, `MIN_NO_VOTES` = 5, `MIN_YES_VOTES` = 5, `MIN_VOTES_PER_LANG` = 5.

---

## 2. Há dispersão real, ou toda a gente se parece?

`[MEDIDO]` — estatísticas descritivas sobre os 887 anotadores (NaN excluídos por feature):

| feature | n | NaN | média | std | min | q25 | mediana | q75 | max |
|---|---|---|---|---|---|---|---|---|---|
| yes_rate | 887.0000 | 0.0000 | 0.5581 | 0.2132 | 0.0000 | 0.4074 | 0.5556 | 0.7060 | 1.0000 |
| unknown_rate_b | 887.0000 | 0.0000 | 0.0054 | 0.0160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1111 |
| unknown_rate_c | 887.0000 | 0.0000 | 0.0028 | 0.0121 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1111 |
| n_distinct_categories | 844.0000 | 43.0000 | 4.2109 | 0.8634 | 2.0000 | 4.0000 | 4.0000 | 5.0000 | 5.0000 |
| category_entropy | 844.0000 | 43.0000 | 0.7582 | 0.1614 | 0.1522 | 0.6695 | 0.7918 | 0.8778 | 0.9937 |
| mean_labels_per_item | 844.0000 | 43.0000 | 1.3199 | 0.3723 | 1.0000 | 1.0489 | 1.2143 | 1.4511 | 4.2500 |
| incoherence_A_to_C | 790.0000 | 97.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| incoherence_A_to_B | 790.0000 | 97.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| incoherence_yes_empty | 844.0000 | 43.0000 | 0.0053 | 0.0233 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2500 |
| yes_rate_en | 450.0000 | 437.0000 | 0.5409 | 0.2203 | 0.0000 | 0.4074 | 0.5556 | 0.7037 | 1.0000 |
| yes_rate_es | 456.0000 | 431.0000 | 0.5743 | 0.2059 | 0.0370 | 0.4231 | 0.5833 | 0.7407 | 1.0000 |
| lang_gap | 19.0000 | 868.0000 | 0.1965 | 0.1372 | 0.0000 | 0.0741 | 0.1852 | 0.2963 | 0.4444 |

### Veredito por feature

| feature | veredito | n_distintos | bin_modal | IQR/amplitude |
|---|---|---|---|---|
| yes_rate | dispersa | 104 | 0.1984 | 0.2986 |
| unknown_rate_b | concentrada | 14 | 0.8805 | 0.0000 |
| unknown_rate_c | concentrada | 12 | 0.9402 | 0.0000 |
| n_distinct_categories | concentrada | 4 | 0.4609 | 0.3333 |
| category_entropy | dispersa | 697 | 0.2358 | 0.2476 |
| mean_labels_per_item | concentrada | 200 | 0.6149 | 0.1238 |
| incoherence_A_to_C | degenerada | 1 | 1.0000 | 0.0000 |
| incoherence_A_to_B | degenerada | 1 | 1.0000 | 0.0000 |
| incoherence_yes_empty | concentrada | 24 | 0.9384 | 0.0000 |
| yes_rate_en | dispersa | 45 | 0.2067 | 0.2963 |
| yes_rate_es | dispersa | 93 | 0.1842 | 0.3299 |
| lang_gap | dispersa | 15 | 0.2632 | 0.5000 |

**Resumo:** 5 dispersas · 5 concentradas · 2 degeneradas (variância nula), de 12 features.

### Histogramas

**`yes_rate`** — dispersa

```
[ 0.0000,  0.1000) #####                                      21
[ 0.1000,  0.2000) ########                                   35
[ 0.2000,  0.3000) #############                              57
[ 0.3000,  0.4000) ###################                        84
[ 0.4000,  0.5000) ################################          139
[ 0.5000,  0.6000) ########################################  176
[ 0.6000,  0.7000) ############################              122
[ 0.7000,  0.8000) #############################             129
[ 0.8000,  0.9000) ###################                        84
[ 0.9000,  1.0000) #########                                  40
```

**`unknown_rate_b`** — concentrada

```
[ 0.0000,  0.0111) ########################################  781
[ 0.0111,  0.0222)                                             2
[ 0.0222,  0.0333)                                             0
[ 0.0333,  0.0444) ####                                       83
[ 0.0444,  0.0556)                                             1
[ 0.0556,  0.0667)                                             1
[ 0.0667,  0.0778) #                                          14
[ 0.0778,  0.0889)                                             2
[ 0.0889,  0.1000)                                             0
[ 0.1000,  0.1111)                                             3
```

**`unknown_rate_c`** — concentrada

```
[ 0.0000,  0.0111) ########################################  834
[ 0.0111,  0.0222)                                             1
[ 0.0222,  0.0333)                                             0
[ 0.0333,  0.0444) ##                                         38
[ 0.0444,  0.0556)                                             2
[ 0.0556,  0.0667)                                             0
[ 0.0667,  0.0778)                                             7
[ 0.0778,  0.0889)                                             4
[ 0.0889,  0.1000)                                             0
[ 0.1000,  0.1111)                                             1
```

**`n_distinct_categories`** — concentrada

```
[ 2.0000,  2.3000) ###                                        33
[ 2.3000,  2.6000)                                             0
[ 2.6000,  2.9000)                                             0
[ 2.9000,  3.2000) ###############                           145
[ 3.2000,  3.5000)                                             0
[ 3.5000,  3.8000)                                             0
[ 3.8000,  4.1000) ############################              277
[ 4.1000,  4.4000)                                             0
[ 4.4000,  4.7000)                                             0
[ 4.7000,  5.0000) ########################################  389
```

**`category_entropy`** — dispersa

```
[ 0.1522,  0.2363) #                                           7
[ 0.2363,  0.3205) ##                                          9
[ 0.3205,  0.4046) ####                                       21
[ 0.4046,  0.4888) ####                                       19
[ 0.4888,  0.5729) ##########                                 51
[ 0.5729,  0.6571) #################                          85
[ 0.6571,  0.7412) ########################                  121
[ 0.7412,  0.8254) #####################################     185
[ 0.8254,  0.9095) ########################################  199
[ 0.9095,  0.9937) ##############################            147
```

**`mean_labels_per_item`** — concentrada

```
[ 1.0000,  1.3250) ########################################  519
[ 1.3250,  1.6500) ###############                           198
[ 1.6500,  1.9750) ######                                     77
[ 1.9750,  2.3000) ##                                         29
[ 2.3000,  2.6250) #                                          13
[ 2.6250,  2.9500)                                             3
[ 2.9500,  3.2750)                                             3
[ 3.2750,  3.6000)                                             0
[ 3.6000,  3.9250)                                             1
[ 3.9250,  4.2500)                                             1
```

**`incoherence_A_to_C`** — degenerada

```
[0.0000] ######################################## 790 (valor único)
```

**`incoherence_A_to_B`** — degenerada

```
[0.0000] ######################################## 790 (valor único)
```

**`incoherence_yes_empty`** — concentrada

```
[ 0.0000,  0.0250) ########################################  792
[ 0.0250,  0.0500)                                             8
[ 0.0500,  0.0750) #                                          20
[ 0.0750,  0.1000) #                                          10
[ 0.1000,  0.1250)                                             6
[ 0.1250,  0.1500)                                             3
[ 0.1500,  0.1750)                                             2
[ 0.1750,  0.2000)                                             2
[ 0.2000,  0.2250)                                             0
[ 0.2250,  0.2500)                                             1
```

**`yes_rate_en`** — dispersa

```
[ 0.0000,  0.1000) #######                                    16
[ 0.1000,  0.2000) ##########                                 24
[ 0.2000,  0.3000) #############                              30
[ 0.3000,  0.4000) #################                          39
[ 0.4000,  0.5000) ###############################            72
[ 0.5000,  0.6000) ########################################   93
[ 0.6000,  0.7000) #######################                    54
[ 0.7000,  0.8000) ############################               64
[ 0.8000,  0.9000) #################                          39
[ 0.9000,  1.0000) ########                                   19
```

**`yes_rate_es`** — dispersa

```
[ 0.0370,  0.1333) ####                                        9
[ 0.1333,  0.2296) ########                                   16
[ 0.2296,  0.3259) #############                              27
[ 0.3259,  0.4222) #############################              60
[ 0.4222,  0.5185) ###################################        73
[ 0.5185,  0.6148) ###############################            65
[ 0.6148,  0.7111) ########################################   84
[ 0.7111,  0.8074) ##########################                 54
[ 0.8074,  0.9037) ######################                     46
[ 0.9037,  1.0000) ##########                                 22
```

**`lang_gap`** — dispersa

```
[ 0.0000,  0.0444) ################                            2
[ 0.0444,  0.0889) ########################################    5
[ 0.0889,  0.1333) ########                                    1
[ 0.1333,  0.1778) ########                                    1
[ 0.1778,  0.2222) ########                                    1
[ 0.2222,  0.2667) ################################            4
[ 0.2667,  0.3111)                                             0
[ 0.3111,  0.3556) ########                                    1
[ 0.3556,  0.4000) ########################                    3
[ 0.4000,  0.4444) ########                                    1
```

---

## 3. Quantos anotadores são anómalos, e em quê?

`[MEDIDO]` — contagens sobre os 887 anotadores. NaN nunca conta como anomalia: denominador insuficiente não é evidência de comportamento anómalo.

| regra | critério | n | % de 887 | NaN (não avaliáveis) |
|---|---|---|---|---|
| yes_rate_extremo | yes_rate quase-constante (>0,90 ou <0,10) | 61 | 6.9 | 0 |
| categoria_unica | usa sempre a mesma categoria (n_distinct_categories == 1) | 0 | 0.0 | 43 |
| incoerencia_AC_qualquer | qualquer contradição A->C (> 0) | 0 | 0.0 | 97 |
| incoerencia_AC_sistematica | contradição A->C sistemática (> 0,10) | 0 | 0.0 | 97 |
| incoerencia_AB_sistematica | contradição A->B sistemática (> 0,10) | 0 | 0.0 | 97 |
| lang_gap_grande | comportamento muito diferente entre línguas (lang_gap > 0,30) | 5 | 0.6 | 868 |

- **yes_rate_extremo** (61): `Annotator_110`, `Annotator_135`, `Annotator_158`, `Annotator_159`, `Annotator_160`, `Annotator_191`, `Annotator_212`, `Annotator_234`, `Annotator_247`, `Annotator_253`, `Annotator_273`, `Annotator_277`, `Annotator_286`, `Annotator_293`, `Annotator_295`, `Annotator_299`, `Annotator_300`, `Annotator_307`, `Annotator_310`, `Annotator_329`, `Annotator_364`, `Annotator_365`, `Annotator_374`, `Annotator_386`, `Annotator_483`, `Annotator_484`, `Annotator_492`, `Annotator_494`, `Annotator_505`, `Annotator_532`, `Annotator_533`, `Annotator_544`, `Annotator_558`, `Annotator_568`, `Annotator_584`, `Annotator_586`, `Annotator_599`, `Annotator_611`, `Annotator_649`, `Annotator_652`, … (+21)
- **categoria_unica** (0): (nenhum)
- **incoerencia_AC_qualquer** (0): (nenhum)
- **incoerencia_AC_sistematica** (0): (nenhum)
- **incoerencia_AB_sistematica** (0): (nenhum)
- **lang_gap_grande** (5): `Annotator_11`, `Annotator_148`, `Annotator_187`, `Annotator_199`, `Annotator_285`

### Sobreposição entre conjuntos de anomalia

| ∩ | yes_rate_extremo | lang_gap_grande |
|---|---|---|
| yes_rate_extremo | 61 | 0 |
| lang_gap_grande | 0 | 5 |

**Anotadores anómalos em ≥ 2 critérios:** 0 — (nenhum)

---

## 4. Estratificação por número de votos

`[MEDIDO]` — 11 anotadores com < 15 votos, 876 com ≥ 15. As features dos primeiros são ruidosas por construção (denominadores de 9 a 14); **os dois estratos não se misturam nas conclusões**.

| regra | < 15 votos | % < 15 votos | ≥ 15 votos | % ≥ 15 votos |
|---|---|---|---|---|
| yes_rate_extremo | 2 / 11 | 18.2 | 59 / 876 | 6.7 |
| categoria_unica | 0 / 11 | 0.0 | 0 / 876 | 0.0 |
| incoerencia_AC_qualquer | 0 / 11 | 0.0 | 0 / 876 | 0.0 |
| incoerencia_AC_sistematica | 0 / 11 | 0.0 | 0 / 876 | 0.0 |
| incoerencia_AB_sistematica | 0 / 11 | 0.0 | 0 / 876 | 0.0 |
| lang_gap_grande | 0 / 11 | 0.0 | 5 / 876 | 0.6 |

- **yes_rate_extremo** no estrato < 15 votos (2 de 11): `Annotator_885`, `Annotator_886`

`[LIMITAÇÃO]` — com 9 a 14 votos, um `yes_rate` de 1,00 exige apenas 9 votos YES consecutivos. No estrato baixo, "quase-constante" e "pouca amostra" não são distinguíveis por estes dados.

---

## 5. Os perfis clusterizam de forma equilibrada?

**Features retidas (7):** `yes_rate`, `unknown_rate_b`, `unknown_rate_c`, `n_distinct_categories`, `category_entropy`, `mean_labels_per_item`, `incoherence_yes_empty`

**Features excluídas (regra pré-registada, não exclusão silenciosa):**
- `incoherence_A_to_C` — variância nula
- `incoherence_A_to_B` — variância nula

`lang_gap`, `yes_rate_en` e `yes_rate_es` ficam fora por desenho: NaN em 97,9% / 49,3% / 48,6% dos anotadores (§1.2). São reportadas à parte, não imputadas.

**Anotadores clusterizados:** 844 de 887 (política `complete-case`; 43 excluídos por NaN).

| K | tamanhos | min_frac | silhouette | veredito |
|---|---|---|---|---|
| 2 | [792, 52] | 0.0616 | 0.5432 | PASS |
| 3 | [52, 317, 475] | 0.0616 | 0.2960 | PASS |
| 4 | [424, 286, 83, 51] | 0.0604 | 0.3226 | PASS |
| 5 | [115, 170, 426, 82, 51] | 0.0604 | 0.2962 | PASS |
| 6 | [51, 231, 163, 89, 228, 82] | 0.0604 | 0.2806 | PASS |

**5 de 5 valores de K passam o critério pré-registado.** A caracterização segue abaixo. Ler em conjunto com a §6, que mostra que a separação do K com melhor silhouette **não** é comportamental.

### K = 2 (PASS) — o que distingue cada cluster

Média por cluster **menos** média global:

| cluster | yes_rate | unknown_rate_b | unknown_rate_c | n_distinct_categories | category_entropy | mean_labels_per_item | incoherence_yes_empty |
|---|---|---|---|---|---|---|---|
| 0 | -0.0009 | -0.0003 | -0.0029 | 0.0088 | 0.0008 | 0.0031 | -0.0053 |
| 1 | 0.0141 | 0.0049 | 0.0444 | -0.1340 | -0.0126 | -0.0472 | 0.0811 |

### K = 3 (PASS) — o que distingue cada cluster

Média por cluster **menos** média global:

| cluster | yes_rate | unknown_rate_b | unknown_rate_c | n_distinct_categories | category_entropy | mean_labels_per_item | incoherence_yes_empty |
|---|---|---|---|---|---|---|---|
| 0 | 0.0141 | 0.0049 | 0.0444 | -0.1340 | -0.0126 | -0.0472 | 0.0811 |
| 1 | -0.1091 | 0.0006 | -0.0029 | -0.8260 | -0.1408 | -0.1830 | -0.0053 |
| 2 | 0.0712 | -0.0009 | -0.0029 | 0.5659 | 0.0954 | 0.1273 | -0.0053 |

### K = 4 (PASS) — o que distingue cada cluster

Média por cluster **menos** média global:

| cluster | yes_rate | unknown_rate_b | unknown_rate_c | n_distinct_categories | category_entropy | mean_labels_per_item | incoherence_yes_empty |
|---|---|---|---|---|---|---|---|
| 0 | 0.0742 | -0.0055 | -0.0029 | 0.5509 | 0.0925 | 0.1286 | -0.0053 |
| 1 | -0.1121 | -0.0042 | -0.0029 | -0.8682 | -0.1498 | -0.1859 | -0.0053 |
| 2 | 0.0013 | 0.0405 | -0.0025 | 0.2590 | 0.0521 | 0.0124 | -0.0048 |
| 3 | 0.0098 | 0.0029 | 0.0446 | -0.1325 | -0.0142 | -0.0466 | 0.0819 |

### K = 5 (PASS) — o que distingue cada cluster

Média por cluster **menos** média global:

| cluster | yes_rate | unknown_rate_b | unknown_rate_c | n_distinct_categories | category_entropy | mean_labels_per_item | incoherence_yes_empty |
|---|---|---|---|---|---|---|---|
| 0 | 0.0680 | -0.0052 | -0.0029 | 0.6587 | 0.1333 | 0.6748 | -0.0053 |
| 1 | -0.1180 | -0.0033 | -0.0029 | -1.3109 | -0.2346 | -0.2045 | -0.0053 |
| 2 | 0.0279 | -0.0054 | -0.0029 | 0.3126 | 0.0495 | -0.0949 | -0.0053 |
| 3 | -0.0020 | 0.0406 | -0.0025 | 0.2525 | 0.0510 | -0.0005 | -0.0048 |
| 4 | 0.0098 | 0.0029 | 0.0446 | -0.1325 | -0.0142 | -0.0466 | 0.0819 |

### K = 6 (PASS) — o que distingue cada cluster

Média por cluster **menos** média global:

| cluster | yes_rate | unknown_rate_b | unknown_rate_c | n_distinct_categories | category_entropy | mean_labels_per_item | incoherence_yes_empty |
|---|---|---|---|---|---|---|---|
| 0 | 0.0098 | 0.0029 | 0.0446 | -0.1325 | -0.0142 | -0.0466 | 0.0819 |
| 1 | -0.1275 | -0.0054 | -0.0029 | 0.2307 | 0.0556 | -0.0642 | -0.0053 |
| 2 | -0.1149 | -0.0032 | -0.0029 | -1.3581 | -0.2399 | -0.2045 | -0.0053 |
| 3 | 0.0585 | -0.0051 | -0.0029 | 0.6655 | 0.1339 | 0.7761 | -0.0053 |
| 4 | 0.1869 | -0.0055 | -0.0029 | 0.4163 | 0.0477 | -0.0811 | -0.0053 |
| 5 | -0.0020 | 0.0406 | -0.0025 | 0.2525 | 0.0510 | -0.0005 | -0.0048 |

### Composição demográfica (K = 2, o melhor silhouette dos que passam)

Descritivo. Os clusters alinharem ou não com demografia é informativo nos dois sentidos: alinhamento sugeriria que o "comportamento" é um proxy do grupo; ausência de alinhamento sugere variação individual.

**gender_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | F | M |
|---|---|---|
| 0 | 51.4 | 48.6 |
| 1 | 53.8 | 46.2 |

**age_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | 18-22 | 23-45 | 46+ |
|---|---|---|---|
| 0 | 31.4 | 34.2 | 34.3 |
| 1 | 38.5 | 30.8 | 30.8 |

**ethnicities_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | Black or African American | Hispano or Latino | White or Caucasian |
|---|---|---|---|
| 0 | 10.4 | 23.2 | 58.3 |
| 1 | 7.7 | 32.7 | 57.7 |

**study_levels_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | Bachelor's degree | High school degree or equivalent | Master's degree |
|---|---|---|---|
| 0 | 48.7 | 29.0 | 16.9 |
| 1 | 44.2 | 32.7 | 17.3 |

**countries_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | Chile | Italy | Mexico | Poland | Portugal | South Africa | Spain | United Kingdom |
|---|---|---|---|---|---|---|---|---|
| 0 | 4.4 | 5.3 | 12.9 | 5.9 | 9.3 | 11.1 | 12.4 | 11.2 |
| 1 | 5.8 | 5.8 | 17.3 | 1.9 | 3.8 | 3.8 | 28.8 | 13.5 |

---

## 6. O que a partição primária está realmente a separar (post-hoc)

> ⚠ **Esta secção é POST-HOC e não estava pré-registada.** Foi acrescentada depois de ver o resultado da §5. Não substitui a análise primária; corre a par dela. Registo completo em `reliability/outputs/behavioral_diagnosis_observacoes.md`.

### 6.1 Três das features retidas são indicadores binários, não escalas

`[MEDIDO]` — fracção de valores exactamente zero (sinalizada a partir de 80%):

| feature | n | zeros | fracção de zeros | indicador? |
|---|---|---|---|---|
| yes_rate | 887 | 2 | 0.0023 | não |
| unknown_rate_b | 887 | 781 | 0.8805 | SIM |
| unknown_rate_c | 887 | 834 | 0.9402 | SIM |
| n_distinct_categories | 844 | 0 | 0.0000 | não |
| category_entropy | 844 | 0 | 0.0000 | não |
| mean_labels_per_item | 844 | 0 | 0.0000 | não |
| incoherence_yes_empty | 844 | 792 | 0.9384 | SIM |

Depois de padronizadas, features com esta forma funcionam como um **flag ligado/desligado**, não como uma escala de comportamento, e dominam a distância euclidiana. Duas delas — `unknown_rate_c` e `incoherence_yes_empty` — sinalizam o mesmo evento e são colineares por construção (ver Parte 1, §2).

### 6.2 O K com melhor silhouette separa quem usou `UNKNOWN`, não perfis de comportamento

`[MEDIDO]` — K = 2 (silhouette 0.5432, tamanhos [792, 52]). Médias por cluster:

| feature | 0 | 1 |
|---|---|---|
| yes_rate | 0.5807 | 0.5957 |
| unknown_rate_b | 0.0052 | 0.0104 |
| unknown_rate_c | 0.0000 | 0.0474 |
| n_distinct_categories | 4.2197 | 4.0769 |
| category_entropy | 0.7591 | 0.7456 |
| mean_labels_per_item | 1.3230 | 1.2727 |
| incoherence_yes_empty | 0.0000 | 0.0864 |

Cruzamento com o evento «usou `UNKNOWN` alguma vez»:

| cluster | False | True |
|---|---|---|
| 0 | 698 | 94 |
| 1 | 0 | 52 |

As features de comportamento propriamente dito são **quase idênticas** entre os grupos; o que os separa é um flag administrativo. Um anotador ter carregado `UNKNOWN` uma vez não é um perfil de comportamento — é um evento pontual de preenchimento.

`[LIMITAÇÃO]` — **o critério pré-registado é insensível a este modo de falha.** Foi desenhado contra clusters degenerados por *tamanho* (o `[883, 2, 1, 1]` das tentativas anteriores). Um flag binário com ~6% de prevalência passa nas duas condições. O critério cumpriu o que prometia e ainda assim deixou passar um resultado que não responde à pergunta. Fica registado para quem reutilizar este protocolo.

### 6.3 Sensibilidade: clustering só com as features contínuas

Removendo as features sinalizadas em 6.1 e mantendo as contínuas — `yes_rate`, `n_distinct_categories`, `category_entropy`, `mean_labels_per_item` (n = 844):

| K | tamanhos | min_frac | silhouette | veredito |
|---|---|---|---|---|
| 2 | [509, 335] | 0.3969 | 0.3415 | PASS |
| 3 | [139, 190, 515] | 0.1647 | 0.3099 | PASS |
| 4 | [279, 186, 111, 268] | 0.1315 | 0.2937 | PASS |
| 5 | [196, 234, 161, 105, 148] | 0.1244 | 0.2863 | PASS |
| 6 | [150, 159, 158, 98, 128, 151] | 0.1161 | 0.2755 | PASS |

Clusters muito mais equilibrados (fracção mínima 0.397 em K = 2, contra 0.062 na análise primária). O silhouette é menor precisamente porque já não há um flag binário a inflacioná-lo.

O que distingue cada cluster em K = 2 (média do cluster menos média global):

| cluster | yes_rate | n_distinct_categories | category_entropy | mean_labels_per_item |
|---|---|---|---|---|
| 0 | 0.0724 | 0.5533 | 0.0942 | 0.1247 |
| 1 | -0.1100 | -0.8408 | -0.1431 | -0.1895 |

**As quatro features movem-se todas na mesma direção** dentro de cada cluster: um grupo está acima da média em tudo, o outro abaixo em tudo. Isso é a assinatura de **uma única dimensão latente** cortada ao meio — anotadores mais expansivos (marcam mais YES, usam mais categorias, mais rótulos por item, entropia maior) contra anotadores mais parcimoniosos. Não são dois *tipos*: são as duas metades de um contínuo.

Composição demográfica desta partição (a substantiva), K = 2:

**gender_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | F | M |
|---|---|---|
| 0 | 55.2 | 44.8 |
| 1 | 46.0 | 54.0 |

**age_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | 18-22 | 23-45 | 46+ |
|---|---|---|---|
| 0 | 31.6 | 34.2 | 34.2 |
| 1 | 32.2 | 33.7 | 34.0 |

**ethnicities_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | Black or African American | Hispano or Latino | White or Caucasian |
|---|---|---|---|
| 0 | 10.0 | 26.5 | 56.2 |
| 1 | 10.4 | 19.7 | 61.5 |

**study_levels_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | Bachelor's degree | High school degree or equivalent | Master's degree |
|---|---|---|---|
| 0 | 50.5 | 27.9 | 16.5 |
| 1 | 45.4 | 31.3 | 17.6 |

**countries_annotators** (% por cluster; categorias com ≥5% nalgum cluster)

| cluster | Chile | Italy | Mexico | Poland | Portugal | South Africa | Spain | United Kingdom |
|---|---|---|---|---|---|---|---|---|
| 0 | 5.7 | 5.1 | 14.3 | 5.7 | 8.4 | 10.6 | 15.3 | 9.8 |
| 1 | 2.7 | 5.7 | 11.3 | 5.7 | 9.9 | 10.7 | 10.4 | 13.7 |

`[HIPÓTESE]` — esta é a estrutura substantiva e a da §5 é um artefacto de features indicadoras. **Não testada:** não existe critério externo que arbitre qual das duas partições é a verdadeira, e por desenho este diagnóstico não tem acesso a nenhum.

---

## 7. Relação com os artefactos das fases anteriores

**Enquadramento obrigatório:** correlação **descritiva**, não validação. Os métodos anteriores foram caracterizados como capturáveis por blocos de anotadores correlacionados (`reliability/outputs/stage1_report.md`, `perann_gate_report.md`). Concordar com eles não é prova de nada; discordar também não.

### NutmegEM theta (demo_base_K4) — n = 887 anotadores em comum

| feature | n | Spearman ρ | p |
|---|---|---|---|
| yes_rate | 887.0000 | -0.1447 | 0.0000 |
| unknown_rate_b | 887.0000 | 0.0640 | 0.0566 |
| unknown_rate_c | 887.0000 | 0.0276 | 0.4109 |
| n_distinct_categories | 844.0000 | -0.0548 | 0.1119 |
| category_entropy | 844.0000 | -0.0037 | 0.9142 |
| mean_labels_per_item | 844.0000 | 0.0192 | 0.5784 |
| incoherence_A_to_C | 790.0000 | — | — |
| incoherence_A_to_B | 790.0000 | — | — |
| incoherence_yes_empty | 844.0000 | 0.0238 | 0.4899 |
| yes_rate_en | 450.0000 | -0.0605 | 0.2002 |
| yes_rate_es | 456.0000 | -0.2310 | 0.0000 |
| lang_gap | 19.0000 | — | — |
| n_votes | 887.0000 | 0.1434 | 0.0000 |

### NutmegEM theta (demo_extended_K4) — n = 887 anotadores em comum

| feature | n | Spearman ρ | p |
|---|---|---|---|
| yes_rate | 887.0000 | -0.2024 | 0.0000 |
| unknown_rate_b | 887.0000 | 0.1146 | 0.0006 |
| unknown_rate_c | 887.0000 | 0.0563 | 0.0937 |
| n_distinct_categories | 844.0000 | -0.0644 | 0.0617 |
| category_entropy | 844.0000 | -0.0239 | 0.4879 |
| mean_labels_per_item | 844.0000 | 0.0158 | 0.6475 |
| incoherence_A_to_C | 790.0000 | — | — |
| incoherence_A_to_B | 790.0000 | — | — |
| incoherence_yes_empty | 844.0000 | 0.0534 | 0.1209 |
| yes_rate_en | 450.0000 | -0.1290 | 0.0061 |
| yes_rate_es | 456.0000 | -0.2803 | 0.0000 |
| lang_gap | 19.0000 | — | — |
| n_votes | 887.0000 | 0.1554 | 0.0000 |

### NutmegEM theta (single_truth_K1) — n = 887 anotadores em comum

| feature | n | Spearman ρ | p |
|---|---|---|---|
| yes_rate | 887.0000 | -0.0723 | 0.0313 |
| unknown_rate_b | 887.0000 | 0.0281 | 0.4035 |
| unknown_rate_c | 887.0000 | -0.0123 | 0.7140 |
| n_distinct_categories | 844.0000 | -0.1393 | 0.0000 |
| category_entropy | 844.0000 | -0.1174 | 0.0006 |
| mean_labels_per_item | 844.0000 | -0.0142 | 0.6803 |
| incoherence_A_to_C | 790.0000 | — | — |
| incoherence_A_to_B | 790.0000 | — | — |
| incoherence_yes_empty | 844.0000 | -0.0217 | 0.5284 |
| yes_rate_en | 450.0000 | -0.0081 | 0.8637 |
| yes_rate_es | 456.0000 | -0.1365 | 0.0035 |
| lang_gap | 19.0000 | — | — |
| n_votes | 887.0000 | 0.0124 | 0.7113 |

`[LIMITAÇÃO]` — **não existe** em `reliability/outputs/` nenhum ficheiro de competência per-anotador calculada sobre o EXIST real; o `perann_synthetic_gate.json` é do gate sintético e não tem IDs do EXIST. A comparação pedida com essa família de estimadores fica por fazer, por ausência do artefacto.

---
## 8. Veredito

### 8.1 Existem padrões comportamentais distinguíveis entre os anotadores do EXIST?

**SIM, parcialmente — e a parte que existe é um contínuo, não uma tipologia.**

`[MEDIDO]` — 5 das 12 features são dispersas pelo critério pré-registado (`yes_rate`, `category_entropy`, `yes_rate_en`, `yes_rate_es`, `lang_gap`); 2 são degeneradas (variância nula).

`[LIMITAÇÃO]` — dessa contagem, `yes_rate_en`, `yes_rate_es` e `lang_gap` são avaliadas sobre subpopulações de 450, 456 e **19** anotadores. Só `yes_rate` e `category_entropy` são dispersas sobre o painel completo. Não citar "5 de 12" sem esta ressalva.

`[MEDIDO]` — o clustering passa o critério pré-registado em 5 dos 5 valores de K testados (K=2, K=3, K=4, K=5, K=6). Melhor silhouette: 0.5432 em K = 2, tamanhos [792, 52].

**Mas o PASS não deve ser lido como "existem tipos de anotador".** A §6 mostra que a partição com melhor silhouette separa quem alguma vez carregou `UNKNOWN` na Task C — as features de comportamento são quase idênticas entre os dois grupos. 3 das 7 features retidas têm ≥ 80% de zeros e funcionam como indicador binário.

Removendo-as (análise post-hoc, §6.3), o clustering continua a passar em 5 de 5 valores de K, agora com clusters **equilibrados** (fracção mínima 0.397 em K = 2, contra 0.062 na análise primária) e silhouette mais baixo (0.3415).

`[MEDIDO]` — **a partição substantiva não é um proxy demográfico.** Na partição da §6.3, a composição por idade é praticamente idêntica nos dois grupos (31,6/34,2/34,2 contra 32,2/33,7/34,0), e etnia, escolaridade e país variam pouco. A maior diferença é de género (55,2% F contra 46,0% F) e continua modesta. O comportamento de anotação varia entre indivíduos, não entre os grupos demográficos que o modelo de memes já codifica no vetor demográfico.

**Formulação defensável para a tese:** os anotadores do EXIST diferem de forma mensurável ao longo de eixos contínuos — sobretudo a propensão a marcar YES (`yes_rate` cobre [0, 1] inteiro) e a forma como usam o espaço de categorias (`category_entropy` cobre [0,15 · 0,99]). Não há evidência de tipos discretos: as partições que o KMeans encontra ou são conduzidas por um flag administrativo (§6.2), ou são cortes num contínuo, sem separação forte (silhouette ≈ 0,28–0,34). `[HIPÓTESE]` — variação individual gradual, não subgrupos.

### 8.2 Quantos anotadores estão envolvidos, e em que tipo de anomalia?

`[MEDIDO]` — **66 anotadores distintos (7.4% do painel)** disparam pelo menos um critério de anomalia. Destes, 64 estão no estrato ≥ 15 votos.

- **yes_rate quase-constante (>0,90 ou <0,10)** — 61
- **usa sempre a mesma categoria (n_distinct_categories == 1)** — 0
- **qualquer contradição A->C (> 0)** — 0
- **contradição A->C sistemática (> 0,10)** — 0
- **contradição A->B sistemática (> 0,10)** — 0
- **comportamento muito diferente entre línguas (lang_gap > 0,30)** — 5

`[MEDIDO]` — **sobreposição zero**: 0 anotadores disparam ≥ 2 critérios. Os conjuntos são disjuntos. Não existe neste painel o perfil "anómalo em várias dimensões" que se procurava — quem tem taxa de YES extrema não é quem diverge entre línguas, e ninguém acumula.

Uma única família de anomalia tem população não-trivial: a **taxa de YES extrema** (61 anotadores, 6,9%). As restantes têm população zero por razões estruturais: `n_distinct_categories` tem mínimo observado 2 (ninguém usa uma só categoria) e as incoerências são nulas em todo o painel (§8.3).

`[LIMITAÇÃO]` — dos 61, dois estão no estrato < 15 votos, onde `yes_rate = 1,00` exige apenas 9 votos YES seguidos. E `lang_gap > 0,30` descreve 5 anotadores de uma subpopulação de 19 — não é uma taxa do painel.

### 8.3 A incoerência hierárquica existe em grau não-trivial, ou é residual?

**Nem uma coisa nem outra: é INEXISTENTE, e por construção.**

`[MEDIDO]` — `incoherence_A_to_C` e `incoherence_A_to_B` valem exactamente 0 nos 887 anotadores. Ao nível do voto (n = 23 904) não há um único caso de `NO` na Task A acompanhado de categoria de sexismo ou de intenção positiva. As contagens batem certo ao voto: 10 618 votos `NO` ↔ 10 618 `-` na Task B ↔ 10 618 `-` na Task C.

Duas consequências para a tese:

1. A feature que a especificação identificava como a mais informativa — a única próxima de um *erro objetivo*, independente de qualquer verdade — **não é computável neste corpus**. Não porque o método falhe, mas porque o fenómeno que mediria não ocorre. É um resultado, não uma lacuna.
2. `[HIPÓTESE]` a hierarquia é imposta pela interface de anotação. Se assim for, a ausência de incoerência **não é evidência de qualidade de anotação** — é evidência de um constrangimento de UI. Não escrever "os anotadores do EXIST são hierarquicamente coerentes" sem esta ressalva. Distinguir as duas leituras exigiria acesso à ferramenta de anotação ou às guidelines operacionais, que não temos.

A única variante que sobrevive, `incoherence_yes_empty` (0 a 0,25), capta os 65 votos `YES` + `UNKNOWN` em C e é colinear com `unknown_rate_c`. Mede uso de `UNKNOWN`, não incoerência.

---

## 9. Limites

1. **Este diagnóstico descreve padrões; não julga legitimidade.** Um anotador com 95% de YES pode ser um spammer **ou** genuinamente mais sensível a sexismo. Estas features não distinguem os dois casos e nunca distinguirão — por desenho, não medem acerto. Sob o paradigma LeWiDi, a segunda leitura é tão admissível como a primeira, e os 61 anotadores da §8.2 não são um conjunto de suspeitos.
2. **Nenhum down-weighting é justificado por estes números.** Nada aqui é uma medida de fiabilidade. Um perfil atípico é um perfil atípico; a validade dos votos não decorre daqui em nenhum sentido.
3. **O critério pré-registado do clustering é insensível a features indicadoras** (§6.2). Passou uma partição conduzida por um flag binário com ~6% de prevalência. Quem reutilizar este protocolo deve acrescentar uma condição sobre a forma das features, não só sobre o tamanho dos clusters.
4. **A §6.3 é post-hoc.** Foi decidida depois de ver os resultados. Não tem o estatuto do resultado pré-registado e não pode ser reportada como se tivesse.
5. **As correlações da §7 não validam nem invalidam os métodos anteriores.** Todas as associações significativas são fracas (|ρ| ≤ 0,24). Os anotadores que o NutmegEM marcou como menos fiáveis **não são** os de perfil comportamental anómalo. Como o NutmegEM foi caracterizado como capturável por blocos correlacionados, nem a concordância nem a discordância com ele transportam garantia.
6. **O estrato < 15 votos (11 anotadores) não suporta conclusões.** Extremos de taxa são esperados por amostragem com denominadores de 9 a 14.
7. **`lang_gap` descreve 19 anotadores**, não o painel: EN e ES são recrutados quase disjuntamente (431 / 437 / 19). Qualquer afirmação sobre comportamento inter-língua dentro do mesmo anotador vive nesta amostra minúscula.
8. **As features de categorias dependem da Task C**, cujo denominador é o número de itens que o anotador marcou como sexistas. Quem marca poucos YES tem estas features estimadas sobre poucos itens, ou a NaN (43 anotadores). A incerteza não é uniforme pelo painel.
