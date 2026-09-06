# Parte 1 — Extração do perfil comportamental por anotador (EXIST 2026, Task 2)

**Estado: entregue, à espera de validação do autor.** Este documento cobre APENAS a Parte 1
(extração). A análise descritiva (Parte 2) não foi corrida e nada aqui é conclusão sobre
padrões, clusters ou anomalias.

- **Código:** `reliability/behavioral_profile.py`
- **Artefacto:** `reliability/outputs/behavioral_profiles.csv` (887 linhas × features + demografia)
- **Fonte:** `EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json`
- **Data:** 2026-08-27

**Convenção** (a mesma de `reliability/NOTAS_PARA_TESE.md`):
`[MEDIDO]` — número observado, com o artefacto que o produziu ·
`[HIPÓTESE]` — interpretação plausível, ainda não testada; **não escrever como facto** ·
`[LIMITAÇÃO]` — restrição a declarar explicitamente no texto.

---

## 1. O que foi construído

Perfil de comportamento **agregado** por anotador. Restrição de desenho, verificada feature
a feature: **nenhuma quantidade depende de outro anotador, de uma verdade estimada, ou de
concordância par-a-par.** Cada feature calcula-se só a partir dos votos do próprio anotador.

Isto responde à razão de a via existir: a clusterização por concordância par-a-par é
impossível neste regime (~0,15 itens partilhados por par; matriz anotador × item ~99%
esparsa; tentativas anteriores deram clusters degenerados `[883, 2, 1, 1]`).

**Reutilização, não cópia:** `_normalize_apostrophes` e a resolução da demografia por moda
são importados de `reliability/exist_data.py`. Como o `load_task_a` desse módulo descarta as
Tasks B/C, este faz uma segunda passagem alinhada por `(item, posição)` sobre o JSON cru.

**Carregamento verificado** `[MEDIDO]`:

| | valor |
|---|---|
| itens | 3 984 |
| votos | 23 904 |
| anotadores | 887 |
| votos/anotador | 9 a 54 (média 26,9); 11 anotadores com < 15 |
| listas desalinhadas (`annotators` vs `labels_task2_*`) | **0** |
| demografia inconsistente resolvida por moda | 2 IDs — `Annotator_125` (país), `Annotator_187` (escolaridade) |

---

## 2. A regra de coerência hierárquica (o ponto a validar)

Segundo as guidelines do EXIST, `-` é o rótulo atribuído a instâncias **não-sexistas** nas
subtasks 2.2 e 2.3. Logo `NO` na Task A seguido de `-` em B/C é **coerente**, não é erro.
Regra tal como está no código (`COHERENCE_RULE`, `behavioral_profile.py`):

```
NO na Task A  +  `-` na B/C                     -> COERENTE (não conta)
NO na Task A  +  >=1 categoria de SEXISM_SET    -> incoherence_A_to_C
NO na Task A  +  DIRECT ou JUDGEMENTAL na B     -> incoherence_A_to_B
YES na Task A +  nem `-` nem categoria real     -> incoherence_yes_empty
UNKNOWN                                          -> NUNCA conta como incoerência A->C ou
                                                    A->B; conta como "vazio" no caso YES
                                                    (é o único valor que satisfaz
                                                    "nem `-` nem categoria")
```

`SEXISM_SET` = as 5 categorias reais (`IDEOLOGICAL-INEQUALITY`, `MISOGYNY-NON-SEXUAL-VIOLENCE`,
`OBJECTIFICATION`, `SEXUAL-VIOLENCE`, `STEREOTYPING-DOMINANCE`). `-` e `UNKNOWN` estão
explicitamente fora do conjunto.

`[LIMITAÇÃO]` — `incoherence_yes_empty` capta, por esta definição, exactamente os votos
`YES` + `UNKNOWN` na Task C. É colinear com `unknown_rate_c` restrita aos votos YES. As duas
**não são evidência independente** e não podem ser somadas como dois sinais.

---

## 3. As 12 features e as suas guardas

Features abaixo do denominador mínimo devolvem **NaN explícito, nunca 0**.

| # | feature | denominador | limiar mínimo |
|---|---|---|---|
| 1 | `yes_rate` | todos os votos | — (mín. observado 9) |
| 2 | `n_votes` | — | — |
| 3 | `unknown_rate_b` | todos os votos | — |
| 4 | `unknown_rate_c` | todos os votos | — |
| 5 | `n_distinct_categories` | itens com ≥1 categoria real | `MIN_SEXIST_ITEMS = 5` |
| 6 | `category_entropy` | idem | `MIN_SEXIST_ITEMS = 5` |
| 7 | `mean_labels_per_item` | idem | `MIN_SEXIST_ITEMS = 5` |
| 8 | `incoherence_A_to_C` | votos `NO` | `MIN_NO_VOTES = 5` |
| 9 | `incoherence_A_to_B` | votos `NO` | `MIN_NO_VOTES = 5` |
| 10 | `incoherence_yes_empty` | votos `YES` | `MIN_YES_VOTES = 5` |
| 11 | `yes_rate_en` / `yes_rate_es` | votos nessa língua | `MIN_VOTES_PER_LANG = 5` |
| 12 | `lang_gap` | ambas as línguas | `MIN_VOTES_PER_LANG = 5` em **cada** |

`category_entropy` é a entropia de Shannon normalizada por `log(5)`: 0 = usa sempre a mesma
categoria, 1 = distribuição uniforme pelas cinco.

**Decisão documentada sobre `n_distinct_categories`:** um anotador sem votos sexistas
suficientes recebe **NaN**, não 0. Assim o critério de anomalia `== 1` (usa sempre a mesma
categoria) nunca se confunde com ausência de dados.

---

## 4. Três factos estruturais que reduzem o alcance do diagnóstico

Descobertos durante a extração. Mudam o que a Parte 2 pode perguntar, por isso ficam aqui.

### 4.1 A coerência hierárquica é perfeita — a família de features "erro objetivo" morre

`[MEDIDO]` — cross-tab ao nível do voto, n = 23 904 (`structural_audit`, impresso pelo módulo):

```
A × B:   NO|'-' 10 618  ·  YES|DIRECT 8 471  ·  YES|JUDGEMENTAL 4 688  ·  YES|UNKNOWN 127
A × C:   NO|DASH 10 618  ·  YES|CATEGORIA 13 221  ·  YES|UNKNOWN 65
```

Zero votos `NO` + categoria de sexismo. Zero votos `NO` + `DIRECT`/`JUDGEMENTAL`.
Consequência directa `[MEDIDO]`:

> `incoherence_A_to_C` e `incoherence_A_to_B` valem **0,0000 nos 887 anotadores**
> (`nunique = 1`, min = max = 0).

Isto elimina a família de features que a especificação designava como "as mais informativas —
erro estrutural, não de consenso". Não há erro estrutural neste dataset para medir.

`[HIPÓTESE]` — a hierarquia foi imposta pela interface de anotação. A coincidência exacta
das contagens (10 618 = 10 618 = 10 618) é consistente com um constrangimento de ferramenta,
mas **não distingue** constrangimento de disciplina do anotador. Não escrever como facto.

`[LIMITAÇÃO]` operacional: features de variância zero não podem ser padronizadas (divisão por
zero) e têm de ser **excluídas do KMeans**. Ficam retidas no CSV e reportadas — a regra de
exclusão vai pré-registada no código da Parte 2, não aplicada em silêncio.

### 4.2 Os painéis EN e ES são quase disjuntos — `lang_gap` fica sem população

`[MEDIDO]` — `audit['annotators_by_lang']`:

| | anotadores |
|---|---|
| só ES | 437 |
| só EN | 431 |
| **ambas** | **19** |

`yes_rate_en` é NaN para 437, `yes_rate_es` para 431, e `lang_gap` para **868 dos 887
(97,9%)**. A pergunta 2.2 sobre `lang_gap > 0,30` responde-se sobre n = 19 — reportável,
mas sem poder estatístico e sem generalização ao painel.

`[LIMITAÇÃO]` — o desenho do EXIST recruta por língua. Comparação de comportamento
inter-língua dentro do mesmo anotador é, neste corpus, quase inexistente por construção.

### 4.3 `incoherence_yes_empty` sobrevive, mas é fraca e não é independente

Varia (0 a 0,25, 24 valores distintos), logo entra na análise. Mas capta exactamente os
65 votos `YES` + `UNKNOWN` em C — a mesma informação que `unknown_rate_c`. Ver `[LIMITAÇÃO]`
da §2.

---

## 5. Cobertura de NaN e variância (887 anotadores)

`[MEDIDO]` — `reliability/outputs/behavioral_profiles.csv`.

| feature | NaN | % | valores distintos | min | max |
|---|---|---|---|---|---|
| `yes_rate` | 0 | 0,0 | 104 | 0,0000 | 1,0000 |
| `n_votes` | 0 | 0,0 | 13 | 9 | 54 |
| `unknown_rate_b` | 0 | 0,0 | 14 | 0,0000 | 0,1111 |
| `unknown_rate_c` | 0 | 0,0 | 12 | 0,0000 | 0,1111 |
| `n_distinct_categories` | 43 | 4,8 | 4 | **2** | 5 |
| `category_entropy` | 43 | 4,8 | 697 | 0,1522 | 0,9937 |
| `mean_labels_per_item` | 43 | 4,8 | 200 | 1,0000 | 4,2500 |
| `incoherence_A_to_C` | 97 | 10,9 | **1** | 0,0000 | 0,0000 |
| `incoherence_A_to_B` | 97 | 10,9 | **1** | 0,0000 | 0,0000 |
| `incoherence_yes_empty` | 43 | 4,8 | 24 | 0,0000 | 0,2500 |
| `yes_rate_en` | 437 | 49,3 | 45 | 0,0000 | 1,0000 |
| `yes_rate_es` | 431 | 48,6 | 93 | 0,0370 | 1,0000 |
| `lang_gap` | 868 | 97,9 | 15 | 0,0000 | 0,4444 |

Duas leituras adiantadas, ambas puramente aritméticas (não são a análise da Parte 2):

- **`n_distinct_categories` tem mínimo 2.** O critério de anomalia da §2.2 da especificação
  (`== 1`, "usa sempre a mesma categoria") terá contagem **zero** — nenhum anotador do painel
  usa uma só categoria.
- **`yes_rate` cobre o intervalo completo [0, 1]** e `category_entropy` quase todo
  [0,15 · 0,99]. São os dois eixos onde há dispersão para investigar.

**Sobram 7 features com variância real** para a análise e o clustering: `yes_rate`,
`unknown_rate_b`, `unknown_rate_c`, `n_distinct_categories`, `category_entropy`,
`mean_labels_per_item`, `incoherence_yes_empty` (+ `n_votes` como estratificador, não como
feature de comportamento).

---

## 6. Cinco anotadores de exemplo

`[MEDIDO]` — saída de `python3 -m reliability.behavioral_profile`.

| annotator_id | n_votes | yes_rate | unk_b | unk_c | n_dist_cat | cat_entropy | mean_lab/item | inc_A→C | inc_A→B | inc_yes_empty | yes_en | yes_es | lang_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Annotator_1 | 27 | 0,7778 | 0,0000 | 0,0000 | 5 | 0,9564 | 2,5714 | 0,0000 | 0,0000 | 0,0000 | NaN | 0,7778 | NaN |
| Annotator_10 | 27 | 0,4444 | 0,0000 | 0,0000 | 4 | 0,8551 | 1,6667 | 0,0000 | 0,0000 | 0,0000 | NaN | 0,4444 | NaN |
| Annotator_100 | 27 | 0,6296 | 0,0000 | 0,0000 | 4 | 0,7368 | 1,0588 | 0,0000 | 0,0000 | 0,0000 | NaN | 0,6296 | NaN |
| Annotator_101 | 27 | 0,3704 | 0,0000 | 0,0000 | 4 | 0,8043 | 1,1000 | 0,0000 | 0,0000 | 0,0000 | NaN | 0,3704 | NaN |
| Annotator_102 | 27 | 0,8519 | 0,0000 | 0,0000 | 3 | 0,6103 | 1,0435 | **NaN** | **NaN** | 0,0000 | NaN | 0,8519 | NaN |

`Annotator_102` tem `inc_A→C` e `inc_A→B` a NaN porque só emitiu 4 votos `NO`
(< `MIN_NO_VOTES = 5`) — a guarda de denominador mínimo a funcionar como especificado.
Os cinco são anotadores ES, daí `yes_rate_en` NaN (ver §4.2).

---

## 7. Nota de implementação

`load_votes` devolve `demo_raw` à parte e **não** o guarda em `votes.attrs`. O pandas propaga
`attrs` copiando-os para cada objecto derivado; com 887 anotadores × 5 `Counter` lá dentro, um
`groupby` por anotador custava ~20 s só em cópias e a extração completa não terminava em 350 s.
Sem os `attrs`, a extração corre em ~4 s. Registado porque o mesmo padrão pode reaparecer
noutro módulo deste repositório.

---

## 8. O que este documento NÃO diz

- Não diz se existem padrões comportamentais distinguíveis. Isso é a Parte 2.
- Não julga a legitimidade de nenhum perfil. Um anotador com 95% de YES pode ser ruído **ou**
  genuinamente mais sensível a sexismo; estas features não distinguem os dois casos e nunca
  distinguirão.
- Não justifica down-weighting de ninguém.

---

## 9. Ponto de decisão para o autor

A validar antes de correr a Parte 2:

1. A regra de coerência da §2 — em particular o tratamento de `-` (coerente) e de `UNKNOWN`
   (nunca incoerência A→B/A→C; conta como "vazio" só no ramo YES).
2. Os limiares de denominador mínimo da §3 (todos a 5).
3. Que as duas features de incoerência A→\* sejam **retidas e reportadas como zero medido**,
   e **excluídas do KMeans** por variância nula.
4. Que `lang_gap` seja reportada sobre os 19 anotadores bilingues, com a ressalva de n
   pequeno, em vez de removida.
