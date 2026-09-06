# Observações sobre a primeira corrida do diagnóstico comportamental

> **NOTA DE LEITURA — os defeitos descritos nas §1–§2 JÁ FORAM CORRIGIDOS.** Ver §5 para o
> estado final. As §1–§4 descrevem a primeira corrida e mantêm-se como registo de auditoria:
> o aviso "a §7 não deve ser lida" aplica-se à versão antiga do relatório, não à atual.

**Estado congelado em 2026-08-28, antes de qualquer correção.** Este documento regista o que
a primeira corrida de `reliability/run_behavioral_diagnosis.py` produziu, **incluindo os dois
defeitos que a invalidam parcialmente**. Existe para que a correção seguinte seja auditável:
o que estava errado, porquê, e o que muda.

`[MEDIDO]` — número observado · `[HIPÓTESE]` — interpretação não testada · `[LIMITAÇÃO]` —
restrição a declarar.

**Ambiente:** local, sem GPU. O módulo `reliability/` é offline por desenho (`exist_data.py`
documenta que não importa `memes_core.py` justamente para evitar `torch`/`pyevall`).
Dependências: `pandas`, `numpy`, `scikit-learn`, `scipy`. Corrida completa: ~4 s.

---

## 1. Os dois defeitos

### 1.1 DEFEITO — o veredito da §7 foi escrito a assumir que o clustering ia falhar

O texto da secção "Veredito" em `run_behavioral_diagnosis.py` (`build_verdict_section`) contém
prosa fixa que afirma **"não há estrutura de grupos"**. Foi escrita antes de correr, na
expectativa de repetir o resultado degenerado das tentativas anteriores (`[883, 2, 1, 1]`).

`[MEDIDO]` — aconteceu o contrário. **Todos os K de 2 a 6 passam** o critério pré-registado:

| K | tamanhos | min. fracção | silhouette | veredito |
|---|---|---|---|---|
| 2 | [792, 52] | 0,062 | **0,5432** | PASS |
| 3 | [52, 317, 475] | 0,062 | 0,2960 | PASS |
| 4 | [424, 286, 83, 51] | 0,060 | 0,3226 | PASS |
| 5 | [115, 170, 426, 82, 51] | 0,060 | 0,2962 | PASS |
| 6 | [51, 231, 163, 89, 228, 82] | 0,060 | 0,2806 | PASS |

O relatório gerado (`behavioral_diagnosis.md`) tem, portanto, tabelas corretas na §5 e uma
§7.1 que as contradiz. **A §7 do ficheiro atual não deve ser lida nem citada.**

Nota metodológica: o critério PASS/FAIL estava pré-registado no código antes de correr e
**não foi alterado**. O defeito é de redação do veredito, não de mudança de critério.

### 1.2 DEFEITO — o K=2 vencedor não é uma tipologia comportamental

`[MEDIDO]` — médias por cluster, K=2 (n=844):

| feature | cluster 0 (n=792) | cluster 1 (n=52) |
|---|---|---|
| `yes_rate` | 0,5807 | 0,5957 |
| `unknown_rate_b` | 0,0052 | 0,0104 |
| **`unknown_rate_c`** | **0,0000** | **0,0474** |
| `n_distinct_categories` | 4,2197 | 4,0769 |
| `category_entropy` | 0,7591 | 0,7456 |
| `mean_labels_per_item` | 1,3230 | 1,2727 |
| **`incoherence_yes_empty`** | **0,0000** | **0,0864** |

Cruzamento com "usou UNKNOWN alguma vez":

```
cluster \ tem algum UNKNOWN    False   True
        0                        698     94
        1                          0     52
```

O cluster 1 é **exatamente o conjunto dos anotadores que carregaram `UNKNOWN` na Task C**.
As features de comportamento propriamente dito são quase idênticas entre os dois grupos
(`yes_rate` 0,581 vs 0,596; entropia 0,759 vs 0,746).

**Causa** `[MEDIDO]`: `unknown_rate_c` e `incoherence_yes_empty` são quase binárias —
respetivamente 834/887 e 792/844 valores exatamente zero. Depois de padronizadas, funcionam
como um **indicador**, não como uma escala. Duas features indicadoras do mesmo evento (são
colineares por construção, ver §2 do relatório da Parte 1) dominam a distância euclidiana e
produzem um silhouette alto (0,5432) que mede a separação de um flag binário, não de um
perfil de comportamento.

`[LIMITAÇÃO]` — o critério pré-registado (silhouette > 0,25 e nenhum cluster < 5%) é
**insensível a este modo de falha**. Foi desenhado contra clusters degenerados por tamanho,
e um flag binário com 6% de prevalência passa nas duas condições. O critério cumpriu o que
prometia e mesmo assim deixa passar um resultado que não responde à pergunta.

### 1.3 Análise de sensibilidade (post-hoc, declarada como tal)

Removendo as três features com inflação de zeros (`unknown_rate_b`, `unknown_rate_c`,
`incoherence_yes_empty`) e mantendo as quatro contínuas — `yes_rate`,
`n_distinct_categories`, `category_entropy`, `mean_labels_per_item` — n=844:

| K | tamanhos | min. fracção | silhouette | veredito |
|---|---|---|---|---|
| 2 | [509, 335] | 0,397 | 0,3415 | PASS |
| 3 | [139, 190, 515] | 0,165 | 0,3099 | PASS |
| 4 | [279, 186, 111, 268] | 0,132 | 0,2937 | PASS |
| 5 | [196, 234, 161, 105, 148] | 0,124 | 0,2863 | PASS |
| 6 | [150, 159, 158, 98, 128, 151] | 0,116 | 0,2755 | PASS |

Clusters muito mais equilibrados (fracção mínima 0,40 em K=2, contra 0,062). O silhouette é
menor em K=2 (0,342 vs 0,543) precisamente porque já não há um flag binário a inflacioná-lo.

`[HIPÓTESE]` — esta é a estrutura substantiva, e a do conjunto completo é um artefacto de
features indicadoras. Não testada: não há critério externo que arbitre qual das duas partições
é "a verdadeira", e por desenho este diagnóstico não tem acesso a nenhum.

**Esta análise é post-hoc.** Não estava pré-registada. O resultado pré-registado continua a
ser o primário; a sensibilidade entra como secundária declarada, não como substituição.

---

## 2. Aviso numérico investigado e descartado

A corrida emite `RuntimeWarning: overflow encountered in matmul` (e `invalid value`,
`divide by zero`) a partir de `sklearn/utils/extmath.py:227`.

`[MEDIDO]` — é **espúrio**. Verificações feitas:

- a matriz padronizada é finita, com amplitude [-3,758 · 10,504];
- recalculado o silhouette pelo caminho sem BLAS (`scipy.spatial.distance.pdist` +
  `metric='precomputed'`), para todos os K:

| K | silhouette (BLAS) | silhouette (precomputed) | Δ |
|---|---|---|---|
| 2 | 0,543230 | 0,543230 | 3,8e-14 |
| 3 | 0,295985 | 0,295985 | 1,7e-13 |
| 4 | 0,322587 | 0,322587 | 4,4e-13 |
| 5 | 0,296176 | 0,296176 | 4,0e-13 |
| 6 | 0,280634 | 0,280634 | 4,8e-13 |

`[HIPÓTESE]` — flag de vírgula flutuante levantada pelo BLAS Accelerate do macOS em operações
vetorizadas com lanes não usadas. Não afeta os resultados. Correção: silenciar com
`np.errstate` **no ponto exato da chamada** e documentar, nunca com um filtro global de
warnings.

---

## 3. Resultados que se mantêm válidos

Estes não dependem do clustering e não são afetados pelos defeitos acima.

### 3.1 Dispersão por feature (§2 do relatório)

| feature | veredito | n distintos | bin modal | IQR/amplitude |
|---|---|---|---|---|
| `yes_rate` | **dispersa** | 104 | 0,1984 | 0,2986 |
| `unknown_rate_b` | concentrada | 14 | 0,8805 | 0,0000 |
| `unknown_rate_c` | concentrada | 12 | 0,9402 | 0,0000 |
| `n_distinct_categories` | concentrada | 4 | 0,4609 | 0,3333 |
| `category_entropy` | **dispersa** | 697 | 0,2358 | 0,2476 |
| `mean_labels_per_item` | concentrada | 200 | 0,6149 | 0,1238 |
| `incoherence_A_to_C` | **degenerada** | 1 | 1,0000 | 0,0000 |
| `incoherence_A_to_B` | **degenerada** | 1 | 1,0000 | 0,0000 |
| `incoherence_yes_empty` | concentrada | 24 | 0,9384 | 0,0000 |
| `yes_rate_en` | dispersa | 45 | 0,2067 | 0,2963 |
| `yes_rate_es` | dispersa | 93 | 0,1842 | 0,3299 |
| `lang_gap` | dispersa (n=19) | 15 | 0,2632 | 0,5000 |

5 dispersas · 5 concentradas · 2 degeneradas.

Observação `[LIMITAÇÃO]` — `yes_rate_en`, `yes_rate_es` e `lang_gap` são classificadas
"dispersas" sobre subpopulações de 450, 456 e **19** anotadores. O veredito de `lang_gap`
sobre n=19 não é comparável com os das features de população completa e não deve entrar na
contagem "5 dispersas" sem essa ressalva.

### 3.2 Anomalias (§3 do relatório)

| regra | n | % de 887 | não avaliáveis (NaN) |
|---|---|---|---|
| `yes_rate` quase-constante (>0,90 ou <0,10) | **61** | 6,9 | 0 |
| usa sempre a mesma categoria (`n_distinct_categories == 1`) | **0** | 0,0 | 43 |
| qualquer contradição A→C (> 0) | **0** | 0,0 | 97 |
| contradição A→C sistemática (> 0,10) | **0** | 0,0 | 97 |
| contradição A→B sistemática (> 0,10) | **0** | 0,0 | 97 |
| `lang_gap` > 0,30 | **5** | 0,6 | 868 |

**Sobreposição: zero.** Nenhum anotador dispara dois critérios. Os 61 de taxa extrema e os 5
de discrepância entre línguas são conjuntos disjuntos. Não existe um perfil "anómalo em
várias dimensões" neste painel.

IDs de `lang_gap > 0,30` (n=5, dos 19 bilingues): `Annotator_11`, `Annotator_148`,
`Annotator_187`, `Annotator_199`, `Annotator_285`.

### 3.3 Estratificação por volume de votos (§4 do relatório)

| regra | < 15 votos | ≥ 15 votos |
|---|---|---|
| `yes_rate` quase-constante | 2 / 11 (18,2%) | 59 / 876 (6,7%) |
| todas as outras | 0 | 0 (exceto `lang_gap`: 5 / 876) |

`[LIMITAÇÃO]` — com 9 a 14 votos, `yes_rate = 1,00` exige apenas 9 votos YES seguidos. No
estrato baixo, "quase-constante" e "amostra pequena" não são distinguíveis. Os 2 casos
(`Annotator_885`, `Annotator_886`) não sustentam conclusão.

### 3.4 Correlação com o θ do NutmegEM (§6 do relatório)

`[MEDIDO]` — Spearman contra `theta_taskA_demo_base_K4`, n=887:

| feature | ρ | p |
|---|---|---|
| `yes_rate` | **−0,1447** | < 0,001 |
| `yes_rate_es` | **−0,2310** | < 0,001 |
| `n_votes` | +0,1434 | < 0,001 |
| `unknown_rate_b` | +0,0640 | 0,057 |
| `unknown_rate_c` | +0,0276 | 0,411 |
| `n_distinct_categories` | −0,0548 | 0,112 |
| `category_entropy` | −0,0037 | 0,914 |
| `mean_labels_per_item` | +0,0192 | 0,578 |
| `incoherence_yes_empty` | +0,0238 | 0,490 |
| `yes_rate_en` | −0,0605 | 0,200 |

Leitura: as correlações significativas são **fracas** (|ρ| ≤ 0,23). Os anotadores que o
NutmegEM marcou como menos fiáveis **não são** os que têm perfis comportamentais anómalos —
a sobreposição é ténue e explica pouca variância.

`[LIMITAÇÃO]` — enquadramento obrigatório: correlação descritiva, **não validação**. O
NutmegEM foi caracterizado como capturável por blocos correlacionados; concordar ou discordar
com ele não transporta garantia nenhuma.

`ρ(n_votes, θ) = +0,143` é um alerta próprio: o θ do NutmegEM cresce com o número de votos do
anotador. `[HIPÓTESE]` — não é competência, é redução de variância na estimação. Já consistente
com o diagnóstico de fragmentação registado em `stage1_report.md`.

### 3.5 Artefacto em falta

`[LIMITAÇÃO]` — não existe em `reliability/outputs/` nenhum ficheiro de competência
per-anotador calculada sobre o EXIST **real**. O `perann_synthetic_gate.json` é do gate
sintético e não tem IDs do EXIST. A comparação pedida com essa família de estimadores fica
por fazer, por ausência do artefacto — não por omissão de análise.

---

## 4. O que a correção vai fazer

1. Silenciar o aviso do BLAS com `np.errstate` no ponto da chamada, com comentário a apontar
   para a §2 deste documento.
2. Reescrever a §7 (Veredito) inteiramente a partir dos números, sem prosa pré-escrita.
3. Acrescentar ao `behavioral_diagnosis.md` uma secção nova com a diagnose do K=2 (§1.2) e a
   análise de sensibilidade (§1.3), marcada como **post-hoc, não pré-registada**.
4. Manter o resultado pré-registado como primário. O critério **não** é alterado
   retroativamente; a sua insensibilidade a features indicadoras é declarada como limitação.

**O que NÃO muda:** os critérios pré-registados, a extração da Parte 1, os limiares de
anomalia, e as secções §1 a §6 do relatório (todas verificadas acima como válidas).

---

## 5. Estado após a correção (2026-08-28)

Os quatro pontos da §4 foram executados. `reliability/outputs/behavioral_diagnosis.md`
foi regenerado e **já não contém o veredito defeituoso**.

| item | estado |
|---|---|
| Aviso do BLAS silenciado com `np.errstate` local + comentário | feito |
| §7 (Veredito) reescrita a partir dos números → agora §8 | feito |
| Secção nova §6 "O que a partição primária está realmente a separar (post-hoc)" | feito |
| Resultado pré-registado mantido como primário; critério não alterado | feito |

**Renumeração:** o relatório passou a ter 9 secções. A antiga §6 (artefactos anteriores) é
agora §7; o veredito é §8; os limites são §9. A §6 é nova.

### 5.1 Achado acrescentado pela análise post-hoc

Na partição de sensibilidade (K=2, features contínuas), **as quatro features movem-se todas na
mesma direção**: um grupo está acima da média em tudo (`yes_rate` +0,072, `n_distinct_categories`
+0,553, `category_entropy` +0,094, `mean_labels_per_item` +0,125), o outro abaixo em tudo.

`[MEDIDO]` — é a assinatura de **uma única dimensão latente cortada ao meio**, não de dois
tipos: anotadores expansivos (marcam mais YES, usam mais categorias, mais rótulos por item)
contra parcimoniosos. Sustenta a formulação "contínuo, não tipologia" do veredito.

### 5.2 A partição substantiva não é um proxy demográfico

`[MEDIDO]` — composição demográfica dos dois clusters da §6.3:

| campo | cluster 0 | cluster 1 |
|---|---|---|
| idade 18-22 / 23-45 / 46+ | 31,6 / 34,2 / 34,2 | 32,2 / 33,7 / 34,0 |
| género F / M | 55,2 / 44,8 | 46,0 / 54,0 |
| etnia (White / Hispano / Black) | 56,2 / 26,5 / 10,0 | 61,5 / 19,7 / 10,4 |
| escolaridade (Bachelor / HS / Master) | 50,5 / 27,9 / 16,5 | 45,4 / 31,3 / 17,6 |

A idade é praticamente idêntica. A maior diferença é de género e continua modesta. O
comportamento de anotação varia **entre indivíduos**, não entre os grupos demográficos que o
modelo de memes já codifica no vetor demográfico.

`[HIPÓTESE]` — se o eixo expansivo/parcimonioso é ortogonal à demografia, então o vetor
demográfico do modelo não captura esta fonte de variação. Não testada aqui: exigiria medir o
efeito no modelo, o que está fora do âmbito deste diagnóstico (que não treina nada).

### 5.3 Estatuto epistémico dos dois resultados

| | análise primária (§5) | sensibilidade (§6.3) |
|---|---|---|
| pré-registada? | **sim** | **não** — post-hoc |
| K que passam | 5 de 5 | 5 de 5 |
| melhor silhouette | 0,5432 (K=2) | 0,3415 (K=2) |
| fracção mínima de cluster | 0,062 | 0,397 |
| o que separa | flag `UNKNOWN` | eixo expansivo/parcimonioso |
| citável como resultado principal | sim, com a §6.2 ao lado | **não sozinha** |

Regra para a tese: a análise primária é a que tem estatuto pré-registado, mas **não pode ser
reportada sem a §6.2**, que mostra o que ela realmente separou. A sensibilidade é mais
interpretável mas é post-hoc e tem de ser rotulada como tal em qualquer tabela.
