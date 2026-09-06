# Validação da dimensão latente "expansivo ↔ parcimonioso"

**O que este documento é:** teste estatístico, ao nível dos anotadores individuais, da afirmação feita em `behavioral_diagnosis.md` §6.3 e §8.1 — a de que as quatro features contínuas formam uma dimensão latente única. A evidência original foi observada **entre centróides do KMeans**, o que é parcialmente circular: o KMeans agrupa por proximidade nesse mesmo espaço.

- **Fonte:** `reliability/outputs/behavioral_profiles.csv` (lido, não regenerado)
- **Código:** `reliability/validate_latent_dimension.py` · seed 42
- **n efetivo:** 844 anotadores sem NaN nas quatro features (de 887)
- **KMeans e labels de cluster não são usados em nenhum ponto desta análise.**

`[MEDIDO]` — número observado · `[HIPÓTESE]` — interpretação não testada · `[LIMITAÇÃO]` — restrição a declarar.

---

## 1. Veredito

### 1.1 As quatro features correlacionam-se entre si ao nível individual?

`[MEDIDO]` — entre as três features de categoria, o |ρ| máximo é **0.8395** (forte). Nos pares que envolvem `yes_rate`, o |ρ| máximo é **0.3501** (moderada).

- `yes_rate × n_distinct_categories` — ρ = +0.3501 (moderada)
- `yes_rate × category_entropy` — ρ = +0.2102 (fraca)
- `yes_rate × mean_labels_per_item` — ρ = +0.1365 (desprezável)
- `n_distinct_categories × category_entropy` — ρ = +0.8395 (forte)
- `n_distinct_categories × mean_labels_per_item` — ρ = +0.4432 (moderada)
- `category_entropy × mean_labels_per_item` — ρ = +0.4570 (moderada)

Magnitudes presentes: desprezável, forte, fraca, moderada. Leitura por magnitude, não por significância — com n = 844, ρ = 0,08 sai "significativo" e não quer dizer nada.

### 1.2 Existe uma dimensão latente única?

`[MEDIDO]` — PC1 explica **57.1%** da variância das quatro features padronizadas (referência de nulidade com 4 features independentes: 25%).

Loadings do PC1: `yes_rate` +0.3205 · `n_distinct_categories` +0.6145 · `category_entropy` +0.5983 · `mean_labels_per_item` +0.4022

Sinais: todos iguais.

**Veredito pela régua pré-registada: DIMENSÃO LATENTE ÚNICA.**

`[MEDIDO]` — margem sobre o limiar de dimensão única: **+2.1 pontos percentuais** (57.1% contra 55%). `[LIMITAÇÃO]` — a passagem é **apertada**. O veredito muda de categoria com uma diferença de poucos pontos, e o limiar de 55% é convencional, não derivado dos dados. Reportar a percentagem, nunca só o rótulo.

### 1.3 A associação sobrevive ao controlo por `n_sexist_items`?

`[MEDIDO]` — os três números lado a lado, por par:

| par | ρ bruto | ρ parcial | ρ estratificado | Δ | magnitude (parcial) | leitura |
|---|---|---|---|---|---|---|
| yes_rate × n_distinct_categories | 0.3501 | 0.0454 | 0.1183 | -0.3047 | desprezável | OVER-CONTROL |
| yes_rate × category_entropy | 0.2102 | 0.0242 | 0.0393 | -0.1861 | desprezável | OVER-CONTROL |
| yes_rate × mean_labels_per_item | 0.1365 | -0.0339 | 0.0211 | -0.1705 | desprezável | OVER-CONTROL |
| n_distinct_categories × category_entropy | 0.8395 | 0.8450 | 0.8286 | 0.0054 | forte | controlo válido |
| n_distinct_categories × mean_labels_per_item | 0.4432 | 0.4076 | 0.4210 | -0.0355 | moderada | controlo válido |
| category_entropy × mean_labels_per_item | 0.4570 | 0.4272 | 0.4438 | -0.0298 | moderada | controlo válido |

Nos 3 pares com controlo válido (entre features de categoria), **0** perdem mais de 50% da magnitude ao controlar por `n_sexist_items`.

`[LIMITAÇÃO]` — **over-control nos pares com `yes_rate`.** ρ(`n_sexist_items`, `yes_rate`) = +0.9254, e `n_sexist_items` == `n_yes` em 834 dos 887 anotadores. A variável de controlo e `yes_rate` são quase a mesma coisa: nesses pares a parcial tende a zero **por construção** e não pode ser lida como "a associação era artefacto". Para eles o controlo interpretável é a estratificação.

`[MEDIDO]` — estratificação: `n_sexist_items` na banda **12–18** (mediana ± 3), **n = 387**. As colunas `ρ estratificado` da tabela acima vêm daí.

`[MEDIDO]` — PCA sobre features residualizadas em `n_sexist_items`:

| variante | PC1 antes | PC1 depois | nulidade | veredito |
|---|---|---|---|---|
| 4 features (contaminada por over-control) | 57.1% | 53.3% | 25% | ESTRUTURA PARCIAL — descrever associações específicas, não dimensão única |
| **3 de categoria (a pergunta limpa)** | 71.3% | 70.7% | 33% | DIMENSÃO LATENTE ÚNICA |

### 1.4 Duas ressalvas que os números impõem

**(a) Duas das três features de categoria estão ligadas por definição.** `[MEDIDO]` ρ(`n_distinct_categories`, `category_entropy`) = +0.8395 — o par mais forte da matriz. `category_entropy` é calculada sobre a distribuição das categorias efetivamente usadas, logo é **limitada por cima** por `log(n_distinct_categories)`: quem usou 2 categorias não pode ter entropia normalizada acima de `log(2)/log(5)` ≈ 0,43. A associação é em parte matemática, não comportamental.

Consequência: o PC1 de 71.3% sobre as três features de categoria **sobrestima a evidência independente** — são mais próximas de duas medições e meia do que de três. `[LIMITAÇÃO]` a declarar sempre que esse número for citado.

**Mas o achado não colapsa com esta ressalva.** `[MEDIDO]` — descontando o par definicional, restam 2 associações **sem** ligação matemática entre si, e nenhuma delas é artefacto do denominador:

- `n_distinct_categories × mean_labels_per_item` — ρ bruto +0.4432 → parcial +0.4076 → estratificado +0.4210 (moderada)
- `category_entropy × mean_labels_per_item` — ρ bruto +0.4570 → parcial +0.4272 → estratificado +0.4438 (moderada)

Ou seja: o par inflacionado é **um** dos três, não todos. O núcleo da estrutura entre features de categoria sobrevive à ressalva definicional **e** ao controlo por `n_sexist_items`. Descontar o achado inteiro por causa de (a) seria sobre-corrigir.

**(b) A questão do `yes_rate` fica por resolver, e não por omissão.** `[MEDIDO]` em bruto, `yes_rate` associa-se às features de categoria até |ρ| = 0.3501 (moderada); no estrato, até |ρ| = 0.1183 (desprezável); na parcial, até |ρ| = 0.0454.

`[LIMITAÇÃO]` — **os dois controlos atenuam esta associação pela mesma razão, e nenhum a resolve.** A estratificação fixa `n_sexist_items` numa banda estreita, e como `yes_rate ≈ n_sexist_items / n_votes`, fixar o numerador deixa `yes_rate` a variar quase só através de `n_votes`. A estratificação é, também ela, um controlo parcial sobre a própria variável — mais suave que a parcial, não independente dela.

`[HIPÓTESE]` — não é possível, com estes dados, separar a associação de `yes_rate` com as features de categoria do efeito mecânico do denominador, porque `yes_rate` **é** o denominador normalizado. Distingui-los exigiria uma medida de propensão a marcar YES que não entrasse no denominador das features de categoria — e não existe nestes dados.

---

## 2. O que pode e não pode ser escrito na tese

### Sustentado pelos números

- `[MEDIDO]` Entre as três features de categoria, as correlações de Spearman ao nível individual chegam a |ρ| = 0.8395 (forte), n = 844.
- `[MEDIDO]` Nos pares que envolvem `yes_rate`, |ρ| máximo = 0.3501 (moderada).
- `[MEDIDO]` O PC1 das quatro features explica 57.1% da variância com todas as loadings do mesmo sinal — passa o limiar pré-registado de 55%, por **+2.1 pontos**. A formulação "dimensão latente única" é sustentável ao nível individual, **desde que dita com a percentagem, com a margem, e com a composição do eixo da §1.4** — a força vem das features de categoria, não de `yes_rate`.
- `[MEDIDO]` As loadings do PC1 têm todas o mesmo sinal — a direção "expansivo ↔ parcimonioso" descreve corretamente o eixo, independentemente da força dele.
- `[MEDIDO]` Nenhuma das 3 associações entre features de categoria colapsa ao controlar por `n_sexist_items`: não são artefacto de tamanho de amostra.
- `[MEDIDO]` Depois de residualizar as três features de categoria em `n_sexist_items`, o PC1 passa de 71.3% para 70.7% (nulidade 33%). É a versão da estrutura livre do artefacto do denominador.
- `[MEDIDO]` A estrutura entre as features de categoria é forte (ρ até +0.8395) **mas parcialmente definicional** — a declarar sempre que o PC1 dessas três for citado.
- `[MEDIDO]` **Descontando o par definicional, o achado mantém-se:** as 2 associações restantes entre features de categoria não têm ligação matemática entre si e sobrevivem ao controlo — `n_distinct_categories × mean_labels_per_item` (bruto +0.443 → parcial +0.408) e `category_entropy × mean_labels_per_item` (bruto +0.457 → parcial +0.427). A ressalva definicional atinge um par, não a estrutura toda.

### NÃO sustentado — não escrever

- ❌ Apresentar as três features de categoria como três medições independentes do mesmo comportamento. ρ(`n_distinct_categories`, `category_entropy`) = +0.8395 e a ligação é em parte **definicional** — a entropia normalizada é limitada por `log(n_distinct_categories)`. O PC1 das três sobrestima a evidência independente.
- ❌ Afirmar que `yes_rate` faz parte da mesma dimensão latente que as features de categoria. Em bruto |ρ| ≤ 0.3501; no estrato cai para |ρ| ≤ 0.1183 (desprezável). A loading de `yes_rate` no PC1 é a mais fraca das quatro (+0.3205).
- ❌ Afirmar o contrário — que `yes_rate` **não** faz parte da dimensão. Os dois controlos disponíveis atenuam a associação pela mesma razão (ambos controlam sobre um quase-duplicado de `yes_rate`) e nenhum a resolve. A questão fica em aberto por limitação dos dados, e é assim que deve ser escrita.
- ❌ Citar a parcial dos pares com `yes_rate` como prova de artefacto. ρ(`n_sexist_items`, `yes_rate`) = +0.9254 — é over-control.
- ❌ Reutilizar a formulação original de `behavioral_diagnosis.md` §6.3, que inferiu a dimensão latente **da comparação entre centróides do KMeans**. Esse argumento é circular e fica substituído por esta análise, qualquer que seja a direção do resultado.
- ❌ Qualquer leitura destes eixos como qualidade, fiabilidade ou competência de anotação. Nenhuma feature aqui mede acerto.

---

## 3. Correlações par-a-par

Spearman é a métrica **primária**: `n_distinct_categories` tem apenas 4 valores distintos (2–5), quase ordinal, e as distribuições são enviesadas. Pearson entra só como verificação de consistência.

**Régua de magnitude (pré-registada):** |ρ| < 0,20 desprezável · 0,20–0,30 fraca · 0,30–0,50 moderada · > 0,50 forte.

### 3.1 Spearman (primária)

| par | n | ρ | p | magnitude |
|---|---|---|---|---|
| yes_rate × n_distinct_categories | 844 | 0.3501 | 0.0000 | moderada |
| yes_rate × category_entropy | 844 | 0.2102 | 0.0000 | fraca |
| yes_rate × mean_labels_per_item | 844 | 0.1365 | 0.0001 | desprezável |
| n_distinct_categories × category_entropy | 844 | 0.8395 | 0.0000 | forte |
| n_distinct_categories × mean_labels_per_item | 844 | 0.4432 | 0.0000 | moderada |
| category_entropy × mean_labels_per_item | 844 | 0.4570 | 0.0000 | moderada |

Matriz completa:

| ρ | yes_rate | n_distinct_categories | category_entropy | mean_labels_per_item |
|---|---|---|---|---|
| yes_rate | 1.0000 | 0.3501 | 0.2102 | 0.1365 |
| n_distinct_categories | 0.3501 | 1.0000 | 0.8395 | 0.4432 |
| category_entropy | 0.2102 | 0.8395 | 1.0000 | 0.4570 |
| mean_labels_per_item | 0.1365 | 0.4432 | 0.4570 | 1.0000 |

### 3.2 Pearson (verificação de consistência)

| par | n | ρ | p | magnitude |
|---|---|---|---|---|
| yes_rate × n_distinct_categories | 844 | 0.3644 | 0.0000 | moderada |
| yes_rate × category_entropy | 844 | 0.2250 | 0.0000 | fraca |
| yes_rate × mean_labels_per_item | 844 | 0.1331 | 0.0001 | desprezável |
| n_distinct_categories × category_entropy | 844 | 0.8703 | 0.0000 | forte |
| n_distinct_categories × mean_labels_per_item | 844 | 0.3793 | 0.0000 | moderada |
| category_entropy × mean_labels_per_item | 844 | 0.4033 | 0.0000 | moderada |

`[MEDIDO]` — nenhuma divergência |ρ − r| acima de 0.1. Spearman e Pearson concordam; não há não-linearidade a assinalar.

---

## 4. PCA bruta

| componente | variância explicada (%) | acumulada (%) |
|---|---|---|
| PC1 | 57.14 | 57.14 |
| PC2 | 22.24 | 79.38 |
| PC3 | 17.70 | 97.08 |
| PC4 | 2.92 | 100.00 |

| feature | loading no PC1 |
|---|---|
| yes_rate | 0.3205 |
| n_distinct_categories | 0.6145 |
| category_entropy | 0.5983 |
| mean_labels_per_item | 0.4022 |

**Régua pré-registada:** PC1 > 55% **e** loadings todas do mesmo sinal → dimensão latente única · 35–55% → estrutura parcial · < 35% → sem dimensão latente · qualquer loading de sinal oposto → a interpretação "expansivo ↔ parcimonioso" está errada.

---

## 5. Controlo do artefacto mecânico

### 5.1 O denominador partilhado

`n_sexist_items` = número de itens que o anotador marcou com ≥ 1 categoria real de sexismo. **Já existia** como coluna em `behavioral_profiles.csv` (escrita por `behavioral_profile.py::_profile_one`); não foi derivada aqui.

`[MEDIDO]` — ρ de Spearman entre o denominador e cada feature:

| feature | n | ρ com n_sexist_items | p | magnitude |
|---|---|---|---|---|
| yes_rate | 844 | 0.9254 | 0.0000 | forte |
| n_distinct_categories | 844 | 0.3771 | 0.0000 | moderada |
| category_entropy | 844 | 0.2244 | 0.0000 | fraca |
| mean_labels_per_item | 844 | 0.1516 | 0.0000 | desprezável |

Hipótese sob teste: quem tem mais itens sexistas tem mecanicamente mais oportunidade de acumular categorias distintas, e a entropia estimada de poucas amostras é sistematicamente subestimada.

### 5.2 Correlações parciais

Método: `scipy` e `sklearn` não expõem correlação parcial diretamente. ρ(X, Y | Z) é obtida regredindo X e Y em Z por OLS (com intercepto, via `numpy.linalg.lstsq`) e correlacionando os resíduos por Spearman — a definição clássica de parcial.

| par | ρ bruto | ρ parcial | Δ | magnitude (parcial) | leitura |
|---|---|---|---|---|---|
| yes_rate × n_distinct_categories | 0.3501 | 0.0454 | -0.3047 | desprezável | OVER-CONTROL |
| yes_rate × category_entropy | 0.2102 | 0.0242 | -0.1861 | desprezável | OVER-CONTROL |
| yes_rate × mean_labels_per_item | 0.1365 | -0.0339 | -0.1705 | desprezável | OVER-CONTROL |
| n_distinct_categories × category_entropy | 0.8395 | 0.8450 | 0.0054 | forte | controlo válido |
| n_distinct_categories × mean_labels_per_item | 0.4432 | 0.4076 | -0.0355 | moderada | controlo válido |
| category_entropy × mean_labels_per_item | 0.4570 | 0.4272 | -0.0298 | moderada | controlo válido |

### 5.3 Estratificação

Banda `n_sexist_items` ∈ [12, 18] (mediana ± 3), n = 387 de 844. Verificação independente da parcial: se a associação persistir dentro de um estrato estreito do denominador, não é artefacto de tamanho de amostra.

| par | ρ bruto | ρ parcial | ρ estratificado | Δ | magnitude (parcial) | leitura |
|---|---|---|---|---|---|---|
| yes_rate × n_distinct_categories | 0.3501 | 0.0454 | 0.1183 | -0.3047 | desprezável | OVER-CONTROL |
| yes_rate × category_entropy | 0.2102 | 0.0242 | 0.0393 | -0.1861 | desprezável | OVER-CONTROL |
| yes_rate × mean_labels_per_item | 0.1365 | -0.0339 | 0.0211 | -0.1705 | desprezável | OVER-CONTROL |
| n_distinct_categories × category_entropy | 0.8395 | 0.8450 | 0.8286 | 0.0054 | forte | controlo válido |
| n_distinct_categories × mean_labels_per_item | 0.4432 | 0.4076 | 0.4210 | -0.0355 | moderada | controlo válido |
| category_entropy × mean_labels_per_item | 0.4570 | 0.4272 | 0.4438 | -0.0298 | moderada | controlo válido |

### 5.4 PCA residualizada

**4 features** — n = 844, referência de nulidade 25%

| componente | variância explicada (%) | acumulada (%) |
|---|---|---|
| PC1 | 53.25 | 53.25 |
| PC2 | 24.78 | 78.04 |
| PC3 | 18.71 | 96.74 |
| PC4 | 3.26 | 100.00 |

| feature | loading no PC1 |
|---|---|
| yes_rate | 0.0931 |
| n_distinct_categories | 0.6340 |
| category_entropy | 0.6378 |
| mean_labels_per_item | 0.4272 |

Veredito: **ESTRUTURA PARCIAL — descrever associações específicas, não dimensão única**

**3 features de categoria** — n = 844, referência de nulidade 33%

| componente | variância explicada (%) | acumulada (%) |
|---|---|---|
| PC1 | 70.68 | 70.68 |
| PC2 | 24.96 | 95.63 |
| PC3 | 4.37 | 100.00 |

| feature | loading no PC1 |
|---|---|
| n_distinct_categories | 0.6361 |
| category_entropy | 0.6417 |
| mean_labels_per_item | 0.4284 |

Veredito: **DIMENSÃO LATENTE ÚNICA**

---

## 6. Limites

1. **Análise descritiva sobre um artefacto já produzido.** Não treina nada, não reamostra, não valida contra critério externo. Descreve a estrutura de covariância das features, não a sua validade.
2. **A parcial nos pares com `yes_rate` é over-control** e está marcada como tal. Não citar essas três linhas como evidência de artefacto.
3. **A estratificação reduz o n** e com ele o poder: uma associação fraca pode desaparecer no estrato por falta de amostra, não por ser artefacto. Comparar magnitudes, não presença/ausência de significância.
4. **43 anotadores estão fora de toda esta análise** (NaN nas features de categoria, por terem menos de 5 itens sexistas). n = 844 de 887.
5. **Nada aqui julga a legitimidade de nenhum perfil.** Continua a valer o limite da Parte 2: estas features não medem acerto e não justificam down-weighting.
