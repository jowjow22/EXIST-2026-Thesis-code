# Notas para a tese — módulo de confiabilidade de anotadores (Task 2, LeWiDi)

**O que este documento é:** registro dos fatos produzidos pelo módulo `reliability/`
(sessões de 2026-08-26/27), no formato em que podem ir para a tese. Cada número tem
fonte. O que é hipótese está marcado como hipótese.

**Convenção** (a mesma de `EXIST_VIDEOS/improvements/13_NOTAS_PARA_TESE.md`):
- `[MEDIDO]` — número observado, com o artefato que o produziu.
- `[HIPÓTESE]` — interpretação plausível, ainda não testada. **Não escrever como fato.**
- `[LIMITAÇÃO]` — restrição a declarar explicitamente no texto.

**Enquadramento narrativo:** este é um **resultado negativo caracterizado**, não uma
funcionalidade entregue. Duas famílias de estimador foram construídas, validadas em
sintético e reprovadas — cada uma por uma razão diferente e mensurável. A tese ganha com
isto se o apresentar como tal: o valor está no diagnóstico, não num θ utilizável.

---

## 1. A tese central (formulação pronta para o texto)

> Qualquer estimador de confiabilidade que infira a verdade de referência a partir dos
> próprios votos é capturável por um bloco de anotadores correlacionados. A vulnerabilidade
> existe nas duas pontas do espectro de densidade do bloco, e a estratégia que protege
> contra uma delas agrava a outra.

**Dados de suporte** `[MEDIDO]`:

| estimador | unidade de estimação | bloco que o captura | evidência |
|---|---|---|---|
| `NutmegEM` (subpop-por-item) | célula (item, subpopulação) | **pequeno** (2–3 anotadores com célula própria) | θ̂ = 0,733 a 3% de spammers enviesados (uma seed 0,981), contra teto pré-registado de 0,60 |
| per-anotador com desconto de dificuldade | anotador (~27 votos) | **grande** (≥ 25% do painel) | gap fiáveis−bloco cai de +0,242 (3%) para +0,044 (35%); **negativo** a 35% no braço do spammer |

O bloco pequeno fabrica a **verdade da sua própria célula**; o bloco grande fabrica a
**dificuldade empírica do item**. Mecanismos diferentes, mesma raiz.

---

## 2. Por que o `NutmegEM` não se sustenta no regime do EXIST

**Formulação:**

> No regime de densidade do EXIST (3 984 itens, 887 anotadores, ~27 votos por anotador,
> ~6 votos por item), a estimação de uma verdade por célula (item, subpopulação) dispõe de
> 1–2 votos por célula. A fragmentação anula o ganho teórico do modelo de subpopulação.

**Dados de suporte** `[MEDIDO — reliability/outputs/recovery_grid*.csv, stage1_report.md]`:

- O fenómeno teórico **reproduz-se** no sintético: com subpopulação, o Pearson r contra o
  θ real mantém-se estável ao longo da divisividade; sem ela degrada monotonicamente
  (0,883 → 0,287). Cruzamento ≈ **0,15** de divisividade.
- `[LIMITAÇÃO]` Abaixo do cruzamento o modelo de verdade única **ganha**. O ganho da
  subpopulação não é incondicional.
- Na densidade real do EXIST (26,9 votos/anotador em vez de 80), r cai de **0,761 para
  0,599**. É o tecto honesto da Task A, independentemente do estimador.
- O teste de preservação de minoria **passou** nas duas densidades (θ̂ da minoria
  0,921–0,960, acima da maioria fiável) — o modelo não rompe o LeWiDi por desenho; o que
  falha é a densidade de dados.

**Divisividade empírica real** `[MEDIDO — reliability/outputs/divisiveness_report.md]`,
com nulo condicionado *within-item*:

| definição de subpopulação | itens divisivos (subconjunto fiável) | z | ratio entre/total | z | α intra | z |
|---|---|---|---|---|---|---|
| `demo_base_K4` | 0,304 | −0,08 | 0,362 | **+2,16** | 0,288 | **+2,03** |
| `demo_extended_K4` | 0,329 | +0,30 | 0,499 | **+2,09** | 0,288 | +1,37 |
| `demo_base_K8` | 0,424 | −0,37 | 0,597 | +0,77 | 0,284 | +0,49 |
| `demo_extended_K8` | 0,467 | +1,48 | 0,802 | −0,85 | 0,285 | −0,02 |
| `behavioral_K4` | 0,411 | +18,3 | 0,460 | +31,8 | 0,336 | +12,8 |

α global = **0,2791**.

Leitura: as definições **demográficas** — as únicas não-circulares — mostram estrutura no
máximo **marginal** (z ≈ 2,1 em 2 de 3 indicadores, só em K=4, nula em K=8). As
**comportamentais** mostram z enormes mas são **circulares**: os clusters saem dos mesmos
votos que se vão testar. Não são evidência independente.

`[LIMITAÇÃO]` `demo_extended_K8` é **inconclusivo por esparsidade**: só 17,5% dos itens têm
duas subpopulações com ≥ 2 votos cada.

**Correção metodológica a declarar** `[MEDIDO]`: o nulo por permutação de anotadores
(Null A) é **confundido**. Permutar anotadores altera a composição de subpopulações dentro
de cada item: itens reais têm 2,77 subpopulações distintas em média contra 3,13 nos
permutados (`demo_base_K4`). Menos células ⇒ menos fragmentação ⇒ menos divisividade
espúria, logo o observado sai *abaixo* do nulo por **geometria**, não por ausência de
estrutura. Todos os vereditos usam o nulo **condicionado within-item**, que preserva
partição e composição e permuta apenas os rótulos dentro do item.

---

## 3. A reformulação por-anotador e por que também falha

**Motivação:** eliminar a célula como unidade de estimação, agregando os ~27 votos de cada
anotador — 1–2 ordens de grandeza mais dados por parâmetro. Para não colapsar em MACE puro
(que pune minorias consistentes e rompe o LeWiDi), a defesa era o **desconto por
dificuldade do item**: discordar num item de pool dividido quase não penaliza; discordar
num item de consenso penaliza.

    difficulty[i] = H(p_i)/log 2,   p_i = fração de YES no pool do item i
    competência[j] = Σ_i (1 − difficulty[i])·[voto_j(i) == maioria(i)] / Σ_i (1 − difficulty[i])

Determinístico, não-iterativo: a dificuldade sai dos votos **crus**, sem ponderar por
competência.

**Veredito** `[MEDIDO — reliability/outputs/perann_gate_report.md,
perann_synthetic_gate.json]`: **FAIL nas duas variantes, nos três testes.** O componente
não foi corrido nos dados reais.

### 3.1 O que o desconto faz certo

`[MEDIDO]` No teste de preservação de minoria com dois braços **matched** (mesma fração
0,25, mesmo θ real 0,95, mesmo número de itens de discordância; só muda o terreno):

| config | hard: minoria | hard: fiáveis | easy: minoria | easy: fiáveis |
|---|---|---|---|---|
| `mace_naive` | 0,940 | 0,845 | 0,728 | 0,851 |
| `discount_raw` | 0,983 | 0,947 | **0,896** | 0,945 |
| `discount_loo` | 0,940 | 0,842 | **0,702** | 0,820 |

O braço `hard` (minoria consistente em terreno genuinamente ambíguo) **passa com folga** em
ambas as variantes: a minoria fica **acima** da maioria fiável, como o θ real dela prevê.
A proteção LeWiDi funciona.

### 3.2 O que o desconto faz errado

`[MEDIDO]` O braço `easy` (minoria consistente em terreno de **consenso** — é ruído, deve
ser punida) **falha** nas duas variantes: 0,896 e 0,702, contra um teto pré-registado de
0,60. O estimador faz metade da distinção que prometia.

`[MEDIDO]` Spammer de viés constante (sempre YES, 20% do painel):

| config | enviesados | fiáveis | gap |
|---|---|---|---|
| `mace_naive` | 0,774 | 0,855 | +0,081 |
| `discount_raw` | 0,906 | 0,944 | **+0,037** |
| `discount_loo` | 0,733 | 0,823 | +0,089 |

**A hipótese de desenho está refutada.** Esperava-se que os erros do spammer caíssem em
itens fáceis e o desconto não o protegesse. O que se mede é o contrário: na forma literal
da especificação o desconto dá ao spammer um gap **menor** que o da linha de base
MACE-ingénua.

### 3.3 O mecanismo

> A dificuldade do item é calculada a partir dos votos. Um bloco dissidente
> suficientemente grande **fabrica** a dificuldade dos itens em que discorda, e o desconto
> passa a protegê-lo — inclusive em terreno de consenso. O desconto quebra a circularidade
> *competência → dificuldade*; **não** quebra a circularidade *bloco → dificuldade*.

`[MEDIDO]` Sweep da densidade do bloco (gap = competência dos fiáveis − competência do
bloco; maior é melhor):

| fração do bloco | easy: `mace` | easy: `raw` | easy: `loo` | bias: `mace` | bias: `raw` | bias: `loo` |
|---|---|---|---|---|---|---|
| 0,03 | +0,242 | +0,150 | +0,302 | +0,162 | +0,094 | +0,209 |
| 0,15 | +0,176 | +0,096 | +0,210 | +0,103 | +0,050 | +0,118 |
| 0,25 | +0,124 | +0,049 | +0,118 | +0,057 | +0,026 | +0,065 |
| 0,35 | +0,044 | +0,014 | +0,038 | **−0,014** | **−0,018** | **−0,043** |

Decaimento **monotónico** nas três configurações. A 35% de spammers de viés constante o gap
fica **negativo** em todas: o bloco recebe competência **maior** que os anotadores fiáveis,
porque a essa densidade o bloco *é* a maioria em boa parte dos itens.

### 3.4 Recuperação de competência

`[MEDIDO]` Pearson r contra o θ real, média de 3 seeds:

| densidade | `hard_frac` | votos/anotador | `mace_naive` | `discount_raw` | `discount_loo` |
|---|---|---|---|---|---|
| confortável | 0,00 | 80 | **0,893** | 0,856 | 0,865 |
| confortável | 0,40 | 80 | 0,712 | 0,730 | **0,736** |
| confortável | 0,60 | 80 | 0,613 | 0,696 | **0,699** |
| exist | 0,40 | 27 | 0,503 | 0,500 | **0,510** |
| exist | 0,60 | 27 | 0,354 | **0,389** | **0,389** |

O desconto **só compensa quando a ambiguidade genuína domina** (+0,083 sobre a linha de
base em `hard_frac`=0,6; empate em 0,4; perde em ≤ 0,2, porque descarta informação de itens
que não precisavam de proteção). O limiar pré-registado exigia vantagem em todo o regime
`hard_frac` ≥ 0,4; a vantagem mínima medida é **−0,003**.

`[MEDIDO]` A densidade do EXIST corta r em ~0,2 em todos os regimes — **o mesmo tecto** já
medido para o `NutmegEM` (§2). O constrangimento dominante é a densidade, não a família de
estimador.

---

## 4. Duas causas de FAIL, a separar no texto

`[MEDIDO]` **Escala (corrigível, e é uma lição de pré-registo).** Os limiares absolutos
(teto 0,60, piso 0,70) foram herdados da fase do `NutmegEM`, onde θ é o parâmetro de spam
de um modelo generativo e tem chão 0. A competência por-anotador é uma **taxa de
concordância**, com chão de acaso ≈ 0,5. Um anotador com θ real 0,25 tem concordância
esperada 0,25 + 0,75 × 0,5 = **0,625** — acima do teto **por construção**, sem que nada
esteja errado. Limiares absolutos não transferem entre escalas; **gaps transferem**.

`[MEDIDO]` **Fundo (não corrigível por limiar).** Olhando só para gaps, `discount_raw`
perde para `mace_naive` em **10/10** linhas do sweep. O desconto na forma literal piora a
deteção de ruído. Nenhuma recalibração muda isto.

`[LIMITAÇÃO]` Os limiares **não foram alterados depois de vistos os resultados**. A
distinção acima é diagnóstico, não recalibração retroativa.

---

## 5. Variante *leave-one-out* (não estava na especificação)

Na forma literal, o voto do próprio anotador entra no cálculo da dificuldade do item que o
vai julgar: um dissidente solitário num pool de 6 votos unânimes dá p=1/6 ⇒ dificuldade
0,65 ⇒ peso 0,35 em vez de 1,0. **Auto-inflação.** A variante `leave_one_out` calcula
dificuldade e maioria vistas por *j* excluindo o voto de *j*.

`[MEDIDO]` Domina a forma literal: r superior em **7/8** linhas da grade de recuperação;
gap superior em **9/10** linhas do sweep; e continua a passar o braço `hard` da preservação
de minoria com folga. O receio de que o LOO destruísse a proteção da minoria legítima
**não se confirmou** nesta parametrização.

`[LIMITAÇÃO]` Contra a linha de base MACE-ingénua a vantagem é modesta: gap superior em
**7/10** linhas, e só em blocos ≤ 0,15. Nenhuma das variantes passa o portão.

---

## 6. Limitações a declarar explicitamente

- `[LIMITAÇÃO]` **A ambiguidade genuína não é observável nos dados reais.** No sintético
  sabe-se por construção que item é ambíguo; no EXIST só existe a dificuldade **empírica**,
  que é precisamente a quantidade capturável por um bloco correlacionado. O desenho depende
  de uma quantidade que só é fiável onde não é necessária.
- `[LIMITAÇÃO]` A discriminação colapsa para blocos ≥ 25% do painel. Regra operacional que
  daí sai: **nunca aceitar definições de subpopulação com clusters pequenos** (guarda
  sugerida: cada cluster ≥ 10% dos anotadores) **nem derivá-los do padrão de voto**. O
  primeiro clustering comportamental tentado produziu `[883, 2, 1, 1]` (K=4) e
  `[873, 2, 2, 2, 2, 2, 2, 3]` (K=8) — exatamente o regime em que o `NutmegEM` atribui θ
  alto a ruído.
- `[LIMITAÇÃO]` Os testes 2.2 e 2.3 correram só na densidade confortável (80
  votos/anotador); a densidade do EXIST está coberta em 2.1. Como o portão já falha na
  densidade mais fácil, estender os outros dois não muda o veredito.
- `[LIMITAÇÃO]` Votos em itens com pool empatado (ou pool LOO vazio/par) são **excluídos**
  do numerador e do denominador nas três configurações, para que a linha de base sem
  desconto seja comparável à variante com desconto (onde o empate tem peso 0 de qualquer
  forma).
- `[LIMITAÇÃO]` Task A (binária) apenas. Task B (3 classes) e Task C (multilabel) não foram
  implementadas. **Um resultado fraco na Task A não é preditivo da utilidade em B/C**, dado
  o gradiente de sinal demográfico já medido no modelo de memes: as condições B1 (fusão de
  biosinais sem embeddings per-viewer) mostraram deltas positivos consistentes justamente
  nas métricas finas.
- `[LIMITAÇÃO]` As definições de subpopulação comportamentais são **circulares** por
  construção e não podem sustentar afirmação de estrutura. Ficam no texto como
  contra-exemplo metodológico, não como resultado.

---

## 7. Disciplina metodológica (vale a pena descrever no capítulo de método)

`[MEDIDO]` Três práticas que produziram achados que de outro modo teriam passado:

1. **Limiares de PASS/FAIL pré-registados em código antes de correr.** Sem isso, os valores
   de §3.2 teriam sido lidos como "razoáveis".
2. **Nulos condicionados em vez de permutação ingénua.** O Null A dava z **negativos** que
   pareciam ausência de estrutura e eram geometria de composição (§2).
3. **Regra "nunca correr nos dados reais um componente que não passou no teste sintético".**
   É a razão de `run_perann_exist.py` não existir.

`[MEDIDO]` Determinismo verificado: duas execuções do estimador sobre a mesma entrada dão
max|Δ| = **0,000e+00**. Seeds `[42, 7, 2025]` em todos os testes.

---

## 8. Artefatos

| ficheiro | conteúdo |
|---|---|
| `reliability/nutmeg_em.py` | estimador EM subpop-por-item (baseline / resultado negativo). **Não alterado** desde a validação. |
| `reliability/perann_reliability.py` | estimador por-anotador com desconto de dificuldade |
| `reliability/synthetic.py` | `generate_synthetic` (subpop) e `generate_difficulty_synthetic` (dificuldade controlada) |
| `reliability/validate.py` | harness sintético, limiares pré-registados, CLI |
| `reliability/divisiveness.py` | divisividade empírica, decomposição entre/dentro, α por subgrupo, nulos A e B |
| `reliability/exist_data.py` | carregador da Task A do EXIST |
| `outputs/stage1_report.md` · `stage1b_report.md` | validação do `NutmegEM` e braço do spammer enviesado |
| `outputs/divisiveness_report.md` | divisividade real, tabela do §2 |
| `outputs/perann_gate_report.md` · `perann_gate_resumo.md` | portão do estimador por-anotador, §3–§5 |
| `outputs/summary_taskA.md` | θ do `NutmegEM` nos dados reais, 7 definições de subpopulação |

Comandos: `python3 -m reliability.validate --all` · `--perann` ·
`python3 -m reliability.divisiveness` · `python3 -m reliability.run_exist`

---

*Escrito em 2026-08-27 a partir de `reliability/outputs/perann_synthetic_gate.json`,
`divisiveness_metrics.json`, `minority_preservation.json`, `recovery_grid*.csv` e dos
relatórios em `reliability/outputs/`. Números sem fonte explícita não existem neste
documento.*
