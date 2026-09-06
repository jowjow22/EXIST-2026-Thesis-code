# Portão sintético — estimador POR-ANOTADOR com desconto de dificuldade

Gerado por `python3 -m reliability.validate --perann`. Seeds `[42, 7, 2025]`. Determinismo: **OK** (max|Δ| = 0,000e+00 entre duas execuções).

O `NutmegEM` não foi tocado. Ficheiros novos: `reliability/perann_reliability.py`; `generate_difficulty_synthetic` em `synthetic.py` (função nova, `generate_synthetic` intacta); testes novos em `validate.py`.

## VEREDITO: FAIL nas duas variantes, nos três testes

Não é um bug de implementação. O portão apanhou duas coisas distintas — uma de escala (esperável, corrigível) e uma de fundo (o achado real). Ambas abaixo.

## As três configurações comparadas

| nome | `discount` | `leave_one_out` | o que é |
|---|---|---|---|
| `mace_naive` | False | False | linha de base: concordância-com-maioria simples (MACE ingénuo), mesmo caminho de código |
| `discount_raw` | True | False | **literal da especificação**: dificuldade do item calculada do pool CRU, incluindo o voto do próprio anotador |
| `discount_loo` | True | True | variante: a dificuldade e a maioria vistas por *j* excluem o voto de *j* |

A variante LOO não estava na especificação. Foi acrescentada — e reportada ao lado, não em substituição — porque na forma literal o dissidente **infla com o próprio voto** a dificuldade do item que o vai julgar, e assim reduz a própria penalização. Com 6 votos, 1 dissidente num item unânime dá p=1/6 → dificuldade 0,65 → peso 0,35 em vez de 1,0.

## 2.1 Recuperação de competência (Pearson *r* vs θ real, média de 3 seeds)

| densidade | `hard_frac` | votos/anotador | `mace_naive` | `discount_raw` | `discount_loo` |
|---|---|---|---|---|---|
| confortavel | 0.00 | 80.0 | 0.893 | 0.856 | 0.865 |
| confortavel | 0.20 | 80.0 | 0.818 | 0.810 | 0.819 |
| confortavel | 0.40 | 80.0 | 0.712 | 0.730 | 0.736 |
| confortavel | 0.60 | 80.0 | 0.613 | 0.696 | 0.699 |
| exist | 0.00 | 27.1 | 0.711 | 0.638 | 0.658 |
| exist | 0.20 | 27.1 | 0.629 | 0.587 | 0.604 |
| exist | 0.40 | 27.1 | 0.503 | 0.500 | 0.510 |
| exist | 0.60 | 27.1 | 0.354 | 0.389 | 0.389 |

- `discount_raw`: r(hard=0,4, confortável) = **0.730** ≥ 0,60 → ok; vantagem mínima sobre `mace_naive` em hard ≥ 0,4 = **-0.003** ≥ 0,05 → **FALHA**. Veredito: **FAIL**.
- `discount_loo`: r(hard=0,4, confortável) = **0.736** ≥ 0,60 → ok; vantagem mínima sobre `mace_naive` em hard ≥ 0,4 = **+0.007** ≥ 0,05 → **FALHA**. Veredito: **FAIL**.

O desconto **só compensa quando a ambiguidade genuína domina**. Em `hard_frac=0,6` ganha (+0,083 sobre o MACE ingénuo na densidade confortável); em `hard_frac=0,4` empata; em `hard_frac ≤ 0,2` perde, porque descarta informação de itens que não precisavam de proteção. O limiar pré-registado exigia vantagem em todo o regime hard ≥ 0,4 e isso não se verifica.

Na densidade real do EXIST (~27 votos/anotador) todo o *r* cai ~0,2 — mesmo tecto honesto já medido no Stage 1 para o `NutmegEM`. A densidade continua a ser o constrangimento dominante, seja qual for o estimador.

## 2.2 Preservação de minoria — dois braços MATCHED

Mesma fração de minoria (0,25), mesmo θ (0,95), mesmo nº de itens de discordância (30%). A **única** diferença é o terreno: braço `hard` = itens genuinamente ambíguos (deve ser **protegida**); braço `easy` = itens de consenso (é ruído, deve ser **punida**).

| config | hard: minoria | hard: fiáveis | easy: minoria | easy: fiáveis |
|---|---|---|---|---|
| `mace_naive` | 0.940 | 0.845 | 0.728 | 0.851 |
| `discount_raw` | 0.983 | 0.947 | 0.896 | 0.945 |
| `discount_loo` | 0.940 | 0.842 | 0.702 | 0.820 |

- `discount_raw`: **hard PASSA** (minoria 0.983 ≥ 0,70; gap -0.036 ≤ 0,10). **easy FALHA** (minoria 0.896 tinha de ficar ≤ 0,60; gap +0.049 tinha de ser ≥ 0,20).
- `discount_loo`: **hard PASSA** (minoria 0.940 ≥ 0,70; gap -0.098 ≤ 0,10). **easy FALHA** (minoria 0.702 tinha de ficar ≤ 0,60; gap +0.118 tinha de ser ≥ 0,20).

**Metade certa, metade errada.** O desconto protege a minoria em terreno ambíguo — que é o objetivo LeWiDi e a razão de existir do desenho. Mas não pune a minoria que discorda em terreno de consenso, que é a outra metade da distinção que ele prometia fazer.

## 2.3 Spammer de viés constante (sempre YES, 20% dos anotadores)

| config | enviesados | fiáveis | gap |
|---|---|---|---|
| `mace_naive` | 0.774 | 0.855 | +0.081 |
| `discount_raw` | 0.906 | 0.944 | +0.037 |
| `discount_loo` | 0.733 | 0.823 | +0.089 |

Ambas as variantes **FALHAM**. Pior: `discount_raw` tem um gap (+0,037) **menor** que o do MACE ingénuo (+0,081) — nesta densidade o desconto *protege* o spammer em vez de o apanhar. A hipótese de desenho («os erros do spammer caem em itens fáceis, logo o desconto não o protege») está **refutada**.

## O diagnóstico que explica os três FAILs

A dificuldade do item é calculada **dos votos**. Um bloco dissidente grande o suficiente **fabrica** a dificuldade dos itens em que discorda, e o desconto passa a protegê-lo — mesmo em terreno de consenso. O desconto quebra a circularidade *competência → dificuldade*, mas **não** quebra a circularidade *bloco → dificuldade*.

Sweep da densidade do bloco (gap = fiáveis − bloco; maior é melhor):

**minoria dissidente em terreno FÁCIL (deve ser punida)**

| fração do bloco | `mace_naive` | `discount_raw` | `discount_loo` |
|---|---|---|---|
| 0.03 | 0.614 (gap +0.242) | 0.797 (gap +0.150) | 0.532 (gap +0.302) |
| 0.08 | 0.621 (gap +0.233) | 0.803 (gap +0.142) | 0.549 (gap +0.276) |
| 0.15 | 0.680 (gap +0.176) | 0.847 (gap +0.096) | 0.610 (gap +0.210) |
| 0.25 | 0.728 (gap +0.124) | 0.896 (gap +0.049) | 0.702 (gap +0.118) |
| 0.35 | 0.788 (gap +0.044) | 0.923 (gap +0.014) | 0.760 (gap +0.038) |

**spammer de viés constante (deve ser punido)**

| fração do bloco | `mace_naive` | `discount_raw` | `discount_loo` |
|---|---|---|---|
| 0.03 | 0.697 (gap +0.162) | 0.852 (gap +0.094) | 0.624 (gap +0.209) |
| 0.08 | 0.713 (gap +0.143) | 0.862 (gap +0.083) | 0.644 (gap +0.184) |
| 0.15 | 0.758 (gap +0.103) | 0.897 (gap +0.050) | 0.714 (gap +0.118) |
| 0.25 | 0.788 (gap +0.057) | 0.913 (gap +0.026) | 0.745 (gap +0.065) |
| 0.35 | 0.833 (gap -0.014) | 0.944 (gap -0.018) | 0.820 (gap -0.043) |

A discriminação decai **monotonicamente** com o tamanho do bloco, nas três configurações. A 35% de spammers de viés constante o gap fica **negativo** em todas: o bloco recebe competência MAIOR que os anotadores fiáveis, porque a 35% de sempre-YES ele *é* a maioria em muitos itens.

Esta é a **mesma geometria de falha** do achado do Stage 1B (spammers enviesados com subpopulação própria recebiam θ alto), mas espelhada: lá a vulnerabilidade era em blocos PEQUENOS com célula própria; aqui é em blocos GRANDES que dominam o pool. As duas pontas do mesmo problema — qualquer estimador que infira a verdade dos próprios votos é capturável por um bloco correlacionado.

## O que é escala e o que é fundo

Duas causas distintas, e é importante não as juntar:

1. **Escala (corrigível).** Os limiares absolutos (teto 0,60, piso 0,70) foram herdados do Stage 1, onde θ é o parâmetro de spam de um modelo generativo e tem chão 0. Aqui a competência é uma **taxa de concordância**, com chão de acaso ≈ 0,5: um anotador que vota aleatoriamente acerta metade das vezes. Um spammer com θ real 0,25 tem concordância esperada ≈ 0,25 + 0,75×0,5 = 0,625 — acima do teto 0,60 **por construção**, sem que nada esteja errado. Os limiares absolutos não transferem entre escalas. O **gap** transfere.

2. **Fundo (não corrigível por limiar).** Mesmo olhando só para os gaps, `discount_raw` perde para o `mace_naive` em **todas as 10 linhas** do sweep. O desconto na forma literal da especificação piora a deteção de ruído. Nenhuma recalibração de limiar muda isto.

## A variante LOO domina a forma literal

`r` de recuperação superior em **7/8** linhas da 2.1 (na restante, `exist`/`hard_frac=0,6`, as duas diferem só para além da 3ª casa decimal — ambas 0,389); gap superior em **9/10** linhas do sweep, nos dois braços; e continua a passar o braço `hard` da 2.2 com folga: minoria 0,940 contra 0,842 dos fiáveis, ou seja gap **−0,098** (a minoria fica ACIMA da maioria fiável, como o θ real dela de 0,95 contra ~0,82 prevê), quando o limiar admitia até +0,10. O receio de que o LOO destruísse a proteção da minoria legítima **não se confirmou** nesta parametrização.

Contra o `mace_naive` a vantagem do `discount_loo` é mais modesta: gap superior em **7/10** linhas do sweep. Ganha nas frações baixas de bloco nos dois braços (≤ 0,15) e perde nas altas (terreno fácil: 0,25 e 0,35; spammer: 0,35), onde o bloco domina o pool e nenhuma variante discrimina.

## Limitações medidas

- **A dificuldade empírica não separa ambiguidade de bloco.** É a limitação central; o sweep quantifica-a. Para blocos ≥ 25% do painel o estimador deixa de discriminar.

- **Densidade do EXIST corta o *r* em ~0,2** em todos os regimes. Mesmo tecto do `NutmegEM`.

- **Ambiguidade genuína não é observável nos dados reais.** No sintético sabemos que item é `hard` por construção; no EXIST só temos a dificuldade empírica, que é precisamente a quantidade capturável por bloco.

- **2.2 e 2.3 correram só na densidade confortável** (80 votos/anotador); a densidade do EXIST está coberta na 2.1. Como o portão já falha na densidade mais fácil, estender 2.2/2.3 à densidade real não muda o veredito.
- **Empates excluídos.** Votos em itens com pool empatado (ou pool LOO vazio/par) são excluídos do numerador e do denominador nas três configurações, para que `discount=False` seja consistente com `discount=True` (onde o empate tem peso 0 de qualquer forma). Sem isto a linha de base não seria comparável.

## Estado

Portão **não passado**. Segundo a regra do projeto — *nunca corras nos dados reais um componente que não passou no teste sintético* — a PARTE 3 (`run_perann_exist.py`) **não foi escrita nem executada**. Decisão sobre como prosseguir é do autor.
