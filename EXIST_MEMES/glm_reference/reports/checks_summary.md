# F4 — Checagens (a)–(i) · doc 27 (distância ao soft label do GLM)

Gerado por `checks.py`. Split de treino: 3386 memes, 887 anotadores, 887 com ≥ 1 meme incluído. Método: `doc27_soft_distance`.

> **Escala (SPEC_27 §16A, P1):** num meme em que o GLM fica 2-1, todo anotador leva pelo menos 1/3 de desvio. Os `D` deste relatório **não são comparáveis** com os do doc 25.

## (a) GLM vs consenso dos pares

ρ de Spearman(r_glm, r_peer) = **0.5008** (n = 887; `D_peer` com d = |voto − fração de YES dos outros 5|).

## (b) O peso só mede o limiar?

- ρ(D_glm, taxa de YES do anotador em todos os seus memes de treino) = **-0.6760**
- ρ(D_glm, |taxa de YES do anotador − média de p_GLM|, ambas nos memes incluídos) = **0.7104**

## (c) r por gênero × idade (anotadores com ≥ 1 meme incluído; IC 95% bootstrap de anotadores)

| gênero | idade | n | r_glm | IC 95% | r_peer | IC 95% |
|---|---|---|---|---|---|---|
| F | 18-22 | 147 | 0.5209 | [0.4846, 0.5567] | 0.4690 | [0.4395, 0.4980] |
| F | 23-45 | 151 | 0.5764 | [0.5450, 0.6080] | 0.4906 | [0.4631, 0.5181] |
| F | 46+ | 148 | 0.5956 | [0.5586, 0.6300] | 0.4775 | [0.4480, 0.5061] |
| M | 18-22 | 148 | 0.4609 | [0.4254, 0.4963] | 0.4524 | [0.4229, 0.4828] |
| M | 23-45 | 147 | 0.5276 | [0.4912, 0.5633] | 0.4833 | [0.4526, 0.5132] |
| M | 46+ | 146 | 0.5706 | [0.5377, 0.6018] | 0.4468 | [0.4193, 0.4738] |

Anotadores em mais de um painel (demografia conferida como consistente): 19.

## (d) Portão — quanto os soft labels se movem

| escopo | share_moved (|Δp| ≥ 1/12) | mudanças de rótulo duro |
|---|---|---|
| total | 0.3461 | 554 |
| en | 0.3469 | 285 |
| es | 0.3453 | 269 |

Tipos de mudança: NO→YES: 36, TIE→NO: 109, TIE→YES: 409

## (e) GLM confiante (3-0) contrariando meme unânime

| escopo | unânimes com GLM 3-0 | contrariados | taxa |
|---|---|---|---|
| total | 608 | 56 | 0.0921 |
| en | 318 | 29 | 0.0912 |
| es | 290 | 27 | 0.0931 |

30 memes sorteados em `manual_review_unanimous.csv` para revisão manual.

## (f) Acordo de glm_vote (maioria das 3 paráfrases) com o rótulo duro humano (empates 3-3 fora)

| escopo | divisão | n | acordo |
|---|---|---|---|
| total | todas | 2863 | 0.7073 |
| total | 6-0 | 790 | 0.8278 |
| total | 5-1 | 1032 | 0.7083 |
| total | 4-2 | 1041 | 0.6148 |
| en | todas | 1452 | 0.7059 |
| en | 6-0 | 401 | 0.8354 |
| en | 5-1 | 519 | 0.7187 |
| en | 4-2 | 532 | 0.5959 |
| es | todas | 1411 | 0.7087 |
| es | 6-0 | 389 | 0.8201 |
| es | 5-1 | 513 | 0.6979 |
| es | 4-2 | 509 | 0.6346 |

## (g) Precisão por anotador

- n_included (todos os 887 anotadores): mín 7 · p25 14 · mediana 17 · p75 20 · máx 45 · média 17.53
- mass_included (Σ w por anotador): n 887 · p25 2.5854 · mediana 3.4279 · p75 4.3989
- fração com < 5 memes incluídos: **0.0000**
- largura mediana do IC 95% de D_glm (bootstrap dos memes de cada anotador, 887 anotadores): **0.4563**

**Parcelas do D_glm** (mesma fórmula, mesmo w; por anotador em `annotator_deviation_parcels.csv`):

- D_glm total: n 887 · p25 0.2797 · mediana 0.4383 · p75 0.6261
- D_glm só com memes GLM 3-0: n 887 · p25 0.1829 · mediana 0.4157 · p75 0.6724
- D_glm só com memes GLM 2-1: n 865 · p25 0.4082 · mediana 0.5000 · p75 0.5980 (piso de 1/3 por construção)

## (h) Fração da massa Σ w que vem de memes GLM 2-1

| escopo | massa 2-1 / massa total |
|---|---|
| total | 0.2879 |
| en | 0.2698 |
| es | 0.3064 |

## (i) Hesitação do GLM × divisão humana

Descritiva; não decide nada e não entra no G4. Contagens e (proporção na linha).

**Padrão do GLM × divisão dos votos humanos:**

| escopo | linha | n | 6-0 | 5-1 | 4-2 | 3-3 | 4-2 + 3-3 |
|---|---|---|---|---|---|---|---|
| total | 3-0 | 2401 | 608 (0.253) | 746 (0.311) | 710 (0.296) | 337 (0.140) | **0.436** |
| total | 2-1 | 980 | 182 (0.186) | 286 (0.292) | 331 (0.338) | 181 (0.185) | **0.522** |
| en | 3-0 | 1246 | 318 (0.255) | 385 (0.309) | 372 (0.299) | 171 (0.137) | **0.436** |
| en | 2-1 | 466 | 83 (0.178) | 134 (0.288) | 160 (0.343) | 89 (0.191) | **0.534** |
| es | 3-0 | 1155 | 290 (0.251) | 361 (0.313) | 338 (0.293) | 166 (0.144) | **0.436** |
| es | 2-1 | 514 | 99 (0.193) | 152 (0.296) | 171 (0.333) | 92 (0.179) | **0.512** |

**Memes GLM 2-1, pela paráfrase que discordou:**

| escopo | linha | n | 6-0 | 5-1 | 4-2 | 3-3 | 4-2 + 3-3 |
|---|---|---|---|---|---|---|---|
| total | P1 | 39 | 5 (0.128) | 11 (0.282) | 13 (0.333) | 10 (0.256) | **0.590** |
| total | P2 | 207 | 66 (0.319) | 56 (0.271) | 52 (0.251) | 33 (0.159) | **0.411** |
| total | P3 | 734 | 111 (0.151) | 219 (0.298) | 266 (0.362) | 138 (0.188) | **0.550** |
| en | P1 | 17 | 2 (0.118) | 6 (0.353) | 4 (0.235) | 5 (0.294) | **0.529** |
| en | P2 | 103 | 30 (0.291) | 32 (0.311) | 22 (0.214) | 19 (0.184) | **0.398** |
| en | P3 | 346 | 51 (0.147) | 96 (0.277) | 134 (0.387) | 65 (0.188) | **0.575** |
| es | P1 | 22 | 3 (0.136) | 5 (0.227) | 9 (0.409) | 5 (0.227) | **0.636** |
| es | P2 | 104 | 36 (0.346) | 24 (0.231) | 30 (0.288) | 14 (0.135) | **0.423** |
| es | P3 | 388 | 60 (0.155) | 123 (0.317) | 132 (0.340) | 73 (0.188) | **0.528** |

- total: 4-2 + 3-3 = 0.522 nos GLM 2-1 vs 0.436 nos GLM 3-0; diferença +0.086 (IC 95% [+0.049, +0.123]) → os memes em que o GLM hesita têm **mais** divisões humanas fortes.
- en: 4-2 + 3-3 = 0.534 nos GLM 2-1 vs 0.436 nos GLM 3-0; diferença +0.099 (IC 95% [+0.046, +0.152]) → os memes em que o GLM hesita têm **mais** divisões humanas fortes.
- es: 4-2 + 3-3 = 0.512 nos GLM 2-1 vs 0.436 nos GLM 3-0; diferença +0.075 (IC 95% [+0.023, +0.127]) → os memes em que o GLM hesita têm **mais** divisões humanas fortes.

Leitura pré-registrada: se os 2-1 **não** tiverem mais 4-2/3-3 que os 3-0, a [HIPÓTESE] "os memes em que o GLM hesita são os ambíguos" não se sustenta. "Mais" só quando o IC 95% da diferença (aproximação normal para duas proporções) exclui zero. Nada muda no cálculo.

