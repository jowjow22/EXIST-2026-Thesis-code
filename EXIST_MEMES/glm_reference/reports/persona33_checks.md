# Doc 33 — F4: checagens (a)–(j) com a Persona como referência

Gerado por `persona_ref.py checks`. Treino: 3386 memes, 887 anotadores, 887 com ≥ 1 meme incluído. Método: `doc33_persona_soft_distance` (fórmulas do doc 27, SPEC_27 §8).

> Escala (SPEC_27 §16A, P1): num meme em que a Persona fica 2-1, todo anotador leva pelo menos 1/3 de desvio.

## (a) Persona vs consenso dos pares

ρ de Spearman(r_persona, r_peer) = **0.7266** (n = 887).

## (b) O peso só mede o limiar?

- ρ(D_persona, taxa de YES do anotador em todos os seus memes de treino) = **-0.1011**
- ρ(D_persona, |taxa de YES do anotador − média de p_Persona|, nos memes incluídos) = **0.4630**

## (c) r por gênero × idade (IC 95% bootstrap de anotadores)

| gênero | idade | n | r_persona | IC 95% | r_peer | IC 95% |
|---|---|---|---|---|---|---|
| F | 18-22 | 147 | 0.5533 | [0.5210, 0.5875] | 0.4685 | [0.4389, 0.4976] |
| F | 23-45 | 151 | 0.5913 | [0.5598, 0.6237] | 0.4905 | [0.4628, 0.5181] |
| F | 46+ | 148 | 0.5511 | [0.5162, 0.5875] | 0.4773 | [0.4478, 0.5059] |
| M | 18-22 | 148 | 0.5475 | [0.5134, 0.5801] | 0.4516 | [0.4223, 0.4825] |
| M | 23-45 | 147 | 0.5597 | [0.5250, 0.5942] | 0.4835 | [0.4528, 0.5134] |
| M | 46+ | 146 | 0.5214 | [0.4887, 0.5529] | 0.4470 | [0.4199, 0.4739] |

Anotadores em mais de um painel (demografia conferida como consistente): 19.

## (d) Portão — quanto os soft labels se movem

| escopo | share_moved (|Δp| ≥ 1/12) | mudanças de rótulo duro | Δp médio |
|---|---|---|---|
| total | 0.2741 | 526 | +0.0054 |
| en | 0.3090 | 266 | +0.0007 |
| es | 0.2382 | 260 | +0.0102 |

Tipos de mudança: NO→YES: 4, TIE→NO: 243, TIE→YES: 275, YES→NO: 4

## (e) Persona confiante (3-0) contrariando meme unânime

| escopo | unânimes com Persona 3-0 | contrariados | taxa |
|---|---|---|---|
| total | 737 | 27 | 0.0366 |
| en | 377 | 18 | 0.0477 |
| es | 360 | 9 | 0.0250 |

27 memes sorteados para revisão manual (`persona_ref/manual_review_unanimous.csv`).

## (f) Acordo do voto da Persona com o rótulo duro humano (empates 3-3 fora)

| escopo | divisão | n | acordo |
|---|---|---|---|
| total | todas | 2868 | 0.8061 |
| total | 6-0 | 792 | 0.9331 |
| total | 5-1 | 1033 | 0.8132 |
| total | 4-2 | 1043 | 0.7028 |
| en | todas | 1455 | 0.8172 |
| en | 6-0 | 402 | 0.9179 |
| en | 5-1 | 520 | 0.8365 |
| en | 4-2 | 533 | 0.7223 |
| es | todas | 1413 | 0.7948 |
| es | 6-0 | 390 | 0.9487 |
| es | 5-1 | 513 | 0.7895 |
| es | 4-2 | 510 | 0.6824 |

## (g) Precisão por anotador

- n_included: mín 7 · p25 14 · mediana 18 · p75 20 · máx 45
- mass_included (Σ w por anotador): n 887 · p25 2.5854 · mediana 3.4279 · p75 4.3989
- fração com < 5 memes incluídos: **0.0000**
- largura mediana do IC 95% de D_persona (887 anotadores): **0.5313**
- D_persona total: n 887 · p25 0.2794 · mediana 0.4461 · p75 0.6026
- só memes Persona 3-0: n 887 · p25 0.2394 · mediana 0.4388 · p75 0.6146
- só memes Persona 2-1: n 829 · p25 0.3598 · mediana 0.5000 · p75 0.6379 (piso de 1/3 por construção)

## (h) Fração da massa Σ w que vem de memes Persona 2-1

| escopo | massa 2-1 / total |
|---|---|
| total | 0.1399 |
| en | 0.1113 |
| es | 0.1690 |

## (i) Hesitação da Persona × divisão humana (descritiva)

| escopo | linha | n | 6-0 | 5-1 | 4-2 | 3-3 | 4-2 + 3-3 |
|---|---|---|---|---|---|---|---|
| total | 3-0 | 2913 | 737 (0.253) | 901 (0.309) | 861 (0.296) | 414 (0.142) | **0.438** |
| total | 2-1 | 473 | 55 (0.116) | 132 (0.279) | 182 (0.385) | 104 (0.220) | **0.605** |
| en | 3-0 | 1496 | 377 (0.252) | 473 (0.316) | 442 (0.295) | 204 (0.136) | **0.432** |
| en | 2-1 | 219 | 25 (0.114) | 47 (0.215) | 91 (0.416) | 56 (0.256) | **0.671** |
| es | 3-0 | 1417 | 360 (0.254) | 428 (0.302) | 419 (0.296) | 210 (0.148) | **0.444** |
| es | 2-1 | 254 | 30 (0.118) | 85 (0.335) | 91 (0.358) | 48 (0.189) | **0.547** |

**Memes Persona 2-1, pelo prompt que discordou:**

| escopo | linha | n | 6-0 | 5-1 | 4-2 | 3-3 | 4-2 + 3-3 |
|---|---|---|---|---|---|---|---|
| total | S1 | 197 | 17 (0.086) | 55 (0.279) | 78 (0.396) | 47 (0.239) | **0.635** |
| total | S2 | 134 | 26 (0.194) | 37 (0.276) | 49 (0.366) | 22 (0.164) | **0.530** |
| total | S3 | 142 | 12 (0.085) | 40 (0.282) | 55 (0.387) | 35 (0.246) | **0.634** |
| en | S1 | 86 | 7 (0.081) | 17 (0.198) | 38 (0.442) | 24 (0.279) | **0.721** |
| en | S2 | 62 | 12 (0.194) | 13 (0.210) | 26 (0.419) | 11 (0.177) | **0.597** |
| en | S3 | 71 | 6 (0.085) | 17 (0.239) | 27 (0.380) | 21 (0.296) | **0.676** |
| es | S1 | 111 | 10 (0.090) | 38 (0.342) | 40 (0.360) | 23 (0.207) | **0.568** |
| es | S2 | 72 | 14 (0.194) | 24 (0.333) | 23 (0.319) | 11 (0.153) | **0.472** |
| es | S3 | 71 | 6 (0.085) | 23 (0.324) | 28 (0.394) | 14 (0.197) | **0.592** |

- total: 4-2 + 3-3 = 0.605 nos 2-1 vs 0.438 nos 3-0; diferença +0.167 (IC 95% [+0.119, +0.215]) → os memes em que a Persona hesita têm **mais** divisões humanas fortes.
- en: 4-2 + 3-3 = 0.671 nos 2-1 vs 0.432 nos 3-0; diferença +0.239 (IC 95% [+0.172, +0.307]) → os memes em que a Persona hesita têm **mais** divisões humanas fortes.
- es: 4-2 + 3-3 = 0.547 nos 2-1 vs 0.444 nos 3-0; diferença +0.103 (IC 95% [+0.037, +0.170]) → os memes em que a Persona hesita têm **mais** divisões humanas fortes.

Leitura pré-registrada (SPEC_27 §16B): "mais" só quando o IC 95% da diferença exclui zero. Aqui S1–S3 não são paráfrases: um 2-1 é discordância entre procedimentos (SPEC_33 §9).

## (j) Persona × GLM (doc 27), mesmos anotadores

- ρ(r_persona, r_glm) = **0.3969** (n = 887)
- (b) lado a lado — ρ(D, taxa de YES do anotador): Persona **-0.1011** · GLM **-0.6760**
- r médio: Persona 0.5542 · GLM 0.5421

## (k) Faixa nula por permutação (200 permutações das triplas S1–S3 entre memes, dentro de cada idioma; default_rng(2026))

Mantém a taxa de YES da Persona e destrói a informação por meme. Uma referência calibrada derruba a (b) sozinha; o que separa informação de taxa é a (a) acima da faixa (SPEC_33 §5).

| estatística | observado | faixa nula p2,5–p97,5 |
|---|---|---|
| rho_a | 0.7266 | [-0.0531, 0.1180] |
| rho_b | -0.1011 | [-0.1277, 0.0129] |
| rho_gap | 0.4630 | [0.0822, 0.2268] |
| share_moved | 0.2741 | [0.1403, 0.1879] |

**Leitura pré-registrada da (b) (SPEC_33 §5): parcial.**

**Contexto — a mesma faixa nula para o GLM (doc 27):** rho_a [0.1950, 0.2895] · rho_b [-0.7086, -0.6490] · rho_gap [0.6460, 0.7141] · share_moved [0.2932, 0.3435] (observados no doc 27: ρ(a) 0,5008, ρ(b) −0,676, ρ(gap) 0,710, share_moved 0,3461).

