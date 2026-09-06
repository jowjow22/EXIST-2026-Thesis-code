# Parte A — divisividade empírica real (EXIST Task A, memes)

Gerado por `python -m reliability.divisiveness`. CPU-only, diagnóstico puro (o `NutmegEM` não é usado nem alterado).

Dados: 3984 itens · 887 anotadores · 23904 votos · Krippendorff α global = **0.2791**

## 0. Dois nulos, e porque o segundo é o que conta

**Nulo A (permutação de anotadores)** — baralha quem pertence a que cluster, mantendo os tamanhos de cluster. Problema descoberto ao validar: também muda a COMPOSIÇÃO de cada item. Os itens reais têm em média **menos** subpops distintas do que os permutados (`demo_base_K4` 2.77 vs 3.13; `demo_extended_K4` 3.43 vs 3.32; `behavioral_K4` 2.59 vs 3.04; `demo_base_K8` 3.97 vs 4.22; `demo_extended_K8` 5.01 vs 4.37; `behavioral_K8` 3.29 vs 4.27), porque os pools de anotadores estão correlacionados com os itens (split ES/EN, clusters dominados por país). Menos células = menos fragmentação = menos divisividade espúria. **Os z negativos do Nulo A são geometria, não ausência de estrutura** — não devem ser lidos como 'menos estruturado que o acaso'.

**Nulo B (condicionado, *within-item*)** — mantém a partição e a composição de cada item exatamente como estão, e permuta apenas os **votos** entre os anotadores do mesmo item. Destrói só a associação anotador↔voto, que é a hipótese em teste. **Os vereditos usam o Nulo B.** O Nulo A fica na tabela como diagnóstico de composição.

### Nota de método original

Com 1–2 votos por célula (item, subpop), **qualquer** partição — inclusive aleatória — produz divisividade e rácio entre-grupos altos, só por fragmentação. Todas as quantidades são por isso comparadas com 50 partições aleatórias das mesmas dimensões (mesmos tamanhos de cluster, anotadores baralhados). **O número que interessa é o z-score, não o valor absoluto.** Um z ≈ 0 significa 'esta partição não explica a discordância melhor do que uma partição ao acaso'.

## A.1 / A.2 / A.3 — NULO B (condicionado) — os números que contam

| definição | divisivo (confiável) | z (Nulo B) | ratio entre/total | z (Nulo B) | α intra (pond.) | z (Nulo B) |
|---|---|---|---|---|---|---|
| `demo_base_K4` | 0.304 | -0.08 | 0.362 | +2.16 | 0.288 | +2.03 |
| `demo_extended_K4` | 0.329 | +0.30 | 0.499 | +2.09 | 0.288 | +1.37 |
| `behavioral_K4` | 0.411 | +18.33 | 0.460 | +31.82 | 0.336 | +12.80 |
| `demo_base_K8` | 0.424 | -0.37 | 0.597 | +0.77 | 0.284 | +0.49 |
| `demo_extended_K8` | 0.467 | +1.48 | 0.802 | -0.85 | 0.285 | -0.02 |
| `behavioral_K8` | 0.459 | +13.02 | 0.591 | +31.16 | 0.355 | +14.46 |

## Tabela completa (inclui o Nulo A, só diagnóstico de composição)

| definição | divisivo (todos) | nulo | z | divisivo (confiável) | nulo | z | itens confiáveis | ratio entre/total | nulo | z | α intra (pond.) | nulo | z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `demo_base_K4` | 0.332 | 0.381 | -4.16 | 0.304 | 0.346 | -2.83 | 2878 (72.2%) | 0.362 | 0.425 | -6.73 | 0.288 | 0.278 | +1.16 |
| `demo_extended_K4` | 0.348 | 0.400 | -3.91 | 0.329 | 0.367 | -2.78 | 3455 (86.7%) | 0.499 | 0.455 | +3.76 | 0.288 | 0.279 | +0.94 |
| `behavioral_K4` | 0.414 | 0.367 | +4.60 | 0.411 | 0.336 | +5.73 | 2750 (69.0%) | 0.460 | 0.410 | +4.57 | 0.336 | 0.278 | +5.92 |
| `demo_base_K8` | 0.507 | 0.544 | -3.41 | 0.424 | 0.426 | -0.14 | 2153 (54.0%) | 0.597 | 0.643 | -3.78 | 0.284 | 0.275 | +0.58 |
| `demo_extended_K8` | 0.645 | 0.563 | +6.29 | 0.467 | 0.439 | +1.31 | 698 (17.5%) | 0.802 | 0.670 | +8.14 | 0.285 | 0.275 | +0.66 |
| `behavioral_K8` | 0.526 | 0.548 | -1.67 | 0.459 | 0.426 | +2.01 | 2071 (52.0%) | 0.591 | 0.648 | -4.04 | 0.355 | 0.275 | +5.19 |

α global (referência para a coluna α intra): **0.2791**

**Critério de divisividade (A.1), explícito:** um item é divisivo se existe uma subpop com taxa-de-YES > 0.5 e outra com taxa < 0.5 nesse item. Empates exatos (0.5) não contam para nenhum lado. 'Confiável' = itens com ≥2 subpops com ≥2 votos cada — o caveat pedido: com 1 voto por célula a 'maioria da subpop' é o voto de uma pessoa.

**A.2, decomposição:** por item, `SS_total = Σ_j (v_ij − média_i)²`, `SS_between = Σ_k n_ik (média_ik − média_i)²`, `SS_within = SS_total − SS_between`, agregado sobre itens com ≥2 votos. Células de 1 voto têm `SS_within = 0` por construção e empurram o rácio para cima — daí o nulo.

## A.3 — α por subgrupo (detalhe)

- `demo_base_K4` — k0: 0.280 (n_itens≥2=2140), k1: 0.301 (n_itens≥2=2591), k2: 0.302 (n_itens≥2=1666), k3: 0.237 (n_itens≥2=836)
- `demo_extended_K4` — k0: 0.307 (n_itens≥2=2556), k1: 0.258 (n_itens≥2=1959), k2: 0.290 (n_itens≥2=2847), k3: 0.293 (n_itens≥2=1589)
- `behavioral_K4` — k0: 0.518 (n_itens≥2=1849), k1: 0.121 (n_itens≥2=296), k2: 0.313 (n_itens≥2=2533), k3: 0.242 (n_itens≥2=2245)
- `demo_base_K8` — k0: 0.307 (n_itens≥2=1286), k1: 0.107 (n_itens≥2=396), k2: 0.310 (n_itens≥2=825), k3: 0.201 (n_itens≥2=593), k4: 0.291 (n_itens≥2=1192), k5: 0.441 (n_itens≥2=132), k6: 0.299 (n_itens≥2=1371), k7: 0.321 (n_itens≥2=533)
- `demo_extended_K8` — k0: 0.311 (n_itens≥2=883), k1: 0.335 (n_itens≥2=157), k2: 0.284 (n_itens≥2=1042), k3: nan (n_itens≥2=0), k4: nan (n_itens≥2=0), k5: 0.316 (n_itens≥2=509), k6: 0.228 (n_itens≥2=647), k7: 0.259 (n_itens≥2=349)
- `behavioral_K8` — k0: 0.313 (n_itens≥2=910), k1: 0.260 (n_itens≥2=898), k2: 0.207 (n_itens≥2=369), k3: 0.486 (n_itens≥2=947), k4: 0.388 (n_itens≥2=1157), k5: 0.123 (n_itens≥2=54), k6: 0.210 (n_itens≥2=1141), k7: 0.582 (n_itens≥2=841)

## A.4 — onde os dados caem na curva de contraste do Stage 1

No grid sintético (densidade EXIST), subpop-aware ultrapassa single-truth a partir de divisividade ≈ **0.15**. Comparação:

| definição | divisividade empírica (confiável) | acima do cruzamento 0.15? | divisividade ACIMA DO NULO (obs − nulo) | acima de 0.15? |
|---|---|---|---|---|
| `demo_base_K4` | 0.304 | SIM | -0.042 | não |
| `demo_extended_K4` | 0.329 | SIM | -0.039 | não |
| `behavioral_K4` | 0.411 | SIM | +0.075 | não |
| `demo_base_K8` | 0.424 | SIM | -0.002 | não |
| `demo_extended_K8` | 0.467 | SIM | +0.028 | não |
| `behavioral_K8` | 0.459 | SIM | +0.033 | não |

⚠ **A comparação com 0.15 é direcional, não pontual.** O cruzamento sintético foi medido com **K=2** e subpopulações de informação equilibrada; a divisividade empírica vem de **K ∈ {4,8}** com células fragmentadas. As duas quantidades não são comensuráveis ponto a ponto — servem para dizer de que lado do regime os dados caem, não para prever um valor de r.

**Como ler estas duas colunas.** A divisividade sintética do Stage 1 é a fração de itens em que as VERDADES das subpops diferem — quantidade limpa, sem ruído. A divisividade empírica é medida sobre votos ruidosos, logo está inflacionada: parte dela seria observada mesmo com subpopulações aleatórias. A comparação honesta com o cruzamento de 0.15 é a coluna **obs − nulo**, não a coluna bruta.

## Aviso de circularidade (aplica-se só a `behavioral`)

As definições `behavioral_*` clusterizam anotadores **pelos próprios votos** (perfil de desvio face ao consenso). Medir depois se esses clusters explicam a discordância dos mesmos votos é **parcialmente tautológico**: um z alto ali é o esperado por construção, não evidência de estrutura latente. O nulo de permutação **não** corrige isto — baralha os rótulos, mas os rótulos originais já foram ajustados aos dados. Só as definições demográficas (`demo_base`, `demo_extended`) usam informação externa aos votos e por isso só elas dão um teste honesto da hipótese de subpopulação.

## Conclusão por definição

- `demo_base_K4`: **subpopulação justifica-se, evidência marginal (nenhum z > 3)** — 2 dos 3 indicadores acima do nulo condicionado.
- `demo_extended_K4`: **inconclusivo** — só 1 dos 3 indicadores excede o nulo.
- `behavioral_K4`: **subpopulação justifica-se, evidência clara** — 3 dos 3 indicadores acima do nulo condicionado ⚠ **mas ver o aviso de circularidade acima — este veredito não é evidência independente**.
- `demo_base_K8`: **não se justifica** — nenhum indicador excede o nulo condicionado: a discordância é ruído difuso, não estrutura de subgrupo.
- `demo_extended_K8`: **inconclusivo por esparsidade** — só 17.5% dos itens têm duas subpops com ≥2 votos, logo o subconjunto de confiança não sustenta o veredito.
- `behavioral_K8`: **subpopulação justifica-se, evidência clara** — 3 dos 3 indicadores acima do nulo condicionado ⚠ **mas ver o aviso de circularidade acima — este veredito não é evidência independente**.

