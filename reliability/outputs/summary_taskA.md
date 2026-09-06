# Task A (memes) — θ por anotador, três definições de subpopulação

Gerado por `python -m reliability.run_exist`. CPU-only, sem GPU.

## 0. Dados

- itens **3984** · anotadores **887** · votos **23904** · descartados 0
- YES = 55.6% dos votos
- votos/anotador: mean 26.9 · min 9 · max 54 · **11 anotadores abaixo de 15 votos** (θ desses é fraco por construção)
- Krippendorff α (nominal, todos os anotadores) = **0.2791**
- concordância-com-maioria: mean 0.771 · min 0.077 · max 1.000
- demografia inconsistente resolvida por moda: `Annotator_125` `countries_annotators` {'Romania': 27, 'Spain': 27}
- demografia inconsistente resolvida por moda: `Annotator_187` `study_levels_annotators` {"Bachelor's degree": 24, "Master's degree": 27}

## 1. Enquadramento obrigatório dos sanity checks

As duas métricas externas — **concordância-com-maioria** e **Krippendorff α leave-one-out** — são **diagnósticos descritivos com viés conhecido para o consenso**, NÃO critérios de correção do θ. Por construção penalizam a minoria sistemática que este método existe para proteger: um anotador de uma subpopulação coerente mas minoritária discorda da maioria *por ter razão dentro do seu grupo*, e ambas as métricas leem isso como erro.

Por isso, qualquer divergência entre θ e estas métricas é interpretada com o **teste sintético de preservação de minoria como árbitro** (`outputs/minority_preservation.json`, veredito PASS nas 3 seeds e nas duas densidades) — só ele distingue ruído de discordância legítima, porque só ele tem acesso à verdade. Um ρ(θ, concordância) alto **não** valida o θ, e um ρ baixo **não** o invalida.

## 2. Distribuição de θ por definição

| definição | K | cobertura células | votos/célula | θ mean | θ std | θ IQR | <0.3 | <0.5 | <0.7 | ρ(θ, concord.) | ρ(θ, Δα) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `demo_base_K4` | 4 | 69.27% | 2.17 | 0.650 | 0.316 | 0.601 | 171 | 327 | 449 | +0.144 | -0.200 |
| `demo_extended_K4` | 4 | 85.74% | 1.75 | 0.697 | 0.322 | 0.585 | 157 | 288 | 364 | +0.017 | -0.069 |
| `behavioral_K4` | 4 | 64.70% | 2.32 | 0.662 | 0.296 | 0.554 | 144 | 280 | 437 | +0.138 | -0.196 |
| `demo_base_K8` | 8 | 49.58% | 1.51 | 0.738 | 0.315 | 0.502 | 134 | 242 | 310 | +0.027 | -0.087 |
| `demo_extended_K8` | 8 | 62.66% | 1.20 | 0.821 | 0.279 | 0.017 | 93 | 157 | 181 | -0.049 | -0.013 |
| `behavioral_K8` | 8 | 41.09% | 1.83 | 0.723 | 0.288 | 0.451 | 109 | 216 | 351 | -0.019 | -0.029 |
| `single_truth_K1` | 1 | 100.00% | 6.00 | 0.535 | 0.259 | 0.403 | 191 | 407 | 638 | +0.622 | -0.647 |

### Tamanhos de cluster

- `demo_base_K4`: [253, 318, 210, 106]
- `demo_extended_K4`: [232, 203, 246, 206]
- `behavioral_K4`: [287, 69, 278, 253]
- `demo_base_K8`: [152, 74, 123, 86, 146, 30, 180, 96]
- `demo_extended_K8`: [120, 90, 144, 147, 105, 93, 84, 104]  **<- CÉLULAS_QUASE_SINGULARES(<1.5 votos/célula: θ inflacionado)**
- `behavioral_K8`: [126, 135, 68, 126, 150, 22, 139, 121]
- `single_truth_K1`: [887]

### Leitura

- O baseline **single_truth (K=1)** tem ρ(θ, concordância-com-maioria) = **+0.622** — quase uma reexpressão da métrica de consenso. É exatamente o comportamento que se espera de um estimador de verdade única: 'competência' e 'concordar com a maioria' colapsam na mesma coisa.
- As definições com subpopulação **descolam** dessa métrica (ρ entre -0.049 e +0.144). Pela §1, isto **não** é evidência de falha: é o efeito pretendido — parte da discordância deixa de ser contabilizada como incompetência. Também não é, por si só, prova de sucesso; a prova está no teste sintético.
- Sem flags de degenerescência, a definição com maior dispersão de θ (separação mais limpa entre anotadores) é **`demo_base_K4`** (std 0.316, IQR 0.601, 327 anotadores < 0.5), com 2.17 votos/célula.
- **Confound a declarar:** a dispersão de θ **não** é comparável entre configs com densidades de célula diferentes. θ médio sobe monotonicamente à medida que votos/célula cai (6.00/0.535 → 2.32/0.662 → 2.17/0.650 → 1.83/0.723 → 1.75/0.697 → 1.51/0.738 → 1.20/0.821 em votos-por-célula/θ̄). A inflação **não é um efeito de limiar**: é contínua, e um std maior sob poucos votos/célula é em parte ruído de fragmentação (anotadores quase sozinhos numa célula recebem θ→1), não separação mais limpa. Por isso o critério acima exige ≥2 votos/célula; sem esse corte a diferença entre `demo_base_K4` e `demo_extended_K4` (std 0.316 vs 0.322) é um empate técnico decidido pelo config que está mais perto do regime inflacionado.
- **K=4 é preferível a K=8** nestes dados: com K=8 a definição `demo_extended` cai para ~1.2 votos por célula não vazia, regime em que a 'verdade da subpopulação' é praticamente o voto do próprio anotador e o θ infla artificialmente (flag `CÉLULAS_QUASE_SINGULARES`). Foi por isso que ambos os K foram corridos.

## 3. Concordância entre as três definições (Spearman de θ)

| | `demo_base_K4` | `demo_extended_K4` | `behavioral_K4` | `demo_base_K8` | `demo_extended_K8` | `behavioral_K8` |
|---|---|---|---|---|---|---|
| `demo_base_K4` | 1.000 | 0.592 | 0.382 | 0.694 | 0.567 | 0.336 |
| `demo_extended_K4` | 0.592 | 1.000 | 0.388 | 0.628 | 0.696 | 0.383 |
| `behavioral_K4` | 0.382 | 0.388 | 1.000 | 0.418 | 0.408 | 0.391 |
| `demo_base_K8` | 0.694 | 0.628 | 0.418 | 1.000 | 0.707 | 0.363 |
| `demo_extended_K8` | 0.567 | 0.696 | 0.408 | 0.707 | 1.000 | 0.423 |
| `behavioral_K8` | 0.336 | 0.383 | 0.391 | 0.363 | 0.423 | 1.000 |

Referência single-truth (K=1, tipo-MACE): ρ(`demo_base_K4`, single) = 0.419, ρ(`demo_extended_K4`, single) = 0.348, ρ(`behavioral_K4`, single) = 0.443, ρ(`demo_base_K8`, single) = 0.358, ρ(`demo_extended_K8`, single) = 0.340, ρ(`behavioral_K8`, single) = 0.293

## 4. Histograma de θ (definições K=4)

**`demo_base_K4`**

| faixa | n | |
|---|---|---|
| 0.0–0.1 |   37 | `####` |
| 0.1–0.2 |   65 | `#######` |
| 0.2–0.3 |   69 | `#######` |
| 0.3–0.4 |   87 | `#########` |
| 0.4–0.5 |   69 | `#######` |
| 0.5–0.6 |   64 | `#######` |
| 0.6–0.7 |   58 | `######` |
| 0.7–0.8 |   33 | `###` |
| 0.8–0.9 |   18 | `##` |
| 0.9–1.0 |  387 | `########################################` |

**`demo_extended_K4`**

| faixa | n | |
|---|---|---|
| 0.0–0.1 |   37 | `###` |
| 0.1–0.2 |   64 | `#####` |
| 0.2–0.3 |   56 | `#####` |
| 0.3–0.4 |   84 | `#######` |
| 0.4–0.5 |   47 | `####` |
| 0.5–0.6 |   38 | `###` |
| 0.6–0.7 |   38 | `###` |
| 0.7–0.8 |   31 | `###` |
| 0.8–0.9 |   12 | `#` |
| 0.9–1.0 |  480 | `########################################` |

**`behavioral_K4`**

| faixa | n | |
|---|---|---|
| 0.0–0.1 |   27 | `###` |
| 0.1–0.2 |   59 | `#######` |
| 0.2–0.3 |   58 | `#######` |
| 0.3–0.4 |   73 | `########` |
| 0.4–0.5 |   63 | `#######` |
| 0.5–0.6 |   76 | `#########` |
| 0.6–0.7 |   81 | `#########` |
| 0.7–0.8 |   67 | `########` |
| 0.8–0.9 |   38 | `####` |
| 0.9–1.0 |  345 | `########################################` |

## 5. Células (item, subpop) vazias — fenómeno esperado, não bug

Com 6 anotadores por item, K clusters fragmentam cada item em no máximo 6 células não vazias de K possíveis. A cobertura na tabela acima mostra o efeito directo, e os votos/célula caem de 6 (K=1) para ~1–2. Verdades por subpopulação assim fracamente determinadas são o limite estrutural da Task A neste dataset — é por isso que K=4 e K=8 são ambos reportados: K menor troca granularidade de subpopulação por evidência por célula.

## 6. Aviso de transferência para Task B/C

**Um resultado fraco na Task A não é preditivo da utilidade em Task B/C.** O gradiente de sinal demográfico já medido neste trabalho vai na direção oposta: a condição B1 (fusão de biosinais/demografia sem embeddings per-viewer) mostrou deltas positivos consistentes em **Task B soft e Task C hard**, e não na deteção binária. A Task A é a tarefa onde o sinal de subpopulação é mais fraco por natureza — julgar o método por ela subestima-o. A extensão a Task B (`n_labels=3`, single-label) é directa na formulação atual; Task C (multilabel) exige um modelo generativo diferente.

## 7. Limites metodológicos desta corrida

- **`behavioral` não é concordância par-a-par** e não deve ser descrito como tal. Com 6 anotadores/item e 887 anotadores, a sobreposição par-a-par esperada é **~0.15 itens por par** — a matriz de concordância par-a-par é praticamente vazia. Duas alternativas foram testadas e REJEITADAS por evidência medida: (i) essa matriz par-a-par; (ii) `TruncatedSVD(32)` da matriz de resíduos 887×3984 + KMeans, que deu clusters degenerados `[883, 2, 1, 1]` (K=4) e `[873, 2, 2, 2, 2, 2, 2, 3]` (K=8) — os suportes de itens são quase disjuntos, logo as distâncias mediam o nº de votos, não o comportamento. O que ficou é um **perfil comportamental 6-d** (yes-rate, desvio médio do consenso, |desvio| médio, concordância-com-maioria, desvio em itens contestados, fração de votos contestados), padronizado e clusterizado — comparável entre anotadores apesar dos suportes disjuntos.
- **Circularidade parcial de `behavioral`**: os clusters são derivados dos mesmos votos que alimentam o EM. O θ resultante não é independente do sinal que o gerou; comparar com as definições demográficas (derivadas de metadados externos aos votos) é o controlo.
- **Ponto cego específico da definição `behavioral`**: um spammer de viés CONSTANTE (vota sempre YES, por exemplo) cai num cluster de votantes semelhantes; o EM concede a esse cluster uma verdade própria, os votos batem certo com ela, e o spammer recebe θ ALTO. Por construção, esta definição **não distingue** um spammer de viés constante de uma subpopulação legítima. O teste sintético de preservação de minoria — o árbitro indicado na §1 — **não cobre este caso**: o spam que ele injeta é uniforme, não enviesado. Cobrir isto exigiria um braço sintético novo (spammers de viés constante), fora do âmbito desta etapa.
- **θ de anotadores com poucos votos** encolhe para a média do prior Beta(0.5,0.5) = 0.5. Ver `n_votes` no JSON de cada definição antes de usar um θ individual.
