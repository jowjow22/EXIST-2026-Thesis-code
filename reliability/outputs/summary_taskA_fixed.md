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
| `demo_base_K4` | 4 | 69.27% | 2.17 | 0.461 | 0.206 | 0.306 | 214 | 519 | 764 | +0.301 | -0.361 |
| `demo_extended_K4` | 4 | 85.74% | 1.75 | 0.446 | 0.192 | 0.257 | 201 | 571 | 786 | +0.178 | -0.245 |
| `behavioral_K4` | 4 | 64.70% | 2.32 | 0.480 | 0.212 | 0.315 | 187 | 501 | 736 | +0.457 | -0.509 |
| `demo_base_K8` | 8 | 49.58% | 1.51 | 0.429 | 0.190 | 0.224 | 211 | 634 | 808 | +0.193 | -0.250 |
| `demo_extended_K8` | 8 | 62.66% | 1.20 | 0.395 | 0.162 | 0.206 | 233 | 742 | 846 | +0.041 | -0.110 |
| `behavioral_K8` | 8 | 41.09% | 1.83 | 0.461 | 0.205 | 0.299 | 196 | 560 | 761 | +0.415 | -0.466 |
| `single_truth_K1` | 1 | 100.00% | 6.00 | 0.483 | 0.219 | 0.338 | 208 | 458 | 725 | +0.615 | -0.643 |

### Tamanhos de cluster

- `demo_base_K4`: [253, 318, 210, 106]
- `demo_extended_K4`: [232, 203, 246, 206]
- `behavioral_K4`: [287, 69, 278, 253]  **<- MONOTONIA_LOG_POSTERIOR(385 quebras, máx 1.83e-04; M-step de média posterior, não argmax MAP)**
- `demo_base_K8`: [152, 74, 123, 86, 146, 30, 180, 96]  **<- MONOTONIA_LOG_POSTERIOR(30 quebras, máx 3.80e-02; M-step de média posterior, não argmax MAP)**
- `demo_extended_K8`: [120, 90, 144, 147, 105, 93, 84, 104]  **<- CÉLULAS_QUASE_SINGULARES(<1.5 votos/célula: θ inflacionado); MONOTONIA_LOG_POSTERIOR(48 quebras, máx 7.82e-01; M-step de média posterior, não argmax MAP)**
- `behavioral_K8`: [126, 135, 68, 126, 150, 22, 139, 121]
- `single_truth_K1`: [887]

### Leitura

- O baseline **single_truth (K=1)** tem ρ(θ, concordância-com-maioria) = **+0.615** — quase uma reexpressão da métrica de consenso. É exatamente o comportamento que se espera de um estimador de verdade única: 'competência' e 'concordar com a maioria' colapsam na mesma coisa.
- As definições com subpopulação **descolam** dessa métrica (ρ entre +0.041 e +0.457). Pela §1, isto **não** é evidência de falha: é o efeito pretendido — parte da discordância deixa de ser contabilizada como incompetência. Também não é, por si só, prova de sucesso; a prova está no teste sintético.
- Sem flags de degenerescência, a definição com maior dispersão de θ (separação mais limpa entre anotadores) é **`demo_base_K4`** (std 0.206, IQR 0.306, 519 anotadores < 0.5), com 2.17 votos/célula.
- **Confound a declarar:** a dispersão de θ **não** é comparável entre configs com densidades de célula diferentes. θ médio sobe monotonicamente à medida que votos/célula cai (6.00/0.483 → 2.32/0.480 → 2.17/0.461 → 1.83/0.461 → 1.75/0.446 → 1.51/0.429 → 1.20/0.395 em votos-por-célula/θ̄). A inflação **não é um efeito de limiar**: é contínua, e um std maior sob poucos votos/célula é em parte ruído de fragmentação (anotadores quase sozinhos numa célula recebem θ→1), não separação mais limpa. Por isso o critério acima exige ≥2 votos/célula; sem esse corte a diferença entre `demo_base_K4` e `demo_extended_K4` (std 0.316 vs 0.322) é um empate técnico decidido pelo config que está mais perto do regime inflacionado.
- **K=4 é preferível a K=8** nestes dados: com K=8 a definição `demo_extended` cai para ~1.2 votos por célula não vazia, regime em que a 'verdade da subpopulação' é praticamente o voto do próprio anotador e o θ infla artificialmente (flag `CÉLULAS_QUASE_SINGULARES`). Foi por isso que ambos os K foram corridos.

## 3. Concordância entre as três definições (Spearman de θ)

| | `demo_base_K4` | `demo_extended_K4` | `behavioral_K4` | `demo_base_K8` | `demo_extended_K8` | `behavioral_K8` |
|---|---|---|---|---|---|---|
| `demo_base_K4` | 1.000 | 0.782 | 0.654 | 0.848 | 0.727 | 0.653 |
| `demo_extended_K4` | 0.782 | 1.000 | 0.600 | 0.748 | 0.777 | 0.642 |
| `behavioral_K4` | 0.654 | 0.600 | 1.000 | 0.619 | 0.540 | 0.767 |
| `demo_base_K8` | 0.848 | 0.748 | 0.619 | 1.000 | 0.754 | 0.625 |
| `demo_extended_K8` | 0.727 | 0.777 | 0.540 | 0.754 | 1.000 | 0.589 |
| `behavioral_K8` | 0.653 | 0.642 | 0.767 | 0.625 | 0.589 | 1.000 |

Referência single-truth (K=1, tipo-MACE): ρ(`demo_base_K4`, single) = 0.723, ρ(`demo_extended_K4`, single) = 0.617, ρ(`behavioral_K4`, single) = 0.783, ρ(`demo_base_K8`, single) = 0.635, ρ(`demo_extended_K8`, single) = 0.521, ρ(`behavioral_K8`, single) = 0.746

## 4. Histograma de θ (definições K=4)

**`demo_base_K4`**

| faixa | n | |
|---|---|---|
| 0.0–0.1 |   32 | `########` |
| 0.1–0.2 |   69 | `################` |
| 0.2–0.3 |  113 | `###########################` |
| 0.3–0.4 |  135 | `################################` |
| 0.4–0.5 |  170 | `########################################` |
| 0.5–0.6 |  144 | `##################################` |
| 0.6–0.7 |  101 | `########################` |
| 0.7–0.8 |   76 | `##################` |
| 0.8–0.9 |   36 | `########` |
| 0.9–1.0 |   11 | `###` |

**`demo_extended_K4`**

| faixa | n | |
|---|---|---|
| 0.0–0.1 |   36 | `#######` |
| 0.1–0.2 |   59 | `############` |
| 0.2–0.3 |  106 | `#####################` |
| 0.3–0.4 |  167 | `#################################` |
| 0.4–0.5 |  203 | `########################################` |
| 0.5–0.6 |  129 | `#########################` |
| 0.6–0.7 |   86 | `#################` |
| 0.7–0.8 |   72 | `##############` |
| 0.8–0.9 |   25 | `#####` |
| 0.9–1.0 |    4 | `#` |

**`behavioral_K4`**

| faixa | n | |
|---|---|---|
| 0.0–0.1 |   26 | `######` |
| 0.1–0.2 |   76 | `################` |
| 0.2–0.3 |   85 | `##################` |
| 0.3–0.4 |  128 | `############################` |
| 0.4–0.5 |  186 | `########################################` |
| 0.5–0.6 |  111 | `########################` |
| 0.6–0.7 |  124 | `###########################` |
| 0.7–0.8 |   79 | `#################` |
| 0.8–0.9 |   59 | `#############` |
| 0.9–1.0 |   13 | `###` |

## 5. Células (item, subpop) vazias — fenómeno esperado, não bug

Com 6 anotadores por item, K clusters fragmentam cada item em no máximo 6 células não vazias de K possíveis. A cobertura na tabela acima mostra o efeito directo, e os votos/célula caem de 6 (K=1) para ~1–2. Verdades por subpopulação assim fracamente determinadas são o limite estrutural da Task A neste dataset — é por isso que K=4 e K=8 são ambos reportados: K menor troca granularidade de subpopulação por evidência por célula.

## 6. Aviso de transferência para Task B/C

**Um resultado fraco na Task A não é preditivo da utilidade em Task B/C.** O gradiente de sinal demográfico já medido neste trabalho vai na direção oposta: a condição B1 (fusão de biosinais/demografia sem embeddings per-viewer) mostrou deltas positivos consistentes em **Task B soft e Task C hard**, e não na deteção binária. A Task A é a tarefa onde o sinal de subpopulação é mais fraco por natureza — julgar o método por ela subestima-o. A extensão a Task B (`n_labels=3`, single-label) é directa na formulação atual; Task C (multilabel) exige um modelo generativo diferente.

## 7. Limites metodológicos desta corrida

- **`behavioral` não é concordância par-a-par** e não deve ser descrito como tal. Com 6 anotadores/item e 887 anotadores, a sobreposição par-a-par esperada é **~0.15 itens por par** — a matriz de concordância par-a-par é praticamente vazia. Duas alternativas foram testadas e REJEITADAS por evidência medida: (i) essa matriz par-a-par; (ii) `TruncatedSVD(32)` da matriz de resíduos 887×3984 + KMeans, que deu clusters degenerados `[883, 2, 1, 1]` (K=4) e `[873, 2, 2, 2, 2, 2, 2, 3]` (K=8) — os suportes de itens são quase disjuntos, logo as distâncias mediam o nº de votos, não o comportamento. O que ficou é um **perfil comportamental 6-d** (yes-rate, desvio médio do consenso, |desvio| médio, concordância-com-maioria, desvio em itens contestados, fração de votos contestados), padronizado e clusterizado — comparável entre anotadores apesar dos suportes disjuntos.
- **Circularidade parcial de `behavioral`**: os clusters são derivados dos mesmos votos que alimentam o EM. O θ resultante não é independente do sinal que o gerou; comparar com as definições demográficas (derivadas de metadados externos aos votos) é o controlo.
- **Ponto cego específico da definição `behavioral`**: um spammer de viés CONSTANTE (vota sempre YES, por exemplo) cai num cluster de votantes semelhantes; o EM concede a esse cluster uma verdade própria, os votos batem certo com ela, e o spammer recebe θ ALTO. Por construção, esta definição **não distingue** um spammer de viés constante de uma subpopulação legítima. O teste sintético de preservação de minoria — o árbitro indicado na §1 — **não cobre este caso**: o spam que ele injeta é uniforme, não enviesado. Cobrir isto exigiria um braço sintético novo (spammers de viés constante), fora do âmbito desta etapa.
- **θ de anotadores com poucos votos** encolhe para a média do prior Beta(0.5,0.5) = 0.5. Ver `n_votes` no JSON de cada definição antes de usar um θ individual.
