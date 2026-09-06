# Parte B — spammer de viés constante (braço sintético novo)

Fecha o ponto cego identificado no Stage 2: um anotador que vota SEMPRE a mesma classe
(preguiça/viés), em vez de uniformemente aleatório. Gerado por
`python -m reliability.validate --bias-spammer`. Artefacto: `bias_spammer_test.json`.

## Setup

Três populações **explícitas** no mesmo dataset (800 itens, 60 anotadores, 6 votos/item,
divisividade 0.30):

| população | fração | θ real | comportamento ao spammar |
|---|---|---|---|
| fiáveis | resto | Beta(μ=0.70) e ≥0.80 no grupo medido | — |
| spam uniforme | 20% | 0.25 | voto aleatório |
| **spam de viés constante** | 20% | 0.25 | vota **sempre YES** |

> Nota de execução: a primeira versão lia o grupo "spam uniforme" da cauda da Beta
> (`true_theta ≤ 0.50`), o que dava 0–4 anotadores — amostra pequena demais para comparar
> populações, e produzia FAILs no critério c3 por ruído de amostragem. O gerador foi
> corrigido para as três populações explícitas que a especificação pedia. **Os limiares
> pré-registados não foram tocados.**

Três modos de subpopulação, porque a suspeita era que o subpop-aware fosse mais vulnerável:
- `subpop_aware` — subpops legítimas; os enviesados espalhados pela maioria;
- **`subpop_adversarial`** — os enviesados recebem uma **subpop só deles**: é o que um
  clustering **comportamental** produziria, já que votam todos igual. É o cenário do ponto cego;
- `single_truth` — K=1 (tipo-MACE).

Limiares **pré-registados** em `validate.py`:
`θ̂_enviesado ≤ 0.60` (c1) · `θ̂_fiável − θ̂_enviesado ≥ 0.20` (c2) ·
`|θ̂_enviesado − θ̂_uniforme| ≤ 0.15` (c3).

## Resultado principal — **PASS nos três modos**

| seed | modo | θ̂ enviesado | θ̂ uniforme | θ̂ fiável | gap | \|Δ\| | veredito |
|---|---|---|---|---|---|---|---|
| 42 | subpop_aware | 0.246 | 0.355 | 0.944 | +0.699 | 0.109 | PASS |
| 42 | **subpop_adversarial** | 0.298 | 0.343 | 0.944 | +0.647 | 0.046 | PASS |
| 42 | single_truth | 0.252 | 0.325 | 0.801 | +0.548 | 0.072 | PASS |
| 7 | subpop_aware | 0.247 | 0.332 | 0.918 | +0.671 | 0.084 | PASS |
| 7 | **subpop_adversarial** | 0.379 | 0.351 | 0.916 | +0.536 | 0.028 | PASS |
| 7 | single_truth | 0.234 | 0.309 | 0.617 | +0.382 | 0.075 | PASS |
| 2025 | subpop_aware | 0.222 | 0.371 | 0.974 | +0.752 | 0.150 | PASS |
| 2025 | **subpop_adversarial** | 0.306 | 0.374 | 0.974 | +0.668 | 0.068 | PASS |
| 2025 | single_truth | 0.224 | 0.367 | 0.821 | +0.597 | 0.143 | PASS |

**A suspeita do Stage 2 não se confirma nesta configuração.** O spammer de viés constante
recebe θ̂ ≈ 0.22–0.38 nos três modos — é detectado. Dar-lhe subpopulação própria
(`subpop_adversarial`) **empurra o θ̂ para cima de forma mensurável** (+0.05 a +0.13 face ao
`subpop_aware`), ou seja, a pressão prevista existe — mas com 20% de enviesados não chega
para inverter o veredito.

## Stress-sweep — **é aqui que o método quebra**

Modo `subpop_adversarial`, variando a fração de enviesados (diagnóstico secundário; não
altera o veredito pré-registado):

| fração de enviesados | θ̂ enviesados (média 3 seeds) | por seed | vs teto 0.60 |
|---|---|---|---|
| **0.03** (≈2 pessoas) | **0.733** | 0.611 · 0.607 · 0.981 | **ACIMA — falha** |
| **0.05** (3 pessoas) | **0.663** | 0.742 · 0.740 · 0.507 | **ACIMA — falha** |
| 0.10 | 0.478 | 0.483 · 0.445 · 0.505 | ok |
| 0.20 | 0.328 | 0.298 · 0.379 · 0.306 | ok |
| 0.35 | 0.326 | 0.301 · 0.385 · 0.293 | ok |
| 0.50 | 0.310 | 0.316 · 0.325 · 0.289 | ok |

**A vulnerabilidade é inversa ao tamanho do grupo.** Quanto MENOR o cluster de spammers
enviesados com subpopulação própria, MAIS ALTO o θ̂ que recebem — numa seed chega a **0.981**,
ou seja, um spammer classificado como anotador quase perfeito. O mecanismo é directo: com
2–3 pessoas numa subpop própria, a "verdade da subpopulação" em cada item é praticamente o
voto delas, e votar sempre a mesma coisa passa a ser "estar sempre certo". Com muitos
enviesados a célula tem votos que se contradizem com a verdade dos itens e o modelo separa-os.

## Consequência prática directa para o Stage 2

Isto **não é hipotético neste projeto**: a primeira tentativa de clustering comportamental
(SVD do resíduo) produziu clusters de tamanho **[883, 2, 1, 1]** e **[873, 2, 2, 2, 2, 2, 2, 3]**
— exatamente o regime de 1–3 pessoas por cluster em que o método atribui θ alto a ruído.

**Regra operacional que sai daqui:** nunca aceitar uma definição de subpopulação com clusters
pequenos (guarda sugerida: **cada cluster ≥ 10% dos anotadores**, ou fundir os pequenos), e
nunca derivar subpopulações do próprio padrão de voto sem essa guarda. As definições
demográficas (`demo_base_K4`: [210, 253, 318, 106]) não têm este problema.

## Limitações deste braço

- **Sempre NO também foi corrido** (`bias_spammer_test_label0.json`), já que YES é a classe
  maioritária real (55,6%) e a prevalência podia interagir com o viés. Resultado quase
  idêntico: θ̂ enviesado 0.215–0.365 nos três modos, c1 e c2 passam em todas as 9 linhas.
  Única diferença: em `subpop_aware`/seed 2025 o c3 dá **0.150**, exatamente no limiar, e
  cai do lado do FAIL (no braço sempre-YES a mesma célula dá 0.150 e cai do lado do PASS).
  É ruído de fronteira num critério secundário, não um efeito de prevalência — os dois
  braços são equivalentes.
- O `subpop_adversarial` é um **oráculo**: coloca os enviesados juntos por construção. Um
  clustering real pode não os isolar tão limpamente — o teste é um limite superior da
  vulnerabilidade, não uma previsão do que o KMeans faz.
- Determinismo verificado (`max|Δ| = 0` entre execuções com a mesma seed); os testes do
  Stage 1 (grid e preservação de minoria) continuam a passar sem alteração após a extensão
  do gerador.
