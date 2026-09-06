# Stage 1 — validação sintética do estimador tipo-NUTMEG (EM)

Gerado por `python -m reliability.validate --all`. Env local: numpy 2.2.6, scipy 1.16.3.
Config da grade: 800 itens, 60 anotadores, 6 anotações/item (80 votos/anotador, min 60),
minoria 25%, seeds [42, 7, 2025], spam_rate ∈ {0.15, 0.30, 0.45}.

## 1. Recuperação de θ — Pearson r médio

| divisiveness | subpop_aware | single_truth (tipo-MACE) | delta |
|---|---|---|---|
| 0.00 | 0.763 | **0.883** | −0.119 |
| 0.15 | 0.768 | 0.723 | +0.045 |
| 0.30 | 0.757 | 0.510 | +0.246 |
| 0.50 | 0.765 | 0.317 | +0.448 |
| 0.70 | 0.754 | 0.287 | +0.467 |
| **GLOBAL** | **0.761** | **0.544** | **+0.217** |

Referência do paper: NUTMEG r≈0.81 vs MACE r≈0.58. A ordem de grandeza e a direção
reproduzem-se.

**Fenómeno central reproduzido:** o subpop-aware é **plano** em divisiveness (0.754–0.768)
enquanto o single-truth **degrada monotonicamente** (0.883 → 0.287). Isto é exatamente o
argumento de tese: a discordância sistemática não é ruído, e um estimador de verdade única
converte-a em "incompetência".

**Reportar honestamente:** em `divisiveness = 0` o single-truth **ganha** (0.883 vs 0.763).
Não é um bug — sem divisiveness a verdade é mesmo única, e K=1 junta todos os votos numa
só célula (6 votos/célula em vez de ~4.5 + ~1.5), logo estima T melhor. O custo do
subpop-aware é fragmentar evidência; o benefício aparece assim que a divisiveness > ~0.10.
Este trade-off tem de ir para a tese, não ser escondido.

Artefacto bruto: `recovery_grid.csv` (90 linhas: divisiveness × spam × seed × estimador).

## 2. Preservação de discordância minoritária — **PASS**

Limiares **pré-registados** em `validate.py` (fixados antes da execução):
`floor θ_minoria ≥ 0.70` · `gap(maioria − minoria) ≤ 0.10` · `separação vs spammers ≥ 0.25`.

Setup: minoria 25% com θ **fixo em 0.90**, divisiveness 0.70 (discorda da maioria na
maioria dos itens), spam médio 0.35 na maioria.

| seed | θ̂ minoria | θ̂ maioria fiável | θ̂ spammers | gap | sep | veredito |
|---|---|---|---|---|---|---|
| 42 | 0.955 | 0.890 | 0.521 | −0.065 | +0.434 | PASS |
| 7 | 0.960 | 0.874 | 0.509 | −0.086 | +0.451 | PASS |
| 2025 | 0.921 | 0.885 | 0.419 | −0.036 | +0.501 | PASS |

**VEREDITO GLOBAL: PASS.** A minoria consistente recebe θ̂ **acima** da maioria fiável
(gap negativo) e ~0.45 acima dos spammers reais. O método não rompe o LeWiDi.

Artefacto: `minority_preservation.json`.

## 3. Determinismo

Mesma seed → θ idêntico bit a bit (`max|Δ| = 0.000e+00`).

## 4. Notas de implementação relevantes

- Init determinístico: `θ=0.8` + `ξ` uniforme faz o 1.º E-step colapsar no voto
  maioritário suavizado por (item, subpop) — é o init recomendado, sem aleatoriedade.
  `n_restarts` existe mas o default é 1.
- Baseline single-truth = **a mesma classe** com `subpop = zeros(M)` (K=1). Sem segundo
  estimador → comparação apples-to-apples garantida.
- Células (item, subpop) sem votos ficam no prior uniforme e **não contribuem** para a
  log-lik. Isto vai ser o regime dominante nos dados reais (6 anotadores/item, K=4–8).
- Convergência típica: ~130 iterações, tol 1e-6.

---

## 5. Validação no REGIME DE DENSIDADE REAL (acrescentado antes do gate)

A grade da §1 corre com 80 votos/anotador. Os dados reais do EXIST têm **~27**
(9–54). Validar só no regime fácil deixaria o gate a decidir com o número errado.
Repetido com `n_items=3984, n_annotators=887, annotations_per_item=6`
(→ 23 904 votos, média 26.9/anotador, **min 10–14: o aviso `< 15` dispara em todas as
seeds**, tal como vai acontecer nos dados reais).

| divisiveness | subpop_aware | single_truth | delta |
|---|---|---|---|
| 0.00 | 0.602 | **0.692** | −0.090 |
| 0.15 | 0.603 | 0.596 | +0.007 |
| 0.30 | 0.602 | 0.458 | +0.144 |
| 0.50 | 0.595 | 0.334 | +0.261 |
| 0.70 | 0.594 | 0.287 | +0.307 |
| **GLOBAL** | **0.599** | **0.473** | **+0.126** |

Artefacto: `recovery_grid_existdensity.csv`.

**Leitura.** O r absoluto cai de ~0.76 para ~0.60 — custo direto de estimar θ com 27
votos em vez de 80. Isto é o **teto realista** do que a Task A do EXIST pode dar, e tem
de ser dito assim na tese: um θ com r≈0.6 ordena anotadores razoavelmente, mas não é uma
medida precisa por indivíduo. O **fenómeno qualitativo mantém-se intacto**: subpop-aware
plano (0.594–0.603), single-truth a degradar (0.692 → 0.287), cruzamento perto de
divisiveness ≈ 0.15.

**Preservação de minoria @ densidade real: PASS** nas 3 seeds
(θ̂ minoria 0.921–0.927 · gap −0.051 a −0.067 · separação +0.42 a +0.45).

**Anotadores com 0 votos:** testado explicitamente (5 slots sem votos) → θ = 0.5 exato
(média do prior Beta(0.5, 0.5)), sem NaN, sem crash.
