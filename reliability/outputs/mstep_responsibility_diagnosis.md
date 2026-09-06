# A responsabilidade do M-step é a causa da quebra de monotonicidade

Gerado por `reliability/diagnose_mstep_responsibility.py`. **Só diagnóstico** —
`nutmeg_em.py` intacto, nenhum ficheiro de θ ou de soft labels regenerado,
nada retreinado. Todos os recálculos correram em memória.

> **SUBSTITUI a conclusão da §6 de `em_convergence_diagnosis.md`.** Aquele
> relatório concluiu «as soft labels ponderadas não precisam de ser refeitas»,
> e a conclusão era correcta **para a causa que testava** (paragem prematura do
> EM, deslocamento 0,000317). Esta etapa encontra uma segunda causa,
> independente, de magnitude 77× maior. A conclusão anterior fica revogada
> quanto às soft labels; tudo o resto dela mantém-se.

> **PORTÃO.** Os 6 retreinos a 15 épocas ficam **em espera** até haver decisão
> sobre as labels. O que **não** depende da decisão e sobrevive intacto: o split
> congelado (`derived_from = soft_labels_taskA_raw.json`, verificado — não toca
> em θ), o braço A0, e o hard gold.

## 0. A quinta explicação estava morta antes de ser testada

`[MEDIDO]` `diagnose_em_convergence.run_instrumented` já avalia `ll` no E-step
de (θ_old, ξ_old) e `log_prior(θ_old, ξ_old)` no **mesmo par**. A série de
log-posterior já era a quantidade certa medida no ponto certo. Desalinhamento
de ciclo está **excluído** — não por experiência nova, por leitura do código
que produziu o número. A tarefa passa a ser identificar a causa, e ela é outra.

## 1. A derivação

Modelo: `P(v | T=l) = θ_j·1{v==l} + (1−θ_j)·ξ_j[v]`.

O E-step do indicador de spam `s_ij`, marginalizando T:

```
r = P(s=0 | V) = Σ_l P(T=l | V) · P(s=0 | T=l, v)
               = p_true · θ / (θ + (1−θ)·ξ[v])          <- CORRIGIDA
```

O que `nutmeg_em._m_step` calcula:

```
r = θ·p_true / (θ·p_true + (1−θ)·ξ[v])                  <- CODE
```

A forma CODE **é** correcta, mas só quando `p_true` é o posterior
**leave-one-out** `P(T=v | votos da célula excluindo j)`. Com `q` = LOO,
`p = q·(θ+(1−θ)ξ)/Z`, logo `p·θ/(θ+(1−θ)ξ) = θq/Z = θq/(θq+(1−θ)ξ)`: as duas
formas correctas coincidem. O código alimenta a forma LOO com o posterior
**completo**, que já contém o voto de j — **o voto é contado duas vezes**.

Direcção do viés: para `p<1`, `CODE ≥ CORRIGIDA` sempre, com factor de
inflação função de `p_true` e `ξ` → **θ enviesado para cima, não uniformemente**.

`[LIMITAÇÃO]` separada e não confundida com o bug: o M-step codificado é
**média posterior** `(Σr+a)/(n+a+b)`, não o argmax MAP `(Σr+a−1)/(n+a+b−2)`.
Médias posteriores não têm garantia de subida mesmo com `r` correcta. Por isso
o PASS/FAIL principal corre na variante **ML-argmax**, onde o M-step é o
maximizador exacto de Q e a garantia sobre a log-lik nua é limpa.

## 2. Critérios pré-registados

- **C1_synthetic_recovery** — RMSE(θ̂, θ_true) da variante CORRIGIDA <= RMSE da variante CODE + 0.005 em 3 sementes sintéticas. PASS = corrigir não piora a recuperação.
- **C2_monotonicity_corrected** — Variante ML-ARGMAX + r CORRIGIDA, dados reais: ZERO quebras da log-lik nua com folga 1e-8. PASS = a responsabilidade era a causa.
- **C2b_control_code** — Controlo: ML-ARGMAX + r CODE tem >0 quebras. Se der 0, C2 não discrimina nada e o teste é declarado insensível.
- **C3_theta_shift** — Descritivo, sem PASS/FAIL: r de Pearson, média|Δθ|, máx|Δθ| e nº de anotadores que cruzam θ=0,5 entre θ_CORRIGIDO e θ do ficheiro em uso.
- **C4_softlabel_shift** — MATERIAL se média|Δp_YES| > 0.0072 (10% do efeito da ponderação, 0.0720); IMATERIAL caso contrário. Padrões de comparação: 0.0720 = efeito da ponderação; 0.000317 = deslocamento por convergir o EM.

## 3. C1 — validação sintética (antes dos dados reais)

| seed | RMSE CODE | RMSE CORRIGIDA | r CODE | r CORRIGIDA | viés CODE | viés CORRIGIDA |
|---|---|---|---|---|---|---|
| 42 | 0.1845 | 0.1907 | 0.6420 | 0.6429 | -0.0496 | -0.0861 |
| 7 | 0.2164 | 0.2315 | 0.2960 | 0.2906 | -0.0552 | -0.1117 |
| 2025 | 0.1801 | 0.1857 | 0.7043 | 0.7002 | -0.0370 | -0.0731 |

**C1: FAIL** (3/3 sementes em que a corrigida piora >0,005). O critério foi pré-registado e **não é alterado retroactivamente**.

### 3.1 Análise post-hoc do FAIL `[POST-HOC]`

Rotulada como post-hoc: não substitui o resultado pré-registado acima.

`[MEDIDO]` **Os dois estimadores enviesam θ para BAIXO**; o corrigido enviesa
mais. A responsabilidade inflacionada do código empurra θ para cima e
**cancela parcialmente** o viés descendente do EM neste regime (~80 votos por
anotador). O RMSE menor do código é esse cancelamento acidental, não
informação a mais.

`[MEDIDO]` **A correlação com o θ verdadeiro é praticamente idêntica** nas duas
variantes (diferenças na 3ª casa, sem direcção consistente): a **ordenação** dos
anotadores não muda. O que muda é o **nível** — e de forma não uniforme, que é
o que sobra depois da normalização de pesos dentro do item (§6).

`[LIMITAÇÃO]` A recuperação absoluta é fraca nas duas variantes
(r ≈ 0,30–0,70 com 6 votos/item). O critério C1 mede diferença entre duas
variantes num regime onde nenhuma recupera bem — é pouco sensível, e
declara-se assim. O critério que discrimina é o C2, e esse é categórico.

## 4. C2 — monotonicidade em dados reais (ML-argmax, log-lik nua)

| variante | transições | quebras | queda máx |
|---|---|---|---|
| r CODE (controlo) | 588 | 90 | 9.063e-01 |
| r CORRIGIDA | 498 | 0 | 0.000e+00 |

**C2 (corrigida sem quebras): PASS** · **C2b (controlo quebra): PASS**

`[MEDIDO]` Secundário — a variante do repositório (MAP média posterior) com
`r` corrigida: **0 quebras** da log-posterior em
291 transições, queda máx `0.000e+00`, convergida=True em 292 iterações.
Resíduo aqui atribui-se à média posterior, não à responsabilidade.

## 5. C3 — quanto θ se move

θ_CORRIGIDO = `r` corrigida + M-step do repositório (MAP média posterior),
convergido. Comparado contra os dois θ que existem hoje.

| comparação | Pearson r | Spearman ρ | média\|Δθ\| | máx\|Δθ\| | cruzam 0,5 |
|---|---|---|---|---|---|
| vs θ do ficheiro (o que A1 usou) | 0.8923 | 0.8977 | 0.0794 | 0.7166 | 95 |
| vs θ CODE convergido | 0.8901 | 0.8962 | 0.0798 | 0.7166 | 95 |

`[MEDIDO]` θ_CORRIGIDO: média 0.4826 · θ_CODE convergido: média 0.5344. θ_CODE ≥ θ_CORRIGIDO em 74.4% dos anotadores — a direcção prevista pela derivação.

## 6. C4 — quanto as soft labels se movem (a pergunta que decide)

| grandeza | valor |
|---|---|
| itens | 3984 |
| média \|Δp_YES\| (θ corrigido vs θ do ficheiro) | **0.024565** |
| p95 \|Δp_YES\| | 0.118038 |
| máx \|Δp_YES\| | 0.436497 |
| itens que cruzam p_YES = 0,5 | 134 |
| padrão: efeito da própria ponderação | 0.0720 |
| padrão: deslocamento por convergir o EM | 0.000317 |
| efeito da ponderação sob θ CORRIGIDO (vs soft label crua) | 0.0617 |
| limiar de materialidade pré-registado | 0.0072 |

**C4: MATERIAL.** O deslocamento é 34.12% do efeito da ponderação e 77× o deslocamento por convergir.

`[MEDIDO]` **Atribuição limpa:** a comparação é θ_CORRIGIDO *convergido* contra
o θ do ficheiro *interrompido*, logo mistura duas causas — mas a componente de
convergência está **limitada por 0.000317** (medida em
`em_convergence_diagnosis.md` §6, mesma população). Praticamente todo o
deslocamento vem da responsabilidade.

`[MEDIDO]` Sob θ corrigido, a **intervenção** (ponderar vs não ponderar) vale
0.0617 de deslocamento médio contra
0.0720 sob o θ actual. Isto entra na tabela de poder: muda o
tamanho do alvo que os retreinos têm de detectar.

`[MEDIDO]` As soft labels ponderadas **dependem** da responsabilidade errada.
A decisão é do autor e tem de ser tomada **antes** dos 6 retreinos — depois
seria comparar dois alvos diferentes. As duas opções, sem recomendação:

- **(a) Manter as labels actuais.** Os resultados a 2 épocas continuam
  internamente válidos. A tese descreve os pesos do A1 como *ponto fixo
  enviesado, próximo do estimador MACE mas não igual a ele*, e diz porquê.
- **(b) Regenerar com θ corrigido.** O nulo a 2 épocas passa a ser o registo
  do alvo antigo, e os 6 retreinos a 15 épocas tornam-se o **primeiro**
  resultado sobre o alvo corrigido — não há comparação 2 vs 15 épocas.

## 7. O que isto obriga a dizer na tese

`[MEDIDO]` O estimador implementado **não** é o EM de MACE tal como derivado:
a responsabilidade do M-step conta o voto do próprio anotador duas vezes. É um
ponto fixo enviesado para cima, próximo do estimador mas não igual a ele.
`[LIMITAÇÃO]` Se a tese afirmar "MACE na formulação original", a afirmação tem
de ser verificada contra o M-step descrito em Hovy et al. (NAACL 2013) — o
artigo suaviza as contagens do M-step e a variante corrigida pode ou não
coincidir com ele. Não afirmar de memória.

`[MEDIDO]` Carregamento: 3984 itens · 887 anotadores · 23904 votos.
