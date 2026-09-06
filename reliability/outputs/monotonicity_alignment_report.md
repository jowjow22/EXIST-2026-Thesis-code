# Monotonicidade lida no fim da iteração completa — fecho

Gerado por `reliability/diagnose_monotonicity_alignment.py`. **Só
diagnóstico** — `nutmeg_em.py` intacto, nada sobrescrito, nada retreinado.

## 0. O que foi medido

Duas séries, cada uma função pura dos parâmetros de **um só estado**:

- **INÍCIO** — `L(θ_t, ξ_t)`, avaliada antes do M-step da iteração `t`.
  É a que `diagnose_em_convergence` já registava (e já estava coerente: a
  log-lik e o `log_prior` vinham ambos de (θ_old, ξ_old)).
- **FIM** — `L(θ_{t+1}, ξ_{t+1})`, com um **E-step novo** corrido depois do
  M-step, nos parâmetros de saída. Nenhum posterior antigo é reutilizado.

No modo MAP a quantidade com garantia de não-decrescimento é a
**log-posterior** (log-lik + log-prior); é por ela que o veredito é dado. A
log-lik nua vai ao lado, como sintoma.

## 1. Critérios pré-registados

- **M1_end_of_iteration_monotone** — Série FIM (log-posterior avaliada DEPOIS do M-step, em (θ_novo, ξ_novo)): zero quebras com folga 1e-8. PASS = a causa era a instrumentação.
- **M2_start_vs_end_identical** — As séries INÍCIO e FIM têm o mesmo número de quebras (a menos das fronteiras). PASS = mudar o ponto de leitura não altera o diagnóstico.

## 2. Resultado — M-step do repositório (responsabilidade CODE) `[MEDIDO]`

| série | ponto do ciclo | transições | quebras | queda máx |
|---|---|---|---|---|
| log-posterior (CODE) | INÍCIO | 461 | **273** | 1.518e-01 |
| log-posterior (CODE) | FIM | 461 | **274** | 1.518e-01 |
| log-lik nua (CODE) | INÍCIO | 461 | **275** | 1.622e-01 |
| log-lik nua (CODE) | FIM | 461 | **276** | 1.622e-01 |

**M1: FAIL** — a log-posterior lida no FIM da iteração
completa quebra **274** vezes, queda máx
`1.518e-01`.

`[MEDIDO]` **As violações PERSISTEM.** Mudar o ponto de leitura não as
remove. A instrumentação **não** era a causa.

### 2.1 Porque tinha de dar isto

`[MEDIDO]` As duas séries são a mesma sequência desfasada de uma posição: o
E-step do fim da iteração `t` e o do início da iteração `t+1` correm nos
mesmos parâmetros. Verificado numericamente — máx
`|lp_end[t] − lp_start[t+1]|` = `0.000e+00` em
461 pares.

**M2: PASS** — o número de quebras coincide
nas duas séries (a menos das fronteiras). Ler no fim em vez de no meio é a
mesma medição, não uma medição diferente.

## 3. A causa real, para não ficar em aberto `[MEDIDO]`

Com a **responsabilidade do M-step corrigida** (derivação em
`mstep_responsibility_diagnosis.md`: o código conta o voto do próprio
anotador duas vezes), no MESMO M-step do repositório e nos MESMOS dois
pontos de leitura:

| série | ponto do ciclo | transições | quebras | queda máx |
|---|---|---|---|---|
| log-posterior (CORRIGIDA) | INÍCIO | 291 | **0** | 0.000e+00 |
| log-posterior (CORRIGIDA) | FIM | 291 | **0** | 0.000e+00 |

A quebra desaparece nos dois pontos de leitura. **Havia bug no estimador** —
na responsabilidade, não na instrumentação.

## 4. Veredito

`[MEDIDO]` A hipótese "a causa era a instrumentação" está **REJEITADA**. A
log-posterior lida no fim da iteração completa, com θ e ξ do mesmo estado,
continua a quebrar 274 vezes. A causa é a
responsabilidade de spam do M-step, e corrigi-la leva as quebras a zero.

`[MEDIDO]` Carregamento: 3984 itens · 887 anotadores · 23904 votos. Corridas: CODE 462 iterações (convergida=True) · CORRIGIDA 292 (convergida=True).
