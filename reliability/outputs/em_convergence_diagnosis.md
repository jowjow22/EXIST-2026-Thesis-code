# Diagnóstico da não-convergência do estimador de θ (Task A, K=1)

Gerado por `reliability/diagnose_em_convergence.py`. **Só diagnóstico** — a
formulação, a tolerância e os priors do estimador não foram tocados.

## 0. O estimador que está no código (não o do enunciado)

`[MEDIDO]` em `reliability/nutmeg_em.py`:

- Modelo de **spam**, não Dawid-Skene simétrico:
  `P(v | T=l) = θ_j · 1{v==l} + (1 − θ_j) · ξ_j[v]`. **θ é P(não spammar)**,
  não P(voto coincide com z).
- Critério de paragem: `abs(ll − prev_ll) < tol`, **tol = 1e-06** — sobre a
  log-verossimilhança, **não** sobre `max|Δθ|`. `converged = false` diz que a
  log-lik ainda se movia, não que θ ainda se movia.
- `THETA_CLIP = (0.001, 0.999)` — limite inferior perto
  de 0. **A simetria θ ↔ 1−θ não está quebrada pelo clip.**
- Prior sobre a verdade: **uniforme** (1/L = 0,5), como o enunciado assume.
- Priors Beta(0.5, 0.5) em θ e Dirichlet(0.5) em ξ:
  o M-step é **MAP, não ML**.
- Init determinístico θ = 0.8, ξ uniforme; `n_restarts=1`, logo **nenhum**
  RNG é consumido — a run é reprodutível sem seed.

`[LIMITAÇÃO]` como o M-step é MAP, a quantidade que o EM garante não-decrescente
é a **log-posterior** (loglik + log-prior), não a loglik nua que o `_e_step`
devolve. As duas são verificadas abaixo; só a violação da **posterior** seria bug.

## 1. Convergiu?

**SIM, na iteração 462** (com `n_iter=5000`, `tol=1e-06`).
O ficheiro original parou em 200 por bater o limite, não por divergir.

Dados: 3984 itens · 887 anotadores · 23904 votos carregados.

## 2. Monotonicidade

- **log-posterior** (a que tem garantia): 273 quebras — máx 1.518e-01
- loglik nua (sem prior, sem garantia sob MAP): 275 quebras — máx 1.622e-01

**Há quebra na log-posterior — isso é bug de implementação, não interpretação.**
Primeiras 10 quebras:

- iteração 83: queda de `2.175e-02`
- iteração 84: queda de `6.427e-02`
- iteração 85: queda de `1.012e-01`
- iteração 86: queda de `1.300e-01`
- iteração 87: queda de `1.480e-01`
- iteração 88: queda de `1.518e-01`
- iteração 89: queda de `1.391e-01`
- iteração 90: queda de `1.099e-01`
- iteração 91: queda de `6.899e-02`
- iteração 92: queda de `2.497e-02`

### 2.1 O que foi excluído como causa

- **Priors** — o M-step de máxima verossimilhança pura (sem prior nenhum) também viola: `90` quebras em `461` transições, queda máx `9.063e-01`. **Priors inocentes.**
- **Clip** — θ nunca se aproxima dos limites: máximo observado `0.9647`, mínimo `0.0204`, contra `(0.001, 0.999)`. Zero anotadores no clip.
- **`EPS = 1e-12`** — o menor `(1−θ)` observado é `3.528e-02`, sete ordens de grandeza acima do EPS. Nunca entra no regime em que perturbaria os logs.
- **Label switching** — sem alternância de período 2 (secção 5).

`[MEDIDO]` **Assinatura:** as quebras concentram-se no regime de passo pequeno — `max|Δθ|` médio `1.03e-03` nas iterações com quebra contra `2.54e-02` nas restantes — e só começam na iteração 83; a fase inicial é limpa.

`[NÃO IDENTIFICADO]` a causa exacta fica por localizar. Localizá-la exige mexer
no E/M, o que está fora do âmbito desta etapa. O que está estabelecido é que
**existe** e que não é nenhuma das quatro explicações benignas acima.

## 3. Trajectória de `max|Δθ|`

| métrica | valor |
|---|---|
| primeira iteração | 6.558e-01 |
| última iteração | 1.245e-07 |
| mínimo observado | 1.245e-07 |
| máximo depois da it. 10 | 9.362e-02 |
| média das últimas 50 | 3.239e-06 |
| sd das últimas 50 | 3.944e-06 |
| gap médio par↔ímpar (últimas 50) | 3.164e-07 |

Monótona decrescente: **False**.

## 4. Clip

Intervalo: `(0.001, 0.999)`. Na última iteração: **0** anotadores no extremo inferior, **0** no superior, de 887.

## 5. Oscilação / label switching

Anotadores cujo movimento de θ troca de sinal em >60% dos passos consecutivos (it. 195–205): **4** de 887.
Amplitude máxima de oscilação no painel: `7.920e-03`.

## 6. θ convergido vs θ do ficheiro original (a pergunta que decide)

| métrica | valor |
|---|---|
| Pearson r | 0.9991 |
| Spearman ρ | 0.9991 |
| diferença média absoluta | 0.0011 |
| diferença máxima absoluta | 0.2370 |
| anotadores que cruzam θ = 0,5 | 0.0000 |

`[MEDIDO]` **O θ interrompido é praticamente o convergido** (diferença média `0.0011`, r = `0.9991`).
O `converged = false` era **cosmético**: as soft labels ponderadas e o
experimento dos dois braços **não precisam de ser refeitos por esta causa**.

Traço completo por iteração: `em_convergence_trace.csv`.

