"""Estimador EM de confiabilidade de anotadores com verdade por subpopulação.

Espírito do NUTMEG (Ivey et al., EMNLP 2025): ao contrário do MACE, existe uma
verdade latente T[i, k] por (item, subpopulação) em vez de uma única por item.
Isso impede o modelo de punir discordância sistemática legítima (o núcleo do
paradigma Learning with Disagreement) como se fosse ruído.

Implementação própria por Expectation-Maximization, com updates fechados e
inspecionáveis. Variational-Bayes fica como extensão futura documentada — NÃO
implementada nesta fase, por escolha de debugabilidade.

Modelo
------
    T[i, k] ~ Uniform({0..L-1})
    voto de j (subpop k=subpop[j]) no item i:
        com prob theta[j]      -> T[i, k]
        com prob 1 - theta[j]  -> ~ Multinomial(xi[j])

E-step  : posterior de T[i, k] dados theta, xi.
M-step  : responsabilidade de spam por voto -> theta[j], xi[j].
Priors  : Beta(0.5, 0.5) em theta, Dirichlet(0.5) em xi (suaves, estabilizam).

CORRECÇÃO DA RESPONSABILIDADE (2026-08-30)
------------------------------------------
A responsabilidade de spam do M-step estava mal derivada. Marginalizando T, o
E-step do indicador s_ij é

    r = P(s=0 | V) = sum_l P(T=l | V) * P(s=0 | T=l, v)
                   = p_true * theta / (theta + (1-theta)*xi[v])          <- CORRECTA

enquanto o código calculava

    r = theta*p_true / (theta*p_true + (1-theta)*xi[v])                  <- LEGADA

A forma legada É correcta, mas só se `p_true` for o posterior LEAVE-ONE-OUT
P(T=v | votos da célula excluindo j). Alimentada com o posterior COMPLETO -- que
já contém o voto de j -- conta o voto do próprio anotador duas vezes e infla
`r`, enviesando theta para cima de forma não uniforme.

Evidência (`outputs/monotonicity_alignment_report.md`,
`outputs/mstep_responsibility_diagnosis.md`): com a forma legada a log-posterior
-- a quantidade que o MAP-EM garante não-decrescente -- quebra 273/274 vezes em
461 transições (lida no início OU no fim da iteração: são a mesma série,
max|delta| = 0 entre elas). Com a forma corrigida: ZERO quebras, nos dois pontos
de leitura e nas duas variantes de M-step.

`responsibility="fixed"` (default) usa a corrigida; `"legacy"` reproduz bit a bit
os resultados das fases anteriores. O DEFAULT DO STATICMETHOD `_m_step` é
"legacy" de propósito: os scripts de diagnóstico que replicam o loop por fora
chamam-no posicionalmente e têm de continuar a reproduzir os números publicados.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12

# Priors suaves (Jeffreys). Constantes de módulo para ficarem inspecionáveis.
THETA_PRIOR_A = 0.5
THETA_PRIOR_B = 0.5
XI_PRIOR = 0.5

THETA_INIT = 0.8  # init determinístico; ver nota de inicialização no fit()
THETA_CLIP = (1e-3, 1.0 - 1e-3)

# Folga da asserção de monotonicidade da log-posterior. Uma queda acima disto é
# erro, não ruído numérico: o EM corrigido corre 292 iterações com queda máxima 0.
MONOTONIC_SLACK = 1e-8

RESPONSIBILITY_MODES = ("fixed", "legacy")


class MonotonicityViolation(RuntimeError):
    """A log-posterior desceu entre iterações do EM.

    O EM garante não-decrescimento da log-posterior quando o M-step MAXIMIZA o
    objectivo. Uma queda grande é bug de implementação -- foi exactamente assim
    que a responsabilidade errada do M-step se manifestou sem ser vista.

    `[LIMITAÇÃO]` medida, não assumida: o M-step deste módulo é uma MÉDIA
    POSTERIOR, (sum r + a)/(n + a + b), e não o argmax. Com priors de Jeffreys
    (a = c = 0.5 < 1) o argmax MAP nem está definido -- Beta(0.5, 0.5) e
    Dirichlet(0.5) não têm modo interior -- logo a média posterior é um
    compromisso deliberado e a garantia de subida não se aplica formalmente a ela.

    Consequência prática, medida em dados reais com a responsabilidade CORRIGIDA:

        K=1 (single_truth, a configuração oficial)  ->   0 quebras
        K=8 (demo_base)                             ->  30 quebras, máx 3.8e-02
        K=8, mesmo r, M-step de ML-argmax           ->   0 quebras

    Ou seja: em K=1 a asserção é limpa e vale como guarda de regressão; em K>1 a
    responsabilidade já está certa e o que sobra é a média posterior. Nessas
    configurações passar `monotonic_check=False` e registar as quebras é o
    procedimento correcto -- é o que `run_exist` faz.
    """


def log_prior(theta, xi):
    """log-prior a menos de constantes de normalização (não afectam deltas).

    Beta(a, b) em cada theta_j  -> (a-1) log th + (b-1) log(1-th)
    Dirichlet(c) em cada linha  -> (c-1) sum_l log xi_jl
    """
    lp_theta = ((THETA_PRIOR_A - 1.0) * np.log(theta)
                + (THETA_PRIOR_B - 1.0) * np.log(1.0 - theta)).sum()
    lp_xi = ((XI_PRIOR - 1.0) * np.log(xi)).sum()
    return float(lp_theta + lp_xi)


class NutmegEM:
    """EM para theta (competência) por anotador, com verdade por subpopulação.

    Atributos pós-`fit`
    -------------------
    theta_        : ndarray (M,)          competência estimada (prob. de NÃO spammar)
    xi_           : ndarray (M, L)        distribuição de ruído por anotador
    posterior_T_  : ndarray (N, K, L)     posterior da verdade por (item, subpop)
    loglik_history_ : list[float]         log-verossimilhança por iteração
    n_votes_      : ndarray (M,)          votos observados por anotador
    cell_has_votes_ : ndarray (N, K) bool células (item, subpop) com >=1 voto
    n_iter_run_   : int
    converged_    : bool
    """

    def __init__(self, n_labels: int = 2, responsibility: str = "fixed",
                 monotonic_check=None):
        # n_labels > 2 já é suportado pela formulação (gancho para a Task B,
        # single-label multiclasse). Task C (multilabel) exigiria um modelo
        # diferente — gancho deixado por escrever, de propósito.
        self.n_labels = int(n_labels)
        if responsibility not in RESPONSIBILITY_MODES:
            raise ValueError(f"responsibility deve ser um de {RESPONSIBILITY_MODES}")
        self.responsibility = responsibility
        # None = automático: verificar em "fixed", não verificar em "legacy".
        # Verificar em "legacy" faria o estimador antigo rebentar sempre, e o
        # ponto de manter o legado é reproduzir as fases anteriores, não corrigi-las.
        self.monotonic_check = (responsibility == "fixed") if monotonic_check is None \
            else bool(monotonic_check)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        annotations,
        subpop,
        n_iter: int = 5000,
        tol: float = 1e-6,
        verbose: bool = False,
        n_restarts: int = 1,
        restart_seed: int = 0,
    ):
        """Ajusta o modelo.

        Parameters
        ----------
        annotations : array-like (n_votes, 3) -> (item_id, annotator_id, label).
            item_id em [0, N), annotator_id em [0, M), label em [0, L).
        subpop : array-like (M,) -> índice de subpopulação em [0, K).
            Passar `np.zeros(M)` reproduz o baseline single-truth (tipo-MACE):
            é o MESMO caminho de código, K=1.
        n_iter : máximo de iterações EM. Default 5000 desde 2026-08-30 — o antigo
            200 era o que produzia `converged=false` em single_truth_K1, que
            parava por bater no limite e não por divergir (o legado converge na
            iteração 462, o corrigido na 292).
        tol : critério de paragem em |delta log-lik|.
        verbose : imprime log-lik e min/max/mean(theta) por iteração.
        n_restarts : >1 perturba o init de theta e fica com o melhor log-lik.
            Default 1 — o init determinístico abaixo já é estável.

        Returns
        -------
        self
        """
        ann = np.asarray(annotations, dtype=int)
        assert ann.ndim == 2 and ann.shape[1] == 3, f"annotations shape {ann.shape}"
        subpop = np.asarray(subpop, dtype=int)

        items, anns, labels = ann[:, 0], ann[:, 1], ann[:, 2]
        M = int(subpop.shape[0])
        N = int(items.max()) + 1 if len(items) else 0
        K = int(subpop.max()) + 1 if M else 0
        L = self.n_labels
        assert anns.max(initial=-1) < M, "annotator_id fora do range de subpop"
        assert labels.max(initial=-1) < L, "label fora do range de n_labels"

        cells = subpop[anns]  # subpop de cada voto, shape (n_votes,)
        n_votes = np.bincount(anns, minlength=M)

        cell_has_votes = np.zeros((N, K), dtype=bool)
        cell_has_votes[items, cells] = True

        if verbose:
            print(
                f"[NutmegEM] N={N} M={M} K={K} L={L} n_votes={len(ann)} | "
                f"células com votos={int(cell_has_votes.sum())}/{N * K} "
                f"({100.0 * cell_has_votes.mean():.1f}%) | "
                f"anotadores com 0 votos={int((n_votes == 0).sum())}"
            )

        best = None
        rng = np.random.default_rng(restart_seed)
        for r in range(max(1, n_restarts)):
            # --- INIT determinístico (r=0) -------------------------------
            # theta=0.8 + xi uniforme faz o primeiro E-step colapsar no voto
            # maioritário suavizado por (item, subpop) -- que é exatamente o
            # init recomendado. Init aleatório converge para soluções
            # degeneradas ("toda a gente spamma, xi explica tudo") ou inverte
            # T nos itens divisivos.
            theta = np.full(M, THETA_INIT)
            if r > 0:
                theta = np.clip(theta + rng.normal(0, 0.1, size=M), *THETA_CLIP)
            xi = np.full((M, L), 1.0 / L)

            loglik_history = []
            logpost_history = []
            prev_ll = -np.inf
            prev_lp = -np.inf
            converged = False
            it = 0
            posterior = None
            for it in range(1, n_iter + 1):
                posterior, ll = self._e_step(
                    items, anns, cells, labels, theta, xi, N, K, L, cell_has_votes
                )
                # log-posterior avaliada no MESMO estado que a log-lik: os
                # parâmetros de ENTRADA do E-step. Misturar theta de entrada com
                # xi de saída daria um objectivo mal formado e inventaria quebras.
                lp = ll + log_prior(theta, xi)
                if self.monotonic_check and lp < prev_lp - MONOTONIC_SLACK:
                    raise MonotonicityViolation(
                        f"log-posterior desceu na iteração {it} do restart {r}: "
                        f"{prev_lp:.10f} -> {lp:.10f} (queda {prev_lp - lp:.3e} > "
                        f"folga {MONOTONIC_SLACK:.1e}). O MAP-EM garante "
                        f"não-decrescimento — isto é bug de implementação, não "
                        f"ruído numérico. responsibility={self.responsibility!r}."
                    )
                prev_lp = lp
                theta, xi = self._m_step(
                    items, anns, cells, labels, theta, xi, posterior, M, L, n_votes,
                    responsibility=self.responsibility,
                )
                loglik_history.append(float(ll))
                logpost_history.append(float(lp))
                if verbose:
                    print(
                        f"  it {it:3d} | loglik={ll:14.4f} | theta "
                        f"min={theta.min():.3f} mean={theta.mean():.3f} "
                        f"max={theta.max():.3f}"
                    )
                if abs(ll - prev_ll) < tol:
                    converged = True
                    break
                prev_ll = ll

            # posterior final coerente com o theta/xi finais
            posterior, ll = self._e_step(
                items, anns, cells, labels, theta, xi, N, K, L, cell_has_votes
            )
            if best is None or ll > best[0]:
                best = (ll, theta, xi, posterior, loglik_history, logpost_history,
                        converged, it)

        ll, theta, xi, posterior, loglik_history, logpost_history, converged, it = best
        assert theta.shape == (M,), theta.shape
        assert xi.shape == (M, L), xi.shape
        assert posterior.shape == (N, K, L), posterior.shape
        assert np.isfinite(theta).all(), "theta contém NaN/inf"

        self.theta_ = theta
        self.xi_ = xi
        self.posterior_T_ = posterior
        self.loglik_history_ = loglik_history
        self.logpost_history_ = logpost_history
        self.responsibility_ = self.responsibility
        self.monotonic_checked_ = bool(self.monotonic_check)
        self.final_loglik_ = float(ll)
        self.n_votes_ = n_votes
        self.cell_has_votes_ = cell_has_votes
        self.n_iter_run_ = it
        self.converged_ = converged
        return self

    # --------------------------------------------------------------- E-step
    @staticmethod
    def _e_step(items, anns, cells, labels, theta, xi, N, K, L, cell_has_votes):
        """Posterior de T[i, k] dados theta e xi. Devolve (posterior (N,K,L), loglik).

        Para cada voto v de j: P(v | T=l) = theta_j * 1{v==l} + (1-theta_j) * xi_j[v].
        Acumulado em log-espaço; células (i,k) sem votos ficam no prior uniforme
        e contribuem 0 para a log-lik.
        """
        th = theta[anns]                       # (n_votes,)
        xv = xi[anns, labels]                  # (n_votes,)
        base = np.log((1.0 - th) * xv + EPS)   # contribuição para l != voto
        agree = np.log(th + (1.0 - th) * xv + EPS)  # contribuição para l == voto

        contrib = np.repeat(base[:, None], L, axis=1)      # (n_votes, L)
        contrib[np.arange(len(labels)), labels] = agree

        logp = np.zeros((N, K, L))             # prior uniforme -> constante, omitida
        np.add.at(logp, (items, cells), contrib)

        mx = logp.max(axis=2, keepdims=True)
        exp = np.exp(logp - mx)
        s = exp.sum(axis=2, keepdims=True)
        posterior = exp / (s + EPS)
        # log P(votos da célula) = log sum_l exp(logp) - log L (prior uniforme)
        cell_ll = (mx[..., 0] + np.log(s[..., 0] + EPS)) - np.log(L)
        loglik = float(cell_ll[cell_has_votes].sum())
        return posterior, loglik

    # --------------------------------------------------------------- M-step
    @staticmethod
    def _m_step(items, anns, cells, labels, theta, xi, posterior, M, L, n_votes,
                responsibility: str = "legacy"):
        """Reestima theta e xi a partir das responsabilidades de spam por voto.

        responsibility="fixed"  (a derivada; usada pelo `fit()` por default):
            r_notspam = P(T=voto) * theta_j / (theta_j + (1-theta_j) * xi_j[voto])
        responsibility="legacy" (a das fases anteriores; DEFAULT DESTE MÉTODO):
            r_notspam = theta_j * P(T=voto) / (theta_j * P(T=voto) + (1-theta_j) * xi_j[voto])

        A forma legada só é correcta se `P(T=voto)` for o posterior LEAVE-ONE-OUT;
        alimentada com o posterior completo conta o voto do próprio anotador duas
        vezes. Ver o cabeçalho do módulo.

        O default deste staticmethod é "legacy" DE PROPÓSITO: os scripts de
        diagnóstico chamam-no posicionalmente para replicar o loop antigo e têm
        de continuar a reproduzir os números já reportados. O `fit()` passa
        explicitamente o modo da instância, cujo default é "fixed".

        theta_j   = (sum r_notspam + a) / (n_j + a + b)              [Beta(a,b)]
        xi_j[v]   = (sum_{voto=v} r_spam + c) / (sum r_spam + c*L)   [Dirichlet(c)]
        """
        if responsibility not in RESPONSIBILITY_MODES:
            raise ValueError(f"responsibility deve ser um de {RESPONSIBILITY_MODES}")
        p_true = posterior[items, cells, labels]   # (n_votes,) P(T[i,k] == voto)
        th = theta[anns]
        xv = xi[anns, labels]
        if responsibility == "fixed":
            r_notspam = p_true * th / (th + (1.0 - th) * xv + EPS)
        else:
            num = th * p_true
            r_notspam = num / (num + (1.0 - th) * xv + EPS)
        r_spam = 1.0 - r_notspam

        sum_notspam = np.bincount(anns, weights=r_notspam, minlength=M)
        new_theta = (sum_notspam + THETA_PRIOR_A) / (
            n_votes + THETA_PRIOR_A + THETA_PRIOR_B
        )
        # anotador com 0 votos -> media do prior (0.5), sem NaN: a fórmula já dá isso.
        new_theta = np.clip(new_theta, *THETA_CLIP)

        flat = anns * L + labels
        spam_counts = np.bincount(flat, weights=r_spam, minlength=M * L).reshape(M, L)
        new_xi = (spam_counts + XI_PRIOR) / (
            spam_counts.sum(axis=1, keepdims=True) + XI_PRIOR * L
        )
        assert new_theta.shape == (M,) and new_xi.shape == (M, L)
        return new_theta, new_xi
