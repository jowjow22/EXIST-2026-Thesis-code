"""Gerador de anotações sintéticas com competência (theta) conhecida por construção.

Modelo generativo (o mesmo que `nutmeg_em.NutmegEM` inverte):

    Por item i e subpopulação k: verdade T[i, k] in {0, 1}.
    Anotador j pertence à subpop subpop[j].
    Para cada voto de j no item i:
        S ~ Bernoulli(1 - theta[j])          # S=1 => spam
        S=0 -> vota T[i, subpop[j]]          # a verdade da SUA subpop
        S=1 -> vota uniforme em {0, ..., L-1}

Como theta é sorteado antes de gerar os votos, ele é a verdade contra a qual a
recuperação do EM é medida (ver `validate.evaluate_recovery`).
"""

from __future__ import annotations

import numpy as np

MIN_VOTES_PER_ANNOTATOR = 15  # abaixo disto o EM não tem repetição de ID suficiente


def generate_synthetic(
    n_items: int = 800,
    n_annotators: int = 60,
    minority_frac: float = 0.25,
    divisiveness_rate: float = 0.3,
    mean_spam_rate: float = 0.3,
    annotations_per_item: int = 6,
    seed: int = 42,
    n_labels: int = 2,
    theta_conc: float = 8.0,
    minority_theta: float | None = None,
    uniform_spammer_frac: float = 0.0,
    uniform_spammer_theta: float = 0.25,
    biased_spammer_frac: float = 0.0,
    biased_spammer_label: int = 1,
    biased_spammer_theta: float = 0.25,
    verbose: bool = False,
):
    """Gera um dataset sintético de anotações.

    Parameters
    ----------
    n_items : nº de itens.
    n_annotators : nº de anotadores (M).
    minority_frac : fração de anotadores atribuídos à subpop 1 (minoria).
    divisiveness_rate : fração de itens em que T[i, 1] != T[i, 0].
    mean_spam_rate : média de (1 - theta); theta ~ Beta com esta média.
    annotations_per_item : anotadores distintos por item.
    seed : semente do RNG (determinismo total).
    n_labels : L (2 = Task A binária).
    theta_conc : concentração da Beta (maior = theta menos disperso).
    minority_theta : se dado, força theta[j] = este valor para toda a minoria.
        Usado pelo teste de preservação de minoria (minoria consistente mas
        discordante da maioria).
    uniform_spammer_frac : fração de anotadores que são SPAMMERS UNIFORMES explícitos
        (theta forçado a `uniform_spammer_theta`, voto aleatório quando spammam).
        Sem isto o grupo de "spam uniforme" teria de ser lido da cauda da Beta, que
        nesta parametrização tem 0–4 anotadores — amostra pequena demais para
        comparar populações.
    uniform_spammer_theta : theta verdadeiro desses anotadores.
    biased_spammer_frac : fração de anotadores que são SPAMMERS DE VIÉS CONSTANTE.
        Quando spammam (prob 1-theta) NÃO votam uniforme: votam sempre
        `biased_spammer_label`. theta real deles é forçado a `biased_spammer_theta`
        (baixo: são ruído). São sorteados só de entre a MAIORIA (subpop 0), para
        não se confundirem com a minoria legítima do outro teste.
    biased_spammer_label : classe fixa que o spammer enviesado repete (default 1 = YES).
    biased_spammer_theta : theta verdadeiro atribuído a esses anotadores.
    verbose : imprime shapes e estatísticas de votos/anotador.

    Returns
    -------
    annotations : ndarray[int] shape (n_votes, 3) -> (item_id, annotator_id, label)
    true_theta  : ndarray[float] shape (M,)
    subpop      : ndarray[int] shape (M,), valores em {0, 1}
    T           : ndarray[int] shape (n_items, 2)
    groups : dict com máscaras bool (M,): "biased_spammer", "uniform_spammer"
    """
    if annotations_per_item > n_annotators:
        raise ValueError("annotations_per_item > n_annotators")
    rng = np.random.default_rng(seed)

    # --- subpopulações -----------------------------------------------------
    n_minority = max(1, int(round(minority_frac * n_annotators)))
    subpop = np.zeros(n_annotators, dtype=int)
    minority_ids = rng.choice(n_annotators, size=n_minority, replace=False)
    subpop[minority_ids] = 1

    # --- competência verdadeira -------------------------------------------
    mean_theta = 1.0 - mean_spam_rate
    a = mean_theta * theta_conc
    b = (1.0 - mean_theta) * theta_conc
    true_theta = rng.beta(a, b, size=n_annotators)
    true_theta = np.clip(true_theta, 0.02, 0.98)
    if minority_theta is not None:
        true_theta[subpop == 1] = minority_theta

    # --- spammers explícitos (subgrupos da MAIORIA, disjuntos entre si) -----
    is_biased_spammer = np.zeros(n_annotators, dtype=bool)
    is_uniform_spammer = np.zeros(n_annotators, dtype=bool)
    if uniform_spammer_frac > 0:
        pool = np.flatnonzero(subpop == 0)
        n_uni = min(len(pool), max(1, int(round(uniform_spammer_frac * n_annotators))))
        chosen = rng.choice(pool, size=n_uni, replace=False)
        is_uniform_spammer[chosen] = True
        true_theta[chosen] = uniform_spammer_theta
    if biased_spammer_frac > 0:
        pool = np.flatnonzero((subpop == 0) & ~is_uniform_spammer)
        n_bias = min(len(pool), max(1, int(round(biased_spammer_frac * n_annotators))))
        chosen = rng.choice(pool, size=n_bias, replace=False)
        is_biased_spammer[chosen] = True
        true_theta[chosen] = biased_spammer_theta

    # --- verdades por subpopulação ----------------------------------------
    T = np.zeros((n_items, 2), dtype=int)
    T[:, 0] = rng.integers(0, n_labels, size=n_items)
    divisive = rng.random(n_items) < divisiveness_rate
    T[:, 1] = np.where(divisive, (T[:, 0] + 1) % n_labels, T[:, 0])

    # --- votos -------------------------------------------------------------
    rows = []
    for i in range(n_items):
        voters = rng.choice(n_annotators, size=annotations_per_item, replace=False)
        for j in voters:
            if rng.random() < true_theta[j]:
                label = T[i, subpop[j]]
            elif is_biased_spammer[j]:
                label = biased_spammer_label  # spam CONSTANTE, não uniforme
            else:
                label = rng.integers(0, n_labels)
            rows.append((i, int(j), int(label)))
    annotations = np.asarray(rows, dtype=int)

    # --- guarda de votos por anotador --------------------------------------
    counts = np.bincount(annotations[:, 1], minlength=n_annotators)
    if verbose or counts.min() < MIN_VOTES_PER_ANNOTATOR:
        msg = (
            f"[synthetic] votos/anotador: mean={counts.mean():.1f} "
            f"min={counts.min()} max={counts.max()} (M={n_annotators}, "
            f"n_votes={len(annotations)})"
        )
        if counts.min() < MIN_VOTES_PER_ANNOTATOR:
            msg += f"  ** AVISO: min < {MIN_VOTES_PER_ANNOTATOR} **"
        print(msg)
    if verbose:
        print(
            f"[synthetic] shapes: annotations={annotations.shape} "
            f"true_theta={true_theta.shape} subpop={subpop.shape} T={T.shape} | "
            f"itens divisivos={int(divisive.sum())}/{n_items} | "
            f"minoria={n_minority}/{n_annotators} | "
            f"spam uniforme={int(is_uniform_spammer.sum())} "
            f"spam enviesado={int(is_biased_spammer.sum())}"
        )
    return annotations, true_theta, subpop, T, {
        "biased_spammer": is_biased_spammer,
        "uniform_spammer": is_uniform_spammer,
    }


# ===========================================================================
# Gerador com DIFICULDADE DE ITEM controlada (para o estimador por-anotador).
# Função NOVA e independente de `generate_synthetic` — a assinatura daquela já
# quebrou call sites uma vez; esta não lhe toca.
# ===========================================================================

def generate_difficulty_synthetic(
    n_items: int = 800,
    n_annotators: int = 60,
    annotations_per_item: int = 6,
    hard_frac: float = 0.4,
    hard_q: float = 0.5,
    mean_spam_rate: float = 0.3,
    theta_conc: float = 8.0,
    seed: int = 42,
    minority_frac: float = 0.0,
    minority_theta: float = 0.95,
    minority_mode: str | None = None,       # None | "hard" | "easy"
    disagree_frac: float = 0.30,
    biased_spammer_frac: float = 0.0,
    biased_spammer_label: int = 1,
    biased_spammer_theta: float = 0.25,
    verbose: bool = False,
):
    """Anotações sintéticas com dificuldade de item CONTROLADA e theta conhecido.

    Modelo de convicção (separado da competência, ao contrário de `generate_synthetic`,
    onde toda a discordância vinha de spam):

        item FÁCIL   -> existe verdade T[i]; a convicção de qualquer anotador é T[i]
                        => o pool concorda e difficulty ≈ 0.
        item DIFÍCIL -> ambiguidade GENUÍNA; a convicção de cada anotador é
                        Bernoulli(hard_q) independente => o pool divide-se e
                        difficulty ≈ 1, mesmo com todos os anotadores competentes.

        voto = convicção            com prob theta[j]
             = ruído                com prob 1 - theta[j]
               (uniforme, ou `biased_spammer_label` fixo se j é spammer enviesado)

    Braços de minoria (`minority_mode`), desenhados MATCHED — mesma fração de
    minoria, mesmo `minority_theta` e o MESMO número de itens de discordância;
    a ÚNICA diferença é o terreno onde a discordância cai:

        "hard" -> a minoria discorda em itens GENUINAMENTE ambíguos: nesses itens
                  a convicção dela é fixa em 1 (não é Bernoulli). É perspetiva
                  legítima em terreno dividido; o desconto DEVE protegê-la.
        "easy" -> a minoria discorda em itens de CONSENSO: convicção = 1 - T[i].
                  É ruído em terreno claro; o desconto NÃO a deve proteger.

    Returns
    -------
    annotations : (n_votes, 3) -> (item, anotador, label)
    true_theta  : (M,)
    is_hard     : (N,) bool — item genuinamente ambíguo por construção
    groups      : dict de máscaras (M,): "minority", "biased_spammer", "reliable_majority"
    info        : dict (T, itens de discordância, contagens)
    """
    if annotations_per_item > n_annotators:
        raise ValueError("annotations_per_item > n_annotators")
    if minority_mode not in (None, "hard", "easy"):
        raise ValueError("minority_mode tem de ser None, 'hard' ou 'easy'")
    rng = np.random.default_rng(seed)

    # --- itens ---------------------------------------------------------------
    is_hard = rng.random(n_items) < hard_frac
    T = rng.integers(0, 2, size=n_items)          # verdade dos itens fáceis

    # --- competência ---------------------------------------------------------
    mean_theta = 1.0 - mean_spam_rate
    true_theta = rng.beta(mean_theta * theta_conc,
                          (1.0 - mean_theta) * theta_conc, size=n_annotators)
    true_theta = np.clip(true_theta, 0.02, 0.98)

    # --- minoria -------------------------------------------------------------
    is_minority = np.zeros(n_annotators, dtype=bool)
    if minority_frac > 0 and minority_mode is not None:
        n_min = max(1, int(round(minority_frac * n_annotators)))
        is_minority[rng.choice(n_annotators, size=n_min, replace=False)] = True
        true_theta[is_minority] = minority_theta

    # --- spammers de viés constante (nunca da minoria) -----------------------
    is_biased = np.zeros(n_annotators, dtype=bool)
    if biased_spammer_frac > 0:
        pool = np.flatnonzero(~is_minority)
        n_b = min(len(pool), max(1, int(round(biased_spammer_frac * n_annotators))))
        is_biased[rng.choice(pool, size=n_b, replace=False)] = True
        true_theta[is_biased] = biased_spammer_theta

    # --- itens de discordância da minoria (MATCHED entre braços) -------------
    n_dis = int(round(disagree_frac * n_items))
    if minority_mode == "hard":
        pool_items = np.flatnonzero(is_hard)
    elif minority_mode == "easy":
        pool_items = np.flatnonzero(~is_hard)
    else:
        pool_items = np.empty(0, dtype=int)
    n_dis = min(n_dis, len(pool_items))
    disagree_items = np.sort(rng.choice(pool_items, size=n_dis, replace=False)) \
        if n_dis > 0 else np.empty(0, dtype=int)
    is_disagree = np.zeros(n_items, dtype=bool)
    is_disagree[disagree_items] = True

    # --- votos ---------------------------------------------------------------
    rows = []
    for i in range(n_items):
        voters = rng.choice(n_annotators, size=annotations_per_item, replace=False)
        for j in voters:
            # convicção
            if is_minority[j] and is_disagree[i]:
                conviction = 1 if is_hard[i] else int(1 - T[i])
            elif is_hard[i]:
                conviction = int(rng.random() < hard_q)
            else:
                conviction = int(T[i])
            # voto
            if rng.random() < true_theta[j]:
                label = conviction
            elif is_biased[j]:
                label = biased_spammer_label
            else:
                label = int(rng.integers(0, 2))
            rows.append((i, int(j), label))
    annotations = np.asarray(rows, dtype=int)

    counts = np.bincount(annotations[:, 1], minlength=n_annotators)
    groups = {
        "minority": is_minority,
        "biased_spammer": is_biased,
        "reliable_majority": (~is_minority) & (~is_biased) & (true_theta >= 0.70),
    }
    info = {
        "T": T,
        "is_disagree_item": is_disagree,
        "n_hard": int(is_hard.sum()),
        "n_disagree_items": int(n_dis),
        "votes_per_annotator": counts,
    }
    if verbose or counts.min() < MIN_VOTES_PER_ANNOTATOR:
        msg = (f"[synth-diff] votos/anotador: mean={counts.mean():.1f} "
               f"min={counts.min()} max={counts.max()} (M={n_annotators}, "
               f"n_votes={len(annotations)})")
        if counts.min() < MIN_VOTES_PER_ANNOTATOR:
            msg += f"  ** AVISO: min < {MIN_VOTES_PER_ANNOTATOR} **"
        print(msg)
    if verbose:
        print(f"[synth-diff] itens difíceis={info['n_hard']}/{n_items} | "
              f"itens de discordância da minoria={n_dis} (modo={minority_mode}) | "
              f"minoria={int(is_minority.sum())} spam enviesado={int(is_biased.sum())} | "
              f"maioria fiável={int(groups['reliable_majority'].sum())}")
    return annotations, true_theta, is_hard, groups, info
