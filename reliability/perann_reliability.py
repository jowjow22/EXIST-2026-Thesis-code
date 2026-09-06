"""Estimador de confiabilidade POR-ANOTADOR com desconto de dificuldade do item.

Motivação (ver reliability/outputs/divisiveness_report.md e stage1b_report.md):
o `NutmegEM` estima uma verdade por célula (item, subpopulação). No regime do
EXIST isso dá 1–2 votos por célula — fragmentação. Aqui a célula desaparece: a
competência de cada anotador é estimada agregando TODOS os ~27 votos dele.

O risco de um estimador por-anotador ingénuo é ser MACE puro ("concordância com a
maioria"), que pune minorias consistentes e rompe o LeWiDi. A defesa é o DESCONTO
POR DIFICULDADE: discordar num item cujo pool está genuinamente dividido quase não
penaliza; discordar num item de consenso claro penaliza a sério.

    difficulty[i] = H(p_i) / log 2,   p_i = fração de YES no pool do item i
    peso_i        = 1 - difficulty[i]
    competência[j] = Σ_i peso_i · [voto_j(i) == maioria(i)] / Σ_i peso_i

PROPRIEDADES DE DESENHO
-----------------------
* DETERMINÍSTICO e NÃO-ITERATIVO: `difficulty` sai dos votos CRUS, sem ponderar por
  competência. É isto que quebra a circularidade e dispensa EM/convergência.
* Sem RNG, sem I/O. Função pura sobre a matriz de anotações.
* `discount=False` reproduz a linha de base MACE-ingénua no MESMO caminho de código
  (mesma decisão que o baseline K=1 do `NutmegEM`: comparação apples-to-apples).

VARIANTE `leave_one_out`
------------------------
Na forma literal da especificação (`leave_one_out=False`, o default), o voto do
próprio anotador entra no cálculo da dificuldade do item que o vai julgar. Um
dissidente solitário num item de consenso INFLA a dificuldade desse item com o
próprio voto e assim reduz a própria penalização (com 6 votos, 1 dissidente ->
p=1/6 -> difficulty=0,65 -> peso 0,35 em vez de 1,0).
Com `leave_one_out=True`, a dificuldade e a maioria vistas pelo anotador j no item i
são calculadas SEM o voto de j. Isso remove a auto-inflação, mas também remove
proteção a uma minoria que aparece pouco no pool de cada item.
As duas variantes correm o mesmo portão sintético e são reportadas lado a lado —
não se substitui a especificação em silêncio.

REGRA DE DESEMPATE
------------------
Empate no pool (3-3) => p=0,5 => difficulty=1 => peso 0. O voto não contribui de
qualquer forma, e a maioria fica indefinida. Para que `discount=False` (que não tem
o peso a anular o empate) seja consistente com isto, votos em itens empatados são
EXCLUÍDOS do numerador E do denominador nas duas variantes. Idem para pools LOO
pares e para pools LOO vazios (itens com 1 voto).

GANCHOS (documentados, NÃO implementados nesta fase)
-----------------------------------------------------
* Versão iterativa: ponderar a dificuldade pela competência já estimada
  (difficulty ponderada -> nova competência -> ...). Reintroduz a circularidade e
  exige iteração e critério de convergência. Extensão futura.
* Task B (3 classes): a entropia generaliza para H(p)/log(L) sobre o vetor de
  contagens; a "maioria" passa a argmax com desempate documentado.
* Task C (multilabel): não é uma distribuição sobre classes — exige dificuldade por
  rótulo e uma regra de agregação. Modelo diferente, não é generalização direta.
"""

from __future__ import annotations

import numpy as np

LOW_CONFIDENCE_MIN_VOTES = 15   # abaixo disto a competência é ruidosa (11 anotadores no EXIST)
NEUTRAL_COMPETENCE = 0.5        # valor para massa de peso ~0 ("indeterminada por dificuldade")
WEIGHT_MASS_EPS = 1e-9


def item_difficulty(annotations, n_items=None):
    """Dificuldade por item a partir dos votos CRUS (entropia binária normalizada).

    Returns
    -------
    difficulty : (N,) em [0, 1]; NaN em itens sem votos.
    n0, n1     : (N,) contagens de NO/YES por item.
    """
    items = annotations[:, 0]
    labels = annotations[:, 2]
    N = int(items.max()) + 1 if n_items is None else n_items
    n1 = np.bincount(items[labels == 1], minlength=N).astype(float)
    n0 = np.bincount(items[labels == 0], minlength=N).astype(float)
    tot = n0 + n1
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, n1 / np.maximum(tot, 1), np.nan)
    return _binary_entropy(p), n0, n1


def _binary_entropy(p):
    """H(p)/log 2, com H(0)=H(1)=0. Propaga NaN."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    ok = np.isfinite(p) & (p > 0) & (p < 1)
    pk = p[ok]
    out[ok] = -(pk * np.log(pk) + (1 - pk) * np.log(1 - pk)) / np.log(2.0)
    out[~np.isfinite(p)] = np.nan
    return out


def compute_competence(
    annotations,
    n_annotators=None,
    n_items=None,
    discount=True,
    leave_one_out=False,
    verbose=False,
):
    """Competência por anotador, determinística e não-iterativa.

    Parameters
    ----------
    annotations : (n_votes, 3) -> (item_idx, annotator_idx, label 0/1)
    discount : True = pesa cada item por (1 - difficulty). False = concordância-com-
        maioria simples (linha de base MACE-ingénua), mesmo caminho de código.
    leave_one_out : True = dificuldade e maioria vistas por j excluem o voto de j.

    Returns
    -------
    dict com:
      competence (M,), n_votes (M,), weight_mass (M,), agree_mass (M,),
      indeterminate (M,) bool, low_confidence (M,) bool, n_used (M,) votos que
      contribuíram (não empatados), difficulty (N,) [sempre a dificuldade CRUA do
      item, para reporte], config, e diagnósticos agregados.
    """
    items = annotations[:, 0].astype(int)
    anns = annotations[:, 1].astype(int)
    labels = annotations[:, 2].astype(int)
    N = int(items.max()) + 1 if n_items is None else n_items
    M = int(anns.max()) + 1 if n_annotators is None else n_annotators

    difficulty_raw, n0, n1 = item_difficulty(annotations, n_items=N)

    # --- pool visto por cada voto -------------------------------------------
    if leave_one_out:
        c1 = n1[items] - (labels == 1)
        c0 = n0[items] - (labels == 0)
    else:
        c1 = n1[items]
        c0 = n0[items]
    tot = c0 + c1
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, c1 / np.maximum(tot, 1), np.nan)
    diff_v = _binary_entropy(p)                   # dificuldade vista por este voto

    # --- maioria e empates ---------------------------------------------------
    tie = (c0 == c1) | (tot <= 0)                 # empate ou pool vazio -> excluído
    majority = (c1 > c0).astype(int)
    agree = (labels == majority).astype(float)

    weight = (1.0 - diff_v) if discount else np.ones_like(diff_v)
    weight = np.where(tie, 0.0, weight)
    weight = np.nan_to_num(weight, nan=0.0)
    contrib = weight * agree

    weight_mass = np.bincount(anns, weights=weight, minlength=M)
    agree_mass = np.bincount(anns, weights=contrib, minlength=M)
    n_votes = np.bincount(anns, minlength=M)
    n_used = np.bincount(anns[~tie], minlength=M)

    indeterminate = weight_mass <= WEIGHT_MASS_EPS
    competence = np.full(M, NEUTRAL_COMPETENCE, dtype=float)
    ok = ~indeterminate
    competence[ok] = agree_mass[ok] / weight_mass[ok]
    low_conf = n_votes < LOW_CONFIDENCE_MIN_VOTES

    out = {
        "competence": competence,
        "n_votes": n_votes,
        "n_used": n_used,
        "weight_mass": weight_mass,
        "agree_mass": agree_mass,
        "indeterminate": indeterminate,
        "low_confidence": low_conf,
        "difficulty": difficulty_raw,
        "config": {"discount": bool(discount), "leave_one_out": bool(leave_one_out)},
        "n_tied_votes": int(tie.sum()),
        "n_items": N,
        "n_annotators": M,
    }
    if verbose:
        fin = difficulty_raw[np.isfinite(difficulty_raw)]
        print(
            f"[perann] discount={discount} loo={leave_one_out} | "
            f"shapes: competence={competence.shape} difficulty={difficulty_raw.shape} "
            f"votos={len(annotations)}"
        )
        print(
            f"[perann] difficulty(crua): mean={fin.mean():.3f} "
            f"min={fin.min():.3f} max={fin.max():.3f} | "
            f"fáceis(<0.2)={int((fin < 0.2).sum())} difíceis(>0.8)={int((fin > 0.8).sum())}"
            f" de {len(fin)}"
        )
        print(
            f"[perann] competência: mean={competence.mean():.3f} "
            f"min={competence.min():.3f} max={competence.max():.3f} | "
            f"indeterminadas={int(indeterminate.sum())} "
            f"baixa-confiança(<{LOW_CONFIDENCE_MIN_VOTES} votos)={int(low_conf.sum())} | "
            f"votos empatados excluídos={out['n_tied_votes']}"
        )
        print(
            f"[perann] massa de peso/anotador: mean={weight_mass.mean():.2f} "
            f"min={weight_mass.min():.2f} max={weight_mass.max():.2f}"
        )
    return out
