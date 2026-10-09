"""Testes do Apêndice A do SPEC_25.md, da linha de base dos pares e do parser (§5.1).

Uso: python3 EXIST_MEMES/glm_annotator/tests/test_weights.py   (também roda com pytest)
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import parse                                          # noqa: E402
from weights import (annotator_deviation, binary_entropy, vote_weights,  # noqa: E402
                     weighted_soft_label)

ANNS = "ABCDEF"
YES_BY = {"m1": "ABCDEF", "m2": "ABCDE", "m3": "ABCD", "m4": "ABC", "m5": "BCDEF"}
GLM = {"m1": "NO", "m2": "YES", "m3": "NO", "m4": "YES", "m5": "NO"}
EXP_D = {"A": 0.0217, "B": 0.5000, "C": 0.5000, "D": 0.5473, "E": 0.3558, "F": 0.8246}
EXP_P_OLD = {"m1": 1.0, "m2": 0.8333, "m3": 0.6667, "m4": 0.5, "m5": 0.8333}
EXP_P_NEW = {"m1": 1.0, "m2": 0.9460, "m3": 0.7478, "m4": 0.6086, "m5": 0.6991}
TOL = 1e-4


def panels():
    return {m: {a: ("YES" if a in y else "NO") for a in ANNS} for m, y in YES_BY.items()}


def p_old(votes):
    return sum(v == "YES" for v in votes.values()) / len(votes)


def test_appendix_values():
    D = annotator_deviation(panels(), GLM)
    r = vote_weights(D)
    for a in ANNS:
        assert abs(D[a] - EXP_D[a]) < TOL, (a, D[a])
        assert abs(r[a] - (1 - EXP_D[a])) < TOL, (a, r[a])
    for m, votes in panels().items():
        assert abs(p_old(votes) - EXP_P_OLD[m]) < TOL, m
        assert abs(weighted_soft_label(votes, r) - EXP_P_NEW[m]) < TOL, (m, weighted_soft_label(votes, r))


def test_entropy():                                                   # propriedade 1
    assert abs(binary_entropy(0.2) - 0.7219) < TOL
    assert abs(binary_entropy(0.4) - 0.9710) < TOL
    assert binary_entropy(0.5) == 1.0 and binary_entropy(0) == 0


def test_unanimous_glm_irrelevant():                                  # propriedade 2
    assert annotator_deviation(panels(), GLM) == annotator_deviation(panels(), {**GLM, "m1": "YES"})


def test_no_included_memes():                                         # propriedade 3
    P = panels()
    r = vote_weights(annotator_deviation(P, {m: None for m in P}))
    assert all(v == 1.0 for v in r.values())
    assert all(weighted_soft_label(v, r) == p_old(v) for v in P.values())


def test_unanimous_unchanged_any_weights():                           # propriedade 4
    rng = random.Random(0)
    for _ in range(100):
        r = {a: rng.random() + 1e-6 for a in ANNS}
        assert weighted_soft_label({a: "YES" for a in ANNS}, r) == 1.0
        assert weighted_soft_label({a: "NO" for a in ANNS}, r) == 0.0


def test_equal_weights_no_change():                                   # propriedade 5
    for c in (0.37, 1.0, 0.5):
        r = {a: c for a in ANNS}
        assert all(abs(weighted_soft_label(v, r) - p_old(v)) < 1e-12 for v in panels().values())


def test_tie_can_flip():                                              # propriedade 6
    r = vote_weights(annotator_deviation(panels(), GLM))
    assert p_old(panels()["m4"]) == 0.5 and weighted_soft_label(panels()["m4"], r) > 0.5


def test_peer_baseline():
    """Conta à mão, referência = maioria dos 5 outros. F: m2 (5-0, d=1), m3 (4-1, d=1),
    m4 (3-2, d=1), m5 (4-1, d=0) → 1,3071/1,5852. A: m2 (4-1, d=0), m3 (3-2, d=0),
    m4 (2-3 → NO, d=1), m5 (5-0, d=1) → 1,0290/1,3361."""
    D = annotator_deviation(panels(), GLM, peer=True)
    assert abs(D["F"] - 1.3071 / 1.5852) < 1e-3, D["F"]
    assert abs(D["A"] - 1.0290 / 1.3361) < 1e-3, D["A"]
    assert annotator_deviation(panels(), {**GLM, "m3": None}, peer=True) != D   # mesmo filtro do GLM


def test_parser():
    cases = [
        ("<|begin_of_box|>YES<|end_of_box|>", "stop", "YES", "box"),
        ("<|begin_of_box|> **no.** <|end_of_box|>", "stop", "NO", "box"),
        ("<|begin_of_box|>YES<|end_of_box|> not NO", "stop", "YES", "box"),
        ("Yes.", "stop", "YES", "plain"),
        ("The answer is NO because it is a joke", "stop", "NO", "single_token"),
        ("<|begin_of_box|>maybe<|end_of_box|> NO", "stop", "NO", "single_token"),
        ("", "stop", "?", "empty"),
        (None, "length", "?", "empty"),
        ("Let me think about the image", "length", "?", "length"),
        ("YES and NO both apply", "stop", "?", "both"),
        ("I cannot help with that.", "stop", "?", "other"),
        ("It is NOT sexist", "stop", "?", "other"),
    ]
    for content, fr, label, method in cases:
        assert parse(content, fr) == (label, method), (content, parse(content, fr))


def run_all() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    return fails


if __name__ == "__main__":
    sys.exit(run_all())
