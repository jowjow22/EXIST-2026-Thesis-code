"""Testes do Apêndice A do SPEC_27.md (distância ao soft label do GLM), do doc 25 como caso
particular (propriedade 7) e do parser (SPEC_25 §5.1).

Uso: python3 EXIST_MEMES/glm_annotator/tests/test_weights.py   (também roda com pytest)
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import parse                                          # noqa: E402
from weights import (annotator_deviation, binary_entropy, deviation_terms, glm_soft,  # noqa: E402
                     vote_weights, weighted_soft_label)

ANNS = "ABCDEF"
YES_BY = {"m1": "ABCDEF", "m2": "ABCDE", "m3": "ABCD", "m4": "ABC", "m5": "BCDEF"}
GLM3 = {"m1": ["NO"] * 3, "m2": ["YES"] * 3, "m3": ["NO", "NO", "YES"],
        "m4": ["YES", "YES", "NO"], "m5": ["NO"] * 3}
EXP_D = dict(zip(ANNS, [0.0217, 0.5000, 0.5000, 0.5158, 0.4519, 0.8769]))
EXP_PEER = dict(zip(ANNS, [0.8118, 0.2284, 0.2284, 0.2284, 0.4067, 0.8173]))
EXP_P_OLD = {"m1": 1.0, "m2": 0.8333, "m3": 0.6667, "m4": 0.5, "m5": 0.8333}
EXP_P_NEW = {"m1": 1.0, "m2": 0.9607, "m3": 0.7858, "m4": 0.6313, "m5": 0.6878}
DOC25_P = {"m1": 0.0, "m2": 1.0, "m3": 0.0, "m4": 1.0, "m5": 0.0}       # voto duro do doc 25
DOC25_D = dict(zip(ANNS, [0.0217, 0.5000, 0.5000, 0.5473, 0.3558, 0.8246]))
TOL = 1e-4


def panels():
    return {m: {a: ("YES" if a in y else "NO") for a in ANNS} for m, y in YES_BY.items()}


def p_glm():
    return {m: glm_soft(l) for m, l in GLM3.items()}


def p_old(votes):
    return sum(v == "YES" for v in votes.values()) / len(votes)


def test_glm_soft():
    assert [glm_soft(GLM3[m]) for m in YES_BY] == [0.0, 1.0, 1 / 3, 2 / 3, 0.0]
    assert glm_soft(["YES", "?", "NO"]) is None


def test_appendix_values():
    D = annotator_deviation(panels(), p_glm())
    r = vote_weights(D)
    for a in ANNS:
        assert abs(D[a] - EXP_D[a]) < TOL, (a, D[a])
        assert abs(r[a] - (1 - EXP_D[a])) < TOL, (a, r[a])
    for m, votes in panels().items():
        assert abs(p_old(votes) - EXP_P_OLD[m]) < TOL, m
        assert abs(weighted_soft_label(votes, r) - EXP_P_NEW[m]) < TOL, (m, weighted_soft_label(votes, r))


def test_peer_baseline():
    D = annotator_deviation(panels(), p_glm(), peer=True)
    for a in ANNS:
        assert abs(D[a] - EXP_PEER[a]) < TOL, (a, D[a])


def test_entropy():                                                   # propriedade 1
    assert abs(binary_entropy(0.2) - 0.7219) < TOL
    assert abs(binary_entropy(0.4) - 0.9710) < TOL
    assert binary_entropy(0.5) == 1.0 and binary_entropy(0) == 0


def test_unanimous_glm_irrelevant():                                  # propriedade 2
    assert annotator_deviation(panels(), p_glm()) == annotator_deviation(panels(), {**p_glm(), "m1": 1.0})


def test_no_reference():                                              # propriedade 3
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
    r = vote_weights(annotator_deviation(panels(), p_glm()))
    assert p_old(panels()["m4"]) == 0.5 and abs(weighted_soft_label(panels()["m4"], r) - 0.6313) < TOL


def test_doc25_special_case():                                        # propriedade 7
    D = annotator_deviation(panels(), DOC25_P)
    for a in ANNS:
        assert abs(D[a] - DOC25_D[a]) < TOL, (a, D[a])


def test_distance_range():                                            # propriedade 8
    P = panels()
    terms = deviation_terms(P, {m: 0.5 for m in P})
    assert terms and all(d == 0.5 for *_, d in terms)
    assert all(0.0 <= d <= 1.0 for *_, d in deviation_terms(P, p_glm()))


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
