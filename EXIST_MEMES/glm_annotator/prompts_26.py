"""Doc 26 — fatorial 2×2 a partir do P1 (SPEC_26.md).

Fator ORDEM:        texto do meme depois da instrução (P1) | antes da instrução
Fator ENQUADRAMENTO: pergunta direta (P1) | "Task: decide whether ..."
Todo o resto do texto do P1 fica idêntico. A célula direta+depois É o P1 (assert).
"""
from prompts import P1, sha256

QUESTION = ("Is the meme sexist, in any form, or does it describe situations in which such "
            "discrimination occurs (i.e., it is sexist itself, describes a sexist situation or "
            "criticizes a sexist behavior)?")
TASK = ("Task: decide whether the meme is sexist, in any form, or whether it describes situations "
        "in which such discrimination occurs (i.e., it is sexist itself, describes a sexist "
        "situation or criticizes a sexist behavior).")
BODY = """YES: the meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.
NO: the meme does not prejudice, underestimate or discriminate women.
A meme that condemns or reports sexism is still YES.

Remember that sexism may be expressed in different forms! It may sound friendly, funny, offensive, hateful or even violent. Sexism comprises any form of oppression or prejudice against women because of their sex. This discrimination can be based on different beliefs such as stereotypes, the belief that men are superior to women or irrational hatred of women (misogyny). Note that misogyny is a radical/hate-driven sexism."""
JUDGE = "Judge the image and the text together."
TEXT = "Text extracted from the meme (English or Spanish, do not translate it):\n{text}"
END = "Answer with one word: YES or NO.\n"


def build(opening: str, text_first: bool) -> str:
    if text_first:
        return f"{TEXT}\n\n{opening}\n\n{BODY}\n\n{JUDGE}\n\n{END}"
    return f"{opening}\n\n{BODY}\n\n{JUDGE} {TEXT}\n\n{END}"


CELLS = {
    "direct_after": build(QUESTION, False),    # = P1 (controle de ruído)
    "direct_before": build(QUESTION, True),    # só a ORDEM muda
    "task_after": build(TASK, False),          # só o ENQUADRAMENTO muda
    "task_before": build(TASK, True),          # os dois mudam
}
CELL_SHA256 = {k: sha256(v) for k, v in CELLS.items()}

assert CELLS["direct_after"] == P1, "a célula de controle tem de ser o P1 byte a byte"
assert all(v.count("{text}") == 1 and v.endswith("\n") for v in CELLS.values())
