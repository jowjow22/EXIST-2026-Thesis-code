"""Prompts P1–P3 com o texto exato da §4 do SPEC_25.md, e a sonda visual do G0.3.

`{text}` é substituído com `str.replace`, nunca com `.format`: o texto dos memes
pode conter chaves. Cada prompt termina com `\n` (o template acrescenta `/nothink`
ao fim quando o thinking está desligado).
"""
import hashlib

P1 = """Is the meme sexist, in any form, or does it describe situations in which such discrimination occurs (i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behavior)?

YES: the meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.
NO: the meme does not prejudice, underestimate or discriminate women.
A meme that condemns or reports sexism is still YES.

Remember that sexism may be expressed in different forms! It may sound friendly, funny, offensive, hateful or even violent. Sexism comprises any form of oppression or prejudice against women because of their sex. This discrimination can be based on different beliefs such as stereotypes, the belief that men are superior to women or irrational hatred of women (misogyny). Note that misogyny is a radical/hate-driven sexism.

Judge the image and the text together. Text extracted from the meme (English or Spanish, do not translate it):
{text}

Answer with one word: YES or NO.
"""

P2 = """Sexism is any form of oppression or prejudice against women because of their sex. It can sound friendly, funny, offensive, hateful or violent, and it can rest on stereotypes, on the belief that men are superior to women, or on hatred of women (misogyny, the radical, hate-driven form of sexism).

Look at the meme, image and text together. Label it YES if it is sexist itself, if it describes a sexist situation, or if it criticizes a sexist behavior. A meme that condemns or reports sexism also counts as YES. Label it NO if it does not prejudice, underestimate or discriminate women.

Text extracted from the meme (English or Spanish, keep it in its original language):
{text}

Answer with one word: YES or NO.
"""

P3 = """Text extracted from the meme (English or Spanish, original language):
{text}

Task: decide whether this meme, considering the image and the text together, contains sexist expressions or behaviors. This includes memes that are sexist themselves, memes that describe a sexist situation, and memes that criticize or condemn a sexist behavior. Sexism means any oppression or prejudice against women because of their sex, whether it sounds friendly, funny, offensive, hateful or violent, including stereotypes, the belief that men are superior to women, and misogyny.

Is this meme sexist under this definition? Answer with one word: YES or NO.
"""

PROMPTS = {1: P1, 2: P2, 3: P3}
PROBE_PROMPT = "Describe the image in one sentence. Do not transcribe its text.\n"


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def render(paraphrase: int, text: str) -> str:
    return PROMPTS[paraphrase].replace("{text}", text)


PROMPT_SHA256 = {f"P{k}": sha256(v) for k, v in PROMPTS.items()}
PROBE_SHA256 = sha256(PROBE_PROMPT)

assert all(p.endswith("\n") and p.count("{text}") == 1 for p in PROMPTS.values())
