"""Zero-shot da Task 2.1 (meme sexista? YES/NO) com o GLM-4.6V-Flash servido por vLLM (API OpenAI).

    python3 EXIST_MEMES/glm_zeroshot.py [N]    # N primeiros memes do treino (padrão 5)
"""
import base64
import json
import mimetypes
import re
import sys
from pathlib import Path

from openai import OpenAI

from memes_core import pre_process_text  # mesma limpeza de texto do pipeline de memes

DATA = Path(__file__).parent / "Dataset/memes/training"
client = OpenAI(base_url="http://193.136.195.31:8000/v1", api_key="EMPTY")  # o servidor não exige chave
PROMPT = "Is this meme sexist? Answer only YES or NO.\n\nMeme text: {text}"


def parse(content):
    """O GLM responde '<|begin_of_box|>YES<|end_of_box|>'; o raciocínio vem à parte (reasoning_content)."""
    found = re.findall(r"\b(YES|NO)\b", (content or "").upper())
    return found[-1] if found else "?"


assert parse("\n<|begin_of_box|>NO<|end_of_box|>") == "NO"


def ask(image, text):
    url = f"data:{mimetypes.guess_type(image)[0]};base64,{base64.b64encode(image.read_bytes()).decode()}"
    r = client.chat.completions.create(
        model="glm-4.6v-flash",
        max_tokens=2048,  # o raciocínio conta no limite; cortado = content vazio = "?"
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": PROMPT.format(text=text)},
        ]}],
    )
    return parse(r.choices[0].message.content)


n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
items = list(json.loads((DATA / "EXIST2026_training.json").read_text()).values())[:n]
hits = counted = 0
for it in items:
    yes = it["labels_task2_1"].count("YES")
    gold = "YES" if yes > 3 else "NO" if yes < 3 else "TIE"  # 6 votos por meme; 3-3 fica fora da conta
    pred = ask(DATA / it["path_memes"], pre_process_text(it["text"]))
    if gold != "TIE":
        counted += 1
        hits += pred == gold
    print(f"{it['id_EXIST']}  votos YES={yes}/6  gold={gold:3}  pred={pred}")
print(f"acertos: {hits}/{counted} (empates fora)")
