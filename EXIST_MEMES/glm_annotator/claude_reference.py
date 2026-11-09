"""Doc 31 (SPEC_31.md): os mesmos memes extremos (select_extremes.py) e os mesmos P1–P3 do GLM,
agora no Claude, pela Message Batches API (50% do preço; resultado em até 24 h, em geral < 1 h).

Pré-requisitos (no Mac, uma vez):
    pip install anthropic
    export ANTHROPIC_API_KEY=...            # nunca gravar a chave no repositório

Uso (da raiz do repositório):
    # 1. teste síncrono: 1 meme × 3 prompts, nada é gravado, mostra resposta, tokens e custo estimado
    python3 EXIST_MEMES/glm_annotator/claude_reference.py --model claude-sonnet-5-5 --arm think_off --smoke
    # 2. piloto: os 20 primeiros de cada grupo (a ordem já é aleatória)
    python3 EXIST_MEMES/glm_annotator/claude_reference.py --model claude-sonnet-5-5 --arm think_off --per-group 20
    # 3. todos os 1 551 memes extremos
    python3 EXIST_MEMES/glm_annotator/claude_reference.py --model claude-sonnet-5-5 --arm think_off

Retomável: os lotes enviados ficam em claude_reference/batches.json. Ctrl+C ou queda de rede não
perdem nada; o mesmo comando de novo recolhe os lotes pendentes e só envia o que falta.

Diferenças inevitáveis em relação ao GLM (registradas em SPEC_31 §1): os modelos Claude 5.5 não
aceitam temperature/top_p/top_k fora do padrão nem seed, e a Batch API não aceita `fallbacks`
(uma recusa fica gravada como "?" com a categoria, sem trocar de modelo).
"""
import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from client import majority, now, parse, read_jsonl
from common import LABELS, MEMES_DIR, REF, REPORTS, clean_texts, image_path, load_memes
from prompts import PROMPTS, sha256

OUT = MEMES_DIR / "claude_reference"
RAW = OUT / "raw"
STATE = OUT / "batches.json"
EXTREMES = REF / "extremes" / "extremes.csv"

EFFORT = "medium"           # explícito nos dois braços (o padrão muda por modelo)
MAX_TOKENS = {"think_off": 256, "think_on": 16000}   # think_on: o raciocínio conta em max_tokens
# Como desligar o thinking em cada modelo 5.5. O Opus 5.5 não desliga (400 em qualquer effort).
THINK_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}, "claude-haiku-5-5": {"type": "disabled"}}
THINK_ON = {"type": "adaptive", "display": "summarized"}   # summarized: grava um resumo do raciocínio
PRICE = {"claude-opus-5-5": (4, 20), "claude-sonnet-5-5": (2, 10), "claude-haiku-5-5": (0.10, 0.50)}
MAX_BATCH_BYTES = 150_000_000   # limite da API: 256 MB e 100 000 pedidos por lote
POLL_S = 60


def thinking_cfg(model: str, arm: str) -> dict:
    if arm == "think_on":
        return THINK_ON
    assert model in THINK_OFF, f"{model} não permite desligar o thinking; use --arm think_on"
    return THINK_OFF[model]


def request_params(model: str, arm: str, image_b64: str, media_type: str, text: str) -> dict:
    """Uma mensagem user, sem system prompt, partes [imagem, texto] — igual ao GLM."""
    return {"model": model, "max_tokens": MAX_TOKENS[arm], "thinking": thinking_cfg(model, arm),
            "output_config": {"effort": EFFORT},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": text}]}]}


def logged_params(model: str, arm: str) -> dict:
    return {"model": model, "max_tokens": MAX_TOKENS[arm], "thinking": thinking_cfg(model, arm),
            "effort": EFFORT}


def image_b64(item, reencode: bool = False) -> tuple[str, str, str]:
    """(base64, media type, sha256 do arquivo original).
    - Acima do limite de 5 MB por imagem da API (nos extremos: só 210851), reduz para 1568 px
      de lado e JPEG: é o tamanho a que o Claude reduz as imagens de qualquer forma.
    - `reencode`: a API devolveu "Could not process image" para este JPEG (que o PIL e o GLM
      leem sem erro). Decodifica e manda como PNG: mesmos pixels, sem nova compressão."""
    import io
    from PIL import Image
    p = image_path(item)
    raw = p.read_bytes()
    sha, mt = hashlib.sha256(raw).hexdigest(), mimetypes.guess_type(p)[0]
    if len(raw) * 4 / 3 > 5 * 1024 * 1024:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((1568, 1568))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        raw, mt = buf.getvalue(), "image/jpeg"
        print(f"aviso: imagem {item['id_EXIST']} reduzida para {im.size} ({len(raw) / 1e6:.1f} MB)")
    elif reencode:
        buf = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert("RGB").save(buf, "PNG")
        raw, mt = buf.getvalue(), "image/png"
    return base64.standard_b64encode(raw).decode(), mt, sha


IMAGE_ERROR = "Could not process image"


def image_failed(r: dict) -> bool:
    """Registro de erro de imagem ainda não tentado com a imagem reconvertida."""
    return IMAGE_ERROR in (r.get("error") or "") and not r.get("image_reencoded")


def raw_path(model: str, arm: str):
    return RAW / f"{model}_{arm}.jsonl"


def custom_id(meme_id, p) -> str:
    return f"{meme_id}_p{p}"


def from_custom_id(cid: str) -> tuple[str, int]:
    meme_id, p = cid.rsplit("_p", 1)
    return meme_id, int(p)


def load_state() -> list:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []


def save_state(state: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def select_ids(per_group: int | None) -> list:
    with open(EXTREMES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = {}
    out = []
    for r in rows:                      # a ordem dentro do grupo já é aleatória (rng 2026)
        seen[r["group"]] = seen.get(r["group"], 0) + 1
        if per_group is None or seen[r["group"]] <= per_group:
            out.append(r["meme_id"])
    return out


def message_fields(msg) -> dict:
    """Campos gravados de uma resposta bem-sucedida, no formato do JSONL do GLM."""
    content = "".join(b.text for b in msg.content if b.type == "text")
    thinking = "\n".join(b.thinking for b in msg.content if b.type == "thinking" and b.thinking) or None
    refusal = msg.stop_reason == "refusal"
    if refusal:
        label, method = "?", "refusal"
    else:
        label, method = parse(content, "length" if msg.stop_reason == "max_tokens" else msg.stop_reason)
    return {"content": content, "reasoning_content": thinking, "finish_reason": msg.stop_reason,
            "refusal_category": msg.stop_details.category if refusal and msg.stop_details else None,
            "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens},
            "label": label, "parse_method": method, "server_model": msg.model}


# ------------------------------------------------------------------ teste síncrono

def smoke(client, model, arm, memes, ids, texts):
    i = ids[0]
    b64, mt, _ = image_b64(memes[i])
    tokens = []
    for p in (1, 2, 3):
        msg = client.messages.create(**request_params(model, arm, b64, mt, PROMPTS[p].replace("{text}", texts[i])))
        f = message_fields(msg)
        tokens.append((f["usage"]["input_tokens"], f["usage"]["output_tokens"]))
        print(f"meme {i} P{p} · stop={f['finish_reason']} · label={f['label']} ({f['parse_method']}) · "
              f"tokens in/out={tokens[-1]} · resposta={f['content']!r}")
        if f["reasoning_content"]:
            print(f"   raciocínio (resumo): {f['reasoning_content'][:300]!r}")
    if model in PRICE:
        n = 3 * len(select_ids(None))
        tin = sum(t[0] for t in tokens) / 3
        tout = sum(t[1] for t in tokens) / 3
        cost = n * (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6 / 2
        print(f"\nestimativa para os {n // 3} memes extremos ({n} pedidos, batch −50%): ~US$ {cost:.2f}")


# ------------------------------------------------------------------ Batch API

def collect(client, b: dict, out) -> int:
    """Grava os resultados de um lote terminado. Erros transitórios (servidor, expirado,
    cancelado) não são gravados: voltam a ser enviados na próxima execução."""
    written = 0
    for res in client.messages.batches.results(b["id"]):
        meme_id, p = from_custom_id(res.custom_id)
        base = {"meme_id": meme_id, "lang": b["lang"].get(meme_id), "arm": b["arm"], "paraphrase": p,
                "seed": None, "params": logged_params(b["model"], b["arm"]),
                "prompt_sha256": sha256(PROMPTS[p]), "image_sha256": b["image_sha256"].get(meme_id),
                "image_reencoded": meme_id in b.get("reencoded", []), "batch_id": b["id"]}
        t = res.result.type
        if t == "succeeded":
            rec = {**base, **message_fields(res.result.message), "error": None}
        elif t == "errored" and "invalid_request" in repr(res.result.error):   # 4xx: determinístico
            rec = {**base, "content": None, "reasoning_content": None, "finish_reason": None,
                   "refusal_category": None, "usage": None, "label": "?", "parse_method": "error",
                   "server_model": None, "error": repr(res.result.error)}
        else:
            print(f"  {res.custom_id}: {t} — será reenviado")
            continue
        rec["timestamp"] = now()
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
    out.flush()
    return written


def submit(client, model, arm, memes, todo, texts, state, reencode=frozenset()) -> None:
    """Envia `todo` [(meme_id, p)] em lotes de até MAX_BATCH_BYTES. Memes em `reencode` vão
    com a imagem reconvertida para PNG (ver image_b64)."""
    reqs, size, lang, img_sha, cache = [], 0, {}, {}, {}

    def flush():
        nonlocal reqs, size, lang, img_sha
        if not reqs:
            return
        batch = client.messages.batches.create(requests=reqs)
        state.append({"id": batch.id, "model": model, "arm": arm, "submitted": now(), "collected": False,
                      "custom_ids": [r["custom_id"] for r in reqs], "lang": lang, "image_sha256": img_sha,
                      "reencoded": sorted(set(lang) & reencode)})
        save_state(state)
        print(f"lote enviado: {batch.id} · {len(reqs)} pedidos · {size / 1e6:.0f} MB")
        reqs, size, lang, img_sha = [], 0, {}, {}

    for i, p in todo:
        if i not in cache:
            cache.clear()               # todo vem ordenado por meme: só a imagem atual em memória
            cache[i] = image_b64(memes[i], reencode=i in reencode)
        b64, mt, sha = cache[i]
        params = request_params(model, arm, b64, mt, PROMPTS[p].replace("{text}", texts[i]))
        n = len(b64) + 4000
        if size + n > MAX_BATCH_BYTES:
            flush()
        reqs.append(Request(custom_id=custom_id(i, p), params=MessageCreateParamsNonStreaming(**params)))
        size += n
        lang[i], img_sha[i] = memes[i]["lang"], sha
    flush()


def run(client, model, arm, memes, ids, texts):
    RAW.mkdir(parents=True, exist_ok=True)
    state = load_state()
    path = raw_path(model, arm)
    first = True
    while True:
        mine = [b for b in state if b["model"] == model and b["arm"] == arm and not b["collected"]]
        with open(path, "a", encoding="utf-8") as out:
            for b in mine:
                st = client.messages.batches.retrieve(b["id"])
                if st.processing_status == "ended":
                    print(f"lote {b['id']} terminou: gravados {collect(client, b, out)}")
                    b["collected"] = True
                    save_state(state)
                else:
                    c = st.request_counts
                    print(f"lote {b['id']}: {st.processing_status} · processando {c.processing} · "
                          f"ok {c.succeeded} · erro {c.errored}")
        pending = {from_custom_id(c) for b in mine if not b["collected"] for c in b["custom_ids"]}
        recs = read_jsonl(path)
        done = {(r["meme_id"], r["paraphrase"]) for r in recs if not image_failed(r)}
        reencode = frozenset(r["meme_id"] for r in recs if IMAGE_ERROR in (r.get("error") or ""))
        todo = [(i, p) for i in ids for p in (1, 2, 3) if (i, p) not in done and (i, p) not in pending]
        print(f"[{model} {arm}] {3 * len(ids)} pedidos · {len(done & {(i, p) for i in ids for p in (1, 2, 3)})} "
              f"gravados · {len(pending)} em lote · {len(todo)} a enviar")
        if todo and first:              # envia uma vez por execução: erro persistente não vira laço
            submit(client, model, arm, memes, todo, texts, state, reencode)
            first = False
            continue
        if not pending:
            if todo:
                print(f"{len(todo)} pedidos falharam de forma transitória: rode o mesmo comando de novo")
            return not todo
        print(f"aguardando {POLL_S} s (Ctrl+C é seguro: rode o mesmo comando depois para recolher)")
        time.sleep(POLL_S)


# ------------------------------------------------------------------ relatório

def report(model, arm, ids):
    recs = {(r["meme_id"], r["paraphrase"]): r for r in read_jsonl(raw_path(model, arm))}
    with open(EXTREMES, encoding="utf-8") as f:
        ext = {r["meme_id"]: r for r in csv.DictReader(f)}
    tag = f"{model} {arm}"
    L = [f"# Doc 31 — {tag} nos memes extremos (treino)", "",
         f"Mesmos P1–P3 e memes do GLM; effort {EFFORT}; thinking {json.dumps(thinking_cfg(model, arm))}. "
         "Taxa de YES = fração de YES nas 3 paráfrases parseadas. Voto = maioria das 3.", "",
         "| grupo | n | humanos | voto YES GLM off | voto YES GLM on | voto YES Claude | YES Claude "
         "(paráfrases) | parse | recusas |", "|---|---|---|---|---|---|---|---|---|"]
    for g, k in (("1Y5N", 1), ("3Y3N", 3), ("5Y1N", 5)):
        for lang in ("total", "en", "es"):
            s = [i for i in ids if ext[i]["group"] == g and (lang == "total" or ext[i]["lang"] == lang)]
            labs = {i: [recs[(i, p)]["label"] for p in (1, 2, 3)] for i in s}
            flat = [l for v in labs.values() for l in v]
            parsed = [l for l in flat if l in LABELS]
            cell = lambda xs: f"{sum(xs) / len(xs):.3f}" if xs else "—"
            L.append(f"| {g if lang == 'total' else '↳ ' + lang} | {len(s)} | {k / 6:.3f} | "
                     f"{cell([ext[i]['think_off_vote'] == 'YES' for i in s])} | "
                     f"{cell([ext[i]['think_on_vote'] == 'YES' for i in s])} | "
                     f"{cell([majority(v) == 'YES' for v in labs.values()])} | "
                     f"{cell([l == 'YES' for l in parsed])} | {cell([l in LABELS for l in flat])} | "
                     f"{sum(recs[(i, p)]['parse_method'] == 'refusal' for i in s for p in (1, 2, 3))} |")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"claude_{model}_{arm}_extremes.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L) + f"\n\ngravado: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PRICE))
    ap.add_argument("--arm", required=True, choices=sorted(MAX_TOKENS))
    ap.add_argument("--per-group", type=int, help="só os N primeiros de cada grupo (piloto)")
    ap.add_argument("--smoke", action="store_true", help="1 meme × 3 prompts síncrono; não grava")
    a = ap.parse_args()
    thinking_cfg(a.model, a.arm)                     # falha cedo se o modelo não desliga o thinking
    memes, ids = load_memes(), select_ids(a.per_group)
    texts = clean_texts(memes, ids)                  # mesmo texto limpo que o GLM recebeu
    client = anthropic.Anthropic()
    if a.smoke:
        return smoke(client, a.model, a.arm, memes, ids, texts)
    if run(client, a.model, a.arm, memes, ids, texts):
        report(a.model, a.arm, ids)


if __name__ == "__main__":
    main()
