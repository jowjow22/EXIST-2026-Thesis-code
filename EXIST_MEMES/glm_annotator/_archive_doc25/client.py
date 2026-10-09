"""Cliente do GLM (§5.1): chamadas assíncronas, parser, retomada e registro JSONL.

Uma linha por chamada concluída em `glm_reference/raw/<arm>.jsonl`. Chave de
retomada: (meme_id, arm, paraphrase, seed). Uma chamada gravada nunca é refeita.
Falha de transporte que esgota as retentativas NÃO é gravada ali (é uma chamada
interrompida): vai para `raw/<arm>.errors.jsonl` e é refeita na próxima execução.
"""
import asyncio
import datetime
import json
import re
import time

import httpx
import openai
from openai import AsyncOpenAI

from common import BASE_URL, FIXED, LABELS, MODEL, RAW_DIR, image_data_url
from prompts import PROMPT_SHA256, render

BOX = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.S)
WORD = re.compile(r"\b(YES|NO)\b")
# Só estes são retentados (§3). APITimeoutError é subclasse de APIConnectionError;
# InternalServerError é o que o SDK levanta para HTTP >= 500. 429 (servidor saturado)
# conta como transporte: gravá-lo como "?" seria permanente (SPEC_25 §18).
TRANSPORT = (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError)


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s).strip().upper()     # tira pontuação e `*`


def parse(content, finish_reason):
    """§5.1, só sobre `content`. Devolve (label, parse_method)."""
    c = content or ""
    m = BOX.search(c)
    if m and _norm(m.group(1)) in LABELS:
        return _norm(m.group(1)), "box"
    if _norm(c) in LABELS:
        return _norm(c), "plain"
    found = set(WORD.findall(c.upper()))
    if len(found) == 1:
        return found.pop(), "single_token"
    if not c.strip():
        return "?", "empty"
    if finish_reason == "length":
        return "?", "length"
    if len(found) == 2:
        return "?", "both"
    return "?", "other"


def confident(labels) -> bool:
    """§3: as 3 paráfrases parseadas e iguais."""
    return all(l in LABELS for l in labels) and len(set(labels)) == 1


def majority(labels):
    """§7: maioria dos labels parseados quando >= 2 concordam; senão None."""
    for lab in LABELS:
        if sum(l == lab for l in labels) >= 2:
            return lab
    return None


def request_params(arm: str, seed: int) -> dict:
    """Tudo o que vai no pedido além de model/messages; é gravado em `params`."""
    extra = {"top_k": FIXED["top_k"], "repetition_penalty": FIXED["repetition_penalty"]}
    if arm == "think_off":
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    return {"temperature": FIXED["temperature"], "top_p": FIXED["top_p"], "seed": seed,
            "max_tokens": FIXED["max_tokens"][arm], "extra_body": extra}


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY", max_retries=0)  # retentativas são nossas


def server_info() -> tuple[list, str]:
    """G0.1: ids dos modelos em /v1/models e versão do vLLM em /version (ou 'unknown')."""
    ids = [m["id"] for m in httpx.get(f"{BASE_URL}/models", timeout=30).json()["data"]]
    try:
        version = httpx.get(BASE_URL.rsplit("/v1", 1)[0] + "/version", timeout=30).json()["version"]
    except Exception:
        version = "unknown"
    return ids, version


async def chat(client, *, image_url: str, text: str, arm: str, params: dict):
    """Uma chamada; até 3 retentativas com backoff, só para transporte."""
    retries = FIXED["transport_retries"]
    for attempt in range(retries + 1):
        try:
            t0 = time.monotonic()
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text}]}],
                timeout=FIXED["timeout_s"][arm], **params)
            return r, time.monotonic() - t0
        except TRANSPORT:
            if attempt == retries:
                raise
            await asyncio.sleep(5 * 2 ** attempt)


def reasoning_of(msg):
    extra = msg.model_extra or {}
    return extra.get("reasoning_content") or extra.get("reasoning")


def response_fields(r) -> dict:
    ch = r.choices[0]
    label, method = parse(ch.message.content, ch.finish_reason)
    usage = ({"prompt_tokens": r.usage.prompt_tokens,
              "completion_tokens": r.usage.completion_tokens} if r.usage else None)
    return {"content": ch.message.content, "reasoning_content": reasoning_of(ch.message),
            "finish_reason": ch.finish_reason, "usage": usage,
            "label": label, "parse_method": method, "server_model": r.model}


def log_call(tag: str, idx: int, total: int, meme_id, ok: bool, text) -> None:
    """Uma linha por request: ✓ + resposta do modelo, ou ✗ + erro."""
    out = " ".join(str(text if text not in (None, "") else "(vazio)").split())
    print(f"[{tag}] {'✓' if ok else '✗'} {idx}/{total} · meme {meme_id} · {out[:200]}", flush=True)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_jsonl(path) -> list:
    """Uma última linha cortada (queda no meio da escrita) é descartada: era uma chamada
    interrompida e será refeita. Corrupção em qualquer outra linha continua sendo erro."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    out = [json.loads(line) for line in lines[:-1]]
    if lines:
        try:
            out.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            print(f"aviso: última linha de {path.name} incompleta, descartada (será refeita)")
            path.write_text("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""), encoding="utf-8")
    return out


def call_key(meme_id, arm, paraphrase, seed) -> tuple:
    return (str(meme_id), arm, int(paraphrase), int(seed))


def load_records(arm: str) -> dict:
    """{chave: registro} do JSONL do braço."""
    return {call_key(r["meme_id"], r["arm"], r["paraphrase"], r["seed"]): r
            for r in read_jsonl(RAW_DIR / f"{arm}.jsonl")}


def make_jobs(memes: dict, ids, texts: dict, paraphrases, seeds) -> list:
    return [{"meme_id": i, "lang": memes[i]["lang"], "item": memes[i], "text": texts[i],
             "paraphrase": p, "seed": s}
            for i in ids for p in paraphrases for s in seeds]


async def run_calls(jobs: list, arm: str, concurrency: int) -> dict:
    """Faz as chamadas que faltam no JSONL do braço. Devolve contagens."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = set(load_records(arm))
    todo = [j for j in jobs if call_key(j["meme_id"], arm, j["paraphrase"], j["seed"]) not in done]
    print(f"[{arm}] {len(jobs)} pedidas · {len(jobs) - len(todo)} já gravadas · {len(todo)} a fazer")
    client, sem = make_client(), asyncio.Semaphore(concurrency)
    stats = {"written": 0, "transport_failed": 0, "n": 0}
    t0 = time.monotonic()
    with open(RAW_DIR / f"{arm}.jsonl", "a", encoding="utf-8") as f, \
         open(RAW_DIR / f"{arm}.errors.jsonl", "a", encoding="utf-8") as fe:

        async def one(j):
            base = {"meme_id": j["meme_id"], "lang": j["lang"], "arm": arm,
                    "paraphrase": j["paraphrase"], "seed": j["seed"]}
            params = request_params(arm, j["seed"])
            tag = f"{arm} P{j['paraphrase']} s{j['seed']}"
            async with sem:                   # imagem só é lida com o slot na mão
                url, img_sha = image_data_url(j["item"])
                meta = {**base, "params": params,
                        "prompt_sha256": PROMPT_SHA256[f"P{j['paraphrase']}"],
                        "image_sha256": img_sha}
                try:
                    r, lat = await chat(client, image_url=url, text=render(j["paraphrase"], j["text"]),
                                        arm=arm, params=params)
                    rec = {**meta, **response_fields(r), "latency_s": round(lat, 3), "error": None}
                except TRANSPORT as e:
                    fe.write(json.dumps({**base, "error": repr(e), "timestamp": now()}) + "\n")
                    fe.flush()
                    stats["transport_failed"] += 1
                    stats["n"] += 1
                    log_call(tag, stats["n"], len(todo), j["meme_id"], False, repr(e))
                    return
                except openai.APIStatusError as e:     # 4xx: determinístico, sem retentativa
                    rec = {**meta, "content": None, "reasoning_content": None,
                           "finish_reason": None, "usage": None, "latency_s": None,
                           "label": "?", "parse_method": "error", "error": repr(e),
                           "server_model": None}
            stats["n"] += 1
            log_call(tag, stats["n"], len(todo), j["meme_id"], rec["error"] is None,
                     rec["error"] or rec["content"])
            rec["timestamp"] = now()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            stats["written"] += 1
            if stats["written"] % 100 == 0:
                rate = stats["written"] / (time.monotonic() - t0) * 60
                print(f"[{arm}] {stats['written']}/{len(todo)} · {rate:.1f} chamadas/min")

        await asyncio.gather(*(one(j) for j in todo))
    await client.close()
    print(f"[{arm}] gravadas {stats['written']} · falhas de transporte {stats['transport_failed']}")
    return stats


def assert_complete(arm: str, jobs: list) -> dict:
    """A fase só segue com todas as chamadas pedidas gravadas."""
    recs = load_records(arm)
    missing = [j for j in jobs if call_key(j["meme_id"], arm, j["paraphrase"], j["seed"]) not in recs]
    assert not missing, (f"[{arm}] faltam {len(missing)} chamadas (falha de transporte?). "
                         "Rode o mesmo comando de novo: a retomada refaz só as que faltam.")
    return recs
