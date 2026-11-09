"""Doc 32 (SPEC_32.md): cópia de claude_reference.py (doc 31) para a nova onda de testes.
Mesmos memes extremos, modelo, braços e Batch API; o que muda é a mensagem:
- system: um dos três prompts novos S1–S3 (prompts_32.py), um pedido por prompt;
- user: só [imagem, texto do meme dentro de <meme_text></meme_text>].

Nada do doc 31 é tocado: esta cópia grava em claude_reference_32/ e em
reports/claude32_<modelo>_<braço>_extremes.md. Do doc 31 só se lê (nunca se escreve):
- os 21 memes que deram "Could not process image", que aqui já vão como PNG no primeiro envio;
- os votos do Claude com P1–P3, para a coluna de comparação do relatório.

Pré-requisitos (no Mac, uma vez):
    pip install anthropic
    export ANTHROPIC_API_KEY=...            # nunca gravar a chave no repositório

Uso (da raiz do repositório):
    # 1. teste síncrono: 1 meme × 3 prompts, nada é gravado, mostra resposta, tokens e custo estimado
    python3 EXIST_MEMES/glm_annotator/claude_reference_32.py --model claude-sonnet-5-5 --arm think_off --smoke
    # 2. piloto: os 20 primeiros de cada grupo (a ordem já é aleatória)
    python3 EXIST_MEMES/glm_annotator/claude_reference_32.py --model claude-sonnet-5-5 --arm think_off --per-group 20
    # 3. todos os 1 551 memes extremos
    python3 EXIST_MEMES/glm_annotator/claude_reference_32.py --model claude-sonnet-5-5 --arm think_off
    # 4. doc 33 (SPEC_33.md): os 1 835 memes de treino que não são extremos, com os mesmos S1–S3
    python3 EXIST_MEMES/glm_annotator/claude_reference_32.py --model claude-sonnet-5-5 --arm think_off --scope train
    # 5. doc 33: os 60 memes do piloto de novo, num arquivo separado (ruído entre rodadas)
    python3 EXIST_MEMES/glm_annotator/claude_reference_32.py --model claude-sonnet-5-5 --arm think_off --scope repeat

Com --scope train (ou repeat) os registros vão para raw/<modelo>_<braço>_<escopo>.jsonl, separados
do JSONL dos extremos (que analyze_claude32.py e o SPEC_32 §7 leem como está). persona_ref.py junta
extremos e treino; o repeat só serve para medir o ruído entre rodadas.

Retomável: os lotes ficam em claude_reference_32/batches.json. Ctrl+C, queda de rede ou da montagem
não perdem nada nem pagam duas vezes; o mesmo comando de novo recolhe os lotes pendentes e só envia
o que falta.
- Cada lote é gravado como intenção (id None) ANTES de ser criado, e o POST de criação vai sem retry
  automático (o SDK repetiria sem chave de idempotência). Se o envio cair antes de o id voltar, a
  execução seguinte procura o lote na API (adopt_stubs) em vez de pagar de novo.
- Se registros de um lote já recolhido sumirem do JSONL, o lote é baixado de novo (grátis por 29
  dias) em vez de reenviado (recover_lost).

Diferenças inevitáveis em relação ao GLM (registradas em SPEC_31 §1): os modelos Claude 5.5 não
aceitam temperature/top_p/top_k fora do padrão nem seed, e a Batch API não aceita `fallbacks`
(uma recusa fica gravada como "?" com a categoria, sem trocar de modelo).
"""
import argparse
import base64
import csv
import datetime
import hashlib
import itertools
import json
import mimetypes
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from client import majority, now, parse, read_jsonl
from common import LABELS, MEMES_DIR, REF, REPORTS, clean_texts, image_path, load_memes
from prompts_32 import SYSTEM, USER_TEXT, sha256

OUT = MEMES_DIR / "claude_reference_32"
RAW = OUT / "raw"
STATE = OUT / "batches.json"
EXTREMES = REF / "extremes" / "extremes.csv"
TRAIN_IDS = REF / "train_ids.json"
SCOPES = ("extremes", "train", "repeat")
DOC31_RAW = MEMES_DIR / "claude_reference" / "raw"    # só leitura

EFFORT = "medium"           # explícito nos dois braços (o padrão muda por modelo)
MAX_TOKENS = {"think_off": 256, "think_on": 16000}   # think_on: o raciocínio conta em max_tokens
# Como desligar o thinking em cada modelo 5.5. O Opus 5.5 não desliga (400 em qualquer effort).
THINK_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}, "claude-haiku-5-5": {"type": "disabled"}}
THINK_ON = {"type": "adaptive", "display": "summarized"}   # summarized: grava um resumo do raciocínio
PRICE = {"claude-opus-5-5": (4, 20), "claude-sonnet-5-5": (2, 10), "claude-haiku-5-5": (0.10, 0.50)}
MAX_BATCH_BYTES = 150_000_000   # limite da API: 256 MB e 100 000 pedidos por lote
POLL_S = 60
CREATE_TRIES, CREATE_WAIT_S = 5, 30   # conexão caída na criação: confere na API e tenta de novo


def thinking_cfg(model: str, arm: str) -> dict:
    if arm == "think_on":
        return THINK_ON
    assert model in THINK_OFF, f"{model} não permite desligar o thinking; use --arm think_on"
    return THINK_OFF[model]


def request_params(model: str, arm: str, p: int, image_b64: str, media_type: str, text: str) -> dict:
    """System = S<p>; uma mensagem user com [imagem, texto do meme entre tags]."""
    assert text.strip(), "texto do meme vazio: a API recusa bloco de texto vazio"
    return {"model": model, "max_tokens": MAX_TOKENS[arm], "thinking": thinking_cfg(model, arm),
            "output_config": {"effort": EFFORT}, "system": SYSTEM[p],
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": USER_TEXT.replace("{text}", text)}]}]}


def logged_params(model: str, arm: str, p: int) -> dict:
    return {"model": model, "max_tokens": MAX_TOKENS[arm], "thinking": thinking_cfg(model, arm),
            "effort": EFFORT, "system": f"S{p}", "user_text": USER_TEXT}


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
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "PNG")
        if len(buf.getvalue()) * 4 / 3 > 5 * 1024 * 1024:     # PNG acima do limite: reduz como acima
            im.thumbnail((1568, 1568))
            buf = io.BytesIO()
            im.save(buf, "PNG")
            print(f"aviso: PNG de {item['id_EXIST']} reduzido para {im.size}")
        raw, mt = buf.getvalue(), "image/png"
    return base64.standard_b64encode(raw).decode(), mt, sha


IMAGE_ERROR = "Could not process image"


def image_failed(r: dict) -> bool:
    """Registro de erro de imagem ainda não tentado com a imagem reconvertida."""
    return IMAGE_ERROR in (r.get("error") or "") and not r.get("image_reencoded")


def doc31_records(name: str) -> list:
    """Registros de um JSONL do doc 31, só leitura (read_jsonl pode reescrever o arquivo)."""
    p = DOC31_RAW / name
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line] if p.exists() else []


def doc31_image_failures() -> frozenset:
    """Memes com "Could not process image" no doc 31: aqui vão como PNG desde o primeiro envio."""
    names = sorted(p.name for p in DOC31_RAW.glob("*.jsonl") if not p.name.startswith("._"))  # ._* = AppleDouble
    return frozenset(r["meme_id"] for n in names for r in doc31_records(n) if IMAGE_ERROR in (r.get("error") or ""))


def raw_path(model: str, arm: str, scope: str = "extremes"):
    return RAW / (f"{model}_{arm}.jsonl" if scope == "extremes" else f"{model}_{arm}_{scope}.jsonl")


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


def train_rest_ids() -> list:
    """Doc 33: memes de treino fora dos extremos, na ordem de train_ids.json."""
    ext = set(select_ids(None))
    return [i for i in json.loads(TRAIN_IDS.read_text(encoding="utf-8")) if i not in ext]


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
    b64, mt, _ = image_b64(memes[i], reencode=i in doc31_image_failures())
    tokens = []
    for p in (1, 2, 3):
        msg = client.messages.create(**request_params(model, arm, p, b64, mt, texts[i]))
        f = message_fields(msg)
        tokens.append((f["usage"]["input_tokens"], f["usage"]["output_tokens"]))
        print(f"meme {i} P{p} · stop={f['finish_reason']} · label={f['label']} ({f['parse_method']}) · "
              f"tokens in/out={tokens[-1]} · resposta={f['content']!r}")
        if f["reasoning_content"]:
            print(f"   raciocínio (resumo): {f['reasoning_content'][:300]!r}")
    if model in PRICE:
        n = 3 * len(ids)
        tin = sum(t[0] for t in tokens) / 3
        tout = sum(t[1] for t in tokens) / 3
        cost = n * (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6 / 2
        print(f"\nestimativa para os {n // 3} memes deste escopo ({n} pedidos, batch −50%): ~US$ {cost:.2f}")


# ------------------------------------------------------------------ Batch API

def collect(client, b: dict, out, have_ok=frozenset(), have_any=frozenset()) -> int:
    """Grava os resultados de um lote terminado. Erros transitórios (servidor, expirado,
    cancelado) não são gravados: ficam em b["transient"] e voltam a ser enviados na próxima
    execução. Num lote baixado de novo (recover_lost) não se regrava o que já está no JSONL:
    pula a chave com registro bom (have_ok) e não grava um erro onde já há registro (have_any),
    para um erro antigo nunca ficar por último sobre o reenvio PNG."""
    written = 0
    for res in client.messages.batches.results(b["id"]):
        meme_id, p = from_custom_id(res.custom_id)
        if (meme_id, p) in have_ok or ((meme_id, p) in have_any and res.result.type != "succeeded"):
            continue
        base = {"meme_id": meme_id, "lang": b["lang"].get(meme_id), "arm": b["arm"], "paraphrase": p,
                "seed": None, "params": logged_params(b["model"], b["arm"], p),
                "prompt_sha256": sha256(SYSTEM[p]), "image_sha256": b["image_sha256"].get(meme_id),
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
            b.setdefault("transient", []).append(res.custom_id)
            continue
        rec["timestamp"] = now()
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
    out.flush()
    return written


def submit(client, model, arm, memes, todo, texts, state, reencode=frozenset(), scope="extremes") -> None:
    """Envia `todo` [(meme_id, p)] em lotes de até MAX_BATCH_BYTES. Memes em `reencode` vão
    com a imagem reconvertida para PNG (ver image_b64)."""
    reqs, size, lang, img_sha, cache = [], 0, {}, {}, {}

    def flush():
        nonlocal reqs, size, lang, img_sha
        if not reqs:
            return
        stub = {"id": None, "model": model, "arm": arm, "scope": scope, "submitted": now(), "collected": False,
                "custom_ids": [r["custom_id"] for r in reqs], "lang": lang, "image_sha256": img_sha,
                "reencoded": sorted(set(lang) & reencode)}
        for attempt in range(1, CREATE_TRIES + 1):
            if stub not in state:         # descartada por adopt_stubs na tentativa anterior
                stub["submitted"] = now()
                state.append(stub)
            save_state(state)             # a intenção fica gravada antes de pagar (adopt_stubs)
            try:
                # sem retry do SDK: ele repetiria o POST sem chave de idempotência e criaria lotes em dobro
                stub["id"] = client.with_options(max_retries=0).messages.batches.create(requests=reqs).id
                break
            except anthropic.APIConnectionError as e:
                print(f"conexão caiu na criação do lote ({attempt}/{CREATE_TRIES}, {e.__class__.__name__}); "
                      f"conferindo na API em {CREATE_WAIT_S} s…")
                time.sleep(CREATE_WAIT_S)
                adopt_stubs(client, state)    # adota se o servidor criou; senão descarta a intenção
                if stub["id"]:
                    break
                if attempt == CREATE_TRIES:
                    raise
        save_state(state)
        print(f"lote enviado: {stub['id']} · {len(reqs)} pedidos · {size / 1e6:.0f} MB")
        reqs, size, lang, img_sha = [], 0, {}, {}

    for i, p in todo:
        if i not in cache:
            cache.clear()               # todo vem ordenado por meme: só a imagem atual em memória
            cache[i] = image_b64(memes[i], reencode=i in reencode)
        b64, mt, sha = cache[i]
        params = request_params(model, arm, p, b64, mt, texts[i])
        n = len(b64) + 4000
        if size + n > MAX_BATCH_BYTES:
            flush()
        reqs.append(Request(custom_id=custom_id(i, p), params=MessageCreateParamsNonStreaming(**params)))
        size += n
        lang[i], img_sha[i] = memes[i]["lang"], sha
    flush()


def adopt_stubs(client, state: list) -> None:
    """Intenções de lote sem id: o envio caiu (Ctrl+C, rede, montagem) antes de o id voltar. Procura
    na API um lote criado depois da intenção, com o mesmo nº de pedidos e ainda não registrado:
    adota se houver exatamente um; descarta a intenção se não houver nenhum (nada foi cobrado);
    para se houver mais de um."""
    stubs = [b for b in state if b["id"] is None]
    if not stubs:
        return
    known = {b["id"] for b in state if b["id"]}
    recent = [x for x in itertools.islice(client.messages.batches.list(limit=100), 200) if x.id not in known]
    for b in stubs:
        t0 = datetime.datetime.fromisoformat(b["submitted"]) - datetime.timedelta(minutes=5)
        n = len(b["custom_ids"])
        cands = [x for x in recent if x.created_at >= t0 and n == sum(
            getattr(x.request_counts, k) for k in ("processing", "succeeded", "errored", "canceled", "expired"))]
        if len(cands) > 1:
            raise SystemExit(f"intenção de {n} pedidos ({b['submitted']}) casa com {len(cands)} lotes da API: "
                             f"{[c.id for c in cands]}. Resolver à mão em batches.json.")
        if cands:
            b["id"] = cands[0].id
            recent.remove(cands[0])
            print(f"lote recuperado da API: {b['id']} ({n} pedidos)")
        else:
            state.remove(b)
            print(f"intenção de {n} pedidos sem lote na API: descartada, os pedidos voltam a ser enviados")
    save_state(state)


def trim_cut_line(path) -> None:
    """Apara uma última linha cortada (queda da montagem) ou completa a quebra de linha, trocando o
    arquivo de uma vez (tmp + replace): uma segunda queda não deixa o JSONL pela metade."""
    if not path.exists() or not path.stat().st_size:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    last = data[data.rfind(b"\n") + 1:]
    try:
        json.loads(last)
        fixed = data + b"\n"
    except json.JSONDecodeError:
        fixed = data[:len(data) - len(last)]
        print(f"aviso: última linha de {path.name} incompleta, descartada (será refeita)")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(fixed)
    tmp.replace(path)


def recover_lost(state: list, have: set, model, arm, scope) -> None:
    """Lote marcado como recolhido cujos registros sumiram do JSONL: volta a pendente, para ser
    baixado de novo (os resultados ficam 29 dias na API) em vez de reenviado e pago de novo."""
    for b in state:
        if not (b["collected"] and b["id"] and b["model"] == model and b["arm"] == arm
                and b.get("scope", "extremes") == scope):
            continue
        miss = ({from_custom_id(c) for c in b["custom_ids"]} - have
                - {from_custom_id(c) for c in b.get("transient", [])})
        if miss:
            print(f"lote {b['id']}: {len(miss)} registros sumiram do JSONL; será baixado de novo, sem custo")
            b["collected"] = False
    save_state(state)


def run(client, model, arm, memes, ids, texts, scope="extremes"):
    RAW.mkdir(parents=True, exist_ok=True)
    state = load_state()
    path = raw_path(model, arm, scope)
    known_bad = doc31_image_failures()
    adopt_stubs(client, state)
    trim_cut_line(path)
    recover_lost(state, {(r["meme_id"], r["paraphrase"]) for r in read_jsonl(path)}, model, arm, scope)
    first = True
    while True:
        mine = [b for b in state if b["model"] == model and b["arm"] == arm and not b["collected"]
                and b.get("scope", "extremes") == scope]       # lotes antigos não têm "scope": extremos
        trim_cut_line(path)
        recs = read_jsonl(path)
        have_any = {(r["meme_id"], r["paraphrase"]) for r in recs}
        have_ok = {(r["meme_id"], r["paraphrase"]) for r in recs if not image_failed(r)}
        with open(path, "a", encoding="utf-8") as out:
            for b in mine:
                st = client.messages.batches.retrieve(b["id"])
                if st.processing_status == "ended":
                    print(f"lote {b['id']} terminou: gravados {collect(client, b, out, have_ok, have_any)}")
                    b["collected"] = True
                    save_state(state)
                else:
                    c = st.request_counts
                    print(f"lote {b['id']}: {st.processing_status} · processando {c.processing} · "
                          f"ok {c.succeeded} · erro {c.errored}")
        pending = {from_custom_id(c) for b in mine if not b["collected"] for c in b["custom_ids"]}
        recs = read_jsonl(path)
        done = {(r["meme_id"], r["paraphrase"]) for r in recs if not image_failed(r)}
        reencode = known_bad | {r["meme_id"] for r in recs if IMAGE_ERROR in (r.get("error") or "")}
        todo = [(i, p) for i in ids for p in (1, 2, 3) if (i, p) not in done and (i, p) not in pending]
        print(f"[{model} {arm}] {3 * len(ids)} pedidos · {len(done & {(i, p) for i in ids for p in (1, 2, 3)})} "
              f"gravados · {len(pending)} em lote · {len(todo)} a enviar")
        if todo and first:              # envia uma vez por execução: erro persistente não vira laço
            submit(client, model, arm, memes, todo, texts, state, reencode, scope)
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
    # doc 31, mesmo modelo e braço (P1–P3 na mensagem user); o último registro vence (reenvio PNG)
    r31 = {(r["meme_id"], r["paraphrase"]): r["label"] for r in doc31_records(f"{model}_{arm}.jsonl")}
    with open(EXTREMES, encoding="utf-8") as f:
        ext = {r["meme_id"]: r for r in csv.DictReader(f)}
    tag = f"{model} {arm}"
    L = [f"# Doc 32 — {tag} com os system prompts S1–S3 nos memes extremos (treino)", "",
         f"Mesmos memes do doc 31; system = S1, S2 ou S3 (prompts_32.py); user = [imagem, <meme_text>]; "
         f"effort {EFFORT}; thinking {json.dumps(thinking_cfg(model, arm))}. Voto = maioria dos 3 prompts, "
         "sobre todos os memes do grupo (sem maioria de 2 conta como não-YES, como no doc 31). "
         "YES Sk = fração de YES entre as respostas parseadas do prompt k. Claude doc 31 = voto com P1–P3 "
         "na mensagem user, mesmo modelo e braço.", "",
         "| grupo | n | humanos | voto YES GLM off | voto YES GLM on | voto YES Claude doc 31 | "
         "voto YES Claude S1–S3 | YES S1 | YES S2 | YES S3 | parse | recusas |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cell = lambda xs: f"{sum(xs) / len(xs):.3f}" if xs else "—"
    for g, k in (("1Y5N", 1), ("3Y3N", 3), ("5Y1N", 5)):
        for lang in ("total", "en", "es"):
            s = [i for i in ids if ext[i]["group"] == g and (lang == "total" or ext[i]["lang"] == lang)]
            labs = {i: [recs[(i, p)]["label"] for p in (1, 2, 3)] for i in s}
            flat = [l for v in labs.values() for l in v]
            old = [majority([r31[(i, p)] for p in (1, 2, 3)]) == "YES" for i in s
                   if all((i, p) in r31 for p in (1, 2, 3))]
            per_p = [cell([labs[i][p - 1] == "YES" for i in s if labs[i][p - 1] in LABELS]) for p in (1, 2, 3)]
            L.append(f"| {g if lang == 'total' else '↳ ' + lang} | {len(s)} | {k / 6:.3f} | "
                     f"{cell([ext[i]['think_off_vote'] == 'YES' for i in s])} | "
                     f"{cell([ext[i]['think_on_vote'] == 'YES' for i in s])} | {cell(old)} | "
                     f"{cell([majority(v) == 'YES' for v in labs.values()])} | {' | '.join(per_p)} | "
                     f"{cell([l in LABELS for l in flat])} | "
                     f"{sum(recs[(i, p)]['parse_method'] == 'refusal' for i in s for p in (1, 2, 3))} |")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"claude32_{model}_{arm}_extremes.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L) + f"\n\ngravado: {out}")


def main():
    global MAX_BATCH_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PRICE))
    ap.add_argument("--arm", required=True, choices=sorted(MAX_TOKENS))
    ap.add_argument("--per-group", type=int, help="só os N primeiros de cada grupo (piloto)")
    ap.add_argument("--smoke", action="store_true", help="1 meme × 3 prompts síncrono; não grava")
    ap.add_argument("--batch-mb", type=int, default=MAX_BATCH_BYTES // 1_000_000,
                    help="tamanho máximo de cada lote em MB (padrão 150); menor = upload e resposta mais curtos, "
                         "para redes que derrubam a conexão enquanto a API processa o lote")
    ap.add_argument("--scope", choices=SCOPES, default="extremes",
                    help="train = os 1 835 memes de treino fora dos extremos (doc 33)")
    a = ap.parse_args()
    MAX_BATCH_BYTES = a.batch_mb * 1_000_000          # só muda o fatiamento dos lotes, não os pedidos
    assert not (a.scope != "extremes" and a.per_group), "--per-group só vale para os extremos"
    thinking_cfg(a.model, a.arm)                     # falha cedo se o modelo não desliga o thinking
    memes = load_memes()
    ids = {"extremes": lambda: select_ids(a.per_group), "train": train_rest_ids,
           "repeat": lambda: select_ids(20)}[a.scope]()     # repeat: os 60 memes do piloto do doc 32
    texts = clean_texts(memes, ids)                  # mesmo texto limpo que o GLM recebeu
    client = anthropic.Anthropic()
    if a.smoke:
        return smoke(client, a.model, a.arm, memes, ids, texts)
    if run(client, a.model, a.arm, memes, ids, texts, a.scope):
        if a.scope == "extremes":
            report(a.model, a.arm, ids)
        else:
            print(f"{a.scope}: {len(ids)} memes × 3 gravados em {raw_path(a.model, a.arm, a.scope)}. Próximo: "
                  f"python3 EXIST_MEMES/glm_annotator/persona_ref.py {'votes' if a.scope == 'train' else 'drift'}")


if __name__ == "__main__":
    main()
