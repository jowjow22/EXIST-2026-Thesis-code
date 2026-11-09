"""Teste offline de claude_reference_32.py: nenhuma chamada à API (cliente falso), nada gravado
no repositório (OUT/RAW/STATE/REPORTS redirecionados para uma pasta temporária).
Uso (da raiz do repositório): python3 EXIST_MEMES/glm_annotator/tests/test_claude32_offline.py"""
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

GA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GA))
import claude_reference_32 as C  # noqa: E402

M, ARM = "claude-sonnet-5-5", "think_off"
doc31_raw = C.DOC31_RAW / f"{M}_{ARM}.jsonl"
doc31_bytes = doc31_raw.read_bytes()

# 1. memes e reencode herdado do doc 31
ids = C.select_ids(None)
assert len(ids) == 1551 and len(C.select_ids(20)) == 60
bad = C.doc31_image_failures()
assert len(bad) == 21, len(bad)
print("ok 1: 1 551 memes, piloto 60, 21 memes PNG herdados do doc 31")

# 2. formato do pedido
memes = C.load_memes()
i0, b = ids[0], sorted(bad)[0]
b64, mt, _ = C.image_b64(memes[i0])
for p in (1, 2, 3):
    r = C.request_params(M, ARM, p, b64, mt, "hey girl")
    assert r["system"] == C.SYSTEM[p] and set(r) == {"model", "max_tokens", "thinking", "output_config", "system", "messages"}
    (msg,) = r["messages"]
    assert msg["role"] == "user" and [x["type"] for x in msg["content"]] == ["image", "text"]
    assert msg["content"][1]["text"] == "<meme_text>\nhey girl\n</meme_text>"
    assert r["thinking"] == {"type": "between_tools"} and r["output_config"] == {"effort": "medium"} and r["max_tokens"] == 256
assert C.request_params(M, "think_on", 1, b64, mt, "x")["thinking"] == {"type": "adaptive", "display": "summarized"}
try:
    C.request_params(M, ARM, 1, b64, mt, "  \n")
    raise SystemExit("FALHA: texto vazio passou")
except AssertionError:
    pass
print("ok 2: system = S1/S2/S3, user = [imagem, <meme_text>], sem temperature/top_p/top_k, texto vazio barrado")

# 3. ciclo completo com cliente falso: envia, recolhe, retoma sem reenviar
tmp = Path(tempfile.mkdtemp())
C.OUT, C.RAW, C.STATE, C.REPORTS = tmp, tmp / "raw", tmp / "batches.json", tmp / "reports"
C.POLL_S = 0


class FakeBatches:
    def __init__(self):
        self.sent = {}

    def create(self, requests):
        bid = f"msgbatch_fake{len(self.sent) + 1}"
        self.sent[bid] = list(requests)
        return NS(id=bid)

    def retrieve(self, bid):
        return NS(processing_status="ended", request_counts=NS(processing=0, succeeded=len(self.sent[bid]), errored=0))

    def results(self, bid):
        for q in self.sent[bid]:
            ans = "YES" if q["custom_id"].endswith("_p1") else "NO"
            yield NS(custom_id=q["custom_id"], result=NS(type="succeeded", message=NS(
                content=[NS(type="text", text=ans)], stop_reason="end_turn", stop_details=None,
                usage=NS(input_tokens=1500, output_tokens=1), model=M)))


fake = NS(messages=NS(batches=FakeBatches()))
fake.with_options = lambda **_: fake      # o envio de lote desliga o retry do SDK (doc 33)
sub = [i0, b]
texts = {i0: "hey girl", b: "honey nut"}
assert C.run(fake, M, ARM, memes, sub, texts) is True
(bid, reqs), = fake.messages.batches.sent.items()
assert [q["custom_id"] for q in reqs] == [f"{i}_p{p}" for i in sub for p in (1, 2, 3)]
media = {q["custom_id"]: q["params"]["messages"][0]["content"][0]["source"]["media_type"] for q in reqs}
assert all(media[f"{b}_p{p}"] == "image/png" for p in (1, 2, 3)), media
assert [q["params"]["system"] for q in reqs] == [C.SYSTEM[p] for _ in sub for p in (1, 2, 3)]
recs = [json.loads(line) for line in C.raw_path(M, ARM).read_text().splitlines()]
assert len(recs) == 6
for r in recs:
    p = r["paraphrase"]
    import prompts_32
    assert r["prompt_sha256"] == prompts_32.SYSTEM_SHA256[f"S{p}"]
    assert r["params"]["system"] == f"S{p}" and r["label"] == ("YES" if p == 1 else "NO")
    assert r["image_reencoded"] == (r["meme_id"] == b) and r["batch_id"] == bid
state = json.loads(C.STATE.read_text())
assert len(state) == 1 and state[0]["collected"] and state[0]["reencoded"] == [b]
assert C.run(fake, M, ARM, memes, sub, texts) is True and len(fake.messages.batches.sent) == 1
print("ok 3: envio → recolha → 6 registros; PNG no meme herdado; segunda execução não reenvia nada")

# 4. relatório: com as respostas do doc 31 no lugar das novas, as duas colunas de voto têm de coincidir
#    e bater com o relatório publicado do doc 31 (1Y5N = 0,286)
C.raw_path(M, ARM).write_bytes(doc31_bytes)
C.report(M, ARM, ids)
md = (C.REPORTS / f"claude32_{M}_{ARM}_extremes.md").read_text()
rows = [line.split("|")[1:-1] for line in md.splitlines() if line.startswith("| ") and "grupo" not in line]
for row in rows:
    c = [x.strip() for x in row]
    assert c[5] == c[6], row                    # voto doc 31 == voto "S1–S3" (mesmos dados)
one = next(c for c in ([x.strip() for x in r] for r in rows) if c[0] == "1Y5N")
assert one[1] == "388" and one[5] == "0.286", one
pub = (C.MEMES_DIR / "glm_reference/reports" / f"claude_{M}_{ARM}_extremes.md")
if pub.exists():
    old = {r[0]: r for r in ([x.strip() for x in line.split("|")[1:-1]] for line in pub.read_text().splitlines()
                              if re.match(r"\| (1Y5N|3Y3N|5Y1N|↳)", line))}
    print("relatório publicado do doc 31 lido:", len(old), "linhas (rótulos ↳ repetem; comparação só nas totais)")
    for c in ([x.strip() for x in r] for r in rows):
        if c[0] in ("1Y5N", "3Y3N", "5Y1N"):
            assert old[c[0]][5] == c[6], (old[c[0]], c)
print("ok 4: relatório reproduz o doc 31 (1Y5N 0,286) quando alimentado com os dados do doc 31")
print(md)

assert doc31_raw.read_bytes() == doc31_bytes
print("ok 5: JSONL do doc 31 intocado")
print("\nTODOS OS TESTES PASSARAM")
