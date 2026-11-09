"""Teste offline do doc 33: nenhuma chamada à API, nada gravado no repositório (saídas e config.json em
pasta temporária).

1. equivalência com o doc 27: persona_ref alimentado com os votos do GLM reproduz pesos, soft labels,
   checagens (a)–(i), G4, vote/confident e o G2 do doc 25;
2. votes: junta extremos (doc 32, só leitura) + treino sintético; barra cobertura incompleta e reenvio
   PNG pendente;
3. ponta a ponta com config.json temporário: votes → build → checks; build congelado é idempotente e
   retoma depois de uma queda; checks recusa votos trocados depois do build;
4. claude_reference_32 --scope train: arquivo próprio, escopos não se misturam, lote perdido na criação é
   recuperado da API sem pagar de novo, registros perdidos são baixados de novo, reenvio PNG é gravado.

Uso (da raiz do repositório): python3 EXIST_MEMES/glm_annotator/tests/test_persona33_offline.py
"""
import csv
import datetime
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

GA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GA))
import claude_reference_32 as C  # noqa: E402  (antes do build, que mexe no sys.path)
import common  # noqa: E402
import persona_ref as P  # noqa: E402
from common import REF, REPORTS, SOFT_GLMW, SOFT_RAW, TRAIN_IDS, VAL_IDS, load_config, load_json, load_memes  # noqa: E402
from prompts_32 import SYSTEM_SHA256  # noqa: E402

tmp = Path(tempfile.mkdtemp())
memes, train, val, raw = load_memes(), load_json(TRAIN_IDS), load_json(VAL_IDS), load_json(SOFT_RAW)
protected = {p: p.read_bytes() for p in (REF / "glm_votes.csv", REF / "annotator_weights.csv", SOFT_GLMW,
                                          REF / "config.json", REPORTS / "checks_summary.md", P.RAWS[0])}
glm = {r["meme_id"]: r for r in csv.DictReader(open(REF / "glm_votes.csv", encoding="utf-8"))}
ext = {r["meme_id"] for r in csv.DictReader(open(P.EXTREMES, encoding="utf-8"))}


def table_rows(md: str, title: str) -> list:
    """Linhas de dados das tabelas de uma seção '## (x)' de um relatório."""
    sec = md.split(f"## {title}", 1)[1].split("\n## ", 1)[0]
    return [l for l in sec.splitlines() if l.startswith("| ") and not l.startswith("| escopo") and "---" not in l
            and not l.startswith("| gênero")]


# 1. equivalência com o doc 27 -------------------------------------------------------------------
glm_csv = tmp / "glm_as_persona.csv"
with open(glm_csv, "w", newline="", encoding="utf-8") as g:
    w = csv.DictWriter(g, fieldnames=P.FIELDS)
    w.writeheader()
    for i in train:
        r = glm[i]
        w.writerow({**{k: r[k] for k in P.FIELDS[:5]}, "vote": r["glm_vote"], "confident": r["glm_confident"]})
rows, soft, full = P.build_core(glm_csv, memes, train, val, raw)
ref = {r["annotator_id"]: r for r in csv.DictReader(open(REF / "annotator_weights.csv", encoding="utf-8"))}
assert set(ref) == {x["annotator_id"] for x in rows}
for x in rows:
    for k in ("D_glm", "r_glm", "D_peer", "r_peer"):
        a, b = x[k], ref[x["annotator_id"]][k]
        assert (a is None and b == "") or float(b) == a, (x["annotator_id"], k, a, b)
assert soft == load_json(REF / "soft_labels_weighted.json"), "soft labels de treino ≠ doc 27"
assert full == load_json(SOFT_GLMW), "soft_labels_taskA_glmw.json ≠ reconstrução"
print(f"ok 1a: build_core com os votos do GLM reproduz o doc 27 ({len(rows)} anotadores, {len(full)} soft labels)")

wcsv = tmp / "w.csv"                                   # com os nomes da Persona (caminho real do checks)
with open(wcsv, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(P.rename(rows[0])))
    w.writeheader()
    w.writerows(P.rename(x) for x in rows)
L, res = P.checks_core(glm_csv, wcsv, soft, memes, train, glm_weights_csv=REF / "annotator_weights.csv",
                       raw=raw, n_perm=3)
g4 = load_config()["runtime"]["gates"]["G4"]
assert abs(res["share_moved"] - g4["share_moved"]) < 1e-12 and abs(res["rho_a"] - g4["rho_r_glm_r_peer"]) < 1e-12
old = (REPORTS / "checks_summary.md").read_text(encoding="utf-8")
new = "\n".join(L)
for sec in ("(c)", "(e)", "(f)", "(h)"):
    assert table_rows(new, sec) == table_rows(old, sec), f"tabela {sec} ≠ doc 27"
cut = lambda rows_: [r.rsplit("|", 2)[0] for r in rows_]                            # noqa: E731  (sem a coluna Δp)
assert cut(table_rows(new, "(d)")) == [r.rsplit("|", 1)[0] for r in table_rows(old, "(d)")], "tabela (d) ≠ doc 27"
assert [r.replace("| S", "| P") for r in table_rows(new, "(i)")] == table_rows(old, "(i)"), "tabela (i) ≠ doc 27"
rho_b_old = float(re.search(r"ρ\(D_glm, taxa de YES do anotador[^*]*\*\*(-?[\d.]+)\*\*", old).group(1))
assert abs(res["rho_b_yes_rate"] - rho_b_old) < 5e-5
assert res["band"] and len(res["parcels"]) == len(rows) and "Persona × GLM" in new
print("ok 1b: checks_core (pesos com nomes da Persona) reproduz G4, (b) e as tabelas (c)–(i) do doc 27; (j) e (k) rodam")

fake_recs = {(i, p): {"label": glm[i][f"label_p{p}"] if glm[i][f"label_p{p}"] in ("YES", "NO") else "?",
                      "prompt_sha256": SYSTEM_SHA256[f"S{p}"]} for i in train for p in (1, 2, 3)}
vr = P.vote_rows(fake_recs, train, memes)
assert all(r["vote"] == glm[r["meme_id"]]["glm_vote"] and r["confident"] == glm[r["meme_id"]]["glm_confident"]
           for r in vr), "vote/confident ≠ glm_votes.csv"
g2 = P.gate_g2(vr)
g2_old = load_config()["runtime"]["gates"]["G2"]["stats"]
for lang in ("en", "es"):
    assert abs(g2["by_lang"][lang]["parse_rate"] - g2_old[lang]["parse_rate"]) < 1e-12
    assert abs(g2["by_lang"][lang]["confident"] - g2_old[lang]["confident_frac"]) < 1e-12
assert g2["pass"] is False
print("ok 1c: vote_rows e gate_g2 nos rótulos do GLM reproduzem glm_votes.csv e o G2 do doc 25 (falha)")


# 2. votes ----------------------------------------------------------------------------------------
def write_fake_train(path, rows_override=None):
    """Resto do treino com o voto do GLM repetido 3 vezes (G2 passa), formato do JSONL real."""
    with open(path, "w", encoding="utf-8") as fh:
        for i in train:
            if i in ext:
                continue
            for p in (1, 2, 3):
                rec = {"meme_id": i, "paraphrase": p, "label": glm[i]["glm_vote"] or "NO", "parse_method": "plain",
                       "prompt_sha256": SYSTEM_SHA256[f"S{p}"], "image_reencoded": False, "error": None,
                       "usage": {"input_tokens": 1000, "output_tokens": 4}}
                fh.write(json.dumps({**rec, **(rows_override or {}).get((i, p), {})}) + "\n")


fake = tmp / "fake_train.jsonl"
write_fake_train(fake)
out = tmp / "persona_votes.csv"
g2v = P.votes(raws=(P.RAWS[0], fake), out=out, save=False)
vcsv = {r["meme_id"]: r for r in csv.DictReader(open(out, encoding="utf-8"))}
d32 = P.read_records(P.RAWS[:1])
assert len(vcsv) == 3386 and g2v["pass"]
assert all(vcsv[i][f"label_p{p}"] == d32[(i, p)]["label"] for i in ext for p in (1, 2, 3))
for bad_raws, msg in (((P.RAWS[0],), "cobertura incompleta"), (None, "reenvio PNG pendente")):
    if bad_raws is None:
        k = next(i for i in train if i not in ext)
        pend = tmp / "pend.jsonl"
        write_fake_train(pend, {(k, 1): {"label": "?", "parse_method": "error",
                                         "error": "invalid_request_error: Could not process image"}})
        bad_raws = (P.RAWS[0], pend)
    try:
        P.votes(raws=bad_raws, out=tmp / "x.csv", save=False)
        raise SystemExit(f"FALHA: votes aceitou {msg}")
    except AssertionError:
        pass
print("ok 2: votes junta extremos + resto do treino; barra cobertura incompleta e reenvio PNG pendente")

# 3. ponta a ponta com config.json temporário -----------------------------------------------------
cfg_tmp = tmp / "config.json"
shutil.copyfile(REF / "config.json", cfg_tmp)
common.CONFIG_PATH = P.CONFIG_PATH = cfg_tmp
for name in ("OUT", "REPORTS"):
    setattr(P, name, tmp / name.lower())
P.VOTES, P.WEIGHTS, P.SOFT_W = P.OUT / "persona_votes.csv", P.OUT / "w.csv", P.OUT / "sw.json"
P.PARCELS, P.SOFT_PERSONAW, P.SOFT_SHIFT = P.OUT / "parcels.csv", tmp / "personaw.json", tmp / "personashift.json"
P.N_PERM = 3
before = json.loads(cfg_tmp.read_text())
assert P.votes(raws=(P.RAWS[0], fake), out=P.VOTES)["pass"]
real_write, calls = P.write_json, []


def crash_on_config(path, obj):                  # queda da montagem na escrita do config.json do build
    calls.append(Path(path).name)
    if Path(path) == cfg_tmp:
        raise OSError("Input/output error (montagem caiu)")
    real_write(path, obj)


P.write_json = crash_on_config
try:
    P.build()
    raise SystemExit("FALHA: a queda simulada não aconteceu")
except OSError:
    pass
P.write_json = real_write
assert P.SOFT_PERSONAW.exists() and "build" not in json.loads(cfg_tmp.read_text())["runtime"]["doc33"]
P.build()                                         # retoma: sem SHA congelado, reconstrói
P.build()                                         # congelado: só confere
cfg = json.loads(cfg_tmp.read_text())
d33 = cfg["runtime"]["doc33"]
assert d33["sha256"] and d33["control"]["sha256"] and cfg["runtime"]["gates"]["doc33_F3"]["pass"]
try:
    P.checks()
except SystemExit:
    pass                                          # G4 pode falhar com votos sintéticos; o que importa é rodar
cfg = json.loads(cfg_tmp.read_text())
assert "doc33_G4" in cfg["runtime"]["gates"] and P.PARCELS.exists()
for k in set(before["runtime"]) - {"doc33", "gates"}:
    assert cfg["runtime"][k] == before["runtime"][k], f"runtime.{k} do GLM mudou"
for k, v in before["runtime"]["gates"].items():
    assert cfg["runtime"]["gates"][k] == v, f"gate {k} do GLM mudou"
try:
    P.votes(raws=(P.RAWS[0], fake), out=P.VOTES)
    raise SystemExit("FALHA: votes regravou depois do build")
except AssertionError:
    pass
P.VOTES.write_text(P.VOTES.read_text() + "\n")    # votos trocados por fora depois do build
try:
    P.checks()
    raise SystemExit("FALHA: checks aceitou votos trocados depois do build")
except AssertionError:
    pass
print("ok 3: votes → build → checks em config temporário; build retoma após queda e fica idempotente; "
      "votes congelados; checks recusa votos trocados; nada do GLM mudou no config")

# 4. claude_reference_32 ---------------------------------------------------------------------------
rest = C.train_rest_ids()
assert len(rest) == 1835 and not set(rest) & ext and set(rest) | ext == set(train)
C.OUT, C.RAW, C.STATE, C.POLL_S = tmp / "c32", tmp / "c32" / "raw", tmp / "c32" / "batches.json", 0
M, A = "claude-sonnet-5-5", "think_off"


class FakeBatches:
    """API falsa: guarda os lotes; `lose_reply` simula o POST aceito cuja resposta não volta."""
    def __init__(self):
        self.sent, self.lose_reply, self.errors = {}, False, set()

    def create(self, requests):
        bid = f"msgbatch_fake{len(self.sent) + 1}"
        self.sent[bid] = (list(requests), datetime.datetime.now(datetime.timezone.utc))
        if self.lose_reply:
            self.lose_reply = False
            raise KeyboardInterrupt("Ctrl+C durante o upload")
        return NS(id=bid)

    def list(self, limit=20):
        for bid, (reqs, t) in reversed(self.sent.items()):
            yield NS(id=bid, created_at=t, request_counts=NS(processing=0, succeeded=len(reqs), errored=0,
                                                             canceled=0, expired=0))

    def retrieve(self, bid):
        return NS(processing_status="ended", request_counts=NS(processing=0, succeeded=len(self.sent[bid][0]), errored=0))

    def results(self, bid):
        for q in self.sent[bid][0]:
            if q["custom_id"] in self.errors and q["params"]["messages"][0]["content"][0]["source"]["media_type"] != "image/png":
                yield NS(custom_id=q["custom_id"], result=NS(type="errored", error=NS(
                    type="invalid_request_error", message=C.IMAGE_ERROR)))
                continue
            yield NS(custom_id=q["custom_id"], result=NS(type="succeeded", message=NS(
                content=[NS(type="text", text="NO")], stop_reason="end_turn", stop_details=None,
                usage=NS(input_tokens=1500, output_tokens=4), model=M)))


api = FakeBatches()
client = NS(messages=NS(batches=api))
client.with_options = lambda **_: client
sub = rest[:3]
texts = {i: "hey" for i in sub}
tr = C.raw_path(M, A, "train")

api.lose_reply = True                             # (a) Ctrl+C depois de o servidor aceitar o lote
try:
    C.run(client, M, A, memes, sub, texts, scope="train")
    raise SystemExit("FALHA: interrupção simulada não aconteceu")
except KeyboardInterrupt:
    pass
st = json.loads(C.STATE.read_text())
assert len(st) == 1 and st[0]["id"] is None and len(api.sent) == 1
assert C.run(client, M, A, memes, sub, texts, scope="train") is True
st = json.loads(C.STATE.read_text())
assert len(api.sent) == 1, f"pagou de novo: {len(api.sent)} lotes na API"
assert [b["id"] for b in st] == ["msgbatch_fake1"] and st[0]["collected"] and st[0]["scope"] == "train"
assert len(tr.read_text().splitlines()) == 9 and not C.raw_path(M, A).exists()
print("ok 4a: lote criado com a resposta perdida é recuperado da API, sem segundo pagamento")

lines = tr.read_text().splitlines()               # (b) registros somem depois de o lote ser recolhido
tr.write_text("\n".join(lines[:4]) + "\n" + lines[4][:20])
assert C.run(client, M, A, memes, sub, texts, scope="train") is True
assert len(api.sent) == 1, "reenviou em vez de baixar de novo"
recs = [json.loads(l) for l in tr.read_text().splitlines()]
assert {(r["meme_id"], r["paraphrase"]) for r in recs} == {(i, p) for i in sub for p in (1, 2, 3)} and len(recs) == 9
print("ok 4b: registros perdidos são baixados de novo (sem reenvio) e a linha cortada é aparada")

jpg = next(i for i in rest[3:] if common.image_path(memes[i]).suffix.lower() in (".jpg", ".jpeg"))
api.errors = {C.custom_id(jpg, 1)}               # (c) erro de imagem num JPEG → reenvio PNG gravado
sub2 = [jpg]
assert C.run(client, M, A, memes, sub2, {jpg: "hey"}, scope="train") is False
assert C.run(client, M, A, memes, sub2, {jpg: "hey"}, scope="train") is True
last = {}
for line in tr.read_text().splitlines():
    r = json.loads(line)
    last[(r["meme_id"], r["paraphrase"])] = r
assert last[(jpg, 1)]["image_reencoded"] and last[(jpg, 1)]["label"] == "NO"
assert C.run(client, M, A, memes, sub2, {jpg: "hey"}, scope="train") is True and len(api.sent) == 3
print("ok 4c: erro de imagem → reenvio PNG uma vez, registro PNG fica por último, sem novo envio depois")

C.CREATE_WAIT_S = 0                                # (e) conexão cai na criação: o script resolve sozinho
import anthropic  # noqa: E402
conn_err = anthropic.APIConnectionError(request=None)
real_create = api.create
for created in (True, False):                      # o servidor criou o lote / não criou
    n_before, calls = len(api.sent), []

    def flaky(requests, created=created):
        calls.append(1)
        if len(calls) == 1:
            if created:
                real_create(requests)              # aceito no servidor, resposta perdida
            raise conn_err
        return real_create(requests)
    api.create = flaky
    k = rest[5 + int(created)]
    assert C.run(client, M, A, memes, [k], {k: "hey"}, scope="train") is True
    assert len(api.sent) == n_before + 1, f"created={created}: {len(api.sent) - n_before} lotes na API"
    assert all(b["id"] for b in json.loads(C.STATE.read_text())), "ficou intenção sem id"
api.create = real_create
print("ok 4e: conexão caída na criação é resolvida no mesmo comando (adota ou reenvia), um lote só na API")

st = json.loads(C.STATE.read_text())               # (d) escopos não se misturam
st.append({**st[0], "id": "msgbatch_pendente_train", "collected": False})
C.save_state(st)
assert C.run(client, M, A, memes, [], {}, scope="extremes") is True
assert not json.loads(C.STATE.read_text())[-1]["collected"], "escopo extremos recolheu lote de treino"
print("ok 4d: escopo extremos não toca lote de treino")

for p, b in protected.items():
    assert p.read_bytes() == b, f"{p.name} foi modificado"
print("ok 5: votos do GLM, pesos do doc 27, glmw, config.json, checks_summary.md e JSONL do doc 32 intocados")
print("\nTODOS OS TESTES PASSARAM")
