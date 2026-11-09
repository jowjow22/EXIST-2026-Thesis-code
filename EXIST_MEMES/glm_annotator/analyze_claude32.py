"""Doc 32: integridade, voto YES com Wilson 95% (§3) e análises exploratórias da rodada completa.
Só leitura; stdlib apenas. Uso (da raiz do repositório): python3 -I EXIST_MEMES/glm_annotator/analyze_claude32.py"""
import csv, json, math, sys
from collections import Counter
from pathlib import Path

M = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RAW = M / "claude_reference_32/raw/claude-sonnet-5-5_think_off.jsonl"
D31 = M / "claude_reference/raw/claude-sonnet-5-5_think_off.jsonl"
EXT = M / "glm_reference/extremes/extremes.csv"
SHA = {1: "9bdf1e12e1cbc02c296fd902ab208950ef780ffce712c2bce204a488770fb573",
       2: "b545a0951465a19852f5cae9394038fe7625fe1eec899af08b511268751adbcf",
       3: "eeaf482e82f83356186676e38739c112e191df829fcae1d3779f3ae5aa0cc6a6"}

lines = [l for l in RAW.read_text(encoding="utf-8").splitlines() if l]
recs = [json.loads(l) for l in lines]
ext = {r["meme_id"]: r for r in csv.DictReader(open(EXT, encoding="utf-8"))}
ids = list(ext)
keys = [(r["meme_id"], r["paraphrase"]) for r in recs]

print("== integridade ==")
print("registros", len(recs), "· chaves únicas", len(set(keys)), "· esperado", 3 * len(ids))
assert set(keys) == {(i, p) for i in ids for p in (1, 2, 3)}
print("parse_method", Counter(r["parse_method"] for r in recs))
print("finish_reason", Counter(r["finish_reason"] for r in recs))
print("label", Counter(r["label"] for r in recs))
print("content", Counter(r["content"] for r in recs).most_common(5))
print("erros", sum(r["error"] is not None for r in recs), "· recusas", sum(r["refusal_category"] is not None for r in recs))
print("SHA confere", all(r["prompt_sha256"] == SHA[r["paraphrase"]] for r in recs),
      "· system", Counter((r["paraphrase"], r["params"]["system"]) for r in recs))
print("server_model", Counter(r["server_model"] for r in recs))
re_ = {r["meme_id"] for r in recs if r["image_reencoded"]}
print("reencoded memes", len(re_), "· registros", sum(r["image_reencoded"] for r in recs))
print("max output tokens", max(r["usage"]["output_tokens"] for r in recs),
      "· output", Counter(r["usage"]["output_tokens"] for r in recs))
print("lotes", Counter(r["batch_id"] for r in recs))
print("registros do piloto reaproveitados", sum(r["batch_id"] == "msgbatch_01D9QLhJxKPKXqipS2DxN2Pt" for r in recs))
tin = sum(r["usage"]["input_tokens"] for r in recs); tout = sum(r["usage"]["output_tokens"] for r in recs)
cost = (tin * 2 + tout * 10) / 1e6 / 2
pil = [r for r in recs if r["batch_id"] == "msgbatch_01D9QLhJxKPKXqipS2DxN2Pt"]
cpil = (sum(r["usage"]["input_tokens"] for r in pil) * 2 + sum(r["usage"]["output_tokens"] for r in pil) * 10) / 1e6 / 2
print(f"tokens in {tin} · out {tout} · custo batch total US$ {cost:.2f} (piloto {cpil:.2f}, rodada {cost - cpil:.2f})")
for p in (1, 2, 3):
    xs = [r["usage"]["input_tokens"] for r in recs if r["paraphrase"] == p]
    print(f"  S{p} input médio {sum(xs) / len(xs):.0f}")

lab = {(r["meme_id"], r["paraphrase"]): r["label"] for r in recs}
d31 = {}
for l in D31.read_text(encoding="utf-8").splitlines():
    if l:
        r = json.loads(l); d31[(r["meme_id"], r["paraphrase"])] = r["label"]   # último vence (reenvio PNG)


def maj(v):
    for x in ("YES", "NO"):
        if sum(y == x for y in v) >= 2:
            return x
    return None


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


v32 = {i: maj([lab[(i, p)] for p in (1, 2, 3)]) for i in ids}
v31 = {i: maj([d31[(i, p)] for p in (1, 2, 3)]) for i in ids}
print("\nsem maioria: doc32", sum(v is None for v in v32.values()), "· doc31", sum(v is None for v in v31.values()))
print("\n== voto YES com Wilson 95% (pré-registrado: §3) ==")
for g, k in (("1Y5N", 1), ("3Y3N", 3), ("5Y1N", 5)):
    for L in ("total", "en", "es"):
        s = [i for i in ids if ext[i]["group"] == g and (L == "total" or ext[i]["lang"] == L)]
        n = len(s); y = sum(v32[i] == "YES" for i in s); y31 = sum(v31[i] == "YES" for i in s)
        lo, hi = wilson(y, n); lo31, hi31 = wilson(y31, n)
        pp = []
        for p in (1, 2, 3):
            yp = sum(lab[(i, p)] == "YES" for i in s); a, b_ = wilson(yp, n)
            pp.append(f"S{p} {yp / n:.3f} [{a:.3f}–{b_:.3f}]")
        print(f"{g:5} {L:5} n={n:3} humanos {k / 6:.3f} · S1–S3 {y / n:.3f} [{lo:.3f}–{hi:.3f}] · doc31 {y31 / n:.3f} "
              f"[{lo31:.3f}–{hi31:.3f}] · " + " · ".join(pp))

print("\n== exploratório (não pré-registrado) ==")
print("pareado doc31 → doc32 por meme (voto): b = doc31 YES→doc32 não-YES, c = doc31 não-YES→doc32 YES")
for g in ("1Y5N", "3Y3N", "5Y1N"):
    s = [i for i in ids if ext[i]["group"] == g]
    b = sum(v31[i] == "YES" and v32[i] != "YES" for i in s); c = sum(v31[i] != "YES" and v32[i] == "YES" for i in s)
    same = len(s) - b - c
    print(f"  {g}: concordam {same}/{len(s)} ({same / len(s):.3f}) · b={b} · c={c} · McNemar exato p={mcnemar(b, c):.2g}")
print("\nseparação entre extremos (exploratório): TPR = YES em 5Y1N, TNR = 1 − YES em 1Y5N")
neg = [i for i in ids if ext[i]["group"] == "1Y5N"]; pos = [i for i in ids if ext[i]["group"] == "5Y1N"]


def ba(fn):
    tpr = sum(fn(i) for i in pos) / len(pos); tnr = 1 - sum(fn(i) for i in neg) / len(neg)
    return tpr, tnr, (tpr + tnr) / 2


rows = [("Claude S1–S3 voto", lambda i: v32[i] == "YES"), ("Claude doc31 voto", lambda i: v31[i] == "YES"),
        ("GLM off voto", lambda i: ext[i]["think_off_vote"] == "YES"), ("GLM on voto", lambda i: ext[i]["think_on_vote"] == "YES")]
rows += [(f"Claude S{p}", (lambda p: lambda i: lab[(i, p)] == "YES")(p)) for p in (1, 2, 3)]
rows += [(f"Claude doc31 P{p}", (lambda p: lambda i: d31[(i, p)] == "YES")(p)) for p in (1, 2, 3)]
for name, fn in rows:
    t, n_, b_ = ba(fn); print(f"  {name:20} TPR {t:.3f} · TNR {n_:.3f} · acurácia balanceada {b_:.3f}")


def auc(score):
    """AUC (Mann–Whitney) 5Y1N vs 1Y5N com score ordinal = nº de YES em 3 respostas."""
    w = 0.0
    for a in pos:
        for b in neg:
            w += 1.0 if score(a) > score(b) else 0.5 if score(a) == score(b) else 0.0
    return w / (len(pos) * len(neg))


print(f"  AUC nº de YES (0–3): S1–S3 {auc(lambda i: sum(lab[(i, p)] == 'YES' for p in (1, 2, 3))):.3f} · "
      f"doc31 {auc(lambda i: sum(d31[(i, p)] == 'YES' for p in (1, 2, 3))):.3f} · "
      f"GLM off {auc(lambda i: sum(ext[i][f'think_off_p{p}'] == 'YES' for p in (1, 2, 3))):.3f} · "
      f"GLM on {auc(lambda i: sum(ext[i][f'think_on_p{p}'] == 'YES' for p in (1, 2, 3))):.3f}")

print("\nconcordância entre as 3 respostas (exploratório): S1–S3 (doc 32) e P1–P3 (doc 31)")
for name, L_ in (("doc32 S1–S3", lab), ("doc31 P1–P3", d31)):
    for g in ("1Y5N", "3Y3N", "5Y1N", "todos"):
        s = [i for i in ids if g == "todos" or ext[i]["group"] == g]
        ny = {i: sum(L_[(i, p)] == "YES" for p in (1, 2, 3)) for i in s}
        cnt = Counter(ny.values())
        # Fleiss κ, 3 respostas, 2 categorias (todas parseadas: YES + NO = 3)
        Pb = sum((y * (y - 1) + (3 - y) * (2 - y)) / 6 for y in ny.values()) / len(s)
        pY = sum(ny.values()) / (3 * len(s)); Pe = pY ** 2 + (1 - pY) ** 2
        print(f"  {name} {g:5} nº de YES 0/1/2/3 = {cnt[0]}/{cnt[1]}/{cnt[2]}/{cnt[3]} · "
              f"unânime {(cnt[0] + cnt[3]) / len(s):.3f} · Fleiss κ {(Pb - Pe) / (1 - Pe):.3f} · YES por resposta {pY:.3f}")
