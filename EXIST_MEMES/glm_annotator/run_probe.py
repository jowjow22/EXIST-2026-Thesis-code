"""F0 (§5.2): split, piloto, config.json e testes de fumaça G0.1–G0.4.

Uso (da raiz do repositório):
    python3 EXIST_MEMES/glm_annotator/run_probe.py --concurrency 4
    # revisar glm_reference/probe_vision.csv à mão (G0.3) e então:
    python3 EXIST_MEMES/glm_annotator/run_probe.py --approve-vision

Nenhuma chamada desta fase vai para raw/<arm>.jsonl.
"""
import argparse
import asyncio
import csv
import inspect
import io
import statistics
import sys
import time

import numpy as np
from PIL import Image

from client import chat, log_call, make_client, now, request_params, response_fields, server_info
from common import (CONFIG_PATH, FIXED, MODEL, PILOT_IDS, REF, ROOT, SOFT_RAW, SPLIT_PATH,
                    TRAIN_IDS, VAL_IDS, assert_config, clean_texts, data_url, image_data_url,
                    image_path, load_config, load_json, load_memes, set_gate, update_runtime,
                    write_json)
from prompts import PROBE_PROMPT, render


def export_split(memes: dict) -> list:
    """Reproduz o split do memes_model.ipynb e exige igualdade com o congelado."""
    from reliability.make_soft_labels_taskA import build_frozen_split
    frozen = load_json(SPLIT_PATH)
    train, val, _ = build_frozen_split(list(memes), load_json(SOFT_RAW))
    assert train == frozen["train_ids"] and val == frozen["val_ids"], \
        "split recalculado difere de split_taskA_frozen.json"
    assert (len(train), len(val)) == (3386, 598), (len(train), len(val))
    assert not set(train) & set(val), "treino ∩ validação ≠ ∅ (portão 12.3)"
    write_json(TRAIN_IDS, train)
    write_json(VAL_IDS, val)
    return train


def select_pilot(ids, langs) -> list:
    """Portão 12.2: só ids e idiomas entram aqui, nunca votos."""
    rng = np.random.default_rng(FIXED["pilot"]["rng"])
    out = []
    for lang in FIXED["pilot"]["langs"]:
        pool = sorted(i for i, l in zip(ids, langs) if l == lang)
        out += sorted(rng.choice(pool, size=FIXED["pilot"]["per_lang"], replace=False).tolist())
    return out


assert list(inspect.signature(select_pilot).parameters) == ["ids", "langs"]


def white_png_url() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), "white").save(buf, "PNG")
    return data_url(buf.getvalue(), "image/png")


def init_config(concurrency: int) -> None:
    if CONFIG_PATH.exists():
        cfg = assert_config()
        assert cfg["runtime"]["concurrency"] == concurrency, \
            f"concorrência gravada é {cfg['runtime']['concurrency']}; não muda depois da F0"
        return
    write_json(CONFIG_PATH, {"fixed": FIXED,
                             "runtime": {"concurrency": concurrency, "created": now()}})


async def smoke(memes, pilot, texts, concurrency):
    client = make_client()
    done = {}

    async def call(tag, total, meme_id, **kw):
        """chat() com uma linha de log por request (✓ resposta / ✗ erro)."""
        try:
            r, lat = await chat(client, **kw)
        except Exception as e:
            done[tag] = done.get(tag, 0) + 1
            log_call(tag, done[tag], total, meme_id, False, repr(e))
            raise
        done[tag] = done.get(tag, 0) + 1
        log_call(tag, done[tag], total, meme_id, True, response_fields(r)["content"])
        return r, lat

    # G0.2 — thinking desligado: P1 nos 3 primeiros memes do piloto
    g02 = []
    for i in pilot[:3]:
        url, _ = image_data_url(memes[i])
        r, lat = await call("G0.2", 3, i, image_url=url, text=render(1, texts[i]), arm="think_off",
                            params=request_params("think_off", FIXED["seed_reference"]))
        rf = response_fields(r)
        g02.append({"meme_id": i, "reasoning_chars": len(rf["reasoning_content"] or ""),
                    "completion_tokens": rf["usage"]["completion_tokens"], "content": rf["content"],
                    "latency_s": round(lat, 3)})
    off_ok = all(g["reasoning_chars"] == 0 and g["completion_tokens"] <= 32 for g in g02)

    # G0.3 — sonda visual (imagem real vs PNG branco), com a concorrência registrada
    arm = "think_off" if off_ok else "think_on"
    sem, lats, white = asyncio.Semaphore(concurrency), [], white_png_url()

    async def describe(url, tag, meme_id):
        async with sem:
            r, lat = await call(tag, 20, meme_id, image_url=url, text=PROBE_PROMPT, arm=arm,
                                params=request_params(arm, FIXED["seed_reference"]))
        lats.append(lat)
        return response_fields(r)["content"]

    ids20 = pilot[:20]
    t0 = time.monotonic()
    descs = await asyncio.gather(*[describe(image_data_url(memes[i])[0], "G0.3 real", i) for i in ids20],
                                 *[describe(white, "G0.3 branca", i) for i in ids20])
    wall = time.monotonic() - t0
    await client.close()
    rows = [{"meme_id": i, "image_path": str(image_path(memes[i]).relative_to(ROOT)),
             "desc_real": descs[k], "desc_white": descs[20 + k]} for k, i in enumerate(ids20)]
    g04 = {"calls": len(descs), "arm": arm, "concurrency": concurrency,
           "median_latency_s": round(statistics.median(lats), 3),
           "calls_per_min": round(len(descs) / wall * 60, 1)}
    return g02, off_ok, rows, g04


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--approve-vision", action="store_true",
                    help="registra a aprovação manual do G0.3 depois de revisar probe_vision.csv")
    args = ap.parse_args()

    if args.approve_vision:
        cfg = assert_config()
        assert "G0.3" in cfg["runtime"].get("gates", {}), "rode a F0 antes de aprovar o G0.3"
        set_gate("G0.3", True, review="aprovado manualmente pelo Jonathan", at=now())
        print("G0.3 aprovado. Próximo: run_pilot.py --stage pilot")
        return

    REF.mkdir(parents=True, exist_ok=True)
    init_config(args.concurrency)
    memes = load_memes()
    train = export_split(memes)
    pilot = select_pilot(train, [memes[i]["lang"] for i in train])
    write_json(PILOT_IDS, pilot)
    print(f"split 3386/598 conferido · piloto {len(pilot)} memes")

    # G0.1 — servidor
    models, version = server_info()
    update_runtime(server_version=version, server_models=models)
    set_gate("G0.1", MODEL in models, models=models, version=version)
    if MODEL not in models:
        sys.exit(f"G0.1 FALHOU: {MODEL} não está em /v1/models ({models}). Parar.")
    print(f"G0.1 ok · versão do servidor: {version}")

    texts = clean_texts(memes, pilot[:3])
    g02, off_ok, rows, g04 = asyncio.run(smoke(memes, pilot, texts, load_config()["runtime"]["concurrency"]))
    set_gate("G0.2", True, think_off_enabled=off_ok, calls=g02)
    update_runtime(arms_enabled=["think_on", "think_off"] if off_ok else ["think_on"],
                   throughput_g04=g04)
    print(f"G0.2 thinking desligado {'ok' if off_ok else 'FALHOU → braço think_off removido da F1'}")

    with open(REF / "probe_vision.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    set_gate("G0.3", False, review="pendente: revisão manual de probe_vision.csv")
    print(f"G0.4 {g04}")
    print("\nPróximo: revisar glm_reference/probe_vision.csv. Passa se as descrições da imagem "
          "branca descrevem algo branco/vazio e >= 16/20 descrições reais correspondem ao meme.\n"
          "Se passar: run_probe.py --approve-vision. Se não: parar (§11).")


if __name__ == "__main__":
    main()
