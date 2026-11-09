"""Parâmetros fixados (§3/§16 do SPEC_25.md), caminhos e utilidades partilhadas.

`config.json` tem duas partes:
- `fixed`: tudo o que foi pré-registrado. É gravado na F0 e, no início de cada fase,
  `assert_config()` exige que seja idêntico ao que este módulo declara (portão 12.1).
- `runtime`: o que a execução descobre (versão do servidor, braços habilitados,
  resultado de cada portão, braço escolhido, hashes dos soft labels).
"""
import base64
import hashlib
import json
import mimetypes
import sys
from pathlib import Path

from prompts import PROBE_SHA256, PROMPT_SHA256

ROOT = Path(__file__).resolve().parents[2]
MEMES_DIR = ROOT / "EXIST_MEMES"
DATA = MEMES_DIR / "Dataset/memes/training"
JSON_PATH = DATA / "EXIST2026_training.json"

REF = MEMES_DIR / "glm_reference"
RAW_DIR = REF / "raw"
REPORTS = REF / "reports"
CONFIG_PATH = REF / "config.json"
TRAIN_IDS = REF / "train_ids.json"
VAL_IDS = REF / "val_ids.json"
PILOT_IDS = REF / "pilot_ids.json"
VOTES_CSV = REF / "glm_votes.csv"

REL_OUT = ROOT / "reliability/outputs"
SPLIT_PATH = REL_OUT / "split_taskA_frozen.json"
SOFT_RAW = REL_OUT / "soft_labels_taskA_raw.json"
SOFT_GLMW = REL_OUT / "soft_labels_taskA_glmw.json"
SOFT_GLMBASE = REL_OUT / "soft_labels_taskA_glmbase.json"

for p in (str(MEMES_DIR), str(ROOT)):          # memes_core e reliability.*
    if p not in sys.path:
        sys.path.insert(0, p)

BASE_URL = "http://193.136.195.31:8000/v1"
MODEL = "glm-4.6v-flash"
ARMS = ("think_on", "think_off")
LABELS = ("YES", "NO")

FIXED = {
    "base_url": BASE_URL,
    "model": MODEL,
    "temperature": 0.1,
    "top_p": 0.6,
    "top_k": 2,
    "repetition_penalty": 1.1,
    "max_tokens": {"think_on": 16384, "think_off": 256},
    "timeout_s": {"think_on": 600, "think_off": 60},
    "seed_reference": 42,
    "seeds_stability": [43, 44],
    "transport_retries": 3,
    "prompt_sha256": PROMPT_SHA256,
    "probe_prompt_sha256": PROBE_SHA256,
    "message": "uma mensagem user, sem system prompt, partes [image_url, text]",
    "pilot": {"per_lang": 120, "langs": ["en", "es"], "rng": 2026,
              "procedure": "por idioma (en, depois es): ids de treino ordenados como "
                           "string; um único default_rng(2026); rng.choice sem reposição"},
    "split": "reliability/outputs/split_taskA_frozen.json (3386/598)",
    "thresholds": {"parse": 0.98, "yes_lo": 0.05, "yes_hi": 0.95, "confident": 0.70,
                   "think_gain": 0.05, "flip": 0.10, "share_moved": 0.05,
                   "rho": 0.8, "half_vote": 1 / 12, "tol": 1e-9},
    "deviation_to_weight": "r = 1 - D",
    "bootstrap": {"n": 2000, "rng": 2026},
}
T = FIXED["thresholds"]


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    """Troca o arquivo de uma vez (tmp + replace): uma queda da montagem no meio não deixa o
    config.json ou um arquivo de labels pela metade."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_memes() -> dict:
    """{id_EXIST: item}, na ordem de inserção do JSON."""
    return load_json(JSON_PATH)


def clean_texts(memes: dict, ids) -> dict:
    """Texto do meme limpo por memes_core.pre_process_text (import lento: torch)."""
    from memes_core import pre_process_text
    return {i: pre_process_text(memes[i]["text"]) for i in ids}


def image_path(item) -> Path:
    return DATA / item["path_memes"]


def data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def image_data_url(item) -> tuple[str, str]:
    """(data URL, sha256 dos bytes) da imagem do meme, como em glm_zeroshot.py."""
    p = image_path(item)
    raw = p.read_bytes()
    return data_url(raw, mimetypes.guess_type(p)[0]), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------- config.json

def load_config() -> dict:
    assert CONFIG_PATH.exists(), f"{CONFIG_PATH} não existe: rode run_probe.py (F0) antes"
    return load_json(CONFIG_PATH)


def assert_config() -> dict:
    """Portão 12.1: os parâmetros gravados na F0 são exatamente os pré-registrados."""
    cfg = load_config()
    assert cfg["fixed"] == json.loads(json.dumps(FIXED)), \
        "config.json['fixed'] diverge dos parâmetros pré-registrados (portão 12.1)"
    return cfg


def update_runtime(**kv) -> dict:
    cfg = load_config()
    cfg["runtime"].update(kv)
    write_json(CONFIG_PATH, cfg)
    return cfg


def set_gate(name: str, passed: bool, **info) -> None:
    cfg = load_config()
    cfg["runtime"].setdefault("gates", {})[name] = {"pass": bool(passed), **info}
    write_json(CONFIG_PATH, cfg)


def require_gate(cfg: dict, name: str) -> None:
    g = cfg["runtime"].get("gates", {}).get(name)
    assert g is not None and g["pass"], f"portão {name} não aprovado: a fase não pode rodar"


def assert_server_version(cfg: dict, current: str) -> None:
    """Portão 12.1: a versão do servidor não mudou desde a F0."""
    assert current == cfg["runtime"]["server_version"], (
        f"versão do servidor mudou: F0 gravou {cfg['runtime']['server_version']!r}, "
        f"agora {current!r}")
