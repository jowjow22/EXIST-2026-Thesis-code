"""Visualizador local dos casos extremos (select_extremes.py) com avaliação manual.

Uso (da raiz do repositório, no Mac):
    python3 EXIST_MEMES/glm_annotator/select_extremes.py   # uma vez, gera extremes.json
    python3 EXIST_MEMES/glm_annotator/viewer.py            # abre http://127.0.0.1:8765

Cada avaliação é gravada na hora em glm_reference/extremes/avaliacao.json (e .csv).
Só escuta em 127.0.0.1; não faz chamadas externas.
"""
import csv
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from common import DATA, REF

HERE = Path(__file__).resolve().parent
EXT = REF / "extremes"
EVAL_JSON, EVAL_CSV = EXT / "avaliacao.json", EXT / "avaliacao.csv"
PORT = 8765


def load_evals() -> dict:
    return json.loads(EVAL_JSON.read_text(encoding="utf-8")) if EVAL_JSON.exists() else {}


def save_evals(ev: dict) -> None:
    tmp = EVAL_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(EVAL_JSON)                      # troca atômica: queda no meio não corrompe
    with open(EVAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["meme_id", "label", "note", "timestamp"])
        w.writerows([i, e["label"], e["note"], e["ts"]] for i, e in ev.items())


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, (HERE / "viewer.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/data":
            self.send(200, (EXT / "extremes.json").read_bytes(), "application/json")
        elif self.path == "/evals":
            self.send(200, json.dumps(load_evals()).encode(), "application/json")
        elif self.path.startswith("/img/"):
            p = (DATA / self.path[5:]).resolve()
            if not p.is_relative_to(DATA.resolve()) or not p.is_file():
                return self.send(404, b"not found", "text/plain")
            ctype = {".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(p.suffix.lower(),
                                                                                        "image/jpeg")
            self.send(200, p.read_bytes(), ctype)
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/eval":
            return self.send(404, b"not found", "text/plain")
        e = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ev = load_evals()
        if e.get("label") is None and not e.get("note"):
            ev.pop(e["id"], None)               # desmarcar
        else:
            ev[e["id"]] = {"label": e.get("label"), "note": e.get("note", ""), "ts": e["ts"]}
        save_evals(ev)
        self.send(200, b"ok", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    assert (EXT / "extremes.json").exists(), "rode select_extremes.py antes"
    url = f"http://127.0.0.1:{PORT}"
    print(f"visualizador em {url}  (Ctrl+C para sair) · avaliações em {EVAL_JSON}")
    webbrowser.open(url)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()   # uma thread: sem corrida ao gravar
