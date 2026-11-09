"""Gera o HTML do relatório Persona (doc 32) a partir de RELATORIO_PERSONA.md; o PDF sai do Chrome headless.

Uso (no Mac, da raiz do repositório; precisa do pacote `markdown` e do Google Chrome):
    python3 EXIST_MEMES/glm_annotator/relatorio_persona_pdf.py "" EXIST_MEMES /tmp/relatorio_persona.html
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
        --print-to-pdf=EXIST_MEMES/glm_reference/reports/relatorio_persona.pdf "file:///tmp/relatorio_persona.html"

1º argumento: pasta extra no sys.path onde está o pacote markdown ("" se instalado); 2º: a pasta EXIST_MEMES;
3º: onde gravar o HTML intermediário (fora do repositório). As figuras são referenciadas por caminho absoluto.
"""
import pathlib
import re
import sys

sys.path.insert(0, sys.argv[1])
import markdown

B = pathlib.Path(sys.argv[2]).resolve()
OUT = pathlib.Path(sys.argv[3])
# mesmo estilo do relatório GLM/Claude (relatorio_pdf.py)
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#1a1a18; --muted:#55534e; --line:#dcdad3; --bg:#ffffff; }
body { background:var(--bg); color:var(--ink); font-family: "Charter","Georgia",serif; font-size:10.3pt; line-height:1.45; max-width: 100%; }
h1 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:19pt; line-height:1.2; margin:0 0 8pt; }
h2 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:14pt; margin:20pt 0 6pt; border-bottom:1px solid var(--line); padding-bottom:3pt; break-after:avoid; }
h3 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:11.5pt; margin:14pt 0 4pt; break-after:avoid; }
p, li { orphans:3; widows:3; }
table { border-collapse:collapse; width:100%; margin:6pt 0 10pt; font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.2pt; break-inside:auto; }
th, td { border-bottom:1px solid var(--line); padding:3pt 5pt; text-align:left; vertical-align:top; }
th { background:#f3f2ee; font-weight:600; }
tr { break-inside:avoid; }
code { font-family:"SF Mono",Menlo,monospace; font-size:8.6pt; background:#f3f2ee; padding:0 1pt; border-radius:2pt; }
table code { font-size:0.9em; }
pre { background:#f6f5f1; border:1px solid var(--line); padding:6pt 8pt; white-space:pre-wrap; font-size:7.6pt; line-height:1.3; }
pre code { background:none; padding:0; }
img { max-width:100%; display:block; margin:8pt auto 2pt; break-inside:avoid; }
p:has(> img) { break-inside:avoid; break-after:avoid; }
p:has(+ table), p:has(+ ul), p:has(+ ol), p:has(+ pre) { break-after:avoid; }
.nw { white-space:nowrap; }
em { color:var(--muted); }
hr { border:none; border-top:1px solid var(--line); margin:14pt 0; }
.pagebreak { break-before:page; }
.cover { margin-top:60mm; }
.cover h1 { font-size:24pt; }
.cover p { color:var(--muted); font-size:11pt; }
"""


def lists(md):
    """O pacote markdown só abre lista depois de linha em branco e só aninha com 4 espaços; o .md segue o
    estilo GitHub (lista colada no parágrafo, aninhamento com 2–3 espaços). Fora dos blocos de código."""
    out, code = [], False
    for line in md.split("\n"):
        if line.startswith("```"):
            code = not code
        m = None if code else re.match(r"^( *)([-*]|\d+\.) ", line)
        if m:
            ind = len(m.group(1))
            line = " " * (0 if ind == 0 else 4 if ind <= 4 else 8) + line[ind:]
            prev = out[-1] if out else ""
            if prev.strip() and not re.match(r"^ *([-*]|\d+\.) ", prev):
                out.append("")
        out.append(line)
    return "\n".join(out)


md = lists((B / "glm_annotator/RELATORIO_PERSONA.md").read_text(encoding="utf-8"))
md = md.split("\n", 1)[1]                      # o título já está na capa
md = md.replace("](../glm_reference/reports/", f"]({(B / 'glm_reference/reports').as_uri()}/")
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
# números com espaço de milhar, "n = 388" e "US$ 8,68" não quebram de linha (fora de <code>/<pre>)
NW = re.compile(r"\d{1,3}(?: \d{3})+\b|\bn = [\d ]*\d|(?:≈ )?US\$ [\d,]+|— · \d+ \d{3}")
body = "".join(t if t.startswith("<") else NW.sub(lambda m: f'<span class="nw">{m.group(0)}</span>', t)
               for t in re.split(r"(<pre>.*?</pre>|<code>.*?</code>|<[^>]+>)", body, flags=re.S))
cover = """<div class="cover"><h1>Persona: três system prompts no Claude Sonnet 5.5</h1>
<p>Comparação de quatro anotadores LLM nos 1 551 memes extremos: GLM-4.6V-Flash (thinking desligado e ligado),
Claude Sonnet 5.5 com as paráfrases P1–P3 (doc 31) e Claude Sonnet 5.5 Persona, com os system prompts S1–S3 (doc 32)</p>
<p>Jonathan Santos da Silva · EXIST 2026 (Task 2.1) · versão de 2026-10-28</p>
<p>Parte A: Método · Parte B: Resultados · Parte C: Interpretação, limitações e pendências</p></div>"""
html = (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Persona</title><style>{CSS}</style>'
        f'</head><body>{cover}<div class="pagebreak"></div>{body}</body></html>')
OUT.write_text(html, encoding="utf-8")
print("html ok", OUT, len(html))
