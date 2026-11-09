"""Gera o PDF de METODOLOGIA_GLM_CLAUDE.md + RELATORIO_GLM_CLAUDE.md (capa + as duas partes).

Uso (no Mac, da raiz do repositório; precisa de `pip install markdown` e do Google Chrome):
    python3 EXIST_MEMES/glm_annotator/relatorio_pdf.py "" EXIST_MEMES
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
        --print-to-pdf=EXIST_MEMES/glm_reference/reports/relatorio_glm_claude.pdf \
        "file://$PWD/EXIST_MEMES/glm_annotator/relatorio.html"

1º argumento: pasta extra no sys.path onde está o pacote markdown ("" se instalado); 2º: a pasta EXIST_MEMES.
Grava glm_annotator/relatorio.html (arquivo gerado, intermediário): fica ao lado dos .md para que os caminhos
relativos das figuras (../glm_reference/reports/*.png) funcionem.
"""
import sys, pathlib, re
sys.path.insert(0, sys.argv[1])
import markdown
B = pathlib.Path(sys.argv[2])
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#1a1a18; --muted:#55534e; --line:#dcdad3; --bg:#ffffff; --accent:#2a78d6; }
body { background:var(--bg); color:var(--ink); font-family: "Charter","Georgia",serif; font-size:10.3pt; line-height:1.45; max-width: 100%; }
h1 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:19pt; line-height:1.2; margin:0 0 8pt; }
h2 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:14pt; margin:20pt 0 6pt; border-bottom:1px solid var(--line); padding-bottom:3pt; break-after:avoid; }
h3 { font-family:"Helvetica Neue",Arial,sans-serif; font-size:11.5pt; margin:14pt 0 4pt; break-after:avoid; }
p, li { orphans:3; widows:3; }
table { border-collapse:collapse; width:100%; margin:6pt 0 10pt; font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.2pt; break-inside:auto; }
th, td { border-bottom:1px solid var(--line); padding:3pt 5pt; text-align:left; vertical-align:top; }
th { background:#f3f2ee; font-weight:600; }
tr { break-inside:avoid; }
code { font-family:"SF Mono",Menlo,monospace; font-size:8.6pt; background:#f3f2ee; padding:0 2pt; border-radius:2pt; }
pre { background:#f6f5f1; border:1px solid var(--line); padding:6pt 8pt; white-space:pre-wrap; font-size:8pt; line-height:1.35; break-inside:avoid; }
pre code { background:none; padding:0; }
img { max-width:100%; display:block; margin:8pt auto 2pt; break-inside:avoid; }
p:has(> img) { break-inside:avoid; }
em { color:var(--muted); }
hr { border:none; border-top:1px solid var(--line); margin:14pt 0; }
.pagebreak { break-before:page; }
.cover { margin-top:60mm; }
.cover h1 { font-size:24pt; }
.cover p { color:var(--muted); font-size:11pt; }
"""
def conv(p):
    t = p.read_text(encoding="utf-8")
    return markdown.markdown(t, extensions=["tables", "fenced_code", "sane_lists"])
cover = """<div class="cover"><h1>LLMs multimodais como anotadores de referência para sexismo em memes</h1>
<p>GLM-4.6V-Flash e Claude Sonnet 5.5 — metodologia e resultados dos docs 25 a 31</p>
<p>Jonathan Santos da Silva · EXIST 2026 (Task 2.1) · versão de 2026-10-22</p>
<p>Parte A: Metodologia (implementação) · Parte B: Relatório de resultados</p></div>"""
html = (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>GLM vs Claude</title><style>{CSS}</style></head><body>'
        + cover + '<div class="pagebreak"></div>' + conv(B/"glm_annotator/METODOLOGIA_GLM_CLAUDE.md")
        + '<div class="pagebreak"></div>' + conv(B/"glm_annotator/RELATORIO_GLM_CLAUDE.md") + "</body></html>")
(B/"glm_annotator/relatorio.html").write_text(html, encoding="utf-8")
print("html ok", len(html))
