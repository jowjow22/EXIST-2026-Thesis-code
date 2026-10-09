# Código do doc 25 (versão anterior ao doc 26/27), arquivado em 2026-10-09

Cópia fiel dos arquivos de `glm_annotator/` como estavam quando o doc 25 rodou (F0–F2, 2026-10-07/08),
antes de serem reescritos no próprio arquivo:

| arquivo | o que mudou depois |
|---|---|
| `client.py` | é a versão que executou F0, F1, F1b e F2. Depois foi generalizada para o doc 26 (`templates`, `stem`); as chamadas do doc 25 não mudaram de comportamento |
| `weights.py` | fórmula do doc 25: voto duro do GLM, só memes 3-0, `d ∈ {0, 1}`. **Nunca rodou nos dados reais** (o G2 falhou antes da F3) |
| `checks.py` | checagens (a)–(g) do doc 25. Nunca rodou nos dados reais |
| `analyze_f5.py` | antes de fixar o estado primário (menor `eval_loss`) e o teste bilateral (SPEC_27 §16A) |
| `tests/test_weights.py` | Apêndice A do doc 25 (9 testes) |

Não executar a partir desta pasta: os imports apontam para os módulos de `glm_annotator/`. Serve de registro e de referência.
