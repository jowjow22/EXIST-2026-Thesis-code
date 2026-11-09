# Doc 30 — GLM com thinking ligado vs desligado (treino)

P1–P3, seed 42, 3386 memes de treino, mesmos parâmetros do doc 25.

| escopo | humanos (média p_YES) | YES think_off | YES think_on | Δ (on − off) | parse think_on | confiantes think_on | voto igual on/off |
|---|---|---|---|---|---|---|---|
| total | 0.555 | 0.804 | 0.711 | -0.093 | 0.9985 | 0.747 | 0.853 |
| en | 0.542 | 0.793 | 0.708 | -0.085 | 0.9983 | 0.764 | 0.875 |
| es | 0.568 | 0.817 | 0.715 | -0.102 | 0.9988 | 0.730 | 0.831 |

Distribuição de p_GLM (fração de YES nas 3 paráfrases):

| braço | 0 | 1/3 | 2/3 | 1 |
|---|---|---|---|---|
| think_on | 608 | 255 | 596 | 1922 |
| think_off | 254 | 241 | 739 | 2147 |

**Decisão (SPEC_30 §3): o viés de YES PERSISTE com thinking ligado (0.711 ≥ 0.555 + 0.1). Próximo: testar outros modelos.**
