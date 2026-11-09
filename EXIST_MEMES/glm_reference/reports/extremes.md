# Casos extremos de concordância humana — comportamento do GLM (treino)

Taxa de YES = fração de YES nas 3 paráfrases (P1–P3). Voto = maioria das 3. Humanos: fração de YES dos 6 anotadores.

| grupo | n | humanos | YES think_off | YES think_on | voto YES off | voto YES on | p_GLM off (0/⅓/⅔/1) | p_GLM on (0/⅓/⅔/1) |
|---|---|---|---|---|---|---|---|---|
| 1Y5N | 388 | 0.167 | 0.648 | 0.492 | 0.719 | 0.510 | 58/51/134/145 | 139/51/72/126 |
| ↳ en | 210 | 0.167 | 0.594 | 0.468 | 0.643 | 0.471 | 44/31/62/73 | 85/26/28/71 |
| ↳ es | 178 | 0.167 | 0.712 | 0.521 | 0.809 | 0.556 | 14/20/72/72 | 54/25/44/55 |
| 3Y3N | 518 | 0.500 | 0.817 | 0.715 | 0.880 | 0.753 | 21/41/140/316 | 77/51/110/280 |
| ↳ en | 260 | 0.500 | 0.808 | 0.731 | 0.865 | 0.765 | 13/22/67/158 | 36/25/52/147 |
| ↳ es | 258 | 0.500 | 0.826 | 0.699 | 0.895 | 0.740 | 8/19/73/158 | 41/26/58/133 |
| 5Y1N | 645 | 0.833 | 0.928 | 0.879 | 0.964 | 0.918 | 7/15/86/536 | 25/27/103/489 |
| ↳ en | 310 | 0.833 | 0.930 | 0.881 | 0.961 | 0.916 | 5/6/35/263 | 13/12/45/239 |
| ↳ es | 335 | 0.833 | 0.925 | 0.877 | 0.967 | 0.919 | 2/9/51/273 | 12/15/58/250 |

Leitura: em 1Y5N o esperado de um anotador típico é ~17% de YES; em 5Y1N, ~83%; em 3Y3N, ~50%. p_GLM não soma n quando alguma paráfrase não foi parseada.

Arquivos: `glm_reference/extremes/extremes.csv` e `extremes.json`. Visualizador: `python3 EXIST_MEMES/glm_annotator/viewer.py`.
