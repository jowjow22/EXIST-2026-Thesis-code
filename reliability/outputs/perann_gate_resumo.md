# Resumo do portão sintético — estimador por-anotador com desconto de dificuldade

Portão sintético corrido. **FAIL nas duas variantes, nos três testes.** PARTE 3 não escrita — regra do projeto proíbe ("nunca corras nos dados reais um componente que não passou no teste sintético").

- Relatório completo: `reliability/outputs/perann_gate_report.md`
- Dados brutos: `reliability/outputs/perann_synthetic_gate.json`
- Comando: `python3 -m reliability.validate --perann`

Ficheiros novos: `reliability/perann_reliability.py`, `generate_difficulty_synthetic` em `synthetic.py` (função nova, `generate_synthetic` intacta), testes novos em `validate.py`. **`NutmegEM` não tocado.** Determinismo max|Δ| = 0,000e+00.

## Três configurações

| nome | `discount` | `leave_one_out` | |
|---|---|---|---|
| `mace_naive` | False | False | linha de base MACE-ingénua, mesmo caminho de código |
| `discount_raw` | True | False | **literal da spec** |
| `discount_loo` | True | True | variante acrescentada |

LOO acrescentado porque na forma literal o dissidente **infla com o próprio voto** a dificuldade do item que o vai julgar: 1 dissidente em 6 votos → p=1/6 → dificuldade 0,65 → peso 0,35 em vez de 1,0. Reportado ao lado, não em substituição.

## 2.1 Recuperação (Pearson r vs θ real)

| densidade | hard | votos/an | `mace_naive` | `discount_raw` | `discount_loo` |
|---|---|---|---|---|---|
| confortável | 0,00 | 80 | **0,893** | 0,856 | 0,865 |
| confortável | 0,40 | 80 | 0,712 | 0,730 | **0,736** |
| confortável | 0,60 | 80 | 0,613 | 0,696 | **0,699** |
| exist | 0,40 | 27 | 0,503 | 0,500 | **0,510** |
| exist | 0,60 | 27 | 0,354 | **0,389** | **0,389** |

Desconto só compensa quando ambiguidade genuína domina (+0,083 em hard=0,6; empate em 0,4; perde em ≤0,2). Vantagem mínima exigida 0,05, medida −0,003 → **FAIL**. Densidade EXIST corta r em ~0,2 — mesmo tecto do `NutmegEM`.

## 2.2 Minoria, dois braços MATCHED

| config | hard: minoria | hard: fiáveis | easy: minoria | easy: fiáveis |
|---|---|---|---|---|
| `mace_naive` | 0,940 | 0,845 | 0,728 | 0,851 |
| `discount_raw` | 0,983 | 0,947 | **0,896** | 0,945 |
| `discount_loo` | 0,940 | 0,842 | **0,702** | 0,820 |

Braço `hard` **passa com folga** nas duas. Braço `easy` **falha** nas duas (tinha de cair ≤ 0,60). Metade certa: protege minoria em terreno ambíguo, mas não pune quem discorda em terreno de consenso.

## 2.3 Spammer de viés constante (20%, sempre YES)

| config | enviesados | fiáveis | gap |
|---|---|---|---|
| `mace_naive` | 0,774 | 0,855 | +0,081 |
| `discount_raw` | 0,906 | 0,944 | **+0,037** |
| `discount_loo` | 0,733 | 0,823 | +0,089 |

`discount_raw` tem gap **menor** que o MACE ingénuo. A hipótese de desenho — "os erros do spammer caem em itens fáceis, logo o desconto não o protege" — está **refutada**.

## O diagnóstico

Dificuldade sai dos votos. Um bloco dissidente grande **fabrica** a dificuldade dos itens onde discorda. O desconto quebra a circularidade *competência → dificuldade*, **não** a circularidade *bloco → dificuldade*.

Sweep (gap = fiáveis − bloco, maior é melhor):

| fração | easy: `mace` | easy: `raw` | easy: `loo` | bias: `mace` | bias: `raw` | bias: `loo` |
|---|---|---|---|---|---|---|
| 0,03 | +0,242 | +0,150 | **+0,302** | +0,162 | +0,094 | **+0,209** |
| 0,15 | +0,176 | +0,096 | **+0,210** | +0,103 | +0,050 | **+0,118** |
| 0,25 | **+0,124** | +0,049 | +0,118 | +0,057 | +0,026 | **+0,065** |
| 0,35 | **+0,044** | +0,014 | +0,038 | −0,014 | −0,018 | **−0,043** |

Decai monotonicamente nas três. A 35% de spammers o gap fica **negativo** — o bloco recebe competência maior que os fiáveis, porque a 35% de sempre-YES ele *é* a maioria.

Mesma geometria do Stage 1B, espelhada: lá blocos **pequenos** capturavam células próprias do `NutmegEM`; aqui blocos **grandes** capturam a dificuldade empírica. Duas pontas do mesmo problema — qualquer estimador que infira a verdade dos próprios votos é capturável por um bloco correlacionado.

## Escala vs fundo

1. **Escala (corrigível).** Limiares absolutos (teto 0,60, piso 0,70) vieram do Stage 1, onde θ tem chão 0. Aqui competência é taxa de concordância, chão de acaso ≈ 0,5: spammer com θ real 0,25 tem concordância esperada 0,25 + 0,75×0,5 = **0,625**, acima do teto por construção. Limiares absolutos não transferem entre escalas; o **gap** transfere. Os limiares não foram mexidos.
2. **Fundo (não corrigível por limiar).** Só em gaps, `discount_raw` perde para `mace_naive` em **10/10** linhas do sweep. Nenhuma recalibração muda isto.

## LOO domina a forma literal

r superior em 7/8 linhas da 2.1 · gap superior em **9/10** do sweep · passa o braço `hard` com folga. Contra o `mace_naive` a vantagem é modesta: 7/10, ganhando só em blocos ≤ 0,15.

## Estado

Decisão de como prosseguir é do autor. PARTE 3 (`run_perann_exist.py`) não escrita nem executada.
