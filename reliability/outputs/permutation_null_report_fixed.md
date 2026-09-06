# Nulo por permutação — geração e validação

Gerado por `reliability/make_permuted_soft_labels.py`.

- θ de origem: `reliability/outputs/theta_taskA_single_truth_K1_fixed.json`
- sementes de permutação (fixas): `[101, 202, 303, 404, 505]`
- ficheiros escritos: **sim**

## 0. Porquê estratificar por painel linguístico

Os painéis ES e EN são quase disjuntos. Permutar θ entre línguas destruiria
estrutura que **não é confiabilidade** — que anotador viu que língua — e daria
um nulo artificialmente fácil de bater. Três estratos disjuntos, cada um
permutado dentro de si. Os **bilingues formam estrato próprio**: simétrico, não
privilegia nenhuma língua, preserva a estrutura de painel exactamente.

## 1. Estratos `[MEDIDO]`

| estrato | n |
|---|---|
| `es_only` | 437 |
| `en_only` | 431 |
| `bilingual` | 19 |
| `orphan` | 0 |
| **total** | **887** |

## 2. Critérios pré-registados

- **P1_distribution_preserved** — Dentro de cada estrato, o multiconjunto ordenado de θ é IDÊNTICO antes e depois (igualdade exacta de floats). FAIL = a permutação não é permutação.
- **P2_no_cross_language** — Todo o anotador recebe um θ de um anotador do MESMO estrato. FAIL = houve fuga entre painéis linguísticos.
- **P3_fixed_points** — Fração de pontos fixos (anotador que mantém o próprio θ) < 0.05 em cada permutação e estrato com n >= 20. FAIL = a permutação mal mexeu.
- **P4_strata_sizes** — Os três estratos particionam os anotadores do ficheiro de θ, sem sobras. Os tamanhos são MEDIDOS e reportados, não assumidos.
- **P5_perturbation_magnitude** — Descritivo, sem PASS/FAIL: média|Δp_YES| das labels permutadas contra a crua, comparada com a mesma quantidade para o A1 real. Se as permutadas perturbarem MUITO menos, o nulo é fraco e declara-se a limitação.

## 3. Resultado das verificações `[MEDIDO]`

| seed | P1 distribuição | P2 sem fuga | P3 pontos fixos | máx frac. fixos |
|---|---|---|---|---|
| 101 | PASS | PASS | PASS | 0.0000 |
| 202 | PASS | PASS | PASS | 0.0046 |
| 303 | PASS | PASS | PASS | 0.0046 |
| 404 | PASS | PASS | PASS | 0.0046 |
| 505 | PASS | PASS | PASS | 0.0046 |

## 4. P5 — magnitude da perturbação `[MEDIDO]`

média|Δp_YES| contra a soft label **crua**. O A1 real é o padrão: se as
permutadas perturbarem muito menos, o nulo é fraco.

| braço | média\|Δp_YES\| vs crua |
|---|---|
| A1 real (θ na sua atribuição) | **0.061732** |
| permutado seed 101 | 0.051031 |
| permutado seed 202 | 0.051312 |
| permutado seed 303 | 0.053081 |
| permutado seed 404 | 0.051224 |
| permutado seed 505 | 0.050107 |
| permutado: média | 0.051351 |

Razão permutado/real = **0.832**.

## 5. Ficheiros

- `soft_labels_taskA_permuted_seed101_fixed.json`
- `soft_labels_taskA_permuted_seed202_fixed.json`
- `soft_labels_taskA_permuted_seed303_fixed.json`
- `soft_labels_taskA_permuted_seed404_fixed.json`
- `soft_labels_taskA_permuted_seed505_fixed.json`
- `soft_labels_taskA_permuted_manifest.json` — metadados, verificações e o mapa
  doador→receptor de cada permutação. Fica **fora** dos ficheiros de
  labels para que o formato seja idêntico ao
  `soft_labels_taskA_weighted.json`: dict plano `id → [p_YES, p_NO]`.

`[LIMITAÇÃO]` Estes ficheiros foram construídos a partir do θ indicado em
cima. Se a decisão sobre `mstep_responsibility_diagnosis.md` for regenerar
o θ, estes cinco ficheiros têm de ser regenerados com ele — a permutação é
do mesmo vector que alimenta o A1, senão o nulo não é o nulo do A1.
