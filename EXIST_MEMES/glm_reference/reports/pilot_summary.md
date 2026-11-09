# F1 — Piloto (escolha do thinking)

Gerado por `run_pilot.py --stage pilot`. 240 memes, P1–P3, seed 42. Nenhum voto humano foi lido.

| braço | escopo | n_memes | parse_rate | length_rate | yes_rate | paraphrase_agreement | refusals_other | api_errors | cjk_reasoning_frac | median_completion_tokens | median_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| think_on | total | 240 | 1.0000 | 0.0000 | 0.7000 | 0.7583 | 0 | 0 | 0.0000 | 171.0000 | 15.7225 |
| think_on | en | 120 | 1.0000 | 0.0000 | 0.6944 | 0.7667 | 0 | 0 | 0.0000 | 154.5000 | 15.3155 |
| think_on | es | 120 | 1.0000 | 0.0000 | 0.7056 | 0.7500 | 0 | 0 | 0.0000 | 176.0000 | 17.2060 |
| think_off | total | 240 | 1.0000 | 0.0000 | 0.7931 | 0.7500 | 0 | 0 | None | 4.0000 | 0.7120 |
| think_off | en | 120 | 1.0000 | 0.0000 | 0.7972 | 0.7750 | 0 | 0 | None | 4.0000 | 0.6535 |
| think_off | es | 120 | 1.0000 | 0.0000 | 0.7889 | 0.7250 | 0 | 0 | None | 4.0000 | 0.8175 |

## Regra de escolha (§6)

- `think_on` elegível: **True**
- `think_off` elegível: **True**

Braço escolhido: **`think_off`**. Próximo: `run_pilot.py --stage stability`.

## F1b — Estabilidade

Braço `think_off`, P1, seeds [42, 43, 44]. `?` conta como flip.

| escopo | flip_rate |
|---|---|
| total | 0.0042 |
| en | 0.0000 |
| es | 0.0083 |

**G1 APROVADO** (flip_rate total 0.0042, limite 0.1). Próximo: `run_reference.py`.
