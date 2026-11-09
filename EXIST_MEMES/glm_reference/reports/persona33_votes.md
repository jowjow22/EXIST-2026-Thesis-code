# Doc 33 — votos da Persona no treino (S1–S3, think_off)

Gerado por `persona_ref.py votes`. 3386 memes de treino × 3 prompts = 10158 registros (o último de cada chave vale). Sem rótulo: nenhum. Imagens reenviadas em PNG: 33 memes. Tokens: 18438973 de entrada, 40632 de saída (≈ US$ 18.64 no preço batch, extremos incluídos).

## Portão G2 (SPEC_33 §3: parse ≥ 0,98 e confiante ≥ 0,70 em cada idioma)

| idioma | n | parse | confiante (3-0) |
|---|---|---|---|
| en | 1715 | 1.0000 | 0.8723 |
| es | 1671 | 1.0000 | 0.8480 |

**G2: aprovado**

Descritivo: os 1 551 extremos já eram conhecidos do doc 32; só nos 1 835 memes novos: en parse 1.0000, confiante 0.8759 · es parse 1.0000, confiante 0.8656.

## Calibração por nº de YES humanos (descritivo; a linha dos novos é o teste fora dos extremos, SPEC_33 §2)

Fração de YES sobre as respostas parseadas; humanos = fração de YES dos 6 votos.

| conjunto | YES humanos | n | humanos | S1 | S2 | S3 | Persona (3 respostas) | voto Persona | GLM off (3 respostas) | parse | recusas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| novos (fora dos extremos) | 0 | 316 | 0.000 | 0.057 | 0.054 | 0.054 | 0.055 | 0.047 | 0.400 | 1.000 | 0 |
| novos (fora dos extremos) | 2 | 458 | 0.333 | 0.225 | 0.269 | 0.288 | 0.261 | 0.255 | 0.737 | 1.000 | 0 |
| novos (fora dos extremos) | 4 | 585 | 0.667 | 0.609 | 0.685 | 0.691 | 0.662 | 0.670 | 0.902 | 1.000 | 0 |
| novos (fora dos extremos) | 6 | 476 | 1.000 | 0.903 | 0.933 | 0.933 | 0.923 | 0.920 | 0.964 | 1.000 | 0 |
| novos (fora dos extremos) | todos | 1835 | 0.555 | 0.494 | 0.537 | 0.543 | 0.525 | 0.524 | 0.790 | 1.000 | 0 |
| extremos (doc 32) | 1 | 388 | 0.167 | 0.129 | 0.175 | 0.155 | 0.153 | 0.157 | 0.648 | 1.000 | 0 |
| extremos (doc 32) | 3 | 518 | 0.500 | 0.429 | 0.496 | 0.506 | 0.477 | 0.481 | 0.817 | 1.000 | 0 |
| extremos (doc 32) | 5 | 645 | 0.833 | 0.755 | 0.814 | 0.803 | 0.791 | 0.795 | 0.929 | 1.000 | 0 |
| extremos (doc 32) | todos | 1551 | 0.555 | 0.489 | 0.548 | 0.542 | 0.526 | 0.531 | 0.821 | 1.000 | 0 |
| treino inteiro | 0 | 316 | 0.000 | 0.057 | 0.054 | 0.054 | 0.055 | 0.047 | 0.400 | 1.000 | 0 |
| treino inteiro | 1 | 388 | 0.167 | 0.129 | 0.175 | 0.155 | 0.153 | 0.157 | 0.648 | 1.000 | 0 |
| treino inteiro | 2 | 458 | 0.333 | 0.225 | 0.269 | 0.288 | 0.261 | 0.255 | 0.737 | 1.000 | 0 |
| treino inteiro | 3 | 518 | 0.500 | 0.429 | 0.496 | 0.506 | 0.477 | 0.481 | 0.817 | 1.000 | 0 |
| treino inteiro | 4 | 585 | 0.667 | 0.609 | 0.685 | 0.691 | 0.662 | 0.670 | 0.902 | 1.000 | 0 |
| treino inteiro | 5 | 645 | 0.833 | 0.755 | 0.814 | 0.803 | 0.791 | 0.795 | 0.929 | 1.000 | 0 |
| treino inteiro | 6 | 476 | 1.000 | 0.903 | 0.933 | 0.933 | 0.923 | 0.920 | 0.964 | 1.000 | 0 |
| treino inteiro | todos | 3386 | 0.555 | 0.492 | 0.542 | 0.543 | 0.525 | 0.527 | 0.804 | 1.000 | 0 |

