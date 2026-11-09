# 31 — O viés de YES é do GLM ou também aparece no Claude? (memes extremos)

**Pré-registro aprovado em 2026-10-19, antes do piloto: modelo `claude-sonnet-5-5`, regra do §3 com os limites 0,40 e 0,25. Mudanças depois de rodar só como seção nova datada.** Antes da aprovação rodou apenas o `--smoke` (3 chamadas síncronas no meme 110096, só para checar chave e parâmetros; nada gravado).

**Estatuto: descritivo.** O doc 30 mostrou que o viés de YES do GLM persiste com o thinking ligado (0,711 de YES contra 0,555 dos humanos). O orientador pediu para testar outro LLM. O foco está nos casos extremos de concordância humana (`select_extremes.py`, treino):

| grupo | n | humanos | voto YES GLM off | voto YES GLM on |
|---|---|---|---|---|
| 1Y5N (1 YES, 5 NO) | 388 | 0,167 | 0,719 | 0,510 |
| 3Y3N (empate) | 518 | 0,500 | 0,880 | 0,753 |
| 5Y1N (5 YES, 1 NO) | 645 | 0,833 | 0,964 | 0,918 |

## 1. O que roda

- `claude_reference.py`: os 1 551 memes acima × P1–P3 (os mesmos textos, já limpos, e as mesmas imagens que o GLM recebeu). Uma mensagem user, sem system prompt, partes [imagem, texto], via Message Batches API.
- Modelo: **`claude-sonnet-5-5`** (o 5.5 de porte médio com thinking desligável).
- Braços:
  - `think_off`: `thinking: between_tools` no Sonnet 5.5 e `disabled` no Haiku 5.5, `max_tokens` 256;
  - `think_on`: `thinking: adaptive` (resumo gravado), `max_tokens` 16 000.
  - Os dois com effort `medium`.
- Ordem: primeiro `--smoke` (3 chamadas síncronas), depois o piloto `--per-group 20` (180 pedidos), depois todos.
- Grava `claude_reference/raw/<modelo>_<braço>.jsonl` (mesmo formato do GLM, mais `refusal_category` e `batch_id`) e `reports/claude_<modelo>_<braço>_extremes.md`. Não toca em nada do GLM.

**Diferenças inevitáveis em relação ao GLM** (limitação a registrar na tese):
- os modelos Claude 5.5 rejeitam temperature/top_p/top_k fora do padrão e não têm seed, então não dá para fixar temp 0,1 · top_p 0,6 · seed 42;
- a Batch API não aceita `fallbacks`: uma recusa fica como `?` com a categoria, sem trocar de modelo;
- o meme 210851 (5,1 MB) passa do limite de 5 MB por imagem e é reduzido para 1568 px de lado, que é o tamanho a que o Claude reduz as imagens de qualquer forma.

## 2. O que se mede (por grupo; total, en, es)

- Voto YES (maioria das 3 paráfrases).
- Fração de YES nas paráfrases.
- Parse rate e número de recusas.
- As mesmas colunas do GLM off/on para comparar.

## 3. Regra de decisão (fixada em 2026-10-19)

Métrica principal: **voto YES do Claude no grupo 1Y5N** (5 de 6 humanos dizem NO).

- **≥ 0,40:** o Claude também tem o viés, logo ele não é exclusivo do GLM. (0,40 fica abaixo do GLM com thinking ligado, 0,51, com folga de 0,1.)
- **≤ 0,25:** o viés é, em boa parte, do GLM. (0,25 é perto do esperado de um anotador humano típico, 0,167.)
- **Entre as duas:** parcial; reportar sem conclusão forte.

Secundária: voto YES no 3Y3N comparado com 0,5. Mesma leitura por idioma.

## 4. O que NÃO fazer

- Não decidir o modelo nem a regra depois de ver o piloto.
- Não usar o piloto para ajustar prompts: os P1–P3 são os do SPEC_25, sem mudança.
- Não usar `fallbacks` nem outro modelo para cobrir recusas.

## 5. Resultado

### Piloto de 2026-10-19 (`think_off`, `--per-group 20`, 60 memes, 180 pedidos, lote `msgbatch_01FkUCMPxzxwLmcQuaPTbDST`)

Parse 1,000, 0 recusas. Voto YES (Claude / GLM off / GLM on, nos mesmos memes): 1Y5N 0,200 / 0,750 / 0,550 · 3Y3N 0,750 / 0,950 / 0,800 · 5Y1N 0,900 / 1,000 / 0,850. 1Y5N por idioma (Claude): en 0,000 · es 0,364. Com n = 20 por grupo, o IC 95% de 0,20 vai de ~0,06 a ~0,44: **o piloto não aplica a regra do §3**, só confirma que o pipeline funciona. O relatório `reports/claude_claude-sonnet-5-5_think_off_extremes.md` será sobrescrito pela rodada completa; os números do piloto ficam aqui.

### Rodada completa de 2026-10-19 (`think_off`, 1 551 memes, 4 653 pedidos)

| grupo | n | humanos | voto YES GLM off | voto YES GLM on | voto YES Claude | YES Claude (paráfrases) | parse |
|---|---|---|---|---|---|---|---|
| 1Y5N | 388 | 0,167 | 0,719 | 0,510 | 0,278 | 0,293 | 0,979 |
| 3Y3N | 518 | 0,500 | 0,880 | 0,753 | 0,683 | 0,676 | 0,994 |
| 5Y1N | 645 | 0,833 | 0,964 | 0,918 | 0,910 | 0,911 | 0,984 |

0 recusas. Por idioma, 1Y5N: en 0,271 · es 0,287. Os 63 pedidos não parseados são **21 memes inteiros** (3 paráfrases cada) com `invalid_request_error: Could not process image`, em JPEGs válidos que o PIL e o GLM leem sem erro. Destes, 8 são do 1Y5N e não têm voto. Para essa contagem, o denominador continua sendo 388.

**Regra do §3:** voto YES 1Y5N = 108/388 = **0,278** (Wilson 95%: 0,236–0,325), **entre 0,25 e 0,40, logo PARCIAL.** O Claude fica bem abaixo do GLM (0,719 off / 0,510 on), mas acima do anotador humano típico (0,167). Mesmo que os 8 memes sem voto saiam todos YES, o valor fica em 116/388 = 0,299, ainda na faixa parcial.

Nos mesmos 388 memes, cruzando os votos (GLM off, Claude): (YES, NO) 171 · (YES, YES) 101 · (NO, NO) 101 · (NO, YES) 7 · sem voto do Claude 8.

**Desvio desta data (antes de ver qualquer resposta reenviada):** os 21 memes com "Could not process image" são reenviados uma vez com a imagem decodificada e reconvertida para PNG (mesmos pixels, sem nova compressão). Registros marcados com `image_reencoded: true`; os registros de erro originais ficam no JSONL. Se falhar de novo, fica `?` definitivo.

### Resultado final de 2026-10-19 (`think_off`, depois do reenvio em PNG)

Os 63 pedidos reenviados (21 memes, `image_reencoded: true`) foram todos processados. Parse **1,000**, 0 recusas.

| grupo | n | humanos | voto YES GLM off | voto YES GLM on | voto YES Claude (Wilson 95%) | YES Claude (paráfrases) |
|---|---|---|---|---|---|---|
| 1Y5N | 388 | 0,167 | 0,719 | 0,510 | **0,286** (0,243–0,333) | 0,292 |
| 3Y3N | 518 | 0,500 | 0,880 | 0,753 | 0,689 (0,648–0,728) | 0,678 |
| 5Y1N | 645 | 0,833 | 0,964 | 0,918 | 0,924 (0,901–0,942) | 0,911 |

1Y5N por idioma: en 0,271 · es 0,303.

**Decisão pela regra do §3: PARCIAL** (111/388 = 0,286, entre 0,25 e 0,40). O viés de YES é em grande parte do GLM: no 1Y5N, o Claude discorda de 175 dos 279 YES do GLM off e só diz YES sozinho em 7 memes. Mas não é só do GLM: o Claude também fica acima dos humanos no 1Y5N (0,286 contra 0,167) e no 3Y3N (0,689 contra 0,500).

Voto (GLM off, Claude) por grupo:
- 1Y5N: (YES, NO) 175 · (YES, YES) 104 · (NO, NO) 102 · (NO, YES) 7
- 3Y3N: (YES, YES) 333 · (YES, NO) 123 · (NO, NO) 38 · (NO, YES) 24
- 5Y1N: (YES, YES) 575 · (YES, NO) 47 · (NO, YES) 20 · (NO, NO) 2 · (sem voto GLM, YES) 1

### Nota de 2026-10-22 — análise descritiva pós-hoc (fora da regra do §3)

Concordância entre paráfrases, paráfrase dissidente, acordo com a maioria humana nos extremos, κ de Cohen, McNemar e cruzamentos GLM × Claude: calculados depois do resultado final, a partir dos mesmos arquivos, só como descrição. Estão em `RELATORIO_GLM_CLAUDE.md` §11–§14 (figuras de `plot_doc31.py`); o método está em `METODOLOGIA_GLM_CLAUDE.md` §7–§9. Nada aqui altera a decisão PARCIAL.
