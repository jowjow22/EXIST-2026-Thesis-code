# 25 — Anotador de referência (GLM) e ponderação de votos por desvio: pré-registro e especificação para o agente de código

**Escrito em 2026-10-04, ANTES de rodar.** Regra do projeto: nada neste documento muda depois de ver o primeiro resultado. Se algo tiver de mudar, a mudança entra como seção nova datada, com o motivo, nunca por edição do que está aqui.

**Revisão de 2026-10-06, antes de qualquer execução:** acrescentado o mínimo de 70% de memes em que o GLM está confiante, em cada idioma (§3, §6, §7, §11; portões G1 e G2).

**Para o agente de código.** Este documento é a especificação completa. Implemente exatamente o que está aqui. Onde houver ambiguidade, pare e pergunte ao Jonathan; nunca escolha um valor olhando resultados. Todo parâmetro marcado como fixado vai para `config.json` e é verificado por `assert` antes de cada fase. O treino na GPU (F5) é sempre disparado pelo Jonathan, nunca pelo agente.

**Diagramas.** O fluxo de cada fase está desenhado no draw.io (uma página por fase). Em caso de divergência entre diagrama e texto, vale o texto.

---

## 0. De onde isto vem

- **Pedido do orientador:** construir um "anotador perfeito" com um LLM, medir o desvio de cada anotador real em relação a ele e ponderar o voto de cada anotador por esse desvio. É um teste único, sabidamente fora da prática LeWiDi.
- **Papel do GLM:** não é o preditor final. Ele gera uma referência; os pesos dos anotadores produzem novos soft labels da Task 2.1 para treinar o **modelo de memes atual** (XLM-R + CLIP). Só a Task 2.1 muda; 2.2 e 2.3 ficam como estão.
- **Critério de sucesso do autor:** ganho mensurável em ICM ou `eval_loss` na Task 2.1. Contribuição só metodológica não basta.
- **Expectativa, dita antes:** a ponderação por MACE deu Δ = −0,0025 (p = 0,164); o desvio de cada anotador já medido no EXIST é um limiar pessoal estável (split-half 0,850), não descuido; a coleta tem 151 painéis fixos de 6 anotadores (~26 memes por painel; 868 dos 887 anotadores em um único painel). O que este teste traz de novo é uma referência externa aos votos. Os portões da F4 existem para descobrir sem GPU se a ponderação move os labels.

## 1. Ponto de partida no código

| Item | Estado atual |
|---|---|
| Script | `EXIST_MEMES/glm_zeroshot.py` (uso: `python3 EXIST_MEMES/glm_zeroshot.py N`). **Não editar**: o código novo vai em `EXIST_MEMES/glm_annotator/`. Reaproveitar dele a leitura do JSON, a resolução do caminho da imagem e o data URL base64 (MIME via `mimetypes`) |
| Ambiente das chamadas | Mac local, Python 3.13.5, SDK `openai` 2.14.0. Repositório montado via SSHFS em `~/Documents/projects/VM-THESIS`. Não precisa da VM |
| Servidor | `http://193.136.195.31:8000/v1`, `api_key="EMPTY"`, modelo `glm-4.6v-flash` (`cont1037/GLM-4.6V-Flash-FP8-DYNAMIC`, contexto 32 768) |
| Saída do modelo | raciocínio em `reasoning_content`; resposta em `content` no formato `<\|begin_of_box\|>YES<\|end_of_box\|>` |
| Texto do meme | campo `text`, limpo por `memes_core.pre_process_text` (mesma limpeza do pipeline) |
| Dados | `EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json`: 3 984 memes, 6 votos cada. Campos a confirmar no JSON antes de codar: `labels_task2_1` (lista de YES/NO), `annotators` (ids), `lang`, e os campos de gênero e idade usados por `compute_annotator_vector` (provavelmente `gender_annotators` e `age_annotators`) |
| Mac sem PyEvALL | o import do `pyevall` em `memes_core.py` já é opcional. Não instalar nada que rebaixe numpy/pandas. Sem SciPy: Spearman = Pearson dos ranks (`x.rank().corr(y.rank())`) |

## 2. Visão geral das fases

| Fase | O que faz | Chamadas ao GLM | Usa votos humanos? | Portão ao final |
|---|---|---|---|---|
| F0 | Cliente, parser, registro; testes de fumaça e sonda visual | ~50 | Não | G0 (visão íntegra) |
| F1 | Piloto: thinking ligado/desligado × 3 paráfrases; depois estabilidade | 1 440 + 480 | Não | G1 (braço elegível, com ≥ 70% de memes confiantes; flip ≤ 10%) |
| F2 | Referência: braço escolhido × 3 paráfrases nos memes de treino | ≈ 3·N_train − 720 | Não | G2 (parse ≥ 98% e ≥ 70% de memes confiantes, por idioma) |
| F3 | Pesos dos anotadores e novos soft labels (só cálculo) | 0 | Sim | testes unitários |
| F4 | Checagens (a)–(g) e portão de retreino | 0 | Sim | G4 (labels mudam ≥ 5% e ρ < 0,8) |
| F5 | Retreino do modelo de memes, BASE vs GLMW (Jonathan dispara) | 0 | Sim (avaliação) | leitura §11 |

Nenhuma decisão de F0 a F2 olha voto humano. Isso elimina a circularidade: a referência não é ajustada aos anotadores que ela vai auditar.

## 3. Parâmetros fixados

| Parâmetro | Valor | Por quê |
|---|---|---|
| `temperature` | **0,1** | decisão do autor: reduzir o ruído de amostragem em todos os testes |
| `top_p`, `top_k`, `repetition_penalty` | **0,6 · 2 · 1,1** (enviados explicitamente; `top_k` e `repetition_penalty` via `extra_body`) | valores do model card. O servidor aplica `generation_config.json` campo a campo se o cliente não mandar; mandar explícito torna a config auditável |
| `seed` | 42 na referência; 43 e 44 nas repetições de estabilidade | registrado por chamada |
| `max_tokens` | thinking ligado: **16 384** (recomendação do model card); thinking desligado: **256** | 2 048 truncava o raciocínio, e um raciocínio truncado deixa `content` vazio, o que provavelmente gerou os "?" do teste de 5 memes |
| Thinking desligado | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | o template acrescenta `/nothink` ao fim do texto do usuário; por isso o prompt termina com `\n` |
| Mensagem | uma mensagem `user`, sem system prompt; partes na ordem `[imagem, texto]` | igual ao script atual |
| Prompts | P1, P2, P3 da §4, texto exato; SHA-256 de cada um em `config.json` | paráfrases medem a sensibilidade à redação |
| Idioma da instrução | inglês para todos os memes; texto do meme no idioma original | evidência mista para instrução em espanhol; braço cortado |
| Voto do GLM por meme | maioria das 3 paráfrases | ímpar, sem empate |
| GLM "confiante" | as 3 paráfrases parseadas e iguais | só memes com GLM confiante entram no desvio |
| Mínimo de memes confiantes | **70%** em cada idioma, no piloto (G1) e na referência completa (G2) | abaixo disso, cada anotador fica com poucos memes para medir o desvio (§6) |
| Retentativas | só para erro de transporte (timeout, conexão, HTTP 5xx): até 3, com backoff. **Nunca** por falha de parse ou `finish_reason="length"` | repetir até "dar certo" seria seleção |
| Concorrência | `--concurrency 8` (registrar o valor usado) | a ordem de lote altera resultados numéricos no vLLM; manter igual entre braços |
| Piloto | 240 memes do split de treino: 120 `lang=en` + 120 `lang=es`, sorteio com `numpy.random.default_rng(2026)` sobre ids ordenados | a seleção usa só id e idioma, nunca votos |

## 4. Prompts (texto exato)

Proveniência. P1 adapta a pergunta mostrada aos anotadores de tweets (EXIST 2023, visão geral estendida, Seção 3.3) e a regra de escopo para memes (EXIST 2026, visão geral, Seção 2.1: "it is sexist itself, describes a sexist situation or criticizes a sexist behavior"). As frases "A meme that condemns or reports sexism is still YES" e "Judge the image and the text together" não vêm da EXIST; explicitam a regra. P2 e P3 dizem o mesmo com outra redação e outra ordem. As diretrizes de memes da EXIST não são públicas; isto é a melhor aproximação pública.

`{text}` é o texto já limpo por `pre_process_text`. Cada prompt termina com uma quebra de linha.

**P1**
```text
Is the meme sexist, in any form, or does it describe situations in which such discrimination occurs (i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behavior)?

YES: the meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.
NO: the meme does not prejudice, underestimate or discriminate women.
A meme that condemns or reports sexism is still YES.

Remember that sexism may be expressed in different forms! It may sound friendly, funny, offensive, hateful or even violent. Sexism comprises any form of oppression or prejudice against women because of their sex. This discrimination can be based on different beliefs such as stereotypes, the belief that men are superior to women or irrational hatred of women (misogyny). Note that misogyny is a radical/hate-driven sexism.

Judge the image and the text together. Text extracted from the meme (English or Spanish, do not translate it):
{text}

Answer with one word: YES or NO.
```

**P2**
```text
Sexism is any form of oppression or prejudice against women because of their sex. It can sound friendly, funny, offensive, hateful or violent, and it can rest on stereotypes, on the belief that men are superior to women, or on hatred of women (misogyny, the radical, hate-driven form of sexism).

Look at the meme, image and text together. Label it YES if it is sexist itself, if it describes a sexist situation, or if it criticizes a sexist behavior. A meme that condemns or reports sexism also counts as YES. Label it NO if it does not prejudice, underestimate or discriminate women.

Text extracted from the meme (English or Spanish, keep it in its original language):
{text}

Answer with one word: YES or NO.
```

**P3**
```text
Text extracted from the meme (English or Spanish, original language):
{text}

Task: decide whether this meme, considering the image and the text together, contains sexist expressions or behaviors. This includes memes that are sexist themselves, memes that describe a sexist situation, and memes that criticize or condemn a sexist behavior. Sexism means any oppression or prejudice against women because of their sex, whether it sounds friendly, funny, offensive, hateful or violent, including stereotypes, the belief that men are superior to women, and misogyny.

Is this meme sexist under this definition? Answer with one word: YES or NO.
```

## 5. F0 — Infraestrutura e testes de fumaça

### 5.1 Cliente (`glm_annotator/client.py`)

- `openai.AsyncOpenAI` com semáforo de concorrência; timeout por chamada de 600 s (thinking ligado) e 60 s (desligado).
- **Retomável:** antes de chamar, pular toda chave `(meme_id, arm, paraphrase, seed)` já presente no JSONL do braço. Uma chamada interrompida não é gravada; chamada gravada nunca é refeita.
- **Uma linha JSONL por chamada**, em `glm_reference/raw/<arm>.jsonl`, com: `meme_id`, `lang`, `arm` (`think_on`/`think_off`), `paraphrase` (1–3), `seed`, `params` (todos os parâmetros enviados), `prompt_sha256`, `image_sha256`, `content`, `reasoning_content`, `finish_reason`, `usage` (`prompt_tokens`, `completion_tokens`), `latency_s`, `label`, `parse_method`, `error`, `timestamp`, `server_model`.
- **Parser** (só em `content`; nunca em `reasoning_content`):
  1. primeiro trecho `<|begin_of_box|>…<|end_of_box|>`; normalizar (strip, maiúsculas, remover pontuação e `*`); se for `YES` ou `NO`, `parse_method="box"`;
  2. senão, se o `content` inteiro normalizado for `YES` ou `NO`, `parse_method="plain"`;
  3. senão, se aparecer exatamente um dos dois rótulos como palavra isolada, `parse_method="single_token"`;
  4. senão `label="?"`, com `parse_method` = `"empty"` (content vazio), `"length"` (`finish_reason="length"`), `"both"` (YES e NO presentes) ou `"other"` (recusa ou texto sem rótulo).

### 5.2 Testes de fumaça (portão G0)

| Teste | Como | Passa se | Se falhar |
|---|---|---|---|
| G0.1 servidor | `GET /v1/models`; tentar `GET /version` (vLLM) | `glm-4.6v-flash` listado; versão gravada (ou `"unknown"`) | parar |
| G0.2 thinking desligado | 3 memes com `enable_thinking=False` | `reasoning_content` vazio/None e `completion_tokens ≤ 32` nas 3 | remover o braço `think_off` da F1 (registrar; não abortar) |
| G0.3 sonda visual | 20 primeiros memes do piloto, prompt `"Describe the image in one sentence. Do not transcribe its text.\n"`, uma vez com a imagem real e uma com PNG branco 512×512. Salvar `probe_vision.csv` (id, caminho da imagem, descrição real, descrição branca) | **revisão manual do Jonathan:** as respostas para a imagem branca descrevem algo branco/vazio, e ≥ 16 de 20 descrições reais correspondem ao meme | **parar**: a torre de visão do checkpoint FP8 pode estar quebrada. Falar com quem mantém o servidor. Não é resultado sobre o método |
| G0.4 vazão | latência e chamadas/min com concorrência 8 nos testes acima | — (só registrar) | — |

O G0.3 vem de um auto-relato no README do build FP8 (`cont1037/...`): checkpoints públicos quantizados podem quantizar a torre de visão sem erro visível, com texto normal e imagem sem sentido. Não foi verificado de forma independente, por isso a sonda.

## 6. F1 — Piloto (escolha do thinking) e estabilidade

**Amostra:** 240 memes (§3), salvos em `pilot_ids.json`. A função de sorteio recebe só ids e idiomas.

**Braços:** `think_on` e `think_off` × P1, P2, P3, seed 42 → 2 × 3 × 240 = **1 440 chamadas**.

**Métricas sem rótulo** (por braço, total e por idioma), em `reports/pilot_summary.md`:
`parse_rate` (fração com label YES/NO), `length_rate`, `yes_rate`, `paraphrase_agreement` (fração de memes com as 3 paráfrases parseadas e iguais), contagem de recusas (`parse_method="other"`), fração de raciocínios com caracteres CJK (`think_on`), mediana de `completion_tokens` e de latência.

**Regra de escolha (fixada):**
1. Braço **elegível** se `parse_rate ≥ 0,98` em cada idioma **e** `0,05 ≤ yes_rate ≤ 0,95` (guarda contra um braço degenerado que responde sempre a mesma coisa) **e** `paraphrase_agreement ≥ 0,70` em cada idioma (fração de memes em que o GLM está confiante).
2. Os dois elegíveis: escolher `think_off`, a não ser que `paraphrase_agreement(on) − paraphrase_agreement(off) > 0,05`. Thinking desligado é o padrão porque é muito mais barato e porque a literatura não mostra ganho do raciocínio em classificação binária subjetiva.
3. Só um elegível: ele. Nenhum: **parar** (portão G1).

**Por que 70% de memes confiantes.** Cada anotador vota em cerca de 26 memes. Tirando os unânimes (cerca de 24% dos memes), sobram uns 20 que podem entrar no desvio. Com o GLM confiante em 70% deles, ficam uns 14 memes por anotador; com 50%, uns 10. O desvio é uma proporção, e o erro-padrão de uma proporção chega a 0,13 com 14 memes e a 0,16 com 10. O limite impede que os pesos fiquem ainda mais ruidosos do que já são. O valor de 70% é uma escolha do autor, não vem da literatura. A conta supõe que a confiança do GLM é igual em todos os memes; como os memes em que os anotadores divergem tendem a ser os mais difíceis, o número real de memes incluídos por anotador pode ser menor, e ele é reportado na checagem (g).

**Estabilidade (F1b):** braço escolhido, P1, seeds 43 e 44 nos 240 memes → **480 chamadas**. `flip_rate` = fração de memes cujos 3 labels (seeds 42, 43, 44) não são todos iguais; `?` conta como flip. Por idioma também.
**Portão G1:** `flip_rate ≤ 0,10` no total. Se maior, **parar**: a referência não é estável o bastante para pesar anotadores.

## 7. F2 — Referência

- **Memes:** todos os do **split de treino** do modelo de memes. O agente exporta `train_ids.json` e `val_ids.json` reproduzindo exatamente o split do `memes_model.ipynb` (`SPLIT_SEED` fixo) e confere as contagens contra as impressas no notebook. Os memes de validação **não** entram na referência nem no cálculo dos pesos.
- **Chamadas:** braço escolhido × P1–P3 × seed 42. As chamadas do piloto desse braço são reaproveitadas pela chave de retomada → ≈ 3·N_train − 720 chamadas novas.
- **Saída** `glm_votes.csv`: `meme_id, lang, label_p1, label_p2, label_p3, glm_vote, glm_confident`. `glm_vote` = maioria dos labels parseados quando pelo menos 2 concordam; senão vazio. `glm_confident` = 3 parseados e iguais.
- **Portão G2:** `parse_rate ≥ 0,98` **e** fração de memes com `glm_confident = True` ≥ 0,70, ambos em cada idioma. Se falhar, parar e reportar.

## 8. F3 — Pesos e novos soft labels (só cálculo)

Código em `glm_annotator/weights.py`, funções puras, sem I/O dentro delas. Implementação de referência e testes no Apêndice A; o código final tem de passar nesses testes.

**Entradas:** memes do split de treino; para cada meme, os votos `labels_task2_1` alinhados a `annotators`; `glm_votes.csv`. Conferir que cada lista tem o mesmo tamanho; votos fora de {YES, NO} são contados, reportados e removidos daquele meme.

**Memes que entram no cálculo do desvio** (os demais ficam fora só do desvio, não dos novos soft labels):
- **fora:** memes unânimes (todos os votos iguais). Neles os 6 anotadores do painel têm o mesmo desvio, qualquer que seja o voto do GLM; punir os 6 igualmente não distingue ninguém e só dilui o sinal;
- **fora:** memes em que o GLM não está confiante (`glm_confident = False`);
- **dentro:** todos os outros, empates 3-3 incluídos.

**Fórmula**, para o anotador *a* e o meme incluído *i*:
- `d_ai = 1` se o voto de *a* difere de `glm_vote_i`, senão 0;
- `H_i,−a` = entropia binária, em bits, da fração de YES entre os **outros** votos do meme (leave-one-out: o voto do próprio anotador não entra);
- peso da punição `w_ai = 1 − H_i,−a`. Com 5 outros votos: 5-0 → 1,000; 4-1 → 0,278; 3-2 → 0,029. Consenso entre os outros pune mais; divisão entre eles pune menos;
- desvio `D_a = Σ_i w_ai·d_ai / Σ_i w_ai` sobre os memes incluídos do anotador;
- peso do voto `r_a = 1 − D_a`. Anotador sem nenhum meme incluído (ou com Σ w = 0): `r_a = 1` e marcado em `no_included_memes`.

**Novos soft labels** para **todo** meme do split de treino: `p_new = Σ_a r_a·[voto_a = YES] / Σ_a r_a`; o soft label da Task 2.1 vira `[p_new, 1 − p_new]` na ordem de `LABELS_2_1 = ['YES', 'NO']`. Consequências que viram `assert`: memes unânimes mantêm 0 ou 1; um painel com todos os `r` iguais não muda nenhum soft label.

**Saídas:** `annotator_weights.csv` (`annotator_id, gender, age, n_memes, n_included, D_glm, r_glm, D_peer, r_peer, yes_rate, no_included_memes`) e `soft_labels_weighted.json` (`meme_id → {"old": [p, 1-p], "new": [p, 1-p]}`).

## 9. F4 — Checagens e portão de retreino

Código em `glm_annotator/checks.py`; relatório em `reports/checks_summary.md`. Tudo no split de treino.

| Checagem | O que calcula | Para que serve |
|---|---|---|
| (a) linha de base dos pares | mesma fórmula, mesmos memes incluídos, trocando `glm_vote_i` pela maioria dos outros 5 votos (`D_peer`, `r_peer`). ρ de Spearman entre `r_glm` e `r_peer`, anotadores com ≥ 1 meme incluído | se ρ for alto, o GLM não acrescenta nada ao consenso dos pares |
| (b) limiar | ρ de Spearman entre `D_glm` e a taxa de YES do anotador; e entre `D_glm` e \|taxa de YES do anotador − taxa de YES do GLM nos mesmos memes\| | mede se o peso só reflete a distância entre o limiar do anotador e o do GLM |
| (c) demografia | média de `r_glm` (e de `r_peer`, para contraste) por gênero × idade, com IC 95% por bootstrap de anotadores (2 000 reamostragens, `default_rng(2026)`) | LLMs tendem a concordar mais com anotadoras |
| (d) **portão** | `share_moved` = fração dos memes de treino com \|p_new − p_old\| ≥ 1/12 (meio voto; comparar com tolerância 1e-9); número de mudanças de rótulo duro (YES > 0,5 · NO < 0,5 · empate = 0,5), por idioma | decide o retreino |
| (e) GLM contra unanimidade | taxa com que o GLM confiante contraria um meme unânime, por idioma; 30 desses memes sorteados (`default_rng(2026)`) em `manual_review_unanimous.csv` (id, idioma, caminho da imagem, texto, votos, labels P1–P3, raciocínio se houver) | diagnóstico da qualidade da referência; o Jonathan revisa à mão |
| (f) concordância descritiva | acordo de `glm_vote` com o rótulo duro humano (empates fora), por idioma e por divisão de votos (6-0, 5-1, 4-2) | descrição; não escolhe nada |
| (g) precisão por anotador | distribuição de `n_included`; fração de anotadores com < 5 memes incluídos; largura mediana do IC 95% de `D_glm` por bootstrap dos memes de cada anotador | com ~26 memes por anotador, o desvio é ruidoso; isso tem de aparecer |

**Portão G4 (fixado):** o retreino só é autorizado se **`share_moved ≥ 0,05`** **e** **ρ(`r_glm`, `r_peer`) < 0,8**. Senão, parar e registrar o resultado (§11). A decisão vai para `reports/gate_decision.md`, com os dois números.

## 10. F5 — Retreino (Jonathan dispara na VM)

- **Mecanismo:** o mesmo usado no experimento de ponderação por MACE para trocar os soft labels da Task 2.1 no treino. O agente localiza esse código no repositório e o reaproveita; se não encontrar, pergunta.
- **Braços:** `BASE` (labels originais) e `GLMW` (Task 2.1 de treino trocada por `soft_labels_weighted.json`). Tudo o mais idêntico: Tasks 2.2/2.3, `demo_vec`, hiperparâmetros, split. Se as runs BASE do experimento MACE existirem com a mesma configuração (conferir `model_fingerprint()` e `meta.json`), reaproveitá-las em vez de retreinar.
- **Seeds e épocas:** as mesmas seeds pareadas (`INIT_SEED`) e o mesmo número de épocas e regra de estado reportado do experimento MACE; `SPLIT_SEED` fixo. Confirmar os valores no repositório antes de rodar.
- **Avaliação:** memes de validação com os **labels originais**. Primária: Task 2.1 ICM-Soft-Norm. Secundárias: Task 2.1 ICM-Hard-Norm e `eval_loss` (calculada contra os labels originais). Reportar 2.2 e 2.3 para ver dano colateral.
- **Estatística:** Δ por seed (`GLMW − BASE`), média e o mesmo teste pareado do experimento MACE.

## 11. Leitura pré-registrada

| Desfecho | Leitura |
|---|---|
| G0.3 falha | caminho visual quebrado; parar. Não diz nada sobre o método |
| G1 falha (nenhum braço elegível ou `flip_rate > 0,10`) | o GLM não serve como referência estável nesta configuração; registrar como limite e encerrar |
| G2 falha (parse < 0,98 ou menos de 70% de memes confiantes em algum idioma) | a referência completa não reproduz o piloto; registrar como limite e encerrar |
| G4 falha por `share_moved < 0,05` | a ponderação não move os labels; resultado nulo por construção, sem gastar GPU; encerra |
| G4 falha por ρ ≥ 0,8 | o GLM reproduz o consenso dos pares, família já testada com MACE; encerra |
| F5: média de Δ ICM-Soft-Norm > 0, p < 0,05 e `eval_loss` sem piora | hipótese sustentada. Só então variações (mapeamento D→r, thinking) podem ser discutidas, em seção nova |
| F5: Δ nulo ou negativo | encerra; soma-se às evidências da linha de reliability |

**Nenhum desfecho autoriza** ajustar prompt, temperatura, limiares, agregação ou o mapeamento D→r e rodar de novo mirando um resultado melhor. É uma medição, não uma busca.

## 12. Portões de execução (falha dura, não aviso)

1. `config.json` gravado antes da F1 e conferido por `assert` no início de cada fase: parâmetros da §3, SHA-256 dos 3 prompts, concorrência, versão do servidor.
2. A função de sorteio do piloto não recebe votos (assinatura só com ids e idiomas).
3. `set(train_ids) ∩ set(val_ids) == ∅`, e o cálculo de pesos só toca `train_ids`.
4. Testes do Apêndice A passam antes de rodar a F3 de verdade.
5. Depois da F3: todo soft label em [0, 1] e somando 1; memes unânimes com `p_new == p_old`; labels de validação idênticos aos originais (hash igual).
6. Antes da F5: o arquivo de soft labels carregado no notebook tem o mesmo SHA-256 do gerado na F3, e só a coluna da Task 2.1 de treino difere entre `BASE` e `GLMW`.

## 13. Estrutura de arquivos

```
EXIST_MEMES/glm_annotator/
  client.py        # chamadas, parser, retomada, JSONL
  prompts.py       # P1–P3 exatos + hashes
  run_probe.py     # F0
  run_pilot.py     # F1 e F1b
  run_reference.py # F2
  weights.py       # F3 (funções puras)
  checks.py        # F4
  tests/test_weights.py
EXIST_MEMES/glm_reference/
  config.json  pilot_ids.json  train_ids.json  val_ids.json
  raw/think_on.jsonl  raw/think_off.jsonl
  probe_vision.csv  glm_votes.csv
  annotator_weights.csv  soft_labels_weighted.json
  manual_review_unanimous.csv
  reports/pilot_summary.md  reports/reference_summary.md
  reports/checks_summary.md  reports/gate_decision.md
```

## 14. Custo e tempo

| Fase | Chamadas | Thinking desligado (~1,5 s/chamada, hipótese) | Thinking ligado (~12 s/chamada, medido em 5 memes) |
|---|---|---|---|
| F0 | ~50 | minutos | minutos |
| F1 | 1 440 + 480 | 720 chamadas ≈ 0,3 h; estabilidade ≈ 0,2 h se este braço for o escolhido | 720 chamadas ≈ 2,4 h; estabilidade ≈ 1,6 h se este braço for o escolhido |
| F2 (N_train ≈ 3 200, se o split for 80/20) | ≈ 8 900 | ≈ 3,7 h sequencial | ≈ 30 h sequencial |
| F3–F4 | 0 | minutos | minutos |

Com concorrência 8 o tempo real deve cair bastante, mas o ganho depende da GPU do servidor; medir no G0.4. O tempo de GPU da F5 é o de um experimento como o do MACE (ou metade, se as runs BASE forem reaproveitadas). Trabalho humano estimado: ~1,5 dia de código, ~0,5 dia de análise do piloto, ~1,5 a 2 dias para pesos e checagens, ~0,5 dia para o retreino.

## 15. O que NÃO fazer

- Não ajustar prompt, temperatura, limiar, agregação ou o mapeamento D→r olhando voto humano.
- Não parsear `reasoning_content`.
- Não editar `glm_zeroshot.py` nem os notebooks de treino originais; criar cópias.
- Não mudar os labels das Tasks 2.2 e 2.3, nem os de validação.
- Não rodar a F2 antes do G1 decidir o braço, nem a F5 sem o G4 autorizar.
- Não refazer chamadas que falharam no parse para "conseguir" um rótulo.

## 16. Decisões a confirmar pelo Jonathan antes de congelar

Estes valores foram propostos na conversa e ainda não foram confirmados com o orientador. Confirmados, viram fixados e a seção é datada.
1. `temperature=0,1` com `top_p=0,6`, `top_k=2`, `repetition_penalty=1,1` (valores do model card para os demais).
2. Limiares: parse ≥ 0,98; flip ≤ 0,10; `share_moved` ≥ 0,05; ρ < 0,8. (Memes confiantes ≥ 0,70 por idioma: **fixado pelo Jonathan em 2026-10-06**.)
3. Thinking desligado como padrão, salvo ganho > 0,05 em concordância entre paráfrases.
4. Mapeamento `r = 1 − D`.
5. Seeds, épocas e teste estatístico da F5 iguais aos do experimento MACE.

## 17. Resultado

*(preencher depois de rodar, como seção datada; não editar as seções acima)*

---

## Apêndice A — Implementação de referência da fórmula e testes

O código de `weights.py` pode ser organizado de outro jeito, mas tem de produzir os mesmos números.

```python
from math import log2

def binary_entropy(p: float) -> float:
    return 0.0 if p in (0.0, 1.0) else -(p * log2(p) + (1 - p) * log2(1 - p))

def annotator_deviation(memes: dict, glm_vote: dict) -> dict:
    """memes: {meme_id: {annotator_id: 'YES'|'NO'}}
    glm_vote: {meme_id: 'YES'|'NO'|None}  (None = GLM não confiante)
    Retorna {annotator_id: D_a or None}."""
    num, den = {}, {}
    for mid, votes in memes.items():
        anns = list(votes)
        n_yes = sum(v == 'YES' for v in votes.values())
        if n_yes in (0, len(anns)):          # unânime: fora do desvio
            continue
        g = glm_vote.get(mid)
        if g is None:                        # GLM não confiante: fora
            continue
        for a in anns:
            others = [votes[b] for b in anns if b != a]
            w = 1 - binary_entropy(sum(o == 'YES' for o in others) / len(others))
            d = int(votes[a] != g)
            num[a] = num.get(a, 0.0) + w * d
            den[a] = den.get(a, 0.0) + w
    all_anns = {a for v in memes.values() for a in v}
    return {a: (num[a] / den[a] if den.get(a, 0) > 0 else None) for a in all_anns}

def vote_weights(D: dict) -> dict:
    return {a: (1.0 if d is None else 1.0 - d) for a, d in D.items()}

def weighted_soft_label(votes: dict, r: dict) -> float:
    return sum(r[a] * (v == 'YES') for a, v in votes.items()) / sum(r[a] for a in votes)
```

**Caso de teste (painel de 6 anotadores A–F, 5 memes):**

| Meme | Votam YES | Divisão | Voto do GLM | Situação |
|---|---|---|---|---|
| m1 | A B C D E F | 6-0 | NO | unânime contrariado pelo GLM → fora do desvio |
| m2 | A B C D E | 5-1 | YES | F discorda sozinho; GLM com a maioria |
| m3 | A B C D | 4-2 | NO | GLM com a minoria |
| m4 | A B C | 3-3 | YES | empate |
| m5 | B C D E F | 5-1 | NO | A discorda sozinho; GLM com A |

Valores esperados (tolerância 1e-4):

| Anotador | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| D_a | 0,0217 | 0,5000 | 0,5000 | 0,5473 | 0,3558 | 0,8246 |
| r_a | 0,9783 | 0,5000 | 0,5000 | 0,4527 | 0,6442 | 0,1754 |

Conta de F, para conferir: m2 (outros 5-0 → w = 1, d = 1), m3 (outros 4-1 → w = 0,2781, d = 0), m4 (outros 3-2 → w = 0,0290, d = 1), m5 (outros 4-1 → w = 0,2781, d = 1). D = (1 + 0,0290 + 0,2781) / (1 + 0,2781 + 0,0290 + 0,2781) = 1,3071 / 1,5852 = 0,8246.

| Meme | p_old | p_new |
|---|---|---|
| m1 | 1,0000 | 1,0000 |
| m2 | 0,8333 | 0,9460 |
| m3 | 0,6667 | 0,7478 |
| m4 | 0,5000 | 0,6086 |
| m5 | 0,8333 | 0,6991 |

**Testes de propriedade:**
1. `binary_entropy(0.2) ≈ 0.7219`, `binary_entropy(0.4) ≈ 0.9710`, `binary_entropy(0.5) == 1.0`, `binary_entropy(0) == 0`.
2. Trocar o voto do GLM em m1 (unânime) de NO para YES não muda nenhum `D_a`.
3. Se nenhum meme é incluído, todo `r_a = 1` e todo `p_new == p_old`.
4. Num meme unânime, `p_new == p_old` para quaisquer pesos.
5. Se todos os `r_a` de um painel forem iguais, nenhum soft label do painel muda.
6. m4 mostra que um empate 3-3 pode virar YES depois da ponderação; isso conta como mudança de rótulo duro na checagem (d).

---

## 18. Adendo de 2026-10-06 — decisões confirmadas e encaixe no repositório (antes de qualquer execução)

Seção nova, como manda a regra do topo. Nada acima foi editado. Nenhuma chamada ao GLM tinha sido feita quando isto foi escrito.

**§16 confirmada pelo Jonathan em 2026-10-06.** Os cinco itens viram fixados com os valores propostos: `temperature=0,1`, `top_p=0,6`, `top_k=2`, `repetition_penalty=1,1`; limiares parse ≥ 0,98, flip ≤ 0,10, `share_moved` ≥ 0,05, ρ < 0,8, confiantes ≥ 0,70 por idioma; thinking desligado salvo ganho > 0,05; `r = 1 − D`; seeds, épocas e teste da F5 iguais aos do MACE. Todos estão em `glm_annotator/common.py` e são conferidos contra `config.json` por `assert`.

**Decisões do Jonathan sobre pontos em aberto (2026-10-06):**
- **BASE é retreinado** na F5. Motivo: não existe `meta.json` para os memes (o `model_fingerprint()` só é impresso nas células), e `memes_model.ipynb` mudou depois das runs MACE, então a §10 ("reaproveitar se a configuração for a mesma") não tem como ser verificada.
- **Checagem (b), segunda correlação:** a taxa de YES do anotador e a do GLM são calculadas sobre os **memes incluídos** daquele anotador (os mesmos que entram no `D_glm`). A primeira correlação usa a `yes_rate` do anotador em todos os memes dele (coluna de `annotator_weights.csv`).
- **O agente de código só escreve código.** Todas as fases que chamam o servidor são disparadas pelo Jonathan.

**Encaixe no repositório (fatos verificados, não escolhas):**
- **Split.** O split do `memes_model.ipynb` está congelado em `reliability/outputs/split_taskA_frozen.json` (`random_state=42`, `test_size=0,15`, estratificado pelo voto majoritário do `raw`), e o notebook o carrega. Treino **3 386**, validação **598**: são as contagens impressas pelo notebook. `run_probe.py` recalcula o split com `reliability/make_soft_labels_taskA.build_frozen_split` e exige igualdade com o arquivo antes de exportar `train_ids.json`/`val_ids.json`.
- **§14 subestimava a F2.** Com N_train = 3 386, a F2 tem ≈ 3·3 386 − 720 = **9 438** chamadas novas, e não ≈ 8 900. A §14 era estimativa; nada muda nos limiares.
- **`INIT_SEED` ≡ `SEED`.** No modelo de memes não existe `INIT_SEED`: `SEED` alimenta `seed` e `data_seed` do `TrainingArguments`, e o split vem do arquivo congelado, independente de `SEED`. As seeds pareadas do MACE são `SEED ∈ {42, 7, 2025}`, com 15 épocas, estado reportado em E = argmin(`eval_loss`) e estado final (regra de simetria), e t pareado com df = 2 (`reliability/analyze_arm_experiment.py`).
- **Formato dos soft labels para a F5.** O mecanismo do MACE é a célula 6 do `memes_model.ipynb`: `ARM` escolhe `../reliability/outputs/soft_labels_taskA_{ARM}.json`, um dicionário `{id_EXIST: [p_YES, p_NO]}` com os 3 984 memes. A validação é sempre remapeada para `soft_labels_taskA_raw.json`. Por isso a F3 grava, além de `soft_labels_weighted.json` (formato da §8):
  - `reliability/outputs/soft_labels_taskA_glmw.json`: treino = `p_new`; validação copiada de `_raw.json` sem alteração;
  - `reliability/outputs/soft_labels_taskA_glmbase.json`: cópia byte a byte de `_raw.json`, só para o BASE retreinado ter uma tag própria (`glmbase_seed*`) e não se misturar às runs `raw_seed*` de 15 épocas do MACE.
  Os dois SHA-256 vão para `config.json` e são o que o portão 12.6 confere. O notebook não é editado; muda-se só `ARM` e `SEED`.
- **Falhas de transporte** que esgotam as 3 retentativas **não** são gravadas no JSONL (chamada interrompida, §5.1). Vão para `raw/<arm>.errors.jsonl`, e a fase seguinte recusa rodar enquanto faltar chamada. Erros HTTP 4xx (não são transporte, logo não há retentativa) são gravados com `label="?"` e `parse_method="error"`. É um valor que a §5.1 não prevê; existe para que um erro determinístico não seja refeito para sempre. **Exceção: HTTP 429** (servidor saturado) é tratado como transporte, com retentativa e sem gravação. Gravá-lo derrubaria o `parse_rate` por motivo de infraestrutura, e não haveria como recuperar sem editar o JSONL à mão. Refazer um 429 não é seleção: o modelo não chegou a responder.
- **Métricas de agregação não especificadas na §6**, resolvidas sem olhar resultado: `yes_rate` = fração de YES entre as chamadas parseadas, no total (a §6 só pede "em cada idioma" para parse e concordância); o ganho de concordância da regra 2 é comparado no total; `length_rate` vem de `finish_reason == "length"`, não de `parse_method`.
- **G0.2** usa P1 nos 3 primeiros memes do piloto. **G0.3** roda no braço `think_off` se o G0.2 o manteve; senão no `think_on`. Nenhuma chamada da F0 vai para `raw/<arm>.jsonl`.

## 19. Adendo de 2026-10-07 — concorrência 4 (antes de qualquer chamada de chat)

Decisão do Jonathan: **`--concurrency 4`** no lugar do 8 da §3. Motivo: não sobrecarregar o servidor GLM, que é partilhado. A F0 de 2026-10-06 parou no G0.1 (servidor inacessível) antes de qualquer chamada de chat. O `config.json` dela, que só tinha `concurrency: 8` e `created`, foi apagado, e não existia `raw/`. A concorrência é a mesma em todos os braços e fases, como pede a §3. Ela muda só a vazão, não os parâmetros de amostragem.

Mudança só de observabilidade: cada request imprime uma linha com o status (✓/✗), o ID do meme e a resposta do modelo ou o erro. Nenhum voto humano é exibido.

## 20. Resultado de 2026-10-08 (preenche o §17) — G2 falhou, experimento encerrado

Seção datada, como manda o §17; nada acima foi editado.

| Portão | Resultado |
|---|---|
| G0 | aprovado. Sonda visual 20/20 + 20/20 (revisão manual); vLLM `0.29.1rc1.dev452+g3df4ae153` |
| G1 | aprovado. Os dois braços foram elegíveis; ganho do `think_on` de 0,008 < 0,05 → `think_off`. `flip_rate` 0,0042 |
| **G2** | **falhou.** Confiantes: en 0,7265 · **es 0,6912** (1 155/1 671; o mínimo era 1 170). `parse_rate` 0,9983 / 0,9988 |

As 15 chamadas sem rótulo são HTTP 400 em 5 memes (3 en, 2 es): a imagem excede o cache de encoder do servidor (6 084 tokens). Mesmo com os 2 memes em espanhol confiantes, o resultado seria 1 157 < 1 170, então a falha não se deve a isso.

**Leitura (§11):** a referência completa não reproduz o piloto (es 0,725 → 0,691). Fica registrado como limite, e o experimento termina aqui. F3, F4 e F5 não foram executadas e nenhum peso foi calculado. Detalhes e achados descritivos (a P3 é a dissidente em 75% dos memes não confiantes) estão em `NOTAS_PARA_TESE.md` §4.

Nada aqui autoriza ajustar paráfrases, agregação ou limiares e rodar de novo (§11, §15). Uma continuação seria um experimento novo, com pré-registro próprio e declarado como motivado por este resultado.
