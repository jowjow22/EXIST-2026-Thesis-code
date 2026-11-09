# 33 — A Persona como referência: votos no treino inteiro e ponderação de anotadores pelo método do doc 27

**Pré-registro escrito em 2026-10-29, antes de qualquer chamada desta onda.** Mudanças depois de rodar só como seção nova datada.

**Estado: APROVADO pelo autor em 2026-10-29, antes de qualquer envio: decisões 1 a 5 do §0, incluindo a 5 (repetir o piloto).**

**Estatuto: exploratório**, como o doc 27. É uma continuação declarada dos docs 27 (GLM como referência) e 32 (Persona nos extremos). Não reabre nenhum dos dois.

## 0. Decisões propostas (a confirmar pelo autor)

1. **Votos só no treino.** A Persona vota nos 1 835 memes de treino que ainda não têm voto; os 1 551 extremos do doc 32 são reaproveitados. A validação (598) não recebe voto: os labels dela ficam crus, como no doc 27, e votar nela não serve ao método.
2. **Peso r = 1 − D**, a fórmula do doc 27 (SPEC_27 §8), sem nenhuma mudança. O r⁴ do doc 28 fica para depois, como triagem opcional numa seção datada própria.
3. **Portão G2 completo** do doc 25 (definição no SPEC_25 §7; leitura no §11): parse ≥ 0,98 **e** fração de memes confiantes (3-0) ≥ 0,70, nos dois idiomas. O GLM falhou nesse portão, e o doc 27 seguiu com o G2′ (só parse). Aqui o G2 completo vale desde o início; se falhar, a onda para (§3).
4. **BASE reaproveitado:** as runs `glmbase` do doc 27 (seeds 42, 7, 2025), **se** uma nova run `glmbase` seed 42 reproduzir a antiga exatamente (§7).
5. **Medir o ruído entre rodadas:** repetir os 180 pedidos do piloto do doc 32 (60 memes × S1–S3), num arquivo separado, por ≈ US$ 0,33. É descritivo (§2, §9) e não entra em nenhum portão.

## 1. Por que

Nos docs 27 e 28, ponderar os anotadores pela distância ao GLM não melhorou o modelo de memes.

- **O diagnóstico do doc 27:** o peso media sobretudo o **limiar de YES** de cada anotador (ρ(D_glm, taxa de YES) = −0,676), e o GLM diz YES demais: 0,804 por resposta no treino, contra 0,555 dos humanos.
- **A Persona é calibrada:** 0,526 por resposta nos 1 551 extremos (voto 0,531), contra 0,555 dos humanos nos mesmos memes.
- **Faixa nula do GLM** `[MEDIDO em 2026-10-29, descritivo]`: com 200 permutações das triplas de rótulos do GLM entre memes, dentro de cada idioma, ρ(b) cai em [−0,709; −0,649] e ρ(gap) em [0,646; 0,714]. Os observados (−0,676 e 0,710) caem **dentro** da faixa: o vínculo do peso do GLM com o limiar é o que a taxa de YES do GLM produz sozinha. Só o ρ(a) observado (0,501) fica acima da faixa nula ([0,195; 0,289]), sinal de alguma informação por meme. `share_moved` fica em [0,293; 0,344] contra 0,346 observado.

**A pergunta desta onda:** com uma referência sem viés de taxa, o peso passa a carregar informação por meme, e isso muda o resultado do retreino?

## 2. Votos no treino inteiro (F2 da Persona)

- **O que roda:** `claude_reference_32.py --scope train`. Usa os mesmos `claude-sonnet-5-5`, `think_off`, S1–S3 (mesmos SHA-256 do SPEC_32 §2), tags `<meme_text>`, `effort` e Batch API do doc 32. Os 1 835 memes × S1–S3 = **5 505 pedidos**.
- **Onde grava:** `claude_reference_32/raw/claude-sonnet-5-5_think_off_train.jsonl`, separado do JSONL dos extremos, que o `analyze_claude32.py` e o SPEC_32 §7 leem como está. Os lotes entram no mesmo `batches.json`, marcados com `"scope": "train"`.
- **Os 1 835 memes:** o treino menos os extremos. Divisões humanas: 0 YES (316), 2 YES (458), 4 YES (585), 6 YES (476); en 935, es 900. Checado em 2026-10-29:
  - nenhum tem texto vazio nem imagem acima de 5 MB;
  - 1 702 são JPEG e 133 PNG;
  - ≈ 250 MB de imagens, ou ≈ 1 GB enviado (3 pedidos por meme, em base64), em ~7 lotes de até 150 MB.
- **Imagens:** um erro "Could not process image" é reenviado uma vez em PNG ao rodar o mesmo comando de novo, como no doc 31. Um PNG acima de 5 MB é reduzido para 1 568 px de lado. Se falhar de novo, fica `?` definitivo.
- **Envio robusto:** cada lote é gravado como intenção antes de ser criado, e o POST de criação vai sem retry automático do SDK. Um envio interrompido é recuperado da API na execução seguinte em vez de pago de novo, e registros perdidos de um lote já recolhido são baixados de novo (grátis por 29 dias). Testado em `tests/test_persona33_offline.py`.
  - **Consequência prática:** sem retry, um erro passageiro da API (429/529) na criação de um lote faz o comando parar no meio do upload. Rodar o mesmo comando de novo é seguro: a intenção é adotada ou descartada, e só o upload daquele lote se repete.
- **Sem `fallbacks`** e sem trocar de modelo. Uma recusa fica `?`.
- **Junção:** `persona_ref.py votes` junta os dois JSONL (vale o último registro de cada chave). Ele:
  - confere que cobrem exatamente treino × S1–S3 e que todo registro tem o SHA-256 de S1–S3;
  - recusa rodar enquanto houver reenvio PNG pendente;
  - grava `claude_reference_32/persona_ref/persona_votes.csv` (`meme_id, lang, label_p1..p3, vote, confident`), com o SHA-256 em `config.json` (`runtime.doc33`).

**Calibração fora dos extremos (descritivo, registrado agora):** os S1–S3 foram escritos depois do doc 31 e avaliados nos mesmos 1 551 extremos (Relatório Persona §13). Os memes com 0, 2, 4 e 6 YES humanos não foram usados para avaliar S1–S3. `persona_ref.py votes` reporta, por número de YES humanos, a fração de YES de cada prompt e do voto contra k/6, separando novos, extremos e treino inteiro, com o GLM off ao lado.

Leitura de referência (não é regra de decisão), para a fração de YES das 3 respostas nos 1 835 novos contra a taxa humana deles (0,555: 6 112 YES em 11 010 votos):

| distância | leitura |
|---|---|
| ≤ 0,05 | a calibração dos extremos se mantém fora deles |
| 0,05 a 0,10 | inconclusivo |
| > 0,10 | não se mantém, e a tese diz isso |

**Ruído entre rodadas (decisão 5):** `claude_reference_32.py --scope repeat` reenvia os 60 memes do piloto do doc 32 para `..._repeat.jsonl`. `persona_ref.py drift` conta os rótulos e votos que mudaram em relação ao piloto original. É descritivo e serve à limitação (1) do §9.

## 3. Portão G2

Sobre os 3 386 memes de treino, em cada idioma: **parse ≥ 0,98 e confiante ≥ 0,70**. Calculado por `persona_ref.py votes` e gravado em `gates.doc33_G2`.

- **Aprovado:** segue para a F3.
- **Basta um idioma falhar para a onda encerrar.** Não se segue com um só idioma nem com o G2′ dentro do doc 33. Qualquer continuação é um documento novo, aprovado pelo orientador.

O portão é calculado em parte sobre dados já conhecidos: nos extremos, a Persona teve 86,8 % (en) e 82,7 % (es) de memes 3-0. Por isso o `votes` também reporta o G2 só nos 1 835 novos, como descritivo.

## 4. F3 — Pesos, soft labels `personaw` e o controle `personashift`

- **Fórmulas:** as do SPEC_27 §8, sem mudança, com `p_Persona,i` (fração de YES entre S1, S2 e S3) no lugar de `p_GLM,i`:
  - `d_ai = |1[voto_a = YES] − p_Persona,i|`;
  - `w_ai = 1 − H(fração de YES entre os outros 5 votos)`;
  - `D_a = Σ w·d / Σ w`; `r_a = 1 − D_a` (sem meme incluído: `r_a = 1`);
  - `p_new = Σ r·1[YES] / Σ r`, em todo meme de treino.
- **Memes incluídos no desvio:** não unânimes entre os humanos e com as três respostas da Persona parseadas.
- **Código:** `persona_ref.py build`, que chama as funções puras de `weights.py` (sem cópia) e roda os testes do Apêndice A do SPEC_27 antes.
- **Teste de equivalência** (`tests/test_persona33_offline.py`, feito em 2026-10-29 antes deste pré-registro valer). Alimentado com os votos do GLM, o código desta onda reproduz exatamente o doc 27:
  - `annotator_weights.csv` (D, r, D_peer, r_peer), `soft_labels_weighted.json` e `soft_labels_taskA_glmw.json`;
  - o G4 (`share_moved` 0,3461; ρ(a) 0,5008) e a checagem (b) (−0,676);
  - as tabelas (c), (d), (e), (f), (h) e (i) de `reports/checks_summary.md`, com os pesos gravados com os nomes da Persona;
  - `vote`/`confident` de `glm_votes.csv` e o G2 do doc 25 (en 0,7265 · es 0,6912; falha).

  Uma diferença entre Persona e GLM não vem, portanto, do código.
- **Asserts (portão 12.5 do doc 25):** soft labels em [0, 1]; memes unânimes com `p_new == p_old`; validação byte a byte igual ao `raw`; 3 984 chaves.
- **Controle `personashift`, congelado no mesmo passo:** igual ao `glmshift` do SPEC_27 §16C (`control_glmshift.shifted`). Nos memes de treino não unânimes, `p_ctrl = clip(p_old + Δ̄, 0, 1)`, com Δ̄ = deslocamento médio do `personaw` nesses memes. Ele é gerado antes de qualquer run da F5, com o SHA-256 e o Δ̄ em `runtime.doc33.control`.
- **Congelamento:** vale pelo `config.json` (`runtime.doc33.sha256`), gravado numa escrita só junto com o portão `doc33_F3`. Se o `build` cair antes dessa escrita, roda de novo; depois dela, só confere. O `votes` recusa regravar os votos depois do `build`.
- **Saídas:**
  - `claude_reference_32/persona_ref/annotator_weights_persona.csv` e `soft_labels_weighted_persona.json`;
  - `reliability/outputs/soft_labels_taskA_personaw.json` e `soft_labels_taskA_personashift.json`.

  Nada do GLM é reescrito.

## 5. F4 — Checagens e portão G4

`persona_ref.py checks` confere primeiro os SHA-256 dos votos, pesos e soft labels congelados no `build`. Depois roda, com a Persona no lugar do GLM, as checagens (a)–(i) do SPEC_27 §9 e §16A, incluindo as parcelas por anotador (`annotator_deviation_parcels_persona.csv`). Mais duas:

- **(j) Persona × GLM:** ρ(r_persona, r_glm) nos anotadores com meme incluído nas duas referências; a checagem (b) lado a lado; r médio.
- **(k) Faixa nula:** 200 permutações das triplas S1–S3 entre memes, dentro de cada idioma (`default_rng(2026)`). Reporta a faixa p2,5–p97,5 de ρ(a), ρ(b), ρ(gap) e `share_moved`, para a Persona e, como contexto, para o GLM (§1).

**Portão G4** (o do doc 27): `share_moved ≥ 0,05` **e** ρ(r_persona, r_peer) < 0,8. Se falhar, a onda encerra sem gastar GPU.

**Leitura pré-registrada da checagem (b)**, a pergunta central da onda. Uma referência calibrada já derruba ρ(b) sozinha, então a leitura usa a faixa nula:

| condição (nesta ordem) | leitura |
|---|---|
| ρ(a) observado ≤ p97,5 da faixa nula | o peso não carrega informação por meme além da taxa de YES da referência |
| \|ρ(b)\| < 0,40 **e** ρ(gap) < 0,40 | o peso deixou de medir sobretudo o limiar de YES e carrega informação por meme |
| \|ρ(b)\| > 0,60 | o peso continua medindo sobretudo o limiar, como no GLM |
| outro caso | parcial |

O `checks` imprime e grava essa leitura (`gates.doc33_G4.reading_b`). Não é regra de decisão: a F5 roda se o G4 passar, qualquer que seja a (b).

## 6. Leitura pré-registrada dos desfechos

| Desfecho | Leitura |
|---|---|
| G2 falha (em qualquer idioma) | a Persona não é estável o bastante no treino inteiro; encerra (§3) |
| G4 falha por `share_moved < 0,05` | a ponderação não move os labels; nulo por construção, sem GPU |
| G4 falha por ρ ≥ 0,8 | a Persona reproduz o consenso dos pares; encerra |
| F5: Δ médio de ICM-Soft-Norm > 0, p < 0,05 e `eval_loss` sem piora | evidência exploratória a favor de uma referência calibrada; roda o controle `personashift` (§7) |
| F5: Δ médio de ICM-Soft-Norm < 0 e p < 0,05 | efeito negativo, registrado como resultado |
| F5: qualquer outro caso (por exemplo Δ > 0 com p ≥ 0,05, ou `eval_loss` pior) | efeito não detectado com 3 seeds; encerra |

**Nenhum desfecho autoriza** trocar de variante (r⁴, entropia, outro D→r), mudar S1–S3 ou limiares e calcular de novo mirando um resultado melhor dentro desta onda.

## 7. F5 — Retreino (o autor dispara na VM)

- **Braço tratado:** `personaw` = Task 2.1 de treino trocada por `p_new`. A validação fica sempre crua; as Tasks 2.2 e 2.3 ficam intocadas.
- **Notebook:** `memes_model.ipynb`, sem edição além de `ARM = 'personaw'` e `SEED` na célula 6. Antes do treino, a célula de verificação `from analyze_f5 import precheck; precheck(ARM, _SOFT_A)` confere o SHA-256 congelado e o portão `doc33_G4`. Os arquivos `soft_labels_taskA_personaw.json` e `config.json` precisam estar na cópia do repositório que a VM usa.
- **Seeds:** 42, 7, 2025; 15 épocas.
- **BASE.** Antes das runs `personaw`, uma run `ARM = 'glmbase'`, `SEED = 42`.
  - **"Reproduz"** = as 15 `eval_loss` iguais, sem tolerância, às da run `20261009_144803_glmbase_seed42`. O treino é determinístico: as duas runs completas de `glmbase` seed 7 do doc 27 têm as 15 `eval_loss` idênticas.
  - O `analyze_f5.py` checa isso sozinho: duas runs do mesmo braço e seed com `eval_loss` diferente fazem a análise parar.
  - **Se reproduzir:** as três `glmbase` do doc 27 servem de BASE.
  - **Se não reproduzir:** o ambiente mudou. As runs `glmbase` antigas vão para `metrics_history/doc27_glmbase/` (fora da busca) e as três seeds são retreinadas. Os relatórios F5 dos docs 27 e 28 ficam como artefatos congelados; para reproduzi-los, os arquivos arquivados têm de voltar para `metrics_history/`.
- **Análise:** `analyze_f5.py --treat personaw`, igual à do doc 27:
  - Δ = personaw − glmbase por seed;
  - **estado primário E = argmin(`eval_loss`)**; estado final secundário;
  - t pareado **bilateral**, df = 2;
  - métrica primária: Task A ICM-Soft-Norm. Secundárias: ICM-Norm hard e `eval_loss`; Tasks B e C para dano colateral.
- **Critério:** Δ médio de ICM-Soft-Norm > 0, p < 0,05 e `eval_loss` sem piora (= Δ médio de `eval_loss` ≤ 0 em E, como no `analyze_f5.py`).
- **Controle `personashift`:** já congelado no `build` (§4). **Só roda se o critério for satisfeito**, com as mesmas seeds, 15 épocas e `ARM = 'personashift'`. A análise é `analyze_f5.py --treat personaw --base personashift`, com a leitura do SPEC_27 §16C: Δ(personaw − personashift) > 0 com p < 0,05 indica efeito da ponderação além do deslocamento. Se |Δ̄| < 1/24 (meio voto de 12), não roda: seria indistinguível do `glmbase`, e o resultado se registra assim.
- **Ambiente:** como no SPEC_27 §16C, o ambiente da VM (bibliotecas, GPU, notebook) não pode mudar entre as runs `personaw` e as do controle. Se mudar, o BASE também é retreinado.
- **Descritivo, sem teste novo:** a F5 do `personaw` ao lado dos resultados do `glmw` (doc 27) e do `glmw4` (doc 28).

## 8. Custo e tempo

- **Votos:** ≈ US$ 10 para os 5 505 pedidos, estimado pela entrada média dos extremos (1 845 tokens por pedido) no preço *batch* do Sonnet 5.5. Os tokens de imagem seguem o tamanho em pixels, então o valor real só sai na rodada. ~1 h, quase toda de upload (≈ 1 GB).
- **Ruído entre rodadas:** ≈ US$ 0,33 (180 pedidos, medido no piloto do doc 32), minutos.
- **GPU:**
  - 1 run `glmbase` de verificação + 3 runs `personaw`, 15 épocas cada;
  - mais 2 se o BASE tiver de ser retreinado (a seed 42 já terá rodado);
  - mais 3 do controle, se o critério for satisfeito.
- **Sem custo:** `votes`, `build`, `checks` (≈ 10 s com as permutações) e `drift` rodam no Mac.

## 9. Limitações

1. **A referência mistura duas rodadas, sem seed.** Os extremos foram votados em 2026-10-28 e os outros memes serão votados depois. Dos memes que entram no desvio (não unânimes), os com 1, 3 ou 5 YES humanos vêm da primeira rodada e os com 2 ou 4, da segunda. Uma mudança de comportamento entre rodadas coincidiria com a paridade da divisão humana. A repetição do piloto (decisão 5) mede o tamanho desse ruído, mas não o elimina.
2. **S1–S3 não são paráfrases.** Um meme 2-1 é discordância entre procedimentos (o S1 tem o desempate `[CALIBRATE: default NO …]`), não instabilidade de redação. O `p_Persona` mede acordo entre personas.
3. **Recusas e erros viram `?`** e tiram o meme do desvio. Eles provavelmente não são aleatórios em relação ao conteúdo.
4. **A validação (598 memes) é reutilizada pela terceira vez** numa F5 (docs 27, 28 e 33), sem correção para comparações múltiplas entre documentos.
5. **Só `think_off`** e **só um modelo** (Claude Sonnet 5.5), como no doc 32.

## 10. O que NÃO fazer

- Não votar na validação nem no teste.
- Não mudar S1–S3, o modelo, o braço ou as tags (são os do SPEC_32).
- Não usar `fallbacks` nem outro modelo para cobrir recusas.
- Não reescrever nenhum artefato do GLM (`glm_votes.csv`, `annotator_weights.csv`, `soft_labels_taskA_glmw*.json`, gates F3/G4) nem o JSONL dos extremos do doc 32.
- Não editar o notebook além de `ARM` e `SEED`.
- Não trocar o estado primário, o teste ou a leitura da (b) depois de ver os resultados.

## 11. Comandos (da raiz do repositório)

```bash
cp EXIST_MEMES/glm_reference/config.json EXIST_MEMES/glm_reference/config.json.bak-doc33   # cópia de segurança
# votos (≈ US$ 10; chave e cabeçalho exportados no terminal, nunca gravados no repositório)
python3 EXIST_MEMES/glm_annotator/claude_reference_32.py --model claude-sonnet-5-5 --arm think_off --scope train
# se aparecer "rode o mesmo comando de novo" (reenvio PNG), rodar o mesmo comando
python3 EXIST_MEMES/glm_annotator/claude_reference_32.py --model claude-sonnet-5-5 --arm think_off --scope repeat
python3 EXIST_MEMES/glm_annotator/persona_ref.py votes     # G2 e calibração
python3 EXIST_MEMES/glm_annotator/persona_ref.py drift     # ruído entre rodadas
python3 EXIST_MEMES/glm_annotator/persona_ref.py build     # F3 e controle congelado
python3 EXIST_MEMES/glm_annotator/persona_ref.py checks    # F4, faixa nula e G4
# F5 na VM: ARM = 'glmbase', SEED = 42 (verificação); depois ARM = 'personaw', SEED ∈ {42, 7, 2025}
python3 EXIST_MEMES/glm_annotator/analyze_f5.py --treat personaw
```

## 12. Resultado

Seções datadas, preenchidas depois de cada fase.

### Votos, G2, ruído, F3 e F4 — 2026-10-29

**Execução dos votos (F2 da Persona).**
- 5 505 pedidos em 26 lotes, 5 541 contando os 36 reenvios em PNG de 12 memes novos com "Could not process image". Nenhum `?` ficou no fim. A repetição do piloto (decisão 5) foi em 3 lotes de 180 pedidos.
- **Custo medido:** ≈ US$ 9,96 no treino (US$ 18,64 junto com os extremos do doc 32) e US$ 0,33 na repetição. A estimativa do §8 era ≈ US$ 10.
- **Desvio de infraestrutura** (registrado aqui; nada muda nos pedidos nem nas respostas):
  - a rede derrubou cerca de metade das criações de lote com `SSL: SSLV3_ALERT_BAD_RECORD_MAC`, depois do upload e antes da resposta;
  - acrescentados `--batch-mb` (a rodada foi feita com lotes de 150, depois 40 e 20 MB) e a nova tentativa automática: ao cair a conexão na criação, o script espera 30 s, confere na API e adota o lote ou o reenvia, até 5 vezes;
  - toda tentativa caída foi conferida na API, e o servidor não tinha criado o lote em nenhuma. A listagem dos lotes da conta confirmou que não ficou lote órfão, então nada foi cobrado em dobro;
  - testado em `tests/test_persona33_offline.py` (4e).

**G2 (§3): aprovado.**

| idioma | n | parse | confiante (3-0) | só nos 1 835 novos (descritivo) |
|---|---|---|---|---|
| en | 1 715 | 1,000 | 0,872 | 0,876 |
| es | 1 671 | 1,000 | 0,848 | 0,866 |

**Calibração fora dos extremos (§2, descritivo):** a calibração dos extremos se mantém fora deles. Nos 1 835 novos, a fração de YES das 3 respostas é 0,525, contra 0,555 dos humanos: distância de 0,030, na faixa ≤ 0,05.

| YES humanos (k/6) | 0 (n 316) | 2 (n 458) | 4 (n 585) | 6 (n 476) |
|---|---|---|---|---|
| humanos | 0,000 | 0,333 | 0,667 | 1,000 |
| Persona (3 respostas) | 0,055 | 0,261 | 0,662 | 0,923 |
| GLM off (3 respostas) | 0,400 | 0,737 | 0,902 | 0,964 |

No treino inteiro: Persona 0,525, GLM off 0,804 e humanos 0,555. Tabela completa em `reports/persona33_votes.md`.

**Ruído entre rodadas (decisão 5, descritivo):** repetindo os 180 pedidos do piloto do doc 32, 10 rótulos mudaram (5,6 %) e 1 de 60 votos por maioria. A fração de YES foi de 0,483 para 0,472. No 1Y5N nenhum rótulo mudou; as mudanças ficam no 3Y3N e no 5Y1N. `reports/persona33_drift.md`.

**F3 (§4):**
- 887 anotadores, todos com meme incluído.
- Deslocamento médio de p: +0,0054 (|Δp| médio 0,053). `share_moved` 0,274.
- `soft_labels_taskA_personaw.json` congelado com SHA-256 `a9445c9f8285d2128d01f0f418502327e87675dd18dffd548b83f82b0d3f55ae`.
- **Controle `personashift`:** Δ̄ = +0,0070, abaixo de 1/24. Pela regra do §7, ele **não roda**: seria indistinguível do `glmbase`. Isso vale qualquer que seja o resultado da F5.

**F4 (§5): G4 aprovado.** `share_moved` 0,274 (≥ 0,05) e ρ(r_persona, r_peer) 0,727 (< 0,8). Retreino autorizado.

**Checagem (b) com a faixa nula (k):**

| estatística | Persona | faixa nula da Persona | GLM (doc 27) | faixa nula do GLM |
|---|---|---|---|---|
| ρ(a) = ρ(r, r_peer) | 0,727 | [−0,053; 0,118] | 0,501 | [0,195; 0,290] |
| ρ(b) = ρ(D, taxa de YES) | −0,101 | [−0,128; 0,013] | −0,676 | [−0,709; −0,649] |
| ρ(gap) | 0,463 | [0,082; 0,227] | 0,710 | [0,646; 0,714] |
| `share_moved` | 0,274 | [0,140; 0,188] | 0,346 | [0,293; 0,344] |

**Leitura pré-registrada: parcial.**
- O ρ(a) fica muito acima da faixa nula: o peso carrega informação por meme.
- O ρ(b) cai para −0,10, dentro da faixa nula: o peso deixou de seguir a taxa de YES do anotador.
- O ρ(gap) = 0,463 fica acima de 0,40 e acima da faixa nula: o peso ainda segue a distância entre o limiar do anotador e o da Persona.

**Outras checagens (descritivo, ao lado do doc 27):**
- (d) As 526 mudanças de rótulo duro se dividem: TIE→NO 243, TIE→YES 275, NO→YES 4, YES→NO 4. No GLM, quase todas iam para o YES (NO→YES 36, TIE→NO 109, TIE→YES 409).
- (e) A Persona confiante contraria 27 de 737 memes unânimes (3,7 %); o GLM, 56 de 608 (9,2 %).
- (f) O voto da Persona concorda com o rótulo duro humano em 0,806 dos memes sem empate; o do GLM, em 0,707.
- (i) Os memes em que a Persona fica 2-1 têm mais divisões humanas fortes: 0,605 contra 0,438 nos 3-0, com IC 95 % [+0,119; +0,215]. No GLM era 0,522 contra 0,436.
- (j) ρ(r_persona, r_glm) = 0,397; r médio 0,554 (Persona) e 0,542 (GLM).

Relatórios: `reports/persona33_votes.md`, `persona33_drift.md`, `persona33_checks.md` e `persona33_gate.md`.

**Próximo:** F5 na VM (§7). Primeiro `glmbase` seed 42 (verificação do BASE), depois `personaw` nas seeds 42, 7 e 2025.

### F5 — verificação do BASE — 2026-10-29

- Run nova: `20261029_124907_glmbase_seed42` (15 épocas, `precheck` ok: `soft_labels_taskA_glmbase.json`, validação idêntica).
- **As 15 `eval_loss` são iguais, sem tolerância, às de `20261009_144803_glmbase_seed42`.** As 15 `loss` de treino também.
- Diferenças só em `eval_task_c_icm_hard_default`/`icmnorm_hard_default` (época 2) e `eval_task_c_icm_hard` (época 14), de no máximo 2,2e−16: arredondamento de ponto flutuante na métrica da Task C, não no modelo.
- **Decisão (§7): BASE reaproveitado.** As runs `glmbase` do doc 27 ficam em `metrics_history/`: seed 42 `20261009_144803`, seed 7 `20261009_164210`, seed 2025 `20261011_130127`. O `analyze_f5.discover` confirma: a run nova e `20261010_174208_glmbase_seed7` entram como repetições idênticas e ficam de fora; `20261010_165944_glmbase_seed7` (11 épocas) é ignorada.

**Próximo:** `ARM = 'personaw'` nas seeds 42, 7 e 2025.

### F5 — `personaw` vs BASE — 2026-10-29 — critério não satisfeito; controle não roda

Runs: `20261029_151046_personaw_seed42`, `20261029_161954_personaw_seed7`, `20261029_180306_personaw_seed2025` (15 épocas, `precheck` ok), contra o BASE do doc 27. Saída: `analyze_f5.py --treat personaw`. Os números foram recalculados à parte a partir dos JSON e batem.

**E = argmin(`eval_loss`), primário** (a 3ª época em todas as runs, nos dois braços):

| métrica | Δ 42 | Δ 7 | Δ 2025 | média | t | p |
|---|---|---|---|---|---|---|
| A ICM-Soft-Norm (primária) | +0,0082 | +0,0131 | +0,0140 | **+0,0118** | 6,49 | **0,023** |
| A ICM-Norm hard | +0,0032 | +0,0305 | −0,0215 | +0,0041 | 0,27 | 0,811 |
| `eval_loss` (↓ melhor) | +0,0106 | +0,0052 | +0,0106 | **+0,0088** | 4,88 | **0,039** |
| B ICM-Soft-Norm | +0,0012 | −0,0009 | −0,0006 | −0,0001 | −0,10 | 0,928 |
| C ICM-Soft-Norm | −0,0008 | −0,0092 | −0,0037 | −0,0046 | −1,87 | 0,202 |

**Estado final (secundário, só descritivo):** A ICM-Soft-Norm +0,0028 (p = 0,581); `eval_loss` +0,0448 (p = 0,073).

**Critério (§7): não satisfeito.** Duas das três condições passam: Δ > 0 e p < 0,05. A terceira falha: a `eval_loss` piora nas 3 seeds (p = 0,039). Pela tabela do §6 ("`eval_loss` pior"), a leitura é **efeito não detectado com 3 seeds; encerra**. O controle `personashift` não roda, por duas regras: o critério falhou, e |Δ̄| = 0,0070 < 1/24 (§12, F3).

**Descritivo (sem teste novo), Δ em E = argmin(`eval_loss`):**

| braço | Δ A ICM-Soft-Norm | p | Δ `eval_loss` | p |
|---|---|---|---|---|
| `glmw` (doc 27) | +0,0074 | 0,107 | +0,0093 | 0,105 |
| `glmw4` (doc 28) | −0,0247 | 0,209 | +0,0988 | 0,029 |
| `personaw` (doc 33) | +0,0118 | 0,023 | +0,0088 | 0,039 |

- O `personaw` dá o maior Δ da métrica primária dos três braços e o único com p < 0,05. A piora da `eval_loss` é do mesmo tamanho da do `glmw` e cerca de 1/11 da do `glmw4`.
- Nos três braços, a `eval_loss` em E = argmin sobe em todas as seeds, qualquer que seja o sinal do Δ da métrica primária.
- Isso não muda a leitura pré-registrada, e esta onda não autoriza nova variante (§6).

Relatórios: `reports/f5_report_personaw.md` e `f5_report_personaw.json`. Os relatórios F5 dos docs 27 (`f5_report.md`) e 28 (`f5_report_glmw4.md`) ficam intactos.
