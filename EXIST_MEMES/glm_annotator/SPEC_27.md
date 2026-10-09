# 27 — Ponderação de votos pela distância ao soft label do GLM: continuação exploratória do doc 25

**Escrito em 2026-10-09, ANTES de calcular qualquer peso.** Mesma regra dos docs 25 e 26: nada aqui muda depois do primeiro resultado. Mudanças entram como seção nova datada, com o motivo.

**Estatuto: continuação exploratória, aprovada pelo orientador em 2026-10-09 e declarada como motivada pela falha do portão G2 do doc 25.** Este documento não reabre o doc 25, cujo resultado (G2 falhou, `SPEC_25.md` §20) fica registrado como está. Qualquer ganho na F5 daqui deve ser apresentado como **evidência exploratória**, ao lado do resultado do doc 25 e nunca no lugar dele.

**Para o agente de código.** Este documento especifica só o que muda em relação ao `SPEC_25.md`. Tudo o que não está aqui segue o doc 25 sem alteração. Onde houver ambiguidade, pare e pergunte ao Jonathan.

---

## Resumo das mudanças em relação ao doc 25 (para atualizar o contexto)

| Item | Doc 25 | Doc 27 |
|---|---|---|
| Referência do GLM por meme | voto duro: maioria das 3 paráfrases | **soft label**: `p_GLM` = fração de YES entre as 3 paráfrases (0, 1/3, 2/3 ou 1) |
| Memes que entram no desvio | não unânimes **e** GLM confiante (3-0) | não unânimes **e** as 3 paráfrases parseadas (3-0 **e** 2-1) |
| Desvio do anotador no meme | `d = 1` se voto ≠ voto do GLM, senão 0 | **`d = |1[voto = YES] − p_GLM|`** ∈ {0, 1/3, 2/3, 1} |
| Portão G2 | parse ≥ 0,98 **e** confiantes ≥ 0,70, por idioma | **só parse ≥ 0,98 por idioma** (já atendido: en 0,9983 · es 0,9988) |
| Linha de base dos pares, checagem (a) | `d` contra a maioria dos outros 5 | `d = |1[voto = YES] − p_outros|`, com `p_outros` = fração de YES dos outros 5 (mesma lógica soft) |
| Chamadas ao GLM | 1 440 + 480 + 9 438 | **0 novas**: reaproveita tudo o que o doc 25 gravou |
| Peso da punição `w`, `r = 1 − D`, `p_new`, G4, F5 | — | **inalterados** |

---

## 0. De onde isto vem

- **Doc 25, F2 (2026-10-08):** com a regra "confiante = as 3 paráfrases concordam", o GLM ficou confiante em 72,7% dos memes de treino em inglês e em **69,1% em espanhol** (1 155/1 671, mínimo 1 170). G2 falhou por 15 memes, e o experimento foi encerrado (`SPEC_25.md` §20).
- **Diagnóstico (só dados do GLM, sem voto humano):** nos 980 memes não confiantes, a P3 foi a paráfrase dissidente em 734 (75%). Excluir esses memes do desvio tem dois custos:
  1. o corte é binário e arbitrário: 69,1% e 70% dão praticamente a mesma precisão (~13,8 contra ~14,0 memes por anotador);
  2. `[HIPÓTESE]` os memes excluídos tendem a ser os ambíguos, justamente onde o desvio de cada anotador seria mais informativo. Excluí-los mede os pesos sobretudo nos memes fáceis.
- **Proposta discutida em 2026-10-09:** usar o grau de concordância entre as 3 paráfrases em vez de descartar os memes 2-1. A variante com entropia foi descartada: com 3 paráfrases, `1 − H` só vale 1 (3-0) ou 0,082 (2-1), o que quase reproduz a exclusão (+3% de massa) e não informa a **direção** da hesitação do GLM. A variante escolhida é a **distância ao soft label**.
- **Aprovação:** orientador, 2026-10-09.

## 1. Ponto de partida (o que já existe e é reaproveitado)

| Artefato | Estado |
|---|---|
| `glm_reference/config.json` | parâmetros do doc 25; portões G0.1–G1 aprovados; G2 reprovado; vLLM `0.29.1rc1.dev452+g3df4ae153` |
| `glm_reference/raw/think_off.jsonl` | 10 638 chamadas (piloto + estabilidade + F2), braço `think_off`, seed 42 |
| `glm_reference/glm_votes.csv` | `label_p1..p3` dos 3 386 memes de treino |
| `glm_reference/train_ids.json` · `val_ids.json` | split 3 386 / 598 conferido |
| Código | `weights.py`, `checks.py`, `analyze_f5.py`, `tests/test_weights.py`: adaptar (§13) |

**Distribuição da referência na F2** `[MEDIDO — glm_votes.csv; só GLM, nenhum voto humano lido]`:

| escopo | 3-0 YES | 3-0 NO | 2-1 YES | 2-1 NO | sem resposta |
|---|---|---|---|---|---|
| total (3 386) | 0,634 | 0,075 | 0,218 | 0,071 | 0,001 (5 memes) |
| en (1 715) | 0,633 | 0,094 | 0,204 | 0,068 | 0,002 (3) |
| es (1 671) | 0,636 | 0,056 | 0,233 | 0,075 | 0,001 (2) |

Os 5 memes sem resposta são os rejeitados pelo servidor (imagem acima do cache de encoder de 6 084 tokens, `SPEC_25.md` §20). Ficam fora do desvio, mas recebem o soft label novo como todos os outros.

## 2. Visão geral das fases

| Fase | O que faz | Chamadas | Usa votos humanos? | Portão |
|---|---|---|---|---|
| F0–F2 | **não rodam de novo**: reaproveitadas do doc 25 | 0 | Não | G2′ (parse ≥ 0,98 por idioma), já atendido |
| F3′ | pesos pela distância ao soft label; novos soft labels | 0 | Sim | testes do Apêndice A deste documento |
| F4 | checagens (a)–(g) adaptadas; portão de retreino | 0 | Sim | G4, inalterado |
| F5 | retreino BASE vs GLMW (o Jonathan dispara) | 0 | Sim (avaliação) | leitura §11 |

**Declaração sobre o G2′:** ele é avaliado sobre dados já conhecidos (`reference_summary.md`). Não é um teste: é a condição mínima de que a referência existe para quase todos os memes. A substituição do critério de 70% foi decidida porque o desenho novo não exclui mais os memes 2-1, e um limite sobre a fração 3-0 deixa de ter a função de garantir memes por anotador (§0).

## 3. Parâmetros fixados

Os da §3 do doc 25 continuam valendo para tudo o que foi gerado (amostragem, prompts, concorrência). Novos ou alterados:

| Parâmetro | Valor | Por quê |
|---|---|---|
| Soft label do GLM | `p_GLM,i = (nº de YES entre P1, P2, P3) / 3` | usa a hesitação e a direção do GLM |
| Inclusão no desvio | meme não unânime entre os humanos **e** P1, P2 e P3 parseadas | exclui só o que não tem referência (5 memes) |
| Desvio por meme | `d_ai = |1[voto_a = YES] − p_GLM,i|` | nos memes 3-0 é idêntico ao doc 25; nos 2-1 pune na direção da maioria das paráfrases, proporcionalmente |
| Peso da punição | `w_ai = 1 − H(fração de YES entre os outros votos)` | inalterado (leave-one-out) |
| Desvio do anotador | `D_a = Σ w·d / Σ w` | inalterado |
| Peso do voto | `r_a = 1 − D_a`; sem meme incluído (ou Σw = 0) → `r_a = 1` | inalterado |
| Portão G2′ | parse ≥ 0,98 por idioma | ver §2 |
| Portão G4 | `share_moved ≥ 0,05` e ρ(r_glm, r_peer) < 0,8 | inalterado |

## 4. Prompts

Inalterados: P1, P2 e P3 do `SPEC_25.md` §4 (SHA-256 em `config.json`). O doc 26 (fatorial 2×2 da estrutura do prompt) é independente: **não** altera a referência usada aqui. Serve só para interpretar a sensibilidade à P3.

## 5–7. F0, F1, F2

Não são executadas de novo. Resultados no `SPEC_25.md` §20 e em `NOTAS_PARA_TESE.md` §2–§4.

## 8. F3′ — Pesos e novos soft labels (só cálculo)

Código em `glm_annotator/weights.py`, funções puras, sem I/O dentro delas. A implementação de referência e os testes estão no Apêndice A; o código final tem de passar neles.

**Entradas:** memes do split de treino; `labels_task2_1` alinhados a `annotators`; `glm_votes.csv` (`label_p1..p3`). Votos humanos fora de {YES, NO} são contados, reportados e removidos daquele meme.

**Memes no cálculo do desvio:**
- **fora:** memes unânimes entre os humanos (motivo do doc 25 §8 inalterado);
- **fora:** memes com alguma paráfrase sem rótulo (`?`);
- **dentro:** todos os outros, empates 3-3 humanos incluídos, GLM 3-0 **e** 2-1.

**Novos soft labels:** como no doc 25, para **todo** meme de treino: `p_new = Σ r_a·[voto_a = YES] / Σ r_a`, gravado como `[p_new, 1 − p_new]` na ordem `['YES', 'NO']`.

**Saídas** (os nomes do doc 25 se mantêm, porque a F3 do doc 25 nunca rodou):
- `annotator_weights.csv`: as colunas do doc 25 mais `mass_included` (= Σ w do anotador);
- `soft_labels_weighted.json`;
- `reliability/outputs/soft_labels_taskA_glmw.json` (treino = `p_new`; validação = `raw` sem alteração);
- `reliability/outputs/soft_labels_taskA_glmbase.json` (cópia byte a byte do `raw`);
- os SHA-256 dos dois arquivos planos vão para `config.json`, junto com `"method": "doc27_soft_distance"`.

## 9. F4 — Checagens e portão de retreino

Como no doc 25 §9, com estas adaptações:

| Checagem | Mudança |
|---|---|
| (a) pares | `D_peer` com `d = |1[voto = YES] − p_outros|`, nos mesmos memes incluídos. ρ(r_glm, r_peer) entra no G4 |
| (b) limiar | inalterada; a "taxa de YES do GLM nos memes incluídos" passa a ser a média de `p_GLM` nesses memes |
| (c) demografia | inalterada |
| (d) portão | inalterada (`share_moved` com meio voto = 1/12; mudanças de rótulo duro) |
| (e) GLM contra unanimidade | "GLM confiante" = 3-0, como no doc 25. Descritiva |
| (f) concordância | `glm_vote` = maioria das 3 paráfrases. Descritiva |
| (g) precisão | acrescentar a distribuição de `mass_included` (Σ w por anotador) ao lado de `n_included`; largura do IC 95% de `D_glm` por bootstrap dos memes |
| **(h) nova, descritiva** | fração da massa `Σ w` que vem de memes 2-1, por idioma: mostra quanto do peso depende dos memes em que o prompt importa |

**Portão G4 (inalterado):** retreino só se `share_moved ≥ 0,05` **e** ρ(r_glm, r_peer) < 0,8. A decisão vai para `reports/gate_decision.md`.

## 10. F5 — Retreino (Jonathan dispara na VM)

Inalterada em relação ao doc 25 §10 e ao adendo §18:
- braços `glmbase` (cópia do `raw`) e `glmw` (Task 2.1 de treino ponderada pela F3′);
- `SEED ∈ {42, 7, 2025}`, 15 épocas, estado em E = argmin(`eval_loss`) e estado final;
- validação sempre com os labels originais;
- t pareado com df = 2;
- **BASE retreinado** (decisão de 2026-10-06);
- portão 12.6 via `analyze_f5.py --precheck` e a célula de verificação no notebook.

## 11. Leitura pré-registrada

| Desfecho | Leitura |
|---|---|
| G4 falha por `share_moved < 0,05` | a ponderação por distância não move os labels; nulo por construção, sem gastar GPU; encerra |
| G4 falha por ρ ≥ 0,8 | o GLM reproduz o consenso dos pares (família já testada com MACE); encerra |
| F5: média de Δ ICM-Soft-Norm > 0, p < 0,05 e `eval_loss` sem piora | **evidência exploratória** a favor de uma referência externa graduada; não confirmatória (estatuto deste documento) |
| F5: Δ nulo ou negativo | encerra; soma-se ao doc 25 e à linha de reliability |

Em qualquer desfecho, a tese apresenta o doc 25 (G2 falhou) e o doc 27 juntos, com o estatuto de cada um.

**Nenhum desfecho autoriza** trocar de variante (entropia, outro mapeamento D→r), mudar prompts ou limiares e calcular de novo mirando um resultado melhor.

## 12. Portões de execução (falha dura)

Os do doc 25 §12, com estas mudanças:
1. `assert_config()` continua valendo: os parâmetros de geração não mudam.
2. A F3′ exige no `config.json` o G2′ (parse ≥ 0,98 por idioma, recalculado do `glm_votes.csv`) e grava `"method": "doc27_soft_distance"`.
3. Os testes do Apêndice A **deste** documento passam antes da F3′, e os do doc 25 continuam passando como caso particular (propriedade 7).
4. Depois da F3′: soft labels em [0, 1]; memes unânimes com `p_new == p_old`; validação idêntica ao `raw` (hash).
5. Antes da F5: igual ao doc 25 §12.6.

## 13. Estrutura de arquivos (o que muda)

```
EXIST_MEMES/glm_annotator/
  SPEC_27.md              # este documento
  weights.py              # + glm_soft(), d por distância, peer soft, mass_included
  checks.py               # (a) peer soft, (b) média de p_GLM, (g) massa, (h) nova
  tests/test_weights.py   # + Apêndice A do doc 27; doc 25 mantido como caso particular
EXIST_MEMES/glm_reference/
  (mesmos arquivos de saída do doc 25 §13; a F3 do doc 25 nunca os gerou)
```

`client.py`, `run_*.py`, `prompts.py` e `analyze_f5.py` não mudam.

## 14. Custo e tempo

| Fase | Chamadas | Tempo |
|---|---|---|
| F3′ + F4 | 0 | minutos |
| F5 | 0 | 6 runs × 15 épocas na VM (o mesmo que o experimento MACE) |
| Código | — | ~1–2 h de adaptação + testes |

## 15. O que NÃO fazer

- Não refazer chamadas ao GLM nem trocar prompts.
- Não usar só P1/P2, nem tirar a P3: as três entram em `p_GLM`.
- Não trocar de variante (entropia) depois de ver os pesos.
- Não mudar os labels de validação nem os das Tasks 2.2/2.3.
- Não rodar a F5 sem o G4 autorizar.
- Não apresentar o resultado da F5 como confirmatório.

## 16. Decisões confirmadas

- 2026-10-09, Jonathan + orientador: seguir como continuação exploratória; variante **distância ao soft label**; G2′ = parse ≥ 0,98 por idioma; o resto como no doc 25.
- Herdadas do doc 25 (§16 e §18): temperatura e amostragem, limiares do G4, `r = 1 − D`, seeds/épocas/teste da F5, BASE retreinado, concorrência 4 (§19).

## 16A. Adendo de 2026-10-09 — pontos de atenção (antes da F3′)

Seção nova datada; as seções acima não foram editadas. Nada aqui muda a fórmula, os prompts ou os portões. Tudo isto foi para o documento e para o código **antes** de calcular qualquer peso.

### P1 — Escala do desvio nos memes 2-1 (documentação + descritivo)

- **Fato:** com `d = |1[voto = YES] − p_GLM|`, num meme 2-1 a diferença entre quem vota YES e quem vota NO é 1/3 (contra 1 nos memes 3-0), e todo anotador leva pelo menos 1/3 nesses memes. Isso cria um piso em `D_a`.
- **Consequência:** os `D_a` do doc 27 **não estão na mesma escala** dos do doc 25 e não são comparáveis. O relatório da F4 diz isso explicitamente.
- **Ação (checagem g):** além do `D_glm` total, reportar por anotador duas parcelas, ambas com a mesma fórmula e o mesmo `w`: `D_glm` só com os memes GLM 3-0 e `D_glm` só com os memes GLM 2-1. Descritivo.
- **Proibido:** alterar a fórmula ou o mapeamento `r = 1 − D` por causa disso (§11, §15).

### P2 — A hesitação do GLM pode ser efeito da P3, não ambiguidade do meme (checagem (i), descritiva)

- **Fato:** a P3 é a paráfrase discordante em 75% dos memes 2-1 (§0). A hipótese da §0 ("os memes excluídos no doc 25 são os ambíguos") está marcada como `[HIPÓTESE]` e não foi testada.
- **Checagem (i), nos memes de treino, em `checks.py`:**
  1. tabela cruzada, por idioma: padrão do GLM (3-0 / 2-1) × divisão dos votos humanos (6-0, 5-1, 4-2, 3-3), com contagens e proporções por linha;
  2. nos memes GLM 2-1, a mesma tabela separada pela paráfrase que discordou (P1, P2 ou P3);
  3. gravada em `reports/checks_summary.md`, seção "(i) hesitação do GLM × divisão humana".
- **Leitura:** se os memes 2-1 do GLM não tiverem mais divisões humanas 4-2/3-3 do que os 3-0, a `[HIPÓTESE]` da §0 não se sustenta. Registra-se, sem mudar nada no cálculo.
- A checagem (i) não decide nada e não entra no G4.

### P3 — Estado primário do retreino (F5): fixado pelo Jonathan em 2026-10-09

- **O que estava em aberto:** a §10 manda reportar dois estados de cada run de 15 épocas (a época de menor `eval_loss` e o estado ao fim da época 15), mas a §11 não dizia em qual deles se aplica o critério de sucesso. Dois estados seriam duas chances de passar. O `SPEC_25.md` §18 também não fixava.
- **Decisão:** o **estado primário é E = argmin(`eval_loss`)**, a época de menor loss na validação, escolhida pela loss e nunca pela métrica. É o mesmo estado oficial do experimento MACE (`reliability/analyze_arm_experiment.py`: "Época reportada: E = argmin(eval_loss)"). O critério da §11 aplica-se **só** em E. O estado final continua reportado, apenas como **secundário** (regra de simetria).

### P4 — Poder e lateralidade do teste da F5: fixado pelo Jonathan em 2026-10-09

- **Fato:** com 3 seeds, o t pareado tem df = 2. Para p < 0,05 é preciso |t| > 4,30 (bilateral) ou t > 2,92 (unilateral).
- **Decisão:** teste **bilateral**, igual ao MACE e ao `analyze_f5.py` (`p_two_sided`).
- O relatório da F5 reporta sempre o Δ de cada seed, a média de Δ e o t, não só o p.
- Um resultado nulo é registrado como **"efeito não detectado com 3 seeds"**, nunca como "efeito zero".

## 16B. Nota de 2026-10-09 — leitura da checagem (i) (antes de qualquer cálculo real)

Testado com votos do GLM **sintéticos e aleatórios**, a comparação crua "proporção de 4-2/3-3 nos memes 2-1 > nos 3-0" deu "mais" por puro ruído (0,479 contra 0,455). Para que a leitura do P2 não confunda ruído com efeito, a checagem (i) reporta a **diferença com IC 95%** (aproximação normal para duas proporções) e só diz "mais" quando o intervalo exclui zero. Se o IC incluir zero, a leitura é "não diferem", e a `[HIPÓTESE]` da §0 não se sustenta. Continua descritiva e fora do G4.

## 17. Resultado

*(preencher depois de rodar, como seção datada; não editar as seções acima)*

---

## Apêndice A — Implementação de referência e testes

```python
from math import log2

def binary_entropy(p: float) -> float:
    return 0.0 if p in (0.0, 1.0) else -(p * log2(p) + (1 - p) * log2(1 - p))

def glm_soft(labels: list):
    """Fração de YES nas 3 paráfrases; None se alguma não tiver rótulo."""
    return None if any(l not in ('YES', 'NO') for l in labels) else sum(l == 'YES' for l in labels) / len(labels)

def annotator_deviation(memes: dict, p_glm: dict, peer: bool = False) -> dict:
    """memes: {meme_id: {annotator_id: 'YES'|'NO'}}
    p_glm: {meme_id: float in [0,1] | None}
    peer=True: referência = fração de YES dos outros votos do meme (checagem a)."""
    num, den = {}, {}
    for mid, votes in memes.items():
        anns = list(votes)
        n_yes = sum(v == 'YES' for v in votes.values())
        if n_yes in (0, len(anns)):              # unânime: fora do desvio
            continue
        p = p_glm.get(mid)
        if p is None:                            # sem referência: fora
            continue
        for a in anns:
            others = [votes[b] for b in anns if b != a]
            k = sum(o == 'YES' for o in others) / len(others)
            w = 1 - binary_entropy(k)
            ref = k if peer else p
            d = abs((votes[a] == 'YES') - ref)
            num[a] = num.get(a, 0.0) + w * d
            den[a] = den.get(a, 0.0) + w
    all_anns = {a for v in memes.values() for a in v}
    return {a: (num[a] / den[a] if den.get(a, 0) > 0 else None) for a in all_anns}

def vote_weights(D: dict) -> dict:
    return {a: (1.0 if d is None else 1.0 - d) for a, d in D.items()}

def weighted_soft_label(votes: dict, r: dict) -> float:
    return sum(r[a] * (v == 'YES') for a, v in votes.items()) / sum(r[a] for a in votes)
```

**Caso de teste:** o mesmo painel do doc 25 (A–F, m1–m5), agora com as 3 paráfrases do GLM:

| Meme | Votam YES (humanos) | Divisão | GLM (P1, P2, P3) | `p_GLM` |
|---|---|---|---|---|
| m1 | A B C D E F | 6-0 | NO, NO, NO | 0 (unânime humano → fora) |
| m2 | A B C D E | 5-1 | YES, YES, YES | 1 |
| m3 | A B C D | 4-2 | NO, NO, YES | 1/3 |
| m4 | A B C | 3-3 | YES, YES, NO | 2/3 |
| m5 | B C D E F | 5-1 | NO, NO, NO | 0 |

Valores esperados (tolerância 1e-4):

| Anotador | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| D_glm | 0,0217 | 0,5000 | 0,5000 | 0,5158 | 0,4519 | 0,8769 |
| r_glm | 0,9783 | 0,5000 | 0,5000 | 0,4842 | 0,5481 | 0,1231 |
| D_peer | 0,8118 | 0,2284 | 0,2284 | 0,2284 | 0,4067 | 0,8173 |

Conta de F, para conferir: m2 (w = 1, d = |0 − 1| = 1); m3 (w = 0,2781, d = |0 − 1/3| = 0,3333); m4 (w = 0,0290, d = |0 − 2/3| = 0,6667); m5 (w = 0,2781, d = |1 − 0| = 1). D = (1 + 0,0927 + 0,0193 + 0,2781) / (1 + 0,2781 + 0,0290 + 0,2781) = 1,3901 / 1,5852 = 0,8769.

Conta de `D_peer` de F: m2 (outros 5/5 YES → p = 1, d = 1, w = 1); m3 (4/5 → d = 0,8, w = 0,2781); m4 (3/5 → d = 0,6, w = 0,0290); m5 (4/5 → F vota YES, d = 0,2, w = 0,2781). D = 1,2955 / 1,5852 = 0,8173.

| Meme | p_old | p_new |
|---|---|---|
| m1 | 1,0000 | 1,0000 |
| m2 | 0,8333 | 0,9607 |
| m3 | 0,6667 | 0,7858 |
| m4 | 0,5000 | 0,6313 |
| m5 | 0,8333 | 0,6878 |

**Testes de propriedade:**
1. `binary_entropy` como no doc 25.
2. Mudar `p_GLM` de m1 (unânime) não muda nenhum `D_a`.
3. Se todo `p_GLM` for None, todo `r_a = 1` e todo `p_new == p_old`.
4. Num meme unânime, `p_new == p_old` para quaisquer pesos.
5. Se todos os `r_a` de um painel forem iguais, nenhum soft label do painel muda.
6. m4 (3-3) vira YES depois da ponderação (0,6313), uma mudança de rótulo duro na checagem (d).
7. **Caso particular do doc 25:** se todo `p_GLM` ∈ {0, 1}, os `D_a` são exatamente os do Apêndice A do doc 25 (0,0217 · 0,5000 · 0,5000 · 0,5473 · 0,3558 · 0,8246, com m3 → 0 e m4 → 1).
8. `d ∈ [0, 1]`; com `p_GLM = 0,5`, qualquer voto tem `d = 0,5`.
