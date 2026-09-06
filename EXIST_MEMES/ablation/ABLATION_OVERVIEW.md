# Overview da Suite de Ablação — Task 2 (Memes), EXIST 2026

Documento de trabalho: resume cada teste de ablação rodado (`ablation/runner.py`),
os resultados obtidos (3 seeds pareadas: 42, 7, 2025; 30 épocas cada; mesmo split
treino/val — `VAL_SPLIT_SEED=42` fixo em toda a suite), e quais diferenças são
sinal real vs. ruído de seed. Baseline usada em todos os testes: `warmup_steps=0.1`,
`task_a_weight=2.0`, `unfrozen_vision_layers=4`, `use_vision=True`,
`use_demographics=True`, `demo_feature_set="base"` (66-d), `demo_norm_mode="sum"`.

## Metodologia de comparação (revisado)

Como as 3 seeds são **pareadas** (mesma seed em ambos os lados de cada comparação,
mesmo split de validação), o teste mais forte disponível não é comparar médias ±
std não-pareado — é olhar o **delta por seed** e a **consistência de sinal** entre
as 3 seeds. Um efeito com sinal consistente (3/3 mesmo sinal) é evidência de
direção real mesmo se a magnitude for pequena; um efeito com sinais que trocam
(ex. +/-/+ ) é ruído, mesmo que a média pareça grande.

Cada tabela abaixo reporta: delta médio pareado, e se o sinal é consistente
(3/3) ou não (ex. `2/3`, `-/-/+ `). "YES" = consistente.

## ⚠️ Vieses de protocolo que afetam TODAS as tabelas abaixo

Estes vieses são idênticos em todas as configs comparadas (mesmo protocolo), então
comparações *relativas* entre configs continuam válidas — mas precisam ser
declaradas para qualquer leitura absoluta dos números:

1. **Seleção de época com vazamento.** `load_best_model_at_end=True` com
   `metric_for_best_model="task_a_icmsoftnorm_soft"` escolhe a melhor época no
   mesmo conjunto de validação sobre o qual as métricas finais são reportadas.
2. **Threshold da Task C ajustado por época, no conjunto de validação.**
   `_search_thresholds_task_c` (busca gulosa por coordenada, 12 valores × 6
   categorias = 72 chamadas PyEvALL) roda a CADA avaliação e o resultado
   sobrescreve `task_c_icmnorm_hard` — isto é ajuste de threshold durante o
   treino, no set de validação, o que o `CLAUDE.md` proíbe para o modelo final
   ("ajuste de threshold apenas pós-treino"). Para comparações de ablação isso é
   aceitável (mesmo viés em todas as configs), mas **para qualquer leitura
   absoluta do valor hard de Task C, preferir `task_c_icmnorm_hard_default`**
   (threshold fixo 0.5, sem tuning), já logado em todas as linhas.

Por isso, abaixo, sempre que a Task C hard for citada como sinal real, reporto
tanto `_hard` (tuned) quanto `_hard_default` (threshold 0.5) — quando os dois
concordam em sinal, a conclusão é robusta ao vazamento de threshold.

---

## ABL-1 — Schedule de warmup (`warmup_steps`: 0.0 / 0.1 / 0.2)

| comparação | métrica | delta médio | consistente? |
|---|---|---|---|
| 0.1 vs 0.0 | task_a soft | -0.0033 | não (-/-/+ ) |
| 0.1 vs 0.0 | task_a hard | -0.0122 | **YES (-)**, pequeno |
| 0.1 vs 0.0 | task_c hard | -0.0034 | não |
| 0.1 vs 0.0 | task_c hard_default | -0.0117 | não |
| 0.1 vs 0.2 | task_b soft | +0.0030 | **YES (+)**, pequeno |
| 0.1 vs 0.2 | demais | — | não |

**Resultado: efeito nulo a marginal.** Só 2 de 14 comparações mostram sinal
consistente, ambos com magnitude pequena (<0.012) e sem confirmação cruzada
entre métricas relacionadas (task_a hard consistente mas task_a soft não; task_b
soft consistente mas task_b hard não). Não há evidência robusta de que o warmup
schedule, nessa faixa (0–20%), importe.

---

## ABL-2 — Vetor demográfico expandido (`demo_feature_set`: base 66-d vs extended 71-d)

Extended adiciona `gender_annotators` (2 valores) e `age_annotators` (3 valores).

| métrica | delta médio (extended - base) | consistente? |
|---|---|---|
| task_a soft | +0.0016 | não (-/+/+) |
| task_b soft | **+0.0081** | **YES (+)** |
| task_c soft | -0.0122 | não (-/-/+) |
| task_a hard | -0.0089 | não |
| task_b hard | -0.0156 | não |
| task_c hard (tuned) | **-0.0322** | **YES (-)** |
| task_c hard_default (0.5) | **-0.0274** | **YES (-)** |

**Resultado: sinal real e robusto, com trade-off.** Task B soft sobe de forma
consistente (3/3) — adicionar gênero/idade ajuda a estimar a intenção da fonte.
Task C hard cai de forma consistente **tanto com threshold ajustado quanto com
threshold fixo 0.5** — o efeito não é artefato do tuning de threshold, é real.
Task A e Task C soft: sinal inconsistente, tratar como ruído.

**Nota de qualidade de dado:** o std não-pareado de `abl2_config_B_demo_extended`
em task_c soft é 0.0003 (valores por seed: 0.27237 / 0.27213 / 0.27271) — muito
mais apertado que a variação típica de ~0.01 nas outras configs. Isso é
consistente com a cabeça de Task C convergindo para uma saída quase-constante
sob o demo_vec extended (possível colapso, não apenas "métrica estável"). Não foi
possível confirmar examinando as probabilidades por categoria (o diretório
temporário de predições é apagado ao fim de cada run) — fica registrado como
hipótese a verificar numa run futura antes de reportar isso como achado positivo
no paper.

---

## ABL-3 — Peso da loss da Task A (`task_a_weight`: 2.0 / 1.0 / 3.0)

| comparação | métrica | delta médio | consistente? |
|---|---|---|---|
| 2.0 vs 3.0 | task_a soft | -0.0032 | **YES (-)**, pequeno |
| 2.0 vs 1.0 | task_a soft | +0.0037 | não (+/-/+) |
| demais (12 comparações) | — | — | não |

**Resultado: praticamente nulo.** Único sinal consistente é pequeno (-0.003) e
na direção "peso 3.0 piora ligeiramente task_a soft" — mas nem 1.0 vs 2.0 mostra
sinal, então não há uma relação monotônica clara. Reponderar a loss de Task A
entre 1×/2×/3× não move as métricas de forma útil. Negative result relevante:
a loss de Task A já é dominada por outro fator (a magnitude absoluta do total de
loss muda — de ~2.79 com peso 2.0 para ~2.17 com peso 1.0 — mas isso não se
traduz em métrica melhor).

---

## ABL-4 — Camadas do CLIP destravadas (`unfrozen_vision_layers`: 4 / 0 / 8)

| comparação | métrica | delta médio | consistente? |
|---|---|---|---|
| 4 vs 0 | task_a soft | -0.0081 | **YES (-)** |
| 4 vs 0 | task_c soft | -0.0335 | **YES (-)** |
| 4 vs 0 | task_c hard (tuned) | -0.0551 | **YES (-)** |
| 4 vs 0 | task_c hard_default | -0.0502 | **YES (-)** |
| 4 vs 8 | task_b soft | +0.0150 | **YES (+)** |
| 4 vs 8 | task_a hard | +0.0216 | **YES (+)** |

**Resultado: sinal real forte e robusto.** Congelar o CLIP inteiro (0 layers)
derruba task_a soft, task_c soft e task_c hard — **e o efeito em task_c hard é
idêntico em sinal com threshold tuned e com threshold fixo 0.5**, então não é
artefato do vazamento de threshold. Destravar mais camadas (8) melhora task_b
soft e task_a hard de forma consistente em relação à baseline (4 camadas).
**Conclusão robusta: o encoder de visão precisa de fine-tuning parcial; mais
camadas destravadas tende a ser ≥ baseline.**

---

## ABL-5 — Ablação de modalidades (visão / demografia) — ⚠️ reinterpretado

**Correção de rotulagem:** o texto (`XLMRobertaModel`, 24 camadas, sempre
totalmente treinável) está presente em **todas as 4 configs** — não existe um
braço "sem texto" nesta suite. O que foi de fato testado:

| nome no CSV | conteúdo real |
|---|---|
| `vision_demo` (baseline) | texto + visão + demografia |
| `vision_only` | texto + visão (sem demografia) |
| `demo_only` | texto + demografia (sem visão) |
| `text_only` | texto sozinho |

Ou seja: a suite testa "o que acontece ao remover visão e/ou demografia do
texto", não "visão vs. texto". A afirmação anterior ("visão domina o resultado")
não é sustentada por este desenho — o que os dados mostram é mais específico.

**Viés de implementação a declarar:** quando `use_vision=False`,
`vision_cls = torch.zeros_like(text_cls)` e o tensor concatenado (texto real +
512 zeros) passa por `fusion_norm` (`LayerNorm` sobre os 1024-d inteiros) antes
das 3 cabeças. Isso distorce a normalização do lado de texto (que é real) e
desperdiça metade da largura de entrada das cabeças em zeros constantes. As
quedas abaixo ao remover visão são, portanto, **remoção de informação + um
caminho de fusão degenerado somados** — a direção do efeito é confiável, a
magnitude está inflada. Ver seção de melhorias arquiteturais, item 1.

| comparação (delta = variante - full) | métrica | delta médio | consistente? |
|---|---|---|---|
| full vs vision_only (remove só demo) | task_a soft | +0.0067 | não (-/+/+) |
| full vs vision_only | **qualquer métrica** | — | **nenhuma consistente** |
| full vs demo_only (remove visão) | task_a soft | **-0.0242** | **YES (-)** |
| full vs demo_only | task_c soft | **-0.0486** | **YES (-)** |
| full vs demo_only | task_b hard | **-0.0392** | **YES (-)** |
| full vs demo_only | task_c hard (tuned) | **-0.0801** | **YES (-)** |
| full vs demo_only | task_c hard_default | **-0.0767** | **YES (-)** |
| full vs text_only (remove visão e demo) | task_a soft | **-0.0316** | **YES (-)** |
| full vs text_only | task_b soft | **-0.0112** | **YES (-)** |
| full vs text_only | task_c soft | **-0.0576** | **YES (-)** |
| full vs text_only | task_b hard | **-0.0461** | **YES (-)** |
| full vs text_only | task_c hard (tuned) | **-0.0824** | **YES (-)** |
| full vs text_only | task_c hard_default | **-0.0799** | **YES (-)** |

**Resultado (reinterpretado): remover demografia sozinha (full → vision_only)
não tem NENHUM sinal consistente em nenhuma métrica — demo não ajuda nem
atrapalha de forma mensurável quando a visão já está presente.** Remover visão
(full → demo_only, full → text_only), por outro lado, derruba quase todas as
métricas de forma robusta e consistente — inclusive task_c hard sob threshold
fixo, então não é artefato de vazamento de threshold. `demo_only` e `text_only`
têm magnitudes de queda muito parecidas entre si (ex. task_c hard_default:
-0.0767 vs -0.0799), reforçando que **a demografia sozinha não compensa a falta
de visão nem agrega sobre o texto puro nesse regime**.

**Reformulação do achado central:** não é "visão domina" — é **"adicionar visão
ao texto dá ganhos grandes e consistentes (especialmente em task_c); adicionar
demografia ao texto+visão não muda nada de forma mensurável"**. Essa formulação
é a que os dados sustentam.

### Contraste isolado da visão: `text_only` → `vision_only`

As comparações acima usam `full` como referência, então `full − text_only`
adiciona **duas** coisas ao mesmo tempo (visão e demografia). Para isolar só a
visão, o par correto é `text_only` vs `vision_only` — demografia desligada em
ambos, diferindo apenas em `use_vision`:

| métrica | texto só | texto + visão | Δ | Δ relativo | consistente |
|---|---|---|---|---|---|
| A soft | 0.4609 | 0.4992 | **+0.0384** | +8.3% | **3/3** |
| B soft | 0.3857 | 0.3962 | +0.0105 | +2.7% | **3/3** |
| C soft | 0.2270 | 0.2898 | **+0.0628** | **+27.7%** | **3/3** |
| A hard | 0.5073 | 0.5315 | +0.0243 | +4.8% | **3/3** |
| B hard | 0.3975 | 0.4209 | +0.0234 | +5.9% | **3/3** |
| C hard (tuned) | 0.3233 | 0.4160 | **+0.0928** | **+28.7%** | **3/3** |
| C hard (thr=0.5) | 0.3050 | 0.3899 | **+0.0849** | **+27.9%** | **3/3** |

**Resultado mais forte da suite inteira: 7/7 métricas com sinal consistente
nas 3 seeds, todas positivas.** Ganho concentrado em Task C (~+28% relativo,
soft e hard) — coerente com categorias como `OBJECTIFICATION` e
`SEXUAL-VIOLENCE` serem expressas visualmente, não no texto sobreposto. Menor
ganho em Task B (+2.7%) — intenção da fonte é distinção mais linguística. O
ganho em C hard sobrevive ao threshold fixo 0.5, então não é artefato do
tuning.

**Duas ressalvas:** (a) a magnitude é teto, não estimativa limpa — `text_only`
sofre também o confound de fusão com zeros (ver item 1 da seção de melhorias);
(b) isto autoriza dizer "o modelo extrai informação relevante da imagem", mas
**não** "a imagem carrega mais informação que o texto" — texto está presente
nos dois braços, então o que se mede é ganho incremental da visão, não ranking
entre modalidades.

---

## ABL-6 — Normalização do vetor demográfico (`demo_norm_mode`: sum vs L2)

| métrica | delta médio (L2 - sum) | consistente? |
|---|---|---|
| task_b soft | +0.0040 | **YES (+)**, pequeno |
| task_c hard (tuned) | -0.0121 | **YES (-)**, pequeno |
| demais | — | não |

**Resultado: efeito pequeno e sem direção consistente entre métricas** — L2
"ganha" em task_b soft e "perde" em task_c hard, ambos com magnitude pequena
(<0.012) e sem confirmação em `task_c_icmnorm_hard_default` (não consistente).
Não há um "L2 destrói o sinal" nesses dados.

### Reconciliação com a claim histórica do projeto

`CLAUDE.md` registra: *"normalização por soma dos vetores one-hot demográficos
preservou o sinal; normalização L2 destruiu-o"*. Rastreando a origem: o
`memes_model.ipynb` (célula markdown, seção 8) tem a frase quase idêntica —
*"testamos normalizar por norma L2 e o sinal demográfico foi destruído"* — escrita
**antes** da correção de `_normalize_apostrophes` (2026-08-01, mesma seção do
notebook). Ou seja, a claim original foi testada sobre um vetor demográfico
**com bug conhecido**: 64,6% dos votos de escolaridade eram descartados
silenciosamente por mismatch de apóstrofo (`'` vs `’`), deixando a fatia de
escolaridade quase sempre zerada. Um teste de normalização L2 sobre um vetor já
majoritariamente esparso/zerado tende a amplificar artificialmente poucas
posições não-zero — não é comparável ao teste atual, feito sobre o vetor
corrigido, com 3 seeds pareadas.

**Conclusão: a claim histórica não se aplica ao pipeline atual — foi medida
(aparentemente em execução única, não documentada com seeds) sobre uma versão
do vetor com bug de dados já corrigido nesta revisão.** Não reproduzida sob o
protocolo atual; tratar como não aplicável, não como contradição a resolver.

---

## ABL-INJ — Mecanismo de injeção da demografia (`demo_injection_mode`) × `demo_feature_set`

Executa a proposta nº 4 da seção seguinte. Três modos, cruzados com os dois
feature sets, todos com `use_vision=True` e `use_demographics=True`:
`current` (injeção somada ao CLS de texto depois do LayerNorm — a baseline),
`renorm` (re-aplica LayerNorm depois da injeção) e `head_inject` (concatena o
sinal demográfico ao vetor fundido depois do `fusion_norm`, elevando as cabeças
de 1024 para 1536 in_features).

**Âncora verificada:** o modo `current` nasce bit-a-bit idêntico à baseline nas
3 seeds (`fingerprint_init` igual em 42/7/2025), então os deltas abaixo medem só
o mecanismo de injeção.

| config | métricas com sinal consistente 3/3 (de 7) |
|---|---|
| `renorm` (base) | 1/7 |
| `head_inject` (base) | **6/7** |
| `renorm` (extended) | 1/7 |
| `head_inject` (extended) | **6/7** |

**Resultado 1 — a hipótese de escala está refutada.** O `renorm` corrige
exatamente a distorção de magnitude descrita na proposta nº 4 e **não move
nada** (1/7 nos dois feature sets, médias dentro do ruído). Se a inflação da
metade de texto antes do `fusion_norm` fosse o mecanismo que suprime a
demografia, o `renorm` teria funcionado. Não funcionou.

**Resultado 2 — o `head_inject` funciona, por outro mecanismo.** 6/7 métricas
sobem com sinal consistente nos dois feature sets (ganhos de +0.008 a +0.028).
O que ajuda não é consertar a escala, é dar às cabeças acesso direto ao sinal.
A única exceção é `task_b_icmnorm_hard` no base (−0.0012), que no extended
inverte para +0.0329 — inconsistente entre feature sets, não generalizar.

**Resultado 3 — o colapso da cabeça de Task C não se reproduziu.** A suspeita
registrada na nota de qualidade da ABL-2 foi instrumentada explicitamente
(`diagnose_task_c_collapse`, com JSON persistido por config × seed). Em **0 de
3 seeds** de **todas as 6 configs**, incluindo `current` + `extended` — onde o
colapso teria sido observado. A variância inter-item ficou em ~2,5e-02, cerca de
250× acima do limiar de colapso, e |média − taxa base do treino| em ~0,34, ~17×
acima. A cabeça discrimina claramente entre itens.

Duas leituras seguem vivas e esta suite não as separa: o colapso era artefato do
framework antigo (braços com pesos partilhados; o diretório de predições foi
apagado e não há como reconferir), ou é real mas não aparece sob o regime de
treino corrigido. **Tratar como observação não replicada, nunca como refutação.**
A ressalva de qualidade da ABL-2 deve ser lida à luz disto.

Detalhamento: `ablation/results/summary_INJ.md`.

---

## ABL-INJ-CTRL — Controle de capacidade do `head_inject`

O `head_inject` faz duas coisas ao mesmo tempo: dá às cabeças acesso ao sinal
demográfico **e** acrescenta 512 dimensões aprendidas a cada uma (1024 → 1536).
A ABL-INJ não separa as duas. Este braço isola: configuração **idêntica** ao
`inj_headinject_base` — mesma arquitetura, mesma capacidade, `fingerprint_init`
idêntico nas 3 seeds — com o `demo_vec` permutado entre itens (derangement de
permutação fixa, seed dedicada, aplicado antes do split). Preserva a marginal e
a capacidade utilizável; destrói só o vínculo item↔demografia.

Permutar, e não zerar: com `demo_vec=0`, `demo_proj(0)` é constante, o slot
concatenado vira viés fixo e as 512 dimensões deixam de ser utilizáveis — zerar
removeria sinal **e** capacidade, remedindo as duas juntas.

**Decomposição do ganho (mediana entre as métricas com ganho positivo):**

| origem do ganho do `head_inject` | fração |
|---|---|
| capacidade das 512 dims extras | **~86%** |
| vínculo item↔demografia | **~14%** (intervalo 0,7%–29,4%) |

O efeito demográfico concentra-se na Task C (20,7%–29,4%) e é quase nulo nas
Tasks A e B (0,7%–7,9%).

**Veredito pelo critério pré-registrado: parcial, não conclusivo — não adotar
o `head_inject` como evidência de que a demografia importa.** A degradação por
embaralhamento tem sinal consistente 3/3 em apenas 3 das 7 métricas
(`task_c_icmsoftnorm_soft`, `task_b_icmnorm_hard`,
`task_c_icmnorm_hard_default`); o critério exigia maioria.

Observação descritiva que **não** altera o veredito: a média do delta é negativa
em **7/7** métricas — embaralhar nunca ajudou em média. Compatível com um efeito
demográfico real porém pequeno demais para superar o ruído de seed na maioria
das métricas. Descrever assim, sem converter em evidência de efeito.

**Caracterização do `demo_vec` (medida ao desenhar o controle):** assume apenas
**151 valores distintos entre 3.984 itens** (mediana 27 itens por valor) — é, na
prática, um identificador categórico do grupo de anotadores. Esse grupo explica
**15,1% da variância de P(YES)** na Task A. Há sinal real no vetor; o que os
resultados dizem é que a arquitetura atual aproveita pouco dele.

Método completo, pré-registro e tabelas: `ablation/results/ABL-INJ-CTRL.md`.

---

## Síntese consolidada (revisada)

**Achados robustos (sinal consistente 3/3, inclusive sob threshold fixo quando
aplicável a Task C), por magnitude:**

1. **Adicionar visão melhora todas as métricas, em todas as seeds** (ABL-5,
   contraste isolado `text_only` → `vision_only`) — 7/7 métricas consistentes,
   ganho de ~+28% relativo em task_c, confirmado sob threshold fixo. É o
   resultado mais forte e mais limpo da suite. Simetricamente, remover visão a
   partir de `full` produz as maiores quedas medidas (até -0.08 em task_c hard).
2. **Congelar o CLIP inteiro prejudica task_a e task_c** (ABL-4, config B) —
   consistente e confirmado sob threshold fixo. Destravar mais camadas (8 vs 4)
   ajuda task_b soft e task_a hard.
3. **Demo_vec extended tem trade-off real** (ABL-2): ajuda task_b soft,
   prejudica task_c hard — confirmado sob threshold fixo. Ressalva: possível
   colapso da cabeça de Task C nessa config (ver nota de qualidade acima).
4. **Remover só a demografia (mantendo visão) não tem efeito mensurável em
   nenhuma métrica** (ABL-5, `vision_only` vs `full`) — nenhum sinal consistente
   em nenhuma comparação.
5. **Levar a demografia até as cabeças (`head_inject`) melhora 6/7 métricas nos
   dois feature sets** (ABL-INJ) — mas o controle pareado (ABL-INJ-CTRL) atribui
   só ~14% desse ganho à demografia; ~86% é capacidade das 512 dimensões extras.

**Negative results (sem sinal consistente ou sinal fraco/não confirmado):**
- Warmup schedule (ABL-1) — só 2/14 comparações consistentes, magnitude pequena.
- Peso da loss de Task A, 1×/2×/3× (ABL-3) — só 1/14, magnitude pequena, sem
  relação monotônica.
- Normalização sum vs L2 (ABL-6) — 2/14, direções conflitantes entre métricas,
  não reproduz a claim histórica (ver reconciliação acima).
- Re-normalizar depois da injeção de demografia (ABL-INJ, `renorm`) — 1/7 nos
  dois feature sets. **Refuta a hipótese de que a distorção de escala era a causa
  da demografia não contribuir.**
- Colapso da cabeça de Task C sob `demo_vec` extended (ABL-INJ, diagnóstico
  instrumentado) — 0/3 seeds em todas as 6 configs. Observação não replicada.

**Implicação prática (revisada):** a arquitetura atual precisa de visão fine-
tuned para funcionar bem — visão congelada ou ausente derruba o modelo. O
vetor demográfico, como injetado hoje (soma + gate + adição pós-LayerNorm ao
CLS de texto), não demonstra valor incremental mensurável sobre texto+visão,
exceto no caso específico de features expandidas (gênero/idade) que trocam
ganho em B por perda em C. Isso não significa que demografia seja inútil — pode
significar que **o mecanismo de injeção atual não deixa o sinal chegar às
cabeças de forma eficaz**.

**Atualização após ABL-INJ e ABL-INJ-CTRL:** essa hipótese foi testada e é
apenas parcialmente verdadeira. Levar o sinal até as cabeças de fato melhora as
métricas, mas o controle pareado mostra que a maior parte do ganho é capacidade,
não demografia. A leitura defensável na tese é: *o vetor demográfico carrega
sinal — o grupo de anotadores explica 15,1% da variância de P(YES) — mas a
arquitetura atual extrai dele uma fração minoritária (~14% do ganho, mediana),
concentrada na Task C.* Afirmar que "a correção da injeção fez a demografia
passar a ter efeito mensurável" **não** é sustentável.

---

## Melhorias arquiteturais propostas (para otimizar todas as métricas)

Ordenadas por payoff esperado por hora de GPU, todas compatíveis com o
framework de ablação existente (parametrizáveis via `AblationConfig` com custo
mínimo de implementação).

### 1. Corrigir a fusão para modalidades ausentes (pré-requisito para as demais)

**Problema:** em `memes_core.py`, quando `use_vision=False`:
```python
vision_cls = torch.zeros_like(text_cls)
fused = torch.cat([text_cls, vision_cls], dim=-1)   # [B, 1024], metade constante-zero
fused = self.fusion_norm(fused)                      # LayerNorm sobre os 1024-d inteiros
```
`fusion_norm` calcula média/variância sobre os 1024-d inteiros, então 512 zeros
constantes distorcem as estatísticas da metade viva (texto), e as 3 cabeças
gastam metade da largura de entrada em zeros. Isso não é uma ablação limpa — é
remoção de informação + degradação do caminho de fusão, ao mesmo tempo.

**Correção proposta:** montar `fused` só a partir dos ramos presentes (dimensão
variável por config), ou aplicar `LayerNorm` por-ramo antes da concatenação e
não normalizar globalmente sobre dimensões mortas. Pré-requisito para qualquer
ablação de modalidade futura ser interpretável sem ressalva.

**Custo:** baixo (mudança estrutural pequena em `forward` + cabeças com
`in_features` dinâmico, ou padding explícito documentado). **Risco:** médio —
muda a dimensão de entrada das cabeças, precisa retreinar baseline para
comparação justa.

### 2. Modality dropout no ramo de visão

**Motivação direta da ABL-5:** o sintoma observado — uma modalidade (visão)
"monopoliza" a representação fundida, com as demais (texto, demografia) não
mostrando ganho incremental mensurável — é o cenário clássico endereçado por
modality dropout.

**Proposta:** durante o treino, zerar `vision_cls` (após a correção do item 1)
com probabilidade `p ∈ {0.15, 0.30}`, forçando o caminho de texto a permanecer
preditivo de forma independente, sem depender de sempre ter visão disponível.

**Custo:** muito baixo — poucas linhas em `forward`, um novo campo em
`AblationConfig` (`vision_dropout_p`). **Risco:** baixo. **Payoff esperado:**
alto — ataca diretamente o problema relatado pelo usuário ("fazer o texto ser
mais influente").

### 3. Supervisão auxiliar só de texto (deep supervision)

**Proposta:** adicionar uma cabeça extra `Linear(512, 1)` sobre `text_cls`
(antes da fusão), com sua própria loss de Task A, ponderada com peso pequeno
(ex. 0.2–0.5) somada à loss principal. Força o encoder de texto a manter um
sinal preditivo por conta própria, independente da fusão.

**Custo:** baixo — uma `Linear` nova, um termo de loss extra, sem mudar
dimensões existentes. **Risco:** baixo. Complementar ao item 2, não competitivo.

### 4. Repensar a injeção de demografia — ✅ EXECUTADA (ABL-INJ + ABL-INJ-CTRL)

> **Status: concluída.** As duas alternativas propostas abaixo foram
> implementadas e medidas. `renorm` (re-normalizar depois da injeção) **não move
> nada** — a hipótese de escala está refutada. `head_inject` (re-injetar depois
> da fusão, nas cabeças) melhora 6/7 métricas, mas o controle de capacidade
> atribui ~86% desse ganho às 512 dimensões extras e só ~14% à demografia.
> Ver as seções ABL-INJ e ABL-INJ-CTRL acima. O texto original fica preservado
> abaixo como registro da motivação.

**Problema:** hoje,
```python
text_cls = text_cls + gate * self.demo_proj(demo_vec)   # depois do LayerNorm de text_proj
```
a injeção acontece **depois** do `LayerNorm` de `text_proj`, então ela infla a
magnitude da metade de texto antes do `fusion_norm` global — texto e visão
entram na fusão em escalas diferentes. A ABL-5 mostra que a injeção atual não
compra nada (`vision_only` ≈ `full` em todas as métricas) — não há o que perder
testando uma alternativa.

**Proposta:** re-normalizar (`LayerNorm`) depois da injeção de demografia, antes
da concatenação com visão, ou considerar re-injetar demografia depois da fusão
(nas cabeças) em vez de só no CLS de texto.

**Custo:** muito baixo (mudança de 1 linha + parametrização). **Risco:** baixo.

### 5. Não investir em capacidade de texto ou reponderação de loss

O XLM-R já está totalmente destravado (todas as 24 camadas treináveis, sem
`requires_grad=False` em lugar nenhum) — não há capacidade adicional óbvia a
liberar aí. A ABL-3 mostrou que reponderar a loss de Task A (1×/2×/3×) não move
nenhuma métrica de forma consistente, apesar de mudar a magnitude absoluta da
loss total (~2.17 a ~2.79). **A alavanca não é escalar (peso de loss), é
estrutural (itens 1–4 acima).** Não alocar runs futuras testando mais variações
de peso de loss ou tamanho de projeção de texto sem antes testar 1–4.

### Ordem de execução recomendada

(Item 4 já executado — ver ABL-INJ / ABL-INJ-CTRL.)

1 (pré-requisito) → 2 e 4 em paralelo (ambos baratos, endereçam o mesmo sintoma
por ângulos diferentes) → 3 (complementar) → nova rodada de ABL-5 com a fusão
corrigida para confirmar se a "não-diferença" de demografia era artefato do
item 1 ou é real mesmo com fusão limpa.
