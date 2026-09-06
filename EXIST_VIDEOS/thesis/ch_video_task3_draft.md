# Capítulo N — Deteção de sexismo em vídeo (EXIST 2026, Task 3): auditoria, investigação causal e arco experimental

> **Estado deste ficheiro:** rascunho de tese, escrito em 2026-08-16 a partir dos documentos
> de rastreabilidade `improvements/09,16,17,18,20,21,22` e do `CLAUDE.md` do projeto.
> Todo número aqui tem fonte nesses documentos. Onde um valor seria necessário e não existe
> na documentação, aparece um marcador `[NÚMERO A CONFIRMAR: …]` em vez de uma estimativa.
>
> **Convenção de marcação herdada dos documentos de origem e mantida deliberadamente:**
> `[MEDIDO]` — produzido por execução, com artefato; `[VERIFICADO]` — conferido no código ou
> por aritmética; `[HIPÓTESE]` — interpretação, **não escrever como facto**. A conversão para
> LaTeX deve preservar esta distinção, não achatá-la.

---

## N.0 O que este capítulo relata

O sistema de vídeo da Task 3 foi construído, submetido ao EXIST 2026, e depois submetido a
uma auditoria de código e a um programa de investigação experimental que ocupou a maior
parte do trabalho de tese. Este capítulo relata esse programa.

A tese que o capítulo defende não é "o sistema melhorou". É mais específica e mais
defensável: **o ramo de vídeo deste sistema tem um teto de informação medido, esse teto foi
localizado por sondas causais sem gastar treino, e quatro intervenções desenhadas para o
ultrapassar falharam de formas que se confirmam mutuamente.** O contributo é metodológico e
negativo — e o material negativo aqui é mais sólido do que o ganho agregado que se procurava.

### N.0.1 Distinção obrigatória: sistema submetido vs. sistema corrigido

Estes são **dois objetos diferentes** e não devem partilhar tabela sem legenda explícita:

- **Sistema submetido** — o que produziu as submissões do EXIST 2026, em
  `predictions/cross_att/` e `predictions/swa/`. Contém `cross_attn`, SWA, `max_length=128`
  e a tabela de ablação com os problemas descritos em §N.2. **Estas submissões não podem ser
  regeneradas**: o sistema que as produziu deixou de existir no repositório.
- **Sistema corrigido** — o que resultou da auditoria (§N.2), sem `cross_attn`, sem SWA, com
  `max_length=256`, sobre features de áudio v2. É este que sustenta todos os números
  experimentais deste capítulo, com a exceção explícita do baseline congelado (§N.1.3), que
  é uma re-execução controlada do sistema submetido.

Escrever isto na tese não é uma formalidade defensiva: as duas leituras dão números
diferentes e a comparação entre elas foi a origem de várias armadilhas descritas adiante.

---

## N.1 O sistema de vídeo: arquitetura e baseline congelado

### N.1.1 Modalidades e fusão

O sistema recebe quatro modalidades por amostra e funde-as por concatenação de vetores,
seguida de uma projeção:

| slot | encoder | dimensão |
|---|---|---:|
| vídeo | VideoMAE-base, LoRA `r=16`, `α=32` | 768 |
| áudio | Whisper-large-v3, pré-extraído offline, projetado | 768 |
| texto | XLM-RoBERTa-large, LoRA `r=16` | 1024 |
| bio | Bio-MLP (eye-tracking + heart rate, `RobustScaler`) | 128 |

A largura da fusão é **função da condição de ablação**: `2688 → 1024` nas condições com
biosinais (B1/B2) e `2560 → 1024` na condição sem biosinais (B0) `[VERIFICADO — SIMP-16]`.

> ⚠ **Correção a propagar para o texto do artigo CEUR-WS e para o diagrama TikZ.** O artigo
> submetido descreve a fusão da Task 3 como `2592`, com `bio=128` e os rótulos de texto e
> áudio trocados. O valor `2592` **não corresponde a nenhuma configuração do código**. A
> aritmética correta é `768 (vídeo) + 768 (áudio) + 1024 (texto) + 128 (bio) = 2688`.

A fusão é `Linear → GELU → Dropout`, **sem LayerNorm depois da concatenação**. Isto é uma
restrição de desenho, não um esquecimento: enquanto qualquer braço de ablação zerar um slot,
um LayerNorm sobre o vetor concatenado redistribuiria a massa dos slots ativos e
reintroduziria o confound de modalidade ausente. Com a fusão linear, um slot zerado
contribui exatamente 0 e recebe gradiente 0 `[VERIFICADO — video_core.py:1041-1045]`.

### N.1.2 A `cross_attn` removida — e por que não era atenção

O sistema submetido incluía um módulo de cross-attention entre modalidades. A auditoria
`[VERIFICADO]` mostrou que ele atendia sobre uma **única** key — o vetor de vídeo já
colapsado pelo pooling — pelo que o softmax operava sobre um elemento e devolvia
identicamente 1,0. O módulo não era atenção: era um mapa linear fixo, e custava
**4 987 904 parâmetros, 45 % dos 11,0 M treináveis** do sistema.

Foi removido no sistema corrigido (ARCH-03b). O diagrama de arquitetura do artigo mostra-o e
precisa de ser corrigido junto com a largura de fusão.

### N.1.3 Baseline congelado (`09_BASELINE_RESULTS.md`)

Para que "o sistema melhorou" pudesse significar alguma coisa, o sistema submetido foi
re-treinado sobre o split congelado (`val_split_42.json`, 2 019 treino / 505 validação,
`SEED=42`), com tratamento intacto — a única mudança foi o split, isto é, o **instrumento de
medida**, não o objeto medido. A reprodutibilidade foi verificada: uma segunda execução
reproduziu os fingerprints de inicialização e finais `[MEDIDO]`.

| tarefa | ICM-Norm (hard) | ICM-Soft-Norm (soft) |
|---|---:|---:|
| A — sexismo sim/não | 0,5483 | 0,4972 |
| B — intenção da fonte | 0,4579 | 0,3713 |
| C — categorias finas | 0,3784 | 0,1378 |

Com limiares ajustados, a Task C hard sobe para 0,4267 `[MEDIDO]` — **mas** esse ajuste foi
feito e reportado no mesmo conjunto de validação, o que é o regime otimista do sistema
antigo. O sistema corrigido ajusta numa metade da validação e reporta na outra (ARCH-11b).
**Os dois regimes não são comparáveis** e a tese não deve pô-los na mesma coluna.

#### A definição de baseline e a regra de simetria que ela impõe

Havia duas leituras defensáveis do que é "o baseline": o **estado final da execução** (o
modelo SWA, 0,5483 em A hard) ou a **melhor época da curva** (0,5872, época 6). Foi escolhido
o estado final, com a justificação de que assim o baseline reflete o que o código de facto
produz quando executado de ponta a ponta, sem uma escolha de época feita depois de ver os
resultados.

Essa escolha cria uma **regra de simetria obrigatória** que atravessa todo o resto do
capítulo: se o baseline é o estado final da execução, o sistema corrigido tem de ser
reportado pelo seu estado final também. Nunca o fim de um lado contra o pico do outro — isso
fabricaria ganho. Toda decisão de seleção de época descrita em §N.5 deriva desta regra.

#### A atribuição que tem de ser declarada

O sistema corrigido removeu o SWA. Comparado contra o estado final do baseline, **parte de
qualquer delta vem dessa remoção**, não das restantes correções. Isto está medido na própria
execução do baseline: o modelo SWA é pior que o melhor ponto da curva em **6/6 métricas**
`[MEDIDO]`.

---

## N.2 Auditoria de código: o que estava errado e o que cada erro custou

A auditoria foi conduzida por três análises independentes (arquitetura, simplificação,
otimização), seguidas de reconciliação e revisão de código. Os itens abaixo são os que foram
**confirmados por evidência direta**, não hipóteses.

### N.2.1 ARCH-01 — os biases `q_bias`/`v_bias` do VideoMAE chegavam como zeros

Em `transformers==5.6.0`, o checkpoint `MCG-NJU/videomae-base` traz `q_bias`/`v_bias`
(resquício da projeção `qkv` fundida do VideoMAE original), enquanto a implementação da v5
espera `query.bias`/`value.bias` das camadas `Linear` separadas. O mapeamento dos **pesos**
foi feito na refatoração da biblioteca; o dos **biases**, não.

Medido em runtime na VM `[MEDIDO]`, não inferido do relatório de carregamento:

```
query.weight carregou?              True     <- o prefixo `videomae.` resolve
norma do q_bias no checkpoint:     17.541    <- o bias treinado existe
norma do query.bias no modelo:      0.000    <- e vira ZERO ao carregar
q_bias carregou?                    False
```

O facto de `query.weight` carregar prova que o problema **não é o prefixo** — se fosse, os
86 M de parâmetros do encoder estariam todos em falta. É o **nome**. Nas 12 camadas, um bias
treinado de norma ≈12,9 por camada (mín. 10,5, máx. 17,8) era substituído por zeros
`[MEDIDO]`.

Nota de desenho: `key.bias` também aparece em falta, e a correção **zera-o de propósito** — o
VideoMAE não tem `k_bias` por construção, só `q_bias` e `v_bias`, logo zero é o valor
correto, não um remendo.

### N.2.2 ARCH-04 — diluição das features de áudio por padding

`whisper.pad_or_trim` força uma janela fixa de 30 s, e a média sobre os 1 500 frames do
encoder inclui o silêncio sintético. Medido na população completa (n = 2 524 treino, 674
teste, via `ffprobe`) `[MEDIDO]`:

| | treino | teste |
|---|---|---|
| duração mediana | 18,19 s | 15,05 s |
| < 30 s (diluídos) | 66,6 % | 76,0 % |
| > 30 s (truncados) | 33,4 % | 24,0 % |
| `valid_fraction` média | **0,655** | **0,582** |
| fração do vetor `[1280]` que é silêncio | 34,5 % | 41,8 % |

No treino, 16 104 s (24,5 %) de áudio real **nunca chegam ao encoder**, e 26 087 s de silêncio
sintético são injetados.

Dois pontos merecem estatuto próprio no texto da tese:

1. **As features não estavam corrompidas — estavam diluídas.** Zero de 2 524 vetores com
   norma nula; norma média 14,96. Corrupção deteta-se; diluição passa em toda a verificação
   de sanidade. É um argumento metodológico transferível.
2. **Treino e teste sofrem o confound em intensidades diferentes** (0,655 vs 0,582), o que
   constitui *dataset shift induzido pelo pipeline de extração* — não foi reportado por
   nenhuma das análises independentes e só apareceu ao medir a população inteira.

Corrigido na extração v2 (máscara de padding + média ponderada por janela). O baseline B15 e
todos os braços deste capítulo treinam já sobre o áudio v2.

### N.2.3 ARCH-05 — truncagem de transcrições

`max_length=128` truncava **42,9 %** das transcrições (mediana 105 tokens, p90 364, máximo
1 122) `[MEDIDO]`. Corrigido para 256, que cobre 78 % integralmente.

### N.2.4 ARCH-06 — SWA produzia um modelo final pior que a época intermédia

15 épocas com SWA a partir da 5 produziam um estado final pior que a época 4/6 em **todas as
seis métricas** (Task A ICM-Norm hard 0,6199 → 0,5647 na medição original) `[MEDIDO]`. O
achado **reproduziu** em split e seed diferentes: 0,5872 (época 6) → 0,5483 (SWA), mesma
direção e magnitude parecida `[MEDIDO]` — não era artefato do split antigo.

SWA foi removido. A época passa a ser escolhida por `E = argmin(eval_loss)`, **nunca** pela
métrica reportada — essa seria a definição operacional de leakage neste projeto.

Nota técnica que vale registar: SWA entra em conflito com `load_best_model_at_end` do HF
Trainer — os pesos SWA são sobrescritos se o carregamento do melhor checkpoint estiver ativo.
Exige escolha explícita de estratégia, e no sistema submetido essa escolha não era explícita.

### N.2.5 ARCH-20 e SIMP-01 — a tabela de ablação do artigo

Três problemas independentes, todos com consequência sobre uma tabela **já publicada**:

- **ARCH-20:** os braços de ablação **partilhavam os mesmos pesos**, sempre treinados com
  bio+viewer; as flags apenas zeravam a entrada no `forward`. `B0_content_only` não era um
  modelo content-only — era uma sonda de zeragem de input. Braços que partilham pesos não são
  ablações.
- **SIMP-01:** `ablation_results.csv` é md5-idêntico a `ablation_seed_2025.csv`
  (`718264812c8df36df087a1484c95b6b2`) `[VERIFICADO]`. A tabela de ablação do artigo é **uma
  seed apresentada como agregado de três**. A função `run_ablations_multi_seed` existe no
  código e nunca foi chamada.
- **SIMP-01b:** `B2_bio_viewer_no_tau` e `FULL_bio_viewer_tau` tinham dicionários de flags
  **idênticos**; não existia parâmetro `use_tau` (era absorvido por `**kwargs`). Um dos braços
  da tabela é um fantasma.

Correção de desenho adotada: **flags de ablação vão no construtor do modelo, não no
`forward`**.

### N.2.6 OPT-06 — "três seeds" que eram repeated holdout

`SEED` controlava simultaneamente o split, a inicialização e a ordem dos dados
(`train_test_split(random_state=SEED)`) `[VERIFICADO]`. As "3 seeds" eram, portanto,
**repeated holdout sobre três conjuntos de validação diferentes**, não seeds pareadas.
Importar a análise pareada do modelo de memes para cá teria sido estatisticamente errado.

Corrigido separando `SPLIT_SEED` (fixo, artefato commitado) de `INIT_SEED`.

### N.2.7 Dois erros menores com efeito não-menor

- **BUG-I:** a `cross_attn` executava incondicionalmente, pelo que uma modalidade zerada pelo
  modality dropout ainda era misturada e **re-normalizada** por `cross_norm`. Ativo em todos
  os treinos realizados até à auditoria.
- **Modality dropout por batch, não por amostra:** usava `torch.rand(1)`. Corrigido para
  `torch.rand(B)` com faixas posicionais, mantendo a taxa por modalidade igual em todos os
  braços.

### N.2.8 Reprodutibilidade

As execuções eram irreproduzíveis por duas causas independentes `[VERIFICADO]`: o modelo era
instanciado antes do `Trainer`, pelo que as camadas novas nasciam de um RNG nunca semeado (o
`set_seed` interno do Trainer chega tarde demais); e kernels CUDA não-determinísticos
amplificavam as diferenças ao longo das épocas.

A correção — semeadura completa, `cudnn.benchmark=False`, `use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, re-seed imediatamente antes de cada inicialização de pesos,
e fingerprints SHA-256 impressos no init e pós-treino — custa ~10 % de velocidade e foi
verificada na VM: fingerprints iniciais e finais idênticos entre duas execuções `[MEDIDO]`.

Esta infraestrutura não é acessória. Sem ela, nenhuma das comparações de §N.4–N.5 seria
interpretável: várias delas assentam em runs que têm de ser bit-a-bit reproduzíveis para que
a diferença observada seja atribuível ao eixo manipulado.

---

## N.3 Investigação causal: onde está, de facto, o sinal

Antes de gastar GPU a melhorar o ramo de vídeo, foi conduzida uma investigação por sondas
sobre o checkpoint B15 (`run_B2_bio_viewer_seed42_B15/`, frames v2 + áudio v2,
`restore_biases=True`, 15 épocas), na validação congelada de 505 amostras. **Nada foi
treinado**: sete testes, todos sobre um forward cacheado.

### N.3.1 Método: permutação, não zeragem

O método é *um forward + replay exato*: um hook captura o vetor concatenado como o modelo o
funde, e os testes são reexecutados correndo apenas `fusão → heads`. Isto é válido porque em
`eval()` o forward não tem nenhuma operação entre amostras — e é proibido por regra do
projeto haver normalização depois da concatenação (§N.1.1).

A escolha de **permutar** em vez de **zerar** é o núcleo metodológico do capítulo, e três
medições justificam-na `[MEDIDO]`:

1. **Zerar a entrada normalizada engana.** Zero pós-normalização é o pixel médio do ImageNet
   — uma imagem cinzenta plausível — que produz um embedding constante mas não-nulo, absorvido
   como viés fixo. As métricas quase não mudam e algumas **melhoram**, sugerindo falsamente
   que o vídeo não contribui.
2. **Zerar o vetor pós-pooling subestima por construção**: reproduz exatamente a operação do
   modality dropout, para a qual o treino criou robustez. Permutar magoa **3,5×** mais do que
   remover.
3. **Ruído isotrópico é o controlo errado.** `perm_across` vs `noise_across` difere 0,0278,
   abaixo da tolerância pré-registada de 0,0501: **sem poder estatístico a n=505**.
   "Indistinguível" não é "confirmado" — e a árvore de leitura original do notebook caiu nesta
   armadilha, imprimindo uma conclusão que teve de ser corrigida.

O controlo que discrimina é **permutação condicional à classe, com magnitude casada**.

### N.3.2 A hierarquia causal entre modalidades

Δ médio sobre 5 derangements `[MEDIDO]`:

| slot | dim | ‖Δ‖ | A hard | B hard | C hard |
|---|---:|---:|---:|---:|---:|
| vídeo | 768 | 13,154 | −0,0079 | −0,0631 | +0,0032 |
| áudio | 768 | 13,066 | −0,0120 | −0,0574 | −0,0098 |
| **texto** | 1024 | 18,596 | **−0,2313** | **−0,2541** | **−0,2160** |
| bio | 128 | 2,820 | −0,0078 | −0,0207 | −0,0022 |

**O texto domina por uma margem enorme.** A contribuição pequena do vídeo é *proporcional,
não anómala* — o conteúdo sexista chega pela transcrição.

⚠ **Caveat obrigatório:** estes Δ medem **dependência do modelo treinado**, não **teto de
informação da modalidade**. Um ramo que o treino aprendeu a ignorar dá Δ≈0 mesmo que a
modalidade contenha sinal. Esta frase tem de acompanhar a tabela sempre que ela for citada.

### N.3.3 A afirmação de teto: o vídeo é um classificador fraco de intenção

O teste decisivo compara dois derangements com magnitude de perturbação praticamente idêntica
e que diferem apenas na **classe gold de B do doador** — informação que o modelo nunca vê
`[MEDIDO]`:

| arm | ‖Δ vídeo‖ | B hard |
|---|---:|---:|
| `perm_across` (entre classes) | 13,154 | **−0,0631 ± 0,050** |
| `perm_within` (dentro da classe) | 13,119 | **−0,0002 ± 0,021** |

A diferença (0,0629) excede a tolerância pré-registada (0,0501). Permutar o vídeo dentro da
mesma classe de B custa **essencialmente nada, nas seis métricas** — não apenas em B.

**Leitura:** o slot de vídeo não carrega nada **além** da classe de B. Ele funciona como um
**classificador fraco de intenção da fonte, e nada mais.** Se carregasse intensidade,
categoria fina ou detalhe de enquadramento, trocar dentro da classe custaria alguma coisa.

Dois caveats que têm de ser escritos **antes** que a defesa os levante:

- **`perm_within` é oracular na construção** — agrupa pelo gold de B. É legítimo (o gold
  escolhe o doador, nunca entra no modelo), mas tem de estar declarado.
- **É teto do vetor deste checkpoint, não teto da modalidade vídeo.** Nenhuma sonda sobre
  features congeladas pode fechar essa questão; só um braço treinado com outro encoder — que
  é exatamente o que §N.4 faz.

### N.3.4 O mecanismo, com o eixo corrigido

A hipótese inicial era que o visual separa DIRECT de JUDGEMENTAL. A matriz de flips
(2 525 decisões, 574 flips, 22,7 %, concentrados em 231 ids únicos) diz outra coisa
`[MEDIDO]`:

| eixo | flips | % | acaso |
|---|---:|---:|---:|
| **DIRECT ↔ NONE** | 302 | **52,6 %** | 33,3 % |
| DIRECT ↔ JUDGEMENTAL | 140 | 24,4 % | 33,3 % |
| JUDGEMENTAL ↔ NONE | 132 | 23,0 % | 33,3 % |

O eixo D↔J está **abaixo do acaso**. A reformulação correta é: *o visual entrega
enquadramento, e o enquadramento discrimina a enunciação em primeira pessoa (DIRECT) do
não-sexista (NONE)*.

`[HIPÓTESE]` Mecanismo plausível: "a fonte fala para a câmara" é visualmente distintivo,
enquanto separar "comenta material de terceiros" (JUDGEMENTAL) de NONE pelo visual é muito
mais difícil — ambos podem ser reação, duet ou greenscreen. **Não escrever como facto.**

Nos 574 flips, o acerto contra o gold próprio cai de 51,9 % para 34,3 % `[MEDIDO]`: o slot de
vídeo estava a entregar classe **correta** nesses itens. Junto com `perm_within ≈ 0`, fecha o
argumento de que se trata de informação alinhada ao rótulo, não de ruído.

### N.3.5 Três descrições de arquitetura que a tese tem de corrigir

Independentes de qualquer resultado — são descrições do que o código faz `[VERIFICADO]`:

1. **O `TemporalAttentionPool` não é temporal.** O softmax opera sobre os 1 568 tokens
   espácio-temporais (8 posições × 196 patches), não sobre frames. Espaço e tempo competem no
   mesmo softmax e nenhuma informação de *onde* ou *quando* sobrevive à soma global.
2. **Não é pooling, é quase um seletor esparso.** Peso mediano do token máximo = 0,502; massa
   mediana do top-10 = 0,970; ~5 tokens efetivos de 1 568; 5 % dos vídeos operam com 1,42
   `[MEDIDO]`. Uma média ponderada em que um elemento tem peso 0,5 não é agregação.
3. **A fusão dá ao vídeo 18,3 %, não "mais que ao texto".** Em peso × escala: texto 52,2 %,
   vídeo 18,3 %, áudio 26,5 %, bio 3,0 % `[MEDIDO]`. A norma de coluna maior do vídeo mede
   aproximadamente o inverso da escala da feature, e a camada linear compensa por unidade. A
   leitura anterior ("a fusão privilegia o vídeo") está invertida e propagou-se para o artigo.

### N.3.6 Achados transversais

- **O confound de duração atravessa três modalidades e é pré-treinado, não aprendido.**
  Decodificabilidade de "duração > mediana" (AUC − nulo): áudio +0,494, bio +0,361, texto
  +0,323, tokens crus de vídeo +0,297. E é **idêntico com e sem LoRA** (+0,297 vs +0,297)
  `[MEDIDO]` — o que **exonera o treino** e localiza o confound no encoder e no pipeline de
  extração.
- **O pool aprendido funciona como filtro de nuisance.** Duração: +0,297 nos tokens crus →
  +0,117 no vetor final. Língua: +0,055 → −0,076 `[MEDIDO]`. O seletor descarta a variável de
  captura — ponto **a favor** do pool.
- **O vetor de vídeo não codifica língua** (AUC 0,447, abaixo do nulo) `[MEDIDO]`, ao
  contrário de texto (+0,466), áudio (+0,424) e bio (+0,411). Achado limpo, e relevante porque
  foi o confound de língua que arruinou a normalização per-viewer.
- **Um vetor de áudio de 1 280-d pré-extraído e congelado iguala 16 frames processados por um
  encoder adaptável** (áudio −0,0574 vs vídeo −0,0631 em B hard) `[MEDIDO]`. Vai diretamente
  ao argumento de capacidade absoluta.

### N.3.7 Rotas fechadas por medição, sem gastar treino

| rota | razão do fecho |
|---|---|
| trocar o agregador do vídeo | `mean_all`, `mean_temporal` e `top5` empatam com o pool aprendido dentro da tolerância; `top5 ≈ pool` mostra que não há sinal na cauda de 1 563 tokens |
| "a fusão ignora o vídeo" | o slot pesa 18,3 % — não é descarte |
| mais adaptação LoRA no vídeo | primeiro incremento rende abaixo do ruído (+0,004/+0,005 sob `mean_all`) |
| cross-attention sobre os 1 568 tokens | o critério de reabertura foi atendido, mas o teste do agregador fechou a rota: seria uma forma mais cara de não encontrar nada |
| concatenar os 1 568 tokens na fusão | 1 233 125 376 parâmetros só na camada de fusão |
| braço de 32 frames | `num_frames` é livre no VideoMAE `[VERIFICADO na VM]`, mas o gap continua a ser função da duração: a razão p90/p10 permanece 7,2× para 16, 32 ou 64 frames |

O que sobra é a hipótese H2 na forma corrigida — **capacidade absoluta + objetivo de
pré-treino** — e a única forma de a testar é trocar o encoder.

> **Nota sobre a razão texto:visão.** A formulação inicial da H2 ("o vídeo tem pouca
> capacidade *relativamente ao texto*") foi **refutada** `[VERIFICADO]`: no modelo de memes,
> que funciona, o texto recebe ~355 M parâmetros treináveis contra 50,4 M da visão (≈7×),
> enquanto no vídeo a razão é ≈2,7× (1,57 M vs 0,59 M). O modelo de memes é *mais* dominado
> por texto em termos relativos e a sua visão contribui. A H2 tem de assentar em capacidade
> **absoluta** (0,59 M em rank-16 contra 50,4 M full-rank) e no **objetivo de pré-treino**
> (contrastivo vs reconstrução de pixels).

---

## N.4 Troca de encoder: o braço CLIP

### N.4.1 Desenho e o que foi mantido constante

O braço substitui o VideoMAE por CLIP-ViT-L/14 aplicado **frame a frame**, com as 4 últimas
camadas de visão destravadas full-rank (~50,4 M parâmetros treináveis) — transplante literal
do padrão já em produção no sistema de memes. Os mesmos 16 frames alimentam o encoder; a
fusão passa a `2944 → 1024` (com bio).

Tudo o resto é herdado sem alteração — amostragem, split, áudio v2, bio, texto+LoRA, pesos de
loss, caminho de avaliação e orçamento de 15 épocas — precisamente para que qualquer delta
seja atribuível ao encoder. O código vive num módulo *overlay* que importa do original em vez
de o duplicar; `video_core.py` nunca é editado.

**Duas limitações declaradas antes de treinar:**

- **Interpolação:** os frames em disco foram redimensionados com interpolação bilinear (o
  processador do VideoMAE); o CLIP nativo usaria bicúbica. O braço não re-extrai pixels,
  apenas reaplica a normalização, pelo que herda o resize bilinear. `[HIPÓTESE]` o efeito é
  uma pequena perda de alta frequência face ao regime de pré-treino do CLIP; não é mensurável
  a partir do repositório.
- **Temporalidade:** o CLIP processa cada frame isoladamente — não há pareamento entre frames.
  O argumento de que isto não é custo severo assenta na medição de §N.3.5 (o pool já não é
  temporal na prática e o sinal existente comporta-se como propriedade de frame único), mas é
  `[HIPÓTESE]` por analogia, não medição do braço CLIP.

### N.4.2 Resultado da primeira execução, e a anomalia

`[MEDIDO]` Estado final da época 15:

```
A 0,5741 / 0,5172  ·  B 0,4986 / 0,4122  ·  C 0,3640 / 0,1368
```

Contra o baseline B15 (A 0,5536/0,5058 · B 0,4968/0,3947 · C 0,3436/0,1514), os números
agregados sobem em cinco das seis métricas e descem em C soft. Mas a sonda causal aplicada ao
novo checkpoint revelou um padrão que nenhum número agregado mostrava:

**o ramo de texto colapsou.** `‖Δ_texto‖ = 0,785` — **24× menor** que os 18,596 do B15 — com o
mesmo código de texto, o mesmo LoRA e os mesmos dados. Simultaneamente, o vídeo passou a
dominar causalmente todas as tarefas (`‖Δ_vídeo‖ = 23,658`, com Δ de −0,20 a −0,33 nas seis
métricas — a ordem de grandeza que o texto causava no B15) `[MEDIDO]`.

Ou seja: o encoder novo não acrescentou informação ao sistema. **Substituiu o canal por onde
a informação passava.**

### N.4.3 Duas hipóteses de correção, ambas refutadas

**Hipótese 1 — overtraining.** Se o colapso fosse degeneração tardia, selecionar
`E = argmin(eval_loss)` recuperá-lo-ia. O mínimo cai na **época 2** (1,5158), subindo
monotonicamente depois. No checkpoint dessa época: `‖Δ_texto‖ = 3,138` — melhoria de 4× sobre
o colapso, mas ainda 6× abaixo do B15 — e `‖Δ_vídeo‖ = 17,315`, ainda dominante `[MEDIDO]`.
**Refutada:** o colapso já está presente cedo no treino; não é acumulação tardia.

**Hipótese 2 — desequilíbrio de capacidade do encoder.** Se o CLIP full-rank (~50,4 M) esmaga
o LoRA do texto (~1,57 M), reduzir `UNFROZEN_VISION_LAYERS` de 4 para 2 deveria reequilibrar.
Resultado: `‖Δ_texto‖ = 1,046` (**piorou**, quase de volta ao colapso) e `‖Δ_vídeo‖ = 18,433`
(**subiu**, agora acima do B15) `[MEDIDO]`. **Refutada — e na direção oposta à prevista:**
menos capacidade no encoder tornou o vídeo *mais* dominante.

`[HIPÓTESE]` bem fundamentada para o mecanismo, não medida diretamente: `video_pool` e
`fusion` são sempre full-rank e treinados com `lr=1e-3`, independentemente daquela constante —
ela controla quanto o CLIP *se adapta*, não a força do que ele já entrega congelado. As
features do CLIP-L pré-treinado são semanticamente fortes mesmo sem adaptação, pelo que a
competição na fusão é decidida pela **qualidade relativa das features**, não pela contagem de
parâmetros treináveis do encoder. Com 2 camadas o CLIP fica mais genérico e estável — um
atalho ainda mais fiável para a fusão. O `‖Δ_texto‖` baixo significa que o XLM-R+LoRA foi
**treinado até ficar quase constante**: o gradiente conjunto ensina o texto a calar-se porque
o vídeo já resolve.

### N.4.4 A decisão de estado, e o debate que ela encerrou

Após as duas refutações, o braço foi revertido para 4 camadas e passou a ser reportado no
**estado final da época 15**, abandonando `argmin(eval_loss)` — por simetria com o baseline
(§N.1.3) e porque o padrão passou a ser lido como **trade-off real de trocar o encoder**, não
como bug corrigível por seleção de época.

A época 7 chegou a ser considerada (vale local de `eval_loss` em 1,6121, e picos absolutos de
A hard 0,5801 e B hard 0,5046). **Rejeitada por dois motivos independentes:** não é o argmin
global (a época 2 é mais funda), e só ganhou consideração *depois* de se olhar para as
métricas — que é exatamente o mecanismo de leakage proibido; e não existe "época de
equilíbrio" nesta curva, porque cada subtarefa pica numa época diferente (A em 7, B em 10, C
hard em 10-11, C soft em 15, ainda a subir no fim do orçamento).

### N.4.5 O que o braço CLIP estabeleceu

- A H2 foi **testada**, não apenas argumentada. Trocar o encoder muda o comportamento do
  sistema de forma substancial e mensurável.
- O ganho agregado existe em cinco das seis métricas face ao B15, mas vem acompanhado de um
  **custo estrutural**: o sistema passa a depender de um canal que carrega formato, não
  conteúdo (§N.6.2).
- Os artefatos oficiais do braço são os da execução de 4 camadas em estado final da época 15.
  As execuções descritas em §N.5 são **braços de atribuição**, não substituem este.

---

## N.5 A questão da Task C: quatro tentativas, quatro resultados

A Task C soft é o número aberrante do sistema — 0,1514 no B15, menos de metade da C hard e
menos de um terço da A soft. No braço CLIP, ela continua a subir muito depois do `argmin` da
loss: 0,0388 na época 2 (o argmin global) contra 0,1368 na época 15, ainda a subir no fim do
orçamento, enquanto A e B saturam mais cedo `[MEDIDO — curva das 15 épocas]`.

> ⚠ **Discrepância entre documentos, a resolver antes da versão final.**
> `18_CLIP_VIDEO_BRANCH_SPEC.md` §9.3 descreve esta divergência como "0,0524 na época 4 →
> 0,1368 na 15" atribuindo-a ao braço CLIP de 4 camadas. Mas a tabela da própria execução
> (§9.1 do mesmo documento) dá C soft = 0,0834 na época 4, e o valor 0,0524 pertence ao braço
> **CLIP_L2** (2 camadas), conforme `20_EPOCH_SELECTION_TASK_C.md` §0. Os números usados acima
> vêm da tabela primária da execução, não da prosa do §9.3. O fenómeno — C soft a subir muito
> depois do argmin — não depende de qual dos dois pares se usa.

Isto abriu uma bifurcação com duas leituras incompatíveis:

- **(i)** a Task C está **sub-treinada** dentro do treino conjunto; ou
- **(ii)** a **representação do trunk não carrega sinal suficiente de C** para ser extraído.

Quatro tentativas atacaram esta questão. Nenhuma produziu ganho, e é a *forma* como falharam
que constitui o resultado.

### N.5.0 Um facto aritmético que enquadra tudo

`loss_task_c` reduz as 6 classes com `.mean(dim=-1)`, enquanto `loss_task_b` usa
`.sum(dim=-1)` sobre as 3 `[VERIFICADO]`. Com `LOSS_W_A=1,0`, `LOSS_W_B=0,7`, `LOSS_W_C=0,7`,
o peso **efetivo** da Task C é `0,7 × 1/6 ≈ 0,117`. O critério de seleção de época é, por
aritmética, dominado por A e B. E **não existe `eval_loss_C`** — as perdas por tarefa só são
registadas nos logs de treino, pelo que nunca se soube onde cai `argmin(eval_loss_C)`
`[VERIFICADO]`.

### N.5.1 Tentativa 1 — seleção de época por tarefa

Rejeitada como rota principal por duas razões: qualquer critério adotado *depois* de ver a
curva da Task C é pós-hoc e irrecuperável para as execuções já feitas; e o "checkpoint
Frankenstein" (trunk de uma época + head de outra) **não tem nenhuma fonte que o autorize**.
Os *model soups* mediam modelos inteiros dentro da mesma bacia; e o mecanismo *Resurrection*
do MAGPIE é evidência **contra**, existindo precisamente porque uma head parada precisa de
se reajustar à representação latente mais recente.

A variante limpa seria seleção de **modelo inteiro por tarefa** (precedente publicado em
Hitachi@MRP 2019 e Harutyunyan et al. 2019), ao preço de "o sistema" passar a ser dois
modelos e não um — o que teria de ser declarado como tal.

### N.5.2 Tentativa 2 — probing das heads sobre trunk congelado: INCONCLUSIVO

Protocolo pré-registado antes de correr: congelar o trunk, extrair o vetor pós-fusão (1024-d)
e treinar cada head isoladamente sobre features congeladas, com as heads A e B a servirem de
**controlo** — se o protocolo for sólido, re-treinar a head A sobre a mesma representação deve
reproduzir aproximadamente o resultado original.

**Os controlos falharam nos 4 casos** (A e B, nos dois trunks), sempre na direção errada e com
maior amplitude no soft (ex.: CLIP-A soft 0,5172 → 0,3890) `[MEDIDO]`. Pelo *gate*
pré-registado, o veredito sobre C que o notebook chegou a calcular **não é válido** e não deve
ser citado: não é "C sem sinal" nem "C sub-treinada" — é falha do instrumento.

`[HIPÓTESE]` para o mecanismo: as `best_epoch` das probes (3 a 15, de um orçamento de 300)
mostram que pararam pelo argmin da própria BCE de validação muito cedo, antes de a superfície
de decisão calibrar para o que o ICM mede. É a mesma divergência loss-cedo/métrica-tarde que
o experimento pretendia estudar, a reaparecer *dentro do instrumento de diagnóstico*.

**Decisão registada:** rota fechada, **não consertada**. Ajustar LR ou épocas até os controlos
empatarem seria calibrar o instrumento mirando o resultado esperado — e, mesmo bem-sucedido,
devolveria diagnóstico, não ganho.

### N.5.3 Tentativa 3 — repesar a Task C na loss (`LOSS_W_C` 0,7 → 4,2)

Valor **principiado, não afinado por busca**: `0,7 × 6 = 4,2` desfaz exatamente a assimetria
`.mean` vs `.sum` de §N.5.0, sendo equivalente a trocar a redução para `.sum`. Nenhum grid,
nenhuma escolha por métrica de validação. Todo o resto congelado face ao braço CLIP.

`[MEDIDO]` Estado final da época 15:

| métrica | `CLIP` | `CLIP_WC42` | Δ |
|---|---:|---:|---:|
| A hard | 0,5741 | 0,5353 | **−0,0388** |
| A soft | 0,5172 | 0,5163 | −0,0009 |
| B hard | 0,4986 | 0,4733 | **−0,0253** |
| B soft | 0,4122 | 0,4124 | +0,0002 |
| C hard | 0,3640 | 0,3318 | **−0,0322** |
| C soft | 0,1368 | 0,1456 | +0,0088 |

Com limiares ajustados (mesmo protocolo metade-ajuste/metade-reporte dos dois lados), a C hard
vai de 0,3697 para 0,3322 — **−0,0376**. A queda **não** é artefato de calibração de limiar.

**Refutada por medição.** A realocação ocorreu, mas paga A hard, B hard **e C hard** por
+0,0088 em C soft — o único ganho, de uma seed, e ainda abaixo do B15 (0,1514). Não é sequer o
trade-off que a intervenção prevê: a própria Task C piora na métrica hard.

**Consequência para a bifurcação:** multiplicar o gradiente efetivo da Task C por 6 — a
correção de causa do ramo (i) — **não fez a Task C render mais**. Evidência direta contra (i)
e a favor de (ii). ⚠ Uma seed, sem replicação: evidência convergente, não prova.

### N.5.4 Tentativa 4 — congelar as heads A/B, com controlo de orçamento

Hipótese: se A e B disputam o trunk com C, congelar `head_a`/`head_b` liberta capacidade para
C sem que A/B degradem.

Duas notas de desenho que são, elas próprias, resultado metodológico:

- **Congelar uma head não corta o gradiente do trunk.** `requires_grad=False` impede que os
  *pesos da head* mudem, mas o gradiente continua a fluir através dela. As losses de A e B
  foram mantidas ativas com os pesos originais precisamente por isso: as heads congeladas
  passam a funcionar como *probes fixas* que **ancoram** o trunk. Remover as losses de A/B é
  que soltaria o trunk.
- **Foi executado um braço de controlo.** Como a fase 2 acrescenta 5 épocas que o baseline não
  teve, "o freeze ajudou" seria inseparável de "5 épocas extras ajudaram". O controlo parte do
  mesmo checkpoint, corre as mesmas 5 épocas, e difere **apenas** no freeze.

O freeze foi verificado nos artefatos gravados, não inferido: `head_a` e `head_b` estão
bit-a-bit idênticas ao checkpoint-base no braço com freeze, e diferentes no controlo;
`head_c` mudou nos dois `[MEDIDO]`.

`[MEDIDO]` Estado final da época 5:

| métrica | `CLIP` (ép. 15) | `FT5_FREEZEAB` | `FT5_CTRL` | Δ freeze |
|---|---:|---:|---:|---:|
| A hard | 0,5741 | 0,5657 | 0,5625 | +0,0032 |
| A soft | 0,5172 | 0,4976 | 0,4883 | +0,0093 |
| B hard | 0,4986 | 0,5139 | 0,4984 | +0,0155 |
| B soft | 0,4122 | 0,4063 | 0,3901 | +0,0162 |
| C hard | 0,3640 | 0,3486 | 0,3498 | −0,0012 |
| **C hard, limiares ajustados** | 0,3697 | 0,3126 | 0,3390 | **−0,0264** |
| C soft | 0,1368 | 0,1400 | 0,1278 | +0,0122 |

**Hipótese não sustentada.** O freeze preserva A e B — o que é quase mecânico e não era a
hipótese — e **não move C**: os dois sinais contradizem-se (soft +0,0122, hard pós-tuning
−0,0264). Congelar A/B não libertou capacidade para C; libertou-a para A e B, que já eram as
tarefas bem servidas.

**Achado colateral:** os **dois** braços de fase 2 ficam abaixo da fase 1 em C hard ajustada e
em A. Treinar além das 15 épocas piora, com ou sem freeze — coerente com a `eval_loss` a subir
monotonicamente desde a época 2. O orçamento de 15 épocas não é curto. ⚠ Parte da degradação
pode vir do reinício de warmup e scheduler (a fase 2 não é *resume* de otimização); isto vale
igual nos dois braços, pelo que não afeta a comparação primária, mas mantém este colateral
como **indício**, não prova.

### N.5.5 A conclusão convergente

Duas intervenções de mecanismos **opostos** — dar 6× mais gradiente à Task C, e retirar a
competição de A/B sobre o trunk — falharam em mover C de forma consistente.

**A leitura convergente é (ii): o gargalo da Task C é a representação do trunk, não a alocação
de treino entre tarefas.** É a mesma conclusão que o probing de heads pretendia alcançar e não
conseguiu, obtida aqui por duas vias independentes que não partilham o problema de protocolo
daquele.

⚠ **n = 1 seed em ambas.** Os deltas (0,003–0,038) estão na ordem de grandeza que o projeto já
documentou como variância de seed. O que sustenta a conclusão é a **convergência das duas
falhas**, não a magnitude de qualquer uma delas. Esta ressalva é obrigatória no texto final.

Consequência prática registada: a linha de investigação sobre a Task C está encerrada, e
qualquer procura por outro valor de peso ou outra combinação de heads congeladas seria uma
busca contra a métrica reportada.

---

## N.6 Discussão

### N.6.1 O que se aprende com resultados negativos

O programa experimental deste capítulo produziu, no essencial, achados negativos. Três razões
pelas quais isso constitui contributo e não falha:

**1. As rotas foram fechadas por medição, não por opinião ou por custo.** Trocar o agregador,
aumentar a adaptação LoRA no vídeo, cross-attention sobre os tokens, seleção de época por
tarefa, repesagem da loss, congelamento de heads — cada uma tem um número associado e um
protocolo que a fechou. A regra assimétrica adotada é explícita: **métricas não caírem fecha
uma opção; métricas caírem é apenas gatilho para gastar uma execução treinada.** Quem prova é
o braço treinado.

**2. O método é transferível.** A permutação condicional à classe com magnitude casada é o
contributo metodológico mais forte: distingue "o modelo não usa esta modalidade" de "esta
modalidade só carrega esta classe", ao custo de um forward e nenhum treino. Junto com os três
modos de falha documentados em §N.3.1 — zerar entrada normalizada engana, zerar pós-pooling
subestima, ruído isotrópico não tem poder a este n — constitui um pequeno guia de como *não*
medir contribuição de modalidade.

**3. Vários dos achados só apareceram porque se mediu a população inteira.** O *dataset shift*
induzido pelo pipeline de áudio (0,655 vs 0,582) não foi reportado por nenhuma das análises
independentes; o confound de duração ser **pré-treinado** e não aprendido só se estabeleceu
comparando com e sem LoRA; a tabela de ablação do artigo só se revelou uma seed única por
comparação de md5.

**4. Alguns erros são invisíveis a toda a verificação de sanidade.** As features de áudio
diluídas tinham norma média 14,96 e zero vetores nulos — corrupção deteta-se, diluição não. Os
biases zerados do VideoMAE não produzem exceção nem aviso: o treino corre e as métricas ficam
piores, indistinguível de o encoder simplesmente não ajudar. É um argumento a favor de auditar
o carregamento de pesos por medição direta, não pelo relatório da biblioteca.

### N.6.2 O achado de robustez: o vídeo carrega formato, não conteúdo

O resultado com maior consequência para a interpretação do sistema é o de §N.3.3: o slot de
vídeo funciona como classificador fraco de intenção da fonte e nada mais.

No braço CLIP isto agrava-se. O sistema não só depende de um canal de formato — passa a
depender **preferencialmente** dele, com o ramo de texto reduzido a quase constante
(`‖Δ_texto‖` 24× menor). Um modelo que troca o canal de conteúdo (a transcrição, onde o
conteúdo sexista de facto está) por um canal de formato pode **generalizar pior no teste real
mesmo com métricas de validação melhores** — porque "a pessoa fala para a câmara" é uma
propriedade do registo de produção do vídeo, não do que está a ser dito.

Este é um argumento de robustez **contra** o braço CLIP, independente do seu ganho agregado, e
deve ser apresentado como tal: um ganho de 0,02 em ICM-Norm obtido por um atalho de formato
não é o mesmo objeto que um ganho de 0,02 obtido por compreensão de conteúdo, ainda que a
tabela não os distinga.

### N.6.3 Limitações do capítulo, sem suavizar

- **Seed única na maioria dos braços.** Os braços WC42 e de freeze correram com `INIT_SEED=42`
  apenas. A validação cross-seed pareada existe como procedimento no projeto, mas não foi
  aplicada a estes braços.
- **n = 505 na validação.** Várias comparações são declaradas **inconclusivas a este n**, e
  isso está registado onde ocorre em vez de ser dissolvido em prosa.
- **As sondas lineares não provam uso pelo modelo nem ausência de informação não-linear.**
  Uma fusão treinada sobre tokens com mais capacidade poderia extrair de forma não-linear algo
  que nenhuma sonda deste capítulo veria.
- **Dois caveats bloqueiam achados específicos.** O perfil temporal de atenção (a alegação de
  que "o meio do vídeo é ignorado") depende de saber como a extração v2 tratou vídeos com
  menos de 64 frames decodificáveis — se houve repetição do último frame, o pico observado é
  artefato de pipeline. E os números do regime de pré-treino do VideoMAE (~7,6 fps, gap
  ~0,13 s) vêm da configuração citada no artigo original e **não foram confirmados em fonte
  primária**, o que afeta toda a afirmação da forma "o regime atual é ~8,6× mais esparso".
  Nenhum dos dois deve ir para a versão final antes de resolvido.
- **A divergência de interpolação bilinear/bicúbica do braço CLIP** não foi medida e não é
  mensurável a partir do repositório.

### N.6.4 O que fica em aberto

- Confirmar em fonte primária os parâmetros de pré-treino do VideoMAE.
- Resolver o confound de frames repetidos antes de usar o perfil temporal.
- Corrigir, no artigo e no diagrama, a largura de fusão (2688/2560, não 2592), a remoção da
  `cross_attn`, e a leitura invertida sobre o peso do vídeo na fusão.
- O ramo de texto — o único onde as sondas mostraram que o sinal reside — nunca recebeu mais
  capacidade de adaptação. É a rota não testada com melhor razão evidência/custo.

---

## Apêndice A — Tabela consolidada dos braços

Todos os valores em ICM-Norm (hard) / ICM-Soft-Norm (soft), validação congelada de 505
amostras, estado final da execução.

| braço | épocas | A hard | A soft | B hard | B soft | C hard | C soft |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline submetido (SWA) | 15 | 0,5483 | 0,4972 | 0,4579 | 0,3713 | 0,3784 | 0,1378 |
| B15 (corrigido, VideoMAE) | 15 | 0,5536 | 0,5058 | 0,4968 | 0,3947 | 0,3436 | 0,1514 |
| CLIP (4 camadas) | 15 | 0,5741 | 0,5172 | 0,4986 | 0,4122 | 0,3640 | 0,1368 |
| CLIP + `LOSS_W_C=4,2` | 15 | 0,5353 | 0,5163 | 0,4733 | 0,4124 | 0,3318 | 0,1456 |
| CLIP + freeze A/B (fase 2) | 15+5 | 0,5657 | 0,4976 | 0,5139 | 0,4063 | 0,3486 | 0,1400 |
| CLIP + controlo (fase 2) | 15+5 | 0,5625 | 0,4883 | 0,4984 | 0,3901 | 0,3498 | 0,1278 |

⚠ **Esta tabela não é uma tabela de ablação.** As linhas não são comparáveis todas entre si: o
baseline submetido inclui SWA e `cross_attn`; o B15 e o CLIP diferem em encoder **e** largura
de fusão; as duas últimas linhas partem do checkpoint CLIP e têm orçamento diferente. As
comparações legítimas são: B15 ↔ CLIP (troca de encoder), CLIP ↔ WC42 (repesagem isolada), e
FREEZEAB ↔ CTRL (freeze isolado). Qualquer outra leitura desta tabela mistura eixos.

Braços de seleção de época intermédios (CLIP época 2; CLIP 2 camadas época 4) estão
documentados em `18_CLIP_VIDEO_BRANCH_SPEC.md` §9.1 e foram omitidos aqui por não serem
estados finais de execuções com o protocolo definitivo.

---

## Apêndice B — Marcadores pendentes

Valores que este rascunho **não** pôde preencher a partir da documentação disponível:

- `[NÚMERO A CONFIRMAR: métricas exatas reportadas na tabela do artigo CEUR-WS submetido]` —
  necessárias para a comparação explícita "sistema submetido vs. sistema corrigido" de §N.0.1.
- `[NÚMERO A CONFIRMAR: contagem de parâmetros treináveis do sistema corrigido sem cross_attn]`
  — o valor documentado (11,0 M) é do sistema **com** `cross_attn`, da qual 4 987 904 eram do
  próprio módulo removido.
- `[NÚMERO A CONFIRMAR: composição exata do vetor biométrico de 17-d e a lista de features
  selecionadas por informação mútua]` — §N.1.1 descreve o slot pela dimensão de saída (128),
  não pela entrada.
- `[NÚMERO A CONFIRMAR: nº de vídeos e distribuição de rótulos do conjunto de treino oficial
  da Task 3]` — o capítulo usa 2 524 (treino) e 674 (teste) das medições de duração, e 2 019/505
  do split congelado; a relação exata entre estes conjuntos deve ser declarada na secção de
  dados.
- `[NÚMERO A CONFIRMAR: ‖Δ_texto‖ e ‖Δ_vídeo‖ da sonda causal sobre os braços WC42 e de fase 2]`
  — a sonda não foi re-executada sobre esses artefatos; §N.5 não faz nenhuma afirmação causal
  sobre eles.
