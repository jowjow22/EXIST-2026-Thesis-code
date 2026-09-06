# 17 — Opções de melhoria para RE-TREINO (Task 3, sem re-extração)

**Escrito em:** 2026-08-13
**Escopo (revisado):** a versão original deste documento só cobria mudanças de re-treino
sobre os dados já extraídos. **Você removeu essa restrição** — re-extração deixou de ser
descartada por padrão, entra na mesa se o ganho justificar. Os itens 1-10 abaixo continuam
sem pedir re-extração; os itens 11-12 pedem, e estão marcados como tal.
**Fontes:** `16_VIDEO_INVESTIGATION_RESULTS.md` (medições sobre o checkpoint
`run_B2_bio_viewer_seed42_B15/`), `09_BASELINE_RESULTS.md`, leitura direta de
`video_core.py`, e um levantamento de literatura feito para esta sessão.

---

## 0. Antes de tudo: o alvo de +0,2 de ICM

Você pediu "algo como +0,2 de ICM em alguma task". **Preciso dizer isso sem suavizar: um
ganho de +0,2 vindo de UMA mudança isolada de arquitetura ou de treino não tem precedente
na literatura consultada.** Nem uma fonte reporta efeito dessa ordem para uma escolha de
função de perda, de pooling ou de fusão.

O que a literatura efetivamente documenta, para uma técnica isolada, em datasets da ordem
de 2 000 amostras:

| item | valor |
|---|---|
| faixa típica de ganho por técnica isolada | **+0,01 a +0,05** absoluto |
| teto observado (caso muito favorável) | **+0,07** — e só quando o baseline estava mal casado com o problema |
| piso de ruído de seed | **~0,01** — ganho abaixo de 0,02 numa única run é indistinguível de sorte |
| ganho "líquido" depois do efeito reality-check de MTL | mais perto de **+0,01–0,03** |

E há uma ressalva que muda a leitura de tudo: esses números são **acurácia/F1/mAP em
benchmarks equilibrados**. A nossa métrica é ICM-Norm / ICM-Soft-Norm do PyEvALL, com
faixa dinâmica e sensibilidade diferentes. **Não existe conversão conhecida** de "+7,7
pontos de acurácia" para "+0,077 de ICM-Norm". O que transfere entre os dois mundos é
*direção* e *ordem de magnitude*, não o número.

Onde deltas da ordem de 0,1–0,2 aparecem de verdade na literatura? Em **troca de
backbone/arquitetura inteira** ou em **correção de bug**. Não em escolha de loss, pooling
ou fusão. (Aliás, no nosso próprio projeto isso se confirma: o maior salto documentado até
hoje veio do ARCH-01, que era um bug de carregamento de pesos.)

**Alvo defensável, na minha avaliação:**

- **+0,02 a +0,05** por técnica isolada, na melhor das hipóteses;
- **+0,03 a +0,08** para uma combinação bem escolhida (e mesmo isso é hipótese, não soma);
- **uma exceção condicional**, a Task C soft (hoje 0,1514), que está num piso anômalo e
  onde um movimento maior é *concebível* — ver item 3 e item 4.

Isso não é um veredito sobre prosseguir ou não. É o número real. A decisão é sua.

> **Nota:** você esclareceu depois que +0,2 era só um exemplo, não um alvo fixo. Os números
> abaixo continuam os mesmos — a análise não muda — só o tom de "confrontar o alvo" da
> abertura passa a valer menos: não há alvo fixo para confrontar, só ordens de magnitude
> reais para decidir com que expectativa entrar em cada aposta.

---

## 1. O que já sabemos medido, e que restringe tudo abaixo

### 1.0 De onde estamos partindo

Todos os "+0,0X" deste documento são somados a **estes** números — o checkpoint
`run_B2_bio_viewer_seed42_B15/` (frames v2 + áudio v2, `restore_biases=True`, 15 épocas),
medidos na val congelada de 505 amostras. Sem esta tabela à vista, "+0,03" não quer dizer
nada.

| task | ICM-Norm (**hard**) | ICM-Soft-Norm (**soft**) |
|---|---:|---:|
| **A** — sexismo sim/não | 0,5536 | 0,5058 |
| **B** — intenção da fonte | 0,4968 | 0,3947 |
| **C** — categorias finas | 0,3436 | **0,1514** |

Leitura rápida: **a Task C soft é o número aberrante do sistema.** É menos de metade da C
hard, e menos de um terço da A soft. É por isso que ela aparece como a única exceção
condicional ao teto de ganho da §0.

### 1.1 Os cinco fatos que restringem tudo

Cinco fatos do doc 16 que qualquer proposta tem de respeitar:

1. **O texto domina, por uma margem enorme.** Embaralhar o slot de texto entre amostras
   custa **−0,231 / −0,254 / −0,216** (ICM hard de A / B / C). Embaralhar o vídeo custa
   **−0,008 / −0,063 / +0,003**. O áudio custa algo parecido com o vídeo.
2. **O ramo de vídeo tem um teto medido:** ele funciona como um classificador fraco de
   intenção da fonte (Task B) e **nada mais**. Trocar vídeo entre amostras da *mesma*
   classe de B custa −0,0002.
3. **Trocar o agregador do vídeo não recupera nada** (TESTE 6) e **mais adaptação LoRA no
   vídeo rende abaixo do ruído** (TESTE 7). Essas duas rotas estão fechadas por medição.
4. **A cabeça da Task B é instável:** 45,7 % das amostras trocam de predição sob
   perturbação do vídeo, e um único bit de arredondamento em `bf16` já vira um `argmax`.
5. **O peso efetivo da Task C na loss é 0,117, não 0,7** — ver item 3. É a única
   discrepância grande entre o que a documentação diz e o que o código faz.

Consequência prática: **as apostas de maior retorno esperado não estão no ramo de vídeo.**
Estão no ramo de texto, no balanceamento das perdas e no pós-processamento das
probabilidades.

---

## 2. Itens INCLUÍDOS

Ordenados por retorno esperado (minha avaliação), do maior para o menor.

---

### Item 1 — Ampliar a adaptação do ramo de TEXTO (LoRA no MLP e/ou rank maior, só no XLM-R)

**1. O que é.**
Hoje o modelo não treina os encoders inteiros: ele congela quase tudo e treina só umas
"peças de encaixe" pequenas chamadas **LoRA** (adaptadores de baixo posto — matrizes
pequenas somadas às originais, com um "posto" `r` que controla o tamanho delas). Hoje
esses adaptadores estão presos a **dois** lugares dentro de cada camada de atenção
(`query` e `value`) e com `r=16`. A proposta é dar mais espaço de adaptação **só ao
encoder de texto**: alcançar também as camadas densas do bloco (o "MLP" que vem depois da
atenção) e/ou subir o `r` do texto.

**2. Por que pode ajudar AQUI.**
Porque é o único ramo em que está provado que o sinal mora: permutar o texto derruba as
métricas em ~0,23–0,25, contra ~0 do vídeo. E há um paralelo direto dentro do próprio
projeto: **no modelo de memes, o XLM-R é treinado inteiro** (~355 M de parâmetros,
`memes_core.py:473`, sem LoRA e sem freeze), enquanto aqui ele tem **1,57 M treináveis**.
Ou seja, no sistema que funciona bem, o ramo dominante recebe ~226× mais capacidade de
adaptação do que aqui. Não é um argumento genérico de "LoRA maior é melhor" — é apontar
capacidade para o ramo que a nossa própria sonda mostrou carregar o sinal.
A literatura reforça a direção pelo lado dos alvos: no único estudo sistemático de seleção
de módulos que li, alvejar **só `q`/`v`** é o perfil mais instável dos testados, e
acrescentar módulos do MLP dá os maiores ganhos justamente nas tarefas difíceis.

**3. Onde mexe no código.**
- `video_core.py:1167` — `make_lora_config()`. **Hoje ela é a MESMA para os dois
  backbones** (VideoMAE em `video_core.py:1260`, XLM-R em `video_core.py:1264`). Para
  configurar só o texto é preciso **mudar a assinatura** dessa função (ex.:
  `make_lora_config(r=LORA_R, alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES)`) e
  passar valores diferentes nas duas chamadas.
- `video_core.py:312-315` — constantes `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`,
  `LORA_TARGET_MODULES`; acrescentar um conjunto `*_TEXT`.
- ⚠ **Não copiar nomes de módulo da literatura.** Os papers falam em `o_proj`, `gate_proj`
  etc. — **esses nomes são do Llama e não existem no XLM-R nem no VideoMAE.** Em modelos
  RoBERTa do HuggingFace os nomes reais são `query`, `key`, `value`,
  `attention.output.dense`, `intermediate.dense`, `output.dense`. E o PEFT casa por
  **sufixo**: passar `"dense"` sozinho pegaria mais módulos do que se quer. Antes de
  rodar, imprimir na VM `[n for n,_ in xlmr.named_modules()][:80]` e escolher os sufixos
  exatos.
- Ponto de atenção de custo: subir `r` do XLM-R-large muda a contagem de treináveis e o
  tempo de época. Manter a convenção `alpha = 2r` recomendada pela literatura.

**4. Ganho esperado realista.**
**+0,01 a +0,04**, concentrado em A e B (as tasks onde o texto mais pesa). É a aposta com
melhor razão entre evidência interna e custo. Não é +0,2.

**5. Esforço/risco.**
**Médio.** Uma função muda de assinatura, duas chamadas mudam. Risco real: **overfitting**
— 2 019 amostras de treino e mais capacidade adaptativa é a receita clássica. E dá para
pôr número nisso (⚠ conferir a aritmética na VM antes de citar na tese):

| configuração do LoRA do texto | parâmetros treináveis |
|---|---:|
| hoje: 24 camadas × (`query`+`value`) × r=16 → 24 × 2 × 16 × (1024+1024) | **1 572 864** (≈1,57 M — bate com o doc 16) |
| acrescentando `intermediate.dense` (1024→4096) e `output.dense` (4096→1024), r=16 → 24 × 2 × 16 × 5120 | **+3 932 160** |
| total do degrau 1 | **≈5,5 M — cerca de 3,5× a capacidade atual** |

3,5× mais parâmetros treináveis sobre 2 019 amostras é exatamente o motivo de fazer **um
degrau por run**: se `r=32` entrar junto, a capacidade dobra de novo (~11 M) e não há como
saber qual das duas mudanças causou o que se observar. Mitigar
subindo um degrau de cada vez (primeiro só `target_modules`+MLP com `r=16`; só depois
`r=32`), nunca os dois juntos na mesma run.

**6. Conflitos com regras do projeto.**
Nenhum direto. Mas **cada degrau é um braço separado** — mudar alvos e rank na mesma run
torna o resultado não atribuível, que é exatamente a objeção registrada em
`16_...§11 ponto 3`.

---

### Item 2 — `MAX_LENGTH` de 256 para 384

**1. O que é.**
O texto (transcrição do vídeo) é cortado num número máximo de "tokens" (pedaços de
palavra). Hoje o corte é em 256. A proposta é subir para 384.

**2. Por que pode ajudar AQUI.**
Está medido no `CLAUDE.md`: a mediana das transcrições é 105 tokens, o **p90 é 364** e o
máximo 1 122. Com 256, 22 % das amostras continuam truncadas; com 384 cobrimos
integralmente ~90 %. E, de novo: é o ramo dominante — informação perdida ali é a que mais
custa. (Historicamente, ir de 128 para 256 já foi o ARCH-05, que truncava 42,9 %.)

**3. Onde mexe no código.**
`video_core.py:281` — `MAX_LENGTH = 256` → `384`. Uma constante. É usada por default em
`EXISTVideoDataset.__init__` (`video_core.py:743`) e registrada no `meta` do run
(`video_core.py:1932`), então a rastreabilidade já vem de graça.

**4. Ganho esperado realista.**
**+0,00 a +0,02.** Provavelmente pequeno — a mediana é 105, então a maioria das amostras
não muda nada. O ganho vem da cauda longa.

**5. Esforço/risco.**
**Trivial** do lado do código. O custo é de **memória/tempo**: a atenção do XLM-R-large
cresce com o quadrado do comprimento; 256→384 é ~2,25× no custo de atenção do ramo de
texto. Se a GPU der out-of-memory, o batch cai e o tempo de run sobe. Testar 384 antes de
sonhar com 512 (que é o teto do XLM-R).

**6. Conflitos com regras do projeto.**
Nenhum.

---

### Item 3 — Corrigir o peso EFETIVO da Task C na loss

**1. O que é.**
O modelo aprende três tarefas ao mesmo tempo (A, B, C) e soma as três "penalidades" (as
losses) numa só, com pesos. Os pesos escritos no código são
`LOSS_W_A=1.0, LOSS_W_B=0.7, LOSS_W_C=0.7`. Só que a loss da Task C tira uma **média**
sobre as 6 categorias, enquanto a da Task B tira uma **soma** sobre as 3 classes. Média
divide por 6; soma não divide por nada.

**2. Por que pode ajudar AQUI.**
Porque o peso **documentado** é (1,0 · 0,7 · 0,7) mas o peso **efetivo** é
**(1,0 · 0,7 · ~0,117)**. A Task C está treinando com ~1/6 da força que o número no
arquivo sugere — e a Task C é, de longe, a mais fraca do sistema: ICM-Norm hard **0,3436**
e ICM-Soft-Norm **0,1514** no checkpoint B15. Uma cabeça que quase não recebe gradiente e
que tem a pior métrica do sistema é o lugar mais óbvio para procurar folga.
O próprio código já registra isso como hipótese em `video_core.py:1338-1341` — não é
descoberta nova, é um item que nunca foi testado.
Note bem, e isto é importante: **a literatura de multi-task é majoritariamente cética**
com métodos sofisticados de balanceamento (pesos aleatórios empatam com 12 métodos SOTA;
em CelebA nenhuma estratégia bateu pesos uniformes fixos). Mas ela é **igualmente
explícita** de que um **pequeno grid manual sobre pesos fixos** é um baseline mais forte
do que se supõe. É exatamente isso que este item é.

**3. Onde mexe no código.**
Duas formas equivalentes — escolher **uma**:
- `video_core.py:1346-1348` — trocar `.mean(dim=-1)` por `.sum(dim=-1)` em `loss_task_c`.
- ou `video_core.py:274` — `LOSS_W_C = 0.7` → algo entre `1.5` e `4.2`
  (`0.7 × 6 = 4.2` reproduz exatamente o efeito do `.sum`).
A segunda é melhor: mantém a loss intacta e deixa o experimento como um **grid de uma
constante**, mais fácil de documentar e reverter. Sugestão de grid: `{0.7 (baseline), 1.5,
2.8, 4.2}`.

**4. Ganho esperado realista.**
**+0,02 a +0,06 em Task C**, e — condicionalmente — possivelmente mais em **Task C soft**,
porque 0,1514 é um piso anômalo e a cabeça mal foi treinada. Registre como **hipótese
explícita**, não como previsão.
⚠ **É uma REALOCAÇÃO, não ganho de graça.** Subir C vai muito provavelmente **custar** A
e/ou B. O documento final tem de reportar as seis métricas, nunca só o ΔC.

**5. Esforço/risco.**
**Trivial** no código (uma constante), **caro em GPU** (cada ponto do grid é uma run de
~3,5 h). Com 4 pontos, ~14 h. Alternativa mais barata: rodar o grid com os **backbones
congelados** (só fusão + cabeças treinam), que é a triagem OPT-09 já prevista, e levar
só o vencedor para uma run completa. ⚠ triagem congelada é *indicativa*, não conclusiva.

**6. Conflitos com regras do projeto.**
Nenhum de regra. Mas o `.mean → .sum` **muda o baseline silenciosamente** se for aplicado
sem braço de comparação — o próprio comentário do código avisa disso. Fazer como grid
explícito com o 0,7 dentro dele.

---

### Item 4 — Calibração pós-treino (temperature / vector scaling por task)

**1. O que é.**
Depois que o modelo termina de treinar, a gente não mexe mais nos pesos — só divide os
números crus que ele produz (os "logits") por uma constante `T` aprendida. `T > 1` deixa o
modelo menos confiante, `T < 1` mais confiante. É 1 parâmetro por task (ou 6 para a Task
C, um por categoria — isso se chama *vector scaling*). Ajusta-se `T` numa metade da
validação e reporta-se na outra.

**2. Por que pode ajudar AQUI.**
Porque metade das nossas seis métricas é **soft** (ICM-Soft-Norm), e soft mede o quanto a
distribuição de probabilidade do modelo bate com a distribuição de votos dos anotadores —
não só quem ganhou. Duas evidências apontam para desalinhamento de escala:
- **Task C soft = 0,1514 contra hard = 0,3436.** O modelo acerta razoavelmente quem é a
  categoria, mas a *magnitude* das probabilidades está longe das proporções de voto.
- **Task B é instável** (45,7 % de flips, 1 bit de `bf16` vira um argmax) — sinal clássico
  de logits mal escalados perto da fronteira.

**Verificação que fiz nesta sessão (importante para não vender ilusão):** o caminho soft
**não está quebrado**. `build_soft_label_c` (`video_core.py:690-710`) produz proporções de
voto por categoria, e `_metrics_task_c` (`video_core.py:1642-1649`) manda sigmoides
independentes no mesmo espaço — gold e predição são comensuráveis. Então **não há bug de
encanamento** a colher aqui; é calibração legítima, não conserto. Ver também item 9.

**3. Onde mexe no código.**
Não existe hoje. Função nova, no mesmo molde de `tune_task_c_thresholds`
(`video_core.py:1724`), que já implementa o protocolo honesto de "ajusta numa metade da
val, reporta na outra". Ela recebe os logits de `trainer.predict` e devolve
`{"T_a": …, "T_b": …, "T_c": [...]}` gravado em JSON. Os pontos onde `T` entraria na
avaliação são as três linhas que convertem logit em probabilidade:
`video_core.py:1593-1594` (A, sigmoide), `video_core.py:1614` (B, softmax) e
`video_core.py:1642` (C, sigmoide).

**4. Ganho esperado realista.**
**+0,01 a +0,04 nas métricas SOFT**, com a Task C soft como candidata a mais que isso
(hipótese, por causa do piso de 0,1514).
⚠ **Nas métricas HARD o ganho esperado é ZERO por construção**, nas três tasks — e vale a
pena entender por quê, porque isso evita um experimento desperdiçado:
- **A e B:** dividir todos os logits pela mesma constante não muda quem é o maior. O
  `argmax` é idêntico.
- **C:** aqui a decisão é por limiar, não por `argmax`, então *pareceria* que uma
  temperatura por classe (`vector scaling`) ajudaria. **Não ajuda:** comparar
  `sigmoid(logit_i / T_i)` contra o limiar `t_i` é a mesma coisa que comparar
  `sigmoid(logit_i)` contra um limiar reescalado — é uma reparametrização monótona da
  **mesma família** que o `_search_thresholds_task_c` (`video_core.py:1510`) já varre. Ou
  seja, no lado hard da C, o ajuste de limiar existente **já cobre** o que o vector scaling
  faria.
**Conclusão operacional:** calibrar **só para as métricas soft**, e não esperar (nem
reportar) movimento no lado hard.

**5. Esforço/risco.**
**Médio** (função nova + JSON de artefato), mas **custo ZERO de GPU de treino** — roda
sobre checkpoints que já existem. Melhor razão esforço/retorno do documento inteiro.
Risco documentado na literatura: temperature scaling degrada sob mudança de distribuição e
perde eficácia com classes muito sobrepostas.

**6. Conflitos com regras do projeto.**
Nenhum — desde que o ajuste use **metade da val e o report use a outra metade**,
exatamente como `tune_task_c_thresholds` já faz. Fazer numa val inteira seria o mesmo
vazamento que o `09_BASELINE_RESULTS.md` já marca como "regime otimista".

---

### Item 5 — Amostragem DENSA de frames (janela contígua dentro dos 64 já gravados)

**1. O que é.**
No disco há **64 frames por vídeo**, espalhados uniformemente do começo ao fim. O modelo
consome só 16, escolhidos com `linspace` sobre esses 64 — ou seja, 16 frames espalhados
pelo vídeo inteiro. A proposta é continuar consumindo 16, mas **contíguos**: uma janela de
16 frames consecutivos dos 64 (por exemplo, os índices 24–39), em vez de espalhados.

**2. Por que pode ajudar AQUI.**
O VideoMAE foi pré-treinado com clipes de ~2 segundos e `tubelet_size=2`, que **pareia
frames adjacentes** para captar movimento curto. Hoje o intervalo mediano entre frames
consecutivos é **1,21 s** — o tubelet vê duas imagens praticamente sem relação, e a
modelação temporal é degenerada por construção. Uma janela contígua dos 64 divide esse
intervalo por ~4 (mediana ~0,28 s), aproximando o regime em que o encoder foi treinado.
**Por que isso não está refutado pelos TESTES 6 e 7?** Porque os dois operam **depois** do
encoder: o 6 trocou o agregador sobre tokens fixos; o 7 trocou a adaptação sobre tokens
fixos. **Nenhum dos dois mudou o que o encoder VÊ.** O `perm_within ≈ −0,0002` é teto do
vetor `vid_feat` *deste checkpoint*, não da modalidade — o próprio doc 16 escreve isso.
Amostragem densa é a única intervenção barata que muda os tokens de entrada.

**3. Onde mexe no código.**
- `video_core.py:441-449` — `_subsample_uniform`. Acrescentar (não substituir) uma função
  `_subsample_dense(t, n_frames, center=…)` que fatia uma janela contígua.
- `video_core.py:452-481` — `load_frames`, que hoje chama `_subsample_uniform` na linha
  479. Passar a escolher a estratégia por uma constante nova (ex.:
  `FRAME_SAMPLING = "uniform" | "dense"`) perto de `video_core.py:244`.
- ⚠ Gravar a estratégia no `meta` do run (`video_core.py:1932` está perto do lugar certo)
  — sem isso, dois checkpoints ficam indistinguíveis.

**4. Ganho esperado realista.**
**+0,00 a +0,03, e essencialmente só na Task B.** Motivo do teto baixo: a contribuição
causal total do vídeo é −0,063 em **uma** métrica de seis. Mesmo um sucesso completo mexe
pouco no agregado.
⚠ E há uma crítica que sobrevive: janela densa **encolhe o intervalo absoluto mas NÃO
remove o confound de duração** — a razão p90/p10 continua 7,2×, porque a taxa de
amostragem continua sendo função da duração do vídeo. Remover isso exige amostragem a fps
fixo, ou seja, **re-extração** — fora do escopo desta rodada.
Enquadrar como **braço de teste de hipótese**, não como jogada de performance.

**5. Esforço/risco.**
**Médio.** Uma função nova + uma constante + o registro no meta. Risco: a janela contígua
vê **um quarto** do vídeo, então perde-se cobertura global. Escolher a janela central é o
default menos arbitrário.

**6. Conflitos com regras do projeto.**
Nenhum de regra. Mas obedece ao princípio de atribuição: **não juntar com o item 6.**

---

### Item 6 — Augmentação temporal (janela densa com deslocamento aleatório no treino)

**1. O que é.**
Mesma janela contígua do item 5, mas com o ponto de início **sorteado a cada época**
durante o treino (e fixo no centro na validação). Cada época o modelo vê um trecho
diferente do mesmo vídeo — é uma forma barata de "criar mais dados".

**2. Por que pode ajudar AQUI.**
Porque 2 019 amostras de treino é pouquíssimo e todo o resto deste documento aponta para
risco de overfitting. Os 64 frames no disco são um estoque de variação que hoje é
literalmente descartado (usamos sempre os mesmos 16 índices). É augmentação **de graça**,
sem tocar nos dados.

**3. Onde mexe no código.**
`video_core.py:768-780` — `EXISTVideoDataset.__getitem__`, que chama `load_frames` na
linha 771. É onde o offset aleatório entraria (o dataset saberia se é treino ou val).
⚠ Cuidado com determinismo: o projeto tem `use_deterministic_algorithms(True)` e
fingerprints SHA-256 dos pesos para comparar runs. Um sorteio no `__getitem__` **quebra a
reprodutibilidade bit a bit entre runs** a menos que use um gerador semeado por
`(SPLIT_SEED, epoch, idx)`. Especificar isso explicitamente antes de implementar.

**4. Ganho esperado realista.**
**+0,00 a +0,02**, provavelmente concentrado em estabilidade (menos variância entre seeds)
mais do que em métrica média.

**5. Esforço/risco.**
**Médio/alto** — não pelo código, mas pelo conflito com o regime de determinismo, que é
uma propriedade cara do projeto e conquistada com esforço.

**6. Conflitos com regras do projeto.**
**Sim, dois.** (a) Reprodutibilidade bit a bit — ver acima. (b) **Braço separado do item
5**, obrigatoriamente. Mudar amostragem e augmentação na mesma run reproduz exatamente a
não-atribuibilidade que o `16_...§11 ponto 3` rejeitou.

---

### Item 7 — Uncertainty weighting (pesos de loss aprendidos, Kendall et al.)

**1. O que é.**
Em vez de fixar os pesos das três tarefas à mão, deixa-se o próprio treino aprendê-los. O
truque: cria-se um parâmetro por tarefa que representa "quanto essa tarefa é ruidosa", e a
loss vira `Σ (1/2σ²)·loss_i + log σ_i`. Tarefas ruidosas ganham peso menor automaticamente.
São **3 parâmetros treináveis a mais** — o método mais barato dessa família.

**2. Por que pode ajudar AQUI.**
Porque as três tarefas do sistema têm níveis de dificuldade muito diferentes (A 0,55 · B
0,50 · C 0,34) e os pesos atuais nunca foram ajustados — são constantes herdadas. Se o
item 3 mostrar que o balanceamento importa, este é o passo seguinte natural: em vez de
mais um grid de runs de 3,5 h, uma única run que procura o balanço sozinha.

**3. Onde mexe no código.**
`video_core.py:1352-1359` — `compute_loss`, especificamente a linha 1359
(`total = LOSS_W_A*la + LOSS_W_B*lb + LOSS_W_C*lc`). Os 3 `log σ` precisam ser
`nn.Parameter` **dentro do modelo** (`EXISTVideoModel.__init__`,
`video_core.py:1005-1053`) para serem vistos pelo optimizer, e precisam de um grupo
próprio em `get_optimizer` (`video_core.py:1893-1896`) — provavelmente `weight_decay=0.0`,
como o `viewer_emb`.

**4. Ganho esperado realista.**
**+0,00 a +0,02.** Prior baixo, e digo por quê: a literatura recente é **cética**. Pesos
aleatórios empatam com 12 métodos SOTA em 5 datasets; quando o baseline é bem ajustado, o
ganho atribuído a métodos de MTL desaparece. E há crítica documentada específica ao
uncertainty weighting ("move erro de treino e de teste na mesma direção; não é bom
regularizador").

**5. Esforço/risco.**
**Médio** (parâmetros novos no modelo + grupo no optimizer + registro no state_dict).
Risco de estabilidade num dataset pequeno.

**6. Conflitos com regras do projeto.**
Muda o `state_dict`, então `load_state_dict(strict=True)` passa a distinguir checkpoints
com e sem o mecanismo — o que na verdade é **bom** (é o mesmo tipo de proteção que o
ARCH-20 dá às flags de ablação).

---

### Item 8 — Média sobre seeds (3 seeds de init, mesmo split)

**1. O que é.**
Rodar o mesmo treino 3 vezes mudando só a semente de inicialização (`INIT_SEED`) e (a)
reportar média ± amplitude em vez de um número, ou (b) **combinar** as três predições
(média das probabilidades) e reportar o conjunto.

**2. Por que pode ajudar AQUI.**
Porque a Task B é comprovadamente instável — 45,7 % de flips sob perturbação, e um bit de
`bf16` bastando para virar um argmax. Esse é o retrato exato de um sistema onde a média
sobre seeds ajuda. E o projeto já tem a infraestrutura certa: `SPLIT_SEED` e `INIT_SEED`
foram separados justamente para isso (OPT-06).

**3. Onde mexe no código.**
Nada estrutural. `video_core.py:84` (`INIT_SEED = 42`) já é parâmetro de
`build_model(init_seed=…)` (`video_core.py:1269`). A combinação das predições seria uma
função nova de análise, fora do caminho de treino.

**4. Ganho esperado realista.**
Para (a) **reportar média**: ganho de métrica **zero** — o ganho é de *credibilidade*, e é
grande para a tese. Para (b) **ensemble**: **+0,01 a +0,03**, tipicamente.

**5. Esforço/risco.**
**Trivial** no código, **3× o custo de GPU** (~10,5 h por configuração).

**6. Conflitos com regras do projeto.**
**Sim, e é sério.** O `CLAUDE.md` fixa o baseline como *"o estado FINAL da run"* com uma
**regra de simetria obrigatória**: se um lado é ensemble, o outro tem de ser também. Um
ensemble de 3 seeds **não é comparável** com um baseline de uma run só. Ou se ensembla os
dois lados, ou o ensemble entra como análise separada, nunca como "sistema corrigido".

---

### Item 9 — Investigar os NaN do PyEvALL (custo zero de GPU)

**1. O que é.**
Três métricas voltam **NaN** no checkpoint B15: `task_b_fmeasure_hard`,
`task_b_crossentropy_soft` e `task_c_fmeasure_hard`. A causa nunca foi identificada — o
doc 16 só refutou a explicação antiga ("classe nunca predita") e registrou "vem de outro
motivo, interno ao PyEvALL".

**2. Por que pode ajudar AQUI.**
Não é uma técnica de melhoria; é **higiene que gateia a leitura de todo o resto**. Se
houver algo torto no caminho até o PyEvALL, qualquer delta medido nos itens acima está
sendo lido através de um instrumento com defeito conhecido e não diagnosticado. E se
houver um bug de formatação, essa é a *única* categoria em que a literatura admite deltas
de 0,1+ — o que faria dele o item de maior retorno do documento.
**Ressalva honesta:** eu **verifiquei nesta sessão** que o caminho soft está estruturalmente
correto para A, B e C (gold e predição no mesmo espaço), então não estou prometendo um bug.
Estou dizendo que três NaN sem causa conhecida são um risco não quantificado.

**3. Onde mexe no código.**
`video_core.py:1403-1465` (`_format_for_pyevall`) e `video_core.py:1467`
(`_run_pyevall`) — inspeção, não edição, até saber a causa.

**4. Ganho esperado realista.**
**0 na maioria dos cenários; desconhecido e potencialmente grande num cenário de bug.**
É a única linha do documento onde eu escreveria "pode ser mais de +0,05" sem me sentir
desonesto — e ainda assim como *condicional*, não como previsão.

**5. Esforço/risco.**
**Baixo.** Roda em CPU sobre artefatos existentes. Algumas horas.

**6. Conflitos com regras do projeto.**
Nenhum.

---

### Item 10 — Trocar o encoder de vídeo (VideoMAE → CLIP sobre frames, pooled no tempo)

⚠ **Reclassificado nesta revisão.** Na primeira versão deste documento este item estava em
"Descartadas" com a razão "fora de escopo". Isso estava **errado por instrução minha, não
por evidência**: eu havia pedido ao agente que escrevia o documento para excluir qualquer
coisa que "só fizesse sentido combinado com re-extração", supondo que trocar de encoder
exigisse re-extrair pixels. **Não exige.** Os frames no disco (v2) são pixels **crus**,
`uint8`, `[64,3,224,224]` por vídeo. A normalização (subtrair média, dividir por desvio) só
acontece na hora de **carregar**, não na extração — confirmado agora em
`video_core.py:452-481` (`load_frames`): `t.float().div_(255.0).sub_(mean).div_(std)`, com
`mean`/`std` lidos de um JSON por diretório de extração, nunca hardcoded. Trocar de encoder
é trocar essas duas constantes e o backbone — **cabe nesta rodada**.

**1. O que é.**
Hoje o vídeo passa inteiro (16 frames como um clipe) pelo VideoMAE, que foi treinado para
**reconstruir pixels** que faltam de um clipe (like um "complete a imagem"). A proposta é
processar cada frame **individualmente** com o CLIP — o mesmo encoder de imagem que o
sistema de memes já usa com sucesso — e depois juntar os 16 vetores resultantes num só
(por média, ou pela mesma atenção que já existe hoje).

**2. Por que pode ajudar AQUI.**
O doc 16 fecha em H2: o que sobra para explicar o vídeo fraco é **capacidade absoluta +
objetivo de pré-treino**, não um defeito no pooling ou na adaptação (essas duas rotas já
foram fechadas por medição — TESTES 6 e 7). Trocar para CLIP mexe nas **duas** pernas de
H2 ao mesmo tempo:
- **Objetivo de pré-treino:** o CLIP foi pré-treinado para casar imagem com **descrição em
  texto** (contrastivo, 400 M pares) — os vetores que produz já nascem carregando
  significado. O VideoMAE foi pré-treinado para reconstruir pixels perdidos — medido no
  doc 16, os tokens que produz são **locais** (concentração 10,88× no bloco trocado), não
  semânticos. É a diferença entre "sabe o que a imagem *é*" e "sabe reconstruir a imagem".
- **Capacidade absoluta:** o sistema de memes já treina o CLIP **de verdade** — 4 camadas
  destravadas, full-rank, ~50,4 M parâmetros (`memes_core.py:489-494`) — e o ramo visual
  dele contribui. Aqui, hoje, o vídeo tem só 0,59 M treináveis (LoRA r=16 em 2 módulos).
  **Existe precedente dentro do próprio projeto, no mesmo tipo de dado (imagem), na mesma
  ordem de tamanho de dataset.** É a evidência mais forte de todo este documento — mais
  forte que qualquer coisa que a pesquisa na internet trouxe para o item 1.
- ⚠ **O que este item NÃO resolve:** perder a modelagem temporal por `tubelet`. O
  VideoMAE, mesmo esparso, pareia frames adjacentes; CLIP por frame trata cada imagem como
  isolada — a única temporalidade que sobra é a do pooling. Só vale a pena se o valor do
  vídeo estiver mesmo em "conteúdo de cada frame" e não em "movimento entre frames" — e o
  doc 16 já mediu que o sinal existente é de **classe de intenção/formato**, que é uma
  propriedade de frame único (como a pessoa está enquadrada, se fala pra câmera), não de
  movimento. Isso favorece a aposta.
- **`perm_within ≈ −0,0002` é teto do *vetor atual* (deste checkpoint), não da
  modalidade — o próprio doc 16 declara isso (§6).** Nada medido até agora refuta um
  encoder melhor.

**3. Onde mexe no código.**
Mudança de arquitetura de verdade, não um parâmetro:
- `video_core.py:249` — `VIDEO_DIM = 768` → **1024** (hidden do CLIP ViT-L, igual ao
  `vision_dim` de `memes_core.py`). Isso propaga para `fusion_input_dim`
  (`video_core.py:970`) e muda a largura da fusão — **quebra compatibilidade de state_dict
  com as 5 runs já treinadas**. Não é fine-tuning de um checkpoint existente, é um braço
  novo do zero.
- `video_core.py:1128-1129` — troca `self.videomae(pixel_values=frames).last_hidden_state`
  (um forward, todo o clipe) por: achatar `[B,16,3,224,224]` em `[B·16,3,224,224]`, rodar o
  CLIP (`CLIPVisionModel`, mesma classe de `memes_core.py:482-484`) frame a frame,
  reagrupar em `[B,16,1024]`, e então poolar no tempo — pode reaproveitar o
  `TemporalAttentionPool` já existente (`video_core.py:949-967`), só trocando `hidden_dim`.
- `_frames_norm_constants` (`video_core.py:430-438`) já lê mean/std de um JSON por
  diretório — só é preciso um `_extraction_meta.json` novo (ou um parâmetro) com as
  constantes do CLIP: `mean=(0.4815,0.4578,0.4082)`, `std=(0.2686,0.2613,0.2758)` (as
  mesmas do `memes_core.py`). **Não precisa tocar nos arquivos `.pt` de frames.**
- `build_backbones` (`video_core.py:1246`) — trocar `VideoMAEModel.from_pretrained(...)`
  por `CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")` (mesmo nome já
  usado em `memes_core.py:483`). `restore_videomae_qv_biases` (linha 1258) deixa de se
  aplicar — o CLIP não tem esse problema de nomenclatura de bias (ARCH-01 era específico
  do VideoMAE).
- **Decisão em aberto, não resolvida aqui:** quantos frames mandar pro CLIP. Rodar os 16
  atuais multiplica o custo de forward por ~16× em relação ao VideoMAE (que processa o
  clipe inteiro numa passada via `tubelet_size=2`). "Frames-chave" — um subconjunto menor
  (4-8), escolhido por amostragem uniforme dos 64 já gravados ou por um critério barato de
  diversidade entre frames — é o jeito de controlar esse custo. **Isso é uma sub-decisão de
  desenho que fica para depois, não decidida por este documento.**

**4. Ganho esperado realista.**
Categoria diferente do resto do documento — é troca de backbone, a **única** categoria
onde a literatura (item 6 da pesquisa) documenta deltas de 0,1+. Mas dois freios seguram a
expectativa: (a) a dependência causal *medida* do slot de vídeo hoje é −0,063 numa métrica
de seis — um encoder melhor pode **subir esse teto**, mas o teto de hoje não é evidência de
quanto ele sobe, é só o piso a partir do qual se parte; (b) perder a modelagem temporal é
uma perda real, não hipotética.
**`[HIPÓTESE]`: +0,02 a +0,10 na Task B** (onde o vídeo já mostrou ter algum sinal),
**quase nada esperado em A e C** (mesma lógica do resto do documento — o vídeo nunca
mostrou sinal lá). O canto superior dessa faixa (perto de 0,10) só é plausível se o
objetivo de pré-treino for de fato o gargalo principal, o que é a leitura mais provável do
H2 mas continua sendo leitura, não medição.

**5. Esforço/risco.**
**Alto.** Não é ajuste de hiperparâmetro — é uma classe de encoder nova, uma dimensão de
fusão nova, um branch de código novo em paralelo ao atual (nunca substituir o VideoMAE
"no lugar" sem manter o caminho antigo comparável), e uma run que não é comparável ao
baseline sem a mesma disciplina de atribuição que os 3 braços A/A'/B do doc 09 já usaram.
Custo de GPU desconhecido até medir — plausivelmente **maior por época** que o VideoMAE
mesmo com poucos frames-chave, porque o CLIP ViT-L é um encoder maior. Risco de
overfitting: real, mas com precedente favorável (o mesmo encoder, unfrozen, já funciona
com um dataset de ordem de grandeza parecida em `memes_core.py`).

**6. Conflitos com regras do projeto.**
Nenhum de regra. Mas **exige tratamento como braço de atribuição isolado**, no mesmo
molde do `A′/A/B` do doc 09 — nunca reportar como substituição direta do baseline sem essa
disciplina.

---

### Item 11 — ATUALIZADO: ARCH-04 já corrigido; resta questão de atenção ao padding (ver spec 19)

⚠ **Reescrito nesta revisão (2026-08-13).** A versão original deste item descrevia o ARCH-04
(diluição de áudio por padding do Whisper) como um bug **ainda aberto**, citando os números
de população do `CLAUDE.md`/doc 07 (34,5 %/41,8 % do vetor de 1500 quadros é silêncio) como
se fossem o estado atual do sistema. **Isso estava desatualizado.** Nesta sessão, lendo
`EXIST_VIDEOS/extract_audio_v2.ipynb` diretamente (não só a documentação sobre ele),
descobrimos que o ARCH-04 **já foi corrigido em v2**: a extração já aplica máscara de
padding (A1) + média ponderada por janela de 30s (A2), e o checkpoint que serve de baseline
para esta rodada (`run_B2_bio_viewer_seed42_B15/`) já treina sobre esse áudio v2 —
`meta.json: "audio_version": "v2"` `[VERIFICADO — meta.json de run_B2_bio_viewer_seed42_B15]`.
Os números 34,5 %/41,8 % descrevem o problema da extração **v1**, já resolvido; não são o
estado do sistema hoje.

**Por que isso aconteceu:** o item 11 original foi escrito lendo o `CLAUDE.md` e o doc 07
(`07_ARCH04_FULL_POPULATION.md`) como fonte da verdade sobre o estado do áudio, sem checar o
notebook de extração real que roda hoje. O `CLAUDE.md` documenta a *história* do ARCH-04
(inclui a medição v1 que motivou a correção), não necessariamente o *estado atual* pós-v2 —
a correção acabou registrada no notebook `extract_audio_v2.ipynb` e no `meta.json` dos
checkpoints, mas não foi propagada de volta para uma atualização explícita deste documento
até agora.

**O que a v2 já faz** (`extract_audio_v2.ipynb`, célula `extract_whisper_features_v2`):
- **A1 (máscara do padding):** dentro de cada janela de 30s, a média usa só os
  `n_valid = round(segundos_válidos × 50)` primeiros quadros do encoder — o resto (padding)
  fica fora da média.
- **A2 (janelas múltiplas + média ponderada):** o áudio inteiro é percorrido em janelas de
  30s e a agregação final é a média das janelas ponderada por `n_valid` — matematicamente
  igual à média sobre todos os frames válidos do áudio inteiro, não uma heurística.

**Ressalva residual real e não resolvida:** o próprio comentário do notebook v2
declara isso explicitamente — a **auto-atenção** do encoder do Whisper, dentro de cada
janela de 30s, ainda **processa o padding internamente**; só a média final o ignora. Ou
seja, A1+A2 removem a diluição do vetor agregado, mas não removem o padding do que o
encoder efetivamente "viu" durante a auto-atenção.
**Contra-argumento a levar a sério, registrado no pré-registro do probe:** o Whisper foi
**treinado** com padding de 30s fixo — processar padding pode ser regime *in-distribution*
para o encoder, não contaminação. Não há como saber a priori se essa auto-atenção sobre
silêncio distorce as representações dos quadros válidos ou se é inócua por ser exatamente o
regime de pré-treino do modelo.

**Isto é diagnóstico, não correção.** Nenhuma re-extração v3 de áudio está decidida nesta
rodada. A questão de se essa ressalva residual vale a pena investigar mais — e, se valer, em
que direção — é o objeto do probe pré-registrado em
`EXIST_VIDEOS/improvements/19_WHISPER_PADDING_PROBE_SPEC.md` (escrito em paralelo a este
documento por outro agente desta sessão; conteúdo não antecipado aqui). O probe compara, sob
leitura pré-registrada e assimétrica, uma variante que evita processar o padding dentro da
auto-atenção contra o v2 atual, sobre uma sonda linear diagnóstica — sem gerar nenhum
`.pt` novo em árvore de produção.

**Consequência prática para o resto deste documento:** o item 11 deixa de ser "o item de
maior teto do documento inteiro" (como a versão anterior desta seção e o §5 afirmavam) — essa
caracterização dependia dos números v1 como estado atual. Com a diluição já corrigida, não há
mais uma categoria de "bug de pipeline ainda ativo" a resgatar aqui; o que resta é uma
pergunta de pesquisa mais estreita e incerta (auto-atenção sobre padding), cuja resposta
começa pelo probe do doc 19, não por uma run de re-treino cara.

---

### Item 12 — Amostragem de frames a taxa fixa (fps constante) `[REQUER RE-EXTRAÇÃO DE FRAMES]`

**1. O que é.**
Hoje os 16 frames usados no treino são escolhidos por posição relativa no vídeo (16
espalhados uniformemente entre o início e o fim, dos 64 guardados). Isso faz o intervalo
entre frames **depender da duração do vídeo**: quanto mais longo, maior o intervalo. A
proposta é amostrar a uma **taxa fixa de quadros por segundo**, igual para todo vídeo.

**2. Por que pode ajudar AQUI — e por que é diferente do item 5 (janela densa).**
O item 5 (já incluído, sem re-extração) encurta o intervalo médio mas **não remove** o
confound: a razão entre o vídeo mais longo e o mais curto do dataset continua 7,2× em
qualquer contagem de frames, porque a fonte do problema é a *duração variar* e a
amostragem continuar proporcional a ela. **Só fps fixo remove isso na raiz** — e o doc 16
mediu que esse é exatamente o confound que aparece igual com e sem LoRA (pré-treinado, não
aprendido): +0,297 de AUC−nulo pra prever duração, idêntico ligado/desligado o adaptador.
Isso significa que nenhuma correção de re-treino chega nele — só mudar o que entra no
encoder resolve.

**3. Onde mexe no código/pipeline.**
Extração de frames, não `video_core.py`. Decisão de desenho que precisa ser tomada **antes**
de escrever a spec (não decidida aqui):
- **Janela fixa** (ex.: 8s a 4fps = 32 frames): remove o confound de verdade, mas descarta
  o resto de vídeos longos — um vídeo de 60s perde 52s de conteúdo potencialmente
  relevante.
- **fps fixo + padding com máscara**: cobre o vídeo inteiro, mas reintroduz comprimento
  variável e exige que o encoder saiba ignorar o padding — a mesma doença do item 11 se
  malfeito.
A primeira é a que resolve o confound; a segunda o transforma em "quanto de padding" —
que é literalmente o problema do item 11, replicado no vídeo.

**4. Ganho esperado realista.**
**`[HIPÓTESE]` +0,01 a +0,04 na Task B**, mesma ordem do item 5, porque o teto causal
medido do vídeo continua sendo −0,063 numa métrica — remover o confound não cria sinal que
não existe, só limpa o sinal que já existe. **Aqui não há categoria de "bug" a resgatar**
(nem havia no item 11 pelo áudio — ver revisão do item 11 acima: a diluição por padding do
áudio já estava corrigida em v2 antes desta rodada) — é limpeza de confound sobre um sinal
que já está mapeado como fraco. Não esperar salto grande só por causa da categoria
"re-extração".

**5. Esforço/risco.**
**Alto.** Escolher janela fixa vs. fps+máscara é decisão de arquitetura, não só de
extração; muda `NUM_FRAMES`/`FRAME_SAMPLING` e potencialmente o `tubelet_size` efetivo.
~1 dia de GPU pra re-extrair (mesma ordem da v2), e exige `frames_v2_sample/` preservado
antes de mexer (já existe, `frames_v1_sample/` está preservado; confirmar se
`frames_v2_sample/` foi criado antes desta rodada — se não, criar primeiro).

**6. Conflitos com regras do projeto.**
Nenhum de regra, mas **decisão do autor obrigatória antes de qualquer código**: qual das
duas variantes (janela fixa ou fps+máscara). Já foi levantada e não resolvida numa sessão
anterior.

---

## 3. COMBINAÇÕES

Regra que vale para as três: **o ganho combinado NÃO é a soma dos itens.** A literatura de
multi-task mostra explicitamente que ganhos não são aditivos, e que métodos sofisticados
empatam com um baseline bem ajustado. As faixas abaixo são **hipóteses**, e a faixa
realista é **+0,03 a +0,08 mesmo combinando** — salvo a exceção condicional já nomeada.

---

### Combo A — "Alimentar o ramo que carrega o sinal" (recomendado)

**Itens:** 1 (LoRA no texto) + 2 (`MAX_LENGTH` 384) + 3 (peso efetivo da Task C) + 4
(calibração pós-treino).

**Mecanismo de por que fazem sentido JUNTOS neste sistema — não é genérico:**
os quatro atacam pontos diferentes da mesma cadeia. O item 2 faz **chegar** mais texto ao
encoder (hoje 22 % é cortado); o item 1 dá ao encoder **capacidade de transformar** esse
texto (hoje 226× menor que no sistema de memes, que funciona); o item 3 garante que a
cabeça C consiga **usar** o que chega (hoje ela treina com 1/6 da força documentada); e o
item 4 converte, no fim, distribuições melhores em **ICM-Soft**, que é metade das nossas
métricas. É uma cadeia, não uma pilha: mais sinal → mais capacidade de processá-lo → mais
gradiente para a cabeça mais fraca → melhor aproveitamento na métrica soft.

**Ganho combinado — HIPÓTESE:** **+0,03 a +0,08** agregado, mais provável em A e B (via
itens 1-2) e em C (via itens 3-4). ⚠ Lembrar que o item 3 é **realocação**: parte do ΔC
sai do bolso de A/B.

**Custo:** o item 4 é grátis em GPU. Os itens 1-3 valem ~3 a 5 runs (10-18 h) se cada
degrau for isolado.

**Ordem sugerida (cada linha é uma run, para ficar atribuível):**
1. Item 2 sozinho (é uma constante; barato de descartar se der OOM).
2. Item 1, degrau 1: `target_modules` do texto incluindo MLP, `r` mantido em 16.
3. Item 3: grid de `LOSS_W_C`, de preferência na triagem com backbones congelados.
4. Item 4 sobre o melhor checkpoint. Sem custo de treino.
5. Só se 2 der ganho: item 1, degrau 2 (`r=32` no texto).

---

### Combo B — "Vídeo em regime menos esparso" (braço de hipótese, não jogada de performance)

**Itens:** 5 (janela densa) — e o 6 (offset aleatório) **como run separada, nunca junto**.

**Mecanismo:** é o único par de intervenções que muda **o que o encoder de vídeo vê**,
enquanto todas as rotas já fechadas (TESTE 6 pooling, TESTE 7 LoRA de vídeo,
cross-attention, concatenação de tokens) mexiam apenas **depois** do encoder. Se o vídeo
tem mais a dar, é por aqui ou por troca de encoder — e troca de encoder está fora de
escopo.

**Ganho combinado — HIPÓTESE:** **+0,00 a +0,03**, e essencialmente só na Task B. O teto é
imposto por medição, não por pessimismo: a dependência causal total do slot de vídeo é
−0,063 numa métrica de seis.

**Valor real deste combo não é a métrica** — é fechar cientificamente a última rota barata
do ramo de vídeo. Se não der nada, isso vira **achado negativo publicável**, com a mesma
força do resto do doc 16. É defensável gastar 2 runs nisso pela tese, não pelo placar.

**Custo:** 2 runs (~7 h).

---

### Combo C — "Pós-treino, zero GPU de treino"

**Itens:** 4 (calibração) + 9 (NaN do PyEvALL) + o `tune_task_c_thresholds` **que já
existe**.

**Mecanismo:** todos operam sobre checkpoints já treinados. O item 9 garante que o
instrumento de medição está íntegro; o item 4 realinha as probabilidades com as
proporções de voto que o ICM-Soft cobra; e o ajuste de limiar da Task C já converte
probabilidades melhores em decisões melhores (`video_core.py:1724`, com o protocolo
honesto de metade/metade).

**Ganho combinado — HIPÓTESE:** **+0,01 a +0,04**, concentrado nas métricas soft.
⚠ **O ajuste de limiar NÃO conta como ganho novo** — ele já está implementado e já está
contabilizado no baseline. O número 0,3784 → 0,4267 do `09_BASELINE_RESULTS.md` está
marcado lá mesmo como *"mesmo val — regime otimista, não comparar"*; importá-lo aqui como
crédito seria importar um vazamento para dentro das estimativas.

**Custo:** algumas horas de CPU. **É o melhor retorno por hora de trabalho do documento
inteiro**, ainda que não seja o maior ganho absoluto.

---

### Combo D — "Capacidade nos dois ramos com sinal" (maior teto, maior custo/risco)

**Itens:** 1 (LoRA no texto) + 10 (CLIP no vídeo).

**Mecanismo:** os dois ramos que sobrevivem à investigação — texto (sinal provado, −0,25) e
vídeo (sinal fraco mas real dentro da classe de B) — hoje têm capacidade de adaptação muito
abaixo do precedente que já funciona no sistema de memes (XLM-R inteiro; CLIP com 4
camadas destravadas). Este combo não é "somar dois números" — é a hipótese de que o
sistema atual está capacidade-limitado nos dois ramos que carregam sinal, não só num.

**Ganho combinado — HIPÓTESE, não soma:** **+0,03 a +0,12**, concentrado em A e B. A parte
de vídeo (item 10) domina a incerteza — é o item de maior variância de todo o documento,
pra cima e pra baixo.

**Por que NÃO é o primeiro combo a rodar, apesar do teto maior:** custo e risco de
atribuição. Trocar dois ramos na mesma run torna o resultado não separável (qual dos dois
causou o quê); e o item 10 sozinho já é uma run cara e de resultado incerto. **Ordem
recomendada:** validar o item 10 isolado primeiro (é a incógnita), só depois combinar com o
item 1 (que já tem base de evidência própria mais sólida).

**Custo:** 3-4 runs (item 10 isolado, item 1 isolado se ainda não rodado, e a combinação) —
ordem de 12-16 h.

---

## 4. DESCARTADAS

| técnica | por que ficou de fora |
|---|---|
| **Pooling alternativo no vídeo** (GMAP, GeM, regularização de entropia, top-k) | TESTE 6 mediu: `mean_all`, `mean_temporal` e `top5` empatam com o pool aprendido dentro da tolerância. Nenhum operador cria informação que os tokens não carregam. |
| **Mais LoRA no VÍDEO** (r=64, MLP no VideoMAE) | TESTE 7: o **primeiro** incremento de adaptação já rende abaixo do ruído (+0,004/+0,005). Aumentar é apostar contra uma medição. |
| **Cross-attention sobre os 1 568 tokens** | Rota fechada no doc 16: sem sinal na cauda dos tokens, e custaria ~852 k parâmetros num dataset de 2 019 amostras — o mesmo risco que matou a versão anterior. |
| **OGM / OGM-GE, Gradient Blending** (rebalancear gradiente por modalidade) | Os ganhos grandes ocorrem quando a modalidade fraca é **genuinamente informativa mas abafada**. Aqui ela é quase não-informativa (0,063 em B, ~0 em A/C) — não há o que redistribuir. |
| **Gated fusion / GMU** (gate aprendido sobre as modalidades) | Um gate é função do slot; num braço de ablação que zera uma modalidade, ele redistribui massa e reintroduz o mesmo tipo de confound que a proibição de LayerNorm pós-concat existe para evitar. Conflito direto com a regra do projeto. |
| **LayerNorm/BatchNorm sobre o vetor concatenado** | Proibido por regra (`video_core.py:1042-1046`). |
| **GradNorm, PCGrad** | Exigem gradientes por tarefa (backward extra) — custo real num orçamento de 1 GPU e 3,5 h por run, sem evidência de ganho em datasets pequenos. |
| **Focal loss / Class-Balanced Loss na Task B** | O desbalanceamento é 2,9:1 (268/92/145) — **leve** para os padrões dessa literatura, que mira 100:1. E há aviso explícito de que focal loss desestabiliza treino em datasets muito pequenos. |
| **Label smoothing na Task B** | **Redundante:** `loss_task_b` (`video_core.py:1323-1330`) já treina contra a **distribuição de votos dos anotadores**, que já é suave. Não é o remédio para os 45,7 % de flips. |
| **32 ou 64 frames no encoder, MANTENDO amostragem proporcional à duração** | A razão p90/p10 do intervalo continua 7,2× para 16, 32 ou 64 frames enquanto a amostragem for proporcional à duração — mais frames sozinho não remove o confound. **Ver item 12** para a variante que resolve isso (fps fixo, exige re-extração, agora em escopo). |
| **Ajuste de limiar da Task C como "nova melhoria"** | **Já implementado** (`video_core.py:1724`) e já contabilizado no baseline. Listá-lo como ganho seria contar duas vezes, e com um número que a própria fonte marca como otimista. |
| **Normalização por ramo em vídeo/texto/bio** (LayerNorm antes do concat) | Permitida por regra, mas o bio já teve o dela removida deliberadamente (ARCH-08, `video_core.py:1023-1030`) com justificativa medida. Sem hipótese própria, seria mexer por mexer. |

---

## 5. Resumo executivo

- **Alvo de +0,2 numa mudança isolada: sem precedente na literatura consultada** para
  técnicas de loss/pooling/fusão. A única categoria onde a literatura documenta deltas
  dessa ordem é **troca de backbone** — e o item 10 (abaixo) é exatamente essa categoria.
  Alvo defensável para o resto do documento: +0,02–0,05 isolado, +0,03–0,08 combinado.
- **Duas apostas de perfil diferente, não uma só:**
  - **Maior teto, maior risco: item 10** (trocar VideoMAE por CLIP no vídeo). Reclassificado
    nesta revisão — a versão anterior deste documento o descartava por "fora de escopo",
    o que estava errado: os frames já são pixels crus no disco, trocar de encoder não pede
    re-extração. É a única mudança que ataca as duas pernas do H2 do doc 16 (capacidade
    absoluta + objetivo de pré-treino) ao mesmo tempo, e tem precedente **dentro do próprio
    projeto** (o CLIP já funciona no modelo de memes, unfrozen, dataset de ordem parecida).
    Custo e incerteza também são os maiores do documento — é a run que resolve a maior
    dúvida em aberto, não a mais segura.
  - **Maior confiança, menor teto: item 1** (LoRA no texto). É o único ramo onde a nossa
    própria sonda provou que o sinal mora (−0,25 contra ~0 do vídeo hoje) e onde o sistema
    de memes investe 226× mais capacidade. Ganho esperado menor, mas evidência mais direta.
- **A aposta de melhor retorno por hora de trabalho continua o Combo C** (calibração + NaN),
  porque não custa GPU de treino.
- **O item mais barato de todos com efeito plausível é o item 3** (peso efetivo da Task C:
  0,117 em vez dos 0,7 documentados) — uma constante, mas com custo de GPU no grid, e é
  **realocação**, não ganho de graça.
- **A conclusão do doc 16 sobre o vídeo atual não muda** — ela é sobre o checkpoint medido,
  não sobre o teto da modalidade. `perm_within ≈ 0` é teto do *vetor deste checkpoint*; o
  doc 16 já registra isso. O item 10 é a forma de testar se um vetor diferente muda o
  quadro. O Combo B continua existindo para fechar a última rota barata dentro do
  VideoMAE atual, caso o item 10 não seja perseguido.
- **Você removeu a restrição de "sem re-extração".** Isso abriu espaço para o **item 11**
  (ARCH-04, diluição de áudio por padding). **Atualização desta revisão (2026-08-13):** o
  item 11 deixou de ser um item de re-treino em aberto — descobrimos, lendo
  `extract_audio_v2.ipynb` diretamente, que a diluição **já estava corrigida em v2** antes
  desta rodada (máscara de padding + média ponderada por janela; o baseline B15 já treina
  sobre esse áudio). A caracterização anterior deste resumo ("item de maior teto do
  documento inteiro", com precedente no ARCH-01) descrevia o problema **v1**, não o estado
  atual — ver a seção "Item 11" acima para a correção completa. O que resta do item 11 é uma
  ressalva mais estreita (auto-atenção do Whisper ainda processa padding dentro da janela de
  30s, só a média final o ignora), tratada como **diagnóstico**, não correção, via o probe
  pré-registrado em `19_WHISPER_PADDING_PROBE_SPEC.md` — nenhuma re-extração v3 está
  decidida. O **item 12** (fps fixo no vídeo) continua de pé como descrito: remove de vez o
  confound de duração que o item 5 só atenua, mas não tem categoria de "bug" a resgatar —
  não esperar o mesmo salto que se esperava do item 11 na leitura antiga.
