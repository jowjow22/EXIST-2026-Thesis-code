# 21 — Pré-registro: probing das 3 heads sobre trunk congelado (variação, não substituição)

**Escrito em:** 2026-08-15, **antes** de qualquer forward de probe rodar.
**Formato:** pré-registro de protocolo — decisões travadas aqui não podem mudar depois de
ver as curvas de nenhum dos dois trunks (CLIP ou B15). Se algo aqui se mostrar errado, o
documento é corrigido e a correção é datada — nunca a decisão é "ajustada silenciosamente"
depois do resultado.
**Origem:** opção 4 do `20_EPOCH_SELECTION_TASK_C.md`, generalizada das três heads (A/B/C),
por decisão do autor — validada com o advisor antes deste documento.
**Nada de código foi escrito ainda.** Este documento vem primeiro.

---

## 0. O que isto é, e o que NÃO é

**É:** uma **variação** do braço CLIP (e do B15, obrigatoriamente, ver §2) — trunk
congelado no estado oficial já salvo, três heads (`head_a`, `head_b`, `head_c`, cada uma
uma `nn.Linear` sobre `FUSION_HIDDEN_DIM`) retreinadas **independentemente**, cada uma
parando pela **própria** loss de validação. Linear probing clássico (Alain & Bengio 2016,
arxiv 1610.01644), aplicado às três heads de forma uniforme — não só à C.

**NÃO é:** uma reabertura da seleção de época. O trunk fica exatamente no estado oficial
já decidido (CLIP: estado FINAL da época 15, `run_..._CLIP`; B15: o estado oficial daquele
braço, o que já está salvo em `run_..._B15`). Nenhum peso do trunk muda.

**NÃO é:** o "checkpoint Frankenstein" (enxerto trunk-de-uma-época + head-de-outra) que o
doc 20 §3 já vetou por falta de fonte. Aqui só existe UM trunk por sistema, fixo; só as
heads mudam.

**Por que uniforme (A+B+C) e não só C:** se só a C fosse probada, o motivo de escolha seria
"essa é a task com número ruim" — a métrica reportada decidindo onde intervir, o mesmo
mecanismo de leakage que a regra de seleção de época já proíbe, só que aplicado à escolha
de ONDE mexer em vez de QUAL época escolher. Aplicando às três, o protocolo é simétrico e
a decisão de intervir não depende de ter visto o resultado de nenhuma head individualmente.

---

## 1. Protocolo, travado antes de ver qualquer curva

### 1.1 Extração de features

- **Treino (2.019 amostras):** forward em `eval()` (sem dropout) sobre `train_dataset`,
  hook em `model.fusion[0]` (mesmo ponto de captura já usado nos probes causais) —
  **extração nova**, o cache de `fusion_in` que já existe (`probe_cache_*/val_cache.pt`)
  é só da val (505) e não serve para treinar a probe.
- **Val (505):** reusa o cache já existente por `RUN_DIR`, se presente; senão, mesma
  extração.
- **Caveat declarado, não corrigido:** a head original viu features com dropout ATIVO
  durante o treino end-to-end; a probe treina sobre features em `eval()` (dropout =
  identidade). Distribuição de entrada diferente da que gerou a head original — compensado
  por weight decay + early stopping na probe, nunca por reintroduzir dropout na extração
  (isso tornaria a extração não-determinística e complicaria o cache).

### 1.2 Arquitetura e init da probe

- Mesma arquitetura da head original (`nn.Linear(FUSION_HIDDEN_DIM, out_dim)` — confirmar
  `out_dim` por head: A=1, B=3, C=6, olhando `video_core.py`, não assumir).
- Init: `vc.set_all_seeds(INIT_SEED)` re-chamado imediatamente antes de cada `nn.Linear`
  novo — mesmo protocolo de determinismo do resto do projeto. As três heads (A, B, C) são
  inicializadas em sequência, sempre com re-seed imediatamente antes de cada uma, para que
  a ordem de instanciação não afete o RNG consumido por cada head.
- Otimizador: `AdamW`, mesmo LR base do treino original (`vc.get_optimizer`/`get_optimizer_clip`
  usam grupos — para a probe, LR único, valor a definir por um smoke test rápido, não por
  grid buscando a melhor métrica).

### 1.3 Critério de parada — decisão travada, não a mais purista

**Parar pela loss da própria head, computada sobre o VAL INTEIRO** (as mesmas 505
amostras que depois reportam a métrica ICM) — não um holdout separado do treino.

Justificativa (nota 3 do advisor, aceita e registrada): o `E = argmin(eval_loss)` que o
projeto já usa para selecionar época também usa o val inteiro tanto para decidir quanto
para depois reportar ICM no mesmo val — mesmo nível de reuso de informação. A variante mais
estrita (separar um holdout do treino só para a parada, mantendo o val intocado até o
relatório final) existe e é mais defensável em teoria, mas **não foi escolhida** porque
decidir isso DEPOIS de ver que a variante simples dá números bons seria exatamente a
armadilha pós-hoc do doc 20 §6. Está travado aqui como a escolha, ponto.

### 1.4 O que separa "funcionou" de "não funcionou" — definido ANTES de rodar

- **Probes A e B são controle, não candidatas a ganho.** O trunk foi otimizado sob uma loss
  conjunta dominada aritmeticamente por A e B (doc 20 §0.1a) — as heads A/B originais já
  tiveram bastante sinal. Expectativa: **A_probe ≈ A_original, B_probe ≈ B_original**
  (dentro de ruído). **Se A ou B inflar muito em relação ao original, é motivo de
  desconfiar do protocolo (bug de extração, mismatch de split, etc.), não de comemorar.**
- **Probe-C é o discriminador.** Duas leituras possíveis, sem meio-termo:
  - **C_probe ≫ C_original:** a head C estava sub-treinada dentro do treino conjunto — a
    representação do trunk já continha sinal suficiente, só a head não teve gradiente
    dedicado o bastante. Ganho barato, sem retreinar o trunk. **Caso fechado nessa
    direção.**
  - **C_probe ≈ C_original:** a representação CONGELADA do trunk não carrega sinal de C
    suficiente — nenhum ajuste de head vai resolver isso. Só resta a opção 5 do doc 20
    (repesar `LOSS_W_C` no treino, retreino completo do trunk).

---

## 2. Condição bloqueante: rodar em AMBOS os trunks (CLIP e B15)

Se o braço CLIP virar "trunk congelado + 3 heads probadas" e o B15 continuar reportado só
como "end-to-end conjunto", a comparação entre os dois deixa de fazer sentido — mesma regra
de simetria que o doc 20 já impõe para a opção 3 ("se o corrigido é selecionado por task, o
baseline é reportado por task também", aqui adaptado para "se o corrigido é probado por
head, o baseline é probado por head também").

**Consequência prática:** o mesmo código de extração + probe roda duas vezes, uma vez sobre
`run_..._CLIP` (trunk CLIP-ViT-L/14, 4 camadas, estado final época 15) e uma vez sobre
`run_..._B15` (trunk VideoMAE, estado oficial daquele braço). Sem os dois, nenhum número
de nenhum dos dois lados entra em relatório comparativo.

---

## 3. Estado do sistema depois disto — declaração explícita, não escondida

Isto é a **3ª definição de "o que é o braço CLIP"** nesta investigação (argmin → final →
final+heads-probadas). Cada braço passa a ter **dois números reportáveis**, nunca só um
escondendo o outro:

1. **End-to-end conjunto** (já reportado, `final_metrics.json` de cada `RUN_DIR`) — o
   sistema como foi treinado, uma única passada, sem pós-processamento de head.
2. **Trunk congelado + heads probadas** (esta variação) — mesmo trunk, heads
   re-otimizadas independentemente pós-treino.

Os dois ficam lado a lado na tese/paper, rotulados como tal. **Números da variação 2 sobre
o braço CLIP são marcados POST-HOC** na mesma linha em que aparecem (mesmo padrão do
`09_BASELINE_RESULTS.md`, "regime otimista, não comparar com meia-val") — porque a decisão
de tentar esta variação só veio depois de já ter visto a Task C do CLIP ficar atrás do
B15. Para o B15, se a variação 2 for aplicada também a ele, os números são de primeira
inspeção nesse braço (não pós-hoc) — mas só se o B15 nunca tiver sido examinado com essa
lente antes deste documento.

---

## 4. Depois disto — calibração (opção 6) NUNCA antes

Se, e só se, os números da variação 2 (heads probadas) forem adotados como o estado
reportado de algum braço, **então** a opção 6 do doc 20 (calibração/temperature scaling
pós-treino nas métricas soft) pode ser aplicada por cima da head que ficou oficial — nunca
antes, e nunca misturada com a decisão de qual head usar (senão dois ajustes pós-hoc
acontecem ao mesmo tempo e fica impossível atribuir o ganho a qual dos dois).

---

## 5. Ordem de execução

1. Confirmar `FUSION_HIDDEN_DIM` e `out_dim` de cada head lendo `video_core.py` — não
   assumir.
2. Extrair `fusion_in` do TREINO para os dois trunks (CLIP-final, B15-oficial) — extração
   nova, hook em `model.fusion[0]`, `eval()`, sem dropout.
3. Treinar as 3 probes (A, B, C) sobre cada trunk, parada pela própria loss no val inteiro
   (§1.3), com o smoke test de LR antes de qualquer treino "de verdade".
4. Reportar tabela: `{trunk} × {head} × {original vs probe}` nas 6 métricas ICM-Norm.
5. Aplicar a leitura pré-registada do §1.4 — sem reformular depois de ver os números.
6. Só depois: decisão do autor sobre adotar a variação 2 como oficial de algum braço.

---

## 6. Resultado (2026-08-15) `[VERIFICADO]`

Protocolo executado sobre os dois trunks (`run_..._CLIP`, 4 camadas, estado final época
15; `run_..._B15`, oficial) — extração nova de `fused` (saída de `model.fusion`, 1024-d,
correção de um bug de hook que capturava `fusion_in` pré-fusão, 2944-d, por engano),
smoke test de LR por convergência de treino (`PROBE_LR=1e-3`), 6 probes treinadas
(3 heads × 2 trunks, 300 épocas full-batch cada, `argmin` da própria BCE val loss).

**Tabela — original vs probe, isolando cada head:**

| trunk | variante | A hard | A soft | B hard | B soft | C hard | C soft |
|---|---|---|---|---|---|---|---|
| CLIP | original | 0,5741 | 0,5172 | 0,4986 | 0,4122 | 0,3640 | 0,1368 |
| CLIP | probe_a | 0,4953 | 0,3890 | 0,4986 | 0,4122 | 0,3640 | 0,1368 |
| CLIP | probe_b | 0,5741 | 0,5172 | 0,4604 | 0,2780 | 0,3640 | 0,1368 |
| CLIP | probe_c | 0,5741 | 0,5172 | 0,4986 | 0,4122 | 0,3108 | 0,0000 |
| B15 | original | 0,5536 | 0,5058 | 0,4968 | 0,3947 | 0,3436 | 0,1514 |
| B15 | probe_a | 0,5588 | 0,4298 | 0,4968 | 0,3947 | 0,3436 | 0,1514 |
| B15 | probe_b | 0,5536 | 0,5058 | 0,4493 | 0,2767 | 0,3436 | 0,1514 |
| B15 | probe_c | 0,5536 | 0,5058 | 0,4986 | 0,4122 | 0,3120 | 0,0000 |

`best_epoch` de cada probe (argmin da própria BCE val loss, orçamento de 300 épocas):
CLIP → head_a=5, head_b=3, head_c=11. B15 → head_a=6, head_b=5, head_c=15.

### 6.1 Veredito: INCONCLUSIVO por falha de controle

Os controles A/B **pioraram nos 4 casos** (CLIP-A, CLIP-B, B15-A, B15-B) — nunca
empataram, sempre na direção errada, mais forte no soft que no hard (ex.: CLIP-A soft
0,5172→0,3890, Δ=−0,1282; CLIP-B soft 0,4122→0,2780, Δ=−0,1342). `task_c_icmsoftnorm_soft`
caiu para exatamente **0,000000** nos dois trunks.

Pelo gate pré-registado no §1.4 deste documento (controles são checagem de sanidade, não
candidatos a ganho — se falharem, desconfiar do protocolo, não do conteúdo), **o veredito
sobre C que o notebook chegou a calcular ("representação sem sinal") NÃO É VÁLIDO** e não
deve ser citado em nenhum lugar como conclusão. Não é "C sem sinal". Não é "C
sub-treinada". É falha do instrumento de medição. A célula de leitura do
`probe_head_training.ipynb` foi corrigida para checar isso automaticamente (gate
`controls_ok`) antes de imprimir qualquer veredito sobre C — impede que uma reprodução
futura reimprima a conclusão inválida.

### 6.2 Diagnóstico do mecanismo `[HIPÓTESE]`

Dois pontos, nenhum verificado além do raciocínio sobre os números:

1. **`C_soft=0,000000` idêntico nos dois trunks é consistente com o clip do
   ICM-Soft-Norm** (ICM cru negativo → normalizado clipa em 0). Duas probes
   independentes, dois trunks com pesos totalmente diferentes, ambas afundando abaixo do
   floor — mais coerente com "ambas ficaram ruins o bastante para bater no piso" do que
   com um bug determinístico. **Não confirmado**: checar `eval_task_c_icmsoft_soft` (o
   valor CRU, sem normalização) no log — dois valores negativos DIFERENTES confirmam essa
   leitura; valores idênticos indicariam bug real de código.

   **Checagem parcial (2026-08-15, `analysis/probe_head_training_report.txt`):** o valor
   CRU (`ICMSoft` não-normalizado) **não foi impresso** — a tabela do notebook só guardou
   as métricas normalizadas, então a confirmação direta exigiria re-rodar o probing (que
   está fechado como rota, §6.3) e **não foi feita**. Evidência indireta a favor do clip,
   registrada no lugar: na mesma linha `probe_c`, o `task_c_icmnorm_hard` dos dois trunks
   é **diferente** (CLIP 0,310759 · B15 0,312008) enquanto o soft é idêntico em 0,000000.
   Um bug determinístico (valor copiado/constante) teria colado as duas colunas; duas
   probes com pesos diferentes produzindo hard diferente e soft no mesmo piso exato é o
   comportamento esperado de um clip em zero. **Continua `[HIPÓTESE]`**, agora com um
   argumento a mais — não promover a `[VERIFICADO]` sem o valor cru.
2. **Os `best_epoch` (3, 3, 5, 6, 11, 15 de um orçamento de 300) contam a história
   principal: comparação de dosagem de treino, não de conteúdo de representação.** As
   heads originais receberam ~960 steps de mini-batch com dropout ativo nas features
   durante o treino end-to-end; as probes convergiram o BCE de validação em poucos steps
   full-batch e pararam ali — muito antes de a superfície de decisão calibrar para o que
   o ICM mede. É a mesma divergência loss-cedo/métrica-tarde (padrão Guo et al., ICML
   2017, já documentada nos docs 16/20 para o treino do trunk inteiro) reaparecendo
   *dentro* do próprio instrumento de diagnóstico — a probe herdou o problema que deveria
   medir.

### 6.3 Decisão: rota fechada por medição, não consertada

Seguindo a diretriz explícita do autor nesta sessão ("se o teste não for direto e
decisivo, não ficar andando em círculos"): **descartar a rota, não consertar a probe.**
Ajustar LR/épocas/regularização até os controles empatarem seria calibrar o instrumento
mirando o resultado esperado — borderline com o leakage que o projeto proíbe — e mesmo
que funcionasse, devolveria só um diagnóstico, não um ganho de métrica. Não repetir esta
rota. Fechada pelo mesmo estatuto das rotas fechadas por medição no doc 16.

### 6.4 O que isto NÃO decidiu — dito sem disfarce ⚠ SUPERADO, ver §6.6

> ⚠ **Escrito em 2026-08-15, antes das runs WC42 e da fase 2.** A afirmação "continua em
> aberto" era correta naquele momento e **deixou de ser** — ver §6.6. Mantido como
> registo do que se sabia quando o probing fechou; **não citar isoladamente**.

A bifurcação original do doc 20 (Task C sub-treinada dentro do treino conjunto vs a
representação congelada do trunk não carregando sinal suficiente de C) **continua em
aberto**. Este experimento não respondeu isso — só eliminou uma via barata de tentar
respondê-lo.

### 6.5 Caminhos restantes — decisão do autor ⚠ HISTÓRICO, ambos já percorridos

> ⚠ **Escrito em 2026-08-15.** O autor escolheu a opção 1; ela foi executada e
> **refutada** (doc 18 §9.5). Uma 4ª hipótese (freeze de A/B) foi testada depois e também
> refutada (doc 22 §6). Secção mantida como registo da decisão; o estado atual está em
> §6.6.

1. **Opção 5 do doc 20** — repesar `LOSS_W_C` no treino (0,7→4,2, valor **principiado**:
   desfaz a divisão por 6 da redução `.mean` em `loss_task_c`, não é valor afinado por
   busca), retreino completo do trunk, ~3,5h de GPU. Corrige a causa nos dois cenários da
   bifurcação em aberto (funciona seja C sub-treinada ou representação insuficiente,
   porque muda o que é aprendido, não só o que é lido). É **REALOCAÇÃO** — subir C deve
   custar A e/ou B; o relatório final tem que trazer as seis métricas, não só a que
   melhorou.
2. **Aceitar o estado atual como resultado final do braço CLIP** — 4 camadas, estado
   final da época 15, já documentado em `18_CLIP_VIDEO_BRANCH_SPEC.md` §9. A tese já tem
   um arco experimental completo sem essa run adicional (H2 testada, duas hipóteses de
   correção testadas e refutadas, achado de robustez sobre o vídeo carregar
   formato-não-conteúdo já registrado). Encerrar o braço aqui é decisão legítima, não
   desistência.

### 6.6 Epílogo (2026-08-15) — a pergunta acabou respondida por outro caminho

A bifurcação que este probing deveria decidir ("C sub-treinada" vs "representação sem
sinal de C") foi decidida pela run `CLIP_WC42` (opção 5 do doc 20, `LOSS_W_C` 0,7→4,2):
6× de gradiente efetivo na Task C **piorou** C hard (−0,032; −0,038 pós-tuning de
limiares) e rendeu só +0,009 em C soft. Evidência direta contra o ramo "sub-treinada",
vinda de uma intervenção no treino e sem o problema de protocolo que invalidou o probing.
Detalhe em `18_CLIP_VIDEO_BRANCH_SPEC.md` §9.5. ⚠ n=1 seed — evidência convergente,
não prova. Isto **não reabre** esta rota: o probing continua fechado.

**Atualização (2026-08-16):** uma quarta hipótese — congelar `head_a`/`head_b` e treinar
trunk + `head_c` por mais 5 épocas, com braço de controle sem freeze — também não moveu a
Task C (doc 22 §6). Duas intervenções de mecanismos **opostos** (mais gradiente para C;
menos competição de A/B sobre o trunk) falharam na mesma direção. A leitura convergente é
que o gargalo da Task C é a **representação do trunk**, não a alocação de treino entre
tasks. O eixo está declarado fechado em doc 22 §6.5.
