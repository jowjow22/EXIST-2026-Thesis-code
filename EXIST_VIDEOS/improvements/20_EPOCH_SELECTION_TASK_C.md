# 20 — Seleção de época em MTL quando a Task C atinge o pico DEPOIS do `argmin(eval_loss)`

**Escrito em:** 2026-08-14
**Formato:** o mesmo do `17_RETRAIN_IMPROVEMENT_OPTIONS.md` — pesquisa em paralelo (3
agentes, um por ângulo) → segundo passe crítico → opções numeradas com *mecanismo · o que
resolve · esforço · exige re-treino? · risco de leakage · citação*.
**Fontes:** leitura direta de `video_core.py` e `Thesis_video_clip.ipynb`,
`meta.json`/`final_metrics.json` das runs, `09_BASELINE_RESULTS.md`,
`17_RETRAIN_IMPROVEMENT_OPTIONS.md`, e um levantamento de literatura feito nesta sessão
com **verificação individual de cada citação por fetch da página real**.
**Nada de código foi alterado.** Este documento é decisão de autor primeiro.

---

## 0. O problema, com os números que estão no disco

A regra do projeto é `E = argmin(eval_loss)` **conjunta**, fixada a priori
(`video_core.py:283-307`, ARCH-06). Escolher a época pela métrica reportada é leakage e
está proibido.

No braço CLIP_L2 (`run_B2_bio_viewer_seed42_CLIP_L2/meta.json` — verificado):

```
selected_epoch          = 4
eval_loss_at_selection  = 1.4802
epoch_selection         = "argmin(eval_loss)"
num_train_epochs        = 15
```

E a Task C (`icmsoftnorm_soft`) reportada pelo autor a partir da curva daquela run:

| época | Task C soft |
|---|---:|
| 4 (`argmin` oficial) | 0,0524 |
| 10 | **0,1172** |
| 15 (final) | 0,1138 |

Mais que o dobro do valor da época escolhida. A e B não mostram esse padrão.

### 0.1 Três fatos que reenquadram o problema antes de qualquer proposta

**(a) A loss conjunta que decide o `argmin` quase não contém a Task C.**
`loss_task_c` (`video_core.py:1334-1349`) faz `.mean(dim=-1)` sobre as 6 classes;
`loss_task_b` (`:1323-1331`) faz `.sum(dim=-1)` sobre as 3. Com
`LOSS_W_A=1.0, LOSS_W_B=0.7, LOSS_W_C=0.7` (`:272-274`), o peso **efetivo** é
**1,0 · 0,7 · ~0,117**. O critério de seleção é, por aritmética, dominado por A e B.
(Isto já estava registado como hipótese no próprio código, `:1337-1341`, e é o item 3 do
doc 17 — a novidade aqui é que ele também contamina a **seleção de época**, não só o
treino.)

**(b) Não existe `eval_loss_C`. Nunca foi medida.**
`MultiTaskTrainer.log` (`:1857-1866`) só injeta `loss_a/b/c` quando `"loss" in logs` — ou
seja, **só nos logs de treino**. Os logs de avaliação usam `eval_loss` e não recebem nada.
Portanto **ninguém sabe onde cai `argmin(eval_loss_C)`**, e é essa incógnita que decide
qual das opções abaixo é a certa.
⚠ Achado secundário **[NÃO CONFIRMADO NA VM — raciocínio sobre o código-fonte do HF
Trainer, transformers==5.6.0]**: o `prediction_step` chama `compute_loss`, então
`_loss_a/b/c_accum` acumulariam batches de *eval* e descarregariam no log de *treino*
seguinte. As per-task losses de treino registadas **poderiam** estar levemente
contaminadas. Antes de tratar como fato, confirmar na VM (mesmo padrão de cautela do
`warmup_ratio`/`VideoMAEImageProcessorFast` do `CLAUDE.md`). Se confirmado, é uma linha
de correção, não um bug de resultado.

**(c) Os pesos por época existem — ou existiam.**
`Thesis_video_clip.ipynb` (seção 10) regista um `SaveTrainableStateCallback` que grava
`checkpoints_.../epoch_states/epoch_NN.safetensors` (~120 MB, só params treináveis) a
**cada época**. Se as 15 sobreviveram na VM, tudo neste documento que envolve "olhar
outra época" custa **passes de avaliação**, não re-treino.
**Estado verificado localmente: só `epoch_15.safetensors` sincronizou** (nos dois braços
CLIP). Confirmar na VM é o pré-requisito de custo de metade deste documento.

### 0.2 Segundo passe crítico — quatro objeções ao enquadramento "a regra falhou"

1. **A métrica que sobe tarde é SOFT.** O precedente clássico (Guo et al. 2017, §1.1) é
   sobre NLL a subir enquanto o **erro 0/1** melhora — loss probabilística contra métrica
   de decisão. Aqui a métrica que sobe é distribucional (ICM-Soft-Norm), que *deveria*
   acompanhar a loss. Isso enfraquece a leitura "é só calibração" e reforça a leitura
   "a cabeça C está sub-treinada" — mas **as duas são indistinguíveis sem medir
   `eval_loss_C`**.
2. **Escala:** 0,1172 na época 10 do CLIP_L2 continua **abaixo** do baseline B15
   (0,1514, `17_...§1.0`). O braço não está a ser prejudicado pela regra até um valor bom;
   está sub-treinado em C até um valor ainda pior que o baseline.
3. **O braço é atípico.** O próprio notebook regista que no CLIP_L2 o ramo de **texto
   colapsou** (‖Δ‖ 24× menor que no VideoMAE). Mudar uma regra global do projeto a partir
   da curva de um braço anómalo é fraco. Confirmar o padrão no B15 antes.
4. **Armadilha pós-hoc.** Qualquer critério adotado agora foi escolhido **depois** de ver
   a curva da Task C desta run. "Definido a priori" é irrecuperável para o CLIP_L2. Ver §5.

---

## 1. O que a literatura sustenta — e o que NÃO sustenta

Regra desta seção: **só entra citação cuja página foi aberta e lida nesta sessão**. Onde
não há fonte, está escrito "sem fonte verificada" — a ausência é achado, não lacuna.

### 1.1 O fenómeno é conhecido e publicado

- **Guo, Pleiss, Sun, Weinberger — *On Calibration of Modern Neural Networks*, ICML 2017,
  PMLR v70.** https://arxiv.org/abs/1706.04599 — §3 / Fig. 3, verbatim:
  *"NLL overfits during the remainder of training. Surprisingly, overfitting to NLL is
  beneficial to classification accuracy"*; no CIFAR-100 o erro de teste cai de 29 % para
  27 % **na região em que a NLL sobe**.
  → Loss de validação no mínimo e métrica ainda a subir **não é anomalia do nosso setup**.
  ⚠ Mas o paper fala de métrica *hard*; a nossa é soft (ver §0.2 ponto 1).

- **Harutyunyan, Khachatrian, Kale, Ver Steeg, Galstyan — *Multitask learning and
  benchmarking with clinical time series data*, Scientific Data 6:96 (2019).**
  https://arxiv.org/abs/1703.07771 — seção Methods, verbatim:
  *"the model converges and then overfits at very different rates during training. This is
  often addressed through the use of heuristics, including a multitask variant of early
  stopping, in which we identify the best epoch for each task based on individual task
  validation loss."*
  → Precedente direto: época por task, escolhida pela **loss** daquela task.
  📌 **Nota de auditoria:** um dos agentes de pesquisa desta sessão afirmou que esta
  atribuição "não sobreviveu à verificação" e mandou não a usar. **Verifiquei eu mesmo por
  fetch do texto completo: a frase está lá, literal.** O agente errou. Registado aqui
  porque a política do projeto é rastrear afirmações sobre citações nos dois sentidos.

- **Koreeda, Morio, Morishita, Ozaki, Yanai — *Hitachi at MRP 2019: Unified
  Encoder-to-Biaffine Network for Cross-Framework Meaning Representation Parsing*, Shared
  Task on Cross-Framework MRP @ CoNLL 2019.** https://arxiv.org/abs/1910.01299 —
  Apêndice A.1, verbatim: *"For the multi-task variants, we employ a slightly different
  strategy for early stopping. For the multi-task variant without fine-tuning, we apply
  early stopping separately to each framework with respect to the framework-specific
  validation loss."*
  → É o protocolo mais limpo disponível: treino com perda somada, **seleção por task pela
  loss de validação daquela task** — critério cego à métrica reportada.

- **Horych, Wessel, Wahle, Ruas, Waßmuth, Greiner-Petter, Aizawa, Gipp, Spinde — *MAGPIE*,
  LREC-COLING 2024.** https://arxiv.org/abs/2403.07910 — *Head-Specific Early Stopping*:
  a head para pela **própria val loss** e *"we stop updating its specific head parameters
  while still backpropagating its language model gradients"*; o mecanismo *Resurrection*
  existe porque a head parada precisa voltar a adaptar-se *"to the latest latent
  representation"*.
  → Duplo uso: sustenta parada por head **e** é a evidência mais direta **contra** enxertar
  uma head tardia num trunk antigo (§3).

### 1.2 O que a literatura NÃO sustenta (achados negativos, úteis)

- **Ponderação dinâmica de tarefas não ataca este sintoma.** Verificados: Kendall/Gal/
  Cipolla (CVPR 2018, arxiv 1705.07115), GradNorm (Chen et al., ICML 2018, arxiv
  1711.02257), PCGrad (Yu et al., NeurIPS 2020, arxiv 2001.06782), CAGrad (arxiv
  2110.14048), IMTL (ICLR 2021), Nash-MTL (ICML 2022, arxiv 2202.01017), DWA
  (Liu/Johns/Davison, CVPR 2019, arxiv 1803.10704), MultiNet++ (CVPRW 2019, arxiv
  1904.08492), survey de Vandenhende et al. (T-PAMI, arxiv 2004.13379).
  **Nenhum deles mede ou reivindica alinhamento temporal dos ótimos de validação por
  task.** Todos medem performance final/média. GradNorm equaliza *training rates* — sobre
  a loss de **treino**, não de validação. → Responde ao ângulo 2 da pergunta: **é o eixo
  errado do problema.**
- **Ceticismo documentado:** Xin, Ghorbani, Gilmer, Garg, Firat, *Do Current Multi-Task
  Optimization Methods in Deep Learning Even Help?*, NeurIPS 2022; Kurin, De Palma,
  Kostrikov, Whiteson, Kumar, *In Defense of the Unitary Scalarization*, NeurIPS 2022
  (arxiv 2201.04122) — §5.1 verbatim: *"SMTOs delay overfitting, requiring less early
  stopping compared to unitary scalarization"*; §4.1: *"we select a model on the validation
  set, and later report test metrics"*. Lin et al., *Reasonable Effectiveness of Random
  Weighting* (arxiv 2111.10603): pesos aleatórios empatam com o estado da arte.
- **"Um critério único penaliza sistematicamente a task mais lenta" — sem fonte
  verificada** como resultado *medido*. O fenómeno é afirmado no lado do **treino**
  (convergência assíncrona), não no lado da **seleção de checkpoint**.
- **"Cabeça multirrótulo esparsa atinge o pico mais tarde que binária/multiclasse no mesmo
  treino conjunto" — sem fonte verificada.** Existe literatura sobre a *dominância dos
  negativos* em BCE multirrótulo (Ben-Baruch/Ridnik et al., *Asymmetric Loss*, ICCV 2021,
  arxiv 2009.14119: o desbalanço *"dominates the optimization process… under-emphasizing
  gradients from positive labels"*; Wu et al., *Distribution-Balanced Loss*, ECCV 2020,
  arxiv 2007.09654), mas **nada sobre número de épocas**. → O ângulo 5 da pergunta
  ("é só multirrótulo, muda quando avaliar") **não tem apoio direto na literatura**; é
  plausível por composição, não demonstrado.
- **Regra a-priori de peso por nº de classes / tamanho da head — sem fonte verificada.** A
  única regra a-priori com base publicada é a de ruído homocedástico de Kendall et al., e
  ela é *aprendida durante o treino*, não fixada antes. O que **existe** verificado são
  normalizações adimensionais: IMTL-L (igualar magnitudes), DWA (razão de losses),
  MultiNet++ (média geométrica, motivada por *"difference in convergence rates"*), Δ_MTL
  do survey T-PAMI (queda relativa por task). Todas usadas no **treino** ou no **report**;
  usá-las como **critério de validação** é extensão minha, não claim de paper.

---

## 2. PASSO 0 — medir antes de propor (custo zero de treino)

A pergunta que bifurca tudo: **onde cai `argmin(eval_loss_C)`?**

- **Se cair tarde (~8-12):** a divergência é real e a seleção por task pela loss da task
  (opção 3) resolve, com precedente publicado.
- **Se cair cedo, junto com a conjunta:** é o padrão Guo et al. — a loss piora enquanto a
  métrica melhora — e **nenhum critério cego à métrica ratifica a época 10**. Nesse caso a
  resposta honesta é: manter E=4 e mover o esforço para repesagem no treino (opção 5) e
  calibração (opção 6).

**Procedimento (na VM):**
1. Confirmar `ls checkpoints_B2_bio_viewer_seed42_CLIP_L2/epoch_states/` → 15 ficheiros.
2. Para cada época: recarregar o `epoch_NN.safetensors` por cima do modelo final — **o
   código exato já existe**, célula 30 do `Thesis_video_clip.ipynb` (`load_state_dict`
   com asserts de chaves).
3. Rodar `trainer.predict(val_dataset)` e computar `loss_a/b/c` a partir dos logits com as
   funções que já existem (`vc.loss_task_a/b/c`).
4. Tabela: época × (`eval_loss`, `loss_a`, `loss_b`, `loss_c`, métricas já registadas).

**Probe irmão, também de custo zero:** recomputar a loss **conjunta** por época com
`LOSS_W_C ∈ {0,7 · 1,5 · 2,8 · 4,2}` sobre os mesmos estados e ver para onde o `argmin`
se move. Isso separa **"repesar muda o que é SELECIONADO"** de **"repesar muda o que é
APRENDIDO"** — separação que uma run re-treinada sozinha não consegue fazer.

⚠ Se os `epoch_states` não sobreviveram: o passo 0 exige **uma** re-run de ~3,5 h com o
callback ativo (e com a opção 1 já instrumentada). Nada além disso.

---

## 3. "Checkpoint Frankenstein" — veredito

**Enxertar pesos (trunk da época 4 + head C da época 10) não tem NENHUMA fonte
verificada que o autorize.** As duas técnicas que superficialmente parecem autorizar não
autorizam:
- *Model soups* (Wortsman et al., ICML 2022, arxiv 2203.05482) e *checkpoint averaging*
  mediam **modelos inteiros** dentro da mesma bacia de erro; não recombinam componentes de
  épocas diferentes.
- O *Resurrection* do MAGPIE é evidência **contra**: existe precisamente porque a head
  parada precisa reajustar-se à representação latente mais recente. Head e trunk têm de
  ser mutuamente consistentes.

**A variante limpa não é costurar pesos — é seleção de MODELO por task.** As submissões
EXIST são ficheiros por subtask, e há (ou havia) checkpoints **inteiros** por época:
usar o checkpoint inteiro `E_AB` para A/B e o checkpoint inteiro `E_C` para C é
internamente consistente, sem mismatch trunk/head, e tem precedente publicado (Hitachi
2019, Harutyunyan 2019). É a opção 3.
⚠ Preço: "o sistema" passa a ser 2 modelos, não 1 — tem de ser declarado assim na tese e
no paper, não escondido atrás de uma tabela de métricas.

---

## 4. Opções

Ranqueadas por **(a)** não exigir mudança de arquitetura/re-extração, **(b)** esforço de
código, **(c)** força da evidência — **não** por promessa de ganho.

Cada opção traz um **eixo de regra**:
- **[LETRA]** — continua a ser *um único critério conjunto, cego à métrica reportada*.
  Compatível com a regra como está escrita.
- **[EMENDA]** — mantém o *espírito* (cego à métrica) mas deixa de ser critério único.
  **Exige decisão explícita do autor + orientadores**; não é algo que eu decida.

---

### Opção 1 — Instrumentar `eval_loss_a/b/c` (habilita tudo o resto)

**Mecanismo.** Computar as três losses por task dentro de `compute_metrics`
(`video_core.py:1674`), em numpy, a partir dos logits e labels que a função **já recebe**.
Replicar exatamente `loss_task_a/b/c` (incluindo as sentinelas `SENTINEL_A`/`SENTINEL_BC`).
Devolver `eval_loss_a`, `eval_loss_b`, `eval_loss_c` como métricas.
Correção irmã: descarregar os acumuladores do `MultiTaskTrainer` em `on_evaluate` (ou
gate por `"eval_loss" not in logs`), para os logs de treino deixarem de misturar batches
de avaliação.

**O que resolve.** Torna o passo 0 possível. Sem isto, todas as outras opções são
especulação sobre uma curva que ninguém viu.

**Esforço.** Baixo — uma função de ~25 linhas, sem tocar no caminho de treino.
**Re-treino?** Não, para runs futuras. Para a curva do CLIP_L2, só se os `epoch_states`
tiverem desaparecido.
**Risco de leakage.** **Zero.** `eval_loss_*` não é métrica reportada e não passa pelo
PyEvALL.
**Citação.** Não precisa — é instrumentação. O uso a que serve está em Harutyunyan et al.
2019 e Hitachi/MRP 2019 (§1.1).

---

### Opção 2 — Critério conjunto com escala normalizada, fixado a priori **[LETRA]**

**Mecanismo.** Manter *um único critério conjunto*, mas corrigir a assimetria de redução
que o distorce: no **critério de validação**, usar a loss C com redução `sum` (equivalente
a `LOSS_W_C = 4,2 = 0,7 × 6`), de modo que os pesos efetivos passem a ser os documentados
(1,0 / 0,7 / 0,7). **O treino não muda.** Alternativas da mesma família, todas
adimensionais: média geométrica das três losses (MultiNet++), ou losses z-scoradas por
task ao longo das épocas.

**O que resolve.** Ataca a causa aritmética identificada em §0.1(a): hoje o `argmin`
ignora a Task C por construção. Se o passo 0 mostrar que `argmin(loss_C)` cai tarde, esta
opção move o `argmin` conjunto na direção certa **sem quebrar a regra do critério único**.
⚠ Se `argmin(loss_C)` cair cedo, esta opção não move nada — e isso é uma resposta, não um
fracasso.

**Esforço.** Baixo (constante + registo no `meta`). Testável **sem GPU de treino** pelo
probe irmão do passo 0.
**Re-treino?** Não.
**Risco de leakage.** **Baixo, com uma ressalva séria:** a correção `0,7 → 4,2` é
*principiada* (desfaz a divisão por 6 da redução `.mean`), não um número afinado — é isso
que a mantém do lado da letra da regra. Varrer um grid `{1,5 · 2,8 · 4,2}` e escolher o
que dá a melhor Task C **seria leakage**. Se se usar grid, ele é diagnóstico; o valor
adotado tem de ser o principiado.
**Citação.** IMTL-L (ICLR 2021, igualar magnitudes de loss por task); DWA
(Liu/Johns/Davison, CVPR 2019, arxiv 1803.10704 — razão de losses, adimensional);
MultiNet++ (CVPRW 2019, arxiv 1904.08492 — média geométrica, motivada por *"difference in
convergence rates of different tasks"*); Δ_MTL do survey de Vandenhende et al. (T-PAMI,
arxiv 2004.13379). ⚠ Todos usam essas normalizações no **treino** ou no **report**;
aplicá-las ao **critério de validação** é extensão minha — declarar como tal na tese.

---

### Opção 3 — Seleção por task, checkpoint INTEIRO, pela loss daquela task **[EMENDA]**

**Mecanismo.** `E_A = argmin(eval_loss_A)`, `E_B = argmin(eval_loss_B)`,
`E_C = argmin(eval_loss_C)`. Cada subtask é reportada com o checkpoint **inteiro** da sua
época. Sem enxerto de pesos (§3). Critério de cada task continua **cego à métrica
reportada** — é a loss, não a ICM.

**O que resolve.** É o remédio direto para o sintoma, *se* o passo 0 mostrar divergência
real entre os `argmin` por task. Operacionalmente natural no EXIST (submissão por
subtask).

**Esforço.** Baixo depois da opção 1 (é escolher índices numa tabela) — mas exige guardar
`epoch_states` em todas as runs futuras, e disciplina de artefatos: cada subtask passa a
apontar para um `run_dir`/época diferente, e o `meta.json` tem de registar isso.
**Re-treino?** Não (com `epoch_states`).
**Risco de leakage.** **Baixo pelo critério, alto pela narrativa.** O critério é cego. Mas
"o sistema" deixa de ser um modelo e passa a ser 2-3, e a **regra de simetria** do
`CLAUDE.md` obriga: se o corrigido é selecionado por task, o baseline tem de ser
reportado por task também. Comparar "por task" contra "época única" seria comparar
maçã com laranja.
**Citação.** Koreeda et al., Hitachi at MRP 2019 (arxiv 1910.01299, Apêndice A.1) —
*"we apply early stopping separately to each framework with respect to the
framework-specific validation loss"*; Harutyunyan et al. 2019 (arxiv 1703.07771) —
*"identify the best epoch for each task based on individual task validation loss"*;
MAGPIE (arxiv 2403.07910) para a variante dentro do treino.

---

### Opção 4 — Head C re-treinada sobre trunk congelado em E **[LETRA]**

**Mecanismo.** Congelar o modelo inteiro no `E` oficial, extrair o vetor de fusão para
todo o treino/val, e treinar **só a head C** (uma `Linear`) sobre essas features
congeladas, parando pela própria `val_loss_C`. É *linear probing* clássico.

**O que resolve.** Dá à cabeça C mais treino sem mover o trunk — logo sem arrastar A e B
para a cauda de overfitting, e sem o problema de consistência do §3 (o trunk não muda,
então não há mismatch por construção).

**Esforço.** Médio (extrair features + loop de treino pequeno), mas **minutos de GPU**,
não horas.
**Re-treino?** Não do sistema; sim de uma camada.
**Risco de leakage.** Baixo, desde que a parada seja pela loss e o report continue a
respeitar o protocolo de metade/metade do `tune_task_c_thresholds`
(`video_core.py:1724`).
**Citação.** Alain & Bengio, *Understanding intermediate layers using linear classifier
probes*, 2016, arxiv 1610.01644 — probes treinados sobre representações congeladas,
*"trained entirely independently of the model itself"*. ⚠ O paper é sobre **diagnóstico**;
usar a probe como componente do sistema é extensão minha.

---

### Opção 5 — Repesar `LOSS_W_C` no TREINO (= item 3 do doc 17) **[LETRA]**

> ❌ **EXECUTADA E REFUTADA (2026-08-15)** `[VERIFICADO]` — `LOSS_W_C = 4,2` via
> `vcc.apply_loss_reweight_c()`, run `run_B2_bio_viewer_seed42_CLIP_WC42`, tudo o mais
> congelado. Resultado: **A hard −0,0388 · B hard −0,0253 · C hard −0,0322
> (−0,0376 pós-tuning de limiares) · C soft +0,0088** contra o braço `CLIP`. A própria
> Task C piora no hard; o único ganho é +0,009 em C soft, uma seed, ainda abaixo do B15.
> Tabela completa e leitura em `18_CLIP_VIDEO_BRANCH_SPEC.md` §9.5.
>
> **Consequência para este documento:** o ramo "Task C sub-treinada dentro do treino
> conjunto" perde suporte — 6× de gradiente em C não fez C render mais. Evidência
> convergente (não prova, n=1 seed) a favor do outro ramo: a representação do trunk não
> carrega mais sinal de C. **Não repetir com outro valor de peso** — seria grid contra a
> métrica reportada (leakage) e a direção do efeito já é conhecida.

**Mecanismo.** `LOSS_W_C = 0,7 → 4,2` (ou `.mean → .sum` em `loss_task_c`) no **treino**.
Muda o que é aprendido, não só o que é selecionado.

**O que resolve.** Se o passo 0 mostrar que a C está sub-treinada (loss C ainda a cair na
época 15), esta é a correção de causa, não de sintoma. Coerente com §0.2 ponto 2: a C do
CLIP_L2 está abaixo até do baseline B15.

**Esforço.** Trivial no código, **caro em GPU** (~3,5 h por ponto). Triagem barata:
backbones congelados (OPT-09).
**Re-treino?** **Sim.**
**Risco de leakage.** Baixo se o valor for o principiado (4,2) e o `0,7` estiver no braço
de comparação. **É REALOCAÇÃO, não ganho de graça** — subir C deve custar A e/ou B; o
report tem de trazer as seis métricas.
**Citação.** Kendall et al. (CVPR 2018) para a família; ⚠ mas ver §1.2: a literatura é
cética com balanceamento sofisticado (Xin et al. NeurIPS 2022; Kurin et al. NeurIPS 2022;
Lin et al. arxiv 2111.10603) e **explicitamente favorável a um pequeno grid manual sobre
pesos fixos**, que é exatamente esta opção.

---

### Opção 6 — Calibração pós-treino para as métricas soft (= item 4 do doc 17) **[LETRA]**

> ❌ **EXECUTADA E REFUTADA na sua forma pré-registada (2026-08-16)** `[VERIFICADO]` —
> temperature scaling por NLL soft (escalar em A/B, vector na C), ajustado numa metade da
> val e reportado na outra, sobre `run_..._CLIP` e `run_..._B15`. **Piora o ICMSoftNorm em
> 6/6 casos** (Δ REPORTE de −0,059 a −0,131), nas duas metades e nos dois encoders — não é
> overfit da calibração, é sistemático. Todo T ajustado ficou em 2,4–3,9: a NLL quer
> suavizar, e o ICM-Soft — métrica de informação — penaliza exatamente isso.
> `[HIPÓTESE]`: NLL e ICM-Soft-Norm são objetivos antagónicos, logo o método padrão de
> calibração é o objetivo errado para esta métrica. Tabela completa e diagnóstico em
> `23_CALIBRATION_AND_NAN_RESULTS.md`.
>
> **Em aberto:** a variante que ajusta T contra o **ICM-Soft-Norm** (não a NLL) na metade
> de ajuste — doc 23 §2.3. Não executada, aguarda decisão do autor.

**Mecanismo.** Temperatura por task (vector scaling na C), ajustada numa metade da val e
reportada na outra, no molde de `tune_task_c_thresholds`.

**O que resolve.** Se o passo 0 mostrar o padrão Guo et al. (loss C sobe, métrica soft
sobe), então o que se degrada é a **calibração** — e esse é o remédio nomeado por esse
mesmo paper. Não move as métricas hard (argumento completo no doc 17, item 4).

**Esforço.** Médio; **zero GPU de treino**.
**Re-treino?** Não.
**Risco de leakage.** Baixo com o protocolo metade/metade; alto se ajustado e reportado na
val inteira (o "regime otimista" do `09_BASELINE_RESULTS.md`).
**Citação.** Guo et al., ICML 2017 (arxiv 1706.04599) — temperature scaling; e §3/Fig. 3
como diagnóstico do próprio fenómeno.

---

## 5. Descartadas, com a razão citada

| rota | por que fica de fora |
|---|---|
| **SWA / EMA sobre a cauda** | Já **medido pior em 6/6 métricas** neste código (ARCH-06, `09_BASELINE_RESULTS.md`). E a premissa do método não se cumpre aqui: Izmailov et al. (UAI 2018, arxiv 1803.05407) definem SWA com **LR cíclico ou constante**; o projeto usa `lr_scheduler_type="cosine"`, que decai a ~0 — os pontos da cauda estão praticamente no mesmo sítio do espaço de pesos. Rota fechada por medição **e** por premissa. |
| **Model soups (greedy)** | Wortsman et al., ICML 2022 (arxiv 2203.05482), Alg. 1: os ingredientes são ordenados por *validation set accuracy* e só entram se a val melhora. Isso é **seleção pela métrica em held-out** — leakage sob a regra do projeto. Fechada **por regra**, não por medição. |
| **GradNorm / PCGrad / CAGrad / Nash-MTL / uncertainty weighting** | Nenhum mede alinhamento temporal dos ótimos por task (§1.2) — é o eixo errado. Somam custo (backward por task em PCGrad/GradNorm) num orçamento de 1 GPU e 3,5 h/run, e a literatura recente é cética (Xin et al. 2022; Kurin et al. 2022; Lin et al. 2021). |
| **"Frankenstein" por enxerto de pesos (trunk@4 + head C@10)** | Sem nenhuma fonte verificada que o autorize; o *Resurrection* do MAGPIE (arxiv 2403.07910) é evidência direta contra. Substituído pela opção 3. |
| **Escolher a época pela métrica de C, mesmo "só para a C"** | É literalmente o vazamento nº 2 do precedente dos memes, já codificado em `video_core.py:303-307`. Não negociável. |
| **`load_best_model_at_end=True` com `metric_for_best_model`** | Mesma coisa que a linha acima, com automação. `LOAD_BEST_MODEL_AT_END=False` (`:307`) e `metric_for_best_model` deliberadamente ausente existem para isso. |

---

## 6. A armadilha pós-hoc — dita sem suavizar

Qualquer critério escolhido agora foi escolhido **depois** de ver a curva da Task C do
CLIP_L2. Para **esta run**, "definido a priori" é irrecuperável. Nenhuma reformulação
conserta isso.

Protocolo honesto, e é o único que a tese pode defender:

1. **Pré-registar** a regra nova aqui, por escrito, com data, **antes** de olhar qualquer
   outra curva (este documento serve como esse registo).
2. **Aplicá-la a braços/seeds ainda não inspecionados** (B15 é o candidato natural: mesmo
   split, encoder diferente) e reportar esses números como a evidência principal.
3. **Marcar explicitamente** qualquer número do CLIP_L2 sob a regra nova como **pós-hoc**,
   na mesma linha em que aparece — do mesmo jeito que o `09_BASELINE_RESULTS.md` marca
   "mesmo val — regime otimista, não comparar".
4. Manter a **regra de simetria**: se o sistema corrigido é selecionado por task, o
   baseline é reportado por task também.

---

## 7. Ordem recomendada

1. **Confirmar na VM se `epoch_states/` do CLIP_L2 tem as 15 épocas.** Uma linha de `ls`.
   Decide se o resto custa horas de CPU ou uma re-run.
2. **Opção 1** (instrumentar `eval_loss_a/b/c`) — habilita tudo, risco zero.
3. **Passo 0** (§2): curva de `loss_C` por época + probe do `argmin` sob `LOSS_W_C`
   alternativo. **Só aqui se sabe qual das opções é a certa.**
4. Bifurcação:
   - `argmin(loss_C)` **tarde** → opção 2 (letra) e/ou opção 3 (emenda, decisão do autor).
   - `argmin(loss_C)` **cedo** → o padrão é Guo et al.; o esforço vai para a opção 5
     (repesar no treino) e opção 6 (calibração). A regra atual **fica como está**.
5. **Opção 4** como ablação barata em qualquer dos ramos.
6. Repetir o passo 0 no **B15** antes de propor qualquer mudança de regra global — o
   CLIP_L2 tem o ramo de texto colapsado e não é uma base segura para legislar.

**Nenhuma destas opções é uma promessa de ganho de métrica.** Três delas (1, passo 0, 3)
são sobre *medir e reportar corretamente*; duas (2, 6) são sobre *ler o modelo que já
existe*; só a 5 muda o que o modelo aprende.
