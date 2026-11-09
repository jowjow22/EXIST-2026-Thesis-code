# Handoff — extrair sinal de anotador: viés congelado + métrica de votos retidos

## Objectivo

Linha de caracterização pós-submissão, EXIST 2026 Task 2 (memes). A pergunta é
**se é possível extrair sinal de anotador utilizável** desta base — reliability,
deteção de anomalias — não melhorar a métrica do sistema. Um nulo é resultado
válido, desde que seja atribuível. Nada aqui é para seguir à risca: o critério
é que cada resultado seja interpretável.

Contexto e resultados já obtidos: `reliability/CONFUSION_EXPERIMENT.md` (protocolo
completo, todas as decisões e desvios declarados) e a subsecção
`\subsection{Annotator Confusion Matrices as an Auxiliary Objective}` já escrita
em `~/Documents/projects/Tese_arquivo_final_2_0_agora_vai_true/chapters/cap07.tex`.

## O que já está feito e medido (não repetir)

Sistema: XLM-R large + CLIP ViT-L/14, três cabeças (A binária, B intenção, C
multirrótulo), treino sobre soft labels. 3984 memes, 6 votos por item, 887
anotadores, mediana 27 votos/anotador. Split congelado 3386/598
(`reliability/outputs/split_taskA_frozen.json`). Métrica: Task A ICM-Soft-Norm na
época `E = argmin(eval_loss)`, 15 épocas, seeds pareadas {42, 7, 2025}.

Camada de confusão como perda auxiliar, `L = L_soft + λ·L_aux`, só na Task A,
`p_voto(a,i) = p_latente(i) @ C_a`, `C_a` 2x2 com linhas normalizadas.
Implementação em `EXIST_MEMES/confusion_core.py`, notebook
`EXIST_MEMES/memes_confusion.ipynb`, análise
`reliability/analyze_confusion_experiment.py`, testes
`EXIST_MEMES/confusion_selftest.py` (8 testes, passam em CPU local).

Resultados, todos replicados em 3 seeds:

1. **Controlo λ=0 passou.** Matrizes congeladas a 7e-9, baseline reproduz
   bit-a-bit. A divergência residual vem do `clip_grad_norm_` (6,08e-8 relativo);
   a `loss` do passo 1 é bit-a-bit igual, logo o forward do baseline é intocado.
2. **Com o LR do encoder (1e-5) as matrizes não se movem** (6e-4, o tecto de 795
   passos de AdamW) e `λ` não corrige, porque o AdamW é invariante à escala do
   gradiente e a auxiliar é a única fonte de gradiente delas.
3. **Com `C≈I` a auxiliar É o BCE da Task A** (`L_aux` implícito 0,767 / 0,751 /
   0,716 para λ 0,1 / 0,3 / 1,0), logo a grelha de λ degenera em reponderar a
   Task A: 2,0 → 2,0+λ. Δ = +0,0005 / +0,0003 / +0,0016 na seed 42.
4. **Com LR próprio (`CONF_LR=1e-2`) as matrizes movem-se** e alteram o alvo a
   sério: ajuste médio 0,125, máx 0,349, 331/3386 itens mudam de lado.
5. **Mas colapsam no viés marginal.** Viés = 88,7 % / 88,8 % / 88,6 % da variância;
   `viés = 0,827·yes_rate − 0,420`, R² 0,910 / 0,906 / 0,908. Competência sd 0,066.
6. **E não movem a métrica:** Δ = +0,0019 / +0,0010 / −0,0003, média +0,0009,
   sd pareado 0,0011, t=1,380, p=0,30. Δ_null (controlo) = −0,0017; sd do A0 entre
   seeds = 0,0059.
7. **Previsão registada (2026-09-10, antes de qualquer código novo):** a
   competência aprendida já é em grande parte o desvio à taxa base.
   `r(comp, |yes_rate − 0,55|)` = **−0,758 / −0,731 / −0,747** nas três seeds,
   contra `r(comp, yes_rate)` linear de apenas −0,15. `r(comp, concordância com a
   maioria)` = +0,217 / +0,231 / +0,248.

## Fase 2 — o que implementar

### A. Métrica de votos retidos (fazer PRIMEIRO — é a régua)

O ICM não é a régua certa para modelação de anotador. A régua certa é predizer
votos individuais que o modelo nunca viu.

**Usar os votos dos 598 itens de VALIDAÇÃO (3588 votos).** Nunca entraram nem no
`L_soft` nem no `L_aux`, e os anotadores são quase todos conhecidos do treino:
é a pergunta certa, *anotador conhecido, item novo*.

⚠ **Não reter votos de itens de treino** — foi a primeira ideia e tem leakage: o
`L_soft` treina na soft label desses itens, que é a média dos 6 votos, incluindo
o voto supostamente retirado.

Implementar como callback com o seu próprio loader sobre a validação, em
`on_evaluate`, modelo em `eval`, um forward para `logits_a`, e depois log-probs
por voto. Acrescentar `ConfusionAux.vote_logprobs(logits_a, ann_idx, ann_vote,
use_matrices)` reutilizando os mesmos lookups por comparação+matmul, com autocast
desligado. Gravar por época no `*_confusion.json`. **Não** passar votos pela perda
do Trainer: o `eval_loss` tem de continuar a ser `L_soft` puro, porque é o
selector de época.

Quatro preditores, todos sobre os mesmos votos, NLL e acerto por época:

- (a) `p_lat` sozinho (identidade — sem modelo de anotador)
- (b) `p_lat @ C_a` (a matriz aprendida)
- (c) taxa de YES do anotador no treino, constante (ignora o item)
- (d) **regressão logística de 2 parâmetros** sobre `[logit p_lat, logit yes_rate_a]`,
  ajustada nos votos de treino

**(d) é a que decide.** (b) pode ganhar a (a) e continuar a não valer nada. Se
(b) não bater (d), a matriz de 887×4 parâmetros não codifica nada que um ajuste
de duas variáveis não dê. Acrescentar a taxa de concordância com a maioria como
terceira feature de (d) se a competência aparecer a correlacionar com ela.

### B. Braço com viés congelado

Reparametrizar `C_a` explicitamente nas suas duas direções, em vez do `delta[2,2]`
livre. Com `u = logit P(YES|YES)` e `v = logit P(NO|NO)`:

```
s = (u+v)/2   -> competência = 2σ(s) − 1
d = (u−v)/2   -> viés        = σ(u) − σ(v)
```

M1 fica `u_a = s0 + d0 + s_a`, `v_a = s0 − d0 + s_a`, com o `d0` **comum livre**
(calibração do painel, não é o sinal de anomalia) e `d_a` **congelado a zero**.
Init `s0 = 3` reproduz o `softmax([3,0]) = 0,9526` das runs anteriores.

Só o modo **congelado**. Não implementar variante "penalizada": o peso da
penalização seria hiperparâmetro novo sem valor a priori, e viraria busca.

Decidir e declarar o que fazer ao `reg_weight=0.01` que hoje incide no `Δ_a`
inteiro: em `s_a` puxa a competência para o valor do painel.

Teste de sanidade novo: em modo congelado, `C[:,0,1] == C[:,1,0]` para todos os
anotadores, com precisão de vírgula flutuante.

**Previsão a verificar (já registada):** bloqueado o viés, a direcção seguinte
mais fácil é concordância com a média dos votos, e um anotador com `yes_rate`
extremo discorda mais da média a uma taxa base de 0,55. Esperar que a competência
absorva `|yes_rate − 0,55|`. Medir `r(comp, |yes_rate−0,55|)` e
`r(comp, concordância)`, **não** a correlação linear com a `yes_rate`, que esconde
a forma em U.

### C. Dificuldade por item — ADIADA, não implementar agora

Dois motivos: um `g_i` por item de treino não existe para itens de validação, logo
a métrica de A não o consegue avaliar; e a 6 votos/item vai re-aprender a entropia
dos votos, como a dificuldade do IRT re-aprendeu a proporção (resultado já na
tese, `\subsection{Item Response Theory as a Non-Circular Estimator}`). Correr
junto com B tornaria um nulo inatribuível.

Se for retomada: parametrizar `g` como função de `p_lat` ou uma cabeça pequena
sobre `fused`, para transferir a itens novos, e como braço separado.

## Critério, a fixar antes da primeira run

O braço congelado **extrai sinal** se e só se a NLL de (b) for menor que a de (d)
nos votos retidos, nas **três** seeds. Registar também (b) contra (a), e o sd da
competência contra os 0,066 do braço de viés livre.

## Plano de runs (~4 h)

1. **1 run** do braço actual (viés livre, `CONF_LR=1e-2`, λ=0,1, seed 42) **com a
   métrica nova** — sem isto não há termo de comparação para o congelado, e a
   reparametrização torna as runs antigas não comparáveis bit-a-bit.
2. **3 runs** do braço congelado, seeds 42/7/2025, mesmos λ=0,1 e `CONF_LR=1e-2`.

`λ` e `CONF_LR` ficam fixos: já está demonstrado que `λ` é irrelevante para as
matrizes.

## Armadilhas do ambiente (custaram tempo nesta sessão)

- A pasta local `~/Documents/projects/VM-THESIS` **é a VM montada por sshfs**.
  Com o disco cheio, `open(...,'w')` trunca o ficheiro e o `write` falha: já
  destruiu dois ficheiros. **Escrever sempre no scratchpad local e só depois
  copiar**, e verificar `df -h` antes.
- Disco da VM: 126 GB, ~11 GB livres. `save_strategy="no"` já está no notebook.
  Não reactivar checkpoints (~3,3 GB cada).
- Depois de alterar `confusion_core.py`, **restart do kernel** antes de correr:
  o `import` não relê o ficheiro.
- Os logs demoram a sincronizar da VM. Verificar por `ls -lt` e não assumir.
- Tese: bibliografia é **biber**, não `bibtex` (correr `bibtex` apaga o `.bbl`).
  Citações com `\cite`, não `\citep`.

## Pendente na tese

A subsecção nova em `cap07.tex` diz que a competência não correlaciona com nada,
com base na correlação **linear** (−0,15). Corrigir com o achado 7: não
correlaciona linearmente com a taxa de positivos, mas correlaciona −0,74 com o
desvio à taxa base. Também está por corrigir a frase da subsecção do IRT que diz
"four independent families have now been evaluated" — com a camada de confusão são
cinco. E confirmar as três referências acrescentadas a `lib/refs.bib`
(`chu2021common`, `tanno2019regularized`, `rodrigues2018crowds`): foram escritas
sem verificação em fonte primária, e a auditoria do paper já apanhou duas
referências fabricadas.

## Como trabalhar

Português do Brasil no chat. Fluxo iterativo: propor desenho, explicar decisões
antes de escrever código, funções de substituição completas em vez de diffs, correr
diagnósticos e interpretar o output. Nunca escolher a época pela métrica reportada.
Nunca alterar o A0, o split, nem a métrica pré-registada. Desvios ao protocolo são
aceitáveis desde que declarados no `CONFUSION_EXPERIMENT.md` com a razão e a data.
