# 05 — Notas de implementação (Agente 4)

Escrito a partir de `04_RECONCILIATION.md` (especificação primária). Nada aqui
re-audita nem re-decide — documenta **o que foi escrito, onde, e o que ficou de fora**.

## Arquivos produzidos

| Arquivo | O que é |
|---|---|
| `EXIST_VIDEOS/video_core.py` | módulo importável, ~1 100 linhas, sem execução de treino no import |
| `EXIST_VIDEOS/Thesis_video_improved.ipynb` | notebook de orquestração, 35 células (17 código / 18 markdown) |
| `EXIST_VIDEOS/Dataset/val_split_42.json` | **artefato do split congelado** — gerado e verificado localmente (ver §"Portão do split") |
| `EXIST_VIDEOS/improvements/05_IMPLEMENTATION_NOTES.md` | este arquivo |
| `.gitignore` (3 linhas acrescentadas) | exceção mínima para que o split congelado seja versionável — ver Decisão 9 |

**Nenhum arquivo original foi modificado.** `Thesis_video_fixed.ipynb`,
`extract_video.ipynb`, `extract_audio.ipynb`, `inference_video.py`,
`ablation_harness.py` e os relatórios `0X_*.md` estão intocados.

## Verificado localmente (fato, não afirmação)

```
$ python3 -m py_compile video_core.py           → OK
$ python3 -c "json.load(...); nbformat.validate(...)" → OK, nbformat 4.5, sem warnings
                                                   17 células de código, todas parseiam com ast
$ python3 -m video_core (self-checks, com pyevall stubado)
  ✓ convenção Task A (sigmoid = P(YES), eixo [NO, YES]) verificada
  ✓ fusion_input_dim: 2688 com bio (B1/B2), 2560 sem bio (B0)
  ✓ modality dropout: taxas por modalidade ≈ [0.0503, 0.0495, 0.0499, 0.0501] (p=0.05)
$ formato hard da Task C (ARCH-10), 3 casos:
  categorias batem → ['IDEOLOGICAL-INEQUALITY', 'OBJECTIFICATION']   (sem "NO" junto)
  só NO bate       → ['NO']
  nada bate        → ['IDEOLOGICAL-INEQUALITY']                      (fallback, não-NO)
$ build_frozen_split sobre EXIST2026_training.json (2 524 linhas) → 2019 / 505, portão OK
```

`peft` e `pyevall` **não existem na máquina local** (só na VM). `peft` é importado dentro
das funções que o usam; `pyevall` é import de topo e foi stubado só no script de
verificação — o módulo em si continua honesto.

**Nada foi treinado. Nenhum número de métrica deste trabalho existe.** Tudo que exige GPU
está na seção "Verificação pendente na VM".

---

## 1. Mapa finding → implementação

### Arquitetura

| Finding | Onde | Nota |
|---|---|---|
| **ARCH-01** q_bias/v_bias | `video_core.restore_videomae_qv_biases()`, chamada por `build_backbones()` **antes** de `get_peft_model` | Falha **alto** com o comando de diagnóstico na mensagem se o caminho de atributo estiver errado (é o item não verificado nº 1). Ressalva ARCH-01 × OPT-03 resolvida em `load_run()`, que reconstrói via `build_backbones(restore_biases=True)` — os biases são *frozen*, não estão no checkpoint enxuto, e `meta.json` grava `"videomae_qv_biases": "restored_at_build_not_in_checkpoint"` |
| **ARCH-02** | `MultiTaskTrainer.__init__` → `self.model_accepts_loss_kwargs = False` | Patch defensivo; a hipótese de LR 16× continua refutada. Célula 7 do notebook imprime `"labels" in sample` (tem de ser False) |
| **ARCH-03 (b)** | `EXISTVideoModel.__init__` — sem `cross_attn`/`cross_proj`/`cross_norm` | Critério de reabertura `probe_zero_video` e a **assimetria de leitura** estão registados em comentário no próprio `__init__`. Nenhum caminho de máscara em tempo de avaliação foi criado (seria reintroduzir o padrão de flags do ARCH-20) |
| **ARCH-05** | `MAX_LENGTH = 256` | 78 % das amostras cobertas integralmente; 384 custaria ~+49 % de compute contra ~+22 % |
| **ARCH-06** | `NUM_TRAIN_EPOCHS_PROVISIONAL = 6` / `CALIBRATION_EPOCHS = 15` | Marcado PROVISÓRIO num bloco de comentário longo; critério `E = argmin(eval_loss)` documentado ali e na célula 12 do notebook, que calcula E a partir de `trainer.state.log_history`. `load_best_model_at_end=False`; `metric_for_best_model` **deliberadamente não definido** |
| **ARCH-07** | `EXISTVideoModel.forward` — `band = clamp(floor(r/p), max=4)` | Por amostra, faixas fixas posicionais. Invariante (taxa por modalidade = `p` em todos os braços) verificada por Monte-Carlo em `selfcheck_modality_dropout()` |
| **ARCH-08** | sem `bio_ln`, sem `BatchNorm1d` | `cvpc` entra no `cat` junto com `bio_in`, como antes |
| **ARCH-10** | `_format_for_pyevall`, ramo `'2_3'` hard | Nunca mais emite `NO` junto com categorias. `NONE` excluído do grid em `_search_thresholds_task_c` (loop começa em 1). Comentário justifica **só** por corretude — a economia de chamadas foi retirada do argumento (§4) |
| **ARCH-11 (b)** | `split_val_halves()` + `tune_task_c_thresholds()` | Ajusta numa metade, reporta na outra; o JSON grava `scores_on_report_half` (o número reportável), `scores_on_tune_half` e o baseline 0,5 na metade de reporte |
| **ARCH-12 / SIMP-02 / BUG-A** | `run_dir()` / `ckpt_dir()` derivados de (condição, INIT_SEED); células 13–14 do notebook | A validação de checkpoint agora lê o que a gravação escreveu |
| **ARCH-13 / SIMP-01** | `condition_tag()` — só 3 condições | `FULL_bio_viewer_tau` removido |
| **ARCH-14** | `_extract_metrics_from_report` e o `except` de `_run_pyevall` devolvem `nan`; `_proxy_metrics()` | Proxies sklearn (`f1_macro` A/B/C + `roc_auc` A) com prefixo `proxy_` — diagnóstico, nunca paper |
| **ARCH-15 / SIMP-08** | notebook, célula 9: `warmup_steps=0.1` | Comentado explicitamente como **durabilidade, não bugfix** — `warmup_ratio` funciona em 5.6.0 |
| **ARCH-16** | SWA **removido** | Bloco de comentário no §14 do módulo com os 3 motivos (cauda de overfitting, `swa_lr` nunca usado, 2,6 GB de VRAM) |
| **ARCH-17.3** | imports mortos fora; `cross_ids` fora; guarda `labels_a.ndim == 2` mantida com comentário de herança | |
| **ARCH-17.4** | `get_optimizer()` — grupo `base_no_decay` para biases e LayerNorm | |
| **ARCH-19** | `record_fingerprint()` grava e **compara** com `RUN_DIR/fingerprints.json` | Chamado em "init" e "final" no notebook. Comentário avisa que fingerprints só comparam dentro do mesmo par (braço, seed) |
| **ARCH-20 / OPT-13** | `use_bio`/`use_viewer_emb` são **kwargs do construtor**; `fusion_input_dim(use_bio)` → 2560/2688 | Regra escrita em comentário no `__init__`: **proibido normalizar o vetor concatenado** enquanto qualquer braço zerar slot. A justificativa correta (auditabilidade + blindagem, **não** "os zeros distorcem as estatísticas") está no docstring de `fusion_input_dim` |

### Otimização / protocolo

| Finding | Onde |
|---|---|
| **OPT-03** | `trainable_state_dict()` / `save_run()` / `load_run()`; `save_strategy="no"` no notebook |
| **OPT-05** | `dataloader_persistent_workers=True`, `pin_memory=True`; `num_workers=2` mantido com nota (OPT-05 sugere 8, depende de `nproc` na VM) |
| **OPT-06** | `SPLIT_SEED` / `INIT_SEED` separados; `build_frozen_split()` com artefato em disco |
| **OPT-08** | caminho único de avaliação: `trainer.evaluate()` / `trainer.predict()`. Não há loop de inferência paralelo neste código |
| **OPT-11** | `VAL_FRACTION = 0.20` |
| **OPT-12** | `frames_dir()`/`audio_dir()` com nota sobre árvore versionada se houver re-extração |

### Simplificação / divergências

| Finding | Onde |
|---|---|
| **SIMP-03** | o módulo inteiro é a extração pedida |
| **SIMP-05** | `compute_metrics(eval_pred)` — um argumento; o ramo `tune_thresholds_c` deixou de existir |
| **SIMP-06** | `build_bio_vector` devolve `(bio_vec, cvpc, user_ids)`; dataset e `forward` não emitem/aceitam `rt_norm`/`fix_norm` |
| **SIMP-07** | `_metrics_task_a/_b/_c` separados, com a ordem de inserção assimétrica preservada e comentada |
| **SIMP-09** | `video_collate()` à prova de campos não-tensoriais |
| **SIMP-12** | imports mortos removidos; ausências deliberadas listadas em comentário no topo |
| **SIMP-13** | `forward` re-indentado |
| **SIMP-14** | `BIO_MLP_HIDDEN`, `BIO_DROPOUT`, `FUSION_DROPOUT`, `MODALITY_DROPOUT_P`, `LOSS_W_*`, `make_lora_config()` |
| **DIV-01** | dataset devolve sempre `id_EXIST` |
| **DIV-02** | flags no construtor + `meta.json` → carregar B1 numa arquitetura B2 quebra alto |
| **DIV-03 / BUG-C** | `TEST_CASE = "EXIST2026"`, parametrizável em `_format_for_pyevall(test_case=...)`, com ⚠ de confirmar contra o gold oficial |
| **DIV-04** | sem arredondamento das probabilidades soft |
| **DIV-06** | `_reaction_time()` — semântica de `collect_bio_vectors` (não-castável → NaN → imputa), fonte única para os dois call sites; contagem de imputações impressa |
| **DIV-07** | `LABELS_2_1` **removida**; `TASK_A_PROB_ORDER = ("NO","YES")` indexada em `_format_for_pyevall`; `selfcheck_task_a_convention()` amarra loss e formatter |
| **BUG-B / ARCH-18** | não aplicável a estes arquivos; `load_run` avisa alto se o `scaler.pkl` faltar, e o docstring de `collect_bio_vectors` registra a regra |
| **BUG-G** | nenhuma ação — impacto exatamente zero, verificado (0 amostras com todos os votos UNKNOWN nas 3 tasks) |

---

## 2. Findings NÃO implementados, e por quê

| Finding | Motivo |
|---|---|
| **ARCH-04** (áudio: janelas múltiplas, média só do válido) | Exige **re-extração**, e vive em `extract_audio.ipynb`, que não posso modificar. `load_audio_features` documenta o defeito em docstring. ~10 min de GPU, mas invalida todas as features atuais |
| **ARCH-09** (decode sequencial, dupla conversão BGR2RGB) + **OPT-01** (fp16/uint8) | Idem, em `extract_video.ipynb`. `load_frames` já faz `.float()` no retorno, então é **agnóstico ao dtype de armazenamento** — quando a re-extração acontecer, nada no treino muda |
| **SIMP-04** (constante `SPLIT` + `_extraction_meta.json`) | Mesmos arquivos de extração |
| **BUG-D/BUG-E** (zeros no disco mascarando vídeos faltantes) | Mesma razão; documentado em docstring de `load_frames` |
| **OPT-04** (batch 8 / sem gradient checkpointing) | `[MUDA]` e depende de VRAM não medida e do OPT-01. Fica como bloco **comentado** na célula 9 do notebook, com a célula de benchmark 9.1 ao lado. Ver "Decisão 2" |
| **OPT-02 (ii)** (subir rank/módulos do LoRA) | Explicitamente "no máximo um braço de ablação, não uma mudança de baseline" |
| **OPT-09** (triagem com backbones congelados) | Infraestrutura separada; e não pode ser usada para decidir a remoção do `bio_ln`, que é decisão de desenho |
| **ARCH-17.1** (unificar sentinelas) | **Decisão consciente de não fazer** — ver "Decisão 1" |
| **ARCH-17.2** (`sum` vs `mean` na Task C) | Hipótese, não bug. Documentado no docstring de `loss_task_c` |
| **BUG-F** (pré-processamento de texto) | "Nenhuma ação nesta ronda" por decisão do §7.2 |
| **SIMP-16** (`CLAUDE.md`, TikZ, paper) | Fora de escopo de código. A linha correta passa a ser **"2688→1024 (B1/B2); 2560→1024 (B0)"**, e o diagrama não pode mais mostrar a cross-attention |
| **`inference_video.py` e `ablation_harness.py`** | A regra 1 proíbe modificá-los. **Consequência importante:** ver "O que este código invalida", item 6 |
| **OPT-07 (b)** (análise pareada) | O código de agregação vive no harness. O procedimento está escrito na célula 16 do notebook |

---

## 3. Decisões que tive de tomar

**Decisão 1 — sentinelas ficam assimétricas (`-100.0` para A, `-1.0` para B/C).**
O ARCH-17.1 propõe unificar em `-100.0` e o Agente 3 concordou, mas isso não está na
lista obrigatória e a reconciliação não o menciona. Unificar muda **quais linhas são
mascaradas**, e `inference_video.py` — que eu **não posso tocar** — ainda constrói o gold
com a convenção antiga. Uma segunda incompatibilidade silenciosa entre os dois arquivos
custa mais do que o ganho de estilo. Estão nomeadas (`SENTINEL_A`, `SENTINEL_BC`) e usadas
em toda parte, para que a unificação futura seja uma edição em dois lugares.

**Decisão 2 — batch fica em 2 × 16 com gradient checkpointing.**
O OPT-04 é `[MUDA]`, o Agente 3 titulou "medir antes de fixar", declarou não saber a VRAM
disponível e amarrou-o ao OPT-01 (fora do meu escopo). O §7.3 exige congelar **antes** da
calibração, não adivinhar agora. Fica a configuração que sabidamente roda, com o
candidato do OPT-04 num bloco comentado e a célula de benchmark ao lado. Batch efetivo é
32 nas duas — trocar depois toca uma célula.

**Decisão 3 — `dataloader_num_workers=2` mantido.** O OPT-05 sugere 8 "ajustar ao nº de
vCPUs da VM — não sei quantas são". Mantive 2 e pus os dois ganhos gratuitos e sem risco
(`persistent_workers`, `pin_memory`).

**Decisão 4 — `video_collate` obrigatório.** Emitir `id_EXIST` (DIV-01) e
`remove_unused_columns=False` são incompatíveis com o `default_data_collator` do HF, que
faria `torch.tensor` sobre uma lista de strings. O collate custom é a única forma de ter
os dois. **Não verifiquei** que o `_prepare_inputs` de `transformers==5.6.0` deixa passar
campos não-tensoriais sem tocar — é comportamento esperado, não medido (pendente nº 6).

**Decisão 5 — portão do split com contagem permissiva.** O §6.3 diz "contagem de
positivos por categoria". Implementei **duas** contagens: `algum voto` (proporção > 0) e
`maioria` (≥ 0,5, que é a regra de binarização do gold hard). O portão gateia na
permissiva (leitura literal do §6.3) e **imprime as duas**. Na prática ambas passaram com
folga. O portão **recusa-se a gravar** o artefato se falhar — é o que o faz portão.

**Decisão 6 — `NONE` fixo em 0,5 na busca de limiares**, em vez de removido da lista.
A lista continua com 6 posições (é a ordem de `LABELS_2_3`, partilhada com o formatter);
só o índice 0 sai do grid.

**Decisão 7 — `peft` importado dentro das funções.** Mantém o custo de importar
`video_core` baixo e a dependência explícita. `pyevall` continua no topo, como em
`memes_core.py`.

**Decisão 9 — exceção no `.gitignore` para o split congelado.**
`Dataset/` inteiro estava ignorado, então gravar `Dataset/val_split_42.json` **não o
tornava commitável** — e o §6.2 exige explicitamente commitar, porque é isso que faz de
"as 9 runs partilham o mesmo val" uma afirmação verificável em vez de uma promessa.
Acrescentei três linhas, com o efeito verificado:

```
!*/Dataset/
*/Dataset/*
!*/Dataset/val_split_*.json
```

`git ls-files --others --exclude-standard EXIST_VIDEOS/Dataset EXIST_MEMES/Dataset`
devolve **exatamente um arquivo**: `EXIST_VIDEOS/Dataset/val_split_42.json`. Os JSONs de
treino/test, os `val_seed_*.json` antigos e os vídeos continuam ignorados. Foi a
alteração mínima; a alternativa seria mudar o caminho que a especificação nomeia.

**Decisão 8 — `save_strategy="no"`.** Das duas opções do OPT-03, escolhi a que remove IO
inútil por completo. Numa run de ~22 min a perda de recuperação de falha é aceitável, e o
salvamento enxuto na seção 13 é o consumidor real.

---

## 4. Portão do split (§6.3) — **descarregado**

Rodado localmente sobre `Dataset/training/EXIST2026_training.json` (2 524 linhas),
`SPLIT_SEED=42`, `VAL_FRACTION=0.20`. Passou **na primeira tentativa**, com
estratificação só por Task A — a estratificação conjunta não foi necessária:

```
[split] positivos por categoria da Task C no val (n=505):
  NONE                          algum voto:  329 | maioria:  261
  IDEOLOGICAL-INEQUALITY        algum voto:  122 | maioria:   97
  MISOGYNY-NON-SEXUAL-VIOLENCE  algum voto:   47 | maioria:   37
  SEXUAL-VIOLENCE               algum voto:   31 | maioria:   28
  OBJECTIFICATION               algum voto:   59 | maioria:   48
  STEREOTYPING-DOMINANCE        algum voto:  205 | maioria:  163
```

Split: **2 019 treino / 505 val**. Artefato gravado em `Dataset/val_split_42.json` e
recarregado com sucesso pelo caminho de leitura.

> **A partir daqui o split está congelado. COMMITAR o arquivo.** Qualquer mexida
> posterior é split-shopping e invalida todas as runs.

---

## 5. Verificação pendente na VM

Em ordem. Nenhuma decisão da tabela de ablação deve ser tomada antes de 1–6.

| # | O quê | Comando | Resultado esperado |
|---|---|---|---|
| **0** | **Copiar `Dataset/val_split_42.json` para a VM** — o artefato foi criado e verificado LOCALMENTE, e `Dataset/` não é sincronizado. Gateia todas as runs | célula 4 do notebook (tem um `assert` que falha cedo se o arquivo faltar) | `[split] carregado de .../val_split_42.json ... 2019 treino / 505 val`. Se imprimir **`[split] artefato gravado`**, o split foi RECALCULADO na VM e a run está **fora de protocolo** |
| 1 | **Caminho de atributo do ARCH-01** (o item mais crítico: se estiver errado, `build_model` levanta `RuntimeError`) | `videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")`<br>`print([n for n,_ in videomae.named_parameters() if 'layer.0.attention' in n])` | Aparecerem `...attention.attention.query.bias` / `.value.bias`. Se aparecer `qkv`/`q_bias`, ajustar `restore_videomae_qv_biases`. **Checar também se uma versão mais nova do `transformers` já mapeia certo** — atualizar a lib é mais durável que o remendo |
| 2 | O remendo de facto rodou | `vc.build_model(42)` | `[fix ARCH-01] q_bias/v_bias restaurados em 12 camadas do VideoMAE` |
| 3 | **Contagem de params por braço** (§1.3 manda verificar, não assumir) | `for ub, uv in [(False,False),(True,False),(True,True)]: print(vc.condition_tag(ub,uv), vc.count_params(vc.build_model(42, use_bio=ub, use_viewer_emb=uv)))` | ≈ 5,88 M (B0) e ≈ 6,03 M (B1/B2) treináveis. **Reconciliar com a tabela do §1.4 antes de qualquer número ir para a tese** |
| 4 | ARCH-02 continua inócuo em 5.6.0 | `print(trainer.model_accepts_loss_kwargs)` e `print("labels" in train_dataset[0])` | `False` (travado) e `False` |
| 5 | Equivalência `warmup_ratio` → `warmup_steps` | célula 10 do notebook imprime `training_args.get_warmup_steps(total_steps)` | ~10 % dos steps totais |
| 6 | **Um forward completo** (é o teste que exercita collate + máscaras + fusão) | `vc.forward_sanity_check(model, train_dataset)` | `fusion_in` 2688/2560 conforme a condição; `logits_a [1]`, `logits_b [1,3]`, `logits_c [1,6]`; sem erro no `_prepare_inputs` com `id_EXIST` |
| 7 | **Benchmark do batch (OPT-04)** e congelamento | célula 9.1 do notebook, `max_steps=30` × {bs 2, 8} × {ckpt on/off} | escolher, escrever na célula 9 e **não mexer mais** |
| 8 | Buffers do modelo (ARCH-16.2, agora só curiosidade) | `print([n for n,_ in model.named_buffers()])` | esperado: nada relevante |
| 9 | **Run de calibração** — 15 épocas, `INIT_SEED=42`, sistema pós-correção | `NUM_EPOCHS = vc.CALIBRATION_EPOCHS`, Run All | célula 12 imprime `E = argmin(eval_loss)`. Congelar E, registar as 6 métricas em E e E±2 |
| 10 | Determinismo por braço (ARCH-19) | rodar 2× a mesma (condição, seed) e ler `RUN_DIR/fingerprints.json` | `✓ idêntico à run anterior` em "init" e "final" |
| 11 | Round-trip do checkpoint enxuto | célula 14 | `✓ Checkpoint válido` |
| 12 | **`test_case` (DIV-03/BUG-C)** | abrir o gold oficial do EXIST 2026 e copiar a string literal | confirmar `TEST_CASE`. **Não deduzir por analogia com 2025** |
| 13 | Imputações de RT no split de **test** (DIV-06) | `vc.collect_bio_vectors(test_df)` | contagem impressa; se for alta, investigar antes de submeter |
| 14 | Tabela de ablação | 9 runs (3 condições × 3 `INIT_SEED`) | ~4,1–5,0 h. Análise: **delta pareado por seed + consistência de sinal** |
| 15 | `probe_zero_video` (§2.3) | depois das 9 runs, sobre checkpoints já treinados | **análise separada, nunca na tabela**; leitura assimétrica |

---

## 6. O que este código invalida

Tudo abaixo deixa de ser comparável com qualquer número novo. **Não misturar as duas
gerações na mesma tabela do paper.**

1. **Todos os checkpoints** — `best_model_seed_*`, `best_model_seed_*_cross_att`,
   `checkpoints_seed_*`. A arquitetura mudou (sem cross-attention, sem `bio_ln`, fusão de
   largura variável), os `q_bias` mudaram, o `max_length` mudou e o split mudou. Nenhum
   carrega neste código, e é bom que não carregue.
2. **`ablation_results.csv`, `ablation_results.tex`, `ablation_seed_{42,7,2025}.csv`** —
   além dos pesos, eram sondas de zeragem de input, não ablações treinadas (ARCH-20), e o
   `.tex` continha um braço fantasma e números de uma seed só (SIMP-01).
3. **`task_c_thresholds_seed_{42,7,2025}.json`** — triplamente: pesos novos, o formato
   hard do gold mudou (ARCH-10) e o ajuste passou a ser em meia-val (ARCH-11 b).
4. **A tabela de treino da célula 52 e a avaliação da célula 55** do notebook antigo.
5. **`predictions/cross_att/` e `predictions/swa/`** — são as submissões de um sistema que
   deixa de existir. Os resultados oficiais da competição derivados delas **não podem ser
   regenerados** e **não podem partilhar tabela** com números pós-correção. Na tese têm de
   aparecer como *"sistema submetido"*, separados de *"sistema corrigido"*.
6. **`inference_video.py` está agora DESSINCRONIZADO deste código** e **não deve ser
   usado** com checkpoints novos até ser reescrito sobre `video_core`. Divergências ativas:
   ele reimplementa `EXISTVideoModel` **com** cross-attention e **com** `bio_ln`, espera
   `rt_norm`/`fix_norm` do dataset, usa `max_length=128`, não tem as flags de condição, tem
   `--scaler` opcional (BUG-B) e reimplementa o formato hard da Task C sem a correção do
   ARCH-10. `load_state_dict` estrito vai falhar alto — o que é a proteção funcionando —,
   mas o arquivo tem de ser reescrito antes de qualquer submissão nova.
7. **`ablation_harness.py`** perde a razão de existir na forma atual: com as flags no
   construtor, `run_inference`/`run_all_ablations`/`run_ablations_multi_seed` devem ser
   substituídos por um caminho único de avaliação (`trainer.evaluate()` por checkpoint) mais
   uma função de agregação com delta pareado por seed. `ABLATION_CONDITIONS` ainda contém o
   braço fantasma.
8. **`CLAUDE.md`, o diagrama TikZ e a seção de arquitetura do paper** — a linha da Task 3
   passa a ser **"2688→1024 (B1/B2); 2560→1024 (B0)"**, e o diagrama não pode mais mostrar
   a cross-attention.
