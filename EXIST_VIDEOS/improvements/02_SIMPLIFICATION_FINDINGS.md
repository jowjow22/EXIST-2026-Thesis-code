# Agente 2 — Simplificação (DRY / KISS) — modelo de vídeo, Task 3

**Escopo:** duplicação real, complexidade acidental, consistência interna,
constantes/configuração, estrutura extraível, pipelines de extração.
**Restrição inegociável:** cada proposta é classificada como
**`[EQUIV]` equivalente numericamente** ou
**`[MUDA]` muda comportamento — requer aprovação do Agente 1**.
Na dúvida, classifiquei como `[MUDA]`.
**Severidade** (regra 3 do brief): ALTA = afeta resultado publicado ou
corretude; MÉDIA = manutenção/risco latente; BAIXA = estilo.

**Arquivos analisados** (versões limpas em `scratchpad/video_code/`, mapeadas
para os originais):
`Thesis_video_fixed_CODE.py` (células citadas = células do
`Thesis_video_fixed.ipynb`), `inference_video.py` e `ablation_harness.py`
(citados por **número de linha**, pois não têm células),
`extract_video_CODE.py`, `extract_audio_CODE.py`.

---

## Sumário executivo — as 5 de maior valor

| # | ID | Uma linha |
|---|----|-----------|
| 1 | SIMP-01 | A tabela de ablação do paper (`ablation_results.tex`) reporta uma condição **que nunca foi executada** (`FULL` tem flags idênticas a `B2`; `use_tau` não existe no modelo) e contém números de **uma única seed** (2025), não de uma agregação. |
| 2 | SIMP-02 | A célula 54 ("validação do checkpoint") **crashou com `RuntimeError`** porque grava em `best_model_seed_{SEED}_cross_att` e lê de `best_model_seed_{SEED}` — o checkpoint salvo nunca foi validado. |
| 3 | SIMP-03 | ~450 das 794 linhas de `inference_video.py` são cópia do notebook de treino (constantes, bio, dataset, modelo, PyEvALL), já com divergências reais — candidato direto ao mesmo padrão `memes_core.py`. |
| 4 | SIMP-04 | Os scripts de extração estão commitados na configuração de **test**, mas as docstrings dizem **train**: a versão que gerou os tensores de treino não existe mais no repositório (lacuna de proveniência). |
| 5 | SIMP-16 | As dimensões da fusão registradas no `CLAUDE.md` (`2592`, bio 32-d, texto/áudio trocados) **não correspondem ao código** (`2688`, bio 128-d) — verificar se o paper e o diagrama TikZ herdaram os números errados. |
| 6 | SIMP-05 | `compute_metrics(..., tune_thresholds_c=False)` — o ramo de busca de threshold **nunca executa** (todos os chamadores passam 1 argumento). É por isso que o modelo de vídeo escapou do vazamento de validação que a auditoria de memes encontrou — mas por acidente de argumento default, não por construção. |

Ordenação geral: SIMP-01, SIMP-02, SIMP-04 e SIMP-16 têm impacto em
resultado/paper; SIMP-03/05 são estruturais; SIMP-06 em diante são DRY/KISS
puros. **Os três achados mais graves (SIMP-01, SIMP-02, SIMP-16) são, tecnicamente,
findings de configuração/código morto — e é exatamente por isso que passaram
despercebidos numa revisão de arquitetura.**

---

## Findings detalhados

### SIMP-01 — Condição de ablação fantasma + tabela do paper de uma seed só
**Tipo:** código morto + consistência (impacto em publicação)  
**Severidade:** ALTA  
**Local:** `ablation_harness.py` linhas 33–38 (registry), 272–281
(`save_for_paper`); artefatos `ablation_results.csv`, `ablation_results.tex`,
`ablation_seed_{42,7,2025}.csv`; consumidor em `Thesis_video_fixed_CODE.py`
célula 59.

```python
ABLATION_CONDITIONS = OrderedDict([
    ("B0_content_only",       {"use_bio": False, "use_viewer_emb": False}),
    ("B1_bio_no_viewer",      {"use_bio": True,  "use_viewer_emb": False}),
    ("B2_bio_viewer_no_tau",  {"use_bio": True,  "use_viewer_emb": True}),
    ("FULL_bio_viewer_tau",   {"use_bio": True,  "use_viewer_emb": True}),
])
```

**Problema (fatos verificados):**
1. `B2` e `FULL` têm **dicionários de flags idênticos**. O docstring do módulo
   (linhas 10–14) descreve `use_tau`, mas `EXISTVideoModel.forward`
   (célula 28) não tem esse parâmetro — se fosse passado, seria engolido por
   `**kwargs` silenciosamente. A rede-τ foi removida do projeto (ver
   `CLAUDE.md`), o registry não acompanhou.
2. Consequência verificável nos dados: as linhas `B2` e `FULL` são
   **bit-idênticas nas três seeds** (conferido em `ablation_seed_42.csv`,
   `ablation_seed_7.csv`, `ablation_seed_2025.csv`). Ex. seed 42,
   `task_c_icmsoftnorm_soft` = `0.1395275616815516` nas duas linhas.
3. `save_for_paper(df)` (célula 59) usa os defaults
   `csv_path="ablation_results.csv"`, `latex_path="ablation_results.tex"`.
   Como cada seed roda a mesma célula, **a última execução sobrescreve**:
   `ablation_results.csv` é byte-a-byte igual a `ablation_seed_2025.csv`.
   `ablation_results.tex` — o artefato que entra no paper — carrega tanto a
   linha fantasma `FULL` quanto números de uma seed só, com aparência de
   resultado consolidado.

**Por que importa:** é o achado de maior impacto do meu escopo, e não é de
estilo — é integridade do resultado publicado. Equivale ao padrão #4 do brief
("rotulagem enganosa de ablação") encontrado na auditoria de memes.

**Proposta:** remover a condição fantasma, renomear para o que de fato varia, e
tornar o caminho de saída dependente da seed.

```python
# ablation_harness.py — substitui o registry (linhas 33-38)
# Só existem 3 condições reais: a rede-τ foi removida do projeto e nunca teve
# parâmetro em EXISTVideoModel.forward. Um 4º braço com flags idênticas a B2
# produzia uma linha duplicada na tabela do paper.
ABLATION_CONDITIONS = OrderedDict([
    ("B0_content_only",  {"use_bio": False, "use_viewer_emb": False}),
    ("B1_bio_no_viewer", {"use_bio": True,  "use_viewer_emb": False}),
    ("B2_bio_viewer",    {"use_bio": True,  "use_viewer_emb": True}),
])


def save_for_paper(df, csv_path, latex_path=None):
    """Grava a tabela. `csv_path` é OBRIGATÓRIO e deve conter a seed — o
    default anterior fazia cada seed sobrescrever o arquivo da anterior,
    e o .tex do paper acabou sendo um retrato de uma única seed."""
    if latex_path is None:
        latex_path = str(csv_path).rsplit(".", 1)[0] + ".tex"
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV salvo em {csv_path}")
    try:
        with open(latex_path, "w") as f:
            f.write(df.to_latex(index=False, escape=True, float_format="%.3f"))
        print(f"✓ LaTeX salvo em {latex_path}")
    except Exception as e:
        print(f"  ! Export LaTeX falhou: {e}")
```
e na célula 59: `save_for_paper(df, csv_path=f"ablation_seed_{SEED}.csv")`.

**Classificação:** `[EQUIV]` para os números por condição (B0/B1/B2 não mudam;
a linha removida era uma cópia de B2). **`[MUDA]` para o paper**: a tabela
publicada perde uma linha e precisa ser regerada a partir das 3 seeds — o
agregado multi-seed (`run_ablations_multi_seed` / `print_multi_seed_table`,
linhas 168–266) já existe e nunca foi usado. **Requer decisão do Agente 1 e
provavelmente correção no texto do paper.**

---

### SIMP-02 — Diretório de checkpoint divergente entre salvar e validar (célula 53 vs 54)
**Tipo:** constante duplicada / consistência (bug confirmado)  
**Severidade:** ALTA  
**Local:** `Thesis_video_fixed_CODE.py` células 53 e 54.

```python
# célula 53 — grava
os.makedirs(f"./best_model_seed_{SEED}_cross_att", exist_ok=True)
safetensors.torch.save_file(state_dict, f"./best_model_seed_{SEED}_cross_att/model.safetensors")

# célula 54 — valida... outro diretório
state_dict_check = safetensors.torch.load_file(
    f"./best_model_seed_{SEED}/model.safetensors", device="cpu"
)
model_check.load_state_dict(state_dict_check)
```

E ainda um terceiro uso do prefixo: a célula 17 grava o `scaler.pkl` em
`./best_model_seed_{SEED}/` (sem sufixo), enquanto a célula 53 grava outra
cópia do mesmo scaler em `..._cross_att/`.

**Fato verificado (não hipótese):** a célula 54 do
`Thesis_video_fixed.ipynb` tem output do tipo `error`:

```
RuntimeError: Error(s) in loading state_dict for EXISTVideoModel:
    Missing key(s) in state_dict: "cross_attn.in_proj_weight",
    "cross_attn.in_proj_bias", "cross_attn.out_proj.weight",
    "cross_attn.out_proj.bias", "cross_proj.weight", "cross_proj.bias",
    "cross_norm.weight", "cross_norm.bias".
```

Ou seja: leu um checkpoint **antigo, anterior à cross-attention**. A célula que
existe justamente para provar que o checkpoint é válido nunca chegou a rodar as
comparações — nenhum `OK:` / `MISMATCH` foi impresso. Nenhuma verificação de
integridade do modelo salvo aconteceu, e ainda ficam dois checkpoints diferentes
no disco com nomes parecidos.

**Proposta:** uma única constante derivada da seed, usada por todos os
consumidores (incluindo `inference_video.py --model-dir`).

```python
# célula 5 (bloco de constantes globais) — fonte única de verdade
MODEL_DIR = f"./best_model_seed_{SEED}"   # scaler.pkl + model.safetensors + tokenizer
CKPT_DIR  = f"./checkpoints_seed_{SEED}"  # checkpoints intermediários do Trainer
```
e trocar TODAS as ocorrências (células 17, 33, 53, 54) por `MODEL_DIR` /
`CKPT_DIR`. Se o sufixo `_cross_att` for informação de experimento, ele deve
entrar na constante (`MODEL_DIR = f"./best_model_seed_{SEED}_cross_att"`), não
ser digitado em uma célula e esquecido na outra.

**Classificação:** `[EQUIV]` — nenhum peso ou métrica muda; o que muda é que a
validação passa a rodar de fato (e pode acusar um problema real, aí é do
Agente 1). Ver "Bugs encontrados" → BUG-A.

---

### SIMP-03 — `inference_video.py` é uma cópia manual do notebook (≈450 de 794 linhas)
**Tipo:** DRY (o mais caro do arquivo)  
**Severidade:** MÉDIA  
**Local:** `inference_video.py` linhas 60–98 (constantes), 103–205 (bio),
210–225 (loaders), 230–272 (soft labels), 277–334 (dataset), 339–410 (modelo),
420–463 (`format_records`), 521–555 (`_run_pyevall`).

Blocos duplicados verbatim ou quase-verbatim a partir de
`Thesis_video_fixed_CODE.py`:

| Bloco | Notebook | inference_video.py | Estado |
|---|---|---|---|
| `LABELS_2_*`, hierarquias | células 1, 38 | 62–70 | idêntico |
| Constantes de dimensão | célula 5 | 72–98 | idêntico em valor, comentários perdidos |
| `safe_float`, `build_bio_vector`, `collect_bio_vectors` | células 8, 9, 14 | 103–205 | idêntico |
| `load_frames`, `load_audio_features` | células 6, 7 | 210–225 | idêntico (só mensagens de WARN diferem) |
| `build_soft_label_a/b/c` | células 10–12 | 230–272 | idêntico |
| `EXISTVideoDataset` | célula 15 | 277–334 | **divergente** (ver DIV-01) |
| `TemporalAttentionPool`, `EXISTVideoModel` | célula 28 | 339–410 | **divergente** (ver DIV-02) |
| `_format_for_pyevall` → `format_records` | célula 37 | 420–463 | **divergente** (ver DIV-03, DIV-04) |
| `_run_pyevall` + extração de métricas | células 39, 40 | 521–555 | **divergente** (assinatura e inline) |

**Por que importa:** já divergiu em 5 pontos (seção "Divergências"). Toda
mudança de arquitetura proposta pelo Agente 1 precisa hoje ser aplicada duas
vezes, e a segunda aplicação não tem teste que a cubra.

**Proposta:** extrair `video_core.py` (desenho concreto na seção "Desenho
proposto de módulo extraível") e reduzir `inference_video.py` a: CLI, loop de
inferência, gravação dos JSONs. Estimativa: 794 → ~300 linhas.

**Classificação:** `[EQUIV]` para os blocos idênticos (é mover código sem
alterá-lo). **`[MUDA]` para cada bloco divergente** — unificar obriga a
escolher uma das duas versões, e essa escolha é do Agente 1. Não unifique
"pegando a versão do notebook" por default sem ler DIV-01..DIV-05.

---

### SIMP-04 — Scripts de extração commitados em configuração de *test*, docstring dizendo *train*
**Tipo:** configuração / consistência (lacuna de proveniência)  
**Severidade:** ALTA  
**Local:** `extract_video_CODE.py` células 0–1 (linhas 24–28);
`extract_audio_CODE.py` célula 0 (linhas 22–27).

```python
# extract_video: docstring diz "→ frames/train/{id_EXIST}.pt"
VIDEO_DIR   = Path("./Dataset/test/videos/")
OUTPUT_DIR  = Path("./frames/test/")
JSON_PATH   = "./Dataset/test/EXIST2026_test_clean.json"
```
```python
# extract_audio: docstring diz "Saves → audio_features/train/{id_EXIST}.pt"
VIDEO_DIR   = Path("./Dataset/test/videos/")
OUTPUT_DIR  = Path("./audio_features/test/")
JSON_PATH   = "./Dataset/test/EXIST2026_test_clean.json"
```

**Problema:** os tensores de treino (`frames/train/`, `audio_features/train/`)
foram produzidos por uma versão editada à mão desses arquivos que **não existe
mais no repositório**. Ninguém consegue verificar hoje que train e test foram
extraídos com `NUM_FRAMES`, normalização e `TARGET_SR` idênticos — e um
descasamento aqui é exatamente o tipo de bug de pré-processamento que o brief
aponta como candidato forte para explicar performance ruim. Não é arrumação: é
fechar uma lacuna de proveniência.

**Proposta:** um único `SPLIT` com caminhos derivados, nos dois scripts.

```python
# ── Config ───────────────────────────────────────────────────────────────
# Trocar SPLIT é a ÚNICA edição necessária entre train e test. Antes, cada
# caminho era editado à mão, e a versão que gerou os tensores de treino não
# ficou registada em lado nenhum.
SPLIT       = "train"                     # "train" | "test"
JSON_PATH   = {"train": "./Dataset/training/EXIST2026_training.json",
               "test":  "./Dataset/test/EXIST2026_test_clean.json"}[SPLIT]
VIDEO_DIR   = Path({"train": "./Dataset/training/videos/",
                    "test":  "./Dataset/test/videos/"}[SPLIT])
OUTPUT_DIR  = Path(f"./frames/{SPLIT}/")          # audio: f"./audio_features/{SPLIT}/"
NUM_FRAMES  = 16
MODEL_NAME  = "MCG-NJU/videomae-base"
```
E gravar um `OUTPUT_DIR/_extraction_meta.json` com `{SPLIT, NUM_FRAMES,
MODEL_NAME, transformers_version, n_skipped}` ao final do loop, para que a
proveniência fique no disco junto com os tensores.

**Classificação:** `[EQUIV]` para uma nova extração com os mesmos valores.
**Aviso:** se uma re-extração for feita, os tensores de treino serão
regravados — o loop tem `if out_path.exists(): continue`, então na prática nada
é sobrescrito sem apagar antes. Uma re-extração completa é decisão do Agente 1
(ver BUG-D/BUG-E).

---

### SIMP-05 — Parâmetro `tune_thresholds_c` de `compute_metrics` nunca é usado
**Tipo:** código morto (com consequência positiva acidental)  
**Severidade:** MÉDIA  
**Local:** `Thesis_video_fixed_CODE.py` célula 41, linhas 1123 e 1188–1196.

```python
def compute_metrics(eval_pred, tune_thresholds_c: bool = False):
    ...
    if tune_thresholds_c:
        best_thresholds, tuned = _search_thresholds_task_c(...)
```

**Problema:** os dois únicos chamadores — o HF `Trainer` (célula 50,
`compute_metrics=compute_metrics`) e `run_all_ablations`
(`ablation_harness.py` linha 153, `compute_metrics_fn(eval_pred)`) — passam
**um argumento só**. O ramo é inalcançável.

**Por que importa (além do lixo):** é exatamente por causa desse default que o
modelo de vídeo **não** sofre o vazamento de validação #2 do brief (busca de
threshold a cada época sobre o set de validação) que a auditoria de memes
encontrou. Hoje essa propriedade depende de um argumento default; deve depender
da estrutura.

**Proposta:** apagar o parâmetro e o bloco `if tune_thresholds_c:` inteiro
(incluindo o log `task_c_thresh_*`), deixando `tune_task_c_thresholds()`
(célula 35) como o **único** ponto onde a busca roda, com o comentário de
anti-leakage que já existe lá preservado. Assinatura final:
`def compute_metrics(eval_pred):` — igual à de `memes_core.py`.

**Classificação:** `[EQUIV]` — remove um ramo que nunca executou.

---

### SIMP-06 — `rt_norm` e `fix_norm` percorrem o pipeline inteiro sem serem usados
**Tipo:** código morto  
**Severidade:** BAIXA  
**Local:** produzidos em `Thesis_video_fixed_CODE.py` célula 9
(`build_bio_vector`, retorno 3º e 4º), expostos na célula 15 (`__getitem__`),
colateados em `ablation_harness.py` (linha 55), movidos para GPU (linha 86),
aceitos por `forward` (célula 28, linhas 687–688) — e **nunca lidos no corpo do
`forward`**. Idem `inference_video.py` linhas 165–166, 324–325, 380.

**Problema:** dois tensores por amostra atravessam dataset → collate → GPU →
assinatura do modelo sem efeito nenhum. Pior, o nome mente: são a média bruta
de `reaction_time` e `fixations_count`, sem normalização alguma (`_norm`
sugere o contrário). A informação de RT já entra no modelo — é a feature 12 do
`bio_vec`, essa sim escalada pelo `RobustScaler`.

**Proposta:** `build_bio_vector` passa a devolver 3 valores
`(bio_vec, cvpc, user_ids)`; remover as chaves `rt_norm`/`fix_norm` do
`__getitem__` e os parâmetros homônimos de `forward` nos dois arquivos.
As listas `rt_vals`/`fix_vals` internas continuam necessárias? Não: `rt_vals`
só alimentava `rt_mean`; a variável `rt` local (já clipada/imputada) continua
entrando no vetor. `fix_vals` pode sair inteiro.

**Classificação:** `[EQUIV]` — nenhum desses tensores participa de qualquer
cálculo que produza logits, loss ou consumo de RNG.
**Nota de conflito:** se o Agente 1 quiser *passar a usar* RT/fixações como
sinal auxiliar (injeção aditiva, como no modelo de memes), esta remoção entra
em conflito direto. Ver seção de conflitos.

---

### SIMP-07 — `compute_metrics` com três blocos quase idênticos (padrão já resolvido em memes)
**Tipo:** DRY / KISS  
**Severidade:** BAIXA  
**Local:** `Thesis_video_fixed_CODE.py` célula 41, linhas 1134–1198 (função de
~75 linhas com três seções `# === Task X ===`).

**Problema:** as três seções repetem o mesmo esqueleto (máscara → probabilidades
→ ids posicionais → loop hard/soft → prefixar chaves). O
`EXIST_MEMES/memes_core.py` já resolveu isso quebrando em
`_metrics_task_a/_metrics_task_b/_metrics_task_c` (linhas 751–858) — é o
estilo e a granularidade que o projeto já aceitou.

**Proposta:** replicar exatamente esse desenho:

```python
def _metrics_task_a(logits_a, labels_a, temp_dir):
    """Task A. Devolve {} se nenhuma amostra tem voto válido."""
    metrics = {}
    # guarda herdada do modelo de memes: lá labels_a vinha [N,2]. No vídeo é
    # sempre [N] — mantida por segurança, custa nada.
    if labels_a.ndim == 2:
        labels_a = labels_a[:, 1]

    mask_a = labels_a != -100.0
    if not mask_a.any():
        return metrics

    # CONVENÇÃO DO MODELO DE VÍDEO: sigmoid(logit_a) = P(YES)
    # (oposta à do memes_core.py — ver aviso na seção "Desenho proposto")
    prob_yes  = 1 / (1 + np.exp(-logits_a[mask_a].reshape(-1)))
    probs_a   = np.stack([1 - prob_yes, prob_yes], axis=-1)      # [P(NO), P(YES)]
    targets_a = np.stack([1 - labels_a[mask_a], labels_a[mask_a]], axis=-1)
    ids_a     = [str(i) for i in range(len(probs_a))]

    for evaluation in ('hard', 'soft'):
        for k, v in _run_pyevall(probs_a, targets_a, ids_a, '2_1', evaluation,
                                 temp_dir=temp_dir).items():
            metrics[f'task_a_{k.lower()}_{evaluation}'] = v
    return metrics


def _metrics_task_b(logits_b, labels_b, temp_dir):
    metrics = {}
    mask_b = labels_b[:, 0] != -1.0
    if not mask_b.any():
        return metrics

    x         = logits_b[mask_b]
    x         = x - x.max(axis=-1, keepdims=True)   # estabilidade do softmax
    probs_b   = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    targets_b = labels_b[mask_b]
    ids_b     = [str(i) for i in range(len(probs_b))]

    for evaluation in ('hard', 'soft'):
        for k, v in _run_pyevall(probs_b, targets_b, ids_b, '2_2', evaluation,
                                 temp_dir=temp_dir).items():
            metrics[f'task_b_{k.lower()}_{evaluation}'] = v
    return metrics


def _metrics_task_c(logits_c, labels_c, temp_dir):
    """Task C. Threshold FIXO em 0.5 aqui — a busca de threshold roda uma única
    vez, pós-treino, em tune_task_c_thresholds(). Rodar por época vazaria o
    set de validação (ver auditoria do modelo de memes)."""
    metrics = {}
    mask_c = labels_c[:, 0] != -1.0
    if not mask_c.any():
        return metrics

    probs_c   = 1 / (1 + np.exp(-logits_c[mask_c]))
    targets_c = labels_c[mask_c]
    ids_c     = [str(i) for i in range(len(probs_c))]

    # A ordem ('soft', 'hard') aqui é PROPOSITAL e assimétrica em relação às
    # tasks A e B (('hard', 'soft')): cada uma reproduz a ordem de inserção do
    # código original, para que o dicionário de métricas saia idêntico. Não
    # "uniformizar".
    for evaluation in ('soft', 'hard'):
        for k, v in _run_pyevall(probs_c, targets_c, ids_c, '2_3', evaluation,
                                 temp_dir=temp_dir).items():
            metrics[f'task_c_{k.lower()}_{evaluation}'] = v
    return metrics


def compute_metrics(eval_pred):
    logits_a, logits_b, logits_c = _unpack_predictions(eval_pred.predictions)
    labels_a, labels_b, labels_c = eval_pred.label_ids
    logits_a, logits_b, logits_c = map(np.asarray, (logits_a, logits_b, logits_c))
    labels_a, labels_b, labels_c = map(np.asarray, (labels_a, labels_b, labels_c))

    metrics = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics.update(_metrics_task_a(logits_a, labels_a, temp_dir))
        metrics.update(_metrics_task_b(logits_b, labels_b, temp_dir))
        metrics.update(_metrics_task_c(logits_c, labels_c, temp_dir))
    return metrics
```

**Classificação:** `[EQUIV]` — mesmas máscaras, mesmas probabilidades, mesmas
chaves de saída, mesma ordem de inserção. Já incorpora SIMP-05 (ramo
`tune_thresholds_c` removido).
**Cuidado que herdei do memes:** no `memes_core._metrics_task_c` há um
comentário forte dizendo que a ordem `hard_default` → `hard tuned` é
proposital porque a segunda **sobrescreve** a primeira. No vídeo isso não se
aplica (só existe o hard com 0.5), mas nomes de chave `_hard` significam coisas
diferentes nos dois modelos — não copie o comentário junto.

---

### SIMP-08 — `warmup_ratio` deprecado (célula 33)
**Tipo:** consistência / dívida de API  
**Severidade:** MÉDIA  
**Local:** `Thesis_video_fixed_CODE.py` célula 33, `warmup_ratio=0.1`.

**Evidência (output real da célula 33 no `Thesis_video_fixed.ipynb`), citada
verbatim:**
```
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
```

**Leitura:** na versão instalada na VM o argumento **ainda funciona** (é um
warning de deprecação, não um erro). Isso é mais brando do que o
`CLAUDE.md`/brief afirmam ("`warmup_ratio` foi removido em 5.6.0"). Duas
consequências: (a) não há bug de execução aqui e nada quebrou; (b) a afirmação
do `CLAUDE.md` deve ser reconfirmada — vale rodar `pip show transformers` na VM
e checar a linha de `MIGRATION_GUIDE_V5.md`, porque o notebook de memes já
migrou para `warmup_steps=0.1` e os dois notebooks estão em estados
diferentes.

**Proposta:** trocar por `warmup_steps=0.1`, alinhando com
`memes_model.ipynb`, e **verificar na VM** antes de considerar fechado:
imprimir `training_args.get_warmup_steps(total_steps)` antes e depois da
troca e conferir que o número é o mesmo.

**Classificação:** `[MUDA]` até essa verificação. Pelo mapeamento documentado
da deprecação deveria ser idêntico (0.1 = 10 % dos passos nos dois casos), mas
não posso rodar na VM, então não afirmo equivalência numérica.

---

### SIMP-09 — Ramo morto no `ablation_collate`
**Tipo:** código morto / KISS  
**Severidade:** BAIXA  
**Local:** `ablation_harness.py` linhas 55–65.

```python
if vals[0].dim() == 0:
    out[k] = torch.stack(vals)
else:
    out[k] = torch.stack(vals)
```
Os dois ramos são idênticos.

⚠ **Colisão com a minha própria proposta DIV-01.** Se o dataset unificado
passar a devolver sempre `id_EXIST` (uma `str`), um collate que faz
`torch.stack` em **todas** as chaves quebra com
`TypeError: expected Tensor as element 0`. Hoje o caminho de inferência
sobrevive porque usa o `default_collate` do PyTorch, que aceita strings.
O protocolo de migração da seção do módulo **não pegaria isso** —
`model_fingerprint` e o replay de `compute_metrics` não exercitam o collate.
Por isso a substituição já vem à prova de campos não-tensoriais:

```python
def ablation_collate(batch):
    """Empilha só os campos tensoriais.

    `id_EXIST` é str: o default_collate do PyTorch aceitaria, torch.stack não.
    Devolvido como lista, e run_inference deve dar pop nele ANTES do
    model(**batch) — o forward não tem esse parâmetro.
    """
    out = {k: torch.stack([b[k] for b in batch])
           for k in batch[0] if torch.is_tensor(batch[0][k])}
    if "id_EXIST" in batch[0]:
        out["id_EXIST"] = [b["id_EXIST"] for b in batch]
    return out
```
E em `run_inference` (`ablation_harness.py` linhas 85–92), antes do
`batch = {k: v.to(device) ...}`: `batch.pop("id_EXIST", None)`.

**Classificação:** `[EQUIV]`.
**Dependência de ordem:** se DIV-01 for aceita, SIMP-09 tem que entrar
**junto ou antes** — nunca depois.

---

### SIMP-10 — Resolução de caminho de vídeo com fallback que é no-op
**Tipo:** código morto  
**Severidade:** BAIXA  
**Local:** `extract_video_CODE.py` linhas 91–94.

```python
video_path = VIDEO_DIR / str(row["path_video"]).split("/")[-1]
if not video_path.exists():
    # try with just filename from path_video field
    video_path = VIDEO_DIR / Path(row["path_video"]).name
```
Em POSIX, `str(x).split("/")[-1]` e `Path(x).name` produzem exatamente a mesma
string — o segundo teste nunca pode ter sucesso depois que o primeiro falhou.
`extract_audio_CODE.py` (linha 90) já usa só a segunda forma.

**Proposta:** manter apenas `video_path = VIDEO_DIR / Path(row["path_video"]).name`
nos dois scripts (idealmente na função compartilhada — ver desenho de módulo).
**Classificação:** `[EQUIV]`.

---

### SIMP-11 — Ramo `else` vazio no loader de áudio
**Tipo:** código morto / KISS  
**Severidade:** BAIXA  
**Local:** `extract_audio_CODE.py` linhas 48–51.
```python
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
else:
    waveform = waveform          # no-op
```
Remover o `else`. **Classificação:** `[EQUIV]`.

---

### SIMP-12 — Imports não usados no notebook de treino e na inferência
**Tipo:** código morto  
**Severidade:** BAIXA  
**Local:** `Thesis_video_fixed_CODE.py` célula 2; `inference_video.py`
linhas 39–58.

Sem uso em nenhum ponto do arquivo:

| Arquivo | Imports mortos |
|---|---|
| notebook, célula 2 | `re`, `emoji`, `wordsegment.segment`, `PIL.Image`, `tqdm`, `sklearn.metrics.f1_score`, `DataLoader`, `CLIPProcessor`, `CLIPVisionModel`, `SWALR`, `update_bn` (só citado em comentário), `EarlyStoppingCallback`, `whisper`, `torchaudio` |
| `inference_video.py` | `re`, `emoji`, `wordsegment.segment` |

(`train_test_split` **é** usado, célula 16 — manter.)

**Constante morta, e não é inofensiva:** `LABELS_2_1 = ['YES', 'NO']` está
definida nos dois arquivos (célula 1; `inference_video.py` linha 62) e **nunca
é indexada em lugar nenhum** — todos os ramos da Task A escrevem `"YES"`/`"NO"`
literalmente e acessam `probs[i][0]` / `probs[i][1]` por posição. Essa morte é
justamente a raiz do risco descrito em DIV-07: não existe nada que amarre a
ordem do array a uma lista nomeada, então as convenções de memes e de vídeo
puderam divergir sem ninguém notar. **Não basta apagar:** ou se apaga e se
documenta a convenção no docstring de `_format_for_pyevall`, ou se passa a
indexar de verdade (`LABELS_2_1.index('YES')`) — a segunda opção é `[MUDA]`
enquanto as duas convenções não forem unificadas.

`whisper` e `torchaudio` no notebook de treino são particularmente enganosos:
sugerem que o áudio é extraído ali, quando na verdade vem pré-extraído do
`extract_audio.ipynb`. `CLIPProcessor`/`CLIPVisionModel` são resíduo do modelo
de memes.

**Atenção:** `emoji` / `wordsegment` / `re` importados **nos dois** arquivos
não são só lixo — são o rastro de um pré-processamento de texto que existe no
modelo de memes e **não existe** no de vídeo (o `tokenize_text` da célula 13
tokeniza o transcript cru). Removê-los é seguro; a ausência do
pré-processamento é assunto do Agente 1 (BUG-F).

**Classificação:** `[EQUIV]`. `EarlyStoppingCallback` também está morto (não
há `callbacks=[EarlyStoppingCallback...]`) — remover o import é `[EQUIV]`;
*adicionar* o callback seria `[MUDA]`.

---

### SIMP-13 — Corpo do `forward` indentado num nível extra
**Tipo:** consistência (estilo)  
**Severidade:** BAIXA  
**Local:** `Thesis_video_fixed_CODE.py` célula 28, linhas 679–780 — a
assinatura e todo o corpo de `EXISTVideoModel.forward` estão 8 espaços à
direita do esperado. É Python válido, mas destoa do resto do arquivo e do
`memes_core.py`, e torna diffs futuros ruidosos.
**Proposta:** re-indentar para 4/8 espaços padrão.
**Classificação:** `[EQUIV]` (whitespace).

---

### SIMP-14 — Números mágicos e constantes que não derivam da config
**Tipo:** configuração / consistência  
**Severidade:** MÉDIA  
**Locais e correções (todos `[EQUIV]`, valores preservados):**

| Local | Hoje | Proposta |
|---|---|---|
| célula 28, `bio_mlp` | `nn.Linear(BIO_INPUT_DIM, 128)` e `nn.Linear(128, BIO_HIDDEN_DIM)` | `BIO_MLP_HIDDEN = 128` na célula 5 (hoje o `128` literal coincide com `BIO_HIDDEN_DIM=128` por acaso; se alguém mudar `BIO_HIDDEN_DIM`, a camada interna fica fora de sintonia sem aviso) |
| célula 28, `bio_mlp` | `nn.Dropout(0.2)` | `BIO_DROPOUT = 0.2` (é o dropout mais alto do modelo e não está documentado em lugar nenhum) |
| célula 28, `cross_attn` | `num_heads=8`, `dropout=0.1` | `CROSS_ATTN_HEADS = 8`, `CROSS_ATTN_DROPOUT = 0.1` |
| célula 28, `fusion` | `nn.Dropout(0.1)` | `FUSION_DROPOUT = 0.1` |
| célula 30, `compute_loss` | `total = la + 0.7 * lb + 0.7 * lc` | `LOSS_W_B = LOSS_W_C = 0.7` (o docstring do `MultiTaskTrainer` já fala em "1.0/0.7/0.7" — a constante amarra comentário e código) |
| célula 22/24/54 e `inference_video.py` 651–658 | `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["query","value"])` repetido **5 vezes** | `def make_lora_config(): return LoraConfig(...)` em um só lugar |
| célula 28 `forward` | `modality_dropout_prob=0.05` como default de parâmetro | `MODALITY_DROPOUT_P = 0.05` na célula 5 (o markdown da célula 0 documenta "0.05 (was 0.15)" — a decisão fica junto da constante) |
| extração | `torch.zeros(NUM_FRAMES, 3, 224, 224)`, `torch.zeros(1280)` | usar as constantes de dimensão compartilhadas |

**Verificação de consistência das dimensões (pedida no escopo), parte 1 —
dentro do código:** conferi todas e **batem**. `FUSION_INPUT_DIM = 768 + 768 + 1024 + 128 = 2688` ✓ e o comentário
in-line "= 2688" está correto. `VIEWER_PROJ_DIM = BIO_RAW_DIM = 16`, que é o
que a subtração `bio_vec - (base1+base2)/2` exige ✓. `BIO_INPUT_DIM = 17 = 16 +
cvpc` ✓. `THRESHOLD_GRID` tem 12 valores (0.05…0.60), batendo com o comentário
"12 candidates × 6 classes = 72 PyEvALL calls" ✓.
**Duas legendas erradas, porém:**
1. O markdown da célula 0 diz "Biosignal features (ET + HR + viewer
   demographics) → 128-dim". **Não há demografia nenhuma** no modelo de vídeo
   (não existe vetor demográfico em lugar algum do arquivo) — é texto colado do
   modelo de memes. Corrigir para "ET + HR + CVPC".
2. A anotação de shape em `fused = self.fusion(torch.cat(...))  # [B, 2688]`
   (célula 28, linha 757) está errada: 2688 é a **entrada** do `fusion`; a saída
   é `[B, 1024]`. `inference_video.py` linha 406 tem a anotação certa
   (`# [B, 1024]`). Corrigir o notebook.

**Parte 2 — fora do código: ver SIMP-16.** As dimensões batem entre si, mas
**não batem com o `CLAUDE.md`**, e a divergência pode ter chegado ao paper.

---

### SIMP-16 — As dimensões da fusão no `CLAUDE.md` (e possivelmente no paper) não correspondem ao código
**Tipo:** configuração / consistência (constante definida em mais de um lugar)  
**Severidade:** ALTA (se o paper repetir os números) / MÉDIA (se só o `CLAUDE.md`)  
**Local:** `CLAUDE.md` linha 49 vs `Thesis_video_fixed_CODE.py` célula 5 e
`inference_video.py` linha 88.

```
CLAUDE.md:  Task 3 — fusão: 2592→1024-d (768 texto + 768 vídeo + 1024 áudio + 32 bio)
código:     FUSION_INPUT_DIM = VIDEO_DIM + AUDIO_PROJ_DIM + TEXT_DIM + BIO_HIDDEN_DIM
                             =    768    +      768       +   1024   +      128       = 2688
```

**Três divergências numa linha só:**
1. **bio = 128, não 32.** O próprio comentário do código diz por que e quando:
   `BIO_HIDDEN_DIM = 128  # bumped from 32 to balance vs text=1024/video=768/audio=768`.
   O `CLAUDE.md` registrou o estado anterior a esse "bump" e nunca foi
   atualizado. (Nota: o `VIEWER_EMB_DIM` **é** 32 — provável origem da
   confusão.)
2. **total = 2688, não 2592.** 2592 = 768+768+1024+32, ou seja, é o total
   coerente com o bio de 32-d antigo.
3. **rótulos de modalidade trocados:** o `CLAUDE.md` atribui 768 ao texto e
   1024 ao áudio. No código é o contrário — texto = 1024 (XLM-R-**large**) e
   áudio = 768 (`AUDIO_PROJ_DIM`, projetado a partir dos 1280 do Whisper).

**Por que importa:** o `CLAUDE.md` é a fonte que alimenta descrições de
arquitetura na tese e no paper. É o padrão #5 do brief (afirmação histórica no
`CLAUDE.md` que não sobrevive à verificação) reaparecendo no modelo de vídeo.

**Ação (para o Agente 1 / autor):**
- Corrigir a linha 49 do `CLAUDE.md` para
  `2688→1024-d (1024 texto + 768 vídeo + 768 áudio + 128 bio)`.
- **Verificar a seção de arquitetura do paper e o diagrama TikZ da Task 3.**
  Não consegui verificar: os fontes LaTeX **não estão neste repositório**
  (`grep -rn "2592\|2688"` em `VM-THESIS/` só acha o `CLAUDE.md`, os notebooks
  e o `inference_video.py`). Se o paper trouxer 2592 ou um ramo bio de 32-d,
  ele descreve uma arquitetura diferente da submetida — e isso passa à frente
  de tudo o mais neste documento.

**Classificação:** `[EQUIV]` — nenhuma linha de código muda; o código está
certo e a documentação errada.

---

### SIMP-15 — Convenções de retorno e de erro inconsistentes
**Tipo:** consistência  
**Severidade:** BAIXA  
1. `forward` devolve **tupla de tamanho variável** (3 ou 4 conforme haja
   labels), o que obriga `ablation_harness.py` (linhas 98–103) a fazer
   `len(out)` e `_unpack_predictions` (célula 41) a lidar com 4 formatos
   diferentes. Um retorno estável seria mais simples — mas o `Trainer` do HF
   depende de `loss` ser o elemento 0. **Não proponho mexer** enquanto o
   Agente 1 não decidir; registro como dívida.
2. Tratamento de erro assimétrico: `load_frames`/`load_audio_features` engolem
   qualquer exceção e devolvem zeros com um `print`, enquanto
   `_unpack_predictions` levanta `RuntimeError` com mensagem detalhada. O
   primeiro estilo esconde falhas de dados (ver BUG-D).
3. `inference_video.py` define `head_b = nn.Linear(FUSION_HIDDEN_DIM,
   len(LABELS_2_2))` onde o notebook usa `NUM_TASK_B`. Mesmo valor, nomes
   diferentes — resolvido pela extração do módulo.

**Classificação:** (2) e (3) `[EQUIV]` após a extração; (1) `[MUDA]`, não
proposto agora.

---

### Simplificações que considerei e **rejeitei** (para não serem re-levantadas)

- **`self._zeros(B, D, device, dtype)` (célula 28, linhas 675–677)** — parece
  supérfluo, mas o `ref_dtype = self.fusion[0].weight.dtype` interage com o
  autocast bf16 e com a promoção de tipos do `torch.cat`. Funciona hoje;
  qualquer "simplificação" exige teste em runtime na VM. **Não mexer.**
- **`set_all_seeds` chamando `set_seed()` do HF além dos 4 RNGs** — redundante
  na aparência, intencional pelo protocolo de determinismo do projeto.
  **Não mexer** (o brief manda preservar o protocolo intacto).
- **Guarda `if labels_a.ndim == 2` em `compute_metrics`** — herdada do modelo
  de memes e inalcançável no vídeo, mas é barata e protege contra uma mudança
  de formato do collator. Mantida, com comentário explicando a herança.
- **Comentários "Key fixes vs previous version" (célula 0, markdown)** —
  memória institucional (documentam bugs já corrigidos). **Preservar.**

---

## Divergências entre cópias duplicadas

> Esta seção é separada porque são **bugs potenciais**, não apenas duplicação.

### DIV-01 — `EXISTVideoDataset`: chave `id_EXIST` só na inferência
**Severidade:** BAIXA  
`inference_video.py` linha 317 devolve `"id_EXIST": str(item_id)`; a versão de
treino (célula 15) não. Consequência: no treino o PyEvALL recebe **ids
posicionais** (`str(i)` sobre a subamostra mascarada, células 35 e 41),
enquanto na inferência recebe os ids reais do EXIST. Como gold e pred usam a
mesma lista em cada caso, as métricas não são afetadas; mas nenhum erro de
alinhamento por-amostra é detectável no treino, e as máscaras diferentes por
task (`mask_a`, `mask_b`, `mask_c`) reindexam cada task de forma independente.
**Impacto:** baixo em métrica, alto em diagnosticabilidade.
**Proposta:** o dataset unificado sempre devolve `id_EXIST`; `compute_metrics`
continua com ids posicionais enquanto não houver forma de propagá-los pelo
`Trainer`. `[EQUIV]`.

### DIV-02 — `EXISTVideoModel.forward`: a versão de inferência não tem as flags de ablação
**Severidade:** MÉDIA  
Treino (célula 28) aceita `use_bio`, `use_viewer_emb`, `modality_dropout_prob`;
`inference_video.py` (linha 379) aceita nenhuma delas e **sempre** aplica bio +
viewer embedding. Os `state_dict` são compatíveis, então o `load_state_dict`
passa e nada avisa.
**Impacto:** se algum dia se decidir servir o modelo na configuração B1 (a
única com deltas positivos consistentes, segundo o `CLAUDE.md`), a inferência
silenciosamente rodará B2. Hoje o treino é sempre FULL, então não há erro
ativo — **divergência latente**.
**Proposta:** a classe unificada mantém as flags; `inference_video.py` ganha
`--use-bio/--no-use-bio` e `--use-viewer-emb/--no-use-viewer-emb`, com default
igual ao treino. `[EQUIV]` no default.

### DIV-03 — `test_case`: `"EXIST2025"` no treino vs `"EXIST2026"` na inferência
**Severidade:** ALTA  
Célula 37 tem `"test_case": "EXIST2025"` **hardcoded**;
`inference_video.py` usa `--test-case` com default `"EXIST2026"` (linha 783).
No treino a string é a mesma em pred e gold, então não afeta métrica alguma.
**Mas é a string que vai nos arquivos de submissão.**
**Proposta:** parametrizar em `video_core` com uma constante
`TEST_CASE = "EXIST2026"` e passar explicitamente nos dois lados.
**Classificação:** `[EQUIV]` para métricas de treino; **`[MUDA]` para
submissão** — qual string o PyEvALL do EXIST 2026 espera é uma pergunta para o
Agente 1 (BUG-C). Não altere sem confirmar contra o gold oficial.

### DIV-04 — Arredondamento das probabilidades soft
**Severidade:** BAIXA  
`format_records` (inferência, linhas 436/445/460) faz `round(v, 6)` em todos os
valores soft; `_format_for_pyevall` (treino, célula 37) grava o float completo.
**Impacto:** ≤1e-6 por probabilidade; ICM-Soft calculado no treino e na
inferência sobre os mesmos logits pode diferir na 6ª casa.
**Proposta:** unificar **sem** arredondamento (a versão do treino), já que o
arredondamento existia só para legibilidade do JSON.
**Classificação:** `[MUDA]` — mudança minúscula, mas mensurável nos JSONs de
submissão. Decisão do Agente 1.

### DIV-05 — `_run_pyevall`: duas assinaturas, uma delas com a extração de métricas inline
**Severidade:** BAIXA  
Treino (célula 40): `_run_pyevall(probs, gold_dist, sample_ids, task,
evaluation, temp_dir, thresholds_c)` — formata pred **e** gold internamente,
com `_extract_metrics_from_report` isolado (célula 39).
Inferência (linhas 521–555): `_run_pyevall(pred_records, gold_records, task,
evaluation, temp_dir)` — recebe registros já formatados, e reimplementa a
extração de métricas **inline** com `except (KeyError, TypeError)` — sem
`ValueError`, que a versão do treino captura.
Além disso a inferência tem `_gold_soft_records`/`_gold_hard_records`
(linhas 475–518) que refazem, com código totalmente diferente, o que
`_format_for_pyevall` já fazia no gold. **Conferi caso a caso: as regras de
binarização do gold são equivalentes** (Task A `g >= 0.5`; Task B `argmax` com
empate no primeiro índice nas duas versões; Task C `p >= 0.5` com fallback no
argmax). São ~45 linhas de código redundante que por sorte ainda concordam.
**Proposta:** apagar `_gold_soft_records`/`_gold_hard_records` e o
`_run_pyevall` da inferência; usar os do `video_core`.
**Classificação:** `[EQUIV]` — verifiquei a equivalência das três regras de
binarização.

### DIV-06 — `reaction_time` não-numérico é tratado de forma diferente ao ajustar o scaler e ao servir a feature
**Severidade:** MÉDIA  
`build_bio_vector` (célula 9): `rt = et.get("reaction_time", None)`;
imputa se `None` **ou** NaN; caso contrário `min(safe_float(rt), RT_CLIP)` —
e `safe_float` devolve **0.0** para um valor não-castável.
`collect_bio_vectors` (célula 14): `safe_float(..., default=float("nan"))` →
um valor não-castável vira NaN → **é imputado** com `RT_IMPUTE[lang]`.
Ou seja, para um `reaction_time` presente, não-`None` e não-castável, o scaler
seria ajustado com ~29.0/17.5 e o modelo receberia 0.0.

**Verificação feita** (`Dataset/training/EXIST2026_training.json`, 2 524
registros / 5 048 linhas usuário-vídeo): `reaction_time` ausente = 0,
`None` = 97, NaN = 0, numérico = 4 951, **não-castável = 0**. Também confirmei
que todos os registros têm exatamente 2 usuários (a indexação `user_vecs[0]`/
`[1]` é segura).
**Conclusão: divergência latente, sem impacto nos dados atuais.** Rebaixada de
bug para dívida — mas continua sendo uma armadilha em dados de teste não
inspecionados.

**Consequência para o desenho do módulo (importante):** extrair o construtor de
16 features para uma função única **não é DRY seguro**. Qualquer que seja a
semântica escolhida, um dos dois call sites muda: mexer em
`collect_bio_vectors` muda o `RobustScaler` ajustado (**exige re-treino**);
mexer em `build_bio_vector` dessincroniza as features de runtime do
`scaler.pkl` já salvo. **Classificação: `[MUDA]` — nível re-treino, não
cosmético. Requer aprovação do Agente 1.**

### DIV-07 — `_format_for_pyevall` do vídeo e do memes têm a **ordem dos índices da Task A invertida**
**Severidade:** MÉDIA  
`memes_core.py` linha 646: `p_yes = float(probs[i][0])`, `p_no = probs[i][1]`.
`Thesis_video_fixed_CODE.py` célula 37: `p_no = float(probs[i][0])`,
`p_yes = probs[i][1]`.
Cada uma está **correta em relação ao seu próprio chamador** (memes empilha
`[1-p_no, p_no]`; vídeo empilha `[1-p_yes, p_yes]`), e o cabeçalho de
`inference_video.py` linhas 30–36 documenta bem essa diferença. Não há bug
hoje. É um risco de fusão — tratado na seção do módulo.

---

## Bugs encontrados durante a simplificação (para o Agente 1)

> **Não corrigi nenhum destes.** Registro para não mascará-los dentro de uma
> refatoração.

- **[ALTA] BUG-A — A validação de checkpoint nunca rodou.** Célula 54 termina com
  `RuntimeError` (traceback no SIMP-02). Depois de corrigir o caminho, a
  validação pode acusar um problema real de salvamento — vale rodar antes de
  qualquer conclusão sobre os pesos. Além disso existem dois diretórios de
  modelo por seed (`best_model_seed_N` e `best_model_seed_N_cross_att`) com
  arquiteturas diferentes; é preciso decidir qual é o oficial e o que
  `inference_video.py --model-dir` deve apontar.
- **[ALTA] BUG-B — `--scaler` opcional na inferência ajusta o `RobustScaler` no
  próprio conjunto avaliado** (`inference_video.py` linhas 635–640). Isso troca
  a normalização biométrica de treino por uma ajustada no test set: mismatch
  de features silencioso (e, num cenário de avaliação, vazamento). Sugestão:
  tornar `--scaler` obrigatório. **Não é simplificação — é decisão de
  corretude.**
- **[ALTA] BUG-C — `test_case` da submissão.** Ver DIV-03: o treino usa `"EXIST2025"`.
  É preciso confirmar contra o gold oficial do EXIST 2026 qual string deve
  constar nos JSONs.
- **[ALTA] BUG-D — Zeros gravados no disco mascaram vídeos faltantes.** Os dois
  scripts de extração gravam `torch.zeros(...)` quando o vídeo não é
  encontrado/legível (`extract_video_CODE.py` linhas 99/106,
  `extract_audio_CODE.py` linhas 94/100). Efeitos: (i) o `try/except` de
  `load_frames`/`load_audio_features` no treino vira código morto — o arquivo
  sempre existe, então o `[WARN]` nunca aparece e ninguém sabe quantas amostras
  são zeros; (ii) frames são salvos **já normalizados** pelo processador do
  VideoMAE, logo um tensor de zeros não é "vídeo preto", é o **pixel médio** do
  ImageNet — uma entrada plausível, não neutra. Sugestão: gravar um manifesto
  de faltantes e decidir explicitamente a política (excluir amostra vs zeros).
  Conecta com o achado #1 do brief (confound de modalidade ausente).
- **[ALTA] BUG-E — Ordem de canais possivelmente invertida no frame de fallback.**
  `extract_video_CODE.py` linhas 53–57: quando `cap.read()` falha, o código faz
  `frame = frames[-1]` — um frame **já convertido para RGB** — e em seguida
  aplica `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` de novo, trocando R e B desse
  frame duplicado. Além disso, se a falha acontece no **primeiro** frame, o
  fallback é `np.zeros((224,224,3))` com resolução fixa, que vai para
  `np.stack` junto de frames na resolução nativa → `ValueError` → capturado
  pelo `except` externo → o vídeo inteiro vira zeros silenciosamente. (Note que
  o `CLAUDE.md` já registra "guards antes de `np.stack`" como lição aprendida.)
- **[MÉDIA] BUG-F — Zero pré-processamento de texto no modelo de vídeo.** A célula 13
  tokeniza o transcript cru. Os imports `emoji`/`wordsegment`/`re` (célula 2)
  são resíduo do modelo de memes, que aplica normalização de hashtags, emojis,
  URLs e menções. Se os transcripts do TikTok contêm hashtags/emojis, há um
  descasamento de domínio entre os dois sistemas.
- **[MÉDIA] BUG-G — `MultiTaskTrainer.compute_loss` ignora `num_items_in_batch`**
  (célula 48, linha 1264). Com `gradient_accumulation_steps=16` e máscaras cujo
  `sum()` varia por microbatch, as médias por microbatch são ponderadas
  igualmente — microbatches com poucas amostras válidas pesam demais. Só
  registro; não mexer.
- **[ALTA] BUG-H — A tabela de ablação e as métricas do paper estão em regimes
  numéricos diferentes.**
  **Fato (verificado nos artefatos):** para a mesma seed 2025, mesmo modelo,
  mesmo val set, `trainer.evaluate()` (output da célula 55) reporta
  `task_c_icmnorm_hard = 0.3531`, enquanto `ablation_seed_2025.csv` reporta
  `0.34763985` na linha B2/FULL. Os dois números não são comparáveis, apesar de
  aparecerem lado a lado no paper.
  **Mecanismo (inferido, NÃO verificado — confirmar na VM):** `run_inference`
  (`ablation_harness.py` linhas 71–80) faz `model.eval()` e chama o forward
  **sem contexto de autocast**, logo provavelmente roda fp32, enquanto o
  `Trainer` com `bf16=True` avalia sob autocast. Soma-se a isso o batch size
  (2 no Trainer vs 4 no harness). Não rodei nada na VM; a explicação vem do
  comportamento documentado das libs.
  **Como confirmar em uma linha:** imprimir `logits_c.dtype` dentro de
  `run_inference` e dentro do `prediction_step` do Trainer e comparar.
  **Os deltas internos da tabela continuam válidos** (as 4 condições
  compartilham o mesmo harness). Para os Agentes 1 e 3.
- **[ALTA] BUG-I — `cross_attn` é aplicada mesmo quando texto ou vídeo foram
  derrubados pelo modality dropout** (célula 28, linhas 752–757): o bloco
  cross-attention roda incondicionalmente, então `txt_feat` "zerado" ainda é
  misturado com vídeo e re-normalizado, e vice-versa. Isso é exatamente o
  achado #1 do brief (confound de fusão com modalidade ausente) na sua versão
  de vídeo. **Arquitetura — Agente 1.**

---

## Desenho proposto de módulo extraível

Mesmo padrão já aceito no projeto (`EXIST_MEMES/memes_core.py` + notebook/harness
só orquestrando). Proposta: **`EXIST_VIDEOS/video_core.py`**.

### `video_core.py` — o que vai para lá (ordem sugerida do arquivo)

| Seção | Funções / classes | Origem |
|---|---|---|
| 1. Reprodutibilidade | `set_all_seeds`, `model_fingerprint`, `CUBLAS_WORKSPACE_CONFIG` | células 3–4 (**copiar intacto**) |
| 2. Labels e hierarquias | `LABELS_2_1/2/3`, `TASK_2_2_HIERARCHY`, `TASK_2_3_HIERARCHY`, `THRESHOLD_GRID`, `TEST_CASE` | células 1, 38, 5 |
| 3. Constantes de dimensão | `NUM_FRAMES`, `FRAME_H/W`, `NUM_CHANNELS`, `VIDEO_DIM`, `AUDIO_RAW_DIM`, `AUDIO_PROJ_DIM`, `TEXT_DIM`, `BIO_*`, `VIEWER_*`, `FUSION_*`, `NUM_TASK_*`, `VIEWER_TO_IDX`, `RT_IMPUTE`, `RT_CLIP`, + as novas de SIMP-14 | célula 5 |
| 4. Caminhos | `MODEL_DIR(seed)`, `CKPT_DIR(seed)`, `frames_dir(split)`, `audio_dir(split)` | resolve SIMP-02 e SIMP-04 |
| 5. IO de features | `load_frames`, `load_audio_features` | células 6–7 |
| 6. Bio | `safe_float`, `_raw_bio_features(et, hr, rt)` *(ver aviso abaixo)*, `build_bio_vector`, `collect_bio_vectors` | células 8, 9, 14 |
| 7. Soft labels | `build_soft_label_a/b/c` | células 10–12 |
| 8. Dataset | `EXISTVideoDataset(..., has_labels=True)` — versão unificada, sempre devolve `id_EXIST` | célula 15 + `inference_video.py` 277–334 |
| 9. Modelo | `TemporalAttentionPool`, `EXISTVideoModel`, `make_lora_config()`, `build_backbones()` | célula 28 + 22/24 |
| 10. Loss | `loss_task_a/b/c`, `compute_loss` | célula 30 |
| 11. Métricas | `_extract_metrics_from_report`, `_format_for_pyevall`, `_run_pyevall`, `_search_thresholds_task_c`, `_metrics_task_a/b/c`, `compute_metrics`, `_unpack_predictions` | células 35, 37, 39–41 (com SIMP-05 e SIMP-07 aplicados) |
| 12. Treino (peças) | `SWACallback`, `MultiTaskTrainer`, `get_optimizer` | células 26, 48, 32 |

### O que **fica fora** de `video_core.py`
Igual ao precedente de memes: nada que execute treino.
`pd.read_json` + `train_test_split` (célula 16), o `fit` do scaler (célula 17),
`TrainingArguments` (célula 33), a construção do `Trainer` e `trainer.train()`
(células 50–52), o salvamento/validação de checkpoint (53–54), a chamada de
`tune_task_c_thresholds` (57) e a chamada de ablação (59) — tudo isso continua
no notebook (ou migra para um `ablation/train_one.py` análogo ao de memes, se o
Agente 3 propuser).

### O que sobra em cada arquivo depois
- `Thesis_video_fixed.ipynb`: ~10 células de orquestração + as células de
  sanity check (43–46). De 1 418 linhas de código para ~250.
- `inference_video.py`: CLI + loop de inferência + gravação dos 6 JSONs.
  De 794 para ~300 linhas.
- `ablation_harness.py`: registry (3 condições), `run_inference`,
  `run_all_ablations`, impressão/gravação. Praticamente inalterado.
- `extract_video.ipynb` / `extract_audio.ipynb`: importam as constantes de
  dimensão e o helper de caminho de `video_core.py`; ver abaixo.

### Duplicação entre os dois pipelines de extração
Os dois scripts compartilham: bloco de config, `pd.read_json(...).transpose()`,
o loop `tqdm` com skip resume-safe, a resolução de `path_video`, o fallback de
zeros, a lista `skipped` e o relatório final — ~40 % de cada arquivo.
Proposta: um gerador único em `video_core.py`, com os scripts fornecendo só a
função de extração:

```python
def iter_extraction_items(json_path, video_dir, output_dir):
    """Percorre o dataset devolvendo (item_id, video_path, out_path) só para os
    itens que ainda não têm saída gravada (resume-safe) e cujo vídeo existe.
    Itens sem vídeo são devolvidos com video_path=None para que o chamador
    decida a política — hoje cada script grava zeros por conta própria, o que
    torna amostras faltantes indistinguíveis de amostras reais (ver BUG-D)."""
```
**Classificação:** `[EQUIV]` desde que a política de zeros seja preservada
exatamente como está enquanto o Agente 1 não decidir outra (BUG-D).

### ⚠ Perigo nº 1 desta extração: **não fundir `video_core.py` com `memes_core.py`**
Os dois módulos terão funções com **o mesmo nome e semântica oposta**:

```python
# memes_core._format_for_pyevall, task '2_1'
p_yes = float(probs[i][0]); p_no = float(probs[i][1])
# video _format_for_pyevall, task '2_1'
p_no  = float(probs[i][0]); p_yes = float(probs[i][1])
```

Cada um é consistente com seu próprio chamador (memes: `sigmoid = P(NO)`;
vídeo: `sigmoid = P(YES)`), mas **um merge feito na confiança de que nomes
iguais significam comportamento igual inverte a Task A silenciosamente** — sem
erro, sem shape errado, só métricas ruins. Se um dia houver um `exist_core.py`
comum, a convenção da Task A tem que ser um parâmetro explícito, e a migração
precisa de um teste que compare os JSONs gerados antes e depois.
Mesmo cuidado com `compute_metrics`: a chave `task_c_..._hard` significa
"threshold ajustado" no memes e "threshold 0.5" no vídeo.

### Protocolo de migração sugerido (para não quebrar nada)
1. Criar `video_core.py` copiando **verbatim** (mesma disciplina do
   `memes_core.py`: "copiado função a função, sem alterações de
   comportamento"), sem aplicar nenhuma das simplificações.
2. Rodar `compute_metrics` sobre um `eval_pred` salvo em disco, com o notebook
   e com o módulo, e comparar dicionários chave a chave.
3. Comparar `model_fingerprint()` de um `EXISTVideoModel` recém-instanciado
   (após `set_all_seeds(SEED)`) construído pelas duas vias — tem que bater bit
   a bit.
4. Só depois aplicar SIMP-05..SIMP-15, uma de cada vez, repetindo (2) e (3).
5. As `[MUDA]` (SIMP-01, SIMP-08, DIV-03, DIV-04, DIV-06) só depois da
   reconciliação com o Agente 1.

---

## Possíveis conflitos com outros agentes

| Trecho | Minha proposta | Quem mais deve querer mexer | Como reconciliar |
|---|---|---|---|
| Célula 28, bloco de fusão + cross-attention (linhas 752–757) | Só constantes nomeadas (SIMP-14) e re-indentação (SIMP-13) | **Agente 1** — BUG-I (cross-attn ignora modality dropout) e o achado #1 do brief provavelmente reescrevem o bloco todo | O Agente 1 escreve primeiro; eu só aplico nomes de constante por cima |
| `rt_norm` / `fix_norm` (SIMP-06) | Remover | **Agente 1**, se quiser usá-los como sinal auxiliar aditivo (padrão que funcionou no modelo de memes) | Decidir **antes** de escrever: usar ou remover. Manter como está é a única opção ruim |
| `ABLATION_CONDITIONS` + `save_for_paper` (SIMP-01) | Remover braço fantasma, forçar caminho por seed | **Agente 1** (validade da tabela) e **Agente 3** (agregação multi-seed, teste pareado por seed — achado #3 do brief) | Meu patch é o mínimo; o Agente 3 provavelmente substitui `print_ablation_table` por uma análise pareada |
| `compute_metrics` (SIMP-05, SIMP-07) | Quebrar em 3 e apagar o ramo `tune_thresholds_c` | **Agente 3** — política de threshold/anti-leakage; **Agente 1** — se mudar as métricas reportadas | Aplicar SIMP-05 primeiro (é consenso: o ramo é inalcançável), depois discutir SIMP-07 |
| `run_inference` do harness (BUG-H) | Só o `ablation_collate` (SIMP-09) | **Agente 3** — vai querer alinhar autocast/batch size com o `trainer.evaluate()` | Meu patch é ortogonal; sem conflito de linhas, mas o mesmo arquivo |
| Scripts de extração (SIMP-04, SIMP-10, SIMP-11) | Constante `SPLIT`, remoção de ramos mortos | **Agente 1** — BUG-D/BUG-E podem forçar re-extração completa | Se houver re-extração, aplicar SIMP-04 **antes**, para que a nova extração fique documentada |
| `TrainingArguments`, célula 33 (SIMP-08) | `warmup_ratio` → `warmup_steps` | **Agente 3** — hiperparâmetros, `load_best_model_at_end` vs SWA, `save_total_limit` | Um único patch da célula 33, escrito pelo Agente 3, com minha troca incluída |
| `inference_video.py` inteiro (SIMP-03) | Reduzir a CLI + loop | **Agente 1** — BUG-B (`--scaler` opcional) e DIV-02 (flags de ablação) | A extração do módulo deve vir **depois** das decisões do Agente 1, ou vira retrabalho |
