# Auditoria: a afirmação de cross-attention da Task 2 (memes)

**Data:** 2026-09-20
**Âmbito:** repositório local `VM-THESIS` (ver "Limites do âmbito" no fim — há uma pista
fora dele que não pode ser fechada a partir daqui).
**Objetivo:** substanciar ou corrigir, com evidência do repositório, a afirmação do paper
CEUR-WS submetido:

> "An earlier version of the system used cross-modal attention between the text and vision
> CLS tokens. On the development set, attention weights collapsed to near-uniform
> distributions, which is a known failure mode on small datasets. CLS concatenation with a
> linear projection was therefore adopted."

**Restrição:** nada foi treinado, nenhum experimento novo foi corrido, nenhum valor foi
estimado. Só leitura de código, git e artefactos já existentes.

---

## Veredito

**A afirmação não é substanciável a partir deste repositório, e o seu núcleo empírico
("attention weights collapsed to near-uniform distributions") não tem qualquer evidência.**

Não existe, em lugar nenhum do repositório local, uma implementação de cross-attention para
o modelo de memes — nem por módulo nomeado (`nn.MultiheadAttention`), nem escrita à mão
(`bmm`/`einsum`/`matmul` + `softmax`), nem comentada, nem em backup, nem em checkpoint de
notebook, nem nos pesos treinados, nem no git. Consequentemente não existe nenhuma medição
dos pesos de atenção, nenhuma comparação cross-attention vs concatenação, e nenhum commit
de remoção.

A pergunta prioritária — "o defeito da Task 3 (softmax sobre 1 chave) também estava na
Task 2?" — **não é respondível**, porque não há forward da Task 2 com atenção para
inspecionar. Não é "sim" nem "não": é INCONCLUSIVO por ausência total do artefacto.

---

## Item a — Localizar a implementação de cross-attention do modelo de memes

**NÃO ENCONTRADO.**

Duas varreduras independentes sobre todos os `*.py`, `*.ipynb` e `*.md` de `EXIST_MEMES/`
(incluindo `.ipynb_checkpoints/`, `ablation/`, ficheiros de lock e cópias):

1. **Atenção por módulo nomeado** — `MultiheadAttention`, `MultiHeadAttention`,
   `cross_attn`, `CrossAttention`, `need_weights`, `attn_weights`, `attention_weights`,
   `attn_output_weights`, `register_forward_hook`: **0 ocorrências.**
   Controlo positivo na mesma invocação: `MultiHeadModel` → **23 ficheiros**.
   Repetido sem filtro de extensão sobre a árvore `EXIST_MEMES/` completa
   (**5 307 ficheiros**, mesmo controlo): também 0 — os únicos hits são este próprio
   relatório.
2. **Atenção escrita à mão** — `scaled_dot_product_attention`, `.bmm(`, `einsum(`,
   `torch.matmul(`, `transpose(-2,-1)`, `transpose(1,2)`: 5 ocorrências, **nenhuma de
   atenção**. São todas `einsum` de álgebra de matrizes de confusão / IRT
   (`confusion_core.py:385`, `dataset_verifications.ipynb:221,222,512,513`).
   Controlo positivo: `torch.cat` → **17 ficheiros**.

Adicionalmente, todas as células que continham `softmax`, `attention`, `attn` ou
`multihead` e ainda não tinham sido abertas foram inspecionadas linha a linha
(`memes_model copy.ipynb` células 1, 6, 55, 61, 69; `Untitled.ipynb` célula 9;
`.ipynb_checkpoints/Untitled-checkpoint.ipynb` célula 0, 411 linhas). Todos os *hits* são
`attention_mask` do tokenizer, o nome da classe `MultiHeadModel`, ou o `F.softmax` da loss
da Task B. Nenhum é um mecanismo de atenção entre modalidades.

Cronologia dos snapshots, do mais antigo ao mais recente:

| Ficheiro | mtime (última gravação, não autoria) | Arquitetura | Cross-attn |
|---|---|---|---|
| `.ipynb_checkpoints/inference_sample-checkpoint.ipynb` célula 12 | 2026-04-16 | só texto (XLM-R), sem CLIP, sem fusão | não |
| `.ipynb_checkpoints/Untitled-checkpoint.ipynb` célula 0 | 2026-05-13 | script de inferência, `MultiHeadModel` com `torch.cat` | não |
| `.ipynb_checkpoints/Thesis_main-checkpoint.ipynb` célula 41 | 2026-05-14 | texto + CLIP, `torch.cat` (confirmado por grep, não inferido) | não |
| `exist2026_Jow.zip` / `run3.zip` | 2026-05-14 | **só predições** — não contêm código | n/a |
| `.~Thesis_main.ipynb` | 2026-06-03 | texto + CLIP, `torch.cat` | não |
| `Thesis_main.ipynb` célula 42 (**sistema submetido**) | 2026-09-16 | texto + CLIP, `torch.cat` | não |
| `memes_model.ipynb` célula 35 | 2026-09-07 | idem | não |
| `memes_core.py:641` | 2026-09-16 | idem | não |

A versão mais antiga do repositório (16/abr) é **text-only**: não tinha ramo de visão
nenhum, logo não havia duas modalidades entre as quais atender. A primeira versão
multimodal que sobreviveu (13–14/mai) já usa `torch.cat`.

O único snapshot que falhou a leitura como JSON foi `.~Thesis_main.ipynb` (ficheiro de
lock/parcial do Jupyter, truncado na linha 4613); foi inspecionado por grep em texto bruto
— 0 ocorrências de qualquer construto de atenção, e a linha de fusão presente é
`torch.cat([text_cls, vision_cls], dim=-1)`.

### Confirmação pelos pesos treinados

Os *checkpoints* confirmam a arquitetura independentemente do código-fonte. Cabeçalho do
`model.safetensors` de `best_model/` (sistema submetido), 804 tensores, módulos de topo:

```
demo_gate, demo_proj, encoder, fusion_norm, head_a, head_b, head_c,
text_proj, vision_encoder, vision_proj
```

Nenhuma chave contendo `cross`, e nenhum `in_proj_weight` — que é o tensor `[3E, E]` que
`nn.MultiheadAttention` obrigatoriamente grava. Idem em
`best_model_conf_M1_lam0.0_seed42/` (806 tensores, módulos `base` e `confusion`).

## Itens b e c — Nº de chaves do softmax e shapes de Q/K/V

**INCONCLUSIVO — falta o artefacto.** Não há forward com atenção para medir.

O que existe, verbatim, é a fusão por concatenação (`Thesis_main.ipynb` célula 42,
sistema submetido):

```python
# === Text ===
text_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
text_cls = self.text_proj(text_out.last_hidden_state[:, 0, :])  # [B, 512]
...
# === Vision ===
vision_out = self.vision_encoder(pixel_values=pixel_values)
vision_cls = self.vision_proj(vision_out.last_hidden_state[:, 0, :])
# === Fusion ===
fused = torch.cat([text_cls, vision_cls], dim=-1)  # [B, 1024]
fused = self.fusion_norm(fused)
```

Ambos os ramos colapsam para o CLS (`[:, 0, :]`) **antes** da fusão.

### Contraste: na Task 3 o defeito é verificável, porque o código existe

`EXIST_VIDEOS/inference_video.py:373-403` (sistema de vídeo submetido):

```python
373:        self.cross_attn = nn.MultiheadAttention(
377:        self.cross_norm = nn.LayerNorm(TEXT_DIM)
400:        vid_proj  = self.cross_proj(vid_feat).unsqueeze(1)
401:        txt_q     = txt_feat.unsqueeze(1)
402:        cross_out, _ = self.cross_attn(query=txt_q, key=vid_proj, value=vid_proj)
403:        txt_feat  = self.cross_norm(txt_feat + cross_out.squeeze(1))
```

Q: `[B, 1, D]` · K: `[B, 1, D]` · V: `[B, 1, D]`. **Uma chave** → softmax sobre 1 elemento
≡ 1,0. Nota adicional: os pesos de atenção são descartados na própria linha 402 (`, _`),
portanto nem no modelo de vídeo eles foram alguma vez observados.

É esta a assimetria que a tese tem de respeitar: o defeito degenerado está **provado** na
Task 3 e **não existe artefacto** na Task 2.

## Item 1 — Medições dos pesos de atenção (entropia, desvio da uniformidade, plots)

**NÃO ENCONTRADO — e não podia existir.**

`need_weights`, `attn_weights`, `attention_weights`, `attn_output_weights` e
`register_forward_hook`: 0 ocorrências em todo o `EXIST_MEMES/` (grep validado por
controlo). Sem módulo de atenção e sem hook, nunca houve onde capturar pesos.

Existe no repositório um diagnóstico de colapso — mas é **outra coisa**: as colunas
`collapse_mean_inter_item_variance`, `collapse_mean_abs_diff_vs_baserate`,
`collapse_mean_pairwise_cosine`, `collapse_collapsed`, `collapse_verdict` em
`ablation/results/raw_results.csv` medem colapso das **previsões de saída**, não dos pesos
de atenção. Não servem para sustentar a frase.

## Item 2 — Comparação direta cross-attention vs concatenação

**NÃO ENCONTRADO.**

`ablation/results/raw_results.csv` tem 25 braços distintos e nenhum é de fusão:

```
abl1_config_{A,B,C}_warmup_{0.1,0.0,0.2}
abl2_config_{A,B}_demo_{base,extended}
abl3_config_{A,B,C}_taskaw_{2.0,1.0,3.0}
abl4_config_{A,B,C}_unfrozen_{4,0,8}
abl5_config_{A,B,C,D}_{vision_demo,vision_only,demo_only,text_only}
abl6_config_{A,B}_norm_{sum,l2}
inj_{current,renorm,headinject}_{base,extended}
injctrl_headinject_shuffled_base
baseline
```

Os quatro logs de treino foram varridos por `cross_attn|MultiheadAttention|CrossAttention`,
cada um com controlo positivo: `runner.log` (52 MB) **0**, `runner_INJ.log` (45 MB) **0**
(controlo `config_name` = 7), `runner_INJCTRL.log` (7,6 MB) **0**, `phase0.log` (9 MB) **0**.

`EXIST_MEMES/results/` e `EXIST_MEMES/runs_baseline/` estão vazios (`find … -type f` = 0
ficheiros). `metrics_history/` começa em `20260804_155433` — posterior à decisão de
arquitetura. Os dois zips de submissão (`exist2026_Jow.zip`, `run3.zip`, ambos de 14/mai)
contêm apenas `predictions/`, sem código.

Não existe, portanto, nenhum par de valores ICM-Soft-Norm / ICM-Hard-Norm atribuível a uma
condição cross-attention, em nenhuma época, seed ou conjunto.

⚠ `EXIST_VIDEOS/best_model_seed_42_cross_att/` e `EXIST_VIDEOS/predictions/cross_att/`
existem — são da **Task 3**. Não podem ser citados na secção da Task 2.

## Item 3 — Datas e commits de introdução/remoção

**NÃO ENCONTRADO — o git não cobre o período relevante.**

```
d6b6e03  2026-09-06 16:15:01 +0100  feat: update 6/09
ff37b33  2026-08-04 16:45:34 +0100  first save
```

O repositório tem **dois commits**, e o primeiro é de 2026-08-04 — posterior tanto à
submissão como à decisão de arquitetura (a versão multimodal já usava `torch.cat` em
2026-05-13/14). Não existe commit de introdução nem de remoção, e portanto não existe
mensagem de commit com a justificação real.

A cronologia utilizável vem de mtimes de ficheiro, não do git — é a tabela do item (a).

## Item 4 — Flag parametrizável de cross-attention no framework de ablação

**NÃO ENCONTRADO.** `ablation/config.py:27-53`, `AblationConfig` (frozen dataclass), campos
completos:

```
name, seed, num_epochs, warmup_steps, task_a_weight, unfrozen_vision_layers,
use_vision, use_demographics, demo_feature_set, demo_norm_mode,
demo_injection_mode, run_collapse_diagnostics, demo_shuffle
```

Não há `use_cross_attention` nem qualquer campo de fusão. `ablation/train_one.py` e
`ablation/runner.py`: 0 ocorrências de `cross`.

---

## O que existe: prosa, toda ela derivada da própria afirmação

Quatro textos repetem a alegação. Nenhum é fonte primária — todos a reformulam:

- `EXIST_MEMES/memes_core.py:641` — `# === Fusão: concatenação simples, sem cross-attention ===`
- `EXIST_MEMES/ablation/ABLATION_REPORT.md:227-229` — *"decisão de projeto documentada:
  cross-attention colapsou para padrões quase uniformes no dataset pequeno, ~5.000 memes"*
  — note-se o "documentada", que aponta para a documentação, não para uma medição
- `EXIST_MEMES/memes_model.ipynb` célula 34 (markdown) — *"Cross-attention foi testada e
  descartada durante o desenvolvimento: com um dataset pequeno (~5.000 memes), ela
  colapsava para padrões quase uniformes"*
- `CLAUDE.md:242-243` — mesma frase

Todos posteriores ao facto e sem número associado. Citar qualquer um deles como evidência
seria circular.

---

## Implicação para a redação da tese

A frase do paper tem duas partes com estatutos diferentes:

1. **"An earlier version used cross-modal attention ... CLS concatenation was therefore
   adopted"** — plausível como memória do autor, mas **sem vestígio no repositório local**.
   A versão multimodal mais antiga preservada já é concatenação. Se houve um protótipo com
   atenção, não sobreviveu aqui (ver "Limites do âmbito").
2. **"attention weights collapsed to near-uniform distributions"** — **não sustentável**.
   Nunca existiu, neste repositório, código capaz de expor esses pesos (`need_weights` /
   hook: 0 ocorrências), e não há número, plot ou log.

### O argumento estrutural, e o seu limite

Vale a pena registar o argumento, mas com o cuidado que os itens (b)/(c) exigem. A frase do
paper descreve atenção *entre os CLS tokens de texto e visão* — dois vetores **já
agrupados**. Qualquer desenho compatível com essa descrição tem conjunto de chaves de
comprimento ≤ 2:

- **K = 1** (q = texto, k = v = imagem): o softmax é sobre um elemento e vale identicamente
  1,0. "Quase uniforme" é uma tautologia, não uma observação. É o desenho da Task 3.
- **K = 2** (ambos os CLS empilhados numa sequência de 2 tokens): o softmax é sobre dois
  elementos e reduz-se a um único escalar aprendido. "Quase uniforme" significaria apenas
  que esse escalar ficou perto de 0,5.

Em nenhum dos dois casos se trata do fenómeno que o paper invoca — o colapso da atenção em
datasets pequenos pressupõe uma **sequência longa de chaves** sobre a qual a massa se
espalha. **Mas qual dos dois desenhos existiu é indecidível**, porque o código não existe;
o argumento acima é dedução a partir da frase do paper, não leitura de código.

### Redações defensáveis, por ordem de conservadorismo

- **Mais segura:** descrever a concatenação como escolha de projeto e justificá-la pelo
  argumento estrutural acima, apresentado como argumento (não como resultado experimental),
  citando a Task 3 como o caso em que o mesmo desenho é verificável no código.
- **Intermédia:** manter a nota histórica ("um protótipo inicial explorou cross-attention")
  e declarar explicitamente que não foi conservado nem medido, removendo a afirmação sobre
  os pesos.
- **A evitar:** repetir "attention weights collapsed to near-uniform distributions". É o
  único elemento da frase que soa quantitativo e é o único sem qualquer suporte.

---

## Limites do âmbito

Esta auditoria cobre o repositório local. Duas fontes **não** foram verificadas e podem
conter um protótipo de maio que nunca foi copiado para cá:

- **Histórico de revisões do Colab / Google Drive.** O CLAUDE.md regista que o
  desenvolvimento se fez em Jupyter/Colab; o Colab guarda revisões por notebook,
  independentes do ficheiro local. É a pista com maior probabilidade.
- **A VM de treino** (`/home/a67606/jupyter-env/`), onde os notebooks correm e onde podem
  existir versões que nunca desceram para a máquina local.

Se nenhuma das duas tiver o protótipo, a alegação passa de "não substanciável" a "sem
artefacto conhecido", e a redação "intermédia" acima deixa de ser sustentável junto com a
"mais segura".

## Nota sobre fiabilidade

O volume do projeto desmontou-se várias vezes durante a sessão (`ENXIO: no such device or
address`, `error: unable to open loose object ...: Device not configured`). Por isso:

- Todo grep negativo reportado aqui foi corrido **com um controlo positivo na mesma
  invocação** e **sem `2>/dev/null`**, para que um erro de I/O não se disfarçasse de "sem
  resultados". Uma primeira passagem por `runner_INJ.log` devolveu saída vazia sem erro;
  foi repetida com `LC_ALL=C --binary-files=text` e só então deu `0` com controlo `7`.
- Os `git log -S` (pickaxe) **falharam** por objetos ilegíveis durante um desmonte
  (`fatal: unable to read 61c9624...`) e **não** são a base de nenhuma conclusão. O item 3
  assenta apenas em `git log --format='%h %ai %s' --all`, que correu limpo.
