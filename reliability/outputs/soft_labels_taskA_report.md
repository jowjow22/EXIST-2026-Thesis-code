# Soft labels da Task A ponderadas por θ — geração e diagnóstico

Gerado por `reliability/make_soft_labels_taskA.py`. Experimento controlado: mede
se o θ de competência por anotador carrega informação quando entra na construção
da soft label. **Não é uma tentativa de melhorar o modelo.** Resultado nulo é
aceitável.

## 1. Veredito — o retreino tem poder?

**SIM.** O deslocamento médio é `0.0720` (sd `0.0742`), a correlação de Pearson entre raw e
weighted é `0.9572`, e **443 itens (11.1%) mudam de lado da
maioria**. Isto não é ruído numérico: é uma mudança de alvo que o treino pode
registar. Gastar GPU é defensável.

`[LIMITAÇÃO]` isto mede quanto os **alvos** mudaram, não se a mudança é na direcção
certa. Um θ que carrega ruído desloca as labels tanto quanto um θ que carrega sinal.

## 2. Entradas

- Dataset: `EXIST_MEMES/Dataset/memes/training/EXIST2026_training.json` — **3984 itens**, 6 votos no primeiro item.
- θ: `reliability/outputs/theta_taskA_single_truth_K1.json` (`single_truth_K1`) — **887 anotadores**.
  θ ∈ [`0.0223`, `0.9647`], mediana `0.5300`, `converged = False`.
- Mediana do painel usada como fallback: `0.529955` (0 anotadores precisaram dela).

Auditoria dos votos de Task A: nenhum `UNKNOWN` no corpus, logo o raw é exactamente
`n_yes / n_votos` e a identidade com o pipeline é exacta, não aproximada.

## 3. Verificações obrigatórias

- **PASS** — 1. identidade do baseline (raw == n_yes/n_votos, |Δ|<1e-9)  
  3984/3984 itens · maxΔ = 0.000e+00
- **PASS** — 2. domínio [0,1], sem NaN, contagens e IDs iguais ao dataset  
  3984 raw · 3984 weighted · 3984 dataset · 0 violações de domínio
- **PASS** — 3. Σ pesos normalizados == 1 por item (|Δ|<1e-9)  
  maxΔ = 3.331e-16 sobre 3984 itens
- **PASS** — 4. anotadores sem θ (recebem mediana do painel)  
  0 anotadores em 0 itens
- **PASS** — 5. split congelado particiona os itens sem sobreposição  
  train 3386 + val 598 = 3984 de 3984

## 4. Diagnóstico de poder (weighted vs raw)

| métrica | valor |
|---|---|
| média |weighted − raw| | 0.0720 |
| desvio-padrão |weighted − raw| | 0.0742 |
| percentil 50 | 0.0555 |
| percentil 90 | 0.1781 |
| percentil 99 | 0.2934 |
| máximo | 0.4589 |
| itens com Δ > 0,05 | 2078.0000 |
| itens com Δ > 0,10 | 1187.0000 |
| Pearson r (raw, weighted) | 0.9572 |
| itens que cruzam 0,5 | 443.0000 |

n = 3984 itens. Δ>0,05 = 52.2% · Δ>0,10 = 29.8% · cruzam 0,5 = 11.1%.

## 5. Efeito colateral do θ não-clipado

`[DECISÃO DO AUTOR]` a fórmula do spec (`Σθv/Σθ`) é seguida **literalmente**, sem
clip nem piso. Sob EM single-truth, θ<0,5 significa "discorda da verdade inferida
mais do que acerta"; a fórmula **atenua** esse anotador mas nunca inverte o voto
dele. Os números abaixo quantificam o que isso faz — são diagnóstico, não um
segundo braço de treino.

| métrica | valor |
|---|---|
| anotadores com θ < 0,50 (de 887) | 407.0000 |
| anotadores com θ < 0,30 | 191.0000 |
| fracção dos votos emitida por θ < 0,50 | 0.4540 |
| fracção dos votos emitida por θ < 0,30 | 0.2113 |
| peso normalizado mínimo observado | 0.0074 |
| peso normalizado mediano | 0.1585 |
| peso normalizado máximo | 0.5648 |
| média |literal − clipado em [0.5, 0.999]| | 0.0420 |
| máximo da mesma diferença | 0.3396 |
| itens que cruzam 0,5 entre literal e clipado | 121.0000 |

Com 6 votos por item, o peso uniforme seria `0.1667`. O intervalo observado
(`0.0074`–`0.5648`) mostra quanto um voto
individual chega a valer face a outro no mesmo item.

## 6. Split congelado

`[DECISÃO DO AUTOR]` a estratificação da célula 44 lê a mesma coluna de soft label.
Como 443 itens cruzam 0,5, o braço ponderado teria um split
**diferente** do baseline — parte de qualquer delta de ICM viria daí, não do θ.

`split_taskA_frozen.json` congela as listas de `id_EXIST` resolvidas **uma
vez a partir do raw**, com `random_state=42`, `test_size=0.15`
e estratificação pelo voto majoritário do raw: **3386 treino / 598 validação**.
Os **dois** braços carregam esse ficheiro. O baseline tem de ser re-treinado com ele
também — comparar contra uma run que usou o `train_test_split` in-line reintroduz o
confound que este ficheiro existe para eliminar.

## 7. Como correr o retreino

Nenhum ficheiro do pipeline foi modificado. `memes_core.py` está bit-a-bit intacto;
as duas edições abaixo são no **notebook**, e são as únicas necessárias.

**Célula 35** — substituir a atribuição de `labels_task2_1_soft`:

```python
with open('../reliability/outputs/soft_labels_taskA_weighted.json') as f:
    _SOFT_A = json.load(f)
dataset['labels_task2_1_soft'] = dataset.index.map(lambda i: _SOFT_A[str(i)])
```

**Célula 44** — substituir o `train_test_split` (nos dois braços):

```python
with open('../reliability/outputs/split_taskA_frozen.json') as f:
    _SPLIT = json.load(f)
dataset_train = dataset.loc[_SPLIT['train_ids']].reset_index(drop=True)
dataset_val   = dataset.loc[_SPLIT['val_ids']].reset_index(drop=True)
```

**Voltar ao baseline:** apontar a célula 35 para `soft_labels_taskA_raw.json` (é
idêntico ao que `get_soft_distribution` produz — verificação 1), mantendo a célula 44
no split congelado. Ou restaurar a linha original com `get_soft_distribution`.

**Medir ICM em A, B e C.** Os alvos de B e C não mudaram; qualquer movimento neles é
spillover exploratório vindo do encoder partilhado, e tem de ser lido como tal.

## 8. O que estes números sustentam

**Sustentado:**

- As duas variantes de soft label diferem materialmente: média `0.0720`, 443 itens cruzam a maioria.
- O baseline é reproduzido exactamente (verificação 1), logo qualquer delta de ICM
  é atribuível à ponderação e não a uma reimplementação divergente.
- O split é idêntico nos dois braços por construção (secção 6).

**Não sustentado (não escrever):**

- Que a ponderação melhora o modelo. Isto ainda não foi medido — só o alvo mudou.
- Que θ mede competência. θ vem de EM single-truth, que **define** competência como
  concordância com a verdade que ele próprio infere. Sob LeWiDi, um anotador de θ
  baixo pode ser uma perspectiva minoritária legítima, não ruído.
- Que o θ está convergido: `converged = False` após 200 iterações.

