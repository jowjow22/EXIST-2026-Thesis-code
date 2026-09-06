# 08 — Re-extração de áudio: o que muda e por quê

Guia para executar e para **explicar depois**. Cada correção tem: o defeito, a
causa, a correção, e o que dizer quando perguntarem.

**Custo total: ~10 min de GPU.** Invalida todas as features de áudio atuais.

---

## O plano em 6 passos

1. Criar `audio_features/v2/{training,test}/` — árvore nova, não sobrescrever
2. Trocar `extract_whisper_features` pela versão com janelas + máscara
3. Trocar o fallback de `torch.zeros` por manifesto de faltantes
4. Adicionar constante `SPLIT` + gravar `_extraction_meta.json`
5. Rodar para `SPLIT="training"`, depois `SPLIT="test"` (~10 min total)
6. Apontar `video_core.audio_dir()` para `v2/` e re-treinar

Passos 1–4 são edição de código (~30 min). Passo 5 é GPU. Passo 6 destrava o resto.

---

# PARTE A — Correções de arquitetura (corretude)

## A1 — Máscara do padding: promediar só o áudio real

**Severidade: ALTA.** É a correção principal.

### O defeito

```python
audio = whisper.pad_or_trim(waveform.numpy())     # SEMPRE 480000 amostras = 30 s
encoder_out = model.encoder(mel)                  # [1, 1500, 1280]
features = encoder_out.squeeze(0).mean(dim=0)     # média sobre TODOS os 1500
```

`pad_or_trim` preenche com zeros até 30 s. Vídeo de 12 s → 12 s de fala + 18 s
de silêncio. O `.mean(dim=0)` promedia os 1500 frames do encoder, **silêncio
incluído**.

### O número

Medido na população completa (`analysis/audio_durations.csv`, n=2 524 treino):

| | treino | teste |
|---|---|---|
| itens < 30 s | 66,6 % | 76,0 % |
| `valid_fraction` média | 0,655 | 0,582 |
| do vetor `[1280]` que é silêncio | **34,5 %** | **41,8 %** |

Em segundos agregados no treino: **26 087 s de silêncio sintético injetados**.

### Por que isto é grave — o argumento a decorar

Não é ruído, é **variável de confusão**.

Ruído é aleatório: atrapalha mas não engana. Confound é sistemático e
correlaciona com o alvo. Aqui a diluição é **função da duração do vídeo**:

```
duração ──> quanto silêncio entra na média ──> magnitude do embedding
   │
   └──────> (possivelmente) probabilidade de ser sexista
```

Se vídeos sexistas tiverem duração média diferente — plausível, formatos de
TikTok diferem por tipo de conteúdo — o modelo pode acertar a Task A **lendo a
duração**, não o áudio. Métrica boa, aprendizado errado, quebra em dados novos.

Confound é pior que ruído porque **pode subir a métrica pelo motivo errado**.
Não aparece como problema no número.

E as features não estão corrompidas — verificado: 0/2 524 com norma zero, norma
média 14,96. **Estão diluídas.** Corrompido você detecta. Diluído passa em toda
checagem de sanidade.

### A correção

```python
n_valid = max(1, min(1500, int(round(valid_seconds * 50))))   # 1500 frames / 30 s
features = encoder_out[:n_valid].mean(dim=0)                  # média SÓ do válido
```

O `50` é `1500 frames ÷ 30 s`. Um áudio de 12 s ocupa os primeiros 600 frames;
promediamos 600, não 1500.

### Ressalva honesta — diga isto, não omita

**A self-attention do encoder ainda viu o padding.** Mascarar depois corrige a
*média*, não o *encoder*. Os 600 frames que sobram foram computados numa
atenção que atendeu sobre os 1500.

Isso não invalida a correção — o Whisper é treinado com esse padding, então o
encoder está desenhado para ele, e a média é onde o dano é grande e mensurável.
Mas a afirmação correta é **"removemos a diluição da agregação"**, não
"removemos o efeito do padding".

---

## A2 — Janelas múltiplas: parar de descartar áudio

**Severidade: ALTA.** Mesmo `pad_or_trim`, defeito oposto.

### O defeito

`pad_or_trim` também **corta**. Vídeo de 60 s → os primeiros 30 s são
codificados, os outros 30 s são descartados. Sem aviso, sem loop, sem janela
deslizante.

### O número

**33,4 % dos vídeos de treino passam de 30 s.** Entre eles, perde-se em média
36,1 % do áudio, no máximo 50,5 %. Agregado: **16 104 s (24,5 %) do áudio de
treino nunca chega ao encoder.**

No teste a cauda é pior: máximo 135,1 s, e esse item perde 77,8 %.

### A correção

Percorrer o áudio em janelas de 30 s e promediar as janelas, **ponderando pelos
segundos válidos de cada uma**:

```python
for start in range(0, n_samples, 30 * TARGET_SR):
    chunk = waveform[start:start + 30 * TARGET_SR]
    valid = chunk.shape[-1] / TARGET_SR
    ...
    feats.append(out[:n_valid].mean(dim=0))
    weights.append(valid)
return (stack(feats) * w[:, None]).sum(0) / w.sum()
```

### Por que ponderar, e não `mean()` simples

Vídeo de 50 s → janela 1 tem 30 s, janela 2 tem 20 s. Média simples daria peso
igual a um trecho de 30 s e a um de 20 s.

**Com pesos `(30, 20)`, o resultado é exatamente a média sobre todos os frames
válidos do áudio inteiro** — como se existisse uma janela de 50 s. É o mesmo
número que A1 produziria se o Whisper não tivesse limite. Essa é a justificação
matemática, não uma heurística.

---

## A3 — O achado novo: treino e teste têm o confound em intensidades diferentes

**Severidade: ALTA. Nenhum agente reportou** — a amostra de 60 do ARCH-04 era só
de treino.

### O defeito

| | `valid_fraction` média |
|---|---|
| treino | 0,655 |
| teste | **0,582** |

O teste é sistematicamente mais curto (76,0 % abaixo de 30 s, contra 66,6 %).

### Por que isto é o argumento mais forte a favor da re-extração

O modelo é **calibrado sob um regime de diluição e avaliado sob outro**.

Isso é *dataset shift induzido pelo pipeline de extração* — não é uma
propriedade dos dados, é um artefato que nós criamos. Se o modelo aprendeu
qualquer coisa que dependa da magnitude do embedding de áudio, essa coisa está
mal calibrada no teste por construção.

**Corrigido de graça por A1 + A2:** com a máscara, `valid_fraction` deixa de
existir como variável. Os dois splits passam a ser medidos na mesma escala.

### A frase para a tese

> A extração de áudio introduzia um fator de diluição dependente da duração, com
> médias diferentes entre treino (0,655) e teste (0,582), tornando as duas
> distribuições de features não-comparáveis por construção.

---

## A4 — Zeros no disco mascarando vídeos faltantes (BUG-D)

**Severidade: ALTA (operacional).**

### O defeito

```python
if not video_path.exists():
    torch.save(torch.zeros(1280), out_path)      # grava e segue
    continue
```

O arquivo `.pt` **sempre existe**. Consequência: o `try/except` de
`load_audio_features` no treino é **código morto** — o `[WARN]` nunca dispara, e
ninguém sabe quantas amostras são zeros.

Neste dataset o dano medido é zero (0 arquivos ausentes, 0 sem stream de áudio,
0 falhas de probe — reconciliação fecha em 3 198). **Mas a armadilha continua
armada** para a próxima extração.

### A correção

Gravar um **manifesto de faltantes** e decidir a política explicitamente:

```python
missing = []
...
    missing.append(item_id)
    continue                    # NÃO grava .pt
...
json.dump(missing, open(OUTPUT_DIR / "_missing.json", "w"))
```

Sem `.pt`, o `except` do `load_audio_features` volta a ser alcançável e o
`[WARN]` volta a significar alguma coisa.

### O paralelo que explica o princípio

No `extract_video`, zeros são **piores**: os frames são gravados já normalizados
pelo processador do VideoMAE, então um tensor de zeros não é "vídeo preto" — é o
**pixel médio do ImageNet**. Uma entrada plausível, não neutra. O modelo não tem
como saber que aquilo é ausência de dado.

Regra geral: **ausência de dado nunca deve ser representada por um valor que o
modelo interpreta como dado válido.**

---

## A5 — Ramo `else` vazio (SIMP-11)

**Severidade: BAIXA.** Trivial, listado por completude.

```python
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
else:
    waveform = waveform          # no-op
```

Apagar o `else`. Sem efeito funcional.

---

# PARTE B — Correções de otimização (processo)

## O1 — Árvore versionada `v2/` — a mais importante desta parte

**Severidade: MÉDIA, mas custaria uma onda inteira de re-treino.**

### O defeito

```python
if out_path.exists():
    continue     # resume-safe
```

Os scripts são resume-safe. **Depois de corrigir A1 e A2, re-rodar não faz
nada** — todos os `.pt` já existem. Zero arquivos novos, e o treino seguinte usa
features velhas.

O modo de falha: tudo roda sem erro, e as métricas parecem simplesmente **"não
melhorou"**. Você conclui que o ARCH-04 era irrelevante, quando na verdade a
correção nunca foi aplicada.

### A correção

Gravar em árvore nova em vez de apagar:

```python
OUTPUT_DIR = Path(f"./audio_features/v2/{SPLIT}/")
```

**Três ganhos:** (i) o skip resume-safe volta a ser correto porque o diretório
está vazio; (ii) as duas gerações coexistem, então dá para comparar; (iii) a
proveniência de qual treino usou qual extração fica gravada em disco.

Custo em disco: `[1280] float32` = 5 KB por item × 3 198 = **~16 MB**.
Irrelevante.

---

## O2 — Constante `SPLIT` + `_extraction_meta.json` (SIMP-04)

**Severidade: ALTA — é lacuna de proveniência, não arrumação.**

### O defeito

Os scripts estão commitados apontando para `test`, com a docstring dizendo
`train`:

```python
"""Saves → audio_features/train/{id_EXIST}.pt"""     # docstring
OUTPUT_DIR = Path("./audio_features/test/")          # código
```

**Os tensores de treino foram produzidos por uma versão editada à mão que não
existe mais no repositório.** Hoje ninguém consegue verificar que train e test
foram extraídos com o mesmo `TARGET_SR`, o mesmo modelo Whisper, a mesma
normalização.

Um descasamento aqui é exatamente o tipo de bug de pré-processamento que
explicaria performance ruim — e seria invisível.

### A correção

```python
SPLIT = "training"          # <- unica linha a editar
VIDEO_DIR  = Path(f"./Dataset/{SPLIT}/videos/")
OUTPUT_DIR = Path(f"./audio_features/v2/{SPLIT}/")
JSON_PATH  = f"./Dataset/{SPLIT}/EXIST2026_{SPLIT}.json"
```

Mais um `_extraction_meta.json` gravado ao lado das features:

```json
{"split": "training", "whisper_model": "large-v3", "target_sr": 16000,
 "window_s": 30, "multi_window": true, "mask_padding": true,
 "n_items": 2524, "n_missing": 0, "torch": "2.6.0+cu124", "date": "..."}
```

**A regra:** se dois splits podem divergir em pré-processamento, a diferença tem
de ser impossível de criar (uma constante) **ou** detectável depois (um
manifesto). Preferencialmente as duas.

---

## O3 — Batching do encoder (opcional, ganho pequeno)

**Severidade: BAIXA. Incluído para ser dispensado com argumento.**

Medido: **7,50 it/s** → 3 198 vídeos em ~7 min. Com as janelas múltiplas de A2,
~33 % dos itens precisam de 2 passagens → **~10 min**.

Dava para empilhar N mels num batch e ganhar talvez 3×. **Não vale.** Economiza
~7 min num pipeline onde o treino custa 4–8 h de GPU, e acrescenta código que
precisa lidar com número variável de janelas por item.

**O princípio (é o achado central do Agente 3):** o custo nunca foi o gargalo
deste projeto. A tabela de ablação "impossível" custa ~4–5 h. O PyEvALL, tratado
como caro, é **1,1 % do wall-clock**. Otimize corretude, não segundos.

---

## O4 — Armazenamento fp16? Não

`[1280] float32` = 5 KB. Todo o dataset de áudio = 16 MB. **Nada a ganhar.**

Isto é diferente dos **frames**, onde OPT-01 é real: 9,19 MiB por tensor, 24,3 GB
lidos por época, e é o teto que bloqueia qualquer aumento de batch. Não confundir
os dois.

---

# Ordem de execução e dependências

| # | Passo | Custo | Bloqueia |
|---|---|---|---|
| 1 | Editar script (A1, A2, A4, A5, O1, O2) | ~30 min | tudo |
| 2 | Rodar `SPLIT="training"` | ~8 min GPU | passo 4 |
| 3 | Rodar `SPLIT="test"` | ~2 min GPU | submissão |
| 4 | Apontar `video_core.audio_dir()` para `v2/` | ~2 min | re-treino |
| 5 | Sanidade: normas antes vs depois | ~5 min | — |

**Passo 5 é o que prova que a correção pegou.** Comparar norma média dos vetores
v1 vs v2: v1 era 14,96 (std 0,50). Se v2 der praticamente o mesmo número, a
correção **não** foi aplicada — vá verificar o skip resume-safe.

---

# O que estes números NÃO provam

Diga isto antes que perguntem.

1. **Não medem ganho de métrica.** Que a re-extração melhore o ICM continua
   **hipótese não verificada**. A cadeia é: defeito medido (fato) → correção
   plausível → *portanto melhora* (hipótese). Só re-treino confirma, e o ganho
   pode ser pequeno.
2. **Duração de stream ≠ duração de fala.** Um vídeo de 40 s pode ter 5 s de
   fala e 35 s de música. `valid_fraction` mede quanto da janela é *sinal*, não
   quanto é *conteúdo linguístico*. O dano medido é um **limite inferior**.
3. **Não testam alinhamento áudio-vídeo.** Frames e áudio são amostrados por
   caminhos independentes. Nada aqui verifica se concordam temporalmente.

---

# O que a re-extração invalida

- Todas as features em `audio_features/{training,test}/` (por isso `v2/`)
- Todo checkpoint treinado — já invalidados por outras mudanças (ver
  `05_IMPLEMENTATION_NOTES.md` §6)
- `predictions/cross_att/` e `predictions/swa/` — submissões de um sistema que
  deixa de existir. Na tese têm de aparecer como *"sistema submetido"*,
  separados de *"sistema corrigido"*

---

# Fontes

| Afirmação | Onde |
|---|---|
| durações, percentagens, `valid_fraction` | `analysis/audio_durations.csv` (n=3 198), notebook `analysis/audio_duration_analysis.ipynb` |
| shift treino/teste, reconciliação | `improvements/07_ARCH04_FULL_POPULATION.md` |
| defeito original, correção proposta | `improvements/01_ARCHITECTURE_FINDINGS.md` ARCH-04 |
| resume-safe, custos, batching | `improvements/03_OPTIMIZATION_FINDINGS.md` OPT-12, tabela de custos |
| zeros, proveniência, `else` morto | `improvements/02_SIMPLIFICATION_FINDINGS.md` BUG-D, SIMP-04, SIMP-11 |
| normas das features v1 (14,96 ± 0,50) | `01_ARCHITECTURE_FINDINGS.md` ARCH-04, verificação de saúde |
