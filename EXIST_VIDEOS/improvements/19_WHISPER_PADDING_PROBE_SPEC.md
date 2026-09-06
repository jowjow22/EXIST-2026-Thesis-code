# 19 — Pré-registro: probe de atenção ao padding dentro da janela de 30s do Whisper

**Escrito em:** 2026-08-13, ANTES de rodar `probe_whisper_attention_padding.ipynb` (D) e antes
de ver qualquer número. Segue a convenção do doc 16 §0. Item 11 do plano
`zippy-shimmying-glacier.md`.

**Estado:** pré-registro. Nada foi executado. Este documento fixa a pergunta, o contra-
argumento e a leitura ANTES do notebook D existir, para que os números não moldem a
interpretação.

---

## 0. Convenção de marcação (idêntica ao doc 16 §0)

- `[MEDIDO]` — número produzido por execução, com o artefato que o contém.
- `[VERIFICADO]` — conferido diretamente no código, com `arquivo:linha`, ou por aritmética.
- `[HIPÓTESE]` — interpretação. Não escrever na tese como fato.

Este documento não contém `[MEDIDO]` — é pré-registro. Contém `[VERIFICADO]` (o que já está
no código hoje) e a árvore de leitura pré-registada para os números que o notebook D vai
produzir.

---

## 1. A pergunta

`extract_audio_v2.ipynb` (A1) já corrige a diluição da MÉDIA do vetor de áudio: em vez de
promediar os 1500 frames do encoder do Whisper incluindo o silêncio sintético do
`pad_or_trim`, a média usa só os `n_valid` primeiros frames válidos
`[VERIFICADO — extract_audio_v2.ipynb célula 3]`. Isso já removeu o confound medido no doc
07/spec (34,5%/41,8% do vetor médio era silêncio) — **não é reaberto aqui**.

Mas a correção A1 age só na agregação (a média, no final). Ela não muda o que a
**self-attention dentro do encoder** vê: cada janela ainda é sempre construída a
`pad_or_trim`-30s, e os blocos de atenção do Whisper processam os 1500 frames (válidos +
padding de silêncio) juntos, antes de qualquer seleção de `n_valid` acontecer. A pergunta
deste probe:

> **A self-attention dentro da janela de 30s "vê" o padding de silêncio, e isso é o
> mecanismo por trás do *nuisance* de duração medido no doc 16 (AUC−nulo = 0,985 para
> decodificar "duração > mediana" a partir do vetor de áudio, doc 16 linha 269/505)
> `[VERIFICADO — 16_VIDEO_INVESTIGATION_RESULTS.md:269,505]`?**

Se sim, um vetor calculado sem o padding jamais entrar no encoder (não só excluído da
média) deveria decodificar duração pior — os frames válidos deixariam de "saber", via
atenção, quanto silêncio havia ao redor deles.

## 2. O contra-argumento a levar a sério

**O Whisper foi TREINADO com janelas de 30s com padding de silêncio.** `pad_or_trim` a 30s
não é uma decisão do pipeline deste projeto — é o contrato de entrada do encoder tal como
pré-treinado pela OpenAI. Um encoder que sempre viu janelas padded em pré-treino pode ter
aprendido a **ignorar** o padding via atenção (é um sinal de baixa informação, quase
constante, log-mel de silêncio) tão bem quanto a correção A1 já ignora ele na média. Nesse
caso, processar a janela **inteira** (com padding) é o regime **in-distribution**, e é o
caminho SEM padding (janela cortada, `positional_embedding` fatiado, comprimento nunca visto
em pré-treino) que seria **out-of-distribution** — e qualquer piora medida seria artefato da
mudança de regime, não evidência de que o padding "contamina" o vetor.

Este probe existe para **decidir entre as duas leituras**, não para confirmar a suspeita de
contaminação. A hipótese nula (Whisper já ignora o padding via atenção; a janela cortada é
que está fora de distribuição) é tratada com o mesmo peso a priori que a hipótese de
contaminação.

## 3. Leitura pré-registada e assimétrica (as 3 condições)

Sonda linear (§5) roda sobre os vetores dos dois caminhos (A = como em produção hoje, com
padding entrando na atenção; B = mel cortado para `2·n_valid` colunas ANTES do encoder, sem
padding em lugar nenhum) para três alvos: duração (primário), língua e A_hard (controles).
Comparação é `AUC−nulo` do caminho B versus caminho A, no mesmo alvo.

| condição | leitura | consequência |
|---|---|---|
| **B decodifica duração IGUALMENTE BEM que A** (Δ pequeno, dentro do ruído de permutação) | a atenção ao padding **não é** o mecanismo — o Whisper já era robusto a ele | **rota FECHA.** Achado negativo publicável: o nuisance de duração não vem da atenção intra-janela; é outra coisa (ex.: o log-mel dos frames válidos em si já carrega informação de prosódia/ritmo correlacionada com duração — não investigado aqui). Nenhuma v3 é justificada por este eixo. |
| **B decodifica duração MUITO PIOR que A, E os controles (língua, A_hard) se MANTÊM** | consistente com "o padding realmente carregava o sinal de duração via atenção, e removê-lo não destrói informação semântica geral" | **GATILHO, nunca prova**, para considerar uma extração v3 que evite padding na atenção (ex.: `attention_mask` do encoder, se a API permitir, ou reamostragem de janela variável). Decisão de rodar v3 fica com o autor — este probe não a implementa. |
| **B piora em duração E TAMBÉM nos controles** (língua, A_hard caem junto) | é degradação genérica do caminho B (regime out-of-distribution do encoder, per o contra-argumento §2), não remoção seletiva de um confound | **rota fecha igual**, mas por um motivo diferente da condição 1: não é "o padding não importava", é "cortar a janela quebra o encoder de um jeito que não discrimina duração de semântica" — não é evidência utilizável em nenhuma direção. |

Nenhuma das três condições produz, por si só, uma re-extração v3. A tabela existe para que
a leitura dos números não seja escolhida depois de vistos.

## 4. Método

### 4.1 Caminho A — recompute do v2 atual

Cópia declarada (não importação) das ~20 linhas de `extract_whisper_features_v2` de
`extract_audio_v2.ipynb` célula 3: mel completo de 30s (`whisper.pad_or_trim` + `log_mel_spectrogram`),
forward do encoder via a API normal do pacote (`model.encoder(mel.unsqueeze(0))`), média
mascarada aos `n_valid` primeiros frames. Para os itens da amostra (`duration_s < 30`), cada
item tem exatamente 1 janela (o `for start in range(0, n_samples, CHUNK_SAMPLES)` do original
roda uma vez só), então não há necessidade de reproduzir a agregação ponderada entre janelas
(A2) — ela é uma no-op aqui, mas o código copiado a mantém por fidelidade.

**Portão (obrigatório, aborta se falhar):** cosseno > 0,999 entre o vetor recomputado e o
`.pt` já salvo em `audio_features/v2/training/{id}.pt` para os mesmos IDs. Se falhar,
investigar dtype/versão do Whisper (`openai-whisper` na VM: `20250625`, ver CLAUDE.md) ANTES
de ler qualquer AUC — divergência no portão invalida os dois caminhos, não só o B.

### 4.2 Caminho B — mel cortado, forward manual do encoder

Mesmo mel do caminho A (mesma chamada `log_mel_spectrogram` sobre o mesmo waveform), mas
**cortado para as primeiras `2·n_valid` colunas ANTES de entrar no encoder** — o padding de
silêncio nunca é materializado como entrada de nenhum bloco de atenção.

A API de alto nível do pacote `openai-whisper` (`model.encoder(mel)`) não aceita comprimento
de entrada variável: o `positional_embedding` do encoder é uma tabela fixa de tamanho
`n_ctx=1500`, somada por broadcasting a uma entrada que a classe `AudioEncoder` assume
sempre ter exatamente 1500 posições (`assert x.shape[1:] == positional_embedding.shape` no
forward original). Por isso o caminho B replica o forward do encoder **manualmente**:
`conv1 → gelu → conv2 → gelu → permute → + positional_embedding[:n_valid] → blocks →
ln_post`, com o `positional_embedding` fatiado às primeiras `n_valid` posições.

O corte de `2·n_valid` colunas de mel produz **exatamente** `n_valid` posições na saída:
`conv1` é `stride=1, padding=1` (preserva o comprimento), `conv2` é `stride=2, padding=1`
(divide por 2, arredondando pra baixo mas compensado pelo padding — para `L=2n` par,
`floor((L+2-3)/2)+1 = floor((2n-1)/2)+1 = (n-1)+1 = n`). Isso casa exatamente com
`positional_embedding[:n_valid]`, sem interpolação nem truncamento adicional.

### 4.3 Sonda linear (receita verbatim, célula 23 de `probe_video_contribution.ipynb`)

`StandardScaler` + `LogisticRegression(C=0.05, max_iter=3000)`, ajuste na metade `fit` de
`vc.split_val_halves(N)`, relato de AUC na metade `report`, nulo por 10 permutações do rótulo
(seeds `1000+r`) — mesma função `probe()` da célula 23, reaplicada sobre a amostra deste
probe em vez do cache do val de 505. Métrica reportada: `AUC − média(AUC nulo)`, por caminho
(A/B), por alvo.

**Alvos:**
- **duração > mediana da amostra** (primário — é o que este probe decide)
- **língua (es vs en)** (controle — precisa se manter estável nas 3 condições da §3)
- **A_hard (sexista? gold hard)** (controle — idem)

## 5. Amostra

~300–400 itens do split `training` com `duration_s < 30`, lidos de
`EXIST_VIDEOS/analysis/audio_durations.csv` (filtro `split == "training"`), amostrados com
seed fixa. `[VERIFICADO]`: 1682 de 2524 itens de treino têm `duration_s < 30`
(`analysis/audio_durations.csv`, contagem direta) — a amostra é um subconjunto determinístico
desses.

Restrição a `duration_s < 30`: só esses itens têm o padding de silêncio como fenômeno
possível dentro de uma única janela de 30s (é exatamente o caso que A1/A2 tratam por
mascaramento/ponderação de janelas — aqui isolamos o caso de janela única para que o caminho
B, de forward manual de janela única, seja a comparação direta e sem confundir com a lógica
de múltiplas janelas).

## 6. O que este item NÃO é

- **Não é uma correção.** Nenhum `.pt` novo é gravado em `audio_features/`. Nenhuma árvore
  `v3/` é criada.
- **Não decide, por si**, se uma re-extração v3 deve acontecer — na melhor das leituras
  (condição 2 da §3) ele é só **gatilho**, e a decisão de gastar uma re-extração fica com o
  autor.
- **É diagnóstico**, no mesmo espírito do `probe_video_contribution.ipynb`: sondas lineares
  não provam que o modelo usa (ou deixa de usar) a informação de forma não-linear; provam
  só o que é linearmente decodificável nos vetores testados.

## 7. Saída esperada

`EXIST_VIDEOS/analysis/whisper_padding_probe.json` (AUC−nulo por caminho por alvo, Δ entre
caminhos, resultado do portão de cosseno) e
`EXIST_VIDEOS/analysis/whisper_padding_probe_report.txt` (log completo, tee de stdout, mesmo
padrão de `probe_report.txt`).
