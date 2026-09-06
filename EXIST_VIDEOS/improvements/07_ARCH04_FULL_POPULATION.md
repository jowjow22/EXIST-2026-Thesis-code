# 07 — ARCH-04 medido na população completa (adenda)

**Não edita o `01_ARCHITECTURE_FINDINGS.md`.** Aquele documento é do Agente 1 e
o número que ele cita (`n=60`) continua sendo o que ele mediu. Esta adenda
substitui a base de evidência por medição da população inteira.

- **Notebook:** `EXIST_VIDEOS/analysis/audio_duration_analysis.ipynb` (executado, com outputs)
- **Dados brutos:** `analysis/audio_durations.csv` — 3 198 linhas
- **Gráficos:** `analysis/arch04_audio_durations.png`, `analysis/arch04_audio_cdf.png`
- **Data:** 2026-08-08. **Medido localmente**, não na VM (os mp4 estão nos dois lados).

## Método

`ffprobe -select_streams a:0 -show_entries stream=duration` — **stream de áudio**,
não duração do container. `whisper.pad_or_trim` opera sobre a forma de onda que
`torchaudio.load` devolve, que é o stream; usar `format=duration` mediria a coisa
errada. Iteração pelos manifestos (`id_EXIST` → `Path(row["path_video"]).name`),
mesma resolução de caminho de `extract_audio.ipynb`, não por glob de disco.

## Reconciliação de contagens — fecha

| | training | test |
|---|---|---|
| linhas de manifesto | 2 524 | 674 |
| mensuráveis (`ok`) | 2 524 | 674 |
| sem stream de áudio | **0** | **0** |
| arquivo ausente | 0 | 0 |
| `ffprobe` falhou | 0 | 0 |

Total 3 198 = número de `.pt` em `audio_features/`. O 3 199º mp4 em disco é
`6567557435626622214-checkpoint.mp4`, artefato de checkpoint do Jupyter — não é
referenciado por nenhum manifesto e não gerou feature. **Sem divergência real.**

Nenhum arquivo sem stream de áudio: o pior caso hipotético (100 % padding) não
ocorre neste dataset.

## Resultado — training (n=2 524)

| | valor |
|---|---|
| p10 | 8,08 s |
| p25 | 11,26 s |
| **mediana** | **18,19 s** |
| p75 | 40,97 s |
| p90 | 57,81 s |
| média | 26,04 s |
| min / max | 3,34 s / 60,65 s |
| **< 30 s (diluídos)** | **1 682 (66,6 %)** |
| < 15 s | 949 (37,6 %) |
| **> 30 s (truncados)** | **842 (33,4 %)** |
| **`valid_fraction` média** | **0,655** → 34,5 % do vetor `[1280]` médio é encoder de silêncio |
| descartado entre truncados | média 36,1 %, máx 50,5 % |

**Em segundos agregados:** 65 737 s de áudio real no split de treino, dos quais
**16 104 s (24,5 %) nunca chegam ao encoder**, e **26 087 s de silêncio sintético
são injetados** pelo padding. O dataset que o Whisper vê tem mais segundos de
padding do que de áudio descartado.

## Resultado — test (n=674)

Distribuição **deslocada para mais curto** que o treino: mediana 15,05 s,
**76,0 % < 30 s**, `valid_fraction` média **0,582** (41,8 % de silêncio).

**Isto é o achado novo mais importante desta adenda.** O fator de diluição médio
difere entre treino (0,655) e teste (0,582). Como a diluição é uma variável de
confusão embutida em cada embedding, treino e teste têm o confound em
**intensidades diferentes** — o modelo é calibrado sob um regime de diluição e
avaliado sob outro. Isso é *dataset shift induzido pelo pipeline de extração*,
não uma propriedade dos dados. Nenhum agente reportou isso, porque a amostra de
60 era só de treino.

O test também tem uma cauda que o treino não tem: máx 135,1 s (contra 60,65 s), e
o item mais longo perde **77,8 %** do áudio.

## Comparação com a amostra n=60 do ARCH-04

| métrica | n=60 | população (n=2 524) | Δ rel |
|---|---|---|---|
| mediana | 16,40 s | 18,19 s | +10,9 % |
| média | 25,30 s | 26,04 s | +2,9 % |
| p10 | 7,80 s | 8,08 s | +3,6 % |
| máx | 59,10 s | 60,65 s | +2,6 % |
| % < 30 s | 61,7 % | 66,6 % | +8,0 % |
| % < 15 s | 45,0 % | **37,6 %** | **−16,4 %** |
| % > 30 s | 38,0 % | 33,4 % | −12,2 % |
| `valid_fraction` | 0,63 | 0,655 | +4,0 % |

**Veredicto: a amostra segurou.** Todos os números da população caem perto dos
da amostra e **nenhum muda a direção do finding** — a maioria dos itens é
diluída, um terço é truncado, e ~1/3 do vetor médio de áudio é silêncio. O
ARCH-04 fica **mais firme**, não mais fraco.

Uma correção pontual: o `% < 15 s` da amostra (45,0 %) superestimava em 16 %
relativo — o valor real é 37,6 %. É o único número do ARCH-04 que deve ser
citado com o valor novo se for para a tese.

## Estrutura da distribuição — não é suave

O histograma de treino tem **dois picos** que não são ruído:

- **~15 s:** 298 itens (11,8 %) entre 14–16 s
- **~60 s:** 348 itens (13,8 %) entre 55–61 s

São os limites de formato do TikTok. A consequência é que o dano não está
espalhado uniformemente: o cluster de 60 s perde **exatamente cerca de metade**
do áudio (daí o `máx 50,5 % descartado`), e o cluster de 15 s recebe
**exatamente cerca de metade** de padding. Duas modas grandes com defeitos
opostos e de magnitude quase idêntica.

## O que estes números NÃO dizem

1. **Não medem ganho de métrica.** A correção do ARCH-04 melhorar o ICM continua
   hipótese não verificada. Só re-treino confirma.
2. **Duração do stream ≠ duração de fala.** Um vídeo de 40 s pode ter 5 s de fala
   e 35 s de música. `valid_fraction` mede quanto da janela é *sinal*, não quanto
   é *conteúdo linguístico*. O dano medido é um **limite inferior**.
3. **Não validam alinhamento áudio-vídeo.** Frames e áudio são amostrados por
   caminhos independentes; nada aqui testa concordância temporal.

## Consequência para o plano

A correção proposta no ARCH-04 (janelas múltiplas + média só do trecho válido)
endereça os dois defeitos. A adenda acrescenta um argumento que não existia: além
de limpar o confound *dentro* de cada split, ela **iguala o regime de diluição
entre treino e teste** — que hoje diferem em 0,655 vs 0,582.
