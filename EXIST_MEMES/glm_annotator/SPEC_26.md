# 26 — Fatorial 2×2 da estrutura do prompt: pré-registro

**Escrito em 2026-10-08, ANTES de rodar.** Mesma regra do doc 25: nada aqui muda depois do primeiro resultado. Mudanças entram como seção nova datada.

## 0. Motivação e estatuto

- O doc 25 terminou na F2: G2 falhou (espanhol com 0,6912 de memes confiantes, mínimo 0,70; `SPEC_25.md` §20). Nos 980 memes não confiantes, a **P3** foi a paráfrase dissidente em **734 (75%)** e respondeu YES bem menos (0,636, contra 0,861 do P1 e 0,913 do P2).
- A P3 difere do P1 em cinco coisas ao mesmo tempo: (1) o texto do meme vem **antes** da instrução; (2) enquadramento "**Task: decide whether…**"; (3) pergunta *"contains sexist expressions or behaviors"*; (4) não define o NO; (5) termina com *"Is this meme sexist under this definition?"*. A hipótese levantada pelo autor é que a **estrutura**, ou seja (1) e (2), é a causa.
- **Estatuto: experimento exploratório, motivado pelo resultado do doc 25 e declarado como tal.** Ele **não** reabre o doc 25, **não** escolhe prompt para calcular pesos e **não** autoriza rodar a F2 de novo. Pergunta só *por que* a P3 diverge.

## 1. Desenho

Fatorial 2×2 a partir do P1, mudando um fator de cada vez e deixando todo o resto do texto idêntico:

| | pergunta direta | "Task: decide whether…" |
|---|---|---|
| **texto do meme depois da instrução** | `direct_after` (= P1, controle) | `task_after` (só enquadramento) |
| **texto do meme antes da instrução** | `direct_before` (só ordem) | `task_before` (os dois) |

- **Memes:** os 240 do piloto do doc 25 (`pilot_ids.json`; 120 en + 120 es), sorteados só por id e idioma.
- **Chamadas:** 4 células × 240 = **960**, braço `think_off`, seed 42, com os mesmos parâmetros, servidor, versão e concorrência (4) do doc 25. Tudo isso é conferido por `assert` contra `config.json`.
- **Referência:** os rótulos P1 e P3 do piloto do doc 25 (`raw/think_off.jsonl`, seed 42). Esses rótulos já existiam antes deste documento; os das 4 células não.
- **Nenhum voto humano é lido.**

## 2. O que NÃO é testado

As diferenças (3), (4) e (5) da P3 ficam fora. Se ordem e enquadramento juntos não reproduzirem a divergência, a conclusão é só "**não é a estrutura**"; a atribuição à redação seria por eliminação, não testada.

## 3. Prompts (texto exato; código em `prompts_26.py`)

**`direct_after`** — SHA-256 `eda963068a2270d49f149c082a8989ca310a68ca655a9c635efb4464a112cd54`
```text
Is the meme sexist, in any form, or does it describe situations in which such discrimination occurs (i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behavior)?

YES: the meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.
NO: the meme does not prejudice, underestimate or discriminate women.
A meme that condemns or reports sexism is still YES.

Remember that sexism may be expressed in different forms! It may sound friendly, funny, offensive, hateful or even violent. Sexism comprises any form of oppression or prejudice against women because of their sex. This discrimination can be based on different beliefs such as stereotypes, the belief that men are superior to women or irrational hatred of women (misogyny). Note that misogyny is a radical/hate-driven sexism.

Judge the image and the text together. Text extracted from the meme (English or Spanish, do not translate it):
{text}

Answer with one word: YES or NO.
```

**`direct_before`** — SHA-256 `c94ed4fbeabcbbca88caa76a0cd55e2356cbceca25649b03d4aa4e63ff79c517`
```text
Text extracted from the meme (English or Spanish, do not translate it):
{text}

Is the meme sexist, in any form, or does it describe situations in which such discrimination occurs (i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behavior)?

YES: the meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.
NO: the meme does not prejudice, underestimate or discriminate women.
A meme that condemns or reports sexism is still YES.

Remember that sexism may be expressed in different forms! It may sound friendly, funny, offensive, hateful or even violent. Sexism comprises any form of oppression or prejudice against women because of their sex. This discrimination can be based on different beliefs such as stereotypes, the belief that men are superior to women or irrational hatred of women (misogyny). Note that misogyny is a radical/hate-driven sexism.

Judge the image and the text together.

Answer with one word: YES or NO.
```

**`task_after`** — SHA-256 `c6abfe45a7604cc517cbdbd1f5ec386371a9484443f7552908fa872418531f1c`
```text
Task: decide whether the meme is sexist, in any form, or whether it describes situations in which such discrimination occurs (i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behavior).

YES: the meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.
NO: the meme does not prejudice, underestimate or discriminate women.
A meme that condemns or reports sexism is still YES.

Remember that sexism may be expressed in different forms! It may sound friendly, funny, offensive, hateful or even violent. Sexism comprises any form of oppression or prejudice against women because of their sex. This discrimination can be based on different beliefs such as stereotypes, the belief that men are superior to women or irrational hatred of women (misogyny). Note that misogyny is a radical/hate-driven sexism.

Judge the image and the text together. Text extracted from the meme (English or Spanish, do not translate it):
{text}

Answer with one word: YES or NO.
```

**`task_before`** — SHA-256 `551b61bdb9e729473ec157eb71509b7c256bdfb8ea9d627f87a317ff60999ecf`
```text
Text extracted from the meme (English or Spanish, do not translate it):
{text}

Task: decide whether the meme is sexist, in any form, or whether it describes situations in which such discrimination occurs (i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behavior).

YES: the meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.
NO: the meme does not prejudice, underestimate or discriminate women.
A meme that condemns or reports sexism is still YES.

Remember that sexism may be expressed in different forms! It may sound friendly, funny, offensive, hateful or even violent. Sexism comprises any form of oppression or prejudice against women because of their sex. This discrimination can be based on different beliefs such as stereotypes, the belief that men are superior to women or irrational hatred of women (misogyny). Note that misogyny is a radical/hate-driven sexism.

Judge the image and the text together.

Answer with one word: YES or NO.
```

## 4. Métricas e leitura (fixadas pelo Jonathan em 2026-10-08)

- `D(x, y)`: fração dos 240 memes em que os rótulos de x e y diferem (`?` em qualquer lado conta como diferente).
- **Controle de ruído:** `D(direct_after, P1 original)` ≤ **0,02**. Se passar disso, o servidor mudou desde o doc 25, o teste é **inválido** e nenhuma leitura vale.
- **Medida principal:** `R(célula) = D(célula, P1) ÷ D(P3, P1)`, a fração da divergência da P3 que a célula reproduz.
- **Leitura**, para cada célula testada (`direct_before`, `task_after`, `task_before`):
  - R ≥ **0,5**: o fator explica a maior parte da divergência da P3;
  - **0,2** ≤ R < 0,5: explica parte;
  - R < 0,2: não explica.
- Se `task_before` (os dois fatores juntos) tiver R < 0,5, a maior parte da divergência vem das diferenças de redação **não testadas** (§2).
- **Descritivos**, sem regra de decisão: taxa de YES por célula (total, en, es); `D` por idioma; `D(célula, P3)`; McNemar exato contra o P1; efeitos principais e interação na taxa de YES.

## 5. Limitações declaradas antes

- n = 240 e uma seed: os efeitos pequenos não terão precisão.
- Uma só realização de cada fator: "enquadramento Task" é **esta** redação, não qualquer enquadramento de tarefa.
- O resultado descreve a sensibilidade **deste** modelo (GLM-4.6V-Flash FP8) a **estas** variações.

## 6. Execução

```bash
cd ~/Documents/projects/VM-THESIS && caffeinate -i python3 -u EXIST_MEMES/glm_annotator/factorial_26.py 2>&1 | tee EXIST_MEMES/glm_reference/factorial26.log
```

Saídas: `raw/factorial26.jsonl`, `reports/factorial26_summary.md`, `reports/factorial26.json`.

## 7. Resultado

*(preencher depois de rodar, como seção datada)*

