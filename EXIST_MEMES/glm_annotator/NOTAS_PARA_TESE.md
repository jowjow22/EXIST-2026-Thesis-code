# Notas para a tese — anotador de referência (GLM) e ponderação de votos por desvio (doc 25)

**O que este documento é:** registro dos fatos produzidos pelo experimento do doc 25
(`EXIST_MEMES/glm_annotator/`), no formato em que podem ir para a tese. Cada número tem
fonte. O que é hipótese está marcado como hipótese. É atualizado fase a fase; as seções
de fases ainda não executadas ficam marcadas como **pendente**.

**Convenção** (a mesma de `reliability/NOTAS_PARA_TESE.md`):
- `[MEDIDO]` — número observado, com o artefato que o produziu.
- `[HIPÓTESE]` — interpretação plausível, ainda não testada. **Não escrever como fato.**
- `[LIMITAÇÃO]` — restrição a declarar explicitamente no texto.

**Enquadramento narrativo:** um teste único, pedido pelo orientador e sabidamente fora da
prática LeWiDi. Um LLM multimodal faz o papel de "anotador de referência"; mede-se o
desvio de cada anotador humano em relação a ele e pondera-se o voto de cada um por esse
desvio. A diferença para a linha de reliability anterior (MACE: Δ = −0,0025, p = 0,164) é
que **a referência é externa aos votos**: não é inferida dos próprios anotadores que vai
auditar. Critério de sucesso fixado pelo autor antes de rodar: ganho mensurável em ICM ou
`eval_loss` na Task 2.1. Contribuição só metodológica não basta.

---

## 1. Desenho (formulação para o capítulo de método)

> O protocolo foi pré-registrado (`SPEC_25.md`, 2026-10-04) antes de qualquer chamada ao
> modelo, com seis fases separadas por portões de falha dura. As fases que constroem a
> referência (F0–F2) não leem nenhum voto humano, para que a referência não seja ajustada
> aos anotadores que vai auditar. Só a partir da F3 os votos humanos entram, para calcular
> os pesos.

| fase | o que faz | portão |
|---|---|---|
| F0 | integridade do servidor e do caminho visual | G0 (sonda visual, revisão manual) |
| F1 | piloto: escolha do modo de raciocínio; estabilidade entre seeds | G1 |
| F2 | referência nos 3 386 memes de treino | G2 |
| F3 | pesos dos anotadores e novos soft labels (só cálculo) | testes do Apêndice A |
| F4 | checagens (a)–(g) e decisão de retreino, sem GPU | G4 |
| F5 | retreino BASE vs GLMW, 3 seeds pareadas | leitura pré-registrada (§11) |

**Configuração do modelo** `[MEDIDO — glm_reference/config.json]`: `glm-4.6v-flash`
(checkpoint `cont1037/GLM-4.6V-Flash-FP8-DYNAMIC`), servido por vLLM
`0.29.1rc1.dev452+g3df4ae153`; `temperature=0,1`, `top_p=0,6`, `top_k=2`,
`repetition_penalty=1,1`; uma mensagem de usuário `[imagem, texto]`, sem system prompt;
instrução em inglês e texto do meme no idioma original. O voto do GLM por meme é a maioria
de **três paráfrases** do prompt (P1–P3); o GLM é "confiante" num meme quando as três
concordam, e só esses memes entram no cálculo do desvio.

`[LIMITAÇÃO]` As diretrizes de anotação de memes do EXIST não são públicas. O prompt P1
adapta a pergunta mostrada aos anotadores de tweets (EXIST 2023) e a regra de escopo dos
memes (EXIST 2026, §2.1); é a melhor aproximação pública, não a instrução original.

`[LIMITAÇÃO]` O checkpoint é quantizado (FP8). Ver §2 para a sonda que o justifica.

---

## 2. F0 — Por que se testou a visão antes de tudo

**Formulação:**

> Checkpoints públicos quantizados podem degradar a torre de visão sem erro visível: o
> modelo continua a responder em formato válido, decidindo só pelo texto. Como o texto do
> meme também vai no prompt, esse modo de falha não seria detectável pelas métricas da
> classificação. Antes de qualquer chamada do experimento, o caminho visual foi sondado
> com um controle negativo.

**Procedimento:** 20 memes do piloto, prompt *"Describe the image in one sentence. Do not
transcribe its text."*, uma vez com a imagem real e outra com um PNG branco de 512×512.
Critério pré-registrado: as descrições da imagem branca descrevem algo vazio **e** ≥ 16/20
descrições reais correspondem ao meme.

**Resultado** `[MEDIDO — glm_reference/probe_vision.csv, revisão manual do autor em 2026-10-07]`:
- imagem branca: **20/20** descrições do tipo *"a blank white square with no visible
  content, text, or discernible features"*. Nenhuma alucinação de cena;
- imagem real: **20/20** descrições correspondentes ao meme (revisão manual).

**Demais testes de fumaça** `[MEDIDO — config.json → runtime]`:
- thinking desligado funciona como declarado: 3/3 respostas sem raciocínio, com **4 tokens**;
- vazão com 4 requisições simultâneas: latência mediana **2,5 s** nas descrições,
  **73,6 chamadas/min**.

`[LIMITAÇÃO]` A sonda verifica que a visão **funciona**, não que é **tão boa** quanto a do
modelo não quantizado. Não há comparação com o checkpoint BF16.

---

## 3. F1 — Piloto: escolha do modo de raciocínio

**Formulação:**

> A escolha entre raciocínio ligado e desligado foi feita num piloto de 240 memes de treino
> (120 em inglês, 120 em espanhol, sorteados só por id e idioma), sem consultar votos
> humanos. O critério foi a **consistência do modelo consigo mesmo** sob reformulação do
> prompt, não a concordância com anotadores, que tornaria a referência circular.

**Regra pré-registrada (§6 do SPEC_25):** um braço é elegível se `parse_rate ≥ 0,98` e
concordância entre paráfrases ≥ 0,70, ambos em cada idioma, e `0,05 ≤ yes_rate ≤ 0,95`.
Se os dois forem elegíveis, fica o raciocínio **desligado**, salvo ganho de concordância
> 0,05 do ligado.

**Resultado** `[MEDIDO — glm_reference/reports/pilot_summary.md; 1 440 chamadas, seed 42]`:

| braço | concordância entre paráfrases (total · en · es) | `yes_rate` | latência mediana | tokens gerados (mediana) |
|---|---|---|---|---|
| raciocínio ligado | 0,758 · 0,767 · 0,750 | 0,700 | 15,7 s | 171 |
| raciocínio desligado | 0,750 · 0,775 · **0,725** | 0,793 | 0,7 s | 4 |

- `parse_rate` = **1,000** nos dois braços, em todos os escopos; 0 respostas truncadas,
  0 recusas, 0 erros de API.
- Os dois braços foram elegíveis. Ganho do raciocínio = 0,758 − 0,750 = **0,008**, abaixo do
  limiar de 0,05 → **escolhido o raciocínio desligado**, ~22× mais rápido.

**Achado — o raciocínio não aumentou a consistência** `[MEDIDO]`:

> Com o raciocínio ligado, o modelo produziu cadeias de raciocínio reais (mediana de 753
> caracteres, nenhuma vazia, nenhuma com caracteres CJK, máximo de 1 188 tokens) e ainda
> assim concordou entre paráfrases praticamente na mesma taxa (0,758 vs 0,750). Neste
> piloto, raciocinar não tornou o julgamento binário de sexismo mais estável à redação do
> prompt.

`[LIMITAÇÃO]` Uma amostra de 240 memes e uma seed. A diferença de 0,008 não é um teste de
igualdade, só fica abaixo do limiar de decisão pré-registrado.

**Achado — sensibilidade à redação concentrada numa paráfrase**
`[MEDIDO — calculado de glm_reference/raw/<braço>.jsonl, análise descritiva de 2026-10-07]`:

| braço | YES em P1 | YES em P2 | YES em P3 | paráfrase dissidente nos memes não confiantes |
|---|---|---|---|---|
| raciocínio desligado | 0,838 | 0,887 | **0,654** | P1: 2 · P2: 14 · **P3: 44** (de 60) |
| raciocínio ligado | 0,771 | 0,754 | **0,575** | P1: 13 · P2: 9 · **P3: 36** (de 58) |

Nos dois braços, a P3 é a fonte principal da não confiança e a que menos responde YES.

`[HIPÓTESE]` A P3 é a única que põe o texto do meme **antes** da instrução e pergunta se o
meme *"contains sexist expressions or behaviors"*. Essa formulação pode ser lida como mais
restritiva que a de P1/P2 ("is sexist itself, describes a sexist situation or criticizes a
sexist behavior"). Não testado: separar ordem e redação exigiria paráfrases novas, o que o
pré-registro proíbe neste experimento.

**Viés para o YES** `[MEDIDO]`: os dois braços respondem YES em 70–79% das chamadas,
dentro da faixa de não degenerescência (0,05–0,95). A comparação com a taxa humana é
deliberadamente adiada para a F4, checagem (f), para não contaminar a escolha do braço.

`[LIMITAÇÃO]` **Margem apertada em espanhol:** 0,725 contra o mínimo de 0,70. A F2 repete o
critério nos 3 386 memes de treino (portão G2); se o espanhol cair abaixo de 0,70 ali, o
experimento encerra (§11).

### 3.1 F1b — Estabilidade entre seeds

**Procedimento:** braço escolhido (raciocínio desligado), P1, seeds 42, 43 e 44 nos 240 memes
do piloto. `flip_rate` = fração de memes cujos três rótulos não são todos iguais (`?` conta
como flip). Portão G1: `flip_rate` total ≤ 0,10.

**Resultado** `[MEDIDO — glm_reference/reports/pilot_summary.md; raw/think_off.jsonl, 480 chamadas novas]`:

| escopo | `flip_rate` |
|---|---|
| total | **0,0042** (1/240) |
| en | 0,0000 |
| es | 0,0083 (1/120) |

- Um único meme mudou de rótulo: `110614` (NO · YES · NO nas seeds 42 · 43 · 44).
- Em **237/240** memes o texto da resposta foi idêntico, caractere a caractere, nas três seeds.
- 0 erros, 0 falhas de transporte, 100% de respostas parseadas. **G1 aprovado.**

**Achado — a redação do prompt pesa ~60× mais que a amostragem** `[MEDIDO]`:

> Nos 240 memes do piloto, trocar a seed muda o rótulo do GLM em 0,4% deles (1/240);
> trocar a redação do prompt (P1/P2/P3, mesma seed) produz respostas divergentes em 25%
> (60/240; concordância entre paráfrases de 0,750, §3). Estes memes não perdem o voto do
> GLM (que é a maioria das três paráfrases): saem apenas do cálculo do desvio.
> A instabilidade relevante da referência não vem do ruído de amostragem, vem da
> formulação da pergunta. Isso justifica, depois do fato, o desenho com três paráfrases e
> a regra de "confiante = as três concordam": uma referência de prompt único teria
> parecido estável (flip de 0,4%) enquanto escondia uma dependência de redação de 25%.

`[LIMITAÇÃO]` Medida de piloto (n = 240). A taxa na base de treino completa (n = 3 386) é a
que a F2 reporta (§4); o IC 95% aproximado dos 25% do piloto é ~20–31%.

`[LIMITAÇÃO]` Com `temperature=0,1` e `top_k=2`, a geração é quase determinística por
construção. Um `flip_rate` baixo era esperado e é um teste fraco de estabilidade: mede o
ruído de amostragem residual, não a robustez do julgamento. A robustez é o que a
concordância entre paráfrases mede (§3).

---

## 4. F2 — Referência completa: **G2 falhou, experimento encerrado**

**Formulação:**

> Na base de treino completa, o GLM ficou confiante (três paráfrases concordando) em 72,7%
> dos memes em inglês e em 69,1% dos memes em espanhol. O critério pré-registrado exigia
> ≥ 70% em cada idioma. Pela leitura fixada antes da execução (§11 do SPEC_25), a
> referência não sustenta a ponderação dos anotadores, e o experimento foi encerrado na F2,
> sem calcular pesos e sem gastar GPU.

**Resultado** `[MEDIDO — glm_reference/reports/reference_summary.md, glm_votes.csv; 3 386 memes de treino, 10 158 chamadas (9 438 novas + 720 do piloto)]`:

| escopo | memes | `parse_rate` | confiantes | exigido |
|---|---|---|---|---|
| total | 3 386 | 0,9985 | 0,7091 | — |
| en | 1 715 | 0,9983 | **0,7265** (1 246) | ≥ 0,70 ✓ |
| es | 1 671 | 0,9988 | **0,6912** (1 155) | ≥ 0,70 ✗ (mínimo 1 170: **faltaram 15 memes**) |

**A falha não se deve à infraestrutura** `[MEDIDO — raw/think_off.jsonl]`: as 15 chamadas
não parseadas são HTTP 400 em **5 memes** (3 en, 2 es), todas pela mesma causa: a imagem gera
6 090–6 144 tokens, acima do cache de encoder do servidor (6 084). Mesmo contando os 2 memes
em espanhol como confiantes, seriam 1 157 < 1 170.

**O piloto superestimou a confiança** `[MEDIDO]`: piloto (n = 240) → base completa
(n = 3 386): total 0,750 → 0,709; espanhol 0,725 → 0,691. O espanhol completo (30,9% de não
confiantes) ficou no limite superior do intervalo aproximado do piloto (~20–31%), §3.1.

**A sensibilidade à redação se confirmou e está concentrada numa paráfrase**
`[MEDIDO — raw/think_off.jsonl, seed 42, análise descritiva de 2026-10-08]`:
- YES por paráfrase: P1 0,861 · P2 0,913 · **P3 0,636**;
- nos 980 memes não confiantes com as três respostas parseadas, a paráfrase dissidente foi
  P1 em 39, P2 em 207 e **P3 em 734 (75%)**.

`[HIPÓTESE]` A P3 (texto antes da instrução; pergunta se o meme *"contains sexist expressions
or behaviors"*) induz uma leitura mais restritiva do conceito. Com outra redação, ou sem a P3,
o critério provavelmente passaria. **Isto não pode ser testado neste experimento:** escolher
ou descartar paráfrases depois de ver o resultado é exatamente o ajuste que o pré-registro
proíbe (§11, §15). Uma continuação teria de ser um experimento novo, declarado como motivado
por este resultado.

**Leitura para a tese:**

> Um LLM multimodal usado como anotador de referência para sexismo em memes produz um
> julgamento binário estável sob amostragem (0,4% de flips entre seeds), mas dependente da
> formulação da pergunta: três paráfrases da mesma definição discordam em ~29% dos memes,
> e uma única paráfrase responde por três quartos dessa discordância. A "verdade" que o
> modelo oferece é, em parte, um artefato do prompt. Isso desqualifica-o como referência
> externa para auditar anotadores humanos no regime testado. É um resultado negativo
> coerente com a linha de reliability da tese: nem a referência interna (MACE, θ) nem a
> externa (LLM) produziram uma régua estável o bastante para ponderar anotadores do EXIST.

`[LIMITAÇÃO]` Um modelo (GLM-4.6V-Flash, FP8), uma configuração de amostragem, três paráfrases
e só a Task 2.1. A conclusão vale para essa referência, não para LLMs em geral.

`[LIMITAÇÃO]` Limite de infraestrutura: imagens acima de ~6 084 tokens de encoder são
rejeitadas pelo servidor (5/3 386 memes). Um experimento futuro precisa redimensionar as
imagens ou aumentar o cache de encoder do vLLM.

## 5. F3 — Pesos e novos soft labels — **não executada** (G2 falhou)

## 6. F4 — Checagens e decisão de retreino — **não executada** (G2 falhou)

## 7. F5 — Retreino — **não executada** (G2 falhou)

---

## 8. Disciplina metodológica (para o capítulo de método)

`[MEDIDO]` Práticas aplicadas neste experimento:

1. **Pré-registro com portões de falha dura.** Todos os parâmetros e limiares estão em
   `config.json` e são conferidos por `assert` no início de cada fase. Uma mudança após ver
   resultados aborta a execução.
2. **Separação estrita entre construir e auditar.** F0–F2 não leem votos humanos; a função
   que sorteia o piloto recebe só ids e idiomas, e a assinatura dela é conferida em código.
3. **Retentativa só para falha de transporte, nunca de parse.** Repetir uma chamada até
   "dar certo" seria seleção. Chamada gravada nunca é refeita; a execução é retomável.
4. **Desvios documentados como adendos datados, sem editar o original:** §18 (encaixe no
   repositório, 2026-10-06) e §19 (concorrência 8 → 4 para não sobrecarregar o servidor
   partilhado, 2026-10-07, antes de qualquer chamada de chat).
5. **Implementação validada contra casos calculados à mão** (Apêndice A, 9 testes) e de
   ponta a ponta contra um servidor falso, antes de qualquer chamada real.

---

## 9. Artefatos

| arquivo | conteúdo |
|---|---|
| `EXIST_MEMES/glm_annotator/SPEC_25.md` | pré-registro completo + adendos §18, §19 |
| `EXIST_MEMES/glm_annotator/RUNBOOK.md` | passo a passo de execução |
| `EXIST_MEMES/glm_annotator/*.py` | cliente, fases F0–F5, testes |
| `EXIST_MEMES/glm_reference/config.json` | parâmetros fixados + estado dos portões e versão do servidor |
| `EXIST_MEMES/glm_reference/probe_vision.csv` | sonda visual (§2) |
| `EXIST_MEMES/glm_reference/raw/think_on.jsonl` · `think_off.jsonl` | uma linha por chamada, com resposta, raciocínio, parâmetros e hashes |
| `EXIST_MEMES/glm_reference/reports/pilot_summary.md` | métricas do piloto e regra de escolha (§3) |
| `EXIST_MEMES/glm_reference/reports/reference_summary.md` · `glm_votes.csv` | referência completa e portão G2 (§4) |

---

*Escrito em 2026-10-07 (F0, F1) e atualizado em 2026-10-08 (F1b, F2: G2 falhou, experimento encerrado). Números sem fonte explícita
não existem neste documento.*
