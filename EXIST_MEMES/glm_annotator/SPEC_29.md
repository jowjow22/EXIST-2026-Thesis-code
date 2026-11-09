# 29 — Mistura do GLM no soft label, com o viés de YES corrigido por quantis, e controle de suavização

**RASCUNHO de 2026-10-12, para aprovação do orientador. Vira pré-registro quando o Jonathan aprovar; a partir daí, nada aqui muda (mudanças entram como seção nova datada). Nenhum label foi gerado e nenhuma run foi feita sob este documento.**

**Estatuto: exploratório**, motivado pelos docs 27 e 28. Ponderar anotadores pelo desvio ao GLM acabou medindo o limiar de YES de cada anotador: o GLM diz YES em 80% dos memes de treino, os humanos em 55%. Por isso a ponderação empurrou os labels para o YES, e a dose maior (doc 28) piorou o modelo (`eval_loss` +0,099, p = 0,029). Este doc muda a pergunta: em vez de pesar anotadores, **o GLM entra direto no label, depois de removido o viés de YES dele**.

## 1. Os dois braços novos

Para cada meme de treino `i` com referência do GLM (as 3 paráfrases deram YES/NO; 3 381 de 3 386):

- `p_h(i)`: fração de YES entre os votos humanos (o `soft_labels_taskA_raw.json`).
- `p_GLM(i)`: fração de YES entre as 3 paráfrases (`glm_votes.csv`, braço `think_off`): 0, ⅓, ⅔ ou 1.

**Braço `glmq50` (tratamento):** `p_novo = 0,5 · p_h + 0,5 · q(p_GLM)`, onde `q` é a correção por quantis:
1. Ordenar os 3 381 valores de `p_h` (crescente).
2. Os níveis do GLM, do menor para o maior, ocupam blocos consecutivos dessa fila, do tamanho do número de memes em cada nível (254, 241, 739, 2 147).
3. `q(nível)` = média dos `p_h` no bloco correspondente.

Mapa medido em 2026-10-12, antes de qualquer treino: `q(0) = 0`, `q(⅓) = 0,124`, `q(⅔) = 0,303`, `q(1) = 0,756`. Por construção, a média de `q(p_GLM)` é igual à média de `p_h` (0,555): o deslocamento médio dos labels é zero. O GLM decide a **ordem** dos memes, e os humanos decidem a **escala**.

**Braço `smooth50` (controle):** `p_novo = 0,5 · p_h + 0,5 · c`, com `c` = média de `p_h` nos mesmos 3 381 memes (0,555). É a mesma mistura, nos mesmos memes, com a mesma média, **sem nenhuma informação do GLM** (uma suavização de label cega). Sem este braço, um ganho do `glmq50` não pode ser atribuído ao GLM.

**Iguais nos dois braços e ao doc 27:** a validação usa os labels originais (`raw`); as Tasks 2.2/2.3 ficam intocadas; os 5 memes sem referência do GLM ficam com `p_h`. **Diferente do doc 27:** os memes unânimes entre os humanos **também** são misturados, nos dois braços. A mistura é aplicada a todos os memes com referência, porque a pergunta aqui é "o GLM como fonte de label", e não "quais anotadores pesar".

**Efeito nos labels de treino, medido antes de fixar o desenho:**

| braço | deslocamento médio p/ YES | \|Δp\| médio | memes com \|Δp\| ≥ 1/12 | rótulos duros: sobem / descem |
|---|---|---|---|---|
| `glmq50` | 0,000 | 0,114 | 55% | 532 / 364 |
| `smooth50` | 0,000 | 0,135 | 67% | 518 / 0 |

Os 518 rótulos duros que sobem no `smooth50` são todos memes empatados (3-3): 0,5·0,5 + 0,5·0,555 = 0,528. O controle não move nenhum meme para baixo; o `glmq50` move nos dois sentidos.

**Teste offline que motivou o desenho (split-half; 3 anotadores preveem os outros 3; as 20 divisões por meme, nos dois sentidos; só treino; `splithalf_29.py` → `reports/splithalf_29.md`):**

| previsor da outra metade | MSE ↓ | entropia cruzada ↓ |
|---|---|---|
| só a metade (humano puro) | 0,1193 | 1,108 |
| 0,5 · metade + 0,5 · 0,555 (= `smooth50`) | 0,0916 | **0,611** |
| 0,5 · metade + 0,5 · GLM sem correção | 0,1062 | 0,856 |
| 0,5 · metade + 0,5 · GLM quantis (= `glmq50`) | **0,0885** | 0,624 |
| só GLM quantis | 0,1154 | 0,724 |

`[LIMITAÇÃO]` declarada antes: no teste offline, o GLM corrigido supera a suavização cega no MSE (−0,0031, IC 95% [−0,0043; −0,0019]), mas perde na entropia cruzada (+0,0130, IC 95% [+0,0077; +0,0182]). O motivo: quando a metade humana é 0 e o GLM diz 0 de 3 (`q(0) = 0`), o label vai a 0, e a entropia cruzada pune isso sempre que a outra metade tem algum YES. **Expectativa a priori: efeito pequeno.** O teste usa metades de 3 anotadores, e o treino real usa 6, com menos ruído, o que reduz o ganho de qualquer mistura.

## 2. Runs

- BASE reaproveitado: `glmbase` (seeds 42, 7, 2025), pelo mesmo motivo do SPEC_28 §2.
- Notebook: `memes_model.ipynb`, célula 6, `ARM = 'glmq50'` ou `'smooth50'`; 15 épocas; o resto igual.

## 3. Fase 1 — Triagem (seed 42, os dois braços)

Comparar na **época de menor `eval_loss`** de cada run, com os mesmos limites do doc 28 (|Δ| ≥ 0,03 hard, ≥ 0,02 soft, ≥ 0,03 `eval_loss`, Task A, qualquer direção):

- (T1) `glmq50` − `glmbase`
- (T2) `glmq50` − `smooth50`

**Continua (os dois braços, seeds 7 e 2025) se T1 ou T2 passar em pelo menos uma métrica.** Se nenhum passar, o doc 29 encerra com "sem diferença relevante na seed 42".

## 4. Fase 2 — Confirmação

t pareado bilateral, df = 2, estado primário = menor `eval_loss`, como nos docs 27/28.

- **Comparação principal: `glmq50` − `smooth50`** (o efeito atribuível à informação do GLM).
- Secundárias: `glmq50` − `glmbase` e `smooth50` − `glmbase`.

**Critério de "o GLM ajuda":** na comparação principal, Δ médio do ICM-Soft-Norm A > 0 com p < 0,05, **e** `glmq50` − `glmbase` com ICM-Soft-Norm A > 0 (o tratamento também tem de superar o BASE, e não só o controle). Tudo o que ficar abaixo disso se registra como "efeito não detectado com 3 seeds". Um efeito negativo significativo também é registrado.

## 5. O que NÃO fazer

- Não mudar α, o mapa `q`, a constante `c` nem os limites depois de gerar os labels.
- Não rodar α = 0 (só GLM) sem uma seção nova datada, escrita antes de ver a triagem.
- Não usar a seed 42 como evidência se a fase 2 não rodar.

## 6. Resultado

*(preencher depois de rodar, como seção datada)*
