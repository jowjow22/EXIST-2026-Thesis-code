# 28 — Pesos extremos (r⁴): dose maior da ponderação do doc 27, com triagem em 1 seed

**Escrito em 2026-10-11, ANTES de gerar os labels e de qualquer run.** Mesma regra dos docs 25–27: nada aqui muda depois do primeiro resultado. Mudanças entram como seção nova datada.

**Estatuto: exploratório**, motivado pelo resultado nulo do doc 27 (critério não satisfeito: Δ ICM-Soft-Norm +0,0074, p = 0,107; `eval_loss` pior nas 3 seeds). O objetivo declarado pelo autor é testar uma intervenção com **efeito maior, em qualquer direção**. Os docs 25 e 27 ficam como estão.

## 1. O que muda em relação ao doc 27

Só o peso do voto: **`r⁴` em vez de `r`**, onde `r = r_glm` vem do `annotator_weights.csv` do doc 27 (SHA-256 em `config.json`). O restante segue o doc 27 sem alteração: o desvio, os memes incluídos, `p_new = Σ r⁴·[YES] / Σ r⁴` nos memes de treino, validação igual ao `raw`, e Tasks 2.2/2.3 intocadas.

**Efeito nos labels de treino, medido antes de fixar a dose** (só pesos e votos de treino, nenhum resultado de modelo lido):

| versão | deslocamento médio p/ YES | \|Δp\| médio | memes com \|Δp\| ≥ 1/12 | rótulos duros que mudam |
|---|---|---|---|---|
| doc 27 (r) | +0,054 | 0,065 | 34,6% | 554 |
| **doc 28 (r⁴)** | **+0,120** | **0,154** | **62,5%** | **849** (19 de YES → NO) |

`[LIMITAÇÃO]` declarada antes: r⁴ amplifica o mesmo problema da checagem (b) do doc 27. O peso reflete sobretudo o limiar de YES de cada anotador, então a dose maior é em boa parte "mais YES". Um efeito do doc 28 deve ser lido como o efeito de deslocar o treino para o YES pelo julgamento do GLM, e não como o efeito de "anotadores melhores".

## 2. Braço e runs

- Braço **`glmw4`**: `reliability/outputs/soft_labels_taskA_glmw4.json`, gerado por `doc28.py build` (SHA-256 em `config.json → runtime.doc28`).
- **BASE reaproveitado:** as runs `glmbase` do doc 27 (seeds 42, 7, 2025). O treino é determinístico (três execuções de `glmbase` seed 7 deram `eval_loss` idêntica época a época), e o ambiente da VM não mudou. Se o ambiente mudar, o BASE tem de ser retreinado.
- Notebook: `memes_model.ipynb`, célula 6, `ARM = 'glmw4'`; 15 épocas; o resto igual.

## 3. Fase 1 — Triagem (seed 42) — regra fixada pelo Jonathan em 2026-10-11

Rodar só `glmw4` com seed 42 e comparar com `glmbase` seed 42 na **época de menor `eval_loss`** de cada run.

**Continua para as seeds 7 e 2025 se pelo menos um destes, em qualquer direção:**
- |Δ ICM-Norm hard, Task A| ≥ **0,03**
- |Δ ICM-Soft-Norm soft, Task A| ≥ **0,02**
- |Δ `eval_loss`| ≥ **0,03**

Os limites são o dobro do Δ da seed 42 no doc 27 (+0,016 hard, +0,004 soft, +0,015 loss), que acabou nulo.

**Se nenhum passar:** o doc 28 encerra com "a dose maior não produziu diferença relevante na seed 42", e segue-se para a opção A (mistura do GLM no label, doc seguinte).

`[LIMITAÇÃO]` A triagem não é um teste de significância. Ela decide só se vale gastar as outras duas seeds. Um "passou" na triagem **não** é resultado.

## 4. Fase 2 — Confirmação (se a triagem passar)

Seeds 7 e 2025 do `glmw4`. Análise idêntica à do doc 27: Δ = glmw4 − glmbase por seed, estado primário = menor `eval_loss`, t pareado **bilateral**, df = 2, critério "Δ médio de ICM-Soft-Norm > 0, p < 0,05 e `eval_loss` sem piora". Reportar sempre o Δ por seed, a média e o t. Um nulo se registra como "efeito não detectado com 3 seeds". Um efeito **negativo** significativo (p < 0,05 com Δ < 0) também é registrado como resultado, já que o objetivo declarado é ver efeito em qualquer direção.

Comando: `python3 EXIST_MEMES/glm_annotator/analyze_f5.py --treat glmw4`.

## 5. O que NÃO fazer

- Não trocar a dose (k) depois de ver a seed 42.
- Não mudar os limites da triagem.
- Não usar a seed 42 da triagem como "evidência" se a fase 2 não rodar.

## 6. Resultado

*(preencher depois de rodar, como seção datada)*

## Resultado de 2026-10-12 (preenche o §6) — triagem passou; critério não satisfeito; `eval_loss` significativamente pior

Seção datada; nada acima foi editado.

**Triagem (seed 42, §3)** (`reports/doc28_screen.md`): passou nas 3 métricas: hard −0,0778, soft −0,0492, `eval_loss` +0,1269. Por isso as seeds 7 e 2025 rodaram.
Na seed 42, a menor `eval_loss` do `glmw4` caiu na época 2 (2,7808), quase empatada com a época 4 (2,7815). Na época 4 os Δ seriam hard −0,006 e soft −0,009. A triagem passaria mesmo assim, pela `eval_loss`. Nas seeds 7 e 2025, E = época 3 nos dois braços.

**Fase 2, estado primário (E = argmin `eval_loss`), t pareado bilateral, df = 2** (`reports/f5_report_glmw4.md`):

| métrica | Δ 42 | Δ 7 | Δ 2025 | média | t | p |
|---|---|---|---|---|---|---|
| A ICM-Soft-Norm soft (primária) | −0,0492 | −0,0222 | −0,0026 | **−0,0247** | −1,83 | 0,209 |
| A ICM-Norm hard | −0,0778 | −0,0350 | −0,0257 | −0,0462 | −2,88 | 0,102 |
| `eval_loss` (↓ melhor) | +0,1269 | +0,1014 | +0,0680 | **+0,0988** (pior) | 5,79 | **0,029** |
| B / C (ICM hard e soft) | — | — | — | −0,011 a −0,024 | ≥ −1,97 | ≥ 0,187 |

**Critério do §4: não satisfeito** (Δ soft < 0). **Efeito negativo registrado (§4):** a `eval_loss` piora nas 3 seeds, com p = 0,029. Nas métricas ICM, as 3 seeds dão Δ negativo na Task A, mas sem p < 0,05.

**Estado final (secundário, só descritivo):** Task A soft −0,0465 (3/3 seeds, p = 0,002); hard −0,0058 (p = 0,787; sinal muda por seed); `eval_loss` +0,288 (p < 0,001).
