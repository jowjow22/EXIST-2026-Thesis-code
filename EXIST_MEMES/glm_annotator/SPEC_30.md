# 30 — Referência do GLM com thinking ligado (o viés de YES se mantém?)

**Pré-registro de 2026-10-12, escrito antes de qualquer chamada nova. Mudanças só como seção nova datada.**

**Estatuto: descritivo.** Docs 27–29 mostraram que o GLM (`think_off`) diz YES em 80,4% dos memes de treino, contra 55,5% dos humanos. No piloto (240 memes, 720 chamadas), o `think_on` deu 0,70 e o `think_off` 0,79. A pergunta: com o thinking ligado em todo o treino, o viés some ou se mantém? Se se mantiver, o passo seguinte é testar outros modelos (Claude, GPT, DeepSeek), como pediu o orientador.

## 1. O que roda

- `run_reference_think_on.py`: braço `think_on`, P1–P3, seed 42, os 3 386 memes de treino. Parâmetros idênticos ao doc 25 (`config.json['fixed']`: temp 0,1 · top_p 0,6 · top_k 2 · rep 1,1 · `max_tokens` 16 384 · timeout 600 s · concorrência 4).
- As 720 chamadas `think_on` do piloto são reaproveitadas (mesma chave). Faltam 9 438.
- Grava `glm_reference/glm_votes_think_on.csv`, `reports/think_on_vs_off.md` e `runtime.doc30`. **Não toca** em `glm_votes.csv`, `chosen_arm`, portões nem nos soft labels dos docs 27–29.

## 2. O que se mede (total, en, es)

Taxa de YES (`think_off` vs `think_on`), média de p_YES dos humanos, parse rate e fração de memes confiantes do `think_on`, concordância de voto on/off e distribuição de p_GLM (0, ⅓, ⅔, 1).

## 3. Regra de decisão (fixada agora)

O viés **persiste** se a taxa de YES do `think_on` no treino ≥ média humana + 0,10 (≥ 0,655). Nesse caso: testar outros modelos. Abaixo disso: o viés é, em boa parte, efeito do thinking desligado; discutir com o orientador antes de gastar em API.

## 4. O que NÃO fazer

- Não trocar a referência dos docs 27–29 pelo `think_on` sem uma seção nova datada.
- Não mexer na regra do §3 depois de ver o resultado.

## 5. Resultado

### Resultado de 2026-10-13

10 158 chamadas `think_on` completas; parse 0,9985 (en 0,9983 · es 0,9988). Relatório: `reports/think_on_vs_off.md`.

| escopo | humanos | YES think_off | YES think_on | Δ (on − off) | confiantes on | voto igual on/off |
|---|---|---|---|---|---|---|
| total | 0,555 | 0,804 | 0,711 | −0,093 | 0,747 | 0,853 |
| en | 0,542 | 0,793 | 0,708 | −0,085 | 0,764 | 0,875 |
| es | 0,568 | 0,817 | 0,715 | −0,102 | 0,730 | 0,831 |

p_GLM (0 / ⅓ / ⅔ / 1): `think_on` 608 / 255 / 596 / 1 922 · `think_off` 254 / 241 / 739 / 2 147.

**Decisão pela regra do §3: o viés de YES PERSISTE** (0,711 ≥ 0,655). Próximo: testar outros modelos.

Descritivo extra (fora da regra; 3 381 memes com referência; acerto exclui os empates 3-3):

| braço | acerto vs maioria humana | Spearman(p_GLM, p_h) | YES quando humanos dizem NO | NO quando humanos dizem YES |
|---|---|---|---|---|
| `think_off` | 0,707 | 0,496 | 0,674 | 0,032 |
| `think_on` | 0,753 | 0,501 | 0,488 | 0,082 |

Leitura: o thinking reduz o viés (o YES em memes que a maioria humana diz NO cai de 67% para 49%), mas a **ordenação** dos memes quase não muda (Spearman 0,496 → 0,501). Ou seja, o thinking desloca o limiar do GLM, não melhora o que ele vê. Para um método que usa quantis (doc 29), os dois braços dão praticamente a mesma informação.
