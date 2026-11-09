# Camada de confusão como perda auxiliar da Task A — relatório

Gerado por `reliability/analyze_confusion_experiment.py` a partir dos logs de
`EXIST_MEMES/metrics_history/`. Não treina nem regenera nada.

## 1. O que este experimento pergunta

Se modelar o anotador com uma matriz de confusão 2x2 acrescenta sinal ao
treino sobre soft labels: `L = L_soft(A,B,C) + lambda * L_aux(A)`, com
`p_voto(a,i) = p_latente(i) @ C_a`. O objectivo **não é melhorar o sistema** —
é caracterização. Um nulo é resultado válido e esperado.

**Intervenção:** só um termo somado à perda, e só na Task A. B e C intocadas.
A auxiliar entra apenas no treino: `eval_loss` continua a ser a `L_soft` pura,
logo `E = argmin(eval_loss)` é o mesmo selector de época nos dois lados.

## 2. Integridade `[MEDIDO]`

- [PASS] as 3 runs A0 de 15 épocas existem — todas presentes
- [PASS] existe run de controlo (lambda = 0) — M1/42
- [FAIL] células (método, lambda) com as 3 sementes — M1@0.1, M1@0.3, M1@1.0
- [PASS] cada run tem 15 avaliações — ok

Runs A0 usadas como referência (não re-treinadas): seed 7: `20260831_075933_raw_seed7.json`, seed 42: `20260831_065521_raw_seed42.json`, seed 2025: `20260830_225133_raw_seed2025.json`

## 3. Controlo obrigatório — lambda = 0 `[MEDIDO]`

Com `lambda = 0` o caminho auxiliar é calculado e multiplicado por zero
(não é saltado): um NaN lá dentro ainda contaminaria a perda. A trajectória
tem de reproduzir o A0 da mesma seed.

| método | seed | E (A0 / controlo) | Δ eval_loss em E | Δ primária em E | sd do A0 entre seeds | máx \|Δ\| eval_loss (15 ép, contexto) | veredicto |
|---|---|---|---|---|---|---|---|
| `M1` | 42 | 3 / 3 | +1.78e-03 | -0.0017 | 0.0059 | 1.66e-02 | **PASS** |

**Critério do controlo (recalibrado — ver nota):** as matrizes têm de ficar
congeladas, o A0 tem de reproduzir, e `|Δ da métrica primária em E|` tem de
ficar abaixo de **1 sd do A0 entre seeds**. O máximo ao longo das 15 épocas
aparece só como contexto.

> **Nota — um degrau retirado.** A versão inicial deste protocolo exigia
> reprodução **bit-a-bit** (máx \|Δ\| de `eval_loss` < 1e-3 nas 15 épocas).
> Essa estatística está mal calibrada para um sistema caótico: uma
> perturbação de um ulp no passo 1 amplifica para escala macro na época 15,
> logo o máximo sobre 15 épocas **não distingue um ulp de um bug**. A
> evidência que discrimina é independente e já está medida: as matrizes
> ficam congeladas (sem fuga de gradiente nem weight decay) e o A0
> reproduz bit-a-bit ao longo de 6 dias. O degrau foi retirado **depois**
> de ver o resultado do controlo — está declarado como tal. O critério do
> RESULTADO (secção 5) não foi tocado.

## 4. Grelha método x lambda x seed `[MEDIDO]`

| método | lambda | seed | E | eval_loss(E) | A ICM-Soft-Norm | A0 na mesma seed | Δ |
|---|---|---|---|---|---|---|---|
| `M1` | 0.1 | 42 | 4 | 2.67554 | 0.4716 | 0.4711 | +0.0005 |
| `M1` | 0.3 | 42 | 3 | 2.65824 | 0.4715 | 0.4711 | +0.0003 |
| `M1` | 1.0 | 42 | 3 | 2.66530 | 0.4727 | 0.4711 | +0.0016 |

### 4.1 As seis métricas na época E

| método | lambda | seed | A ICM-Norm hard | A ICM-Soft-Norm soft | B ICM-Norm hard | B ICM-Soft-Norm soft | C ICM-Norm hard | C ICM-Soft-Norm soft |
|---|---|---|---|---|---|---|---|---|
| `M1` | 0.1 | 42 | 0.5261 | 0.4716 | 0.4337 | 0.3897 | 0.2898 | 0.1942 |
| `M1` | 0.3 | 42 | 0.5201 | 0.4715 | 0.4317 | 0.3856 | 0.2887 | 0.1845 |
| `M1` | 1.0 | 42 | 0.5261 | 0.4727 | 0.4209 | 0.3827 | 0.2511 | 0.1785 |

## 5. Resultado por método, no melhor lambda `[MEDIDO]`

A escolha do melhor lambda é **post-hoc** (maior Δ médio) e está declarada
como tal. Critério pré-registado: Δ > 0 e p < 0.0167 
(Bonferroni sobre 3 métodos), t pareado bilateral sobre 3 sementes.

| método | melhor lambda | média A0 | média método | Δ médio | sd | t | p | d_z | mesmo sinal | veredicto |
|---|---|---|---|---|---|---|---|---|---|---|
| `M1` | — | — | — | — | — | — | — | — | — | não corrido |
| `M2` | — | — | — | — | — | — | — | — | — | não corrido |
| `M3` | — | — | — | — | — | — | — | — | — | não corrido |

**Veredicto da família:** NULO — nenhum método passa o critério pré-registado.

## 5b. Braços exploratórios — learning rate próprio para as matrizes `[MEDIDO]`

**Fora da grelha pré-registada e fora da estatística confirmatória.** Com o
`learning_rate` do encoder (1e-5), as matrizes movem-se no máximo ~6e-4 em 15
épocas — o tecto de 795 passos de AdamW — e `lambda` não muda isso, porque o
AdamW é invariante à escala do gradiente e a auxiliar é a única fonte de
gradiente delas. Estes braços dão às matrizes um grupo de parâmetros com lr
próprio, para responder se o nulo sobrevive quando o mecanismo tem margem
para se mover.

| método | lambda | lr das matrizes | seed | E | A ICM-Soft-Norm | A0 | Δ | dist média a I | viés sd | r(viés, yes_rate) |
|---|---|---|---|---|---|---|---|---|---|---|
| `M1` | 0.1 | 0.01 | 7 | 3 | 0.4620 | 0.4610 | +0.0010 | 0.3745 | 0.1728 | +0.952 |
| `M1` | 0.1 | 0.01 | 42 | 3 | 0.4730 | 0.4711 | +0.0019 | 0.3873 | 0.1848 | +0.954 |
| `M1` | 0.1 | 0.01 | 2025 | 3 | 0.4605 | 0.4608 | -0.0003 | 0.3737 | 0.1775 | +0.953 |

## 6. Diagnóstico das matrizes `[MEDIDO]` (não é critério)

Distância média `||C_a − I||_F` na época E e correlação com a `yes_rate` do
anotador. Distância ~0 = o mecanismo colapsou na identidade (não aprendeu
nada sobre os anotadores); correlação com a `yes_rate` = aprendeu o viés de
resposta, que é o que a matriz deveria capturar.

O **viés** é `C_a[NO->YES] − C_a[YES->NO]`, com sinal: é a quantidade que
deveria correlacionar com a `yes_rate` se o mecanismo aprendeu alguma coisa.
A distância de Frobenius é sem sinal e mistura as duas off-diagonais — pode
ficar plana enquanto o viés está estruturado.

| método | lambda | seed | E | dist média | sd | min | max | diag média | viés médio | r(dist, yes_rate) | rho(dist) | r(viés, yes_rate) | rho(viés) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `M1` | 0.1 | 42 | 4 | 0.0948 | 0.0000 | 0.0947 | 0.0948 | 0.9526 | +0.0000 | -0.068 | -0.084 | +0.947 | +0.945 |
| `M1` | 0.3 | 42 | 3 | 0.0948 | 0.0000 | 0.0948 | 0.0948 | 0.9526 | +0.0000 | -0.082 | -0.106 | +0.951 | +0.949 |
| `M1` | 1.0 | 42 | 3 | 0.0948 | 0.0000 | 0.0948 | 0.0948 | 0.9526 | +0.0000 | -0.066 | -0.091 | +0.951 | +0.948 |

## 7. Limitações declaradas

- O split é por **item**, não por anotador: anotadores de treino aparecem em
  validação, o que **favorece** o método. Declarado, não corrigido.
- `L_aux` é a **média** sobre os pares (anotador, item) do batch, não a soma —
  mantém `lambda` na escala da `L_soft`, que também é média.
- `C_a[i,j] = P(voto=j | latente=i)` com linhas normalizadas e
  `p_voto = p_latente @ C_a`. A forma `C_a @ p_latente` do handoff, com linhas
  normalizadas, não devolve uma distribuição.
- Os parâmetros de confusão estão **fora do weight decay** (0.05): o decaimento
  puxaria os logits para a matriz uniforme, o oposto da inicialização perto da
  identidade, e no M3 seria regularização que a especificação proíbe.
- Os pesos de regularização (M1 e M2) foram fixados **a priori**, um valor cada,
  sem grelha — uma grelha neles inflaria as comparações para além do Bonferroni-3.
- O M1 é uma **adaptação** de Chu et al.: implementa a decomposição
  comum + individual, não o portão dependente da instância do CoNAL.
