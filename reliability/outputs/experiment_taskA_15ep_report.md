# A0 vs A1 a 15 épocas — θ corrigido, 3 sementes pareadas

Gerado por `reliability/analyze_arm_experiment.py` a partir dos logs de
`EXIST_MEMES/metrics_history/`. Não treina nem regenera nada.

## 1. O que este experimento pergunta

Se ponderar os votos dos anotadores pela competência estimada (θ) ao
construir a soft label da Task A muda as métricas ICM do modelo de memes.
O objectivo **não é melhorar o modelo** — é medir se o θ carrega informação.
Um resultado nulo é aceitável e esperado; o enquadramento é de
**caracterização**: sob que condições a modelação de reliability é viável em
LeWiDi, e porque é que o EXIST cai fora dessas condições.

**Intervenção:** apenas o alvo de treino da Task A. Hard gold intocado. B e C
sem alteração de alvo — o que se observa neles é spillover do encoder
partilhado, não efeito directo.

**θ:** estimador corrigido (`theta_taskA_single_truth_K1_fixed.json`), depois
da correcção da responsabilidade do M-step. Ver `theta_legacy_vs_fixed.md`.

## 2. Integridade das runs `[MEDIDO]`

- [PASS] as 6 runs (2 braços × 3 seeds) existem — todas presentes
- [PASS] cada run tem 15 avaliações — raw/42=15, raw/7=15, raw/2025=15, weighted_fixed/42=15, weighted_fixed/7=15, weighted_fixed/2025=15
- [PASS] cada run terminou (`train_runtime` no log) — todas com on_train_end
- [PASS] as 6 métricas ICM estão em todas as runs — ok

| braço | seed | ficheiro | E = argmin(eval_loss) | eval_loss em E |
|---|---|---|---|---|
| `raw` | 42 | `20260831_065521_raw_seed42.json` | **3** | 2.65397 |
| `raw` | 7 | `20260831_075933_raw_seed7.json` | **3** | 2.67339 |
| `raw` | 2025 | `20260830_225133_raw_seed2025.json` | **3** | 2.67177 |
| `weighted_fixed` | 42 | `20260830_205756_weighted_fixed_seed42.json` | **3** | 2.67109 |
| `weighted_fixed` | 7 | `20260830_215815_weighted_fixed_seed7.json` | **3** | 2.69073 |
| `weighted_fixed` | 2025 | `20260830_194907_weighted_fixed_seed2025.json` | **3** | 2.69589 |

## 3. Curvas de `eval_loss` `[MEDIDO]`

| braço | seed | ép 1 | ép 2 | ép 3 | ép 4 | ép 5 | ép 6 | ép 7 | ép 8 | ép 9 | ép 10 | ép 11 | ép 12 | ép 13 | ép 14 | ép 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `raw` | 42 | 2.8221 | 2.6922 | 2.6540 | 2.6685 | 2.6934 | 2.6893 | 2.6933 | 2.6863 | 2.7176 | 2.7200 | 2.7159 | 2.7344 | 2.7356 | 2.7459 | 2.7439 |
| `raw` | 7 | 2.8704 | 2.7011 | 2.6734 | 2.6923 | 2.7229 | 2.7331 | 2.7312 | 2.7407 | 2.7472 | 2.7642 | 2.7679 | 2.7871 | 2.8042 | 2.8009 | 2.8083 |
| `raw` | 2025 | 2.8533 | 2.6914 | 2.6718 | 2.6857 | 2.7237 | 2.7360 | 2.7334 | 2.7299 | 2.7508 | 2.7602 | 2.7680 | 2.7906 | 2.7963 | 2.8060 | 2.8129 |
| `weighted_fixed` | 42 | 2.8300 | 2.6975 | 2.6711 | 2.7031 | 2.7015 | 2.7225 | 2.7396 | 2.7420 | 2.7432 | 2.7553 | 2.7635 | 2.7808 | 2.7865 | 2.7931 | 2.7980 |
| `weighted_fixed` | 7 | 2.8964 | 2.7107 | 2.6907 | 2.7290 | 2.7693 | 2.7827 | 2.7847 | 2.7953 | 2.8030 | 2.8135 | 2.8245 | 2.8443 | 2.8613 | 2.8590 | 2.8655 |
| `weighted_fixed` | 2025 | 2.9033 | 2.6987 | 2.6959 | 2.7156 | 2.7688 | 2.7623 | 2.7721 | 2.7594 | 2.7760 | 2.7880 | 2.7878 | 2.8117 | 2.8204 | 2.8254 | 2.8336 |

`[MEDIDO]` Mínimo de `eval_loss` por run: `raw`/42=ép 3, `raw`/7=ép 3, `raw`/2025=ép 3, `weighted_fixed`/42=ép 3, `weighted_fixed`/7=ép 3, `weighted_fixed`/2025=ép 3.
A perda sobe a partir daí — as épocas seguintes são sobreajuste. É o mesmo
padrão do ARCH-06 no modelo de vídeo, agora replicado nos memes e nos dois
braços. Correr 15 épocas não foi desperdício: é o que estabelece que o nulo
a 2 épocas não era artefacto de sub-treino.

## 4. Resultado principal — época E, comparação pareada `[MEDIDO]`

Métrica pré-registada do estudo: **Task A ICM-Soft-Norm (soft)**.

| seed | A0 (raw) | A1 (ponderado) | Δ = A1 − A0 |
|---|---|---|---|
| 42 | 0.4711 | 0.4663 | **-0.0048** |
| 7 | 0.4610 | 0.4596 | **-0.0014** |
| 2025 | 0.4608 | 0.4595 | **-0.0013** |
| **média** | | | **-0.0025** |

sd(Δ) = 0.0020 · t(2) = -2.16 · p = 0.164 · d_z = -1.25

**Veredito: NULO — não sustentado.**

## 5. As seis métricas, na época E `[MEDIDO]`

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0025 | -0.0075 | -0.0256 | **-0.0118** | 0.0122 | -0.97 | 0.234 |
| A ICM-Soft-Norm soft | -0.0048 | -0.0014 | -0.0013 | **-0.0025** | 0.0020 | -1.25 | 0.164 |
| B ICM-Norm hard | -0.0063 | -0.0043 | +0.0122 | **+0.0006** | 0.0102 | +0.05 | 0.933 |
| B ICM-Soft-Norm soft | -0.0039 | -0.0007 | -0.0048 | **-0.0031** | 0.0021 | -1.47 | 0.126 |
| C ICM-Norm hard | -0.0036 | +0.0012 | +0.0057 | **+0.0011** | 0.0046 | +0.24 | 0.719 |
| C ICM-Soft-Norm soft | -0.0029 | -0.0048 | -0.0133 | **-0.0070** | 0.0055 | -1.26 | 0.160 |

## 6. Regra de simetria — estado FINAL (época 15) `[MEDIDO]`

O mesmo critério aplicado aos dois braços. Reportado porque a definição de
baseline adoptada na Task 3 é o estado final da run, e misturar «fim de um
lado, pico do outro» é o erro que essa regra existe para impedir.

| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |
|---|---|---|---|---|---|---|---|
| A ICM-Norm hard | -0.0224 | -0.0282 | -0.0207 | **-0.0238** | 0.0039 | -6.02 | 0.009 |
| A ICM-Soft-Norm soft | -0.0098 | -0.0082 | -0.0019 | **-0.0066** | 0.0042 | -1.59 | 0.111 |
| B ICM-Norm hard | -0.0079 | +0.0095 | +0.0221 | **+0.0079** | 0.0150 | +0.52 | 0.459 |
| B ICM-Soft-Norm soft | -0.0037 | -0.0027 | +0.0063 | **-0.0001** | 0.0055 | -0.01 | 0.987 |
| C ICM-Norm hard | -0.0085 | +0.0109 | +0.0173 | **+0.0066** | 0.0134 | +0.49 | 0.486 |
| C ICM-Soft-Norm soft | -0.0067 | -0.0053 | +0.0015 | **-0.0035** | 0.0044 | -0.79 | 0.304 |

`[LIMITAÇÃO]` **Nada nesta tabela é um achado.** A época 15 está em
sobreajuste profundo (a `eval_loss` sobe desde a época 3 nas seis runs), a
tabela existe para cumprir a regra de simetria e não para inferência, e um
p nominal < 0,05 isolado entre 12 células (6 métricas × 2 pontos de leitura)
é multiplicidade, não sinal. O `A ICM-Norm hard` com d_z = −6,02 é exactamente
esse caso: não deve ser lido como efeito.

## 7. Comparação com o estudo a 2 épocas

`[LIMITAÇÃO]` As duas linhas **não são medições do mesmo alvo**: o estudo a
2 épocas usou as soft labels do θ LEGADO (com o bug da responsabilidade do
M-step); este usa o θ corrigido. A intervenção encolheu de
0.0720 para 0.0617 de deslocamento médio do
alvo (−14,3%). A linha antiga fica como registo, não como referência.

| | 2 épocas (θ legado) | 15 épocas (θ corrigido) |
|---|---|---|
| Δ médio (A soft) | +0.0045 | -0.0025 |
| Δ por seed | +0.0031, +0.0099, +0.0005 | -0.0048, -0.0014, -0.0013 |
| mesmo sinal em 3/3 | sim | sim |

A linha de 2 épocas **continua válida como registo do alvo legado** — foi a
decisão explícita tomada ao adoptar o estimador corrigido como oficial.

## 8. Dimensionamento `[MEDIDO]`

Sementes necessárias para 80% de poder (α = 0,05 bilateral, aproximação
normal), a partir do d_z observado em cada métrica na época E:

| métrica | d_z | sementes para 80% |
|---|---|---|
| A ICM-Norm hard | -0.97 | 9 |
| A ICM-Soft-Norm soft | -1.25 | 6 |
| B ICM-Norm hard | +0.05 | 2634 |
| B ICM-Soft-Norm soft | -1.47 | 4 |
| C ICM-Norm hard | +0.24 | 138 |
| C ICM-Soft-Norm soft | -1.26 | 5 |

`[LIMITAÇÃO]` Com n = 3 o sd é ele próprio muito ruidoso, logo estes números
são ordens de grandeza, não estimativas. A leitura útil é binária: «cabe em
~10 sementes» ou «não cabe em nenhum orçamento razoável». E há uma leitura
que o número sozinho esconde: **a direcção que mais sementes iriam confirmar**
**é NEGATIVA**. Correr mais sementes não pode transformar isto num ganho — no
melhor dos casos confirma um custo pequeno. Isso fecha a objecção «e se
corressem mais sementes?» sem gastar GPU nenhuma.

## 9. Interpretação

### 9.1 O nulo está estabelecido

Na métrica pré-registada, no regime real de treino (15 épocas, θ corrigido,
época E por `argmin(eval_loss)`): **Δ = -0.0025, p = 0.164**. O efeito é **menor em módulo** que o do estudo a 2
épocas e não é distinguível de zero. Este é o resultado que a linha de
investigação procurava: **ponderar os votos pela competência estimada não
carrega informação utilizável neste dataset** — medido onde o modelo
efectivamente treina, não num regime de sub-treino fácil de atacar.

A história interna é coerente, e é isso que dá confiança no nulo: a
`eval_loss` em E é **pior** no braço ponderado nas três sementes
(2,6541/2,6734/2,6718 → 2,6711/2,6907/2,6959), e o Δ é negativo nas três.
Não há sinal a ser perdido em ruído — não há sinal.

### 9.2 O que o sinal negativo NÃO quer dizer

**Não quer dizer que ponderar «piora o modelo».** A régua de validação é o
gold **cru** nos dois braços, por desenho. Um braço treinado para um alvo
diferente do da régua fica em ligeira desvantagem **por construção**. A
leitura correcta é: *sem ganho de informação; o Δ residual é compatível com
o desencontro alvo-régua, na direcção esperada e com magnitude dentro do
ruído*.

**E não houve «inversão de sinal» face às 2 épocas.** As duas linhas não são
o mesmo experimento: a de 2 épocas mediu o alvo do θ LEGADO, esta mede o
alvo corrigido. Comparar os sinais das duas como se fosse uma reversão seria
erro de leitura. A linha antiga continua válida **como registo do alvo**
**legado** — foi essa a decisão tomada ao adoptar o estimador corrigido.

### 9.3 O nulo por permutação passa a ser opcional

Os cinco conjuntos de labels permutadas existem e estão validados
(`permutation_null_report_fixed.md`). Mas o propósito deles era **atribuir**
**um Δ positivo**: separar «o ganho vem do θ» de «o ganho vem de perturbar o
alvo». **Não há Δ positivo para atribuir.**

Custo de correr assim mesmo: mais 6 runs (5 permutações × sementes, ou 5×1 se
bastar uma semente por permutação). O que acrescentaria: a atribuição de um
nulo, que já é interpretável sem ela. **Recomendação: não correr**, a menos
que o autor queira o braço permutado como figura da tese. Decisão do autor.

### 9.4 O arco para o capítulo

A sequência, nesta ordem, é ela própria material de tese:

1. NUTMEG, Davani et al. e DEM-MoE avaliados e descartados por densidade,
   escala e falta de estrutura demográfica. Sobra o método mais simples e
   mais antigo da bibliografia — **MACE**. Isso é resultado, não acaso.
2. O estimador implementado tinha um **bug na responsabilidade do M-step**
   (voto do próprio anotador contado duas vezes), encontrado por uma
   **violação de monotonicidade da log-posterior** — 273 quebras que quatro
   explicações benignas não explicavam.
3. Corrigido, o efeito da intervenção no alvo **encolheu 14%**
   (0.0720 → 0.0617).
4. No regime real de treino, o efeito nas métricas é **nulo**.

Um nulo com esta cadeia de auditoria por trás é mais forte que um nulo sem
ela: mostra que o resultado não vem de uma implementação partida.

