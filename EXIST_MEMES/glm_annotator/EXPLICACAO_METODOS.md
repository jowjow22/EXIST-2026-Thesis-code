# Como funcionam os métodos dos docs 25, 27 e 28 (e a proposta do doc 29)

*Documento de apoio para a reunião de orientação. Os resultados de desempenho estão nos gráficos. Aqui o foco é o mecanismo: o que cada método calcula, por que existiu e em que difere dos outros. Toda conta do exemplo numérico foi feita rodando as funções do próprio código (`weights.py` e `_archive_doc25/weights.py`).*

---

## 0. Termos usados ao longo do texto

- **Soft label.** No EXIST, cada meme foi rotulado por 6 anotadores humanos. Em vez de reduzir os 6 votos a um rótulo único (YES ou NO), o paradigma *Learning with Disagreement* (LeWiDi) treina o modelo com a distribuição dos votos. Na Task 2.1 (binária: o meme é sexista?), o soft label é `p_YES` = fração de votos YES. Com 4 YES e 2 NO, por exemplo, `p_YES` = 4/6 = 0,667.
- **Entropia binária** `H(q)`. Mede, em bits, o quanto uma proporção `q` está dividida: `H(q) = −q·log₂q − (1−q)·log₂(1−q)`. Vale 0 quando todos concordam (`q` = 0 ou 1) e 1 quando a divisão é meio a meio (`q` = 0,5). Aqui ela serve para medir o consenso entre os anotadores.
- **ICM-Norm e ICM-Soft-Norm.** O ICM (*Information Contrast Measure*) é a métrica oficial do EXIST, calculada pela biblioteca PyEvALL. É uma medida de base informacional que valoriza mais os acertos nas classes menos frequentes. A versão *Norm* é normalizada para [0, 1], e quanto maior, melhor. O ICM-Norm (*hard*) compara a classe prevista com a classe de referência. O ICM-Soft-Norm compara a distribuição prevista com o soft label humano e é a métrica primária sob LeWiDi.
- **`eval_loss`.** Perda do modelo no conjunto de validação, calculada sempre contra os labels humanos originais. Quanto menor, melhor.
- **Portão.** Nos três documentos, o protocolo foi pré-registrado (escrito antes de rodar), com condições de parada fixadas de antemão. Se uma condição falha, o experimento para ali, sem ajustes. Quando este texto fala em "portão", é uma dessas condições.

---

## 1. Visão geral

Os 6 anotadores de cada meme frequentemente discordam, e o soft label original trata todos os votos como igualmente válidos. A pergunta comum aos três documentos, feita pelo orientador, é se vale a pena dar menos peso aos anotadores que se afastam de um julgamento de referência **externo** aos próprios votos. A referência é o modelo multimodal GLM-4.6V-Flash, que respondeu YES ou NO para cada meme de treino com três redações diferentes do mesmo pedido (paráfrases P1, P2 e P3). Mede-se então o desvio `D` de cada anotador em relação ao GLM, transforma-se esse desvio em um peso `r` e recalcula-se o soft label de treino como uma média ponderada dos votos. O modelo de memes (XLM-R + CLIP) é retreinado com esses novos labels e avaliado na validação com os labels originais.

---

## 2. A estrutura comum aos três métodos

Os três métodos seguem a mesma linha de montagem, em quatro passos. Eles diferem só nos passos 1 e 3.

1. **Referência do GLM por meme**, construída a partir das três paráfrases.
2. **Desvio do anotador.** Para cada anotador `a` e cada meme `i` que entra no cálculo:
   - `d_ai` = o quanto o voto de `a` discorda da referência do GLM no meme `i` (a definição muda entre os docs);
   - `w_ai = 1 − H(fração de YES entre os **outros** 5 votos do meme)`, o **peso de consenso**. O voto do próprio anotador não entra (*leave-one-out*). Se os outros 5 estão de acordo entre si (5-0), `w` = 1. Com 4-1, `w` = 0,278. Com 3-2, `w` = 0,029. A ideia é que discordar do GLM quando todos os outros concordam com ele é um sinal forte, enquanto discordar num meme em que os próprios humanos se dividem quase não conta;
   - `D_a = Σ_i w_ai·d_ai / Σ_i w_ai`: média ponderada dos desvios do anotador sobre os memes dele que entram no cálculo. Cada anotador vê cerca de 26 memes. Se nenhum meme dele entra, `r_a` = 1.
3. **Peso do voto:** `r_a = 1 − D_a` (no doc 28, `r_a⁴`).
4. **Novo soft label** de cada meme de treino: `p_novo = Σ_a r_a·[voto_a = YES] / Σ_a r_a`. É a fração de YES em que cada voto conta com o peso do seu anotador. Se todos os pesos do painel forem iguais, `p_novo = p_original`.

**Memes excluídos do cálculo do desvio, nos três métodos:** os **unânimes** (6-0). Num meme unânime, os 6 anotadores têm o mesmo `d`, qualquer que seja a resposta do GLM. Punir os seis igualmente não distingue ninguém e só dilui o sinal. Atenção: ficar fora do **desvio** não é o mesmo que ficar fora da **reponderação**. Todo meme de treino recebe `p_novo`, e nos unânimes ele é igual ao original por construção, já que todos os votos são iguais.

**O que fica igual nos três:**
- a validação usa sempre os soft labels humanos originais (o arquivo de validação é conferido por hash);
- só os labels de **treino** da Task 2.1 mudam. As Tasks 2.2 (intenção) e 2.3 (categorias) ficam intocadas;
- os memes unânimes mantêm o label 0 ou 1;
- o GLM nunca é o preditor final. Ele só fabrica os pesos dos anotadores.

---

## 3. Doc 25: voto duro do GLM, só nos memes em que ele está seguro (pré-registro original)

**Como o GLM vira referência.** Cada meme recebe o **voto da maioria** das três paráfrases. O GLM é considerado **confiante** no meme quando as três respostas existem e são iguais (padrão 3-0). Só os memes confiantes servem de referência. Os memes em que as paráfrases discordam (2-1) ficam fora do cálculo do desvio, embora continuem recebendo o soft label novo.

**Como se mede o desvio.** `d_ai = 1` se o voto do anotador difere do voto do GLM, e 0 se coincide. Entram no desvio os memes **não unânimes entre os humanos e com o GLM confiante**. Os empates humanos 3-3 entram. O peso de consenso `w` e a média `D_a` seguem a §2.

**Como D vira peso e o peso vira label.** `r_a = 1 − D_a`, e `p_novo` pela média ponderada da §2. Um anotador que nunca discorda do GLM nos memes em que os outros concordam fica com `r` próximo de 1. Um anotador que sempre discorda fica com `r` próximo de 0, e o voto dele praticamente some.

**Por que existiu.** Foi o desenho pedido pelo orientador: construir um "anotador perfeito" com um LLM, medir o desvio de cada anotador real em relação a ele e ponderar os votos por esse desvio. A novidade em relação à tentativa anterior da tese (ponderação por MACE, um modelo que estima a confiabilidade de cada anotador a partir dos **próprios votos**) é que aqui a referência vem de fora dos votos. Para evitar circularidade, nenhuma decisão sobre o GLM (prompt, modo de raciocínio etc.) olhou votos humanos.

**Por que o mínimo de 70 %.** O pré-registro exigia que o GLM estivesse confiante em pelo menos 70 % dos memes de treino **em cada idioma**. A conta foi a seguinte: cada anotador vota em cerca de 26 memes; tirando os unânimes, sobram uns 20. Com 70 % deles confiantes, ficam uns 14 memes por anotador para estimar `D`. Abaixo disso, os pesos ficariam ruidosos demais.

**Desfecho.** O método parou no portão dos 70 %. Em espanhol, as três paráfrases concordaram em 69,1 % dos memes (1 155 de 1 671; o mínimo era 1 170). Nenhum peso foi calculado e nada foi treinado. A paráfrase P3 era a dissidente em 75 % dos memes não confiantes (734 de 980).

---

## 4. Doc 27: o GLM como soft label (continuação exploratória do 25)

**Como o GLM vira referência.** Em vez de um voto duro, o GLM passa a ter seu próprio soft label: `p_GLM` = fração de YES nas três paráfrases, que só pode valer 0, ⅓, ⅔ ou 1. Um meme 2-1 deixa de ser descartado e passa a valer como uma referência "inclinada": `p_GLM` = ⅔ quando duas paráfrases dizem YES, ⅓ quando só uma diz.

**Como se mede o desvio.** `d_ai = |1[voto_a = YES] − p_GLM,i|`, em que `1[voto = YES]` vale 1 para um voto YES e 0 para um voto NO. Nos memes 3-0 do GLM, isso é exatamente o `d` do doc 25 (0 ou 1). Nos memes 2-1, o `d` vale ⅓ para quem vota com a maioria das paráfrases e ⅔ para quem vota contra ela. Entram no desvio os memes **não unânimes entre os humanos e com as três paráfrases respondidas**, sejam 3-0 ou 2-1. Ficam fora só os 5 memes cujas imagens o servidor recusou. O peso de consenso `w` e a média `D_a` não mudam.

Há um efeito colateral, registrado no próprio pré-registro: num meme 2-1, **todo** anotador leva pelo menos ⅓ de desvio, qualquer que seja o voto. Isso cria um piso em `D_a`, e por isso os valores de `D` do doc 27 **não estão na mesma escala** que os do doc 25 e não devem ser comparados diretamente.

**Como D vira peso e o peso vira label.** Exatamente como no doc 25: `r_a = 1 − D_a` e `p_novo = Σ r·[YES] / Σ r`.

**Por que existiu.** A falha do doc 25 por 15 memes levou a dois diagnósticos, feitos só com dados do GLM:
1. o corte de 70 % é binário e arbitrário: 69,1 % e 70 % dão praticamente a mesma precisão por anotador (~13,8 contra ~14,0 memes);
2. descartar os memes 2-1 provavelmente jogava fora justamente os memes ambíguos, onde o desvio de cada anotador seria mais informativo. Isso foi confirmado depois: nos memes em que as paráfrases discordam, 52,2 % têm divisão humana 4-2 ou 3-3, contra 43,6 % nos memes 3-0 (diferença de +0,086, IC 95 % de +0,049 a +0,123).

Uma variante com entropia (pesar cada meme por `1 − H` das paráfrases) foi descartada antes de rodar. Com três paráfrases, esse peso só vale 1 (3-0) ou 0,082 (2-1), o que quase reproduz a exclusão e ainda ignora a **direção** da hesitação do GLM. O doc 27 foi aprovado pelo orientador em 2026-10-09 como continuação **exploratória**, declarada como motivada pela falha do doc 25, e não substitui o resultado do 25.

**Portão de retreino.** Antes de gastar GPU, o protocolo exigia que os labels se movessem de verdade e que o GLM não reproduzisse apenas o consenso dos pares. As condições eram: pelo menos 5 % dos memes de treino mudando meio voto ou mais (`|Δp| ≥ 1/12`) e correlação de Spearman abaixo de 0,8 entre os pesos via GLM e os pesos obtidos trocando o GLM pela fração de YES dos outros 5 anotadores. Passou com 34,6 % e ρ = 0,50.

**Desfecho.** O retreino foi feito com 3 seeds pareadas. O ICM-Soft-Norm subiu +0,0074 em média (as 3 seeds no mesmo sentido, p = 0,107) e a `eval_loss` piorou nas 3. O critério pré-registrado não foi satisfeito: efeito não detectado com 3 seeds.

---

## 5. Doc 28: a mesma ponderação do 27 com peso r⁴

**Como o GLM vira referência.** Igual ao doc 27: `p_GLM` ∈ {0, ⅓, ⅔, 1}.

**Como se mede o desvio.** Igual ao doc 27. Os `D_a` e os `r_a` não foram recalculados: o doc 28 lê o `r_glm` gravado pelo doc 27 (`annotator_weights.csv`, com SHA-256 conferido).

**Como D vira peso e o peso vira label.** O peso do voto passa a ser `r_a⁴ = (1 − D_a)⁴`, e `p_novo = Σ r⁴·[YES] / Σ r⁴`. Elevar à quarta potência preserva a ordem dos anotadores, mas aumenta muito a distância entre eles. Um anotador com o dobro do `r` de outro passa a ter 2⁴ = 16 vezes o peso. Com a mediana real, `r` = 0,562 vira 0,562⁴ ≈ 0,10. Na prática, dentro de cada painel, o voto passa a ser decidido pelos um ou dois anotadores mais alinhados ao GLM.

**O que fica igual.** Memes incluídos, validação original, Tasks 2.2 e 2.3, memes unânimes. O BASE (modelo treinado com os labels originais) foi reaproveitado do doc 27, porque o treino se mostrou determinístico: três execuções da mesma seed deram `eval_loss` idêntica época a época.

**Por que existiu.** O nulo do doc 27 deixou aberta a possibilidade de o efeito existir e ser apenas pequeno demais para 3 seeds. O doc 28 testa uma **dose maior** da mesma intervenção, com o objetivo declarado de produzir um efeito maior **em qualquer direção**. O pré-registro não justifica a escolha do expoente 4 em particular, além de ser "uma dose maior", e diz explicitamente que ele não poderia ser trocado depois de ver a primeira seed. O desenho foi uma triagem com a seed 42 e limites fixados antes: o experimento só seguiria para as seeds 7 e 2025 se alguma métrica mudasse acima do dobro do Δ observado no doc 27.

**Desfecho.** A triagem passou e as 3 seeds rodaram. O efeito trocou de sinal: o ICM-Soft-Norm caiu −0,0247 em média (3 seeds no mesmo sentido, p = 0,209) e a `eval_loss` piorou nas 3 (+0,0988, p = 0,029).

---

## 6. Exemplo numérico: um painel passado pelos três métodos

**O painel.** O desvio de um anotador é uma média sobre vários memes, então o exemplo precisa de um painel pequeno, e não de um meme isolado. São 6 anotadores (A a F) e 5 memes. Os votos e as respostas do GLM são **o mesmo caso de teste do Apêndice A do `SPEC_27.md`**, e os valores de `D` e `r` do doc 27 abaixo coincidem com os desse apêndice, o que permite conferir as contas.

| meme | votam YES | divisão humana | GLM (P1, P2, P3) | referência no doc 25 | `p_GLM` no doc 27/28 |
|---|---|---|---|---|---|
| m1 | A B C D E F | 6-0 | NO, NO, NO | unânime: fora do desvio | 0 (unânime: fora) |
| m2 | A B C D E | 5-1 | YES, YES, YES | YES (confiante) | 1 |
| m3 | A B C D | 4-2 | NO, NO, YES | **2-1: fora do desvio** | ⅓ |
| m4 | A B C | 3-3 | YES, YES, NO | **2-1: fora do desvio** | ⅔ |
| m5 | B C D E F | 5-1 | NO, NO, NO | NO (confiante) | 0 |

O meme de destaque é o **m4**: empate humano 3-3, com as paráfrases do GLM divididas 2-1 para o YES. Os anotadores de destaque são A (o mais alinhado ao GLM), E (intermediário) e F (o que mais diverge).

**Nota para quem conferir com o `SPEC_25.md`.** O Apêndice A do doc 25 dá outros valores para este painel (por exemplo, `D_F` = 0,8246). Isso não é um erro: aquele teste supunha o GLM confiante nos 5 memes. Aqui, m3 e m4 têm paráfrases 2-1 e, pela regra real do doc 25, saem do desvio.

### 6.1 Desvio de A, E e F, termo a termo

Cada célula mostra `w · d` para o meme (o par `w`, `d`). "—" indica um meme fora do desvio.

| anotador | m2 | m3 | m4 | m5 | `D` |
|---|---|---|---|---|---|
| **doc 25** | | | | | |
| A (YES, YES, YES, NO) | w 0,278 · d 0 | — | — | w 1,000 · d 0 | 0 / 1,278 = **0,0000** |
| E (YES, NO, NO, YES) | w 0,278 · d 0 | — | — | w 0,278 · d 1 | 0,278 / 0,556 = **0,5000** |
| F (NO, NO, NO, YES) | w 1,000 · d 1 | — | — | w 0,278 · d 1 | 1,278 / 1,278 = **1,0000** |
| **doc 27 / 28** | | | | | |
| A | w 0,278 · d 0 | w 0,029 · d ⅔ | w 0,029 · d ⅓ | w 1,000 · d 0 | 0,0290 / 1,3362 = **0,0217** |
| E | w 0,278 · d 0 | w 0,278 · d ⅓ | w 0,029 · d ⅔ | w 0,278 · d 1 | 0,3901 / 0,8633 = **0,4519** |
| F | w 1,000 · d 1 | w 0,278 · d ⅓ | w 0,029 · d ⅔ | w 0,278 · d 1 | 1,3901 / 1,5852 = **0,8769** |

Como ler: em m2, F é o único NO contra 5 YES dos outros e contra o GLM (YES). Os outros 5 concordam entre si, então `w` = 1 e a punição é cheia. Em m5, A é o único NO, os outros 5 dizem YES e o GLM diz NO. A "acerta" com o GLM e não leva desvio, enquanto E e F levam `d` = 1, mas com `w` = 0,278, porque, do ponto de vista de cada um deles, os outros 5 votos já estão divididos 4-1. No doc 27, os memes 2-1 (m3 e m4) passam a contar, com `d` fracionário: todos levam pelo menos ⅓, o que puxa o `D` de A para cima (0 → 0,0217) e reduz um pouco o de E (0,5000 → 0,4519) e o de F (1,0000 → 0,8769).

### 6.2 Pesos dos seis anotadores

| anotador | `D` (doc 25) | `r` (doc 25) | `D` (doc 27/28) | `r` (doc 27) | `r⁴` (doc 28) |
|---|---|---|---|---|---|
| A | 0,0000 | 1,0000 | 0,0217 | 0,9783 | 0,9158 |
| B | 0,5000 | 0,5000 | 0,5000 | 0,5000 | 0,0625 |
| C | 0,5000 | 0,5000 | 0,5000 | 0,5000 | 0,0625 |
| D | 0,5000 | 0,5000 | 0,5158 | 0,4842 | 0,0550 |
| E | 0,5000 | 0,5000 | 0,4519 | 0,5481 | 0,0902 |
| F | 1,0000 | **0,0000** | 0,8769 | 0,1231 | **0,0002** |

No doc 25, F chega a `r` = 0, ou seja, o voto dele é simplesmente apagado. No doc 28, ninguém chega a zero, mas a quarta potência deixa A com 0,92 e todos os outros entre 0,0002 e 0,09: dentro deste painel, A passa a valer mais que os outros cinco juntos.

### 6.3 Novos soft labels

| meme | votos YES | `p` original | doc 25 | doc 27 | doc 28 |
|---|---|---|---|---|---|
| m1 | 6/6 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| m2 | 5/6 | 0,8333 | 1,0000 | 0,9607 | 0,9998 |
| m3 | 4/6 | 0,6667 | 0,8333 | 0,7858 | 0,9237 |
| **m4** | 3/6 | 0,5000 | **0,6667** | **0,6313** | **0,8774** |
| m5 | 5/6 | 0,8333 | 0,6667 | 0,6878 | **0,2280** |

**O meme m4, passo a passo** (votam YES: A, B e C; votam NO: D, E e F):
- **doc 25:** (1 + 0,5 + 0,5) / (1 + 0,5 + 0,5 + 0,5 + 0,5 + 0) = 2,0000 / 3,0000 = **0,6667**. O empate vira YES porque o NO de F não pesa nada. Note que o m4 está fora do desvio no doc 25 (GLM 2-1) e mesmo assim tem o label alterado, porque a reponderação vale para todo meme;
- **doc 27:** (0,9783 + 0,5 + 0,5) / 3,1336 = 1,9783 / 3,1336 = **0,6313**. Também vira YES, um pouco menos;
- **doc 28:** (0,9158 + 0,0625 + 0,0625) / 1,1863 = 1,0408 / 1,1863 = **0,8774**. Como A está do lado do YES e domina o painel, o empate vira um YES forte.

**O meme m5 mostra o mecanismo do r⁴ às claras.** Cinco anotadores dizem YES e só A diz NO, junto com o GLM. No doc 27, o label cai de 0,833 para 0,688. No doc 28, o peso de A (0,9158) supera a soma dos cinco outros (0,2704), e o label cai para **0,2280**: um meme que os humanos chamaram de sexista por 5 a 1 passa a ser treinado como não sexista. A direção do empurrão segue o GLM. Neste exemplo ele foi para o NO, porque o GLM disse NO em m5. Nos dados reais, como se verá na §8, o GLM diz YES na grande maioria dos memes, e o empurrão vai para o YES.

---

## 7. Comparação lado a lado

| | doc 25 | doc 27 | doc 28 |
|---|---|---|---|
| Referência do GLM | voto duro (maioria das 3 paráfrases); só vale se as 3 concordarem | soft label `p_GLM` ∈ {0, ⅓, ⅔, 1} | igual ao 27 |
| Memes usados no desvio | não unânimes **e** GLM 3-0 | não unânimes **e** 3 paráfrases respondidas (3-0 e 2-1) | igual ao 27 |
| Desvio por meme `d` | 0 ou 1 | `|1[voto = YES] − p_GLM|` ∈ {0, ⅓, ⅔, 1} | igual ao 27 |
| Peso de consenso `w` | `1 − H(outros 5)` | igual | igual |
| Peso do voto | `r = 1 − D` | `r = 1 − D` | `r⁴ = (1 − D)⁴` |
| Deslocamento médio dos labels de treino para o YES | não gerado | +0,054 | +0,120 |
| Memes de treino que mudam ≥ meio voto (`|Δp|` ≥ 1/12) | não gerado | 34,6 % | 62,5 % |
| Rótulos duros que mudam (maioria YES/NO/empate) | não gerado | 554 (nenhum de YES para NO) | 849 (19 de YES para NO) |
| Treinou? | **não** (parou no portão dos 70 %) | sim, 3 seeds | sim: triagem com 1 seed, depois 3 seeds |

Os deslocamentos foram conferidos nos arquivos: +0,0540 e 34,6 % recalculados de `soft_labels_taskA_glmw.json`, e +0,1200 e 62,5 % lidos do `config.json` (registro do doc 28). Lembrete da §4: os valores de `D` dos docs 25 e 27 não são comparáveis entre si, por causa do piso de ⅓ nos memes 2-1.

---

## 8. O fio que liga os três

O achado mais importante não está no retreino. Está na checagem descritiva que o doc 27 fez **antes** dele (a "checagem (b)" do protocolo, que pergunta se o peso só mede o limiar do anotador).

**O peso acabou medindo sobretudo o limiar de YES de cada anotador, e não a "qualidade" dele.** A correlação de Spearman entre o desvio `D` e a taxa de YES do anotador (a fração dos memes dele em que votou YES) é ρ = −0,676: quem diz YES com mais facilidade tem desvio menor e, portanto, peso maior. A correlação entre `D` e a distância entre o limiar do anotador e o do GLM é ainda maior, ρ = 0,710.

**Por que isso acontece.** O GLM diz YES muito mais do que os humanos. Nos 3 381 memes de treino com referência, a média de `p_GLM` é 0,804 (por paráfrase: P1 0,861, P2 0,913, P3 0,636). A média do soft label humano nesses memes é 0,555. Num meme não unânime típico, então, o GLM diz YES, e quem votou YES leva `d` = 0, enquanto quem votou NO leva `d` = 1. Um anotador com limiar baixo, que chama mais memes de sexistas, acumula pouco desvio quase automaticamente, sem que isso diga algo sobre o cuidado com que anotou. Isso é coerente com um resultado anterior da tese, citado no doc 25: o desvio de cada anotador no EXIST se comporta como um **limiar pessoal estável** (correlação de 0,850 entre duas metades dos memes do mesmo anotador), e não como descuido.

**Consequência: ponderar ≈ empurrar os labels para o YES.** Se o peso favorece quem diz YES, a média ponderada dos votos sobe. É o que se vê nos dados do doc 27: das 554 mudanças de rótulo duro, 409 são empates que viram YES, 109 empates que viram NO, 36 NO que viram YES e **nenhuma** YES que vira NO. A distribuição demográfica dos pesos vai na mesma linha: mulheres e faixas etárias mais altas recebem peso maior (por exemplo, 0,596 para mulheres com 46 anos ou mais contra 0,461 para homens de 18 a 22). A hipótese é que esses grupos simplesmente dizem mais YES.

**Mais dose, mais empurrão, pior resultado.** O doc 28 não muda o que o peso mede, só a força com que ele age. O deslocamento médio para o YES passa de +0,054 para +0,120, e o modelo, treinado em labels mais distantes dos labels humanos da validação, piora. O padrão do doc 27 (ICM-Soft-Norm um pouco acima e `eval_loss` um pouco pior, sempre no mesmo sentido) é compatível com uma mudança de calibração. O do doc 28 mostra que, ao aumentar essa mudança, o custo aparece.

**Ressalva honesta.** "O efeito é só o deslocamento para o YES" é uma inferência, não um resultado medido. O doc 27 definiu, antes do retreino, um braço de controle (`glmshift`) que aplicaria o mesmo deslocamento médio a todos os memes não unânimes, sem ponderar anotador nenhum. A regra fixada foi rodá-lo só se a ponderação desse um ganho significativo. Como não deu, ele não rodou. Parte da piora da `eval_loss` no doc 28 também é quase mecânica: o modelo treina para uma distribuição deslocada e é avaliado contra a original.

---

## 9. Próximo passo proposto: doc 29 (**proposta para discussão, ainda NÃO pré-registrada**)

*Esta seção descreve uma ideia a discutir com o orientador. Nada aqui foi fixado nem treinado. Os números são de uma análise exploratória offline nos labels de treino, sem nenhum modelo treinado.*

**A ideia.** Os docs 25 a 28 usaram o GLM de forma **indireta**: o GLM julga os anotadores, e os pesos dos anotadores mudam o label. O §8 mostra que esse caminho mede o limiar do anotador. A proposta é usar o GLM de forma **direta**, como mais uma fonte de evidência sobre o meme, depois de corrigir o viés dele para o YES:

`p_novo = 0,5 · p_humano + 0,5 · p_GLM_corrigido`

Não há mais peso por anotador.

**A correção por quantis.** O `p_GLM` bruto tem a escala errada (média 0,804 contra 0,555 dos humanos), mas pode ter a **ordem** certa: um meme em que as três paráfrases dizem YES provavelmente é mais sexista, aos olhos humanos, do que um em que só duas dizem. A correção mantém a ordem do GLM e impõe a escala humana. Ordenam-se os memes de treino pelo `p_GLM`, atribuem-se a eles os soft labels humanos também ordenados, e cada nível do GLM recebe a média dos valores humanos que caíram na sua faixa. O mapa resultante é:

| `p_GLM` bruto | 0 | ⅓ | ⅔ | 1 |
|---|---|---|---|---|
| `p_GLM` corrigido | 0 | 0,124 | 0,303 | 0,756 |

Por construção, o deslocamento médio é zero. O mapa foi reproduzido nesta sessão a partir de `glm_votes.csv` e dos labels de treino: dá exatamente esses quatro valores, e a média do GLM corrigido (0,5549) é igual à média humana nos mesmos memes. Note que um 2-1 para o YES (⅔), que no doc 27 empurrava para o YES, vira 0,303 depois da correção, abaixo da média humana: para o padrão do GLM, "só duas paráfrases dizem YES" é um sinal fraco.

**O braço de controle `smooth50`.** Misturar qualquer label com uma constante é uma forma de *label smoothing* (suavização de labels), que por si só pode ajudar, porque um soft label de 6 votos é uma estimativa ruidosa. Para separar "o GLM traz informação" de "suavizar ajuda", o controle usa a mesma mistura com a constante 0,555 (a média humana) no lugar do GLM: `p_novo = 0,5 · p_humano + 0,5 · 0,555`. O GLM só merece crédito se superar esse controle.

**No exemplo da §6.** Para o m4 (`p_humano` = 0,5 e `p_GLM` = ⅔, que vira 0,303), a proposta daria 0,5·0,5 + 0,5·0,303 = **0,4015**, e o `smooth50` daria 0,5275. Para o m5 (`p_humano` = 0,833 e GLM 0), daria 0,4167. Para o m1, que é **unânime** (6 YES) mas com o GLM em 0, daria **0,5000**. Isso expõe um ponto em aberto: diferente dos docs 25 a 28, a mistura direta, tal como formulada, **altera os memes unânimes**. Se eles devem ser preservados é uma decisão a tomar no pré-registro.

**Teste offline *split-half*.** Para estimar, sem GPU, se o GLM tem informação sobre o que os humanos pensam, cada meme de treino é dividido em duas metades de 3 anotadores. A fração de YES de uma metade, combinada ou não com o GLM, tenta prever a fração de YES da outra metade, que funciona como uma amostra independente da mesma população de anotadores. Medem-se o erro quadrático médio (MSE) e a entropia cruzada, e para os dois, quanto menor, melhor:

| previsor da outra metade | MSE | entropia cruzada |
|---|---|---|
| só humano (3 anotadores) | 0,1193 | 1,108 |
| suavização cega (`smooth50`) | 0,0916 | **0,611** |
| mistura com GLM sem correção | 0,1062 | 0,856 |
| mistura com GLM corrigido por quantis | **0,0885** | 0,624 |
| só GLM (corrigido) | 0,1154 | 0,724 |

*(Protocolo: as 20 divisões 3 contra 3 de cada meme, nos dois sentidos; o mapa de quantis e a constante são ajustados uma vez sobre o soft label de 6 votos do treino; a entropia cruzada corta p em [0,001; 0,999]. Script `splithalf_29.py`, saída em `glm_reference/reports/splithalf_29.md`.)*

**Leitura.** O GLM tem informação: corrigido, ele é o melhor previsor em MSE e supera bastante o GLM sem correção. Mas o ganho além da simples suavização é pequeno. Em MSE, o GLM corrigido vence o `smooth50` por 0,0031 (IC 95% [0,0019; 0,0043]). Em entropia cruzada, perde (0,624 contra 0,611; diferença +0,0130, IC 95% [+0,0077; +0,0182]). Boa parte do benefício da mistura viria, portanto, da suavização em si, e não do GLM. `[Hipótese]` No treino, os soft labels têm 6 votos, e não 3, de modo que o espaço para ganho com qualquer suavização deve ser ainda menor do que no teste *split-half*.

**Pontos a fixar no pré-registro, se o doc 29 for adiante:**
1. **Protocolo do *split-half*:** já fixado e salvo (`splithalf_29.py`; protocolo na nota abaixo da tabela). Uma primeira versão exploratória usava só 10 das 20 divisões. Os números mudaram na terceira casa, e as conclusões são as mesmas.
2. **Memes unânimes:** o rascunho do SPEC_29 propõe **misturar** também os unânimes, nos dois braços (`glmq50` e `smooth50`), porque a pergunta passa a ser "o GLM como fonte de label". Consequência: um meme 6/6 YES em que o GLM diz 0 de 3 cai para 0,5. A alternativa é preservar os unânimes, como nos docs 25 a 28. Decisão a tomar com o orientador.
3. **Onde ajustar o mapa de quantis:** só no treino, nunca na validação, para não vazar informação.
4. **Peso da mistura** (0,5) fixado antes, sem busca.
5. **Critério de sucesso:** o braço com GLM tem de superar o `smooth50`, e não só o BASE.

---

## Fontes

`glm_annotator/SPEC_25.md` (§0, §3, §6, §8, §20 e Apêndice A), `SPEC_27.md` (§0, §3, §8, §9, §16A, §16C, Apêndice A e resultado de 2026-10-11), `SPEC_28.md` (§1, §3 e resultado de 2026-10-12), `NOTAS_PARA_TESE.md` (§1–§4, §7A e §7B), `glm_reference/reports/checks_summary.md` e `glm_reference/config.json`. Código: `weights.py` (doc 27), `_archive_doc25/weights.py` (doc 25), `doc28.py` e `prompts.py`. O exemplo numérico da §6 e o mapa de quantis da §9 foram calculados nesta sessão com essas funções e esses arquivos. Os números do *split-half* vêm da análise exploratória descrita na §9.
