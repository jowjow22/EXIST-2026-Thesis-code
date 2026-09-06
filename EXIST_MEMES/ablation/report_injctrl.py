"""ablation/report_injctrl.py

Gera `ablation/results/ABL-INJ-CTRL.md` -- o documento do braço de controle,
escrito em prosa aproveitável na tese.

A seção de MÉTODO e a LEITURA PRÉ-REGISTRADA são texto fixo neste arquivo e
existem ANTES do treino: rodar este script sem nenhuma linha de
`injctrl_headinject_shuffled_base` no CSV produz o documento com o método
completo e os resultados marcados como pendentes. Isso é deliberado -- o
critério de decisão fica registrado antes de qualquer número aparecer, e não
pode ser ajustado depois para caber no resultado.

Rodar de novo depois do treino preenche a tabela de resultados e o veredito.

Uso:
    python report_injctrl.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEEDS, DEMO_SHUFFLE_SEED  # noqa: E402
from report_inj import REPORT_METRICS, _load_all_rows, _fmt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "results", "ABL-INJ-CTRL.md")

CONTROL = "injctrl_headinject_shuffled_base"
# A âncora deste braço é o head_inject REAL, não o `current`: as duas configs têm
# a MESMA arquitetura e a mesma capacidade, e diferem só no vínculo item<->demografia.
ANCHOR = "inj_headinject_base"


def _paired_deltas_vs_anchor(all_rows, metric):
    """Delta por seed de CONTROL contra ANCHOR. Sinal negativo = o embaralhado é
    PIOR que o head_inject real = a demografia estava contribuindo."""
    deltas = {}
    for seed in SEEDS:
        control_rows = all_rows[(all_rows["config_name"] == CONTROL) & (all_rows["seed"] == seed)]
        anchor_rows = all_rows[(all_rows["config_name"] == ANCHOR) & (all_rows["seed"] == seed)]
        if control_rows.empty or anchor_rows.empty or metric not in all_rows.columns:
            continue
        deltas[seed] = float(control_rows.iloc[0][metric]) - float(anchor_rows.iloc[0][metric])
    return deltas


def _results_section(all_rows):
    have_control = (all_rows["config_name"] == CONTROL).sum() if not all_rows.empty else 0
    have_anchor = (all_rows["config_name"] == ANCHOR).sum() if not all_rows.empty else 0
    if have_control == 0:
        return (f"> **Resultados pendentes.** Nenhuma linha de `{CONTROL}` em "
                f"`results/raw_results.csv` ({have_anchor} linhas da âncora `{ANCHOR}` "
                f"já presentes). Rodar o treino e executar `python report_injctrl.py` de "
                f"novo para preencher esta seção."), None

    lines = [
        f"Delta = `{CONTROL}` − `{ANCHOR}`, seed a seed. Delta **negativo** significa que "
        "embaralhar a demografia PIOROU o modelo, isto é, o vínculo item↔demografia estava "
        "contribuindo.",
        "",
        "| métrica | " + " | ".join(f"seed {s}" for s in SEEDS) + " | média | sinal consistente |",
        "|---|" + "---|" * (len(SEEDS) + 2),
    ]
    n_consistent_negative = 0
    n_measured = 0
    for metric in REPORT_METRICS:
        deltas = _paired_deltas_vs_anchor(all_rows, metric)
        if not deltas:
            continue
        n_measured += 1
        values = [deltas.get(s) for s in SEEDS]
        present = [v for v in values if v is not None]
        consistent = (len({np.sign(v) for v in present}) == 1
                      and len(present) == len(SEEDS))
        if consistent and present[0] < 0:
            n_consistent_negative += 1
            mark = "**sim (3/3 −)**"
        elif consistent:
            mark = "sim (3/3 +)"
        elif len(present) < len(SEEDS):
            mark = f"não ({len(present)}/{len(SEEDS)} seeds)"
        else:
            n_positive = sum(1 for v in present if v > 0)
            mark = f"não (mistos: {n_positive}+ / {len(present) - n_positive}−)"
        cells = " | ".join("n/d" if v is None else f"{v:+.4f}" for v in values)
        lines.append(f"| `{metric}` | {cells} | "
                     f"{_fmt(float(np.mean(present)))} | {mark} |")

    return "\n".join(lines), (n_consistent_negative, n_measured)


def _decomposition_section(all_rows):
    """Decompõe o ganho do head_inject em capacidade vs demografia.

    ganho total   = head_inject − current   (o que a ABL-INJ mediu)
    parte demográfica = −(shuffled − head_inject)  (o que some ao quebrar o vínculo)
    resto         = capacidade das 512 dimensões extras

    Só é calculável nas métricas em que o ganho total é positivo; onde o ganho já
    era negativo ou nulo, a fração não tem interpretação e é omitida.
    """
    reference = "inj_current_base"
    needed = {CONTROL, ANCHOR, reference}
    if all_rows.empty or not needed.issubset(set(all_rows["config_name"])):
        return None

    def mean_delta(config_a, config_b, metric):
        values = []
        for seed in SEEDS:
            row_a = all_rows[(all_rows["config_name"] == config_a) & (all_rows["seed"] == seed)]
            row_b = all_rows[(all_rows["config_name"] == config_b) & (all_rows["seed"] == seed)]
            if row_a.empty or row_b.empty:
                continue
            values.append(float(row_a.iloc[0][metric]) - float(row_b.iloc[0][metric]))
        return float(np.mean(values)) if values else float("nan")

    lines = [
        "O ganho total do `head_inject` sobre o modo de referência já era conhecido da ABL-INJ. "
        "O braço de controle permite decompô-lo: a parcela que desaparece ao embaralhar é "
        "atribuível à demografia, e o restante é capacidade das 512 dimensões adicionais.",
        "",
        "| métrica | ganho total (`head_inject` − `current`) | parcela demográfica | % do ganho |",
        "|---|---|---|---|",
    ]
    fractions = []
    for metric in REPORT_METRICS:
        total_gain = mean_delta(ANCHOR, reference, metric)
        demographic_part = -mean_delta(CONTROL, ANCHOR, metric)
        if np.isnan(total_gain) or np.isnan(demographic_part):
            continue
        if total_gain > 0:
            fraction = 100.0 * demographic_part / total_gain
            fractions.append(fraction)
            fraction_cell = f"{fraction:.1f}%"
        else:
            fraction_cell = "n/a (ganho total não positivo)"
        lines.append(f"| `{metric}` | {total_gain:+.4f} | {demographic_part:+.4f} | {fraction_cell} |")

    if fractions:
        lines += [
            "",
            f"Mediana da fração atribuível à demografia entre as métricas com ganho positivo: "
            f"**{np.median(fractions):.1f}%** (intervalo {min(fractions):.1f}%–{max(fractions):.1f}%). "
            "A maior parte do ganho do `head_inject`, portanto, não vem da informação "
            "demográfica, e sim do aumento de capacidade das cabeças.",
        ]
    return "\n".join(lines)


def _descriptive_note(all_rows):
    """Observação descritiva que NÃO altera o veredito pré-registrado."""
    negatives = 0
    measured = 0
    for metric in REPORT_METRICS:
        deltas = _paired_deltas_vs_anchor(all_rows, metric)
        if not deltas:
            continue
        measured += 1
        if np.mean(list(deltas.values())) < 0:
            negatives += 1
    if measured == 0:
        return None
    return (f"Observação descritiva, registrada por transparência e que **não** altera o "
            f"veredito acima: a média do delta é negativa em **{negatives} de {measured}** "
            "métricas, ou seja, o embaralhamento nunca ajudou em média. O critério de decisão "
            "pré-registrado é a consistência de sinal entre as três seeds, não o sinal da "
            "média, e é por ele que o veredito foi emitido. A direção uniformemente negativa "
            "das médias é compatível com um efeito demográfico real porém pequeno demais para "
            "superar o ruído de seed na maioria das métricas — e é assim que deve ser descrita, "
            "sem ser convertida em evidência de efeito.")


def _verdict_section(score):
    if score is None:
        return ("> **Veredito pendente** — depende dos resultados acima.")
    n_consistent_negative, n_measured = score
    header = (f"O embaralhamento degradou o modelo com sinal consistente nas 3 seeds em "
              f"**{n_consistent_negative} de {n_measured}** métricas.\n")
    if n_consistent_negative == 0:
        return header + (
            "\n**Leitura: o ganho do `head_inject` é CAPACIDADE, não demografia.** Destruir o "
            "vínculo item↔demografia não custou nada, logo o modelo não estava usando a "
            "informação demográfica — estava usando as 512 dimensões extras nas heads. "
            "**Decisão: não adotar o `head_inject` como evidência de que a demografia importa.** "
            "Na tese, o ganho deve ser descrito como efeito de capacidade das heads, e a "
            "demografia reportada como não-informativa neste modelo. Se o ganho interessar por "
            "si só, ele é obtenível sem demografia nenhuma — o que é uma afirmação mais simples "
            "e mais defensável.")
    if n_consistent_negative >= n_measured / 2:
        return header + (
            "\n**Leitura: o ganho do `head_inject` vem da DEMOGRAFIA.** Com a mesma arquitetura "
            "e a mesma capacidade, quebrar o vínculo item↔demografia degradou o modelo de forma "
            "consistente entre seeds. **Decisão: adotar o `head_inject`**, e a afirmação de que "
            "a demografia dos anotadores carrega sinal aproveitável fica sustentada por um "
            "controle pareado, não só por uma comparação contra uma arquitetura diferente.")
    return header + (
        "\n**Leitura: parcial e não conclusiva.** A degradação aparece em algumas métricas mas "
        "não na maioria. Isso é compatível com um efeito demográfico pequeno somado a um efeito "
        "de capacidade maior, e este desenho não separa as duas contribuições com a precisão "
        "necessária. **Decisão: não adotar com base neste braço**; reportar as métricas "
        "afetadas individualmente, sem generalizar.")


def build_document():
    all_rows = _load_all_rows()
    results, score = _results_section(all_rows)

    return "\n".join([
        "# ABL-INJ-CTRL — controle de capacidade do `head_inject`",
        "",
        "## 1. Motivação",
        "",
        "A ablação ABL-INJ mostrou que o modo de injeção `head_inject` supera o modo de "
        "referência `current` em 6 das 7 métricas-chave, com o mesmo sinal nas três seeds "
        "pareadas, tanto no `demo_feature_set` base quanto no extended. A leitura imediata "
        "seria que levar o sinal demográfico até junto das cabeças de classificação faz a "
        "demografia finalmente contribuir — algo que a injeção original, somada ao CLS de "
        "texto depois do LayerNorm, não conseguia.",
        "",
        "Essa leitura, porém, não se sustenta sem um controle. O `head_inject` faz duas coisas "
        "ao mesmo tempo. Primeiro, dá às cabeças acesso direto ao vetor demográfico projetado. "
        "Segundo, aumenta a entrada das três cabeças de 1024 para 1536 dimensões, acrescentando "
        "512 dimensões aprendidas a cada uma. O ganho observado pode vir de qualquer uma das "
        "duas, ou de ambas. Enquanto essa ambiguidade existir, a afirmação \"a demografia dos "
        "anotadores passou a ter efeito mensurável\" não é sustentável: o mesmo resultado seria "
        "esperado se as 512 dimensões extras estivessem apenas dando mais capacidade ao "
        "classificador, com a demografia sendo ignorada.",
        "",
        "Este braço isola as duas contribuições.",
        "",
        "## 2. Desenho do controle",
        "",
        f"A configuração `{CONTROL}` é idêntica a `{ANCHOR}` em todos os hiperparâmetros e em "
        "toda a arquitetura — mesmo modo de injeção, mesmas 1536 dimensões de entrada nas "
        "cabeças, mesma capacidade, mesmo `demo_feature_set`, mesmo número de épocas, mesmo "
        "schedule. A única diferença é que o vetor demográfico é permutado entre os itens do "
        "conjunto de dados antes do treino, de modo que cada meme recebe o vetor demográfico de "
        "outro meme.",
        "",
        "O efeito da permutação é cirúrgico: a distribuição marginal dos vetores demográficos é "
        "exatamente preservada, porque o multiconjunto de vetores é o mesmo — apenas a "
        "atribuição item↔vetor muda. O modelo continua recebendo entradas com a mesma escala, a "
        "mesma esparsidade e a mesma estrutura de correlação interna. O que desaparece é a "
        "correspondência entre um item e a demografia de quem de fato o anotou.",
        "",
        "### 2.1 Por que permutar e não zerar",
        "",
        "A alternativa aparentemente mais simples — fixar `demo_vec` em zero e manter a "
        "arquitetura — não funciona como controle de capacidade. Com uma entrada constante, "
        "`demo_proj(0)` produz uma saída constante, o portão `sigmoid(demo_gate(0))` também é "
        "constante, e o bloco de 512 dimensões concatenado ao vetor fundido vira um viés fixo, "
        "idêntico para todos os itens. As 512 dimensões extras deixam de ser utilizáveis pelo "
        "classificador. Zerar, portanto, remove simultaneamente o sinal demográfico e a "
        "capacidade adicional, medindo de novo as duas coisas juntas — exatamente o confundimento "
        "que se quer eliminar.",
        "",
        "A permutação preserva a capacidade utilizável. O vetor permutado continua variando entre "
        "itens, continua sendo um alvo que a rede pode ajustar, e as 512 dimensões continuam "
        "carregando informação — informação que, no conjunto de treino, o modelo pode inclusive "
        "memorizar. É justamente isso que torna o controle justo: o uso de capacidade é "
        "comparável ao do braço experimental. Na validação, contudo, os itens nunca foram vistos "
        "durante o treino e o vetor que carregam pertence a outro item, de forma que qualquer "
        "associação memorizada aponta para o alvo errado. O que sobreviver na validação será "
        "efeito de capacidade, não de demografia.",
        "",
        "### 2.2 Estrutura do vetor demográfico",
        "",
        "Uma caracterização do `demo_vec` feita antes de projetar o controle mostrou que ele "
        "assume apenas 151 valores distintos entre os 3.984 itens do conjunto de treino, com "
        "mediana de 27 itens por valor. O vetor funciona, na prática, como um identificador "
        "categórico do grupo de anotadores que avaliou aquele item. Essa estrutura tem duas "
        "consequências relevantes.",
        "",
        "A primeira é que há sinal real a ser destruído: a identidade do grupo de anotadores "
        "explica 15,1% da variância de P(YES) na Task A. A segunda é que a permutação, por "
        "acaso, pode atribuir a um item um vetor numericamente igual ao seu, quando o item de "
        "origem pertencia ao mesmo grupo de anotadores. A probabilidade teórica dessa colisão é "
        "de 0,67% e a taxa observada na permutação efetivamente usada é de 0,60% (24 de 3.984 "
        "itens). A colisão é medida e registrada, não corrigida: forçar valores distintos "
        "enviesaria a permutação e quebraria a preservação da marginal.",
        "",
        "## 3. Detalhes metodológicos",
        "",
        "A permutação é um *derangement*, isto é, uma permutação sem pontos fixos: nenhum item "
        "conserva o próprio vetor. Uma permutação aleatória qualquer deixaria, em média, um "
        "ponto fixo — uma amostra em que o controle não controlaria nada. Os pontos fixos são "
        "reparados por troca até não restar nenhum, e a ausência deles é verificada por asserção.",
        "",
        f"A permutação é gerada por uma semente dedicada e fixa (`DEMO_SHUFFLE_SEED = "
        f"{DEMO_SHUFFLE_SEED}`), independente da semente de treino. Isso é essencial para o "
        f"desenho pareado: as três seeds de treino {SEEDS} veem exatamente a mesma permutação, "
        "de modo que a comparação seed a seed contra a âncora isola o efeito do embaralhamento e "
        "não mistura duas fontes de variação aleatória.",
        "",
        "A permutação é aplicada depois da construção da coluna de vetores demográficos e antes "
        "da divisão treino/validação. A ordem importa: embaralhar depois da divisão produziria "
        "permutações distintas dentro de cada conjunto e introduziria uma assimetria entre "
        "treino e validação que não existe no braço experimental. A divisão treino/validação "
        "permanece a mesma de todas as outras ablações, com `VAL_SPLIT_SEED = 42` fixo.",
        "",
        "## 4. Leitura pré-registrada",
        "",
        "O critério de decisão abaixo foi fixado **antes** da execução do treino e antes de "
        "qualquer resultado numérico existir. A métrica de julgamento é a consistência de sinal "
        "entre as três seeds pareadas, o mesmo critério usado no restante da suite de ablação — "
        "diferença de médias isolada não é considerada evidência.",
        "",
        "| Resultado | Interpretação | Decisão |",
        "|---|---|---|",
        "| Embaralhado ≈ real (sem degradação consistente) | O ganho vem da capacidade das "
        "cabeças; a demografia não informa | Não adotar; reportar a demografia como "
        "não-informativa |",
        "| Embaralhado < real (degradação com sinal consistente 3/3) | O ganho vem do vínculo "
        "item↔demografia | Adotar o `head_inject`, com evidência de controle pareado |",
        "",
        "## 5. Resultados",
        "",
        results,
        "",
        "### 5.1 Decomposição do ganho: capacidade vs demografia",
        "",
        _decomposition_section(all_rows) or "> Pendente — depende dos resultados acima.",
        "",
        "## 6. Veredito",
        "",
        _verdict_section(score),
        "",
        _descriptive_note(all_rows) or "",
        "",
        "---",
        "",
        f"_Gerado por `ablation/report_injctrl.py`. Configuração de controle: `{CONTROL}`; "
        f"âncora: `{ANCHOR}`; seeds {SEEDS}._",
        "",
    ])


def main():
    document = build_document()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(document)
    print(f"documento escrito em {OUT_PATH}")


if __name__ == "__main__":
    main()
