# θ legado vs θ corrigido — mapeamento e impacto

Gerado por `reliability/compare_theta_legacy_vs_fixed.py`. Lê ficheiros já
escritos; não ajusta nem regenera nada.

## 1. Mapeamento de ficheiros

Os ficheiros legados **não foram tocados**. Os novos têm sufixo `_fixed`.

| legado (fases anteriores) | corrigido (oficial) |
|---|---|
| `theta_taskA_demo_base_K4.json` | `theta_taskA_demo_base_K4_fixed.json` |
| `theta_taskA_demo_extended_K4.json` | `theta_taskA_demo_extended_K4_fixed.json` |
| `theta_taskA_behavioral_K4.json` | `theta_taskA_behavioral_K4_fixed.json` |
| `theta_taskA_demo_base_K8.json` | `theta_taskA_demo_base_K8_fixed.json` |
| `theta_taskA_demo_extended_K8.json` | `theta_taskA_demo_extended_K8_fixed.json` |
| `theta_taskA_behavioral_K8.json` | `theta_taskA_behavioral_K8_fixed.json` |
| `theta_taskA_single_truth_K1.json` | `theta_taskA_single_truth_K1_fixed.json` |
| `summary_taskA.md` | `summary_taskA_fixed.md` |
| `soft_labels_taskA_weighted.json` | `soft_labels_taskA_weighted_fixed.json` |
| `soft_labels_taskA_raw.json` | `soft_labels_taskA_raw_fixed.json` (idêntico byte a byte — o raw não depende de θ) |
| `split_taskA_frozen.json` | `split_taskA_frozen_fixed.json` (mesmas listas de IDs — o split deriva do raw) |
| `soft_labels_taskA_report.md` | `soft_labels_taskA_report_fixed.md` |

O estimador legado continua acessível por flag:
`NutmegEM(responsibility="legacy")`, ou `python -m reliability.run_exist --responsibility legacy`.

## 2. θ: legado vs corrigido, nas sete configurações `[MEDIDO]`

| configuração | Pearson r | média\|Δθ\| | máx\|Δθ\| | cruzam 0,5 | \|Δ\|>0,05 |
|---|---|---|---|---|---|
| `demo_base_K4` | 0.5908 | 0.2263 | 0.7898 | 258 | 626 (70.6%) |
| `demo_extended_K4` | 0.5703 | 0.2880 | 0.7891 | 343 | 739 (83.3%) |
| `behavioral_K4` | 0.6103 | 0.2073 | 0.7898 | 259 | 600 (67.6%) |
| `demo_base_K8` | 0.4902 | 0.3339 | 0.7898 | 434 | 739 (83.3%) |
| `demo_extended_K8` | 0.4314 | 0.4419 | 0.8388 | 619 | 809 (91.2%) |
| `behavioral_K8` | 0.4651 | 0.2858 | 0.8388 | 380 | 706 (79.6%) |
| `single_truth_K1` | 0.8923 | 0.0794 | 0.7166 | 95 | 409 (46.1%) |

## 3. `single_truth_K1` em detalhe — a configuração que alimenta o A1

| grandeza | valor |
|---|---|
| anotadores comparados | 887 |
| Pearson r | **0.8923** |
| Spearman ρ | 0.8977 |
| média \|Δθ\| | **0.0794** |
| máx \|Δθ\| | **0.7166** |
| anotadores que cruzam θ = 0,5 | **95** |
| anotadores com \|Δθ\| > 0,05 | **409** (46.1%) |
| θ médio legado → corrigido | 0.5352 → 0.4826 |
| θ_legado ≥ θ_corrigido | 74.6% dos anotadores |

`[MEDIDO]` A direcção bate com a derivação: a responsabilidade legada infla
`r` (conta o voto do próprio anotador duas vezes), logo θ legado é o maior
na esmagadora maioria dos anotadores.

| | legado | corrigido |
|---|---|---|
| iterações | 200 | 292 |
| convergiu | False | True |
| quebras da log-posterior | 273 (medidas à parte) | 0 |

`[MEDIDO]` O `converged=false` original vinha do limite de 200 iterações. Com
`n_iter=5000` (novo default) o legado converge em 462 e o corrigido em 292.

## 4. Diagnóstico de poder das soft labels `[MEDIDO]`

Todas as colunas são `weighted − raw` sobre p_YES, no mesmo formato do
relatório original. A última coluna é a comparação directa entre os dois
alvos ponderados — quanto o A1 se move ao trocar de estimador.

| grandeza | legado (ref.) | corrigido | corrigido − legado |
|---|---|---|---|
| média \|Δ\| | 0.0720 | 0.0617 | -0.0103 |
| sd \|Δ\| | 0.0742 | 0.0656 | -0.0087 |
| p50 | 0.0555 | 0.0445 | -0.0110 |
| p90 | 0.1781 | 0.1511 | -0.0270 |
| p99 | 0.2934 | 0.2638 | -0.0296 |
| máx \|Δ\| | 0.4589 | 0.3525 | -0.1064 |
| itens \|Δ\|>0,05 | 2078 | 1861 | -217 |
| itens \|Δ\|>0,10 | 1187 | 934 | -253 |
| Pearson r (ponderado vs cru) | 0.9572 | 0.9661 | +0.0090 |
| itens que cruzam 0,5 | 443 | 421 | -22 |

| grandeza | valor |
|---|---|
| média \|p_YES corrigido − p_YES legado\| | **0.0246** |
| p90 | 0.0727 |
| máx | 0.4365 |
| Pearson r entre os dois alvos ponderados | 0.9893 |
| itens que cruzam 0,5 entre os dois alvos | **134** |

## 5. O que muda face ao diagnóstico anterior

- **Magnitude da intervenção desce**: média \|weighted − raw\| passa de
  **0.0720** para **0.0617**
  (-0.0103, -14.3%). O alvo que os
  retreinos têm de detectar ficou **menor**.
- **Itens que cruzam 0,5**: **443** → **421** (-22).
- **Correlação com o alvo cru sobe**: 0.9572 → 0.9661 — o alvo corrigido está **mais perto** do
  baseline, não mais longe.

`[LIMITAÇÃO]` Menor efeito no alvo significa **menos poder** para o mesmo
número de sementes. A tabela de dimensionamento do plano foi calculada sobre
o efeito antigo; com um alvo mais próximo do baseline, um nulo a 15 épocas
passa a ser ainda mais provável, e a análise de poder tem de ser refeita a
partir do d observado nesta configuração — não reaproveitada.

`[MEDIDO]` 3984 itens · 887 anotadores.
