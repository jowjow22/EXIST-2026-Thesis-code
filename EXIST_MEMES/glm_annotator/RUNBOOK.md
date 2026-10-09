# Runbook — Doc 25 (anotador GLM): passo a passo de execução

Spec: [`SPEC_25.md`](SPEC_25.md); o §18 lista as decisões de encaixe tomadas em 2026-10-06.
Todos os comandos rodam **da raiz do repositório** (`~/Documents/projects/VM-THESIS`), no Mac.
F0 a F4 rodam no Mac; F5 roda na VM.

Regras que valem em todos os passos:
- **Portão falhou: pare.** Não ajuste prompt, parâmetro nem limiar para rodar de novo (§11, §15).
- **Caiu no meio (rede, servidor):** rode **o mesmo comando** de novo. A retomada refaz só as chamadas que faltam, e chamada gravada nunca é refeita.
- O estado de todos os portões fica em `EXIST_MEMES/glm_reference/config.json` → `runtime.gates`.
- **Log:** cada request imprime uma linha `[braço P1 s42] ✓ 12/720 · meme 110887 · <resposta do modelo>`, ou `✗ ... · <erro>`. A cada 100 chamadas sai também a taxa em chamadas/min.
- **Concorrência:** 4 requests simultâneas, gravadas no `config.json` na F0. Não muda depois disso.

---

## Passo 1 — F0: sonda e testes de fumaça (~2–5 min, 43 chamadas)

```bash
python3 EXIST_MEMES/glm_annotator/run_probe.py --concurrency 4
```

O que este passo faz:
- grava `config.json`, `train_ids.json`, `val_ids.json` e `pilot_ids.json`;
- G0.1: confere o servidor e grava a versão;
- G0.2: testa o thinking desligado;
- G0.3: grava `probe_vision.csv`;
- G0.4: mede a vazão.

**Se der erro em G0.1:** o modelo não está no servidor. Pare e fale com quem mantém o servidor.

## Passo 2 — G0.3: revisão manual da sonda visual (~10 min)

Abra `EXIST_MEMES/glm_reference/probe_vision.csv`. Para cada linha, abra a imagem em `image_path` e compare com as descrições.

| Passa se | Falha se |
|---|---|
| todas as `desc_white` descrevem algo branco/vazio **e** ≥ 16 de 20 `desc_real` correspondem ao meme | a imagem branca vira uma "cena", ou as descrições reais não batem com os memes |

Se passou:
```bash
python3 EXIST_MEMES/glm_annotator/run_probe.py --approve-vision
```
Se falhou: **pare**. A torre de visão do FP8 pode estar quebrada; isso não é resultado sobre o método (§11).

## Passo 3 — F1: piloto, thinking ligado vs desligado (1 440 chamadas)

```bash
python3 EXIST_MEMES/glm_annotator/run_pilot.py --stage pilot
```

Tempo: ~0,3 h com o thinking desligado e ~2,4 h com ele ligado (sequencial, pela §14). Deve cair com concorrência 4 (até 4 requests simultâneas; ~2–3 req/s com thinking desligado).

Leia `reports/pilot_summary.md`. A regra da §6 já escolhe o braço sozinha.
**Se der "G1 FALHOU: nenhum braço elegível":** pare e registre como limite.

## Passo 4 — F1b: estabilidade (480 chamadas)

```bash
python3 EXIST_MEMES/glm_annotator/run_pilot.py --stage stability
```

O resultado é acrescentado ao `pilot_summary.md`. Com `flip_rate` total > 0,10, **pare** (G1 falhou).

## Passo 5 — F2: referência nos memes de treino (9 438 chamadas novas)

```bash
python3 EXIST_MEMES/glm_annotator/run_reference.py
```

Tempo: ~3,7 h sequencial com o thinking desligado; ~30 h com ele ligado.
Para não depender do terminal aberto, rode com `caffeinate` e `nohup`:

```bash
caffeinate -i nohup python3 EXIST_MEMES/glm_annotator/run_reference.py > EXIST_MEMES/glm_reference/f2.log 2>&1 &
tail -f EXIST_MEMES/glm_reference/f2.log
```

Saídas: `glm_votes.csv` e `reports/reference_summary.md`.
**G2 falhou** (parse < 0,98 ou confiantes < 0,70 em algum idioma): **pare**.

> **Desde 2026-10-09 os passos 6–8 seguem o `SPEC_27.md`** (continuação exploratória: desvio = distância ao soft label do GLM, portão G2′ = parse ≥ 0,98, estado primário da F5 = menor `eval_loss`, teste bilateral). Os comandos são os mesmos.

## Passo 6 — F3: pesos e novos soft labels (minutos, 0 chamadas)

```bash
python3 EXIST_MEMES/glm_annotator/weights.py
```

Roda os testes do Apêndice A antes de tudo. Grava:
- `annotator_weights.csv` e `soft_labels_weighted.json`;
- `reliability/outputs/soft_labels_taskA_glmw.json` e `soft_labels_taskA_glmbase.json`.

## Passo 7 — F4: checagens e portão G4 (minutos, 0 chamadas)

```bash
python3 EXIST_MEMES/glm_annotator/checks.py
```

1. Leia `reports/gate_decision.md`.
   - **NÃO AUTORIZADO:** encerre, sem gastar GPU.
   - **AUTORIZADO:** siga para o item 2.
2. Revise à mão `manual_review_unanimous.csv` (checagem (e): 30 memes unânimes que o GLM contrariou).
3. Leia `reports/checks_summary.md`.

## Passo 8 — F5: retreino na VM (6 runs × 15 épocas)

1. Antes de tudo:
   ```bash
   python3 EXIST_MEMES/glm_annotator/analyze_f5.py --precheck
   ```
2. No `memes_model.ipynb`, **célula 6**, mude só estas duas linhas, uma run por vez:

   | Run | `ARM` | `SEED` |
   |---|---|---|
   | 1–3 | `'glmbase'` | `42`, `7`, `2025` |
   | 4–6 | `'glmw'` | `42`, `7`, `2025` |

3. Numa célula **nova**, logo depois da que carrega `_SOFT_A` (portão 12.6):
   ```python
   import sys; sys.path.insert(0, 'glm_annotator')
   from analyze_f5 import precheck; precheck(ARM, _SOFT_A)
   ```
4. Depois das 6 runs:
   ```bash
   python3 EXIST_MEMES/glm_annotator/analyze_f5.py
   ```
   Saída: `reports/f5_report.md`, com o critério da §11 avaliado em E = argmin(eval_loss) e no estado final.

## Passo 9 — Registrar o resultado

Preencha o **§17 do `SPEC_25.md`** como seção datada, sem editar as seções acima. Isso vale qualquer que seja o desfecho, inclusive portão falho.

---

## Se algo sair do roteiro

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `faltam N chamadas (falha de transporte?)` | rede ou servidor caiu, ou HTTP 429/5xx esgotou as retentativas | rodar o mesmo comando de novo |
| `versão do servidor mudou` | o servidor foi atualizado entre fases | **parar**: a comparação entre fases deixa de valer. Falar com o orientador |
| `config.json['fixed'] diverge` | alguém mudou um parâmetro em `common.py` | reverter a mudança. Parâmetro pré-registrado não muda |
| `portão X não aprovado` | a fase anterior falhou ou não rodou | ver `config.json` → `runtime.gates` |
