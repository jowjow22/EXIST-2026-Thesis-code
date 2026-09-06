"""Calibração pós-treino por task (item 4 do doc 17 / opção 6 do doc 20).

Zero GPU de treino: roda uma vez sobre as predições de um checkpoint já treinado.

**Por que agora.** O doc 20 §4 barrava explicitamente a calibração até o probing de
heads terminar ("Opção 6 — NUNCA antes"). O probing terminou (doc 21, INCONCLUSIVO) e as
duas rotas de realocação de treino entre tasks fecharam (docs 18 §9.5 e 22 §6). A
calibração é ortogonal a esse eixo: não muda a representação — muda como as
probabilidades são espalhadas, que é exatamente o que o ICM-Soft mede.

**Protocolo anti-leakage — idêntico ao de `vc.tune_task_c_thresholds` (ARCH-11 b):**
a temperatura é ajustada numa metade da validação e REPORTADA na outra, com o mesmo
`vc.split_val_halves(n, SPLIT_SEED)`. O número reportável é sempre
`scores_on_report_half`. Ajustar e reportar na mesma metade é o vazamento que essa
função existe para evitar.

⚠ **A escolha escalar-vs-vetor é PRÉ-REGISTRADA, não escolhida pelo resultado:**
temperatura escalar em A e B, vector scaling (uma temperatura por classe) na C — como
consta no item 4 do doc 17, escrito antes de qualquer medição. A variante escalar da C é
calculada e gravada **só como registro**; escolher entre as duas olhando a métrica de
reporte seria leakage.

⚠ **Assimetria a declarar na tese** (a mesma de `tune_task_c_thresholds`): os números
desta função vêm de ~metade do val, enquanto os de `compute_metrics` vêm do val inteiro.
Não colocar lado a lado sem essa nota.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

import video_core as vc


# ============================================================================
# Perdas soft usadas como objetivo de calibração
# ============================================================================
# O alvo é a métrica SOFT (ICM-Soft-Norm), logo o objetivo é a log-verosimilhança das
# distribuições de votos — não a accuracy. Calibrar contra rótulos hard destruiria
# exatamente a informação que o paradigma LeWiDi quer preservar.

def _nll_binary_soft(logits, targets, temperature):
    """NLL binária com alvo soft (proporção de votos YES). `logits` e `targets` são [N]."""
    z = logits / temperature
    # log-sigmoid estável: -log(1+exp(-z)) para z>=0, z - log(1+exp(z)) para z<0
    log_p = np.where(z >= 0, -np.log1p(np.exp(-np.abs(z))), z - np.log1p(np.exp(-np.abs(z))))
    log_1mp = log_p - z
    return -np.mean(targets * log_p + (1 - targets) * log_1mp)


def _nll_categorical_soft(logits, targets, temperature):
    """Cross-entropia soft sobre softmax. `logits`/`targets` são [N, K]."""
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return -np.mean((targets * log_p).sum(axis=-1))


def _nll_multilabel_soft(logits, targets, temperatures):
    """BCE soft por classe, com uma temperatura por classe (vector scaling).

    `temperatures` pode ser escalar (broadcast) ou [K]. A redução é `.mean` sobre as
    classes — a mesma de `vc.loss_task_c`, para o objetivo de calibração não introduzir
    uma assimetria diferente da que o treino usou.
    """
    z = logits / np.asarray(temperatures).reshape(1, -1)
    log_p = np.where(z >= 0, -np.log1p(np.exp(-np.abs(z))), z - np.log1p(np.exp(-np.abs(z))))
    log_1mp = log_p - z
    per_class = -(targets * log_p + (1 - targets) * log_1mp)
    return float(per_class.mean())


# ============================================================================
# Ajuste das temperaturas (só na metade de AJUSTE)
# ============================================================================

def _fit_scalar_temperature(nll_fn, lo: float = 0.05, hi: float = 20.0) -> float:
    """Minimiza `nll_fn(T)` em T > 0. Busca em log-T: a escala é multiplicativa
    (T=0,5 e T=2 são simétricos), e em log o mínimo é bem-comportado.

    Grid grosso + refino local, sem scipy — 200 avaliações de uma NLL vetorizada são
    milissegundos, e evita uma dependência a mais só para uma otimização 1-D.
    """
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 120))
    vals = [nll_fn(float(t)) for t in grid]
    i = int(np.argmin(vals))

    left = grid[max(i - 1, 0)]
    right = grid[min(i + 1, len(grid) - 1)]
    fine = np.exp(np.linspace(np.log(left), np.log(right), 80))
    vals_fine = [nll_fn(float(t)) for t in fine]
    return float(fine[int(np.argmin(vals_fine))])


def _fit_vector_temperature(logits, targets, n_classes: int,
                            rounds: int = 3) -> list:
    """Vector scaling: uma temperatura por classe, por descida coordenada.

    Cada classe da Task C é uma BCE independente (as categorias não somam 1), logo o
    objetivo separa exatamente por classe — a descida coordenada converge na primeira
    passagem. As `rounds` extras são verificação barata, não necessidade.
    """
    temps = [1.0] * n_classes
    for _ in range(rounds):
        for k in range(n_classes):
            def nll_k(t, k=k):
                return _nll_binary_soft(logits[:, k], targets[:, k], t)
            temps[k] = _fit_scalar_temperature(nll_k)
    return [float(t) for t in temps]


# ============================================================================
# Avaliação: PyEvALL soft nas duas metades, com e sem calibração
# ============================================================================

def _eval_soft(probs, targets, task, temp_dir) -> dict:
    ids = [str(i) for i in range(len(probs))]
    return vc._run_pyevall(probs, targets, ids, task, 'soft', temp_dir=temp_dir)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_from_predictions(logits_a, labels_a, logits_b, labels_b,
                               logits_c, labels_c,
                               split_seed: int = None) -> dict:
    """Ajusta temperatura por task na metade de AJUSTE e reporta na metade de REPORTE.

    Devolve um dict pronto para gravar em JSON, com — para cada task — a temperatura
    ajustada, as métricas soft ANTES (T=1) e DEPOIS nas duas metades, e o delta.

    O número reportável é sempre o da metade de REPORTE. O da metade de ajuste está lá
    para diagnosticar overfit da própria calibração (se o ganho existe só na metade de
    ajuste, a temperatura ajustou ruído).
    """
    split_seed = vc.SPLIT_SEED if split_seed is None else split_seed
    logits_a, labels_a = np.asarray(logits_a), np.asarray(labels_a)
    logits_b, labels_b = np.asarray(logits_b), np.asarray(labels_b)
    logits_c, labels_c = np.asarray(logits_c), np.asarray(labels_c)
    if labels_a.ndim == 2:                 # mesma guarda de vc.compute_metrics
        labels_a = labels_a[:, 1]

    out = {"split_seed": int(split_seed),
           "protocol": ("Temperatura ajustada por NLL soft na metade de AJUSTE, "
                        "reportada na metade de REPORTE (vc.split_val_halves). "
                        "Escalar em A/B, vector scaling na C — pré-registrado no item 4 "
                        "do doc 17, NÃO escolhido pelo resultado."),
           "tasks": {}}

    with tempfile.TemporaryDirectory() as temp_dir:
        # ---------------- Task A: temperatura escalar ----------------
        mask = labels_a != vc.SENTINEL_A
        if mask.any():
            z, y = logits_a[mask].reshape(-1), labels_a[mask]
            idx_tune, idx_report = vc.split_val_halves(len(z), split_seed)
            T = _fit_scalar_temperature(
                lambda t: _nll_binary_soft(z[idx_tune], y[idx_tune], t))

            def probs_a(zz, t):
                p = _sigmoid(zz / t)
                return np.stack([1 - p, p], axis=-1)   # eixo = TASK_A_PROB_ORDER

            targ = lambda yy: np.stack([1 - yy, yy], axis=-1)
            out["tasks"]["a"] = {
                "kind": "scalar", "temperature": T,
                "n_tune": int(len(idx_tune)), "n_report": int(len(idx_report)),
                "nll_tune_before": float(_nll_binary_soft(z[idx_tune], y[idx_tune], 1.0)),
                "nll_tune_after": float(_nll_binary_soft(z[idx_tune], y[idx_tune], T)),
                "before_on_report_half": _eval_soft(probs_a(z[idx_report], 1.0),
                                                    targ(y[idx_report]), '2_1', temp_dir),
                "after_on_report_half": _eval_soft(probs_a(z[idx_report], T),
                                                   targ(y[idx_report]), '2_1', temp_dir),
                "before_on_tune_half": _eval_soft(probs_a(z[idx_tune], 1.0),
                                                  targ(y[idx_tune]), '2_1', temp_dir),
                "after_on_tune_half": _eval_soft(probs_a(z[idx_tune], T),
                                                 targ(y[idx_tune]), '2_1', temp_dir),
            }

        # ---------------- Task B: temperatura escalar ----------------
        mask = labels_b[:, 0] != vc.SENTINEL_BC
        if mask.any():
            z, y = logits_b[mask], labels_b[mask]
            idx_tune, idx_report = vc.split_val_halves(len(z), split_seed)
            T = _fit_scalar_temperature(
                lambda t: _nll_categorical_soft(z[idx_tune], y[idx_tune], t))

            def probs_b(zz, t):
                x = zz / t
                x = x - x.max(axis=-1, keepdims=True)
                e = np.exp(x)
                return e / e.sum(axis=-1, keepdims=True)

            out["tasks"]["b"] = {
                "kind": "scalar", "temperature": T,
                "n_tune": int(len(idx_tune)), "n_report": int(len(idx_report)),
                "nll_tune_before": float(_nll_categorical_soft(z[idx_tune], y[idx_tune], 1.0)),
                "nll_tune_after": float(_nll_categorical_soft(z[idx_tune], y[idx_tune], T)),
                "before_on_report_half": _eval_soft(probs_b(z[idx_report], 1.0),
                                                    y[idx_report], '2_2', temp_dir),
                "after_on_report_half": _eval_soft(probs_b(z[idx_report], T),
                                                   y[idx_report], '2_2', temp_dir),
                "before_on_tune_half": _eval_soft(probs_b(z[idx_tune], 1.0),
                                                  y[idx_tune], '2_2', temp_dir),
                "after_on_tune_half": _eval_soft(probs_b(z[idx_tune], T),
                                                 y[idx_tune], '2_2', temp_dir),
            }

        # ---------------- Task C: vector scaling (pré-registrado) ----------------
        mask = labels_c[:, 0] != vc.SENTINEL_BC
        if mask.any():
            z, y = logits_c[mask], labels_c[mask]
            idx_tune, idx_report = vc.split_val_halves(len(z), split_seed)
            n_cls = z.shape[-1]
            T_vec = _fit_vector_temperature(z[idx_tune], y[idx_tune], n_cls)
            # Escalar só como REGISTRO — não é a variante reportável (ver docstring).
            T_scalar = _fit_scalar_temperature(
                lambda t: _nll_multilabel_soft(z[idx_tune], y[idx_tune], [t] * n_cls))

            def probs_c(zz, temps):
                return _sigmoid(zz / np.asarray(temps).reshape(1, -1))

            ones = [1.0] * n_cls
            out["tasks"]["c"] = {
                "kind": "vector", "temperature": T_vec,
                "label_order": vc.LABELS_2_3,
                "temperature_scalar_REGISTRO_APENAS": T_scalar,
                "n_tune": int(len(idx_tune)), "n_report": int(len(idx_report)),
                "nll_tune_before": _nll_multilabel_soft(z[idx_tune], y[idx_tune], ones),
                "nll_tune_after": _nll_multilabel_soft(z[idx_tune], y[idx_tune], T_vec),
                "before_on_report_half": _eval_soft(probs_c(z[idx_report], ones),
                                                    y[idx_report], '2_3', temp_dir),
                "after_on_report_half": _eval_soft(probs_c(z[idx_report], T_vec),
                                                   y[idx_report], '2_3', temp_dir),
                "before_on_tune_half": _eval_soft(probs_c(z[idx_tune], ones),
                                                  y[idx_tune], '2_3', temp_dir),
                "after_on_tune_half": _eval_soft(probs_c(z[idx_tune], T_vec),
                                                 y[idx_tune], '2_3', temp_dir),
            }

    return out


def calibrate_run(trainer, eval_dataset, save_path=None,
                  split_seed: int = None) -> dict:
    """Wrapper que roda `trainer.predict` e chama `calibrate_from_predictions`.

    ⚠ Rodar SÓ UMA VEZ, depois do treino — mesma regra de `vc.tune_task_c_thresholds`.
    """
    print("[calibração] gerando predições no conjunto de validação...")
    pred_out = trainer.predict(eval_dataset)
    logits_a, logits_b, logits_c = vc._unpack_predictions(pred_out.predictions)
    labels_a, labels_b, labels_c = pred_out.label_ids

    result = calibrate_from_predictions(logits_a, labels_a, logits_b, labels_b,
                                        logits_c, labels_c, split_seed=split_seed)
    print_calibration_report(result)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[calibração] gravado em {save_path}")
    return result


def print_calibration_report(result: dict) -> None:
    """Tabela legível. Só o delta da metade de REPORTE é reportável."""
    metric_by_task = {"a": "ICMSoftNorm", "b": "ICMSoftNorm", "c": "ICMSoftNorm"}
    print("\n" + "=" * 78)
    print("CALIBRAÇÃO POR TASK — número reportável = Δ da metade de REPORTE")
    print("=" * 78)
    for task, d in result.get("tasks", {}).items():
        m = metric_by_task[task]
        rep_before = d["before_on_report_half"].get(m, float("nan"))
        rep_after = d["after_on_report_half"].get(m, float("nan"))
        tune_before = d["before_on_tune_half"].get(m, float("nan"))
        tune_after = d["after_on_tune_half"].get(m, float("nan"))
        d_rep, d_tune = rep_after - rep_before, tune_after - tune_before
        temp = d["temperature"]
        temp_str = (f"{temp:.4f}" if d["kind"] == "scalar"
                    else "[" + ", ".join(f"{t:.3f}" for t in temp) + "]")

        print(f"\nTask {task.upper()}  ({d['kind']} scaling, T = {temp_str})")
        print(f"  NLL na metade de ajuste : {d['nll_tune_before']:.4f} -> "
              f"{d['nll_tune_after']:.4f}   (cai sempre — é o objetivo otimizado, "
              f"não é evidência de nada)")
        print(f"  {m} REPORTE : {rep_before:.4f} -> {rep_after:.4f}   "
              f"(Δ {d_rep:+.4f})  <- REPORTÁVEL")
        print(f"  {m} ajuste  : {tune_before:.4f} -> {tune_after:.4f}   (Δ {d_tune:+.4f})")

        # Overfit da própria calibração: ganho grande no ajuste que não sobrevive no
        # reporte. Comparar os DOIS deltas — o delta do ajuste sozinho não diz nada.
        if d_tune > 0.01 and d_rep <= 0:
            print("    ⚠ ganha no ajuste e NÃO no reporte -> a temperatura ajustou "
                  "ruído. Reportar como nulo, não como ganho.")
        elif d["kind"] == "scalar" and abs(temp - 1.0) < 0.05:
            print("    T ≈ 1 -> esta task já estava calibrada. Resultado legítimo.")

    print("\n" + "=" * 78)
    print("⚠ Métricas HARD são invariantes a temperature scaling por construção "
          "(argmax em B; o cruzamento de 0,5 da sigmoide em A/C fica em logit 0 para "
          "qualquer T>0). Se alguma métrica hard se mexer, o problema é no pipeline, "
          "não na calibração — é um teste de resposta conhecida, de graça.")
    print("⚠ Estes números vêm de ~metade do val; os de compute_metrics vêm do val "
          "inteiro. Não colocar lado a lado sem essa nota (mesma assimetria de "
          "tune_task_c_thresholds).")


# ============================================================================
# Item 9 do doc 17 — diagnóstico dos NaN do PyEvALL (custo zero de GPU)
# ============================================================================

def diagnose_pyevall_nans(trainer, eval_dataset, n_show: int = 3) -> dict:
    """Descobre POR QUE `fmeasure_hard` (B e C) e `crossentropy_soft` (B e C) vêm `nan`.

    `vc._extract_metrics_from_report` devolve `nan` em dois casos que ele não distingue:
    a métrica não veio no relatório, ou veio com `status != 'OK'` (ARCH-14 — a escolha
    de devolver `nan` em vez de omitir foi deliberada, para a coluna não sumir da tabela
    do Trainer). Esta função abre o relatório cru e imprime qual dos dois é, por métrica,
    com a mensagem do PyEvALL quando existe.

    Diagnóstico puro: não corrige nada, não altera nenhuma métrica reportada.
    """
    print("[NaN] gerando predições...")
    pred_out = trainer.predict(eval_dataset)
    logits_a, logits_b, logits_c = vc._unpack_predictions(pred_out.predictions)
    labels_a, labels_b, labels_c = pred_out.label_ids
    logits_a, logits_b, logits_c = map(np.asarray, (logits_a, logits_b, logits_c))
    labels_a, labels_b, labels_c = map(np.asarray, (labels_a, labels_b, labels_c))
    if labels_a.ndim == 2:
        labels_a = labels_a[:, 1]

    prob_yes = _sigmoid(logits_a[labels_a != vc.SENTINEL_A].reshape(-1))
    mask_b = labels_b[:, 0] != vc.SENTINEL_BC
    xb = logits_b[mask_b] - logits_b[mask_b].max(axis=-1, keepdims=True)
    mask_c = labels_c[:, 0] != vc.SENTINEL_BC

    cases = [
        ('2_1', np.stack([1 - prob_yes, prob_yes], axis=-1),
         np.stack([1 - labels_a[labels_a != vc.SENTINEL_A],
                   labels_a[labels_a != vc.SENTINEL_A]], axis=-1)),
        ('2_2', np.exp(xb) / np.exp(xb).sum(axis=-1, keepdims=True), labels_b[mask_b]),
        ('2_3', _sigmoid(logits_c[mask_c]), labels_c[mask_c]),
    ]

    findings = {}
    from pyevall.evaluation import PyEvALLEvaluation
    from pyevall.utils.utils import PyEvALLUtils

    with tempfile.TemporaryDirectory() as td:
        for task, probs, gold in cases:
            for evaluation in ('hard', 'soft'):
                metric_list = vc._metric_list(task, evaluation)
                preds_f = vc._format_for_pyevall(probs, [str(i) for i in range(len(probs))],
                                                 task, evaluation)
                gold_f = vc._format_for_pyevall(gold, [str(i) for i in range(len(gold))],
                                                task, evaluation)
                pp, gp = f"{td}/p_{task}_{evaluation}.json", f"{td}/g_{task}_{evaluation}.json"
                json.dump(preds_f, open(pp, "w"))
                json.dump(gold_f, open(gp, "w"))

                kwargs = {}
                hierarchy = getattr(vc, f"TASK_{task.upper()}_HIERARCHY", None)
                if hierarchy is not None:
                    kwargs[PyEvALLUtils.PARAM_HIERARCHY] = hierarchy
                rep = PyEvALLEvaluation().evaluate(pp, gp, metric_list, **kwargs)

                metrics_dict = getattr(rep, "report", {}).get("metrics", {})
                for m in metric_list:
                    key = f"{task}_{evaluation}_{m}"
                    if m not in metrics_dict:
                        findings[key] = {"causa": "AUSENTE do relatório"}
                        continue
                    entry = metrics_dict[m]
                    status = entry.get("status")
                    if status != "OK":
                        findings[key] = {"causa": f"status={status}",
                                         "detalhe": {k: v for k, v in entry.items()
                                                     if k != "results"}}
                    else:
                        try:
                            val = float(entry["results"]["average_per_test_case"])
                            findings[key] = {"causa": "OK", "valor": val}
                        except (KeyError, TypeError, ValueError) as e:
                            findings[key] = {"causa": f"OK mas valor ilegível: {e!r}"}

    print("\n" + "=" * 78)
    print("DIAGNÓSTICO DOS NaN (item 9 do doc 17)")
    print("=" * 78)
    for key, info in findings.items():
        marca = "  " if info["causa"] == "OK" else "⚠ "
        extra = f"  valor={info['valor']:.4f}" if "valor" in info else ""
        print(f"{marca}{key:36s} {info['causa']}{extra}")
        if "detalhe" in info:
            for k, v in list(info["detalhe"].items())[:n_show]:
                print(f"      {k}: {v}")
    return findings
