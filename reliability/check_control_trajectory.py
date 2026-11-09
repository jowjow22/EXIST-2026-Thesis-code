"""Compara a trajectória de uma run de CONTROLO (lambda = 0) com a referência.

Porquê existe
-------------
A Fase 2 acrescenta um callback com dois `DataLoader` próprios. O
`_BaseDataLoaderIter.__init__` consome do RNG global do CPU a cada `iter()`, e
se isso acontecesse a ordem dos batches de treino e o dropout mudariam a partir
da primeira avaliação -- a run deixaria de ser comparável com o A0, em silêncio.

O callback tem geradores próprios e uma guarda que aborta se o RNG se mexer, e
o teste T12c cobre o caso em CPU com um modelo falso. Este script fecha o
último buraco: a run REAL, em CUDA, com o modelo a sério.

O que verificar
---------------
1. `eval_loss` da época 1 **bit-a-bit** igual à referência. A avaliação da
   época 1 corre ANTES de o callback poder ter causado qualquer deriva, logo
   esta linha testa o FORWARD (a reparametrização `sd` no init tem de dar
   exactamente as mesmas matrizes que a `delta` -- teste T9).
2. As perdas de TREINO da época 2, casadas por `step`. Estas correm DEPOIS da
   primeira passagem do callback, logo testam o RNG. O piso medido na Fase 1 é
   ~6e-8 relativo, vindo do `clip_grad_norm_`; qualquer coisa acima disso é
   deriva a sério.

Uso:
    python3 reliability/check_control_trajectory.py <run.json> [referencia.json]

Sem referência explícita, usa a run de controlo da Fase 1:
    EXIST_MEMES/metrics_history/20260907_083944_conf_M1_lam0.0_seed42.json
"""

import json
import os
import sys

DEFAULT_REF = ("EXIST_MEMES/metrics_history/"
               "20260907_083944_conf_M1_lam0.0_seed42.json")
FLOOR = 1e-7          # piso do `clip_grad_norm_` medido na Fase 1 (~6e-8)


def rows(hist, key):
    return {int(r["step"]): float(r[key]) for r in hist if key in r and "step" in r}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cur_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REF
    for p in (cur_path, ref_path):
        if not os.path.exists(p):
            print(f"FALTA: {p}")
            return 2
    cur = json.load(open(cur_path, encoding="utf-8"))
    ref = json.load(open(ref_path, encoding="utf-8"))
    print(f"run        : {os.path.basename(cur_path)}")
    print(f"referência : {os.path.basename(ref_path)}\n")

    fails = []

    # -- (1) eval_loss da época 1: testa o FORWARD ---------------------------
    ce = [r for r in cur if "eval_loss" in r]
    re_ = [r for r in ref if "eval_loss" in r]
    if ce and re_:
        d = ce[0]["eval_loss"] - re_[0]["eval_loss"]
        ok = (d == 0.0)
        print(f"[{'PASS' if ok else 'FAIL'}] eval_loss da época 1 bit-a-bit  "
              f"ref={re_[0]['eval_loss']:.10f} run={ce[0]['eval_loss']:.10f} "
              f"delta={d:+.2e}")
        if not ok:
            fails.append("forward (eval_loss ép.1)")
            print("        -> o FORWARD mudou. Suspeitos: `param='sd'` não "
                  "reproduz o init da `delta` (T9), ou LAMBDA != 0.")
    else:
        print("[SKIP] sem avaliações em algum dos ficheiros")

    # -- (1b) eval_loss das épocas seguintes: também testam o RNG ------------
    # Com `logging_steps=50` e ~53 passos por época, há aproximadamente UMA
    # linha de `loss` de treino por época -- pouco para julgar deriva. As
    # avaliações da época 2 em diante já correm depois de o callback ter
    # passado, logo servem para o mesmo efeito.
    if len(ce) > 1 and len(re_) > 1:
        print()
        worst_e, worst_ep = 0.0, None
        for k, (x, y) in enumerate(zip(re_[1:], ce[1:]), start=2):
            rel = abs(y["eval_loss"] - x["eval_loss"]) / max(abs(x["eval_loss"]), 1e-12)
            if rel > worst_e:
                worst_e, worst_ep = rel, k
            print(f"  ép.{k:<3} eval_loss ref={x['eval_loss']:.10f} "
                  f"run={y['eval_loss']:.10f} rel={rel:.2e}")
        ok_e = worst_e <= FLOOR
        print(f"[{'PASS' if ok_e else 'FAIL'}] eval_loss das épocas 2+  "
              f"pior delta relativo = {worst_e:.2e} (ép. {worst_ep})")
        if not ok_e:
            fails.append("RNG (eval_loss ép. 2+)")

    # -- (2) perdas de treino: testam o RNG ----------------------------------
    a, b = rows(ref, "loss"), rows(cur, "loss")
    shared = sorted(set(a) & set(b))
    if not shared:
        print("[SKIP] sem `loss` de treino em comum (logging_steps?)")
    else:
        worst, worst_step = 0.0, None
        print(f"\n{'step':>7} {'referência':>14} {'run':>14} {'delta rel.':>12}")
        for s in shared:
            rel = abs(b[s] - a[s]) / max(abs(a[s]), 1e-12)
            if rel > worst:
                worst, worst_step = rel, s
            print(f"{s:>7} {a[s]:>14.8f} {b[s]:>14.8f} {rel:>12.2e}")
        ok = worst <= FLOOR
        print(f"\n[{'PASS' if ok else 'FAIL'}] perdas de treino  "
              f"pior delta relativo = {worst:.2e} no step {worst_step} "
              f"(piso do clip_grad_norm_ = {FLOOR:.0e})")
        if not ok:
            fails.append("RNG (perdas de treino)")
            print("        -> a trajectória DERIVOU. O callback provavelmente "
                  "consumiu do RNG global; ver a guarda em `_evaluate`.")

    print()
    if fails:
        print("FALHAS:", ", ".join(fails))
        print("NÃO avançar para a run A antes de resolver.")
        return 1
    print("PASS — a régua não tocou na trajectória. Pode avançar para a run A.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
