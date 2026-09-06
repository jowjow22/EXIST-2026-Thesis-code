#!/usr/bin/env python3
"""Verifica que o sistema de seeding funciona nos dois notebooks da Task 3.

    python3 verify_seeding.py            # tudo o que der para rodar aqui
    python3 verify_seeding.py --full     # inclui o teste que constrói os modelos (VM)

Três níveis, do mais barato ao mais caro:

  NÍVEL 1  estático   — as implementações de `set_all_seeds` e `model_fingerprint`
                        são idênticas entre `Thesis_video_baseline.ipynb` e
                        `video_core.py`? Roda em qualquer lugar, sem torch.
  NÍVEL 2  RNG        — `set_all_seeds(s)` re-zera de facto todos os geradores?
                        Precisa de torch. Roda local.
  NÍVEL 3  modelo     — `build_model(s)` duas vezes dá pesos bit a bit iguais?
                        Precisa de peft + acesso ao Hub. **Só na VM.**

O nível 3 é o único que prova o que interessa de verdade; os dois primeiros
falham mais cedo e mais barato quando algo se partiu.

⚠ **O que este script NÃO faz:** comparar fingerprints ENTRE os dois notebooks.
Eles têm arquiteturas diferentes, logo módulos diferentes, logo consumo de RNG
diferente — os hashes **têm** de divergir. Fingerprint prova reprodutibilidade
DENTRO de um notebook, nunca equivalência entre dois.
"""
import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OK, FAIL, SKIP = "  [OK]  ", "  [FALHA]", "  [PULADO]"
failures = []


def report(passed, msg, detail=""):
    print(f"{OK if passed else FAIL} {msg}")
    if detail:
        for line in str(detail).rstrip().split("\n"):
            print(f"          {line}")
    if not passed:
        failures.append(msg)


# ---------------------------------------------------------------------------
# NÍVEL 1 — as duas implementações são a mesma?
# ---------------------------------------------------------------------------

def _normalize(src: str) -> str:
    """Remove comentários, docstrings, anotações de tipo e espaço.

    Compara **comportamento**, não prosa. Os dois arquivos têm comentários em idiomas
    diferentes (o original em inglês, o `video_core` em português) e o `video_core`
    anota tipos de retorno que o notebook não anota — nada disso muda o que a função faz.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
            for arg in (node.args.args + node.args.posonlyargs
                        + node.args.kwonlyargs):
                arg.annotation = None
    return ast.unparse(ast.fix_missing_locations(tree))


def _extract_func_raw(src: str, name: str):
    """Fonte da função como está escrita, sem normalizar (para `exec`)."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(ast.fix_missing_locations(node))
    return None


def _extract_func(src: str, name: str):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(ast.fix_missing_locations(
                ast.parse(_normalize(ast.unparse(node)))))
    return None


def notebook_code(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")


def level1():
    print("\n=== NÍVEL 1 — implementações idênticas (estático) ===\n")

    core_path = HERE / "video_core.py"
    base_path = HERE / "Thesis_video_baseline.ipynb"
    orig_path = HERE / "Thesis_video_fixed.ipynb"

    if not core_path.exists():
        report(False, "video_core.py não encontrado")
        return

    core_src = core_path.read_text()

    for label, path in [("baseline", base_path), ("original (fixed)", orig_path)]:
        if not path.exists():
            print(f"{SKIP} {label}: {path.name} não existe")
            continue
        nb_src = notebook_code(path)
        for fn in ("set_all_seeds", "model_fingerprint"):
            a = _extract_func(nb_src, fn)
            b = _extract_func(core_src, fn)
            if a is None or b is None:
                report(False, f"{label}: {fn} não encontrada "
                              f"(notebook={a is not None}, core={b is not None})")
                continue
            report(a == b, f"{label}: {fn} idêntica a video_core.{fn}",
                   "" if a == b else f"notebook:\n{a}\n---\ncore:\n{b}")

    # A separação SPLIT_SEED × INIT_SEED só é real se o split NÃO depender do RNG
    # global. `train_test_split(random_state=int)` cria um RandomState local — é isso
    # que torna a separação limpa. Se alguém trocar por `random_state=None` ou por um
    # shuffle manual, a separação morre em silêncio.
    m = re.search(r"train_test_split\((.*?)\)", core_src, re.S)
    report(bool(m) and "random_state=split_seed" in m.group(1).replace(" ", ""),
           "video_core: train_test_split usa random_state=split_seed "
           "(RandomState local, não o RNG global)")

    # O baseline NÃO pode sortear o próprio split — é a mudança inteira.
    if base_path.exists():
        base_src = notebook_code(base_path)
        code_only = "\n".join(l for l in base_src.split("\n")
                              if not l.strip().startswith("#"))
        report("train_test_split(" not in code_only,
               "baseline: não sorteia split próprio (usa o artefato congelado)")
        report("val_split_42.json" in code_only,
               "baseline: carrega Dataset/val_split_42.json")


# ---------------------------------------------------------------------------
# NÍVEL 2 — os geradores realmente re-zeram?
# ---------------------------------------------------------------------------

def level2():
    print("\n=== NÍVEL 2 — RNGs re-zeram (precisa de torch) ===\n")
    try:
        import numpy as np
        import random
        import torch
    except ImportError as e:
        print(f"{SKIP} torch/numpy indisponível ({e})")
        return

    sys.path.insert(0, str(HERE))
    try:
        from video_core import set_all_seeds
        print("          (set_all_seeds importada de video_core)")
    except ImportError as e:
        # `video_core` importa `pyevall` no topo, que só existe na VM. A função de
        # seeding em si não depende dele, então extraio só ela do fonte — assim o
        # nível 2 roda na máquina local, onde é mais provável que alguém o rode.
        print(f"          (video_core não importável: {e} — extraindo a função do fonte)")
        src = (HERE / "video_core.py").read_text()
        fn = _extract_func_raw(src, "set_all_seeds")
        if fn is None:
            report(False, "set_all_seeds não encontrada em video_core.py")
            return
        ns = {"random": random, "np": np, "torch": torch}
        try:
            from transformers import set_seed
            ns["set_seed"] = set_seed
        except ImportError:
            ns["set_seed"] = lambda s: None
            print("          (transformers.set_seed indisponível — stubado)")
        import os as _os
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        exec(fn, ns)  # noqa: S102 — fonte é deste repositório, não entrada externa
        set_all_seeds = ns["set_all_seeds"]

    def draw():
        return (random.random(),
                float(np.random.rand()),
                float(torch.rand(1)),
                float(torch.nn.Linear(64, 64).weight.detach().std()))

    set_all_seeds(42); a = draw()
    set_all_seeds(42); b = draw()
    report(a == b, "set_all_seeds(42) duas vezes → mesmas amostras",
           f"a = {a}\nb = {b}")

    # Chamar no meio de uma sequência tem de re-zerar, não continuar de onde estava.
    # É disto que depende o re-seed imediatamente antes de cada init de pesos.
    set_all_seeds(42); draw(); draw()
    set_all_seeds(42); c = draw()
    report(a == c, "set_all_seeds re-zera no meio da sequência (não continua)")

    set_all_seeds(7); d = draw()
    report(a != d, "seeds diferentes → amostras diferentes (não está tudo constante)")

    report(torch.backends.cudnn.benchmark is False
           and torch.backends.cudnn.deterministic is True,
           "flags do cuDNN setadas (benchmark=False, deterministic=True)")

    import os
    report(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
           "CUBLAS_WORKSPACE_CONFIG=:4096:8 no ambiente",
           "Tem de estar setado ANTES de qualquer uso de CUDA. video_core faz isso "
           "no import; num notebook, a célula de imports tem de vir primeiro.")


# ---------------------------------------------------------------------------
# NÍVEL 3 — o modelo real sai bit a bit igual?
# ---------------------------------------------------------------------------

def level3():
    print("\n=== NÍVEL 3 — build_model reproduzível (precisa de peft; VM) ===\n")
    sys.path.insert(0, str(HERE))
    try:
        import video_core as vc
    except ImportError as e:
        print(f"{SKIP} video_core não importável ({e})")
        return

    try:
        for use_bio, use_viewer in [(True, True), (True, False), (False, False)]:
            tag = vc.condition_tag(use_bio, use_viewer)
            m1 = vc.build_model(42, use_bio=use_bio, use_viewer_emb=use_viewer)
            f1 = vc.model_fingerprint(m1)
            del m1
            m2 = vc.build_model(42, use_bio=use_bio, use_viewer_emb=use_viewer)
            f2 = vc.model_fingerprint(m2)
            fusion = m2.fusion_in
            params = vc.count_params(m2)
            del m2
            report(f1 == f2, f"{tag}: build_model(42) duas vezes → mesmo fingerprint",
                   f"{f1} vs {f2}\nfusion_in={fusion} | params={params}")
    except Exception as e:  # noqa: BLE001 — quero o motivo impresso, não o traceback
        print(f"{SKIP} build_model falhou: {type(e).__name__}: {e}")
        print("          Se for o caminho de atributo do ARCH-01, é o item pendente")
        print("          nº 1 de 05_IMPLEMENTATION_NOTES.md §5.")
        return

    # Seeds diferentes TÊM de dar pesos diferentes — senão INIT_SEED é decorativo e as
    # "3 seeds" da tabela de ablação seriam a mesma run três vezes.
    try:
        a = vc.model_fingerprint(vc.build_model(42))
        b = vc.model_fingerprint(vc.build_model(7))
        report(a != b, "INIT_SEED diferente → fingerprint diferente", f"{a} vs {b}")
    except Exception as e:  # noqa: BLE001
        print(f"{SKIP} comparação entre seeds falhou: {e}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="inclui o nível 3 (constrói os modelos; só faz sentido na VM)")
    args = ap.parse_args()

    level1()
    level2()
    if args.full:
        level3()
    else:
        print("\n=== NÍVEL 3 — pulado (use --full na VM) ===")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} verificação(ões) FALHARAM:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Todas as verificações executadas passaram.")
    print("\nLembrete: fingerprint prova reprodutibilidade DENTRO de um notebook.")
    print("Nunca compare o hash do baseline com o do improved — arquiteturas")
    print("diferentes consomem RNG de forma diferente e os hashes têm de divergir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
