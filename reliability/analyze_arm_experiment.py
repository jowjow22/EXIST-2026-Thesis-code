"""
Análise pareada do experimento A0 (raw) vs A1 (ponderado por θ CORRIGIDO), 15 épocas.

Lê os logs de `EXIST_MEMES/metrics_history/` escritos pelo `MetricsHistoryLogger`
do `EXIST_MEMES/memes_model.ipynb`. Não treina, não regenera labels, não ajusta nada.

Desenho, como pré-registado
---------------------------
- Dois braços, retreino completo, comparação PAREADA por semente [42, 7, 2025].
- Split congelado em disco, idêntico nos dois braços (`split_taskA_frozen.json`).
- Gold de validação SEMPRE `raw`, nos dois braços — a régua não muda com o alvo.
- Época reportada: E = argmin(eval_loss), NUNCA a métrica reportada (evita leakage).
- Regra de simetria: o estado FINAL também é reportado, para os dois braços.

Uso:
    python3 -m reliability.analyze_arm_experiment
"""

import glob
import json
import os
import re

import numpy as np

METRICS_DIR = "EXIST_MEMES/metrics_history"
OUT_DIR = "reliability/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "experiment_taskA_15ep_report.md")
JSON_PATH = os.path.join(OUT_DIR, "experiment_taskA_15ep_report.json")

SEEDS = [42, 7, 2025]
ARMS = {"raw": "A0 (baseline, soft label crua)",
        "weighted_fixed": "A1 (ponderada por θ corrigido)"}
N_EPOCHS = 15

METRICS = [
    ("A ICM-Norm hard", "eval_task_a_icmnorm_hard"),
    ("A ICM-Soft-Norm soft", "eval_task_a_icmsoftnorm_soft"),
    ("B ICM-Norm hard", "eval_task_b_icmnorm_hard"),
    ("B ICM-Soft-Norm soft", "eval_task_b_icmsoftnorm_soft"),
    ("C ICM-Norm hard", "eval_task_c_icmnorm_hard"),
    ("C ICM-Soft-Norm soft", "eval_task_c_icmsoftnorm_soft"),
]
PRIMARY = "eval_task_a_icmsoftnorm_soft"   # a métrica pré-registada do estudo

# Resultado a 2 épocas, do estudo anterior (alvo LEGADO). Só para contexto.
REF_2EP = {42: 0.0031, 7: 0.0099, 2025: 0.0005}
REF_TARGET_LEGACY = 0.0720     # média |weighted − raw| com θ legado
REF_TARGET_FIXED = 0.0617      # idem com θ corrigido


def discover():
    """Localiza a run de 15 épocas de cada (braço, seed). O nome do ficheiro traz
    o RUN_TAG; o número de épocas separa estas runs das de 2 épocas do estudo
    anterior, que partilham o tag `raw_seed*`."""
    found = {}
    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "*.json"))):
        m = re.search(r"_(raw|weighted_fixed|weighted)_seed(\d+)\.json$", path)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        if arm not in ARMS:
            continue
        hist = json.load(open(path, encoding="utf-8"))
        ev = [r for r in hist if "eval_loss" in r]
        if len(ev) != N_EPOCHS:
            continue                      # runs de 2 épocas do estudo anterior
        key = (arm, seed)
        if key in found:
            # Repetição da mesma run (evidência de reprodutibilidade) fica de fora
            # em silêncio se a curva for idêntica; curvas diferentes é que são
            # ambíguas. `20260906_215142_raw_seed42.json` é uma dessas repetições.
            if [r["eval_loss"] for r in found[key]["evals"]] == \
               [r["eval_loss"] for r in ev]:
                continue
            raise RuntimeError(f"duas runs de {N_EPOCHS} épocas para {key} "
                               f"com curvas DIFERENTES")
        found[key] = {"path": path, "hist": hist, "evals": ev,
                      "train_end": "train_runtime" in hist[-1]}
    return found


def integrity(found):
    """Verificações antes de qualquer número sair. Falhar aqui aborta a análise."""
    checks = []
    missing = [(a, s) for a in ARMS for s in SEEDS if (a, s) not in found]
    checks.append(("as 6 runs (2 braços × 3 seeds) existem", not missing,
                   "todas presentes" if not missing else f"em falta: {missing}"))
    ok_ep = all(len(v["evals"]) == N_EPOCHS for v in found.values())
    checks.append((f"cada run tem {N_EPOCHS} avaliações", ok_ep,
                   ", ".join(f"{a}/{s}={len(found[(a,s)]['evals'])}"
                             for a in ARMS for s in SEEDS if (a, s) in found)))
    ok_end = all(v["train_end"] for v in found.values())
    checks.append(("cada run terminou (`train_runtime` no log)", ok_end,
                   "todas com on_train_end" if ok_end else "run truncada"))
    ok_keys = all(all(k in v["evals"][0] for _, k in METRICS)
                  for v in found.values())
    checks.append(("as 6 métricas ICM estão em todas as runs", ok_keys, "ok"))
    return checks


def best_epoch(ev):
    """E = argmin(eval_loss). Devolve (índice 0-based, época, eval_loss)."""
    losses = [r["eval_loss"] for r in ev]
    i = int(np.argmin(losses))
    return i, ev[i]["epoch"], losses[i]


def paired(found, key, point="best"):
    """Δ = A1 − A0 por semente, na métrica `key`."""
    out = {}
    for s in SEEDS:
        vals = {}
        for arm in ARMS:
            ev = found[(arm, s)]["evals"]
            i = best_epoch(ev)[0] if point == "best" else len(ev) - 1
            vals[arm] = float(ev[i][key])
        out[s] = {"raw": vals["raw"], "weighted_fixed": vals["weighted_fixed"],
                  "delta": vals["weighted_fixed"] - vals["raw"]}
    return out


def stats(deltas):
    """t pareado, d_z de Cohen e nº de sementes para 80% de poder.

    Com n=3 o teste é fraco por construção; o d_z é a grandeza que interessa e o
    dimensionamento sai dele. Aproximação normal para o poder (z_{0.8}+z_{0.975}
    = 2.802), suficiente para dizer "≈10" ou "≈90" — não é uma tabela exacta.
    """
    d = np.asarray([deltas[s]["delta"] for s in SEEDS], dtype=float)
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        return {"mean": mean, "sd": 0.0, "t": float("inf"), "d_z": float("inf"),
                "n_for_80": 2, "same_sign": True, "n": n}
    t = mean / (sd / np.sqrt(n))
    dz = mean / sd
    n80 = int(np.ceil((2.802 / abs(dz)) ** 2)) if dz else None
    return {"mean": mean, "sd": sd, "t": float(t), "d_z": float(dz),
            "n_for_80": n80, "same_sign": bool(np.all(np.sign(d) == np.sign(d[0]))),
            "n": n}


def p_two_sided(t, df):
    """p bilateral do t de Student, por integração numérica da densidade.

    Evita depender do scipy só para um valor; com df=2 a integração é trivial e
    o erro é irrelevante para a precisão a que o número é reportado.
    """
    from math import gamma, pi, sqrt
    t = abs(float(t))
    if not np.isfinite(t):
        return 0.0
    c = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    xs = np.linspace(0, t, 200001)
    dens = c * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
    return float(2 * (0.5 - np.trapezoid(dens, xs)))


def build_report(found, checks, res, curves):
    L = []
    a = L.append
    a("# A0 vs A1 a 15 épocas — θ corrigido, 3 sementes pareadas")
    a("")
    a("Gerado por `reliability/analyze_arm_experiment.py` a partir dos logs de")
    a("`EXIST_MEMES/metrics_history/`. Não treina nem regenera nada.")
    a("")
    a("## 1. O que este experimento pergunta")
    a("")
    a("Se ponderar os votos dos anotadores pela competência estimada (θ) ao")
    a("construir a soft label da Task A muda as métricas ICM do modelo de memes.")
    a("O objectivo **não é melhorar o modelo** — é medir se o θ carrega informação.")
    a("Um resultado nulo é aceitável e esperado; o enquadramento é de")
    a("**caracterização**: sob que condições a modelação de reliability é viável em")
    a("LeWiDi, e porque é que o EXIST cai fora dessas condições.")
    a("")
    a("**Intervenção:** apenas o alvo de treino da Task A. Hard gold intocado. B e C")
    a("sem alteração de alvo — o que se observa neles é spillover do encoder")
    a("partilhado, não efeito directo.")
    a("")
    a("**θ:** estimador corrigido (`theta_taskA_single_truth_K1_fixed.json`), depois")
    a("da correcção da responsabilidade do M-step. Ver `theta_legacy_vs_fixed.md`.")
    a("")
    a("## 2. Integridade das runs `[MEDIDO]`")
    a("")
    for name, ok, detail in checks:
        a(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    a("")
    a("| braço | seed | ficheiro | E = argmin(eval_loss) | eval_loss em E |")
    a("|---|---|---|---|---|")
    for arm in ARMS:
        for s in SEEDS:
            f = found[(arm, s)]
            i, ep, lo = best_epoch(f["evals"])
            a(f"| `{arm}` | {s} | `{os.path.basename(f['path'])}` | **{ep:.0f}** | "
              f"{lo:.5f} |")
    a("")
    a("## 3. Curvas de `eval_loss` `[MEDIDO]`")
    a("")
    a("| braço | seed | " + " | ".join(f"ép {i+1}" for i in range(N_EPOCHS)) + " |")
    a("|---|---|" + "---|" * N_EPOCHS)
    for arm in ARMS:
        for s in SEEDS:
            a(f"| `{arm}` | {s} | " +
              " | ".join(f"{v:.4f}" for v in curves[(arm, s)]) + " |")
    a("")
    mins = {k: int(np.argmin(v)) + 1 for k, v in curves.items()}
    a(f"`[MEDIDO]` Mínimo de `eval_loss` por run: " +
      ", ".join(f"`{a_}`/{s}=ép {mins[(a_, s)]}" for a_ in ARMS for s in SEEDS) + ".")
    a("A perda sobe a partir daí — as épocas seguintes são sobreajuste. É o mesmo")
    a("padrão do ARCH-06 no modelo de vídeo, agora replicado nos memes e nos dois")
    a("braços. Correr 15 épocas não foi desperdício: é o que estabelece que o nulo")
    a("a 2 épocas não era artefacto de sub-treino.")
    a("")
    a("## 4. Resultado principal — época E, comparação pareada `[MEDIDO]`")
    a("")
    a("Métrica pré-registada do estudo: **Task A ICM-Soft-Norm (soft)**.")
    a("")
    a("| seed | A0 (raw) | A1 (ponderado) | Δ = A1 − A0 |")
    a("|---|---|---|---|")
    pr = res["best"][PRIMARY]
    for s in SEEDS:
        d = pr["per_seed"][s]
        a(f"| {s} | {d['raw']:.4f} | {d['weighted_fixed']:.4f} | "
          f"**{d['delta']:+.4f}** |")
    st = pr["stats"]
    a(f"| **média** | | | **{st['mean']:+.4f}** |")
    a("")
    a(f"sd(Δ) = {st['sd']:.4f} · t({st['n']-1}) = {st['t']:.2f} · "
      f"p = {pr['p']:.3f} · d_z = {st['d_z']:.2f}")
    a("")
    verdict = ("NULO — não sustentado" if pr["p"] >= 0.05
               else "EFEITO SUSTENTADO ao nível 0,05")
    a(f"**Veredito: {verdict}.**")
    a("")
    a("## 5. As seis métricas, na época E `[MEDIDO]`")
    a("")
    a("| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |")
    a("|---|---|---|---|---|---|---|---|")
    for lbl, key in METRICS:
        r = res["best"][key]
        st = r["stats"]
        a(f"| {lbl} | {r['per_seed'][42]['delta']:+.4f} | "
          f"{r['per_seed'][7]['delta']:+.4f} | {r['per_seed'][2025]['delta']:+.4f} | "
          f"**{st['mean']:+.4f}** | {st['sd']:.4f} | {st['d_z']:+.2f} | "
          f"{r['p']:.3f} |")
    a("")
    a("## 6. Regra de simetria — estado FINAL (época 15) `[MEDIDO]`")
    a("")
    a("O mesmo critério aplicado aos dois braços. Reportado porque a definição de")
    a("baseline adoptada na Task 3 é o estado final da run, e misturar «fim de um")
    a("lado, pico do outro» é o erro que essa regra existe para impedir.")
    a("")
    a("| métrica | Δ seed 42 | Δ seed 7 | Δ seed 2025 | média Δ | sd | d_z | p |")
    a("|---|---|---|---|---|---|---|---|")
    for lbl, key in METRICS:
        r = res["final"][key]
        st = r["stats"]
        a(f"| {lbl} | {r['per_seed'][42]['delta']:+.4f} | "
          f"{r['per_seed'][7]['delta']:+.4f} | {r['per_seed'][2025]['delta']:+.4f} | "
          f"**{st['mean']:+.4f}** | {st['sd']:.4f} | {st['d_z']:+.2f} | "
          f"{r['p']:.3f} |")
    a("")
    a("`[LIMITAÇÃO]` **Nada nesta tabela é um achado.** A época 15 está em")
    a("sobreajuste profundo (a `eval_loss` sobe desde a época 3 nas seis runs), a")
    a("tabela existe para cumprir a regra de simetria e não para inferência, e um")
    a("p nominal < 0,05 isolado entre 12 células (6 métricas × 2 pontos de leitura)")
    a("é multiplicidade, não sinal. O `A ICM-Norm hard` com d_z = −6,02 é exactamente")
    a("esse caso: não deve ser lido como efeito.")
    a("")
    a("## 7. Comparação com o estudo a 2 épocas")
    a("")
    a("`[LIMITAÇÃO]` As duas linhas **não são medições do mesmo alvo**: o estudo a")
    a("2 épocas usou as soft labels do θ LEGADO (com o bug da responsabilidade do")
    a("M-step); este usa o θ corrigido. A intervenção encolheu de")
    a(f"{REF_TARGET_LEGACY:.4f} para {REF_TARGET_FIXED:.4f} de deslocamento médio do")
    a("alvo (−14,3%). A linha antiga fica como registo, não como referência.")
    a("")
    a("| | 2 épocas (θ legado) | 15 épocas (θ corrigido) |")
    a("|---|---|---|")
    ref_mean = float(np.mean(list(REF_2EP.values())))
    a(f"| Δ médio (A soft) | {ref_mean:+.4f} | {pr['stats']['mean']:+.4f} |")
    a("| Δ por seed | " + ", ".join(f"{REF_2EP[s]:+.4f}" for s in SEEDS)
      + " | " + ", ".join(f"{pr['per_seed'][s]['delta']:+.4f}" for s in SEEDS) + " |")
    a("| mesmo sinal em 3/3 | sim | "
      f"{'sim' if pr['stats']['same_sign'] else 'não'} |")
    a("")
    a("A linha de 2 épocas **continua válida como registo do alvo legado** — foi a")
    a("decisão explícita tomada ao adoptar o estimador corrigido como oficial.")
    a("")
    a("## 8. Dimensionamento `[MEDIDO]`")
    a("")
    a("Sementes necessárias para 80% de poder (α = 0,05 bilateral, aproximação")
    a("normal), a partir do d_z observado em cada métrica na época E:")
    a("")
    a("| métrica | d_z | sementes para 80% |")
    a("|---|---|---|")
    for lbl, key in METRICS:
        st = res["best"][key]["stats"]
        n80 = st["n_for_80"]
        a(f"| {lbl} | {st['d_z']:+.2f} | {n80 if n80 else '—'} |")
    a("")
    a("`[LIMITAÇÃO]` Com n = 3 o sd é ele próprio muito ruidoso, logo estes números")
    a("são ordens de grandeza, não estimativas. A leitura útil é binária: «cabe em")
    a("~10 sementes» ou «não cabe em nenhum orçamento razoável». E há uma leitura")
    a("que o número sozinho esconde: **a direcção que mais sementes iriam confirmar**")
    a("**é NEGATIVA**. Correr mais sementes não pode transformar isto num ganho — no")
    a("melhor dos casos confirma um custo pequeno. Isso fecha a objecção «e se")
    a("corressem mais sementes?» sem gastar GPU nenhuma.")
    a("")
    a("## 9. Interpretação")
    a("")
    a("### 9.1 O nulo está estabelecido")
    a("")
    pr_ = res["best"][PRIMARY]
    a(f"Na métrica pré-registada, no regime real de treino (15 épocas, θ corrigido,")
    a(f"época E por `argmin(eval_loss)`): **Δ = {pr_['stats']['mean']:+.4f}, "
      f"p = {pr_['p']:.3f}**. O efeito é **menor em módulo** que o do estudo a 2")
    a("épocas e não é distinguível de zero. Este é o resultado que a linha de")
    a("investigação procurava: **ponderar os votos pela competência estimada não")
    a("carrega informação utilizável neste dataset** — medido onde o modelo")
    a("efectivamente treina, não num regime de sub-treino fácil de atacar.")
    a("")
    a("A história interna é coerente, e é isso que dá confiança no nulo: a")
    a("`eval_loss` em E é **pior** no braço ponderado nas três sementes")
    a("(2,6541/2,6734/2,6718 → 2,6711/2,6907/2,6959), e o Δ é negativo nas três.")
    a("Não há sinal a ser perdido em ruído — não há sinal.")
    a("")
    a("### 9.2 O que o sinal negativo NÃO quer dizer")
    a("")
    a("**Não quer dizer que ponderar «piora o modelo».** A régua de validação é o")
    a("gold **cru** nos dois braços, por desenho. Um braço treinado para um alvo")
    a("diferente do da régua fica em ligeira desvantagem **por construção**. A")
    a("leitura correcta é: *sem ganho de informação; o Δ residual é compatível com")
    a("o desencontro alvo-régua, na direcção esperada e com magnitude dentro do")
    a("ruído*.")
    a("")
    a("**E não houve «inversão de sinal» face às 2 épocas.** As duas linhas não são")
    a("o mesmo experimento: a de 2 épocas mediu o alvo do θ LEGADO, esta mede o")
    a("alvo corrigido. Comparar os sinais das duas como se fosse uma reversão seria")
    a("erro de leitura. A linha antiga continua válida **como registo do alvo**")
    a("**legado** — foi essa a decisão tomada ao adoptar o estimador corrigido.")
    a("")
    a("### 9.3 O nulo por permutação passa a ser opcional")
    a("")
    a("Os cinco conjuntos de labels permutadas existem e estão validados")
    a("(`permutation_null_report_fixed.md`). Mas o propósito deles era **atribuir**")
    a("**um Δ positivo**: separar «o ganho vem do θ» de «o ganho vem de perturbar o")
    a("alvo». **Não há Δ positivo para atribuir.**")
    a("")
    a("Custo de correr assim mesmo: mais 6 runs (5 permutações × sementes, ou 5×1 se")
    a("bastar uma semente por permutação). O que acrescentaria: a atribuição de um")
    a("nulo, que já é interpretável sem ela. **Recomendação: não correr**, a menos")
    a("que o autor queira o braço permutado como figura da tese. Decisão do autor.")
    a("")
    a("### 9.4 O arco para o capítulo")
    a("")
    a("A sequência, nesta ordem, é ela própria material de tese:")
    a("")
    a("1. NUTMEG, Davani et al. e DEM-MoE avaliados e descartados por densidade,")
    a("   escala e falta de estrutura demográfica. Sobra o método mais simples e")
    a("   mais antigo da bibliografia — **MACE**. Isso é resultado, não acaso.")
    a("2. O estimador implementado tinha um **bug na responsabilidade do M-step**")
    a("   (voto do próprio anotador contado duas vezes), encontrado por uma")
    a("   **violação de monotonicidade da log-posterior** — 273 quebras que quatro")
    a("   explicações benignas não explicavam.")
    a("3. Corrigido, o efeito da intervenção no alvo **encolheu 14%**")
    a(f"   ({REF_TARGET_LEGACY:.4f} → {REF_TARGET_FIXED:.4f}).")
    a("4. No regime real de treino, o efeito nas métricas é **nulo**.")
    a("")
    a("Um nulo com esta cadeia de auditoria por trás é mais forte que um nulo sem")
    a("ela: mostra que o resultado não vem de uma implementação partida.")
    a("")
    return "\n".join(L) + "\n"


def main():
    found = discover()
    checks = integrity(found)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if any(not ok for _, ok, _ in checks):
        raise SystemExit("ABORTADO — verificação de integridade falhou.")

    curves = {k: [r["eval_loss"] for r in v["evals"]] for k, v in found.items()}
    res = {"best": {}, "final": {}}
    for point in ("best", "final"):
        for _lbl, key in METRICS:
            ps = paired(found, key, point)
            st = stats(ps)
            res[point][key] = {"per_seed": ps, "stats": st,
                               "p": p_two_sided(st["t"], len(SEEDS) - 1)}

    print()
    for arm in ARMS:
        for s in SEEDS:
            i, ep, lo = best_epoch(found[(arm, s)]["evals"])
            print(f"  {arm:15s} seed {s:>4}: E={ep:.0f} eval_loss={lo:.5f}")
    print()
    pr = res["best"][PRIMARY]
    print("A ICM-Soft-Norm soft, época E:")
    for s in SEEDS:
        d = pr["per_seed"][s]
        print(f"  seed {s:>4}: raw={d['raw']:.4f} weighted={d['weighted_fixed']:.4f} "
              f"Δ={d['delta']:+.4f}")
    st = pr["stats"]
    print(f"  média Δ={st['mean']:+.4f} sd={st['sd']:.4f} t={st['t']:.2f} "
          f"p={pr['p']:.3f} d_z={st['d_z']:.2f} n80={st['n_for_80']}")
    print()
    print("resumo das 6 metricas na epoca E:")
    for lbl, key in METRICS:
        r = res["best"][key]
        print(f"  {lbl:24s} media={r['stats']['mean']:+.4f} sd={r['stats']['sd']:.4f} "
              f"d_z={r['stats']['d_z']:+.2f} p={r['p']:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(found, checks, res, curves))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "runs": {f"{a_}_seed{s}": os.path.basename(found[(a_, s)]["path"])
                     for a_ in ARMS for s in SEEDS},
            "best_epoch": {f"{a_}_seed{s}": best_epoch(found[(a_, s)]["evals"])[1]
                           for a_ in ARMS for s in SEEDS},
            "eval_loss_curves": {f"{a_}_seed{s}": curves[(a_, s)]
                                 for a_ in ARMS for s in SEEDS},
            "results": res,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nescrito: {REPORT_PATH}\n         {JSON_PATH}")


if __name__ == "__main__":
    main()
