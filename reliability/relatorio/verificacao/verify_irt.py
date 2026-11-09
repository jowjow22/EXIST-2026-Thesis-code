"""Reexecução das células de EXIST_MEMES/dataset_verifications.ipynb (0, 2–12, 14) com o caminho do dataset como argumento (verificação de 2026-10-24).
Uso: python3 verify_irt.py <caminho para EXIST2026_training.json>"""
import sys
import matplotlib
matplotlib.use("Agg")
import json
from collections import Counter
import os



PATH = sys.argv[1]  # ajusta ao teu caminho

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

print(f"tipo da raiz: {type(data).__name__}")

if isinstance(data, dict):
    items = list(data.values())
    print(f"n itens: {len(items)}")
    print(f"chaves de topo (3 primeiras): {list(data.keys())[:3]}")
else:
    items = data
    print(f"n itens: {len(items)}")

key_freq = Counter()
for it in items:
    if isinstance(it, dict):
        key_freq.update(it.keys())

print(f"\n--- chaves ao nivel do item ({len(key_freq)} distintas) ---")
for k, n in sorted(key_freq.items()):
    exemplo = next((it[k] for it in items if isinstance(it, dict) and k in it), None)
    amostra = repr(exemplo)
    if len(amostra) > 90:
        amostra = amostra[:90] + "..."
    print(f"{k:34s} n={n:6d}  {type(exemplo).__name__:5s}  {amostra}")

import numpy as np

def gerar_2pl(n_itens, n_anotadores, votos_por_item, seed):
    """
    Gera respostas binárias sob o modelo IRT 2PL.
    Devolve (respostas, theta, b, a).
    respostas: matriz (n_anotadores x n_itens), NaN onde não houve voto.
    """
    rng = np.random.default_rng(seed)

    theta = rng.normal(0.0, 1.0, size=n_anotadores)
    b     = rng.normal(0.0, 1.0, size=n_itens)
    a     = rng.lognormal(mean=0.0, sigma=0.3, size=n_itens)

    respostas = np.full((n_anotadores, n_itens), np.nan)

    for j in range(n_itens):
        quem = rng.choice(n_anotadores, size=votos_por_item, replace=False)
        logit = a[j] * (theta[quem] - b[j])
        p = 1.0 / (1.0 + np.exp(-logit))
        respostas[quem, j] = (rng.random(votos_por_item) < p).astype(float)

    return respostas, theta, b, a


resp, theta_true, b_true, a_true = gerar_2pl(
    n_itens=1000, n_anotadores=200, votos_por_item=50, seed=42
)

n_por_anotador = np.sum(~np.isnan(resp), axis=1)
n_por_item     = np.sum(~np.isnan(resp), axis=0)

print(f"matriz: {resp.shape}")
print(f"votos totais: {int(np.sum(~np.isnan(resp)))}")
print(f"votos/item:      min={n_por_item.min()}  max={n_por_item.max()}")
print(f"votos/anotador:  min={n_por_anotador.min()}  mediana={np.median(n_por_anotador):.0f}  max={n_por_anotador.max()}")
print(f"densidade da matriz: {np.mean(~np.isnan(resp)):.3f}")
print(f"taxa global de 1s: {np.nanmean(resp):.3f}")
print(f"a: media={a_true.mean():.3f}  min={a_true.min():.3f}  max={a_true.max():.3f}")

from scipy.optimize import minimize

def estimar_2pl_mml(respostas, n_pontos=21, max_iter=200, tol=1e-5,
                    prior_a_sigma=0.5, verbose=True):
    """
    IRT 2PL por MML-EM com quadratura de Gauss-Hermite.
    prior_a_sigma: sd do prior log-normal em a (None desliga).
    Devolve (theta_eap, b_hat, a_hat, historico_loglik).
    """
    n_ann, n_it = respostas.shape
    obs = ~np.isnan(respostas)
    R = np.nan_to_num(respostas, nan=0.0)

    nos, pesos = np.polynomial.hermite_e.hermegauss(n_pontos)
    pesos = pesos / pesos.sum()

    a_hat = np.ones(n_it)
    b_hat = np.zeros(n_it)
    historico = []

    for it_num in range(max_iter):
        logit = a_hat[None, :, None] * (nos[None, None, :] - b_hat[None, :, None])
        logp1 = -np.logaddexp(0.0, -logit)
        logp0 = -np.logaddexp(0.0, logit)

        contrib = obs[:, :, None] * (R[:, :, None] * logp1 + (1.0 - R)[:, :, None] * logp0)
        loglik_ann = contrib.sum(axis=1)

        m = loglik_ann.max(axis=1, keepdims=True)
        naolog = np.exp(loglik_ann - m) * pesos[None, :]
        marg = naolog.sum(axis=1, keepdims=True)
        post = naolog / marg
        loglik = float(np.sum(np.log(marg) + m))
        historico.append(loglik)

        n_kj = np.einsum('ik,ij->jk', post, obs.astype(float))
        r_kj = np.einsum('ik,ij->jk', post, R * obs)

        for j in range(n_it):
            def neg(par):
                aj, bj = par
                lg = aj * (nos - bj)
                lp1 = -np.logaddexp(0.0, -lg)
                lp0 = -np.logaddexp(0.0, lg)
                ll = np.sum(r_kj[j] * lp1 + (n_kj[j] - r_kj[j]) * lp0)
                if prior_a_sigma is not None:
                    ll -= (np.log(aj) ** 2) / (2.0 * prior_a_sigma ** 2)
                return -ll
            res = minimize(neg, x0=[a_hat[j], b_hat[j]], method='L-BFGS-B',
                           bounds=[(0.05, 5.0), (-10.0, 10.0)])
            a_hat[j], b_hat[j] = res.x

        if it_num > 0 and abs(historico[-1] - historico[-2]) < tol:
            if verbose:
                print(f"convergiu em {it_num+1} iteracoes")
            break

    quebras = sum(1 for i in range(1, len(historico)) if historico[i] < historico[i-1] - 1e-6)
    if verbose:
        print(f"loglik final: {historico[-1]:.2f}")
        print(f"quebras de monotonicidade: {quebras} / {len(historico)-1}")
        n_lim = int(np.sum((a_hat <= 0.06) | (a_hat >= 4.9) | (np.abs(b_hat) >= 9.9)))
        print(f"itens no bordo: {n_lim} / {n_it}")

    theta_eap = post @ nos
    return theta_eap, b_hat, a_hat, historico


theta_hat, b_hat, a_hat, hist = estimar_2pl_mml(resp)

from scipy.stats import pearsonr, spearmanr

def avaliar_recuperacao(theta_true, b_true, a_true, theta_hat, b_hat, a_hat, rotulo=""):
    print(f"--- recuperacao {rotulo} ---")
    linhas = []
    for nome, vt, vh in [("theta", theta_true, theta_hat),
                         ("b    ", b_true, b_hat),
                         ("a    ", a_true, a_hat)]:
        r, p_r = pearsonr(vt, vh)
        rho, _ = spearmanr(vt, vh)
        print(f"{nome}  Pearson r = {r:+.4f}  (p={p_r:.2e})   Spearman = {rho:+.4f}")
        linhas.append((nome.strip(), r))

    r_log, _ = pearsonr(np.log(a_true), np.log(a_hat))
    print(f"a     Pearson r em log(a) = {r_log:+.4f}")

    print()
    print(f"theta:  true media={theta_true.mean():+.3f} sd={theta_true.std():.3f} | "
          f"est media={theta_hat.mean():+.3f} sd={theta_hat.std():.3f}")
    print(f"b:      true media={b_true.mean():+.3f} sd={b_true.std():.3f} | "
          f"est media={b_hat.mean():+.3f} sd={b_hat.std():.3f}")
    print(f"a:      true media={a_true.mean():.3f} sd={a_true.std():.3f} | "
          f"est media={a_hat.mean():.3f} sd={a_hat.std():.3f}")

    n_lim = int(np.sum((a_hat <= 0.06) | (a_hat >= 4.9) | (np.abs(b_hat) >= 5.9)))
    print(f"\nitens nos limites das bounds: {n_lim} / {len(a_hat)}")
    return dict(linhas)


res_sondagem = avaliar_recuperacao(theta_true, b_true, a_true,
                                   theta_hat, b_hat, a_hat,
                                   rotulo="sondagem 50 votos/item")

PASS_SONDAGEM = 0.80
veredicto = all(r >= PASS_SONDAGEM for r in res_sondagem.values())
print(f"\nveredicto sondagem (r >= {PASS_SONDAGEM} nos tres): "
      f"{'IMPLEMENTACAO CORRETA' if veredicto else 'FALHA — investigar'}")

def diagnostico_escala(theta_hat, b_hat, a_hat, b_true, a_true, theta_true):
    lim = (a_hat <= 0.06) | (a_hat >= 4.9) | (np.abs(b_hat) >= 5.9)
    print(f"itens no bordo: {lim.sum()}")
    print(f"  destes, a no bordo inferior (<=0.06): {int(np.sum(a_hat <= 0.06))}")
    print(f"  a no bordo superior (>=4.9):          {int(np.sum(a_hat >= 4.9))}")
    print(f"  |b| no bordo (>=5.9):                 {int(np.sum(np.abs(b_hat) >= 5.9))}")

    ok = ~lim
    r_b, _ = pearsonr(b_true[ok], b_hat[ok])
    r_a, _ = pearsonr(a_true[ok], a_hat[ok])
    print(f"\nexcluindo itens no bordo (n={ok.sum()}):")
    print(f"  b  Pearson r = {r_b:+.4f}")
    print(f"  a  Pearson r = {r_a:+.4f}")

    s = theta_true.std() / theta_hat.std()
    d = theta_true.mean() - s * theta_hat.mean()
    th_t = s * theta_hat + d
    b_t  = s * b_hat + d
    a_t  = a_hat / s
    print(f"\napos alinhamento linear (escala={s:.4f}, desloc={d:+.4f}):")
    for nome, vt, vh in [("theta", theta_true, th_t), ("b", b_true, b_t), ("a", a_true, a_t)]:
        r, _ = pearsonr(vt, vh)
        print(f"  {nome:5s} r = {r:+.4f}   est media={vh.mean():+.3f} sd={vh.std():.3f}")

diagnostico_escala(theta_hat, b_hat, a_hat, b_true, a_true, theta_true)

for sigma in [0.3, 0.6, 1.0]:
    r_, tt, bt, at = gerar_2pl(n_itens=1000, n_anotadores=200,
                               votos_por_item=50, seed=42)
    rng = np.random.default_rng(seed=1000 + int(sigma * 10))
    at = rng.lognormal(mean=0.0, sigma=sigma, size=1000)
    obs = ~np.isnan(r_)
    for j in range(1000):
        quem = np.where(obs[:, j])[0]
        p = 1.0 / (1.0 + np.exp(-at[j] * (tt[quem] - bt[j])))
        r_[quem, j] = (rng.random(len(quem)) < p).astype(float)

    th, bh, ah, _ = estimar_2pl_mml(r_, verbose=False)
    r_th, _ = pearsonr(tt, th)
    r_b, _  = pearsonr(bt, bh)
    r_a, _  = pearsonr(at, ah)
    print(f"sigma_a={sigma:.1f}  sd(a)={at.std():.3f}  |  "
          f"theta={r_th:+.3f}  b={r_b:+.3f}  a={r_a:+.3f}")

resultados = {}
for seed in [42, 7, 2025]:
    r_, tt, bt, at = gerar_2pl(n_itens=1000, n_anotadores=220,
                               votos_por_item=6, seed=seed)
    n_ann = np.sum(~np.isnan(r_), axis=1)
    th, bh, ah, _ = estimar_2pl_mml(r_, verbose=False)
    r_th, _ = pearsonr(tt, th)
    r_b, _  = pearsonr(bt, bh)
    r_a, _  = pearsonr(at, ah)
    resultados[seed] = (r_th, r_b, r_a)
    print(f"seed {seed:4d}  votos/anotador mediana={np.median(n_ann):.0f}  |  "
          f"theta={r_th:+.4f}  b={r_b:+.4f}  a={r_a:+.4f}")

m = np.array(list(resultados.values())).mean(axis=0)
print(f"\nMEDIA  theta={m[0]:+.4f}  b={m[1]:+.4f}  a={m[2]:+.4f}")
print("\n--- criterio pre-registado (densidade real) ---")
for nome, valor, lim in [("theta", m[0], 0.60), ("b", m[1], 0.50), ("a", m[2], 0.40)]:
    print(f"{nome:6s} {valor:+.4f}  vs  {lim:.2f}   {'PASS' if valor >= lim else 'FAIL'}")

def estimar_1pl_mml(respostas, n_pontos=21, max_iter=200, tol=1e-5, verbose=True):
    """
    IRT 1PL (Rasch) por MML-EM. Discriminacao fixa em a=1 para todos os itens.
    Devolve (theta_eap, b_hat, historico_loglik).
    """
    n_ann, n_it = respostas.shape
    obs = ~np.isnan(respostas)
    R = np.nan_to_num(respostas, nan=0.0)

    nos, pesos = np.polynomial.hermite_e.hermegauss(n_pontos)
    pesos = pesos / pesos.sum()

    b_hat = np.zeros(n_it)
    historico = []

    for it_num in range(max_iter):
        logit = nos[None, None, :] - b_hat[None, :, None]
        logp1 = -np.logaddexp(0.0, -logit)
        logp0 = -np.logaddexp(0.0, logit)

        contrib = obs[:, :, None] * (R[:, :, None] * logp1 + (1.0 - R)[:, :, None] * logp0)
        loglik_ann = contrib.sum(axis=1)

        m = loglik_ann.max(axis=1, keepdims=True)
        naolog = np.exp(loglik_ann - m) * pesos[None, :]
        marg = naolog.sum(axis=1, keepdims=True)
        post = naolog / marg
        loglik = float(np.sum(np.log(marg) + m))
        historico.append(loglik)

        n_kj = np.einsum('ik,ij->jk', post, obs.astype(float))
        r_kj = np.einsum('ik,ij->jk', post, R * obs)

        for j in range(n_it):
            def neg(par):
                bj = par[0]
                lg = nos - bj
                lp1 = -np.logaddexp(0.0, -lg)
                lp0 = -np.logaddexp(0.0, lg)
                return -np.sum(r_kj[j] * lp1 + (n_kj[j] - r_kj[j]) * lp0)
            res = minimize(neg, x0=[b_hat[j]], method='L-BFGS-B',
                           bounds=[(-10.0, 10.0)])
            b_hat[j] = res.x[0]

        if it_num > 0 and abs(historico[-1] - historico[-2]) < tol:
            if verbose:
                print(f"convergiu em {it_num+1} iteracoes")
            break

    quebras = sum(1 for i in range(1, len(historico)) if historico[i] < historico[i-1] - 1e-6)
    if verbose:
        print(f"quebras de monotonicidade: {quebras} / {len(historico)-1}")
        print(f"itens no bordo: {int(np.sum(np.abs(b_hat) >= 9.9))} / {n_it}")

    theta_eap = post @ nos
    return theta_eap, b_hat, historico


res_1pl = {}
for seed in [42, 7, 2025]:
    r_, tt, bt, at = gerar_2pl(n_itens=1000, n_anotadores=220,
                               votos_por_item=6, seed=seed)
    th, bh, _ = estimar_1pl_mml(r_, verbose=False)
    r_th, _ = pearsonr(tt, th)
    r_b, _  = pearsonr(bt, bh)
    res_1pl[seed] = (r_th, r_b)
    print(f"seed {seed:4d}  theta={r_th:+.4f}  b={r_b:+.4f}")

m1 = np.array(list(res_1pl.values())).mean(axis=0)
print(f"\nMEDIA 1PL  theta={m1[0]:+.4f}  b={m1[1]:+.4f}")
print(f"MEDIA 2PL  theta=+0.8772  b=+0.6019")
print(f"delta      theta={m1[0]-0.8772:+.4f}  b={m1[1]-0.6019:+.4f}")
print("\n--- decisao pre-registada (tolerancia -0.01) ---")
print(f"theta: {'OK' if m1[0] >= 0.8772 - 0.01 else 'PIOROU'}")
print(f"b:     {'OK' if m1[1] >= 0.6019 - 0.01 else 'PIOROU'}")

def gerar_1pl_com_bloco(n_itens, n_anotadores, votos_por_item,
                        k_bloco, delta, seed):
    """
    Gera respostas 1PL onde os primeiros k_bloco anotadores veem
    todos os itens com dificuldade deslocada por delta.
    Competencia verdadeira identica nos dois grupos.
    Devolve (respostas, theta, b, is_bloco).
    """
    rng = np.random.default_rng(seed)

    theta = rng.normal(0.0, 1.0, size=n_anotadores)
    b     = rng.normal(0.0, 1.0, size=n_itens)

    is_bloco = np.zeros(n_anotadores, dtype=bool)
    is_bloco[:k_bloco] = True

    respostas = np.full((n_anotadores, n_itens), np.nan)

    for j in range(n_itens):
        quem = rng.choice(n_anotadores, size=votos_por_item, replace=False)
        b_efetivo = np.where(is_bloco[quem], b[j] + delta, b[j])
        p = 1.0 / (1.0 + np.exp(-(theta[quem] - b_efetivo)))
        respostas[quem, j] = (rng.random(votos_por_item) < p).astype(float)

    return respostas, theta, b, is_bloco


linhas = []
for k in [5, 20, 60, 120]:
    for delta in [0.0, 1.0, 2.0]:
        gaps, r_ths, r_bs = [], [], []
        for seed in [42, 7, 2025]:
            r_, tt, bt, blk = gerar_1pl_com_bloco(
                n_itens=1000, n_anotadores=220, votos_por_item=6,
                k_bloco=k, delta=delta, seed=seed)
            th, bh, _ = estimar_1pl_mml(r_, verbose=False)
            gap = th[blk].mean() - th[~blk].mean()
            gaps.append(gap)
            r_ths.append(pearsonr(tt, th)[0])
            r_bs.append(pearsonr(bt, bh)[0])
        linhas.append((k, delta, np.mean(gaps), np.std(gaps),
                       np.mean(r_ths), np.mean(r_bs)))
        print(f"k={k:3d} ({100*k/220:4.1f}%)  delta={delta:.1f}  |  "
              f"gap={np.mean(gaps):+.4f} (sd={np.std(gaps):.4f})  |  "
              f"theta={np.mean(r_ths):+.3f}  b={np.mean(r_bs):+.3f}")

print("\n--- criterio pre-registado ---")
ctrl = [abs(g) for k, d, g, s, rt, rb in linhas if d == 0.0]
print(f"CONTROLO delta=0: max |gap| = {max(ctrl):.4f}  "
      f"{'OK' if max(ctrl) < 0.10 else 'FALHA — bug'}")

trat = [abs(g) for k, d, g, s, rt, rb in linhas if d > 0.0]
mx = max(trat)
if mx < 0.20:
    ver = "RESISTE"
elif mx >= 0.50:
    ver = "CAPTURA"
else:
    ver = "PARCIAL"
print(f"TRATAMENTO delta>0: max |gap| = {mx:.4f}  ->  {ver}")

gaps_k5 = []
for seed in range(1000, 1030):
    r_, tt, bt, blk = gerar_1pl_com_bloco(
        n_itens=1000, n_anotadores=220, votos_por_item=6,
        k_bloco=5, delta=0.0, seed=seed)
    th, bh, _ = estimar_1pl_mml(r_, verbose=False)
    gaps_k5.append(th[blk].mean() - th[~blk].mean())

g = np.array(gaps_k5)
ep = g.std(ddof=1) / np.sqrt(len(g))
print(f"k=5, delta=0, 30 seeds")
print(f"  gap medio  = {g.mean():+.4f}")
print(f"  sd         = {g.std(ddof=1):.4f}")
print(f"  erro-padrao= {ep:.4f}")
print(f"  IC95       = [{g.mean()-1.96*ep:+.4f}, {g.mean()+1.96*ep:+.4f}]")
print(f"  zero no IC: {abs(g.mean()) < 1.96*ep}")

import numpy as np, json
from collections import defaultdict

MAP = {"YES": 1.0, "NO": 0.0}

def construir_matriz(items, lang):
    sel = [it for it in items if it.get("lang") == lang]
    anns = sorted({a for it in sel for a in it["annotators"]})
    a_idx = {a: i for i, a in enumerate(anns)}
    ids = [it["id_EXIST"] for it in sel]

    M = np.full((len(anns), len(sel)), np.nan)
    for j, it in enumerate(sel):
        for a, lab in zip(it["annotators"], it["labels_task2_1"]):
            v = MAP.get(str(lab).strip().upper())
            if v is not None:
                M[a_idx[a], j] = v
    return M, anns, ids

for lang in ["es", "en"]:
    M, anns, ids = construir_matriz(items, lang)
    n_it = np.sum(~np.isnan(M), axis=0)
    n_an = np.sum(~np.isnan(M), axis=1)
    print(f"--- painel {lang} ---")
    print(f"  itens={M.shape[1]}  anotadores={M.shape[0]}  votos={int(np.sum(~np.isnan(M)))}")
    print(f"  votos/item:     min={n_it.min()} mediana={np.median(n_it):.0f} max={n_it.max()}")
    print(f"  votos/anotador: min={n_an.min()} mediana={np.median(n_an):.0f} max={n_an.max()}")
    print(f"  itens unanimes: {int(np.sum((np.nanmean(M,axis=0)==0)|(np.nanmean(M,axis=0)==1)))}")
    print(f"  taxa global YES: {np.nanmean(M):.3f}")
    globals()[f"M_{lang}"] = M
    globals()[f"anns_{lang}"] = anns
    globals()[f"ids_{lang}"] = ids

res_reais = {}
for lang in ["es", "en"]:
    M = globals()[f"M_{lang}"]
    th, bh, hist = estimar_1pl_mml(M, verbose=False)

    quebras = sum(1 for i in range(1, len(hist)) if hist[i] < hist[i-1] - 1e-6)
    yes = np.nanmean(M, axis=0)
    unanime = (yes == 0.0) | (yes == 1.0)
    saturado = np.abs(bh) >= 9.9

    print(f"--- painel {lang} ---")
    print(f"  iteracoes={len(hist)}  quebras de monotonicidade={quebras}")
    print(f"  itens unanimes={int(unanime.sum())}  itens saturados (|b|>=9.9)={int(saturado.sum())}")
    print(f"  sobreposicao unanime/saturado: {int(np.sum(unanime & saturado))}")

    ok = ~unanime
    eps = 1e-6
    logit_yes = np.log((yes[ok] + eps) / (1 - yes[ok] + eps))
    r_b, _ = pearsonr(bh[ok], logit_yes)
    rho_b, _ = spearmanr(bh[ok], logit_yes)

    print(f"  itens nao unanimes: {int(ok.sum())}")
    print(f"  b vs logit(yes_rate):  Pearson={r_b:+.4f}  Spearman={rho_b:+.4f}")
    print(f"  b (nao unanimes): media={bh[ok].mean():+.3f} sd={bh[ok].std():.3f} "
          f"min={bh[ok].min():+.3f} max={bh[ok].max():+.3f}")
    print(f"  theta: media={th.mean():+.3f} sd={th.std():.3f}")

    res_reais[lang] = dict(theta=th, b=bh, unanime=unanime, ok=ok)

from collections import Counter

vocab_b = Counter()
votos_b = Counter()
votos_a = Counter()

for it in items:
    for a, la, lb in zip(it["annotators"], it["labels_task2_1"], it["labels_task2_2"]):
        votos_a[a] += 1
        vocab_b[str(lb).strip()] += 1
        if str(la).strip().upper() == "YES":
            votos_b[a] += 1

print("vocabulario de labels_task2_2:")
for k, n in vocab_b.most_common():
    print(f"  {k!r:20s} {n}")

nb = np.array([votos_b.get(a, 0) for a in votos_a])
na = np.array([votos_a[a] for a in votos_a])
print(f"\nanotadores: {len(na)}")
print(f"votos em A: mediana={np.median(na):.0f}")
print(f"votos em B: mediana={np.median(nb):.0f}  min={nb.min()}  max={nb.max()}")
print(f"  com <9 votos em B (menos que parametros): {int(np.sum(nb < 9))} ({100*np.mean(nb<9):.1f}%)")
print(f"  com <18 votos em B (menos de 2 por parametro): {int(np.sum(nb < 18))} ({100*np.mean(nb<18):.1f}%)")
print(f"observacoes por parametro em B: mediana={np.median(nb)/9:.2f}")
print(f"observacoes por parametro em A: mediana={np.median(na)/4:.2f}")