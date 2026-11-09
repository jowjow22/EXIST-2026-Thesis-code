"""Reexecução das células de EXIST_MEMES/Untitled.ipynb (0, 1, 3–8, 12) com o caminho do dataset como argumento (verificação de 2026-10-24).
Uso: python3 verify_panel.py <caminho para EXIST2026_training.json>"""
import sys
# Passo 1 — inventario demografico dos anotadores
import json, pathlib
import pandas as pd
import os


TRAIN_JSON = sys.argv[1]   # AJUSTE para o caminho real

raw = json.loads(pathlib.Path(TRAIN_JSON).read_text(encoding="utf-8"))
items = list(raw.values()) if isinstance(raw, dict) else list(raw)
first = items[0]

print("chaves de um item:")
for k, v in first.items():
    amostra = v[:3] if isinstance(v, list) else v
    print(f"  {k:34s} {type(v).__name__:5s}  {str(amostra)[:56]}")

ANNOT_KEY = next(k for k in first if "annotator" in k.lower() and isinstance(first[k], list))
n0 = len(first[ANNOT_KEY])

PADROES = {
    "genero": ["gender", "sex"],
    "idade":  ["age"],
    "etnia":  ["ethnic"],
    "estudo": ["study", "education"],
    "pais":   ["countr"],
}
demo_keys = {}
for k, v in first.items():
    if k == ANNOT_KEY or not isinstance(v, list) or len(v) != n0:
        continue
    kl = k.lower()
    for nome, pads in PADROES.items():
        if any(p in kl for p in pads) and nome not in demo_keys:
            demo_keys[nome] = k

print("\ncolunas demograficas detectadas:")
for nome, k in demo_keys.items():
    print(f"  {nome:8s} <- {k}")
faltando = set(PADROES) - set(demo_keys)
if faltando:
    print(f"  NAO ENCONTRADAS: {sorted(faltando)}  (me mande a lista de chaves acima)")

linhas = []
for it in items:
    ids = it[ANNOT_KEY]
    for i, a in enumerate(ids):
        linhas.append({"anotador": a, **{nome: it[k][i] for nome, k in demo_keys.items()}})
longo = pd.DataFrame(linhas)

vpa = longo.anotador.value_counts()
print(f"\nitens: {len(items):,}")
print(f"votos totais: {len(longo):,}")
print(f"anotadores distintos: {longo.anotador.nunique():,}")
print(f"votos por item: media {len(longo)/len(items):.2f}")
print(f"votos por anotador: mediana {vpa.median():.0f} | min {vpa.min()} | max {vpa.max()}")

perfis = longo.drop_duplicates()
dup = perfis.anotador.value_counts()
print(f"anotadores com metadados inconsistentes entre itens: {(dup > 1).sum()}")

ann = perfis.drop_duplicates("anotador")
for nome in demo_keys:
    vc = ann[nome].value_counts(dropna=False)
    print(f"\n{nome}  —  {len(vc)} valores distintos")
    for val, c in vc.head(15).items():
        print(f"   {str(val)[:40]:42s} {c:5d}  ({100*c/len(ann):4.1f}%)")
    if len(vc) > 15:
        print(f"   ... mais {len(vc)-15} valores")

# Passo 3 (autossuficiente) — metricas de desvio por anotador, N=6 vs controlo N=1
import numpy as np, pandas as pd

ANN_K   = "annotators"
DEMO_K  = {"genero": "gender_annotators", "idade": "age_annotators",
           "etnia": "ethnicities_annotators", "estudo": "study_levels_annotators",
           "pais": "countries_annotators"}
LAB_K   = "labels_task2_1"

# --- perfil demografico, uma linha por anotador (moda resolve os 2 inconsistentes)
pl = []
for it in items:
    for i, a in enumerate(it[ANN_K]):
        pl.append({"anotador": a, **{n: it[k][i] for n, k in DEMO_K.items()}})
perfil = (pd.DataFrame(pl).groupby("anotador")[list(DEMO_K)]
            .agg(lambda s: s.mode().iat[0]).reset_index())
print(f"perfis: {len(perfil)}  |  cardinalidades: "
      f"{ {c: perfil[c].nunique() for c in DEMO_K} }")

# --- votos
recs = []
for it in items:
    if LAB_K not in it:
        continue
    for a, l in zip(it[ANN_K], it[LAB_K]):
        recs.append({"item": it["id_EXIST"], "anotador": a, "lab": str(l).strip().upper()})
v = pd.DataFrame(recs)
print("\nrotulos brutos:"); print(v.lab.value_counts().to_string())
v["y"] = v.lab.map({"YES": 1, "NO": 0})
print(f"votos descartados (nao YES/NO): {int(v.y.isna().sum())}")
v = v.dropna(subset=["y"])

# --- desvio com o item controlado: cada voto contra os 5 colegas do mesmo meme
g = v.groupby("item")["y"].agg(["sum", "count"])
v = v.join(g, on="item")
v = v[v["count"] > 1].copy()
v["pares"] = (v["sum"] - v.y) / (v["count"] - 1)
v["desac"] = (v.y - v.pares).abs()
v["resid"] = v.y - v.pares

ann = (v.groupby("anotador")
         .agg(n=("y", "size"), yes=("y", "mean"),
              desac=("desac", "mean"), resid=("resid", "mean"))
         .reset_index()
         .merge(perfil, on="anotador", how="left"))
assert ann[list(DEMO_K)].notna().all().all(), "anotador sem perfil"
print(f"\nanotadores: {len(ann)}  |  votos usados: {len(v):,}")

# --- quanto do espalhamento e real e quanto e so amostragem
rng = np.random.default_rng(42)
nulo = [v.assign(anotador=rng.permutation(v.anotador.values))
         .groupby("anotador")["resid"].mean().std() for _ in range(200)]
sd_nulo = float(np.mean(nulo))
print(f"\nsd de resid sob o nulo : {sd_nulo:.4f}")
print(f"sd observado           : {ann.resid.std():.4f}")
print(f"razao obs/nulo         : {ann.resid.std()/sd_nulo:.2f}   (1.0 = nada a medir)")

# --- esquemas
ESQ = {"N1": None, "N6": ["genero", "idade"]}
flag = {}
for nome, cols in ESQ.items():
    key = pd.Series("todos", index=ann.index) if cols is None \
          else ann[cols].astype(str).agg(" | ".join, axis=1)
    ann[f"cl_{nome}"] = key
    tam = key.value_counts()
    print(f"\n{nome}: {len(tam)} clusters | menor {tam.min()} | "
          f"mediana {int(tam.median())} | maior {tam.max()}")
    for m in ("yes", "resid", "desac"):
        mu = ann.groupby(key)[m].transform("mean")
        sd = ann.groupby(key)[m].transform("std").replace(0, np.nan)
        ann[f"z_{m}_{nome}"] = (ann[m] - mu) / sd
    flag[nome] = set(ann.nlargest(61, f"z_resid_{nome}").anotador) | \
                 set(ann.nsmallest(61, f"z_resid_{nome}").anotador)

inter, uni = len(flag["N1"] & flag["N6"]), len(flag["N1"] | flag["N6"])
print(f"\nsobreposicao N1 vs N6: {inter}/{uni} = {100*inter/uni:.0f}%")
print("\ncorrelacao entre metricas:")
print(ann[["yes", "resid", "desac"]].corr().round(3).to_string())
print("\nmedia de resid por cluster (N6):")
print(ann.groupby("cl_N6")["resid"].agg(["size", "mean", "std"]).round(3).to_string())

# Passo 4 (corrigido) — decomposicao de variancia e estabilidade interna
import numpy as np, pandas as pd

# --- 1. quanto da variacao e real
var_obs   = float(ann.resid.var(ddof=1))
var_ruido = float(sd_nulo ** 2)
var_real  = max(var_obs - var_ruido, 0.0)
print(f"variancia observada  : {var_obs:.5f}")
print(f"  amostragem         : {var_ruido:.5f}  ({100*var_ruido/var_obs:4.1f}%)")
print(f"  real entre pessoas : {var_real:.5f}  ({100*var_real/var_obs:4.1f}%)")

# --- 2. quanto disso a demografia explica
mu  = float(ann.resid.mean())
btw = ann.groupby("cl_N6").resid.agg(["size", "mean"])
var_entre = float(np.average((btw["mean"] - mu) ** 2, weights=btw["size"]))
sd_dentro = float(ann.groupby("cl_N6").resid.transform(lambda s: s - s.mean()).std())
print(f"\nvariancia entre clusters N6 : {var_entre:.5f}")
print(f"  {100*var_entre/var_obs:4.1f}% da observada | {100*var_entre/var_real:4.1f}% da real")
print(f"amplitude das 6 medias      : {btw['mean'].max() - btw['mean'].min():.3f}")
print(f"sd individual apos centrar  : {sd_dentro:.3f}")

# --- 3. o desvio de uma metade preve o da outra?
def metades(df, col="resid", n_rep=25, min_votos=12, seed0=300):
    r_obs, r_nul, diffs, n_ult = [], [], [], 0
    for s in range(n_rep):
        rng_s = np.random.default_rng(seed0 + s)
        vv = df.sample(frac=1, random_state=seed0 + s).copy()
        vv["h"] = vv.groupby("anotador").cumcount() % 2
        m = vv.pivot_table(index="anotador", columns="h", values=col, aggfunc="mean")
        cnt = vv.groupby("anotador").size()
        m = m.loc[cnt[cnt >= min_votos].index].dropna()
        assert m.shape[1] == 2, f"pivot devolveu {m.shape[1]} colunas"
        r_obs.append(m.iloc[:, 0].corr(m.iloc[:, 1]))
        diffs.append((m.iloc[:, 0] - m.iloc[:, 1]).abs())
        n_ult = len(m)

        pp = vv.assign(anotador=rng_s.permutation(vv.anotador.values))
        mp = pp.pivot_table(index="anotador", columns="h", values=col, aggfunc="mean").dropna()
        r_nul.append(mp.iloc[:, 0].corr(mp.iloc[:, 1]))
    instab = pd.concat(diffs, axis=1).mean(axis=1)
    return float(np.mean(r_obs)), float(np.mean(r_nul)), instab, n_ult

r_obs, r_nul, instab, n_usados = metades(v)
sb = 2 * r_obs / (1 + r_obs) if r_obs > -1 else float("nan")
print(f"\nestabilidade entre metades ({n_usados} anotadores com >= 12 votos):")
print(f"  observada            : {r_obs:.3f}")
print(f"  sob o nulo           : {r_nul:.3f}")
print(f"  Spearman-Brown       : {sb:.3f}")

# --- 4. instabilidade por pessoa, anexada sem merge
assert isinstance(instab, pd.Series) and instab.index.name == "anotador"
ann["instab"] = ann["anotador"].map(instab)
print(f"\ninstab preenchida para {ann.instab.notna().sum()} de {len(ann)} anotadores")
print(ann.instab.describe().round(3).to_string())

print("\ncorrelacoes:")
print(ann[["instab", "resid", "yes", "n"]].assign(abs_resid=ann.resid.abs())
        .corr().round(3).to_string())

# Passo 5 — a instabilidade e um traco real ou so ruido de amostragem?
import numpy as np, pandas as pd

# --- A. instab tem confiabilidade propria? (quartos em vez de metades)
def instab_de(df, col="resid", seed=0):
    vv = df.sample(frac=1, random_state=seed).copy()
    vv["h"] = vv.groupby("anotador").cumcount() % 2
    m = vv.pivot_table(index="anotador", columns="h", values=col, aggfunc="mean").dropna()
    return (m.iloc[:, 0] - m.iloc[:, 1]).abs()

rs = []
for s in range(25):
    vv = v.sample(frac=1, random_state=700 + s).copy()
    vv["q"] = vv.groupby("anotador").cumcount() % 4
    a = instab_de(vv[vv.q.isin([0, 1])], seed=s)
    b = instab_de(vv[vv.q.isin([2, 3])], seed=s)
    j = a.index.intersection(b.index)
    rs.append(a[j].corr(b[j]))
r_i = float(np.mean(rs))
print(f"confiabilidade da instab (metades independentes): {r_i:.3f}")
print(f"  Spearman-Brown: {2*r_i/(1+r_i):.3f}   (perto de 0 = nao ha traco a medir)")

# --- B. simular um mundo sem ninguem eratico e comparar
sh = ann.set_index("anotador").resid
sim_stats = []
for s in range(20):
    rng_s = np.random.default_rng(1000 + s)
    p = (v.pares + v.anotador.map(sh)).clip(0.02, 0.98)
    vs = v.assign(y=rng_s.binomial(1, p))
    vs["resid"] = vs.y - vs.pares
    i_sim = instab_de(vs, seed=s)
    a_sim = vs.groupby("anotador")["resid"].mean()
    sim_stats.append({"instab_med": i_sim.mean(), "instab_sd": i_sim.std(),
                      "sd_resid": a_sim.std()})
sim = pd.DataFrame(sim_stats)

print("\n                    observado   simulado (regua fixa)")
print(f"media da instab   :   {ann.instab.mean():.3f}        {sim.instab_med.mean():.3f}")
print(f"sd da instab      :   {ann.instab.std():.3f}        {sim.instab_sd.mean():.3f}")
print(f"sd do resid       :   {ann.resid.std():.3f}        {sim.sd_resid.mean():.3f}")

# --- C. sobra alguem genuinamente eratico?
lim = sim.instab_med.mean() + 2.5 * sim.instab_sd.mean()
extremos = ann[ann.instab > lim]
print(f"\nlimiar (simulado + 2.5 sd): {lim:.3f}")
print(f"anotadores acima: {len(extremos)} de {ann.instab.notna().sum()}")
if len(extremos):
    print(extremos[["anotador", "n", "yes", "resid", "instab"]]
          .sort_values("instab", ascending=False).head(10).to_string(index=False))

# Passo 6 — vies de painel, estimado sem circularidade
import numpy as np, pandas as pd

REL = 0.850   # confiabilidade medida no passo 4 (Spearman-Brown)

it = v.groupby("item")["y"].agg(m="mean", k="size")
v6 = v.join(it, on="item")
v6["d"] = v6.y - v6.m                       # desvio face a media do item

tot = v6.groupby("anotador")["d"].agg(s="sum", c="size")
v6 = v6.join(tot, on="anotador")
v6 = v6[v6.c > 1].copy()
v6["theta_loo"] = (v6.s - v6.d) / (v6.c - 1)   # limiar sem o item atual

pan = v6.groupby("item").agg(bias=("theta_loo", "mean"),
                             obs=("y", "mean"), k=("y", "size"))
pan["bias_enc"] = pan.bias * REL
pan["corr"] = (pan.obs - pan.bias_enc).clip(0, 1)

print(f"itens: {len(pan):,}")
print(f"sd do vies de painel : {pan.bias.std():.4f}")
print(f"sd apos encolhimento : {pan.bias_enc.std():.4f}")
print(f"em votos (de 6)      : {pan.bias_enc.std()*6:.2f}")
print(f"amplitude do vies    : {pan.bias_enc.min():+.3f} a {pan.bias_enc.max():+.3f}")

print(f"\ncorrelacao vies x taxa observada: {pan.bias.corr(pan.obs):+.3f}")
d = (pan['corr'] - pan.obs).abs()
print(f"deslocamento medio do rotulo    : {d.mean():.4f}")
print(f"itens que mudam >= 1/6 (um voto): {(d >= 1/6).sum()} ({100*(d>=1/6).mean():.1f}%)")
print(f"itens que mudam >= 1/12         : {(d >= 1/12).sum()} ({100*(d>=1/12).mean():.1f}%)")

rng = np.random.default_rng(11)
nul = [v6.assign(anotador=rng.permutation(v6.anotador.values))
          .pipe(lambda x: x.join(x.groupby("anotador")["d"].agg(s2="sum", c2="size"),
                                 on="anotador"))
          .assign(t=lambda x: (x.s2 - x.d) / (x.c2 - 1).clip(lower=1))
          .groupby("item")["t"].mean().std() for _ in range(50)]
print(f"\nsd do vies sob o nulo: {np.mean(nul):.4f}")
print(f"razao obs/nulo       : {pan.bias.std()/np.mean(nul):.2f}")

print("\nitens com maior vies de painel:")
print(pan.reindex(pan.bias_enc.abs().sort_values(ascending=False).index)
         .head(8)[["obs", "bias_enc", "corr"]].round(3).to_string())

# Passo 7 — a estrutura de paineis: blocos fechados ou grafo ligado?
import numpy as np, pandas as pd
from collections import defaultdict

pain = v.groupby("item")["anotador"].apply(frozenset)
print(f"itens: {len(pain):,} | paineis distintos: {pain.nunique():,}")
print(f"itens por painel: {(len(pain)/pain.nunique()):.1f}")
print("\ntamanho dos paineis:"); print(pain.map(len).value_counts().to_string())

por_ann = defaultdict(set)
for i, p in pain.items():
    for a in p:
        por_ann[a].add(p)
n_pain = pd.Series({a: len(s) for a, s in por_ann.items()})
print(f"\npaineis por anotador: mediana {n_pain.median():.0f} | "
      f"min {n_pain.min()} | max {n_pain.max()}")
print(f"anotadores num painel unico: {(n_pain == 1).sum()} de {len(n_pain)}")

parent = {a: a for a in por_ann}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for p in pain.unique():
    it_ = iter(p); r = find(next(it_))
    for a in it_:
        parent[find(a)] = r
comp = pd.Series({a: find(a) for a in parent}).value_counts()
print(f"\ncomponentes ligadas no grafo de co-anotacao: {len(comp)}")
print(f"maior componente: {comp.iloc[0]} anotadores ({100*comp.iloc[0]/len(parent):.1f}%)")
print("tamanhos (top 10):", comp.head(10).tolist())

# soma dos limiares dentro do painel: zero por construcao?
th = v6.groupby("anotador")["theta_loo"].mean()
comp_id = pd.Series({a: find(a) for a in parent})
soma = th.groupby(comp_id).sum()
print(f"\nsoma dos limiares por componente: media {soma.mean():+.4f} | "
      f"max |.| {soma.abs().max():.4f}")

# quantos colegas distintos cada pessoa encontrou
cols = {a: set().union(*s) - {a} for a, s in por_ann.items()}
nc = pd.Series({a: len(c) for a, c in cols.items()})
print(f"\ncolegas distintos por anotador: mediana {nc.median():.0f} | "
      f"min {nc.min()} | max {nc.max()}")
print(f"densidade do grafo: {100*nc.mean()/(len(nc)-1):.2f}% dos pares possiveis")

# Passo 8 — composicao dos paineis e o que isso faz ao demo_vec
import numpy as np, pandas as pd

pid = pd.Series(pd.factorize(pain.astype(str))[0], index=pain.index, name="painel")
v8 = v.join(pid, on="item")
memb = (v8[["painel", "anotador"]].drop_duplicates()
          .merge(perfil, on="anotador", how="left"))
memb["ga"] = memb.genero + " | " + memb.idade

comp = memb.groupby(["painel", "ga"]).size().unstack(fill_value=0)
print(f"paineis: {len(comp)} | celulas genero x idade: {comp.shape[1]}")
print(f"paineis com exatamente 1 de cada celula: {(comp == 1).all(axis=1).sum()} de {len(comp)}")
print("\ndistribuicao das contagens por celula:")
print(pd.Series(comp.values.ravel()).value_counts().sort_index().to_string())

print("\nvariacao da composicao ENTRE paineis (sd da contagem):")
for atr in ["genero", "idade", "etnia", "estudo", "pais"]:
    c = memb.groupby(["painel", atr]).size().unstack(fill_value=0)
    print(f"  {atr:7s} sd medio {c.std().mean():.3f} | "
          f"valores {c.shape[1]:3d} | celulas sempre iguais: {(c.std() < 1e-9).all()}")

# efeito de celula com intervalo, reamostrando paineis inteiros
res = v8.groupby("anotador").resid.mean().rename("r")
base = memb.join(res, on="anotador").dropna(subset=["r"])
pontual = base.groupby("ga").r.mean()
rng = np.random.default_rng(7)
pans = base.painel.unique()
bs = []
for _ in range(2000):
    s = rng.choice(pans, len(pans), replace=True)
    bs.append(base[base.painel.isin(s)].groupby("ga").r.mean())
B = pd.DataFrame(bs)
out = pd.DataFrame({"efeito": pontual,
                    "ic2.5": B.quantile(.025), "ic97.5": B.quantile(.975)}).round(3)
print("\nefeito de celula (contraste dentro do painel, IC por reamostragem de paineis):")
print(out.sort_values("efeito").to_string())

# Passo 9 — o efeito de celula resiste aos atributos nao equilibrados?
import numpy as np, pandas as pd

r_ann = v8.groupby("anotador").resid.mean().rename("r")
b = memb.join(r_ann, on="anotador").dropna(subset=["r"]).copy()
b["etnia3"] = np.where(b.etnia.isin(["White or Caucasian", "Hispano or Latino"]),
                       b.etnia, "Outros")
ORD = ["M | 18-22", "M | 23-45", "F | 18-22", "F | 23-45", "M | 46+", "F | 46+"]

print("as celulas tem composicao parecida nos atributos nao equilibrados?")
for atr in ["etnia3", "estudo"]:
    print(f"\n{atr} (% dentro de cada celula):")
    print((100 * pd.crosstab(b.ga, b[atr], normalize="index"))
          .reindex(ORD).round(1).to_string())

print("\nefeito bruto vs efeito apos remover etnia+escolaridade:")
aj = b.copy()
for atr in ["etnia3", "estudo"]:
    aj["r"] = aj.r - aj.groupby(atr).r.transform("mean")
cmp = pd.DataFrame({"bruto": b.groupby("ga").r.mean(),
                    "ajustado": aj.groupby("ga").r.mean()}).reindex(ORD)
cmp["delta"] = (cmp.ajustado - cmp.bruto).abs()
print(cmp.round(3).to_string())

print("\nconsistencia dentro de cada painel (sem modelo nenhum):")
b["rank"] = b.groupby("painel").r.rank()
print(b.groupby("ga")["rank"].mean().reindex(ORD).round(2).to_string())
piores = b.loc[b.groupby("painel").r.idxmin()].ga.value_counts()
print(f"\nquantas vezes cada celula foi a MENOR do seu painel (de {b.painel.nunique()}):")
print(piores.reindex(ORD).fillna(0).astype(int).to_string())

lang_p = v8.merge(pd.DataFrame({"item": [i["id_EXIST"] for i in items],
                                "lang": [i["lang"] for i in items]}), on="item")
pl = lang_p.groupby("painel").lang.agg(["nunique", lambda s: s.mode().iat[0]])
pl.columns = ["n_langs", "lang"]
print(f"\npaineis com uma so lingua: {(pl.n_langs == 1).sum()} de {len(pl)}")
b = b.merge(pl["lang"], left_on="painel", right_index=True, how="left")
print("\nefeito por lingua:")
print(b.pivot_table(index="ga", columns="lang", values="r", aggfunc="mean")
       .reindex(ORD).round(3).to_string())

# Perfis: existe bimodalidade, e a composicao varia entre paineis?
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture

r = ann.resid.values.reshape(-1, 1)
for k in (1, 2, 3):
    g = GaussianMixture(k, random_state=0, n_init=10).fit(r)
    print(f"  {k} componente(s): BIC {g.bic(r):9.1f}")
g2 = GaussianMixture(2, random_state=0, n_init=10).fit(r)
print(f"\nmedias dos 2 componentes: {np.sort(g2.means_.ravel()).round(3)}")
print(f"pesos: {g2.weights_[np.argsort(g2.means_.ravel())].round(3)}")
print("(1 componente com menor BIC = distribuicao unimodal, sem dois perfis)")

print("\nvariacao da composicao de perfil ENTRE paineis:")
pa = v8[["anotador", "painel"]].drop_duplicates().merge(
        ann[["anotador", "resid"]], on="anotador", how="left")
g = pa.groupby("painel").resid.agg(["mean", "size"])
vb = float(np.average((g["mean"] - pa.resid.mean())**2, weights=g["size"]))
print(f"  variancia entre paineis : {vb:.6f}")
print(f"  variancia entre pessoas : {pa.resid.var(ddof=1):.6f}")
print(f"  razao                   : {vb/pa.resid.var(ddof=1):.4f}")
print("  (perto de 0 = todo painel tem a mesma mistura de perfis)")

yr = v.groupby("anotador").y.mean().rename("bruta")
pb = pa.merge(yr, on="anotador", how="left")
gb = pb.groupby("painel").bruta.agg(["mean", "size"])
print(f"\ncom taxa bruta de sim (confundida com o bloco de memes):")
print(f"  variancia entre paineis : {float(np.average((gb['mean'] - pb.bruta.mean())**2, weights=gb['size'])):.6f}")