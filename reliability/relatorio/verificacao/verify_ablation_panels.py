"""Densidade por item e concentração por painel dos anotadores removidos em R1 (extremos) e R2 (controlo).
Uso: python3 verify_ablation_panels.py <raiz do repositório>"""
import json, sys, pathlib
import pandas as pd
sys.argv = [sys.argv[0], sys.argv[1]]
src = (pathlib.Path(__file__).resolve().parent.parent / "plot_relatorio.py").read_text(encoding="utf-8")
exec(src.replace('if __name__ == "__main__":', "if False:"))          # reutiliza DATA e controle42()
ds = pd.DataFrame(list(json.load(open(DATA, encoding="utf-8")).values()))
v = ds[["id_EXIST", "lang", "annotators", "labels_task2_1"]].explode(["annotators", "labels_task2_1"])
v["yes"] = (v.labels_task2_1 == "YES").astype(int)
perfil = v.groupby("annotators").agg(n_votes=("yes", "size"), yes_rate=("yes", "mean"))
perfil["extremo"] = (perfil.yes_rate > 0.9) | (perfil.yes_rate < 0.1)
perfil["lang"] = v.groupby("annotators")["lang"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "ambas")
pan = ds.annotators.apply(frozenset)
for nome, S in (("R1 extremos", set(perfil.index[perfil.extremo])), ("R2 controlo", set(controle42(perfil)))):
    por_painel = pd.Series({p: len(p & S) for p in set(pan)})
    n = v[~v.annotators.isin(S)].groupby("id_EXIST").size()
    print(f"{nome}: painéis com ≥1 = {(por_painel > 0).sum()}, por painel {por_painel[por_painel > 0].value_counts().sort_index().to_dict()}, "
          f"itens com ≥1 voto removido = {(pan.apply(lambda p: len(p & S)) > 0).sum()}, votos/item {n.value_counts().sort_index().to_dict()}")
