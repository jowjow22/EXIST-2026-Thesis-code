"""Testes de sanidade da camada de confusão. Correm em CPU, sem transformers.

    python3 confusion_selftest.py

O teste central é o T2: com C = identidade exacta, `L_aux` tem de ser
NUMERICAMENTE IGUAL ao BCE da Task A contra a soft label crua (a soft label crua
é a frequência dos votos, e todos os itens têm 6 votos). Se a polaridade de
`head_a` estiver invertida, este teste falha -- e o controlo lambda=0 NÃO
apanharia essa inversão.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

from confusion_core import (ConfusionAux, ConfusionMultiHeadModel, VOTE_MAP,
                            build_vote_table, IDENT_LOGIT, build_annotator_strata,
                            build_annotator_stats, build_permutation,
                            corrupt_vote_table, make_vote_metric_logger_class,
                            fit_logistic, STRATUM_NAMES, CORRUPTION_LEVELS,
                            soft_labels_from_votes)

DATA = "./Dataset/memes/training/EXIST2026_training.json"
SOFT_RAW = "../reliability/outputs/soft_labels_taskA_raw.json"
SPLIT = "../reliability/outputs/split_taskA_frozen.json"
OUT_TMP = tempfile.mkdtemp(prefix="confusion_selftest_")

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# --- T1: tabela de votos ---------------------------------------------------
table, ann_ids, stats = build_vote_table(DATA)
split = json.load(open(SPLIT))
all_ids = [str(i) for i in split["train_ids"] + split["val_ids"]]
check("T1a chaves do split cobertas pela tabela de votos",
      all(i in table for i in all_ids), f"{len(all_ids)} ids")
check("T1b 6 votos válidos em todos os itens",
      all((v >= 0).sum() == 6 for _, v in table.values()),
      f"n_itens={stats['n_items']}")
check("T1c nº de anotadores", stats["n_annotators"] == 887,
      str(stats["n_annotators"]))

# --- T2: identidade exacta == BCE contra a soft label crua ------------------
soft = json.load(open(SOFT_RAW))
ids = all_ids[:64]
votes = np.stack([table[i][1] for i in ids])
anns = np.stack([table[i][0] for i in ids])
p_no_soft = torch.tensor([soft[i][1] for i in ids], dtype=torch.float32)  # [B]

g = torch.Generator().manual_seed(0)
logits_a = torch.randn(len(ids), 1, generator=g) * 2.0

aux = ConfusionAux(stats["n_annotators"], method="M3")
aux.force_identity = True
l_aux, _ = aux(logits_a, torch.from_numpy(anns), torch.from_numpy(votes))
bce = torch.nn.functional.binary_cross_entropy_with_logits(
    logits_a.view(-1), p_no_soft)
check("T2 L_aux(identidade) == BCE(logits, P(NO) crua)",
      bool(torch.allclose(l_aux, bce, atol=1e-6)),
      f"l_aux={l_aux.item():.8f} bce={bce.item():.8f} "
      f"delta={abs(l_aux.item()-bce.item()):.2e}")

# --- T2b: polaridade invertida TEM de falhar (o teste tem poder) -----------
flipped = np.where(votes >= 0, 1 - votes, votes)
l_flip, _ = aux(logits_a, torch.from_numpy(anns), torch.from_numpy(flipped))
check("T2b polaridade invertida diverge (o T2 tem poder)",
      not bool(torch.allclose(l_flip, bce, atol=1e-3)),
      f"l_aux_flip={l_flip.item():.6f} vs bce={bce.item():.6f}")

# --- T3: matrizes bem formadas e perto da identidade no init ---------------
for m in ("M1", "M2", "M3"):
    a = ConfusionAux(stats["n_annotators"], method=m)
    C = a.confusion_matrices()
    rows_ok = bool(torch.allclose(C.sum(-1), torch.ones_like(C.sum(-1)), atol=1e-6))
    diag = C.diagonal(dim1=-2, dim2=-1)
    check(f"T3 {m}: linhas somam 1 e init perto da identidade", rows_ok,
          f"diag={diag.mean().item():.4f} (softmax([{IDENT_LOGIT},0])) "
          f"dist_id={a.identity_distance().mean():.4f}")

# --- T4: init não consome RNG ---------------------------------------------
torch.manual_seed(123)
before = torch.get_rng_state().clone()
_ = ConfusionAux(stats["n_annotators"], method="M1")
_ = ConfusionAux(stats["n_annotators"], method="M2")
check("T4 construir ConfusionAux não consome RNG",
      bool(torch.equal(before, torch.get_rng_state())))

# --- T5: gradiente flui para as matrizes e para os logits ------------------
for m in ("M1", "M2", "M3"):
    a = ConfusionAux(stats["n_annotators"], method=m)
    lg = logits_a.clone().requires_grad_(True)
    l, r = a(lg, torch.from_numpy(anns), torch.from_numpy(votes))
    total = l if r is None else l + r
    total.backward()
    touched = (a.delta.grad.abs().sum(dim=(1, 2)) > 0).sum().item()
    n_uniq = len(np.unique(anns[votes >= 0]))
    ok = lg.grad.abs().sum() > 0 and (touched == n_uniq or m == "M2")
    check(f"T5 {m}: gradiente flui", bool(ok),
          f"anotadores com grad={touched} (no batch={n_uniq}), "
          f"grad_logits={lg.grad.abs().sum().item():.4f}")

# --- T6: com algoritmos determinísticos ligados, o forward/backward corre ---
torch.use_deterministic_algorithms(True)
try:
    a = ConfusionAux(stats["n_annotators"], method="M1")
    lg = logits_a.clone().requires_grad_(True)
    l, r = a(lg, torch.from_numpy(anns), torch.from_numpy(votes))
    (l + r).backward()
    check("T6 corre sob use_deterministic_algorithms(True)", True, "CPU")
except RuntimeError as e:
    check("T6 corre sob use_deterministic_algorithms(True)", False, str(e))
finally:
    torch.use_deterministic_algorithms(False)

# --- T7: lambda=0 mantém o caminho vivo (0 * NaN = NaN) --------------------
class _FakeBase(torch.nn.Module):
    def forward(self, input_ids, attention_mask, **kw):
        lg = input_ids.float().sum(-1, keepdim=True)
        return (lg.sum() * 0.0 + 1.0, lg, lg, lg)

model = ConfusionMultiHeadModel(_FakeBase(), stats["n_annotators"], method="M3", lam=0.0)
model.train()
out = model(input_ids=torch.zeros(len(ids), 4), attention_mask=None,
            ann_idx=torch.from_numpy(anns), ann_vote=torch.from_numpy(votes))
check("T7a lambda=0: perda total == perda base", float(out[0]) == 1.0,
      f"loss={float(out[0])} aux_registada={model.last_aux:.6f}")
model.confusion.delta.data[0, 0, 0] = float("nan")
out_nan = model(input_ids=torch.zeros(len(ids), 4), attention_mask=None,
                ann_idx=torch.from_numpy(anns), ann_vote=torch.from_numpy(votes))
check("T7b lambda=0 ainda detecta NaN na auxiliar",
      bool(torch.isnan(out_nan[0])), "0 * NaN = NaN, como pretendido")

# --- T8: eval não usa a auxiliar ------------------------------------------
model2 = ConfusionMultiHeadModel(_FakeBase(), stats["n_annotators"], method="M3", lam=1.0)
model2.eval()
out_eval = model2(input_ids=torch.zeros(len(ids), 4), attention_mask=None,
                  ann_idx=torch.from_numpy(anns), ann_vote=torch.from_numpy(votes))
check("T8 em eval a auxiliar não entra (eval_loss = L_soft)",
      float(out_eval[0]) == 1.0 and model2.last_aux is None)


# ===========================================================================
# FASE 2
# ===========================================================================
M = stats["n_annotators"]
anns_t = torch.from_numpy(anns)
votes_t = torch.from_numpy(votes)

# --- T9: parametrização `sd` nasce no MESMO sítio que a `delta` -------------
a_delta = ConfusionAux(M, method="M1", param="delta")
a_sd = ConfusionAux(M, method="M1", param="sd")
C_d, C_s = a_delta.confusion_matrices(), a_sd.confusion_matrices()
check("T9a sd e delta produzem matrizes idênticas no init",
      bool(torch.allclose(C_d, C_s, atol=1e-6)),
      f"maxdiff={float((C_d - C_s).abs().max()):.2e} diag={float(C_s[0,0,0]):.6f}")
torch.manual_seed(321)
_before = torch.get_rng_state().clone()
_ = ConfusionAux(M, method="M1", param="sd")
_ = ConfusionAux(M, method="M1", param="sd", freeze_bias=True)
check("T9b construir em `sd` não consome RNG",
      bool(torch.equal(_before, torch.get_rng_state())))

# --- T10: invariante do modo congelado --------------------------------------
a_frz = ConfusionAux(M, method="M1", param="sd", freeze_bias=True)
with torch.no_grad():
    a_frz.s_ind.normal_(0.0, 1.0)          # competência individual mexida à mão
    a_frz.d_common.fill_(0.37)             # calibração do painel != 0
gap = a_frz.frozen_bias_gap()
expected = 2 * float(a_frz.d_common)
check("T10a congelado: u_a - v_a == 2*d_common para TODOS os anotadores",
      bool(np.allclose(gap, expected, atol=1e-6)),
      f"esperado={expected:.6f} max|desvio|={np.abs(gap - expected).max():.2e}")
Cf = a_frz.confusion_matrices().detach()
prob_bias = (Cf[:, 1, 0] - Cf[:, 0, 1]).numpy()
check("T10b em PROBABILIDADE o viés varia (artefacto da sigmóide, não fuga)",
      float(prob_bias.std()) > 1e-6,
      f"sd(viés_prob)={prob_bias.std():.4f} vs sd(viés_logit)={gap.std():.2e}")
names = dict(a_frz.named_parameters())
check("T10c congelado: `d_ind` não é Parameter", "d_ind" not in names,
      f"params={sorted(names)}")
lg = logits_a.clone().requires_grad_(True)
l, r = a_frz(lg, anns_t, votes_t)
(l + r).backward()
check("T10d congelado: `d_ind` não recebe gradiente",
      a_frz.d_ind.grad is None and a_frz.s_ind.grad is not None
      and float(a_frz.s_ind.grad.abs().sum()) > 0,
      f"grad(s_ind)={float(a_frz.s_ind.grad.abs().sum()):.4f}")
a_free = ConfusionAux(M, method="M1", param="sd", freeze_bias=False)
check("T10e livre: `d_ind` É Parameter", "d_ind" in dict(a_free.named_parameters()))

# --- T11: `vote_logprobs` é a identidade da perda auxiliar -------------------
a11 = ConfusionAux(M, method="M1", param="sd")
with torch.no_grad():
    a11.s_ind.normal_(0.0, 0.5)
l_aux, _ = a11(logits_a, anns_t, votes_t)
lp, p_yes, f_idx, f_vote = a11.vote_logprobs(logits_a, anns_t, votes_t)
check("T11a l_aux == -vote_logprobs(...).mean()",
      bool(torch.allclose(l_aux, -lp.mean(), atol=0.0)),
      f"delta={abs(float(l_aux) + float(lp.mean())):.2e}")
lp_id, p_id, _, _ = a11.vote_logprobs(logits_a, anns_t, votes_t, use_matrices=False)
p_lat_yes = (1.0 - torch.sigmoid(logits_a.view(-1)))
mask11 = votes_t >= 0
item_of = torch.arange(votes_t.shape[0]).unsqueeze(1).expand_as(votes_t)[mask11]
check("T11b use_matrices=False devolve a probabilidade CRUA do meme",
      bool(torch.allclose(p_id, p_lat_yes[item_of], atol=1e-6)),
      f"maxdiff={float((p_id - p_lat_yes[item_of]).abs().max()):.2e}")
perm11 = np.roll(np.arange(M), 1)
lp_p, _, _, _ = a11.vote_logprobs(logits_a, anns_t, votes_t, perm=perm11)
check("T11c perm troca a ficha (o resultado muda) mas não o voto nem o item",
      not bool(torch.allclose(lp_p, lp, atol=1e-6)),
      f"delta médio={float((lp_p - lp).abs().mean()):.4f}")

# --- T12: o callback da régua NÃO toca no RNG global ------------------------
strata = build_annotator_strata(DATA, ann_ids, verbose=False)
check("T12a estratos linguísticos batem com o nulo por permutação",
      [int((strata == k).sum()) for k in range(3)] == [437, 431, 19],
      str({n: int((strata == k).sum()) for k, n in enumerate(STRATUM_NAMES)}))
ann_stats = build_annotator_stats(table, split["train_ids"], M, verbose=False)
perm, prep = build_permutation(strata, seed=0, verbose=False)
check("T12b permutação fica dentro do estrato",
      bool((strata[perm] == strata).all()),
      f"pontos fixos: { {k: round(v['fixed_frac'], 4) for k, v in prep.items()} }")


class _FakeVoteDS(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        a, v = table[self.ids[i]]
        return {"input_ids": torch.full((4,), float(i % 7)),
                "attention_mask": torch.ones(4),
                "pixel_values": torch.zeros(2),
                "demo_vec": torch.zeros(3),
                "ann_idx": torch.from_numpy(a.copy()),
                "ann_vote": torch.from_numpy(v.copy())}


class _FakeVoteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)
        self.confusion = ConfusionAux(M, method="M1", param="sd")
        self.lam = 0.1

    def forward(self, input_ids, attention_mask, pixel_values=None,
                demo_vec=None, **kw):
        lg = self.lin(input_ids.float())
        return (lg.sum() * 0.0, lg, lg, lg)


class _State:
    epoch, global_step = 1.0, 10


VoteLogger = make_vote_metric_logger_class(object, torch.utils.data.DataLoader)
_fm = _FakeVoteModel()
_val_ids = [str(i) for i in split["val_ids"][:64]]
_trn_ids = [str(i) for i in split["train_ids"][:64]]
_vl = VoteLogger(_fm, _FakeVoteDS(_val_ids), _FakeVoteDS(_trn_ids), ann_stats,
                 perm, os.path.join(OUT_TMP, "t12_votes.json"),
                 batch_size=8, n_train_fit_items=64, use_fp16=False)
torch.manual_seed(999)
_rng_before = torch.get_rng_state().clone()
_vl._evaluate(_State())
check("T12c uma passagem do callback não altera o RNG global",
      bool(torch.equal(_rng_before, torch.get_rng_state())))
_rec = _vl.records[-1]["metrics"]
check("T12d o callback produz os 8 preditores, todos finitos",
      sorted(_rec) == ["a", "b", "b_perm", "c", "d3_train", "d3_val",
                       "d4_train", "d4_val"]
      and all(np.isfinite(v["nll"]) and np.isfinite(v["acc"])
              for v in _rec.values()),
      " ".join(f"{k}={v['nll']:.4f}" for k, v in sorted(_rec.items())))
# ⚠ (b) já bate (a) NO INIT, sem saber nada sobre pessoa nenhuma: C = softmax([3,0])
# encolhe a previsão para 0.5, o que é CALIBRAÇÃO GLOBAL, não sinal de anotador.
# Logo `b - a` não prova nada; o contraste limpo do individual é `b - b_perm`.
check("T12e (b) bate (a) já no init -- é encolhimento, não sinal",
      _rec["b"]["nll"] < _rec["a"]["nll"],
      f"a={_rec['a']['nll']:.4f} b={_rec['b']['nll']:.4f} "
      f"(delta={_rec['a']['nll'] - _rec['b']['nll']:.4f} sem aprender nada)")
check("T12f no init (b) == (b_perm) -- fichas iguais, permutar não muda nada",
      abs(_rec["b"]["nll"] - _rec["b_perm"]["nll"]) < 1e-9,
      "confirma que b-b_perm isola SÓ a parte individual")

# --- T13: corrupção sintética ----------------------------------------------
ct, manifest = corrupt_vote_table(table, ann_ids, strata,
                                  levels=CORRUPTION_LEVELS, n_per_level=20,
                                  seed=0, verbose=False)
ent = manifest["entries"]
check("T13a 120 anotadores, 20 por nível", len(ent) == 120 and all(
    sum(1 for e in ent if e["level"] == f) == 20 for f in CORRUPTION_LEVELS),
    f"n={len(ent)}")
bad = [e for e in ent if abs(e["n_flipped"] - round(e["level"] * e["n_votes"])) > 0]
check("T13b a fracção invertida bate o nível pedido", not bad,
      f"desvios={len(bad)}")
lvl0 = [e for e in ent if e["level"] == 0.0]
check("T13c nível 0% não altera voto nenhum",
      all(e["n_flipped"] == 0 for e in lvl0), f"n={len(lvl0)}")
n_diff = sum(int((ct[k][1] != table[k][1]).sum()) for k in table)
check("T13d total de votos invertidos bate com o manifesto",
      n_diff == manifest["n_votes_flipped"],
      f"tabela={n_diff} manifesto={manifest['n_votes_flipped']} "
      f"({100 * manifest['frac_votes_flipped']:.2f}% de todos)")
check("T13e a corrupção não muda quem votou nem quantos votos há",
      all(bool((ct[k][0] == table[k][0]).all()) for k in table)
      and all(int((ct[k][1] >= 0).sum()) == 6 for k in ct))
strata_sel = {STRATUM_NAMES[int(strata[e["annotator_idx"]])] for e in ent}
check("T13f selecção cobre os painéis linguísticos", len(strata_sel) >= 2,
      str(sorted(strata_sel)))
_y = (np.arange(200) % 2).astype(float)
_X = np.column_stack([_y * 3 - 1.5, np.random.default_rng(0).normal(size=200)])
_b = fit_logistic(_X, _y)
check("T13g fit_logistic converge num caso separável-ish",
      bool(np.all(np.isfinite(_b))) and _b[1] > 0, f"beta={np.round(_b, 3)}")

# --- T14: soft labels reconstruídas dos votos == as do A0 -------------------
_rebuilt = soft_labels_from_votes(table)
_worst = max(max(abs(_rebuilt[i][0] - soft[i][0]), abs(_rebuilt[i][1] - soft[i][1]))
             for i in all_ids)
check("T14a soft labels reconstruídas reproduzem soft_labels_taskA_raw.json",
      _worst < 1e-9, f"max|delta|={_worst:.2e} em {len(all_ids)} itens")
_soft_c = soft_labels_from_votes(ct)
_moved = sum(1 for i in all_ids if abs(_soft_c[i][0] - soft[i][0]) > 1e-9)
check("T14b corromper votos MOVE as soft labels (o mundo fica consistente)",
      _moved > 0, f"{_moved}/{len(all_ids)} itens com alvo alterado")

print()
print("FALHAS:", fails if fails else "nenhuma")
sys.exit(1 if fails else 0)
