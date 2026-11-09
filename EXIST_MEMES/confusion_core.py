"""Camada de confusão por anotador como PERDA AUXILIAR da Task A (memes).

Linha de caracterização pós-submissão. O objectivo NÃO é melhorar o sistema:
é medir se modelar o anotador com uma matriz de confusão acrescenta sinal ao
treino sobre soft labels. Um nulo é resultado válido.

    L = L_soft(A, B, C)  +  lambda * L_aux(A)

`L_soft` é a perda do A0, INALTERADA (`MultiHeadModel.forward`). Este módulo só
acrescenta um termo; nenhuma linha do baseline é tocada.

Convenções (fixadas aqui, verificadas contra `memes_core.py`)
-------------------------------------------------------------
- `LABELS_2_1 = ['YES', 'NO']` -> índice latente 0 = YES, 1 = NO.
- `head_a` devolve 1 logit e `sigmoid(logit) = P(NO)` (memes_core.py:837 e :722).
  Logo `p_latente = [1 - sigmoid, sigmoid]`. Os votos usam O MESMO mapa
  (`VOTE_MAP = {'YES': 0, 'NO': 1}`) -- note que `reliability/exist_data.py` usa
  a convenção INVERSA (NO=0/YES=1); não misturar os dois.
- `C_a[i, j] = P(voto = j | latente = i)`, linhas somando 1 (softmax por linha),
  logo `p_voto = p_latente @ C_a`. O handoff escrevia `C_a @ p_latente` com
  linhas normalizadas -- isso não devolve uma distribuição; a forma usada aqui é
  a equivalente correcta e está declarada como tal no relatório.
- `L_aux` = média (não soma) do cross-entropy sobre os pares (anotador, item)
  com voto presente no batch. Média mantém `lambda` na mesma escala da
  `L_soft`, que também é média. Declarado no relatório.

Métodos
-------
- `M1`  Chu et al. (AAAI 2021), ADAPTADO: confusão comum + desvio individual,
        `C_a = row_softmax(L_comum + Delta_a)`, com `Delta_a` regularizado para
        zero. (A decomposição de CoNAL tem ainda um portão dependente da
        instância; não é implementado -- por isso "adaptado", não "é CoNAL".)
- `M2`  Tanno et al. (CVPR 2019): `C_a` individual completa + penalização do
        traço somada à perda.
- `M3`  Rodrigues & Pereira (AAAI 2018), crowd layer: `C_a` individual, sem
        partilha e sem regularização. Baseline de atribuição da família.

Determinismo
------------
- A inicialização é uma CONSTANTE (zeros num offset de identidade); não consome
  RNG. Com `lambda = 0` a trajectória de treino tem de ser bit-a-bit igual à do
  A0 para a mesma seed -- é esse o controlo obrigatório.
- Todos os lookups (anotador, item, classe do voto) são feitos por
  COMPARAÇÃO + MATMUL, nunca por `F.one_hot`, indexação avançada ou
  `nn.Embedding`: sob `torch.use_deterministic_algorithms(True)` o backward de
  scatter/index_put em CUDA aborta ou é não-determinístico.
- A auxiliar corre com o autocast EXPLICITAMENTE DESLIGADO: `fp16=True` faria
  matmul/einsum em fp16 mesmo depois de um `.float()`, e log-probs perto de 0/1
  em fp16 são ruído.

FASE 2 (2026-09-10)
-------------------
A Fase 1 mediu que as matrizes, quando finalmente se movem (`CONF_LR=1e-2`),
COLAPSAM NO VIÉS MARGINAL (88,7% da variância, R^2 = 0,91 contra a `yes_rate`) e
não movem o ICM (Δ = +0,0009, p = 0,30, contra sd 0,0059 do próprio A0 entre
seeds). Três acrescentos, nenhum deles altera o caminho da Fase 1:

1. `param="sd"` -- reparametriza `C_a` nas suas duas direcções COM SIGNIFICADO,
   competência e viés, uma por anotador em vez de quatro números redundantes.
   Permite o braço com o VIÉS CONGELADO (`freeze_bias=True`), que remove a
   única direcção que a Fase 1 mostrou ser estimável, para ver o que sobra.
2. `vote_logprobs` + `make_vote_metric_logger_class` -- a RÉGUA: prever os
   3588 votos individuais dos 598 itens de validação, que nunca entraram na
   `L_soft` nem na `L_aux`. O ICM compara dois TREINOS e por isso tem o ruído
   de seed por baixo; esta régua compara PREDITORES dentro do mesmo treino.
3. `corrupt_vote_table` -- o CONTROLO POSITIVO: estragar de propósito os votos
   de anotadores escolhidos, em níveis conhecidos. Sem ele não se distingue
   "o instrumento não funciona" de "as pessoas não são anómalas", e nenhum
   nulo é interpretável.

⚠ Armadilha medida no T12: `b` (ficha aprendida) já bate `a` (sem modelo de
anotador) NO INIT, por encolhimento da previsão -- calibração global, não sinal
de pessoa. O contraste limpo é `b` contra `b_perm` (ficha de outra pessoa do
mesmo painel linguístico) e contra `d4`.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Convenções de rótulo (ver docstring)
# ---------------------------------------------------------------------------
LABELS_A = ["YES", "NO"]
VOTE_MAP = {"YES": 0, "NO": 1}
N_CLASSES_A = 2
MAX_VOTES_PER_ITEM = 6          # medido: todos os 3984 itens têm exactamente 6
PAD_ANN = 0                     # índice qualquer; a máscara é `vote >= 0`
PAD_VOTE = -1

METHODS = ("M1", "M2", "M3")

# Parametrização das matrizes:
#   "delta" -- LEGADO (Fase 1): desvio livre [M, 2, 2] sobre um offset de
#              identidade. Tem 2 graus de liberdade redundantes por anotador
#              (a softmax por linha é invariante a somar uma constante à linha).
#   "sd"    -- FASE 2: as duas direcções com significado, competência e viés,
#              uma por anotador. Mínima, e permite congelar SÓ o viés.
PARAMS = ("delta", "sd")

# Offset de identidade na parametrização em logits: softmax([IDENT_LOGIT, 0])
# = [0.953, 0.047]. "Perto da identidade", não a identidade exacta (impossível
# com softmax). O valor é constante e igual nos três métodos.
IDENT_LOGIT = 3.0

# Pesos de regularização FIXADOS A PRIORI (não são grelha; ver protocolo).
DEFAULT_REG = {
    "M1": 0.01,   # L2 sobre Delta_a (desvio individual -> 0)
    "M2": 0.01,   # penalização do traço (valor de partida de Tanno et al.)
    "M3": 0.0,    # por desenho: sem regularização
}


# ---------------------------------------------------------------------------
# 1. Tabela de votos por item
# ---------------------------------------------------------------------------
def build_vote_table(json_path, verbose=True):
    """Lê o JSON de treino e devolve os votos da Task A por item.

    A chave é a CHAVE DO DICIONÁRIO do JSON -- a mesma que `_id_frozen` e o
    `split_taskA_frozen.json` usam (verificado: chave == `id_EXIST` em 3984/3984).

    Returns
    -------
    table : dict[str, (ann_idx int64[6], vote int64[6])]  com padding PAD_VOTE
    annotator_ids : list[str]  índice -> ID persistente ("Annotator_193")
    stats : dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ann_to_idx = {}
    annotator_ids = []
    table = {}
    n_votes = 0
    n_dropped = 0

    for key, rec in raw.items():
        annotators = rec.get("annotators") or []
        labels = rec.get("labels_task2_1") or []
        idxs, votes = [], []
        for aid, lab in zip(annotators, labels):
            lab = str(lab).strip().upper()
            if lab not in VOTE_MAP:            # UNKNOWN / '-' / REPORTED
                n_dropped += 1
                continue
            if aid not in ann_to_idx:
                ann_to_idx[aid] = len(annotator_ids)
                annotator_ids.append(aid)
            idxs.append(ann_to_idx[aid])
            votes.append(VOTE_MAP[lab])
        n_votes += len(votes)
        if len(votes) > MAX_VOTES_PER_ITEM:
            raise ValueError(
                f"item {key}: {len(votes)} votos > MAX_VOTES_PER_ITEM={MAX_VOTES_PER_ITEM}"
            )
        pad = MAX_VOTES_PER_ITEM - len(votes)
        table[str(key)] = (
            np.asarray(idxs + [PAD_ANN] * pad, dtype=np.int64),
            np.asarray(votes + [PAD_VOTE] * pad, dtype=np.int64),
        )

    stats = {
        "n_items": len(table),
        "n_annotators": len(annotator_ids),
        "n_votes": n_votes,
        "n_dropped_votes": n_dropped,
        "votes_per_annotator_mean": n_votes / max(len(annotator_ids), 1),
    }
    if verbose:
        print(f"[confusion] itens={stats['n_items']} anotadores={stats['n_annotators']} "
              f"votos={stats['n_votes']} descartados={stats['n_dropped_votes']} "
              f"votos/anotador={stats['votes_per_annotator_mean']:.2f}")
    return table, annotator_ids, stats


# ---------------------------------------------------------------------------
# 2. Dataset: acrescenta os votos ao batch, sem tocar no resto
# ---------------------------------------------------------------------------
class VoteAugmentedDataset(Dataset):
    """Embrulha um `SexismDataset` e junta `ann_idx` e `ann_vote` ao batch.

    Nenhuma chave contém a substring "label": o `Trainer` monta
    `eval_pred.label_ids` a partir da assinatura do `forward` procurando nomes
    com "label", e uma chave nova ali partiria o `compute_metrics`.

    Só o dataset de TREINO deve ser embrulhado. O de validação fica cru, logo
    `eval_loss` continua a ser exactamente a `L_soft` do A0 -- que é o selector
    de época (E = argmin(eval_loss)). Mudar o selector seria confound.
    """

    def __init__(self, base_dataset, frozen_ids, vote_table):
        self.base = base_dataset
        self.ids = [str(i) for i in frozen_ids]
        if len(self.ids) != len(base_dataset):
            raise ValueError(
                f"frozen_ids ({len(self.ids)}) != len(dataset) ({len(base_dataset)})"
            )
        missing = [i for i in self.ids if i not in vote_table]
        if missing:
            raise KeyError(f"{len(missing)} ids sem votos, ex.: {missing[:3]}")
        self.vote_table = vote_table

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        batch = self.base[idx]
        ann_idx, votes = self.vote_table[self.ids[idx]]
        batch["ann_idx"] = torch.from_numpy(ann_idx.copy())
        batch["ann_vote"] = torch.from_numpy(votes.copy())
        return batch


# ---------------------------------------------------------------------------
# 3. A camada de confusão
# ---------------------------------------------------------------------------
class ConfusionAux(nn.Module):
    """Matrizes de confusão por anotador + a perda auxiliar da Task A.

    Parâmetros (todos inicializados com CONSTANTES -- não consomem RNG):
      - `delta` [M, 2, 2] : desvio individual, init 0.
      - `common` [2, 2]   : só em M1, confusão partilhada pelo painel, init
                            IDENT_LOGIT * I.
      Em M2/M3 o offset de identidade é um buffer constante somado ao `delta`,
      logo o efeito de init é o mesmo nos três métodos.
    """

    def __init__(self, n_annotators, method="M1", reg_weight=None,
                 param="delta", freeze_bias=False):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method desconhecido: {method!r} (esperado {METHODS})")
        if param not in PARAMS:
            raise ValueError(f"param desconhecido: {param!r} (esperado {PARAMS})")
        if param == "sd" and method != "M1":
            raise ValueError("param='sd' só está definido para method='M1' (a "
                             "decomposição competência/viés supõe o termo comum)")
        if freeze_bias and param != "sd":
            raise ValueError("freeze_bias exige param='sd' -- em 'delta' as duas "
                             "direcções não estão separadas")
        self.method = method
        self.param = param
        self.freeze_bias = bool(freeze_bias)
        self.n_annotators = n_annotators
        self.reg_weight = DEFAULT_REG[method] if reg_weight is None else float(reg_weight)

        if param == "delta":
            ident = IDENT_LOGIT * torch.eye(N_CLASSES_A)
            self.delta = nn.Parameter(torch.zeros(n_annotators, N_CLASSES_A, N_CLASSES_A))
            if method == "M1":
                self.common = nn.Parameter(ident.clone())
            else:
                self.register_buffer("ident", ident)
        else:
            # Parametrização em duas direcções COM SIGNIFICADO (Fase 2).
            #   u_a = s_common + d_common + s_ind[a]   (linha do latente YES)
            #   v_a = s_common - d_common + s_ind[a]   (linha do latente NO)
            # competência = 2*sigma(s) - 1 ; viés = sigma(u) - sigma(v).
            # Init s_common=IDENT_LOGIT, resto zero, reproduz softmax([3,0]) das
            # runs anteriores por construção (teste T9).
            self.s_common = nn.Parameter(torch.tensor(float(IDENT_LOGIT)))
            self.d_common = nn.Parameter(torch.zeros(()))
            self.s_ind = nn.Parameter(torch.zeros(n_annotators))
            if freeze_bias:
                # Buffer, não Parameter: sai de `named_parameters()` e do
                # optimizador de graça, sem precisar de filtro em lado nenhum.
                self.register_buffer("d_ind", torch.zeros(n_annotators))
            else:
                self.d_ind = nn.Parameter(torch.zeros(n_annotators))

        # Identidade EXACTA, só para o teste de sanidade (nunca no treino).
        self.force_identity = False

    # -- matrizes ----------------------------------------------------------
    @property
    def _device(self):
        return next(iter(self.buffers()), next(self.parameters())).device

    def logits_matrix(self):
        """[M, 2, 2] em espaço de logits, antes do softmax por linha."""
        if self.param == "sd":
            u = self.s_common + self.d_common + self.s_ind + self.d_ind
            v = self.s_common - self.d_common + self.s_ind - self.d_ind
            zero = torch.zeros_like(u)
            # [[u, 0], [0, v]] -- e softmax([u,0]) = [sigma(u), 1-sigma(u)]
            # EXACTAMENTE, logo o resto do módulo (force_identity,
            # identity_distance, T3) continua válido sem alteração.
            return torch.stack([torch.stack([u, zero], dim=-1),
                                torch.stack([zero, v], dim=-1)], dim=-2)
        if self.method == "M1":
            return self.common.unsqueeze(0) + self.delta
        return self.ident.unsqueeze(0) + self.delta

    def confusion_matrices(self):
        """[M, 2, 2] com linhas somando 1: C[a, i, j] = P(voto=j | latente=i)."""
        if self.force_identity:
            return torch.eye(N_CLASSES_A, device=self._device,
                             dtype=torch.float32).expand(self.n_annotators, -1, -1)
        return F.softmax(self.logits_matrix().float(), dim=-1)

    # -- log-probabilidade por voto ---------------------------------------
    def vote_logprobs(self, logits_a, ann_idx, ann_vote, use_matrices=True,
                      perm=None):
        """Log-probabilidade que o modelo dá a CADA voto observado.

        É o núcleo partilhado pela perda auxiliar e pela régua de votos retidos
        da Fase 2 -- `l_aux == -vote_logprobs(...)[0].mean()` é uma identidade
        exacta, verificada no teste T11.

        Parameters
        ----------
        logits_a : [B, 1] ou [B]  -- `sigmoid(logits_a) = P(NO)`
        ann_idx  : [B, K] int64
        ann_vote : [B, K] int64, PAD_VOTE onde não há voto
        use_matrices : se False, `C = identidade` -> é o preditor (a) da régua,
            "sem modelo de anotador". Passa pelo MESMO código, para que a
            diferença entre (a) e (b) seja só a matriz e mais nada.
        perm : [M] int64 opcional. `perm[a]` diz de que anotador vem a ficha
            usada para o anotador `a`. É o controlo negativo (b_embaralhado):
            se (b) continua a ganhar com fichas trocadas, a vantagem não vinha
            de conhecer AQUELA pessoa. A permutação tem de ser construída
            estratificada por painel linguístico (ver `build_permutation`).

        Returns
        -------
        (logp[V], p_yes[V], flat_idx[V], flat_vote[V]) ou (None, None, None, None)
        `p_yes` é P(voto = YES) previsto; `flat_vote` usa VOTE_MAP (0 = YES).
        """
        mask = ann_vote >= 0
        if not bool(mask.any()):
            return None, None, None, None

        # `fp16=True` corre o forward sob autocast, e o autocast converte as
        # entradas de matmul/einsum para fp16 mesmo depois de um `.float()`.
        # Desligá-lo aqui é o que torna verdadeira a afirmação "a auxiliar corre
        # em float32": log-probs perto de 0/1 em fp16 são ruído.
        with torch.autocast(device_type=logits_a.device.type, enabled=False):
            p_no = torch.sigmoid(logits_a.float().view(-1))      # [B]
            p_lat = torch.stack([1.0 - p_no, p_no], dim=-1)      # [B, 2] = [YES, NO]

            if use_matrices:
                C = self.confusion_matrices()                    # [M, 2, 2]
            else:
                C = torch.eye(N_CLASSES_A, device=p_lat.device,
                              dtype=torch.float32).expand(self.n_annotators, -1, -1)

            flat_idx = ann_idx[mask]                             # [V]
            flat_vote = ann_vote[mask]                           # [V]
            B, K = ann_vote.shape
            item_of_vote = torch.arange(
                B, device=ann_vote.device
            ).unsqueeze(1).expand(B, K)[mask]                    # [V]

            if perm is not None:
                # Só o índice da FICHA é trocado; o voto e o item ficam onde
                # estavam. Forward-only (a permutação é uma constante), logo o
                # gather aqui não cria backward não-determinístico.
                perm_t = torch.as_tensor(perm, device=flat_idx.device,
                                         dtype=flat_idx.dtype)
                idx_for_matrix = perm_t[flat_idx]
            else:
                idx_for_matrix = flat_idx

            # Tudo por comparação + matmul: sem `F.one_hot` (scatter_), sem
            # indexação avançada (index_put no backward). Sob
            # `use_deterministic_algorithms(True)` em CUDA, essas duas famílias
            # abortam ou não são determinísticas; matmul é sempre segura.
            ar_ann = torch.arange(self.n_annotators, device=C.device)
            onehot_ann = (idx_for_matrix.unsqueeze(1) == ar_ann).to(C.dtype)   # [V, M]
            C_v = (onehot_ann @ C.reshape(self.n_annotators, -1)).view(
                -1, N_CLASSES_A, N_CLASSES_A)                                   # [V, 2, 2]

            ar_item = torch.arange(B, device=p_lat.device)
            onehot_item = (item_of_vote.unsqueeze(1) == ar_item).to(p_lat.dtype)  # [V, B]
            p_lat_v = onehot_item @ p_lat                                        # [V, 2]

            p_vote = torch.einsum("vi,vij->vj", p_lat_v, C_v)                    # [V, 2]

            ar_cls = torch.arange(N_CLASSES_A, device=p_vote.device)
            onehot_vote = (flat_vote.unsqueeze(1) == ar_cls).to(p_vote.dtype)    # [V, 2]
            logp = (onehot_vote * torch.log(p_vote.clamp_min(1e-8))).sum(-1)     # [V]
            p_yes = p_vote[:, VOTE_MAP["YES"]]                                   # [V]
        return logp, p_yes, flat_idx, flat_vote

    # -- perdas ------------------------------------------------------------
    def forward(self, logits_a, ann_idx, ann_vote):
        """Devolve `(l_aux, l_reg)`, ambos escalares (ou None se não há votos)."""
        logp, _, _, _ = self.vote_logprobs(logits_a, ann_idx, ann_vote)
        if logp is None:
            return None, None
        with torch.autocast(device_type=logits_a.device.type, enabled=False):
            l_aux = -logp.mean()
            # A matriz só é recalculada quando o regularizador precisa dela
            # (só o M2 usa `C`); com `reg_weight = 0` não há custo nenhum.
            l_reg = (None if self.reg_weight == 0.0
                     else self._regularizer(self.confusion_matrices()))
        return l_aux, l_reg

    def _regularizer(self, C):
        if self.reg_weight == 0.0:
            return None
        if self.method == "M1":
            if self.param == "sd":
                # Desvio INDIVIDUAL -> zero, na parametrização nova. Em `s_ind`
                # puxa a competência para o valor do painel (`s_common`); em
                # `d_ind` puxa o viés para a calibração do painel.
                # No braço congelado `d_ind` é um buffer de zeros e não entra
                # (o `.mean()` fica só sobre os parâmetros livres, mantendo o
                # mesmo peso POR PARÂMETRO LIVRE nos dois braços).
                terms = [self.s_ind.float().pow(2)]
                if not self.freeze_bias:
                    terms.append(self.d_ind.float().pow(2))
                return self.reg_weight * torch.cat(terms).mean()
            # desvio individual -> zero
            return self.reg_weight * self.delta.float().pow(2).mean()
        if self.method == "M2":
            # Tanno et al.: penalizar o traço das matrizes (empurra para longe
            # da identidade degenerada; o mínimo da CE já puxa na direcção oposta)
            trace = C.diagonal(dim1=-2, dim2=-1).sum(-1)          # [M]
            return self.reg_weight * trace.mean()
        return None

    # -- diagnóstico -------------------------------------------------------
    @torch.no_grad()
    def identity_distance(self):
        """||C_a - I||_F por anotador, [M] numpy. Diagnóstico, não critério."""
        C = self.confusion_matrices().detach().float().cpu()
        eye = torch.eye(N_CLASSES_A)
        return (C - eye).pow(2).sum(dim=(-2, -1)).sqrt().numpy()

    @torch.no_grad()
    def frozen_bias_gap(self):
        """`u_a - v_a` por anotador, [M] numpy. Só faz sentido em `param='sd'`.

        ⚠ O invariante do modo congelado é ESTE, em espaço de logit:
        `u_a - v_a == 2 * d_common` para TODO `a`. O teste que o handoff da
        Fase 2 propunha (`C[:, 0, 1] == C[:, 1, 0]`) está errado sempre que
        `d_common != 0`.

        E em espaço de PROBABILIDADE, `sigma(u_a) - sigma(v_a)` **varia** com
        `s_ind[a]` mesmo com `d_ind ≡ 0`, por não-linearidade da sigmóide. Sem
        esta distinção, o braço congelado pareceria continuar a aprender viés.
        """
        if self.param != "sd":
            raise ValueError("frozen_bias_gap() exige param='sd'")
        u = self.s_common + self.d_common + self.s_ind + self.d_ind
        v = self.s_common - self.d_common + self.s_ind - self.d_ind
        return (u - v).detach().float().cpu().numpy()

    @torch.no_grad()
    def competence_bias(self):
        """`(competencia[M], vies_logit[M])` numpy, em `param='sd'`.

        competência = 2*sigma(s_common + s_ind) - 1   (1 = perfeito, 0 = moeda ao ar)
        viés_logit  = u_a - v_a                        (0 = sem preferência)
        """
        if self.param != "sd":
            raise ValueError("competence_bias() exige param='sd'")
        s = (self.s_common + self.s_ind).detach().float().cpu()
        comp = (2.0 * torch.sigmoid(s) - 1.0).numpy()
        return comp, self.frozen_bias_gap()


class ConfusionMultiHeadModel(nn.Module):
    """`MultiHeadModel` + `ConfusionAux`, sem alterar o baseline.

    Composição em vez de herança para o `MultiHeadModel` ser construído
    EXACTAMENTE como no A0 (mesma sequência de draws de RNG) e a camada de
    confusão nascer depois, de constantes.

    A auxiliar só entra quando `self.training` E o batch traz votos -- a
    validação continua a medir `L_soft` pura.
    """

    def __init__(self, base_model, n_annotators, method="M1", lam=0.0,
                 reg_weight=None, param="delta", freeze_bias=False):
        super().__init__()
        self.base = base_model
        self.confusion = ConfusionAux(n_annotators, method=method,
                                      reg_weight=reg_weight,
                                      param=param, freeze_bias=freeze_bias)
        self.lam = float(lam)
        self.last_aux = None      # diagnóstico: valores do último forward
        self.last_reg = None

    def __getattr__(self, name):
        """Atributos não encontrados no invólucro caem para o modelo-base.

        As células de diagnóstico do notebook chamam `model.vision_encoder` e
        afins directamente; sem isto, embrulhar o modelo partia-as.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base":
                raise
            return getattr(super().__getattr__("base"), name)

    def forward(self, input_ids, attention_mask, pixel_values=None, demo_vec=None,
                labels_a=None, labels_b=None, labels_c=None,
                ann_idx=None, ann_vote=None, **kwargs):
        loss, logits_a, logits_b, logits_c = self.base(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, demo_vec=demo_vec,
            labels_a=labels_a, labels_b=labels_b, labels_c=labels_c, **kwargs
        )

        if self.training and ann_idx is not None and ann_vote is not None:
            # O caminho auxiliar fica SEMPRE vivo, mesmo com lambda = 0: é o que
            # transforma o controlo lambda=0 em detector de bug (0 * NaN = NaN).
            l_aux, l_reg = self.confusion(logits_a, ann_idx, ann_vote)
            if l_aux is not None:
                self.last_aux = float(l_aux.detach())
                loss = loss + self.lam * l_aux
                if l_reg is not None:
                    self.last_reg = float(l_reg.detach())
                    loss = loss + self.lam * l_reg

        return (loss, logits_a, logits_b, logits_c)


# ---------------------------------------------------------------------------
# 4. Trainer: tirar as matrizes do weight decay
# ---------------------------------------------------------------------------
def make_confusion_trainer_class(trainer_cls, conf_lr=None):
    """Subclasse de `Trainer` para os parâmetros de confusão.

    Faz duas coisas:

    1. **Exclui `confusion.*` do weight decay.** `TrainingArguments(weight_decay=
       0.05)` aplicaria decaimento aos logits das matrizes, puxando-os para ZERO
       -- isto é, para a matriz UNIFORME, o oposto de "perto da identidade", e em
       M3 seria uma regularização que a spec proíbe. Cada método fica só com o
       seu próprio regularizador explícito.

    2. **`conf_lr`: learning rate próprio para as matrizes.** Com `conf_lr=None`
       (pré-registado) elas herdam o `learning_rate` do encoder -- 1e-5, tunado
       para afinar um XLM-R pré-treinado. MEDIDO a 2026-09-07: com AdamW, cujo
       passo é ~`lr` por iteração quando o sinal do gradiente é consistente, 795
       passos dão no máximo 7,95e-3 de deslocamento em espaço de logit, ou seja
       Δp ~3,6e-4 no ponto de init. As matrizes moveram-se 6e-4 -- praticamente
       o máximo que o LR permite. Não colapsaram na identidade: nunca saíram do
       sítio onde nasceram.

       Pior: **AdamW é invariante à escala do gradiente** (`update = lr*m/(√v+ε)`,
       e um gradiente λ·g escala `m` e `√v` juntos). Para as matrizes, a auxiliar
       é a ÚNICA fonte de gradiente, logo `lambda` cancela-se e a grelha de
       lambda não muda quanto elas se movem -- só quanto a auxiliar perturba o
       encoder.

       Passar `conf_lr` põe `confusion.*` num grupo de parâmetros próprio. O
       scheduler do HF aplica o mesmo factor multiplicativo a todos os grupos,
       logo o grupo novo também tem warmup e decaimento, a partir da SUA base.
    """

    class ConfusionTrainer(trainer_cls):
        confusion_lr = conf_lr

        def get_decay_parameter_names(self, model):
            names = super().get_decay_parameter_names(model)
            return [n for n in names if not n.startswith("confusion.")]

        _conf_group_added = False

        def create_optimizer(self):
            optimizer = super().create_optimizer()
            # `super().create_optimizer()` devolve o optimizador já existente se
            # for chamado duas vezes (o `Trainer` chama-o pelo menos uma, e o
            # utilizador pode chamá-lo antes). Sem esta guarda, o grupo seria
            # acrescentado outra vez e os mesmos parâmetros levariam DOIS updates
            # por passo.
            if self.confusion_lr is None or self._conf_group_added:
                return optimizer
            model = self.model
            conf_ids = {id(p) for n, p in model.named_parameters()
                        if n.startswith("confusion.")}
            moved = []
            for group in optimizer.param_groups:
                keep = []
                for prm in group["params"]:
                    (moved if id(prm) in conf_ids else keep).append(prm)
                group["params"] = keep
            if moved:
                # `add_param_group` completa os restantes hiperparâmetros a partir
                # dos defaults do optimizador (betas, eps). Weight decay fica a
                # zero, como no caminho pré-registado.
                optimizer.add_param_group(
                    {"params": moved, "lr": float(self.confusion_lr),
                     "weight_decay": 0.0}
                )
                self._conf_group_added = True
            # O scheduler é construído DEPOIS do optimizador, logo apanha este
            # grupo e o seu `initial_lr`.
            return optimizer

    return ConfusionTrainer


# ---------------------------------------------------------------------------
# 5. Callback: guardar as matrizes por época (E é escolhida post-hoc)
# ---------------------------------------------------------------------------
def make_confusion_logger_class(callback_cls):
    """Callback que despeja as matrizes de confusão a cada avaliação."""

    class ConfusionMatrixLogger(callback_cls):
        def __init__(self, model, annotator_ids, out_path):
            self.model = model
            self.annotator_ids = list(annotator_ids)
            self.out_path = out_path
            self.snapshots = []
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        def _dump(self, state):
            with torch.no_grad():
                C = self.model.confusion.confusion_matrices().detach().float().cpu().numpy()
            snap = {
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "global_step": int(state.global_step),
                "matrices": C.tolist(),
            }
            if self.model.confusion.param == "sd":
                # Por época, não só no fim: `E = argmin(eval_loss)` é escolhida
                # post-hoc e a verificação do invariante tem de correr NESSA
                # época, não na última.
                with torch.no_grad():
                    snap["s_common"] = float(self.model.confusion.s_common)
                    snap["d_common"] = float(self.model.confusion.d_common)
            self.snapshots.append(snap)
            conf = self.model.confusion
            meta = {"annotator_ids": self.annotator_ids,
                    "method": conf.method,
                    "param": conf.param,
                    "freeze_bias": bool(conf.freeze_bias),
                    "lam": self.model.lam,
                    "reg_weight": conf.reg_weight,
                    "snapshots": self.snapshots}
            if conf.param == "sd":
                # Sem `d_common` não é possível verificar o invariante do modo
                # congelado (`u_a - v_a == 2*d_common`) a partir do ficheiro, e
                # sem `param`/`freeze_bias` a análise não distingue um braço do
                # outro só pelas matrizes.
                with torch.no_grad():
                    meta["s_common"] = float(conf.s_common)
                    meta["d_common"] = float(conf.d_common)
                    meta["frozen_bias_gap_max_dev"] = float(np.abs(
                        conf.frozen_bias_gap() - 2 * float(conf.d_common)).max())
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

        def on_evaluate(self, args, state, control, **kwargs):
            self._dump(state)

        def on_train_end(self, args, state, control, **kwargs):
            self._dump(state)

    return ConfusionMatrixLogger


# ---------------------------------------------------------------------------
# 6. FASE 2 -- estatísticas de anotador, estratos e permutação
# ---------------------------------------------------------------------------
STRATUM_NAMES = ("es_only", "en_only", "bilingual")


def build_annotator_strata(json_path, annotator_ids, verbose=True):
    """Estrato de painel linguístico de cada anotador. [M] int64.

    0 = `es_only` | 1 = `en_only` | 2 = `bilingual`.

    Porquê estratificar (herdado de `reliability/make_permuted_soft_labels.py`,
    e obrigatório): os painéis ES e EN são quase disjuntos. Permutar fichas
    ENTRE línguas destruiria estrutura que **não é comportamento de anotador**
    -- quem viu que língua -- e daria um nulo artificialmente fácil de bater.
    Os bilingues formam estrato próprio: é simétrico e preserva o painel.

    Contagens esperadas (medidas em `permutation_null_report_fixed.md`):
    437 es_only / 431 en_only / 19 bilingual, num total de 887.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    langs = {aid: set() for aid in annotator_ids}
    for rec in raw.values():
        lang = str(rec.get("lang", "")).strip().lower()
        for aid in (rec.get("annotators") or []):
            if aid in langs:
                langs[aid].add(lang)

    strata = np.empty(len(annotator_ids), dtype=np.int64)
    for i, aid in enumerate(annotator_ids):
        seen = langs[aid]
        if len(seen) > 1:
            strata[i] = 2
        elif "en" in seen:
            strata[i] = 1
        else:
            strata[i] = 0
    if verbose:
        counts = {n: int((strata == k).sum()) for k, n in enumerate(STRATUM_NAMES)}
        print(f"[confusion] estratos linguísticos: {counts}")
    return strata


def build_annotator_stats(vote_table, train_ids, n_annotators, verbose=True):
    """Taxa de YES e concordância por anotador, calculadas SÓ do TREINO.

    ⚠ Nunca usar a `yes_rate` de `reliability/outputs/behavioral_profiles.csv`
    para isto: é calculada sobre os 3984 itens, logo inclui os votos de
    VALIDAÇÃO -- que são exactamente os que a régua da Fase 2 tenta prever.

    - Suavização de Laplace `(n_yes + 1) / (n + 2)`. Sem ela, anotadores
      todo-YES ou todo-NO dão `logit = ±inf` nos preditores (c) e (d).
    - `agree` = concordância com a maioria DEIXANDO-UM-DE-FORA (os outros 5
      votos do item). 5 é ímpar, logo não há empates; itens com um número par
      de outros votos são ignorados para esta estatística e contados à parte.
    - Anotadores sem votos de treino recuam para a taxa base do painel. É
      justo: em (b) esses mesmos anotadores têm `C_a` = painel, porque
      `s_ind = d_ind = 0` nunca recebeu gradiente.
    """
    n_yes = np.zeros(n_annotators, dtype=np.int64)
    n_tot = np.zeros(n_annotators, dtype=np.int64)
    n_agree = np.zeros(n_annotators, dtype=np.int64)
    n_agree_tot = np.zeros(n_annotators, dtype=np.int64)
    n_ties = 0

    for key in train_ids:
        anns, votes = vote_table[str(key)]
        valid = votes >= 0
        v_ann, v_vote = anns[valid], votes[valid]
        yes_total = int((v_vote == VOTE_MAP["YES"]).sum())
        k = len(v_vote)
        for j in range(k):
            a = int(v_ann[j])
            n_tot[a] += 1
            if v_vote[j] == VOTE_MAP["YES"]:
                n_yes[a] += 1
            others = k - 1
            if others <= 0:
                continue
            yes_others = yes_total - int(v_vote[j] == VOTE_MAP["YES"])
            if 2 * yes_others == others:          # empate: sem maioria definida
                n_ties += 1
                continue
            maj = VOTE_MAP["YES"] if 2 * yes_others > others else VOTE_MAP["NO"]
            n_agree_tot[a] += 1
            if v_vote[j] == maj:
                n_agree[a] += 1

    panel_yes_rate = float(n_yes.sum() / max(n_tot.sum(), 1))
    yes_rate = (n_yes + 1.0) / (n_tot + 2.0)
    agree = (n_agree + 1.0) / (n_agree_tot + 2.0)
    unseen = n_tot == 0
    yes_rate[unseen] = panel_yes_rate
    agree[unseen] = float(n_agree.sum() / max(n_agree_tot.sum(), 1))

    stats = {
        "yes_rate": yes_rate, "agree": agree, "n_votes": n_tot,
        "panel_yes_rate": panel_yes_rate,
        "n_unseen_annotators": int(unseen.sum()),
        "n_ties_skipped": n_ties,
        "n_train_votes": int(n_tot.sum()),
    }
    if verbose:
        print(f"[confusion] estatísticas de TREINO: {stats['n_train_votes']} votos | "
              f"taxa YES do painel {panel_yes_rate:.4f} | "
              f"anotadores sem votos de treino {stats['n_unseen_annotators']} | "
              f"empates ignorados {n_ties}")
    return stats


def build_permutation(strata, seed=0, verbose=True):
    """Permutação de fichas DENTRO de cada estrato linguístico. [M] int64.

    `perm[a]` = índice do anotador cuja ficha o anotador `a` recebe. É o
    controlo negativo (b_embaralhado) da régua de votos: se (b) continua a
    ganhar com fichas trocadas, a vantagem não vinha de conhecer aquela pessoa.

    Reporta a fracção de pontos fixos, pelo mesmo critério do nulo por
    permutação já existente no repositório (< 0.05 em estratos com n >= 20).
    """
    rng = np.random.default_rng(seed)
    perm = np.arange(len(strata), dtype=np.int64)
    report = {}
    for k, name in enumerate(STRATUM_NAMES):
        members = np.flatnonzero(strata == k)
        if len(members) < 2:
            report[name] = {"n": int(len(members)), "fixed_frac": 1.0}
            continue
        shuffled = members.copy()
        for _ in range(20):                       # tenta reduzir pontos fixos
            rng.shuffle(shuffled)
            if float((shuffled == members).mean()) < 0.05 or len(members) < 20:
                break
        perm[members] = shuffled
        report[name] = {"n": int(len(members)),
                        "fixed_frac": float((shuffled == members).mean())}
    if verbose:
        print(f"[confusion] permutação de fichas (estratificada): {report}")
    return perm, report


# ---------------------------------------------------------------------------
# 7. FASE 2 -- corrupção sintética (CONTROLO POSITIVO)
# ---------------------------------------------------------------------------
CORRUPTION_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def corrupt_vote_table(vote_table, annotator_ids, strata,
                       levels=CORRUPTION_LEVELS, n_per_level=20, seed=0,
                       verbose=True):
    """Inverte uma fracção conhecida dos votos de anotadores escolhidos.

    É o **controlo positivo** de toda a linha: sem saber se o mecanismo detecta
    um anotador que SABIDAMENTE difere, nenhum nulo dele é interpretável.

    Desenho
    -------
    - `n_per_level` anotadores por nível. O nível **0.0 é o grupo de controlo
      interno** -- escolhido do mesmo modo, votos intocados -- e dá a
      distribuição nula da competência entre seleccionados.
    - Selecção **estratificada por painel linguístico** (proporcional) **e por
      número de votos**: os seleccionados são ordenados por contagem de votos e
      os níveis distribuídos à vez sobre essa ordem. Sem isso, o efeito do
      nível confunde-se com "quem tem poucos votos é mais difícil de detectar".
    - Inverte-se em TREINO E VALIDAÇÃO (a tabela é global). Nesta run, o
      comportamento *verdadeiro* daquela pessoa **é** o corrompido -- um
      spammer real spamma em todo o lado. Corromper só o treino testaria outra
      coisa: um traço aprendido que não existe no teste.

    Porque é este o cenário certo: inverter ao acaso uma fracção `f` dos votos
    move a taxa de YES de `y` para `y + f(1-2y)` -- com `y ~ 0.556` e `f = 0.5`,
    de 0.556 para 0.500, ou seja QUASE NADA. Mas destrói a competência por
    completo. É exactamente onde a ficha tem vantagem estrutural sobre os
    preditores baseados na taxa marginal.
    """
    rng = np.random.default_rng(seed)
    n_ann = len(annotator_ids)
    n_select = len(levels) * n_per_level

    # Contagem de votos calculada AQUI, da tabela completa. Passá-la de fora
    # convidava ao erro de usar a contagem só-de-treino para ordenar por
    # tamanho enquanto a corrupção mexe na tabela inteira.
    n_votes_per_ann = np.zeros(n_ann, dtype=np.int64)
    for anns, votes in vote_table.values():
        for a in anns[votes >= 0]:
            n_votes_per_ann[int(a)] += 1

    # -- selecção: quota por estrato, proporcional ao tamanho ----------------
    chosen = []
    sizes = {k: int((strata == k).sum()) for k in range(len(STRATUM_NAMES))}
    total = sum(sizes.values())
    quotas = {k: int(round(n_select * sizes[k] / total)) for k in sizes}
    # acerta o arredondamento no maior estrato
    biggest = max(sizes, key=lambda k: sizes[k])
    quotas[biggest] += n_select - sum(quotas.values())
    for k, q in quotas.items():
        members = np.flatnonzero((strata == k) & (n_votes_per_ann > 0))
        q = min(q, len(members))
        chosen.extend(rng.choice(members, size=q, replace=False).tolist())
    chosen = np.asarray(sorted(chosen, key=lambda a: (n_votes_per_ann[a], a)))

    # -- níveis distribuídos à vez sobre a ordem por contagem de votos -------
    # Em blocos de `len(levels)`, com a ORDEM DOS NÍVEIS SORTEADA dentro de
    # cada bloco. Sem o sorteio, `levels[i % 6]` faria o nível de 50% cair
    # sempre no anotador com mais votos de cada bloco -- um confound sistemático
    # na direcção de uma curva de detecção artificialmente mais íngreme.
    n_lv = len(levels)
    level_of = {}
    for start in range(0, len(chosen), n_lv):
        block = chosen[start:start + n_lv]
        order = rng.permutation(n_lv)[:len(block)]
        for j, a in enumerate(block):
            level_of[int(a)] = float(levels[order[j]])

    # -- índice inverso anotador -> [(chave, posição)] -----------------------
    where = {int(a): [] for a in chosen}
    for key, (anns, votes) in vote_table.items():
        for pos in range(len(votes)):
            if votes[pos] < 0:
                continue
            a = int(anns[pos])
            if a in where:
                where[a].append((key, pos))

    # -- inversão ------------------------------------------------------------
    new_table = {k: (a.copy(), v.copy()) for k, (a, v) in vote_table.items()}
    entries, n_flipped_total = [], 0
    for a in chosen:
        a = int(a)
        f = level_of[a]
        spots = where[a]
        n_flip = int(round(f * len(spots)))
        picks = rng.choice(len(spots), size=n_flip, replace=False) if n_flip else []
        for p in picks:
            key, pos = spots[int(p)]
            new_table[key][1][pos] = 1 - new_table[key][1][pos]
        n_flipped_total += n_flip
        entries.append({"annotator_id": annotator_ids[a], "annotator_idx": a,
                        "stratum": STRATUM_NAMES[int(strata[a])],
                        "level": f, "n_votes": len(spots), "n_flipped": n_flip})

    n_all_votes = sum(int((v >= 0).sum()) for _, v in vote_table.values())
    manifest = {
        "seed": int(seed), "levels": list(levels), "n_per_level": int(n_per_level),
        "n_selected": len(entries), "n_annotators_total": n_ann,
        "n_votes_flipped": n_flipped_total, "n_votes_total": n_all_votes,
        "frac_votes_flipped": n_flipped_total / max(n_all_votes, 1),
        "entries": entries,
    }
    if verbose:
        print(f"[confusion] CORRUPÇÃO: {len(entries)} anotadores, "
              f"{n_flipped_total} votos invertidos "
              f"({100 * manifest['frac_votes_flipped']:.2f}% de todos os votos)")
        for f in levels:
            sel = [e for e in entries if e["level"] == f]
            print(f"           nível {f:.0%}: {len(sel)} anotadores, "
                  f"{sum(e['n_flipped'] for e in sel)} votos invertidos")
    return new_table, manifest


# ---------------------------------------------------------------------------
# 8. FASE 2 -- a régua: prever votos individuais RETIDOS
# ---------------------------------------------------------------------------
LOGIT_CLIP = 15.0


def _logit(p, clip=LOGIT_CLIP):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.clip(np.log(p / (1.0 - p)), -clip, clip)


def fit_logistic(X, y, max_iter=100, tol=1e-9, clip=LOGIT_CLIP):
    """Regressão logística por IRLS, SEM penalização. -> beta [1 + k].

    `X` entra sem coluna de intercepto; ela é acrescentada aqui. Ajuste em
    numpy, no CPU, em milissegundos: isto é o ADVERSÁRIO da ficha, não uma
    peça do sistema, e nunca toca na rede.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xd.shape[1], dtype=np.float64)
    if y.std() == 0:                      # degenerado: só uma classe
        beta[0] = clip if y.mean() > 0.5 else -clip
        return beta
    for _ in range(max_iter):
        eta = np.clip(Xd @ beta, -clip, clip)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        z = eta + (y - p) / w
        XtW = Xd.T * w
        H = XtW @ Xd + 1e-9 * np.eye(Xd.shape[1])
        try:
            new = np.linalg.solve(H, XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def _apply_logistic(beta, X, clip=LOGIT_CLIP):
    Xd = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
    return 1.0 / (1.0 + np.exp(-np.clip(Xd @ beta, -clip, clip)))


def _score(p_yes, flat_vote):
    """Surpresa média (NLL) e acerto de um preditor de P(voto = YES)."""
    is_yes = flat_vote == VOTE_MAP["YES"]
    p_obs = np.where(is_yes, p_yes, 1.0 - p_yes)
    return {"nll": float(-np.log(np.clip(p_obs, 1e-12, 1.0)).mean()),
            "acc": float(((p_yes > 0.5) == is_yes).mean())}


def make_vote_metric_logger_class(callback_cls, dataloader_cls):
    """Callback que mede, por época, a previsão de VOTOS INDIVIDUAIS RETIDOS.

    Porquê esta régua e não o ICM
    ------------------------------
    O ICM compara dois TREINOS, logo tem por baixo o ruído de seed (sd 0.0059
    medido no A0). Esta régua compara PREDITORES DENTRO DO MESMO TREINO, nos
    mesmos votos: o sorteio é o mesmo dos dois lados e cancela.

    Os votos usados são os dos 598 itens de VALIDAÇÃO -- nunca entraram na
    `L_soft` nem na `L_aux`, e os anotadores são quase todos conhecidos do
    treino. É a pergunta certa: **anotador conhecido, item novo**.

    ⚠ Reter votos de itens de TREINO teria fuga: o alvo de treino desses itens
    é a média dos 6 votos, logo o voto "retirado" já entrou pela porta de trás.

    Preditores (todos sobre os MESMOS votos)
    ----------------------------------------
    - `a`       -- sem modelo de anotador (matriz = identidade)
    - `b`       -- a ficha aprendida
    - `b_perm`  -- a ficha de OUTRA pessoa do mesmo painel linguístico
                   (controlo negativo: se `b` continua a ganhar, a vantagem não
                   vinha de conhecer aquela pessoa)
    - `c`       -- só a taxa de YES da pessoa, constante, ignora o item
    - `d3/d4`   -- fórmula logística de 3 e de 4 ingredientes. **`d4` decide.**

    `d4` acrescenta `|logit(yes_rate) - logit(base)|`. Sem ele, uma fórmula
    LINEAR em `logit(yes_rate)` não consegue representar a forma em U que a
    competência deve absorver -- e a ficha "passaria" justamente por confirmar
    a previsão nula. Cada uma é ajustada duas vezes: no subconjunto de treino
    (**critério**) e na própria validação (**oráculo**, barra mais alta).

    ⚠ MEDIDO no teste T12 (2026-09-10): **`b` já bate `a` NO INIT**, antes de
    aprender fosse o que fosse sobre pessoa nenhuma. Causa: `C` no init é
    `softmax([3, 0])`, com diagonal 0.9526, e isso **encolhe a previsão na
    direcção de 0.5** -- um efeito de calibração global, não de anotador. E
    `s_common` é livre, logo o treino pode afinar essa temperatura à vontade.

    Consequência para o critério: **`b` contra `a` mede calibração global
    misturada com sinal individual e não serve de prova de nada.** O contraste
    limpo do sinal INDIVIDUAL é `b` contra `b_perm`: as duas têm exactamente a
    mesma calibração global (o mesmo `s_common`, o mesmo conjunto de fichas),
    e diferem só em cada pessoa receber a SUA ficha ou a de outra. Reportar
    `b - a` como descritivo; decidir por `b - b_perm` e `b - d4`.

    Determinismo
    ------------
    `_BaseDataLoaderIter.__init__` consome do RNG global do CPU a cada `iter()`.
    Os dois loaders levam gerador PRÓPRIO e `num_workers=0`, e o teste T12
    exige que uma passagem completa deixe `torch.get_rng_state()` intacto. O
    controlo lambda=0 reproduzir o A0 bit-a-bit prova que hoje não há nenhum
    consumidor espúrio; este callback seria o primeiro.

    ⚠ Nada disto entra na perda do Trainer, e o dataset de votos NUNCA é
    passado como `eval_dataset`: `eval_loss` tem de continuar a ser `L_soft`
    pura, porque é o selector de época.
    """

    class VoteMetricLogger(callback_cls):
        def __init__(self, model, val_votes_dataset, train_votes_dataset,
                     ann_stats, perm, out_path, batch_size=8,
                     n_train_fit_items=600, fit_seed=0, use_fp16=True,
                     npz_path=None):
            self.model = model
            self.val_ds = val_votes_dataset
            self.ann_stats = ann_stats
            self.perm = np.asarray(perm, dtype=np.int64)
            self.out_path = out_path
            self.npz_path = npz_path or out_path.replace(".json", "_pervote.npz")
            self.batch_size = int(batch_size)
            self.use_fp16 = bool(use_fp16)
            self.records = []
            self._pervote = {}
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            # Subconjunto FIXO de itens de treino onde (d) é ajustada. Sorteado
            # UMA vez, com um gerador independente do RNG global. 5 parâmetros
            # sobre ~3600 votos chegam e sobram, e o custo fica igual ao de uma
            # passagem de avaliação em vez de 6x.
            n = len(train_votes_dataset)
            k = min(int(n_train_fit_items), n)
            rng = np.random.default_rng(fit_seed)
            idx = np.sort(rng.choice(n, size=k, replace=False))
            from torch.utils.data import Subset
            self.train_ds = Subset(train_votes_dataset, idx.tolist())
            self.train_fit_items = idx.tolist()

        # -- recolha ------------------------------------------------------
        @torch.no_grad()
        def _collect(self, dataset):
            device = next(self.model.parameters()).device
            loader = dataloader_cls(
                dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
                generator=torch.Generator().manual_seed(0),
            )
            logits, anns, votes = [], [], []
            amp = (self.use_fp16 and device.type == "cuda")
            for batch in loader:
                out = None
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16, enabled=amp):
                    out = self.model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        pixel_values=batch["pixel_values"].to(device),
                        demo_vec=batch["demo_vec"].to(device),
                    )
                # O encoder corre em fp16 (como na avaliação do Trainer), mas a
                # aritmética de log-probabilidades é SEMPRE float32.
                logits.append(out[1].detach().float().view(-1))
                anns.append(batch["ann_idx"].to(device))
                votes.append(batch["ann_vote"].to(device))
            return (torch.cat(logits), torch.cat(anns, dim=0),
                    torch.cat(votes, dim=0))

        def _predict_pack(self, dataset):
            """-> dict com p_yes de (a), (b), (b_perm) e os arrays achatados."""
            logits, ann_idx, ann_vote = self._collect(dataset)
            conf = self.model.confusion
            out = {}
            for name, kw in (("a", {"use_matrices": False}),
                             ("b", {"use_matrices": True}),
                             ("b_perm", {"use_matrices": True, "perm": self.perm})):
                logp, p_yes, f_idx, f_vote = conf.vote_logprobs(
                    logits, ann_idx, ann_vote, **kw)
                out[name] = p_yes.detach().float().cpu().numpy().astype(np.float64)
                out["_idx"] = f_idx.detach().cpu().numpy()
                out["_vote"] = f_vote.detach().cpu().numpy()
            return out

        def _features(self, pack):
            idx = pack["_idx"]
            x1 = _logit(pack["a"])                              # prob. do meme
            x2 = _logit(self.ann_stats["yes_rate"][idx])        # taxa da pessoa
            x3 = _logit(self.ann_stats["agree"][idx])           # concordância
            base = _logit(np.full(len(idx), self.ann_stats["panel_yes_rate"]))
            x4 = np.abs(x2 - base)                              # desvio à base
            return np.column_stack([x1, x2, x3]), np.column_stack([x1, x2, x3, x4])

        # -- avaliação ----------------------------------------------------
        def _evaluate(self, state):
            # GUARDA DE RNG, activa em TODAS as runs reais (o T12c só cobre o
            # CPU e um modelo falso). Se o callback deslocar o gerador, a ordem
            # dos batches de treino e o dropout mudam a partir daqui e a run
            # deixa de ser comparável com o A0 -- falhar alto é melhor que
            # descobrir depois de 15 épocas.
            rng_cpu = torch.get_rng_state().clone()
            rng_cuda = (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None)

            was_training = self.model.training
            self.model.eval()
            try:
                val = self._predict_pack(self.val_ds)
                trn = self._predict_pack(self.train_ds)
            finally:
                self.model.train(was_training)

            if not torch.equal(rng_cpu, torch.get_rng_state()):
                raise RuntimeError(
                    "VoteMetricLogger deslocou o RNG do CPU -- a trajectória de "
                    "treino deixou de ser comparável com o A0.")
            if rng_cuda is not None and any(
                    not torch.equal(a, b) for a, b in
                    zip(rng_cuda, torch.cuda.get_rng_state_all())):
                raise RuntimeError(
                    "VoteMetricLogger deslocou o RNG do CUDA -- ver acima.")

            v_vote, t_vote = val["_vote"], trn["_vote"]
            v_y = (v_vote == VOTE_MAP["YES"]).astype(np.float64)
            t_y = (t_vote == VOTE_MAP["YES"]).astype(np.float64)
            V3, V4 = self._features(val)
            T3, T4 = self._features(trn)

            metrics = {k: _score(val[k], v_vote) for k in ("a", "b", "b_perm")}
            metrics["c"] = _score(self.ann_stats["yes_rate"][val["_idx"]], v_vote)

            coefs = {}
            for tag, Xt, Xv in (("d3", T3, V3), ("d4", T4, V4)):
                b_tr = fit_logistic(Xt, t_y)          # CRITÉRIO: ajustada no treino
                b_or = fit_logistic(Xv, v_y)          # ORÁCULO: ajustada na validação
                metrics[f"{tag}_train"] = _score(_apply_logistic(b_tr, Xv), v_vote)
                metrics[f"{tag}_val"] = _score(_apply_logistic(b_or, Xv), v_vote)
                coefs[f"{tag}_train"] = b_tr.tolist()
                coefs[f"{tag}_val"] = b_or.tolist()

            self.records.append({
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "global_step": int(state.global_step),
                "n_votes_val": int(len(v_vote)),
                "n_votes_train_fit": int(len(t_vote)),
                "metrics": metrics,
                "coefs": coefs,
            })
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "n_annotators": int(self.model.confusion.n_annotators),
                    "b_minus_a_is_confounded": (
                        "b bate a já no init por encolhimento (calibração "
                        "global). Decidir por b-b_perm e b-d4_train."),
                    "param": self.model.confusion.param,
                    "freeze_bias": bool(self.model.confusion.freeze_bias),
                    "lam": self.model.lam,
                    "panel_yes_rate": self.ann_stats["panel_yes_rate"],
                    "n_unseen_annotators": self.ann_stats["n_unseen_annotators"],
                    "train_fit_items": self.train_fit_items,
                    "predictors": {
                        "a": "sem modelo de anotador (matriz = identidade)",
                        "b": "ficha aprendida",
                        "b_perm": "ficha de outra pessoa do mesmo painel linguístico",
                        "c": "taxa de YES da pessoa, constante",
                        "d3_train": "logística [logit p_meme, logit yes_rate, logit agree]",
                        "d4_train": "d3 + |logit yes_rate - logit base|  <- CRITÉRIO",
                        "d3_val": "d3 ajustada na validação (oráculo)",
                        "d4_val": "d4 ajustada na validação (oráculo)",
                    },
                    "records": self.records,
                }, f, indent=2, ensure_ascii=False)

            # Probabilidade prevista POR VOTO, guardada em todas as épocas. É o
            # que permite qualquer recorte post-hoc (por anotador, por nível de
            # corrupção) sem voltar a treinar. ~1.3 MB por run.
            ep = len(self.records) - 1
            for k in ("a", "b", "b_perm"):
                self._pervote[f"p_{k}_ep{ep:02d}"] = val[k].astype(np.float32)
            np.savez_compressed(
                self.npz_path,
                ann_idx=val["_idx"], ann_vote=val["_vote"], perm=self.perm,
                **self._pervote,
            )

        def on_evaluate(self, args, state, control, **kwargs):
            self._evaluate(state)

        def on_train_end(self, args, state, control, **kwargs):
            self._evaluate(state)

    return VoteMetricLogger


def soft_labels_from_votes(vote_table):
    """Soft labels da Task A reconstruídas a partir da tabela de votos.

    `-> {id: [P(YES), P(NO)]}`, na convenção do notebook (índice 0 = YES).

    Existe por causa da run B. Se a tabela for corrompida mas a `L_soft`
    continuar a treinar sobre `soft_labels_taskA_raw.json`, o braço fica
    INCONSISTENTE: a camada de confusão recebe um **latente-oráculo limpo** e só
    tem de explicar porque é que 120 pessoas discordam de um consenso que nunca
    foi contaminado. Isso sobrestima a deteção (na prática a soft label É a
    média dos votos corrompidos) e parte a identidade do T2
    (`L_aux(identidade) == BCE contra a soft label`), que é o que permitia ler
    `lambda` como reponderação da Task A.

    Com esta função, a run B passa a ser **um mundo consistente**: estas 120
    pessoas são spammers em todo o lado -- no alvo de treino, no gold de
    validação e nos votos que a régua tenta prever.

    Sobre a tabela LIMPA tem de reproduzir `soft_labels_taskA_raw.json` exacto;
    é o que o teste T14 verifica.
    """
    out = {}
    for key, (_, votes) in vote_table.items():
        valid = votes[votes >= 0]
        n = len(valid)
        if n == 0:
            out[str(key)] = None
            continue
        p_yes = float((valid == VOTE_MAP["YES"]).sum()) / n
        out[str(key)] = [p_yes, 1.0 - p_yes]
    return out
