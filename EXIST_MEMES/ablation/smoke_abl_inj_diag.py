"""Smoke test do diagnóstico de colapso -- não carrega XLM-R/CLIP.
Usa um modelo-brinquedo com a MESMA assinatura de saída do MultiHeadModel."""
import os, sys, json, tempfile
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import types
# pyevall não está instalado localmente (só na VM). O diagnóstico não usa PyEvALL --
# stub só para o import de topo de memes_core passar.
# Stub SÓ para estes smokes: eles não chamam PyEvALL, mas o import de topo de
# memes_core exige o módulo. `setdefault` não sobrescreve um pyevall real já
# importado -- ainda assim, NUNCA copiar este padrão para código de treino.
for name in ("pyevall", "pyevall.evaluation", "pyevall.utils", "pyevall.utils.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyevall.evaluation"].PyEvALLEvaluation = object
sys.modules["pyevall.utils.utils"].PyEvALLUtils = object
import memes_core as core

class ToyModel(nn.Module):
    def __init__(self, collapse):
        super().__init__(); self.collapse = collapse
    def forward(self, input_ids=None, demo_vec=None, **kw):
        B = input_ids.shape[0]
        if self.collapse:                      # mesma dist. para todo item
            logits = torch.zeros(B, 6) + torch.tensor([2.0, -1.0, -1.5, -2.0, -1.2, -0.8])
        else:                                  # discrimina por item
            logits = torch.randn(B, 6) * 3.0
        return (torch.tensor(0.0), torch.zeros(B, 1), torch.zeros(B, 3), logits)

torch.manual_seed(0)
N = 40
loader = [{"input_ids": torch.zeros(8, 4, dtype=torch.long),
           "demo_vec": torch.zeros(8, 66),
           "labels_c": torch.zeros(8, 6)} for _ in range(N // 8)]

# taxa base do "treino": inclui None e sentinela -1.0, que devem ser EXCLUÍDOS
rng = np.random.default_rng(0)
train_c = []
target_rate = np.array([0.88, 0.27, 0.18, 0.12, 0.23, 0.31])
for _ in range(500):
    train_c.append((rng.random(6) < target_rate).astype(np.float32))
train_c += [None, np.full(6, -1.0, dtype=np.float32)]

out = os.path.join(tempfile.mkdtemp(), "sub", "task_c_collapse_toy_seed42.json")
for collapse in (True, False):
    d = core.diagnose_task_c_collapse(
        ToyModel(collapse), loader, torch.device("cpu"), out,
        train_labels_c=train_c, config_name=f"toy_collapse={collapse}", seed=42)
    assert os.path.exists(out), "JSON não gravado"
    assert d["collapsed"] is collapse, f"veredito errado: {d['collapsed']} != {collapse}"
    assert d["n_val_items"] == N
    assert not any(np.isnan(d["train_base_rate_per_class"]))

# sem train_labels_c -> inconclusivo, nunca "OK" falso
d = core.diagnose_task_c_collapse(ToyModel(True), loader, torch.device("cpu"), out,
                                  config_name="toy_no_train", seed=42)
assert d["collapsed"] is None and "INCONCLUSIVO" in d["verdict"]

# base rate ignora None/sentinela
br = core.compute_train_base_rate_task_c(train_c)
assert np.allclose(br, (np.stack([r for r in train_c[:500]]) > 0).mean(axis=0))
print("\nSMOKE OK: veredito, JSON persistente, base rate sem sentinelas, cosseno.")
