"""Smoke da arquitetura nos 3 demo_injection_mode, com XLM-R/CLIP stubados.
Verifica shapes, no-op sem demografia e equivalência numérica de "current"."""
import os
import sys, types, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Stub SÓ para estes smokes: eles não chamam PyEvALL, mas o import de topo de
# memes_core exige o módulo. `setdefault` não sobrescreve um pyevall real já
# importado -- ainda assim, NUNCA copiar este padrão para código de treino.
for name in ("pyevall", "pyevall.evaluation", "pyevall.utils", "pyevall.utils.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyevall.evaluation"].PyEvALLEvaluation = object
sys.modules["pyevall.utils.utils"].PyEvALLUtils = object
import memes_core as core

class Cfg:
    def __init__(self, h): self.hidden_size = h
class FakeEncoder(nn.Module):
    """Devolve last_hidden_state [B, T, H] -- só o CLS (posição 0) é usado."""
    def __init__(self, h):
        super().__init__(); self.config = Cfg(h); self.lin = nn.Linear(h, h)
        self.encoder = types.SimpleNamespace(layers=nn.ModuleList([nn.Linear(4, 4) for _ in range(24)]))
    @classmethod
    def make(cls, h): return lambda *a, **k: cls(h)
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        src = input_ids if input_ids is not None else pixel_values
        B = src.shape[0]
        x = torch.arange(B * 8 * self.config.hidden_size, dtype=torch.float32)
        x = (x.reshape(B, 8, self.config.hidden_size) % 7) / 7.0
        return types.SimpleNamespace(last_hidden_state=self.lin(x))

core.XLMRobertaModel = types.SimpleNamespace(from_pretrained=FakeEncoder.make(1024))
core.CLIPVisionModel = types.SimpleNamespace(from_pretrained=FakeEncoder.make(1024))

B, D = 4, 66
ids = torch.ones(B, 8, dtype=torch.long); att = torch.ones(B, 8, dtype=torch.long)
pix = torch.ones(B, 3, 4, 4); demo = torch.rand(B, D); demo = demo / demo.sum(1, keepdim=True)

def build(mode, use_demo=True, seed=42):
    torch.manual_seed(seed)
    m = core.MultiHeadModel(num_labels_task_b=3, num_labels_task_c=6, demo_vec_dim=D,
                            use_demographics=use_demo, demo_injection_mode=mode)
    m.eval(); return m

for mode, expected in (("current", 1024), ("renorm", 1024), ("head_inject", 1536)):
    m = build(mode)
    assert m.head_input_dim == expected, (mode, m.head_input_dim)
    loss, la, lb, lc = m(input_ids=ids, attention_mask=att, pixel_values=pix, demo_vec=demo)
    assert la.shape == (B, 1) and lb.shape == (B, 3) and lc.shape == (B, 6)
    has_pn = hasattr(m, "demo_post_norm")
    assert has_pn == (mode == "renorm"), mode
    print(f"{mode:<12} head_in={m.head_input_dim:<5} demo_post_norm={has_pn}  logits_c ok")

# "current" == código antigo: a soma pós-LayerNorm, sem módulo extra
cur = build("current")
extra = set(build("renorm").state_dict()) - set(cur.state_dict())
assert extra == {"demo_post_norm.weight", "demo_post_norm.bias"}, extra
print("renorm acrescenta exatamente:", sorted(extra))

# pesos compartilhados entre current e renorm são bit-a-bit iguais (LayerNorm não consome RNG)
ren = build("renorm")
diffs = [k for k in cur.state_dict() if not torch.equal(cur.state_dict()[k], ren.state_dict()[k])]
assert not diffs, diffs[:5]
print("current vs renorm: todos os pesos comuns bit-a-bit iguais ->", len(cur.state_dict()), "tensores")

# no-op sem demografia: os 3 modos dão logits idênticos e nenhum módulo de demo
outs = []
for mode in ("current", "renorm", "head_inject"):
    m = build(mode, use_demo=False)
    assert m.head_input_dim == 1024 and not hasattr(m, "demo_proj")
    outs.append(m(input_ids=ids, attention_mask=att, pixel_values=pix, demo_vec=demo)[3])
assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
print("use_demographics=False: 3 modos no-op, logits idênticos")

# head_inject sem demo_vec em runtime: slot de zeros, sem erro de shape
m = build("head_inject")
lc = m(input_ids=ids, attention_mask=att, pixel_values=pix, demo_vec=None)[3]
assert lc.shape == (B, 6)
print("head_inject com demo_vec=None: fallback de zeros ok")

try:
    build("bogus")
except ValueError as e:
    print("modo inválido rejeitado:", str(e)[:60])
print("\nSMOKE ARQ OK")
