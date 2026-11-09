import os, sys, json, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch.nn as nn
torch.set_num_threads(4)

def breakdown(model, label):
    rows = collections.OrderedDict()
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        if top in ("videomae", "xlmr"):
            sub = "lora" if "lora_" in n else ("base")
            key = f"{top}.{sub}"
        else:
            key = top
        r = rows.setdefault(key, [0, 0])
        r[0] += p.numel(); r[1] += p.numel() if p.requires_grad else 0
    tot = sum(p.numel() for p in model.parameters()); tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {label}: total={tot:,} trainable={tr:,} frozen={tot-tr:,} ({100*tr/tot:.3f}%)")
    for k, (a, b) in rows.items():
        print(f"   {k:28s} total={a:>13,}  trainable={b:>12,}  frozen={a-b:>13,}")
    return tot, tr

which = sys.argv[1] if len(sys.argv) > 1 else "all"

if which in ("all", "submitted"):
    import inference_video as iv   # = arquitetura do Jow_3 (md5 == .ipynb_checkpoints/inference_video-checkpoint.py)
    from transformers import VideoMAEModel, XLMRobertaModel
    from peft import LoraConfig, get_peft_model
    cfg = lambda: LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["query", "value"])
    vm = get_peft_model(VideoMAEModel.from_pretrained("MCG-NJU/videomae-base"), cfg())
    xl = get_peft_model(XLMRobertaModel.from_pretrained("xlm-roberta-large"), cfg())
    m = iv.EXISTVideoModel(vm, xl)
    breakdown(m, "SUBMETIDA Jow_3 (inference_video.EXISTVideoModel, com cross_attn)")
    for name in ["video_pool", "audio_proj", "viewer_emb", "viewer_proj", "bio_ln", "bio_mlp", "fusion", "head_a", "head_b", "head_c", "cross_attn", "cross_proj", "cross_norm"]:
        print(f"   [{name}] {sum(p.numel() for p in getattr(m, name).parameters()):,}")
    # variante descrita no paper (Thesis_video.ipynb, 2592, sem bio_ln / pool / cross_attn) — só cabeças/novas camadas
    class Old(nn.Module):
        def __init__(s):
            super().__init__()
            s.audio_proj = nn.Linear(1280, 768); s.viewer_emb = nn.Embedding(10, 32); s.viewer_proj = nn.Linear(32, 16)
            s.bio_mlp = nn.Sequential(nn.Linear(17, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 32))
            s.fusion = nn.Sequential(nn.Linear(768 + 768 + 1024 + 32, 1024), nn.GELU(), nn.Dropout(0.1))
            s.head_a = nn.Linear(1024, 1); s.head_b = nn.Linear(1024, 3); s.head_c = nn.Linear(1024, 6)
    o = Old(); n_old = sum(p.numel() for p in o.parameters())
    lora = sum(p.numel() for n, p in m.named_parameters() if "lora_" in n)
    print(f"\n=== Thesis_video.ipynb (paper, 2592): camadas novas={n_old:,} + LoRA={lora:,} => treinavel={n_old+lora:,}")
    for k, v in o.named_children(): print(f"   [{k}] {sum(p.numel() for p in v.parameters()):,}")
    del m, vm, xl

if which in ("all", "b15"):
    import video_core as vc
    m = vc.build_model(42, use_bio=True, use_viewer_emb=True, restore_biases=True)
    breakdown(m, "CORRIGIDA B15 (video_core.build_model, B2)")
    for name in ["video_pool", "audio_proj", "viewer_emb", "viewer_proj", "bio_mlp", "fusion", "head_a", "head_b", "head_c"]:
        print(f"   [{name}] {sum(p.numel() for p in getattr(m, name).parameters()):,}")
    q = m.videomae.base_model.model.encoder.layer[0].attention.attention.query.bias
    print("   q_bias L2 camada0 apos restore:", float(q.norm()))
    print("   fusion_in:", m.fusion_in)
    del m

if which in ("all", "clip"):
    import video_core as vc, video_core_clip as vcc
    m = vcc.build_model_clip(42, use_bio=True, use_viewer_emb=True)
    breakdown(m, "CORRIGIDA CLIP (video_core_clip.build_model_clip, B2)")
    clip_tr = sum(p.numel() for n, p in m.named_parameters() if n.startswith("videomae.") and p.requires_grad)
    clip_tot = sum(p.numel() for n, p in m.named_parameters() if n.startswith("videomae."))
    print(f"   CLIP vision total={clip_tot:,} trainable(4 ultimas camadas)={clip_tr:,}")
    print("   post_layernorm treinavel?", any(p.requires_grad for p in m.videomae.clip.vision_model.post_layernorm.parameters()) if hasattr(m.videomae.clip, 'vision_model') else 'n/a')
    for name in ["video_pool", "fusion"]:
        print(f"   [{name}] {sum(p.numel() for p in getattr(m, name).parameters()):,}")
    print("   fusion_in:", m.fusion_in)
    mean, std = vcc.clip_norm_constants(); print("   CLIP mean/std:", mean.flatten().tolist(), std.flatten().tolist())
