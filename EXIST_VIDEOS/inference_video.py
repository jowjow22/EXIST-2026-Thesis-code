#!/usr/bin/env python
"""
inference_video.py
------------------
Load the trained EXISTVideoModel, run inference on a dataset split,
and either export 6 PyEvALL-compatible JSON files (test mode) or
print metrics against gold labels (eval mode).

Usage — test:
    python inference_video.py \
        --model-dir   ./checkpoints_seed_2025/best \
        --test-json   ./Dataset/test/EXIST2026_test_clean.json \
        --frames-dir  ./frames/test/ \
        --audio-dir   ./audio_features/test/ \
        --scaler      ./scaler.pkl \
        --output-dir  ./predictions/

Usage — eval against training data:
    python inference_video.py \
        --model-dir   ./checkpoints_seed_2025/best \
        --test-json   ./Dataset/training/EXIST2026_training.json \
        --frames-dir  ./frames/train/ \
        --audio-dir   ./audio_features/train/ \
        --scaler      ./scaler.pkl \
        --eval

Optional tuned Task C thresholds (saved by tune_task_c_thresholds):
    --thresholds-json ./task_c_thresholds_seed_2025.json

NOTE on Task A sigmoid convention (video model):
    sigmoid(logit_a) = p_YES   (opposite of the meme model)
    p_NO = 1 - sigmoid(logit_a)
    This matches compute_metrics in the training notebook:
        prob_yes = 1 / (1 + exp(-logits_a))
        probs_a  = [1 - prob_yes, prob_yes]  →  [p_NO, p_YES]
    and _format_for_pyevall which reads probs[i][1] as p_YES.
"""

import argparse
import json
import os
import pickle
import re
import tempfile

import emoji
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, VideoMAEModel
from wordsegment import segment
import safetensors.torch
import pandas as pd
from sklearn.preprocessing import RobustScaler
from peft import LoraConfig, get_peft_model

# ── Constants (must match training) ─────────────────────────────────────────

LABELS_2_1 = ['YES', 'NO']
LABELS_2_2 = ['NONE', 'JUDGEMENTAL', 'DIRECT']
LABELS_2_3 = ['NONE', 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE',
               'SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'STEREOTYPING-DOMINANCE']

TASK_2_2_HIERARCHY = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
TASK_2_3_HIERARCHY = {"YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
                               "OBJECTIFICATION", "SEXUAL-VIOLENCE",
                               "MISOGYNY-NON-SEXUAL-VIOLENCE"], "NO": []}

NUM_FRAMES    = 16
FRAME_H = FRAME_W = 224
NUM_CHANNELS  = 3

VIDEO_DIM      = 768
AUDIO_RAW_DIM  = 1280
AUDIO_PROJ_DIM = 768
TEXT_DIM       = 1024

BIO_RAW_DIM    = 16
BIO_INPUT_DIM  = BIO_RAW_DIM + 1   # +1 for CVPC
BIO_HIDDEN_DIM = 128

VIEWER_EMB_DIM  = 32
VIEWER_PROJ_DIM = BIO_RAW_DIM

FUSION_INPUT_DIM  = VIDEO_DIM + AUDIO_PROJ_DIM + TEXT_DIM + BIO_HIDDEN_DIM  # 2688
FUSION_HIDDEN_DIM = 1024

VIEWER_TO_IDX = {
    "EN1_2": 0, "EN2_2": 1, "EN3_2": 2, "EN4_2": 3, "EN5_2": 4,
    "ES1_2": 5, "ES2_2": 6, "ES3_2": 7, "ES4_2": 8, "ES5_2": 9,
}
NUM_VIEWERS = len(VIEWER_TO_IDX)

RT_IMPUTE = {"en": 29.0, "es": 17.5}
RT_CLIP   = 120.0


# ── Bio helpers ───────────────────────────────────────────────────────────────

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_bio_vector(sensorial, lang, scaler):
    users   = sensorial["users"]
    et_data = sensorial["modalities"]["ET"]["by_user"]
    hr_data = sensorial["modalities"]["HR"]["by_user"]

    user_vecs = []
    user_ids  = []
    rt_vals   = []
    fix_vals  = []

    for user in users:
        et = et_data[user]
        hr = hr_data[user]

        rt = et.get("reaction_time", None)
        if rt is None or safe_float(rt) != safe_float(rt):
            rt = RT_IMPUTE[lang]
        rt = min(safe_float(rt), RT_CLIP)
        rt_vals.append(rt)
        fix_vals.append(safe_float(et.get("fixations_count"), 0.0))

        vec = np.array([
            safe_float(et.get("3d_eye_states_pupil diameter left [mm]_mean")),
            safe_float(et.get("3d_eye_states_pupil diameter left [mm]_std")),
            safe_float(et.get("3d_eye_states_pupil diameter right [mm]_mean")),
            safe_float(et.get("3d_eye_states_pupil diameter right [mm]_std")),
            safe_float(et.get("blinks_count")),
            safe_float(et.get("blinks_duration_mean_ns")),
            safe_float(et.get("fixations_count")),
            safe_float(et.get("fixations_duration_mean_ns")),
            safe_float(et.get("fixations_duration_std_ns")),
            safe_float(et.get("saccades_count")),
            safe_float(et.get("saccades_duration_mean_ns")),
            rt,
            safe_float(hr.get("garmin_hr_mean")),
            safe_float(hr.get("garmin_hr_std")),
            safe_float(hr.get("garmin_hr_max")),
            safe_float(hr.get("garmin_hr_min")),
        ], dtype=np.float32)

        user_vecs.append(vec)
        user_ids.append(VIEWER_TO_IDX[user])

    v1_scaled = scaler.transform(user_vecs[0].reshape(1, -1)).flatten()
    v2_scaled = scaler.transform(user_vecs[1].reshape(1, -1)).flatten()

    dot   = np.dot(v1_scaled, v2_scaled)
    norms = (np.linalg.norm(v1_scaled) * np.linalg.norm(v2_scaled)) + 1e-8
    cvpc  = float(dot / norms)

    bio_scaled = np.mean([v1_scaled, v2_scaled], axis=0)

    return (
        torch.tensor(bio_scaled,              dtype=torch.float32),  # [16]
        torch.tensor([cvpc],                  dtype=torch.float32),  # [1]
        torch.tensor([float(np.mean(rt_vals))],  dtype=torch.float32),  # [1]
        torch.tensor([float(np.mean(fix_vals))], dtype=torch.float32),  # [1]
        torch.tensor(user_ids,                dtype=torch.long),     # [2]
    )


def collect_bio_vectors(dataframe):
    """Flatten per-user bio vectors for fitting the scaler."""
    vectors = []
    for _, row in dataframe.iterrows():
        sensorial = row["sensorial"]
        lang      = row["lang"]
        et_data   = sensorial["modalities"]["ET"]["by_user"]
        hr_data   = sensorial["modalities"]["HR"]["by_user"]
        for user in sensorial["users"]:
            et = et_data[user]
            hr = hr_data[user]
            rt = safe_float(et.get("reaction_time"), default=float("nan"))
            if rt != rt:
                rt = RT_IMPUTE[lang]
            rt = min(rt, RT_CLIP)
            vec = np.array([
                safe_float(et.get("3d_eye_states_pupil diameter left [mm]_mean")),
                safe_float(et.get("3d_eye_states_pupil diameter left [mm]_std")),
                safe_float(et.get("3d_eye_states_pupil diameter right [mm]_mean")),
                safe_float(et.get("3d_eye_states_pupil diameter right [mm]_std")),
                safe_float(et.get("blinks_count")),
                safe_float(et.get("blinks_duration_mean_ns")),
                safe_float(et.get("fixations_count")),
                safe_float(et.get("fixations_duration_mean_ns")),
                safe_float(et.get("fixations_duration_std_ns")),
                safe_float(et.get("saccades_count")),
                safe_float(et.get("saccades_duration_mean_ns")),
                rt,
                safe_float(hr.get("garmin_hr_mean")),
                safe_float(hr.get("garmin_hr_std")),
                safe_float(hr.get("garmin_hr_max")),
                safe_float(hr.get("garmin_hr_min")),
            ], dtype=np.float32)
            vectors.append(vec)
    return np.stack(vectors)


# ── Frame / audio loaders ─────────────────────────────────────────────────────

def load_frames(frames_dir, item_id):
    path = Path(frames_dir) / f"{item_id}.pt"
    try:
        return torch.load(path, weights_only=True).float()
    except Exception:
        print(f"[WARN] Frames missing for {item_id}, using zeros")
        return torch.zeros(NUM_FRAMES, NUM_CHANNELS, FRAME_H, FRAME_W, dtype=torch.float32)


def load_audio_features(audio_dir, item_id):
    path = Path(audio_dir) / f"{item_id}.pt"
    try:
        return torch.load(path, weights_only=True).float()
    except Exception:
        print(f"[WARN] Audio missing for {item_id}, using zeros")
        return torch.zeros(AUDIO_RAW_DIM, dtype=torch.float32)


# ── Soft label builders (needed for --eval mode) ──────────────────────────────

def build_soft_label_a(labels):
    valid = [l for l in labels if l != "UNKNOWN"]
    if not valid:
        return torch.tensor(-100.0, dtype=torch.float32)
    soft = sum(1 for l in valid if l == "YES") / len(valid)
    return torch.tensor(soft, dtype=torch.float32)


def build_soft_label_b(labels):
    remapped = []
    for l in labels:
        if l == "UNKNOWN":
            continue
        remapped.append("NONE" if l == "-" else l)
    if not remapped:
        return torch.full((len(LABELS_2_2),), -1.0, dtype=torch.float32)
    counts = {lbl: 0 for lbl in LABELS_2_2}
    for l in remapped:
        if l in counts:
            counts[l] += 1
    soft = np.array([counts[lbl] / len(remapped) for lbl in LABELS_2_2], dtype=np.float32)
    return torch.tensor(soft, dtype=torch.float32)


def build_soft_label_c(labels):
    remapped = []
    for ann_labels in labels:
        cleaned = []
        for lbl in ann_labels:
            if lbl == "UNKNOWN":
                continue
            cleaned.append("NONE" if lbl == "-" else lbl)
        if cleaned:
            remapped.append(cleaned)
    if not remapped:
        return torch.full((len(LABELS_2_3),), -1.0, dtype=torch.float32)
    counts = {lbl: 0 for lbl in LABELS_2_3}
    for ann_labels in remapped:
        for lbl in ann_labels:
            if lbl in counts:
                counts[lbl] += 1
    soft = np.array([counts[lbl] / len(remapped) for lbl in LABELS_2_3], dtype=np.float32)
    return torch.tensor(soft, dtype=torch.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class EXISTVideoDataset(Dataset):
    def __init__(self, dataframe, tokenizer, scaler, frames_dir, audio_dir,
                 max_length=128, has_labels=False):
        self.tokenizer  = tokenizer
        self.scaler     = scaler
        self.frames_dir = frames_dir
        self.audio_dir  = audio_dir
        self.max_length = max_length
        self.has_labels = has_labels

        self.item_ids   = dataframe["id_EXIST"].values
        self.langs      = dataframe["lang"].values
        self.texts      = dataframe["text"].values
        self.sensorials = dataframe["sensorial"].values

        if has_labels:
            self.labels_a = dataframe["labels_task3_1"].values
            self.labels_b = dataframe["labels_task3_2"].values
            self.labels_c = dataframe["labels_task3_3"].values

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        item_id   = self.item_ids[idx]
        lang      = self.langs[idx]
        text      = self.texts[idx]
        sensorial = self.sensorials[idx]

        frames    = load_frames(self.frames_dir, item_id)
        audio     = load_audio_features(self.audio_dir, item_id)
        tokenized = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        bio_vec, cvpc, rt_norm, fix_norm, user_ids = build_bio_vector(
            sensorial, lang, self.scaler
        )

        item = {
            "id_EXIST":       str(item_id),
            "frames":         frames,
            "audio":          audio,
            "input_ids":      tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "bio_vec":        bio_vec,
            "cvpc":           cvpc,
            "rt_norm":        rt_norm,
            "fix_norm":       fix_norm,
            "user_ids":       user_ids,
        }

        if self.has_labels:
            item["gold_a"] = build_soft_label_a(self.labels_a[idx])
            item["gold_b"] = build_soft_label_b(self.labels_b[idx])
            item["gold_c"] = build_soft_label_c(self.labels_c[idx])

        return item


# ── Model ─────────────────────────────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class EXISTVideoModel(nn.Module):
    def __init__(self, videomae, xlmr):
        super().__init__()
        self.videomae   = videomae
        self.xlmr       = xlmr
        self.video_pool = TemporalAttentionPool(VIDEO_DIM)
        self.audio_proj = nn.Sequential(
            nn.Linear(AUDIO_RAW_DIM, AUDIO_PROJ_DIM), nn.GELU(), nn.LayerNorm(AUDIO_PROJ_DIM)
        )
        self.viewer_emb  = nn.Embedding(NUM_VIEWERS, VIEWER_EMB_DIM)
        self.viewer_proj = nn.Sequential(nn.Linear(VIEWER_EMB_DIM, VIEWER_PROJ_DIM), nn.Tanh())
        self.bio_ln      = nn.LayerNorm(BIO_RAW_DIM)
        self.bio_mlp     = nn.Sequential(
            nn.Linear(BIO_INPUT_DIM, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, BIO_HIDDEN_DIM), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(FUSION_INPUT_DIM, FUSION_HIDDEN_DIM), nn.GELU(), nn.Dropout(0.1)
        )
        self.head_a = nn.Linear(FUSION_HIDDEN_DIM, 1)
        self.head_b = nn.Linear(FUSION_HIDDEN_DIM, len(LABELS_2_2))
        self.head_c = nn.Linear(FUSION_HIDDEN_DIM, len(LABELS_2_3))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=TEXT_DIM, num_heads=8, batch_first=True, dropout=0.1,
        )
        self.cross_proj = nn.Linear(VIDEO_DIM, TEXT_DIM)
        self.cross_norm = nn.LayerNorm(TEXT_DIM)

    def forward(self, frames, audio, input_ids, attention_mask,
                bio_vec, cvpc, rt_norm, fix_norm, user_ids, **kwargs):
        B      = frames.shape[0]
        device = frames.device
        dtype  = self.fusion[0].weight.dtype

        vid_feat = self.video_pool(
            self.videomae(pixel_values=frames).last_hidden_state
        )                                                              # [B, 768]
        aud_feat = self.audio_proj(audio)                              # [B, 768]
        txt_feat = self.xlmr(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]                                   # [B, 1024]

        base1       = self.viewer_proj(self.viewer_emb(user_ids[:, 0]))
        base2       = self.viewer_proj(self.viewer_emb(user_ids[:, 1]))
        bio_centered = bio_vec - (base1 + base2) / 2
        bio_norm    = self.bio_ln(bio_centered)
        bio_input   = torch.cat([bio_norm, cvpc], dim=-1)
        bio_feat    = self.bio_mlp(bio_input)                          # [B, 128]

        vid_proj  = self.cross_proj(vid_feat).unsqueeze(1)
        txt_q     = txt_feat.unsqueeze(1)
        cross_out, _ = self.cross_attn(query=txt_q, key=vid_proj, value=vid_proj)
        txt_feat  = self.cross_norm(txt_feat + cross_out.squeeze(1))
        fused    = self.fusion(
            torch.cat([vid_feat, aud_feat, txt_feat, bio_feat], dim=-1)
        )                                                       # [B, 1024]
        logits_a = self.head_a(fused).squeeze(-1)                      # [B]
        logits_b = self.head_b(fused)                                  # [B, 3]
        logits_c = self.head_c(fused)                                  # [B, 6]
        return logits_a, logits_b, logits_c


# ── JSON formatting ───────────────────────────────────────────────────────────
#
# Task A convention (video model, matches _format_for_pyevall in training):
#   sigmoid(logit_a) = p_YES
#   p_NO = 1 - p_YES
#   probs_a stored as [p_NO, p_YES]  →  probs[i][0]=p_NO, probs[i][1]=p_YES
#
def format_records(sample_ids, all_probs_a, all_probs_b, all_probs_c,
                   task, evaluation, thresholds_c=None, test_case="EXIST2026"):
    """
    all_probs_a: list of (p_NO, p_YES) tuples
    all_probs_b: list of [p_NONE, p_JUDGEMENTAL, p_DIRECT]
    all_probs_c: list of [p_NONE, p_IDEOLOGICAL, ...]
    """
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": test_case, "id": str(sid)}

        if task == '2_1':
            p_no, p_yes = all_probs_a[i]
            if evaluation == 'hard':
                entry["value"] = "YES" if p_yes >= p_no else "NO"
            else:
                entry["value"] = {"YES": round(p_yes, 6), "NO": round(p_no, 6)}

        elif task == '2_2':
            label_map = {"NONE": "NO", "JUDGEMENTAL": "JUDGEMENTAL", "DIRECT": "DIRECT"}
            probs = {label_map[lbl]: float(all_probs_b[i][idx])
                     for idx, lbl in enumerate(LABELS_2_2)}
            if evaluation == 'hard':
                entry["value"] = max(probs, key=probs.get)
            else:
                entry["value"] = {k: round(v, 6) for k, v in probs.items()}

        elif task == '2_3':
            label_map = {"NONE": "NO"}
            label_map.update({lbl: lbl for lbl in LABELS_2_3[1:]})
            probs = {label_map[lbl]: float(all_probs_c[i][idx])
                     for idx, lbl in enumerate(LABELS_2_3)}
            if evaluation == 'hard':
                threshs    = thresholds_c if thresholds_c is not None else [0.5] * len(LABELS_2_3)
                thresh_map = {label_map[lbl]: threshs[idx] for idx, lbl in enumerate(LABELS_2_3)}
                hard_labels = [lbl for lbl, p in probs.items() if p >= thresh_map[lbl]]
                if not hard_labels:
                    hard_labels = [max(probs, key=probs.get)]
                entry["value"] = hard_labels
            else:
                entry["value"] = {k: round(v, 6) for k, v in probs.items()}

        out.append(entry)
    return out


def save_json(records, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(records)} records → {path}")


# ── Eval helpers ──────────────────────────────────────────────────────────────

def _gold_soft_records(sample_ids, gold_dist, task, test_case):
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": test_case, "id": str(sid)}
        if task == '2_1':
            g = gold_dist[i]   # scalar p_YES
            entry["value"] = {"YES": round(float(g), 6), "NO": round(1.0 - float(g), 6)}
        elif task == '2_2':
            g = gold_dist[i]
            label_map = {"NONE": "NO", "JUDGEMENTAL": "JUDGEMENTAL", "DIRECT": "DIRECT"}
            entry["value"] = {label_map[lbl]: round(float(g[idx]), 6)
                              for idx, lbl in enumerate(LABELS_2_2)}
        elif task == '2_3':
            g = gold_dist[i]
            label_map = {"NONE": "NO"}
            label_map.update({lbl: lbl for lbl in LABELS_2_3[1:]})
            entry["value"] = {label_map[lbl]: round(float(g[idx]), 6)
                              for idx, lbl in enumerate(LABELS_2_3)}
        out.append(entry)
    return out


def _gold_hard_records(sample_ids, gold_dist, task, test_case):
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": test_case, "id": str(sid)}
        if task == '2_1':
            g = float(gold_dist[i])
            entry["value"] = "YES" if g >= 0.5 else "NO"
        elif task == '2_2':
            g = gold_dist[i]
            label_map = {"NONE": "NO", "JUDGEMENTAL": "JUDGEMENTAL", "DIRECT": "DIRECT"}
            best = LABELS_2_2[int(np.argmax(g))]
            entry["value"] = label_map[best]
        elif task == '2_3':
            g = gold_dist[i]
            label_map = {"NONE": "NO"}
            label_map.update({lbl: lbl for lbl in LABELS_2_3[1:]})
            hard_labels = [label_map[LABELS_2_3[j]] for j, p in enumerate(g) if p >= 0.5]
            if not hard_labels:
                hard_labels = [label_map[LABELS_2_3[int(np.argmax(g))]]]
            entry["value"] = hard_labels
        out.append(entry)
    return out


def _run_pyevall(pred_records, gold_records, task, evaluation, temp_dir):
    from pyevall.evaluation import PyEvALLEvaluation
    from pyevall.utils.utils import PyEvALLUtils

    pred_path = os.path.join(temp_dir, f'pred_{task}_{evaluation}.json')
    gold_path = os.path.join(temp_dir, f'gold_{task}_{evaluation}.json')
    with open(pred_path, 'w') as f: json.dump(pred_records, f)
    with open(gold_path, 'w') as f: json.dump(gold_records, f)

    if evaluation == 'hard':
        metrics = ['ICM', 'ICMNorm', 'FMeasure']
    else:
        metrics = ['ICMSoft', 'ICMSoftNorm'] + (['CrossEntropy'] if task != '2_3' else [])

    params = {}
    if task == '2_2':
        params[PyEvALLUtils.PARAM_HIERARCHY] = TASK_2_2_HIERARCHY
    elif task == '2_3':
        params[PyEvALLUtils.PARAM_HIERARCHY] = TASK_2_3_HIERARCHY

    try:
        report = PyEvALLEvaluation().evaluate(pred_path, gold_path, metrics, **params)
        results = {}
        if hasattr(report, 'report') and 'metrics' in report.report:
            for m in metrics:
                entry = report.report['metrics'].get(m, {})
                if entry.get('status') == 'OK':
                    try:
                        results[m] = float(entry['results']['average_per_test_case'])
                    except (KeyError, TypeError):
                        pass
        return results
    except Exception as e:
        print(f"    PyEvALL error ({task} {evaluation}): {e}")
        return {}


def print_eval_results(all_ids, all_probs_a, all_probs_b, all_probs_c,
                       all_gold_a, all_gold_b, all_gold_c,
                       thresholds_c, test_case):
    gold_a = np.array(all_gold_a)   # [N]    — scalar p_YES per sample
    gold_b = np.array(all_gold_b)   # [N, 3]
    gold_c = np.array(all_gold_c)   # [N, 6]

    mask_a = gold_a != -100.0
    mask_b = gold_b[:, 0] >= 0
    mask_c = gold_c[:, 0] >= 0

    def filt(lst, mask): return [lst[i] for i in range(len(lst)) if mask[i]]

    ids_a = filt(all_ids, mask_a);  ids_b = filt(all_ids, mask_b);  ids_c = filt(all_ids, mask_c)
    pa    = filt(all_probs_a, mask_a)   # (p_NO, p_YES)
    pb    = filt(all_probs_b, mask_b)
    pc    = filt(all_probs_c, mask_c)
    ga    = gold_a[mask_a].tolist()     # scalar p_YES values
    gb    = gold_b[mask_b].tolist()
    gc    = gold_c[mask_c].tolist()

    print(f"\n  Samples — Task 2_1: {len(ids_a)}  |  Task 2_2: {len(ids_b)}  |  Task 2_3: {len(ids_c)}")

    with tempfile.TemporaryDirectory() as tmp:
        scores = {}
        for task, sids, probs, gold in [('2_1', ids_a, pa, ga),
                                         ('2_2', ids_b, pb, gb),
                                         ('2_3', ids_c, pc, gc)]:
            pa_ = probs if task == '2_1' else None
            pb_ = probs if task == '2_2' else None
            pc_ = probs if task == '2_3' else None

            pred_soft = format_records(sids, pa_, pb_, pc_, task, 'soft', test_case=test_case)
            pred_hard = format_records(sids, pa_, pb_, pc_, task, 'hard',
                                       thresholds_c=thresholds_c, test_case=test_case)
            gold_soft = _gold_soft_records(sids, gold, task, test_case)
            gold_hard = _gold_hard_records(sids, gold, task, test_case)

            scores[f'{task}_soft'] = _run_pyevall(pred_soft, gold_soft, task, 'soft', tmp)
            scores[f'{task}_hard'] = _run_pyevall(pred_hard, gold_hard, task, 'hard', tmp)

    print("\n" + "=" * 60)
    print("EVAL RESULTS (against training labels)")
    print("=" * 60)
    for key, result in scores.items():
        parts   = key.split('_')
        task_id = f"{parts[0]}_{parts[1]}"
        mode    = parts[2]
        print(f"\n  Task {task_id}  [{mode}]")
        if result:
            for metric, val in result.items():
                print(f"    {metric:<20} {val:.4f}")
        else:
            print("    (no results — check PyEvALL error above)")
    print("=" * 60)

    # Sanity: Task A accuracy
    if pa:
        pred_hard_labels = ["YES" if p_yes >= p_no else "NO" for p_no, p_yes in pa]
        gold_hard_labels = ["YES" if g >= 0.5 else "NO" for g in ga]
        acc = sum(p == g for p, g in zip(pred_hard_labels, gold_hard_labels)) / len(pred_hard_labels)
        print(f"\n  Sanity — Task 2_1 accuracy (threshold 0.5): {acc:.4f}")
        print(f"  Predicted YES: {pred_hard_labels.count('YES')}/{len(pred_hard_labels)}"
              f"  |  Gold YES: {gold_hard_labels.count('YES')}/{len(gold_hard_labels)}\n")


# ── Core run ──────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Scaler ────────────────────────────────────────────────────────────────
    if args.scaler:
        print(f"Loading scaler from {args.scaler}...")
        with open(args.scaler, 'rb') as f:
            scaler = pickle.load(f)
    else:
        # Fit scaler on the provided dataset (useful when no saved scaler exists)
        print("No scaler provided — fitting RobustScaler on the dataset bio vectors...")
        raw_for_scaler = pd.read_json(args.test_json).transpose().reset_index(drop=True)
        scaler = RobustScaler().fit(collect_bio_vectors(raw_for_scaler))
        print("Scaler fitted.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        args.model_dir if os.path.isdir(args.model_dir) else "xlm-roberta-large"
    )

    # ── Backbone models with LoRA ─────────────────────────────────────────────
    print("Building model backbones...")
    videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["query", "value"])
    videomae = get_peft_model(videomae, lora_cfg)

    xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-large")
    lora_cfg_text = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["query", "value"])
    xlmr = get_peft_model(xlmr, lora_cfg_text)

    model = EXISTVideoModel(videomae, xlmr)

    # ── Load weights ──────────────────────────────────────────────────────────
    weights_path = os.path.join(args.model_dir, "model.safetensors")
    print(f"Loading weights from {weights_path}...")
    state_dict = safetensors.torch.load_file(weights_path, device=str(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Thresholds ────────────────────────────────────────────────────────────
    thresholds_c = None
    if args.thresholds_json:
        with open(args.thresholds_json) as f:
            thresh_data = json.load(f)
        thresholds_c = thresh_data["thresholds"]
        print(f"Task C thresholds loaded: {thresholds_c}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading data from {args.test_json}...")
    raw = pd.read_json(args.test_json).transpose().reset_index(drop=True)

    dataset = EXISTVideoDataset(
        raw, tokenizer, scaler,
        frames_dir=args.frames_dir,
        audio_dir=args.audio_dir,
        has_labels=args.eval,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )
    print(f"Total samples: {len(dataset)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_ids     = []
    all_probs_a = []   # (p_NO, p_YES) per sample
    all_probs_b = []
    all_probs_c = []
    all_gold_a  = []
    all_gold_b  = []
    all_gold_c  = []

    with torch.no_grad():
        for batch in loader:
            ids = batch.pop('id_EXIST')

            # pull gold before moving to device (they stay on CPU)
            gold_a = batch.pop('gold_a', None)
            gold_b = batch.pop('gold_b', None)
            gold_c = batch.pop('gold_c', None)

            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            logits_a, logits_b, logits_c = model(**batch)

            # sigmoid(logit_a) = p_YES  (video model convention)
            p_yes = torch.sigmoid(logits_a).cpu().tolist()
            p_no  = [1.0 - p for p in p_yes]

            all_ids.extend(ids)
            all_probs_a.extend(list(zip(p_no, p_yes)))   # (p_NO, p_YES)
            all_probs_b.extend(F.softmax(logits_b.float(), dim=-1).cpu().tolist())
            all_probs_c.extend(torch.sigmoid(logits_c.float()).cpu().tolist())

            if args.eval:
                all_gold_a.extend(gold_a.tolist())
                all_gold_b.extend(gold_b.tolist())
                all_gold_c.extend(gold_c.tolist())

    print(f"Inference complete — {len(all_ids)} samples.")

    # ── Eval mode ─────────────────────────────────────────────────────────────
    if args.eval:
        print_eval_results(all_ids, all_probs_a, all_probs_b, all_probs_c,
                           all_gold_a, all_gold_b, all_gold_c,
                           thresholds_c, args.test_case)
        return

    # ── Test mode: export JSON files ──────────────────────────────────────────
    tc = args.test_case
    print("Writing output files...")
    tn = args.team_name
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c,
                             '2_1', 'soft', test_case=tc),
              os.path.join(args.output_dir, f'task3_1_soft_{tn}_1.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c,
                             '2_1', 'hard', test_case=tc),
              os.path.join(args.output_dir, f'task3_1_hard_{tn}_1.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c,
                             '2_2', 'soft', test_case=tc),
              os.path.join(args.output_dir, f'task3_2_soft_{tn}_1.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c,
                             '2_2', 'hard', test_case=tc),
              os.path.join(args.output_dir, f'task3_2_hard_{tn}_1.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c,
                             '2_3', 'soft', test_case=tc),
              os.path.join(args.output_dir, f'task3_3_soft_{tn}_1.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c,
                             '2_3', 'hard', thresholds_c=thresholds_c, test_case=tc),
              os.path.join(args.output_dir, f'task3_3_hard_{tn}_1.json'))
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir',       required=True,
                   help='Directory with model.safetensors (and optionally tokenizer files)')
    p.add_argument('--test-json',       required=True,
                   help='Path to EXIST2026 JSON (training or test)')
    p.add_argument('--frames-dir',      required=True,
                   help='Directory with pre-extracted frame tensors ({id}.pt)')
    p.add_argument('--audio-dir',       required=True,
                   help='Directory with pre-extracted Whisper features ({id}.pt)')
    p.add_argument('--scaler',          default=None,
                   help='Path to fitted RobustScaler pickle (scaler.pkl). '
                        'If omitted, scaler is fitted on the provided dataset.')
    p.add_argument('--output-dir',      default='./predictions')
    p.add_argument('--eval',            action='store_true',
                   help='Evaluate against gold labels instead of exporting JSONs')
    p.add_argument('--test-case',       default='EXIST2026')
    p.add_argument('--batch-size',      type=int, default=4)
    p.add_argument('--num-workers',     type=int, default=2)
    p.add_argument('--thresholds-json', default=None,
                   help='Path to task_c_thresholds_seed_*.json saved by tune_task_c_thresholds()')
    p.add_argument('--team-name', default='Jow',
               help='Team name used in output filenames.')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())