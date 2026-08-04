#!/usr/bin/env python
"""
inference.py
------------
Load the trained MultiHeadModel and either:
  (a) run inference on the test set and export 6 PyEvALL JSON files, or
  (b) run --eval on the training set to check predictions against gold labels.

Usage — test (default):
    python inference.py \
        --model-dir  ./best_model \
        --test-json  ./Dataset/memes/test/EXIST2026_test.json \
        --image-dir  ./Dataset/memes/test/ \
        --output-dir ./predictions

Usage — eval against training data:
    python inference.py \
        --model-dir  ./best_model \
        --test-json  ./Dataset/memes/training/EXIST2026_training.json \
        --image-dir  ./Dataset/memes/training/ \
        --eval

Optional Task-C threshold overrides (default 0.5 for all):
    --thresholds 0.5 0.35 0.40 0.30 0.45 0.40
    (order: NONE IDEOLOGICAL-INEQUALITY MISOGYNY-NON-SEXUAL-VIOLENCE
            SEXUAL-VIOLENCE OBJECTIFICATION STEREOTYPING-DOMINANCE)

NOTE on Task 2_1 sigmoid convention
-------------------------------------
The model is trained with BCEWithLogitsLoss against labels_a[:, 1] which is
p(NO) (LABELS_2_1 = ['YES','NO'], index 1 = NO).
Therefore:  sigmoid(logit) = p_NO,  p_YES = 1 - sigmoid(logit).
"""

import argparse
import json
import os
import re
import tempfile

import emoji
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, CLIPProcessor, CLIPVisionModel
from wordsegment import segment
import safetensors.torch
import pandas as pd

# ── Label definitions ────────────────────────────────────────────────────────

LABELS_2_1 = ['YES', 'NO']
LABELS_2_2 = ['NONE', 'JUDGEMENTAL', 'DIRECT']
LABELS_2_3 = ['NONE', 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE',
               'SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'STEREOTYPING-DOMINANCE']

TASK_2_2_HIERARCHY = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
TASK_2_3_HIERARCHY = {"YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
                               "OBJECTIFICATION", "SEXUAL-VIOLENCE",
                               "MISOGYNY-NON-SEXUAL-VIOLENCE"], "NO": []}

DEMO_ETHNICITY_VALS = sorted(['Asian', 'Black or African American', 'Hispano or Latino',
                               'Middle Eastern', 'Multiracial', 'White or Caucasian', 'other'])
DEMO_STUDY_VALS     = sorted(["Bachelor's degree", 'Doctorate',
                               'High school degree or equivalent',
                               'Less than high school diploma', "Master's degree", 'other'])
DEMO_COUNTRY_VALS   = sorted(['Afghanistan', 'Argentina', 'Australia', 'Austria', 'Azerbaijan',
                               'Bangladesh', 'Belgium', 'Brazil', 'Bulgaria', 'Canada', 'Chile',
                               'China', 'Colombia', 'Croatia', 'Cuba', 'Czech Republic',
                               'Dominican Republic', 'Ecuador', 'Estonia', 'Finland', 'France',
                               'Germany', 'Greece', 'Hungary', 'India', 'Ireland', 'Israel',
                               'Italy', 'Latvia', 'Mexico', 'Morocco', 'Netherlands',
                               'New Zealand', 'Nicaragua', 'Nigeria', 'Peru', 'Poland',
                               'Portugal', 'Romania', 'Russian Federation', 'Serbia', 'Slovenia',
                               'South Africa', 'Spain', 'Sweden', 'Turkey', 'United Arab Emirates',
                               'United Kingdom', 'United States', 'Uruguay', 'Venezuela',
                               'Viet Nam', 'Zimbabwe'])
DEMO_FEATURE_GROUPS = [
    ('ethnicities_annotators',  DEMO_ETHNICITY_VALS),
    ('study_levels_annotators', DEMO_STUDY_VALS),
    ('countries_annotators',    DEMO_COUNTRY_VALS),
]
DEMO_VEC_DIM = len(DEMO_ETHNICITY_VALS) + len(DEMO_STUDY_VALS) + len(DEMO_COUNTRY_VALS)


# ── Text preprocessing (verbatim from training pipeline) ────────────────────

def replace_user_mentions(text):
    return re.sub(r'@\w+', '<USER_MENTION>', text)

def replace_urls(text):
    return re.sub(r'http\S+|www\S+', '<URL>', text)

def normalize_hashtags(text):
    def process_hashtag(match):
        tag = match.group()[1:]
        if "_" in tag:
            words = tag.split("_")
        elif re.search(r'[A-Z]', tag):
            words = re.sub(r'(?<!^)(?=[A-Z])', ' ', tag).split()
        else:
            try:
                words = segment(tag)
            except ValueError:
                words = [tag]
        return " ".join(words)
    return re.sub(r'#\w+', process_hashtag, text)

def convert_emojis_to_text(text):
    return emoji.demojize(text)

def replace_profanity(text):
    return re.sub(r'\b[a-zA-Z]*[*!@#$%^&]+[a-zA-Z]*\b', '<PROFANITY>', text)

def limit_token_repetition(text, max_repeats=2):
    words = text.split()
    result, prev, count = [], None, 0
    for w in words:
        count = count + 1 if w == prev else 1
        prev = w
        if count <= max_repeats:
            result.append(w)
    return " ".join(result)

def remove_random_chars_and_spaces(text):
    text = re.sub(r'[^\w\s<>]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def pre_process_text(text):
    return remove_random_chars_and_spaces(limit_token_repetition(replace_profanity(
           replace_user_mentions(convert_emojis_to_text(replace_urls(
               normalize_hashtags(text).lower()))))))


# ── Annotator demographic vector ─────────────────────────────────────────────

def compute_annotator_vector(row, feature_groups):
    parts = []
    for col, vocab in feature_groups:
        vocab_index = {v: i for i, v in enumerate(vocab)}
        counts = np.zeros(len(vocab), dtype=np.float32)
        cell = row.get(col, None)
        if isinstance(cell, list):
            for val in cell:
                if val is not None:
                    idx = vocab_index.get(str(val).strip(), None)
                    if idx is not None:
                        counts[idx] += 1.0
        parts.append(counts)
    vec = np.concatenate(parts)
    total = vec.sum()
    if total > 1e-9:
        vec = vec / total
    return vec


# ── Label helpers (same as training pipeline) ─────────────────────────────────

def remap_labels_flat(annotations):
    return ["NONE" if l == "-" else l for l in annotations if l != "UNKNOWN"]

def remap_labels_nested(annotations):
    result = []
    for annotator in annotations:
        cleaned = ["NONE" if l == "-" else l for l in annotator if l != "UNKNOWN"]
        if cleaned:
            result.append(cleaned)
    return result

def get_soft_distribution(label_series, available_labels):
    if not label_series:
        return None
    counts = {label: 0 for label in available_labels}
    for label in label_series:
        if label in counts:
            counts[label] += 1
    return [counts[l] / len(label_series) for l in available_labels]

def annotations_matrix_to_distribution(annotations, labels):
    if not annotations:
        return None
    counts = {label: 0 for label in labels}
    for annotator in annotations:
        for label in annotator:
            if label in counts:
                counts[label] += 1
    return [counts[label] / len(annotations) for label in labels]


# ── Dataset ──────────────────────────────────────────────────────────────────

class SexismDataset(Dataset):
    def __init__(self, dataframe, tokenizer, clip_processor, image_dir,
                 max_length=128, has_labels=False):
        self.tokenizer      = tokenizer
        self.clip_processor = clip_processor
        self.image_dir      = image_dir
        self.max_length     = max_length
        self.has_labels     = has_labels

        self.ids       = dataframe['id_EXIST'].values
        self.texts     = dataframe['text'].apply(pre_process_text).values
        self.img_paths = dataframe['path_memes'].values
        self.demo_vecs = dataframe.apply(
            lambda row: compute_annotator_vector(row, DEMO_FEATURE_GROUPS), axis=1
        ).values

        if has_labels:
            self.labels_a = dataframe['labels_task2_1_soft'].values
            self.labels_b = dataframe['labels_task2_2_soft'].values
            self.labels_c = dataframe['labels_task2_3_soft'].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        image_path = os.path.join(self.image_dir, self.img_paths[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=0)

        item = {
            'id_EXIST':       str(self.ids[idx]),
            'input_ids':      tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'pixel_values':   self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0),
            'demo_vec':       torch.tensor(self.demo_vecs[idx], dtype=torch.float32),
        }

        if self.has_labels:
            la = self.labels_a[idx]
            lb = self.labels_b[idx]
            lc = self.labels_c[idx]
            item['gold_a'] = torch.tensor(la, dtype=torch.float32) if la is not None \
                             else torch.full((len(LABELS_2_1),), -1.0)
            item['gold_b'] = torch.tensor(lb, dtype=torch.float32) if lb is not None \
                             else torch.full((len(LABELS_2_2),), -1.0)
            item['gold_c'] = torch.tensor(lc, dtype=torch.float32) if lc is not None \
                             else torch.full((len(LABELS_2_3),), -1.0)

        return item


# ── Model ────────────────────────────────────────────────────────────────────

class MultiHeadModel(nn.Module):
    def __init__(self, num_labels_task_b, num_labels_task_c,
                 demo_vec_dim=66, dropout_prob=0.1):
        super().__init__()

        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        text_dim = self.encoder.config.hidden_size  # 1024

        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        vision_dim = self.vision_encoder.config.hidden_size  # 1024

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for layer in self.vision_encoder.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        shared_dim = 512
        self.demo_gate = nn.Linear(demo_vec_dim, shared_dim)
        self.text_proj   = nn.Sequential(nn.Linear(text_dim,     shared_dim), nn.LayerNorm(shared_dim))
        self.vision_proj = nn.Sequential(nn.Linear(vision_dim,   shared_dim), nn.LayerNorm(shared_dim))
        self.demo_proj   = nn.Sequential(nn.Linear(demo_vec_dim, shared_dim), nn.LayerNorm(shared_dim))

        fused_dim = shared_dim * 2
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.dropout     = nn.Dropout(dropout_prob)

        self.head_a = nn.Linear(fused_dim, 1)
        self.head_b = nn.Linear(fused_dim, num_labels_task_b)
        self.head_c = nn.Linear(fused_dim, num_labels_task_c)

        self.loss_a_fn = nn.BCEWithLogitsLoss()
        self.loss_c_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input_ids, attention_mask, pixel_values=None, demo_vec=None, **kwargs):
        text_cls = self.text_proj(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                .last_hidden_state[:, 0, :]
        )
        if demo_vec is not None:
            gate = torch.sigmoid(self.demo_gate(demo_vec))  # [B, 512]
            text_cls = text_cls + gate * self.demo_proj(demo_vec)

        if pixel_values is not None:
            vision_cls = self.vision_proj(
                self.vision_encoder(pixel_values=pixel_values)
                    .last_hidden_state[:, 0, :]
            )
        else:
            vision_cls = torch.zeros_like(text_cls)

        fused = self.dropout(self.fusion_norm(torch.cat([text_cls, vision_cls], dim=-1)))
        return self.head_a(fused).squeeze(-1), self.head_b(fused), self.head_c(fused)


# ── Probability helpers ───────────────────────────────────────────────────────
#
# Task 2_1 sigmoid convention (matches training):
#   The model is trained against labels_a[:, 1] = p(NO).
#   Therefore: sigmoid(logit) = p_NO,  p_YES = 1 - sigmoid(logit).
#
def logit_to_probs_a(logit_scalar):
    """Return (p_YES, p_NO) from a single Task 2_1 logit scalar."""
    p_no  = float(torch.sigmoid(logit_scalar) if isinstance(logit_scalar, torch.Tensor)
                  else 1 / (1 + np.exp(-logit_scalar)))
    p_yes = 1.0 - p_no
    return p_yes, p_no


# ── JSON formatting ───────────────────────────────────────────────────────────

def format_records(sample_ids, all_probs_a, all_probs_b, all_probs_c,
                   task, evaluation, thresholds_c=None, test_case="EXIST2026"):
    """
    all_probs_a : list of (p_YES, p_NO) tuples
    all_probs_b : list of [p_NONE, p_JUDGEMENTAL, p_DIRECT]
    all_probs_c : list of [p_NONE, p_IDEOLOGICAL, ...]
    """
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": test_case, "id": str(sid)}

        if task == '2_1':
            p_yes, p_no = all_probs_a[i]
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
    """Soft gold JSON — probability dicts, same format as pred soft."""
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": test_case, "id": str(sid)}
        if task == '2_1':
            g = gold_dist[i]
            # gold_dist stores [p_YES, p_NO] (LABELS_2_1 order)
            entry["value"] = {"YES": round(float(g[0]), 6), "NO": round(float(g[1]), 6)}
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
    """
    Hard gold JSON — string labels (or list for 2_3), derived from the soft
    gold by majority vote / argmax. This is what PyEvALL hard mode expects.
    """
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": test_case, "id": str(sid)}
        if task == '2_1':
            g = gold_dist[i]   # [p_YES, p_NO]
            entry["value"] = "YES" if g[0] >= g[1] else "NO"
        elif task == '2_2':
            g = gold_dist[i]   # [p_NONE, p_JUDGEMENTAL, p_DIRECT]
            label_map = {"NONE": "NO", "JUDGEMENTAL": "JUDGEMENTAL", "DIRECT": "DIRECT"}
            best = LABELS_2_2[int(np.argmax(g))]
            entry["value"] = label_map[best]
        elif task == '2_3':
            g = gold_dist[i]   # [p_NONE, p_IDEOLOGICAL, ...]
            label_map = {"NONE": "NO"}
            label_map.update({lbl: lbl for lbl in LABELS_2_3[1:]})
            # threshold at 0.5 for gold hard (standard EXIST practice)
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
    gold_a = np.array(all_gold_a)  # [N, 2]  — [p_YES, p_NO]
    gold_b = np.array(all_gold_b)  # [N, 3]
    gold_c = np.array(all_gold_c)  # [N, 6]

    mask_a = gold_a[:, 0] >= 0
    mask_b = gold_b[:, 0] >= 0
    mask_c = gold_c[:, 0] >= 0

    def filt(lst, mask): return [lst[i] for i in range(len(lst)) if mask[i]]

    ids_a = filt(all_ids, mask_a)
    ids_b = filt(all_ids, mask_b)
    ids_c = filt(all_ids, mask_c)
    pa    = filt(all_probs_a, mask_a)   # list of (p_YES, p_NO)
    pb    = filt(all_probs_b, mask_b)
    pc    = filt(all_probs_c, mask_c)
    ga    = gold_a[mask_a].tolist()     # [p_YES, p_NO]
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
            gold_hard = _gold_hard_records(sids, gold, task, test_case)   # ← fixed

            scores[f'{task}_soft'] = _run_pyevall(pred_soft, gold_soft, task, 'soft', tmp)
            scores[f'{task}_hard'] = _run_pyevall(pred_hard, gold_hard, task, 'hard', tmp)  # ← fixed

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

    # Sanity check — no external lib needed
    if pa:
        pred_hard_labels = ["YES" if p_yes >= p_no else "NO" for p_yes, p_no in pa]
        gold_hard_labels = ["YES" if g[0] >= g[1] else "NO" for g in ga]
        acc = sum(p == g for p, g in zip(pred_hard_labels, gold_hard_labels)) / len(pred_hard_labels)
        print(f"\n  Sanity — Task 2_1 accuracy (threshold 0.5): {acc:.4f}")
        print(f"  Predicted YES: {pred_hard_labels.count('YES')}/{len(pred_hard_labels)}"
              f"  |  Gold YES: {gold_hard_labels.count('YES')}/{len(gold_hard_labels)}\n")


# ── Core run ──────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading tokenizer and CLIP processor...")
    tokenizer      = XLMRobertaTokenizer.from_pretrained(args.model_dir)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print("Loading model...")
    model = MultiHeadModel(
        num_labels_task_b=len(LABELS_2_2),
        num_labels_task_c=len(LABELS_2_3),
        demo_vec_dim=DEMO_VEC_DIM,
    )
    state_dict = safetensors.torch.load_file(
        os.path.join(args.model_dir, "model.safetensors"), device=str(device)
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    print(f"Loading data from {args.test_json}...")
    raw = pd.read_json(args.test_json).transpose().reset_index(drop=True)

    for col, _ in DEMO_FEATURE_GROUPS:
        if col not in raw.columns:
            raw[col] = [[] for _ in range(len(raw))]

    if args.eval:
        raw['labels_task2_1_soft'] = raw['labels_task2_1'].apply(
            lambda x: get_soft_distribution(remap_labels_flat(x), LABELS_2_1)
        )
        raw['labels_task2_2_soft'] = raw['labels_task2_2'].apply(
            lambda x: get_soft_distribution(remap_labels_flat(x), LABELS_2_2)
        )
        raw['labels_task2_3_soft'] = raw['labels_task2_3'].apply(
            lambda x: annotations_matrix_to_distribution(remap_labels_nested(x), LABELS_2_3)
        )
        raw = raw[raw['labels_task2_1_soft'].notna()].reset_index(drop=True)
        print(f"Samples with valid Task 2_1 labels: {len(raw)}")

    dataset = SexismDataset(raw, tokenizer, clip_processor, args.image_dir,
                            has_labels=args.eval)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=(device.type == 'cuda'))
    print(f"Total samples: {len(dataset)}")

    # ── Inference loop ────────────────────────────────────────────────────────
    all_ids     = []
    all_probs_a = []   # list of (p_YES, p_NO) tuples
    all_probs_b = []
    all_probs_c = []
    all_gold_a  = []
    all_gold_b  = []
    all_gold_c  = []

    with torch.no_grad():
        for batch in loader:
            ids            = batch.pop('id_EXIST')
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values   = batch['pixel_values'].to(device)
            demo_vec       = batch['demo_vec'].to(device)

            logits_a, logits_b, logits_c = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                demo_vec=demo_vec,
            )

            # sigmoid(logit_a) = p_NO  →  p_YES = 1 - sigmoid
            p_no  = torch.sigmoid(logits_a).cpu().tolist()
            p_yes = [1.0 - p for p in p_no]

            all_ids.extend(ids)
            all_probs_a.extend(list(zip(p_yes, p_no)))   # (p_YES, p_NO) per sample
            all_probs_b.extend(F.softmax(logits_b.float(), dim=-1).cpu().tolist())
            all_probs_c.extend(torch.sigmoid(logits_c.float()).cpu().tolist())

            if args.eval:
                all_gold_a.extend(batch['gold_a'].tolist())
                all_gold_b.extend(batch['gold_b'].tolist())
                all_gold_c.extend(batch['gold_c'].tolist())

    print(f"Inference complete — {len(all_ids)} samples.")

    if args.eval:
        print_eval_results(all_ids, all_probs_a, all_probs_b, all_probs_c,
                           all_gold_a, all_gold_b, all_gold_c,
                           args.thresholds, args.test_case)
        return

    # ── Test mode: export JSON files ──────────────────────────────────────────
    tc = args.test_case
    print("Writing output files...")
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c, '2_1', 'soft', test_case=tc),
              os.path.join(args.output_dir, 'task_2_1_soft.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c, '2_1', 'hard', test_case=tc),
              os.path.join(args.output_dir, 'task_2_1_hard.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c, '2_2', 'soft', test_case=tc),
              os.path.join(args.output_dir, 'task_2_2_soft.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c, '2_2', 'hard', test_case=tc),
              os.path.join(args.output_dir, 'task_2_2_hard.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c, '2_3', 'soft', test_case=tc),
              os.path.join(args.output_dir, 'task_2_3_soft.json'))
    save_json(format_records(all_ids, all_probs_a, all_probs_b, all_probs_c, '2_3', 'hard',
                             thresholds_c=args.thresholds, test_case=tc),
              os.path.join(args.output_dir, 'task_2_3_hard.json'))
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir',   required=True)
    p.add_argument('--test-json',   required=True)
    p.add_argument('--image-dir',   required=True)
    p.add_argument('--output-dir',  default='./predictions')
    p.add_argument('--eval',        action='store_true',
                   help='Evaluate against gold labels instead of exporting JSONs')
    p.add_argument('--test-case',   default='EXIST2026')
    p.add_argument('--batch-size',  type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--thresholds',  type=float, nargs=6, default=None,
                   metavar=('NONE', 'IDEOLOGICAL', 'MISOGYNY', 'SEXUAL', 'OBJECTIFICATION', 'STEREOTYPING'),
                   help='Per-class thresholds for Task 2_3 hard (6 floats, default 0.5 each)')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())