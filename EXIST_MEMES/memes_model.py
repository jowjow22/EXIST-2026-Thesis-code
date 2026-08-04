#!/usr/bin/env python
# coding: utf-8

import os
import re
import json
import csv
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import emoji
from wordsegment import load as load_wordsegment_dictionary, segment

from sklearn.model_selection import train_test_split

from transformers import (
    XLMRobertaTokenizer, XLMRobertaModel,
    CLIPProcessor, CLIPVisionModel,
    TrainingArguments, Trainer, TrainerCallback,
)

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

os.chdir("/home/a67606/Thesis/EXIST_MEMES")
load_wordsegment_dictionary()

import hashlib
import random

SEED = 42

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_full_determinism(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


set_full_determinism(SEED)


raw_data = pd.read_json('./Dataset/memes/training/EXIST2026_training.json')
raw_data = raw_data.transpose()
raw_data.head()


def has_biometric_modality(row, modality_name):
    sensorial_data = row.get('sensorial')
    if sensorial_data is None or not isinstance(sensorial_data, dict):
        return False
    users_with_data = sensorial_data.get('modalities', {}).get(modality_name, {}).get('by_user', {})
    return len(users_with_data) > 0

print(f"Total de memes: {len(raw_data)}")
print(f"Com algum dado biométrico: {raw_data.apply(lambda r: r.get('sensorial') is not None, axis=1).sum()}")
for modality in ['ET', 'HR', 'EEG']:
    count = raw_data.apply(lambda r, modality=modality: has_biometric_modality(r, modality), axis=1).sum()
    print(f"Com {modality}: {count}")


dataset = raw_data.copy()
dataset.drop(['id_EXIST', 'number_annotators', 'annotators', 'lang', 'split'], axis=1, inplace=True)
dataset.describe()


for i in range(3):
    path = os.path.join('./Dataset/memes/training/', dataset['path_memes'].iloc[i])
    exists = os.path.exists(path)
    print(f"{path} -> {'existe' if exists else 'FALTANDO'}")


LABELS_2_1 = ['YES', 'NO']
LABELS_2_2 = ['NONE', 'JUDGEMENTAL', 'DIRECT']
LABELS_2_3 = ['NONE', 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE',
              'SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'STEREOTYPING-DOMINANCE']

THRESHOLD_GRID = np.arange(0.05, 0.65, 0.05).tolist()

TASK_2_2_HIERARCHY = {
    "YES": ["DIRECT", "JUDGEMENTAL"],
    "NO": []
}

TASK_2_3_HIERARCHY = {
    "YES": [
        "IDEOLOGICAL-INEQUALITY",
        "STEREOTYPING-DOMINANCE",
        "OBJECTIFICATION",
        "SEXUAL-VIOLENCE",
        "MISOGYNY-NON-SEXUAL-VIOLENCE"
    ],
    "NO": []
}


def normalize_hashtags(text):
    def split_one_hashtag(match):
        tag_without_hash = match.group()[1:]
        if "_" in tag_without_hash:
            words = tag_without_hash.split("_")
        elif re.search(r'[A-Z]', tag_without_hash):
            words = re.sub(r'(?<!^)(?=[A-Z])', ' ', tag_without_hash).split()
        else:
            try:
                words = segment(tag_without_hash)
            except ValueError:
                words = [tag_without_hash]
        return " ".join(words)
    return re.sub(r'#\w+', split_one_hashtag, text)


def convert_emojis_to_text(text):
    return emoji.demojize(text)


def replace_urls(text):
    return re.sub(r'http\S+|www\S+', '<URL>', text)


def replace_user_mentions(text):
    return re.sub(r'@\w+', '<USER_MENTION>', text)


def replace_profanity(text):
    profanity_pattern = r'\b[a-zA-Z]*[*!@#$%^&]+[a-zA-Z]*\b'
    return re.sub(profanity_pattern, "<PROFANITY>", text)


def limit_token_repetition(text, max_repeats=2):
    words = text.split()
    kept_words = []
    previous_word = None
    repeat_count = 0
    for word in words:
        if word == previous_word:
            repeat_count += 1
        else:
            previous_word = word
            repeat_count = 1
        if repeat_count <= max_repeats:
            kept_words.append(word)
    return " ".join(kept_words)


def strip_punctuation_and_extra_spaces(text):
    text = re.sub(r'[^\w\s<>]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def pre_process_text(text):
    text = normalize_hashtags(text)
    text = text.lower()
    text = replace_urls(text)
    text = convert_emojis_to_text(text)
    text = replace_user_mentions(text)
    text = replace_profanity(text)
    text = limit_token_repetition(text)
    text = strip_punctuation_and_extra_spaces(text)
    return text


example_raw = dataset['text'].iloc[0]
example_clean = pre_process_text(example_raw)
print("ANTES:")
print(example_raw)
print()
print("DEPOIS:")
print(example_clean)


def remap_labels_flat(annotator_labels):
    return [
        "NONE" if label == "-" else label
        for label in annotator_labels
        if label != "UNKNOWN"
    ]

def remap_labels_nested(per_annotator_label_lists):
    cleaned_lists = []
    for this_annotators_labels in per_annotator_label_lists:
        cleaned = [
            "NONE" if label == "-" else label
            for label in this_annotators_labels
            if label != "UNKNOWN"
        ]
        if cleaned:
            cleaned_lists.append(cleaned)
    return cleaned_lists


def get_soft_distribution(annotator_labels, all_possible_labels):
    if not annotator_labels:
        return None
    vote_counts = dict.fromkeys(all_possible_labels, 0)
    for vote in annotator_labels:
        if vote in vote_counts:
            vote_counts[vote] += 1
    num_votes = len(annotator_labels)
    return [vote_counts[label] / num_votes for label in all_possible_labels]

def annotations_matrix_to_distribution(per_annotator_label_lists, all_possible_labels):
    if not per_annotator_label_lists:
        return None
    label_counts = dict.fromkeys(all_possible_labels, 0)
    num_annotators = len(per_annotator_label_lists)
    for this_annotators_labels in per_annotator_label_lists:
        for label in this_annotators_labels:
            if label in label_counts:
                label_counts[label] += 1
    return [label_counts[label] / num_annotators for label in all_possible_labels]


DEMO_ETHNICITY_VALS = sorted(['Asian', 'Black or African American', 'Hispano or Latino',
                               'Middle Eastern', 'Multiracial', 'White or Caucasian', 'other'])
DEMO_STUDY_VALS = sorted(["Bachelor's degree", 'Doctorate',
                           'High school degree or equivalent',
                           'Less than high school diploma', "Master's degree", 'other'])
DEMO_COUNTRY_VALS = sorted(['Afghanistan', 'Argentina', 'Australia', 'Austria', 'Azerbaijan',
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
print(f"Dimensão do vetor demográfico: {DEMO_VEC_DIM}")


def _normalize_apostrophes(text):
    return text.replace("'", "'")


def _count_votes_per_category(annotator_values, vocabulary):
    category_to_index = {
        _normalize_apostrophes(category): i for i, category in enumerate(vocabulary)
    }
    counts = np.zeros(len(vocabulary), dtype=np.float32)
    if not isinstance(annotator_values, list):
        return counts
    for value in annotator_values:
        if value is None:
            continue
        idx = category_to_index.get(_normalize_apostrophes(str(value).strip()), None)
        if idx is not None:
            counts[idx] += 1.0
    return counts


def compute_annotator_vector(row, feature_groups):
    category_count_vectors = [
        _count_votes_per_category(row.get(column_name, None), vocabulary)
        for column_name, vocabulary in feature_groups
    ]

    full_vector = np.concatenate(category_count_vectors)
    total_votes = full_vector.sum()
    if total_votes > 1e-9:
        full_vector = full_vector / total_votes
    return full_vector


dataset['text_preprocessed'] = dataset['text'].apply(pre_process_text)

dataset['labels_task2_1_soft'] = dataset['labels_task2_1'].apply(
    lambda votes: get_soft_distribution(votes, LABELS_2_1)
)

dataset['labels_task2_2'] = dataset['labels_task2_2'].apply(remap_labels_flat)
dataset['labels_task2_2_soft'] = dataset['labels_task2_2'].apply(
    lambda votes: get_soft_distribution(votes, LABELS_2_2)
)

dataset['labels_task2_3'] = dataset['labels_task2_3'].apply(remap_labels_nested)
dataset['labels_task2_3_soft'] = dataset['labels_task2_3'].apply(
    lambda votes: annotations_matrix_to_distribution(votes, LABELS_2_3)
)

dataset['demo_vec'] = dataset.apply(
    lambda row: compute_annotator_vector(row, DEMO_FEATURE_GROUPS), axis=1
)

print(f"Antes de filtrar: {len(dataset)} amostras")
dataset = dataset[dataset['labels_task2_1_soft'].notna()].reset_index(drop=True)
print(f"Depois de filtrar (Task A precisa de ao menos 1 voto válido): {len(dataset)} amostras")
dataset.head()


class SexismDataset(Dataset):
    def __init__(self, dataframe, tokenizer, clip_processor, max_length, image_dir):
        self.tokenizer      = tokenizer
        self.clip_processor = clip_processor
        self.max_length     = max_length
        self.image_dir      = image_dir

        self.text        = dataframe['text_preprocessed'].values
        self.image_paths = dataframe['path_memes'].values
        self.labels_a    = dataframe['labels_task2_1_soft'].values
        self.labels_b    = dataframe['labels_task2_2_soft'].values
        self.labels_c    = dataframe['labels_task2_3_soft'].values
        self.demo_vecs   = dataframe['demo_vec'].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.text[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            print("Imagem não encontrada:", image_path)
            image = Image.new('RGB', (224, 224), color=0)

        image_inputs = self.clip_processor(images=image, return_tensors='pt')

        batch = {
            'input_ids':      tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'pixel_values':   image_inputs['pixel_values'].squeeze(0),
            'demo_vec':       torch.tensor(self.demo_vecs[idx], dtype=torch.float32),
            'labels_a':       torch.tensor(self.labels_a[idx], dtype=torch.float32),
        }

        label_b = self.labels_b[idx]
        if label_b is None:
            batch['labels_b'] = torch.full((len(LABELS_2_2),), -1.0, dtype=torch.float32)
        else:
            label_b = np.array(label_b, dtype=np.float32)
            if label_b[0] != -1.0:
                batch['labels_b'] = torch.tensor(label_b, dtype=torch.float32)
            else:
                batch['labels_b'] = torch.full((len(label_b),), -1.0, dtype=torch.float32)

        label_c = self.labels_c[idx]
        if label_c is None:
            batch['labels_c'] = torch.full((len(LABELS_2_3),), -1.0, dtype=torch.float32)
        else:
            label_c = np.array(label_c, dtype=np.float32)
            if label_c[0] != -1.0:
                batch['labels_c'] = torch.tensor(label_c, dtype=torch.float32)
            else:
                batch['labels_c'] = torch.full((len(label_c),), -1.0, dtype=torch.float32)

        return batch


class MultiHeadModel(nn.Module):
    def __init__(self, num_labels_task_b, num_labels_task_c, demo_vec_dim=66,
                 dropout_prob=0.1, use_vision=True, use_demographics=True):
        super().__init__()
        self.use_vision = use_vision
        self.use_demographics = use_demographics

        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        text_dim = self.encoder.config.hidden_size

        shared_dim = 512
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared_dim), nn.LayerNorm(shared_dim)
        )

        if self.use_vision:
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            vision_dim = self.vision_encoder.config.hidden_size
            frozen_layers = 4
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for layer in self.vision_encoder.encoder.layers[-frozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, shared_dim), nn.LayerNorm(shared_dim)
            )

        if self.use_demographics:
            self.demo_gate = nn.Linear(demo_vec_dim, shared_dim)
            self.demo_proj = nn.Sequential(
                nn.Linear(demo_vec_dim, shared_dim),
                nn.LayerNorm(shared_dim),
            )

        fused_dim = shared_dim * 2
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout_prob)

        self.head_a = nn.Linear(fused_dim, 1)
        self.head_b = nn.Linear(fused_dim, num_labels_task_b)
        self.head_c = nn.Linear(fused_dim, num_labels_task_c)

        self.loss_a_fn = nn.BCEWithLogitsLoss()
        self.loss_c_fn = nn.BCEWithLogitsLoss(reduction='none')

    def _loss_task_a(self, logits_a, labels_a):
        if labels_a is None:
            return None
        if labels_a.dim() == 2:
            labels_a = labels_a[:, 1]
        mask_a = labels_a != -100.0
        if not mask_a.any():
            return None
        loss_a = self.loss_a_fn(logits_a[mask_a].view(-1), labels_a[mask_a])
        return loss_a * 2.0

    def _loss_task_b(self, logits_b, labels_b):
        if labels_b is None:
            return None
        mask_b = labels_b[:, 0] != -1.0
        if not mask_b.any():
            return None
        log_probs_b = F.log_softmax(logits_b[mask_b], dim=-1)
        return -(labels_b[mask_b] * log_probs_b).sum(dim=-1).mean()

    def _loss_task_c(self, logits_c, labels_c):
        if labels_c is None:
            return None
        mask_c = labels_c[:, 0] != -1.0
        if not mask_c.any():
            return None
        return self.loss_c_fn(logits_c[mask_c], labels_c[mask_c]).mean()

    def forward(self, input_ids, attention_mask,
                pixel_values=None, demo_vec=None,
                labels_a=None, labels_b=None, labels_c=None, **kwargs):

        text_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = self.text_proj(text_out.last_hidden_state[:, 0, :])

        if self.use_demographics and demo_vec is not None:
            gate = torch.sigmoid(self.demo_gate(demo_vec))
            text_cls = text_cls + gate * self.demo_proj(demo_vec)

        if self.use_vision and pixel_values is not None:
            vision_out = self.vision_encoder(pixel_values=pixel_values)
            vision_cls = self.vision_proj(vision_out.last_hidden_state[:, 0, :])
        else:
            vision_cls = torch.zeros_like(text_cls)

        fused = torch.cat([text_cls, vision_cls], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)

        logits_a = self.head_a(fused)
        logits_b = self.head_b(fused)
        logits_c = self.head_c(fused)

        task_losses = [
            self._loss_task_a(logits_a, labels_a),
            self._loss_task_b(logits_b, labels_b),
            self._loss_task_c(logits_c, labels_c),
        ]
        task_losses = [l for l in task_losses if l is not None]

        if task_losses:
            loss = sum(task_losses)
        else:
            loss = logits_a.sum() * 0.0

        return (loss, logits_a, logits_b, logits_c)


set_full_determinism(SEED)

model = MultiHeadModel(
    num_labels_task_b=len(LABELS_2_2),
    num_labels_task_c=len(LABELS_2_3),
    demo_vec_dim=DEMO_VEC_DIM,
)

print("Fingerprint dos pesos iniciais:", model_fingerprint(model))


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
frozen_params = total_params - trainable_params

print(f"Parâmetros totais:     {total_params:,}")
print(f"Parâmetros treináveis: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
print(f"Parâmetros congelados: {frozen_params:,}")


tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def majority_vote_task_a(soft_label):
    return 'YES' if soft_label[0] >= 0.5 else 'NO'

stratify_col = dataset['labels_task2_1_soft'].apply(majority_vote_task_a)

dataset_train, dataset_val = train_test_split(
    dataset,
    test_size=0.15,
    random_state=42,
    stratify=stratify_col,
)
dataset_train = dataset_train.reset_index(drop=True)
dataset_val = dataset_val.reset_index(drop=True)

IMAGE_DIR = './Dataset/memes/training/'

train_dataset = SexismDataset(
    dataframe=dataset_train,
    tokenizer=tokenizer,
    clip_processor=clip_processor,
    max_length=128,
    image_dir=IMAGE_DIR,
)

val_dataset = SexismDataset(
    dataframe=dataset_val,
    tokenizer=tokenizer,
    clip_processor=clip_processor,
    max_length=128,
    image_dir=IMAGE_DIR,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4)

print(f"Treino: {len(train_dataset)} amostras | Validação: {len(val_dataset)} amostras")


training_args = TrainingArguments(
    output_dir="./results",
    seed=SEED,
    data_seed=SEED,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=0.1,
    weight_decay=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="task_a_icmsoftnorm_soft",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=50,
    fp16=True,
    report_to="none",
    dataloader_pin_memory=True,
)


def _extract_metrics_from_report(report, metric_list):
    results = {}
    if not hasattr(report, 'report') or 'metrics' not in report.report:
        return results

    metrics_dict = report.report['metrics']
    for metric in metric_list:
        if metric not in metrics_dict:
            continue
        if metrics_dict[metric].get('status') != 'OK':
            continue
        try:
            results[metric] = float(
                metrics_dict[metric]['results']['average_per_test_case']
            )
        except (KeyError, TypeError, ValueError):
            pass
    return results


def _format_for_pyevall(probs, sample_ids, task, evaluation, thresholds_c=None):
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": "EXIST2025", "id": str(sid)}

        if task == '2_1':
            p_yes = float(probs[i][0])
            p_no  = float(probs[i][1])
            if evaluation == 'hard':
                entry["value"] = "YES" if p_yes >= p_no else "NO"
            else:
                entry["value"] = {"YES": p_yes, "NO": p_no}

        elif task == '2_2':
            label_map = {"NONE": "NO", "JUDGEMENTAL": "JUDGEMENTAL", "DIRECT": "DIRECT"}
            relevant  = {label_map[lbl]: float(probs[i][idx])
                         for idx, lbl in enumerate(LABELS_2_2)}
            if evaluation == 'hard':
                entry["value"] = max(relevant, key=relevant.get)
            else:
                entry["value"] = relevant

        elif task == '2_3':
            label_map = {"NONE": "NO"}
            label_map.update({lbl: lbl for lbl in LABELS_2_3[1:]})
            relevant  = {label_map[lbl]: float(probs[i][idx])
                         for idx, lbl in enumerate(LABELS_2_3)}

            if evaluation == 'hard':
                threshs = thresholds_c if thresholds_c is not None else [0.5] * len(LABELS_2_3)
                thresh_map = {label_map[lbl]: threshs[idx] for idx, lbl in enumerate(LABELS_2_3)}
                hard_labels = [lbl for lbl, p in relevant.items() if p >= thresh_map[lbl]]
                if not hard_labels:
                    hard_labels = [max(relevant, key=relevant.get)]
                entry["value"] = hard_labels
            else:
                entry["value"] = relevant

        out.append(entry)
    return out


def _run_pyevall(probs, gold_dist, sample_ids, task, evaluation,
                 temp_dir='./pyevall_temp_train', thresholds_c=None):
    os.makedirs(temp_dir, exist_ok=True)

    preds = _format_for_pyevall(probs, sample_ids, task, evaluation,
                                 thresholds_c=thresholds_c)
    gold  = _format_for_pyevall(gold_dist, sample_ids, task, evaluation)

    pred_path = f'{temp_dir}/pred_{task}_{evaluation}.json'
    gold_path = f'{temp_dir}/gold_{task}_{evaluation}.json'
    with open(pred_path, 'w') as f: json.dump(preds, f)
    with open(gold_path, 'w') as f: json.dump(gold, f)

    if evaluation == 'hard':
        metric_list = ['ICM', 'ICMNorm', 'FMeasure']
    else:
        if task == '2_3':
            metric_list = ['ICMSoft', 'ICMSoftNorm']
        else:
            metric_list = ['ICMSoft', 'ICMSoftNorm', 'CrossEntropy']

    params = {}
    if task == '2_2':
        params[PyEvALLUtils.PARAM_HIERARCHY] = TASK_2_2_HIERARCHY
    elif task == '2_3':
        params[PyEvALLUtils.PARAM_HIERARCHY] = TASK_2_3_HIERARCHY

    try:
        evaluator = PyEvALLEvaluation()
        report    = evaluator.evaluate(pred_path, gold_path, metric_list, **params)
        return _extract_metrics_from_report(report, metric_list)
    except Exception as e:
        print(f"  ! PyEvALL falhou para task {task} ({evaluation}): {e}")
        return {}


def _search_thresholds_task_c(probs_c, targets_c, ids_c, temp_dir):
    best_thresholds = [0.5] * len(LABELS_2_3)

    for class_idx in range(len(LABELS_2_3)):
        best_t   = best_thresholds[class_idx]
        best_icm = -np.inf

        for t in THRESHOLD_GRID:
            candidate = best_thresholds.copy()
            candidate[class_idx] = float(t)

            scores = _run_pyevall(probs_c, targets_c, ids_c, '2_3', 'hard',
                                  temp_dir=temp_dir, thresholds_c=candidate)
            icm = scores.get('ICMNorm', -np.inf)
            if icm > best_icm:
                best_icm = icm
                best_t   = float(t)

        best_thresholds[class_idx] = best_t

    final_scores = _run_pyevall(probs_c, targets_c, ids_c, '2_3', 'hard',
                                temp_dir=temp_dir, thresholds_c=best_thresholds)
    return best_thresholds, final_scores


def _metrics_task_a(logits_a, labels_a, temp_dir):
    metrics = {}
    if labels_a.ndim == 2:
        labels_a = labels_a[:, 1]

    mask_a = labels_a != -100.0
    if mask_a.any():
        prob_no  = 1 / (1 + np.exp(-logits_a[mask_a].squeeze(-1)))
        probs_a  = np.stack([1 - prob_no, prob_no], axis=-1)
        targets_a = np.stack([1 - labels_a[mask_a], labels_a[mask_a]], axis=-1)
        ids_a    = [str(i) for i in range(len(probs_a))]

        hard_scores = _run_pyevall(probs_a, targets_a, ids_a, '2_1', 'hard',
                                   temp_dir=temp_dir)
        for k, v in hard_scores.items():
            metrics[f'task_a_{k.lower()}_hard'] = v

        soft_scores = _run_pyevall(probs_a, targets_a, ids_a, '2_1', 'soft',
                                   temp_dir=temp_dir)
        for k, v in soft_scores.items():
            metrics[f'task_a_{k.lower()}_soft'] = v

    return metrics


def _metrics_task_b(logits_b, labels_b, temp_dir):
    metrics = {}
    mask_b = labels_b[:, 0] != -1.0
    if mask_b.any():
        x        = logits_b[mask_b]
        x        = x - x.max(axis=-1, keepdims=True)
        probs_b  = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
        targets_b = labels_b[mask_b]
        ids_b    = [str(i) for i in range(len(probs_b))]

        hard_scores = _run_pyevall(probs_b, targets_b, ids_b, '2_2', 'hard',
                                   temp_dir=temp_dir)
        for k, v in hard_scores.items():
            metrics[f'task_b_{k.lower()}_hard'] = v

        soft_scores = _run_pyevall(probs_b, targets_b, ids_b, '2_2', 'soft',
                                   temp_dir=temp_dir)
        for k, v in soft_scores.items():
            metrics[f'task_b_{k.lower()}_soft'] = v

    return metrics


def _metrics_task_c(logits_c, labels_c, temp_dir):
    metrics = {}
    mask_c = labels_c[:, 0] != -1.0
    if mask_c.any():
        probs_c  = 1 / (1 + np.exp(-logits_c[mask_c]))
        targets_c = labels_c[mask_c]
        ids_c    = [str(i) for i in range(len(probs_c))]

        soft_scores = _run_pyevall(probs_c, targets_c, ids_c, '2_3', 'soft',
                                   temp_dir=temp_dir)
        for k, v in soft_scores.items():
            metrics[f'task_c_{k.lower()}_soft'] = v

        hard_scores_default = _run_pyevall(probs_c, targets_c, ids_c, '2_3', 'hard',
                                           temp_dir=temp_dir)
        for k, v in hard_scores_default.items():
            metrics[f'task_c_{k.lower()}_hard_default'] = v

        best_thresholds, hard_scores_tuned = _search_thresholds_task_c(
            probs_c, targets_c, ids_c, temp_dir
        )
        for k, v in hard_scores_tuned.items():
            metrics[f'task_c_{k.lower()}_hard'] = v

        for j, label in enumerate(LABELS_2_3):
            metrics[f'task_c_thresh_{label.lower().replace("-", "_")}'] = round(best_thresholds[j], 2)

    return metrics


def compute_metrics(eval_pred):
    logits_a, logits_b, logits_c = eval_pred.predictions
    labels_a, labels_b, labels_c = eval_pred.label_ids
    logits_a = np.asarray(logits_a)
    logits_b = np.asarray(logits_b)
    logits_c = np.asarray(logits_c)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    labels_c = np.asarray(labels_c)
    metrics = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        metrics.update(_metrics_task_a(logits_a, labels_a, temp_dir))
        metrics.update(_metrics_task_b(logits_b, labels_b, temp_dir))
        metrics.update(_metrics_task_c(logits_c, labels_c, temp_dir))

    return metrics


class MetricsHistoryLogger(TrainerCallback):

    def __init__(self, output_dir="./metrics_history"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = os.path.join(self.output_dir, f"{self.run_timestamp}.json")
        self.csv_path = os.path.join(self.output_dir, f"{self.run_timestamp}.csv")

    def _save(self, log_history):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=2, ensure_ascii=False)

        all_keys = []
        for row in log_history:
            for key in row:
                if key not in all_keys:
                    all_keys.append(key)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(log_history)

    def on_evaluate(self, args, state, control, **kwargs):
        self._save(state.log_history)

    def on_train_end(self, args, state, control, **kwargs):
        self._save(state.log_history)


metrics_logger = MetricsHistoryLogger()
print(f"Métricas desta run serão salvas em: {metrics_logger.json_path} e {metrics_logger.csv_path}")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[metrics_logger],
)


trainer.train()


import safetensors.torch
from accelerate.utils import extract_model_from_parallel

actual_model = extract_model_from_parallel(trainer.model)
state_dict = {k: v.cpu().clone() for k, v in actual_model.state_dict().items()}
safetensors.torch.save_file(state_dict, "./best_model/model.safetensors")
tokenizer.save_pretrained("./best_model")
print("Modelo e tokenizer salvos em ./best_model")
print("Fingerprint dos pesos finais:", model_fingerprint(actual_model))


def diagnose_vision_contribution(model, val_loader, device='cuda'):
    model.eval()

    results = {'real_images': [], 'zero_images': []}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_pixels = batch['pixel_values'].to(device)
            zero_pixels = torch.zeros_like(real_pixels)

            _, logits_a_real, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=real_pixels,
            )

            _, logits_a_zero, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=zero_pixels,
            )

            probs_a_real = torch.sigmoid(logits_a_real).cpu().numpy()
            probs_a_zero = torch.sigmoid(logits_a_zero).cpu().numpy()

            results['real_images'].append(probs_a_real)
            results['zero_images'].append(probs_a_zero)

    real = np.concatenate(results['real_images']).flatten()
    zero = np.concatenate(results['zero_images']).flatten()

    diff = np.abs(real - zero)
    correlation = np.corrcoef(real, zero)[0, 1]

    print("=" * 60)
    print("DIAGNÓSTICO DE ABLAÇÃO DE VISÃO")
    print("=" * 60)
    print(f"Diferença média absoluta entre previsões com imagem real vs zerada:")
    print(f"  {diff.mean():.4f}  (apenas Task A)")
    print(f"")
    print(f"Correlação entre previsões reais e zeradas: {correlation:.4f}")
    print(f"  - 1.00 = a imagem não tem NENHUM efeito (modelo ignora imagens)")
    print(f"  - 0.50 = contribuição moderada da imagem")
    print(f"  - 0.00 = imagem domina a decisão / aleatório")
    print(f"")

    if correlation > 0.99:
        print("[AVISO] A imagem está sendo essencialmente ignorada!")
    elif correlation > 0.95:
        print("[AVISO] A imagem contribui muito pouco.")
    elif correlation > 0.85:
        print("[AVISO] A imagem contribui de forma modesta.")
    else:
        print("[OK] A imagem está contribuindo de forma significativa.")

    return correlation, diff.mean()


diagnose_vision_contribution(actual_model, val_loader)


def check_clip_output_diversity(model, val_loader, device='cuda', n_samples=20):
    model.eval()
    cls_outputs = []

    with torch.no_grad():
        count = 0
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)

            vision_out = model.vision_encoder(pixel_values=pixel_values)
            cls_token = vision_out.last_hidden_state[:, 0, :]

            cls_outputs.append(cls_token.cpu().numpy())

            count += pixel_values.shape[0]
            if count >= n_samples:
                break

    cls_outputs = np.concatenate(cls_outputs, axis=0)[:n_samples]

    from numpy.linalg import norm
    similarities = []
    for i in range(len(cls_outputs)):
        for j in range(i + 1, len(cls_outputs)):
            sim = np.dot(cls_outputs[i], cls_outputs[j]) / (
                norm(cls_outputs[i]) * norm(cls_outputs[j]) + 1e-8
            )
            similarities.append(sim)

    similarities = np.array(similarities)

    print("=" * 60)
    print("CHECAGEM DE DIVERSIDADE DE SAÍDA DO CLIP")
    print("=" * 60)
    print(f"Similaridade de cosseno entre pares, {n_samples} amostras:")
    print(f"  Média: {similarities.mean():.4f}")
    print(f"  Desvio padrão:  {similarities.std():.4f}")
    print(f"  Mínimo:  {similarities.min():.4f}")
    print(f"  Máximo:  {similarities.max():.4f}")
    print()

    if similarities.mean() > 0.99:
        print("[AVISO] As saídas do CLIP são quase idênticas para todas as imagens!")
        print("        Isso indica que o CLIP não está vendo conteúdo diverso.")
        print("        Causas possíveis: todas as imagens caíram no fallback preto,")
        print("                          ou o pré-processamento está quebrado.")
    elif similarities.mean() > 0.95:
        print("[AVISO] As saídas do CLIP são muito parecidas (baixa diversidade).")
    else:
        print("[OK] O CLIP está produzindo saídas diversas.")

    return similarities

check_clip_output_diversity(actual_model, val_loader)