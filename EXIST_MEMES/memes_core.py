"""memes_core.py

Lógica reutilizável extraída VERBATIM de `memes_model.ipynb` (Task 2 — Memes, EXIST 2026).

Contém: constantes de labels, pré-processamento de texto, remapeamento de labels,
distribuições soft, vetor demográfico dos anotadores, `SexismDataset`, `MultiHeadModel`
e todas as funções de métricas (PyEvALL). NÃO contém código de topo que execute treino —
carregamento de dataset, split treino/validação, `TrainingArguments`, `Trainer` e
`trainer.train()` ficam no framework de ablação (`ablation/train_one.py`), não aqui.

Copiado função a função do notebook, sem alterações de comportamento.
"""

import os
import re
import json
import tempfile
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
import emoji
from wordsegment import load as load_wordsegment_dictionary, segment

from transformers import XLMRobertaModel, CLIPVisionModel

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

# wordsegment precisa do dicionário de frequência de palavras carregado antes de segment() funcionar
load_wordsegment_dictionary()


# ============================================================================
# 2.1 Reprodutibilidade — seed global e kernels determinísticos
# ============================================================================

SEED = 42

# O cuBLAS lê esta variável quando cria o handle (na primeira operação em GPU);
# precisa estar definida antes de qualquer uso de CUDA para ter efeito.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_full_determinism(seed=SEED):
    """Fixa todos os geradores de números aleatórios e força kernels determinísticos.

    Chamar imediatamente antes de cada passo cuja aleatoriedade precisa ser
    reproduzível (criação do modelo, treino). Chamar de novo "re-zera" os
    geradores para o mesmo ponto, independentemente do que rodou antes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # gerador da CPU (init de pesos, dropout em CPU)
    torch.cuda.manual_seed_all(seed)  # geradores de todas as GPUs (dropout em GPU)

    # Sem isto, o cuDNN escolhe algoritmos por benchmark de tempo de execução
    # (varia de run para run) e usa kernels com somas atômicas em ordem
    # arbitrária — diferenças de arredondamento que se amplificam no treino.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Exige implementação determinística de TODO operador; se algum não tiver,
    # o PyTorch levanta RuntimeError apontando o operador culpado.
    torch.use_deterministic_algorithms(True)


def model_fingerprint(model):
    """Hash SHA-256 (16 hex) de todos os pesos do modelo.

    Dois modelos com o mesmo fingerprint têm pesos bit a bit idênticos —
    comparar o valor impresso entre runs verifica a reprodutibilidade.
    """
    digest = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


# ============================================================================
# 5. Definições de labels (rótulos)
# ============================================================================

LABELS_2_1 = ['YES', 'NO']
LABELS_2_2 = ['NONE', 'JUDGEMENTAL', 'DIRECT']
LABELS_2_3 = ['NONE', 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE',
              'SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'STEREOTYPING-DOMINANCE']

# Valores de corte testados na busca de threshold da Task C (seção 11)
THRESHOLD_GRID = np.arange(0.05, 0.65, 0.05).tolist()   # [0.05, 0.10, ..., 0.60]

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


# ============================================================================
# 6. Limpeza de texto
# ============================================================================

def normalize_hashtags(text):
    """Transforma hashtags em palavras legíveis.
    Ex.: '#MyGreatIdea' -> 'my great idea', '#happy_day' -> 'happy day'.
    Tenta 3 estratégias, na ordem: separar por underscore, separar por
    CamelCase, ou usar um dicionário de segmentação de palavras (wordsegment)
    como último recurso."""
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
                words = [tag_without_hash]  # não conseguiu segmentar, mantém como está
        return " ".join(words)
    return re.sub(r'#\w+', split_one_hashtag, text)


def convert_emojis_to_text(text):
    """Troca emoji por sua descrição textual, ex.: 😂 -> ':face_with_tears_of_joy:'.
    O tokenizador entende palavras muito melhor do que glifos de emoji."""
    return emoji.demojize(text)


def replace_urls(text):
    """Qualquer link vira o placeholder <URL> — o conteúdo do link em si não importa
    para detectar sexismo, só atrapalharia o tokenizador."""
    return re.sub(r'http\S+|www\S+', '<URL>', text)


def replace_user_mentions(text):
    """@fulano -> <USER_MENTION>. Anonimiza e generaliza menções a usuários."""
    return re.sub(r'@\w+', '<USER_MENTION>', text)


def replace_profanity(text):
    """Palavrões censurados com símbolos (ex.: 'f*ck', 'sh!t') viram <PROFANITY>.
    Detecta: qualquer 'palavra' que contenha um dos símbolos * ! @ # $ % ^ &."""
    profanity_pattern = r'\b[a-zA-Z]*[*!@#$%^&]+[a-zA-Z]*\b'
    return re.sub(profanity_pattern, "<PROFANITY>", text)


def limit_token_repetition(text, max_repeats=2):
    """Corta repetições consecutivas da mesma palavra em no máximo `max_repeats`.
    Ex.: 'não não não não não' -> 'não não'. Mantém a ênfase sem deixar o texto
    virar uma parede da mesma palavra repetida, que confundiria o tokenizador."""
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
    """Remove qualquer caractere que não seja letra/número/underscore/espaço/< >,
    depois colapsa espaços múltiplos em um só. Os placeholders <URL>, <USER_MENTION>,
    <PROFANITY> sobrevivem porque < e > estão na lista de exceções."""
    text = re.sub(r'[^\w\s<>]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def pre_process_text(text):
    """Pipeline completo de limpeza de texto, aplicado em ordem:
    1. hashtags -> palavras legíveis
    2. tudo em minúsculas
    3. URLs -> <URL>
    4. emoji -> descrição textual
    5. @menções -> <USER_MENTION>
    6. palavrões censurados -> <PROFANITY>
    7. limita repetição de palavras consecutivas
    8. remove pontuação sobrando e espaços extras
    """
    text = normalize_hashtags(text)
    text = text.lower()
    text = replace_urls(text)
    text = convert_emojis_to_text(text)
    text = replace_user_mentions(text)
    text = replace_profanity(text)
    text = limit_token_repetition(text)
    text = strip_punctuation_and_extra_spaces(text)
    return text


# ============================================================================
# 7. Rótulos suaves (soft labels) e remapeamento
# ============================================================================

def remap_labels_flat(annotator_labels):
    """Para tasks onde cada anotador dá UM rótulo (Task A, Task B).
    '-' (não-sexista) -> 'NONE'. 'UNKNOWN' (indeciso) é descartado."""
    return [
        "NONE" if label == "-" else label
        for label in annotator_labels
        if label != "UNKNOWN"
    ]


def remap_labels_nested(per_annotator_label_lists):
    """Mesma regra acima, mas para tasks onde cada anotador dá VÁRIOS rótulos (Task C).
    Um anotador cuja lista fica vazia depois do remapeamento é removido inteiramente
    (ele não deu nenhuma opinião válida)."""
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
    """Conta os votos de uma lista FLAT (Task A, Task B) e devolve a fração de cada rótulo.
    Ex.: ['YES','YES','NO'] com labels ['YES','NO'] -> [0.667, 0.333].
    Devolve None se não houve nenhum voto (a amostra deve ser descartada)."""
    if not annotator_labels:
        return None
    vote_counts = dict.fromkeys(all_possible_labels, 0)
    for vote in annotator_labels:
        if vote in vote_counts:
            vote_counts[vote] += 1
    num_votes = len(annotator_labels)
    return [vote_counts[label] / num_votes for label in all_possible_labels]


def annotations_matrix_to_distribution(per_annotator_label_lists, all_possible_labels):
    """Mesma ideia, mas para a Task C, onde cada anotador pode escolher MAIS DE UM
    rótulo. Conta quantos ANOTADORES (não quantas escolhas) marcaram cada categoria,
    dividido pelo número de anotadores."""
    if not per_annotator_label_lists:
        return None
    label_counts = dict.fromkeys(all_possible_labels, 0)
    num_annotators = len(per_annotator_label_lists)
    for this_annotators_labels in per_annotator_label_lists:
        for label in this_annotators_labels:
            if label in label_counts:
                label_counts[label] += 1
    return [label_counts[label] / num_annotators for label in all_possible_labels]


# ============================================================================
# 8. Vetor demográfico dos anotadores
# ============================================================================

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

# (nome da coluna no dataset, vocabulário fixo daquela característica)
DEMO_FEATURE_GROUPS = [
    ('ethnicities_annotators',  DEMO_ETHNICITY_VALS),
    ('study_levels_annotators', DEMO_STUDY_VALS),
    ('countries_annotators',    DEMO_COUNTRY_VALS),
]

DEMO_VEC_DIM = len(DEMO_ETHNICITY_VALS) + len(DEMO_STUDY_VALS) + len(DEMO_COUNTRY_VALS)  # 7+6+53 = 66

# --- ABL-2 (demo_vec expandido): vocabulários novos, não usados na baseline ---
DEMO_GENDER_VALS = sorted(['F', 'M'])
DEMO_AGE_VALS = sorted(['18-22', '23-45', '46+'])

DEMO_FEATURE_GROUPS_EXTENDED = DEMO_FEATURE_GROUPS + [
    ('gender_annotators', DEMO_GENDER_VALS),
    ('age_annotators',    DEMO_AGE_VALS),
]


def get_demo_feature_groups(feature_set="base"):
    """feature_set: "base" (66-d, ethnicity+study+country) ou "extended"
    (66-d + gender + age). Usado pela ABL-2."""
    if feature_set == "base":
        return DEMO_FEATURE_GROUPS
    if feature_set == "extended":
        return DEMO_FEATURE_GROUPS_EXTENDED
    raise ValueError(f"demo_feature_set desconhecido: {feature_set!r}")


def get_demo_vec_dim(feature_set="base"):
    return sum(len(vocabulary) for _, vocabulary in get_demo_feature_groups(feature_set))


def _normalize_apostrophes(text):
    """Trata apóstrofo curvo (’, U+2019) e reto (', U+0027) como o mesmo caractere.
    Os dados brutos usam ’ em "Bachelor’s degree"/"Master’s degree"; o vocabulário fixo
    abaixo usa '. Sem essa normalização, 64,6% dos votos de escolaridade nunca batiam
    com o vocabulário e eram descartados silenciosamente (achado desta revisão de código,
    corrigido aqui -- ver seção 8)."""
    return text.replace("’", "'")


def _count_votes_per_category(annotator_values, vocabulary):
    """Para UM grupo de característica (ex.: escolaridade): conta quantos anotadores
    caem em cada categoria do vocabulário. Devolve um vetor de contagens, na mesma
    ordem do vocabulário."""
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


def compute_annotator_vector(row, feature_groups, norm_mode="sum"):
    """Monta o vetor demográfico (66-d na baseline) para um meme.

    Para cada grupo de característica: conta quantos anotadores caem em cada
    categoria daquele vocabulário (_count_votes_per_category), concatena as
    contagens, e normaliza o vetor inteiro.

    norm_mode: "sum" (baseline -- divide pela soma total, preserva o sinal
    demográfico) ou "l2" (ABL-6 -- divide pela norma L2; se a norma for
    ~0, o vetor fica como está, sem divisão por zero).
    """
    category_count_vectors = [
        _count_votes_per_category(row.get(column_name, None), vocabulary)
        for column_name, vocabulary in feature_groups
    ]

    full_vector = np.concatenate(category_count_vectors)  # [66] na baseline

    if norm_mode == "sum":
        total_votes = full_vector.sum()
        if total_votes > 1e-9:
            full_vector = full_vector / total_votes
    elif norm_mode == "l2":
        norm = np.linalg.norm(full_vector)
        if norm > 1e-9:
            full_vector = full_vector / norm
    else:
        raise ValueError(f"norm_mode desconhecido: {norm_mode!r}")

    return full_vector


# ============================================================================
# 10. O "Dataset" do PyTorch
# ============================================================================

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
        self.demo_vecs   = dataframe['demo_vec'].values  # [N,] de np.ndarray [66]

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
            image = Image.new('RGB', (224, 224), color=0)  # fallback: imagem preta

        image_inputs = self.clip_processor(images=image, return_tensors='pt')

        batch = {
            'input_ids':      tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'pixel_values':   image_inputs['pixel_values'].squeeze(0),
            'demo_vec':       torch.tensor(self.demo_vecs[idx], dtype=torch.float32),  # [66]
            'labels_a':       torch.tensor(self.labels_a[idx], dtype=torch.float32),
        }

        # Task B: se não há soft label válido, usa o vetor sentinela -1.0
        # (o modelo detecta isso e ignora a amostra no cálculo da loss_b)
        label_b = self.labels_b[idx]
        if label_b is None:
            batch['labels_b'] = torch.full((len(LABELS_2_2),), -1.0, dtype=torch.float32)
        else:
            label_b = np.array(label_b, dtype=np.float32)
            if label_b[0] != -1.0:
                batch['labels_b'] = torch.tensor(label_b, dtype=torch.float32)
            else:
                batch['labels_b'] = torch.full((len(label_b),), -1.0, dtype=torch.float32)

        # Task C: mesma lógica de sentinela -1.0
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


# ============================================================================
# 11. A arquitetura do modelo (`MultiHeadModel`)
# ============================================================================

DEMO_INJECTION_MODES = ("current", "renorm", "head_inject")


class MultiHeadModel(nn.Module):
    """Modelo multi-head (Task A/B/C) sobre XLM-R + CLIP + demografia.

    `demo_injection_mode` (ABL-INJ) escolhe COMO o sinal demográfico entra na rede:

    - "current"     : baseline atual, injeção somada ao CLS de texto DEPOIS do
                      LayerNorm de `text_proj` (e antes do `fusion_norm`).
                      NENHUM módulo novo é criado neste modo -- a sequência de
                      draws do RNG é idêntica à do código antes desta mudança,
                      logo `model_fingerprint` inicial é bit-a-bit igual ao da
                      baseline para a mesma seed. É a âncora de comparação.
    - "renorm"      : igual a "current", mas re-aplica um LayerNorm
                      (`self.demo_post_norm`) DEPOIS da injeção, antes de
                      concatenar com visão -- devolve o CLS de texto à escala
                      unitária que o `text_proj` garantia e a injeção destruía.
                      `nn.LayerNorm` inicializa com ones/zeros (não consome RNG),
                      então os pesos das outras camadas continuam bit-a-bit
                      iguais; o fingerprint muda apenas porque o `state_dict`
                      ganha duas chaves novas.
    - "head_inject" : NÃO injeta no CLS de texto. O sinal demográfico projetado é
                      concatenado ao vetor `fused` DEPOIS do `fusion_norm`, logo
                      antes das heads. As heads passam de 1024 para 1536
                      in_features (1024 fundidos + 512 demo) -- isto MUDA o
                      fingerprint por construção (as heads têm outro shape e
                      consomem outra quantidade de RNG); é esperado, não é
                      regressão.
                      Concatenar DEPOIS do `fusion_norm` (e não antes) respeita a
                      regra do CLAUDE.md de nunca normalizar um vetor concatenado
                      enquanto algum braço de ablação puder zerar um slot.

    Com `use_demographics=False` (ou `demo_vec=None` no forward), TODOS os modos
    são no-ops: nenhum cria módulo de demografia e o forward não muda nada.
    """

    def __init__(self, num_labels_task_b, num_labels_task_c, demo_vec_dim=66,
                 dropout_prob=0.1, use_vision=True, use_demographics=True,
                 task_a_weight=2.0, unfrozen_vision_layers=4,
                 demo_injection_mode="current"):
        super().__init__()
        if demo_injection_mode not in DEMO_INJECTION_MODES:
            raise ValueError(
                f"demo_injection_mode desconhecido: {demo_injection_mode!r} "
                f"(esperado um de {DEMO_INJECTION_MODES})"
            )
        self.use_vision = use_vision
        self.use_demographics = use_demographics
        self.demo_injection_mode = demo_injection_mode
        self.task_a_weight = task_a_weight  # ABL-3: peso da loss da Task A (baseline 2.0)

        # === Codificador de texto ===
        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        text_dim = self.encoder.config.hidden_size  # 1024

        shared_dim = 512
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared_dim), nn.LayerNorm(shared_dim)
        )

        # === Codificador de imagem (CLIP), parcialmente congelado ===
        if self.use_vision:
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            vision_dim = self.vision_encoder.config.hidden_size  # 1024
            # ABL-4: nº de últimas camadas do CLIP treináveis (baseline 4; 0 = tudo congelado)
            frozen_layers = unfrozen_vision_layers
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            if frozen_layers > 0:
                for layer in self.vision_encoder.encoder.layers[-frozen_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, shared_dim), nn.LayerNorm(shared_dim)
            )

        # === Demografia: projetada, com portão aprendido ===
        # A projeção e o portão são IDÊNTICOS nos 3 modos (mesma ordem de criação,
        # mesmos draws de RNG) -- só o ponto de aplicação muda no forward.
        if self.use_demographics:
            self.demo_gate = nn.Linear(demo_vec_dim, shared_dim)
            self.demo_proj = nn.Sequential(
                nn.Linear(demo_vec_dim, shared_dim),
                nn.LayerNorm(shared_dim),
            )
            # Só o modo "renorm" cria este módulo -- em "current" nada muda.
            if self.demo_injection_mode == "renorm":
                self.demo_post_norm = nn.LayerNorm(shared_dim)

        # === Fusão: 512 (texto+demo) + 512 (visão) = 1024 ===
        fused_dim = shared_dim * 2
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # "head_inject" concatena o sinal demográfico [B, 512] ao vetor fundido
        # DEPOIS do fusion_norm -> as heads passam a ver 1024 + 512 = 1536.
        self.head_input_dim = fused_dim
        if self.use_demographics and self.demo_injection_mode == "head_inject":
            self.head_input_dim = fused_dim + shared_dim  # 1536

        self.head_a = nn.Linear(self.head_input_dim, 1)                  # Task A: 1 logit (sexista?)
        self.head_b = nn.Linear(self.head_input_dim, num_labels_task_b)  # Task B: 3 logits (intenção)
        self.head_c = nn.Linear(self.head_input_dim, num_labels_task_c)  # Task C: 6 logits (categorias)

        self.loss_a_fn = nn.BCEWithLogitsLoss()
        self.loss_c_fn = nn.BCEWithLogitsLoss(reduction='none')

    def _loss_task_a(self, logits_a, labels_a):
        """Task A: sexista sim/não. Devolve None se nenhuma amostra do batch tem
        voto válido (sentinela -100.0)."""
        if labels_a is None:
            return None
        # labels_a chega como [B, 2] do coletor padrão do Trainer; a coluna 1
        # é P(NO) -- ver LABELS_2_1 = ['YES', 'NO']. Depois desta linha vira [B].
        if labels_a.dim() == 2:
            labels_a = labels_a[:, 1]
        mask_a = labels_a != -100.0  # sentinela de "sem voto válido" para Task A
        if not mask_a.any():
            return None
        loss_a = self.loss_a_fn(logits_a[mask_a].view(-1), labels_a[mask_a])
        return loss_a * self.task_a_weight  # ABL-3: peso da Task A (baseline 2.0)

    def _loss_task_b(self, logits_b, labels_b):
        """Task B: intenção (NONE/JUDGEMENTAL/DIRECT). Cross-entropy contra soft label.
        Devolve None se nenhuma amostra do batch tem voto válido (sentinela -1.0)."""
        if labels_b is None:
            return None
        mask_b = labels_b[:, 0] != -1.0
        if not mask_b.any():
            return None
        log_probs_b = F.log_softmax(logits_b[mask_b], dim=-1)
        # cross-entropy contra soft label: -sum(p_verdadeiro * log(p_previsto))
        return -(labels_b[mask_b] * log_probs_b).sum(dim=-1).mean()

    def _loss_task_c(self, logits_c, labels_c):
        """Task C: categorias finas (multilabel). BCE por categoria.
        Devolve None se nenhuma amostra do batch tem voto válido (sentinela -1.0)."""
        if labels_c is None:
            return None
        mask_c = labels_c[:, 0] != -1.0
        if not mask_c.any():
            return None
        return self.loss_c_fn(logits_c[mask_c], labels_c[mask_c]).mean()

    def forward(self, input_ids, attention_mask,
                pixel_values=None, demo_vec=None,
                labels_a=None, labels_b=None, labels_c=None, **kwargs):

        # === Texto ===
        text_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = self.text_proj(text_out.last_hidden_state[:, 0, :])  # [B, 512]

        # === Sinal demográfico (ABL-INJ: onde ele entra depende do modo) ===
        demo_signal = None
        if self.use_demographics and demo_vec is not None:
            gate = torch.sigmoid(self.demo_gate(demo_vec))     # [B, 512], valores em (0,1)
            demo_signal = gate * self.demo_proj(demo_vec)      # [B, 512]

            if self.demo_injection_mode == "current":
                # baseline: soma pós-LayerNorm de text_proj, sem re-normalizar
                text_cls = text_cls + demo_signal
            elif self.demo_injection_mode == "renorm":
                # mesma soma, mas devolve o CLS à escala unitária antes da concatenação
                text_cls = self.demo_post_norm(text_cls + demo_signal)
            # "head_inject": text_cls fica intocado; demo_signal entra depois das heads
        # Sem demo_vec (ou use_demographics=False): text_cls fica como está em TODOS
        # os modos (inclusive "renorm" -- sem demo, sem post-norm).

        # === Imagem ===
        if self.use_vision and pixel_values is not None:
            vision_out = self.vision_encoder(pixel_values=pixel_values)
            vision_cls = self.vision_proj(vision_out.last_hidden_state[:, 0, :])
        else:
            vision_cls = torch.zeros_like(text_cls)

        # === Fusão: concatenação simples, sem cross-attention ===
        fused = torch.cat([text_cls, vision_cls], dim=-1)  # [B, 1024]
        fused = self.fusion_norm(fused)

        # === "head_inject": demografia entra aqui, DEPOIS do fusion_norm ===
        # Nunca antes: normalizar um cat com slot zerável reintroduziria o confound
        # de modalidade ausente (CLAUDE.md). O dropout vem depois do cat para o slot
        # demográfico ser regularizado como o resto.
        if self.use_demographics and self.demo_injection_mode == "head_inject":
            if demo_signal is None:
                # demo_vec ausente em runtime: slot de zeros, mantém o shape das heads
                demo_signal = torch.zeros(
                    fused.shape[0], fused.shape[-1] // 2,
                    dtype=fused.dtype, device=fused.device,
                )
            fused = torch.cat([fused, demo_signal], dim=-1)  # [B, 1536]

        fused = self.dropout(fused)

        logits_a = self.head_a(fused)
        logits_b = self.head_b(fused)
        logits_c = self.head_c(fused)

        # === Loss ===
        # Cada amostra pode não ter voto válido em Task B e/ou C (ver seção 10) —
        # cada _loss_task_* devolve None nesse caso, e é ignorada na soma abaixo,
        # sem afetar as outras tasks.
        task_losses = [
            self._loss_task_a(logits_a, labels_a),
            self._loss_task_b(logits_b, labels_b),
            self._loss_task_c(logits_c, labels_c),
        ]
        task_losses = [l for l in task_losses if l is not None]

        if task_losses:
            loss = sum(task_losses)
        else:
            # nenhuma task tinha label válido neste batch -- loss zero que ainda
            # participa do grafo computacional (evita erro do Trainer)
            loss = logits_a.sum() * 0.0

        return (loss, logits_a, logits_b, logits_c)


# ============================================================================
# 14. Métricas de avaliação (PyEvALL)
# ============================================================================

def _extract_metrics_from_report(report, metric_list):
    """Puxa os valores numéricos das métricas de dentro do objeto de relatório do PyEvALL."""
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
    """Converte um array de probabilidades para o formato JSON que o PyEvALL espera.

    thresholds_c: lista opcional de 6 floats, usada só na Task C modo 'hard' — o valor
                  de corte por categoria para decidir se ela "está presente" ou não.
                  Se None, usa 0.5 para todas as categorias.
    """
    out = []
    for i, sid in enumerate(sample_ids):
        entry = {"test_case": "EXIST2025", "id": str(sid)}

        if task == '2_1':
            # Task A: sigmoid(logit) = P(NO) -- ver nota na seção 11 sobre a convenção
            # de índice de LABELS_2_1 = ['YES', 'NO']. probs[i][0] já vem como P(YES).
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
                # marca como presente toda categoria cuja probabilidade bate o threshold dela
                hard_labels = [lbl for lbl, p in relevant.items() if p >= thresh_map[lbl]]
                if not hard_labels:
                    # nenhuma categoria bateu o threshold -- em vez de devolver uma lista
                    # vazia (o PyEvALL rejeitaria), usa a categoria mais provável mesmo assim
                    hard_labels = [max(relevant, key=relevant.get)]
                entry["value"] = hard_labels
            else:
                entry["value"] = relevant

        out.append(entry)
    return out


def _run_pyevall(probs, gold_dist, sample_ids, task, evaluation,
                 temp_dir='./pyevall_temp_train', thresholds_c=None):
    """Roda o PyEvALL uma vez, para uma task e um modo de avaliação (soft ou hard)."""
    os.makedirs(temp_dir, exist_ok=True)

    preds = _format_for_pyevall(probs, sample_ids, task, evaluation,
                                 thresholds_c=thresholds_c)
    # thresholds_c NÃO é passado para o gold -- o gold usa a distribuição real de votos,
    # e o PyEvALL binariza essa distribuição internamente quando o modo é 'hard'.
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
    """Busca gulosa por coordenada: ajusta 1 threshold por vez, mantendo os outros fixos."""
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

    # avaliação final com todos os thresholds já travados nos melhores valores encontrados
    final_scores = _run_pyevall(probs_c, targets_c, ids_c, '2_3', 'hard',
                                temp_dir=temp_dir, thresholds_c=best_thresholds)
    return best_thresholds, final_scores


def _metrics_task_a(logits_a, labels_a, temp_dir):
    """Task A: sexista sim/não. Devolve {} se nenhuma amostra tem voto válido."""
    metrics = {}
    if labels_a.ndim == 2:
        labels_a = labels_a[:, 1]

    mask_a = labels_a != -100.0
    if mask_a.any():
        prob_no  = 1 / (1 + np.exp(-logits_a[mask_a].squeeze(-1)))  # sigmoid = P(NO)
        probs_a  = np.stack([1 - prob_no, prob_no], axis=-1)        # [P(YES), P(NO)]
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
    """Task B: intenção (NONE/JUDGEMENTAL/DIRECT). Devolve {} se nenhuma amostra tem voto válido."""
    metrics = {}
    mask_b = labels_b[:, 0] != -1.0
    if mask_b.any():
        x        = logits_b[mask_b]
        x        = x - x.max(axis=-1, keepdims=True)  # estabilidade numérica do softmax
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
    """Task C: categorias finas (multilabel), incluindo a busca de threshold (seção 14.2).
    Devolve {} se nenhuma amostra tem voto válido.

    IMPORTANTE: a ordem das atribuições abaixo é proposital -- 'task_c_..._hard' é
    escrito primeiro pelo bloco 'hard_default' (threshold=0.5) e depois SOBRESCRITO
    pelo bloco de threshold ajustado, que se torna o score hard oficial. Não separar
    essas duas atribuições em funções diferentes chamadas fora de ordem.
    """
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

        # threshold padrão (0.5), só de referência para comparação
        hard_scores_default = _run_pyevall(probs_c, targets_c, ids_c, '2_3', 'hard',
                                           temp_dir=temp_dir)
        for k, v in hard_scores_default.items():
            metrics[f'task_c_{k.lower()}_hard_default'] = v

        # threshold ajustado pela busca gulosa (seção 14.2) -- vira o score hard oficial
        best_thresholds, hard_scores_tuned = _search_thresholds_task_c(
            probs_c, targets_c, ids_c, temp_dir
        )
        for k, v in hard_scores_tuned.items():
            metrics[f'task_c_{k.lower()}_hard'] = v

        # loga os thresholds por categoria, para acompanhar como evoluem entre épocas
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


# ============================================================================
# 15. Diagnóstico de colapso da head de Task C (ABL-INJ)
# ============================================================================

# Limiares do veredito. Escolhidos para serem lidos como "praticamente constante":
# variância inter-item < 1e-4 = desvio-padrão < 0,01 na probabilidade prevista;
# |média - taxa base| < 0,02 = a head reproduz a marginal do treino a 2 pontos.
COLLAPSE_VARIANCE_THRESHOLD = 1e-4
COLLAPSE_BASERATE_DIFF_THRESHOLD = 0.02


def compute_train_base_rate_task_c(train_labels_c, num_labels=None):
    """Taxa base de cada categoria da Task C no conjunto de TREINO.

    `train_labels_c`: iterável de soft targets [6] (ou None / sentinela -1.0 para
    itens sem voto válido -- esses são EXCLUÍDOS, não contados como zeros).
    Devolve a proporção de itens válidos em que a categoria aparece (soft > 0).
    """
    num_labels = num_labels or len(LABELS_2_3)
    valid_rows = []
    for label in train_labels_c:
        if label is None:
            continue
        vector = np.asarray(label, dtype=np.float32)
        if vector.size != num_labels or vector[0] == -1.0:
            continue
        valid_rows.append(vector)

    if not valid_rows:
        return [float("nan")] * num_labels

    matrix = np.stack(valid_rows)              # [N_valid, 6]
    return (matrix > 0).mean(axis=0).astype(float).tolist()


def _mean_pairwise_cosine(prob_matrix):
    """Cosseno médio entre pares DISTINTOS de linhas de [N, 6].

    Próximo de 1 = todos os itens recebem praticamente a mesma distribuição.
    Linhas de norma ~0 recebem guarda de epsilon (cosseno 0 contra tudo, em vez
    de NaN). Devolve NaN se N < 2.
    """
    matrix = np.asarray(prob_matrix, dtype=np.float64)
    n_items = matrix.shape[0]
    if n_items < 2:
        return float("nan")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-12)
    similarity = unit @ unit.T
    off_diagonal = ~np.eye(n_items, dtype=bool)
    return float(similarity[off_diagonal].mean())


@torch.no_grad()
def diagnose_task_c_collapse(model, val_loader, device, out_path,
                             train_labels_c=None, config_name=None, seed=None):
    """Roda o modelo em TODO o val set e mede se a head de Task C colapsou.

    Colapso = a head ignora o input e devolve sempre a distribuição marginal do
    treino. Isso NÃO aparece nas métricas agregadas (o ICM de uma head que prevê
    a marginal não é catastrófico), por isso é medido explicitamente aqui.

    Chamar UMA VEZ, na avaliação final (não a cada época).

    `train_labels_c` é opcional só para não quebrar chamadas de diagnóstico
    rápido; sem ele, `train_base_rate_per_class` e `abs_diff_mean_vs_baserate`
    vêm como NaN e o veredito de colapso não pode ser emitido.

    `out_path` TEM de ser um caminho persistente (ex.:
    `ablation/diagnostics/task_c_collapse_<config>_seed<seed>.json`). Nunca dentro
    de um tempfile.TemporaryDirectory -- foi exatamente isso que impediu confirmar
    o colapso da vez anterior.

    Devolve o dict que também é gravado em `out_path`.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    prob_batches = []
    for batch in val_loader:
        model_inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and not key.startswith("labels_")
        }
        _, _, _, logits_c = model(**model_inputs)
        prob_batches.append(torch.sigmoid(logits_c.float()).cpu().numpy())

    if was_training:
        model.train()

    probs_c = np.concatenate(prob_batches, axis=0)  # [N_val, 6]

    inter_item_variance = probs_c.var(axis=0).astype(float).tolist()
    mean_pred = probs_c.mean(axis=0).astype(float).tolist()

    if train_labels_c is not None:
        base_rate = compute_train_base_rate_task_c(train_labels_c, probs_c.shape[1])
        abs_diff = np.abs(np.asarray(mean_pred) - np.asarray(base_rate)).astype(float).tolist()
    else:
        base_rate = [float("nan")] * probs_c.shape[1]
        abs_diff = [float("nan")] * probs_c.shape[1]

    mean_variance = float(np.mean(inter_item_variance))
    mean_abs_diff = float(np.mean(abs_diff))
    mean_cosine = _mean_pairwise_cosine(probs_c)

    if np.isnan(mean_abs_diff):
        verdict = ("[INCONCLUSIVO] taxa base do treino não fornecida "
                   "(train_labels_c=None) -- só a variância inter-item foi medida.")
        collapsed = None
    elif (mean_variance < COLLAPSE_VARIANCE_THRESHOLD
          and mean_abs_diff < COLLAPSE_BASERATE_DIFF_THRESHOLD):
        verdict = "[COLAPSO CONFIRMADO] head de Task C prevê a marginal, ignora o input."
        collapsed = True
    else:
        verdict = "[OK] head de Task C discrimina entre itens."
        collapsed = False

    diagnostics = {
        "config_name": config_name,
        "seed": seed,
        "n_val_items": int(probs_c.shape[0]),
        "labels": list(LABELS_2_3),
        "inter_item_variance_per_class": inter_item_variance,
        "mean_pred_per_class": mean_pred,
        "train_base_rate_per_class": base_rate,
        "abs_diff_mean_vs_baserate": abs_diff,
        "mean_pairwise_cosine_between_items": mean_cosine,
        "mean_inter_item_variance": mean_variance,
        "mean_abs_diff_vs_baserate": mean_abs_diff,
        "collapse_variance_threshold": COLLAPSE_VARIANCE_THRESHOLD,
        "collapse_baserate_diff_threshold": COLLAPSE_BASERATE_DIFF_THRESHOLD,
        "collapsed": collapsed,
        "verdict": verdict,
    }

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n--- diagnóstico de colapso da Task C ({config_name}, seed={seed}) ---")
    for j, label in enumerate(LABELS_2_3):
        print(f"  {label:<32} var={inter_item_variance[j]:.3e}  "
              f"mean={mean_pred[j]:.4f}  base={base_rate[j]:.4f}  "
              f"|dif|={abs_diff[j]:.4f}")
    print(f"  variância inter-item média : {mean_variance:.3e}")
    print(f"  |média - taxa base| média  : {mean_abs_diff:.4f}")
    print(f"  cosseno médio entre itens  : {mean_cosine:.4f}")
    print(f"  {verdict}")
    print(f"  JSON: {out_path}")

    return diagnostics


if __name__ == "__main__":
    print("memes_core.py: módulo de definições, sem execução de treino.")
    print(f"DEMO_VEC_DIM = {DEMO_VEC_DIM}")
    print(f"LABELS_2_1={LABELS_2_1} LABELS_2_2={LABELS_2_2} LABELS_2_3={LABELS_2_3}")
