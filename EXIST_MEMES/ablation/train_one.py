"""ablation/train_one.py

`run_single(config)`: roda UM treino completo (1 config, 1 seed) e devolve um dict
com todas as métricas de `memes_core.compute_metrics` + fingerprints + metadados.

Importa toda a lógica de `EXIST_MEMES/memes_core.py` -- nunca do notebook.
Dataset/demo_vec são reconstruídos aqui porque ABL-2 (demo_feature_set) e
ABL-6 (demo_norm_mode) mudam como o demo_vec é montado.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from transformers import (
    XLMRobertaTokenizer, CLIPProcessor,
    TrainingArguments, Trainer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import memes_core as core  # noqa: E402
from config import DEMO_SHUFFLE_SEED  # noqa: E402

DIAGNOSTICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")

# Paths ancorados em EXIST_MEMES/ (pai de ablation/), NÃO no cwd. Com "./Dataset/..."
# o runner só funcionava se lançado de dentro de EXIST_MEMES; lançado de
# ablation/ quebrava com FileNotFoundError no primeiro pd.read_json.
EXIST_MEMES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_JSON = os.environ.get(
    "MEMES_DATASET_JSON",
    os.path.join(EXIST_MEMES_DIR, "Dataset", "memes", "training", "EXIST2026_training.json"),
)
IMAGE_DIR = os.environ.get(
    "MEMES_IMAGE_DIR",
    os.path.join(EXIST_MEMES_DIR, "Dataset", "memes", "training") + os.sep,
)
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8
# Eval não tem gradiente/optimizer state -- folga de VRAM que o treino não tem.
# Maior batch só muda quantas amostras processam por forward call; a agregação
# das previsões (e portanto todas as métricas) é idêntica. Puramente velocidade.
EVAL_BATCH_SIZE = int(os.environ.get("MEMES_EVAL_BATCH_SIZE", "32"))
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
VAL_SPLIT_SEED = 42  # split treino/val É FIXO entre configs -- só o seed de TREINO varia por config.seed
# Sobrepõe tokenização/CLIP-processing (CPU) com compute da GPU -- não muda
# nenhum resultado, só evita a GPU ficar ociosa esperando __getitem__.
DATALOADER_NUM_WORKERS = int(os.environ.get("MEMES_DATALOADER_WORKERS", "4"))


def _load_and_prepare_dataset(demo_feature_set):
    """Carrega o JSON, aplica pré-processamento/remapeamento/soft-labels (idênticos
    para toda config), e monta o demo_vec com o feature_set pedido.

    Levanta ValueError com mensagem clara se `demo_feature_set="extended"` for
    pedido mas as colunas `gender_annotators`/`age_annotators` não existirem
    no dataset (ABL-2).
    """
    raw_data = pd.read_json(DATASET_JSON)
    raw_data = raw_data.transpose()

    dataset = raw_data.copy()
    drop_cols = [c for c in ['id_EXIST', 'number_annotators', 'annotators', 'lang', 'split']
                 if c in dataset.columns]
    dataset.drop(drop_cols, axis=1, inplace=True)

    feature_groups = core.get_demo_feature_groups(demo_feature_set)
    required_columns = {column_name for column_name, _ in feature_groups}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"demo_feature_set={demo_feature_set!r} precisa das colunas "
            f"{sorted(missing_columns)}, que não existem no dataset "
            f"({DATASET_JSON}). Colunas disponíveis: {sorted(dataset.columns)}"
        )

    dataset['text_preprocessed'] = dataset['text'].apply(core.pre_process_text)

    dataset['labels_task2_1_soft'] = dataset['labels_task2_1'].apply(
        lambda votes: core.get_soft_distribution(votes, core.LABELS_2_1)
    )

    dataset['labels_task2_2'] = dataset['labels_task2_2'].apply(core.remap_labels_flat)
    dataset['labels_task2_2_soft'] = dataset['labels_task2_2'].apply(
        lambda votes: core.get_soft_distribution(votes, core.LABELS_2_2)
    )

    dataset['labels_task2_3'] = dataset['labels_task2_3'].apply(core.remap_labels_nested)
    dataset['labels_task2_3_soft'] = dataset['labels_task2_3'].apply(
        lambda votes: core.annotations_matrix_to_distribution(votes, core.LABELS_2_3)
    )

    dataset = dataset[dataset['labels_task2_1_soft'].notna()].reset_index(drop=True)
    return dataset, feature_groups


def _build_demo_vec_column(dataset, feature_groups, demo_norm_mode):
    return dataset.apply(
        lambda row: core.compute_annotator_vector(row, feature_groups, norm_mode=demo_norm_mode),
        axis=1,
    )


def _build_demo_vec_derangement(n_items, seed):
    """Permutação SEM pontos fixos (derangement) de `n_items` índices.

    Uma permutação aleatória qualquer deixa, em média, 1 ponto fixo -- e um ponto
    fixo é um item que ficou com o próprio demo_vec, ou seja, uma amostra em que
    o controle não controla nada. Aqui os pontos fixos são reparados por troca
    até não sobrar nenhum.

    `seed` é `config.DEMO_SHUFFLE_SEED` (fixo, NÃO o seed de treino): as 3 seeds
    de treino veem a mesma permutação, então a comparação pareada seed-a-seed
    contra `inj_headinject_base` isola só o embaralhamento.
    """
    if n_items < 2:
        raise ValueError(f"derangement impossível com n_items={n_items}")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_items)

    # Repara pontos fixos trocando cada um com outra posição ao acaso; repete até
    # a permutação não ter nenhum. Converge em poucas iterações (E[fixos] = 1).
    for _ in range(1000):
        fixed_points = np.flatnonzero(permutation == np.arange(n_items))
        if fixed_points.size == 0:
            break
        for i in fixed_points:
            j = int(rng.integers(0, n_items))
            while j == i:
                j = int(rng.integers(0, n_items))
            permutation[i], permutation[j] = permutation[j], permutation[i]
    else:
        raise RuntimeError("não foi possível construir um derangement em 1000 tentativas")

    assert not np.any(permutation == np.arange(n_items)), "derangement com ponto fixo"
    return permutation


def _apply_demo_vec_shuffle(dataset, seed):
    """Embaralha a coluna `demo_vec` ENTRE itens (ABL-INJ-CTRL).

    Chamado DEPOIS de montar o demo_vec e ANTES do split treino/val -- embaralhar
    depois do split criaria assimetria entre os dois conjuntos (permutações
    distintas dentro de cada um) e contaminaria a comparação.

    Preserva exatamente: a arquitetura, a dimensão do demo_vec, e a distribuição
    marginal dos vetores (é uma permutação -- o multiconjunto de vetores é o
    mesmo). Destrói só o vínculo item<->demografia.

    Devolve (dataset, info) onde `info` registra a taxa de colisão de VALOR: o
    demo_vec tem só ~151 valores distintos em ~3984 itens (cada valor é um pool
    de anotadores, ~27 itens cada), então uma fração pequena dos itens recebe,
    por acaso, um vetor numericamente igual ao seu. Medido e registrado, não
    corrigido -- forçar valor diferente enviesaria a permutação.
    """
    original = np.stack(dataset['demo_vec'].values)
    permutation = _build_demo_vec_derangement(len(dataset), seed)
    shuffled = original[permutation]

    value_collisions = int(np.all(np.isclose(original, shuffled), axis=1).sum())
    info = {
        "demo_shuffle_seed": seed,
        "demo_shuffle_value_collisions": value_collisions,
        "demo_shuffle_value_collision_rate": value_collisions / len(dataset),
    }

    dataset = dataset.copy()
    dataset['demo_vec'] = list(shuffled)
    return dataset, info


def _split_train_val(dataset):
    def majority_vote_task_a(soft_label):
        return 'YES' if soft_label[0] >= 0.5 else 'NO'

    stratify_col = dataset['labels_task2_1_soft'].apply(majority_vote_task_a)
    dataset_train, dataset_val = train_test_split(
        dataset,
        test_size=0.15,
        random_state=VAL_SPLIT_SEED,
        stratify=stratify_col,
    )
    return dataset_train.reset_index(drop=True), dataset_val.reset_index(drop=True)


def run_single(config):
    """Roda 1 treino completo para `config` (ablation.config.AblationConfig).

    Devolve um dict com: config_name, ablation_seed, fingerprint_init,
    fingerprint_final, demo_vec_dim_effective, + todas as chaves devolvidas por
    `memes_core.compute_metrics` na avaliação final.
    """
    print(f"\n{'=' * 70}")
    print(f"RUN: {config.name}  (seed={config.seed})")
    print(f"  {config}")

    dataset, feature_groups = _load_and_prepare_dataset(config.demo_feature_set)
    demo_vec_dim = core.get_demo_vec_dim(config.demo_feature_set)
    print(f"  DEMO_VEC_DIM efetivo: {demo_vec_dim}")

    dataset['demo_vec'] = _build_demo_vec_column(dataset, feature_groups, config.demo_norm_mode)

    # ABL-INJ-CTRL: embaralhamento ANTES do split, com seed própria e fixa.
    shuffle_info = {"demo_shuffle_seed": None,
                    "demo_shuffle_value_collisions": None,
                    "demo_shuffle_value_collision_rate": None}
    if config.demo_shuffle:
        dataset, shuffle_info = _apply_demo_vec_shuffle(dataset, DEMO_SHUFFLE_SEED)
        print(f"  demo_vec EMBARALHADO (derangement, seed={DEMO_SHUFFLE_SEED}): "
              f"{shuffle_info['demo_shuffle_value_collisions']} itens receberam um vetor "
              f"numericamente igual ao próprio "
              f"({100 * shuffle_info['demo_shuffle_value_collision_rate']:.2f}%)")

    dataset_train, dataset_val = _split_train_val(dataset)

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    train_dataset = core.SexismDataset(
        dataframe=dataset_train, tokenizer=tokenizer, clip_processor=clip_processor,
        max_length=MAX_LENGTH, image_dir=IMAGE_DIR,
    )
    val_dataset = core.SexismDataset(
        dataframe=dataset_val, tokenizer=tokenizer, clip_processor=clip_processor,
        max_length=MAX_LENGTH, image_dir=IMAGE_DIR,
    )

    # Re-zera os geradores imediatamente antes de criar o modelo -- garante que
    # as camadas novas nascem com os mesmos pesos em qualquer run desta seed,
    # independentemente do que rodou antes (ver memes_core.set_full_determinism).
    core.set_full_determinism(config.seed)

    model = core.MultiHeadModel(
        num_labels_task_b=len(core.LABELS_2_2),
        num_labels_task_c=len(core.LABELS_2_3),
        demo_vec_dim=demo_vec_dim,
        use_vision=config.use_vision,
        use_demographics=config.use_demographics,
        task_a_weight=config.task_a_weight,
        unfrozen_vision_layers=config.unfrozen_vision_layers,
        demo_injection_mode=config.demo_injection_mode,   # ABL-INJ
    )
    fingerprint_init = core.model_fingerprint(model)
    print(f"  fingerprint_init: {fingerprint_init}")

    # Checkpoints de época (pesos + optimizer state de XLM-R-large + CLIP) passam de
    # 5GB cada -- 36 runs gravando isso em `results_dir` (montado na VM, disco pequeno)
    # esgota o disco em 1 run só (já aconteceu). Usa um diretório TEMPORÁRIO do SO
    # (normalmente outra partição, com mais espaço) e apaga tudo ao fim do run --
    # não precisamos reter o checkpoint em disco, só as métricas (linha do CSV) e os
    # fingerprints. `save_only_model=True` também evita gravar optimizer.pt/scheduler.pt
    # (não precisamos retomar treino no meio de uma run de ablação).
    with tempfile.TemporaryDirectory(prefix=f"ablation_{config.name}_") as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            seed=config.seed,
            data_seed=config.seed,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=config.warmup_steps,   # float [0,1) = fração. NUNCA warmup_ratio.
            weight_decay=WEIGHT_DECAY,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="task_a_icmsoftnorm_soft",
            greater_is_better=True,
            save_total_limit=1,
            save_only_model=True,
            logging_steps=50,
            fp16=True,
            report_to="none",
            dataloader_pin_memory=True,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=core.compute_metrics,
        )

        trainer.train()

        from accelerate.utils import extract_model_from_parallel
        actual_model = extract_model_from_parallel(trainer.model)
        fingerprint_final = core.model_fingerprint(actual_model)
        print(f"  fingerprint_final: {fingerprint_final}")

        eval_metrics = trainer.evaluate()

        # ABL-INJ: diagnóstico de colapso da head de Task C, UMA vez, no fim.
        # Roda aqui dentro (trainer ainda vivo) mas grava FORA do tempdir, em
        # DIAGNOSTICS_DIR -- com load_best_model_at_end=True o modelo avaliado
        # acima é o mesmo diagnosticado abaixo, então o JSON e a linha do CSV
        # descrevem o MESMO checkpoint.
        collapse_diagnostics = None
        diagnostics_path = None
        if config.run_collapse_diagnostics:
            os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
            diagnostics_path = os.path.join(
                DIAGNOSTICS_DIR,
                f"task_c_collapse_{config.name}_seed{config.seed}.json",
            )
            collapse_diagnostics = core.diagnose_task_c_collapse(
                model=trainer.model,
                val_loader=trainer.get_eval_dataloader(),
                device=trainer.args.device,
                out_path=diagnostics_path,
                train_labels_c=dataset_train['labels_task2_3_soft'].values,
                config_name=config.name,
                seed=config.seed,
            )
    # saída do `with`: output_dir (e todos os checkpoints dentro dele) já foi apagado

    row = {
        "config_name": config.name,
        "seed": config.seed,
        "fingerprint_init": fingerprint_init,
        "fingerprint_final": fingerprint_final,
        "demo_vec_dim_effective": demo_vec_dim,
        "num_epochs": config.num_epochs,
        "warmup_steps": config.warmup_steps,
        "task_a_weight": config.task_a_weight,
        "unfrozen_vision_layers": config.unfrozen_vision_layers,
        "use_vision": config.use_vision,
        "use_demographics": config.use_demographics,
        "demo_feature_set": config.demo_feature_set,
        "demo_norm_mode": config.demo_norm_mode,
        "demo_injection_mode": config.demo_injection_mode,
        "demo_shuffle": config.demo_shuffle,
        **shuffle_info,
        "head_input_dim": model.head_input_dim,   # 1024, ou 1536 em "head_inject"
    }
    if collapse_diagnostics is not None:
        row["collapse_mean_inter_item_variance"] = collapse_diagnostics["mean_inter_item_variance"]
        row["collapse_mean_abs_diff_vs_baserate"] = collapse_diagnostics["mean_abs_diff_vs_baserate"]
        row["collapse_mean_pairwise_cosine"] = collapse_diagnostics["mean_pairwise_cosine_between_items"]
        row["collapse_collapsed"] = collapse_diagnostics["collapsed"]
        row["collapse_verdict"] = collapse_diagnostics["verdict"]
        row["collapse_diagnostics_path"] = diagnostics_path
    for key, value in eval_metrics.items():
        clean_key = key[len("eval_"):] if key.startswith("eval_") else key
        row[clean_key] = value

    return row


if __name__ == "__main__":
    from config import get_baseline_config
    result = run_single(get_baseline_config(name="smoke_test"))
    print(result)
