# 09_brazil_columbia_finetuning.py

"""
Fine-tune unet_finetuned_gold.h5 on the gold labels with the two coffee
classes merged into one, giving a 3-class model:

    Gold label value          ->  Model class
    0  Background             ->  0  Background
    1  Forest                 ->  1  Forest
    2  Unshaded coffee        ->  2  Coffee
    3  Shaded coffee          ->  2  Coffee

The 4-channel softmax head of the base model is replaced by a 3-channel head.
Background/forest weights are copied; the coffee channel is initialised from
the two coffee channels so training starts close to the base model.

This script is self-contained: it does not modify src/config.py,
finetuning_config.json, gold_data_split.json or any existing model file.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

from src.config import (
    MODEL_FOLDER, GOLD_DATA_SPLIT_PATH, IMAGE_BAND_NAMES,
    GOLD_LABEL_BAND_NAME, PATCH_SIZE, RANDOM_SEED
)
from src.data_loader import load_geotiff_by_name
from src.combined_loss_system import CombinedFocalJaccardLoss
from src.model import SparseMeanIoU, SparseMetricWrapper, SparseF1Score
from src.per_class_metrics import (
    calculate_per_class_metrics_simple, macro_average,
    analyze_class_distribution, print_class_distribution
)
from src.optimized_hyperparameters import load_optimized_parameters
from src.transfer_learning_config import load_finetuning_parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOLD_TO_3CLASS = {0: 0, 1: 1, 2: 2, 3: 2}
NUM_CLASSES = 3
CLASS_NAMES = ['Background', 'Forest', 'Coffee']

DEFAULT_BASE_MODEL = os.path.join(MODEL_FOLDER, 'unet_finetuned_gold.h5')
DEFAULT_OUTPUT_MODEL = os.path.join(MODEL_FOLDER, 'unet_finetuned_gold_3class.h5')
HISTORY_PATH = os.path.join(MODEL_FOLDER, 'finetuning_history_3class.json')
RESULTS_PATH = os.path.join(BASE_DIR, 'results_brazil_columbia_3class.md')

CUSTOM_OBJECTS = {
    'CombinedFocalJaccardLoss': CombinedFocalJaccardLoss,
    'SparseMeanIoU': SparseMeanIoU,
    'SparseMetricWrapper': SparseMetricWrapper,
    'SparseF1Score': SparseF1Score,
}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def _parse_3class(file_path, num_image_bands, table):
    data = tf.py_function(
        load_geotiff_by_name,
        [file_path, IMAGE_BAND_NAMES, GOLD_LABEL_BAND_NAME],
        tf.float32
    )
    image = tf.transpose(data[:num_image_bands], [1, 2, 0])
    mask = tf.transpose(data[num_image_bands:], [1, 2, 0])

    image = tf.where(tf.math.is_nan(image), 0.0, image)
    mask = table.lookup(tf.cast(mask, tf.int32))

    image.set_shape([PATCH_SIZE, PATCH_SIZE, num_image_bands])
    mask.set_shape([PATCH_SIZE, PATCH_SIZE, 1])

    # Same scaling as src/data_loader.py so inputs match what the base model saw.
    image = image / 255.0
    return image, mask


def create_3class_dataset(file_paths, batch_size, shuffle=False):
    num_image_bands = len(IMAGE_BAND_NAMES)
    # Unknown label values fall back to background, consistent with src/data_loader.py.
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(GOLD_TO_3CLASS.keys()), dtype=tf.int32),
            tf.constant(list(GOLD_TO_3CLASS.values()), dtype=tf.int32),
        ),
        default_value=0,
    )

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        dataset = dataset.shuffle(len(file_paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda x: _parse_3class(x, num_image_bands, table),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# -----------------------------------------------------------------------------
# Model surgery
# -----------------------------------------------------------------------------

def base_num_classes(base_model):
    return int(base_model.get_layer('output').get_weights()[1].shape[0])


def build_3class_model(base_model):
    """Replace the 4-class softmax head with a 3-class head (coffee merged)."""
    old_head = base_model.get_layer('output')
    kernel, bias = old_head.get_weights()

    # P(coffee) = P(unshaded) + P(shaded). With logits z2, z3 close together,
    # exp(z2) + exp(z3) ~= 2 * exp(mean(z2, z3)), hence the +log(2) on the bias.
    new_kernel = np.stack([
        kernel[..., 0],
        kernel[..., 1],
        0.5 * (kernel[..., 2] + kernel[..., 3]),
    ], axis=-1)
    new_bias = np.array([
        bias[0],
        bias[1],
        0.5 * (bias[2] + bias[3]) + np.log(2.0),
    ], dtype=bias.dtype)

    features = old_head.input
    new_head = tf.keras.layers.Conv2D(
        NUM_CLASSES, (1, 1), activation='softmax', name='output'
    )
    outputs = new_head(features)
    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name='U-Net-3class')
    new_head.set_weights([new_kernel, new_bias])
    return model


def build_collapsed_baseline(base_model):
    """Exact 3-class view of the 4-class base model (sums coffee probabilities)."""
    probs = base_model.output
    collapsed = tf.keras.layers.Lambda(
        lambda p: tf.concat([p[..., 0:1], p[..., 1:2], p[..., 2:3] + p[..., 3:4]], axis=-1),
        name='collapse_coffee',
    )(probs)
    return tf.keras.Model(inputs=base_model.inputs, outputs=collapsed, name='U-Net-4to3')


def compile_model(model, learning_rate, class_weights, loss_cfg, optimal_params):
    loss_function = CombinedFocalJaccardLoss(
        class_weights=class_weights,
        alpha=float(loss_cfg.get('alpha', 1.0)),
        gamma=optimal_params['gamma'],
        focal_weight=optimal_params['focal_weight'],
        jaccard_weight=optimal_params['jaccard_weight'],
        dice_weight=optimal_params.get('dice_weight', 0.3),
        smooth=float(loss_cfg.get('smooth', 1e-5)),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=[
            'accuracy',
            SparseMeanIoU(num_classes=NUM_CLASSES, name='mean_iou'),
            SparseMetricWrapper(tf.keras.metrics.Precision(name='precision')),
            SparseMetricWrapper(tf.keras.metrics.Recall(name='recall')),
            SparseF1Score(num_classes=NUM_CLASSES, name='f1_score'),
        ],
    )


# -----------------------------------------------------------------------------
# Evaluation / reporting
# -----------------------------------------------------------------------------

def evaluate(model, dataset, label):
    print(f"\nEvaluating {label}...")
    metrics = model.evaluate(dataset, verbose=0, return_dict=True)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    per_class = calculate_per_class_metrics_simple(model, dataset, NUM_CLASSES)
    macro = macro_average(per_class)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:<11} IoU={per_class['per_class_iou'][i]:.4f} "
            f"F1={per_class['per_class_f1'][i]:.4f}"
        )
    print(f"  Macro IoU={macro['macro_iou']:.4f}  Macro F1={macro['macro_f1']:.4f}")
    return {'metrics': metrics, 'per_class': per_class, 'macro': macro}


def _write_per_class_table(f, result):
    per_class, macro = result['per_class'], result['macro']
    f.write("| Class | Accuracy | IoU | Precision | Recall | F1 |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(
            f"| {name} | {per_class['per_class_accuracy'][i]:.4f} | "
            f"{per_class['per_class_iou'][i]:.4f} | "
            f"{per_class['per_class_precision'][i]:.4f} | "
            f"{per_class['per_class_recall'][i]:.4f} | "
            f"{per_class['per_class_f1'][i]:.4f} |\n"
        )
    f.write(
        f"| **Macro** | **{macro['macro_accuracy']:.4f}** | **{macro['macro_iou']:.4f}** | "
        f"**{macro['macro_precision']:.4f}** | **{macro['macro_recall']:.4f}** | "
        f"**{macro['macro_f1']:.4f}** |\n"
    )


def _write_comparison(f, base, finetuned):
    f.write("| Metric | Base | Fine-tuned 3-class | Δ |\n")
    f.write("|---|---:|---:|---:|\n")
    for key, label in (
        ('macro_accuracy', 'Macro accuracy'),
        ('macro_iou', 'Macro IoU'),
        ('macro_precision', 'Macro precision'),
        ('macro_recall', 'Macro recall'),
        ('macro_f1', 'Macro F1'),
    ):
        b, n = base['macro'][key], finetuned['macro'][key]
        f.write(f"| {label} | {b:.4f} | {n:.4f} | {n - b:+.4f} |\n")


def write_results(results, run_info):
    with open(RESULTS_PATH, 'w') as f:
        f.write("# Brazil/Colombia 3-class Fine-tuning Results\n\n")
        f.write("Classes: Background, Forest, Coffee (unshaded + shaded merged).\n\n")
        f.write(
            "A 4-class base model is evaluated by summing its unshaded and shaded coffee "
            "probabilities, so both models are compared on the same 3-class labels.\n"
        )
        for split in ('validation', 'test'):
            if split not in results:
                continue
            base, finetuned = results[split]['base'], results[split]['finetuned']
            f.write(f"\n## Gold {split.capitalize()}\n")
            f.write(f"\n### {run_info['Base model head']}\n")
            _write_per_class_table(f, base)
            f.write("\n### Fine-tuned 3-class model\n")
            _write_per_class_table(f, finetuned)
            f.write("\n### Macro comparison\n")
            _write_comparison(f, base, finetuned)

        f.write("\n## Training Info\n")
        for key, value in run_info.items():
            f.write(f"- {key}: {value}\n")
    print(f"\nResults summary saved to: {RESULTS_PATH}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune unet_finetuned_gold.h5 to 3 classes (coffee merged)."
    )
    parser.add_argument('--base-model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--output-model', default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument('--epochs', type=int, default=None,
                        help="Defaults to finetuning_config.json")
    parser.add_argument('--learning-rate', type=float, default=None,
                        help="Defaults to finetuning_config.json")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Defaults to finetuning_config.json")
    parser.add_argument('--class-weights', type=float, nargs=3, default=None,
                        metavar=('BG', 'FOREST', 'COFFEE'),
                        help="Defaults to finetuning_config.json weights with coffee averaged")
    parser.add_argument('--skip-test', action='store_true',
                        help="Do not evaluate on the gold test split")
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("=" * 70)
    print("BRAZIL/COLOMBIA FINE-TUNING: 3 CLASSES (BACKGROUND, FOREST, COFFEE)")
    print("=" * 70)

    # --- 1. Configuration (read-only) ---
    ft = load_finetuning_parameters()
    epochs = args.epochs or int(ft['epochs'])
    learning_rate = args.learning_rate or float(ft['learning_rate'])
    batch_size = args.batch_size or int(ft['batch_size'])
    es_cfg = ft['early_stopping']
    ckpt_cfg = ft['model_checkpoint']
    rlrop_cfg = ft['reduce_lr_on_plateau']
    loss_cfg = ft['loss']

    if args.class_weights:
        class_weights = list(args.class_weights)
    else:
        w = list(ft['class_weights'])
        class_weights = w[:2] + [float(np.mean(w[2:4]))] if len(w) == 4 else w
    if len(class_weights) != NUM_CLASSES:
        raise ValueError(f"Need {NUM_CLASSES} class weights, got {class_weights}")

    with open(GOLD_DATA_SPLIT_PATH, 'r') as f:
        splits = json.load(f)
    train_files = splits['train']
    val_files = splits['validation']
    test_files = [] if args.skip_test else splits.get('test', [])

    print(f"Label mapping (gold -> model): {GOLD_TO_3CLASS}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Files: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    print(f"LR={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    print(f"Class weights: {class_weights}")

    # --- 2. Datasets ---
    train_dataset = create_3class_dataset(train_files, batch_size, shuffle=True)
    val_dataset = create_3class_dataset(val_files, batch_size)
    test_dataset = create_3class_dataset(test_files, batch_size) if test_files else None

    print_class_distribution(
        analyze_class_distribution(create_3class_dataset(train_files, batch_size), NUM_CLASSES),
        CLASS_NAMES,
    )

    # --- 3. Models ---
    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model not found at '{args.base_model}'.")
    print(f"\nLoading base model from: {args.base_model}")
    base_model = tf.keras.models.load_model(
        args.base_model, custom_objects=CUSTOM_OBJECTS, compile=False
    )

    optimal_params = load_optimized_parameters()

    n_base = base_num_classes(base_model)
    if n_base == 4:
        print("Base model has a 4-class head -> merging coffee channels into a 3-class head.")
        baseline = build_collapsed_baseline(base_model)
        compile_model(baseline, learning_rate, class_weights, loss_cfg, optimal_params)
        model = build_3class_model(base_model)
        base_label = "base model (4→3)"
    elif n_base == NUM_CLASSES:
        print("Base model already has a 3-class head -> fine-tuning it directly.")
        # Evaluated before training, so its numbers are the pre-fine-tuning baseline.
        baseline = base_model
        model = base_model
        base_label = "base model (3-class)"
    else:
        raise ValueError(f"Unsupported base model head with {n_base} output channels.")
    compile_model(model, learning_rate, class_weights, loss_cfg, optimal_params)

    # --- 4. Baseline evaluation (before fine-tuning) ---
    results = {'validation': {}, 'test': {}}
    results['validation']['base'] = evaluate(baseline, val_dataset, f"{base_label} on gold validation")
    if test_dataset is not None:
        results['test']['base'] = evaluate(baseline, test_dataset, f"{base_label} on gold test")
    if n_base == 4:
        evaluate(model, val_dataset, "initialised 3-class head on gold validation")

    # --- 5. Fine-tuning ---
    es_start_epoch = int(es_cfg.get('start_from_epoch', 0))
    checkpoint_path = args.output_model.replace('.h5', '_checkpoint.h5')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg.get('monitor', 'val_mean_iou'),
            mode=es_cfg.get('mode', 'max'),
            patience=int(es_cfg.get('patience', 10)),
            restore_best_weights=bool(es_cfg.get('restore_best_weights', True)),
            start_from_epoch=es_start_epoch,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=ckpt_cfg.get('monitor', 'val_mean_iou'),
            mode=ckpt_cfg.get('mode', 'max'),
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=rlrop_cfg.get('monitor', 'val_loss'),
            factor=float(rlrop_cfg.get('factor', 0.2)),
            patience=int(rlrop_cfg.get('patience', 5)),
            min_lr=float(rlrop_cfg.get('min_lr', 1e-7)),
            verbose=1,
        ),
    ]

    print("\nStarting 3-class fine-tuning...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
    )

    with open(HISTORY_PATH, 'w') as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2,
        )
    print(f"History saved to: {HISTORY_PATH}")

    model.save(args.output_model)
    print(f"Fine-tuned 3-class model saved to: {args.output_model}")

    # --- 6. Evaluation after fine-tuning ---
    results['validation']['finetuned'] = evaluate(model, val_dataset, "fine-tuned model on gold validation")
    if test_dataset is not None:
        results['test']['finetuned'] = evaluate(model, test_dataset, "fine-tuned model on gold test")
    else:
        results.pop('test')

    for split, res in results.items():
        print(f"\nMacro comparison on gold {split} ({base_label} -> fine-tuned 3-class):")
        for key in ('macro_accuracy', 'macro_iou', 'macro_precision', 'macro_recall', 'macro_f1'):
            print(f"  {key}: {res['base']['macro'][key]:.4f} -> {res['finetuned']['macro'][key]:.4f}")

    write_results(results, {
        'Base model': args.base_model,
        'Base model head': base_label,
        'Label mapping (gold -> model)': GOLD_TO_3CLASS,
        'Gold training files': len(train_files),
        'Gold validation files': len(val_files),
        'Gold test files': len(test_files),
        'Epochs (max)': epochs,
        'Epochs actually run': len(history.history.get('loss', [])),
        'Learning rate': learning_rate,
        'Batch size': batch_size,
        'Class weights': class_weights,
        'Early stopping': (
            f"monitor {es_cfg.get('monitor', 'val_mean_iou')}, "
            f"patience {es_cfg.get('patience', 10)}, start_from_epoch {es_start_epoch}"
        ),
        'History': HISTORY_PATH,
        'Model saved to': args.output_model,
    })

    print(f"\nSUCCESS: 3-class fine-tuning complete. Model saved to '{args.output_model}'")


if __name__ == "__main__":
    main()
