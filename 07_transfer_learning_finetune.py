# 07_transfer_learning_finetune.py

import os
import json
import tensorflow as tf
from src.config import (
    MODEL_FOLDER, GOLD_DATA_SPLIT_PATH, IMAGE_BAND_NAMES,
    TRAINING_CONFIG_PATH, GOLD_LABEL_BAND_NAME
)
from src.data_loader import create_dataset
from src.combined_loss_system import CombinedFocalJaccardLoss
from src.model import SparseMeanIoU, SparseMetricWrapper, SparseF1Score
from src.per_class_metrics import calculate_per_class_metrics_simple, macro_average
from src.optimized_hyperparameters import load_optimized_parameters
from src.transfer_learning_config import load_finetuning_parameters

def main():
    """
    Fine-tuning workflow using gold data with the SAME class aggregation
    as the base model. Hyperparameters come from finetuning_config.json
    (optionally populated by run_finetuning_hyperparameter_optimization.py).
    """
    print("=" * 70)
    print("FINE-TUNING WORKFLOW (CONSISTENT AGGREGATION)")
    print("=" * 70)

    # --- 1. Load Configurations ---
    with open(TRAINING_CONFIG_PATH, 'r') as f:
        training_config = json.load(f)
    NUM_CLASSES = training_config['num_classes']
    class_names = training_config.get('class_names', [f'Class {i}' for i in range(NUM_CLASSES)])

    finetune_params = load_finetuning_parameters()
    CLASS_WEIGHTS = finetune_params.get(
        'class_weights', training_config.get('class_weights')
    )
    batch_size = int(finetune_params['batch_size'])
    fine_tune_lr = float(finetune_params['learning_rate'])
    fine_tune_epochs = int(finetune_params['epochs'])
    shuffle_buffer = int(finetune_params['shuffle_buffer_size'])
    base_model_path = finetune_params['base_model_path']
    finetuned_model_path = finetune_params['finetuned_model_path']
    es_cfg = finetune_params['early_stopping']
    ckpt_cfg = finetune_params['model_checkpoint']
    rlrop_cfg = finetune_params['reduce_lr_on_plateau']
    loss_cfg = finetune_params['loss']

    # Load the gold data splits
    with open(GOLD_DATA_SPLIT_PATH, 'r') as f:
        gold_splits = json.load(f)
    train_files = gold_splits['train']
    val_files = gold_splits['validation']

    print(f"Loaded configuration for a {NUM_CLASSES}-class problem.")
    print(f"Found {len(train_files)} training and {len(val_files)} validation files for fine-tuning.")
    print(f"Fine-tune LR={fine_tune_lr}, batch_size={batch_size}, epochs={fine_tune_epochs}")
    print(f"Class weights (frozen): {CLASS_WEIGHTS}")

    # --- 2. Load the Pre-trained Base Model ---
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found at '{base_model_path}'. Run 02_...py first.")
    
    print(f"Loading base model from: {base_model_path}")
    model = tf.keras.models.load_model(
        base_model_path,
        custom_objects={
            'CombinedFocalJaccardLoss': CombinedFocalJaccardLoss,
            'SparseMeanIoU': SparseMeanIoU
        },
        compile=False
    )

    # --- 3. Prepare Datasets WITH Aggregation ---
    train_dataset = create_dataset(
        train_files, batch_size, IMAGE_BAND_NAMES, GOLD_LABEL_BAND_NAME
    ).shuffle(shuffle_buffer)
    val_dataset = create_dataset(
        val_files, batch_size, IMAGE_BAND_NAMES, GOLD_LABEL_BAND_NAME
    )
    print("Gold datasets created with on-the-fly class aggregation.")

    # --- 4. Compile the Model for Fine-tuning ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    
    optimal_params = load_optimized_parameters()
    loss_function = CombinedFocalJaccardLoss(
        class_weights=CLASS_WEIGHTS,
        alpha=float(loss_cfg.get('alpha', 1.0)),
        gamma=optimal_params['gamma'],
        focal_weight=optimal_params['focal_weight'],
        jaccard_weight=optimal_params['jaccard_weight'],
        dice_weight=optimal_params.get('dice_weight', 0.3),
        smooth=float(loss_cfg.get('smooth', 1e-5)),
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[
            'accuracy',
            SparseMeanIoU(num_classes=NUM_CLASSES, name='mean_iou'),
            SparseMetricWrapper(tf.keras.metrics.Precision(name='precision')),
            SparseMetricWrapper(tf.keras.metrics.Recall(name='recall')),
            SparseF1Score(num_classes=NUM_CLASSES, name='f1_score')
        ]
    )
    print(f"\nModel re-compiled for fine-tuning with learning rate: {fine_tune_lr}")

    # --- 4B. Evaluate Base Model (Before Fine-tuning) ---
    print("\nEvaluating base model on gold validation set (before fine-tuning)...")
    base_metrics = model.evaluate(val_dataset, verbose=0, return_dict=True)
    base_f1 = base_metrics.get('f1_score')
    for name, value in base_metrics.items():
        print(f"  {name}: {value:.4f}")
    base_per_class = calculate_per_class_metrics_simple(model, val_dataset, NUM_CLASSES)
    base_macro = macro_average(base_per_class)

    # --- 5. Set Up Callbacks and Run Fine-tuning ---
    es_start_epoch = int(es_cfg.get('start_from_epoch', 0))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg.get('monitor', 'val_mean_iou'),
            mode=es_cfg.get('mode', 'max'),
            patience=int(es_cfg.get('patience', 10)),
            verbose=1,
            restore_best_weights=bool(es_cfg.get('restore_best_weights', True)),
            start_from_epoch=es_start_epoch,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=rlrop_cfg.get('monitor', 'val_loss'),
            factor=float(rlrop_cfg.get('factor', 0.2)),
            patience=int(rlrop_cfg.get('patience', 5)),
            min_lr=float(rlrop_cfg.get('min_lr', 1e-7)),
            verbose=1,
        ),
    ]
    if bool(ckpt_cfg.get('save_best_only', True)):
        # Checkpoint to a crash-recovery path; the deployed file is written after
        # fit so it always matches the weights reported below.
        checkpoint_path = finetuned_model_path.replace('.h5', '_checkpoint.h5')
        callbacks.insert(
            1,
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=ckpt_cfg.get('monitor', 'val_mean_iou'),
                mode=ckpt_cfg.get('mode', 'max'),
                save_best_only=True,
            ),
        )

    print("\nStarting fine-tuning on aggregated gold dataset...")
    print(f"Early stopping monitors {es_cfg.get('monitor', 'val_mean_iou')} "
          f"from epoch {es_start_epoch + 1} with patience {es_cfg.get('patience', 10)}")
    history = model.fit(
        train_dataset,
        epochs=fine_tune_epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    history_path = os.path.join(MODEL_FOLDER, 'finetuning_history.json')
    with open(history_path, 'w') as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )
    print(f"Fine-tuning history saved to: {history_path}")

    model.save(finetuned_model_path)
    print(f"Fine-tuned model saved to: {finetuned_model_path}")

    # --- 6. Evaluate Fine-tuned Model ---
    print("\nEvaluating fine-tuned model on gold validation set...")
    finetuned_metrics = model.evaluate(val_dataset, verbose=0, return_dict=True)
    finetuned_f1 = finetuned_metrics.get('f1_score')
    for name, value in finetuned_metrics.items():
        print(f"  {name}: {value:.4f}")
    finetuned_per_class = calculate_per_class_metrics_simple(model, val_dataset, NUM_CLASSES)
    finetuned_macro = macro_average(finetuned_per_class)

    print("\nMacro comparison on gold validation (base -> fine-tuned):")
    for key in ('macro_accuracy', 'macro_iou', 'macro_precision', 'macro_recall', 'macro_f1'):
        print(f"  {key}: {base_macro[key]:.4f} -> {finetuned_macro[key]:.4f}")

    # --- 7. Save Results Summary ---
    results_path = os.path.join(os.path.dirname(__file__), "results.md")
    with open(results_path, "w") as f:
        f.write("# Fine-tuning Results Summary\n\n")

        f.write("## Base Model (Before Fine-tuning)\n")
        f.write(f"- accuracy: {base_metrics.get('accuracy', 0.0):.6f}\n")
        f.write(f"- mean_iou: {base_metrics.get('mean_iou', 0.0):.6f}\n")
        f.write(f"- precision: {base_metrics.get('precision', 0.0):.6f}\n")
        f.write(f"- recall: {base_metrics.get('recall', 0.0):.6f}\n")
        if base_f1 is not None:
            f.write(f"- f1_score: {base_f1:.6f}\n")

        f.write("\n### Per-class Metrics (Before Fine-tuning)\n")
        f.write("| Class | Accuracy | IoU | Precision | Recall | F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for i, name in enumerate(class_names):
            f.write(
                f"| {name} | "
                f"{base_per_class['per_class_accuracy'][i]:.4f} | "
                f"{base_per_class['per_class_iou'][i]:.4f} | "
                f"{base_per_class['per_class_precision'][i]:.4f} | "
                f"{base_per_class['per_class_recall'][i]:.4f} | "
                f"{base_per_class['per_class_f1'][i]:.4f} |\n"
            )
        f.write(
            f"| **Macro** | **{base_macro['macro_accuracy']:.4f}** | "
            f"**{base_macro['macro_iou']:.4f}** | "
            f"**{base_macro['macro_precision']:.4f}** | "
            f"**{base_macro['macro_recall']:.4f}** | "
            f"**{base_macro['macro_f1']:.4f}** |\n"
        )

        f.write("\n## Fine-tuned Model (After Fine-tuning)\n")
        f.write(f"- accuracy: {finetuned_metrics.get('accuracy', 0.0):.6f}\n")
        f.write(f"- mean_iou: {finetuned_metrics.get('mean_iou', 0.0):.6f}\n")
        f.write(f"- precision: {finetuned_metrics.get('precision', 0.0):.6f}\n")
        f.write(f"- recall: {finetuned_metrics.get('recall', 0.0):.6f}\n")
        if finetuned_f1 is not None:
            f.write(f"- f1_score: {finetuned_f1:.6f}\n")

        f.write("\n### Per-class Metrics (After Fine-tuning)\n")
        f.write("| Class | Accuracy | IoU | Precision | Recall | F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for i, name in enumerate(class_names):
            f.write(
                f"| {name} | "
                f"{finetuned_per_class['per_class_accuracy'][i]:.4f} | "
                f"{finetuned_per_class['per_class_iou'][i]:.4f} | "
                f"{finetuned_per_class['per_class_precision'][i]:.4f} | "
                f"{finetuned_per_class['per_class_recall'][i]:.4f} | "
                f"{finetuned_per_class['per_class_f1'][i]:.4f} |\n"
            )
        f.write(
            f"| **Macro** | **{finetuned_macro['macro_accuracy']:.4f}** | "
            f"**{finetuned_macro['macro_iou']:.4f}** | "
            f"**{finetuned_macro['macro_precision']:.4f}** | "
            f"**{finetuned_macro['macro_recall']:.4f}** | "
            f"**{finetuned_macro['macro_f1']:.4f}** |\n"
        )

        f.write("\n## Macro Comparison (Gold Validation)\n")
        f.write("| Metric | Base | Fine-tuned | Δ |\n")
        f.write("|---|---:|---:|---:|\n")
        for key, label in (
            ('macro_accuracy', 'Macro accuracy'),
            ('macro_iou', 'Macro IoU'),
            ('macro_precision', 'Macro precision'),
            ('macro_recall', 'Macro recall'),
            ('macro_f1', 'Macro F1'),
        ):
            delta = finetuned_macro[key] - base_macro[key]
            f.write(
                f"| {label} | {base_macro[key]:.4f} | "
                f"{finetuned_macro[key]:.4f} | {delta:+.4f} |\n"
            )

        f.write("\n## Training Info\n")
        f.write(f"- Gold training files: {len(train_files)}\n")
        f.write(f"- Gold validation files: {len(val_files)}\n")
        f.write(f"- Fine-tune epochs (max): {fine_tune_epochs}\n")
        f.write(f"- Epochs actually run: {len(history.history.get('loss', []))}\n")
        f.write(f"- Learning rate: {fine_tune_lr}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Class weights: {CLASS_WEIGHTS}\n")
        f.write(
            f"- Early stopping: monitor {es_cfg.get('monitor', 'val_mean_iou')}, "
            f"patience {es_cfg.get('patience', 10)}, "
            f"start_from_epoch {es_start_epoch}\n"
        )
        f.write(f"- History: {history_path}\n")
        f.write(f"- Model saved to: {finetuned_model_path}\n")
    print(f"\nResults summary saved to: {results_path}")

    print(f"\nSUCCESS: Fine-tuning complete. Best model saved to '{finetuned_model_path}'")

if __name__ == "__main__":
    main()
