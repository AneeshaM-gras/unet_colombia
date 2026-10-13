#!/usr/bin/env python3
"""
Optimized Training Script with Combined Loss + Hyperparameter Optimization
Uses the best parameters found through Bayesian optimization
"""

import os
import tensorflow as tf
import json
import numpy as np
import datetime

from src.config import (
    FINAL_PATCHES_FOLDER, IMAGE_BAND_NAMES, LABEL_BAND_NAME,
    NUM_IMAGE_BANDS, PATCH_SIZE, TRAINING_CONFIG_PATH, MODEL_FOLDER,
    DATA_SPLIT_PATH
)
from src.optimized_hyperparameters import load_optimized_parameters
from src.data_loader import create_dataset
from src.optimized_model import unet
from src.combined_loss_system import CombinedFocalJaccardLoss
from src.per_class_metrics import (
    calculate_per_class_metrics_simple, print_per_class_metrics,
    analyze_class_distribution, print_class_distribution
)

def load_datasets_from_split(batch_size, shuffle_buffer_size=500):
    """Loads train and validation datasets from the pre-defined split file.
    
    Args:
        batch_size: Batch size for training
        shuffle_buffer_size: Size of shuffle buffer (default 500, good for 48GB RAM)
    """
    print("Loading pre-defined data splits...")
    if not os.path.exists(DATA_SPLIT_PATH):
        raise FileNotFoundError(f"Data split file not found at '{DATA_SPLIT_PATH}'. Run '01_prepare_environment.py' first.")
    
    with open(DATA_SPLIT_PATH, 'r') as f:
        splits = json.load(f)
    
    # Create datasets with proper pipeline: map -> shuffle -> batch -> prefetch
    from src.data_loader import parse_function
    num_image_bands = len(IMAGE_BAND_NAMES)
    
    # Training dataset: shuffle individual samples before batching
    train_dataset = tf.data.Dataset.from_tensor_slices(splits['train'])
    train_dataset = train_dataset.map(
        lambda x: parse_function(x, IMAGE_BAND_NAMES, LABEL_BAND_NAME, num_image_bands),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(shuffle_buffer_size)  # Shuffle BEFORE batching
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Validation dataset: no shuffle needed
    val_dataset = tf.data.Dataset.from_tensor_slices(splits['validation'])
    val_dataset = val_dataset.map(
        lambda x: parse_function(x, IMAGE_BAND_NAMES, LABEL_BAND_NAME, num_image_bands),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


class OptimizedCombinedLossCallback(tf.keras.callbacks.Callback):
    """Callback for monitoring optimized combined loss performance"""
    
    def __init__(self, val_dataset, num_classes, class_names=None, log_frequency=3):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        self.log_frequency = log_frequency
        self.best_iou = 0.0
        self.best_rare_class_iou = 0.0
        self.best_composite_score = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_frequency == 0:
            # Calculate per-class metrics
            results = calculate_per_class_metrics_simple(self.model, self.val_dataset, self.num_classes)
            
            # Print per-class metrics
            print(f"\nEpoch {epoch + 1} - Per-Class Metrics:")
            print_per_class_metrics(results, self.class_names)
            
            # Calculate composite score
            avg_iou = np.mean(results['per_class_iou'])
            # Instead of hard-coding [3, 4], we'll calculate the mean IoU
            # for all non-background classes. This is robust and adapts to any number of classes.
            if self.num_classes > 1:
                non_background_ious = results['per_class_iou'][1:]
                rare_class_iou = np.mean(non_background_ious)
            else:
                rare_class_iou = 0.0 # Avoid errors if there's only one class

            iou_std = np.std(results['per_class_iou'])
            
            composite_score = (0.6 * avg_iou + 
                             0.4 * (1.0 - min(iou_std, 0.5)))
            
            # Track best scores
            if avg_iou > self.best_iou:
                self.best_iou = avg_iou
                print(f"New best average IoU: {avg_iou:.4f}")
            
            if rare_class_iou > self.best_rare_class_iou:
                self.best_rare_class_iou = rare_class_iou
                print(f"New best rare class IoU: {rare_class_iou:.4f}")
            
            if composite_score > self.best_composite_score:
                self.best_composite_score = composite_score
                print(f"New best composite score: {composite_score:.4f}")
            
            # Check for balanced predictions
            self.check_prediction_balance(results)
            
            # Add to logs
            if logs is not None:
                for i, (acc, iou) in enumerate(zip(results['per_class_accuracy'], results['per_class_iou'])):
                    logs[f'per_class_accuracy_class_{i}'] = acc
                    logs[f'per_class_iou_class_{i}'] = iou
                logs['avg_per_class_iou'] = avg_iou
                logs['avg_rare_class_iou'] = rare_class_iou
                logs['composite_score'] = composite_score
    
    def check_prediction_balance(self, results):
        """Check for balanced predictions"""
        print("Balance Check:")
        
        # Check overall balance
        ious = results['per_class_iou']
        iou_std = np.std(ious)
        if iou_std > 0.3:
            print(f"  WARNING: High class imbalance (IoU std: {iou_std:.3f})")
        else:
            print(f"  SUCCESS: Good class balance (IoU std: {iou_std:.3f})")

def main():
    print("OPTIMIZED COMBINED LOSS TRAINING")
    print("=" * 60)
    
    # --- 0. Configure GPU Memory (if using GPU) ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected, using CPU")
    
    # --- 1. Load Configuration ---
    with open(TRAINING_CONFIG_PATH, 'r') as f:
        training_config = json.load(f)
    NUM_CLASSES = training_config['num_classes']
    CLASS_WEIGHTS = training_config['class_weights']
    CLASS_NAMES = training_config['class_names'] # DYNAMIC
    loss_config = training_config.get('loss', {})

    if 'batch_size' not in training_config:
        raise KeyError("'batch_size' missing from training_config.json")
    batch_size = int(training_config['batch_size'])

    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class weights: {CLASS_WEIGHTS}")
    print(f"Class names: {CLASS_NAMES}")

    # --- 2. Load Optimized Parameters ---
    # Batch size comes from training_config.json (not hardcoded / not HPO JSON alone).
    optimal_params = load_optimized_parameters()

    print("Using optimized hyperparameters:")
    print(f"  Learning Rate: {optimal_params['learning_rate']:.6f}")
    print(f"  Batch Size: {batch_size} (from training_config.json)")
    print(f"  Focal Weight: {optimal_params['focal_weight']:.3f}")
    print(f"  Jaccard Weight: {optimal_params['jaccard_weight']:.3f}")
    print(f"  Gamma: {optimal_params['gamma']:.3f}")
    print(f"  Dropout Encoder: {optimal_params['dropout_encoder']:.3f}")
    print(f"  Dropout Decoder: {optimal_params['dropout_decoder']:.3f}")

    # --- 3. Load Training Dataset ---
    train_dataset, val_dataset = load_datasets_from_split(batch_size)

    # --- 5. Analyze Class Distribution ---
    print("\nAnalyzing class distribution...")

    train_distribution = analyze_class_distribution(train_dataset, NUM_CLASSES)
    print("Training Data Distribution:")
    print_class_distribution(train_distribution, CLASS_NAMES)

    # --- 6. Build Model ---
    print("\nBuilding U-Net...")
    
    INPUT_SHAPE = (PATCH_SIZE, PATCH_SIZE, NUM_IMAGE_BANDS)
    model = unet(
        input_shape=INPUT_SHAPE, 
        n_classes=NUM_CLASSES,
        dropout_encoder=optimal_params['dropout_encoder'],
        dropout_decoder=optimal_params['dropout_decoder']
    )
    
    print(f"Model created with {model.count_params():,} parameters")
    model.summary()

    # --- 7. Compile Model with Optimized Combined Loss ---
    print("\nCompiling model with optimized Combined Focal + Jaccard Loss...")
    
    # Use optimized combined loss
    loss_function = CombinedFocalJaccardLoss(
        class_weights=CLASS_WEIGHTS,
        alpha=float(loss_config.get('alpha', 1.0)),
        gamma=optimal_params['gamma'],
        focal_weight=optimal_params['focal_weight'],
        jaccard_weight=optimal_params['jaccard_weight'],
        dice_weight=optimal_params.get('dice_weight', 0.3),
        smooth=float(loss_config.get('smooth', 1e-5)),
    )
    
    # Use optimized learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimal_params['learning_rate'])
    
    # Metrics
    from src.model import SparseMeanIoU, SparseMetricWrapper, SparseF1Score
    metrics = [
        'accuracy',
        SparseMeanIoU(num_classes=NUM_CLASSES, name='mean_iou'),
        SparseMetricWrapper(tf.keras.metrics.Precision(name='precision')),
        SparseMetricWrapper(tf.keras.metrics.Recall(name='recall')),
        SparseF1Score(num_classes=NUM_CLASSES, name='f1_score')
    ]
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    print("Model compiled with optimized Combined Focal + Jaccard Loss")

    # --- 8. Setup Callbacks ---
    print("\nSetting up callbacks...")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    callbacks = []
    
    # Model checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_FOLDER, "unet_optimized_combined_loss.h5"),
        monitor='val_mean_iou',
        mode='max',
        save_best_only=True,
        verbose=1
    ))
    
    # Early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=int(optimal_params['early_stopping_patience']),
        restore_best_weights=True,
        verbose=1
    ))
    
    # Learning rate scheduling
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=optimal_params['reduce_lr_factor'],
        patience=int(optimal_params['reduce_lr_patience']),
        min_lr=optimal_params['min_lr'],
        verbose=1
    ))
    
    # TensorBoard
    log_dir = os.path.join(MODEL_FOLDER, "logs_optimized_combined", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    ))
    
    # Optimized combined loss metrics callback
    per_class_callback = OptimizedCombinedLossCallback(
        val_dataset, NUM_CLASSES, CLASS_NAMES, log_frequency=3
    )
    callbacks.append(per_class_callback)
    
    print(f"{len(callbacks)} callbacks configured")

    # --- 9. Train Model ---
    print(f"\nStarting optimized combined loss training for {optimal_params['epochs']} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {optimal_params['learning_rate']:.6f}")
    print(f"Loss function: Combined Focal ({optimal_params['focal_weight']:.3f}) + Jaccard ({optimal_params['jaccard_weight']:.3f})")
    print(f"Gamma: {optimal_params['gamma']:.3f}")
    print(f"Architecture: U-Net")
    print(f"Focus: Optimized class imbalance + IoU optimization")
    
    start_time = datetime.datetime.now()
    
    history = model.fit(
        train_dataset,
        epochs=int(optimal_params['epochs']),
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    print(f"Training completed in {training_time}")

    # --- 10. Final Evaluation ---
    print("\nFinal evaluation...")
    
    # Calculate final per-class metrics
    final_results = calculate_per_class_metrics_simple(model, val_dataset, NUM_CLASSES)
    print("\nFinal Per-Class Metrics:")
    print_per_class_metrics(final_results, CLASS_NAMES)
    
    # Training summary
    best_val_acc = max(history.history['val_accuracy'])
    best_val_iou = max(history.history['val_mean_iou']) if 'val_mean_iou' in history.history else 0
    best_val_f1 = max(history.history['val_f1_score']) if 'val_f1_score' in history.history else 0
    final_val_loss = history.history['val_loss'][-1]
    
    avg_iou = np.mean(final_results['per_class_iou'])
    iou_std = np.std(final_results['per_class_iou'])

    if NUM_CLASSES > 1:
        avg_non_background_iou = np.mean(final_results['per_class_iou'][1:])
    else:
        avg_non_background_iou = 0.0

    composite_score = (0.6 * avg_iou + 0.4 * (1.0 - min(iou_std, 0.5)))
    
    print(f"\nTraining Summary:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"   Best Validation IoU: {best_val_iou:.4f}")
    print(f"   Best Validation F1 Score: {best_val_f1:.4f}")
    print(f"   Final Validation Loss: {final_val_loss:.4f}")
    print(f"   Average Per-Class IoU: {avg_iou:.4f}")
    print(f"   Average Non-Background IoU: {avg_non_background_iou:.4f}")
    print(f"   Composite Score: {composite_score:.4f}")
    
    print(f"\nFinal Balance Analysis:")
    print(f"   IoU Standard Deviation: {iou_std:.4f} (target: < 0.3)")
    
    # Generalize success criteria
    success_criteria = [
        iou_std < 0.3,
        avg_non_background_iou > 0.25,
        composite_score > 0.45
    ]

    success_count = sum(success_criteria)
    print(f"\nSuccess Criteria Met: {success_count}/5")
    if success_count >= 4:
        print("SUCCESS: Optimized combined loss training highly successful!")
    elif success_count >= 3:
        print("SUCCESS: Optimized combined loss training successful!")
    else:
        print("WARNING: Some issues remain, consider running optimization again")

    # --- 11. Save Results ---
    print("\nSaving results...")
    
    # Save training history
    history_path = os.path.join(MODEL_FOLDER, "training_history_optimized_combined.json")
    with open(history_path, 'w') as f:
            # new dictionary and explicitly cast all values to standard Python floats.
            serializable_history = {k: [float(vi) for vi in v] for k, v in history.history.items()}
            json.dump(serializable_history, f, indent=4)
    
    # Save per-class metrics
    metrics_path = os.path.join(MODEL_FOLDER, "per_class_metrics_optimized_combined.json")
    with open(metrics_path, 'w') as f:

        json.dump({
            'per_class_accuracy': final_results['per_class_accuracy'].tolist(),
            'per_class_iou': final_results['per_class_iou'].tolist(),
            'training_summary': {
                'best_val_accuracy': float(best_val_acc),
                'best_val_iou': float(best_val_iou),
                'best_val_f1_score': float(best_val_f1),
                'final_val_loss': float(final_val_loss),
                'training_time': str(training_time),
                'model_parameters': int(model.count_params()),
                'avg_non_background_iou': float(avg_non_background_iou),
                'iou_std': float(iou_std),
                'composite_score': float(composite_score),
                'success_criteria_met': f"{int(success_count)}/{len(success_criteria)}",
                'optimal_parameters': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in optimal_params.items()}
            }
        }, f, indent=4)
    
    print(f"Training history saved to: {history_path}")
    print(f"Per-class metrics saved to: {metrics_path}")
    print(f"Model saved to: {os.path.join(MODEL_FOLDER, 'unet_optimized_combined_loss.h5')}")
    
    print("\nOPTIMIZED COMBINED LOSS TRAINING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
