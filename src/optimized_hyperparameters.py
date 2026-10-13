# src/optimized_hyperparameters.py

"""Best hyperparameters from Bayesian optimization (with file overrides)."""

import os
import json

OPTIMIZED_PARAMETERS = {
    'learning_rate': 0.001164,
    'epochs': 12,
    'focal_weight': 0.3,
    'jaccard_weight': 0.4,
    'dice_weight': 0.3,
    'gamma': 3.5,
    'smooth': 1e-5,
    'loss_alpha': 1.0,
    'shuffle_buffer_size': 500,
    'log_frequency': 3,
    'early_stopping_monitor': 'val_loss',
    'checkpoint_monitor': 'val_mean_iou',
    'dropout_encoder': 0.05,
    'dropout_decoder': 0.233,
    'l2_regularization': 1e-5,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 8,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
    'use_data_augmentation': True,
    'smart_augmentation': True,
    'model_name': 'U-Net-Optimized',
}


def load_optimized_parameters():
    """Load optimized params from HPO results JSON when available.

    Expects models/saved_models/combined_loss_optimization_results.json, typically
    produced by base HPO on gold (--data gold). Those params are then used for
    weak-label training in 02_train_optimized_combined_loss.py.

    Batch size for training/eval comes from training_config.json (not this JSON alone).
    """
    from .config import MODEL_FOLDER, get_training_batch_size

    optimization_results_path = os.path.join(
        MODEL_FOLDER, 'combined_loss_optimization_results.json'
    )

    if os.path.exists(optimization_results_path):
        try:
            with open(optimization_results_path, 'r') as f:
                results = json.load(f)

            summary = results.get('optimization_summary', results)
            best_params = summary['best_params']
            hpo_source = (
                results.get('hpo_data_source')
                or summary.get('hpo_data_source')
                or 'unknown'
            )
            optimized_params = OPTIMIZED_PARAMETERS.copy()
            optimized_params.update({
                'learning_rate': float(best_params.get('learning_rate', 0.001164)),
                # Prefer training_config batch_size; fall back to HPO best if needed
                'batch_size': int(
                    best_params.get('batch_size', get_training_batch_size())
                ),
                'focal_weight': float(best_params.get('focal_weight', 0.3)),
                'jaccard_weight': float(best_params.get('jaccard_weight', 0.4)),
                'dice_weight': float(best_params.get('dice_weight', 0.3)),
                'gamma': float(best_params.get('gamma', 3.5)),
                'dropout_encoder': float(best_params.get('dropout_encoder', 0.05)),
                'dropout_decoder': float(best_params.get('dropout_decoder', 0.233)),
                'hpo_data_source': hpo_source,
            })
            # Runtime batch size always from training_config.json
            optimized_params['batch_size'] = get_training_batch_size()
            print(
                "SUCCESS: Loaded optimized parameters from optimization results "
                f"(hpo_data_source={hpo_source})"
            )
            return optimized_params
        except Exception as e:
            print(f"WARNING: Could not load optimization results: {e}")
            print("Using default optimized parameters")

    print("Using default optimized parameters")
    params = OPTIMIZED_PARAMETERS.copy()
    params['batch_size'] = get_training_batch_size()
    return params


def get_optimized_config():
    from .config import (
        FINAL_PATCHES_FOLDER, IMAGE_BAND_NAMES, LABEL_BAND_NAME,
        NUM_IMAGE_BANDS, PATCH_SIZE, TRAINING_CONFIG_PATH, MODEL_FOLDER,
        TRAIN_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, RANDOM_SEED,
        load_training_config,
    )
    training_config = load_training_config()
    opt_params = load_optimized_parameters()
    return {
        'FINAL_PATCHES_FOLDER': FINAL_PATCHES_FOLDER,
        'IMAGE_BAND_NAMES': IMAGE_BAND_NAMES,
        'LABEL_BAND_NAME': LABEL_BAND_NAME,
        'NUM_IMAGE_BANDS': NUM_IMAGE_BANDS,
        'PATCH_SIZE': PATCH_SIZE,
        'TRAINING_CONFIG_PATH': TRAINING_CONFIG_PATH,
        'MODEL_FOLDER': MODEL_FOLDER,
        'TRAIN_SPLIT_RATIO': TRAIN_SPLIT_RATIO,
        'VALIDATION_SPLIT_RATIO': VALIDATION_SPLIT_RATIO,
        'RANDOM_SEED': RANDOM_SEED,
        'NUM_CLASSES': training_config['num_classes'],
        'CLASS_WEIGHTS': training_config['class_weights'],
        **opt_params,
    }


def get_learning_rate():
    return load_optimized_parameters()['learning_rate']


def get_batch_size():
    from .config import get_training_batch_size
    return get_training_batch_size()


def get_epochs():
    return load_optimized_parameters()['epochs']


def get_loss_weights():
    params = load_optimized_parameters()
    return params['focal_weight'], params['jaccard_weight']


def get_dropout_rates():
    params = load_optimized_parameters()
    return params['dropout_encoder'], params['dropout_decoder']
