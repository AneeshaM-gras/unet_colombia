# src/transfer_learning_config.py

"""
Transfer Learning Configuration
Configuration for transfer learning and fine-tuning workflow
"""

import json
import os

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Transfer Learning Paths
TRANSFER_LEARNING_CONFIG = {
    # Model paths
    'base_model_path': os.path.join(BASE_DIR, "models", "saved_models", "unet_base_model.h5"),
    'finetuned_model_path': os.path.join(BASE_DIR, "models", "saved_models", "unet_finetuned_gold.h5"),
    'combined_model_path': os.path.join(BASE_DIR, "models", "saved_models", "unet_combined_data.h5"),
    
    # Current data paths (for base model training)
    'current_data_path': os.path.join(BASE_DIR, "data", "input_patches_training"),
    
    # Data gold workflow paths
    'gold_raw_data_path': os.path.join(BASE_DIR, "data_gold", "raw"),
    'gold_merged_tiles_path': os.path.join(BASE_DIR, "data_gold", "merged_tiles"),
    'gold_patches_path': os.path.join(BASE_DIR, "data_gold", "input_patches_training"),
    # 'gold_synthetic_patches_path' removed - synthetic data generation no longer used
    
    # Legacy paths (for backward compatibility)
    'gold_data_path': os.path.join(BASE_DIR, "data_gold", "input_patches_training"),
    # 'synthetic_data_path' removed - synthetic data generation no longer used
    
    # Output paths
    'transfer_learning_report': os.path.join(BASE_DIR, "models", "saved_models", "transfer_learning_report.json"),
    'finetuning_history': os.path.join(BASE_DIR, "models", "saved_models", "finetuning_history.json"),
    'data_preparation_log': os.path.join(BASE_DIR, "models", "saved_models", "data_preparation_log.json"),
}

# Transfer Learning Parameters
TRANSFER_LEARNING_PARAMS = {
    # Base model training
    'base_training': {
        'epochs': 8,  # Same as current configuration
        'learning_rate': 0.001164,  # Same as current configuration
        'batch_size': 16,
        'patience': 15,
        'reduce_lr_patience': 8,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-7,
    },
    
    # Fine-tuning parameters
    'finetuning': {
        'epochs': 4,  # Half of base training
        'learning_rate': 0.0001164,  # 10% of base learning rate
        'batch_size': 16,
        'patience': 10,  # Shorter patience
        'reduce_lr_patience': 5,  # Shorter patience
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-8,
    },
    
    # Combined training parameters
    'combined_training': {
        'epochs': 6,  # Between base and fine-tuning
        'learning_rate': 0.000582,  # 50% of base learning rate
        'batch_size': 16,
        'patience': 12,
        'reduce_lr_patience': 6,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-7,
    }
}

# Data Preparation Workflow
DATA_PREPARATION_WORKFLOW = {
    'step_1': {
        'description': 'Prepare raw data from data_gold/raw/',
        'script': '01_prepare_data.py',
        'input': 'data_gold/raw/',
        'output': 'data_gold/input_patches_training/',
        'command': 'python 01_prepare_data.py --input_dir data_gold/raw --output_dir data_gold/input_patches_training'
    },
    
    'step_2': {
        'description': 'Transfer learning with prepared data',
        'script': '07_transfer_learning_finetune.py',
        'input': 'data_gold/input_patches_training/',
        'output': 'models/saved_models/unet_finetuned_gold.h5',
        'command': 'python 07_transfer_learning_finetune.py'
    }
}

# Data Mixing Strategies
DATA_MIXING_STRATEGIES = {
    'sequential': {
        'description': 'Train on current data, then fine-tune on gold data',
        'phases': ['base_training', 'finetuning'],
        'advantage': 'Clear separation, easy to understand',
        'disadvantage': 'May forget base knowledge'
    },
    
    'combined': {
        'description': 'Mix current and gold data for training',
        'phases': ['combined_training'],
        'advantage': 'Learns from both datasets simultaneously',
        'disadvantage': 'May be confused by different distributions'
    },
    
    'progressive': {
        'description': 'Gradually increase gold data proportion',
        'phases': ['base_training', 'progressive_finetuning'],
        'advantage': 'Smooth transition between datasets',
        'disadvantage': 'More complex implementation'
    }
}

# Evaluation Metrics
EVALUATION_METRICS = {
    'primary': ['mean_iou', 'overall_accuracy'],
    'per_class': ['per_class_iou', 'per_class_accuracy', 'per_class_precision', 'per_class_recall'],
    'loss': ['training_loss', 'validation_loss'],
    'additional': ['f1_score', 'dice_coefficient']
}

# Transfer Learning Strategies
TRANSFER_STRATEGIES = {
    'feature_extraction': {
        'description': 'Freeze encoder, train only decoder',
        'trainable_layers': 'decoder_only',
        'learning_rate': 0.001,  # Higher LR for feature extraction
        'use_case': 'When base features are good'
    },
    
    'fine_tuning': {
        'description': 'Train entire model with lower learning rate',
        'trainable_layers': 'all',
        'learning_rate': 0.0001,  # Lower LR for fine-tuning
        'use_case': 'When need to adapt to new domain'
    },
    
    'progressive_unfreezing': {
        'description': 'Gradually unfreeze layers from top to bottom',
        'trainable_layers': 'progressive',
        'learning_rate': 0.0005,  # Medium LR
        'use_case': 'When need careful adaptation'
    }
}

def get_transfer_learning_config(strategy='sequential'):
    """
    Get transfer learning configuration for a specific strategy
    
    Args:
        strategy (str): Transfer learning strategy ('sequential', 'combined', 'progressive')
    
    Returns:
        dict: Configuration for the specified strategy
    """
    if strategy not in DATA_MIXING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(DATA_MIXING_STRATEGIES.keys())}")
    
    config = {
        'strategy': strategy,
        'strategy_info': DATA_MIXING_STRATEGIES[strategy],
        'paths': TRANSFER_LEARNING_CONFIG,
        'parameters': TRANSFER_LEARNING_PARAMS,
        'evaluation': EVALUATION_METRICS,
        'transfer_strategies': TRANSFER_STRATEGIES
    }
    
    return config

def get_data_paths():
    """Get all data paths for transfer learning"""
    return TRANSFER_LEARNING_CONFIG

def get_training_params(phase):
    """Get training parameters for a specific phase"""
    if phase not in TRANSFER_LEARNING_PARAMS:
        raise ValueError(f"Unknown phase: {phase}. Available: {list(TRANSFER_LEARNING_PARAMS.keys())}")
    
    return TRANSFER_LEARNING_PARAMS[phase]

def get_model_paths():
    """Get model paths for transfer learning"""
    return {
        'base_model': TRANSFER_LEARNING_CONFIG['base_model_path'],
        'finetuned_model': TRANSFER_LEARNING_CONFIG['finetuned_model_path'],
        'combined_model': TRANSFER_LEARNING_CONFIG['combined_model_path']
    }


def _deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


FINETUNING_DEFAULTS = {
    'epochs': 50,
    'batch_size': 8,
    'learning_rate': 1e-5,
    'shuffle_buffer_size': 500,
    'class_weights': [1.0, 1.0, 1.0, 1.0],
    'loss': {
        'alpha': 1.0,
        'smooth': 1e-5,
        'use_base_optimal_weights': True,
    },
    'early_stopping': {
        'monitor': 'val_mean_iou',
        'mode': 'max',
        'patience': 10,
        'restore_best_weights': True,
        # Gold validation is tiny (4 patches); ignore the noisy first epochs so a
        # near-base checkpoint cannot win the monitor and undo fine-tuning.
        'start_from_epoch': 5,
    },
    'model_checkpoint': {
        'monitor': 'val_mean_iou',
        'mode': 'max',
        'save_best_only': True,
    },
    'reduce_lr_on_plateau': {
        'monitor': 'val_loss',
        'factor': 0.2,
        'patience': 5,
        'min_lr': 1e-7,
    },
    'base_model_filename': 'unet_optimized_combined_loss.h5',
    'finetuned_model_filename': 'unet_finetuned_gold.h5',
}


def load_finetuning_parameters():
    """
    Load fine-tuning hyperparameters from finetuning_config.json,
    falling back to FINETUNING_DEFAULTS.
    """
    from src.config import FINETUNING_CONFIG_PATH, MODEL_FOLDER

    params = FINETUNING_DEFAULTS.copy()
    params['loss'] = FINETUNING_DEFAULTS['loss'].copy()
    params['early_stopping'] = FINETUNING_DEFAULTS['early_stopping'].copy()
    params['model_checkpoint'] = FINETUNING_DEFAULTS['model_checkpoint'].copy()
    params['reduce_lr_on_plateau'] = FINETUNING_DEFAULTS['reduce_lr_on_plateau'].copy()

    if os.path.exists(FINETUNING_CONFIG_PATH):
        with open(FINETUNING_CONFIG_PATH, 'r') as f:
            file_params = json.load(f)
        params = _deep_merge(params, file_params)
        print(f"Loaded fine-tuning parameters from: {FINETUNING_CONFIG_PATH}")
    else:
        print(f"Using default fine-tuning parameters (no file at {FINETUNING_CONFIG_PATH})")

    if 'batch_size' not in params:
        raise KeyError(f"'batch_size' missing from {FINETUNING_CONFIG_PATH}")

    params['base_model_path'] = os.path.join(MODEL_FOLDER, params['base_model_filename'])
    params['finetuned_model_path'] = os.path.join(MODEL_FOLDER, params['finetuned_model_filename'])
    return params


if __name__ == "__main__":
    # Test configuration
    print("=== TRANSFER LEARNING CONFIGURATION TEST ===")
    
    # Test strategy configuration
    for strategy in DATA_MIXING_STRATEGIES.keys():
        config = get_transfer_learning_config(strategy)
        print(f"\nStrategy: {strategy}")
        print(f"Description: {config['strategy_info']['description']}")
        print(f"Phases: {config['strategy_info']['phases']}")
    
    # Test data paths
    print(f"\nData Paths:")
    paths = get_data_paths()
    for key, path in paths.items():
        print(f"  {key}: {path}")
    
    # Test training parameters
    print(f"\nTraining Parameters:")
    for phase, params in TRANSFER_LEARNING_PARAMS.items():
        print(f"  {phase}: {params['epochs']} epochs, LR={params['learning_rate']}")
    
    print("\n✓ Transfer learning configuration loaded successfully")
