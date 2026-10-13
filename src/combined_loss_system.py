# src/combined_loss_system.py

"""
Combined Loss System: Loss Function + Optimization
This file contains both the CombinedFocalJaccardLoss implementation
and the CombinedLossOptimizer for hyperparameter optimization.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from src.config import (
    FINAL_PATCHES_FOLDER, IMAGE_BAND_NAMES, LABEL_BAND_NAME,
    GOLD_LABEL_BAND_NAME, GOLD_DATA_SPLIT_PATH, FINETUNING_CONFIG_PATH,
    NUM_IMAGE_BANDS, PATCH_SIZE, TRAINING_CONFIG_PATH, MODEL_FOLDER,
    TRAIN_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, RANDOM_SEED,
    HPO_SUBSET_SIZE, HPO_SUBSET_FRACTION, HPO_SHUFFLE_ALL, HPO_RANDOM_SAMPLE
)
from src.data_loader import create_dataset
from src.optimized_model import unet
from src.optimized_hyperparameters import load_optimized_parameters
from src.per_class_metrics import calculate_per_class_metrics_simple, macro_average
from src.model import SparseMeanIoU, SparseMetricWrapper, SparseF1Score
from src.transfer_learning_config import load_finetuning_parameters

# =============================================================================
# COMBINED FOCAL + JACCARD LOSS FUNCTION
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class CombinedFocalJaccardLoss(tf.keras.losses.Loss):
    """
    Combined Focal + Jaccard + Dice Loss
    Combines the benefits of all three loss functions:
    - Focal Loss: Handles class imbalance and focuses on hard examples
    - Jaccard Loss: Directly optimizes Intersection over Union
    - Dice Loss: Optimizes Dice coefficient (similar to IoU but with different formulation)
    """
    
    def __init__(self, class_weights=None, alpha=1.0, gamma=2.0, 
                 focal_weight=0.33, jaccard_weight=0.33, dice_weight=0.34, smooth=1e-5, 
                 name="combined_focal_jaccard_dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        # Ensure class_weights is a float32 tensor if provided
        if class_weights is not None:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        else:
            self.class_weights = None
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.jaccard_weight = jaccard_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
    
    def focal_loss(self, y_true, y_pred):
        """Calculate Focal Loss component"""
        y_true = tf.cast(y_true, tf.int32)
        # Squeeze the trailing dimension if it exists (shape becomes [batch, H, W])
        y_true = tf.squeeze(y_true, axis=-1)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        ce = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred, from_logits=False)
        
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = self.alpha * tf.pow(1.0 - p_t, self.gamma)
        focal_loss_per_pixel = focal_weight * ce
        
        if self.class_weights is not None:
            class_indices = tf.cast(y_true, dtype=tf.int32)
            # Clip class indices to valid range [0, num_classes) to handle any invalid values
            num_classes = tf.shape(y_pred)[-1]
            class_indices = tf.clip_by_value(class_indices, 0, num_classes - 1)
            weights_for_pixels = tf.gather(self.class_weights, class_indices)
            focal_loss_per_pixel *= weights_for_pixels
        
        return tf.reduce_mean(focal_loss_per_pixel)
    
    def jaccard_loss(self, y_true, y_pred):
        """Calculate Jaccard Loss component"""
        y_true = tf.cast(y_true, tf.int32)
        # Squeeze the trailing dimension if it exists (shape becomes [batch, H, W])
        y_true = tf.squeeze(y_true, axis=-1)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        # Flatten for intersection/union calculation
        y_true_flat = tf.reshape(y_true_one_hot, [-1, tf.shape(y_pred)[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0) - intersection
        
        jaccard_coeff = (intersection + self.smooth) / (union + self.smooth)
        
        if self.class_weights is not None:
            # Weight the IoU of each class before averaging
            jaccard_coeff *= self.class_weights
        
        return 1.0 - tf.reduce_mean(jaccard_coeff)
    
    def dice_loss(self, y_true, y_pred):
        """Calculate Dice Loss component"""
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true, axis=-1)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        # Flatten for calculation
        y_true_flat = tf.reshape(y_true_one_hot, [-1, tf.shape(y_pred)[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        sum_pred = tf.reduce_sum(y_pred_flat, axis=0)
        sum_true = tf.reduce_sum(y_true_flat, axis=0)
        
        dice_coeff = (2.0 * intersection + self.smooth) / (sum_pred + sum_true + self.smooth)
        
        if self.class_weights is not None:
            dice_coeff *= self.class_weights
        
        return 1.0 - tf.reduce_mean(dice_coeff)
    
    def call(self, y_true, y_pred):
        """
        Calculates the combined loss.
        The hard-coded penalties for specific classes have been removed for robustness.
        Class imbalance should be handled by the 'class_weights' parameter.
        """
        focal_loss_val = self.focal_loss(y_true, y_pred)
        jaccard_loss_val = self.jaccard_loss(y_true, y_pred)
        dice_loss_val = self.dice_loss(y_true, y_pred)
        
        # Combine losses using their respective weights
        combined_loss = (self.focal_weight * focal_loss_val + 
                         self.jaccard_weight * jaccard_loss_val +
                         self.dice_weight * dice_loss_val)
        
        return combined_loss
    
    def get_config(self):
        """Get configuration for serialization"""
        config = super().get_config()
        config.update({
            'class_weights': self.class_weights.numpy().tolist() if self.class_weights is not None else None,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'focal_weight': self.focal_weight,
            'jaccard_weight': self.jaccard_weight,
            'dice_weight': self.dice_weight,
            'smooth': self.smooth
        })
        return config

# =============================================================================
# COMBINED LOSS OPTIMIZER
# =============================================================================

class CombinedLossOptimizer:
    def __init__(self, hpo_data_source='gold'):
        """
        Args:
            hpo_data_source: 'gold' — search base params on gold train/val, then use
                for weak-label training via combined_loss_optimization_results.json.
                'weak' — legacy search on weak-label subset.
        """
        if hpo_data_source not in ('gold', 'weak'):
            raise ValueError("hpo_data_source must be 'gold' or 'weak'")
        self.hpo_data_source = hpo_data_source
        self.best_score = 0.0
        self.best_params = None
        self.results = []
        self._gold_train_files = None
        self._gold_val_files = None

        # Load training config for num_classes / class_weights
        if os.path.exists(TRAINING_CONFIG_PATH):
            with open(TRAINING_CONFIG_PATH, 'r') as f:
                training_config = json.load(f)
            self.num_classes = training_config.get('num_classes', 4)
            self.class_weights = training_config.get('class_weights')
        else:
            self.num_classes = 4
            self.class_weights = None

    def load_enhanced_dataset(self):
        """Load weak-label patch paths for legacy HPO."""
        print("Loading weak-label dataset...")

        all_files = [
            os.path.join(FINAL_PATCHES_FOLDER, f)
            for f in os.listdir(FINAL_PATCHES_FOLDER)
            if f.lower().endswith(('.tif', '.tiff'))
        ]
        print(f"Using original dataset: {len(all_files)} patches")
        return all_files

    def create_optimization_datasets(self, all_files, batch_size, subset_size=None):
        """Create smaller weak-label datasets for optimization.

        Args:
            all_files: List of all patch files
            batch_size: Batch size for training (from HPO trial)
            subset_size: Number of files to use for optimization (defaults to HPO_SUBSET_SIZE)
        """
        if subset_size is None:
            if HPO_SUBSET_FRACTION is not None:
                subset_size = int(len(all_files) * HPO_SUBSET_FRACTION)
            else:
                subset_size = HPO_SUBSET_SIZE

        if HPO_SUBSET_FRACTION is not None and (subset_size is None or subset_size <= 0):
            subset_size = max(1, int(len(all_files) * HPO_SUBSET_FRACTION))

        if subset_size is None:
            subset_files = list(all_files)
        else:
            subset_size = min(subset_size, len(all_files))
            if HPO_RANDOM_SAMPLE:
                random.seed(RANDOM_SEED)
                subset_files = random.sample(all_files, k=subset_size)
            else:
                if HPO_SHUFFLE_ALL:
                    random.seed(RANDOM_SEED)
                    random.shuffle(all_files)
                subset_files = all_files[:subset_size]

        random.seed(RANDOM_SEED)
        random.shuffle(subset_files)

        train_split = int(TRAIN_SPLIT_RATIO * len(subset_files))
        val_split = int(VALIDATION_SPLIT_RATIO * len(subset_files))

        train_files = subset_files[:train_split]
        val_files = subset_files[train_split:train_split + val_split]

        print(f"Optimization datasets: {len(train_files)} train, {len(val_files)} val")

        train_dataset = create_dataset(train_files, batch_size, IMAGE_BAND_NAMES, LABEL_BAND_NAME)
        val_dataset = create_dataset(val_files, batch_size, IMAGE_BAND_NAMES, LABEL_BAND_NAME)

        return train_dataset, val_dataset

    def create_gold_base_hpo_datasets(self, batch_size):
        """Gold train/val for base-parameter HPO (gold test never used)."""
        if self._gold_train_files is None or self._gold_val_files is None:
            self._gold_train_files, self._gold_val_files = self.load_gold_splits()
        return self.create_finetune_datasets(
            self._gold_train_files,
            self._gold_val_files,
            batch_size,
            shuffle_buffer_size=min(500, max(1, len(self._gold_train_files))),
        )

    def evaluate_parameters(self, params, epochs=10):
        """Evaluate a set of parameters (train from scratch on gold or weak).

        Args:
            params: Dictionary of hyperparameters to test
            epochs: Number of epochs to train (default: 10)
        """
        try:
            tf.keras.backend.clear_session()

            if self.hpo_data_source == 'gold':
                train_dataset, val_dataset = self.create_gold_base_hpo_datasets(
                    int(params['batch_size'])
                )
            else:
                all_files = self.load_enhanced_dataset()
                train_dataset, val_dataset = self.create_optimization_datasets(
                    all_files, params['batch_size'], subset_size=HPO_SUBSET_SIZE
                )

            INPUT_SHAPE = (PATCH_SIZE, PATCH_SIZE, NUM_IMAGE_BANDS)
            model = unet(
                input_shape=INPUT_SHAPE,
                n_classes=self.num_classes,
                dropout_encoder=params['dropout_encoder'],
                dropout_decoder=params['dropout_decoder']
            )

            loss_function = CombinedFocalJaccardLoss(
                class_weights=self.class_weights,
                gamma=params['gamma'],
                focal_weight=params['focal_weight'],
                jaccard_weight=params['jaccard_weight'],
                dice_weight=params['dice_weight']
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=['accuracy']
            )

            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=max(3, epochs // 3),
                        restore_best_weights=True,
                        verbose=0
                    )
                ]
            )

            val_loss = min(history.history['val_loss'])
            val_acc = max(history.history['val_accuracy'])

            results = calculate_per_class_metrics_simple(model, val_dataset, self.num_classes)
            avg_iou = np.mean(results['per_class_iou'])

            composite_score = 0.6 * val_acc + 0.4 * avg_iou

            print(f"[{self.hpo_data_source}] Params: {params}")
            print(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Avg IoU: {avg_iou:.4f}, Score: {composite_score:.4f}"
            )

            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_params = params.copy()
                print(f"New best score: {composite_score:.4f}")

            self.results.append({
                'params': params.copy(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'avg_iou': avg_iou,
                'composite_score': composite_score,
                'hpo_data_source': self.hpo_data_source,
            })

            return -composite_score

        except Exception as e:
            print(f"Error evaluating parameters {params}: {e}")
            return 0.0

    def optimize_hyperparameters(self, n_calls=50, epochs_per_trial=10):
        """Run Bayesian optimization for base (from-scratch) training params.

        When hpo_data_source='gold', trials train on gold train/val. Best params
        are saved for weak-label full training (02_train_optimized_combined_loss.py).
        """
        print("Starting Bayesian optimization for Combined Loss...")
        print(f"HPO data source: {self.hpo_data_source}")
        print(f"Configuration: {n_calls} trials, {epochs_per_trial} epochs per trial")

        if self.hpo_data_source == 'gold':
            self._gold_train_files, self._gold_val_files = self.load_gold_splits()
            print(
                "NOTE: Base HPO is scored on gold validation. Best params will be "
                "applied to weak-label training. Gold set is small — interpret carefully."
            )
            space = [
                Real(1e-5, 1e-2, name='learning_rate', prior='log-uniform'),
                Categorical([2, 4, 8], name='batch_size'),
                Real(0.15, 0.5, name='focal_weight'),
                Real(0.3, 0.6, name='jaccard_weight'),
                Real(0.2, 0.5, name='dice_weight'),
                Real(2.0, 5.0, name='gamma'),
                Real(0.01, 0.5, name='dropout_encoder'),
                Real(0.01, 0.5, name='dropout_decoder'),
            ]
        else:
            space = [
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform'),
                Integer(8, 32, name='batch_size'),
                Real(0.15, 0.5, name='focal_weight'),
                Real(0.3, 0.6, name='jaccard_weight'),
                Real(0.2, 0.5, name='dice_weight'),
                Real(2.0, 5.0, name='gamma'),
                Real(0.01, 0.5, name='dropout_encoder'),
                Real(0.01, 0.5, name='dropout_decoder'),
            ]

        @use_named_args(space)
        def objective(**params):
            total_weight = (
                params['focal_weight'] + params['jaccard_weight'] + params['dice_weight']
            )
            params['focal_weight'] = params['focal_weight'] / total_weight
            params['jaccard_weight'] = params['jaccard_weight'] / total_weight
            params['dice_weight'] = params['dice_weight'] / total_weight

            normalized_jaccard = params['jaccard_weight']
            gamma_scale_factor = 2.5
            gamma_base = params['gamma']
            params['gamma'] = gamma_base + (normalized_jaccard - 0.3) * gamma_scale_factor
            params['gamma'] = max(2.0, min(6.0, params['gamma']))

            return self.evaluate_parameters(params, epochs=epochs_per_trial)

        gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=RANDOM_SEED,
            acq_func='EI'
        )

        print(f"\nOptimization completed!")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")

        return self.best_params, self.results

    def write_training_config_batch_size(self, best_params):
        """Write winning batch_size into training_config.json (runtime source of truth)."""
        if not best_params or 'batch_size' not in best_params:
            return None
        with open(TRAINING_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        config['batch_size'] = int(best_params['batch_size'])
        with open(TRAINING_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
            f.write('\n')
        print(f"Updated training_config.json batch_size -> {config['batch_size']}")
        return TRAINING_CONFIG_PATH

    def save_results(self, best_params, results):
        """Save optimization results for weak-label training to consume."""
        optimization_results = {
            'best_params': best_params,
            'best_score': self.best_score,
            'hpo_data_source': self.hpo_data_source,
            'intended_use': (
                'Apply best_params to weak-label base training '
                '(02_train_optimized_combined_loss.py via load_optimized_parameters).'
                if self.hpo_data_source == 'gold'
                else 'Legacy weak-subset HPO for weak-label base training.'
            ),
            'all_results': results,
            'optimization_summary': {
                'best_params': best_params,
                'best_score': self.best_score,
                'total_trials': len(results),
                'hpo_data_source': self.hpo_data_source,
            }
        }

        os.makedirs(MODEL_FOLDER, exist_ok=True)
        results_path = os.path.join(MODEL_FOLDER, 'combined_loss_optimization_results.json')

        with open(results_path, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)

        if self.hpo_data_source == 'gold':
            gold_copy = os.path.join(
                MODEL_FOLDER, 'combined_loss_optimization_results_from_gold.json'
            )
            with open(gold_copy, 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)
            print(f"Gold HPO copy saved to: {gold_copy}")

        if best_params:
            self.write_training_config_batch_size(best_params)

        print(f"Results saved to: {results_path}")
        return results_path

    # -------------------------------------------------------------------------
    # Fine-tuning HPO (gold labels) — LR + batch_size only (no coffee weight)
    # -------------------------------------------------------------------------

    def load_gold_splits(self):
        """Load fixed gold train/validation file lists. Never loads the test split."""
        if not os.path.exists(GOLD_DATA_SPLIT_PATH):
            raise FileNotFoundError(
                f"Gold data split not found at '{GOLD_DATA_SPLIT_PATH}'. "
                "Run 06_prepare_gold_data.py first."
            )
        with open(GOLD_DATA_SPLIT_PATH, 'r') as f:
            splits = json.load(f)
        train_files = splits['train']
        val_files = splits['validation']
        print(
            f"Gold HPO splits (fixed): {len(train_files)} train, "
            f"{len(val_files)} validation (test excluded)"
        )
        return train_files, val_files

    def create_finetune_datasets(self, train_files, val_files, batch_size, shuffle_buffer_size=500):
        """Create gold train/val datasets from fixed splits."""
        train_dataset = create_dataset(
            train_files, batch_size, IMAGE_BAND_NAMES, GOLD_LABEL_BAND_NAME
        ).shuffle(shuffle_buffer_size)
        val_dataset = create_dataset(
            val_files, batch_size, IMAGE_BAND_NAMES, GOLD_LABEL_BAND_NAME
        )
        return train_dataset, val_dataset

    def _finetune_custom_objects(self):
        return {
            'CombinedFocalJaccardLoss': CombinedFocalJaccardLoss,
            'SparseMeanIoU': SparseMeanIoU,
            'SparseMetricWrapper': SparseMetricWrapper,
            'SparseF1Score': SparseF1Score,
        }

    def evaluate_finetune_parameters(self, params, epochs=20, early_stopping_patience=8):
        """
        One fine-tuning HPO trial:
        reload pretrained base model, fine-tune on fixed gold train,
        early-stop / score on gold validation only (never gold/AnnualTS test).
        """
        trial_idx = len(self.results) + 1
        try:
            tf.keras.backend.clear_session()

            train_dataset, val_dataset = self.create_finetune_datasets(
                self._gold_train_files,
                self._gold_val_files,
                int(params['batch_size']),
                shuffle_buffer_size=self._finetune_shuffle_buffer,
            )

            if not os.path.exists(self._base_model_path):
                raise FileNotFoundError(
                    f"Base model not found at '{self._base_model_path}'. "
                    "Run 02_train_optimized_combined_loss.py first."
                )

            model = tf.keras.models.load_model(
                self._base_model_path,
                custom_objects=self._finetune_custom_objects(),
                compile=False,
            )

            # Class weights are frozen (no coffee-weight search)
            class_weights = list(self._finetune_class_weights)

            loss_function = CombinedFocalJaccardLoss(
                class_weights=class_weights,
                alpha=self._finetune_loss_alpha,
                gamma=self._base_loss_params.get('gamma', 3.5),
                focal_weight=self._base_loss_params.get('focal_weight', 0.3),
                jaccard_weight=self._base_loss_params.get('jaccard_weight', 0.4),
                dice_weight=self._base_loss_params.get('dice_weight', 0.3),
                smooth=self._finetune_loss_smooth,
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=float(params['learning_rate'])
                ),
                loss=loss_function,
                metrics=[
                    'accuracy',
                    SparseMeanIoU(num_classes=self.num_classes, name='mean_iou'),
                    SparseF1Score(num_classes=self.num_classes, name='f1_score'),
                ],
            )

            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_mean_iou',
                        mode='max',
                        patience=early_stopping_patience,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
            )

            val_mean_iou = float(max(history.history['val_mean_iou']))
            per_class = calculate_per_class_metrics_simple(
                model, val_dataset, self.num_classes
            )
            macro = macro_average(per_class)
            macro_f1 = float(macro.get('macro_f1', 0.0))

            trial_record = {
                'trial': trial_idx,
                'params': {
                    'learning_rate': float(params['learning_rate']),
                    'batch_size': int(params['batch_size']),
                    'class_weights': class_weights,
                },
                'val_mean_iou': val_mean_iou,
                'macro_f1': macro_f1,
                'per_class_iou': [float(x) for x in per_class['per_class_iou']],
                'per_class_f1': [float(x) for x in per_class['per_class_f1']],
            }
            self.results.append(trial_record)
            self._persist_finetune_trial_log()

            print(
                f"[Trial {trial_idx}] lr={params['learning_rate']:.6g}, "
                f"batch={int(params['batch_size'])} | "
                f"val_mean_iou={val_mean_iou:.4f}, macro_f1={macro_f1:.4f}"
            )

            if val_mean_iou > self.best_score:
                self.best_score = val_mean_iou
                self.best_params = {
                    'learning_rate': float(params['learning_rate']),
                    'batch_size': int(params['batch_size']),
                    'class_weights': class_weights,
                }
                print(f"  -> New best val_mean_iou: {val_mean_iou:.4f}")

            del model
            tf.keras.backend.clear_session()
            return -val_mean_iou

        except Exception as e:
            print(f"Error evaluating fine-tune parameters {params}: {e}")
            self.results.append({
                'trial': trial_idx,
                'params': {
                    'learning_rate': float(params.get('learning_rate', 0.0)),
                    'batch_size': int(params.get('batch_size', 0)),
                },
                'val_mean_iou': 0.0,
                'macro_f1': 0.0,
                'error': str(e),
            })
            self._persist_finetune_trial_log()
            tf.keras.backend.clear_session()
            return 0.0

    def _persist_finetune_trial_log(self):
        """Write all trials so far to a persistent JSON file."""
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        payload = {
            'best_params': self.best_params,
            'best_val_mean_iou': self.best_score,
            'total_trials': len(self.results),
            'all_trials': self.results,
        }
        with open(self._finetune_results_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)

    def optimize_finetune_hyperparameters(
        self, n_calls=30, epochs_per_trial=20, early_stopping_patience=8
    ):
        """Bayesian HPO for gold fine-tuning (learning_rate + batch_size only)."""
        finetune_params = load_finetuning_parameters()
        self._base_model_path = finetune_params['base_model_path']
        self._finetune_shuffle_buffer = int(finetune_params['shuffle_buffer_size'])
        self._finetune_loss_alpha = float(finetune_params['loss'].get('alpha', 1.0))
        self._finetune_loss_smooth = float(finetune_params['loss'].get('smooth', 1e-5))

        # Prefer class weights from training_config (frozen; not searched)
        if os.path.exists(TRAINING_CONFIG_PATH):
            with open(TRAINING_CONFIG_PATH, 'r') as f:
                training_config = json.load(f)
            self._finetune_class_weights = training_config.get(
                'class_weights', finetune_params['class_weights']
            )
            self.num_classes = training_config.get('num_classes', self.num_classes)
        else:
            self._finetune_class_weights = finetune_params['class_weights']

        self._base_loss_params = load_optimized_parameters()
        self._finetune_results_path = os.path.join(
            MODEL_FOLDER, 'gold_finetune_hpo_results.json'
        )

        # Reset trial state for a fresh search
        self.best_score = 0.0
        self.best_params = None
        self.results = []
        self._gold_train_files, self._gold_val_files = self.load_gold_splits()

        print("Starting Bayesian optimization for gold fine-tuning...")
        print(f"Base model: {self._base_model_path}")
        print(
            f"Configuration: {n_calls} trials, {epochs_per_trial} epochs/trial, "
            f"ES patience={early_stopping_patience}"
        )
        print(
            "Search space: lr log-uniform [1e-5, 1e-3], batch in {2,4,6,8}; "
            "class_weights frozen from training_config; loss mixture frozen from base HPO"
        )
        print(f"Frozen class_weights: {self._finetune_class_weights}")
        print(f"Trial log: {self._finetune_results_path}")

        space = [
            Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
            Categorical([2, 4, 6, 8], name='batch_size'),
        ]

        @use_named_args(space)
        def objective(**params):
            return self.evaluate_finetune_parameters(
                params,
                epochs=epochs_per_trial,
                early_stopping_patience=early_stopping_patience,
            )

        gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=RANDOM_SEED,
            acq_func='EI',
        )

        print(f"\nFine-tuning HPO completed!")
        print(f"Best val_mean_iou: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        self._persist_finetune_trial_log()
        return self.best_params, self.results

    def write_finetuning_config(self, best_params):
        """
        Update learning_rate and batch_size in finetuning_config.json.
        Leaves class_weights and other fields intact (no coffee-weight search).
        """
        if best_params is None:
            raise ValueError("No best parameters to write; HPO found no successful trial.")

        if os.path.exists(FINETUNING_CONFIG_PATH):
            with open(FINETUNING_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        config['learning_rate'] = float(best_params['learning_rate'])
        config['batch_size'] = int(best_params['batch_size'])
        # Keep class_weights frozen unless already present
        if 'class_weights' not in config and 'class_weights' in best_params:
            config['class_weights'] = [float(w) for w in best_params['class_weights']]

        with open(FINETUNING_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
            f.write('\n')

        print(f"Updated finetuning config: {FINETUNING_CONFIG_PATH}")
        print(f"  learning_rate: {config['learning_rate']}")
        print(f"  batch_size: {config['batch_size']}")
        if 'class_weights' in config:
            print(f"  class_weights (frozen): {config['class_weights']}")
        return FINETUNING_CONFIG_PATH

# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

def run_finetuning_optimization(
    n_calls=30, epochs_per_trial=20, early_stopping_patience=8, update_config=True
):
    """
    Run Bayesian HPO for the gold fine-tuning stage.

    Searches learning_rate and batch_size on the fixed gold validation set.
    Class weights are frozen from training_config.json (no coffee-weight search).
    Does not run final fine-tuning or AnnualTS test evaluation.
    """
    print("=" * 60)
    print("GOLD FINE-TUNING HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Trials: {n_calls}")
    print(f"Epochs per trial: {epochs_per_trial}")
    print(f"Early-stopping patience: {early_stopping_patience}")
    print("=" * 60)

    optimizer = CombinedLossOptimizer()
    best_params, results = optimizer.optimize_finetune_hyperparameters(
        n_calls=n_calls,
        epochs_per_trial=epochs_per_trial,
        early_stopping_patience=early_stopping_patience,
    )

    results_path = optimizer._finetune_results_path
    config_path = None
    if update_config and best_params is not None:
        config_path = optimizer.write_finetuning_config(best_params)

    print("\n" + "=" * 60)
    print("FINE-TUNING HPO COMPLETE")
    print("=" * 60)
    if best_params:
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest val_mean_iou: {optimizer.best_score:.4f}")
    print(f"Full trial log: {results_path}")
    if config_path:
        print(f"Updated: {config_path}")
    print("\nNext step (manual):")
    print("  python 07_transfer_learning_finetune.py")

    return best_params, results


def run_combined_loss_optimization(n_calls=50, epochs_per_trial=10, data_source='gold'):
    """
    Run base hyperparameter optimization, then use best params for weak training.

    Args:
        n_calls: Number of trials to run (default: 50)
        epochs_per_trial: Epochs per trial (default: 10)
        data_source: 'gold' (default) — HPO on gold train/val, params applied to
            weak-label training via combined_loss_optimization_results.json.
            'weak' — legacy HPO on a weak-label subset.
    """
    print("=" * 60)
    print("COMBINED LOSS OPTIMIZATION (BASE PARAMS)")
    print("=" * 60)
    print(f"Data source for HPO: {data_source}")
    print(f"Trials: {n_calls}")
    print(f"Epochs per trial: {epochs_per_trial}")
    if data_source == 'gold':
        print("Workflow: HPO on gold → save params → train weak with 02_train_...")
    print("=" * 60)

    optimizer = CombinedLossOptimizer(hpo_data_source=data_source)
    best_params, results = optimizer.optimize_hyperparameters(
        n_calls=n_calls, epochs_per_trial=epochs_per_trial
    )
    results_path = optimizer.save_results(best_params, results)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best parameters found (hpo_data_source={data_source}):")
    if best_params:
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    print(f"\nBest composite score: {optimizer.best_score:.4f}")
    print(f"Results saved to: {results_path}")
    print("\nNext step:")
    print("  python 02_train_optimized_combined_loss.py")
    print("  (loads these params for weak-label base training)")

    return best_params, results


if __name__ == "__main__":
    run_combined_loss_optimization(
        n_calls=50, epochs_per_trial=10, data_source='gold'
    )
