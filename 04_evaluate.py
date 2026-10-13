# 04_evaluate.py

import os
import tensorflow as tf
import json

from src.config import (
    IMAGE_BAND_NAMES,
    LABEL_BAND_NAME,
    TRAINING_CONFIG_PATH,
    MODEL_FOLDER,
    DATA_SPLIT_PATH,
    get_training_batch_size,
    load_training_config,
)
from src.data_loader import create_dataset
from src.model import SparseMeanIoU, SparseMetricWrapper
from src.combined_loss_system import CombinedFocalJaccardLoss
from src.optimized_hyperparameters import load_optimized_parameters
from src.per_class_metrics import calculate_per_class_metrics_simple, print_per_class_metrics

def main():
    print("--- STARTING MODEL EVALUATION ---")

    # --- 1. Load the Dynamic Training Configuration ---
    if not os.path.exists(TRAINING_CONFIG_PATH):
        print(f"ERROR: Training config file not found at '{TRAINING_CONFIG_PATH}'")
        return
        
    training_config = load_training_config()
    NUM_CLASSES = training_config['num_classes']
    batch_size = get_training_batch_size()
    print(f"Loaded NUM_CLASSES = {NUM_CLASSES} from training config.")
    print(f"Batch size = {batch_size} (from training_config.json)")
    
    # --- 2. Load the Test Data File Paths ---
    if not os.path.exists(DATA_SPLIT_PATH):
        raise FileNotFoundError(f"Data split file not found at '{DATA_SPLIT_PATH}'. Run '01_prepare_environment.py' first.")
    
    with open(DATA_SPLIT_PATH, 'r') as f:
        test_files = json.load(f)['test']


    # --- 3. Create the TensorFlow Dataset Object ---
    test_dataset = create_dataset(test_files, batch_size, IMAGE_BAND_NAMES, LABEL_BAND_NAME)

    # --- 4. Auto-detect Best Available Model ---
    model_candidates = [
        os.path.join(MODEL_FOLDER, "unet_optimized_combined_loss.h5"),
        os.path.join(MODEL_FOLDER, "unet_finetuned_gold.h5"),
        os.path.join(MODEL_FOLDER, "unet_best_model_jaccard.h5"),
        os.path.join(MODEL_FOLDER, "unet_best_model.h5")
    ]

    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            print(f"Found model: {os.path.basename(candidate)}")
            break

    if model_path is None:
        print("ERROR: No trained model found. Available models:")
        for candidate in model_candidates:
            print(f"  - {candidate}")
        print("Please train a model first using: python 02_train_optimized_combined_loss.py")
        return

    # --- 5. Load the Trained Model ---
    custom_objects = {
        "CombinedFocalJaccardLoss": CombinedFocalJaccardLoss,
        "SparseMeanIoU": SparseMeanIoU,
        "SparseMetricWrapper": SparseMetricWrapper
    }

    print(f"Loading model from: {model_path}")
    try:
        trained_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"WARNING: Could not load with CombinedFocalJaccardLoss: {e}")
        print("Trying with basic custom objects...")
        basic_custom_objects = {
            "SparseMeanIoU": SparseMeanIoU,
            "SparseMetricWrapper": SparseMetricWrapper
        }
        trained_model = tf.keras.models.load_model(model_path, custom_objects=basic_custom_objects, compile=False)
        print("Model loaded with basic custom objects.")

    # Compile the model for evaluation with same loss function as training
    print("Compiling model for evaluation...")
    
    # Load optimized parameters for consistent loss function
    optimal_params = load_optimized_parameters()
    
    # Use same loss function as training
    loss_function = CombinedFocalJaccardLoss(
        class_weights=training_config['class_weights'],
        alpha=1.0,
        gamma=optimal_params['gamma'],
        focal_weight=optimal_params['focal_weight'],
        jaccard_weight=optimal_params['jaccard_weight'],
        dice_weight=optimal_params.get('dice_weight', 0.3),
        smooth=1e-5
    )
    
    # Use same optimizer as training
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimal_params['learning_rate'])
    
    trained_model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy', SparseMeanIoU(num_classes=NUM_CLASSES)]
    )
    print("Model compiled successfully.")

    # --- 6. Evaluate the Model ---
    # This call will now work because 'test_dataset' is a defined tf.data.Dataset object.
    print("\nEvaluating model performance on the test dataset...")
    results = trained_model.evaluate(test_dataset)
    
    # Create a dictionary of the results for nice printing
    metric_names = [name.replace('_', '').title() for name in trained_model.metrics_names]
    results_dict = dict(zip(metric_names, results))
    
    print("\n--- FINAL TEST SET PERFORMANCE ---")
    for name, value in results_dict.items():
        print(f"{name:<15}: {value:.4f}")
    print("------------------------------------")

    # --- 7. Per-class Metrics (Precision/Recall/F1/IoU) ---
    print("\nCalculating per-class metrics on test dataset...")
    per_class_results = calculate_per_class_metrics_simple(trained_model, test_dataset, NUM_CLASSES)
    print_per_class_metrics(per_class_results, training_config.get('class_names'))

if __name__ == "__main__":
    main()