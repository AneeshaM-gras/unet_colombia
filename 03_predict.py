# 03_predict.py

import os
import tensorflow as tf
import json

# --- IMPORTS FROM YOUR PROJECT'S SOURCE CODE ---
from src.config import (
    MERGED_TILES_FOLDER, # Input for this script
    PREDICTION_FOLDER,   # Output for this script
    MODEL_FOLDER,        # Model directory for auto-detection
    IMAGE_BAND_NAMES,
    TRAINING_CONFIG_PATH
)
from src.model import SparseMeanIoU, SparseMetricWrapper
from src.combined_loss_system import CombinedFocalJaccardLoss
from src.optimized_hyperparameters import load_optimized_parameters
from src.utils import run_batch_prediction

def main():
    print("--- PREDICTION WORKFLOW ON TRAINING SOURCE DATA ---")
    print("Using optimized model with Combined Focal + Jaccard Loss")

    # --- 1. Load Dynamic Config (Needed for NUM_CLASSES) ---
    if not os.path.exists(TRAINING_CONFIG_PATH):
        print(f"ERROR: Training config file not found at '{TRAINING_CONFIG_PATH}'")
        return

    with open(TRAINING_CONFIG_PATH, 'r') as f:
        training_config = json.load(f)
    NUM_CLASSES = training_config['num_classes']
    CLASS_WEIGHTS = training_config['class_weights']
    print(f"Loaded NUM_CLASSES = {NUM_CLASSES} from training config.")

    # --- 2. Auto-detect Best Available Model ---
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

    # --- 3. Define Custom Objects and Load Model ---
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

    # --- 4. Load Optimized Parameters and Compile Model ---
    print("Loading optimized parameters for consistent model compilation...")
    try:
        optimal_params = load_optimized_parameters()
        
        # Use same loss function as training for consistency
        loss_function = CombinedFocalJaccardLoss(
            class_weights=CLASS_WEIGHTS,
            alpha=1.0,
            gamma=optimal_params['gamma'],
            focal_weight=optimal_params['focal_weight'],
            jaccard_weight=optimal_params['jaccard_weight'],
            dice_weight=optimal_params.get('dice_weight', 0.3),
            smooth=1e-5
        )
        
        # Use same optimizer as training
        optimizer = tf.keras.optimizers.Adam(learning_rate=optimal_params['learning_rate'])
        
        # Compile model with same configuration as training
        trained_model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=['accuracy', SparseMeanIoU(num_classes=NUM_CLASSES)]
        )
        print("Model compiled with optimized parameters successfully.")
        
    except Exception as e:
        print(f"WARNING: Could not load optimized parameters: {e}")
        print("Model will be used without compilation.")

    # --- 5. Run Batch Prediction ---
    print(f"Running prediction on: {MERGED_TILES_FOLDER}")
    print(f"Output will be saved to: {PREDICTION_FOLDER}")

    run_batch_prediction(
        model=trained_model,
        input_folder=MERGED_TILES_FOLDER,
        output_folder=PREDICTION_FOLDER,
        image_band_names=IMAGE_BAND_NAMES
    )
    print("--- PREDICTION WORKFLOW COMPLETE ---")

if __name__ == "__main__":
    main()