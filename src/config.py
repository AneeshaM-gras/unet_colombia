# src/config.py

import os

# --- 1. File Path Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Raw data input directory
ROOT_DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")

# Processed data directories
MERGED_TILES_FOLDER = os.path.join(BASE_DIR, "data", "merged_tiles")
FINAL_PATCHES_FOLDER = os.path.join(BASE_DIR, "data", "input_patches_training")

PREDICTION_FOLDER = os.path.join(BASE_DIR, "data", "predictions_training")

# --- INFERENCE PATHS (New Section) ---
# --- INFERENCE PATHS ---
INFERENCE_DATA_FOLDER = os.path.join(BASE_DIR, "inference_data")  # Raw inference data with subfolders (at project root)
INFERENCE_TILES_FOLDER = os.path.join(BASE_DIR, "data", "input_inference_tiles")  # Legacy: kept for backward compatibility
INFERENCE_PREDICTIONS_FOLDER = os.path.join(BASE_DIR, "data", "predictions_inference")
# --- NEW: A temporary workspace for all intermediate inference files ---
TEMP_INFERENCE_WORKSPACE = os.path.join(BASE_DIR, "data", "temp_inference_workspace")

# Add the path for the data split JSON file
DATA_SPLIT_PATH = os.path.join(BASE_DIR, "data_split.json")

# --- 2. Data and Model Configuration ---
#IMAGE_BAND_NAMES = [f'embedding_{i:02d}' for i in range(64)]
IMAGE_BAND_NAMES = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60', 'A61', 'A62', 'A63']
#LABEL_BAND_NAME = '0_label'
LABEL_BAND_NAME = 'weaklabel' #'classification_2023'
GOLD_LABEL_BAND_NAME = 'label'
NUM_IMAGE_BANDS = len(IMAGE_BAND_NAMES)

# --- NEW: CLASS AGGREGATION AND REMAPPING ---
# Define the mapping from ORIGINAL class IDs in your files to the FINAL class IDs for the model.
# Set to None or {} to disable remapping.
CLASS_MAPPING = {
    0: 0,  # Background -> Background
    1: 1,  # Original Class 1 -> Aggregated Class 1
    2: 2,  # Original Class 2 -> Final Class 2
    3: 2 #Original Class 3 -> Final class 3

}
# If you don't want to aggregate, set this to None:
# CLASS_MAPPING = None

# Optional: Provide names for the FINAL classes after mapping.
# The order should correspond to the new class IDs (0, 1, 2, ...).
FINAL_CLASS_NAMES = [
    'Background',      # Final Class 0
    'Forest',          # Final Class 1
    'Unshaded Coffee',           # Final Class 2 (originally class 2)
    'Shaded Coffee'           # Final Class 3 (originally class 2)
]

# --- 3. Training Hyperparameters ---
PATCH_SIZE = 256
TRAIN_SPLIT_RATIO = 0.7
VALIDATION_SPLIT_RATIO = 0.15
# Fixed gold validation size for fine-tuning splits (overrides ratio when set)
GOLD_VALIDATION_SIZE = 4
RANDOM_SEED = 42

# --- Hyperparameter Optimization (HPO) Dataset Sampling ---
# Set HPO_SUBSET_FRACTION to a value in (0, 1] to sample a percentage.
# Set HPO_SUBSET_FRACTION to None to fall back to HPO_SUBSET_SIZE.
HPO_SUBSET_FRACTION = None
# Set HPO_SUBSET_SIZE to None to use all patches for HPO (slow).
HPO_SUBSET_SIZE = 50
# If True, shuffle the full file list before slicing.
HPO_SHUFFLE_ALL = True
# If True, use random.sample across the full dataset.
HPO_RANDOM_SAMPLE = False

# --- 4. Config paths (single source of truth for batch sizes) ---
MODEL_FOLDER = os.path.join(BASE_DIR, "models", "saved_models")
TRAINED_MODEL_PATH = os.path.join(MODEL_FOLDER, "unet_best_model.h5")
TRAINING_CONFIG_PATH = os.path.join(BASE_DIR, "training_config.json")
FINETUNING_CONFIG_PATH = os.path.join(BASE_DIR, "finetuning_config.json")


def load_training_config():
    """Load model configuration (num_classes, batch_size, class_weights, loss)."""
    import json
    with open(TRAINING_CONFIG_PATH, 'r') as f:
        return json.load(f)


def get_training_batch_size():
    """Batch size for base/weak training and related eval from training_config.json."""
    config = load_training_config()
    if 'batch_size' not in config:
        raise KeyError(f"'batch_size' missing from {TRAINING_CONFIG_PATH}")
    return int(config['batch_size'])


def get_finetuning_batch_size():
    """Batch size for fine-tuning and gold test/inference from finetuning_config.json."""
    from .transfer_learning_config import load_finetuning_parameters
    params = load_finetuning_parameters()
    if 'batch_size' not in params:
        raise KeyError(f"'batch_size' missing from {FINETUNING_CONFIG_PATH}")
    return int(params['batch_size'])


def get_optimized_batch_size():
    """Batch size from training_config.json (base training source of truth)."""
    return get_training_batch_size()


def get_optimized_epochs():
    try:
        from .optimized_hyperparameters import load_optimized_parameters
        return load_optimized_parameters()['epochs']
    except Exception:
        return load_training_config().get('epochs', 60)


def get_optimized_learning_rate():
    try:
        from .optimized_hyperparameters import load_optimized_parameters
        return load_optimized_parameters()['learning_rate']
    except Exception:
        return float(load_training_config().get('learning_rate', 0.001164))


# Backward-compatible aliases: batch from training_config; LR/epochs from HPO JSON.
BATCH_SIZE = get_training_batch_size()
try:
    from .optimized_hyperparameters import load_optimized_parameters
    _opt_params = load_optimized_parameters()
    EPOCHS = _opt_params['epochs']
    LEARNING_RATE = _opt_params['learning_rate']
except Exception:
    EPOCHS = 12
    LEARNING_RATE = 0.001164

# Paths for the standalone prediction script (05_predict_new_data.py)
#PREDICT_INPUT_FOLDER = os.path.join(BASE_DIR, "data", "input_patches_total") # merge tiles
#PREDICT_OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "predictions_total") 


# --- 5. Fine-tuning Configuration ---

# Paths for the high-quality "gold" dataset
GOLD_PATCHES_FOLDER = os.path.join(BASE_DIR, "data_gold", "input_patches_training")
# GOLD_SYNTHETIC_FOLDER removed - synthetic data generation no longer used

# Path for the JSON file that will store the train/val/test splits for the gold data
GOLD_DATA_SPLIT_PATH = os.path.join(BASE_DIR, "gold_data_split.json")

# Fixed hold-out test set lives under test_data/AnnualTS (not in gold train pool)
TEST_DATA_FOLDER = os.path.join(BASE_DIR, "test_data", "AnnualTS")
GOLD_TEST_PATCH_IDS = sorted([
    f for f in os.listdir(TEST_DATA_FOLDER)
    if f.lower().endswith((".tif", ".tiff"))
]) if os.path.isdir(TEST_DATA_FOLDER) else []