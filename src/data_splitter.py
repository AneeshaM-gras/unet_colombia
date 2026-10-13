# src/data_splitter.py

import os
import random
import json
import rasterio
import numpy as np
from .config import (
    FINAL_PATCHES_FOLDER, TRAIN_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, 
    RANDOM_SEED, DATA_SPLIT_PATH
)

def _analyze_dominant_class(file_path):
    """(Internal) Analyzes the dominant class in a single GeoTIFF file."""
    try:
        with rasterio.open(file_path) as src:
            label_band = src.read(src.count)
            unique, counts = np.unique(label_band, return_counts=True)
            dominant_class = unique[counts.argmax()]
            return int(dominant_class)
    except Exception:
        return 0

def create_and_save_splits(force_rerun=False):
    """
    Performs a stratified train/validation/test split on the entire dataset
    and saves the file lists. Skips if the output file already exists,
    unless force_rerun is True.
    """
    # This check will now work correctly because force_rerun is a defined parameter.
    if os.path.exists(DATA_SPLIT_PATH) and not force_rerun:
        print(f"INFO: Data split file '{os.path.basename(DATA_SPLIT_PATH)}' already exists. Skipping split creation.")
        return
    
    if os.path.exists(DATA_SPLIT_PATH) and force_rerun:
        print(f"INFO: `force_rerun` is True. Overwriting existing data split file.")
        
    # --- END OF FIX ---

    print("--- Creating Consistent Data Splits ---")
    
    all_files = [os.path.join(FINAL_PATCHES_FOLDER, f) 
                 for f in os.listdir(FINAL_PATCHES_FOLDER) 
                 if f.lower().endswith(('.tif', '.tiff'))]
                 
    if not all_files:
        raise FileNotFoundError(f"ERROR: No patch files found in '{FINAL_PATCHES_FOLDER}'. Cannot create splits.")

    print(f"Found {len(all_files)} total patch files. Stratifying...")

    class_groups = {}
    for file_path in all_files:
        dominant_class = _analyze_dominant_class(file_path)
        if dominant_class not in class_groups:
            class_groups[dominant_class] = []
        class_groups[dominant_class].append(file_path)

    train_files, val_files, test_files = [], [], []
    random.seed(RANDOM_SEED)

    for files in class_groups.values():
        random.shuffle(files)
        n_files = len(files)
        train_count = int(TRAIN_SPLIT_RATIO * n_files)
        val_count = int(VALIDATION_SPLIT_RATIO * n_files)

        train_files.extend(files[:train_count])
        val_files.extend(files[train_count : train_count + val_count])
        test_files.extend(files[train_count + val_count :])
    
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    split_data = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }

    with open(DATA_SPLIT_PATH, 'w') as f:
        json.dump(split_data, f, indent=4)
        
    print("\n--- Data Split Summary ---")
    print(f"Training files:   {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files:       {len(test_files)}")
    print(f"SUCCESS: Data splits saved to '{DATA_SPLIT_PATH}'")