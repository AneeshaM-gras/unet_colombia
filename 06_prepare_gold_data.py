#!/usr/bin/env python3
"""
Prepare gold data and create fine-tuning splits in one step.
"""

import argparse
import json
import os
import random

import rasterio
from rasterio import Affine

from src.config import (
    GOLD_DATA_SPLIT_PATH,
    GOLD_PATCHES_FOLDER,
    GOLD_TEST_PATCH_IDS,
    GOLD_VALIDATION_SIZE,
    PATCH_SIZE,
    RANDOM_SEED,
    TEST_DATA_FOLDER,
    VALIDATION_SPLIT_RATIO,
)
from src.patch_utils import center_crop_array


GOLD_IMAGE_BAND_NAMES = [f'A{i:02d}' for i in range(64)]
GOLD_LABEL_BAND_NAME = 'label'


def write_center_cropped_patch(src, output_path, descriptions):
    """Write a center-cropped PATCH_SIZE x PATCH_SIZE GeoTIFF from an open rasterio dataset."""
    data = src.read()
    cropped, row_offset, col_offset = center_crop_array(data, PATCH_SIZE)
    out_transform = src.transform * Affine.translation(col_offset, row_offset)

    out_meta = src.meta.copy()
    out_meta.update({
        'height': PATCH_SIZE,
        'width': PATCH_SIZE,
        'transform': out_transform,
    })

    temp_path = f"{output_path}.tmp.tif"
    with rasterio.open(temp_path, 'w', **out_meta) as dst:
        dst.write(cropped)
        dst.descriptions = descriptions

    os.replace(temp_path, output_path)


def prepare_gold_patches(input_dir, output_dir, force=False):
    print("Preparing gold patches...")

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if os.path.exists(output_dir) and not force:
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
        if existing_files:
            response = input(
                f"Output directory already contains {len(existing_files)} files. Continue anyway? (y/N): "
            )
            if response.lower() != 'y':
                print("Aborted.")
                return 0

    os.makedirs(output_dir, exist_ok=True)

    raw_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                raw_files.append(os.path.join(root, file))

    print(f"Found {len(raw_files)} raw files")
    print(f"Center-cropping all valid patches to {PATCH_SIZE}x{PATCH_SIZE}")

    patch_count = 0
    for raw_file in raw_files:
        try:
            with rasterio.open(raw_file) as src:
                if src.count != 65:
                    print(f"Skipping {raw_file}: Expected 65 bands, got {src.count}")
                    continue

                descriptions = list(src.descriptions)
                if len(descriptions) < 65:
                    print(f"Skipping {raw_file}: Insufficient band descriptions")
                    continue

                if descriptions[64] != GOLD_LABEL_BAND_NAME:
                    print(
                        f"Skipping {raw_file}: Label band is '{descriptions[64]}', "
                        f"expected '{GOLD_LABEL_BAND_NAME}'"
                    )
                    continue

                if src.height < PATCH_SIZE or src.width < PATCH_SIZE:
                    print(
                        f"Skipping {raw_file}: Size {src.height}x{src.width} is smaller than "
                        f"{PATCH_SIZE}x{PATCH_SIZE}"
                    )
                    continue

                base_name = os.path.splitext(os.path.basename(raw_file))[0]
                patch_filename = f"{base_name}.tif"
                patch_path = os.path.join(output_dir, patch_filename)

                write_center_cropped_patch(src, patch_path, descriptions)
                patch_count += 1

                if patch_count % 10 == 0:
                    print(f"  Processed {patch_count} patches...")

        except Exception as e:
            print(f"Error processing {raw_file}: {e}")
            continue

    print(f"Generated {patch_count} gold patches")
    return patch_count


def create_gold_splits(patches_dir):
    """
    Creates train, validation, and a fixed test split for the gold dataset.
    - Train/val come from gold patches in patches_dir.
    - Test comes from TEST_DATA_FOLDER (AnnualTS), linked via GOLD_TEST_PATCH_IDS.
    """
    print("--- CREATING CONSISTENT FINE-TUNING DATA SPLITS ---")

    # --- 1. Load all available gold files ---
    gold_files = {f for f in os.listdir(patches_dir) if f.endswith('.tif')}

    # --- 2. Define the Test Set from test_data/AnnualTS ---
    fixed_test_ids = set(GOLD_TEST_PATCH_IDS)
    if not os.path.isdir(TEST_DATA_FOLDER):
        raise FileNotFoundError(f"Test data folder not found: {TEST_DATA_FOLDER}")

    missing = [
        f for f in fixed_test_ids
        if not os.path.exists(os.path.join(TEST_DATA_FOLDER, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"The following test files defined in config.py were not found in "
            f"{TEST_DATA_FOLDER}: {missing}"
        )

    test_files = [os.path.join(TEST_DATA_FOLDER, f) for f in sorted(fixed_test_ids)]
    print(f"Defined a fixed test set with {len(test_files)} files from {TEST_DATA_FOLDER}.")

    # --- 3. Create the Training/Validation Pool ---
    # Gold train/val exclude any filename that also appears in the hold-out test set
    train_val_pool_ids = gold_files - fixed_test_ids
    train_val_pool_paths = [os.path.join(patches_dir, f) for f in train_val_pool_ids]

    print(f"Training/Validation pool contains {len(train_val_pool_paths)} files (all from gold data).")

    # --- 4. Split the Pool into Training and Validation ---
    random.seed(RANDOM_SEED)
    random.shuffle(train_val_pool_paths)

    if GOLD_VALIDATION_SIZE is not None:
        val_count = min(int(GOLD_VALIDATION_SIZE), len(train_val_pool_paths))
    else:
        val_count = int(len(train_val_pool_paths) * VALIDATION_SPLIT_RATIO)
    val_files = train_val_pool_paths[:val_count]
    train_files = train_val_pool_paths[val_count:]

    # --- 5. Save the Splits to gold_data_split.json ---
    split_data = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }

    with open(GOLD_DATA_SPLIT_PATH, 'w') as f:
        json.dump(split_data, f, indent=4)

    print("\n--- SUMMARY ---")
    print(f"Fine-tuning Training files:   {len(train_files)}")
    print(f"Fine-tuning Validation files: {len(val_files)}")
    print(f"Fine-tuning Test files:       {len(test_files)} (Fixed from AnnualTS)")
    print(f"\nSUCCESS: Gold data splits saved to '{GOLD_DATA_SPLIT_PATH}'")


def main():
    parser = argparse.ArgumentParser(description='Prepare gold data and create splits')
    parser.add_argument('--input_dir', type=str, default='data_gold/raw',
                        help='Input directory containing raw data')
    parser.add_argument('--output_dir', type=str, default=GOLD_PATCHES_FOLDER,
                        help='Output directory for gold patches')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if output exists')

    args = parser.parse_args()

    print("=== DATA GOLD PREPARATION ===")
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    patch_count = prepare_gold_patches(args.input_dir, args.output_dir, force=args.force)
    existing = (
        [f for f in os.listdir(args.output_dir) if f.endswith('.tif')]
        if os.path.isdir(args.output_dir) else []
    )
    if patch_count == 0 and not existing:
        return

    if args.output_dir != GOLD_PATCHES_FOLDER:
        print(
            f"WARNING: output_dir '{args.output_dir}' does not match GOLD_PATCHES_FOLDER "
            f"('{GOLD_PATCHES_FOLDER}'). Splits will be created from output_dir."
        )

    create_gold_splits(args.output_dir)

    print("\n=== DATA GOLD PREPARATION COMPLETE ===")


if __name__ == "__main__":
    main()
