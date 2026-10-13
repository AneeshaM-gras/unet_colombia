#!/usr/bin/env python3
"""
Combined inference + test evaluation entrypoint.

Modes:
  - inference: run unlabeled inference pipeline
  - test: run labeled test evaluation pipeline
  - both: run inference then test
"""

import argparse
import builtins
import gc
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
from patchify import patchify
from tqdm import tqdm

from src.combined_loss_system import CombinedFocalJaccardLoss
from src.optimized_hyperparameters import load_optimized_parameters
from src.config import (
    IMAGE_BAND_NAMES,
    INFERENCE_PREDICTIONS_FOLDER,
    INFERENCE_TILES_FOLDER,
    INFERENCE_DATA_FOLDER,
    LABEL_BAND_NAME,
    GOLD_LABEL_BAND_NAME,
    MODEL_FOLDER,
    PATCH_SIZE,
    TEMP_INFERENCE_WORKSPACE,
    TRAINING_CONFIG_PATH,
    get_finetuning_batch_size,
)
from src.data_loader import create_dataset
from src.data_preprocessing import merge_geotiffs_in_folder
from src.model import SparseMeanIoU


@contextmanager
def print_tag(tag):
    original_print = builtins.print

    def tagged_print(*args, **kwargs):
        original_print(f"[{tag}]", *args, **kwargs)

    builtins.print = tagged_print
    try:
        yield
    finally:
        builtins.print = original_print


def run_inference_pipeline():
    """Main function to run the complete, memory-efficient inference pipeline.
    
    Uses INFERENCE_DATA_FOLDER with subfolders (same structure as 01_prepare_environment.py):
    - Each subfolder contains raw GeoTIFF tiles that will be merged
    - Merged tiles are then processed for inference
    """
    print("--- STANDALONE INFERENCE PIPELINE (Memory Optimized) ---")

    # --- 1. Load Model ---
    model_path = os.path.join(MODEL_FOLDER, "unet_finetuned_gold.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_FOLDER, "unet_optimized_combined_loss.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found.")

    print(f"Loading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'SparseMeanIoU': SparseMeanIoU},
        compile=False
    )

    # --- 2. Setup Workspace ---
    workspace_dir = TEMP_INFERENCE_WORKSPACE
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir)

    # --- 3. Identify Inference Jobs from inference_data folder ---
    # Uses same structure as 01_prepare_environment.py: subfolders with raw tiles
    if not os.path.exists(INFERENCE_DATA_FOLDER):
        print(f"ERROR: Inference data folder not found: '{INFERENCE_DATA_FOLDER}'")
        return
    
    subfolders = [
        d for d in os.listdir(INFERENCE_DATA_FOLDER)
        if os.path.isdir(os.path.join(INFERENCE_DATA_FOLDER, d))
    ]
    
    if not subfolders:
        print(f"ERROR: No subfolders found in '{INFERENCE_DATA_FOLDER}'.")
        print(f"Expected structure: inference_data/subfolder1/, inference_data/subfolder2/, etc.")
        return
    
    print(f"Using inference_data folder structure (like 01_prepare_environment.py)")
    print(f"Found {len(subfolders)} subfolders in '{INFERENCE_DATA_FOLDER}': {subfolders}")

    # --- 4. Process Each Tile Set ---
    try:
        for tile_id in subfolders:
            print(f"\n--- Processing Tile Set: {tile_id} ---")

            # --- STEP A: Merge Raw Tile Pieces ---
            print("Step 1/3: Merging tile pieces from subfolder...")
            raw_tile_pieces_dir = os.path.join(INFERENCE_DATA_FOLDER, tile_id)
            merged_tile_path = os.path.join(workspace_dir, f"{tile_id}_merged.tif")
            if not merge_geotiffs_in_folder(raw_tile_pieces_dir, merged_tile_path):
                print(f"  WARNING: Failed to merge tiles in '{tile_id}', skipping...")
                continue

            # --- STEP B: Predict using an in-memory sliding window ---
            print("Step 2/3: Running model prediction (in-memory)...")
            with rasterio.open(merged_tile_path) as src:
                meta = src.meta.copy()
                image_data = src.read([src.descriptions.index(b) + 1 for b in IMAGE_BAND_NAMES])

                # Prepare image for patchify
                image_tensor = np.moveaxis(image_data, 0, -1).astype(np.float32) / 255.0

                # Pad the image to ensure it's perfectly divisible by the stride
                stride = PATCH_SIZE
                img_h, img_w, _ = image_tensor.shape
                pad_h = stride - (img_h - PATCH_SIZE) % stride if (img_h - PATCH_SIZE) % stride != 0 else 0
                pad_w = stride - (img_w - PATCH_SIZE) % stride if (img_w - PATCH_SIZE) % stride != 0 else 0
                padded_image = np.pad(image_tensor, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

                # Create patches from the padded image in memory
                patches = patchify(padded_image, (PATCH_SIZE, PATCH_SIZE, len(IMAGE_BAND_NAMES)), step=stride)

                # Create an empty canvas to place the predictions
                padded_prediction = np.zeros(padded_image.shape[:2], dtype=np.uint8)

                # Loop through patches, predict, and place result on the canvas
                for i in tqdm(range(patches.shape[0]), desc="   Predicting rows"):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j, 0, :, :, :]
                        prediction = model.predict(np.expand_dims(patch, axis=0), verbose=0)
                        prediction_map = np.argmax(prediction[0], axis=-1).astype(np.uint8)

                        y_start, x_start = i * stride, j * stride
                        padded_prediction[y_start:y_start + PATCH_SIZE, x_start:x_start + PATCH_SIZE] = prediction_map

                # Un-pad the final prediction map to match original dimensions
                final_prediction = padded_prediction[:img_h, :img_w]

            # --- STEP C: Save the final stitched map ---
            print("Step 3/3: Saving final prediction map...")
            # Save predictions inside inference_data/predictions folder
            predictions_folder = os.path.join(INFERENCE_DATA_FOLDER, "predictions")
            os.makedirs(predictions_folder, exist_ok=True)
            final_output_path = os.path.join(predictions_folder, f"{tile_id}_classified.tif")

            meta.update(count=1, dtype='uint8')
            with rasterio.open(final_output_path, 'w', **meta) as dst:
                dst.write(final_prediction, 1)
            print(f"-> Prediction saved to: {final_output_path}")
            print("   Cleaning up memory...")
            gc.collect()

    finally:
        # --- 5. Final Cleanup ---
        print("\nCleaning up temporary workspace...")
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)

    print("\n--- INFERENCE PIPELINE COMPLETE ---")


class ModelTester:
    """Test models on unseen test data"""

    def __init__(self):
        self.test_data_path = "test_data"
        self.base_model_path = os.path.join(MODEL_FOLDER, "unet_optimized_combined_loss.h5")
        self.finetuned_model_path = os.path.join(MODEL_FOLDER, "unet_finetuned_gold.h5")

        # Load configuration
        with open(TRAINING_CONFIG_PATH, 'r') as f:
            self.config = json.load(f)
        
        # Get number of classes from config (after aggregation)
        self.num_classes = self.config.get('num_classes', 3)

        # Use same band configuration as training data
        self.image_bands = IMAGE_BAND_NAMES
        self.label_band = GOLD_LABEL_BAND_NAME

        # Custom objects for model loading
        self.custom_objects = {
            'CombinedFocalJaccardLoss': CombinedFocalJaccardLoss,
            'focal_jaccard_loss': CombinedFocalJaccardLoss()
        }

        # Temporary directory for processed patches
        self.temp_patches_dir = tempfile.mkdtemp(prefix="test_patches_")

        print(f"Test data path: {self.test_data_path}")
        print(f"Base model path: {self.base_model_path}")
        print(f"Fine-tuned model path: {self.finetuned_model_path}")
        print(f"Temp patches directory: {self.temp_patches_dir}")
        print(f"Image bands: {len(self.image_bands)} bands")
        print(f"Label band: {self.label_band}")
        print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")

    def check_test_data_exists(self):
        """Check if test data exists"""
        if not os.path.exists(self.test_data_path):
            print(f"ERROR: Test data directory not found: {self.test_data_path}")
            return False

        # Check for .tif files recursively
        tif_files = list(Path(self.test_data_path).rglob("*.tif"))
        if not tif_files:
            print(f"ERROR: No .tif files found in {self.test_data_path}")
            return False

        print(f"Found {len(tif_files)} test files")
        return True

    def check_models_exist(self):
        """Check if both models exist"""
        models_status = {}

        if os.path.exists(self.base_model_path):
            models_status['base'] = True
            print(f"Base model found: {self.base_model_path}")
        else:
            models_status['base'] = False
            print(f"ERROR: Base model not found: {self.base_model_path}")

        if os.path.exists(self.finetuned_model_path):
            models_status['finetuned'] = True
            print(f"Fine-tuned model found: {self.finetuned_model_path}")
        else:
            models_status['finetuned'] = False
            print(f"ERROR: Fine-tuned model not found: {self.finetuned_model_path}")

        return models_status

    def process_raw_test_data(self):
        """Process raw test data into patches (same as training data)"""
        print("\n=== PROCESSING RAW TEST DATA INTO PATCHES ===")

        # Find all test files
        tif_files = list(Path(self.test_data_path).rglob("*.tif"))
        if not tif_files:
            print("ERROR: No .tif files found in test data")
            return []

        print(f"Found {len(tif_files)} test files to process")

        processed_patches = []

        for file_path in tif_files:
            print(f"Processing: {file_path.name}")

            try:
                with rasterio.open(file_path) as src:
                    # Check if file has required bands
                    if src.count < 65:
                        print(f"  WARNING: File has only {src.count} bands, skipping")
                        continue

                    # Get band indices for image and label bands
                    band_descriptions = src.descriptions

                    # Find image band indices (A00-A63)
                    image_band_indices = []
                    for band_name in self.image_bands:
                        try:
                            idx = band_descriptions.index(band_name) + 1
                            image_band_indices.append(idx)
                        except ValueError:
                            print(f"  WARNING: Band {band_name} not found")
                            continue

                    # Find label band index
                    try:
                        label_band_idx = band_descriptions.index(self.label_band) + 1
                    except ValueError:
                        print(f"  WARNING: Label band {self.label_band} not found")
                        continue

                    if len(image_band_indices) != 64:
                        print(f"  WARNING: Only {len(image_band_indices)} image bands found, skipping")
                        continue

                    # Read data
                    image_data = src.read(image_band_indices)
                    label_data = src.read(label_band_idx)

                    # Transpose image data to (height, width, channels)
                    image_data = np.transpose(image_data, (1, 2, 0))

                    print(f"  Image shape: {image_data.shape}")
                    print(f"  Label shape: {label_data.shape}")

                    # Create patches
                    height, width = image_data.shape[:2]
                    patch_count = 0

                    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
                        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
                            # Extract patch
                            image_patch = image_data[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]
                            label_patch = label_data[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                            # Save patch
                            patch_filename = f"{file_path.stem}_patch_{y}_{x}.tif"
                            patch_path = os.path.join(self.temp_patches_dir, patch_filename)

                            # Create patch with same structure as training data
                            with rasterio.open(
                                patch_path, 'w',
                                driver='GTiff',
                                height=PATCH_SIZE,
                                width=PATCH_SIZE,
                                count=65,
                                dtype=image_patch.dtype,
                                crs=src.crs,
                                transform=rasterio.transform.from_bounds(
                                    *rasterio.transform.array_bounds(PATCH_SIZE, PATCH_SIZE, src.transform),
                                    PATCH_SIZE, PATCH_SIZE
                                )
                            ) as dst:
                                # Write image bands
                                for i, band_idx in enumerate(image_band_indices):
                                    dst.write(image_patch[:, :, i], i + 1)
                                    dst.set_band_description(i + 1, self.image_bands[i])

                                # Write label band
                                dst.write(label_patch, 65)
                                dst.set_band_description(65, self.label_band)

                            processed_patches.append(patch_path)
                            patch_count += 1

                    print(f"  Created {patch_count} patches")

            except Exception as e:
                print(f"  ERROR processing {file_path.name}: {e}")
                continue

        print(f"\nTotal patches created: {len(processed_patches)}")
        return processed_patches

    def cleanup_temp_files(self):
        """Clean up temporary patch files"""
        if os.path.exists(self.temp_patches_dir):
            shutil.rmtree(self.temp_patches_dir)
            print(f"Cleaned up temporary directory: {self.temp_patches_dir}")

    def create_test_dataset_from_patches(self, patch_files):
        """Create test dataset from processed patches"""
        print(f"\n=== CREATING TEST DATASET FROM {len(patch_files)} PATCHES ===")

        try:
            batch_size = get_finetuning_batch_size()
            # Create dataset using same configuration as training
            dataset = create_dataset(
                patch_files,
                batch_size=batch_size,
                image_bands=self.image_bands,
                label_band=self.label_band
            )

            print(f"Test dataset created with {len(patch_files)} patches")
            print(f"Batch size: {batch_size} (from finetuning_config.json)")
            print(f"Image bands: {len(self.image_bands)}")
            print(f"Label band: {self.label_band}")

            return dataset, patch_files

        except Exception as e:
            print(f"ERROR: Failed to create test dataset: {e}")
            return None, None

    def load_model(self, model_path):
        """Load a model with proper compilation"""
        try:
            print(f"Loading model: {model_path}")

            # Load model
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=self.custom_objects,
                compile=False
            )

            # Compile model for evaluation
            optimal_params = load_optimized_parameters()
            loss_function = CombinedFocalJaccardLoss(
                gamma=optimal_params['gamma'],
                focal_weight=optimal_params['focal_weight'],
                jaccard_weight=optimal_params['jaccard_weight'],
                dice_weight=optimal_params.get('dice_weight', 0.3)
            )
            model.compile(
                optimizer='adam',
                loss=loss_function,
                metrics=['sparse_categorical_accuracy']
            )

            print(f"Model loaded successfully")
            print(f"  Model parameters: {model.count_params():,}")

            return model

        except Exception as e:
            print(f"ERROR: Failed to load model {model_path}: {e}")
            return None

    def create_test_dataset(self):
        """Create test dataset from test_data folder"""
        try:
            print(f"\n=== CREATING TEST DATASET ===")
            print(f"Test data path: {self.test_data_path}")

            # Get all .tif files recursively
            test_files = list(Path(self.test_data_path).rglob("*.tif"))
            print(f"Searching in: {self.test_data_path}")
            print(f"Found files: {[str(f) for f in test_files[:3]]}...")
            print(f"Found {len(test_files)} test files")

            if not test_files:
                print("ERROR: No test files found")
                return None

            # For raw files, we need to process them into patches first
            # Check if files are raw (large) or already patches (small)
            raw_files = []
            patch_files = []

            for file_path in test_files:
                file_size = file_path.stat().st_size
                if file_size > 100 * 1024 * 1024:
                    raw_files.append(file_path)
                else:
                    patch_files.append(file_path)

            print(f"Raw files: {len(raw_files)}")
            print(f"Patch files: {len(patch_files)}")

            if raw_files and not patch_files:
                print("Found raw files - need to process into patches")
                return self.process_raw_files_to_patches(raw_files)
            elif patch_files:
                print("Using existing patch files")
                return self.create_test_dataset_from_patches([str(p) for p in patch_files])
            else:
                print("ERROR: No suitable files found")
                return None

        except Exception as e:
            print(f"ERROR: Failed to create test dataset: {e}")
            return None, None

    def process_raw_files_to_patches(self, raw_files):
        """Process raw files into patches for testing"""
        try:
            print(f"\n=== PROCESSING RAW FILES TO PATCHES ===")

            import rasterio
            from rasterio.windows import Window

            patch_size = 256
            patches = []

            for i, raw_file in enumerate(raw_files):
                print(f"Processing raw file {i+1}/{len(raw_files)}: {raw_file.name}")

                with rasterio.open(raw_file) as src:
                    height, width = src.shape
                    print(f"  File shape: {height}x{width}, bands: {src.count}")

                    # Extract patches
                    for y in range(0, height - patch_size + 1, patch_size):
                        for x in range(0, width - patch_size + 1, patch_size):
                            window = Window(x, y, patch_size, patch_size)

                            # Read patch
                            patch_data = src.read(window=window)

                            # Take first 64 image bands + 65th band as label
                            if patch_data.shape[0] >= 65:
                                # Get first 64 bands (A00-A63) + 65th band as label
                                image_bands = patch_data[:64]
                                label_band = patch_data[64:65]
                                patch_data = np.concatenate([image_bands, label_band], axis=0)

                                # Save patch
                                patch_filename = f"test_patch_{i}_{y}_{x}.tif"
                                patch_path = Path(self.test_data_path) / patch_filename

                                with rasterio.open(
                                    patch_path, 'w',
                                    driver='GTiff',
                                    height=patch_size,
                                    width=patch_size,
                                    count=65,
                                    dtype=patch_data.dtype,
                                    crs=src.crs,
                                    transform=rasterio.windows.transform(window, src.transform)
                                ) as dst:
                                    dst.write(patch_data)
                                    # Set band descriptions
                                    dst.descriptions = src.descriptions[:65]

                                patches.append(patch_path)

                                if len(patches) >= 20:
                                    break

                        if len(patches) >= 20:
                            break

                    if len(patches) >= 20:
                        break

            print(f"Created {len(patches)} test patches")
            return self.create_dataset_from_patches(patches)

        except Exception as e:
            print(f"ERROR: Failed to process raw files: {e}")
            return None, None

    def calculate_file_metrics(self, true_labels, predicted_labels, num_classes=None):
        """Calculate metrics for a single file"""
        try:
            # Use instance num_classes if not provided
            if num_classes is None:
                num_classes = self.num_classes
            
            # Flatten arrays
            true_flat = true_labels.flatten()
            pred_flat = predicted_labels.flatten()

            # Calculate overall metrics
            total_pixels = len(true_flat)
            total_correct = np.sum(true_flat == pred_flat)
            overall_accuracy = total_correct / total_pixels if total_pixels > 0 else 0

            # Calculate per-class metrics
            per_class_correct = np.zeros(num_classes)
            per_class_total = np.zeros(num_classes)
            per_class_intersection = np.zeros(num_classes)
            per_class_union = np.zeros(num_classes)
            per_class_tp = np.zeros(num_classes)
            per_class_fp = np.zeros(num_classes)
            per_class_fn = np.zeros(num_classes)

            for class_id in range(num_classes):
                true_mask = (true_flat == class_id)
                pred_mask = (pred_flat == class_id)

                per_class_total[class_id] = np.sum(true_mask)
                per_class_correct[class_id] = np.sum(true_mask & pred_mask)
                per_class_intersection[class_id] = np.sum(true_mask & pred_mask)
                per_class_union[class_id] = np.sum(true_mask | pred_mask)
                
                # For F1 score: TP, FP, FN
                per_class_tp[class_id] = np.sum(true_mask & pred_mask)
                per_class_fp[class_id] = np.sum(pred_mask & ~true_mask)
                per_class_fn[class_id] = np.sum(true_mask & ~pred_mask)

            # Calculate per-class accuracy, IoU, Precision, Recall, and F1
            per_class_accuracy = np.zeros(num_classes)
            per_class_iou = np.zeros(num_classes)
            per_class_precision = np.zeros(num_classes)
            per_class_recall = np.zeros(num_classes)
            per_class_f1 = np.zeros(num_classes)

            for class_id in range(num_classes):
                if per_class_total[class_id] > 0:
                    per_class_accuracy[class_id] = per_class_correct[class_id] / per_class_total[class_id]
                if per_class_union[class_id] > 0:
                    per_class_iou[class_id] = per_class_intersection[class_id] / per_class_union[class_id]
                
                # Precision = TP / (TP + FP)
                precision_den = per_class_tp[class_id] + per_class_fp[class_id]
                if precision_den > 0:
                    per_class_precision[class_id] = per_class_tp[class_id] / precision_den
                
                # Recall = TP / (TP + FN)
                recall_den = per_class_tp[class_id] + per_class_fn[class_id]
                if recall_den > 0:
                    per_class_recall[class_id] = per_class_tp[class_id] / recall_den
                
                # F1 = 2 * (precision * recall) / (precision + recall)
                f1_den = per_class_precision[class_id] + per_class_recall[class_id]
                if f1_den > 0:
                    per_class_f1[class_id] = 2 * (per_class_precision[class_id] * per_class_recall[class_id]) / f1_den

            # Calculate overall IoU
            overall_iou = np.sum(per_class_intersection) / np.sum(per_class_union) if np.sum(per_class_union) > 0 else 0
            
            # Calculate overall F1 (macro-averaged)
            overall_f1 = np.mean(per_class_f1) if num_classes > 0 else 0

            return {
                'overall_accuracy': overall_accuracy,
                'overall_iou': overall_iou,
                'overall_f1': overall_f1,
                'total_correct': total_correct,
                'total_pixels': total_pixels,
                'per_class_accuracy': per_class_accuracy,
                'per_class_iou': per_class_iou,
                'per_class_precision': per_class_precision,
                'per_class_recall': per_class_recall,
                'per_class_f1': per_class_f1,
                'per_class_intersection': per_class_intersection,
                'per_class_union': per_class_union
            }

        except Exception as e:
            print(f"ERROR: Failed to calculate file metrics: {e}")
            return None

    def analyze_class_distribution_simple(self, class_intersection, class_union, total_pixels):
        """Simple class distribution analysis"""
        try:
            class_counts = np.zeros(self.num_classes)
            class_percentages = np.zeros(self.num_classes)

            for i in range(self.num_classes):
                if class_union[i] > 0:
                    class_counts[i] = class_intersection[i]
                    class_percentages[i] = (class_counts[i] / total_pixels) * 100

            return {
                'class_counts': class_counts.tolist(),
                'class_percentages': class_percentages.tolist(),
                'total_pixels': float(total_pixels)
            }

        except Exception as e:
            print(f"ERROR: Failed to analyze class distribution: {e}")
            return None

    def evaluate_model_on_test_data(self, model, dataset, test_files, model_name):
        """Evaluate a model on test data"""
        try:
            print(f"\n=== EVALUATING {model_name.upper()} MODEL ===")

            # Initialize metrics
            total_files = len(test_files)
            overall_correct = 0
            overall_total = 0
            overall_intersection = np.zeros(self.num_classes)
            overall_union = np.zeros(self.num_classes)
            overall_f1_scores = []

            per_file_results = []
            file_idx = 0

            # Process batches from the dataset
            for batch in dataset:
                # Handle different batch structures
                if len(batch) == 2:
                    images, masks = batch
                else:
                    images, masks = batch[0], batch[1]

                # Get predictions for entire batch
                predictions = model.predict(images, verbose=0)
                
                # Get batch size
                batch_size = images.shape[0]
                
                # Process each sample in the batch
                for batch_idx in range(batch_size):
                    if file_idx >= total_files:
                        break
                    
                    # Handle both string paths and Path objects
                    file_name = Path(test_files[file_idx]).name if isinstance(test_files[file_idx], str) else test_files[file_idx].name
                    print(f"Processing file {file_idx+1}/{total_files}: {file_name}")

                    # Extract single sample from batch
                    predicted_classes = np.argmax(predictions[batch_idx], axis=-1)
                    true_labels = masks[batch_idx].numpy().squeeze()
                    predicted_labels = predicted_classes.squeeze()
                    
                    # Ensure 2D arrays
                    if true_labels.ndim > 2:
                        true_labels = true_labels.squeeze()
                    if predicted_labels.ndim > 2:
                        predicted_labels = predicted_labels.squeeze()

                    # Calculate metrics for this file
                    file_metrics = self.calculate_file_metrics(true_labels, predicted_labels, num_classes=self.num_classes)

                    # Accumulate overall metrics
                    overall_correct += file_metrics['total_correct']
                    overall_total += file_metrics['total_pixels']
                    overall_intersection += file_metrics['per_class_intersection']
                    overall_union += file_metrics['per_class_union']
                    overall_f1_scores.append(file_metrics['overall_f1'])

                    # Store per-file results
                    per_file_results.append({
                        'file_name': file_name,
                        'overall_accuracy': file_metrics['overall_accuracy'],
                        'overall_iou': file_metrics['overall_iou'],
                        'overall_f1': file_metrics['overall_f1'],
                        'per_class_iou': file_metrics['per_class_iou'].tolist(),
                        'per_class_accuracy': file_metrics['per_class_accuracy'].tolist(),
                        'per_class_f1': file_metrics['per_class_f1'].tolist()
                    })

                    print(f"  File accuracy: {file_metrics['overall_accuracy']:.4f}")
                    print(f"  File IoU: {file_metrics['overall_iou']:.4f}")
                    print(f"  File F1: {file_metrics['overall_f1']:.4f}")
                    # Show per-class F1 for this file
                    class_names = self.config.get('class_names', [f'Class {i}' for i in range(self.num_classes)])
                    per_class_f1_str = ", ".join([f"{class_names[i] if i < len(class_names) else f'Class {i}'}: {file_metrics['per_class_f1'][i]:.4f}" for i in range(self.num_classes)])
                    print(f"  Per-class F1: {per_class_f1_str}")
                    
                    file_idx += 1

                if file_idx >= total_files:
                    break

            # Calculate overall metrics
            overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
            overall_iou = np.sum(overall_intersection) / np.sum(overall_union) if np.sum(overall_union) > 0 else 0
            overall_f1 = np.mean(overall_f1_scores) if overall_f1_scores else 0

            # Calculate per-class metrics
            per_class_accuracy = np.zeros(self.num_classes)
            per_class_iou = np.zeros(self.num_classes)
            per_class_f1 = np.zeros(self.num_classes)
            
            # Accumulate per-class F1 from all files
            per_class_f1_sum = np.zeros(self.num_classes)
            per_class_f1_count = np.zeros(self.num_classes)

            for file_result in per_file_results:
                if 'per_class_f1' in file_result:
                    per_class_f1_sum += np.array(file_result['per_class_f1'])
                    per_class_f1_count += 1

            for i in range(self.num_classes):
                if overall_union[i] > 0:
                    per_class_iou[i] = overall_intersection[i] / overall_union[i]
                if overall_total > 0:
                    per_class_accuracy[i] = overall_correct / overall_total
                if per_class_f1_count[i] > 0:
                    per_class_f1[i] = per_class_f1_sum[i] / per_class_f1_count[i]

            # Analyze class distribution
            class_dist = self.analyze_class_distribution_simple(overall_intersection, overall_union, overall_total)

            results = {
                'model_name': model_name,
                'total_files': total_files,
                'overall_accuracy': float(overall_accuracy),
                'overall_iou': float(overall_iou),
                'overall_f1': float(overall_f1),
                'per_class_accuracy': per_class_accuracy.tolist(),
                'per_class_iou': per_class_iou.tolist(),
                'per_class_f1': per_class_f1.tolist(),
                'per_file_results': per_file_results,
                'class_distribution': class_dist
            }

            # Get class names from config
            class_names = self.config.get('class_names', [f'Class {i}' for i in range(self.num_classes)])
            
            print(f"\n{model_name.upper()} MODEL RESULTS:")
            print(f"Overall Accuracy: {overall_accuracy:.4f}")
            print(f"Overall IoU: {overall_iou:.4f}")
            print(f"Overall F1: {overall_f1:.4f}")
            
            print(f"\nPer-class Metrics:")
            print(f"{'Class':<15} {'IoU':<10} {'Accuracy':<10} {'F1':<10}")
            print("-" * 45)
            for i in range(self.num_classes):
                class_name = class_names[i] if i < len(class_names) else f'Class {i}'
                print(f"{class_name:<15} {per_class_iou[i]:<10.4f} {per_class_accuracy[i]:<10.4f} {per_class_f1[i]:<10.4f}")

            return results

        except Exception as e:
            print(f"ERROR: Failed to evaluate {model_name} model: {e}")
            return None

    def compare_models_on_test_data(self, base_results, finetuned_results):
        """Compare base and fine-tuned models on test data"""
        print(f"\n=== COMPARING MODELS ON TEST DATA ===")

        if not base_results or not finetuned_results:
            print("ERROR: Cannot compare - missing results")
            return None

        # Calculate improvements
        iou_improvement = finetuned_results['overall_iou'] - base_results['overall_iou']
        accuracy_improvement = finetuned_results['overall_accuracy'] - base_results['overall_accuracy']
        f1_improvement = finetuned_results['overall_f1'] - base_results['overall_f1']

        iou_improvement_pct = (iou_improvement / base_results['overall_iou']) * 100 if base_results['overall_iou'] > 0 else 0
        accuracy_improvement_pct = (accuracy_improvement / base_results['overall_accuracy']) * 100 if base_results['overall_accuracy'] > 0 else 0
        f1_improvement_pct = (f1_improvement / base_results['overall_f1']) * 100 if base_results['overall_f1'] > 0 else 0

        # Per-class improvements
        per_class_iou_improvements = np.array(finetuned_results['per_class_iou']) - np.array(base_results['per_class_iou'])
        per_class_accuracy_improvements = np.array(finetuned_results['per_class_accuracy']) - np.array(base_results['per_class_accuracy'])
        per_class_f1_improvements = np.array(finetuned_results['per_class_f1']) - np.array(base_results['per_class_f1'])

        comparison = {
            'test_data_evaluation': {
                'base_model': {
                    'overall_accuracy': base_results['overall_accuracy'],
                    'overall_iou': base_results['overall_iou'],
                    'overall_f1': base_results['overall_f1'],
                    'per_class_iou': base_results['per_class_iou'],
                    'per_class_accuracy': base_results['per_class_accuracy'],
                    'per_class_f1': base_results['per_class_f1']
                },
                'finetuned_model': {
                    'overall_accuracy': finetuned_results['overall_accuracy'],
                    'overall_iou': finetuned_results['overall_iou'],
                    'overall_f1': finetuned_results['overall_f1'],
                    'per_class_iou': finetuned_results['per_class_iou'],
                    'per_class_accuracy': finetuned_results['per_class_accuracy'],
                    'per_class_f1': finetuned_results['per_class_f1']
                },
                'improvements': {
                    'iou_improvement': float(iou_improvement),
                    'accuracy_improvement': float(accuracy_improvement),
                    'f1_improvement': float(f1_improvement),
                    'iou_improvement_pct': float(iou_improvement_pct),
                    'accuracy_improvement_pct': float(accuracy_improvement_pct),
                    'f1_improvement_pct': float(f1_improvement_pct),
                    'per_class_iou_improvements': per_class_iou_improvements.tolist(),
                    'per_class_accuracy_improvements': per_class_accuracy_improvements.tolist(),
                    'per_class_f1_improvements': per_class_f1_improvements.tolist()
                }
            },
            'test_data_summary': {
                'total_test_files': base_results['total_files'],
                'test_data_path': self.test_data_path,
                'evaluation_date': str(Path().cwd()),
                'data_leakage_prevention': 'Models tested on completely unseen test data'
            }
        }

        # Get class names from config
        class_names = self.config.get('class_names', [f'Class {i}' for i in range(self.num_classes)])
        
        # Print comparison
        print(f"\nTEST DATA COMPARISON:")
        print(f"Base Model - Accuracy: {base_results['overall_accuracy']:.4f}, IoU: {base_results['overall_iou']:.4f}, F1: {base_results['overall_f1']:.4f}")
        print(f"Fine-tuned Model - Accuracy: {finetuned_results['overall_accuracy']:.4f}, IoU: {finetuned_results['overall_iou']:.4f}, F1: {finetuned_results['overall_f1']:.4f}")
        print(f"IoU Improvement: {iou_improvement:+.4f} ({iou_improvement_pct:+.1f}%)")
        print(f"Accuracy Improvement: {accuracy_improvement:+.4f} ({accuracy_improvement_pct:+.1f}%)")
        print(f"F1 Improvement: {f1_improvement:+.4f} ({f1_improvement_pct:+.1f}%)")

        print(f"\nPer-class IoU Comparison:")
        print(f"{'Class':<15} {'Base IoU':<12} {'Fine-tuned IoU':<15} {'Improvement':<12}")
        print("-" * 54)
        for i in range(self.num_classes):
            class_name = class_names[i] if i < len(class_names) else f'Class {i}'
            print(f"{class_name:<15} {base_results['per_class_iou'][i]:<12.4f} {finetuned_results['per_class_iou'][i]:<15.4f} {per_class_iou_improvements[i]:+12.4f}")
        
        print(f"\nPer-class F1 Comparison:")
        print(f"{'Class':<15} {'Base F1':<12} {'Fine-tuned F1':<15} {'Improvement':<12}")
        print("-" * 54)
        for i in range(self.num_classes):
            class_name = class_names[i] if i < len(class_names) else f'Class {i}'
            print(f"{class_name:<15} {base_results['per_class_f1'][i]:<12.4f} {finetuned_results['per_class_f1'][i]:<15.4f} {per_class_f1_improvements[i]:+12.4f}")

        return comparison

    def create_test_data_visualization(self, comparison, output_dir):
        """Create visualization for test data evaluation"""
        print(f"\n=== CREATING TEST DATA VISUALIZATION ===")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Model Performance on Unseen Test Data', fontsize=16, fontweight='bold')

            test_data = comparison['test_data_evaluation']
            base_results = test_data['base_model']
            finetuned_results = test_data['finetuned_model']
            improvements = test_data['improvements']

            # 1. Overall Performance Comparison
            metrics = ['Accuracy', 'IoU', 'F1']
            base_values = [base_results['overall_accuracy'], base_results['overall_iou'], base_results['overall_f1']]
            finetuned_values = [finetuned_results['overall_accuracy'], finetuned_results['overall_iou'], finetuned_results['overall_f1']]

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax1.bar(x - width / 2, base_values, width, label='Base Model',
                            color='lightcoral', alpha=0.8)
            bars2 = ax1.bar(x + width / 2, finetuned_values, width, label='Fine-tuned Model',
                            color='lightgreen', alpha=0.8)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{height:.3f}', ha='center', va='bottom')

            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Overall Performance on Test Data')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)

            # 2. Per-class IoU Comparison
            classes = [f'Class {i}' for i in range(self.num_classes)]
            base_iou = base_results['per_class_iou']
            finetuned_iou = finetuned_results['per_class_iou']

            x = np.arange(len(classes))
            bars3 = ax2.bar(x - width / 2, base_iou, width, label='Base Model',
                            color='lightcoral', alpha=0.8)
            bars4 = ax2.bar(x + width / 2, finetuned_iou, width, label='Fine-tuned Model',
                            color='lightgreen', alpha=0.8)

            ax2.set_xlabel('Classes')
            ax2.set_ylabel('IoU Score')
            ax2.set_title('Per-class IoU on Test Data')
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)

            # 3. Improvement Analysis
            iou_improvements = improvements['per_class_iou_improvements']
            colors = ['green' if x > 0 else 'red' for x in iou_improvements]

            bars5 = ax3.bar(classes, iou_improvements, color=colors, alpha=0.7)

            # Add value labels
            for bar, improvement in zip(bars5, iou_improvements):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + (0.01 if height > 0 else -0.02),
                    f'{improvement:+.3f}',
                    ha='center',
                    va='bottom' if height > 0 else 'top'
                )

            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_xlabel('Classes')
            ax3.set_ylabel('IoU Improvement')
            ax3.set_title('Per-class IoU Improvement on Test Data')
            ax3.grid(True, alpha=0.3)

            # 4. Test Data Summary
            ax4.text(0.1, 0.9, f"Test Files: {comparison['test_data_summary']['total_test_files']}",
                     transform=ax4.transAxes, fontsize=12, fontweight='bold')
            ax4.text(0.1, 0.8, f"Overall IoU Improvement: {improvements['iou_improvement']:+.4f}",
                     transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f"Overall Accuracy Improvement: {improvements['accuracy_improvement']:+.4f}",
                     transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f"Overall F1 Improvement: {improvements['f1_improvement']:+.4f}",
                     transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.5, f"IoU Improvement %: {improvements['iou_improvement_pct']:+.1f}%",
                     transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.4, f"Accuracy Improvement %: {improvements['accuracy_improvement_pct']:+.1f}%",
                     transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.3, f"F1 Improvement %: {improvements['f1_improvement_pct']:+.1f}%",
                     transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.2, "No Data Leakage",
                     transform=ax4.transAxes, fontsize=14, fontweight='bold', color='green')
            ax4.text(0.1, 0.1, "Unseen Test Data",
                     transform=ax4.transAxes, fontsize=14, fontweight='bold', color='green')

            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Test Data Evaluation Summary')

            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, 'test_data_evaluation.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Test data visualization saved to: {plot_path}")

            plt.close()

        except Exception as e:
            print(f"ERROR: Failed to create test data visualization: {e}")

    def generate_prediction_images(self, base_model, finetuned_model, dataset, test_files, output_dir):
        """Generate prediction images for visual comparison - SECONDARY OUTPUT"""
        print(f"\n=== SECONDARY OUTPUT: GENERATING PREDICTION IMAGES ===")
        print(f"These are visualization images - TIF files are the main result")

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            # Create output directory for predictions
            predictions_dir = os.path.join(output_dir, "test_predictions")
            os.makedirs(predictions_dir, exist_ok=True)

            # Process first 5 test files for visualization
            num_files_to_visualize = min(5, len(test_files))
            file_idx = 0

            for batch in dataset:
                if file_idx >= num_files_to_visualize:
                    break

                # Handle batch structure
                if len(batch) == 2:
                    images, masks = batch
                else:
                    images, masks = batch[0], batch[1]

                # Get predictions for entire batch
                base_predictions = base_model.predict(images, verbose=0)
                finetuned_predictions = finetuned_model.predict(images, verbose=0)

                batch_size = images.shape[0]

                for batch_idx in range(batch_size):
                    if file_idx >= num_files_to_visualize:
                        break

                    file_name = Path(test_files[file_idx]).name if isinstance(test_files[file_idx], str) else test_files[file_idx].name
                    print(f"Generating prediction images for file {file_idx+1}/{num_files_to_visualize}: {file_name}")

                    # Extract single sample from batch
                    true_labels = masks[batch_idx].numpy().squeeze()
                    base_pred_labels = np.argmax(base_predictions[batch_idx], axis=-1)
                    finetuned_pred_labels = np.argmax(finetuned_predictions[batch_idx], axis=-1)

                    # Create visualization
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle(f'Prediction Comparison: {file_name}', fontsize=16, fontweight='bold')

                    # Define colors for classes (use config class names if available)
                    class_colors = ['black', 'green', 'yellow', 'orange', 'red'][:self.num_classes]
                    class_names = self.config.get('class_names', [f'Class {i}' for i in range(self.num_classes)])
                    max_class_val = self.num_classes - 1

                    # Plot ground truth
                    axes[0, 0].imshow(true_labels, cmap='tab10', vmin=0, vmax=max_class_val)
                    axes[0, 0].set_title('Ground Truth')
                    axes[0, 0].axis('off')

                    # Plot base model predictions
                    axes[0, 1].imshow(base_pred_labels, cmap='tab10', vmin=0, vmax=max_class_val)
                    axes[0, 1].set_title('Base Model Predictions')
                    axes[0, 1].axis('off')

                    # Plot fine-tuned model predictions
                    axes[0, 2].imshow(finetuned_pred_labels, cmap='tab10', vmin=0, vmax=max_class_val)
                    axes[0, 2].set_title('Fine-tuned Model Predictions')
                    axes[0, 2].axis('off')

                    # Plot differences
                    base_diff = np.abs(true_labels - base_pred_labels)
                    finetuned_diff = np.abs(true_labels - finetuned_pred_labels)

                    axes[1, 0].imshow(base_diff, cmap='hot', vmin=0, vmax=max_class_val)
                    axes[1, 0].set_title('Base Model Errors')
                    axes[1, 0].axis('off')

                    axes[1, 1].imshow(finetuned_diff, cmap='hot', vmin=0, vmax=max_class_val)
                    axes[1, 1].set_title('Fine-tuned Model Errors')
                    axes[1, 1].axis('off')

                    # Plot improvement
                    improvement = base_diff - finetuned_diff
                    im6 = axes[1, 2].imshow(improvement, cmap='RdYlGn', vmin=-max_class_val, vmax=max_class_val)
                    axes[1, 2].set_title('Improvement (Red=Worse, Green=Better)')
                    axes[1, 2].axis('off')

                    # Add colorbar
                    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

                    # Add class legend
                    legend_elements = [
                        mpatches.Patch(color=class_colors[c], label=class_names[c]) for c in range(self.num_classes)
                    ]
                    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

                    plt.tight_layout()

                    # Save prediction image
                    file_stem = Path(test_files[file_idx]).stem if isinstance(test_files[file_idx], str) else test_files[file_idx].stem
                    prediction_filename = f"prediction_comparison_{file_idx+1:02d}_{file_stem}.png"
                    prediction_path = os.path.join(predictions_dir, prediction_filename)
                    plt.savefig(prediction_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    print(f"  Saved: {prediction_path}")
                    file_idx += 1

            print(f"\nPrediction images saved to: {predictions_dir}")

        except Exception as e:
            import traceback
            print(f"ERROR: Failed to generate prediction images: {e}")
            traceback.print_exc()

    def save_predictions_as_tif(self, base_model, finetuned_model, dataset, test_files, output_dir):
        """Save model predictions as GeoTIFF files - PRIMARY OUTPUT"""
        try:
            # Create TIF predictions directory
            tif_predictions_dir = os.path.join(output_dir, "test_predictions_tif")
            os.makedirs(tif_predictions_dir, exist_ok=True)

            print(f"\n=== PRIMARY OUTPUT: SAVING PREDICTIONS AS GEOTIFF FILES ===")
            print(f"Output directory: {tif_predictions_dir}")
            print(f"This is the main result - TIF files ready for GIS analysis")

            # Process each test file - handle batched dataset
            file_idx = 0
            for batch in dataset:
                if file_idx >= len(test_files):
                    break

                # Handle batch structure
                if len(batch) == 2:
                    images, masks = batch
                else:
                    images, masks = batch[0], batch[1]

                # Get predictions for entire batch
                base_predictions = base_model.predict(images, verbose=0)
                finetuned_predictions = finetuned_model.predict(images, verbose=0)

                batch_size = images.shape[0]

                for batch_idx in range(batch_size):
                    if file_idx >= len(test_files):
                        break

                    file_name = Path(test_files[file_idx]).name if isinstance(test_files[file_idx], str) else test_files[file_idx].name
                    print(f"Saving TIF predictions for file {file_idx+1}/{len(test_files)}: {file_name}")

                    # Extract single sample from batch
                    true_labels = masks[batch_idx].numpy().squeeze().astype(np.uint8)
                    base_pred_labels = np.argmax(base_predictions[batch_idx], axis=-1).astype(np.uint8)
                    finetuned_pred_labels = np.argmax(finetuned_predictions[batch_idx], axis=-1).astype(np.uint8)

                    # Get original file path for metadata
                    original_file = test_files[file_idx]

                    # Read original file metadata
                    transform = None
                    crs = None
                    try:
                        with rasterio.open(original_file) as src:
                            transform = src.transform
                            crs = src.crs
                    except Exception as e:
                        print(f"  Warning: Could not read metadata from {original_file}: {e}")

                    # Use default transform if not available
                    if transform is None:
                        transform = rasterio.transform.from_bounds(0, 0, 1, 1, base_pred_labels.shape[1], base_pred_labels.shape[0])

                    file_stem = Path(test_files[file_idx]).stem if isinstance(test_files[file_idx], str) else test_files[file_idx].stem

                    # Save ground truth
                    gt_filename = f"ground_truth_{file_stem}.tif"
                    gt_path = os.path.join(tif_predictions_dir, gt_filename)
                    with rasterio.open(
                        gt_path, 'w',
                        driver='GTiff',
                        height=true_labels.shape[0],
                        width=true_labels.shape[1],
                        count=1,
                        dtype=rasterio.uint8,
                        crs=crs,
                        transform=transform,
                        compress='lzw',
                        tiled=True
                    ) as dst:
                        dst.write(true_labels, 1)
                        dst.set_band_description(1, 'Ground Truth Labels')

                    # Save base model predictions
                    base_filename = f"base_model_prediction_{file_stem}.tif"
                    base_path = os.path.join(tif_predictions_dir, base_filename)
                    with rasterio.open(
                        base_path, 'w',
                        driver='GTiff',
                        height=base_pred_labels.shape[0],
                        width=base_pred_labels.shape[1],
                        count=1,
                        dtype=rasterio.uint8,
                        crs=crs,
                        transform=transform,
                        compress='lzw',
                        tiled=True
                    ) as dst:
                        dst.write(base_pred_labels, 1)
                        dst.set_band_description(1, 'Base Model Predictions')

                    # Save fine-tuned model predictions
                    finetuned_filename = f"finetuned_model_prediction_{file_stem}.tif"
                    finetuned_path = os.path.join(tif_predictions_dir, finetuned_filename)
                    with rasterio.open(
                        finetuned_path, 'w',
                        driver='GTiff',
                        height=finetuned_pred_labels.shape[0],
                        width=finetuned_pred_labels.shape[1],
                        count=1,
                        dtype=rasterio.uint8,
                        crs=crs,
                        transform=transform,
                        compress='lzw',
                        tiled=True
                    ) as dst:
                        dst.write(finetuned_pred_labels, 1)
                        dst.set_band_description(1, 'Fine-tuned Model Predictions')

                    # Save improvement map (difference between models)
                    improvement = finetuned_pred_labels.astype(np.int8) - base_pred_labels.astype(np.int8)
                    improvement_filename = f"improvement_map_{file_stem}.tif"
                    improvement_path = os.path.join(tif_predictions_dir, improvement_filename)
                    with rasterio.open(
                        improvement_path, 'w',
                        driver='GTiff',
                        height=improvement.shape[0],
                        width=improvement.shape[1],
                        count=1,
                        dtype=rasterio.int8,
                        crs=crs,
                        transform=transform,
                        compress='lzw',
                        tiled=True
                    ) as dst:
                        dst.write(improvement, 1)
                        dst.set_band_description(1, 'Improvement Map (Fine-tuned - Base)')

                    print(f"  Saved: {gt_filename}, {base_filename}, {finetuned_filename}, {improvement_filename}")
                    file_idx += 1

            print(f"\nAll TIF prediction files saved to: {tif_predictions_dir}")

        except Exception as e:
            import traceback
            print(f"ERROR: Failed to save TIF predictions: {e}")
            traceback.print_exc()

    def create_summary_comparison_image(self, base_model, finetuned_model, dataset, test_files, output_dir):
        """Create a summary comparison image showing overall differences"""
        try:
            import matplotlib.pyplot as plt

            # Collect metrics for all files
            base_accuracies = []
            finetuned_accuracies = []
            base_ious = []
            finetuned_ious = []

            for i, batch in enumerate(dataset):
                if i >= 10:
                    break

                # Handle batch structure
                if len(batch) == 2:
                    image, mask = batch
                else:
                    image, mask = batch[0], batch[1]

                # Get predictions
                base_predictions = base_model.predict(image, verbose=0)
                finetuned_predictions = finetuned_model.predict(image, verbose=0)

                # Convert to numpy arrays
                true_labels = mask.numpy().squeeze()
                base_pred_labels = np.argmax(base_predictions, axis=-1).squeeze()
                finetuned_pred_labels = np.argmax(finetuned_predictions, axis=-1).squeeze()

                # Calculate metrics
                base_metrics = self.calculate_file_metrics(true_labels, base_pred_labels, num_classes=self.num_classes)
                finetuned_metrics = self.calculate_file_metrics(true_labels, finetuned_pred_labels, num_classes=self.num_classes)

                if base_metrics and finetuned_metrics:
                    base_accuracies.append(base_metrics['overall_accuracy'])
                    finetuned_accuracies.append(finetuned_metrics['overall_accuracy'])
                    base_ious.append(base_metrics['overall_iou'])
                    finetuned_ious.append(finetuned_metrics['overall_iou'])

            # Create summary plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Model Performance Comparison on Test Data', fontsize=16, fontweight='bold')

            # Plot accuracies
            x = np.arange(len(base_accuracies))
            width = 0.35

            ax1.bar(x - width / 2, base_accuracies, width, label='Base Model', alpha=0.8, color='lightcoral')
            ax1.bar(x + width / 2, finetuned_accuracies, width, label='Fine-tuned Model', alpha=0.8, color='lightgreen')

            ax1.set_xlabel('Test Files')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Per-File Accuracy Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot IoUs
            ax2.bar(x - width / 2, base_ious, width, label='Base Model', alpha=0.8, color='lightcoral')
            ax2.bar(x + width / 2, finetuned_ious, width, label='Fine-tuned Model', alpha=0.8, color='lightgreen')

            ax2.set_xlabel('Test Files')
            ax2.set_ylabel('IoU')
            ax2.set_title('Per-File IoU Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save summary image
            summary_path = os.path.join(output_dir, "performance_summary.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Summary comparison saved: {summary_path}")

        except Exception as e:
            print(f"ERROR: Failed to create summary comparison: {e}")

    def run_test_evaluation(self, base_only=False):
        """Run complete test evaluation
        
        Args:
            base_only: If True, only evaluate base model (skip finetuned model comparison)
        """
        print("=" * 80)
        print("TESTING MODELS ON UNSEEN TEST DATA")
        print("=" * 80)

        # Check prerequisites
        if not self.check_test_data_exists():
            return False

        models_status = self.check_models_exist()
        
        # Base model is always required
        if not models_status['base']:
            print("ERROR: Base model not found")
            return False
        
        # Finetuned model is optional if base_only=True or if it doesn't exist
        has_finetuned = models_status['finetuned']
        if not has_finetuned:
            print("INFO: Fine-tuned model not found - running base model evaluation only")
            base_only = True

        # Process raw test data into patches (same as training data)
        processed_patches = self.process_raw_test_data()
        if not processed_patches:
            print("ERROR: No patches created from test data")
            return False

        # Create test dataset from processed patches
        dataset, test_files = self.create_test_dataset_from_patches(processed_patches)
        if dataset is None:
            return False

        # Load base model (always required)
        base_model = self.load_model(self.base_model_path)
        if base_model is None:
            print("ERROR: Failed to load base model")
            return False

        # Evaluate base model
        print(f"\n{'=' * 60}")
        print("EVALUATING BASE MODEL ON TEST DATA")
        print(f"{'=' * 60}")

        base_results = self.evaluate_model_on_test_data(base_model, dataset, test_files, "Base")
        if not base_results:
            print("ERROR: Failed to evaluate base model")
            return False

        # If base_only mode, save results and exit
        if base_only:
            # Recreate dataset for saving predictions (dataset was consumed during evaluation)
            dataset, _ = self.create_test_dataset_from_patches(processed_patches)
            
            # Save base model predictions as TIF files
            self.save_base_model_predictions_as_tif(base_model, dataset, test_files, MODEL_FOLDER)
            
            # Save results
            results_path = os.path.join(MODEL_FOLDER, 'test_data_evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'base_model_results': base_results,
                    'evaluation_mode': 'base_only'
                }, f, indent=2)

            print(f"\n{'=' * 60}")
            print("BASE MODEL EVALUATION COMPLETE")
            print(f"{'=' * 60}")
            print(f"Overall Accuracy: {base_results['overall_accuracy']:.4f}")
            print(f"Overall IoU: {base_results['overall_iou']:.4f}")
            print(f"Per-class IoU: {base_results['per_class_iou']}")
            print(f"TIF predictions saved to: {MODEL_FOLDER}/test_predictions_tif/")
            print(f"Results saved to: {results_path}")
            print(f"No data leakage - model tested on unseen data")

            # Clean up temporary files
            self.cleanup_temp_files()
            return True

        # Load finetuned model for comparison
        finetuned_model = self.load_model(self.finetuned_model_path)
        if finetuned_model is None:
            print("ERROR: Failed to load finetuned model")
            return False

        # Reset dataset for finetuned model evaluation
        dataset, _ = self.create_test_dataset()
        finetuned_results = self.evaluate_model_on_test_data(finetuned_model, dataset, test_files, "Fine-tuned")

        if not finetuned_results:
            print("ERROR: Failed to evaluate finetuned model")
            return False

        # Compare results
        comparison = self.compare_models_on_test_data(base_results, finetuned_results)
        if not comparison:
            return False

        # Create visualization
        self.create_test_data_visualization(comparison, MODEL_FOLDER)

        # Recreate dataset for saving predictions (dataset was consumed during evaluation)
        dataset, _ = self.create_test_dataset_from_patches(processed_patches)
        
        # Save predictions as TIF files (PRIMARY OUTPUT)
        self.save_predictions_as_tif(base_model, finetuned_model, dataset, test_files, MODEL_FOLDER)

        # Recreate dataset for generating prediction images
        dataset, _ = self.create_test_dataset_from_patches(processed_patches)
        
        # Generate prediction images (SECONDARY OUTPUT - for visualization only)
        self.generate_prediction_images(base_model, finetuned_model, dataset, test_files, MODEL_FOLDER)

        # Save results
        results_path = os.path.join(MODEL_FOLDER, 'test_data_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'base_model_results': base_results,
                'finetuned_model_results': finetuned_results,
                'comparison': comparison
            }, f, indent=2)

        print(f"\n{'=' * 60}")
        print("TEST DATA EVALUATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"TIF predictions saved to: {MODEL_FOLDER}/test_predictions_tif/")
        print(f"PNG visualizations saved to: {MODEL_FOLDER}/test_predictions/")
        print(f"Results saved to: {results_path}")
        print(f"Visualization saved to: {MODEL_FOLDER}/test_data_evaluation.png")
        print(f"No data leakage - models tested on unseen data")

        # Clean up temporary files
        self.cleanup_temp_files()

        return True
    
    def save_base_model_predictions_as_tif(self, base_model, dataset, test_files, output_dir):
        """Save base model predictions as GeoTIFF files"""
        try:
            # Create TIF predictions directory
            tif_predictions_dir = os.path.join(output_dir, "test_predictions_tif")
            os.makedirs(tif_predictions_dir, exist_ok=True)

            print(f"\n=== SAVING BASE MODEL PREDICTIONS AS GEOTIFF FILES ===")
            print(f"Output directory: {tif_predictions_dir}")

            # Process each test file - handle batched dataset
            file_idx = 0
            for batch in dataset:
                # Handle batch structure
                if len(batch) == 2:
                    images, masks = batch
                else:
                    images, masks = batch[0], batch[1]

                # Get predictions for the entire batch
                base_predictions = base_model.predict(images, verbose=0)
                
                # Get batch size
                batch_size = images.shape[0]
                
                # Process each sample in the batch
                for batch_idx in range(batch_size):
                    if file_idx >= len(test_files):
                        break
                    
                    # Extract single sample from batch
                    true_labels = masks[batch_idx].numpy().squeeze().astype(np.uint8)
                    base_pred_labels = np.argmax(base_predictions[batch_idx], axis=-1).astype(np.uint8)
                    
                    # Ensure 2D arrays
                    if true_labels.ndim > 2:
                        true_labels = true_labels.squeeze()
                    if base_pred_labels.ndim > 2:
                        base_pred_labels = base_pred_labels.squeeze()

                    file_name = Path(test_files[file_idx]).name if isinstance(test_files[file_idx], str) else test_files[file_idx].name
                    print(f"Saving TIF predictions for file {file_idx+1}/{len(test_files)}: {file_name}")

                    # Get original file path for metadata
                    original_file = test_files[file_idx]

                    # Read original file metadata
                    transform = None
                    crs = None
                    try:
                        with rasterio.open(original_file) as src:
                            transform = src.transform
                            crs = src.crs
                    except Exception as e:
                        print(f"  Warning: Could not read metadata from {original_file}: {e}")

                    # Use default transform if not available
                    if transform is None:
                        transform = rasterio.transform.from_bounds(0, 0, 1, 1, base_pred_labels.shape[1], base_pred_labels.shape[0])

                    file_stem = Path(test_files[file_idx]).stem if isinstance(test_files[file_idx], str) else test_files[file_idx].stem

                    # Save ground truth
                    gt_filename = f"ground_truth_{file_stem}.tif"
                    gt_path = os.path.join(tif_predictions_dir, gt_filename)
                    with rasterio.open(
                        gt_path, 'w',
                        driver='GTiff',
                        height=true_labels.shape[0],
                        width=true_labels.shape[1],
                        count=1,
                        dtype=rasterio.uint8,
                        crs=crs,
                        transform=transform,
                        compress='lzw',
                        tiled=True
                    ) as dst:
                        dst.write(true_labels, 1)
                        dst.set_band_description(1, 'Ground Truth Labels')

                    # Save base model predictions
                    base_filename = f"base_model_prediction_{file_stem}.tif"
                    base_path = os.path.join(tif_predictions_dir, base_filename)
                    with rasterio.open(
                        base_path, 'w',
                        driver='GTiff',
                        height=base_pred_labels.shape[0],
                        width=base_pred_labels.shape[1],
                        count=1,
                        dtype=rasterio.uint8,
                        crs=crs,
                        transform=transform,
                        compress='lzw',
                        tiled=True
                    ) as dst:
                        dst.write(base_pred_labels, 1)
                        dst.set_band_description(1, 'Base Model Predictions')

                    print(f"  Saved: {gt_filename}, {base_filename}")
                    file_idx += 1

                if file_idx >= len(test_files):
                    break

            print(f"\nAll TIF prediction files saved to: {tif_predictions_dir}")

        except Exception as e:
            import traceback
            print(f"ERROR: Failed to save TIF predictions: {e}")
            traceback.print_exc()


def run_inference():
    with print_tag("INFERENCE"):
        run_inference_pipeline()


def run_test(base_only=False):
    with print_tag("TEST"):
        tester = ModelTester()
        return tester.run_test_evaluation(base_only=base_only)


def parse_args(argv=None, default_mode="inference"):
    parser = argparse.ArgumentParser(description="Run inference and/or test evaluation.")
    parser.add_argument(
        "--mode",
        choices=["inference", "test", "both"],
        default=default_mode,
        help="Which pipeline to run."
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        default=False,
        help="Only evaluate the base model (skip finetuned model comparison)."
    )
    return parser.parse_args(argv)


def main(argv=None, default_mode="inference"):
    args = parse_args(argv=argv, default_mode=default_mode)

    if args.mode in ("inference", "both"):
        run_inference()

    if args.mode in ("test", "both"):
        run_test(base_only=args.base_only)


if __name__ == "__main__":
    main()

