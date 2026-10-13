# src/utils.py

import os
import rasterio
import numpy as np
from tqdm import tqdm
from patchify import patchify
from osgeo import gdal


def stitch_predictions(patches_folder, original_tile_path, output_path):
    """
    Stitches predicted patches back together into a single GeoTIFF.

    Args:
        patches_folder (str): Folder containing the predicted patches.
        original_tile_path (str): Path to the original, large GeoTIFF tile that was patched.
                                  Used to get georeferencing information.
        output_path (str): Path to save the final stitched prediction map.
    """
    print(f"Stitching predictions for {os.path.basename(original_tile_path)}...")
    
    # Use gdal to build a virtual raster (VRT). This is memory-efficient.
    patch_files = [os.path.join(patches_folder, f) for f in os.listdir(patches_folder) if f.endswith('.tif')]
    if not patch_files:
        print("WARNING: No patches found to stitch.")
        return

    vrt_path = os.path.join(os.path.dirname(output_path), "temp_stitched.vrt")
    gdal.BuildVRT(vrt_path, patch_files)
    
    # Use the VRT to create the final GeoTIFF, preserving the original's georeferencing.
    with rasterio.open(vrt_path) as src:
        vrt_data = src.read(1)
        vrt_profile = src.profile

    with rasterio.open(original_tile_path) as orig_src:
        orig_profile = orig_src.profile
    
    # Update profile for a single-band uint8 classification map
    orig_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'width': vrt_profile['width'],
        'height': vrt_profile['height'],
        'transform': vrt_profile['transform']
    })
    
    with rasterio.open(output_path, 'w', **orig_profile) as dst:
        dst.write(vrt_data.astype(np.uint8), 1)
        
    os.remove(vrt_path) # Clean up the temporary VRT file
    print(f"-> Successfully stitched prediction saved to: {output_path}")

def validate_dataset_labels(patches_folder, label_band_name):
    """
    Scans all patch files to find all unique label values and their total pixel counts.

    Returns:
        dict: A dictionary where keys are the unique class IDs and values are their
              total pixel counts across the entire dataset. Returns None on error.
    """
    if not os.path.exists(patches_folder):
        print(f"ERROR: Patches folder not found at '{patches_folder}'")
        return None

    all_files = [os.path.join(patches_folder, f) for f in os.listdir(patches_folder) if f.lower().endswith(('.tif', '.tiff'))]

    if not all_files:
        print(f"WARNING: No .tif files found in '{patches_folder}'")
        return None

    master_class_counts = {}
    print(f"Inspecting {len(all_files)} patch files to find all unique label values and their counts...")

    for patch_file in tqdm(all_files, desc="Validating Labels"):
        try:
            with rasterio.open(patch_file) as src:
                all_bands_in_file = list(src.descriptions)
                label_band_index = all_bands_in_file.index(label_band_name) + 1
                mask_data = src.read(label_band_index)
                
                # Get unique classes and their counts in the current file
                unique_in_file, counts_in_file = np.unique(mask_data, return_counts=True)
                
                # Aggregate the counts into the master dictionary
                for class_id, count in zip(unique_in_file, counts_in_file):
                    # Ensure the key is a standard integer, not a numpy type
                    class_id = int(class_id)
                    master_class_counts[class_id] = master_class_counts.get(class_id, 0) + count

        except Exception as e:
            print(f"\nCould not process file {os.path.basename(patch_file)}: {e}")

    # --- Final Report ---
    print("\n--- DATASET VALIDATION REPORT ---")
    if not master_class_counts:
        print("WARNING: No label values were found. The dataset may be empty or the label band name is incorrect.")
        return None

    # Sort the dictionary by class ID for clean and predictable printing
    sorted_counts = dict(sorted(master_class_counts.items()))
    
    print("Found the following raw class labels and their total pixel counts:")
    for class_id, count in sorted_counts.items():
        print(f"  - Class {class_id}: {count:,} pixels")
    print("-----------------------------------")
    
    # Return the dictionary of counts, which is what the new workflow requires
    return sorted_counts


def predict_with_overlap(model, input_geotiff_path, output_geotiff_path, image_band_names, patch_size=256, stride=192):
    """
    Performs prediction on a single GeoTIFF using an overlapping sliding window.
    """
    print(f"\nStarting prediction for: {os.path.basename(input_geotiff_path)}")
    
    with rasterio.open(input_geotiff_path) as src:
        meta = src.meta.copy()
        all_bands_in_file = list(src.descriptions)
        try:
            image_band_indices = [all_bands_in_file.index(name) + 1 for name in image_band_names]
        except ValueError as e:
            print(f"ERROR: A required band was not found in {input_geotiff_path}. Details: {e}")
            return

        image = src.read(image_band_indices)
        image = np.moveaxis(image, 0, -1)

    image = np.nan_to_num(image, nan=0.0)
    image = image.astype(np.float32) / 255.0

    img_h, img_w, num_channels = image.shape
    pad_h = stride - (img_h - patch_size) % stride if (img_h - patch_size) % stride != 0 else 0
    pad_w = stride - (img_w - patch_size) % stride if (img_w - patch_size) % stride != 0 else 0
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    patches = patchify(padded_image, (patch_size, patch_size, num_channels), step=stride)
    num_classes = model.output_shape[-1]
    padded_prediction = np.zeros(padded_image.shape[:2] + (num_classes,), dtype=np.float32)
    overlap_counter = np.zeros(padded_image.shape[:2], dtype=np.float32)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch_image = patches[i, j, 0, :, :, :]
            predicted_patch_logits = model.predict(np.expand_dims(patch_image, axis=0), verbose=0)[0]
            y_start, x_start = i * stride, j * stride
            y_end, x_end = y_start + patch_size, x_start + patch_size
            padded_prediction[y_start:y_end, x_start:x_end] += predicted_patch_logits
            overlap_counter[y_start:y_end, x_start:x_end] += 1
            
    averaged_prediction_logits = padded_prediction / (np.expand_dims(overlap_counter, axis=-1) + 1e-7)
    final_prediction = np.argmax(averaged_prediction_logits, axis=-1)
    final_prediction = final_prediction[:img_h, :img_w]

    meta.update(count=1, dtype='uint8', driver='GTiff')
    with rasterio.open(output_geotiff_path, 'w', **meta) as dst:
        dst.write(final_prediction.astype(np.uint8), 1)
        
    print(f"-> Successfully saved classification map to: {output_geotiff_path}")


def run_batch_prediction(model, input_folder, output_folder, image_band_names):
    """
    Finds all GeoTIFFs in an input folder and runs prediction on each one.
    """
    print(f"\n--- Starting Batch Prediction ---")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    files_to_predict = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
    
    if not files_to_predict:
        print("WARNING: No GeoTIFF files found in the input folder.")
        return

    print(f"Found {len(files_to_predict)} files to process.")
    
    for filename in files_to_predict:
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace('_merged.tif', '_classified.tif')
        output_path = os.path.join(output_folder, output_filename)
        
        predict_with_overlap(
            model=model,
            input_geotiff_path=input_path,
            output_geotiff_path=output_path,
            image_band_names=image_band_names,
            patch_size=256,
            stride=192
        )

    print("\n--- Batch Prediction Complete ---")
