# src/data_preprocessing.py

import os
import rasterio
from rasterio.merge import merge
from patchify import patchify
import numpy as np
import shutil

def merge_geotiffs_in_folder(source_folder, output_path, search_pattern=".tif"):
    """
    Merges multiple GeoTIFF files from a source folder into a single mosaic,
    PRESERVING THE BAND DESCRIPTIONS (NAMES).
    """
    src_files_to_mosaic = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(search_pattern)]
    
    if not src_files_to_mosaic:
        print(f"INFO: No files found in '{source_folder}'. Skipping merge.")
        return False

    print(f"Merging {len(src_files_to_mosaic)} files from '{os.path.basename(source_folder)}'...")

    srcs = [rasterio.open(fp) for fp in src_files_to_mosaic]

    try:
        source_band_descriptions = list(srcs[0].descriptions)
        if not all(source_band_descriptions):
            print(f"WARNING: A source file in '{os.path.basename(source_folder)}' has missing band names. Using what's available.")

        # The merge function returns a single array and the affine transform
        mosaic, out_trans = merge(srcs, method='first')

        # Copy the metadata from the first source file
        out_meta = srcs[0].meta.copy()

        # Update the metadata with the dimensions and transform of the merged mosaic
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        })

        # Write the mosaic to a new GeoTIFF file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
            dest.descriptions = source_band_descriptions

        print(f"-> Saved merged file with band names to '{output_path}'")
        return True

    finally:
        for src in srcs:
            src.close()

def create_geotiff_patches(image_path, output_folder, patch_size=256, stride=256):
    """
    Creates patches from a GeoTIFF image and saves them,
    CRITICALLY PRESERVING THE BAND DESCRIPTIONS (NAMES).
    """

    print(f"Creating patches for '{os.path.basename(image_path)}'...")
    try:
        with rasterio.open(image_path) as src:
            # --- FIX: Read the band descriptions from the source file ---
            source_band_descriptions = list(src.descriptions)
            
            # Verify that the source file actually has band names
            if not all(source_band_descriptions):
                print(f"WARNING: Source file '{os.path.basename(image_path)}' has missing band names. Got: {source_band_descriptions}")
                # You might want to raise an error here depending on your workflow's strictness
                # raise ValueError("Source file is missing band descriptions.")

            image = src.read()
            source_filename_base = os.path.splitext(os.path.basename(image_path))[0]
            
            image = np.moveaxis(image, 0, -1)
            patches = patchify(image, (patch_size, patch_size, image.shape[2]), step=stride)
            
            num_patches = 0
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0, :, :, :]
                    patch = np.moveaxis(patch, -1, 0)

                    transform = src.transform * rasterio.Affine.translation(j * stride, i * stride)
                    
                    output_filename = os.path.join(output_folder, f"{source_filename_base}_patch_{i}_{j}.tif")
                    
                    # Open the new patch file for writing
                    with rasterio.open(
                        output_filename, 'w', 
                        driver='GTiff',
                        height=patch.shape[1], 
                        width=patch.shape[2],
                        count=patch.shape[0], 
                        dtype=patch.dtype,
                        crs=src.crs, 
                        transform=transform,
                    ) as dst:
                        # Write the patch data
                        dst.write(patch)
                        
                        # --- FIX: Set the band descriptions for the new patch file ---
                        dst.descriptions = source_band_descriptions
                        
                    num_patches += 1
            print(f"-> Saved {num_patches} patches with band names to '{output_folder}'")
            
    except Exception as e:
        print(f"ERROR: Could not process {image_path}. Reason: {e}")


# def process_and_patch_all_subfolders(root_input_dir, merged_output_dir, patches_output_dir):
#     """
#     Orchestrates the entire workflow:
#     1. Iterates through subfolders in root_input_dir.
#     2. Merges GeoTIFFs in each subfolder.
#     3. Creates patches from each merged GeoTIFF.
#     """

#     # Create the output directories if they don't exist
#     os.makedirs(merged_output_dir, exist_ok=True)
#     os.makedirs(patches_output_dir, exist_ok=True)
    
#     print(f"Starting processing for subfolders in: {root_input_dir}")
#     print("---")

#     # Get a list of all items in the root directory that are directories
#     subfolders = [f.path for f in os.scandir(root_input_dir) if f.is_dir()]
    
#     if not subfolders:
#         print("No subfolders found. Exiting.")
#         return

#     for folder_path in subfolders:
#         folder_name = os.path.basename(folder_path)
        
#         # Define the output path for the merged file for this subfolder
#         merged_file_path = os.path.join(merged_output_dir, f"{folder_name}_merged.tif")

#         # --- MERGE STEP ---
#         # Merge all .tif files within the current subfolder
#         success = merge_geotiffs_in_folder(folder_path, merged_file_path)

#         # --- PATCHING STEP ---
#         # If merging was successful, create patches from the new merged file
#         if success:
#             create_geotiff_patches(
#                 image_path=merged_file_path,
#                 output_folder=patches_output_dir,
#                 patch_size=256,
#                 stride=192  # Non-overlapping patches
#             )
#         print("---")
            
#     print("All subfolders processed successfully!")
#     print(f"All merged tiles are in: {merged_output_dir}")
#     print(f"All training patches are in: {patches_output_dir}")

def run_merge_step(root_input_dir, merged_output_dir, force_rerun=False):
    """
    Orchestrates ONLY the merging of GeoTIFFs from subfolders.
    Skips if output files already exist, unless force_rerun is True.
    """
    os.makedirs(merged_output_dir, exist_ok=True)
    print(f"--- Running Merge Step ---")
    subfolders = [f.path for f in os.scandir(root_input_dir) if f.is_dir()]
    if not subfolders:
        print("No subfolders found. Exiting merge step.")
        return

    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        merged_file_path = os.path.join(merged_output_dir, f"{folder_name}_merged.tif")

        if os.path.exists(merged_file_path) and not force_rerun:
            print(f"INFO: Merged file '{os.path.basename(merged_file_path)}' already exists. Skipping.")
            continue

        merge_geotiffs_in_folder(folder_path, merged_file_path)

def run_patch_step(merged_input_dir, patches_output_dir, patch_size=256, stride=192, force_rerun=False):
    """
    Orchestrates ONLY the creation of patches from already merged GeoTIFFs.
    If force_rerun is True, it will delete existing patches and regenerate them.
    Otherwise, it will only run if the patches_output_dir is empty.
    """
    print(f"\n--- Running Patching Step ---")

    # --- START OF FIX ---
    
    # Check if the output directory exists and has files
    output_exists = os.path.exists(patches_output_dir) and os.listdir(patches_output_dir)

    if output_exists and force_rerun:
        print(f"INFO: `force_rerun` is True. Deleting existing patches in '{patches_output_dir}'...")
        # A safe way to clear a directory
        shutil.rmtree(patches_output_dir)
        os.makedirs(patches_output_dir) # Recreate the empty directory
    elif output_exists and not force_rerun:
        print(f"INFO: Patches already exist in '{patches_output_dir}' and `force_rerun` is False. Skipping.")
        return # Skip the patching process as intended
    
    # If we reach here, it's because the directory was empty or was just cleared.
    # We still need to ensure the directory exists before proceeding.
    os.makedirs(patches_output_dir, exist_ok=True)
    
    # --- END OF FIX ---

    merged_files = [os.path.join(merged_input_dir, f) for f in os.listdir(merged_input_dir) if f.endswith('.tif')]
    if not merged_files:
        print("WARNING: No merged files found to patch. Cannot create patches.")
        return
    
    print(f"Generating patches from {len(merged_files)} merged tiles...")
    for merged_file_path in merged_files:
        create_geotiff_patches(
            image_path=merged_file_path,
            output_folder=patches_output_dir,
            patch_size=patch_size,
            stride=stride
        )


def create_inference_patches(image_path, output_folder, image_band_names, patch_size=256, stride=256):
    """
    Creates patches from a GeoTIFF for inference, saving ONLY the image bands.
    This function specifically excludes the label band.
    """
    print(f"Creating INFERENCE patches for '{os.path.basename(image_path)}'...")
    try:
        with rasterio.open(image_path) as src:
            all_bands_in_file = list(src.descriptions)
            
            # Find the indices of ONLY the image bands
            try:
                image_band_indices = [all_bands_in_file.index(name) + 1 for name in image_band_names]
            except ValueError as e:
                raise ValueError(f"A required image band was not found in {image_path}. Details: {e}")

            # Read ONLY the image data, excluding the label
            image = src.read(image_band_indices)
            
            source_filename_base = os.path.splitext(os.path.basename(image_path))[0]
            
            image = np.moveaxis(image, 0, -1)
            # The shape will now be (H, W, 64), which is correct
            patches = patchify(image, (patch_size, patch_size, image.shape[2]), step=stride)
            
            num_patches = 0
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0, :, :, :]
                    patch = np.moveaxis(patch, -1, 0)

                    transform = src.transform * rasterio.Affine.translation(j * stride, i * stride)
                    
                    output_filename = os.path.join(output_folder, f"{source_filename_base}_patch_{i}_{j}.tif")
                    
                    with rasterio.open(
                        output_filename, 'w', 
                        driver='GTiff',
                        height=patch.shape[1], 
                        width=patch.shape[2],
                        count=patch.shape[0], # This will be 64
                        dtype=patch.dtype,
                        crs=src.crs, 
                        transform=transform,
                    ) as dst:
                        dst.write(patch)
                        # Set the band descriptions for the new 64-band patch file
                        dst.descriptions = image_band_names
                        
                    num_patches += 1
            print(f"-> Saved {num_patches} INFERENCE patches (64 bands each) to '{output_folder}'")
            
    except Exception as e:
        print(f"ERROR: Could not process {image_path} for inference. Reason: {e}")