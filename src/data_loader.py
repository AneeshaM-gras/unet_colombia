# src/data_loader.py

import tensorflow as tf
import numpy as np
import rasterio
import os
from . import config
from .patch_utils import center_crop_array

def load_geotiff_by_name(file_path_tensor, image_band_names_tensor, label_band_name_tensor):
    """
    Loads specific bands from a GeoTIFF file based on their names.
    This version correctly handles EagerTensor inputs from tf.py_function.
    """
    try:
        # --- THE FIX: Use .numpy() to extract the bytes value from the Tensor ---
        file_path = file_path_tensor.numpy().decode('utf-8')
        
        # .numpy() on a list-like tensor returns a numpy array of bytes.
        image_band_names = [name.decode('utf-8') for name in image_band_names_tensor.numpy()]
        
        label_band_name = label_band_name_tensor.numpy().decode('utf-8')
        
        # The rest of the logic remains the same
        with rasterio.open(file_path) as src:
            all_bands_in_file = list(src.descriptions)

            # Check for band names
            for band in image_band_names:
                if band not in all_bands_in_file:
                    raise ValueError(f"Required image band '{band}' not found. Available: {all_bands_in_file}")
            
            if label_band_name not in all_bands_in_file:
                raise ValueError(f"Required label band '{label_band_name}' not found. Available: {all_bands_in_file}")

            # Get indices and read data
            image_band_indices = [all_bands_in_file.index(name) + 1 for name in image_band_names]
            label_band_index = all_bands_in_file.index(label_band_name) + 1

            image_data = src.read(image_band_indices)
            mask_data = src.read(label_band_index)
            
            full_data = np.concatenate([image_data, np.expand_dims(mask_data, axis=0)], axis=0)

            if (
                full_data.shape[1] != config.PATCH_SIZE
                or full_data.shape[2] != config.PATCH_SIZE
            ):
                full_data, _, _ = center_crop_array(full_data, config.PATCH_SIZE)

        return full_data.astype(np.float32)

    except Exception as e:
        # This error handling is now safe, though the file_path is only decoded inside the try block.
        # It's better to report the error concisely.
        raise IOError(f"Failed to process file. Reason: {e}")

def parse_function(file_path, image_band_names, label_band_name, num_image_bands):
    """
    Correct version for this solution. Produces a mask of shape (H, W, 1).
    """
    data = tf.py_function(
        load_geotiff_by_name, 
        [file_path, image_band_names, label_band_name], 
        tf.float32
    )
    
    image = data[:num_image_bands, :, :]
    mask = data[num_image_bands:, :, :]
    
    # Transpose to (height, width, channels)
    image = tf.transpose(image, [1, 2, 0])
    mask = tf.transpose(mask, [1, 2, 0]) # Mask shape is (H, W, 1)

    image = tf.where(tf.math.is_nan(image), 0.0, image)

    # Cast the mask to integer type
    mask = tf.cast(mask, tf.int32)
    
    # --- NEW: ON-THE-FLY CLASS AGGREGATION ---
    if config.CLASS_MAPPING:
        keys = tf.constant(list(config.CLASS_MAPPING.keys()), dtype=tf.int32)
        values = tf.constant(list(config.CLASS_MAPPING.values()), dtype=tf.int32)
        # Map unknown classes to background (0) instead of -1 to avoid invalid indices
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=0  # Map unknown classes to background
        )
        mask = table.lookup(mask)
        # Ensure mask values are non-negative (clip any remaining invalid values)
        mask = tf.maximum(mask, 0)

    target_size = config.PATCH_SIZE
    image.set_shape([target_size, target_size, num_image_bands])
    mask.set_shape([target_size, target_size, 1])

    image = image / 255.0
    
    return image, mask

def create_dataset(file_paths, batch_size, image_bands, label_band):
    """
    Creates a TensorFlow dataset from file paths, configured with band names.
    """
    # Calculate the number of image bands here, as a Python integer
    num_image_bands = len(image_bands)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    # Use a lambda function to "capture" the variables and pass them to the parser
    # This now correctly passes the integer `num_image_bands`
    dataset = dataset.map(
        lambda x: parse_function(x, image_bands, label_band, num_image_bands), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset