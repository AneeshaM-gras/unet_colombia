# src/patch_utils.py


def center_crop_array(data, target_size):
    """
    Center-crop an array shaped (bands, height, width) to target_size x target_size.
    Returns (cropped_data, row_offset, col_offset).
    """
    _, height, width = data.shape
    if height < target_size or width < target_size:
        raise ValueError(
            f"Cannot center crop {height}x{width} to {target_size}x{target_size}: "
            "patch is smaller than target size"
        )

    if height == target_size and width == target_size:
        return data, 0, 0

    row_offset = (height - target_size) // 2
    col_offset = (width - target_size) // 2
    cropped = data[
        :,
        row_offset:row_offset + target_size,
        col_offset:col_offset + target_size,
    ]
    return cropped, row_offset, col_offset
