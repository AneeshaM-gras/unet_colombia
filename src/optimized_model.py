# src/optimized_model.py
"""
U-Net Model Architecture

This module contains the U-Net model implementation used for training and hyperparameter optimization.
The model follows the standard 3-level U-Net architecture from the original paper.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model

def conv_block_optimized(input_tensor, num_filters, dropout_rate=0.2, use_batch_norm=True):
    """
    Optimized convolution block with enhanced regularization
    """
    # First convolution
    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(input_tensor)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    
    # Second convolution
    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x

def unet(input_shape, n_classes, dropout_encoder=0.1, dropout_decoder=0.2):
    """
    Standard 3-level U-Net architecture matching the original U-Net paper.
    
    Architecture:
    - Encoder: 3 downsampling levels (64 -> 128 -> 256 -> 1024 bottleneck)
    - Decoder: 3 upsampling levels with skip connections (1024 -> 512 -> 256 -> 128 -> 64)
    - Each level: 2 conv blocks (3x3 conv + BN + ReLU) x2
    - Skip connections: Concatenate encoder features with decoder features
    
    This matches the standard U-Net architecture for 256x256 input images.
    
    Args:
        input_shape: Shape of input images (H, W, C)
        n_classes: Number of output classes
        dropout_encoder: Dropout rate in encoder path (default: 0.1)
        dropout_decoder: Dropout rate in decoder path (default: 0.2)
    
    Returns:
        Keras Model with U-Net architecture
    """
    inputs = Input(shape=input_shape)

    # ========== ENCODER PATH (Contracting Path) ==========
    # Level 1: 256x256 -> 128x128
    c1 = conv_block_optimized(inputs, 64, dropout_rate=dropout_encoder)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout_encoder)(p1)

    # Level 2: 128x128 -> 64x64
    c2 = conv_block_optimized(p1, 128, dropout_rate=dropout_encoder)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout_encoder)(p2)

    # Level 3: 64x64 -> 32x32
    c3 = conv_block_optimized(p2, 256, dropout_rate=dropout_encoder)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout_encoder)(p3)

    # ========== BOTTLENECK ==========
    # 32x32 -> 32x32
    b = conv_block_optimized(p3, 1024, dropout_rate=dropout_encoder)

    # ========== DECODER PATH (Expansive Path) ==========
    # Level 1: 32x32 -> 64x64
    u1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(b)
    u1 = concatenate([u1, c3])  # Skip connection
    u1 = Dropout(dropout_decoder)(u1)
    c4 = conv_block_optimized(u1, 512, dropout_rate=dropout_decoder)

    # Level 2: 64x64 -> 128x128
    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c4)
    u2 = concatenate([u2, c2])  # Skip connection
    u2 = Dropout(dropout_decoder)(u2)
    c5 = conv_block_optimized(u2, 128, dropout_rate=dropout_decoder)

    # Level 3: 128x128 -> 256x256
    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = concatenate([u3, c1])  # Skip connection
    u3 = Dropout(dropout_decoder)(u3)
    c6 = conv_block_optimized(u3, 64, dropout_rate=dropout_decoder)

    # ========== OUTPUT LAYER ==========
    activation = 'softmax' if n_classes > 1 else 'sigmoid'
    outputs = Conv2D(n_classes, (1, 1), activation=activation, name="output")(c6)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model
