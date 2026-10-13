# src/data_augmentation.py

import tensorflow as tf
import numpy as np

def augment_image_and_mask(image, mask):
    """
    Apply the same augmentation to both image and mask
    """
    # Random horizontal flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Random vertical flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    # Random rotation (90, 180, 270 degrees)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    
    # Random brightness adjustment (only for image)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast adjustment (only for image)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    return image, mask

def create_augmented_dataset(dataset, augment_probability=0.7):
    """
    Create an augmented dataset with specified probability
    """
    def augment_fn(image, mask):
        # Apply augmentation with given probability
        should_augment = tf.random.uniform([]) < augment_probability
        
        def augment():
            return augment_image_and_mask(image, mask)
        
        def no_augment():
            return image, mask
        
        return tf.cond(should_augment, augment, no_augment)
    
    return dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

def create_balanced_augmented_dataset(dataset, class_weights, augment_probability=0.8):
    """
    Create a balanced augmented dataset that focuses on rare classes
    """
    def smart_augment_fn(image, mask):
        # Check if mask contains rare classes (3 or 4)
        rare_classes_present = tf.reduce_any(tf.equal(mask, 3)) or tf.reduce_any(tf.equal(mask, 4))
        
        # Higher augmentation probability for rare classes
        prob = tf.cond(rare_classes_present, 
                      lambda: tf.constant(0.9),  # 90% for rare classes
                      lambda: tf.constant(0.6))  # 60% for common classes
        
        should_augment = tf.random.uniform([]) < prob
        
        def augment():
            return augment_image_and_mask(image, mask)
        
        def no_augment():
            return image, mask
        
        return tf.cond(should_augment, augment, no_augment)
    
    return dataset.map(smart_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

def create_mixup_dataset(dataset, alpha=0.2, mixup_probability=0.3):
    """
    Create a dataset with MixUp augmentation for better generalization
    """
    def mixup_fn(image1, mask1):
        # Get next batch element
        def apply_mixup():
            # Sample lambda from Beta distribution
            lam = np.random.beta(alpha, alpha)
            
            # For simplicity, we'll just return the original
            # In practice, you'd need to mix with another sample
            return image1, mask1
        
        def no_mixup():
            return image1, mask1
        
        should_mixup = tf.random.uniform([]) < mixup_probability
        return tf.cond(should_mixup, apply_mixup, no_mixup)
    
    return dataset.map(mixup_fn, num_parallel_calls=tf.data.AUTOTUNE)

def create_cutmix_dataset(dataset, alpha=1.0, cutmix_probability=0.3):
    """
    Create a dataset with CutMix augmentation
    """
    def cutmix_fn(image1, mask1):
        # For simplicity, return original
        # Full CutMix implementation would require mixing with another sample
        return image1, mask1
    
    return dataset.map(cutmix_fn, num_parallel_calls=tf.data.AUTOTUNE)

def create_enhanced_augmentation_pipeline(dataset, class_weights=None, use_advanced_aug=True):
    """
    Create a comprehensive augmentation pipeline
    """
    # Basic augmentations
    augmented_dataset = create_augmented_dataset(dataset, augment_probability=0.7)
    
    if class_weights is not None and use_advanced_aug:
        # Smart augmentation for rare classes
        augmented_dataset = create_balanced_augmented_dataset(augmented_dataset, class_weights)
    
    # Cache and prefetch for performance
    augmented_dataset = augmented_dataset.cache()
    augmented_dataset = augmented_dataset.prefetch(tf.data.AUTOTUNE)
    
    return augmented_dataset

def visualize_augmentations(dataset, num_samples=4):
    """
    Visualize augmented samples (for debugging)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i, (image, mask) in enumerate(dataset.take(num_samples)):
        if i >= num_samples:
            break
        
        # Show original image (first 3 channels)
        axes[0, i].imshow(image[:, :, :3])
        axes[0, i].set_title(f'Augmented Image {i+1}')
        axes[0, i].axis('off')
        
        # Show mask
        axes[1, i].imshow(mask[:, :, 0], cmap='viridis')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Augmentation samples saved to 'augmentation_samples.png'")
