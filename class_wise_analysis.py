#!/usr/bin/env python3
"""
Calculate Class-wise Accuracy for U-Net Models
"""

import os
import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    FINAL_PATCHES_FOLDER, IMAGE_BAND_NAMES, LABEL_BAND_NAME,
    NUM_IMAGE_BANDS, PATCH_SIZE, TRAINING_CONFIG_PATH, get_training_batch_size
)
from src.data_loader import create_dataset
from src.model import WeightedCombinedDiceCE, SparseMeanIoU, SparseMetricWrapper
from src.combined_loss_system import CombinedFocalJaccardLoss

def load_model(model_path, model_name):
    """Load model with appropriate custom objects"""
    try:
        if 'finetuned' in model_path.lower():
            print(f"Loading {model_name} (Fine-tuned Model)...")
            custom_objects = {
                'CombinedFocalJaccardLoss': CombinedFocalJaccardLoss,
                'SparseMeanIoU': SparseMeanIoU,
                'SparseMetricWrapper': SparseMetricWrapper
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print(f"{model_name} loaded successfully")
            return model, "finetuned"
        else:
            print(f"Loading {model_name} (Combined Loss)...")
            custom_objects = {
                'CombinedFocalJaccardLoss': CombinedFocalJaccardLoss,
                'SparseMeanIoU': SparseMeanIoU,
                'SparseMetricWrapper': SparseMetricWrapper
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print(f"{model_name} loaded successfully")
            return model, "baseline"
    except Exception as e:
        print(f"ERROR: Failed to load {model_name}: {e}")
        return None, None

def get_predictions_and_labels(model, dataset):
    """Get predictions and true labels from dataset"""
    print("Generating predictions...")
    
    all_predictions = []
    all_labels = []
    
    for batch_images, batch_labels in dataset:
        # Get predictions
        batch_predictions = model.predict(batch_images, verbose=0)
        predicted_classes = np.argmax(batch_predictions, axis=-1)
        
        # Flatten arrays
        all_predictions.extend(predicted_classes.flatten())
        all_labels.extend(batch_labels.numpy().flatten())
    
    return np.array(all_predictions), np.array(all_labels)

def calculate_class_metrics(y_true, y_pred, num_classes, model_name):
    """Calculate detailed class-wise metrics"""
    print(f"\n--- Class-wise Metrics for {model_name} ---")
    
    # Get unique classes present in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    print(f"Classes present in data: {unique_classes}")
    
    # Calculate per-class accuracy
    class_accuracy = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    
    for class_id in range(num_classes):
        if class_id in unique_classes:
            # True positives, false positives, false negatives
            tp = np.sum((y_true == class_id) & (y_pred == class_id))
            fp = np.sum((y_true != class_id) & (y_pred == class_id))
            fn = np.sum((y_true == class_id) & (y_pred != class_id))
            tn = np.sum((y_true != class_id) & (y_pred != class_id))
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_accuracy[class_id] = accuracy
            class_precision[class_id] = precision
            class_recall[class_id] = recall
            class_f1[class_id] = f1
            
            print(f"Class {class_id}:")
            print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
            print(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)")
            print(f"  F1-Score:  {f1:.4f} ({f1*100:.1f}%)")
            print(f"  Samples:   {np.sum(y_true == class_id)}")
        else:
            print(f"Class {class_id}: Not present in test data")
            class_accuracy[class_id] = 0
            class_precision[class_id] = 0
            class_recall[class_id] = 0
            class_f1[class_id] = 0
    
    return {
        'accuracy': class_accuracy,
        'precision': class_precision,
        'recall': class_recall,
        'f1': class_f1
    }

def plot_class_comparison(baseline_metrics, optimized_metrics, num_classes):
    """Plot comparison of class-wise metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Class-wise Performance Comparison: Baseline vs Optimized', fontsize=16)
    
    classes = list(range(num_classes))
    
    # Accuracy comparison
    baseline_acc = [baseline_metrics['accuracy'][i] for i in classes]
    optimized_acc = [optimized_metrics['accuracy'][i] for i in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.8)
    axes[0, 0].bar(x + width/2, optimized_acc, width, label='Optimized', alpha=0.8)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Class-wise Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'Class {i}' for i in classes])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision comparison
    baseline_prec = [baseline_metrics['precision'][i] for i in classes]
    optimized_prec = [optimized_metrics['precision'][i] for i in classes]
    
    axes[0, 1].bar(x - width/2, baseline_prec, width, label='Baseline', alpha=0.8)
    axes[0, 1].bar(x + width/2, optimized_prec, width, label='Optimized', alpha=0.8)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Class-wise Precision')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'Class {i}' for i in classes])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Recall comparison
    baseline_rec = [baseline_metrics['recall'][i] for i in classes]
    optimized_rec = [optimized_metrics['recall'][i] for i in classes]
    
    axes[1, 0].bar(x - width/2, baseline_rec, width, label='Baseline', alpha=0.8)
    axes[1, 0].bar(x + width/2, optimized_rec, width, label='Optimized', alpha=0.8)
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Class-wise Recall')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'Class {i}' for i in classes])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1-Score comparison
    baseline_f1 = [baseline_metrics['f1'][i] for i in classes]
    optimized_f1 = [optimized_metrics['f1'][i] for i in classes]
    
    axes[1, 1].bar(x - width/2, baseline_f1, width, label='Baseline', alpha=0.8)
    axes[1, 1].bar(x + width/2, optimized_f1, width, label='Optimized', alpha=0.8)
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('Class-wise F1-Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'Class {i}' for i in classes])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('class_wise_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== CLASS-WISE ACCURACY ANALYSIS ===\n")
    
    # Load training config
    with open(TRAINING_CONFIG_PATH, 'r') as f:
        training_config = json.load(f)
    NUM_CLASSES = training_config['num_classes']
    
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class weights: {training_config['class_weights']}")
    
    # Prepare test data
    print("\nPreparing test data...")
    all_files = [os.path.join(FINAL_PATCHES_FOLDER, f) 
                for f in os.listdir(FINAL_PATCHES_FOLDER) 
                if f.lower().endswith(('.tif', '.tiff'))]
    
    # Use last 11 files for testing
    test_files = all_files[-11:]
    print(f"Using {len(test_files)} files for testing")
    
    test_dataset = create_dataset(
        test_files, get_training_batch_size(), IMAGE_BAND_NAMES, LABEL_BAND_NAME
    )
    
    # Load models
    baseline_model, _ = load_model(
        './models/saved_models/unet_optimized_combined_loss.h5',
        'Base Model (Combined Loss)'
    )
    
    optimized_model, _ = load_model(
        './models/saved_models/unet_finetuned_gold.h5',
        'Fine-tuned Model'
    )
    
    # Compile models for prediction
    if baseline_model:
        baseline_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    if optimized_model:
        optimized_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Get predictions and calculate metrics
    results = {}
    
    if baseline_model:
        print("\n" + "="*50)
        baseline_pred, baseline_true = get_predictions_and_labels(baseline_model, test_dataset)
        results['baseline'] = calculate_class_metrics(baseline_true, baseline_pred, NUM_CLASSES, 'Base Model')
    
    if optimized_model:
        print("\n" + "="*50)
        optimized_pred, optimized_true = get_predictions_and_labels(optimized_model, test_dataset)
        results['optimized'] = calculate_class_metrics(optimized_true, optimized_pred, NUM_CLASSES, 'Fine-tuned Model')
    
    # Print comparison table
    if 'baseline' in results and 'optimized' in results:
        print("\n" + "="*80)
        print("CLASS-WISE COMPARISON TABLE")
        print("="*80)
        
        print(f"{'Class':<8} {'Metric':<10} {'Base Model':<12} {'Fine-tuned':<12} {'Improvement':<15}")
        print("-" * 80)
        
        for class_id in range(NUM_CLASSES):
            baseline_acc = results['baseline']['accuracy'][class_id]
            optimized_acc = results['optimized']['accuracy'][class_id]
            improvement = ((optimized_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            
            print(f"{class_id:<8} {'Accuracy':<10} {baseline_acc:<12.4f} {optimized_acc:<12.4f} {improvement:+.1f}%")
            
            baseline_prec = results['baseline']['precision'][class_id]
            optimized_prec = results['optimized']['precision'][class_id]
            prec_improvement = ((optimized_prec - baseline_prec) / baseline_prec * 100) if baseline_prec > 0 else 0
            
            print(f"{'':<8} {'Precision':<10} {baseline_prec:<12.4f} {optimized_prec:<12.4f} {prec_improvement:+.1f}%")
            
            baseline_rec = results['baseline']['recall'][class_id]
            optimized_rec = results['optimized']['recall'][class_id]
            rec_improvement = ((optimized_rec - baseline_rec) / baseline_rec * 100) if baseline_rec > 0 else 0
            
            print(f"{'':<8} {'Recall':<10} {baseline_rec:<12.4f} {optimized_rec:<12.4f} {rec_improvement:+.1f}%")
            
            baseline_f1 = results['baseline']['f1'][class_id]
            optimized_f1 = results['optimized']['f1'][class_id]
            f1_improvement = ((optimized_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            
            print(f"{'':<8} {'F1-Score':<10} {baseline_f1:<12.4f} {optimized_f1:<12.4f} {f1_improvement:+.1f}%")
            print("-" * 80)
        
        # Plot comparison
        plot_class_comparison(results['baseline'], results['optimized'], NUM_CLASSES)
        
        # Save results
        with open('class_wise_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to 'class_wise_results.json'")
        print(f"Plots saved to 'class_wise_comparison.png'")
    
    print("\n=== CLASS-WISE ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
