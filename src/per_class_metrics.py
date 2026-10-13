# src/per_class_metrics.py

import tensorflow as tf
import numpy as np

def calculate_per_class_metrics_simple(model, dataset, num_classes):
    """
    Calculate per-class metrics for a trained model on a dataset.
    Simple implementation without custom metrics.
    
    Args:
        model: Trained Keras model
        dataset: TensorFlow dataset
        num_classes: Number of classes
    
    Returns:
        Dictionary with per-class metrics
    """
    # Initialize counters
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    class_intersection = np.zeros(num_classes)
    class_union = np.zeros(num_classes)
    class_tp = np.zeros(num_classes)
    class_fp = np.zeros(num_classes)
    class_fn = np.zeros(num_classes)
    
    # Calculate metrics on dataset
    for batch in dataset:
        x_batch, y_batch = batch
        y_pred = model.predict(x_batch, verbose=0)
        
        # Convert predictions to class indices
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # Ensure y_true is the right shape
        y_true = tf.squeeze(y_batch, axis=-1)
        
        # Convert to numpy for easier computation
        y_true_np = y_true.numpy()
        y_pred_np = y_pred_classes.numpy()
        
        # Calculate metrics for each class
        for class_id in range(num_classes):
            # True positives for this class
            true_mask = (y_true_np == class_id)
            pred_mask = (y_pred_np == class_id)
            
            # Correct predictions for this class
            correct_mask = true_mask & pred_mask
            
            # Count correct and total for this class
            correct_count = np.sum(correct_mask)
            total_count = np.sum(true_mask)
            
            # Intersection and union for IoU
            intersection_mask = true_mask & pred_mask
            union_mask = true_mask | pred_mask
            
            intersection_count = np.sum(intersection_mask)
            union_count = np.sum(union_mask)
            
            # Update counters
            class_correct[class_id] += correct_count
            class_total[class_id] += total_count
            class_intersection[class_id] += intersection_count
            class_union[class_id] += union_count
            
            # Precision/Recall components
            tp = correct_count
            fp = np.sum(pred_mask) - tp
            fn = np.sum(true_mask) - tp
            class_tp[class_id] += tp
            class_fp[class_id] += fp
            class_fn[class_id] += fn
    
    # Calculate final metrics with proper zero-division handling
    per_class_accuracy = np.zeros_like(class_correct, dtype=np.float64)
    per_class_iou = np.zeros_like(class_intersection, dtype=np.float64)
    per_class_precision = np.zeros_like(class_tp, dtype=np.float64)
    per_class_recall = np.zeros_like(class_tp, dtype=np.float64)
    per_class_f1 = np.zeros_like(class_tp, dtype=np.float64)
    
    # Calculate accuracy for classes that exist
    valid_classes = class_total > 0
    per_class_accuracy[valid_classes] = class_correct[valid_classes] / class_total[valid_classes]
    
    # Calculate IoU for classes that exist
    valid_iou_classes = class_union > 0
    per_class_iou[valid_iou_classes] = class_intersection[valid_iou_classes] / class_union[valid_iou_classes]
    
    # Precision/Recall/F1 with safe division
    precision_den = class_tp + class_fp
    recall_den = class_tp + class_fn
    valid_precision = precision_den > 0
    valid_recall = recall_den > 0
    per_class_precision[valid_precision] = class_tp[valid_precision] / precision_den[valid_precision]
    per_class_recall[valid_recall] = class_tp[valid_recall] / recall_den[valid_recall]
    f1_den = per_class_precision + per_class_recall
    valid_f1 = f1_den > 0
    per_class_f1[valid_f1] = 2 * (per_class_precision[valid_f1] * per_class_recall[valid_f1]) / f1_den[valid_f1]
    
    results = {
        'per_class_accuracy': per_class_accuracy,
        'per_class_iou': per_class_iou,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1
    }
    
    return results

def print_per_class_metrics(metrics_dict, class_names=None):
    """
    Print per-class metrics in a formatted way.
    
    Args:
        metrics_dict: Dictionary containing per-class metrics
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics_dict.get('per_class_accuracy', [])))]
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    
    # Print header
    print(f"{'Class':<12} {'Accuracy':<10} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)
    
    # Print metrics for each class
    accuracies = metrics_dict.get('per_class_accuracy', [])
    ious = metrics_dict.get('per_class_iou', [])
    precisions = metrics_dict.get('per_class_precision', [])
    recalls = metrics_dict.get('per_class_recall', [])
    f1s = metrics_dict.get('per_class_f1', [])
    
    for i, class_name in enumerate(class_names):
        acc = accuracies[i] if i < len(accuracies) else 0.0
        iou = ious[i] if i < len(ious) else 0.0
        prec = precisions[i] if i < len(precisions) else 0.0
        rec = recalls[i] if i < len(recalls) else 0.0
        f1 = f1s[i] if i < len(f1s) else 0.0
        
        print(f"{class_name:<12} {acc:<10.4f} {iou:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    
    # Print averages
    if len(accuracies) > 0:
        avg_acc = np.mean(accuracies)
        avg_iou = np.mean(ious)
        avg_prec = np.mean(precisions) if len(precisions) > 0 else 0.0
        avg_rec = np.mean(recalls) if len(recalls) > 0 else 0.0
        avg_f1 = np.mean(f1s) if len(f1s) > 0 else 0.0
        
        print("-" * 80)
        print(f"{'Average':<12} {avg_acc:<10.4f} {avg_iou:<10.4f} {avg_prec:<10.4f} {avg_rec:<10.4f} {avg_f1:<10.4f}")
    
    print("="*80)

def analyze_class_distribution(dataset, num_classes):
    """
    Analyze the distribution of classes in the dataset.
    
    Args:
        dataset: TensorFlow dataset
        num_classes: Number of classes
    
    Returns:
        Dictionary with class distribution information
    """
    class_counts = np.zeros(num_classes)
    total_pixels = 0
    
    for batch in dataset:
        x_batch, y_batch = batch
        y_true = tf.squeeze(y_batch, axis=-1)
        y_true_np = y_true.numpy()
        
        # Count pixels for each class
        for class_id in range(num_classes):
            count = np.sum(y_true_np == class_id)
            class_counts[class_id] += count
        
        total_pixels += y_true_np.size
    
    # Calculate percentages
    class_percentages = (class_counts / total_pixels) * 100
    
    distribution = {
        'class_counts': class_counts.tolist(),
        'class_percentages': class_percentages.tolist(),
        'total_pixels': float(total_pixels)
    }
    
    return distribution

def print_class_distribution(distribution, class_names=None):
    """
    Print class distribution in a formatted way.
    
    Args:
        distribution: Dictionary with class distribution information
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(distribution['class_counts']))]
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    
    print(f"{'Class':<12} {'Count':<15} {'Percentage':<12}")
    print("-" * 60)
    
    counts = distribution['class_counts']
    percentages = distribution['class_percentages']
    
    for i, class_name in enumerate(class_names):
        count = counts[i]
        percentage = percentages[i]
        print(f"{class_name:<12} {count:<15,} {percentage:<12.2f}%")
    
    print("-" * 60)
    print(f"{'Total':<12} {distribution['total_pixels']:<15,} {'100.00%':<12}")
    print("="*60)


def macro_average(metrics_dict):
    """Return macro-averaged metrics from a per-class metrics dictionary."""
    return {
        'macro_accuracy': float(np.mean(metrics_dict.get('per_class_accuracy', [0.0]))),
        'macro_iou': float(np.mean(metrics_dict.get('per_class_iou', [0.0]))),
        'macro_precision': float(np.mean(metrics_dict.get('per_class_precision', [0.0]))),
        'macro_recall': float(np.mean(metrics_dict.get('per_class_recall', [0.0]))),
        'macro_f1': float(np.mean(metrics_dict.get('per_class_f1', [0.0]))),
    }
