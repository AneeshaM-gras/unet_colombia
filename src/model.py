# src/model.py

import tensorflow as tf

# --- Custom Loss and Metrics ---
@tf.keras.utils.register_keras_serializable()
class WeightedCombinedDiceCE(tf.keras.losses.Loss):
    """
    A custom loss class that combines Dice Loss and a WEIGHTED version of
    Sparse Categorical Cross-Entropy. This version is robust to extra
    keyword arguments during model loading.
    """
    def __init__(self, num_classes, class_weights, weight_ce=0.5, weight_dice=0.5, name="weighted_combined_dice_ce", **kwargs):
        # **kwargs catches any unexpected arguments (like 'reduction') from Keras
        # and passes them to the parent class, preventing errors.
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.cce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none' # reduction='none' is crucial for manual weighting
        )

    def dice_loss(self, y_true, y_pred):
        """Calculates the multi-class Dice Loss."""
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes, axis=-1)
        y_true_one_hot = tf.squeeze(y_true_one_hot, axis=-2)
        
        y_true_flat = tf.reshape(y_true_one_hot, [-1, self.num_classes])
        y_pred_flat = tf.reshape(y_pred, [-1, self.num_classes])
        
        epsilon = 1e-5
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        cardinality = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)
        
        dice_coefficient = (2. * intersection + epsilon) / (cardinality + epsilon)
        return 1 - tf.reduce_mean(dice_coefficient)

    def call(self, y_true, y_pred):
        """This method is the core of the loss calculation during training."""
        # Calculate the base per-pixel CCE loss
        unweighted_cce = self.cce_loss_fn(y_true, y_pred)
        
        # Apply the weights for each pixel based on its true class
        y_true_squeezed = tf.squeeze(y_true, axis=-1)
        weights = tf.gather(self.class_weights, tf.cast(y_true_squeezed, tf.int32))
        weighted_cce_loss = tf.reduce_mean(unweighted_cce * weights)
        
        # Calculate the dice loss component
        dice_loss_val = self.dice_loss(y_true, y_pred)
        
        # Return the final combined loss
        return (self.weight_ce * weighted_cce_loss) + (self.weight_dice * dice_loss_val)

    def get_config(self):
        """Enables the class to be saved and loaded correctly."""
        base_config = super().get_config()
        config = {
            "num_classes": self.num_classes,
            "class_weights": self.class_weights.numpy().tolist(),
            "weight_ce": self.weight_ce,
            "weight_dice": self.weight_dice,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Creates the loss class from its saved configuration."""
        return cls(**config)

class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    """
    A wrapper for MeanIoU that handles sparse labels and probability predictions.
    It applies argmax to the predictions before passing them to the underlying metric.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert model's probability predictions to class indices
        y_pred = tf.math.argmax(y_pred, axis=-1)
        # Call the parent's update_state with corrected predictions
        super().update_state(y_true, y_pred, sample_weight)

class SparseMetricWrapper(tf.keras.metrics.Metric):
    """
    A generic wrapper that makes standard Keras metrics compatible with
    sparse integer labels and probability predictions from a model.
    This version includes get_config for model saving and loading.
    """
    def __init__(self, metric, name=None, **kwargs):
        if name is None:
            name = metric.name
        super().__init__(name=name, **kwargs)
        self.metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()
        
    # --- THE CRITICAL FIX IS HERE ---
    def get_config(self):
        # Get the config from the parent class (e.g., name, dtype)
        base_config = super().get_config()
        # Add the configuration of our inner metric so Keras can rebuild it.
        # tf.keras.metrics.serialize will convert the metric object to a dict.
        config = {"metric": tf.keras.metrics.serialize(self.metric)}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # Keras will call this method to recreate the instance.
        # We deserialize the inner metric from its config dict.
        metric = tf.keras.metrics.deserialize(config.pop("metric"))
        # Create the new wrapper instance with the recreated inner metric.
        return cls(metric, **config)

@tf.keras.utils.register_keras_serializable()
class SparseF1Score(tf.keras.metrics.Metric):
    """
    F1 Score metric that handles sparse integer labels and probability predictions.
    Calculates F1 = 2 * (precision * recall) / (precision + recall)
    """
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices
        y_pred = tf.math.argmax(y_pred, axis=-1)
        # Squeeze y_true if needed
        y_true = tf.squeeze(y_true, axis=-1)
        # Update precision and recall
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        # Calculate F1 score with safe division
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_classes": self.num_classes,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)