# ⚡ Optimization Techniques

> **Master TinyML optimization: quantization, pruning, distillation, and advanced compression techniques**

## 🎯 **Learning Objectives**

- Understand quantization techniques (float32 → int8, dynamic, static)
- Master model pruning and weight sharing strategies
- Learn knowledge distillation for model compression
- Explore advanced optimization techniques
- Build production-ready optimized models

## 📚 **Table of Contents**

1. [Quantization Techniques](#quantization-techniques)
2. [Model Pruning](#model-pruning)
3. [Knowledge Distillation](#knowledge-distillation)
4. [Advanced Optimization](#advanced-optimization)
5. [Case Studies](#case-studies)

---

## 🔢 **Quantization Techniques**

### **Quantization Fundamentals**

#### **Concept**

Quantization reduces model precision from 32-bit floating-point to lower precision (16-bit, 8-bit, or even 1-bit) to decrease model size and inference time.

#### **Types of Quantization**

- **Post-training Quantization**: Quantize after training
- **Quantization-aware Training**: Train with quantization in mind
- **Dynamic Quantization**: Quantize weights, keep activations in float
- **Static Quantization**: Quantize both weights and activations

#### **Mathematical Foundation**

- **Quantization Formula**: `Q(x) = round((x - zero_point) / scale)`
- **Dequantization**: `x = scale * Q(x) + zero_point`
- **Scale**: `scale = (max - min) / (2^n - 1)`
- **Zero Point**: `zero_point = -min / scale`

#### **Code Example**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationEngine:
    """Advanced quantization engine for TinyML models"""

    def __init__(self):
        self.quantization_configs = {}
        self.quantized_models = {}

    def post_training_quantization(self, model: tf.keras.Model,
                                 quantization_type: str = "int8") -> tf.keras.Model:
        """Apply post-training quantization"""

        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quantization_type == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        elif quantization_type == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert model
        tflite_model = converter.convert()

        # Save quantized model
        model_name = f"quantized_{quantization_type}_model"
        self.quantized_models[model_name] = tflite_model

        logger.info(f"Model quantized to {quantization_type}")
        return tflite_model

    def quantization_aware_training(self, model: tf.keras.Model,
                                  train_data: np.ndarray,
                                  train_labels: np.ndarray) -> tf.keras.Model:
        """Apply quantization-aware training"""

        # Clone model for quantization-aware training
        qat_model = tf.keras.models.clone_model(model)
        qat_model.set_weights(model.get_weights())

        # Apply quantization to layers
        qat_model = self._apply_quantization_layers(qat_model)

        # Compile model
        qat_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train with quantization
        qat_model.fit(train_data, train_labels, epochs=5, batch_size=32, verbose=0)

        return qat_model

    def _apply_quantization_layers(self, model: tf.keras.Model) -> tf.keras.Model:
        """Apply quantization layers to model"""

        # This is a simplified example
        # In practice, you would use tf.quantization.quantize_and_dequantize

        quantized_layers = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Apply quantization to dense layers
                quantized_layer = tf.keras.layers.Dense(
                    layer.units,
                    activation=layer.activation,
                    kernel_quantizer='quantized_bits(8, 0, 1)',
                    bias_quantizer='quantized_bits(8, 0, 1)'
                )
                quantized_layers.append(quantized_layer)
            else:
                quantized_layers.append(layer)

        # Create new model with quantized layers
        quantized_model = tf.keras.Sequential(quantized_layers)
        return quantized_model

    def analyze_quantization_impact(self, original_model: tf.keras.Model,
                                  quantized_model: bytes,
                                  test_data: np.ndarray) -> Dict[str, Any]:
        """Analyze the impact of quantization"""

        # Original model predictions
        original_predictions = original_model.predict(test_data)

        # Quantized model predictions
        interpreter = tf.lite.Interpreter(model_content=quantized_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        quantized_predictions = []
        for sample in test_data:
            interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            quantized_predictions.append(output[0])

        quantized_predictions = np.array(quantized_predictions)

        # Calculate metrics
        mse = np.mean((original_predictions - quantized_predictions) ** 2)
        mae = np.mean(np.abs(original_predictions - quantized_predictions))

        # Model size comparison
        original_size = original_model.count_params() * 4  # Float32
        quantized_size = len(quantized_model)
        compression_ratio = original_size / quantized_size

        return {
            "mse": mse,
            "mae": mae,
            "original_size_bytes": original_size,
            "quantized_size_bytes": quantized_size,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": (1 - quantized_size / original_size) * 100
        }

    def benchmark_quantized_model(self, quantized_model: bytes,
                                test_data: np.ndarray,
                                num_runs: int = 100) -> Dict[str, float]:
        """Benchmark quantized model performance"""
        import time

        interpreter = tf.lite.Interpreter(model_content=quantized_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Warmup
        interpreter.set_tensor(input_details[0]['index'], test_data[:1])
        interpreter.invoke()

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], test_data[:1])
            interpreter.invoke()
        end_time = time.time()

        avg_time_ms = (end_time - start_time) * 1000 / num_runs
        throughput = 1000 / avg_time_ms

        return {
            "avg_inference_time_ms": avg_time_ms,
            "throughput_inferences_per_sec": throughput,
            "num_runs": num_runs
        }

class AdvancedQuantization:
    """Advanced quantization techniques"""

    def __init__(self):
        self.quantization_methods = {}

    def per_channel_quantization(self, weights: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply per-channel quantization"""

        # Quantize each channel separately
        quantized_weights = np.zeros_like(weights, dtype=np.int8)
        scales = np.zeros(weights.shape[-1])
        zero_points = np.zeros(weights.shape[-1])

        for channel in range(weights.shape[-1]):
            channel_weights = weights[..., channel]

            # Calculate scale and zero point for this channel
            min_val = np.min(channel_weights)
            max_val = np.max(channel_weights)

            scale = (max_val - min_val) / 255.0
            zero_point = -min_val / scale

            # Quantize
            quantized_channel = np.round(channel_weights / scale + zero_point)
            quantized_channel = np.clip(quantized_channel, 0, 255)

            quantized_weights[..., channel] = quantized_channel.astype(np.int8)
            scales[channel] = scale
            zero_points[channel] = zero_point

        return {
            "quantized_weights": quantized_weights,
            "scales": scales,
            "zero_points": zero_points
        }

    def symmetric_quantization(self, weights: np.ndarray) -> Dict[str, Any]:
        """Apply symmetric quantization"""

        # Find maximum absolute value
        max_abs = np.max(np.abs(weights))

        # Calculate scale
        scale = max_abs / 127.0  # For int8 symmetric

        # Quantize
        quantized_weights = np.round(weights / scale)
        quantized_weights = np.clip(quantized_weights, -128, 127)

        return {
            "quantized_weights": quantized_weights.astype(np.int8),
            "scale": scale,
            "zero_point": 0  # Symmetric quantization has zero point at 0
        }

    def asymmetric_quantization(self, weights: np.ndarray) -> Dict[str, Any]:
        """Apply asymmetric quantization"""

        # Find min and max values
        min_val = np.min(weights)
        max_val = np.max(weights)

        # Calculate scale and zero point
        scale = (max_val - min_val) / 255.0
        zero_point = -min_val / scale

        # Quantize
        quantized_weights = np.round(weights / scale + zero_point)
        quantized_weights = np.clip(quantized_weights, 0, 255)

        return {
            "quantized_weights": quantized_weights.astype(np.uint8),
            "scale": scale,
            "zero_point": zero_point
        }

# Example usage
def main():
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Generate test data
    test_data = np.random.randn(100, 10).astype(np.float32)
    test_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, 100))

    # Train model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(test_data, test_labels, epochs=10, batch_size=32, verbose=0)

    # Initialize quantization engine
    quantizer = QuantizationEngine()

    # Apply different quantization techniques
    quantization_types = ["int8", "float16", "dynamic"]
    results = {}

    for qtype in quantization_types:
        quantized_model = quantizer.post_training_quantization(model, qtype)

        # Analyze impact
        impact = quantizer.analyze_quantization_impact(model, quantized_model, test_data[:10])
        results[qtype] = impact

        print(f"\n{qtype.upper()} Quantization Results:")
        print(f"  MSE: {impact['mse']:.6f}")
        print(f"  MAE: {impact['mae']:.6f}")
        print(f"  Compression Ratio: {impact['compression_ratio']:.2f}x")
        print(f"  Size Reduction: {impact['size_reduction_percent']:.1f}%")

    # Test advanced quantization
    advanced_quantizer = AdvancedQuantization()

    # Get model weights
    weights = model.layers[0].get_weights()[0]  # First dense layer weights

    # Apply different quantization methods
    symmetric_result = advanced_quantizer.symmetric_quantization(weights)
    asymmetric_result = advanced_quantizer.asymmetric_quantization(weights)

    print(f"\nSymmetric Quantization:")
    print(f"  Scale: {symmetric_result['scale']:.6f}")
    print(f"  Zero Point: {symmetric_result['zero_point']}")

    print(f"\nAsymmetric Quantization:")
    print(f"  Scale: {asymmetric_result['scale']:.6f}")
    print(f"  Zero Point: {asymmetric_result['zero_point']:.2f}")

if __name__ == "__main__":
    main()
```

---

## ✂️ **Model Pruning**

### **Pruning Fundamentals**

#### **Concept**

Model pruning removes unnecessary weights or neurons from a neural network to reduce model size and computational requirements.

#### **Types of Pruning**

- **Magnitude-based Pruning**: Remove weights with smallest absolute values
- **Gradient-based Pruning**: Remove weights with smallest gradients
- **Structured Pruning**: Remove entire neurons or channels
- **Unstructured Pruning**: Remove individual weights

#### **Code Example**

```python
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PruningEngine:
    """Advanced pruning engine for TinyML models"""

    def __init__(self):
        self.pruning_configs = {}
        self.pruned_models = {}

    def magnitude_based_pruning(self, model: tf.keras.Model,
                               sparsity: float = 0.5) -> tf.keras.Model:
        """Apply magnitude-based pruning"""

        # Clone model
        pruned_model = tf.keras.models.clone_model(model)
        pruned_model.set_weights(model.get_weights())

        # Apply pruning to each layer
        for layer in pruned_model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                # Get weights
                weights = layer.kernel.numpy()

                # Calculate threshold
                threshold = np.percentile(np.abs(weights), sparsity * 100)

                # Create mask
                mask = np.abs(weights) > threshold

                # Apply mask
                pruned_weights = weights * mask
                layer.kernel.assign(pruned_weights)

                logger.info(f"Pruned layer {layer.name}: {np.sum(mask)}/{weights.size} weights kept")

        return pruned_model

    def structured_pruning(self, model: tf.keras.Model,
                          sparsity: float = 0.5) -> tf.keras.Model:
        """Apply structured pruning (remove entire neurons)"""

        # Clone model
        pruned_model = tf.keras.models.clone_model(model)
        pruned_model.set_weights(model.get_weights())

        # Apply structured pruning to dense layers
        for i, layer in enumerate(pruned_model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and i < len(pruned_model.layers) - 1:
                # Get weights
                weights = layer.kernel.numpy()

                # Calculate importance of each neuron (L2 norm of weights)
                neuron_importance = np.linalg.norm(weights, axis=0)

                # Select neurons to keep
                num_neurons_to_keep = int(weights.shape[1] * (1 - sparsity))
                top_neurons = np.argsort(neuron_importance)[-num_neurons_to_keep:]

                # Create mask
                mask = np.zeros(weights.shape[1], dtype=bool)
                mask[top_neurons] = True

                # Apply mask
                pruned_weights = weights[:, mask]
                layer.kernel.assign(pruned_weights)

                # Update layer units
                layer.units = num_neurons_to_keep

                logger.info(f"Structured pruning layer {layer.name}: {num_neurons_to_keep}/{weights.shape[1]} neurons kept")

        return pruned_model

    def iterative_pruning(self, model: tf.keras.Model,
                         target_sparsity: float = 0.8,
                         pruning_steps: int = 5) -> tf.keras.Model:
        """Apply iterative pruning"""

        current_model = model
        current_sparsity = 0.0
        sparsity_increment = target_sparsity / pruning_steps

        for step in range(pruning_steps):
            current_sparsity += sparsity_increment

            # Prune model
            current_model = self.magnitude_based_pruning(current_model, current_sparsity)

            # Fine-tune (simplified - in practice you would retrain)
            logger.info(f"Pruning step {step + 1}/{pruning_steps}, sparsity: {current_sparsity:.2f}")

        return current_model

    def analyze_pruning_impact(self, original_model: tf.keras.Model,
                             pruned_model: tf.keras.Model,
                             test_data: np.ndarray) -> Dict[str, Any]:
        """Analyze the impact of pruning"""

        # Get predictions
        original_predictions = original_model.predict(test_data)
        pruned_predictions = pruned_model.predict(test_data)

        # Calculate accuracy difference
        accuracy_diff = np.mean(np.abs(original_predictions - pruned_predictions))

        # Calculate model size reduction
        original_params = original_model.count_params()
        pruned_params = pruned_model.count_params()

        size_reduction = (original_params - pruned_params) / original_params

        # Calculate sparsity
        total_weights = 0
        zero_weights = 0

        for layer in pruned_model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                weights = layer.kernel.numpy()
                total_weights += weights.size
                zero_weights += np.sum(weights == 0)

        sparsity = zero_weights / total_weights if total_weights > 0 else 0

        return {
            "accuracy_difference": accuracy_diff,
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "size_reduction": size_reduction,
            "sparsity": sparsity,
            "zero_weights": zero_weights,
            "total_weights": total_weights
        }

class AdvancedPruning:
    """Advanced pruning techniques"""

    def __init__(self):
        self.pruning_methods = {}

    def gradient_based_pruning(self, model: tf.keras.Model,
                              train_data: np.ndarray,
                              train_labels: np.ndarray,
                              sparsity: float = 0.5) -> tf.keras.Model:
        """Apply gradient-based pruning"""

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Calculate gradients
        with tf.GradientTape() as tape:
            predictions = model(train_data)
            loss = tf.keras.losses.categorical_crossentropy(train_labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Apply gradient-based pruning
        for layer, grad in zip(model.layers, gradients):
            if hasattr(layer, 'kernel') and layer.kernel is not None and grad is not None:
                # Get weights and gradients
                weights = layer.kernel.numpy()
                grad_weights = grad.numpy()

                # Calculate importance (magnitude of gradients)
                importance = np.abs(grad_weights)

                # Calculate threshold
                threshold = np.percentile(importance, sparsity * 100)

                # Create mask
                mask = importance > threshold

                # Apply mask
                pruned_weights = weights * mask
                layer.kernel.assign(pruned_weights)

        return model

    def lottery_ticket_hypothesis(self, model: tf.keras.Model,
                                 train_data: np.ndarray,
                                 train_labels: np.ndarray,
                                 sparsity: float = 0.8) -> tf.keras.Model:
        """Apply lottery ticket hypothesis pruning"""

        # Save original weights
        original_weights = [layer.get_weights() for layer in model.layers]

        # Train model to find important weights
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=10, batch_size=32, verbose=0)

        # Create mask based on final weights
        masks = []
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                weights = layer.kernel.numpy()
                threshold = np.percentile(np.abs(weights), sparsity * 100)
                mask = np.abs(weights) > threshold
                masks.append(mask)
            else:
                masks.append(None)

        # Reset to original weights
        for layer, original_weight in zip(model.layers, original_weights):
            if original_weight:
                layer.set_weights(original_weight)

        # Apply masks
        for layer, mask in zip(model.layers, masks):
            if mask is not None and hasattr(layer, 'kernel'):
                weights = layer.kernel.numpy()
                pruned_weights = weights * mask
                layer.kernel.assign(pruned_weights)

        return model

# Example usage
def main():
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Generate test data
    test_data = np.random.randn(100, 10).astype(np.float32)
    test_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, 100))

    # Train model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(test_data, test_labels, epochs=10, batch_size=32, verbose=0)

    # Initialize pruning engine
    pruner = PruningEngine()

    # Apply different pruning techniques
    pruning_methods = {
        "magnitude_based": lambda m: pruner.magnitude_based_pruning(m, 0.5),
        "structured": lambda m: pruner.structured_pruning(m, 0.5),
        "iterative": lambda m: pruner.iterative_pruning(m, 0.8, 5)
    }

    results = {}

    for method_name, method_func in pruning_methods.items():
        pruned_model = method_func(model)

        # Analyze impact
        impact = pruner.analyze_pruning_impact(model, pruned_model, test_data[:10])
        results[method_name] = impact

        print(f"\n{method_name.replace('_', ' ').title()} Pruning Results:")
        print(f"  Accuracy Difference: {impact['accuracy_difference']:.6f}")
        print(f"  Size Reduction: {impact['size_reduction']:.2%}")
        print(f"  Sparsity: {impact['sparsity']:.2%}")
        print(f"  Zero Weights: {impact['zero_weights']}/{impact['total_weights']}")

    # Test advanced pruning
    advanced_pruner = AdvancedPruning()

    # Gradient-based pruning
    gradient_pruned = advanced_pruner.gradient_based_pruning(
        tf.keras.models.clone_model(model), test_data, test_labels, 0.5
    )

    print(f"\nGradient-based Pruning:")
    gradient_impact = pruner.analyze_pruning_impact(model, gradient_pruned, test_data[:10])
    print(f"  Accuracy Difference: {gradient_impact['accuracy_difference']:.6f}")
    print(f"  Sparsity: {gradient_impact['sparsity']:.2%}")

if __name__ == "__main__":
    main()
```

---

## 🎓 **Knowledge Distillation**

### **Distillation Fundamentals**

#### **Concept**

Knowledge distillation transfers knowledge from a large, complex teacher model to a smaller, simpler student model while maintaining performance.

#### **Mathematical Foundation**

- **Softmax with Temperature**: `softmax(x/T) = exp(x_i/T) / Σ exp(x_j/T)`
- **Distillation Loss**: `L = α * L_hard + (1-α) * L_soft`
- **Hard Loss**: Standard cross-entropy with true labels
- **Soft Loss**: Cross-entropy with teacher's soft predictions

#### **Code Example**

```python
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeDistillation:
    """Knowledge distillation for model compression"""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = None
        self.student_model = None

    def create_teacher_model(self, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        """Create a large teacher model"""
        teacher = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.teacher_model = teacher
        return teacher

    def create_student_model(self, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        """Create a small student model"""
        student = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.student_model = student
        return student

    def softmax_with_temperature(self, logits: tf.Tensor, temperature: float) -> tf.Tensor:
        """Apply softmax with temperature"""
        return tf.nn.softmax(logits / temperature)

    def distillation_loss(self, student_logits: tf.Tensor,
                         teacher_logits: tf.Tensor,
                         true_labels: tf.Tensor) -> tf.Tensor:
        """Calculate distillation loss"""

        # Soft loss (student learns from teacher)
        soft_student = self.softmax_with_temperature(student_logits, self.temperature)
        soft_teacher = self.softmax_with_temperature(teacher_logits, self.temperature)

        soft_loss = tf.keras.losses.categorical_crossentropy(soft_teacher, soft_student)
        soft_loss = tf.reduce_mean(soft_loss) * (self.temperature ** 2)

        # Hard loss (student learns from true labels)
        hard_loss = tf.keras.losses.categorical_crossentropy(true_labels, soft_student)
        hard_loss = tf.reduce_mean(hard_loss)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss

    def train_with_distillation(self, train_data: np.ndarray,
                               train_labels: np.ndarray,
                               epochs: int = 20) -> tf.keras.Model:
        """Train student model with knowledge distillation"""

        if self.teacher_model is None or self.student_model is None:
            raise ValueError("Teacher and student models must be created first")

        # Train teacher model first
        self.teacher_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Training teacher model...")
        self.teacher_model.fit(train_data, train_labels, epochs=10, batch_size=32, verbose=0)

        # Get teacher predictions
        teacher_predictions = self.teacher_model.predict(train_data)

        # Create custom training loop for distillation
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            batch_size = 32
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                batch_teacher_preds = teacher_predictions[i:i + batch_size]

                with tf.GradientTape() as tape:
                    # Get student predictions
                    student_logits = self.student_model(batch_data, training=True)

                    # Get teacher predictions (convert from probabilities to logits)
                    teacher_logits = tf.math.log(batch_teacher_preds + 1e-8)

                    # Calculate distillation loss
                    loss = self.distillation_loss(student_logits, teacher_logits, batch_labels)

                # Update student model
                gradients = tape.gradient(loss, self.student_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

                epoch_loss += loss.numpy()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return self.student_model

    def compare_models(self, test_data: np.ndarray,
                      test_labels: np.ndarray) -> Dict[str, Any]:
        """Compare teacher and student models"""

        # Get predictions
        teacher_preds = self.teacher_model.predict(test_data)
        student_preds = self.student_model.predict(test_data)

        # Calculate accuracies
        teacher_accuracy = np.mean(np.argmax(teacher_preds, axis=1) == np.argmax(test_labels, axis=1))
        student_accuracy = np.mean(np.argmax(student_preds, axis=1) == np.argmax(test_labels, axis=1))

        # Calculate model sizes
        teacher_params = self.teacher_model.count_params()
        student_params = self.student_model.count_params()

        # Calculate compression ratio
        compression_ratio = teacher_params / student_params

        return {
            "teacher_accuracy": teacher_accuracy,
            "student_accuracy": student_accuracy,
            "accuracy_difference": teacher_accuracy - student_accuracy,
            "teacher_parameters": teacher_params,
            "student_parameters": student_params,
            "compression_ratio": compression_ratio,
            "size_reduction": (teacher_params - student_params) / teacher_params
        }

class AdvancedDistillation:
    """Advanced knowledge distillation techniques"""

    def __init__(self):
        self.distillation_methods = {}

    def feature_distillation(self, teacher_model: tf.keras.Model,
                           student_model: tf.keras.Model,
                           train_data: np.ndarray,
                           train_labels: np.ndarray) -> tf.keras.Model:
        """Apply feature-level distillation"""

        # Extract intermediate features from teacher
        teacher_features = []
        for layer in teacher_model.layers[:-1]:  # Exclude output layer
            if isinstance(layer, tf.keras.layers.Dense):
                teacher_features.append(layer.output)

        # Create feature extraction model
        feature_extractor = tf.keras.Model(
            inputs=teacher_model.input,
            outputs=teacher_features
        )

        # Get teacher features
        teacher_feature_maps = feature_extractor.predict(train_data)

        # Train student with feature matching
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(10):
            epoch_loss = 0
            num_batches = 0

            batch_size = 32
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                batch_teacher_features = [feat[i:i + batch_size] for feat in teacher_feature_maps]

                with tf.GradientTape() as tape:
                    # Get student features
                    student_features = []
                    x = batch_data
                    for layer in student_model.layers[:-1]:  # Exclude output layer
                        x = layer(x)
                        if isinstance(layer, tf.keras.layers.Dense):
                            student_features.append(x)

                    # Calculate feature matching loss
                    feature_loss = 0
                    for student_feat, teacher_feat in zip(student_features, batch_teacher_features):
                        # Match feature dimensions if needed
                        if student_feat.shape[-1] != teacher_feat.shape[-1]:
                            # Use a projection layer
                            projection = tf.keras.layers.Dense(teacher_feat.shape[-1])
                            student_feat = projection(student_feat)

                        feature_loss += tf.reduce_mean(tf.square(student_feat - teacher_feat))

                    # Add classification loss
                    student_output = student_model(batch_data, training=True)
                    classification_loss = tf.keras.losses.categorical_crossentropy(batch_labels, student_output)
                    classification_loss = tf.reduce_mean(classification_loss)

                    total_loss = feature_loss + classification_loss

                # Update student model
                gradients = tape.gradient(total_loss, student_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

                epoch_loss += total_loss.numpy()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            logger.info(f"Feature distillation epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        return student_model

    def self_distillation(self, model: tf.keras.Model,
                         train_data: np.ndarray,
                         train_labels: np.ndarray) -> tf.keras.Model:
        """Apply self-distillation (model learns from itself)"""

        # Train model normally first
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=5, batch_size=32, verbose=0)

        # Get model predictions
        model_predictions = model.predict(train_data)

        # Train model to match its own predictions (with temperature)
        temperature = 3.0

        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(5):
            epoch_loss = 0
            num_batches = 0

            batch_size = 32
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_predictions = model_predictions[i:i + batch_size]

                with tf.GradientTape() as tape:
                    # Get current predictions
                    current_predictions = model(batch_data, training=True)

                    # Apply temperature to both predictions
                    soft_current = tf.nn.softmax(current_predictions / temperature)
                    soft_target = tf.nn.softmax(tf.math.log(batch_predictions + 1e-8) / temperature)

                    # Calculate loss
                    loss = tf.keras.losses.categorical_crossentropy(soft_target, soft_current)
                    loss = tf.reduce_mean(loss) * (temperature ** 2)

                # Update model
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                epoch_loss += loss.numpy()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            logger.info(f"Self-distillation epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        return model

# Example usage
def main():
    # Generate sample data
    train_data = np.random.randn(1000, 20).astype(np.float32)
    train_labels = tf.keras.utils.to_categorical(np.random.randint(0, 5, 1000))

    test_data = np.random.randn(200, 20).astype(np.float32)
    test_labels = tf.keras.utils.to_categorical(np.random.randint(0, 5, 200))

    # Initialize knowledge distillation
    distiller = KnowledgeDistillation(temperature=3.0, alpha=0.7)

    # Create models
    teacher = distiller.create_teacher_model((20,), 5)
    student = distiller.create_student_model((20,), 5)

    print(f"Teacher model parameters: {teacher.count_params()}")
    print(f"Student model parameters: {student.count_params()}")

    # Train with distillation
    trained_student = distiller.train_with_distillation(train_data, train_labels, epochs=10)

    # Compare models
    comparison = distiller.compare_models(test_data, test_labels)

    print(f"\nModel Comparison:")
    print(f"Teacher accuracy: {comparison['teacher_accuracy']:.4f}")
    print(f"Student accuracy: {comparison['student_accuracy']:.4f}")
    print(f"Accuracy difference: {comparison['accuracy_difference']:.4f}")
    print(f"Compression ratio: {comparison['compression_ratio']:.2f}x")
    print(f"Size reduction: {comparison['size_reduction']:.2%}")

    # Test advanced distillation
    advanced_distiller = AdvancedDistillation()

    # Feature distillation
    feature_distilled = advanced_distiller.feature_distillation(
        teacher, tf.keras.models.clone_model(student), train_data, train_labels
    )

    # Self-distillation
    self_distilled = advanced_distiller.self_distillation(
        tf.keras.models.clone_model(student), train_data, train_labels
    )

    print(f"\nAdvanced Distillation Results:")
    print(f"Feature distillation completed")
    print(f"Self-distillation completed")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Optimization Techniques**

#### **Q1: What is quantization and why is it important for TinyML?**

**Answer**:

- **Definition**: Reducing model precision from 32-bit to lower precision (8-bit, 16-bit)
- **Benefits**:
  - Reduces model size by 4x (float32 → int8)
  - Faster inference on integer-only hardware
  - Lower memory bandwidth requirements
  - Better power efficiency
- **Types**: Post-training, quantization-aware training, dynamic, static

#### **Q2: What are the trade-offs between different quantization methods?**

**Answer**:

- **Post-training**: Easy to apply, but may lose accuracy
- **Quantization-aware Training**: Better accuracy, but requires retraining
- **Dynamic Quantization**: Weights quantized, activations in float
- **Static Quantization**: Both weights and activations quantized
- **Per-channel vs Per-tensor**: Per-channel more accurate but complex

#### **Q3: How does model pruning work and what are the different types?**

**Answer**:

- **Magnitude-based**: Remove weights with smallest absolute values
- **Gradient-based**: Remove weights with smallest gradients
- **Structured**: Remove entire neurons or channels
- **Unstructured**: Remove individual weights
- **Iterative**: Gradually increase sparsity over multiple steps

#### **Q4: What is knowledge distillation and how does it help with model compression?**

**Answer**:

- **Concept**: Transfer knowledge from large teacher to small student model
- **Process**: Student learns from teacher's soft predictions and true labels
- **Benefits**:
  - Maintains accuracy while reducing model size
  - Can compress models by 10x or more
  - Works well with other optimization techniques
- **Types**: Response distillation, feature distillation, self-distillation

#### **Q5: How do you choose the right optimization technique for a specific use case?**

**Answer**:

- **Memory constraints**: Quantization + pruning
- **Latency constraints**: Structured pruning + quantization
- **Accuracy critical**: Quantization-aware training + knowledge distillation
- **Power constraints**: Aggressive quantization + model compression
- **Hardware specific**: Choose techniques supported by target hardware

---

**Ready to explore real-world applications? Let's dive into [Use Cases](./UseCases.md) next!** 🚀
