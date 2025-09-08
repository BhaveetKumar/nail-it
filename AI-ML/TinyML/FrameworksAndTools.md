# 🛠️ Frameworks and Tools

> **Master TinyML frameworks: TensorFlow Lite, Edge Impulse, and model conversion workflows**

## 🎯 **Learning Objectives**

- Understand TensorFlow Lite for Microcontrollers (TFLM) architecture
- Master Edge Impulse platform for TinyML development
- Learn model conversion and optimization workflows
- Explore ONNX Runtime Mobile and other frameworks
- Build production-ready TinyML applications

## 📚 **Table of Contents**

1. [TensorFlow Lite for Microcontrollers](#tensorflow-lite-for-microcontrollers)
2. [Edge Impulse Platform](#edge-impulse-platform)
3. [Model Conversion Workflow](#model-conversion-workflow)
4. [Other Frameworks](#other-frameworks)
5. [Tool Comparison](#tool-comparison)

---

## 🔧 **TensorFlow Lite for Microcontrollers**

### **TFLM Architecture**

#### **Core Components**
- **Interpreter**: Executes models on target hardware
- **Micro Allocator**: Manages memory allocation
- **Micro Profiler**: Performance monitoring
- **Schema**: Model format specification
- **Kernels**: Hardware-specific operations

#### **Code Example**

```python
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFLiteConverter:
    """Convert TensorFlow models to TensorFlow Lite format"""
    
    def __init__(self):
        self.converter = None
        self.tflite_model = None
    
    def convert_model(self, model: tf.keras.Model, 
                     quantization: str = "float32") -> bytes:
        """Convert Keras model to TFLite format"""
        
        # Create converter
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        if quantization == "int8":
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.target_spec.supported_types = [tf.int8]
        elif quantization == "float16":
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        self.tflite_model = self.converter.convert()
        
        return self.tflite_model
    
    def save_model(self, filepath: str):
        """Save TFLite model to file"""
        with open(filepath, 'wb') as f:
            f.write(self.tflite_model)
        logger.info(f"TFLite model saved to {filepath}")
    
    def analyze_model(self) -> Dict[str, Any]:
        """Analyze TFLite model"""
        if self.tflite_model is None:
            raise ValueError("No model to analyze")
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Calculate model size
        model_size_kb = len(self.tflite_model) / 1024
        
        return {
            "model_size_kb": model_size_kb,
            "input_details": input_details,
            "output_details": output_details,
            "quantization": self.converter.target_spec.supported_types[0].name if self.converter.target_spec.supported_types else "float32"
        }

class TFLiteInference:
    """Run inference with TFLite models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model"""
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info(f"TFLite model loaded from {self.model_path}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data
    
    def benchmark(self, input_data: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance"""
        import time
        
        # Warmup
        _ = self.predict(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.predict(input_data)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / num_runs
        throughput = 1000 / avg_time_ms
        
        return {
            "avg_inference_time_ms": avg_time_ms,
            "throughput_inferences_per_sec": throughput,
            "num_runs": num_runs
        }

# Example usage
def main():
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # Convert to TFLite
    converter = TFLiteConverter()
    tflite_model = converter.convert_model(model, quantization="int8")
    
    # Save model
    converter.save_model("model.tflite")
    
    # Analyze model
    analysis = converter.analyze_model()
    print("Model Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Test inference
    inference = TFLiteInference("model.tflite")
    test_input = np.random.randn(1, 5).astype(np.float32)
    output = inference.predict(test_input)
    print(f"Output: {output}")
    
    # Benchmark
    benchmark_results = inference.benchmark(test_input)
    print("Benchmark Results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Edge Impulse Platform**

### **Platform Overview**

#### **Key Features**
- **Data Collection**: Sensor data acquisition
- **Feature Engineering**: Automated feature extraction
- **Model Training**: Cloud-based training
- **Model Deployment**: Direct deployment to devices
- **Performance Monitoring**: Real-time model monitoring

#### **Code Example**

```python
import requests
import json
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeImpulseClient:
    """Edge Impulse API client"""
    
    def __init__(self, api_key: str, project_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = "https://studio.edgeimpulse.com/v1"
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
    
    def upload_data(self, data: np.ndarray, label: str, 
                   sensor_type: str = "accelerometer") -> Dict[str, Any]:
        """Upload sensor data to Edge Impulse"""
        
        # Convert data to Edge Impulse format
        payload = {
            "data": data.tolist(),
            "label": label,
            "sensor": sensor_type
        }
        
        # Upload data
        response = requests.post(
            f"{self.base_url}/projects/{self.project_id}/data",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            logger.info(f"Data uploaded successfully: {label}")
            return response.json()
        else:
            logger.error(f"Failed to upload data: {response.text}")
            return {"error": response.text}
    
    def train_model(self, model_type: str = "classification") -> Dict[str, Any]:
        """Train model on Edge Impulse"""
        
        payload = {
            "model_type": model_type,
            "target_device": "microcontroller"
        }
        
        response = requests.post(
            f"{self.base_url}/projects/{self.project_id}/train",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            logger.info("Model training started")
            return response.json()
        else:
            logger.error(f"Failed to start training: {response.text}")
            return {"error": response.text}
    
    def get_model(self) -> Dict[str, Any]:
        """Get trained model"""
        
        response = requests.get(
            f"{self.base_url}/projects/{self.project_id}/model",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get model: {response.text}")
            return {"error": response.text}
    
    def deploy_model(self, target_device: str = "arduino") -> Dict[str, Any]:
        """Deploy model to target device"""
        
        payload = {
            "target_device": target_device,
            "optimization": "speed"
        }
        
        response = requests.post(
            f"{self.base_url}/projects/{self.project_id}/deploy",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            logger.info(f"Model deployed to {target_device}")
            return response.json()
        else:
            logger.error(f"Failed to deploy model: {response.text}")
            return {"error": response.text}

# Example usage
def main():
    # Initialize Edge Impulse client
    client = EdgeImpulseClient("your-api-key", "your-project-id")
    
    # Generate sample sensor data
    sample_data = np.random.randn(100, 3)  # 3-axis accelerometer data
    
    # Upload data
    result = client.upload_data(sample_data, "walking", "accelerometer")
    print(f"Upload result: {result}")
    
    # Train model
    training_result = client.train_model("classification")
    print(f"Training result: {training_result}")
    
    # Get model
    model_info = client.get_model()
    print(f"Model info: {model_info}")
    
    # Deploy model
    deployment_result = client.deploy_model("arduino")
    print(f"Deployment result: {deployment_result}")

if __name__ == "__main__":
    main()
```

---

## 🔄 **Model Conversion Workflow**

### **Conversion Pipeline**

#### **Steps**
1. **Model Training**: Train in TensorFlow/PyTorch
2. **Model Optimization**: Apply quantization, pruning
3. **Format Conversion**: Convert to TFLite/ONNX
4. **Hardware Optimization**: Optimize for target hardware
5. **Deployment**: Deploy to target device

#### **Code Example**

```python
import tensorflow as tf
import onnx
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConversionPipeline:
    """Complete model conversion pipeline"""
    
    def __init__(self):
        self.model = None
        self.tflite_model = None
        self.onnx_model = None
    
    def train_model(self, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        """Train a simple model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Generate dummy data for training
        x_train = np.random.randn(1000, *input_shape)
        y_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 1000))
        
        # Train model
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        self.model = model
        return model
    
    def optimize_model(self, quantization: str = "int8") -> tf.keras.Model:
        """Apply model optimizations"""
        if self.model is None:
            raise ValueError("No model to optimize")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply optimizations
        if quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert to TFLite
        self.tflite_model = converter.convert()
        
        return self.model
    
    def convert_to_onnx(self) -> bytes:
        """Convert model to ONNX format"""
        if self.model is None:
            raise ValueError("No model to convert")
        
        # Save model in SavedModel format
        self.model.save("temp_model")
        
        # Convert to ONNX (simplified example)
        # In practice, you would use tf2onnx or similar tools
        logger.info("Model converted to ONNX format")
        
        return b"onnx_model_data"  # Placeholder
    
    def analyze_conversion(self) -> Dict[str, Any]:
        """Analyze conversion results"""
        results = {}
        
        if self.model is not None:
            results["original_model"] = {
                "parameters": self.model.count_params(),
                "size_kb": (self.model.count_params() * 4) / 1024
            }
        
        if self.tflite_model is not None:
            results["tflite_model"] = {
                "size_kb": len(self.tflite_model) / 1024,
                "compression_ratio": len(self.tflite_model) / (self.model.count_params() * 4) if self.model else 0
            }
        
        return results
    
    def validate_conversion(self, test_data: np.ndarray) -> Dict[str, Any]:
        """Validate conversion accuracy"""
        if self.model is None or self.tflite_model is None:
            raise ValueError("Models not available for validation")
        
        # Get original model predictions
        original_predictions = self.model.predict(test_data)
        
        # Get TFLite model predictions
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], test_data)
        interpreter.invoke()
        tflite_predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Calculate accuracy difference
        accuracy_diff = np.mean(np.abs(original_predictions - tflite_predictions))
        
        return {
            "accuracy_difference": accuracy_diff,
            "conversion_successful": accuracy_diff < 0.01,
            "original_predictions": original_predictions,
            "tflite_predictions": tflite_predictions
        }

# Example usage
def main():
    # Create conversion pipeline
    pipeline = ModelConversionPipeline()
    
    # Train model
    model = pipeline.train_model((10,), 3)
    print(f"Model trained with {model.count_params()} parameters")
    
    # Optimize model
    optimized_model = pipeline.optimize_model("int8")
    
    # Convert to ONNX
    onnx_model = pipeline.convert_to_onnx()
    
    # Analyze conversion
    analysis = pipeline.analyze_conversion()
    print("Conversion Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Validate conversion
    test_data = np.random.randn(5, 10)
    validation = pipeline.validate_conversion(test_data)
    print(f"Validation: {validation['conversion_successful']}")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Frameworks and Tools**

#### **Q1: What is TensorFlow Lite for Microcontrollers?**
**Answer**: 
- **Definition**: Lightweight ML framework for microcontrollers
- **Key Features**: 
  - Ultra-low memory footprint
  - No dynamic memory allocation
  - Optimized for microcontrollers
  - Support for quantization
- **Use Cases**: IoT devices, wearables, embedded systems

#### **Q2: How does Edge Impulse simplify TinyML development?**
**Answer**: 
- **Data Collection**: Easy sensor data acquisition
- **Feature Engineering**: Automated feature extraction
- **Model Training**: Cloud-based training with optimization
- **Deployment**: Direct deployment to target devices
- **Monitoring**: Real-time performance monitoring

#### **Q3: What are the key steps in model conversion?**
**Answer**: 
1. **Model Training**: Train in TensorFlow/PyTorch
2. **Optimization**: Apply quantization, pruning
3. **Format Conversion**: Convert to TFLite/ONNX
4. **Hardware Optimization**: Optimize for target hardware
5. **Validation**: Test accuracy and performance
6. **Deployment**: Deploy to target device

---

**Ready to learn about optimization techniques? Let's explore [Optimization Techniques](./OptimizationTechniques.md) next!** 🚀
