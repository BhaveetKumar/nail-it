# 💻 **Code Examples**

> **Practical Python and Go implementations for TinyML applications**

## 🎯 **Learning Objectives**

- Master Python training and model export for TinyML
- Learn Go inference implementations for edge devices
- Build end-to-end TinyML applications
- Understand model conversion and deployment
- Implement real-time inference pipelines

## 📚 **Table of Contents**

1. [Python Training Examples](#python-training-examples)
2. [Go Inference Examples](#go-inference-examples)
3. [Model Conversion Pipeline](#model-conversion-pipeline)
4. [Real-time Inference](#real-time-inference)
5. [Performance Optimization](#performance-optimization)

---

## 🐍 **Python Training Examples**

### **Image Classification Model**

```python
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyMLImageClassifier:
    """TinyML image classification model trainer"""
    
    def __init__(self, input_shape: tuple = (32, 32, 3), num_classes: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_model(self) -> tf.keras.Model:
        """Create a lightweight CNN model for TinyML"""
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.input_shape),
            
            # Convolutional layers
            tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            
            # Global average pooling instead of dense layers
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Output layer
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray, 
                   val_data: np.ndarray, val_labels: np.ndarray,
                   epochs: int = 50) -> tf.keras.Model:
        """Train the TinyML model"""
        
        # Create model
        self.model = self.create_model()
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return self.model
    
    def export_to_tflite(self, model: tf.keras.Model, 
                        quantization: str = "int8") -> bytes:
        """Export model to TensorFlow Lite format"""
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization
        if quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        tflite_model = converter.convert()
        
        return tflite_model

# Example usage
def main():
    # Generate sample data
    train_data = np.random.randn(1000, 32, 32, 3).astype(np.float32)
    train_labels = tf.keras.utils.to_categorical(np.random.randint(0, 10, 1000))
    
    val_data = np.random.randn(200, 32, 32, 3).astype(np.float32)
    val_labels = tf.keras.utils.to_categorical(np.random.randint(0, 10, 200))
    
    # Initialize trainer
    trainer = TinyMLImageClassifier()
    
    # Train model
    model = trainer.train_model(train_data, train_labels, val_data, val_labels, epochs=10)
    
    # Export to TFLite
    tflite_model = trainer.export_to_tflite(model, quantization="int8")
    
    print(f"Model exported to TFLite: {len(tflite_model)} bytes")
    print(f"Model parameters: {model.count_params()}")

if __name__ == "__main__":
    main()
```

### **Audio Classification Model**

```python
import librosa
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class TinyMLAudioClassifier:
    """TinyML audio classification model trainer"""
    
    def __init__(self, sampling_rate: int = 16000, duration: float = 1.0):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_mfcc = 13
        self.model = None
        
    def extract_mfcc_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        
        # Ensure audio is the right length
        if len(audio_data) != int(self.sampling_rate * self.duration):
            audio_data = librosa.util.fix_length(audio_data, int(self.sampling_rate * self.duration))
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=256
        )
        
        return mfccs.T  # Transpose to get time x features
    
    def create_model(self, input_shape: tuple) -> tf.keras.Model:
        """Create a lightweight model for audio classification"""
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # Convolutional layers
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Global average pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(5, activation='softmax')  # 5 audio classes
        ])
        
        return model
    
    def train_model(self, audio_data: List[np.ndarray], labels: List[int],
                   epochs: int = 50) -> tf.keras.Model:
        """Train the audio classification model"""
        
        # Extract features
        features = []
        for audio in audio_data:
            mfcc = self.extract_mfcc_features(audio)
            features.append(mfcc)
        
        # Convert to numpy array
        X = np.array(features)
        y = tf.keras.utils.to_categorical(labels, 5)
        
        # Reshape for model input
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Create model
        self.model = self.create_model(X.shape[1:])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
        
        return self.model

# Example usage
def main():
    # Generate sample audio data
    audio_data = []
    labels = []
    
    for i in range(100):
        # Generate synthetic audio
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        audio_data.append(audio)
        labels.append(i % 5)  # 5 classes
    
    # Initialize trainer
    trainer = TinyMLAudioClassifier()
    
    # Train model
    model = trainer.train_model(audio_data, labels, epochs=10)
    
    print(f"Audio model trained with {model.count_params()} parameters")

if __name__ == "__main__":
    main()
```

---

## 🐹 **Go Inference Examples**

### **TFLite Inference in Go**

```go
package main

import (
    "fmt"
    "log"
    "unsafe"
    
    "github.com/mattn/go-tflite"
)

// TinyMLInference handles TFLite model inference
type TinyMLInference struct {
    model     *tflite.Model
    interpreter *tflite.Interpreter
    inputTensor  *tflite.Tensor
    outputTensor *tflite.Tensor
}

// NewTinyMLInference creates a new inference instance
func NewTinyMLInference(modelPath string) (*TinyMLInference, error) {
    // Load model
    model := tflite.NewModelFromFile(modelPath)
    if model == nil {
        return nil, fmt.Errorf("failed to load model from %s", modelPath)
    }
    
    // Create interpreter options
    options := tflite.NewInterpreterOptions()
    options.SetNumThread(1) // Single thread for TinyML
    
    // Create interpreter
    interpreter := tflite.NewInterpreter(model, options)
    if interpreter == nil {
        return nil, fmt.Errorf("failed to create interpreter")
    }
    
    // Allocate tensors
    if status := interpreter.AllocateTensors(); status != tflite.OK {
        return nil, fmt.Errorf("failed to allocate tensors")
    }
    
    // Get input and output tensors
    inputTensor := interpreter.GetInputTensor(0)
    outputTensor := interpreter.GetOutputTensor(0)
    
    return &TinyMLInference{
        model:        model,
        interpreter:  interpreter,
        inputTensor:  inputTensor,
        outputTensor: outputTensor,
    }, nil
}

// Predict performs inference on input data
func (t *TinyMLInference) Predict(inputData []float32) ([]float32, error) {
    // Copy input data to tensor
    inputBytes := (*[1 << 30]byte)(unsafe.Pointer(&inputData[0]))[:len(inputData)*4]
    copy(t.inputTensor.ByteSlice(), inputBytes)
    
    // Run inference
    if status := t.interpreter.Invoke(); status != tflite.OK {
        return nil, fmt.Errorf("inference failed")
    }
    
    // Get output data
    outputBytes := t.outputTensor.ByteSlice()
    outputData := (*[1 << 30]float32)(unsafe.Pointer(&outputBytes[0]))[:len(outputBytes)/4]
    
    return outputData, nil
}

// Close releases resources
func (t *TinyMLInference) Close() {
    if t.interpreter != nil {
        t.interpreter.Delete()
    }
    if t.model != nil {
        t.model.Delete()
    }
}

// ImageClassification performs image classification
func (t *TinyMLInference) ImageClassification(imageData []float32) (int, float32, error) {
    // Run inference
    output, err := t.Predict(imageData)
    if err != nil {
        return 0, 0, err
    }
    
    // Find class with highest probability
    maxProb := float32(0)
    predictedClass := 0
    
    for i, prob := range output {
        if prob > maxProb {
            maxProb = prob
            predictedClass = i
        }
    }
    
    return predictedClass, maxProb, nil
}

// AudioClassification performs audio classification
func (t *TinyMLInference) AudioClassification(audioData []float32) (int, float32, error) {
    // Run inference
    output, err := t.Predict(audioData)
    if err != nil {
        return 0, 0, err
    }
    
    // Find class with highest probability
    maxProb := float32(0)
    predictedClass := 0
    
    for i, prob := range output {
        if prob > maxProb {
            maxProb = prob
            predictedClass = i
        }
    }
    
    return predictedClass, maxProb, nil
}

func main() {
    // Example usage
    inference, err := NewTinyMLInference("model.tflite")
    if err != nil {
        log.Fatal(err)
    }
    defer inference.Close()
    
    // Example image classification
    imageData := make([]float32, 32*32*3) // 32x32 RGB image
    for i := range imageData {
        imageData[i] = float32(i) / float32(len(imageData))
    }
    
    class, prob, err := inference.ImageClassification(imageData)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Predicted class: %d, Probability: %.3f\n", class, prob)
    
    // Example audio classification
    audioData := make([]float32, 63*13) // MFCC features
    for i := range audioData {
        audioData[i] = float32(i) / float32(len(audioData))
    }
    
    audioClass, audioProb, err := inference.AudioClassification(audioData)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Audio class: %d, Probability: %.3f\n", audioClass, audioProb)
}
```

### **Real-time Inference Pipeline**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

// InferencePipeline handles real-time inference
type InferencePipeline struct {
    inference    *TinyMLInference
    inputChan    chan []float32
    outputChan   chan InferenceResult
    ctx          context.Context
    cancel       context.CancelFunc
    wg           sync.WaitGroup
}

// InferenceResult contains inference results
type InferenceResult struct {
    Class       int
    Probability float32
    Timestamp   time.Time
    Error       error
}

// NewInferencePipeline creates a new inference pipeline
func NewInferencePipeline(modelPath string) (*InferencePipeline, error) {
    inference, err := NewTinyMLInference(modelPath)
    if err != nil {
        return nil, err
    }
    
    ctx, cancel := context.WithCancel(context.Background())
    
    return &InferencePipeline{
        inference:  inference,
        inputChan:  make(chan []float32, 10),
        outputChan: make(chan InferenceResult, 10),
        ctx:        ctx,
        cancel:     cancel,
    }, nil
}

// Start begins the inference pipeline
func (p *InferencePipeline) Start() {
    p.wg.Add(1)
    go p.processInference()
}

// Stop stops the inference pipeline
func (p *InferencePipeline) Stop() {
    p.cancel()
    p.wg.Wait()
    close(p.inputChan)
    close(p.outputChan)
    p.inference.Close()
}

// ProcessInference processes inference requests
func (p *InferencePipeline) processInference() {
    defer p.wg.Done()
    
    for {
        select {
        case <-p.ctx.Done():
            return
        case inputData := <-p.inputChan:
            // Run inference
            class, prob, err := p.inference.ImageClassification(inputData)
            
            // Send result
            result := InferenceResult{
                Class:       class,
                Probability: prob,
                Timestamp:   time.Now(),
                Error:       err,
            }
            
            select {
            case p.outputChan <- result:
            case <-p.ctx.Done():
                return
            }
        }
    }
}

// PredictAsync performs asynchronous inference
func (p *InferencePipeline) PredictAsync(inputData []float32) {
    select {
    case p.inputChan <- inputData:
    case <-p.ctx.Done():
    default:
        // Channel full, skip this inference
        log.Println("Inference pipeline full, skipping")
    }
}

// GetResult gets the next inference result
func (p *InferencePipeline) GetResult() (InferenceResult, bool) {
    select {
    case result := <-p.outputChan:
        return result, true
    case <-p.ctx.Done():
        return InferenceResult{}, false
    default:
        return InferenceResult{}, false
    }
}

// BenchmarkInference benchmarks inference performance
func (p *InferencePipeline) BenchmarkInference(inputData []float32, numRuns int) {
    start := time.Now()
    
    for i := 0; i < numRuns; i++ {
        p.PredictAsync(inputData)
    }
    
    // Collect results
    results := make([]InferenceResult, 0, numRuns)
    for i := 0; i < numRuns; i++ {
        if result, ok := p.GetResult(); ok {
            results = append(results, result)
        }
    }
    
    elapsed := time.Since(start)
    avgTime := elapsed / time.Duration(len(results))
    
    fmt.Printf("Benchmark Results:\n")
    fmt.Printf("  Total runs: %d\n", len(results))
    fmt.Printf("  Total time: %v\n", elapsed)
    fmt.Printf("  Average time: %v\n", avgTime)
    fmt.Printf("  Throughput: %.2f inferences/sec\n", float64(len(results))/elapsed.Seconds())
}

func main() {
    // Create inference pipeline
    pipeline, err := NewInferencePipeline("model.tflite")
    if err != nil {
        log.Fatal(err)
    }
    defer pipeline.Stop()
    
    // Start pipeline
    pipeline.Start()
    
    // Example usage
    inputData := make([]float32, 32*32*3)
    for i := range inputData {
        inputData[i] = float32(i) / float32(len(inputData))
    }
    
    // Benchmark inference
    pipeline.BenchmarkInference(inputData, 100)
}
```

---

## 🔄 **Model Conversion Pipeline**

### **Python to TFLite Conversion**

```python
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class ModelConverter:
    """Convert models to TinyML formats"""
    
    def __init__(self):
        self.conversion_configs = {}
    
    def convert_to_tflite(self, model: tf.keras.Model, 
                         quantization: str = "int8",
                         representative_dataset: Optional[List[np.ndarray]] = None) -> bytes:
        """Convert Keras model to TFLite format"""
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization
        if quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            
            # Use representative dataset for calibration
            if representative_dataset:
                def representative_data_gen():
                    for data in representative_dataset:
                        yield [data]
                converter.representative_dataset = representative_data_gen
        
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        tflite_model = converter.convert()
        
        return tflite_model
    
    def validate_conversion(self, original_model: tf.keras.Model, 
                          tflite_model: bytes, 
                          test_data: np.ndarray) -> Dict[str, Any]:
        """Validate TFLite conversion accuracy"""
        
        # Original model prediction
        original_pred = original_model.predict(test_data)
        
        # TFLite model prediction
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        tflite_pred = []
        for sample in test_data:
            interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            tflite_pred.append(output[0])
        
        tflite_pred = np.array(tflite_pred)
        
        # Calculate accuracy difference
        mse = np.mean((original_pred - tflite_pred) ** 2)
        mae = np.mean(np.abs(original_pred - tflite_pred))
        
        return {
            "mse": mse,
            "mae": mae,
            "original_size": original_model.count_params() * 4,  # Float32
            "tflite_size": len(tflite_model),
            "compression_ratio": (original_model.count_params() * 4) / len(tflite_model)
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
    
    # Convert model
    converter = ModelConverter()
    tflite_model = converter.convert_to_tflite(model, quantization="int8")
    
    # Validate conversion
    validation = converter.validate_conversion(model, tflite_model, test_data[:10])
    
    print(f"Conversion validation:")
    print(f"  MSE: {validation['mse']:.6f}")
    print(f"  MAE: {validation['mae']:.6f}")
    print(f"  Compression ratio: {validation['compression_ratio']:.2f}x")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Code Examples**

#### **Q1: How do you implement real-time inference in Go for TinyML?**
**Answer**: 
- **TFLite Integration**: Use go-tflite library for model loading
- **Memory Management**: Efficient tensor allocation and deallocation
- **Pipeline Design**: Asynchronous processing with channels
- **Error Handling**: Robust error handling for edge cases
- **Performance**: Single-threaded inference for resource constraints
- **Buffering**: Input/output buffering for continuous processing

#### **Q2: What are the key considerations when converting models to TFLite?**
**Answer**: 
- **Quantization**: Choose appropriate quantization (int8, float16)
- **Representative Dataset**: Use calibration data for quantization
- **Model Validation**: Compare accuracy before/after conversion
- **Size Optimization**: Balance model size vs accuracy
- **Hardware Compatibility**: Ensure target hardware support
- **Performance Testing**: Benchmark inference speed and memory usage

#### **Q3: How do you handle real-time data processing in TinyML applications?**
**Answer**: 
- **Streaming Processing**: Process data in windows/batches
- **Buffer Management**: Efficient memory management for continuous data
- **Latency Optimization**: Minimize processing delays
- **Error Recovery**: Handle sensor failures and data corruption
- **Resource Monitoring**: Monitor memory and CPU usage
- **Adaptive Processing**: Adjust processing based on available resources

---

**Ready to explore hardware and deployment? Let's dive into [Hardware and Deployment](./HardwareAndDeployment.md) next!** 🚀
