# 🎯 **TinyML Use Cases**

> **Explore real-world TinyML applications: health monitoring, voice wake-up, gesture recognition, vision, and predictive maintenance**

## 🎯 **Learning Objectives**

- Understand real-world TinyML applications across industries
- Learn implementation patterns for common use cases
- Master sensor data processing and feature extraction
- Build end-to-end TinyML applications
- Explore industry-specific challenges and solutions

## 📚 **Table of Contents**

1. [Healthcare and Medical Devices](#healthcare-and-medical-devices)
2. [Voice and Audio Processing](#voice-and-audio-processing)
3. [Gesture and Motion Recognition](#gesture-and-motion-recognition)
4. [Computer Vision on Edge](#computer-vision-on-edge)
5. [Predictive Maintenance](#predictive-maintenance)
6. [Environmental Monitoring](#environmental-monitoring)
7. [Smart Agriculture](#smart-agriculture)
8. [Case Studies](#case-studies)

---

## 🏥 **Healthcare and Medical Devices**

### **Heart Rate Monitoring**

#### **Concept**
Continuous heart rate monitoring using photoplethysmography (PPG) sensors to detect heart rate variability and anomalies.

#### **Technical Implementation**
- **Sensor**: PPG sensor with LED and photodiode
- **Sampling Rate**: 100-1000 Hz
- **Processing**: Peak detection, filtering, heart rate calculation
- **Output**: Heart rate (BPM), heart rate variability (HRV)

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartRateMonitor:
    """TinyML-based heart rate monitoring system"""
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.buffer_size = 1000  # 10 seconds at 100Hz
        self.ppg_buffer = []
        self.heart_rate_history = []
        
    def preprocess_ppg_signal(self, ppg_data: np.ndarray) -> np.ndarray:
        """Preprocess PPG signal for heart rate detection"""
        
        # Remove DC component
        ppg_data = ppg_data - np.mean(ppg_data)
        
        # Apply bandpass filter (0.5-5 Hz for heart rate)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 5.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, ppg_data)
        
        # Normalize signal
        filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
        
        return filtered_signal
    
    def detect_peaks(self, signal: np.ndarray, threshold: float = 0.3) -> List[int]:
        """Detect R-peaks in PPG signal"""
        
        # Find peaks using scipy
        peaks, _ = signal.find_peaks(signal, height=threshold, distance=self.sampling_rate//3)
        
        return peaks.tolist()
    
    def calculate_heart_rate(self, peaks: List[int], window_size: int = 30) -> float:
        """Calculate heart rate from peak intervals"""
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate intervals between peaks
        intervals = np.diff(peaks) / self.sampling_rate  # Convert to seconds
        
        # Filter out unrealistic intervals (20-200 BPM)
        valid_intervals = intervals[(intervals > 0.3) & (intervals < 3.0)]
        
        if len(valid_intervals) == 0:
            return 0.0
        
        # Calculate average heart rate
        avg_interval = np.mean(valid_intervals)
        heart_rate = 60.0 / avg_interval
        
        return heart_rate
    
    def detect_anomalies(self, heart_rate: float, threshold_std: float = 2.0) -> Dict[str, Any]:
        """Detect heart rate anomalies"""
        
        if len(self.heart_rate_history) < 10:
            self.heart_rate_history.append(heart_rate)
            return {"anomaly": False, "confidence": 0.0}
        
        # Calculate statistics
        mean_hr = np.mean(self.heart_rate_history)
        std_hr = np.std(self.heart_rate_history)
        
        # Check for anomalies
        z_score = abs(heart_rate - mean_hr) / std_hr if std_hr > 0 else 0
        is_anomaly = z_score > threshold_std
        
        # Update history
        self.heart_rate_history.append(heart_rate)
        if len(self.heart_rate_history) > 100:  # Keep last 100 readings
            self.heart_rate_history.pop(0)
        
        return {
            "anomaly": is_anomaly,
            "confidence": min(z_score / threshold_std, 1.0),
            "z_score": z_score,
            "mean_hr": mean_hr,
            "std_hr": std_hr
        }
    
    def process_ppg_data(self, ppg_data: np.ndarray) -> Dict[str, Any]:
        """Process PPG data and extract heart rate information"""
        
        # Preprocess signal
        filtered_signal = self.preprocess_ppg_signal(ppg_data)
        
        # Detect peaks
        peaks = self.detect_peaks(filtered_signal)
        
        # Calculate heart rate
        heart_rate = self.calculate_heart_rate(peaks)
        
        # Detect anomalies
        anomaly_info = self.detect_anomalies(heart_rate)
        
        # Calculate heart rate variability
        hrv = self.calculate_hrv(peaks)
        
        return {
            "heart_rate": heart_rate,
            "hrv": hrv,
            "peaks": peaks,
            "filtered_signal": filtered_signal,
            "anomaly": anomaly_info
        }
    
    def calculate_hrv(self, peaks: List[int]) -> float:
        """Calculate heart rate variability (RMSSD)"""
        
        if len(peaks) < 3:
            return 0.0
        
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / self.sampling_rate
        
        # Filter realistic intervals
        valid_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 3.0)]
        
        if len(valid_intervals) < 2:
            return 0.0
        
        # Calculate RMSSD
        rr_diffs = np.diff(valid_intervals)
        rmssd = np.sqrt(np.mean(rr_diffs**2))
        
        return rmssd * 1000  # Convert to milliseconds

class FallDetection:
    """TinyML-based fall detection system"""
    
    def __init__(self, sampling_rate: int = 50):
        self.sampling_rate = sampling_rate
        self.accel_buffer = []
        self.gyro_buffer = []
        self.fall_threshold = 2.5  # g
        self.impact_threshold = 3.0  # g
        
    def preprocess_imu_data(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess IMU data for fall detection"""
        
        # Remove gravity component from accelerometer
        accel_data = accel_data - np.mean(accel_data, axis=0)
        
        # Apply low-pass filter to reduce noise
        nyquist = self.sampling_rate / 2
        cutoff = 5.0 / nyquist
        
        b, a = signal.butter(4, cutoff, btype='low')
        filtered_accel = signal.filtfilt(b, a, accel_data, axis=0)
        filtered_gyro = signal.filtfilt(b, a, gyro_data, axis=0)
        
        return filtered_accel, filtered_gyro
    
    def extract_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict[str, float]:
        """Extract features for fall detection"""
        
        # Calculate magnitude
        accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
        gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
        
        # Statistical features
        features = {
            "accel_max": np.max(accel_magnitude),
            "accel_mean": np.mean(accel_magnitude),
            "accel_std": np.std(accel_magnitude),
            "gyro_max": np.max(gyro_magnitude),
            "gyro_mean": np.mean(gyro_magnitude),
            "gyro_std": np.std(gyro_magnitude),
            "accel_range": np.max(accel_magnitude) - np.min(accel_magnitude),
            "gyro_range": np.max(gyro_magnitude) - np.min(gyro_magnitude)
        }
        
        return features
    
    def detect_fall(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict[str, Any]:
        """Detect fall using IMU data"""
        
        # Preprocess data
        filtered_accel, filtered_gyro = self.preprocess_imu_data(accel_data, gyro_data)
        
        # Extract features
        features = self.extract_features(filtered_accel, filtered_gyro)
        
        # Simple rule-based fall detection
        is_fall = (
            features["accel_max"] > self.fall_threshold or
            features["gyro_max"] > self.impact_threshold or
            features["accel_range"] > 4.0
        )
        
        # Calculate confidence
        confidence = min(
            features["accel_max"] / self.fall_threshold,
            features["gyro_max"] / self.impact_threshold,
            1.0
        )
        
        return {
            "is_fall": is_fall,
            "confidence": confidence,
            "features": features,
            "accel_magnitude": np.sqrt(np.sum(filtered_accel**2, axis=1)),
            "gyro_magnitude": np.sqrt(np.sum(filtered_gyro**2, axis=1))
        }

# Example usage
def main():
    # Simulate PPG data for heart rate monitoring
    t = np.linspace(0, 10, 1000)  # 10 seconds at 100Hz
    heart_rate = 75  # BPM
    ppg_freq = heart_rate / 60.0  # Hz
    
    # Generate synthetic PPG signal
    ppg_signal = np.sin(2 * np.pi * ppg_freq * t) + 0.1 * np.random.randn(len(t))
    
    # Initialize heart rate monitor
    hr_monitor = HeartRateMonitor(sampling_rate=100)
    
    # Process PPG data
    hr_result = hr_monitor.process_ppg_data(ppg_signal)
    
    print("Heart Rate Monitoring Results:")
    print(f"  Heart Rate: {hr_result['heart_rate']:.1f} BPM")
    print(f"  HRV: {hr_result['hrv']:.1f} ms")
    print(f"  Anomaly: {hr_result['anomaly']['anomaly']}")
    print(f"  Confidence: {hr_result['anomaly']['confidence']:.2f}")
    
    # Simulate IMU data for fall detection
    t_imu = np.linspace(0, 2, 100)  # 2 seconds at 50Hz
    
    # Generate synthetic IMU data
    accel_data = np.random.randn(len(t_imu), 3) * 0.1
    gyro_data = np.random.randn(len(t_imu), 3) * 0.05
    
    # Add fall event
    fall_start = 50
    fall_end = 60
    accel_data[fall_start:fall_end, 2] += 3.0  # Z-axis impact
    
    # Initialize fall detection
    fall_detector = FallDetection(sampling_rate=50)
    
    # Detect fall
    fall_result = fall_detector.detect_fall(accel_data, gyro_data)
    
    print("\nFall Detection Results:")
    print(f"  Fall Detected: {fall_result['is_fall']}")
    print(f"  Confidence: {fall_result['confidence']:.2f}")
    print(f"  Max Acceleration: {fall_result['features']['accel_max']:.2f} g")
    print(f"  Max Gyroscope: {fall_result['features']['gyro_max']:.2f} rad/s")

if __name__ == "__main__":
    main()
```

---

## 🎤 **Voice and Audio Processing**

### **Wake Word Detection**

#### **Concept**
Detect specific wake words (like "Hey Google", "Alexa") using audio processing and machine learning on edge devices.

#### **Technical Implementation**
- **Audio Input**: Microphone with 16kHz sampling rate
- **Preprocessing**: Noise reduction, voice activity detection
- **Feature Extraction**: MFCC, spectrograms, or raw audio
- **Model**: Small CNN or RNN for classification

#### **Code Example**

```python
import numpy as np
import librosa
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """TinyML-based wake word detection system"""
    
    def __init__(self, sampling_rate: int = 16000, window_size: int = 16000):
        self.sampling_rate = sampling_rate
        self.window_size = window_size  # 1 second at 16kHz
        self.mfcc_features = 13
        self.audio_buffer = []
        
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio for wake word detection"""
        
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Apply window function
        windowed_audio = emphasized_audio * np.hamming(len(emphasized_audio))
        
        return windowed_audio
    
    def extract_mfcc_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_data)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=processed_audio,
            sr=self.sampling_rate,
            n_mfcc=self.mfcc_features,
            n_fft=512,
            hop_length=256
        )
        
        return mfccs.T  # Transpose to get time x features
    
    def extract_spectrogram_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract spectrogram features from audio"""
        
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_data)
        
        # Compute spectrogram
        stft = librosa.stft(processed_audio, n_fft=512, hop_length=256)
        spectrogram = np.abs(stft)
        
        # Convert to log scale
        log_spectrogram = np.log(spectrogram + 1e-8)
        
        return log_spectrogram.T  # Transpose to get time x frequency
    
    def create_wake_word_model(self, input_shape: tuple) -> tf.keras.Model:
        """Create a small CNN model for wake word detection"""
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # Convolutional layers
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Global average pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation='softmax')  # Wake word or not
        ])
        
        return model
    
    def detect_wake_word(self, audio_data: np.ndarray, model: tf.keras.Model) -> Dict[str, Any]:
        """Detect wake word in audio data"""
        
        # Extract features
        mfcc_features = self.extract_mfcc_features(audio_data)
        
        # Reshape for model input
        if len(mfcc_features.shape) == 2:
            mfcc_features = mfcc_features.reshape(1, *mfcc_features.shape, 1)
        
        # Make prediction
        prediction = model.predict(mfcc_features)
        wake_word_prob = prediction[0][1]  # Probability of wake word
        
        # Apply threshold
        threshold = 0.7
        is_wake_word = wake_word_prob > threshold
        
        return {
            "is_wake_word": is_wake_word,
            "confidence": wake_word_prob,
            "threshold": threshold,
            "features": mfcc_features
        }

class NoiseClassification:
    """TinyML-based noise classification system"""
    
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        self.noise_types = ["silence", "speech", "music", "noise", "traffic"]
        
    def extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract features for noise classification"""
        
        # Preprocess audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Extract various audio features
        features = {}
        
        # Time domain features
        features["rms"] = np.sqrt(np.mean(audio_data**2))
        features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sampling_rate))
        features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=self.sampling_rate))
        features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sampling_rate))
        
        # Frequency domain features
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        features["spectral_contrast"] = np.mean(librosa.feature.spectral_contrast(S=magnitude))
        features["mfcc_mean"] = np.mean(librosa.feature.mfcc(y=audio_data, sr=self.sampling_rate))
        features["mfcc_std"] = np.std(librosa.feature.mfcc(y=audio_data, sr=self.sampling_rate))
        
        return features
    
    def classify_noise(self, audio_data: np.ndarray, model: tf.keras.Model) -> Dict[str, Any]:
        """Classify noise type in audio data"""
        
        # Extract features
        features = self.extract_audio_features(audio_data)
        
        # Convert to array for model input
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_array)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            "predicted_class": self.noise_types[predicted_class],
            "confidence": confidence,
            "all_probabilities": dict(zip(self.noise_types, prediction[0])),
            "features": features
        }

# Example usage
def main():
    # Simulate audio data for wake word detection
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(16000 * duration))
    
    # Generate synthetic wake word signal
    wake_word_signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # Initialize wake word detector
    wake_detector = WakeWordDetector(sampling_rate=16000)
    
    # Create model
    model = wake_detector.create_wake_word_model((63, 13, 1))  # MFCC shape
    
    # Detect wake word
    wake_result = wake_detector.detect_wake_word(wake_word_signal, model)
    
    print("Wake Word Detection Results:")
    print(f"  Wake Word Detected: {wake_result['is_wake_word']}")
    print(f"  Confidence: {wake_result['confidence']:.3f}")
    print(f"  Threshold: {wake_result['threshold']}")
    
    # Simulate audio data for noise classification
    noise_signal = np.random.randn(16000) * 0.1  # Random noise
    
    # Initialize noise classifier
    noise_classifier = NoiseClassification(sampling_rate=16000)
    
    # Create simple model for noise classification
    noise_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 noise types
    ])
    
    # Classify noise
    noise_result = noise_classifier.classify_noise(noise_signal, noise_model)
    
    print("\nNoise Classification Results:")
    print(f"  Predicted Class: {noise_result['predicted_class']}")
    print(f"  Confidence: {noise_result['confidence']:.3f}")
    print(f"  All Probabilities: {noise_result['all_probabilities']}")

if __name__ == "__main__":
    main()
```

---

## 🤲 **Gesture and Motion Recognition**

### **Hand Gesture Recognition**

#### **Concept**
Recognize hand gestures using accelerometer and gyroscope data from smartwatches or mobile devices.

#### **Technical Implementation**
- **Sensors**: 6-axis IMU (accelerometer + gyroscope)
- **Sampling Rate**: 50-100 Hz
- **Features**: Statistical features, frequency domain features
- **Model**: LSTM or CNN for sequence classification

#### **Code Example**

```python
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureRecognizer:
    """TinyML-based gesture recognition system"""
    
    def __init__(self, sampling_rate: int = 50, window_size: int = 100):
        self.sampling_rate = sampling_rate
        self.window_size = window_size  # 2 seconds at 50Hz
        self.gesture_types = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "tap", "double_tap", "no_gesture"]
        
    def preprocess_imu_data(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess IMU data for gesture recognition"""
        
        # Remove gravity component
        accel_data = accel_data - np.mean(accel_data, axis=0)
        
        # Apply low-pass filter
        from scipy import signal
        nyquist = self.sampling_rate / 2
        cutoff = 10.0 / nyquist
        
        b, a = signal.butter(4, cutoff, btype='low')
        filtered_accel = signal.filtfilt(b, a, accel_data, axis=0)
        filtered_gyro = signal.filtfilt(b, a, gyro_data, axis=0)
        
        return filtered_accel, filtered_gyro
    
    def extract_gesture_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> np.ndarray:
        """Extract features for gesture recognition"""
        
        # Preprocess data
        filtered_accel, filtered_gyro = self.preprocess_imu_data(accel_data, gyro_data)
        
        # Calculate magnitude
        accel_magnitude = np.sqrt(np.sum(filtered_accel**2, axis=1))
        gyro_magnitude = np.sqrt(np.sum(filtered_gyro**2, axis=1))
        
        # Statistical features
        features = []
        
        # Accelerometer features
        features.extend([
            np.mean(accel_magnitude),
            np.std(accel_magnitude),
            np.max(accel_magnitude),
            np.min(accel_magnitude),
            np.percentile(accel_magnitude, 25),
            np.percentile(accel_magnitude, 75)
        ])
        
        # Gyroscope features
        features.extend([
            np.mean(gyro_magnitude),
            np.std(gyro_magnitude),
            np.max(gyro_magnitude),
            np.min(gyro_magnitude),
            np.percentile(gyro_magnitude, 25),
            np.percentile(gyro_magnitude, 75)
        ])
        
        # Directional features
        for axis in range(3):
            features.extend([
                np.mean(filtered_accel[:, axis]),
                np.std(filtered_accel[:, axis]),
                np.max(filtered_accel[:, axis]),
                np.min(filtered_accel[:, axis])
            ])
        
        # Frequency domain features
        fft_accel = np.fft.fft(accel_magnitude)
        fft_gyro = np.fft.fft(gyro_magnitude)
        
        features.extend([
            np.mean(np.abs(fft_accel[:len(fft_accel)//2])),
            np.std(np.abs(fft_accel[:len(fft_accel)//2])),
            np.mean(np.abs(fft_gyro[:len(fft_gyro)//2])),
            np.std(np.abs(fft_gyro[:len(fft_gyro)//2]))
        ])
        
        return np.array(features)
    
    def create_gesture_model(self, input_shape: tuple) -> tf.keras.Model:
        """Create a model for gesture recognition"""
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # LSTM layers for sequence processing
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(16, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            
            # Dense layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.gesture_types), activation='softmax')
        ])
        
        return model
    
    def recognize_gesture(self, accel_data: np.ndarray, gyro_data: np.ndarray, model: tf.keras.Model) -> Dict[str, Any]:
        """Recognize gesture from IMU data"""
        
        # Ensure data has correct shape
        if len(accel_data) != self.window_size or len(gyro_data) != self.window_size:
            # Pad or truncate data
            if len(accel_data) < self.window_size:
                accel_data = np.pad(accel_data, ((0, self.window_size - len(accel_data)), (0, 0)), mode='constant')
                gyro_data = np.pad(gyro_data, ((0, self.window_size - len(gyro_data)), (0, 0)), mode='constant')
            else:
                accel_data = accel_data[:self.window_size]
                gyro_data = gyro_data[:self.window_size]
        
        # Combine accel and gyro data
        combined_data = np.concatenate([accel_data, gyro_data], axis=1)
        
        # Reshape for model input
        input_data = combined_data.reshape(1, self.window_size, 6)
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            "gesture": self.gesture_types[predicted_class],
            "confidence": confidence,
            "all_probabilities": dict(zip(self.gesture_types, prediction[0])),
            "input_data": input_data
        }

class ActivityRecognition:
    """TinyML-based activity recognition system"""
    
    def __init__(self, sampling_rate: int = 50):
        self.sampling_rate = sampling_rate
        self.activities = ["walking", "running", "sitting", "standing", "lying", "cycling"]
        
    def extract_activity_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict[str, float]:
        """Extract features for activity recognition"""
        
        # Calculate magnitude
        accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
        gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
        
        # Statistical features
        features = {}
        
        # Time domain features
        features["accel_mean"] = np.mean(accel_magnitude)
        features["accel_std"] = np.std(accel_magnitude)
        features["accel_max"] = np.max(accel_magnitude)
        features["accel_min"] = np.min(accel_magnitude)
        features["accel_range"] = features["accel_max"] - features["accel_min"]
        
        features["gyro_mean"] = np.mean(gyro_magnitude)
        features["gyro_std"] = np.std(gyro_magnitude)
        features["gyro_max"] = np.max(gyro_magnitude)
        features["gyro_min"] = np.min(gyro_magnitude)
        features["gyro_range"] = features["gyro_max"] - features["gyro_min"]
        
        # Frequency domain features
        fft_accel = np.fft.fft(accel_magnitude)
        fft_gyro = np.fft.fft(gyro_magnitude)
        
        features["accel_energy"] = np.sum(np.abs(fft_accel)**2)
        features["gyro_energy"] = np.sum(np.abs(fft_gyro)**2)
        
        # Dominant frequency
        freqs = np.fft.fftfreq(len(accel_magnitude), 1/self.sampling_rate)
        accel_spectrum = np.abs(fft_accel[:len(fft_accel)//2])
        gyro_spectrum = np.abs(fft_gyro[:len(fft_gyro)//2])
        
        features["accel_dominant_freq"] = freqs[np.argmax(accel_spectrum)]
        features["gyro_dominant_freq"] = freqs[np.argmax(gyro_spectrum)]
        
        return features
    
    def recognize_activity(self, accel_data: np.ndarray, gyro_data: np.ndarray, model: tf.keras.Model) -> Dict[str, Any]:
        """Recognize activity from IMU data"""
        
        # Extract features
        features = self.extract_activity_features(accel_data, gyro_data)
        
        # Convert to array for model input
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_array)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            "activity": self.activities[predicted_class],
            "confidence": confidence,
            "all_probabilities": dict(zip(self.activities, prediction[0])),
            "features": features
        }

# Example usage
def main():
    # Simulate IMU data for gesture recognition
    window_size = 100  # 2 seconds at 50Hz
    t = np.linspace(0, 2, window_size)
    
    # Generate synthetic gesture data (swipe right)
    accel_data = np.random.randn(window_size, 3) * 0.1
    gyro_data = np.random.randn(window_size, 3) * 0.05
    
    # Add swipe gesture
    accel_data[:, 0] += np.sin(2 * np.pi * t) * 0.5  # X-axis movement
    gyro_data[:, 2] += np.cos(2 * np.pi * t) * 0.3  # Z-axis rotation
    
    # Initialize gesture recognizer
    gesture_recognizer = GestureRecognizer(sampling_rate=50, window_size=window_size)
    
    # Create model
    model = gesture_recognizer.create_gesture_model((window_size, 6))
    
    # Recognize gesture
    gesture_result = gesture_recognizer.recognize_gesture(accel_data, gyro_data, model)
    
    print("Gesture Recognition Results:")
    print(f"  Gesture: {gesture_result['gesture']}")
    print(f"  Confidence: {gesture_result['confidence']:.3f}")
    print(f"  All Probabilities: {gesture_result['all_probabilities']}")
    
    # Simulate IMU data for activity recognition
    activity_accel = np.random.randn(100, 3) * 0.1
    activity_gyro = np.random.randn(100, 3) * 0.05
    
    # Add walking pattern
    activity_accel[:, 2] += np.sin(4 * np.pi * t) * 0.3  # Vertical movement
    
    # Initialize activity recognizer
    activity_recognizer = ActivityRecognition(sampling_rate=50)
    
    # Create simple model for activity recognition
    activity_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(12,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # 6 activities
    ])
    
    # Recognize activity
    activity_result = activity_recognizer.recognize_activity(activity_accel, activity_gyro, activity_model)
    
    print("\nActivity Recognition Results:")
    print(f"  Activity: {activity_result['activity']}")
    print(f"  Confidence: {activity_result['confidence']:.3f}")
    print(f"  All Probabilities: {activity_result['all_probabilities']}")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Use Cases**

#### **Q1: How would you implement a heart rate monitoring system using TinyML?**
**Answer**: 
- **Sensor**: PPG sensor with LED and photodiode
- **Preprocessing**: Bandpass filter (0.5-5 Hz), DC removal, normalization
- **Peak Detection**: Find R-peaks using threshold and distance constraints
- **Heart Rate Calculation**: Convert peak intervals to BPM
- **Anomaly Detection**: Use statistical methods (z-score) to detect irregularities
- **Optimization**: Use quantized models, efficient peak detection algorithms

#### **Q2: What are the challenges in implementing wake word detection on edge devices?**
**Answer**: 
- **Power Consumption**: Continuous audio processing drains battery
- **False Positives**: Background noise can trigger false wake words
- **Latency**: Must respond quickly (< 100ms) to user commands
- **Model Size**: Must fit in limited memory while maintaining accuracy
- **Noise Robustness**: Must work in various acoustic environments
- **Privacy**: Processing must be done locally without cloud connectivity

#### **Q3: How do you handle gesture recognition with limited sensor data?**
**Answer**: 
- **Feature Engineering**: Extract statistical and frequency domain features
- **Data Augmentation**: Generate synthetic gestures for training
- **Model Architecture**: Use LSTM or CNN for sequence classification
- **Preprocessing**: Filter noise, remove gravity, normalize data
- **Thresholding**: Use confidence thresholds to reduce false positives
- **Context Awareness**: Consider user behavior patterns and device orientation

#### **Q4: What are the key considerations for deploying TinyML in healthcare applications?**
**Answer**: 
- **Accuracy**: High accuracy required for medical decisions
- **Reliability**: Must work consistently in various conditions
- **Privacy**: Patient data must be processed locally
- **Regulatory Compliance**: Must meet medical device standards
- **Power Efficiency**: Long battery life for continuous monitoring
- **Calibration**: Regular calibration for sensor drift
- **Fallback Mechanisms**: Cloud connectivity for critical alerts

#### **Q5: How would you optimize a TinyML model for real-time performance?**
**Answer**: 
- **Model Architecture**: Use depthwise separable convolutions, global average pooling
- **Quantization**: Convert to int8 for faster inference
- **Pruning**: Remove unnecessary weights and neurons
- **Knowledge Distillation**: Train smaller student models
- **Hardware Optimization**: Use specialized accelerators (Edge TPU, Neural Engine)
- **Pipeline Optimization**: Parallel processing, efficient memory management
- **Profiling**: Identify bottlenecks and optimize critical paths

---

**Ready to explore code examples and implementations? Let's dive into [Code Examples](./CodeExamples.md) next!** 🚀
