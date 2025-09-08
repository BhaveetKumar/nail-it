# 🏗️ **System Design**

> **Design end-to-end TinyML systems: architecture, privacy, scalability, and production patterns**

## 🎯 **Learning Objectives**

- Master TinyML system architecture patterns
- Design privacy-preserving ML systems
- Build scalable edge AI pipelines
- Implement production-ready TinyML systems
- Understand distributed TinyML architectures

## 📚 **Table of Contents**

1. [System Architecture Patterns](#system-architecture-patterns)
2. [Privacy-Preserving ML](#privacy-preserving-ml)
3. [Scalable Edge AI](#scalable-edge-ai)
4. [Production Patterns](#production-patterns)
5. [Case Studies](#case-studies)

---

## 🏗️ **System Architecture Patterns**

### **Edge-First Architecture**

#### **Concept**
Design systems where AI inference happens at the edge, with minimal cloud dependency for real-time processing.

#### **Architecture Components**
- **Edge Devices**: MCUs, smartphones, IoT sensors
- **Edge Gateway**: Local processing and coordination
- **Cloud Backend**: Model training, updates, analytics
- **Communication Layer**: MQTT, HTTP, WebSocket

#### **Code Example: Edge-First System**

```python
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    UPDATING = "updating"

@dataclass
class DeviceInfo:
    device_id: str
    status: DeviceStatus
    last_seen: float
    model_version: str
    inference_count: int
    error_count: int

class EdgeDevice:
    """Represents an edge device in the TinyML system"""
    
    def __init__(self, device_id: str, model_path: str):
        self.device_id = device_id
        self.model_path = model_path
        self.status = DeviceStatus.ONLINE
        self.model_version = "1.0.0"
        self.inference_count = 0
        self.error_count = 0
        self.last_seen = time.time()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the TinyML model"""
        try:
            # Simulate model loading
            self.model = f"Model loaded from {self.model_path}"
            logger.info(f"Device {self.device_id} loaded model {self.model_version}")
        except Exception as e:
            logger.error(f"Failed to load model for device {self.device_id}: {e}")
            self.status = DeviceStatus.ERROR
            self.error_count += 1
    
    def run_inference(self, input_data: List[float]) -> Dict[str, Any]:
        """Run inference on the device"""
        try:
            if self.status != DeviceStatus.ONLINE:
                raise Exception("Device not available")
            
            # Simulate inference
            start_time = time.time()
            
            # Mock inference result
            result = {
                "prediction": [0.1, 0.2, 0.7],  # Mock probabilities
                "confidence": 0.7,
                "inference_time_ms": (time.time() - start_time) * 1000,
                "device_id": self.device_id,
                "timestamp": time.time()
            }
            
            self.inference_count += 1
            self.last_seen = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed on device {self.device_id}: {e}")
            self.error_count += 1
            return {"error": str(e), "device_id": self.device_id}
    
    def update_model(self, new_model_path: str, new_version: str):
        """Update the model on the device"""
        try:
            self.status = DeviceStatus.UPDATING
            logger.info(f"Updating model on device {self.device_id} to version {new_version}")
            
            # Simulate model update
            time.sleep(0.1)  # Mock update time
            
            self.model_path = new_model_path
            self.model_version = new_version
            self.load_model()
            
            self.status = DeviceStatus.ONLINE
            logger.info(f"Model updated successfully on device {self.device_id}")
            
        except Exception as e:
            logger.error(f"Model update failed on device {self.device_id}: {e}")
            self.status = DeviceStatus.ERROR
            self.error_count += 1
    
    def get_device_info(self) -> DeviceInfo:
        """Get device information"""
        return DeviceInfo(
            device_id=self.device_id,
            status=self.status,
            last_seen=self.last_seen,
            model_version=self.model_version,
            inference_count=self.inference_count,
            error_count=self.error_count
        )

class EdgeGateway:
    """Edge gateway for coordinating multiple devices"""
    
    def __init__(self, gateway_id: str):
        self.gateway_id = gateway_id
        self.devices: Dict[str, EdgeDevice] = {}
        self.inference_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.running = False
    
    def add_device(self, device: EdgeDevice):
        """Add a device to the gateway"""
        self.devices[device.device_id] = device
        logger.info(f"Added device {device.device_id} to gateway {self.gateway_id}")
    
    def remove_device(self, device_id: str):
        """Remove a device from the gateway"""
        if device_id in self.devices:
            del self.devices[device_id]
            logger.info(f"Removed device {device_id} from gateway {self.gateway_id}")
    
    async def process_inference_requests(self):
        """Process inference requests from the queue"""
        while self.running:
            try:
                # Get request from queue
                request = await asyncio.wait_for(self.inference_queue.get(), timeout=1.0)
                
                # Find available device
                available_device = self.find_available_device()
                if available_device:
                    # Run inference
                    result = available_device.run_inference(request["input_data"])
                    result["request_id"] = request["request_id"]
                    
                    # Put result in results queue
                    await self.results_queue.put(result)
                else:
                    # No available devices
                    error_result = {
                        "error": "No available devices",
                        "request_id": request["request_id"],
                        "timestamp": time.time()
                    }
                    await self.results_queue.put(error_result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing inference request: {e}")
    
    def find_available_device(self) -> Optional[EdgeDevice]:
        """Find an available device for inference"""
        for device in self.devices.values():
            if device.status == DeviceStatus.ONLINE:
                return device
        return None
    
    async def submit_inference_request(self, input_data: List[float], request_id: str):
        """Submit an inference request"""
        request = {
            "input_data": input_data,
            "request_id": request_id,
            "timestamp": time.time()
        }
        await self.inference_queue.put(request)
    
    async def get_inference_result(self) -> Optional[Dict[str, Any]]:
        """Get the next inference result"""
        try:
            return await asyncio.wait_for(self.results_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get gateway status information"""
        device_statuses = {}
        for device_id, device in self.devices.items():
            device_statuses[device_id] = device.get_device_info()
        
        return {
            "gateway_id": self.gateway_id,
            "total_devices": len(self.devices),
            "online_devices": sum(1 for d in self.devices.values() if d.status == DeviceStatus.ONLINE),
            "total_inferences": sum(d.inference_count for d in self.devices.values()),
            "total_errors": sum(d.error_count for d in self.devices.values()),
            "devices": device_statuses
        }
    
    async def start(self):
        """Start the gateway"""
        self.running = True
        logger.info(f"Starting edge gateway {self.gateway_id}")
        
        # Start inference processing task
        asyncio.create_task(self.process_inference_requests())
    
    async def stop(self):
        """Stop the gateway"""
        self.running = False
        logger.info(f"Stopping edge gateway {self.gateway_id}")

class CloudBackend:
    """Cloud backend for model management and analytics"""
    
    def __init__(self):
        self.model_registry = {}
        self.device_analytics = {}
        self.model_versions = {}
    
    def register_model(self, model_id: str, model_path: str, version: str, metadata: Dict[str, Any]):
        """Register a new model version"""
        if model_id not in self.model_registry:
            self.model_registry[model_id] = {}
        
        self.model_registry[model_id][version] = {
            "model_path": model_path,
            "metadata": metadata,
            "created_at": time.time(),
            "deployment_count": 0
        }
        
        logger.info(f"Registered model {model_id} version {version}")
    
    def get_latest_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest model version"""
        if model_id not in self.model_registry:
            return None
        
        versions = list(self.model_registry[model_id].keys())
        if not versions:
            return None
        
        latest_version = max(versions)
        return self.model_registry[model_id][latest_version]
    
    def update_device_analytics(self, device_id: str, analytics_data: Dict[str, Any]):
        """Update device analytics"""
        if device_id not in self.device_analytics:
            self.device_analytics[device_id] = []
        
        analytics_data["timestamp"] = time.time()
        self.device_analytics[device_id].append(analytics_data)
        
        # Keep only last 1000 entries
        if len(self.device_analytics[device_id]) > 1000:
            self.device_analytics[device_id] = self.device_analytics[device_id][-1000:]
    
    def get_device_analytics(self, device_id: str) -> List[Dict[str, Any]]:
        """Get device analytics"""
        return self.device_analytics.get(device_id, [])
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics"""
        total_devices = len(self.device_analytics)
        total_inferences = sum(len(analytics) for analytics in self.device_analytics.values())
        
        return {
            "total_devices": total_devices,
            "total_inferences": total_inferences,
            "model_registry_size": len(self.model_registry),
            "timestamp": time.time()
        }

# Example usage
async def main():
    # Create edge devices
    device1 = EdgeDevice("device_001", "models/model_v1.tflite")
    device2 = EdgeDevice("device_002", "models/model_v1.tflite")
    device3 = EdgeDevice("device_003", "models/model_v1.tflite")
    
    # Create edge gateway
    gateway = EdgeGateway("gateway_001")
    gateway.add_device(device1)
    gateway.add_device(device2)
    gateway.add_device(device3)
    
    # Create cloud backend
    cloud = CloudBackend()
    cloud.register_model("image_classifier", "models/model_v1.tflite", "1.0.0", {
        "accuracy": 0.95,
        "size_mb": 2.5,
        "quantization": "int8"
    })
    
    # Start gateway
    await gateway.start()
    
    # Submit inference requests
    for i in range(10):
        input_data = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock input
        await gateway.submit_inference_request(input_data, f"request_{i}")
    
    # Collect results
    results = []
    for _ in range(10):
        result = await gateway.get_inference_result()
        if result:
            results.append(result)
            print(f"Result: {result}")
    
    # Update device analytics
    for device in gateway.devices.values():
        device_info = device.get_device_info()
        cloud.update_device_analytics(device.device_id, {
            "inference_count": device_info.inference_count,
            "error_count": device_info.error_count,
            "status": device_info.status.value
        })
    
    # Print system status
    print(f"\nGateway Status: {gateway.get_gateway_status()}")
    print(f"System Analytics: {cloud.get_system_analytics()}")
    
    # Stop gateway
    await gateway.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔒 **Privacy-Preserving ML**

### **Federated Learning**

#### **Concept**
Train models across multiple devices without sharing raw data, maintaining privacy while improving model performance.

#### **Architecture**
- **Local Training**: Each device trains on local data
- **Model Aggregation**: Central server aggregates model updates
- **Differential Privacy**: Add noise to protect individual data
- **Secure Aggregation**: Cryptographic protocols for privacy

#### **Code Example: Federated Learning System**

```python
import numpy as np
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelUpdate:
    device_id: str
    model_weights: List[float]
    sample_count: int
    round_number: int
    timestamp: float

class FederatedLearningClient:
    """Federated learning client on edge device"""
    
    def __init__(self, device_id: str, local_data: np.ndarray, local_labels: np.ndarray):
        self.device_id = device_id
        self.local_data = local_data
        self.local_labels = local_labels
        self.model_weights = None
        self.round_number = 0
        self.privacy_budget = 1.0  # Differential privacy budget
    
    def initialize_model(self, initial_weights: List[float]):
        """Initialize model with global weights"""
        self.model_weights = initial_weights.copy()
        logger.info(f"Device {self.device_id} initialized with global model")
    
    def train_local_model(self, epochs: int = 5) -> ModelUpdate:
        """Train model on local data"""
        logger.info(f"Device {self.device_id} starting local training")
        
        # Simulate local training
        # In practice, this would be actual model training
        local_weights = self.model_weights.copy()
        
        # Mock training process
        for epoch in range(epochs):
            # Simulate gradient updates
            gradient = np.random.randn(len(local_weights)) * 0.01
            local_weights = [w - 0.1 * g for w, g in zip(local_weights, gradient)]
        
        # Apply differential privacy
        if self.privacy_budget > 0:
            noise_scale = 0.1 / self.privacy_budget
            noise = np.random.normal(0, noise_scale, len(local_weights))
            local_weights = [w + n for w, n in zip(local_weights, noise)]
            self.privacy_budget -= 0.1
        
        # Create model update
        update = ModelUpdate(
            device_id=self.device_id,
            model_weights=local_weights,
            sample_count=len(self.local_data),
            round_number=self.round_number,
            timestamp=time.time()
        )
        
        self.round_number += 1
        logger.info(f"Device {self.device_id} completed local training")
        
        return update
    
    def update_global_model(self, global_weights: List[float]):
        """Update local model with global weights"""
        self.model_weights = global_weights.copy()
        logger.info(f"Device {self.device_id} updated with global model")

class FederatedLearningServer:
    """Federated learning server for model aggregation"""
    
    def __init__(self):
        self.global_model_weights = None
        self.client_updates = []
        self.round_number = 0
        self.aggregation_strategy = "fedavg"  # Federated averaging
    
    def initialize_global_model(self, model_size: int):
        """Initialize global model"""
        self.global_model_weights = np.random.randn(model_size).tolist()
        logger.info(f"Initialized global model with {model_size} parameters")
    
    def add_client_update(self, update: ModelUpdate):
        """Add client model update"""
        self.client_updates.append(update)
        logger.info(f"Received update from device {update.device_id}")
    
    def aggregate_models(self) -> List[float]:
        """Aggregate client model updates"""
        if not self.client_updates:
            return self.global_model_weights
        
        logger.info(f"Aggregating {len(self.client_updates)} client updates")
        
        if self.aggregation_strategy == "fedavg":
            # Federated averaging
            total_samples = sum(update.sample_count for update in self.client_updates)
            
            aggregated_weights = [0.0] * len(self.client_updates[0].model_weights)
            
            for update in self.client_updates:
                weight_factor = update.sample_count / total_samples
                for i, weight in enumerate(update.model_weights):
                    aggregated_weights[i] += weight * weight_factor
        
        elif self.aggregation_strategy == "fedprox":
            # FedProx with proximal term
            aggregated_weights = self.global_model_weights.copy()
            total_samples = sum(update.sample_count for update in self.client_updates)
            
            for update in self.client_updates:
                weight_factor = update.sample_count / total_samples
                for i, weight in enumerate(update.model_weights):
                    # Add proximal term
                    proximal_term = 0.01 * (weight - self.global_model_weights[i])
                    aggregated_weights[i] += weight_factor * (weight - proximal_term)
        
        # Update global model
        self.global_model_weights = aggregated_weights
        self.round_number += 1
        
        # Clear client updates
        self.client_updates = []
        
        logger.info(f"Global model updated for round {self.round_number}")
        return self.global_model_weights
    
    def get_global_model(self) -> List[float]:
        """Get current global model weights"""
        return self.global_model_weights.copy()
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            "round_number": self.round_number,
            "pending_updates": len(self.client_updates),
            "model_size": len(self.global_model_weights) if self.global_model_weights else 0,
            "aggregation_strategy": self.aggregation_strategy
        }

class PrivacyPreservingML:
    """Privacy-preserving ML system"""
    
    def __init__(self):
        self.server = FederatedLearningServer()
        self.clients = {}
        self.privacy_metrics = {}
    
    def add_client(self, client: FederatedLearningClient):
        """Add a federated learning client"""
        self.clients[client.device_id] = client
        logger.info(f"Added client {client.device_id}")
    
    def initialize_system(self, model_size: int):
        """Initialize the federated learning system"""
        self.server.initialize_global_model(model_size)
        
        # Initialize all clients with global model
        for client in self.clients.values():
            client.initialize_model(self.server.get_global_model())
    
    async def run_federated_round(self, epochs: int = 5):
        """Run a federated learning round"""
        logger.info(f"Starting federated learning round {self.server.round_number + 1}")
        
        # Collect updates from all clients
        updates = []
        for client in self.clients.values():
            update = client.train_local_model(epochs)
            updates.append(update)
            self.server.add_client_update(update)
        
        # Aggregate models
        new_global_weights = self.server.aggregate_models()
        
        # Update all clients with new global model
        for client in self.clients.values():
            client.update_global_model(new_global_weights)
        
        # Update privacy metrics
        self.update_privacy_metrics(updates)
        
        logger.info(f"Completed federated learning round {self.server.round_number}")
    
    def update_privacy_metrics(self, updates: List[ModelUpdate]):
        """Update privacy metrics"""
        total_samples = sum(update.sample_count for update in updates)
        avg_privacy_budget = np.mean([client.privacy_budget for client in self.clients.values()])
        
        self.privacy_metrics[self.server.round_number] = {
            "total_samples": total_samples,
            "participating_clients": len(updates),
            "avg_privacy_budget": avg_privacy_budget,
            "timestamp": time.time()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "server_status": self.server.get_server_status(),
            "total_clients": len(self.clients),
            "privacy_metrics": self.privacy_metrics,
            "timestamp": time.time()
        }

# Example usage
async def main():
    # Create federated learning system
    fl_system = PrivacyPreservingML()
    
    # Create clients with local data
    for i in range(5):
        device_id = f"device_{i:03d}"
        local_data = np.random.randn(100, 10)  # Mock local data
        local_labels = np.random.randint(0, 3, 100)  # Mock labels
        
        client = FederatedLearningClient(device_id, local_data, local_labels)
        fl_system.add_client(client)
    
    # Initialize system
    fl_system.initialize_system(model_size=50)
    
    # Run federated learning rounds
    for round_num in range(3):
        await fl_system.run_federated_round(epochs=3)
        
        # Print status
        status = fl_system.get_system_status()
        print(f"\nRound {round_num + 1} Status:")
        print(f"  Server: {status['server_status']}")
        print(f"  Privacy Metrics: {status['privacy_metrics']}")
    
    print("\nFederated learning completed!")

if __name__ == "__main__":
    import time
    asyncio.run(main())
```

---

## 📈 **Scalable Edge AI**

### **Load Balancing and Auto-scaling**

#### **Concepts**
- **Device Pool Management**: Manage multiple edge devices
- **Load Balancing**: Distribute inference requests
- **Auto-scaling**: Scale based on demand
- **Health Monitoring**: Monitor device health and performance

#### **Code Example: Scalable Edge AI System**

```python
import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

@dataclass
class InferenceRequest:
    request_id: str
    input_data: List[float]
    priority: int
    timestamp: float
    timeout: float

@dataclass
class InferenceResult:
    request_id: str
    result: Dict[str, Any]
    device_id: str
    processing_time: float
    timestamp: float

class EdgeDevice:
    """Enhanced edge device with health monitoring"""
    
    def __init__(self, device_id: str, capacity: int = 10):
        self.device_id = device_id
        self.capacity = capacity
        self.current_load = 0
        self.health = DeviceHealth.HEALTHY
        self.response_times = []
        self.error_count = 0
        self.total_requests = 0
        self.last_health_check = time.time()
    
    def can_handle_request(self) -> bool:
        """Check if device can handle more requests"""
        return (self.current_load < self.capacity and 
                self.health != DeviceHealth.OFFLINE)
    
    async def process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process an inference request"""
        start_time = time.time()
        self.current_load += 1
        self.total_requests += 1
        
        try:
            # Simulate inference processing
            processing_time = random.uniform(0.1, 0.5)  # Mock processing time
            await asyncio.sleep(processing_time)
            
            # Mock result
            result = {
                "prediction": [0.1, 0.2, 0.7],
                "confidence": random.uniform(0.8, 0.95),
                "model_version": "1.0.0"
            }
            
            # Update response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            return InferenceResult(
                request_id=request.request_id,
                result=result,
                device_id=self.device_id,
                processing_time=response_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing request on device {self.device_id}: {e}")
            raise
        finally:
            self.current_load -= 1
    
    def update_health(self):
        """Update device health based on metrics"""
        current_time = time.time()
        
        # Check if device is responsive
        if current_time - self.last_health_check > 30:  # 30 seconds timeout
            self.health = DeviceHealth.OFFLINE
            return
        
        # Calculate health metrics
        if len(self.response_times) > 0:
            avg_response_time = statistics.mean(self.response_times)
            error_rate = self.error_count / max(self.total_requests, 1)
            
            if error_rate > 0.1 or avg_response_time > 1.0:
                self.health = DeviceHealth.UNHEALTHY
            elif error_rate > 0.05 or avg_response_time > 0.5:
                self.health = DeviceHealth.DEGRADED
            else:
                self.health = DeviceHealth.HEALTHY
        
        self.last_health_check = current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get device metrics"""
        return {
            "device_id": self.device_id,
            "health": self.health.value,
            "current_load": self.current_load,
            "capacity": self.capacity,
            "utilization": self.current_load / self.capacity,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0,
            "last_health_check": self.last_health_check
        }

class LoadBalancer:
    """Load balancer for edge devices"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.devices: Dict[str, EdgeDevice] = {}
        self.current_index = 0
        self.request_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.running = False
    
    def add_device(self, device: EdgeDevice):
        """Add a device to the load balancer"""
        self.devices[device.device_id] = device
        logger.info(f"Added device {device.device_id} to load balancer")
    
    def remove_device(self, device_id: str):
        """Remove a device from the load balancer"""
        if device_id in self.devices:
            del self.devices[device_id]
            logger.info(f"Removed device {device_id} from load balancer")
    
    def select_device(self, request: InferenceRequest) -> Optional[EdgeDevice]:
        """Select a device based on load balancing strategy"""
        available_devices = [d for d in self.devices.values() if d.can_handle_request()]
        
        if not available_devices:
            return None
        
        if self.strategy == "round_robin":
            device = available_devices[self.current_index % len(available_devices)]
            self.current_index += 1
            return device
        
        elif self.strategy == "least_loaded":
            return min(available_devices, key=lambda d: d.current_load)
        
        elif self.strategy == "health_based":
            # Prefer healthy devices
            healthy_devices = [d for d in available_devices if d.health == DeviceHealth.HEALTHY]
            if healthy_devices:
                return min(healthy_devices, key=lambda d: d.current_load)
            else:
                return min(available_devices, key=lambda d: d.current_load)
        
        return available_devices[0]
    
    async def process_requests(self):
        """Process inference requests"""
        while self.running:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                
                # Select device
                device = self.select_device(request)
                if not device:
                    # No available devices
                    error_result = InferenceResult(
                        request_id=request.request_id,
                        result={"error": "No available devices"},
                        device_id="",
                        processing_time=0,
                        timestamp=time.time()
                    )
                    await self.result_queue.put(error_result)
                    continue
                
                # Process request
                try:
                    result = await device.process_request(request)
                    await self.result_queue.put(result)
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    error_result = InferenceResult(
                        request_id=request.request_id,
                        result={"error": str(e)},
                        device_id=device.device_id,
                        processing_time=0,
                        timestamp=time.time()
                    )
                    await self.result_queue.put(error_result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in request processing: {e}")
    
    async def submit_request(self, request: InferenceRequest):
        """Submit an inference request"""
        await self.request_queue.put(request)
    
    async def get_result(self) -> Optional[InferenceResult]:
        """Get the next inference result"""
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer status"""
        device_metrics = {}
        for device_id, device in self.devices.items():
            device_metrics[device_id] = device.get_metrics()
        
        return {
            "strategy": self.strategy,
            "total_devices": len(self.devices),
            "available_devices": len([d for d in self.devices.values() if d.can_handle_request()]),
            "queue_size": self.request_queue.qsize(),
            "devices": device_metrics
        }
    
    async def start(self):
        """Start the load balancer"""
        self.running = True
        logger.info("Starting load balancer")
        asyncio.create_task(self.process_requests())
    
    async def stop(self):
        """Stop the load balancer"""
        self.running = False
        logger.info("Stopping load balancer")

class AutoScaler:
    """Auto-scaler for edge devices"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.scaling_metrics = []
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.min_devices = 2
        self.max_devices = 10
        self.running = False
    
    def add_device(self, device: EdgeDevice):
        """Add a new device"""
        self.load_balancer.add_device(device)
        logger.info(f"Auto-scaler added device {device.device_id}")
    
    def remove_device(self, device_id: str):
        """Remove a device"""
        self.load_balancer.remove_device(device_id)
        logger.info(f"Auto-scaler removed device {device_id}")
    
    async def monitor_and_scale(self):
        """Monitor system and scale as needed"""
        while self.running:
            try:
                # Get system metrics
                status = self.load_balancer.get_load_balancer_status()
                
                # Calculate average utilization
                total_utilization = 0
                healthy_devices = 0
                
                for device_metrics in status["devices"].values():
                    if device_metrics["health"] != "offline":
                        total_utilization += device_metrics["utilization"]
                        healthy_devices += 1
                
                avg_utilization = total_utilization / max(healthy_devices, 1)
                
                # Record metrics
                self.scaling_metrics.append({
                    "timestamp": time.time(),
                    "avg_utilization": avg_utilization,
                    "total_devices": status["total_devices"],
                    "available_devices": status["available_devices"],
                    "queue_size": status["queue_size"]
                })
                
                # Keep only last 100 metrics
                if len(self.scaling_metrics) > 100:
                    self.scaling_metrics = self.scaling_metrics[-100:]
                
                # Scale up if needed
                if (avg_utilization > self.scale_up_threshold and 
                    status["total_devices"] < self.max_devices):
                    await self.scale_up()
                
                # Scale down if needed
                elif (avg_utilization < self.scale_down_threshold and 
                      status["total_devices"] > self.min_devices):
                    await self.scale_down()
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")
                await asyncio.sleep(10)
    
    async def scale_up(self):
        """Scale up by adding a new device"""
        if self.load_balancer.get_load_balancer_status()["total_devices"] >= self.max_devices:
            return
        
        # Create new device
        new_device_id = f"device_{len(self.load_balancer.devices):03d}"
        new_device = EdgeDevice(new_device_id, capacity=10)
        
        self.add_device(new_device)
        logger.info(f"Scaled up: added device {new_device_id}")
    
    async def scale_down(self):
        """Scale down by removing a device"""
        if self.load_balancer.get_load_balancer_status()["total_devices"] <= self.min_devices:
            return
        
        # Find device with lowest utilization
        devices = list(self.load_balancer.devices.values())
        if not devices:
            return
        
        device_to_remove = min(devices, key=lambda d: d.current_load)
        self.remove_device(device_to_remove.device_id)
        logger.info(f"Scaled down: removed device {device_to_remove.device_id}")
    
    def get_scaling_metrics(self) -> List[Dict[str, Any]]:
        """Get scaling metrics"""
        return self.scaling_metrics.copy()
    
    async def start(self):
        """Start auto-scaler"""
        self.running = True
        logger.info("Starting auto-scaler")
        asyncio.create_task(self.monitor_and_scale())
    
    async def stop(self):
        """Stop auto-scaler"""
        self.running = False
        logger.info("Stopping auto-scaler")

# Example usage
async def main():
    # Create load balancer
    load_balancer = LoadBalancer(strategy="health_based")
    
    # Create initial devices
    for i in range(3):
        device = EdgeDevice(f"device_{i:03d}", capacity=5)
        load_balancer.add_device(device)
    
    # Create auto-scaler
    auto_scaler = AutoScaler(load_balancer)
    
    # Start systems
    await load_balancer.start()
    await auto_scaler.start()
    
    # Submit requests
    for i in range(20):
        request = InferenceRequest(
            request_id=f"req_{i}",
            input_data=[0.1, 0.2, 0.3, 0.4, 0.5],
            priority=1,
            timestamp=time.time(),
            timeout=5.0
        )
        await load_balancer.submit_request(request)
    
    # Collect results
    results = []
    for _ in range(20):
        result = await load_balancer.get_result()
        if result:
            results.append(result)
            print(f"Result: {result.request_id} from {result.device_id}")
    
    # Print status
    print(f"\nLoad Balancer Status: {load_balancer.get_load_balancer_status()}")
    print(f"Scaling Metrics: {len(auto_scaler.get_scaling_metrics())} records")
    
    # Stop systems
    await auto_scaler.stop()
    await load_balancer.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 **Interview Questions**

### **System Design**

#### **Q1: How would you design a scalable TinyML system for IoT devices?**
**Answer**: 
- **Edge-First Architecture**: Local inference with cloud coordination
- **Device Management**: Health monitoring, load balancing, auto-scaling
- **Communication**: MQTT for device communication, REST for management
- **Data Pipeline**: Edge processing, selective cloud upload
- **Security**: Device authentication, encrypted communication
- **Monitoring**: Real-time metrics, alerting, performance tracking

#### **Q2: What are the key considerations for privacy-preserving ML in TinyML?**
**Answer**: 
- **Federated Learning**: Train models without sharing raw data
- **Differential Privacy**: Add noise to protect individual data
- **Secure Aggregation**: Cryptographic protocols for model updates
- **Local Processing**: Keep sensitive data on device
- **Data Minimization**: Collect only necessary data
- **Audit Trails**: Track data usage and model updates

#### **Q3: How do you handle device failures and network issues in distributed TinyML?**
**Answer**: 
- **Health Monitoring**: Continuous device health checks
- **Fault Tolerance**: Redundant devices and failover mechanisms
- **Graceful Degradation**: Reduce functionality when devices fail
- **Network Resilience**: Offline operation, message queuing
- **Recovery Strategies**: Automatic device recovery and reconnection
- **Load Redistribution**: Rebalance load when devices fail

---

**Ready to explore interview questions? Let's dive into [Interview Questions](./InterviewQuestions.md) next!** 🚀
