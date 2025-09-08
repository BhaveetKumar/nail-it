# 🚀 Model Serving

> **Master AI model serving: from deployment strategies to production optimization**

## 🎯 **Learning Objectives**

- Understand model serving architectures and patterns
- Implement model serving systems in Python and Go
- Master load balancing and scaling strategies
- Handle model versioning and A/B testing
- Build production-ready model serving infrastructure

## 📚 **Table of Contents**

1. [Serving Architectures](#serving-architectures)
2. [Implementation Examples](#implementation-examples)
3. [Load Balancing and Scaling](#load-balancing-and-scaling)
4. [Model Versioning](#model-versioning)
5. [Production Optimization](#production-optimization)
6. [Interview Questions](#interview-questions)

---

## 🏗️ **Serving Architectures**

### **Model Serving Patterns**

#### **Concept**
Model serving involves deploying trained models to production environments with high availability, scalability, and performance.

#### **Architecture Types**
- **Synchronous**: Request-response pattern with immediate results
- **Asynchronous**: Queue-based processing for batch inference
- **Streaming**: Real-time processing of data streams
- **Edge**: Deploy models closer to data sources

#### **Code Example**

```python
import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    version: str
    path: str
    input_shape: tuple
    output_shape: tuple
    batch_size: int = 32
    max_latency_ms: int = 100
    device: str = "cpu"

class ModelServer(ABC):
    """Abstract base class for model servers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_loaded = False
        self.load_time = None
        self.inference_count = 0
        self.total_inference_time = 0.0
    
    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions"""
        pass
    
    @abstractmethod
    def batch_predict(self, input_batch: List[Any]) -> List[Any]:
        """Make batch predictions"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model metrics"""
        avg_latency = self.total_inference_time / max(self.inference_count, 1)
        return {
            "model_name": self.config.name,
            "version": self.config.version,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "inference_count": self.inference_count,
            "average_latency_ms": avg_latency * 1000,
            "total_inference_time": self.total_inference_time
        }

class PyTorchModelServer(ModelServer):
    """PyTorch model server implementation"""
    
    def load_model(self):
        """Load PyTorch model"""
        try:
            start_time = time.time()
            self.model = torch.load(self.config.path, map_location=self.config.device)
            self.model.eval()
            self.is_loaded = True
            self.load_time = time.time() - start_time
            logger.info(f"Model {self.config.name} loaded in {self.load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model {self.config.name}: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make single prediction"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert to tensor
            input_tensor = torch.from_numpy(input_data).float()
            if self.config.device == "cuda":
                input_tensor = input_tensor.cuda()
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                else:
                    output = [t.cpu().numpy() for t in output]
            
            # Update metrics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return output
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def batch_predict(self, input_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Make batch predictions"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Stack inputs
            batch_tensor = torch.from_numpy(np.stack(input_batch)).float()
            if self.config.device == "cuda":
                batch_tensor = batch_tensor.cuda()
            
            # Make batch prediction
            with torch.no_grad():
                output = self.model(batch_tensor)
                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                else:
                    output = [t.cpu().numpy() for t in output]
            
            # Update metrics
            inference_time = time.time() - start_time
            self.inference_count += len(input_batch)
            self.total_inference_time += inference_time
            
            return output
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

class ModelRegistry:
    """Model registry for managing multiple models"""
    
    def __init__(self):
        self.models: Dict[str, ModelServer] = {}
        self.model_versions: Dict[str, List[str]] = {}
        self.default_models: Dict[str, str] = {}
    
    def register_model(self, model: ModelServer):
        """Register a model"""
        model_key = f"{model.config.name}:{model.config.version}"
        self.models[model_key] = model
        
        # Track versions
        if model.config.name not in self.model_versions:
            self.model_versions[model.config.name] = []
        self.model_versions[model.config.name].append(model.config.version)
        
        # Set as default if first version
        if model.config.name not in self.default_models:
            self.default_models[model.config.name] = model.config.version
        
        logger.info(f"Registered model {model_key}")
    
    def get_model(self, name: str, version: Optional[str] = None) -> ModelServer:
        """Get a model by name and version"""
        if version is None:
            version = self.default_models.get(name)
            if version is None:
                raise ValueError(f"No default version for model {name}")
        
        model_key = f"{name}:{version}"
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        return self.models[model_key]
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models"""
        return self.model_versions.copy()
    
    def get_model_metrics(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model metrics"""
        model = self.get_model(name, version)
        return model.get_metrics()

# Request/Response models
class PredictionRequest(BaseModel):
    """Prediction request model"""
    model_name: str
    model_version: Optional[str] = None
    input_data: List[List[float]]
    batch_id: Optional[str] = None

class PredictionResponse(BaseModel):
    """Prediction response model"""
    predictions: List[List[float]]
    model_name: str
    model_version: str
    inference_time_ms: float
    batch_id: Optional[str] = None

class ModelMetricsResponse(BaseModel):
    """Model metrics response model"""
    metrics: Dict[str, Any]

# FastAPI application
app = FastAPI(title="AI Model Serving API", version="1.0.0")

# Global model registry
model_registry = ModelRegistry()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    # Example: Load a sample model
    config = ModelConfig(
        name="sample_model",
        version="1.0.0",
        path="path/to/model.pth",
        input_shape=(1, 784),
        output_shape=(1, 10),
        device="cpu"
    )
    
    # Note: In production, you would load actual models
    # model = PyTorchModelServer(config)
    # model.load_model()
    # model_registry.register_model(model)
    
    logger.info("Model serving API started")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions"""
    try:
        # Get model
        model = model_registry.get_model(request.model_name, request.model_version)
        
        # Convert input data
        input_data = np.array(request.input_data)
        
        # Make prediction
        start_time = time.time()
        predictions = model.predict(input_data)
        inference_time = time.time() - start_time
        
        # Convert predictions to list
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        else:
            predictions = [p.tolist() for p in predictions]
        
        return PredictionResponse(
            predictions=predictions,
            model_name=request.model_name,
            model_version=model.config.version,
            inference_time_ms=inference_time * 1000,
            batch_id=request.batch_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=List[PredictionResponse])
async def batch_predict(requests: List[PredictionRequest]):
    """Make batch predictions"""
    responses = []
    
    for request in requests:
        try:
            response = await predict(request)
            responses.append(response)
        except Exception as e:
            logger.error(f"Batch prediction failed for {request.model_name}: {e}")
            # Continue with other requests
    
    return responses

@app.get("/models", response_model=Dict[str, List[str]])
async def list_models():
    """List all available models"""
    return model_registry.list_models()

@app.get("/models/{model_name}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(model_name: str, version: Optional[str] = None):
    """Get model metrics"""
    try:
        metrics = model_registry.get_model_metrics(model_name, version)
        return ModelMetricsResponse(metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## ⚖️ **Load Balancing and Scaling**

### **Advanced Scaling Strategies**

#### **Concept**
Load balancing distributes requests across multiple model instances to improve performance and availability.

#### **Code Example**

```python
import random
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import statistics

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"

@dataclass
class ModelInstance:
    """Model instance configuration"""
    id: str
    url: str
    weight: int = 1
    max_connections: int = 100
    health_check_interval: int = 30
    is_healthy: bool = True
    active_connections: int = 0
    response_times: List[float] = None
    last_health_check: float = 0
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []

class LoadBalancer:
    """Load balancer for model instances"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.instances: List[ModelInstance] = []
        self.current_index = 0
        self.health_check_executor = ThreadPoolExecutor(max_workers=2)
        self.start_health_checks()
    
    def add_instance(self, instance: ModelInstance):
        """Add a model instance"""
        self.instances.append(instance)
        logger.info(f"Added model instance {instance.id}")
    
    def remove_instance(self, instance_id: str):
        """Remove a model instance"""
        self.instances = [i for i in self.instances if i.id != instance_id]
        logger.info(f"Removed model instance {instance_id}")
    
    def get_healthy_instances(self) -> List[ModelInstance]:
        """Get only healthy instances"""
        return [i for i in self.instances if i.is_healthy]
    
    def select_instance(self) -> ModelInstance:
        """Select an instance based on the load balancing strategy"""
        healthy_instances = self.get_healthy_instances()
        
        if not healthy_instances:
            raise RuntimeError("No healthy instances available")
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_selection(healthy_instances)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _round_robin_selection(self, instances: List[ModelInstance]) -> ModelInstance:
        """Round robin selection"""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _least_connections_selection(self, instances: List[ModelInstance]) -> ModelInstance:
        """Select instance with least active connections"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _weighted_round_robin_selection(self, instances: List[ModelInstance]) -> ModelInstance:
        """Weighted round robin selection"""
        total_weight = sum(i.weight for i in instances)
        if total_weight == 0:
            return random.choice(instances)
        
        # Simple weighted selection
        rand = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if rand <= current_weight:
                return instance
        
        return instances[-1]  # Fallback
    
    def _least_response_time_selection(self, instances: List[ModelInstance]) -> ModelInstance:
        """Select instance with least average response time"""
        def get_avg_response_time(instance):
            if not instance.response_times:
                return float('inf')
            return statistics.mean(instance.response_times[-10:])  # Last 10 requests
        
        return min(instances, key=get_avg_response_time)
    
    def _random_selection(self, instances: List[ModelInstance]) -> ModelInstance:
        """Random selection"""
        return random.choice(instances)
    
    def start_health_checks(self):
        """Start health check background task"""
        asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """Health check loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all instances"""
        tasks = []
        for instance in self.instances:
            task = asyncio.create_task(self._check_instance_health(instance))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_instance_health(self, instance: ModelInstance):
        """Check health of a single instance"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{instance.url}/health", timeout=5) as response:
                    instance.is_healthy = response.status == 200
                    instance.last_health_check = time.time()
        except Exception as e:
            instance.is_healthy = False
            logger.warning(f"Health check failed for {instance.id}: {e}")
    
    async def make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request through the load balancer"""
        instance = self.select_instance()
        
        # Increment connection count
        instance.active_connections += 1
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{instance.url}{endpoint}", json=data) as response:
                    result = await response.json()
                    
                    # Record response time
                    response_time = time.time() - start_time
                    instance.response_times.append(response_time)
                    
                    # Keep only last 100 response times
                    if len(instance.response_times) > 100:
                        instance.response_times = instance.response_times[-100:]
                    
                    return result
        
        except Exception as e:
            logger.error(f"Request failed for {instance.id}: {e}")
            raise
        
        finally:
            # Decrement connection count
            instance.active_connections = max(0, instance.active_connections - 1)

class AutoScaler:
    """Auto-scaler for model instances"""
    
    def __init__(self, load_balancer: LoadBalancer, min_instances: int = 1, max_instances: int = 10):
        self.load_balancer = load_balancer
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = 0.8  # CPU/Memory usage
        self.scale_down_threshold = 0.3
        self.scale_cooldown = 300  # 5 minutes
        self.last_scale_time = 0
    
    async def check_and_scale(self):
        """Check metrics and scale if needed"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        # Get current metrics
        metrics = await self._get_cluster_metrics()
        
        # Scale up if needed
        if metrics['avg_cpu'] > self.scale_up_threshold and len(self.load_balancer.instances) < self.max_instances:
            await self._scale_up()
            self.last_scale_time = current_time
        
        # Scale down if needed
        elif metrics['avg_cpu'] < self.scale_down_threshold and len(self.load_balancer.instances) > self.min_instances:
            await self._scale_down()
            self.last_scale_time = current_time
    
    async def _get_cluster_metrics(self) -> Dict[str, float]:
        """Get cluster metrics"""
        # In production, this would query your monitoring system
        # For demo, return mock metrics
        return {
            'avg_cpu': random.uniform(0.2, 0.9),
            'avg_memory': random.uniform(0.3, 0.8),
            'avg_response_time': random.uniform(50, 200)
        }
    
    async def _scale_up(self):
        """Scale up by adding new instances"""
        # In production, this would create new instances
        logger.info("Scaling up - adding new instances")
        # Implementation would depend on your infrastructure (Kubernetes, Docker, etc.)
    
    async def _scale_down(self):
        """Scale down by removing instances"""
        # In production, this would remove instances
        logger.info("Scaling down - removing instances")
        # Implementation would depend on your infrastructure

# Example usage
async def main():
    # Create load balancer
    lb = LoadBalancer(LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    
    # Add instances
    instances = [
        ModelInstance("instance-1", "http://localhost:8001", weight=2),
        ModelInstance("instance-2", "http://localhost:8002", weight=1),
        ModelInstance("instance-3", "http://localhost:8003", weight=1),
    ]
    
    for instance in instances:
        lb.add_instance(instance)
    
    # Create auto-scaler
    scaler = AutoScaler(lb, min_instances=2, max_instances=5)
    
    # Make some requests
    for i in range(10):
        try:
            result = await lb.make_request("/predict", {"input_data": [[1, 2, 3, 4, 5]]})
            print(f"Request {i}: {result}")
        except Exception as e:
            print(f"Request {i} failed: {e}")
    
    # Check scaling
    await scaler.check_and_scale()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 **Interview Questions**

### **Model Serving Theory**

#### **Q1: What are the different model serving patterns?**
**Answer**: 
- **Synchronous**: Request-response with immediate results, good for real-time applications
- **Asynchronous**: Queue-based processing, good for batch inference
- **Streaming**: Real-time processing of data streams, good for continuous data
- **Edge**: Deploy models closer to data sources, good for low latency requirements

#### **Q2: How do you handle model versioning in production?**
**Answer**: 
- **Semantic Versioning**: Use MAJOR.MINOR.PATCH format
- **A/B Testing**: Deploy multiple versions and compare performance
- **Blue-Green Deployment**: Switch between versions with zero downtime
- **Canary Deployment**: Gradually roll out new versions to subset of traffic
- **Model Registry**: Track model metadata, performance, and lineage

#### **Q3: What are the key considerations for model serving performance?**
**Answer**: 
- **Latency**: Minimize inference time and network overhead
- **Throughput**: Handle high request volumes efficiently
- **Resource Usage**: Optimize CPU, memory, and GPU utilization
- **Caching**: Cache predictions for repeated inputs
- **Batching**: Process multiple requests together
- **Load Balancing**: Distribute load across multiple instances

#### **Q4: How do you monitor model performance in production?**
**Answer**: 
- **Metrics**: Track latency, throughput, error rates, resource usage
- **Logging**: Log predictions, errors, and system events
- **Alerting**: Set up alerts for performance degradation
- **Dashboards**: Visualize metrics and trends
- **Model Drift**: Monitor input/output distributions for drift
- **A/B Testing**: Compare model versions statistically

#### **Q5: What are the challenges in scaling model serving?**
**Answer**: 
- **Resource Constraints**: Limited CPU, memory, and GPU resources
- **Cold Starts**: Initialization time for new instances
- **State Management**: Handling model state and caching
- **Load Balancing**: Distributing requests efficiently
- **Data Consistency**: Ensuring consistent predictions across instances
- **Cost Management**: Balancing performance with infrastructure costs

### **Implementation Questions**

#### **Q6: Implement a model serving system**
**Answer**: See the implementation above with FastAPI, model registry, and load balancing.

#### **Q7: How would you handle model updates without downtime?**
**Answer**: 
- **Blue-Green Deployment**: Maintain two identical environments
- **Rolling Updates**: Update instances one at a time
- **Feature Flags**: Control which version serves requests
- **Model Warmup**: Pre-load models before serving traffic
- **Health Checks**: Verify new models before switching traffic
- **Rollback Strategy**: Quick rollback to previous version if issues occur

#### **Q8: How do you optimize model inference for production?**
**Answer**: 
- **Model Optimization**: Quantization, pruning, distillation
- **Hardware Acceleration**: Use GPUs, TPUs, or specialized chips
- **Caching**: Cache frequent predictions
- **Batching**: Process multiple requests together
- **Async Processing**: Use async/await for I/O operations
- **Connection Pooling**: Reuse connections to reduce overhead

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different models
2. **Optimize**: Focus on performance and scalability
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about MLOps and CI/CD for ML
5. **Interview**: Practice model serving interview questions

---

**Ready to learn about MLOps? Let's move to the next section!** 🎯
