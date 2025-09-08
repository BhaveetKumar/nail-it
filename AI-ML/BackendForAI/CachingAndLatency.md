# ⚡ Caching and Latency: Optimizing AI System Performance

> **Complete guide to caching strategies and latency optimization for AI systems**

## 🎯 **Learning Objectives**

- Master caching strategies for AI applications
- Understand latency optimization techniques
- Implement multi-level caching systems
- Optimize model inference performance
- Handle real-time AI workloads efficiently

## 📚 **Table of Contents**

1. [Caching Fundamentals](#caching-fundamentals)
2. [Caching Strategies](#caching-strategies)
3. [Latency Optimization](#latency-optimization)
4. [Real-time Systems](#real-time-systems)
5. [Performance Monitoring](#performance-monitoring)
6. [Interview Questions](#interview-questions)

---

## ⚡ **Caching Fundamentals**

### **Concept**

Caching is the process of storing frequently accessed data in fast storage to reduce latency and improve system performance. In AI systems, caching can significantly reduce inference time and computational costs.

### **Types of Caching**

1. **Prediction Caching**: Cache model predictions
2. **Feature Caching**: Cache computed features
3. **Model Caching**: Cache model weights and parameters
4. **Data Caching**: Cache input data and preprocessed results
5. **Result Caching**: Cache final outputs and responses

### **Caching Benefits**

- **Reduced Latency**: Faster response times
- **Lower Costs**: Reduced computational requirements
- **Better Scalability**: Handle more requests with same resources
- **Improved User Experience**: Faster application responses
- **Resource Optimization**: Better utilization of compute resources

---

## 🗄️ **Caching Strategies**

### **1. In-Memory Caching**

**Code Example**:
```python
import time
import hashlib
import json
from typing import Any, Dict, List, Optional
from functools import lru_cache
import threading
from collections import OrderedDict

class InMemoryCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, (list, dict)):
            key_string = json.dumps(data, sort_keys=True)
        else:
            key_string = str(data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def _evict_lru(self):
        """Remove least recently used entry"""
        if self.cache:
            self.cache.popitem(last=False)
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key in self.cache:
                if self._is_expired(cache_key):
                    # Remove expired entry
                    self.cache.pop(cache_key, None)
                    self.timestamps.pop(cache_key, None)
                    return None
                
                # Move to end (most recently used)
                value = self.cache.pop(cache_key)
                self.cache[cache_key] = value
                return value
            
            return None
    
    def set(self, key: Any, value: Any):
        """Set value in cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            
            # Remove if already exists
            if cache_key in self.cache:
                self.cache.pop(cache_key, None)
                self.timestamps.pop(cache_key, None)
            
            # Evict expired entries
            self._evict_expired()
            
            # Evict LRU if at capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self.cache[cache_key] = value
            self.timestamps[cache_key] = time.time()
    
    def delete(self, key: Any):
        """Delete value from cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            self.cache.pop(cache_key, None)
            self.timestamps.pop(cache_key, None)
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl
            }

# Example usage
cache = InMemoryCache(max_size=100, ttl=60)

# Cache some data
cache.set("user:123", {"name": "John", "age": 30})
cache.set("model:prediction", {"result": [0.8, 0.2], "confidence": 0.95})

# Retrieve data
user_data = cache.get("user:123")
prediction = cache.get("model:prediction")

print(f"User data: {user_data}")
print(f"Prediction: {prediction}")
print(f"Cache stats: {cache.stats()}")
```

### **2. Redis Caching**

**Code Example**:
```python
import redis
import json
import pickle
import time
from typing import Any, Dict, List, Optional
import hashlib

class RedisCache:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.default_ttl = 3600  # 1 hour
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage"""
        if isinstance(data, (dict, list)):
            return json.dumps(data).encode('utf-8')
        elif isinstance(data, (str, int, float, bool)):
            return json.dumps(data).encode('utf-8')
        else:
            return pickle.dumps(data)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return pickle.loads(data)
    
    def _generate_key(self, key: Any) -> str:
        """Generate cache key"""
        if isinstance(key, (list, dict)):
            key_string = json.dumps(key, sort_keys=True)
        else:
            key_string = str(key)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(key)
        data = self.redis_client.get(cache_key)
        
        if data is not None:
            return self._deserialize(data)
        return None
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        cache_key = self._generate_key(key)
        serialized_value = self._serialize(value)
        
        if ttl is None:
            ttl = self.default_ttl
        
        self.redis_client.setex(cache_key, ttl, serialized_value)
    
    def delete(self, key: Any):
        """Delete value from cache"""
        cache_key = self._generate_key(key)
        self.redis_client.delete(cache_key)
    
    def exists(self, key: Any) -> bool:
        """Check if key exists in cache"""
        cache_key = self._generate_key(key)
        return self.redis_client.exists(cache_key) > 0
    
    def get_ttl(self, key: Any) -> int:
        """Get time to live for key"""
        cache_key = self._generate_key(key)
        return self.redis_client.ttl(cache_key)
    
    def set_ttl(self, key: Any, ttl: int):
        """Set time to live for key"""
        cache_key = self._generate_key(key)
        self.redis_client.expire(cache_key, ttl)
    
    def clear(self):
        """Clear all cache entries"""
        self.redis_client.flushdb()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        info = self.redis_client.info()
        return {
            "used_memory": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "total_commands_processed": info.get("total_commands_processed"),
            "keyspace_hits": info.get("keyspace_hits"),
            "keyspace_misses": info.get("keyspace_misses")
        }

# Example usage
redis_cache = RedisCache()

# Cache some data
redis_cache.set("model:bert", {"weights": [0.1, 0.2, 0.3]}, ttl=7200)
redis_cache.set("prediction:123", {"result": [0.8, 0.2]}, ttl=300)

# Retrieve data
model_data = redis_cache.get("model:bert")
prediction = redis_cache.get("prediction:123")

print(f"Model data: {model_data}")
print(f"Prediction: {prediction}")
print(f"Cache stats: {redis_cache.stats()}")
```

### **3. Multi-Level Caching**

**Code Example**:
```python
class MultiLevelCache:
    def __init__(self, l1_size: int = 100, l1_ttl: int = 300, 
                 redis_host: str = "localhost", redis_port: int = 6379):
        # Level 1: In-memory cache (fastest)
        self.l1_cache = InMemoryCache(max_size=l1_size, ttl=l1_ttl)
        
        # Level 2: Redis cache (fast)
        self.l2_cache = RedisCache(host=redis_host, port=redis_port, db=1)
        
        # Level 3: Database/File system (slowest)
        self.l3_storage = {}  # Mock database
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try Level 1 (in-memory)
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try Level 2 (Redis)
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to Level 1
            self.l1_cache.set(key, value)
            return value
        
        # Try Level 3 (database)
        value = self.l3_storage.get(key)
        if value is not None:
            # Promote to Level 2 and Level 1
            self.l2_cache.set(key, value)
            self.l1_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None):
        """Set value in multi-level cache"""
        # Set in all levels
        self.l1_cache.set(key, value)
        self.l2_cache.set(key, value, ttl)
        self.l3_storage[key] = value
    
    def delete(self, key: Any):
        """Delete value from all cache levels"""
        self.l1_cache.delete(key)
        self.l2_cache.delete(key)
        self.l3_storage.pop(key, None)
    
    def clear(self):
        """Clear all cache levels"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_storage.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics for all cache levels"""
        return {
            "l1_cache": self.l1_cache.stats(),
            "l2_cache": self.l2_cache.stats(),
            "l3_storage": {"size": len(self.l3_storage)}
        }

# Example usage
multi_cache = MultiLevelCache()

# Set data
multi_cache.set("user:123", {"name": "John", "age": 30})
multi_cache.set("model:prediction", {"result": [0.8, 0.2]})

# Get data (will try all levels)
user_data = multi_cache.get("user:123")
prediction = multi_cache.get("model:prediction")

print(f"User data: {user_data}")
print(f"Prediction: {prediction}")
print(f"Multi-level cache stats: {multi_cache.stats()}")
```

---

## ⚡ **Latency Optimization**

### **1. Model Inference Optimization**

**Code Example**:
```python
import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedModelServer:
    def __init__(self, model: nn.Module, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Warm up model
        self._warmup()
    
    def _warmup(self):
        """Warm up model for faster inference"""
        dummy_input = torch.randn(1, 784).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
    
    def predict_single(self, input_data: List[float]) -> Dict[str, Any]:
        """Single prediction with optimization"""
        start_time = time.time()
        
        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.cpu().numpy().tolist()
        
        inference_time = time.time() - start_time
        
        return {
            "prediction": prediction,
            "inference_time": inference_time,
            "device": str(self.device)
        }
    
    def predict_batch(self, batch_data: List[List[float]]) -> Dict[str, Any]:
        """Batch prediction for efficiency"""
        start_time = time.time()
        
        # Convert to tensor
        input_tensor = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            predictions = output.cpu().numpy().tolist()
        
        inference_time = time.time() - start_time
        
        return {
            "predictions": predictions,
            "batch_size": len(batch_data),
            "inference_time": inference_time,
            "throughput": len(batch_data) / inference_time
        }
    
    async def predict_async(self, input_data: List[float]) -> Dict[str, Any]:
        """Async prediction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.predict_single, input_data)
    
    def benchmark(self, test_data: List[List[float]], num_runs: int = 100):
        """Benchmark model performance"""
        # Single prediction benchmark
        single_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.predict_single(test_data[0])
            single_times.append(time.time() - start_time)
        
        # Batch prediction benchmark
        batch_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.predict_batch(test_data[:self.batch_size])
            batch_times.append(time.time() - start_time)
        
        return {
            "single_prediction": {
                "avg_time": np.mean(single_times),
                "std_time": np.std(single_times),
                "min_time": np.min(single_times),
                "max_time": np.max(single_times)
            },
            "batch_prediction": {
                "avg_time": np.mean(batch_times),
                "std_time": np.std(batch_times),
                "min_time": np.min(batch_times),
                "max_time": np.max(batch_times),
                "throughput": self.batch_size / np.mean(batch_times)
            }
        }

# Example usage
def model_optimization_example():
    """Example of model optimization"""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create optimized server
    server = OptimizedModelServer(model, batch_size=32)
    
    # Test data
    test_data = [np.random.randn(784).tolist() for _ in range(100)]
    
    # Benchmark
    results = server.benchmark(test_data, num_runs=50)
    
    print("Model Performance Benchmark:")
    print(f"Single Prediction: {results['single_prediction']}")
    print(f"Batch Prediction: {results['batch_prediction']}")

if __name__ == "__main__":
    model_optimization_example()
```

### **2. Request Batching**

**Code Example**:
```python
import asyncio
import time
from typing import List, Dict, Any, Callable
from collections import deque
import threading

class RequestBatcher:
    def __init__(self, batch_size: int = 32, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batch_queue = deque()
        self.pending_requests = {}
        self.lock = threading.Lock()
        self.batch_id = 0
    
    async def add_request(self, request_data: Any, request_id: str) -> Any:
        """Add request to batch queue"""
        future = asyncio.Future()
        
        with self.lock:
            self.batch_queue.append({
                "data": request_data,
                "request_id": request_id,
                "future": future,
                "timestamp": time.time()
            })
            
            # Check if we should process batch
            if len(self.batch_queue) >= self.batch_size:
                await self._process_batch()
            else:
                # Schedule batch processing after max_wait_time
                asyncio.create_task(self._schedule_batch_processing())
        
        return await future
    
    async def _schedule_batch_processing(self):
        """Schedule batch processing after max_wait_time"""
        await asyncio.sleep(self.max_wait_time)
        
        with self.lock:
            if self.batch_queue:
                await self._process_batch()
    
    async def _process_batch(self):
        """Process current batch"""
        if not self.batch_queue:
            return
        
        # Extract batch
        batch = []
        futures = []
        
        with self.lock:
            batch_size = min(len(self.batch_queue), self.batch_size)
            for _ in range(batch_size):
                request = self.batch_queue.popleft()
                batch.append(request["data"])
                futures.append(request["future"])
        
        # Process batch
        try:
            results = await self._process_batch_data(batch)
            
            # Set results for each future
            for i, future in enumerate(futures):
                if i < len(results):
                    future.set_result(results[i])
                else:
                    future.set_exception(Exception("Batch processing error"))
        
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                future.set_exception(e)
    
    async def _process_batch_data(self, batch_data: List[Any]) -> List[Any]:
        """Process batch data (mock implementation)"""
        # Simulate batch processing
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Mock results
        results = []
        for data in batch_data:
            if isinstance(data, list):
                result = [sum(data) / len(data)]
            else:
                result = data
            results.append(result)
        
        return results

# Example usage
async def batching_example():
    """Example of request batching"""
    batcher = RequestBatcher(batch_size=5, max_wait_time=0.1)
    
    # Add multiple requests
    tasks = []
    for i in range(10):
        task = batcher.add_request([i, i+1, i+2], f"request_{i}")
        tasks.append(task)
    
    # Wait for all results
    results = await asyncio.gather(*tasks)
    
    print("Batching Results:")
    for i, result in enumerate(results):
        print(f"Request {i}: {result}")

if __name__ == "__main__":
    asyncio.run(batching_example())
```

---

## 🚀 **Real-time Systems**

### **1. WebSocket-based Real-time AI**

**Code Example**:
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
import time
from typing import List, Dict, Any
import numpy as np

app = FastAPI()

class RealTimeAIServer:
    def __init__(self):
        self.active_connections = []
        self.model_cache = {}
        self.prediction_cache = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)
    
    async def process_realtime_prediction(self, data: Dict[str, Any], websocket: WebSocket):
        """Process real-time prediction"""
        try:
            # Extract input data
            input_data = data.get("input", [])
            model_id = data.get("model_id", "default")
            
            # Check cache first
            cache_key = f"{model_id}:{hash(tuple(input_data))}"
            if cache_key in self.prediction_cache:
                result = self.prediction_cache[cache_key]
                result["cached"] = True
            else:
                # Make prediction
                start_time = time.time()
                prediction = await self._make_prediction(input_data, model_id)
                inference_time = time.time() - start_time
                
                result = {
                    "prediction": prediction,
                    "inference_time": inference_time,
                    "model_id": model_id,
                    "cached": False
                }
                
                # Cache result
                self.prediction_cache[cache_key] = result
            
            # Send result back to client
            await self.send_personal_message(json.dumps(result), websocket)
            
        except Exception as e:
            error_result = {"error": str(e)}
            await self.send_personal_message(json.dumps(error_result), websocket)
    
    async def _make_prediction(self, input_data: List[float], model_id: str) -> List[float]:
        """Make prediction (mock implementation)"""
        # Simulate model inference
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Mock prediction
        if model_id == "classification":
            prediction = [0.8, 0.2]  # Binary classification
        elif model_id == "regression":
            prediction = [sum(input_data) / len(input_data)]  # Regression
        else:
            prediction = [0.5]  # Default
        
        return prediction

# Initialize server
ai_server = RealTimeAIServer()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time AI"""
    await ai_server.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process message
            if message.get("type") == "prediction":
                await ai_server.process_realtime_prediction(message, websocket)
            elif message.get("type") == "ping":
                await ai_server.send_personal_message(json.dumps({"type": "pong"}), websocket)
            
    except WebSocketDisconnect:
        ai_server.disconnect(websocket)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Real-time AI Server", "connections": len(ai_server.active_connections)}

# Client example
class RealTimeAIClient:
    def __init__(self, websocket_url: str = "ws://localhost:8000/ws"):
        self.websocket_url = websocket_url
        self.websocket = None
    
    async def connect(self):
        """Connect to WebSocket server"""
        import websockets
        self.websocket = await websockets.connect(self.websocket_url)
    
    async def send_prediction_request(self, input_data: List[float], model_id: str = "default"):
        """Send prediction request"""
        message = {
            "type": "prediction",
            "input": input_data,
            "model_id": model_id
        }
        
        await self.websocket.send(json.dumps(message))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()

# Example usage
async def realtime_example():
    """Example of real-time AI client"""
    client = RealTimeAIClient()
    await client.connect()
    
    # Send prediction requests
    for i in range(5):
        input_data = [i, i+1, i+2, i+3]
        result = await client.send_prediction_request(input_data, "classification")
        print(f"Prediction {i}: {result}")
        
        # Wait between requests
        await asyncio.sleep(0.1)
    
    await client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### **2. Stream Processing**

**Code Example**:
```python
import asyncio
import time
from typing import AsyncGenerator, List, Dict, Any
import json

class StreamProcessor:
    def __init__(self, batch_size: int = 10, window_size: int = 100):
        self.batch_size = batch_size
        self.window_size = window_size
        self.data_window = []
        self.processing_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
    
    async def add_data(self, data: Dict[str, Any]):
        """Add data to processing stream"""
        await self.processing_queue.put(data)
    
    async def process_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Process data stream"""
        while True:
            try:
                # Get data from queue
                data = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                # Add to window
                self.data_window.append(data)
                
                # Maintain window size
                if len(self.data_window) > self.window_size:
                    self.data_window.pop(0)
                
                # Process if batch is ready
                if len(self.data_window) >= self.batch_size:
                    result = await self._process_batch(self.data_window[-self.batch_size:])
                    yield result
                
            except asyncio.TimeoutError:
                # Process remaining data if any
                if self.data_window:
                    result = await self._process_batch(self.data_window)
                    yield result
                    self.data_window.clear()
    
    async def _process_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch of data"""
        # Simulate processing
        await asyncio.sleep(0.01)
        
        # Mock processing result
        result = {
            "timestamp": time.time(),
            "batch_size": len(batch_data),
            "processed_data": [item.get("value", 0) for item in batch_data],
            "summary": {
                "count": len(batch_data),
                "avg": sum(item.get("value", 0) for item in batch_data) / len(batch_data)
            }
        }
        
        return result

# Example usage
async def stream_processing_example():
    """Example of stream processing"""
    processor = StreamProcessor(batch_size=5, window_size=20)
    
    # Start processing
    processing_task = asyncio.create_task(processor.process_stream())
    
    # Add data to stream
    for i in range(25):
        data = {"value": i, "timestamp": time.time()}
        await processor.add_data(data)
        await asyncio.sleep(0.05)  # Simulate data arrival
    
    # Wait for processing to complete
    await asyncio.sleep(1)
    processing_task.cancel()
    
    print("Stream processing completed")

if __name__ == "__main__":
    asyncio.run(stream_processing_example())
```

---

## 📊 **Performance Monitoring**

### **1. Latency Monitoring**

**Code Example**:
```python
import time
import statistics
from typing import List, Dict, Any
from collections import deque
import threading

class LatencyMonitor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.start_times = {}
    
    def start_request(self, request_id: str):
        """Start timing a request"""
        with self.lock:
            self.start_times[request_id] = time.time()
    
    def end_request(self, request_id: str):
        """End timing a request"""
        with self.lock:
            if request_id in self.start_times:
                latency = time.time() - self.start_times[request_id]
                self.latencies.append(latency)
                del self.start_times[request_id]
                return latency
        return None
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        with self.lock:
            if not self.latencies:
                return {}
            
            latencies_list = list(self.latencies)
            return {
                "count": len(latencies_list),
                "min": min(latencies_list),
                "max": max(latencies_list),
                "mean": statistics.mean(latencies_list),
                "median": statistics.median(latencies_list),
                "p95": self._percentile(latencies_list, 95),
                "p99": self._percentile(latencies_list, 99),
                "std": statistics.stdev(latencies_list) if len(latencies_list) > 1 else 0
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_recent_latencies(self, count: int = 100) -> List[float]:
        """Get recent latencies"""
        with self.lock:
            return list(self.latencies)[-count:]

# Example usage
def latency_monitoring_example():
    """Example of latency monitoring"""
    monitor = LatencyMonitor()
    
    # Simulate requests
    for i in range(100):
        request_id = f"request_{i}"
        monitor.start_request(request_id)
        
        # Simulate processing time
        time.sleep(0.01 + (i % 10) * 0.001)
        
        latency = monitor.end_request(request_id)
        print(f"Request {i}: {latency:.4f}s")
    
    # Get statistics
    stats = monitor.get_stats()
    print(f"Latency Statistics: {stats}")

if __name__ == "__main__":
    latency_monitoring_example()
```

---

## 🎯 **Interview Questions**

### **1. How do you optimize latency in AI systems?**

**Answer:**
- **Caching**: Cache predictions and intermediate results
- **Model Optimization**: Use quantization, pruning, and JIT compilation
- **Batch Processing**: Process multiple requests together
- **Hardware Acceleration**: Use GPUs and specialized hardware
- **Load Balancing**: Distribute requests across multiple instances
- **Connection Pooling**: Reuse database and service connections

### **2. What are the different types of caching strategies?**

**Answer:**
- **Write-through**: Write to cache and storage simultaneously
- **Write-behind**: Write to cache first, then to storage asynchronously
- **Write-around**: Write directly to storage, bypassing cache
- **Cache-aside**: Application manages cache explicitly
- **Read-through**: Cache loads data from storage on miss
- **Refresh-ahead**: Proactively refresh cache before expiration

### **3. How do you handle cache invalidation?**

**Answer:**
- **TTL (Time To Live)**: Automatic expiration after time
- **LRU (Least Recently Used)**: Remove least recently accessed items
- **LFU (Least Frequently Used)**: Remove least frequently accessed items
- **Event-based**: Invalidate based on data changes
- **Version-based**: Use version numbers to detect changes
- **Manual**: Explicit invalidation by application

### **4. What are the trade-offs of different caching levels?**

**Answer:**
- **L1 (In-memory)**: Fastest but limited size, lost on restart
- **L2 (Redis)**: Fast, persistent, but network overhead
- **L3 (Database)**: Slowest but most reliable and persistent
- **CDN**: Fast for static content, but limited to specific use cases
- **Browser**: Fastest for users, but limited control

### **5. How do you monitor and debug performance issues?**

**Answer:**
- **Metrics**: Track latency, throughput, error rates
- **Profiling**: Identify bottlenecks in code
- **Tracing**: Track requests across services
- **Logging**: Detailed logs for debugging
- **Alerting**: Proactive monitoring with alerts
- **Load Testing**: Simulate high load scenarios

---

**🎉 Caching and latency optimization are crucial for building high-performance AI systems!**
