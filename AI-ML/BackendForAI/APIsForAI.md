# 🌐 APIs for AI: Building Production-Ready AI APIs

> **Complete guide to designing, implementing, and scaling AI APIs for production**

## 🎯 **Learning Objectives**

- Master API design patterns for AI applications
- Implement RESTful and GraphQL APIs for ML models
- Handle authentication, rate limiting, and security
- Optimize API performance and scalability
- Monitor and maintain AI APIs in production

## 📚 **Table of Contents**

1. [API Design Patterns](#api-design-patterns)
2. [RESTful AI APIs](#restful-ai-apis)
3. [GraphQL for AI](#graphql-for-ai)
4. [Authentication & Security](#authentication--security)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring & Observability](#monitoring--observability)
7. [Interview Questions](#interview-questions)

---

## 🎨 **API Design Patterns**

### **Concept**

AI APIs require special considerations for handling model inference, batch processing, real-time predictions, and resource management.

### **Key Design Principles**

1. **Stateless**: APIs should not maintain state between requests
2. **Idempotent**: Multiple identical requests should produce the same result
3. **Scalable**: Handle varying loads and traffic patterns
4. **Secure**: Protect against attacks and unauthorized access
5. **Observable**: Provide metrics, logs, and traces

### **API Types for AI**

- **Synchronous**: Real-time predictions with immediate response
- **Asynchronous**: Batch processing with job queues
- **Streaming**: Real-time data processing with WebSockets
- **Webhook**: Event-driven processing with callbacks

---

## 🔄 **RESTful AI APIs**

### **1. Basic AI API Structure**

**Code Example**:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import uuid
from datetime import datetime

app = FastAPI(title="AI API", version="1.0.0")

# Request/Response Models
class PredictionRequest(BaseModel):
    input_data: List[float]
    model_version: Optional[str] = "latest"
    timeout: Optional[int] = 30

class PredictionResponse(BaseModel):
    prediction_id: str
    prediction: List[float]
    confidence: float
    model_version: str
    processing_time: float
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    inputs: List[List[float]]
    model_version: Optional[str] = "latest"
    callback_url: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    job_id: str
    status: str
    estimated_completion: datetime
    total_inputs: int

# In-memory storage for demo (use Redis/DB in production)
predictions = {}
batch_jobs = {}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Synchronous prediction endpoint"""
    try:
        start_time = datetime.now()

        # Validate input
        if not request.input_data:
            raise HTTPException(status_code=400, detail="Input data cannot be empty")

        # Generate prediction ID
        prediction_id = str(uuid.uuid4())

        # Simulate model inference
        await asyncio.sleep(0.1)  # Simulate processing time

        # Mock prediction (replace with actual model)
        prediction = [sum(request.input_data) / len(request.input_data)]
        confidence = 0.95

        processing_time = (datetime.now() - start_time).total_seconds()

        response = PredictionResponse(
            prediction_id=prediction_id,
            prediction=prediction,
            confidence=confidence,
            model_version=request.model_version,
            processing_time=processing_time,
            timestamp=datetime.now()
        )

        # Store prediction for reference
        predictions[prediction_id] = response

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Asynchronous batch prediction endpoint"""
    try:
        job_id = str(uuid.uuid4())

        # Store job information
        batch_jobs[job_id] = {
            "status": "queued",
            "total_inputs": len(request.inputs),
            "completed": 0,
            "results": [],
            "created_at": datetime.now()
        }

        # Start background processing
        background_tasks.add_task(process_batch, job_id, request.inputs, request.model_version)

        return BatchPredictionResponse(
            job_id=job_id,
            status="queued",
            estimated_completion=datetime.now(),
            total_inputs=len(request.inputs)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch(job_id: str, inputs: List[List[float]], model_version: str):
    """Background task for batch processing"""
    try:
        batch_jobs[job_id]["status"] = "processing"

        results = []
        for i, input_data in enumerate(inputs):
            # Simulate processing
            await asyncio.sleep(0.1)

            # Mock prediction
            prediction = [sum(input_data) / len(input_data)]
            confidence = 0.95

            results.append({
                "input_index": i,
                "prediction": prediction,
                "confidence": confidence
            })

            batch_jobs[job_id]["completed"] = i + 1

        batch_jobs[job_id]["results"] = results
        batch_jobs[job_id]["status"] = "completed"

    except Exception as e:
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["error"] = str(e)

@app.get("/predict/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Get batch prediction status"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return batch_jobs[job_id]

@app.get("/predict/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Get prediction by ID"""
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return predictions[prediction_id]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    return {
        "total_predictions": len(predictions),
        "active_batch_jobs": len([job for job in batch_jobs.values() if job["status"] == "processing"]),
        "completed_batch_jobs": len([job for job in batch_jobs.values() if job["status"] == "completed"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### **2. Advanced API Features**

**Code Example**:

```python
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
import time
import json
from typing import AsyncGenerator

app = FastAPI()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting (simplified)
from collections import defaultdict
import time

rate_limit_storage = defaultdict(list)

def rate_limit(max_requests: int = 100, window: int = 60):
    """Simple rate limiting decorator"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            current_time = time.time()

            # Clean old requests
            rate_limit_storage[client_ip] = [
                req_time for req_time in rate_limit_storage[client_ip]
                if current_time - req_time < window
            ]

            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Add current request
            rate_limit_storage[client_ip].append(current_time)

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

@app.post("/predict/stream")
@rate_limit(max_requests=10, window=60)
async def predict_stream(request: PredictionRequest):
    """Streaming prediction endpoint"""
    async def generate_stream():
        yield f"data: {json.dumps({'status': 'started', 'timestamp': time.time()})}\n\n"

        # Simulate streaming prediction
        for i in range(5):
            await asyncio.sleep(0.2)
            progress = (i + 1) / 5 * 100
            yield f"data: {json.dumps({'progress': progress, 'step': f'Processing step {i+1}'})}\n\n"

        # Final result
        prediction = [sum(request.input_data) / len(request.input_data)]
        yield f"data: {json.dumps({'status': 'completed', 'prediction': prediction})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/plain")

# Model versioning
@app.post("/predict/v1")
async def predict_v1(request: PredictionRequest):
    """Version 1 prediction endpoint"""
    # Legacy model logic
    pass

@app.post("/predict/v2")
async def predict_v2(request: PredictionRequest):
    """Version 2 prediction endpoint"""
    # New model logic
    pass

# A/B testing endpoint
@app.post("/predict/ab-test")
async def predict_ab_test(request: PredictionRequest):
    """A/B testing endpoint"""
    import random

    # Randomly choose model version
    model_version = "v1" if random.random() < 0.5 else "v2"

    if model_version == "v1":
        return await predict_v1(request)
    else:
        return await predict_v2(request)
```

---

## 🔍 **GraphQL for AI**

### **Concept**

GraphQL provides a flexible query language for APIs, allowing clients to request exactly the data they need.

### **Code Example**

```python
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
import strawberry
from typing import List, Optional
from datetime import datetime

@strawberry.type
class Prediction:
    id: str
    prediction: List[float]
    confidence: float
    model_version: str
    timestamp: datetime

@strawberry.type
class BatchJob:
    id: str
    status: str
    total_inputs: int
    completed: int
    results: Optional[List[Prediction]]

@strawberry.input
class PredictionInput:
    input_data: List[float]
    model_version: Optional[str] = "latest"

@strawberry.input
class BatchPredictionInput:
    inputs: List[List[float]]
    model_version: Optional[str] = "latest"

@strawberry.type
class Query:
    @strawberry.field
    def prediction(self, id: str) -> Optional[Prediction]:
        """Get prediction by ID"""
        return predictions.get(id)

    @strawberry.field
    def predictions(self, limit: int = 10) -> List[Prediction]:
        """Get recent predictions"""
        return list(predictions.values())[-limit:]

    @strawberry.field
    def batch_job(self, id: str) -> Optional[BatchJob]:
        """Get batch job by ID"""
        job = batch_jobs.get(id)
        if not job:
            return None

        return BatchJob(
            id=id,
            status=job["status"],
            total_inputs=job["total_inputs"],
            completed=job["completed"],
            results=job.get("results", [])
        )

@strawberry.type
class Mutation:
    @strawberry.field
    def predict(self, input: PredictionInput) -> Prediction:
        """Make a prediction"""
        prediction_id = str(uuid.uuid4())

        # Simulate model inference
        prediction = [sum(input.input_data) / len(input.input_data)]
        confidence = 0.95

        pred = Prediction(
            id=prediction_id,
            prediction=prediction,
            confidence=confidence,
            model_version=input.model_version,
            timestamp=datetime.now()
        )

        predictions[prediction_id] = pred
        return pred

    @strawberry.field
    def predict_batch(self, input: BatchPredictionInput) -> BatchJob:
        """Start batch prediction"""
        job_id = str(uuid.uuid4())

        batch_jobs[job_id] = {
            "status": "queued",
            "total_inputs": len(input.inputs),
            "completed": 0,
            "results": [],
            "created_at": datetime.now()
        }

        return BatchJob(
            id=job_id,
            status="queued",
            total_inputs=len(input.inputs),
            completed=0,
            results=[]
        )

# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)

# Add to FastAPI app
app.include_router(graphql_app, prefix="/graphql")
```

---

## 🔐 **Authentication & Security**

### **1. JWT Authentication**

**Code Example**:

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()
security = HTTPBearer()

# JWT configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/auth/login")
async def login(username: str, password: str):
    """Login endpoint"""
    # Validate credentials (implement your logic)
    if username == "admin" and password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/predict/secure")
async def predict_secure(
    request: PredictionRequest,
    current_user: str = Depends(verify_token)
):
    """Secure prediction endpoint"""
    # Only authenticated users can access this endpoint
    return await predict(request)
```

### **2. API Key Authentication**

**Code Example**:

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
import hashlib
import secrets

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")

# In-memory API key storage (use database in production)
api_keys = {
    "sk-1234567890abcdef": {"user_id": "user1", "permissions": ["predict", "batch"]},
    "sk-abcdef1234567890": {"user_id": "user2", "permissions": ["predict"]},
}

def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key"""
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_keys[api_key]

def check_permission(required_permission: str):
    """Check if user has required permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get API key from dependencies
            api_key_info = kwargs.get("api_key_info")
            if not api_key_info:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )

            if required_permission not in api_key_info["permissions"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/predict/api-key")
async def predict_with_api_key(
    request: PredictionRequest,
    api_key_info: dict = Depends(verify_api_key)
):
    """Prediction endpoint with API key authentication"""
    return await predict(request)

@app.post("/predict/batch/api-key")
@check_permission("batch")
async def predict_batch_with_api_key(
    request: BatchPredictionRequest,
    api_key_info: dict = Depends(verify_api_key)
):
    """Batch prediction endpoint with API key authentication"""
    return await predict_batch(request, BackgroundTasks())
```

---

## ⚡ **Performance Optimization**

### **1. Caching**

**Code Example**:

```python
from fastapi import FastAPI, Request
from functools import lru_cache
import redis
import json
import hashlib

app = FastAPI()

# Redis client for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_key(request: PredictionRequest) -> str:
    """Generate cache key from request"""
    key_data = {
        "input_data": request.input_data,
        "model_version": request.model_version
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

@app.post("/predict/cached")
async def predict_cached(request: PredictionRequest):
    """Cached prediction endpoint"""
    # Generate cache key
    key = cache_key(request)

    # Check cache
    cached_result = redis_client.get(key)
    if cached_result:
        return json.loads(cached_result)

    # Make prediction
    result = await predict(request)

    # Cache result (expire in 1 hour)
    redis_client.setex(key, 3600, json.dumps(result.dict()))

    return result

# In-memory caching with LRU
@lru_cache(maxsize=1000)
def expensive_computation(input_data: tuple) -> float:
    """Expensive computation that can be cached"""
    # Simulate expensive computation
    import time
    time.sleep(0.1)
    return sum(input_data) / len(input_data)

@app.post("/predict/lru-cached")
async def predict_lru_cached(request: PredictionRequest):
    """LRU cached prediction endpoint"""
    input_tuple = tuple(request.input_data)
    result = expensive_computation(input_tuple)

    return {
        "prediction": [result],
        "confidence": 0.95,
        "cached": True
    }
```

### **2. Async Processing**

**Code Example**:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

async def cpu_intensive_prediction(input_data: List[float]) -> List[float]:
    """CPU-intensive prediction that runs in thread pool"""
    def _predict():
        # Simulate CPU-intensive computation
        result = np.array(input_data)
        for _ in range(1000):
            result = np.sin(result) + np.cos(result)
        return result.tolist()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _predict)

@app.post("/predict/async")
async def predict_async(request: PredictionRequest):
    """Async prediction endpoint"""
    # Run multiple predictions concurrently
    tasks = [
        cpu_intensive_prediction(request.input_data)
        for _ in range(3)  # Run 3 predictions in parallel
    ]

    results = await asyncio.gather(*tasks)

    # Average the results
    avg_result = [sum(col) / len(col) for col in zip(*results)]

    return {
        "prediction": avg_result,
        "confidence": 0.95,
        "parallel_predictions": len(results)
    }
```

---

## 📊 **Monitoring & Observability**

### **1. Metrics Collection**

**Code Example**:

```python
from fastapi import FastAPI, Request
import time
from collections import defaultdict
import logging

app = FastAPI()

# Metrics storage
metrics = {
    "request_count": 0,
    "request_duration": [],
    "error_count": 0,
    "predictions_by_model": defaultdict(int)
}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics"""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Update metrics
    metrics["request_count"] += 1
    metrics["request_duration"].append(duration)

    # Keep only last 1000 durations
    if len(metrics["request_duration"]) > 1000:
        metrics["request_duration"] = metrics["request_duration"][-1000:]

    # Count errors
    if response.status_code >= 400:
        metrics["error_count"] += 1

    return response

@app.post("/predict/metrics")
async def predict_with_metrics(request: PredictionRequest):
    """Prediction endpoint with metrics"""
    # Update model-specific metrics
    metrics["predictions_by_model"][request.model_version] += 1

    return await predict(request)

@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    durations = metrics["request_duration"]
    avg_duration = sum(durations) / len(durations) if durations else 0

    return {
        "request_count": metrics["request_count"],
        "error_count": metrics["error_count"],
        "error_rate": metrics["error_count"] / max(metrics["request_count"], 1),
        "average_duration": avg_duration,
        "predictions_by_model": dict(metrics["predictions_by_model"])
    }
```

### **2. Logging**

**Code Example**:

```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_prediction(request: PredictionRequest, response: PredictionResponse, duration: float):
    """Log prediction details"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": "prediction",
        "request_id": response.prediction_id,
        "input_size": len(request.input_data),
        "model_version": request.model_version,
        "duration": duration,
        "confidence": response.confidence,
        "status": "success"
    }

    logger.info(json.dumps(log_data))

@app.post("/predict/logged")
async def predict_logged(request: PredictionRequest):
    """Prediction endpoint with logging"""
    start_time = time.time()

    try:
        response = await predict(request)
        duration = time.time() - start_time

        # Log successful prediction
        log_prediction(request, response, duration)

        return response

    except Exception as e:
        duration = time.time() - start_time

        # Log error
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event": "prediction_error",
            "error": str(e),
            "duration": duration,
            "status": "error"
        }

        logger.error(json.dumps(log_data))
        raise
```

---

## 🎯 **Interview Questions**

### **1. How do you design an API for ML model serving?**

**Answer:**

- **Stateless Design**: APIs should not maintain state between requests
- **Versioning**: Support multiple model versions for A/B testing
- **Async Processing**: Handle both real-time and batch predictions
- **Error Handling**: Graceful degradation and meaningful error messages
- **Rate Limiting**: Prevent abuse and ensure fair usage
- **Monitoring**: Comprehensive metrics and logging

### **2. What are the differences between synchronous and asynchronous AI APIs?**

**Answer:**
**Synchronous APIs:**

- Immediate response required
- Good for real-time applications
- Limited by processing time
- Simple client implementation

**Asynchronous APIs:**

- Background processing with job queues
- Good for batch processing
- Can handle large volumes
- Requires polling or webhooks for results

### **3. How do you handle authentication in AI APIs?**

**Answer:**

- **API Keys**: Simple authentication for programmatic access
- **JWT Tokens**: Stateless authentication with expiration
- **OAuth2**: Standardized authorization framework
- **Rate Limiting**: Prevent abuse and ensure fair usage
- **Permission-based**: Different access levels for different users

### **4. What are the key considerations for scaling AI APIs?**

**Answer:**

- **Load Balancing**: Distribute requests across multiple instances
- **Caching**: Cache predictions for repeated inputs
- **Async Processing**: Use queues for batch processing
- **Resource Management**: Monitor CPU, memory, and GPU usage
- **Auto-scaling**: Automatically scale based on demand
- **Database Optimization**: Efficient storage and retrieval

### **5. How do you monitor and debug AI APIs in production?**

**Answer:**

- **Metrics**: Request count, latency, error rate, throughput
- **Logging**: Structured logs with correlation IDs
- **Tracing**: Distributed tracing for request flow
- **Alerting**: Proactive monitoring with alerts
- **Health Checks**: Regular health checks and status endpoints
- **Performance Profiling**: Identify bottlenecks and optimize

---

**🎉 Building production-ready AI APIs requires careful consideration of performance, security, and scalability!**
