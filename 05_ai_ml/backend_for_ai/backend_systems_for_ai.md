# Backend Systems for AI

Comprehensive guide to building backend systems that support AI/ML applications.

## 🎯 AI Infrastructure Architecture

### Model Serving Infrastructure
```python
# Model Serving Service
from typing import Dict, List, Any, Optional, Union
import asyncio
import aiohttp
import json
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
import logging

class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADING = "unloading"

@dataclass
class ModelInfo:
    name: str
    version: str
    path: str
    framework: str
    input_shape: List[int]
    output_shape: List[int]
    memory_usage: int
    load_time: float
    status: ModelStatus

@dataclass
class PredictionRequest:
    model_name: str
    inputs: Union[List, Dict, np.ndarray]
    request_id: str
    timeout: float = 30.0
    metadata: Dict[str, Any] = None

@dataclass
class PredictionResponse:
    request_id: str
    outputs: Union[List, Dict, np.ndarray]
    inference_time: float
    model_name: str
    status: str
    error: Optional[str] = None

class ModelServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, ModelInfo] = {}
        self.model_instances: Dict[str, Any] = {}
        self.request_queue = asyncio.Queue()
        self.workers = config.get('workers', 4)
        self.max_batch_size = config.get('max_batch_size', 32)
        self.batch_timeout = config.get('batch_timeout', 0.1)
        self.logger = logging.getLogger(__name__)
        
        # Start worker tasks
        for i in range(self.workers):
            asyncio.create_task(self._worker(f"worker-{i}"))
    
    async def load_model(self, model_name: str, model_path: str, 
                        framework: str = "pytorch") -> bool:
        """Load a model for serving"""
        try:
            self.logger.info(f"Loading model {model_name} from {model_path}")
            
            # Update model status
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.LOADING
            else:
                self.models[model_name] = ModelInfo(
                    name=model_name,
                    version="1.0",
                    path=model_path,
                    framework=framework,
                    input_shape=[],
                    output_shape=[],
                    memory_usage=0,
                    load_time=0,
                    status=ModelStatus.LOADING
                )
            
            start_time = time.time()
            
            # Load model based on framework
            if framework == "pytorch":
                model_instance = await self._load_pytorch_model(model_path)
            elif framework == "tensorflow":
                model_instance = await self._load_tensorflow_model(model_path)
            elif framework == "onnx":
                model_instance = await self._load_onnx_model(model_path)
            else:
                raise ValueError(f"Unsupported framework: {framework}")
            
            load_time = time.time() - start_time
            
            # Update model info
            self.models[model_name].status = ModelStatus.READY
            self.models[model_name].load_time = load_time
            self.models[model_name].memory_usage = self._get_model_memory_usage(model_instance)
            
            self.model_instances[model_name] = model_instance
            
            self.logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.ERROR
            return False
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a prediction using the specified model"""
        if request.model_name not in self.models:
            return PredictionResponse(
                request_id=request.request_id,
                outputs=None,
                inference_time=0,
                model_name=request.model_name,
                status="error",
                error=f"Model {request.model_name} not found"
            )
        
        if self.models[request.model_name].status != ModelStatus.READY:
            return PredictionResponse(
                request_id=request.request_id,
                outputs=None,
                inference_time=0,
                model_name=request.model_name,
                status="error",
                error=f"Model {request.model_name} is not ready"
            )
        
        try:
            start_time = time.time()
            
            # Get model instance
            model = self.model_instances[request.model_name]
            
            # Preprocess inputs
            processed_inputs = await self._preprocess_inputs(request.inputs, request.model_name)
            
            # Make prediction
            outputs = await self._run_inference(model, processed_inputs)
            
            # Postprocess outputs
            processed_outputs = await self._postprocess_outputs(outputs, request.model_name)
            
            inference_time = time.time() - start_time
            
            return PredictionResponse(
                request_id=request.request_id,
                outputs=processed_outputs,
                inference_time=inference_time,
                model_name=request.model_name,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {request.model_name}: {e}")
            return PredictionResponse(
                request_id=request.request_id,
                outputs=None,
                inference_time=0,
                model_name=request.model_name,
                status="error",
                error=str(e)
            )
    
    async def _worker(self, worker_name: str):
        """Worker task for processing predictions"""
        self.logger.info(f"Starting worker {worker_name}")
        
        while True:
            try:
                # Get request from queue
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=1.0
                )
                
                # Process request
                response = await self.predict(request)
                
                # Send response (implement based on your needs)
                await self._send_response(response)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
    
    async def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model"""
        import torch
        
        # Load model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        return model
    
    async def _load_tensorflow_model(self, model_path: str):
        """Load TensorFlow model"""
        import tensorflow as tf
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        return model
    
    async def _load_onnx_model(self, model_path: str):
        """Load ONNX model"""
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(model_path)
        
        return session
    
    async def _preprocess_inputs(self, inputs: Any, model_name: str) -> Any:
        """Preprocess inputs for the model"""
        # Implement preprocessing logic based on model requirements
        return inputs
    
    async def _postprocess_outputs(self, outputs: Any, model_name: str) -> Any:
        """Postprocess outputs from the model"""
        # Implement postprocessing logic based on model requirements
        return outputs
    
    async def _run_inference(self, model: Any, inputs: Any) -> Any:
        """Run inference on the model"""
        # Implement inference logic based on framework
        if hasattr(model, 'predict'):
            return model.predict(inputs)
        elif hasattr(model, 'forward'):
            return model.forward(inputs)
        else:
            # ONNX runtime
            return model.run(None, inputs)
    
    def _get_model_memory_usage(self, model: Any) -> int:
        """Get memory usage of the model"""
        # Implement memory usage calculation
        return 0
    
    async def _send_response(self, response: PredictionResponse):
        """Send response back to client"""
        # Implement response sending logic
        pass
```

### Data Pipeline for AI
```python
# AI Data Pipeline
from typing import Dict, List, Any, Optional, Callable
import asyncio
import aiofiles
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class DataSource:
    name: str
    type: str  # 'file', 'database', 'api', 'stream'
    config: Dict[str, Any]
    schema: Dict[str, str] = None

@dataclass
class DataTransform:
    name: str
    function: Callable
    input_columns: List[str]
    output_columns: List[str]
    parameters: Dict[str, Any] = None

@dataclass
class DataSink:
    name: str
    type: str  # 'file', 'database', 'api', 'model'
    config: Dict[str, Any]

class AIDataPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources: Dict[str, DataSource] = {}
        self.transforms: List[DataTransform] = []
        self.sinks: Dict[str, DataSink] = {}
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def add_source(self, source: DataSource):
        """Add a data source to the pipeline"""
        self.sources[source.name] = source
    
    def add_transform(self, transform: DataTransform):
        """Add a data transform to the pipeline"""
        self.transforms.append(transform)
    
    def add_sink(self, sink: DataSink):
        """Add a data sink to the pipeline"""
        self.sinks[sink.name] = sink
    
    async def run_pipeline(self, source_name: str, sink_name: str, 
                          cache_key: str = None) -> Dict[str, Any]:
        """Run the data pipeline from source to sink"""
        try:
            # Check cache
            if cache_key and cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.logger.info(f"Using cached data for {cache_key}")
                    return cache_entry['data']
            
            # Load data from source
            data = await self._load_data(source_name)
            
            # Apply transforms
            for transform in self.transforms:
                data = await self._apply_transform(data, transform)
            
            # Save data to sink
            result = await self._save_data(data, sink_name)
            
            # Cache result
            if cache_key:
                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time()
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def _load_data(self, source_name: str) -> pd.DataFrame:
        """Load data from the specified source"""
        if source_name not in self.sources:
            raise ValueError(f"Source {source_name} not found")
        
        source = self.sources[source_name]
        
        if source.type == 'file':
            return await self._load_from_file(source)
        elif source.type == 'database':
            return await self._load_from_database(source)
        elif source.type == 'api':
            return await self._load_from_api(source)
        elif source.type == 'stream':
            return await self._load_from_stream(source)
        else:
            raise ValueError(f"Unsupported source type: {source.type}")
    
    async def _load_from_file(self, source: DataSource) -> pd.DataFrame:
        """Load data from file"""
        file_path = source.config['path']
        file_format = source.config.get('format', 'csv')
        
        if file_format == 'csv':
            return pd.read_csv(file_path)
        elif file_format == 'json':
            return pd.read_json(file_path)
        elif file_format == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    async def _load_from_database(self, source: DataSource) -> pd.DataFrame:
        """Load data from database"""
        # Implement database loading logic
        pass
    
    async def _load_from_api(self, source: DataSource) -> pd.DataFrame:
        """Load data from API"""
        # Implement API loading logic
        pass
    
    async def _load_from_stream(self, source: DataSource) -> pd.DataFrame:
        """Load data from stream"""
        # Implement stream loading logic
        pass
    
    async def _apply_transform(self, data: pd.DataFrame, transform: DataTransform) -> pd.DataFrame:
        """Apply a transform to the data"""
        try:
            # Prepare inputs
            inputs = data[transform.input_columns].values
            
            # Apply transform function
            if transform.parameters:
                outputs = transform.function(inputs, **transform.parameters)
            else:
                outputs = transform.function(inputs)
            
            # Add outputs to dataframe
            if len(transform.output_columns) == 1:
                data[transform.output_columns[0]] = outputs
            else:
                for i, col in enumerate(transform.output_columns):
                    data[col] = outputs[:, i] if outputs.ndim > 1 else outputs
            
            return data
            
        except Exception as e:
            self.logger.error(f"Transform {transform.name} failed: {e}")
            raise
    
    async def _save_data(self, data: pd.DataFrame, sink_name: str) -> Dict[str, Any]:
        """Save data to the specified sink"""
        if sink_name not in self.sinks:
            raise ValueError(f"Sink {sink_name} not found")
        
        sink = self.sinks[sink_name]
        
        if sink.type == 'file':
            return await self._save_to_file(data, sink)
        elif sink.type == 'database':
            return await self._save_to_database(data, sink)
        elif sink.type == 'api':
            return await self._save_to_api(data, sink)
        elif sink.type == 'model':
            return await self._save_to_model(data, sink)
        else:
            raise ValueError(f"Unsupported sink type: {sink.type}")
    
    async def _save_to_file(self, data: pd.DataFrame, sink: DataSink) -> Dict[str, Any]:
        """Save data to file"""
        file_path = sink.config['path']
        file_format = sink.config.get('format', 'csv')
        
        if file_format == 'csv':
            data.to_csv(file_path, index=False)
        elif file_format == 'json':
            data.to_json(file_path, orient='records')
        elif file_format == 'parquet':
            data.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return {'status': 'success', 'rows': len(data)}
    
    async def _save_to_database(self, data: pd.DataFrame, sink: DataSink) -> Dict[str, Any]:
        """Save data to database"""
        # Implement database saving logic
        pass
    
    async def _save_to_api(self, data: pd.DataFrame, sink: DataSink) -> Dict[str, Any]:
        """Save data to API"""
        # Implement API saving logic
        pass
    
    async def _save_to_model(self, data: pd.DataFrame, sink: DataSink) -> Dict[str, Any]:
        """Save data to model"""
        # Implement model saving logic
        pass
```

## 🚀 MLOps and Model Management

### Model Registry and Versioning
```python
# Model Registry Service
from typing import Dict, List, Any, Optional
import asyncio
import aiofiles
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelVersion:
    name: str
    version: str
    stage: ModelStage
    path: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: str
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['stage'] = self.stage.value
        data['created_at'] = self.created_at.isoformat()
        return data

class ModelRegistry:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, List[ModelVersion]] = {}
        self.storage_path = config.get('storage_path', './model_registry')
        self.logger = logging.getLogger(__name__)
    
    async def register_model(self, model_version: ModelVersion) -> bool:
        """Register a new model version"""
        try:
            # Validate model version
            if not self._validate_model_version(model_version):
                return False
            
            # Add to registry
            if model_version.name not in self.models:
                self.models[model_version.name] = []
            
            self.models[model_version.name].append(model_version)
            
            # Save to storage
            await self._save_model_version(model_version)
            
            self.logger.info(f"Registered model {model_version.name} version {model_version.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return False
    
    async def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        return self.models.get(model_name, [])
    
    async def get_latest_version(self, model_name: str, stage: ModelStage = None) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = self.models.get(model_name, [])
        
        if not versions:
            return None
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        if not versions:
            return None
        
        # Sort by creation time and return latest
        return max(versions, key=lambda v: v.created_at)
    
    async def promote_model(self, model_name: str, version: str, 
                           new_stage: ModelStage) -> bool:
        """Promote a model to a new stage"""
        try:
            versions = self.models.get(model_name, [])
            model_version = next((v for v in versions if v.version == version), None)
            
            if not model_version:
                self.logger.error(f"Model {model_name} version {version} not found")
                return False
            
            # Update stage
            model_version.stage = new_stage
            
            # Save updated version
            await self._save_model_version(model_version)
            
            self.logger.info(f"Promoted model {model_name} version {version} to {new_stage.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to promote model: {e}")
            return False
    
    async def compare_models(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        versions = self.models.get(model_name, [])
        v1 = next((v for v in versions if v.version == version1), None)
        v2 = next((v for v in versions if v.version == version2), None)
        
        if not v1 or not v2:
            raise ValueError("One or both model versions not found")
        
        comparison = {
            'model_name': model_name,
            'version1': v1.to_dict(),
            'version2': v2.to_dict(),
            'metrics_comparison': {},
            'metadata_comparison': {}
        }
        
        # Compare metrics
        for metric in set(v1.metrics.keys()) | set(v2.metrics.keys()):
            val1 = v1.metrics.get(metric, 0)
            val2 = v2.metrics.get(metric, 0)
            comparison['metrics_comparison'][metric] = {
                'version1': val1,
                'version2': val2,
                'difference': val2 - val1,
                'improvement': val2 > val1 if 'accuracy' in metric.lower() or 'f1' in metric.lower() else val2 < val1
            }
        
        return comparison
    
    def _validate_model_version(self, model_version: ModelVersion) -> bool:
        """Validate a model version"""
        # Check required fields
        if not model_version.name or not model_version.version:
            return False
        
        # Check if version already exists
        existing_versions = self.models.get(model_version.name, [])
        if any(v.version == model_version.version for v in existing_versions):
            self.logger.warning(f"Model {model_version.name} version {model_version.version} already exists")
            return False
        
        return True
    
    async def _save_model_version(self, model_version: ModelVersion):
        """Save model version to storage"""
        # Create directory if it doesn't exist
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Save model version metadata
        file_path = os.path.join(self.storage_path, f"{model_version.name}_{model_version.version}.json")
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(model_version.to_dict(), indent=2))
    
    async def load_registry(self):
        """Load model registry from storage"""
        try:
            import os
            import glob
            
            # Load all model version files
            pattern = os.path.join(self.storage_path, "*.json")
            files = glob.glob(pattern)
            
            for file_path in files:
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                    
                    # Convert back to ModelVersion
                    model_version = ModelVersion(
                        name=data['name'],
                        version=data['version'],
                        stage=ModelStage(data['stage']),
                        path=data['path'],
                        metrics=data['metrics'],
                        metadata=data['metadata'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        created_by=data['created_by'],
                        description=data.get('description', '')
                    )
                    
                    # Add to registry
                    if model_version.name not in self.models:
                        self.models[model_version.name] = []
                    self.models[model_version.name].append(model_version)
            
            self.logger.info(f"Loaded {len(files)} model versions from registry")
            
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
```

## 🎯 Best Practices

### Performance Optimization
1. **Model Optimization**: Use quantization, pruning, and distillation
2. **Caching**: Implement model and prediction caching
3. **Batching**: Process multiple requests in batches
4. **Load Balancing**: Distribute load across multiple model instances
5. **Resource Management**: Monitor and optimize resource usage

### Security and Privacy
1. **Data Privacy**: Implement data anonymization and encryption
2. **Model Security**: Protect models from adversarial attacks
3. **Access Control**: Implement proper authentication and authorization
4. **Audit Logging**: Log all model-related activities
5. **Compliance**: Ensure compliance with regulations

### Monitoring and Observability
1. **Model Performance**: Monitor model accuracy and drift
2. **System Metrics**: Track latency, throughput, and resource usage
3. **Business Metrics**: Monitor business impact of models
4. **Alerting**: Set up alerts for anomalies and failures
5. **Dashboards**: Create comprehensive monitoring dashboards

---

**Last Updated**: December 2024  
**Category**: Backend Systems for AI  
**Complexity**: Expert Level
