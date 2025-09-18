# Advanced ML Systems Comprehensive

Comprehensive guide to advanced machine learning systems for senior backend engineers.

## 🎯 MLOps and Production Systems

### Model Serving Infrastructure
```python
# Advanced Model Serving with TensorFlow Serving
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import grpc
import numpy as np
from typing import Dict, Any, List
import asyncio
import logging

class AdvancedModelServer:
    def __init__(self, model_path: str, model_name: str, 
                 max_batch_size: int = 32, timeout: int = 30):
        self.model_path = model_path
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.model = None
        self.preprocessor = None
        self.postprocessor = None
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
        
    async def load_model(self):
        """Load and initialize the model"""
        try:
            # Load the model
            self.model = tf.saved_model.load(self.model_path)
            
            # Initialize preprocessor and postprocessor
            self.preprocessor = self._create_preprocessor()
            self.postprocessor = self._create_postprocessor()
            
            # Warm up the model
            await self._warmup_model()
            
            self.logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    async def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions on input data"""
        try:
            # Preprocess inputs
            processed_inputs = await self.preprocessor.process(inputs)
            
            # Make prediction
            predictions = self.model(processed_inputs)
            
            # Postprocess outputs
            outputs = await self.postprocessor.process(predictions)
            
            # Update metrics
            self._update_metrics(len(processed_inputs))
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    async def batch_predict(self, batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        results = []
        
        # Process in batches
        for i in range(0, len(batch_inputs), self.max_batch_size):
            batch = batch_inputs[i:i + self.max_batch_size]
            batch_results = await self.predict_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def predict_batch(self, batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a single batch of inputs"""
        try:
            # Preprocess batch
            processed_batch = await self.preprocessor.process_batch(batch_inputs)
            
            # Make batch prediction
            batch_predictions = self.model(processed_batch)
            
            # Postprocess batch
            batch_outputs = await self.postprocessor.process_batch(batch_predictions)
            
            return batch_outputs
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _create_preprocessor(self):
        """Create input preprocessor"""
        class Preprocessor:
            def __init__(self, model_input_spec):
                self.model_input_spec = model_input_spec
            
            async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                # Implement preprocessing logic
                processed = {}
                for key, value in inputs.items():
                    if key in self.model_input_spec:
                        processed[key] = self._preprocess_value(value, key)
                return processed
            
            async def process_batch(self, batch_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
                # Implement batch preprocessing
                batch_processed = {}
                for key in self.model_input_spec.keys():
                    batch_processed[key] = np.array([
                        self._preprocess_value(inputs.get(key), key) 
                        for inputs in batch_inputs
                    ])
                return batch_processed
            
            def _preprocess_value(self, value: Any, key: str) -> Any:
                # Implement specific preprocessing for each input
                if key == "text":
                    return self._preprocess_text(value)
                elif key == "image":
                    return self._preprocess_image(value)
                else:
                    return value
            
            def _preprocess_text(self, text: str) -> str:
                # Text preprocessing logic
                return text.lower().strip()
            
            def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
                # Image preprocessing logic
                return image / 255.0
        
        return Preprocessor(self.model.signatures['serving_default'].inputs)
    
    def _create_postprocessor(self):
        """Create output postprocessor"""
        class Postprocessor:
            def __init__(self, model_output_spec):
                self.model_output_spec = model_output_spec
            
            async def process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
                # Implement postprocessing logic
                processed = {}
                for key, value in predictions.items():
                    processed[key] = self._postprocess_value(value, key)
                return processed
            
            async def process_batch(self, batch_predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
                # Implement batch postprocessing
                batch_size = len(next(iter(batch_predictions.values())))
                results = []
                
                for i in range(batch_size):
                    single_prediction = {
                        key: value[i] for key, value in batch_predictions.items()
                    }
                    processed = await self.process(single_prediction)
                    results.append(processed)
                
                return results
            
            def _postprocess_value(self, value: Any, key: str) -> Any:
                # Implement specific postprocessing for each output
                if key == "probabilities":
                    return tf.nn.softmax(value).numpy()
                elif key == "logits":
                    return value.numpy()
                else:
                    return value.numpy()
        
        return Postprocessor(self.model.signatures['serving_default'].outputs)
    
    async def _warmup_model(self):
        """Warm up the model with dummy data"""
        dummy_inputs = self._create_dummy_inputs()
        await self.predict(dummy_inputs)
    
    def _create_dummy_inputs(self) -> Dict[str, Any]:
        """Create dummy inputs for warmup"""
        # Implement based on your model's input requirements
        return {
            "text": "dummy text",
            "image": np.random.rand(224, 224, 3)
        }
    
    def _update_metrics(self, batch_size: int):
        """Update serving metrics"""
        self.metrics["total_requests"] = self.metrics.get("total_requests", 0) + 1
        self.metrics["total_samples"] = self.metrics.get("total_samples", 0) + batch_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
```

### Model Registry and Versioning
```python
# Advanced Model Registry with Versioning
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, List, Optional
import os
import json
import hashlib
from datetime import datetime

class AdvancedModelRegistry:
    def __init__(self, tracking_uri: str, experiment_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.experiment_name = experiment_name
    
    def register_model(self, model_path: str, model_name: str, 
                      metadata: Dict[str, Any], tags: Dict[str, str] = None) -> str:
        """Register a new model version"""
        try:
            with mlflow.start_run() as run:
                # Log model
                mlflow.tensorflow.log_model(
                    tf_model_path=model_path,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
                # Log metadata
                mlflow.log_params(metadata.get("params", {}))
                mlflow.log_metrics(metadata.get("metrics", {}))
                
                # Log tags
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                # Log model hash
                model_hash = self._calculate_model_hash(model_path)
                mlflow.set_tag("model_hash", model_hash)
                
                # Log timestamp
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                return run.info.run_id
                
        except Exception as e:
            raise Exception(f"Failed to register model: {e}")
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            result = []
            for version in versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "description": version.description,
                    "tags": version.tags
                }
                result.append(version_info)
            
            return sorted(result, key=lambda x: x["version"], reverse=True)
            
        except Exception as e:
            raise Exception(f"Failed to get model versions: {e}")
    
    def promote_model(self, model_name: str, version: str, stage: str) -> bool:
        """Promote a model to a specific stage"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            return True
            
        except Exception as e:
            raise Exception(f"Failed to promote model: {e}")
    
    def get_model_by_stage(self, model_name: str, stage: str) -> Optional[Dict[str, Any]]:
        """Get the latest model in a specific stage"""
        try:
            versions = self.client.get_latest_versions(
                model_name, 
                stages=[stage]
            )
            
            if not versions:
                return None
            
            latest_version = versions[0]
            return {
                "name": latest_version.name,
                "version": latest_version.version,
                "stage": latest_version.current_stage,
                "run_id": latest_version.run_id,
                "creation_timestamp": latest_version.creation_timestamp,
                "last_updated_timestamp": latest_version.last_updated_timestamp,
                "description": latest_version.description,
                "tags": latest_version.tags
            }
            
        except Exception as e:
            raise Exception(f"Failed to get model by stage: {e}")
    
    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            # Get run information for both versions
            run1 = self.client.get_run(self._get_run_id(model_name, version1))
            run2 = self.client.get_run(self._get_run_id(model_name, version2))
            
            comparison = {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "metrics_comparison": self._compare_metrics(run1.data.metrics, run2.data.metrics),
                "params_comparison": self._compare_params(run1.data.params, run2.data.params),
                "tags_comparison": self._compare_tags(run1.data.tags, run2.data.tags)
            }
            
            return comparison
            
        except Exception as e:
            raise Exception(f"Failed to compare models: {e}")
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model files"""
        hasher = hashlib.sha256()
        
        for root, dirs, files in os.walk(model_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _get_run_id(self, model_name: str, version: str) -> str:
        """Get run ID for a model version"""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.version == version:
                return v.run_id
        raise Exception(f"Version {version} not found for model {model_name}")
    
    def _compare_metrics(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> Dict[str, Any]:
        """Compare metrics between two runs"""
        comparison = {}
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric, None)
            val2 = metrics2.get(metric, None)
            
            if val1 is not None and val2 is not None:
                comparison[metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
            elif val1 is not None:
                comparison[metric] = {
                    "version1": val1,
                    "version2": None,
                    "difference": None,
                    "percent_change": None
                }
            else:
                comparison[metric] = {
                    "version1": None,
                    "version2": val2,
                    "difference": None,
                    "percent_change": None
                }
        
        return comparison
    
    def _compare_params(self, params1: Dict[str, str], params2: Dict[str, str]) -> Dict[str, Any]:
        """Compare parameters between two runs"""
        comparison = {}
        
        all_params = set(params1.keys()) | set(params2.keys())
        
        for param in all_params:
            val1 = params1.get(param, None)
            val2 = params2.get(param, None)
            
            comparison[param] = {
                "version1": val1,
                "version2": val2,
                "changed": val1 != val2
            }
        
        return comparison
    
    def _compare_tags(self, tags1: Dict[str, str], tags2: Dict[str, str]) -> Dict[str, Any]:
        """Compare tags between two runs"""
        comparison = {}
        
        all_tags = set(tags1.keys()) | set(tags2.keys())
        
        for tag in all_tags:
            val1 = tags1.get(tag, None)
            val2 = tags2.get(tag, None)
            
            comparison[tag] = {
                "version1": val1,
                "version2": val2,
                "changed": val1 != val2
            }
        
        return comparison
```

## 🚀 Advanced ML Pipeline Architecture

### Feature Store Implementation
```python
# Advanced Feature Store with Real-time and Batch Features
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import redis
import sqlalchemy
from sqlalchemy import create_engine, text
import json
import asyncio
from datetime import datetime, timedelta
import logging

class AdvancedFeatureStore:
    def __init__(self, redis_client: redis.Redis, 
                 batch_db_engine: sqlalchemy.Engine,
                 realtime_db_engine: sqlalchemy.Engine):
        self.redis = redis_client
        self.batch_db = batch_db_engine
        self.realtime_db = realtime_db_engine
        self.logger = logging.getLogger(__name__)
    
    async def get_feature(self, feature_name: str, entity_id: str, 
                         timestamp: Optional[datetime] = None) -> Optional[Any]:
        """Get a single feature value"""
        try:
            # Try real-time store first
            realtime_value = await self._get_realtime_feature(feature_name, entity_id, timestamp)
            if realtime_value is not None:
                return realtime_value
            
            # Fall back to batch store
            batch_value = await self._get_batch_feature(feature_name, entity_id, timestamp)
            return batch_value
            
        except Exception as e:
            self.logger.error(f"Failed to get feature {feature_name}: {e}")
            return None
    
    async def get_features(self, feature_names: List[str], entity_id: str,
                          timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get multiple feature values"""
        try:
            # Try real-time store first
            realtime_features = await self._get_realtime_features(feature_names, entity_id, timestamp)
            
            # Get missing features from batch store
            missing_features = [name for name in feature_names if name not in realtime_features]
            if missing_features:
                batch_features = await self._get_batch_features(missing_features, entity_id, timestamp)
                realtime_features.update(batch_features)
            
            return realtime_features
            
        except Exception as e:
            self.logger.error(f"Failed to get features: {e}")
            return {}
    
    async def set_feature(self, feature_name: str, entity_id: str, 
                         value: Any, ttl: Optional[int] = None) -> bool:
        """Set a feature value in real-time store"""
        try:
            feature_key = f"feature:{feature_name}:{entity_id}"
            
            feature_data = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "entity_id": entity_id,
                "feature_name": feature_name
            }
            
            if ttl:
                self.redis.setex(feature_key, ttl, json.dumps(feature_data))
            else:
                self.redis.set(feature_key, json.dumps(feature_data))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set feature {feature_name}: {e}")
            return False
    
    async def _get_realtime_feature(self, feature_name: str, entity_id: str,
                                   timestamp: Optional[datetime] = None) -> Optional[Any]:
        """Get feature from real-time store"""
        try:
            feature_key = f"feature:{feature_name}:{entity_id}"
            feature_data = self.redis.get(feature_key)
            
            if feature_data:
                data = json.loads(feature_data)
                
                # Check timestamp if provided
                if timestamp:
                    feature_timestamp = datetime.fromisoformat(data["timestamp"])
                    if feature_timestamp < timestamp:
                        return None
                
                return data["value"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get realtime feature: {e}")
            return None
    
    async def _get_batch_feature(self, feature_name: str, entity_id: str,
                                timestamp: Optional[datetime] = None) -> Optional[Any]:
        """Get feature from batch store"""
        try:
            query = """
                SELECT value, timestamp 
                FROM features 
                WHERE feature_name = :feature_name 
                AND entity_id = :entity_id
            """
            
            params = {
                "feature_name": feature_name,
                "entity_id": entity_id
            }
            
            if timestamp:
                query += " AND timestamp <= :timestamp"
                params["timestamp"] = timestamp
            
            query += " ORDER BY timestamp DESC LIMIT 1"
            
            with self.batch_db.connect() as conn:
                result = conn.execute(text(query), params)
                row = result.fetchone()
                
                if row:
                    return row[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get batch feature: {e}")
            return None
    
    async def _get_realtime_features(self, feature_names: List[str], entity_id: str,
                                    timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get multiple features from real-time store"""
        try:
            pipeline = self.redis.pipeline()
            
            for feature_name in feature_names:
                feature_key = f"feature:{feature_name}:{entity_id}"
                pipeline.get(feature_key)
            
            results = pipeline.execute()
            
            features = {}
            for i, feature_name in enumerate(feature_names):
                if results[i]:
                    data = json.loads(results[i])
                    
                    # Check timestamp if provided
                    if timestamp:
                        feature_timestamp = datetime.fromisoformat(data["timestamp"])
                        if feature_timestamp >= timestamp:
                            features[feature_name] = data["value"]
                    else:
                        features[feature_name] = data["value"]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to get realtime features: {e}")
            return {}
    
    async def _get_batch_features(self, feature_names: List[str], entity_id: str,
                                 timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get multiple features from batch store"""
        try:
            placeholders = ",".join([f":feature_{i}" for i in range(len(feature_names))])
            
            query = f"""
                SELECT feature_name, value, timestamp 
                FROM features 
                WHERE feature_name IN ({placeholders})
                AND entity_id = :entity_id
            """
            
            params = {"entity_id": entity_id}
            for i, feature_name in enumerate(feature_names):
                params[f"feature_{i}"] = feature_name
            
            if timestamp:
                query += " AND timestamp <= :timestamp"
                params["timestamp"] = timestamp
            
            query += " ORDER BY feature_name, timestamp DESC"
            
            with self.batch_db.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                
                features = {}
                for row in rows:
                    if row[0] not in features:  # Take the latest value for each feature
                        features[row[0]] = row[1]
                
                return features
            
        except Exception as e:
            self.logger.error(f"Failed to get batch features: {e}")
            return {}
```

## 🎯 Best Practices

### MLOps Principles
1. **Reproducibility**: Ensure experiments can be reproduced
2. **Versioning**: Track model and data versions
3. **Monitoring**: Monitor model performance in production
4. **Testing**: Implement comprehensive testing strategies
5. **Documentation**: Maintain detailed documentation

### Production Considerations
1. **Scalability**: Design for horizontal scaling
2. **Reliability**: Implement fault tolerance and recovery
3. **Security**: Secure model serving and data access
4. **Performance**: Optimize for latency and throughput
5. **Cost**: Optimize resource usage and costs

### Monitoring and Observability
1. **Metrics**: Track model performance and system health
2. **Logging**: Implement structured logging
3. **Alerting**: Set up proactive alerts
4. **Dashboards**: Create monitoring dashboards
5. **Tracing**: Use distributed tracing for debugging

---

**Last Updated**: December 2024  
**Category**: Advanced ML Systems Comprehensive  
**Complexity**: Expert Level
