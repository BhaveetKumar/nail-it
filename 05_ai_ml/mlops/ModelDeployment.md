---
# Auto-generated front matter
Title: Modeldeployment
LastUpdated: 2025-11-06T20:45:58.318437
Tags: []
Status: draft
---

# 🚀 Model Deployment

> **Master MLOps: from model deployment to production monitoring and CI/CD pipelines**

## 🎯 **Learning Objectives**

- Understand MLOps principles and deployment strategies
- Implement CI/CD pipelines for ML models
- Master model versioning and rollback strategies
- Handle A/B testing and canary deployments
- Build production-ready MLOps infrastructure

## 📚 **Table of Contents**

1. [MLOps Fundamentals](#mlops-fundamentals)
2. [Deployment Strategies](#deployment-strategies)
3. [CI/CD for ML](#cicd-for-ml)
4. [Model Monitoring](#model-monitoring)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🏗️ **MLOps Fundamentals**

### **MLOps Principles**

#### **Concept**
MLOps (Machine Learning Operations) is the practice of applying DevOps principles to machine learning workflows, ensuring reliable, scalable, and maintainable ML systems.

#### **Key Components**
- **Data Pipeline**: Automated data collection, validation, and preprocessing
- **Model Training**: Automated model training and validation
- **Model Deployment**: Automated deployment and rollback
- **Monitoring**: Continuous monitoring of model performance
- **Governance**: Model versioning, lineage, and compliance

#### **Code Example**

```python
import os
import json
import yaml
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import docker
import kubernetes
from kubernetes import client, config
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata for tracking"""
    name: str
    version: str
    created_at: datetime
    model_type: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    validation_metrics: Dict[str, float]
    model_size_mb: float
    deployment_status: str = "pending"
    deployed_at: Optional[datetime] = None
    rollback_version: Optional[str] = None

class ModelRegistry:
    """Model registry for tracking model versions and metadata"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.metadata_file = os.path.join(registry_path, "metadata.json")
        self.models: Dict[str, List[ModelMetadata]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from disk"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                for model_name, versions in data.items():
                    self.models[model_name] = [
                        ModelMetadata(**version) for version in versions
                    ]
        else:
            os.makedirs(self.registry_path, exist_ok=True)
    
    def _save_registry(self):
        """Save model registry to disk"""
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = [asdict(version) for version in versions]
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def register_model(self, metadata: ModelMetadata):
        """Register a new model version"""
        if metadata.name not in self.models:
            self.models[metadata.name] = []
        
        self.models[metadata.name].append(metadata)
        self._save_registry()
        logger.info(f"Registered model {metadata.name} version {metadata.version}")
    
    def get_latest_version(self, model_name: str) -> Optional[ModelMetadata]:
        """Get the latest version of a model"""
        if model_name not in self.models or not self.models[model_name]:
            return None
        
        return max(self.models[model_name], key=lambda x: x.version)
    
    def get_model_versions(self, model_name: str) -> List[ModelMetadata]:
        """Get all versions of a model"""
        return self.models.get(model_name, [])
    
    def update_deployment_status(self, model_name: str, version: str, status: str):
        """Update deployment status of a model"""
        for model in self.models.get(model_name, []):
            if model.version == version:
                model.deployment_status = status
                if status == "deployed":
                    model.deployed_at = datetime.now()
                self._save_registry()
                break

class DataValidator:
    """Data validation for ML pipelines"""
    
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load data schema"""
        with open(self.schema_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data against schema"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check required columns
        required_columns = self.schema.get("required_columns", [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        for column, expected_type in self.schema.get("column_types", {}).items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if expected_type not in actual_type:
                    validation_results["warnings"].append(
                        f"Column {column} has type {actual_type}, expected {expected_type}"
                    )
        
        # Check data quality
        for column in data.columns:
            null_count = data[column].isnull().sum()
            if null_count > 0:
                validation_results["warnings"].append(
                    f"Column {column} has {null_count} null values"
                )
        
        # Calculate statistics
        validation_results["statistics"] = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return validation_results
    
    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current data"""
        drift_results = {
            "has_drift": False,
            "drift_score": 0.0,
            "column_drifts": {}
        }
        
        for column in reference_data.columns:
            if column in current_data.columns:
                # Calculate statistical drift
                ref_mean = reference_data[column].mean()
                ref_std = reference_data[column].std()
                curr_mean = current_data[column].mean()
                curr_std = current_data[column].std()
                
                # Simple drift detection based on mean and std changes
                mean_drift = abs(ref_mean - curr_mean) / (ref_std + 1e-8)
                std_drift = abs(ref_std - curr_std) / (ref_std + 1e-8)
                
                drift_score = (mean_drift + std_drift) / 2
                drift_results["column_drifts"][column] = drift_score
                
                if drift_score > 0.5:  # Threshold for drift
                    drift_results["has_drift"] = True
                    drift_results["drift_score"] = max(drift_results["drift_score"], drift_score)
        
        return drift_results

class ModelEvaluator:
    """Model evaluation and validation"""
    
    def __init__(self, metrics_config: Dict[str, Any]):
        self.metrics_config = metrics_config
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        metrics = {}
        
        # Classification metrics
        if self.metrics_config.get("classification", False):
            metrics.update({
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average='weighted'),
                "recall": recall_score(y_test, predictions, average='weighted'),
                "f1_score": f1_score(y_test, predictions, average='weighted')
            })
        
        # Regression metrics
        if self.metrics_config.get("regression", False):
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics.update({
                "mse": mean_squared_error(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
                "r2_score": r2_score(y_test, predictions)
            })
        
        return metrics
    
    def validate_model(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
        """Validate model against performance thresholds"""
        for metric, threshold in thresholds.items():
            if metric in metrics:
                if metrics[metric] < threshold:
                    logger.warning(f"Model failed validation: {metric} = {metrics[metric]:.4f} < {threshold}")
                    return False
        
        return True

class DeploymentManager:
    """Manage model deployments"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.deployed_models: Dict[str, str] = {}  # model_name -> version
    
    def deploy_model(self, model_name: str, version: str, deployment_config: Dict[str, Any]) -> bool:
        """Deploy a model version"""
        try:
            # Get model metadata
            model_metadata = None
            for model in self.registry.get_model_versions(model_name):
                if model.version == version:
                    model_metadata = model
                    break
            
            if not model_metadata:
                logger.error(f"Model {model_name} version {version} not found")
                return False
            
            # Deploy based on strategy
            strategy = deployment_config.get("strategy", "rolling")
            
            if strategy == "rolling":
                success = self._rolling_deployment(model_name, version, deployment_config)
            elif strategy == "blue_green":
                success = self._blue_green_deployment(model_name, version, deployment_config)
            elif strategy == "canary":
                success = self._canary_deployment(model_name, version, deployment_config)
            else:
                logger.error(f"Unknown deployment strategy: {strategy}")
                return False
            
            if success:
                self.deployed_models[model_name] = version
                self.registry.update_deployment_status(model_name, version, "deployed")
                logger.info(f"Successfully deployed {model_name} version {version}")
            
            return success
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _rolling_deployment(self, model_name: str, version: str, config: Dict[str, Any]) -> bool:
        """Rolling deployment strategy"""
        logger.info(f"Starting rolling deployment for {model_name} version {version}")
        
        # In production, this would:
        # 1. Create new model instances
        # 2. Gradually shift traffic to new instances
        # 3. Remove old instances
        # 4. Verify deployment success
        
        # Simulate deployment
        time.sleep(2)  # Simulate deployment time
        return True
    
    def _blue_green_deployment(self, model_name: str, version: str, config: Dict[str, Any]) -> bool:
        """Blue-green deployment strategy"""
        logger.info(f"Starting blue-green deployment for {model_name} version {version}")
        
        # In production, this would:
        # 1. Deploy new version to green environment
        # 2. Run health checks on green environment
        # 3. Switch traffic from blue to green
        # 4. Keep blue environment for rollback
        
        # Simulate deployment
        time.sleep(3)  # Simulate deployment time
        return True
    
    def _canary_deployment(self, model_name: str, version: str, config: Dict[str, Any]) -> bool:
        """Canary deployment strategy"""
        logger.info(f"Starting canary deployment for {model_name} version {version}")
        
        # In production, this would:
        # 1. Deploy new version to subset of instances
        # 2. Route small percentage of traffic to new version
        # 3. Monitor performance and gradually increase traffic
        # 4. Complete deployment or rollback based on results
        
        # Simulate deployment
        time.sleep(4)  # Simulate deployment time
        return True
    
    def rollback_model(self, model_name: str, target_version: str) -> bool:
        """Rollback model to previous version"""
        try:
            current_version = self.deployed_models.get(model_name)
            if not current_version:
                logger.error(f"No deployed version found for {model_name}")
                return False
            
            logger.info(f"Rolling back {model_name} from {current_version} to {target_version}")
            
            # In production, this would:
            # 1. Stop traffic to current version
            # 2. Deploy target version
            # 3. Route traffic to target version
            # 4. Verify rollback success
            
            # Simulate rollback
            time.sleep(2)
            
            self.deployed_models[model_name] = target_version
            self.registry.update_deployment_status(model_name, target_version, "deployed")
            
            logger.info(f"Successfully rolled back {model_name} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

class MLOpsPipeline:
    """Complete MLOps pipeline"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.registry = ModelRegistry(self.config["registry_path"])
        self.data_validator = DataValidator(self.config["data_schema_path"])
        self.model_evaluator = ModelEvaluator(self.config["evaluation_metrics"])
        self.deployment_manager = DeploymentManager(self.registry)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MLOps configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_pipeline(self, model_name: str, model, training_data: pd.DataFrame, 
                    test_data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> bool:
        """Run complete MLOps pipeline"""
        try:
            # 1. Data validation
            logger.info("Step 1: Data validation")
            validation_results = self.data_validator.validate_data(training_data)
            if not validation_results["is_valid"]:
                logger.error(f"Data validation failed: {validation_results['errors']}")
                return False
            
            # 2. Model training
            logger.info("Step 2: Model training")
            model.fit(training_data.drop('target', axis=1), training_data['target'])
            
            # 3. Model evaluation
            logger.info("Step 3: Model evaluation")
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']
            metrics = self.model_evaluator.evaluate_model(model, X_test, y_test)
            
            # 4. Model validation
            logger.info("Step 4: Model validation")
            if not self.model_evaluator.validate_model(metrics, self.config["performance_thresholds"]):
                logger.error("Model validation failed")
                return False
            
            # 5. Model registration
            logger.info("Step 5: Model registration")
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            training_data_hash = hashlib.md5(training_data.to_string().encode()).hexdigest()
            
            metadata = ModelMetadata(
                name=model_name,
                version=model_version,
                created_at=datetime.now(),
                model_type=type(model).__name__,
                algorithm=type(model).__name__,
                hyperparameters=hyperparameters,
                training_data_hash=training_data_hash,
                validation_metrics=metrics,
                model_size_mb=0.1  # Would calculate actual size
            )
            
            self.registry.register_model(metadata)
            
            # 6. Model deployment
            logger.info("Step 6: Model deployment")
            deployment_config = self.config["deployment"]
            if self.deployment_manager.deploy_model(model_name, model_version, deployment_config):
                logger.info(f"MLOps pipeline completed successfully for {model_name}")
                return True
            else:
                logger.error("Model deployment failed")
                return False
                
        except Exception as e:
            logger.error(f"MLOps pipeline failed: {e}")
            return False

# Example usage
def main():
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    
    # Split data
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create MLOps configuration
    config = {
        "registry_path": "./model_registry",
        "data_schema_path": "./data_schema.yaml",
        "evaluation_metrics": {
            "classification": True,
            "regression": False
        },
        "performance_thresholds": {
            "accuracy": 0.8,
            "f1_score": 0.75
        },
        "deployment": {
            "strategy": "rolling",
            "replicas": 3,
            "resources": {
                "cpu": "500m",
                "memory": "1Gi"
            }
        }
    }
    
    # Save configuration
    with open("mlops_config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Create data schema
    schema = {
        "required_columns": [f'feature_{i}' for i in range(20)] + ['target'],
        "column_types": {
            "target": "int64"
        }
    }
    
    with open("data_schema.yaml", 'w') as f:
        yaml.dump(schema, f)
    
    # Initialize MLOps pipeline
    pipeline = MLOpsPipeline("mlops_config.yaml")
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    hyperparameters = {"n_estimators": 100, "random_state": 42}
    
    # Run pipeline
    success = pipeline.run_pipeline("sample_model", model, train_data, test_data, hyperparameters)
    
    if success:
        print("MLOps pipeline completed successfully!")
    else:
        print("MLOps pipeline failed!")

if __name__ == "__main__":
    main()
```

---

## 🚀 **Deployment Strategies**

### **Advanced Deployment Patterns**

#### **Concept**
Different deployment strategies for different use cases and risk tolerance levels.

#### **Code Example**

```python
import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import kubernetes
from kubernetes import client, config
import docker
import subprocess
import json

class DeploymentStrategy(Enum):
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    strategy: DeploymentStrategy
    replicas: int
    resources: Dict[str, str]
    health_check_path: str
    readiness_probe: Dict[str, Any]
    liveness_probe: Dict[str, Any]
    traffic_split: Optional[Dict[str, float]] = None
    rollback_threshold: Optional[float] = None

class KubernetesDeployer:
    """Kubernetes-based model deployment"""
    
    def __init__(self, namespace: str = "ml-models"):
        self.namespace = namespace
        self.k8s_client = client.ApiClient()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.networking_v1 = client.NetworkingV1Api()
    
    async def deploy_model(self, model_name: str, version: str, config: DeploymentConfig) -> bool:
        """Deploy model using specified strategy"""
        try:
            if config.strategy == DeploymentStrategy.ROLLING:
                return await self._rolling_deployment(model_name, version, config)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._blue_green_deployment(model_name, version, config)
            elif config.strategy == DeploymentStrategy.CANARY:
                return await self._canary_deployment(model_name, version, config)
            elif config.strategy == DeploymentStrategy.A_B_TESTING:
                return await self._ab_testing_deployment(model_name, version, config)
            else:
                raise ValueError(f"Unknown deployment strategy: {config.strategy}")
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    async def _rolling_deployment(self, model_name: str, version: str, config: DeploymentConfig) -> bool:
        """Rolling deployment - gradually replace instances"""
        logger.info(f"Starting rolling deployment for {model_name}:{version}")
        
        # Create new deployment
        deployment = self._create_deployment_manifest(model_name, version, config)
        
        # Apply deployment
        try:
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            logger.info(f"Created deployment for {model_name}:{version}")
        except client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                # Update existing deployment
                self.apps_v1.patch_namespaced_deployment(
                    name=f"{model_name}-{version}",
                    namespace=self.namespace,
                    body=deployment
                )
                logger.info(f"Updated deployment for {model_name}:{version}")
            else:
                raise
        
        # Wait for rollout to complete
        return await self._wait_for_rollout(model_name, version)
    
    async def _blue_green_deployment(self, model_name: str, version: str, config: DeploymentConfig) -> bool:
        """Blue-green deployment - switch traffic between environments"""
        logger.info(f"Starting blue-green deployment for {model_name}:{version}")
        
        # Deploy to green environment
        green_deployment = self._create_deployment_manifest(
            f"{model_name}-green", version, config
        )
        
        try:
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=green_deployment
            )
        except client.rest.ApiException as e:
            if e.status == 409:
                self.apps_v1.patch_namespaced_deployment(
                    name=f"{model_name}-green-{version}",
                    namespace=self.namespace,
                    body=green_deployment
                )
            else:
                raise
        
        # Wait for green deployment to be ready
        if not await self._wait_for_rollout(f"{model_name}-green", version):
            return False
        
        # Run health checks on green environment
        if not await self._run_health_checks(f"{model_name}-green", config):
            logger.error("Health checks failed on green environment")
            return False
        
        # Switch traffic to green
        await self._switch_traffic(model_name, "green")
        
        # Keep blue environment for rollback
        logger.info("Blue-green deployment completed successfully")
        return True
    
    async def _canary_deployment(self, model_name: str, version: str, config: DeploymentConfig) -> bool:
        """Canary deployment - gradual traffic increase"""
        logger.info(f"Starting canary deployment for {model_name}:{version}")
        
        # Deploy canary version
        canary_deployment = self._create_deployment_manifest(
            f"{model_name}-canary", version, config
        )
        
        try:
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=canary_deployment
            )
        except client.rest.ApiException as e:
            if e.status == 409:
                self.apps_v1.patch_namespaced_deployment(
                    name=f"{model_name}-canary-{version}",
                    namespace=self.namespace,
                    body=canary_deployment
                )
            else:
                raise
        
        # Wait for canary to be ready
        if not await self._wait_for_rollout(f"{model_name}-canary", version):
            return False
        
        # Gradually increase traffic
        traffic_splits = [0.1, 0.25, 0.5, 0.75, 1.0]  # 10%, 25%, 50%, 75%, 100%
        
        for split in traffic_splits:
            logger.info(f"Setting traffic split to {split*100}% for canary")
            
            # Update traffic split
            await self._update_traffic_split(model_name, "canary", split)
            
            # Monitor for issues
            await asyncio.sleep(60)  # Monitor for 1 minute
            
            # Check if rollback is needed
            if await self._should_rollback(model_name, config):
                logger.warning("Rollback triggered during canary deployment")
                await self._rollback_canary(model_name)
                return False
        
        # Promote canary to stable
        await self._promote_canary(model_name, version)
        logger.info("Canary deployment completed successfully")
        return True
    
    async def _ab_testing_deployment(self, model_name: str, version: str, config: DeploymentConfig) -> bool:
        """A/B testing deployment - split traffic between versions"""
        logger.info(f"Starting A/B testing deployment for {model_name}:{version}")
        
        # Deploy new version
        new_deployment = self._create_deployment_manifest(
            f"{model_name}-b", version, config
        )
        
        try:
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=new_deployment
            )
        except client.rest.ApiException as e:
            if e.status == 409:
                self.apps_v1.patch_namespaced_deployment(
                    name=f"{model_name}-b-{version}",
                    namespace=self.namespace,
                    body=new_deployment
                )
            else:
                raise
        
        # Wait for new version to be ready
        if not await self._wait_for_rollout(f"{model_name}-b", version):
            return False
        
        # Set up A/B testing traffic split
        traffic_split = config.traffic_split or {"a": 0.5, "b": 0.5}
        await self._setup_ab_testing(model_name, traffic_split)
        
        logger.info("A/B testing deployment completed successfully")
        return True
    
    def _create_deployment_manifest(self, name: str, version: str, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{name}-{version}",
                "namespace": self.namespace,
                "labels": {
                    "app": name,
                    "version": version
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": name,
                        "version": version
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": name,
                            "version": version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": name,
                            "image": f"ml-model:{version}",
                            "ports": [{"containerPort": 8080}],
                            "resources": {
                                "requests": config.resources,
                                "limits": config.resources
                            },
                            "readinessProbe": config.readiness_probe,
                            "livenessProbe": config.liveness_probe
                        }]
                    }
                }
            }
        }
    
    async def _wait_for_rollout(self, name: str, version: str, timeout: int = 300) -> bool:
        """Wait for deployment rollout to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=f"{name}-{version}",
                    namespace=self.namespace
                )
                
                if (deployment.status.ready_replicas == deployment.spec.replicas and
                    deployment.status.updated_replicas == deployment.spec.replicas):
                    logger.info(f"Deployment {name}:{version} is ready")
                    return True
                
                await asyncio.sleep(5)
                
            except client.rest.ApiException as e:
                if e.status == 404:
                    logger.warning(f"Deployment {name}:{version} not found")
                    return False
                raise
        
        logger.error(f"Deployment {name}:{version} rollout timeout")
        return False
    
    async def _run_health_checks(self, name: str, config: DeploymentConfig) -> bool:
        """Run health checks on deployment"""
        try:
            # Get service endpoint
            service = self.core_v1.read_namespaced_service(
                name=name,
                namespace=self.namespace
            )
            
            # Run health check
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{service.spec.cluster_ip}:8080{config.health_check_path}",
                    timeout=10
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _switch_traffic(self, model_name: str, environment: str):
        """Switch traffic to specified environment"""
        # Update service selector
        service = self.core_v1.read_namespaced_service(
            name=model_name,
            namespace=self.namespace
        )
        
        service.spec.selector["environment"] = environment
        
        self.core_v1.patch_namespaced_service(
            name=model_name,
            namespace=self.namespace,
            body=service
        )
        
        logger.info(f"Switched traffic to {environment} environment")
    
    async def _update_traffic_split(self, model_name: str, version: str, split: float):
        """Update traffic split for canary deployment"""
        # This would typically use Istio or similar service mesh
        # For demo purposes, we'll simulate it
        logger.info(f"Updated traffic split for {model_name}:{version} to {split*100}%")
    
    async def _should_rollback(self, model_name: str, config: DeploymentConfig) -> bool:
        """Check if rollback is needed based on metrics"""
        # In production, this would check:
        # - Error rates
        # - Response times
        # - Business metrics
        
        # Simulate rollback check
        error_rate = 0.05  # 5% error rate
        threshold = config.rollback_threshold or 0.1
        
        return error_rate > threshold
    
    async def _rollback_canary(self, model_name: str):
        """Rollback canary deployment"""
        # Remove canary deployment
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=f"{model_name}-canary",
                namespace=self.namespace
            )
            logger.info(f"Rolled back canary deployment for {model_name}")
        except client.rest.ApiException as e:
            if e.status != 404:
                raise
    
    async def _promote_canary(self, model_name: str, version: str):
        """Promote canary to stable version"""
        # Update stable deployment to canary version
        # Remove canary deployment
        # Update service to point to stable version
        
        logger.info(f"Promoted canary {model_name}:{version} to stable")
    
    async def _setup_ab_testing(self, model_name: str, traffic_split: Dict[str, float]):
        """Set up A/B testing traffic split"""
        # This would typically use Istio or similar service mesh
        logger.info(f"Set up A/B testing for {model_name} with traffic split: {traffic_split}")

# Example usage
async def main():
    # Initialize Kubernetes deployer
    deployer = KubernetesDeployer()
    
    # Create deployment configuration
    config = DeploymentConfig(
        strategy=DeploymentStrategy.CANARY,
        replicas=3,
        resources={"cpu": "500m", "memory": "1Gi"},
        health_check_path="/health",
        readiness_probe={
            "httpGet": {"path": "/ready", "port": 8080},
            "initialDelaySeconds": 10,
            "periodSeconds": 5
        },
        liveness_probe={
            "httpGet": {"path": "/health", "port": 8080},
            "initialDelaySeconds": 30,
            "periodSeconds": 10
        },
        rollback_threshold=0.1
    )
    
    # Deploy model
    success = await deployer.deploy_model("sample-model", "v1.0.0", config)
    
    if success:
        print("Model deployed successfully!")
    else:
        print("Model deployment failed!")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 **Interview Questions**

### **MLOps Theory**

#### **Q1: What is MLOps and how does it differ from DevOps?**
**Answer**: 
- **MLOps**: Machine Learning Operations, applying DevOps principles to ML workflows
- **Key Differences**: 
  - Data versioning and lineage tracking
  - Model versioning and experiment tracking
  - Continuous training and retraining
  - Model monitoring and drift detection
  - A/B testing and model comparison
- **Benefits**: Faster deployment, better reproducibility, improved model performance

#### **Q2: What are the key components of an MLOps pipeline?**
**Answer**: 
- **Data Pipeline**: Automated data collection, validation, and preprocessing
- **Model Training**: Automated model training and validation
- **Model Registry**: Centralized model storage and versioning
- **Model Deployment**: Automated deployment with rollback capabilities
- **Monitoring**: Continuous monitoring of model performance and data drift
- **Governance**: Model approval, compliance, and audit trails

#### **Q3: How do you handle model versioning in production?**
**Answer**: 
- **Semantic Versioning**: Use MAJOR.MINOR.PATCH format
- **Model Registry**: Track model metadata, performance, and lineage
- **Experiment Tracking**: Use tools like MLflow, Weights & Biases
- **Data Versioning**: Track training data versions and hashes
- **Model Artifacts**: Store model files, configurations, and dependencies
- **Rollback Strategy**: Quick rollback to previous versions if issues occur

#### **Q4: What are the different deployment strategies for ML models?**
**Answer**: 
- **Rolling Deployment**: Gradually replace instances one at a time
- **Blue-Green Deployment**: Switch between two identical environments
- **Canary Deployment**: Gradually increase traffic to new version
- **A/B Testing**: Split traffic between different model versions
- **Shadow Deployment**: Run new model alongside old one without affecting traffic

#### **Q5: How do you monitor model performance in production?**
**Answer**: 
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Business Metrics**: Conversion rates, revenue impact, user satisfaction
- **Data Drift**: Monitor input data distribution changes
- **Model Drift**: Monitor model performance degradation over time
- **Infrastructure Metrics**: Latency, throughput, error rates, resource usage
- **Alerting**: Set up alerts for performance degradation or anomalies

### **Implementation Questions**

#### **Q6: Implement a complete MLOps pipeline**
**Answer**: See the implementation above with model registry, data validation, evaluation, and deployment.

#### **Q7: How would you handle data drift in production?**
**Answer**: 
- **Detection**: Statistical tests, distribution comparisons, drift scores
- **Monitoring**: Continuous monitoring of input data distributions
- **Alerting**: Set up alerts when drift exceeds thresholds
- **Response**: Retrain model, update preprocessing, or rollback
- **Prevention**: Regular retraining, data quality checks, feature engineering

#### **Q8: How do you ensure model reproducibility?**
**Answer**: 
- **Environment**: Use containers, virtual environments, dependency management
- **Data**: Version control data, track data lineage, use data catalogs
- **Code**: Version control, code reviews, automated testing
- **Experiments**: Track hyperparameters, random seeds, model configurations
- **Infrastructure**: Use Infrastructure as Code, consistent environments
- **Documentation**: Document assumptions, decisions, and processes

---

## 🚀 **Next Steps**

1. **Practice**: Implement all deployment strategies and test with different models
2. **Optimize**: Focus on automation and monitoring
3. **Deploy**: Build production MLOps infrastructure
4. **Extend**: Learn about advanced monitoring and governance
5. **Interview**: Practice MLOps interview questions

---

**Ready to learn about Case Studies? Let's move to the final section!** 🎯


## Cicd For Ml

<!-- AUTO-GENERATED ANCHOR: originally referenced as #cicd-for-ml -->

Placeholder content. Please replace with proper section.


## Model Monitoring

<!-- AUTO-GENERATED ANCHOR: originally referenced as #model-monitoring -->

Placeholder content. Please replace with proper section.


## Production Implementation

<!-- AUTO-GENERATED ANCHOR: originally referenced as #production-implementation -->

Placeholder content. Please replace with proper section.
