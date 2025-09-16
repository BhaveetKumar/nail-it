# 📋 Model Registry: Managing ML Models in Production

> **Complete guide to model registry, versioning, and lifecycle management**

## 🎯 **Learning Objectives**

- Master model registry design and implementation
- Understand model versioning and lifecycle management
- Implement model metadata tracking and lineage
- Build automated model deployment pipelines
- Handle model rollback and A/B testing

## 📚 **Table of Contents**

1. [Model Registry Fundamentals](#model-registry-fundamentals/)
2. [Model Versioning](#model-versioning/)
3. [Model Metadata](#model-metadata/)
4. [Model Lifecycle Management](#model-lifecycle-management/)
5. [Model Deployment Pipeline](#model-deployment-pipeline/)
6. [Interview Questions](#interview-questions/)

---

## 📋 **Model Registry Fundamentals**

### **Concept**

A model registry is a centralized system for managing machine learning models throughout their lifecycle, from development to production deployment. It provides versioning, metadata tracking, and deployment management capabilities.

### **Key Features**

1. **Model Versioning**: Track different versions of models
2. **Metadata Management**: Store model information and performance metrics
3. **Lifecycle Management**: Manage model states (development, staging, production)
4. **Deployment Tracking**: Monitor model deployments and rollbacks
5. **Access Control**: Manage permissions and approvals
6. **Integration**: Connect with CI/CD pipelines and monitoring systems

### **Benefits**

- **Reproducibility**: Track model lineage and dependencies
- **Governance**: Ensure model quality and compliance
- **Collaboration**: Enable team collaboration on models
- **Automation**: Automate model deployment and monitoring
- **Auditability**: Maintain audit trails for compliance

---

## 🔄 **Model Versioning**

### **1. Semantic Versioning for Models**

**Code Example**:
```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import hashlib
import json
import time
from datetime import datetime

class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelVersion:
    major: int
    minor: int
    patch: int
    stage: ModelStage
    created_at: datetime
    model_id: str
    metadata: Dict[str, Any]
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}-{self.stage.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self),
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "model_id": self.model_id,
            "metadata": self.metadata
        }

class ModelVersionManager:
    def __init__(self):
        self.versions: Dict[str, List[ModelVersion]] = {}
    
    def create_version(self, model_id: str, stage: ModelStage, 
                      metadata: Dict[str, Any]) -> ModelVersion:
        """Create a new model version"""
        if model_id not in self.versions:
            self.versions[model_id] = []
        
        # Get latest version
        latest_version = self.get_latest_version(model_id, stage)
        
        if latest_version is None:
            # First version
            major, minor, patch = 1, 0, 0
        else:
            major, minor, patch = latest_version.major, latest_version.minor, latest_version.patch
            
            # Increment version based on stage
            if stage == ModelStage.DEVELOPMENT:
                patch += 1
            elif stage == ModelStage.STAGING:
                minor += 1
                patch = 0
            elif stage == ModelStage.PRODUCTION:
                major += 1
                minor = 0
                patch = 0
        
        version = ModelVersion(
            major=major,
            minor=minor,
            patch=patch,
            stage=stage,
            created_at=datetime.now(),
            model_id=model_id,
            metadata=metadata
        )
        
        self.versions[model_id].append(version)
        return version
    
    def get_latest_version(self, model_id: str, stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """Get latest version for a model"""
        if model_id not in self.versions:
            return None
        
        versions = self.versions[model_id]
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        if not versions:
            return None
        
        # Sort by version number and return latest
        versions.sort(key=lambda v: (v.major, v.minor, v.patch), reverse=True)
        return versions[0]
    
    def promote_version(self, model_id: str, version: str, target_stage: ModelStage) -> ModelVersion:
        """Promote a model version to a new stage"""
        # Find the version
        model_versions = self.versions.get(model_id, [])
        source_version = None
        
        for v in model_versions:
            if str(v) == version:
                source_version = v
                break
        
        if not source_version:
            raise ValueError(f"Version {version} not found for model {model_id}")
        
        # Create new version in target stage
        new_metadata = source_version.metadata.copy()
        new_metadata["promoted_from"] = str(source_version)
        new_metadata["promoted_at"] = datetime.now().isoformat()
        
        return self.create_version(model_id, target_stage, new_metadata)
    
    def get_version_history(self, model_id: str) -> List[ModelVersion]:
        """Get version history for a model"""
        if model_id not in self.versions:
            return []
        
        versions = self.versions[model_id].copy()
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

# Example usage
def model_versioning_example():
    """Example of model versioning"""
    manager = ModelVersionManager()
    
    # Create initial development version
    v1 = manager.create_version(
        "sentiment_classifier",
        ModelStage.DEVELOPMENT,
        {"accuracy": 0.85, "dataset": "train_v1", "algorithm": "bert"}
    )
    print(f"Created version: {v1}")
    
    # Create another development version
    v2 = manager.create_version(
        "sentiment_classifier",
        ModelStage.DEVELOPMENT,
        {"accuracy": 0.87, "dataset": "train_v1", "algorithm": "bert", "optimized": True}
    )
    print(f"Created version: {v2}")
    
    # Promote to staging
    v3 = manager.promote_version("sentiment_classifier", str(v2), ModelStage.STAGING)
    print(f"Promoted to staging: {v3}")
    
    # Promote to production
    v4 = manager.promote_version("sentiment_classifier", str(v3), ModelStage.PRODUCTION)
    print(f"Promoted to production: {v4}")
    
    # Get version history
    history = manager.get_version_history("sentiment_classifier")
    print("Version History:")
    for version in history:
        print(f"  {version} - {version.stage.value}")

if __name__ == "__main__":
    model_versioning_example()
```

### **2. Model Artifact Management**

**Code Example**:
```python
import os
import shutil
import pickle
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib

class ModelArtifactManager:
    def __init__(self, base_path: str = "./model_registry"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def store_model(self, model_id: str, version: str, model: Any, 
                   metadata: Dict[str, Any]) -> str:
        """Store model artifact"""
        model_path = self.base_path / model_id / version
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Calculate checksum
        checksum = self._calculate_checksum(model_file)
        
        # Save checksum
        checksum_file = model_path / "checksum.txt"
        with open(checksum_file, 'w') as f:
            f.write(checksum)
        
        return str(model_path)
    
    def load_model(self, model_id: str, version: str) -> tuple[Any, Dict[str, Any]]:
        """Load model artifact"""
        model_path = self.base_path / model_id / version
        
        # Load model
        model_file = model_path / "model.pkl"
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify checksum
        if not self._verify_checksum(model_file):
            raise ValueError("Model checksum verification failed")
        
        return model, metadata
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _verify_checksum(self, file_path: Path) -> bool:
        """Verify file checksum"""
        checksum_file = file_path.parent / "checksum.txt"
        if not checksum_file.exists():
            return False
        
        with open(checksum_file, 'r') as f:
            stored_checksum = f.read().strip()
        
        calculated_checksum = self._calculate_checksum(file_path)
        return stored_checksum == calculated_checksum
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models"""
        models = []
        for model_dir in self.base_path.iterdir():
            if model_dir.is_dir():
                versions = []
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_file = version_dir / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            versions.append({
                                "version": version_dir.name,
                                "metadata": metadata
                            })
                
                models.append({
                    "model_id": model_dir.name,
                    "versions": versions
                })
        
        return models
    
    def delete_model(self, model_id: str, version: Optional[str] = None):
        """Delete model or version"""
        if version:
            # Delete specific version
            model_path = self.base_path / model_id / version
            if model_path.exists():
                shutil.rmtree(model_path)
        else:
            # Delete entire model
            model_path = self.base_path / model_id
            if model_path.exists():
                shutil.rmtree(model_path)

# Example usage
def model_artifact_example():
    """Example of model artifact management"""
    manager = ModelArtifactManager()
    
    # Mock model
    class MockModel:
        def __init__(self, accuracy):
            self.accuracy = accuracy
        
        def predict(self, data):
            return [0.8, 0.2]  # Mock prediction
    
    # Store model
    model = MockModel(0.85)
    metadata = {
        "accuracy": 0.85,
        "dataset": "train_v1",
        "algorithm": "random_forest",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    path = manager.store_model("sentiment_classifier", "1.0.0", model, metadata)
    print(f"Model stored at: {path}")
    
    # Load model
    loaded_model, loaded_metadata = manager.load_model("sentiment_classifier", "1.0.0")
    print(f"Loaded model accuracy: {loaded_model.accuracy}")
    print(f"Loaded metadata: {loaded_metadata}")
    
    # List models
    models = manager.list_models()
    print("Available models:")
    for model_info in models:
        print(f"  {model_info['model_id']}: {len(model_info['versions'])} versions")

if __name__ == "__main__":
    model_artifact_example()
```

---

## 📊 **Model Metadata**

### **1. Comprehensive Metadata Tracking**

**Code Example**:
```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

@dataclass
class ModelMetadata:
    # Basic information
    model_id: str
    version: str
    name: str
    description: str
    
    # Technical details
    algorithm: str
    framework: str
    language: str
    dependencies: List[str] = field(default_factory=list)
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    
    # Dataset information
    training_dataset: Optional[str] = None
    validation_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    dataset_size: Optional[int] = None
    
    # Training details
    training_start_time: Optional[datetime] = None
    training_end_time: Optional[datetime] = None
    training_duration: Optional[float] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Model characteristics
    model_size: Optional[float] = None  # in MB
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    num_parameters: Optional[int] = None
    
    # Deployment information
    deployment_target: Optional[str] = None
    deployment_time: Optional[datetime] = None
    deployment_status: Optional[str] = None
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Convert datetime strings back to datetime objects
        for key in ['training_start_time', 'training_end_time', 'deployment_time']:
            if key in data and data[key]:
                data[key] = datetime.fromisoformat(data[key])
        
        return cls(**data)

class MetadataManager:
    def __init__(self, storage_path: str = "./metadata_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def store_metadata(self, metadata: ModelMetadata):
        """Store model metadata"""
        model_dir = self.storage_path / metadata.model_id
        model_dir.mkdir(exist_ok=True)
        
        metadata_file = model_dir / f"{metadata.version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def load_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Load model metadata"""
        metadata_file = self.storage_path / model_id / f"{version}.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return ModelMetadata.from_dict(data)
    
    def update_metadata(self, model_id: str, version: str, updates: Dict[str, Any]):
        """Update model metadata"""
        metadata = self.load_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Metadata not found for {model_id}:{version}")
        
        # Update fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
            else:
                metadata.custom_metadata[key] = value
        
        # Store updated metadata
        self.store_metadata(metadata)
    
    def search_models(self, criteria: Dict[str, Any]) -> List[ModelMetadata]:
        """Search models by criteria"""
        results = []
        
        for model_dir in self.storage_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            for metadata_file in model_dir.glob("*.json"):
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                metadata = ModelMetadata.from_dict(data)
                
                # Check criteria
                match = True
                for key, value in criteria.items():
                    if key in metadata.custom_metadata:
                        if metadata.custom_metadata[key] != value:
                            match = False
                            break
                    elif hasattr(metadata, key):
                        if getattr(metadata, key) != value:
                            match = False
                            break
                    else:
                        match = False
                        break
                
                if match:
                    results.append(metadata)
        
        return results

# Example usage
def metadata_example():
    """Example of metadata management"""
    manager = MetadataManager()
    
    # Create metadata
    metadata = ModelMetadata(
        model_id="sentiment_classifier",
        version="1.0.0",
        name="Sentiment Analysis Model",
        description="BERT-based sentiment classification model",
        algorithm="BERT",
        framework="PyTorch",
        language="Python",
        dependencies=["torch", "transformers", "numpy"],
        accuracy=0.85,
        precision=0.83,
        recall=0.87,
        f1_score=0.85,
        training_dataset="sentiment_train_v1",
        validation_dataset="sentiment_val_v1",
        dataset_size=10000,
        training_start_time=datetime(2024, 1, 1, 10, 0, 0),
        training_end_time=datetime(2024, 1, 1, 12, 0, 0),
        training_duration=7200,  # 2 hours
        hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        },
        model_size=500.0,  # 500 MB
        input_shape=[512],  # Max sequence length
        output_shape=[2],   # Binary classification
        num_parameters=110000000,  # 110M parameters
        custom_metadata={
            "team": "ML Team",
            "project": "Sentiment Analysis",
            "business_impact": "high"
        }
    )
    
    # Store metadata
    manager.store_metadata(metadata)
    print("Metadata stored successfully")
    
    # Load metadata
    loaded_metadata = manager.load_metadata("sentiment_classifier", "1.0.0")
    print(f"Loaded metadata: {loaded_metadata.name}")
    
    # Update metadata
    manager.update_metadata("sentiment_classifier", "1.0.0", {
        "deployment_status": "deployed",
        "deployment_time": datetime.now()
    })
    
    # Search models
    results = manager.search_models({"team": "ML Team"})
    print(f"Found {len(results)} models for ML Team")

if __name__ == "__main__":
    metadata_example()
```

---

## 🔄 **Model Lifecycle Management**

### **1. Model Lifecycle States**

**Code Example**:
```python
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

class ModelLifecycleState(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ModelLifecycleManager:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.transitions = {
            ModelLifecycleState.DEVELOPMENT: [ModelLifecycleState.TESTING, ModelLifecycleState.ARCHIVED],
            ModelLifecycleState.TESTING: [ModelLifecycleState.DEVELOPMENT, ModelLifecycleState.STAGING, ModelLifecycleState.ARCHIVED],
            ModelLifecycleState.STAGING: [ModelLifecycleState.TESTING, ModelLifecycleState.PRODUCTION, ModelLifecycleState.ARCHIVED],
            ModelLifecycleState.PRODUCTION: [ModelLifecycleState.STAGING, ModelLifecycleState.DEPRECATED],
            ModelLifecycleState.DEPRECATED: [ModelLifecycleState.ARCHIVED],
            ModelLifecycleState.ARCHIVED: []
        }
    
    def create_model(self, model_id: str, version: str, initial_state: ModelLifecycleState = ModelLifecycleState.DEVELOPMENT) -> Dict[str, Any]:
        """Create a new model in the lifecycle"""
        model_key = f"{model_id}:{version}"
        
        model_info = {
            "model_id": model_id,
            "version": version,
            "state": initial_state,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "history": [{
                "state": initial_state,
                "timestamp": datetime.now(),
                "action": "created"
            }]
        }
        
        self.models[model_key] = model_info
        return model_info
    
    def transition_model(self, model_id: str, version: str, new_state: ModelLifecycleState, 
                        reason: str = "", metadata: Dict[str, Any] = None) -> bool:
        """Transition model to new state"""
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        current_state = self.models[model_key]["state"]
        
        # Check if transition is allowed
        if new_state not in self.transitions[current_state]:
            raise ValueError(f"Transition from {current_state.value} to {new_state.value} not allowed")
        
        # Update model state
        self.models[model_key]["state"] = new_state
        self.models[model_key]["updated_at"] = datetime.now()
        
        # Add to history
        history_entry = {
            "state": new_state,
            "timestamp": datetime.now(),
            "action": "transitioned",
            "reason": reason,
            "metadata": metadata or {}
        }
        self.models[model_key]["history"].append(history_entry)
        
        return True
    
    def get_model_state(self, model_id: str, version: str) -> Optional[ModelLifecycleState]:
        """Get current model state"""
        model_key = f"{model_id}:{version}"
        if model_key in self.models:
            return self.models[model_key]["state"]
        return None
    
    def get_model_history(self, model_id: str, version: str) -> List[Dict[str, Any]]:
        """Get model lifecycle history"""
        model_key = f"{model_id}:{version}"
        if model_key in self.models:
            return self.models[model_key]["history"]
        return []
    
    def get_models_by_state(self, state: ModelLifecycleState) -> List[Dict[str, Any]]:
        """Get all models in a specific state"""
        return [
            model_info for model_info in self.models.values()
            if model_info["state"] == state
        ]
    
    async def auto_transition(self, model_id: str, version: str, 
                            current_state: ModelLifecycleState, 
                            new_state: ModelLifecycleState,
                            condition_func: callable) -> bool:
        """Automatically transition model based on condition"""
        if condition_func():
            return self.transition_model(model_id, version, new_state, "auto_transition")
        return False
    
    def cleanup_archived_models(self, older_than_days: int = 30):
        """Clean up archived models older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        models_to_remove = []
        for model_key, model_info in self.models.items():
            if (model_info["state"] == ModelLifecycleState.ARCHIVED and 
                model_info["updated_at"] < cutoff_date):
                models_to_remove.append(model_key)
        
        for model_key in models_to_remove:
            del self.models[model_key]
        
        return len(models_to_remove)

# Example usage
def lifecycle_example():
    """Example of model lifecycle management"""
    manager = ModelLifecycleManager()
    
    # Create model
    model_info = manager.create_model("sentiment_classifier", "1.0.0")
    print(f"Created model: {model_info['model_id']}:{model_info['version']} in {model_info['state'].value}")
    
    # Transition through lifecycle
    manager.transition_model("sentiment_classifier", "1.0.0", ModelLifecycleState.TESTING, "Ready for testing")
    print(f"Model state: {manager.get_model_state('sentiment_classifier', '1.0.0').value}")
    
    manager.transition_model("sentiment_classifier", "1.0.0", ModelLifecycleState.STAGING, "Testing passed")
    print(f"Model state: {manager.get_model_state('sentiment_classifier', '1.0.0').value}")
    
    manager.transition_model("sentiment_classifier", "1.0.0", ModelLifecycleState.PRODUCTION, "Staging validation successful")
    print(f"Model state: {manager.get_model_state('sentiment_classifier', '1.0.0').value}")
    
    # Get history
    history = manager.get_model_history("sentiment_classifier", "1.0.0")
    print("Model History:")
    for entry in history:
        print(f"  {entry['timestamp']}: {entry['action']} -> {entry['state'].value}")

if __name__ == "__main__":
    lifecycle_example()
```

---

## 🎯 **Interview Questions**

### **1. What is a model registry and why is it important?**

**Answer:**
- **Definition**: Centralized system for managing ML models throughout their lifecycle
- **Importance**: Ensures reproducibility, governance, and collaboration
- **Benefits**: Version control, metadata tracking, deployment management
- **Use Cases**: Model versioning, A/B testing, rollback capabilities
- **Integration**: Connects with CI/CD pipelines and monitoring systems

### **2. How do you handle model versioning in production?**

**Answer:**
- **Semantic Versioning**: Major.Minor.Patch format
- **Stage-based Versioning**: Development, staging, production stages
- **Metadata Tracking**: Performance metrics, dataset information
- **Artifact Management**: Model files, dependencies, checksums
- **Promotion Process**: Automated or manual promotion between stages

### **3. What metadata should be tracked for ML models?**

**Answer:**
- **Technical**: Algorithm, framework, dependencies, hyperparameters
- **Performance**: Accuracy, precision, recall, F1-score, AUC
- **Dataset**: Training/validation/test datasets, dataset size
- **Training**: Start/end time, duration, compute resources
- **Model**: Size, input/output shapes, number of parameters
- **Deployment**: Target environment, deployment time, status

### **4. How do you implement model rollback in production?**

**Answer:**
- **Version Tracking**: Maintain history of deployed versions
- **Health Monitoring**: Monitor model performance and errors
- **Automated Rollback**: Trigger rollback based on metrics
- **Manual Rollback**: Allow manual intervention when needed
- **Data Consistency**: Ensure data compatibility between versions
- **Testing**: Validate rollback process in staging environment

### **5. What are the challenges of model lifecycle management?**

**Answer:**
- **Version Control**: Managing multiple model versions
- **Dependencies**: Handling model and data dependencies
- **Testing**: Comprehensive testing at each stage
- **Deployment**: Coordinating deployments across environments
- **Monitoring**: Tracking model performance and health
- **Compliance**: Meeting regulatory and audit requirements

---

**🎉 Model registry is essential for managing ML models in production environments!**
