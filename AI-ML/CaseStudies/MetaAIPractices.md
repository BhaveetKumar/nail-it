# 🏢 Meta AI Practices

> **Learn from Meta's AI engineering practices: from research to production deployment**

## 🎯 **Learning Objectives**

- Understand Meta's AI research and engineering practices
- Learn about Meta's ML infrastructure and tooling
- Master Meta's approach to large-scale AI systems
- Understand Meta's AI ethics and responsible AI practices
- Apply Meta's best practices to your own AI projects

## 📚 **Table of Contents**

1. [Meta AI Overview](#meta-ai-overview)
2. [Research Practices](#research-practices)
3. [Engineering Practices](#engineering-practices)
4. [Infrastructure and Tooling](#infrastructure-and-tooling)
5. [Production Systems](#production-systems)
6. [Interview Questions](#interview-questions)

---

## 🏢 **Meta AI Overview**

### **Meta AI Organization**

#### **Concept**
Meta AI is one of the world's leading AI research organizations, focusing on advancing AI capabilities while ensuring responsible development.

#### **Key Areas**
- **Computer Vision**: Image recognition, object detection, video understanding
- **Natural Language Processing**: Language models, translation, understanding
- **Speech and Audio**: Speech recognition, synthesis, audio processing
- **Robotics**: Embodied AI, manipulation, navigation
- **AI Infrastructure**: Distributed training, model serving, optimization

#### **Code Example**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import logging
import time
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetaAIConfig:
    """Meta AI configuration"""
    model_name: str
    model_size: str  # small, medium, large, xlarge
    training_data_size: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    weight_decay: float = 0.01

class MetaAIDataset(Dataset):
    """Meta AI dataset with advanced preprocessing"""
    
    def __init__(self, data_path: str, config: MetaAIConfig):
        self.data_path = data_path
        self.config = config
        self.data = self._load_data()
        self.preprocessing_pipeline = self._create_preprocessing_pipeline()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and validate data"""
        # In production, this would load from Meta's data infrastructure
        # For demo, create synthetic data
        data = []
        for i in range(self.config.training_data_size):
            data.append({
                "input": np.random.randn(512),  # Input features
                "target": np.random.randint(0, 10),  # Target class
                "metadata": {
                    "user_id": f"user_{i}",
                    "timestamp": time.time(),
                    "source": "synthetic"
                }
            })
        return data
    
    def _create_preprocessing_pipeline(self):
        """Create preprocessing pipeline"""
        # Meta uses sophisticated preprocessing pipelines
        # This is a simplified version
        return {
            "normalization": True,
            "augmentation": True,
            "tokenization": True,
            "padding": True
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Apply preprocessing
        input_data = self._preprocess_input(item["input"])
        target = item["target"]
        
        return {
            "input": torch.tensor(input_data, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
            "metadata": item["metadata"]
        }
    
    def _preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess input data"""
        # Normalization
        if self.preprocessing_pipeline["normalization"]:
            input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-8)
        
        # Data augmentation (simplified)
        if self.preprocessing_pipeline["augmentation"]:
            # Add noise for augmentation
            noise = np.random.normal(0, 0.1, input_data.shape)
            input_data = input_data + noise
        
        return input_data

class MetaAIModel(nn.Module):
    """Meta AI model architecture"""
    
    def __init__(self, config: MetaAIConfig):
        super(MetaAIModel, self).__init__()
        self.config = config
        self.model_size = config.model_size
        
        # Model architecture based on size
        if self.model_size == "small":
            self._build_small_model()
        elif self.model_size == "medium":
            self._build_medium_model()
        elif self.model_size == "large":
            self._build_large_model()
        elif self.model_size == "xlarge":
            self._build_xlarge_model()
        else:
            raise ValueError(f"Unknown model size: {self.model_size}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_small_model(self):
        """Build small model architecture"""
        self.input_layer = nn.Linear(512, 256)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 128)
        ])
        self.output_layer = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(256)
    
    def _build_medium_model(self):
        """Build medium model architecture"""
        self.input_layer = nn.Linear(512, 512)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 256)
        ])
        self.output_layer = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(512)
    
    def _build_large_model(self):
        """Build large model architecture"""
        self.input_layer = nn.Linear(512, 1024)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 256)
        ])
        self.output_layer = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(1024)
    
    def _build_xlarge_model(self):
        """Build xlarge model architecture"""
        self.input_layer = nn.Linear(512, 2048)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(2048, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 512)
        ])
        self.output_layer = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(2048)
    
    def _initialize_weights(self):
        """Initialize model weights using Meta's practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.input_layer(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection if dimensions match
            if x.shape == residual.shape:
                x = x + residual
        
        x = self.output_layer(x)
        return x

class MetaAITrainer:
    """Meta AI training practices"""
    
    def __init__(self, model: MetaAIModel, config: MetaAIConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Initialize optimizer with Meta's practices
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
            eta_min=config.learning_rate * 0.01
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training metrics
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "learning_rate": []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Update training history
        self.training_history["loss"].append(epoch_loss)
        self.training_history["accuracy"].append(epoch_accuracy)
        self.training_history["learning_rate"].append(current_lr)
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "learning_rate": current_lr
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": 100 * correct / total
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Train the model"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            logger.info(f"Learning Rate: {train_metrics['learning_rate']:.6f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
                logger.info(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

class MetaAIInfrastructure:
    """Meta AI infrastructure practices"""
    
    def __init__(self):
        self.distributed_training = True
        self.model_serving = True
        self.monitoring = True
        self.experiment_tracking = True
    
    def setup_distributed_training(self, world_size: int, rank: int):
        """Setup distributed training"""
        if self.distributed_training:
            torch.distributed.init_process_group(
                backend='nccl',
                world_size=world_size,
                rank=rank
            )
            logger.info(f"Distributed training initialized: rank {rank}/{world_size}")
    
    def setup_model_serving(self, model_path: str):
        """Setup model serving infrastructure"""
        if self.model_serving:
            # In production, this would integrate with Meta's serving infrastructure
            logger.info(f"Model serving setup for {model_path}")
    
    def setup_monitoring(self, model_name: str):
        """Setup monitoring and logging"""
        if self.monitoring:
            # In production, this would integrate with Meta's monitoring systems
            logger.info(f"Monitoring setup for {model_name}")
    
    def setup_experiment_tracking(self, experiment_name: str):
        """Setup experiment tracking"""
        if self.experiment_tracking:
            # In production, this would integrate with Meta's experiment tracking
            logger.info(f"Experiment tracking setup for {experiment_name}")

# Example usage
def main():
    # Create Meta AI configuration
    config = MetaAIConfig(
        model_name="meta_ai_demo",
        model_size="medium",
        training_data_size=10000,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=True,
        gradient_accumulation_steps=2,
        warmup_steps=1000,
        weight_decay=0.01
    )
    
    # Create dataset
    train_dataset = MetaAIDataset("train_data", config)
    val_dataset = MetaAIDataset("val_data", config)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = MetaAIModel(config)
    
    # Create trainer
    trainer = MetaAITrainer(model, config)
    
    # Setup infrastructure
    infrastructure = MetaAIInfrastructure()
    infrastructure.setup_experiment_tracking("meta_ai_demo_experiment")
    infrastructure.setup_monitoring("meta_ai_demo")
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Setup model serving
    infrastructure.setup_model_serving("best_model_epoch_10.pth")

if __name__ == "__main__":
    main()
```

---

## 🔬 **Research Practices**

### **Meta AI Research Methodology**

#### **Concept**
Meta AI follows rigorous research practices with emphasis on reproducibility, collaboration, and open science.

#### **Key Practices**
- **Open Research**: Publishing papers and open-sourcing code
- **Collaboration**: Cross-team collaboration and external partnerships
- **Reproducibility**: Detailed documentation and code sharing
- **Ethics**: Responsible AI development and deployment
- **Diversity**: Inclusive research teams and diverse perspectives

#### **Code Example**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import logging
import time
from dataclasses import dataclass
import json
import wandb  # Weights & Biases for experiment tracking

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    """Research configuration"""
    experiment_name: str
    model_architecture: str
    dataset_name: str
    hyperparameters: Dict[str, Any]
    reproducibility_seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    weight_decay: float = 0.01

class MetaAIResearch:
    """Meta AI research practices"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set reproducibility seed
        self._set_seed(config.reproducibility_seed)
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training metrics
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": []
        }
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed}")
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking"""
        # Initialize Weights & Biases
        wandb.init(
            project="meta_ai_research",
            name=self.config.experiment_name,
            config=self.config.hyperparameters
        )
        
        # Log model architecture
        wandb.watch(self.model, log="all", log_freq=100)
        
        logger.info(f"Experiment tracking initialized for {self.config.experiment_name}")
    
    def _create_model(self) -> nn.Module:
        """Create model based on architecture"""
        if self.config.model_architecture == "transformer":
            return self._create_transformer_model()
        elif self.config.model_architecture == "cnn":
            return self._create_cnn_model()
        elif self.config.model_architecture == "mlp":
            return self._create_mlp_model()
        else:
            raise ValueError(f"Unknown model architecture: {self.config.model_architecture}")
    
    def _create_transformer_model(self) -> nn.Module:
        """Create transformer model"""
        # Simplified transformer for demo
        class TransformerModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embedding = nn.Embedding(10000, 512)
                self.pos_encoding = nn.Parameter(torch.randn(1000, 512))
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.1),
                    num_layers=6
                )
                self.classifier = nn.Linear(512, 10)
            
            def forward(self, x):
                x = self.embedding(x) + self.pos_encoding[:x.size(1)]
                x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
                x = self.transformer(x)
                x = x.mean(dim=0)  # Global average pooling
                return self.classifier(x)
        
        return TransformerModel(self.config)
    
    def _create_cnn_model(self) -> nn.Module:
        """Create CNN model"""
        class CNNModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(256 * 4 * 4, 512)
                self.fc2 = nn.Linear(512, 10)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.conv3(x)))
                x = x.view(-1, 256 * 4 * 4)
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        return CNNModel(self.config)
    
    def _create_mlp_model(self) -> nn.Module:
        """Create MLP model"""
        class MLPModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.fc1 = nn.Linear(784, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, 10)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = x.view(-1, 784)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = F.relu(self.fc3(x))
                x = self.dropout(x)
                x = self.fc4(x)
                return x
        
        return MLPModel(self.config)
    
    def _create_optimizer(self):
        """Create optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.hyperparameters.get("learning_rate", 0.001),
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.warmup_steps,
            T_mult=2,
            eta_min=self.config.hyperparameters.get("learning_rate", 0.001) * 0.01
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "learning_rate": current_lr
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": 100 * correct / total
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int):
        """Train the model"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Update training history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["train_accuracy"].append(train_metrics["accuracy"])
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])
            self.training_history["learning_rate"].append(train_metrics["learning_rate"])
            
            # Log to Weights & Biases
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": train_metrics["learning_rate"]
            })
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            logger.info(f"Learning Rate: {train_metrics['learning_rate']:.6f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
                logger.info(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Finish experiment tracking
        wandb.finish()
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate research report"""
        return {
            "experiment_name": self.config.experiment_name,
            "model_architecture": self.config.model_architecture,
            "dataset_name": self.config.dataset_name,
            "hyperparameters": self.config.hyperparameters,
            "best_val_accuracy": max(self.training_history["val_accuracy"]),
            "final_train_loss": self.training_history["train_loss"][-1],
            "final_val_loss": self.training_history["val_loss"][-1],
            "training_history": self.training_history
        }

# Example usage
def main():
    # Create research configuration
    config = ResearchConfig(
        experiment_name="meta_ai_research_demo",
        model_architecture="transformer",
        dataset_name="synthetic_dataset",
        hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 10,
            "hidden_size": 512,
            "num_heads": 8,
            "num_layers": 6
        },
        reproducibility_seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=True,
        gradient_accumulation_steps=2,
        warmup_steps=1000,
        weight_decay=0.01
    )
    
    # Create research instance
    research = MetaAIResearch(config)
    
    # Create synthetic dataset
    train_dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 10000, (1000, 100)),
        torch.randint(0, 10, (1000,))
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 10000, (200, 100)),
        torch.randint(0, 10, (200,))
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model
    research.train(train_loader, val_loader, num_epochs=10)
    
    # Generate report
    report = research.generate_report()
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Meta AI Practices**

#### **Q1: What are Meta's key AI research areas?**
**Answer**: 
- **Computer Vision**: Image recognition, object detection, video understanding
- **Natural Language Processing**: Language models, translation, understanding
- **Speech and Audio**: Speech recognition, synthesis, audio processing
- **Robotics**: Embodied AI, manipulation, navigation
- **AI Infrastructure**: Distributed training, model serving, optimization
- **AI Ethics**: Responsible AI development and deployment

#### **Q2: How does Meta approach AI research collaboration?**
**Answer**: 
- **Open Research**: Publishing papers and open-sourcing code
- **Cross-team Collaboration**: Internal collaboration between different AI teams
- **External Partnerships**: Collaborations with universities and research institutions
- **Open Source**: Contributing to open source AI projects
- **Knowledge Sharing**: Regular internal and external presentations

#### **Q3: What are Meta's practices for AI model training?**
**Answer**: 
- **Distributed Training**: Large-scale distributed training across multiple GPUs
- **Mixed Precision**: Using FP16 for faster training and reduced memory usage
- **Gradient Accumulation**: Accumulating gradients across multiple batches
- **Learning Rate Scheduling**: Using cosine annealing with warm restarts
- **Regularization**: Dropout, weight decay, and other regularization techniques

#### **Q4: How does Meta ensure AI model reproducibility?**
**Answer**: 
- **Random Seeds**: Setting reproducible random seeds
- **Deterministic Operations**: Using deterministic CUDA operations
- **Version Control**: Tracking code, data, and model versions
- **Documentation**: Detailed documentation of experiments and results
- **Code Sharing**: Open-sourcing research code and models

#### **Q5: What are Meta's AI ethics and responsible AI practices?**
**Answer**: 
- **Fairness**: Ensuring AI systems are fair and unbiased
- **Transparency**: Making AI systems transparent and explainable
- **Privacy**: Protecting user privacy in AI systems
- **Safety**: Ensuring AI systems are safe and reliable
- **Accountability**: Taking responsibility for AI system outcomes

### **Implementation Questions**

#### **Q6: Implement Meta's AI training practices**
**Answer**: See the implementation above with distributed training, mixed precision, and experiment tracking.

#### **Q7: How would you scale Meta's AI training to handle large datasets?**
**Answer**: 
- **Data Parallelism**: Distribute data across multiple GPUs
- **Model Parallelism**: Split large models across multiple GPUs
- **Pipeline Parallelism**: Overlap computation and communication
- **Gradient Accumulation**: Accumulate gradients across multiple batches
- **Mixed Precision**: Use FP16 for faster training and reduced memory usage

#### **Q8: How do you ensure AI model quality at Meta's scale?**
**Answer**: 
- **Automated Testing**: Comprehensive automated testing of AI models
- **A/B Testing**: Large-scale A/B testing of AI models
- **Monitoring**: Continuous monitoring of AI model performance
- **Feedback Loops**: Incorporating user feedback into model improvements
- **Quality Gates**: Quality gates at each stage of the AI pipeline

---

## 🚀 **Next Steps**

1. **Practice**: Implement Meta's AI practices in your own projects
2. **Research**: Follow Meta's AI research publications and open source projects
3. **Collaborate**: Engage with Meta's AI community and open source projects
4. **Learn**: Continuously learn about new AI techniques and best practices
5. **Apply**: Apply Meta's practices to solve real-world AI problems

---

**Ready to learn about Google Brain practices? Let's move to the next section!** 🎯
