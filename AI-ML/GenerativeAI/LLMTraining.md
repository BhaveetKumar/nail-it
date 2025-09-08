# 🧠 LLM Training

> **Master Large Language Model Training: from data preparation to production deployment**

## 🎯 **Learning Objectives**

- Understand LLM training pipeline and infrastructure
- Implement distributed training strategies
- Master data preprocessing and tokenization
- Handle model optimization and memory management
- Build production-ready LLM training systems

## 📚 **Table of Contents**

1. [Training Pipeline](#training-pipeline)
2. [Data Processing](#data-processing)
3. [Distributed Training](#distributed-training)
4. [Optimization Techniques](#optimization-techniques)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🏗️ **Training Pipeline**

### **LLM Training Architecture**

#### **Concept**
LLM training involves massive datasets, distributed computing, and sophisticated optimization techniques to train models with billions of parameters.

#### **Key Components**
- **Data Pipeline**: Large-scale data processing and tokenization
- **Model Architecture**: Transformer-based language models
- **Training Infrastructure**: Distributed training across multiple GPUs/TPUs
- **Optimization**: Memory-efficient training techniques
- **Monitoring**: Training progress and model performance tracking

#### **Code Example**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMTrainingConfig:
    """LLM training configuration"""
    model_name: str = "gpt2"
    model_size: str = "small"  # small, medium, large, xlarge
    vocab_size: int = 50257
    max_length: int = 1024
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cuda"
    mixed_precision: bool = True
    distributed_training: bool = False
    world_size: int = 1
    rank: int = 0

class LLMDataset(Dataset):
    """LLM dataset with advanced preprocessing"""
    
    def __init__(self, data_path: str, tokenizer, config: LLMTrainingConfig):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.data = self._load_data()
        self.max_length = config.max_length
    
    def _load_data(self) -> List[str]:
        """Load and preprocess data"""
        data = []
        
        # In production, this would load from various sources
        # For demo, create synthetic text data
        synthetic_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Large language models are trained on massive datasets.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers.",
            "Transformers revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of input.",
            "Pre-training and fine-tuning are common approaches in NLP.",
            "Transfer learning enables models to leverage pre-trained knowledge.",
            "Ethical AI development requires careful consideration of bias and fairness."
        ]
        
        # Repeat synthetic data to create larger dataset
        for _ in range(1000):
            data.extend(synthetic_texts)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create input and target (shifted by 1)
        input_ids = tokens[0, :-1]
        target_ids = tokens[0, 1:]
        
        return {
            'input_ids': input_ids,
            'labels': target_ids
        }

class LLMModel(nn.Module):
    """LLM model with advanced features"""
    
    def __init__(self, config: LLMTrainingConfig):
        super(LLMModel, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_length = config.max_length
        
        # Model components
        self.embedding = nn.Embedding(config.vocab_size, 768)
        self.pos_embedding = nn.Embedding(config.max_length, 768)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=12
        )
        self.ln_f = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.pos_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Transformer
        hidden_states = self.transformer(embeddings, src_key_padding_mask=~attention_mask.bool())
        
        # Layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }

class LLMTrainer:
    """LLM trainer with advanced features"""
    
    def __init__(self, model: LLMModel, config: LLMTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup distributed training
        if config.distributed_training:
            self._setup_distributed()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
            eta_min=config.learning_rate * 0.1
        )
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training metrics
        self.training_history = {
            "loss": [],
            "learning_rate": [],
            "perplexity": []
        }
        
        # Setup experiment tracking
        if config.rank == 0:  # Only log from rank 0
            wandb.init(
                project="llm_training",
                name=f"{config.model_name}_{config.model_size}",
                config=config.__dict__
            )
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if not dist.is_initialized():
            init_process_group(backend='nccl')
        
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.config.rank],
            output_device=self.config.rank
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, labels)
                    loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(input_ids, labels)
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_tokens += input_ids.numel()
            
            # Log progress
            if batch_idx % 100 == 0 and self.config.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "learning_rate": current_lr,
            "tokens_processed": total_tokens
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids, labels)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "tokens_processed": total_tokens
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Train the model"""
        if self.config.rank == 0:
            logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        best_val_perplexity = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Update training history
            self.training_history["loss"].append(train_metrics["loss"])
            self.training_history["learning_rate"].append(train_metrics["learning_rate"])
            self.training_history["perplexity"].append(train_metrics["perplexity"])
            
            # Log metrics
            if self.config.rank == 0:
                # Log to Weights & Biases
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_perplexity": train_metrics["perplexity"],
                    "val_perplexity": val_metrics["perplexity"],
                    "learning_rate": train_metrics["learning_rate"]
                })
                
                # Log epoch results
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Perplexity: {train_metrics['perplexity']:.2f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Perplexity: {val_metrics['perplexity']:.2f}")
                logger.info(f"Learning Rate: {train_metrics['learning_rate']:.6f}")
                
                # Save best model
                if val_metrics['perplexity'] < best_val_perplexity:
                    best_val_perplexity = val_metrics['perplexity']
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth")
                    logger.info(f"New best model saved with perplexity: {best_val_perplexity:.2f}")
        
        if self.config.rank == 0:
            logger.info(f"Training completed. Best validation perplexity: {best_val_perplexity:.2f}")
            wandb.finish()
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        if self.config.rank == 0:  # Only save from rank 0
            checkpoint = {
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'training_history': self.training_history
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Model saved to {filepath}")
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text from prompt"""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text

# Example usage
def main():
    # Create configuration
    config = LLMTrainingConfig(
        model_name="gpt2",
        model_size="small",
        vocab_size=50257,
        max_length=1024,
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_epochs=3,
        warmup_steps=1000,
        weight_decay=0.01,
        max_grad_norm=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=True,
        distributed_training=False,
        world_size=1,
        rank=0
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    train_dataset = LLMDataset("train_data", tokenizer, config)
    val_dataset = LLMDataset("val_data", tokenizer, config)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = LLMModel(config)
    
    # Create trainer
    trainer = LLMTrainer(model, config)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Generate text
    generated_text = trainer.generate_text("The future of artificial intelligence is", max_length=50)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **LLM Training Theory**

#### **Q1: What are the key challenges in training large language models?**
**Answer**: 
- **Computational Resources**: Massive compute requirements (GPUs/TPUs)
- **Memory Management**: Large models require significant memory
- **Data Quality**: Need high-quality, diverse training data
- **Training Stability**: Ensuring stable training across long periods
- **Distributed Training**: Coordinating training across multiple devices
- **Cost**: Expensive to train and maintain large models

#### **Q2: How do you handle memory optimization in LLM training?**
**Answer**: 
- **Gradient Checkpointing**: Trade compute for memory by recomputing activations
- **Mixed Precision Training**: Use FP16 to reduce memory usage
- **Model Parallelism**: Split model across multiple devices
- **Gradient Accumulation**: Simulate larger batch sizes
- **Parameter Sharding**: Distribute parameters across devices
- **Activation Offloading**: Move activations to CPU memory

#### **Q3: What are the different distributed training strategies for LLMs?**
**Answer**: 
- **Data Parallelism**: Replicate model, split data across devices
- **Model Parallelism**: Split model across devices
- **Pipeline Parallelism**: Split model layers across devices
- **Tensor Parallelism**: Split tensor operations across devices
- **Hybrid Approaches**: Combine multiple strategies

#### **Q4: How do you ensure training stability in LLM training?**
**Answer**: 
- **Gradient Clipping**: Prevent exploding gradients
- **Learning Rate Scheduling**: Careful learning rate management
- **Weight Initialization**: Proper weight initialization
- **Batch Normalization**: Stabilize training
- **Monitoring**: Continuous monitoring of training metrics
- **Checkpointing**: Regular model saving

#### **Q5: What are the key metrics to monitor during LLM training?**
**Answer**: 
- **Loss**: Training and validation loss
- **Perplexity**: Measure of model uncertainty
- **Learning Rate**: Current learning rate
- **Gradient Norm**: Gradient magnitude
- **Memory Usage**: GPU/CPU memory utilization
- **Throughput**: Tokens processed per second

### **Implementation Questions**

#### **Q6: Implement LLM training from scratch**
**Answer**: See the implementation above with distributed training, mixed precision, and optimization techniques.

#### **Q7: How would you scale LLM training to handle larger models?**
**Answer**: 
- **Model Parallelism**: Split model across multiple devices
- **Pipeline Parallelism**: Overlap computation and communication
- **Tensor Parallelism**: Distribute tensor operations
- **Memory Optimization**: Use gradient checkpointing and mixed precision
- **Data Pipeline**: Optimize data loading and preprocessing
- **Infrastructure**: Use specialized hardware (TPUs, high-memory GPUs)

#### **Q8: How do you handle data preprocessing for LLM training?**
**Answer**: 
- **Tokenization**: Convert text to tokens
- **Data Cleaning**: Remove noise and irrelevant content
- **Data Augmentation**: Increase dataset diversity
- **Data Validation**: Ensure data quality
- **Data Streaming**: Handle large datasets efficiently
- **Data Versioning**: Track data changes and versions

---

## 🚀 **Next Steps**

1. **Practice**: Implement all training strategies and test with different models
2. **Optimize**: Focus on memory efficiency and training speed
3. **Deploy**: Build production training infrastructure
4. **Extend**: Learn about fine-tuning and instruction following
5. **Interview**: Practice LLM training interview questions

---

**Ready to learn about Prompt Engineering? Let's move to the next section!** 🎯
