# 🔧 Fine-tuning LLMs

> **Master Fine-tuning: from basic techniques to advanced strategies for customizing LLMs**

## 🎯 **Learning Objectives**

- Understand fine-tuning fundamentals and strategies
- Implement different fine-tuning techniques (LoRA, QLoRA, etc.)
- Master parameter-efficient fine-tuning methods
- Handle instruction tuning and alignment
- Build production-ready fine-tuning systems

## 📚 **Table of Contents**

1. [Fine-tuning Fundamentals](#fine-tuning-fundamentals)
2. [Parameter-Efficient Methods](#parameter-efficient-methods)
3. [Instruction Tuning](#instruction-tuning)
4. [Alignment and Safety](#alignment-and-safety)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🔧 **Fine-tuning Fundamentals**

### **Core Concepts**

#### **Concept**
Fine-tuning adapts pre-trained language models to specific tasks or domains by training on task-specific data while preserving the model's general knowledge.

#### **Key Strategies**
- **Full Fine-tuning**: Update all model parameters
- **Parameter-Efficient Fine-tuning**: Update only a subset of parameters
- **Instruction Tuning**: Train on instruction-following data
- **Alignment**: Align model behavior with human preferences
- **Domain Adaptation**: Adapt to specific domains or use cases

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
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
import wandb
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import bitsandbytes as bnb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Fine-tuning configuration"""
    model_name: str = "microsoft/DialoGPT-medium"
    dataset_name: str = "custom_dataset"
    output_dir: str = "./fine_tuned_model"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    device: str = "cuda"
    mixed_precision: bool = True
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_qlora: bool = False
    quantization_config: Optional[BitsAndBytesConfig] = None

class FineTuningDataset(Dataset):
    """Dataset for fine-tuning"""
    
    def __init__(self, data_path: str, tokenizer, config: FineTuningConfig):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.data = self._load_data()
        self.max_length = config.max_length
    
    def _load_data(self) -> List[Dict[str, str]]:
        """Load and preprocess data"""
        data = []
        
        # In production, this would load from various sources
        # For demo, create synthetic instruction-following data
        synthetic_data = [
            {
                "instruction": "Explain what machine learning is",
                "input": "",
                "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
            },
            {
                "instruction": "Translate the following text to French",
                "input": "Hello, how are you?",
                "output": "Bonjour, comment allez-vous?"
            },
            {
                "instruction": "Summarize the following text",
                "input": "Artificial intelligence is transforming various industries by automating tasks and providing intelligent insights.",
                "output": "AI is changing industries through automation and intelligent insights."
            },
            {
                "instruction": "Write a creative story about a robot",
                "input": "",
                "output": "Once upon a time, there was a robot named Alex who dreamed of understanding human emotions..."
            },
            {
                "instruction": "Solve this math problem",
                "input": "What is 15 + 27?",
                "output": "15 + 27 = 42"
            }
        ]
        
        # Repeat synthetic data to create larger dataset
        for _ in range(200):
            data.extend(synthetic_data)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-following prompt
        if item["input"]:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        
        # Tokenize prompt and response
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(item["output"], add_special_tokens=False)
        
        # Combine and add special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(prompt_tokens + response_tokens)
        
        # Create labels (only for response part)
        labels = [-100] * len(prompt_tokens) + response_tokens + [self.tokenizer.eos_token_id]
        
        # Pad or truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            labels.extend([-100] * padding_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor([1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids], dtype=torch.long)
        }

class FineTuningTrainer:
    """Fine-tuning trainer with advanced features"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = self._load_model()
        
        # Setup LoRA if enabled
        if config.use_lora:
            self.model = self._setup_lora()
        
        # Setup training arguments
        self.training_args = self._setup_training_args()
        
        # Setup data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training metrics
        self.training_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": []
        }
    
    def _load_model(self):
        """Load model with optional quantization"""
        if self.config.use_qlora and self.config.quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=self.config.quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                device_map="auto" if self.config.device == "cuda" else None
            )
        
        return model
    
    def _setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation)"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _setup_training_args(self):
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.mixed_precision,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to="wandb" if wandb.api.api_key else None,
            run_name=f"fine_tuning_{self.config.model_name}_{int(time.time())}"
        )
    
    def train(self, train_dataset: FineTuningDataset, eval_dataset: Optional[FineTuningDataset] = None):
        """Train the model"""
        logger.info("Starting fine-tuning")
        
        # Initialize wandb
        if wandb.api.api_key:
            wandb.init(
                project="llm_fine_tuning",
                name=f"{self.config.model_name}_{self.config.dataset_name}",
                config=self.config.__dict__
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save LoRA adapters if using LoRA
        if self.config.use_lora:
            self.model.save_pretrained(f"{self.config.output_dir}/lora_adapters")
        
        logger.info(f"Fine-tuning completed. Model saved to {self.config.output_dir}")
        
        if wandb.api.api_key:
            wandb.finish()
    
    def evaluate(self, eval_dataset: FineTuningDataset) -> Dict[str, float]:
        """Evaluate the model"""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        
        eval_results = trainer.evaluate()
        return eval_results
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        self.model.eval()
        
        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        response_start = generated_text.find("### Response:\n") + len("### Response:\n")
        response = generated_text[response_start:].strip()
        
        return response
    
    def save_model(self, path: str):
        """Save the fine-tuned model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a fine-tuned model"""
        if self.config.use_lora:
            # Load base model and LoRA adapters
            base_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            self.model = PeftModel.from_pretrained(base_model, path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")

class FineTuningEvaluator:
    """Evaluator for fine-tuned models"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def evaluate_instruction_following(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate instruction following capability"""
        results = []
        
        for item in test_data:
            prompt = item["instruction"]
            expected = item["output"]
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Evaluate
            score = self._calculate_similarity(response, expected)
            results.append(score)
        
        return {
            "average_score": np.mean(results),
            "std_score": np.std(results),
            "min_score": np.min(results),
            "max_score": np.max(results)
        }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response for evaluation"""
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = generated_text.find("### Response:\n") + len("### Response:\n")
        response = generated_text[response_start:].strip()
        
        return response
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

# Example usage
def main():
    # Create configuration
    config = FineTuningConfig(
        model_name="microsoft/DialoGPT-medium",
        dataset_name="instruction_dataset",
        output_dir="./fine_tuned_model",
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-5,
        warmup_steps=100,
        max_length=512,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        weight_decay=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=True,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_qlora=False
    )
    
    # Create trainer
    trainer = FineTuningTrainer(config)
    
    # Create datasets
    train_dataset = FineTuningDataset("train_data", trainer.tokenizer, config)
    eval_dataset = FineTuningDataset("eval_data", trainer.tokenizer, config)
    
    # Train model
    trainer.train(train_dataset, eval_dataset)
    
    # Evaluate model
    eval_results = trainer.evaluate(eval_dataset)
    print(f"Evaluation results: {eval_results}")
    
    # Test generation
    test_prompts = [
        "Explain what artificial intelligence is",
        "Write a short poem about technology",
        "What is the capital of France?"
    ]
    
    for prompt in test_prompts:
        response = trainer.generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 50)
    
    # Create evaluator
    evaluator = FineTuningEvaluator(trainer.model, trainer.tokenizer)
    
    # Evaluate instruction following
    test_data = [
        {
            "instruction": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence."
        },
        {
            "instruction": "Translate 'hello' to Spanish",
            "output": "Hola"
        }
    ]
    
    eval_results = evaluator.evaluate_instruction_following(test_data)
    print(f"Instruction following evaluation: {eval_results}")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Fine-tuning Theory**

#### **Q1: What is fine-tuning and why is it important?**
**Answer**: 
- **Definition**: Adapting pre-trained language models to specific tasks or domains
- **Importance**: 
  - Leverages pre-trained knowledge while adapting to specific needs
  - More efficient than training from scratch
  - Enables customization for specific use cases
  - Improves performance on target tasks
  - Reduces computational requirements compared to full training

#### **Q2: What are the different fine-tuning strategies?**
**Answer**: 
- **Full Fine-tuning**: Update all model parameters
- **Parameter-Efficient Fine-tuning**: Update only a subset of parameters
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
  - AdaLoRA
  - Prefix Tuning
  - P-Tuning v2
- **Instruction Tuning**: Train on instruction-following data
- **Alignment**: Align model behavior with human preferences

#### **Q3: What is LoRA and how does it work?**
**Answer**: 
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **How it works**: 
  - Decomposes weight updates into low-rank matrices
  - Freezes original model weights
  - Only trains small adapter matrices
  - Reduces trainable parameters significantly
- **Benefits**: 
  - Memory efficient
  - Fast training
  - Easy to switch between tasks
  - Preserves original model capabilities

#### **Q4: What is instruction tuning?**
**Answer**: 
- **Definition**: Training models to follow human instructions
- **Process**: 
  - Collect instruction-following datasets
  - Format as instruction-input-output triplets
  - Train model to generate appropriate responses
- **Benefits**: 
  - Improves model helpfulness
  - Enables better task following
  - Reduces harmful outputs
  - Makes models more controllable

#### **Q5: How do you evaluate fine-tuned models?**
**Answer**: 
- **Task-specific metrics**: Accuracy, F1-score, BLEU, etc.
- **Instruction following**: Measure adherence to instructions
- **Safety evaluation**: Check for harmful outputs
- **Generalization**: Test on unseen tasks
- **Human evaluation**: Collect human feedback
- **Automated evaluation**: Use metrics and benchmarks

### **Implementation Questions**

#### **Q6: Implement fine-tuning from scratch**
**Answer**: See the implementation above with LoRA, instruction tuning, and evaluation.

#### **Q7: How would you optimize fine-tuning for production?**
**Answer**: 
- **Memory optimization**: Use LoRA, quantization, gradient checkpointing
- **Speed optimization**: Use mixed precision, gradient accumulation
- **Quality optimization**: Use instruction tuning, alignment
- **Cost optimization**: Use parameter-efficient methods
- **Monitoring**: Track training metrics and model performance
- **Versioning**: Track model versions and performance

#### **Q8: How do you handle data quality in fine-tuning?**
**Answer**: 
- **Data validation**: Check for errors and inconsistencies
- **Data cleaning**: Remove noise and irrelevant content
- **Data augmentation**: Increase dataset diversity
- **Data balancing**: Ensure balanced representation
- **Quality control**: Implement quality checks and filters
- **Human review**: Review and validate training data

---

## 🚀 **Next Steps**

1. **Practice**: Implement all fine-tuning techniques
2. **Optimize**: Focus on efficiency and quality
3. **Deploy**: Build production fine-tuning systems
4. **Extend**: Learn about advanced techniques and best practices
5. **Interview**: Practice fine-tuning interview questions

---

**Ready to learn about Vector Databases? Let's move to the next section!** 🎯
