# 🎨 Diffusion Models

> **Master Diffusion Models: from mathematical foundations to production implementation**

## 🎯 **Learning Objectives**

- Understand diffusion model theory and mathematical foundations
- Implement diffusion models from scratch in Python
- Master denoising processes and sampling strategies
- Handle conditional generation and fine-tuning
- Build production-ready diffusion model systems

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations)
2. [Implementation from Scratch](#implementation-from-scratch)
3. [Sampling Strategies](#sampling-strategies)
4. [Conditional Generation](#conditional-generation)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🧮 **Mathematical Foundations**

### **Diffusion Model Theory**

#### **Concept**
Diffusion models generate data by learning to reverse a noise corruption process, gradually denoising random noise into meaningful data.

#### **Math Behind**
- **Forward Process**: `q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)`
- **Reverse Process**: `p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))`
- **Loss Function**: `L = E[||ε - ε_θ(x_t, t)||²]`
- **DDPM Sampling**: `x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t,t)) + σ_t z`

#### **Code Example**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import math
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Diffusion model configuration"""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    image_size: int = 32
    num_channels: int = 3
    hidden_dim: int = 128
    num_layers: int = 4
    device: str = "cuda"

class DiffusionModel(nn.Module):
    """Diffusion model implementation"""
    
    def __init__(self, config: DiffusionConfig):
        super(DiffusionModel, self).__init__()
        self.config = config
        self.num_timesteps = config.num_timesteps
        
        # Create noise schedule
        self.betas = self._create_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate variance schedule
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculate posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        # Create denoising network
        self.denoising_net = self._create_denoising_network()
    
    def _create_beta_schedule(self) -> torch.Tensor:
        """Create noise schedule"""
        return torch.linspace(self.config.beta_start, self.config.beta_end, self.num_timesteps)
    
    def _create_denoising_network(self) -> nn.Module:
        """Create denoising network"""
        return UNet(
            in_channels=self.config.num_channels,
            out_channels=self.config.num_channels,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_timesteps=self.num_timesteps
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t|x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from p_θ(x_{t-1}|x_t)"""
        # Predict noise
        predicted_noise = self.denoising_net(x, t)
        
        # Calculate mean
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        mean = sqrt_recip_alphas_t * (x - self.betas[t].reshape(-1, 1, 1, 1) * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # Calculate variance
        if t[0] == 0:
            return mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from the model"""
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        
        return x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Add noise to input
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        predicted_noise = self.denoising_net(x_noisy, t)
        
        return predicted_noise, noise

class UNet(nn.Module):
    """U-Net architecture for diffusion models"""
    
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, num_layers: int, num_timesteps: int):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Input projection
        self.input_projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(ResidualBlock(hidden_dim, hidden_dim, hidden_dim))
        
        for i in range(num_layers - 1):
            self.down_blocks.append(ResidualBlock(hidden_dim * (2 ** i), hidden_dim * (2 ** (i + 1)), hidden_dim))
            self.down_blocks.append(Downsample(hidden_dim * (2 ** (i + 1))))
        
        # Middle block
        self.middle_block = ResidualBlock(hidden_dim * (2 ** (num_layers - 1)), hidden_dim * (2 ** (num_layers - 1)), hidden_dim)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_layers - 1)):
            self.up_blocks.append(Upsample(hidden_dim * (2 ** (i + 1))))
            self.up_blocks.append(ResidualBlock(hidden_dim * (2 ** (i + 1)) + hidden_dim * (2 ** i), hidden_dim * (2 ** i), hidden_dim))
        
        self.up_blocks.append(ResidualBlock(hidden_dim * 2, hidden_dim, hidden_dim))
        
        # Output projection
        self.output_projection = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self._get_time_embedding(t)
        
        # Input projection
        x = self.input_projection(x)
        
        # Downsampling
        skip_connections = []
        for block in self.down_blocks:
            if isinstance(block, Downsample):
                skip_connections.append(x)
                x = block(x)
            else:
                x = block(x, t_emb)
        
        # Middle block
        x = self.middle_block(x, t_emb)
        
        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, Upsample):
                x = block(x)
                skip_connection = skip_connections.pop()
                x = torch.cat([x, skip_connection], dim=1)
            else:
                x = block(x, t_emb)
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Get time embedding"""
        # Sinusoidal positional encoding
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        return self.time_embedding(emb)

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Normalization
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Activation
        self.activation = nn.SiLU()
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First convolution
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        # Add time embedding
        t_emb = self.time_proj(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        
        # Second convolution
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        # Residual connection
        return x + residual

class Downsample(nn.Module):
    """Downsampling layer"""
    
    def __init__(self, channels: int):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """Upsampling layer"""
    
    def __init__(self, channels: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DiffusionTrainer:
    """Diffusion model trainer"""
    
    def __init__(self, model: DiffusionModel, config: DiffusionConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training metrics
        self.training_history = {
            "loss": [],
            "epoch": []
        }
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(self.device)
            batch_size = x.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
            
            # Forward pass
            predicted_noise, noise = self.model(x, t)
            
            # Calculate loss
            loss = self.criterion(predicted_noise, noise)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def train(self, dataloader, num_epochs: int):
        """Train the model"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(dataloader)
            
            self.training_history["loss"].append(epoch_loss)
            self.training_history["epoch"].append(epoch)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"diffusion_model_epoch_{epoch+1}.pth")
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """Generate samples from the model"""
        self.model.eval()
        
        with torch.no_grad():
            shape = (num_samples, self.config.num_channels, self.config.image_size, self.config.image_size)
            samples = self.model.p_sample_loop(shape, self.device)
        
        return samples

# Example usage
def main():
    # Create configuration
    config = DiffusionConfig(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        image_size=32,
        num_channels=3,
        hidden_dim=128,
        num_layers=4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create model
    model = DiffusionModel(config)
    
    # Create trainer
    trainer = DiffusionTrainer(model, config)
    
    # Create synthetic dataset
    from torch.utils.data import DataLoader, TensorDataset
    
    # Generate synthetic images
    synthetic_data = torch.randn(1000, 3, 32, 32)
    dataset = TensorDataset(synthetic_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    trainer.train(dataloader, num_epochs=50)
    
    # Generate samples
    samples = trainer.generate_samples(num_samples=16)
    print(f"Generated samples shape: {samples.shape}")
    
    # Visualize samples
    samples_np = samples.cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        # Convert from [-1, 1] to [0, 1] for visualization
        img = (samples_np[i].transpose(1, 2, 0) + 1) / 2
        img = np.clip(img, 0, 1)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Diffusion Model Theory**

#### **Q1: What are diffusion models and how do they work?**
**Answer**: 
- **Concept**: Diffusion models generate data by learning to reverse a noise corruption process
- **Forward Process**: Gradually add noise to data until it becomes pure noise
- **Reverse Process**: Learn to denoise and recover original data
- **Training**: Train a neural network to predict the noise added at each timestep
- **Sampling**: Start with noise and iteratively denoise to generate new data

#### **Q2: What is the mathematical foundation of diffusion models?**
**Answer**: 
- **Forward Process**: `q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)`
- **Reverse Process**: `p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))`
- **Loss Function**: `L = E[||ε - ε_θ(x_t, t)||²]`
- **DDPM Sampling**: `x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t,t)) + σ_t z`

#### **Q3: How do diffusion models compare to GANs and VAEs?**
**Answer**: 
- **GANs**: Adversarial training, mode collapse issues, unstable training
- **VAEs**: Probabilistic framework, blurry reconstructions, limited quality
- **Diffusion Models**: Stable training, high quality samples, no mode collapse
- **Trade-offs**: Diffusion models are slower to sample but more stable to train

#### **Q4: What are the advantages and disadvantages of diffusion models?**
**Answer**: 
- **Advantages**: 
  - Stable training process
  - High quality samples
  - No mode collapse
  - Good theoretical foundation
- **Disadvantages**: 
  - Slow sampling (many timesteps)
  - Computationally expensive
  - Requires careful noise scheduling

#### **Q5: How do you improve diffusion model sampling speed?**
**Answer**: 
- **DDIM**: Deterministic sampling with fewer steps
- **DPM-Solver**: Fast ODE solver for diffusion models
- **Progressive Distillation**: Train smaller models to match larger ones
- **Latent Diffusion**: Work in lower-dimensional latent space
- **Few-step Sampling**: Train models for fewer timesteps

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different datasets
2. **Optimize**: Focus on sampling speed and quality
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about conditional generation and fine-tuning
5. **Interview**: Practice diffusion model interview questions

---

**Ready to learn about LLM Training? Let's move to the next section!** 🎯
