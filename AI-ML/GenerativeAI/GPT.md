# 🤖 GPT (Generative Pre-trained Transformer)

> **Master GPT: from mathematical foundations to production implementation of Large Language Models**

## 🎯 **Learning Objectives**

- Understand GPT architecture and transformer mechanisms
- Implement GPT from scratch in Python and Go
- Master attention mechanisms and positional encoding
- Handle tokenization and text preprocessing
- Build production-ready GPT systems

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations)
2. [Implementation from Scratch](#implementation-from-scratch)
3. [Attention Mechanisms](#attention-mechanisms)
4. [Training and Fine-tuning](#training-and-fine-tuning)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🧮 **Mathematical Foundations**

### **GPT Theory**

#### **Concept**
GPT (Generative Pre-trained Transformer) is an autoregressive language model that uses transformer architecture to generate human-like text.

#### **Math Behind**
- **Self-Attention**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Multi-Head Attention**: `MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O`
- **Positional Encoding**: `PE(pos,2i) = sin(pos/10000^(2i/d_model))`
- **Layer Normalization**: `LayerNorm(x) = γ * (x - μ)/σ + β`

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super(GPT, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)
        
        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        return logits, attention_weights
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Generate text using the model"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == 0:  # Assuming 0 is EOS token
                    break
        
        return generated

# Example usage
# Create a simple GPT model
vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048

gpt_model = GPT(vocab_size, d_model, num_heads, num_layers, d_ff)

# Test forward pass
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

logits, attention_weights = gpt_model(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Number of attention weight tensors: {len(attention_weights)}")

# Test generation
start_tokens = torch.randint(0, vocab_size, (1, 5))
generated = gpt_model.generate(start_tokens, max_length=20, temperature=0.8)
print(f"Generated sequence shape: {generated.shape}")
```

---

## 🎯 **Attention Mechanisms**

### **Advanced Attention Implementations**

#### **Concept**
Attention mechanisms allow models to focus on relevant parts of the input sequence.

#### **Code Example**

```python
class AdvancedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(AdvancedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None, attention_type='scaled_dot_product'):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if attention_type == 'scaled_dot_product':
            output, weights = self.scaled_dot_product_attention(Q, K, V, mask)
        elif attention_type == 'additive':
            output, weights = self.additive_attention(Q, K, V, mask)
        elif attention_type == 'local':
            output, weights = self.local_attention(Q, K, V, mask)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Standard scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def additive_attention(self, Q, K, V, mask=None):
        """Additive attention mechanism"""
        # Learnable parameters
        W_a = nn.Parameter(torch.randn(self.d_k, self.d_k))
        v_a = nn.Parameter(torch.randn(self.d_k))
        
        # Compute attention scores
        Q_transformed = torch.matmul(Q, W_a)
        scores = torch.matmul(torch.tanh(Q_transformed + K.transpose(-2, -1)), v_a)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def local_attention(self, Q, K, V, mask=None, window_size=10):
        """Local attention with sliding window"""
        seq_len = Q.size(2)
        output = torch.zeros_like(Q)
        attention_weights = torch.zeros(Q.size(0), Q.size(1), seq_len, seq_len)
        
        for i in range(seq_len):
            # Define local window
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            
            # Extract local keys and values
            K_local = K[:, :, start:end, :]
            V_local = V[:, :, start:end, :]
            
            # Compute attention for local window
            scores = torch.matmul(Q[:, :, i:i+1, :], K_local.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                local_mask = mask[:, :, i, start:end]
                scores = scores.masked_fill(local_mask == 0, -1e9)
            
            local_weights = F.softmax(scores, dim=-1)
            local_weights = self.dropout(local_weights)
            
            # Compute output
            output[:, :, i:i+1, :] = torch.matmul(local_weights, V_local)
            attention_weights[:, :, i, start:end] = local_weights.squeeze(2)
        
        return output, attention_weights

class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_pattern='strided', dropout=0.1):
        super(SparseAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sparsity_pattern = sparsity_pattern
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def create_sparse_mask(self, seq_len, sparsity_pattern):
        """Create sparse attention mask"""
        mask = torch.zeros(seq_len, seq_len)
        
        if sparsity_pattern == 'strided':
            # Strided pattern: attend to every k-th token
            stride = 4
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    if j <= i:  # Causal constraint
                        mask[i, j] = 1
        elif sparsity_pattern == 'local':
            # Local pattern: attend to nearby tokens
            window_size = 8
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = 1
        elif sparsity_pattern == 'random':
            # Random pattern: attend to random tokens
            for i in range(seq_len):
                num_attended = min(8, i + 1)
                attended_indices = torch.randperm(i + 1)[:num_attended]
                mask[i, attended_indices] = 1
        
        return mask
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Create sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, self.sparsity_pattern)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).to(query.device)
        
        # Combine with input mask
        if mask is not None:
            combined_mask = mask * sparse_mask
        else:
            combined_mask = sparse_mask
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention with sparse mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(combined_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights

# Example usage
# Test different attention mechanisms
d_model = 512
num_heads = 8
seq_len = 20
batch_size = 2

# Create test data
query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

# Test advanced attention
advanced_attention = AdvancedAttention(d_model, num_heads)
output, weights = advanced_attention(query, key, value, attention_type='scaled_dot_product')
print(f"Advanced attention output shape: {output.shape}")

# Test sparse attention
sparse_attention = SparseAttention(d_model, num_heads, sparsity_pattern='strided')
output, weights = sparse_attention(query, key, value)
print(f"Sparse attention output shape: {output.shape}")
```

---

## 🎯 **Interview Questions**

### **GPT Theory**

#### **Q1: What is the difference between GPT and BERT?**
**Answer**: 
- **GPT**: Autoregressive, generates text left-to-right, uses causal masking
- **BERT**: Bidirectional, understands context from both directions, uses masked language modeling
- **Training**: GPT uses next token prediction, BERT uses masked token prediction
- **Use Cases**: GPT for generation, BERT for understanding/classification

#### **Q2: Explain the transformer architecture in GPT**
**Answer**: 
- **Self-Attention**: Allows each position to attend to all previous positions
- **Multi-Head Attention**: Multiple attention heads capture different types of relationships
- **Feed-Forward Networks**: Apply point-wise transformations
- **Residual Connections**: Help with gradient flow and training stability
- **Layer Normalization**: Stabilizes training and improves convergence

#### **Q3: What is positional encoding and why is it needed?**
**Answer**: 
- **Purpose**: Provides sequence order information since transformers have no inherent notion of position
- **Implementation**: Sinusoidal functions with different frequencies
- **Formula**: `PE(pos,2i) = sin(pos/10000^(2i/d_model))`
- **Alternative**: Learned positional embeddings
- **Why Needed**: Without it, the model would treat input as a bag of tokens

#### **Q4: How does GPT handle variable-length sequences?**
**Answer**: 
- **Padding**: Pad shorter sequences to match the longest in the batch
- **Masking**: Use attention masks to ignore padded tokens
- **Causal Masking**: Ensure autoregressive property by masking future tokens
- **Dynamic Batching**: Group sequences of similar lengths together
- **Truncation**: Limit sequence length to maximum model capacity

#### **Q5: What are the challenges in training large language models?**
**Answer**: 
- **Computational**: Requires massive compute resources and time
- **Memory**: Large models need significant GPU memory
- **Data**: Need large, high-quality text datasets
- **Stability**: Training can be unstable with large models
- **Cost**: Expensive to train and maintain
- **Bias**: Models can learn and amplify biases in training data

### **Implementation Questions**

#### **Q6: Implement GPT from scratch**
**Answer**: See the implementation above with transformer blocks, attention mechanisms, and generation.

#### **Q7: How would you optimize GPT inference for production?**
**Answer**: 
- **Quantization**: Reduce precision (FP16, INT8) to decrease memory usage
- **Pruning**: Remove less important weights
- **Knowledge Distillation**: Train smaller models to mimic larger ones
- **Caching**: Cache attention computations for repeated sequences
- **Batching**: Process multiple requests together
- **Model Parallelism**: Split model across multiple devices

#### **Q8: How do you handle different languages in GPT?**
**Answer**: 
- **Multilingual Training**: Train on text from multiple languages
- **Tokenization**: Use subword tokenization (BPE, SentencePiece) for better handling
- **Cross-lingual Transfer**: Leverage similarities between languages
- **Language-specific Models**: Train separate models for different languages
- **Code-switching**: Handle mixed-language text in training data

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different datasets
2. **Optimize**: Focus on performance and memory efficiency
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about diffusion models and other generative techniques
5. **Interview**: Practice GPT and transformer interview questions

---

**Ready to learn about Diffusion Models? Let's move to the next section!** 🎯
