# 🔄 Transformers: Attention Is All You Need

> **Complete guide to Transformer architecture, attention mechanisms, and modern NLP**

## 🎯 **Learning Objectives**

- Master Transformer architecture and attention mechanisms
- Understand self-attention and multi-head attention
- Implement Transformers for various NLP tasks
- Apply Transformers to real-world problems
- Optimize Transformer performance and training

## 📚 **Table of Contents**

1. [Transformer Architecture](#transformer-architecture)
2. [Attention Mechanisms](#attention-mechanisms)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Implementation Examples](#implementation-examples)
5. [Applications](#applications)
6. [Interview Questions](#interview-questions)

---

## 🏗️ **Transformer Architecture**

### **Concept**

Transformers are neural network architectures that rely entirely on attention mechanisms, eliminating the need for recurrent or convolutional layers.

### **Key Components**

1. **Multi-Head Attention**: Parallel attention mechanisms
2. **Position Encoding**: Adds positional information to input embeddings
3. **Feed-Forward Networks**: Point-wise fully connected layers
4. **Layer Normalization**: Stabilizes training
5. **Residual Connections**: Helps with gradient flow

### **Architecture Diagram**

```
Input Embeddings + Position Encoding
              │
              ▼
    ┌─────────────────────┐
    │   Multi-Head        │
    │   Attention         │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Add & Norm        │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Feed Forward      │
    │   Network           │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Add & Norm        │
    └─────────────────────┘
              │
              ▼
         Output
```

---

## 🎯 **Attention Mechanisms**

### **1. Self-Attention**

**Purpose**: Allows each position to attend to all positions in the input sequence.

**Formula**: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`

**Code Example**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Example usage
d_model = 512
seq_len = 100
batch_size = 32

self_attention = SelfAttention(d_model)
x = torch.randn(batch_size, seq_len, d_model)
output, attention_weights = self_attention(x)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

### **2. Multi-Head Attention**

**Purpose**: Allows the model to jointly attend to information from different representation subspaces.

**Code Example**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output, attention_weights

# Example usage
d_model = 512
num_heads = 8
seq_len = 100
batch_size = 32

multi_head_attention = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)
output, attention_weights = multi_head_attention(x)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

### **3. Position Encoding**

**Purpose**: Adds positional information to input embeddings since Transformers have no inherent notion of position.

**Formula**: 
- `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

**Code Example**:
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Example usage
d_model = 512
max_seq_len = 1000

pos_encoding = PositionalEncoding(d_model, max_seq_len)
x = torch.randn(100, 32, d_model)  # (seq_len, batch_size, d_model)
output = pos_encoding(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

---

## 🧮 **Mathematical Foundations**

### **Complete Transformer Implementation**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output layer
        output = self.output_layer(x)
        
        return output

# Example usage
vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_len = 1000

transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)
batch_size = 32
seq_len = 100

# Input: batch of tokenized sequences
x = torch.randint(0, vocab_size, (batch_size, seq_len))
output = transformer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### **Golang Implementation**

```go
package main

import (
    "fmt"
    "math"
)

type MultiHeadAttention struct {
    DModel   int
    NumHeads int
    DK       int
    
    // Weight matrices
    WQ [][]float64
    WK [][]float64
    WV [][]float64
    WO [][]float64
}

func NewMultiHeadAttention(dModel, numHeads int) *MultiHeadAttention {
    mha := &MultiHeadAttention{
        DModel:   dModel,
        NumHeads: numHeads,
        DK:       dModel / numHeads,
    }
    
    // Initialize weight matrices
    mha.WQ = make([][]float64, dModel)
    mha.WK = make([][]float64, dModel)
    mha.WV = make([][]float64, dModel)
    mha.WO = make([][]float64, dModel)
    
    for i := 0; i < dModel; i++ {
        mha.WQ[i] = make([]float64, dModel)
        mha.WK[i] = make([]float64, dModel)
        mha.WV[i] = make([]float64, dModel)
        mha.WO[i] = make([]float64, dModel)
    }
    
    return mha
}

func (mha *MultiHeadAttention) Forward(x [][]float64) [][]float64 {
    batchSize := len(x)
    seqLen := len(x[0])
    
    // Compute Q, K, V
    Q := mha.computeQ(x)
    K := mha.computeK(x)
    V := mha.computeV(x)
    
    // Reshape for multi-head attention
    Q = mha.reshapeForHeads(Q, batchSize, seqLen)
    K = mha.reshapeForHeads(K, batchSize, seqLen)
    V = mha.reshapeForHeads(V, batchSize, seqLen)
    
    // Compute attention scores
    scores := mha.computeAttentionScores(Q, K)
    
    // Apply softmax
    attentionWeights := mha.softmax(scores)
    
    // Apply attention to values
    context := mha.applyAttention(attentionWeights, V)
    
    // Concatenate heads
    context = mha.concatenateHeads(context, batchSize, seqLen)
    
    // Final linear transformation
    output := mha.computeOutput(context)
    
    return output
}

func (mha *MultiHeadAttention) computeQ(x [][]float64) [][]float64 {
    // Implementation of Q computation
    return x // Simplified for example
}

func (mha *MultiHeadAttention) computeK(x [][]float64) [][]float64 {
    // Implementation of K computation
    return x // Simplified for example
}

func (mha *MultiHeadAttention) computeV(x [][]float64) [][]float64 {
    // Implementation of V computation
    return x // Simplified for example
}

func (mha *MultiHeadAttention) reshapeForHeads(x [][]float64, batchSize, seqLen int) [][]float64 {
    // Reshape for multi-head attention
    return x // Simplified for example
}

func (mha *MultiHeadAttention) computeAttentionScores(Q, K [][]float64) [][]float64 {
    // Compute attention scores: Q * K^T / sqrt(d_k)
    return Q // Simplified for example
}

func (mha *MultiHeadAttention) softmax(x [][]float64) [][]float64 {
    // Apply softmax
    return x // Simplified for example
}

func (mha *MultiHeadAttention) applyAttention(weights, values [][]float64) [][]float64 {
    // Apply attention weights to values
    return values // Simplified for example
}

func (mha *MultiHeadAttention) concatenateHeads(x [][]float64, batchSize, seqLen int) [][]float64 {
    // Concatenate heads
    return x // Simplified for example
}

func (mha *MultiHeadAttention) computeOutput(x [][]float64) [][]float64 {
    // Final linear transformation
    return x // Simplified for example
}

func main() {
    dModel := 512
    numHeads := 8
    
    mha := NewMultiHeadAttention(dModel, numHeads)
    
    // Example input
    batchSize := 32
    seqLen := 100
    x := make([][]float64, batchSize)
    for i := range x {
        x[i] = make([]float64, seqLen)
    }
    
    output := mha.Forward(x)
    
    fmt.Printf("Output shape: %dx%d\n", len(output), len(output[0]))
}
```

---

## 🎯 **Applications**

### **1. Text Classification**

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, max_seq_len):
        super(TransformerClassifier, self).__init__()
        
        self.transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff=2048, max_seq_len=max_seq_len)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Get transformer output
        transformer_output = self.transformer(x)
        
        # Use [CLS] token or mean pooling
        pooled_output = transformer_output.mean(dim=1)  # Mean pooling
        
        # Classification
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        
        return output

# Example usage
vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
num_classes = 5
max_seq_len = 512

model = TransformerClassifier(vocab_size, d_model, num_heads, num_layers, num_classes, max_seq_len)
batch_size = 32
seq_len = 100

x = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(x)

print(f"Classification output shape: {output.shape}")
```

### **2. Machine Translation**

```python
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super(EncoderDecoderTransformer, self).__init__()
        
        # Encoder
        self.encoder = Transformer(src_vocab_size, d_model, num_heads, num_layers, d_ff=2048, max_seq_len=max_seq_len)
        
        # Decoder
        self.decoder = Transformer(tgt_vocab_size, d_model, num_heads, num_layers, d_ff=2048, max_seq_len=max_seq_len)
        
        # Cross-attention (simplified)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
    def forward(self, src, tgt):
        # Encode source
        encoder_output = self.encoder(src)
        
        # Decode target
        decoder_output = self.decoder(tgt)
        
        # Cross-attention (simplified)
        output, _ = self.cross_attention(decoder_output, encoder_output)
        
        return output

# Example usage
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
max_seq_len = 512

model = EncoderDecoderTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, max_seq_len)
batch_size = 32
src_len = 100
tgt_len = 100

src = torch.randint(0, src_vocab_size, (batch_size, src_len))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
output = model(src, tgt)

print(f"Translation output shape: {output.shape}")
```

### **3. Question Answering**

```python
class QATransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super(QATransformer, self).__init__()
        
        self.transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff=2048, max_seq_len=max_seq_len)
        
        # Answer span prediction
        self.start_classifier = nn.Linear(d_model, 1)
        self.end_classifier = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # Get transformer output
        transformer_output = self.transformer(x)
        
        # Predict start and end positions
        start_logits = self.start_classifier(transformer_output).squeeze(-1)
        end_logits = self.end_classifier(transformer_output).squeeze(-1)
        
        return start_logits, end_logits

# Example usage
vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
max_seq_len = 512

model = QATransformer(vocab_size, d_model, num_heads, num_layers, max_seq_len)
batch_size = 32
seq_len = 100

x = torch.randint(0, vocab_size, (batch_size, seq_len))
start_logits, end_logits = model(x)

print(f"Start logits shape: {start_logits.shape}")
print(f"End logits shape: {end_logits.shape}")
```

---

## 🎯 **Interview Questions**

### **1. What is the key innovation of Transformers?**

**Answer:**
The key innovation is the **attention mechanism** that allows the model to:
- Process all positions in parallel (no sequential dependency)
- Capture long-range dependencies directly
- Learn which parts of the input to focus on
- Eliminate the need for recurrent or convolutional layers

### **2. How does self-attention work?**

**Answer:**
Self-attention allows each position to attend to all positions in the input sequence:
1. **Query (Q)**: What am I looking for?
2. **Key (K)**: What do I have to offer?
3. **Value (V)**: What is the actual content?
4. **Attention**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

### **3. Why is the scaling factor √d_k used in attention?**

**Answer:**
The scaling factor √d_k prevents the dot products from becoming too large:
- Large dot products lead to extreme softmax values
- This causes gradients to become very small
- Scaling by √d_k keeps the variance of attention scores stable
- Helps with training stability and convergence

### **4. What is the purpose of multi-head attention?**

**Answer:**
Multi-head attention allows the model to:
- Attend to different types of relationships simultaneously
- Learn different representation subspaces
- Capture various types of dependencies (syntactic, semantic, etc.)
- Improve model expressiveness and performance

### **5. How do Transformers handle positional information?**

**Answer:**
Transformers use **positional encoding** because they have no inherent notion of position:
- **Sinusoidal encoding**: Uses sine and cosine functions
- **Learned encoding**: Learnable position embeddings
- **Relative encoding**: Encode relative positions between tokens
- **Rotary encoding**: More recent approach with better performance

### **6. What are the advantages and disadvantages of Transformers?**

**Answer:**
**Advantages:**
- Parallel processing (faster training)
- Long-range dependencies
- State-of-the-art performance
- Flexible architecture

**Disadvantages:**
- High computational complexity O(n²)
- Large memory requirements
- Requires large datasets
- Less interpretable than RNNs

---

**🎉 Transformers revolutionized NLP and understanding them is essential for modern AI/ML engineering roles!**
