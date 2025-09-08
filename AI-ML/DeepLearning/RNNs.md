# 🔄 Recurrent Neural Networks (RNNs)

> **Understanding RNNs, LSTMs, and GRUs for sequential data processing**

## 🎯 **Learning Objectives**

- Master RNN architectures and their applications
- Understand LSTM and GRU mechanisms
- Implement RNNs for sequence modeling
- Apply RNNs to real-world problems
- Optimize RNN performance and training

## 📚 **Table of Contents**

1. [RNN Fundamentals](#rnn-fundamentals)
2. [LSTM Networks](#lstm-networks)
3. [GRU Networks](#gru-networks)
4. [Implementation Examples](#implementation-examples)
5. [Applications](#applications)
6. [Interview Questions](#interview-questions)

---

## 🔄 **RNN Fundamentals**

### **Concept**

RNNs are designed to process sequential data by maintaining hidden states that carry information from previous time steps.

### **Math Behind**

- **Forward Pass**: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
- **Output**: `y_t = W_hy * h_t + b_y`
- **Backpropagation Through Time (BPTT)**: Gradients flow backward through time

### **Code Example**

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden):
        # x: (batch_size, input_size)
        # hidden: (batch_size, hidden_size)
        
        # Compute new hidden state
        hidden = self.tanh(self.W_xh(x) + self.W_hh(hidden))
        
        # Compute output
        output = self.W_hy(hidden)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Example usage
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
batch_size = 32
sequence_length = 50

# Initialize hidden state
hidden = rnn.init_hidden(batch_size)

# Process sequence
for t in range(sequence_length):
    x_t = torch.randn(batch_size, 10)  # Input at time t
    output, hidden = rnn(x_t, hidden)
    
print(f"Output shape: {output.shape}")
print(f"Hidden shape: {hidden.shape}")
```

### **Golang Implementation**

```go
package main

import (
    "fmt"
    "math"
)

type RNN struct {
    InputSize  int
    HiddenSize int
    OutputSize int
    
    // Weight matrices
    Wxh [][]float64  // Input to hidden
    Whh [][]float64  // Hidden to hidden
    Why [][]float64  // Hidden to output
    
    // Biases
    bh []float64     // Hidden bias
    by []float64     // Output bias
}

func NewRNN(inputSize, hiddenSize, outputSize int) *RNN {
    rnn := &RNN{
        InputSize:  inputSize,
        HiddenSize: hiddenSize,
        OutputSize: outputSize,
    }
    
    // Initialize weight matrices
    rnn.Wxh = make([][]float64, hiddenSize)
    rnn.Whh = make([][]float64, hiddenSize)
    rnn.Why = make([][]float64, outputSize)
    
    for i := range rnn.Wxh {
        rnn.Wxh[i] = make([]float64, inputSize)
        rnn.Whh[i] = make([]float64, hiddenSize)
    }
    
    for i := range rnn.Why {
        rnn.Why[i] = make([]float64, hiddenSize)
    }
    
    // Initialize biases
    rnn.bh = make([]float64, hiddenSize)
    rnn.by = make([]float64, outputSize)
    
    return rnn
}

func (rnn *RNN) Forward(x, hidden []float64) ([]float64, []float64) {
    // Compute new hidden state
    newHidden := make([]float64, rnn.HiddenSize)
    for i := 0; i < rnn.HiddenSize; i++ {
        sum := rnn.bh[i]
        
        // Input contribution
        for j := 0; j < rnn.InputSize; j++ {
            sum += rnn.Wxh[i][j] * x[j]
        }
        
        // Hidden contribution
        for j := 0; j < rnn.HiddenSize; j++ {
            sum += rnn.Whh[i][j] * hidden[j]
        }
        
        newHidden[i] = math.Tanh(sum)
    }
    
    // Compute output
    output := make([]float64, rnn.OutputSize)
    for i := 0; i < rnn.OutputSize; i++ {
        sum := rnn.by[i]
        for j := 0; j < rnn.HiddenSize; j++ {
            sum += rnn.Why[i][j] * newHidden[j]
        }
        output[i] = sum
    }
    
    return output, newHidden
}

func main() {
    rnn := NewRNN(10, 20, 5)
    
    // Example forward pass
    x := make([]float64, 10)
    hidden := make([]float64, 20)
    
    output, newHidden := rnn.Forward(x, hidden)
    
    fmt.Printf("Output: %v\n", output)
    fmt.Printf("New Hidden: %v\n", newHidden)
}
```

---

## 🧠 **LSTM Networks**

### **Concept**

LSTMs solve the vanishing gradient problem in RNNs by using gates to control information flow.

### **Math Behind**

- **Forget Gate**: `f_t = σ(W_f * [h_{t-1}, x_t] + b_f)`
- **Input Gate**: `i_t = σ(W_i * [h_{t-1}, x_t] + b_i)`
- **Candidate Values**: `C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)`
- **Cell State**: `C_t = f_t * C_{t-1} + i_t * C̃_t`
- **Output Gate**: `o_t = σ(W_o * [h_{t-1}, x_t] + b_o)`
- **Hidden State**: `h_t = o_t * tanh(C_t)`

### **Code Example**

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gate weights
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # Input gate
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate values
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # Output gate
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden, cell):
        # Concatenate input and hidden state
        combined = torch.cat((x, hidden), dim=1)
        
        # Compute gates
        forget_gate = self.sigmoid(self.W_f(combined))
        input_gate = self.sigmoid(self.W_i(combined))
        candidate_values = self.tanh(self.W_C(combined))
        output_gate = self.sigmoid(self.W_o(combined))
        
        # Update cell state
        cell = forget_gate * cell + input_gate * candidate_values
        
        # Compute hidden state
        hidden = output_gate * self.tanh(cell)
        
        return hidden, cell

# Example usage
lstm_cell = LSTMCell(input_size=10, hidden_size=20)
batch_size = 32

# Initialize states
hidden = torch.zeros(batch_size, 20)
cell = torch.zeros(batch_size, 20)

# Process input
x = torch.randn(batch_size, 10)
hidden, cell = lstm_cell(x, hidden, cell)

print(f"Hidden shape: {hidden.shape}")
print(f"Cell shape: {cell.shape}")
```

---

## 🔄 **GRU Networks**

### **Concept**

GRUs are a simplified version of LSTMs with fewer parameters but similar performance.

### **Math Behind**

- **Reset Gate**: `r_t = σ(W_r * [h_{t-1}, x_t] + b_r)`
- **Update Gate**: `z_t = σ(W_z * [h_{t-1}, x_t] + b_z)`
- **Candidate Hidden**: `h̃_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)`
- **Hidden State**: `h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t`

### **Code Example**

```python
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gate weights
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)  # Reset gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # Update gate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate hidden
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden):
        # Concatenate input and hidden state
        combined = torch.cat((x, hidden), dim=1)
        
        # Compute gates
        reset_gate = self.sigmoid(self.W_r(combined))
        update_gate = self.sigmoid(self.W_z(combined))
        
        # Compute candidate hidden state
        reset_hidden = reset_gate * hidden
        combined_reset = torch.cat((x, reset_hidden), dim=1)
        candidate_hidden = self.tanh(self.W_h(combined_reset))
        
        # Update hidden state
        hidden = (1 - update_gate) * hidden + update_gate * candidate_hidden
        
        return hidden

# Example usage
gru_cell = GRUCell(input_size=10, hidden_size=20)
batch_size = 32

# Initialize hidden state
hidden = torch.zeros(batch_size, 20)

# Process input
x = torch.randn(batch_size, 10)
hidden = gru_cell(x, hidden)

print(f"Hidden shape: {hidden.shape}")
```

---

## 🎯 **Applications**

### **1. Natural Language Processing**

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x: (batch_size, sequence_length)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        
        # Classification
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output

# Example usage
vocab_size = 10000
embed_size = 128
hidden_size = 256
num_classes = 5
sequence_length = 100

model = TextClassifier(vocab_size, embed_size, hidden_size, num_classes)
batch_size = 32

# Input: batch of tokenized sequences
x = torch.randint(0, vocab_size, (batch_size, sequence_length))
output = model(x)

print(f"Output shape: {output.shape}")
```

### **2. Time Series Prediction**

```python
import torch
import torch.nn as nn

class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(TimeSeriesPredictor, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Prediction
        prediction = self.fc(last_output)
        
        return prediction

# Example usage
input_size = 5  # 5 features
hidden_size = 64
output_size = 1  # Predict 1 value
sequence_length = 50

model = TimeSeriesPredictor(input_size, hidden_size, output_size)
batch_size = 16

# Input: time series data
x = torch.randn(batch_size, sequence_length, input_size)
prediction = model(x)

print(f"Prediction shape: {prediction.shape}")
```

---

## 🎯 **Interview Questions**

### **1. What is the vanishing gradient problem in RNNs?**

**Answer:**
The vanishing gradient problem occurs when gradients become exponentially small as they propagate backward through time, making it difficult to learn long-term dependencies.

**Solutions:**
- LSTM networks with gating mechanisms
- GRU networks with reset and update gates
- Gradient clipping
- Residual connections

### **2. How do LSTMs solve the vanishing gradient problem?**

**Answer:**
LSTMs solve the vanishing gradient problem through:
- **Cell State**: Maintains information across long sequences
- **Gates**: Control information flow (forget, input, output)
- **Additive Updates**: Cell state updates are additive, not multiplicative
- **Gradient Flow**: Gradients can flow through the cell state without vanishing

### **3. What's the difference between LSTM and GRU?**

**Answer:**
- **LSTM**: 3 gates (forget, input, output) + cell state
- **GRU**: 2 gates (reset, update) + no cell state
- **Parameters**: GRU has fewer parameters than LSTM
- **Performance**: Similar performance, GRU is faster to train
- **Memory**: LSTM has better memory for long sequences

### **4. When would you use RNNs vs Transformers?**

**Answer:**
**Use RNNs when:**
- Sequential processing is important
- Limited computational resources
- Real-time processing required
- Short to medium sequences

**Use Transformers when:**
- Long-range dependencies are crucial
- Parallel processing is needed
- Large datasets available
- State-of-the-art performance required

### **5. How do you handle variable-length sequences in RNNs?**

**Answer:**
- **Padding**: Pad shorter sequences to match longest sequence
- **Masking**: Use attention masks to ignore padded tokens
- **Packing**: Use PyTorch's pack_padded_sequence for efficiency
- **Dynamic Batching**: Group sequences by length for better efficiency

---

**🎉 RNNs are fundamental for sequential data processing and understanding them is crucial for AI/ML engineering roles!**
