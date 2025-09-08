# 🧠 Long Short-Term Memory (LSTM) Networks

> **Deep dive into LSTM architecture, mechanisms, and applications**

## 🎯 **Learning Objectives**

- Master LSTM architecture and gating mechanisms
- Understand how LSTMs solve the vanishing gradient problem
- Implement LSTMs for various sequence modeling tasks
- Apply LSTMs to real-world problems
- Optimize LSTM performance and training

## 📚 **Table of Contents**

1. [LSTM Architecture](#lstm-architecture)
2. [Gating Mechanisms](#gating-mechanisms)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Implementation Examples](#implementation-examples)
5. [Applications](#applications)
6. [Interview Questions](#interview-questions)

---

## 🏗️ **LSTM Architecture**

### **Concept**

LSTMs are a type of RNN designed to solve the vanishing gradient problem by using gating mechanisms to control information flow.

### **Key Components**

1. **Cell State (C_t)**: Long-term memory that flows through the network
2. **Hidden State (h_t)**: Short-term memory used for predictions
3. **Forget Gate**: Decides what information to discard
4. **Input Gate**: Decides what new information to store
5. **Output Gate**: Decides what parts of the cell state to output

### **Architecture Diagram**

```
Input (x_t) ──┐
              ├─→ [Forget Gate] ──┐
              ├─→ [Input Gate] ───┤
              ├─→ [Candidate] ────┤
              └─→ [Output Gate] ──┘
                              │
Previous Hidden (h_{t-1}) ────┘
                              │
Previous Cell (C_{t-1}) ──────┘
                              │
                              ▼
                    [Cell State Update]
                              │
                              ▼
                    [Hidden State Update]
                              │
                              ▼
                    Output (h_t)
```

---

## 🚪 **Gating Mechanisms**

### **1. Forget Gate**

**Purpose**: Decides what information to discard from the cell state.

**Formula**: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`

**Code Example**:
```python
import torch
import torch.nn as nn

class ForgetGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ForgetGate, self).__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        forget_gate = self.sigmoid(self.linear(combined))
        return forget_gate

# Example usage
forget_gate = ForgetGate(input_size=10, hidden_size=20)
x = torch.randn(32, 10)
hidden = torch.randn(32, 20)
forget_output = forget_gate(x, hidden)
print(f"Forget gate output shape: {forget_output.shape}")
```

### **2. Input Gate**

**Purpose**: Decides what new information to store in the cell state.

**Formula**: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`

**Code Example**:
```python
class InputGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputGate, self).__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        input_gate = self.sigmoid(self.linear(combined))
        return input_gate

# Example usage
input_gate = InputGate(input_size=10, hidden_size=20)
input_output = input_gate(x, hidden)
print(f"Input gate output shape: {input_output.shape}")
```

### **3. Candidate Values**

**Purpose**: Creates new candidate values to be added to the cell state.

**Formula**: `C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)`

**Code Example**:
```python
class CandidateValues(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CandidateValues, self).__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        candidate = self.tanh(self.linear(combined))
        return candidate

# Example usage
candidate_layer = CandidateValues(input_size=10, hidden_size=20)
candidate_output = candidate_layer(x, hidden)
print(f"Candidate values shape: {candidate_output.shape}")
```

### **4. Output Gate**

**Purpose**: Decides what parts of the cell state to output as the hidden state.

**Formula**: `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`

**Code Example**:
```python
class OutputGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OutputGate, self).__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        output_gate = self.sigmoid(self.linear(combined))
        return output_gate

# Example usage
output_gate = OutputGate(input_size=10, hidden_size=20)
output_gate_output = output_gate(x, hidden)
print(f"Output gate shape: {output_gate_output.shape}")
```

---

## 🧮 **Mathematical Foundations**

### **Complete LSTM Equations**

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # All gates in one linear layer for efficiency
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
    def forward(self, x, hidden, cell):
        # Concatenate input and hidden state
        combined = torch.cat((x, hidden), dim=1)
        
        # Compute all gates at once
        gates_output = self.gates(combined)
        
        # Split into individual gates
        forget_gate = torch.sigmoid(gates_output[:, :self.hidden_size])
        input_gate = torch.sigmoid(gates_output[:, self.hidden_size:2*self.hidden_size])
        candidate_values = torch.tanh(gates_output[:, 2*self.hidden_size:3*self.hidden_size])
        output_gate = torch.sigmoid(gates_output[:, 3*self.hidden_size:])
        
        # Update cell state
        cell = forget_gate * cell + input_gate * candidate_values
        
        # Compute hidden state
        hidden = output_gate * torch.tanh(cell)
        
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

### **Golang Implementation**

```go
package main

import (
    "fmt"
    "math"
)

type LSTMCell struct {
    InputSize  int
    HiddenSize int
    
    // Weight matrices for all gates
    Wf [][]float64  // Forget gate weights
    Wi [][]float64  // Input gate weights
    Wc [][]float64  // Candidate values weights
    Wo [][]float64  // Output gate weights
    
    // Biases
    bf []float64    // Forget gate bias
    bi []float64    // Input gate bias
    bc []float64    // Candidate values bias
    bo []float64    // Output gate bias
}

func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
    lstm := &LSTMCell{
        InputSize:  inputSize,
        HiddenSize: hiddenSize,
    }
    
    // Initialize weight matrices
    lstm.Wf = make([][]float64, hiddenSize)
    lstm.Wi = make([][]float64, hiddenSize)
    lstm.Wc = make([][]float64, hiddenSize)
    lstm.Wo = make([][]float64, hiddenSize)
    
    for i := 0; i < hiddenSize; i++ {
        lstm.Wf[i] = make([]float64, inputSize+hiddenSize)
        lstm.Wi[i] = make([]float64, inputSize+hiddenSize)
        lstm.Wc[i] = make([]float64, inputSize+hiddenSize)
        lstm.Wo[i] = make([]float64, inputSize+hiddenSize)
    }
    
    // Initialize biases
    lstm.bf = make([]float64, hiddenSize)
    lstm.bi = make([]float64, hiddenSize)
    lstm.bc = make([]float64, hiddenSize)
    lstm.bo = make([]float64, hiddenSize)
    
    return lstm
}

func (lstm *LSTMCell) Forward(x, hidden, cell []float64) ([]float64, []float64) {
    // Concatenate input and hidden state
    combined := append(x, hidden...)
    
    // Compute forget gate
    forgetGate := make([]float64, lstm.HiddenSize)
    for i := 0; i < lstm.HiddenSize; i++ {
        sum := lstm.bf[i]
        for j := 0; j < len(combined); j++ {
            sum += lstm.Wf[i][j] * combined[j]
        }
        forgetGate[i] = sigmoid(sum)
    }
    
    // Compute input gate
    inputGate := make([]float64, lstm.HiddenSize)
    for i := 0; i < lstm.HiddenSize; i++ {
        sum := lstm.bi[i]
        for j := 0; j < len(combined); j++ {
            sum += lstm.Wi[i][j] * combined[j]
        }
        inputGate[i] = sigmoid(sum)
    }
    
    // Compute candidate values
    candidateValues := make([]float64, lstm.HiddenSize)
    for i := 0; i < lstm.HiddenSize; i++ {
        sum := lstm.bc[i]
        for j := 0; j < len(combined); j++ {
            sum += lstm.Wc[i][j] * combined[j]
        }
        candidateValues[i] = math.Tanh(sum)
    }
    
    // Update cell state
    newCell := make([]float64, lstm.HiddenSize)
    for i := 0; i < lstm.HiddenSize; i++ {
        newCell[i] = forgetGate[i]*cell[i] + inputGate[i]*candidateValues[i]
    }
    
    // Compute output gate
    outputGate := make([]float64, lstm.HiddenSize)
    for i := 0; i < lstm.HiddenSize; i++ {
        sum := lstm.bo[i]
        for j := 0; j < len(combined); j++ {
            sum += lstm.Wo[i][j] * combined[j]
        }
        outputGate[i] = sigmoid(sum)
    }
    
    // Compute new hidden state
    newHidden := make([]float64, lstm.HiddenSize)
    for i := 0; i < lstm.HiddenSize; i++ {
        newHidden[i] = outputGate[i] * math.Tanh(newCell[i])
    }
    
    return newHidden, newCell
}

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func main() {
    lstm := NewLSTMCell(10, 20)
    
    // Example forward pass
    x := make([]float64, 10)
    hidden := make([]float64, 20)
    cell := make([]float64, 20)
    
    newHidden, newCell := lstm.Forward(x, hidden, cell)
    
    fmt.Printf("New Hidden: %v\n", newHidden)
    fmt.Printf("New Cell: %v\n", newCell)
}
```

---

## 🎯 **Applications**

### **1. Sentiment Analysis**

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers=2):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = hidden[-1]
        
        # Classification
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output

# Example usage
vocab_size = 10000
embed_size = 128
hidden_size = 256
num_classes = 3  # Positive, Negative, Neutral
sequence_length = 100

model = SentimentLSTM(vocab_size, embed_size, hidden_size, num_classes)
batch_size = 32

# Input: batch of tokenized sequences
x = torch.randint(0, vocab_size, (batch_size, sequence_length))
output = model(x)

print(f"Sentiment output shape: {output.shape}")
```

### **2. Time Series Forecasting**

```python
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(TimeSeriesLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Prediction
        prediction = self.fc(last_output)
        
        return prediction

# Example usage
input_size = 5  # 5 features
hidden_size = 64
output_size = 1  # Predict 1 value
sequence_length = 50

model = TimeSeriesLSTM(input_size, hidden_size, output_size)
batch_size = 16

# Input: time series data
x = torch.randn(batch_size, sequence_length, input_size)
prediction = model(x)

print(f"Time series prediction shape: {prediction.shape}")
```

### **3. Machine Translation**

```python
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size):
        super(EncoderDecoderLSTM, self).__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_size)
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_vocab_size)
        
    def encode(self, x):
        embedded = self.encoder_embedding(x)
        _, (hidden, cell) = self.encoder_lstm(embedded)
        return hidden, cell
    
    def decode(self, x, hidden, cell):
        embedded = self.decoder_embedding(x)
        output, (hidden, cell) = self.decoder_lstm(embedded, (hidden, cell))
        output = self.output_layer(output)
        return output, hidden, cell

# Example usage
input_vocab_size = 5000
output_vocab_size = 5000
embed_size = 128
hidden_size = 256

model = EncoderDecoderLSTM(input_vocab_size, output_vocab_size, embed_size, hidden_size)
batch_size = 32
sequence_length = 50

# Encoder input
encoder_input = torch.randint(0, input_vocab_size, (batch_size, sequence_length))
hidden, cell = model.encode(encoder_input)

# Decoder input
decoder_input = torch.randint(0, output_vocab_size, (batch_size, 1))
output, _, _ = model.decode(decoder_input, hidden, cell)

print(f"Translation output shape: {output.shape}")
```

---

## 🎯 **Interview Questions**

### **1. How do LSTMs solve the vanishing gradient problem?**

**Answer:**
LSTMs solve the vanishing gradient problem through:
- **Cell State**: Maintains information across long sequences without multiplicative updates
- **Gating Mechanisms**: Control information flow to prevent gradient explosion/vanishing
- **Additive Updates**: Cell state updates are additive, allowing gradients to flow freely
- **Selective Memory**: Forget gate removes irrelevant information, input gate adds new information

### **2. What's the difference between LSTM and GRU?**

**Answer:**
- **LSTM**: 3 gates (forget, input, output) + cell state
- **GRU**: 2 gates (reset, update) + no cell state
- **Parameters**: GRU has fewer parameters (~25% less)
- **Performance**: Similar performance, GRU is faster to train
- **Memory**: LSTM has better memory for very long sequences
- **Complexity**: GRU is simpler and easier to understand

### **3. When would you use LSTM vs Transformer?**

**Answer:**
**Use LSTM when:**
- Sequential processing is important
- Limited computational resources
- Real-time processing required
- Short to medium sequences (< 100 tokens)
- Need to maintain state across time steps

**Use Transformer when:**
- Long-range dependencies are crucial
- Parallel processing is needed
- Large datasets available
- State-of-the-art performance required
- Can afford computational overhead

### **4. How do you handle variable-length sequences in LSTMs?**

**Answer:**
- **Padding**: Pad shorter sequences to match longest sequence
- **Masking**: Use attention masks to ignore padded tokens
- **Packing**: Use PyTorch's pack_padded_sequence for efficiency
- **Dynamic Batching**: Group sequences by length for better efficiency
- **Truncation**: Truncate very long sequences to fixed length

### **5. What are the computational complexity considerations for LSTMs?**

**Answer:**
- **Time Complexity**: O(n × h²) where n is sequence length, h is hidden size
- **Space Complexity**: O(n × h) for storing hidden states
- **Memory Usage**: High due to storing all hidden states for backpropagation
- **Parallelization**: Limited due to sequential nature
- **Optimization**: Use gradient clipping, batch normalization, and dropout

---

**🎉 LSTMs are powerful tools for sequence modeling and understanding them is essential for AI/ML engineering roles!**
