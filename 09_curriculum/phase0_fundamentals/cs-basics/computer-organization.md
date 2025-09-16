# Computer Organization

## Table of Contents

1. [Overview](#overview/)
2. [CPU Architecture](#cpu-architecture/)
3. [Memory Hierarchy](#memory-hierarchy/)
4. [Assembly Language](#assembly-language/)
5. [Performance Optimization](#performance-optimization/)
6. [Implementations](#implementations/)
7. [Follow-up Questions](#follow-up-questions/)
8. [Sources](#sources/)
9. [Projects](#projects/)

## Overview

### Learning Objectives

- Understand CPU architecture and instruction sets
- Master memory hierarchy and caching concepts
- Learn assembly language basics
- Apply performance optimization techniques
- Implement computer organization concepts in code

### What is Computer Organization?

Computer Organization involves understanding how computer hardware components work together to execute programs, including CPU design, memory systems, and performance optimization.

## CPU Architecture

### 1. Basic CPU Components

#### CPU Structure
```go
package main

import "fmt"

type CPU struct {
    Registers map[string]int
    ALU       *ArithmeticLogicUnit
    ControlUnit *ControlUnit
    PC        int // Program Counter
}

type ArithmeticLogicUnit struct {
    A, B, Result int
}

type ControlUnit struct {
    Instruction string
    OpCode      string
    Operands    []string
}

func NewCPU() *CPU {
    return &CPU{
        Registers: map[string]int{
            "R0": 0, "R1": 0, "R2": 0, "R3": 0,
            "R4": 0, "R5": 0, "R6": 0, "R7": 0,
        },
        ALU:        &ArithmeticLogicUnit{},
        ControlUnit: &ControlUnit{},
        PC:         0,
    }
}

func (cpu *CPU) ExecuteInstruction(instruction string) {
    fmt.Printf("Executing: %s\n", instruction)
    
    // Parse instruction
    parts := []string{instruction}
    if len(parts) >= 2 {
        cpu.ControlUnit.OpCode = parts[0]
        cpu.ControlUnit.Operands = parts[1:]
    }
    
    // Execute based on opcode
    switch cpu.ControlUnit.OpCode {
    case "ADD":
        cpu.ADD(parts[1], parts[2], parts[3])
    case "SUB":
        cpu.SUB(parts[1], parts[2], parts[3])
    case "MOV":
        cpu.MOV(parts[1], parts[2])
    case "JMP":
        cpu.JMP(parts[1])
    }
    
    cpu.PC++
}

func (cpu *CPU) ADD(dest, src1, src2 string) {
    cpu.ALU.A = cpu.Registers[src1]
    cpu.ALU.B = cpu.Registers[src2]
    cpu.ALU.Result = cpu.ALU.A + cpu.ALU.B
    cpu.Registers[dest] = cpu.ALU.Result
    fmt.Printf("  %s = %d + %d = %d\n", dest, cpu.ALU.A, cpu.ALU.B, cpu.ALU.Result)
}

func (cpu *CPU) SUB(dest, src1, src2 string) {
    cpu.ALU.A = cpu.Registers[src1]
    cpu.ALU.B = cpu.Registers[src2]
    cpu.ALU.Result = cpu.ALU.A - cpu.ALU.B
    cpu.Registers[dest] = cpu.ALU.Result
    fmt.Printf("  %s = %d - %d = %d\n", dest, cpu.ALU.A, cpu.ALU.B, cpu.ALU.Result)
}

func (cpu *CPU) MOV(dest, src string) {
    if val, ok := cpu.Registers[src]; ok {
        cpu.Registers[dest] = val
    } else {
        // Assume immediate value
        fmt.Sscanf(src, "%d", &cpu.Registers[dest])
    }
    fmt.Printf("  %s = %s\n", dest, src)
}

func (cpu *CPU) JMP(label string) {
    // Simplified - in real CPU, this would jump to label address
    fmt.Printf("  Jump to %s\n", label)
}

func (cpu *CPU) PrintRegisters() {
    fmt.Println("CPU Registers:")
    for reg, val := range cpu.Registers {
        fmt.Printf("  %s: %d\n", reg, val)
    }
    fmt.Printf("  PC: %d\n", cpu.PC)
}

func main() {
    cpu := NewCPU()
    
    // Load immediate values
    cpu.ExecuteInstruction("MOV R1 10")
    cpu.ExecuteInstruction("MOV R2 20")
    
    // Perform arithmetic
    cpu.ExecuteInstruction("ADD R3 R1 R2")
    cpu.ExecuteInstruction("SUB R4 R2 R1")
    
    cpu.PrintRegisters()
}
```

#### Node.js Implementation
```javascript
class CPU {
    constructor() {
        this.registers = {
            R0: 0, R1: 0, R2: 0, R3: 0,
            R4: 0, R5: 0, R6: 0, R7: 0
        };
        this.alu = new ArithmeticLogicUnit();
        this.controlUnit = new ControlUnit();
        this.pc = 0;
    }
    
    executeInstruction(instruction) {
        console.log(`Executing: ${instruction}`);
        
        const parts = instruction.split(' ');
        if (parts.length >= 2) {
            this.controlUnit.opCode = parts[0];
            this.controlUnit.operands = parts.slice(1);
        }
        
        switch (this.controlUnit.opCode) {
            case 'ADD':
                this.ADD(parts[1], parts[2], parts[3]);
                break;
            case 'SUB':
                this.SUB(parts[1], parts[2], parts[3]);
                break;
            case 'MOV':
                this.MOV(parts[1], parts[2]);
                break;
            case 'JMP':
                this.JMP(parts[1]);
                break;
        }
        
        this.pc++;
    }
    
    ADD(dest, src1, src2) {
        this.alu.A = this.registers[src1];
        this.alu.B = this.registers[src2];
        this.alu.result = this.alu.A + this.alu.B;
        this.registers[dest] = this.alu.result;
        console.log(`  ${dest} = ${this.alu.A} + ${this.alu.B} = ${this.alu.result}`);
    }
    
    SUB(dest, src1, src2) {
        this.alu.A = this.registers[src1];
        this.alu.B = this.registers[src2];
        this.alu.result = this.alu.A - this.alu.B;
        this.registers[dest] = this.alu.result;
        console.log(`  ${dest} = ${this.alu.A} - ${this.alu.B} = ${this.alu.result}`);
    }
    
    MOV(dest, src) {
        if (this.registers.hasOwnProperty(src)) {
            this.registers[dest] = this.registers[src];
        } else {
            this.registers[dest] = parseInt(src);
        }
        console.log(`  ${dest} = ${src}`);
    }
    
    JMP(label) {
        console.log(`  Jump to ${label}`);
    }
    
    printRegisters() {
        console.log('CPU Registers:');
        for (const [reg, val] of Object.entries(this.registers)) {
            console.log(`  ${reg}: ${val}`);
        }
        console.log(`  PC: ${this.pc}`);
    }
}

class ArithmeticLogicUnit {
    constructor() {
        this.A = 0;
        this.B = 0;
        this.result = 0;
    }
}

class ControlUnit {
    constructor() {
        this.instruction = '';
        this.opCode = '';
        this.operands = [];
    }
}

// Example usage
const cpu = new CPU();
cpu.executeInstruction('MOV R1 10');
cpu.executeInstruction('MOV R2 20');
cpu.executeInstruction('ADD R3 R1 R2');
cpu.executeInstruction('SUB R4 R2 R1');
cpu.printRegisters();
```

### 2. Instruction Set Architecture

#### RISC vs CISC
```go
package main

import "fmt"

type InstructionSet struct {
    Type        string
    Instructions map[string]Instruction
}

type Instruction struct {
    OpCode      string
    Operands    int
    Cycles      int
    Description string
}

func NewRISC() *InstructionSet {
    return &InstructionSet{
        Type: "RISC",
        Instructions: map[string]Instruction{
            "ADD": {"ADD", 3, 1, "Add two registers"},
            "SUB": {"SUB", 3, 1, "Subtract two registers"},
            "MUL": {"MUL", 3, 2, "Multiply two registers"},
            "DIV": {"DIV", 3, 4, "Divide two registers"},
            "LOAD": {"LOAD", 2, 2, "Load from memory"},
            "STORE": {"STORE", 2, 2, "Store to memory"},
            "JMP": {"JMP", 1, 1, "Unconditional jump"},
            "BEQ": {"BEQ", 3, 1, "Branch if equal"},
        },
    }
}

func NewCISC() *InstructionSet {
    return &InstructionSet{
        Type: "CISC",
        Instructions: map[string]Instruction{
            "ADD": {"ADD", 2, 1, "Add with addressing modes"},
            "SUB": {"SUB", 2, 1, "Subtract with addressing modes"},
            "MUL": {"MUL", 2, 3, "Multiply with addressing modes"},
            "DIV": {"DIV", 2, 6, "Divide with addressing modes"},
            "MOV": {"MOV", 2, 1, "Move with addressing modes"},
            "CALL": {"CALL", 1, 3, "Call subroutine"},
            "RET": {"RET", 0, 2, "Return from subroutine"},
            "PUSH": {"PUSH", 1, 1, "Push to stack"},
            "POP": {"POP", 1, 1, "Pop from stack"},
        },
    }
}

func (isa *InstructionSet) CompareWith(other *InstructionSet) {
    fmt.Printf("Comparing %s vs %s:\n", isa.Type, other.Type)
    fmt.Println("=====================================")
    
    fmt.Printf("\n%s Characteristics:\n", isa.Type)
    fmt.Printf("- Instruction Count: %d\n", len(isa.Instructions))
    fmt.Printf("- Average Cycles: %.1f\n", isa.averageCycles())
    fmt.Printf("- Addressing Modes: %s\n", isa.getAddressingModes())
    
    fmt.Printf("\n%s Characteristics:\n", other.Type)
    fmt.Printf("- Instruction Count: %d\n", len(other.Instructions))
    fmt.Printf("- Average Cycles: %.1f\n", other.averageCycles())
    fmt.Printf("- Addressing Modes: %s\n", other.getAddressingModes())
}

func (isa *InstructionSet) averageCycles() float64 {
    total := 0
    for _, inst := range isa.Instructions {
        total += inst.Cycles
    }
    return float64(total) / float64(len(isa.Instructions))
}

func (isa *InstructionSet) getAddressingModes() string {
    if isa.Type == "RISC" {
        return "Register, Immediate"
    }
    return "Register, Immediate, Direct, Indirect, Indexed"
}

func main() {
    risc := NewRISC()
    cisc := NewCISC()
    
    risc.CompareWith(cisc)
}
```

## Memory Hierarchy

### 1. Cache Implementation

#### Multi-Level Cache
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CacheLevel struct {
    Size       int
    BlockSize  int
    Associativity int
    HitTime    time.Duration
    MissTime   time.Duration
    Data       map[int]CacheBlock
    mutex      sync.RWMutex
}

type CacheBlock struct {
    Valid      bool
    Dirty      bool
    Tag        int
    Data       []byte
    LastAccess time.Time
}

type MemoryHierarchy struct {
    L1Cache    *CacheLevel
    L2Cache    *CacheLevel
    L3Cache    *CacheLevel
    MainMemory *MainMemory
    Stats      *CacheStats
}

type MainMemory struct {
    Data map[int]byte
    AccessTime time.Duration
}

type CacheStats struct {
    L1Hits, L1Misses int
    L2Hits, L2Misses int
    L3Hits, L3Misses int
    MemoryAccesses   int
}

func NewMemoryHierarchy() *MemoryHierarchy {
    return &MemoryHierarchy{
        L1Cache: &CacheLevel{
            Size:         32 * 1024,  // 32KB
            BlockSize:    64,         // 64 bytes
            Associativity: 4,         // 4-way set associative
            HitTime:      1 * time.Nanosecond,
            MissTime:     10 * time.Nanosecond,
            Data:         make(map[int]CacheBlock),
        },
        L2Cache: &CacheLevel{
            Size:         256 * 1024, // 256KB
            BlockSize:    64,
            Associativity: 8,
            HitTime:      10 * time.Nanosecond,
            MissTime:     50 * time.Nanosecond,
            Data:         make(map[int]CacheBlock),
        },
        L3Cache: &CacheLevel{
            Size:         8 * 1024 * 1024, // 8MB
            BlockSize:    64,
            Associativity: 16,
            HitTime:      50 * time.Nanosecond,
            MissTime:     200 * time.Nanosecond,
            Data:         make(map[int]CacheBlock),
        },
        MainMemory: &MainMemory{
            Data:       make(map[int]byte),
            AccessTime: 200 * time.Nanosecond,
        },
        Stats: &CacheStats{},
    }
}

func (mh *MemoryHierarchy) Read(address int) (byte, time.Duration) {
    start := time.Now()
    
    // Try L1 cache
    if data, hit := mh.readFromCache(mh.L1Cache, address); hit {
        mh.Stats.L1Hits++
        return data, time.Since(start)
    }
    mh.Stats.L1Misses++
    
    // Try L2 cache
    if data, hit := mh.readFromCache(mh.L2Cache, address); hit {
        mh.Stats.L2Hits++
        // Bring to L1
        mh.writeToCache(mh.L1Cache, address, data)
        return data, time.Since(start)
    }
    mh.Stats.L2Misses++
    
    // Try L3 cache
    if data, hit := mh.readFromCache(mh.L3Cache, address); hit {
        mh.Stats.L3Hits++
        // Bring to L2 and L1
        mh.writeToCache(mh.L2Cache, address, data)
        mh.writeToCache(mh.L1Cache, address, data)
        return data, time.Since(start)
    }
    mh.Stats.L3Misses++
    
    // Access main memory
    mh.Stats.MemoryAccesses++
    data := mh.MainMemory.Data[address]
    
    // Bring to all caches
    mh.writeToCache(mh.L3Cache, address, data)
    mh.writeToCache(mh.L2Cache, address, data)
    mh.writeToCache(mh.L1Cache, address, data)
    
    return data, time.Since(start)
}

func (mh *MemoryHierarchy) readFromCache(cache *CacheLevel, address int) (byte, bool) {
    cache.mutex.RLock()
    defer cache.mutex.RUnlock()
    
    blockIndex := address / cache.BlockSize
    if block, exists := cache.Data[blockIndex]; exists && block.Valid {
        block.LastAccess = time.Now()
        offset := address % cache.BlockSize
        if offset < len(block.Data) {
            return block.Data[offset], true
        }
    }
    return 0, false
}

func (mh *MemoryHierarchy) writeToCache(cache *CacheLevel, address int, data byte) {
    cache.mutex.Lock()
    defer cache.mutex.Unlock()
    
    blockIndex := address / cache.BlockSize
    offset := address % cache.BlockSize
    
    block := cache.Data[blockIndex]
    if !block.Valid {
        block = CacheBlock{
            Valid:      true,
            Dirty:      false,
            Tag:        address / cache.BlockSize,
            Data:       make([]byte, cache.BlockSize),
            LastAccess: time.Now(),
        }
    }
    
    if offset < len(block.Data) {
        block.Data[offset] = data
        block.LastAccess = time.Now()
        cache.Data[blockIndex] = block
    }
}

func (mh *MemoryHierarchy) PrintStats() {
    fmt.Println("Memory Hierarchy Statistics:")
    fmt.Println("===========================")
    fmt.Printf("L1 Cache - Hits: %d, Misses: %d, Hit Rate: %.2f%%\n", 
        mh.Stats.L1Hits, mh.Stats.L1Misses, 
        float64(mh.Stats.L1Hits)/float64(mh.Stats.L1Hits+mh.Stats.L1Misses)*100)
    fmt.Printf("L2 Cache - Hits: %d, Misses: %d, Hit Rate: %.2f%%\n", 
        mh.Stats.L2Hits, mh.Stats.L2Misses, 
        float64(mh.Stats.L2Hits)/float64(mh.Stats.L2Hits+mh.Stats.L2Misses)*100)
    fmt.Printf("L3 Cache - Hits: %d, Misses: %d, Hit Rate: %.2f%%\n", 
        mh.Stats.L3Hits, mh.Stats.L3Misses, 
        float64(mh.Stats.L3Hits)/float64(mh.Stats.L3Hits+mh.Stats.L3Misses)*100)
    fmt.Printf("Main Memory Accesses: %d\n", mh.Stats.MemoryAccesses)
}

func main() {
    mh := NewMemoryHierarchy()
    
    // Simulate memory accesses
    for i := 0; i < 1000; i++ {
        address := i % 10000 // Simulate some locality
        data, duration := mh.Read(address)
        fmt.Printf("Read address %d: %d (took %v)\n", address, data, duration)
    }
    
    mh.PrintStats()
}
```

## Assembly Language

### 1. Basic Assembly Instructions

#### Assembly Simulator
```go
package main

import (
    "fmt"
    "strconv"
    "strings"
)

type AssemblySimulator struct {
    Registers map[string]int
    Memory    map[int]int
    PC        int
    Labels    map[string]int
}

func NewAssemblySimulator() *AssemblySimulator {
    return &AssemblySimulator{
        Registers: map[string]int{
            "R0": 0, "R1": 0, "R2": 0, "R3": 0,
            "R4": 0, "R5": 0, "R6": 0, "R7": 0,
        },
        Memory: make(map[int]int),
        PC:     0,
        Labels: make(map[string]int),
    }
}

func (as *AssemblySimulator) LoadProgram(program []string) {
    // First pass: collect labels
    for i, line := range program {
        line = strings.TrimSpace(line)
        if strings.HasSuffix(line, ":") {
            label := strings.TrimSuffix(line, ":")
            as.Labels[label] = i
        }
    }
    
    // Second pass: execute instructions
    for as.PC < len(program) {
        line := strings.TrimSpace(program[as.PC])
        if line == "" || strings.HasSuffix(line, ":") {
            as.PC++
            continue
        }
        
        as.executeInstruction(line)
        as.PC++
    }
}

func (as *AssemblySimulator) executeInstruction(instruction string) {
    parts := strings.Fields(instruction)
    if len(parts) == 0 {
        return
    }
    
    opcode := parts[0]
    
    switch opcode {
    case "MOV":
        as.MOV(parts[1], parts[2])
    case "ADD":
        as.ADD(parts[1], parts[2], parts[3])
    case "SUB":
        as.SUB(parts[1], parts[2], parts[3])
    case "MUL":
        as.MUL(parts[1], parts[2], parts[3])
    case "DIV":
        as.DIV(parts[1], parts[2], parts[3])
    case "CMP":
        as.CMP(parts[1], parts[2])
    case "JMP":
        as.JMP(parts[1])
    case "JE":
        as.JE(parts[1])
    case "JNE":
        as.JNE(parts[1])
    case "LOAD":
        as.LOAD(parts[1], parts[2])
    case "STORE":
        as.STORE(parts[1], parts[2])
    case "PUSH":
        as.PUSH(parts[1])
    case "POP":
        as.POP(parts[1])
    case "CALL":
        as.CALL(parts[1])
    case "RET":
        as.RET()
    }
}

func (as *AssemblySimulator) MOV(dest, src string) {
    if val, ok := as.Registers[src]; ok {
        as.Registers[dest] = val
    } else {
        val, _ := strconv.Atoi(src)
        as.Registers[dest] = val
    }
    fmt.Printf("MOV %s, %s -> %s = %d\n", dest, src, dest, as.Registers[dest])
}

func (as *AssemblySimulator) ADD(dest, src1, src2 string) {
    val1 := as.getOperand(src1)
    val2 := as.getOperand(src2)
    as.Registers[dest] = val1 + val2
    fmt.Printf("ADD %s, %s, %s -> %s = %d\n", dest, src1, src2, dest, as.Registers[dest])
}

func (as *AssemblySimulator) SUB(dest, src1, src2 string) {
    val1 := as.getOperand(src1)
    val2 := as.getOperand(src2)
    as.Registers[dest] = val1 - val2
    fmt.Printf("SUB %s, %s, %s -> %s = %d\n", dest, src1, src2, dest, as.Registers[dest])
}

func (as *AssemblySimulator) MUL(dest, src1, src2 string) {
    val1 := as.getOperand(src1)
    val2 := as.getOperand(src2)
    as.Registers[dest] = val1 * val2
    fmt.Printf("MUL %s, %s, %s -> %s = %d\n", dest, src1, src2, dest, as.Registers[dest])
}

func (as *AssemblySimulator) DIV(dest, src1, src2 string) {
    val1 := as.getOperand(src1)
    val2 := as.getOperand(src2)
    if val2 != 0 {
        as.Registers[dest] = val1 / val2
    }
    fmt.Printf("DIV %s, %s, %s -> %s = %d\n", dest, src1, src2, dest, as.Registers[dest])
}

func (as *AssemblySimulator) CMP(op1, op2 string) {
    val1 := as.getOperand(op1)
    val2 := as.getOperand(op2)
    fmt.Printf("CMP %s, %s -> %d vs %d\n", op1, op2, val1, val2)
}

func (as *AssemblySimulator) JMP(label string) {
    if addr, ok := as.Labels[label]; ok {
        as.PC = addr
        fmt.Printf("JMP %s -> PC = %d\n", label, as.PC)
    }
}

func (as *AssemblySimulator) JE(label string) {
    // Simplified - assume comparison result is in a flag
    if addr, ok := as.Labels[label]; ok {
        as.PC = addr
        fmt.Printf("JE %s -> PC = %d\n", label, as.PC)
    }
}

func (as *AssemblySimulator) JNE(label string) {
    // Simplified - assume comparison result is in a flag
    if addr, ok := as.Labels[label]; ok {
        as.PC = addr
        fmt.Printf("JNE %s -> PC = %d\n", label, as.PC)
    }
}

func (as *AssemblySimulator) LOAD(reg, addr string) {
    address := as.getOperand(addr)
    as.Registers[reg] = as.Memory[address]
    fmt.Printf("LOAD %s, [%s] -> %s = %d\n", reg, addr, reg, as.Registers[reg])
}

func (as *AssemblySimulator) STORE(addr, reg string) {
    address := as.getOperand(addr)
    as.Memory[address] = as.Registers[reg]
    fmt.Printf("STORE [%s], %s -> [%d] = %d\n", addr, reg, address, as.Memory[address])
}

func (as *AssemblySimulator) PUSH(reg string) {
    // Simplified stack implementation
    as.Memory[as.Registers["R7"]] = as.Registers[reg]
    as.Registers["R7"]++
    fmt.Printf("PUSH %s\n", reg)
}

func (as *AssemblySimulator) POP(reg string) {
    as.Registers["R7"]--
    as.Registers[reg] = as.Memory[as.Registers["R7"]]
    fmt.Printf("POP %s\n", reg)
}

func (as *AssemblySimulator) CALL(label string) {
    as.PUSH("PC")
    as.JMP(label)
    fmt.Printf("CALL %s\n", label)
}

func (as *AssemblySimulator) RET() {
    as.POP("PC")
    fmt.Printf("RET\n")
}

func (as *AssemblySimulator) getOperand(operand string) int {
    if val, ok := as.Registers[operand]; ok {
        return val
    }
    val, _ := strconv.Atoi(operand)
    return val
}

func (as *AssemblySimulator) PrintState() {
    fmt.Println("\nAssembly Simulator State:")
    fmt.Println("========================")
    fmt.Println("Registers:")
    for reg, val := range as.Registers {
        fmt.Printf("  %s: %d\n", reg, val)
    }
    fmt.Printf("PC: %d\n", as.PC)
    fmt.Println("Memory:")
    for addr, val := range as.Memory {
        fmt.Printf("  [%d]: %d\n", addr, val)
    }
}

func main() {
    simulator := NewAssemblySimulator()
    
    program := []string{
        "MOV R1, 10",
        "MOV R2, 20",
        "ADD R3, R1, R2",
        "SUB R4, R2, R1",
        "MUL R5, R1, R2",
        "DIV R6, R2, R1",
        "STORE 100, R3",
        "LOAD R7, 100",
    }
    
    simulator.LoadProgram(program)
    simulator.PrintState()
}
```

## Performance Optimization

### 1. CPU Performance Analysis

#### Performance Profiler
```go
package main

import (
    "fmt"
    "runtime"
    "time"
)

type PerformanceProfiler struct {
    StartTime    time.Time
    EndTime      time.Time
    Instructions int
    Cycles       int
    CacheHits    int
    CacheMisses  int
    BranchHits   int
    BranchMisses int
}

func NewPerformanceProfiler() *PerformanceProfiler {
    return &PerformanceProfiler{
        StartTime: time.Now(),
    }
}

func (pp *PerformanceProfiler) Start() {
    pp.StartTime = time.Now()
    pp.Instructions = 0
    pp.Cycles = 0
    pp.CacheHits = 0
    pp.CacheMisses = 0
    pp.BranchHits = 0
    pp.BranchMisses = 0
}

func (pp *PerformanceProfiler) Stop() {
    pp.EndTime = time.Now()
}

func (pp *PerformanceProfiler) RecordInstruction(cycles int) {
    pp.Instructions++
    pp.Cycles += cycles
}

func (pp *PerformanceProfiler) RecordCacheAccess(hit bool) {
    if hit {
        pp.CacheHits++
    } else {
        pp.CacheMisses++
    }
}

func (pp *PerformanceProfiler) RecordBranch(hit bool) {
    if hit {
        pp.BranchHits++
    } else {
        pp.BranchMisses++
    }
}

func (pp *PerformanceProfiler) GetMetrics() map[string]float64 {
    duration := pp.EndTime.Sub(pp.StartTime)
    
    return map[string]float64{
        "ExecutionTime":    duration.Seconds(),
        "Instructions":     float64(pp.Instructions),
        "Cycles":           float64(pp.Cycles),
        "IPC":              float64(pp.Instructions) / float64(pp.Cycles),
        "CacheHitRate":     float64(pp.CacheHits) / float64(pp.CacheHits+pp.CacheMisses),
        "BranchHitRate":    float64(pp.BranchHits) / float64(pp.BranchHits+pp.BranchMisses),
        "InstructionsPerSecond": float64(pp.Instructions) / duration.Seconds(),
    }
}

func (pp *PerformanceProfiler) PrintReport() {
    metrics := pp.GetMetrics()
    
    fmt.Println("Performance Profiler Report:")
    fmt.Println("===========================")
    fmt.Printf("Execution Time: %.6f seconds\n", metrics["ExecutionTime"])
    fmt.Printf("Instructions: %.0f\n", metrics["Instructions"])
    fmt.Printf("Cycles: %.0f\n", metrics["Cycles"])
    fmt.Printf("IPC (Instructions Per Cycle): %.2f\n", metrics["IPC"])
    fmt.Printf("Cache Hit Rate: %.2f%%\n", metrics["CacheHitRate"]*100)
    fmt.Printf("Branch Hit Rate: %.2f%%\n", metrics["BranchHitRate"]*100)
    fmt.Printf("Instructions Per Second: %.0f\n", metrics["InstructionsPerSecond"])
}

func simulateWorkload(profiler *PerformanceProfiler) {
    // Simulate CPU-intensive workload
    for i := 0; i < 1000000; i++ {
        // Simulate instruction execution
        profiler.RecordInstruction(1)
        
        // Simulate cache access (90% hit rate)
        hit := i%10 != 0
        profiler.RecordCacheAccess(hit)
        
        // Simulate branch prediction (85% hit rate)
        branchHit := i%7 != 0
        profiler.RecordBranch(branchHit)
    }
}

func main() {
    profiler := NewPerformanceProfiler()
    
    profiler.Start()
    simulateWorkload(profiler)
    profiler.Stop()
    
    profiler.PrintReport()
    
    // Print system information
    fmt.Println("\nSystem Information:")
    fmt.Println("==================")
    fmt.Printf("CPU Cores: %d\n", runtime.NumCPU())
    fmt.Printf("Go Version: %s\n", runtime.Version())
    
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("Memory Allocated: %d KB\n", m.Alloc/1024)
    fmt.Printf("Total Allocations: %d\n", m.TotalAlloc)
    fmt.Printf("GC Cycles: %d\n", m.NumGC)
}
```

## Follow-up Questions

### 1. CPU Architecture
**Q: What's the difference between RISC and CISC architectures?**
A: RISC uses simple, fixed-length instructions with many registers, while CISC uses complex, variable-length instructions with fewer registers and more addressing modes.

### 2. Memory Hierarchy
**Q: Why do we need multiple levels of cache?**
A: Multiple cache levels provide a balance between speed and cost - L1 is fastest but smallest, L2/L3 are larger but slower, reducing the need for expensive main memory access.

### 3. Assembly Language
**Q: How does assembly language relate to high-level programming?**
A: Assembly is the human-readable form of machine code that compilers generate from high-level languages, providing direct control over hardware operations.

## Sources

### Books
- **Computer Systems: A Programmer's Perspective** by Bryant & O'Hallaron
- **Computer Organization and Design** by Patterson & Hennessy
- **Modern Processor Design** by Shen & Lipasti

### Online Resources
- **MIT 6.004**: Computation Structures
- **Coursera**: Computer Architecture courses
- **Intel Developer Zone**: CPU optimization guides

## Projects

### 1. CPU Simulator
**Objective**: Build a complete CPU simulator
**Requirements**: Instruction set, registers, memory, execution engine
**Deliverables**: Working CPU simulator with assembly language support

### 2. Cache Simulator
**Objective**: Implement a multi-level cache simulator
**Requirements**: Different cache levels, replacement policies, performance metrics
**Deliverables**: Cache simulator with performance analysis

### 3. Assembly Compiler
**Objective**: Create a simple assembly language compiler
**Requirements**: Lexical analysis, parsing, code generation
**Deliverables**: Assembly compiler with optimization features

---

**Next**: [Operating Systems](operating-systems-concepts.md/) | **Previous**: [CS Basics Overview](README.md/) | **Up**: [Phase 0](README.md/)
