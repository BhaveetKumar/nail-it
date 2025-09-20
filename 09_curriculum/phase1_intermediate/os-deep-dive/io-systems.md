# I/O Systems

## Overview

This module covers I/O system concepts including device drivers, interrupt handling, I/O scheduling, buffering strategies, and I/O optimization. These concepts are essential for understanding how operating systems manage input/output operations efficiently.

## Table of Contents

1. [Device Drivers](#device-drivers)
2. [Interrupt Handling](#interrupt-handling)
3. [I/O Scheduling](#io-scheduling)
4. [Buffering Strategies](#buffering-strategies)
5. [I/O Optimization](#io-optimization)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Device Drivers

### Theory

Device drivers are software components that provide an interface between the operating system and hardware devices. They abstract hardware-specific details and provide a standardized interface for I/O operations.

### Device Driver Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type DeviceType int

const (
    BlockDevice DeviceType = iota
    CharacterDevice
    NetworkDevice
)

type Device struct {
    ID          int
    Name        string
    Type        DeviceType
    Status      string
    Buffer      []byte
    mutex       sync.Mutex
    InterruptChannel chan bool
}

type DeviceDriver struct {
    devices map[int]*Device
    mutex   sync.RWMutex
}

func NewDeviceDriver() *DeviceDriver {
    return &DeviceDriver{
        devices: make(map[int]*Device),
    }
}

func (dd *DeviceDriver) RegisterDevice(id int, name string, deviceType DeviceType) *Device {
    device := &Device{
        ID:               id,
        Name:             name,
        Type:             deviceType,
        Status:           "idle",
        Buffer:           make([]byte, 1024),
        InterruptChannel: make(chan bool, 1),
    }
    
    dd.mutex.Lock()
    dd.devices[id] = device
    dd.mutex.Unlock()
    
    fmt.Printf("Registered device %d: %s (type: %v)\n", id, name, deviceType)
    return device
}

func (dd *DeviceDriver) OpenDevice(id int) error {
    dd.mutex.RLock()
    device, exists := dd.devices[id]
    dd.mutex.RUnlock()
    
    if !exists {
        return fmt.Errorf("device %d not found", id)
    }
    
    device.mutex.Lock()
    defer device.mutex.Unlock()
    
    if device.Status != "idle" {
        return fmt.Errorf("device %d is not available", id)
    }
    
    device.Status = "open"
    fmt.Printf("Opened device %d: %s\n", id, device.Name)
    return nil
}

func (dd *DeviceDriver) CloseDevice(id int) error {
    dd.mutex.RLock()
    device, exists := dd.devices[id]
    dd.mutex.RUnlock()
    
    if !exists {
        return fmt.Errorf("device %d not found", id)
    }
    
    device.mutex.Lock()
    defer device.mutex.Unlock()
    
    device.Status = "idle"
    fmt.Printf("Closed device %d: %s\n", id, device.Name)
    return nil
}

func (dd *DeviceDriver) ReadDevice(id int, buffer []byte) (int, error) {
    dd.mutex.RLock()
    device, exists := dd.devices[id]
    dd.mutex.RUnlock()
    
    if !exists {
        return 0, fmt.Errorf("device %d not found", id)
    }
    
    device.mutex.Lock()
    defer device.mutex.Unlock()
    
    if device.Status != "open" {
        return 0, fmt.Errorf("device %d is not open", id)
    }
    
    // Simulate device read operation
    device.Status = "reading"
    fmt.Printf("Reading from device %d: %s\n", id, device.Name)
    
    // Simulate I/O delay
    time.Sleep(100 * time.Millisecond)
    
    // Copy data from device buffer
    bytesToRead := len(buffer)
    if bytesToRead > len(device.Buffer) {
        bytesToRead = len(device.Buffer)
    }
    
    copy(buffer, device.Buffer[:bytesToRead])
    
    device.Status = "open"
    fmt.Printf("Read %d bytes from device %d\n", bytesToRead, id)
    return bytesToRead, nil
}

func (dd *DeviceDriver) WriteDevice(id int, data []byte) (int, error) {
    dd.mutex.RLock()
    device, exists := dd.devices[id]
    dd.mutex.RUnlock()
    
    if !exists {
        return 0, fmt.Errorf("device %d not found", id)
    }
    
    device.mutex.Lock()
    defer device.mutex.Unlock()
    
    if device.Status != "open" {
        return 0, fmt.Errorf("device %d is not open", id)
    }
    
    // Simulate device write operation
    device.Status = "writing"
    fmt.Printf("Writing to device %d: %s\n", id, device.Name)
    
    // Simulate I/O delay
    time.Sleep(100 * time.Millisecond)
    
    // Copy data to device buffer
    bytesToWrite := len(data)
    if bytesToWrite > len(device.Buffer) {
        bytesToWrite = len(device.Buffer)
    }
    
    copy(device.Buffer, data[:bytesToWrite])
    
    device.Status = "open"
    fmt.Printf("Wrote %d bytes to device %d\n", bytesToWrite, id)
    return bytesToWrite, nil
}

func (dd *DeviceDriver) GetDeviceStatus(id int) (string, error) {
    dd.mutex.RLock()
    device, exists := dd.devices[id]
    dd.mutex.RUnlock()
    
    if !exists {
        return "", fmt.Errorf("device %d not found", id)
    }
    
    device.mutex.Lock()
    defer device.mutex.Unlock()
    
    return device.Status, nil
}

func (dd *DeviceDriver) ListDevices() {
    dd.mutex.RLock()
    defer dd.mutex.RUnlock()
    
    fmt.Println("Registered Devices:")
    fmt.Println("ID\tName\t\tType\t\tStatus")
    fmt.Println("--\t----\t\t----\t\t------")
    
    for _, device := range dd.devices {
        fmt.Printf("%d\t%s\t\t%v\t\t%s\n", 
                   device.ID, device.Name, device.Type, device.Status)
    }
}

func main() {
    driver := NewDeviceDriver()
    
    fmt.Println("Device Driver Demo:")
    
    // Register some devices
    driver.RegisterDevice(1, "hda", BlockDevice)
    driver.RegisterDevice(2, "tty1", CharacterDevice)
    driver.RegisterDevice(3, "eth0", NetworkDevice)
    
    driver.ListDevices()
    
    // Open and use a device
    err := driver.OpenDevice(1)
    if err != nil {
        fmt.Printf("Error opening device: %v\n", err)
        return
    }
    
    // Write to device
    data := []byte("Hello, Device!")
    bytesWritten, err := driver.WriteDevice(1, data)
    if err != nil {
        fmt.Printf("Error writing to device: %v\n", err)
    } else {
        fmt.Printf("Successfully wrote %d bytes\n", bytesWritten)
    }
    
    // Read from device
    buffer := make([]byte, 100)
    bytesRead, err := driver.ReadDevice(1, buffer)
    if err != nil {
        fmt.Printf("Error reading from device: %v\n", err)
    } else {
        fmt.Printf("Successfully read %d bytes: %s\n", bytesRead, string(buffer[:bytesRead]))
    }
    
    // Close device
    err = driver.CloseDevice(1)
    if err != nil {
        fmt.Printf("Error closing device: %v\n", err)
    }
}
```

## Interrupt Handling

### Theory

Interrupts are signals sent by hardware devices to the CPU to indicate that an event has occurred. The operating system must handle these interrupts efficiently to maintain system responsiveness.

### Interrupt Handler Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type InterruptType int

const (
    TimerInterrupt InterruptType = iota
    IOInterrupt
    KeyboardInterrupt
    NetworkInterrupt
)

type InterruptHandler struct {
    interruptQueue chan InterruptType
    handlers       map[InterruptType]func()
    mutex          sync.RWMutex
    running        bool
}

type InterruptManager struct {
    handler *InterruptHandler
    devices map[int]*Device
}

func NewInterruptHandler() *InterruptHandler {
    return &InterruptHandler{
        interruptQueue: make(chan InterruptType, 100),
        handlers:       make(map[InterruptType]func()),
        running:        false,
    }
}

func (ih *InterruptHandler) RegisterHandler(interruptType InterruptType, handler func()) {
    ih.mutex.Lock()
    defer ih.mutex.Unlock()
    
    ih.handlers[interruptType] = handler
    fmt.Printf("Registered handler for interrupt type %v\n", interruptType)
}

func (ih *InterruptHandler) TriggerInterrupt(interruptType InterruptType) {
    select {
    case ih.interruptQueue <- interruptType:
        fmt.Printf("Triggered interrupt: %v\n", interruptType)
    default:
        fmt.Printf("Interrupt queue full, dropping interrupt: %v\n", interruptType)
    }
}

func (ih *InterruptHandler) Start() {
    ih.mutex.Lock()
    ih.running = true
    ih.mutex.Unlock()
    
    fmt.Println("Starting interrupt handler...")
    
    go func() {
        for {
            select {
            case interruptType := <-ih.interruptQueue:
                ih.handleInterrupt(interruptType)
            case <-time.After(1 * time.Second):
                // Check if we should stop
                ih.mutex.RLock()
                running := ih.running
                ih.mutex.RUnlock()
                
                if !running {
                    return
                }
            }
        }
    }()
}

func (ih *InterruptHandler) Stop() {
    ih.mutex.Lock()
    ih.running = false
    ih.mutex.Unlock()
    
    fmt.Println("Stopping interrupt handler...")
}

func (ih *InterruptHandler) handleInterrupt(interruptType InterruptType) {
    ih.mutex.RLock()
    handler, exists := ih.handlers[interruptType]
    ih.mutex.RUnlock()
    
    if exists {
        fmt.Printf("Handling interrupt: %v\n", interruptType)
        handler()
    } else {
        fmt.Printf("No handler registered for interrupt: %v\n", interruptType)
    }
}

func NewInterruptManager() *InterruptManager {
    return &InterruptManager{
        handler: NewInterruptHandler(),
        devices: make(map[int]*Device),
    }
}

func (im *InterruptManager) RegisterDevice(device *Device) {
    im.devices[device.ID] = device
    fmt.Printf("Registered device %d with interrupt manager\n", device.ID)
}

func (im *InterruptManager) Start() {
    // Register interrupt handlers
    im.handler.RegisterHandler(TimerInterrupt, im.handleTimerInterrupt)
    im.handler.RegisterHandler(IOInterrupt, im.handleIOInterrupt)
    im.handler.RegisterHandler(KeyboardInterrupt, im.handleKeyboardInterrupt)
    im.handler.RegisterHandler(NetworkInterrupt, im.handleNetworkInterrupt)
    
    im.handler.Start()
}

func (im *InterruptManager) Stop() {
    im.handler.Stop()
}

func (im *InterruptManager) SimulateInterrupt(interruptType InterruptType) {
    im.handler.TriggerInterrupt(interruptType)
}

func (im *InterruptManager) handleTimerInterrupt() {
    fmt.Println("Timer interrupt: Updating system clock")
    // Simulate timer interrupt handling
    time.Sleep(10 * time.Millisecond)
}

func (im *InterruptManager) handleIOInterrupt() {
    fmt.Println("I/O interrupt: Processing I/O completion")
    // Simulate I/O interrupt handling
    time.Sleep(5 * time.Millisecond)
}

func (im *InterruptManager) handleKeyboardInterrupt() {
    fmt.Println("Keyboard interrupt: Processing key input")
    // Simulate keyboard interrupt handling
    time.Sleep(2 * time.Millisecond)
}

func (im *InterruptManager) handleNetworkInterrupt() {
    fmt.Println("Network interrupt: Processing network packet")
    // Simulate network interrupt handling
    time.Sleep(15 * time.Millisecond)
}

func main() {
    im := NewInterruptManager()
    
    fmt.Println("Interrupt Manager Demo:")
    
    // Start interrupt manager
    im.Start()
    
    // Simulate some interrupts
    time.Sleep(100 * time.Millisecond)
    im.SimulateInterrupt(TimerInterrupt)
    
    time.Sleep(100 * time.Millisecond)
    im.SimulateInterrupt(IOInterrupt)
    
    time.Sleep(100 * time.Millisecond)
    im.SimulateInterrupt(KeyboardInterrupt)
    
    time.Sleep(100 * time.Millisecond)
    im.SimulateInterrupt(NetworkInterrupt)
    
    // Let the system run for a bit
    time.Sleep(1 * time.Second)
    
    // Stop interrupt manager
    im.Stop()
}
```

## I/O Scheduling

### Theory

I/O scheduling determines the order in which I/O requests are processed. Different scheduling algorithms optimize for different criteria like throughput, response time, and fairness.

### I/O Scheduler Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
    "sync"
    "time"
)

type IORequest struct {
    ID        int
    DeviceID  int
    Operation string // "read" or "write"
    Address   int64
    Size      int
    Priority  int
    Timestamp time.Time
}

type IOScheduler struct {
    requests    []IORequest
    mutex       sync.Mutex
    nextID      int
    maxRequests int
}

func NewIOScheduler(maxRequests int) *IOScheduler {
    return &IOScheduler{
        requests:    make([]IORequest, 0),
        nextID:      1,
        maxRequests: maxRequests,
    }
}

func (ios *IOScheduler) AddRequest(deviceID int, operation string, address int64, size int, priority int) int {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    if len(ios.requests) >= ios.maxRequests {
        fmt.Println("I/O scheduler queue is full")
        return -1
    }
    
    request := IORequest{
        ID:        ios.nextID,
        DeviceID:  deviceID,
        Operation: operation,
        Address:   address,
        Size:      size,
        Priority:  priority,
        Timestamp: time.Now(),
    }
    
    ios.requests = append(ios.requests, request)
    ios.nextID++
    
    fmt.Printf("Added I/O request %d: %s %d bytes at address %d (priority: %d)\n", 
               request.ID, operation, size, address, priority)
    
    return request.ID
}

func (ios *IOScheduler) RemoveRequest(id int) bool {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    for i, request := range ios.requests {
        if request.ID == id {
            ios.requests = append(ios.requests[:i], ios.requests[i+1:]...)
            fmt.Printf("Removed I/O request %d\n", id)
            return true
        }
    }
    
    fmt.Printf("I/O request %d not found\n", id)
    return false
}

func (ios *IOScheduler) ScheduleFIFO() []IORequest {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    if len(ios.requests) == 0 {
        return []IORequest{}
    }
    
    // Sort by timestamp (FIFO)
    sortedRequests := make([]IORequest, len(ios.requests))
    copy(sortedRequests, ios.requests)
    
    sort.Slice(sortedRequests, func(i, j int) bool {
        return sortedRequests[i].Timestamp.Before(sortedRequests[j].Timestamp)
    })
    
    fmt.Println("FIFO I/O Schedule:")
    for _, request := range sortedRequests {
        fmt.Printf("  Request %d: %s %d bytes at address %d\n", 
                   request.ID, request.Operation, request.Size, request.Address)
    }
    
    return sortedRequests
}

func (ios *IOScheduler) SchedulePriority() []IORequest {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    if len(ios.requests) == 0 {
        return []IORequest{}
    }
    
    // Sort by priority (higher priority first)
    sortedRequests := make([]IORequest, len(ios.requests))
    copy(sortedRequests, ios.requests)
    
    sort.Slice(sortedRequests, func(i, j int) bool {
        if sortedRequests[i].Priority == sortedRequests[j].Priority {
            return sortedRequests[i].Timestamp.Before(sortedRequests[j].Timestamp)
        }
        return sortedRequests[i].Priority > sortedRequests[j].Priority
    })
    
    fmt.Println("Priority I/O Schedule:")
    for _, request := range sortedRequests {
        fmt.Printf("  Request %d: %s %d bytes at address %d (priority: %d)\n", 
                   request.ID, request.Operation, request.Size, request.Address, request.Priority)
    }
    
    return sortedRequests
}

func (ios *IOScheduler) ScheduleSCAN() []IORequest {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    if len(ios.requests) == 0 {
        return []IORequest{}
    }
    
    // Sort by address (SCAN algorithm)
    sortedRequests := make([]IORequest, len(ios.requests))
    copy(sortedRequests, ios.requests)
    
    sort.Slice(sortedRequests, func(i, j int) bool {
        return sortedRequests[i].Address < sortedRequests[j].Address
    })
    
    fmt.Println("SCAN I/O Schedule:")
    for _, request := range sortedRequests {
        fmt.Printf("  Request %d: %s %d bytes at address %d\n", 
                   request.ID, request.Operation, request.Size, request.Address)
    }
    
    return sortedRequests
}

func (ios *IOScheduler) ScheduleCSCAN() []IORequest {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    if len(ios.requests) == 0 {
        return []IORequest{}
    }
    
    // Sort by address (C-SCAN algorithm)
    sortedRequests := make([]IORequest, len(ios.requests))
    copy(sortedRequests, ios.requests)
    
    sort.Slice(sortedRequests, func(i, j int) bool {
        return sortedRequests[i].Address < sortedRequests[j].Address
    })
    
    fmt.Println("C-SCAN I/O Schedule:")
    for _, request := range sortedRequests {
        fmt.Printf("  Request %d: %s %d bytes at address %d\n", 
                   request.ID, request.Operation, request.Size, request.Address)
    }
    
    return sortedRequests
}

func (ios *IOScheduler) GetQueueLength() int {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    return len(ios.requests)
}

func (ios *IOScheduler) ClearQueue() {
    ios.mutex.Lock()
    defer ios.mutex.Unlock()
    
    ios.requests = ios.requests[:0]
    fmt.Println("Cleared I/O scheduler queue")
}

func main() {
    scheduler := NewIOScheduler(10)
    
    fmt.Println("I/O Scheduler Demo:")
    
    // Add some I/O requests
    scheduler.AddRequest(1, "read", 100, 512, 1)
    scheduler.AddRequest(2, "write", 200, 1024, 3)
    scheduler.AddRequest(3, "read", 50, 256, 2)
    scheduler.AddRequest(4, "write", 300, 2048, 1)
    scheduler.AddRequest(5, "read", 150, 768, 3)
    
    fmt.Printf("Queue length: %d\n", scheduler.GetQueueLength())
    
    // Test different scheduling algorithms
    scheduler.ScheduleFIFO()
    fmt.Println()
    
    scheduler.SchedulePriority()
    fmt.Println()
    
    scheduler.ScheduleSCAN()
    fmt.Println()
    
    scheduler.ScheduleCSCAN()
    fmt.Println()
    
    // Remove a request
    scheduler.RemoveRequest(2)
    fmt.Printf("Queue length after removal: %d\n", scheduler.GetQueueLength())
    
    // Clear queue
    scheduler.ClearQueue()
    fmt.Printf("Queue length after clear: %d\n", scheduler.GetQueueLength())
}
```

## Buffering Strategies

### Theory

Buffering strategies determine how data is temporarily stored in memory to improve I/O performance. Common strategies include single buffering, double buffering, and circular buffering.

### Buffer Manager Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Buffer struct {
    ID       int
    Data     []byte
    Size     int
    InUse    bool
    Dirty    bool
    LastUsed time.Time
    mutex    sync.Mutex
}

type BufferManager struct {
    buffers    map[int]*Buffer
    freeList   []int
    usedList   []int
    maxBuffers int
    bufferSize int
    mutex      sync.RWMutex
}

func NewBufferManager(maxBuffers, bufferSize int) *BufferManager {
    bm := &BufferManager{
        buffers:    make(map[int]*Buffer),
        freeList:   make([]int, 0),
        usedList:   make([]int, 0),
        maxBuffers: maxBuffers,
        bufferSize: bufferSize,
    }
    
    // Initialize buffers
    for i := 0; i < maxBuffers; i++ {
        buffer := &Buffer{
            ID:       i,
            Data:     make([]byte, bufferSize),
            Size:     0,
            InUse:    false,
            Dirty:    false,
            LastUsed: time.Now(),
        }
        
        bm.buffers[i] = buffer
        bm.freeList = append(bm.freeList, i)
    }
    
    return bm
}

func (bm *BufferManager) GetBuffer() (*Buffer, error) {
    bm.mutex.Lock()
    defer bm.mutex.Unlock()
    
    if len(bm.freeList) == 0 {
        // No free buffers, try to evict one
        if len(bm.usedList) == 0 {
            return nil, fmt.Errorf("no buffers available")
        }
        
        // Evict least recently used buffer
        bm.evictLRUBuffer()
    }
    
    // Get buffer from free list
    bufferID := bm.freeList[0]
    bm.freeList = bm.freeList[1:]
    
    buffer := bm.buffers[bufferID]
    buffer.InUse = true
    buffer.LastUsed = time.Now()
    
    bm.usedList = append(bm.usedList, bufferID)
    
    fmt.Printf("Allocated buffer %d\n", bufferID)
    return buffer, nil
}

func (bm *BufferManager) ReleaseBuffer(bufferID int) {
    bm.mutex.Lock()
    defer bm.mutex.Unlock()
    
    buffer, exists := bm.buffers[bufferID]
    if !exists {
        fmt.Printf("Buffer %d not found\n", bufferID)
        return
    }
    
    buffer.InUse = false
    buffer.LastUsed = time.Now()
    
    // Remove from used list
    for i, id := range bm.usedList {
        if id == bufferID {
            bm.usedList = append(bm.usedList[:i], bm.usedList[i+1:]...)
            break
        }
    }
    
    // Add to free list
    bm.freeList = append(bm.freeList, bufferID)
    
    fmt.Printf("Released buffer %d\n", bufferID)
}

func (bm *BufferManager) evictLRUBuffer() {
    if len(bm.usedList) == 0 {
        return
    }
    
    // Find least recently used buffer
    lruIndex := 0
    lruTime := bm.buffers[bm.usedList[0]].LastUsed
    
    for i, bufferID := range bm.usedList {
        if bm.buffers[bufferID].LastUsed.Before(lruTime) {
            lruTime = bm.buffers[bufferID].LastUsed
            lruIndex = i
        }
    }
    
    // Evict the LRU buffer
    bufferID := bm.usedList[lruIndex]
    buffer := bm.buffers[bufferID]
    
    if buffer.Dirty {
        // Write back to disk if dirty
        fmt.Printf("Writing back dirty buffer %d\n", bufferID)
        buffer.Dirty = false
    }
    
    buffer.InUse = false
    buffer.Size = 0
    
    // Remove from used list
    bm.usedList = append(bm.usedList[:lruIndex], bm.usedList[lruIndex+1:]...)
    
    // Add to free list
    bm.freeList = append(bm.freeList, bufferID)
    
    fmt.Printf("Evicted buffer %d\n", bufferID)
}

func (bm *BufferManager) WriteToBuffer(bufferID int, data []byte, offset int) error {
    bm.mutex.RLock()
    buffer, exists := bm.buffers[bufferID]
    bm.mutex.RUnlock()
    
    if !exists {
        return fmt.Errorf("buffer %d not found", bufferID)
    }
    
    buffer.mutex.Lock()
    defer buffer.mutex.Unlock()
    
    if !buffer.InUse {
        return fmt.Errorf("buffer %d is not in use", bufferID)
    }
    
    // Check bounds
    if offset+len(data) > len(buffer.Data) {
        return fmt.Errorf("write would exceed buffer size")
    }
    
    // Write data
    copy(buffer.Data[offset:], data)
    buffer.Size = offset + len(data)
    buffer.Dirty = true
    buffer.LastUsed = time.Now()
    
    fmt.Printf("Wrote %d bytes to buffer %d at offset %d\n", len(data), bufferID, offset)
    return nil
}

func (bm *BufferManager) ReadFromBuffer(bufferID int, data []byte, offset int) (int, error) {
    bm.mutex.RLock()
    buffer, exists := bm.buffers[bufferID]
    bm.mutex.RUnlock()
    
    if !exists {
        return 0, fmt.Errorf("buffer %d not found", bufferID)
    }
    
    buffer.mutex.Lock()
    defer buffer.mutex.Unlock()
    
    if !buffer.InUse {
        return 0, fmt.Errorf("buffer %d is not in use", bufferID)
    }
    
    // Check bounds
    if offset >= buffer.Size {
        return 0, fmt.Errorf("offset beyond buffer size")
    }
    
    // Calculate how much to read
    bytesToRead := len(data)
    if offset+bytesToRead > buffer.Size {
        bytesToRead = buffer.Size - offset
    }
    
    // Read data
    copy(data, buffer.Data[offset:offset+bytesToRead])
    buffer.LastUsed = time.Now()
    
    fmt.Printf("Read %d bytes from buffer %d at offset %d\n", bytesToRead, bufferID, offset)
    return bytesToRead, nil
}

func (bm *BufferManager) GetBufferStatus() {
    bm.mutex.RLock()
    defer bm.mutex.RUnlock()
    
    fmt.Printf("Buffer Manager Status:\n")
    fmt.Printf("  Total buffers: %d\n", bm.maxBuffers)
    fmt.Printf("  Free buffers: %d\n", len(bm.freeList))
    fmt.Printf("  Used buffers: %d\n", len(bm.usedList))
    
    fmt.Println("  Buffer details:")
    for _, buffer := range bm.buffers {
        status := "FREE"
        if buffer.InUse {
            status = "USED"
        }
        dirty := "CLEAN"
        if buffer.Dirty {
            dirty = "DIRTY"
        }
        
        fmt.Printf("    Buffer %d: %s, %s, size=%d\n", 
                   buffer.ID, status, dirty, buffer.Size)
    }
}

func main() {
    bm := NewBufferManager(5, 1024)
    
    fmt.Println("Buffer Manager Demo:")
    bm.GetBufferStatus()
    
    // Get some buffers
    buffer1, err := bm.GetBuffer()
    if err != nil {
        fmt.Printf("Error getting buffer: %v\n", err)
        return
    }
    
    buffer2, err := bm.GetBuffer()
    if err != nil {
        fmt.Printf("Error getting buffer: %v\n", err)
        return
    }
    
    bm.GetBufferStatus()
    
    // Write to buffers
    data1 := []byte("Hello, Buffer 1!")
    data2 := []byte("Hello, Buffer 2!")
    
    bm.WriteToBuffer(buffer1.ID, data1, 0)
    bm.WriteToBuffer(buffer2.ID, data2, 0)
    
    bm.GetBufferStatus()
    
    // Read from buffers
    readData1 := make([]byte, len(data1))
    readData2 := make([]byte, len(data2))
    
    bm.ReadFromBuffer(buffer1.ID, readData1, 0)
    bm.ReadFromBuffer(buffer2.ID, readData2, 0)
    
    fmt.Printf("Read from buffer 1: %s\n", string(readData1))
    fmt.Printf("Read from buffer 2: %s\n", string(readData2))
    
    // Release buffers
    bm.ReleaseBuffer(buffer1.ID)
    bm.ReleaseBuffer(buffer2.ID)
    
    bm.GetBufferStatus()
}
```

## Follow-up Questions

### 1. Device Drivers
**Q: What are the key responsibilities of a device driver?**
A: Device drivers are responsible for hardware abstraction, interrupt handling, device initialization, I/O operations, error handling, and providing a standardized interface between the operating system and hardware devices.

### 2. Interrupt Handling
**Q: How do you prioritize different types of interrupts?**
A: Interrupts are typically prioritized based on their criticality and system impact. Hardware interrupts (like power failure) have highest priority, followed by I/O interrupts, timer interrupts, and software interrupts. The interrupt controller manages this prioritization.

### 3. I/O Scheduling
**Q: When would you choose SCAN over FIFO for disk I/O scheduling?**
A: Choose SCAN when you want to minimize disk head movement and improve throughput for sequential access patterns. Choose FIFO when you need predictable response times and the workload doesn't have strong spatial locality.

## Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Device Registration | O(1) | O(1) | Hash table insertion |
| Interrupt Handling | O(1) | O(1) | Direct function call |
| I/O Scheduling | O(n log n) | O(n) | Sorting operations |
| Buffer Management | O(1) | O(n) | Hash table operations |

## Applications

1. **Device Drivers**: Operating systems, embedded systems
2. **Interrupt Handling**: Real-time systems, embedded systems
3. **I/O Scheduling**: Database systems, file systems
4. **Buffering**: Web servers, database systems, multimedia applications

---

**Next**: [Concurrency and Synchronization](concurrency-synchronization.md) | **Previous**: [OS Deep Dive](README.md) | **Up**: [OS Deep Dive](README.md)
