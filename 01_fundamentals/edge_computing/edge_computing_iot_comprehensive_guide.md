# 🌐 Edge Computing & IoT Comprehensive Guide

## Table of Contents
1. [Edge Computing Fundamentals](#edge-computing-fundamentals/)
2. [IoT Device Management](#iot-device-management/)
3. [Edge Analytics](#edge-analytics/)
4. [Fog Computing](#fog-computing/)
5. [5G and Edge Networks](#5g-and-edge-networks/)
6. [Edge Security](#edge-security/)
7. [Go Implementation Examples](#go-implementation-examples/)
8. [Interview Questions](#interview-questions/)

## Edge Computing Fundamentals

### Edge Computing Architecture

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type EdgeNode struct {
    ID          string
    Location    Location
    Resources   Resources
    Services    map[string]*EdgeService
    mutex       sync.RWMutex
}

type Location struct {
    Latitude  float64
    Longitude float64
    Region    string
    Zone      string
}

type Resources struct {
    CPU    float64
    Memory int64
    Storage int64
    Network float64
}

type EdgeService struct {
    Name        string
    Version     string
    Endpoints   []string
    Resources   Resources
    Status      ServiceStatus
    mutex       sync.RWMutex
}

type ServiceStatus int

const (
    STOPPED ServiceStatus = iota
    STARTING
    RUNNING
    STOPPING
    ERROR
)

type EdgeOrchestrator struct {
    nodes    map[string]*EdgeNode
    services map[string]*EdgeService
    mutex    sync.RWMutex
}

func NewEdgeOrchestrator() *EdgeOrchestrator {
    return &EdgeOrchestrator{
        nodes:    make(map[string]*EdgeNode),
        services: make(map[string]*EdgeService),
    }
}

func (eo *EdgeOrchestrator) RegisterNode(node *EdgeNode) {
    eo.mutex.Lock()
    defer eo.mutex.Unlock()
    eo.nodes[node.ID] = node
}

func (eo *EdgeOrchestrator) DeployService(serviceName string, requirements Resources) (*EdgeNode, error) {
    eo.mutex.RLock()
    var bestNode *EdgeNode
    bestScore := 0.0
    
    for _, node := range eo.nodes {
        if eo.canDeployOnNode(node, requirements) {
            score := eo.calculateScore(node, requirements)
            if score > bestScore {
                bestScore = score
                bestNode = node
            }
        }
    }
    eo.mutex.RUnlock()
    
    if bestNode == nil {
        return nil, fmt.Errorf("no suitable node found")
    }
    
    // Deploy service on best node
    service := &EdgeService{
        Name:      serviceName,
        Version:   "1.0.0",
        Resources: requirements,
        Status:    STARTING,
    }
    
    bestNode.mutex.Lock()
    bestNode.Services[serviceName] = service
    bestNode.Resources.CPU -= requirements.CPU
    bestNode.Resources.Memory -= requirements.Memory
    bestNode.Resources.Storage -= requirements.Storage
    bestNode.mutex.Unlock()
    
    // Start service
    go eo.startService(bestNode, service)
    
    return bestNode, nil
}

func (eo *EdgeOrchestrator) canDeployOnNode(node *EdgeNode, requirements Resources) bool {
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    return node.Resources.CPU >= requirements.CPU &&
           node.Resources.Memory >= requirements.Memory &&
           node.Resources.Storage >= requirements.Storage
}

func (eo *EdgeOrchestrator) calculateScore(node *EdgeNode, requirements Resources) float64 {
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    // Calculate resource utilization score
    cpuUtilization := requirements.CPU / node.Resources.CPU
    memoryUtilization := float64(requirements.Memory) / float64(node.Resources.Memory)
    storageUtilization := float64(requirements.Storage) / float64(node.Resources.Storage)
    
    // Lower utilization is better
    return 1.0 - (cpuUtilization + memoryUtilization + storageUtilization) / 3.0
}

func (eo *EdgeOrchestrator) startService(node *EdgeNode, service *EdgeService) {
    // Simulate service startup
    time.Sleep(2 * time.Second)
    
    service.mutex.Lock()
    service.Status = RUNNING
    service.mutex.Unlock()
    
    fmt.Printf("Service %s started on node %s\n", service.Name, node.ID)
}

// Edge Load Balancer
type EdgeLoadBalancer struct {
    nodes    map[string]*EdgeNode
    strategy LoadBalancingStrategy
    mutex    sync.RWMutex
}

type LoadBalancingStrategy interface {
    SelectNode(nodes map[string]*EdgeNode, request *Request) *EdgeNode
}

type RoundRobinStrategy struct {
    current int
    mutex   sync.Mutex
}

func (rr *RoundRobinStrategy) SelectNode(nodes map[string]*EdgeNode, request *Request) *EdgeNode {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()
    
    nodeList := make([]*EdgeNode, 0, len(nodes))
    for _, node := range nodes {
        if node.isHealthy() {
            nodeList = append(nodeList, node)
        }
    }
    
    if len(nodeList) == 0 {
        return nil
    }
    
    selected := nodeList[rr.current%len(nodeList)]
    rr.current++
    return selected
}

type LeastConnectionsStrategy struct{}

func (lc *LeastConnectionsStrategy) SelectNode(nodes map[string]*EdgeNode, request *Request) *EdgeNode {
    var bestNode *EdgeNode
    minConnections := int64(^uint64(0) >> 1) // Max int64
    
    for _, node := range nodes {
        if !node.isHealthy() {
            continue
        }
        
        connections := node.getActiveConnections()
        if connections < minConnections {
            minConnections = connections
            bestNode = node
        }
    }
    
    return bestNode
}

func (node *EdgeNode) isHealthy() bool {
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    // Check if node has sufficient resources
    return node.Resources.CPU > 0.1 && node.Resources.Memory > 100*1024*1024 // 100MB
}

func (node *EdgeNode) getActiveConnections() int64 {
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    // Simplified: return number of running services
    count := int64(0)
    for _, service := range node.Services {
        if service.Status == RUNNING {
            count++
        }
    }
    return count
}
```

## IoT Device Management

### IoT Device Registry

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type IoTDevice struct {
    ID          string
    Name        string
    Type        DeviceType
    Location    Location
    Status      DeviceStatus
    Properties  map[string]interface{}
    LastSeen    time.Time
    mutex       sync.RWMutex
}

type DeviceType int

const (
    SENSOR DeviceType = iota
    ACTUATOR
    GATEWAY
    CAMERA
    SENSOR_ARRAY
)

type DeviceStatus int

const (
    OFFLINE DeviceStatus = iota
    ONLINE
    MAINTENANCE
    ERROR
)

type DeviceRegistry struct {
    devices map[string]*IoTDevice
    mutex   sync.RWMutex
}

type DeviceManager struct {
    registry *DeviceRegistry
    handlers map[string]DeviceHandler
    mutex    sync.RWMutex
}

type DeviceHandler interface {
    HandleData(device *IoTDevice, data []byte) error
    HandleCommand(device *IoTDevice, command string, params map[string]interface{}) error
}

func NewDeviceRegistry() *DeviceRegistry {
    return &DeviceRegistry{
        devices: make(map[string]*IoTDevice),
    }
}

func (dr *DeviceRegistry) RegisterDevice(device *IoTDevice) {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()
    dr.devices[device.ID] = device
}

func (dr *DeviceRegistry) GetDevice(id string) (*IoTDevice, bool) {
    dr.mutex.RLock()
    defer dr.mutex.RUnlock()
    device, exists := dr.devices[id]
    return device, exists
}

func (dr *DeviceRegistry) UpdateDeviceStatus(id string, status DeviceStatus) {
    dr.mutex.RLock()
    device, exists := dr.devices[id]
    dr.mutex.RUnlock()
    
    if exists {
        device.mutex.Lock()
        device.Status = status
        device.LastSeen = time.Now()
        device.mutex.Unlock()
    }
}

func (dr *DeviceRegistry) GetDevicesByType(deviceType DeviceType) []*IoTDevice {
    dr.mutex.RLock()
    defer dr.mutex.RUnlock()
    
    var devices []*IoTDevice
    for _, device := range dr.devices {
        if device.Type == deviceType {
            devices = append(devices, device)
        }
    }
    return devices
}

func (dr *DeviceRegistry) GetDevicesByLocation(location Location) []*IoTDevice {
    dr.mutex.RLock()
    defer dr.mutex.RUnlock()
    
    var devices []*IoTDevice
    for _, device := range dr.devices {
        if device.Location.Region == location.Region {
            devices = append(devices, device)
        }
    }
    return devices
}

func NewDeviceManager() *DeviceManager {
    return &DeviceManager{
        registry: NewDeviceRegistry(),
        handlers: make(map[string]DeviceHandler),
    }
}

func (dm *DeviceManager) RegisterHandler(deviceType DeviceType, handler DeviceHandler) {
    dm.mutex.Lock()
    defer dm.mutex.Unlock()
    dm.handlers[fmt.Sprintf("%d", deviceType)] = handler
}

func (dm *DeviceManager) ProcessDeviceData(deviceID string, data []byte) error {
    device, exists := dm.registry.GetDevice(deviceID)
    if !exists {
        return fmt.Errorf("device not found")
    }
    
    handler, exists := dm.handlers[fmt.Sprintf("%d", device.Type)]
    if !exists {
        return fmt.Errorf("no handler for device type")
    }
    
    return handler.HandleData(device, data)
}

func (dm *DeviceManager) SendCommand(deviceID string, command string, params map[string]interface{}) error {
    device, exists := dm.registry.GetDevice(deviceID)
    if !exists {
        return fmt.Errorf("device not found")
    }
    
    handler, exists := dm.handlers[fmt.Sprintf("%d", device.Type)]
    if !exists {
        return fmt.Errorf("no handler for device type")
    }
    
    return handler.HandleCommand(device, command, params)
}

// Sensor Handler
type SensorHandler struct{}

func (sh *SensorHandler) HandleData(device *IoTDevice, data []byte) error {
    device.mutex.Lock()
    device.LastSeen = time.Now()
    device.mutex.Unlock()
    
    // Process sensor data
    fmt.Printf("Processing sensor data from %s: %s\n", device.ID, string(data))
    
    // Update device properties
    device.mutex.Lock()
    device.Properties["last_reading"] = string(data)
    device.Properties["timestamp"] = time.Now().Unix()
    device.mutex.Unlock()
    
    return nil
}

func (sh *SensorHandler) HandleCommand(device *IoTDevice, command string, params map[string]interface{}) error {
    switch command {
    case "calibrate":
        return sh.calibrateSensor(device, params)
    case "reset":
        return sh.resetSensor(device)
    default:
        return fmt.Errorf("unknown command: %s", command)
    }
}

func (sh *SensorHandler) calibrateSensor(device *IoTDevice, params map[string]interface{}) error {
    fmt.Printf("Calibrating sensor %s with params: %+v\n", device.ID, params)
    return nil
}

func (sh *SensorHandler) resetSensor(device *IoTDevice) error {
    fmt.Printf("Resetting sensor %s\n", device.ID)
    return nil
}

// Actuator Handler
type ActuatorHandler struct{}

func (ah *ActuatorHandler) HandleData(device *IoTDevice, data []byte) error {
    // Actuators typically don't send data, they receive commands
    return nil
}

func (ah *ActuatorHandler) HandleCommand(device *IoTDevice, command string, params map[string]interface{}) error {
    switch command {
    case "turn_on":
        return ah.turnOn(device, params)
    case "turn_off":
        return ah.turnOff(device, params)
    case "set_value":
        return ah.setValue(device, params)
    default:
        return fmt.Errorf("unknown command: %s", command)
    }
}

func (ah *ActuatorHandler) turnOn(device *IoTDevice, params map[string]interface{}) error {
    fmt.Printf("Turning on actuator %s\n", device.ID)
    return nil
}

func (ah *ActuatorHandler) turnOff(device *IoTDevice, params map[string]interface{}) error {
    fmt.Printf("Turning off actuator %s\n", device.ID)
    return nil
}

func (ah *ActuatorHandler) setValue(device *IoTDevice, params map[string]interface{}) error {
    value, ok := params["value"].(float64)
    if !ok {
        return fmt.Errorf("invalid value parameter")
    }
    
    fmt.Printf("Setting actuator %s to value %f\n", device.ID, value)
    return nil
}
```

## Edge Analytics

### Real-time Data Processing

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type EdgeAnalytics struct {
    processors map[string]*DataProcessor
    streams    map[string]*DataStream
    mutex      sync.RWMutex
}

type DataProcessor struct {
    Name        string
    Input       chan DataPoint
    Output      chan ProcessedData
    Filters     []Filter
    Aggregators []Aggregator
    mutex       sync.RWMutex
}

type DataStream struct {
    Name     string
    Source   string
    DataType string
    mutex    sync.RWMutex
}

type DataPoint struct {
    Timestamp time.Time
    Value     float64
    Metadata  map[string]interface{}
    Source    string
}

type ProcessedData struct {
    Timestamp time.Time
    Value     float64
    Type      string
    Metadata  map[string]interface{}
}

type Filter interface {
    Filter(data DataPoint) bool
}

type Aggregator interface {
    Aggregate(data []DataPoint) ProcessedData
}

func NewEdgeAnalytics() *EdgeAnalytics {
    return &EdgeAnalytics{
        processors: make(map[string]*DataProcessor),
        streams:    make(map[string]*DataStream),
    }
}

func (ea *EdgeAnalytics) CreateDataStream(name, source, dataType string) *DataStream {
    stream := &DataStream{
        Name:     name,
        Source:   source,
        DataType: dataType,
    }
    
    ea.mutex.Lock()
    ea.streams[name] = stream
    ea.mutex.Unlock()
    
    return stream
}

func (ea *EdgeAnalytics) CreateProcessor(name string) *DataProcessor {
    processor := &DataProcessor{
        Name:        name,
        Input:       make(chan DataPoint, 1000),
        Output:      make(chan ProcessedData, 1000),
        Filters:     make([]Filter, 0),
        Aggregators: make([]Aggregator, 0),
    }
    
    ea.mutex.Lock()
    ea.processors[name] = processor
    ea.mutex.Unlock()
    
    go ea.runProcessor(processor)
    
    return processor
}

func (ea *EdgeAnalytics) runProcessor(processor *DataProcessor) {
    for dataPoint := range processor.Input {
        // Apply filters
        filtered := true
        for _, filter := range processor.Filters {
            if !filter.Filter(dataPoint) {
                filtered = false
                break
            }
        }
        
        if !filtered {
            continue
        }
        
        // Process data
        processed := ProcessedData{
            Timestamp: dataPoint.Timestamp,
            Value:     dataPoint.Value,
            Type:      "processed",
            Metadata:  dataPoint.Metadata,
        }
        
        // Apply aggregators if any
        if len(processor.Aggregators) > 0 {
            // In a real implementation, this would work on windows of data
            processed = processor.Aggregators[0].Aggregate([]DataPoint{dataPoint})
        }
        
        select {
        case processor.Output <- processed:
        default:
            // Output channel is full, drop data
            fmt.Printf("Dropped processed data from %s\n", processor.Name)
        }
    }
}

func (ea *EdgeAnalytics) ConnectStreamToProcessor(streamName, processorName string) error {
    ea.mutex.RLock()
    stream, streamExists := ea.streams[streamName]
    processor, processorExists := ea.processors[processorName]
    ea.mutex.RUnlock()
    
    if !streamExists || !processorExists {
        return fmt.Errorf("stream or processor not found")
    }
    
    // In a real implementation, this would establish the connection
    fmt.Printf("Connected stream %s to processor %s\n", streamName, processorName)
    return nil
}

// Example Filters
type RangeFilter struct {
    Min float64
    Max float64
}

func (rf *RangeFilter) Filter(data DataPoint) bool {
    return data.Value >= rf.Min && data.Value <= rf.Max
}

type AnomalyFilter struct {
    Threshold float64
    History   []DataPoint
    mutex     sync.RWMutex
}

func (af *AnomalyFilter) Filter(data DataPoint) bool {
    af.mutex.Lock()
    defer af.mutex.Unlock()
    
    if len(af.History) < 10 {
        af.History = append(af.History, data)
        return true
    }
    
    // Calculate moving average
    sum := 0.0
    for _, point := range af.History {
        sum += point.Value
    }
    avg := sum / float64(len(af.History))
    
    // Check if current value is within threshold
    if abs(data.Value-avg) > af.Threshold {
        return false // Anomaly detected
    }
    
    // Update history
    af.History = append(af.History[1:], data)
    return true
}

func abs(x float64) float64 {
    if x < 0 {
        return -x
    }
    return x
}

// Example Aggregators
type MovingAverageAggregator struct {
    WindowSize int
    Data       []DataPoint
    mutex      sync.RWMutex
}

func (maa *MovingAverageAggregator) Aggregate(data []DataPoint) ProcessedData {
    maa.mutex.Lock()
    defer maa.mutex.Unlock()
    
    maa.Data = append(maa.Data, data...)
    
    if len(maa.Data) > maa.WindowSize {
        maa.Data = maa.Data[len(maa.Data)-maa.WindowSize:]
    }
    
    if len(maa.Data) == 0 {
        return ProcessedData{}
    }
    
    sum := 0.0
    for _, point := range maa.Data {
        sum += point.Value
    }
    
    return ProcessedData{
        Timestamp: time.Now(),
        Value:     sum / float64(len(maa.Data)),
        Type:      "moving_average",
        Metadata:  map[string]interface{}{"window_size": len(maa.Data)},
    }
}

type MaxAggregator struct{}

func (ma *MaxAggregator) Aggregate(data []DataPoint) ProcessedData {
    if len(data) == 0 {
        return ProcessedData{}
    }
    
    max := data[0].Value
    for _, point := range data {
        if point.Value > max {
            max = point.Value
        }
    }
    
    return ProcessedData{
        Timestamp: time.Now(),
        Value:     max,
        Type:      "max",
        Metadata:  map[string]interface{}{"count": len(data)},
    }
}
```

## Interview Questions

### Basic Concepts
1. **What is edge computing and how does it differ from cloud computing?**
2. **What are the benefits and challenges of edge computing?**
3. **How do you manage IoT devices at scale?**
4. **What is the difference between edge and fog computing?**
5. **How do you ensure security in edge computing environments?**

### Advanced Topics
1. **How would you implement edge analytics for real-time data processing?**
2. **What are the challenges of deploying services at the edge?**
3. **How do you handle device failures in IoT systems?**
4. **What are the considerations for edge-to-cloud communication?**
5. **How do you implement edge load balancing?**

### System Design
1. **Design an edge computing platform.**
2. **How would you implement IoT device management?**
3. **Design a real-time analytics system for edge devices.**
4. **How would you implement edge security?**
5. **Design a fog computing architecture.**

## Conclusion

Edge computing and IoT represent the future of distributed computing, bringing processing closer to data sources. Key areas to master:

- **Edge Computing**: Distributed processing, resource management, service orchestration
- **IoT Systems**: Device management, data collection, real-time processing
- **Edge Analytics**: Real-time data processing, filtering, aggregation
- **Fog Computing**: Intermediate processing layer, edge-to-cloud coordination
- **5G Networks**: High-speed connectivity, low latency, massive device support
- **Edge Security**: Device authentication, data encryption, secure communication

Understanding these concepts helps in:
- Building edge computing systems
- Managing IoT devices
- Implementing real-time analytics
- Designing distributed architectures
- Preparing for technical interviews

This guide provides a comprehensive foundation for edge computing and IoT concepts and their practical implementation in Go.
