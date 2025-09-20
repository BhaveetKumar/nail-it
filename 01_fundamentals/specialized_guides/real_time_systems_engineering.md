# Real-Time Systems Engineering Guide

## Table of Contents
- [Introduction](#introduction)
- [Real-Time System Fundamentals](#real-time-system-fundamentals)
- [Low-Latency Architecture](#low-latency-architecture)
- [Event-Driven Systems](#event-driven-systems)
- [Stream Processing](#stream-processing)
- [High-Frequency Trading Systems](#high-frequency-trading-systems)
- [Gaming and Interactive Systems](#gaming-and-interactive-systems)
- [IoT and Edge Computing](#iot-and-edge-computing)
- [Real-Time Communication](#real-time-communication)
- [Performance Optimization](#performance-optimization)

## Introduction

Real-time systems engineering focuses on building systems that must respond to events within strict time constraints. This guide covers the essential concepts, architectures, and technologies needed to build high-performance real-time systems.

## Real-Time System Fundamentals

### Real-Time System Characteristics

```go
// Real-Time System Requirements
type RealTimeSystem struct {
    MaxLatency    time.Duration
    MinThroughput int64
    Reliability   float64
    Predictability bool
}

type RealTimeTask struct {
    ID           string
    Priority     int
    Deadline     time.Duration
    ExecutionTime time.Duration
    Period       time.Duration
    IsPeriodic   bool
}

// Real-Time Scheduler
type RealTimeScheduler struct {
    tasks        []*RealTimeTask
    currentTime  time.Time
    readyQueue   *PriorityQueue
    mu           sync.RWMutex
}

func (rts *RealTimeScheduler) Schedule() error {
    rts.mu.Lock()
    defer rts.mu.Unlock()
    
    // Rate Monotonic Scheduling
    sort.Slice(rts.tasks, func(i, j int) bool {
        return rts.tasks[i].Period < rts.tasks[j].Period
    })
    
    // Check schedulability
    if !rts.isSchedulable() {
        return fmt.Errorf("tasks are not schedulable")
    }
    
    // Execute tasks
    return rts.executeTasks()
}

func (rts *RealTimeScheduler) isSchedulable() bool {
    utilization := 0.0
    for _, task := range rts.tasks {
        utilization += float64(task.ExecutionTime) / float64(task.Period)
    }
    
    // Liu-Layland bound for RM scheduling
    n := len(rts.tasks)
    bound := float64(n) * (math.Pow(2, 1.0/float64(n)) - 1)
    
    return utilization <= bound
}
```

### Hard vs Soft Real-Time Systems

```go
// Hard Real-Time System (Safety-Critical)
type HardRealTimeSystem struct {
    maxResponseTime time.Duration
    failureMode     string
    redundancy      int
    watchdog        *Watchdog
}

type Watchdog struct {
    timeout    time.Duration
    lastPing   time.Time
    mu         sync.RWMutex
    resetCh    chan struct{}
}

func (w *Watchdog) Start() {
    go func() {
        ticker := time.NewTicker(w.timeout / 2)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                w.mu.RLock()
                if time.Since(w.lastPing) > w.timeout {
                    w.mu.RUnlock()
                    w.handleTimeout()
                    return
                }
                w.mu.RUnlock()
            case <-w.resetCh:
                w.mu.Lock()
                w.lastPing = time.Now()
                w.mu.Unlock()
            }
        }
    }()
}

// Soft Real-Time System (Best Effort)
type SoftRealTimeSystem struct {
    targetLatency   time.Duration
    acceptableLoss  float64
    adaptiveControl *AdaptiveController
}

type AdaptiveController struct {
    currentLatency time.Duration
    targetLatency  time.Duration
    controlParams  map[string]float64
    mu             sync.RWMutex
}

func (ac *AdaptiveController) AdjustParameters() {
    ac.mu.Lock()
    defer ac.mu.Unlock()
    
    // PID Controller for latency adjustment
    error := float64(ac.currentLatency - ac.targetLatency)
    
    // Proportional term
    p := ac.controlParams["kp"] * error
    
    // Integral term (simplified)
    i := ac.controlParams["ki"] * error
    
    // Derivative term (simplified)
    d := ac.controlParams["kd"] * error
    
    adjustment := p + i + d
    
    // Apply adjustment to system parameters
    ac.applyAdjustment(adjustment)
}
```

## Low-Latency Architecture

### Zero-Copy Networking

```go
// Zero-Copy Network Stack
type ZeroCopyNetwork struct {
    ringBuffer    *RingBuffer
    memoryPool    *MemoryPool
    eventLoop     *EventLoop
    connections   map[int]*Connection
    mu            sync.RWMutex
}

type RingBuffer struct {
    buffer    []byte
    head      int
    tail      int
    size      int
    capacity  int
    mu        sync.Mutex
}

func (rb *RingBuffer) Write(data []byte) (int, error) {
    rb.mu.Lock()
    defer rb.mu.Unlock()
    
    if rb.size+len(data) > rb.capacity {
        return 0, fmt.Errorf("ring buffer full")
    }
    
    written := 0
    for _, b := range data {
        rb.buffer[rb.tail] = b
        rb.tail = (rb.tail + 1) % rb.capacity
        rb.size++
        written++
    }
    
    return written, nil
}

func (rb *RingBuffer) Read(data []byte) (int, error) {
    rb.mu.Lock()
    defer rb.mu.Unlock()
    
    if rb.size == 0 {
        return 0, fmt.Errorf("ring buffer empty")
    }
    
    read := 0
    for i := 0; i < len(data) && rb.size > 0; i++ {
        data[i] = rb.buffer[rb.head]
        rb.head = (rb.head + 1) % rb.capacity
        rb.size--
        read++
    }
    
    return read, nil
}

// Memory Pool for efficient allocation
type MemoryPool struct {
    pools    map[int]*sync.Pool
    maxSize  int
    mu       sync.RWMutex
}

func NewMemoryPool(maxSize int) *MemoryPool {
    mp := &MemoryPool{
        pools:   make(map[int]*sync.Pool),
        maxSize: maxSize,
    }
    
    // Create pools for common sizes
    for size := 64; size <= maxSize; size *= 2 {
        size := size
        mp.pools[size] = &sync.Pool{
            New: func() interface{} {
                return make([]byte, size)
            },
        }
    }
    
    return mp
}

func (mp *MemoryPool) Get(size int) []byte {
    mp.mu.RLock()
    pool, exists := mp.pools[size]
    mp.mu.RUnlock()
    
    if exists {
        return pool.Get().([]byte)
    }
    
    // Fallback to direct allocation
    return make([]byte, size)
}

func (mp *MemoryPool) Put(buf []byte) {
    size := cap(buf)
    mp.mu.RLock()
    pool, exists := mp.pools[size]
    mp.mu.RUnlock()
    
    if exists {
        pool.Put(buf[:0]) // Reset length
    }
}
```

### Lock-Free Data Structures

```go
// Lock-Free Ring Buffer
type LockFreeRingBuffer struct {
    buffer   []interface{}
    capacity int
    head     int64
    tail     int64
}

func NewLockFreeRingBuffer(capacity int) *LockFreeRingBuffer {
    return &LockFreeRingBuffer{
        buffer:   make([]interface{}, capacity),
        capacity: capacity,
    }
}

func (lfrb *LockFreeRingBuffer) Enqueue(item interface{}) bool {
    currentTail := atomic.LoadInt64(&lfrb.tail)
    nextTail := (currentTail + 1) % int64(lfrb.capacity)
    
    // Check if buffer is full
    if nextTail == atomic.LoadInt64(&lfrb.head) {
        return false
    }
    
    lfrb.buffer[currentTail] = item
    atomic.StoreInt64(&lfrb.tail, nextTail)
    return true
}

func (lfrb *LockFreeRingBuffer) Dequeue() (interface{}, bool) {
    currentHead := atomic.LoadInt64(&lfrb.head)
    
    // Check if buffer is empty
    if currentHead == atomic.LoadInt64(&lfrb.tail) {
        return nil, false
    }
    
    item := lfrb.buffer[currentHead]
    nextHead := (currentHead + 1) % int64(lfrb.capacity)
    atomic.StoreInt64(&lfrb.head, nextHead)
    
    return item, true
}

// Lock-Free Hash Map
type LockFreeHashMap struct {
    buckets []*LockFreeBucket
    size    int
    mask    int
}

type LockFreeBucket struct {
    key   string
    value interface{}
    next  *LockFreeBucket
}

func NewLockFreeHashMap(size int) *LockFreeHashMap {
    // Ensure size is power of 2
    actualSize := 1
    for actualSize < size {
        actualSize <<= 1
    }
    
    return &LockFreeHashMap{
        buckets: make([]*LockFreeBucket, actualSize),
        size:    actualSize,
        mask:    actualSize - 1,
    }
}

func (lfhm *LockFreeHashMap) Get(key string) (interface{}, bool) {
    hash := lfhm.hash(key)
    bucket := lfhm.buckets[hash&lfhm.mask]
    
    for bucket != nil {
        if bucket.key == key {
            return bucket.value, true
        }
        bucket = bucket.next
    }
    
    return nil, false
}

func (lfhm *LockFreeHashMap) Put(key string, value interface{}) {
    hash := lfhm.hash(key)
    index := hash & lfhm.mask
    
    newBucket := &LockFreeBucket{
        key:   key,
        value: value,
        next:  lfhm.buckets[index],
    }
    
    lfhm.buckets[index] = newBucket
}

func (lfhm *LockFreeHashMap) hash(key string) int {
    h := 0
    for _, c := range key {
        h = h*31 + int(c)
    }
    return h
}
```

## Event-Driven Systems

### Event Sourcing

```go
// Event Sourcing Implementation
type EventStore struct {
    events    []*Event
    snapshots map[string]*Snapshot
    mu        sync.RWMutex
}

type Event struct {
    ID        string
    Type      string
    AggregateID string
    Data      interface{}
    Timestamp time.Time
    Version   int
}

type Snapshot struct {
    AggregateID string
    Data        interface{}
    Version     int
    Timestamp   time.Time
}

func (es *EventStore) AppendEvent(event *Event) error {
    es.mu.Lock()
    defer es.mu.Unlock()
    
    // Validate event version
    lastVersion := es.getLastVersion(event.AggregateID)
    if event.Version != lastVersion+1 {
        return fmt.Errorf("invalid event version")
    }
    
    es.events = append(es.events, event)
    return nil
}

func (es *EventStore) GetEvents(aggregateID string, fromVersion int) ([]*Event, error) {
    es.mu.RLock()
    defer es.mu.RUnlock()
    
    var events []*Event
    for _, event := range es.events {
        if event.AggregateID == aggregateID && event.Version >= fromVersion {
            events = append(events, event)
        }
    }
    
    return events, nil
}

func (es *EventStore) CreateSnapshot(aggregateID string, data interface{}, version int) {
    es.mu.Lock()
    defer es.mu.Unlock()
    
    es.snapshots[aggregateID] = &Snapshot{
        AggregateID: aggregateID,
        Data:        data,
        Version:     version,
        Timestamp:   time.Now(),
    }
}

// Event Handler
type EventHandler struct {
    handlers map[string][]func(*Event) error
    mu       sync.RWMutex
}

func (eh *EventHandler) RegisterHandler(eventType string, handler func(*Event) error) {
    eh.mu.Lock()
    defer eh.mu.Unlock()
    
    eh.handlers[eventType] = append(eh.handlers[eventType], handler)
}

func (eh *EventHandler) HandleEvent(event *Event) error {
    eh.mu.RLock()
    handlers := eh.handlers[event.Type]
    eh.mu.RUnlock()
    
    for _, handler := range handlers {
        if err := handler(event); err != nil {
            return err
        }
    }
    
    return nil
}
```

### CQRS (Command Query Responsibility Segregation)

```go
// CQRS Implementation
type CQRS struct {
    commandBus  *CommandBus
    queryBus    *QueryBus
    eventStore  *EventStore
    readModels  map[string]*ReadModel
    mu          sync.RWMutex
}

type Command interface {
    GetAggregateID() string
    GetType() string
}

type Query interface {
    GetType() string
}

type CommandHandler interface {
    Handle(command Command) error
}

type QueryHandler interface {
    Handle(query Query) (interface{}, error)
}

type CommandBus struct {
    handlers map[string]CommandHandler
    mu       sync.RWMutex
}

func (cb *CommandBus) RegisterHandler(commandType string, handler CommandHandler) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    cb.handlers[commandType] = handler
}

func (cb *CommandBus) Execute(command Command) error {
    cb.mu.RLock()
    handler := cb.handlers[command.GetType()]
    cb.mu.RUnlock()
    
    if handler == nil {
        return fmt.Errorf("no handler for command type: %s", command.GetType())
    }
    
    return handler.Handle(command)
}

type QueryBus struct {
    handlers map[string]QueryHandler
    mu       sync.RWMutex
}

func (qb *QueryBus) RegisterHandler(queryType string, handler QueryHandler) {
    qb.mu.Lock()
    defer qb.mu.Unlock()
    
    qb.handlers[queryType] = handler
}

func (qb *QueryBus) Execute(query Query) (interface{}, error) {
    qb.mu.RLock()
    handler := qb.handlers[query.GetType()]
    qb.mu.RUnlock()
    
    if handler == nil {
        return nil, fmt.Errorf("no handler for query type: %s", query.GetType())
    }
    
    return handler.Handle(query)
}

// Read Model for queries
type ReadModel struct {
    data    map[string]interface{}
    version int
    mu      sync.RWMutex
}

func (rm *ReadModel) Update(event *Event) {
    rm.mu.Lock()
    defer rm.mu.Unlock()
    
    // Update read model based on event
    switch event.Type {
    case "UserCreated":
        rm.data["user"] = event.Data
    case "UserUpdated":
        rm.data["user"] = event.Data
    }
    
    rm.version = event.Version
}

func (rm *ReadModel) GetData() map[string]interface{} {
    rm.mu.RLock()
    defer rm.mu.RUnlock()
    
    return rm.data
}
```

## Stream Processing

### Apache Kafka Integration

```go
// Kafka Stream Processor
type KafkaStreamProcessor struct {
    producer  *kafka.Producer
    consumer  *kafka.Consumer
    processor StreamProcessor
    config    *KafkaConfig
}

type StreamProcessor interface {
    Process(record *kafka.Record) (*kafka.Record, error)
    Filter(record *kafka.Record) bool
    Transform(record *kafka.Record) *kafka.Record
}

type KafkaConfig struct {
    BootstrapServers []string
    GroupID         string
    Topics          []string
    AutoOffsetReset string
}

func (ksp *KafkaStreamProcessor) Start() error {
    // Start consumer
    go ksp.consume()
    
    return nil
}

func (ksp *KafkaStreamProcessor) consume() {
    for {
        msg, err := ksp.consumer.ReadMessage(-1)
        if err != nil {
            log.Printf("Error reading message: %v", err)
            continue
        }
        
        // Process message
        processedMsg, err := ksp.processor.Process(msg)
        if err != nil {
            log.Printf("Error processing message: %v", err)
            continue
        }
        
        // Send processed message
        if processedMsg != nil {
            ksp.producer.Produce(processedMsg, nil)
        }
    }
}

// Real-time Analytics Processor
type RealTimeAnalyticsProcessor struct {
    aggregators map[string]*Aggregator
    windows     map[string]*TimeWindow
    mu          sync.RWMutex
}

type Aggregator struct {
    sum   float64
    count int64
    min   float64
    max   float64
    mu    sync.RWMutex
}

func (a *Aggregator) Add(value float64) {
    a.mu.Lock()
    defer a.mu.Unlock()
    
    a.sum += value
    a.count++
    
    if a.count == 1 {
        a.min = value
        a.max = value
    } else {
        if value < a.min {
            a.min = value
        }
        if value > a.max {
            a.max = value
        }
    }
}

func (a *Aggregator) GetStats() *Stats {
    a.mu.RLock()
    defer a.mu.RUnlock()
    
    avg := 0.0
    if a.count > 0 {
        avg = a.sum / float64(a.count)
    }
    
    return &Stats{
        Sum:   a.sum,
        Count: a.count,
        Avg:   avg,
        Min:   a.min,
        Max:   a.max,
    }
}

type TimeWindow struct {
    duration time.Duration
    buckets  map[int64]*Aggregator
    mu       sync.RWMutex
}

func (tw *TimeWindow) Add(timestamp time.Time, value float64) {
    bucket := timestamp.Unix() / int64(tw.duration.Seconds())
    
    tw.mu.Lock()
    if tw.buckets[bucket] == nil {
        tw.buckets[bucket] = &Aggregator{}
    }
    tw.mu.Unlock()
    
    tw.buckets[bucket].Add(value)
}

func (tw *TimeWindow) GetStats(timestamp time.Time) *Stats {
    bucket := timestamp.Unix() / int64(tw.duration.Seconds())
    
    tw.mu.RLock()
    aggregator := tw.buckets[bucket]
    tw.mu.RUnlock()
    
    if aggregator == nil {
        return &Stats{}
    }
    
    return aggregator.GetStats()
}
```

## High-Frequency Trading Systems

### Order Matching Engine

```go
// High-Frequency Trading Order Book
type HFTOrderBook struct {
    symbol    string
    bids      *PriceLevel
    asks      *PriceLevel
    orders    map[string]*Order
    trades    []*Trade
    mu        sync.RWMutex
}

type PriceLevel struct {
    price  decimal.Decimal
    orders []*Order
    volume decimal.Decimal
    next   *PriceLevel
    prev   *PriceLevel
}

type Order struct {
    ID        string
    Symbol    string
    Side      string // "buy" or "sell"
    Price     decimal.Decimal
    Quantity  decimal.Decimal
    Timestamp time.Time
    Status    string
}

type Trade struct {
    ID        string
    Symbol    string
    Price     decimal.Decimal
    Quantity  decimal.Decimal
    Timestamp time.Time
    BuyOrderID  string
    SellOrderID string
}

func (ob *HFTOrderBook) AddOrder(order *Order) ([]*Trade, error) {
    ob.mu.Lock()
    defer ob.mu.Unlock()
    
    ob.orders[order.ID] = order
    
    var trades []*Trade
    
    if order.Side == "buy" {
        trades = ob.matchBuyOrder(order)
    } else {
        trades = ob.matchSellOrder(order)
    }
    
    // Add remaining quantity to order book
    if order.Quantity.GreaterThan(decimal.Zero) {
        ob.addToOrderBook(order)
    }
    
    return trades, nil
}

func (ob *HFTOrderBook) matchBuyOrder(order *Order) []*Trade {
    var trades []*Trade
    
    for order.Quantity.GreaterThan(decimal.Zero) && ob.asks != nil {
        askLevel := ob.asks
        
        if order.Price.LessThan(askLevel.price) {
            break // No more matches possible
        }
        
        // Match with orders at this price level
        for i := 0; i < len(askLevel.orders) && order.Quantity.GreaterThan(decimal.Zero); i++ {
            askOrder := askLevel.orders[i]
            
            if askOrder.Quantity.LessThanOrEqual(order.Quantity) {
                // Full match
                trade := &Trade{
                    ID:          generateTradeID(),
                    Symbol:      order.Symbol,
                    Price:       askOrder.Price,
                    Quantity:    askOrder.Quantity,
                    Timestamp:   time.Now(),
                    BuyOrderID:  order.ID,
                    SellOrderID: askOrder.ID,
                }
                
                trades = append(trades, trade)
                
                order.Quantity = order.Quantity.Sub(askOrder.Quantity)
                askOrder.Quantity = decimal.Zero
                askOrder.Status = "filled"
                
                // Remove from order book
                ob.removeOrder(askOrder)
            } else {
                // Partial match
                trade := &Trade{
                    ID:          generateTradeID(),
                    Symbol:      order.Symbol,
                    Price:       askOrder.Price,
                    Quantity:    order.Quantity,
                    Timestamp:   time.Now(),
                    BuyOrderID:  order.ID,
                    SellOrderID: askOrder.ID,
                }
                
                trades = append(trades, trade)
                
                askOrder.Quantity = askOrder.Quantity.Sub(order.Quantity)
                order.Quantity = decimal.Zero
                order.Status = "filled"
            }
        }
        
        // Move to next price level
        ob.asks = askLevel.next
    }
    
    return trades
}

// Market Data Feed
type MarketDataFeed struct {
    symbol    string
    subscribers []chan *MarketData
    mu         sync.RWMutex
    running    bool
}

type MarketData struct {
    Symbol    string
    Price     decimal.Decimal
    Volume    decimal.Decimal
    Timestamp time.Time
    Source    string
}

func (mdf *MarketDataFeed) Subscribe() <-chan *MarketData {
    mdf.mu.Lock()
    defer mdf.mu.Unlock()
    
    ch := make(chan *MarketData, 1000)
    mdf.subscribers = append(mdf.subscribers, ch)
    
    return ch
}

func (mdf *MarketDataFeed) Publish(data *MarketData) {
    mdf.mu.RLock()
    subscribers := make([]chan *MarketData, len(mdf.subscribers))
    copy(subscribers, mdf.subscribers)
    mdf.mu.RUnlock()
    
    for _, ch := range subscribers {
        select {
        case ch <- data:
        default:
            // Channel full, skip
        }
    }
}
```

## Gaming and Interactive Systems

### Game Server Architecture

```go
// Game Server
type GameServer struct {
    rooms      map[string]*GameRoom
    players    map[string]*Player
    tickRate   time.Duration
    running    bool
    mu         sync.RWMutex
}

type GameRoom struct {
    ID       string
    Players  map[string]*Player
    State    *GameState
    TickRate time.Duration
    mu       sync.RWMutex
}

type Player struct {
    ID       string
    Position Vector3
    Rotation Vector3
    Health   int
    Score    int
    LastSeen time.Time
}

type GameState struct {
    Players    map[string]*Player
    GameObjects []*GameObject
    Timestamp  time.Time
}

type GameObject struct {
    ID       string
    Type     string
    Position Vector3
    Rotation Vector3
    Data     map[string]interface{}
}

type Vector3 struct {
    X float64
    Y float64
    Z float64
}

func (gs *GameServer) Start() {
    gs.running = true
    
    // Start game loop
    go gs.gameLoop()
    
    // Start player management
    go gs.playerManagement()
}

func (gs *GameServer) gameLoop() {
    ticker := time.NewTicker(gs.tickRate)
    defer ticker.Stop()
    
    for gs.running {
        select {
        case <-ticker.C:
            gs.updateGameState()
        }
    }
}

func (gs *GameServer) updateGameState() {
    gs.mu.RLock()
    rooms := make([]*GameRoom, 0, len(gs.rooms))
    for _, room := range gs.rooms {
        rooms = append(rooms, room)
    }
    gs.mu.RUnlock()
    
    for _, room := range rooms {
        room.updateState()
    }
}

func (gr *GameRoom) updateState() {
    gr.mu.Lock()
    defer gr.mu.Unlock()
    
    // Update game state
    gr.State.Timestamp = time.Now()
    
    // Process player actions
    for _, player := range gr.Players {
        gr.processPlayerActions(player)
    }
    
    // Update game objects
    gr.updateGameObjects()
    
    // Check win conditions
    gr.checkWinConditions()
}

// Real-time Physics Engine
type PhysicsEngine struct {
    objects    []*PhysicsObject
    gravity    Vector3
    timestep   float64
    iterations int
    mu         sync.RWMutex
}

type PhysicsObject struct {
    ID       string
    Position Vector3
    Velocity Vector3
    Mass     float64
    Radius   float64
    Type     string // "static", "dynamic"
}

func (pe *PhysicsEngine) Update(deltaTime float64) {
    pe.mu.Lock()
    defer pe.mu.Unlock()
    
    // Update dynamic objects
    for _, obj := range pe.objects {
        if obj.Type == "dynamic" {
            pe.updateObject(obj, deltaTime)
        }
    }
    
    // Check collisions
    pe.checkCollisions()
}

func (pe *PhysicsEngine) updateObject(obj *PhysicsObject, deltaTime float64) {
    // Apply gravity
    obj.Velocity = obj.Velocity.Add(pe.gravity.Multiply(deltaTime))
    
    // Update position
    obj.Position = obj.Position.Add(obj.Velocity.Multiply(deltaTime))
}

func (pe *PhysicsEngine) checkCollisions() {
    for i := 0; i < len(pe.objects); i++ {
        for j := i + 1; j < len(pe.objects); j++ {
            obj1 := pe.objects[i]
            obj2 := pe.objects[j]
            
            if pe.isColliding(obj1, obj2) {
                pe.resolveCollision(obj1, obj2)
            }
        }
    }
}
```

## IoT and Edge Computing

### Edge Computing Node

```go
// Edge Computing Node
type EdgeNode struct {
    ID          string
    Location    *Location
    Resources   *Resources
    Tasks       []*Task
    Neighbors   []*EdgeNode
    mu          sync.RWMutex
}

type Location struct {
    Latitude  float64
    Longitude float64
    Altitude  float64
}

type Resources struct {
    CPU    float64
    Memory float64
    Storage float64
    Network float64
}

type Task struct {
    ID          string
    Type        string
    Priority    int
    Requirements *Resources
    Data        []byte
    Result      []byte
    Status      string
    Deadline    time.Time
}

func (en *EdgeNode) ProcessTask(task *Task) error {
    en.mu.Lock()
    defer en.mu.Unlock()
    
    // Check if node has sufficient resources
    if !en.hasResources(task.Requirements) {
        return fmt.Errorf("insufficient resources")
    }
    
    // Allocate resources
    en.allocateResources(task.Requirements)
    
    // Process task
    result, err := en.executeTask(task)
    if err != nil {
        en.deallocateResources(task.Requirements)
        return err
    }
    
    task.Result = result
    task.Status = "completed"
    
    // Deallocate resources
    en.deallocateResources(task.Requirements)
    
    return nil
}

func (en *EdgeNode) hasResources(requirements *Resources) bool {
    return en.Resources.CPU >= requirements.CPU &&
           en.Resources.Memory >= requirements.Memory &&
           en.Resources.Storage >= requirements.Storage &&
           en.Resources.Network >= requirements.Network
}

// IoT Device Manager
type IoTDeviceManager struct {
    devices    map[string]*IoTDevice
    gateway    *Gateway
    cloud      *CloudService
    mu         sync.RWMutex
}

type IoTDevice struct {
    ID          string
    Type        string
    Location    *Location
    Sensors     []*Sensor
    Actuators   []*Actuator
    LastSeen    time.Time
    Status      string
}

type Sensor struct {
    ID       string
    Type     string
    Value    float64
    Unit     string
    Timestamp time.Time
}

type Actuator struct {
    ID    string
    Type  string
    State string
    Value interface{}
}

func (idm *IoTDeviceManager) ProcessSensorData(deviceID string, sensorData []*SensorData) error {
    idm.mu.RLock()
    device := idm.devices[deviceID]
    idm.mu.RUnlock()
    
    if device == nil {
        return fmt.Errorf("device not found")
    }
    
    // Process sensor data
    for _, data := range sensorData {
        // Update sensor value
        device.updateSensor(data)
        
        // Check for alerts
        if idm.checkAlert(device, data) {
            idm.sendAlert(device, data)
        }
        
        // Send to cloud if needed
        if idm.shouldSendToCloud(data) {
            idm.cloud.SendData(deviceID, data)
        }
    }
    
    device.LastSeen = time.Now()
    
    return nil
}
```

## Real-Time Communication

### WebSocket Server

```go
// WebSocket Server
type WebSocketServer struct {
    clients    map[*websocket.Conn]*Client
    rooms      map[string]*Room
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
    mu         sync.RWMutex
}

type Client struct {
    conn   *websocket.Conn
    send   chan []byte
    ID     string
    RoomID string
}

type Room struct {
    ID      string
    Clients map[*websocket.Conn]*Client
    mu      sync.RWMutex
}

func (wss *WebSocketServer) Start() {
    for {
        select {
        case client := <-wss.register:
            wss.registerClient(client)
            
        case client := <-wss.unregister:
            wss.unregisterClient(client)
            
        case message := <-wss.broadcast:
            wss.broadcastMessage(message)
        }
    }
}

func (wss *WebSocketServer) registerClient(client *Client) {
    wss.mu.Lock()
    defer wss.mu.Unlock()
    
    wss.clients[client.conn] = client
    
    // Add to room if specified
    if client.RoomID != "" {
        if room, exists := wss.rooms[client.RoomID]; exists {
            room.mu.Lock()
            room.Clients[client.conn] = client
            room.mu.Unlock()
        }
    }
}

func (wss *WebSocketServer) broadcastMessage(message []byte) {
    wss.mu.RLock()
    defer wss.mu.RUnlock()
    
    for client := range wss.clients {
        select {
        case client.send <- message:
        default:
            close(client.send)
            delete(wss.clients, client)
        }
    }
}

// Real-time Chat System
type ChatSystem struct {
    rooms      map[string]*ChatRoom
    users      map[string]*User
    messageQueue chan *Message
    mu         sync.RWMutex
}

type ChatRoom struct {
    ID       string
    Name     string
    Users    map[string]*User
    Messages []*Message
    mu       sync.RWMutex
}

type User struct {
    ID       string
    Username string
    conn     *websocket.Conn
    send     chan *Message
}

type Message struct {
    ID        string
    RoomID    string
    UserID    string
    Content   string
    Timestamp time.Time
    Type      string // "text", "image", "file"
}

func (cs *ChatSystem) SendMessage(message *Message) error {
    cs.mu.RLock()
    room := cs.rooms[message.RoomID]
    cs.mu.RUnlock()
    
    if room == nil {
        return fmt.Errorf("room not found")
    }
    
    // Add message to room
    room.mu.Lock()
    room.Messages = append(room.Messages, message)
    room.mu.Unlock()
    
    // Broadcast to all users in room
    room.mu.RLock()
    for _, user := range room.Users {
        select {
        case user.send <- message:
        default:
            // User channel full, skip
        }
    }
    room.mu.RUnlock()
    
    return nil
}
```

## Performance Optimization

### CPU Optimization

```go
// CPU Optimization Techniques
type CPUOptimizer struct {
    affinity    []int
    governor    string
    frequency   int
    cacheSize   int
    numCores    int
}

func (co *CPUOptimizer) SetCPUAffinity(pid int, cores []int) error {
    // Set CPU affinity for process
    var cpuset unix.CPUSet
    for _, core := range cores {
        cpuset.Set(core)
    }
    
    return unix.SchedSetaffinity(pid, &cpuset)
}

func (co *CPUOptimizer) SetCPUFrequency(frequency int) error {
    // Set CPU frequency governor
    return ioutil.WriteFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", 
                           []byte("performance"), 0644)
}

// Memory Optimization
type MemoryOptimizer struct {
    pageSize    int
    hugePages   bool
    numaNodes   []int
    memoryPool  *MemoryPool
}

func (mo *MemoryOptimizer) EnableHugePages() error {
    // Enable huge pages
    return ioutil.WriteFile("/proc/sys/vm/nr_hugepages", 
                           []byte("1024"), 0644)
}

func (mo *MemoryOptimizer) SetNUMAAffinity(node int) error {
    // Set NUMA node affinity
    return ioutil.WriteFile("/proc/self/numa_maps", 
                           []byte(fmt.Sprintf("bind=%d", node)), 0644)
}

// Network Optimization
type NetworkOptimizer struct {
    tcpNoDelay     bool
    tcpQuickAck    bool
    tcpCongestion  string
    bufferSize     int
    reuseAddr      bool
}

func (no *NetworkOptimizer) OptimizeSocket(conn net.Conn) error {
    tcpConn := conn.(*net.TCPConn)
    
    // Set TCP_NODELAY
    if err := tcpConn.SetNoDelay(no.tcpNoDelay); err != nil {
        return err
    }
    
    // Set TCP_QUICKACK
    if err := tcpConn.SetQuickAck(no.tcpQuickAck); err != nil {
        return err
    }
    
    // Set buffer sizes
    if err := tcpConn.SetReadBuffer(no.bufferSize); err != nil {
        return err
    }
    
    if err := tcpConn.SetWriteBuffer(no.bufferSize); err != nil {
        return err
    }
    
    return nil
}
```

## Conclusion

Real-time systems engineering requires deep understanding of:

1. **Timing Constraints**: Hard vs soft real-time requirements
2. **Low-Latency Architecture**: Zero-copy networking, lock-free data structures
3. **Event-Driven Systems**: Event sourcing, CQRS patterns
4. **Stream Processing**: Real-time data processing and analytics
5. **High-Frequency Trading**: Order matching, market data feeds
6. **Gaming Systems**: Game loops, physics engines, multiplayer networking
7. **IoT and Edge Computing**: Edge nodes, device management
8. **Real-Time Communication**: WebSockets, chat systems
9. **Performance Optimization**: CPU, memory, and network optimization

Mastering these areas will prepare you for building high-performance real-time systems in industries like finance, gaming, IoT, and telecommunications.

## Additional Resources

- [Real-Time Systems Design](https://www.oreilly.com/library/view/real-time-systems/9780132494282/)
- [High-Performance Go](https://github.com/geohot/minikeyvalue/)
- [Lock-Free Programming](https://preshing.com/20120612/an-introduction-to-lock-free-programming/)
- [Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html/)
- [CQRS Pattern](https://martinfowler.com/bliki/CQRS.html/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [WebSocket RFC](https://tools.ietf.org/html/rfc6455/)
- [Linux Performance Tuning](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/performance_tuning_guide/)
