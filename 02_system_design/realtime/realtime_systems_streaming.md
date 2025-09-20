# ⚡ Real-time Systems & Streaming

## Table of Contents
1. [Real-time System Architecture](#real-time-system-architecture)
2. [Event Streaming](#event-streaming)
3. [WebSocket Implementation](#websocket-implementation)
4. [Message Queues](#message-queues)
5. [Stream Processing](#stream-processing)
6. [Low Latency Optimization](#low-latency-optimization)
7. [Go Implementation Examples](#go-implementation-examples)
8. [Interview Questions](#interview-questions)

## Real-time System Architecture

### Core Real-time System

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type RealTimeSystem struct {
    eventBus    *EventBus
    processors  map[string]*StreamProcessor
    subscribers map[string][]Subscriber
    mutex       sync.RWMutex
}

type Event struct {
    ID        string
    Type      string
    Data      interface{}
    Timestamp time.Time
    Source    string
}

type EventBus struct {
    channels map[string]chan Event
    mutex    sync.RWMutex
}

type StreamProcessor struct {
    Name     string
    Input    chan Event
    Output   chan Event
    Process  func(Event) Event
    mutex    sync.RWMutex
}

type Subscriber interface {
    OnEvent(event Event) error
    GetID() string
}

func NewRealTimeSystem() *RealTimeSystem {
    return &RealTimeSystem{
        eventBus:    NewEventBus(),
        processors:  make(map[string]*StreamProcessor),
        subscribers: make(map[string][]Subscriber),
    }
}

func NewEventBus() *EventBus {
    return &EventBus{
        channels: make(map[string]chan Event),
    }
}

func (eb *EventBus) Publish(eventType string, event Event) {
    eb.mutex.RLock()
    channel, exists := eb.channels[eventType]
    eb.mutex.RUnlock()
    
    if !exists {
        eb.mutex.Lock()
        channel = make(chan Event, 1000)
        eb.channels[eventType] = channel
        eb.mutex.Unlock()
    }
    
    select {
    case channel <- event:
    default:
        // Channel is full, drop event
        fmt.Printf("Dropped event %s due to full channel\n", event.ID)
    }
}

func (eb *EventBus) Subscribe(eventType string) <-chan Event {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    channel, exists := eb.channels[eventType]
    if !exists {
        channel = make(chan Event, 1000)
        eb.channels[eventType] = channel
    }
    
    return channel
}

func (rts *RealTimeSystem) AddProcessor(name string, processor *StreamProcessor) {
    rts.mutex.Lock()
    defer rts.mutex.Unlock()
    rts.processors[name] = processor
}

func (rts *RealTimeSystem) AddSubscriber(eventType string, subscriber Subscriber) {
    rts.mutex.Lock()
    defer rts.mutex.Unlock()
    rts.subscribers[eventType] = append(rts.subscribers[eventType], subscriber)
}

func (rts *RealTimeSystem) PublishEvent(eventType string, event Event) {
    rts.eventBus.Publish(eventType, event)
    
    // Notify subscribers
    rts.mutex.RLock()
    subscribers := rts.subscribers[eventType]
    rts.mutex.RUnlock()
    
    for _, subscriber := range subscribers {
        go func(sub Subscriber) {
            if err := sub.OnEvent(event); err != nil {
                fmt.Printf("Subscriber %s error: %v\n", sub.GetID(), err)
            }
        }(subscriber)
    }
}

func (rts *RealTimeSystem) StartProcessor(name string) {
    rts.mutex.RLock()
    processor, exists := rts.processors[name]
    rts.mutex.RUnlock()
    
    if !exists {
        return
    }
    
    go func() {
        for event := range processor.Input {
            processed := processor.Process(event)
            select {
            case processor.Output <- processed:
            default:
                // Output channel is full
                fmt.Printf("Dropped processed event %s\n", processed.ID)
            }
        }
    }()
}
```

## Event Streaming

### Kafka-style Event Streaming

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type EventStream struct {
    name     string
    partitions []*Partition
    mutex    sync.RWMutex
}

type Partition struct {
    ID       int
    messages []Message
    offset   int64
    mutex    sync.RWMutex
}

type Message struct {
    Key       string
    Value     []byte
    Offset    int64
    Timestamp time.Time
    Headers   map[string]string
}

type Producer struct {
    stream *EventStream
}

type Consumer struct {
    stream     *EventStream
    partition  int
    offset     int64
    groupID    string
}

func NewEventStream(name string, partitionCount int) *EventStream {
    partitions := make([]*Partition, partitionCount)
    for i := 0; i < partitionCount; i++ {
        partitions[i] = &Partition{
            ID:       i,
            messages: make([]Message, 0),
            offset:   0,
        }
    }
    
    return &EventStream{
        name:       name,
        partitions: partitions,
    }
}

func (es *EventStream) GetPartition(key string) int {
    // Simple hash-based partitioning
    hash := 0
    for _, c := range key {
        hash = hash*31 + int(c)
    }
    return hash % len(es.partitions)
}

func (es *EventStream) Produce(key string, value []byte, headers map[string]string) (int64, error) {
    partition := es.GetPartition(key)
    
    es.mutex.RLock()
    p := es.partitions[partition]
    es.mutex.RUnlock()
    
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    message := Message{
        Key:       key,
        Value:     value,
        Offset:    p.offset,
        Timestamp: time.Now(),
        Headers:   headers,
    }
    
    p.messages = append(p.messages, message)
    p.offset++
    
    return message.Offset, nil
}

func (es *EventStream) Consume(partition, offset int64, maxMessages int) ([]Message, error) {
    es.mutex.RLock()
    if int(partition) >= len(es.partitions) {
        es.mutex.RUnlock()
        return nil, fmt.Errorf("invalid partition")
    }
    p := es.partitions[partition]
    es.mutex.RUnlock()
    
    p.mutex.RLock()
    defer p.mutex.RUnlock()
    
    if offset >= int64(len(p.messages)) {
        return []Message{}, nil
    }
    
    end := offset + int64(maxMessages)
    if end > int64(len(p.messages)) {
        end = int64(len(p.messages))
    }
    
    return p.messages[offset:end], nil
}

func NewProducer(stream *EventStream) *Producer {
    return &Producer{stream: stream}
}

func (p *Producer) Send(key string, value []byte, headers map[string]string) (int64, error) {
    return p.stream.Produce(key, value, headers)
}

func NewConsumer(stream *EventStream, partition int, groupID string) *Consumer {
    return &Consumer{
        stream:    stream,
        partition: partition,
        offset:    0,
        groupID:   groupID,
    }
}

func (c *Consumer) Poll(maxMessages int) ([]Message, error) {
    messages, err := c.stream.Consume(c.partition, c.offset, maxMessages)
    if err != nil {
        return nil, err
    }
    
    if len(messages) > 0 {
        c.offset = messages[len(messages)-1].Offset + 1
    }
    
    return messages, nil
}

func (c *Consumer) Commit() {
    // In production, this would commit the offset to a persistent store
    fmt.Printf("Committed offset %d for consumer group %s\n", c.offset, c.groupID)
}
```

## WebSocket Implementation

### WebSocket Server

```go
package main

import (
    "context"
    "fmt"
    "net/http"
    "sync"
    "time"
    
    "github.com/gorilla/websocket"
)

type WebSocketServer struct {
    upgrader    websocket.Upgrader
    connections map[string]*Connection
    mutex       sync.RWMutex
    hub         *Hub
}

type Connection struct {
    ID       string
    WS       *websocket.Conn
    Send     chan []byte
    Hub      *Hub
    mutex    sync.RWMutex
}

type Hub struct {
    connections map[string]*Connection
    broadcast   chan []byte
    register    chan *Connection
    unregister  chan *Connection
    mutex       sync.RWMutex
}

func NewWebSocketServer() *WebSocketServer {
    hub := NewHub()
    go hub.Run()
    
    return &WebSocketServer{
        upgrader: websocket.Upgrader{
            CheckOrigin: func(r *http.Request) bool {
                return true // Allow all origins in development
            },
        },
        connections: make(map[string]*Connection),
        hub:         hub,
    }
}

func NewHub() *Hub {
    return &Hub{
        connections: make(map[string]*Connection),
        broadcast:   make(chan []byte),
        register:    make(chan *Connection),
        unregister:  make(chan *Connection),
    }
}

func (h *Hub) Run() {
    for {
        select {
        case conn := <-h.register:
            h.mutex.Lock()
            h.connections[conn.ID] = conn
            h.mutex.Unlock()
            fmt.Printf("Connection %s registered\n", conn.ID)
            
        case conn := <-h.unregister:
            h.mutex.Lock()
            if _, ok := h.connections[conn.ID]; ok {
                delete(h.connections, conn.ID)
                close(conn.Send)
            }
            h.mutex.Unlock()
            fmt.Printf("Connection %s unregistered\n", conn.ID)
            
        case message := <-h.broadcast:
            h.mutex.RLock()
            for _, conn := range h.connections {
                select {
                case conn.Send <- message:
                default:
                    close(conn.Send)
                    delete(h.connections, conn.ID)
                }
            }
            h.mutex.RUnlock()
        }
    }
}

func (wss *WebSocketServer) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
    conn, err := wss.upgrader.Upgrade(w, r, nil)
    if err != nil {
        fmt.Printf("WebSocket upgrade error: %v\n", err)
        return
    }
    
    connectionID := fmt.Sprintf("conn_%d", time.Now().UnixNano())
    connection := &Connection{
        ID:   connectionID,
        WS:   conn,
        Send: make(chan []byte, 256),
        Hub:  wss.hub,
    }
    
    wss.hub.register <- connection
    
    go connection.writePump()
    go connection.readPump()
}

func (c *Connection) readPump() {
    defer func() {
        c.Hub.unregister <- c
        c.WS.Close()
    }()
    
    c.WS.SetReadLimit(512)
    c.WS.SetReadDeadline(time.Now().Add(60 * time.Second))
    c.WS.SetPongHandler(func(string) error {
        c.WS.SetReadDeadline(time.Now().Add(60 * time.Second))
        return nil
    })
    
    for {
        _, message, err := c.WS.ReadMessage()
        if err != nil {
            if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
                fmt.Printf("WebSocket error: %v\n", err)
            }
            break
        }
        
        // Process message
        c.handleMessage(message)
    }
}

func (c *Connection) writePump() {
    ticker := time.NewTicker(54 * time.Second)
    defer func() {
        ticker.Stop()
        c.WS.Close()
    }()
    
    for {
        select {
        case message, ok := <-c.Send:
            c.WS.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if !ok {
                c.WS.WriteMessage(websocket.CloseMessage, []byte{})
                return
            }
            
            if err := c.WS.WriteMessage(websocket.TextMessage, message); err != nil {
                return
            }
            
        case <-ticker.C:
            c.WS.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if err := c.WS.WriteMessage(websocket.PingMessage, nil); err != nil {
                return
            }
        }
    }
}

func (c *Connection) handleMessage(message []byte) {
    // Process incoming message
    fmt.Printf("Received message from %s: %s\n", c.ID, string(message))
    
    // Echo back to sender
    c.Send <- message
}

func (c *Connection) SendMessage(message []byte) {
    select {
    case c.Send <- message:
    default:
        close(c.Send)
    }
}
```

## Message Queues

### Advanced Message Queue

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type MessageQueue struct {
    name        string
    messages    []Message
    consumers   []Consumer
    mutex       sync.RWMutex
    maxSize     int
    retention   time.Duration
}

type Consumer struct {
    ID           string
    GroupID      string
    Offset       int64
    LastSeen     time.Time
    MessageChan  chan Message
    mutex        sync.RWMutex
}

type QueueManager struct {
    queues map[string]*MessageQueue
    mutex  sync.RWMutex
}

func NewQueueManager() *QueueManager {
    return &QueueManager{
        queues: make(map[string]*MessageQueue),
    }
}

func (qm *QueueManager) CreateQueue(name string, maxSize int, retention time.Duration) *MessageQueue {
    qm.mutex.Lock()
    defer qm.mutex.Unlock()
    
    queue := &MessageQueue{
        name:      name,
        messages:  make([]Message, 0),
        consumers: make([]Consumer, 0),
        maxSize:   maxSize,
        retention: retention,
    }
    
    qm.queues[name] = queue
    return queue
}

func (qm *QueueManager) GetQueue(name string) (*MessageQueue, bool) {
    qm.mutex.RLock()
    defer qm.mutex.RUnlock()
    
    queue, exists := qm.queues[name]
    return queue, exists
}

func (mq *MessageQueue) Publish(message Message) error {
    mq.mutex.Lock()
    defer mq.mutex.Unlock()
    
    if len(mq.messages) >= mq.maxSize {
        return fmt.Errorf("queue is full")
    }
    
    message.Offset = int64(len(mq.messages))
    message.Timestamp = time.Now()
    mq.messages = append(mq.messages, message)
    
    // Notify consumers
    for i := range mq.consumers {
        select {
        case mq.consumers[i].MessageChan <- message:
        default:
            // Consumer channel is full
        }
    }
    
    return nil
}

func (mq *MessageQueue) Subscribe(consumerID, groupID string) (*Consumer, error) {
    mq.mutex.Lock()
    defer mq.mutex.Unlock()
    
    consumer := &Consumer{
        ID:          consumerID,
        GroupID:     groupID,
        Offset:      0,
        LastSeen:    time.Now(),
        MessageChan: make(chan Message, 100),
    }
    
    mq.consumers = append(mq.consumers, *consumer)
    return consumer, nil
}

func (mq *MessageQueue) Consume(consumerID string, maxMessages int) ([]Message, error) {
    mq.mutex.RLock()
    defer mq.mutex.RUnlock()
    
    var consumer *Consumer
    for i := range mq.consumers {
        if mq.consumers[i].ID == consumerID {
            consumer = &mq.consumers[i]
            break
        }
    }
    
    if consumer == nil {
        return nil, fmt.Errorf("consumer not found")
    }
    
    if consumer.Offset >= int64(len(mq.messages)) {
        return []Message{}, nil
    }
    
    end := consumer.Offset + int64(maxMessages)
    if end > int64(len(mq.messages)) {
        end = int64(len(mq.messages))
    }
    
    messages := mq.messages[consumer.Offset:end]
    consumer.Offset = end
    consumer.LastSeen = time.Now()
    
    return messages, nil
}

func (mq *MessageQueue) Cleanup() {
    mq.mutex.Lock()
    defer mq.mutex.Unlock()
    
    cutoff := time.Now().Add(-mq.retention)
    var validMessages []Message
    
    for _, message := range mq.messages {
        if message.Timestamp.After(cutoff) {
            validMessages = append(validMessages, message)
        }
    }
    
    mq.messages = validMessages
}

// Dead Letter Queue
type DeadLetterQueue struct {
    messages []Message
    mutex    sync.RWMutex
}

func NewDeadLetterQueue() *DeadLetterQueue {
    return &DeadLetterQueue{
        messages: make([]Message, 0),
    }
}

func (dlq *DeadLetterQueue) AddMessage(message Message, reason string) {
    dlq.mutex.Lock()
    defer dlq.mutex.Unlock()
    
    message.Headers["dlq_reason"] = reason
    message.Headers["dlq_timestamp"] = time.Now().Format(time.RFC3339)
    dlq.messages = append(dlq.messages, message)
}

func (dlq *DeadLetterQueue) GetMessages() []Message {
    dlq.mutex.RLock()
    defer dlq.mutex.RUnlock()
    
    return dlq.messages
}
```

## Stream Processing

### Stream Processing Engine

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type StreamProcessor struct {
    name     string
    input    chan Event
    output   chan Event
    filters  []Filter
    mappers  []Mapper
    reducers []Reducer
    mutex    sync.RWMutex
}

type Filter interface {
    Filter(event Event) bool
}

type Mapper interface {
    Map(event Event) Event
}

type Reducer interface {
    Reduce(events []Event) Event
}

type StreamProcessingEngine struct {
    processors map[string]*StreamProcessor
    mutex      sync.RWMutex
}

func NewStreamProcessingEngine() *StreamProcessingEngine {
    return &StreamProcessingEngine{
        processors: make(map[string]*StreamProcessor),
    }
}

func (spe *StreamProcessingEngine) AddProcessor(name string, processor *StreamProcessor) {
    spe.mutex.Lock()
    defer spe.mutex.Unlock()
    spe.processors[name] = processor
}

func (spe *StreamProcessingEngine) StartProcessor(name string) {
    spe.mutex.RLock()
    processor, exists := spe.processors[name]
    spe.mutex.RUnlock()
    
    if !exists {
        return
    }
    
    go func() {
        for event := range processor.input {
            processed := processor.ProcessEvent(event)
            if processed != nil {
                select {
                case processor.output <- *processed:
                default:
                    // Output channel is full
                    fmt.Printf("Dropped processed event %s\n", processed.ID)
                }
            }
        }
    }()
}

func (sp *StreamProcessor) ProcessEvent(event Event) *Event {
    sp.mutex.RLock()
    filters := sp.filters
    mappers := sp.mappers
    reducers := sp.reducers
    sp.mutex.RUnlock()
    
    // Apply filters
    for _, filter := range filters {
        if !filter.Filter(event) {
            return nil // Event filtered out
        }
    }
    
    // Apply mappers
    for _, mapper := range mappers {
        event = mapper.Map(event)
    }
    
    // Apply reducers (simplified - in production, this would be more complex)
    if len(reducers) > 0 {
        // In a real implementation, reducers would work on windows of events
        event = reducers[0].Reduce([]Event{event})
    }
    
    return &event
}

func (sp *StreamProcessor) AddFilter(filter Filter) {
    sp.mutex.Lock()
    defer sp.mutex.Unlock()
    sp.filters = append(sp.filters, filter)
}

func (sp *StreamProcessor) AddMapper(mapper Mapper) {
    sp.mutex.Lock()
    defer sp.mutex.Unlock()
    sp.mappers = append(sp.mappers, mapper)
}

func (sp *StreamProcessor) AddReducer(reducer Reducer) {
    sp.mutex.Lock()
    defer sp.mutex.Unlock()
    sp.reducers = append(sp.reducers, reducer)
}

// Example filters
type AmountFilter struct {
    MinAmount int64
    MaxAmount int64
}

func (af *AmountFilter) Filter(event Event) bool {
    if amount, ok := event.Data.(map[string]interface{})["amount"]; ok {
        if amt, ok := amount.(int64); ok {
            return amt >= af.MinAmount && amt <= af.MaxAmount
        }
    }
    return false
}

// Example mappers
type TimestampMapper struct{}

func (tm *TimestampMapper) Map(event Event) Event {
    event.Timestamp = time.Now()
    return event
}

// Example reducers
type CountReducer struct {
    count int64
}

func (cr *CountReducer) Reduce(events []Event) Event {
    cr.count++
    return Event{
        ID:        fmt.Sprintf("count_%d", cr.count),
        Type:      "count",
        Data:      map[string]interface{}{"count": cr.count},
        Timestamp: time.Now(),
    }
}
```

## Low Latency Optimization

### Performance Optimizations

```go
package main

import (
    "context"
    "fmt"
    "runtime"
    "sync"
    "time"
)

type LowLatencySystem struct {
    eventPool    sync.Pool
    workerPool   *WorkerPool
    metrics      *LatencyMetrics
    mutex        sync.RWMutex
}

type Event struct {
    ID        string
    Data      interface{}
    Timestamp time.Time
}

type WorkerPool struct {
    workers    []*Worker
    jobQueue   chan Job
    quit       chan bool
    wg         sync.WaitGroup
}

type Worker struct {
    ID       int
    JobQueue chan Job
    Quit     chan bool
    WG       *sync.WaitGroup
}

type Job struct {
    ID   string
    Data interface{}
    Fn   func(interface{}) interface{}
}

type LatencyMetrics struct {
    totalLatency   time.Duration
    requestCount   int64
    mutex          sync.RWMutex
}

func NewLowLatencySystem() *LowLatencySystem {
    return &LowLatencySystem{
        eventPool: sync.Pool{
            New: func() interface{} {
                return &Event{}
            },
        },
        workerPool: NewWorkerPool(runtime.NumCPU()),
        metrics:    &LatencyMetrics{},
    }
}

func NewWorkerPool(numWorkers int) *WorkerPool {
    wp := &WorkerPool{
        workers:  make([]*Worker, numWorkers),
        jobQueue: make(chan Job, 1000),
        quit:     make(chan bool),
    }
    
    for i := 0; i < numWorkers; i++ {
        worker := &Worker{
            ID:       i,
            JobQueue: make(chan Job, 100),
            Quit:     make(chan bool),
            WG:       &wp.wg,
        }
        wp.workers[i] = worker
        go worker.Start()
    }
    
    return wp
}

func (w *Worker) Start() {
    w.WG.Add(1)
    defer w.WG.Done()
    
    for {
        select {
        case job := <-w.JobQueue:
            start := time.Now()
            result := job.Fn(job.Data)
            latency := time.Since(start)
            
            // Update metrics
            fmt.Printf("Worker %d processed job %s in %v\n", w.ID, job.ID, latency)
            
        case <-w.Quit:
            return
        }
    }
}

func (wp *WorkerPool) SubmitJob(job Job) {
    // Simple round-robin job distribution
    workerID := job.ID[0] % len(wp.workers)
    wp.workers[workerID].JobQueue <- job
}

func (wp *WorkerPool) Stop() {
    for _, worker := range wp.workers {
        close(worker.Quit)
    }
    wp.wg.Wait()
}

func (lls *LowLatencySystem) ProcessEvent(data interface{}) {
    start := time.Now()
    
    // Get event from pool
    event := lls.eventPool.Get().(*Event)
    event.ID = fmt.Sprintf("event_%d", time.Now().UnixNano())
    event.Data = data
    event.Timestamp = time.Now()
    
    // Submit to worker pool
    job := Job{
        ID:   event.ID,
        Data: event,
        Fn:   lls.processEvent,
    }
    
    lls.workerPool.SubmitJob(job)
    
    // Update metrics
    latency := time.Since(start)
    lls.metrics.UpdateLatency(latency)
    
    // Return event to pool
    lls.eventPool.Put(event)
}

func (lls *LowLatencySystem) processEvent(data interface{}) interface{} {
    event := data.(*Event)
    
    // Simulate processing
    time.Sleep(1 * time.Millisecond)
    
    return event
}

func (lm *LatencyMetrics) UpdateLatency(latency time.Duration) {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    lm.totalLatency += latency
    lm.requestCount++
}

func (lm *LatencyMetrics) GetAverageLatency() time.Duration {
    lm.mutex.RLock()
    defer lm.mutex.RUnlock()
    
    if lm.requestCount == 0 {
        return 0
    }
    
    return lm.totalLatency / time.Duration(lm.requestCount)
}

// Memory Pool for frequent allocations
type ObjectPool struct {
    pool chan interface{}
    new  func() interface{}
}

func NewObjectPool(size int, newFunc func() interface{}) *ObjectPool {
    return &ObjectPool{
        pool: make(chan interface{}, size),
        new:  newFunc,
    }
}

func (op *ObjectPool) Get() interface{} {
    select {
    case obj := <-op.pool:
        return obj
    default:
        return op.new()
    }
}

func (op *ObjectPool) Put(obj interface{}) {
    select {
    case op.pool <- obj:
    default:
        // Pool is full, discard object
    }
}

// Lock-free data structures
type LockFreeQueue struct {
    head *Node
    tail *Node
}

type Node struct {
    value interface{}
    next  *Node
}

func NewLockFreeQueue() *LockFreeQueue {
    dummy := &Node{}
    return &LockFreeQueue{
        head: dummy,
        tail: dummy,
    }
}

func (q *LockFreeQueue) Enqueue(value interface{}) {
    node := &Node{value: value}
    
    for {
        tail := q.tail
        next := tail.next
        
        if tail == q.tail {
            if next == nil {
                if tail.next == nil {
                    tail.next = node
                    q.tail = node
                    return
                }
            } else {
                q.tail = next
            }
        }
    }
}

func (q *LockFreeQueue) Dequeue() interface{} {
    for {
        head := q.head
        tail := q.tail
        next := head.next
        
        if head == q.head {
            if head == tail {
                if next == nil {
                    return nil
                }
                q.tail = next
            } else {
                value := next.value
                q.head = next
                return value
            }
        }
    }
}
```

## Interview Questions

### Basic Concepts
1. **What are the challenges of building real-time systems?**
2. **How do you implement event streaming?**
3. **What is the difference between WebSockets and HTTP polling?**
4. **How do you handle message ordering in distributed systems?**
5. **What are the trade-offs of different message queue implementations?**

### Advanced Topics
1. **How would you implement low-latency event processing?**
2. **How do you handle backpressure in streaming systems?**
3. **What are the challenges of real-time data consistency?**
4. **How do you optimize for low latency?**
5. **How do you handle failure recovery in streaming systems?**

### System Design
1. **Design a real-time chat system.**
2. **How would you implement a real-time analytics system?**
3. **Design a streaming data processing pipeline.**
4. **How would you implement real-time notifications?**
5. **Design a low-latency trading system.**

## Conclusion

Real-time systems and streaming require careful attention to performance, scalability, and reliability. Key areas to master:

- **Real-time Architecture**: Event-driven design, streaming patterns
- **Event Streaming**: Message queues, event sourcing, CQRS
- **WebSockets**: Real-time communication, connection management
- **Stream Processing**: Real-time data processing, windowing
- **Low Latency**: Performance optimization, memory management
- **Scalability**: Horizontal scaling, load balancing

Understanding these concepts helps in:
- Building real-time applications
- Implementing streaming systems
- Optimizing for performance
- Handling high-throughput data
- Preparing for technical interviews

This guide provides a comprehensive foundation for real-time systems concepts and their practical implementation in Go.


## Go Implementation Examples

<!-- AUTO-GENERATED ANCHOR: originally referenced as #go-implementation-examples -->

Placeholder content. Please replace with proper section.
