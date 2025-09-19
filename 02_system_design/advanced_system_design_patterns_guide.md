# 🏗️ **Advanced System Design Patterns Guide**

## **Table of Contents**

1. [🔄 Event Sourcing at Scale](#-event-sourcing-at-scale)
2. [📊 CQRS with Microservices](#-cqrs-with-microservices)
3. [🎭 Saga Orchestration Patterns](#-saga-orchestration-patterns)
4. [🚀 Advanced Caching Strategies](#-advanced-caching-strategies)
5. [⚡ Circuit Breaker and Resilience Patterns](#-circuit-breaker-and-resilience-patterns)
6. [🏛️ Bulkhead Isolation Patterns](#-bulkhead-isolation-patterns)
7. [🚪 API Gateway Patterns](#-api-gateway-patterns)
8. [❓ Interview Questions](#-interview-questions)

---

## 🔄 **Event Sourcing at Scale**

### **Enterprise Event Store Implementation**

```go
package eventsourcing

import (
    "context"
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Scalable event store for high-throughput systems
type EventStore interface {
    AppendEvents(ctx context.Context, streamID string, expectedVersion int64, events []Event) error
    ReadEvents(ctx context.Context, streamID string, fromVersion int64, maxCount int) ([]Event, error)
    ReadEventsBackward(ctx context.Context, streamID string, fromVersion int64, maxCount int) ([]Event, error)
    Subscribe(ctx context.Context, streamID string, fromVersion int64) (<-chan Event, error)
    CreateSnapshot(ctx context.Context, streamID string, version int64, snapshot interface{}) error
    LoadSnapshot(ctx context.Context, streamID string) (*Snapshot, error)
}

type DistributedEventStore struct {
    storage        EventStorage
    snapshotStore  SnapshotStorage
    eventBus       EventBus
    partitioner    StreamPartitioner
    replication    ReplicationManager
    cache          EventCache
    metrics        EventStoreMetrics
    mu             sync.RWMutex
}

type Event struct {
    ID            string                 `json:"id"`
    StreamID      string                 `json:"stream_id"`
    Version       int64                  `json:"version"`
    EventType     string                 `json:"event_type"`
    Data          json.RawMessage        `json:"data"`
    Metadata      map[string]interface{} `json:"metadata"`
    Timestamp     time.Time              `json:"timestamp"`
    CorrelationID string                 `json:"correlation_id,omitempty"`
    CausationID   string                 `json:"causation_id,omitempty"`
}

type Snapshot struct {
    StreamID    string      `json:"stream_id"`
    Version     int64       `json:"version"`
    Data        interface{} `json:"data"`
    Timestamp   time.Time   `json:"timestamp"`
}

func NewDistributedEventStore(config EventStoreConfig) *DistributedEventStore {
    return &DistributedEventStore{
        storage:       NewDistributedStorage(config.StorageConfig),
        snapshotStore: NewSnapshotStorage(config.SnapshotConfig),
        eventBus:      NewDistributedEventBus(config.EventBusConfig),
        partitioner:   NewConsistentHashPartitioner(config.PartitionConfig),
        replication:   NewReplicationManager(config.ReplicationConfig),
        cache:         NewDistributedEventCache(config.CacheConfig),
        metrics:       NewEventStoreMetrics(),
    }
}

func (des *DistributedEventStore) AppendEvents(ctx context.Context, streamID string, expectedVersion int64, events []Event) error {
    des.mu.Lock()
    defer des.mu.Unlock()
    
    startTime := time.Now()
    defer func() {
        des.metrics.RecordAppendDuration(time.Since(startTime))
    }()
    
    // Determine partition for the stream
    partition := des.partitioner.GetPartition(streamID)
    
    // Check current version for optimistic concurrency control
    currentVersion, err := des.storage.GetStreamVersion(ctx, streamID)
    if err != nil {
        return fmt.Errorf("failed to get current version: %w", err)
    }
    
    if currentVersion != expectedVersion {
        return &ConcurrencyError{
            StreamID:        streamID,
            ExpectedVersion: expectedVersion,
            ActualVersion:   currentVersion,
        }
    }
    
    // Assign versions and prepare events
    for i, event := range events {
        event.Version = expectedVersion + int64(i) + 1
        event.StreamID = streamID
        event.Timestamp = time.Now()
        if event.ID == "" {
            event.ID = generateEventID()
        }
        events[i] = event
    }
    
    // Append to storage with replication
    if err := des.replication.ReplicateEvents(ctx, partition, events); err != nil {
        return fmt.Errorf("replication failed: %w", err)
    }
    
    // Invalidate cache
    des.cache.Invalidate(streamID)
    
    // Publish events to event bus
    for _, event := range events {
        if err := des.eventBus.Publish(ctx, event); err != nil {
            // Log error but don't fail the append
            des.metrics.RecordPublishError()
        }
    }
    
    des.metrics.RecordEventsAppended(len(events))
    return nil
}

func (des *DistributedEventStore) ReadEvents(ctx context.Context, streamID string, fromVersion int64, maxCount int) ([]Event, error) {
    startTime := time.Now()
    defer func() {
        des.metrics.RecordReadDuration(time.Since(startTime))
    }()
    
    // Check cache first
    if cached := des.cache.Get(streamID, fromVersion, maxCount); cached != nil {
        des.metrics.RecordCacheHit()
        return cached, nil
    }
    
    des.metrics.RecordCacheMiss()
    
    // Read from storage
    events, err := des.storage.ReadEvents(ctx, streamID, fromVersion, maxCount)
    if err != nil {
        return nil, err
    }
    
    // Cache the results
    des.cache.Put(streamID, fromVersion, events)
    
    des.metrics.RecordEventsRead(len(events))
    return events, nil
}

// Aggregate root with event sourcing
type AggregateRoot interface {
    GetID() string
    GetVersion() int64
    GetUncommittedEvents() []Event
    MarkEventsAsCommitted()
    LoadFromHistory(events []Event) error
    CreateSnapshot() (interface{}, error)
    LoadFromSnapshot(snapshot interface{}) error
}

type BaseAggregate struct {
    ID                string  `json:"id"`
    Version           int64   `json:"version"`
    UncommittedEvents []Event `json:"-"`
}

func (ba *BaseAggregate) GetID() string {
    return ba.ID
}

func (ba *BaseAggregate) GetVersion() int64 {
    return ba.Version
}

func (ba *BaseAggregate) GetUncommittedEvents() []Event {
    return ba.UncommittedEvents
}

func (ba *BaseAggregate) MarkEventsAsCommitted() {
    ba.UncommittedEvents = ba.UncommittedEvents[:0]
}

func (ba *BaseAggregate) raiseEvent(eventType string, data interface{}) error {
    eventData, err := json.Marshal(data)
    if err != nil {
        return err
    }
    
    event := Event{
        ID:        generateEventID(),
        StreamID:  ba.ID,
        Version:   ba.Version + 1,
        EventType: eventType,
        Data:      eventData,
        Timestamp: time.Now(),
        Metadata:  make(map[string]interface{}),
    }
    
    ba.UncommittedEvents = append(ba.UncommittedEvents, event)
    ba.Version++
    
    return nil
}

// Example aggregate: Order
type Order struct {
    BaseAggregate
    CustomerID    string      `json:"customer_id"`
    Items         []OrderItem `json:"items"`
    Status        OrderStatus `json:"status"`
    TotalAmount   float64     `json:"total_amount"`
    CreatedAt     time.Time   `json:"created_at"`
    UpdatedAt     time.Time   `json:"updated_at"`
}

type OrderItem struct {
    ProductID string  `json:"product_id"`
    Quantity  int     `json:"quantity"`
    Price     float64 `json:"price"`
}

type OrderStatus string

const (
    OrderPending   OrderStatus = "pending"
    OrderConfirmed OrderStatus = "confirmed"
    OrderShipped   OrderStatus = "shipped"
    OrderDelivered OrderStatus = "delivered"
    OrderCancelled OrderStatus = "cancelled"
)

// Order events
type OrderCreatedEvent struct {
    OrderID     string      `json:"order_id"`
    CustomerID  string      `json:"customer_id"`
    Items       []OrderItem `json:"items"`
    TotalAmount float64     `json:"total_amount"`
    CreatedAt   time.Time   `json:"created_at"`
}

type OrderConfirmedEvent struct {
    OrderID     string    `json:"order_id"`
    ConfirmedAt time.Time `json:"confirmed_at"`
}

type OrderShippedEvent struct {
    OrderID      string    `json:"order_id"`
    TrackingID   string    `json:"tracking_id"`
    ShippedAt    time.Time `json:"shipped_at"`
}

type OrderCancelledEvent struct {
    OrderID     string    `json:"order_id"`
    Reason      string    `json:"reason"`
    CancelledAt time.Time `json:"cancelled_at"`
}

// Order business logic
func NewOrder(customerID string, items []OrderItem) (*Order, error) {
    if len(items) == 0 {
        return nil, fmt.Errorf("order must have at least one item")
    }
    
    order := &Order{
        BaseAggregate: BaseAggregate{
            ID:      generateOrderID(),
            Version: 0,
        },
        CustomerID: customerID,
        Items:      items,
        Status:     OrderPending,
        CreatedAt:  time.Now(),
    }
    
    // Calculate total amount
    var total float64
    for _, item := range items {
        total += item.Price * float64(item.Quantity)
    }
    order.TotalAmount = total
    
    // Raise order created event
    if err := order.raiseEvent("OrderCreated", OrderCreatedEvent{
        OrderID:     order.ID,
        CustomerID:  customerID,
        Items:       items,
        TotalAmount: total,
        CreatedAt:   order.CreatedAt,
    }); err != nil {
        return nil, err
    }
    
    return order, nil
}

func (o *Order) Confirm() error {
    if o.Status != OrderPending {
        return fmt.Errorf("cannot confirm order in status: %s", o.Status)
    }
    
    o.Status = OrderConfirmed
    o.UpdatedAt = time.Now()
    
    return o.raiseEvent("OrderConfirmed", OrderConfirmedEvent{
        OrderID:     o.ID,
        ConfirmedAt: o.UpdatedAt,
    })
}

func (o *Order) Ship(trackingID string) error {
    if o.Status != OrderConfirmed {
        return fmt.Errorf("cannot ship order in status: %s", o.Status)
    }
    
    o.Status = OrderShipped
    o.UpdatedAt = time.Now()
    
    return o.raiseEvent("OrderShipped", OrderShippedEvent{
        OrderID:    o.ID,
        TrackingID: trackingID,
        ShippedAt:  o.UpdatedAt,
    })
}

func (o *Order) Cancel(reason string) error {
    if o.Status == OrderDelivered {
        return fmt.Errorf("cannot cancel delivered order")
    }
    
    o.Status = OrderCancelled
    o.UpdatedAt = time.Now()
    
    return o.raiseEvent("OrderCancelled", OrderCancelledEvent{
        OrderID:     o.ID,
        Reason:      reason,
        CancelledAt: o.UpdatedAt,
    })
}

func (o *Order) LoadFromHistory(events []Event) error {
    for _, event := range events {
        if err := o.Apply(event); err != nil {
            return err
        }
        o.Version = event.Version
    }
    return nil
}

func (o *Order) Apply(event Event) error {
    switch event.EventType {
    case "OrderCreated":
        var e OrderCreatedEvent
        if err := json.Unmarshal(event.Data, &e); err != nil {
            return err
        }
        o.ID = e.OrderID
        o.CustomerID = e.CustomerID
        o.Items = e.Items
        o.TotalAmount = e.TotalAmount
        o.Status = OrderPending
        o.CreatedAt = e.CreatedAt
        
    case "OrderConfirmed":
        var e OrderConfirmedEvent
        if err := json.Unmarshal(event.Data, &e); err != nil {
            return err
        }
        o.Status = OrderConfirmed
        o.UpdatedAt = e.ConfirmedAt
        
    case "OrderShipped":
        var e OrderShippedEvent
        if err := json.Unmarshal(event.Data, &e); err != nil {
            return err
        }
        o.Status = OrderShipped
        o.UpdatedAt = e.ShippedAt
        
    case "OrderCancelled":
        var e OrderCancelledEvent
        if err := json.Unmarshal(event.Data, &e); err != nil {
            return err
        }
        o.Status = OrderCancelled
        o.UpdatedAt = e.CancelledAt
        
    default:
        return fmt.Errorf("unknown event type: %s", event.EventType)
    }
    
    return nil
}

func (o *Order) CreateSnapshot() (interface{}, error) {
    return OrderSnapshot{
        ID:          o.ID,
        CustomerID:  o.CustomerID,
        Items:       o.Items,
        Status:      o.Status,
        TotalAmount: o.TotalAmount,
        CreatedAt:   o.CreatedAt,
        UpdatedAt:   o.UpdatedAt,
        Version:     o.Version,
    }, nil
}

type OrderSnapshot struct {
    ID          string      `json:"id"`
    CustomerID  string      `json:"customer_id"`
    Items       []OrderItem `json:"items"`
    Status      OrderStatus `json:"status"`
    TotalAmount float64     `json:"total_amount"`
    CreatedAt   time.Time   `json:"created_at"`
    UpdatedAt   time.Time   `json:"updated_at"`
    Version     int64       `json:"version"`
}

func (o *Order) LoadFromSnapshot(snapshot interface{}) error {
    orderSnapshot, ok := snapshot.(OrderSnapshot)
    if !ok {
        return fmt.Errorf("invalid snapshot type")
    }
    
    o.ID = orderSnapshot.ID
    o.CustomerID = orderSnapshot.CustomerID
    o.Items = orderSnapshot.Items
    o.Status = orderSnapshot.Status
    o.TotalAmount = orderSnapshot.TotalAmount
    o.CreatedAt = orderSnapshot.CreatedAt
    o.UpdatedAt = orderSnapshot.UpdatedAt
    o.Version = orderSnapshot.Version
    
    return nil
}

// Repository pattern for aggregates
type OrderRepository struct {
    eventStore    EventStore
    snapshotFreq  int // Take snapshot every N events
}

func NewOrderRepository(eventStore EventStore) *OrderRepository {
    return &OrderRepository{
        eventStore:   eventStore,
        snapshotFreq: 10, // Snapshot every 10 events
    }
}

func (repo *OrderRepository) Save(ctx context.Context, order *Order) error {
    events := order.GetUncommittedEvents()
    if len(events) == 0 {
        return nil // Nothing to save
    }
    
    expectedVersion := order.GetVersion() - int64(len(events))
    
    if err := repo.eventStore.AppendEvents(ctx, order.GetID(), expectedVersion, events); err != nil {
        return err
    }
    
    order.MarkEventsAsCommitted()
    
    // Create snapshot if needed
    if order.GetVersion()%int64(repo.snapshotFreq) == 0 {
        snapshot, err := order.CreateSnapshot()
        if err != nil {
            return fmt.Errorf("failed to create snapshot: %w", err)
        }
        
        if err := repo.eventStore.CreateSnapshot(ctx, order.GetID(), order.GetVersion(), snapshot); err != nil {
            // Log error but don't fail the save
            fmt.Printf("Failed to save snapshot: %v\n", err)
        }
    }
    
    return nil
}

func (repo *OrderRepository) Load(ctx context.Context, orderID string) (*Order, error) {
    order := &Order{}
    
    // Try to load from snapshot first
    snapshot, err := repo.eventStore.LoadSnapshot(ctx, orderID)
    if err == nil && snapshot != nil {
        if err := order.LoadFromSnapshot(snapshot.Data); err != nil {
            return nil, fmt.Errorf("failed to load from snapshot: %w", err)
        }
        
        // Load events after snapshot
        events, err := repo.eventStore.ReadEvents(ctx, orderID, snapshot.Version+1, 1000)
        if err != nil {
            return nil, fmt.Errorf("failed to read events after snapshot: %w", err)
        }
        
        if err := order.LoadFromHistory(events); err != nil {
            return nil, fmt.Errorf("failed to apply events after snapshot: %w", err)
        }
    } else {
        // Load all events from beginning
        events, err := repo.eventStore.ReadEvents(ctx, orderID, 0, 1000)
        if err != nil {
            return nil, fmt.Errorf("failed to read events: %w", err)
        }
        
        if len(events) == 0 {
            return nil, fmt.Errorf("order not found: %s", orderID)
        }
        
        if err := order.LoadFromHistory(events); err != nil {
            return nil, fmt.Errorf("failed to load from history: %w", err)
        }
    }
    
    return order, nil
}

// Event projections for read models
type OrderProjection struct {
    projectionStore ProjectionStore
    eventBus        EventBus
    handlers        map[string]EventHandler
}

type ProjectionStore interface {
    UpdateOrderSummary(ctx context.Context, orderSummary OrderSummary) error
    GetOrderSummary(ctx context.Context, orderID string) (*OrderSummary, error)
    UpdateCustomerOrderStats(ctx context.Context, customerID string, stats CustomerOrderStats) error
}

type OrderSummary struct {
    OrderID       string      `json:"order_id"`
    CustomerID    string      `json:"customer_id"`
    Status        OrderStatus `json:"status"`
    TotalAmount   float64     `json:"total_amount"`
    ItemCount     int         `json:"item_count"`
    CreatedAt     time.Time   `json:"created_at"`
    LastUpdatedAt time.Time   `json:"last_updated_at"`
}

type CustomerOrderStats struct {
    CustomerID   string  `json:"customer_id"`
    TotalOrders  int     `json:"total_orders"`
    TotalSpent   float64 `json:"total_spent"`
    LastOrderAt  time.Time `json:"last_order_at"`
}

func NewOrderProjection(projectionStore ProjectionStore, eventBus EventBus) *OrderProjection {
    projection := &OrderProjection{
        projectionStore: projectionStore,
        eventBus:        eventBus,
        handlers:        make(map[string]EventHandler),
    }
    
    // Register event handlers
    projection.handlers["OrderCreated"] = projection.handleOrderCreated
    projection.handlers["OrderConfirmed"] = projection.handleOrderConfirmed
    projection.handlers["OrderShipped"] = projection.handleOrderShipped
    projection.handlers["OrderCancelled"] = projection.handleOrderCancelled
    
    return projection
}

func (op *OrderProjection) handleOrderCreated(ctx context.Context, event Event) error {
    var e OrderCreatedEvent
    if err := json.Unmarshal(event.Data, &e); err != nil {
        return err
    }
    
    orderSummary := OrderSummary{
        OrderID:       e.OrderID,
        CustomerID:    e.CustomerID,
        Status:        OrderPending,
        TotalAmount:   e.TotalAmount,
        ItemCount:     len(e.Items),
        CreatedAt:     e.CreatedAt,
        LastUpdatedAt: e.CreatedAt,
    }
    
    return op.projectionStore.UpdateOrderSummary(ctx, orderSummary)
}

func (op *OrderProjection) handleOrderConfirmed(ctx context.Context, event Event) error {
    var e OrderConfirmedEvent
    if err := json.Unmarshal(event.Data, &e); err != nil {
        return err
    }
    
    orderSummary, err := op.projectionStore.GetOrderSummary(ctx, e.OrderID)
    if err != nil {
        return err
    }
    
    orderSummary.Status = OrderConfirmed
    orderSummary.LastUpdatedAt = e.ConfirmedAt
    
    return op.projectionStore.UpdateOrderSummary(ctx, *orderSummary)
}

// Event-driven process managers (Sagas)
type OrderProcessManager struct {
    eventBus        EventBus
    commandBus      CommandBus
    paymentService  PaymentService
    inventoryService InventoryService
    shippingService ShippingService
}

func (opm *OrderProcessManager) HandleOrderCreated(ctx context.Context, event Event) error {
    var e OrderCreatedEvent
    if err := json.Unmarshal(event.Data, &e); err != nil {
        return err
    }
    
    // Reserve inventory
    reserveCmd := ReserveInventoryCommand{
        OrderID: e.OrderID,
        Items:   e.Items,
    }
    
    if err := opm.commandBus.Send(ctx, reserveCmd); err != nil {
        return err
    }
    
    // Process payment
    paymentCmd := ProcessPaymentCommand{
        OrderID:     e.OrderID,
        CustomerID:  e.CustomerID,
        Amount:      e.TotalAmount,
    }
    
    return opm.commandBus.Send(ctx, paymentCmd)
}
```

---

## 📊 **CQRS with Microservices**

### **Advanced CQRS Implementation**

```go
package cqrs

import (
    "context"
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Command Query Responsibility Segregation implementation
type CQRSFramework struct {
    commandBus    CommandBus
    queryBus      QueryBus
    eventStore    EventStore
    readModelRepo ReadModelRepository
    projections   []Projection
    commandHandler map[string]CommandHandler
    queryHandler   map[string]QueryHandler
}

// Command side (Write operations)
type Command interface {
    GetID() string
    GetType() string
    GetMetadata() map[string]interface{}
    Validate() error
}

type CommandHandler interface {
    Handle(ctx context.Context, command Command) error
    CanHandle(commandType string) bool
}

type CommandBus interface {
    Send(ctx context.Context, command Command) error
    RegisterHandler(commandType string, handler CommandHandler)
}

// Query side (Read operations)
type Query interface {
    GetID() string
    GetType() string
    GetCriteria() map[string]interface{}
    Validate() error
}

type QueryHandler interface {
    Handle(ctx context.Context, query Query) (interface{}, error)
    CanHandle(queryType string) bool
}

type QueryBus interface {
    Execute(ctx context.Context, query Query) (interface{}, error)
    RegisterHandler(queryType string, handler QueryHandler)
}

// Distributed command bus implementation
type DistributedCommandBus struct {
    localHandlers  map[string]CommandHandler
    messageQueue   MessageQueue
    serializer     CommandSerializer
    middleware     []CommandMiddleware
    retryPolicy    RetryPolicy
    deadLetterQueue DeadLetterQueue
    metrics        CommandBusMetrics
    mu             sync.RWMutex
}

type CommandMiddleware interface {
    Execute(ctx context.Context, command Command, next func() error) error
}

func NewDistributedCommandBus(config CommandBusConfig) *DistributedCommandBus {
    return &DistributedCommandBus{
        localHandlers:   make(map[string]CommandHandler),
        messageQueue:    NewDistributedMessageQueue(config.QueueConfig),
        serializer:      NewJSONCommandSerializer(),
        retryPolicy:     NewExponentialBackoffRetry(config.RetryConfig),
        deadLetterQueue: NewDeadLetterQueue(config.DLQConfig),
        metrics:         NewCommandBusMetrics(),
    }
}

func (dcb *DistributedCommandBus) Send(ctx context.Context, command Command) error {
    startTime := time.Now()
    defer func() {
        dcb.metrics.RecordCommandDuration(command.GetType(), time.Since(startTime))
    }()
    
    // Validate command
    if err := command.Validate(); err != nil {
        dcb.metrics.RecordCommandValidationError(command.GetType())
        return fmt.Errorf("command validation failed: %w", err)
    }
    
    // Apply middleware
    return dcb.executeWithMiddleware(ctx, command, func() error {
        return dcb.sendCommand(ctx, command)
    })
}

func (dcb *DistributedCommandBus) executeWithMiddleware(ctx context.Context, command Command, handler func() error) error {
    if len(dcb.middleware) == 0 {
        return handler()
    }
    
    // Build middleware chain
    var execute func(int) error
    execute = func(index int) error {
        if index >= len(dcb.middleware) {
            return handler()
        }
        
        return dcb.middleware[index].Execute(ctx, command, func() error {
            return execute(index + 1)
        })
    }
    
    return execute(0)
}

func (dcb *DistributedCommandBus) sendCommand(ctx context.Context, command Command) error {
    dcb.mu.RLock()
    handler, hasLocal := dcb.localHandlers[command.GetType()]
    dcb.mu.RUnlock()
    
    if hasLocal {
        // Handle locally
        dcb.metrics.RecordLocalCommand(command.GetType())
        return handler.Handle(ctx, command)
    }
    
    // Send to distributed queue
    dcb.metrics.RecordDistributedCommand(command.GetType())
    return dcb.sendToQueue(ctx, command)
}

func (dcb *DistributedCommandBus) sendToQueue(ctx context.Context, command Command) error {
    message := CommandMessage{
        ID:        generateMessageID(),
        Type:      command.GetType(),
        Payload:   dcb.serializer.Serialize(command),
        Metadata:  command.GetMetadata(),
        Timestamp: time.Now(),
        RetryCount: 0,
    }
    
    // Send with retry policy
    return dcb.retryPolicy.Execute(func() error {
        return dcb.messageQueue.Publish(ctx, command.GetType(), message)
    })
}

// Advanced query bus with caching and sharding
type DistributedQueryBus struct {
    localHandlers  map[string]QueryHandler
    shardManager   QueryShardManager
    cache          QueryCache
    loadBalancer   QueryLoadBalancer
    circuitBreaker QueryCircuitBreaker
    metrics        QueryBusMetrics
    mu             sync.RWMutex
}

type QueryShardManager interface {
    GetShard(query Query) (ShardInfo, error)
    RouteQuery(ctx context.Context, query Query, shard ShardInfo) (interface{}, error)
}

type QueryCache interface {
    Get(ctx context.Context, cacheKey string) (interface{}, bool)
    Set(ctx context.Context, cacheKey string, result interface{}, ttl time.Duration) error
    Invalidate(ctx context.Context, pattern string) error
}

func NewDistributedQueryBus(config QueryBusConfig) *DistributedQueryBus {
    return &DistributedQueryBus{
        localHandlers:  make(map[string]QueryHandler),
        shardManager:   NewConsistentHashShardManager(config.ShardConfig),
        cache:          NewDistributedQueryCache(config.CacheConfig),
        loadBalancer:   NewRoundRobinLoadBalancer(),
        circuitBreaker: NewQueryCircuitBreaker(config.CircuitBreakerConfig),
        metrics:        NewQueryBusMetrics(),
    }
}

func (dqb *DistributedQueryBus) Execute(ctx context.Context, query Query) (interface{}, error) {
    startTime := time.Now()
    defer func() {
        dqb.metrics.RecordQueryDuration(query.GetType(), time.Since(startTime))
    }()
    
    // Validate query
    if err := query.Validate(); err != nil {
        dqb.metrics.RecordQueryValidationError(query.GetType())
        return nil, fmt.Errorf("query validation failed: %w", err)
    }
    
    // Check cache first
    cacheKey := dqb.generateCacheKey(query)
    if cached, found := dqb.cache.Get(ctx, cacheKey); found {
        dqb.metrics.RecordCacheHit(query.GetType())
        return cached, nil
    }
    
    dqb.metrics.RecordCacheMiss(query.GetType())
    
    // Execute query with circuit breaker
    result, err := dqb.circuitBreaker.Execute(func() (interface{}, error) {
        return dqb.executeQuery(ctx, query)
    })
    
    if err != nil {
        return nil, err
    }
    
    // Cache successful results
    cacheTTL := dqb.getCacheTTL(query)
    if err := dqb.cache.Set(ctx, cacheKey, result, cacheTTL); err != nil {
        // Log cache error but don't fail the query
        dqb.metrics.RecordCacheError()
    }
    
    return result, nil
}

func (dqb *DistributedQueryBus) executeQuery(ctx context.Context, query Query) (interface{}, error) {
    dqb.mu.RLock()
    handler, hasLocal := dqb.localHandlers[query.GetType()]
    dqb.mu.RUnlock()
    
    if hasLocal {
        // Handle locally
        dqb.metrics.RecordLocalQuery(query.GetType())
        return handler.Handle(ctx, query)
    }
    
    // Route to appropriate shard
    shard, err := dqb.shardManager.GetShard(query)
    if err != nil {
        return nil, fmt.Errorf("failed to determine shard: %w", err)
    }
    
    dqb.metrics.RecordDistributedQuery(query.GetType())
    return dqb.shardManager.RouteQuery(ctx, query, shard)
}

// Read model synchronization
type ReadModelSynchronizer struct {
    eventStore      EventStore
    readModelStore  ReadModelRepository
    projections     []Projection
    checkpointStore CheckpointStore
    eventProcessor  EventProcessor
    batchProcessor  BatchProcessor
    errorHandler    SyncErrorHandler
}

type Projection interface {
    GetName() string
    GetHandledEvents() []string
    Handle(ctx context.Context, event Event) error
    Initialize(ctx context.Context) error
    Reset(ctx context.Context) error
}

type CheckpointStore interface {
    SaveCheckpoint(ctx context.Context, projectionName string, position int64) error
    LoadCheckpoint(ctx context.Context, projectionName string) (int64, error)
}

func NewReadModelSynchronizer(config SyncConfig) *ReadModelSynchronizer {
    return &ReadModelSynchronizer{
        eventStore:      config.EventStore,
        readModelStore:  config.ReadModelStore,
        checkpointStore: config.CheckpointStore,
        eventProcessor:  NewParallelEventProcessor(config.ProcessorConfig),
        batchProcessor:  NewBatchProcessor(config.BatchConfig),
        errorHandler:    NewRetryErrorHandler(config.ErrorConfig),
    }
}

func (rms *ReadModelSynchronizer) StartSynchronization(ctx context.Context) error {
    for _, projection := range rms.projections {
        go rms.synchronizeProjection(ctx, projection)
    }
    return nil
}

func (rms *ReadModelSynchronizer) synchronizeProjection(ctx context.Context, projection Projection) {
    // Load checkpoint
    checkpoint, err := rms.checkpointStore.LoadCheckpoint(ctx, projection.GetName())
    if err != nil {
        // Start from beginning if no checkpoint
        checkpoint = 0
    }
    
    // Subscribe to events from checkpoint
    eventStream, err := rms.eventStore.Subscribe(ctx, "", checkpoint)
    if err != nil {
        rms.errorHandler.HandleError(fmt.Errorf("failed to subscribe to events: %w", err))
        return
    }
    
    // Process events in batches
    eventBatch := make([]Event, 0, rms.batchProcessor.GetBatchSize())
    
    for {
        select {
        case event := <-eventStream:
            // Check if projection handles this event type
            if rms.projectionHandlesEvent(projection, event.EventType) {
                eventBatch = append(eventBatch, event)
                
                // Process batch when full
                if len(eventBatch) >= rms.batchProcessor.GetBatchSize() {
                    if err := rms.processBatch(ctx, projection, eventBatch); err != nil {
                        rms.errorHandler.HandleError(err)
                        continue
                    }
                    eventBatch = eventBatch[:0]
                }
            }
            
        case <-ctx.Done():
            // Process remaining events before shutting down
            if len(eventBatch) > 0 {
                rms.processBatch(ctx, projection, eventBatch)
            }
            return
            
        case <-time.After(5 * time.Second):
            // Process partial batch after timeout
            if len(eventBatch) > 0 {
                if err := rms.processBatch(ctx, projection, eventBatch); err != nil {
                    rms.errorHandler.HandleError(err)
                    continue
                }
                eventBatch = eventBatch[:0]
            }
        }
    }
}

func (rms *ReadModelSynchronizer) processBatch(ctx context.Context, projection Projection, events []Event) error {
    // Process events in parallel if possible
    if parallelProjection, ok := projection.(ParallelProjection); ok {
        return rms.eventProcessor.ProcessParallel(ctx, parallelProjection, events)
    }
    
    // Sequential processing
    for _, event := range events {
        if err := projection.Handle(ctx, event); err != nil {
            return fmt.Errorf("failed to handle event %s: %w", event.ID, err)
        }
        
        // Save checkpoint after each event
        if err := rms.checkpointStore.SaveCheckpoint(ctx, projection.GetName(), event.Version); err != nil {
            return fmt.Errorf("failed to save checkpoint: %w", err)
        }
    }
    
    return nil
}

// Example CQRS implementation for Order domain
type OrderCommandHandler struct {
    repository    OrderRepository
    eventBus      EventBus
    validator     OrderValidator
    domainService OrderDomainService
}

// Commands
type CreateOrderCommand struct {
    ID         string      `json:"id"`
    CustomerID string      `json:"customer_id"`
    Items      []OrderItem `json:"items"`
    Metadata   map[string]interface{} `json:"metadata"`
}

func (c CreateOrderCommand) GetID() string { return c.ID }
func (c CreateOrderCommand) GetType() string { return "CreateOrder" }
func (c CreateOrderCommand) GetMetadata() map[string]interface{} { return c.Metadata }

func (c CreateOrderCommand) Validate() error {
    if c.CustomerID == "" {
        return fmt.Errorf("customer ID is required")
    }
    if len(c.Items) == 0 {
        return fmt.Errorf("order must have at least one item")
    }
    return nil
}

type ConfirmOrderCommand struct {
    ID       string                 `json:"id"`
    OrderID  string                 `json:"order_id"`
    Metadata map[string]interface{} `json:"metadata"`
}

func (c ConfirmOrderCommand) GetID() string { return c.ID }
func (c ConfirmOrderCommand) GetType() string { return "ConfirmOrder" }
func (c ConfirmOrderCommand) GetMetadata() map[string]interface{} { return c.Metadata }

func (c ConfirmOrderCommand) Validate() error {
    if c.OrderID == "" {
        return fmt.Errorf("order ID is required")
    }
    return nil
}

func (och *OrderCommandHandler) Handle(ctx context.Context, command Command) error {
    switch cmd := command.(type) {
    case CreateOrderCommand:
        return och.handleCreateOrder(ctx, cmd)
    case ConfirmOrderCommand:
        return och.handleConfirmOrder(ctx, cmd)
    default:
        return fmt.Errorf("unsupported command type: %s", command.GetType())
    }
}

func (och *OrderCommandHandler) handleCreateOrder(ctx context.Context, cmd CreateOrderCommand) error {
    // Validate business rules
    if err := och.validator.ValidateOrderCreation(ctx, cmd.CustomerID, cmd.Items); err != nil {
        return fmt.Errorf("order validation failed: %w", err)
    }
    
    // Create order aggregate
    order, err := NewOrder(cmd.CustomerID, cmd.Items)
    if err != nil {
        return fmt.Errorf("failed to create order: %w", err)
    }
    
    // Apply domain services
    if err := och.domainService.ApplyPricingRules(ctx, order); err != nil {
        return fmt.Errorf("failed to apply pricing: %w", err)
    }
    
    // Save to repository
    if err := och.repository.Save(ctx, order); err != nil {
        return fmt.Errorf("failed to save order: %w", err)
    }
    
    return nil
}

func (och *OrderCommandHandler) handleConfirmOrder(ctx context.Context, cmd ConfirmOrderCommand) error {
    // Load order aggregate
    order, err := och.repository.Load(ctx, cmd.OrderID)
    if err != nil {
        return fmt.Errorf("failed to load order: %w", err)
    }
    
    // Execute business operation
    if err := order.Confirm(); err != nil {
        return fmt.Errorf("failed to confirm order: %w", err)
    }
    
    // Save changes
    if err := och.repository.Save(ctx, order); err != nil {
        return fmt.Errorf("failed to save order: %w", err)
    }
    
    return nil
}

// Query handlers
type OrderQueryHandler struct {
    readModelRepo ReadModelRepository
    cache         QueryCache
    metrics       QueryMetrics
}

// Queries
type GetOrderQuery struct {
    ID      string                 `json:"id"`
    OrderID string                 `json:"order_id"`
    Criteria map[string]interface{} `json:"criteria"`
}

func (q GetOrderQuery) GetID() string { return q.ID }
func (q GetOrderQuery) GetType() string { return "GetOrder" }
func (q GetOrderQuery) GetCriteria() map[string]interface{} { return q.Criteria }

func (q GetOrderQuery) Validate() error {
    if q.OrderID == "" {
        return fmt.Errorf("order ID is required")
    }
    return nil
}

type GetCustomerOrdersQuery struct {
    ID         string                 `json:"id"`
    CustomerID string                 `json:"customer_id"`
    PageSize   int                    `json:"page_size"`
    PageToken  string                 `json:"page_token"`
    Criteria   map[string]interface{} `json:"criteria"`
}

func (q GetCustomerOrdersQuery) GetID() string { return q.ID }
func (q GetCustomerOrdersQuery) GetType() string { return "GetCustomerOrders" }
func (q GetCustomerOrdersQuery) GetCriteria() map[string]interface{} { return q.Criteria }

func (q GetCustomerOrdersQuery) Validate() error {
    if q.CustomerID == "" {
        return fmt.Errorf("customer ID is required")
    }
    if q.PageSize <= 0 || q.PageSize > 100 {
        return fmt.Errorf("page size must be between 1 and 100")
    }
    return nil
}

func (oqh *OrderQueryHandler) Handle(ctx context.Context, query Query) (interface{}, error) {
    switch q := query.(type) {
    case GetOrderQuery:
        return oqh.handleGetOrder(ctx, q)
    case GetCustomerOrdersQuery:
        return oqh.handleGetCustomerOrders(ctx, q)
    default:
        return nil, fmt.Errorf("unsupported query type: %s", query.GetType())
    }
}

func (oqh *OrderQueryHandler) handleGetOrder(ctx context.Context, query GetOrderQuery) (interface{}, error) {
    orderSummary, err := oqh.readModelRepo.GetOrderSummary(ctx, query.OrderID)
    if err != nil {
        return nil, fmt.Errorf("failed to get order: %w", err)
    }
    
    return orderSummary, nil
}

func (oqh *OrderQueryHandler) handleGetCustomerOrders(ctx context.Context, query GetCustomerOrdersQuery) (interface{}, error) {
    orders, nextToken, err := oqh.readModelRepo.GetCustomerOrders(ctx, 
        query.CustomerID, query.PageSize, query.PageToken)
    if err != nil {
        return nil, fmt.Errorf("failed to get customer orders: %w", err)
    }
    
    return CustomerOrdersResponse{
        Orders:        orders,
        NextPageToken: nextToken,
        TotalCount:    len(orders),
    }, nil
}

type CustomerOrdersResponse struct {
    Orders        []OrderSummary `json:"orders"`
    NextPageToken string         `json:"next_page_token"`
    TotalCount    int            `json:"total_count"`
}
```

---

## 🎭 **Saga Orchestration Patterns**

### **Advanced Saga Implementation**

```go
package saga

import (
    "context"
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Saga orchestration for distributed transactions
type SagaOrchestrator struct {
    sagaStore       SagaStore
    commandBus      CommandBus
    eventBus        EventBus
    compensationMgr CompensationManager
    timeoutMgr      TimeoutManager
    sagaDefinitions map[string]SagaDefinition
    stepExecutors   map[string]StepExecutor
    mu              sync.RWMutex
}

type SagaDefinition struct {
    Name            string        `json:"name"`
    Steps           []SagaStep    `json:"steps"`
    CompensationSteps []SagaStep  `json:"compensation_steps"`
    Timeout         time.Duration `json:"timeout"`
    RetryPolicy     RetryPolicy   `json:"retry_policy"`
    IsolationLevel  IsolationLevel `json:"isolation_level"`
}

type SagaStep struct {
    Name            string                 `json:"name"`
    Command         string                 `json:"command"`
    Parameters      map[string]interface{} `json:"parameters"`
    CompensationCmd string                 `json:"compensation_command"`
    Timeout         time.Duration          `json:"timeout"`
    RetryCount      int                    `json:"retry_count"`
    CriticalStep    bool                   `json:"critical_step"`
    ParallelGroup   string                 `json:"parallel_group,omitempty"`
}

type SagaInstance struct {
    ID              string                 `json:"id"`
    SagaType        string                 `json:"saga_type"`
    Status          SagaStatus             `json:"status"`
    CurrentStep     int                    `json:"current_step"`
    StepResults     map[string]interface{} `json:"step_results"`
    CompensationStack []string             `json:"compensation_stack"`
    StartTime       time.Time              `json:"start_time"`
    EndTime         time.Time              `json:"end_time,omitempty"`
    Context         map[string]interface{} `json:"context"`
    ErrorDetails    *SagaError             `json:"error_details,omitempty"`
    Version         int64                  `json:"version"`
}

type SagaStatus int

const (
    SagaStarted SagaStatus = iota
    SagaRunning
    SagaCompleted
    SagaFailed
    SagaCompensating
    SagaCompensated
    SagaAborted
)

type SagaError struct {
    Step        string    `json:"step"`
    Message     string    `json:"message"`
    ErrorCode   string    `json:"error_code"`
    Retryable   bool      `json:"retryable"`
    OccurredAt  time.Time `json:"occurred_at"`
}

func NewSagaOrchestrator(config SagaConfig) *SagaOrchestrator {
    return &SagaOrchestrator{
        sagaStore:       NewDistributedSagaStore(config.StoreConfig),
        commandBus:      config.CommandBus,
        eventBus:        config.EventBus,
        compensationMgr: NewCompensationManager(),
        timeoutMgr:      NewTimeoutManager(),
        sagaDefinitions: make(map[string]SagaDefinition),
        stepExecutors:   make(map[string]StepExecutor),
    }
}

func (so *SagaOrchestrator) StartSaga(ctx context.Context, sagaType string, initialData map[string]interface{}) (*SagaInstance, error) {
    so.mu.RLock()
    definition, exists := so.sagaDefinitions[sagaType]
    so.mu.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("saga definition not found: %s", sagaType)
    }
    
    instance := &SagaInstance{
        ID:              generateSagaID(),
        SagaType:        sagaType,
        Status:          SagaStarted,
        CurrentStep:     0,
        StepResults:     make(map[string]interface{}),
        CompensationStack: []string{},
        StartTime:       time.Now(),
        Context:         initialData,
        Version:         1,
    }
    
    // Save initial saga instance
    if err := so.sagaStore.SaveSaga(ctx, instance); err != nil {
        return nil, fmt.Errorf("failed to save saga instance: %w", err)
    }
    
    // Start saga execution
    go so.executeSaga(ctx, instance, definition)
    
    return instance, nil
}

func (so *SagaOrchestrator) executeSaga(ctx context.Context, instance *SagaInstance, definition SagaDefinition) {
    defer func() {
        if r := recover(); r != nil {
            so.handleSagaPanic(ctx, instance, r)
        }
    }()
    
    // Set up timeout
    sagaCtx, cancel := context.WithTimeout(ctx, definition.Timeout)
    defer cancel()
    
    instance.Status = SagaRunning
    so.sagaStore.SaveSaga(sagaCtx, instance)
    
    // Execute steps
    for i, step := range definition.Steps {
        instance.CurrentStep = i
        
        // Check for parallel execution
        if step.ParallelGroup != "" {
            if err := so.executeParallelSteps(sagaCtx, instance, definition, step.ParallelGroup); err != nil {
                so.handleSagaFailure(sagaCtx, instance, definition, err)
                return
            }
            continue
        }
        
        // Execute single step
        if err := so.executeStep(sagaCtx, instance, step); err != nil {
            so.handleSagaFailure(sagaCtx, instance, definition, err)
            return
        }
        
        // Save progress
        instance.Version++
        if err := so.sagaStore.SaveSaga(sagaCtx, instance); err != nil {
            so.handleSagaFailure(sagaCtx, instance, definition, err)
            return
        }
    }
    
    // Saga completed successfully
    so.completeSaga(sagaCtx, instance)
}

func (so *SagaOrchestrator) executeStep(ctx context.Context, instance *SagaInstance, step SagaStep) error {
    stepCtx, cancel := context.WithTimeout(ctx, step.Timeout)
    defer cancel()
    
    executor, exists := so.stepExecutors[step.Command]
    if !exists {
        return fmt.Errorf("step executor not found: %s", step.Command)
    }
    
    // Prepare step context
    stepContext := StepContext{
        SagaID:      instance.ID,
        StepName:    step.Name,
        Parameters:  step.Parameters,
        SagaContext: instance.Context,
        StepResults: instance.StepResults,
    }
    
    // Execute with retry
    var lastErr error
    for attempt := 0; attempt <= step.RetryCount; attempt++ {
        result, err := executor.Execute(stepCtx, stepContext)
        if err == nil {
            // Step succeeded
            instance.StepResults[step.Name] = result
            
            // Add to compensation stack if compensation command exists
            if step.CompensationCmd != "" {
                instance.CompensationStack = append(instance.CompensationStack, step.Name)
            }
            
            return nil
        }
        
        lastErr = err
        
        // Check if error is retryable
        if stepErr, ok := err.(*StepExecutionError); ok && !stepErr.Retryable {
            break
        }
        
        // Wait before retry
        if attempt < step.RetryCount {
            backoff := time.Duration(attempt+1) * time.Second
            time.Sleep(backoff)
        }
    }
    
    return fmt.Errorf("step execution failed after %d attempts: %w", step.RetryCount+1, lastErr)
}

func (so *SagaOrchestrator) executeParallelSteps(ctx context.Context, instance *SagaInstance, definition SagaDefinition, parallelGroup string) error {
    // Find all steps in the parallel group
    var parallelSteps []SagaStep
    for _, step := range definition.Steps {
        if step.ParallelGroup == parallelGroup {
            parallelSteps = append(parallelSteps, step)
        }
    }
    
    if len(parallelSteps) == 0 {
        return nil
    }
    
    // Execute steps in parallel
    results := make(chan StepResult, len(parallelSteps))
    var wg sync.WaitGroup
    
    for _, step := range parallelSteps {
        wg.Add(1)
        go func(s SagaStep) {
            defer wg.Done()
            
            err := so.executeStep(ctx, instance, s)
            results <- StepResult{
                StepName: s.Name,
                Error:    err,
            }
        }(step)
    }
    
    wg.Wait()
    close(results)
    
    // Check results
    var failures []string
    for result := range results {
        if result.Error != nil {
            failures = append(failures, fmt.Sprintf("%s: %v", result.StepName, result.Error))
        }
    }
    
    if len(failures) > 0 {
        return fmt.Errorf("parallel step failures: %v", failures)
    }
    
    return nil
}

func (so *SagaOrchestrator) handleSagaFailure(ctx context.Context, instance *SagaInstance, definition SagaDefinition, err error) {
    instance.Status = SagaFailed
    instance.ErrorDetails = &SagaError{
        Step:       definition.Steps[instance.CurrentStep].Name,
        Message:    err.Error(),
        OccurredAt: time.Now(),
        Retryable:  false,
    }
    
    // Start compensation
    so.startCompensation(ctx, instance, definition)
}

func (so *SagaOrchestrator) startCompensation(ctx context.Context, instance *SagaInstance, definition SagaDefinition) {
    instance.Status = SagaCompensating
    so.sagaStore.SaveSaga(ctx, instance)
    
    // Execute compensation commands in reverse order
    for i := len(instance.CompensationStack) - 1; i >= 0; i-- {
        stepName := instance.CompensationStack[i]
        
        // Find compensation command for this step
        var compensationCmd string
        for _, step := range definition.Steps {
            if step.Name == stepName {
                compensationCmd = step.CompensationCmd
                break
            }
        }
        
        if compensationCmd == "" {
            continue
        }
        
        // Execute compensation
        if err := so.executeCompensation(ctx, instance, stepName, compensationCmd); err != nil {
            // Log compensation error but continue with other compensations
            fmt.Printf("Compensation failed for step %s: %v\n", stepName, err)
        }
    }
    
    instance.Status = SagaCompensated
    instance.EndTime = time.Now()
    so.sagaStore.SaveSaga(ctx, instance)
}

func (so *SagaOrchestrator) executeCompensation(ctx context.Context, instance *SagaInstance, stepName, compensationCmd string) error {
    executor, exists := so.stepExecutors[compensationCmd]
    if !exists {
        return fmt.Errorf("compensation executor not found: %s", compensationCmd)
    }
    
    stepContext := StepContext{
        SagaID:      instance.ID,
        StepName:    stepName,
        Parameters:  make(map[string]interface{}),
        SagaContext: instance.Context,
        StepResults: instance.StepResults,
    }
    
    _, err := executor.Execute(ctx, stepContext)
    return err
}

// Choreography-based saga (event-driven)
type SagaChoreographer struct {
    eventBus        EventBus
    sagaStore       SagaStore
    policyHandlers  map[string]PolicyHandler
    eventHandlers   map[string]EventHandler
    mu              sync.RWMutex
}

type PolicyHandler interface {
    Handle(ctx context.Context, event Event) ([]Command, error)
    GetHandledEvents() []string
}

func NewSagaChoreographer(eventBus EventBus, sagaStore SagaStore) *SagaChoreographer {
    return &SagaChoreographer{
        eventBus:       eventBus,
        sagaStore:      sagaStore,
        policyHandlers: make(map[string]PolicyHandler),
        eventHandlers:  make(map[string]EventHandler),
    }
}

func (sc *SagaChoreographer) RegisterPolicy(eventType string, handler PolicyHandler) {
    sc.mu.Lock()
    defer sc.mu.Unlock()
    sc.policyHandlers[eventType] = handler
}

func (sc *SagaChoreographer) HandleEvent(ctx context.Context, event Event) error {
    sc.mu.RLock()
    handler, exists := sc.policyHandlers[event.EventType]
    sc.mu.RUnlock()
    
    if !exists {
        return nil // No policy for this event type
    }
    
    // Generate commands based on policy
    commands, err := handler.Handle(ctx, event)
    if err != nil {
        return fmt.Errorf("policy handler failed: %w", err)
    }
    
    // Execute generated commands
    for _, command := range commands {
        if err := sc.eventBus.Publish(ctx, CommandEvent{
            CommandType: command.GetType(),
            CommandData: command,
            CorrelationID: event.CorrelationID,
        }); err != nil {
            return fmt.Errorf("failed to publish command: %w", err)
        }
    }
    
    return nil
}

// Example: Order Processing Saga
type OrderProcessingSaga struct {
    orchestrator *SagaOrchestrator
}

func (ops *OrderProcessingSaga) CreateOrderProcessingDefinition() SagaDefinition {
    return SagaDefinition{
        Name:    "OrderProcessing",
        Timeout: 30 * time.Minute,
        Steps: []SagaStep{
            {
                Name:            "ValidateOrder",
                Command:         "ValidateOrderCommand",
                CompensationCmd: "RejectOrderCommand",
                Timeout:         30 * time.Second,
                RetryCount:      2,
                CriticalStep:    true,
            },
            {
                Name:            "ReserveInventory",
                Command:         "ReserveInventoryCommand",
                CompensationCmd: "ReleaseInventoryCommand",
                Timeout:         45 * time.Second,
                RetryCount:      3,
                CriticalStep:    true,
            },
            {
                Name:            "ProcessPayment",
                Command:         "ProcessPaymentCommand",
                CompensationCmd: "RefundPaymentCommand",
                Timeout:         60 * time.Second,
                RetryCount:      2,
                CriticalStep:    true,
            },
            {
                Name:         "UpdateInventory",
                Command:      "UpdateInventoryCommand",
                Timeout:      30 * time.Second,
                RetryCount:   1,
                ParallelGroup: "finalization",
            },
            {
                Name:         "SendConfirmationEmail",
                Command:      "SendEmailCommand",
                Timeout:      15 * time.Second,
                RetryCount:   3,
                ParallelGroup: "finalization",
            },
            {
                Name:         "UpdateOrderStatus",
                Command:      "UpdateOrderStatusCommand",
                Timeout:      10 * time.Second,
                RetryCount:   1,
                ParallelGroup: "finalization",
            },
        },
        RetryPolicy: ExponentialBackoffRetry{
            InitialDelay: time.Second,
            MaxDelay:     time.Minute,
            Multiplier:   2.0,
        },
    }
}

// Step executors
type ValidateOrderExecutor struct {
    orderService OrderService
    validator    OrderValidator
}

func (voe *ValidateOrderExecutor) Execute(ctx context.Context, stepCtx StepContext) (interface{}, error) {
    orderID := stepCtx.Parameters["order_id"].(string)
    
    // Validate order details
    order, err := voe.orderService.GetOrder(ctx, orderID)
    if err != nil {
        return nil, &StepExecutionError{
            Message:   fmt.Sprintf("failed to get order: %v", err),
            Retryable: true,
        }
    }
    
    if err := voe.validator.ValidateOrder(ctx, order); err != nil {
        return nil, &StepExecutionError{
            Message:   fmt.Sprintf("order validation failed: %v", err),
            Retryable: false,
        }
    }
    
    return ValidationResult{
        OrderID:    orderID,
        ValidatedAt: time.Now(),
        TotalAmount: order.TotalAmount,
    }, nil
}

type ReserveInventoryExecutor struct {
    inventoryService InventoryService
}

func (rie *ReserveInventoryExecutor) Execute(ctx context.Context, stepCtx StepContext) (interface{}, error) {
    orderID := stepCtx.Parameters["order_id"].(string)
    items := stepCtx.Parameters["items"].([]OrderItem)
    
    reservationID, err := rie.inventoryService.ReserveItems(ctx, orderID, items)
    if err != nil {
        if IsInventoryNotAvailable(err) {
            return nil, &StepExecutionError{
                Message:   fmt.Sprintf("inventory not available: %v", err),
                Retryable: false,
            }
        }
        
        return nil, &StepExecutionError{
            Message:   fmt.Sprintf("failed to reserve inventory: %v", err),
            Retryable: true,
        }
    }
    
    return InventoryReservationResult{
        ReservationID: reservationID,
        OrderID:       orderID,
        ReservedAt:    time.Now(),
    }, nil
}

type ProcessPaymentExecutor struct {
    paymentService PaymentService
}

func (ppe *ProcessPaymentExecutor) Execute(ctx context.Context, stepCtx StepContext) (interface{}, error) {
    orderID := stepCtx.Parameters["order_id"].(string)
    amount := stepCtx.StepResults["ValidateOrder"].(ValidationResult).TotalAmount
    customerID := stepCtx.Parameters["customer_id"].(string)
    
    paymentResult, err := ppe.paymentService.ProcessPayment(ctx, PaymentRequest{
        OrderID:    orderID,
        CustomerID: customerID,
        Amount:     amount,
        Currency:   "USD",
    })
    
    if err != nil {
        if IsPaymentDeclined(err) {
            return nil, &StepExecutionError{
                Message:   fmt.Sprintf("payment declined: %v", err),
                Retryable: false,
            }
        }
        
        return nil, &StepExecutionError{
            Message:   fmt.Sprintf("payment processing failed: %v", err),
            Retryable: true,
        }
    }
    
    return PaymentResult{
        TransactionID: paymentResult.TransactionID,
        OrderID:       orderID,
        Amount:        amount,
        ProcessedAt:   time.Now(),
    }, nil
}

// Saga monitoring and management
type SagaMonitor struct {
    sagaStore  SagaStore
    metrics    SagaMetrics
    alerting   AlertingService
}

func (sm *SagaMonitor) MonitorSagas(ctx context.Context) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            sm.checkRunningS agas(ctx)
            sm.checkFailedSagas(ctx)
            sm.updateMetrics(ctx)
            
        case <-ctx.Done():
            return
        }
    }
}

func (sm *SagaMonitor) checkRunningS agas(ctx context.Context) {
    runningSagas, err := sm.sagaStore.GetSagasByStatus(ctx, SagaRunning)
    if err != nil {
        return
    }
    
    for _, saga := range runningSagas {
        // Check for timeouts
        if time.Since(saga.StartTime) > 1*time.Hour {
            sm.alerting.SendAlert(Alert{
                Type:    "saga_timeout",
                Message: fmt.Sprintf("Saga %s has been running for over 1 hour", saga.ID),
                Severity: "high",
                Data: map[string]interface{}{
                    "saga_id":   saga.ID,
                    "saga_type": saga.SagaType,
                    "duration":  time.Since(saga.StartTime).String(),
                },
            })
        }
    }
}

func (sm *SagaMonitor) checkFailedSagas(ctx context.Context) {
    failedSagas, err := sm.sagaStore.GetSagasByStatus(ctx, SagaFailed)
    if err != nil {
        return
    }
    
    sm.metrics.RecordFailedSagas(len(failedSagas))
    
    for _, saga := range failedSagas {
        sm.alerting.SendAlert(Alert{
            Type:    "saga_failure",
            Message: fmt.Sprintf("Saga %s failed: %s", saga.ID, saga.ErrorDetails.Message),
            Severity: "critical",
            Data: map[string]interface{}{
                "saga_id":     saga.ID,
                "saga_type":   saga.SagaType,
                "error_step":  saga.ErrorDetails.Step,
                "error_message": saga.ErrorDetails.Message,
            },
        })
    }
}
```

---

## 🚀 **Advanced Caching Strategies**

### **Multi-Level Caching System**

```go
package caching

import (
    "context"
    "encoding/json"
    "fmt"
    "hash/fnv"
    "sync"
    "time"
)

// Multi-level cache hierarchy
type MultiLevelCache struct {
    l1Cache    LocalCache    // In-memory cache (fastest)
    l2Cache    LocalCache    // SSD-backed cache (fast)
    l3Cache    DistributedCache // Network cache (Redis/Memcached)
    l4Cache    PersistentCache  // Database cache (slowest)
    
    policies   CachePolicies
    metrics    CacheMetrics
    promoter   CachePromoter
    eviction   EvictionManager
    mu         sync.RWMutex
}

type CacheLevel int

const (
    L1Cache CacheLevel = iota
    L2Cache
    L3Cache
    L4Cache
)

type CacheEntry struct {
    Key         string                 `json:"key"`
    Value       interface{}            `json:"value"`
    Level       CacheLevel             `json:"level"`
    Size        int64                  `json:"size"`
    TTL         time.Duration          `json:"ttl"`
    CreatedAt   time.Time              `json:"created_at"`
    LastAccessed time.Time             `json:"last_accessed"`
    AccessCount int64                  `json:"access_count"`
    Metadata    map[string]interface{} `json:"metadata"`
}

type CachePolicies struct {
    L1Policy    CachePolicy `json:"l1_policy"`
    L2Policy    CachePolicy `json:"l2_policy"`
    L3Policy    CachePolicy `json:"l3_policy"`
    L4Policy    CachePolicy `json:"l4_policy"`
    PromotionPolicy PromotionPolicy `json:"promotion_policy"`
    EvictionPolicy  EvictionPolicy  `json:"eviction_policy"`
}

type CachePolicy struct {
    MaxSize     int64         `json:"max_size"`
    MaxEntries  int           `json:"max_entries"`
    DefaultTTL  time.Duration `json:"default_ttl"`
    Strategy    EvictionStrategy `json:"strategy"`
}

type EvictionStrategy string

const (
    LRUEviction  EvictionStrategy = "lru"
    LFUEviction  EvictionStrategy = "lfu"
    TTLEviction  EvictionStrategy = "ttl"
    SizeEviction EvictionStrategy = "size"
    AdaptiveEviction EvictionStrategy = "adaptive"
)

func NewMultiLevelCache(config CacheConfig) *MultiLevelCache {
    return &MultiLevelCache{
        l1Cache:  NewInMemoryCache(config.L1Config),
        l2Cache:  NewSSDCache(config.L2Config),
        l3Cache:  NewRedisCache(config.L3Config),
        l4Cache:  NewDatabaseCache(config.L4Config),
        policies: config.Policies,
        metrics:  NewCacheMetrics(),
        promoter: NewIntelligentPromoter(config.PromoterConfig),
        eviction: NewAdaptiveEvictionManager(config.EvictionConfig),
    }
}

func (mlc *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, bool) {
    startTime := time.Now()
    defer func() {
        mlc.metrics.RecordOperationDuration("get", time.Since(startTime))
    }()
    
    // Try L1 cache first (fastest)
    if value, found := mlc.l1Cache.Get(ctx, key); found {
        mlc.metrics.RecordHit(L1Cache)
        mlc.updateAccessStats(key, L1Cache)
        return value, true
    }
    
    // Try L2 cache
    if value, found := mlc.l2Cache.Get(ctx, key); found {
        mlc.metrics.RecordHit(L2Cache)
        mlc.updateAccessStats(key, L2Cache)
        
        // Promote to L1 if warranted
        if mlc.promoter.ShouldPromote(key, L2Cache, L1Cache) {
            mlc.l1Cache.Set(ctx, key, value, mlc.policies.L1Policy.DefaultTTL)
        }
        
        return value, true
    }
    
    // Try L3 cache
    if value, found := mlc.l3Cache.Get(ctx, key); found {
        mlc.metrics.RecordHit(L3Cache)
        mlc.updateAccessStats(key, L3Cache)
        
        // Consider promotion to higher levels
        if mlc.promoter.ShouldPromote(key, L3Cache, L2Cache) {
            mlc.l2Cache.Set(ctx, key, value, mlc.policies.L2Policy.DefaultTTL)
        }
        if mlc.promoter.ShouldPromote(key, L3Cache, L1Cache) {
            mlc.l1Cache.Set(ctx, key, value, mlc.policies.L1Policy.DefaultTTL)
        }
        
        return value, true
    }
    
    // Try L4 cache (slowest but most comprehensive)
    if value, found := mlc.l4Cache.Get(ctx, key); found {
        mlc.metrics.RecordHit(L4Cache)
        mlc.updateAccessStats(key, L4Cache)
        
        // Promote based on access patterns
        mlc.promoteFromL4(ctx, key, value)
        
        return value, true
    }
    
    // Cache miss at all levels
    mlc.metrics.RecordMiss()
    return nil, false
}

func (mlc *MultiLevelCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    startTime := time.Now()
    defer func() {
        mlc.metrics.RecordOperationDuration("set", time.Since(startTime))
    }()
    
    entry := &CacheEntry{
        Key:       key,
        Value:     value,
        TTL:       ttl,
        CreatedAt: time.Now(),
        LastAccessed: time.Now(),
        AccessCount: 1,
        Size:      mlc.calculateSize(value),
        Metadata:  make(map[string]interface{}),
    }
    
    // Determine optimal cache level for new entry
    targetLevel := mlc.determineOptimalLevel(entry)
    
    switch targetLevel {
    case L1Cache:
        return mlc.setInLevel(ctx, mlc.l1Cache, entry)
    case L2Cache:
        return mlc.setInLevel(ctx, mlc.l2Cache, entry)
    case L3Cache:
        return mlc.setInLevel(ctx, mlc.l3Cache, entry)
    case L4Cache:
        return mlc.setInLevel(ctx, mlc.l4Cache, entry)
    default:
        return fmt.Errorf("invalid cache level: %d", targetLevel)
    }
}

func (mlc *MultiLevelCache) determineOptimalLevel(entry *CacheEntry) CacheLevel {
    // Small, frequently accessed items go to L1
    if entry.Size < 1024 && mlc.promoter.IsHotData(entry.Key) {
        return L1Cache
    }
    
    // Medium-sized items with good locality go to L2
    if entry.Size < 10*1024 && mlc.promoter.HasGoodLocality(entry.Key) {
        return L2Cache
    }
    
    // Larger items or less frequently accessed go to L3
    if entry.Size < 100*1024 {
        return L3Cache
    }
    
    // Very large items go to L4
    return L4Cache
}

func (mlc *MultiLevelCache) promoteFromL4(ctx context.Context, key string, value interface{}) {
    accessPattern := mlc.promoter.GetAccessPattern(key)
    
    if accessPattern.Frequency > 10 && accessPattern.Recency < time.Hour {
        // Promote to L3
        mlc.l3Cache.Set(ctx, key, value, mlc.policies.L3Policy.DefaultTTL)
        
        if accessPattern.Frequency > 50 && accessPattern.Recency < 15*time.Minute {
            // Promote to L2
            mlc.l2Cache.Set(ctx, key, value, mlc.policies.L2Policy.DefaultTTL)
            
            if accessPattern.Frequency > 100 && accessPattern.Recency < 5*time.Minute {
                // Promote to L1
                mlc.l1Cache.Set(ctx, key, value, mlc.policies.L1Policy.DefaultTTL)
            }
        }
    }
}

// Intelligent cache promoter
type IntelligentPromoter struct {
    accessTracker  AccessTracker
    patternAnalyzer PatternAnalyzer
    predictor      AccessPredictor
    config         PromoterConfig
}

type AccessTracker struct {
    accessHistory map[string][]AccessEvent
    mu            sync.RWMutex
}

type AccessEvent struct {
    Timestamp time.Time `json:"timestamp"`
    Level     CacheLevel `json:"level"`
    Hit       bool      `json:"hit"`
}

type AccessPattern struct {
    Key           string        `json:"key"`
    Frequency     int64         `json:"frequency"`
    Recency       time.Duration `json:"recency"`
    Locality      float64       `json:"locality"`
    Seasonality   float64       `json:"seasonality"`
    Predictability float64      `json:"predictability"`
}

func (ip *IntelligentPromoter) ShouldPromote(key string, fromLevel, toLevel CacheLevel) bool {
    pattern := ip.GetAccessPattern(key)
    
    // Don't promote if target level is already at capacity
    if !ip.hasCapacity(toLevel) {
        return false
    }
    
    // Promotion criteria based on access patterns
    switch toLevel {
    case L1Cache:
        return pattern.Frequency > 50 && 
               pattern.Recency < 5*time.Minute &&
               pattern.Locality > 0.8
               
    case L2Cache:
        return pattern.Frequency > 10 && 
               pattern.Recency < 15*time.Minute &&
               pattern.Locality > 0.6
               
    case L3Cache:
        return pattern.Frequency > 3 && 
               pattern.Recency < time.Hour &&
               pattern.Locality > 0.4
               
    default:
        return false
    }
}

func (ip *IntelligentPromoter) GetAccessPattern(key string) AccessPattern {
    ip.accessTracker.mu.RLock()
    events := ip.accessTracker.accessHistory[key]
    ip.accessTracker.mu.RUnlock()
    
    if len(events) == 0 {
        return AccessPattern{Key: key}
    }
    
    // Calculate frequency (accesses per hour)
    now := time.Now()
    hourAgo := now.Add(-time.Hour)
    
    var recentAccesses int64
    var lastAccess time.Time
    
    for _, event := range events {
        if event.Timestamp.After(hourAgo) {
            recentAccesses++
        }
        if event.Timestamp.After(lastAccess) {
            lastAccess = event.Timestamp
        }
    }
    
    recency := now.Sub(lastAccess)
    locality := ip.patternAnalyzer.CalculateLocality(key, events)
    
    return AccessPattern{
        Key:         key,
        Frequency:   recentAccesses,
        Recency:     recency,
        Locality:    locality,
        Seasonality: ip.patternAnalyzer.CalculateSeasonality(events),
        Predictability: ip.predictor.GetPredictability(key),
    }
}

// Write-through and Write-behind strategies
type WriteStrategy interface {
    Write(ctx context.Context, key string, value interface{}) error
    WriteBatch(ctx context.Context, entries map[string]interface{}) error
}

type WriteThroughStrategy struct {
    cache       MultiLevelCache
    backend     PersistentStore
    consistency ConsistencyLevel
}

type WriteBehindStrategy struct {
    cache       MultiLevelCache
    backend     PersistentStore
    writeQueue  WriteQueue
    batchSize   int
    flushInterval time.Duration
    conflictResolver ConflictResolver
}

func (wbs *WriteBehindStrategy) Write(ctx context.Context, key string, value interface{}) error {
    // Write to cache immediately
    if err := wbs.cache.Set(ctx, key, value, time.Hour); err != nil {
        return err
    }
    
    // Queue for background write to persistent store
    writeOp := WriteOperation{
        Key:       key,
        Value:     value,
        Timestamp: time.Now(),
        Operation: WriteOpSet,
    }
    
    return wbs.writeQueue.Enqueue(writeOp)
}

func (wbs *WriteBehindStrategy) startBackgroundWriter(ctx context.Context) {
    ticker := time.NewTicker(wbs.flushInterval)
    defer ticker.Stop()
    
    batch := make([]WriteOperation, 0, wbs.batchSize)
    
    for {
        select {
        case <-ticker.C:
            // Flush current batch
            if len(batch) > 0 {
                wbs.flushBatch(ctx, batch)
                batch = batch[:0]
            }
            
        case <-ctx.Done():
            // Flush remaining operations before shutdown
            if len(batch) > 0 {
                wbs.flushBatch(ctx, batch)
            }
            return
        }
        
        // Drain queue into batch
        for len(batch) < wbs.batchSize {
            op, ok := wbs.writeQueue.Dequeue()
            if !ok {
                break
            }
            batch = append(batch, op)
        }
        
        // Flush if batch is full
        if len(batch) >= wbs.batchSize {
            wbs.flushBatch(ctx, batch)
            batch = batch[:0]
        }
    }
}

func (wbs *WriteBehindStrategy) flushBatch(ctx context.Context, batch []WriteOperation) {
    // Group operations by key to handle conflicts
    keyOps := make(map[string]WriteOperation)
    
    for _, op := range batch {
        existing, exists := keyOps[op.Key]
        if !exists || op.Timestamp.After(existing.Timestamp) {
            keyOps[op.Key] = op
        }
    }
    
    // Write to persistent store
    for _, op := range keyOps {
        if err := wbs.backend.Write(ctx, op.Key, op.Value); err != nil {
            // Handle write failure - could retry or send to DLQ
            wbs.handleWriteFailure(op, err)
        }
    }
}

// Cache warming and preloading
type CacheWarmer struct {
    cache         MultiLevelCache
    dataSource    DataSource
    predictor     AccessPredictor
    scheduler     WarmingScheduler
    loadBalancer  LoadBalancer
}

func (cw *CacheWarmer) WarmCache(ctx context.Context, strategy WarmingStrategy) error {
    switch strategy.Type {
    case WarmingTypePopular:
        return cw.warmPopularKeys(ctx, strategy)
    case WarmingTypePredictive:
        return cw.warmPredictedKeys(ctx, strategy)
    case WarmingTypeScheduled:
        return cw.warmScheduledKeys(ctx, strategy)
    default:
        return fmt.Errorf("unsupported warming strategy: %s", strategy.Type)
    }
}

func (cw *CacheWarmer) warmPopularKeys(ctx context.Context, strategy WarmingStrategy) error {
    // Get popular keys from analytics
    popularKeys := cw.predictor.GetPopularKeys(strategy.KeyCount)
    
    // Load keys in parallel with rate limiting
    semaphore := make(chan struct{}, strategy.Concurrency)
    var wg sync.WaitGroup
    
    for _, key := range popularKeys {
        semaphore <- struct{}{}
        wg.Add(1)
        
        go func(k string) {
            defer func() {
                <-semaphore
                wg.Done()
            }()
            
            // Load from data source
            value, err := cw.dataSource.Load(ctx, k)
            if err != nil {
                return
            }
            
            // Store in appropriate cache level
            cw.cache.Set(ctx, k, value, strategy.TTL)
        }(key)
    }
    
    wg.Wait()
    return nil
}

// Cache coherence and consistency
type CacheCoherenceManager struct {
    caches        []CacheNode
    invalidator   InvalidationManager
    versionVector VersionVector
    conflictResolver ConflictResolver
}

type InvalidationManager struct {
    invalidationQueue MessageQueue
    subscribers       map[string][]InvalidationSubscriber
    patterns          []InvalidationPattern
}

type InvalidationPattern struct {
    Pattern     string `json:"pattern"`
    Strategy    InvalidationStrategy `json:"strategy"`
    Propagation PropagationStrategy `json:"propagation"`
}

type InvalidationStrategy string

const (
    InvalidateImmediate InvalidationStrategy = "immediate"
    InvalidateLazy      InvalidationStrategy = "lazy"
    InvalidateTTL       InvalidationStrategy = "ttl"
)

func (ccm *CacheCoherenceManager) InvalidateKey(ctx context.Context, key string, version int64) error {
    // Update version vector
    ccm.versionVector.Update(key, version)
    
    // Find matching invalidation patterns
    patterns := ccm.findMatchingPatterns(key)
    
    for _, pattern := range patterns {
        switch pattern.Strategy {
        case InvalidateImmediate:
            return ccm.immediateInvalidation(ctx, key, pattern)
        case InvalidateLazy:
            return ccm.lazyInvalidation(ctx, key, pattern)
        case InvalidateTTL:
            return ccm.ttlInvalidation(ctx, key, pattern)
        }
    }
    
    return nil
}

func (ccm *CacheCoherenceManager) immediateInvalidation(ctx context.Context, key string, pattern InvalidationPattern) error {
    invalidationMsg := InvalidationMessage{
        Key:       key,
        Pattern:   pattern.Pattern,
        Timestamp: time.Now(),
        Version:   ccm.versionVector.Get(key),
    }
    
    // Broadcast to all cache nodes
    for _, cache := range ccm.caches {
        if err := cache.Invalidate(ctx, invalidationMsg); err != nil {
            // Log error but continue with other nodes
            continue
        }
    }
    
    return nil
}
```

This comprehensive Advanced System Design Patterns Guide provides production-ready implementations of sophisticated patterns including Event Sourcing, CQRS, Saga orchestration, and multi-level caching strategies. The code demonstrates enterprise-level system design expertise essential for senior backend engineering roles.

---

## ⚡ **Circuit Breaker & Resilience Patterns**

### **Advanced Circuit Breaker Implementation**

```go
package resilience

import (
    "context"
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

// Circuit Breaker with advanced failure detection
type AdvancedCircuitBreaker struct {
    name            string
    maxRequests     uint64
    interval        time.Duration
    timeout         time.Duration
    readyToTrip     ReadyToTripFunc
    onStateChange   StateChangeFunc
    
    // State management
    state           CircuitState
    generation      uint64
    counts          Counts
    expiry          time.Time
    
    // Advanced features
    failureDetector FailureDetector
    healthChecker   HealthChecker
    metrics         CircuitMetrics
    rateLimiter     RateLimiter
    
    mu              sync.RWMutex
}

type CircuitState int32

const (
    StateClosed CircuitState = iota
    StateHalfOpen
    StateOpen
)

type Counts struct {
    Requests             uint64
    TotalSuccesses       uint64
    TotalFailures        uint64
    ConsecutiveSuccesses uint64
    ConsecutiveFailures  uint64
}

type ReadyToTripFunc func(counts Counts) bool
type StateChangeFunc func(name string, from, to CircuitState)

type FailureDetector interface {
    IsFailure(err error) bool
    GetFailureType(err error) FailureType
    ShouldIgnore(err error) bool
}

type FailureType int

const (
    FailureTypeTimeout FailureType = iota
    FailureTypeRateLimit
    FailureTypeServerError
    FailureTypeClientError
    FailureTypeNetworkError
    FailureTypeCircuitOpen
)

type AdvancedFailureDetector struct {
    timeoutDetector    TimeoutDetector
    errorClassifier    ErrorClassifier
    patternMatcher     PatternMatcher
    thresholds         FailureThresholds
}

type FailureThresholds struct {
    ErrorRateThreshold    float64       `json:"error_rate_threshold"`
    ResponseTimeThreshold time.Duration `json:"response_time_threshold"`
    ConsecutiveFailures   int           `json:"consecutive_failures"`
    WindowSize            time.Duration `json:"window_size"`
}

func NewAdvancedCircuitBreaker(config CircuitBreakerConfig) *AdvancedCircuitBreaker {
    cb := &AdvancedCircuitBreaker{
        name:            config.Name,
        maxRequests:     config.MaxRequests,
        interval:        config.Interval,
        timeout:         config.Timeout,
        readyToTrip:     config.ReadyToTrip,
        onStateChange:   config.OnStateChange,
        failureDetector: NewAdvancedFailureDetector(config.FailureConfig),
        healthChecker:   NewHealthChecker(config.HealthConfig),
        metrics:         NewCircuitMetrics(config.MetricsConfig),
        rateLimiter:     NewAdaptiveRateLimiter(config.RateLimitConfig),
        state:           StateClosed,
        expiry:          time.Now().Add(config.Interval),
    }
    
    // Start background health checking
    go cb.startHealthChecking()
    
    return cb
}

func (acb *AdvancedCircuitBreaker) Execute(ctx context.Context, req func() (interface{}, error)) (interface{}, error) {
    // Check rate limiting first
    if !acb.rateLimiter.Allow() {
        acb.metrics.RecordRateLimited()
        return nil, ErrRateLimited
    }
    
    generation, err := acb.beforeRequest()
    if err != nil {
        acb.metrics.RecordRejected()
        return nil, err
    }
    
    defer func() {
        acb.afterRequest(generation, err)
    }()
    
    // Execute with timeout
    result, err := acb.executeWithTimeout(ctx, req)
    
    // Classify the result
    if err != nil && acb.failureDetector.IsFailure(err) {
        acb.metrics.RecordFailure(acb.failureDetector.GetFailureType(err))
        return result, err
    }
    
    acb.metrics.RecordSuccess()
    return result, err
}

func (acb *AdvancedCircuitBreaker) executeWithTimeout(ctx context.Context, req func() (interface{}, error)) (interface{}, error) {
    if acb.timeout <= 0 {
        return req()
    }
    
    ctx, cancel := context.WithTimeout(ctx, acb.timeout)
    defer cancel()
    
    type result struct {
        data interface{}
        err  error
    }
    
    ch := make(chan result, 1)
    
    go func() {
        data, err := req()
        ch <- result{data: data, err: err}
    }()
    
    select {
    case res := <-ch:
        return res.data, res.err
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

func (acb *AdvancedCircuitBreaker) beforeRequest() (uint64, error) {
    acb.mu.Lock()
    defer acb.mu.Unlock()
    
    now := time.Now()
    state, generation := acb.currentState(now)
    
    if state == StateOpen {
        return generation, ErrOpenState
    }
    
    if state == StateHalfOpen && acb.counts.Requests >= acb.maxRequests {
        return generation, ErrTooManyRequests
    }
    
    acb.counts.Requests++
    return generation, nil
}

func (acb *AdvancedCircuitBreaker) afterRequest(before uint64, err error) {
    acb.mu.Lock()
    defer acb.mu.Unlock()
    
    now := time.Now()
    state, generation := acb.currentState(now)
    
    if generation != before {
        return
    }
    
    if err != nil && acb.failureDetector.IsFailure(err) {
        acb.onFailure(state, now)
    } else {
        acb.onSuccess(state, now)
    }
}

func (acb *AdvancedCircuitBreaker) onFailure(state CircuitState, now time.Time) {
    acb.counts.TotalFailures++
    acb.counts.ConsecutiveFailures++
    acb.counts.ConsecutiveSuccesses = 0
    
    if state == StateClosed {
        if acb.readyToTrip(acb.counts) {
            acb.setState(StateOpen, now)
        }
    } else if state == StateHalfOpen {
        acb.setState(StateOpen, now)
    }
}

func (acb *AdvancedCircuitBreaker) onSuccess(state CircuitState, now time.Time) {
    acb.counts.TotalSuccesses++
    acb.counts.ConsecutiveSuccesses++
    acb.counts.ConsecutiveFailures = 0
    
    if state == StateHalfOpen {
        if acb.counts.ConsecutiveSuccesses >= acb.maxRequests {
            acb.setState(StateClosed, now)
        }
    }
}

func (acb *AdvancedCircuitBreaker) setState(state CircuitState, now time.Time) {
    if acb.state == state {
        return
    }
    
    prev := acb.state
    acb.state = state
    acb.generation++
    acb.counts = Counts{}
    
    var expiry time.Time
    switch state {
    case StateClosed:
        expiry = now.Add(acb.interval)
    case StateOpen:
        expiry = now.Add(acb.timeout)
    default:
        expiry = now.Add(acb.interval)
    }
    acb.expiry = expiry
    
    if acb.onStateChange != nil {
        acb.onStateChange(acb.name, prev, state)
    }
    
    acb.metrics.RecordStateChange(prev, state)
}

func (acb *AdvancedCircuitBreaker) currentState(now time.Time) (CircuitState, uint64) {
    switch acb.state {
    case StateClosed:
        if !acb.expiry.IsZero() && acb.expiry.Before(now) {
            acb.toNewGeneration(now)
        }
    case StateOpen:
        if acb.expiry.Before(now) {
            acb.setState(StateHalfOpen, now)
        }
    }
    
    return acb.state, acb.generation
}

func (acb *AdvancedCircuitBreaker) toNewGeneration(now time.Time) {
    acb.generation++
    acb.counts = Counts{}
    
    var expiry time.Time
    switch acb.state {
    case StateClosed:
        expiry = now.Add(acb.interval)
    case StateOpen:
        expiry = now.Add(acb.timeout)
    default:
        expiry = now.Add(acb.interval)
    }
    acb.expiry = expiry
}

// Health checker for proactive circuit management
type HealthChecker struct {
    endpoint        string
    interval        time.Duration
    timeout         time.Duration
    healthyThreshold int
    unhealthyThreshold int
    
    client          HTTPClient
    currentHealth   int32
    metrics         HealthMetrics
}

func (hc *HealthChecker) StartHealthChecking(ctx context.Context, callback HealthCallback) {
    ticker := time.NewTicker(hc.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            healthy := hc.checkHealth(ctx)
            hc.updateHealthStatus(healthy, callback)
            
        case <-ctx.Done():
            return
        }
    }
}

func (hc *HealthChecker) checkHealth(ctx context.Context) bool {
    ctx, cancel := context.WithTimeout(ctx, hc.timeout)
    defer cancel()
    
    resp, err := hc.client.Get(ctx, hc.endpoint)
    if err != nil {
        hc.metrics.RecordHealthCheckFailure(err)
        return false
    }
    defer resp.Body.Close()
    
    healthy := resp.StatusCode >= 200 && resp.StatusCode < 300
    if healthy {
        hc.metrics.RecordHealthCheckSuccess()
    } else {
        hc.metrics.RecordHealthCheckFailure(fmt.Errorf("unhealthy status: %d", resp.StatusCode))
    }
    
    return healthy
}

func (hc *HealthChecker) updateHealthStatus(healthy bool, callback HealthCallback) {
    currentHealth := atomic.LoadInt32(&hc.currentHealth)
    
    if healthy {
        if currentHealth < int32(hc.healthyThreshold) {
            newHealth := atomic.AddInt32(&hc.currentHealth, 1)
            if newHealth == int32(hc.healthyThreshold) && callback != nil {
                callback(true)
            }
        }
    } else {
        if currentHealth > -int32(hc.unhealthyThreshold) {
            newHealth := atomic.AddInt32(&hc.currentHealth, -1)
            if newHealth == -int32(hc.unhealthyThreshold) && callback != nil {
                callback(false)
            }
        }
    }
}

// Bulkhead isolation pattern
type BulkheadIsolator struct {
    pools       map[string]*ResourcePool
    router      RequestRouter
    metrics     BulkheadMetrics
    fallback    FallbackHandler
    mu          sync.RWMutex
}

type ResourcePool struct {
    name         string
    maxSize      int
    currentSize  int
    queue        chan Request
    workers      []*Worker
    circuit      *AdvancedCircuitBreaker
    metrics      PoolMetrics
    rateLimiter  RateLimiter
    mu           sync.RWMutex
}

type Worker struct {
    id       int
    pool     *ResourcePool
    requests chan Request
    quit     chan bool
    busy     int32
}

type Request struct {
    ID          string                         `json:"id"`
    Type        string                         `json:"type"`
    Priority    Priority                       `json:"priority"`
    Payload     interface{}                    `json:"payload"`
    Context     context.Context                `json:"-"`
    Response    chan Response                  `json:"-"`
    Timeout     time.Duration                  `json:"timeout"`
    Retries     int                           `json:"retries"`
    Metadata    map[string]interface{}         `json:"metadata"`
}

type Response struct {
    Data  interface{} `json:"data"`
    Error error      `json:"error"`
    Stats RequestStats `json:"stats"`
}

type RequestStats struct {
    QueueTime     time.Duration `json:"queue_time"`
    ProcessTime   time.Duration `json:"process_time"`
    TotalTime     time.Duration `json:"total_time"`
    WorkerID      int          `json:"worker_id"`
    RetryCount    int          `json:"retry_count"`
}

type Priority int

const (
    PriorityLow Priority = iota
    PriorityNormal
    PriorityHigh
    PriorityCritical
)

func NewBulkheadIsolator(config BulkheadConfig) *BulkheadIsolator {
    bi := &BulkheadIsolator{
        pools:    make(map[string]*ResourcePool),
        router:   NewRequestRouter(config.RoutingRules),
        metrics:  NewBulkheadMetrics(),
        fallback: NewFallbackHandler(config.FallbackConfig),
    }
    
    // Initialize resource pools
    for name, poolConfig := range config.Pools {
        bi.pools[name] = bi.createResourcePool(name, poolConfig)
    }
    
    return bi
}

func (bi *BulkheadIsolator) createResourcePool(name string, config PoolConfig) *ResourcePool {
    pool := &ResourcePool{
        name:        name,
        maxSize:     config.MaxSize,
        queue:       make(chan Request, config.QueueSize),
        workers:     make([]*Worker, config.MaxSize),
        circuit:     NewAdvancedCircuitBreaker(config.CircuitConfig),
        metrics:     NewPoolMetrics(name),
        rateLimiter: NewTokenBucketRateLimiter(config.RateLimitConfig),
    }
    
    // Create workers
    for i := 0; i < config.MaxSize; i++ {
        worker := &Worker{
            id:       i,
            pool:     pool,
            requests: make(chan Request, 1),
            quit:     make(chan bool),
        }
        pool.workers[i] = worker
        go worker.start()
    }
    
    // Start pool manager
    go pool.manage()
    
    return pool
}

func (bi *BulkheadIsolator) Execute(ctx context.Context, req Request) (*Response, error) {
    startTime := time.Now()
    
    // Route request to appropriate pool
    poolName := bi.router.Route(req)
    
    bi.mu.RLock()
    pool, exists := bi.pools[poolName]
    bi.mu.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("pool not found: %s", poolName)
    }
    
    // Check rate limiting
    if !pool.rateLimiter.Allow() {
        bi.metrics.RecordRateLimited(poolName)
        return bi.fallback.Handle(req, ErrRateLimited)
    }
    
    // Execute through circuit breaker
    result, err := pool.circuit.Execute(ctx, func() (interface{}, error) {
        return bi.executeInPool(ctx, pool, req, startTime)
    })
    
    if err != nil {
        // Try fallback if available
        if fallbackResp, fallbackErr := bi.fallback.Handle(req, err); fallbackErr == nil {
            return fallbackResp, nil
        }
        return nil, err
    }
    
    return result.(*Response), nil
}

func (bi *BulkheadIsolator) executeInPool(ctx context.Context, pool *ResourcePool, req Request, startTime time.Time) (*Response, error) {
    req.Context = ctx
    req.Response = make(chan Response, 1)
    
    // Add to queue
    select {
    case pool.queue <- req:
        pool.metrics.RecordEnqueued()
    default:
        pool.metrics.RecordQueueFull()
        return nil, ErrQueueFull
    }
    
    // Wait for response
    select {
    case resp := <-req.Response:
        resp.Stats.QueueTime = time.Since(startTime)
        resp.Stats.TotalTime = time.Since(startTime)
        
        if resp.Error != nil {
            pool.metrics.RecordFailure()
            return nil, resp.Error
        }
        
        pool.metrics.RecordSuccess()
        return &resp, nil
        
    case <-ctx.Done():
        pool.metrics.RecordTimeout()
        return nil, ctx.Err()
    }
}

func (w *Worker) start() {
    for {
        select {
        case req := <-w.pool.queue:
            w.processRequest(req)
            
        case <-w.quit:
            return
        }
    }
}

func (w *Worker) processRequest(req Request) {
    atomic.StoreInt32(&w.busy, 1)
    defer atomic.StoreInt32(&w.busy, 0)
    
    startTime := time.Now()
    
    // Process with timeout
    ctx := req.Context
    if req.Timeout > 0 {
        var cancel context.CancelFunc
        ctx, cancel = context.WithTimeout(ctx, req.Timeout)
        defer cancel()
    }
    
    // Execute the actual work
    data, err := w.executeWork(ctx, req)
    
    response := Response{
        Data:  data,
        Error: err,
        Stats: RequestStats{
            ProcessTime: time.Since(startTime),
            WorkerID:    w.id,
        },
    }
    
    // Send response
    select {
    case req.Response <- response:
    default:
        // Response channel was closed or request was cancelled
    }
}

func (w *Worker) executeWork(ctx context.Context, req Request) (interface{}, error) {
    // This would contain the actual business logic
    // For demonstration, we'll simulate some work
    
    processor, exists := GetProcessor(req.Type)
    if !exists {
        return nil, fmt.Errorf("no processor for request type: %s", req.Type)
    }
    
    return processor.Process(ctx, req.Payload)
}

// Retry patterns with exponential backoff and jitter
type RetryPolicy struct {
    MaxRetries      int           `json:"max_retries"`
    InitialDelay    time.Duration `json:"initial_delay"`
    MaxDelay        time.Duration `json:"max_delay"`
    Multiplier      float64       `json:"multiplier"`
    Jitter          JitterType    `json:"jitter"`
    RetryableErrors []string      `json:"retryable_errors"`
}

type JitterType string

const (
    NoJitter     JitterType = "none"
    FullJitter   JitterType = "full"
    EqualJitter  JitterType = "equal"
    DecorrJitter JitterType = "decorr"
)

type RetryExecutor struct {
    policy     RetryPolicy
    classifier ErrorClassifier
    metrics    RetryMetrics
}

func (re *RetryExecutor) Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error) {
    var lastErr error
    
    for attempt := 0; attempt <= re.policy.MaxRetries; attempt++ {
        result, err := operation()
        if err == nil {
            re.metrics.RecordSuccess(attempt)
            return result, nil
        }
        
        lastErr = err
        
        // Check if error is retryable
        if !re.classifier.IsRetryable(err) {
            re.metrics.RecordNonRetryableFailure(attempt)
            return nil, err
        }
        
        // Don't delay after the last attempt
        if attempt == re.policy.MaxRetries {
            break
        }
        
        // Calculate delay with jitter
        delay := re.calculateDelay(attempt)
        re.metrics.RecordRetry(attempt, delay)
        
        // Wait for delay or context cancellation
        select {
        case <-time.After(delay):
            continue
        case <-ctx.Done():
            return nil, ctx.Err()
        }
    }
    
    re.metrics.RecordFinalFailure(re.policy.MaxRetries)
    return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

func (re *RetryExecutor) calculateDelay(attempt int) time.Duration {
    delay := time.Duration(float64(re.policy.InitialDelay) * 
                           math.Pow(re.policy.Multiplier, float64(attempt)))
    
    if delay > re.policy.MaxDelay {
        delay = re.policy.MaxDelay
    }
    
    switch re.policy.Jitter {
    case FullJitter:
        delay = time.Duration(rand.Float64() * float64(delay))
    case EqualJitter:
        delay = delay/2 + time.Duration(rand.Float64()*float64(delay/2))
    case DecorrJitter:
        delay = time.Duration(rand.Float64() * float64(delay) * 3)
        if delay > re.policy.MaxDelay {
            delay = re.policy.MaxDelay
        }
    }
    
    return delay
}

// Timeout and deadline management
type TimeoutManager struct {
    defaultTimeout  time.Duration
    serviceTimeouts map[string]time.Duration
    adaptiveTimeout *AdaptiveTimeout
    metrics         TimeoutMetrics
}

type AdaptiveTimeout struct {
    percentile      float64
    windowSize      time.Duration
    minTimeout      time.Duration
    maxTimeout      time.Duration
    responseTimeHistory *RingBuffer
    mu              sync.RWMutex
}

func (tm *TimeoutManager) GetTimeout(ctx context.Context, service string) time.Duration {
    // Check if context already has a deadline
    if deadline, ok := ctx.Deadline(); ok {
        remaining := time.Until(deadline)
        if remaining > 0 {
            return remaining
        }
    }
    
    // Check service-specific timeout
    if timeout, exists := tm.serviceTimeouts[service]; exists {
        // Apply adaptive adjustment if enabled
        if tm.adaptiveTimeout != nil {
            return tm.adaptiveTimeout.AdjustTimeout(service, timeout)
        }
        return timeout
    }
    
    return tm.defaultTimeout
}

func (at *AdaptiveTimeout) AdjustTimeout(service string, baseTimeout time.Duration) time.Duration {
    at.mu.RLock()
    defer at.mu.RUnlock()
    
    // Calculate percentile from recent response times
    responseTimes := at.responseTimeHistory.GetValues()
    if len(responseTimes) < 10 {
        return baseTimeout
    }
    
    percentileTime := calculatePercentile(responseTimes, at.percentile)
    adaptiveTimeout := time.Duration(float64(percentileTime) * 1.5) // 50% buffer
    
    // Clamp to min/max bounds
    if adaptiveTimeout < at.minTimeout {
        adaptiveTimeout = at.minTimeout
    }
    if adaptiveTimeout > at.maxTimeout {
        adaptiveTimeout = at.maxTimeout
    }
    
    return adaptiveTimeout
}

func calculatePercentile(values []time.Duration, percentile float64) time.Duration {
    if len(values) == 0 {
        return 0
    }
    
    sorted := make([]time.Duration, len(values))
    copy(sorted, values)
    sort.Slice(sorted, func(i, j int) bool {
        return sorted[i] < sorted[j]
    })
    
    index := int(float64(len(sorted)) * percentile / 100.0)
    if index >= len(sorted) {
        index = len(sorted) - 1
    }
    
    return sorted[index]
}
```

---

## 🏗️ **Bulkhead Isolation Patterns**

### **Resource Isolation & Thread Pool Management**

```go
package bulkhead

import (
    "context"
    "fmt"
    "runtime"
    "sync"
    "sync/atomic"
    "time"
)

// Thread pool bulkhead for CPU-intensive operations
type ThreadPoolBulkhead struct {
    pools      map[string]*ThreadPool
    router     ResourceRouter
    monitor    ResourceMonitor
    isolator   ResourceIsolator
    metrics    BulkheadMetrics
    mu         sync.RWMutex
}

type ThreadPool struct {
    name           string
    coreSize       int
    maxSize        int
    queueCapacity  int
    keepAliveTime  time.Duration
    
    workers        []*PoolWorker
    taskQueue      chan Task
    idleWorkers    chan *PoolWorker
    
    activeWorkers  int32
    totalWorkers   int32
    
    rejectionPolicy RejectionPolicy
    factory         WorkerFactory
    metrics         ThreadPoolMetrics
    
    shutdown       int32
    shutdownCh     chan struct{}
    wg             sync.WaitGroup
    mu             sync.RWMutex
}

type Task struct {
    ID          string
    Type        string
    Priority    int
    Payload     interface{}
    Context     context.Context
    Handler     TaskHandler
    ResultChan  chan TaskResult
    SubmittedAt time.Time
    Deadline    time.Time
}

type TaskResult struct {
    Data      interface{}   `json:"data"`
    Error     error        `json:"error"`
    Duration  time.Duration `json:"duration"`
    WorkerID  string       `json:"worker_id"`
    Stats     TaskStats    `json:"stats"`
}

type TaskStats struct {
    QueueTime    time.Duration `json:"queue_time"`
    ExecuteTime  time.Duration `json:"execute_time"`
    TotalTime    time.Duration `json:"total_time"`
    MemoryUsed   int64        `json:"memory_used"`
    CPUTime      time.Duration `json:"cpu_time"`
}

type PoolWorker struct {
    id         string
    pool       *ThreadPool
    taskChan   chan Task
    lastUsed   time.Time
    stats      WorkerStats
    goroutineID int
    mu         sync.RWMutex
}

type WorkerStats struct {
    TasksProcessed int64         `json:"tasks_processed"`
    TotalTime      time.Duration `json:"total_time"`
    IdleTime       time.Duration `json:"idle_time"`
    LastActive     time.Time     `json:"last_active"`
    Errors         int64         `json:"errors"`
}

func NewThreadPoolBulkhead(config BulkheadConfig) *ThreadPoolBulkhead {
    tpb := &ThreadPoolBulkhead{
        pools:   make(map[string]*ThreadPool),
        router:  NewResourceRouter(config.RoutingConfig),
        monitor: NewResourceMonitor(config.MonitorConfig),
        isolator: NewResourceIsolator(config.IsolationConfig),
        metrics: NewBulkheadMetrics(),
    }
    
    // Initialize thread pools
    for name, poolConfig := range config.ThreadPools {
        tpb.pools[name] = NewThreadPool(name, poolConfig)
    }
    
    // Start monitoring
    go tpb.monitor.Start(context.Background())
    
    return tpb
}

func NewThreadPool(name string, config ThreadPoolConfig) *ThreadPool {
    tp := &ThreadPool{
        name:           name,
        coreSize:       config.CoreSize,
        maxSize:        config.MaxSize,
        queueCapacity:  config.QueueCapacity,
        keepAliveTime:  config.KeepAliveTime,
        workers:        make([]*PoolWorker, 0, config.MaxSize),
        taskQueue:      make(chan Task, config.QueueCapacity),
        idleWorkers:    make(chan *PoolWorker, config.MaxSize),
        rejectionPolicy: config.RejectionPolicy,
        factory:        config.WorkerFactory,
        metrics:        NewThreadPoolMetrics(name),
        shutdownCh:     make(chan struct{}),
    }
    
    // Pre-create core workers
    tp.ensureCoreWorkers()
    
    // Start pool manager
    go tp.manage()
    
    return tp
}

func (tp *ThreadPool) Submit(ctx context.Context, task Task) (*TaskResult, error) {
    if atomic.LoadInt32(&tp.shutdown) == 1 {
        return nil, ErrPoolShutdown
    }
    
    task.Context = ctx
    task.SubmittedAt = time.Now()
    task.ResultChan = make(chan TaskResult, 1)
    
    // Set deadline from context
    if deadline, ok := ctx.Deadline(); ok {
        task.Deadline = deadline
    }
    
    // Try to submit task
    select {
    case tp.taskQueue <- task:
        tp.metrics.RecordTaskSubmitted()
        
        // Ensure adequate workers
        tp.ensureWorkers()
        
        // Wait for result
        select {
        case result := <-task.ResultChan:
            if result.Error != nil {
                tp.metrics.RecordTaskFailed()
            } else {
                tp.metrics.RecordTaskCompleted()
            }
            return &result, result.Error
            
        case <-ctx.Done():
            tp.metrics.RecordTaskCancelled()
            return nil, ctx.Err()
        }
        
    default:
        // Queue is full, apply rejection policy
        return tp.handleRejection(task)
    }
}

func (tp *ThreadPool) handleRejection(task Task) (*TaskResult, error) {
    tp.metrics.RecordTaskRejected()
    
    switch tp.rejectionPolicy {
    case RejectPolicyAbort:
        return nil, ErrTaskRejected
        
    case RejectPolicyCallerRuns:
        // Execute in caller's goroutine
        start := time.Now()
        result, err := task.Handler.Handle(task.Context, task.Payload)
        duration := time.Since(start)
        
        return &TaskResult{
            Data:     result,
            Error:    err,
            Duration: duration,
            WorkerID: "caller",
            Stats: TaskStats{
                QueueTime:   0,
                ExecuteTime: duration,
                TotalTime:   duration,
            },
        }, err
        
    case RejectPolicyDiscardOldest:
        // Try to remove oldest task and submit new one
        select {
        case oldTask := <-tp.taskQueue:
            tp.metrics.RecordTaskDiscarded()
            // Send error to discarded task
            oldTask.ResultChan <- TaskResult{
                Error: ErrTaskDiscarded,
            }
            
            // Submit new task
            select {
            case tp.taskQueue <- task:
                tp.metrics.RecordTaskSubmitted()
                return nil, nil
            default:
                return nil, ErrTaskRejected
            }
        default:
            return nil, ErrTaskRejected
        }
        
    default:
        return nil, ErrTaskRejected
    }
}

func (tp *ThreadPool) ensureWorkers() {
    tp.mu.Lock()
    defer tp.mu.Unlock()
    
    activeWorkers := atomic.LoadInt32(&tp.activeWorkers)
    totalWorkers := atomic.LoadInt32(&tp.totalWorkers)
    queueSize := len(tp.taskQueue)
    
    // Create more workers if needed
    if queueSize > 0 && int(activeWorkers) < tp.maxSize && int(totalWorkers) < tp.maxSize {
        worker := tp.createWorker()
        tp.workers = append(tp.workers, worker)
        atomic.AddInt32(&tp.totalWorkers, 1)
        go worker.run()
    }
}

func (tp *ThreadPool) ensureCoreWorkers() {
    tp.mu.Lock()
    defer tp.mu.Unlock()
    
    for i := 0; i < tp.coreSize; i++ {
        worker := tp.createWorker()
        tp.workers = append(tp.workers, worker)
        atomic.AddInt32(&tp.totalWorkers, 1)
        go worker.run()
    }
}

func (tp *ThreadPool) createWorker() *PoolWorker {
    worker := &PoolWorker{
        id:         fmt.Sprintf("%s-worker-%d", tp.name, len(tp.workers)),
        pool:       tp,
        taskChan:   make(chan Task, 1),
        lastUsed:   time.Now(),
        stats:      WorkerStats{LastActive: time.Now()},
        goroutineID: runtime.NumGoroutine(),
    }
    
    if tp.factory != nil {
        return tp.factory.CreateWorker(worker)
    }
    
    return worker
}

func (pw *PoolWorker) run() {
    defer func() {
        atomic.AddInt32(&pw.pool.totalWorkers, -1)
        pw.pool.wg.Done()
    }()
    
    pw.pool.wg.Add(1)
    
    for {
        select {
        case task := <-pw.pool.taskQueue:
            pw.processTask(task)
            
        case <-time.After(pw.pool.keepAliveTime):
            // Check if worker should be terminated
            if pw.shouldTerminate() {
                return
            }
            
        case <-pw.pool.shutdownCh:
            return
        }
    }
}

func (pw *PoolWorker) processTask(task Task) {
    atomic.AddInt32(&pw.pool.activeWorkers, 1)
    defer atomic.AddInt32(&pw.pool.activeWorkers, -1)
    
    startTime := time.Now()
    queueTime := startTime.Sub(task.SubmittedAt)
    
    pw.mu.Lock()
    pw.lastUsed = startTime
    pw.stats.LastActive = startTime
    pw.mu.Unlock()
    
    // Check if task has already timed out
    if !task.Deadline.IsZero() && time.Now().After(task.Deadline) {
        task.ResultChan <- TaskResult{
            Error:    ErrTaskTimeout,
            Duration: time.Since(startTime),
            WorkerID: pw.id,
            Stats: TaskStats{
                QueueTime: queueTime,
                TotalTime: time.Since(task.SubmittedAt),
            },
        }
        return
    }
    
    // Execute task with resource monitoring
    var memBefore runtime.MemStats
    runtime.ReadMemStats(&memBefore)
    
    result, err := pw.executeTask(task)
    
    var memAfter runtime.MemStats
    runtime.ReadMemStats(&memAfter)
    
    executeTime := time.Since(startTime)
    totalTime := time.Since(task.SubmittedAt)
    
    // Update worker stats
    pw.mu.Lock()
    pw.stats.TasksProcessed++
    pw.stats.TotalTime += executeTime
    if err != nil {
        pw.stats.Errors++
    }
    pw.mu.Unlock()
    
    // Send result
    taskResult := TaskResult{
        Data:     result,
        Error:    err,
        Duration: executeTime,
        WorkerID: pw.id,
        Stats: TaskStats{
            QueueTime:   queueTime,
            ExecuteTime: executeTime,
            TotalTime:   totalTime,
            MemoryUsed:  int64(memAfter.Alloc - memBefore.Alloc),
        },
    }
    
    select {
    case task.ResultChan <- taskResult:
    default:
        // Result channel was closed or task was cancelled
    }
    
    // Update pool metrics
    pw.pool.metrics.RecordTaskExecution(executeTime, err == nil)
}

func (pw *PoolWorker) executeTask(task Task) (interface{}, error) {
    // Create task-specific context with timeout if needed
    ctx := task.Context
    if !task.Deadline.IsZero() {
        var cancel context.CancelFunc
        ctx, cancel = context.WithDeadline(ctx, task.Deadline)
        defer cancel()
    }
    
    // Execute the task
    return task.Handler.Handle(ctx, task.Payload)
}

func (pw *PoolWorker) shouldTerminate() bool {
    pw.mu.RLock()
    defer pw.mu.RUnlock()
    
    // Don't terminate core workers
    totalWorkers := atomic.LoadInt32(&pw.pool.totalWorkers)
    if int(totalWorkers) <= pw.pool.coreSize {
        return false
    }
    
    // Terminate if idle too long
    return time.Since(pw.lastUsed) > pw.pool.keepAliveTime
}

// Memory bulkhead for preventing OOM
type MemoryBulkhead struct {
    pools       map[string]*MemoryPool
    monitor     MemoryMonitor
    gc          GCManager
    limits      MemoryLimits
    metrics     MemoryMetrics
    alerts      AlertManager
    mu          sync.RWMutex
}

type MemoryPool struct {
    name        string
    maxSize     int64
    currentSize int64
    allocator   MemoryAllocator
    tracker     AllocationTracker
    policy      EvictionPolicy
    mu          sync.RWMutex
}

type MemoryAllocator interface {
    Allocate(size int64) ([]byte, error)
    Deallocate(ptr []byte) error
    GetStats() AllocationStats
}

type AllocationStats struct {
    TotalAllocated   int64 `json:"total_allocated"`
    TotalDeallocated int64 `json:"total_deallocated"`
    CurrentUsage     int64 `json:"current_usage"`
    AllocationCount  int64 `json:"allocation_count"`
    DeallocationCount int64 `json:"deallocation_count"`
}

func (mb *MemoryBulkhead) Allocate(poolName string, size int64) ([]byte, error) {
    mb.mu.RLock()
    pool, exists := mb.pools[poolName]
    mb.mu.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("memory pool not found: %s", poolName)
    }
    
    pool.mu.Lock()
    defer pool.mu.Unlock()
    
    // Check if allocation would exceed limit
    if pool.currentSize+size > pool.maxSize {
        // Try to evict some memory
        if evicted := pool.policy.Evict(size); evicted < size {
            mb.metrics.RecordAllocationFailure(poolName, size)
            return nil, ErrMemoryLimitExceeded
        }
    }
    
    // Allocate memory
    memory, err := pool.allocator.Allocate(size)
    if err != nil {
        mb.metrics.RecordAllocationFailure(poolName, size)
        return nil, err
    }
    
    pool.currentSize += size
    pool.tracker.Track(memory, size)
    mb.metrics.RecordAllocation(poolName, size)
    
    return memory, nil
}

// Connection pool bulkhead
type ConnectionBulkhead struct {
    pools      map[string]*ConnectionPool
    router     ConnectionRouter
    monitor    ConnectionMonitor
    balancer   LoadBalancer
    circuitBreaker *AdvancedCircuitBreaker
    metrics    ConnectionMetrics
    mu         sync.RWMutex
}

type ConnectionPool struct {
    name           string
    factory        ConnectionFactory
    validator      ConnectionValidator
    maxActive      int
    maxIdle        int
    minIdle        int
    maxLifetime    time.Duration
    idleTimeout    time.Duration
    
    active         map[string]*PooledConnection
    idle           []*PooledConnection
    waiting        []chan *PooledConnection
    
    activeCount    int32
    metrics        PoolMetrics
    mu             sync.RWMutex
}

type PooledConnection struct {
    conn        Connection
    createdAt   time.Time
    lastUsed    time.Time
    usageCount  int64
    pool        *ConnectionPool
    id          string
    healthy     bool
}

func (cp *ConnectionPool) Get(ctx context.Context) (*PooledConnection, error) {
    // Try to get from idle connections first
    if conn := cp.getIdleConnection(); conn != nil {
        return conn, nil
    }
    
    // Create new connection if under limit
    if atomic.LoadInt32(&cp.activeCount) < int32(cp.maxActive) {
        return cp.createConnection(ctx)
    }
    
    // Wait for available connection
    return cp.waitForConnection(ctx)
}

func (cp *ConnectionPool) getIdleConnection() *PooledConnection {
    cp.mu.Lock()
    defer cp.mu.Unlock()
    
    for len(cp.idle) > 0 {
        conn := cp.idle[len(cp.idle)-1]
        cp.idle = cp.idle[:len(cp.idle)-1]
        
        // Validate connection
        if cp.isConnectionValid(conn) {
            cp.active[conn.id] = conn
            conn.lastUsed = time.Now()
            conn.usageCount++
            return conn
        } else {
            // Connection is invalid, close it
            cp.closeConnection(conn)
        }
    }
    
    return nil
}

func (cp *ConnectionPool) createConnection(ctx context.Context) (*PooledConnection, error) {
    conn, err := cp.factory.Create(ctx)
    if err != nil {
        return nil, err
    }
    
    pooledConn := &PooledConnection{
        conn:       conn,
        createdAt:  time.Now(),
        lastUsed:   time.Now(),
        usageCount: 1,
        pool:       cp,
        id:         generateConnectionID(),
        healthy:    true,
    }
    
    cp.mu.Lock()
    cp.active[pooledConn.id] = pooledConn
    atomic.AddInt32(&cp.activeCount, 1)
    cp.mu.Unlock()
    
    cp.metrics.RecordConnectionCreated()
    return pooledConn, nil
}

func (cp *ConnectionPool) Return(conn *PooledConnection) {
    cp.mu.Lock()
    defer cp.mu.Unlock()
    
    // Remove from active connections
    delete(cp.active, conn.id)
    
    // Check if connection is still healthy
    if !cp.isConnectionValid(conn) {
        cp.closeConnection(conn)
        return
    }
    
    // Check if we have waiting requests
    if len(cp.waiting) > 0 {
        waiter := cp.waiting[0]
        cp.waiting = cp.waiting[1:]
        
        conn.lastUsed = time.Now()
        conn.usageCount++
        cp.active[conn.id] = conn
        
        select {
        case waiter <- conn:
        default:
            // Waiter gave up, return to idle
            cp.returnToIdle(conn)
        }
        return
    }
    
    // Return to idle pool
    cp.returnToIdle(conn)
}

func (cp *ConnectionPool) returnToIdle(conn *PooledConnection) {
    if len(cp.idle) < cp.maxIdle {
        cp.idle = append(cp.idle, conn)
    } else {
        cp.closeConnection(conn)
    }
}

func (cp *ConnectionPool) waitForConnection(ctx context.Context) (*PooledConnection, error) {
    waiter := make(chan *PooledConnection, 1)
    
    cp.mu.Lock()
    cp.waiting = append(cp.waiting, waiter)
    cp.mu.Unlock()
    
    select {
    case conn := <-waiter:
        return conn, nil
    case <-ctx.Done():
        // Remove from waiting list
        cp.mu.Lock()
        for i, w := range cp.waiting {
            if w == waiter {
                cp.waiting = append(cp.waiting[:i], cp.waiting[i+1:]...)
                break
            }
        }
        cp.mu.Unlock()
        
        return nil, ctx.Err()
    }
}

func (cp *ConnectionPool) isConnectionValid(conn *PooledConnection) bool {
    now := time.Now()
    
    // Check maximum lifetime
    if cp.maxLifetime > 0 && now.Sub(conn.createdAt) > cp.maxLifetime {
        return false
    }
    
    // Check idle timeout
    if cp.idleTimeout > 0 && now.Sub(conn.lastUsed) > cp.idleTimeout {
        return false
    }
    
    // Validate using custom validator
    if cp.validator != nil && !cp.validator.IsValid(conn.conn) {
        return false
    }
    
    return conn.healthy
}
```

---

## 🌐 **API Gateway Patterns**

### **Advanced API Gateway Implementation**

```go
package gateway

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
    "sync"
    "time"
)

// Advanced API Gateway with comprehensive features
type APIGateway struct {
    router          *Router
    middleware      MiddlewareChain
    rateLimiter     RateLimiter
    auth            AuthenticationManager
    circuitBreaker  *AdvancedCircuitBreaker
    loadBalancer    LoadBalancer
    cache           *APICache
    transformer     RequestTransformer
    validator       RequestValidator
    monitor         GatewayMonitor
    config          GatewayConfig
    mu              sync.RWMutex
}

type Router struct {
    routes      map[string]*Route
    middleware  []Middleware
    fallback    http.Handler
    mu          sync.RWMutex
}

type Route struct {
    Path            string            `json:"path"`
    Method          string            `json:"method"`
    Service         string            `json:"service"`
    Backends        []Backend         `json:"backends"`
    Timeout         time.Duration     `json:"timeout"`
    Retries         int              `json:"retries"`
    RateLimit       RateLimitConfig   `json:"rate_limit"`
    Auth            AuthConfig        `json:"auth"`
    Cache           CacheConfig       `json:"cache"`
    Transform       TransformConfig   `json:"transform"`
    Validation      ValidationConfig  `json:"validation"`
    CircuitBreaker  CircuitConfig     `json:"circuit_breaker"`
    Metadata        map[string]interface{} `json:"metadata"`
}

type Backend struct {
    URL         string            `json:"url"`
    Weight      int              `json:"weight"`
    Health      HealthCheck       `json:"health"`
    Timeout     time.Duration     `json:"timeout"`
    Headers     map[string]string `json:"headers"`
    Auth        BackendAuth       `json:"auth"`
    TLS         TLSConfig         `json:"tls"`
}

func NewAPIGateway(config GatewayConfig) *APIGateway {
    gw := &APIGateway{
        router:          NewRouter(),
        middleware:      NewMiddlewareChain(),
        rateLimiter:     NewDistributedRateLimiter(config.RateLimitConfig),
        auth:           NewAuthenticationManager(config.AuthConfig),
        circuitBreaker:  NewAdvancedCircuitBreaker(config.CircuitConfig),
        loadBalancer:    NewAdvancedLoadBalancer(config.LoadBalancerConfig),
        cache:          NewAPICache(config.CacheConfig),
        transformer:    NewRequestTransformer(config.TransformConfig),
        validator:      NewRequestValidator(config.ValidationConfig),
        monitor:        NewGatewayMonitor(config.MonitorConfig),
        config:         config,
    }
    
    // Setup default middleware
    gw.setupDefaultMiddleware()
    
    // Start background processes
    go gw.startHealthChecking()
    go gw.startMetricsCollection()
    
    return gw
}

func (gw *APIGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    startTime := time.Now()
    requestID := generateRequestID()
    
    // Create request context
    ctx := context.WithValue(r.Context(), "request_id", requestID)
    ctx = context.WithValue(ctx, "start_time", startTime)
    r = r.WithContext(ctx)
    
    // Execute middleware chain
    gw.middleware.Execute(w, r, func(w http.ResponseWriter, r *http.Request) {
        gw.handleRequest(w, r)
    })
}

func (gw *APIGateway) handleRequest(w http.ResponseWriter, r *http.Request) {
    requestID := r.Context().Value("request_id").(string)
    startTime := r.Context().Value("start_time").(time.Time)
    
    // Find matching route
    route := gw.router.Match(r)
    if route == nil {
        gw.monitor.RecordError("route_not_found", r.URL.Path)
        http.NotFound(w, r)
        return
    }
    
    // Apply rate limiting
    if !gw.rateLimiter.Allow(r.Context(), route.RateLimit, r) {
        gw.monitor.RecordRateLimit(route.Path)
        http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
        return
    }
    
    // Authentication
    if route.Auth.Required {
        user, err := gw.auth.Authenticate(r, route.Auth)
        if err != nil {
            gw.monitor.RecordAuthFailure(route.Path, err)
            http.Error(w, "Authentication failed", http.StatusUnauthorized)
            return
        }
        r = r.WithContext(context.WithValue(r.Context(), "user", user))
    }
    
    // Check cache
    if route.Cache.Enabled {
        if response := gw.cache.Get(r.Context(), route, r); response != nil {
            gw.monitor.RecordCacheHit(route.Path)
            gw.writeResponse(w, response)
            return
        }
    }
    
    // Request validation
    if route.Validation.Enabled {
        if err := gw.validator.Validate(r, route.Validation); err != nil {
            gw.monitor.RecordValidationError(route.Path, err)
            http.Error(w, fmt.Sprintf("Validation error: %v", err), http.StatusBadRequest)
            return
        }
    }
    
    // Request transformation
    if route.Transform.Enabled {
        var err error
        r, err = gw.transformer.Transform(r, route.Transform)
        if err != nil {
            gw.monitor.RecordTransformError(route.Path, err)
            http.Error(w, "Request transformation failed", http.StatusInternalServerError)
            return
        }
    }
    
    // Execute through circuit breaker
    response, err := gw.circuitBreaker.Execute(r.Context(), func() (interface{}, error) {
        return gw.proxyRequest(r, route)
    })
    
    if err != nil {
        gw.monitor.RecordBackendError(route.Path, err)
        gw.handleError(w, r, route, err)
        return
    }
    
    httpResponse := response.(*http.Response)
    
    // Cache response if applicable
    if route.Cache.Enabled && gw.shouldCache(httpResponse) {
        gw.cache.Set(r.Context(), route, r, httpResponse)
    }
    
    // Record metrics
    duration := time.Since(startTime)
    gw.monitor.RecordRequest(route.Path, httpResponse.StatusCode, duration)
    
    // Write response
    gw.writeHTTPResponse(w, httpResponse)
}

func (gw *APIGateway) proxyRequest(r *http.Request, route *Route) (*http.Response, error) {
    // Select backend using load balancer
    backend, err := gw.loadBalancer.Select(route.Backends, r)
    if err != nil {
        return nil, fmt.Errorf("no available backend: %w", err)
    }
    
    // Create proxy request
    proxyReq, err := gw.createProxyRequest(r, backend)
    if err != nil {
        return nil, fmt.Errorf("failed to create proxy request: %w", err)
    }
    
    // Execute request with timeout and retry
    client := gw.getHTTPClient(backend)
    
    var lastErr error
    for attempt := 0; attempt <= route.Retries; attempt++ {
        ctx, cancel := context.WithTimeout(proxyReq.Context(), route.Timeout)
        proxyReq = proxyReq.WithContext(ctx)
        
        resp, err := client.Do(proxyReq)
        cancel()
        
        if err == nil {
            return resp, nil
        }
        
        lastErr = err
        
        // Check if we should retry
        if !gw.shouldRetry(err, resp) || attempt == route.Retries {
            break
        }
        
        // Exponential backoff
        backoff := time.Duration(attempt+1) * 100 * time.Millisecond
        time.Sleep(backoff)
    }
    
    return nil, lastErr
}

func (gw *APIGateway) createProxyRequest(r *http.Request, backend Backend) (*http.Request, error) {
    // Parse backend URL
    backendURL, err := url.Parse(backend.URL)
    if err != nil {
        return nil, err
    }
    
    // Create new request
    proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, backendURL.String()+r.URL.Path, r.Body)
    if err != nil {
        return nil, err
    }
    
    // Copy headers
    for name, values := range r.Header {
        for _, value := range values {
            proxyReq.Header.Add(name, value)
        }
    }
    
    // Add backend-specific headers
    for name, value := range backend.Headers {
        proxyReq.Header.Set(name, value)
    }
    
    // Add authentication headers
    if backend.Auth.Type != "" {
        if err := gw.addBackendAuth(proxyReq, backend.Auth); err != nil {
            return nil, err
        }
    }
    
    // Set forwarded headers
    proxyReq.Header.Set("X-Forwarded-For", r.RemoteAddr)
    proxyReq.Header.Set("X-Forwarded-Host", r.Host)
    proxyReq.Header.Set("X-Forwarded-Proto", "https")
    
    return proxyReq, nil
}

// Advanced middleware implementations
type MiddlewareChain struct {
    middlewares []Middleware
    mu          sync.RWMutex
}

type Middleware interface {
    Handle(w http.ResponseWriter, r *http.Request, next http.HandlerFunc)
    Priority() int
    Name() string
}

type LoggingMiddleware struct {
    logger Logger
    config LoggingConfig
}

func (lm *LoggingMiddleware) Handle(w http.ResponseWriter, r *http.Request, next http.HandlerFunc) {
    startTime := time.Now()
    requestID := r.Context().Value("request_id").(string)
    
    // Wrap response writer to capture status code
    wrapper := &responseWrapper{ResponseWriter: w}
    
    // Log request
    lm.logger.Info("Request started", map[string]interface{}{
        "request_id": requestID,
        "method":     r.Method,
        "path":       r.URL.Path,
        "remote_ip":  r.RemoteAddr,
        "user_agent": r.Header.Get("User-Agent"),
    })
    
    next(wrapper, r)
    
    // Log response
    duration := time.Since(startTime)
    lm.logger.Info("Request completed", map[string]interface{}{
        "request_id":  requestID,
        "status_code": wrapper.statusCode,
        "duration_ms": duration.Milliseconds(),
        "bytes_written": wrapper.bytesWritten,
    })
}

type SecurityMiddleware struct {
    config SecurityConfig
}

func (sm *SecurityMiddleware) Handle(w http.ResponseWriter, r *http.Request, next http.HandlerFunc) {
    // CORS headers
    if sm.config.CORS.Enabled {
        sm.setCORSHeaders(w, r)
        
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
    }
    
    // Security headers
    w.Header().Set("X-Content-Type-Options", "nosniff")
    w.Header().Set("X-Frame-Options", "DENY")
    w.Header().Set("X-XSS-Protection", "1; mode=block")
    w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    
    // Check for suspicious patterns
    if sm.detectSuspiciousRequest(r) {
        http.Error(w, "Suspicious request detected", http.StatusBadRequest)
        return
    }
    
    next(w, r)
}

func (sm *SecurityMiddleware) setCORSHeaders(w http.ResponseWriter, r *http.Request) {
    origin := r.Header.Get("Origin")
    if sm.isAllowedOrigin(origin) {
        w.Header().Set("Access-Control-Allow-Origin", origin)
        w.Header().Set("Access-Control-Allow-Methods", strings.Join(sm.config.CORS.AllowedMethods, ", "))
        w.Header().Set("Access-Control-Allow-Headers", strings.Join(sm.config.CORS.AllowedHeaders, ", "))
        w.Header().Set("Access-Control-Max-Age", "3600")
        
        if sm.config.CORS.AllowCredentials {
            w.Header().Set("Access-Control-Allow-Credentials", "true")
        }
    }
}

type CompressionMiddleware struct {
    config CompressionConfig
}

func (cm *CompressionMiddleware) Handle(w http.ResponseWriter, r *http.Request, next http.HandlerFunc) {
    // Check if client accepts compression
    acceptEncoding := r.Header.Get("Accept-Encoding")
    
    var writer http.ResponseWriter = w
    var shouldCompress bool
    
    if strings.Contains(acceptEncoding, "gzip") && cm.shouldCompress(r) {
        shouldCompress = true
        writer = NewGzipResponseWriter(w)
        w.Header().Set("Content-Encoding", "gzip")
        w.Header().Set("Vary", "Accept-Encoding")
    }
    
    next(writer, r)
    
    if shouldCompress {
        if gzipWriter, ok := writer.(*GzipResponseWriter); ok {
            gzipWriter.Close()
        }
    }
}

// Load balancing strategies
type LoadBalancer interface {
    Select(backends []Backend, r *http.Request) (Backend, error)
    UpdateHealth(backend Backend, healthy bool)
    GetStats() LoadBalancerStats
}

type RoundRobinLoadBalancer struct {
    counter uint64
    mu      sync.Mutex
}

func (rr *RoundRobinLoadBalancer) Select(backends []Backend, r *http.Request) (Backend, error) {
    healthyBackends := filterHealthyBackends(backends)
    if len(healthyBackends) == 0 {
        return Backend{}, ErrNoHealthyBackends
    }
    
    rr.mu.Lock()
    index := rr.counter % uint64(len(healthyBackends))
    rr.counter++
    rr.mu.Unlock()
    
    return healthyBackends[index], nil
}

type WeightedRoundRobinLoadBalancer struct {
    weights map[string]int
    current map[string]int
    mu      sync.Mutex
}

func (wrr *WeightedRoundRobinLoadBalancer) Select(backends []Backend, r *http.Request) (Backend, error) {
    healthyBackends := filterHealthyBackends(backends)
    if len(healthyBackends) == 0 {
        return Backend{}, ErrNoHealthyBackends
    }
    
    wrr.mu.Lock()
    defer wrr.mu.Unlock()
    
    var selected Backend
    maxWeight := -1
    
    for _, backend := range healthyBackends {
        // Increase current weight
        current := wrr.current[backend.URL]
        current += backend.Weight
        wrr.current[backend.URL] = current
        
        // Select backend with highest current weight
        if current > maxWeight {
            maxWeight = current
            selected = backend
        }
    }
    
    // Decrease current weight of selected backend
    wrr.current[selected.URL] -= wrr.getTotalWeight(healthyBackends)
    
    return selected, nil
}

func (wrr *WeightedRoundRobinLoadBalancer) getTotalWeight(backends []Backend) int {
    total := 0
    for _, backend := range backends {
        total += backend.Weight
    }
    return total
}

type ConsistentHashLoadBalancer struct {
    hashRing *HashRing
    mu       sync.RWMutex
}

func (ch *ConsistentHashLoadBalancer) Select(backends []Backend, r *http.Request) (Backend, error) {
    healthyBackends := filterHealthyBackends(backends)
    if len(healthyBackends) == 0 {
        return Backend{}, ErrNoHealthyBackends
    }
    
    // Use request path or session ID for consistent hashing
    key := r.URL.Path
    if sessionID := r.Header.Get("X-Session-ID"); sessionID != "" {
        key = sessionID
    }
    
    ch.mu.RLock()
    backendURL := ch.hashRing.Get(key)
    ch.mu.RUnlock()
    
    // Find backend by URL
    for _, backend := range healthyBackends {
        if backend.URL == backendURL {
            return backend, nil
        }
    }
    
    // Fallback to first healthy backend
    return healthyBackends[0], nil
}
```

---

## 📚 **System Design Interview Questions & Answers**

### **Event Sourcing Questions**

**Q1: How would you implement eventual consistency in a distributed event sourced system?**

**Answer:**

Eventual consistency in event sourcing can be achieved through:

1. Event Replication Strategy:

   - Async replication with conflict resolution
   - Vector clocks for ordering
   - Merkle trees for consistency verification

2. Read Models:

   - Eventually consistent projections
   - Sagas for cross-aggregate consistency
   - Compensation patterns for failures

3. Implementation approach:

   - Use distributed event log (Kafka, EventStore)
   - Implement projection rebuilding mechanisms
   - Handle network partitions with conflict resolution

**Q2: Design a payment processing system using Event Sourcing at scale.**

**Answer:**
The payment system would include:
- Payment aggregate with events (PaymentInitiated, PaymentProcessed, PaymentFailed)
- Separate command and query sides (CQRS)
- Saga orchestration for multi-step transactions
- Event replay for audit compliance
- Snapshot optimization for performance

**Q3: How do you handle schema evolution in event sourced systems?**

**Answer:**
- Versioned events with upcasting transformers
- Event schema registry for compatibility
- Graceful degradation strategies
- Migration sagas for data transformation

### **CQRS Questions**

**Q4: When would you not use CQRS? What are the tradeoffs?**

**Answer:**
Avoid CQRS when:
- Simple CRUD applications
- Low complexity domains
- Small team with limited expertise
- Strong consistency requirements

Tradeoffs:
- Increased complexity vs. scalability
- Eventual consistency vs. performance
- Development overhead vs. maintainability

**Q5: How do you ensure data consistency between command and query models?**

**Answer:**
- Event-driven synchronization
- Saga patterns for complex workflows
- Conflict resolution strategies
- Monitoring and alerting for sync failures

### **Saga Pattern Questions**

**Q6: Compare orchestration vs choreography in saga patterns.**

**Answer:**
Orchestration:
- Central coordinator manages workflow
- Easier to understand and debug
- Single point of failure
- Better for complex workflows

Choreography:
- Event-driven, decentralized
- Better fault tolerance
- Harder to track and debug
- Looser coupling between services

**Q7: How do you handle partial failures in long-running sagas?**

**Answer:**
- Compensation actions for rollback
- Idempotent operations
- Timeout handling with retries
- Dead letter queues for failed steps
- Circuit breakers for external services

### **Caching Strategy Questions**

**Q8: Design a multi-level caching system for a high-traffic e-commerce platform.**

**Answer:**
The system would include:
- L1: Application cache (in-memory)
- L2: Distributed cache (Redis cluster)
- L3: CDN for static content
- L4: Database query cache
- Cache promotion/demotion strategies
- Intelligent prefetching based on user behavior

**Q9: How do you handle cache invalidation in a microservices architecture?**

**Answer:**
- Event-driven invalidation
- Cache tags and dependencies
- Time-based expiration with refresh-ahead
- Versioned cache keys
- Distributed cache coherence protocols

### **Circuit Breaker Questions**

**Q10: How do you tune circuit breaker parameters for different types of services?**

**Answer:**
Parameters depend on service characteristics:
- Fast services: Lower failure threshold, shorter timeout
- Slow services: Higher threshold, longer timeout
- Critical services: More conservative settings
- Non-critical: More aggressive settings
- Use adaptive algorithms based on historical data

---

## 🎯 **Key Interview Tips**

### **System Design Approach**

1. **Clarify Requirements**
   - Functional requirements
   - Non-functional requirements (scale, performance)
   - Constraints and assumptions

2. **High-Level Design**
   - Start with simple design
   - Identify major components
   - Define data flow

3. **Detailed Design**
   - Choose appropriate patterns
   - Consider failure scenarios
   - Plan for scalability

4. **Scale the Design**
   - Identify bottlenecks
   - Apply scaling patterns
   - Discuss tradeoffs

### **Common Patterns to Mention**

- Event Sourcing for audit trails
- CQRS for read/write separation
- Saga for distributed transactions
- Circuit Breaker for fault tolerance
- Bulkhead for resource isolation
- Multi-level caching for performance

### **Real-World Examples**

- Netflix: Circuit breakers and bulkhead isolation
- Amazon: Event sourcing for order processing
- Uber: Saga patterns for trip management
- Facebook: Multi-level caching architecture
- Google: Consistent hashing for load balancing

This comprehensive guide provides advanced system design patterns with production-ready implementations essential for senior backend engineering interviews at companies like Razorpay, FAANG, and other high-scale technology organizations.