# Event Sourcing & CQRS Patterns

Advanced patterns for building event-driven systems with command-query responsibility segregation.

## 🎯 Overview

### Event Sourcing
- **Store events** instead of current state
- **Replay events** to rebuild state
- **Audit trail** of all changes
- **Time travel** capabilities

### CQRS (Command Query Responsibility Segregation)
- **Separate read and write** models
- **Optimize for different** use cases
- **Independent scaling** of read/write
- **Different data models** for commands and queries

## 🏗️ Architecture Patterns

### Event Sourcing Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Commands      │    │  Event Store    │    │   Read Models   │
│   (Write)       │────│  (Event Log)    │────│   (Projections) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Event Handlers │
                       │  (Side Effects) │
                       └─────────────────┘
```

### CQRS Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Commands      │    │  Command Side   │    │   Query Side    │
│   (Write)       │────│  (Write Model)  │    │   (Read Model)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Event Bus      │
                       │  (Synchronization)│
                       └─────────────────┘
```

## 🔧 Implementation

### Event Store
```go
type EventStore interface {
    AppendEvents(streamID string, events []Event, expectedVersion int) error
    GetEvents(streamID string, fromVersion int) ([]Event, error)
    GetEventsFrom(streamID string, fromVersion int, count int) ([]Event, error)
}

type Event struct {
    ID          string                 `json:"id"`
    StreamID    string                 `json:"stream_id"`
    Type        string                 `json:"type"`
    Data        map[string]interface{} `json:"data"`
    Metadata    map[string]interface{} `json:"metadata"`
    Version     int                    `json:"version"`
    Timestamp   time.Time              `json:"timestamp"`
}

type InMemoryEventStore struct {
    events map[string][]Event
    mutex  sync.RWMutex
}

func (s *InMemoryEventStore) AppendEvents(streamID string, events []Event, expectedVersion int) error {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    currentEvents := s.events[streamID]
    if len(currentEvents) != expectedVersion {
        return ErrConcurrencyConflict
    }
    
    for i, event := range events {
        event.Version = expectedVersion + i + 1
        event.Timestamp = time.Now()
        currentEvents = append(currentEvents, event)
    }
    
    s.events[streamID] = currentEvents
    return nil
}

func (s *InMemoryEventStore) GetEvents(streamID string, fromVersion int) ([]Event, error) {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    
    events := s.events[streamID]
    if fromVersion >= len(events) {
        return []Event{}, nil
    }
    
    return events[fromVersion:], nil
}
```

### Aggregate Root
```go
type AggregateRoot struct {
    ID      string
    Version int
    Events  []Event
}

func (ar *AggregateRoot) ApplyEvent(event Event) {
    ar.Events = append(ar.Events, event)
    ar.Version++
}

func (ar *AggregateRoot) GetUncommittedEvents() []Event {
    return ar.Events
}

func (ar *AggregateRoot) MarkEventsAsCommitted() {
    ar.Events = []Event{}
}

// Example: Order Aggregate
type Order struct {
    AggregateRoot
    CustomerID string
    Items      []OrderItem
    Status     OrderStatus
    Total      decimal.Decimal
}

func (o *Order) CreateOrder(customerID string, items []OrderItem) error {
    if len(items) == 0 {
        return ErrEmptyOrder
    }
    
    total := decimal.Zero
    for _, item := range items {
        total = total.Add(item.Price.Mul(decimal.NewFromInt(int64(item.Quantity))))
    }
    
    event := Event{
        ID:   generateEventID(),
        Type: "OrderCreated",
        Data: map[string]interface{}{
            "customer_id": customerID,
            "items":       items,
            "total":       total.String(),
        },
    }
    
    o.ApplyEvent(event)
    o.CustomerID = customerID
    o.Items = items
    o.Total = total
    o.Status = OrderPending
    
    return nil
}

func (o *Order) AddItem(item OrderItem) error {
    if o.Status != OrderPending {
        return ErrOrderNotPending
    }
    
    event := Event{
        ID:   generateEventID(),
        Type: "ItemAdded",
        Data: map[string]interface{}{
            "item": item,
        },
    }
    
    o.ApplyEvent(event)
    o.Items = append(o.Items, item)
    o.Total = o.Total.Add(item.Price.Mul(decimal.NewFromInt(int64(item.Quantity))))
    
    return nil
}

func (o *Order) ConfirmOrder() error {
    if o.Status != OrderPending {
        return ErrOrderNotPending
    }
    
    event := Event{
        ID:   generateEventID(),
        Type: "OrderConfirmed",
        Data: map[string]interface{}{
            "order_id": o.ID,
        },
    }
    
    o.ApplyEvent(event)
    o.Status = OrderConfirmed
    
    return nil
}
```

### Command Handler
```go
type CommandHandler interface {
    Handle(command Command) error
}

type CreateOrderCommand struct {
    CustomerID string
    Items      []OrderItem
}

type CreateOrderHandler struct {
    eventStore EventStore
    repository OrderRepository
}

func (h *CreateOrderHandler) Handle(command CreateOrderCommand) error {
    // Create new aggregate
    order := &Order{
        AggregateRoot: AggregateRoot{
            ID: generateOrderID(),
        },
    }
    
    // Apply business logic
    if err := order.CreateOrder(command.CustomerID, command.Items); err != nil {
        return err
    }
    
    // Save to event store
    events := order.GetUncommittedEvents()
    if err := h.eventStore.AppendEvents(order.ID, events, 0); err != nil {
        return err
    }
    
    // Update read model
    if err := h.repository.Save(order); err != nil {
        return err
    }
    
    order.MarkEventsAsCommitted()
    return nil
}
```

### Query Handler
```go
type QueryHandler interface {
    Handle(query Query) (interface{}, error)
}

type GetOrderQuery struct {
    OrderID string
}

type GetOrderHandler struct {
    readModel OrderReadModel
}

func (h *GetOrderHandler) Handle(query GetOrderQuery) (*OrderView, error) {
    return h.readModel.GetOrder(query.OrderID)
}

type OrderView struct {
    ID         string      `json:"id"`
    CustomerID string      `json:"customer_id"`
    Items      []OrderItem `json:"items"`
    Status     string      `json:"status"`
    Total      string      `json:"total"`
    CreatedAt  time.Time   `json:"created_at"`
    UpdatedAt  time.Time   `json:"updated_at"`
}
```

### Event Handler
```go
type EventHandler interface {
    Handle(event Event) error
}

type OrderEventHandler struct {
    readModel OrderReadModel
    emailService EmailService
    inventoryService InventoryService
}

func (h *OrderEventHandler) Handle(event Event) error {
    switch event.Type {
    case "OrderCreated":
        return h.handleOrderCreated(event)
    case "OrderConfirmed":
        return h.handleOrderConfirmed(event)
    case "ItemAdded":
        return h.handleItemAdded(event)
    default:
        return nil
    }
}

func (h *OrderEventHandler) handleOrderCreated(event Event) error {
    // Update read model
    orderView := &OrderView{
        ID:         event.Data["order_id"].(string),
        CustomerID: event.Data["customer_id"].(string),
        Items:      event.Data["items"].([]OrderItem),
        Status:     "pending",
        Total:      event.Data["total"].(string),
        CreatedAt:  event.Timestamp,
        UpdatedAt:  event.Timestamp,
    }
    
    return h.readModel.SaveOrder(orderView)
}

func (h *OrderEventHandler) handleOrderConfirmed(event Event) error {
    orderID := event.Data["order_id"].(string)
    
    // Update read model
    if err := h.readModel.UpdateOrderStatus(orderID, "confirmed"); err != nil {
        return err
    }
    
    // Send confirmation email
    order, err := h.readModel.GetOrder(orderID)
    if err != nil {
        return err
    }
    
    return h.emailService.SendOrderConfirmation(order.CustomerID, order)
}

func (h *OrderEventHandler) handleItemAdded(event Event) error {
    item := event.Data["item"].(OrderItem)
    
    // Update inventory
    return h.inventoryService.ReserveItem(item.ProductID, item.Quantity)
}
```

## 📊 Projection Patterns

### Read Model Projection
```go
type Projection interface {
    Process(event Event) error
    GetName() string
}

type OrderProjection struct {
    readModel OrderReadModel
}

func (p *OrderProjection) Process(event Event) error {
    switch event.Type {
    case "OrderCreated":
        return p.handleOrderCreated(event)
    case "OrderConfirmed":
        return p.handleOrderConfirmed(event)
    case "ItemAdded":
        return p.handleItemAdded(event)
    default:
        return nil
    }
}

func (p *OrderProjection) GetName() string {
    return "order_projection"
}

func (p *OrderProjection) handleOrderCreated(event Event) error {
    orderView := &OrderView{
        ID:         event.Data["order_id"].(string),
        CustomerID: event.Data["customer_id"].(string),
        Items:      event.Data["items"].([]OrderItem),
        Status:     "pending",
        Total:      event.Data["total"].(string),
        CreatedAt:  event.Timestamp,
        UpdatedAt:  event.Timestamp,
    }
    
    return p.readModel.SaveOrder(orderView)
}
```

### Event Replay
```go
type EventReplay struct {
    eventStore EventStore
    projections []Projection
}

func (r *EventReplay) ReplayAllEvents() error {
    // Get all events from event store
    events, err := r.eventStore.GetAllEvents()
    if err != nil {
        return err
    }
    
    // Process each event through all projections
    for _, event := range events {
        for _, projection := range r.projections {
            if err := projection.Process(event); err != nil {
                return err
            }
        }
    }
    
    return nil
}

func (r *EventReplay) ReplayFromVersion(version int) error {
    events, err := r.eventStore.GetEventsFromVersion(version)
    if err != nil {
        return err
    }
    
    for _, event := range events {
        for _, projection := range r.projections {
            if err := projection.Process(event); err != nil {
                return err
            }
        }
    }
    
    return nil
}
```

## 🔄 CQRS Implementation

### Command Bus
```go
type CommandBus struct {
    handlers map[string]CommandHandler
}

func (b *CommandBus) RegisterHandler(commandType string, handler CommandHandler) {
    b.handlers[commandType] = handler
}

func (b *CommandBus) Execute(command Command) error {
    commandType := reflect.TypeOf(command).Name()
    handler, exists := b.handlers[commandType]
    if !exists {
        return ErrHandlerNotFound
    }
    
    return handler.Handle(command)
}
```

### Query Bus
```go
type QueryBus struct {
    handlers map[string]QueryHandler
}

func (b *QueryBus) RegisterHandler(queryType string, handler QueryHandler) {
    b.handlers[queryType] = handler
}

func (b *QueryBus) Execute(query Query) (interface{}, error) {
    queryType := reflect.TypeOf(query).Name()
    handler, exists := b.handlers[queryType]
    if !exists {
        return nil, ErrHandlerNotFound
    }
    
    return handler.Handle(query)
}
```

### Event Bus
```go
type EventBus struct {
    handlers map[string][]EventHandler
}

func (b *EventBus) RegisterHandler(eventType string, handler EventHandler) {
    b.handlers[eventType] = append(b.handlers[eventType], handler)
}

func (b *EventBus) Publish(event Event) error {
    handlers := b.handlers[event.Type]
    
    for _, handler := range handlers {
        if err := handler.Handle(event); err != nil {
            return err
        }
    }
    
    return nil
}
```

## 🚀 Advanced Patterns

### Saga Pattern
```go
type Saga struct {
    ID        string
    Steps     []SagaStep
    Status    SagaStatus
    Events    []Event
}

type SagaStep struct {
    ID           string
    Command      Command
    Compensation Command
    Status       StepStatus
}

func (s *Saga) Execute() error {
    for _, step := range s.Steps {
        if err := s.executeStep(step); err != nil {
            // Compensate previous steps
            return s.compensate(step)
        }
    }
    
    s.Status = SagaCompleted
    return nil
}

func (s *Saga) executeStep(step SagaStep) error {
    // Execute command
    if err := s.commandBus.Execute(step.Command); err != nil {
        return err
    }
    
    step.Status = StepCompleted
    return nil
}

func (s *Saga) compensate(failedStep SagaStep) error {
    // Find failed step index
    failedIndex := -1
    for i, step := range s.Steps {
        if step.ID == failedStep.ID {
            failedIndex = i
            break
        }
    }
    
    // Compensate in reverse order
    for i := failedIndex - 1; i >= 0; i-- {
        step := s.Steps[i]
        if step.Status == StepCompleted {
            if err := s.commandBus.Execute(step.Compensation); err != nil {
                return err
            }
            step.Status = StepCompensated
        }
    }
    
    s.Status = SagaFailed
    return nil
}
```

### Event Sourcing with Snapshots
```go
type Snapshot struct {
    AggregateID string
    Version     int
    Data        map[string]interface{}
    Timestamp   time.Time
}

type SnapshotStore interface {
    Save(snapshot Snapshot) error
    Get(aggregateID string) (*Snapshot, error)
}

type SnapshotPolicy struct {
    SnapshotInterval int
    MaxEvents        int
}

func (p *SnapshotPolicy) ShouldCreateSnapshot(currentVersion, lastSnapshotVersion int) bool {
    return currentVersion-lastSnapshotVersion >= p.SnapshotInterval
}

type SnapshotManager struct {
    eventStore     EventStore
    snapshotStore  SnapshotStore
    policy         SnapshotPolicy
}

func (m *SnapshotManager) CreateSnapshot(aggregateID string, aggregate AggregateRoot) error {
    snapshot := Snapshot{
        AggregateID: aggregateID,
        Version:     aggregate.Version,
        Data:        aggregate.GetState(),
        Timestamp:   time.Now(),
    }
    
    return m.snapshotStore.Save(snapshot)
}

func (m *SnapshotManager) LoadAggregate(aggregateID string, aggregate AggregateRoot) error {
    // Try to load from snapshot
    snapshot, err := m.snapshotStore.Get(aggregateID)
    if err == nil {
        aggregate.LoadFromSnapshot(snapshot)
    }
    
    // Load events since snapshot
    fromVersion := 0
    if snapshot != nil {
        fromVersion = snapshot.Version + 1
    }
    
    events, err := m.eventStore.GetEvents(aggregateID, fromVersion)
    if err != nil {
        return err
    }
    
    // Apply events to aggregate
    for _, event := range events {
        aggregate.ApplyEvent(event)
    }
    
    return nil
}
```

## 📊 Performance Optimization

### Event Store Optimization
```go
type OptimizedEventStore struct {
    eventStore    EventStore
    cache         *cache.Cache
    batchSize     int
    writeBuffer   []Event
    mutex         sync.Mutex
}

func (s *OptimizedEventStore) AppendEvents(streamID string, events []Event, expectedVersion int) error {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    // Add to write buffer
    s.writeBuffer = append(s.writeBuffer, events...)
    
    // Flush if buffer is full
    if len(s.writeBuffer) >= s.batchSize {
        return s.flush()
    }
    
    return nil
}

func (s *OptimizedEventStore) flush() error {
    if len(s.writeBuffer) == 0 {
        return nil
    }
    
    // Group events by stream
    eventsByStream := make(map[string][]Event)
    for _, event := range s.writeBuffer {
        eventsByStream[event.StreamID] = append(eventsByStream[event.StreamID], event)
    }
    
    // Write to event store
    for streamID, events := range eventsByStream {
        if err := s.eventStore.AppendEvents(streamID, events, 0); err != nil {
            return err
        }
    }
    
    // Clear buffer
    s.writeBuffer = []Event{}
    return nil
}
```

### Read Model Optimization
```go
type OptimizedReadModel struct {
    database    *sql.DB
    cache       *cache.Cache
    indexer     *Indexer
}

func (r *OptimizedReadModel) GetOrder(orderID string) (*OrderView, error) {
    // Try cache first
    if cached, found := r.cache.Get(orderID); found {
        return cached.(*OrderView), nil
    }
    
    // Query database
    order, err := r.queryOrder(orderID)
    if err != nil {
        return nil, err
    }
    
    // Cache result
    r.cache.Set(orderID, order, 5*time.Minute)
    
    return order, nil
}

func (r *OptimizedReadModel) SearchOrders(query SearchQuery) ([]*OrderView, error) {
    // Use index for search
    orderIDs, err := r.indexer.Search(query)
    if err != nil {
        return nil, err
    }
    
    // Batch load orders
    orders := make([]*OrderView, 0, len(orderIDs))
    for _, orderID := range orderIDs {
        order, err := r.GetOrder(orderID)
        if err != nil {
            continue
        }
        orders = append(orders, order)
    }
    
    return orders, nil
}
```

## 🔍 Testing Strategies

### Event Sourcing Tests
```go
func TestOrderAggregate(t *testing.T) {
    order := &Order{
        AggregateRoot: AggregateRoot{
            ID: "order-123",
        },
    }
    
    // Test order creation
    items := []OrderItem{
        {ProductID: "prod-1", Quantity: 2, Price: decimal.NewFromFloat(10.99)},
    }
    
    err := order.CreateOrder("customer-123", items)
    assert.NoError(t, err)
    assert.Equal(t, "customer-123", order.CustomerID)
    assert.Len(t, order.Items, 1)
    assert.Equal(t, OrderPending, order.Status)
    
    // Test event generation
    events := order.GetUncommittedEvents()
    assert.Len(t, events, 1)
    assert.Equal(t, "OrderCreated", events[0].Type)
    
    // Test order confirmation
    err = order.ConfirmOrder()
    assert.NoError(t, err)
    assert.Equal(t, OrderConfirmed, order.Status)
    
    events = order.GetUncommittedEvents()
    assert.Len(t, events, 2)
    assert.Equal(t, "OrderConfirmed", events[1].Type)
}

func TestEventStore(t *testing.T) {
    eventStore := NewInMemoryEventStore()
    
    events := []Event{
        {ID: "evt-1", Type: "OrderCreated", Data: map[string]interface{}{"order_id": "order-123"}},
        {ID: "evt-2", Type: "OrderConfirmed", Data: map[string]interface{}{"order_id": "order-123"}},
    }
    
    // Append events
    err := eventStore.AppendEvents("order-123", events, 0)
    assert.NoError(t, err)
    
    // Retrieve events
    retrieved, err := eventStore.GetEvents("order-123", 0)
    assert.NoError(t, err)
    assert.Len(t, retrieved, 2)
    assert.Equal(t, "evt-1", retrieved[0].ID)
    assert.Equal(t, "evt-2", retrieved[1].ID)
}
```

## 📚 Best Practices

### Event Design
1. **Immutable**: Events should never change
2. **Versioned**: Include version information
3. **Rich**: Include all necessary data
4. **Named**: Use clear, descriptive names
5. **Small**: Keep events focused and small

### CQRS Design
1. **Separate Models**: Different models for read/write
2. **Eventual Consistency**: Accept eventual consistency
3. **Optimize for Use Case**: Optimize each side independently
4. **Handle Failures**: Implement proper error handling
5. **Monitor**: Monitor both sides separately

### Performance Considerations
1. **Batch Operations**: Batch events and commands
2. **Caching**: Cache read models
3. **Indexing**: Index for common queries
4. **Partitioning**: Partition by aggregate ID
5. **Compression**: Compress event data

---

**Last Updated**: December 2024  
**Category**: Advanced System Design Patterns  
**Complexity**: Expert Level
